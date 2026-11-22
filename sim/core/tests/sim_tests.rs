use potok_protocol as proto;
use potok_sim::{
    atmosphere::Atmosphere,
    config::Config,
    dynamics::{derivative, Phase, State, Vehicle},
    math::{episode_seed, lerp_table},
    sensor::{gating, wire_flags, FLAG_AI_READY, FLAG_FAULT, FLAG_MOTOR_ARMED},
    validation::{
        read_validation, write_validation, Header, FILE_NAME, HEADER_SIZE, META_V2, TARGET_COUNT,
        VERSION,
    },
    Frame, Simulator,
};

fn sim() -> Simulator {
    Simulator::new(Config::default()).expect("default config must load")
}

#[test]
fn same_seed_twice_is_bit_identical() {
    let s = sim();
    let a = s.run_episode(0).unwrap();
    let b = s.run_episode(0).unwrap();
    assert_eq!(a.frames.len(), b.frames.len());
    for (fa, fb) in a.frames.iter().zip(b.frames.iter()) {
        assert_eq!(fa.to_bytes(), fb.to_bytes());
        assert_eq!(fa.crc, fb.crc);
    }
    for (ta, tb) in a.targets.iter().zip(b.targets.iter()) {
        assert_eq!(ta.g_theta, tb.g_theta);
        assert_eq!(ta.confidence, tb.confidence);
    }
    assert_eq!(a.features, b.features);
    assert_eq!(a.meta.mass_kg, b.meta.mass_kg);
    assert_eq!(a.phase_counts, b.phase_counts);
}

#[test]
fn different_seeds_differ() {
    let mut cfg = Config::default();
    cfg.run.seed = 1;
    let a = Simulator::new(cfg).unwrap().run_episode(0).unwrap();
    let mut cfg = Config::default();
    cfg.run.seed = 2;
    let b = Simulator::new(cfg).unwrap().run_episode(0).unwrap();
    assert_ne!(a.features, b.features);
    assert_ne!(a.meta.mass_kg, b.meta.mass_kg);
    assert_ne!(episode_seed(1, 0), episode_seed(2, 0));
}

#[test]
fn frame_rate_and_count_are_exact() {
    let cfg = Config::default();
    assert_eq!(cfg.steps(), 15_000);
    assert_eq!(cfg.steps_per_frame(), 10);
    assert_eq!(cfg.frames_per_episode(), 1_500);
    let e = sim().run_episode(0).unwrap();
    assert_eq!(e.frames.len(), 1_500);
    assert_eq!(e.targets.len(), 1_500);
    assert_eq!(e.features.len(), 1_500 * cfg.feature_count());
    for (i, f) in e.frames.iter().enumerate() {
        assert_eq!(f.seq, i as u16);
        assert_eq!(f.time_us, (i as u32) * 20_000);
    }
}

#[test]
fn no_nan_or_infinite_values_over_30s() {
    let e = sim().run_episode(0).unwrap();
    assert_eq!(e.nan_frames, 0);
    assert!(e.features.iter().all(|v| v.is_finite()));
    assert!(e
        .targets
        .iter()
        .all(|t| t.g_theta.iter().all(|v| v.is_finite()) && t.confidence.is_finite()));
    assert!(e.frames.iter().all(|f| f.to_bytes().len() == 56));
}

#[test]
fn gating_flags_follow_phase_and_sensor_health() {
    let e = sim().run_episode(0).unwrap();
    let boost = e.frames.iter().filter(|f| f.phase == Phase::Boost).count();
    let coast = e
        .frames
        .iter()
        .filter(|f| {
            f.phase == Phase::Coast || f.phase == Phase::Apogee || f.phase == Phase::Descent
        })
        .count();
    assert!(boost > 0 && coast > 0, "phase counts: {:?}", e.phase_counts);
    assert!(e
        .frames
        .iter()
        .filter(|f| f.phase == Phase::Boost)
        .all(|f| f.flags & FLAG_AI_READY == 0 && f.flags & FLAG_MOTOR_ARMED != 0));
    for f in &e.frames {
        let sensor_ok = f.flags().has(potok_sim::sensor::FLAG_IMU_OK);
        assert_eq!(
            f.flags & FLAG_AI_READY != 0,
            gating(f.phase, sensor_ok, f.pitot_valid, f.saturated),
        );
    }
    assert!(!gating(Phase::Boost, true, true, false));
    assert!(gating(Phase::Coast, true, true, false));
    assert!(!gating(Phase::Coast, false, true, false));
    assert!(!gating(Phase::Coast, true, true, true));
    // Gust/shear have no protocol bit; they stay simulator-side context.
    let e3 = sim().run_episode(3).unwrap();
    if e3.frames.iter().any(|f| f.gust) {
        assert!(e3.meta.gust_start_s.is_some());
    }
    // A frame rejected by gating never produces a healthy target.
    for (f, t) in e.frames.iter().zip(e.targets.iter()) {
        if !f.ai_input_valid() {
            assert!(!t.ai_healthy);
        }
        if f.saturated {
            assert!(f.flags().has(FLAG_FAULT));
        }
    }
}

#[test]
fn atmosphere_and_aerodynamics_are_sane() {
    let cfg = Config::default();
    let atm = Atmosphere::load(
        &{
            let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../config/boundary_profile.csv");
            p
        },
        cfg.sim.altitude_grid_m,
    )
    .unwrap();
    let d0 = atm.at(0.0).density;
    let d1 = atm.at(1000.0).density;
    let d5 = atm.at(5000.0).density;
    assert!(d0 > d1 && d1 > d5, "densities {d0} {d1} {d5}");
    assert!((atm.at(0.0).speed_of_sound - 340.3).abs() < 1.0);
    assert!((d0 - 1.225).abs() < 0.01);
    assert_eq!(lerp_table(&[(0.0, 1.0), (2.0, 3.0)], 1.0), 2.0);

    let v = Vehicle::from_config(&cfg, 130.0, 1.0, [0.0, 0.0]);
    assert!(v.burn_end() > 2.0 && v.burn_end() <= 30.0);
    // Resting on the rail before liftoff: inertial accel is zero, so the
    // accelerometer reads +g on body Z (f = a - g).
    let s = State::new(130.0, 0.0, 0.0);
    let d = derivative(0.0, &s, [0.0; 3], &atm, &v);
    assert!(d.vel[2].abs() < 1e-6, "a_z = {}", d.vel[2]);
    let spec_z = d.vel[2] + 9.80665;
    assert!((spec_z - 9.80665).abs() < 1e-6);

    // Drag must oppose a purely lateral flow (cna = 0 isolates pure drag).
    let mut no_lift = cfg.clone();
    no_lift.aero.cna = 0.0;
    let v_drag = Vehicle::from_config(&no_lift, 130.0, 1.0, [0.0, 0.0]);
    let mut moving = s;
    moving.vel[0] = 50.0;
    let dm = derivative(10.0, &moving, [0.0; 3], &atm, &v_drag);
    assert!(dm.vel[0] < 0.0, "drag should decelerate, got {}", dm.vel[0]);
    assert!(cfg
        .aero
        .cd_mach
        .iter()
        .map(|p| p.1)
        .all(|c| c > 0.0 && c.is_finite()));
}

#[test]
fn crc_and_frame_roundtrip() {
    let e = sim().run_episode(0).unwrap();
    for f in e.frames.iter().step_by(97) {
        let bytes = f.to_bytes();
        assert_eq!(bytes.len(), 56);
        let back = Frame::from_bytes(&bytes).unwrap();
        assert_eq!(&back.to_bytes(), &bytes);
        assert_eq!(back.crc, proto::crc16(&bytes[..54]));
        assert_eq!(proto::decode_request(&bytes).unwrap(), back.to_wire());
    }
    assert_eq!(episode_seed(7, 0), episode_seed(7, 0));
    assert_ne!(episode_seed(7, 0), episode_seed(7, 1));
}

#[test]
fn validation_file_roundtrip() {
    let dir = std::env::temp_dir().join(format!("potok-sim-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(FILE_NAME);
    let cfg = Config::default();
    let sim = Simulator::new(cfg.clone()).unwrap();
    let episodes: Vec<_> = sim.episodes(2).map(|e| e.unwrap()).collect();
    assert_eq!(HEADER_SIZE, 80);
    assert_eq!(Header::header_size(), HEADER_SIZE);
    write_validation(&path, &cfg, &episodes).unwrap();
    let raw = std::fs::read(&path).unwrap();
    assert_eq!(&raw[..8], b"POTOKSIM");
    assert_eq!(u32::from_le_bytes(raw[8..12].try_into().unwrap()), VERSION);
    assert_eq!(
        u32::from_le_bytes(raw[48..52].try_into().unwrap()),
        META_V2 as u32
    );
    assert!(raw[52..HEADER_SIZE].iter().all(|b| *b == 0));
    let per_episode =
        8 + META_V2 * 4 + cfg.frames_per_episode() * (cfg.feature_count() + TARGET_COUNT) * 4;
    assert_eq!(raw.len(), HEADER_SIZE + 2 * per_episode);
    let rs = read_validation(&path).unwrap();
    assert_eq!(rs.version, VERSION);
    assert_eq!(rs.header.meta_count as usize, META_V2);
    assert_eq!(rs.header.seed, cfg.run.seed);
    assert_eq!(rs.header.episode_count, 2);
    assert_eq!(
        rs.header.frames_per_episode as usize,
        cfg.frames_per_episode()
    );
    assert_eq!(rs.header.feature_count as usize, cfg.feature_count());
    assert_eq!(rs.header.target_count as usize, TARGET_COUNT);
    assert_eq!(rs.episodes.len(), 2);
    for (src, dst) in episodes.iter().zip(rs.episodes.iter()) {
        assert_eq!(dst.conditioning, src.meta.conditioning);
        assert_eq!(dst.scenario_id, src.meta.scenario_id);
        assert_eq!(dst.split, src.meta.split);
        assert_eq!(src.features.len(), dst.features.len());
        assert_eq!(dst.features[[0, 0]], src.features[0]);
        assert_eq!(
            dst.features[[1499, cfg.feature_count() - 1]],
            src.features[src.features.len() - 1]
        );
        assert_eq!(dst.targets[[0, 0]], src.targets[0].g_theta[0]);
        assert_eq!(
            dst.targets[[1499, 3]],
            src.targets[1499].ai_healthy as u8 as f32
        );
    }
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn fast_field_matches_exact_analytic_profile() {
    let mut fast = Config::default();
    fast.run.seed = 7;
    fast.run.fast_field = true;
    let mut exact = fast.clone();
    exact.run.fast_field = false;
    let a = Simulator::new(fast).unwrap().run_episode(0).unwrap();
    let b = Simulator::new(exact).unwrap().run_episode(0).unwrap();
    let max_rel = a
        .features
        .iter()
        .zip(b.features.iter())
        .map(|(x, y)| (x - y).abs() / (1.0 + y.abs()))
        .fold(0.0_f32, f32::max);
    assert!(max_rel < 1e-2, "fast lookup deviates by {max_rel} relative");
    // Both paths remain deterministic.
    let sim_fast = Simulator::new({
        let mut c = Config::default();
        c.run.seed = 7;
        c.run.fast_field = true;
        c
    })
    .unwrap();
    assert_eq!(sim_fast.run_episode(0).unwrap().features, a.features);
}

#[test]
fn parallel_episodes_match_serial() {
    let sim = Simulator::new(Config::default()).unwrap();
    let serial: Vec<Vec<f32>> = sim.episodes(4).map(|e| e.unwrap().features).collect();
    let parallel: Vec<Vec<f32>> = sim
        .run_episodes_parallel(4)
        .into_iter()
        .map(|e| e.unwrap().features)
        .collect();
    assert_eq!(serial, parallel);
}

#[test]
fn defaults_match_real_vehicle_ttx() {
    let cfg = Config::default();
    assert!((cfg.initial.mass_kg.0 - 12.5).abs() < 1e-9);
    assert!((cfg.initial.mass_kg.1 - 13.5).abs() < 1e-9);
    assert!((cfg.aero.area_m2 - 0.00817).abs() < 1e-9);
    assert!((cfg.aero.static_margin_m - 0.0765).abs() < 1e-9);
    assert!((cfg.sensor.accel_noise - 0.011).abs() < 1e-9);
    assert!((cfg.sensor.gyro_noise - 0.0015).abs() < 1e-9);

    let v = Vehicle::from_config(&cfg, 13.0, 1.0, [0.0, 0.0]);
    assert!(
        (v.propellant_mass() - 2.881).abs() < 0.05,
        "propellant {}",
        v.propellant_mass()
    );
    assert!(
        v.burn_end() >= 8.5 && v.burn_end() <= 9.0,
        "burn_end {}",
        v.burn_end()
    );
    // Motor passports minimum 3.2 kNs: 2.2 / 1.0 kNs over the 5.5 / 3.2 s burns.
    let stage1 = v.thrust_at(3.0);
    let stage2 = v.thrust_at(7.0);
    assert!((stage1 - 400.0).abs() < 1e-6, "stage 1 {stage1}");
    assert!((stage2 - 312.5).abs() < 1e-6, "stage 2 {stage2}");
    assert!(stage1 < 550.0 && stage2 < 600.0, "below passport peaks");
    assert!(v.thrust_at(9.0).abs() < 1e-9);
}

#[test]
fn nominal_trajectory_is_plausible() {
    let e = sim().run_episode(0).unwrap();
    eprintln!(
        "nominal: apogee={:.1} m, vmax={:.1} m/s",
        e.apogee_m, e.max_speed_mps
    );
    assert!(
        (1_900.0..=2_200.0).contains(&e.apogee_m),
        "apogee {} m outside 1.9-2.2 km",
        e.apogee_m
    );
    assert!(
        (165.0..=185.0).contains(&e.max_speed_mps),
        "max speed {} m/s outside 165-185",
        e.max_speed_mps
    );
}

#[test]
fn protocol_phase_codes_match_uart_yaml() {
    assert_eq!(Phase::Pad.protocol_code(), 0);
    assert_eq!(Phase::Boost.protocol_code(), 0);
    assert_eq!(Phase::Coast.protocol_code(), 1);
    assert_eq!(Phase::Apogee.protocol_code(), 2);
    assert_eq!(Phase::Descent.protocol_code(), 3);
    assert_eq!(Phase::Landed.protocol_code(), 4);
    for code in 0..=4u8 {
        assert_eq!(
            Phase::from_protocol_code(code).unwrap().protocol_code(),
            code
        );
    }
    assert!(Phase::from_protocol_code(5).is_none());
    let boost_flags = wire_flags(Phase::Boost, true, true, false);
    assert_eq!(boost_flags, FLAG_MOTOR_ARMED | 0x0e);
    assert_eq!(boost_flags & FLAG_AI_READY, 0);
    assert_eq!(
        wire_flags(Phase::Coast, true, true, false) & FLAG_AI_READY,
        FLAG_AI_READY
    );
}

#[test]
fn canonical_golden_request_is_byte_identical() {
    const GOLDEN: &str = "4d5034127bf1536502011200cdcccc3dcdcc4cbe33335340c3f51c41000000000000c0bf00004841000050c0000030420080f742bffec67f";
    let f = Frame {
        seq: 0x1234,
        time_us: 1_700_000_123,
        phase: Phase::Apogee,
        gyro: [0.1, -0.2, 3.3],
        accel: [9.81, 0.0, -1.5],
        mag: [12.5, -3.25, 44.0],
        height: 123.75,
        pitot_dp: -321,
        flags: potok_sim::sensor::FLAG_IMU_OK | FLAG_AI_READY,
        crc: 0,
        gust: false,
        shear: false,
        saturated: false,
        spike: false,
        pitot_valid: true,
        physics: [0.0; 6],
    };
    let hex: String = f.to_bytes().iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(hex, GOLDEN);
    let bytes: Vec<u8> = (0..GOLDEN.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&GOLDEN[i..i + 2], 16).unwrap())
        .collect();
    assert_eq!(bytes.len(), proto::REQUEST_SIZE);
    let wire = proto::decode_request(&bytes).unwrap();
    assert_eq!(wire, f.to_wire());
    assert_eq!(Frame::from_bytes(&bytes).unwrap().to_bytes(), bytes);
}

#[test]
fn config_validation_rejects_bad_frame_rate() {
    let mut cfg = Config::default();
    cfg.run.frame_rate_hz = 33.0;
    assert!(cfg.validate().is_err());
    let mut cfg = Config::default();
    cfg.run.dt = -1.0;
    assert!(cfg.validate().is_err());
}
