//! Ideal-loop physics lane: aero card interpolation, truth v2 metadata and
//! seeded sensor realism. Every new knob defaults to off, so all legacy tests
//! keep their original behaviour.

use std::{
    path::{Path, PathBuf},
    sync::Mutex,
};

use potok_sim::{
    aero::{placeholder_uses, AeroChannel},
    config::AeroSource,
    sensor::FLAG_FAULT,
    validation::{
        read_validation, write_validation, write_validation_with_version, FILE_NAME, HEADER_SIZE,
        META_V1, META_V2, META_V2_TRUTH, TARGET_COUNT, TRUTH_COUNT, VERSION, VERSION_V1,
    },
    AeroCard, Config, SimError, Simulator,
};

// The placeholder usage marker is process-global; serialize the tests that
// observe it so parallel test threads cannot perturb the count.
static COUNTER_LOCK: Mutex<()> = Mutex::new(());

// The generated placeholder card is kept as a separate fixture: these tests
// assert legacy parity, while config/aero_card.toml now holds the real
// OpenRocket-derived card (covered by core/tests/aero_card_tests.rs).
fn card_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../config/aero_card_placeholder.toml")
}

fn card_config() -> Config {
    let mut cfg = Config::default();
    cfg.aero.source = AeroSource::Card;
    cfg.aero.card_path = Some(PathBuf::from("config/aero_card_placeholder.toml"));
    cfg
}

fn truth_config() -> Config {
    let mut cfg = Config::default();
    cfg.output.truth_metadata = true;
    cfg
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("potok-ideal-{tag}-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    dir
}

#[test]
fn card_interpolation_matches_grid_nodes_and_is_linear_between_them() {
    let card = AeroCard::from_file(&card_path()).unwrap();
    for ch in [
        AeroChannel::Cd0,
        AeroChannel::Cna,
        AeroChannel::PitchDamping,
        AeroChannel::CmDelta,
    ] {
        for (ai, alpha) in card.alpha_deg.iter().enumerate() {
            for (mi, mach) in card.mach.iter().enumerate() {
                let got = card.interpolate(ch, *mach, *alpha).unwrap();
                let want = card.channel(ch)[ai][mi];
                assert_eq!(got.to_bits(), want.to_bits(), "{ch:?} at ({mach}, {alpha})");
            }
        }
    }
    // Flat-alpha card reproduces the legacy lerp exactly at a mach midpoint.
    let m0 = card.mach[1];
    let m1 = card.mach[2];
    let mid = 0.5 * (m0 + m1);
    let want = 0.5 * (card.channel(AeroChannel::Cd0)[4][1] + card.channel(AeroChannel::Cd0)[4][2]);
    let got = card.interpolate(AeroChannel::Cd0, mid, 0.0).unwrap();
    assert!((got - want).abs() < 1e-9, "got {got}, want {want}");

    // A genuinely 2-D grid exercises bilinear, not just the flat fast path.
    let tilted = AeroCard::from_toml_str(
        r#"
        baseline_cd0 = 0.0
        [grid]
        mach = [0.0, 1.0]
        alpha_deg = [0.0, 10.0]
        [coefficients]
        cd0 = [[0.0, 1.0], [2.0, 3.0]]
        cna = [[1.0, 1.0], [1.0, 1.0]]
        pitch_damping = [[0.0, 0.0], [0.0, 0.0]]
        cm_delta = [[0.0, 0.0], [0.0, 0.0]]
        "#,
    )
    .unwrap();
    assert_eq!(tilted.interpolate(AeroChannel::Cd0, 0.5, 5.0).unwrap(), 1.5);
    assert_eq!(
        tilted.interpolate(AeroChannel::Cd0, 0.25, 2.5).unwrap(),
        0.75
    );
}

#[test]
fn card_out_of_grid_queries_are_loud_and_never_clamped() {
    let card = AeroCard::from_file(&card_path()).unwrap();
    for (mach, alpha, axis) in [
        (-1e-6, 0.0, "mach"),
        (3.1, 0.0, "mach"),
        (1.0, -180.1, "alpha_deg"),
        (1.0, 180.1, "alpha_deg"),
        (f64::NAN, 0.0, "mach"),
    ] {
        match card.interpolate(AeroChannel::Cd0, mach, alpha) {
            Err(SimError::AeroOutOfGrid { axis: a, .. }) => assert_eq!(a, axis),
            other => panic!("expected AeroOutOfGrid for ({mach}, {alpha}), got {other:?}"),
        }
    }
    // Malformed grids are rejected at load, also loudly.
    assert!(matches!(
        AeroCard::from_toml_str(
            r#"
            baseline_cd0 = 0.0
            [grid]
            mach = [0.0, 1.0, 2.0]
            alpha_deg = [0.0, 10.0]
            [coefficients]
            cd0 = [[0.0, 1.0], [2.0, 3.0]]
            cna = [[0.0, 0.0], [0.0, 0.0]]
            pitch_damping = [[0.0, 0.0], [0.0, 0.0]]
            cm_delta = [[0.0, 0.0], [0.0, 0.0]]
            "#
        ),
        Err(SimError::AeroCard(_))
    ));
}

#[test]
fn card_out_of_grid_during_a_run_returns_an_error() {
    let dir = temp_dir("narrow");
    let path = dir.join("narrow.toml");
    std::fs::write(
        &path,
        r#"
        baseline_cd0 = 0.0
        [grid]
        mach = [0.0, 0.1]
        alpha_deg = [-1.0, 1.0]
        [coefficients]
        cd0 = [[0.0, 0.0], [0.0, 0.0]]
        cna = [[1.0, 1.0], [1.0, 1.0]]
        pitch_damping = [[0.0, 0.0], [0.0, 0.0]]
        cm_delta = [[0.0, 0.0], [0.0, 0.0]]
        "#,
    )
    .unwrap();
    let mut cfg = Config::default();
    cfg.aero.source = AeroSource::Card;
    cfg.aero.card_path = Some(path.clone());
    let sim = Simulator::new(cfg).unwrap();
    let err = sim.run_episode(0).unwrap_err();
    assert!(matches!(err, SimError::AeroOutOfGrid { .. }), "got {err:?}");
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn generated_card_reproduces_placeholder_and_placeholder_is_counted() {
    let _guard = COUNTER_LOCK.lock().unwrap();
    let base = placeholder_uses();
    let placeholder = Simulator::new(Config::default())
        .unwrap()
        .run_episode(0)
        .unwrap();
    let after_placeholder = placeholder_uses();
    assert!(after_placeholder > base, "placeholder path must be COUNTED");

    let card = Simulator::new(card_config())
        .unwrap()
        .run_episode(0)
        .unwrap();
    assert_eq!(
        placeholder_uses(),
        after_placeholder,
        "card path must not mark"
    );
    assert_eq!(card.features.len(), placeholder.features.len());
    for (a, b) in card.features.iter().zip(placeholder.features.iter()) {
        assert!((a - b).abs() < 1e-2, "feature {a} vs {b}");
    }
    assert!(
        (card.apogee_m - placeholder.apogee_m).abs() < 1e-3,
        "card apogee mismatch"
    );
}

#[test]
fn truth_extended_v2_roundtrip_and_legacy_readers_still_work() {
    let _guard = COUNTER_LOCK.lock().unwrap();
    let cfg = truth_config();
    let sim = Simulator::new(cfg.clone()).unwrap();
    let episodes: Vec<_> = (0..2).map(|i| sim.run_episode(i).unwrap()).collect();
    let dir = temp_dir("truth");
    let path = dir.join(FILE_NAME);

    write_validation(&path, &cfg, &episodes).unwrap();
    let raw = std::fs::read(&path).unwrap();
    assert_eq!(u32::from_le_bytes(raw[8..12].try_into().unwrap()), VERSION);
    assert_eq!(
        u32::from_le_bytes(raw[48..52].try_into().unwrap()),
        META_V2_TRUTH as u32
    );
    let per_episode =
        8 + META_V2_TRUTH * 4 + cfg.frames_per_episode() * (cfg.feature_count() + TARGET_COUNT) * 4;
    assert_eq!(raw.len(), HEADER_SIZE + episodes.len() * per_episode);

    let rs = read_validation(&path).unwrap();
    assert_eq!(rs.header.meta_count as usize, META_V2_TRUTH);
    for (src, dst) in episodes.iter().zip(rs.episodes.iter()) {
        let t = src.truth.unwrap();
        assert_eq!(dst.truth, Some(t));
        assert!(t.wind.iter().all(|v| v.is_finite()));
        assert!(t.pos.iter().all(|v| v.is_finite()));
        assert!(t.vel.iter().all(|v| v.is_finite()));
        assert!(t.quat.iter().all(|v| v.is_finite()));
        assert!(t.omega.iter().all(|v| v.is_finite()));
        assert!(t.mass_kg.is_finite() && t.mass_kg > 0.0);
        assert_eq!(t.as_array().len(), TRUTH_COUNT);
        assert_eq!(dst.features[[0, 0]], src.features[0]);
        assert_eq!(dst.conditioning, src.meta.conditioning);
    }

    // A classic v2 file has 17 metadata and no truth block.
    write_validation_with_version(&path, &cfg, &episodes, VERSION).unwrap();
    let classic = read_validation(&path).unwrap();
    assert_eq!(classic.header.meta_count as usize, META_V2);
    assert!(classic.episodes.iter().all(|e| e.truth.is_none()));
    assert_eq!(classic.episodes[0].features, rs.episodes[0].features);

    // A v1 file still reads with reconstructed conditioning.
    write_validation_with_version(&path, &cfg, &episodes, VERSION_V1).unwrap();
    let legacy = read_validation(&path).unwrap();
    assert_eq!(legacy.version, VERSION_V1);
    assert_eq!(legacy.header.meta_count as usize, META_V1);
    assert!(legacy.episodes.iter().all(|e| e.truth.is_none()));
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn seeded_sensor_faults_are_deterministic_and_gated() {
    let mut cfg = card_config();
    cfg.run.seed = 4242;
    cfg.sensor.gyro_arw = 0.01;
    cfg.sensor.gyro_bias_stability = 0.001;
    cfg.sensor.accel_arw = 0.05;
    cfg.sensor.accel_bias_stability = 0.002;
    cfg.sensor.latency_steps = 2;
    cfg.sensor.gyro_quant_step = 0.001;
    cfg.sensor.accel_quant_step = 0.01;
    cfg.sensor.spike_probability = 0.02;
    cfg.sensor.spike_amplitude = 5.0;

    let a = Simulator::new(cfg.clone()).unwrap().run_episode(3).unwrap();
    let b = Simulator::new(cfg.clone()).unwrap().run_episode(3).unwrap();
    assert_eq!(a.features, b.features);
    assert_eq!(
        a.frames.iter().map(|f| f.spike).collect::<Vec<_>>(),
        b.frames.iter().map(|f| f.spike).collect::<Vec<_>>()
    );
    for (fa, fb) in a.frames.iter().zip(b.frames.iter()) {
        assert_eq!(fa.to_bytes(), fb.to_bytes());
    }

    cfg.run.seed = 4243;
    let c = Simulator::new(cfg).unwrap().run_episode(3).unwrap();
    assert_ne!(a.features, c.features);

    let spikes = a.frames.iter().filter(|f| f.spike).count();
    assert!(spikes > 0, "fault injection never fired");
    assert!(a
        .frames
        .iter()
        .filter(|f| f.spike)
        .all(|f| f.flags & FLAG_FAULT != 0 && !f.ai_input_valid()));
}

#[test]
fn legacy_sensor_defaults_stay_fault_free_and_allan_knobs_change_output() {
    let _guard = COUNTER_LOCK.lock().unwrap();
    let legacy = Simulator::new(Config::default())
        .unwrap()
        .run_episode(0)
        .unwrap();
    assert!(legacy.frames.iter().all(|f| !f.spike));

    let mut cfg = Config::default();
    cfg.sensor.gyro_arw = 0.02;
    cfg.sensor.gyro_bias_stability = 0.005;
    cfg.sensor.accel_arw = 0.08;
    cfg.sensor.accel_bias_stability = 0.01;
    cfg.sensor.latency_steps = 3;
    let noisy = Simulator::new(cfg).unwrap().run_episode(0).unwrap();
    assert!(noisy.features.iter().all(|v| v.is_finite()));
    assert_ne!(legacy.features, noisy.features);
}
