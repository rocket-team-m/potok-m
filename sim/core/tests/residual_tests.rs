//! Physics-residual features in the validation stream.
//!
//! `physics::PhysicsFeatures` already computes these six values from the true
//! 6-DoF state, true wind, atmosphere and measured specific force, and they are
//! already wired into `sensor::FEATURE_NAMES`, `Frame::physics` and the
//! versioned validation feature matrix (enabled by listing their names in
//! `output.features`). These tests pin the behaviour, not a duplicate module.

use std::path::PathBuf;

use potok_sim::{
    atmosphere::Atmosphere,
    dynamics::derivative,
    math::{mat_t_vec, quat_to_matrix},
    physics::PhysicsFeatures,
    sensor::FEATURE_NAMES,
    validation::{
        read_validation, write_validation, write_validation_with_version, FILE_NAME, HEADER_SIZE,
        LEGACY_TARGET_COUNT, TARGET_COUNT, VERSION_V1,
    },
    Config, Episode, Simulator, State, Vehicle, PHYSICS_FEATURES,
};

const A_LAT_RESID: usize = 3;

fn smart_config() -> Config {
    let mut cfg = Config::default();
    cfg.output
        .features
        .extend(PHYSICS_FEATURES.iter().map(|s| s.to_string()));
    cfg
}

fn sim(seed: u64, cfg: &Config) -> Simulator {
    let mut c = cfg.clone();
    c.run.seed = seed;
    Simulator::new(c).unwrap()
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("potok-residual-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Coast frames only: the thrust table is empty after 9.95 s, so the physics
/// baseline (drag + normal force) fully explains the true acceleration there.
fn coast_range(ep: &Episode) -> std::ops::Range<usize> {
    let start = ep
        .frames
        .iter()
        .position(|f| f.time_us as f64 >= 10.5e6)
        .unwrap_or(ep.frames.len());
    let end = ep
        .frames
        .iter()
        .position(|f| f.time_us as f64 >= 28.0e6)
        .unwrap_or(ep.frames.len());
    start..end
}

#[test]
fn field_order_is_stable_and_appended_after_raw_telemetry() {
    assert_eq!(
        &FEATURE_NAMES[FEATURE_NAMES.len() - 6..],
        &PHYSICS_FEATURES
    );
    assert_eq!(
        PHYSICS_FEATURES,
        [
            "alpha_deg",
            "beta_deg",
            "q_dyn_kpa",
            "a_lat_resid",
            "drag_est",
            "static_margin_cal"
        ]
    );
    let cfg = smart_config();
    assert_eq!(cfg.feature_count(), 14);
    let ep = sim(1337, &cfg).run_episode(0).unwrap();
    assert_eq!(ep.feature_count(), 14);
    for (i, f) in ep.frames.iter().enumerate() {
        for (j, want) in f.physics.iter().enumerate() {
            assert_eq!(ep.features[i * 14 + 8 + j], *want);
        }
    }
}

#[test]
fn steady_glide_has_small_lateral_residual() {
    let cfg = smart_config();
    // Scenario 0 (index 0) has no wind, no gust and no shear; zero control.
    let ep = sim(1337, &cfg).run_episode(0).unwrap();
    assert_eq!(ep.meta.scenario_id, 0);
    let coast = coast_range(&ep);
    assert!(coast.start < coast.end, "no coast frames");
    let mut max_resid = 0.0_f32;
    let mut sum_resid = 0.0_f32;
    for f in &ep.frames[coast.clone()] {
        let p = f.physics;
        assert!(p.iter().all(|v| v.is_finite()), "{p:?}");
        assert!(p[2] > 0.0, "dynamic pressure must be positive");
        assert!(p[4] > 0.0, "drag estimate must be positive");
        max_resid = max_resid.max(p[A_LAT_RESID]);
        sum_resid += p[A_LAT_RESID];
    }
    let mean = sum_resid / coast.len() as f32;
    eprintln!(
        "steady glide: frames={} mean_resid={mean:.6} max_resid={max_resid:.6}",
        coast.len()
    );
    // In coast with no control the measured specific force equals the
    // drag+normal baseline up to sensor bias/noise (~0.03 m/s^2 here).
    assert!(mean < 0.5, "mean residual too large: {mean}");
    assert!(max_resid < 2.0, "max residual too large: {max_resid}");
}

/// A 20 m/s crosswind on an otherwise forward-flying vehicle must move the
/// sideslip angle and open a gap between the measured specific force and the
/// drag+normal baseline, producing a nonzero lateral-acceleration residual.
#[test]
fn crosswind_gust_raises_lateral_residual() {
    let cfg = Config::default();
    let v = Vehicle::from_config(&cfg, 11.0, 1.0, [0.0, 0.0]);
    let mut s = State::new(11.0, 0.0, 0.0);
    s.vel = [120.0, 0.0, 0.0];
    let profile = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../config/boundary_profile.csv");
    let atm = Atmosphere::load_with_fast(&profile, cfg.sim.altitude_grid_m, cfg.run.fast_field)
        .unwrap();
    let air = atm.at(s.pos()[2]);
    // Specific force the sensor would report on the calm flight at coast:
    // the baseline then explains it, so the residual is ~0. Freeze that
    // measurement and inject the gust into the true wind only — exactly the
    // transient the residual is meant to expose.
    let accel = derivative(12.0, &s, [0.0; 3], &atm, &v).vel;
    let spec_world = [accel[0], accel[1], accel[2] + 9.80665];
    let sb = mat_t_vec(quat_to_matrix(s.quat()), spec_world);
    let measured = [sb[0] as f32, sb[1] as f32, sb[2] as f32];
    let calm = PhysicsFeatures::from_state(&s, [0.0; 3], air, &v, measured);
    let gust = PhysicsFeatures::from_state(&s, [0.0, 20.0, 0.0], air, &v, measured);
    eprintln!(
        "calm: beta={:.4} resid={:.4} | gust: beta={:.4} resid={:.4}",
        calm.beta_deg, calm.a_lat_resid, gust.beta_deg, gust.a_lat_resid
    );
    assert!(calm.a_lat_resid < 1e-3, "calm residual: {}", calm.a_lat_resid);
    assert!(gust.beta_deg.abs() > 5.0, "sideslip: {}", gust.beta_deg);
    assert!(gust.a_lat_resid > 1.0, "gust residual: {}", gust.a_lat_resid);
    assert!(gust.q_dyn_kpa > calm.q_dyn_kpa);
}

/// Full-path check: a real gust episode flags gust frames and those frames
/// carry a finite residual above the sensor-bias floor of a calm glide.
#[test]
fn gust_episode_carries_residual_in_the_feature_path() {
    let cfg = smart_config();
    let s = sim(1337, &cfg);
    let idx = (0..60u64)
        .find(|&i| {
            let e = s.run_episode(i).unwrap();
            e.meta.gust_start_s.is_some_and(|t| t >= 11.0)
        })
        .expect("no episode with a coast-phase gust");
    let ep = s.run_episode(idx).unwrap();
    let gust: Vec<f32> = ep
        .frames
        .iter()
        .filter(|f| f.gust)
        .map(|f| f.physics[A_LAT_RESID])
        .collect();
    assert!(!gust.is_empty(), "gust flags missing");
    assert!(gust.iter().all(|r| r.is_finite()));
    let gust_mean = gust.iter().sum::<f32>() / gust.len() as f32;
    let coast = coast_range(&ep);
    let calm: Vec<f32> = ep.frames[coast.clone()]
        .iter()
        .zip(ep.trajectory[coast].iter())
        .filter(|(f, _)| !f.gust)
        .map(|(f, _)| f.physics[A_LAT_RESID])
        .collect();
    let calm_mean = calm.iter().sum::<f32>() / calm.len() as f32;
    eprintln!(
        "episode {idx}: gust_frames={} gust_mean={gust_mean:.6} calm_mean={calm_mean:.6}",
        gust.len()
    );
    assert!(gust_mean > 0.01, "gust residual too small: {gust_mean}");
    assert!(gust_mean > calm_mean, "{gust_mean} <= {calm_mean}");
}

#[test]
fn validation_formats_keep_physics_features_without_layout_changes() {
    let cfg = smart_config();
    let s = sim(1337, &cfg);
    let episodes: Vec<_> = (0..2u64).map(|i| s.run_episode(i).unwrap()).collect();
    let frames = cfg.frames_per_episode() as usize;
    let dir = temp_dir("formats");

    for (version, meta, targets) in [
        (VERSION_V1, 9usize, LEGACY_TARGET_COUNT),
        (2, 17, LEGACY_TARGET_COUNT),
        (3, 17, TARGET_COUNT),
    ] {
        let path = dir.join(format!("v{version}.bin"));
        write_validation_with_version(&path, &cfg, &episodes, version).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(
            bytes.len(),
            HEADER_SIZE + episodes.len() * (8 + meta * 4 + frames * (14 + targets) * 4),
            "v{version} byte layout changed"
        );
        let set = read_validation(&path).unwrap();
        assert_eq!(set.version, version);
        assert_eq!(set.header.feature_count as usize, 14);
        assert_eq!(set.header.target_count as usize, targets);
        for (src, dst) in episodes.iter().zip(set.episodes.iter()) {
            assert_eq!(dst.features.ncols(), 14);
            for (i, f) in src.frames.iter().enumerate() {
                for j in 0..6 {
                    assert_eq!(dst.features[[i, 8 + j]], f.physics[j]);
                }
            }
        }
    }

    // The current writer (v3) also round-trips the six physics columns.
    let path = dir.join(FILE_NAME);
    write_validation(&path, &cfg, &episodes).unwrap();
    let set = read_validation(&path).unwrap();
    assert_eq!(set.episodes[0].features[[0, 8]], episodes[0].frames[0].physics[0]);
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Adding the physics features to `output.features` changes only the feature
/// matrix width. The zero-command actuator path stays byte-identical in all
/// three validation versions, including the six new columns.
#[test]
fn zero_command_stays_bit_identical_with_physics_features() {
    let cfg = smart_config();
    let s = sim(1337, &cfg);
    let legacy = s.run_episode(0).unwrap();
    let zero = s.run_episode_with_control(0, &|_| (0.0, 0.0)).unwrap();
    assert_eq!(legacy.features, zero.features);
    assert_eq!(legacy.trajectory, zero.trajectory);

    let dir = temp_dir("zero-bit");
    for (version, meta, targets) in [
        (VERSION_V1, 9usize, LEGACY_TARGET_COUNT),
        (2, 17, LEGACY_TARGET_COUNT),
        (3, 17, TARGET_COUNT),
    ] {
        for (tag, ep) in [("legacy", &legacy), ("zero", &zero)] {
            let path = dir.join(format!("v{version}-{tag}.bin"));
            write_validation_with_version(&path, &cfg, std::slice::from_ref(ep), version).unwrap();
            let bytes = std::fs::read(&path).unwrap();
            assert_eq!(
                bytes.len(),
                HEADER_SIZE + 8 + meta * 4 + cfg.frames_per_episode() as usize * (14 + targets) * 4
            );
        }
        assert_eq!(
            std::fs::read(dir.join(format!("v{version}-legacy.bin"))).unwrap(),
            std::fs::read(dir.join(format!("v{version}-zero.bin"))).unwrap(),
            "v{version} must be byte-identical with physics features enabled"
        );
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
