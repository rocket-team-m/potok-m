//! The 14-value model stream: exact column order and the gust residual slot.
//!
//! `config/default.toml` lists eight raw telemetry channels; `--physics-features`
//! appends the six `PHYSICS_FEATURES`, giving the canonical 14-column frame.
//! These tests pin the order at every place that can silently reorder it —
//! config, the `sensor::FEATURE_NAMES` lookup table, the row-major feature
//! matrix and the validation header — and pin the lateral acceleration
//! residual to its exact column (11). Nothing is re-implemented
//! here: the tests exercise the existing config, episode and physics modules.

use std::path::Path;

use potok_sim::{
    atmosphere::Atmosphere,
    dynamics::derivative,
    math::{mat_t_vec, quat_to_matrix},
    sensor::{self, FEATURE_NAMES},
    validation::{read_validation, write_validation, FILE_NAME},
    Config, Episode, PhysicsFeatures, Simulator, State, Vehicle, PHYSICS_FEATURES,
};

/// The canonical model frame, in wire-to-model order. Eight raw channels
/// followed by the six physics residuals. Changing this array is a breaking
/// change to every downstream model input.
const STREAM: [&str; 14] = [
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "acc_x",
    "acc_y",
    "acc_z",
    "baro_h",
    "pitot_q",
    "alpha_deg",
    "beta_deg",
    "q_dyn_kpa",
    "a_lat_resid",
    "drag_est",
    "static_margin_cal",
];

const RAW_COUNT: usize = 8;
const A_LAT_RESID_SLOT: usize = 3;
const GUST_RESIDUAL_COLUMN: usize = RAW_COUNT + A_LAT_RESID_SLOT;

fn smart_config() -> Config {
    let mut cfg = Config::default();
    cfg.output
        .features
        .extend(PHYSICS_FEATURES.iter().map(|s| s.to_string()));
    cfg
}

fn episode(seed: u64, index: u64, cfg: &Config) -> Episode {
    let mut c = cfg.clone();
    c.run.seed = seed;
    Simulator::new(c).unwrap().run_episode(index).unwrap()
}

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "potok-residual-stream-{tag}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn canonical_stream_is_default_telemetry_then_physics() {
    let cfg = Config::default();
    let want: Vec<String> = STREAM.iter().map(|s| s.to_string()).collect();

    assert_eq!(cfg.output.features.as_slice(), &want[..RAW_COUNT]);
    assert_eq!(PHYSICS_FEATURES, &STREAM[RAW_COUNT..]);
    assert_eq!(smart_config().output.features, want);
    assert_eq!(GUST_RESIDUAL_COLUMN, 11);
    assert_eq!(PHYSICS_FEATURES[A_LAT_RESID_SLOT], "a_lat_resid");

    // The lookup table may carry aliases (mag_x/y/z, pitot_dp), but the 14
    // canonical names must still appear in the same relative order, and the
    // six physics names must be its final six entries.
    let mut cursor = 0;
    for name in STREAM {
        let at = FEATURE_NAMES[cursor..]
            .iter()
            .position(|n| *n == name)
            .unwrap_or_else(|| panic!("{name} missing from FEATURE_NAMES"));
        cursor += at + 1;
    }
    assert_eq!(&FEATURE_NAMES[FEATURE_NAMES.len() - 6..], PHYSICS_FEATURES);
}

#[test]
fn canonical_matrix_columns_match_named_feature_lookup() {
    let cfg = smart_config();
    let ep = episode(1337, 0, &cfg);
    assert_eq!(cfg.feature_count(), STREAM.len());
    assert_eq!(ep.feature_count(), STREAM.len());
    assert_eq!(ep.features.len(), ep.frames.len() * STREAM.len());

    for (i, frame) in ep.frames.iter().enumerate() {
        let row = &ep.features[i * STREAM.len()..(i + 1) * STREAM.len()];
        for (col, name) in STREAM.iter().enumerate() {
            assert_eq!(
                row[col],
                sensor::feature(frame, name).unwrap(),
                "frame {i} {name}"
            );
        }
        for slot in 0..PHYSICS_FEATURES.len() {
            assert_eq!(
                row[RAW_COUNT + slot],
                frame.physics[slot],
                "frame {i} physics[{slot}]"
            );
        }
    }
}

#[test]
fn configured_order_drives_columns_not_the_lookup_table_order() {
    let mut cfg = Config::default();
    let names = ["pitot_q", "alpha_deg", "gyro_z", "acc_x"];
    cfg.output.features = names.iter().map(|s| s.to_string()).collect();
    let ep = episode(1337, 0, &cfg);
    assert_eq!(ep.feature_count(), names.len());

    for (i, frame) in ep.frames.iter().enumerate() {
        let row = &ep.features[i * names.len()..(i + 1) * names.len()];
        for (col, name) in names.iter().enumerate() {
            assert_eq!(
                row[col],
                sensor::feature(frame, name).unwrap(),
                "frame {i} {name}"
            );
        }
    }
}

#[test]
fn validation_header_pins_stream_width() {
    let cfg = smart_config();
    let episodes: Vec<_> = (0..2).map(|i| episode(1337, i, &cfg)).collect();
    let dir = temp_dir("header");
    let path = dir.join(FILE_NAME);
    write_validation(&path, &cfg, &episodes).unwrap();

    let set = read_validation(&path).unwrap();
    assert_eq!(set.header.feature_count as usize, STREAM.len());
    for src in &episodes {
        let dst = set
            .episodes
            .iter()
            .find(|e| e.index == src.meta.index)
            .unwrap();
        assert_eq!(dst.features.ncols(), STREAM.len());
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Freeze the measurement of a calm coast frame, then inject a 20 m/s
/// crosswind into the true wind only. That isolates the quantity
/// `a_lat_resid` is designed to expose: the gap between measured specific
/// force and the drag+normal baseline. The same value must land in the
/// stream's column 11, not in any other physics slot.
#[test]
fn crosswind_opens_lateral_residual_at_stream_column_eleven() {
    let cfg = Config::default();
    let vehicle = Vehicle::from_config(&cfg, 11.0, 1.0, [0.0, 0.0]);
    let mut state = State::new(11.0, 0.0, 0.0);
    state.vel = [120.0, 0.0, 0.0];
    let profile = Path::new(env!("CARGO_MANIFEST_DIR")).join("../config/boundary_profile.csv");
    let atm =
        Atmosphere::load_with_fast(&profile, cfg.sim.altitude_grid_m, cfg.run.fast_field).unwrap();
    let air = atm.at(state.pos()[2]);

    let accel = derivative(12.0, &state, [0.0; 3], &atm, &vehicle).vel;
    let spec_world = [accel[0], accel[1], accel[2] + 9.80665];
    let body = mat_t_vec(quat_to_matrix(state.quat()), spec_world);
    let measured = [body[0] as f32, body[1] as f32, body[2] as f32];

    let calm = PhysicsFeatures::from_state(&state, [0.0; 3], air, &vehicle, measured);
    let gust = PhysicsFeatures::from_state(&state, [0.0, 20.0, 0.0], air, &vehicle, measured);

    assert_eq!(calm.as_array()[A_LAT_RESID_SLOT], calm.a_lat_resid);
    assert_eq!(gust.as_array()[A_LAT_RESID_SLOT], gust.a_lat_resid);
    assert!(
        calm.a_lat_resid < 1e-3,
        "calm residual: {}",
        calm.a_lat_resid
    );
    assert!(gust.beta_deg.abs() > 5.0, "sideslip: {}", gust.beta_deg);
    assert!(
        gust.a_lat_resid > 1.0,
        "gust residual: {}",
        gust.a_lat_resid
    );
    assert!(gust.q_dyn_kpa > calm.q_dyn_kpa);
}

#[test]
fn gust_frames_carry_residual_in_stream_column_eleven() {
    let cfg = smart_config();
    let mut sim_cfg = cfg.clone();
    // Prefer a coast-phase gust so boost transients cannot dominate the
    // residual. The first scenario cycle always contains at least one.
    let ep = (0..potok_sim::SCENARIO_COUNT as u64)
        .map(|i| {
            sim_cfg.run.seed = 1337;
            Simulator::new(sim_cfg.clone())
                .unwrap()
                .run_episode(i)
                .unwrap()
        })
        .find(|e| e.meta.gust_start_s.is_some_and(|t| t >= 11.0))
        .expect("no coast-phase gust in the first scenario cycle");

    let rows: Vec<&[f32]> = ep.features.chunks_exact(STREAM.len()).collect();
    assert_eq!(rows.len(), ep.frames.len());

    let mut gust_mean = 0.0_f32;
    let mut gust_count = 0usize;
    let mut calm_mean = 0.0_f32;
    let mut calm_count = 0usize;
    for (frame, row) in ep.frames.iter().zip(rows.iter()) {
        assert_eq!(row[GUST_RESIDUAL_COLUMN], frame.physics[A_LAT_RESID_SLOT]);
        let value = row[GUST_RESIDUAL_COLUMN];
        assert!(
            value.is_finite(),
            "non-finite residual at {:?} s",
            frame.time_us
        );
        if frame.gust {
            gust_mean += value;
            gust_count += 1;
        } else if frame.time_us >= 10_500_000 {
            calm_mean += value;
            calm_count += 1;
        }
    }
    assert!(gust_count > 0, "gust flags missing");
    assert!(calm_count > 0, "no calm coast frames");
    gust_mean /= gust_count as f32;
    calm_mean /= calm_count as f32;
    eprintln!(
        "episode {}: gust_frames={gust_count} gust_mean={gust_mean:.6} calm_mean={calm_mean:.6}",
        ep.meta.index
    );
    assert!(gust_mean > 0.01, "gust residual too small: {gust_mean}");
    assert!(gust_mean > calm_mean, "{gust_mean} <= {calm_mean}");
}

#[test]
fn stream_is_bit_identical_for_the_same_seed() {
    let cfg = smart_config();
    let a = episode(1337, 2, &cfg);
    let b = episode(1337, 2, &cfg);
    assert_eq!(a.feature_count(), STREAM.len());
    assert_eq!(a.frames.len(), b.frames.len());
    assert_eq!(
        a.features, b.features,
        "same seed produced different streams"
    );
}
