//! Dataset format and quality gates: v1/v2 truth sidecars, manifest
//! round-trip, LOUD gate failures and deterministic bytes.

use std::path::PathBuf;

use potok_sim::{
    config_hash,
    manifest::{
        manifest_from_str, manifest_to_string, read_manifest, sha256_file, sha256_hex,
        write_manifest, DatasetManifest,
    },
    quality::{
        check_dataset, check_determinism, check_split_disjoint, observed_bounds,
        QUALITY_SCHEMA_VERSION,
    },
    read_trajectory,
    trajectory::{
        sidecar_path, write_trajectory, write_trajectory_v2, write_trajectory_v3,
        write_trajectory_with_version, TRUTH_STRIDE, TRUTH_TICK_NAMES, TRUTH_V3_STRIDE,
        TRUTH_V3_TICK_NAMES, VERSION, VERSION_V2, VERSION_V3,
    },
    validation::{
        read_validation, write_validation, write_validation_with_version, HEADER_SIZE,
        LEGACY_TARGET_COUNT, TARGET_COUNT, TARGET_NAMES, VERSION_V1,
    },
    Config, Simulator,
};

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("potok-dataset-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn fixture(tag: &str, count: usize) -> (Config, Vec<potok_sim::Episode>, PathBuf) {
    let cfg = Config::default();
    let sim = Simulator::new(cfg.clone()).unwrap();
    let episodes: Vec<_> = sim.episodes(count).map(|e| e.unwrap()).collect();
    let dir = temp_dir(tag);
    (cfg, episodes, dir)
}

fn write_and_read(
    tag: &str,
    count: usize,
) -> (
    Config,
    potok_sim::validation::ValidationSet,
    potok_sim::TrajectorySet,
    PathBuf,
) {
    let (cfg, episodes, dir) = fixture(tag, count);
    let validation = dir.join("validation.bin");
    write_validation(&validation, &cfg, &episodes).unwrap();
    let sidecar = sidecar_path(&validation);
    write_trajectory_v3(&sidecar, &cfg, &episodes).unwrap();
    (
        cfg,
        read_validation(&validation).unwrap(),
        read_trajectory(&sidecar).unwrap(),
        dir,
    )
}

#[test]
fn v1_sidecar_bytes_and_reader_stay_compatible() {
    let (cfg, episodes, dir) = fixture("v1", 2);
    let validation = dir.join("validation.bin");
    let sidecar = sidecar_path(&validation);
    write_trajectory(&sidecar, &cfg, &episodes).unwrap();

    let bytes = std::fs::read(&sidecar).unwrap();
    let frames = cfg.frames_per_episode();
    assert_eq!(
        bytes.len(),
        64 + episodes.len() * (8 + frames * 6 * 4),
        "v1 layout must stay 6 floats per tick"
    );
    assert_eq!(&bytes[0..8], b"POTOKTRJ");
    assert_eq!(
        u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
        VERSION
    );
    assert_eq!(u32::from_le_bytes(bytes[20..24].try_into().unwrap()), 6);

    let set = read_trajectory(&sidecar).unwrap();
    assert_eq!(set.version, VERSION);
    assert!(!set.is_v2());
    assert_eq!(set.episodes.len(), 2);
    assert_eq!(set.episodes[0].ticks.len(), frames);
    assert_eq!(set.episodes[0].ticks[0].pos, episodes[0].trajectory[0].pos);
    assert_eq!(set.episodes[0].ticks[0].vel, episodes[0].trajectory[0].vel);
    // V1 has no privileged fields; the reader must say so explicitly.
    assert_eq!(set.episodes[0].ticks[0].gust_start_s, -1.0);
    assert_eq!(set.episodes[0].ticks[0].g_theta, [0.0, 0.0]);
    assert_eq!(set.episodes[0].ticks[0].quat, [0.0; 4]);
}

#[test]
fn v2_sidecar_carries_privileged_truth_at_every_tick() {
    let (cfg, episodes, dir) = fixture("v2", 2);
    let sidecar = dir.join("truth.trajectory");
    write_trajectory_v2(&sidecar, &cfg, &episodes).unwrap();
    let bytes = std::fs::read(&sidecar).unwrap();
    let frames = cfg.frames_per_episode();
    assert_eq!(
        bytes.len(),
        64 + episodes.len() * (8 + frames * TRUTH_STRIDE * 4)
    );
    assert_eq!(
        u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
        VERSION_V2
    );
    assert_eq!(
        u32::from_le_bytes(bytes[20..24].try_into().unwrap()),
        TRUTH_STRIDE as u32
    );
    assert_eq!(TRUTH_TICK_NAMES.len(), TRUTH_STRIDE);

    let set = read_trajectory(&sidecar).unwrap();
    assert_eq!(set.version, VERSION_V2);
    assert!(set.is_v2());
    for (got, want) in set.episodes.iter().zip(episodes.iter()) {
        for (t, src) in got.ticks.iter().zip(want.trajectory.iter()) {
            // v2 carries the first 24 fields; the v3-only g_accel pair is
            // explicitly zero in a v2 read and is checked by the v3 test.
            assert_eq!(
                t.as_truth_array(),
                src.as_truth_array(),
                "v2 round trip must preserve every v2 field"
            );
            assert_eq!(t.g_accel, [0.0, 0.0]);
            assert!(t.mass_kg > 0.0);
            let q2: f64 = t.quat.iter().map(|x| (*x as f64).powi(2)).sum();
            assert!((q2.sqrt() - 1.0).abs() < 0.1);
        }
    }
    assert!(
        set.episodes[0]
            .ticks
            .iter()
            .any(|t| t.wind.iter().any(|w| w.abs() > 1e-3)),
        "true wind must be populated"
    );
    assert!(
        set.episodes[0]
            .ticks
            .iter()
            .any(|t| t.g_theta.iter().any(|g| g.abs() > 1e-6)),
        "reference optimizer corrections must be present"
    );
}

#[test]
fn v3_sidecar_carries_g_accel_per_tick_and_keeps_the_v2_prefix() {
    let (cfg, episodes, dir) = fixture("v3-traj", 2);
    let sidecar = dir.join("truth-v3.trajectory");
    write_trajectory_v3(&sidecar, &cfg, &episodes).unwrap();
    let bytes = std::fs::read(&sidecar).unwrap();
    let frames = cfg.frames_per_episode();
    assert_eq!(
        bytes.len(),
        64 + episodes.len() * (8 + frames * TRUTH_V3_STRIDE * 4)
    );
    assert_eq!(
        u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
        VERSION_V3
    );
    assert_eq!(
        u32::from_le_bytes(bytes[20..24].try_into().unwrap()),
        TRUTH_V3_STRIDE as u32
    );
    assert_eq!(TRUTH_V3_TICK_NAMES.len(), TRUTH_V3_STRIDE);
    assert_eq!(&TRUTH_V3_TICK_NAMES[..TRUTH_STRIDE], &TRUTH_TICK_NAMES);
    assert_eq!(TRUTH_V3_TICK_NAMES[24], "g_accel_y");
    assert_eq!(TRUTH_V3_TICK_NAMES[25], "g_accel_x");

    let set = read_trajectory(&sidecar).unwrap();
    assert_eq!(set.version, VERSION_V3);
    assert!(set.is_v3() && set.is_privileged() && !set.is_v2());
    for (got, want) in set.episodes.iter().zip(episodes.iter()) {
        for (t, src) in got.ticks.iter().zip(want.trajectory.iter()) {
            assert_eq!(*t, *src, "v3 round trip must preserve every field");
            assert!(t.g_accel.iter().all(|g| g.is_finite()));
            assert!(t.g_accel.iter().all(|g| g.abs() <= 10.0));
        }
    }
    assert!(
        set.episodes[0]
            .ticks
            .iter()
            .any(|t| t.g_accel.iter().any(|g| g.abs() > 1e-6)),
        "the command target must be populated on a coast/descent tick"
    );
}

#[test]
fn validation_v1_v2_v3_target_layout_and_field_order() {
    let (cfg, episodes, dir) = fixture("v3-validation", 2);
    let frames = cfg.frames_per_episode();
    let features = cfg.feature_count();
    let sizes = [
        (VERSION_V1, 9usize, LEGACY_TARGET_COUNT),
        (VERSION_V2, 17, LEGACY_TARGET_COUNT),
        (VERSION_V3, 17, TARGET_COUNT),
    ];
    for (version, meta, targets) in sizes {
        let path = dir.join(format!("v{version}.bin"));
        write_validation_with_version(&path, &cfg, &episodes, version).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(
            bytes.len(),
            HEADER_SIZE + episodes.len() * (8 + meta * 4 + frames * (features + targets) * 4)
        );
        assert_eq!(
            u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            version
        );
        assert_eq!(
            u32::from_le_bytes(bytes[32..36].try_into().unwrap()),
            targets as u32
        );
        let set = read_validation(&path).unwrap();
        assert_eq!(set.version, version);
        assert_eq!(set.header.target_count as usize, targets);
        assert_eq!(set.episodes[0].targets.ncols(), targets);
        // The legacy four columns are byte-identical in every version.
        for row in 0..frames {
            let got = &set.episodes[0].targets[[row, 0]];
            let src = &episodes[0].targets[row];
            assert_eq!(*got, src.g_theta[0]);
            assert_eq!(set.episodes[0].targets[[row, 1]], src.g_theta[1]);
            assert_eq!(set.episodes[0].targets[[row, 2]], src.confidence);
            assert_eq!(
                set.episodes[0].targets[[row, 3]],
                src.ai_healthy as u8 as f32
            );
        }
        if version == VERSION_V3 {
            assert_eq!(TARGET_NAMES.len(), TARGET_COUNT);
            assert_eq!(TARGET_NAMES[4], "g_accel_y");
            assert_eq!(TARGET_NAMES[5], "g_accel_x");
            for row in 0..frames {
                assert_eq!(
                    set.episodes[0].targets[[row, 4]],
                    episodes[0].targets[row].g_accel[0]
                );
                assert_eq!(
                    set.episodes[0].targets[[row, 5]],
                    episodes[0].targets[row].g_accel[1]
                );
            }
            assert!(
                (0..frames).any(|r| set.episodes[0].targets[[r, 4]].abs() > 1e-6),
                "v3 must carry a nonzero command target"
            );
        }
    }
}

#[test]
fn version_and_target_count_mismatches_are_loud() {
    let (cfg, episodes, dir) = fixture("v3-mismatch", 1);
    let v2 = dir.join("v2.bin");
    write_validation_with_version(&v2, &cfg, &episodes, VERSION_V2).unwrap();
    let v3 = dir.join("v3.bin");
    write_validation_with_version(&v3, &cfg, &episodes, VERSION_V3).unwrap();

    let mut bytes = std::fs::read(&v2).unwrap();
    bytes[32..36].copy_from_slice(&(TARGET_COUNT as u32).to_le_bytes());
    let bad = dir.join("bad-v2.bin");
    std::fs::write(&bad, &bytes).unwrap();
    assert!(read_validation(&bad)
        .unwrap_err()
        .to_string()
        .contains("targets"));

    let mut bytes = std::fs::read(&v3).unwrap();
    bytes[32..36].copy_from_slice(&(LEGACY_TARGET_COUNT as u32).to_le_bytes());
    std::fs::write(&bad, &bytes).unwrap();
    assert!(read_validation(&bad)
        .unwrap_err()
        .to_string()
        .contains("targets"));

    let mut bytes = std::fs::read(&v3).unwrap();
    bytes[8..12].copy_from_slice(&4u32.to_le_bytes());
    std::fs::write(&bad, &bytes).unwrap();
    assert!(read_validation(&bad)
        .unwrap_err()
        .to_string()
        .contains("version"));
}

#[test]
fn fin_accel_command_matches_the_exported_command_target() {
    use potok_sim::{ControlContext, ReferenceOptimizer};
    use std::cell::RefCell;

    let sim = Simulator::new(Config::default()).unwrap();
    let opt = ReferenceOptimizer::new(sim.config());
    let flown: RefCell<Vec<(f64, f64)>> = RefCell::new(Vec::new());
    let ep = sim
        .run_episode_with_control_ctx(17, &|ctx: &ControlContext| {
            let g = opt.fin_accel_command(ctx.state, ctx.wind, ctx.vehicle, ctx.phase, ctx.frame);
            flown.borrow_mut().push((g[0], g[1]));
            (g[0], g[1])
        })
        .unwrap();
    let commands = flown.into_inner();
    assert_eq!(commands.len(), ep.targets.len());
    for (i, ((roll, pitch), target)) in commands.iter().zip(ep.targets.iter()).enumerate() {
        let want_y = target.g_accel[0] as f64;
        let want_x = target.g_accel[1] as f64;
        assert!(
            (roll - want_y).abs() < 2e-6,
            "tick {i}: roll {roll} vs g_accel_y {want_y}"
        );
        assert!(
            (pitch + want_x).abs() < 2e-6,
            "tick {i}: pitch {pitch} vs -g_accel_x {want_x}"
        );
        if !target.ai_healthy {
            assert_eq!(*roll, 0.0);
            assert_eq!(*pitch, 0.0);
        }
    }

    // The same command survives both v3 exports.
    let dir = temp_dir("fin-accel");
    let validation = dir.join("validation.bin");
    write_validation(&validation, &Config::default(), &[ep.clone()]).unwrap();
    let sidecar = sidecar_path(&validation);
    write_trajectory_v3(&sidecar, &Config::default(), &[ep.clone()]).unwrap();
    let v = read_validation(&validation).unwrap();
    let t = read_trajectory(&sidecar).unwrap();
    for i in 0..ep.targets.len() {
        let y = v.episodes[0].targets[[i, 4]];
        let x = v.episodes[0].targets[[i, 5]];
        assert_eq!(y, t.episodes[0].ticks[i].g_accel[0]);
        assert_eq!(x, t.episodes[0].ticks[i].g_accel[1]);
        assert!((commands[i].0 - y as f64).abs() < 2e-6);
        assert!((commands[i].1 + x as f64).abs() < 2e-6);
    }
}

#[test]
fn zero_command_keeps_every_serialized_format_bit_identical() {
    let mut cfg = Config::default();
    cfg.run.seed = 1337;
    let sim = Simulator::new(cfg.clone()).unwrap();
    let legacy = sim.run_episode(0).unwrap();
    let zero = sim.run_episode_with_control(0, &|_| (0.0, 0.0)).unwrap();
    assert_eq!(zero.targets, legacy.targets);
    assert_eq!(zero.trajectory, legacy.trajectory);
    let dir = temp_dir("zero-bit");
    for (version, meta) in [(VERSION_V1, 9), (VERSION_V2, 17), (VERSION_V3, 17)] {
        let targets = if version == VERSION_V3 {
            TARGET_COUNT
        } else {
            LEGACY_TARGET_COUNT
        };
        for (tag, ep) in [("legacy", &legacy), ("zero", &zero)] {
            let path = dir.join(format!("v{version}-{tag}.bin"));
            write_validation_with_version(&path, &cfg, std::slice::from_ref(ep), version).unwrap();
            let bytes = std::fs::read(&path).unwrap();
            assert_eq!(
                bytes.len(),
                HEADER_SIZE
                    + 8
                    + meta * 4
                    + cfg.frames_per_episode() * (cfg.feature_count() + targets) * 4
            );
        }
        assert_eq!(
            std::fs::read(dir.join(format!("v{version}-legacy.bin"))).unwrap(),
            std::fs::read(dir.join(format!("v{version}-zero.bin"))).unwrap(),
            "v{version} validation bytes must match"
        );
    }
    for (version, stride) in [
        (VERSION, 6usize),
        (VERSION_V2, 24),
        (VERSION_V3, TRUTH_V3_STRIDE),
    ] {
        for (tag, ep) in [("legacy", &legacy), ("zero", &zero)] {
            let path = dir.join(format!("traj-v{version}-{tag}"));
            write_trajectory_with_version(&path, &cfg, std::slice::from_ref(ep), version).unwrap();
            assert_eq!(
                std::fs::metadata(&path).unwrap().len() as usize,
                64 + 8 + cfg.frames_per_episode() * stride * 4
            );
        }
        assert_eq!(
            std::fs::read(dir.join(format!("traj-v{version}-legacy"))).unwrap(),
            std::fs::read(dir.join(format!("traj-v{version}-zero"))).unwrap(),
            "v{version} sidecar bytes must match"
        );
    }
}

#[test]
fn manifest_roundtrip_is_deterministic_and_hashes_the_files() {
    let (cfg, _v, t, dir) = write_and_read("manifest", 2);
    let validation = dir.join("validation.bin");
    let sidecar = sidecar_path(&validation);
    let mut scenario_ids: Vec<u16> = _v.episodes.iter().map(|e| e.scenario_id).collect();
    scenario_ids.sort_unstable();
    scenario_ids.dedup();
    let manifest = DatasetManifest {
        schema_version: QUALITY_SCHEMA_VERSION,
        config_hash: config_hash(&cfg).unwrap(),
        seed: cfg.run.seed,
        split: "train".into(),
        episode_count: 2,
        frame_count: 2 * cfg.frames_per_episode() as u64,
        frames_per_episode: cfg.frames_per_episode() as u32,
        scenario_ids,
        feature_names: cfg.output.features.clone(),
        target_names: potok_sim::validation::TARGET_NAMES
            .iter()
            .map(|s| s.to_string())
            .collect(),
        truth_names: TRUTH_V3_TICK_NAMES.iter().map(|s| s.to_string()).collect(),
        bounds: observed_bounds(&cfg, &_v, &t),
        files: vec![
            potok_sim::file_record("validation.bin", &validation).unwrap(),
            potok_sim::file_record("validation.trajectory", &sidecar).unwrap(),
        ],
    };
    let text = manifest_to_string(&manifest).unwrap();
    assert_eq!(text, manifest_to_string(&manifest).unwrap());
    assert!(
        !text.contains("timestamp"),
        "no timestamps in deterministic bytes"
    );
    assert_eq!(manifest_from_str(&text).unwrap(), manifest);
    assert_eq!(sha256_file(&validation).unwrap(), manifest.files[0].sha256);
    assert_eq!(sha256_file(&sidecar).unwrap(), manifest.files[1].sha256);
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );

    let path = dir.join("manifest.json");
    write_manifest(&path, &manifest).unwrap();
    assert_eq!(read_manifest(&path).unwrap(), manifest);
    assert_eq!(std::fs::read_to_string(&path).unwrap(), text);
}

#[test]
fn quality_gates_fail_loudly_on_corrupt_data() {
    let (cfg, mut v, mut t, _dir) = write_and_read("gates", 2);
    let min = 2;

    // Baseline must pass before each mutation.
    check_dataset("train", &cfg, &v, &t, min).unwrap();

    v.episodes[0].features[[0, 0]] = f32::NAN;
    let err = check_dataset("train", &cfg, &v, &t, min)
        .unwrap_err()
        .to_string();
    assert!(err.contains("finite") && err.contains("features"), "{err}");
    v.episodes[0].features[[0, 0]] = 0.0;

    v.episodes[0].targets[[0, 0]] = 0.9;
    let err = check_dataset("train", &cfg, &v, &t, min)
        .unwrap_err()
        .to_string();
    assert!(err.contains("target_bounds"), "{err}");
    v.episodes[0].targets[[0, 0]] = 0.0;

    v.episodes[0].features[[0, 0]] = 1.0e9;
    let err = check_dataset("train", &cfg, &v, &t, min)
        .unwrap_err()
        .to_string();
    assert!(err.contains("feature_bounds"), "{err}");
    v.episodes[0].features[[0, 0]] = 0.0;

    t.episodes[0].ticks[0].omega[0] = f32::INFINITY;
    let err = check_dataset("train", &cfg, &v, &t, min)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("finite") && err.contains("per-tick truth"),
        "{err}"
    );
    t.episodes[0].ticks[0].omega[0] = 0.0;

    t.episodes[0].ticks.pop();
    let err = check_dataset("train", &cfg, &v, &t, min)
        .unwrap_err()
        .to_string();
    assert!(err.contains("frame_count"), "{err}");
    let (_, _, t_ok, _) = write_and_read("gates-restore", 2);
    t = t_ok;

    let err = check_dataset("train", &cfg, &v, &t, 3)
        .unwrap_err()
        .to_string();
    assert!(err.contains("episode_count"), "{err}");

    let err = check_split_disjoint(&v, &v).unwrap_err().to_string();
    assert!(err.contains("split_scenarios_disjoint"), "{err}");

    let err = check_determinism(&[("validation.bin", b"a", b"b")])
        .unwrap_err()
        .to_string();
    assert!(err.contains("deterministic_regeneration"), "{err}");
}

#[test]
fn disjoint_split_gate_accepts_distinct_scenario_sets() {
    let (cfg, train, _t, dir) = write_and_read("disjoint", 4);
    let sim = Simulator::new(cfg.clone()).unwrap();
    let held_episodes: Vec<_> = (16..20).map(|i| sim.run_episode(i).unwrap()).collect();
    let held_path = dir.join("held.bin");
    write_validation(&held_path, &cfg, &held_episodes).unwrap();
    let held = read_validation(&held_path).unwrap();
    let check = check_split_disjoint(&train, &held).unwrap();
    assert!(check.ok);
    assert!(check.detail.contains("overlap=[]"));
}
