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
        sidecar_path, write_trajectory, write_trajectory_v2, TRUTH_STRIDE, TRUTH_TICK_NAMES,
        VERSION, VERSION_V2,
    },
    validation::{read_validation, write_validation},
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
    write_trajectory_v2(&sidecar, &cfg, &episodes).unwrap();
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
            assert_eq!(*t, *src, "v2 round trip must preserve every field");
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
        truth_names: TRUTH_TICK_NAMES.iter().map(|s| s.to_string()).collect(),
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
