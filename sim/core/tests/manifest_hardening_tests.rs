// SPDX-License-Identifier: GPL-3.0-or-later
//! SIM04 — hardening gates for the deterministic dataset manifest.
//!
//! The manifest is the trust boundary of the generated dataset: every field is
//! derived from the config, the seed and the bytes actually written. These
//! tests pin that contract — schema v1, the config->hash binding, the recorded
//! seed, disjoint train/held-out scenarios, sha256 over both files, observed
//! bounds, the absence of timestamps, and LOUD failure when anything is
//! corrupted. No source module is edited here; a red test means the contract
//! regressed.

use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};

use potok_sim::{
    config_hash,
    manifest::{
        canonical_config_bytes, default_target_names, default_truth_names, file_record,
        manifest_from_str, manifest_to_string, read_manifest, sha256_file, sha256_hex,
        write_manifest, DatasetManifest, SCHEMA_VERSION,
    },
    quality::{
        check_dataset, check_determinism, check_split_disjoint, observed_bounds,
        QUALITY_SCHEMA_VERSION,
    },
    read_trajectory, scenario, sidecar_path,
    trajectory::{write_trajectory_v3, TRUTH_V3_TICK_NAMES},
    validation::{read_validation, write_validation, ValidationSet, FILE_NAME, TARGET_NAMES},
    Config, Episode, Simulator, Split, TrajectorySet,
};

fn temp(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("potok-sim04-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");
    dir
}

fn sim_with_seed(seed: u64) -> (Config, Simulator) {
    let mut cfg = Config::default();
    cfg.run.seed = seed;
    let sim = Simulator::new(cfg.clone()).expect("simulator");
    (cfg, sim)
}

fn split_indices(split: Split, count: usize) -> Vec<u64> {
    (0..)
        .filter(|&i| scenario(i).split == split)
        .take(count)
        .collect()
}

fn run_episodes(sim: &Simulator, ids: &[u64]) -> Vec<Episode> {
    ids.iter().map(|&i| sim.run_episode(i).unwrap()).collect()
}

fn write_validation_of(path: &Path, cfg: &Config, eps: &[Episode]) -> ValidationSet {
    write_validation(path, cfg, eps).unwrap();
    read_validation(path).unwrap()
}

/// Writes one split exactly as the dataset generator does: validation.bin, the
/// v3 truth sidecar, and the manifest. Returns the read-back sets so the gates
/// check the bytes that are actually on disk.
fn write_split(
    dir: &Path,
    cfg: &Config,
    split: Split,
    eps: &[Episode],
) -> (ValidationSet, TrajectorySet, DatasetManifest) {
    std::fs::create_dir_all(dir).unwrap();
    let validation = dir.join(FILE_NAME);
    let sidecar = sidecar_path(&validation);
    let v = write_validation_of(&validation, cfg, eps);
    write_trajectory_v3(&sidecar, cfg, eps).unwrap();
    let t = read_trajectory(&sidecar).unwrap();

    let mut scenario_ids: Vec<u16> = v.episodes.iter().map(|e| e.scenario_id).collect();
    scenario_ids.sort_unstable();
    scenario_ids.dedup();

    let manifest = DatasetManifest {
        schema_version: SCHEMA_VERSION,
        config_hash: config_hash(cfg).unwrap(),
        seed: cfg.run.seed,
        split: split.label().to_string(),
        scenario_ids,
        episode_count: v.episodes.len() as u32,
        frame_count: (v.episodes.len() as u64) * (v.header.frames_per_episode as u64),
        frames_per_episode: v.header.frames_per_episode,
        feature_names: cfg.output.features.clone(),
        target_names: default_target_names(),
        truth_names: default_truth_names(),
        bounds: observed_bounds(cfg, &v, &t),
        files: vec![
            file_record(FILE_NAME, &validation).unwrap(),
            file_record("validation.trajectory", &sidecar).unwrap(),
        ],
    };
    write_manifest(&dir.join("manifest.json"), &manifest).unwrap();
    (v, t, manifest)
}

#[test]
fn manifest_schema_is_v1_with_exactly_the_expected_shape() {
    assert_eq!(SCHEMA_VERSION, 1, "manifest wire schema must be v1");
    assert_eq!(
        QUALITY_SCHEMA_VERSION, 1,
        "quality report schema must be v1"
    );

    let dir = temp("schema");
    let (cfg, sim) = sim_with_seed(11);
    let (_v, _t, m) = write_split(&dir, &cfg, Split::Train, &run_episodes(&sim, &[0]));
    assert_eq!(m.schema_version, SCHEMA_VERSION);

    let text = manifest_to_string(&m).unwrap();
    assert!(text.contains("\"schema_version\": 1"), "{text}");
    assert_eq!(manifest_from_str(&text).unwrap(), m);
    assert_eq!(read_manifest(&dir.join("manifest.json")).unwrap(), m);

    // The exact key set is the schema: an added field is a wire change and an
    // added timestamp field would be a determinism regression.
    let value: serde_json::Value = serde_json::from_str(&text).unwrap();
    let obj = value.as_object().unwrap();
    let keys: BTreeSet<&str> = obj.keys().map(String::as_str).collect();
    let expected: BTreeSet<&str> = [
        "schema_version",
        "config_hash",
        "seed",
        "split",
        "scenario_ids",
        "episode_count",
        "frame_count",
        "frames_per_episode",
        "feature_names",
        "target_names",
        "truth_names",
        "bounds",
        "files",
    ]
    .into_iter()
    .collect();
    assert_eq!(keys, expected, "manifest top-level fields changed");

    let bounds: BTreeSet<&str> = obj["bounds"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(
        bounds,
        ["features", "targets", "truth"]
            .into_iter()
            .collect::<BTreeSet<&str>>()
    );
    for rec in obj["files"].as_array().unwrap() {
        let file_keys: BTreeSet<&str> = rec
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(
            file_keys,
            ["name", "bytes", "sha256"]
                .into_iter()
                .collect::<BTreeSet<&str>>()
        );
    }

    // A foreign schema version is not silently rewritten: it stays visible so
    // a reader can reject it loudly at its own boundary.
    let mut bumped = value.clone();
    bumped["schema_version"] = serde_json::json!(99);
    let parsed = manifest_from_str(&bumped.to_string()).unwrap();
    assert_eq!(parsed.schema_version, 99);
    assert_ne!(parsed.schema_version, SCHEMA_VERSION);
}

#[test]
fn config_hash_binds_the_canonical_toml_and_moves_with_any_field() {
    let cfg = Config::default();
    let canonical = canonical_config_bytes(&cfg).unwrap();
    let hash = config_hash(&cfg).unwrap();
    assert_eq!(hash, sha256_hex(&canonical));
    assert_eq!(hash.len(), 64);
    assert!(
        hash.bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
        "hash must be lowercase hex: {hash}"
    );
    assert_eq!(
        config_hash(&cfg.clone()).unwrap(),
        hash,
        "hash is a pure function"
    );

    let mut seed = cfg.clone();
    seed.run.seed += 1;
    assert_ne!(
        config_hash(&seed).unwrap(),
        hash,
        "seed is part of the hash"
    );

    let mut wind = cfg.clone();
    wind.wind.v_10 += 0.5;
    assert_ne!(
        config_hash(&wind).unwrap(),
        hash,
        "every config field is bound"
    );

    let mut dt = cfg.clone();
    dt.run.dt *= 2.0;
    assert_ne!(config_hash(&dt).unwrap(), hash, "numeric fields are bound");
}

#[test]
fn manifest_records_the_seed_and_a_new_seed_moves_the_bytes() {
    let root = temp("seed");
    let (c1, s1) = sim_with_seed(101);
    let (c2, s2) = sim_with_seed(102);
    let ids = split_indices(Split::Train, 1);

    let (_, _, m1) = write_split(&root.join("a"), &c1, Split::Train, &run_episodes(&s1, &ids));
    let (_, _, m2) = write_split(&root.join("b"), &c2, Split::Train, &run_episodes(&s2, &ids));

    assert_eq!(m1.seed, c1.run.seed);
    assert_eq!(m2.seed, c2.run.seed);
    assert_ne!(m1.seed, m2.seed);
    assert_ne!(m1.config_hash, m2.config_hash);

    let a = std::fs::read(root.join("a").join(FILE_NAME)).unwrap();
    let b = std::fs::read(root.join("b").join(FILE_NAME)).unwrap();
    assert_ne!(
        a, b,
        "a different seed must change the data bytes, not just the hash"
    );

    // Rewriting the same inputs is byte-identical, including the manifest.
    for tag in ["regen1", "regen2"] {
        write_split(&root.join(tag), &c1, Split::Train, &run_episodes(&s1, &ids));
    }
    assert_eq!(
        std::fs::read(root.join("regen1/manifest.json")).unwrap(),
        std::fs::read(root.join("regen2/manifest.json")).unwrap()
    );
}

#[test]
fn split_disjointness_is_a_loud_gate() {
    let dir = temp("split");
    let (cfg, sim) = sim_with_seed(7);
    let train_ids = split_indices(Split::Train, 2);
    let held_ids = split_indices(Split::HeldOut, 2);
    let train = write_validation_of(
        &dir.join("train.bin"),
        &cfg,
        &run_episodes(&sim, &train_ids),
    );
    let held = write_validation_of(&dir.join("held.bin"), &cfg, &run_episodes(&sim, &held_ids));

    let ok = check_split_disjoint(&train, &held).unwrap();
    assert!(ok.ok);
    assert!(ok.detail.contains("overlap=[]"), "{}", ok.detail);

    let err = check_split_disjoint(&train, &train)
        .unwrap_err()
        .to_string();
    assert!(err.contains("split_scenarios_disjoint"), "{err}");

    let mixed = write_validation_of(
        &dir.join("mixed.bin"),
        &cfg,
        &run_episodes(&sim, &[held_ids[0], train_ids[0]]),
    );
    let err = check_split_disjoint(&train, &mixed)
        .unwrap_err()
        .to_string();
    assert!(err.contains("split_scenarios_disjoint"), "{err}");
    assert!(
        err.contains(&scenario(train_ids[0]).id.to_string()),
        "{err}"
    );

    let empty = write_validation_of(&dir.join("empty.bin"), &cfg, &[]);
    let err = check_split_disjoint(&train, &empty)
        .unwrap_err()
        .to_string();
    assert!(err.contains("split_scenarios_disjoint"), "{err}");
}

#[test]
fn sha256_detects_single_byte_corruption_in_every_manifest_file() {
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );

    let dir = temp("sha");
    let (cfg, sim) = sim_with_seed(55);
    let (_v, _t, m) = write_split(&dir, &cfg, Split::Train, &run_episodes(&sim, &[0]));
    assert_eq!(
        m.files.len(),
        2,
        "validation.bin and its v3 sidecar are both hashed"
    );

    for rec in &m.files {
        let path = dir.join(&rec.name);
        assert_eq!(rec.sha256, sha256_file(&path).unwrap());
        assert_eq!(rec.sha256.len(), 64);
        assert!(rec.name.ends_with(".bin") || rec.name.ends_with(".trajectory"));
        assert!(
            !rec.name.contains('/') && !rec.name.contains('\\'),
            "relative flat name"
        );

        let mut bytes = std::fs::read(&path).unwrap();
        let len = bytes.len() as u64;
        assert_eq!(rec.bytes, len);
        *bytes.last_mut().unwrap() ^= 0x01;
        std::fs::write(&path, &bytes).unwrap();

        let changed = file_record(&rec.name, &path).unwrap();
        assert_ne!(
            changed.sha256, rec.sha256,
            "one flipped byte must move sha256: {}",
            rec.name
        );
        assert_eq!(changed.sha256, sha256_hex(&bytes));
        assert_eq!(changed.bytes, len, "length alone cannot detect corruption");

        // Restore so the on-disk split again matches the manifest.
        *bytes.last_mut().unwrap() ^= 0x01;
        std::fs::write(&path, &bytes).unwrap();
        assert_eq!(sha256_file(&path).unwrap(), rec.sha256);
    }
}

#[test]
fn bounds_cover_every_exported_channel_and_stay_finite() {
    let dir = temp("bounds");
    let (cfg, sim) = sim_with_seed(2026);
    let (v, t, m) = write_split(&dir, &cfg, Split::Train, &run_episodes(&sim, &[3]));
    let _ = (v, t);

    let names: Vec<&str> = m.bounds.features.iter().map(|r| r.name.as_str()).collect();
    assert_eq!(
        names,
        cfg.output
            .features
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
    );
    let names: Vec<&str> = m.bounds.targets.iter().map(|r| r.name.as_str()).collect();
    assert_eq!(names, TARGET_NAMES.to_vec());
    assert_eq!(
        names,
        default_target_names()
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
    );
    let names: Vec<&str> = m.bounds.truth.iter().map(|r| r.name.as_str()).collect();
    assert_eq!(names, TRUTH_V3_TICK_NAMES.to_vec());
    assert_eq!(
        names,
        default_truth_names()
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
    );
    assert_eq!(m.truth_names, default_truth_names());
    assert_eq!(m.target_names, default_target_names());
    assert_eq!(m.feature_names, cfg.output.features);

    let all = m
        .bounds
        .features
        .iter()
        .chain(m.bounds.targets.iter())
        .chain(m.bounds.truth.iter());
    let mut non_degenerate = 0;
    for r in all {
        assert!(
            r.min.is_finite() && r.max.is_finite(),
            "{}: {:?}",
            r.name,
            r
        );
        assert!(r.min <= r.max, "{}: {:?}", r.name, r);
        if r.max > r.min {
            non_degenerate += 1;
        }
    }
    assert!(
        non_degenerate > 0,
        "observed bounds must describe real data"
    );

    // Bounds survive the JSON round trip unchanged.
    assert_eq!(
        manifest_from_str(&manifest_to_string(&m).unwrap())
            .unwrap()
            .bounds,
        m.bounds
    );
}

#[test]
fn manifest_bytes_carry_no_timestamps_or_process_artifacts() {
    let dir = temp("no-time");
    let (cfg, sim) = sim_with_seed(31337);
    let eps = run_episodes(&sim, &split_indices(Split::Train, 1));
    let (a_v, a_t, a_m) = write_split(&dir.join("a"), &cfg, Split::Train, &eps);
    let (b_v, b_t, b_m) = write_split(&dir.join("b"), &cfg, Split::Train, &eps);

    let text = std::fs::read_to_string(dir.join("a/manifest.json")).unwrap();
    let lower = text.to_ascii_lowercase();
    for token in [
        "timestamp",
        "created_at",
        "modified_at",
        "generated_at",
        "wall_clock",
        "process_id",
        "pid",
        "utc",
        "iso8601",
    ] {
        assert!(
            !lower.contains(token),
            "manifest leaked nondeterministic field {token}"
        );
    }
    assert_eq!(a_m, b_m);
    assert_eq!(
        text,
        std::fs::read_to_string(dir.join("b/manifest.json")).unwrap()
    );
    assert_eq!(manifest_to_string(&a_m).unwrap(), text);

    let a_bytes = std::fs::read(dir.join("a/manifest.json")).unwrap();
    let b_bytes = std::fs::read(dir.join("b/manifest.json")).unwrap();
    let ok = check_determinism(&[
        ("train/manifest.json", &a_bytes, &b_bytes),
        (
            "train/validation.bin",
            &std::fs::read(dir.join("a/validation.bin")).unwrap(),
            &std::fs::read(dir.join("b/validation.bin")).unwrap(),
        ),
        (
            "train/validation.trajectory",
            &std::fs::read(dir.join("a/validation.trajectory")).unwrap(),
            &std::fs::read(dir.join("b/validation.trajectory")).unwrap(),
        ),
    ])
    .unwrap();
    assert!(ok.ok);
    assert!(
        ok.detail.contains("3 files compared, mismatches=[]"),
        "{}",
        ok.detail
    );

    let _ = (a_v, a_t, b_v, b_t);
}

#[test]
fn corruption_fails_loudly_at_every_boundary() {
    let dir = temp("loud");
    let (cfg, sim) = sim_with_seed(99);
    let (v, t, _m) = write_split(&dir, &cfg, Split::Train, &run_episodes(&sim, &[0]));

    // Malformed manifest JSON is an error that names the operation.
    let err = manifest_from_str("{ not json").unwrap_err().to_string();
    assert!(err.contains("manifest parse"), "{err}");
    let bad = dir.join("bad.json");
    std::fs::write(&bad, "{ not json").unwrap();
    let err = read_manifest(&bad).unwrap_err().to_string();
    assert!(err.contains("manifest parse"), "{err}");

    // The dataset gates reject corrupt data instead of clamping or skipping.
    check_dataset("train", &cfg, &v, &t, 1).unwrap();

    let mut nan = v.clone();
    nan.episodes[0].features[[0, 0]] = f32::NAN;
    let err = check_dataset("train", &cfg, &nan, &t, 1)
        .unwrap_err()
        .to_string();
    assert!(err.contains("finite") && err.contains("features"), "{err}");

    let mut target = v.clone();
    target.episodes[0].targets[[0, 0]] = 0.9;
    let err = check_dataset("train", &cfg, &target, &t, 1)
        .unwrap_err()
        .to_string();
    assert!(err.contains("target_bounds"), "{err}");

    let mut short = t.clone();
    short.episodes[0].ticks.pop();
    let err = check_dataset("train", &cfg, &v, &short, 1)
        .unwrap_err()
        .to_string();
    assert!(err.contains("frame_count"), "{err}");

    let err = check_dataset("train", &cfg, &v, &t, 3)
        .unwrap_err()
        .to_string();
    assert!(err.contains("episode_count"), "{err}");

    let err = check_determinism(&[("train/validation.bin", b"a", b"b")])
        .unwrap_err()
        .to_string();
    assert!(err.contains("deterministic_regeneration"), "{err}");
    assert!(err.contains("train/validation.bin"), "{err}");
}
