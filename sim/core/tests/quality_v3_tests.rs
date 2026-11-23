//! Quality gate v3: g_accel bounds, split disjointness, manifest hash and
//! deterministic regeneration. Every failure below must be a LOUD gate error,
//! never a clamp, skip or silent fallback.

use std::path::{Path, PathBuf};

use potok_sim::{
    config_hash,
    manifest::{
        canonical_config_bytes, default_target_names, default_truth_names, file_record,
        manifest_to_string, read_manifest, sha256_file, sha256_hex, write_manifest,
        DatasetManifest, SCHEMA_VERSION,
    },
    quality::{
        check_dataset, check_determinism, check_split_disjoint, observed_bounds, target_limits,
    },
    read_trajectory, scenario, sidecar_path,
    trajectory::write_trajectory_v3,
    validation::{
        read_validation, write_validation, write_validation_with_version, ValidationSet, FILE_NAME,
        HEADER_SIZE, LEGACY_TARGET_COUNT, TARGET_COUNT, TARGET_NAMES, VERSION_V1,
    },
    Config, Episode, Simulator, TrajectorySet, MAX_COMMAND_MPS2, PHYSICS_FEATURES,
};

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("potok-qv3-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn new_sim(seed: u64) -> (Config, Simulator) {
    let mut cfg = Config::default();
    cfg.run.seed = seed;
    let sim = Simulator::new(cfg.clone()).unwrap();
    (cfg, sim)
}

fn episodes(sim: &Simulator, indices: &[u64]) -> Vec<Episode> {
    indices
        .iter()
        .map(|&i| sim.run_episode(i).unwrap())
        .collect()
}

fn indices(split: potok_sim::Split, count: usize) -> Vec<u64> {
    (0..)
        .filter(|&i| scenario(i).split == split)
        .take(count)
        .collect()
}

fn write_validation_of(path: &Path, cfg: &Config, eps: &[Episode]) -> ValidationSet {
    write_validation(path, cfg, eps).unwrap();
    read_validation(path).unwrap()
}

/// Writes one split exactly as `generate-dataset` does: validation, v3 truth
/// sidecar, manifest. Returns the read-back sets so gates run on the real bytes.
fn write_split(
    dir: &Path,
    cfg: &Config,
    split: potok_sim::Split,
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
fn v3_g_accel_limits_are_the_command_envelope() {
    assert_eq!(target_limits(4), (-MAX_COMMAND_MPS2, MAX_COMMAND_MPS2));
    assert_eq!(target_limits(5), (-MAX_COMMAND_MPS2, MAX_COMMAND_MPS2));
    assert_eq!(
        (TARGET_NAMES[4], TARGET_NAMES[5]),
        ("g_accel_y", "g_accel_x")
    );
    // The two legacy correction columns are unchanged by the v3 extension.
    assert!(target_limits(0).1 < 1.0);
    assert_eq!(target_limits(2), (0.0, 1.0));
    assert_eq!(target_limits(3), (0.0, 1.0));
}

#[test]
fn g_accel_targets_pass_the_gate_and_match_the_sidecar() {
    let dir = temp_dir("gaccel-ok");
    let (cfg, sim) = new_sim(2026);
    let eps = episodes(&sim, &[3]);
    let validation = dir.join(FILE_NAME);
    let sidecar = sidecar_path(&validation);
    let v = write_validation_of(&validation, &cfg, &eps);
    write_trajectory_v3(&sidecar, &cfg, &eps).unwrap();
    let t = read_trajectory(&sidecar).unwrap();

    let report = check_dataset("train", &cfg, &v, &t, 1).unwrap();
    let targets = report
        .checks
        .iter()
        .find(|c| c.name == "target_bounds")
        .unwrap();
    assert!(targets.ok);
    assert!(targets.detail.contains("g_accel"), "{}", targets.detail);

    for (row, tick) in v.episodes[0]
        .targets
        .rows()
        .into_iter()
        .zip(&t.episodes[0].ticks)
    {
        let (y, x) = (row[4] as f64, row[5] as f64);
        assert!(y.is_finite() && x.is_finite());
        assert!(y.abs() <= MAX_COMMAND_MPS2 && x.abs() <= MAX_COMMAND_MPS2);
        assert_eq!(row[4], tick.g_accel[0]);
        assert_eq!(row[5], tick.g_accel[1]);
        assert!(tick
            .g_accel
            .iter()
            .all(|g| (*g as f64).abs() <= MAX_COMMAND_MPS2));
    }

    // The envelope edge itself is legal: the gate uses an inclusive range.
    let mut edge = v.clone();
    let last = edge.episodes[0].targets.nrows() - 1;
    edge.episodes[0].targets[[last, 4]] = MAX_COMMAND_MPS2 as f32;
    edge.episodes[0].targets[[last, 5]] = -(MAX_COMMAND_MPS2 as f32);
    assert!(check_dataset("train", &cfg, &edge, &t, 1).is_ok());
}

#[test]
fn g_accel_target_violations_fail_loudly() {
    let dir = temp_dir("gaccel-bad");
    let (cfg, sim) = new_sim(77);
    let eps = episodes(&sim, &[5]);
    let validation = dir.join(FILE_NAME);
    let sidecar = sidecar_path(&validation);
    let v = write_validation_of(&validation, &cfg, &eps);
    write_trajectory_v3(&sidecar, &cfg, &eps).unwrap();
    let t = read_trajectory(&sidecar).unwrap();

    for (col, name) in [(4, "g_accel_y"), (5, "g_accel_x")] {
        for value in [MAX_COMMAND_MPS2 + 1.0, -MAX_COMMAND_MPS2 - 1.0] {
            let mut bad = v.clone();
            bad.episodes[0].targets[[0, col]] = value as f32;
            let err = check_dataset("train", &cfg, &bad, &t, 1)
                .unwrap_err()
                .to_string();
            assert!(err.contains("train"), "{err}");
            assert!(err.contains("target_bounds"), "{err}");
            assert!(err.contains(name), "{err}");
        }
    }
    let mut nan = v.clone();
    nan.episodes[0].targets[[0, 4]] = f32::NAN;
    let err = check_dataset("train", &cfg, &nan, &t, 1)
        .unwrap_err()
        .to_string();
    assert!(err.contains("finite") && err.contains("targets"), "{err}");
}

#[test]
fn g_accel_truth_violations_fail_loudly() {
    let dir = temp_dir("gaccel-truth");
    let (cfg, sim) = new_sim(78);
    let eps = episodes(&sim, &[5]);
    let validation = dir.join(FILE_NAME);
    let sidecar = sidecar_path(&validation);
    let v = write_validation_of(&validation, &cfg, &eps);
    write_trajectory_v3(&sidecar, &cfg, &eps).unwrap();
    let t = read_trajectory(&sidecar).unwrap();

    for axis in 0..2 {
        let mut bad = t.clone();
        bad.episodes[0].ticks[0].g_accel[axis] = (MAX_COMMAND_MPS2 + 1.0) as f32;
        let err = check_dataset("train", &cfg, &v, &bad, 1)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("truth_bounds") && err.contains("g_accel"),
            "{err}"
        );
    }
}

#[test]
fn split_gate_accepts_disjoint_scenarios_and_rejects_contamination() {
    let dir = temp_dir("split");
    let (cfg, sim) = new_sim(7);
    let train_idx = indices(potok_sim::Split::Train, 2);
    let held_idx = indices(potok_sim::Split::HeldOut, 2);
    let train = write_validation_of(&dir.join("train.bin"), &cfg, &episodes(&sim, &train_idx));
    let held = write_validation_of(&dir.join("held.bin"), &cfg, &episodes(&sim, &held_idx));

    let ok = check_split_disjoint(&train, &held).unwrap();
    assert!(ok.ok);
    assert!(ok.detail.contains("overlap=[]"), "{}", ok.detail);
    let a: Vec<u16> = train.episodes.iter().map(|e| e.scenario_id).collect();
    let b: Vec<u16> = held.episodes.iter().map(|e| e.scenario_id).collect();
    assert!(a.iter().all(|id| !b.contains(id)), "{a:?} vs {b:?}");

    let err = check_split_disjoint(&train, &train)
        .unwrap_err()
        .to_string();
    assert!(err.contains("split_scenarios_disjoint"), "{err}");

    // One held-out episode carrying a training scenario id is contamination.
    let mixed_idx = [held_idx[0], train_idx[0]];
    let mixed = write_validation_of(&dir.join("mixed.bin"), &cfg, &episodes(&sim, &mixed_idx));
    let err = check_split_disjoint(&train, &mixed)
        .unwrap_err()
        .to_string();
    assert!(err.contains("split_scenarios_disjoint"), "{err}");
    assert!(
        err.contains(&scenario(train_idx[0]).id.to_string()),
        "{err}"
    );

    let empty = write_validation_of(&dir.join("empty.bin"), &cfg, &[]);
    let err = check_split_disjoint(&train, &empty)
        .unwrap_err()
        .to_string();
    assert!(err.contains("split_scenarios_disjoint"), "{err}");
}

#[test]
fn manifest_hash_binds_config_and_file_bytes() {
    let dir = temp_dir("manifest-hash");
    let (cfg, sim) = new_sim(99);
    let (_v, _t, manifest) =
        write_split(&dir, &cfg, potok_sim::Split::Train, &episodes(&sim, &[0]));
    assert_eq!(read_manifest(&dir.join("manifest.json")).unwrap(), manifest);
    assert_eq!(
        manifest.config_hash,
        sha256_hex(&canonical_config_bytes(&cfg).unwrap())
    );
    assert_eq!(manifest.config_hash, config_hash(&cfg.clone()).unwrap());
    for rec in &manifest.files {
        assert_eq!(rec.sha256, sha256_file(&dir.join(&rec.name)).unwrap());
    }

    let text = manifest_to_string(&manifest).unwrap();
    assert_eq!(text, manifest_to_string(&manifest).unwrap());
    assert_eq!(
        sha256_hex(text.as_bytes()),
        sha256_hex(&std::fs::read(dir.join("manifest.json")).unwrap())
    );

    let mut other = cfg.clone();
    other.run.seed += 1;
    assert_ne!(config_hash(&other).unwrap(), manifest.config_hash);

    let path = dir.join(FILE_NAME);
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[HEADER_SIZE] ^= 0x01;
    std::fs::write(&path, &bytes).unwrap();
    let changed = file_record(FILE_NAME, &path).unwrap();
    assert_eq!(changed.sha256, sha256_hex(&bytes));
    assert_eq!(changed.bytes, bytes.len() as u64);
    assert_ne!(changed.sha256, manifest.files[0].sha256);
}

#[test]
fn regeneration_is_byte_identical_and_the_gate_can_fail() {
    let root = temp_dir("regen");
    let (cfg, sim) = new_sim(4242);
    let train_idx = indices(potok_sim::Split::Train, 2);
    let held_idx = indices(potok_sim::Split::HeldOut, 2);
    for tag in ["a", "b"] {
        let dir = root.join(tag);
        let (tv, tt, train_manifest) = write_split(
            &dir.join("train"),
            &cfg,
            potok_sim::Split::Train,
            &episodes(&sim, &train_idx),
        );
        let (hv, ht, held_manifest) = write_split(
            &dir.join("held-out"),
            &cfg,
            potok_sim::Split::HeldOut,
            &episodes(&sim, &held_idx),
        );
        check_dataset("train", &cfg, &tv, &tt, 1).unwrap();
        check_dataset("held-out", &cfg, &hv, &ht, 1).unwrap();
        check_split_disjoint(&tv, &hv).unwrap();
        assert_eq!(train_manifest.config_hash, held_manifest.config_hash);
    }

    let mut pairs = Vec::new();
    for label in ["train", "held-out"] {
        for name in [FILE_NAME, "validation.trajectory", "manifest.json"] {
            let rel = format!("{label}/{name}");
            pairs.push((
                rel,
                std::fs::read(root.join("a").join(label).join(name)).unwrap(),
                std::fs::read(root.join("b").join(label).join(name)).unwrap(),
            ));
        }
    }
    let refs: Vec<(&str, &[u8], &[u8])> = pairs
        .iter()
        .map(|(n, a, b)| (n.as_str(), a.as_slice(), b.as_slice()))
        .collect();
    let ok = check_determinism(&refs).unwrap();
    assert!(ok.ok);
    assert!(
        ok.detail.contains("6 files compared, mismatches=[]"),
        "{}",
        ok.detail
    );

    let err = check_determinism(&[("train/validation.bin", b"one", b"two")])
        .unwrap_err()
        .to_string();
    assert!(err.contains("deterministic_regeneration"), "{err}");
    assert!(err.contains("train/validation.bin"), "{err}");

    let (other_cfg, other_sim) = new_sim(4243);
    let other = root.join("other-seed");
    let (_, _, manifest) = write_split(
        &other,
        &other_cfg,
        potok_sim::Split::Train,
        &episodes(&other_sim, &train_idx),
    );
    assert_ne!(manifest.config_hash, config_hash(&cfg).unwrap());
    assert_ne!(
        std::fs::read(root.join("a/train").join(FILE_NAME)).unwrap(),
        std::fs::read(other.join(FILE_NAME)).unwrap(),
        "a different seed must move the bytes, not just the hash"
    );
}

/// The post-separation mass floor must reach the v3 export without moving a
/// byte of the v1/v2/v3 layouts (SIMDBG finding 2). `drag_est` is
/// `cd0 * q_dyn * area / mass`, so the mass implied by the exported pair must
/// equal the trajectory tick's mass on every aerodynamic frame — including
/// after separation, where the old dry_mass clamp forged ~8.1 kg over ~6.3 kg.
#[test]
fn post_sep_mass_reaches_v3_features_without_layout_changes() {
    let dir = temp_dir("post-sep-v3");
    let mut cfg = Config::default();
    cfg.output
        .features
        .extend(PHYSICS_FEATURES.iter().map(|s| s.to_string()));
    let sim = Simulator::new(cfg.clone()).unwrap();
    let eps = episodes(&sim, &[0]);
    let frames = cfg.frames_per_episode() as usize;

    // The fix changes feature VALUES, not the container stride.
    for (version, meta, targets) in [
        (VERSION_V1, 9usize, LEGACY_TARGET_COUNT),
        (2, 17, LEGACY_TARGET_COUNT),
        (3, 17, TARGET_COUNT),
    ] {
        let path = dir.join(format!("v{version}.bin"));
        write_validation_with_version(&path, &cfg, &eps, version).unwrap();
        assert_eq!(
            std::fs::read(&path).unwrap().len(),
            HEADER_SIZE + 8 + meta * 4 + frames * (14 + targets) * 4,
            "v{version} layout changed"
        );
    }

    let validation = dir.join(FILE_NAME);
    let sidecar = sidecar_path(&validation);
    let set = write_validation_of(&validation, &cfg, &eps);
    write_trajectory_v3(&sidecar, &cfg, &eps).unwrap();
    let traj = read_trajectory(&sidecar).unwrap();

    // The episode overrides `vehicle.aero.cd0` with its sampled domain value,
    // which is exactly what the conditioning record carries.
    let cd0 = eps[0].meta.conditioning.cd0 as f64;
    let area = cfg.aero.area_m2;
    let (q_col, drag_col) = (8 + 2, 8 + 4); // q_dyn_kpa, drag_est inside the 8 raw channels
    let mut first = 0.0_f64;
    let mut last = None;
    for (i, tick) in traj.episodes[0].ticks.iter().enumerate() {
        let f = set.episodes[0].features.row(i);
        let mass = tick.mass_kg as f64;
        if i == 0 {
            first = mass;
        }
        let qdyn = f[q_col] as f64 * 1000.0;
        let drag = f[drag_col] as f64;
        if qdyn < 1.0 || drag <= 0.0 {
            continue;
        }
        let implied = cd0 * qdyn * area / drag;
        assert!(
            (implied - mass).abs() / mass < 1e-3,
            "frame {i}: mass implied by drag_est {implied:.4} != tick mass {mass:.4}"
        );
        if eps[0].frames[i].time_us as f64 >= 7.0e6 {
            last = Some(mass);
        }
    }
    let last = last.expect("no post-separation frame");
    assert!(
        last < first - 3.0,
        "stage separation did not lower the mass: {first:.3} -> {last:.3}"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}
