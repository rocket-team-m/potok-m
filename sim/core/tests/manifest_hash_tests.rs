// SPDX-License-Identifier: GPL-3.0-or-later
//! SIMHASH — provenance binding for generated datasets.
//!
//! The companion `manifest.provenance.json` binds a dataset to the exact code
//! commit, protocol schema, measurement inputs and golden episode that produced
//! it. Every field is derived from bytes already on disk or from an explicit
//! input: no timestamps, no process ids, no map iteration order. These tests
//! pin determinism, v1/v2/v3 compatibility and LOUD failure on any tampering.

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use potok_sim::{
    config_hash,
    manifest::{
        build_provenance, default_target_names, default_truth_names, file_record, git_head_hash,
        golden_episode_hash, measurement_hashes, protocol_schema_hash, provenance_from_str,
        provenance_path, provenance_to_string, read_provenance, verify_code_hash,
        verify_golden_episode, verify_measurement_inputs, verify_protocol_schema,
        verify_provenance_binding, write_manifest, write_provenance, DatasetManifest, FileRecord,
        ProvenanceManifest, ProvenanceSources, PROVENANCE_SCHEMA_VERSION, SCHEMA_VERSION,
        SUPPORTED_SCHEMA_VERSIONS,
    },
    quality::{check_golden_episode, check_provenance, observed_bounds},
    read_trajectory, sidecar_path,
    trajectory::write_trajectory_v3,
    validation::{write_validation, FILE_NAME},
    Config, Simulator, Split,
};

const REPO: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
const PROTOCOL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../protocol/uart.yaml");

/// Pinned seed-1337 episode-0 validation.bin + validation.trajectory hash.
/// Changing the RNG draw order or any exported byte breaks this literal.
const GOLDEN_PIN: &str = "2e1395c0532d6d934fcac89f07885254ffe4838e2ae22d4f4afe73187b480d02";

/// The golden fixture freezes the exported feature list so an unrelated edit
/// to `config/default.toml` cannot invalidate the pin; only a change to the
/// RNG draw order or to the exported bytes should.
fn frozen_golden_config() -> Config {
    let mut cfg = Config::default();
    cfg.run.seed = 1337;
    cfg.output.features = vec![
        "gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z", "baro_h", "pitot_q",
    ]
    .into_iter()
    .map(str::to_string)
    .collect();
    cfg
}

fn temp(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("potok-simhash-{tag}-{}", std::process::id()));
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

/// Writes one split the way the generator does and returns its manifest plus
/// the file records of the dataset bytes (used by the golden pin).
fn write_split(dir: &Path, cfg: &Config, sim: &Simulator) -> (DatasetManifest, Vec<FileRecord>) {
    std::fs::create_dir_all(dir).unwrap();
    let eps = vec![sim.run_episode(0).unwrap()];
    let validation = dir.join(FILE_NAME);
    let sidecar = sidecar_path(&validation);
    write_validation(&validation, cfg, &eps).unwrap();
    write_trajectory_v3(&sidecar, cfg, &eps).unwrap();
    let v = potok_sim::validation::read_validation(&validation).unwrap();
    let t = read_trajectory(&sidecar).unwrap();

    let manifest = DatasetManifest {
        schema_version: SCHEMA_VERSION,
        config_hash: config_hash(cfg).unwrap(),
        seed: cfg.run.seed,
        split: Split::Train.label().to_string(),
        scenario_ids: vec![0],
        episode_count: 1,
        frame_count: cfg.frames_per_episode() as u64,
        frames_per_episode: cfg.frames_per_episode() as u32,
        feature_names: cfg.output.features.clone(),
        target_names: default_target_names(),
        truth_names: default_truth_names(),
        bounds: observed_bounds(cfg, &v, &t),
        files: vec![
            file_record(FILE_NAME, &validation).unwrap(),
            file_record("validation.trajectory", &sidecar).unwrap(),
        ],
    };
    let files = manifest_files(&manifest, dir);
    write_manifest(&dir.join("manifest.json"), &manifest).unwrap();
    (manifest, files)
}

fn manifest_files(manifest: &DatasetManifest, dir: &Path) -> Vec<FileRecord> {
    manifest
        .files
        .iter()
        .map(|r| file_record(&r.name, &dir.join(&r.name)).unwrap())
        .collect()
}

fn measurements(dir: &Path) -> BTreeMap<String, String> {
    write_measurement_inputs(dir);
    measurement_hashes(dir).unwrap()
}

fn write_measurement_inputs(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(
        dir.join("THRUST_curve.csv"),
        b"t_s,thrust_n\n0,100\n1,200\n",
    )
    .unwrap();
    std::fs::write(dir.join("imu_allan.json"), b"{\"tau\":1,\"sigma\":0.01}").unwrap();
    std::fs::write(dir.join("servo_step.csv"), b"t_s,deg\n0,0\n0.1,5\n").unwrap();
    // A source file is not a measurement even though its name contains a token.
    std::fs::write(dir.join("servo.rs"), b"fn main() {}").unwrap();
}

fn good_provenance(
    manifest: &DatasetManifest,
    golden: &[FileRecord],
    measurements: BTreeMap<String, String>,
) -> ProvenanceManifest {
    build_provenance(
        manifest,
        &git_head_hash(Path::new(REPO)).unwrap(),
        &protocol_schema_hash(Path::new(PROTOCOL)).unwrap(),
        measurements,
        &golden_episode_hash(golden).unwrap(),
    )
    .unwrap()
}

fn sources<'a>(
    repo: &'a Path,
    protocol: &'a Path,
    measurements: Option<&'a Path>,
) -> ProvenanceSources<'a> {
    ProvenanceSources {
        repo,
        protocol_schema: protocol,
        measurement_dir: measurements,
    }
}

#[test]
fn provenance_json_is_deterministic_and_carries_no_timestamps() {
    let dir = temp("determinism");
    let (cfg, sim) = sim_with_seed(31337);
    let (manifest, golden) = write_split(&dir.join("a"), &cfg, &sim);
    let (_, golden_b) = write_split(&dir.join("b"), &cfg, &sim);
    assert_eq!(golden, golden_b);

    let m = measurements(&dir.join("meas"));
    let mut reversed = BTreeMap::new();
    for (name, hash) in m.iter().rev() {
        reversed.insert(name.clone(), hash.clone());
    }
    let p = good_provenance(&manifest, &golden, m.clone());
    let p2 = good_provenance(&manifest, &golden, reversed);
    assert_eq!(p, p2);
    assert_eq!(p.measurement_hashes.len(), 3, "only data files are bound");

    let text = provenance_to_string(&p).unwrap();
    assert_eq!(text, provenance_to_string(&p2).unwrap());
    assert_eq!(provenance_from_str(&text).unwrap(), p);
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
        assert!(!lower.contains(token), "provenance leaked {token}");
    }

    let pa = dir.join("a/manifest.provenance.json");
    let pb = dir.join("b/manifest.provenance.json");
    assert_eq!(provenance_path(&dir.join("a/manifest.json")), pa);
    write_provenance(&pa, &p).unwrap();
    write_provenance(&pb, &p).unwrap();
    assert_eq!(std::fs::read(&pa).unwrap(), std::fs::read(&pb).unwrap());
    assert_eq!(read_provenance(&pa).unwrap(), p);
}

#[test]
fn every_bound_hash_is_verified_against_its_live_source() {
    let dir = temp("verify");
    let (cfg, sim) = sim_with_seed(4242);
    let (manifest, golden) = write_split(&dir.join("dataset"), &cfg, &sim);
    let mdir = dir.join("meas");
    let p = good_provenance(&manifest, &golden, measurements(&mdir));

    let s = sources(Path::new(REPO), Path::new(PROTOCOL), Some(&mdir));
    let checks = check_provenance("train", &manifest, &p, &s).unwrap();
    assert_eq!(
        checks.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
        [
            "provenance_binding",
            "code_hash",
            "protocol_schema_hash",
            "measurement_hashes"
        ]
    );
    assert!(checks.iter().all(|c| c.ok));
    let gold = check_golden_episode("train", &p, &golden).unwrap();
    assert!(gold.ok && gold.detail.contains(&p.golden_episode_hash));
}

#[test]
fn tampering_with_any_bound_field_fails_loudly() {
    let dir = temp("tamper");
    let (cfg, sim) = sim_with_seed(99);
    let (manifest, golden) = write_split(&dir.join("dataset"), &cfg, &sim);
    let mdir = dir.join("meas");
    let p = good_provenance(&manifest, &golden, measurements(&mdir));

    // Manifest bytes changed after the binding was written.
    let mut changed = manifest.clone();
    changed.seed += 1;
    let err = verify_provenance_binding(&changed, &p)
        .unwrap_err()
        .to_string();
    assert!(err.contains("manifest_hash"), "{err}");
    assert!(err.contains("expected"), "{err}");

    // A different code commit.
    let mut bad = p.clone();
    bad.code_hash = "0".repeat(40);
    let err = verify_code_hash(&bad, Path::new(REPO))
        .unwrap_err()
        .to_string();
    assert!(err.contains("code_hash"), "{err}");

    // A different protocol schema.
    let mut bad = p.clone();
    bad.protocol_schema_hash = "1".repeat(64);
    let err = verify_protocol_schema(&bad, Path::new(PROTOCOL))
        .unwrap_err()
        .to_string();
    assert!(err.contains("protocol_schema_hash"), "{err}");

    // A changed measurement file, a removed measurement file, and an
    // unexpected extra measurement all fail.
    let altered = dir.join("altered-meas");
    write_measurement_inputs(&altered);
    std::fs::write(altered.join("servo_step.csv"), b"t_s,deg\n0,999\n").unwrap();
    let err = verify_measurement_inputs(&p, Some(&altered))
        .unwrap_err()
        .to_string();
    assert!(err.contains("measurement_hashes"), "{err}");

    let removed = dir.join("removed-meas");
    write_measurement_inputs(&removed);
    std::fs::remove_file(removed.join("imu_allan.json")).unwrap();
    let err = verify_measurement_inputs(&p, Some(&removed))
        .unwrap_err()
        .to_string();
    assert!(err.contains("measurement_hashes"), "{err}");
    let err = verify_measurement_inputs(&p, None).unwrap_err().to_string();
    assert!(err.contains("measurement_hashes"), "{err}");

    // One flipped byte in a golden file.
    let mut tampered = golden.clone();
    let path = dir.join("dataset").join(&tampered[0].name);
    let mut bytes = std::fs::read(&path).unwrap();
    *bytes.last_mut().unwrap() ^= 0x01;
    std::fs::write(&path, &bytes).unwrap();
    tampered[0] = file_record(&tampered[0].name, &path).unwrap();
    let err = verify_golden_episode(&p, &tampered)
        .unwrap_err()
        .to_string();
    assert!(err.contains("golden_episode_hash"), "{err}");

    // Shape errors are LOUD too, not silently accepted.
    let mut bad = p.clone();
    bad.golden_episode_hash = "deadbeef".into();
    let err = verify_provenance_binding(&manifest, &bad)
        .unwrap_err()
        .to_string();
    assert!(err.contains("sha256"), "{err}");
    let err = build_provenance(
        &manifest,
        "not-a-sha",
        &"a".repeat(64),
        BTreeMap::new(),
        &"b".repeat(64),
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("code_hash"), "{err}");
    assert!(provenance_from_str("{ not json").is_err());
}

#[test]
fn golden_episode_pin_detects_a_seed_change() {
    let dir = temp("pin");
    let cfg = frozen_golden_config();
    let sim = Simulator::new(cfg.clone()).unwrap();
    let (_, golden) = write_split(&dir.join("1337"), &cfg, &sim);
    let hash = golden_episode_hash(&golden).unwrap();
    assert_eq!(hash.len(), 64);
    assert_eq!(
        hash, GOLDEN_PIN,
        "seed-1337 episode-0 bytes changed; record the new pin deliberately"
    );

    let (other_cfg, other_sim) = sim_with_seed(1338);
    let (_, other) = write_split(&dir.join("1338"), &other_cfg, &other_sim);
    assert_ne!(golden_episode_hash(&other).unwrap(), hash);
}

#[test]
fn v1_v2_v3_schemas_bind_and_v4_is_rejected_loudly() {
    assert_eq!(SUPPORTED_SCHEMA_VERSIONS, [1, 2, 3]);
    assert_eq!(SCHEMA_VERSION, 1);
    assert_eq!(PROVENANCE_SCHEMA_VERSION, 1);

    let dir = temp("versions");
    let (cfg, sim) = sim_with_seed(7);
    let (manifest, golden) = write_split(&dir.join("dataset"), &cfg, &sim);
    let code = git_head_hash(Path::new(REPO)).unwrap();
    let proto = protocol_schema_hash(Path::new(PROTOCOL)).unwrap();

    for version in SUPPORTED_SCHEMA_VERSIONS {
        let mut m = manifest.clone();
        m.schema_version = version;
        let p = build_provenance(
            &m,
            &code,
            &proto,
            BTreeMap::new(),
            &golden_episode_hash(&golden).unwrap(),
        )
        .unwrap();
        verify_provenance_binding(&m, &p)
            .unwrap_or_else(|e| panic!("v{version} must remain bindable: {e}"));
    }

    let mut future = manifest.clone();
    future.schema_version = 4;
    let p = build_provenance(
        &future,
        &code,
        &proto,
        BTreeMap::new(),
        &golden_episode_hash(&golden).unwrap(),
    )
    .unwrap();
    let err = verify_provenance_binding(&future, &p)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("unsupported manifest schema_version 4"),
        "{err}"
    );

    let mut bad = p.clone();
    bad.schema_version = 99;
    bad.manifest_hash = p.manifest_hash.clone();
    let err = verify_provenance_binding(&future, &bad)
        .unwrap_err()
        .to_string();
    assert!(err.contains("unsupported"), "{err}");
}

#[test]
fn absent_measurement_inputs_are_recorded_as_empty_not_invented() {
    let dir = temp("absent");
    let (cfg, sim) = sim_with_seed(5);
    let (manifest, golden) = write_split(&dir.join("dataset"), &cfg, &sim);
    let empty = dir.join("no-measurements");
    assert!(measurement_hashes(&empty).unwrap().is_empty());
    assert!(measurement_hashes(&dir.join("also-absent"))
        .unwrap()
        .is_empty());

    let p = good_provenance(&manifest, &golden, BTreeMap::new());
    verify_measurement_inputs(&p, None).unwrap();
    verify_measurement_inputs(&p, Some(&empty)).unwrap();

    std::fs::create_dir_all(&empty).unwrap();
    std::fs::write(empty.join("thrust_stand.csv"), b"t,n\n0,1\n").unwrap();
    let err = verify_measurement_inputs(&p, Some(&empty))
        .unwrap_err()
        .to_string();
    assert!(err.contains("thrust_stand.csv"), "{err}");
}
