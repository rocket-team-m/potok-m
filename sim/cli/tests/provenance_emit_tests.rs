// SPDX-License-Identifier: GPL-3.0-or-later
//! SIMIDEAL follow-up: `generate-dataset` must emit the provenance companion
//! that `core/src/manifest.rs` already knows how to build and verify. The v1
//! manifest and the v1/v2/v3 dataset byte layouts stay untouched.

use std::{path::Path, path::PathBuf, process::Command};

use potok_sim::{
    manifest::{
        git_head_hash, golden_episode_hash, manifest_to_string, provenance_path, read_manifest,
        read_provenance, sha256_file, sha256_hex, verify_code_hash, verify_golden_episode,
        verify_measurement_inputs, verify_protocol_schema, verify_provenance_binding,
        write_golden_episode,
    },
    read_trajectory,
    validation::read_validation,
};

const REPO: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
const PROTOCOL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../protocol/uart.yaml");

/// Pinned seed-1337 episode-0 validation + trajectory hash, independently
/// asserted in `core/tests/manifest_hash_tests.rs`.
const GOLDEN_PIN: &str = "2e1395c0532d6d934fcac89f07885254ffe4838e2ae22d4f4afe73187b480d02";

fn dataset_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("potok-provenance-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn generate(out: &Path, extra: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_potok-cli"))
        .arg("generate-dataset")
        .args([
            "--seed",
            "42",
            "--train-episodes",
            "8",
            "--held-out-episodes",
            "4",
        ])
        .arg("--out")
        .arg(out)
        .args(extra)
        .output()
        .expect("run generate-dataset")
}

#[test]
fn generate_dataset_emits_a_verified_provenance_companion_per_split() {
    let dir = dataset_dir("emit");
    let out = generate(&dir, &[]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("provenance:"), "{stdout}");
    assert!(
        !dir.join(".provenance-golden").exists(),
        "golden temp leaked"
    );

    let code = git_head_hash(Path::new(REPO)).unwrap();
    let protocol = sha256_file(Path::new(PROTOCOL)).unwrap();
    for label in ["train", "held-out"] {
        let split = dir.join(label);
        let manifest_path = split.join("manifest.json");
        let prov_path = provenance_path(&manifest_path);
        assert_eq!(prov_path, split.join("manifest.provenance.json"));

        let manifest = read_manifest(&manifest_path).unwrap();
        let provenance = read_provenance(&prov_path).unwrap();
        assert_eq!(provenance.schema_version, 1);
        assert_eq!(
            provenance.manifest_hash,
            sha256_hex(manifest_to_string(&manifest).unwrap().as_bytes())
        );
        assert_eq!(provenance.code_hash, code);
        assert_eq!(provenance.protocol_schema_hash, protocol);
        assert!(provenance.measurement_hashes.is_empty());
        assert_eq!(provenance.golden_episode_hash, GOLDEN_PIN);

        verify_provenance_binding(&manifest, &provenance).unwrap();
        verify_code_hash(&provenance, Path::new(REPO)).unwrap();
        verify_protocol_schema(&provenance, Path::new(PROTOCOL)).unwrap();
        verify_measurement_inputs(&provenance, None).unwrap();

        // The golden hash is the seed-1337 episode-0 bytes, not a hash of the
        // seed-42 dataset files recorded in this split's manifest.
        let golden_dir = dir.join(format!(".golden-check-{label}"));
        let golden = write_golden_episode(&golden_dir).unwrap();
        assert_eq!(golden_episode_hash(&golden).unwrap(), GOLDEN_PIN);
        verify_golden_episode(&provenance, &golden).unwrap();
        std::fs::remove_dir_all(&golden_dir).unwrap();
    }

    let report = std::fs::read_to_string(dir.join("quality_report.json")).unwrap();
    for gate in [
        "provenance_binding",
        "code_hash",
        "protocol_schema_hash",
        "measurement_hashes",
        "golden_episode",
    ] {
        assert!(report.contains(gate), "{gate} gate missing: {report}");
    }
    assert!(report.contains("\"all_passed\": true"), "{report}");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn provenance_companions_are_byte_identical_on_regeneration() {
    let a = dataset_dir("det-a");
    let b = dataset_dir("det-b");
    assert!(generate(&a, &[]).status.success());
    assert!(generate(&b, &[]).status.success());
    for name in [
        "train/manifest.provenance.json",
        "held-out/manifest.provenance.json",
        "train/manifest.json",
        "held-out/manifest.json",
        "quality_report.json",
    ] {
        assert_eq!(
            std::fs::read(a.join(name)).unwrap(),
            std::fs::read(b.join(name)).unwrap(),
            "{name} must be byte-identical across runs"
        );
    }
    std::fs::remove_dir_all(&a).unwrap();
    std::fs::remove_dir_all(&b).unwrap();
}

#[test]
fn provenance_companion_leaves_v1_manifest_and_v1_v2_v3_bytes_untouched() {
    let dir = dataset_dir("wire");
    assert!(generate(&dir, &[]).status.success());
    for label in ["train", "held-out"] {
        let split = dir.join(label);
        let manifest = read_manifest(&split.join("manifest.json")).unwrap();
        assert_eq!(manifest.schema_version, 1);
        assert_eq!(manifest.files.len(), 2, "companion is not a dataset file");
        assert_eq!(manifest.files[0].name, "validation.bin");
        assert_eq!(manifest.files[1].name, "validation.trajectory");
        for rec in &manifest.files {
            let path = split.join(&rec.name);
            assert_eq!(rec.sha256, sha256_file(&path).unwrap());
            assert_eq!(rec.bytes, std::fs::metadata(&path).unwrap().len());
        }

        let validation = read_validation(&split.join("validation.bin")).unwrap();
        let trajectory = read_trajectory(&split.join("validation.trajectory")).unwrap();
        assert_eq!(validation.header.target_count, 6, "v3 validation targets");
        assert!(trajectory.is_v3(), "v3 truth sidecar");
        assert_eq!(manifest.truth_names.len(), 26);
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn missing_or_unreadable_provenance_inputs_fail_loudly() {
    let dir = dataset_dir("loud");
    let missing = dir.join("absent.yaml");
    let out = generate(&dir, &["--protocol-schema", missing.to_str().unwrap()]);
    assert!(!out.status.success(), "missing protocol schema must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("provenance: protocol schema"), "{stderr}");
    assert!(stderr.contains("absent.yaml"), "{stderr}");
    assert!(
        !dir.join("train/manifest.json").exists(),
        "inputs are checked before dataset bytes are written"
    );

    let not_a_dir = dir.join("not-a-dir");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(&not_a_dir, b"not a directory").unwrap();
    let out = generate(&dir, &["--measurement-dir", not_a_dir.to_str().unwrap()]);
    assert!(
        !out.status.success(),
        "unreadable measurement dir must fail"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("cannot read measurement directory"),
        "{stderr}"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn measurement_inputs_are_hashed_by_name_without_source_files() {
    let dir = dataset_dir("meas");
    let meas = dir.join("inputs");
    std::fs::create_dir_all(&meas).unwrap();
    let names = ["THRUST_curve.csv", "imu_allan.json", "servo_step.csv"];
    for name in names {
        std::fs::write(meas.join(name), format!("data for {name}\n")).unwrap();
    }
    // A source file with a measurement token in its name is not an input.
    std::fs::write(meas.join("servo.rs"), b"fn main() {}").unwrap();

    let out = generate(&dir, &["--measurement-dir", meas.to_str().unwrap()]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let provenance = read_provenance(&dir.join("train/manifest.provenance.json")).unwrap();
    assert_eq!(provenance.measurement_hashes.len(), 3);
    for name in names {
        assert_eq!(
            provenance.measurement_hashes.get(name).unwrap(),
            &sha256_file(&meas.join(name)).unwrap()
        );
    }
    assert!(!provenance.measurement_hashes.contains_key("servo.rs"));
    std::fs::remove_dir_all(&dir).unwrap();
}
