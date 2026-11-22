//! CI-readable gate for the `potok-protocol` no_std contract.
//!
//! Three independent checks, all runnable with plain `cargo test`:
//!   1. the crate builds for the real firmware target with no default features;
//!   2. a separate `#![no_std]` consumer (`src/no_std.rs`) builds against it;
//!   3. `std::error::Error` is present only with the `std` feature (compile
//!      fail without it, compile pass with it).
//!
//! Every failure message names the exact command/config that broke, so a CI
//! log is diagnostic without a profiler or a local reproducer.

use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const EMBEDDED_TARGET: &str = "thumbv7em-none-eabihf";

const STD_ERROR_PROBE: &str = "\
fn assert_std_error<T: std::error::Error>() {}\n\
pub fn check() { assert_std_error::<potok_protocol::Error>(); }\n";

fn cargo() -> OsString {
    std::env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo"))
}

fn crate_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn workspace_manifest() -> PathBuf {
    crate_dir()
        .parent()
        .expect("crate lives inside the protocol workspace")
        .join("Cargo.toml")
}

fn toml_path(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "\\\\")
}

/// Fresh standalone probe crate; `[workspace]` keeps it out of the parent
/// workspace and its own `--target-dir` avoids nested-cargo lock contention.
fn probe_dir(name: &str, std_feature: bool) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "potok-protocol-no-std-gate-{}-{}-{}",
        std::process::id(),
        name,
        if std_feature { "std" } else { "nostd" }
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(dir.join("src")).expect("create probe src dir");
    fs::create_dir_all(dir.join("target")).expect("create probe target dir");

    let features = if std_feature {
        ", features = [\"std\"]"
    } else {
        ""
    };
    let manifest = format!(
        "[package]\nname = \"gate_probe\"\nversion = \"0.0.0\"\nedition = \"2021\"\n\n\
         [workspace]\n\n\
         [dependencies]\npotok-protocol = {{ path = \"{}\", default-features = false{} }}\n",
        toml_path(&crate_dir()),
        features
    );
    fs::write(dir.join("Cargo.toml"), manifest).expect("write probe manifest");
    dir
}

fn check_probe(dir: &Path, target: Option<&str>) -> Output {
    let mut cmd = Command::new(cargo());
    cmd.arg("check")
        .arg("--offline")
        .arg("--quiet")
        .arg("--target-dir")
        .arg(dir.join("target"))
        .arg("--manifest-path")
        .arg(dir.join("Cargo.toml"));
    if let Some(target) = target {
        cmd.arg("--target").arg(target);
    }
    cmd.output().expect("run cargo check")
}

#[test]
fn generated_crate_compiles_no_std_on_embedded_target() {
    let mut cmd = Command::new(cargo());
    cmd.arg("check")
        .arg("--offline")
        .arg("--quiet")
        .arg("--manifest-path")
        .arg(workspace_manifest())
        .arg("-p")
        .arg("potok-protocol")
        .arg("--no-default-features")
        .arg("--target")
        .arg(EMBEDDED_TARGET)
        .arg("--target-dir")
        .arg(crate_dir().join("target").join("no-std-gate"));
    let output = cmd.output().expect("run cargo check");
    assert!(
        output.status.success(),
        "potok-protocol must compile with --no-default-features for {EMBEDDED_TARGET}:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn no_std_consumer_probe_compiles_on_embedded_target() {
    let dir = probe_dir("consumer", false);
    fs::write(dir.join("src/lib.rs"), include_str!("../src/no_std.rs"))
        .expect("write no_std probe source");
    let output = check_probe(&dir, Some(EMBEDDED_TARGET));
    assert!(
        output.status.success(),
        "a #![no_std] consumer of potok-protocol (src/no_std.rs) must compile for {EMBEDDED_TARGET}:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn std_error_impl_is_feature_gated() {
    let without = probe_dir("without-std", false);
    fs::write(without.join("src/lib.rs"), STD_ERROR_PROBE).expect("write negative probe");
    let negative = check_probe(&without, None);
    assert!(
        !negative.status.success(),
        "Error must NOT implement std::error::Error without the `std` feature; \
         the impl in lib.rs lost its #[cfg(feature = \"std\")] gate"
    );
    assert!(
        String::from_utf8_lossy(&negative.stderr).contains("std::error::Error"),
        "negative probe should fail on the std::error::Error bound, got:\n{}",
        String::from_utf8_lossy(&negative.stderr)
    );

    let with = probe_dir("with-std", true);
    fs::write(with.join("src/lib.rs"), STD_ERROR_PROBE).expect("write positive probe");
    let positive = check_probe(&with, None);
    assert!(
        positive.status.success(),
        "Error must implement std::error::Error with the `std` feature:\n{}",
        String::from_utf8_lossy(&positive.stderr)
    );
}
