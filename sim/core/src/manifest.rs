//! Deterministic JSON manifest for a generated validation split.
//!
//! One manifest lives next to each `validation.bin`. Every field is derived
//! from the config, the seed and the written bytes — no timestamps, no process
//! ids, no map iteration order — so regenerating with the same seed produces
//! byte-identical JSON.

use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    config::Config,
    error::{Result, SimError},
    trajectory::TRUTH_V3_TICK_NAMES,
    validation::TARGET_NAMES,
};

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NamedRange {
    pub name: String,
    pub min: f64,
    pub max: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Bounds {
    pub features: Vec<NamedRange>,
    pub targets: Vec<NamedRange>,
    /// Per-tick privileged truth from the v2 trajectory sidecar.
    pub truth: Vec<NamedRange>,
}

impl Bounds {
    pub fn empty() -> Self {
        Self {
            features: Vec::new(),
            targets: Vec::new(),
            truth: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FileRecord {
    /// Path relative to the manifest's directory, using `/` separators.
    pub name: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DatasetManifest {
    pub schema_version: u32,
    /// SHA-256 of the canonical TOML serialization of the effective config.
    pub config_hash: String,
    pub seed: u64,
    /// `train` or `held-out`.
    pub split: String,
    /// Sorted unique scenario ids actually present in this split.
    pub scenario_ids: Vec<u16>,
    pub episode_count: u32,
    /// Total frames in this split (`episode_count * frames_per_episode`).
    pub frame_count: u64,
    pub frames_per_episode: u32,
    pub feature_names: Vec<String>,
    pub target_names: Vec<String>,
    pub truth_names: Vec<String>,
    pub bounds: Bounds,
    /// `validation.bin` and its v2 `validation.trajectory` sidecar.
    pub files: Vec<FileRecord>,
}

pub fn canonical_config_bytes(cfg: &Config) -> Result<Vec<u8>> {
    toml::to_string(cfg)
        .map(|s| s.into_bytes())
        .map_err(|e| SimError::Dataset(format!("config is not serializable: {e}")))
}

pub fn config_hash(cfg: &Config) -> Result<String> {
    Ok(sha256_hex(&canonical_config_bytes(cfg)?))
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let mut out = String::with_capacity(64);
    for byte in hasher.finalize() {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let mut buf = Vec::new();
    BufReader::new(File::open(path)?).read_to_end(&mut buf)?;
    Ok(sha256_hex(&buf))
}

pub fn file_record(name: &str, path: &Path) -> Result<FileRecord> {
    let bytes = std::fs::metadata(path)?.len();
    Ok(FileRecord {
        name: name.to_string(),
        bytes,
        sha256: sha256_file(path)?,
    })
}

pub fn manifest_to_string(manifest: &DatasetManifest) -> Result<String> {
    let mut s = serde_json::to_string_pretty(manifest)
        .map_err(|e| SimError::Dataset(format!("manifest serialization: {e}")))?;
    s.push('\n');
    Ok(s)
}

pub fn manifest_from_str(text: &str) -> Result<DatasetManifest> {
    serde_json::from_str(text).map_err(|e| SimError::Dataset(format!("manifest parse: {e}")))
}

pub fn write_manifest(path: &Path, manifest: &DatasetManifest) -> Result<()> {
    let text = manifest_to_string(manifest)?;
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(text.as_bytes())?;
    w.flush()?;
    Ok(())
}

pub fn read_manifest(path: &Path) -> Result<DatasetManifest> {
    let mut text = String::new();
    BufReader::new(File::open(path)?).read_to_string(&mut text)?;
    manifest_from_str(&text)
}

/// Names recorded in the manifest; single authority shared with the writer.
pub fn default_target_names() -> Vec<String> {
    TARGET_NAMES.iter().map(|s| s.to_string()).collect()
}

/// Per-tick truth names of the generated (v3) sidecar.
pub fn default_truth_names() -> Vec<String> {
    TRUTH_V3_TICK_NAMES.iter().map(|s| s.to_string()).collect()
}
