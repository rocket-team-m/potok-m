//! Deterministic JSON manifest for a generated validation split.
//!
//! One manifest lives next to each `validation.bin`. Every field is derived
//! from the config, the seed and the written bytes — no timestamps, no process
//! ids, no map iteration order — so regenerating with the same seed produces
//! byte-identical JSON.

use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    process::Command,
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    config::Config,
    episode::Episode,
    error::{Result, SimError},
    parachute::CHUTE_MANIFEST_FIELDS,
    trajectory::TRUTH_V3_TICK_NAMES,
    validation::TARGET_NAMES,
};

pub const SCHEMA_VERSION: u32 = 1;

/// Manifest wire versions the provenance binding accepts. The dataset itself
/// has three live wire versions: v1 legacy validation, v2 privileged truth,
/// v3 per-tick truth. A binding must not invalidate older splits.
pub const SUPPORTED_SCHEMA_VERSIONS: [u32; 3] = [1, 2, 3];

/// Wire version of the provenance companion document. New fields are
/// `#[serde(default)]` so an older, shorter JSON still parses and is then
/// rejected loudly if a bound hash is missing.
pub const PROVENANCE_SCHEMA_VERSION: u32 = 1;

/// Filename substrings recognised as measurement inputs. Matching is
/// case-insensitive and restricted to data extensions, so a source file named
/// `servo.rs` is not mistaken for a servo measurement.
pub const MEASUREMENT_TOKENS: [&str; 3] = ["thrust", "allan", "servo"];

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

// --- CHUTE03: optional canopy manifest block -------------------------------
//
// A separate block, not a field on `DatasetManifest`: that wire shape is built
// with exhaustive struct literals by the generator, the CLI and the firmware
// writer, none of which this task may touch. The nine names come from the
// parachute model so the emitter and the model cannot drift.

/// The nine canopy manifest fields, as min/max ranges over episodes that
/// landed. Empty when `[parachute] enabled = false`: fields are absent, not
/// zero-filled. With the canopy on, a missing landing is a LOUD error.
pub fn chute_manifest_bounds(cfg: &Config, episodes: &[Episode]) -> Result<Vec<NamedRange>> {
    if !cfg.parachute.enabled {
        return Ok(Vec::new());
    }
    let mut spans: Vec<(f64, f64)> = Vec::new();
    for e in episodes {
        let chute = e.chute.as_ref().ok_or_else(|| {
            SimError::Dataset(format!(
                "episode {} has no canopy telemetry; enable [parachute] and run to landing \
                 before emitting chute manifest fields",
                e.meta.index
            ))
        })?;
        let floats = chute.landing.manifest_named_floats(true);
        if spans.is_empty() {
            spans = floats.iter().map(|(_, v)| (*v, *v)).collect();
        } else {
            for (span, (_, v)) in spans.iter_mut().zip(floats.iter()) {
                span.0 = span.0.min(*v);
                span.1 = span.1.max(*v);
            }
        }
    }
    if spans.is_empty() {
        return Ok(Vec::new());
    }
    Ok(CHUTE_MANIFEST_FIELDS
        .iter()
        .zip(spans)
        .map(|(name, (min, max))| NamedRange {
            name: (*name).to_string(),
            min,
            max,
        })
        .collect())
}

/// `dir/manifest.json` -> `dir/manifest.chute.json`.
pub fn chute_manifest_path(manifest_path: &Path) -> PathBuf {
    manifest_path.with_extension("chute.json")
}

/// Write the optional canopy manifest block as JSON. An empty field list (the
/// disabled case) writes nothing at all, leaving no sidecar behind.
pub fn write_chute_manifest(path: &Path, fields: &[NamedRange]) -> Result<()> {
    if fields.is_empty() {
        return Ok(());
    }
    let text = serde_json::to_string_pretty(fields)
        .map_err(|e| SimError::Dataset(format!("chute manifest serialization: {e}")))?;
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(text.as_bytes())?;
    w.write_all(b"\n")?;
    w.flush()?;
    Ok(())
}

/// Read back the optional canopy manifest block written by
/// [`write_chute_manifest`].
pub fn read_chute_manifest(path: &Path) -> Result<Vec<NamedRange>> {
    let mut text = String::new();
    BufReader::new(File::open(path)?).read_to_string(&mut text)?;
    serde_json::from_str(&text).map_err(|e| SimError::Dataset(format!("chute manifest parse: {e}")))
}

// --- Provenance binding -----------------------------------------------------
//
// These are the manifest's audit fields. They live in a companion JSON next
// to `manifest.json` because the v1 `DatasetManifest` wire shape is shared by
// the generator, the CLI and the firmware dataset writer, all of which build
// it with exhaustive struct literals. A separate document keeps v1/v2/v3
// splits readable while binding the exact stack that produced them.

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProvenanceManifest {
    pub schema_version: u32,
    /// sha256 of the canonical JSON of the companion `manifest.json`.
    pub manifest_hash: String,
    /// `git rev-parse HEAD` of the code that produced the dataset.
    pub code_hash: String,
    /// sha256 of the canonical protocol schema (`protocol/uart.yaml`).
    pub protocol_schema_hash: String,
    /// `filename -> sha256` for measurement inputs found, sorted by filename.
    #[serde(default)]
    pub measurement_hashes: BTreeMap<String, String>,
    /// sha256 over the pinned seed-1337 episode-0 file hashes.
    pub golden_episode_hash: String,
}

/// External inputs a provenance document is checked against. `None` for
/// `measurement_dir` means the dataset has no measurement inputs, so the
/// recorded map must be empty.
#[derive(Debug, Clone, Copy)]
pub struct ProvenanceSources<'a> {
    pub repo: &'a Path,
    pub protocol_schema: &'a Path,
    pub measurement_dir: Option<&'a Path>,
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

fn require_sha256(label: &str, value: &str) -> Result<()> {
    if is_sha256_hex(value) {
        Ok(())
    } else {
        Err(SimError::Dataset(format!(
            "provenance: {label} is not a lowercase 64-char sha256: {value:?}; regenerate the binding"
        )))
    }
}

fn mismatch(field: &str, expected: &str, got: &str) -> SimError {
    SimError::Dataset(format!(
        "provenance mismatch: {field}: expected {expected}, got {got}; regenerate the dataset at the recorded inputs or update the binding"
    ))
}

/// LOUD git HEAD reader: no environment, no clock, no fallback. The escape in
/// the error is to pass the hash explicitly to [`build_provenance`].
pub fn git_head_hash(repo: &Path) -> Result<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(["rev-parse", "HEAD"])
        .output()
        .map_err(|e| {
            SimError::Dataset(format!(
                "provenance: cannot run git at {}: {e}; install git or pass the code hash explicitly",
                repo.display()
            ))
        })?;
    if !out.status.success() {
        return Err(SimError::Dataset(format!(
            "provenance: git rev-parse HEAD failed at {}: {}",
            repo.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        )));
    }
    let head = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if head.len() != 40
        || !head
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    {
        return Err(SimError::Dataset(format!(
            "provenance: git HEAD {head:?} is not a lowercase 40-char SHA-1"
        )));
    }
    Ok(head)
}

/// sha256 of the protocol schema bytes. A missing or unreadable schema is an
/// error that names the path.
pub fn protocol_schema_hash(path: &Path) -> Result<String> {
    sha256_file(path).map_err(|e| {
        SimError::Dataset(format!(
            "provenance: protocol schema {} is unreadable: {e}",
            path.display()
        ))
    })
}

/// Hashes thrust/Allan/servo measurement files found directly in `dir`.
/// Absent files are omitted, not zero-filled; a missing directory means no
/// measurement inputs. The map is keyed by bare filename and sorted, so the
/// JSON is byte-identical regardless of filesystem enumeration order.
pub fn measurement_hashes(dir: &Path) -> Result<BTreeMap<String, String>> {
    let entries = match std::fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(BTreeMap::new()),
        Err(e) => {
            return Err(SimError::Dataset(format!(
                "provenance: cannot read measurement directory {}: {e}",
                dir.display()
            )))
        }
    };
    let mut names = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| {
            SimError::Dataset(format!(
                "provenance: cannot enumerate measurement directory {}: {e}",
                dir.display()
            ))
        })?;
        if !entry
            .file_type()
            .map_err(|e| SimError::Dataset(format!("provenance: file type: {e}")))?
            .is_file()
        {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        let lower = name.to_ascii_lowercase();
        let data_file = [".csv", ".json", ".toml", ".txt", ".dat"]
            .iter()
            .any(|ext| lower.ends_with(ext));
        if data_file && MEASUREMENT_TOKENS.iter().any(|token| lower.contains(token)) {
            names.push(name);
        }
    }
    names.sort();
    let mut out = BTreeMap::new();
    for name in names {
        let hash = sha256_file(&dir.join(&name))?;
        out.insert(name, hash);
    }
    Ok(out)
}

pub const GOLDEN_SEED: u64 = 1337;

/// The eight raw channels of the golden export. Not the 14-channel model frame:
/// an unrelated physics-feature edit must not move the golden pin.
const GOLDEN_FEATURES: [&str; 8] = [
    "gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z", "baro_h", "pitot_q",
];

/// Regenerate the pinned seed-1337 episode-0 files in `dir` and return their
/// records. The bytes are the golden pin asserted in
/// `core/tests/manifest_hash_tests.rs`; any RNG or export change breaks both.
pub fn write_golden_episode(dir: &Path) -> Result<Vec<FileRecord>> {
    let mut cfg = Config::default();
    cfg.run.seed = GOLDEN_SEED;
    cfg.output.features = GOLDEN_FEATURES.iter().map(|s| (*s).to_string()).collect();
    let sim = crate::episode::Simulator::new(cfg.clone())?;
    let episodes = vec![sim.run_episode(0)?];
    std::fs::create_dir_all(dir)?;
    let validation = dir.join(crate::validation::FILE_NAME);
    let sidecar = crate::trajectory::sidecar_path(&validation);
    crate::validation::write_validation(&validation, &cfg, &episodes)?;
    crate::trajectory::write_trajectory_v3(&sidecar, &cfg, &episodes)?;
    Ok(vec![
        file_record(crate::validation::FILE_NAME, &validation)?,
        file_record("validation.trajectory", &sidecar)?,
    ])
}

/// Canonical sha256 over the pinned episode's file records. Each record is
/// encoded as `name NUL bytes NUL sha256 LF`, records are sorted by name; the
/// NUL separators make the encoding unambiguous.
pub fn golden_episode_hash(files: &[FileRecord]) -> Result<String> {
    if files.is_empty() {
        return Err(SimError::Dataset(
            "provenance: golden_episode_hash needs at least one file record".into(),
        ));
    }
    let mut rows: Vec<&FileRecord> = files.iter().collect();
    rows.sort_by(|a, b| a.name.cmp(&b.name));
    let mut buf = Vec::new();
    for rec in rows {
        if rec.name.is_empty() || rec.name.contains('\0') || rec.name.contains('\n') {
            return Err(SimError::Dataset(format!(
                "provenance: golden file name {:?} contains a separator",
                rec.name
            )));
        }
        require_sha256(&format!("golden file {}", rec.name), &rec.sha256)?;
        buf.extend_from_slice(rec.name.as_bytes());
        buf.push(0);
        buf.extend_from_slice(rec.bytes.to_string().as_bytes());
        buf.push(0);
        buf.extend_from_slice(rec.sha256.as_bytes());
        buf.push(b'\n');
    }
    Ok(sha256_hex(&buf))
}

/// Assemble a provenance document from already-computed inputs. Every hash is
/// checked for shape here; equality is checked by the `verify_*` functions.
pub fn build_provenance(
    manifest: &DatasetManifest,
    code_hash: &str,
    protocol_hash: &str,
    measurements: BTreeMap<String, String>,
    golden_hash: &str,
) -> Result<ProvenanceManifest> {
    if code_hash.len() != 40
        || !code_hash
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    {
        return Err(SimError::Dataset(format!(
            "provenance: code_hash {code_hash:?} is not a lowercase 40-char git SHA-1; pass `git rev-parse HEAD`"
        )));
    }
    require_sha256("protocol_schema_hash", protocol_hash)?;
    require_sha256("golden_episode_hash", golden_hash)?;
    for (name, hash) in &measurements {
        require_sha256(&format!("measurement hash for {name}"), hash)?;
    }
    Ok(ProvenanceManifest {
        schema_version: PROVENANCE_SCHEMA_VERSION,
        manifest_hash: sha256_hex(manifest_to_string(manifest)?.as_bytes()),
        code_hash: code_hash.to_string(),
        protocol_schema_hash: protocol_hash.to_string(),
        measurement_hashes: measurements,
        golden_episode_hash: golden_hash.to_string(),
    })
}

/// Deterministic JSON: one key order, no timestamps, trailing newline.
pub fn provenance_to_string(provenance: &ProvenanceManifest) -> Result<String> {
    let mut s = serde_json::to_string_pretty(provenance)
        .map_err(|e| SimError::Dataset(format!("provenance serialization: {e}")))?;
    s.push('\n');
    Ok(s)
}

pub fn provenance_from_str(text: &str) -> Result<ProvenanceManifest> {
    serde_json::from_str(text).map_err(|e| SimError::Dataset(format!("provenance parse: {e}")))
}

/// Sidecar path for a manifest: `manifest.json` -> `manifest.provenance.json`.
pub fn provenance_path(manifest_path: &Path) -> PathBuf {
    let name = manifest_path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy();
    let stem = name.strip_suffix(".json").unwrap_or(&name);
    manifest_path.with_file_name(format!("{stem}.provenance.json"))
}

pub fn write_provenance(path: &Path, provenance: &ProvenanceManifest) -> Result<()> {
    let text = provenance_to_string(provenance)?;
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(text.as_bytes())?;
    w.flush()?;
    Ok(())
}

pub fn read_provenance(path: &Path) -> Result<ProvenanceManifest> {
    let mut text = String::new();
    BufReader::new(File::open(path)?).read_to_string(&mut text)?;
    provenance_from_str(&text)
}

/// Bind the provenance document to its dataset manifest and check hash shape.
/// Accepts v1/v2/v3 wire versions; anything else is LOUD.
pub fn verify_provenance_binding(
    manifest: &DatasetManifest,
    provenance: &ProvenanceManifest,
) -> Result<()> {
    if !SUPPORTED_SCHEMA_VERSIONS.contains(&manifest.schema_version) {
        return Err(SimError::Dataset(format!(
            "provenance: unsupported manifest schema_version {}; supported {:?}",
            manifest.schema_version, SUPPORTED_SCHEMA_VERSIONS
        )));
    }
    if !SUPPORTED_SCHEMA_VERSIONS.contains(&provenance.schema_version) {
        return Err(SimError::Dataset(format!(
            "provenance: unsupported provenance schema_version {}; supported {:?}",
            provenance.schema_version, SUPPORTED_SCHEMA_VERSIONS
        )));
    }
    require_sha256("manifest_hash", &provenance.manifest_hash)?;
    require_sha256("protocol_schema_hash", &provenance.protocol_schema_hash)?;
    require_sha256("golden_episode_hash", &provenance.golden_episode_hash)?;
    for (name, hash) in &provenance.measurement_hashes {
        require_sha256(&format!("measurement hash for {name}"), hash)?;
    }
    let actual = sha256_hex(manifest_to_string(manifest)?.as_bytes());
    if actual != provenance.manifest_hash {
        return Err(mismatch(
            "manifest_hash",
            &provenance.manifest_hash,
            &actual,
        ));
    }
    Ok(())
}

pub fn verify_code_hash(provenance: &ProvenanceManifest, repo: &Path) -> Result<()> {
    let actual = git_head_hash(repo)?;
    if actual != provenance.code_hash {
        return Err(mismatch("code_hash", &provenance.code_hash, &actual));
    }
    Ok(())
}

pub fn verify_protocol_schema(provenance: &ProvenanceManifest, path: &Path) -> Result<()> {
    let actual = protocol_schema_hash(path)?;
    if actual != provenance.protocol_schema_hash {
        return Err(mismatch(
            "protocol_schema_hash",
            &provenance.protocol_schema_hash,
            &actual,
        ));
    }
    Ok(())
}

pub fn verify_measurement_inputs(
    provenance: &ProvenanceManifest,
    dir: Option<&Path>,
) -> Result<()> {
    let actual = match dir {
        Some(d) => measurement_hashes(d)?,
        None => BTreeMap::new(),
    };
    if actual != provenance.measurement_hashes {
        return Err(mismatch(
            "measurement_hashes",
            &format!("{:?}", provenance.measurement_hashes),
            &format!("{actual:?}"),
        ));
    }
    Ok(())
}

pub fn verify_golden_episode(provenance: &ProvenanceManifest, files: &[FileRecord]) -> Result<()> {
    let actual = golden_episode_hash(files)?;
    if actual != provenance.golden_episode_hash {
        return Err(mismatch(
            "golden_episode_hash",
            &provenance.golden_episode_hash,
            &actual,
        ));
    }
    Ok(())
}
