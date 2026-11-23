//! Dataset generation whose command stream comes from the RLVR01 firmware
//! adapter instead of the offline `ReferenceOptimizer`.
//!
//! The simulator hard-codes the reference optimizer's `solve()` as the source
//! of the exported `g_accel` target (`episode.rs`). This module keeps those
//! four legacy diagnostic columns untouched but replaces the authoritative
//! v3 command with a [`firmware_adapter::FirmwareAdapter`] output, drives the
//! fin actuator with that very command, and records the origin of every 50 Hz
//! tick in a separate deterministic sidecar.
//!
//! # Contract
//!
//! * `validation.bin` stays a validation v3 file (six targets per frame,
//!   `g_accel_y` then `g_accel_x`) written by the standard writer. The truth
//!   metadata follows the caller's config, as in the standard generator.
//! * `validation.trajectory` stays a v3 privileged-truth sidecar (stride 26).
//!   Its per-tick `g_accel` is overwritten with the same firmware command.
//! * `manifest.json` is a standard `DatasetManifest`. The provenance sidecar
//!   is its third file record, so the record is hashed with the rest.
//! * `validation.command_provenance` is additive: one fixed-size record per
//!   tick with the adapter identity, its declared source, the fin command sent
//!   to the actuator, and the world-frame command exported to the dataset.
//!
//! No timestamps, process ids or global RNG are involved; regeneration from
//! the same `(config, seed, indices, provider)` is byte-identical.
//!
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::{
    cell::RefCell,
    collections::HashSet,
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
};

use crate::{
    config::Config,
    control::validate_command,
    episode::{ControlContext, Episode},
    error::{Result, SimError},
    firmware_adapter::FirmwareAdapter,
    file_record,
    manifest::{write_manifest, DatasetManifest, SCHEMA_VERSION},
    math::episode_seed,
    optimizer::ReferenceOptimizer,
    quality::observed_bounds,
    scenario::{scenario, Split},
    trajectory::{sidecar_path, write_trajectory_v3, TRUTH_V3_TICK_NAMES},
    validation::{read_validation, FILE_NAME, TARGET_NAMES},
    Simulator,
};

pub const PROVENANCE_MAGIC: &[u8; 8] = b"POTOKCMD";
pub const PROVENANCE_VERSION: u32 = 1;
pub const PROVENANCE_FILE_NAME: &str = "validation.command_provenance";
/// One byte is reserved for the NUL terminator.
pub const ADAPTER_NAME_MAX: usize = 63;
const NAME_FIELD: usize = ADAPTER_NAME_MAX + 1;
/// frame u32 + source u8 + health u8 + reserved u16 + name 64 + four f32 = 88.
pub const RECORD_SIZE: usize = 4 + 1 + 1 + 2 + NAME_FIELD + 4 * 4;
const HEADER_SIZE: usize = 64;
const _: () = assert!(8 + 4 + 4 + 4 + 4 + 40 == HEADER_SIZE);
const _: () = assert!(RECORD_SIZE == 88);

/// Who produced one command. Recorded per tick, never inferred from a Rust
/// type: a firmware implementation may replay a reference controller without
/// forging a firmware claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandSource {
    /// The offline privileged LQR reference (`ReferenceOptimizer`).
    OfflineOptimizer,
    /// An RLVR01 `FirmwareAdapter`.
    Firmware,
}

impl CommandSource {
    pub fn as_u8(self) -> u8 {
        match self {
            Self::OfflineOptimizer => 0,
            Self::Firmware => 1,
        }
    }

    fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(Self::OfflineOptimizer),
            1 => Ok(Self::Firmware),
            _ => Err(SimError::Dataset(format!("unknown command source tag {v}"))),
        }
    }
}

/// Identity of the command producer for one generation call. RLVR01's trait
/// deliberately has no name or source method, so the caller supplies it.
#[derive(Debug, Clone, PartialEq)]
pub struct CommandProvenanceMeta {
    pub adapter: String,
    pub source: CommandSource,
}

impl CommandProvenanceMeta {
    pub fn firmware(name: impl Into<String>) -> Self {
        Self {
            adapter: name.into(),
            source: CommandSource::Firmware,
        }
    }

    pub fn offline_optimizer() -> Self {
        Self {
            adapter: "offline_optimizer".into(),
            source: CommandSource::OfflineOptimizer,
        }
    }
}

/// Selects the command source for a whole split.
pub enum CommandProvider {
    /// Drive with an RLVR01 adapter. The adapter is reset once per episode
    /// with `episode_seed(config.run.seed, index)` and receives the phase at
    /// every tick before `step(frame)`.
    Firmware {
        adapter: Box<dyn FirmwareAdapter>,
        meta: CommandProvenanceMeta,
    },
    /// Byte-for-byte control arm: the privileged offline reference controller.
    OfflineReference,
}

/// Provenance of one tick's command, as written to the sidecar.
#[derive(Debug, Clone, PartialEq)]
pub struct FirmwareSample {
    pub frame: u32,
    pub source: CommandSource,
    pub adapter: String,
    /// Command handed to the fin actuator, `[roll, pitch]` m/s^2.
    pub fin_roll: f32,
    pub fin_pitch: f32,
    /// Command exported as the authoritative v3 target, `[ay, ax]` m/s^2.
    pub world_ay: f32,
    pub world_ax: f32,
    /// Whether the tick was outside the AI interlock with valid telemetry.
    pub ai_healthy: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EpisodeProvenance {
    pub index: u64,
    pub samples: Vec<FirmwareSample>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProvenanceSet {
    pub frames_per_episode: u32,
    pub episodes: Vec<EpisodeProvenance>,
}

impl ProvenanceSet {
    pub fn tick_count(&self) -> u64 {
        self.episodes.iter().map(|e| e.samples.len() as u64).sum()
    }
}

fn check_name(name: &str) -> Result<()> {
    if name.is_empty() || name.as_bytes().len() > ADAPTER_NAME_MAX || name.as_bytes().contains(&0) {
        return Err(SimError::Dataset(format!(
            "adapter name {name:?} must be 1..={ADAPTER_NAME_MAX} UTF-8 bytes without NUL"
        )));
    }
    Ok(())
}

fn encode_name(name: &str) -> Result<[u8; NAME_FIELD]> {
    check_name(name)?;
    let mut out = [0u8; NAME_FIELD];
    out[..name.len()].copy_from_slice(name.as_bytes());
    Ok(out)
}

fn decode_name(raw: &[u8; NAME_FIELD]) -> Result<String> {
    let end = raw.iter().position(|b| *b == 0).unwrap_or(NAME_FIELD);
    if end == 0 {
        return Err(SimError::Dataset("empty adapter name in provenance".into()));
    }
    if raw[end..].iter().any(|b| *b != 0) {
        return Err(SimError::Dataset(
            "adapter name in provenance is not NUL-terminated".into(),
        ));
    }
    std::str::from_utf8(&raw[..end])
        .map(|s| s.to_string())
        .map_err(|e| SimError::Dataset(format!("adapter name in provenance is not UTF-8: {e}")))
}

/// Deterministic binary writer for per-tick command provenance.
pub fn write_provenance(path: &Path, set: &ProvenanceSet) -> Result<()> {
    let frames = set.frames_per_episode as usize;
    for ep in &set.episodes {
        if ep.samples.len() != frames {
            return Err(SimError::Dataset(format!(
                "episode {} has {} provenance samples, expected {frames}",
                ep.index,
                ep.samples.len()
            )));
        }
    }
    let mut w = BufWriter::new(File::create(path)?);
    let mut b = [0u8; HEADER_SIZE];
    b[0..8].copy_from_slice(PROVENANCE_MAGIC);
    b[8..12].copy_from_slice(&PROVENANCE_VERSION.to_le_bytes());
    b[12..16].copy_from_slice(&(set.episodes.len() as u32).to_le_bytes());
    b[16..20].copy_from_slice(&set.frames_per_episode.to_le_bytes());
    b[20..24].copy_from_slice(&(RECORD_SIZE as u32).to_le_bytes());
    w.write_all(&b)?;
    for ep in &set.episodes {
        w.write_all(&ep.index.to_le_bytes())?;
        for (frame, s) in ep.samples.iter().enumerate() {
            if s.frame as usize != frame {
                return Err(SimError::Dataset(format!(
                    "episode {} sample {} has frame {}",
                    ep.index, frame, s.frame
                )));
            }
            let name = encode_name(&s.adapter)?;
            let mut r = [0u8; RECORD_SIZE];
            r[0..4].copy_from_slice(&s.frame.to_le_bytes());
            r[4] = s.source.as_u8();
            r[5] = s.ai_healthy as u8;
            // 6..8 reserved, already zero.
            r[8..8 + NAME_FIELD].copy_from_slice(&name);
            let mut o = 8 + NAME_FIELD;
            for v in [s.fin_roll, s.fin_pitch, s.world_ay, s.world_ax] {
                r[o..o + 4].copy_from_slice(&v.to_le_bytes());
                o += 4;
            }
            debug_assert_eq!(o, RECORD_SIZE);
            w.write_all(&r)?;
        }
    }
    w.flush()?;
    Ok(())
}

/// Strict reader: a wrong magic, version, stride, tag, frame index or name is
/// a LOUD error. No field is ever synthesized or zero-filled.
pub fn read_provenance(path: &Path) -> Result<ProvenanceSet> {
    let mut r = BufReader::new(File::open(path)?);
    let mut b = [0u8; HEADER_SIZE];
    r.read_exact(&mut b)?;
    if &b[0..8] != PROVENANCE_MAGIC {
        return Err(SimError::Dataset(format!(
            "{}: bad provenance magic",
            path.display()
        )));
    }
    let u32_at = |off: usize| u32::from_le_bytes(b[off..off + 4].try_into().unwrap());
    let version = u32_at(8);
    if version != PROVENANCE_VERSION {
        return Err(SimError::Dataset(format!(
            "{}: unsupported provenance version {version}",
            path.display()
        )));
    }
    let episode_count = u32_at(12) as usize;
    let frames = u32_at(16) as usize;
    if u32_at(20) as usize != RECORD_SIZE {
        return Err(SimError::Dataset(format!(
            "{}: provenance record size {} != {RECORD_SIZE}",
            path.display(),
            u32_at(20)
        )));
    }
    if episode_count == 0 || frames == 0 {
        return Err(SimError::Dataset(format!(
            "{}: empty provenance (episodes={episode_count}, frames={frames})",
            path.display()
        )));
    }
    let mut episodes = Vec::with_capacity(episode_count);
    for _ in 0..episode_count {
        let mut idx = [0u8; 8];
        r.read_exact(&mut idx)?;
        let index = u64::from_le_bytes(idx);
        let mut samples = Vec::with_capacity(frames);
        for frame in 0..frames {
            let mut rec = [0u8; RECORD_SIZE];
            r.read_exact(&mut rec)?;
            let got_frame = u32::from_le_bytes(rec[0..4].try_into().unwrap());
            if got_frame as usize != frame {
                return Err(SimError::Dataset(format!(
                    "episode {index}: provenance frame {got_frame}, expected {frame}"
                )));
            }
            let source = CommandSource::from_u8(rec[4])?;
            let health = rec[5];
            if health > 1 {
                return Err(SimError::Dataset(format!(
                    "episode {index} frame {frame}: bad ai_healthy {health}"
                )));
            }
            if rec[6..8].iter().any(|v| *v != 0) {
                return Err(SimError::Dataset(format!(
                    "episode {index} frame {frame}: nonzero reserved bytes"
                )));
            }
            let mut name_bytes = [0u8; NAME_FIELD];
            name_bytes.copy_from_slice(&rec[8..8 + NAME_FIELD]);
            let name = decode_name(&name_bytes)?;
            let mut values = [0f32; 4];
            let mut o = 8 + NAME_FIELD;
            for v in values.iter_mut() {
                *v = f32::from_le_bytes(rec[o..o + 4].try_into().unwrap());
                o += 4;
            }
            for (label, v) in ["fin_roll", "fin_pitch", "world_ay", "world_ax"]
                .iter()
                .zip(values)
            {
                if !v.is_finite() || (v as f64).abs() > crate::control::MAX_COMMAND_MPS2 {
                    return Err(SimError::Dataset(format!(
                        "episode {index} frame {frame}: {label}={v} is not a bounded command"
                    )));
                }
            }
            samples.push(FirmwareSample {
                frame: got_frame,
                source,
                adapter: name,
                fin_roll: values[0],
                fin_pitch: values[1],
                world_ay: values[2],
                world_ax: values[3],
                ai_healthy: health == 1,
            });
        }
        episodes.push(EpisodeProvenance { index, samples });
    }
    Ok(ProvenanceSet {
        frames_per_episode: frames as u32,
        episodes,
    })
}

/// One proposed command. `fin_*` stays f64 so the offline reference arm can
/// reproduce the oracle's exact fin command; `world_*` is the f32 value
/// exported to the dataset and recorded in provenance.
struct Proposed {
    world_ay: f32,
    world_ax: f32,
    fin_roll: f64,
    fin_pitch: f64,
    ai_healthy: bool,
    valid: bool,
}

/// Shared seam: run one episode with a privileged tick context, convert the
/// producer's world-frame command into a fin command, and record provenance.
fn generate_episode_inner(
    sim: &Simulator,
    index: u64,
    meta: &CommandProvenanceMeta,
    provide: impl FnMut(&ControlContext<'_>) -> Proposed,
) -> Result<(Episode, Vec<FirmwareSample>)> {
    check_name(&meta.adapter)?;
    let source = meta.source;
    let adapter_name = meta.adapter.clone();
    let samples = RefCell::new(Vec::new());
    let early_error = RefCell::new(None::<SimError>);
    let provide = RefCell::new(provide);
    let mut episode = sim.run_episode_with_control_ctx(index, &|ctx: &ControlContext<'_>| {
        let proposed = provide.borrow_mut()(ctx);
        if !proposed.valid {
            *early_error.borrow_mut() = Some(SimError::Dataset(format!(
                "episode {index} frame {}: adapter {adapter_name} produced an unbounded command",
                ctx.frame.seq
            )));
            return (0.0, 0.0);
        }
        match validate_command(proposed.fin_roll, proposed.fin_pitch) {
            Ok((fin_roll, fin_pitch)) => {
                samples.borrow_mut().push(FirmwareSample {
                    frame: ctx.frame.seq as u32,
                    source,
                    adapter: adapter_name.clone(),
                    fin_roll: fin_roll as f32,
                    fin_pitch: fin_pitch as f32,
                    world_ay: proposed.world_ay,
                    world_ax: proposed.world_ax,
                    ai_healthy: proposed.ai_healthy,
                });
                (fin_roll, fin_pitch)
            }
            Err(e) => {
                *early_error.borrow_mut() = Some(e);
                (0.0, 0.0)
            }
        }
    })?;
    if let Some(e) = early_error.borrow_mut().take() {
        return Err(e);
    }
    let samples = samples.into_inner();
    let frames = sim.config().frames_per_episode();
    if samples.len() != frames {
        return Err(SimError::Dataset(format!(
            "episode {index}: adapter produced {} commands for {frames} ticks",
            samples.len()
        )));
    }
    if episode.targets.len() != frames || episode.trajectory.len() != frames {
        return Err(SimError::Dataset(format!(
            "episode {index}: simulator exported {} targets / {} truth ticks for {frames} frames",
            episode.targets.len(),
            episode.trajectory.len()
        )));
    }
    for (i, s) in samples.iter().enumerate() {
        let command = [s.world_ay, s.world_ax];
        episode.targets[i].g_accel = command;
        episode.trajectory[i].g_accel = command;
    }
    Ok((episode, samples))
}

/// Run one episode with an RLVR01 firmware adapter as the command source.
/// The adapter is reset with a per-episode seed and told the phase before
/// every `step(frame)`, as its contract requires.
pub fn generate_episode(
    sim: &Simulator,
    index: u64,
    adapter: &mut dyn FirmwareAdapter,
    meta: &CommandProvenanceMeta,
) -> Result<(Episode, Vec<FirmwareSample>)> {
    let seed = episode_seed(sim.config().run.seed, index);
    adapter.reset(seed);
    generate_episode_inner(sim, index, meta, |ctx| {
        adapter.on_phase(ctx.phase);
        let command = adapter.step(ctx.frame);
        let healthy = command.ai_healthy && ctx.frame.ai_input_valid() && ctx.phase.ai_allowed();
        Proposed {
            world_ay: command.ay,
            world_ax: command.ax,
            fin_roll: command.ay as f64,
            fin_pitch: -(command.ax as f64),
            ai_healthy: healthy,
            valid: command.checked().is_some(),
        }
    })
}

/// Byte-for-byte control arm: drive the fin with the offline reference
/// optimizer and label every tick as [`CommandSource::OfflineOptimizer`].
pub fn generate_reference_episode(
    sim: &Simulator,
    index: u64,
) -> Result<(Episode, Vec<FirmwareSample>)> {
    let opt = ReferenceOptimizer::new(sim.config());
    generate_episode_inner(sim, index, &CommandProvenanceMeta::offline_optimizer(), |ctx| {
        let [ax, ay] = opt.world_accel_command(ctx.state, ctx.wind, ctx.phase, ctx.frame);
        let healthy = ctx.frame.ai_input_valid() && ctx.phase.ai_allowed();
        Proposed {
            world_ay: ay as f32,
            world_ax: ax as f32,
            fin_roll: ay,
            fin_pitch: -ax,
            ai_healthy: healthy,
            valid: true,
        }
    })
}

/// Episode indices of one scenario split, ascending. Shared with the CLI so a
/// split's membership is decided in exactly one place.
pub fn indices_for(split: Split, count: usize) -> Vec<u64> {
    (0u64..)
        .filter(|i| scenario(*i).split == split)
        .take(count)
        .collect()
}

pub struct FirmwareDatasetSpec<'a> {
    pub cfg: &'a Config,
    pub provider: CommandProvider,
    pub indices: &'a [u64],
    pub split: Split,
    pub out_dir: &'a Path,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FirmwareDatasetOutput {
    pub validation: PathBuf,
    pub trajectory: PathBuf,
    pub provenance: PathBuf,
    pub manifest: PathBuf,
}

/// Write one complete split: v3 validation bytes, v3 truth sidecar, command
/// provenance and a manifest that hashes all three files.
pub fn generate_dataset(mut spec: FirmwareDatasetSpec<'_>) -> Result<FirmwareDatasetOutput> {
    if spec.indices.is_empty() {
        return Err(SimError::Dataset(format!(
            "{} split needs at least one episode",
            spec.split.label()
        )));
    }
    let mut seen = HashSet::with_capacity(spec.indices.len());
    for &i in spec.indices {
        if !seen.insert(i) {
            return Err(SimError::Dataset(format!(
                "{} split repeats episode index {i}",
                spec.split.label()
            )));
        }
        if scenario(i).split != spec.split {
            return Err(SimError::Dataset(format!(
                "episode {i} belongs to {}, not {}",
                scenario(i).split.label(),
                spec.split.label()
            )));
        }
    }
    let sim = Simulator::new(spec.cfg.clone())?;
    let mut episodes = Vec::with_capacity(spec.indices.len());
    let mut set = ProvenanceSet {
        frames_per_episode: spec.cfg.frames_per_episode() as u32,
        episodes: Vec::with_capacity(spec.indices.len()),
    };
    for &i in spec.indices {
        let (episode, samples) = match &mut spec.provider {
            CommandProvider::Firmware { adapter, meta } => {
                generate_episode(&sim, i, adapter.as_mut(), meta)?
            }
            CommandProvider::OfflineReference => generate_reference_episode(&sim, i)?,
        };
        episodes.push(episode);
        set.episodes.push(EpisodeProvenance { index: i, samples });
    }

    std::fs::create_dir_all(spec.out_dir)?;
    let validation = spec.out_dir.join(FILE_NAME);
    let trajectory = sidecar_path(&validation);
    let provenance = spec.out_dir.join(PROVENANCE_FILE_NAME);
    let manifest = spec.out_dir.join("manifest.json");
    crate::validation::write_validation(&validation, spec.cfg, &episodes)?;
    write_trajectory_v3(&trajectory, spec.cfg, &episodes)?;
    write_provenance(&provenance, &set)?;

    let v = read_validation(&validation)?;
    let t = crate::read_trajectory(&trajectory)?;
    let mut scenario_ids: Vec<u16> = v.episodes.iter().map(|e| e.scenario_id).collect();
    scenario_ids.sort_unstable();
    scenario_ids.dedup();
    let manifest_data = DatasetManifest {
        schema_version: SCHEMA_VERSION,
        config_hash: crate::config_hash(spec.cfg)?,
        seed: spec.cfg.run.seed,
        split: spec.split.label().to_string(),
        frame_count: spec.indices.len() as u64 * set.frames_per_episode as u64,
        episode_count: spec.indices.len() as u32,
        frames_per_episode: set.frames_per_episode,
        scenario_ids,
        feature_names: spec.cfg.output.features.clone(),
        target_names: TARGET_NAMES.iter().map(|s| s.to_string()).collect(),
        truth_names: TRUTH_V3_TICK_NAMES.iter().map(|s| s.to_string()).collect(),
        bounds: observed_bounds(spec.cfg, &v, &t),
        files: vec![
            file_record(FILE_NAME, &validation)?,
            file_record("validation.trajectory", &trajectory)?,
            file_record(PROVENANCE_FILE_NAME, &provenance)?,
        ],
    };
    write_manifest(&manifest, &manifest_data)?;
    Ok(FirmwareDatasetOutput {
        validation,
        trajectory,
        provenance,
        manifest,
    })
}
