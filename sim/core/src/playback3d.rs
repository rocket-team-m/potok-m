// SPDX-License-Identifier: GPL-3.0-or-later
//! Deterministic 3D playback artifact exported from a POTOKTRJ v3 sidecar.
//!
//! Pure export: reads the truth sidecar produced by `trajectory::write_trajectory_v3`
//! and emits a text glTF 2.0 document. No simulation, no physics changes, no GUI,
//! no clock, no RNG. The same sidecar bytes always produce the same document bytes.
//!
//! # Artifact contract
//!
//! - `asset.version = "2.0"`, one root node per flown tick with `translation`
//!   and `rotation` (glTF `[x, y, z, w]`), so any glTF loader can position a
//!   vehicle at every instant of the flight.
//! - `extras.playback3d` carries the schema, the 50 Hz tick rate, the coordinate
//!   convention and one entry per episode with its gust/shear markers.
//! - Each tick node's `extras` carries the unmodified v3 truth (position,
//!   quaternion, velocity, angular rate, mass), the derived flight phase, the
//!   rocket body axis in the sim frame, and the gust/shear fields.
//!
//! Sim world is +Z up / +X downrange; glTF is +Y up. Mapping is
//! `(x, y, z)_sim -> (x, z, -y)_gltf`, a -90 degree rotation about X. Node
//! rotations are the sim body-to-world quaternion pre-multiplied by that same
//! rotation, so a node's local +Z remains the rocket axis. Raw sim-frame values
//! stay in `extras` for the binary-sidecar consumers.
//!
//! The sidecar has no burn-end or phase field: `phase` is reconstructed from
//! the mass plateau and the flown state, mirroring `dynamics::phase_at` and the
//! workbench's `trajectory_overlay::phase_at`. Deployment is never invented.

use std::{
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

use serde::Serialize;

use crate::{
    error::{Result, SimError},
    trajectory::{read_trajectory, TrajectorySet, TrajectoryTick, TRUTH_V3_TICK_NAMES},
};

/// Schema of the `extras.playback3d` object and the per-tick `extras`.
pub const SCHEMA_VERSION: u32 = 1;
/// The truth sidecar stores one record every 50 Hz tick.
pub const TICK_HZ: f64 = 50.0;
/// Extension used for the artifact written beside `validation.bin`.
pub const FILE_SUFFIX: &str = "playback3d.json";
/// f32 mass resolution at ~10 kg; smaller drops are rounding noise.
const MASS_EPS: f32 = 1e-6;
/// q_M in `[w, x, y, z]`: -90 degrees about X (sim +Z up -> glTF +Y up).
const MAP_Q: [f64; 4] = [
    std::f64::consts::FRAC_1_SQRT_2,
    -std::f64::consts::FRAC_1_SQRT_2,
    0.0,
    0.0,
];

/// Sidecar path for a validation file: `dir/validation.bin` ->
/// `dir/validation.playback3d.json`.
pub fn playback3d_path(validation_path: &Path) -> PathBuf {
    validation_path.with_extension(FILE_SUFFIX)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Asset {
    pub version: &'static str,
    pub generator: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Scene {
    pub nodes: Vec<u32>,
}

/// glTF node extras: the full v3 truth plus derived playback aids.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TickExtras {
    pub tick: usize,
    pub time_s: f64,
    pub phase: &'static str,
    pub phase_code: u8,
    /// Unmodified v3 world-frame position (m, +Z up).
    pub pos_sim: [f64; 3],
    /// Unmodified v3 body-to-world quaternion `[w, x, y, z]`.
    pub quat_wxyz: [f64; 4],
    /// Unmodified v3 world-frame velocity (m/s).
    pub vel_sim: [f64; 3],
    pub omega_sim: [f64; 3],
    pub mass_kg: f64,
    /// World-frame rocket axis (local +Z rotated by `quat_wxyz`), sim frame.
    pub body_axis_sim: [f64; 3],
    pub gust_start_s: f64,
    pub gust_amplitude: f64,
    /// True when the v3 tick carries a gust start time; never inferred.
    pub gust_armed: bool,
    pub shear_altitude_m: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Node {
    pub name: String,
    /// glTF world position `[x, y, z]`.
    pub translation: [f64; 3],
    /// glTF rotation `[x, y, z, w]`, body frame -> glTF world.
    pub rotation: [f64; 4],
    pub extras: TickExtras,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Marker {
    /// `gust` (start time and amplitude from v3) or `shear` (altitude crossing).
    pub kind: &'static str,
    pub tick: usize,
    pub time_s: f64,
    pub pos_sim: [f64; 3],
    /// Gust amplitude (m/s) or shear altitude (m), depending on `kind`.
    pub value: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EpisodeExtras {
    pub episode_index: u64,
    pub node_start: usize,
    pub tick_count: usize,
    pub markers: Vec<Marker>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PlaybackExtras {
    pub schema: u32,
    pub source: &'static str,
    pub tick_hz: u32,
    /// Coordinate convention: sim +Z up, glTF +Y up via `(x, z, -y)`.
    pub frame: &'static str,
    pub episodes: Vec<EpisodeExtras>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Document {
    pub asset: Asset,
    pub scene: u32,
    pub scenes: Vec<Scene>,
    pub nodes: Vec<Node>,
    pub extras: PlaybackExtras,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExportInfo {
    pub episodes: usize,
    pub ticks: usize,
    pub markers: usize,
    pub bytes: usize,
}

/// Build the in-memory document. Strictly v3: any other version, empty episode,
/// non-finite truth or zero-length quaternion is a loud error.
pub fn document(set: &TrajectorySet) -> Result<Document> {
    if !set.is_v3() {
        return Err(SimError::Trajectory(format!(
            "playback3d requires POTOKTRJ v3, got version {}",
            set.version
        )));
    }
    if set.episodes.is_empty() {
        return Err(SimError::Trajectory(
            "playback3d requires at least one episode".into(),
        ));
    }

    let mut nodes = Vec::new();
    let mut episodes = Vec::new();
    for episode in &set.episodes {
        if episode.ticks.is_empty() {
            return Err(SimError::Trajectory(format!(
                "playback3d: episode {} has no ticks",
                episode.index
            )));
        }
        for (i, tick) in episode.ticks.iter().enumerate() {
            validate_tick(episode.index, i, tick)?;
        }
        let node_start = nodes.len();
        let burn = burnout_tick(&episode.ticks);
        for (i, tick) in episode.ticks.iter().enumerate() {
            nodes.push(tick_node(episode.index, i, tick, burn));
        }
        let markers = markers_for(&episode.ticks);
        episodes.push(EpisodeExtras {
            episode_index: episode.index,
            node_start,
            tick_count: episode.ticks.len(),
            markers,
        });
    }

    let scene_nodes = (0..nodes.len() as u32).collect();
    Ok(Document {
        asset: Asset {
            version: "2.0",
            generator: "potok-sim playback3d",
        },
        scene: 0,
        scenes: vec![Scene { nodes: scene_nodes }],
        nodes,
        extras: PlaybackExtras {
            schema: SCHEMA_VERSION,
            source: "POTOKTRJ v3",
            tick_hz: TICK_HZ as u32,
            frame: "sim +Z up; glTF +Y up via (x, z, -y)",
            episodes,
        },
    })
}

/// Serialize the document as deterministic pretty-printed glTF JSON.
pub fn to_json(set: &TrajectorySet) -> Result<String> {
    let doc = document(set)?;
    serde_json::to_string_pretty(&doc)
        .map_err(|e| SimError::Trajectory(format!("playback3d serialize failed: {e}")))
}

/// Write the artifact. `out` is created/truncated; the byte count includes the
/// trailing newline.
pub fn write_file(set: &TrajectorySet, out: &Path) -> Result<ExportInfo> {
    let doc = document(set)?;
    let mut text = serde_json::to_string_pretty(&doc)
        .map_err(|e| SimError::Trajectory(format!("playback3d serialize failed: {e}")))?;
    text.push('\n');
    let bytes = text.len();
    let info = ExportInfo {
        episodes: doc.extras.episodes.len(),
        ticks: doc.nodes.len(),
        markers: doc.extras.episodes.iter().map(|e| e.markers.len()).sum(),
        bytes,
    };
    let mut writer = BufWriter::new(File::create(out)?);
    writer.write_all(text.as_bytes())?;
    writer.flush()?;
    Ok(info)
}

/// Read a POTOKTRJ v3 sidecar and write its playback artifact.
pub fn export_file(trajectory: &Path, out: &Path) -> Result<ExportInfo> {
    let set = read_trajectory(trajectory)?;
    write_file(&set, out)
}

fn validate_tick(episode: u64, tick: usize, t: &TrajectoryTick) -> Result<()> {
    let values = t.as_truth_v3_array();
    for (name, value) in TRUTH_V3_TICK_NAMES.iter().zip(values.iter()) {
        if !value.is_finite() {
            return Err(SimError::Trajectory(format!(
                "playback3d: non-finite {name} at episode {episode} tick {tick}"
            )));
        }
    }
    let q = t.quat;
    if q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3] < 1e-12 {
        return Err(SimError::Trajectory(format!(
            "playback3d: zero quaternion at episode {episode} tick {tick}"
        )));
    }
    Ok(())
}

fn tick_node(episode: u64, tick: usize, t: &TrajectoryTick, burn: Option<usize>) -> Node {
    let q = [
        t.quat[0] as f64,
        t.quat[1] as f64,
        t.quat[2] as f64,
        t.quat[3] as f64,
    ];
    let phase = phase_at(tick, t, burn);
    Node {
        name: format!("ep{episode}/t{tick}"),
        translation: map_pos(t.pos),
        rotation: map_quat(q),
        extras: TickExtras {
            tick,
            time_s: tick as f64 / TICK_HZ,
            phase,
            phase_code: phase_code(phase),
            pos_sim: [t.pos[0] as f64, t.pos[1] as f64, t.pos[2] as f64],
            quat_wxyz: q,
            vel_sim: [t.vel[0] as f64, t.vel[1] as f64, t.vel[2] as f64],
            omega_sim: [t.omega[0] as f64, t.omega[1] as f64, t.omega[2] as f64],
            mass_kg: t.mass_kg as f64,
            body_axis_sim: body_axis_sim(q),
            gust_start_s: t.gust_start_s as f64,
            gust_amplitude: t.gust_amplitude as f64,
            gust_armed: t.gust_start_s >= 0.0,
            shear_altitude_m: t.shear_altitude_m as f64,
        },
    }
}

/// Last tick with a meaningful mass drop: the final powered tick.
fn burnout_tick(ticks: &[TrajectoryTick]) -> Option<usize> {
    let mut last = None;
    for i in 1..ticks.len() {
        if ticks[i].mass_kg < ticks[i - 1].mass_kg - MASS_EPS {
            last = Some(i);
        }
    }
    last
}

/// Reconstructed from flown truth; mirrors `dynamics::phase_at` with the mass
/// plateau standing in for the sidecar's missing burn-end field.
fn phase_at(tick: usize, t: &TrajectoryTick, burn: Option<usize>) -> &'static str {
    let time_s = tick as f64 / TICK_HZ;
    if time_s > 0.5 && t.pos[2] <= 0.0 {
        return "LANDED";
    }
    if tick == 0 {
        return "PAD";
    }
    if burn.is_some_and(|b| tick <= b) {
        return "BOOST";
    }
    if t.vel[2] < 0.0 {
        return if t.pos[2] < 5.0 { "LANDED" } else { "DESCENT" };
    }
    let speed_sq = t.vel[0].powi(2) + t.vel[1].powi(2) + t.vel[2].powi(2);
    if speed_sq < 4.0 && t.pos[2] > 10.0 {
        "APOGEE"
    } else {
        "COAST"
    }
}

fn phase_code(phase: &'static str) -> u8 {
    match phase {
        "PAD" => 0,
        "BOOST" => 1,
        "COAST" => 2,
        "APOGEE" => 3,
        "DESCENT" => 4,
        "LANDED" => 5,
        other => unreachable!("unknown playback phase {other}"),
    }
}

fn markers_for(ticks: &[TrajectoryTick]) -> Vec<Marker> {
    let mut out = Vec::new();
    if let Some((start, amplitude)) = ticks
        .iter()
        .map(|t| (t.gust_start_s, t.gust_amplitude))
        .find(|(s, _)| s.is_finite() && *s >= 0.0)
    {
        let at = ((start as f64) * TICK_HZ).round();
        let at = at.clamp(0.0, ticks.len().saturating_sub(1) as f64) as usize;
        out.push(Marker {
            kind: "gust",
            tick: at,
            time_s: at as f64 / TICK_HZ,
            pos_sim: sim_pos(ticks[at].pos),
            value: amplitude as f64,
        });
    }
    let shear_alt = ticks.first().map_or(0.0, |t| t.shear_altitude_m);
    if shear_alt > 0.0 {
        if let Some(cross) = (1..ticks.len())
            .find(|&i| ticks[i - 1].pos[2] < shear_alt && ticks[i].pos[2] >= shear_alt)
        {
            out.push(Marker {
                kind: "shear",
                tick: cross,
                time_s: cross as f64 / TICK_HZ,
                pos_sim: sim_pos(ticks[cross].pos),
                value: shear_alt as f64,
            });
        }
    }
    out.sort_by_key(|m| (m.tick, m.kind));
    out
}

fn sim_pos(p: [f32; 3]) -> [f64; 3] {
    [p[0] as f64, p[1] as f64, p[2] as f64]
}

fn map_pos(p: [f32; 3]) -> [f64; 3] {
    [p[0] as f64, p[2] as f64, -(p[1] as f64)]
}

/// Hamilton product of two `[w, x, y, z]` quaternions: `R(a ⊗ b) = R(a) R(b)`.
fn quat_mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    let (aw, ax, ay, az) = (a[0], a[1], a[2], a[3]);
    let (bw, bx, by, bz) = (b[0], b[1], b[2], b[3]);
    [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]
}

/// Sim body-to-world quaternion -> glTF body-to-world quaternion `[x, y, z, w]`.
fn map_quat(q: [f64; 4]) -> [f64; 4] {
    let m = quat_mul(MAP_Q, q);
    [m[1], m[2], m[3], m[0]]
}

/// World-frame rocket axis: the body +Z column of the sim rotation matrix.
fn body_axis_sim(q: [f64; 4]) -> [f64; 3] {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    [
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        1.0 - 2.0 * (x * x + y * y),
    ]
}
