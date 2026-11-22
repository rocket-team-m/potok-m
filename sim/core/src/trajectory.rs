//! Per-tick flown truth in a sidecar next to `validation.bin`.
//!
//! The canonical validation file is untouched, so existing AI readers keep
//! working. This sidecar is where apogee and lateral drift are measured from
//! the trajectory the vehicle actually flew, and where a privileged LoopWM
//! teacher reads the full state it is allowed to see.
//!
//! Binary layout (little-endian): 64-byte header — magic `POTOKTRJ` (8),
//! version (4), episode_count (4), frames_per_episode (4), stride (4),
//! reserved (40) — then per episode a u64 index and `frames x stride` f32.
//!
//! Version 1 (stride 6): `x, y, z, vx, vy, vz` in the world frame (m, m/s) at
//! each 50 Hz tick. `write_trajectory` still emits exactly these bytes.
//!
//! Version 2 (stride 24) adds the privileged truth at every tick:
//! `x, y, z (0..3), vx, vy, vz (3..6), quat w/x/y/z (6..10), omega x/y/z
//! (10..13), mass_kg (13), wind x/y/z (14..17), gust_start_s (17),
//! gust_amplitude (18), shear_altitude_m (19), shear_delta (20),
//! shear_thickness_m (21), g_theta roll/pitch (22..24)`. `g_theta` is the
//! reference optimizer target in radians; it is zero on the AI interlock and
//! on dynamics-only runs. The version and stride in the header are the
//! authority: readers accept both layouts and v1 ticks carry no privileged
//! fields (`gust_start_s = -1`, other extras zero).

use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
};

use crate::{
    config::Config,
    episode::Episode,
    error::{Result, SimError},
};

pub const MAGIC: &[u8; 8] = b"POTOKTRJ";
pub const VERSION: u32 = 1;
pub const VERSION_V2: u32 = 2;
pub const STRIDE: usize = 6;
pub const TRUTH_STRIDE: usize = 24;
pub const HEADER_SIZE: usize = 64;
const _: () = assert!(8 + 4 + 4 + 4 + 4 + 40 == HEADER_SIZE);
/// Extension used for the sidecar written beside `validation.bin`.
pub const FILE_SUFFIX: &str = "trajectory";

/// Names of the 24 f32 fields of a v2 tick, in on-disk order.
pub const TRUTH_TICK_NAMES: [&str; TRUTH_STRIDE] = [
    "pos_x",
    "pos_y",
    "pos_z",
    "vel_x",
    "vel_y",
    "vel_z",
    "quat_w",
    "quat_x",
    "quat_y",
    "quat_z",
    "omega_x",
    "omega_y",
    "omega_z",
    "mass_kg",
    "wind_x",
    "wind_y",
    "wind_z",
    "gust_start_s",
    "gust_amplitude",
    "shear_altitude_m",
    "shear_delta",
    "shear_thickness_m",
    "g_theta_roll",
    "g_theta_pitch",
];

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrajectoryTick {
    /// World-frame position (m). Present in v1 and v2.
    pub pos: [f32; 3],
    /// World-frame velocity (m/s). Present in v1 and v2.
    pub vel: [f32; 3],
    /// `(w, x, y, z)`, un-noised RK4 state. V2 only.
    pub quat: [f32; 4],
    /// Body angular rate (rad/s). V2 only.
    pub omega: [f32; 3],
    /// Un-noised RK4 mass (kg). V2 only.
    pub mass_kg: f32,
    /// True wind at the tick, world frame (m/s). V2 only.
    pub wind: [f32; 3],
    /// Gust start (s), -1 when no gust. V2 only.
    pub gust_start_s: f32,
    /// Gust amplitude (m/s). V2 only.
    pub gust_amplitude: f32,
    /// Effective shear-layer altitude (m). V2 only.
    pub shear_altitude_m: f32,
    /// Effective shear velocity jump (m/s). V2 only.
    pub shear_delta: f32,
    /// Shear-layer thickness (m). V2 only.
    pub shear_thickness_m: f32,
    /// Reference optimizer target `[roll, pitch]` (rad). V2 only.
    pub g_theta: [f32; 2],
}

impl Default for TrajectoryTick {
    fn default() -> Self {
        Self {
            pos: [0.0; 3],
            vel: [0.0; 3],
            quat: [0.0; 4],
            omega: [0.0; 3],
            mass_kg: 0.0,
            wind: [0.0; 3],
            gust_start_s: -1.0,
            gust_amplitude: 0.0,
            shear_altitude_m: 0.0,
            shear_delta: 0.0,
            shear_thickness_m: 0.0,
            g_theta: [0.0; 2],
        }
    }
}

impl TrajectoryTick {
    /// Classic v1 record: six floats, byte-identical to the old writer.
    pub fn as_array(self) -> [f32; STRIDE] {
        let mut a = [0.0; STRIDE];
        a[0..3].copy_from_slice(&self.pos);
        a[3..6].copy_from_slice(&self.vel);
        a
    }

    pub fn from_array(a: [f32; STRIDE]) -> Self {
        Self {
            pos: [a[0], a[1], a[2]],
            vel: [a[3], a[4], a[5]],
            ..Self::default()
        }
    }

    /// Privileged v2 record: 24 floats in the order of `TRUTH_TICK_NAMES`.
    pub fn as_truth_array(self) -> [f32; TRUTH_STRIDE] {
        let mut a = [0.0; TRUTH_STRIDE];
        a[0..3].copy_from_slice(&self.pos);
        a[3..6].copy_from_slice(&self.vel);
        a[6..10].copy_from_slice(&self.quat);
        a[10..13].copy_from_slice(&self.omega);
        a[13] = self.mass_kg;
        a[14..17].copy_from_slice(&self.wind);
        a[17] = self.gust_start_s;
        a[18] = self.gust_amplitude;
        a[19] = self.shear_altitude_m;
        a[20] = self.shear_delta;
        a[21] = self.shear_thickness_m;
        a[22..24].copy_from_slice(&self.g_theta);
        a
    }

    pub fn from_truth_array(a: [f32; TRUTH_STRIDE]) -> Self {
        Self {
            pos: [a[0], a[1], a[2]],
            vel: [a[3], a[4], a[5]],
            quat: [a[6], a[7], a[8], a[9]],
            omega: [a[10], a[11], a[12]],
            mass_kg: a[13],
            wind: [a[14], a[15], a[16]],
            gust_start_s: a[17],
            gust_amplitude: a[18],
            shear_altitude_m: a[19],
            shear_delta: a[20],
            shear_thickness_m: a[21],
            g_theta: [a[22], a[23]],
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EpisodeTrajectory {
    pub index: u64,
    pub ticks: Vec<TrajectoryTick>,
}

impl EpisodeTrajectory {
    pub fn apogee_m(&self) -> f64 {
        apogee_m(&self.ticks)
    }

    pub fn lateral_rms_m(&self) -> f64 {
        lateral_rms_m(&self.ticks)
    }
}

pub fn apogee_m(ticks: &[TrajectoryTick]) -> f64 {
    ticks
        .iter()
        .map(|t| t.pos[2] as f64)
        .fold(0.0_f64, f64::max)
}

/// RMS of horizontal displacement over the flown ticks, metres.
pub fn lateral_rms_m(ticks: &[TrajectoryTick]) -> f64 {
    if ticks.is_empty() {
        return 0.0;
    }
    let sum: f64 = ticks
        .iter()
        .map(|t| (t.pos[0] as f64).powi(2) + (t.pos[1] as f64).powi(2))
        .sum();
    (sum / ticks.len() as f64).sqrt()
}

#[derive(Debug, Clone)]
pub struct TrajectorySet {
    /// 1 = six-float flown truth, 2 = 24-float privileged truth.
    pub version: u32,
    pub episode_count: u32,
    pub frames_per_episode: u32,
    pub episodes: Vec<EpisodeTrajectory>,
}

impl TrajectorySet {
    pub fn is_v2(&self) -> bool {
        self.version == VERSION_V2
    }
}

/// Sidecar path for a validation file: `dir/validation.bin` ->
/// `dir/validation.trajectory`.
pub fn sidecar_path(validation_path: &Path) -> PathBuf {
    validation_path.with_extension(FILE_SUFFIX)
}

/// Legacy v1 writer. The emitted bytes are unchanged from the original tool.
pub fn write_trajectory(path: &Path, cfg: &Config, episodes: &[Episode]) -> Result<()> {
    write_trajectory_with_version(path, cfg, episodes, VERSION)
}

/// Privileged v2 writer: 24 f32 per tick, version 2 in the header.
pub fn write_trajectory_v2(path: &Path, cfg: &Config, episodes: &[Episode]) -> Result<()> {
    write_trajectory_with_version(path, cfg, episodes, VERSION_V2)
}

pub fn write_trajectory_with_version(
    path: &Path,
    cfg: &Config,
    episodes: &[Episode],
    version: u32,
) -> Result<()> {
    let stride = match version {
        VERSION => STRIDE,
        VERSION_V2 => TRUTH_STRIDE,
        _ => {
            return Err(SimError::Trajectory(format!(
                "unsupported version {version}"
            )))
        }
    };
    let frames = cfg.frames_per_episode();
    for e in episodes {
        if e.trajectory.len() != frames {
            return Err(SimError::Trajectory(format!(
                "episode {} has {} truth ticks, expected {frames}",
                e.meta.index,
                e.trajectory.len()
            )));
        }
    }
    let mut w = BufWriter::new(File::create(path)?);
    let mut b = [0u8; HEADER_SIZE];
    b[0..8].copy_from_slice(MAGIC);
    b[8..12].copy_from_slice(&version.to_le_bytes());
    b[12..16].copy_from_slice(&(episodes.len() as u32).to_le_bytes());
    b[16..20].copy_from_slice(&(frames as u32).to_le_bytes());
    b[20..24].copy_from_slice(&(stride as u32).to_le_bytes());
    w.write_all(&b)?;
    for e in episodes {
        w.write_all(&e.meta.index.to_le_bytes())?;
        for tick in &e.trajectory {
            let values = if version == VERSION {
                let mut a = [0f32; TRUTH_STRIDE];
                a[..STRIDE].copy_from_slice(&tick.as_array());
                a
            } else {
                tick.as_truth_array()
            };
            for v in values.iter().take(stride) {
                w.write_all(&v.to_le_bytes())?;
            }
        }
    }
    w.flush()?;
    Ok(())
}

pub fn read_trajectory(path: &Path) -> Result<TrajectorySet> {
    let mut r = BufReader::new(File::open(path)?);
    let mut b = [0u8; HEADER_SIZE];
    r.read_exact(&mut b)?;
    if &b[0..8] != MAGIC {
        return Err(SimError::Trajectory(format!(
            "{}: bad magic",
            path.display()
        )));
    }
    let u32_at = |off: usize| u32::from_le_bytes(b[off..off + 4].try_into().unwrap());
    let version = u32_at(8);
    let expected_stride = match version {
        VERSION => STRIDE,
        VERSION_V2 => TRUTH_STRIDE,
        _ => {
            return Err(SimError::Trajectory(format!(
                "{}: unsupported version {version}",
                path.display()
            )))
        }
    };
    let episode_count = u32_at(12);
    let frames_per_episode = u32_at(16) as usize;
    if u32_at(20) as usize != expected_stride {
        return Err(SimError::Trajectory(format!(
            "{}: stride {} does not match version {version}",
            path.display(),
            u32_at(20)
        )));
    }
    let mut episodes = Vec::with_capacity(episode_count as usize);
    for _ in 0..episode_count {
        let mut idx = [0u8; 8];
        r.read_exact(&mut idx)?;
        let mut ticks = Vec::with_capacity(frames_per_episode);
        for _ in 0..frames_per_episode {
            let mut a = [0f32; TRUTH_STRIDE];
            for v in a.iter_mut().take(expected_stride) {
                let mut raw = [0u8; 4];
                r.read_exact(&mut raw)?;
                *v = f32::from_le_bytes(raw);
            }
            ticks.push(if version == VERSION {
                let mut v1 = [0f32; STRIDE];
                v1.copy_from_slice(&a[..STRIDE]);
                TrajectoryTick::from_array(v1)
            } else {
                TrajectoryTick::from_truth_array(a)
            });
        }
        episodes.push(EpisodeTrajectory {
            index: u64::from_le_bytes(idx),
            ticks,
        });
    }
    Ok(TrajectorySet {
        version,
        episode_count,
        frames_per_episode: frames_per_episode as u32,
        episodes,
    })
}
