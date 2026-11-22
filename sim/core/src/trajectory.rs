//! Per-tick flown truth in a sidecar next to `validation.bin`.
//!
//! The canonical validation file is untouched, so existing AI readers keep
//! working. This sidecar is where apogee and lateral drift are measured from
//! the trajectory the vehicle actually flew.
//!
//! Binary layout (little-endian): 64-byte header — magic `POTOKTRJ` (8),
//! version (4), episode_count (4), frames_per_episode (4), stride = 6 (4),
//! reserved (40) — then per episode a u64 index and `frames x 6` f32:
//! `x, y, z, vx, vy, vz` in the world frame (m and m/s) at each 50 Hz tick.

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
pub const STRIDE: usize = 6;
pub const HEADER_SIZE: usize = 64;
const _: () = assert!(8 + 4 + 4 + 4 + 4 + 40 == HEADER_SIZE);
/// Extension used for the sidecar written beside `validation.bin`.
pub const FILE_SUFFIX: &str = "trajectory";

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrajectoryTick {
    pub pos: [f32; 3],
    pub vel: [f32; 3],
}

impl TrajectoryTick {
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
    pub episode_count: u32,
    pub frames_per_episode: u32,
    pub episodes: Vec<EpisodeTrajectory>,
}

/// Sidecar path for a validation file: `dir/validation.bin` ->
/// `dir/validation.trajectory`.
pub fn sidecar_path(validation_path: &Path) -> PathBuf {
    validation_path.with_extension(FILE_SUFFIX)
}

pub fn write_trajectory(path: &Path, cfg: &Config, episodes: &[Episode]) -> Result<()> {
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
    b[8..12].copy_from_slice(&VERSION.to_le_bytes());
    b[12..16].copy_from_slice(&(episodes.len() as u32).to_le_bytes());
    b[16..20].copy_from_slice(&(frames as u32).to_le_bytes());
    b[20..24].copy_from_slice(&(STRIDE as u32).to_le_bytes());
    w.write_all(&b)?;
    for e in episodes {
        w.write_all(&e.meta.index.to_le_bytes())?;
        for tick in &e.trajectory {
            for v in tick.as_array() {
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
    if version != VERSION {
        return Err(SimError::Trajectory(format!(
            "{}: unsupported version {version}",
            path.display()
        )));
    }
    let episode_count = u32_at(12);
    let frames_per_episode = u32_at(16) as usize;
    if u32_at(20) as usize != STRIDE {
        return Err(SimError::Trajectory(format!(
            "{}: unsupported stride",
            path.display()
        )));
    }
    let mut episodes = Vec::with_capacity(episode_count as usize);
    for _ in 0..episode_count {
        let mut idx = [0u8; 8];
        r.read_exact(&mut idx)?;
        let mut ticks = Vec::with_capacity(frames_per_episode);
        for _ in 0..frames_per_episode {
            let mut a = [0f32; STRIDE];
            for v in a.iter_mut() {
                let mut raw = [0u8; 4];
                r.read_exact(&mut raw)?;
                *v = f32::from_le_bytes(raw);
            }
            ticks.push(TrajectoryTick::from_array(a));
        }
        episodes.push(EpisodeTrajectory {
            index: u64::from_le_bytes(idx),
            ticks,
        });
    }
    Ok(TrajectorySet {
        episode_count,
        frames_per_episode: frames_per_episode as u32,
        episodes,
    })
}
