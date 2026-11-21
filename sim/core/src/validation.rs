//! Compact binary validation set: custom header followed by f32 arrays.
//!
//! Layout (little-endian), see sim/README.md for the full description:
//!   header, then for each episode: 9 metadata f32,
//!   frames x features f32 (row-major), frames x 4 targets f32 (row-major).

use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
};

use ndarray::Array2;

use crate::{
    config::Config,
    episode::Episode,
    error::{Result, SimError},
};

pub const MAGIC: &[u8; 8] = b"POTOKSIM";
pub const VERSION: u32 = 1;
pub const TARGET_COUNT: usize = 4; // g_roll, g_pitch, confidence, ai_healthy
pub const FILE_NAME: &str = "validation.bin";

/// The single authority for the on-disk header length: 8 magic + 4 version +
/// 8 seed + 4x4 counts + 3x4 floats + 32 reserved = 80 bytes.
pub const HEADER_SIZE: usize = 80;
const _: () = assert!(8 + 4 + 8 + 4 * 4 + 3 * 4 + 32 == HEADER_SIZE);

#[derive(Debug, Clone, PartialEq)]
pub struct Header {
    pub seed: u64,
    pub episode_count: u32,
    pub frames_per_episode: u32,
    pub feature_count: u32,
    pub target_count: u32,
    pub frame_rate_hz: f32,
    pub dt: f32,
    pub duration_s: f32,
}

impl Header {
    fn write_to<W: Write>(&self, w: &mut W) -> Result<()> {
        let mut b = [0u8; HEADER_SIZE];
        b[0..8].copy_from_slice(MAGIC);
        b[8..12].copy_from_slice(&VERSION.to_le_bytes());
        b[12..20].copy_from_slice(&self.seed.to_le_bytes());
        b[20..24].copy_from_slice(&self.episode_count.to_le_bytes());
        b[24..28].copy_from_slice(&self.frames_per_episode.to_le_bytes());
        b[28..32].copy_from_slice(&self.feature_count.to_le_bytes());
        b[32..36].copy_from_slice(&self.target_count.to_le_bytes());
        b[36..40].copy_from_slice(&self.frame_rate_hz.to_le_bytes());
        b[40..44].copy_from_slice(&self.dt.to_le_bytes());
        b[44..48].copy_from_slice(&self.duration_s.to_le_bytes());
        // bytes 48..HEADER_SIZE stay zero (reserved)
        w.write_all(&b)?;
        Ok(())
    }

    fn read_from<R: Read>(r: &mut R) -> Result<Self> {
        let mut b = [0u8; HEADER_SIZE];
        r.read_exact(&mut b)?;
        if &b[0..8] != MAGIC {
            return Err(SimError::InvalidValidation("bad magic".into()));
        }
        if u32::from_le_bytes(b[8..12].try_into().unwrap()) != VERSION {
            return Err(SimError::InvalidValidation("bad version".into()));
        }
        let seed = u64::from_le_bytes(b[12..20].try_into().unwrap());
        let mut vals = [0u32; 4];
        for (i, v) in vals.iter_mut().enumerate() {
            *v = u32::from_le_bytes(b[20 + i * 4..24 + i * 4].try_into().unwrap());
        }
        let mut floats = [0f32; 3];
        for (i, f) in floats.iter_mut().enumerate() {
            *f = f32::from_le_bytes(b[36 + i * 4..40 + i * 4].try_into().unwrap());
        }
        if vals[3] as usize != TARGET_COUNT {
            return Err(SimError::InvalidValidation(
                "unsupported target count".into(),
            ));
        }
        Ok(Self {
            seed,
            episode_count: vals[0],
            frames_per_episode: vals[1],
            feature_count: vals[2],
            target_count: vals[3],
            frame_rate_hz: floats[0],
            dt: floats[1],
            duration_s: floats[2],
        })
    }

    pub fn header_size() -> usize {
        HEADER_SIZE
    }
}

#[derive(Debug, Clone)]
pub struct ValidationEpisode {
    pub index: u64,
    /// mass, thrust_scale, pitch, yaw, gust_start, gust_duration, gust_amp, v_10, shear_delta
    pub meta: [f32; 9],
    pub features: Array2<f32>,
    pub targets: Array2<f32>,
}

#[derive(Debug, Clone)]
pub struct ValidationSet {
    pub header: Header,
    pub episodes: Vec<ValidationEpisode>,
}

fn meta_of(e: &Episode) -> [f32; 9] {
    [
        e.meta.mass_kg as f32,
        e.meta.thrust_scale as f32,
        e.meta.initial_pitch_deg as f32,
        e.meta.initial_yaw_deg as f32,
        e.meta.gust_start_s.unwrap_or(-1.0) as f32,
        e.meta.gust_duration_s as f32,
        e.meta.gust_amplitude as f32,
        e.meta.v_10 as f32,
        e.meta.shear_delta as f32,
    ]
}

pub fn write_validation(path: &Path, cfg: &Config, episodes: &[Episode]) -> Result<()> {
    let header = Header {
        seed: cfg.run.seed,
        episode_count: episodes.len() as u32,
        frames_per_episode: cfg.frames_per_episode() as u32,
        feature_count: cfg.feature_count() as u32,
        target_count: TARGET_COUNT as u32,
        frame_rate_hz: cfg.run.frame_rate_hz as f32,
        dt: cfg.run.dt as f32,
        duration_s: cfg.run.duration_s as f32,
    };
    let mut w = BufWriter::new(File::create(path)?);
    header.write_to(&mut w)?;
    for e in episodes {
        w.write_all(&e.meta.index.to_le_bytes())?;
        let meta = meta_of(e);
        for v in meta {
            w.write_all(&v.to_le_bytes())?;
        }
        for v in &e.features {
            w.write_all(&v.to_le_bytes())?;
        }
        for t in &e.targets {
            for v in [
                t.g_theta[0],
                t.g_theta[1],
                t.confidence,
                if t.ai_healthy { 1.0 } else { 0.0 },
            ] {
                w.write_all(&v.to_le_bytes())?;
            }
        }
    }
    w.flush()?;
    Ok(())
}

pub fn read_validation(path: &Path) -> Result<ValidationSet> {
    let mut r = BufReader::new(File::open(path)?);
    let header = Header::read_from(&mut r)?;
    let mut episodes = Vec::with_capacity(header.episode_count as usize);
    for _ in 0..header.episode_count {
        let mut index = [0u8; 8];
        r.read_exact(&mut index)?;
        let mut meta = [0f32; 9];
        for v in meta.iter_mut() {
            let mut b = [0u8; 4];
            r.read_exact(&mut b)?;
            *v = f32::from_le_bytes(b);
        }
        let (nf, nc, nt) = (
            header.frames_per_episode as usize,
            header.feature_count as usize,
            header.target_count as usize,
        );
        let mut features = Array2::<f32>::zeros((nf, nc));
        for v in features.iter_mut() {
            let mut b = [0u8; 4];
            r.read_exact(&mut b)?;
            *v = f32::from_le_bytes(b);
        }
        let mut targets = Array2::<f32>::zeros((nf, nt));
        for v in targets.iter_mut() {
            let mut b = [0u8; 4];
            r.read_exact(&mut b)?;
            *v = f32::from_le_bytes(b);
        }
        episodes.push(ValidationEpisode {
            index: u64::from_le_bytes(index),
            meta,
            features,
            targets,
        });
    }
    Ok(ValidationSet { header, episodes })
}
