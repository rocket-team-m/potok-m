//! Compact binary validation set: custom header followed by f32 arrays.
//!
//! Layout (little-endian), see sim/README.md for the full description:
//!   header, then for each episode: `meta_count` metadata f32,
//!   frames x features f32 (row-major), frames x 4 targets f32 (row-major).
//! Version 1 stores the original 9 metadata; version 2 stores those 9 plus
//! 6 conditioning numbers, a scenario id and the train/withheld split flag
//! (17 total). A v2 file may carry 22 extra truth floats (39 total) when
//! written by `write_validation_with_truth`; `meta_count` in the header is
//! the authority, so v1/v2 readers of either layout keep working.

use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
};

use ndarray::Array2;

use crate::{
    config::{Conditioning, Config},
    episode::Episode,
    error::{Result, SimError},
    scenario::Split,
};

pub const MAGIC: &[u8; 8] = b"POTOKSIM";
pub const VERSION: u32 = 2;
pub const VERSION_V1: u32 = 1;
pub const META_V1: usize = 9;
pub const META_V2: usize = 17;
/// 22 extra floats appended to v2 metadata: wind(3), gust start+amplitude(2),
/// shear altitude/delta/thickness(3), 6-DoF state(14).
pub const TRUTH_COUNT: usize = 22;
pub const META_V2_TRUTH: usize = META_V2 + TRUTH_COUNT;
pub const META_COUNT: usize = META_V2;
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
    /// Metadata f32 per episode: 9 in v1, 17 in classic v2, 39 in a
    /// truth-extended v2. Stored in bytes 48..52 of the header (zero for v1).
    pub meta_count: u32,
}

impl Header {
    fn write_to<W: Write>(&self, w: &mut W, version: u32) -> Result<()> {
        let mut b = [0u8; HEADER_SIZE];
        b[0..8].copy_from_slice(MAGIC);
        b[8..12].copy_from_slice(&version.to_le_bytes());
        b[12..20].copy_from_slice(&self.seed.to_le_bytes());
        b[20..24].copy_from_slice(&self.episode_count.to_le_bytes());
        b[24..28].copy_from_slice(&self.frames_per_episode.to_le_bytes());
        b[28..32].copy_from_slice(&self.feature_count.to_le_bytes());
        b[32..36].copy_from_slice(&self.target_count.to_le_bytes());
        b[36..40].copy_from_slice(&self.frame_rate_hz.to_le_bytes());
        b[40..44].copy_from_slice(&self.dt.to_le_bytes());
        b[44..48].copy_from_slice(&self.duration_s.to_le_bytes());
        if version != VERSION_V1 {
            b[48..52].copy_from_slice(&self.meta_count.to_le_bytes());
        }
        // bytes 52..HEADER_SIZE stay zero (reserved)
        w.write_all(&b)?;
        Ok(())
    }

    fn read_from<R: Read>(r: &mut R) -> Result<VersionedHeader> {
        let mut b = [0u8; HEADER_SIZE];
        r.read_exact(&mut b)?;
        if &b[0..8] != MAGIC {
            return Err(SimError::InvalidValidation("bad magic".into()));
        }
        let version = u32::from_le_bytes(b[8..12].try_into().unwrap());
        let meta_count = match version {
            VERSION_V1 => META_V1 as u32,
            VERSION => {
                let n = u32::from_le_bytes(b[48..52].try_into().unwrap());
                if n as usize != META_V2 && n as usize != META_V2_TRUTH {
                    return Err(SimError::InvalidValidation(format!(
                        "unsupported metadata count {n}"
                    )));
                }
                n
            }
            _ => return Err(SimError::InvalidValidation("bad version".into())),
        };
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
        Ok(VersionedHeader {
            header: Header {
                seed,
                episode_count: vals[0],
                frames_per_episode: vals[1],
                feature_count: vals[2],
                target_count: vals[3],
                frame_rate_hz: floats[0],
                dt: floats[1],
                duration_s: floats[2],
                meta_count,
            },
            version,
        })
    }

    pub fn header_size() -> usize {
        HEADER_SIZE
    }
}

struct VersionedHeader {
    header: Header,
    version: u32,
}

#[derive(Debug, Clone)]
pub struct ValidationEpisode {
    pub index: u64,
    /// mass, thrust_scale, pitch, yaw, gust_start, gust_duration, gust_amp, v_10, shear_delta
    pub meta: [f32; META_V1],
    pub conditioning: Conditioning,
    pub scenario_id: u16,
    pub split: Split,
    pub features: Array2<f32>,
    pub targets: Array2<f32>,
    /// Present only in truth-extended v2 files. Ground truth at the last
    /// 50 Hz tick of the episode.
    pub truth: Option<TruthRecord>,
}

/// Ground truth appended to v2 metadata. `wind` is the true wind vector at
/// the tick (m/s, world frame); `gust_start_s` is -1 when no gust was active;
/// `pos/vel/quat/omega/mass_kg` are the un-noised RK4 state.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct TruthRecord {
    pub wind: [f32; 3],
    pub gust_start_s: f32,
    pub gust_amplitude: f32,
    pub shear_altitude_m: f32,
    pub shear_delta: f32,
    pub shear_thickness_m: f32,
    pub pos: [f32; 3],
    pub vel: [f32; 3],
    pub quat: [f32; 4],
    pub omega: [f32; 3],
    pub mass_kg: f32,
}

impl TruthRecord {
    pub fn as_array(self) -> [f32; TRUTH_COUNT] {
        let mut a = [0.0; TRUTH_COUNT];
        a[0..3].copy_from_slice(&self.wind);
        a[3] = self.gust_start_s;
        a[4] = self.gust_amplitude;
        a[5] = self.shear_altitude_m;
        a[6] = self.shear_delta;
        a[7] = self.shear_thickness_m;
        a[8..11].copy_from_slice(&self.pos);
        a[11..14].copy_from_slice(&self.vel);
        a[14..18].copy_from_slice(&self.quat);
        a[18..21].copy_from_slice(&self.omega);
        a[21] = self.mass_kg;
        a
    }

    pub fn from_array(a: [f32; TRUTH_COUNT]) -> Self {
        Self {
            wind: [a[0], a[1], a[2]],
            gust_start_s: a[3],
            gust_amplitude: a[4],
            shear_altitude_m: a[5],
            shear_delta: a[6],
            shear_thickness_m: a[7],
            pos: [a[8], a[9], a[10]],
            vel: [a[11], a[12], a[13]],
            quat: [a[14], a[15], a[16], a[17]],
            omega: [a[18], a[19], a[20]],
            mass_kg: a[21],
        }
    }
}

#[derive(Debug, Clone)]
pub struct ValidationSet {
    pub header: Header,
    pub version: u32,
    pub episodes: Vec<ValidationEpisode>,
}

fn meta_of(e: &Episode) -> [f32; META_V1] {
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

/// Writes the classic format (version 2, 17 metadata). Use
/// `write_validation_with_version` for a legacy v1 file, or
/// `write_validation_with_truth` for the truth-extended v2 layout.
pub fn write_validation(path: &Path, cfg: &Config, episodes: &[Episode]) -> Result<()> {
    write_validation_impl(path, cfg, episodes, VERSION, cfg.output.truth_metadata)
}

pub fn write_validation_with_version(
    path: &Path,
    cfg: &Config,
    episodes: &[Episode],
    version: u32,
) -> Result<()> {
    write_validation_impl(path, cfg, episodes, version, false)
}

/// Writes version 2 with 39 metadata floats per episode: the classic 17 plus
/// the `TruthRecord` captured at the episode's last frame tick.
pub fn write_validation_with_truth(
    path: &Path,
    cfg: &Config,
    episodes: &[Episode],
) -> Result<()> {
    write_validation_impl(path, cfg, episodes, VERSION, true)
}

fn write_validation_impl(
    path: &Path,
    cfg: &Config,
    episodes: &[Episode],
    version: u32,
    truth: bool,
) -> Result<()> {
    if version != VERSION && version != VERSION_V1 {
        return Err(SimError::InvalidValidation(format!(
            "unsupported validation version {version}"
        )));
    }
    if truth && episodes.iter().any(|e| e.truth.is_none()) {
        return Err(SimError::InvalidValidation(
            "truth-extended v2 needs a TruthRecord on every episode".into(),
        ));
    }
    let header = Header {
        seed: cfg.run.seed,
        episode_count: episodes.len() as u32,
        frames_per_episode: cfg.frames_per_episode() as u32,
        feature_count: cfg.feature_count() as u32,
        target_count: TARGET_COUNT as u32,
        frame_rate_hz: cfg.run.frame_rate_hz as f32,
        dt: cfg.run.dt as f32,
        duration_s: cfg.run.duration_s as f32,
        meta_count: if version == VERSION_V1 {
            META_V1 as u32
        } else if truth {
            META_V2_TRUTH as u32
        } else {
            META_V2 as u32
        },
    };
    let mut w = BufWriter::new(File::create(path)?);
    header.write_to(&mut w, version)?;
    for e in episodes {
        w.write_all(&e.meta.index.to_le_bytes())?;
        let mut meta = Vec::with_capacity(header.meta_count as usize);
        meta.extend_from_slice(&meta_of(e));
        if version == VERSION {
            meta.extend_from_slice(&e.meta.conditioning.as_array());
            meta.push(e.meta.scenario_id as f32);
            meta.push(e.meta.split.as_u8() as f32);
            if truth {
                meta.extend_from_slice(&e.truth.unwrap().as_array());
            }
        }
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
    let vh = Header::read_from(&mut r)?;
    let legacy_default = Config::default();
    let mut episodes = Vec::with_capacity(vh.header.episode_count as usize);
    for _ in 0..vh.header.episode_count {
        let mut index = [0u8; 8];
        r.read_exact(&mut index)?;
        let mut meta = [0f32; META_V1];
        for v in meta.iter_mut() {
            let mut b = [0u8; 4];
            r.read_exact(&mut b)?;
            *v = f32::from_le_bytes(b);
        }
        let (conditioning, scenario_id, split) = if vh.version == VERSION {
            let mut c = [0f32; Conditioning::COUNT];
            for v in c.iter_mut() {
                let mut b = [0u8; 4];
                r.read_exact(&mut b)?;
                *v = f32::from_le_bytes(b);
            }
            let mut b = [0u8; 4];
            r.read_exact(&mut b)?;
            let id = f32::from_le_bytes(b);
            if !id.is_finite() || id.fract() != 0.0 || id < 0.0 || id > u16::MAX as f32 {
                return Err(SimError::InvalidValidation("bad scenario id".into()));
            }
            r.read_exact(&mut b)?;
            let flag = f32::from_le_bytes(b);
            if !flag.is_finite() || flag.fract() != 0.0 {
                return Err(SimError::InvalidValidation("bad split flag".into()));
            }
            let split = Split::from_u8(flag as u8)
                .ok_or_else(|| SimError::InvalidValidation("bad split flag".into()))?;
            (Conditioning::from_array(c), id as u16, split)
        } else {
            // Legacy files were all nominal Potok-M training episodes.
            let default = &legacy_default;
            let c = Conditioning {
                mass_kg: meta[0],
                area_m2: default.aero.area_m2 as f32,
                cd0: default.aero.cd0 as f32,
                static_margin_cal: (default.aero.static_margin_m
                    / (2.0 * (default.aero.area_m2 / std::f64::consts::PI).sqrt()))
                    as f32,
                thrust_peak_n: default
                    .aero
                    .thrust_profile
                    .iter()
                    .map(|p| p.1)
                    .fold(0.0_f64, f64::max) as f32,
                gyro_noise: default.sensor.gyro_noise as f32,
            };
            (c, 0, Split::Train)
        };
        let truth = if vh.version == VERSION
            && vh.header.meta_count as usize == META_V2_TRUTH
        {
            let mut a = [0f32; TRUTH_COUNT];
            for v in a.iter_mut() {
                let mut b = [0u8; 4];
                r.read_exact(&mut b)?;
                *v = f32::from_le_bytes(b);
            }
            Some(TruthRecord::from_array(a))
        } else {
            None
        };
        let (nf, nc, nt) = (
            vh.header.frames_per_episode as usize,
            vh.header.feature_count as usize,
            vh.header.target_count as usize,
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
            conditioning,
            scenario_id,
            split,
            features,
            targets,
            truth,
        });
    }
    Ok(ValidationSet {
        header: vh.header,
        version: vh.version,
        episodes,
    })
}
