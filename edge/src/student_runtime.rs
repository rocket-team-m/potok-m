//! Stateful student model: v2 `model_kind = 2` file reader and per-frame runtime.
//!
//! The file is written by `ai/src/potok_ai/export.py::export_student` (read
//! only). Canonical v2 layout, all little-endian, f32 floats, SHA-256 over
//! every preceding byte:
//!
//! | offset | type | field |
//! |---:|---|---|
//! | 0 | 8 bytes | magic `POTOKM1\0` |
//! | 8 | u32 | version = 2 |
//! | 12 | u32 | `model_kind` = 2 (teacher v2 has no such field) |
//! | 16 | u32 | `in_features` (10 = 8 sensors + phase + flags) |
//! | 20 | u32 | width |
//! | 24 | u32 | `out_features` (2 corrections) |
//! | 28 | u32 | TSCT rank |
//! | 32 | u32 | number of blocks |
//! | 36 | u32 | loop iterations per tick (1..=3) |
//! | 40 | f32 | geometric decay |
//! | 44 | f32 | g_max |
//! | 48 | u32 | flags |
//! | 52 | u32 | layer record count |
//!
//! Records use the canonical v2 encoding: `kind = 1` TSCT (in/out/rank,
//! alpha_u/alpha_v, `in*rank` ternary codes, `rank` f32 scales, `out*rank`
//! ternary codes) and `kind = 2` fp32 tensor (ndim, i32 dims, row-major data).
//! Required names: `blocks.{b}.proj` (TSCT), `blocks.{b}.rec_h.{weight,bias}`,
//! `blocks.{b}.rec_x.{weight,bias}`, `stem.{weight,bias}`, `norm.{weight,bias}`,
//! `alloc_logits`, `head_corr.{weight,bias}`, `head_conf.{weight,bias}`,
//! `g_max`. Ternary factors are dequantized once; the hot loop never touches
//! a code byte and allocates nothing.
//!
//! Forward mirrors `ai/src/potok_ai/student.py::StatefulTCN.step`: a stem
//! projects the frame to width; for each loop iteration all blocks run in
//! sequence with weight sharing (`candidate = tanh(rec_x(x) + rec_h(h))`,
//! `h = 0.8*h + 0.2*candidate`, then the TSCT projection of the new `h`
//! becomes the next block's input), and iteration outputs are accumulated with
//! `softmax(alloc_logits) * decay^loop`. The final feature is
//! `LayerNorm(total + source)` where `source` is the last block's output;
//! a `tanh` head scaled by `tanh(|g_max|)` and a sigmoid confidence finish
//! the tick. These three points deliberately supersede the older
//! SPEC_STUDENT.md text; the canonical Python implementation is the authority.
//!
//! `in_features` comes from the header: **10** for the legacy frame (8 sensors
//! + phase + flags) or **20** for the smart frame (14 physics features followed
//! by the 6 vehicle-conditioning numbers). `step` accepts any frame up to that
//! width and zero-pads the tail, matching the Python `_as_frame` helper.

use crate::simd::dot;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::path::Path;

pub const MAGIC: [u8; 8] = *b"POTOKM1\0";
pub const VERSION: u32 = 2;
pub const MODEL_KIND_STUDENT: u32 = 2;
pub const HEADER_BYTES: usize = 56;
pub const TSCT_LAYER: u8 = 1;
pub const FP32_LAYER: u8 = 2;
pub const WINDOW: usize = 50;
pub const SENSOR_FEATURES: usize = 8;
pub const PHASE_INDEX: usize = 8;
pub const FLAGS_INDEX: usize = 9;
pub const LEGACY_STUDENT_FEATURES: usize = 10;
pub const PHYSICS_FEATURES: usize = 14;
pub const COND_FEATURES: usize = 6;
pub const SMART_FRAME_FEATURES: usize = PHYSICS_FEATURES + COND_FEATURES;
const CHECKSUM_BYTES: usize = 32;
const LN_EPS: f32 = 1e-5;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StudentError {
    Truncated,
    BadMagic,
    UnsupportedVersion(u32),
    BadModelKind(u32),
    ChecksumMismatch,
    InvalidField(&'static str),
    UnknownKind(u8),
    MissingLayer(String),
    ShapeMismatch(&'static str),
    InvalidTernaryCode(i8),
    Utf8,
}

impl fmt::Display for StudentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => write!(f, "student model file truncated"),
            Self::BadMagic => write!(f, "not a POTOKM1 model file"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported export version {v}"),
            Self::BadModelKind(k) => write!(f, "model_kind {k} is not the stateful student (2)"),
            Self::ChecksumMismatch => write!(f, "student SHA256 checksum mismatch"),
            Self::InvalidField(s) => write!(f, "invalid student field: {s}"),
            Self::UnknownKind(k) => write!(f, "unknown student layer kind {k}"),
            Self::MissingLayer(s) => write!(f, "missing student layer {s}"),
            Self::ShapeMismatch(s) => write!(f, "student shape mismatch: {s}"),
            Self::InvalidTernaryCode(c) => write!(f, "invalid ternary code {c}"),
            Self::Utf8 => write!(f, "student layer name is not UTF-8"),
        }
    }
}

impl std::error::Error for StudentError {}

/// Build a 20-float smart frame: 14 physics features then 6 conditioning
/// numbers, in the order written by `potok_sim::Conditioning::as_array`.
pub fn smart_frame(
    physics: &[f32; PHYSICS_FEATURES],
    conditioning: &[f32; COND_FEATURES],
) -> [f32; SMART_FRAME_FEATURES] {
    let mut frame = [0.0f32; SMART_FRAME_FEATURES];
    frame[..PHYSICS_FEATURES].copy_from_slice(physics);
    frame[PHYSICS_FEATURES..].copy_from_slice(conditioning);
    frame
}

/// The `model_kind` field. A canonical windowed-teacher v2 file has no such
/// field: offset 12 holds `in_features = 8` there. Anything other than 2 is
/// therefore "not the student" and is routed to the existing teacher reader.
pub fn model_kind(bytes: &[u8]) -> Result<u32, StudentError> {
    if bytes.len() < 16 {
        return Err(StudentError::Truncated);
    }
    if &bytes[..8] != MAGIC {
        return Err(StudentError::BadMagic);
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    if version != VERSION {
        return Err(StudentError::UnsupportedVersion(version));
    }
    Ok(u32::from_le_bytes(bytes[12..16].try_into().unwrap()))
}

pub fn is_student_file(bytes: &[u8]) -> bool {
    matches!(model_kind(bytes), Ok(MODEL_KIND_STUDENT))
}

struct Reader<'a> {
    b: &'a [u8],
    off: usize,
}

impl<'a> Reader<'a> {
    fn new(b: &'a [u8]) -> Self {
        Self { b, off: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], StudentError> {
        let end = self.off.checked_add(n).ok_or(StudentError::Truncated)?;
        let s = self.b.get(self.off..end).ok_or(StudentError::Truncated)?;
        self.off = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, StudentError> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, StudentError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn i32(&mut self) -> Result<i32, StudentError> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32, StudentError> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f32s(&mut self, n: usize) -> Result<Vec<f32>, StudentError> {
        let b = self.take(n.checked_mul(4).ok_or(StudentError::Truncated)?)?;
        Ok(b.chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }
    fn codes(&mut self, n: usize) -> Result<Vec<i8>, StudentError> {
        let b = self.take(n)?;
        let v: Vec<i8> = b.iter().map(|&x| x as i8).collect();
        if let Some(&bad) = v.iter().find(|c| !matches!(c, -1 | 0 | 1)) {
            return Err(StudentError::InvalidTernaryCode(bad));
        }
        Ok(v)
    }
}

enum LayerData {
    Tsct {
        in_dim: usize,
        out_dim: usize,
        rank: usize,
        alpha_u: f32,
        alpha_v: f32,
        u: Vec<i8>,
        s: Vec<f32>,
        v: Vec<i8>,
    },
    F32 {
        shape: Vec<usize>,
        data: Vec<f32>,
    },
}

#[derive(Debug)]
struct StudentBlock {
    rec_h_w: Vec<f32>,
    rec_h_b: Vec<f32>,
    rec_x_w: Vec<f32>,
    rec_x_b: Vec<f32>,
    /// `[width, width]` dequantized TSCT map.
    proj_w: Vec<f32>,
}

/// A dequantized student model.
#[derive(Debug)]
pub struct StudentModel {
    pub in_features: usize,
    pub width: usize,
    pub out_features: usize,
    pub rank: usize,
    pub n_blocks: usize,
    pub n_loops: usize,
    pub decay: f32,
    pub g_max: f32,
    pub flags: u32,
    /// `tanh(|g_max|)`, the head's correction bound in the Python model.
    pub correction_bound: f32,
    blocks: Vec<StudentBlock>,
    stem_w: Vec<f32>,
    stem_b: Vec<f32>,
    norm_w: Vec<f32>,
    norm_b: Vec<f32>,
    alloc: Vec<f32>,
    head_corr_w: Vec<f32>,
    head_corr_b: Vec<f32>,
    head_conf_w: Vec<f32>,
    head_conf_b: Vec<f32>,
}

impl StudentModel {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, StudentError> {
        Self::from_bytes(&fs::read(path).map_err(|_| StudentError::Truncated)?)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, StudentError> {
        if bytes.len() < HEADER_BYTES + CHECKSUM_BYTES {
            return Err(StudentError::Truncated);
        }
        let split = bytes.len() - CHECKSUM_BYTES;
        let body = &bytes[..split];
        if Sha256::digest(body).as_slice() != &bytes[split..] {
            return Err(StudentError::ChecksumMismatch);
        }
        let mut r = Reader::new(body);
        if r.take(8)? != MAGIC {
            return Err(StudentError::BadMagic);
        }
        let version = r.u32()?;
        if version != VERSION {
            return Err(StudentError::UnsupportedVersion(version));
        }
        let kind = r.u32()?;
        if kind != MODEL_KIND_STUDENT {
            return Err(StudentError::BadModelKind(kind));
        }
        let in_features = r.u32()? as usize;
        let width = r.u32()? as usize;
        let out_features = r.u32()? as usize;
        let rank = r.u32()? as usize;
        let n_blocks = r.u32()? as usize;
        let n_loops = r.u32()? as usize;
        let decay = r.f32()?;
        let g_max = r.f32()?;
        let flags = r.u32()?;
        let n_records = r.u32()? as usize;

        if in_features < SENSOR_FEATURES
            || width == 0
            || out_features == 0
            || rank == 0
            || n_blocks == 0
            || !(1..=3).contains(&n_loops)
            || !decay.is_finite()
            || !(0.0..=1.0).contains(&decay)
            || !g_max.is_finite()
            || n_records < 6 * n_blocks + 8
        {
            return Err(StudentError::InvalidField("header counts"));
        }

        let mut layers: HashMap<String, LayerData> = HashMap::with_capacity(n_records);
        for _ in 0..n_records {
            let kind = r.u8()?;
            let name_len = r.u32()? as usize;
            let name = std::str::from_utf8(r.take(name_len)?)
                .map_err(|_| StudentError::Utf8)?
                .to_owned();
            let layer = match kind {
                TSCT_LAYER => {
                    let d_in = r.u32()? as usize;
                    let d_out = r.u32()? as usize;
                    let d_rank = r.u32()? as usize;
                    let alpha_u = r.f32()?;
                    let alpha_v = r.f32()?;
                    let u = r.codes(d_in.checked_mul(d_rank).ok_or(StudentError::Truncated)?)?;
                    let s = r.f32s(d_rank)?;
                    let v = r.codes(d_out.checked_mul(d_rank).ok_or(StudentError::Truncated)?)?;
                    LayerData::Tsct {
                        in_dim: d_in,
                        out_dim: d_out,
                        rank: d_rank,
                        alpha_u,
                        alpha_v,
                        u,
                        s,
                        v,
                    }
                }
                FP32_LAYER => {
                    let ndim = r.u32()? as usize;
                    let mut shape = Vec::with_capacity(ndim);
                    let mut n = 1usize;
                    for _ in 0..ndim {
                        let d = r.i32()?;
                        if d <= 0 {
                            return Err(StudentError::ShapeMismatch("non-positive dimension"));
                        }
                        n = n.checked_mul(d as usize).ok_or(StudentError::Truncated)?;
                        shape.push(d as usize);
                    }
                    LayerData::F32 {
                        shape,
                        data: r.f32s(n)?,
                    }
                }
                other => return Err(StudentError::UnknownKind(other)),
            };
            if layers.insert(name, layer).is_some() {
                return Err(StudentError::InvalidField("duplicate layer name"));
            }
        }
        if r.off != body.len() {
            return Err(StudentError::InvalidField("trailing bytes"));
        }

        let tensor = |name: &str, want: &[usize]| -> Result<Vec<f32>, StudentError> {
            match layers.get(name) {
                Some(LayerData::F32 { shape, data }) if shape.as_slice() == want => {
                    Ok(data.clone())
                }
                Some(LayerData::F32 { .. }) => Err(StudentError::ShapeMismatch("tensor shape")),
                Some(_) => Err(StudentError::InvalidField("tensor layer kind")),
                None => Err(StudentError::MissingLayer(name.to_owned())),
            }
        };
        let at_least = |name: &str, min: usize| -> Result<Vec<f32>, StudentError> {
            match layers.get(name) {
                Some(LayerData::F32 { data, .. }) if data.len() >= min => Ok(data[..min].to_vec()),
                Some(_) => Err(StudentError::ShapeMismatch("tensor length")),
                None => Err(StudentError::MissingLayer(name.to_owned())),
            }
        };

        let mut blocks = Vec::with_capacity(n_blocks);
        for b in 0..n_blocks {
            let prefix = format!("blocks.{b}");
            let rec_h_w = tensor(&format!("{prefix}.rec_h.weight"), &[width, width])?;
            let rec_h_b = tensor(&format!("{prefix}.rec_h.bias"), &[width])?;
            let rec_x_w = tensor(&format!("{prefix}.rec_x.weight"), &[width, width])?;
            let rec_x_b = tensor(&format!("{prefix}.rec_x.bias"), &[width])?;
            let name = format!("{prefix}.proj");
            let (d_in, d_out, d_rank, alpha_u, alpha_v, uc, sc, vc) = match layers.get(&name) {
                Some(LayerData::Tsct {
                    in_dim,
                    out_dim,
                    rank,
                    alpha_u,
                    alpha_v,
                    u,
                    s,
                    v,
                }) => (*in_dim, *out_dim, *rank, *alpha_u, *alpha_v, u, s, v),
                Some(_) => return Err(StudentError::InvalidField("proj kind")),
                None => return Err(StudentError::MissingLayer(name)),
            };
            if d_in != width || d_out != width || d_rank != rank || sc.len() != rank {
                return Err(StudentError::ShapeMismatch("TSCT factors"));
            }
            // Same square-map convention as the canonical v2 teacher reader:
            // W[out, in] = U[out,k]*alpha_u*s[k]*V[in,k]*alpha_v.
            let mut proj_w = vec![0.0f32; width * width];
            for k in 0..rank {
                for out in 0..width {
                    let uu = uc[out * rank + k] as f32 * alpha_u;
                    for input in 0..width {
                        let vv = vc[input * rank + k] as f32 * alpha_v;
                        proj_w[out * width + input] += (uu * sc[k]) * vv;
                    }
                }
            }
            blocks.push(StudentBlock {
                rec_h_w,
                rec_h_b,
                rec_x_w,
                rec_x_b,
                proj_w,
            });
        }

        let stem_w = tensor("stem.weight", &[width, in_features])?;
        let stem_b = tensor("stem.bias", &[width])?;
        let norm_w = tensor("norm.weight", &[width])?;
        let norm_b = tensor("norm.bias", &[width])?;
        let alloc = at_least("alloc_logits", n_loops)?;
        let head_corr_w = tensor("head_corr.weight", &[out_features, width])?;
        let head_corr_b = tensor("head_corr.bias", &[out_features])?;
        let head_conf_w = tensor("head_conf.weight", &[1, width])?;
        let head_conf_b = tensor("head_conf.bias", &[1])?;
        let g_max_tensor = tensor("g_max", &[1])?[0];
        let correction_bound = g_max_tensor.abs().tanh();

        Ok(Self {
            in_features,
            width,
            out_features,
            rank,
            n_blocks,
            n_loops,
            decay,
            g_max,
            flags,
            correction_bound,
            blocks,
            stem_w,
            stem_b,
            norm_w,
            norm_b,
            alloc,
            head_corr_w,
            head_corr_b,
            head_conf_w,
            head_conf_b,
        })
    }

    pub fn memory_bytes(&self) -> usize {
        let fixed = self.stem_w.len()
            + self.stem_b.len()
            + self.norm_w.len()
            + self.norm_b.len()
            + self.alloc.len()
            + self.head_corr_w.len()
            + self.head_corr_b.len()
            + self.head_conf_w.len()
            + self.head_conf_b.len();
        let blocks: usize = self
            .blocks
            .iter()
            .map(|b| {
                b.rec_h_w.len()
                    + b.rec_h_b.len()
                    + b.rec_x_w.len()
                    + b.rec_x_b.len()
                    + b.proj_w.len()
            })
            .sum();
        (fixed + blocks) * std::mem::size_of::<f32>()
    }
}

/// LayerNorm with the biased variance and epsilon used by PyTorch's
/// `nn.LayerNorm` (and by the canonical v2 teacher reader). The export's
/// weights were trained against this normalization.
fn layer_norm(src: &[f32], gain: &[f32], bias: &[f32], dst: &mut [f32]) {
    let n = src.len();
    let mean = src.iter().sum::<f32>() / n as f32;
    let var = src.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let inv = 1.0 / (var + LN_EPS).sqrt();
    for i in 0..n {
        dst[i] = (src[i] - mean) * inv * gain[i] + bias[i];
    }
}

/// The per-frame state machine: one hidden vector per block, all buffers
/// preallocated. `step` does no heap work and no I/O.
pub struct StudentRuntime {
    model: StudentModel,
    h: Vec<f32>,
    cand: Vec<f32>,
    next_source: Vec<f32>,
    source: Vec<f32>,
    total: Vec<f32>,
    z: Vec<f32>,
    xpad: Vec<f32>,
    alloc: Vec<f32>,
}

impl StudentRuntime {
    pub fn new(model: StudentModel) -> Self {
        let n = model.n_blocks * model.width;
        let mut alloc = vec![0.0f32; model.n_loops];
        let mut max = f32::NEG_INFINITY;
        for l in 0..model.n_loops {
            alloc[l] = model.alloc[l];
            max = max.max(alloc[l]);
        }
        let mut sum = 0.0f32;
        for a in alloc.iter_mut() {
            *a = (*a - max).exp();
            sum += *a;
        }
        for a in alloc.iter_mut() {
            *a /= sum;
        }
        Self {
            h: vec![0.0; n],
            cand: vec![0.0; model.width],
            next_source: vec![0.0; model.width],
            source: vec![0.0; model.width],
            total: vec![0.0; model.width],
            z: vec![0.0; model.width],
            xpad: vec![0.0; model.in_features],
            alloc,
            model,
        }
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, StudentError> {
        Ok(Self::new(StudentModel::from_file(path)?))
    }

    pub fn model(&self) -> &StudentModel {
        &self.model
    }

    pub fn reset_state(&mut self) {
        self.h.fill(0.0);
    }

    pub fn state(&self) -> &[f32] {
        &self.h
    }

    /// Rebuild state by replaying a window of at most [`WINDOW`] frames, each
    /// `in_features` floats, oldest first.
    pub fn warmup(&mut self, window: &[f32]) {
        let f = self.model.in_features;
        assert_eq!(window.len() % f, 0, "warmup window must hold whole frames");
        assert!(
            window.len() <= WINDOW * f,
            "warmup window is at most 50 frames"
        );
        self.reset_state();
        for frame in window.chunks_exact(f) {
            self.step(frame);
        }
    }

    /// One tick. `frame` holds at most `in_features` values; missing trailing
    /// values are zero-filled and the flags slot defaults to "valid", exactly
    /// as the Python `_as_frame` helper does. Returns `[g0, g1, confidence]`,
    /// with both corrections in `[-correction_bound, correction_bound]`.
    pub fn step(&mut self, frame: &[f32]) -> [f32; 3] {
        assert!(frame.len() <= self.model.in_features, "frame too wide");
        self.xpad.fill(0.0);
        self.xpad[..frame.len()].copy_from_slice(frame);
        if frame.len() <= FLAGS_INDEX && FLAGS_INDEX < self.model.in_features {
            self.xpad[FLAGS_INDEX] = 1.0;
        }

        let w = self.model.width;
        for o in 0..w {
            let row =
                &self.model.stem_w[o * self.model.in_features..(o + 1) * self.model.in_features];
            self.source[o] = dot(row, &self.xpad) + self.model.stem_b[o];
        }
        self.total.fill(0.0);

        for iteration in 0..self.model.n_loops {
            for (b, block) in self.model.blocks.iter().enumerate() {
                let h_prev = &self.h[b * w..(b + 1) * w];
                let src = &self.source;
                for o in 0..w {
                    let rh = &block.rec_h_w[o * w..(o + 1) * w];
                    let rx = &block.rec_x_w[o * w..(o + 1) * w];
                    let acc = dot(rx, src) + dot(rh, h_prev) + block.rec_x_b[o] + block.rec_h_b[o];
                    // Canonical Python: leaky state update, not tanh-only.
                    self.cand[o] = 0.8 * h_prev[o] + 0.2 * acc.tanh();
                }
                for o in 0..w {
                    self.next_source[o] = dot(&block.proj_w[o * w..(o + 1) * w], &self.cand);
                }
                self.source.copy_from_slice(&self.next_source);
                self.h[b * w..(b + 1) * w].copy_from_slice(&self.cand);
            }
            let weight = self.alloc[iteration] * self.model.decay.powi(iteration as i32);
            for o in 0..w {
                self.total[o] += weight * self.source[o];
            }
        }

        // Canonical Python: norm(total + source), where `source` is the
        // last block's output of the last iteration.
        for o in 0..w {
            self.next_source[o] = self.total[o] + self.source[o];
        }
        layer_norm(
            &self.next_source,
            &self.model.norm_w,
            &self.model.norm_b,
            &mut self.z,
        );
        let mut out = [0.0f32; 3];
        for k in 0..self.model.out_features {
            out[k] = dot(&self.model.head_corr_w[k * w..(k + 1) * w], &self.z)
                + self.model.head_corr_b[k];
            out[k] = out[k].tanh() * self.model.correction_bound;
        }
        let conf = dot(&self.model.head_conf_w[..w], &self.z) + self.model.head_conf_b[0];
        out[2] = 1.0 / (1.0 + (-conf).exp());
        out
    }

    pub fn memory_bytes(&self) -> usize {
        self.model.memory_bytes()
            + (self.h.len()
                + self.cand.len()
                + self.next_source.len()
                + self.source.len()
                + self.total.len()
                + self.z.len()
                + self.xpad.len()
                + self.alloc.len())
                * std::mem::size_of::<f32>()
    }
}

// ---------------------------------------------------------------------------
// Deterministic tiny fixture for tests and the synthetic benchmark. The same
// canonical layout is emitted so the reader is exercised exactly as it is by
// the real `ai/` export. Same seed, same bytes.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct StudentFixtureConfig {
    pub in_features: usize,
    pub width: usize,
    pub out_features: usize,
    pub rank: usize,
    pub n_blocks: usize,
    pub n_loops: usize,
    pub decay: f32,
    pub g_max: f32,
}

impl Default for StudentFixtureConfig {
    fn default() -> Self {
        Self {
            in_features: 10,
            width: 64, // SPEC_STUDENT.md: two recurrent blocks of width 64
            out_features: 2,
            rank: 16,
            n_blocks: 2,
            n_loops: 3,
            decay: 0.8,
            g_max: 0.1,
        }
    }
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn f32(&mut self) -> f32 {
        (self.next() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
    fn code(&mut self) -> i8 {
        (self.next() % 3) as i8 - 1
    }
}

fn put_u32(b: &mut Vec<u8>, off: usize, v: u32) {
    b[off..off + 4].copy_from_slice(&v.to_le_bytes());
}
fn push_u32(b: &mut Vec<u8>, v: u32) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn push_name(body: &mut Vec<u8>, kind: u8, name: &str) {
    body.push(kind);
    push_u32(body, name.len() as u32);
    body.extend_from_slice(name.as_bytes());
}
fn f32_record(body: &mut Vec<u8>, name: &str, shape: &[usize], vals: &[f32]) {
    assert_eq!(shape.iter().product::<usize>(), vals.len());
    push_name(body, FP32_LAYER, name);
    push_u32(body, shape.len() as u32);
    for &d in shape {
        body.extend_from_slice(&(d as i32).to_le_bytes());
    }
    for &v in vals {
        body.extend_from_slice(&v.to_le_bytes());
    }
}
fn tsct_record(body: &mut Vec<u8>, rng: &mut Lcg, name: &str, rank: usize, w: usize) {
    push_name(body, TSCT_LAYER, name);
    push_u32(body, w as u32);
    push_u32(body, w as u32);
    push_u32(body, rank as u32);
    body.extend_from_slice(&1.0f32.to_le_bytes()); // alpha_u
    body.extend_from_slice(&1.0f32.to_le_bytes()); // alpha_v
    for _ in 0..w * rank {
        body.push(rng.code() as u8);
    }
    for _ in 0..rank {
        body.extend_from_slice(&(0.05 + (rng.next() % 20) as f32 * 0.01).to_le_bytes());
    }
    for _ in 0..w * rank {
        body.push(rng.code() as u8);
    }
}

pub fn build_student_fixture(cfg: StudentFixtureConfig, seed: u64) -> Vec<u8> {
    let mut rng = Lcg(seed);
    let n_records = 6 * cfg.n_blocks + 8;
    let mut body = vec![0u8; HEADER_BYTES];
    body[..8].copy_from_slice(&MAGIC);
    put_u32(&mut body, 8, VERSION);
    put_u32(&mut body, 12, MODEL_KIND_STUDENT);
    put_u32(&mut body, 16, cfg.in_features as u32);
    put_u32(&mut body, 20, cfg.width as u32);
    put_u32(&mut body, 24, cfg.out_features as u32);
    put_u32(&mut body, 28, cfg.rank as u32);
    put_u32(&mut body, 32, cfg.n_blocks as u32);
    put_u32(&mut body, 36, cfg.n_loops as u32);
    body[40..44].copy_from_slice(&cfg.decay.to_le_bytes());
    body[44..48].copy_from_slice(&cfg.g_max.to_le_bytes());
    put_u32(&mut body, 48, 0); // flags
    put_u32(&mut body, 52, n_records as u32);

    let w = cfg.width;
    let f = cfg.in_features;
    let o = cfg.out_features;
    for b in 0..cfg.n_blocks {
        let rh_w: Vec<f32> = (0..w * w).map(|_| rng.f32() * 0.08).collect();
        let rh_b: Vec<f32> = (0..w).map(|_| rng.f32() * 0.02).collect();
        let rx_w: Vec<f32> = (0..w * w).map(|_| rng.f32() * 0.08).collect();
        let rx_b: Vec<f32> = (0..w).map(|_| rng.f32() * 0.02).collect();
        f32_record(
            &mut body,
            &format!("blocks.{b}.rec_h.weight"),
            &[w, w],
            &rh_w,
        );
        f32_record(&mut body, &format!("blocks.{b}.rec_h.bias"), &[w], &rh_b);
        f32_record(
            &mut body,
            &format!("blocks.{b}.rec_x.weight"),
            &[w, w],
            &rx_w,
        );
        f32_record(&mut body, &format!("blocks.{b}.rec_x.bias"), &[w], &rx_b);
        tsct_record(
            &mut body,
            &mut rng,
            &format!("blocks.{b}.proj"),
            cfg.rank,
            w,
        );
    }
    let sw: Vec<f32> = (0..w * f).map(|_| rng.f32() * 0.3).collect();
    let sb: Vec<f32> = (0..w).map(|_| rng.f32() * 0.02).collect();
    let nw: Vec<f32> = (0..w)
        .map(|_| 0.9 + (rng.next() % 10) as f32 * 0.02)
        .collect();
    let nb: Vec<f32> = (0..w).map(|_| rng.f32() * 0.01).collect();
    let al: Vec<f32> = (0..3).map(|_| rng.f32() * 0.2).collect();
    let hcw: Vec<f32> = (0..o * w).map(|_| rng.f32() * 0.3).collect();
    let hcb: Vec<f32> = (0..o).map(|_| rng.f32() * 0.05).collect();
    let hfw: Vec<f32> = (0..w).map(|_| rng.f32() * 0.3).collect();
    let hfb: Vec<f32> = (0..1).map(|_| rng.f32() * 0.05).collect();
    f32_record(&mut body, "stem.weight", &[w, f], &sw);
    f32_record(&mut body, "stem.bias", &[w], &sb);
    f32_record(&mut body, "norm.weight", &[w], &nw);
    f32_record(&mut body, "norm.bias", &[w], &nb);
    f32_record(&mut body, "alloc_logits", &[3], &al);
    f32_record(&mut body, "head_corr.weight", &[o, w], &hcw);
    f32_record(&mut body, "head_corr.bias", &[o], &hcb);
    f32_record(&mut body, "head_conf.weight", &[1, w], &hfw);
    f32_record(&mut body, "head_conf.bias", &[1], &hfb);
    f32_record(&mut body, "g_max", &[1], &[cfg.g_max]);

    let digest = Sha256::digest(&body);
    body.extend_from_slice(&digest);
    body
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_roundtrips_through_reader() {
        let cfg = StudentFixtureConfig {
            width: 8,
            rank: 2,
            n_loops: 3,
            ..StudentFixtureConfig::default()
        };
        let bytes = build_student_fixture(cfg, 7);
        assert!(is_student_file(&bytes));
        let m = StudentModel::from_bytes(&bytes).unwrap();
        assert_eq!(
            (m.in_features, m.width, m.n_blocks, m.n_loops),
            (10, 8, 2, 3)
        );
        assert_eq!(m.g_max, 0.1);
        let mut rt = StudentRuntime::new(m);
        let out = rt.step(&[0.1; 10]);
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(out[0].abs() <= 0.1 && out[1].abs() <= 0.1);
        assert!((0.0..=1.0).contains(&out[2]));
    }

    #[test]
    fn checksum_refusal() {
        let mut bytes = build_student_fixture(StudentFixtureConfig::default(), 7);
        *bytes.last_mut().unwrap() ^= 1;
        assert_eq!(
            StudentModel::from_bytes(&bytes).unwrap_err(),
            StudentError::ChecksumMismatch
        );
    }
}
