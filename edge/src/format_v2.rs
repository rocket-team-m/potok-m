//! Reader and inference for the canonical Potok-M export format v2.
//!
//! Spec: `ai/FORMAT.md`. The writer is `ai/src/potok_ai/export.py`.
//! No requantization happens here: the file stores ternary codes plus their
//! scales, and those are the final inference weights. The 4-bit FWHT
//! activation quantizer and the ternary STE are training-only and absent.

use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::path::Path;

pub const MAGIC: [u8; 8] = *b"POTOKM1\0";
pub const VERSION: u32 = 2;
pub const TSCT_LAYER: u8 = 1;
pub const FP32_LAYER: u8 = 2;
const LN_EPS: f32 = 1e-5;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum V2Error {
    Truncated,
    BadMagic,
    UnsupportedVersion(u32),
    ChecksumMismatch,
    InvalidField(&'static str),
    UnknownKind(u8),
    UnknownLayer(String),
    MissingLayer(String),
    ShapeMismatch(&'static str),
    InvalidTernaryCode(i8),
    Utf8,
}

impl fmt::Display for V2Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => write!(f, "v2 model file truncated"),
            Self::BadMagic => write!(f, "not a POTOKM1 model file"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported export version {v}"),
            Self::ChecksumMismatch => write!(f, "SHA256 checksum mismatch"),
            Self::InvalidField(s) => write!(f, "invalid v2 field: {s}"),
            Self::UnknownKind(k) => write!(f, "unknown layer kind {k}"),
            Self::UnknownLayer(s) => write!(f, "unexpected layer {s}"),
            Self::MissingLayer(s) => write!(f, "missing layer {s}"),
            Self::ShapeMismatch(s) => write!(f, "shape mismatch: {s}"),
            Self::InvalidTernaryCode(c) => write!(f, "invalid ternary code {c}"),
            Self::Utf8 => write!(f, "layer name is not UTF-8"),
        }
    }
}

impl std::error::Error for V2Error {}

struct Reader<'a> {
    b: &'a [u8],
    off: usize,
}

impl<'a> Reader<'a> {
    fn new(b: &'a [u8]) -> Self {
        Self { b, off: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], V2Error> {
        let end = self.off.checked_add(n).ok_or(V2Error::Truncated)?;
        let s = self.b.get(self.off..end).ok_or(V2Error::Truncated)?;
        self.off = end;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8, V2Error> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, V2Error> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn i32(&mut self) -> Result<i32, V2Error> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn f32(&mut self) -> Result<f32, V2Error> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn f32s(&mut self, n: usize) -> Result<Vec<f32>, V2Error> {
        let bytes = self.take(n.checked_mul(4).ok_or(V2Error::Truncated)?)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }

    fn codes(&mut self, n: usize) -> Result<Vec<i8>, V2Error> {
        let bytes = self.take(n)?;
        let v: Vec<i8> = bytes.iter().map(|&b| b as i8).collect();
        if let Some(&bad) = v.iter().find(|c| !matches!(c, -1 | 0 | 1)) {
            return Err(V2Error::InvalidTernaryCode(bad));
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
struct Block {
    dw_w: Vec<f32>,
    dw_b: Vec<f32>,
    norm_w: Vec<f32>,
    norm_b: Vec<f32>,
    /// `W[out, in] = (U*alpha_u*s) @ (V*alpha_v)^T`, dense f32, dequantized once.
    pw_w: Vec<f32>,
}

/// A v2 model with all factors dequantized and reusable buffers allocated.
#[derive(Debug)]
pub struct ModelV2 {
    pub in_features: usize,
    pub width: usize,
    pub out_features: usize,
    pub rank: usize,
    pub kernel: usize,
    pub n_blocks: usize,
    pub window: usize,
    pub dilations: Vec<usize>,
    pub g_max: f32,
    pub flags: u32,
    stem_w: Vec<f32>,
    stem_b: Vec<f32>,
    blocks: Vec<Block>,
    norm_w: Vec<f32>,
    norm_b: Vec<f32>,
    head_w: Vec<f32>,
    head_b: Vec<f32>,
    x: Vec<f32>,
    /// Frozen pre-block input rows; the dilated conv must read the block's
    /// input, not a partially updated residual stream.
    block_in: Vec<f32>,
    conv: Vec<f32>,
    ln: Vec<f32>,
    head: Vec<f32>,
}

impl ModelV2 {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, V2Error> {
        let bytes = fs::read(path).map_err(|_| V2Error::Truncated)?;
        Self::from_bytes(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, V2Error> {
        if bytes.len() < MAGIC.len() + 4 + 32 {
            return Err(V2Error::Truncated);
        }
        let split = bytes.len() - 32;
        let body = &bytes[..split];
        let computed = Sha256::digest(body);
        if computed.as_slice() != &bytes[split..] {
            return Err(V2Error::ChecksumMismatch);
        }
        let mut r = Reader::new(body);
        if r.take(8)? != MAGIC {
            return Err(V2Error::BadMagic);
        }
        let version = r.u32()?;
        if version != VERSION {
            return Err(V2Error::UnsupportedVersion(version));
        }
        let in_features = r.u32()? as usize;
        let width = r.u32()? as usize;
        let out_features = r.u32()? as usize;
        let rank = r.u32()? as usize;
        let kernel = r.u32()? as usize;
        let n_blocks = r.u32()? as usize;
        let g_max = r.f32()?;
        let dilations: Vec<usize> = (0..n_blocks)
            .map(|_| r.u32().map(|d| d as usize))
            .collect::<Result<_, _>>()?;
        let flags = r.u32()?;
        let n_records = r.u32()? as usize;

        if in_features != 8
            || width == 0
            || out_features != 3
            || rank == 0
            || kernel == 0
            || n_blocks == 0
            || dilations.iter().any(|&d| d == 0)
            || n_records != 5 * n_blocks + 6
        {
            return Err(V2Error::InvalidField("header counts"));
        }
        // The edge runtime window is a fixed 8-feature frame.
        let window = 50usize;

        let mut layers: HashMap<String, LayerData> = HashMap::with_capacity(n_records);
        for _ in 0..n_records {
            let kind = r.u8()?;
            let name_len = r.u32()? as usize;
            let name_bytes = r.take(name_len)?;
            let name = std::str::from_utf8(name_bytes)
                .map_err(|_| V2Error::Utf8)?
                .to_owned();
            let layer = match kind {
                TSCT_LAYER => {
                    let d_in = r.u32()? as usize;
                    let d_out = r.u32()? as usize;
                    let d_rank = r.u32()? as usize;
                    let alpha_u = r.f32()?;
                    let alpha_v = r.f32()?;
                    let u = r.codes(d_in.checked_mul(d_rank).ok_or(V2Error::Truncated)?)?;
                    let s = r.f32s(d_rank)?;
                    let v = r.codes(d_out.checked_mul(d_rank).ok_or(V2Error::Truncated)?)?;
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
                            return Err(V2Error::ShapeMismatch("non-positive dimension"));
                        }
                        let d = d as usize;
                        n = n.checked_mul(d).ok_or(V2Error::Truncated)?;
                        shape.push(d);
                    }
                    LayerData::F32 {
                        shape,
                        data: r.f32s(n)?,
                    }
                }
                other => return Err(V2Error::UnknownKind(other)),
            };
            if layers.insert(name, layer).is_some() {
                return Err(V2Error::InvalidField("duplicate layer name"));
            }
        }
        if r.off != body.len() {
            return Err(V2Error::InvalidField("trailing bytes"));
        }

        let tensor = |name: &str, want: &[usize]| -> Result<Vec<f32>, V2Error> {
            match layers.get(name) {
                Some(LayerData::F32 { shape, data }) if shape.as_slice() == want => {
                    Ok(data.clone())
                }
                Some(LayerData::F32 { .. }) => Err(V2Error::ShapeMismatch("tensor shape")),
                Some(_) => Err(V2Error::InvalidField("tensor layer kind")),
                None => Err(V2Error::MissingLayer(name.to_owned())),
            }
        };
        let tsct = |name: &str| -> Result<&LayerData, V2Error> {
            layers
                .get(name)
                .ok_or_else(|| V2Error::MissingLayer(name.to_owned()))
        };

        let stem_w = tensor("stem.weight", &[width, in_features, 1])?;
        let stem_b = tensor("stem.bias", &[width])?;
        let mut blocks = Vec::with_capacity(n_blocks);
        for i in 0..n_blocks {
            let prefix = format!("blocks.{i}");
            let dw_w = tensor(&format!("{prefix}.dw.weight"), &[width, 1, kernel])?;
            let dw_b = tensor(&format!("{prefix}.dw.bias"), &[width])?;
            let norm_w = tensor(&format!("{prefix}.norm.weight"), &[width])?;
            let norm_b = tensor(&format!("{prefix}.norm.bias"), &[width])?;
            let name = format!("{prefix}.pw");
            let (d_in, d_out, d_rank, alpha_u, alpha_v, uc, sc, vc) = match tsct(&name)? {
                LayerData::Tsct {
                    in_dim,
                    out_dim,
                    rank: r,
                    alpha_u,
                    alpha_v,
                    u,
                    s,
                    v,
                } => (*in_dim, *out_dim, *r, *alpha_u, *alpha_v, u, s, v),
                LayerData::F32 { .. } => return Err(V2Error::InvalidField("pw kind")),
            };
            if d_in != width || d_out != width || d_rank != rank || sc.len() != rank {
                return Err(V2Error::ShapeMismatch("TSCT factors"));
            }
            // `F.linear(x, W)` computes x @ W^T with W = (U*s) @ V^T.
            // Therefore W[row=b, col=a] = (U[b,k]*s[k]) * (V[a,k]) * alpha's:
            // the U index is the LINEAR OUTPUT, the V index is the INPUT.
            // Dequantize once here; codes are used as-is, never requantized.
            let mut pw_w = vec![0.0f32; width * width];
            for k in 0..rank {
                for out in 0..width {
                    let uu = uc[out * rank + k] as f32 * alpha_u;
                    for input in 0..width {
                        let vv = vc[input * rank + k] as f32 * alpha_v;
                        pw_w[out * width + input] += (uu * sc[k]) * vv;
                    }
                }
            }
            blocks.push(Block {
                dw_w,
                dw_b,
                norm_w,
                norm_b,
                pw_w,
            });
        }
        let norm_w = tensor("norm.weight", &[width])?;
        let norm_b = tensor("norm.bias", &[width])?;
        let head_w = tensor("head.weight", &[out_features, width])?;
        let head_b = tensor("head.bias", &[out_features])?;

        Ok(Self {
            in_features,
            width,
            out_features,
            rank,
            kernel,
            n_blocks,
            window,
            dilations,
            g_max,
            flags,
            stem_w,
            stem_b,
            blocks,
            norm_w,
            norm_b,
            head_w,
            head_b,
            x: vec![0.0; window * width],
            block_in: vec![0.0; window * width],
            conv: vec![0.0; width],
            ln: vec![0.0; width],
            head: vec![0.0; width],
        })
    }

    pub fn memory_bytes(&self) -> usize {
        let weights = self.stem_w.len()
            + self.stem_b.len()
            + self.norm_w.len()
            + self.norm_b.len()
            + self.head_w.len()
            + self.head_b.len()
            + self.x.len()
            + self.block_in.len()
            + self.conv.len()
            + self.ln.len()
            + self.head.len();
        let blocks: usize = self
            .blocks
            .iter()
            .map(|b| b.dw_w.len() + b.dw_b.len() + b.norm_w.len() + b.norm_b.len() + b.pw_w.len())
            .sum();
        (weights + blocks) * std::mem::size_of::<f32>()
    }

    fn layer_norm(src: &[f32], gain: &[f32], bias: &[f32], dst: &mut [f32]) {
        let n = src.len();
        let mean = src.iter().sum::<f32>() / n as f32;
        let var = src.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
        let inv = 1.0 / (var + LN_EPS).sqrt();
        for i in 0..n {
            dst[i] = (src[i] - mean) * inv * gain[i] + bias[i];
        }
    }

    /// Raw `[g_pitch, g_yaw, confidence]`, float32, before protocol scaling.
    /// Mirrors `PotokModel.forward` in `ai/src/potok_ai/model.py` exactly:
    /// stem, dilated causal depthwise conv, LayerNorm after conv, TSCT
    /// pointwise on the normalized conv with a residual from the un-normalized
    /// input, final LayerNorm, tanh head with g_max, sigmoid confidence.
    pub fn forward(&mut self, window: &[f32]) -> [f32; 3] {
        let t = self.window;
        let c = self.width;
        let fi = self.in_features;
        let dot = crate::simd::dot;
        debug_assert!(window.len() >= t * fi);

        for ti in 0..t {
            let wrow = &window[ti * fi..(ti + 1) * fi];
            let xrow = &mut self.x[ti * c..(ti + 1) * c];
            for (ci, dst) in xrow.iter_mut().enumerate() {
                let w = &self.stem_w[ci * fi..(ci + 1) * fi];
                *dst = dot(w, wrow) + self.stem_b[ci];
            }
        }

        for (bi, block) in self.blocks.iter().enumerate() {
            let dilation = self.dilations[bi];
            let pad = (self.kernel - 1) * dilation;
            for ti in 0..t {
                // Freeze this timestep's pre-block input before any residual
                // write; earlier rows are already frozen and later ones still
                // hold the block input. The conv reads only those rows.
                self.block_in[ti * c..(ti + 1) * c].copy_from_slice(&self.x[ti * c..(ti + 1) * c]);
                // Causal dilated depthwise convolution into a per-timestep row.
                for ci in 0..c {
                    let mut acc = block.dw_b[ci];
                    let kw = &block.dw_w[ci * self.kernel..(ci + 1) * self.kernel];
                    for j in 0..self.kernel {
                        let src = ti + j * dilation;
                        if src >= pad {
                            acc += kw[j] * self.block_in[(src - pad) * c + ci];
                        }
                    }
                    self.conv[ci] = acc;
                }
                // LayerNorm AFTER the conv; TSCT consumes the normalized row.
                Self::layer_norm(&self.conv, &block.norm_w, &block.norm_b, &mut self.ln);
                for out in 0..c {
                    let w = &block.pw_w[out * c..(out + 1) * c];
                    self.x[ti * c + out] += dot(w, &self.ln);
                }
            }
        }

        let last = &self.x[(t - 1) * c..t * c];
        Self::layer_norm(last, &self.norm_w, &self.norm_b, &mut self.head);
        let mut raw = [0.0f32; 3];
        for k in 0..self.out_features {
            let w = &self.head_w[k * c..(k + 1) * c];
            raw[k] = dot(w, &self.head) + self.head_b[k];
        }
        [
            raw[0].tanh() * self.g_max,
            raw[1].tanh() * self.g_max,
            1.0 / (1.0 + (-raw[2]).exp()),
        ]
    }
}
