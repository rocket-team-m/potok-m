//! Reader and dequantized representation of the canonical Potok-M model file.
//!
//! The wire layout is documented in README.md. All integers are little-endian.
//! The writer lives in `ai/` (owned by another agent); this reader only accepts
//! the canonical v1 layout and refuses anything it cannot fully account for.

use sha2::{Digest, Sha256};
use std::fmt;
use std::fs;
use std::path::Path;

pub const MAGIC: [u8; 4] = *b"POTM";
pub const FORMAT_VERSION: u32 = 1;
pub const HEADER_BYTES: usize = 48;
pub const LAYER_DESC_BYTES: usize = 8;
pub const TENSOR_DESC_BYTES: usize = 20;
pub const CHECKSUM_BYTES: usize = 32;

/// Number of tensors per residual block, in the order recorded in README.md.
pub const TENSORS_PER_LAYER: usize = 10;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelError {
    BadMagic,
    UnsupportedVersion(u32),
    Truncated,
    ChecksumMismatch,
    InvalidTensorKind(u8),
    InvalidTernaryCode(u8),
    ShapeMismatch(&'static str),
    BadTensorIndex(usize),
    TrailingBytes,
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadMagic => write!(f, "not a POTM model file"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported model format version {v}"),
            Self::Truncated => write!(f, "model file truncated"),
            Self::ChecksumMismatch => write!(f, "SHA256 checksum mismatch: file is corrupt"),
            Self::InvalidTensorKind(k) => write!(f, "unknown tensor kind {k}"),
            Self::InvalidTernaryCode(c) => write!(f, "invalid ternary code {c} (expected 0,1,2)"),
            Self::ShapeMismatch(what) => write!(f, "shape mismatch: {what}"),
            Self::BadTensorIndex(i) => write!(f, "tensor index {i} out of range"),
            Self::TrailingBytes => write!(f, "unexpected bytes after tensor payload"),
        }
    }
}

impl std::error::Error for ModelError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TensorKind {
    F32 = 1,
    Ternary = 2,
}

/// One dequantized residual depthwise-separable + TSCT block.
#[derive(Debug)]
pub(crate) struct Layer {
    /// Depthwise causal kernel `[channels, kernel]`, dequantized to f32.
    pub(crate) dw_w: Vec<f32>,
    pub(crate) dw_b: Vec<f32>,
    pub(crate) norm1: Vec<f32>,
    pub(crate) norm2: Vec<f32>,
    /// `W1 = U1 diag(s1) V1^T`, shape `[ff_width, channels]`.
    pub(crate) up_w: Vec<f32>,
    /// `W2 = U2 diag(s2) V2^T`, shape `[channels, ff_width]`.
    pub(crate) down_w: Vec<f32>,
}

/// A fully dequantized model. All weights are dense f32 so the hot loop is
/// exactly one multiply-add per weight; no dequantization happens at runtime.
#[derive(Debug)]
pub struct Model {
    pub channels: usize,
    pub kernel: usize,
    pub rank: usize,
    pub window: usize,
    pub ff_width: usize,
    pub in_features: usize,
    pub out_features: usize,
    pub(crate) in_w: Vec<f32>,
    pub(crate) in_b: Vec<f32>,
    pub(crate) final_norm: Vec<f32>,
    pub(crate) out_w: Vec<f32>,
    pub(crate) out_b: Vec<f32>,
    layers: Vec<Layer>,
}

impl Model {
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    pub(crate) fn layers(&self) -> &[Layer] {
        &self.layers
    }

    /// Bytes held by the dequantized weights plus derived buffers.
    pub fn memory_bytes(&self) -> usize {
        let mut n = self.in_w.len()
            + self.in_b.len()
            + self.final_norm.len()
            + self.out_w.len()
            + self.out_b.len();
        for l in &self.layers {
            n += l.dw_w.len()
                + l.dw_b.len()
                + l.norm1.len()
                + l.norm2.len()
                + l.up_w.len()
                + l.down_w.len();
        }
        n * std::mem::size_of::<f32>()
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, ModelError> {
        let bytes = fs::read(path).map_err(|_| ModelError::Truncated)?;
        Self::from_bytes(&bytes)
    }

    /// Verify checksum, parse descriptors, dequantize every ternary tensor.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ModelError> {
        if bytes.len() < HEADER_BYTES + CHECKSUM_BYTES {
            return Err(ModelError::Truncated);
        }
        let split = bytes.len() - CHECKSUM_BYTES;
        let body = &bytes[..split];
        let digest = &bytes[split..];
        let mut hasher = Sha256::new();
        hasher.update(body);
        let expected = hasher.finalize();
        if expected.as_slice() != digest {
            return Err(ModelError::ChecksumMismatch);
        }
        if &body[0..4] != MAGIC {
            return Err(ModelError::BadMagic);
        }
        let version = u32::from_le_bytes(body[4..8].try_into().unwrap());
        if version != FORMAT_VERSION {
            return Err(ModelError::UnsupportedVersion(version));
        }

        let mut h = 8usize;
        let mut header_u32 = |b: &[u8]| -> u32 {
            let v = u32::from_le_bytes(b[h..h + 4].try_into().unwrap());
            h += 4;
            v
        };
        let n_layers = header_u32(body) as usize;
        let channels = header_u32(body) as usize;
        let kernel = header_u32(body) as usize;
        let rank = header_u32(body) as usize;
        let window = header_u32(body) as usize;
        let in_features = header_u32(body) as usize;
        let out_features = header_u32(body) as usize;
        let ff_width = header_u32(body) as usize;
        let n_tensors = header_u32(body) as usize;
        let reserved = header_u32(body);
        debug_assert_eq!(h, HEADER_BYTES);
        if reserved != 0 {
            return Err(ModelError::ShapeMismatch("header reserved field"));
        }

        if n_layers == 0
            || channels == 0
            || kernel == 0
            || rank == 0
            || ff_width == 0
            || window == 0
            || in_features == 0
            || out_features != 3
            || n_tensors != 5 + n_layers * TENSORS_PER_LAYER
        {
            return Err(ModelError::ShapeMismatch("header counts"));
        }

        // Layer descriptors: only the base tensor index is used; the ten tensors
        // of each block follow in the documented order.
        if body.len() < HEADER_BYTES + n_layers * LAYER_DESC_BYTES {
            return Err(ModelError::Truncated);
        }
        let mut layer_bases = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let off = HEADER_BYTES + l * LAYER_DESC_BYTES;
            layer_bases.push(u32::from_le_bytes(body[off..off + 4].try_into().unwrap()) as usize);
        }

        // Tensor table.
        let table_off = HEADER_BYTES + n_layers * LAYER_DESC_BYTES;
        let table_end = table_off + n_tensors * TENSOR_DESC_BYTES;
        if body.len() < table_end {
            return Err(ModelError::Truncated);
        }
        let table_end = table_off + n_tensors * TENSOR_DESC_BYTES;
        if body.len() < table_end {
            return Err(ModelError::Truncated);
        }
        let mut descs = Vec::with_capacity(n_tensors);
        for i in 0..n_tensors {
            let off = table_off + i * TENSOR_DESC_BYTES;
            let kind = match body[off] {
                1 => TensorKind::F32,
                2 => TensorKind::Ternary,
                k => return Err(ModelError::InvalidTensorKind(k)),
            };
            let rows = u32::from_le_bytes(body[off + 4..off + 8].try_into().unwrap()) as usize;
            let cols = u32::from_le_bytes(body[off + 8..off + 12].try_into().unwrap()) as usize;
            let offset = u32::from_le_bytes(body[off + 12..off + 16].try_into().unwrap()) as usize;
            let bytes = u32::from_le_bytes(body[off + 16..off + 20].try_into().unwrap()) as usize;
            descs.push((kind, rows, cols, offset, bytes));
        }

        let expected_bytes = |kind: TensorKind, rows: usize, cols: usize| -> usize {
            let n = if cols == 0 { rows } else { rows * cols };
            match kind {
                TensorKind::F32 => n * 4,
                TensorKind::Ternary => rows * 4 + n,
            }
        };
        for (kind, rows, cols, offset, len) in &descs {
            if table_end + offset + len > body.len() || *len != expected_bytes(*kind, *rows, *cols)
            {
                return Err(ModelError::ShapeMismatch("tensor payload size"));
            }
        }

        // The payload region must be exactly covered by the descriptors.
        let mut spans: Vec<(usize, usize)> = descs
            .iter()
            .map(|(_, _, _, off, len)| (*off, *off + *len))
            .collect();
        spans.sort_unstable();
        let mut cursor = 0usize;
        for (start, end) in spans {
            if start != cursor {
                return Err(ModelError::TrailingBytes);
            }
            cursor = end;
        }
        if cursor != body.len() - table_end {
            return Err(ModelError::TrailingBytes);
        }

        let f32_tensor = |idx: usize, rows: usize, cols: usize| -> Result<Vec<f32>, ModelError> {
            let (kind, r, c, off, _) = descs.get(idx).ok_or(ModelError::BadTensorIndex(idx))?;
            if *kind != TensorKind::F32 || *r != rows || *c != cols {
                return Err(ModelError::ShapeMismatch("f32 tensor"));
            }
            let base = table_end + off;
            let mut v = Vec::with_capacity(rows * cols.max(1));
            for i in 0..rows * cols.max(1) {
                v.push(f32::from_le_bytes(
                    body[base + i * 4..base + i * 4 + 4].try_into().unwrap(),
                ));
            }
            Ok(v)
        };
        let ternary_tensor =
            |idx: usize, rows: usize, cols: usize| -> Result<Vec<f32>, ModelError> {
                let (kind, r, c, off, _) = descs.get(idx).ok_or(ModelError::BadTensorIndex(idx))?;
                if *kind != TensorKind::Ternary || *r != rows || *c != cols {
                    return Err(ModelError::ShapeMismatch("ternary tensor"));
                }
                let base = table_end + off;
                let mut scales = Vec::with_capacity(rows);
                for i in 0..rows {
                    scales.push(f32::from_le_bytes(
                        body[base + i * 4..base + i * 4 + 4].try_into().unwrap(),
                    ));
                }
                let codes_at = base + rows * 4;
                let mut out = Vec::with_capacity(rows * cols);
                for r in 0..rows {
                    for c in 0..cols {
                        let code = body[codes_at + r * cols + c];
                        let q = match code {
                            0 => -1.0f32,
                            1 => 0.0,
                            2 => 1.0,
                            other => return Err(ModelError::InvalidTernaryCode(other)),
                        };
                        out.push(q * scales[r]);
                    }
                }
                Ok(out)
            };
        // Dense TSCT reconstruction: W = U diag(s) V^T.
        let low_rank = |u_idx: usize,
                        v_idx: usize,
                        s_idx: usize,
                        u_rows: usize,
                        v_rows: usize,
                        rank: usize|
         -> Result<Vec<f32>, ModelError> {
            let u = ternary_tensor(u_idx, u_rows, rank)?;
            let v = ternary_tensor(v_idx, v_rows, rank)?;
            let s = f32_tensor(s_idx, rank, 0)?;
            let mut w = vec![0.0f32; u_rows * v_rows];
            for i in 0..u_rows {
                for j in 0..v_rows {
                    let mut acc = 0.0f32;
                    for k in 0..rank {
                        acc += u[i * rank + k] * s[k] * v[j * rank + k];
                    }
                    w[i * v_rows + j] = acc;
                }
            }
            Ok(w)
        };

        let in_w = f32_tensor(0, channels, in_features)?;
        let in_b = f32_tensor(1, channels, 0)?;
        let final_norm = f32_tensor(2, channels, 0)?;
        let out_w = f32_tensor(3, out_features, channels)?;
        let out_b = f32_tensor(4, out_features, 0)?;

        let mut layers = Vec::with_capacity(n_layers);
        for (l, &base) in layer_bases.iter().enumerate() {
            let expected_base = 5 + l * TENSORS_PER_LAYER;
            if base != expected_base {
                return Err(ModelError::ShapeMismatch("layer descriptor base"));
            }
            layers.push(Layer {
                dw_w: ternary_tensor(base, channels, kernel)?,
                dw_b: f32_tensor(base + 1, channels, 0)?,
                norm1: f32_tensor(base + 2, channels, 0)?,
                norm2: f32_tensor(base + 3, channels, 0)?,
                up_w: low_rank(base + 4, base + 5, base + 6, ff_width, channels, rank)?,
                down_w: low_rank(base + 7, base + 8, base + 9, channels, ff_width, rank)?,
            });
        }

        Ok(Self {
            channels,
            kernel,
            rank,
            window,
            ff_width,
            in_features,
            out_features,
            in_w,
            in_b,
            final_norm,
            out_w,
            out_b,
            layers,
        })
    }

    /// Last-timestep logit head.
    pub(crate) fn head(&self, last: &[f32], scratch: &mut [f32]) -> (f32, f32, f32) {
        let c = self.channels;
        let last = &last[..c];
        let mean = last.iter().map(|v| v * v).sum::<f32>() / c as f32;
        let inv = 1.0 / (mean + 1e-6f32).sqrt();
        for i in 0..c {
            scratch[i] = last[i] * inv * self.final_norm[i];
        }
        let mut o = [0.0f32; 3];
        for k in 0..self.out_features {
            let row = &self.out_w[k * c..(k + 1) * c];
            o[k] = crate::simd::dot(row, &scratch[..c]) + self.out_b[k];
        }
        (o[0], o[1], o[2])
    }

    /// One window forward pass. All buffers are preallocated in `Scratch`.
    ///
    /// The five blocks are fused into one pass over timesteps: for each `t` we
    /// compute norm1, the causal depthwise convolution, the residual write,
    /// norm2 and the TSCT pointwise map before moving to `t+1`. The depthwise
    /// kernel only needs norm1 rows `t-K..=t`, which are already final, so the
    /// fusion is exact. `norm2` and `up` are per-timestep buffers (C and D
    /// floats instead of T*C and T*D), and every inner dot uses pre-sliced
    /// rows plus a NEON/scalar kernel.
    pub(crate) fn forward(&self, window: &[f32], scratch: &mut Scratch) -> (f32, f32, f32) {
        let t = self.window;
        let c = self.channels;
        let d = self.ff_width;
        let k = self.kernel;
        let fi = self.in_features;
        let dot = crate::simd::dot;

        let x = &mut scratch.x[..t * c];
        x.fill(0.0);
        for ti in 0..t {
            let wrow = &window[ti * fi..(ti + 1) * fi];
            let xrow = &mut x[ti * c..(ti + 1) * c];
            for (ci, dst) in xrow.iter_mut().enumerate() {
                let w = &self.in_w[ci * fi..(ci + 1) * fi];
                *dst = dot(w, wrow) + self.in_b[ci];
            }
        }

        for layer in self.layers() {
            for ti in 0..t {
                let base = ti * c;
                // norm1, current timestep (kept for causal lookback)
                {
                    let src = &x[base..base + c];
                    let mean = src.iter().map(|v| v * v).sum::<f32>() / c as f32;
                    let inv = 1.0 / (mean + 1e-6f32).sqrt();
                    let dst = &mut scratch.norm1[base..base + c];
                    for ((dst, src), gain) in dst.iter_mut().zip(src.iter()).zip(layer.norm1.iter())
                    {
                        *dst = *src * inv * *gain;
                    }
                }
                let xrow = &mut x[base..base + c];
                // depthwise causal conv + SiLU + residual
                for ci in 0..c {
                    let mut acc = layer.dw_b[ci];
                    let kw = &layer.dw_w[ci * k..(ci + 1) * k];
                    for lag in 0..k {
                        if ti >= lag {
                            let v = if lag == 0 {
                                scratch.norm1[base + ci]
                            } else {
                                scratch.norm1[(ti - lag) * c + ci]
                            };
                            acc += v * kw[lag];
                        }
                    }
                    xrow[ci] += acc / (1.0 + (-acc).exp());
                }
                // norm2 over channels of the post-convolution residual
                let mean = xrow.iter().map(|v| v * v).sum::<f32>() / c as f32;
                let inv = 1.0 / (mean + 1e-6f32).sqrt();
                for ((dst, src), gain) in scratch.norm2[..c]
                    .iter_mut()
                    .zip(xrow.iter())
                    .zip(layer.norm2.iter())
                {
                    *dst = *src * inv * *gain;
                }
                // TSCT pointwise: up -> SiLU -> down -> residual
                let norm2 = &scratch.norm2[..c];
                for di in 0..d {
                    let w = &layer.up_w[di * c..(di + 1) * c];
                    let a = dot(w, norm2);
                    scratch.up[di] = a / (1.0 + (-a).exp());
                }
                let up = &scratch.up[..d];
                for ci in 0..c {
                    let w = &layer.down_w[ci * d..(ci + 1) * d];
                    xrow[ci] += dot(w, up);
                }
            }
        }

        let last = &x[(t - 1) * c..t * c];
        self.head(last, &mut scratch.head[..c])
    }
}

/// Preallocated scratch space. Built once per Runtime; reused every frame.
pub(crate) struct Scratch {
    x: Vec<f32>,
    norm1: Vec<f32>, // T*C, causal history for the depthwise kernel
    norm2: Vec<f32>, // C, one timestep at a time
    up: Vec<f32>,    // D, one timestep at a time
    head: Vec<f32>,  // C
}

impl Scratch {
    pub(crate) fn new(model: &Model) -> Self {
        let n = model.window * model.channels;
        Self {
            x: vec![0.0; n],
            norm1: vec![0.0; n],
            norm2: vec![0.0; model.channels],
            up: vec![0.0; model.ff_width],
            head: vec![0.0; model.channels],
        }
    }

    pub(crate) fn memory_bytes(&self) -> usize {
        (self.x.len() + self.norm1.len() + self.norm2.len() + self.up.len() + self.head.len())
            * std::mem::size_of::<f32>()
    }
}
