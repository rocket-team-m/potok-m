//! Handcrafted canonical model fixture, generated entirely in Rust.
//! Used by the test suite and by `make-fixture` for hardware-in-the-loop runs.
//! No Python, no checkpoint files and no network are needed.

use crate::model::{
    CHECKSUM_BYTES, FORMAT_VERSION, HEADER_BYTES, LAYER_DESC_BYTES, MAGIC, TENSORS_PER_LAYER,
    TENSOR_DESC_BYTES,
};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Copy)]
pub struct FixtureConfig {
    pub layers: usize,
    pub channels: usize,
    pub kernel: usize,
    pub rank: usize,
    pub window: usize,
    pub ff_width: usize,
    pub in_features: usize,
    pub out_features: usize,
}

impl Default for FixtureConfig {
    fn default() -> Self {
        Self {
            layers: 5,
            channels: 8,
            kernel: 3,
            rank: 2,
            window: 50,
            ff_width: 12,
            in_features: 8,
            out_features: 3,
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

    fn code(&mut self) -> u8 {
        (self.next() % 3) as u8
    }

    fn scale(&mut self) -> f32 {
        0.05 + (self.next() % 20) as f32 * 0.01
    }
}

enum Blob {
    F32(Vec<f32>),
    Ternary(Vec<f32>, Vec<u8>), // scales per row, codes row-major
}

impl Blob {
    fn bytes(&self) -> Vec<u8> {
        match self {
            Self::F32(v) => {
                let mut b = Vec::with_capacity(v.len() * 4);
                for x in v {
                    b.extend_from_slice(&x.to_le_bytes());
                }
                b
            }
            Self::Ternary(scales, codes) => {
                let mut b = Vec::with_capacity(scales.len() * 4 + codes.len());
                for x in scales {
                    b.extend_from_slice(&x.to_le_bytes());
                }
                b.extend_from_slice(codes);
                b
            }
        }
    }

    fn kind(&self) -> u8 {
        match self {
            Self::F32(_) => 1,
            Self::Ternary(_, _) => 2,
        }
    }
}

fn f32s(rng: &mut Lcg, n: usize, magnitude: f32) -> Blob {
    Blob::F32((0..n).map(|_| rng.f32() * magnitude).collect())
}

fn ternary(rng: &mut Lcg, rows: usize, cols: usize) -> Blob {
    let scales: Vec<f32> = (0..rows).map(|_| rng.scale()).collect();
    let codes: Vec<u8> = (0..rows * cols).map(|_| rng.code()).collect();
    Blob::Ternary(scales, codes)
}

/// Build a complete, checksummed canonical model file.
pub fn build_fixture(cfg: FixtureConfig, seed: u64) -> Vec<u8> {
    let mut rng = Lcg(seed);
    let n_tensors = 5 + cfg.layers * TENSORS_PER_LAYER;
    let mut blobs: Vec<(Blob, usize, usize)> = Vec::with_capacity(n_tensors);
    let mut push = |blob: Blob, rows: usize, cols: usize| {
        blobs.push((blob, rows, cols));
    };

    let c = cfg.channels;
    let d = cfg.ff_width;
    let r = cfg.rank;
    let k = cfg.kernel;
    let i = cfg.in_features;

    push(f32s(&mut rng, c * i, 0.4), c, i); // in_w
    push(f32s(&mut rng, c, 0.1), c, 0); // in_b
    push(Blob::F32(vec![1.0; c]), c, 0); // final_norm
    push(
        f32s(&mut rng, cfg.out_features * c, 0.3),
        cfg.out_features,
        c,
    ); // out_w
    push(f32s(&mut rng, cfg.out_features, 0.05), cfg.out_features, 0); // out_b

    for _ in 0..cfg.layers {
        push(ternary(&mut rng, c, k), c, k); // dw kernel
        push(f32s(&mut rng, c, 0.02), c, 0); // dw bias
        push(Blob::F32(vec![1.0; c]), c, 0); // norm1
        push(Blob::F32(vec![1.0; c]), c, 0); // norm2
        push(ternary(&mut rng, d, r), d, r); // up_u
        push(ternary(&mut rng, c, r), c, r); // up_v
        push(f32s(&mut rng, r, 0.5), r, 0); // up_s
        push(ternary(&mut rng, c, r), c, r); // down_u
        push(ternary(&mut rng, d, r), d, r); // down_v
        push(f32s(&mut rng, r, 0.5), r, 0); // down_s
    }
    assert_eq!(blobs.len(), n_tensors);

    let table_off = HEADER_BYTES + cfg.layers * LAYER_DESC_BYTES;
    let payload_off = table_off + n_tensors * TENSOR_DESC_BYTES;
    let mut body = vec![0u8; payload_off];
    body[0..4].copy_from_slice(&MAGIC);
    let put_u32 = |body: &mut Vec<u8>, at: usize, v: u32| {
        body[at..at + 4].copy_from_slice(&v.to_le_bytes());
    };
    put_u32(&mut body, 4, FORMAT_VERSION);
    put_u32(&mut body, 8, cfg.layers as u32);
    put_u32(&mut body, 12, c as u32);
    put_u32(&mut body, 16, k as u32);
    put_u32(&mut body, 20, r as u32);
    put_u32(&mut body, 24, cfg.window as u32);
    put_u32(&mut body, 28, i as u32);
    put_u32(&mut body, 32, cfg.out_features as u32);
    put_u32(&mut body, 36, d as u32);
    put_u32(&mut body, 40, n_tensors as u32);
    for l in 0..cfg.layers {
        put_u32(
            &mut body,
            HEADER_BYTES + l * LAYER_DESC_BYTES,
            (5 + l * TENSORS_PER_LAYER) as u32,
        );
    }

    let mut cursor = 0usize;
    for (idx, (blob, rows, cols)) in blobs.iter().enumerate() {
        let off = table_off + idx * TENSOR_DESC_BYTES;
        let encoded = blob.bytes();
        body[off] = blob.kind();
        put_u32(&mut body, off + 4, *rows as u32);
        put_u32(&mut body, off + 8, *cols as u32);
        put_u32(&mut body, off + 12, cursor as u32);
        put_u32(&mut body, off + 16, encoded.len() as u32);
        body.extend_from_slice(&encoded);
        cursor += encoded.len();
    }
    debug_assert_eq!(body.len(), payload_off + cursor);

    let digest = Sha256::digest(&body);
    body.extend_from_slice(&digest);
    debug_assert_eq!(body.len(), payload_off + cursor + CHECKSUM_BYTES);
    body
}
