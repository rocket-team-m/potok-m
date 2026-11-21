//! Naive reference implementation of the same forward pass.
//!
//! Deliberately unoptimized: separate passes over all timesteps per stage,
//! per-frame heap allocations, indexed scalar loops, no fusion and no SIMD.
//! It exists only as the control for `optimized_matches_baseline` and for
//! `bench --compare`, so any speed or output difference has a named cause.

use crate::model::Model;
use crate::simd::dot_scalar;

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

/// Straightforward reference forward. Returns raw `[g_pitch, g_yaw, confidence]`.
pub fn forward_naive(model: &Model, window: &[f32]) -> [f32; 3] {
    let t = model.window;
    let c = model.channels;
    let k = model.kernel;
    let d = model.ff_width;
    let fi = model.in_features;

    let mut x = vec![0.0f32; t * c];
    let mut norm = vec![0.0f32; t * c];
    let mut up = vec![0.0f32; t * d];

    for ti in 0..t {
        for ci in 0..c {
            let mut acc = 0.0f32;
            for f in 0..fi {
                acc += model.in_w[ci * fi + f] * window[ti * fi + f];
            }
            x[ti * c + ci] = acc + model.in_b[ci];
        }
    }

    for layer in model.layers() {
        for ti in 0..t {
            let base = ti * c;
            let mean: f32 = (0..c).map(|ci| x[base + ci] * x[base + ci]).sum::<f32>() / c as f32;
            let inv = 1.0 / (mean + 1e-6f32).sqrt();
            for ci in 0..c {
                norm[base + ci] = x[base + ci] * inv * layer.norm1[ci];
            }
        }
        for ti in 0..t {
            for ci in 0..c {
                let mut acc = layer.dw_b[ci];
                for lag in 0..k {
                    if ti >= lag {
                        acc += norm[(ti - lag) * c + ci] * layer.dw_w[ci * k + lag];
                    }
                }
                x[ti * c + ci] += silu(acc);
            }
        }
        for ti in 0..t {
            let base = ti * c;
            let mean: f32 = (0..c).map(|ci| x[base + ci] * x[base + ci]).sum::<f32>() / c as f32;
            let inv = 1.0 / (mean + 1e-6f32).sqrt();
            for ci in 0..c {
                norm[base + ci] = x[base + ci] * inv * layer.norm2[ci];
            }
        }
        for ti in 0..t {
            for di in 0..d {
                let mut acc = 0.0f32;
                for ci in 0..c {
                    acc += layer.up_w[di * c + ci] * norm[ti * c + ci];
                }
                up[ti * d + di] = silu(acc);
            }
        }
        for ti in 0..t {
            for ci in 0..c {
                let mut acc = 0.0f32;
                for di in 0..d {
                    acc += layer.down_w[ci * d + di] * up[ti * d + di];
                }
                x[ti * c + ci] += acc;
            }
        }
    }

    let last = &x[(t - 1) * c..t * c];
    let mean: f32 = last.iter().map(|v| v * v).sum::<f32>() / c as f32;
    let inv = 1.0 / (mean + 1e-6f32).sqrt();
    let mut head = vec![0.0f32; c];
    for ci in 0..c {
        head[ci] = last[ci] * inv * model.final_norm[ci];
    }
    let mut out = [0.0f32; 3];
    for o in 0..model.out_features {
        let row = &model.out_w[o * c..(o + 1) * c];
        out[o] = dot_scalar(row, &head) + model.out_b[o];
    }
    out
}
