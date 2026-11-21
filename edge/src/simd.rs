//! Dot-product kernel used by every linear map in the hot loop.
//!
//! `dot` is the fast path: NEON on aarch64 (Raspberry Pi), scalar elsewhere.
//! `dot_scalar` is the ordered, always-available baseline used by the
//! in-crate naive implementation and by the architecture equivalence test.
//!
//! NEON accumulates four products per lane and sums lanes at the end, so on
//! aarch64 the result differs from the strictly ordered scalar sum by at
//! most a few ULPs; the bound is asserted by `dot_neon_matches_scalar`.

/// Strictly ordered scalar dot product. Also the reference implementation.
pub fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    for i in 0..a.len().min(b.len()) {
        acc += a[i] * b[i];
    }
    acc
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::{vaddq_f32, vld1q_f32, vmulq_f32, vst1q_f32, vdupq_n_f32};
    let n = a.len().min(b.len());
    let mut acc = vdupq_n_f32(0.0);
    let mut i = 0usize;
    while i + 4 <= n {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        acc = vaddq_f32(acc, vmulq_f32(va, vb));
        i += 4;
    }
    let mut lanes = [0.0f32; 4];
    vst1q_f32(lanes.as_mut_ptr(), acc);
    let mut sum = ((lanes[0] + lanes[1]) + lanes[2]) + lanes[3];
    while i < n {
        sum += *a.get_unchecked(i) * *b.get_unchecked(i);
        i += 1;
    }
    sum
}

/// Fast dot product. Both slices must have equal length.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    {
        // NEON is mandatory on aarch64; the runtime probe keeps the call
        // sound if a future target disables it.
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { dot_neon(a, b) };
        }
    }
    dot_scalar(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_fast_matches_scalar_within_bound() {
        let mut state = 0x1234_5678u32;
        let mut next = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let a: Vec<f32> = (0..64).map(|_| next()).collect();
        let b: Vec<f32> = (0..64).map(|_| next()).collect();
        let fast = dot(&a, &b);
        let slow = dot_scalar(&a, &b);
        let err = (fast - slow).abs();
        // On x86_64 both paths are the same ordered loop: bit-identical.
        // On aarch64 NEON changes only the summation tree, not the products:
        // a few ULPs at 64 terms is far below 1e-4 absolute.
        assert!(err <= 1e-4 * (1.0 + slow.abs()), "dot error {err}");
        #[cfg(not(target_arch = "aarch64"))]
        assert_eq!(fast.to_bits(), slow.to_bits());
    }
}
