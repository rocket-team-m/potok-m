mod common;

use common::{fixture, request};
use potok_edge::baseline::forward_naive;
use potok_edge::model::Model;
use potok_edge::runtime::Runtime;

/// The fused/NEON path must agree with the naive reference on the same window.
/// On x86_64 both reduce to the same ordered scalar loop (bit-identical); on
/// aarch64 NEON changes only the summation tree, so the bound is a few ULPs.
#[test]
fn optimized_matches_baseline() {
    let model = Model::from_bytes(&fixture()).unwrap();
    let mut rt = Runtime::new(model);
    let mut max_err = 0.0f32;
    let mut bit_identical = 0usize;
    let n = 40;
    for seq in 0..n as u16 {
        rt.on_request(&request(seq));
        let fast = rt.last_raw();
        let slow = forward_naive(rt.potm_model().unwrap(), rt.last_window());
        for (a, b) in fast.iter().zip(slow.iter()) {
            max_err = max_err.max((a - b).abs());
            assert!(
                (a - b).abs() <= 1e-4 * (1.0 + b.abs()),
                "optimized {a} vs baseline {b}"
            );
        }
        if fast
            .iter()
            .zip(slow.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits())
        {
            bit_identical += 1;
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    assert_eq!(
        bit_identical, n,
        "scalar fallback must be bit-identical to baseline"
    );
    #[cfg(target_arch = "aarch64")]
    assert!(max_err < 1e-4, "NEON summation error {max_err}");
}
