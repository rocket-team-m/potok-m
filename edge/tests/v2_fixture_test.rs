//! Cross-check against the canonical Python fixture produced by `ai/`.
//! Read-only: this test never writes into another directory.

use potok_edge::format_v2::ModelV2;
use potok_edge::frame::{RequestFrame, FLAG_IMU_OK, PHASE_FAULT, PHASE_LANDING, PROTOCOL_VERSION};
use potok_edge::runtime::Runtime;
use std::fs;
use std::path::{Path, PathBuf};

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// Minimal NumPy `.npy` v1 reader: little-endian float32, C order.
fn read_npy_f32(path: PathBuf) -> Vec<f32> {
    let d = fs::read(path).unwrap();
    assert_eq!(&d[..6], b"\x93NUMPY");
    assert_eq!(d[6], 1, "only npy v1 needed for this fixture");
    let hlen = u16::from_le_bytes([d[8], d[9]]) as usize;
    let header = std::str::from_utf8(&d[10..10 + hlen]).unwrap();
    assert!(header.contains("'<f4'"), "expected float32 fixture");
    assert!(!header.contains("'fortran_order': True"));
    let shape_text = header.split("'shape':").nth(1).unwrap();
    let inside = shape_text
        .split('(')
        .nth(1)
        .unwrap()
        .split(')')
        .next()
        .unwrap();
    let dims: Vec<usize> = inside
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let count: usize = dims.iter().product();
    let data = &d[10 + hlen..10 + hlen + count * 4];
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// `(arange(400, dtype=float32).reshape(1,50,8) - 200) / 100` from FORMAT.md.
fn fixture_window() -> Vec<f32> {
    (0..400).map(|i| (i as f32 - 200.0) / 100.0).collect()
}

#[test]
fn loads_python_v2_fixture_and_matches_expected() {
    let mut model = ModelV2::from_file(fixture_dir().join("tiny_model.bin"))
        .expect("canonical v2 fixture must load");
    assert_eq!(model.in_features, 8);
    assert_eq!(model.width, 16);
    assert_eq!(model.rank, 8);
    assert_eq!(model.n_blocks, 5);
    assert_eq!(model.dilations, vec![1, 2, 4, 8, 16]);

    let expected = read_npy_f32(fixture_dir().join("tiny_expected.npy"));
    assert_eq!(expected.len(), 3);
    let got = model.forward(&fixture_window());
    for i in 0..3 {
        let err = (got[i] - expected[i]).abs();
        println!(
            "out[{i}]: got {:?}, expected {:?}, abs err {err:.3e}",
            got[i], expected[i]
        );
        // fp32 accumulation order differs between Rust and PyTorch's CPU
        // kernels. Measured max error on this fixture is 1.2e-7 (~1 ULP);
        // the documented bound is 1e-6 absolute (~8 ULPs at these magnitudes).
        assert!(err <= 1e-6, "output {i} outside documented fp32 tolerance");
    }
}

#[test]
fn service_loads_v2_fixture_and_mutes_on_forbidden_phase() {
    let mut rt = Runtime::load(fixture_dir().join("tiny_model.bin")).unwrap();
    assert!(rt.is_v2());
    let req = |seq: u16, phase: u8| RequestFrame {
        seq,
        time_us: seq as u32 * 20_000,
        phase,
        version: PROTOCOL_VERSION,
        flags: FLAG_IMU_OK,
        gyro: [0.0; 3],
        accel: [0.0, 9.81, 0.0],
        mag: [0.0; 3],
        height: 1000.0,
        pitot_dp: 0,
    };
    let ok = rt.on_request(&req(1, PHASE_LANDING));
    assert_eq!(ok.ai_healthy, 1);
    let muted = rt.on_request(&req(2, PHASE_FAULT));
    assert_eq!((muted.g_pitch, muted.g_yaw, muted.confidence), (0, 0, 0));
    assert_eq!(muted.ai_healthy, 1);
}

#[test]
fn v2_fixture_refuses_tampered_checksum() {
    let mut bytes = fs::read(fixture_dir().join("tiny_model.bin")).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0x01;
    assert_eq!(
        ModelV2::from_bytes(&bytes).unwrap_err(),
        potok_edge::format_v2::V2Error::ChecksumMismatch
    );
}
