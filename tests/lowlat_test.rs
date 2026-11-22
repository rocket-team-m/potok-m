//! Lowlat profile integration: wire codec, safety gates and dispatch.
//! Golden bytes come from `protocol/potok-protocol/tests/lowlat_tests.rs`
//! (read-only); nothing here writes outside `edge/`.

use potok_edge::fixture::{build_fixture, FixtureConfig};
use potok_edge::frame::{
    build_request, FrameError, FrameLayout, PotokUartV1, ReplyFrame, RequestFrame, FLAG_IMU_OK,
    PHASE_LANDING, PROTOCOL_VERSION, REQUEST_LEN,
};
use potok_edge::lowlat::LowlatLayout;
use potok_edge::model::Model;
use potok_edge::runtime::Runtime;
use potok_edge::student_runtime::StudentRuntime;
use potok_edge::uart::handle_spec_request;
use potok_edge::SpeculativeRuntime;
use potok_protocol::{
    crc16, decode_lowlat_reply, decode_lowlat_request, encode_lowlat_request, quantize_i16,
    LowlatRequestFrame, Phase, LOWLAT_REPLY_VERSION, LOWLAT_REQUEST_CRC_OFF,
    LOWLAT_REQUEST_FEATURES_SCALES, LOWLAT_REQUEST_OFF_PHASE, LOWLAT_REQUEST_SIZE,
    LOWLAT_REQUEST_VERSION,
};
use std::path::Path;
use std::time::Duration;

const REQ_GOLDEN: &str =
    "544c34127bf15365020112000100feff03000400fbff06000700f8ff09000a000b00f4ff0d000e000a2b";
const REP_GOLDEN: &str = "524c34127d0006ffc801012a0000fd1b";
const SEED: u64 = 0x504F544B;

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

fn set_crc(buf: &mut [u8]) {
    let crc = crc16(&buf[..LOWLAT_REQUEST_CRC_OFF]);
    buf[LOWLAT_REQUEST_CRC_OFF] = crc as u8;
    buf[LOWLAT_REQUEST_CRC_OFF + 1] = (crc >> 8) as u8;
}

fn sample_lowlat() -> LowlatRequestFrame {
    LowlatRequestFrame {
        seq: 0x1234,
        time_us: 1_700_000_123,
        phase: Phase::APOGEE,
        version: LOWLAT_REQUEST_VERSION,
        flags: FLAG_IMU_OK | 0x10, // IMU_OK | AI_READY
        reserved: 0,
        features: [1, -2, 3, 4, -5, 6, 7, -8, 9, 10, 11, -12, 13, 14],
    }
}

fn legacy_runtime() -> Runtime {
    Runtime::new(Model::from_bytes(&build_fixture(FixtureConfig::default(), SEED)).unwrap())
}

fn spec_runtime(student: StudentRuntime) -> SpeculativeRuntime {
    SpeculativeRuntime::new(legacy_runtime(), student)
}

#[test]
fn lowlat_round_trip_and_dequantization() {
    let layout = LowlatLayout::new();
    let frame = sample_lowlat();
    let mut bytes = [0u8; LOWLAT_REQUEST_SIZE];
    encode_lowlat_request(&frame, &mut bytes).unwrap();
    let decoded = layout.decode_request(&bytes).unwrap();
    assert_eq!(decoded.seq, 0x1234);
    assert_eq!(decoded.time_us, 1_700_000_123);
    assert_eq!(decoded.phase, 2); // APOGEE
    assert_eq!(decoded.flags, frame.flags);
    assert_eq!(decoded.pitot_dp, -12);
    assert!((decoded.gyro[0] - 0.001).abs() < 1e-9);
    assert!((decoded.mag[2] - 0.0009).abs() < 1e-9);
    assert!((decoded.height - 0.10).abs() < 1e-9);

    let reply = ReplyFrame {
        seq: decoded.seq,
        g_pitch: 125,
        g_yaw: -250,
        confidence: 200,
        ai_healthy: 1,
        version: LOWLAT_REPLY_VERSION as u16,
    };
    layout.set_compute(Duration::from_micros(4_200)); // 42 units of 100 us
    let mut out = [0u8; 16];
    layout.encode_reply(&reply, &mut out).unwrap();
    let decoded_reply = decode_lowlat_reply(&out).unwrap();
    assert_eq!(decoded_reply.compute_us, 42);
    assert_eq!((decoded_reply.g_pitch, decoded_reply.g_yaw), (125, -250));
    assert_eq!(decoded_reply.reserved, [0, 0]);
    assert_eq!(layout.wire_bytes(), 42 + 16);
}

#[test]
fn lowlat_rejects_bad_crc_magic_version_and_phase() {
    let layout = LowlatLayout::new();
    let good = sample_lowlat();
    let mut bytes = [0u8; LOWLAT_REQUEST_SIZE];
    encode_lowlat_request(&good, &mut bytes).unwrap();

    bytes[0] ^= 0xFF;
    assert!(matches!(
        layout.decode_request(&bytes),
        Err(FrameError::BadMagic(_))
    ));

    let mut bytes = [0u8; LOWLAT_REQUEST_SIZE];
    encode_lowlat_request(&good, &mut bytes).unwrap();
    bytes[LOWLAT_REQUEST_OFF_PHASE] ^= 1;
    assert!(matches!(
        layout.decode_request(&bytes),
        Err(FrameError::CrcMismatch { .. })
    ));

    let mut bytes = [0u8; LOWLAT_REQUEST_SIZE];
    encode_lowlat_request(&good, &mut bytes).unwrap();
    bytes[LOWLAT_REQUEST_OFF_PHASE] = 9;
    set_crc(&mut bytes);
    assert_eq!(
        layout.decode_request(&bytes).unwrap_err(),
        FrameError::BadPhase(9)
    );

    let mut bad = good;
    bad.version = 99;
    let mut bytes = [0u8; LOWLAT_REQUEST_SIZE];
    encode_lowlat_request(&bad, &mut bytes).unwrap();
    assert_eq!(
        layout.decode_request(&bytes).unwrap_err(),
        FrameError::VersionMismatch(99)
    );
}

#[test]
fn lowlat_int16_features_saturate() {
    assert_eq!(quantize_i16(1e9, 1e-3), i16::MAX);
    assert_eq!(quantize_i16(-1e9, 1e-3), i16::MIN);
    assert_eq!(quantize_i16(1.0, LOWLAT_REQUEST_FEATURES_SCALES[6]), 10_000);

    let mut frame = sample_lowlat();
    frame.features[0] = i16::MAX;
    frame.features[1] = i16::MIN;
    let mut bytes = [0u8; LOWLAT_REQUEST_SIZE];
    encode_lowlat_request(&frame, &mut bytes).unwrap();
    let decoded = LowlatLayout::new().decode_request(&bytes).unwrap();
    assert_eq!(decoded.gyro[0], 32.767);
    assert_eq!(decoded.gyro[1], -32.768);
}

#[test]
fn lowlat_golden_bytes_match_generated_artifacts() {
    let req = unhex(REQ_GOLDEN);
    let layout = LowlatLayout::new();
    let decoded = layout.decode_request(&req).unwrap();
    assert_eq!(decode_lowlat_request(&req).unwrap(), sample_lowlat());
    assert_eq!(decoded.seq, 0x1234);
    assert_eq!(decoded.flags, FLAG_IMU_OK | 0x10);

    layout.set_compute(Duration::from_micros(4_200));
    let reply = ReplyFrame {
        seq: decoded.seq,
        g_pitch: 125,
        g_yaw: -250,
        confidence: 200,
        ai_healthy: 1,
        version: LOWLAT_REPLY_VERSION as u16,
    };
    let mut out = [0u8; 16];
    layout.encode_reply(&reply, &mut out).unwrap();
    assert_eq!(out.to_vec(), unhex(REP_GOLDEN));
}

#[test]
fn lowlat_phase_gate_and_three_miss_disable() {
    let layout = LowlatLayout::new();
    let mut rt = spec_runtime(StudentRuntime::new(
        potok_edge::StudentModel::from_bytes(&build_student_fixture()).unwrap(),
    ));
    let mut frame = sample_lowlat();
    frame.phase = Phase::BURN;
    let mut bytes = [0u8; LOWLAT_REQUEST_SIZE];
    encode_lowlat_request(&frame, &mut bytes).unwrap();
    let mut out = [0u8; 16];
    handle_spec_request(&layout, &mut rt, &bytes, &mut out).unwrap();
    assert_eq!((out[4], out[5], out[6], out[7]), (0, 0, 0, 0)); // g_pitch, g_yaw
    assert_eq!(out[8], 0, "BURN confidence is zero");
    assert_eq!(out[9], 1, "muting is policy, not a fault");
    assert_eq!(rt.misses(), 0);

    // Three corrupted frames latch the assistant off.
    bytes[0] ^= 0xFF;
    for _ in 0..3 {
        // A bad frame still gets a fault reply, never a model call.
        handle_spec_request(&layout, &mut rt, &bytes, &mut out).unwrap();
        assert_eq!(out[9], 0, "ai_healthy must be 0 on a bad frame");
    }
    assert!(rt.is_disabled());
    assert_eq!(rt.counters.invalid_frames, 3);
    assert!(rt.counters.student_calls == 0 && rt.counters.teacher_calls == 0);
}

#[test]
fn dispatch_works_for_both_profiles_and_both_student_widths() {
    let smart_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/smart_student.bin");
    for (student, label) in [
        (
            StudentRuntime::new(
                potok_edge::StudentModel::from_bytes(&build_student_fixture()).unwrap(),
            ),
            "legacy-10",
        ),
        (StudentRuntime::from_file(&smart_path).unwrap(), "smart-20"),
    ] {
        let _ = label;
        let mut rt = spec_runtime(student);
        rt.set_conditioning([13.0, 0.0082, 0.35, 0.95, 3000.0, 0.01]);

        // Compat profile.
        let req = RequestFrame {
            seq: 1,
            time_us: 20_000,
            phase: PHASE_LANDING,
            version: PROTOCOL_VERSION,
            flags: FLAG_IMU_OK,
            gyro: [0.02, -0.01, 0.005],
            accel: [0.05, 9.79, -0.03],
            mag: [1.0, 2.0, 3.0],
            height: 1500.0,
            pitot_dp: 7,
        };
        let mut compat_bytes = [0u8; REQUEST_LEN];
        build_request(&req, &mut compat_bytes).unwrap();
        let mut out = [0u8; 16];
        handle_spec_request(&PotokUartV1, &mut rt, &compat_bytes, &mut out).unwrap();
        assert_eq!(out[2..4], req.seq.to_le_bytes());

        // Lowlat profile.
        let mut low = sample_lowlat();
        low.seq = 2;
        low.phase = Phase::DESCENT;
        low.flags = FLAG_IMU_OK;
        let mut low_bytes = [0u8; LOWLAT_REQUEST_SIZE];
        encode_lowlat_request(&low, &mut low_bytes).unwrap();
        handle_spec_request(&LowlatLayout::new(), &mut rt, &low_bytes, &mut out).unwrap();
        assert_eq!(u16::from_le_bytes([out[2], out[3]]), 2);
        assert_eq!(rt.counters.frames, 2);
        assert_eq!(rt.counters.student_calls, 2);
        assert!(rt.counters.teacher_calls <= 2);
    }
}

fn build_student_fixture() -> Vec<u8> {
    potok_edge::build_student_fixture(potok_edge::StudentFixtureConfig::default(), 0x504F544B)
}
