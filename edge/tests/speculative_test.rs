//! Acceptance tests for the stateful student and speculative dispatch.

use potok_edge::fixture::{build_fixture, FixtureConfig};
use potok_edge::frame::{
    build_request, PotokUartV1, ReplyFrame, RequestFrame, FLAG_IMU_OK, FLAG_MOTOR_ARMED,
    PHASE_BURN, PHASE_LANDING, PROTOCOL_VERSION, REPLY_LEN, REQUEST_LEN,
};
use potok_edge::model::Model;
use potok_edge::runtime::Runtime;
use potok_edge::student_runtime::{
    build_student_fixture, is_student_file, model_kind, StudentError, StudentFixtureConfig,
    StudentModel, StudentRuntime, WINDOW,
};
use potok_edge::uart::handle_spec_request;
use potok_edge::{DispatchSource, FrameLayout, SpeculativeRuntime};
use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

const SEED: u64 = 0x504F544B;

fn student_bytes() -> Vec<u8> {
    build_student_fixture(StudentFixtureConfig::default(), SEED)
}

fn student() -> StudentRuntime {
    StudentRuntime::new(StudentModel::from_bytes(&student_bytes()).unwrap())
}

fn teacher() -> Runtime {
    Runtime::new(Model::from_bytes(&build_fixture(FixtureConfig::default(), SEED)).unwrap())
}

fn spec() -> SpeculativeRuntime {
    SpeculativeRuntime::new(teacher(), student())
}

fn req(seq: u16, phase: u8, flags: u8) -> RequestFrame {
    RequestFrame {
        seq,
        time_us: seq as u32 * 20_000,
        phase,
        version: PROTOCOL_VERSION,
        flags,
        gyro: [0.02, -0.01, 0.005],
        accel: [0.05, 9.79, -0.03],
        mag: [1.0, 2.0, 3.0],
        height: 1500.0 + seq as f32,
        pitot_dp: (seq % 50) as i16,
    }
}

fn frame(n: usize) -> [f32; 10] {
    let v = n as f32;
    [
        v.sin() * 0.3,
        v.cos() * 0.2,
        (v * 0.3).sin() * 0.1,
        0.05,
        9.79 + (v * 0.1).sin() * 0.2,
        -0.03,
        (v % 50.0) * 0.01,
        1500.0 + v,
        PHASE_LANDING as f32,
        FLAG_IMU_OK as f32,
    ]
}

#[test]
fn model_kind_dispatch() {
    let s = student_bytes();
    assert_eq!(model_kind(&s).unwrap(), 2);
    assert!(is_student_file(&s));
    StudentModel::from_bytes(&s).unwrap();

    // Legacy POTM is not a POTOKM1 file at all.
    let t = build_fixture(FixtureConfig::default(), SEED);
    assert!(!is_student_file(&t));
    assert!(model_kind(&t).is_err());
    assert_eq!(
        StudentModel::from_bytes(&t).unwrap_err(),
        StudentError::BadMagic
    );

    // Canonical teacher v2 has no model_kind field: offset 12 is in_features.
    let v2 = Path::new(env!("CARGO_MANIFEST_DIR")).join("../ai/tests/fixtures/tiny_model.bin");
    let v2 = fs::read(v2).unwrap();
    assert!(!is_student_file(&v2));
    assert_eq!(model_kind(&v2).unwrap(), 8);
}

#[test]
fn student_is_deterministic() {
    let bytes = student_bytes();
    let mut a = StudentRuntime::new(StudentModel::from_bytes(&bytes).unwrap());
    let mut b = StudentRuntime::new(StudentModel::from_bytes(&bytes).unwrap());
    for n in 0..100 {
        let f = frame(n);
        let oa = a.step(&f);
        let ob = b.step(&f);
        for k in 0..3 {
            assert_eq!(oa[k].to_bits(), ob[k].to_bits(), "frame {n}, output {k}");
        }
    }
    assert_eq!(a.state(), b.state());
}

#[test]
fn state_reset_restores_initial_behaviour() {
    let mut rt = student();
    for n in 0..30 {
        rt.step(&frame(n));
    }
    let f0 = frame(0);
    let fresh = student().step(&f0);
    rt.reset_state();
    let after_reset = rt.step(&f0);
    for k in 0..3 {
        assert_eq!(fresh[k].to_bits(), after_reset[k].to_bits());
    }
    assert!(rt.state()[..8].iter().all(|v| v.is_finite()));
}

#[test]
fn warmup_replays_a_window_exactly() {
    let window: Vec<f32> = (0..WINDOW).flat_map(frame).collect();
    let mut a = student();
    a.warmup(&window);
    let mut b = student();
    for f in window.chunks_exact(10) {
        b.step(f);
    }
    assert_eq!(a.state(), b.state());
    let probe = frame(123);
    for (x, y) in a.step(&probe).iter().zip(b.step(&probe)) {
        assert_eq!(x.to_bits(), y.to_bits());
    }
}

#[test]
fn phase_gate_zeroes_burn_and_motor_armed() {
    let mut rt = spec();
    let burn = rt.on_request(&req(1, PHASE_BURN, FLAG_IMU_OK));
    assert_eq!((burn.g_pitch, burn.g_yaw, burn.confidence), (0, 0, 0));
    assert_eq!(burn.ai_healthy, 1, "muting is policy, not a fault");
    assert_eq!(rt.counters.student_calls, 0);
    assert_eq!(rt.counters.teacher_calls, 0);

    let mut armed = req(2, PHASE_LANDING, FLAG_IMU_OK | FLAG_MOTOR_ARMED);
    armed.flags |= FLAG_MOTOR_ARMED;
    let out = rt.on_request(&armed);
    assert_eq!((out.g_pitch, out.g_yaw, out.confidence), (0, 0, 0));
    assert_eq!(rt.counters.teacher_calls, 0);
    assert_eq!(rt.counters.zeroed, 2);
}

#[test]
fn low_confidence_calls_teacher_once_and_applies_it() {
    let mut low = spec();
    low.confidence_threshold = 1.1; // every frame is below threshold
    low.forced_check_every = u32::MAX;
    let reply = low.on_request(&req(1, PHASE_LANDING, FLAG_IMU_OK));
    assert_eq!(low.counters.teacher_calls, 1);
    assert_eq!(low.counters.low_confidence, 1);
    assert_eq!(low.last_source, DispatchSource::TeacherLowConfidence);
    let bound = low.g_max();
    assert_eq!(reply.g_pitch, (low.last_g[0] / 0.001).round() as i16);
    assert!((low.last_g[0].abs() <= bound) && (low.last_g[1].abs() <= bound));
    assert!((reply.g_pitch as f32 * 0.001).abs() <= bound + 1e-6);

    let mut normal = spec();
    normal.confidence_threshold = 0.0;
    normal.forced_check_every = u32::MAX;
    let _ = normal.on_request(&req(1, PHASE_LANDING, FLAG_IMU_OK));
    assert_eq!(normal.counters.teacher_calls, 0);
    assert_eq!(normal.last_source, DispatchSource::Student);
}

#[test]
fn forced_check_cadence_and_reset_on_low_confidence() {
    let mut rt = spec();
    rt.confidence_threshold = 0.0;
    for n in 1..=39 {
        rt.on_request(&req(n, PHASE_LANDING, FLAG_IMU_OK));
    }
    assert_eq!(rt.counters.forced_checks, 1); // frame 20
    assert_eq!(rt.counters.teacher_calls, 1);
    rt.on_request(&req(40, PHASE_LANDING, FLAG_IMU_OK));
    assert_eq!(rt.counters.forced_checks, 2); // frame 40

    // A low-confidence call at 41 restarts the 20-frame counter.
    rt.confidence_threshold = 1.1;
    rt.on_request(&req(41, PHASE_LANDING, FLAG_IMU_OK));
    rt.confidence_threshold = 0.0;
    for n in 42..=60 {
        rt.on_request(&req(n, PHASE_LANDING, FLAG_IMU_OK));
    }
    assert_eq!(rt.counters.forced_checks, 2, "no check at frame 60");
    rt.on_request(&req(61, PHASE_LANDING, FLAG_IMU_OK));
    assert_eq!(rt.counters.forced_checks, 3, "check at frame 61");
    assert_eq!(rt.counters.teacher_calls, 4);
}

#[test]
fn checksum_refusal_and_model_kind_confusion() {
    let s = student_bytes();
    let mut bad = s.clone();
    *bad.last_mut().unwrap() ^= 1;
    assert_eq!(
        StudentModel::from_bytes(&bad).unwrap_err(),
        StudentError::ChecksumMismatch
    );

    let dir = std::env::temp_dir().join(format!("potok-spec-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let sp = dir.join("student.bin");
    let tp = dir.join("teacher.bin");
    fs::write(&sp, &s).unwrap();
    fs::write(&tp, build_fixture(FixtureConfig::default(), SEED)).unwrap();
    assert!(SpeculativeRuntime::load(&tp, &sp).is_ok());
    let swapped = SpeculativeRuntime::load(&sp, &tp).err().unwrap();
    assert!(swapped.contains("model_kind=2"), "{swapped}");
    let doubled = SpeculativeRuntime::load(&tp, &tp).err().unwrap();
    assert!(doubled.contains("model_kind=2"), "{doubled}");
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn divergence_is_logged_and_command_stays_bounded() {
    let mut rt = spec();
    rt.confidence_threshold = 0.0;
    rt.forced_check_every = 1;
    // Both heads are tanh-bounded by g_max, so the literal 3*g_max trip point
    // cannot fire with shared bounds. A negative factor exercises the
    // log + warmup-reset plumbing deterministically.
    rt.divergence_factor = -1.0;
    let reply = rt.on_request(&req(1, PHASE_LANDING, FLAG_IMU_OK));
    assert_eq!(rt.counters.divergences, 1);
    assert_eq!(rt.counters.student_resets, 1);
    assert_eq!(rt.events().len(), 1);
    assert_eq!(rt.events()[0].seq, 1);
    assert!(rt.events()[0].diff >= 0.0);
    let bound = rt.g_max();
    assert!(rt.last_g.iter().all(|v| v.abs() <= bound));
    assert!((reply.g_pitch as f32 * 0.001).abs() <= bound + 1e-6);
    assert!((reply.g_yaw as f32 * 0.001).abs() <= bound + 1e-6);

    // With the specified factor, the bounded heads never diverge by 3*g_max.
    let mut normal = spec();
    normal.confidence_threshold = 0.0;
    normal.forced_check_every = 1;
    for n in 1..=100 {
        normal.on_request(&req(n, PHASE_LANDING, FLAG_IMU_OK));
    }
    assert_eq!(normal.counters.divergences, 0);
    assert!(normal.events().is_empty());
}

#[test]
fn invalid_frames_bypass_both_models() {
    let mut rt = spec();
    rt.confidence_threshold = 0.0;
    rt.forced_check_every = u32::MAX;
    rt.on_invalid_frame();
    assert_eq!(rt.counters.invalid_frames, 1);
    assert_eq!(rt.counters.student_calls, 0);
    assert_eq!(rt.counters.teacher_calls, 0);

    let first = rt.on_request(&req(10, PHASE_LANDING, FLAG_IMU_OK));
    assert_eq!(first.ai_healthy, 1);
    let expected_calls = rt.counters.student_calls;
    let gap = rt.on_request(&req(20, PHASE_LANDING, FLAG_IMU_OK));
    assert_eq!(gap, ReplyFrame::muted(20, false));
    assert_eq!(rt.counters.seq_errors, 1);
    assert_eq!(rt.counters.student_calls, expected_calls);
    assert_eq!(rt.counters.teacher_calls, 0);

    let layout = PotokUartV1;
    let mut bytes = [0u8; REQUEST_LEN];
    build_request(&req(21, PHASE_LANDING, FLAG_IMU_OK), &mut bytes).unwrap();
    bytes[30] ^= 0x80;
    let mut out = [0u8; REPLY_LEN];
    handle_spec_request(&layout, &mut rt, &bytes, &mut out).unwrap();
    assert_eq!(out[9], 0, "bad CRC replies with a fault indication");
    assert_eq!(rt.counters.invalid_frames, 3); // timeout + seq + CRC
    assert_eq!(rt.counters.teacher_calls, 0);
    let _ = PotokUartV1.decode_request(&bytes).err();
}

#[test]
fn thirty_seconds_have_no_nan_and_bounded_state() {
    let mut rt = student();
    for n in 0..(30 * 50) {
        let out = rt.step(&frame(n));
        assert!(out.iter().all(|v| v.is_finite()), "frame {n}: {out:?}");
        assert!(out[0].abs() <= 1.0 && out[1].abs() <= 1.0);
        assert!((0.0..=1.0).contains(&out[2]));
    }
    assert!(
        rt.state().iter().all(|v| v.is_finite() && v.abs() <= 1.0),
        "tanh cell keeps the recurrent state in [-1, 1]"
    );
}

#[test]
fn worst_case_cycle_is_under_10ms_on_host() {
    let mut rt = spec();
    rt.confidence_threshold = 0.0;
    rt.forced_check_every = 1; // teacher every frame = worst case
    let layout = PotokUartV1;
    let mut req_bytes = [0u8; REQUEST_LEN];
    let mut reply_bytes = [0u8; REPLY_LEN];
    let mut samples = Vec::with_capacity(300);
    for n in 1..=300u16 {
        build_request(&req(n, PHASE_LANDING, FLAG_IMU_OK), &mut req_bytes).unwrap();
        let decoded = layout.decode_request(&req_bytes).unwrap();
        let t0 = Instant::now();
        let reply = rt.on_request(&decoded);
        layout.encode_reply(&reply, &mut reply_bytes).unwrap();
        samples.push(t0.elapsed());
    }
    samples.sort_unstable();
    let p99 = samples[((samples.len() - 1) as f64 * 0.99).round() as usize];
    println!(
        "worst-cycle p99 (dispatch + codec): {:.1} us; teacher calls {}",
        p99.as_secs_f64() * 1e6,
        rt.counters.teacher_calls
    );
    assert!(rt.counters.teacher_calls == 300);
    assert!(
        p99 < Duration::from_millis(10),
        "p99 {p99:?} exceeded 10 ms"
    );
}
