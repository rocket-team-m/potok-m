//! Integration against the real exports produced by `ai/`. Skips (does not
//! fail) when the data volume is not mounted, so the suite stays green on a
//! bare checkout. Read-only: nothing here writes outside `edge/`.

use potok_edge::frame::{RequestFrame, FLAG_IMU_OK, PHASE_LANDING, PROTOCOL_VERSION};
use potok_edge::student_runtime::{model_kind, StudentRuntime};
use potok_edge::SpeculativeRuntime;
use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

const TEACHER: &str = "/mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/potok-m-data/models/distill.bin";
const STUDENT: &str = "/mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/potok-m-data/models/student.bin";

fn available() -> bool {
    Path::new(TEACHER).is_file() && Path::new(STUDENT).is_file()
}

fn req(seq: u16) -> RequestFrame {
    let v = seq as f32;
    RequestFrame {
        seq,
        time_us: seq as u32 * 20_000,
        phase: PHASE_LANDING,
        version: PROTOCOL_VERSION,
        flags: FLAG_IMU_OK,
        gyro: [v.sin() * 0.05, v.cos() * 0.03, 0.005],
        accel: [0.05, 9.79 + (v * 0.1).sin() * 0.2, -0.03],
        mag: [1.0, 2.0, 3.0],
        height: 1500.0 + v,
        pitot_dp: (seq % 50) as i16,
    }
}

fn percentile(sorted: &[Duration], q: f64) -> Duration {
    sorted[(((sorted.len() - 1) as f64) * q).round() as usize]
}

#[test]
fn real_exports_load_with_the_documented_model_kinds() {
    if !available() {
        eprintln!("skip: real model files are not mounted");
        return;
    }
    assert_eq!(model_kind(&fs::read(STUDENT).unwrap()).unwrap(), 2);
    // The teacher's v2 header has no model_kind field; offset 12 is in_features.
    assert_eq!(model_kind(&fs::read(TEACHER).unwrap()).unwrap(), 8);

    let rt = SpeculativeRuntime::load(TEACHER, STUDENT).unwrap();
    let m = rt.student().model();
    assert_eq!(
        (m.in_features, m.width, m.out_features, m.rank),
        (10, 64, 2, 16)
    );
    assert_eq!((m.n_blocks, m.n_loops, m.flags), (2, 2, 3));
    assert!((m.decay - 0.8).abs() < 1e-5);
    assert!((m.g_max - 0.910741).abs() < 1e-5);
    assert!(m.correction_bound > 0.0 && m.correction_bound < m.g_max);
}

#[test]
fn real_student_matches_python_reference() {
    if !available() {
        eprintln!("skip: real model files are not mounted");
        return;
    }
    let frame = |n: usize| -> [f32; 10] {
        [
            0.1 + 0.01 * n as f32,
            -0.2,
            0.3,
            0.05,
            9.79,
            -0.03,
            0.42,
            1200.0 + n as f32,
            3.0,
            2.0,
        ]
    };
    // `ai` venv, 2026-11-22, after the leaky-state / total+source update:
    // load_student_export(STUDENT).to_model().step/warmup(frame).
    let cases: [(&str, usize, [f32; 3]); 3] = [
        ("single", 0, [0.01283988, -0.110_565, 0.813_98]),
        ("warm3", 3, [0.00487398, -0.10744341, 0.817_990_2]),
        ("warm50", 50, [-0.00207486, -0.11013591, 0.818_366_6]),
    ];
    for (name, warm, expected) in cases {
        let mut student = StudentRuntime::from_file(STUDENT).unwrap();
        let window: Vec<f32> = (0..warm).flat_map(frame).collect();
        student.warmup(&window);
        let out = student.step(&frame(warm));
        println!("rust student {name}: {out:?}");
        for k in 0..3 {
            assert!(
                (out[k] - expected[k]).abs() <= 1e-5,
                "{name} out[{k}] {} vs Python {}",
                out[k],
                expected[k]
            );
        }
    }
}

#[test]
fn real_exports_dispatch_and_report_all_three_paths() {
    if !available() {
        eprintln!("skip: real model files are not mounted");
        return;
    }
    let requests: Vec<RequestFrame> = (1..=600u16).map(req).collect();
    for (name, threshold, period) in [
        ("normal      ", 0.0, u32::MAX),
        ("low-conf    ", 1.1, u32::MAX),
        ("forced-check", 0.0, 1),
    ] {
        let mut rt = SpeculativeRuntime::load(TEACHER, STUDENT).unwrap();
        rt.confidence_threshold = threshold;
        rt.forced_check_every = period;
        let mut samples = Vec::with_capacity(requests.len());
        for r in &requests {
            let t0 = Instant::now();
            let reply = rt.on_request(r);
            samples.push(t0.elapsed());
            assert_eq!(reply.seq, r.seq);
            assert_eq!(reply.ai_healthy, 1);
            assert!(rt.last_g.iter().all(|v| v.is_finite()));
        }
        samples.sort_unstable();
        let p50 = percentile(&samples, 0.50);
        let p99 = percentile(&samples, 0.99);
        println!(
            "real {name}: p50 {:8.1} us  p99 {:8.1} us  (student {} / teacher {} / forced {})",
            p50.as_secs_f64() * 1e6,
            p99.as_secs_f64() * 1e6,
            rt.counters.student_calls,
            rt.counters.teacher_calls,
            rt.counters.forced_checks
        );
        assert_eq!(rt.counters.student_calls, requests.len() as u64);
        assert!(rt.counters.teacher_calls <= rt.counters.frames);
        assert!(p99 < Duration::from_millis(10), "p99 {p99:?} over budget");
    }
}
