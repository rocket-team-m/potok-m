//! Speculative dispatch benchmark: p50/p99 for the three dispatch paths.
//!
//! Without file arguments it builds the in-crate deterministic fixtures, so
//! it needs no model files and touches no other crate's directory.
//! `cargo run --release --bin bench-spec -- --frames 2000`

use potok_edge::fixture::{build_fixture, FixtureConfig};
use potok_edge::frame::{RequestFrame, FLAG_IMU_OK, PHASE_LANDING, PROTOCOL_VERSION};
use potok_edge::model::Model;
use potok_edge::runtime::Runtime;
use potok_edge::student_runtime::{
    build_student_fixture, StudentFixtureConfig, StudentModel, StudentRuntime,
};
use potok_edge::SpeculativeRuntime;
use std::env;
use std::time::{Duration, Instant};

const UART_WIRE_MS: f64 = 0.61; // spec budget: one 56-byte reply at 921600 baud

fn percentile(sorted: &[Duration], q: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    sorted[(((sorted.len() - 1) as f64) * q).round() as usize]
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut frames = 2000usize;
    let mut teacher_path: Option<String> = None;
    let mut student_path: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--frames" if i + 1 < args.len() => {
                frames = args[i + 1].parse().unwrap_or_else(|_| {
                    eprintln!("invalid frame count");
                    std::process::exit(2);
                });
                i += 2;
            }
            "--teacher" if i + 1 < args.len() => {
                teacher_path = Some(args[i + 1].clone());
                i += 2;
            }
            "--student" if i + 1 < args.len() => {
                student_path = Some(args[i + 1].clone());
                i += 2;
            }
            other => {
                eprintln!("unknown argument: {other}");
                eprintln!("usage: bench-spec [--teacher T.bin] [--student S.bin] [--frames 2000]");
                std::process::exit(2);
            }
        }
    }
    if frames == 0 {
        eprintln!("frame count must be positive");
        std::process::exit(2);
    }

    let requests: Vec<RequestFrame> = (0..frames)
        .map(|n| RequestFrame {
            seq: (n + 1) as u16,
            time_us: (n as u32) * 20_000,
            phase: PHASE_LANDING, // allowed by the conservative gate
            version: PROTOCOL_VERSION,
            flags: FLAG_IMU_OK,
            gyro: [0.02, -0.01, 0.005],
            accel: [0.05, 9.79, -0.03],
            mag: [1.0, 2.0, 3.0],
            height: 1500.0 + n as f32,
            pitot_dp: (n % 50) as i16,
        })
        .collect();

    let make_runtime = || -> SpeculativeRuntime {
        let teacher = match &teacher_path {
            Some(p) => Runtime::load(p).unwrap_or_else(|e| {
                eprintln!("teacher load failed: {e}");
                std::process::exit(1);
            }),
            None => Runtime::new(
                Model::from_bytes(&build_fixture(FixtureConfig::default(), 0x504F544B)).unwrap(),
            ),
        };
        let student = match &student_path {
            Some(p) => StudentRuntime::from_file(p).unwrap_or_else(|e| {
                eprintln!("student load failed: {e}");
                std::process::exit(1);
            }),
            None => StudentRuntime::new(
                StudentModel::from_bytes(&build_student_fixture(
                    StudentFixtureConfig::default(),
                    0x504F544B,
                ))
                .unwrap(),
            ),
        };
        SpeculativeRuntime::new(teacher, student)
    };

    let mut worst_p99 = Duration::ZERO;
    for (name, threshold, period) in [
        ("normal      ", 0.0, u32::MAX),
        ("low-conf    ", 1.1, u32::MAX),
        ("forced-check", 0.0, 1),
    ] {
        let mut rt = make_runtime();
        rt.confidence_threshold = threshold;
        rt.forced_check_every = period;
        let mut samples = Vec::with_capacity(frames);
        for req in &requests {
            let t0 = Instant::now();
            let reply = rt.on_request(req);
            samples.push(t0.elapsed());
            assert_eq!(reply.seq, req.seq);
        }
        samples.sort_unstable();
        let p50 = percentile(&samples, 0.50);
        let p99 = percentile(&samples, 0.99);
        worst_p99 = worst_p99.max(p99);
        let c = rt.counters;
        println!(
            "{name}: p50 {:8.1} us  p99 {:8.1} us  (student {} / teacher {} / forced {})",
            p50.as_secs_f64() * 1e6,
            p99.as_secs_f64() * 1e6,
            c.student_calls,
            c.teacher_calls,
            c.forced_checks
        );
        assert!(
            c.teacher_calls <= c.frames,
            "the one-teacher-per-cycle invariant was broken"
        );
    }
    let cycle = worst_p99.as_secs_f64() * 1e3 + UART_WIRE_MS;
    println!(
        "worst cycle (max path p99 + {UART_WIRE_MS:.2} ms UART wire): {cycle:.3} ms / 10 ms budget"
    );
    if cycle >= 10.0 {
        eprintln!("BUDGET EXCEEDED");
        std::process::exit(1);
    }
}
