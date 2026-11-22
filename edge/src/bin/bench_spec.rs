//! Speculative dispatch benchmark: p50/p99 for the three dispatch paths.
//!
//! Without file arguments it builds the in-crate deterministic fixtures, so
//! it needs no model files and touches no other crate's directory.
//! `cargo run --release --bin bench-spec -- --frames 2000`

use potok_edge::fixture::{build_fixture, FixtureConfig};
use potok_edge::frame::{
    build_request, FrameLayout, PotokUartV1, RequestFrame, FLAG_IMU_OK, PHASE_LANDING,
    PROTOCOL_VERSION,
};
use potok_edge::lowlat::LowlatLayout;
use potok_edge::model::Model;
use potok_edge::runtime::Runtime;
use potok_edge::student_runtime::{
    build_student_fixture, StudentFixtureConfig, StudentModel, StudentRuntime,
};
use potok_edge::SpeculativeRuntime;
use potok_protocol::{
    encode_lowlat_request, quantize_i16, LowlatRequestFrame, Phase, LOWLAT_REQUEST_FEATURES_SCALES,
    LOWLAT_REQUEST_VERSION,
};
use std::env;
use std::time::{Duration, Instant};

const UART_WIRE_MS: f64 = 0.61; // spec budget: one 56-byte reply at 921600 baud

fn percentile(sorted: &[Duration], q: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    sorted[(((sorted.len() - 1) as f64) * q).round() as usize]
}

/// Full-cycle benchmark for one profile: encode request -> decode -> dispatch
/// -> encode reply, including the CRC and quantize/dequantize work.
fn bench_profile<F: Fn() -> SpeculativeRuntime>(
    name: &str,
    baud: u32,
    requests: &[RequestFrame],
    make_runtime: &F,
) {
    let compat = name == "compat";
    let layout: Box<dyn FrameLayout> = if compat {
        Box::new(PotokUartV1)
    } else {
        Box::new(LowlatLayout::new())
    };
    let req_len = layout.request_len();
    let rep_len = layout.reply_len();
    let mut rt = make_runtime();
    let mut req_bytes = [0u8; 64];
    let mut rep_bytes = [0u8; 32];
    let mut samples = Vec::with_capacity(requests.len());

    for req in requests {
        if compat {
            build_request(req, &mut req_bytes[..req_len]).unwrap();
        } else {
            let physical = [
                f64::from(req.gyro[0]),
                f64::from(req.gyro[1]),
                f64::from(req.gyro[2]),
                f64::from(req.accel[0]),
                f64::from(req.accel[1]),
                f64::from(req.accel[2]),
                f64::from(req.mag[0]),
                f64::from(req.mag[1]),
                f64::from(req.mag[2]),
                f64::from(req.height),
                101_325.0,
                f64::from(req.pitot_dp),
                22.0,
                12_000.0,
            ];
            let mut features = [0i16; 14];
            for (k, value) in physical.iter().enumerate() {
                features[k] = quantize_i16(*value, LOWLAT_REQUEST_FEATURES_SCALES[k]);
            }
            let frame = LowlatRequestFrame {
                seq: req.seq,
                time_us: req.time_us,
                phase: Phase::try_from(req.phase).unwrap(),
                version: LOWLAT_REQUEST_VERSION,
                flags: req.flags,
                reserved: 0,
                features,
            };
            encode_lowlat_request(&frame, &mut req_bytes[..req_len]).unwrap();
        }

        let t0 = Instant::now();
        let decoded = layout.decode_request(&req_bytes[..req_len]).unwrap();
        let t_dispatch = Instant::now();
        let reply = rt.on_request(&decoded);
        layout.note_compute(t_dispatch.elapsed());
        layout
            .encode_reply(&reply, &mut rep_bytes[..rep_len])
            .unwrap();
        samples.push(t0.elapsed());
    }
    samples.sort_unstable();
    let p50 = percentile(&samples, 0.50);
    let p99 = percentile(&samples, 0.99);
    let bytes = layout.wire_bytes();
    let wire_us = bytes as f64 * 8.0 / f64::from(baud) * 1e6;
    println!(
        "{name} profile: {bytes} B/cycle, wire {wire_us:7.1} us @ {baud} baud, full cycle p50 {:7.1} us / p99 {:7.1} us (p99 + wire {:7.1} us)",
        p50.as_secs_f64() * 1e6,
        p99.as_secs_f64() * 1e6,
        p99.as_secs_f64() * 1e6 + wire_us
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut frames = 2000usize;
    let mut teacher_path: Option<String> = None;
    let mut student_path: Option<String> = None;
    let mut profile = String::from("both");
    let mut baud: Option<u32> = None;
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
            "--profile" if i + 1 < args.len() => {
                profile = args[i + 1].clone();
                i += 2;
            }
            "--baud" if i + 1 < args.len() => {
                baud = Some(args[i + 1].parse().unwrap_or_else(|_| {
                    eprintln!("invalid baud rate");
                    std::process::exit(2);
                }));
                i += 2;
            }
            other => {
                eprintln!("unknown argument: {other}");
                eprintln!("usage: bench-spec [--teacher T.bin] [--student S.bin] [--frames 2000] [--profile compat|lowlat|both] [--baud N]");
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

    for name in ["compat", "lowlat"] {
        if profile == "both" || profile == name {
            let chosen = baud.unwrap_or(if name == "compat" { 921_600 } else { 2_000_000 });
            bench_profile(name, chosen, &requests, &make_runtime);
        }
    }
}
