//! Synthetic inference benchmark: p50/p99 in microseconds and memory use.
//! With `--compare`, also runs the in-crate naive baseline and reports the
//! maximum output difference between the two implementations.

use potok_edge::baseline::forward_naive;
use potok_edge::frame::{
    build_request, PotokUartV1, RequestFrame, FLAG_IMU_OK, PHASE_LANDING, REPLY_LEN, REQUEST_LEN,
};
use potok_edge::runtime::Runtime;
use potok_edge::FrameLayout;
use std::env;
use std::time::{Duration, Instant};

fn percentile(sorted: &[Duration], q: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx]
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut model_path = String::new();
    let mut frames = 2000usize;
    let mut compare = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" if i + 1 < args.len() => {
                model_path = args[i + 1].clone();
                i += 2;
            }
            "--frames" if i + 1 < args.len() => {
                frames = args[i + 1].parse().unwrap_or_else(|_| {
                    eprintln!("invalid frame count");
                    std::process::exit(2);
                });
                i += 2;
            }
            "--compare" => {
                compare = true;
                i += 1;
            }
            other => {
                eprintln!("unknown argument: {other}");
                eprintln!("usage: bench --model MODEL.bin [--frames 2000] [--compare]");
                std::process::exit(2);
            }
        }
    }
    if model_path.is_empty() || frames == 0 {
        eprintln!("usage: bench --model MODEL.bin [--frames 2000] [--compare]");
        std::process::exit(2);
    }
    let mut runtime = Runtime::load(&model_path).unwrap_or_else(|e| {
        eprintln!("model load failed: {e}");
        std::process::exit(1);
    });
    let memory = runtime.memory_bytes();

    let layout = PotokUartV1;
    let mut req_bytes = [0u8; REQUEST_LEN];
    let mut reply_bytes = [0u8; REPLY_LEN];
    let mut samples = Vec::with_capacity(frames);
    let mut model_samples = Vec::with_capacity(frames);
    let mut overhead_samples = Vec::with_capacity(frames);
    let mut checksum = 0u64;
    let mut max_err = 0f32;
    let mut bit_identical = 0usize;

    for n in 0..frames {
        let req = RequestFrame {
            seq: n as u16,
            time_us: (n as u32) * 20_000,
            phase: PHASE_LANDING,
            version: 1,
            flags: FLAG_IMU_OK,
            gyro: [0.01, -0.02, 0.03],
            accel: [0.1, 9.8, -0.1],
            mag: [1.0, 2.0, 3.0],
            height: 1000.0 + n as f32 * 0.1,
            pitot_dp: (n % 100) as i16,
        };
        build_request(&req, &mut req_bytes).unwrap();
        let t0 = Instant::now();
        let decoded = layout.decode_request(&req_bytes).unwrap();
        let t1 = Instant::now();
        let reply = runtime.on_request(&decoded);
        let t2 = Instant::now();
        layout.encode_reply(&reply, &mut reply_bytes).unwrap();
        let t3 = Instant::now();
        samples.push(t3 - t0);
        model_samples.push(t2 - t1);
        overhead_samples.push((t1 - t0) + (t3 - t2));
        checksum = checksum
            .wrapping_add(reply.g_pitch as u64)
            .wrapping_add(reply.g_yaw as u64);

        if compare && !runtime.is_v2() {
            let fast = runtime.last_raw();
            let slow = forward_naive(runtime.potm_model().unwrap(), runtime.last_window());
            let err = fast
                .iter()
                .zip(slow.iter())
                .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
            max_err = max_err.max(err);
            if fast
                .iter()
                .zip(slow.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits())
            {
                bit_identical += 1;
            }
        }
    }
    let mut baseline_samples = Vec::with_capacity(if compare { frames } else { 0 });
    if compare && !runtime.is_v2() {
        // Same shape, same window: time the naive control in this binary so
        // the before/after numbers are directly comparable.
        for _ in 0..frames {
            let t0 = Instant::now();
            let _ = forward_naive(runtime.potm_model().unwrap(), runtime.last_window());
            baseline_samples.push(t0.elapsed());
        }
        baseline_samples.sort_unstable();
    }
    samples.sort_unstable();
    model_samples.sort_unstable();
    overhead_samples.sort_unstable();
    let p50 = percentile(&samples, 0.50);
    let p99 = percentile(&samples, 0.99);
    let model_p50 = percentile(&model_samples, 0.50);
    let model_p99 = percentile(&model_samples, 0.99);
    let overhead_p50 = percentile(&overhead_samples, 0.50);
    let overhead_p99 = percentile(&overhead_samples, 0.99);
    println!("potok-edge benchmark: {frames} synthetic frames");
    println!("  p50    : {:>9.1} us", p50.as_secs_f64() * 1e6);
    println!("  p99    : {:>9.1} us", p99.as_secs_f64() * 1e6);
    println!(
        "  model      p50 {:>7.1} us / p99 {:>7.1} us",
        model_p50.as_secs_f64() * 1e6,
        model_p99.as_secs_f64() * 1e6
    );
    println!(
        "  overhead   p50 {:>7.1} us / p99 {:>7.1} us (decode+encode, no UART wire)",
        overhead_p50.as_secs_f64() * 1e6,
        overhead_p99.as_secs_f64() * 1e6
    );
    println!(
        "  memory : {} bytes ({} KiB) — weights + preallocated buffers",
        memory,
        memory / 1024
    );
    println!("  checksum: {checksum:#x}");
    if compare && !runtime.is_v2() {
        let b50 = percentile(&baseline_samples, 0.50);
        let b99 = percentile(&baseline_samples, 0.99);
        println!(
            "  naive control: p50 {:>9.1} us, p99 {:>9.1} us",
            b50.as_secs_f64() * 1e6,
            b99.as_secs_f64() * 1e6
        );
        println!("  baseline comparison: max |diff| = {max_err:.3e}, bit-identical {bit_identical}/{frames}");
    } else if compare {
        println!(
            "  baseline comparison: skipped (v2 model; naive control covers legacy POTM only)"
        );
    }
}
