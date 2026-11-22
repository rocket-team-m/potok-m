//! Microbenchmark for the UART hot path: `cargo run --release -p potok-protocol --example hotpath`.
//!
//! Reports ns/frame and MB/s (wire bytes per second) for CRC, encode, decode
//! and encode+decode. `black_box` keeps the compiler from eliding the work.

use potok_protocol::*;
use std::hint::black_box;
use std::time::{Duration, Instant};

fn sample_request() -> RequestFrame {
    RequestFrame {
        seq: 0x1234,
        time_us: 1_700_000_123,
        phase: Phase::APOGEE,
        version: REQUEST_VERSION,
        flags: flags::IMU_OK | flags::AI_READY,
        reserved: 0,
        gyro: [0.1, -0.2, 3.3],
        accel: [9.81, 0.0, -1.5],
        mag: [12.5, -3.25, 44.0],
        height: 123.75,
        pitot_dp: -321,
    }
}

fn sample_reply() -> ReplyFrame {
    ReplyFrame {
        seq: 0x1234,
        g_pitch: 125,
        g_yaw: -250,
        confidence: 200,
        ai_healthy: 1,
        version: REPLY_VERSION,
    }
}

fn sample_lowlat_request() -> LowlatRequestFrame {
    LowlatRequestFrame {
        seq: 0x1234,
        time_us: 1_700_000_123,
        phase: Phase::APOGEE,
        version: LOWLAT_REQUEST_VERSION,
        flags: flags::IMU_OK | flags::AI_READY,
        reserved: 0,
        features: [1, -2, 3, 4, -5, 6, 7, -8, 9, 10, 11, -12, 13, 14],
    }
}

fn sample_lowlat_reply() -> LowlatReplyFrame {
    LowlatReplyFrame {
        seq: 0x1234,
        g_pitch: 125,
        g_yaw: -250,
        confidence: 200,
        ai_healthy: 1,
        version: LOWLAT_REPLY_VERSION,
        compute_us: 42,
        reserved: [0, 0],
    }
}

fn crc16_bitwise(data: &[u8]) -> u16 {
    let mut crc = CRC_INIT;
    for &byte in data {
        crc ^= u16::from(byte) << 8;
        for _ in 0..8 {
            crc = if crc & 0x8000 != 0 {
                (crc << 1) ^ CRC_POLYNOMIAL
            } else {
                crc << 1
            };
        }
    }
    crc ^ CRC_XOR_OUT
}

fn time<F: FnMut()>(iters: u64, f: &mut F) -> Duration {
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    start.elapsed()
}

fn report(name: &str, bytes: u64, iters: u64, reps: usize, mut f: impl FnMut()) {
    for _ in 0..20_000 {
        f();
    }
    let best = (0..reps)
        .map(|_| time(iters, &mut f))
        .min()
        .expect("at least one repetition");
    let ns = best.as_nanos() as f64 / iters as f64;
    let mbs = bytes as f64 * iters as f64 / best.as_secs_f64() / 1e6;
    println!("{name:<28} {ns:10.1} ns/frame  {mbs:10.2} MB/s");
}

fn main() {
    let iters: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);
    let reps = 5;
    println!("iters={iters} reps={reps} (min duration reported)");

    let payload = [0x5Au8; 64];
    report("crc16 bitwise (before)", 54, iters, reps, || {
        let _ = black_box(crc16_bitwise(black_box(&payload[..54])));
    });
    report("crc16 slicing-by-4", 54, iters, reps, || {
        let _ = black_box(crc16(black_box(&payload[..54])));
    });

    let req = sample_request();
    let mut req_buf = [0u8; REQUEST_SIZE];
    report("request encode", REQUEST_SIZE as u64, iters, reps, || {
        let _ = black_box(encode_request(black_box(&req), black_box(&mut req_buf)));
    });
    encode_request(&req, &mut req_buf).unwrap();
    report("request decode", REQUEST_SIZE as u64, iters, reps, || {
        let _ = black_box(decode_request(black_box(&req_buf)));
    });

    let rep = sample_reply();
    let mut rep_buf = [0u8; REPLY_SIZE];
    report("reply encode", REPLY_SIZE as u64, iters, reps, || {
        let _ = black_box(encode_reply(black_box(&rep), black_box(&mut rep_buf)));
    });
    encode_reply(&rep, &mut rep_buf).unwrap();
    report("reply decode", REPLY_SIZE as u64, iters, reps, || {
        let _ = black_box(decode_reply(black_box(&rep_buf)));
    });

    let mut sink = 0u64;
    report(
        "request encode+decode+crc",
        (REQUEST_SIZE * 2) as u64,
        iters,
        reps,
        || {
            let _ = black_box(encode_request(black_box(&req), black_box(&mut req_buf)));
            let out = black_box(decode_request(black_box(&req_buf))).unwrap();
            sink = sink.wrapping_add(u64::from(out.seq) ^ u64::from(out.time_us));
        },
    );
    black_box(sink);

    let mut sink = 0u64;
    report(
        "reply encode+decode+crc",
        (REPLY_SIZE * 2) as u64,
        iters,
        reps,
        || {
            let _ = black_box(encode_reply(black_box(&rep), black_box(&mut rep_buf)));
            let out = black_box(decode_reply(black_box(&rep_buf))).unwrap();
            sink = sink.wrapping_add(u64::from(out.seq) ^ u64::from(out.g_pitch as u16));
        },
    );
    black_box(sink);

    let low_req = sample_lowlat_request();
    let mut low_req_buf = [0u8; LOWLAT_REQUEST_SIZE];
    report(
        "lowlat request encode",
        LOWLAT_REQUEST_SIZE as u64,
        iters,
        reps,
        || {
            let _ = black_box(encode_lowlat_request(
                black_box(&low_req),
                black_box(&mut low_req_buf),
            ));
        },
    );
    encode_lowlat_request(&low_req, &mut low_req_buf).unwrap();
    report(
        "lowlat request decode",
        LOWLAT_REQUEST_SIZE as u64,
        iters,
        reps,
        || {
            let _ = black_box(decode_lowlat_request(black_box(&low_req_buf)));
        },
    );

    let low_rep = sample_lowlat_reply();
    let mut low_rep_buf = [0u8; LOWLAT_REPLY_SIZE];
    report(
        "lowlat reply encode",
        LOWLAT_REPLY_SIZE as u64,
        iters,
        reps,
        || {
            let _ = black_box(encode_lowlat_reply(
                black_box(&low_rep),
                black_box(&mut low_rep_buf),
            ));
        },
    );
    encode_lowlat_reply(&low_rep, &mut low_rep_buf).unwrap();
    report(
        "lowlat reply decode",
        LOWLAT_REPLY_SIZE as u64,
        iters,
        reps,
        || {
            let _ = black_box(decode_lowlat_reply(black_box(&low_rep_buf)));
        },
    );

    let mut sink = 0u64;
    report(
        "lowlat request encode+decode+crc",
        (LOWLAT_REQUEST_SIZE * 2) as u64,
        iters,
        reps,
        || {
            let _ = black_box(encode_lowlat_request(
                black_box(&low_req),
                black_box(&mut low_req_buf),
            ));
            let out = black_box(decode_lowlat_request(black_box(&low_req_buf))).unwrap();
            sink = sink.wrapping_add(u64::from(out.seq) ^ u64::from(out.time_us));
        },
    );
    black_box(sink);

    let mut sink = 0u64;
    report(
        "lowlat reply encode+decode+crc",
        (LOWLAT_REPLY_SIZE * 2) as u64,
        iters,
        reps,
        || {
            let _ = black_box(encode_lowlat_reply(
                black_box(&low_rep),
                black_box(&mut low_rep_buf),
            ));
            let out = black_box(decode_lowlat_reply(black_box(&low_rep_buf))).unwrap();
            sink = sink.wrapping_add(u64::from(out.seq) ^ u64::from(out.compute_us));
        },
    );
    black_box(sink);
}
