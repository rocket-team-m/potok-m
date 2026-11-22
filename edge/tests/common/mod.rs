#![allow(dead_code)]
use potok_edge::fixture::{build_fixture, FixtureConfig};
use potok_edge::frame::{RequestFrame, FLAG_IMU_OK, PHASE_LANDING, PROTOCOL_VERSION};

pub const SEED: u64 = 0x504F544B;

pub fn fixture() -> Vec<u8> {
    build_fixture(FixtureConfig::default(), SEED)
}

pub fn request(seq: u16) -> RequestFrame {
    RequestFrame {
        seq,
        time_us: seq as u32 * 20_000,
        phase: PHASE_LANDING,
        version: PROTOCOL_VERSION,
        flags: FLAG_IMU_OK,
        gyro: [0.02, -0.01, 0.005],
        accel: [0.05, 9.79, -0.03],
        mag: [1.0, 2.0, 3.0],
        height: 1500.0 + seq as f32,
        pitot_dp: (seq % 50) as i16,
    }
}
