// Generated from uart.yaml by `cargo run -p potok-generator`.
// Language: Rust. DO NOT EDIT — edit uart.yaml and regenerate.

pub const PROTOCOL_NAME: &str = "potok-uart";
pub const SPEC_VERSION: u32 = 1;
pub const ENDIANNESS: &str = "little";
pub const CRC_POLYNOMIAL: u16 = 0x1021;
pub const CRC_INIT: u16 = 0xffff;
pub const CRC_XOR_OUT: u16 = 0x0000;
pub const CRC_REFLECT_IN: bool = false;
pub const CRC_REFLECT_OUT: bool = false;
pub const FLAG_WIDTH: usize = 1;

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    BURN = 0,
    COAST = 1,
    APOGEE = 2,
    DESCENT = 3,
    LANDED = 4,
}

impl core::convert::TryFrom<u8> for Phase {
    type Error = u8;
    fn try_from(v: u8) -> Result<Self, Self::Error> {
        match v {
            0 => Ok(Phase::BURN),
            1 => Ok(Phase::COAST),
            2 => Ok(Phase::APOGEE),
            3 => Ok(Phase::DESCENT),
            4 => Ok(Phase::LANDED),
            other => Err(other),
        }
    }
}

impl From<Phase> for u8 {
    fn from(v: Phase) -> u8 { v as u8 }
}

pub mod flags {
    pub const GPS_FIX: u8 = 1u8 << 0;
    pub const IMU_OK: u8 = 1u8 << 1;
    pub const BARO_OK: u8 = 1u8 << 2;
    pub const PITOT_OK: u8 = 1u8 << 3;
    pub const AI_READY: u8 = 1u8 << 4;
    pub const CALIBRATED: u8 = 1u8 << 5;
    pub const MOTOR_ARMED: u8 = 1u8 << 6;
    pub const FAULT: u8 = 1u8 << 7;
    #[inline(always)]
    pub fn has(flags: u8, mask: u8) -> bool { flags & mask != 0 }
    #[inline(always)]
    pub fn set(flags: &mut u8, mask: u8) { *flags |= mask; }
    #[inline(always)]
    pub fn clear(flags: &mut u8, mask: u8) { *flags &= !mask; }
}

pub const REQUEST_SIZE: usize = 56;
pub const REQUEST_MAGIC: u16 = 0x504d;
pub const REQUEST_VERSION: u8 = 1u8;
pub const REQUEST_OFF_MAGIC: usize = 0;
pub const REQUEST_OFF_SEQ: usize = 2;
pub const REQUEST_OFF_TIME_US: usize = 4;
pub const REQUEST_OFF_PHASE: usize = 8;
pub const REQUEST_OFF_VERSION: usize = 9;
pub const REQUEST_OFF_FLAGS: usize = 10;
pub const REQUEST_OFF_RESERVED: usize = 11;
pub const REQUEST_OFF_GYRO: usize = 12;
pub const REQUEST_OFF_ACCEL: usize = 24;
pub const REQUEST_OFF_MAG: usize = 36;
pub const REQUEST_OFF_HEIGHT: usize = 48;
pub const REQUEST_OFF_PITOT_DP: usize = 52;
pub const REQUEST_OFF_CRC16: usize = 54;
pub const REQUEST_CRC_OFF: usize = 54;
pub const REQUEST_CRC_LEN: usize = 2;

pub const REPLY_SIZE: usize = 14;
pub const REPLY_MAGIC: u16 = 0x5250;
pub const REPLY_VERSION: u16 = 1u16;
pub const REPLY_OFF_MAGIC: usize = 0;
pub const REPLY_OFF_SEQ: usize = 2;
pub const REPLY_OFF_G_PITCH: usize = 4;
pub const REPLY_G_PITCH_SCALE: f64 = 0.001;
pub const REPLY_OFF_G_YAW: usize = 6;
pub const REPLY_G_YAW_SCALE: f64 = 0.001;
pub const REPLY_OFF_CONFIDENCE: usize = 8;
pub const REPLY_OFF_AI_HEALTHY: usize = 9;
pub const REPLY_OFF_VERSION: usize = 10;
pub const REPLY_OFF_CRC16: usize = 12;
pub const REPLY_CRC_OFF: usize = 12;
pub const REPLY_CRC_LEN: usize = 2;

pub const LOWLAT_REQUEST_SIZE: usize = 42;
pub const LOWLAT_REQUEST_MAGIC: u16 = 0x4c54;
pub const LOWLAT_REQUEST_VERSION: u8 = 1u8;
pub const LOWLAT_REQUEST_OFF_MAGIC: usize = 0;
pub const LOWLAT_REQUEST_OFF_SEQ: usize = 2;
pub const LOWLAT_REQUEST_OFF_TIME_US: usize = 4;
pub const LOWLAT_REQUEST_OFF_PHASE: usize = 8;
pub const LOWLAT_REQUEST_OFF_VERSION: usize = 9;
pub const LOWLAT_REQUEST_OFF_FLAGS: usize = 10;
pub const LOWLAT_REQUEST_OFF_RESERVED: usize = 11;
pub const LOWLAT_REQUEST_OFF_FEATURES: usize = 12;
pub const LOWLAT_REQUEST_FEATURES_SCALES: [f64; 14] = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.0001, 0.0001, 0.0001, 0.01, 1.0, 1.0, 0.1, 1.0];
pub const LOWLAT_REQUEST_FEATURES_NAMES: [&str; 14] = ["gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z", "mag_x", "mag_y", "mag_z", "height", "pressure", "pitot_dp", "temperature", "battery_mv"];
pub const LOWLAT_REQUEST_OFF_CRC16: usize = 40;
pub const LOWLAT_REQUEST_CRC_OFF: usize = 40;
pub const LOWLAT_REQUEST_CRC_LEN: usize = 2;

pub const LOWLAT_REPLY_SIZE: usize = 16;
pub const LOWLAT_REPLY_MAGIC: u16 = 0x4c52;
pub const LOWLAT_REPLY_VERSION: u8 = 1u8;
pub const LOWLAT_REPLY_OFF_MAGIC: usize = 0;
pub const LOWLAT_REPLY_OFF_SEQ: usize = 2;
pub const LOWLAT_REPLY_OFF_G_PITCH: usize = 4;
pub const LOWLAT_REPLY_G_PITCH_SCALE: f64 = 0.001;
pub const LOWLAT_REPLY_OFF_G_YAW: usize = 6;
pub const LOWLAT_REPLY_G_YAW_SCALE: f64 = 0.001;
pub const LOWLAT_REPLY_OFF_CONFIDENCE: usize = 8;
pub const LOWLAT_REPLY_OFF_AI_HEALTHY: usize = 9;
pub const LOWLAT_REPLY_OFF_VERSION: usize = 10;
pub const LOWLAT_REPLY_OFF_COMPUTE_US: usize = 11;
pub const LOWLAT_REPLY_OFF_RESERVED: usize = 12;
pub const LOWLAT_REPLY_OFF_CRC16: usize = 14;
pub const LOWLAT_REPLY_CRC_OFF: usize = 14;
pub const LOWLAT_REPLY_CRC_LEN: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RequestFrame {
    pub seq: u16,
    pub time_us: u32,
    pub phase: Phase,
    pub version: u8,
    pub flags: u8,
    pub reserved: u8,
    pub gyro: [f32; 3],
    pub accel: [f32; 3],
    pub mag: [f32; 3],
    pub height: f32,
    pub pitot_dp: i16,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReplyFrame {
    pub seq: u16,
    pub g_pitch: i16,
    pub g_yaw: i16,
    pub confidence: u8,
    pub ai_healthy: u8,
    pub version: u16,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LowlatRequestFrame {
    pub seq: u16,
    pub time_us: u32,
    pub phase: Phase,
    pub version: u8,
    pub flags: u8,
    pub reserved: u8,
    pub features: [i16; 14],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LowlatReplyFrame {
    pub seq: u16,
    pub g_pitch: i16,
    pub g_yaw: i16,
    pub confidence: u8,
    pub ai_healthy: u8,
    pub version: u8,
    pub compute_us: u8,
    pub reserved: [u8; 2],
}

pub mod codec {
    #[inline(always)]
    pub fn put_u16(buf: &mut [u8], off: usize, v: u16) { let b = v.to_le_bytes(); buf[off] = b[0];buf[off + 1] = b[1]; }
    #[inline(always)]
    pub fn get_u16(buf: &[u8], off: usize) -> u16 { u16::from_le_bytes([buf[off], buf[off + 1]]) }
    #[inline(always)]
    pub fn put_u32(buf: &mut [u8], off: usize, v: u32) { let b = v.to_le_bytes(); buf[off] = b[0];buf[off + 1] = b[1];buf[off + 2] = b[2];buf[off + 3] = b[3]; }
    #[inline(always)]
    pub fn get_u32(buf: &[u8], off: usize) -> u32 { u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]) }
    #[inline(always)]
    pub fn put_u8(buf: &mut [u8], off: usize, v: u8) { buf[off] = v; }
    #[inline(always)]
    pub fn get_u8(buf: &[u8], off: usize) -> u8 { buf[off] }
    #[inline(always)]
    pub fn put_f32(buf: &mut [u8], off: usize, v: f32) { let b = v.to_le_bytes(); buf[off] = b[0];buf[off + 1] = b[1];buf[off + 2] = b[2];buf[off + 3] = b[3]; }
    #[inline(always)]
    pub fn get_f32(buf: &[u8], off: usize) -> f32 { f32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]) }
    #[inline(always)]
    pub fn put_i16(buf: &mut [u8], off: usize, v: i16) { let b = v.to_le_bytes(); buf[off] = b[0];buf[off + 1] = b[1]; }
    #[inline(always)]
    pub fn get_i16(buf: &[u8], off: usize) -> i16 { i16::from_le_bytes([buf[off], buf[off + 1]]) }
}
