//! Local reimplementation of the Potok-M UART frame layout.
//!
//! The canonical layout lives in `protocol/uart.yaml` (owned by another
//! agent). The protocol crate is not published yet, so this module is a
//! deliberately isolated copy behind the `FrameLayout` trait. To switch to
//! the generated crate later, implement `FrameLayout` for it and change one
//! line in `bin/infer.rs`; nothing else in this crate depends on the bytes.

use std::fmt;

pub const REQUEST_LEN: usize = 56;
pub const REPLY_LEN: usize = 14;
pub const MAGIC_REQUEST: u16 = 0x504D; // "PM"
pub const MAGIC_REPLY: u16 = 0x5250; // "RP"
pub const PROTOCOL_VERSION: u8 = 1;
pub const CORRECTION_SCALE: f32 = 0.001;

pub const PHASE_IDLE: u8 = 0;
pub const PHASE_ARMED: u8 = 1;
pub const PHASE_FLIGHT: u8 = 2;
pub const PHASE_LANDING: u8 = 3;
pub const PHASE_FAULT: u8 = 4;

pub const FLAG_GPS_FIX: u8 = 1 << 0;
pub const FLAG_IMU_OK: u8 = 1 << 1;
pub const FLAG_BARO_OK: u8 = 1 << 2;
pub const FLAG_PITOT_OK: u8 = 1 << 3;
pub const FLAG_AI_READY: u8 = 1 << 4;
pub const FLAG_CALIBRATED: u8 = 1 << 5;
pub const FLAG_MOTOR_ARMED: u8 = 1 << 6;
pub const FLAG_FAULT: u8 = 1 << 7;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrameError {
    TooShort { got: usize, want: usize },
    BadMagic(u16),
    VersionMismatch(u8),
    CrcMismatch { got: u16, want: u16 },
}

impl fmt::Display for FrameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooShort { got, want } => write!(f, "frame is {got} bytes, need {want}"),
            Self::BadMagic(m) => write!(f, "bad frame magic 0x{m:04X}"),
            Self::VersionMismatch(v) => write!(f, "protocol version {v} unsupported"),
            Self::CrcMismatch { got, want } => {
                write!(f, "CRC16 mismatch: got 0x{got:04X}, want 0x{want:04X}")
            }
        }
    }
}

impl std::error::Error for FrameError {}

/// CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, no reflection, xor_out 0.
pub fn crc16(bytes: &[u8]) -> u16 {
    let mut crc = 0xFFFFu16;
    for &b in bytes {
        crc ^= (b as u16) << 8;
        for _ in 0..8 {
            crc = if crc & 0x8000 != 0 {
                (crc << 1) ^ 0x1021
            } else {
                crc << 1
            };
        }
    }
    crc
}

#[derive(Debug, Clone, PartialEq)]
pub struct RequestFrame {
    pub seq: u16,
    pub time_us: u32,
    pub phase: u8,
    pub version: u8,
    pub flags: u8,
    pub gyro: [f32; 3],
    pub accel: [f32; 3],
    pub mag: [f32; 3],
    pub height: f32,
    pub pitot_dp: i16,
}

impl RequestFrame {
    /// The eight model features: gyro(3), accel(3), pitot dp, height.
    /// Scaling is fixed here so firmware, simulator and edge agree.
    pub fn features(&self) -> [f32; 8] {
        [
            self.gyro[0],
            self.gyro[1],
            self.gyro[2],
            self.accel[0],
            self.accel[1],
            self.accel[2],
            self.pitot_dp as f32 * 0.01,
            self.height,
        ]
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplyFrame {
    pub seq: u16,
    /// Signed correction in radians x 1000 (scale 0.001).
    pub g_pitch: i16,
    pub g_yaw: i16,
    pub confidence: u8,
    pub ai_healthy: u8,
    pub version: u16,
}

impl ReplyFrame {
    pub fn muted(seq: u16, healthy: bool) -> Self {
        Self {
            seq,
            g_pitch: 0,
            g_yaw: 0,
            confidence: 0,
            ai_healthy: u8::from(healthy),
            version: PROTOCOL_VERSION as u16,
        }
    }
}

/// The only protocol surface the runtime knows about.
pub trait FrameLayout {
    fn decode_request(&self, bytes: &[u8]) -> Result<RequestFrame, FrameError>;
    fn encode_reply(&self, reply: &ReplyFrame, out: &mut [u8]) -> Result<(), FrameError>;
}

/// Canonical `potok-uart` v1 layout from `protocol/uart.yaml`.
#[derive(Debug, Default, Clone, Copy)]
pub struct PotokUartV1;

impl FrameLayout for PotokUartV1 {
    fn decode_request(&self, bytes: &[u8]) -> Result<RequestFrame, FrameError> {
        if bytes.len() < REQUEST_LEN {
            return Err(FrameError::TooShort { got: bytes.len(), want: REQUEST_LEN });
        }
        let magic = u16::from_le_bytes([bytes[0], bytes[1]]);
        if magic != MAGIC_REQUEST {
            return Err(FrameError::BadMagic(magic));
        }
        let want = crc16(&bytes[..REQUEST_LEN - 2]);
        let got = u16::from_le_bytes([bytes[REQUEST_LEN - 2], bytes[REQUEST_LEN - 1]]);
        if got != want {
            return Err(FrameError::CrcMismatch { got, want });
        }
        let version = bytes[9];
        if version != PROTOCOL_VERSION {
            return Err(FrameError::VersionMismatch(version));
        }
        let seq = u16::from_le_bytes([bytes[2], bytes[3]]);
        let time_us = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let phase = bytes[8];
        let flags = bytes[10];
        let f32_at = |off: usize| -> f32 {
            f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap())
        };
        Ok(RequestFrame {
            seq,
            time_us,
            phase,
            version,
            flags,
            gyro: [f32_at(12), f32_at(16), f32_at(20)],
            accel: [f32_at(24), f32_at(28), f32_at(32)],
            mag: [f32_at(36), f32_at(40), f32_at(44)],
            height: f32_at(48),
            pitot_dp: i16::from_le_bytes([bytes[52], bytes[53]]),
        })
    }

    fn encode_reply(&self, reply: &ReplyFrame, out: &mut [u8]) -> Result<(), FrameError> {
        if out.len() < REPLY_LEN {
            return Err(FrameError::TooShort { got: out.len(), want: REPLY_LEN });
        }
        out.fill(0);
        out[0..2].copy_from_slice(&MAGIC_REPLY.to_le_bytes());
        out[2..4].copy_from_slice(&reply.seq.to_le_bytes());
        out[4..6].copy_from_slice(&reply.g_pitch.to_le_bytes());
        out[6..8].copy_from_slice(&reply.g_yaw.to_le_bytes());
        out[8] = reply.confidence;
        out[9] = reply.ai_healthy;
        out[10..12].copy_from_slice(&reply.version.to_le_bytes());
        let crc = crc16(&out[..REPLY_LEN - 2]);
        out[12..14].copy_from_slice(&crc.to_le_bytes());
        Ok(())
    }
}

/// Build a request frame for tests and synthetic benchmarks.
pub fn build_request(req: &RequestFrame, out: &mut [u8]) -> Result<(), FrameError> {
    if out.len() < REQUEST_LEN {
        return Err(FrameError::TooShort { got: out.len(), want: REQUEST_LEN });
    }
    out.fill(0);
    out[0..2].copy_from_slice(&MAGIC_REQUEST.to_le_bytes());
    out[2..4].copy_from_slice(&req.seq.to_le_bytes());
    out[4..8].copy_from_slice(&req.time_us.to_le_bytes());
    out[8] = req.phase;
    out[9] = req.version;
    out[10] = req.flags;
    let mut at = 12usize;
    for v in req.gyro.iter().chain(req.accel.iter()).chain(req.mag.iter()) {
        out[at..at + 4].copy_from_slice(&v.to_le_bytes());
        at += 4;
    }
    out[at..at + 4].copy_from_slice(&req.height.to_le_bytes());
    at += 4;
    out[at..at + 2].copy_from_slice(&req.pitot_dp.to_le_bytes());
    let crc = crc16(&out[..REQUEST_LEN - 2]);
    out[REQUEST_LEN - 2..REQUEST_LEN].copy_from_slice(&crc.to_le_bytes());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> RequestFrame {
        RequestFrame {
            seq: 7,
            time_us: 1234,
            phase: PHASE_LANDING,
            version: PROTOCOL_VERSION,
            flags: FLAG_IMU_OK,
            gyro: [0.1, -0.2, 0.3],
            accel: [0.0, 9.81, 0.0],
            mag: [1.0, 2.0, 3.0],
            height: 1200.5,
            pitot_dp: -42,
        }
    }

    #[test]
    fn request_roundtrip_and_crc_detection() {
        let layout = PotokUartV1;
        let mut buf = [0u8; REQUEST_LEN];
        build_request(&sample(), &mut buf).unwrap();
        assert_eq!(layout.decode_request(&buf).unwrap(), sample());
        buf[20] ^= 0x01;
        assert!(matches!(layout.decode_request(&buf), Err(FrameError::CrcMismatch { .. })));
    }

    #[test]
    fn reply_roundtrip() {
        let layout = PotokUartV1;
        let reply = ReplyFrame {
            seq: 9,
            g_pitch: -12,
            g_yaw: 34,
            confidence: 200,
            ai_healthy: 1,
            version: PROTOCOL_VERSION as u16,
        };
        let mut buf = [0u8; REPLY_LEN];
        layout.encode_reply(&reply, &mut buf).unwrap();
        assert_eq!(u16::from_le_bytes([buf[4], buf[5]]), (-12i16) as u16);
        assert_eq!(crc16(&buf[..12]), u16::from_le_bytes([buf[12], buf[13]]));
    }
}
