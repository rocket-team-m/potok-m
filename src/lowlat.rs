//! `FrameLayout` adapter for the canonical lowlat profile.
//!
//! All wire bytes go through `potok_protocol`'s generated lowlat codec:
//! magic/CRC/phase/version are checked there, the 14 int16 features are
//! dequantized with `LOWLAT_REQUEST_FEATURES_SCALES`, and the reply is
//! quantized back to scaled int16 with a bounded `compute_us` (100 us units).
//! No allocation happens in `decode_request` or `encode_reply`.

use crate::frame::{FrameError, FrameLayout, ReplyFrame, RequestFrame, CORRECTION_SCALE};
use potok_protocol::{
    decode_lowlat_request, dequantize_i16, encode_lowlat_reply, quantize_i16, Error,
    LowlatReplyFrame, LOWLAT_REPLY_G_PITCH_SCALE, LOWLAT_REPLY_G_YAW_SCALE, LOWLAT_REPLY_SIZE,
    LOWLAT_REPLY_VERSION, LOWLAT_REQUEST_FEATURES_SCALES, LOWLAT_REQUEST_SIZE,
};
use std::cell::Cell;
use std::time::Duration;

fn map_err(e: Error) -> FrameError {
    match e {
        Error::TooShort { need, got } | Error::BufferTooSmall { need, got } => {
            FrameError::TooShort { got, want: need }
        }
        Error::BadMagic { got, .. } => FrameError::BadMagic(got),
        Error::BadCrc { got, expected } => FrameError::CrcMismatch {
            got,
            want: expected,
        },
        Error::VersionMismatch { got, .. } => FrameError::VersionMismatch(got as u8),
        Error::BadPhase(p) => FrameError::BadPhase(p),
    }
}

#[derive(Debug, Default, Clone)]
pub struct LowlatLayout {
    /// Dispatch time in 100 us units, saturating at 255 (25.5 ms).
    compute_us: Cell<u32>,
}

impl LowlatLayout {
    pub fn new() -> Self {
        Self {
            compute_us: Cell::new(0),
        }
    }

    pub fn set_compute(&self, elapsed: Duration) {
        let units = (elapsed.as_micros() / 100).min(255) as u32;
        self.compute_us.set(units);
    }
}

impl FrameLayout for LowlatLayout {
    fn request_len(&self) -> usize {
        LOWLAT_REQUEST_SIZE
    }
    fn reply_len(&self) -> usize {
        LOWLAT_REPLY_SIZE
    }

    fn decode_request(&self, bytes: &[u8]) -> Result<RequestFrame, FrameError> {
        let f = decode_lowlat_request(bytes).map_err(map_err)?;
        let s = LOWLAT_REQUEST_FEATURES_SCALES;
        let mut v = [0.0f64; 14];
        for (i, raw) in f.features.iter().enumerate() {
            v[i] = dequantize_i16(*raw, s[i]);
        }
        Ok(RequestFrame {
            seq: f.seq,
            time_us: f.time_us,
            phase: f.phase as u8,
            version: f.version,
            flags: f.flags,
            gyro: [v[0] as f32, v[1] as f32, v[2] as f32],
            accel: [v[3] as f32, v[4] as f32, v[5] as f32],
            mag: [v[6] as f32, v[7] as f32, v[8] as f32],
            height: v[9] as f32,
            // Lowlat carries pressure/temperature/battery_mv at 10/12/13;
            // the model input uses the standard 8 sensors, height and pitot.
            pitot_dp: f.features[11],
        })
    }

    fn encode_reply(&self, reply: &ReplyFrame, out: &mut [u8]) -> Result<(), FrameError> {
        let f = LowlatReplyFrame {
            seq: reply.seq,
            g_pitch: quantize_i16(
                f64::from(reply.g_pitch) * f64::from(CORRECTION_SCALE),
                LOWLAT_REPLY_G_PITCH_SCALE,
            ),
            g_yaw: quantize_i16(
                f64::from(reply.g_yaw) * f64::from(CORRECTION_SCALE),
                LOWLAT_REPLY_G_YAW_SCALE,
            ),
            confidence: reply.confidence,
            ai_healthy: reply.ai_healthy,
            version: LOWLAT_REPLY_VERSION,
            compute_us: self.compute_us.get() as u8,
            reserved: [0, 0],
        };
        encode_lowlat_reply(&f, out).map_err(map_err)
    }

    fn note_compute(&self, elapsed: Duration) {
        self.set_compute(elapsed);
    }
}
