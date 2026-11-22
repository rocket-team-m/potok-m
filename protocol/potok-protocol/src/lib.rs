//! Potok-M UART protocol: byte-exact encode/decode of the frames in `uart.yaml`.
//!
//! The wire types, offsets, magics, enum and flags are generated; this crate
//! supplies behaviour (CRC, bounds checks, version/phase checks, seq tracking)
//! without allocating on the hot path. `RequestView`/`ReplyView` decode straight
//! from the input slice with no owned copy.

#[allow(dead_code)]
mod generated {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../generated/protocol.rs"
    ));
}
pub use generated::*;

const _: () = assert!(!CRC_REFLECT_IN && !CRC_REFLECT_OUT);
const _: () = assert!(REQUEST_CRC_OFF + REQUEST_CRC_LEN == REQUEST_SIZE);
const _: () = assert!(REPLY_CRC_OFF + REPLY_CRC_LEN == REPLY_SIZE);
const _: () = assert!(LOWLAT_REQUEST_CRC_OFF + LOWLAT_REQUEST_CRC_LEN == LOWLAT_REQUEST_SIZE);
const _: () = assert!(LOWLAT_REPLY_CRC_OFF + LOWLAT_REPLY_CRC_LEN == LOWLAT_REPLY_SIZE);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    BufferTooSmall { need: usize, got: usize },
    TooShort { need: usize, got: usize },
    BadMagic { expected: u16, got: u16 },
    BadCrc { expected: u16, got: u16 },
    BadPhase(u8),
    VersionMismatch { expected: u16, got: u16 },
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::BufferTooSmall { need, got } => {
                write!(f, "output buffer: need {need} bytes, got {got}")
            }
            Error::TooShort { need, got } => {
                write!(f, "frame truncated: need {need} bytes, got {got}")
            }
            Error::BadMagic { expected, got } => {
                write!(f, "bad magic: expected {expected:#06x}, got {got:#06x}")
            }
            Error::BadCrc { expected, got } => {
                write!(f, "bad CRC: expected {expected:#06x}, got {got:#06x}")
            }
            Error::BadPhase(v) => write!(f, "unknown phase {v}"),
            Error::VersionMismatch { expected, got } => {
                write!(f, "version mismatch: expected {expected}, got {got}")
            }
        }
    }
}

impl std::error::Error for Error {}

/// CRC lookup table built from the YAML polynomial at compile time.
const fn make_crc_table() -> [u16; 256] {
    let mut table = [0u16; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut crc = (i as u16) << 8;
        let mut bit = 0usize;
        while bit < 8 {
            crc = if crc & 0x8000 != 0 {
                (crc << 1) ^ CRC_POLYNOMIAL
            } else {
                crc << 1
            };
            bit += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

pub const CRC_TABLE: [u16; 256] = make_crc_table();

const fn crc_advance(x: u16) -> u16 {
    CRC_TABLE[(x >> 8) as usize] ^ (x << 8)
}

/// Positional tables for slicing-by-4. `T0[b]` is one CRC step for byte `b`;
/// `T1[b]` is two steps (`b` followed by a zero byte). Folding two wire bytes
/// into the state and applying `T1/T0` consumes a pair in two table loads.
pub const CRC_TABLE2: [[u16; 256]; 2] = make_crc_table2();

const fn make_crc_table2() -> [[u16; 256]; 2] {
    let mut tables = [[0u16; 256]; 2];
    let mut i = 0usize;
    while i < 256 {
        tables[0][i] = CRC_TABLE[i];
        tables[1][i] = crc_advance(tables[0][i]);
        i += 1;
    }
    tables
}

#[inline(always)]
fn crc_pair(crc: u16, hi: u8, lo: u8) -> u16 {
    let d = crc ^ (u16::from(hi) << 8) ^ u16::from(lo);
    CRC_TABLE2[1][(d >> 8) as usize] ^ CRC_TABLE2[0][(d & 0xff) as usize]
}

/// CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, no reflection, no final XOR.
/// Covers every byte before the trailing CRC field.
#[inline]
pub fn crc16(data: &[u8]) -> u16 {
    let mut crc = CRC_INIT;
    let mut chunks = data.chunks_exact(4);
    for c in &mut chunks {
        crc = crc_pair(crc, c[0], c[1]);
        crc = crc_pair(crc, c[2], c[3]);
    }
    for &b in chunks.remainder() {
        crc = CRC_TABLE[((crc >> 8) ^ u16::from(b)) as usize] ^ (crc << 8);
    }
    crc ^ CRC_XOR_OUT
}

#[inline]
fn check_magic_crc(buf: &[u8], magic: u16, crc_off: usize) -> Result<(), Error> {
    let got_magic = codec::get_u16(buf, 0);
    if got_magic != magic {
        return Err(Error::BadMagic {
            expected: magic,
            got: got_magic,
        });
    }
    let expected = crc16(&buf[..crc_off]);
    let got = codec::get_u16(buf, crc_off);
    if got != expected {
        return Err(Error::BadCrc { expected, got });
    }
    Ok(())
}

#[inline]
pub fn encode_request(frame: &RequestFrame, out: &mut [u8]) -> Result<(), Error> {
    if out.len() < REQUEST_SIZE {
        return Err(Error::BufferTooSmall {
            need: REQUEST_SIZE,
            got: out.len(),
        });
    }
    codec::put_u16(out, REQUEST_OFF_MAGIC, REQUEST_MAGIC);
    codec::put_u16(out, REQUEST_OFF_SEQ, frame.seq);
    codec::put_u32(out, REQUEST_OFF_TIME_US, frame.time_us);
    codec::put_u8(out, REQUEST_OFF_PHASE, u8::from(frame.phase));
    codec::put_u8(out, REQUEST_OFF_VERSION, frame.version);
    codec::put_u8(out, REQUEST_OFF_FLAGS, frame.flags);
    codec::put_u8(out, REQUEST_OFF_RESERVED, frame.reserved);
    codec::put_f32(out, REQUEST_OFF_GYRO, frame.gyro[0]);
    codec::put_f32(out, REQUEST_OFF_GYRO + 4, frame.gyro[1]);
    codec::put_f32(out, REQUEST_OFF_GYRO + 8, frame.gyro[2]);
    codec::put_f32(out, REQUEST_OFF_ACCEL, frame.accel[0]);
    codec::put_f32(out, REQUEST_OFF_ACCEL + 4, frame.accel[1]);
    codec::put_f32(out, REQUEST_OFF_ACCEL + 8, frame.accel[2]);
    codec::put_f32(out, REQUEST_OFF_MAG, frame.mag[0]);
    codec::put_f32(out, REQUEST_OFF_MAG + 4, frame.mag[1]);
    codec::put_f32(out, REQUEST_OFF_MAG + 8, frame.mag[2]);
    codec::put_f32(out, REQUEST_OFF_HEIGHT, frame.height);
    codec::put_i16(out, REQUEST_OFF_PITOT_DP, frame.pitot_dp);
    codec::put_u16(out, REQUEST_CRC_OFF, crc16(&out[..REQUEST_CRC_OFF]));
    Ok(())
}

/// Borrowed view over a validated request frame. Accessors read the original
/// slice; no field is copied until `to_owned()` is called.
#[derive(Clone, Copy, Debug)]
pub struct RequestView<'a> {
    buf: &'a [u8],
    phase: Phase,
}

impl<'a> RequestView<'a> {
    #[inline]
    pub fn new(buf: &'a [u8]) -> Result<Self, Error> {
        if buf.len() < REQUEST_SIZE {
            return Err(Error::TooShort {
                need: REQUEST_SIZE,
                got: buf.len(),
            });
        }
        check_magic_crc(buf, REQUEST_MAGIC, REQUEST_CRC_OFF)?;
        let phase_raw = codec::get_u8(buf, REQUEST_OFF_PHASE);
        let phase = Phase::try_from(phase_raw).map_err(|_| Error::BadPhase(phase_raw))?;
        let version = codec::get_u8(buf, REQUEST_OFF_VERSION);
        if version != REQUEST_VERSION {
            return Err(Error::VersionMismatch {
                expected: u16::from(REQUEST_VERSION),
                got: u16::from(version),
            });
        }
        Ok(Self { buf, phase })
    }

    #[inline(always)]
    pub fn as_bytes(&self) -> &'a [u8] {
        self.buf
    }
    #[inline(always)]
    pub fn seq(&self) -> u16 {
        codec::get_u16(self.buf, REQUEST_OFF_SEQ)
    }
    #[inline(always)]
    pub fn time_us(&self) -> u32 {
        codec::get_u32(self.buf, REQUEST_OFF_TIME_US)
    }
    #[inline(always)]
    pub fn phase(&self) -> Phase {
        self.phase
    }
    #[inline(always)]
    pub fn version(&self) -> u8 {
        codec::get_u8(self.buf, REQUEST_OFF_VERSION)
    }
    #[inline(always)]
    pub fn flags(&self) -> u8 {
        codec::get_u8(self.buf, REQUEST_OFF_FLAGS)
    }
    #[inline(always)]
    pub fn reserved(&self) -> u8 {
        codec::get_u8(self.buf, REQUEST_OFF_RESERVED)
    }
    #[inline(always)]
    pub fn gyro(&self) -> [f32; 3] {
        [
            codec::get_f32(self.buf, REQUEST_OFF_GYRO),
            codec::get_f32(self.buf, REQUEST_OFF_GYRO + 4),
            codec::get_f32(self.buf, REQUEST_OFF_GYRO + 8),
        ]
    }
    #[inline(always)]
    pub fn accel(&self) -> [f32; 3] {
        [
            codec::get_f32(self.buf, REQUEST_OFF_ACCEL),
            codec::get_f32(self.buf, REQUEST_OFF_ACCEL + 4),
            codec::get_f32(self.buf, REQUEST_OFF_ACCEL + 8),
        ]
    }
    #[inline(always)]
    pub fn mag(&self) -> [f32; 3] {
        [
            codec::get_f32(self.buf, REQUEST_OFF_MAG),
            codec::get_f32(self.buf, REQUEST_OFF_MAG + 4),
            codec::get_f32(self.buf, REQUEST_OFF_MAG + 8),
        ]
    }
    #[inline(always)]
    pub fn height(&self) -> f32 {
        codec::get_f32(self.buf, REQUEST_OFF_HEIGHT)
    }
    #[inline(always)]
    pub fn pitot_dp(&self) -> i16 {
        codec::get_i16(self.buf, REQUEST_OFF_PITOT_DP)
    }

    #[inline]
    pub fn to_owned(&self) -> RequestFrame {
        RequestFrame {
            seq: self.seq(),
            time_us: self.time_us(),
            phase: self.phase,
            version: self.version(),
            flags: self.flags(),
            reserved: self.reserved(),
            gyro: self.gyro(),
            accel: self.accel(),
            mag: self.mag(),
            height: self.height(),
            pitot_dp: self.pitot_dp(),
        }
    }
}

#[inline]
pub fn decode_request(buf: &[u8]) -> Result<RequestFrame, Error> {
    if buf.len() < REQUEST_SIZE {
        return Err(Error::TooShort {
            need: REQUEST_SIZE,
            got: buf.len(),
        });
    }
    check_magic_crc(buf, REQUEST_MAGIC, REQUEST_CRC_OFF)?;
    let phase_raw = codec::get_u8(buf, REQUEST_OFF_PHASE);
    let phase = Phase::try_from(phase_raw).map_err(|_| Error::BadPhase(phase_raw))?;
    let version = codec::get_u8(buf, REQUEST_OFF_VERSION);
    if version != REQUEST_VERSION {
        return Err(Error::VersionMismatch {
            expected: u16::from(REQUEST_VERSION),
            got: u16::from(version),
        });
    }
    Ok(RequestFrame {
        seq: codec::get_u16(buf, REQUEST_OFF_SEQ),
        time_us: codec::get_u32(buf, REQUEST_OFF_TIME_US),
        phase,
        version,
        flags: codec::get_u8(buf, REQUEST_OFF_FLAGS),
        reserved: codec::get_u8(buf, REQUEST_OFF_RESERVED),
        gyro: [
            codec::get_f32(buf, REQUEST_OFF_GYRO),
            codec::get_f32(buf, REQUEST_OFF_GYRO + 4),
            codec::get_f32(buf, REQUEST_OFF_GYRO + 8),
        ],
        accel: [
            codec::get_f32(buf, REQUEST_OFF_ACCEL),
            codec::get_f32(buf, REQUEST_OFF_ACCEL + 4),
            codec::get_f32(buf, REQUEST_OFF_ACCEL + 8),
        ],
        mag: [
            codec::get_f32(buf, REQUEST_OFF_MAG),
            codec::get_f32(buf, REQUEST_OFF_MAG + 4),
            codec::get_f32(buf, REQUEST_OFF_MAG + 8),
        ],
        height: codec::get_f32(buf, REQUEST_OFF_HEIGHT),
        pitot_dp: codec::get_i16(buf, REQUEST_OFF_PITOT_DP),
    })
}

#[inline]
pub fn encode_reply(frame: &ReplyFrame, out: &mut [u8]) -> Result<(), Error> {
    if out.len() < REPLY_SIZE {
        return Err(Error::BufferTooSmall {
            need: REPLY_SIZE,
            got: out.len(),
        });
    }
    codec::put_u16(out, REPLY_OFF_MAGIC, REPLY_MAGIC);
    codec::put_u16(out, REPLY_OFF_SEQ, frame.seq);
    codec::put_i16(out, REPLY_OFF_G_PITCH, frame.g_pitch);
    codec::put_i16(out, REPLY_OFF_G_YAW, frame.g_yaw);
    codec::put_u8(out, REPLY_OFF_CONFIDENCE, frame.confidence);
    codec::put_u8(out, REPLY_OFF_AI_HEALTHY, frame.ai_healthy);
    codec::put_u16(out, REPLY_OFF_VERSION, frame.version);
    codec::put_u16(out, REPLY_CRC_OFF, crc16(&out[..REPLY_CRC_OFF]));
    Ok(())
}

/// Borrowed view over a validated reply frame.
#[derive(Clone, Copy, Debug)]
pub struct ReplyView<'a> {
    buf: &'a [u8],
}

impl<'a> ReplyView<'a> {
    #[inline]
    pub fn new(buf: &'a [u8]) -> Result<Self, Error> {
        if buf.len() < REPLY_SIZE {
            return Err(Error::TooShort {
                need: REPLY_SIZE,
                got: buf.len(),
            });
        }
        check_magic_crc(buf, REPLY_MAGIC, REPLY_CRC_OFF)?;
        let version = codec::get_u16(buf, REPLY_OFF_VERSION);
        if version != REPLY_VERSION {
            return Err(Error::VersionMismatch {
                expected: REPLY_VERSION,
                got: version,
            });
        }
        Ok(Self { buf })
    }

    #[inline(always)]
    pub fn as_bytes(&self) -> &'a [u8] {
        self.buf
    }
    #[inline(always)]
    pub fn seq(&self) -> u16 {
        codec::get_u16(self.buf, REPLY_OFF_SEQ)
    }
    #[inline(always)]
    pub fn g_pitch(&self) -> i16 {
        codec::get_i16(self.buf, REPLY_OFF_G_PITCH)
    }
    #[inline(always)]
    pub fn g_yaw(&self) -> i16 {
        codec::get_i16(self.buf, REPLY_OFF_G_YAW)
    }
    #[inline(always)]
    pub fn confidence(&self) -> u8 {
        codec::get_u8(self.buf, REPLY_OFF_CONFIDENCE)
    }
    #[inline(always)]
    pub fn ai_healthy(&self) -> u8 {
        codec::get_u8(self.buf, REPLY_OFF_AI_HEALTHY)
    }
    #[inline(always)]
    pub fn version(&self) -> u16 {
        codec::get_u16(self.buf, REPLY_OFF_VERSION)
    }

    #[inline]
    pub fn to_owned(&self) -> ReplyFrame {
        ReplyFrame {
            seq: self.seq(),
            g_pitch: self.g_pitch(),
            g_yaw: self.g_yaw(),
            confidence: self.confidence(),
            ai_healthy: self.ai_healthy(),
            version: self.version(),
        }
    }
}

#[inline]
pub fn decode_reply(buf: &[u8]) -> Result<ReplyFrame, Error> {
    if buf.len() < REPLY_SIZE {
        return Err(Error::TooShort {
            need: REPLY_SIZE,
            got: buf.len(),
        });
    }
    check_magic_crc(buf, REPLY_MAGIC, REPLY_CRC_OFF)?;
    let version = codec::get_u16(buf, REPLY_OFF_VERSION);
    if version != REPLY_VERSION {
        return Err(Error::VersionMismatch {
            expected: REPLY_VERSION,
            got: version,
        });
    }
    Ok(ReplyFrame {
        seq: codec::get_u16(buf, REPLY_OFF_SEQ),
        g_pitch: codec::get_i16(buf, REPLY_OFF_G_PITCH),
        g_yaw: codec::get_i16(buf, REPLY_OFF_G_YAW),
        confidence: codec::get_u8(buf, REPLY_OFF_CONFIDENCE),
        ai_healthy: codec::get_u8(buf, REPLY_OFF_AI_HEALTHY),
        version,
    })
}

// ---------------------------------------------------------------------------
// lowlat profile (SPEC_PROTOCOL_LOWLAT.md). Additive: compat frames are intact.
// ---------------------------------------------------------------------------

#[inline]
pub fn encode_lowlat_request(frame: &LowlatRequestFrame, out: &mut [u8]) -> Result<(), Error> {
    if out.len() < LOWLAT_REQUEST_SIZE {
        return Err(Error::BufferTooSmall { need: LOWLAT_REQUEST_SIZE, got: out.len() });
    }
    codec::put_u16(out, LOWLAT_REQUEST_OFF_MAGIC, LOWLAT_REQUEST_MAGIC);
    codec::put_u16(out, LOWLAT_REQUEST_OFF_SEQ, frame.seq);
    codec::put_u32(out, LOWLAT_REQUEST_OFF_TIME_US, frame.time_us);
    codec::put_u8(out, LOWLAT_REQUEST_OFF_PHASE, u8::from(frame.phase));
    codec::put_u8(out, LOWLAT_REQUEST_OFF_VERSION, frame.version);
    codec::put_u8(out, LOWLAT_REQUEST_OFF_FLAGS, frame.flags);
    codec::put_u8(out, LOWLAT_REQUEST_OFF_RESERVED, frame.reserved);
    for (i, &feature) in frame.features.iter().enumerate() {
        codec::put_i16(out, LOWLAT_REQUEST_OFF_FEATURES + 2 * i, feature);
    }
    codec::put_u16(out, LOWLAT_REQUEST_CRC_OFF, crc16(&out[..LOWLAT_REQUEST_CRC_OFF]));
    Ok(())
}

/// Borrowed, validated view over a lowlat request frame.
#[derive(Clone, Copy, Debug)]
pub struct LowlatRequestView<'a> {
    buf: &'a [u8],
    phase: Phase,
}

impl<'a> LowlatRequestView<'a> {
    #[inline]
    pub fn new(buf: &'a [u8]) -> Result<Self, Error> {
        if buf.len() < LOWLAT_REQUEST_SIZE {
            return Err(Error::TooShort { need: LOWLAT_REQUEST_SIZE, got: buf.len() });
        }
        check_magic_crc(buf, LOWLAT_REQUEST_MAGIC, LOWLAT_REQUEST_CRC_OFF)?;
        let phase_raw = codec::get_u8(buf, LOWLAT_REQUEST_OFF_PHASE);
        let phase = Phase::try_from(phase_raw).map_err(|_| Error::BadPhase(phase_raw))?;
        let version = codec::get_u8(buf, LOWLAT_REQUEST_OFF_VERSION);
        if version != LOWLAT_REQUEST_VERSION {
            return Err(Error::VersionMismatch {
                expected: u16::from(LOWLAT_REQUEST_VERSION),
                got: u16::from(version),
            });
        }
        Ok(Self { buf, phase })
    }

    #[inline(always)]
    pub fn as_bytes(&self) -> &'a [u8] { self.buf }
    #[inline(always)]
    pub fn seq(&self) -> u16 { codec::get_u16(self.buf, LOWLAT_REQUEST_OFF_SEQ) }
    #[inline(always)]
    pub fn time_us(&self) -> u32 { codec::get_u32(self.buf, LOWLAT_REQUEST_OFF_TIME_US) }
    #[inline(always)]
    pub fn phase(&self) -> Phase { self.phase }
    #[inline(always)]
    pub fn version(&self) -> u8 { codec::get_u8(self.buf, LOWLAT_REQUEST_OFF_VERSION) }
    #[inline(always)]
    pub fn flags(&self) -> u8 { codec::get_u8(self.buf, LOWLAT_REQUEST_OFF_FLAGS) }
    #[inline(always)]
    pub fn reserved(&self) -> u8 { codec::get_u8(self.buf, LOWLAT_REQUEST_OFF_RESERVED) }
    #[inline]
    pub fn features(&self) -> [i16; 14] {
        let mut out = [0i16; 14];
        for (i, item) in out.iter_mut().enumerate() {
            *item = codec::get_i16(self.buf, LOWLAT_REQUEST_OFF_FEATURES + 2 * i);
        }
        out
    }

    #[inline]
    pub fn to_owned(&self) -> LowlatRequestFrame {
        LowlatRequestFrame {
            seq: self.seq(),
            time_us: self.time_us(),
            phase: self.phase,
            version: self.version(),
            flags: self.flags(),
            reserved: self.reserved(),
            features: self.features(),
        }
    }
}

#[inline]
pub fn decode_lowlat_request(buf: &[u8]) -> Result<LowlatRequestFrame, Error> {
    LowlatRequestView::new(buf).map(|v| v.to_owned())
}

#[inline]
pub fn encode_lowlat_reply(frame: &LowlatReplyFrame, out: &mut [u8]) -> Result<(), Error> {
    if out.len() < LOWLAT_REPLY_SIZE {
        return Err(Error::BufferTooSmall { need: LOWLAT_REPLY_SIZE, got: out.len() });
    }
    codec::put_u16(out, LOWLAT_REPLY_OFF_MAGIC, LOWLAT_REPLY_MAGIC);
    codec::put_u16(out, LOWLAT_REPLY_OFF_SEQ, frame.seq);
    codec::put_i16(out, LOWLAT_REPLY_OFF_G_PITCH, frame.g_pitch);
    codec::put_i16(out, LOWLAT_REPLY_OFF_G_YAW, frame.g_yaw);
    codec::put_u8(out, LOWLAT_REPLY_OFF_CONFIDENCE, frame.confidence);
    codec::put_u8(out, LOWLAT_REPLY_OFF_AI_HEALTHY, frame.ai_healthy);
    codec::put_u8(out, LOWLAT_REPLY_OFF_VERSION, frame.version);
    codec::put_u8(out, LOWLAT_REPLY_OFF_COMPUTE_US, frame.compute_us);
    codec::put_u8(out, LOWLAT_REPLY_OFF_RESERVED, frame.reserved[0]);
    codec::put_u8(out, LOWLAT_REPLY_OFF_RESERVED + 1, frame.reserved[1]);
    codec::put_u16(out, LOWLAT_REPLY_CRC_OFF, crc16(&out[..LOWLAT_REPLY_CRC_OFF]));
    Ok(())
}

/// Borrowed, validated view over a lowlat reply frame.
#[derive(Clone, Copy, Debug)]
pub struct LowlatReplyView<'a> {
    buf: &'a [u8],
}

impl<'a> LowlatReplyView<'a> {
    #[inline]
    pub fn new(buf: &'a [u8]) -> Result<Self, Error> {
        if buf.len() < LOWLAT_REPLY_SIZE {
            return Err(Error::TooShort { need: LOWLAT_REPLY_SIZE, got: buf.len() });
        }
        check_magic_crc(buf, LOWLAT_REPLY_MAGIC, LOWLAT_REPLY_CRC_OFF)?;
        let version = codec::get_u8(buf, LOWLAT_REPLY_OFF_VERSION);
        if version != LOWLAT_REPLY_VERSION {
            return Err(Error::VersionMismatch {
                expected: u16::from(LOWLAT_REPLY_VERSION),
                got: u16::from(version),
            });
        }
        Ok(Self { buf })
    }

    #[inline(always)]
    pub fn as_bytes(&self) -> &'a [u8] { self.buf }
    #[inline(always)]
    pub fn seq(&self) -> u16 { codec::get_u16(self.buf, LOWLAT_REPLY_OFF_SEQ) }
    #[inline(always)]
    pub fn g_pitch(&self) -> i16 { codec::get_i16(self.buf, LOWLAT_REPLY_OFF_G_PITCH) }
    #[inline(always)]
    pub fn g_yaw(&self) -> i16 { codec::get_i16(self.buf, LOWLAT_REPLY_OFF_G_YAW) }
    #[inline(always)]
    pub fn confidence(&self) -> u8 { codec::get_u8(self.buf, LOWLAT_REPLY_OFF_CONFIDENCE) }
    #[inline(always)]
    pub fn ai_healthy(&self) -> u8 { codec::get_u8(self.buf, LOWLAT_REPLY_OFF_AI_HEALTHY) }
    #[inline(always)]
    pub fn version(&self) -> u8 { codec::get_u8(self.buf, LOWLAT_REPLY_OFF_VERSION) }
    #[inline(always)]
    pub fn compute_us(&self) -> u8 { codec::get_u8(self.buf, LOWLAT_REPLY_OFF_COMPUTE_US) }
    #[inline]
    pub fn reserved(&self) -> [u8; 2] {
        [
            codec::get_u8(self.buf, LOWLAT_REPLY_OFF_RESERVED),
            codec::get_u8(self.buf, LOWLAT_REPLY_OFF_RESERVED + 1),
        ]
    }

    #[inline]
    pub fn to_owned(&self) -> LowlatReplyFrame {
        LowlatReplyFrame {
            seq: self.seq(),
            g_pitch: self.g_pitch(),
            g_yaw: self.g_yaw(),
            confidence: self.confidence(),
            ai_healthy: self.ai_healthy(),
            version: self.version(),
            compute_us: self.compute_us(),
            reserved: self.reserved(),
        }
    }
}

#[inline]
pub fn decode_lowlat_reply(buf: &[u8]) -> Result<LowlatReplyFrame, Error> {
    LowlatReplyView::new(buf).map(|v| v.to_owned())
}

/// Quantize a physical value to a scaled int16, saturating at the type limits.
#[inline]
pub fn quantize_i16(value: f64, scale: f64) -> i16 {
    assert!(scale.is_finite() && scale > 0.0, "scale must be finite and positive");
    let v = value / scale;
    let r = if v >= 0.0 { v + 0.5 } else { v - 0.5 };
    if r >= 32767.0 { 32767 } else if r <= -32768.0 { -32768 } else { r as i16 }
}

#[inline]
pub fn dequantize_i16(raw: i16, scale: f64) -> f64 {
    f64::from(raw) * scale
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeqError {
    Duplicate,
    OutOfWindow { last: u16, got: u16, delta: u16 },
}

/// Wrap-safe sequence tracking for the request/reply stream.
#[derive(Clone, Copy, Debug)]
pub struct SeqTracker {
    tx: u16,
    rx: Option<u16>,
    window: u16,
}

impl SeqTracker {
    /// Max forward jump accepted in one observation. Must be < 2^15 so that
    /// wraparound is unambiguous.
    pub const DEFAULT_WINDOW: u16 = 64;

    pub fn new(window: u16) -> Self {
        assert!(
            window > 0 && window < 0x8000,
            "seq window must be in 1..=0x7fff"
        );
        Self {
            tx: 0,
            rx: None,
            window,
        }
    }

    pub fn observe(&mut self, seq: u16) -> Result<(), SeqError> {
        let Some(last) = self.rx else {
            self.rx = Some(seq);
            return Ok(());
        };
        let delta = seq.wrapping_sub(last);
        if delta == 0 {
            return Err(SeqError::Duplicate);
        }
        if delta > self.window {
            return Err(SeqError::OutOfWindow {
                last,
                got: seq,
                delta,
            });
        }
        self.rx = Some(seq);
        Ok(())
    }

    /// Returns the sequence number for the next outbound frame.
    pub fn next_tx(&mut self) -> u16 {
        let seq = self.tx;
        self.tx = self.tx.wrapping_add(1);
        seq
    }

    pub fn last_rx(&self) -> Option<u16> {
        self.rx
    }

    pub fn reset(&mut self) {
        self.tx = 0;
        self.rx = None;
    }
}

impl Default for SeqTracker {
    fn default() -> Self {
        Self::new(Self::DEFAULT_WINDOW)
    }
}

#[inline(always)]
pub fn flag_has(flags: u8, mask: u8) -> bool {
    flags::has(flags, mask)
}

#[inline(always)]
pub fn flag_set(flags: &mut u8, mask: u8) {
    flags::set(flags, mask);
}

#[inline(always)]
pub fn flag_clear(flags: &mut u8, mask: u8) {
    flags::clear(flags, mask);
}

#[inline]
fn clamp_i16(v: f64) -> i16 {
    if v >= 32767.0 {
        32767
    } else if v <= -32768.0 {
        -32768
    } else {
        v.round() as i16
    }
}

pub fn reply_g_pitch_rad(raw: i16) -> f64 {
    f64::from(raw) * REPLY_G_PITCH_SCALE
}

pub fn reply_g_pitch_from_rad(rad: f64) -> i16 {
    clamp_i16(rad / REPLY_G_PITCH_SCALE)
}

pub fn reply_g_yaw_rad(raw: i16) -> f64 {
    f64::from(raw) * REPLY_G_YAW_SCALE
}

pub fn reply_g_yaw_from_rad(rad: f64) -> i16 {
    clamp_i16(rad / REPLY_G_YAW_SCALE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc_known_vector() {
        assert_eq!(crc16(b"123456789"), 0x29B1);
    }

    #[test]
    fn crc_table_matches_bitwise_reference() {
        fn bitwise(data: &[u8]) -> u16 {
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
        let mut seed = 0x1234_5678u32;
        let mut data = [0u8; 256];
        for len in 0..data.len() {
            for b in &mut data[..len] {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *b = (seed >> 16) as u8;
            }
            assert_eq!(crc16(&data[..len]), bitwise(&data[..len]), "length {len}");
        }
    }

    #[test]
    fn views_are_zero_copy_and_match_owned_decode() {
        let frame = RequestFrame {
            seq: 7,
            time_us: 42,
            phase: Phase::DESCENT,
            version: REQUEST_VERSION,
            flags: flags::GPS_FIX,
            reserved: 0,
            gyro: [1.0, 2.0, 3.0],
            accel: [4.0, 5.0, 6.0],
            mag: [7.0, 8.0, 9.0],
            height: 10.0,
            pitot_dp: -11,
        };
        let mut buf = [0u8; REQUEST_SIZE];
        encode_request(&frame, &mut buf).unwrap();
        let view = RequestView::new(&buf).unwrap();
        assert_eq!(view.as_bytes().as_ptr(), buf.as_ptr());
        assert_eq!(view.seq(), frame.seq);
        assert_eq!(view.time_us(), frame.time_us);
        assert_eq!(view.phase(), frame.phase);
        assert_eq!(view.flags(), frame.flags);
        assert_eq!(view.gyro(), frame.gyro);
        assert_eq!(view.accel(), frame.accel);
        assert_eq!(view.mag(), frame.mag);
        assert_eq!(view.height(), frame.height);
        assert_eq!(view.pitot_dp(), frame.pitot_dp);
        assert_eq!(view.to_owned(), frame);
        assert_eq!(decode_request(&buf).unwrap(), frame);

        let reply = ReplyFrame {
            seq: 7,
            g_pitch: 1000,
            g_yaw: -1000,
            confidence: 99,
            ai_healthy: 1,
            version: REPLY_VERSION,
        };
        let mut buf = [0u8; REPLY_SIZE];
        encode_reply(&reply, &mut buf).unwrap();
        let view = ReplyView::new(&buf).unwrap();
        assert_eq!(view.as_bytes().as_ptr(), buf.as_ptr());
        assert_eq!(view.g_pitch(), reply.g_pitch);
        assert_eq!(view.g_yaw(), reply.g_yaw);
        assert_eq!(view.confidence(), reply.confidence);
        assert_eq!(view.to_owned(), reply);
        assert_eq!(decode_reply(&buf).unwrap(), reply);
    }
}
