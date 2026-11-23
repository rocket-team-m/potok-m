//! Standalone `no_std` consumer probe for `potok-protocol`.
//!
//! `tests/no_std_gate.rs` compiles this file as the root of a separate
//! `#![no_std]` crate that depends on `potok-protocol` with
//! `default-features = false`. It is a crate root on purpose: do not add it
//! with `mod no_std;` to `lib.rs`. If the public API grows a `std`-only
//! dependency, this probe stops compiling on the embedded target.
#![no_std]

extern crate potok_protocol as protocol;

use protocol::{
    crc16, decode_lowlat_reply, decode_lowlat_request, decode_reply, decode_request,
    encode_lowlat_reply, encode_lowlat_request, encode_reply, encode_request, flag_clear,
    flag_has, flag_set, reply_g_pitch_from_rad, reply_g_pitch_rad, reply_g_yaw_from_rad,
    Error, LowlatReplyFrame, LowlatRequestFrame, Phase, ReplyFrame, ReplyView, RequestFrame,
    RequestView, SeqTracker, LOWLAT_REPLY_SIZE, LOWLAT_REQUEST_SIZE, REPLY_SIZE, REQUEST_SIZE,
};

// Wire layout is part of the no_std contract: firmware sizes its DMA buffers
// from these constants, so a generated-layout drift is a compile-time failure.
const _: () = assert!(protocol::REQUEST_CRC_OFF + protocol::REQUEST_CRC_LEN == REQUEST_SIZE);
const _: () = assert!(protocol::REPLY_CRC_OFF + protocol::REPLY_CRC_LEN == REPLY_SIZE);
const _: () =
    assert!(protocol::LOWLAT_REQUEST_CRC_OFF + protocol::LOWLAT_REQUEST_CRC_LEN == LOWLAT_REQUEST_SIZE);
const _: () =
    assert!(protocol::LOWLAT_REPLY_CRC_OFF + protocol::LOWLAT_REPLY_CRC_LEN == LOWLAT_REPLY_SIZE);
const _: () = assert!(REQUEST_SIZE > LOWLAT_REQUEST_SIZE);

// `Display` must come from `core`, not from `std`.
const _: fn(&Error, &mut core::fmt::Formatter<'_>) -> core::fmt::Result =
    <Error as core::fmt::Display>::fmt;

/// Exercise the whole no_std surface with stack buffers only: no allocator,
/// no `std`, no I/O. Codecs, views, scaling, flags and sequencing all run.
pub fn smoke() -> Result<(), Error> {
    let mut req_buf = [0u8; REQUEST_SIZE];
    let request = RequestFrame {
        seq: 1,
        time_us: 2,
        phase: Phase::DESCENT,
        version: protocol::REQUEST_VERSION,
        flags: protocol::flags::BARO_OK,
        reserved: 0,
        gyro: [1.0, 2.0, 3.0],
        accel: [4.0, 5.0, 6.0],
        mag: [7.0, 8.0, 9.0],
        height: 10.0,
        pitot_dp: -11,
    };
    encode_request(&request, &mut req_buf)?;
    let view = RequestView::new(&req_buf)?;
    if view.to_owned() != request || decode_request(&req_buf)? != request {
        return Err(Error::BadPhase(0));
    }

    let mut reply_buf = [0u8; REPLY_SIZE];
    let reply = ReplyFrame {
        seq: request.seq,
        g_pitch: reply_g_pitch_from_rad(0.25),
        g_yaw: 0,
        confidence: 99,
        ai_healthy: 1,
        version: protocol::REPLY_VERSION,
    };
    encode_reply(&reply, &mut reply_buf)?;
    if ReplyView::new(&reply_buf)?.g_pitch() != reply.g_pitch
        || decode_reply(&reply_buf)? != reply
        || reply_g_pitch_rad(reply.g_pitch) < 0.0
        || reply_g_yaw_from_rad(0.0) != 0
    {
        return Err(Error::BadPhase(1));
    }

    let mut low_req_buf = [0u8; LOWLAT_REQUEST_SIZE];
    let low_req = LowlatRequestFrame {
        seq: 2,
        time_us: 3,
        phase: Phase::COAST,
        version: protocol::LOWLAT_REQUEST_VERSION,
        flags: 0,
        reserved: 0,
        features: [100; 14],
    };
    encode_lowlat_request(&low_req, &mut low_req_buf)?;
    if decode_lowlat_request(&low_req_buf)? != low_req {
        return Err(Error::BadPhase(2));
    }

    let mut low_reply_buf = [0u8; LOWLAT_REPLY_SIZE];
    let low_reply = LowlatReplyFrame {
        seq: 2,
        g_pitch: 1,
        g_yaw: -1,
        confidence: 50,
        ai_healthy: 1,
        version: protocol::LOWLAT_REPLY_VERSION,
        compute_us: 7,
        reserved: [0, 0],
    };
    encode_lowlat_reply(&low_reply, &mut low_reply_buf)?;
    if decode_lowlat_reply(&low_reply_buf)? != low_reply {
        return Err(Error::BadPhase(3));
    }

    if crc16(b"123456789") != 0x29B1 {
        return Err(Error::BadCrc {
            expected: 0x29B1,
            got: 0,
        });
    }

    let mut flags = 0u8;
    flag_set(&mut flags, protocol::flags::IMU_OK);
    if !flag_has(flags, protocol::flags::IMU_OK) {
        return Err(Error::BadPhase(4));
    }
    flag_clear(&mut flags, protocol::flags::IMU_OK);
    if flag_has(flags, protocol::flags::IMU_OK) {
        return Err(Error::BadPhase(5));
    }

    let mut seq = SeqTracker::default();
    seq.observe(7).map_err(|_| Error::BadPhase(6))?;
    if seq.next_tx() != 0 || seq.last_rx() != Some(7) {
        return Err(Error::BadPhase(7));
    }
    Ok(())
}
