mod common;

use common::{fixture, request};
use potok_edge::frame::{
    build_request, crc16, FrameError, PotokUartV1, ReplyFrame, RequestFrame, FLAG_MOTOR_ARMED,
    PHASE_ARMED, PHASE_FAULT, PHASE_FLIGHT, PHASE_LANDING, REPLY_LEN, REQUEST_LEN,
};
use potok_edge::model::Model;
use potok_edge::runtime::{PhasePolicy, Runtime, MAX_CONSECUTIVE_MISSES};
use potok_edge::uart::handle_request;
use potok_edge::FrameLayout;

fn runtime() -> Runtime {
    Runtime::new(Model::from_bytes(&fixture()).unwrap())
}

#[test]
fn deterministic_output_for_fixed_fixture() {
    let mut a = runtime();
    let mut b = runtime();
    let mut replies = Vec::new();
    for seq in 0..12u16 {
        let req = request(seq);
        let ra = a.on_request(&req);
        let rb = b.on_request(&req);
        assert_eq!(
            ra, rb,
            "same fixture + same frames must give bit-identical replies"
        );
        assert_eq!(ra.seq, seq);
        assert_eq!(ra.ai_healthy, 1);
        replies.push(ra);
    }
    assert_eq!((replies[2].g_pitch, replies[2].g_yaw), (100, 100));
    // NEON changes only the summation tree, so allow a 1-LSB quantization
    // difference on aarch64; scalar targets must match the golden value.
    #[cfg(not(target_arch = "aarch64"))]
    assert_eq!(replies[2].confidence, 209);
    #[cfg(target_arch = "aarch64")]
    assert!(replies[2].confidence.abs_diff(209) <= 1);
    assert!(
        replies.iter().any(|r| r.g_pitch != 0 || r.g_yaw != 0),
        "the fixture has non-zero weights, so the assist must not be all zeros"
    );
    let ra = a.on_request(&request(12));
    let rb = b.on_request(&request(12));
    assert_eq!(ra, rb);
}

#[test]
fn forbidden_phases_produce_zero_correction() {
    let mut rt = runtime();
    let allowed = rt.on_request(&request(1));
    assert!(allowed.g_pitch != 0 || allowed.g_yaw != 0);

    let mut armed = request(2);
    armed.flags |= FLAG_MOTOR_ARMED;
    let muted = rt.on_request(&armed);
    assert_eq!(muted.g_pitch, 0);
    assert_eq!(muted.g_yaw, 0);
    assert_eq!(muted.confidence, 0);
    assert_eq!(muted.ai_healthy, 1, "muting is policy, not a fault");

    let mut faulty = request(3);
    faulty.phase = PHASE_FAULT;
    let muted = rt.on_request(&faulty);
    assert_eq!((muted.g_pitch, muted.g_yaw), (0, 0));

    // Policy is configurable: forbidden set can be changed explicitly.
    let mut policy = PhasePolicy::conservative();
    policy.forbidden_phases[PHASE_FLIGHT as usize] = true;
    let mut rt2 = Runtime::with_policy(Model::from_bytes(&fixture()).unwrap(), policy);
    let mut flight = request(1);
    flight.phase = PHASE_FLIGHT;
    assert_eq!(rt2.on_request(&flight).g_pitch, 0);
}

#[test]
fn crc_and_version_errors_are_refused() {
    let layout = PotokUartV1;
    let mut bytes = [0u8; REQUEST_LEN];
    build_request(&request(4), &mut bytes).unwrap();
    assert!(layout.decode_request(&bytes).is_ok());

    bytes[30] ^= 0x80;
    assert!(matches!(
        layout.decode_request(&bytes),
        Err(FrameError::CrcMismatch { .. })
    ));

    let mut bytes = [0u8; REQUEST_LEN];
    let mut bad = request(4);
    bad.version = 2;
    build_request(&bad, &mut bytes).unwrap();
    assert!(matches!(
        layout.decode_request(&bytes),
        Err(FrameError::VersionMismatch(2))
    ));
    assert_eq!(
        crc16(&bytes[..REQUEST_LEN - 2]),
        u16::from_le_bytes([bytes[54], bytes[55]])
    );
}

#[test]
fn sequence_errors_and_three_misses_disable_ai() {
    let mut rt = runtime();
    // First frame establishes the sequence baseline.
    let first = rt.on_request(&request(10));
    assert_eq!(first.seq, 10);
    assert_eq!(first.ai_healthy, 1);

    // Duplicate: counted as a miss, reply muted but still healthy.
    let dup = rt.on_request(&request(10));
    assert_eq!(dup.ai_healthy, 1);
    assert_eq!(rt.misses(), 1);

    // Gap: counted, resync on the new sequence.
    let gap = rt.on_request(&request(20));
    assert_eq!(gap.ai_healthy, 1);
    assert_eq!(rt.misses(), 2);

    // A fully valid frame clears the miss counter.
    let ok = rt.on_request(&request(21));
    assert_eq!(ok.ai_healthy, 1);
    assert_eq!(rt.misses(), 0);

    // Three timeouts latch the assistant off permanently.
    for _ in 0..MAX_CONSECUTIVE_MISSES {
        rt.on_timeout();
    }
    assert!(rt.is_disabled());
    assert_eq!(rt.counters.disables, 1);
    let after = rt.on_request(&request(22));
    assert_eq!(after, ReplyFrame::muted(22, false));
    assert_eq!((after.g_pitch, after.g_yaw, after.confidence), (0, 0, 0));

    // Even a valid frame cannot re-enable it.
    assert_eq!(rt.on_request(&request(23)).ai_healthy, 0);
}

#[test]
fn version_mismatch_disables_immediately() {
    let mut rt = runtime();
    let mut bytes = [0u8; REQUEST_LEN];
    let mut bad = request(3);
    bad.version = 9;
    build_request(&bad, &mut bytes).unwrap();
    let mut reply = [0u8; REPLY_LEN];
    assert!(handle_request(&PotokUartV1, &mut rt, &bytes, &mut reply).is_err());
    assert!(
        rt.is_disabled(),
        "protocol version mismatch is fatal on the first frame"
    );
}

#[test]
fn reply_layout_uses_seq_and_ai_healthy() {
    let layout = PotokUartV1;
    let mut buf = [0u8; REPLY_LEN];
    layout
        .encode_reply(&ReplyFrame::muted(77, false), &mut buf)
        .unwrap();
    assert_eq!(u16::from_le_bytes([buf[2], buf[3]]), 77);
    assert_eq!(buf[9], 0);
    assert_eq!(crc16(&buf[..12]), u16::from_le_bytes([buf[12], buf[13]]));
}

#[test]
fn unknown_phase_is_forbidden() {
    let mut rt = runtime();
    let mut req: RequestFrame = request(1);
    req.phase = 200;
    assert_eq!(rt.on_request(&req).g_pitch, 0);
    let _ = PHASE_ARMED;
    let _ = PHASE_LANDING;
}
