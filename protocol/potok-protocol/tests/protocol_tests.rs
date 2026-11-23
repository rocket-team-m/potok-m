use potok_protocol::*;

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

fn set_crc(buf: &mut [u8], off: usize) {
    let crc = crc16(&buf[..off]);
    buf[off] = crc as u8;
    buf[off + 1] = (crc >> 8) as u8;
}

#[test]
fn request_round_trip() {
    let original = sample_request();
    let mut buf = [0u8; REQUEST_SIZE];
    encode_request(&original, &mut buf).unwrap();
    let decoded = decode_request(&buf).unwrap();
    assert_eq!(decoded, original);
    assert_eq!(u16::from_le_bytes([buf[0], buf[1]]), REQUEST_MAGIC);
}

#[test]
fn reply_round_trip() {
    let original = sample_reply();
    let mut buf = [0u8; REPLY_SIZE];
    encode_reply(&original, &mut buf).unwrap();
    assert_eq!(decode_reply(&buf).unwrap(), original);
    assert_eq!(u16::from_le_bytes([buf[0], buf[1]]), REPLY_MAGIC);
}

#[test]
fn oversized_output_buffer_is_allowed() {
    let mut buf = [0xAAu8; REQUEST_SIZE + 8];
    encode_request(&sample_request(), &mut buf).unwrap();
    assert_eq!(decode_request(&buf).unwrap(), sample_request());
    assert_eq!(&buf[REQUEST_SIZE..], &[0xAA; 8]);
}

#[test]
fn wrong_magic_is_rejected() {
    let mut buf = [0u8; REQUEST_SIZE];
    encode_request(&sample_request(), &mut buf).unwrap();
    buf[0] ^= 0xFF;
    assert_eq!(
        decode_request(&buf),
        Err(Error::BadMagic {
            expected: REQUEST_MAGIC,
            got: (REQUEST_MAGIC & 0xFF00) | 0x00B2
        })
    );

    let mut buf = [0u8; REPLY_SIZE];
    encode_reply(&sample_reply(), &mut buf).unwrap();
    buf[1] ^= 0xFF;
    assert!(matches!(decode_reply(&buf), Err(Error::BadMagic { .. })));
}

#[test]
fn wrong_crc_is_rejected() {
    let mut buf = [0u8; REQUEST_SIZE];
    encode_request(&sample_request(), &mut buf).unwrap();
    buf[REQUEST_OFF_TIME_US] ^= 0x01;
    assert!(matches!(decode_request(&buf), Err(Error::BadCrc { .. })));

    let mut buf = [0u8; REPLY_SIZE];
    encode_reply(&sample_reply(), &mut buf).unwrap();
    buf[REPLY_OFF_CONFIDENCE] ^= 0x80;
    assert!(matches!(decode_reply(&buf), Err(Error::BadCrc { .. })));
}

#[test]
fn truncated_frames_are_rejected() {
    let mut req = [0u8; REQUEST_SIZE];
    encode_request(&sample_request(), &mut req).unwrap();
    for n in 0..REQUEST_SIZE {
        assert_eq!(
            decode_request(&req[..n]),
            Err(Error::TooShort {
                need: REQUEST_SIZE,
                got: n
            })
        );
        let mut out = [0u8; REQUEST_SIZE];
        assert_eq!(
            encode_request(&sample_request(), &mut out[..n]),
            Err(Error::BufferTooSmall {
                need: REQUEST_SIZE,
                got: n
            })
        );
    }

    let mut reply = [0u8; REPLY_SIZE];
    encode_reply(&sample_reply(), &mut reply).unwrap();
    for n in 0..REPLY_SIZE {
        assert_eq!(
            decode_reply(&reply[..n]),
            Err(Error::TooShort {
                need: REPLY_SIZE,
                got: n
            })
        );
    }
}

#[test]
fn seq_wraparound() {
    let mut t = SeqTracker::default();
    assert_eq!(t.observe(u16::MAX - 1), Ok(()));
    assert_eq!(t.observe(u16::MAX), Ok(()));
    assert_eq!(t.observe(0), Ok(()));
    assert_eq!(t.observe(1), Ok(()));
    assert_eq!(t.observe(1), Err(SeqError::Duplicate));
    assert!(matches!(
        t.observe(100),
        Err(SeqError::OutOfWindow {
            last: 1,
            got: 100,
            delta: 99
        })
    ));

    let mut t = SeqTracker::default();
    for _ in 0..=u16::MAX {
        t.next_tx();
    }
    assert_eq!(t.next_tx(), 0);
    assert_eq!(t.next_tx(), 1);

    let mut t = SeqTracker::default();
    assert_eq!(t.last_rx(), None);
    assert_eq!(t.observe(7), Ok(()));
    assert_eq!(t.last_rx(), Some(7));
    t.reset();
    assert_eq!(t.last_rx(), None);
    assert_eq!(t.next_tx(), 0);
}

#[test]
#[should_panic(expected = "seq window must be in 1..=0x7fff")]
fn seq_window_is_bounded() {
    let _ = SeqTracker::new(0x8000);
}

#[test]
fn flag_helpers() {
    let mut f = 0u8;
    flag_set(&mut f, flags::BARO_OK);
    flag_set(&mut f, flags::FAULT);
    assert!(flag_has(f, flags::BARO_OK));
    assert!(flag_has(f, flags::FAULT));
    assert!(!flag_has(f, flags::IMU_OK));
    flag_clear(&mut f, flags::BARO_OK);
    assert!(!flag_has(f, flags::BARO_OK));
    assert_eq!(f, flags::FAULT);
    assert_eq!(flags::FAULT, 1 << 7);
}

#[test]
fn version_mismatch_is_rejected() {
    let mut buf = [0u8; REQUEST_SIZE];
    let mut req = sample_request();
    req.version = 99;
    encode_request(&req, &mut buf).unwrap();
    assert_eq!(
        decode_request(&buf),
        Err(Error::VersionMismatch {
            expected: u16::from(REQUEST_VERSION),
            got: 99
        })
    );

    let mut buf = [0u8; REPLY_SIZE];
    let mut reply = sample_reply();
    reply.version = 0xBEEF;
    encode_reply(&reply, &mut buf).unwrap();
    assert_eq!(
        decode_reply(&buf),
        Err(Error::VersionMismatch {
            expected: REPLY_VERSION,
            got: 0xBEEF
        })
    );
}

#[test]
fn bad_phase_is_rejected() {
    let mut buf = [0u8; REQUEST_SIZE];
    encode_request(&sample_request(), &mut buf).unwrap();
    buf[REQUEST_OFF_PHASE] = 9;
    set_crc(&mut buf, REQUEST_CRC_OFF);
    assert_eq!(decode_request(&buf), Err(Error::BadPhase(9)));
}

#[test]
fn all_phases_map_both_ways() {
    let phases = [
        Phase::BURN,
        Phase::COAST,
        Phase::APOGEE,
        Phase::DESCENT,
        Phase::LANDED,
    ];
    for (i, p) in phases.iter().enumerate() {
        assert_eq!(u8::from(*p), i as u8);
        assert_eq!(Phase::try_from(i as u8).unwrap(), *p);
    }
    assert!(Phase::try_from(5).is_err());
}

#[test]
fn golden_wire_bytes_match_c_and_python() {
    let mut req = [0u8; REQUEST_SIZE];
    encode_request(&sample_request(), &mut req).unwrap();
    let hex: String = req.iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(
        hex,
        "4d5034127bf1536502011200cdcccc3dcdcc4cbe33335340c3f51c41000000000000c0bf00004841000050c0000030420080f742bffec67f"
    );

    let mut rep = [0u8; REPLY_SIZE];
    encode_reply(&sample_reply(), &mut rep).unwrap();
    let hex: String = rep.iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(hex, "505234127d0006ffc8010100c549");
}

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

#[test]
fn committed_golden_fixtures_reencode_byte_identical() {
    let req_hex = "4d5034127bf1536502011200cdcccc3dcdcc4cbe33335340c3f51c41000000000000c0bf00004841000050c0000030420080f742bffec67f";
    let rep_hex = "505234127d0006ffc8010100c549";

    let req_bytes = unhex(req_hex);
    let rep_bytes = unhex(rep_hex);
    assert_eq!(req_bytes.len(), REQUEST_SIZE);
    assert_eq!(rep_bytes.len(), REPLY_SIZE);

    let request = decode_request(&req_bytes).unwrap();
    let view = RequestView::new(&req_bytes).unwrap();
    assert_eq!(view.to_owned(), request);
    let mut reencoded = vec![0u8; REQUEST_SIZE];
    encode_request(&request, &mut reencoded).unwrap();
    assert_eq!(reencoded, req_bytes);
    assert_eq!(crc16(&req_bytes[..REQUEST_CRC_OFF]), {
        let mut b = [0u8; 2];
        b.copy_from_slice(&req_bytes[REQUEST_CRC_OFF..REQUEST_CRC_OFF + 2]);
        u16::from_le_bytes(b)
    });

    let reply = decode_reply(&rep_bytes).unwrap();
    let view = ReplyView::new(&rep_bytes).unwrap();
    assert_eq!(view.to_owned(), reply);
    let mut reencoded = vec![0u8; REPLY_SIZE];
    encode_reply(&reply, &mut reencoded).unwrap();
    assert_eq!(reencoded, rep_bytes);
}

#[test]
fn scaled_angles_are_bijective_at_resolution() {
    for raw in [-32768i16, -250, -1, 0, 1, 125, 32767] {
        assert_eq!(reply_g_pitch_from_rad(reply_g_pitch_rad(raw)), raw);
        assert_eq!(reply_g_yaw_from_rad(reply_g_yaw_rad(raw)), raw);
    }
    assert!((reply_g_pitch_rad(1000) - 1.0).abs() < 1e-12);
    assert_eq!(reply_g_pitch_from_rad(1e9), 32767);
    assert_eq!(reply_g_yaw_from_rad(-1e9), -32768);
}
