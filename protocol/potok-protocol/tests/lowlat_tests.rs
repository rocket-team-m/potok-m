use potok_protocol::*;

const REQ_GOLDEN: &str = "544c34127bf15365020112000100feff03000400fbff06000700f8ff09000a000b00f4ff0d000e000a2b";
const REP_GOLDEN: &str = "524c34127d0006ffc801012a0000fd1b";

fn sample_request() -> LowlatRequestFrame {
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

fn sample_reply() -> LowlatReplyFrame {
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

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

fn set_crc(buf: &mut [u8], off: usize) {
    let crc = crc16(&buf[..off]);
    buf[off] = crc as u8;
    buf[off + 1] = (crc >> 8) as u8;
}

#[test]
fn lowlat_request_round_trip() {
    let frame = sample_request();
    let mut buf = [0u8; LOWLAT_REQUEST_SIZE];
    encode_lowlat_request(&frame, &mut buf).unwrap();
    assert_eq!(decode_lowlat_request(&buf).unwrap(), frame);

    let view = LowlatRequestView::new(&buf).unwrap();
    assert_eq!(view.as_bytes().as_ptr(), buf.as_ptr());
    assert_eq!(view.seq(), frame.seq);
    assert_eq!(view.time_us(), frame.time_us);
    assert_eq!(view.phase(), frame.phase);
    assert_eq!(view.flags(), frame.flags);
    assert_eq!(view.features(), frame.features);
    assert_eq!(view.to_owned(), frame);
}

#[test]
fn lowlat_reply_round_trip() {
    let frame = sample_reply();
    let mut buf = [0u8; LOWLAT_REPLY_SIZE];
    encode_lowlat_reply(&frame, &mut buf).unwrap();
    assert_eq!(decode_lowlat_reply(&buf).unwrap(), frame);

    let view = LowlatReplyView::new(&buf).unwrap();
    assert_eq!(view.as_bytes().as_ptr(), buf.as_ptr());
    assert_eq!(view.compute_us(), 42);
    assert_eq!(view.reserved(), [0, 0]);
    assert_eq!(view.to_owned(), frame);
}

#[test]
fn lowlat_oversized_output_buffer_is_allowed() {
    let mut req = [0xAAu8; LOWLAT_REQUEST_SIZE + 4];
    encode_lowlat_request(&sample_request(), &mut req).unwrap();
    assert_eq!(decode_lowlat_request(&req).unwrap(), sample_request());
    let mut rep = [0xAAu8; LOWLAT_REPLY_SIZE + 4];
    encode_lowlat_reply(&sample_reply(), &mut rep).unwrap();
    assert_eq!(decode_lowlat_reply(&rep).unwrap(), sample_reply());
}

#[test]
fn lowlat_truncation_is_rejected() {
    let mut req = [0u8; LOWLAT_REQUEST_SIZE];
    encode_lowlat_request(&sample_request(), &mut req).unwrap();
    for n in 0..LOWLAT_REQUEST_SIZE {
        assert_eq!(decode_lowlat_request(&req[..n]), Err(Error::TooShort { need: LOWLAT_REQUEST_SIZE, got: n }));
        let mut out = [0u8; LOWLAT_REQUEST_SIZE];
        assert_eq!(encode_lowlat_request(&sample_request(), &mut out[..n]), Err(Error::BufferTooSmall { need: LOWLAT_REQUEST_SIZE, got: n }));
    }

    let mut rep = [0u8; LOWLAT_REPLY_SIZE];
    encode_lowlat_reply(&sample_reply(), &mut rep).unwrap();
    for n in 0..LOWLAT_REPLY_SIZE {
        assert_eq!(decode_lowlat_reply(&rep[..n]), Err(Error::TooShort { need: LOWLAT_REPLY_SIZE, got: n }));
    }
}

#[test]
fn lowlat_wrong_magic_crc_phase_and_version() {
    let mut buf = [0u8; LOWLAT_REQUEST_SIZE];
    encode_lowlat_request(&sample_request(), &mut buf).unwrap();
    buf[0] ^= 0xFF;
    assert!(matches!(decode_lowlat_request(&buf), Err(Error::BadMagic { .. })));

    let mut buf = [0u8; LOWLAT_REQUEST_SIZE];
    encode_lowlat_request(&sample_request(), &mut buf).unwrap();
    buf[LOWLAT_REQUEST_OFF_SEQ] ^= 1;
    assert!(matches!(decode_lowlat_request(&buf), Err(Error::BadCrc { .. })));

    let mut buf = [0u8; LOWLAT_REQUEST_SIZE];
    encode_lowlat_request(&sample_request(), &mut buf).unwrap();
    buf[LOWLAT_REQUEST_OFF_PHASE] = 9;
    set_crc(&mut buf, LOWLAT_REQUEST_CRC_OFF);
    assert_eq!(decode_lowlat_request(&buf), Err(Error::BadPhase(9)));

    let mut buf = [0u8; LOWLAT_REQUEST_SIZE];
    let mut bad = sample_request();
    bad.version = 99;
    encode_lowlat_request(&bad, &mut buf).unwrap();
    assert_eq!(
        decode_lowlat_request(&buf),
        Err(Error::VersionMismatch { expected: u16::from(LOWLAT_REQUEST_VERSION), got: 99 })
    );

    let mut buf = [0u8; LOWLAT_REPLY_SIZE];
    encode_lowlat_reply(&sample_reply(), &mut buf).unwrap();
    buf[0] ^= 0xFF;
    assert!(matches!(decode_lowlat_reply(&buf), Err(Error::BadMagic { .. })));
    let mut buf = [0u8; LOWLAT_REPLY_SIZE];
    encode_lowlat_reply(&sample_reply(), &mut buf).unwrap();
    buf[LOWLAT_REPLY_OFF_CONFIDENCE] ^= 1;
    assert!(matches!(decode_lowlat_reply(&buf), Err(Error::BadCrc { .. })));
    let mut bad = sample_reply();
    bad.version = 99;
    encode_lowlat_reply(&bad, &mut buf).unwrap();
    assert_eq!(
        decode_lowlat_reply(&buf),
        Err(Error::VersionMismatch { expected: u16::from(LOWLAT_REPLY_VERSION), got: 99 })
    );
}

#[test]
fn lowlat_flag_helpers() {
    let mut frame = sample_request();
    frame.flags = 0;
    flag_set(&mut frame.flags, flags::GPS_FIX);
    flag_set(&mut frame.flags, flags::MOTOR_ARMED);
    assert!(flag_has(frame.flags, flags::GPS_FIX));
    assert!(flag_has(frame.flags, flags::MOTOR_ARMED));
    assert!(!flag_has(frame.flags, flags::FAULT));
    flag_clear(&mut frame.flags, flags::GPS_FIX);
    assert_eq!(frame.flags, flags::MOTOR_ARMED);
    let mut buf = [0u8; LOWLAT_REQUEST_SIZE];
    encode_lowlat_request(&frame, &mut buf).unwrap();
    assert_eq!(decode_lowlat_request(&buf).unwrap().flags, flags::MOTOR_ARMED);
}

#[test]
fn lowlat_feature_scales_and_int16_saturation() {
    assert_eq!(LOWLAT_REQUEST_FEATURES_SCALES.len(), 14);
    assert_eq!(LOWLAT_REQUEST_FEATURES_NAMES.len(), 14);
    assert_eq!(LOWLAT_REQUEST_FEATURES_NAMES[0], "gyro_x");
    assert_eq!(LOWLAT_REQUEST_FEATURES_NAMES[8], "mag_z");
    assert_eq!(LOWLAT_REQUEST_FEATURES_NAMES[13], "battery_mv");
    assert_eq!(LOWLAT_REQUEST_FEATURES_SCALES[0], 1e-3);
    assert_eq!(LOWLAT_REQUEST_FEATURES_SCALES[6], 1e-4);
    assert_eq!(LOWLAT_REQUEST_FEATURES_SCALES[9], 1e-2);
    assert_eq!(LOWLAT_REQUEST_FEATURES_SCALES[10], 1.0);

    assert_eq!(quantize_i16(1.0, 1e-3), 1000);
    assert_eq!(quantize_i16(-1.0, 1e-3), -1000);
    assert_eq!(quantize_i16(1e9, 1e-3), i16::MAX);
    assert_eq!(quantize_i16(-1e9, 1e-3), i16::MIN);
    assert_eq!(quantize_i16(1.0, 1e-4), 10000);
    assert_eq!(quantize_i16(1.0, 1e-2), 100);
    assert!((dequantize_i16(1000, 1e-3) - 1.0).abs() < 1e-12);

    let mut frame = sample_request();
    frame.features[0] = i16::MAX;
    frame.features[1] = i16::MIN;
    let mut buf = [0u8; LOWLAT_REQUEST_SIZE];
    encode_lowlat_request(&frame, &mut buf).unwrap();
    assert_eq!(decode_lowlat_request(&buf).unwrap(), frame);
}

#[test]
fn lowlat_golden_bytes_match_c_and_python() {
    let request = unhex(REQ_GOLDEN);
    let reply = unhex(REP_GOLDEN);
    assert_eq!(request.len(), LOWLAT_REQUEST_SIZE);
    assert_eq!(reply.len(), LOWLAT_REPLY_SIZE);

    let decoded = decode_lowlat_request(&request).unwrap();
    assert_eq!(decoded, sample_request());
    assert_eq!(LowlatRequestView::new(&request).unwrap().to_owned(), decoded);
    let mut out = [0u8; LOWLAT_REQUEST_SIZE];
    encode_lowlat_request(&decoded, &mut out).unwrap();
    assert_eq!(out.to_vec(), request);

    let decoded = decode_lowlat_reply(&reply).unwrap();
    assert_eq!(decoded, sample_reply());
    let mut out = [0u8; LOWLAT_REPLY_SIZE];
    encode_lowlat_reply(&decoded, &mut out).unwrap();
    assert_eq!(out.to_vec(), reply);
}

#[test]
fn profiles_have_distinct_magics_and_sizes() {
    assert_ne!(REQUEST_MAGIC, LOWLAT_REQUEST_MAGIC);
    assert_ne!(REPLY_MAGIC, LOWLAT_REPLY_MAGIC);
    assert_eq!((REQUEST_SIZE, REPLY_SIZE), (56, 14));
    assert_eq!((LOWLAT_REQUEST_SIZE, LOWLAT_REPLY_SIZE), (42, 16));
}
