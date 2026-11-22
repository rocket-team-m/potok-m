//! Acceptance tests for the calibration map on the inference side.

use potok_edge::calibration::{Calibration, CalibrationError, FALLBACK_THRESHOLD};
use potok_edge::fixture::{build_fixture, FixtureConfig};
use potok_edge::frame::{RequestFrame, FLAG_IMU_OK, PHASE_LANDING, PROTOCOL_VERSION};
use potok_edge::model::Model;
use potok_edge::runtime::Runtime;
use potok_edge::student_runtime::{
    build_student_fixture, StudentFixtureConfig, StudentModel, StudentRuntime,
};
use potok_edge::{DispatchSource, SpeculativeRuntime};
use std::fs;
use std::path::PathBuf;

const SEED: u64 = 0x504F544B;

fn student_model() -> StudentModel {
    StudentModel::from_bytes(&build_student_fixture(
        StudentFixtureConfig::default(),
        SEED,
    ))
    .unwrap()
}

fn teacher() -> Runtime {
    Runtime::new(Model::from_bytes(&build_fixture(FixtureConfig::default(), SEED)).unwrap())
}

fn frame(n: usize) -> [f32; 10] {
    let v = n as f32;
    [
        v.sin() * 0.3,
        v.cos() * 0.2,
        (v * 0.3).sin() * 0.1,
        0.05,
        9.79 + (v * 0.1).sin() * 0.2,
        -0.03,
        (v % 50.0) * 0.01,
        1500.0 + v,
        PHASE_LANDING as f32,
        FLAG_IMU_OK as f32,
    ]
}

fn req(seq: u16) -> RequestFrame {
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

fn tmp(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("potok-cal-{}-{name}", std::process::id()))
}

fn write_tmp(name: &str, body: &str) -> PathBuf {
    let path = tmp(name);
    fs::write(&path, body).unwrap();
    path
}

// 1. Identity fallback ------------------------------------------------------
#[test]
fn identity_fallback_is_neutral() {
    let c = Calibration::identity();
    assert!(c.is_identity());
    assert_eq!(c.apply(0.73), 0.73);
    assert_eq!(c.threshold(), FALLBACK_THRESHOLD);
    assert_eq!(FALLBACK_THRESHOLD, 0.7);
    // StudentRuntime::new emits a LOUD eprintln before using identity; its
    // output is bit-identical to an explicitly supplied identity map.
    let mut legacy = StudentRuntime::new(student_model());
    let mut explicit = StudentRuntime::with_calibration(student_model(), Calibration::identity());
    for n in 0..20 {
        let a = legacy.step(&frame(n));
        let b = explicit.step(&frame(n));
        for k in 0..3 {
            assert_eq!(a[k].to_bits(), b[k].to_bits());
        }
    }
    assert!(legacy.calibration().is_identity());
}

// 2. Temperature map --------------------------------------------------------
#[test]
fn temperature_map_is_applied_to_confidence_but_not_corrections() {
    let calibration =
        Calibration::from_json(r#"{"method":"temperature","temperature":2.0,"threshold":0.55}"#)
            .unwrap();
    let mut plain = StudentRuntime::with_calibration(student_model(), Calibration::identity());
    let mut calibrated = StudentRuntime::with_calibration(student_model(), calibration);
    for n in 0..20 {
        let f = frame(n);
        let a = plain.step(&f);
        let b = calibrated.step(&f);
        assert_eq!(a[0].to_bits(), b[0].to_bits());
        assert_eq!(a[1].to_bits(), b[1].to_bits());
        let raw = a[2] as f64;
        let logit = (raw / (1.0 - raw)).ln();
        let expected = (1.0 / (1.0 + (-logit / 2.0).exp())) as f32;
        assert!(
            (b[2] - expected).abs() < 1e-6,
            "frame {n}: raw {raw}, expected {expected}, got {}",
            b[2]
        );
    }
}

// 3. Isotonic map -----------------------------------------------------------
#[test]
fn isotonic_map_is_applied_to_confidence() {
    let calibration = Calibration::from_json(
        r#"{"method":"isotonic","threshold":0.6,
            "bins":[0.25,0.5,0.75],"values":[0.1,0.4,0.9]}"#,
    )
    .unwrap();
    let mut plain = StudentRuntime::with_calibration(student_model(), Calibration::identity());
    let mut calibrated = StudentRuntime::with_calibration(student_model(), calibration);
    let f = frame(7);
    let raw = plain.step(&f)[2];
    let got = calibrated.step(&f)[2];
    let expected = if raw <= 0.25 {
        0.1
    } else if raw <= 0.5 {
        0.4
    } else {
        0.9
    };
    assert_eq!(got, expected, "raw {raw} must map to {expected}");
    assert!((raw - got).abs() > 1e-3, "the map must change confidence");
}

// 4. Threshold used ---------------------------------------------------------
#[test]
fn threshold_from_file_drives_dispatch() {
    // Force calibrated confidence to 0.1 regardless of the raw head.
    const MAP: &str = r#"{"method":"isotonic","threshold":0.6,
        "bins":[1.0],"values":[0.1]}"#;
    let low = SpeculativeRuntime::new(
        teacher(),
        StudentRuntime::with_calibration(student_model(), Calibration::from_json(MAP).unwrap()),
    );
    let high = SpeculativeRuntime::new(
        teacher(),
        StudentRuntime::with_calibration(
            student_model(),
            Calibration::from_json(&MAP.replace("0.6", "0.0")).unwrap(),
        ),
    );
    assert_eq!(low.confidence_threshold, 0.6);
    assert_eq!(high.confidence_threshold, 0.0);

    let mut low = low;
    let mut high = high;
    low.on_request(&req(1));
    high.on_request(&req(1));
    assert_eq!(low.last_source, DispatchSource::TeacherLowConfidence);
    assert_eq!(high.last_source, DispatchSource::Student);
}

// 5. Malformed file fails loudly --------------------------------------------
#[test]
fn malformed_file_fails_loudly() {
    let bad_json = write_tmp("bad.json", "{ not json");
    let err = Calibration::from_file(&bad_json).unwrap_err();
    assert!(matches!(err, CalibrationError::Parse(_)), "{err}");
    assert!(err.to_string().contains("not valid JSON"));

    let no_map = write_tmp(
        "no-map.json",
        r#"{"method":"isotonic","threshold":0.6,"n":20}"#,
    );
    let err = Calibration::from_file(&no_map).unwrap_err();
    assert!(matches!(err, CalibrationError::Schema(_)), "{err}");
    assert!(err.to_string().contains("isotonic"));

    let unknown = write_tmp("unknown.json", r#"{"method":"mystery","threshold":0.6}"#);
    assert!(Calibration::from_file(&unknown).is_err());

    let missing = tmp("does-not-exist.json");
    assert!(matches!(
        Calibration::from_file(&missing),
        Err(CalibrationError::Io(_))
    ));

    for p in [bad_json, no_map, unknown] {
        let _ = fs::remove_file(p);
    }
}

// 6. Backward-compatible with no file --------------------------------------
#[test]
fn no_file_keeps_legacy_behaviour() {
    let spec = SpeculativeRuntime::new(teacher(), StudentRuntime::new(student_model()));
    assert_eq!(spec.confidence_threshold, FALLBACK_THRESHOLD);
    assert!(spec.student().calibration().is_identity());

    // Loading a real calibration.json through the file-based constructor wires
    // both the student map and the dispatch threshold.
    let cal = write_tmp(
        "good.json",
        r#"{"method":"temperature","temperature":2.0,"threshold":0.42}"#,
    );
    let student_path = tmp("student.bin");
    let teacher_path = tmp("teacher.bin");
    fs::write(
        &student_path,
        build_student_fixture(StudentFixtureConfig::default(), SEED),
    )
    .unwrap();
    fs::write(&teacher_path, build_fixture(FixtureConfig::default(), SEED)).unwrap();
    let loaded =
        SpeculativeRuntime::load_with_calibration(&teacher_path, &student_path, &cal).unwrap();
    assert_eq!(loaded.confidence_threshold, 0.42);
    assert!(!loaded.student().calibration().is_identity());
    for p in [cal, student_path, teacher_path] {
        let _ = fs::remove_file(&p);
    }
}
