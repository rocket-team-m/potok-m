// SPDX-License-Identifier: GPL-3.0-or-later
//! SIMID — gates for the typed identification input contracts.
//!
//! No measurement data exists in the repo: every fixture below is synthetic.
//! The contracts are compiled from their real source path so these tests
//! exercise the shipped code, not a copy. The `error` shim mirrors the
//! library's module path for that include; `lib.rs` only needs the one-line
//! `pub mod identification;` registration to expose the module publicly.
//!
//! Coverage: LOUD rejection at construction and at query time, bitwise
//! deterministic interpolation (no RNG, no clock, no clamp), and exact
//! knot/endpoint read-back for all three contracts.

mod error {
    pub use potok_sim::error::{Result, SimError};
}

#[path = "../src/identification.rs"]
mod identification;

use error::{Result, SimError};
use identification::{
    AllanPoint, ImuAllanPoints, ImuChannel, ServoStep, ServoStepResponse, ThrustCurve, ThrustSample,
};

fn assert_loud<T: std::fmt::Debug>(result: Result<T>, needle: &str) {
    match result {
        Err(SimError::Config(msg)) => {
            assert!(msg.contains(needle), "error {msg:?} lacks {needle:?}");
        }
        other => panic!("expected a loud Config error containing {needle:?}, got {other:?}"),
    }
}

fn thrust_fixture() -> ThrustCurve {
    ThrustCurve::new(vec![
        ThrustSample {
            t_s: 0.0,
            thrust_n: 0.0,
        },
        ThrustSample {
            t_s: 1.0,
            thrust_n: 100.0,
        },
        ThrustSample {
            t_s: 3.0,
            thrust_n: 300.0,
        },
        ThrustSample {
            t_s: 5.0,
            thrust_n: 100.0,
        },
    ])
    .expect("synthetic static-fire curve")
}

fn allan_fixture() -> ImuAllanPoints {
    let arw = 0.12_f64;
    let points = [0.01_f64, 0.1, 1.0, 10.0]
        .iter()
        .map(|&tau| AllanPoint {
            tau_s: tau,
            sigma: arw / tau.sqrt(),
        })
        .collect();
    ImuAllanPoints::new(ImuChannel::Gyro, points).expect("synthetic Allan series")
}

fn servo_fixture() -> ServoStepResponse {
    ServoStepResponse::new(
        0.1,
        4.0,
        2.0,
        vec![
            ServoStep {
                t_s: 0.0,
                commanded: 0.0,
                measured: 0.0,
            },
            ServoStep {
                t_s: 0.1,
                commanded: 20.0,
                measured: 0.0,
            },
            ServoStep {
                t_s: 0.6,
                commanded: 20.0,
                measured: 2.0,
            },
            ServoStep {
                t_s: 2.1,
                commanded: 20.0,
                measured: 8.0,
            },
        ],
    )
    .expect("synthetic servo step response")
}

// --- Thrust curve ------------------------------------------------------------

#[test]
fn thrust_curve_interpolates_knots_and_midpoints_exactly() {
    let curve = thrust_fixture();
    assert_eq!(curve.samples().len(), 4);
    let cases: [(f64, f64); 8] = [
        (0.0, 0.0),
        (1.0, 100.0),
        (3.0, 300.0),
        (5.0, 100.0),
        (0.5, 50.0),
        (1.25, 125.0),
        (2.0, 200.0),
        (4.0, 200.0),
    ];
    for (t, expected) in cases {
        let got = curve.thrust_at(t).unwrap();
        assert_eq!(got.to_bits(), expected.to_bits(), "thrust_at({t})");
    }
    // Non-round knots catch a table that silently snaps to the nearest sample.
    let other = ThrustCurve::new(vec![
        ThrustSample {
            t_s: 0.0,
            thrust_n: 12.5,
        },
        ThrustSample {
            t_s: 0.3,
            thrust_n: 40.25,
        },
        ThrustSample {
            t_s: 1.7,
            thrust_n: 9.75,
        },
    ])
    .unwrap();
    assert_eq!(
        other.thrust_at(0.15).unwrap().to_bits(),
        26.375_f64.to_bits()
    );
    assert_eq!(other.thrust_at(1.0).unwrap().to_bits(), 25.0_f64.to_bits());
}

#[test]
fn thrust_curve_queries_are_bitwise_deterministic_and_order_independent() {
    let a = thrust_fixture();
    let b = thrust_fixture();
    assert_eq!(format!("{a:?}"), format!("{b:?}"));
    let times = [0.0, 0.13, 0.5, 1.0, 2.7, 3.0, 4.9, 5.0];
    let first: Vec<u64> = times
        .iter()
        .map(|&t| a.thrust_at(t).unwrap().to_bits())
        .collect();
    for _ in 0..3 {
        // Interleave other queries: nothing about a reading may depend on
        // call history, RNG state or the wall clock.
        let _ = a.thrust_at(9.0);
        let again: Vec<u64> = times
            .iter()
            .rev()
            .map(|&t| b.thrust_at(t).unwrap().to_bits())
            .collect();
        let again: Vec<u64> = again.into_iter().rev().collect();
        assert_eq!(first, again);
    }
}

#[test]
fn thrust_curve_rejects_bad_samples_loudly() {
    let s = |t, f| ThrustSample {
        t_s: t,
        thrust_n: f,
    };
    assert_loud(ThrustCurve::new(vec![]), "needs >= 2 knots");
    assert_loud(ThrustCurve::new(vec![s(0.0, 1.0)]), "needs >= 2 knots");
    assert_loud(
        ThrustCurve::new(vec![s(f64::NAN, 1.0), s(1.0, 1.0)]),
        "must be finite",
    );
    assert_loud(
        ThrustCurve::new(vec![s(0.0, f64::INFINITY), s(1.0, 1.0)]),
        "must be finite",
    );
    assert_loud(
        ThrustCurve::new(vec![s(-0.5, 1.0), s(1.0, 1.0)]),
        "negative",
    );
    assert_loud(
        ThrustCurve::new(vec![s(0.0, -1.0), s(1.0, 1.0)]),
        "negative",
    );
    assert_loud(
        ThrustCurve::new(vec![s(0.0, 1.0), s(0.0, 2.0)]),
        "strictly after",
    );
    assert_loud(
        ThrustCurve::new(vec![s(0.0, 1.0), s(2.0, 1.0), s(1.0, 1.0)]),
        "strictly after",
    );
}

#[test]
fn thrust_curve_never_extrapolates_or_clamps_a_query() {
    let curve = thrust_fixture();
    assert_loud(curve.thrust_at(-0.01), "outside measured");
    assert_loud(curve.thrust_at(5.01), "outside measured");
    assert_loud(curve.thrust_at(f64::NAN), "must be finite");
    assert_loud(curve.thrust_at(f64::NEG_INFINITY), "must be finite");
    // The exact measured endpoints are inside the domain and read back exactly.
    assert_eq!(curve.thrust_at(0.0).unwrap().to_bits(), 0.0_f64.to_bits());
    assert_eq!(curve.thrust_at(5.0).unwrap().to_bits(), 100.0_f64.to_bits());
}

// --- IMU Allan points --------------------------------------------------------

#[test]
fn imu_allan_extracts_arw_and_bias_stability_from_pure_fixture() {
    let series = allan_fixture();
    let arw = series.arw();
    assert!((arw - 0.12).abs() < 1e-12, "arw={arw}");
    let floor = 0.12_f64 / 10.0_f64.sqrt();
    let bias = series.bias_stability();
    assert!(
        (bias - floor).abs() < 1e-12,
        "bias_stability={bias}, expected {floor}"
    );
    assert_eq!(series.channel, ImuChannel::Gyro);
    assert_eq!(series.channel.name(), "gyro");
    assert_eq!(ImuChannel::Accel.name(), "accel");
}

#[test]
fn imu_allan_sigma_interpolation_is_log_log_and_deterministic() {
    let series = allan_fixture();
    // Exact knots read back bit-for-bit.
    for (i, p) in series.points().iter().enumerate() {
        assert_eq!(
            series.sigma_at(p.tau_s).unwrap().to_bits(),
            p.sigma.to_bits(),
            "knot {i}"
        );
    }
    // Halfway on a log axis is the geometric mean of tau AND of sigma; a
    // linear-in-tau reading would differ from this by ~0.07 here.
    let lo = series.points()[1];
    let hi = series.points()[2];
    let tau_mid = (lo.tau_s * hi.tau_s).sqrt();
    let sigma_mid = series.sigma_at(tau_mid).unwrap();
    assert!(
        (sigma_mid - (lo.sigma * hi.sigma).sqrt()).abs() < 1e-12,
        "sigma_at({tau_mid})={sigma_mid}"
    );
    assert_eq!(
        sigma_mid.to_bits(),
        series.sigma_at(tau_mid).unwrap().to_bits()
    );
}

#[test]
fn imu_allan_rejects_bad_points_and_queries_loudly() {
    let p = |tau, sigma| AllanPoint { tau_s: tau, sigma };
    assert_loud(
        ImuAllanPoints::new(ImuChannel::Accel, vec![p(0.1, 1.0)]),
        "needs >= 2 points",
    );
    assert_loud(
        ImuAllanPoints::new(ImuChannel::Accel, vec![p(f64::NAN, 1.0), p(1.0, 1.0)]),
        "must be finite",
    );
    assert_loud(
        ImuAllanPoints::new(ImuChannel::Accel, vec![p(0.0, 1.0), p(1.0, 1.0)]),
        "must be > 0",
    );
    assert_loud(
        ImuAllanPoints::new(ImuChannel::Accel, vec![p(0.1, 0.0), p(1.0, 1.0)]),
        "must be > 0",
    );
    assert_loud(
        ImuAllanPoints::new(ImuChannel::Accel, vec![p(1.0, 1.0), p(0.1, 1.0)]),
        "strictly after",
    );
    let series = allan_fixture();
    assert_loud(series.sigma_at(0.0), "must be > 0");
    assert_loud(series.sigma_at(0.005), "outside measured");
    assert_loud(series.sigma_at(11.0), "outside measured");
    assert_loud(series.sigma_at(f64::NAN), "must be finite");
}

// --- Servo step response -----------------------------------------------------

#[test]
fn servo_response_is_deadbanded_delayed_and_rate_limited() {
    let servo = servo_fixture();
    assert_eq!(servo.samples().len(), 4);
    let cases: [(f64, f64); 5] = [
        (1.0, 0.0),
        (-1.5, 0.0),
        (2.0, 0.0),
        (10.0, 8.0),
        (-10.0, -8.0),
    ];
    for (command, expected) in cases {
        let got = servo.effective_command(command).unwrap();
        assert_eq!(
            got.to_bits(),
            expected.to_bits(),
            "effective_command({command})"
        );
    }
    let cases: [(f64, f64); 7] = [
        (0.0, 0.0),
        (0.05, 0.0),
        (0.1, 0.0),
        (0.35, 1.0),
        (0.6, 2.0),
        (2.1, 8.0),
        (5.0, 8.0),
    ];
    for (t, expected) in cases {
        let got = servo.response_at(10.0, t).unwrap();
        // The ramp is deterministic but its intermediate products are not
        // exact in binary (0.35 - 0.1 = 0.2499999...), so compare with a
        // tolerance here; the dedicated determinism test pins bit patterns.
        assert!(
            (got - expected).abs() < 1e-12,
            "response_at(10, {t})={got}, expected {expected}"
        );
    }
    assert_eq!(
        servo.response_at(-10.0, 0.6).unwrap().to_bits(),
        (-2.0_f64).to_bits()
    );
    // A command inside the deadband never moves the output, at any time.
    assert_eq!(
        servo.response_at(1.0, 5.0).unwrap().to_bits(),
        0.0_f64.to_bits()
    );
    assert_eq!(
        servo.settle_time_s(10.0).unwrap().to_bits(),
        2.1_f64.to_bits()
    );
    assert_eq!(
        servo.settle_time_s(1.0).unwrap().to_bits(),
        0.1_f64.to_bits()
    );
}

#[test]
fn servo_contracts_are_deterministic_across_repeated_reads() {
    let a = servo_fixture();
    let b = servo_fixture();
    assert_eq!(format!("{a:?}"), format!("{b:?}"));
    let cases = [
        (0.0, 0.0),
        (9.0, 0.05),
        (9.0, 0.2),
        (-9.0, 0.6),
        (9.0, 3.0),
        (1.0, 9.0),
    ];
    let first: Vec<u64> = cases
        .iter()
        .map(|&(cmd, t)| a.response_at(cmd, t).unwrap().to_bits())
        .collect();
    for _ in 0..3 {
        let again: Vec<u64> = cases
            .iter()
            .map(|&(cmd, t)| b.response_at(cmd, t).unwrap().to_bits())
            .collect();
        assert_eq!(first, again);
    }
}

#[test]
fn servo_rejects_unidentifiable_or_nonfinite_inputs_loudly() {
    let step = |t, c, m| ServoStep {
        t_s: t,
        commanded: c,
        measured: m,
    };
    let good = vec![step(0.0, 10.0, 0.0), step(1.0, 10.0, 8.0)];
    assert_loud(
        ServoStepResponse::new(-0.1, 4.0, 2.0, good.clone()),
        "negative",
    );
    assert_loud(
        ServoStepResponse::new(0.1, 0.0, 2.0, good.clone()),
        "must be > 0",
    );
    assert_loud(
        ServoStepResponse::new(0.1, f64::NAN, 2.0, good.clone()),
        "must be finite",
    );
    assert_loud(
        ServoStepResponse::new(0.1, 4.0, -1.0, good.clone()),
        "negative",
    );
    assert_loud(
        ServoStepResponse::new(
            0.1,
            4.0,
            2.0,
            vec![step(0.0, 1.0, 0.0), step(1.0, 2.0, 0.0)],
        ),
        "never commands outside the deadband",
    );
    assert_loud(
        ServoStepResponse::new(
            0.1,
            4.0,
            2.0,
            vec![step(0.0, 10.0, 0.0), step(0.0, 10.0, 1.0)],
        ),
        "strictly after",
    );
    let servo = servo_fixture();
    assert_loud(servo.response_at(10.0, -0.1), "negative");
    assert_loud(servo.response_at(f64::NAN, 1.0), "must be finite");
    assert_loud(servo.response_at(10.0, f64::INFINITY), "must be finite");
    assert_loud(servo.settle_time_s(f64::NAN), "must be finite");
}
