// SPDX-License-Identifier: GPL-3.0-or-later
// The path-included module is a test-only shim; the symbols it exports are
// exercised by the library and by the integration gates, not by every target.
#![allow(dead_code)]
//! SIMCANOPY gates: typed descent calibration and wind-drift integrals.
//!
//! The module is included by path so these gates compile without another
//! crate-owner wiring step. No drop test exists here: the shipped speeds must
//! stay explicitly PLACEHOLDER, and a MEASURED claim must carry a signature
//! bound to the exact value. The wind gates compare the integrator to the
//! closed-form integral for constant wind — analytic identity, never a tuned
//! landing target.

#[path = "../src/parachute.rs"]
mod parachute;

use parachute::*;

const CALIBRATION_TOML: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../config/parachute.toml");

fn load_config_file() -> ParachuteConfig {
    let text = std::fs::read_to_string(CALIBRATION_TOML).expect("config/parachute.toml must exist");
    toml::from_str(&text).expect("config/parachute.toml must deserialize into ParachuteConfig")
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

fn tol(x: f64) -> f64 {
    1e-8 * x.abs().max(1.0)
}

// --- Calibration provenance --------------------------------------------------

#[test]
fn shipped_defaults_are_explicitly_placeholder_not_measured() {
    let cfg = ParachuteConfig::default();
    assert!(!cfg.enabled, "the shipped default stays opt-in");
    assert_eq!(cfg.drogue.descent.source.label(), "PLACEHOLDER");
    assert_eq!(cfg.main.descent.source.label(), "PLACEHOLDER");
    assert!(!cfg.drogue.descent.source.is_measured());
    assert!(!cfg.main.descent.source.is_measured());
    assert_eq!(cfg.drogue.descent.value_mps, Some(25.0));
    assert_eq!(cfg.main.descent.value_mps, Some(6.5));
    cfg.calibration_harness().unwrap();
    cfg.validate().unwrap();

    // config/default.toml predates the typed section: a numeric-only stage is
    // labelled PLACEHOLDER, never silently promoted, and still validates.
    let mut legacy = cfg.clone();
    legacy.main.descent = DescentCalibration::default();
    legacy.validate().unwrap();
    assert_eq!(legacy.main.descent.source.label(), "PLACEHOLDER");
    assert_eq!(
        legacy
            .main
            .descent
            .value_mps(legacy.main.target_descent_mps),
        6.5
    );
}

#[test]
fn config_file_carries_no_measurement_and_no_uncommented_measured_label() {
    let text = std::fs::read_to_string(CALIBRATION_TOML).expect("config/parachute.toml must exist");
    for line in text.lines() {
        let content = line.split('#').next().unwrap_or("").trim();
        if !content.is_empty() {
            assert!(
                !content.contains("kind = \"measured\""),
                "an uncommented MEASURED label needs signed provenance: {content}"
            );
        }
    }

    let cfg = load_config_file();
    cfg.validate().unwrap();
    assert_eq!(cfg.drogue.descent.value_mps, Some(25.0));
    assert_eq!(cfg.main.descent.value_mps, Some(6.5));
    assert_eq!(cfg.drogue.descent.source.label(), "PLACEHOLDER");
    assert_eq!(cfg.main.descent.source.label(), "PLACEHOLDER");
    assert_eq!(
        toml::from_str::<ParachuteConfig>(&toml::to_string(&cfg).unwrap()).unwrap(),
        cfg
    );
}

#[test]
fn measured_without_signed_provenance_is_refused_loudly() {
    let mut cfg = load_config_file();
    cfg.drogue.descent.source = DescentSource::Measured {
        provenance: SignedProvenance {
            record_id: "DROP-001".to_string(),
            signer: "Test Conductor".to_string(),
            date: "2026-11-24".to_string(),
            signature: String::new(),
        },
    };
    let err = cfg.calibration_harness().unwrap_err();
    assert_eq!(err.field, "drogue.descent.source");
    assert!(err.to_string().contains("MEASURED"));
    assert!(err.to_string().contains("fix the value"));
    assert_eq!(cfg.validate().unwrap_err().field, "drogue.descent.source");
    assert!(
        Parachute::new(cfg).is_err(),
        "construction must reject it too"
    );

    let mut cfg = load_config_file();
    cfg.main.descent.source = DescentSource::Measured {
        provenance: SignedProvenance {
            record_id: String::new(),
            signer: String::new(),
            date: String::new(),
            signature: "0".repeat(64),
        },
    };
    assert_eq!(
        cfg.calibration_harness().unwrap_err().field,
        "main.descent.source"
    );
}

#[test]
fn signed_provenance_promotes_only_its_exact_value_and_stage() {
    let mut cfg = load_config_file();
    let prov = SignedProvenance::sign(25.0, "DROP-001", "A. Signer", "2026-11-24");
    assert_eq!(prov.signature.len(), 64);
    assert!(prov
        .signature
        .chars()
        .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)));

    cfg.drogue.descent.source = DescentSource::Measured {
        provenance: prov.clone(),
    };
    cfg.calibration_harness().unwrap();
    cfg.validate().unwrap();
    Parachute::new(cfg.clone()).unwrap();

    // Both mirrors moved together: the numeric cross-check passes, but the
    // signature still commits to 25.0, so the claim must fail.
    let mut retuned = cfg.clone();
    retuned.drogue.target_descent_mps = 24.5;
    retuned.drogue.descent.value_mps = Some(24.5);
    assert_eq!(
        retuned.calibration_harness().unwrap_err().field,
        "drogue.descent.source"
    );

    // Moving only the numeric mirror breaks the typed/numeric binding first.
    let mut drifted = cfg.clone();
    drifted.drogue.target_descent_mps = 24.5;
    assert_eq!(
        drifted.calibration_harness().unwrap_err().field,
        "drogue.descent.value_mps"
    );

    for (label, bad) in [
        (
            "record",
            SignedProvenance {
                record_id: "DROP-002".to_string(),
                ..prov.clone()
            },
        ),
        (
            "signer",
            SignedProvenance {
                signer: "B. Other".to_string(),
                ..prov.clone()
            },
        ),
        (
            "date",
            SignedProvenance {
                date: "2025-01-01".to_string(),
                ..prov.clone()
            },
        ),
        (
            "signature",
            SignedProvenance {
                signature: "f".repeat(64),
                ..prov.clone()
            },
        ),
    ] {
        let mut tampered = cfg.clone();
        tampered.drogue.descent.source = DescentSource::Measured { provenance: bad };
        assert_eq!(
            tampered.calibration_harness().unwrap_err().field,
            "drogue.descent.source",
            "tampering with {label} was accepted"
        );
    }

    // A real signature for the other stage is still rejected on this stage.
    let mut swapped = cfg.clone();
    swapped.main.descent.source = DescentSource::Measured { provenance: prov };
    assert_eq!(
        swapped.calibration_harness().unwrap_err().field,
        "main.descent.source"
    );
    swapped.main.descent.source = DescentSource::Measured {
        provenance: SignedProvenance::sign(6.5, "DROP-002", "A. Signer", "2026-11-24"),
    };
    swapped.calibration_harness().unwrap();
}

#[test]
fn changing_a_speed_to_chase_a_landing_target_never_buys_provenance() {
    for speed in [4.0, 6.5, 9.0] {
        let mut cfg = load_config_file();
        cfg.main.target_descent_mps = speed;
        cfg.main.descent.value_mps = Some(speed);
        cfg.calibration_harness()
            .unwrap_or_else(|e| panic!("a placeholder speed {speed} must stay legal: {e}"));
        cfg.main.descent.source = DescentSource::Measured {
            provenance: SignedProvenance {
                record_id: "DROP-001".to_string(),
                signer: "A. Signer".to_string(),
                date: "2026-11-24".to_string(),
                signature: String::new(),
            },
        };
        assert!(
            cfg.calibration_harness().is_err(),
            "speed {speed} acquired a MEASURED claim without a drop test"
        );
    }
}

// --- Wind-drift integral gates -----------------------------------------------

fn canopy_cfg(tau_s: f64) -> ParachuteConfig {
    let mut cfg = ParachuteConfig {
        enabled: true,
        ..Default::default()
    };
    cfg.drogue.trigger = DeployTrigger::TimeSinceLaunchS { value: 0.01 };
    cfg.main.trigger = DeployTrigger::TimeSinceLaunchS { value: 1e5 };
    cfg.wind_coupling_s = tau_s;
    cfg.landing_altitude_m = 0.0;
    cfg
}

struct DriftRun {
    /// Deploy-relative horizontal position after each integrating step.
    offsets: Vec<[f64; 2]>,
    final_vel: [f64; 2],
    landed: bool,
    landing: Option<LandingDispersion>,
}

fn run_wind(
    cfg: &ParachuteConfig,
    alt0: f64,
    pos0: [f64; 2],
    v0: [f64; 2],
    wind: [f64; 2],
    dt: f64,
    steps: usize,
) -> DriftRun {
    let mut p = Parachute::new(cfg.clone()).unwrap();
    let t0 = cfg.drogue.trigger.value();
    let mut c = FlightCtx {
        t_s: t0,
        altitude_m: alt0,
        vz_mps: -20.0,
        pos_m: pos0,
        vel_xy_mps: v0,
        true_wind_xy: wind,
        t_apogee_s: Some(0.0),
    };
    p.step(dt, c).unwrap(); // deployment tick records state, integrates nothing
    assert!(p.state().phase.under_canopy());
    let deploy = p.state().pos_m;

    let mut out = DriftRun {
        offsets: Vec::with_capacity(steps),
        final_vel: [0.0; 2],
        landed: false,
        landing: None,
    };
    for k in 1..=steps {
        let s = p.state();
        c.t_s = t0 + k as f64 * dt;
        c.altitude_m = s.altitude_m;
        c.vz_mps = s.vz_mps;
        c.pos_m = s.pos_m;
        c.vel_xy_mps = s.vel_xy_mps;
        c.true_wind_xy = wind;
        p.step(dt, c).unwrap();
        let s = p.state();
        out.offsets
            .push([s.pos_m[0] - deploy[0], s.pos_m[1] - deploy[1]]);
        if s.phase == ChutePhase::Landed {
            out.landed = true;
            out.landing = p.landing().cloned();
            break;
        }
    }
    out.final_vel = p.state().vel_xy_mps;
    out
}

/// Exact solution of the discrete update `v_{k+1} = w + (v_k - w) r`,
/// `x_{k+1} = x_k + v_{k+1} dt`, with `r = exp(-dt/tau)`.
fn analytic_drift(w: [f64; 2], v0: [f64; 2], dt: f64, tau: f64, n: usize) -> [f64; 2] {
    let r = (-dt / tau).exp();
    let geom = dt * r * (1.0 - r.powi(n as i32)) / (1.0 - r);
    let nf = n as f64;
    [
        w[0] * dt * nf + (v0[0] - w[0]) * geom,
        w[1] * dt * nf + (v0[1] - w[1]) * geom,
    ]
}

fn analytic_vel(w: [f64; 2], v0: [f64; 2], dt: f64, tau: f64, n: usize) -> [f64; 2] {
    let r = (-dt / tau).exp().powi(n as i32);
    [w[0] + (v0[0] - w[0]) * r, w[1] + (v0[1] - w[1]) * r]
}

type WindCase = ([f64; 2], [f64; 2], f64, f64, usize);

fn continuous_drift(w: [f64; 2], v0: [f64; 2], t: f64, tau: f64) -> [f64; 2] {
    let decay = 1.0 - (-t / tau).exp();
    [
        w[0] * t + (v0[0] - w[0]) * tau * decay,
        w[1] * t + (v0[1] - w[1]) * tau * decay,
    ]
}

#[test]
fn constant_wind_drift_matches_the_exact_discrete_integral() {
    // wind, initial velocity, dt, coupling tau, steps (high altitude: no landing)
    let cases: [WindCase; 4] = [
        ([4.0, 0.0], [0.0, 0.0], 0.05, 3.0, 600),
        ([-3.0, 4.0], [2.0, -1.0], 0.2, 3.0, 300),
        ([0.0, -5.0], [6.0, 0.0], 1.0, 0.7, 120),
        ([0.0, 0.0], [7.0, -2.0], 0.5, 3.0, 200),
    ];
    for (wind, v0, dt, tau, n) in cases {
        let run = run_wind(&canopy_cfg(tau), 1e6, [0.0; 2], v0, wind, dt, n);
        assert!(!run.landed);
        assert_eq!(run.offsets.len(), n);
        for (k, offset) in run.offsets.iter().enumerate() {
            let expected = analytic_drift(wind, v0, dt, tau, k + 1);
            for axis in 0..2 {
                assert!(
                    close(offset[axis], expected[axis], tol(expected[axis])),
                    "wind={wind:?} dt={dt} tau={tau} step={} axis={axis}: {} vs {}",
                    k + 1,
                    offset[axis],
                    expected[axis]
                );
            }
        }
        let v = analytic_vel(wind, v0, dt, tau, n);
        for axis in 0..2 {
            assert!(
                close(run.final_vel[axis], v[axis], tol(v[axis])),
                "final velocity axis={axis}: {} vs {}",
                run.final_vel[axis],
                v[axis]
            );
        }
    }
}

#[test]
fn small_step_drift_tracks_the_continuous_constant_wind_integral() {
    let dt = 0.01;
    let n = 1000;
    let wind = [3.0, -4.0];
    let v0 = [0.0, 0.0];
    let run = run_wind(&canopy_cfg(3.0), 1e6, [0.0; 2], v0, wind, dt, n);
    let expected = continuous_drift(wind, v0, n as f64 * dt, 3.0);
    for axis in 0..2 {
        // The integrator is first-order in dt; at dt/tau = 1/300 the bias is
        // well inside 0.1 %. This gates the continuous integral, not a target.
        assert!(
            close(
                run.offsets[n - 1][axis],
                expected[axis],
                0.001 * expected[axis].abs()
            ),
            "axis={axis}: {} vs {}",
            run.offsets[n - 1][axis],
            expected[axis]
        );
    }
}

#[test]
fn wind_integral_is_linear_and_rotates_with_the_wind() {
    let cfg = canopy_cfg(2.5);
    let (dt, n) = (0.2, 200);
    let a = run_wind(&cfg, 1e6, [0.0; 2], [0.0; 2], [3.0, 0.0], dt, n).offsets[n - 1];
    let b = run_wind(&cfg, 1e6, [0.0; 2], [0.0; 2], [0.0, 3.0], dt, n).offsets[n - 1];
    let both = run_wind(&cfg, 1e6, [0.0; 2], [0.0; 2], [3.0, 3.0], dt, n).offsets[n - 1];
    assert!(close(a[0], b[1], tol(a[0])));
    assert!(
        close(a[1], 0.0, 1e-9),
        "pure x wind has no y drift: {}",
        a[1]
    );
    assert!(
        close(b[0], 0.0, 1e-9),
        "pure y wind has no x drift: {}",
        b[0]
    );
    assert!(close(both[0], a[0] + b[0], tol(both[0])));
    assert!(close(both[1], a[1] + b[1], tol(both[1])));
}

#[test]
fn piecewise_constant_wind_matches_segmented_analytic_integral() {
    let (tau, dt, n1, n2) = (3.0, 0.25, 80, 120);
    let (w1, w2, v0) = ([4.0, 1.0], [-2.0, 5.0], [1.0, -1.0]);
    let cfg = canopy_cfg(tau);
    let first = run_wind(&cfg, 1e6, [0.0; 2], v0, w1, dt, n1);
    let v1 = analytic_vel(w1, v0, dt, tau, n1);
    let second = run_wind(&cfg, 1e6, first.offsets[n1 - 1], v1, w2, dt, n2);
    let expected_first = analytic_drift(w1, v0, dt, tau, n1);
    let expected_second = analytic_drift(w2, v1, dt, tau, n2);
    for axis in 0..2 {
        let got = first.offsets[n1 - 1][axis] + second.offsets[n2 - 1][axis];
        let expected = expected_first[axis] + expected_second[axis];
        assert!(
            close(got, expected, tol(expected)),
            "axis={axis}: {got} vs {expected}"
        );
    }
}

#[test]
fn landing_dispersion_is_the_canopy_wind_integral() {
    let (tau, dt, wind, v0) = (3.0, 0.5, [4.0, -3.0], [1.5, 0.0]);
    let run = run_wind(&canopy_cfg(tau), 120.0, [0.0; 2], v0, wind, dt, 400);
    assert!(run.landed, "120 m of canopy descent must reach the ground");
    let n = run.offsets.len();
    let expected = analytic_drift(wind, v0, dt, tau, n);
    let landing = run.landing.unwrap();
    for axis in 0..2 {
        assert!(
            close(
                landing.landing_pos_m[axis],
                expected[axis],
                tol(expected[axis])
            ),
            "landing axis={axis}: {} vs {}",
            landing.landing_pos_m[axis],
            expected[axis]
        );
    }
    let expected_offset = expected[0].hypot(expected[1]);
    assert!(
        close(
            landing.lateral_offset_m(),
            expected_offset,
            tol(expected_offset)
        ),
        "dispersion {} vs integral {expected_offset}",
        landing.lateral_offset_m()
    );
}
