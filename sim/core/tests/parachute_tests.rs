// SPDX-License-Identifier: GPL-3.0-or-later
//! CHUTE01 gates for the deterministic two-stage recovery canopy.
//!
//! The module is included by path so this suite compiles without touching
//! `lib.rs` (the crate owner wires `pub mod parachute;` when integrating).
//! These tests pin: opt-in inertness, LOUD config rejection, the drogue->main
//! one-shot sequence, descent-rate settling, true-wind advection, the absence
//! of any control input, bit-identical replay, and the separation of canopy
//! landing dispersion from fin-flight lateral RMS.

#[path = "../src/parachute.rs"]
mod parachute;

use parachute::*;

fn ctx(t: f64, alt: f64, vz: f64, x: f64, y: f64, wind: [f64; 2]) -> FlightCtx {
    FlightCtx {
        t_s: t,
        altitude_m: alt,
        vz_mps: vz,
        pos_m: [x, y],
        vel_xy_mps: [0.0, 0.0],
        true_wind_xy: wind,
        t_apogee_s: Some(0.0),
    }
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn default_is_opt_in_and_disabled_is_inert() {
    let cfg = ParachuteConfig::default();
    assert!(!cfg.enabled, "the shipped default must preserve existing A/Bs");
    cfg.validate().expect("the shipped default must be a valid config");

    let mut p = Parachute::new(cfg).unwrap();
    let mut c = ctx(0.0, 3000.0, -40.0, 100.0, 50.0, [2.0, 0.0]);
    for _ in 0..200 {
        p.step(0.1, c).unwrap();
        c.t_s += 0.1;
        c.altitude_m -= 4.0;
    }
    assert_eq!(p.state().phase, ChutePhase::Stowed);
    assert_eq!(p.state().t_s, 0.0, "a disabled model must not track time");
    assert!(p.landing().is_none());
}

#[test]
fn invalid_config_is_loud_names_the_field_and_offers_an_escape() {
    let mut cases: Vec<(&'static str, ParachuteConfig)> = Vec::new();

    let mut c = ParachuteConfig::default();
    c.wind_coupling_s = 0.0;
    cases.push(("wind_coupling_s", c));

    let mut c = ParachuteConfig::default();
    c.drogue.target_descent_mps = 0.0;
    cases.push(("drogue.target_descent_mps", c));

    let mut c = ParachuteConfig::default();
    c.main.inflation_s = -1.0;
    cases.push(("main.inflation_s", c));

    let mut c = ParachuteConfig::default();
    c.drogue.trigger = DeployTrigger::AltitudeM { value: f64::NAN };
    cases.push(("drogue.trigger", c));

    let mut c = ParachuteConfig::default();
    c.drogue.trigger = DeployTrigger::AltitudeM { value: 3000.0 };
    c.main.trigger = DeployTrigger::AltitudeM { value: 3500.0 };
    cases.push(("main.trigger", c));

    let mut c = ParachuteConfig::default();
    c.main.target_descent_mps = 30.0;
    cases.push(("main.target_descent_mps", c));

    let mut c = ParachuteConfig::default();
    c.landing_altitude_m = -1.0;
    cases.push(("landing_altitude_m", c));

    let mut c = ParachuteConfig::default();
    c.main.trigger = DeployTrigger::TimeSinceLaunchS { value: 0.0 };
    cases.push(("main.trigger", c));

    for (field, cfg) in cases {
        let err = cfg.validate().unwrap_err();
        assert_eq!(err.field, field, "wrong field named for {field}");
        let msg = err.to_string();
        assert!(msg.contains(&format!("parachute.{field}")), "{msg}");
        assert!(msg.contains("fix the value"), "no escape offered: {msg}");
        assert!(
            Parachute::new(cfg).is_err(),
            "construction must reject the same bad config"
        );
    }
}

#[test]
fn drogue_then_main_is_a_one_shot_sequence() {
    let mut cfg = ParachuteConfig::default();
    cfg.enabled = true;
    cfg.drogue.trigger = DeployTrigger::TimeSinceLaunchS { value: 2.0 };
    cfg.main.trigger = DeployTrigger::TimeSinceLaunchS { value: 12.0 };
    let mut p = Parachute::new(cfg.clone()).unwrap();

    let dt = 0.2;
    let mut c = ctx(0.0, 3000.0, -20.0, 100.0, 50.0, [3.0, 4.0]);
    let mut seen = vec![ChutePhase::Stowed];
    for i in 0..5000 {
        p.step(dt, c).unwrap();
        let s = p.state().clone();
        if seen.last() != Some(&s.phase) {
            seen.push(s.phase);
        }
        if s.phase == ChutePhase::Landed {
            break;
        }
        c.t_s = (i + 1) as f64 * dt;
        if s.phase == ChutePhase::Stowed {
            c.altitude_m -= 20.0 * dt;
        } else {
            c.altitude_m = s.altitude_m;
            c.vz_mps = s.vz_mps;
            c.pos_m = s.pos_m;
            c.vel_xy_mps = s.vel_xy_mps;
        }
    }

    assert_eq!(
        seen,
        vec![
            ChutePhase::Stowed,
            ChutePhase::DrogueInflating,
            ChutePhase::Drogue,
            ChutePhase::MainInflating,
            ChutePhase::Main,
            ChutePhase::Landed,
        ]
    );
    assert!(close(p.state().drogue_t_s.unwrap(), 2.0, 1e-9));
    assert!(close(p.state().main_t_s.unwrap(), 12.0, 1e-9));
    assert_eq!(p.state().altitude_m, 0.0, "landing clamps to landing altitude");
    assert_eq!(p.state().vz_mps, 0.0);
    assert!(p.landing().is_some());

    // Redeploy after landing is a no-op: the latch cannot be re-armed by input.
    let before = p.state().clone();
    let landing = p.landing().unwrap().clone();
    let mut wild = ctx(999.0, 5000.0, -99.0, 9.0, 9.0, [50.0, 50.0]);
    wild.t_apogee_s = Some(1.0);
    for _ in 0..20 {
        p.step(1.0, wild).unwrap();
    }
    assert_eq!(*p.state(), before, "landed state must be frozen");
    assert_eq!(*p.landing().unwrap(), landing, "landing fires exactly once");

    // An explicit reset starts a fresh episode with no latches.
    p.reset();
    assert_eq!(p.state().phase, ChutePhase::Stowed);
    assert!(p.landing().is_none());
}

#[test]
fn descent_rate_settles_into_each_stage_band() {
    let mut cfg = ParachuteConfig::default();
    cfg.enabled = true;
    cfg.drogue.trigger = DeployTrigger::TimeSinceLaunchS { value: 0.5 };
    cfg.main.trigger = DeployTrigger::TimeSinceLaunchS { value: 20.0 };
    let mut p = Parachute::new(cfg).unwrap();

    let dt = 0.2;
    let mut c = ctx(0.0, 6000.0, -40.0, 0.0, 0.0, [0.0, 0.0]);
    let mut drogue_vz = f64::NAN;
    let mut main_vz = f64::NAN;
    for _ in 0..500 {
        p.step(dt, c).unwrap();
        let s = p.state().clone();
        if s.phase == ChutePhase::Drogue && close(s.t_s, 15.0, 0.11) {
            drogue_vz = s.vz_mps;
        }
        if s.phase == ChutePhase::Main && close(s.t_s, 60.0, 0.11) {
            main_vz = s.vz_mps;
        }
        c.t_s += dt;
        if s.phase == ChutePhase::Stowed {
            c.altitude_m -= 40.0 * dt;
        } else {
            c.altitude_m = s.altitude_m;
            c.vz_mps = s.vz_mps;
        }
    }
    assert!(close(drogue_vz, -25.0, 0.5), "drogue vz = {drogue_vz}");
    assert!(close(main_vz, -6.5, 0.5), "main vz = {main_vz}");
    assert!(p.landing().is_none(), "6000 m must outlast this window");
}

#[test]
fn horizontal_motion_advects_to_the_true_wind() {
    let mut cfg = ParachuteConfig::default();
    cfg.enabled = true;
    cfg.drogue.trigger = DeployTrigger::TimeSinceLaunchS { value: 0.1 };
    cfg.main.trigger = DeployTrigger::TimeSinceLaunchS { value: 0.2 };
    let mut p = Parachute::new(cfg.clone()).unwrap();
    assert!(p.config().enabled, "config accessor must expose the opt-in flag");

    let dt = 0.2;
    let mut c = ctx(0.0, 5000.0, -10.0, 0.0, 0.0, [4.0, 0.0]);
    for _ in 0..200 {
        p.step(dt, c).unwrap();
        let s = p.state().clone();
        c.t_s += dt;
        if s.phase == ChutePhase::Stowed {
            c.altitude_m -= 10.0 * dt;
        } else {
            c.altitude_m = s.altitude_m;
            c.vz_mps = s.vz_mps;
            c.pos_m = s.pos_m;
            c.vel_xy_mps = s.vel_xy_mps;
        }
    }
    let settled = p.state().vel_xy_mps;
    assert!(close(settled[0], 4.0, 1e-3), "vx = {}", settled[0]);
    assert!(close(settled[1], 0.0, 1e-6), "vy = {}", settled[1]);

    let x0 = p.state().pos_m[0];
    let v0 = p.state().vel_xy_mps[0];
    c.true_wind_xy = [-4.0, 0.0];
    let steps = 200;
    for _ in 0..steps {
        p.step(dt, c).unwrap();
        let s = p.state().clone();
        c.t_s += dt;
        c.altitude_m = s.altitude_m;
        c.vz_mps = s.vz_mps;
        c.pos_m = s.pos_m;
        c.vel_xy_mps = s.vel_xy_mps;
    }
    assert!(close(p.state().vel_xy_mps[0], -4.0, 1e-3));
    let drift = p.state().pos_m[0] - x0;
    // Exact discrete first-order coupling: v_{k+1} = w + (v_k - w) r,
    // r = exp(-dt/tau); x integrates the *updated* velocity each step.
    let r = (-dt / 3.0).exp();
    let expected = -4.0 * dt * steps as f64
        + (v0 + 4.0) * dt * r * (1.0 - r.powi(steps as i32)) / (1.0 - r);
    assert!(close(drift, expected, 1e-6), "drift = {drift}, expected {expected}");
}

#[test]
fn no_active_control_input_exists_under_canopy() {
    assert!(!AI_AUTHORITY_UNDER_CANOPY);
    // The exact step signature has no command parameter. Pinning the fn pointer
    // makes adding one a compile error in this gate.
    let _: fn(&mut Parachute, f64, FlightCtx) -> Result<(), ChuteError> = Parachute::step;
}

#[test]
fn replay_is_bit_identical() {
    let mut cfg = ParachuteConfig::default();
    cfg.enabled = true;
    let mut a = Parachute::new(cfg.clone()).unwrap();
    let mut b = Parachute::new(cfg).unwrap();
    let dt = 0.1;
    let mut c = ctx(0.0, 2000.0, -30.0, 12.0, 7.0, [3.0, -2.0]);
    for _ in 0..500 {
        a.step(dt, c).unwrap();
        b.step(dt, c).unwrap();
        c.t_s += dt;
        if a.state().phase == ChutePhase::Stowed {
            c.altitude_m -= 3.0;
        } else {
            c.altitude_m = a.state().altitude_m;
            c.vz_mps = a.state().vz_mps;
        }
    }
    let x = a.state();
    let y = b.state();
    assert_eq!(x.phase, y.phase);
    for (p, q) in [
        (x.t_s, y.t_s),
        (x.altitude_m, y.altitude_m),
        (x.vz_mps, y.vz_mps),
        (x.inflation, y.inflation),
        (x.pos_m[0], y.pos_m[0]),
        (x.pos_m[1], y.pos_m[1]),
        (x.vel_xy_mps[0], y.vel_xy_mps[0]),
        (x.vel_xy_mps[1], y.vel_xy_mps[1]),
    ] {
        assert_eq!(p.to_bits(), q.to_bits(), "replay diverged");
    }
}

#[test]
fn typed_triggers_and_descent_confirmation() {
    let mut cfg = ParachuteConfig::default();
    cfg.enabled = true;
    cfg.drogue.trigger = DeployTrigger::AltitudeM { value: 1000.0 };
    cfg.main.trigger = DeployTrigger::AltitudeM { value: 500.0 };
    let mut p = Parachute::new(cfg.clone()).unwrap();

    // Altitude reached while still rising: no deployment with descent confirm.
    let rising = ctx(0.0, 900.0, 5.0, 0.0, 0.0, [0.0, 0.0]);
    p.step(0.1, rising).unwrap();
    assert_eq!(p.state().phase, ChutePhase::Stowed);

    let falling = ctx(0.1, 900.0, -5.0, 0.0, 0.0, [0.0, 0.0]);
    p.step(0.1, falling).unwrap();
    assert!(p.state().phase.under_canopy(), "drogue must fire on descent");

    // Without confirmation the same altitude fires while rising.
    cfg.require_descent_confirm = false;
    let mut q = Parachute::new(cfg.clone()).unwrap();
    q.step(0.1, rising).unwrap();
    assert!(q.state().phase.under_canopy());

    // Time-since-apogee waits for a real apogee time; it never fires on None.
    let mut cfg2 = ParachuteConfig::default();
    cfg2.enabled = true;
    cfg2.drogue.trigger = DeployTrigger::TimeSinceApogeeS { value: 2.0 };
    assert!(cfg2.drogue.trigger.needs_apogee());
    assert!(!cfg.drogue.trigger.needs_apogee());
    let mut r = Parachute::new(cfg2).unwrap();
    let mut no_ap = ctx(10.0, 1000.0, -5.0, 0.0, 0.0, [0.0, 0.0]);
    no_ap.t_apogee_s = None;
    r.step(0.1, no_ap).unwrap();
    assert_eq!(r.state().phase, ChutePhase::Stowed);
    let mut with_ap = ctx(3.0, 1000.0, -5.0, 0.0, 0.0, [0.0, 0.0]);
    with_ap.t_apogee_s = Some(0.5);
    r.step(0.1, with_ap).unwrap();
    assert!(r.state().phase.under_canopy());
}

#[test]
fn truth_and_manifest_fields_are_named_and_dispersion_is_separate() {
    assert_eq!(CHUTE_TRUTH_NAMES.len(), 8);
    assert_eq!(CHUTE_MANIFEST_FIELDS.len(), 9);
    let mut seen = std::collections::BTreeSet::<&str>::new();
    for name in CHUTE_MANIFEST_FIELDS {
        assert!(seen.insert(name), "duplicate manifest field {name}");
    }
    assert!(
        !seen.contains("lateral_rms_m"),
        "canopy dispersion must not overwrite the fin-flight RMS"
    );

    let st = ParachuteState {
        phase: ChutePhase::Main,
        t_s: 42.0,
        altitude_m: 800.0,
        vz_mps: -6.5,
        pos_m: [1200.0, 0.0],
        vel_xy_mps: [3.0, 4.0],
        inflation: 1.0,
        drogue_t_s: Some(10.0),
        main_t_s: Some(20.0),
    };
    let truth = st.truth_floats([3.0, 4.0]);
    assert_eq!(truth.len(), CHUTE_TRUTH_NAMES.len());
    assert_eq!(truth[0], ChutePhase::Main.code() as f32);
    assert_eq!(ChutePhase::Main.as_str(), "main");
    assert_eq!(truth[7], 0.0, "not landed yet");

    let landing = LandingDispersion {
        deploy_pos_m: [1000.0, 0.0],
        deploy_altitude_m: 1500.0,
        landing_pos_m: [1200.0, 0.0],
        landing_t_s: 250.0,
        drogue_t_s: 10.0,
        main_t_s: 20.0,
        settled_descent_mps: 6.5,
    };
    assert!(close(landing.lateral_offset_m(), 200.0, 1e-12));
    assert!(close(landing.radial_from_pad_m(), 1200.0, 1e-12));
    let fields = landing.manifest_named_floats(true);
    assert_eq!(fields.len(), CHUTE_MANIFEST_FIELDS.len());
    for ((name, _), expected) in fields.iter().zip(CHUTE_MANIFEST_FIELDS.iter()) {
        assert_eq!(name, expected);
    }
    assert!(close(fields[7].1, 200.0, 1e-12), "dispersion field");
    assert!(close(fields[8].1, 1200.0, 1e-12), "total miss field");
}
