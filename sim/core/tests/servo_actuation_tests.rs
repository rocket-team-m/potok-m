// SPDX-License-Identifier: GPL-3.0-or-later
//! Gates for the measurement-backed fin servo: deadband, pure delay and slew
//! rate between a held command and the deflection that reaches `Cm_delta`.
//!
//! No servo ships enabled: there is no measured step response in the repo, so
//! the legacy instantaneous fin and its trajectory are bit-for-bit unchanged.
//! Every test here installs a synthetic, documented response explicitly.

use potok_sim::dynamics::{derivative, rk4, solve_fin_deflection};
use potok_sim::identification::{ServoActuator, ServoStep, ServoStepResponse};
use potok_sim::{Config, SimError, Simulator, State, Vehicle, PLACEHOLDER_CM_DELTA};

fn response(delay_s: f64, rate_per_s: f64, deadband: f64) -> ServoStepResponse {
    ServoStepResponse::new(
        delay_s,
        rate_per_s,
        deadband,
        vec![
            ServoStep {
                t_s: 0.0,
                commanded: 0.0,
                measured: 0.0,
            },
            ServoStep {
                t_s: 0.01,
                commanded: 10.0,
                measured: 0.0,
            },
            ServoStep {
                t_s: 1.0,
                commanded: 10.0,
                measured: 10.0 - deadband,
            },
        ],
    )
    .expect("synthetic step response")
}

fn assert_close(got: f64, want: f64, tol: f64, what: &str) {
    assert!(
        (got - want).abs() <= tol * want.abs().max(1.0),
        "{what}: got {got}, want {want} (tol {tol})"
    );
}

/// The streamed actuator must reproduce the closed-form step response at every
/// tick once the delay is quantised to whole ticks. Binary-friendly numbers
/// (dt=0.125, delay=0.25) keep the tick boundary exact.
#[test]
fn held_step_matches_response_at_through_deadband_delay_and_ramp() {
    let resp = response(0.25, 4.0, 2.0);
    let dt = 0.125;
    let mut act = ServoActuator::new(resp.clone(), dt).expect("actuator");
    assert_eq!(act.delay_steps(), 2, "ceil(0.25 / 0.125) = 2 ticks");

    let command = 20.0;
    let mut t = 0.0;
    for k in 1..=40 {
        t += dt;
        let out = act.step(command).expect("finite command");
        assert_close(
            out,
            resp.response_at(command, t).expect("in-domain"),
            1e-12,
            "step {k}",
        );
        if k <= 2 {
            assert_eq!(out, 0.0, "pure delay: no motion before {t} s");
        }
    }
    // Deadband is subtracted, not clamped: the settled output is command - band.
    assert_close(act.output(), 18.0, 1e-12, "settled output");
    // The new command enters behind the two-tick delay line: two more steps
    // still drain the old command, then the output slews down by rate*dt.
    assert_eq!(act.step(1.0).expect("command inside band"), 18.0, "delay 1");
    assert_eq!(act.step(1.0).expect("command inside band"), 18.0, "delay 2");
    assert_eq!(
        act.step(1.0).expect("command inside band"),
        17.5,
        "slew down"
    );
}

/// A command entirely inside the deadband never moves the fin, even though the
/// actuator keeps ticking.
#[test]
fn command_inside_deadband_never_moves_fin() {
    let resp = response(0.0, 10.0, 0.5);
    let mut act = ServoActuator::new(resp, 0.01).expect("actuator");
    for _ in 0..50 {
        assert_eq!(act.step(0.25).expect("inside deadband"), 0.0);
    }
}

/// Invalid servo configuration is LOUD at the boundary, not clamped or
/// zero-filled.
#[test]
fn invalid_servo_config_is_loud() {
    let resp = response(0.1, 4.0, 0.5);
    for dt in [0.0, -0.01, f64::NAN, f64::INFINITY] {
        let err = ServoActuator::new(resp.clone(), dt).expect_err("invalid dt");
        match err {
            SimError::Config(msg) => assert!(msg.contains("servo actuator dt"), "{msg}"),
            other => panic!("expected Config, got {other:?}"),
        }
    }
    let bad = ServoStepResponse::new(-0.1, 4.0, 0.5, vec![]).expect_err("negative delay");
    assert!(bad.to_string().contains("delay_s"), "{bad}");
    let mut act = ServoActuator::new(resp, 0.01).expect("actuator");
    let bad = act.step(f64::NAN).expect_err("non-finite command");
    assert!(bad.to_string().contains("servo command"), "{bad}");
}

fn flight(cfg: &Config, mass: f64) -> (Vehicle, State) {
    let v = Vehicle::from_config(cfg, mass, 1.0, [0.0; 2]);
    let mut s = State::new(mass, 0.0, 0.0);
    s.pos[2] = 100.0;
    s.vel[2] = 60.0;
    (v, s)
}

/// With a servo installed the demanded deflection is delayed and rate-limited:
/// the first `delay_steps` ticks see no fin moment at all, then the deflection
/// grows by at most rate*dt each tick. The identical zero-command vehicle is
/// the oracle for "no fin moment" (not the instantaneous-command vehicle,
/// which already deflects on tick 1).
#[test]
fn wired_servo_defers_then_slews_the_fin_moment() {
    let cfg = Config::default();
    let dt = cfg.run.dt;
    let sim = Simulator::new(cfg.clone()).expect("default config");
    let atm = sim.atmosphere();
    let mass = 11.0;
    let resp = response(0.024, 0.5, 0.001);
    let delay_steps = ServoActuator::new(resp.clone(), dt)
        .expect("actuator")
        .delay_steps();

    let (mut idle, idle_state) = flight(&cfg, mass);
    let (served_plain, served_state) = flight(&cfg, mass);
    let mut served = served_plain
        .with_servo(resp.clone(), dt)
        .expect("install servo");
    assert!(idle.control_accel == [0.0; 2] && served.control_accel == [0.0; 2]);
    idle.control_accel = [0.0, 0.0];
    served.control_accel = [0.25, 0.0];

    let wind = [0.0; 3];
    let mut idle_s = idle_state;
    let mut served_s = served_state;
    let mut outputs = Vec::new();
    for k in 0..120 {
        let t = k as f64 * dt;
        idle_s = rk4(t, &idle_s, dt, wind, atm, &idle);
        served_s = rk4(t, &served_s, dt, wind, atm, &served);
        let out = served.fin_output.get();
        outputs.push(out);
        if k < delay_steps {
            assert_eq!(out, [0.0; 2], "tick {k}: still inside the pure delay");
            assert_eq!(served_s.pos, idle_s.pos, "tick {k}: position");
            assert_eq!(served_s.vel, idle_s.vel, "tick {k}: velocity");
            assert_eq!(served_s.quat, idle_s.quat, "tick {k}: attitude");
            assert_eq!(served_s.omega, idle_s.omega, "tick {k}: no fin moment yet");
        }
    }

    let first = outputs
        .iter()
        .position(|o| o[0] != 0.0)
        .expect("fin eventually moves");
    assert_eq!(
        first, delay_steps,
        "first motion is exactly after the delay line drains"
    );
    // qdyn rises during the burn so the demand itself moves; only the slew
    // bound is pinned here, not monotonicity.
    for w in outputs.windows(2) {
        let step = (w[1][0] - w[0][0]).abs();
        assert!(
            step <= 0.5 * dt + 1e-12,
            "slew must respect rate*dt: {step}"
        );
    }

    // The demand never reaches the mechanical stop at this flight condition.
    let air = atm.at(100.0);
    let qdyn0 = 0.5 * air.density * 60.0 * 60.0;
    let demand0 = solve_fin_deflection(
        0.25,
        mass,
        qdyn0,
        cfg.aero.area_m2,
        PLACEHOLDER_CM_DELTA,
        cfg.aero.control_moment_arm_m,
    )
    .expect("below the fin stop");
    assert!(
        outputs.last().unwrap()[0] > 0.5 * demand0,
        "servo tracks the demand"
    );
}

/// Zero command with a servo installed stays bit-for-bit identical to the
/// legacy instantaneous fin on every exported state field.
#[test]
fn zero_command_with_servo_is_bit_identical_to_legacy() {
    let cfg = Config::default();
    let dt = cfg.run.dt;
    let sim = Simulator::new(cfg.clone()).expect("default config");
    let atm = sim.atmosphere();
    let (legacy, legacy_s) = flight(&cfg, 11.0);
    let (plain, plain_s) = flight(&cfg, 11.0);
    let served = plain
        .with_servo(response(0.024, 0.5, 0.001), dt)
        .expect("install servo");

    let mut a = legacy_s;
    let mut b = plain_s;
    for k in 0..200 {
        let t = k as f64 * dt;
        a = rk4(t, &a, dt, [0.0; 3], atm, &legacy);
        b = rk4(t, &b, dt, [0.0; 3], atm, &served);
        assert_eq!(a.pos, b.pos, "pos at {k}");
        assert_eq!(a.vel, b.vel, "vel at {k}");
        assert_eq!(a.quat, b.quat, "quat at {k}");
        assert_eq!(a.omega, b.omega, "omega at {k}");
        assert_eq!(a.mass, b.mass, "mass at {k}");
    }
    assert_eq!(served.fin_output.get(), [0.0; 2]);
}

/// A residual deflection keeps producing a moment after the command returns
/// to zero, until the servo slews home. This is the difference between a
/// rate-limited actuator and an instantaneous one that simply snaps to zero.
#[test]
fn residual_deflection_applies_after_command_returns_to_zero() {
    let cfg = Config::default();
    let dt = cfg.run.dt;
    let sim = Simulator::new(cfg.clone()).expect("default config");
    let atm = sim.atmosphere();
    let (_, state) = flight(&cfg, 11.0);
    let (plain, _) = flight(&cfg, 11.0);
    let mut v = plain
        .with_servo(response(0.0, 0.5, 0.001), dt)
        .expect("install servo");

    v.control_accel = [0.25, 0.0];
    let mut s = state;
    for k in 0..60 {
        s = rk4(k as f64 * dt, &s, dt, [0.0; 3], atm, &v);
    }
    let deflected = v.fin_output.get();
    assert!(deflected[0] > 0.0, "servo has deflected");

    v.control_accel = [0.0, 0.0];
    s = rk4(60.0 * dt, &s, dt, [0.0; 3], atm, &v);
    let residual = v.fin_output.get();
    assert!(
        residual[0] > 0.0 && residual[0] < deflected[0],
        "slewing home: {residual:?}"
    );

    let zero_d = derivative(60.0 * dt + dt, &s, [0.0; 3], atm, &v);
    v.fin_output.set([0.0; 2]); // force the fin neutral for the control reading
    let neutral_d = derivative(60.0 * dt + dt, &s, [0.0; 3], atm, &v);
    assert!(
        zero_d.omega[0] > neutral_d.omega[0],
        "residual fin deflection must still apply a moment"
    );
}
