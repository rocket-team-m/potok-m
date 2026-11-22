// SPDX-License-Identifier: GPL-3.0-or-later
//! Hard gates on the command -> dynamics seam.
//!
//! A command is a desired lateral acceleration in m/s^2, converted to a fin
//! deflection by a moment balance inside `dynamics::derivative`. These tests
//! pin that mapping end to end: the moment that reaches the state derivative,
//! the unit of the command, the mechanical stop, the bit-identical zero path,
//! the phase/health interlocks, and the sign of the closed-loop effect.

use std::f64::consts::PI;

use potok_sim::dynamics::derivative;
use potok_sim::sensor::{Frame, FLAG_AI_READY, FLAG_FAULT};
use potok_sim::{
    fin_clamp_uses, fin_no_authority_uses, lateral_rms_m, solve_fin_deflection, validate_command,
    Config, ControlFault, Phase, ReferenceOptimizer, SimError, Simulator, State, Vehicle,
    MAX_COMMAND_MPS2, MAX_CONTROL_DEG, MAX_FIN_DEFLECTION_RAD, MIN_CONTROL_QDYN_PA,
    PLACEHOLDER_CM_DELTA,
};

const HELD_OUT: [u64; 4] = [16, 17, 18, 19];

fn sim(seed: u64) -> Simulator {
    let mut cfg = Config::default();
    cfg.run.seed = seed;
    Simulator::new(cfg).expect("default config")
}

fn pooled_lateral(rms: &[f64]) -> f64 {
    (rms.iter().map(|v| v * v).sum::<f64>() / rms.len() as f64).sqrt()
}

fn assert_close(got: f64, want: f64, tol: f64, what: &str) {
    assert!(
        (got - want).abs() <= tol * want.abs().max(1.0),
        "{what}: got {got}, want {want} (tol {tol})"
    );
}

fn healthy_frame(phase: Phase) -> Frame {
    Frame {
        seq: 0,
        time_us: 0,
        phase,
        gyro: [0.0; 3],
        accel: [0.0; 3],
        mag: [0.0; 3],
        height: 100.0,
        pitot_dp: 0,
        // AI_READY set, FAULT clear: isolates the phase gate from the health gate.
        flags: FLAG_AI_READY,
        crc: 0,
        gust: false,
        shear: false,
        saturated: false,
        spike: false,
        pitot_valid: true,
        physics: [0.0; 6],
    }
}

/// The command enters the state derivative as a moment on the angular axis
/// only; it is not a force on the translational axis. The applied moment is
/// recomputed here from the documented actuator formula, not from the solver.
#[test]
fn command_maps_to_the_documented_fin_moment_in_dynamics() {
    let cfg = Config::default();
    let sim = Simulator::new(cfg.clone()).expect("default config");
    let atm = sim.atmosphere();
    let mass = 11.0;
    let mut vehicle = Vehicle::from_config(&cfg, mass, 1.0, [0.0; 2]);
    let mut state = State::new(mass, 0.0, 0.0);
    state.pos[2] = 100.0;
    state.vel[2] = 60.0;
    let wind = [0.0; 3];

    vehicle.control_accel = [0.0, 0.0];
    let zero = derivative(3.0, &state, wind, atm, &vehicle);
    // Small enough that neither demand (nor its double) reaches the fin stop.
    vehicle.control_accel = [0.25, -0.5];
    let driven = derivative(3.0, &state, wind, atm, &vehicle);

    // Fin force is not part of body_force: only the angular derivative moves.
    assert_eq!(driven.vel, zero.vel, "command must not enter as a force");
    assert_eq!(driven.pos, zero.pos);
    assert_eq!(driven.mass, zero.mass);

    let air = atm.at(100.0);
    let qdyn = 0.5 * air.density * 60.0 * 60.0;
    let area = cfg.aero.area_m2;
    let diameter = 2.0 * (area / PI).sqrt();
    let arm = cfg.aero.control_moment_arm_m;
    let authority = qdyn * area * diameter * PLACEHOLDER_CM_DELTA;
    let i_xx = 0.1 * mass * 0.15 * 0.15;
    for (axis, g) in [(0usize, 0.25_f64), (1, -0.5_f64)] {
        let delta = solve_fin_deflection(g, mass, qdyn, area, PLACEHOLDER_CM_DELTA, arm)
            .expect("demand below the fin stop");
        assert!(delta.abs() < MAX_FIN_DEFLECTION_RAD);
        let expected_moment = authority * delta;
        let required_moment = mass * g * arm;
        assert_close(expected_moment, required_moment, 1e-12, "moment balance");
        let expected = expected_moment / i_xx;
        let got = driven.omega[axis] - zero.omega[axis];
        assert_close(got, expected, 1e-9, "commanded angular acceleration");
        assert_eq!(
            got.signum(),
            g.signum(),
            "axis {axis} sign must be preserved"
        );
        // Twice the demand must produce twice the angular acceleration.
        vehicle.control_accel = [0.0; 2];
        vehicle.control_accel[axis] = 2.0 * g;
        let doubled = derivative(3.0, &state, wind, atm, &vehicle);
        assert_close(
            doubled.omega[axis] - zero.omega[axis],
            2.0 * got,
            1e-9,
            "linearity",
        );
    }
}

/// All three zero-command entry points must be bit-identical to the legacy
/// path on every exported field, including negative zero (the gate uses
/// `!= 0.0`, not a sign-bit test).
#[test]
fn zero_command_is_bit_identical_to_legacy_on_every_exported_field() {
    for seed in [1u64, 7, 1337] {
        let sim = sim(seed);
        let legacy = sim.run_episode(11).expect("legacy episode");
        let variants = [
            ("frame", sim.run_episode_with_control(11, &|_| (0.0, 0.0))),
            (
                "negative-zero",
                sim.run_episode_with_control(11, &|_| (-0.0, -0.0)),
            ),
            (
                "context",
                sim.run_episode_with_control_ctx(11, &|_| (0.0, 0.0)),
            ),
        ];
        for (name, result) in variants {
            let ep = result.expect(name);
            assert_eq!(ep.frames, legacy.frames, "{seed}/{name}: frames moved");
            assert_eq!(
                ep.features, legacy.features,
                "{seed}/{name}: features moved"
            );
            assert_eq!(ep.targets, legacy.targets, "{seed}/{name}: targets moved");
            assert_eq!(ep.truth, legacy.truth, "{seed}/{name}: truth moved");
            assert_eq!(
                ep.trajectory, legacy.trajectory,
                "{seed}/{name}: flown truth moved"
            );
            assert_eq!(ep.apogee_m.to_bits(), legacy.apogee_m.to_bits());
            assert_eq!(ep.max_speed_mps.to_bits(), legacy.max_speed_mps.to_bits());
            assert_eq!(ep.frames_generated, legacy.frames_generated);
            assert_eq!(ep.phase_counts, legacy.phase_counts);
            assert_eq!(ep.invalid_frames, legacy.invalid_frames);
            assert_eq!(ep.nan_frames, legacy.nan_frames);
        }
    }
}

/// The command is m/s^2 of lateral acceleration, not fin degrees: the solver
/// multiplies by mass (acceleration -> force) and by the moment arm before
/// dividing by the aerodynamic authority.
#[test]
fn command_units_are_lateral_mps2_and_scale_with_mass() {
    assert_eq!(MAX_COMMAND_MPS2, 10.0, "~1 g lateral command envelope");
    assert_eq!(
        MAX_CONTROL_DEG, MAX_COMMAND_MPS2,
        "historical name is an alias"
    );
    assert!((MAX_FIN_DEFLECTION_RAD - 25.0 * PI / 180.0).abs() < 1e-15);
    // Ten m/s^2 is a larger number than the fin stop in radians: treating the
    // command as an angle would hit the stop on the first tick.
    const { assert!(MAX_COMMAND_MPS2 > 20.0 * MAX_FIN_DEFLECTION_RAD) };
    assert_eq!(
        validate_command(MAX_COMMAND_MPS2, -MAX_COMMAND_MPS2).unwrap(),
        (MAX_COMMAND_MPS2, -MAX_COMMAND_MPS2)
    );
    assert!(matches!(
        validate_command(MAX_COMMAND_MPS2 + 1e-9, 0.0),
        Err(SimError::Control(_))
    ));

    let cfg = Config::default();
    let area = cfg.aero.area_m2;
    let arm = cfg.aero.control_moment_arm_m;
    let mass = 11.0;
    let qdyn = 20_000.0;
    let diameter = 2.0 * (area / PI).sqrt();
    for g in [0.25_f64, 1.0, 3.0, 9.80665] {
        let delta = solve_fin_deflection(g, mass, qdyn, area, PLACEHOLDER_CM_DELTA, arm)
            .expect("units demand below the stop");
        let applied_moment = qdyn * area * diameter * PLACEHOLDER_CM_DELTA * delta;
        let required_moment = mass * g * arm;
        assert_close(applied_moment, required_moment, 1e-12, "M = m * a_lat * l");
        let heavier = solve_fin_deflection(g, 2.0 * mass, qdyn, area, PLACEHOLDER_CM_DELTA, arm)
            .expect("heavier vehicle demand");
        assert_close(
            heavier,
            2.0 * delta,
            1e-12,
            "deflection scales with mass at fixed a_lat",
        );
    }
}

/// The mechanical stop is exactly +/-25 degrees, saturation is bit-stable,
/// and every clamped demand is COUNTED rather than silently absorbed.
#[test]
fn fin_deflection_saturates_at_the_mechanical_stop_and_counts() {
    let cfg = Config::default();
    let area = cfg.aero.area_m2;
    let arm = cfg.aero.control_moment_arm_m;
    let mass = 11.0;
    let qdyn = 20_000.0;
    let solve = |g: f64| solve_fin_deflection(g, mass, qdyn, area, PLACEHOLDER_CM_DELTA, arm);

    let mut prev = 0.0_f64;
    for g in [0.0, 1.0, 5.0, 10.0, 50.0, 100.0, 1000.0] {
        let d = solve(g).expect("finite demand");
        assert!(d >= prev, "deflection must be monotone in demand");
        assert!(d <= MAX_FIN_DEFLECTION_RAD);
        prev = d;
    }
    let diameter = 2.0 * (area / PI).sqrt();
    let authority = qdyn * area * diameter * PLACEHOLDER_CM_DELTA;
    let just_inside = 0.99 * MAX_FIN_DEFLECTION_RAD * authority / (mass * arm);
    let inside = solve(just_inside).expect("inside the stop");
    assert!(
        inside > 0.0 && inside < MAX_FIN_DEFLECTION_RAD,
        "no clamp below the stop"
    );

    let before = fin_clamp_uses();
    let big = solve(1000.0).expect("positive stop");
    let bigger = solve(2000.0).expect("positive stop again");
    let neg = solve(-1000.0).expect("negative stop");
    assert_eq!(big.to_bits(), MAX_FIN_DEFLECTION_RAD.to_bits());
    assert_eq!(
        bigger.to_bits(),
        MAX_FIN_DEFLECTION_RAD.to_bits(),
        "saturation is bit-stable"
    );
    assert_eq!(neg.to_bits(), (-MAX_FIN_DEFLECTION_RAD).to_bits());
    assert!(fin_clamp_uses() >= before + 3, "clamps must be COUNTED");

    // Thin air drops a nonzero demand to a safe zero and counts the drop.
    let before = fin_no_authority_uses();
    let thin = solve_fin_deflection(
        2.0,
        mass,
        MIN_CONTROL_QDYN_PA,
        area,
        PLACEHOLDER_CM_DELTA,
        arm,
    )
    .expect("thin-air demand is a counted zero, not a fault");
    assert_eq!(thin, 0.0);
    assert!(
        fin_no_authority_uses() > before,
        "no-authority drop must be COUNTED"
    );
    // Unknown authority is a LOUD fault even for a small command.
    match solve_fin_deflection(0.5, mass, qdyn, area, 0.0, arm) {
        Err(ControlFault {
            reason: "cm_delta",
            value,
        }) => assert_eq!(value, 0.0),
        other => panic!("expected LOUD cm_delta fault, got {other:?}"),
    }
}

/// Pad and Boost forbid AI corrections independently of frame health; a
/// healthy frame must not re-enable the command on the powered segment.
#[test]
fn phase_and_health_interlocks_zero_the_command_independently() {
    let cfg = Config::default();
    let opt = ReferenceOptimizer::new(&cfg);
    let vehicle = Vehicle::from_config(&cfg, 11.0, 1.0, [0.0; 2]);
    let mut state = State::new(11.0, 0.0, 0.0);
    state.pos = [50.0, -40.0, 100.0];
    state.vel = [12.0, -9.0, 30.0];
    let wind = [0.0; 3];
    let mut frame = healthy_frame(Phase::Coast);

    let world = opt.world_accel_command(&state, wind, Phase::Coast, &frame);
    assert_close(world[0], -8.6, 1e-12, "world ax LQR demand");
    assert_close(world[1], 6.55, 1e-12, "world ay LQR demand");
    assert_eq!(
        opt.fin_accel_command(&state, wind, &vehicle, Phase::Coast, &frame),
        [world[1], -world[0]],
        "fin seam is [ay, -ax]"
    );
    for phase in [Phase::Coast, Phase::Apogee, Phase::Descent, Phase::Landed] {
        frame.phase = phase;
        assert!(
            opt.world_accel_command(&state, wind, phase, &frame)[0] != 0.0,
            "{phase:?} must allow AI corrections"
        );
    }
    for phase in [Phase::Pad, Phase::Boost] {
        frame.phase = phase;
        assert!(!phase.ai_allowed());
        assert_eq!(
            opt.world_accel_command(&state, wind, phase, &frame),
            [0.0, 0.0]
        );
        assert_eq!(
            opt.fin_accel_command(&state, wind, &vehicle, phase, &frame),
            [0.0, 0.0]
        );
    }

    // Health gate is independent of the phase gate.
    frame.phase = Phase::Coast;
    frame.flags = 0;
    assert!(!frame.ai_input_valid());
    assert_eq!(
        opt.world_accel_command(&state, wind, Phase::Coast, &frame),
        [0.0, 0.0]
    );
    frame.flags = FLAG_AI_READY | FLAG_FAULT;
    assert_eq!(
        opt.fin_accel_command(&state, wind, &vehicle, Phase::Coast, &frame),
        [0.0, 0.0]
    );
}

/// End-to-end: every powered tick exports a zero acceleration command, and
/// coast ticks carry a live one.
#[test]
fn powered_ticks_export_zero_accel_command() {
    let cfg = Config::default();
    let sim = Simulator::new(cfg.clone()).expect("default config");
    let t_burn = Vehicle::from_config(&cfg, 11.0, 1.0, [0.0; 2]).burn_end();
    assert!(t_burn > 0.0);
    let ep = sim.run_episode_with_oracle(3).expect("oracle episode");
    let mut powered = 0;
    let mut coast_live = 0;
    for (i, tick) in ep.trajectory.iter().enumerate() {
        let t = i as f64 / cfg.run.frame_rate_hz;
        if t <= t_burn {
            assert_eq!(tick.g_accel, [0.0, 0.0], "powered tick {i} at t={t}");
            powered += 1;
        } else if tick.g_accel != [0.0, 0.0] {
            coast_live += 1;
        }
    }
    assert!(powered > 0, "episode must contain powered ticks");
    assert!(coast_live > 0, "coast ticks must carry a nonzero command");
}

/// The physical oracle reduces lateral drift; flipping its sign makes the
/// same actuator actively worse on every held-out episode and every seed.
#[test]
fn wrong_sign_command_worsens_held_out_lateral_rms() {
    let mut report = String::new();
    for seed in [1u64, 2, 3] {
        let sim = sim(seed);
        let collect = |mode: u8| {
            pooled_lateral(
                &HELD_OUT
                    .iter()
                    .map(|&i| {
                        let ep = match mode {
                            0 => sim.run_episode(i),
                            1 => sim.run_episode_with_oracle(i),
                            _ => sim.run_episode_with_control_ctx(i, &|ctx| {
                                let g = ReferenceOptimizer::new(sim.config()).fin_accel_command(
                                    ctx.state,
                                    ctx.wind,
                                    ctx.vehicle,
                                    ctx.phase,
                                    ctx.frame,
                                );
                                (-g[0], -g[1])
                            }),
                        }
                        .expect("episode");
                        lateral_rms_m(&ep.trajectory)
                    })
                    .collect::<Vec<_>>(),
            )
        };
        let base = collect(0);
        let oracle = collect(1);
        let wrong = collect(2);
        report.push_str(&format!(
            "seed {seed}: wrong {wrong:.6} > base {base:.6} > oracle {oracle:.6} m\n"
        ));
        assert!(oracle < base, "oracle must improve lateral RMS:\n{report}");
        assert!(
            wrong > base,
            "flipping the command sign must worsen lateral RMS:\n{report}"
        );
    }
    eprintln!("\n{report}");
}
