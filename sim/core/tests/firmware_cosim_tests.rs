// SPDX-License-Identifier: GPL-3.0-or-later
//! Hard gates on the firmware co-simulation seam.
//!
//! The seam is a host port of the firmware/AI guard decisions in
//! `cli/src/firmware_cosim.rs`, read from the read-only firmware repo. These
//! tests pin three things: the ported constants and decision order, the
//! bit-identical zero-command path, and that an accepted command is the
//! oracle's command unchanged (same positive control, same A/B sign).

#[path = "../../cli/src/firmware_cosim.rs"]
mod firmware_cosim;

use firmware_cosim::{
    run_guarded, run_guarded_with, Command, Decision, FirmwareGuard, Veto, MAX_CONSECUTIVE_MISSES,
    MAX_CORRECTION_RAW, MAX_DELTA, REPLY_SCALE, SIM_TICK_HZ,
};
use potok_sim::{
    lateral_rms_m, sensor::FLAG_AI_READY, sensor::FLAG_FAULT, sensor::FLAG_MOTOR_ARMED, Config,
    Episode, Frame, Phase, Simulator, MAX_COMMAND_MPS2,
};

const HELD_OUT: [u64; 4] = [16, 17, 18, 19];

fn sim(seed: u64) -> Simulator {
    let mut cfg = Config::default();
    cfg.run.seed = seed;
    Simulator::new(cfg).expect("default config")
}

fn short_sim(seed: u64, duration_s: f64) -> Simulator {
    let mut cfg = Config::default();
    cfg.run.seed = seed;
    cfg.run.duration_s = duration_s;
    Simulator::new(cfg).expect("short config")
}

fn frame(phase: Phase, flags: u8) -> Frame {
    Frame {
        seq: 0,
        time_us: 0,
        phase,
        gyro: [0.0; 3],
        accel: [0.0; 3],
        mag: [0.0; 3],
        height: 100.0,
        pitot_dp: 0,
        flags,
        crc: 0,
        gust: false,
        shear: false,
        saturated: false,
        spike: false,
        pitot_valid: true,
        physics: [0.0; 6],
    }
}

fn assert_same_episode(got: &Episode, want: &Episode, what: &str) {
    assert_eq!(got.frames, want.frames, "{what}: frames moved");
    assert_eq!(got.features, want.features, "{what}: features moved");
    assert_eq!(got.targets, want.targets, "{what}: targets moved");
    assert_eq!(got.truth, want.truth, "{what}: truth moved");
    assert_eq!(got.trajectory, want.trajectory, "{what}: flown truth moved");
    assert_eq!(got.apogee_m.to_bits(), want.apogee_m.to_bits(), "{what}");
    assert_eq!(
        got.max_speed_mps.to_bits(),
        want.max_speed_mps.to_bits(),
        "{what}"
    );
    assert_eq!(got.frames_generated, want.frames_generated, "{what}");
    assert_eq!(got.phase_counts, want.phase_counts, "{what}");
    assert_eq!(got.invalid_frames, want.invalid_frames, "{what}");
    assert_eq!(got.nan_frames, want.nan_frames, "{what}");
}

fn pooled_lateral(rms: &[f64]) -> f64 {
    (rms.iter().map(|v| v * v).sum::<f64>() / rms.len() as f64).sqrt()
}

/// The ported numbers are the published ones: no constant is fitted to an arm.
#[test]
fn seam_constants_match_the_read_only_sources() {
    assert_eq!(SIM_TICK_HZ, 50);
    assert_eq!(Config::default().run.frame_rate_hz, SIM_TICK_HZ as f64);
    assert_eq!(MAX_CONSECUTIVE_MISSES, 3);
    assert_eq!(MAX_CORRECTION_RAW, 100);
    assert_eq!(REPLY_SCALE, 0.001);
    assert_eq!(MAX_COMMAND_MPS2, 10.0);
    assert_eq!(MAX_DELTA, 2.0 * MAX_COMMAND_MPS2);
    assert_eq!((FLAG_AI_READY, FLAG_MOTOR_ARMED, FLAG_FAULT), (16, 64, 128));
    assert_eq!(
        (
            Phase::Boost.protocol_code(),
            Phase::Coast.protocol_code(),
            Phase::Apogee.protocol_code(),
            Phase::Descent.protocol_code(),
            Phase::Landed.protocol_code()
        ),
        (0, 1, 2, 3, 4)
    );
    assert!(Veto::Fault.latches() && Veto::AiUnhealthy.latches());
    assert!(Veto::Implausible.latches() && Veto::UnknownPhase(9).latches());
    assert!(!Veto::Boost.latches() && !Veto::MotorArmed.latches());
}

/// Decision order and latch classes are the firmware's, and an accepted
/// command is the proposed one bit-for-bit (no scaling, no shaping, no tuning).
#[test]
fn guard_decision_order_latch_and_pass_through() {
    let healthy = frame(Phase::Coast, FLAG_AI_READY);
    let proposed = Command {
        roll: 2.5,
        pitch: -1.0,
    };

    let mut g = FirmwareGuard::new();
    // Boost and MOTOR_ARMED are transient: no latch, no miss.
    assert_eq!(
        g.step(&frame(Phase::Boost, FLAG_AI_READY), proposed),
        Decision {
            command: Command::ZERO,
            veto: Some(Veto::Boost),
            misses: 0,
            latched: false
        }
    );
    assert_eq!(
        g.step(
            &frame(Phase::Coast, FLAG_AI_READY | FLAG_MOTOR_ARMED),
            proposed
        )
        .veto,
        Some(Veto::MotorArmed)
    );
    assert_eq!((g.misses(), g.latched()), (0, None));
    assert!(g.last_command().is_zero());

    // Health and board faults latch immediately and survive their cause.
    let mut g = FirmwareGuard::new();
    assert_eq!(
        g.step_with(&healthy, proposed, true, true).veto,
        Some(Veto::Fault)
    );
    assert_eq!(g.last_veto(), Some(Veto::Fault));
    assert_eq!(g.latched(), Some(Veto::Fault));
    assert_eq!(
        g.step(&healthy, proposed),
        Decision {
            command: Command::ZERO,
            veto: Some(Veto::Fault),
            misses: 0,
            latched: true
        }
    );
    g.reset();
    assert_eq!(g.resets(), 1);
    let accepted = g.step(&healthy, proposed);
    assert_eq!(accepted.command, proposed);
    assert_eq!(accepted.command.roll.to_bits(), proposed.roll.to_bits());
    assert_eq!(accepted.command.pitch.to_bits(), proposed.pitch.to_bits());
    assert_eq!(accepted.veto, None);

    // An unhealthy reply latches the same way; an invalid sensor sample does
    // not -- it vetoes only the tick on which it was measured.
    let mut g = FirmwareGuard::new();
    assert_eq!(
        g.step_with(&healthy, proposed, false, false).veto,
        Some(Veto::AiUnhealthy)
    );
    assert_eq!(g.latched(), Some(Veto::AiUnhealthy));
    let mut g = FirmwareGuard::new();
    assert_eq!(
        g.step(&frame(Phase::Coast, 0), proposed).veto,
        Some(Veto::SensorInvalid)
    );
    assert_eq!((g.misses(), g.latched()), (0, None));
    assert_eq!(g.step(&healthy, proposed).command, proposed);

    // Three non-finite misses latch; rejected samples never move the rate ref.
    let mut g = FirmwareGuard::new();
    let nan = Command {
        roll: f64::NAN,
        pitch: 0.0,
    };
    for misses in 1..=3 {
        let d = g.step(&healthy, nan);
        assert_eq!(d.veto, Some(Veto::NonFinite));
        assert_eq!((d.misses, d.latched), (misses, misses == 3));
    }
    assert_eq!(g.step(&healthy, proposed).veto, Some(Veto::Latched));
    g.reset();
    assert_eq!(g.step(&healthy, proposed).command, proposed);

    // Out-of-envelope is an immediate plausibility trip, not a silent clamp.
    let mut g = FirmwareGuard::new();
    let big = Command {
        roll: MAX_COMMAND_MPS2 + 1e-9,
        pitch: 0.0,
    };
    assert_eq!(g.step(&healthy, big).veto, Some(Veto::Implausible));
    assert_eq!(g.latched(), Some(Veto::Implausible));
}

/// Every tick of a powered-only episode is vetoed by the BURN/motor gate, so
/// the command reaching the dynamics is exactly zero and the whole exported
/// episode is bit-identical to the controller-absent legacy path.
#[test]
fn zero_command_from_the_seam_is_bit_identical_to_legacy() {
    for seed in [1u64, 7, 1337] {
        let sim = short_sim(seed, 1.0);
        let legacy = sim.run_episode(11).expect("legacy powered episode");
        let run = run_guarded(&sim, 11).expect("guarded powered episode");
        assert_same_episode(&run.episode, &legacy, &format!("seed {seed}"));
        assert_eq!(run.guard.accepted(), 0);
        assert_eq!(run.guard.vetoed() as usize, legacy.frames_generated);
        assert_eq!(run.guard.latched(), None);
        assert!(run.guard.last_command().is_zero());
    }
}

/// The guard is sampled at the simulator's 50 Hz frame tick, exactly once per
/// generated frame, and every tick lands in exactly one decision bucket.
#[test]
fn seam_decides_once_per_50hz_frame() {
    let sim = sim(1);
    let run = run_guarded(&sim, 0).expect("default episode");
    assert_eq!(run.guard.ticks() as usize, run.episode.frames_generated);
    assert_eq!(run.episode.frames.len(), run.episode.frames_generated);
    assert_eq!(run.guard.accepted() + run.guard.vetoed(), run.guard.ticks());
    eprintln!(
        "50 Hz seam: {} frames, {} accepted, {} vetoed, latch={:?}",
        run.guard.ticks(),
        run.guard.accepted(),
        run.guard.vetoed(),
        run.guard.latched()
    );
}

/// Real frames (not hand-built fixtures) drive the decision chain, and one
/// injected board fault latches every later command to zero until reset.
#[test]
fn real_sim_frames_latch_on_a_single_fault_and_recover_only_on_reset() {
    let episode = sim(1).run_episode(0).expect("episode 0");
    let proposed = Command {
        roll: 1.0,
        pitch: -0.5,
    };
    let fault_tick = episode
        .frames
        .iter()
        .enumerate()
        .filter(|(_, f)| f.phase == Phase::Coast && f.ai_input_valid())
        .nth(1)
        .map(|(i, _)| i)
        .expect("a second live coast tick to inject the fault on");
    let mut guard = FirmwareGuard::new();
    let mut accepted_before = 0;
    for (i, frame) in episode.frames.iter().enumerate() {
        let d = guard.step_with(frame, proposed, true, i == fault_tick);
        if i == fault_tick {
            assert_eq!(d.veto, Some(Veto::Fault));
        }
        if i < fault_tick && d.veto.is_none() {
            accepted_before += 1;
        }
        if i > fault_tick {
            assert!(d.command.is_zero(), "latched fault must zero tick {i}");
        }
    }
    assert!(accepted_before > 0, "the pre-fault window must be live");
    assert_eq!(guard.latched(), Some(Veto::Fault));
    let healthy = episode
        .frames
        .iter()
        .find(|f| f.phase == Phase::Coast && f.ai_input_valid())
        .expect("a healthy coast frame");
    guard.reset();
    assert_eq!(guard.step(healthy, proposed).command, proposed);
}

/// Positive control survives the seam: the accepted command is the oracle's,
/// so the flown lateral drift improves and the sign-flipped arm is worse on
/// the same held-out episodes, seeds and guard path.
#[test]
fn oracle_positive_control_is_preserved_through_the_seam() {
    let mut report = String::new();
    for seed in [1u64, 2] {
        let sim = sim(seed);
        let collect = |mode: u8| {
            pooled_lateral(
                &HELD_OUT
                    .iter()
                    .map(|&i| {
                        let ep = match mode {
                            0 => sim.run_episode(i),
                            1 => run_guarded(&sim, i).map(|r| r.episode),
                            _ => run_guarded_with(&sim, i, |c| Command {
                                roll: -c.roll,
                                pitch: -c.pitch,
                            })
                            .map(|r| r.episode),
                        }
                        .expect("episode");
                        lateral_rms_m(&ep.trajectory)
                    })
                    .collect::<Vec<_>>(),
            )
        };
        let base = collect(0);
        let guarded = collect(1);
        let wrong = collect(2);
        report.push_str(&format!(
            "seed {seed}: wrong {wrong:.6} > base {base:.6} > guarded {guarded:.6} m\n"
        ));
        assert!(
            guarded < base,
            "guarded must keep the oracle gain:\n{report}"
        );
        assert!(
            wrong > base,
            "flipped sign must worsen lateral RMS:\n{report}"
        );
    }
    eprintln!("\n{report}");
}
