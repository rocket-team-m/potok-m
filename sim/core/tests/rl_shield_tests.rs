// SPDX-License-Identifier: GPL-3.0-or-later
//! Hard tests for the typed online shield in the RL/env training path.
//!
//! `rl_env.rs` is compiled straight from its source file (same seam as
//! `rl_env_tests.rs`) so no file outside the deliverables is touched.

pub use potok_sim::{config, control, dynamics, episode, error, math, physics, reward, sensor};

#[path = "../src/rl_env.rs"]
mod rl_env;

use config::Config;
use dynamics::Phase;
use reward::{FlightSample, GateLimits, GateObservation, RewardWeights, TruthSidecar};
use rl_env::{
    Action, RlEnv, Shaping, Shield, ShieldCounts, ShieldDecision, ShieldReason, ShieldSample,
};

const WEIGHTS: RewardWeights = RewardWeights {
    lateral: 2.0,
    apogee: 0.5,
    effort: 3.0,
    risk: 7.0,
};

fn short_cfg() -> Config {
    let mut cfg = Config::default();
    cfg.run.duration_s = 2.0;
    cfg.validate().unwrap();
    cfg
}

fn limits() -> GateLimits {
    GateLimits {
        mach_max: 0.6,
        safety_factor_min: 0.0,
        static_margin_cal_min: 0.0,
    }
}

fn permissive_truth(cfg: &Config) -> TruthSidecar {
    let dt = cfg.steps_per_frame() as f64 * cfg.run.dt;
    let reference = (0..cfg.frames_per_episode())
        .map(|i| FlightSample {
            t_s: i as f64 * dt,
            pos_m: [10.0, 0.0, 100.0],
            ctrl_mps2: 0.0,
            safety_violation: false,
        })
        .collect();
    TruthSidecar {
        reference,
        limits: limits(),
        observed: GateObservation {
            mach_peak: 0.5,
            safety_factor_min: 5.0,
            static_margin_cal_min: 1.0,
        },
    }
}

fn env(cfg: Config, truth: TruthSidecar) -> RlEnv {
    RlEnv::new(cfg, 0, truth, WEIGHTS, Shaping::default()).unwrap()
}

fn sample(mach: f64, phase: Phase) -> ShieldSample {
    ShieldSample { mach, phase }
}

#[test]
fn shield_limits_come_from_the_verifier_not_a_new_gate() {
    let shield = Shield::new(limits()).unwrap();
    assert_eq!(shield.mach_max(), 0.6);
    assert_eq!(shield.authority(), control::MAX_COMMAND_MPS2 as f32);
    assert_eq!(shield.counts(), ShieldCounts::default());

    // A malformed ceiling is loud, never defaulted.
    assert!(Shield::new(GateLimits {
        mach_max: f64::NAN,
        ..limits()
    })
    .is_err());
    assert!(Shield::new(GateLimits {
        mach_max: -1.0,
        ..limits()
    })
    .is_err());
}

#[test]
fn accepted_demand_passes_through_bitwise() {
    let mut shield = Shield::new(limits()).unwrap();
    let candidate = Action { ay: 2.5, ax: -1.25 };
    let d = shield.verify(candidate, sample(0.5, Phase::Coast));
    assert_eq!(
        d,
        ShieldDecision {
            action: candidate,
            reason: None
        }
    );
    assert!(!d.intervened());
    assert_eq!(d.action.ay.to_bits(), candidate.ay.to_bits());
    assert_eq!(shield.counts().accepted, 1);
    assert_eq!(shield.counts().interventions(), 0);
}

#[test]
fn each_constraint_has_a_typed_reason_a_zero_fallback_and_a_counter() {
    let mut shield = Shield::new(limits()).unwrap();
    let candidate = Action { ay: 4.0, ax: 3.0 };

    let mach = shield.verify(candidate, sample(0.9, Phase::Coast));
    assert_eq!(
        mach.reason,
        Some(ShieldReason::MachLimit {
            observed: 0.9,
            limit: 0.6
        })
    );
    assert_eq!(mach.action, Action::ZERO);

    let phase = shield.verify(candidate, sample(0.1, Phase::Boost));
    assert_eq!(
        phase.reason,
        Some(ShieldReason::PhaseForbidden {
            phase: Phase::Boost
        })
    );
    assert_eq!(phase.action, Action::ZERO);
    // AI is forbidden on the pad too.
    assert!(shield
        .verify(candidate, sample(0.1, Phase::Pad))
        .intervened());

    let authority = shield.verify(Action { ay: 11.0, ax: 0.0 }, sample(0.1, Phase::Descent));
    assert!(matches!(
        authority.reason,
        Some(ShieldReason::Authority { .. })
    ));
    assert_eq!(authority.action, Action::ZERO);
    // Never a clipped 10.0: the bad demand is refused, not trimmed.
    assert_ne!(authority.action.ay, 10.0);

    let unavailable = shield.verify(candidate, sample(f64::NAN, Phase::Coast));
    assert!(matches!(
        unavailable.reason,
        Some(ShieldReason::MachUnavailable { .. })
    ));
    assert_eq!(unavailable.action, Action::ZERO);

    let counts = shield.counts();
    assert_eq!(counts.mach_limit, 1);
    assert_eq!(counts.phase_forbidden, 2);
    assert_eq!(counts.authority, 1);
    assert_eq!(counts.mach_unavailable, 1);
    assert_eq!(counts.accepted, 0);
    assert_eq!(counts.interventions(), 5);
    assert_eq!(counts.total(), 5);
}

#[test]
fn mach_priority_is_fixed_and_the_reason_names_both_values() {
    let mut shield = Shield::new(limits()).unwrap();
    // Every constraint violated at once: the first check wins deterministically.
    let d = shield.verify(
        Action {
            ay: f32::NAN,
            ax: 0.0,
        },
        sample(f64::INFINITY, Phase::Boost),
    );
    assert_eq!(
        d.reason,
        Some(ShieldReason::MachUnavailable {
            observed: f64::INFINITY
        })
    );
    let text = d.reason.unwrap().to_string();
    assert!(text.contains("shield:"), "{text}");
}

#[test]
fn non_finite_demand_is_an_authority_veto_not_a_panic() {
    let mut shield = Shield::new(limits()).unwrap();
    for bad in [
        Action {
            ay: f32::NAN,
            ax: 0.0,
        },
        Action {
            ay: 0.0,
            ax: f32::INFINITY,
        },
    ] {
        let d = shield.verify(bad, sample(0.1, Phase::Apogee));
        assert!(matches!(d.reason, Some(ShieldReason::Authority { .. })));
        assert_eq!(d.action, Action::ZERO);
    }
    assert_eq!(shield.counts().authority, 2);
}

#[test]
fn reset_counts_is_explicit_and_complete() {
    let mut shield = Shield::new(limits()).unwrap();
    shield.verify(Action::ZERO, sample(0.5, Phase::Coast));
    shield.verify(Action::ZERO, sample(0.9, Phase::Coast));
    assert_eq!(shield.counts().total(), 2);
    shield.reset_counts();
    assert_eq!(shield.counts(), ShieldCounts::default());
}

#[test]
fn shielded_step_sends_the_fallback_command_to_the_simulator() {
    let cfg = short_cfg();

    // Over-Mach: the shield vetoes a nonzero candidate and drives the sim with
    // ZERO. An unshielded ZERO-driven env must produce the identical flight.
    let mut guarded = env(cfg.clone(), permissive_truth(&cfg));
    let mut plain = env(cfg.clone(), permissive_truth(&cfg));
    let (o0, i0) = guarded.reset().unwrap();
    let (p0, j0) = plain.reset().unwrap();
    assert_eq!(o0.tick, p0.tick);

    let candidate = Action { ay: 6.0, ax: -4.0 };
    let s = guarded.step_shielded(candidate, 0.99).unwrap();
    let t = plain.step(Action::ZERO).unwrap();

    let decision = guarded.last_shield_decision().unwrap();
    assert_eq!(
        decision.reason,
        Some(ShieldReason::MachLimit {
            observed: 0.99,
            limit: 0.6
        })
    );
    assert_eq!(decision.action, Action::ZERO);
    assert_eq!(guarded.shield_counts().mach_limit, 1);
    assert_eq!(s.obs.tick, t.obs.tick);
    assert_eq!(s.reward.to_bits(), t.reward.to_bits());
    assert_eq!(s.info.truth.pos_m, t.info.truth.pos_m);
    assert_eq!(i0.truth.pos_m, j0.truth.pos_m);
    assert!(!s.done);
}

#[test]
fn shield_counts_accumulate_across_resets_and_last_decision_clears() {
    let cfg = short_cfg();
    let mut e = env(cfg.clone(), permissive_truth(&cfg));
    e.reset().unwrap();
    e.step_shielded(Action::ZERO, 0.99).unwrap();
    assert_eq!(e.shield_counts().mach_limit, 1);

    e.reset().unwrap();
    assert_eq!(e.last_shield_decision(), None);
    assert_eq!(e.shield_counts().mach_limit, 1);
    e.step_shielded(Action::ZERO, 0.99).unwrap();
    assert_eq!(e.shield_counts().mach_limit, 2);
}

#[test]
fn raw_step_is_unchanged_for_bisect_and_legacy_callers() {
    let cfg = short_cfg();
    let mut e = env(cfg.clone(), permissive_truth(&cfg));
    e.reset().unwrap();
    // The unshielded seam still refuses an out-of-box demand loudly.
    assert!(e.step(Action { ay: 11.0, ax: 0.0 }).is_err());
    assert_eq!(e.shield_counts(), ShieldCounts::default());
    assert_eq!(e.last_shield_decision(), None);
}
