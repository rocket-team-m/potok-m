// SPDX-License-Identifier: GPL-3.0-or-later
//! Hard tests for the verifiable RLVR reward.
//!
//! Synthetic trajectories only: no real flight data, no dataset fixture, no
//! calibrated number. The weights below are arbitrary distinct positives
//! chosen so each term's sign and scale are visible in the total.
//! `reward.rs` is compiled straight from its single source file so that no
//! file other than the two deliverables is touched.

#[path = "../src/reward.rs"]
mod reward;

use reward::{
    dense_reward, evaluate, potential, FlightSample, GateLimits, GateObservation, GateViolation,
    RewardError, RewardWeights, ShapingWeights, TruthSidecar,
};

const WEIGHTS: RewardWeights = RewardWeights {
    lateral: 2.0,
    apogee: 0.5,
    effort: 3.0,
    risk: 7.0,
};

const SHAPING: ShapingWeights = ShapingWeights {
    lateral: 2.0,
    apogee: 0.5,
    gamma: 0.9,
};

fn tick(t: f64, x: f64, z: f64, u: f64, violation: bool) -> FlightSample {
    FlightSample {
        t_s: t,
        pos_m: [x, 0.0, z],
        ctrl_mps2: u,
        safety_violation: violation,
    }
}

fn flat(n: usize, x: f64, z: f64) -> Vec<FlightSample> {
    (0..n).map(|i| tick(i as f64, x, z, 0.0, false)).collect()
}

fn sidecar(reference: Vec<FlightSample>) -> TruthSidecar {
    TruthSidecar {
        reference,
        limits: GateLimits {
            mach_max: 1.0,
            safety_factor_min: 1.5,
            static_margin_cal_min: 1.0,
        },
        observed: GateObservation {
            mach_peak: 0.8,
            safety_factor_min: 2.0,
            static_margin_cal_min: 1.5,
        },
    }
}

#[test]
fn lateral_improvement_has_the_right_sign_in_both_directions() {
    let truth = sidecar(flat(5, 10.0, 100.0));
    let better = evaluate(&flat(5, 2.0, 100.0), Some(&truth), WEIGHTS).unwrap();
    let same = evaluate(&flat(5, 10.0, 100.0), Some(&truth), WEIGHTS).unwrap();
    let worse = evaluate(&flat(5, 20.0, 100.0), Some(&truth), WEIGHTS).unwrap();

    assert!((better.lateral_improvement_m - 8.0).abs() < 1e-12);
    assert!((worse.lateral_improvement_m + 10.0).abs() < 1e-12);
    assert_eq!(same.lateral_improvement_m, 0.0);
    assert!(better.total > same.total && same.total > worse.total);
    assert_eq!(better.apogee_gain_m, 0.0);
    assert_eq!(better.control_integral, 0.0);
    assert_eq!(better.safety_violations, 0);
}

#[test]
fn apogee_gain_has_the_right_sign_in_both_directions() {
    let truth = sidecar(flat(5, 0.0, 100.0));
    let better = evaluate(&flat(5, 0.0, 150.0), Some(&truth), WEIGHTS).unwrap();
    let same = evaluate(&flat(5, 0.0, 100.0), Some(&truth), WEIGHTS).unwrap();
    let worse = evaluate(&flat(5, 0.0, 80.0), Some(&truth), WEIGHTS).unwrap();

    assert_eq!(better.apogee_gain_m, 50.0);
    assert_eq!(worse.apogee_gain_m, -20.0);
    assert!(better.total > same.total && same.total > worse.total);
    assert_eq!(better.lateral_improvement_m, 0.0);
}

#[test]
fn control_integral_is_a_trapezoid_and_reduces_reward() {
    let truth = sidecar(flat(3, 0.0, 0.0));
    let idle = vec![
        tick(0.0, 0.0, 0.0, 0.0, false),
        tick(1.0, 0.0, 0.0, 0.0, false),
        tick(2.0, 0.0, 0.0, 0.0, false),
    ];
    let busy = vec![
        tick(0.0, 0.0, 0.0, 2.0, false),
        tick(1.0, 0.0, 0.0, 2.0, false),
        tick(2.0, 0.0, 0.0, 2.0, false),
    ];

    let a = evaluate(&idle, Some(&truth), WEIGHTS).unwrap();
    let b = evaluate(&busy, Some(&truth), WEIGHTS).unwrap();
    assert_eq!(a.control_integral, 0.0);
    assert_eq!(b.control_integral, 4.0);
    assert_eq!(b.effort_term, -12.0);
    assert!(b.total < a.total);

    let half = evaluate(
        &busy,
        Some(&truth),
        RewardWeights {
            effort: WEIGHTS.effort / 2.0,
            ..WEIGHTS
        },
    )
    .unwrap();
    assert_eq!(half.effort_term, -6.0);
    assert!(half.total > b.total);
}

#[test]
fn safety_violations_are_counted_per_tick_and_penalised() {
    let truth = sidecar(flat(3, 0.0, 0.0));
    let flown = vec![
        tick(0.0, 0.0, 0.0, 0.0, true),
        tick(1.0, 0.0, 0.0, 0.0, false),
        tick(2.0, 0.0, 0.0, 0.0, true),
    ];
    let scored = evaluate(&flown, Some(&truth), WEIGHTS).unwrap();
    let unweighted = evaluate(
        &flown,
        Some(&truth),
        RewardWeights {
            risk: 0.0,
            ..WEIGHTS
        },
    )
    .unwrap();

    assert_eq!(scored.safety_violations, 2);
    assert_eq!(scored.risk_term, -14.0);
    assert_eq!(unweighted.risk_term, 0.0);
    assert!(scored.total < unweighted.total);
}

#[test]
fn breakdown_recomputes_and_tampering_is_detected() {
    let truth = sidecar(flat(4, 3.0, 80.0));
    let flown = vec![
        tick(0.0, 1.0, 60.0, 1.0, false),
        tick(1.0, 2.0, 90.0, 2.0, true),
        tick(2.0, 0.0, 110.0, 3.0, false),
    ];
    let r = evaluate(&flown, Some(&truth), WEIGHTS).unwrap();
    let recomputed = WEIGHTS.lateral * r.lateral_improvement_m + WEIGHTS.apogee * r.apogee_gain_m
        - WEIGHTS.effort * r.control_integral
        - WEIGHTS.risk * f64::from(r.safety_violations);
    assert_eq!(recomputed.to_bits(), r.total.to_bits());
    assert!(r.check(WEIGHTS));
    assert!(!r.check(RewardWeights {
        lateral: 0.0,
        ..WEIGHTS
    }));

    let mut altered = r;
    altered.total += 1.0;
    assert!(!altered.check(WEIGHTS));
}

#[test]
fn mach_gate_rejects_instead_of_penalising() {
    let mut truth = sidecar(flat(3, 0.0, 0.0));
    truth.observed.mach_peak = 1.2;
    let expected = Err(RewardError::GateRejected(GateViolation::Mach {
        observed: 1.2,
        limit: 1.0,
    }));
    let flown = flat(3, 0.0, 0.0);
    assert_eq!(evaluate(&flown, Some(&truth), WEIGHTS), expected);
    assert_eq!(
        evaluate(
            &flown,
            Some(&truth),
            RewardWeights {
                risk: 1e6,
                ..WEIGHTS
            }
        ),
        expected
    );
    assert_eq!(
        evaluate(
            &flown,
            Some(&truth),
            RewardWeights {
                risk: 0.0,
                ..WEIGHTS
            }
        ),
        expected
    );
}

#[test]
fn safety_factor_gate_rejects_instead_of_penalising() {
    let mut truth = sidecar(flat(3, 0.0, 0.0));
    truth.observed.safety_factor_min = 1.2;
    let expected = Err(RewardError::GateRejected(GateViolation::SafetyFactor {
        observed: 1.2,
        limit: 1.5,
    }));
    assert_eq!(
        evaluate(&flat(3, 0.0, 0.0), Some(&truth), WEIGHTS),
        expected
    );
    assert_eq!(
        evaluate(
            &flat(3, 0.0, 0.0),
            Some(&truth),
            RewardWeights {
                risk: 1e6,
                ..WEIGHTS
            }
        ),
        expected
    );
}

#[test]
fn caliber_margin_gate_rejects_instead_of_penalising() {
    let mut truth = sidecar(flat(3, 0.0, 0.0));
    truth.observed.static_margin_cal_min = 0.7;
    let expected = Err(RewardError::GateRejected(GateViolation::StaticMarginCal {
        observed: 0.7,
        limit: 1.0,
    }));
    assert_eq!(
        evaluate(&flat(3, 0.0, 0.0), Some(&truth), WEIGHTS),
        expected
    );
    assert_eq!(
        evaluate(
            &flat(3, 0.0, 0.0),
            Some(&truth),
            RewardWeights {
                risk: 1e6,
                ..WEIGHTS
            }
        ),
        expected
    );
}

#[test]
fn gate_limits_are_inclusive_at_equality() {
    let mut truth = sidecar(flat(3, 0.0, 0.0));
    truth.observed = GateObservation {
        mach_peak: 1.0,
        safety_factor_min: 1.5,
        static_margin_cal_min: 1.0,
    };
    assert!(evaluate(&flat(3, 0.0, 0.0), Some(&truth), WEIGHTS).is_ok());
}

#[test]
fn missing_truth_is_loud() {
    assert_eq!(
        evaluate(&flat(3, 0.0, 0.0), None, WEIGHTS),
        Err(RewardError::MissingTruth)
    );
}

#[test]
fn malformed_inputs_are_loud() {
    let truth = sidecar(flat(3, 0.0, 0.0));
    let flown = flat(3, 0.0, 0.0);

    assert_eq!(
        evaluate(&[], Some(&truth), WEIGHTS),
        Err(RewardError::EmptyTrajectory)
    );
    assert_eq!(
        evaluate(&flown, Some(&sidecar(vec![])), WEIGHTS),
        Err(RewardError::MissingReference)
    );

    let mut non_monotonic = flown.clone();
    non_monotonic[1].t_s = 0.0;
    assert_eq!(
        evaluate(&non_monotonic, Some(&truth), WEIGHTS),
        Err(RewardError::NonMonotonicTime { index: 1 })
    );

    let mut bad_pos = flown.clone();
    bad_pos[1].pos_m[0] = f64::NAN;
    assert_eq!(
        evaluate(&bad_pos, Some(&truth), WEIGHTS),
        Err(RewardError::NonFinite("position"))
    );

    let mut neg_control = flown.clone();
    neg_control[0].ctrl_mps2 = -1.0;
    assert_eq!(
        evaluate(&neg_control, Some(&truth), WEIGHTS),
        Err(RewardError::NegativeControl { index: 0 })
    );

    let mut nan_observed = truth.clone();
    nan_observed.observed.mach_peak = f64::NAN;
    assert_eq!(
        evaluate(&flown, Some(&nan_observed), WEIGHTS),
        Err(RewardError::NonFinite("observed mach"))
    );

    assert_eq!(
        evaluate(
            &flown,
            Some(&truth),
            RewardWeights {
                effort: -1.0,
                ..WEIGHTS
            }
        ),
        Err(RewardError::InvalidWeights("effort"))
    );
    assert_eq!(
        evaluate(
            &flown,
            Some(&truth),
            RewardWeights {
                lateral: f64::NAN,
                ..WEIGHTS
            }
        ),
        Err(RewardError::InvalidWeights("lateral"))
    );
}

#[test]
fn same_inputs_give_bitwise_identical_rewards() {
    let truth = sidecar(flat(3, 3.0, 100.0));
    let flown = vec![
        tick(0.0, 1.0, 90.0, 0.5, false),
        tick(0.5, 2.0, 110.0, 1.5, true),
        tick(1.0, 0.0, 120.0, 0.0, false),
    ];
    let a = evaluate(&flown, Some(&truth), WEIGHTS).unwrap();
    let b = evaluate(&flown, Some(&truth), WEIGHTS).unwrap();
    assert_eq!(a, b);
    assert_eq!(a.total.to_bits(), b.total.to_bits());
    assert!(a.check(WEIGHTS));
}

#[test]
fn potential_matches_the_typed_formula() {
    let truth = sidecar(flat(3, 0.0, 100.0));
    let low = tick(0.0, 3.0, 60.0, 0.0, false);
    let at_apogee = tick(1.0, 0.0, 100.0, 0.0, false);
    let above = tick(2.0, 0.0, 140.0, 0.0, false);

    assert_eq!(potential(&low, Some(&truth), SHAPING).unwrap(), -38.0);
    assert_eq!(potential(&at_apogee, Some(&truth), SHAPING).unwrap(), 0.0);
    assert_eq!(potential(&above, Some(&truth), SHAPING).unwrap(), 0.0);
    assert_eq!(
        potential(
            &low,
            Some(&truth),
            ShapingWeights {
                apogee: 0.0,
                ..SHAPING
            }
        )
        .unwrap(),
        -18.0
    );
    assert_eq!(
        potential(
            &low,
            Some(&truth),
            ShapingWeights {
                lateral: 0.0,
                ..SHAPING
            }
        )
        .unwrap(),
        -20.0
    );
}

#[test]
fn dense_shaping_is_the_discounted_potential_difference() {
    let truth = sidecar(flat(3, 0.0, 100.0));
    let a = tick(0.0, 3.0, 60.0, 0.0, false);
    let b = tick(1.0, 1.0, 80.0, 0.0, false);
    let phi_a = potential(&a, Some(&truth), SHAPING).unwrap();
    let phi_b = potential(&b, Some(&truth), SHAPING).unwrap();
    assert_eq!(phi_a, -38.0);
    assert_eq!(phi_b, -12.0);

    let expected = SHAPING.gamma * phi_b - phi_a;
    assert_eq!(
        dense_reward(&a, &b, Some(&truth), SHAPING)
            .unwrap()
            .to_bits(),
        expected.to_bits()
    );
    assert_eq!(
        dense_reward(
            &a,
            &b,
            Some(&truth),
            ShapingWeights {
                gamma: 0.0,
                ..SHAPING
            }
        )
        .unwrap(),
        -phi_a
    );
}

#[test]
fn dense_shaping_telescopes_over_a_trajectory() {
    let truth = sidecar(flat(3, 0.0, 100.0));
    let flown = vec![
        tick(0.0, 8.0, 0.0, 0.0, false),
        tick(1.0, 4.0, 50.0, 0.0, false),
        tick(2.0, 1.0, 90.0, 0.0, false),
        tick(3.0, 0.0, 110.0, 0.0, false),
    ];

    let mut sum = 0.0;
    let mut discount = 1.0;
    for pair in flown.windows(2) {
        sum += discount * dense_reward(&pair[0], &pair[1], Some(&truth), SHAPING).unwrap();
        discount *= SHAPING.gamma;
    }
    let phi_0 = potential(&flown[0], Some(&truth), SHAPING).unwrap();
    let phi_t = potential(&flown[flown.len() - 1], Some(&truth), SHAPING).unwrap();
    let closed_form = SHAPING.gamma.powi((flown.len() - 1) as i32) * phi_t - phi_0;
    assert!((sum - closed_form).abs() < 1e-9 * (1.0 + closed_form.abs()));
    assert!(sum > 0.0);

    let again: f64 = flown
        .windows(2)
        .enumerate()
        .map(|(i, pair)| {
            SHAPING.gamma.powi(i as i32)
                * dense_reward(&pair[0], &pair[1], Some(&truth), SHAPING).unwrap()
        })
        .sum();
    assert_eq!(sum.to_bits(), again.to_bits());
}

#[test]
fn shaping_rejects_hard_gates_instead_of_penalising() {
    let state = tick(0.0, 3.0, 60.0, 0.0, false);
    let flown = vec![state];
    let base = sidecar(flat(3, 0.0, 100.0)).observed;
    let cases = [
        (
            GateObservation {
                mach_peak: 1.2,
                ..base
            },
            GateViolation::Mach {
                observed: 1.2,
                limit: 1.0,
            },
        ),
        (
            GateObservation {
                safety_factor_min: 1.2,
                ..base
            },
            GateViolation::SafetyFactor {
                observed: 1.2,
                limit: 1.5,
            },
        ),
        (
            GateObservation {
                static_margin_cal_min: 0.7,
                ..base
            },
            GateViolation::StaticMarginCal {
                observed: 0.7,
                limit: 1.0,
            },
        ),
    ];

    for (observed, violation) in cases {
        let mut truth = sidecar(flat(3, 0.0, 100.0));
        truth.observed = observed;
        let expected = Err(RewardError::GateRejected(violation));
        assert_eq!(potential(&state, Some(&truth), SHAPING), expected);
        assert_eq!(
            dense_reward(&state, &state, Some(&truth), SHAPING),
            expected
        );
        assert_eq!(
            evaluate(&flown, Some(&truth), WEIGHTS),
            Err(RewardError::GateRejected(violation))
        );
        assert_eq!(
            potential(
                &state,
                Some(&truth),
                ShapingWeights {
                    apogee: 1e6,
                    ..SHAPING
                }
            ),
            expected
        );
    }
}

#[test]
fn shaping_missing_truth_and_malformed_inputs_are_loud() {
    let truth = sidecar(flat(3, 0.0, 100.0));
    let state = tick(0.0, 3.0, 60.0, 0.0, false);
    let mut bad_position = state;
    bad_position.pos_m[1] = f64::NAN;

    assert_eq!(
        potential(&state, None, SHAPING),
        Err(RewardError::MissingTruth)
    );
    assert_eq!(
        dense_reward(&state, &state, None, SHAPING),
        Err(RewardError::MissingTruth)
    );
    assert_eq!(
        potential(&state, Some(&sidecar(vec![])), SHAPING),
        Err(RewardError::MissingReference)
    );
    assert_eq!(
        potential(&bad_position, Some(&truth), SHAPING),
        Err(RewardError::NonFinite("position"))
    );
    assert_eq!(
        dense_reward(&bad_position, &state, Some(&truth), SHAPING),
        Err(RewardError::NonFinite("position"))
    );
    assert_eq!(
        potential(
            &state,
            Some(&truth),
            ShapingWeights {
                lateral: -1.0,
                ..SHAPING
            }
        ),
        Err(RewardError::InvalidWeights("shaping lateral"))
    );
    assert_eq!(
        dense_reward(
            &state,
            &state,
            Some(&truth),
            ShapingWeights {
                gamma: 1.5,
                ..SHAPING
            }
        ),
        Err(RewardError::InvalidWeights("shaping gamma"))
    );
}
