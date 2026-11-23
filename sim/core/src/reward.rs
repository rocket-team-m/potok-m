// SPDX-License-Identifier: GPL-3.0-or-later
//! Verifiable RLVR reward for a flown trajectory against its truth sidecar.
//!
//! ```text
//! R = w_lat·lateral_improvement + w_apogee·apogee_gain
//!     − w_effort·control_integral − w_risk·safety_violations
//! ```
//!
//! The three hard constraints — peak Mach `M`, minimum structural safety
//! factor `SF`, and minimum static margin in calibers — are **rejection
//! gates**. A violation returns [`RewardError::GateRejected`] and no reward is
//! produced; the gates are never folded into the scalar. The
//! `safety_violations` count in the soft term is a separate per-tick
//! flight-recorder mark (health/actuator limit breached), not a gate.
//!
//! ## Dense potential-based shaping
//!
//! [`evaluate`] is the terminal verifier. On top of it, [`potential`] defines
//! `Φ(s) = −k·lateral_error(s)² − k_apogee·apogee_loss(s)` from privileged
//! truth, and [`dense_reward`] returns `γ·Φ(s′) − Φ(s)` for one transition.
//! Summed over a trajectory that term telescopes to `γᵀ·Φ(s_T) − Φ(s_0)`, so
//! it changes the speed of credit assignment, not the optimal policy (the
//! standard potential-based shaping identity). The weights are typed and
//! caller-supplied, never fitted to a flight log: the identity holds for any
//! finite potential. A hard gate still rejects — shaping never converts a
//! rejection into a soft penalty.
//!
//! This module is deliberately dependency-free: it owns its input types, so a
//! caller maps its own trajectory and sidecar records into them. No physics
//! constants, dataset numbers or calibration live here, and no source file
//! other than this one is needed to compile it. Every value is supplied by
//! the caller, the arithmetic is total, and equal inputs give bitwise equal
//! outputs.

use std::error::Error;
use std::fmt;

/// One recorded tick of the flown trajectory (or of the reference trajectory).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlightSample {
    /// Time since launch (s). Must be finite and strictly increasing in a
    /// series; the effort integral uses these timestamps, not a fixed dt.
    pub t_s: f64,
    /// World-frame position (m); index 2 is altitude. All components finite.
    pub pos_m: [f64; 3],
    /// Magnitude of the commanded lateral acceleration (m/s^2), finite and
    /// non-negative. Feeds the effort integral only — never a hard gate.
    pub ctrl_mps2: f64,
    /// Flight-recorder mark: this tick breached a soft health/actuator limit.
    /// Counted by `w_risk`; the M/SF/caliber gates are NOT represented here.
    pub safety_violation: bool,
}

/// Limits of the three hard constraints. A missing limit is a missing truth
/// record: construct these explicitly, never with a guessed default.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GateLimits {
    /// Maximum permitted Mach number (`M`).
    pub mach_max: f64,
    /// Minimum permitted structural safety factor (`SF`).
    pub safety_factor_min: f64,
    /// Minimum permitted static margin, in calibers.
    pub static_margin_cal_min: f64,
}

/// The observed values of the three hard constraints for this flight, as
/// recorded in the truth sidecar. The reward compares, it does not re-derive
/// aerodynamics; keeping both sides typed makes the comparison auditable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GateObservation {
    pub mach_peak: f64,
    pub safety_factor_min: f64,
    pub static_margin_cal_min: f64,
}

/// Everything the reward needs from the truth sidecar. All fields are
/// mandatory. The `Option` at the [`evaluate`] call site is what makes an
/// absent sidecar loud instead of silently scoring a ghost baseline.
#[derive(Debug, Clone, PartialEq)]
pub struct TruthSidecar {
    /// Reference trajectory: the baseline whose lateral RMS and apogee the
    /// flown trajectory is scored against.
    pub reference: Vec<FlightSample>,
    pub limits: GateLimits,
    pub observed: GateObservation,
}

/// Typed reward weights. All must be finite and non-negative, so the signs in
/// the formula cannot be inverted by a malformed weight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RewardWeights {
    pub lateral: f64,
    pub apogee: f64,
    pub effort: f64,
    pub risk: f64,
}

/// Typed weights for the dense potential-based shaping term, kept separate
/// from [`RewardWeights`] so terminal and dense scales cannot be mixed by
/// accident. There is no default: a caller supplies every field explicitly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShapingWeights {
    /// `k` on the quadratic lateral-error term.
    pub lateral: f64,
    /// `k_apogee` on the apogee-shortfall term.
    pub apogee: f64,
    /// Discount `γ` in `γ·Φ(s′) − Φ(s)`; finite and in `[0, 1]`.
    pub gamma: f64,
}

/// Which hard gate rejected the flight, with the two compared values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GateViolation {
    Mach { observed: f64, limit: f64 },
    SafetyFactor { observed: f64, limit: f64 },
    StaticMarginCal { observed: f64, limit: f64 },
}

/// Every failure is loud and names its cause. There is no clamp, zero-fill or
/// fallback path anywhere in the reward computation.
#[derive(Debug, Clone, PartialEq)]
pub enum RewardError {
    /// The truth sidecar was absent. Never scored as a zero baseline.
    MissingTruth,
    EmptyTrajectory,
    MissingReference,
    NonFinite(&'static str),
    NonMonotonicTime {
        index: usize,
    },
    NegativeControl {
        index: usize,
    },
    InvalidWeights(&'static str),
    /// A hard constraint failed: the flight is rejected, not penalised.
    GateRejected(GateViolation),
}

impl fmt::Display for RewardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingTruth => write!(f, "reward: truth sidecar is missing"),
            Self::EmptyTrajectory => write!(f, "reward: flown trajectory is empty"),
            Self::MissingReference => write!(f, "reward: truth reference trajectory is empty"),
            Self::NonFinite(field) => write!(f, "reward: non-finite {field}"),
            Self::NonMonotonicTime { index } => {
                write!(
                    f,
                    "reward: time is not strictly increasing at sample {index}"
                )
            }
            Self::NegativeControl { index } => {
                write!(f, "reward: negative control magnitude at sample {index}")
            }
            Self::InvalidWeights(field) => write!(f, "reward: invalid weight {field}"),
            Self::GateRejected(v) => match v {
                GateViolation::Mach { observed, limit } => {
                    write!(f, "reward: M gate rejected: mach {observed} > {limit}")
                }
                GateViolation::SafetyFactor { observed, limit } => {
                    write!(f, "reward: SF gate rejected: factor {observed} < {limit}")
                }
                GateViolation::StaticMarginCal { observed, limit } => write!(
                    f,
                    "reward: caliber-margin gate rejected: {observed} cal < {limit} cal"
                ),
            },
        }
    }
}

impl Error for RewardError {}

/// A scored flight with its full arithmetic exposed. `total` is computed once
/// from the four terms; [`Reward::check`] recomputes them for verification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Reward {
    pub lateral_improvement_m: f64,
    pub apogee_gain_m: f64,
    pub control_integral: f64,
    pub safety_violations: u32,
    pub lateral_term: f64,
    pub apogee_term: f64,
    pub effort_term: f64,
    pub risk_term: f64,
    pub total: f64,
}

impl Reward {
    /// Recompute every term from the raw metrics and the weights supplied, and
    /// compare bitwise with the stored values. False means the record was
    /// altered or the caller used different weights.
    pub fn check(&self, weights: RewardWeights) -> bool {
        let lateral = weights.lateral * self.lateral_improvement_m;
        let apogee = weights.apogee * self.apogee_gain_m;
        let effort = -weights.effort * self.control_integral;
        let risk = -weights.risk * f64::from(self.safety_violations);
        let total = lateral + apogee + effort + risk;
        lateral.to_bits() == self.lateral_term.to_bits()
            && apogee.to_bits() == self.apogee_term.to_bits()
            && effort.to_bits() == self.effort_term.to_bits()
            && risk.to_bits() == self.risk_term.to_bits()
            && total.to_bits() == self.total.to_bits()
    }
}

/// Score `flown` against `truth`. A missing sidecar, malformed input or hard
/// gate failure is an error; there is no partial or fallback score.
pub fn evaluate(
    flown: &[FlightSample],
    truth: Option<&TruthSidecar>,
    weights: RewardWeights,
) -> Result<Reward, RewardError> {
    let truth = truth.ok_or(RewardError::MissingTruth)?;
    check_weights(weights)?;
    check_gates(truth)?;

    if flown.is_empty() {
        return Err(RewardError::EmptyTrajectory);
    }
    if truth.reference.is_empty() {
        return Err(RewardError::MissingReference);
    }
    check_series(flown)?;
    check_series(&truth.reference)?;

    let lateral_improvement_m = lateral_rms(&truth.reference) - lateral_rms(flown);
    let apogee_gain_m = apogee_m(flown) - apogee_m(&truth.reference);
    let control_integral = control_integral(flown);
    let safety_violations = flown.iter().filter(|s| s.safety_violation).count() as u32;

    let lateral_term = weights.lateral * lateral_improvement_m;
    let apogee_term = weights.apogee * apogee_gain_m;
    let effort_term = -weights.effort * control_integral;
    let risk_term = -weights.risk * f64::from(safety_violations);
    let total = lateral_term + apogee_term + effort_term + risk_term;

    Ok(Reward {
        lateral_improvement_m,
        apogee_gain_m,
        control_integral,
        safety_violations,
        lateral_term,
        apogee_term,
        effort_term,
        risk_term,
        total,
    })
}

/// The potential `Φ(s) = −lateral·lateral_error(s)² − apogee·apogee_loss(s)`.
///
/// `lateral_error(s)` is the horizontal displacement magnitude `√(x² + y²)`
/// at this tick — the same quantity the terminal verifier aggregates as
/// `lateral_rms_m`. `apogee_loss(s)` is `max(0, reference_apogee − altitude)`:
/// the remaining shortfall below the truth reference's apogee. Both come from
/// privileged truth; no aerodynamics or gravity constant is re-derived here.
/// Missing truth, malformed input, invalid weights or a rejected hard gate is
/// a loud error.
pub fn potential(
    state: &FlightSample,
    truth: Option<&TruthSidecar>,
    weights: ShapingWeights,
) -> Result<f64, RewardError> {
    let truth = truth.ok_or(RewardError::MissingTruth)?;
    check_shaping_weights(weights)?;
    check_gates(truth)?;
    if truth.reference.is_empty() {
        return Err(RewardError::MissingReference);
    }
    for sample in &truth.reference {
        check_position(sample)?;
    }
    check_position(state)?;

    let lateral_sq = state.pos_m[0].powi(2) + state.pos_m[1].powi(2);
    let apogee_loss = (apogee_m(&truth.reference) - state.pos_m[2]).max(0.0);
    Ok(-weights.lateral * lateral_sq - weights.apogee * apogee_loss)
}

/// The dense shaping reward for one transition `s → s'`: `γ·Φ(s′) − Φ(s)`.
/// Add it to the terminal [`evaluate`] reward on each tick; over a whole
/// trajectory the terms telescope to `γᵀ·Φ(s_T) − Φ(s_0)`. Hard gates reject
/// here as well, so a rejected flight earns neither terminal nor dense reward.
pub fn dense_reward(
    state: &FlightSample,
    next: &FlightSample,
    truth: Option<&TruthSidecar>,
    weights: ShapingWeights,
) -> Result<f64, RewardError> {
    let phi = potential(state, truth, weights)?;
    let phi_next = potential(next, truth, weights)?;
    Ok(weights.gamma * phi_next - phi)
}

fn check_shaping_weights(w: ShapingWeights) -> Result<(), RewardError> {
    if !w.lateral.is_finite() || w.lateral < 0.0 {
        return Err(RewardError::InvalidWeights("shaping lateral"));
    }
    if !w.apogee.is_finite() || w.apogee < 0.0 {
        return Err(RewardError::InvalidWeights("shaping apogee"));
    }
    if !w.gamma.is_finite() || !(0.0..=1.0).contains(&w.gamma) {
        return Err(RewardError::InvalidWeights("shaping gamma"));
    }
    Ok(())
}

fn check_position(sample: &FlightSample) -> Result<(), RewardError> {
    if sample.pos_m.iter().any(|v| !v.is_finite()) {
        return Err(RewardError::NonFinite("position"));
    }
    Ok(())
}

fn check_weights(w: RewardWeights) -> Result<(), RewardError> {
    if !w.lateral.is_finite() || w.lateral < 0.0 {
        return Err(RewardError::InvalidWeights("lateral"));
    }
    if !w.apogee.is_finite() || w.apogee < 0.0 {
        return Err(RewardError::InvalidWeights("apogee"));
    }
    if !w.effort.is_finite() || w.effort < 0.0 {
        return Err(RewardError::InvalidWeights("effort"));
    }
    if !w.risk.is_finite() || w.risk < 0.0 {
        return Err(RewardError::InvalidWeights("risk"));
    }
    Ok(())
}

/// Hard gates run before any arithmetic. Finiteness is checked first: a NaN
/// observation would otherwise fail every `>` / `<` comparison and slip
/// through as accepted.
fn check_gates(truth: &TruthSidecar) -> Result<(), RewardError> {
    let o = truth.observed;
    let l = truth.limits;
    for (name, value) in [
        ("observed mach", o.mach_peak),
        ("observed safety factor", o.safety_factor_min),
        ("observed static margin", o.static_margin_cal_min),
        ("mach limit", l.mach_max),
        ("safety factor limit", l.safety_factor_min),
        ("static margin limit", l.static_margin_cal_min),
    ] {
        if !value.is_finite() {
            return Err(RewardError::NonFinite(name));
        }
    }
    if o.mach_peak > l.mach_max {
        return Err(RewardError::GateRejected(GateViolation::Mach {
            observed: o.mach_peak,
            limit: l.mach_max,
        }));
    }
    if o.safety_factor_min < l.safety_factor_min {
        return Err(RewardError::GateRejected(GateViolation::SafetyFactor {
            observed: o.safety_factor_min,
            limit: l.safety_factor_min,
        }));
    }
    if o.static_margin_cal_min < l.static_margin_cal_min {
        return Err(RewardError::GateRejected(GateViolation::StaticMarginCal {
            observed: o.static_margin_cal_min,
            limit: l.static_margin_cal_min,
        }));
    }
    Ok(())
}

fn check_series(samples: &[FlightSample]) -> Result<(), RewardError> {
    let mut prev = f64::NEG_INFINITY;
    for (i, s) in samples.iter().enumerate() {
        if !s.t_s.is_finite() {
            return Err(RewardError::NonFinite("time"));
        }
        if s.pos_m.iter().any(|v| !v.is_finite()) {
            return Err(RewardError::NonFinite("position"));
        }
        if !s.ctrl_mps2.is_finite() {
            return Err(RewardError::NonFinite("control magnitude"));
        }
        if s.t_s <= prev {
            return Err(RewardError::NonMonotonicTime { index: i });
        }
        if s.ctrl_mps2 < 0.0 {
            return Err(RewardError::NegativeControl { index: i });
        }
        prev = s.t_s;
    }
    Ok(())
}

/// RMS horizontal displacement (m), sample-weighted, matching the simulator's
/// `lateral_rms_m`. Sampling is uniform in the recorder, so this is the same
/// quantity as a time-weighted RMS up to the first and last half-interval.
fn lateral_rms(samples: &[FlightSample]) -> f64 {
    let sum: f64 = samples
        .iter()
        .map(|s| s.pos_m[0].powi(2) + s.pos_m[1].powi(2))
        .sum();
    (sum / samples.len() as f64).sqrt()
}

fn apogee_m(samples: &[FlightSample]) -> f64 {
    samples
        .iter()
        .fold(f64::NEG_INFINITY, |a, s| a.max(s.pos_m[2]))
}

/// Trapezoidal ∫ |u| dt. Requires strictly increasing time (checked above),
/// so it is exact for the supplied samples and does not assume a fixed dt.
fn control_integral(samples: &[FlightSample]) -> f64 {
    samples
        .windows(2)
        .map(|w| 0.5 * (w[0].ctrl_mps2 + w[1].ctrl_mps2) * (w[1].t_s - w[0].t_s))
        .sum()
}
