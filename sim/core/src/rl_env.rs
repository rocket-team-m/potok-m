// SPDX-License-Identifier: GPL-3.0-or-later
//! Stepwise RL environment around one deterministic Potok-M episode.
//!
//! The simulator's control seam is push-based: `run_episode_with_control_ctx`
//! calls a closure at every 50 Hz tick and asks for the command to hold until
//! the next tick. This module turns that into the usual pull-based env loop
//! (`reset` → `step(action)` → `step(action)` …) with a lockstep channel
//! handshake on a worker thread. The worker owns the simulator; the caller
//! only ever sees owned values, so no `ControlContext` borrow escapes.
//!
//! Observation = 14 model-frame features + 6 conditioning values (20 f32).
//! Action = world-frame lateral acceleration `[ay, ax]` in m/s², bounded by
//! [`MAX_COMMAND_MPS2`] and mapped to the fin seam as `[ay, -ax]`.
//! Reward = potential-based shaping on privileged truth each tick plus the
//! terminal `reward::evaluate` score. Gates stay rejection, not a scalar:
//! a rejected flight has no verifier value in `info`, only a named violation.
//!
//! [`RlEnv::step_shielded`] is the shielded training entry point: a typed
//! online filter verifies the candidate demand against the runtime Mach
//! sample, the command authority and the current phase before the simulator
//! sees it. A veto falls back to zero with a named reason and is counted; no
//! gate limit or reward weight is changed. This is an in-simulator instrument,
//! not a flight claim.
//!
//! No RNG of its own, no wall clock, no training algorithm, no GPU. Two envs
//! driven by the same seed and action sequence are bitwise identical.

use std::fmt;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread::{self, JoinHandle};

use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::{
    config::{sample_domain, Conditioning, Config},
    control::MAX_COMMAND_MPS2,
    dynamics::Phase,
    episode::{ControlContext, Episode, Simulator},
    error::{Result, SimError},
    math::episode_seed,
    reward::{
        evaluate, FlightSample, GateLimits, GateObservation, GateViolation, Reward, RewardError,
        RewardWeights, TruthSidecar,
    },
    sensor::{feature, Frame},
};

/// Raw 8 + physics 6, in the canonical model-frame order. The tail matches
/// `physics::PHYSICS_FEATURES`; the head is the legacy validation feature set.
pub const STATE_NAMES: [&str; 14] = [
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "acc_x",
    "acc_y",
    "acc_z",
    "baro_h",
    "pitot_q",
    "alpha_deg",
    "beta_deg",
    "q_dyn_kpa",
    "a_lat_resid",
    "drag_est",
    "static_margin_cal",
];
pub const STATE_DIM: usize = STATE_NAMES.len();
pub const CONTEXT_DIM: usize = Conditioning::COUNT;
pub const OBS_DIM: usize = STATE_DIM + CONTEXT_DIM;
pub const ACTION_DIM: usize = 2;

/// Privileged truth available at the control seam. Gust/shear metadata is not
/// exposed there, so it is absent here rather than guessed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TickTruth {
    pub t_s: f64,
    pub pos_m: [f64; 3],
    pub vel_mps: [f64; 3],
    pub quat: [f64; 4],
    pub omega_radps: [f64; 3],
    pub mass_kg: f64,
    pub wind_mps: [f64; 3],
}

impl TickTruth {
    fn from_ctx(ctx: &ControlContext<'_>) -> Self {
        let s = ctx.state;
        Self {
            t_s: f64::from(ctx.frame.time_us) * 1e-6,
            pos_m: s.pos(),
            vel_mps: s.vel(),
            quat: s.quat(),
            omega_radps: s.omega(),
            mass_kg: s.mass(),
            wind_mps: ctx.wind,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Observation {
    pub tick: u32,
    pub phase: Phase,
    pub features: [f32; STATE_DIM],
    pub conditioning: [f32; CONTEXT_DIM],
    pub ai_input_valid: bool,
}

impl Observation {
    /// Concatenate features and conditioning into one flat model input.
    pub fn as_array(self) -> [f32; OBS_DIM] {
        let mut out = [0.0; OBS_DIM];
        out[..STATE_DIM].copy_from_slice(&self.features);
        out[STATE_DIM..].copy_from_slice(&self.conditioning);
        out
    }
}

fn state_features(frame: &Frame) -> [f32; STATE_DIM] {
    let mut out = [0.0; STATE_DIM];
    for (dst, name) in out.iter_mut().zip(STATE_NAMES.iter()) {
        *dst = feature(frame, name).expect("STATE_NAMES are canonical sensor features");
    }
    out
}

/// Bounded world-frame lateral acceleration demand `[ay, ax]` (m/s²).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Action {
    pub ay: f32,
    pub ax: f32,
}

impl Action {
    pub const ZERO: Self = Self { ay: 0.0, ax: 0.0 };

    /// Refuse non-finite or out-of-box demands; never clamp them.
    pub fn checked(self) -> Result<Self> {
        let limit = MAX_COMMAND_MPS2 as f32;
        for (name, v) in [("ay", self.ay), ("ax", self.ax)] {
            if !v.is_finite() || v.abs() > limit {
                return Err(SimError::Control(format!(
                    "rl_env: action {name}={v} outside [-{MAX_COMMAND_MPS2}, {MAX_COMMAND_MPS2}] m/s^2"
                )));
            }
        }
        Ok(self)
    }

    /// World `[ay, ax]` → fin seam `[roll, pitch]`, matching
    /// `ReferenceOptimizer::fin_accel_command`.
    pub fn fin_command(self) -> [f64; 2] {
        [f64::from(self.ay), -f64::from(self.ax)]
    }
}

/// Why a candidate action was replaced by the safe fallback.
///
/// One variant per runtime constraint plus an unverifiable-input case. Every
/// variant names the values that caused the veto, so an intervention is
/// auditable without reading the simulator's state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ShieldReason {
    /// No trustworthy Mach evidence: fail safe without deriving one here.
    MachUnavailable { observed: f64 },
    /// Certified Mach ceiling exceeded. `limit` is the verifier's own gate.
    MachLimit { observed: f64, limit: f64 },
    /// Demand is non-finite or outside the command authority box. Refused,
    /// never clamped to the box.
    Authority { ay: f32, ax: f32, limit: f32 },
    /// AI corrections are forbidden in this flight phase.
    PhaseForbidden { phase: Phase },
}

impl fmt::Display for ShieldReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MachUnavailable { observed } => {
                write!(f, "shield: Mach evidence {observed} is not finite")
            }
            Self::MachLimit { observed, limit } => {
                write!(f, "shield: Mach {observed} above certified ceiling {limit}")
            }
            Self::Authority { ay, ax, limit } => write!(
                f,
                "shield: demand [ay={ay}, ax={ax}] outside authority +/-{limit} m/s^2"
            ),
            Self::PhaseForbidden { phase } => {
                write!(f, "shield: AI corrections forbidden in phase {phase:?}")
            }
        }
    }
}

/// Runtime inputs the shield verifies against but does not derive itself.
///
/// The shield runs no aerodynamics and owns no RNG. `mach` is a caller-derived
/// estimate (the same split the firmware runtime shield uses); `phase` is the
/// flight phase at the moment the action is chosen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShieldSample {
    pub mach: f64,
    pub phase: Phase,
}

/// What actually reaches the simulator, and why it is not the candidate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShieldDecision {
    pub action: Action,
    pub reason: Option<ShieldReason>,
}

impl ShieldDecision {
    pub fn intervened(&self) -> bool {
        self.reason.is_some()
    }
}

/// One counter per reason plus accepted. Counts survive `RlEnv::reset`: they
/// describe a training run, not one episode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ShieldCounts {
    pub accepted: u64,
    pub mach_unavailable: u64,
    pub mach_limit: u64,
    pub authority: u64,
    pub phase_forbidden: u64,
}

impl ShieldCounts {
    fn record(&mut self, reason: Option<ShieldReason>) {
        match reason {
            None => self.accepted += 1,
            Some(ShieldReason::MachUnavailable { .. }) => self.mach_unavailable += 1,
            Some(ShieldReason::MachLimit { .. }) => self.mach_limit += 1,
            Some(ShieldReason::Authority { .. }) => self.authority += 1,
            Some(ShieldReason::PhaseForbidden { .. }) => self.phase_forbidden += 1,
        }
    }

    /// Every veto, whatever its reason.
    pub fn interventions(&self) -> u64 {
        self.mach_unavailable + self.mach_limit + self.authority + self.phase_forbidden
    }

    pub fn total(&self) -> u64 {
        self.accepted + self.interventions()
    }
}

/// Typed online shield on the candidate-action path.
///
/// `verify` is the only way a command leaves this module: it both picks the
/// action that reaches the simulator and records the decision, so an
/// intervention cannot happen uncounted. The safe fallback is always
/// [`Action::ZERO`] (no new demand), never a clipped version of the bad demand.
///
/// Check priority is fixed and loud: unverifiable Mach → Mach ceiling →
/// forbidden phase → authority box → accept. This is an online *veto*, not a
/// reward term: the verifier's gates and weights are unchanged. It is a
/// training/in-simulator instrument only — no flight claim is made here.
#[derive(Debug, Clone, PartialEq)]
pub struct Shield {
    mach_max: f64,
    authority: f32,
    counts: ShieldCounts,
}

impl Shield {
    /// Build from the verifier's own limits and the simulator's command
    /// authority. No new gate values are introduced or tuned here.
    pub fn new(limits: GateLimits) -> Result<Self> {
        if !limits.mach_max.is_finite() || limits.mach_max < 0.0 {
            return Err(SimError::Config(format!(
                "rl_env: shield Mach ceiling {} must be finite and non-negative",
                limits.mach_max
            )));
        }
        Ok(Self {
            mach_max: limits.mach_max,
            authority: MAX_COMMAND_MPS2 as f32,
            counts: ShieldCounts::default(),
        })
    }

    pub fn mach_max(&self) -> f64 {
        self.mach_max
    }
    pub fn authority(&self) -> f32 {
        self.authority
    }
    pub fn counts(&self) -> ShieldCounts {
        self.counts
    }
    pub fn reset_counts(&mut self) {
        self.counts = ShieldCounts::default();
    }

    /// Verify one candidate against the runtime sample. Records the outcome.
    pub fn verify(&mut self, candidate: Action, sample: ShieldSample) -> ShieldDecision {
        let reason = if !sample.mach.is_finite() {
            Some(ShieldReason::MachUnavailable {
                observed: sample.mach,
            })
        } else if sample.mach > self.mach_max {
            Some(ShieldReason::MachLimit {
                observed: sample.mach,
                limit: self.mach_max,
            })
        } else if !sample.phase.ai_allowed() {
            Some(ShieldReason::PhaseForbidden {
                phase: sample.phase,
            })
        } else {
            let limit = self.authority;
            if !candidate.ay.is_finite()
                || !candidate.ax.is_finite()
                || candidate.ay.abs() > limit
                || candidate.ax.abs() > limit
            {
                Some(ShieldReason::Authority {
                    ay: candidate.ay,
                    ax: candidate.ax,
                    limit,
                })
            } else {
                None
            }
        };
        let action = if reason.is_some() {
            Action::ZERO
        } else {
            candidate
        };
        let decision = ShieldDecision { action, reason };
        self.counts.record(decision.reason);
        decision
    }
}

/// Potential-based shaping. `bonus = γ·Φ(s') − Φ(s)` with Φ a function of
/// privileged truth only, so it cannot change the optimal policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shaping {
    pub gamma: f64,
    pub lateral: f64,
    pub apogee: f64,
    pub alpha_deg: f64,
}

impl Default for Shaping {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            lateral: 1.0,
            apogee: 1.0,
            alpha_deg: 0.01,
        }
    }
}

impl Shaping {
    pub fn validate(&self) -> Result<()> {
        if !self.gamma.is_finite() || !(0.0..=1.0).contains(&self.gamma) {
            return Err(SimError::Config(format!(
                "rl_env: shaping gamma {} outside [0, 1]",
                self.gamma
            )));
        }
        for (name, w) in [
            ("lateral", self.lateral),
            ("apogee", self.apogee),
            ("alpha_deg", self.alpha_deg),
        ] {
            if !w.is_finite() || w < 0.0 {
                return Err(SimError::Config(format!(
                    "rl_env: shaping {name} = {w} must be finite and non-negative"
                )));
            }
        }
        Ok(())
    }

    pub fn potential(&self, truth: &TickTruth, alpha_deg: f64) -> f64 {
        let drift = truth.pos_m[0].hypot(truth.pos_m[1]);
        -self.lateral * drift + self.apogee * truth.pos_m[2] - self.alpha_deg * alpha_deg.abs()
    }

    pub fn bonus(&self, phi_before: f64, phi_after: f64) -> f64 {
        self.gamma * phi_after - phi_before
    }
}

/// The three hard constraints as supplied by the verifier's truth sidecar.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GateStatus {
    pub limits: GateLimits,
    pub observed: GateObservation,
    /// Filled only at terminal, from the verifier's own rejection.
    pub rejection: Option<GateViolation>,
}

impl GateStatus {
    fn new(truth: &TruthSidecar) -> Self {
        Self {
            limits: truth.limits,
            observed: truth.observed,
            rejection: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnvInfo {
    pub seed: u64,
    pub episode_index: u64,
    pub tick: u32,
    pub truth: TickTruth,
    pub gates: GateStatus,
    /// Terminal verifier score. Absent when the gates rejected the flight.
    pub verifier: Option<Reward>,
    /// Named verifier failure, when the terminal evaluation did not produce a score.
    pub verify_error: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Step {
    pub obs: Observation,
    pub reward: f64,
    pub done: bool,
    pub info: EnvInfo,
}

type WorkerMsg = (Observation, TickTruth);

/// One episode, driven one 50 Hz tick at a time.
pub struct RlEnv {
    cfg: Config,
    index: u64,
    seed: u64,
    truth: TruthSidecar,
    weights: RewardWeights,
    shaping: Shaping,
    conditioning: Conditioning,
    gates: GateStatus,
    shield: Shield,
    last_shield_decision: Option<ShieldDecision>,
    obs_rx: Option<Receiver<WorkerMsg>>,
    act_tx: Option<Sender<Action>>,
    worker: Option<JoinHandle<Result<Episode>>>,
    last: Option<Observation>,
    last_truth: Option<TickTruth>,
    actions: Vec<[f64; 2]>,
    prev_phi: f64,
    done: bool,
    total_reward: f64,
}

impl RlEnv {
    pub fn new(
        cfg: Config,
        index: u64,
        truth: TruthSidecar,
        weights: RewardWeights,
        shaping: Shaping,
    ) -> Result<Self> {
        cfg.validate()?;
        shaping.validate()?;
        for (name, w) in [
            ("lateral", weights.lateral),
            ("apogee", weights.apogee),
            ("effort", weights.effort),
            ("risk", weights.risk),
        ] {
            if !w.is_finite() || w < 0.0 {
                return Err(SimError::Config(format!(
                    "rl_env: reward weight {name} = {w} must be finite and non-negative"
                )));
            }
        }
        let seed = cfg.run.seed;
        let conditioning = initial_conditioning(&cfg, index);
        let gates = GateStatus::new(&truth);
        let shield = Shield::new(truth.limits)?;
        Ok(Self {
            cfg,
            index,
            seed,
            truth,
            weights,
            shaping,
            conditioning,
            gates,
            shield,
            last_shield_decision: None,
            obs_rx: None,
            act_tx: None,
            worker: None,
            last: None,
            last_truth: None,
            actions: Vec::new(),
            prev_phi: 0.0,
            done: false,
            total_reward: 0.0,
        })
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }
    pub fn seed(&self) -> u64 {
        self.seed
    }
    pub fn conditioning(&self) -> Conditioning {
        self.conditioning
    }
    pub fn is_done(&self) -> bool {
        self.done
    }
    pub fn total_reward(&self) -> f64 {
        self.total_reward
    }
    /// The online shield's cumulative decisions. Survives `reset`.
    pub fn shield_counts(&self) -> ShieldCounts {
        self.shield.counts()
    }
    /// The most recent shielded step's decision, if `step_shielded` was used.
    pub fn last_shield_decision(&self) -> Option<ShieldDecision> {
        self.last_shield_decision
    }

    /// Start a fresh episode and return its first observation (no action yet).
    pub fn reset(&mut self) -> Result<(Observation, EnvInfo)> {
        // Dropping the action sender unblocks the worker's `recv`; join before
        // starting the next episode so no two GPU/CPU-heavy runs overlap.
        self.act_tx.take();
        self.obs_rx.take();
        if let Some(h) = self.worker.take() {
            let _ = h.join();
        }
        self.actions.clear();
        self.last = None;
        self.last_truth = None;
        self.done = false;
        self.total_reward = 0.0;
        self.prev_phi = 0.0;
        self.last_shield_decision = None;

        let (obs_tx, obs_rx) = channel::<WorkerMsg>();
        let (act_tx, act_rx) = channel::<Action>();
        let cfg = self.cfg.clone();
        let conditioning = self.conditioning;
        let worker = thread::spawn(move || -> Result<Episode> {
            let sim = Simulator::new(cfg)?;
            sim.run_episode_with_control_ctx(0, &|ctx: &ControlContext<'_>| {
                let msg = (
                    Observation {
                        tick: u32::from(ctx.frame.seq),
                        phase: ctx.phase,
                        features: state_features(ctx.frame),
                        conditioning: conditioning.as_array(),
                        ai_input_valid: ctx.frame.ai_input_valid(),
                    },
                    TickTruth::from_ctx(ctx),
                );
                if obs_tx.send(msg).is_err() {
                    return (0.0, 0.0);
                }
                match act_rx.recv() {
                    Ok(a) => a.fin_command().into(),
                    Err(_) => (0.0, 0.0),
                }
            })
        });
        self.obs_rx = Some(obs_rx);
        self.act_tx = Some(act_tx);
        self.worker = Some(worker);

        let (obs, truth) = self
            .obs_rx
            .as_ref()
            .expect("receiver just installed")
            .recv()
            .map_err(|_| {
                SimError::Dataset("rl_env: worker stopped before first observation".into())
            })?;
        self.prev_phi = self.shaping.potential(&truth, f64::from(obs.features[8]));
        self.last = Some(obs);
        self.last_truth = Some(truth);
        let info = self.info(obs.tick, truth, None, None);
        Ok((obs, info))
    }

    /// Shielded counterpart of [`RlEnv::step`]: verify a candidate demand
    /// against the runtime Mach sample and the current observation's phase,
    /// then apply the decision. `mach` is caller-derived — this module does no
    /// aerodynamics and never guesses a Mach number. The phase comes from the
    /// environment's own last observation, so it cannot be supplied wrongly.
    ///
    /// The veto replaces the demand with [`Action::ZERO`]; it does not change
    /// the verifier's gate limits or reward weights.
    pub fn step_shielded(&mut self, candidate: Action, mach: f64) -> Result<Step> {
        let phase = self
            .last
            .ok_or_else(|| SimError::Control("rl_env: reset must precede step".into()))?
            .phase;
        let decision = self.shield.verify(candidate, ShieldSample { mach, phase });
        self.last_shield_decision = Some(decision);
        self.step(decision.action)
    }

    /// Apply one action and advance exactly one 50 Hz tick.
    pub fn step(&mut self, action: Action) -> Result<Step> {
        if self.done {
            return Err(SimError::Control(
                "rl_env: step after terminal; call reset".into(),
            ));
        }
        let action = action.checked()?;
        self.last
            .ok_or_else(|| SimError::Control("rl_env: reset must precede step".into()))?;
        let mut last = self.last.expect("checked above");
        let mut last_truth = self.last_truth.expect("reset sets truth");

        self.act_tx
            .as_ref()
            .ok_or_else(|| SimError::Control("rl_env: no active episode".into()))?
            .send(action)
            .map_err(|_| SimError::Dataset("rl_env: worker stopped before action".into()))?;
        self.actions
            .push([f64::from(action.ay), f64::from(action.ax)]);

        let mut shaping = 0.0;
        let (next, next_truth) = match self.recv()? {
            Some(msg) => msg,
            None => return self.finish(last, last_truth, shaping),
        };
        let phi = self
            .shaping
            .potential(&next_truth, f64::from(next.features[8]));
        shaping += self.shaping.bonus(self.prev_phi, phi);
        self.prev_phi = phi;
        last = next;
        last_truth = next_truth;

        let last_tick = self.cfg.frames_per_episode().saturating_sub(1) as u32;
        if next.phase == Phase::Landed || next.tick >= last_tick {
            // Keep the handshake alive until the worker finishes the episode;
            // zero commands after landing do not change the scored trajectory.
            loop {
                self.act_tx
                    .as_ref()
                    .ok_or_else(|| SimError::Control("rl_env: no active episode".into()))?
                    .send(Action::ZERO)
                    .map_err(|_| {
                        SimError::Dataset("rl_env: worker stopped while draining".into())
                    })?;
                self.actions.push([0.0, 0.0]);
                match self.recv()? {
                    Some((o, t)) => {
                        let phi = self.shaping.potential(&t, f64::from(o.features[8]));
                        shaping += self.shaping.bonus(self.prev_phi, phi);
                        self.prev_phi = phi;
                        last = o;
                        last_truth = t;
                    }
                    None => break,
                }
            }
            return self.finish(last, last_truth, shaping);
        }

        self.last = Some(next);
        self.last_truth = Some(next_truth);
        let info = self.info(next.tick, next_truth, None, None);
        Ok(Step {
            obs: next,
            reward: shaping,
            done: false,
            info,
        })
    }

    fn recv(&self) -> Result<Option<WorkerMsg>> {
        match self
            .obs_rx
            .as_ref()
            .ok_or_else(|| SimError::Control("rl_env: no active episode".into()))?
            .recv()
        {
            Ok(msg) => Ok(Some(msg)),
            Err(std::sync::mpsc::RecvError) => Ok(None),
        }
    }

    fn finish(&mut self, last: Observation, last_truth: TickTruth, shaping: f64) -> Result<Step> {
        let handle = self
            .worker
            .take()
            .ok_or_else(|| SimError::Control("rl_env: no worker to join".into()))?;
        let episode = handle
            .join()
            .map_err(|_| SimError::Dataset("rl_env: worker panicked".into()))??;

        let n = episode.trajectory.len();
        if self.actions.len() != n {
            return Err(SimError::Dataset(format!(
                "rl_env: {} actions for {} trajectory ticks",
                self.actions.len(),
                n
            )));
        }
        let tick_s = self.cfg.steps_per_frame() as f64 * self.cfg.run.dt;
        let flown: Vec<FlightSample> = (0..n)
            .map(|i| {
                let t = &episode.trajectory[i];
                let [ay, ax] = self.actions[i];
                FlightSample {
                    t_s: i as f64 * tick_s,
                    pos_m: [
                        f64::from(t.pos[0]),
                        f64::from(t.pos[1]),
                        f64::from(t.pos[2]),
                    ],
                    ctrl_mps2: ay.hypot(ax),
                    safety_violation: episode
                        .frames
                        .get(i)
                        .map(|f| f.saturated || f.spike)
                        .unwrap_or(false),
                }
            })
            .collect();

        let mut gates = self.gates;
        let mut verifier = None;
        let mut verify_error = None;
        match evaluate(&flown, Some(&self.truth), self.weights) {
            Ok(r) => verifier = Some(r),
            Err(e) => {
                if let RewardError::GateRejected(v) = &e {
                    gates.rejection = Some(*v);
                }
                verify_error = Some(e.to_string());
            }
        }
        let terminal = verifier.map(|r| r.total).unwrap_or(0.0);
        let reward = shaping + terminal;
        self.total_reward += reward;
        self.done = true;
        self.last = Some(last);
        self.last_truth = Some(last_truth);
        let mut info = self.info(last.tick, last_truth, verifier, verify_error);
        info.gates = gates;
        Ok(Step {
            obs: last,
            reward,
            done: true,
            info,
        })
    }

    fn info(
        &self,
        tick: u32,
        truth: TickTruth,
        verifier: Option<Reward>,
        verify_error: Option<String>,
    ) -> EnvInfo {
        EnvInfo {
            seed: self.seed,
            episode_index: self.index,
            tick,
            truth,
            gates: self.gates,
            verifier,
            verify_error,
        }
    }
}

impl Drop for RlEnv {
    fn drop(&mut self) {
        // Closing the command channel releases the worker's blocking recv;
        // dropping the observation receiver makes its next send fail and the
        // episode unwind on its own. No join here: Drop must not block.
        self.act_tx.take();
        self.obs_rx.take();
    }
}

/// Reproduce the episode's `INIT` + `DOMA` draws so conditioning is known at
/// reset, before the worker has produced an episode to read it from. Pinned
/// against `run_episode` by a test.
fn initial_conditioning(cfg: &Config, index: u64) -> Conditioning {
    let mut rng = StdRng::seed_from_u64(episode_seed(cfg.run.seed, index ^ 0x494E_4954));
    let mass = cfg.initial.mass_kg.sample(rng.gen());
    let _thrust_scale = cfg.initial.thrust_scale.sample(rng.gen());
    let _pitch = cfg.initial.pitch_deg.sample(rng.gen());
    let _yaw = cfg.initial.yaw_deg.sample(rng.gen());
    let _g0 = cfg.initial.gimbal_deg.sample(rng.gen());
    let _g1 = cfg.initial.gimbal_deg.sample(rng.gen());
    let _cg = cfg.initial.cg_offset_m.sample(rng.gen());
    let domain = sample_domain(cfg, index);
    let thrust_peak = cfg
        .aero
        .thrust_profile
        .iter()
        .map(|p| p.1)
        .fold(0.0_f64, f64::max);
    Conditioning {
        mass_kg: mass as f32,
        area_m2: cfg.aero.area_m2 as f32,
        cd0: domain.cd0 as f32,
        static_margin_cal: domain.static_margin_cal as f32,
        thrust_peak_n: thrust_peak as f32,
        gyro_noise: domain.gyro_noise as f32,
    }
}
