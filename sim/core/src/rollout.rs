// SPDX-License-Identifier: GPL-3.0-or-later
//! Deterministic RLVR rollout harness.
//!
//! Runs one episode per seed through the existing `Simulator` control seam,
//! records the flown trajectory together with the command at every 50 Hz tick,
//! scores it with a pluggable verifiable reward, and emits a JSONL manifest in
//! which every line carries the effective config hash and the seed that
//! produced it.
//!
//! There is no training loop, no global RNG and no GPU: a rollout is a pure
//! function of (config, seed, controller, reward). Re-running the same inputs
//! produces byte-identical JSONL. All degradation is LOUD: an empty or
//! duplicated seed set, a non-finite reward, a malformed controller name and
//! an invalid config are errors, never fallbacks.

use std::{
    cell::RefCell,
    collections::BTreeSet,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
};

use serde::{Deserialize, Serialize};

use crate::{
    config::Config,
    control::validate_command,
    episode::{ControlContext, Episode, Simulator},
    error::{Result, SimError},
    manifest::config_hash,
    optimizer::ReferenceOptimizer,
};

/// Schema version written into every JSONL record.
pub const ROLLOUT_SCHEMA_VERSION: u32 = 1;

/// One recorded 50 Hz tick: what the controller commanded and where the
/// vehicle actually went. All numbers come from the simulator's control seam
/// and flown truth, so a reward built on them is verifiable without a critic.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TrajectoryStep {
    pub tick: u32,
    pub phase: u8,
    /// Command handed to the fin seam: `[g_roll, g_pitch]` in m/s^2.
    pub command: [f64; 2],
    /// Flown world position `[x, y, z]` in metres.
    pub pos_m: [f32; 3],
    /// Flown world velocity `[vx, vy, vz]` in m/s.
    pub vel_mps: [f32; 3],
}

/// Harness counters. Simulator counters are copied verbatim; the controller
/// and effort counters are tallied by the recorder so an arm that skipped a
/// tick is visible in the manifest.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RolloutCounters {
    pub ticks: usize,
    pub controller_calls: usize,
    pub frames: usize,
    pub invalid_frames: usize,
    pub nan_frames: usize,
    pub phase_counts: [usize; 6],
    pub apogee_m: f64,
    pub max_speed_mps: f64,
    /// Integral of command magnitude over the episode, in m/s.
    pub control_integral: f64,
}

/// Auditable inputs of the verifiable reward. Stored next to the score so a
/// reader can recompute it without rerunning the simulator.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RewardParts {
    pub apogee_m: f64,
    pub lateral_rms_m: f64,
    pub control_integral: f64,
    pub fault_frames: usize,
}

/// One JSONL record: one seed, one episode, one score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Rollout {
    pub schema_version: u32,
    /// SHA-256 of the canonical TOML of the effective config (seed included).
    pub config_hash: String,
    pub seed: u64,
    pub episode_index: u64,
    pub controller_id: String,
    pub trajectory: Vec<TrajectoryStep>,
    pub reward: f64,
    pub reward_parts: RewardParts,
    pub counters: RolloutCounters,
}

/// Pluggable per-tick controller. Implementors may keep hidden state; the
/// harness records every command and the true state at the moment of the call.
pub trait Controller {
    fn id(&self) -> &str {
        "custom"
    }
    fn act(&mut self, ctx: &ControlContext<'_>) -> (f64, f64);
}

/// Closures and plain functions are controllers too.
impl<F> Controller for F
where
    F: for<'a> FnMut(&ControlContext<'a>) -> (f64, f64),
{
    fn act(&mut self, ctx: &ControlContext<'_>) -> (f64, f64) {
        self(ctx)
    }
}

/// No command: the historical uncontrolled trajectory.
#[derive(Debug, Clone, Copy, Default)]
pub struct ZeroController;

impl Controller for ZeroController {
    fn id(&self) -> &str {
        "zero"
    }
    fn act(&mut self, _ctx: &ControlContext<'_>) -> (f64, f64) {
        (0.0, 0.0)
    }
}

/// A fixed command for every tick; useful as an A/B arm and as a gate fixture.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConstantController(pub f64, pub f64);

impl Controller for ConstantController {
    fn id(&self) -> &str {
        "constant"
    }
    fn act(&mut self, _ctx: &ControlContext<'_>) -> (f64, f64) {
        (self.0, self.1)
    }
}

/// Privileged physical controller: the reference optimizer's fin demand.
pub struct OracleController {
    opt: ReferenceOptimizer,
}

impl OracleController {
    pub fn new(cfg: &Config) -> Self {
        Self {
            opt: ReferenceOptimizer::new(cfg),
        }
    }
}

impl Controller for OracleController {
    fn id(&self) -> &str {
        "oracle"
    }
    fn act(&mut self, ctx: &ControlContext<'_>) -> (f64, f64) {
        let g = self
            .opt
            .fin_accel_command(ctx.state, ctx.wind, ctx.vehicle, ctx.phase, ctx.frame);
        (g[0], g[1])
    }
}

/// Named controllers selectable from the CLI. Kept as data so one call site
/// builds every arm without a registry or a dynamic library.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ControllerKind {
    Zero,
    Oracle,
    Constant(f64, f64),
}

impl ControllerKind {
    pub fn parse(s: &str) -> Result<Self> {
        match s.trim() {
            "zero" => Ok(Self::Zero),
            "oracle" => Ok(Self::Oracle),
            rest if rest.starts_with("constant:") => {
                let fields: Vec<&str> = rest["constant:".len()..].split(',').collect();
                if fields.len() != 2 {
                    return Err(SimError::Config(format!(
                        "rollout: controller {s:?} needs constant:<roll>,<pitch>"
                    )));
                }
                let roll: f64 = fields[0].trim().parse().map_err(|e| {
                    SimError::Config(format!("rollout: bad constant roll {:?}: {e}", fields[0]))
                })?;
                let pitch: f64 = fields[1].trim().parse().map_err(|e| {
                    SimError::Config(format!("rollout: bad constant pitch {:?}: {e}", fields[1]))
                })?;
                validate_command(roll, pitch)?;
                Ok(Self::Constant(roll, pitch))
            }
            other => Err(SimError::Config(format!(
                "rollout: unknown controller {other:?}; expected zero, oracle or constant:<roll>,<pitch>"
            ))),
        }
    }
}

/// Pluggable verifiable reward. It sees only the recorded trajectory and the
/// counters, never a learned critic or a device tensor.
pub trait RewardFn {
    fn validate(&self) -> Result<()> {
        Ok(())
    }
    fn score(&self, trajectory: &[TrajectoryStep], counters: &RolloutCounters) -> f64;
}

impl<F> RewardFn for F
where
    F: Fn(&[TrajectoryStep], &RolloutCounters) -> f64,
{
    fn score(&self, trajectory: &[TrajectoryStep], counters: &RolloutCounters) -> f64 {
        self(trajectory, counters)
    }
}

/// Objective score from flown truth: higher is better, never clipped.
///
/// `R = -w_apogee·|apogee - target| - w_drift·lateral_rms
///      - w_effort·control_integral - w_fault·(invalid + nan frames)`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlightReward {
    pub target_apogee_m: f64,
    pub apogee_weight: f64,
    pub drift_weight: f64,
    pub effort_weight: f64,
    pub fault_weight: f64,
}

impl Default for FlightReward {
    fn default() -> Self {
        Self {
            target_apogee_m: 3000.0,
            apogee_weight: 1.0,
            drift_weight: 0.01,
            effort_weight: 0.001,
            fault_weight: 1.0,
        }
    }
}

impl RewardFn for FlightReward {
    fn validate(&self) -> Result<()> {
        if !self.target_apogee_m.is_finite() || self.target_apogee_m <= 0.0 {
            return Err(SimError::Config(format!(
                "rollout: target_apogee_m = {} must be positive and finite",
                self.target_apogee_m
            )));
        }
        for (name, w) in [
            ("apogee_weight", self.apogee_weight),
            ("drift_weight", self.drift_weight),
            ("effort_weight", self.effort_weight),
            ("fault_weight", self.fault_weight),
        ] {
            if !w.is_finite() || w < 0.0 {
                return Err(SimError::Config(format!(
                    "rollout: {name} = {w} must be finite and non-negative"
                )));
            }
        }
        Ok(())
    }

    fn score(&self, trajectory: &[TrajectoryStep], counters: &RolloutCounters) -> f64 {
        -self.apogee_weight * (counters.apogee_m - self.target_apogee_m).abs()
            - self.drift_weight * lateral_rms(trajectory)
            - self.effort_weight * counters.control_integral
            - self.fault_weight * f64::from((counters.invalid_frames + counters.nan_frames) as u32)
    }
}

/// Recorder that wraps a controller and captures the true state it saw. A
/// `RefCell` adapter is used at the call site because the simulator's control
/// seam takes `Fn`, while a policy may legitimately carry hidden state.
struct Recorder<C> {
    inner: C,
    steps: Vec<TrajectoryStep>,
}

impl<C: Controller> Recorder<C> {
    fn record(&mut self, ctx: &ControlContext<'_>) -> (f64, f64) {
        let command = self.inner.act(ctx);
        self.steps.push(TrajectoryStep {
            tick: self.steps.len() as u32,
            phase: ctx.phase.code(),
            command: [command.0, command.1],
            pos_m: cast3(ctx.state.pos()),
            vel_mps: cast3(ctx.state.vel()),
        });
        command
    }
}

fn cast3(v: [f64; 3]) -> [f32; 3] {
    [v[0] as f32, v[1] as f32, v[2] as f32]
}

fn lateral_rms(steps: &[TrajectoryStep]) -> f64 {
    let mut sum = 0.0;
    for s in steps {
        let x = f64::from(s.pos_m[0]);
        let y = f64::from(s.pos_m[1]);
        sum += x * x + y * y;
    }
    (sum / steps.len() as f64).sqrt()
}

fn checked_seeds(seeds: &[u64]) -> Result<&[u64]> {
    if seeds.is_empty() {
        return Err(SimError::Config("rollout: seeds must not be empty".into()));
    }
    let mut seen = BTreeSet::new();
    for &s in seeds {
        if !seen.insert(s) {
            return Err(SimError::Config(format!("rollout: duplicate seed {s}")));
        }
    }
    Ok(seeds)
}

fn counters_from(episode: &Episode, steps: &[TrajectoryStep], dt: f64) -> RolloutCounters {
    RolloutCounters {
        ticks: episode.frames_generated,
        controller_calls: steps.len(),
        frames: episode.frames.len(),
        invalid_frames: episode.invalid_frames,
        nan_frames: episode.nan_frames,
        phase_counts: episode.phase_counts,
        apogee_m: episode.apogee_m,
        max_speed_mps: episode.max_speed_mps,
        control_integral: steps
            .iter()
            .map(|s| s.command[0].hypot(s.command[1]))
            .sum::<f64>()
            * dt,
    }
}

/// Run one episode per seed. Each seed becomes `cfg.run.seed` of an otherwise
/// identical config, so the config hash is a function of the seed and nothing
/// else varies between arms.
pub fn run_rollouts<C, F, R>(
    base: &Config,
    seeds: &[u64],
    make_controller: F,
    reward: &R,
) -> Result<Vec<Rollout>>
where
    C: Controller,
    F: Fn(u64) -> C,
    R: RewardFn + ?Sized,
{
    reward.validate()?;
    let seeds = checked_seeds(seeds)?;
    let dt = base.run.dt;
    let mut out = Vec::with_capacity(seeds.len());

    for &seed in seeds {
        let mut cfg = base.clone();
        cfg.run.seed = seed;
        cfg.validate()?;
        let hash = config_hash(&cfg)?;
        let sim = Simulator::new(cfg)?;

        let mut rec = Recorder {
            inner: make_controller(seed),
            steps: Vec::new(),
        };
        let controller_id = rec.inner.id().to_string();
        let episode = {
            let cell = RefCell::new(&mut rec);
            sim.run_episode_with_control_ctx(0, &|ctx| cell.borrow_mut().record(ctx))?
        };
        let steps = std::mem::take(&mut rec.steps);
        if steps.is_empty() {
            return Err(SimError::Dataset(format!(
                "rollout seed {seed}: no ticks were recorded"
            )));
        }

        let counters = counters_from(&episode, &steps, dt);
        let parts = RewardParts {
            apogee_m: counters.apogee_m,
            lateral_rms_m: lateral_rms(&steps),
            control_integral: counters.control_integral,
            fault_frames: counters.invalid_frames + counters.nan_frames,
        };
        let score = reward.score(&steps, &counters);
        if !score.is_finite() {
            return Err(SimError::Dataset(format!(
                "rollout seed {seed}: reward is non-finite ({score})"
            )));
        }

        out.push(Rollout {
            schema_version: ROLLOUT_SCHEMA_VERSION,
            config_hash: hash,
            seed,
            episode_index: 0,
            controller_id,
            trajectory: steps,
            reward: score,
            reward_parts: parts,
            counters,
        });
    }
    Ok(out)
}

/// Convenience entry point for the CLI: build one of the named controllers
/// for every seed and score it with the built-in verifiable reward.
pub fn run_named_rollouts(
    base: &Config,
    seeds: &[u64],
    kind: ControllerKind,
    reward: &FlightReward,
) -> Result<Vec<Rollout>> {
    match kind {
        ControllerKind::Zero => run_rollouts(base, seeds, |_| ZeroController, reward),
        ControllerKind::Oracle => {
            run_rollouts(base, seeds, |_| OracleController::new(base), reward)
        }
        ControllerKind::Constant(roll, pitch) => {
            run_rollouts(base, seeds, |_| ConstantController(roll, pitch), reward)
        }
    }
}

/// Deterministic JSONL rendering: one compact JSON object per line, struct
/// field order only, no timestamps, no map iteration, no pretty printing.
pub fn rollouts_to_jsonl(rollouts: &[Rollout]) -> Result<String> {
    let mut text = String::new();
    for r in rollouts {
        text.push_str(&serde_json::to_string(r).map_err(|e| {
            SimError::Dataset(format!("rollout seed {}: JSON serialization: {e}", r.seed))
        })?);
        text.push('\n');
    }
    Ok(text)
}

/// Write the JSONL manifest. A malformed record fails before the file is
/// truncated, so a bad run cannot leave a half-written manifest behind.
pub fn write_jsonl(path: &Path, rollouts: &[Rollout]) -> Result<()> {
    let text = rollouts_to_jsonl(rollouts)?;
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(text.as_bytes())?;
    w.flush()?;
    Ok(())
}

/// Read a JSONL manifest back. Empty or malformed lines are LOUD errors.
pub fn read_jsonl(path: &Path) -> Result<Vec<Rollout>> {
    let file = File::open(path)?;
    let mut out = Vec::new();
    for (i, line) in BufReader::new(file).lines().enumerate() {
        let line = line?;
        let r: Rollout = serde_json::from_str(&line)
            .map_err(|e| SimError::Dataset(format!("rollout JSONL line {}: {e}", i + 1)))?;
        out.push(r);
    }
    Ok(out)
}
