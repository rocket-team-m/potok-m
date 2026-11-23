// SPDX-License-Identifier: GPL-3.0-or-later
//! CLI front-end for the deterministic RLVR rollout harness.
//!
//! This module holds only argument parsing and IO: the simulation, the reward
//! and the JSONL writer live in `potok_sim::rollout`. Wiring it as a subcommand
//! is a one-line change in `main.rs`, owned by the parent lane.

use std::path::PathBuf;

use clap::Args;
use potok_sim::{
    rollout::{run_named_rollouts, write_jsonl, ControllerKind, FlightReward},
    Config, Result,
};

/// `rollout` subcommand: run N consecutive seeds and emit a JSONL manifest.
#[derive(Debug, Clone, Args)]
pub struct RolloutArgs {
    /// TOML config; built-in defaults are used when omitted.
    #[arg(long, value_name = "PATH")]
    pub config: Option<PathBuf>,
    /// First seed, inclusive.
    #[arg(long, default_value_t = 1, value_name = "N")]
    pub seed: u64,
    /// Number of consecutive seeds to roll out.
    #[arg(long, default_value_t = 4, value_name = "N")]
    pub seeds: usize,
    /// Controller: `zero`, `oracle` or `constant:<g_roll>,<g_pitch>` (m/s^2).
    #[arg(long, default_value = "zero", value_name = "KIND")]
    pub controller: String,
    /// JSONL manifest path (one record per seed).
    #[arg(long, value_name = "PATH")]
    pub out: PathBuf,
    /// Target apogee of the verifiable reward, metres.
    #[arg(long, default_value_t = 3000.0, value_name = "M")]
    pub target_apogee_m: f64,
}

/// Run the subcommand. Every failure is LOUD and comes from the harness's
/// typed errors; nothing is clamped or skipped.
pub fn run(args: RolloutArgs) -> Result<()> {
    if args.seeds == 0 {
        return Err(potok_sim::SimError::Config(
            "rollout: --seeds must be at least 1".into(),
        ));
    }
    let base = match &args.config {
        Some(path) => Config::from_file(path)?,
        None => Config::default(),
    };
    let kind = ControllerKind::parse(&args.controller)?;
    let reward = FlightReward {
        target_apogee_m: args.target_apogee_m,
        ..FlightReward::default()
    };
    let seeds: Vec<u64> = (args.seed..args.seed + args.seeds as u64).collect();

    let rollouts = run_named_rollouts(&base, &seeds, kind, &reward)?;
    write_jsonl(&args.out, &rollouts)?;

    let best = rollouts
        .iter()
        .map(|r| r.reward)
        .fold(f64::NEG_INFINITY, f64::max);
    eprintln!(
        "[potok-rollout] seeds={} controller={} config_hash={} best_reward={} jsonl={}",
        rollouts.len(),
        rollouts[0].controller_id,
        rollouts[0].config_hash,
        best,
        args.out.display()
    );
    Ok(())
}
