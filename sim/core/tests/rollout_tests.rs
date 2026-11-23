// SPDX-License-Identifier: GPL-3.0-or-later
//! RLVR03 — gates for the deterministic rollout harness.
//!
//! `core/src/rollout.rs` is included by path because `lib.rs` is owned by
//! another lane: the shims below mirror the library's module paths for that
//! include, and the one-line `pub mod rollout;` registration exposes the module
//! publicly. The tests exercise the shipped file, not a copy.
//!
//! Coverage: repeat-run byte determinism, per-seed hash/seed provenance in the
//! JSONL, recorder/counter consistency, controller pluggability (closure,
//! built-in and named), and LOUD rejection of malformed inputs.

mod config {
    pub use potok_sim::config::*;
}
mod control {
    pub use potok_sim::control::*;
}
mod episode {
    pub use potok_sim::episode::*;
}
mod error {
    pub use potok_sim::error::*;
}
mod manifest {
    pub use potok_sim::manifest::*;
}
mod optimizer {
    pub use potok_sim::optimizer::*;
}

#[path = "../src/rollout.rs"]
mod rollout;

use potok_sim::{Config, ControlContext};
use rollout::{
    read_jsonl, rollouts_to_jsonl, run_named_rollouts, run_rollouts, write_jsonl, Controller,
    ControllerKind, FlightReward, Rollout, ZeroController, ROLLOUT_SCHEMA_VERSION,
};

fn config_with_duration(duration_s: f64) -> Config {
    let text = potok_sim::config::DEFAULT_TOML
        .replace("duration_s = 30.0", &format!("duration_s = {duration_s}"));
    Config::from_toml_str(&text).expect("short test config must be valid")
}

fn short_config() -> Config {
    config_with_duration(2.0)
}

fn zero_rollouts(cfg: &Config, seeds: &[u64]) -> Vec<Rollout> {
    run_rollouts(cfg, seeds, |_| ZeroController, &FlightReward::default())
        .expect("zero-controller rollouts")
}

#[test]
fn zero_controller_rollouts_are_deterministic_and_well_formed() {
    let cfg = short_config();
    let seeds = [11, 22];
    let a = zero_rollouts(&cfg, &seeds);
    let b = zero_rollouts(&cfg, &seeds);
    assert_eq!(a, b, "same inputs must give identical rollouts");

    assert_eq!(a.len(), seeds.len());
    for (r, &seed) in a.iter().zip(seeds.iter()) {
        assert_eq!(r.schema_version, ROLLOUT_SCHEMA_VERSION);
        assert_eq!(r.seed, seed);
        assert_eq!(r.episode_index, 0);
        assert_eq!(
            r.config_hash.len(),
            64,
            "config hash must be a sha256 hex digest"
        );
        assert_eq!(r.controller_id, "zero");
        assert!(r.trajectory.iter().all(|s| s.command == [0.0, 0.0]));
        assert_eq!(r.trajectory.len(), r.counters.ticks);
        assert_eq!(r.counters.controller_calls, r.counters.ticks);
        assert_eq!(
            r.counters.phase_counts.iter().sum::<usize>(),
            r.counters.ticks
        );
        assert_eq!(r.counters.control_integral, 0.0);
        assert!(r.reward.is_finite());
        assert_eq!(r.reward_parts.control_integral, 0.0);
        assert_eq!(
            r.reward_parts.fault_frames,
            r.counters.invalid_frames + r.counters.nan_frames
        );
    }
}

#[test]
fn jsonl_is_byte_identical_across_runs_and_carries_hash_and_seed() {
    let cfg = short_config();
    let seeds = [3, 5, 8];
    let first = rollouts_to_jsonl(&zero_rollouts(&cfg, &seeds)).unwrap();
    let second = rollouts_to_jsonl(&zero_rollouts(&cfg, &seeds)).unwrap();
    assert_eq!(first, second, "JSONL must be byte-identical across runs");

    let lines: Vec<&str> = first.lines().collect();
    assert_eq!(lines.len(), seeds.len());
    for (line, &seed) in lines.iter().zip(seeds.iter()) {
        let v: serde_json::Value =
            serde_json::from_str(line).expect("each line is one JSON object");
        assert_eq!(v["seed"].as_u64().unwrap(), seed);
        assert_eq!(
            v["schema_version"].as_u64().unwrap(),
            u64::from(ROLLOUT_SCHEMA_VERSION)
        );
        assert_eq!(v["config_hash"].as_str().unwrap().len(), 64);
        assert!(v.get("reward").unwrap().is_number());
        assert!(v.get("trajectory").unwrap().is_array());
    }
}

#[test]
fn different_seeds_change_config_hash_and_trajectory() {
    let cfg = short_config();
    let a = zero_rollouts(&cfg, &[1]).remove(0);
    let b = zero_rollouts(&cfg, &[2]).remove(0);
    assert_ne!(a.config_hash, b.config_hash);
    assert_ne!(a.trajectory, b.trajectory);
    assert_ne!(a.reward_parts.apogee_m, b.reward_parts.apogee_m);
}

#[test]
fn a_closure_is_a_pluggable_controller() {
    let cfg = short_config();
    let policy = |_ctx: &ControlContext<'_>| (1.25, -0.75);
    let rs = run_rollouts(&cfg, &[7], |_| policy, &FlightReward::default()).unwrap();
    assert_eq!(rs.len(), 1);
    assert_eq!(rs[0].controller_id, "custom");
    assert!(rs[0].trajectory.iter().all(|s| s.command == [1.25, -0.75]));
    assert!(rs[0].counters.control_integral > 0.0);
}

#[test]
fn a_custom_controller_trait_records_its_own_id() {
    struct MyPolicy {
        calls: usize,
    }
    impl Controller for MyPolicy {
        fn id(&self) -> &str {
            "my-policy"
        }
        fn act(&mut self, _ctx: &ControlContext<'_>) -> (f64, f64) {
            self.calls += 1;
            let roll = if self.calls % 2 == 0 { 0.5 } else { -0.5 };
            (roll, 0.0)
        }
    }

    let cfg = short_config();
    let rs = run_rollouts(
        &cfg,
        &[9],
        |_| MyPolicy { calls: 0 },
        &FlightReward::default(),
    )
    .unwrap();
    assert_eq!(rs[0].controller_id, "my-policy");
    assert!(rs[0]
        .trajectory
        .iter()
        .enumerate()
        .all(|(i, s)| s.command[0] == if i % 2 == 0 { -0.5 } else { 0.5 }));
}

#[test]
fn oracle_controller_drives_the_fin_seam() {
    // The built-in thrust table burns until ~10 s and the AI interlock is
    // active for the whole boost phase, so score a coast window.
    let cfg = config_with_duration(12.0);
    let rs = run_named_rollouts(
        &cfg,
        &[13],
        ControllerKind::Oracle,
        &FlightReward::default(),
    )
    .unwrap();
    assert_eq!(rs[0].controller_id, "oracle");
    assert!(rs[0]
        .trajectory
        .iter()
        .any(|s| s.command[0] != 0.0 || s.command[1] != 0.0));
    assert!(rs[0].counters.control_integral > 0.0);
    assert!(rs[0].reward.is_finite());
}

#[test]
fn named_and_explicit_controllers_agree() {
    let cfg = short_config();
    let named =
        run_named_rollouts(&cfg, &[4], ControllerKind::Zero, &FlightReward::default()).unwrap();
    let explicit = zero_rollouts(&cfg, &[4]);
    assert_eq!(named, explicit);
}

#[test]
fn jsonl_file_round_trips_exactly() {
    let cfg = short_config();
    let rollouts = zero_rollouts(&cfg, &[1, 2]);
    let dir = std::env::temp_dir().join(format!("potok-rollout-{}-{}", std::process::id(), "file"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("rollouts.jsonl");

    write_jsonl(&path, &rollouts).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(bytes, rollouts_to_jsonl(&rollouts).unwrap().into_bytes());
    assert_eq!(read_jsonl(&path).unwrap(), rollouts);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn malformed_inputs_fail_loudly() {
    let cfg = short_config();

    let err = run_rollouts(&cfg, &[], |_| ZeroController, &FlightReward::default()).unwrap_err();
    assert!(
        format!("{err}").contains("seeds must not be empty"),
        "{err}"
    );

    let err =
        run_rollouts(&cfg, &[6, 6], |_| ZeroController, &FlightReward::default()).unwrap_err();
    assert!(format!("{err}").contains("duplicate seed"), "{err}");

    let err = ControllerKind::parse("telepathy").unwrap_err();
    assert!(format!("{err}").contains("unknown controller"), "{err}");

    let bad = FlightReward {
        target_apogee_m: f64::NAN,
        ..FlightReward::default()
    };
    let err = run_rollouts(&cfg, &[1], |_| ZeroController, &bad).unwrap_err();
    assert!(format!("{err}").contains("target_apogee_m"), "{err}");

    let nan_reward = |_t: &[rollout::TrajectoryStep], _c: &rollout::RolloutCounters| f64::NAN;
    let err = run_rollouts(&cfg, &[1], |_| ZeroController, &nan_reward).unwrap_err();
    assert!(format!("{err}").contains("reward is non-finite"), "{err}");
}
