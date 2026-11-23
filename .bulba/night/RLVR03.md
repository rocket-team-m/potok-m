# RLVR03 — deterministic rollout harness

**Date:** 2026-11-23
**Repo:** `/home/sehaxe/potok-m/sim` (git root `/home/sehaxe/potok-m`)
**Status:** done, committed locally, not pushed.
**Commit:** see `git log -1 --format=%h -- sim/core/src/rollout.rs`.

## Deliverable

Three new files, all `SPDX-License-Identifier: GPL-3.0-or-later`, no existing
file modified:

| file | lines | content |
|---|---:|---|
| `sim/core/src/rollout.rs` | 467 | harness, controller/reward traits, built-in arms, JSONL writer/reader |
| `sim/core/tests/rollout_tests.rs` | 242 | 9 gates on the shipped source (path include + module shims) |
| `sim/cli/src/rollout.rs` | 74 | `RolloutArgs` (clap `Args`) + `run()`, no CLI registration |

No subagents were used. Nothing under `ai/`, `protocol/`, `edge/`,
`.superpowers/` or another lane's untracked files was touched.

## Contract

- **Pluggable controller** — `Controller::act(&mut self, &ControlContext) ->
  (f64, f64)`; blanket impl for `FnMut` closures, plus `ZeroController`,
  `ConstantController`, `OracleController` and a `ControllerKind` parser for
  CLI strings (`zero`, `oracle`, `constant:<g_roll>,<g_pitch>`). A `RefCell`
  adapter lets a stateful policy drive the simulator's `Fn` control seam.
- **Trajectory** — one `TrajectoryStep` per 50 Hz tick: tick, phase code, the
  command actually issued, and the flown world position/velocity from the
  simulator's true state. No learned critic is involved.
- **Reward** — `RewardFn` trait, with a built-in `FlightReward`
  `R = -w_apogee·|apogee-target| - w_drift·lateral_rms - w_effort·∫|u|dt
  - w_fault·(invalid+nan frames)`. Every term is recomputable from the
  stored `reward_parts` and `counters`. This is intentionally separate from
  the RLVR02 hard-gate reward: that module is untracked and unwired in this
  lane, so importing it would have made the harness depend on another
  agent's file.
- **Counters** — simulator counters (ticks, frames, invalid/NaN frames, phase
  histogram, apogee, max speed) copied verbatim, plus controller call count
  and control integral tallied by the recorder.
- **Manifest** — one compact JSON object per seed, fields in fixed struct
  order: `schema_version`, `config_hash`, `seed`, `episode_index`,
  `controller_id`, `trajectory`, `reward`, `reward_parts`, `counters`.
  `config_hash` is the existing sha256 of the effective config's canonical
  TOML, with `run.seed` set to that record's seed. No timestamps, PIDs or
  map iteration: two runs with the same inputs produce byte-identical JSONL.

One episode (index 0) is flown per seed. No training loop, no gradients, no
GPU, no global RNG; all randomness comes from the existing seeded simulator.

## LOUD failures

Empty or duplicated seed set, invalid config, unknown/malformed controller
name, non-finite or negative reward weights, non-finite score and empty tick
record are all `SimError` returns with a cause. A blank or malformed JSONL
line is rejected on read. Nothing is clamped or silently skipped.

## Verification

- `cargo test -p potok-sim --test rollout_tests` → **9 passed, 0 failed** in
  0.28 s (debug): repeat-run equality, per-line hash/seed provenance,
  distinct seeds producing distinct hashes/trajectories, closure and trait
  controllers, oracle actually commanding fins, named-vs-explicit equality,
  JSONL file round-trip, and five malformed-input gates.
- `rustfmt --edition 2021 --check` clean on all three files.
- `cargo check -p potok-cli` clean with `pub mod rollout;` temporarily added
  to both `core/src/lib.rs` and `cli/src/lib.rs`. Those two one-line edits
  were removed immediately afterwards; the final diff touches only the three
  files above. (An unrelated agent's `pub mod firmware_adapter;` line in
  `core/src/lib.rs` was preserved.)

Two initial test failures were fixture mistakes, not code defects: with the
zero controller the AI interlock marks every boost tick invalid (so
`fault_frames` equals the invalid count, not zero), and the default thrust
table burns until ~10 s (so the oracle needs a 12 s window to reach coast).
Both assertions were corrected; no production code was changed to make the
tests pass.

## Not done (out of scope)

- `pub mod rollout;` in `core/src/lib.rs` / `cli/src/lib.rs` and the
  `RolloutArgs` subcommand variant in `cli/src/main.rs` are left to the parent
  lane. Until wired, the core tests reach the module through `#[path]` plus
  six module shims, and `cli/src/rollout.rs` is not compiled by a normal
  workspace build.
- No end-to-end CLI invocation was run, because registering the subcommand
  is outside the three owned files; only a type-check with temporary wiring
  was performed.
- No A/B of controllers: this task builds the instrument, not a verdict.
