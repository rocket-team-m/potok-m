# SIM05 — determinism hard tests (night task)

**Date:** 2026-11-23
**Repo:** `/home/sehaxe/potok-m/sim` (git root `/home/sehaxe/potok-m`)
**Scope:** created exactly one test file, `sim/core/tests/determinism_hard_tests.rs`. No source edits. No subagents. No push.

## What was delivered

One integration-test file with `// SPDX-License-Identifier: GPL-3.0-or-later`, 454 lines after `rustfmt`. It uses the public `potok_sim` API only and writes solely into per-test temp dirs (`potok-sim05-*`), removed by the sidecar test.

| Requirement | Test | What it pins |
|---|---|---|
| bit-identical regeneration | `same_seed_config_regenerates_every_field_bit_identically` | 4 seeds × 3 episode indices × 2 independently constructed `Simulator`s; fingerprint uses `f32/f64::to_bits` over every meta field (including conditioning), the canonical 56-byte wire frames plus sim-only frame context, every target field, all features, every v3 truth tick, phase counts, invalid/NaN counters, apogee/max speed and the truth record; also `episodes()` iterator vs direct calls |
| substream separation | `substream_keys_are_seed_index_and_domain_separated` | the four SplitMix64 keys `INIT`/`DOMA`/`WIND`/`SENS` are pairwise distinct for 5 seeds × 5 indices; no seed collision and no collision over 64 indices within any one substream |
| different seeds, every substream | `different_seeds_move_every_observable_substream` | `INIT` → lift-off mass; `DOMA` → `sample_domain().cd0` and exported conditioning; `WIND` → gust onset and turbulence field; `SENS` → gyro bytes with every config envelope frozen and wind noise off, with meta proven unchanged so the movement is attributable to `SENS` alone |
| no global RNG | `global_rng_state_and_call_order_cannot_change_episodes` | draining 10 000 `u64` + `f64` draws from `thread_rng` before regeneration, forward vs reverse call order, an interleaved different-seed simulator, and rayon `run_episodes_parallel` vs serial — all bit-identical |
| sidecar byte identity | `regenerated_sidecars_are_byte_identical` | `validation.bin` plus v1/v2/v3 truth sidecars from two independent regenerations compared bytewise and by sha256; `check_determinism` gate reports `4 files compared, mismatches=[]`; negative control proves seed+1 changes the v3 sidecar bytes |

## Verification (evidence)

- `cargo test -p potok-sim --test determinism_hard_tests` → **5 passed, 0 failed** (19.79 s first green run; 27.88 s after formatting, under contention from other night tasks).
- `rustfmt --edition 2021 --check sim/core/tests/determinism_hard_tests.rs` → clean; the file was formatted with `rustfmt --edition 2021` before the check.
- `cargo test -p potok-sim --test sim_tests` → 16 passed, 0 failed (pre-existing baseline).
- `cargo test --workspace` → final run green through all test binaries and both doc-test targets. An earlier run failed in `core/tests/control_seam_hard_tests.rs:34` (`linearity: got -251.04, want -444.44`): that file was untracked, concurrently written by another night task, and was fixed by its owner before the final run. No source file was edited for it here.
- The one red result in this file's own history was an inverted assertion in the `SENS` check (asserting byte equality where a difference was the claim). It was corrected in the test file only; the final run is green.

## Limits / not covered

- Cross-process identity uses two independently constructed `Simulator`s in one process, not a spawned child process; OS-level process independence is not asserted here.
- Seed pairs are fixed and verified (101/102, 4242/4243, plus 0/1/7/1337 and 9/11/13), not a randomized property sweep.
- Other agents' untracked/in-flight test files were present in the worktree during verification. Only the two SIM05 paths were staged.

## Git

Local commit `cc7bfb8` — `test(sim): hard determinism gates for seeded regeneration (SIM05)`. This report is committed separately on top. No push.
