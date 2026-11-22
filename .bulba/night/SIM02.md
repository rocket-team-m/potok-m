# SIM02 — sensor realism tests (night task)

**Date:** 2026-11-23
**Repo:** `/home/sehaxe/potok-m/sim` (git root `/home/sehaxe/potok-m`)
**Scope:** created exactly one test file, `sim/core/tests/sensor_realism_tests.rs`. No source edits. No subagents. No push.

## What was delivered

One integration-test file with `// SPDX-License-Identifier: GPL-3.0-or-later`, 596 lines, using only the public `potok_sim` API. It drives `Sensor::measure` directly with a controlled on-rail vehicle (zero thrust, static state) so each effect is isolated instead of inferred from a full episode. The "quiet" rig zeroes noise/bias, lag, quant, delay, limits and spikes — including `gyro_noise_range`/`accel_noise_range`, because `Sensor::new` overwrites the raw `gyro_noise`/`accel_noise` from domain randomization.

| Requirement | Test | What it pins |
|---|---|---|
| deterministic seeds | `same_seed_and_episode_reproduce_every_frame_byte` | two rigs with the production sensor config and seed 4242 produce equal `Frame`s, equal 56-byte wire bytes and equal CRCs over 24 frames; seed 4243 and episode 1 (with nonzero bias so the sub-seed has an observable effect) move the output |
| Allan noise | `allan_arw_sets_the_discrete_noise_std` | 600-frame gyro/accel std matches `arw / sqrt(dt)` within 25 %; doubling `gyro_arw` doubles the std (ratio in 1.6–2.4) |
| bias drift | `allan_bias_stability_drifts_while_zero_stability_is_exactly_flat` | zero stability + zero noise reads exactly 0.0 for 600 frames; positive `gyro_bias_stability`/`accel_bias_stability` produces a drifting random walk (std > 0.01 / 0.02, first-100 vs last-100 mean differ) |
| quantization | `quantization_uses_per_channel_steps_and_zero_step_disables_it` | gyro/accel on their own 0.01/0.2 grids, mag/baro on the 0.05 grid; `gyro_quant_step = 0` falls back to `quant_step` and changes the value; all-zero steps leave the raw value unrounded |
| delay | `latency_steps_shifts_gyro_and_mag_by_exactly_the_configured_frames` | `latency_steps = 3` shifts gyro by exactly 3 frames (step change at k=4 appears at k=7) and shifts the magnetometer's rotated world vector identically; `latency_steps = 0` returns the current sample |
| saturation | `saturation_clamps_each_channel_and_raises_fault` | gyro/accel/mag/baro clamped to their limits and `saturated` set; `pitot_sat` raises the flag; FAULT high, AI_READY low, `ai_input_valid()` false; a nominal-limit rig on the same static state is clean (per-frame, not sticky) |
| fault flags | `wire_flags_and_gating_cover_healthy_and_fault_cases` | exact flag byte for healthy / sensor-bad / no-pitot / saturated / Boost / Pad, plus `gating` agrees with AI_READY across all 6 phases × 2 × 2 × 2 combinations |
| stuck/high fault | `seeded_spikes_latch_fault_high_while_they_keep_firing` | `spike_probability = 1` keeps FAULT high and AI_READY low on all 20 frames with exactly one gyro axis offset by the configured amplitude; probability 0 never faults; probability 0.5 produces spikes followed by clean frames with FAULT low again |
| tilt gravity projection | `accelerometer_reads_tilted_gravity_and_zero_in_free_fall` | upright reads `[0,0,g]`; ±30° X tilt reads `[0,±g·sinθ,g·cosθ]`; +90° Y tilt reads `[-g,0,0]`; magnitude invariant; free fall at 100 m reads ~0; magnetometer sees the same body-frame rotation |

## Verification (evidence)

- `cargo test -p potok-sim --test sensor_realism_tests` → **9 passed, 0 failed** after the final edit (0.09 s).
- `cargo test -p potok-sim` → full package suite green: 21 test binaries, **0 failed** in every `test result:` line, including the 9 new tests and the pre-existing `sim_tests` (16) and `smart_sim_tests` (6).
- `rustfmt --edition 2021 --check core/tests/sensor_realism_tests.rs` → clean (no diff).
- Every failure during development was in the new test's own expectations and was fixed in the test file only: a wrong sign in the Y-tilt analytic expectation (`R^T` maps world +Z to body −X), a seed-determinism check accidentally using the zeroed-noise rig, too-tight statistical margins, and the assumption that `pitot_sat` clamps the wire value.

## Observation (no source edit made)

`pitot_sat` is a **flag-only** limit: `measure` sets `saturated = true` when the raw Pitot pressure exceeds it, but `pitot_dp` is then clamped to the `i16` range, not to `pitot_sat`. The test asserts the real behaviour explicitly so a future change is visible. Likewise, the saturation/fault bits are per-frame, not sticky latches.

## Limits / not covered

- The statistical gates (ARW std, bias-walk spread) use fixed seeds and generous margins; they are regression pins for those seeds, not a randomized property sweep.
- Measurement is through `Sensor::measure` on a static on-rail vehicle; full-episode interaction with wind, thrust and the dataset path is covered by the pre-existing simulator tests, not repeated here.
- Other agents' in-flight files were present in the worktree; only the test path was staged for the first commit and only this report for the second. No push.

## Git

Local commit `c4e8d2b` — `test(sim): sensor realism gates for Allan noise, faults and tilt (SIM02)`. This report is committed separately on top. No push.
