# LINT05 — night-test lint/fmt cleanup (2026-11-23)

**Repo:** `/home/sehaxe/potok-m/sim` (git root `/home/sehaxe/potok-m`)
**Scope:** only the 8 assigned `sim/core/tests/*_tests.rs` files. No source edits. No subagents. No push.
License: `SPDX-License-Identifier: GPL-3.0-or-later` (unchanged on every file).

## Starting state

At handoff all 8 targets already compiled and the full suite was green — the compile
errors named in the older SIM03/SIM04 reports had been fixed by other night agents
(`18cead8`, `cc7bfb8`, `73da928`, …). No compile or test errors remained to fix.
The debt was lint/formatting: **7 clippy warnings in 4 of the 8 files** and
**rustfmt diffs in 1 file**. No gate was weakened; every change is an equivalent
rewrite of the same assertion.

## Fixes

| File | Warning | Fix (gate preserved) |
|---|---|---|
| `manifest_hardening_tests.rs:363,374` | `iter_cloned_collect` ×2 | `TARGET_NAMES.iter().copied().collect::<Vec<_>>()` → `TARGET_NAMES.to_vec()` (same for `TRUTH_V3_TICK_NAMES`); the `assert_eq!` compares the identical `Vec<&str>` |
| `sensor_realism_tests.rs:333,511,571` | `needless_range_loop` ×3 | `for i in 0..3` + indexing → `for (i, (got, want)) in …iter().zip(…).enumerate()`; same f64 comparisons and same axis-labelled failure messages |
| `determinism_hard_tests.rs:273` | `identity_op` | `episode_seed(seed, 0 ^ tag)` → `episode_seed(seed, tag)` — `0 ^ tag` is bit-identical to `tag`; the seed/index collision gates are untouched |
| `control_seam_hard_tests.rs:178` | `assertions_on_constants` | `assert!(MAX_COMMAND_MPS2 > 20.0 * MAX_FIN_DEFLECTION_RAD)` → `const { assert!(…) }` — same condition, now enforced at compile time rather than silently deleted |
| `sensor_realism_tests.rs` (8 hunks) | rustfmt | `rustfmt --edition 2021` applied; line wrapping only |

The other 4 assigned files (`aero_hardening_tests.rs`, `wind_scenario_tests.rs`,
`or_parity_extended_tests.rs`, `trajectory_v3_hard_tests.rs`) were already clean —
no clippy warning, no fmt diff, no edit needed.

## Verification (evidence)

- `cargo clippy -p potok-sim --tests` before: warnings at all 7 lines above.
  After: **zero warnings in all 8 assigned files**. Remaining warnings are in
  out-of-scope files only (`core/src/{config,dynamics,episode,quality,sensor}.rs`,
  `dataset_tests.rs`, `residual_tests.rs`, `residual_stream_tests.rs`) — untouched.
- `rustfmt --edition 2021 --check` on all 8 files → clean.
- `cargo test -p potok-sim` after the edits → **exit 0; 141 passed, 0 failed**
  across 19 integration-test binaries (plus lib and doc-tests, 0 each).
  Full per-binary result log: `/tmp/opencode/lint05_test1.log`.
  Targets of this task: aero_hardening 10, sensor_realism 9, wind_scenario 12,
  manifest_hardening 8, determinism_hard 5, or_parity_extended 4,
  control_seam_hard 7, trajectory_v3_hard 6 — all pass.

**Concurrency note:** three of the edited files landed through other night agents'
commits while these fixes were in flight: `determinism_hard_tests.rs` (`cc7bfb8`),
`control_seam_hard_tests.rs` (`20f4d8d`), `sensor_realism_tests.rs` (`c4e8d2b`).
`git show` on each commit confirms it carries the LINT05 line (the `0 ^ tag` → `tag`
reduction, the `const { assert!(…) }`, and the three zip/enumerate loops). No edit
was lost; those three files were clean by the time LINT05 staged its own commit.

## Git

Local commit only, no push. LINT05's own commit stages exactly the one file still
outstanding for this task (`manifest_hardening_tests.rs`) plus this report; no
`git add .`. The other three fixes are carried in the agents' commits named above.
