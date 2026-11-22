# B08 — Sim residual stream integration — implementation report

## Scope

Repo: `/home/sehaxe/potok-m/sim` (cargo workspace root; git root: `/home/sehaxe/potok-m`).
Only the assigned path was written in the source tree:

- `sim/core/tests/residual_stream_tests.rs` (new)

Plus this report at `.superpowers/sdd/rocketcraft-implementation/task-B08-report.md`.
No other file was touched; in particular `sim/core/src/physics.rs` was **not** edited,
and no source module was changed. Own code is covered by the workspace
`license = "GPL-3.0-or-later"`; no third-party code or corpus was copied.

## What the tests pin

The 14-value model frame is `Config::default().output.features` (eight raw
channels: gyro xyz, acc xyz, baro_h, pitot_q) plus the six `PHYSICS_FEATURES`
appended by `--physics-features`. The test file declares this exact order as a
single `STREAM: [&str; 14]` constant and checks it at each seam that could
silently reorder it:

| Test | What it checks |
|---|---|
| `canonical_stream_is_default_telemetry_then_physics` | Default config = first 8 names; physics = last 6; appended config = exact 14; `sensor::FEATURE_NAMES` preserves the relative order (aliases `mag_*`/`pitot_dp` may be interposed) and ends with the six physics names; residual column constant = 11. |
| `canonical_matrix_columns_match_named_feature_lookup` | On a real 14-feature episode, every row-major column equals `sensor::feature(frame, name)` for that position, and columns 8..14 equal `frame.physics[0..6]`. |
| `configured_order_drives_columns_not_the_lookup_table_order` | With a deliberately permuted 4-feature config, matrix columns follow config order, proving order is config-driven rather than hardcoded. |
| `validation_header_pins_stream_width` | Validation header `feature_count` = 14 and every read-back episode has 14 columns. |
| `crosswind_opens_lateral_residual_at_stream_column_eleven` | With a frozen calm measurement, a 20 m/s crosswind opens `a_lat_resid` (>1.0) and sideslip (>5°), calm stays <1e-3, and `PhysicsFeatures::as_array()[3]` is that same value (the slot that lands in column 11). |
| `gust_frames_carry_residual_in_stream_column_eleven` | On a real coast-phase gust episode, flagged frames carry a finite residual in column 11, equal to `frame.physics[3]`, larger than calm coast frames and above the 0.01 floor. |
| `stream_is_bit_identical_for_the_same_seed` | Same seed, same episode → bit-identical feature vectors. |

The gust tests deliberately reuse the existing modules (`PhysicsFeatures`,
`Sensor::feature`, `Simulator`) rather than duplicating physics. Existing
`core/tests/residual_tests.rs` was left untouched; this file adds the explicit
stream-column/order pin that it does not carry.

## Verification

| Command | Result |
|---|---|
| `cargo test -p potok-sim --test residual_stream_tests` | 7 passed, 0 failed |
| `cargo test -p potok-sim` | 73 passed, 0 failed, 0 ignored across 12 binaries |
| `rustfmt --edition 2021 --check core/tests/residual_stream_tests.rs` | clean |
| `git diff --check -- sim/core/tests/residual_stream_tests.rs` | clean |

No GPU work was performed; this is a CPU-only simulator test target.

## Commit

Local commit only, nothing pushed. Subject: `test(sim): pin 14-feature stream order and gust residual slot`.
Unrelated work in `ai/out/` and other agents' in-progress files was left untouched and unstaged.
