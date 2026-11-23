# SIM06 — extended OpenRocket scalar parity (2026-11-23)

**Scope.** Created exactly one file: `sim/core/tests/or_parity_extended_tests.rs`
(SPDX-License-Identifier: GPL-3.0-or-later). No source, config, coefficient or
fixture edit. No push.

**Evidence (read-only).** `sim/tests/fixtures/or_sim9.csv`, sampled from
`/home/sehaxe/Downloads/Telegram Desktop/гугугага.ork` (ZIP → `rocket.ork`,
`<simulations><simulation><name>Simulation 9</name>`, databranch
`Маршевая ступень`; OR `optimumAltitude = 3127.362628772149`,
`timeToOptimumAltitude = 26.9`). 55 rows: 54 on the 0.5 s grid plus the exact
26.9 s apogee row.

**What it checks.** Four tests:

1. `or_fixture_scalar_anchors` — parses the CSV, asserts 55 strictly rising
   rows, derives apogee (last row), vmax (max over all rows) and apogee time
   from the fixture, and cross-checks them against pinned constants.
2. `episode0_apogee_matches_or_fixture`
3. `episode0_vmax_matches_or_fixture`
4. `episode0_time_to_apogee_matches_or_fixture`

Flight is the existing oracle's flight: `Config::default()`, episode 0,
seed 1337. Per-instant speed is **not** compared — `Frame` has no speed field
(`pitot_dp` is a noisy saturating i16), so speed parity uses only
`Episode::max_speed_mps`. Time-to-apogee comes from the un-noised 50 Hz
`Episode::trajectory`: index of maximum `pos[2]` divided by
`run.frame_rate_hz`, with a cross-check that the same frame's `time_us` agrees
and that the tick altitude is the reported step-sampled apogee.

| metric | OR fixture | sim episode 0 | measured error | threshold | result |
|---|---|---|---|---|---|
| apogee | 3127.363 m | 3185.378 m | 1.86 % rel | 2 % rel | pass |
| vmax | 274.278 m/s | 278.251 m/s | 1.45 % rel | 2 % rel | pass |
| time-to-apogee | 26.9 s | 27.04 s (tick 1352) | 0.14 s abs | 0.25 s abs | pass |

**Threshold rationale (documented in the file header, not fitted).** The 2 %
limits are inherited from `sim/tests/or_parity_tests.rs::episode0_parity_tolerances`
so the new file extends but does not weaken the established oracle. The
0.25 s time limit is ~1.8× the measured 0.14 s bias, 2.5× OR's own 0.1 s
reporting granularity and 12.5× our 0.02 s trajectory tick (0.93 % of the
26.9 s baseline). Absolute rather than relative: a relative gate on a 26.9 s
baseline would hide a drift of the same physical size. These are regression
gates on the current model, not fidelity claims — the sim apogee is 58 m above
OR and vmax 4 m/s above, well inside the inherited envelope.

**No coefficient tweaking.** The only changed path is the new test file;
`config/*.toml`, the OR fixture and every `core/src` file are untouched. The
thresholds were chosen *after* measuring the unmodified model and are recorded
as such.

**Verification.**
- `rustfmt --edition 2021 --check core/tests/or_parity_extended_tests.rs` — clean.
- `cargo test --offline -p potok-sim --test or_parity_extended_tests` —
  `4 passed; 0 failed`.
- `cargo test --offline --workspace --no-fail-fast` — exit 0; all 26 test
  binaries `ok`, 154 tests passed, 0 failed (includes the pre-existing
  `or_parity_tests.rs`, 4 passed).

An earlier workspace run (without `--no-fail-fast`) hit a transient failure in
`core/tests/control_seam_hard_tests.rs`, an untracked test file another agent
was editing concurrently (created after this file). It was not touched here
and passed in the final full run above.

**Commit.** Local commit only; no push.
