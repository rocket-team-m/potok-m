# CHUTE02 — split the single lateral RMS into deployment and landing metrics

**Date:** 2026-11-23. **Scope:** only `sim/core/src/validation.rs`,
`sim/core/src/trajectory.rs`, `sim/core/tests/lateral_metrics_tests.rs` (new).
GPL-3.0-or-later. No subagents, no push.

## 1. Verdict

The old `lateral_rms_m` is a single time-RMS over the whole flown trace; it
silently mixes AI-influenceable fin flight with (future) canopy drift. It is
now explicitly accompanied by two metrics with different authority:

| metric | window | authority | truth source |
|---|---|---|---|
| `lateral_rms_m` (legacy) | every flown tick | mixed | flown trace only; unchanged |
| `lateral_rms_at_deployment_m` | ticks `0..=deployment_tick` | fin-active (AI) | flown trace + `deployment_tick` / `boost_end_tick` |
| `landing_dispersion_m` | deployment → landing | wind (not AI) | flown position at deployment + sidecar `landing_xy` |

`LateralFlightContributions` splits the deployment RMS² additively at
`boost_end_tick`, so the AI-forbidden powered segment and the fin-active
segment are pinned separately: `boost_m2 + fin_active_m2 == at_deployment_m()²`.

## 2. Wire contract

v1/v2/v3 of `validation.bin` and v1/v2/v3 of `validation.trajectory` are
**byte-identical** to before — no header, stride, meta count or target count
was touched. The new truth lives only in a separate optional sidecar:

- path `validation.bin` → `validation.deployment`; magic `POTOKDPL`, version 1;
  64-byte header + 32-byte typed records (`index:u64`, `boost_end_tick:u32`,
  `deployment_tick:u32`, `landing_xy:[f32;2]`, `wind_mean_xy:[f32;2]`).
- `read_deployment_truth` rejects bad magic, unknown version, a byte count that
  does not match `64 + 32·n`, a reversed boost/deployment gate and non-finite
  truth — all LOUD, never zero-filled.
- Neither split metric guesses a deployment tick from flown state: the
  simulator has no canopy (LATERAL01 §0), so passing `None` truth is a hard
  error that names `validation::read_deployment_truth` /
  `write_deployment_truth` as the escape. Out-of-range deployment ticks are
  likewise loud.

## 3. Gates

`cargo test -p potok-sim --test lateral_metrics_tests`: **5 passed, 0 failed.**
The gates pin: (a) the LATERAL01 baseline decomposition — boost **1.89 %** of
fin-flight RMS², fin-active **98.11 %**, deployment RMS ≈ **1528.37 m**, with
explicit `<5 %` / `>95 %` bounds; (b) the deployment metric is a strict prefix
of the legacy metric; (c) fin-flight changes move the deployment metric but
leave landing dispersion bit-identical; (d) missing/malformed truth and
out-of-range ticks fail loudly; (e) the new sidecar round-trips with its exact
byte length and rejects a bumped version.

`cargo test --workspace --no-fail-fast`: all crates and all sim test binaries
pass except **three pre-existing failures in files outside this task's scope**:
`residual_stream_tests.rs:189` (`crosswind_opens_lateral_residual_at_stream_column_eleven`,
"calm residual: 15.876"), `residual_tests.rs:145`
(`crosswind_gust_raises_lateral_residual`, same calm value) and
`residual_tests.rs:249` (`gust_episode_carries_residual_in_the_feature_path`,
gust mean 0.026886 ≤ calm 0.027060). Those files, `config/default.toml` and
`core/src/aero.rs` are dirty from other agents' uncommitted work; none of the
new code is called by any existing test binary, so this change cannot produce
those readings. A separate earlier run also showed a one-off
`manifest_hash_tests.rs:319` golden-pin mismatch that did **not** reproduce on
re-run. Not fixed here: the assigned file scope forbids it.

## 4. Skipped / follow-ups

- No canopy model exists, so no shipped split has a `.deployment` sidecar; the
  split metrics are intentionally unavailable until recovery logic or an
  annotator writes it. This is the honest boundary, not a missing default.
- `lib.rs` is out of scope: new items are imported as
  `potok_sim::trajectory::…` / `potok_sim::validation::…`, not crate-root
  re-exports. Add re-exports in a follow-up if CLI exposure is wanted.
- `landing_dispersion_m` is per-episode, matching the legacy per-episode
  `lateral_rms_m`; pool across episodes the same way the CLI pools today.
