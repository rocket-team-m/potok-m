# SIM03 — wind & scenario test gates (2026-11-23)

Scope: create only `sim/core/tests/wind_scenario_tests.rs`. No source edits, no
subagents, no push. License: `SPDX-License-Identifier: GPL-3.0-or-later`.

## Delivered

12 integration tests (`potok-sim`, crate `core`), all deterministic, no fixtures
and no network/filesystem access:

- **Deterministic wind profile** — `wind_field_is_bitwise_reproducible_for_a_seed_and_episode`,
  `wind_field_changes_with_seed_and_episode`, `mean_profile_follows_power_law_direction_and_wind_factor`.
  Pins bitwise equality across two `Wind` instances (gust flags and `at()` at
  sampled steps × altitudes × wind factors), seed/episode sensitivity, the
  `v_10 (h/10)^α · factor` power law, `factor.max(0)`, the `h.max(0.1)` floor,
  direction and `v_ref = max(|v_10|, 2)`.
- **Shear layer** — `shear_layer_is_a_centered_tanh_jump`,
  `shear_active_band_matches_twice_the_thickness`. Pins scenario-selected
  delta/altitude, the tanh profile below/at/above the layer, odd symmetry
  `f(z−d)+f(z+d)=delta`, monotonicity, and the inclusive `±2·thickness` band.
- **1-cos gust** — `one_cos_gust_envelope_is_zero_at_ends_and_full_at_midpoint`,
  `non_gust_scenarios_never_arm_the_gust`. Isolates the gust by differencing
  against an amplitude-zero wind with identical RNG draw order; checks
  `0.5·A·(1−cos 2πu)`, along-wind-only contribution, active-flag exactness,
  `amp`/`duration`/`t0` envelope, peak ≈ A, window length ≈ `d/dt`, and that
  `gust_u = 0` cells never arm it.
- **Dryden-style turbulence** — `turbulence_slices_are_bitwise_reproducible_and_seed_sensitive`,
  `turbulence_is_a_correlated_process_not_white_noise`,
  `dryden_style_turbulence_matches_the_seeded_ar1_reference`. Pins bitwise
  reproducibility/seed sensitivity, lag-1 vs lag-200 autocorrelation, and the
  exact seeded 3-filter AR(1) recurrence (Box–Muller init, `a=exp(−dt·v_ref/L)`,
  `b=g·√(1−a²)`, scales 1/0.8/0.45) against an independent reference at 128 steps.
- **Withheld split boundaries** — `withheld_split_boundary_is_index_16_and_cycles_with_the_cells`,
  `withheld_cells_are_unseen_extreme_combinations`. Pins 20 cells, exact
  train/held boundary at index 16, cyclic repetition, disjoint signatures,
  `lag_scale` twist, extreme combos, the withheld cell driving the real wind
  field, and `Split` code/label round-trips.

`wind.rs` is described in-tree as "Dryden-like"; no Kármán renderer exists, so
the reproducibility gate targets the implemented Dryden-style filters.

## Verification

- `cargo test -p potok-sim --test wind_scenario_tests` — **12 passed, 0 failed**
  (0.36 s).
- Tracked baseline + new target:
  `cargo test -p potok-sim --lib --test actuator_mapping_tests … --test wind_scenario_tests`
  — **92 passed, 0 failed across 13 binaries**.
- `rustfmt --edition 2021 --check core/tests/wind_scenario_tests.rs` — clean.
  (`cargo fmt --all --check` is not clean repo-wide; the diffs are pre-existing
  in other files and were left untouched.)

**External blocker, not caused by this task:** bare `cargo test -p potok-sim`
does not currently compile because other night agents are writing untracked
test targets in the same checkout: `core/tests/aero_hardening_tests.rs:383`
(invalid `{c.cd0}` format string) and later `core/tests/sensor_realism_tests.rs:320,340,341`
(`f32` vs `f64` `assert_eq!`). Both are outside my scope ("create ONLY …"); the
verification above uses the tracked test set plus this new target.

Commit: local only, no push.
