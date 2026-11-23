<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
# Flight-log NEES / chi-square validation

This document describes what the flight-log replay seam can and cannot claim.
It is intentionally narrow: the instrument is real, the real-data verdict is
**NOT MEASURED**, and no flight or vehicle claim is made anywhere in it.

## What is validated

`core/src/flight_replay.rs` provides a schema (v1), strict JSON/CSV import,
monotone time/phase alignment, residuals against a simulated episode, and the
statistics used to score those residuals:

- `nees` — diagonal normalized error squared, `sum((r_c / sigma_c)^2)`.
  For zero-mean Gaussian residuals with the declared sigma this is chi-square
  with `channels.len()` degrees of freedom.
- `chi2_sf` / `chi2_quantile` / `nees_corridor` — the central chi-square
  survival function, its bisection inverse, and the acceptance interval
  `[Q((1-c)/2), Q((1+c)/2)]`.
- `corridor_report` — counts values inside a corridor and returns empirical
  coverage.

These are consistency instruments. They prove the arithmetic agrees with the
published chi-square distribution; they do not prove a sensor, a simulator or
a recording is accurate.

## Deterministic corridor gate (10^4 runs)

`synthetic_nees_corridor_gate(seed, coverage)` is a pure function of its seed:
10,000 paired Monte-Carlo runs draw eight independent standard-normal residuals
per run. One estimator (the calibrated reference) uses them unchanged, so NEES
is distributed chi-square with 8 degrees of freedom. The other (deliberately
biased) adds +2 sigma on every channel, so NEES is non-central chi-square with
non-centrality 32 — centred near 40, well above the 95 % corridor's upper bound
of about 17.5. The two estimators share the same draws, which makes the
comparison paired and lower-variance.

The gate is exercised by `core/tests/nees_gates_tests.rs`:

| Gate | Assertion | Meaning |
|---|---|---|
| `synthetic_gate_is_deterministic_with_a_pinned_budget` | same seed gives the same result; budget is exactly 10,000 | no hidden RNG or wall-clock state |
| `calibrated_estimator_nees_stays_inside_the_chi2_corridor` | coverage in `[0.94, 0.96]` at 95 % nominal, across three seeds | a calibrated estimator lands inside the corridor |
| `biased_estimator_nees_fails_the_chi2_corridor` | coverage `< 0.05` and at least `0.80` below the calibrated arm, across three seeds | a biased estimator is rejected |
| `real_log_corridor_gate_is_loud_not_measured` | real-log path returns `NotMeasured("NOT MEASURED …")` | no silent pass on the real-data path |

Run with:

```
cargo test -p potok-sim --test nees_gates_tests
```

A green run proves the corridor discriminates a calibrated estimator from a
biased one on synthetic Gaussian data. It is **not** validation of flight data.

## Real flight logs: LOUD `NOT MEASURED`

No real flight log with per-channel sigma ships with this repository, and no
production call site imports one. `validate_real_log_nees(&log)` is the real-data
entry point and always returns `FlightLogError::NotMeasured`, whose message
contains `NOT MEASURED`, names the missing input ("per-channel sigma") and the
escape (supply a real log and implement the path). It is a hard refusal, not a
silent fallback, and no synthetic fixture may be substituted for a measurement
there. Until that path is implemented, real-data corridor coverage is
unmeasured and must be reported as such.

## Ceilings and non-claims

- Diagonal covariance only. Cross-axis correlation would need a full matrix
  and a Cholesky solve.
- Gaussian, zero-mean residuals are an assumption of the test, not a
  measurement of any sensor.
- The synthetic gate tests the instrument, not the vehicle, sensors,
  simulator, or any real recording.
- The replay seam is not wired into `quality.rs` gates or dataset generation.
- No threshold here is fitted to make an arm or controller win; the biased arm
  is a fixed +2 sigma offset, not a tuned target.

There are no real-data results in this document or its tests. Do not add a
number to this page unless it names a real log, its per-channel sigma, the
corridor coverage and the commit it was measured at.
