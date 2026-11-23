# `a_lat_resid`: definition, stale pin, and falsification

This document records the corrected definition of the `a_lat_resid` physics
feature, why the old one was uninformative, and the gates that fail if the
correction is reverted. No coefficient, config value or trajectory target was
changed.

## Definition as shipped

`core/src/physics.rs::PhysicsFeatures::from_state` computes, in body axes:

```
v_rel  = v_world - wind
vb     = R(q)^T v_rel
speed  = ||v_rel||
q_dyn  = 0.5 * rho * speed^2
az     = max(speed, 1)
N_raw  = q_dyn * A * cna * (vb_x / az, vb_y / az, 0)
N      = N_raw - vb * (N_raw . vb) / speed^2      if speed > 1e-6, else N_raw
D      = -q_dyn * A * cd(mach) * vb / speed
m      = s.mass() clamped to the integrator's post-separation floor
a_lat_resid = hypot(a_meas_x - (N_x + D_x)/m,
                    a_meas_y - (N_y + D_y)/m)
```

The projection of `N_raw` and the mass floor are the two load-bearing pieces.
Both must match `dynamics::derivative`, which is what produces the sensor's
measured specific force:

- `derivative` projects `normal_raw` perpendicular to the relative wind so the
  placeholder normal force does no positive work (commit `584b1dc`). Drag is
  the only force doing work.
- The RK4 mass floor drops by `stage_sep_mass` at separation (commit
  `150e7b9`); the feature path mirrors it in `from_state` (commit `08a4c3c`).

`README.md`'s formula table still prints the pre-projection normal force
(`N = q_dyn A cna v_body_i/az`) and is **stale**; `README.md` is outside this
task's scope, so the authoritative definition is the code above.

## Root cause of the three failures

`physics.rs` was last edited by `08a4c3c` (mass floor) — before `584b1dc`
changed the integrator's normal force. Its baseline therefore still used the
unprojected, along-wind `N_raw`. In horizontal flight (`vb = [120, 0, 0]`),
`N_raw` is parallel to `vb`, so the integrator's projected normal is exactly
zero while the stale baseline kept the whole pump term:

| quantity | value at `vb = [120,0,0]`, m = 11 kg |
|---|---:|
| stale calm residual `q_dyn A cna / m` | **15.876 m/s²** |
| corrected calm residual | ~0 |
| gust gap, 20 m/s crosswind (drag change only) | ~0.46 m/s² |
| gust gap, 50 m/s crosswind | 1.2538 m/s² |

Consequences, all measured before the fix:

- `residual_tests.rs:145` — calm residual 15.876 fails `< 1e-3`.
- `residual_stream_tests.rs:189` — same calm residual 15.876.
- `residual_tests.rs:249` — the ~15.9 floor fluctuated frame to frame, so the
  two noise-dominated means crossed (`gust_mean 0.026886 <= calm_mean
  0.027060`); the comparison measured which frames the coast range happened
  to include, not a gust.

## Fix

1. `core/src/physics.rs` — compute the 3-D `normal_raw`, project it with the
   same `dot`/`scale`/`sub` arithmetic as `derivative`, and divide the sum with
   drag by the same mass. One helper is duplicated rather than extracted:
   `derivative` needs `atm` and `t`, which `from_state` does not receive, and
   editing `dynamics.rs` was outside this task's scope.
2. `core/tests/residual_tests.rs`, `core/tests/residual_stream_tests.rs` —
   the crosswind stimulus goes from 20 to 50 m/s. **No threshold was lowered**:
   the `gust > 1.0 m/s²` gate and the `calm < 1e-3` gate are unchanged; the
   corrected metric simply needs a stronger crosswind to produce a 1 m/s² drag
   gap. Coefficients still come from `config/default.toml`.
3. `core/tests/residual_tests.rs` — the episode-level mean-vs-mean assertion
   is replaced with a sensor-band bound. With a correct baseline, measurement
   and baseline use the same true wind, so a gust is *explained* by the aero
   model and cannot raise the residual; only sensor lag/bias/noise remains.
   The gust-sensitivity claim is carried by the two frozen-measurement
   crosswind gates, which are falsifiable.
4. `core/tests/residual_tests.rs` — new
   `lateral_residual_ignores_the_projected_normal_pump` gate: with two
   otherwise identical vehicles (`cna = 1.8` and `cna = 0`) on the same
   horizontal calm state, the residual must be identical and < 1e-3.

## Falsification evidence

Each correction was reverted in `physics.rs` and the gates were re-run:

| reverted formulation | gate | result |
|---|---|---|
| project normal → unprojected `N_raw` | `lateral_residual_ignores_the_projected_normal_pump` | **FAIL**: `15.876` vs `0.0000000059` |
| project normal → unprojected `N_raw` | `crosswind_gust_raises_lateral_residual` | **FAIL**: calm `15.876` |
| post-sep mass floor → `max(s.mass(), dry_mass)` | `post_separation_sample_uses_post_sep_mass` | **FAIL**: drag ratio `1` vs mass ratio `1.2832097` |
| both restored | both test binaries | 7/7 and 8/8 green |

## Verification

- `cargo test --workspace`: **321 passed, 0 failed** (49 result lines,
  including doc-tests).
- `cargo test -p potok-sim --test residual_tests --test residual_stream_tests`:
  **8/8 and 7/7 passed**.
- `rustfmt --edition 2021 --check` on the three edited Rust files: exit 0.
- `cargo clippy --workspace --all-targets`: exit 0. The three clippy warnings
  in the two edited test files (`residual_tests.rs:317,382`,
  `residual_stream_tests.rs:223`) are unchanged from `HEAD`; no new warning
  was introduced.
- `cargo fmt --all -- --check` still reports pre-existing diffs in files owned
  by other lanes (`parachute.rs`, `validation.rs`, `firmware_cosim.rs`,
  `lateral_metrics_tests.rs`, `nees_gates_tests.rs`, …). None is in the three
  edited Rust files.

## Status

This is tested consistency, not external validation: no flight log or measured
aero force exists in the repository. The corrected metric matches the
integrator's own force model exactly; that is the claim being made.
