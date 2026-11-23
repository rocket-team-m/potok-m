# FIXSIM2 — physics features ignored stage separation

**Date:** 2026-11-23
**Repo:** `/home/sehaxe/potok-m/sim` (root `/home/sehaxe/potok-m`)
**From:** SIMDBG finding #2 (HIGH). No subagents.
**Files touched:** `core/src/physics.rs`, `core/tests/residual_tests.rs`, `core/tests/quality_v3_tests.rs`. Nothing else.
**Code commit:** `08a4c3c fix(sim): use post-separation mass floor in physics features` (local, no push). This report is committed separately.

## Root cause

`PhysicsFeatures::from_state` used `let mass = s.mass().max(v.dry_mass)` (old `physics.rs:59`).
The integrator lowers the mass floor once the first-stage casing is jettisoned
(`dynamics.rs:508-514`: `sep_done = stage_sep_mass` for `t >= stage_sep_time`,
`mass_min = dry_mass - sep_done`; `rk4` clamps the state to the same floor at
`dynamics.rs:629-636`). With the shipped config (`stage_sep_time_s = 6.0`,
`stage_sep_mass_kg = 2.55`), after 6 s the true mass is ~6.32 kg at t=7 s while
the feature path clamped to ~8.09 kg — ~28 % too high. `a_lat_resid` and
`drag_est` both divide by that mass, so they were biased low through coast and
descent. `alpha/beta/q_dyn/static_margin` do not depend on mass and are unchanged.

## Fix

`physics.rs` now mirrors the dynamics convention:

```rust
let sep_done = if s.mass() < v.dry_mass { v.stage_sep_mass } else { 0.0 };
let mass = s.mass().max(v.dry_mass - sep_done);
```

`t` is **not** passed to `from_state`: the only production call site is
`episode.rs:330-332`, which is outside this task's file scope. The state passed
in already carries the jettison — the RK4 clamp keeps `s.mass() >= dry_mass`
before separation, so a state mass below `dry_mass` is an exact separation
signal. With the corrected floor, the floor never exceeds the state mass on any
reachable state, so the true mass now passes through unchanged. No other feature
math changed.

## Tests added

- `core/tests/residual_tests.rs::post_separation_sample_uses_post_sep_mass` —
  two states with identical kinematics and zero measured specific force, one at
  the pre-sep floor (`dry_mass`), one below it (`dry_mass - 0.7 * stage_sep_mass`).
  With zero measured force both mass-dependent features are exactly ∝ 1/mass,
  so their ratio pins the mass actually used (expected `dry_mass / post_mass`).
  Also pins the four mass-independent features as unchanged.
- `core/tests/quality_v3_tests.rs::post_sep_mass_reaches_v3_features_without_layout_changes`
  — end-to-end through the v1/v2/v3 writers and the v3 truth sidecar. Asserts
  every writer keeps its exact byte length (layouts unchanged), and for every
  aerodynamic frame the mass implied by `drag_est = cd0*q_dyn*area/mass` equals
  the trajectory tick's mass (using the episode's sampled `cd0` from
  conditioning, which is what `vehicle.aero.cd0` is set to). Also asserts a t>7 s
  frame's mass is >3 kg below lift-off, i.e. separation actually happened.

## Verification

The shared checkout does not compile: a sibling agent's untracked
`core/src/aero_derivatives.rs` uses `use potok_sim::{...}` from inside the crate
and is wired into `core/src/lib.rs` by that agent. Out of scope, not touched.
Verification used a clean `HEAD` snapshot at `/tmp/opencode/potok-clean` with
**only** the three files above overlaid.

- `cargo test -p potok-sim --test residual_tests --test quality_v3_tests`:
  **8 + 7 passed, 0 failed.**
- `cargo test -p potok-sim` (full suite): every test binary green **except one
  pre-existing pin in a file outside this task's scope** (details below).

### Out-of-scope blocker: stale pin in `core/tests/smart_sim_tests.rs`

`physics_features_are_finite_and_bounded` at `core/tests/smart_sim_tests.rs:63`
asserts `f.physics[4] <= at_liftoff * 2.0 + 1e-6`, where `at_liftoff` divides by
the lift-off mass. That bound encodes the old assumption "mass never drops below
half the lift-off mass"; after separation the floor is `dry_mass - 2.55`, which
can be lower. Measured failure with the fix:

```
seed=1 idx=0 frame=456 t=9.12 s
drag_est=26.122181  at_liftoff=13.059529  ratio=2.0002
lift-off mass=10.621 kg   tick mass=5.310 kg (= dry_mass - 2.55)
```

The pin is wrong, not the fix. SIMDBG §"Honest limits" predicted exactly this:
"finding 2 is pinned by existing tests using the wrong mass; fixing them requires
updating those pins, not deleting them." I did **not** edit the file because the
task restricts changes to the three files listed above. The owner should replace
the fixed 2× bound with one derived from the trajectory tick's mass (the same
invariant the new v3 test uses), e.g.:

```rust
let at_tick = at_liftoff * c.mass_kg / e.trajectory[i].mass_kg;
assert!((f.physics[4] - at_tick).abs() / at_tick < 1e-3);
```

**State:** repo-wide `cargo test -p potok-sim` is RED on that single pin until
its owner updates it. My three files and their tests are green.
