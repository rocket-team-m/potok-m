# FIXSIM2b — replace the stale drag_est 2× pin with a mass-derived invariant

**Date:** 2026-11-23
**Repo:** `/home/sehaxe/potok-m/sim` (root `/home/sehaxe/potok-m`)
**Follow-up to:** FIXSIM2 (commit `08a4c3c`), whose report flagged this pin as its out-of-scope blocker.
**Scope:** `core/tests/smart_sim_tests.rs` only. Nothing else. No subagents. GPL-3.0-or-later.
**Code commit:** `7ddf324 fix(sim): pin drag_est to trajectory mass after separation` (local, no push). This report is committed separately.

## What was wrong

`physics_features_are_finite_and_bounded` asserted:

```rust
let at_liftoff = c.cd0 * f.physics[2] * 1000.0 * c.area_m2 / c.mass_kg;
assert!(f.physics[4] >= at_liftoff * 0.99 - 1e-6);
assert!(f.physics[4] <= at_liftoff * 2.0 + 1e-6);
```

The upper bound encoded "mass never drops below half the lift-off mass". FIXSIM2 changed the
physics mass floor to `dry_mass - stage_sep_mass` after separation, so that is no longer true:
with the shipped config the post-sep floor (`5.310 kg`) is below half of a `10.621 kg` lift-off,
and `drag_est` reaches `2.0002×` the lift-off-normalized value. The pin failed on a correct fix.

## Fix

Replace both fixed-multiple assertions with the exact relation the feature is defined by.
`drag_est = cd0·qdyn(Pa)·area / mass`; `qdyn = physics[2]·1000`; therefore the value at the
current tick's mass is `at_liftoff · mass_liftoff / mass_tick`:

```rust
let at_liftoff = c.cd0 * f.physics[2] * 1000.0 * c.area_m2 / c.mass_kg;
let at_tick = at_liftoff * c.mass_kg / e.trajectory[i].mass_kg;
assert!(
    (f.physics[4] - at_tick).abs() <= at_tick * 1e-3 + 1e-6,
    "seed {seed} index {index} frame {i}: drag_est {} vs at_tick {}",
    f.physics[4],
    at_tick
);
```

`e.trajectory[i].mass_kg` is the same `state.mass()` the feature path consumes
(`episode.rs:303-313` ticks at every `step % spf == 0`, alongside `frames.push` at `:362`;
`physics.rs:64-69` derives its floor from that same state). Tolerance is 1e-3 relative; the
`+ 1e-6` absolute floor is the old pin's padding, kept so a zero-qdyn tick (`at_tick == 0`)
cannot become `0/0`. The failure message now names seed, index, frame and both numbers.

## Verification

- `cargo test -p potok-sim --test smart_sim_tests`: **6 passed, 0 failed** (33 s), including
  `physics_features_are_finite_and_bounded`.
- **Falsification** — the pin must fail if mass handling regresses. In a throwaway clean
  snapshot the post-sep floor in `physics.rs` was reverted to `s.mass().max(v.dry_mass)` and
  the single test re-run. It failed as required:
  `seed 1 index 0 frame 300: drag_est 6.3131647 vs at_tick 7.907003` (~20 % gap, far above
  1e-3). Snapshot discarded; no production file was changed by the probe.

### `cargo test --workspace` is not green in this checkout — out of scope

Run as instructed from `sim/`. It does not compile: an unrelated untracked sibling file
`core/tests/servo_actuation_tests.rs` imports `potok_sim::identification::ServoActuator` and calls
`Vehicle::with_servo`, neither of which exists in the current tree (E0432 + 3× E0599). That file is
another agent's WIP; out of scope, not touched. A clean `HEAD` snapshot cannot be used as a
workaround either: `HEAD`'s `lib.rs` does not declare `pub mod reward`, so the tracked
`rl_env_tests.rs` fails to compile there — siblings' staged changes are uncommitted.

Running every other test target in the shared checkout (`--no-fail-fast`) gives **35 targets
green**, including `smart_sim_tests` (6/6), `quality_v3_tests` (8/8) and `residual_tests`'s
FIXSIM2 test. The failures are all in sibling-owned files and moved between two runs of the
same command (the checkout is a moving target):

| target | failing tests |
|---|---|
| `curriculum_tests` | `training_grid_varies_every_previously_fixed_knob`; also `knobs_are_deterministic_and_each_owns_its_substream` in the first run (`lag_scale` all `1.0`) |
| `residual_stream_tests` | `crosswind_opens_lateral_residual_at_stream_column_eleven`, `gust_frames_carry_residual_in_stream_column_eleven` |
| `residual_tests` | `crosswind_gust_raises_lateral_residual`, `gust_episode_carries_residual_in_the_feature_path` |
| `manifest_hash_tests` | 1 failure in the first run; **green (6/6)** in the second |
| `servo_actuation_tests` | compile error (above); excluded |

None of these files imports or calls the changed test. They are not caused by this change.

## State

The FIXSIM2 blocker is closed: the stale 2× pin is replaced by a mass-derived invariant that
still fails on a mass-handling regression. Worktree-wide `cargo test --workspace` stays red until
the sibling agents land their WIP; that is outside this task's file scope.
