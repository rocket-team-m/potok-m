# FIXPUMP — root-cause fix: aerodynamic normal force pumped energy

2026-11-23. Files touched (only these three, plus this report):
`sim/core/src/dynamics.rs`, `sim/core/tests/actuator_mapping_tests.rs`,
`sim/core/tests/or_parity_extended_tests.rs`. No coefficient/config tuned.
No subagents. Local commit only, no push.

## Diagnosis (confirmed)

`dynamics.rs::derivative` built the placeholder normal force as

```
normal = [qdyn·A·cna·vb0/a_z, qdyn·A·cna·vb1/a_z, 0]   (vb = Rᵀ(v - w))
```

i.e. **parallel to the lateral relative wind**, not perpendicular to it. Its
wind-frame power is `N·v_b = qdyn·A·cna·(v_b0²+v_b1²)/a_z ≥ 0`, so every
nonzero angle of attack injected energy. With `cna = 1.8` vs `cd ≈ 0.3` the
pump exceeded drag losses at high qdyn. WORLD04 measured E rising 37.2 →
51.4 kJ/kg from t=10→20 s on the oracle coast; oracle apogee 3971–4000 m beat
the 3819 m energy ceiling by 152–203 m.

## Fix

Project `normal_raw` onto the plane perpendicular to the relative velocity
(Gram–Schmidt against `vb`):

```
let along = dot(normal_raw, vb) / speed²;
let normal = normal_raw - vb * along;      // speed > 1e-6; else raw (qdyn≈0)
```

* Drag (`-qdyn·A·cd·vhat`) is now the only force doing work: aero power
  `= -qdyn·A·cd·|v_rel| ≤ 0`. The normal (lift) does exactly zero work.
* The static-margin restoring moment `cross([0,0,sm], normal)` keeps its
  small-alpha value: `normal[0] ≈ F·(v_z/speed)² → F` as α → 0.
* No new branches on the command path: zero-command stays bit-identical
  between `run_episode` and `run_episode_with_control(0,0)` (test green).
* No coefficient, config value or threshold was fitted to a target.

## Tests added

* `actuator_mapping_tests::aerodynamic_normal_force_does_no_positive_work`
  — calls `dynamics::derivative` directly at a large-angle-of-attack,
  post-burnout state; compares aero power with `cna = 0` (drag only) against
  the real `cna`. Before the fix the lift arm adds ≈ +5.95 W/kg; after it the
  two are **bit-identical** (`-5717.460954 W/kg` both). This is the decisive
  per-force gate.
* `or_parity_extended_tests::oracle_apogee_respects_burnout_energy_ceiling`
  — gate-off boost means baseline and oracle share the burnout state; asserts
  oracle apogee ≤ `z_bo + |u_bo|²/2g + 25 m` from the baseline tick. Coarse
  envelope guard; the direct work test is the real detector.
* Also repaired a pre-existing f32-precision bug in `time_to_apogee_s`:
  trajectory altitude is f32 (ULP ≈ 2.4e-4 m at 3 km) but the tick-vs-apogee
  assertion used an absolute 1e-6 m bound and failed on a correct tick that
  rounded up. Changed to 1e-3 m; the 1 % relative gate is unchanged.

`coast_specific_energy_never_increases` was trialled and **removed**: per-tick
wind-frame energy is not a clean instrument here because gust onset steps the
wind field (`u·Δw` jumps) and turbulence carries a vertical component
(`g·w_z`); the direct derivative gate is both exact and noise-free.

## Measured numbers (before → after)

Same working tree, same config; "before" is the baseline run taken before the
dynamics.rs edit, "after" is post-fix.

| quantity | before | after |
|---|---|---|
| held-out lateral RMS, seed 1 (baseline → oracle) | 1525.97 → **1018.01 m** (−507.96) | 1309.50 → **1300.53 m** (−8.96) |
| seed 2 | 1511.20 → 985.99 m | 1292.16 → 1286.45 m |
| seed 3 | 1528.74 → 1004.56 m | 1309.46 → 1300.64 m |
| episode 0 apogee (OR parity) | 3185.269 m | 3185.269 m (unchanged) |
| episode 0 vmax | 278.257 m/s | 278.257 m/s |
| oracle ep16 apogee | 3971–4000 m *[WORLD04, earlier config]* | 1838–1902 m |

Reading: the oracle's apparent "authority" was largely the pump — keeping the
nose at high α kept qdyn (hence fin authority) high and added energy. With an
energy-conserving normal force the oracle can no longer buy apogee or the
large lateral reduction; its lateral gain collapses from ~−508 m to ~−9 m.
Episode 0 (uncontrolled, near-zero α) is essentially unaffected, which is why
the OpenRocket scalar gates (apogee 1.85 %, vmax 1.45 %, time 0.14 s) still
pass. No claim about a real-rocket apogee is made from these numbers.

## Verification

* `cargo test -p potok-sim --test actuator_mapping_tests --test
  or_parity_extended_tests` → **14 passed, 0 failed**.
* A full `cargo test -p potok-sim` run after the fix showed all test binaries
  green except one golden-pin test that failed once under concurrent load and
  passed immediately when re-run in isolation (race with other agents'
  builds). A later full run failed to compile `curriculum_tests` with 11
  unresolved imports (`config::CurriculumKnobs`, `Config.curriculum`, …) — an
  unrelated in-flight refactor by another agent in `config.rs` /
  `curriculum_tests.rs`, not in any file assigned to this task.
* **Concurrency caveat:** other agents were actively editing this shared
  checkout. `or_parity_extended_tests.rs` was overwritten once mid-task
  (my first test version and the f32 fix disappeared) and both were re-applied.
  `dynamics.rs` also received a concurrent optional servo path
  (`Vehicle::servo`, `fin_output`, `with_servo`, `None` by default); that work
  landed in `aaa25aa` (SIMSERVO) before this commit. My dynamics.rs change was
  staged surgically (`git apply --cached` of a one-hunk diff against HEAD), so
  this commit carries **only** the normal-force fix — no servo code.
