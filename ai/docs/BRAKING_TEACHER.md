# BRAKING_TEACHER — privileged radial-braking reference (BRK01)

**Status:** training-only reference in the staged teacher-v2 privileged and
DAgger stages. CPU, simulation truth, no flight claim, no gate/threshold/
physics constant changed elsewhere.

SPDX-License-Identifier: GPL-3.0-or-later

---

## 1. What it is

`src/potok_ai/braking_teacher.py` turns a true inertial state into the fin
acceleration demand `[u_lat, u_pitch]` in m/s² — the same two channels the v3
dataset's `g_accel` label carries (`sim/core/src/optimizer.rs:127-137`). It is a
*reference*, not a learned model: no parameters, no optimizer, no gradient.

It reads only privileged slots 0..3 (position xyz) and 3..6 (velocity xyz) of
the POTOKTRJ v3 record (`sim/core/src/trajectory.rs:94-121`). Label slots 22..25
are never an input to the demand. If a full 26-slot row is passed, only the
first 22 slots are read.

> **Sibling defect, not touched here:** `objective.py:48-50` declares
> `TRUTH_VEL_START = 11`, but the v3 layout puts velocity at 3..6 and body rate
> at 10..13. The impact objective therefore reads omega/mass as velocity. This
> module uses the wire-format slots (0..3 / 3..6) and does not inherit that
> offset. Fixing the sibling objective needs its own change and A/B.

## 2. Law and where every coefficient comes from

Let `r = hypot(x, y)`, `v_r = (x·vx + y·vy)/r` (outward positive). The signed
inward demand is

```
a_r = clip( -(k_v·v_r + k_p·r), -a_cap, a_cap )
a_world = a_r · (x/r, y/r)          [ax, ay]
[u_lat, u_pitch] = [ay, -ax]        (sim/core/src/optimizer.rs:127-137)
```

Constants, with units and sources:

| symbol | value | unit | source |
|---|---|---|---|
| `a_cap` | 10.0 | m/s² | `sim/core/src/control.rs:34` (`MAX_COMMAND_MPS2`) |
| `T` | 2.0 | s | declared control horizon (design knob, not fitted) |
| `v_ref` | `a_cap·T` = 20 | m/s | speed the cap can null in `T` s |
| `d_ref` | `v_ref²/(2·a_cap)` = 20 | m | stopping distance `d = v²/(2a)` |
| `k_v` | `a_cap/v_ref = 1/T` | 1/s | demand reaches cap at `v_ref` |
| `k_p` | `a_cap/d_ref = 2/T²` | 1/s² | demand reaches cap at `d_ref` |

Every coefficient is algebraic from `a_cap` and `T` via the stopping-distance
identity. No weight was searched, fitted to a trajectory or tuned to the −35 %
/ +15 % closed-loop gates. Changing `T` changes the controller honestly; it does
not re-scale a target to pass a test.

Phases are reported, not switched: `BRAKE` when `v_r > v_ref` (outward),
`CORRECT` otherwise, `SETTLED` inside the 1 m deadband with zero radial speed.
Clipping the same sum already saturates at the cap in the brake phase, so the
phase field labels arithmetic rather than substituting a different law.

## 3. Phase gating, envelope and explicit reasons

- **One-way phase whitelist (BRK03).** `FlightPhase` names the organizer's
  phases. `PAD`, `BOOST` and `MOTOR_ARMED` are **banned** and return zero with
  reason `boost_gate`; `COAST` and `DESCENT` are the **only active** phases in
  which the radial demand is computed; an `UNKNOWN` supplied explicitly is
  banned too. With no clock and no explicit phase (the DAgger wire row) the
  whitelist is not applied and the recorded health bit is the sole interlock.
  The gate tests membership in `ACTIVE_PHASES`, never the complement, so a
  mislabelled clock cannot promote a banned frame to a target. An explicit
  `flight_phases` argument from a caller overrides the clock-derived label.
  `flight_phase()` transcribes `sim/core/src/dynamics.rs:735-755`: `t <= 0` is
  Pad, `0 < t <= boost_end_s = 9.95` s (`sim/core/tests/sim_tests.rs:295`) is
  Boost/motor-armed, and post-burnout is Coast or Descent by the sign of `vz`.
  No threshold or physics constant was changed. The caller may supply a
  separate `ai_allowed` bit that is ANDed with the phase gate.
- **Envelope.** `a_r` is clipped to ±`a_cap` and the two fin channels are
  clipped again to ±10 m/s². A clipped row is counted as `cap_clamped`.
- **Not in range.** Non-finite truth → `non_finite`; `r ≈ 0` (no radial
  direction) → `radius_zero`; `r > 10 km` or `|v| > 1 km/s` → `out_of_range`.
  Each returns zero demand with phase `out_of_range` and its own reason.
- Every row gets exactly one reason; `BrakingTeacher.reason_counts()` exposes
  the histogram, which is copied into `metrics.json` for each stage.

## 4. Wiring

- `TeacherV2Config.braking_teacher` defaults to `True`. CLI:
  `--braking-teacher` (explicit) / `--no-braking-teacher` (legacy escape).
  The flag is part of `config_hash`, so a braking run and a legacy run cannot be
  confused after the fact.
- **Privileged stage:** each training batch's target tensor is replaced by the
  radial demand computed from the same truth rows the stage already reads.
  `_WindowTickCursor` keeps a per-episode window counter for the whole stage:
  window n ends at flight tick `window - 1 + n`, advanced once per batch.
  BRK01's counter was local to each batch, so every row mapped to ticks 11..74
  and 100% were boost-gated (1451/1451 per seed in BRK02). The impact
  objective reads the same ticks; JEPA and privileged latent are unchanged.
  Eval stays label-blind and keeps the recorded target.
- **DAgger stage:** the recorded reference label remains in
  `sample.correction` (its provenance contract is unchanged); the braking
  demand is a parallel `dataset.braking_targets` stream, and the training loop
  uses it as the target when the flag is on. Provenance counts and details say
  `training_target: radial_braking_reference` vs `recorded_label`.
- With no truth sidecar, braking cannot run: both stages report
  `braking.reason = no_truth_sidecar` (or `legacy_braking_teacher` when the
  flag is off) and keep the historical target. This is a counted, visible
  fallback, not a silent no-op.

## 5. Gates and verification

`tests/test_braking_teacher.py`: 13 tests — the 11 BRK01 gates plus two BRK03
falsification gates:

1. `test_phase_gate_is_zero_on_banned_phases_and_active_after_burnout` checks
   the split directly: an explicit phase vector (Pad/Boost/MotorArmed/Coast/
   Descent) produces exactly zero command on the three banned rows, nonzero on
   both active rows, and fails under an inverted gate.
2. `test_privileged_target_is_active_on_post_burnout_rows` runs the real hook
   on 600-frame truth records across many batches. It asserts the cursor does
   not reset (`ticks[64] == ticks[0] + 64`, last tick = 599), that every
   pre-burnout row's target is exactly zero, and that ≥80% of the 204
   post-burnout privileged rows have a nonzero target — both in the direct hook
   call and in `metrics.json`'s `reason_counts` after `ttv.run`. The old
   per-batch counter gives 0 active rows and fails this test.

Full AI suite and `ruff check src tests` are run before commit.

## 6. Honest limits

- This lands a reference and its wiring, **not** a measured closed-loop win.
  AILIMIT records that under the unweakened gate and ±10 m/s² envelope no
  controller can reach the ≤1000 m deployment proxy (cap-10 generous bound
  1211–1217 m); this module does not change that envelope, the gate, or the
  fin stop, and no new flight A/B is claimed here.
- The demand-to-lateral-acceleration transfer is assumed 1:1 in the reference.
  The oracle's measured transfer is worse at high amplitude; a reference is a
  target, not a predictor of flown trajectory.
- The DAgger wire row carries no clock, so its phase is `UNKNOWN`; the health
  bit inferred from the recorded label's zero-ness is the sole interlock there.
  A dataset that emits a non-zero label during boost would defeat that bit; the
  durable fix is to pass an explicit `flight_phases` vector or a real clock from
  the rollout, which the hook now supports. This module will not infer a phase
  from a label number.
- All evidence here is simulator data and CPU tests. No canopy/deployment
  event and no flight log backs any lateral claim.
