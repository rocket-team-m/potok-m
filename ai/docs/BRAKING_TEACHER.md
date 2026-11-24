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

- **No boost.** A row at or before `boost_end_s = 9.95` s
  (`sim/core/tests/sim_tests.rs:295`) returns zero with reason `boost_gate`; the
  caller may supply a separate `ai_allowed` bit that is ANDed with the time
  gate. In the privileged stage the bit comes from the window's flight tick;
  in DAgger from the recorded health flag (the reference emits exactly zero
  during boost). Only the boolean is used, never the recorded number.
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
  radial demand computed from the same truth rows the stage already reads;
  tick-derived time gates boost. The impact objective, JEPA and privileged
  latent are unchanged. Eval stays label-blind and keeps the recorded target.
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

`tests/test_braking_teacher.py`: 11 tests — derived coefficients, correct truth
slots vs labels, cap-saturated brake, position correction after velocity is
nulled, settled deadband, both boost gates, the three not-in-range reasons with
zero demand, cap/histogram over a mixed batch, default-on CLI escape, and both
stage wirings (including that a recorded `[3, 4]` label is *not* the braking
target). Full AI suite and `ruff check src tests` are run before commit.

## 6. Honest limits

- This lands a reference and its wiring, **not** a measured closed-loop win.
  AILIMIT records that under the unweakened gate and ±10 m/s² envelope no
  controller can reach the ≤1000 m deployment proxy (cap-10 generous bound
  1211–1217 m); this module does not change that envelope, the gate, or the
  fin stop, and no new flight A/B is claimed here.
- The demand-to-lateral-acceleration transfer is assumed 1:1 in the reference.
  The oracle's measured transfer is worse at high amplitude; a reference is a
  target, not a predictor of flown trajectory.
- The health bit used in DAgger comes from the recorded label's zero-ness. If a
  future dataset emits a non-zero label during boost, the interlock must move
  to an explicit phase field; this module will not infer it from a number.
- All evidence here is simulator data and CPU tests. No canopy/deployment
  event and no flight log backs any lateral claim.
