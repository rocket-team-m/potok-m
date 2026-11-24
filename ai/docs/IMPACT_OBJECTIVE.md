# IMPACT_OBJECTIVE — privileged-teacher impact objective (AIOBJ)

**Status:** training-only term in the staged teacher-v2 privileged stage.
Simulation data, CPU, no flight claim. No physics constant, gate, threshold,
command envelope or weight from another lane was changed.

SPDX-License-Identifier: GPL-3.0-or-later

---

## 1. What it is

The privileged stage used to supervise only the recorded fin command from truth
slots 22..25 plus a pooled latent. The impact objective adds one differentiable
term that uses the teacher's *privileged* channel (true velocity) and two
already-measured sensor columns. Its loss is

```
L = mean_b[ ((a_lat_resid - u_lat)/g0)^2  +  (drag_est/g0) * gamma(u)^2 ]

gamma(u) = atan2(vz_next, hypot(vx_next, vy_next))       [rad, SI dimensionless]
vx_next  = vx - u_pitch * dt
vy_next  = vy + u_lat  * dt
vz_next  = vz
```

with `u = [u_lat, u_pitch] = command[..., 0:2]` in m/s², the teacher's fin
acceleration demand. The world-frame mapping `[ax, ay] = [-u_pitch, u_lat]` is
the simulator's (`sim/core/src/optimizer.rs:127-137`); the one-step velocity
update is a bookkeeping kinematic, **not** a replacement for the simulator's
dynamics.

### Units (all explicit, none fitted)

| quantity | source | unit | role |
|---|---|---|---|
| `a_lat_resid` | window column 11, last frame | m/s² | lateral residual to be nulled (SPEC_SMART.md) |
| `drag_est` | window column 12, last frame | m/s² | `cd0·q_dyn·area/mass`, measured drag per mass |
| `u_lat`, `u_pitch` | teacher command | m/s² | fin acceleration demand |
| `v_x, v_y, v_z` | v3 truth slots 11, 12, 13 | m/s | privileged flight state |
| `g0` | `G0_MPS2 = 9.80665` | m/s² | standard gravity, physical constant |
| `dt` | `FRAME_DT_S = 0.02` | s | 50 Hz validation frame period |
| `gamma` | `atan2(v_z, v_h)` | rad | flight-path angle; dimensionless in SI |

Both squared terms are dimensionless. They are added with coefficient **1**
each. `g0` and `dt` are physical/measurement constants, not loss weights; there
are no learned or searched weights in this module, and no coefficient was fitted
to a closed-loop target. `drag_est/G0` is a *measured* scale on the angle term:
the penalty grows with both the flight-path excursion and the drag actually
incurred, which is the post-FIXPUMP energy-cost story (normal force does no
work; steering is paid for through drag). It is a drag proxy, not an induced
drag coefficient — no `e`, `AR` or `C_L` was invented.

## 2. Wiring

- `src/potok_ai/objective.py` (new) owns the constants and `ImpactObjective`.
- `train_teacher_v2.py`: `legacy_objective: bool = False` in `TeacherV2Config`;
  the privileged stage builds the term only when a truth sidecar builder with
  `records` is present. Each window's truth row is read at tick
  `window - 1 + window_offset`, using the same `_privileged_snapshot` loud
  checks as the rest of the privileged path (no zero fill, no label slots).
- `--legacy-objective` restores the supervised-only privileged stage. The
  active arm is recorded in `metrics.json` under
  `stages.privileged.details.objective` (`source`, `legacy`, `formula`, `reason`).
- When no truth sidecar records exist, the term cannot run and the details say
  `reason = "no_truth_sidecar_records"`; this is a counted, visible fallback,
  not a silent no-op. Legacy-mode reason is `"legacy_objective"`.
- Config hash includes the flag, so an impact run and a legacy run cannot be
  mistaken for one another.

## 3. Gates and falsification

- 9 new tests in `tests/test_impact_objective.py`: angle units and zero-speed
  refusal, exact unit-normalised loss, lateral cancellation by a matching
  command, command sensitivity and back-propagation, malformed/unphysical input
  refusals, loss hook only when wired, stage default vs `--legacy-objective`,
  no-sidecar fallback, CLI flag.
- Full `ai` suite: **423 passed**. `ruff check src tests`: **clean**.
- Default-on is conditional on those code gates staying green. They are green
  as of this document; if a later change makes them red, flip
  `legacy_objective` back to `True` before shipping rather than weakening a
  test.
- The closed-loop A/B gate is **not** passed by this change and was not re-run:
  AILIMIT records 6 FAIL / 5 PASS and a ≥1211–1217 m cap-10 physical bound.
  No claim is made that the impact objective closes that gap.

## 4. Honest limits

- No new model was trained with this term for a closed-loop A/B; this lands a
  training term and its gates, not a measured improvement. The −35 % / +15 %
  target was not searched, tuned or weakened.
- `alpha_deg` in SPEC_SMART.md is `atan2(omega_body_z, v_body_x)`, a documented
  proxy, not classical AoA; this objective does not use it. `drag_est` is the
  cd0-based drag proxy, not a separated induced-drag measurement.
- The one-step kinematic uses a constant `dt` from the validation header; a
  50 Hz frame with variable integration or a different dataset rate needs its
  own `ImpactObjectiveConfig(frame_dt_s=...)`, not a silent assumption.
- A privileged truth channel is unavailable at the edge. The student still sees
  14 sensor features + 6 conditioning values only; nothing here extends its
  input path.
- All evidence is simulator data from the same generator that flies the A/B;
  there is no flight log, no canopy/deployment event, and no flight benefit is
  claimed.
