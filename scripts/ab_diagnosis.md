# Read-only diagnosis of the negative closed-loop A/B (run4)

This does NOT change the negative result in `ab_report.md`. No gate, physics, gain or threshold was changed; nothing in `ai/` or `sim/` was edited.

## Decision table

| hypothesis | evidence | verdict |
|---|---|---|
| student input order/scale differs from training | 20-value frame = `frames[:, :14] + tile(conditioning)`; max abs diff vs `engine._smart_frames` and local vs `physics.py` baseline = 0 | REFUTED |
| A/B runtime differs from `student.py` | independent reimplementation of `step()` vs `export.to_model()`: max abs diff 0 over 48 frames | REFUTED |
| model correction is the cause | oracle g_theta through the same gated fin path: lateral 1528.366 -> 1530.021 m, apogee delta -5.239 m | REFUTED (oracle also hurts)
| command/fin mapping or actuator authority is the cause | every scale stays worse than zero command (1528.869..1528.589 vs 1528.366 m); lateral decreases monotonically as the correction grows, yet never wins | SUPPORTED |
| confidence threshold rejects AI-active ticks | raw gate rejects loopwm 0.00%, lrs 0.00%, lite 0.00%; lite temperature map rejects 4.04% | not the cause |

## 1. Oracle g_theta (seed 1, 20 episodes)

baseline lateral RMS 1528.366295 m, apogee 1734.977 m.

| seed | oracle lateral RMS (m) | Δ vs baseline | oracle apogee (m) | Δ apogee |
|---:|---:|---:|---:|---:|
| 1 | 1530.021305 | +1.655010 | 1729.738 | -5.239 |
| 2 | 1532.558369 | +1.601391 | 1723.132 | -5.127 |
| 3 | 1529.866155 | +1.692150 | 1740.031 | -5.373 |

g_theta is already in radians; feeding it in degrees (`×180/π`) is the same fin path as the student's normalised command (`×0.12 rad ×180/π`). The same `ai_healthy` gate and ±10° seam clip are applied. No g_theta value reaches the clip (its own clamp is 0.12 rad ≈ 6.88°).

## 2. Diagnostic scale sweep — NOT a tuning exercise

Scale multiplies only the student's Δ; `command = physics_baseline + scale·Δ`. Scale 0 is therefore the physics-baseline-only arm, NOT the zero-command baseline. Same 20 episodes per seed, same gate, same clip.

| scale | seed 1 lateral / apogee | seed 2 lateral / apogee | seed 3 lateral / apogee |
|---:|---:|---:|---:|
| 0 | 1528.8694 / 1733.37 | 1531.4590 / 1726.67 | 1528.7229 / 1743.65 |
| 0.25 | 1528.8326 / 1733.48 | 1531.4223 / 1726.78 | 1528.6850 / 1743.77 |
| 0.5 | 1528.7961 / 1733.59 | 1531.3860 / 1726.89 | 1528.6475 / 1743.88 |
| 1 | 1528.7254 / 1733.81 | 1531.3190 / 1727.10 | 1528.5768 / 1744.11 |
| 2 | 1528.5891 / 1734.25 | 1531.1908 / 1727.51 | 1528.4411 / 1744.55 |

Reading: scale 0 (physics baseline alone) is +0.503 m worse than zero command; larger scale moves monotonically toward baseline (+0.223 m at scale 2) but never wins. The student correction is partially cancelling the P/D baseline, not beating zero command.

## 3. Feature and runtime verification

- Our frame: `frames[:, :14]` then `tile(conditioning, 20/6)` in float32, identical to `ai/src/potok_ai/engine.py:339-346`. Max abs diff 0 over 3 episodes, including the local P/D baseline vs `ai/src/potok_ai/physics.py:37`.
- Sim writer order: `sim/core/src/validation.rs:294` writes `Conditioning::as_array()` = mass, area, cd0, static margin, thrust peak, gyro noise (`sim/core/src/config.rs:298-309`); the AI reader takes exactly those six at `ai/src/potok_ai/data.py:122`.
- Frame columns are the 8 configured raw channels (`ai/out/run4/sim.toml [output].features`) plus 6 physics channels (`sim/core/src/physics.rs:14-21`); `ai/src/potok_ai/physics.py:32-35` reads alpha=8, beta=9, a_lat=11, which is the same layout.
- Runtime: scripts/ab_control contains NO student inference; it only drives the simulator. The A/B uses export.to_model() from ai/src/potok_ai/export.py; this compares an independent reimplementation of student.py::step to that path. Max abs diff 0.

## 4. Confidence gate

| model | method | threshold | AI-active ticks | raw reject % | calibrated reject % |
|---|---|---:|---:|---:|---:|
| loopwm | isotonic | 0 | 18404 | 0.00 | n/a (isotonic map not serialised) |
| lrs | isotonic | 0.313924 | 18404 | 0.00 | n/a (isotonic map not serialised) |
| lite | temperature | 0.011412 | 18404 | 0.00 | 4.04 |

The isotonic maps are not in `calibration.json` (the PAVA fitter does not serialise them, `edge/src/calibration.rs:11-19`); the A/B therefore applies each threshold to raw sigmoid confidence. For loopwm the threshold is 0, so no map can change the gate. For lite (temperature 0.125) the calibrated rejection column is exact and the map changes nothing materially.

## Single most likely root cause

**The command→fin mapping, not the model.** The exact oracle g_theta is the WORST arm (seed 1 lateral +1.655 m, apogee -5.239 m), while the physics-baseline-only arm (scale 0) is also worse than zero command. Increasing the student correction moves lateral RMS monotonically back toward the zero-command baseline but never crosses it: the student's Δ is mostly cancelling an already-harmful P/D baseline, not adding a useful correction. The student input layout and runtime match training exactly, and the gate rejects no material share of active ticks. The reference optimizer (`sim/core/src/optimizer.rs:51-90`) was written for a full-state gimbal-style controller; the seam applies a fin moment (`sim/core/src/dynamics.rs:439-445`) whose authority/sign around the near-vertical 30 s trajectory does not reduce the wind-driven lateral drift `lateral_rms_m` measures. This is a controller/actuator design question, not a student-training question, and is out of scope for this read-only diagnosis.
