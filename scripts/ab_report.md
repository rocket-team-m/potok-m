# PID vs AI assistant — OR-aligned held-out wind/gust scenarios

**This is a deterministic PROXY A/B, not a flown flight test.** The simulator never applies a command to its dynamics (`sim/core/src/episode.rs:233` records `g_theta` as a target, `:257` integrates with unchanged `Vehicle`); validation v2 stores no per-tick truth position/apogee. Real apogee gain is therefore **not measurable** without changing `sim/`. No physics was invented: the proxy uses only recorded commands, recorded baro height, the recorded frame rate, `g = 9.80665` (`sim/core/src/dynamics.rs:437`) and the reference clamp `0.12 rad` (`sim/core/src/optimizer.rs:41`).

## Scope

- seed: **1** (fixed, as required); 3 seeds are **not** used: the data is generated at one fixed seed and the students were trained at that seed. The 4 held-out scenarios are the independent cells; 20 episodes total, no cherry-picking.
- train: 80 episodes / 16 scenarios; held-out: 20 episodes / 4 scenarios (scenario_16, scenario_17, scenario_18, scenario_19).
- simulator reproducibility: byte-identical regeneration confirmed (sha256 `59b0ed4dfaf8287eb16278a916aa03ecf9094711fa49df430646dd8968eaac1c`).
- students: `ai/out/run2/loopwm/loopwm_student.bin` (sha256 419725a92310, g_max 1.2223); `ai/out/run2/lrs/lrs.bin` (sha256 4507be92543c, g_max 1.3626)

## Controllers and proxy definition

- **baseline (AI off)**: `b(x) = physics_baseline(frame)` from `ai/src/potok_ai/physics.py:37`, the frame-local P/D command used by the onboard deployment contract. The simulator's `g_theta` is a distillation *target* computed with privileged full state, not a command sent to the actuator path; adding the student's correction to `g_theta` would not be the flown controller, so it is not used as the baseline. `g_theta` is used only for the `tracking_rmse_norm` diagnostic below.
- **assisted (AI on)**: `b(x) + Delta`, where `Delta` comes from the exported stateful student replayed tick-by-tick with `reset_state()` per episode (canonical Python forward, same as `edge/src/student_runtime.rs`).
- **gate**: reference `ai_healthy` flag AND calibrated confidence >= threshold (loopwm 0.039683, lrs 0.000123). The isotonic map is not stored in calibration.json, so the threshold is applied to the raw sigmoid confidence: an upper bound on gate openness. Gate trips are counted over all ticks.
- **surrogate dynamics**: only on gate-open ticks, with zero initial velocity and displacement: `tilt = command * 0.12 rad`, `a_lat = 9.80665 * tilt_roll`, `a_zloss = 0.5 * 9.80665 * |tilt|^2`, integrated twice by trapezoid at the recorded 50 Hz frame rate. Boost/interlock ticks contribute gate trips but no surrogate dynamics (the reference interlocks them to zero).
- **apogee_m (proxy)**: `max(baro_h_t - zloss_t)` over gate-open ticks, i.e. the recorded biased/noisy barometric altitude minus the surrogate tilt-induced vertical loss at the same tick. `apogee_retention_pct = 100 * assisted / baseline`.
- **lateral_rms_m (proxy)**: RMS of the surrogate lateral displacement over gate-open ticks. **max |g| correction**: largest absolute student correction actually applied (gate-open ticks). **interventions**: gate-open ticks with |Delta| > 1e-6. **tracking_rmse_norm**: RMS of `command - g_theta` in the reference command's normalised units (diagnostic, not metres).

## Student: `loopwm`

| scenario | n ep | gate-open / gate-trip | interventions | max abs Δg | baseline apogee (m) | assisted apogee (m) | retention % | baseline lateral RMS (m) | assisted lateral RMS (m) | tracking RMSE b / b+Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 89 / 2911 | 89 | 0.8311 | 1277.279059 | 1277.283345 | 100.000335 | 0.04002 | 0.00676 | 0.9361 / 0.7094 |
| scenario_17 | 5 | 95 / 2905 | 95 | 0.8311 | 1404.434223 | 1404.438814 | 100.000327 | 0.03874 | 0.00654 | 0.9458 / 0.7080 |
| scenario_18 | 5 | 96 / 2904 | 96 | 0.8311 | 1192.021755 | 1192.026342 | 100.000385 | 0.04188 | 0.00707 | 0.9368 / 0.7093 |
| scenario_19 | 5 | 121 / 2879 | 121 | 0.8311 | 1327.282438 | 1327.290509 | 100.000608 | 0.07593 | 0.01282 | 0.9513 / 0.7072 |
| **ALL (20 ep)** | — | 401 / 11599 | 401 | 0.8311 | 1300.254369 | 1300.259752 | 100.000414 | 0.05153 | 0.00870 | 0.9425 / 0.7085 |

## Student: `lrs`

| scenario | n ep | gate-open / gate-trip | interventions | max abs Δg | baseline apogee (m) | assisted apogee (m) | retention % | baseline lateral RMS (m) | assisted lateral RMS (m) | tracking RMSE b / b+Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 89 / 2911 | 89 | 0.8770 | 1277.279059 | 1277.283503 | 100.000348 | 0.04002 | 0.00492 | 0.9361 / 0.6995 |
| scenario_17 | 5 | 95 / 2905 | 95 | 0.8770 | 1404.434223 | 1404.438894 | 100.000333 | 0.03874 | 0.00477 | 0.9458 / 0.7116 |
| scenario_18 | 5 | 96 / 2904 | 96 | 0.8770 | 1192.021755 | 1192.026650 | 100.000411 | 0.04188 | 0.00515 | 0.9368 / 0.7005 |
| scenario_19 | 5 | 121 / 2879 | 121 | 0.8770 | 1327.282438 | 1327.290750 | 100.000626 | 0.07593 | 0.00934 | 0.9513 / 0.7186 |
| **ALL (20 ep)** | — | 401 / 11599 | 401 | 0.8770 | 1300.254369 | 1300.259949 | 100.000429 | 0.05153 | 0.00634 | 0.9425 / 0.7076 |

## Verdict

Primary model `loopwm` (the non-degenerate loopwm student): assisted beats baseline on apogee retention (100.000414 %, delta 0.005383 m, YES) and on lateral RMS (16.887695 %, YES). Overall: **PASS** (exit 0).

Magnitude check: the surrogate apogee win is 0.005383 m, far below the 0.35 m baro noise floor — it is a tie in physical terms. The larger, still proxy, signal is command tracking: g_theta RMSE 0.9425 -> 0.7085 (24.83% lower) and surrogate lateral RMS 0.05153 -> 0.00870 m, both measured over only 401 gate-open ticks.

## Limitations (do not read this as a flight result)

- The command is never applied to the RK4 state; both arms are scored on the same recorded trajectory, so the apogee/lateral numbers are surrogates, not measurements of a controlled flight.
- Only 401 gate-open ticks exist in the whole held-out set for the primary student (out of 12000 recorded ticks): the AI interlock is closed for almost all of the 12 s scenario because the boost phase lasts ~9.9 s. Any advantage is measured on that short tail.
- The baro height is biased (-0.4 m) and noisy (0.35 m); it is the same recorded signal for both arms, so comparisons are paired, but absolute apogee is not the simulator's truth `apogee_m`.
- The calibration isotonic map is not exported; the threshold is applied to raw sigmoid confidence, so gate openings here are an upper bound.
- 3-seed replication is unavailable by construction (fixed generation seed and seed-1-trained students); the 4 held-out scenarios, 20 episodes, are the replication units. Seeds 2 and 3 were not generated — no seed was cherry-picked.
