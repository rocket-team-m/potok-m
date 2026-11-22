# PID (zero command) vs AI assistant — closed-loop OR-aligned A/B

**FINAL run4 report (full 30 s flights). It SUPERSEDES both earlier reports**: the 12 s closed-loop A/B (commit `42df574`) and the original gravity/tilt surrogate A/B (commit `237920a`). Those files are void, not a smaller version of these results.

Apogee and lateral RMS are now measured directly from the flown-truth `validation.trajectory` sidecar (POTOKTRJ v1, per-tick x/y/z/vx/vy/vz)
produced by `potok-sim`'s command→dynamics seam (`Simulator::run_episode_with_control`, `sim/core/src/episode.rs:130`).
No sim source or config was changed; `scripts/ab_control/` is a read-only driver because the CLI's `--control-file` applies one TSV to every episode.

## Protocol

- Seed 1: 20 OR-aligned held-out episodes (indices 16..19,36..39,56..59,76..79,96..99), 4 scenarios, 1500 frames each, from `ai/out/run4/sim.toml` (30 s, 50 Hz). Baseline validation is byte-identical to `ai/out/run4/data/heldout_validation.bin`.
- Seeds 2 and 3: same 20 indices/scenarios, generated with `--seed 2/3`; the exports were NOT retrained, so these are off-training generalization checks, not a seed sweep of the training procedure.
- Baseline arm: no control file, zero command on every tick.
- Assisted arm: per-tick `command = physics_baseline(frame) + model Delta`. `loopwm`/`lrs` use the exported v2 stateful students with their trained forward (`load_student_export(...).to_model()`). `lite` uses the exported windowed torch model over the last 50 frames (no command before tick 50). A command is emitted only where `ai_healthy` (AI_READY, no FAULT, phase interlock allows) is set AND the model's confidence clears its threshold.
- Calibration: the v2 export binary has NO calibration field; thresholds are read from each run4 `calibration.json` (loopwm 0, lrs 0.313924, lite 0.011412). Nothing is hardcoded.
- Unit mapping: normalised command × 0.12 rad × 180/π = degrees (the reference optimizer's ±0.12 rad clamp, same scale as the old surrogate). Commands beyond the seam's ±10° actuator limit are clipped and counted (`clipped_ticks`); no physics or gains were tuned.
- `interventions` counts gate-open ticks with a non-zero applied command; `gate_trips` counts ticks with the interlock closed. Both are summed over episodes. `tracking_rmse_norm` is the assisted command's RMS error against g_theta on gate-open ticks (diagnostic only).
- Exit status: 0 iff the primary assisted arm reduces lateral RMS on seed 1; nonzero otherwise. Apogee deltas are reported even when tiny or negative.

## Verdict (primary: loopwm student, seed 1)

- lateral RMS 1528.366295 m → 1528.725432 m (0.359136 m, 100.023498 % of baseline): DID NOT BEAT baseline.
- apogee 1734.977 m → 1733.811 m (-1.1666 m, 99.93276 % retention).
- overall **FAIL** (exit 1).

## Seed 1 · student `loopwm` (primary)

| scenario | ep | baseline apogee (m) | assisted apogee (m) | Δ apogee (m) | retention % | baseline lateral RMS (m) | assisted lateral RMS (m) | Δ lateral (m) | lateral % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 1667.467 | 1666.334 | -1.1327 | 99.93207 | 1532.164899 | 1532.497530 | 0.332630 | 100.021710 |
| scenario_17 | 5 | 1962.089 | 1960.905 | -1.1835 | 99.93968 | 1482.130886 | 1482.535198 | 0.404311 | 100.027279 |
| scenario_18 | 5 | 1495.338 | 1494.433 | -0.9047 | 99.93950 | 1599.462601 | 1599.688570 | 0.225969 | 100.014128 |
| scenario_19 | 5 | 1815.016 | 1813.570 | -1.4456 | 99.92035 | 1497.027814 | 1497.512261 | 0.484447 | 100.032361 |
| **ALL (20 ep)** | — | 1734.977 | 1733.811 | -1.1666 | 99.93276 | 1528.366295 | 1528.725432 | 0.359136 | 100.023498 |

interventions 18404 / 30000 ticks; gate trips 11596; clipped ticks 20; max applied command 10.000°; g_theta tracking RMSE 0.8640. Trajectory sha256: baseline `a289e63e8a2e110f4f41029ce15afd5353d33fd10bcc3c41bebbdadd699c147f`, assisted `28535ae0e4f04357e874101ee4e808e0ec418c5e99c88602ae36526419e75ac7`.

## Seed 1 · student `lrs`

| scenario | ep | baseline apogee (m) | assisted apogee (m) | Δ apogee (m) | retention % | baseline lateral RMS (m) | assisted lateral RMS (m) | Δ lateral (m) | lateral % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 1667.467 | 1667.275 | -0.1917 | 99.98851 | 1532.164899 | 1532.226651 | 0.061752 | 100.004030 |
| scenario_17 | 5 | 1962.089 | 1961.876 | -0.2124 | 99.98917 | 1482.130886 | 1482.211534 | 0.080648 | 100.005441 |
| scenario_18 | 5 | 1495.338 | 1495.183 | -0.1547 | 99.98965 | 1599.462601 | 1599.506629 | 0.044028 | 100.002753 |
| scenario_19 | 5 | 1815.016 | 1814.782 | -0.2337 | 99.98712 | 1497.027814 | 1497.111959 | 0.084145 | 100.005621 |
| **ALL (20 ep)** | — | 1734.977 | 1734.779 | -0.1981 | 99.98858 | 1528.366295 | 1528.433448 | 0.067152 | 100.004394 |

interventions 18404 / 30000 ticks; gate trips 11596; clipped ticks 20; max applied command 10.000°; g_theta tracking RMSE 0.1722. Trajectory sha256: baseline `a289e63e8a2e110f4f41029ce15afd5353d33fd10bcc3c41bebbdadd699c147f`, assisted `2459c73110e0833ec0958cfaa6f7d3479df205d8a2bf8b24b87f96725a0d4843`.

## Seed 1 · student `lite`

| scenario | ep | baseline apogee (m) | assisted apogee (m) | Δ apogee (m) | retention % | baseline lateral RMS (m) | assisted lateral RMS (m) | Δ lateral (m) | lateral % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 1667.467 | 1666.983 | -0.4836 | 99.97100 | 1532.164899 | 1532.322517 | 0.157617 | 100.010287 |
| scenario_17 | 5 | 1962.089 | 1961.627 | -0.4621 | 99.97645 | 1482.130886 | 1482.304731 | 0.173844 | 100.011729 |
| scenario_18 | 5 | 1495.338 | 1494.909 | -0.4287 | 99.97133 | 1599.462601 | 1599.590229 | 0.127628 | 100.007979 |
| scenario_19 | 5 | 1815.016 | 1814.495 | -0.5206 | 99.97132 | 1497.027814 | 1497.215284 | 0.187470 | 100.012523 |
| **ALL (20 ep)** | — | 1734.977 | 1734.504 | -0.4737 | 99.97269 | 1528.366295 | 1528.527242 | 0.160946 | 100.010531 |

interventions 18404 / 30000 ticks; gate trips 11596; clipped ticks 16; max applied command 10.000°; g_theta tracking RMSE 0.1691. Trajectory sha256: baseline `a289e63e8a2e110f4f41029ce15afd5353d33fd10bcc3c41bebbdadd699c147f`, assisted `f514750cff4a1476162f40ad719f6d49171c5dca3cbf12916dfb71600f781cda`.

## Seed 2 · student `loopwm` (off-training)

| scenario | ep | baseline apogee (m) | assisted apogee (m) | Δ apogee (m) | retention % | baseline lateral RMS (m) | assisted lateral RMS (m) | Δ lateral (m) | lateral % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 1688.827 | 1687.814 | -1.0129 | 99.94003 | 1523.754604 | 1524.048726 | 0.294122 | 100.019302 |
| scenario_17 | 5 | 1838.721 | 1837.479 | -1.2425 | 99.93243 | 1477.782734 | 1478.208964 | 0.426230 | 100.028843 |
| scenario_18 | 5 | 1573.459 | 1572.446 | -1.0135 | 99.93559 | 1618.990980 | 1619.256863 | 0.265883 | 100.016423 |
| scenario_19 | 5 | 1812.026 | 1810.643 | -1.3827 | 99.92369 | 1499.505046 | 1499.977357 | 0.472310 | 100.031498 |
| **ALL (20 ep)** | — | 1728.258 | 1727.096 | -1.1629 | 99.93271 | 1530.956978 | 1531.318967 | 0.361989 | 100.023645 |

interventions 18359 / 30000 ticks; gate trips 11641; clipped ticks 70; max applied command 10.000°; g_theta tracking RMSE 0.8673. Trajectory sha256: baseline `882199ae15bd29320a230ac5cad71736b954e0659920da8c2a4ef1b115873fa8`, assisted `2281ef0c27f4c61c157f37fafe16da5cc1995f28bee44735c5650e266cdf2546`.

## Seed 2 · student `lrs` (off-training)

| scenario | ep | baseline apogee (m) | assisted apogee (m) | Δ apogee (m) | retention % | baseline lateral RMS (m) | assisted lateral RMS (m) | Δ lateral (m) | lateral % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 1688.827 | 1688.638 | -0.1895 | 99.98878 | 1523.754604 | 1523.817456 | 0.062853 | 100.004125 |
| scenario_17 | 5 | 1838.721 | 1838.499 | -0.2223 | 99.98791 | 1477.782734 | 1477.866667 | 0.083933 | 100.005680 |
| scenario_18 | 5 | 1573.459 | 1573.290 | -0.1697 | 99.98922 | 1618.990980 | 1619.040436 | 0.049456 | 100.003055 |
| scenario_19 | 5 | 1812.026 | 1811.791 | -0.2348 | 99.98704 | 1499.505046 | 1499.589913 | 0.084867 | 100.005660 |
| **ALL (20 ep)** | — | 1728.258 | 1728.054 | -0.2041 | 99.98819 | 1530.956978 | 1531.026728 | 0.069750 | 100.004556 |

interventions 18359 / 30000 ticks; gate trips 11641; clipped ticks 72; max applied command 10.000°; g_theta tracking RMSE 0.2078. Trajectory sha256: baseline `882199ae15bd29320a230ac5cad71736b954e0659920da8c2a4ef1b115873fa8`, assisted `a533af4b376d72e569c21216712db6e34a02dfa496e5d9077005e6c46c01a7ab`.

## Seed 2 · student `lite` (off-training)

| scenario | ep | baseline apogee (m) | assisted apogee (m) | Δ apogee (m) | retention % | baseline lateral RMS (m) | assisted lateral RMS (m) | Δ lateral (m) | lateral % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 1688.827 | 1688.456 | -0.3714 | 99.97801 | 1523.754604 | 1523.870612 | 0.116008 | 100.007613 |
| scenario_17 | 5 | 1838.721 | 1838.362 | -0.3592 | 99.98047 | 1477.782734 | 1477.915759 | 0.133025 | 100.009002 |
| scenario_18 | 5 | 1573.459 | 1573.033 | -0.4264 | 99.97290 | 1618.990980 | 1619.117270 | 0.126290 | 100.007801 |
| scenario_19 | 5 | 1812.026 | 1811.515 | -0.5115 | 99.97177 | 1499.505046 | 1499.690742 | 0.185695 | 100.012384 |
| **ALL (20 ep)** | — | 1728.258 | 1727.841 | -0.4171 | 99.97586 | 1530.956978 | 1531.096804 | 0.139825 | 100.009133 |

interventions 18359 / 30000 ticks; gate trips 11641; clipped ticks 61; max applied command 10.000°; g_theta tracking RMSE 0.1905. Trajectory sha256: baseline `882199ae15bd29320a230ac5cad71736b954e0659920da8c2a4ef1b115873fa8`, assisted `328e083f202c960a3327d7e4cc4a3a0f3be3eb6b4db243c282fcacf3b860110a`.

## Seed 3 · student `loopwm` (off-training)

| scenario | ep | baseline apogee (m) | assisted apogee (m) | Δ apogee (m) | retention % | baseline lateral RMS (m) | assisted lateral RMS (m) | Δ lateral (m) | lateral % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 1728.172 | 1726.744 | -1.4283 | 99.91735 | 1517.289008 | 1517.725559 | 0.436551 | 100.028772 |
| scenario_17 | 5 | 1817.707 | 1816.317 | -1.3898 | 99.92354 | 1474.799609 | 1475.282165 | 0.482555 | 100.032720 |
| scenario_18 | 5 | 1632.301 | 1631.133 | -1.1685 | 99.92842 | 1619.609587 | 1619.925086 | 0.315499 | 100.019480 |
| scenario_19 | 5 | 1803.436 | 1802.234 | -1.2027 | 99.93331 | 1496.972985 | 1497.358313 | 0.385328 | 100.025740 |
| **ALL (20 ep)** | — | 1745.404 | 1744.107 | -1.2973 | 99.92567 | 1528.174005 | 1528.576751 | 0.402747 | 100.026355 |

interventions 18385 / 30000 ticks; gate trips 11615; clipped ticks 79; max applied command 10.000°; g_theta tracking RMSE 0.8695. Trajectory sha256: baseline `32ba089073d24461ae599a3aac73c729b81b59bf147ae81d354bb9c1352b7a43`, assisted `8e4a793cd78255232bbe657efc45ce3dc3266419de9856dc634bff4242e9f690`.

## Seed 3 · student `lrs` (off-training)

| scenario | ep | baseline apogee (m) | assisted apogee (m) | Δ apogee (m) | retention % | baseline lateral RMS (m) | assisted lateral RMS (m) | Δ lateral (m) | lateral % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 1728.172 | 1727.934 | -0.2382 | 99.98622 | 1517.289008 | 1517.367869 | 0.078860 | 100.005197 |
| scenario_17 | 5 | 1817.707 | 1817.469 | -0.2385 | 99.98688 | 1474.799609 | 1474.888178 | 0.088569 | 100.006005 |
| scenario_18 | 5 | 1632.301 | 1632.110 | -0.1913 | 99.98828 | 1619.609587 | 1619.666566 | 0.056979 | 100.003518 |
| scenario_19 | 5 | 1803.436 | 1803.216 | -0.2207 | 99.98776 | 1496.972985 | 1497.051287 | 0.078302 | 100.005231 |
| **ALL (20 ep)** | — | 1745.404 | 1745.182 | -0.2222 | 99.98727 | 1528.174005 | 1528.249221 | 0.075216 | 100.004922 |

interventions 18385 / 30000 ticks; gate trips 11615; clipped ticks 79; max applied command 10.000°; g_theta tracking RMSE 0.2132. Trajectory sha256: baseline `32ba089073d24461ae599a3aac73c729b81b59bf147ae81d354bb9c1352b7a43`, assisted `4fe683ebf33a9222ee8ab20f55c41a7dd8b0cf3c1fd101ab05c47acb13e0c065`.

## Seed 3 · student `lite` (off-training)

| scenario | ep | baseline apogee (m) | assisted apogee (m) | Δ apogee (m) | retention % | baseline lateral RMS (m) | assisted lateral RMS (m) | Δ lateral (m) | lateral % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 1728.172 | 1727.622 | -0.5498 | 99.96819 | 1517.289008 | 1517.471401 | 0.182392 | 100.012021 |
| scenario_17 | 5 | 1817.707 | 1817.264 | -0.4430 | 99.97563 | 1474.799609 | 1474.968878 | 0.169269 | 100.011477 |
| scenario_18 | 5 | 1632.301 | 1631.850 | -0.4510 | 99.97237 | 1619.609587 | 1619.746798 | 0.137211 | 100.008472 |
| scenario_19 | 5 | 1803.436 | 1803.076 | -0.3599 | 99.98004 | 1496.972985 | 1497.101032 | 0.128046 | 100.008554 |
| **ALL (20 ep)** | — | 1745.404 | 1744.953 | -0.4509 | 99.97417 | 1528.174005 | 1528.327831 | 0.153826 | 100.010066 |

interventions 18385 / 30000 ticks; gate trips 11615; clipped ticks 71; max applied command 10.000°; g_theta tracking RMSE 0.1950. Trajectory sha256: baseline `32ba089073d24461ae599a3aac73c729b81b59bf147ae81d354bb9c1352b7a43`, assisted `aec2f997ef33666d7726ad15488176987e4133edf458e66a1e13e235cfe91bae`.

## Honesty notes

- Commands are replayed open-loop from the baseline run's recorded frames; the control TSV is indexed by frame, as the simulator seam requires. Once the assisted command moves the vehicle, later frames differ from the baseline run — that is the point of the closed-loop comparison, but the interlock schedule stays the paired baseline one.
- Lateral RMS is over the whole 30 s flown trajectory, not just the AI-active tail; the earlier 12 s dataset's tiny active window no longer limits the measurement.
- Seeds 2/3 use the seed-1-trained exports; they test input robustness, not training reproducibility. No seed, scenario or episode was dropped.
- The calibration `isotonic` map itself is not exported in calibration.json; only its threshold is available, and it is applied to the raw confidence. This is an upper bound on how often the gate opens.
- `clipped_ticks` is the seam's hard ±10° actuator bound, not a tuning knob.
- Measurement floor: the sidecar stores f32 positions. One ULP is about 1.22e-04 m at the measured lateral scale and 1.22e-04 m at the measured altitude. Compare the deltas above against that floor before calling any difference a win.