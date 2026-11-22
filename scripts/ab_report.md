# PID (zero command) vs AI assistant — closed-loop OR-aligned A/B

**FINAL run4 report (full 30 s flights). It SUPERSEDES both earlier reports**: the 12 s closed-loop A/B (commit `42df574`) and the original gravity/tilt surrogate A/B (commit `237920a`). Those files are void, not a smaller version of these results.

Apogee and lateral RMS are now measured directly from the flown-truth `validation.trajectory` sidecar (POTOKTRJ v1, per-tick x/y/z/vx/vy/vz)
produced by `potok-sim`'s command→dynamics seam (`Simulator::run_episode_with_control`, `sim/core/src/episode.rs:155`).
Commands are desired lateral accelerations in m/s², resolved to fin deflection by `dynamics::solve_fin_deflection`. No sim source or config was changed; `scripts/ab_control/` is a read-only driver because the CLI's `--control-file` applies one TSV to every episode.
The earlier `ab_diagnosis.md` root-cause record used the superseded degree-valued seam; it is kept as history, not as the current contract.

## Protocol

- Seed 1: 20 OR-aligned held-out episodes (indices 16..19,36..39,56..59,76..79,96..99), 4 scenarios, 1500 frames each, from `ai/out/run4/sim.toml` (30 s, 50 Hz). Baseline validation is byte-identical to `ai/out/run4/data/heldout_validation.bin`.
- Seeds 2 and 3: same 20 indices/scenarios, generated with `--seed 2/3`; the exports were NOT retrained, so these are off-training generalization checks, not a seed sweep of the training procedure.
- Baseline arm: no control file, zero command on every tick.
- Positive control `oracle_gtheta`: `sim.run_episode_with_oracle` calls `ReferenceOptimizer::fin_accel_command` on live full state/wind/phase (`sim/core/src/optimizer.rs:98-113`); no degree or acceleration conversion is done by this harness for that arm.
- Assisted arm: per-tick `command = physics_baseline(frame) + model Delta`. `loopwm`/`lrs` use the exported v2 stateful students with their trained forward (`load_student_export(...).to_model()`). `lite` uses the exported windowed torch model over the last 50 frames (no command before tick 50). A command is emitted only where `ai_healthy` (AI_READY, no FAULT, phase interlock allows) is set AND the model's confidence clears its threshold.
- Calibration: the v2 export binary has NO calibration field; thresholds are read from each run4 `calibration.json` (loopwm 0, lrs 0.313924, lite 0.011412). Nothing is hardcoded.
- Student conversion: the command is in `physics_baseline` units (normalised angle error); `ai/src/potok_ai/physics.py:27-28` divides angle errors by BETA0=ALPHA0=10°, so one unit is `g·tan(10°) = 1.729177 m/s²` with g=9.80665. TSV values are `(b+Δ) × 1.729177` m/s², clipped at the seam's ±10 m/s² (`clipped_ticks`). This scale comes from the AI baseline's documented units, not from an A/B fit.
- `interventions` counts gate-open ticks with a non-zero applied command; `gate_trips` counts ticks with the interlock closed. Both are summed over episodes. `tracking_rmse_norm` is the assisted command's RMS error against g_theta on gate-open ticks (diagnostic only).
- Exit status: 0 iff the primary assisted arm reduces lateral RMS on seed 1; nonzero otherwise. Apogee deltas are reported even when tiny or negative.

## Verdict (primary: loopwm student, seed 1)

- lateral RMS 1528.366295 m → 1542.361946 m (13.995650 m, 100.915726 % of baseline): DID NOT BEAT baseline.
- apogee 1734.977 m → 1691.241 m (-43.7361 m, 97.47916 % retention).
- positive control oracle: lateral 1528.366295 -> 1035.227516 m (-493.138780 m), apogee 2236.2201 m.
- overall **FAIL** (exit 1).

## Seed 1 · arm `oracle_gtheta` (positive control)

| scenario | ep | baseline apogee (m) | assisted apogee (m) | Δ apogee (m) | retention % | baseline lateral RMS (m) | assisted lateral RMS (m) | Δ lateral (m) | lateral % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 1667.467 | 3964.762 | 2297.2953 | 237.77158 | 1532.164899 | 1057.578223 | -474.586677 | 69.025091 |
| scenario_17 | 5 | 1962.089 | 4064.499 | 2102.4098 | 207.15161 | 1482.130886 | 939.865181 | -542.265705 | 63.413103 |
| scenario_18 | 5 | 1495.338 | 3830.443 | 2335.1047 | 256.15897 | 1599.462601 | 1156.620621 | -442.841980 | 72.313077 |
| scenario_19 | 5 | 1815.016 | 4025.086 | 2210.0705 | 221.76592 | 1497.027814 | 973.239190 | -523.788624 | 65.011430 |
| **ALL (20 ep)** | — | 1734.977 | 3971.197 | 2236.2201 | 228.89045 | 1528.366295 | 1035.227516 | -493.138780 | 67.734254 |

interventions 18404 / 30000 ticks; gate trips 11596; clipped ticks n/a (sim-internal); max applied command n/a (sim-internal); g_theta tracking RMSE n/a. Trajectory sha256: baseline `a289e63e8a2e110f4f41029ce15afd5353d33fd10bcc3c41bebbdadd699c147f`, assisted `e1a63113749b763aaf4c04678f4788a4e09447e67c2d950389986bb7e08141fd`.

## Seed 1 · arm `loopwm` (primary)

| scenario | ep | baseline apogee (m) | assisted apogee (m) | Δ apogee (m) | retention % | baseline lateral RMS (m) | assisted lateral RMS (m) | Δ lateral (m) | lateral % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 1667.467 | 1626.693 | -40.7737 | 97.55475 | 1532.164899 | 1544.232225 | 12.067325 | 100.787600 |
| scenario_17 | 5 | 1962.089 | 1913.422 | -48.6672 | 97.51962 | 1482.130886 | 1500.567344 | 18.436458 | 101.243916 |
| scenario_18 | 5 | 1495.338 | 1463.883 | -31.4551 | 97.89646 | 1599.462601 | 1606.806752 | 7.344151 | 100.459164 |
| scenario_19 | 5 | 1815.016 | 1760.967 | -54.0483 | 97.02216 | 1497.027814 | 1515.697210 | 18.669396 | 101.247097 |
| **ALL (20 ep)** | — | 1734.977 | 1691.241 | -43.7361 | 97.47916 | 1528.366295 | 1542.361946 | 13.995650 | 100.915726 |

interventions 18404 / 30000 ticks; gate trips 11596; clipped ticks 0; max applied command 3.1737 m/s²; g_theta tracking RMSE 0.8640. Trajectory sha256: baseline `a289e63e8a2e110f4f41029ce15afd5353d33fd10bcc3c41bebbdadd699c147f`, assisted `82b57e59049c568199fdeee361c739d43be284e516f95cbdde7c2ce8128cb292`.

## Seed 1 · arm `lrs`

| scenario | ep | baseline apogee (m) | assisted apogee (m) | Δ apogee (m) | retention % | baseline lateral RMS (m) | assisted lateral RMS (m) | Δ lateral (m) | lateral % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 1667.467 | 1660.255 | -7.2118 | 99.56750 | 1532.164899 | 1534.233539 | 2.068639 | 100.135014 |
| scenario_17 | 5 | 1962.089 | 1953.918 | -8.1710 | 99.58355 | 1482.130886 | 1485.224622 | 3.093736 | 100.208736 |
| scenario_18 | 5 | 1495.338 | 1489.827 | -5.5114 | 99.63143 | 1599.462601 | 1600.762932 | 1.300331 | 100.081298 |
| scenario_19 | 5 | 1815.016 | 1805.888 | -9.1276 | 99.49711 | 1497.027814 | 1500.012123 | 2.984310 | 100.199349 |
| **ALL (20 ep)** | — | 1734.977 | 1727.472 | -7.5055 | 99.56740 | 1528.366295 | 1530.705970 | 2.339675 | 100.153083 |

interventions 18404 / 30000 ticks; gate trips 11596; clipped ticks 0; max applied command 3.3301 m/s²; g_theta tracking RMSE 0.1722. Trajectory sha256: baseline `a289e63e8a2e110f4f41029ce15afd5353d33fd10bcc3c41bebbdadd699c147f`, assisted `9d21192d6ea7fea69057e11cf27e7f5f11e61289cf31aaef64e5e89051933ee4`.

## Seed 1 · arm `lite`

| scenario | ep | baseline apogee (m) | assisted apogee (m) | Δ apogee (m) | retention % | baseline lateral RMS (m) | assisted lateral RMS (m) | Δ lateral (m) | lateral % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 1667.467 | 1649.370 | -18.0966 | 98.91472 | 1532.164899 | 1537.953370 | 5.788470 | 100.377797 |
| scenario_17 | 5 | 1962.089 | 1942.800 | -19.2887 | 99.01693 | 1482.130886 | 1489.425389 | 7.294503 | 100.492163 |
| scenario_18 | 5 | 1495.338 | 1480.182 | -15.1560 | 98.98645 | 1599.462601 | 1603.734428 | 4.271827 | 100.267079 |
| scenario_19 | 5 | 1815.016 | 1794.754 | -20.2621 | 98.88364 | 1497.027814 | 1504.161572 | 7.133758 | 100.476528 |
| **ALL (20 ep)** | — | 1734.977 | 1716.777 | -18.2008 | 98.95095 | 1528.366295 | 1534.450611 | 6.084315 | 100.398093 |

interventions 18404 / 30000 ticks; gate trips 11596; clipped ticks 0; max applied command 2.6473 m/s²; g_theta tracking RMSE 0.1691. Trajectory sha256: baseline `a289e63e8a2e110f4f41029ce15afd5353d33fd10bcc3c41bebbdadd699c147f`, assisted `d138c6d16cb9e41b5a7590346c040d5e3a7b7c53a542da4d98e823f53c307151`.

## Seed 2 · arm `oracle_gtheta` (positive control) (off-training)

| scenario | ep | baseline apogee (m) | assisted apogee (m) | Δ apogee (m) | retention % | baseline lateral RMS (m) | assisted lateral RMS (m) | Δ lateral (m) | lateral % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 1688.827 | 4034.599 | 2345.7722 | 238.89949 | 1523.754604 | 1051.584350 | -472.170254 | 69.012710 |
| scenario_17 | 5 | 1838.721 | 4027.571 | 2188.8494 | 219.04194 | 1477.782734 | 940.245124 | -537.537610 | 63.625397 |
| scenario_18 | 5 | 1573.459 | 3955.250 | 2381.7910 | 251.37289 | 1618.990980 | 1144.625515 | -474.365466 | 70.699932 |
| scenario_19 | 5 | 1812.026 | 3984.499 | 2172.4729 | 219.89192 | 1499.505046 | 985.501732 | -514.003315 | 65.721802 |
| **ALL (20 ep)** | — | 1728.258 | 4000.480 | 2272.2214 | 231.47463 | 1530.956978 | 1033.352768 | -497.604211 | 67.497179 |

interventions 18359 / 30000 ticks; gate trips 11641; clipped ticks n/a (sim-internal); max applied command n/a (sim-internal); g_theta tracking RMSE n/a. Trajectory sha256: baseline `882199ae15bd29320a230ac5cad71736b954e0659920da8c2a4ef1b115873fa8`, assisted `5a4101195fd1dee48d9f0eeb3b7b167799f21bbc58302b8d6ca40039e2677297`.

## Seed 2 · arm `loopwm` (off-training)

| scenario | ep | baseline apogee (m) | assisted apogee (m) | Δ apogee (m) | retention % | baseline lateral RMS (m) | assisted lateral RMS (m) | Δ lateral (m) | lateral % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 1688.827 | 1651.269 | -37.5583 | 97.77607 | 1523.754604 | 1535.945826 | 12.191222 | 100.800078 |
| scenario_17 | 5 | 1838.721 | 1789.244 | -49.4775 | 97.30913 | 1477.782734 | 1497.080950 | 19.298215 | 101.305890 |
| scenario_18 | 5 | 1573.459 | 1538.338 | -35.1212 | 97.76790 | 1618.990980 | 1628.302959 | 9.311979 | 100.575172 |
| scenario_19 | 5 | 1812.026 | 1761.637 | -50.3889 | 97.21919 | 1499.505046 | 1517.160773 | 17.655727 | 101.177437 |
| **ALL (20 ep)** | — | 1728.258 | 1685.122 | -43.1365 | 97.50405 | 1530.956978 | 1545.439120 | 14.482141 | 100.945953 |

interventions 18359 / 30000 ticks; gate trips 11641; clipped ticks 0; max applied command 3.1773 m/s²; g_theta tracking RMSE 0.8673. Trajectory sha256: baseline `882199ae15bd29320a230ac5cad71736b954e0659920da8c2a4ef1b115873fa8`, assisted `1eb1c15a231b22b3322e0285b766ea4b9c3f1388417326119d8f30cdd0070a88`.

## Seed 2 · arm `lrs` (off-training)

| scenario | ep | baseline apogee (m) | assisted apogee (m) | Δ apogee (m) | retention % | baseline lateral RMS (m) | assisted lateral RMS (m) | Δ lateral (m) | lateral % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 1688.827 | 1682.615 | -6.2123 | 99.63216 | 1523.754604 | 1526.337802 | 2.583198 | 100.169528 |
| scenario_17 | 5 | 1838.721 | 1831.290 | -7.4315 | 99.59583 | 1477.782734 | 1481.594319 | 3.811585 | 100.257926 |
| scenario_18 | 5 | 1573.459 | 1567.383 | -6.0763 | 99.61383 | 1618.990980 | 1620.601680 | 1.610700 | 100.099488 |
| scenario_19 | 5 | 1812.026 | 1803.067 | -8.9594 | 99.50556 | 1499.505046 | 1502.537945 | 3.032899 | 100.202260 |
| **ALL (20 ep)** | — | 1728.258 | 1721.089 | -7.1699 | 99.58514 | 1530.956978 | 1533.688274 | 2.731296 | 100.178404 |

interventions 18359 / 30000 ticks; gate trips 11641; clipped ticks 0; max applied command 3.3301 m/s²; g_theta tracking RMSE 0.2078. Trajectory sha256: baseline `882199ae15bd29320a230ac5cad71736b954e0659920da8c2a4ef1b115873fa8`, assisted `40d666b0827bfd6d53d753e588a37f5d9d7f633920e8fa9fcfcb537b9553f3a7`.

## Seed 2 · arm `lite` (off-training)

| scenario | ep | baseline apogee (m) | assisted apogee (m) | Δ apogee (m) | retention % | baseline lateral RMS (m) | assisted lateral RMS (m) | Δ lateral (m) | lateral % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 1688.827 | 1673.914 | -14.9131 | 99.11695 | 1523.754604 | 1528.823375 | 5.068772 | 100.332650 |
| scenario_17 | 5 | 1838.721 | 1823.471 | -15.2500 | 99.17062 | 1477.782734 | 1484.107171 | 6.324437 | 100.427968 |
| scenario_18 | 5 | 1573.459 | 1557.980 | -15.4795 | 99.01621 | 1618.990980 | 1623.422451 | 4.431471 | 100.273718 |
| scenario_19 | 5 | 1812.026 | 1792.525 | -19.5011 | 98.92380 | 1499.505046 | 1506.458527 | 6.953481 | 100.463718 |
| **ALL (20 ep)** | — | 1728.258 | 1711.973 | -16.2859 | 99.05767 | 1530.956978 | 1536.619077 | 5.662099 | 100.369841 |

interventions 18359 / 30000 ticks; gate trips 11641; clipped ticks 0; max applied command 2.7496 m/s²; g_theta tracking RMSE 0.1905. Trajectory sha256: baseline `882199ae15bd29320a230ac5cad71736b954e0659920da8c2a4ef1b115873fa8`, assisted `ecc78770230565c985fc0eca5e436baf1293522b7889007fafa0d9ae779be73e`.

## Seed 3 · arm `oracle_gtheta` (positive control) (off-training)

| scenario | ep | baseline apogee (m) | assisted apogee (m) | Δ apogee (m) | retention % | baseline lateral RMS (m) | assisted lateral RMS (m) | Δ lateral (m) | lateral % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 1728.172 | 4033.050 | 2304.8783 | 233.37088 | 1517.289008 | 1020.417359 | -496.871650 | 67.252669 |
| scenario_17 | 5 | 1817.707 | 4067.187 | 2249.4797 | 223.75370 | 1474.799609 | 939.706767 | -535.092842 | 63.717590 |
| scenario_18 | 5 | 1632.301 | 3940.714 | 2308.4128 | 241.42078 | 1619.609587 | 1140.057020 | -479.552567 | 70.390854 |
| scenario_19 | 5 | 1803.436 | 3949.741 | 2146.3042 | 219.01192 | 1496.972985 | 981.077773 | -515.895213 | 65.537440 |
| **ALL (20 ep)** | — | 1745.404 | 3997.673 | 2252.2688 | 229.03996 | 1528.174005 | 1023.052296 | -505.121709 | 66.946061 |

interventions 18385 / 30000 ticks; gate trips 11615; clipped ticks n/a (sim-internal); max applied command n/a (sim-internal); g_theta tracking RMSE n/a. Trajectory sha256: baseline `32ba089073d24461ae599a3aac73c729b81b59bf147ae81d354bb9c1352b7a43`, assisted `14a499d586f9cb48eaeaf8bd88607838eab7338a58a0190017190f1d5ad75d8f`.

## Seed 3 · arm `loopwm` (off-training)

| scenario | ep | baseline apogee (m) | assisted apogee (m) | Δ apogee (m) | retention % | baseline lateral RMS (m) | assisted lateral RMS (m) | Δ lateral (m) | lateral % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 1728.172 | 1677.885 | -50.2872 | 97.09015 | 1517.289008 | 1533.345759 | 16.056751 | 101.058253 |
| scenario_17 | 5 | 1817.707 | 1763.178 | -54.5287 | 97.00014 | 1474.799609 | 1495.976244 | 21.176634 | 101.435899 |
| scenario_18 | 5 | 1632.301 | 1592.684 | -39.6176 | 97.57290 | 1619.609587 | 1629.969425 | 10.359838 | 100.639650 |
| scenario_19 | 5 | 1803.436 | 1758.461 | -44.9749 | 97.50615 | 1496.972985 | 1512.928241 | 15.955255 | 101.065835 |
| **ALL (20 ep)** | — | 1745.404 | 1698.052 | -47.3521 | 97.28704 | 1528.174005 | 1543.927324 | 15.753319 | 101.030859 |

interventions 18385 / 30000 ticks; gate trips 11615; clipped ticks 0; max applied command 3.1774 m/s²; g_theta tracking RMSE 0.8695. Trajectory sha256: baseline `32ba089073d24461ae599a3aac73c729b81b59bf147ae81d354bb9c1352b7a43`, assisted `1ee0c522983c681c7a2333282df4a683d74b6d8e417437de9095e9eb42861b68`.

## Seed 3 · arm `lrs` (off-training)

| scenario | ep | baseline apogee (m) | assisted apogee (m) | Δ apogee (m) | retention % | baseline lateral RMS (m) | assisted lateral RMS (m) | Δ lateral (m) | lateral % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 1728.172 | 1719.304 | -8.8677 | 99.48687 | 1517.289008 | 1519.942870 | 2.653861 | 100.174908 |
| scenario_17 | 5 | 1817.707 | 1808.871 | -8.8365 | 99.51386 | 1474.799609 | 1478.989666 | 4.190057 | 100.284110 |
| scenario_18 | 5 | 1632.301 | 1625.401 | -6.9006 | 99.57725 | 1619.609587 | 1621.390517 | 1.780930 | 100.109960 |
| scenario_19 | 5 | 1803.436 | 1796.431 | -7.0052 | 99.61157 | 1496.972985 | 1500.504892 | 3.531906 | 100.235937 |
| **ALL (20 ep)** | — | 1745.404 | 1737.502 | -7.9025 | 99.54724 | 1528.174005 | 1531.180827 | 3.006822 | 100.196759 |

interventions 18385 / 30000 ticks; gate trips 11615; clipped ticks 0; max applied command 3.3301 m/s²; g_theta tracking RMSE 0.2132. Trajectory sha256: baseline `32ba089073d24461ae599a3aac73c729b81b59bf147ae81d354bb9c1352b7a43`, assisted `d460c3426de1671fc39a62ea75893d97a759fa6c4ff2dffefde9f74f561c5d46`.

## Seed 3 · arm `lite` (off-training)

| scenario | ep | baseline apogee (m) | assisted apogee (m) | Δ apogee (m) | retention % | baseline lateral RMS (m) | assisted lateral RMS (m) | Δ lateral (m) | lateral % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 1728.172 | 1707.864 | -20.3085 | 98.82486 | 1517.289008 | 1523.921713 | 6.632705 | 100.437142 |
| scenario_17 | 5 | 1817.707 | 1800.077 | -17.6297 | 99.03012 | 1474.799609 | 1482.188438 | 7.388829 | 100.501006 |
| scenario_18 | 5 | 1632.301 | 1616.300 | -16.0007 | 99.01975 | 1619.609587 | 1624.226024 | 4.616436 | 100.285034 |
| scenario_19 | 5 | 1803.436 | 1789.933 | -13.5037 | 99.25122 | 1496.972985 | 1502.434759 | 5.461774 | 100.364855 |
| **ALL (20 ep)** | — | 1745.404 | 1728.544 | -16.8606 | 99.03400 | 1528.174005 | 1534.164295 | 5.990290 | 100.391990 |

interventions 18385 / 30000 ticks; gate trips 11615; clipped ticks 0; max applied command 2.7862 m/s²; g_theta tracking RMSE 0.1950. Trajectory sha256: baseline `32ba089073d24461ae599a3aac73c729b81b59bf147ae81d354bb9c1352b7a43`, assisted `5cdbb92baf4cf27d74fcb3df57682e06a69f997c41c61c5d775ecd54f1e6274b`.

## Why the student still fails while the oracle works

The oracle and the student are not the same signal in different units. `ReferenceOptimizer::fin_accel_command` emits the world-frame lateral acceleration demand `[ay, -ax]` from live full state and wind (`sim/core/src/optimizer.rs:98-113`, `sim/core/src/episode.rs:167-181`). `solve()` converts that SAME demand into a tilt target `g_theta` by `atan(ax/axial)`, attitude error and body-rate damping, clamped to ±0.12 rad (`sim/core/src/optimizer.rs:115-146`). The AI exports were trained against `targets[:,:2] = g_theta`, i.e. the angle-domain signal (`ai/src/potok_ai/engine.py:400-402`), not against `fin_accel_command`. Our `g·tan(10°)` conversion maps the student's normalised angle-error units to an acceleration magnitude; it does not undo the attitude/attitude-rate/axial conversion or the `[ay,-ax]` sign convention in `solve()`. The student therefore feeds an angle-domain P/D residual into an actuator that now expects an acceleration demand. This is a training-target mismatch, not a gate, scale or actuator-fit problem — no gain or threshold was changed to try to fix it.

## Honesty notes

- Commands are replayed open-loop from the baseline run's recorded frames; the control TSV is indexed by frame, as the simulator seam requires. Once the assisted command moves the vehicle, later frames differ from the baseline run — that is the point of the closed-loop comparison, but the interlock schedule stays the paired baseline one.
- Lateral RMS is over the whole 30 s flown trajectory, not just the AI-active tail; the earlier 12 s dataset's tiny active window no longer limits the measurement.
- Seeds 2/3 use the seed-1-trained exports; they test input robustness, not training reproducibility. No seed, scenario or episode was dropped.
- The calibration `isotonic` map itself is not exported in calibration.json; only its threshold is available, and it is applied to the raw confidence. This is an upper bound on how often the gate opens.
- `clipped_ticks` counts demands beyond the seam's ±10 m/s² envelope, not a tuning knob.
- Measurement floor: the sidecar stores f32 positions. One ULP is about 1.22e-04 m at the measured lateral scale and 1.22e-04 m at the measured altitude. Compare the deltas above against that floor before calling any difference a win.