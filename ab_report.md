# v3 closed-loop A/B — zero command vs oracle vs exported students

**This artifact supersedes the v1/v2 closed-loop A/B** (`ab_report.json/.md` at `e57c6bf`): earlier surrogate numbers are void, and every v1/v2 held-out number was scored on the angle-domain target that the acceleration actuator cannot consume.

Verdict: **PASS** (exit 0). Provenance, hashes and the full machine-readable evidence are in `ab_report.json`.

## Instrument and protocol

- Measurement: POTOKTRJ v3 flown-truth sidecar (stride 26, 6-DoF truth), not surrogate.
- Sim: read-only scripts/ab_control_v3 on commit cee6202c97825c0e66dc54283a9926417cd87075; oracle uses run_episode_with_oracle / ReferenceOptimizer::fin_accel_command (v3 g_accel, not g_theta)
- Config: `ai/out/v3_final/sim.toml` (config_hash `1969edcc0829536c4b2ee1bff0ba7d663a22fa36dec588fac67c96107379fed6`); all simulations are read-only through `scripts/ab_control_v3/`.
- Exports: `ai/out/v3_final`; per-file sha256 in `ab_report.json`. Held-out split sha256 `5848e222604b92d659f5778e467bfb359b98d5a2634bdc1b1c2a8257459685ed` (matches `SHA256SUMS` and `heldout_manifest.json`).
- Window: 50 frames of 14 features + 6 conditioning values; 1500 ticks/episode; 20 episodes in scenarios 16..19; seed order [1, 2, 3].
- Baseline: zero command every tick. Oracle: `run_episode_with_oracle` (privileged true state + `fin_accel_command`, v3 m/s^2 demand). Student: `physics_baseline + Delta` from the frozen v3 export, emitted only where the baseline reference says `ai_healthy` and the exported calibration threshold is cleared, clipped only at the seam's hard +/-10 m/s^2 bound. No physics, gain, threshold or scenario was tuned.
- Sidecar f32 floor: one ULP is about 6.10e-05 m at ~600 m lateral distance; a delta below 0.001 m is a numerical tie, not a win.

## Blockers

- lrs seed 3: missing export `ai/out/v3_final/runs/lrs/s3/lrs.bin`; seed not measured, no stand-in was substituted

## Positive control

- Oracle seed 1: lateral RMS 1528.366 -> 1035.228 m, apogee 1734.977 -> 3971.197 m; passes on every seed (see JSON). The seam can detect an improvement; this credits no student.

## Seed 1 · `oracle (fin_accel_command, key kept for gate compatibility)`

| scenario | ep | baseline apogee (m) | assisted apogee (m) | Δ apogee (m) | baseline lateral RMS (m) | assisted lateral RMS (m) | Δ lateral (m) | lateral % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 1667.467 | 3964.762 | +2297.2953 | 1532.165 | 1057.578 | -474.5867 | 69.02509 |
| scenario_17 | 5 | 1962.089 | 4064.499 | +2102.4098 | 1482.131 | 939.865 | -542.2657 | 63.41310 |
| scenario_18 | 5 | 1495.338 | 3830.443 | +2335.1047 | 1599.463 | 1156.621 | -442.8420 | 72.31308 |
| scenario_19 | 5 | 1815.016 | 4025.086 | +2210.0705 | 1497.028 | 973.239 | -523.7886 | 65.01143 |
| **ALL (20 ep)** | — | 1734.977 | 3971.197 | +2236.2201 | 1528.366 | 1035.228 | -493.1388 | 67.73425 |

## Seed 1 · `loopwm`

| scenario | ep | baseline apogee (m) | assisted apogee (m) | Δ apogee (m) | baseline lateral RMS (m) | assisted lateral RMS (m) | Δ lateral (m) | lateral % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 1667.467 | 1782.838 | +115.3710 | 1532.165 | 1509.877 | -22.2883 | 98.54531 |
| scenario_17 | 5 | 1962.089 | 2119.856 | +157.7674 | 1482.131 | 1443.865 | -38.2657 | 97.41820 |
| scenario_18 | 5 | 1495.338 | 1596.261 | +100.9228 | 1599.463 | 1582.775 | -16.6880 | 98.95665 |
| scenario_19 | 5 | 1815.016 | 1941.417 | +126.4017 | 1497.028 | 1468.512 | -28.5160 | 98.09516 |
| **ALL (20 ep)** | — | 1734.977 | 1860.093 | +125.1157 | 1528.366 | 1502.180 | -26.1865 | 98.28663 |

interventions 9135 / 30000 ticks; gate trips 11596; clipped 0; max command 2.000 m/s^2; tracking RMSE 9.054219968923212; first command tick 49. Trajectory sha256: baseline `0be79f08183fcf56fd207cbea1757adb588b9bbd0d7bb7841bb0bd0bb128c9ee`, assisted `3ca4ee1cf4eb3b31df0fa64328de7f6bb594bdc74dac0abc98bf0ed06dd72532`.

## Seed 1 · `lite`

| scenario | ep | baseline apogee (m) | assisted apogee (m) | Δ apogee (m) | baseline lateral RMS (m) | assisted lateral RMS (m) | Δ lateral (m) | lateral % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 1667.467 | 1782.837 | +115.3706 | 1532.165 | 1509.877 | -22.2882 | 98.54532 |
| scenario_17 | 5 | 1962.089 | 2119.856 | +157.7669 | 1482.131 | 1443.865 | -38.2655 | 97.41821 |
| scenario_18 | 5 | 1495.338 | 1596.261 | +100.9224 | 1599.463 | 1582.775 | -16.6879 | 98.95665 |
| scenario_19 | 5 | 1815.016 | 1941.417 | +126.4012 | 1497.028 | 1468.512 | -28.5159 | 98.09517 |
| **ALL (20 ep)** | — | 1734.977 | 1860.093 | +125.1153 | 1528.366 | 1502.180 | -26.1864 | 98.28664 |

interventions 18404 / 30000 ticks; gate trips 11596; clipped 0; max command 2.000 m/s^2; tracking RMSE 9.542279456699712; first command tick 49. Trajectory sha256: baseline `0be79f08183fcf56fd207cbea1757adb588b9bbd0d7bb7841bb0bd0bb128c9ee`, assisted `43c2e0aafc3c2ba63f7d4ef6b3c0c2ff324ce56e056efdebed5bea123e53eb43`.

## Seed 1 · `lrs (partial export: seeds 1/2 only)`

| scenario | ep | baseline apogee (m) | assisted apogee (m) | Δ apogee (m) | baseline lateral RMS (m) | assisted lateral RMS (m) | Δ lateral (m) | lateral % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 1667.467 | 1781.174 | +113.7069 | 1532.165 | 1510.186 | -21.9791 | 98.56548 |
| scenario_17 | 5 | 1962.089 | 2117.613 | +155.5244 | 1482.131 | 1444.336 | -37.7948 | 97.44997 |
| scenario_18 | 5 | 1495.338 | 1594.813 | +99.4745 | 1599.463 | 1583.013 | -16.4499 | 98.97154 |
| scenario_19 | 5 | 1815.016 | 1939.563 | +124.5478 | 1497.028 | 1468.901 | -28.1266 | 98.12117 |
| **ALL (20 ep)** | — | 1734.977 | 1858.291 | +123.3134 | 1528.366 | 1502.529 | -25.8378 | 98.30945 |

interventions 18404 / 30000 ticks; gate trips 11596; clipped 0; max command 1.993 m/s^2; tracking RMSE 9.549632761533159; first command tick 0. Trajectory sha256: baseline `0be79f08183fcf56fd207cbea1757adb588b9bbd0d7bb7841bb0bd0bb128c9ee`, assisted `3166faefa3f9350d8c8eee7f172c927e231d20df6515f9965ac01444f7d831e5`.

## Seed 2 · `oracle (fin_accel_command, key kept for gate compatibility)`

| scenario | ep | baseline apogee (m) | assisted apogee (m) | Δ apogee (m) | baseline lateral RMS (m) | assisted lateral RMS (m) | Δ lateral (m) | lateral % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 1688.827 | 4034.599 | +2345.7722 | 1523.755 | 1051.584 | -472.1703 | 69.01271 |
| scenario_17 | 5 | 1838.721 | 4027.571 | +2188.8494 | 1477.783 | 940.245 | -537.5376 | 63.62540 |
| scenario_18 | 5 | 1573.459 | 3955.250 | +2381.7910 | 1618.991 | 1144.626 | -474.3655 | 70.69993 |
| scenario_19 | 5 | 1812.026 | 3984.499 | +2172.4729 | 1499.505 | 985.502 | -514.0033 | 65.72180 |
| **ALL (20 ep)** | — | 1728.258 | 4000.480 | +2272.2214 | 1530.957 | 1033.353 | -497.6042 | 67.49718 |

## Seed 2 · `loopwm`

| scenario | ep | baseline apogee (m) | assisted apogee (m) | Δ apogee (m) | baseline lateral RMS (m) | assisted lateral RMS (m) | Δ lateral (m) | lateral % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 1688.827 | 1812.374 | +123.5467 | 1523.755 | 1500.028 | -23.7262 | 98.44291 |
| scenario_17 | 5 | 1838.721 | 1976.057 | +137.3357 | 1477.783 | 1445.298 | -32.4852 | 97.80176 |
| scenario_18 | 5 | 1573.459 | 1673.246 | +99.7863 | 1618.991 | 1602.178 | -16.8128 | 98.96153 |
| scenario_19 | 5 | 1812.026 | 1938.466 | +126.4402 | 1499.505 | 1471.437 | -28.0683 | 98.12816 |
| **ALL (20 ep)** | — | 1728.258 | 1850.036 | +121.7772 | 1530.957 | 1505.911 | -25.0460 | 98.36403 |

interventions 18359 / 30000 ticks; gate trips 11641; clipped 0; max command 2.000 m/s^2; tracking RMSE 9.531946115442148; first command tick 49. Trajectory sha256: baseline `60980e80c731d0720a9926363c9035761bd5249f4a66380cd025be66b3c47107`, assisted `a541dd26c543c314228eb515f57223819d28876606455cff6d1f678f2aa351f7`.

## Seed 2 · `lite`

| scenario | ep | baseline apogee (m) | assisted apogee (m) | Δ apogee (m) | baseline lateral RMS (m) | assisted lateral RMS (m) | Δ lateral (m) | lateral % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 1688.827 | 1812.374 | +123.5466 | 1523.755 | 1500.023 | -23.7313 | 98.44258 |
| scenario_17 | 5 | 1838.721 | 1976.057 | +137.3356 | 1477.783 | 1445.298 | -32.4851 | 97.80177 |
| scenario_18 | 5 | 1573.459 | 1673.246 | +99.7863 | 1618.991 | 1602.065 | -16.9256 | 98.95456 |
| scenario_19 | 5 | 1812.026 | 1938.466 | +126.4402 | 1499.505 | 1471.437 | -28.0683 | 98.12816 |
| **ALL (20 ep)** | — | 1728.258 | 1850.036 | +121.7772 | 1530.957 | 1505.880 | -25.0773 | 98.36199 |

interventions 18188 / 30000 ticks; gate trips 11641; clipped 0; max command 2.000 m/s^2; tracking RMSE 9.535901917684138; first command tick 49. Trajectory sha256: baseline `60980e80c731d0720a9926363c9035761bd5249f4a66380cd025be66b3c47107`, assisted `325e7020621321a1d2c720b72cd693ce3af5a792a17f1722afd506afc86607b8`.

## Seed 2 · `lrs (partial export: seeds 1/2 only)`

| scenario | ep | baseline apogee (m) | assisted apogee (m) | Δ apogee (m) | baseline lateral RMS (m) | assisted lateral RMS (m) | Δ lateral (m) | lateral % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 1688.827 | 1810.689 | +121.8620 | 1523.755 | 1500.329 | -23.4255 | 98.46265 |
| scenario_17 | 5 | 1838.721 | 1974.197 | +135.4760 | 1477.783 | 1445.697 | -32.0856 | 97.82880 |
| scenario_18 | 5 | 1573.459 | 1671.889 | +98.4297 | 1618.991 | 1602.413 | -16.5779 | 98.97604 |
| scenario_19 | 5 | 1812.026 | 1936.741 | +124.7144 | 1499.505 | 1471.797 | -27.7084 | 98.15217 |
| **ALL (20 ep)** | — | 1728.258 | 1848.379 | +120.1205 | 1530.957 | 1506.232 | -24.7249 | 98.38501 |

interventions 18359 / 30000 ticks; gate trips 11641; clipped 0; max command 1.993 m/s^2; tracking RMSE 9.53890081544099; first command tick 0. Trajectory sha256: baseline `60980e80c731d0720a9926363c9035761bd5249f4a66380cd025be66b3c47107`, assisted `aaf15ce52eabe5aec5081eea5da4bd2e237681f847b34e40b5be3e7585d8c1e4`.

## Seed 3 · `oracle (fin_accel_command, key kept for gate compatibility)`

| scenario | ep | baseline apogee (m) | assisted apogee (m) | Δ apogee (m) | baseline lateral RMS (m) | assisted lateral RMS (m) | Δ lateral (m) | lateral % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 1728.172 | 4033.050 | +2304.8783 | 1517.289 | 1020.417 | -496.8716 | 67.25267 |
| scenario_17 | 5 | 1817.707 | 4067.187 | +2249.4797 | 1474.800 | 939.707 | -535.0928 | 63.71759 |
| scenario_18 | 5 | 1632.301 | 3940.714 | +2308.4128 | 1619.610 | 1140.057 | -479.5526 | 70.39085 |
| scenario_19 | 5 | 1803.436 | 3949.741 | +2146.3042 | 1496.973 | 981.078 | -515.8952 | 65.53744 |
| **ALL (20 ep)** | — | 1745.404 | 3997.673 | +2252.2688 | 1528.174 | 1023.052 | -505.1217 | 66.94606 |

## Seed 3 · `loopwm`

| scenario | ep | baseline apogee (m) | assisted apogee (m) | Δ apogee (m) | baseline lateral RMS (m) | assisted lateral RMS (m) | Δ lateral (m) | lateral % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 1728.172 | 1859.548 | +131.3756 | 1517.289 | 1491.250 | -26.0388 | 98.28386 |
| scenario_17 | 5 | 1817.707 | 1961.211 | +143.5040 | 1474.800 | 1441.126 | -33.6741 | 97.71670 |
| scenario_18 | 5 | 1632.301 | 1722.127 | +89.8257 | 1619.610 | 1603.621 | -15.9887 | 99.01281 |
| scenario_19 | 5 | 1803.436 | 1928.644 | +125.2075 | 1496.973 | 1468.996 | -27.9774 | 98.13107 |
| **ALL (20 ep)** | — | 1745.404 | 1867.882 | +122.4782 | 1528.174 | 1502.516 | -25.6579 | 98.32101 |

interventions 18385 / 30000 ticks; gate trips 11615; clipped 0; max command 2.000 m/s^2; tracking RMSE 9.541174813842318; first command tick 49. Trajectory sha256: baseline `7b80f3d93e72a8beb1706045c76e254b0c34829b910fabe83e83c48e215a9f3b`, assisted `f42e40dcc7d27f74d329397734d6ec09a1f38ebf0c05f96361803cd62c7e8a5b`.

## Seed 3 · `lite`

| scenario | ep | baseline apogee (m) | assisted apogee (m) | Δ apogee (m) | baseline lateral RMS (m) | assisted lateral RMS (m) | Δ lateral (m) | lateral % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 1728.172 | 1859.547 | +131.3747 | 1517.289 | 1491.251 | -26.0385 | 98.28388 |
| scenario_17 | 5 | 1817.707 | 1961.209 | +143.5023 | 1474.800 | 1441.126 | -33.6737 | 97.71673 |
| scenario_18 | 5 | 1632.301 | 1722.126 | +89.8250 | 1619.610 | 1603.621 | -15.9885 | 99.01282 |
| scenario_19 | 5 | 1803.436 | 1928.642 | +125.2061 | 1496.973 | 1468.996 | -27.9771 | 98.13109 |
| **ALL (20 ep)** | — | 1745.404 | 1867.881 | +122.4770 | 1528.174 | 1502.516 | -25.6576 | 98.32103 |

interventions 18385 / 30000 ticks; gate trips 11615; clipped 0; max command 2.000 m/s^2; tracking RMSE 9.541179667293777; first command tick 49. Trajectory sha256: baseline `7b80f3d93e72a8beb1706045c76e254b0c34829b910fabe83e83c48e215a9f3b`, assisted `75f11a2b7fdbdf4fd9bb94cd2c4078220a0944f71f46bc652823f5f69dcf35d8`.

## Honesty notes

- Commands are generated from the baseline flight's recorded frames and replayed by frame index through the simulator seam; after the assisted command moves the vehicle, later frames differ from the baseline flight, but the interlock schedule stays the paired baseline one.
- Lateral RMS and apogee are over the whole 30 s flown trajectory from the v3 sidecar, not over a gate-open tail and not from a surrogate.
- The v3 target is the fin command [g_accel_y, -g_accel_x] in m/s^2 seam order, so the student's trained output is written directly to the TSV; no degree/radian or g*tan conversion is applied.
- The physics_baseline term keeps its trained normalised scale and is added to the student Delta exactly as in engine.py::final_command; both are interpreted in the seam's m/s^2 units because that is the unit of the v3 supervised target.
- Clipping at +/-10 m/s^2 is the seam's hard actuator bound, counted per episode; no threshold, gain or physics constant was tuned for this report.
- Seeds 2/3 fly seed-matched exports on regenerated held-out flights; they test input robustness, not reproducibility of training. Missing exports are reported as blockers and never replaced with a stand-in.
