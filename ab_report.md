# v3 closed-loop A/B — post-FIXPUMP — zero command vs oracle vs exported students

**post-FIXPUMP recalculation.** The flight helper was rebuilt against corrected dynamics (the aerodynamic normal force does no positive work); the frozen v3 exports are unchanged and no gate, threshold, weight or physics constant was tuned. The zero-command baseline and oracle numbers here supersede the pre-FIXPUMP `ab_report.json/.md`.

**This artifact also supersedes the v1/v2 closed-loop A/B** (`ab_report.json/.md` at `e57c6bf`): earlier surrogate numbers are void, and every v1/v2 held-out number was scored on the angle-domain target that the acceleration actuator cannot consume.

Verdict: **FAIL** (exit 1). Provenance, hashes and the full machine-readable evidence are in `ab_report.json`.

## Instrument and protocol

- Measurement: POTOKTRJ v3 flown-truth sidecar (stride 26, 6-DoF truth), not surrogate.
- Sim: read-only scripts/ab_control_v3 on commit 5acddab285c48964e8a4d52ed36cdfcef83681b8; oracle uses run_episode_with_oracle / ReferenceOptimizer::fin_accel_command (v3 g_accel, not g_theta)
- Dynamics: post-FIXPUMP: aerodynamic normal force does no positive work (584b1dc), drag_est pinned to trajectory mass (7ddf324).
- Config: `ai/out/v3_final/sim.toml` (config_hash `1969edcc0829536c4b2ee1bff0ba7d663a22fa36dec588fac67c96107379fed6`); all simulations are read-only through `scripts/ab_control_v3/`.
- Exports: `ai/out/v3_final`; per-file sha256 in `ab_report.json`. Held-out split sha256 `5848e222604b92d659f5778e467bfb359b98d5a2634bdc1b1c2a8257459685ed` (matches `SHA256SUMS` and `heldout_manifest.json`).
- Window: 50 frames of 14 features + 6 conditioning values; 1500 ticks/episode; 20 episodes in scenarios 16..19; seed order [1, 2, 3].
- Baseline: zero command every tick. Oracle: `run_episode_with_oracle` (privileged true state + `fin_accel_command`, v3 m/s^2 demand). Student: `physics_baseline + Delta` from the frozen v3 export, emitted only where the baseline reference says `ai_healthy` and the exported calibration threshold is cleared, clipped only at the seam's hard +/-10 m/s^2 bound. No physics, gain, threshold or scenario was tuned.
- Sidecar f32 floor: one ULP is about 6.10e-05 m at ~600 m lateral distance; a delta below 0.001 m is a numerical tie, not a win.

## Blockers

- lrs seed 3: missing export `ai/out/v3_final/runs/lrs/s3/lrs.bin`; seed not measured, no stand-in was substituted
- oracle positive control failed on seed 2: lateral 1306.397 -> 1295.894, apogee 1972.859 -> 1970.708; seed 3: lateral 1306.380 -> 1297.999, apogee 1983.003 -> 1975.402: the seam or measurement is broken; no student verdict is valid
- oracle positive control failed on 2 of 3 seeds under corrected dynamics, so no student verdict is valid; seed 1 primary (loopwm) lateral 1309.409 -> 1267.367 m is recorded but not credited. The oracle's pre-FIXPUMP apogee authority was the energy-pumping normal force, now removed.

## Positive control

- **FAILED**: seed 2: lateral 1306.397 -> 1295.894, apogee 1972.859 -> 1970.708; seed 3: lateral 1306.380 -> 1297.999, apogee 1983.003 -> 1975.402

## Seed 1 · `oracle (fin_accel_command, key kept for gate compatibility)`

| scenario | ep | baseline apogee (m) | assisted apogee (m) | Δ apogee (m) | baseline lateral RMS (m) | assisted lateral RMS (m) | Δ lateral (m) | lateral % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 1940.798 | 1927.912 | -12.8856 | 1286.482 | 1280.813 | -5.6686 | 99.55937 |
| scenario_17 | 5 | 2135.255 | 2175.505 | +40.2498 | 1303.543 | 1271.408 | -32.1348 | 97.53481 |
| scenario_18 | 5 | 1756.220 | 1788.625 | +32.4049 | 1376.938 | 1352.296 | -24.6420 | 98.21037 |
| scenario_19 | 5 | 2062.699 | 2049.699 | -13.0007 | 1268.064 | 1261.898 | -6.1657 | 99.51377 |
| **ALL (20 ep)** | — | 1973.743 | 1985.435 | +11.6921 | 1309.409 | 1292.097 | -17.3122 | 98.67786 |

## Seed 1 · `loopwm`

| scenario | ep | baseline apogee (m) | assisted apogee (m) | Δ apogee (m) | baseline lateral RMS (m) | assisted lateral RMS (m) | Δ lateral (m) | lateral % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 1940.798 | 2015.789 | +74.9913 | 1286.482 | 1252.859 | -33.6230 | 97.38643 |
| scenario_17 | 5 | 2135.255 | 2264.592 | +129.3374 | 1303.543 | 1249.394 | -54.1492 | 95.84600 |
| scenario_18 | 5 | 1756.220 | 1843.789 | +87.5690 | 1376.938 | 1342.997 | -33.9418 | 97.53498 |
| scenario_19 | 5 | 2062.699 | 2186.966 | +124.2662 | 1268.064 | 1220.899 | -47.1650 | 96.28055 |
| **ALL (20 ep)** | — | 1973.743 | 2077.784 | +104.0410 | 1309.409 | 1267.367 | -42.0420 | 96.78924 |

interventions 12158 / 30000 ticks; gate trips 11280; clipped 0; max command 2.000 m/s^2; tracking RMSE 9.818938048066364; first command tick 49. Trajectory sha256: baseline `9b1d680d55f635a3a2cf84864498cb533a84132bbe15ba791154a3cd58bb5953`, assisted `063ac2c1143adbcdf985100887929c9b337cf37b13e9e6b17da25ed1ecd90569`.

## Seed 1 · `lite`

| scenario | ep | baseline apogee (m) | assisted apogee (m) | Δ apogee (m) | baseline lateral RMS (m) | assisted lateral RMS (m) | Δ lateral (m) | lateral % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 1940.798 | 2015.795 | +74.9977 | 1286.482 | 1252.862 | -33.6194 | 97.38672 |
| scenario_17 | 5 | 2135.255 | 2264.598 | +129.3425 | 1303.543 | 1249.389 | -54.1545 | 95.84559 |
| scenario_18 | 5 | 1756.220 | 1843.794 | +87.5733 | 1376.938 | 1343.008 | -33.9309 | 97.53577 |
| scenario_19 | 5 | 2062.699 | 2186.968 | +124.2688 | 1268.064 | 1220.902 | -47.1617 | 96.28081 |
| **ALL (20 ep)** | — | 1973.743 | 2077.789 | +104.0456 | 1309.409 | 1267.370 | -42.0387 | 96.78949 |

interventions 18720 / 30000 ticks; gate trips 11280; clipped 0; max command 2.000 m/s^2; tracking RMSE 9.882492158803508; first command tick 49. Trajectory sha256: baseline `9b1d680d55f635a3a2cf84864498cb533a84132bbe15ba791154a3cd58bb5953`, assisted `cbbf923f5635813dd7809cfb68129a4cd7277ac0b3a276042d2e5a28313a8670`.

## Seed 1 · `lrs (partial export: seeds 1/2 only)`

| scenario | ep | baseline apogee (m) | assisted apogee (m) | Δ apogee (m) | baseline lateral RMS (m) | assisted lateral RMS (m) | Δ lateral (m) | lateral % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 1940.798 | 2229.000 | +288.2026 | 1286.482 | 1159.899 | -126.5828 | 90.16054 |
| scenario_17 | 5 | 2135.255 | 2515.077 | +379.8220 | 1303.543 | 1113.622 | -189.9210 | 85.43040 |
| scenario_18 | 5 | 1756.220 | 2037.605 | +281.3846 | 1376.938 | 1255.674 | -121.2647 | 91.19316 |
| scenario_19 | 5 | 2062.699 | 2437.029 | +374.3292 | 1268.064 | 1093.069 | -174.9945 | 86.19986 |
| **ALL (20 ep)** | — | 1973.743 | 2304.678 | +330.9346 | 1309.409 | 1157.264 | -152.1451 | 88.38063 |

interventions 18720 / 30000 ticks; gate trips 11280; clipped 0; max command 3.799 m/s^2; tracking RMSE 8.868337570825933; first command tick 0. Trajectory sha256: baseline `9b1d680d55f635a3a2cf84864498cb533a84132bbe15ba791154a3cd58bb5953`, assisted `8103f17d51db31eb7d3924fcfd30dea2ee6ccee8f516cdf372283dd5db37d53c`.

## Seed 2 · `oracle (fin_accel_command, key kept for gate compatibility)`

| scenario | ep | baseline apogee (m) | assisted apogee (m) | Δ apogee (m) | baseline lateral RMS (m) | assisted lateral RMS (m) | Δ lateral (m) | lateral % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 1955.196 | 1960.743 | +5.5470 | 1278.802 | 1263.796 | -15.0066 | 98.82651 |
| scenario_17 | 5 | 2044.483 | 2028.216 | -16.2673 | 1279.600 | 1275.254 | -4.3462 | 99.66035 |
| scenario_18 | 5 | 1825.667 | 1813.961 | -11.7059 | 1399.654 | 1394.615 | -5.0385 | 99.64002 |
| scenario_19 | 5 | 2066.089 | 2079.911 | +13.8219 | 1262.914 | 1244.573 | -18.3404 | 98.54777 |
| **ALL (20 ep)** | — | 1972.859 | 1970.708 | -2.1510 | 1306.397 | 1295.894 | -10.5029 | 99.19604 |

## Seed 2 · `loopwm`

| scenario | ep | baseline apogee (m) | assisted apogee (m) | Δ apogee (m) | baseline lateral RMS (m) | assisted lateral RMS (m) | Δ lateral (m) | lateral % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 1955.196 | 2056.257 | +101.0613 | 1278.802 | 1236.756 | -42.0460 | 96.71208 |
| scenario_17 | 5 | 2044.483 | 2182.215 | +137.7325 | 1279.600 | 1225.276 | -54.3244 | 95.75458 |
| scenario_18 | 5 | 1825.667 | 1905.588 | +79.9206 | 1399.654 | 1365.027 | -34.6267 | 97.52605 |
| scenario_19 | 5 | 2066.089 | 2127.082 | +60.9937 | 1262.914 | 1228.107 | -34.8068 | 97.24393 |
| **ALL (20 ep)** | — | 1972.859 | 2067.786 | +94.9270 | 1306.397 | 1265.149 | -41.2477 | 96.84264 |

interventions 18742 / 30000 ticks; gate trips 11258; clipped 0; max command 2.000 m/s^2; tracking RMSE 9.990313918090639; first command tick 49. Trajectory sha256: baseline `f0443872b644520d9f413e11b608241b0d8ea9bea65d5605b5e026ce08597aa2`, assisted `96133dca9628285779bb57de99894be58d098f0eb41f1357f878a955f93b4253`.

## Seed 2 · `lite`

| scenario | ep | baseline apogee (m) | assisted apogee (m) | Δ apogee (m) | baseline lateral RMS (m) | assisted lateral RMS (m) | Δ lateral (m) | lateral % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 1955.196 | 2054.609 | +99.4131 | 1278.802 | 1241.679 | -37.1231 | 97.09704 |
| scenario_17 | 5 | 2044.483 | 2180.571 | +136.0881 | 1279.600 | 1228.772 | -50.8284 | 96.02779 |
| scenario_18 | 5 | 1825.667 | 1904.099 | +78.4322 | 1399.654 | 1370.024 | -29.6296 | 97.88308 |
| scenario_19 | 5 | 2066.089 | 2121.970 | +55.8809 | 1262.914 | 1235.191 | -27.7229 | 97.80485 |
| **ALL (20 ep)** | — | 1972.859 | 2065.312 | +92.4536 | 1306.397 | 1270.267 | -36.1304 | 97.23435 |

interventions 16284 / 30000 ticks; gate trips 13716; clipped 0; max command 2.000 m/s^2; tracking RMSE 9.958090668939738; first command tick 49. Trajectory sha256: baseline `f0443872b644520d9f413e11b608241b0d8ea9bea65d5605b5e026ce08597aa2`, assisted `d4435613a9fa441fa59ed3d3f7f2e1e4726ba9288650d7383abd4c44ee900f00`.

## Seed 2 · `lrs (partial export: seeds 1/2 only)`

| scenario | ep | baseline apogee (m) | assisted apogee (m) | Δ apogee (m) | baseline lateral RMS (m) | assisted lateral RMS (m) | Δ lateral (m) | lateral % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 1955.196 | 2275.146 | +319.9504 | 1278.802 | 1136.246 | -142.5566 | 88.85233 |
| scenario_17 | 5 | 2044.483 | 2414.098 | +369.6154 | 1279.600 | 1090.636 | -188.9638 | 85.23259 |
| scenario_18 | 5 | 1825.667 | 2124.604 | +298.9363 | 1399.654 | 1271.371 | -128.2825 | 90.83470 |
| scenario_19 | 5 | 2066.089 | 2264.453 | +198.3643 | 1262.914 | 1155.699 | -107.2144 | 91.51055 |
| **ALL (20 ep)** | — | 1972.859 | 2269.575 | +296.7166 | 1306.397 | 1165.393 | -141.0036 | 89.20668 |

interventions 18742 / 30000 ticks; gate trips 11258; clipped 0; max command 3.827 m/s^2; tracking RMSE 9.195041271196928; first command tick 0. Trajectory sha256: baseline `f0443872b644520d9f413e11b608241b0d8ea9bea65d5605b5e026ce08597aa2`, assisted `8121abc8ae63e93604e5f42bf3475b6cb4151fe093077ce9da6560d96d17a346`.

## Seed 3 · `oracle (fin_accel_command, key kept for gate compatibility)`

| scenario | ep | baseline apogee (m) | assisted apogee (m) | Δ apogee (m) | baseline lateral RMS (m) | assisted lateral RMS (m) | Δ lateral (m) | lateral % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 1983.799 | 2005.173 | +21.3740 | 1272.570 | 1251.882 | -20.6876 | 98.37435 |
| scenario_17 | 5 | 2012.247 | 1995.921 | -16.3258 | 1279.368 | 1274.299 | -5.0683 | 99.60385 |
| scenario_18 | 5 | 1884.696 | 1876.163 | -8.5332 | 1400.991 | 1393.228 | -7.7631 | 99.44589 |
| scenario_19 | 5 | 2051.272 | 2024.353 | -26.9193 | 1267.883 | 1267.703 | -0.1797 | 99.98583 |
| **ALL (20 ep)** | — | 1983.003 | 1975.402 | -7.6011 | 1306.380 | 1297.999 | -8.3815 | 99.35842 |

## Seed 3 · `loopwm`

| scenario | ep | baseline apogee (m) | assisted apogee (m) | Δ apogee (m) | baseline lateral RMS (m) | assisted lateral RMS (m) | Δ lateral (m) | lateral % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 1983.799 | 2096.336 | +112.5367 | 1272.570 | 1225.124 | -47.4459 | 96.27165 |
| scenario_17 | 5 | 2012.247 | 2105.692 | +93.4451 | 1279.368 | 1235.116 | -44.2516 | 96.54114 |
| scenario_18 | 5 | 1884.696 | 1964.180 | +79.4844 | 1400.991 | 1365.436 | -35.5547 | 97.46217 |
| scenario_19 | 5 | 2051.272 | 2166.213 | +114.9412 | 1267.883 | 1222.388 | -45.4947 | 96.41176 |
| **ALL (20 ep)** | — | 1983.003 | 2083.105 | +100.1018 | 1306.380 | 1263.437 | -42.9436 | 96.71278 |

interventions 18750 / 30000 ticks; gate trips 11250; clipped 0; max command 2.000 m/s^2; tracking RMSE 9.930702543287069; first command tick 49. Trajectory sha256: baseline `b1123a635768bdd5d71b1591326477be3fde9f80a22b4f24288f067234ecbc4d`, assisted `ca5269e8ee9c54a2a097ef5a32c4a8ca08fb667c95888a4e81117f6dc2ba3f77`.

## Seed 3 · `lite`

| scenario | ep | baseline apogee (m) | assisted apogee (m) | Δ apogee (m) | baseline lateral RMS (m) | assisted lateral RMS (m) | Δ lateral (m) | lateral % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 1983.799 | 2096.822 | +113.0227 | 1272.570 | 1225.085 | -47.4842 | 96.26863 |
| scenario_17 | 5 | 2012.247 | 2106.436 | +94.1892 | 1279.368 | 1235.263 | -44.1050 | 96.55259 |
| scenario_18 | 5 | 1884.696 | 1964.599 | +79.9037 | 1400.991 | 1365.876 | -35.1145 | 97.49360 |
| scenario_19 | 5 | 2051.272 | 2166.473 | +115.2006 | 1267.883 | 1221.926 | -45.9564 | 96.37534 |
| **ALL (20 ep)** | — | 1983.003 | 2083.582 | +100.5791 | 1306.380 | 1263.470 | -42.9098 | 96.71537 |

interventions 18750 / 30000 ticks; gate trips 11250; clipped 0; max command 2.000 m/s^2; tracking RMSE 9.916981427984576; first command tick 49. Trajectory sha256: baseline `b1123a635768bdd5d71b1591326477be3fde9f80a22b4f24288f067234ecbc4d`, assisted `4780ea91938b3dbde05d0031558a5705cd5ad7fbff164fd30e1ba895d660bedb`.

## Honesty notes

- Commands are generated from the baseline flight's recorded frames and replayed by frame index through the simulator seam; after the assisted command moves the vehicle, later frames differ from the baseline flight, but the interlock schedule stays the paired baseline one.
- Lateral RMS and apogee are over the whole 30 s flown trajectory from the v3 sidecar, not over a gate-open tail and not from a surrogate.
- The v3 target is the fin command [g_accel_y, -g_accel_x] in m/s^2 seam order, so the student's trained output is written directly to the TSV; no degree/radian or g*tan conversion is applied.
- The physics_baseline term keeps its trained normalised scale and is added to the student Delta exactly as in engine.py::final_command; both are interpreted in the seam's m/s^2 units because that is the unit of the v3 supervised target.
- Clipping at +/-10 m/s^2 is the seam's hard actuator bound, counted per episode; no threshold, gain or physics constant was tuned for this report.
- Seeds 2/3 fly seed-matched exports on regenerated held-out flights; they test input robustness, not reproducibility of training. Missing exports are reported as blockers and never replaced with a stand-in.
- Post-FIXPUMP: the corrected zero-command baseline is NOT byte-identical to the frozen v3 held-out export, which was flown with the energy-pumping normal force. The 20 held-out scenario indices and their conditioning vectors are verified identical; only the flown physics changed. Exports were not retrained.
