# PID (zero command) vs AI assistant — closed-loop OR-aligned A/B

**This report SUPERSEDES the surrogate A/B** (`237920a`, the previous `ab_report.json/.md`). The old apogee/lateral numbers were a gravity/tilt surrogate on recorded commands; they are void, not a smaller version of these results.

Apogee and lateral RMS are now measured directly from the flown-truth `validation.trajectory` sidecar (POTOKTRJ v1, per-tick x/y/z/vx/vy/vz)
produced by `potok-sim`'s command→dynamics seam (`Simulator::run_episode_with_control`, `sim/core/src/episode.rs:130`).
No sim source or config was changed; `scripts/ab_control/` is a read-only driver because the CLI's `--control-file` applies one TSV to every episode.

## Protocol

- Seed 1: the 20 OR-aligned held-out episodes (indices 16..19,36..39,56..59,76..79,96..99) across scenario_16..19, from `ai/out/run2/sim.toml` (12 s, 50 Hz). Baseline validation is byte-identical to `ai/out/run2/data/heldout_validation.bin`.
- Seeds 2 and 3: same 20 indices/scenarios, generated with `--seed 2/3`; the students were NOT retrained, so these are off-training generalization checks, not a seed sweep of the training procedure.
- Baseline arm: no control file, zero command on every tick.
- Assisted arm: per-tick `command = physics_baseline(frame) + student Delta`, the same residual correction as the surrogate report. The exported weights are run through the committed (edge-equivalent) student forward in `scripts/ab_pid_vs_ai.py::CanonicalStudent`; the uncommitted change in `ai/src/potok_ai/student.py` is NOT used, because these exports were trained against the old forward. Interlock kept: a command is emitted only where the recorded reference says `ai_healthy` and the calibrated student confidence clears its threshold; otherwise zero.
- Unit mapping: normalised command × 0.12 rad × 180/π = degrees (the reference optimizer's ±0.12 rad clamp, same scale as the old surrogate). Commands beyond the seam's ±10° actuator limit are clipped and counted (`clipped_ticks`); no physics or gains were tuned.
- `interventions` counts gate-open ticks with a non-zero applied command; `gate_trips` counts ticks with the interlock closed. Both are summed over episodes. `tracking_rmse_norm` is the assisted command's RMS error against g_theta on gate-open ticks (diagnostic only).
- Exit status: 0 iff the primary assisted arm reduces lateral RMS on seed 1; nonzero otherwise. Apogee deltas are reported even when tiny or negative.

## Verdict (primary: loopwm student, seed 1)

- lateral RMS 566.002412 m → 566.002452 m (0.000039 m, 100.000007 % of baseline): DID NOT BEAT baseline.
- apogee 1321.346 m → 1321.343 m (-0.0028 m, 99.99979 % retention).
- overall **FAIL** (exit 1).

## Seed 1 · student `loopwm` (primary)

| scenario | ep | baseline apogee (m) | assisted apogee (m) | Δ apogee (m) | retention % | baseline lateral RMS (m) | assisted lateral RMS (m) | Δ lateral (m) | lateral % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 1298.628 | 1298.626 | -0.0018 | 99.99986 | 560.084974 | 560.084991 | 0.000017 | 100.000003 |
| scenario_17 | 5 | 1431.955 | 1431.953 | -0.0014 | 99.99990 | 556.261905 | 556.261915 | 0.000010 | 100.000002 |
| scenario_18 | 5 | 1208.793 | 1208.789 | -0.0034 | 99.99972 | 590.758082 | 590.758118 | 0.000036 | 100.000006 |
| scenario_19 | 5 | 1346.007 | 1346.003 | -0.0045 | 99.99967 | 556.163039 | 556.163134 | 0.000095 | 100.000017 |
| **ALL (20 ep)** | — | 1321.346 | 1321.343 | -0.0028 | 99.99979 | 566.002412 | 566.002452 | 0.000039 | 100.000007 |

interventions 401 / 12000 ticks; gate trips 11599; clipped ticks 0; max applied command 7.790°; g_theta tracking RMSE 1.0018. Trajectory sha256: baseline `7a7117066d8fe7953f97ffc9addf431d1426abbc28a0d0e41d8bcb44a9fba923`, assisted `a2c6e22133a194c1740dd7ee183b6bcf03e6226356c739ffd978c892b16bd3e5`.

## Seed 1 · student `lrs`

| scenario | ep | baseline apogee (m) | assisted apogee (m) | Δ apogee (m) | retention % | baseline lateral RMS (m) | assisted lateral RMS (m) | Δ lateral (m) | lateral % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 1298.628 | 1298.628 | -0.0003 | 99.99998 | 560.084974 | 560.084972 | -0.000002 | 100.000000 |
| scenario_17 | 5 | 1431.955 | 1431.955 | 0.0001 | 100.00001 | 556.261905 | 556.261892 | -0.000012 | 99.999998 |
| scenario_18 | 5 | 1208.793 | 1208.791 | -0.0017 | 99.99986 | 590.758082 | 590.758100 | 0.000018 | 100.000003 |
| scenario_19 | 5 | 1346.007 | 1346.005 | -0.0019 | 99.99986 | 556.163039 | 556.163089 | 0.000050 | 100.000009 |
| **ALL (20 ep)** | — | 1321.346 | 1321.345 | -0.0010 | 99.99993 | 566.002412 | 566.002426 | 0.000013 | 100.000002 |

interventions 401 / 12000 ticks; gate trips 11599; clipped ticks 0; max applied command 6.926°; g_theta tracking RMSE 1.0021. Trajectory sha256: baseline `7a7117066d8fe7953f97ffc9addf431d1426abbc28a0d0e41d8bcb44a9fba923`, assisted `022a914d06d6e0dafb909cbaa885f47278237665d11cb95aa857f93df41467ed`.

## Seed 2 · student `loopwm` (off-training)

| scenario | ep | baseline apogee (m) | assisted apogee (m) | Δ apogee (m) | retention % | baseline lateral RMS (m) | assisted lateral RMS (m) | Δ lateral (m) | lateral % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 1284.671 | 1284.669 | -0.0014 | 99.99989 | 571.915690 | 571.915714 | 0.000024 | 100.000004 |
| scenario_17 | 5 | 1366.894 | 1366.889 | -0.0053 | 99.99961 | 523.193749 | 523.193873 | 0.000124 | 100.000024 |
| scenario_18 | 5 | 1236.375 | 1236.374 | -0.0015 | 99.99988 | 608.541820 | 608.541832 | 0.000011 | 100.000002 |
| scenario_19 | 5 | 1360.423 | 1360.423 | -0.0001 | 99.99999 | 551.791303 | 551.791287 | -0.000016 | 99.999997 |
| **ALL (20 ep)** | — | 1312.091 | 1312.089 | -0.0021 | 99.99984 | 564.715868 | 564.715902 | 0.000034 | 100.000006 |

interventions 391 / 12000 ticks; gate trips 11609; clipped ticks 0; max applied command 7.790°; g_theta tracking RMSE 1.0014. Trajectory sha256: baseline `af9976e2907e309cfc106b8651afda5b5471aebd8f5e30c78517c682cb365f24`, assisted `a98b6192bd25bc89c978528befc0d0e02daa18c82e83bcf84eff4b58e21103f0`.

## Seed 2 · student `lrs` (off-training)

| scenario | ep | baseline apogee (m) | assisted apogee (m) | Δ apogee (m) | retention % | baseline lateral RMS (m) | assisted lateral RMS (m) | Δ lateral (m) | lateral % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 1284.671 | 1284.671 | -0.0001 | 99.99999 | 571.915690 | 571.915696 | 0.000006 | 100.000001 |
| scenario_17 | 5 | 1366.894 | 1366.892 | -0.0019 | 99.99986 | 523.193749 | 523.193795 | 0.000046 | 100.000009 |
| scenario_18 | 5 | 1236.375 | 1236.374 | -0.0008 | 99.99993 | 608.541820 | 608.541825 | 0.000005 | 100.000001 |
| scenario_19 | 5 | 1360.423 | 1360.425 | 0.0024 | 100.00017 | 551.791303 | 551.791244 | -0.000059 | 99.999989 |
| **ALL (20 ep)** | — | 1312.091 | 1312.091 | -0.0001 | 99.99999 | 564.715868 | 564.715867 | -0.000001 | 100.000000 |

interventions 391 / 12000 ticks; gate trips 11609; clipped ticks 0; max applied command 6.926°; g_theta tracking RMSE 1.0055. Trajectory sha256: baseline `af9976e2907e309cfc106b8651afda5b5471aebd8f5e30c78517c682cb365f24`, assisted `5149120fdceb5d8b269de7fb6153fd0c87fd69031b46f46ba1e5c25cfac4be9c`.

## Seed 3 · student `loopwm` (off-training)

| scenario | ep | baseline apogee (m) | assisted apogee (m) | Δ apogee (m) | retention % | baseline lateral RMS (m) | assisted lateral RMS (m) | Δ lateral (m) | lateral % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 1319.460 | 1319.458 | -0.0023 | 99.99982 | 571.922795 | 571.922817 | 0.000022 | 100.000004 |
| scenario_17 | 5 | 1349.590 | 1349.583 | -0.0072 | 99.99947 | 516.635884 | 516.636019 | 0.000135 | 100.000026 |
| scenario_18 | 5 | 1293.819 | 1293.819 | 0.0007 | 100.00005 | 610.778420 | 610.778414 | -0.000006 | 99.999999 |
| scenario_19 | 5 | 1360.607 | 1360.604 | -0.0022 | 99.99983 | 552.972834 | 552.972858 | 0.000023 | 100.000004 |
| **ALL (20 ep)** | — | 1330.869 | 1330.866 | -0.0028 | 99.99979 | 564.100513 | 564.100553 | 0.000041 | 100.000007 |

interventions 392 / 12000 ticks; gate trips 11608; clipped ticks 0; max applied command 7.790°; g_theta tracking RMSE 1.0013. Trajectory sha256: baseline `6157fdd0452d70a2f44f64ec51fb8771783be5f87762c650db0d0eb49231b4db`, assisted `e3a605662bf61a4ed155fdb62403e4250cb4ecf77cda71be2a5a4bedbed0b884`.

## Seed 3 · student `lrs` (off-training)

| scenario | ep | baseline apogee (m) | assisted apogee (m) | Δ apogee (m) | retention % | baseline lateral RMS (m) | assisted lateral RMS (m) | Δ lateral (m) | lateral % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_16 | 5 | 1319.460 | 1319.458 | -0.0015 | 99.99988 | 571.922795 | 571.922809 | 0.000015 | 100.000003 |
| scenario_17 | 5 | 1349.590 | 1349.586 | -0.0036 | 99.99973 | 516.635884 | 516.635941 | 0.000056 | 100.000011 |
| scenario_18 | 5 | 1293.819 | 1293.820 | 0.0012 | 100.00009 | 610.778420 | 610.778411 | -0.000009 | 99.999999 |
| scenario_19 | 5 | 1360.607 | 1360.607 | 0.0007 | 100.00005 | 552.972834 | 552.972798 | -0.000036 | 99.999993 |
| **ALL (20 ep)** | — | 1330.869 | 1330.868 | -0.0008 | 99.99994 | 564.100513 | 564.100518 | 0.000005 | 100.000001 |

interventions 392 / 12000 ticks; gate trips 11608; clipped ticks 0; max applied command 6.926°; g_theta tracking RMSE 1.0058. Trajectory sha256: baseline `6157fdd0452d70a2f44f64ec51fb8771783be5f87762c650db0d0eb49231b4db`, assisted `1f09cb185f5fc3dca4727f1158264351a7ab5cd9d3aeaf108ff5b1adf2bae638`.

## Honesty notes

- Commands are replayed open-loop from the baseline run's recorded frames; the control TSV is indexed by frame, as the simulator seam requires. Once the assisted command moves the vehicle, later frames differ from the baseline run — that is the point of the closed-loop comparison, but the interlock schedule stays the paired baseline one.
- Lateral RMS is over the whole 12 s flown trajectory, not just the interlock-open tail; the old report's 401-tick restriction no longer applies to the measurement (it still applies to when a command can be emitted).
- Seeds 2/3 use the seed-1-trained students; they test input robustness, not training reproducibility. No seed, scenario or episode was dropped.
- `clipped_ticks` is the seam's hard ±10° actuator bound, not a tuning knob.
- Measurement floor: the sidecar stores f32 positions. One ULP is about 6.10e-05 m at ~600 m lateral distance and 1.22e-04 m at ~1300 m altitude. The lateral deltas (~4e-5 m) are below one coordinate ULP — numerically a tie; the apogee deltas (~-0.003 m) are ~20 ULP but still physically negligible. No win survives rounding.