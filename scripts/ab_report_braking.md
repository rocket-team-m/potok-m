# BRK02 closed-loop A/B — braking-teacher student under unweakened v6 gates

Arm verdict: **FAIL** (exit 1). The control is the paired zero-command flight; the oracle is the positive control. Every v6 gate is called verbatim from `scripts/ab_v6.py`. This is a post-FIXPUMP simulator measurement, not a flight claim.

| gate | result | detail |
|---|---|---|
| eval01 rmse_ratio | PASS | verbatim EVAL01 gate |
| eval01 ece_source | PASS | verbatim EVAL01 gate |
| eval01 ece | PASS | verbatim EVAL01 gate |
| eval01 provenance | PASS | verbatim EVAL01 gate |
| eval01 seed_spread | PASS | verbatim EVAL01 gate |
| eval01 closed_loop_improvement | FAIL | see blockers |
| eval01 oracle_positive_control | FAIL | see blockers |
| seed 1 absolute gates | FAIL | deployment lateral 1476.71 m <= 1000 m: False; apogee 2339.45 m >= 2000 m: True |
| seed 2 absolute gates | FAIL | deployment lateral 1544.14 m <= 1000 m: False; apogee 2282.45 m >= 2000 m: True |
| seed 3 absolute gates | FAIL | deployment lateral 1536.59 m <= 1000 m: False; apogee 2279.10 m >= 2000 m: True |
| oracle_deployment_and_apogee_positive_control | FAIL | both metrics improve against paired zero command on every seed |

| seed | arm | deployment lateral (m) | apogee (m) | whole-trajectory lateral RMS (m) |
|---:|---|---:|---:|---:|
| 1 | oracle_gtheta | 1690.76 -> 1650.28 | 1973.74 -> 1985.44 | 1309.41 -> 1292.10 |
| 1 | braking | 1690.76 -> 1476.71 | 1973.74 -> 2339.45 | 1309.41 -> 1121.87 |
| 2 | oracle_gtheta | 1688.42 -> 1650.63 | 1972.86 -> 1970.71 | 1306.40 -> 1295.89 |
| 2 | braking | 1688.42 -> 1544.14 | 1972.86 -> 2282.45 | 1306.40 -> 1159.11 |
| 3 | oracle_gtheta | 1689.19 -> 1653.20 | 1983.00 -> 1975.40 | 1306.38 -> 1298.00 |
| 3 | braking | 1689.19 -> 1536.59 | 1983.00 -> 2279.10 | 1306.38 -> 1164.23 |

Positive control passed: **False**

- seed 1: deployment lateral -40.48 m, apogee +11.69 m, cap ticks 16661/16661
- seed 2: deployment lateral -37.79 m, apogee -2.15 m, cap ticks 16584/16584
- seed 3: deployment lateral -35.99 m, apogee -7.60 m, cap ticks 16531/16531

Blockers:

- EVAL01 closed_loop_improvement failed: seed 1 lateral -14.32% is above -35%; seed 2 lateral -11.27% is above -35%; seed 3 lateral -10.88% is above -35%; seed 3 apogee 14.93% is below 15%
- EVAL01 oracle_positive_control failed: seed 2 oracle did not raise apogee; seed 3 oracle did not raise apogee
- braking: oracle positive control did not improve both metrics on all seeds

## Verdict

`pass`=False, `win_claimed`=False, gates failed=6. A measured FAIL: no win is claimed and no gate was weakened.

## Provenance

| field | value |
|---|---|
| git_head | `f825a50912123989d2bc29103fbb649d76cd9e98` |
| harness | `scripts/ab_braking.py` |
| harness_sha256 | `561dd4340e66880f48be6f0c10b967ca562e962da76d3ba9599a24b81d0b1139` |
| objective_source_sha256 | `64400853113e70b60868add49d63b0f313a6c84b6d0a4801ebddee47be628411` |
| objective_version | `TRUTH_VEL_START=3,TRUTH_VEL_END=6` |
| objective_head | `f825a50912123989d2bc29103fbb649d76cd9e98` |
| config_sha256 | `9c34e71f9d10bcf081c31f15f5332f7bb2fac04dddb28009a27cd30987ab5e91` |
| heldout_sha256 | `fdc1d1400b1c5f111e21f79dead37afa1dd9f058fe457edb78037dc6ebe89807` |
| summary_sha256 | `1f73a14df3945fb851834486c75e1fbb519369db2d89015c3c34b032d05ce9c4` |
| export_manifest_sha256 | `ac4b070c9f4e111503fbd5d2030e1f4341df89559e8b10a67d0d0bd942ecd6c3` |
| exports_sha256 | `None` |
| command_limit_mps2 | `10.0` |
| deployment_proxy | `first DESCENT tick (vz<0 and z>=5), radial distance from pad` |
| lateral_limit_m | `1000.0` |
| apogee_limit_m | `2000.0` |

## Honesty notes

- Simulator-only: the POTOKTRJ sidecar is emitted by the same simulator that flies the episodes; it is not an independent flight truth anchor.
- No canopy/deployment event is scored; the proxy is the first DESCENT tick and is not a landing-dispersion measurement.
- The braking teacher is privileged simulator truth. This A/B measures the distilled student's closed-loop behaviour, not teacher fidelity or flight.
- Commands are generated from the baseline flight's recorded frames and replayed by frame index; they are emitted only where the baseline ai_healthy target is true, so boost-phase flight stays paired with the control.
- provenance.git_head is the checked-out commit at run time; BRK01 and BRK02 source digests are recorded separately so an uncommitted stack is visible.
- The helper was rebuilt from current HEAD; its seed-1 flown sidecar is bit-identical to the frozen instrument, while a small input-frame delta is counted in provenance. No gate threshold was changed.
- No win is claimed unless every unweakened v6 gate passes; a FAIL is a measured miss.
