# BRK04 closed-loop A/B — BRK03-fixed braking-teacher student under unweakened v6 gates

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
| seed 1 absolute gates | FAIL | deployment lateral 1503.98 m <= 1000 m: False; apogee 2294.88 m >= 2000 m: True |
| seed 2 absolute gates | FAIL | deployment lateral 1539.51 m <= 1000 m: False; apogee 2270.66 m >= 2000 m: True |
| seed 3 absolute gates | FAIL | deployment lateral 1521.34 m <= 1000 m: False; apogee 2277.60 m >= 2000 m: True |
| oracle_deployment_and_apogee_positive_control | FAIL | both metrics improve against paired zero command on every seed |

| seed | arm | deployment lateral (m) | apogee (m) | whole-trajectory lateral RMS (m) |
|---:|---|---:|---:|---:|
| 1 | oracle_gtheta | 1690.76 -> 1650.28 | 1973.74 -> 1985.44 | 1309.41 -> 1292.10 |
| 1 | braking_v2 | 1690.76 -> 1503.98 | 1973.74 -> 2294.88 | 1309.41 -> 1136.92 |
| 2 | oracle_gtheta | 1688.42 -> 1650.63 | 1972.86 -> 1970.71 | 1306.40 -> 1295.89 |
| 2 | braking_v2 | 1688.42 -> 1539.51 | 1972.86 -> 2270.66 | 1306.40 -> 1162.86 |
| 3 | oracle_gtheta | 1689.19 -> 1653.20 | 1983.00 -> 1975.40 | 1306.38 -> 1298.00 |
| 3 | braking_v2 | 1689.19 -> 1521.34 | 1983.00 -> 2277.60 | 1306.38 -> 1160.33 |

Positive control passed: **False**

- seed 1: deployment lateral -40.48 m, apogee +11.69 m, cap ticks 16661/16661
- seed 2: deployment lateral -37.79 m, apogee -2.15 m, cap ticks 16584/16584
- seed 3: deployment lateral -35.99 m, apogee -7.60 m, cap ticks 16531/16531

BRK03 privileged target histogram (banned + active rows; cap bucket counted):

| seed | boost_gate | ok | cap_clamped | active |
|---:|---:|---:|---:|---:|
| 1 | 487 | 964 | 0 | 964 |
| 2 | 487 | 964 | 0 | 964 |
| 3 | 487 | 964 | 0 | 964 |

Blockers:

- EVAL01 closed_loop_improvement failed: seed 1 lateral -13.17% is above -35%; seed 2 lateral -10.99% is above -35%; seed 3 lateral -11.18% is above -35%; seed 3 apogee 14.86% is below 15%
- EVAL01 oracle_positive_control failed: seed 2 oracle did not raise apogee; seed 3 oracle did not raise apogee
- braking_v2: oracle positive control did not improve both metrics on all seeds

## Verdict

`pass`=False, `win_claimed`=False, gates failed=6. A measured FAIL: no win is claimed and no gate was weakened.

## Provenance

| field | value |
|---|---|
| git_head | `6f44a25b6f24d668c5cf1cc9039b392f934bef35` |
| harness | `scripts/ab_braking_v2.py` |
| harness_sha256 | `d51dedd5c55b5b295d8b53d9aa1b83f9a9b27a417f48ad3072c688f46b0fe3f1` |
| objective_source_sha256 | `64400853113e70b60868add49d63b0f313a6c84b6d0a4801ebddee47be628411` |
| objective_version | `TRUTH_VEL_START=3,TRUTH_VEL_END=6` |
| objective_head | `6f44a25b6f24d668c5cf1cc9039b392f934bef35` |
| config_sha256 | `9c34e71f9d10bcf081c31f15f5332f7bb2fac04dddb28009a27cd30987ab5e91` |
| heldout_sha256 | `fdc1d1400b1c5f111e21f79dead37afa1dd9f058fe457edb78037dc6ebe89807` |
| summary_sha256 | `68868be7bd29e52455d5d3e9981b314b8864cbccee27fa1893cf72d49888691e` |
| export_manifest_sha256 | `b4b41fe863b72a71c20467f460de5f24452327df66a1b0b5fad2ac475a54aabd` |
| brk03_version | `BRK03 one-way ACTIVE_PHASES whitelist + persistent per-episode window tick cursor` |
| exports_sha256 | `{"1": "4ec9decf073f0611aad338f2ad34e5f03cd21b39ed30bb7afd7f8cdc6425c85f", "2": "90220137ea815c6975189c9fd68c04c8c028b7b0377b5b63c0bef5cee9e98873", "3": "61083ee15f932b1548379c68c39943a6d167860b517624e75c3ebbbb3cf38596"}` |
| command_limit_mps2 | `10.0` |
| deployment_proxy | `first DESCENT tick (vz<0 and z>=5), radial distance from pad` |
| lateral_limit_m | `1000.0` |
| apogee_limit_m | `2000.0` |

## Honesty notes

- Simulator-only: the POTOKTRJ sidecar is emitted by the same simulator that flies the episodes; it is not an independent flight truth anchor.
- No canopy/deployment event is scored; the proxy is the first DESCENT tick and is not a landing-dispersion measurement.
- The braking teacher is privileged simulator truth. This A/B measures the distilled student's closed-loop behaviour, not teacher fidelity or flight.
- Commands are generated from the baseline flight's recorded frames and replayed by frame index; they are emitted only where the baseline ai_healthy target is true, so boost-phase flight stays paired with the control.
- provenance.git_head is the checked-out commit at run time; BRK03 source digests are recorded separately so an uncommitted stack is visible.
- The helper was rebuilt from current HEAD; its seed-1 flown sidecar is bit-identical to the frozen instrument, while a small input-frame delta is counted in provenance. No gate threshold was changed.
- No win is claimed unless every unweakened v6 gate passes; a FAIL is a measured miss.
