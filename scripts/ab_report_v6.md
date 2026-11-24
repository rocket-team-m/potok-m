# v6 closed-loop A/B — unweakened gates, RETRAIN-NIGHT default vs adversarial

Overall verdict: **FAIL** (exit 1). A win is claimed only if every gate of every arm passes. This is a post-FIXPUMP simulator measurement, not a flight claim.

| arm | verdict | gates failed | blockers | win claimed |
|---|---|---:|---:|---|
| night_default | FAIL | 7 | 4 | False |
| night_adversarial | FAIL | 6 | 3 | False |

## Gates (hard-coded, unweakened AIPOST1/EVAL01)

Deployment proxy = first `DESCENT` tick (vz<0 and z>=5) in each flown POTOKTRJ v3 sidecar; the absolute gate uses the pooled RMS of its radial distance from the pad over the 20 held-out episodes (LATERAL01.md:116-127). Apogee is the max z over the whole 30 s trajectory. The repo has no independently scored canopy; nothing here is a landing-dispersion number.

### night_default — RETRAIN-NIGHT default four-stage distill exports

| gate | result | detail |
|---|---|---|
| eval01 rmse_ratio | PASS | verbatim EVAL01 gate |
| eval01 ece_source | PASS | verbatim EVAL01 gate |
| eval01 ece | PASS | verbatim EVAL01 gate |
| eval01 provenance | PASS | verbatim EVAL01 gate |
| eval01 seed_spread | FAIL | see blockers |
| eval01 closed_loop_improvement | FAIL | see blockers |
| eval01 oracle_positive_control | FAIL | see blockers |
| seed 1 absolute gates | FAIL | deployment lateral 1462.98 m <= 1000 m: False; apogee 2366.25 m >= 2000 m: True |
| seed 2 absolute gates | FAIL | deployment lateral 1560.57 m <= 1000 m: False; apogee 2213.94 m >= 2000 m: True |
| seed 3 absolute gates | FAIL | deployment lateral 1454.26 m <= 1000 m: False; apogee 2327.48 m >= 2000 m: True |
| oracle_deployment_and_apogee_positive_control | FAIL | both metrics improve against paired zero command on every seed |

| seed | arm | deployment lateral | apogee | whole-trajectory lateral RMS |
|---:|---|---:|---:|---:|
| 1 | oracle_gtheta | 1690.76 -> 1650.28 | 1973.74 -> 1985.44 | 1309.41 -> 1292.10 |
| 1 | night_default | 1690.76 -> 1462.98 | 1973.74 -> 2366.25 | 1309.41 -> 1118.55 |
| 2 | oracle_gtheta | 1688.42 -> 1650.63 | 1972.86 -> 1970.71 | 1306.40 -> 1295.89 |
| 2 | night_default | 1688.42 -> 1560.57 | 1972.86 -> 2213.94 | 1306.40 -> 1182.50 |
| 3 | oracle_gtheta | 1689.19 -> 1653.20 | 1983.00 -> 1975.40 | 1306.38 -> 1298.00 |
| 3 | night_default | 1689.19 -> 1454.26 | 1983.00 -> 2327.48 | 1306.38 -> 1129.39 |

Positive control passed: **False**

- seed 1: deployment lateral -40.48 m, apogee +11.69 m, actuator cap ticks 16661/16661
- seed 2: deployment lateral -37.79 m, apogee -2.15 m, actuator cap ticks 16584/16584
- seed 3: deployment lateral -35.99 m, apogee -7.60 m, actuator cap ticks 16531/16531

Blockers:

- EVAL01 seed_spread failed: lateral gain 12.54% does not beat 3x seed spread 5.09%; apogee gain 16.49% does not beat 3x seed spread 7.67%
- EVAL01 closed_loop_improvement failed: seed 1 lateral -14.58% is above -35%; seed 2 lateral -9.48% is above -35%; seed 2 apogee 12.22% is below 15%; seed 3 lateral -13.55% is above -35%
- EVAL01 oracle_positive_control failed: seed 2 oracle did not raise apogee; seed 3 oracle did not raise apogee
- night_default: oracle positive control did not improve both metrics on all seeds

### night_adversarial — RETRAIN-NIGHT opt-in adversarial four-stage distill students

| gate | result | detail |
|---|---|---|
| eval01 rmse_ratio | PASS | verbatim EVAL01 gate |
| eval01 ece_source | PASS | verbatim EVAL01 gate |
| eval01 ece | PASS | verbatim EVAL01 gate |
| eval01 provenance | PASS | verbatim EVAL01 gate |
| eval01 seed_spread | PASS | verbatim EVAL01 gate |
| eval01 closed_loop_improvement | FAIL | see blockers |
| eval01 oracle_positive_control | FAIL | see blockers |
| seed 1 absolute gates | FAIL | deployment lateral 1458.08 m <= 1000 m: False; apogee 2347.99 m >= 2000 m: True |
| seed 2 absolute gates | FAIL | deployment lateral 1553.58 m <= 1000 m: False; apogee 2252.12 m >= 2000 m: True |
| seed 3 absolute gates | FAIL | deployment lateral 1456.54 m <= 1000 m: False; apogee 2308.26 m >= 2000 m: True |
| oracle_deployment_and_apogee_positive_control | FAIL | both metrics improve against paired zero command on every seed |

| seed | arm | deployment lateral | apogee | whole-trajectory lateral RMS |
|---:|---|---:|---:|---:|
| 1 | oracle_gtheta | 1690.76 -> 1650.28 | 1973.74 -> 1985.44 | 1309.41 -> 1292.10 |
| 1 | night_adversarial | 1690.76 -> 1458.08 | 1973.74 -> 2347.99 | 1309.41 -> 1132.33 |
| 2 | oracle_gtheta | 1688.42 -> 1650.63 | 1972.86 -> 1970.71 | 1306.40 -> 1295.89 |
| 2 | night_adversarial | 1688.42 -> 1553.58 | 1972.86 -> 2252.12 | 1306.40 -> 1179.59 |
| 3 | oracle_gtheta | 1689.19 -> 1653.20 | 1983.00 -> 1975.40 | 1306.38 -> 1298.00 |
| 3 | night_adversarial | 1689.19 -> 1456.54 | 1983.00 -> 2308.26 | 1306.38 -> 1142.84 |

Positive control passed: **False**

- seed 1: deployment lateral -40.48 m, apogee +11.69 m, actuator cap ticks 16661/16661
- seed 2: deployment lateral -37.79 m, apogee -2.15 m, actuator cap ticks 16584/16584
- seed 3: deployment lateral -35.99 m, apogee -7.60 m, actuator cap ticks 16531/16531

Blockers:

- EVAL01 closed_loop_improvement failed: seed 1 lateral -13.52% is above -35%; seed 2 lateral -9.71% is above -35%; seed 2 apogee 14.16% is below 15%; seed 3 lateral -12.52% is above -35%
- EVAL01 oracle_positive_control failed: seed 2 oracle did not raise apogee; seed 3 oracle did not raise apogee
- night_adversarial: oracle positive control did not improve both metrics on all seeds

## Shared provenance

| field | value |
|---|---|
| generated_at_utc | `2026-11-24T02:04:12+00:00` |
| git_head | `0fe6a4e3bd98db9af49b8d91dbc01604760162c4` |
| config | `ai/out/v3_final/sim.toml` |
| config_sha256 | `9c34e71f9d10bcf081c31f15f5332f7bb2fac04dddb28009a27cd30987ab5e91` |
| config_hash | `9c34e71f9d10bcf081c31f15f5332f7bb2fac04dddb28009a27cd30987ab5e91` |
| heldout | `ai/out/final_v3_postpump/data/held-out/validation.bin` |
| heldout_sha256 | `fdc1d1400b1c5f111e21f79dead37afa1dd9f058fe457edb78037dc6ebe89807` |
| heldout_manifest_sha256 | `a9ff19947618f17790f1257ef6f7e85ba537915f5a2aca782befefa874db675e` |
| quality_report_sha256 | `8ba904125055787c26e95da39102ecd9dcaa0102c3b66a456c5f265316b5b7d8` |
| night_root | `ai/out/night_final` |
| summary_sha256 | `d0128c36d2f06cde2147a6a91c30253494ceb37676c6d1f2e40622f6fb52cd77` |
| adversarial_summary_sha256 | `9c1b76add587803100e17f9d8e551ef805acdc6940187a2fa82b0016bb7535aa` |
| export_manifest_sha256 | `71ab5bbd2b79309665a19db49552b872c9753f2420be0ce7db035221758db1c7` |
| shield_sha256 | `ab277e7825ce7b746d09eb57ee57b01eb3866b489411427b995803cd740b0af2` |
| harness | `scripts/ab_v6.py` |
| harness_sha256 | `2da97169fc0f355c2d655f56985e963b1a578eee64bcf715b9fc8a2c2acc2463` |
| helper | `scripts/ab_control_v3/src/main.rs` |
| helper_sha256 | `1f5564e30d66d82e081e5e435bcac3932a862638163cb0fac9a7261920559fed` |
| helper_binary | `scripts/ab_control_v3/target/release/ab-control-v3` |
| helper_binary_sha256 | `c8fea0bbee7c8e8604a8ad8575aee5b0ba12073ae7570766941e935e12935e36` |
| dynamics_sha256 | `6159cf4bd24862a8ac89f97529e1133d49ef6bd8f6ba2289f2b4d70987063cc2` |
| dynamics_basis | `post-FIXPUMP: aerodynamic normal force does no positive work (584b1dc); seed-1 zero-command sidecar is bit-identical to the frozen held-out trajectory` |
| baseline_trajectory_sha256 | `{"1": "9b1d680d55f635a3a2cf84864498cb533a84132bbe15ba791154a3cd58bb5953", "2": "f0443872b644520d9f413e11b608241b0d8ea9bea65d5605b5e026ce08597aa2", "3": "b1123a635768bdd5d71b1591326477be3fde9f80a22b4f24288f067234ecbc4d"}` |
| arms | `["night_default", "night_adversarial"]` |
| python | `3.13.15` |
| torch | `2.15.0+cpu` |
| numpy | `2.5.5` |
| sample_dt_s | `0.02` |
| command_limit_mps2 | `10.0` |
| deployment_proxy | `first DESCENT tick (vz<0 and z>=5), radial distance from pad` |
| lateral_limit_m | `1000.0` |
| apogee_limit_m | `2000.0` |

## Honesty notes

- Simulator-only: the POTOKTRJ sidecar is emitted by the same simulator that flies the episodes; it is not an independent flight truth anchor.
- No canopy/deployment event is scored; the proxy is the first DESCENT tick and is not a landing-dispersion measurement.
- Commands are generated from the baseline flight's recorded frames and replayed by frame index; after an assisted command moves the vehicle, later frames differ from the baseline flight while the phase/gate schedule stays paired. Commands are emitted only where the baseline ai_healthy target is true, so boost-phase flight stays identical to the zero-command arm.
- The distill target is the applied residual decision (baseline + gain*correction, box projected); deployment adds the export's correction to physics_baseline and clips only at the seam's hard +/-10 m/s^2 bound. No confidence gate, gain, weight or physics constant was introduced or tuned.
- Seeds 2/3 use seed-matched exports on regenerated held-out flights; a missing export is a blocker, never a stand-in.
- The two arms are separate opt-in retrains of the same four-stage teacher-v2 default; adversarial differs only by --adversarial. Each arm carries its own gates and verdict. No arm-to-arm winner is declared: both are compared to the paired zero-command control under the same unweakened gates.
- The helper binary was not rebuilt in this shared checkout; its seed-1 zero-command flight is checked bit-for-bit against the frozen post-FIXPUMP held-out sidecar.
- provenance.git_head is the commit that was checked out when the harness ran, before this harness and report were committed locally; no remote action was taken.
- No win is claimed unless every gate of every arm passes. A FAIL is a measured miss, not a weaker gate.
