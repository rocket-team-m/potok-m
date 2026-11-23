# v4 closed-loop A/B — pre-registered gates, current four-stage residual+shield exports

Verdict: **FAIL** (exit 1). This is a post-FIXPUMP simulator measurement, not a flight claim. No win is claimed unless every gate passes.

## Gates (hard-coded, unweakened)

| gate | result | detail |
|---|---|---|
| eval01 rmse_ratio | PASS | verbatim EVAL01 gate |
| eval01 ece_source | PASS | verbatim EVAL01 gate |
| eval01 ece | PASS | verbatim EVAL01 gate |
| eval01 provenance | PASS | verbatim EVAL01 gate |
| eval01 seed_spread | FAIL | see blockers |
| eval01 closed_loop_improvement | FAIL | see blockers |
| eval01 oracle_positive_control | FAIL | see blockers |
| seed 1 absolute gates | FAIL | deployment lateral 1611.99 m <= 1000 m: False; apogee 2137.89 m >= 2000 m: True |
| seed 2 absolute gates | FAIL | deployment lateral 1632.11 m <= 1000 m: False; apogee 1974.90 m >= 2000 m: False |
| seed 3 absolute gates | FAIL | deployment lateral 1642.11 m <= 1000 m: False; apogee 1965.68 m >= 2000 m: False |
| oracle_deployment_and_apogee_positive_control | FAIL | both metrics improve against paired zero command on every seed |

Deployment proxy = first `DESCENT` tick (vz<0 and z>=5) in each flown POTOKTRJ v3 sidecar; the gate uses the pooled RMS of its radial distance from the pad over the 20 held-out episodes (LATERAL01.md:116-127). Apogee is the max z over the whole 30 s trajectory. The repo has no independently scored canopy; nothing here is a landing-dispersion number.

## Per-seed aggregate (metres)

| seed | arm | deployment lateral | apogee | whole-trajectory lateral RMS |
|---:|---|---:|---:|---:|
| 1 | oracle_gtheta | 1690.76 -> 1650.28 | 1973.74 -> 1985.44 | 1309.41 -> 1292.10 |
| 1 | teacher_v2 | 1690.76 -> 1611.99 | 1973.74 -> 2137.89 | 1309.41 -> 1236.59 |
| 2 | oracle_gtheta | 1688.42 -> 1650.63 | 1972.86 -> 1970.71 | 1306.40 -> 1295.89 |
| 2 | teacher_v2 | 1688.42 -> 1632.11 | 1972.86 -> 1974.90 | 1306.40 -> 1294.59 |
| 3 | oracle_gtheta | 1689.19 -> 1653.20 | 1983.00 -> 1975.40 | 1306.38 -> 1298.00 |
| 3 | teacher_v2 | 1689.19 -> 1642.11 | 1983.00 -> 1965.68 | 1306.38 -> 1301.23 |

## Positive control

- passed: **False**
- seed 1: deployment lateral -40.48 m, apogee +11.69 m, actuator cap ticks 16661/16661
- seed 2: deployment lateral -37.79 m, apogee -2.15 m, actuator cap ticks 16584/16584
- seed 3: deployment lateral -35.99 m, apogee -7.60 m, actuator cap ticks 16531/16531

## Provenance

| field | value |
|---|---|
| generated_at_utc | `2026-11-23T22:59:31+00:00` |
| git_head | `34b14d3b06d0cc6e4a38e62c37cfae5f0e80a514` |
| config | `ai/out/v3_final/sim.toml` |
| config_sha256 | `9c34e71f9d10bcf081c31f15f5332f7bb2fac04dddb28009a27cd30987ab5e91` |
| config_hash | `9c34e71f9d10bcf081c31f15f5332f7bb2fac04dddb28009a27cd30987ab5e91` |
| heldout | `ai/out/final_v3_postpump/data/held-out/validation.bin` |
| heldout_sha256 | `fdc1d1400b1c5f111e21f79dead37afa1dd9f058fe457edb78037dc6ebe89807` |
| heldout_manifest_sha256 | `a9ff19947618f17790f1257ef6f7e85ba537915f5a2aca782befefa874db675e` |
| quality_report_sha256 | `8ba904125055787c26e95da39102ecd9dcaa0102c3b66a456c5f265316b5b7d8` |
| exports_dir | `ai/out/final_v3_postpump_dagger/exports` |
| export_manifest_sha256 | `a23406670a2ecec878ed44f393fd970b642ec1fe00407a7daef79ce2ed462820` |
| exports_sha256 | `{"teacher_v2_s1.pt": "372c49a26d833f694be7de77bc85c00df81a13c36369f643161ee0dda1985bf7", "teacher_v2_s2.pt": "278a9aa2cd4f107aecd1ad2fa60efd0254d996832bd5583a82d383a80b3a8b48", "teacher_v2_s3.pt": "42343b043137358119e037304bfcb5d30f340aa9aa577a66a5b141b3cda0423e"}` |
| metrics_sha256 | `{"s1/metrics.json": "7d034ea1af9078d6cb2ea3d3fe7610f4528414f2c3242561a827efe6a918acb2", "s2/metrics.json": "6d37c3048d3b655c75802ab2a17175e7e8ac0d7925434f574682d02005b7e31b", "s3/metrics.json": "bcb79913e6806377c7f44cc73f2c40980ed73a3e170ac0c207cc93bebc23ec2d"}` |
| metrics_config_hash | `{"1": "7bb5bc69ca995bb6", "2": "852f14851e8c745c", "3": "aafe4d567d73f945"}` |
| residual_gain | `{"1": -11.179696083068848, "2": -11.221317291259766, "3": -11.117688179016113}` |
| stage_order | `["sft", "privileged", "dagger", "distill"]` |
| summary_sha256 | `136628551dab7fd9038ceb3d85ec0f48228f8016eb9d0eefa48d760e6a695c9d` |
| harness | `scripts/ab_v4.py` |
| harness_sha256 | `0d4fb47416604883ce64136684910f54e121e34cbedd5ea2f65638ca4da17d5d` |
| helper | `scripts/ab_control_v3/src/main.rs` |
| helper_sha256 | `1f5564e30d66d82e081e5e435bcac3932a862638163cb0fac9a7261920559fed` |
| helper_binary | `scripts/ab_control_v3/target/release/ab-control-v3` |
| helper_binary_sha256 | `c8fea0bbee7c8e8604a8ad8575aee5b0ba12073ae7570766941e935e12935e36` |
| dynamics_sha256 | `367fa527cf46d67c581200201c7575d4f47e0d103a85b9ac9b5a25330af1ad13` |
| dynamics_basis | `post-FIXPUMP: aerodynamic normal force does no positive work (584b1dc); seed-1 zero-command sidecar is bit-identical to the frozen held-out trajectory` |
| baseline_trajectory_sha256 | `{"1": "9b1d680d55f635a3a2cf84864498cb533a84132bbe15ba791154a3cd58bb5953", "2": "f0443872b644520d9f413e11b608241b0d8ea9bea65d5605b5e026ce08597aa2", "3": "b1123a635768bdd5d71b1591326477be3fde9f80a22b4f24288f067234ecbc4d"}` |
| python | `3.13.15` |
| torch | `2.15.0+cpu` |
| numpy | `2.5.5` |
| sample_dt_s | `0.02` |
| command_limit_mps2 | `10.0` |
| deployment_proxy | `first DESCENT tick (vz<0 and z>=5), radial distance from pad` |
| lateral_limit_m | `1000.0` |
| apogee_limit_m | `2000.0` |

## Blockers

- oracle positive control did not improve both metrics on all seeds
- EVAL01 seed_spread failed: lateral gain 2.29% does not beat 3x seed spread 5.17%; apogee gain 2.52% does not beat 3x seed spread 9.19%
- EVAL01 closed_loop_improvement failed: seed 1 lateral -5.56% is above -35%; seed 1 apogee 8.32% is below 15%; seed 2 lateral -0.90% is above -35%; seed 2 apogee 0.10% is below 15%; seed 3 lateral -0.39% is above -35%; seed 3 apogee -0.87% is below 15%
- EVAL01 oracle_positive_control failed: seed 2 oracle did not raise apogee; seed 3 oracle did not raise apogee

## Honesty notes

- Simulator-only: the POTOKTRJ sidecar is emitted by the same simulator that flies the episodes; it is not an independent flight truth anchor.
- No canopy/deployment event is scored; the proxy is the first DESCENT tick and is not a landing-dispersion measurement.
- Commands are generated from the baseline flight's recorded frames and replayed by frame index; after an assisted command moves the vehicle, later frames differ from the baseline flight while the phase/gate schedule stays paired. Commands are emitted only where the baseline ai_healthy target is true, so boost-phase flight stays identical to the zero-command arm.
- The distill target is the applied residual decision (baseline + gain*correction, box projected); deployment adds the export's correction to physics_baseline and clips only at the seam's hard +/-10 m/s^2 bound. No confidence gate, gain, weight or physics constant was introduced or tuned.
- Seeds 2/3 use the seed-matched exports on regenerated held-out flights; a missing export is a blocker, never a stand-in.
- The helper binary was not rebuilt in this shared checkout; its seed-1 zero-command flight is checked bit-for-bit against the frozen post-FIXPUMP held-out sidecar.
- provenance.git_head is the commit that was checked out when the harness ran, before this harness and report were committed locally; no remote action was taken.
- No win is claimed: every gate must pass. A FAIL is a measured miss, not a weaker gate.
