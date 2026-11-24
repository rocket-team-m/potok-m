# v5 closed-loop A/B — unweakened gates, FINAL all-fix four-stage exports

Verdict: **FAIL** (exit 1). This is a post-FIXPUMP simulator measurement, not a flight claim. No win is claimed unless every gate passes.

## Gates (hard-coded, unweakened)

| gate | result | detail |
|---|---|---|
| eval01 rmse_ratio | PASS | verbatim EVAL01 gate |
| eval01 ece_source | PASS | verbatim EVAL01 gate |
| eval01 ece | PASS | verbatim EVAL01 gate |
| eval01 provenance | PASS | verbatim EVAL01 gate |
| eval01 seed_spread | PASS | verbatim EVAL01 gate |
| eval01 closed_loop_improvement | FAIL | see blockers |
| eval01 oracle_positive_control | FAIL | see blockers |
| seed 1 absolute gates | FAIL | deployment lateral 1473.06 m <= 1000 m: False; apogee 2334.55 m >= 2000 m: True |
| seed 2 absolute gates | FAIL | deployment lateral 1558.44 m <= 1000 m: False; apogee 2251.14 m >= 2000 m: True |
| seed 3 absolute gates | FAIL | deployment lateral 1496.79 m <= 1000 m: False; apogee 2305.87 m >= 2000 m: True |
| oracle_deployment_and_apogee_positive_control | FAIL | both metrics improve against paired zero command on every seed |

Deployment proxy = first `DESCENT` tick (vz<0 and z>=5) in each flown POTOKTRJ v3 sidecar; the gate uses the pooled RMS of its radial distance from the pad over the 20 held-out episodes (LATERAL01.md:116-127). Apogee is the max z over the whole 30 s trajectory. The repo has no independently scored canopy; nothing here is a landing-dispersion number.

## Per-seed aggregate (metres)

| seed | arm | deployment lateral | apogee | whole-trajectory lateral RMS |
|---:|---|---:|---:|---:|
| 1 | oracle_gtheta | 1690.76 -> 1650.28 | 1973.74 -> 1985.44 | 1309.41 -> 1292.10 |
| 1 | teacher_v2 | 1690.76 -> 1473.06 | 1973.74 -> 2334.55 | 1309.41 -> 1138.45 |
| 2 | oracle_gtheta | 1688.42 -> 1650.63 | 1972.86 -> 1970.71 | 1306.40 -> 1295.89 |
| 2 | teacher_v2 | 1688.42 -> 1558.44 | 1972.86 -> 2251.14 | 1306.40 -> 1179.44 |
| 3 | oracle_gtheta | 1689.19 -> 1653.20 | 1983.00 -> 1975.40 | 1306.38 -> 1298.00 |
| 3 | teacher_v2 | 1689.19 -> 1496.79 | 1983.00 -> 2305.87 | 1306.38 -> 1145.63 |

## Positive control

- passed: **False**
- seed 1: deployment lateral -40.48 m, apogee +11.69 m, actuator cap ticks 16661/16661
- seed 2: deployment lateral -37.79 m, apogee -2.15 m, actuator cap ticks 16584/16584
- seed 3: deployment lateral -35.99 m, apogee -7.60 m, actuator cap ticks 16531/16531

## Provenance

| field | value |
|---|---|
| generated_at_utc | `2026-11-24T00:09:09+00:00` |
| git_head | `58421e6518a64537db51549dbc9a87c11936d807` |
| config | `ai/out/v3_final/sim.toml` |
| config_sha256 | `9c34e71f9d10bcf081c31f15f5332f7bb2fac04dddb28009a27cd30987ab5e91` |
| config_hash | `9c34e71f9d10bcf081c31f15f5332f7bb2fac04dddb28009a27cd30987ab5e91` |
| heldout | `ai/out/final_v3_postpump/data/held-out/validation.bin` |
| heldout_sha256 | `fdc1d1400b1c5f111e21f79dead37afa1dd9f058fe457edb78037dc6ebe89807` |
| heldout_manifest_sha256 | `a9ff19947618f17790f1257ef6f7e85ba537915f5a2aca782befefa874db675e` |
| quality_report_sha256 | `8ba904125055787c26e95da39102ecd9dcaa0102c3b66a456c5f265316b5b7d8` |
| allfix_root | `ai/out/final_v3_allfix` |
| exports_dir | `ai/out/final_v3_allfix/exports` |
| export_manifest_sha256 | `4c5040f42185d516b9777d58ee8ec0407f291d0c1566b3d32b3781d8d9a2c0b7` |
| shield_sha256 | `ab277e7825ce7b746d09eb57ee57b01eb3866b489411427b995803cd740b0af2` |
| shield_json | `ai/out/final_v3_allfix/shield.json` |
| exports_sha256 | `{"teacher_v2_s1.pt": "2514b4e46148f70945601ebacc1b3eb2ea207ea819cc0f65f2a543375a8775f4", "teacher_v2_s2.pt": "b2a0a521c7e70070c1f704d776200d643e82ec3c58732738b0756e3f9a9bb3b2", "teacher_v2_s3.pt": "6d4d89d2bd896808edbbe40c176354f51c2186681da7fb7d2a91066812f4a968"}` |
| metrics_sha256 | `{"s1/metrics.json": "330423e584057d50779f215bcabad85bb4af07866d1eda9eeb3b2317e1174a36", "s2/metrics.json": "ec54defcd00bde30a46d6dcb2c7855274a54bbca1b592fd8b8b63c866d5a9bd4", "s3/metrics.json": "2f7b963e0c5f0c689e7e250d708f9bae6786ee182055a87dc2c82dedca92e5eb"}` |
| metrics_config_hash | `{"1": "f5b5106b378aac4b", "2": "7b5ad1b86cfcb3f3", "3": "717d2a5ae810d807"}` |
| residual_gain | `{"1": -4.862932205200195, "2": -6.228996276855469, "3": -5.851548671722412}` |
| stage_order | `["sft", "privileged", "dagger", "distill"]` |
| summary_sha256 | `6dab65d91ab4cbf182a81f57ba6838e24093edd50a3886ab968b52a601853ab2` |
| summary_json | `ai/out/final_v3_allfix/summary.json` |
| summary_ab_win_claimed | `False` |
| summary_rlvr_used | `False` |
| allfix_data_provenance | `{"config_sha256": "9c34e71f9d10bcf081c31f15f5332f7bb2fac04dddb28009a27cd30987ab5e91", "heldout_sha256": "fdc1d1400b1c5f111e21f79dead37afa1dd9f058fe457edb78037dc6ebe89807", "train_sha256": "34b35e0ff67f7eeecbd895b2efde8bebcfa159c1a3e78379da6750338b6930b6", "truth_sidecar_sha256": "7f25bb69124638ea7542be99f43330fda1c5248513f1971f6bfd125f5f759cb8"}` |
| ece_source | `{"1": "held-out", "2": "held-out", "3": "held-out"}` |
| ece_held_out | `{"1": 0.01166445300894342, "2": 0.007868364254322439, "3": 0.010767036888121233}` |
| conformal_coverage | `{"1": 0.9352617079889807, "2": 0.9545454545454546, "3": 0.9641873278236914}` |
| allfix_provenance | `{"1": true, "2": true, "3": true}` |
| student_command_bound | `10.0` |
| legacy_student_bound | `False` |
| residual_policy_enabled | `{"1": true, "2": true, "3": true}` |
| privileged_batches | `{"1": 2300, "2": 2300, "3": 2300}` |
| harness | `scripts/ab_v5.py` |
| harness_sha256 | `98cc203a7452a5bc58a043fae64cfc7665da100ce3959a5b558827ddfb079e90` |
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
- EVAL01 closed_loop_improvement failed: seed 1 lateral -13.06% is above -35%; seed 2 lateral -9.72% is above -35%; seed 2 apogee 14.11% is below 15%; seed 3 lateral -12.31% is above -35%
- EVAL01 oracle_positive_control failed: seed 2 oracle did not raise apogee; seed 3 oracle did not raise apogee

## Honesty notes

- Simulator-only: the POTOKTRJ sidecar is emitted by the same simulator that flies the episodes; it is not an independent flight truth anchor.
- No canopy/deployment event is scored; the proxy is the first DESCENT tick and is not a landing-dispersion measurement.
- Commands are generated from the baseline flight's recorded frames and replayed by frame index; after an assisted command moves the vehicle, later frames differ from the baseline flight while the phase/gate schedule stays paired. Commands are emitted only where the baseline ai_healthy target is true, so boost-phase flight stays identical to the zero-command arm.
- The distill target is the applied residual decision (baseline + gain*correction, box projected); deployment adds the export's correction to physics_baseline and clips only at the seam's hard +/-10 m/s^2 bound. No confidence gate, gain, weight or physics constant was introduced or tuned.
- Seeds 2/3 use the seed-matched exports on regenerated held-out flights; a missing export is a blocker, never a stand-in.
- Primary arm = ai/out/final_v3_allfix/exports/teacher_v2_s{1,2,3}.pt, the FINAL all-fix four-stage distill exports (AIFIX1/2/3 + SHIELD01/02). The harness refuses to run unless the export manifest, summary, shield and per-seed metrics still carry that evidence; no older export set is substituted.
- The all-fix fixes are wiring/provenance fixes validated on a simulator surrogate: world_step and the POTOKTRJ sidecar share a source. A pass here is a simulator measurement under the pre-registered gates, not an independent flight-truth claim.
- The helper binary was not rebuilt in this shared checkout; its seed-1 zero-command flight is checked bit-for-bit against the frozen post-FIXPUMP held-out sidecar.
- provenance.git_head is the commit that was checked out when the harness ran, before this harness and report were committed locally; no remote action was taken.
- No win is claimed: every gate must pass. A FAIL is a measured miss, not a weaker gate.
