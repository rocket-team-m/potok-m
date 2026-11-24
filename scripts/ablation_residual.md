# Residual-policy deployment ablation — unweakened gates, FINAL all-fix exports

Verdict: **FAIL** (exit 1). Simulator measurement only; no win is claimed unless every gate passes.

## Arms (deployment composition only; no retraining, no tuning)

| arm | command written to the frozen seam |
|---|---|
| zero_command | no TSV; Simulator::run_episode ControlMode::None; paired gate control |
| oracle_gtheta | run_episode_with_oracle / fin_accel_command; positive control |
| residual_on | clip(physics_baseline + export_delta); shipped all-fix v5 arm |
| residual_off | clip(export_delta); classical superposition removed |
| classical_only | clip(physics_baseline); learned correction removed = ResidualPolicy(enabled=False) |

## Model quality (exact values from allfix metrics.json)

| seed | rmse_train | rmse_held_out | ratio | ece_held_out | coverage | residual_gain |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.658076677982967 | 5.148904169724766 | 0.9100096132950048 | 0.01166445300894342 | 0.9352617079889807 | -4.862932205200195 |
| 2 | 9.595020625413378 | 10.344431805348933 | 1.0781041760296652 | 0.007868364254322439 | 0.9545454545454546 | -6.228996276855469 |
| 3 | 8.446755869407314 | 8.07331851231966 | 0.9557892565073202 | 0.010767036888121233 | 0.9641873278236914 | -5.851548671722412 |

## Flight metrics (metres; baseline -> assisted, pooled over 20 held-out episodes)

| seed | arm | deployment lateral | apogee | whole-trajectory lateral RMS |
|---:|---|---:|---:|---:|
| 1 | residual_on | 1690.756426 -> 1473.061037 | 1973.743109 -> 2334.545404 | 1309.408733 -> 1138.447709 |
| 1 | residual_off | 1690.756426 -> 1564.528569 | 1973.743109 -> 2184.161621 | 1309.408733 -> 1190.708314 |
| 1 | classical_only | 1690.756426 -> 1658.801783 | 1973.743109 -> 1922.569977 | 1309.408733 -> 1319.533069 |
| 2 | residual_on | 1688.418578 -> 1558.442532 | 1972.858569 -> 2251.142731 | 1306.396947 -> 1179.441255 |
| 2 | residual_off | 1688.418578 -> 1611.285207 | 1972.858569 -> 2105.717175 | 1306.396947 -> 1225.190702 |
| 2 | classical_only | 1688.418578 -> 1656.372738 | 1972.858569 -> 1920.682275 | 1306.396947 -> 1317.047522 |
| 3 | residual_on | 1689.185836 -> 1496.789353 | 1983.003363 -> 2305.874841 | 1306.380164 -> 1145.629080 |
| 3 | residual_off | 1689.185836 -> 1555.356834 | 1983.003363 -> 2185.899200 | 1306.380164 -> 1183.239723 |
| 3 | classical_only | 1689.185836 -> 1657.321562 | 1983.003363 -> 1931.668890 | 1306.380164 -> 1316.336506 |

## Improvement vs paired zero command (whole-trajectory RMS)

| seed | residual_on | residual_off | classical_only |
|---:|---:|---:|---:|
| 1 | -13.06% / +18.28% | -9.07% / +10.66% | +0.77% / -2.59% |
| 2 | -9.72% / +14.11% | -6.22% / +6.73% | +0.82% / -2.64% |
| 3 | -12.31% / +16.28% | -9.43% / +10.23% | +0.76% / -2.59% |

## Gates (hard-coded, unweakened)

| gate | result | detail |
|---|---|---|
| eval01 rmse_ratio | PASS | verbatim EVAL01 gate |
| eval01 ece_source | PASS | verbatim EVAL01 gate |
| eval01 ece | PASS | verbatim EVAL01 gate |
| eval01 provenance | PASS | verbatim EVAL01 gate |
| eval01 seed_spread [residual_on] | PASS | both gains beat 3x the spread of seeds [1, 2, 3] |
| eval01 improvement [residual_on] | FAIL | seed 1 lateral -13.06% is above -35%; seed 2 lateral -9.72% is above -35%; seed 2 apogee 14.11% is below 15%; seed 3 lateral -12.31% is above -35% |
| eval01 seed_spread [residual_off] | FAIL | lateral gain 8.24% does not beat 3x seed spread 3.21%; apogee gain 9.21% does not beat 3x seed spread 3.93% |
| eval01 improvement [residual_off] | FAIL | seed 1 lateral -9.07% is above -35%; seed 1 apogee 10.66% is below 15%; seed 2 lateral -6.22% is above -35%; seed 2 apogee 6.73% is below 15%; seed 3 lateral -9.43% is above -35%; seed 3 apogee 10.23% is below 15% |
| eval01 seed_spread [classical_only] | FAIL | lateral gain -0.78% does not beat 3x seed spread 0.05%; apogee gain -2.61% does not beat 3x seed spread 0.06% |
| eval01 improvement [classical_only] | FAIL | seed 1 lateral 0.77% is above -35%; seed 1 apogee -2.59% is below 15%; seed 2 lateral 0.82% is above -35%; seed 2 apogee -2.64% is below 15%; seed 3 lateral 0.76% is above -35%; seed 3 apogee -2.59% is below 15% |
| eval01 oracle_positive_control | FAIL | seed 2 oracle did not raise apogee; seed 3 oracle did not raise apogee |
| oracle_deployment_and_apogee_positive_control | FAIL | both metrics improve against paired zero command on every seed |
| absolute [residual_on] seed 1 | FAIL | deployment lateral 1473.061037 m <= 1000 m: False; apogee 2334.545404 m >= 2000 m: True |
| absolute [residual_on] seed 2 | FAIL | deployment lateral 1558.442532 m <= 1000 m: False; apogee 2251.142731 m >= 2000 m: True |
| absolute [residual_on] seed 3 | FAIL | deployment lateral 1496.789353 m <= 1000 m: False; apogee 2305.874841 m >= 2000 m: True |
| absolute [residual_off] seed 1 | FAIL | deployment lateral 1564.528569 m <= 1000 m: False; apogee 2184.161621 m >= 2000 m: True |
| absolute [residual_off] seed 2 | FAIL | deployment lateral 1611.285207 m <= 1000 m: False; apogee 2105.717175 m >= 2000 m: True |
| absolute [residual_off] seed 3 | FAIL | deployment lateral 1555.356834 m <= 1000 m: False; apogee 2185.899200 m >= 2000 m: True |
| absolute [classical_only] seed 1 | FAIL | deployment lateral 1658.801783 m <= 1000 m: False; apogee 1922.569977 m >= 2000 m: False |
| absolute [classical_only] seed 2 | FAIL | deployment lateral 1656.372738 m <= 1000 m: False; apogee 1920.682275 m >= 2000 m: False |
| absolute [classical_only] seed 3 | FAIL | deployment lateral 1657.321562 m <= 1000 m: False; apogee 1931.668890 m >= 2000 m: False |

## Positive control

- passed: **False**
- seed 1: deployment lateral -40.476380 m, apogee +11.692065 m, actuator cap ticks 16661/16661
- seed 2: deployment lateral -37.785985 m, apogee -2.151044 m, actuator cap ticks 16584/16584
- seed 3: deployment lateral -35.988458 m, apogee -7.601099 m, actuator cap ticks 16531/16531

## Fidelity to the shipped v5 A/B

- checked: True; passed: **True**; residual_on reproduces scripts/ab_report_v5.json assisted aggregates

## Provenance

| field | value |
|---|---|
| generated_at_utc | `2026-11-24T00:25:43+00:00` |
| git_head | `07a5e43ebcd123065227665a5d6b71c46f0acd7d` |
| config | `ai/out/v3_final/sim.toml` |
| config_sha256 | `9c34e71f9d10bcf081c31f15f5332f7bb2fac04dddb28009a27cd30987ab5e91` |
| config_hash | `9c34e71f9d10bcf081c31f15f5332f7bb2fac04dddb28009a27cd30987ab5e91` |
| heldout | `ai/out/final_v3_postpump/data/held-out/validation.bin` |
| heldout_sha256 | `fdc1d1400b1c5f111e21f79dead37afa1dd9f058fe457edb78037dc6ebe89807` |
| heldout_manifest_sha256 | `a9ff19947618f17790f1257ef6f7e85ba537915f5a2aca782befefa874db675e` |
| quality_report_sha256 | `8ba904125055787c26e95da39102ecd9dcaa0102c3b66a456c5f265316b5b7d8` |
| allfix_root | `ai/out/final_v3_allfix` |
| export_manifest_sha256 | `4c5040f42185d516b9777d58ee8ec0407f291d0c1566b3d32b3781d8d9a2c0b7` |
| summary_sha256 | `6dab65d91ab4cbf182a81f57ba6838e24093edd50a3886ab968b52a601853ab2` |
| shield_sha256 | `ab277e7825ce7b746d09eb57ee57b01eb3866b489411427b995803cd740b0af2` |
| exports_sha256 | `{"teacher_v2_s1.pt": "2514b4e46148f70945601ebacc1b3eb2ea207ea819cc0f65f2a543375a8775f4", "teacher_v2_s2.pt": "b2a0a521c7e70070c1f704d776200d643e82ec3c58732738b0756e3f9a9bb3b2", "teacher_v2_s3.pt": "6d4d89d2bd896808edbbe40c176354f51c2186681da7fb7d2a91066812f4a968"}` |
| metrics_sha256 | `{"s1/metrics.json": "330423e584057d50779f215bcabad85bb4af07866d1eda9eeb3b2317e1174a36", "s2/metrics.json": "ec54defcd00bde30a46d6dcb2c7855274a54bbca1b592fd8b8b63c866d5a9bd4", "s3/metrics.json": "2f7b963e0c5f0c689e7e250d708f9bae6786ee182055a87dc2c82dedca92e5eb"}` |
| metrics_config_hash | `{"1": "f5b5106b378aac4b", "2": "7b5ad1b86cfcb3f3", "3": "717d2a5ae810d807"}` |
| ece_source | `{"1": "held-out", "2": "held-out", "3": "held-out"}` |
| ece_held_out | `{"1": 0.01166445300894342, "2": 0.007868364254322439, "3": 0.010767036888121233}` |
| conformal_coverage | `{"1": 0.9352617079889807, "2": 0.9545454545454546, "3": 0.9641873278236914}` |
| allfix_provenance | `{"1": true, "2": true, "3": true}` |
| residual_gain | `{"1": -4.862932205200195, "2": -6.228996276855469, "3": -5.851548671722412}` |
| residual_policy_enabled | `{"1": true, "2": true, "3": true}` |
| stage_order | `["sft", "privileged", "dagger", "distill"]` |
| summary_ab_win_claimed | `False` |
| summary_rlvr_used | `False` |
| allfix_data_provenance | `{"config_sha256": "9c34e71f9d10bcf081c31f15f5332f7bb2fac04dddb28009a27cd30987ab5e91", "heldout_sha256": "fdc1d1400b1c5f111e21f79dead37afa1dd9f058fe457edb78037dc6ebe89807", "train_sha256": "34b35e0ff67f7eeecbd895b2efde8bebcfa159c1a3e78379da6750338b6930b6", "truth_sidecar_sha256": "7f25bb69124638ea7542be99f43330fda1c5248513f1971f6bfd125f5f759cb8"}` |
| student_command_bound | `10.0` |
| legacy_student_bound | `False` |
| arm_definitions | `{"classical_only": "clip(physics_baseline); learned correction removed = ResidualPolicy(enabled=False)", "oracle_gtheta": "run_episode_with_oracle / fin_accel_command; positive control", "residual_off": "clip(export_delta); classical superposition removed", "residual_on": "clip(physics_baseline + export_delta); shipped all-fix v5 arm", "zero_command": "no TSV; Simulator::run_episode ControlMode::None; paired gate control"}` |
| source_of_off_semantics | `residual_policy.py:139-149 (gain*correction, enabled=False -> baseline.clone()); physics.py:53-56 final_command(residual=True) = baseline + prediction` |
| harness | `scripts/ablation_residual.py` |
| harness_sha256 | `2fa2269374d458fb8e78a4bc50c55c6cf5981da4779781efa10b79a5130d91d6` |
| helper | `scripts/ab_control_v3/src/main.rs` |
| helper_sha256 | `1f5564e30d66d82e081e5e435bcac3932a862638163cb0fac9a7261920559fed` |
| helper_binary | `scripts/ab_control_v3/target/release/ab-control-v3` |
| helper_binary_sha256 | `c8fea0bbee7c8e8604a8ad8575aee5b0ba12073ae7570766941e935e12935e36` |
| dynamics_sha256 | `6159cf4bd24862a8ac89f97529e1133d49ef6bd8f6ba2289f2b4d70987063cc2` |
| v5_harness_sha256 | `98cc203a7452a5bc58a043fae64cfc7665da100ce3959a5b558827ddfb079e90` |
| v5_report_sha256 | `ae61b430b957c7d86d65de9ca83658dbfb7cb5448bc587e8dc2452ff6328ae9a` |
| baseline_trajectory_sha256 | `{"1": "9b1d680d55f635a3a2cf84864498cb533a84132bbe15ba791154a3cd58bb5953", "2": "f0443872b644520d9f413e11b608241b0d8ea9bea65d5605b5e026ce08597aa2", "3": "b1123a635768bdd5d71b1591326477be3fde9f80a22b4f24288f067234ecbc4d"}` |
| assisted_trajectory_sha256 | `{"1/classical_only": "873752c26fb53e57f81b333d00341d44397a2fece5357ea0a7ed917a243bf813", "1/oracle_gtheta": "1d9a28ec83dfac5f9d16e84562450b1fd3317dbd2587885568561d2dca8fca15", "1/residual_off": "ac87d70a05a312066a287a8a500d2af41843172d882f9f4854ca77d75d5c993b", "1/residual_on": "32e4d882333d468f7e946bb09f31183adeff70d52a134b1d9116b10df181e357", "1/zero_command": "9b1d680d55f635a3a2cf84864498cb533a84132bbe15ba791154a3cd58bb5953", "2/classical_only": "ea1e1b84097523701b8b5cc0e11398bf4f12d8b43597cedeab0c195467a8dba5", "2/oracle_gtheta": "92cfa601580d151f83c89d9a8c9ee5790b06118f4b80fe17cc6b188c81c02336", "2/residual_off": "168b470149793c8a6723cd34698471ed5e2f54948093483dd504e70c46d76f4d", "2/residual_on": "d70b48c6c79a3a6d831219aa7b07b9111e98e70df7c29666102e02e470c15c8e", "2/zero_command": "f0443872b644520d9f413e11b608241b0d8ea9bea65d5605b5e026ce08597aa2", "3/classical_only": "6b8ef1fcf049b9d28f32672e1697fccee2b0ec2f6a4b6d1f8ff0388551b1954f", "3/oracle_gtheta": "68973c0f50d554065525c0093aa4deddc72ec69828d2e29d7dbc88e02f9e607d", "3/residual_off": "44342c35c79dabfda0bd5c0b0f15d9920df3413dd960d0fe881ca1ccab931c44", "3/residual_on": "0374ec38eace9089f1f9d0db83ff0f7227c045eba2eb3107b278ca3126d645b8", "3/zero_command": "b1123a635768bdd5d71b1591326477be3fde9f80a22b4f24288f067234ecbc4d"}` |
| python | `3.13.15` |
| torch | `2.15.0+cpu` |
| numpy | `2.5.5` |
| sample_dt_s | `0.02` |
| command_limit_mps2 | `10.0` |
| deployment_proxy | `first DESCENT tick (vz<0 and z>=5), radial distance from pad` |
| lateral_limit_m | `1000.0` |
| apogee_limit_m | `2000.0` |

## Blockers

- EVAL01 improvement (residual_on) failed: seed 1 lateral -13.06% is above -35%; seed 2 lateral -9.72% is above -35%; seed 2 apogee 14.11% is below 15%; seed 3 lateral -12.31% is above -35%
- EVAL01 seed_spread (residual_off) failed: lateral gain 8.24% does not beat 3x seed spread 3.21%; apogee gain 9.21% does not beat 3x seed spread 3.93%
- EVAL01 improvement (residual_off) failed: seed 1 lateral -9.07% is above -35%; seed 1 apogee 10.66% is below 15%; seed 2 lateral -6.22% is above -35%; seed 2 apogee 6.73% is below 15%; seed 3 lateral -9.43% is above -35%; seed 3 apogee 10.23% is below 15%
- EVAL01 seed_spread (classical_only) failed: lateral gain -0.78% does not beat 3x seed spread 0.05%; apogee gain -2.61% does not beat 3x seed spread 0.06%
- EVAL01 improvement (classical_only) failed: seed 1 lateral 0.77% is above -35%; seed 1 apogee -2.59% is below 15%; seed 2 lateral 0.82% is above -35%; seed 2 apogee -2.64% is below 15%; seed 3 lateral 0.76% is above -35%; seed 3 apogee -2.59% is below 15%
- EVAL01 oracle_positive_control failed: seed 2 oracle did not raise apogee; seed 3 oracle did not raise apogee
- oracle positive control did not improve both metrics on all seeds
- absolute gate failed: residual_on seed 1
- absolute gate failed: residual_on seed 2
- absolute gate failed: residual_on seed 3
- absolute gate failed: residual_off seed 1
- absolute gate failed: residual_off seed 2
- absolute gate failed: residual_off seed 3
- absolute gate failed: classical_only seed 1
- absolute gate failed: classical_only seed 2
- absolute gate failed: classical_only seed 3

## Honesty notes

- Simulator-only: the POTOKTRJ sidecar is emitted by the same simulator that flies the episodes; it is not an independent flight-truth anchor.
- No canopy/deployment event is scored; the proxy is the first DESCENT tick and is not a landing-dispersion measurement.
- Commands are generated from the zero-command flight's recorded frames and replayed by frame index; after a command moves the vehicle, later frames differ from that flight while the phase/gate schedule stays paired. Commands are emitted only where the paired baseline ai_healthy target is true, so boost-phase flight stays identical to control.
- No export was retrained. residual_off removes the classical baseline at deployment and applies the residual-trained delta as an absolute command; classical_only removes the learned delta and applies physics_baseline; both are diagnostic interpretations of 'residual OFF' and neither is presented as a shipped policy.
- Seeds 2/3 use seed-matched exports on regenerated held-out flights; a missing export is a loud blocker, never a stand-in.
- Primary arm = ai/out/final_v3_allfix/exports/teacher_v2_s{1,2,3}.pt; the harness refuses to run unless manifest, summary, shield and per-seed metrics still carry the all-fix evidence. No gate, weight, physics constant or threshold was changed.
- The helper binary was not rebuilt in this shared checkout; its seed-1 zero-command flight is checked bit-for-bit against the frozen post-FIXPUMP held-out sidecar.
- Exit 1 is the unweakened gate verdict, not a crash; the JSON and Markdown reports are written before exit. No win is claimed unless every gate passes.
