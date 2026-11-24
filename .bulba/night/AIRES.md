# AIRES — deterministic residual-policy deployment ablation

**Date:** 2026-11-24 · **Repo:** `/home/sehaxe/potok-m` (HEAD at run time `07a5e43ebcd123065227665a5d6b71c46f0acd7d`).
**Status:** done. Local commit only, no push. No subagents. No trainer, physics, weight, gate or threshold file was edited.

## Scope

Exactly four files:

- `scripts/ablation_residual.py` (new, executable, sha256 `2fa2269374d458fb8e78a4bc50c55c6cf5981da4779781efa10b79a5130d91d6`).
- `scripts/ablation_residual.json` / `scripts/ablation_residual.md` (new; generated evidence and table report).
- this report.

Flight work files were written outside the repo, under `/tmp/opencode/ablation_residual/seed{1,2,3}/`.
The pre-existing dirty files in `ai/src/`, `sim/`, `ai/out/` and other night reports were left untouched.

## What was measured

The same shipped FINAL all-fix distill exports (`ai/out/final_v3_allfix/exports/teacher_v2_s{1,2,3}.pt`), the same three seeds (1/2/3), the same 20 held-out episodes (scenarios 16–19, `fdc1d140…`), the same frozen post-FIXPUMP config (`9c34e71f…`) and the same read-only v3 control seam as v5. Nothing was retrained and no physics constant, gate, weight or reward term was changed. Only the **deployment composition** of the export was toggled. Every arm is written through the identical `ai_healthy` authority gate and clipped at the seam's hard ±10 m/s² bound; boost-phase flight stays identical to the control.

| arm | command written to the seam |
|---|---|
| `residual_on` | `clip(physics_baseline + export_delta)` — the shipped v5 arm |
| `residual_off` | `clip(export_delta)` — classical superposition removed |
| `classical_only` | `clip(physics_baseline)` — learned correction removed; also the literal `ResidualPolicy(enabled=False)` fallback (`residual_policy.py:147-149`) |
| `zero_command` | no TSV at all (`Simulator::run_episode`, `ControlMode::None`); paired control the gates measure against |
| `oracle_gtheta` | `run_episode_with_oracle` / `fin_accel_command`; positive control |

Both readings of “residual OFF” therefore have exact numbers: the superposition-off arm is `residual_off`, the module's disabled fallback is `classical_only`. The latter is reported as a diagnostic interpretation, **not** as a shipped policy — the export was trained on the residual-decision target and was not retrained for either ablation arm.

## Exact model-quality numbers (allfix `metrics.json`, unchanged)

| seed | rmse_train | rmse_held_out | ratio | ece_held_out | conformal coverage | residual gain |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.658076677982967 | 5.148904169724766 | 0.9100096132950048 | 0.01166445300894342 | 0.9352617079889807 | −4.862932205200195 |
| 2 | 9.595020625413378 | 10.344431805348933 | 1.0781041760296652 | 0.007868364254322439 | 0.9545454545454546 | −6.228996276855469 |
| 3 | 8.446755869407314 | 8.07331851231966 | 0.9557892565073202 | 0.010767036888121233 | 0.9641873278236914 | −5.851548671722412 |

## Exact flight numbers (metres, pooled over 20 held-out episodes)

Deployment proxy = radial distance at the first `DESCENT` tick (`vz<0 ∧ z≥5`); apogee = max z over the 30 s flight; lateral = whole-trajectory RMS.

| seed | arm | deployment lateral (control → arm) | apogee (control → arm) | lateral RMS (control → arm) |
|---:|---|---:|---:|---:|
| 1 | `residual_on` | 1690.756426 → **1473.061037** | 1973.743109 → **2334.545404** | 1309.408733 → **1138.447709** |
| 1 | `residual_off` | 1690.756426 → 1564.528569 | 1973.743109 → 2184.161621 | 1309.408733 → 1190.708314 |
| 1 | `classical_only` | 1690.756426 → 1658.801783 | 1973.743109 → 1922.569977 | 1309.408733 → 1319.533069 |
| 2 | `residual_on` | 1688.418578 → **1558.442532** | 1972.858569 → **2251.142731** | 1306.396947 → **1179.441255** |
| 2 | `residual_off` | 1688.418578 → 1611.285207 | 1972.858569 → 2105.717175 | 1306.396947 → 1225.190702 |
| 2 | `classical_only` | 1688.418578 → 1656.372738 | 1972.858569 → 1920.682275 | 1306.396947 → 1317.047522 |
| 3 | `residual_on` | 1689.185836 → **1496.789353** | 1983.003363 → **2305.874841** | 1306.380164 → **1145.629080** |
| 3 | `residual_off` | 1689.185836 → 1555.356834 | 1983.003363 → 2185.899200 | 1306.380164 → 1183.239723 |
| 3 | `classical_only` | 1689.185836 → 1657.321562 | 1983.003363 → 1931.668890 | 1306.380164 → 1316.336506 |

Change against the paired zero command (whole-trajectory lateral RMS % / apogee %):

| seed | `residual_on` | `residual_off` | `classical_only` |
|---:|---:|---:|---:|
| 1 | −13.06 % / +18.28 % | −9.07 % / +10.66 % | +0.77 % / −2.59 % |
| 2 | −9.72 % / +14.11 % | −6.22 % / +6.73 % | +0.82 % / −2.64 % |
| 3 | −12.31 % / +16.28 % | −9.43 % / +10.23 % | +0.76 % / −2.59 % |

Reading (descriptive, **not** a win claim): the full composition is the strongest of the three; removing the classical baseline keeps a smaller lateral/apogee effect, while removing the learned correction leaves lateral slightly *worse* and apogee below 2000 m on every seed. That is a decomposition of this export on this simulator, not evidence of mechanism or of flight benefit.

## Gate verdict — FAIL, 5 PASS / 16 FAIL, exit 1

Unweakened limits reused verbatim: deployment lateral ≤ 1000 m and apogee ≥ 2000 m per seed and per arm, the imported `teacher_gates` EVAL01 suite (ratio 1.25, ECE 0.05, disjoint source, provenance, −35 %/+15 %, 3× seed spread, oracle positive control), and the deployment/apogee positive-control check. Exit 1 is the designed gate verdict, not a crash; both report files are written before exit.

- **PASS (5):** `eval01 rmse_ratio`, `eval01 ece_source`, `eval01 ece`, `eval01 provenance`, `eval01 seed_spread [residual_on]`.
- **FAIL (16):** `eval01 improvement [residual_on]` (−13.06/−9.72/−12.31 % lateral above −35 %; seed-2 apogee +14.11 % below +15 %); `eval01 seed_spread`/`improvement [residual_off]`; `eval01 seed_spread`/`improvement [classical_only]`; `eval01 oracle_positive_control` and `oracle_deployment_and_apogee_positive_control` (oracle apogee +11.692065 / −2.151044 / −7.601099 m — seed 2/3 do not rise); and all nine per-arm absolute gates (every arm exceeds 1000 m deployment lateral; `classical_only` also falls below 2000 m apogee on all three seeds).

**No win is claimed.** Every gate must pass; sixteen do not.

## Provenance (exact digests, all re-verified at run time)

| field | value |
|---|---|
| config | `ai/out/v3_final/sim.toml` — `9c34e71f9d10bcf081c31f15f5332f7bb2fac04dddb28009a27cd30987ab5e91` |
| held-out | `ai/out/final_v3_postpump/data/held-out/validation.bin` — `fdc1d1400b1c5f111e21f79dead37afa1dd9f058fe457edb78037dc6ebe89807` |
| held-out manifest | `a9ff19947618f17790f1257ef6f7e85ba537915f5a2aca782befefa874db675e` |
| export manifest | `4c5040f42185d516b9777d58ee8ec0407f291d0c1566b3d32b3781d8d9a2c0b7` |
| allfix summary / shield | `6dab65d91ab4cbf182a81f57ba6838e24093edd50a3886ab968b52a601853ab2` / `ab277e7825ce7b746d09eb57ee57b01eb3866b489411427b995803cd740b0af2` |
| exports s1/s2/s3 | `2514b4e46148f70945601ebacc1b3eb2ea207ea819cc0f65f2a543375a8775f4` / `b2a0a521c7e70070c1f704d776200d643e82ec3c58732738b0756e3f9a9bb3b2` / `6d4d89d2bd896808edbbe40c176354f51c2186681da7fb7d2a91066812f4a968` |
| metrics s1/s2/s3 | `330423e584057d50779f215bcabad85bb4af07866d1eda9eeb3b2317e1174a36` / `ec54defcd00bde30a46d6dcb2c7855274a54bbca1b592fd8b8b63c866d5a9bd4` / `2f7b963e0c5f0c689e7e250d708f9bae6786ee182055a87dc2c82dedca92e5eb` |
| metrics config hashes | `f5b5106b378aac4b` / `7b5ad1b86cfcb3f3` / `717d2a5ae810d807` |
| zero-command trajectories s1/s2/s3 | `9b1d680d55f635a3a2cf84864498cb533a84132bbe15ba791154a3cd58bb5953` / `f0443872b644520d9f413e11b608241b0d8ea9bea65d5605b5e026ce08597aa2` / `b1123a635768bdd5d71b1591326477be3fde9f80a22b4f24288f067234ecbc4d` |
| helper source / binary | `1f5564e30d66d82e081e5e435bcac3932a862638163cb0fac9a7261920559fed` / `c8fea0bbee7c8e8604a8ad8575aee5b0ba12073ae7570766941e935e12935e36` |
| dynamics source | `6159cf4bd24862a8ac89f97529e1133d49ef6bd8f6ba2289f2b4d70987063cc2` |
| v5 harness / v5 report | `98cc203a7452a5bc58a043fae64cfc7665da100ce3959a5b558827ddfb079e90` / `ae61b430b957c7d86d65de9ca83658dbfb7cb5448bc587e8dc2452ff6328ae9a` |
| harness / git HEAD | `2fa2269374d458fb8e78a4bc50c55c6cf5981da4779781efa10b79a5130d91d6` / `07a5e43ebcd123065227665a5d6b71c46f0acd7d` |
| Python / torch / numpy / sample_dt | `3.13.15` / `2.15.0+cpu` / `2.5.5` / `0.02 s` |

Full per-arm assisted-trajectory SHA-256 map is in `ablation_residual.json:provenance.assisted_trajectory_sha256`.

## Fidelity and verification

- `residual_on` **exactly reproduces** the shipped `ab_report_v5.json` `teacher_v2` assisted aggregates on all three seeds (deployment lateral, apogee, whole-trajectory RMS; 1e-6 relative tolerance). This is the check that the new harness did not change the shipped arm — it passed.
- Seed-1 zero-command sidecar is bit-identical to the frozen post-FIXPUMP held-out trajectory (`9b1d680d…`), the loud check that the helper is the corrected-physics binary.
- `sha256sum scripts/ablation_residual.py` equals `provenance.harness_sha256` after the final run.
- `ai/.venv/bin/ruff check scripts/ablation_residual.py` — All checks passed; `ruff format --check` — already formatted.
- `ai/.venv/bin/python scripts/ablation_residual.py` — exit 1 (gate verdict), both report files written. No test was skipped or deleted.

## Honest limits

- Simulator-only: the POTOKTRJ sidecar comes from the same simulator that flies the episodes, so it is not an independent flight-truth anchor. There is no canopy/deployment event; the proxy is a fin-flight tick, not landing dispersion.
- Commands are computed from the control flight's recorded frames and replayed by frame index; after a command moves the vehicle, later frames diverge from the control flight while the phase/gate schedule stays paired.
- Seeds 2/3 use seed-matched exports on regenerated held-out flights — input robustness, not training reproducibility. A missing export is a loud blocker, never a stand-in.
- `residual_off` and `classical_only` are deployment-time recompositions of a residual-trained export, not retrained `--no-residual-policy` teachers; they cannot isolate training-stage effects. A true ON/OFF training A/B remains unrun.
- The fixed oracle positive control still fails the apogee half on seeds 2/3 (post-FIXPUMP behaviour already recorded in v5), so this harness cannot certify an apogee-improving assist even before the primary arm's own misses.
- The helper binary was not rebuilt in this shared checkout; its frozen-baseline bit-identity check is the guard against another worker's in-flight simulator edits.
- No gate was weakened, no threshold searched, no win claimed. Local commit only; nothing pushed.
