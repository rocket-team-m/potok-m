<!-- night-final:begin -->
# Teacher-v2 results — RETRAIN-NIGHT (current)

The shipped four-stage default (`sft -> privileged -> dagger -> distill`) was rerun on the frozen post-FIXPUMP split with the train-split v3 truth sidecar, **3 seeds x 100 epochs, CPU**, and no physics/gate/envelope/weight tuning. Two upgrades that landed after RETRAIN-ALLFIX are active through trainer defaults: the privileged **impact objective (AIOBJ)**, on because a truth sidecar with records is supplied and `--legacy-objective` is NOT passed, and the **split-conformal confidence (AICAL)** hook, whose disjoint ECE and coverage are read from the trainer's own `metrics.json` and `conformal.json`. AIFIX1/2/3, SHIELD01/02 and on-policy DAgger are inherited defaults and are re-checked per seed. RLVR stayed opt-in and was not run. `--limit 1451` caps WINDOWS per batch builder, not episodes.

Held-out RMSE: **7.358037 +/- 2.147744 m/s^2** (sample std).

| seed | train RMSE | held-out RMSE | ratio | disjoint ECE | conformal coverage |
|---:|---:|---:|---:|---:|---:|
| 1 | 6.111624 | 5.474856 | 0.8958 | 0.0231 | 0.9545 |
| 2 | 9.050627 | 9.697134 | 1.0714 | 0.0091 | 0.9559 |
| 3 | 7.132454 | 6.902121 | 0.9677 | 0.0214 | 0.9642 |

## Comparison to RETRAIN-ALLFIX — descriptive, no A/B win claim

| run | held-out RMSE, mean +/- std (m/s^2) |
|---|---:|
| RETRAIN-NIGHT (impact objective + conformal) | 7.358037 +/- 2.147744 |
| RETRAIN-ALLFIX (all root-cause fixes, earlier code) | 7.855551 +/- 2.604600 |

Difference of means: -0.497515 m/s^2; the allfix sample std is 2.604600, so the difference is within one allfix seed spread. The two runs differ in the privileged objective and the confidence measurement at once, no arm isolates either, and there is no closed-loop harness, no independent flight-truth anchor, no positive control and no unweakened gate evaluation. `ab_win_claimed: false` in `summary.json`: **no win is claimed**.

## Adversarial curriculum (AICURR)

**RUN** — 3 seeds, 100 epochs, `--adversarial`; teacher gates: rmse_ratio=PASS, ece_source=PASS, ece=PASS. Descriptive second arm only; no win claimed.

## Gate verdict (teacher gates only)

| check | verdict | detail |
|---|---|---|
| rmse_ratio | PASS | three seeds, held-out/train ratio <= 1.25 |
| ece_source | PASS | three seeds labelled with a disjoint source ['disjoint', 'held-out', 'held_out'] |
| ece | PASS | three disjoint ECE scores <= 0.05 with coverage in [0, 1] |

Closed-loop A/B gates (`seed_spread`, `closed_loop_improvement`, `oracle_positive_control`) are **not evaluated**: no A/B report, no independent truth sidecar and no positive control. The ECE is a disjoint inner-held-out score; stage-local ECE values remain in-sample diagnostics.

## Binding limitation — stated loudly

AIFIX3's `world_step` advances only onboard sensor features through the unchanged `physics_baseline`/`final_command` plus named noise — a surrogate, not a Rust simulator step on (state, command). The v3 truth sidecar is emitted by that same simulator, so generator and verifier share a source; it is not an independent flight-truth anchor. The impact objective uses measured sensor columns and true velocity as an additional training term, not a new closed-loop measurement. A held-out RMSE near the target's own RMS (~7.7 m/s²) is the expected shape of that binding.

## Provenance

- train sha256: `34b35e0ff67f7eeecbd895b2efde8bebcfa159c1a3e78379da6750338b6930b6`; held-out sha256: `fdc1d1400b1c5f111e21f79dead37afa1dd9f058fe457edb78037dc6ebe89807`
- sim config sha256: `9c34e71f9d10bcf081c31f15f5332f7bb2fac04dddb28009a27cd30987ab5e91`; train-split truth sidecar sha256: `7f25bb69124638ea7542be99f43330fda1c5248513f1971f6bfd125f5f759cb8`
- per-seed evidence: `ai/out/night_final/runs/s*/metrics.json` and `conformal.json`; shield: `ai/out/night_final/shield.json`; exports: `ai/out/night_final/exports/manifest.json`; full summary: `ai/out/night_final/summary.json`.
<!-- night-final:end -->


---

# Post-FIXPUMP teacher-v2 results (all AI root-cause fixes active)

The four-stage default (`sft -> privileged -> dagger -> distill`) was run with every landed fix active and checked in each `metrics.json`: the +/-10 m/s² student bound (AIFIX1), the real train-split POTOKTRJ v3 truth sidecar in the privileged stage (AIFIX2), the on-policy student-in-the-loop DAgger rollout (AIFIX3), and the default-on bounded residual policy (SHIELD02). The online shield (SHIELD01) is verified and counted in `ai/out/final_v3_allfix/shield.json`; it is **not** invoked by the staged teacher-v2 trainer, so no training intervention count is claimed. RLVR remains opt-in checkpoint selection and was not run. **100 epochs**, CPU only, seeds [1, 2, 3]. Data, seeds, epochs and every recorded numeric knob are identical to RETRAIN-DAGGER; no physics constant, gate, threshold, weight or recipe value was tuned. `--limit 1451` caps WINDOWS per batch builder, not episodes.

Held-out RMSE over three seeds: **7.855551 +/- 2.604600 m/s^2** (sample standard deviation).

| seed | train RMSE | held-out RMSE | ratio | disjoint ECE | conformal coverage |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.658077 | 5.148904 | 0.9100 | 0.0117 | 0.9353 |
| 2 | 9.595021 | 10.344432 | 1.0781 | 0.0079 | 0.9545 |
| 3 | 8.446756 | 8.073319 | 0.9558 | 0.0108 | 0.9642 |

## Comparison to RETRAIN-DAGGER (no win claimed)

| run | fixes | held-out RMSE, mean +/- std (m/s^2) |
|---|---|---:|
| this run | AIFIX1/2/3 + SHIELD01/02, DAgger default | 7.855551 +/- 2.604600 |
| RETRAIN-DAGGER | DAgger default, older bounded student, no sidecar, recorded-window DAgger | 7.902916 +/- 0.498808 |

Difference of means: -0.047364 m/s^2. The baseline's own sample std is 0.498808 m/s^2, so the delta is within one baseline seed spread. It does not separate from the baseline at the resolution this instrument can resolve. There is no closed-loop harness on these exports, no independent flight-truth anchor, no positive control and no unweakened gate evaluation. `ab_win_claimed: false` in `summary.json`. **No win is claimed until a seeded closed-loop A/B is run.**

## Surrogate dynamics are the only binding mechanism — stated loudly

The fixes make the wiring true; they do not add information about closed-loop flight. AIFIX3's `world_step` advances only the onboard sensor features through the unchanged `physics_baseline`/`final_command` plus a named noise stream — it is a surrogate, not the Rust simulator step on (state, command). The privileged sidecar is emitted by that same simulator, not a flight log, so the verifier and the generator share a source. The DAgger expert label is still the reference optimizer's recorded answer from sidecar slots 22..25, and the privileged stage still supervises an episode-pooled latent with MSE. With those two surrogate links the student-induced states are genuine bookkeeping facts but not flight dynamics. A held-out RMSE near the target's own RMS (~7.7 m/s²) is the expected shape of that binding, not evidence that the fixes failed or that a controller improved. Honest artifacts are produced; the closed-loop A/B and an independent truth anchor remain owed.

## Gate verdict (teacher gates only)

| check | verdict | detail |
|---|---|---|
| rmse_ratio | PASS | three seeds, held-out/train ratio <= 1.25 |
| ece_source | PASS | three seeds labelled with a disjoint source ['disjoint', 'held-out', 'held_out'] |
| ece | PASS | three disjoint ECE scores <= 0.05 with coverage in [0, 1] |

Closed-loop A/B gates (`seed_spread`, `closed_loop_improvement`, `oracle_positive_control`) are **not evaluated**: no A/B report, no flight-truth sidecar, no positive control. `ece_source` is `held-out` because confidence rows come only from the inner scenario split; `ece_held_out` and `conformal_coverage` are a second disjoint fit/score split of those rows.

## Honest reading

The three teacher gates are consistency checks, not a quality result. Held-out RMSE (7.856 m/s²) is close to the v3 target's own RMS, so the student has not learned the command to useful accuracy. The low disjoint ECE reflects a well-behaved confidence head, not a small command error. Seed spread is 2.604600 m/s², 5.2x the baseline's 0.498808; the on-policy surrogate rollout made three identical-config seeds far less reproducible, not more. The -0.047364 m/s² mean difference is not a separable effect. No claim about closed-loop flight benefit follows.

## Provenance

- train `validation.bin` sha256: `34b35e0ff67f7eeecbd895b2efde8bebcfa159c1a3e78379da6750338b6930b6`
- held-out `validation.bin` sha256: `fdc1d1400b1c5f111e21f79dead37afa1dd9f058fe457edb78037dc6ebe89807`
- simulation config sha256: `9c34e71f9d10bcf081c31f15f5332f7bb2fac04dddb28009a27cd30987ab5e91`
- train-split POTOKTRJ v3 sidecar sha256: `7f25bb69124638ea7542be99f43330fda1c5248513f1971f6bfd125f5f759cb8`
- all knobs and per-fix evidence are in `ai/out/final_v3_allfix/runs/s*/metrics.json`; shield verification is in `ai/out/final_v3_allfix/shield.json`; exports and digests are in `ai/out/final_v3_allfix/exports/manifest.json`.
- the earlier RETRAIN-DAGGER artifacts remain untouched in `ai/out/final_v3_postpump_dagger/`.
