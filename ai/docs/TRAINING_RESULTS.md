# Post-FIXPUMP teacher-v2 training results (POSTPIVOT default)

Staged teacher v2 now runs the four-stage default (`sft -> privileged -> dagger -> distill`); RLVR is opt-in checkpoint selection and was **not** enabled here, so no reward gradient and no verifier checkpoint selection ran. **100 epochs**, CPU only, seeds [1, 2, 3]. Physics is the same post-FIXPUMP simulator and the train split is the same full 80-episode v3 file as the earlier five-stage pass, partitioned by scenario with a disjoint 2-scenario inner hold-out per seed. Held-out scenarios 16-19 are never passed to a training batch. No physics constant, gate, weight or threshold was changed. `--limit 1451` caps WINDOWS per batch builder (one episode-equivalent), not episodes: DAgger uses all 80 episodes and every scenario is scored. This is not the old `--limit 2` two-episode recipe.

Held-out RMSE over three seeds: **7.902916 +/- 0.498808 m/s^2** (sample standard deviation).

| seed | train RMSE | held-out RMSE | ratio | disjoint ECE | conformal coverage |
|---:|---:|---:|---:|---:|---:|
| 1 | 7.137926 | 7.572275 | 1.0609 | 0.0259 | 0.9545 |
| 2 | 7.352789 | 7.659804 | 1.0418 | 0.0108 | 0.9614 |
| 3 | 7.911737 | 8.476669 | 1.0714 | 0.0028 | 0.9394 |

## Comparison to the earlier five-stage pass (no win claimed)

| run | stages | held-out RMSE, mean +/- std (m/s^2) |
|---|---|---:|
| this run | sft -> privileged -> dagger -> distill | 7.902916 +/- 0.498808 |
| earlier consistency pass | sft -> privileged -> dagger -> rlvr -> distill | 8.039883 +/- 0.186312 |

Difference of means: -0.136967 m/s^2. The two runs share data, seeds, epochs and every recorded knob except the trainer's stage order (RLVR was a fifth gradient stage then, an opt-in selection stage now). This is a description of two measurements, not an A/B verdict: no closed-loop harness, no flight-truth sidecar, no positive control, and seed spread alone does not establish an improvement. No win is claimed.

A second trainer change landed while this run was prepared: SHIELD02 added a bounded, zero-initialised residual policy that is enabled by default in teacher stages. It is enabled here (see `residual_policy` in each `metrics.json`) and absent from the earlier pass, so the difference of means confounds stage order with residual policy and cannot be attributed to the DAgger ordering. `--no-residual-policy` keeps the pre-SHIELD02 teacher; that isolating run was not performed.

## Gate verdict (teacher gates only)

| check | verdict | detail |
|---|---|---|
| rmse_ratio | PASS | three seeds, held-out/train ratio <= 1.25 |
| ece_source | PASS | three seeds labelled with a disjoint source ['disjoint', 'held-out', 'held_out'] |
| ece | PASS | three disjoint ECE scores <= 0.05 with coverage in [0, 1] |

Closed-loop A/B gates (`seed_spread`, `closed_loop_improvement`, `oracle_positive_control`) are **not evaluated here**: no A/B report was produced and no win is claimed. `ece_source` is `held-out` because the trainer's confidence rows come only from the inner scenario split, which no gradient step sees; `ece_held_out` and `conformal_coverage` come from a second disjoint fit/score split of those rows.

## Honest reading

The three teacher gates pass, but a passing ratio/ECE pair is a consistency check, not a quality result: held-out RMSE (7.903 m/s^2) is close to the v3 target's own RMS (~8 m/s^2), so this student has not learned the command to useful accuracy. The low disjoint ECE reflects a well-behaved confidence head, not a small command error. Seed spread grew to 0.498808 m/s^2 from 0.186312 in the earlier pass, so the -0.136967 m/s^2 mean difference is smaller than one run's scatter and isolates no effect. No claim about closed-loop flight benefit is made here.

## Provenance

- train `validation.bin` sha256: `34b35e0ff67f7eeecbd895b2efde8bebcfa159c1a3e78379da6750338b6930b6`
- held-out `validation.bin` sha256: `fdc1d1400b1c5f111e21f79dead37afa1dd9f058fe457edb78037dc6ebe89807` (matches the post-FIXPUMP zero-command baseline)
- simulation config sha256: `9c34e71f9d10bcf081c31f15f5332f7bb2fac04dddb28009a27cd30987ab5e91`
- all knobs per seed are in `ai/out/final_v3_postpump_dagger/runs/s*/metrics.json`; exports and their digests are in `ai/out/final_v3_postpump_dagger/exports/manifest.json`.
- the trainer default includes SHIELD02 `residual_policy` (enabled); its gain and gate counts are in each `metrics.json`.
- the earlier five-stage artifacts remain untouched in `ai/out/final_v3_postpump/`.
