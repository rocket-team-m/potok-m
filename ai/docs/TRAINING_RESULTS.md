# Post-FIXPUMP teacher-v2 training results (TRAIN-FINAL)

Staged teacher v2 (`sft -> privileged -> dagger -> rlvr -> distill`), **100 epochs**, CPU only, seeds [1, 2, 3]. Physics is the post-FIXPUMP simulator; the train split is the full 80-episode v3 file, partitioned by scenario with a disjoint 2-scenario inner hold-out per seed. Held-out scenarios 16-19 are never passed to a training batch. No physics constant, gate or threshold was changed. `--limit 1451` caps WINDOWS per batch builder (one episode-equivalent), not episodes: DAgger uses all 80 episodes and every scenario is scored. This is not the old `--limit 2` two-episode recipe.

Held-out RMSE over three seeds: **8.039883 +/- 0.186312 m/s^2** (sample standard deviation).

| seed | train RMSE | held-out RMSE | ratio | disjoint ECE | conformal coverage |
|---:|---:|---:|---:|---:|---:|
| 1 | 7.719484 | 8.246161 | 1.0682 | 0.0003 | 0.9311 |
| 2 | 7.372674 | 7.989650 | 1.0837 | 0.0028 | 0.9683 |
| 3 | 7.402216 | 7.883837 | 1.0651 | 0.0025 | 0.9353 |

## Gate verdict (teacher gates only)

| check | verdict | detail |
|---|---|---|
| rmse_ratio | PASS | three seeds, held-out/train ratio <= 1.25 |
| ece_source | PASS | three seeds labelled with a disjoint source ['disjoint', 'held-out', 'held_out'] |
| ece | PASS | three disjoint ECE scores <= 0.05 with coverage in [0, 1] |

Closed-loop A/B gates (`seed_spread`, `closed_loop_improvement`, `oracle_positive_control`) are **not evaluated here**: no A/B report was produced and no win is claimed. `ece_source` is `held-out` because the trainer's confidence rows come only from the inner scenario split, which no gradient step sees; `ece_held_out` and `conformal_coverage` come from a second disjoint fit/score split of those rows.

## Honest reading

The three teacher gates pass, but a passing ratio/ECE pair is a consistency check, not a quality result: held-out RMSE (8.040 m/s^2) is close to the v3 target's own RMS (~8 m/s^2), so this student has not learned the command to useful accuracy. The low disjoint ECE reflects a well-behaved confidence head, not a small command error. No claim about closed-loop flight benefit is made here.

## Provenance

- train `validation.bin` sha256: `34b35e0ff67f7eeecbd895b2efde8bebcfa159c1a3e78379da6750338b6930b6`
- held-out `validation.bin` sha256: `fdc1d1400b1c5f111e21f79dead37afa1dd9f058fe457edb78037dc6ebe89807` (matches the post-FIXPUMP zero-command baseline)
- simulation config sha256: `9c34e71f9d10bcf081c31f15f5332f7bb2fac04dddb28009a27cd30987ab5e91`
- all knobs per seed are in `ai/out/final_v3_postpump/runs/s*/metrics.json`.
