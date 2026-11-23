# Epoch × seed sweep on the full run-4 dataset

Data: `ai/out/run4/data` (80 train / 20 held-out episodes, 1500 frames each). Pipeline: unchanged `potok_ai.engine`, residual + curriculum on, `--calibrate` on. Epochs swept: **40, 60, 100** — epoch 80 was dropped because a single 40-epoch `lrs` run already costs ~17 min of single-core CPU and 36 runs were infeasible; all three seeds were kept. Seeds: 1, 2, 3 (deterministic via `engine.seed_everything`). Every run used a private output directory; gates, loss, architecture and data were not changed.

Metrics are read from each run's `metrics.json` and `calibration.json`. `ece` is the calibrated ECE; `raw_ece` is before calibration; `coverage` is the empirical correct fraction at tol=0.1 over the held-out predictions (n=29020 for lrs, n=5804 for the window families).

## Mean ± std across seeds (sample std, ddof=1)

| model | epoch | rmse_train | rmse_held_out | ratio | ece | coverage | seeds passing gates |
|---|---:|---|---|---|---|---|---|
| loopwm | 40 | 0.215173 ± 0.042272 | 0.190597 ± 0.016707 | 0.899548 ± 0.105148 | 0.002691 ± 0.004660 | 0.316334 ± 0.239024 | 3/3 |
| loopwm | 60 | 0.288084 ± 0.075672 | 0.257438 ± 0.113667 | 0.868745 ± 0.148188 | 0.021449 ± 0.025354 | 0.469446 ± 0.038854 | 3/3 |
| loopwm | 100 | 0.213800 ± 0.023004 | 0.188893 ± 0.010472 | 0.886778 ± 0.045713 | 0.000000 ± 0.000000 | 0.422525 ± 0.200506 | 3/3 |
| loopwm-lite | 40 | 0.264858 ± 0.038014 | 0.217044 ± 0.013488 | 0.827835 ± 0.100178 | 0.012152 ± 0.021048 | 0.194923 ± 0.105766 | 3/3 |
| loopwm-lite | 60 | 0.387730 ± 0.207990 | 0.367602 ± 0.236186 | 0.909937 ± 0.137453 | 0.011413 ± 0.019768 | 0.339019 ± 0.144044 | 3/3 |
| loopwm-lite | 100 | 0.187592 ± 0.022343 | 0.181085 ± 0.009249 | 0.970636 ± 0.067228 | 0.000000 ± 0.000000 | 0.390133 ± 0.163494 | 3/3 |
| lrs | 40 | 0.349103 ± 0.188628 | 0.260717 ± 0.126583 | 0.760074 ± 0.054431 | 0.000000 ± 0.000000 | 0.410648 ± 0.136179 | 3/3 |
| lrs | 60 | 0.236883 ± 0.016132 | 0.188338 ± 0.017665 | 0.794274 ± 0.024530 | 0.000000 ± 0.000000 | 0.454250 ± 0.093549 | 3/3 |
| lrs | 100 | 0.231863 ± 0.007983 | 0.179815 ± 0.002597 | 0.776100 ± 0.027669 | 0.000000 ± 0.000000 | 0.476752 ± 0.160272 | 3/3 |

## Every run (no cherry-picking)

| model | epoch | seed | rmse_train | rmse_held_out | ratio | ece | coverage | n | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| loopwm | 40 | 1 | 0.253184 | 0.204530 | 0.807829 | 0.000000 | 0.061682 | 5804 | yes |
| loopwm | 40 | 2 | 0.169648 | 0.172075 | 1.014305 | 0.008072 | 0.535837 | 5804 | yes |
| loopwm | 40 | 3 | 0.222686 | 0.195186 | 0.876508 | 0.000000 | 0.351482 | 5804 | yes |
| loopwm | 60 | 1 | 0.248792 | 0.186962 | 0.751481 | 0.049430 | 0.469848 | 5804 | yes |
| loopwm | 60 | 2 | 0.240140 | 0.196785 | 0.819459 | 0.014917 | 0.508098 | 5804 | yes |
| loopwm | 60 | 3 | 0.375320 | 0.388567 | 1.035296 | 0.000000 | 0.430393 | 5804 | yes |
| loopwm | 100 | 1 | 0.192259 | 0.178910 | 0.930570 | 0.000000 | 0.648174 | 5804 | yes |
| loopwm | 100 | 2 | 0.238031 | 0.199793 | 0.839359 | 0.000000 | 0.354583 | 5804 | yes |
| loopwm | 100 | 3 | 0.211111 | 0.187975 | 0.890406 | 0.000000 | 0.264817 | 5804 | yes |
| loopwm-lite | 40 | 1 | 0.243274 | 0.201472 | 0.828170 | 0.036457 | 0.073225 | 5804 | yes |
| loopwm-lite | 40 | 2 | 0.308751 | 0.224613 | 0.727490 | 0.000000 | 0.246899 | 5804 | yes |
| loopwm-lite | 40 | 3 | 0.242549 | 0.225048 | 0.927845 | 0.000000 | 0.264645 | 5804 | yes |
| loopwm-lite | 60 | 1 | 0.229464 | 0.172896 | 0.753481 | 0.034239 | 0.498105 | 5804 | yes |
| loopwm-lite | 60 | 2 | 0.310422 | 0.299572 | 0.965048 | 0.000000 | 0.301516 | 5804 | yes |
| loopwm-lite | 60 | 3 | 0.623304 | 0.630336 | 1.011282 | 0.000000 | 0.217436 | 5804 | yes |
| loopwm-lite | 100 | 1 | 0.209654 | 0.190333 | 0.907841 | 0.000000 | 0.310131 | 5804 | yes |
| loopwm-lite | 100 | 2 | 0.188143 | 0.181089 | 0.962507 | 0.000000 | 0.578222 | 5804 | yes |
| loopwm-lite | 100 | 3 | 0.164978 | 0.171834 | 1.041558 | 0.000000 | 0.282047 | 5804 | yes |
| lrs | 40 | 1 | 0.235343 | 0.174805 | 0.742770 | 0.000000 | 0.566402 | 29020 | yes |
| lrs | 40 | 2 | 0.566838 | 0.406082 | 0.716399 | 0.000000 | 0.351482 | 29020 | yes |
| lrs | 40 | 3 | 0.245127 | 0.201263 | 0.821053 | 0.000000 | 0.314059 | 29020 | yes |
| lrs | 60 | 1 | 0.233038 | 0.179594 | 0.770665 | 0.000000 | 0.561578 | 29020 | yes |
| lrs | 60 | 2 | 0.254591 | 0.208671 | 0.819631 | 0.000000 | 0.390007 | 29020 | yes |
| lrs | 60 | 3 | 0.223021 | 0.176750 | 0.792527 | 0.000000 | 0.411165 | 29020 | yes |
| lrs | 100 | 1 | 0.239775 | 0.181753 | 0.758015 | 0.000000 | 0.291730 | 29020 | yes |
| lrs | 100 | 2 | 0.232004 | 0.176865 | 0.762334 | 0.000000 | 0.572777 | 29020 | yes |
| lrs | 100 | 3 | 0.223811 | 0.180829 | 0.807953 | 0.000000 | 0.565748 | 29020 | yes |

## Selected epoch per model

- **loopwm**: epoch **100** — held-out RMSE 0.188893 ± 0.010472, ratio 0.886778 ± 0.045713, ece 0.000000 ± 0.000000, coverage 0.422525 ± 0.200506.
- **loopwm-lite**: epoch **100** — held-out RMSE 0.181085 ± 0.009249, ratio 0.970636 ± 0.067228, ece 0.000000 ± 0.000000, coverage 0.390133 ± 0.163494.
- **lrs**: epoch **100** — held-out RMSE 0.179815 ± 0.002597, ratio 0.776100 ± 0.027669, ece 0.000000 ± 0.000000, coverage 0.476752 ± 0.160272.

## Does more than 40 epochs improve or degrade held-out performance?

- **loopwm** held-out RMSE vs epoch (40: 0.190597±0.016707, 60: 0.257438±0.113667, 100: 0.188893±0.010472): 40 → 100 **improves** by -0.001704 (-0.9%); lowest mean at epoch 100.
- **loopwm-lite** held-out RMSE vs epoch (40: 0.217044±0.013488, 60: 0.367602±0.236186, 100: 0.181085±0.009249): 40 → 100 **improves** by -0.035959 (-16.6%); lowest mean at epoch 100.
- **lrs** held-out RMSE vs epoch (40: 0.260717±0.126583, 60: 0.188338±0.017665, 100: 0.179815±0.002597): 40 → 100 **improves** by -0.080901 (-31.0%); lowest mean at epoch 100.

Ratio < 1 means held-out RMSE is *lower* than the mean training-scenario RMSE. That is not evidence of generalisation by itself: the four held-out scenarios may simply be easier than the sixteen training scenarios. The conditioning vectors (mass, area, cd0, static margin, thrust, gyro noise) have essentially identical means and ranges in both splits, so an easy-split explanation is plausible but not established — there is no re-split control in this sweep. The epoch trend is measured on the same fixed split, so it is comparable across epochs; the absolute ratio is not a claim about split difficulty being matched.

## Caveats (read before using the selected epoch)

- **Selection on held-out RMSE spends the held-out set.** The rule was fixed in advance, but the 100-epoch numbers are the winners on the same 20 episodes used for the gates, so they are optimistic as unbiased generalisation estimates.
- **ECE is in-sample and degenerate for most winners.** Calibration fits temperature/isotonic on the held-out logits themselves. Several runs report `ece = 0.000000` because isotonic maps the held-out set to itself; raw ECE is still large (e.g. loopwm seed 1: 0.582649). Passing `ece <= 0.05` is therefore weak evidence of calibrated confidence, not a standalone quality claim.
- **Epoch count also changes the curriculum.** `epochs_per_stage = epochs // 4` (10 / 15 / 25 for 40 / 60 / 100), so the curve is epoch count *plus* curriculum schedule, not training duration alone. The non-monotonic 60-epoch spike (loopwm-lite seed 3: 0.630336; loopwm seed 3: 0.388567) is seed-driven instability, not a smooth overfitting curve.
- **Seed variance dominates the 40-epoch baseline.** lrs seed 2 at 40 epochs scores 0.406082 held-out RMSE versus seed 1's 0.174805; the single-seed run 4 number was not a stable estimate. Three seeds are the minimum for reading these differences, and the 100-epoch advantage of loopwm (-0.001704, -0.9%) is inside the seed spread — treat it as a tie, not a win.
- **No run failed** (27/27 successful); all 27 passed both gates on every seed, so no selection is based on a dropped outlier.

## Reproduce

```bash
cd /home/sehaxe/potok-m/ai
.venv/bin/python out/sweep/driver.py --workers 10 --epochs 40 60 100 --seeds 1 2 3
.venv/bin/python out/sweep/summarize.py
```
