# Final exports — 100 epochs, seed 1

Date: 2026-11-22. CPU only (`torch 2.15.0+cpu`), no GPU, no push. Source files
outside `ai/` untouched; no gate, loss, architecture, threshold or data was
changed for this deliverable.

## What these artifacts are

The 3-seed sweep (`ai/out/sweep/`) selected **epoch 100** for all three families.
These are the **seed-1, 100-epoch runs from that sweep**, copied byte-for-byte
into `ai/out/final_100/` so the final numbers are exactly the sweep's seed-1
numbers, not a fresh training run with a different RNG stream.

Provenance (sha256, final copy == sweep source):

| model | source | export | sha256 |
|---|---|---|---|
| lrs | `out/sweep/runs/lrs/e100_s1` | `lrs/lrs.bin` | `2bb54c7cb97641a15200890be50db82dee531372ac414fcbee704b452b719a48` |
| loopwm-lite | `out/sweep/runs/loopwm-lite/e100_s1` | `lite/lite.pt` | `5a3cf7ec0d99b7e16fe1f42fdf6ff0c9eb491d1f205c82dbada8269044f90480` |
| loopwm | `out/sweep/runs/loopwm/e100_s1` | `loopwm/loopwm_student.bin` | `458a4b3976d035e82a0f57b854d24caf081eb608397a27fa6c6ddb0a6ac968c8` |

The provenance is independently confirmed by `config_hash`: rebuilding the
train-stage `TrainConfig` with the sweep's exact flags and `out/run4/data`
reproduces each `metrics.json` hash (`aa21e36ca6185ce1`, `993232786e2b4c23`,
`7482d23e9b21faaf`). `config_hash` includes `data` and `stage`, so an eval-stage
or `final_100/data` rebuild necessarily hashes differently even on identical
bytes.

Per-model directories contain the original `metrics.json`, `calibration.json`,
`run.log` and exports. `data/` is a byte-identical copy of `out/run4/data`
(80 train / 20 held-out episodes × 1500 frames), `sim.toml` is the run-4 config,
and `logs/` keeps `results.csv` / `SUMMARY.md` from the sweep.

## Why 100 epochs (sweep rationale)

Rule fixed before reading results: an epoch is admissible only when **every**
seed has `ratio <= 1.25` and `ece <= 0.05`; among admissible epochs pick the
lowest **mean** held-out RMSE across seeds; break ties by highest mean coverage.
All 27 (model × epoch × seed = 3×3×3) runs are admissible, so selection is by
mean held-out RMSE. Epoch 80 was dropped up front (a single 40-epoch `lrs` run
costs ~17 min on one core, so 36 runs were infeasible). No seed was cherry-picked.

| model | selected epoch | mean held-out RMSE ± sample std (ddof=1) | mean ratio ± std | mean coverage ± std |
|---|---:|---|---|---|
| lrs | 100 | 0.179815 ± 0.002597 | 0.776100 ± 0.027669 | 0.476752 ± 0.160272 |
| loopwm-lite | 100 | 0.181085 ± 0.009249 | 0.970636 ± 0.067228 | 0.390133 ± 0.163494 |
| loopwm | 100 | 0.188893 ± 0.010472 | 0.886778 ± 0.045713 | 0.422525 ± 0.200506 |

Against the 40-epoch baseline, 100 epochs improves held-out RMSE by 31.0 %
(lrs), 16.6 % (loopwm-lite) and 0.9 % (loopwm); the loopwm difference is inside
the seed spread, so treat 40 vs 100 as a tie for that family. Full table and
per-run rows: `logs/SUMMARY.md` and `logs/sweep_results.csv`.

## Exact seed-1 numbers (verbatim from each `metrics.json` / `calibration.json`)

### lrs — `lrs/lrs.bin`

| field | value |
|---|---|
| rmse_train | 0.23977497354527916 |
| rmse_held_out | 0.18175296497222185 |
| ratio | 0.7580147430936931 |
| params / mac / bytes | 22439 / 58816 / 78171 |
| mae / max_error | 0.1377689138737775 / 2.018527030944824 |
| ece (calibrated) | 1.3485651832541958e-19 |
| seed / config_hash | 1 / `aa21e36ca6185ce1` |
| calibration | raw_ece 0.3103861131295122, temperature_ece 0.20574074032561196, coverage 0.2917298414886285, temperature 0.1, method isotonic, loops 3, n 29020, target_ece 0.05 |

### loopwm-lite — `lite/lite.pt`

| field | value |
|---|---|
| rmse_train | 0.20965406781013954 |
| rmse_held_out | 0.19033253500748856 |
| ratio | 0.9078408875894154 |
| params / mac / bytes | 34819 / 289984 / 150095 |
| mae / max_error | 0.11587316438499234 / 2.1006813049316406 |
| ece (calibrated) | 1.3046889933380356e-17 |
| seed / config_hash | 1 / `993232786e2b4c23` |
| calibration | raw_ece 0.21477641878789935, temperature_ece 0.0587974683405099, coverage 0.31013094417643006, temperature 0.2734375, method isotonic, loops 3, n 5804, target_ece 0.05 |

### loopwm — `loopwm/loopwm.pt`, `loopwm/loopwm_student.bin`, `loopwm/loopwm_student.pt`

| field | value |
|---|---|
| rmse_train | 0.19225856890709406 |
| rmse_held_out | 0.17891006795923656 |
| ratio | 0.9305700597703505 |
| params / mac / bytes | 83075 / 6231744 / 78171 |
| mae / max_error | 0.10132286183120134 / 1.9500938653945923 |
| ece (calibrated) | 5.5415508310459656e-17 |
| seed / config_hash | 1 / `7482d23e9b21faaf` |
| calibration | raw_ece 0.30570284977463075, temperature_ece 0.09559071201334206, coverage 0.6481736733287388, temperature 2.0, method isotonic, loops 3, n 5804, target_ece 0.05 |

`bytes` is the onboard student binary for all three (78171 for lrs and loopwm,
150095 for the lite checkpoint); `mac` for loopwm is the privileged teacher's
analytic count (SPEC_ENGINE.md convention).

## Exact training flags (sweep driver, `out/sweep/driver.py`)

Shared, all three models:

```
--stage train --data out/run4/data --epochs 100 --seed 1 --device cpu
--window 50 --batch 8 --width 64 --rank 32 --heads 4 --max-iters 2
--lite-width 64 --attn-rank 8 --lite-iters 2 --latent-tokens 16
--residual --curriculum --calibrate
```

Per model:

```
lrs          --limit 2 --eval-limit 1451 --lr 0.03
loopwm-lite  --limit 1451 --eval-limit 1451
loopwm       --limit 1451 --eval-limit 1451
```

Each run used a private `--out` directory, so no teacher checkpoint leaked
between runs (in particular loopwm-lite's teacher is randomly initialised in
its private dir, exactly as in the sweep — its RMSE is not comparable to
loopwm's).

## Verification performed here

- Reload: each final export was copied to `/tmp/opencode/final100_verify/` and
  scored through `potok_ai.engine --stage eval` with the flags above (data from
  `final_100/data`). All three loaded, produced finite RMSE/ratio, and matched
  the stored `params` and `mac`. Reloaded `rmse_train` / `rmse_held_out` differ
  from the train-stage values by at most ~1e-9 (lrs 0.239774974 / 0.181752965,
  lite 0.209654068 / 0.190332535, loopwm 0.192258569 / 0.178910068).
- `uv run pytest -q` → **88 passed**; `uv run ruff check src tests` →
  **All checks passed!**
- Full verifier log: `/tmp/opencode/verify_final100.log` (outside the repo).

Note: the eval stage reports `export_path()` bytes (`loopwm.pt`, 341477) for
loopwm, while the train-stage metrics report the onboard student binary
(78171); this is existing engine behaviour, not a reload defect.

## Caveats (inherited from the sweep, not fixed here)

- Selection on held-out RMSE spends the held-out set; the 100-epoch numbers are
  optimistic as unbiased generalisation estimates.
- Calibrated ECE is in-sample on the held-out set (n reported); `ece ≈ 0` for
  all three winners because isotonic maps that same set to itself. Raw ECE is
  still large (0.21–0.31), so passing the ECE gate is weak evidence of
  calibrated confidence, not a standalone quality claim.
- Epoch count also changes the curriculum (`epochs_per_stage = epochs // 4` =
  25 at 100 vs 10 at 40), so the sweep measures epoch count plus schedule.
- A ratio < 1 means held-out RMSE is lower than the mean training-scenario
  RMSE; the four held-out scenarios may simply be easier. No re-split control
  exists in this sweep.
- No A/B, no GPU, no push was performed for this deliverable.
