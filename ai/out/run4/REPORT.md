# Run 4 — full 30 s OR-aligned flights (CPU, seed 1)

Config: `/tmp/potok-run4-sim.toml`, a copy of the current (OR-derived)
`sim/config/default.toml`; its only difference from `run3/sim.toml` is
`duration_s = 12.0 → 30.0`. The current default already had 30.0, so the temp
config is byte-identical to `sim/config/default.toml`. No `sim/` file was edited.

Data: 100 generated episodes per split with `--physics-features --seed 1`;
the `episode_index % 20` scenario split keeps **80 train** episodes (16 scenarios,
00–15) and **20 held-out** episodes (4 scenarios, 16–19). 1500 frames × 14
features per episode. Files: `data/train_validation.bin`, `data/heldout_validation.bin`.

## AI-active window (the point of this rerun)

`ai_healthy` (validation target column 4) is written from
`frame.ai_input_valid() && finite state && phase.ai_allowed()`, i.e. it requires
AI_READY with no FAULT. Counts of ticks per episode with that flag set:

| split | ticks/episode | AI-active ticks/episode (min / mean / median / max) | first active tick | last active |
|---|---|---|---|---|
| train | 1500 | 892 / 948.0 / 950 / 991 | 509–608 | 1499 |
| held-out | 1500 | 904 / 920.2 / 921 / 932 | 568–596 | 1499 |

The AI-active window is ~18–20 s of real flight (≈61–66% of each episode),
starting around t = 10–12 s. The 12 s run3 data only contained the first few
AI-active ticks; this dataset does not.

## Result (40 epochs, the documented minimum — no gate was changed)

Residual loss + curriculum ON (defaults), `--calibrate` ON, logit-domain confidence
loss from run3. Window families use `--limit/--eval-limit 1451` (a full episode
of windows); LRS uses `--limit 2` and scores all 1500 ticks of every held-out episode.

| model | rmse_train | rmse_held_out | ratio | raw_ece | calibrated_ece | coverage | n | pass |
|---|---|---|---|---|---|---|---|---|
| lrs | 0.235343 | 0.174805 | 0.742770 | 0.271013 | ~0 (isotonic) | 0.566402 (16437/29020) | 29020 | PASS |
| loopwm-lite | 0.243274 | 0.201472 | 0.828170 | 0.367779 | 0.036457 | 0.073225 (425/5804) | 5804 | PASS |
| loopwm | 0.253184 | 0.204530 | 0.807829 | 0.582649 | ~0 (isotonic) | 0.061682 (358/5804) | 5804 | PASS |

Gates: `ratio <= 1.25`, `ece <= 0.05`. All three pass at 40 epochs, so no
increase was needed. ECE is measured in-sample on the held-out windows; `n` is
reported. Raw ECE is large because confidence is mis-scaled before calibration,
not because the head is constant (see run3 root-cause work).

`lrs.bin` and `loopwm_student.bin` are v2 20-input student exports and reload
with `load_student_export`. `lite.pt` is the lite torch export.
