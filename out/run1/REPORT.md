# Run 1 — OR-aligned simulator → AI engine (CPU, seed 1)

Data: `data/train_validation.bin` (80 train episodes, 16 scenarios) and
`data/heldout_validation.bin` (20 held-out episodes, 4 scenarios: 16–19),
generated from `sim.toml` (a copy of `sim/config/default.toml` with **only**
`duration_s = 12.0` instead of 30.0; all OR-derived physics fields unchanged)
via `potok-cli --physics-features --seed 1 --episodes 100 --split …`.
Each episode has 600 frames × 14 features. Residual loss + curriculum are ON
(defaults); calibration is ON. 12 s is used because within the first ~10 s the
reference optimizer is zeroed by the boost-phase AI interlock and no learnable
correction exists.

## Consolidated result (this run)

| model | rmse_train | rmse_held_out | ratio | ece | params | mac | bytes | pass/fail | coverage | n |
|---|---|---|---|---|---|---|---|---|---|---|
| lrs | 0.799218 | 0.767838 | 0.960736 | 0.000192 | 22439 | 58816 | 78171 | PASS | 0.000635 | 11020 |
| loopwm-lite | 0.760144 | 0.779516 | 1.025484 | 0.002864 | 34819 | 289984 | 150095 | PASS | 0.004911 | 2240 |
| loopwm | 0.367503 | 0.398410 | 1.084100 | 0.000000 | 83075 | 6231744 | 78171 | PASS | 0.764286 | 2240 |

Gates (unchanged): `ratio <= 1.25` and `ece <= 0.05`. All three pass.

Honesty notes:

- The LoopWM calibration is genuinely non-degenerate (`coverage = 0.764`,
  isotonic). LRS and loopwm-lite have **non-zero but marginal** coverage
  (7/11020 and 11/2240 correct within tol = 0.1). Their ECE ≤ 0.05 is real but
  reflects a mostly-always-wrong confidence map; do not read it as accuracy.
- The held-out RMSE is dominated by boost-phase windows where the reference
  command is 0 and the physics baseline saturates at ±1, so absolute RMSE stays
  near 1 for the two student families. The ratio gate is the comparison that is
  meaningful here.
- `mac` for `loopwm` is the privileged teacher's analytic count (6.23 M), not
  the exported 20-input student's; `bytes` is the exported v2 student binary.
- `lrs.bin` and `loopwm_student.bin` are v2 (`model_kind = 2`, 20-value frames)
  and reload with `load_student_export`.
