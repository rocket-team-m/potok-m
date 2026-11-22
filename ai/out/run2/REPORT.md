# Run 2 — OR-aligned simulator → AI engine (CPU, seed 1, authoritative)

This run supersedes `../run1/`: the sim owner changed `sim/config/default.toml`
to the OpenRocket-derived config at 20:36, after run 1's data was generated.
Run 2 uses the current OR-derived default (copied verbatim, only `duration_s`
patched 30 → 12). All OR physics fields (`cd0`, `cd0_range`, `static_margin_cal_range`,
`cd_mach`, the 200-row thrust table, wind/turbulence, initial tolerances) are
untouched. `sim.toml` sha256 prefix: `e57ead2c63edfe55`.

Data: `data/train_validation.bin` (80 train episodes, 16 scenarios) and
`data/heldout_validation.bin` (20 held-out episodes, 4 scenarios: 16–19),
generated with `potok-cli --physics-features --seed 1 --episodes 100 --split …`.
Each episode has 600 frames × 14 features. Residual loss + curriculum are ON
(defaults); calibration is ON. 12 s is needed because earlier in the flight the
reference optimizer is zeroed by the boost-phase AI interlock.

## Consolidated result

| model | rmse_train | rmse_held_out | ratio | ece | params | mac | bytes | pass/fail | coverage | n |
|---|---|---|---|---|---|---|---|---|---|---|
| lrs | 0.7629318641176421 | 0.7549866115556559 | 0.989335 | 0.000032 | 22439 | 58816 | 78171 | PASS | 0.000091 | 11020 |
| loopwm-lite | 0.773423373081182 | 0.7573962244702368 | 0.979278 | 0.000374 | 34819 | 289984 | 150095 | PASS | 0.000446 | 2240 |
| loopwm | 0.291805042290809 | 0.358268699214643 | 1.227767 | 0.000000 | 83075 | 6231744 | 78171 | PASS | 0.876786 | 2240 |

Gates (unchanged): `ratio <= 1.25` and `ece <= 0.05`. All three pass.

Exact training settings: seed 1; `lrs` 40 epochs, `--limit 2`, `--lr 0.03`;
`loopwm-lite` 100 epochs, `--limit 560`; `loopwm` 40 epochs, `--limit 560`.
See `../../BUILD_LOG.md` for the full commands and the epoch-selection evidence.

Honesty notes:

- `loopwm` is the genuinely non-degenerate result (88% of held-out samples
  correct within tol = 0.1, calibrated ECE ≈ 0, ratio 1.228).
- `lrs` and `loopwm-lite` have **non-zero but marginal** coverage (1/11020 and
  1/2240 correct). Their ECE ≤ 0.05 is real but largely reflects a
  near-zero-confidence map on an almost-always-wrong model; do not read it as
  accuracy. The physics baseline saturates at ±1 on these features while the
  reference command is ±0.12, so absolute RMSE stays near 1 for these two.
- `bytes`/`mac` for `loopwm` are the exported 20-input student binary and the
  privileged teacher's analytic MAC, respectively (SPEC_ENGINE.md convention).
  `lrs.bin` and `loopwm_student.bin` reload as v2 (`model_kind = 2`) students.
- The smoke harness (`scripts/report.md`, rerun after the config change) also
  passes all gates, but `lrs`/`loopwm-lite` ECE = 0 there is degenerate and
  `loopwm` ECE = 0.0492 only just clears the gate. It is not quality evidence.
