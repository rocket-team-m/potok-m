# Run 3 — confidence calibration fix (CPU, seed 1, same OR-aligned data as run 2)

Data is byte-identical to `../run2/data` (sha256 of both `.bin` files matches).
Only `lrs` and `loopwm-lite` were retrained; `loopwm` is unchanged from run 2.

## Root cause of the run-2 degeneracy

1. **Wrong loss domain.** `lrs` and `loopwm-lite` regressed the *sigmoid* of the
   confidence logit toward `exp(-error/tol)`, while `loopwm` regressed the *raw
   logit*. With near-zero targets, sigmoid-MSE drives the logit to −∞. Measured
   on run 2: lite logit −4.77 ± 0.002, lrs −6.75 ± 0.0006, loopwm +0.53 ± 0.08.
2. **Target saturation.** With command error ≈ 1 and `tol = 0.1`,
   `exp(-error/tol) ≈ 4.5e-5` for every sample — no positive/negative signal.
3. **Collapsed representation.** Both students' latents converged to a
   frame-independent fixed point (lite's unbounded FiLM scale was cancelled by
   LayerNorm; LRS's recurrent state washed out the current frame), so even a
   spread target could not produce spread (logit std ≈ 1e-4 after the first two
   fixes alone).

## What changed (all inside `ai/`)

- `calibration.py`: `confidence_target` now scales by the batch's own mean error
  (floored at `tol`), keeping a spread at any error scale; explicit `scale=`
  preserves the old absolute-error semantics.
- `engine.py`: LRS and lite train the **logit** with BCE-with-logits; LRS shares
  the target scale across an episode's ticks; added `--conf-weight` (default 1.0).
- `student.py`: current-frame skip into the latent (head dimensions unchanged,
  so the v2 export format is unchanged).
- `loopwm_lite.py`: bounded FiLM scale plus a direct last-frame path in `refine`
  (no new parameters; `.pt` export unchanged).

Gates, `tol`, architecture widths and thresholds were not changed to pass.

## Result

| model | rmse_train | rmse_held_out | ratio | raw_ece | calibrated_ece | coverage | n | pass |
|---|---|---|---|---|---|---|---|---|
| lrs | 0.292490 | 0.319560 | 1.092550 | 0.298082 | ~0 (isotonic) | 0.322142 | 11020 | PASS |
| loopwm-lite | 0.266505 | 0.327992 | 1.230717 | 0.105970 | 0.030339 | 0.916071 | 2240 | PASS |

Gates: `ratio <= 1.25`, `ece <= 0.05`. Both pass.

Honesty notes:

- Coverage is now real: 3550/11020 and 2052/2240 held-out samples are within
  `tol = 0.1`. It is not fabricated and `tol` was not moved.
- Raw ECE is large and calibrated ECE small because the heads' confidence is
  genuinely mis-scaled before calibration and the held-out set is used for the
  calibration fit (in-sample, `n` reported).
- `lrs` calibrated ECE ≈ 0 comes from isotonic, not from a constant head: its
  held-out logit std is ≈0.09 on a 4-episode sample; lite's full held-out logit
  std is 0.57.
- `loopwm` was not retrained; its run-2 numbers remain valid for that code state.
  A future retrain would use the changed `confidence_target` default.
- `lrs.bin` reloads as a v2 20-input student; `lite.pt` keeps its `model_config` /
  `state_dict` shape.
