# Task B06 — AI LoopWM scale integration

## Scope
- Owned paths only: `ai/src/potok_ai/loopwm_scale.py`, `ai/tests/test_loopwm_scale.py`.
- `loopwm.py` was not modified (no diff); its `LATENT_SCALE_FLOOR` is imported, not duplicated.
- Own code is GPL-3.0-or-later per the repo `LICENSE`; no third-party source copied.
- No subagents, no GPU work, no outward git operations.

## Implementation
- `normalized_latent_loss(prediction, target, *, scale=None)` — MSE measured in the
  detached target std's units, floored at `LATENT_SCALE_FLOOR`. Optional `scale`
  lets multiple prediction heads share one normalization, matching
  `LoopWM.latent_prediction_loss` (which normalizes every horizon by the std of
  all horizons' targets).
- `scaled_exit_threshold(threshold, latent)` — converts a reference-scale exit
  threshold into raw increment units by multiplying by the latent batch std.
  `LoopWM.refine`'s decision `mean ‖increment‖ < threshold` is thereby invariant
  when latents and increments are multiplied by a common factor.
- `LATENT_SCALE_FLOOR` re-exported for callers.

## Verification
- `pytest tests/test_loopwm_scale.py -q` → 9 passed.
- `pytest tests/test_loopwm_scale.py tests/test_loopwm.py -q` → 21 passed.
- Full AI suite `pytest -q` → 166 passed.
- `ruff check` and `ruff format --check` on both new files → clean.
- `git diff --check` on the staged task paths → clean. Repository-wide check
  reports trailing whitespace only in the pre-existing uncommitted
  `ai/out/v3_final/results.csv` and `ai/out/v3_final/runs/lrs/s2/run.log`; those
  paths are outside this task and were left untouched.

## Notes
- Scale invariance is exact while the target std stays above `LATENT_SCALE_FLOOR`
  (1e-3); below it the floor dominates by design, and a dedicated test pins that
  loss and gradients stay finite.
- One test compares the helper against the live `LoopWM.latent_prediction_loss`
  on a fixed-target model, so the exposed helper cannot silently diverge from
  the in-model normalization.
