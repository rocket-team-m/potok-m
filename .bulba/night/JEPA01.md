# JEPA01 — improved JEPA objective for the privileged teacher

**Date:** 2026-11-23
**Repo:** `/home/sehaxe/potok-m/ai` (git root `/home/sehaxe/potok-m`)
**Status:** done. Local commit only, nothing pushed. No subagents.

## Scope

Exactly two new source/test files plus this report. No existing file was edited:

- `ai/src/potok_ai/jepa_v2.py` — new objective, SPDX `GPL-3.0-or-later`.
- `ai/tests/test_jepa_v2.py` — 8 new gates, same SPDX header.

Pre-existing modified/untracked files elsewhere in the tree (`sim/`, `ai/out/`,
other night tasks) were left untouched.

## What was implemented

- **`EMATeacherV2`** — deep copy of the encoder, eval mode, all parameters
  `requires_grad_(False)`, every forward under `no_grad`, `.detach()` on output.
  `update()` EMA-scales parameters and copies buffers in lockstep (a
  running-stats encoder would otherwise drift); `sync()` hard-copies both.
  Teacher/student parameter and buffer counts and shapes are checked loudly.
- **Augmentation** — one contiguous temporal span per sample (`mask_ratio`,
  default 0.25) plus per-sensor dropout: a channel is dark for the whole window
  (`sensor_dropout`, default 0.1). Both come from a local generator seeded from
  `seed+1`; predictor init uses `seed+0`. The global RNG is never touched by
  this module, and `reset_seed()` rewinds the augmentation stream.
- **One-shot target** — a single `nn.Linear(width, n_horizons*width)` probe maps
  each latent position to all horizon targets at once; no recursive rollout.
  Sources are positions `0..T-h-1`, targets are the teacher latents at `h`
  frames later, for `h ∈ {3, 6, 10}` by default.
- **Normalisation** — `symlog(x) = sign(x)·log1p(|x|)` followed by 5–95
  percentile location/scale over the detached teacher target, with the symlog
  range floored at 1.0 (DreamerV3's robust-target idea, arXiv:2301.04104).
  The scale is recomputed per batch, so it is a property of the batch, not a
  fixed constant.
- **Free bits** — each horizon's normalised MSE is floored with
  `clamp_min(free_bits)` before averaging. Default **0.05**, not DreamerV3's
  "1 nat": that is a KL floor and a different quantity. In this normalised
  space a constant predictor already scores ~1, so a 1.0 floor was measured
  (test run, first iteration) to zero out *all* predictor gradients and make
  the JEPA term inert. The floor's ceiling is documented in the config doc.
- **KoLeo** — negative log of the nearest pairwise distance between L2-normalised
  final-frame student latents, weighted 0.1 by default, differentiable through
  the student only. Returns zero for a one-row batch (defined degenerate case).
- **LOUD validation** — tensor type/ndim, non-empty batch/time/features,
  `time > max(horizons)`, finite window/cond/targets, cond shape, encoder
  `forward_features` contract, encoder width vs `cfg.width`, teacher/student
  structure, prediction-vs-target shapes, and all config ranges. Messages name
  the cause and the fix; there is no zero fill or silent fallback.

## Verification

- `uv run ruff check src tests` — clean.
- `uv run pytest -q tests/test_jepa_v2.py` — **8 passed**: multi-horizon shapes
  (`{3,6,10}` → `(B,T-h,W)`), EMA update and hard sync, no gradient to the
  teacher (teacher grads all `None`, student and predictor grads non-zero),
  free-bits floor (MSE 0 → term exactly 5.0 with zero predictor gradient),
  symlog/percentile scale invariance in the log-like regime plus the 1.0 floor
  on a tight range, seeded augmentation reproducibility/reset, finite KoLeo,
  and malformed-batch refusals.
- Compatibility smoke outside the committed tests: `JEPAV2` with a real
  `SmartTeacher` (`forward_features(window, cond)` path, width 16) produced
  finite loss, correct horizon shapes and no teacher gradients on CPU.
- Full suite: **303 passed, 2 failed**. Both failures are in
  `tests/test_teacher_gates.py::test_lateral_and_apogee_targets_on_every_seed`
  and `::test_oracle_positive_control_is_required_and_must_work`. That module
  and its `potok_ai.teacher_gates` dependency are **untracked files from a
  different concurrent task**; neither imports `jepa_v2`, and this task did not
  touch them. The suite is therefore *not* claimed green end-to-end — only the
  lint and the tests in scope are.

## What is deliberately not in scope

- **No A/B was run and no quality claim is made.** Whether this objective
  improves held-out RMSE, closed-loop lateral/apogee, or anything else against
  `jepa.py`/`loopwm.py` is unmeasured. The references above describe mechanisms,
  not results at our scale.
- Not wired into `train.py`, `train_smart.py` or any preset. It is an importable
  instrument with a `loss(student, window, cond)` call and an explicit
  `update_teacher(student)` after the optimiser step; integration is a separate
  decision and a separate A/B.
- No two-hot symexp loss, no KL balance, no action conditioning, no rollout,
  no actor/critic. Targets are continuous latents, not categorical returns.
- Determinism covers this module's masks, dropout and probe init. If the encoder
  itself uses stochastic layers, its own RNG is outside this claim.

## Commit

Local commit on `main`, subject `feat(ai): add JEPA v2 objective with
one-shot multi-horizon targets`, containing exactly the three paths above.
Nothing pushed.
