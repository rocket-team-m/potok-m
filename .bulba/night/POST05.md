# POST05 — LoopWM spectral residual + learned sigmoid exit (optional, default off)

**Date:** 2026-11-23
**Repo:** `/home/sehaxe/potok-m/ai` (workspace root `/home/sehaxe/potok-m`)
**Status:** done, committed locally, not pushed.
**Code commit:** `6341520` — `feat(ai): POST05 optional LoopWM spectral residual and learned exit`

## Deliverable

Touched exactly two files; both now carry `SPDX-License-Identifier: GPL-3.0-or-later`:

- `ai/src/potok_ai/loopwm.py` (+65/−3)
- `ai/tests/test_loopwm.py` (+86/−1, 4 new tests → module total 16)

No CLI, config-schema, trainer, fixture or report file under `ai/src/` was changed. `train_loopwm.py`
was read only; it keeps constructing `LoopWMConfig` with its existing keyword arguments, so the new
fields take their defaults.

## What was added

Five new `LoopWMConfig` fields, all default-off or inert:

| Field | Default | Role |
|---|---|---|
| `spectral_residual` | `False` | enables the spectral constraint on the shared block's residual branch |
| `spectral_bound` | `1.0` | upper bound on the projected spectral norm (validated `> 0`, finite) |
| `learned_exit` | `False` | enables the learned sigmoid exit gate |
| `exit_gate_threshold` | `0.5` | gate threshold `τ` (validated in `[0, 1]`) |
| `exit_entropy_alpha` | `0.01` | coefficient `α` of Eq. 11's entropy pressure (validated `>= 0`) |

**Spectral residual** (LoopWM 2606.18208 §3.2, Eq. 5–7, p.6). `LoopBlock` gains an identity-initialised
`d×d` retention matrix applied to the residual increment. Each forward projects it with a
row/column norm bound: `σ(W) <= sqrt(||W||₁·||W||∞)`, so clamping the largest absolute row and
column sums to `bound` guarantees `σ(W) <= bound`. Identity init makes the first pass a no-op at
`bound >= 1`; the projection is differentiable and is applied in forward so no optimizer hook or
trainer change is needed. `LoopBlock.projected_residual_weight()` exposes the effective weight for
inspection. The paper's exact negative-diagonal ZOH `Ā = exp(Δ·diag(−exp(a)))` is not reproduced —
this is the cheaper row/column bound the task asked for; the contractive guarantee is the same, the
parameterisation is not.

**Learned sigmoid exit** (§3.3–3.4, Eq. 11–12, p.7). `g⁽ᵗ⁾ = σ(w_gᵀ z_last + b_g)`, one gate per
sample from the final frame of the shared latent; the batch-mean gate above `τ` stops the loop.
Gates are appended to `RefineInfo.gates`, and `LoopWM.exit_entropy_loss(info)` returns
`−α·mean_t H(g⁽ᵗ⁾)` with clamped logs, so minimising it resists exit collapse. The existing
`max_iters` `for` loop is the hard cap and is unchanged; the magnitude threshold still applies, so
the gate can only add an earlier stop.

## Coverage

| Required property | Test | What it pins |
|---|---|---|
| Norm bound | `test_spectral_residual_norm_bound` | `torch.linalg.matrix_norm(..., 2) <= bound` at init and after a 100× weight blow-up; effective weight finite and square; disabled model has no `retention` parameter; zero bound refused. |
| Exit count | `test_learned_exit_count_and_hard_cap` | gate count == iteration count == update count; probabilities in `[0,1]`; saturated gate → 1 iteration; `τ=1` → exactly `max_iters=6`; disabled model has no `exit_gate` and an empty gate trace. |
| Determinism | `test_spectral_exit_determinism` | two same-seed models have bit-equal parameters, latent, iteration count and gate tensors; repeated forward on one model is bit-identical. |
| Backward finiteness | `test_spectral_exit_backward_finite` | entropy term finite and `<= 0`; combined head + entropy loss finite; all gradients finite; new `retention` and `exit_gate` weights receive non-zero gradients; empty trace gives exactly zero pressure. |

## Verification

- `uv run ruff check src/potok_ai/loopwm.py tests/test_loopwm.py`: **All checks passed!**
- `uv run pytest -q tests/test_loopwm.py`: **16 passed** (4.36 s).
- `uv run pytest -q` (whole AI suite): **278 passed** in 168.48 s.
- Default-path equivalence: loaded the pre-change `loopwm.py` from `HEAD` as a standalone module,
  built both models with the same seed, and asserted identical `state_dict` keys and tensors plus
  bit-identical `refine` latents, iteration counts, update traces and `forward` outputs at
  `threshold=0.0` and `1e9`. **Default training results are unchanged** by construction.
- `git diff --check` on both files: clean.

## Notes

- The entropy term is exposed as `LoopWM.exit_entropy_loss(info)` rather than folded into the
  trainer's loss: adding it to `train_teacher` would require editing `ai/src/potok_ai/train_loopwm.py`,
  which is outside this task's file scope. Until that one-line change lands, `learned_exit=True`
  gives deterministic early stopping but no training pressure on the gate.
- No CLI flags were added — scope was two files, and `build_parser` lives in `train_loopwm.py`.
  The features are reachable by constructing `LoopWMConfig` with the new fields.
- Whole-tree `uv run ruff check src tests` is **not** green in this shared checkout: concurrent
  agents' `src/potok_ai/dagger.py` (2× UP040) and `tests/test_dagger.py` (I001, C408) currently fail.
  Those files were not touched. The scoped check on the two POST05 files is green.
- Other agents' modified/untracked paths (`out/`, `sim/`, `docs/`, night reports, etc.) were left
  alone. Only the two code files and this report were staged; no `git add .`, no push, no
  destructive git operation.
