# Task B01 — AI fast model research

**Status: done.** Local commit `b3bdc06` (`feat(ai): B01 fast inference-only model candidate with exact MAC count`). No push.

## Scope

Only the two assigned paths in `/home/sehaxe/potok-m/ai`:

- `src/potok_ai/fast_model.py`
- `tests/test_fast_model.py`

No other file was edited, staged or committed. `src/potok_ai/__init__.py` was deliberately left unchanged (not assigned to this task).

## What was implemented

`FastModelConfig` (plain dataclass, validated) and `FastModel` (inference-only `nn.Module`):

1. `1x1` stem to `width`.
2. One shared `_FastBlock` applied `loops` times (`1` or `2`): causal **depthwise conv** (`groups=width`, left-pad `kernel-1`, so frame `t` never sees a later frame) → residual → **low-rank state** sweep. The recurrent state is itself `rank`-dimensional (`rank <= width`): `s_t = decay*s_{t-1} + V^T z_t`, readout `z_t + U s_t`. Per-frame state cost is `O(width*rank)`, not `O(width^2)`. `U`/`V` are `nn.Linear(..., bias=False)` so every MAC-bearing op is a standard, hook-observable module.
3. Last-frame readout: two bounded corrections (`tanh * g_max`) plus a confidence (`sigmoid`).

All parameters are seeded from a private `torch.Generator().manual_seed(cfg.seed)`; the global RNG is never consulted. No dropout, batch norm or quantisation annealing — `train()` and `eval()` produce bit-identical output.

No accuracy, latency or "beats LRS-TCN / LoopWM-lite" claim is made anywhere. The module docstring states that such claims require an A/B at the same window and MAC budget.

## Exact MAC count

`mac_breakdown(cfg, batch)` and `analytical_mac(cfg, batch)` count one multiply-accumulate per linear/conv kernel application; biases, scalar decays, LayerNorm affine and softmax are excluded (same convention as `loopwm_lite`). Defaults (`in=14, window=50, width=32, rank=8, kernel=3, loops=1, out=3`):

| component | MACs |
|---|---|
| stem | 22,400 |
| depthwise conv | 4,800 |
| state_in (`V^T`) | 12,800 |
| state_out (`U`) | 12,800 |
| head | 96 |
| **total** | **52,896** |

`loops=2` → **83,296** (exactly +30,400, the per-loop block cost). The test attaches hooks to all five multiply-bearing modules and asserts the runtime count equals the analytical breakdown for three configurations (including `batch=2`, non-default kernel/rank/width) — so the count is verified against executed ops, not just a pinned literal.

## Verification

- `uv run pytest -q tests/test_fast_model.py` → **7 passed**.
- `uv run pytest -q` (whole suite) → **145 passed in 14.12 s**.
- `uv run ruff check src/potok_ai/fast_model.py tests/test_fast_model.py` → **clean**.
- `git diff --cached --check` for the two files → **clean**.

Gates cover: exact MAC breakdown + pinned totals, loops scaling, same-seed bitwise identity including under global-RNG perturbation, different-seed separation, train/eval agreement, causal invariance at several frames for both loop counts, output shape/bounds/finiteness, and config/input validation.

## Issues outside my scope (reported, not fixed)

- `uv run ruff check src tests` (whole repo) fails on `src/potok_ai/conf_integration.py:10`: unused `json` import. That is task B04's untracked file; I did not touch it.
- `git diff --check` on the whole worktree fails on pre-existing trailing whitespace in `ai/out/v3_final/results.csv` and `ai/out/v3_final/runs/lrs/s2/run.log` — both were already modified before I started and are unrelated to B01.

## Skipped / not claimed

No A/B runs, no export/protocol integration, no `__init__.py` re-export, no README changes — those are outside B01's paths or need held-out data the task does not provide. The candidate is research-only until an A/B at matched window and MAC budget says otherwise.
