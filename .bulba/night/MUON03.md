# MUON03 — Kimi K3 optimizer deltas, implemented in potok-m/ai

**Status:** done. Local commit only, no push. No subagents. Repo `/home/sehaxe/potok-m`.
**Date:** 2026-11-23. Evidence base: MUON01 (primary sources verified there).

## Scope — exactly four files

- `ai/src/potok_ai/optim.py` (modified; SPDX GPL-3.0-or-later added)
- `ai/tests/test_optim_k3.py` (new; 6 tests, SPDX GPL-3.0-or-later)
- `ai/docs/OPTIMIZER_NOTES.md` (new; GPL notice)
- this report

No other source, config, workflow, gate or artifact was touched. The many
uncommitted files from parallel tasks in this worktree were left alone.

## Applied (evidence-backed only)

1. **Per-head momentum split for Q, K *and* V** — `per_head_muon_update` in
   `optim.py`: momentum `(heads*head_dim, in)` → `(heads, head_dim, in)`, each
   head block through polar+ColRow+scale, reassembled. K3 §2.5 names "the full
   Q, K, and V projection matrices" (verbatim in MUON01 §3.1); the earlier
   dormouse2 port split only Q/K. No V omission here.
2. **QK-Clip, tau=100, typed config, default off** — `qk_clip_` implements
   K2 Alg. 1's `γ_h = min(1, τ/S_max^h)`, scaling q and k head blocks by
   `√γ_h`; `K3Config(num_heads=0, qk_clip=False, qk_clip_tau=100.0)` is frozen
   and validated. τ=100 is **K2's** number; K3 §3.3 invokes the mechanism but
   does not restate τ — documented as such. Defaults reproduce the historical
   Muon+ path, so all pre-existing gates are unchanged.
3. **Citation correction** — the module docstring now cites the Jordan website
   (no arXiv ID) for Muon, arXiv:2607.24653 for K3, arXiv:2507.20534 for
   QK-Clip, arXiv:2502.16982 for scaling, and arXiv:2602.21545 only for **our**
   ColRow. K3 is explicitly stated *not* to use Muon+/ColRow/Polar Express.

## Explicitly not done

- **No NS coefficient invented or changed.** `NS_COEFFS=(3.4445,−4.7750,2.0315)`
  and `POLAR_EXPRESS_COEFFS=(1.875,−1.25,0.375)`, 5 steps, prescale, ColRow and
  `max(1,m/n)^0.5` remain labelled as **our** engineering choices; K3 prints
  none of them. Momentum form, routing, LR and weight decay also unchanged.
- **`2409.16986` was not present in this repo's `optim.py`** (MUON01 found it in
  the separate dormouse2 port, `muon.py:5`). No false fix was fabricated: the
  correct references are stated in the docstring and in `OPTIMIZER_NOTES.md`,
  together with the fact that that ID is a data-selection paper.
- **No CLI wiring.** `train.py` passes `model.parameters()`, so per-head
  detection needs `model.named_parameters()` and `K3Config` to be enabled by a
  future integration task; this stays inert by design and is documented.
- **Weight-space proxy disclosed.** Because an optimizer step has no attention
  logits, S_max defaults to `‖Q_hᵀK_h‖₂/√d`; the measured per-head logit maxima
  can be supplied via `qk_clip_(..., smax=...)` for K2's exact rule. K2's
  MLA-only q_r/k_r split does not apply to dense weights — both are scaled.

## Verification (commands run in `ai/`)

| Command | Result |
|---|---|
| `uv run python -m pytest tests/test_optim.py tests/test_optim_k3.py -q` | 10 passed |
| `uv run python -m pytest -q` | 331 passed in 119.02 s |
| `uv run ruff check src/potok_ai/optim.py tests/test_optim_k3.py` | All checks passed |
| `uv run ruff format --check src/potok_ai/optim.py tests/test_optim_k3.py` | 2 files already formatted |

The 6 new gates pin: per-head math == independent head-wise updates and ≠
unsplit; the split reaches q, k and v; τ caps each head; no-op below τ; a real
`step()` changes only q/k under clip (v identical to the unclipped twin); and
typed-config validation fails loudly (τ≤0, clip without heads, unnamed params).

All code is original; no foreign source was copied. No push was performed.
