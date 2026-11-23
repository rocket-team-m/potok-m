<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
# Optimizer notes: our Muon+ and the Kimi K3 deltas

Scope: `src/potok_ai/optim.py`, `tests/test_optim_k3.py`. Evidence base: the
MUON01 research report (`/home/sehaxe/potok-m/.bulba/night/MUON01.md`), which
read the primary sources listed below. No coefficient or recipe detail is
invented here; where the report is silent, we say so.

## 1. What the primary sources actually say

| Source | Role | What it pins |
|---|---|---|
| Kimi K3, arXiv:2607.24653 | K3 | §2.5: per-head Muon on the momentum matrices of **Q, K and V** (NS per head-block). §3.3: K2's weight clipping, cosine + 1% warmup, weight decay 0.1. |
| Kimi K2, arXiv:2507.20534 | inherited mechanism | §2.1 Alg. 1: `M = μM + G`; `"Match Adam RMS"` scale; QK-Clip `γ = τ/S_max^h`, `q_c,k_c *= √γ`, τ=100 in K2's run. |
| Jordan et al. (2024), https://kellerjordan.github.io/posts/muon/ | Muon itself | Website, **no arXiv ID**; K3 §2.5 cites this. Quintic `(3.4445, −4.7750, 2.0315)` and `ns_steps=5` are Jordan's reference choices. |
| Muon scaling, arXiv:2502.16982 | scale evidence | Cited by K3 §5.2.2 only, not by §2.5. |
| Muon+, arXiv:2602.21545 | **our** base recipe | ColRow = `norm_row(norm_col(X))` (Eq. 7). K3 does **not** cite or use Muon+. |
| Polar Express, arXiv:2505.16932 | our cubic constant | Converged triple `(1.875, −1.25, 0.375)`, used by dormouse's TSCT retraction, not by K3. |

K3 §2.5 verbatim (from MUON01 §3.1): *"instead of applying Newton–Schulz
orthogonalization to the full Q, K, and V projection matrices, we partition
their momentum matrices along the head dimension and orthogonalize each head's
block separately."* The report prints **no** NS coefficients, NS step count,
momentum value, peak LR, or τ; all such numbers below are ours or K2's.

### Citation correction

`arXiv:2409.16986` is **"Harnessing Diversity for Important Data Selection in
Pretraining Large Language Models"** — a data-selection paper, not Muon. The
bogus ID lived in the separate dormouse2 port that MUON01 inspected; it is not
in this repo. This port now cites the Jordan website for Muon, arXiv:2607.24653
for K3, arXiv:2507.20534 for QK-Clip, arXiv:2502.16982 for scaling, and
arXiv:2602.21545 only for **our** ColRow step.

## 2. Changes applied

Both refinements are opt-in through the typed `K3Config`; defaults are
`num_heads=0`, `qk_clip=False`, `qk_clip_tau=100.0`, so every existing gate and
training run keeps the historical Muon+ behaviour.

1. **Per-head Q/K/V momentum split** (`per_head_muon_update`). With
   `num_heads=N`, a parameter named `*_q_proj.weight`, `*_k_proj.weight` or
   `*_v_proj.weight` with output shape `N * head_dim` has its momentum reshaped
   to `(N, head_dim, in_features)`; each block gets the same polar + ColRow +
   `max(1, m/n)^0.5` path as a standalone matrix, then the blocks are
   reassembled. The split covers **V as well as Q and K**, per K3 §2.5. The
   optimizer needs `model.named_parameters()` to identify these projections;
   otherwise no parameter matches and the whole-matrix path is used.
2. **QK-Clip** (`qk_clip_`, K2 Alg. 1 via K3 §3.3). After each step, each
   paired q/k head is scaled by `sqrt(γ_h)` with `γ_h = min(1, τ/S_max^h)`,
   τ defaulting to K2's 100 (K3 does not restate it). Invalid configs fail
   loudly: τ ≤ 0, clip without heads, or clip without named q/k pairs.

`tests/test_optim_k3.py` pins: per-head math equals independent head-wise
updates and differs from the unsplit update; the split reaches q, k **and** v;
the clip caps the measured per-head scale at τ, is a no-op below τ, changes
only q/k in a real `step()`, and the typed config validates its inputs.

## 3. Deliberate non-changes (ours, not K3 claims)

- **NS coefficients and step count stay `(3.4445, −4.7750, 2.0315)` / 5.**
  K3 prints none; we do not attribute our polynomial to K3, and we do not
  switch to a new one.
- **ColRow and `POLAR_EXPRESS_COEFFS` stay.** They are our Muon+/Polar Express
  engineering choices. K3 mentions neither Muon+, ColRow nor Polar Express;
  claiming otherwise is unsupported.
- **Momentum form `m = μm + (1−μ)g` stays** (Jordan reference / Muon+ Alg. 1),
  even though K2's Alg. 1 writes `M = μM + G`. K3 does not restate the form.
- **Routing, AdamW fallback, LR 0.02, Muon weight decay 0.0 stay.** K3's
  "matrix parameters" and "weight decay 0.1 throughout" are not a full recipe;
  changing them would rewrite every historical gate and is not one of the two
  evidence-backed deltas in scope.
- **No new optimizer family.** `Muon = MuonPlus` remains the back-compatible
  alias; the schedule-free / WSD arms are port additions, not K3.

## 4. Usage

```python
from potok_ai.optim import K3Config, MuonPlus

opt = MuonPlus(
    model.named_parameters(),
    lr=0.02,
    k3=K3Config(num_heads=8, qk_clip=True, qk_clip_tau=100.0),
)
```

### Caveats, stated rather than hidden

- The clip's S_max is a **weight-space bound** (`||Q_h^T K_h||_2 / √d`, worst
  case over unit-norm inputs) because the optimizer step has no attention
  logits. K2/K3 measure S_max from actual activations; pass the measured
  per-head maxima via `qk_clip_(..., smax=...)` to apply the paper's exact rule.
- The MLA-specific `q_r *= γ` / `k_r` untouched split cannot be expressed for
  dense weights; both q and k head blocks are scaled symmetrically. This is
  our adaptation of K2's rule, not a verbatim K3 statement.
- Per-head splitting is name-based and therefore inert in callers that pass
  `model.parameters()` instead of `model.named_parameters()` (the current
  `train.py` CLI does the former). Enabling it there is a separate integration
  task outside this change's file scope.

## 5. Verification

```
uv run python -m pytest tests/test_optim.py tests/test_optim_k3.py -q
uv run python -m pytest -q
uv run ruff check src/potok_ai/optim.py tests/test_optim_k3.py
uv run ruff format --check src/potok_ai/optim.py tests/test_optim_k3.py
```
