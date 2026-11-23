# SPDX-License-Identifier: GPL-3.0-or-later
"""Muon+ optimizer plus the two evidence-backed Kimi K3 refinements.

Base recipe (OUR engineering choice, not a K3 claim): Muon+ (arXiv:2602.21545)
-- a Newton-Schulz polar step followed by ColRow (Eq. 7) and a
``max(1, m/n)**0.5`` scale. Muon itself is Jordan et al. (2024),
https://kellerjordan.github.io/posts/muon/ (website, no arXiv ID); the scaling
paper is arXiv:2502.16982. arXiv:2409.16986 is a *data-selection* paper and is
NOT a Muon reference.

Kimi K3 (arXiv:2607.24653) contributes only two optimizer facts:
* §2.5 per-head Muon -- split the momentum matrices of Q, K **and V** along the
  head dimension and orthogonalize each head's block separately;
* §3.3 K2's QK-Clip (arXiv:2507.20534 §2.1 Alg. 1). K2 used tau=100; K3 does
  not restate tau.

Both are opt-in through :class:`K3Config` and off by default, so the historical
training gates are unchanged. K3 does not use Muon+, ColRow or Polar Express;
the NS coefficients, step count and prescale below are ours, not K3 facts.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.optim import AdamW, Optimizer

NS_COEFFS = (3.4445, -4.7750, 2.0315)
# Ours: the converged Polar Express (arXiv:2505.16932) triple used by the
# dormouse TSCT retraction -- NOT a K3 coefficient and NOT the Muon+ polynomial.
POLAR_EXPRESS_COEFFS = (1.875, -1.25, 0.375)


@dataclass(frozen=True)
class K3Config:
    """Opt-in Kimi K3 refinements; defaults reproduce the historical port.

    ``num_heads > 0`` enables per-head Newton-Schulz on parameters whose names
    end in ``q_proj``, ``k_proj`` or ``v_proj`` (weight layout
    ``(heads * head_dim, in_features)``). ``qk_clip`` applies K2's logit clamp
    to paired q/k projections after every step. Pass ``model.named_parameters()``
    to the optimizer so the projections can be identified by name.
    """

    num_heads: int = 0
    qkv_suffixes: tuple[str, ...] = ("q_proj", "k_proj", "v_proj")
    qk_clip: bool = False
    qk_clip_tau: float = 100.0

    def __post_init__(self) -> None:
        if self.num_heads < 0:
            raise ValueError("num_heads must be >= 0 (0 disables per-head Muon)")
        if self.qk_clip_tau <= 0:
            raise ValueError("qk_clip_tau must be positive")
        if len(self.qkv_suffixes) < 2:
            raise ValueError("qkv_suffixes needs at least q and k suffixes")
        if self.qk_clip and self.num_heads < 1:
            raise ValueError("qk_clip requires num_heads >= 1")


def orthNS(
    momentum: torch.Tensor, steps: int = 5, coeffs: tuple[float, float, float] = NS_COEFFS
) -> torch.Tensor:
    """Polar-orthogonalise a matrix with a Newton-Schulz polynomial."""
    transposed = momentum.shape[-2] > momentum.shape[-1]
    x = momentum.mT if transposed else momentum
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    a, b, c = coeffs
    for _ in range(steps):
        gram = x @ x.mT
        x = a * x + (b * gram + c * gram @ gram) @ x
    return x.mT if transposed else x


def norm_col(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.square().sum(dim=-2, keepdim=True) + eps).sqrt()


def norm_row(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.square().sum(dim=-1, keepdim=True) + eps).sqrt()


def colrow(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Muon+ Eq. 7: ``Norm_row(Norm_col(X))`` (ours, not K3)."""
    return norm_row(norm_col(x, eps), eps)


def muon_update(
    momentum: torch.Tensor,
    steps: int = 5,
    coeffs: tuple[float, float, float] = NS_COEFFS,
) -> torch.Tensor:
    orthogonal = colrow(orthNS(momentum, steps, coeffs))
    rows, cols = momentum.shape[-2], momentum.shape[-1]
    return orthogonal * max(1.0, rows / cols) ** 0.5


def per_head_muon_update(
    momentum: torch.Tensor,
    heads: int,
    steps: int = 5,
    coeffs: tuple[float, float, float] = NS_COEFFS,
) -> torch.Tensor:
    """K3 §2.5: orthogonalize each head's momentum block independently.

    The full matrix is partitioned along the output/head axis; each
    ``(head_dim, in_features)`` block goes through the same polar+ColRow path
    as ``muon_update``. The reassembled update has the original shape.
    """
    if momentum.ndim != 2:
        raise ValueError("per-head Muon needs a 2-D matrix")
    rows, cols = momentum.shape
    if heads < 1 or rows % heads:
        raise ValueError("out_features must be divisible by num_heads")
    head_dim = rows // heads
    blocks = momentum.reshape(heads, head_dim, cols)
    updates = torch.stack([muon_update(blocks[i], steps, coeffs) for i in range(heads)])
    return updates.reshape(rows, cols)


@torch.no_grad()
def qk_clip_(
    q: torch.Tensor,
    k: torch.Tensor,
    heads: int,
    tau: float,
    smax: torch.Tensor | None = None,
) -> torch.Tensor:
    """Clamp the per-head attention-logit scale (K2 Alg. 1, used by K3 §3.3).

    ``q`` and ``k`` are dense ``(heads * head_dim, in_features)`` projection
    weights. For each head, ``gamma_h = min(1, tau / S_max^h)`` and both q and k
    are scaled by ``sqrt(gamma_h)``, so the q.k logit scale shrinks by gamma.
    K2's MLA-specific ``q_r *= gamma`` / ``k_r``-untouched split does not apply
    to a dense projection, so the whole head block is scaled symmetrically.

    With ``smax=None`` the measured logit maximum is unavailable to the
    optimizer step, so we use the weight-space bound
    ``max_h ||Q_h^T K_h||_2 / sqrt(head_dim)`` -- OUR adaptation, not the
    paper's rule. Pass the measured per-head logit maxima via ``smax`` to use
    K2's exact clip. Returns the per-head gamma actually applied.
    """
    if q.ndim != 2 or q.shape != k.shape:
        raise ValueError("q and k must be equally shaped 2-D projections")
    rows, cols = q.shape
    if heads < 1 or rows % heads:
        raise ValueError("out_features must be divisible by num_heads")
    head_dim = rows // heads
    q_heads = q.reshape(heads, head_dim, cols)
    k_heads = k.reshape(heads, head_dim, cols)
    if smax is None:
        gram = torch.bmm(q_heads.float().transpose(1, 2), k_heads.float())
        scale = torch.linalg.svdvals(gram).amax(dim=-1) / head_dim**0.5
    else:
        scale = smax.reshape(heads).float()
    if scale.shape != (heads,):
        raise ValueError("smax must hold one value per head")
    gamma = (tau / scale.clamp_min(torch.finfo(torch.float32).tiny)).clamp(max=1.0)
    factor = gamma.sqrt().reshape(heads, 1, 1).to(q.dtype)
    q.copy_((q_heads * factor).reshape(rows, cols))
    k.copy_((k_heads * factor).reshape(rows, cols))
    return gamma


class MuonPlus(Optimizer):
    """Muon+ by default; K3 per-head Q/K/V and QK-Clip are opt-in."""

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        adam_lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        coeffs: tuple[float, float, float] = NS_COEFFS,
        k3: K3Config | None = None,
    ) -> None:
        if lr < 0 or not 0.0 <= momentum < 1.0 or ns_steps < 1:
            raise ValueError("invalid Muon+ hyperparameters")
        self.k3 = k3 or K3Config()
        tensors: list = []
        names: dict[int, str] = {}
        for entry in params:
            if (
                isinstance(entry, tuple)
                and len(entry) == 2
                and isinstance(entry[0], str)
                and torch.is_tensor(entry[1])
            ):
                names[id(entry[1])] = entry[0]
                tensors.append(entry[1])
            else:
                tensors.append(entry)
        defaults = {"lr": lr, "momentum": momentum, "weight_decay": weight_decay}
        super().__init__(tensors, defaults)
        self._names = names
        self.ns_steps = ns_steps
        self.coeffs = coeffs
        self._qk_pairs = self._find_qk_pairs() if self.k3.qk_clip else []
        if self.k3.qk_clip and not self._qk_pairs:
            raise ValueError("qk_clip needs named q_proj/k_proj parameters")
        fallback = [p for group in self.param_groups for p in group["params"] if p.ndim < 2]
        self._adamw = AdamW(fallback, lr=adam_lr, betas=betas, eps=eps) if fallback else None

    @staticmethod
    def _stem(name: str) -> str:
        return name.removesuffix(".weight")

    def _find_qk_pairs(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        q_suffix, k_suffix = self.k3.qkv_suffixes[:2]
        by_id = {id(p): p for group in self.param_groups for p in group["params"]}
        stems = {pid: self._stem(name) for pid, name in self._names.items()}
        pairs = []
        for pid, stem in stems.items():
            if stem == q_suffix or stem.endswith("." + q_suffix):
                k_id = next(
                    (other for other, s in stems.items() if s == stem[: -len(q_suffix)] + k_suffix),
                    None,
                )
                if k_id is not None:
                    pairs.append((by_id[pid], by_id[k_id]))
        return pairs

    def _is_qkv(self, parameter: torch.Tensor) -> bool:
        name = self._names.get(id(parameter))
        if name is None:
            return False
        stem = self._stem(name)
        return any(stem == s or stem.endswith("." + s) for s in self.k3.qkv_suffixes)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.ndim < 2 or parameter.grad is None:
                    continue
                state = self.state[parameter]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(parameter)
                buffer = state["momentum"]
                buffer.mul_(group["momentum"]).add_(parameter.grad, alpha=1.0 - group["momentum"])
                if self.k3.num_heads and self._is_qkv(parameter):
                    update = per_head_muon_update(
                        buffer, self.k3.num_heads, self.ns_steps, self.coeffs
                    )
                else:
                    update = muon_update(buffer, self.ns_steps, self.coeffs)
                if group["weight_decay"]:
                    parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.add_(update, alpha=-group["lr"])
        for q, k in self._qk_pairs:
            qk_clip_(q, k, self.k3.num_heads, self.k3.qk_clip_tau)
        if self._adamw is not None:
            self._adamw.step()
        return loss


Muon = MuonPlus  # back-compatible name; the recipe is Muon+
