"""Muon+ optimiser (Muon+ arXiv:2602.21545 Eq. 4; Kimi K3 arXiv:2607.24653).

Source of the polynomial: the Jordan / ``muon.py`` quintic
``(3.4445, -4.7750, 2.0315)``, found in the reference line at
``dormouse/docs/papers/muon-plus.md`` (row D6, exact match to ``NS_COEFFS``).
Muon+ adds ColRow = ``norm_row(norm_col(X))`` (Eq. 7) after the polar step;
the ``max(1, m/n)**0.5`` scale is the documented Jordan deviation from the
paper's ``(m/n)**0.5`` (row D3). Plain AdamW is a comparison option only.
"""

from __future__ import annotations

import torch
from torch.optim import AdamW, Optimizer

NS_COEFFS = (3.4445, -4.7750, 2.0315)
# dormouse burn-spectral uses this cubic for TSCT retraction (same paper, App. D.3);
# it is NOT the Muon+ polynomial (row D5 of docs/papers/muon-plus.md).
POLAR_EXPRESS_COEFFS = (1.875, -1.25, 0.375)


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
    """Muon+ Eq. 7: ``Norm_row(Norm_col(X))``."""
    return norm_row(norm_col(x, eps), eps)


def muon_update(
    momentum: torch.Tensor,
    steps: int = 5,
    coeffs: tuple[float, float, float] = NS_COEFFS,
) -> torch.Tensor:
    orthogonal = colrow(orthNS(momentum, steps, coeffs))
    rows, cols = momentum.shape[-2], momentum.shape[-1]
    return orthogonal * max(1.0, rows / cols) ** 0.5


class MuonPlus(Optimizer):
    """Muon+ by default; non-matrix parameters follow the paper's AdamW routing."""

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
    ) -> None:
        if lr < 0 or not 0.0 <= momentum < 1.0 or ns_steps < 1:
            raise ValueError("invalid Muon+ hyperparameters")
        defaults = {"lr": lr, "momentum": momentum, "weight_decay": weight_decay}
        super().__init__(params, defaults)
        self.ns_steps = ns_steps
        self.coeffs = coeffs
        fallback = [p for group in self.param_groups for p in group["params"] if p.ndim < 2]
        self._adamw = AdamW(fallback, lr=adam_lr, betas=betas, eps=eps) if fallback else None

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
                update = muon_update(buffer, self.ns_steps, self.coeffs)
                if group["weight_decay"]:
                    parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.add_(update, alpha=-group["lr"])
        if self._adamw is not None:
            self._adamw.step()
        return loss


Muon = MuonPlus  # back-compatible name; the recipe is Muon+
