# SPDX-License-Identifier: GPL-3.0-or-later
"""Typed residual policy for the teacher stages (SHIELD02 / RES-SHIELD).

The action is the classical controller plus a bounded learned correction::

    command = clamp(baseline + gain * learned_correction, -envelope, envelope)

``gain`` starts at exactly zero, so an untrained policy *is* the classical
controller (the jump-start property). Every correction is projected into the
typed :class:`~potok_ai.student.CommandEnvelope` before it is returned, and each
sample carries the reason for the decision.

This is a supervised-stage composition rule, not a flight controller: it does
not tune the physics baseline, add a CBF/QP shield, or make any flight claim.

Evidence: RES-SHIELD (2026-11-24) §5.2/§6 — residual superposition over a
classical controller (arXiv:1812.03201, arXiv:2611.19318) and jump-starting from
the classical guide (arXiv:2407.15083) motivate the deployable student form.
No certificate is implied: the filter here is a box projection on the command.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch
from torch import nn

from .student import CommandEnvelope


class GateReason(str, Enum):
    """Why a correction was applied, bounded or dropped (fixed check order)."""

    DISABLED = "disabled"
    NON_FINITE = "non_finite"
    CLAMPED = "clamped"
    ZERO_GAIN = "zero_gain"
    OK = "ok"


@dataclass(frozen=True)
class ResidualConfig:
    """Typed knobs. The only bound is the simulator's command envelope."""

    envelope: CommandEnvelope = field(default_factory=CommandEnvelope)

    def __post_init__(self) -> None:
        if not isinstance(self.envelope, CommandEnvelope):
            raise TypeError(f"envelope must be a CommandEnvelope, got {self.envelope!r}")


@dataclass(frozen=True)
class ResidualDecision:
    """Final command, applied correction and one gate reason per sample."""

    command: torch.Tensor
    correction: torch.Tensor
    reasons: tuple[GateReason, ...]


class ResidualPolicy(nn.Module):
    """Superpose a bounded learned correction on a classical baseline.

    The zero-initialised scalar ``gain`` is the jump-start: at init the returned
    command equals the baseline. Gradients flow to ``gain`` and to the
    correction's producer once training moves it off zero.
    """

    def __init__(self, cfg: ResidualConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ResidualConfig()
        self.gain = nn.Parameter(torch.zeros(()))
        self.reset_reason_counts()

    def reset_reason_counts(self) -> None:
        """Clear the observed reason histogram (a plain counter, no graph state)."""
        self._reason_counts = {reason.value: 0 for reason in GateReason}

    def reason_counts(self) -> dict[str, int]:
        """Snapshot of reasons seen since the last reset, keyed by ``GateReason`` value."""
        return dict(self._reason_counts)

    def forward(
        self,
        baseline: torch.Tensor,
        learned_correction: torch.Tensor,
        *,
        enabled: bool = True,
    ) -> ResidualDecision:
        """Return the gated action; both inputs have leading shape ``[..., 2]``."""
        if baseline.shape != learned_correction.shape:
            raise ValueError(
                f"baseline {tuple(baseline.shape)} and correction "
                f"{tuple(learned_correction.shape)} must have the same shape"
            )
        if baseline.shape[-1] != 2:
            raise ValueError(f"expected a two-channel command, got {tuple(baseline.shape)}")

        gain = self.gain.to(dtype=baseline.dtype)
        scaled = gain * learned_correction
        finite = torch.isfinite(scaled).all(dim=-1) & torch.isfinite(baseline).all(dim=-1)
        safe = torch.where(finite.unsqueeze(-1), scaled, torch.zeros_like(scaled))
        limit = self.cfg.envelope.accel
        command = (baseline + safe).clamp(-limit, limit)
        applied = command - baseline
        clamped = finite & (command != baseline + safe).any(dim=-1)
        if not enabled:
            command = baseline.clone()
            applied = torch.zeros_like(baseline)

        zero_gain = bool(gain.detach() == 0)
        finite_rows = finite.reshape(-1).detach().cpu().tolist()
        clamped_rows = clamped.reshape(-1).detach().cpu().tolist()
        reasons: list[GateReason] = []
        for row_finite, row_clamped in zip(finite_rows, clamped_rows):
            if not enabled:
                reasons.append(GateReason.DISABLED)
            elif not row_finite:
                reasons.append(GateReason.NON_FINITE)
            elif row_clamped:
                reasons.append(GateReason.CLAMPED)
            elif zero_gain:
                reasons.append(GateReason.ZERO_GAIN)
            else:
                reasons.append(GateReason.OK)
        if len(reasons) != finite.numel():
            raise AssertionError("one gate reason per command row is required")
        for reason in reasons:
            self._reason_counts[reason.value] += 1
        return ResidualDecision(command, applied, tuple(reasons))
