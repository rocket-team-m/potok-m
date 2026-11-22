# SPDX-License-Identifier: GPL-3.0-or-later
"""Fast inference-only model candidate (task B01).

Composition: a 1x1 stem, ONE shared block of (depthwise causal convolution +
low-rank state recurrence), applied 1 or 2 times, then a bounded readout. The
recurrent state itself is rank-dimensional (rank << width), so per-frame state
memory and update cost are O(rank) instead of O(width^2).

This is a research candidate, not a result. It makes NO accuracy, latency or
"better than LRS-TCN / LoopWM-lite" claim: any such claim needs an A/B on
held-out data at the same window and MAC budget. What this module does provide
is an exact MAC accounting (``mac_breakdown`` / ``analytical_mac``) and
bit-reproducible inference from a private seed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

MIN_LOOPS = 1
MAX_LOOPS = 2


@dataclass
class FastModelConfig:
    """Plain-data architecture of the candidate. Own seed, no global RNG."""

    in_features: int = 14
    window: int = 50
    width: int = 32
    rank: int = 8
    kernel: int = 3
    loops: int = MIN_LOOPS
    out_features: int = 3  # g_roll, g_pitch, confidence
    decay: float = 0.8
    g_max: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        dims = (self.in_features, self.window, self.width, self.rank, self.kernel)
        if min(dims) < 1:
            raise ValueError("feature counts, window, width, rank and kernel must be positive")
        if self.rank > self.width:
            raise ValueError("rank must not exceed width: the state is low-rank by construction")
        if not MIN_LOOPS <= self.loops <= MAX_LOOPS:
            raise ValueError(f"loops must be in {MIN_LOOPS}..{MAX_LOOPS}")
        if self.out_features < 3:
            raise ValueError("out_features must be >= 3: two corrections and one confidence")
        if not 0.0 <= self.decay < 1.0:
            raise ValueError("decay must be in [0, 1)")
        if self.g_max <= 0.0:
            raise ValueError("g_max must be positive")


class _FastBlock(nn.Module):
    """Shared block: causal depthwise conv, then a low-rank state sweep."""

    def __init__(self, cfg: FastModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # groups=width keeps the temporal mixing per channel; padding is added
        # on the left only, so frame t never sees a later frame.
        self.dw = nn.Conv1d(cfg.width, cfg.width, cfg.kernel, groups=cfg.width)
        self.state_in = nn.Linear(cfg.width, cfg.rank, bias=False)
        self.state_out = nn.Linear(cfg.rank, cfg.width, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch = z.shape[0]
        pad = self.cfg.kernel - 1
        causal = F.pad(z.transpose(1, 2), (pad, 0))
        z = z + self.dw(causal).transpose(1, 2)

        state = z.new_zeros((batch, self.cfg.rank))
        frames: list[torch.Tensor] = []
        for frame in z.unbind(dim=1):
            key = self.state_in(frame)
            state = self.cfg.decay * state + key
            frames.append(frame + self.state_out(state))
        return torch.stack(frames, dim=1)


class FastModel(nn.Module):
    """Window ``(B, T, in_features)`` -> bounded ``(B, 3)``. Inference only.

    No dropout, no batch norm, no quantisation annealing and no training path:
    the same weights give the same output in ``train()`` and ``eval()``. Loops
    reuse a single block module, so loop 2 adds compute, not parameters.
    """

    def __init__(self, cfg: FastModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or FastModelConfig()
        c = self.cfg
        self.stem = nn.Conv1d(c.in_features, c.width, 1)
        self.block = _FastBlock(c)
        self.norm = nn.LayerNorm(c.width)
        self.head = nn.Linear(c.width, c.out_features)
        self._init_parameters()

    def _init_parameters(self) -> None:
        """Seed from a private generator; the global RNG is never consulted."""
        generator = torch.Generator().manual_seed(self.cfg.seed)
        scale = self.cfg.width**-0.5
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                if name == "norm.weight":
                    parameter.fill_(1.0)
                elif name.endswith(".bias"):
                    parameter.zero_()
                else:
                    nn.init.normal_(parameter, 0.0, scale, generator=generator)

    def forward_features(self, window: torch.Tensor) -> torch.Tensor:
        """Per-frame causal latent ``(B, T, width)`` before the readout."""
        if window.ndim != 3 or window.shape[-1] != self.cfg.in_features:
            raise ValueError(f"expected a window of shape (batch, time, {self.cfg.in_features})")
        if window.shape[1] < 1:
            raise ValueError("a window needs at least one frame")
        z = self.stem(window.transpose(1, 2)).transpose(1, 2)
        for _ in range(self.cfg.loops):
            z = self.block(z)
        return self.norm(z)

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        latent = self.forward_features(window)
        raw = self.head(latent[:, -1, :])
        corrections = torch.tanh(raw[:, :2]) * self.cfg.g_max
        confidence = torch.sigmoid(raw[:, 2:3])
        return torch.cat([corrections, confidence], dim=-1)


def mac_breakdown(cfg: FastModelConfig | None = None, batch: int = 1) -> dict[str, int]:
    """Exact MACs per component for one inference over a ``window`` frames.

    Convention (the one ``loopwm_lite`` uses): a MAC is one multiply-accumulate
    inside a linear map or a convolution kernel application. Biases, scalar
    decays, LayerNorm affine terms and softmax are not MACs here. The per-frame
    state work is ``width*rank`` going in and ``rank*width`` coming out.
    """
    c = cfg or FastModelConfig()
    if batch < 1:
        raise ValueError("batch must be positive")
    b, t, w, r = batch, c.window, c.width, c.rank
    return {
        "stem": b * t * c.in_features * w,
        "depthwise_conv": b * c.loops * t * w * c.kernel,
        "state_in": b * c.loops * t * w * r,
        "state_out": b * c.loops * t * r * w,
        "head": b * w * c.out_features,
    }


def analytical_mac(cfg: FastModelConfig | None = None, batch: int = 1) -> int:
    """Total exact MAC count; equals the hook-counted runtime number."""
    return sum(mac_breakdown(cfg, batch).values())
