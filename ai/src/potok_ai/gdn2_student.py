# SPDX-License-Identifier: GPL-3.0-or-later
"""Experimental stateful low-rank gated-delta recurrent student (POST07).

One frame per tick, with a fixed recurrent state carried between calls. The
state of each of the two blocks is a per-head ``[head_dim, head_dim]`` matrix
(4 heads x 16 x 16 = 4 KB per block in fp32). The update is the gated delta
rule: a channel-wise erase gate and a scalar decay contract the old memory, a
rank-1 delta term removes the component along the current key, and a separate
channel-wise write gate commits a new key/value association. The readout is a
matrix-vector product against the updated state. There is no attention over a
window, no convolution and no history buffer: history lives only in the state.

This is a research candidate, not a result. It makes NO accuracy, latency or
"beats LRS-TCN / LoopWM-lite" claim: such a claim needs an A/B on held-out data
with 3 seeds at the same MAC budget (WORLD07 §6). Pure PyTorch, CPU, no
external CUDA, no quantisation and no training loop are involved here.

MAC convention: one MAC is one multiply-accumulate inside an ``nn.Linear`` or
inside a recurrent matrix-vector product. Biases, sigmoid/tanh gates, RMSNorm,
L2 key normalisation, scalar decay and the rank-1 outer products are not MACs.
``mac_breakdown`` / ``analytical_mac`` report the exact per-tick count.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

FRAME_FEATURES = 14  # 8 raw sensors + 6 physics features (SPEC_SMART.md)
COND_FEATURES = 6  # mass, area, cd0, static margin, thrust, gyro noise
MIN_WIDTH = 48
MAX_WIDTH = 64


@dataclass
class GDN2StudentConfig:
    """Plain-data architecture of the candidate. Own seed, no global RNG."""

    in_features: int = FRAME_FEATURES
    cond_features: int = COND_FEATURES
    width: int = MIN_WIDTH
    heads: int = 4
    head_dim: int = 16
    n_blocks: int = 2
    out_features: int = 3  # g_roll, g_pitch, confidence
    g_max: float = 1.0
    decay: float = 0.9
    seed: int = 0

    def __post_init__(self) -> None:
        if self.in_features < 8 or self.cond_features < 1:
            raise ValueError("a frame needs >=8 sensors and >=1 conditioning value")
        if not MIN_WIDTH <= self.width <= MAX_WIDTH:
            raise ValueError(f"width must be in {MIN_WIDTH}..{MAX_WIDTH}")
        if self.heads < 1 or self.head_dim < 1:
            raise ValueError("heads and head_dim must be positive")
        if self.n_blocks < 1:
            raise ValueError("n_blocks must be positive")
        if self.out_features != 3:
            raise ValueError("out_features must be 3: two corrections and one confidence")
        if not 0.0 < self.decay < 1.0:
            raise ValueError("decay must be in (0, 1) for bounded state")
        if self.g_max <= 0.0:
            raise ValueError("g_max must be positive")


class _RMSNorm(nn.Module):
    """Root-mean-square normalisation with a learned gain, no bias."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        return x * scale * self.weight


class _GDN2Block(nn.Module):
    """One gated-delta block with its own state; no parameters are shared."""

    def __init__(self, cfg: GDN2StudentConfig) -> None:
        super().__init__()
        self.cfg = cfg
        inner = cfg.heads * cfg.head_dim
        self.q = nn.Linear(cfg.width, inner)
        self.k = nn.Linear(cfg.width, inner)
        self.v = nn.Linear(cfg.width, inner)
        self.erase = nn.Linear(cfg.width, inner)
        self.write = nn.Linear(cfg.width, inner)
        self.beta = nn.Linear(cfg.width, cfg.heads)
        self.out = nn.Linear(inner, cfg.width)

    def forward(self, z: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Consume one batch of latents and return ``(increment, new_state)``."""
        h, d = self.cfg.heads, self.cfg.head_dim
        shape = z.shape[:-1] + (h, d)
        q = self.q(z).view(shape)
        k = F.normalize(self.k(z).view(shape), dim=-1)
        v = torch.tanh(self.v(z).view(shape))
        erase = torch.sigmoid(self.erase(z).view(shape)).unsqueeze(-1)
        write = torch.sigmoid(self.write(z).view(shape)).unsqueeze(-2)
        lr = torch.sigmoid(self.beta(z)).unsqueeze(-1).unsqueeze(-1)

        # Contract the old memory, erase along the current key, commit v.
        memory = self.cfg.decay * (1.0 - erase) * state
        key_projection = torch.einsum("...k,...kv->...v", k, memory)
        memory = memory - lr * k.unsqueeze(-1) * key_projection.unsqueeze(-2)
        memory = memory + lr * k.unsqueeze(-1) * (write * v.unsqueeze(-2))

        readout = torch.einsum("...k,...kv->...v", q, memory)
        increment = self.out(readout.reshape(z.shape[:-1] + (h * d,)))
        return increment, memory.detach()


class GDN2Student(nn.Module):
    """One frame per tick -> two bounded corrections and a confidence.

    ``step`` is the runtime API: it consumes one frame, advances the persistent
    state and returns ``(corrections[2], confidence[1])``. ``forward`` is a pure
    window sweep from zero state (analysis / causality tests only); it returns
    per-tick outputs ``(batch, time, 3)`` and never touches the persistent state.
    """

    def __init__(self, cfg: GDN2StudentConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or GDN2StudentConfig()
        c = self.cfg
        self.stem = nn.Linear(c.in_features, c.width)
        self.film = nn.Linear(c.cond_features, 2 * c.width)
        self.norm = _RMSNorm(c.width)
        self.blocks = nn.ModuleList(_GDN2Block(c) for _ in range(c.n_blocks))
        self.norm_out = _RMSNorm(c.width)
        self.head = nn.Linear(c.width, c.out_features)
        # Non-persistent: a fresh process always starts from zero state.
        self.register_buffer("state", torch.zeros(c.n_blocks, c.heads, c.head_dim, c.head_dim))
        self._init_parameters()

    def _init_parameters(self) -> None:
        """Seed every parameter from a local generator; global RNG is unused."""
        generator = torch.Generator().manual_seed(self.cfg.seed)
        scale = self.cfg.width**-0.5
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                if name.endswith(("norm.weight", "norm_out.weight")):
                    parameter.fill_(1.0)
                elif name.endswith("beta.bias"):
                    parameter.fill_(-2.0)  # gentle initial delta edits
                elif name.endswith(".bias"):
                    parameter.zero_()
                else:
                    nn.init.normal_(parameter, 0.0, scale, generator=generator)

    def reset_state(self) -> None:
        """Zero the recurrent state of every block (episode / phase boundary)."""
        self.state.zero_()

    def _as_cond(self, cond, leading: tuple[int, ...]) -> torch.Tensor:
        need = self.cfg.cond_features
        if cond is None:
            return torch.zeros(leading + (need,), dtype=torch.float32)
        tensor = torch.as_tensor(cond, dtype=torch.float32)
        if tensor.shape == (need,):
            return tensor.expand(leading + (need,))
        if tensor.ndim == 2 and tensor.shape[-1] == need:
            if len(leading) == 1 and tensor.shape[0] == leading[0]:
                return tensor
            if len(leading) == 2:
                if tensor.shape[0] == leading[0]:
                    return tensor.unsqueeze(1).expand(leading + (need,))
                if tensor.shape[0] == leading[1]:
                    return tensor.unsqueeze(0).expand(leading + (need,))
        if tensor.ndim == 3 and tensor.shape == leading + (need,):
            return tensor
        raise ValueError(f"conditioning must be ({need},) or broadcastable to {leading + (need,)}")

    def _tick(
        self, frame: torch.Tensor, cond: torch.Tensor, state: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """One tick over a batch of frames; no state is written to self."""
        z = self.stem(frame)
        gamma, shift = self.film(cond).chunk(2, dim=-1)
        z = self.norm(z * (1.0 + torch.tanh(gamma)) + torch.tanh(shift))
        next_state: list[torch.Tensor] = []
        for block, previous in zip(self.blocks, state):
            increment, updated = block(z, previous)
            z = z + increment
            next_state.append(updated)
        raw = self.head(self.norm_out(z))
        corrections = torch.tanh(raw[..., :2]) * self.cfg.g_max
        confidence = torch.sigmoid(raw[..., 2:3])
        return corrections, confidence, next_state

    def _sweep(
        self, frames: torch.Tensor, conds: torch.Tensor, state: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        corrections: list[torch.Tensor] = []
        confidences: list[torch.Tensor] = []
        for frame, cond in zip(frames.unbind(1), conds.unbind(1)):
            corr, conf, state = self._tick(frame, cond, state)
            corrections.append(corr)
            confidences.append(conf)
        return torch.stack(corrections, dim=1), torch.stack(confidences, dim=1), state

    def step(self, frame, cond=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Consume one frame and return ``(corrections[2], confidence[1])``."""
        tensor = torch.as_tensor(frame, dtype=torch.float32).reshape(-1)
        if tensor.numel() > self.cfg.in_features:
            raise ValueError(
                f"frame has {tensor.numel()} values, expected <= {self.cfg.in_features}"
            )
        if tensor.numel() < self.cfg.in_features:
            tensor = torch.cat(
                [tensor, torch.zeros(self.cfg.in_features - tensor.numel(), dtype=tensor.dtype)]
            )
        previous = [self.state[index].detach().unsqueeze(0) for index in range(self.cfg.n_blocks)]
        corrections, confidence, updated = self._tick(
            tensor.unsqueeze(0), self._as_cond(cond, (1,)), previous
        )
        for index, value in enumerate(updated):
            self.state[index] = value[0].detach()
        return corrections[0], confidence[0]

    def forward(self, window, cond=None) -> torch.Tensor:
        """Pure sweep over ``(time, features)`` or ``(batch, time, features)``."""
        frames = torch.as_tensor(window, dtype=torch.float32)
        if frames.ndim == 2:
            frames = frames.unsqueeze(0)
        if frames.ndim != 3 or frames.shape[-1] != self.cfg.in_features:
            raise ValueError(f"expected (batch, time, {self.cfg.in_features}) frames")
        if frames.shape[1] < 1:
            raise ValueError("a window needs at least one frame")
        batch = frames.shape[0]
        conds = self._as_cond(cond, (batch, frames.shape[1]))
        state = [
            torch.zeros_like(self.state[index]).unsqueeze(0).expand(batch, -1, -1, -1)
            for index in range(self.cfg.n_blocks)
        ]
        corrections, confidence, _ = self._sweep(frames, conds, state)
        return torch.cat([corrections, confidence], dim=-1)


def mac_breakdown(cfg: GDN2StudentConfig | None = None, batch: int = 1) -> dict[str, int]:
    """Exact MACs per tick. Biases/gates/norms/outer products are not MACs."""
    c = cfg or GDN2StudentConfig()
    if batch < 1:
        raise ValueError("batch must be positive")
    inner = c.heads * c.head_dim
    return {
        "stem": batch * c.in_features * c.width,
        "film": batch * c.cond_features * 2 * c.width,
        "block_projections": batch * c.n_blocks * 5 * c.width * inner,
        "block_beta": batch * c.n_blocks * c.width * c.heads,
        "block_recurrence": batch * c.n_blocks * 2 * c.heads * c.head_dim**2,
        "block_out": batch * c.n_blocks * inner * c.width,
        "head": batch * c.width * c.out_features,
    }


def analytical_mac(cfg: GDN2StudentConfig | None = None, batch: int = 1) -> int:
    """Total exact MAC count for one tick."""
    return sum(mac_breakdown(cfg, batch).values())
