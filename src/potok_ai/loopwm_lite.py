"""Onboard LoopWM-lite verifier (SPEC_ENGINE.md v1).

50x14 window -> Perceiver encoder (16 latent tokens, width 64) -> ONE
parameter-shared Transformer block applied at most twice with adaptive exit on
the mean latent-update norm. Heads predict the 3/6-frame-ahead latents of the
full LoopWM teacher, a two-value correction and a confidence.

There is no raw reconstruction anywhere: the latent targets are detached outputs
of the privileged teacher's own latent heads, and the reference correction is a
loss target only, never a forward input.

MAC is counted analytically in ``analytical_mac`` (one window inference). With
the defaults (16 tokens, width 64, rank-8 attention, two passes, bottlenecked
FFN) it lands at ~0.29 M, inside the 0.2-0.4 M band sketched in SPEC_ENGINE.md.
A plain full-width attention block here would cost ~0.8 M, so the attention and
FFN are bottlenecked; this is the deliberate accuracy/latency trade.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .loopwm import RefineInfo

LATENT_TOKENS = 16
LITE_WIDTH = 64
HARD_ITER_CAP = 2


@dataclass
class LiteConfig:
    """Architecture of the lightweight onboard verifier. All fields are data."""

    in_features: int = 14
    cond_features: int = 6
    window: int = 50
    latent_tokens: int = LATENT_TOKENS
    width: int = LITE_WIDTH
    attn_rank: int = 8  # bottleneck rank of every attention projection
    heads: int = 2
    max_iters: int = HARD_ITER_CAP  # hard cap; threshold may stop earlier
    exit_threshold: float = 1e-2
    horizons: tuple[int, ...] = (3, 6)
    out_features: int = 3  # g_roll, g_pitch, confidence
    g_max: float = 1.0
    target_width: int = 128  # full LoopWM width, for the latent-target projection

    def __post_init__(self) -> None:
        if min(self.in_features, self.cond_features, self.window, self.latent_tokens) < 1:
            raise ValueError("feature counts, window and latent_tokens must be positive")
        if self.width % self.heads or self.attn_rank % self.heads:
            raise ValueError("width and attn_rank must be divisible by heads")
        if not 1 <= self.max_iters <= HARD_ITER_CAP:
            raise ValueError(f"max_iters must be in 1..{HARD_ITER_CAP}")
        if self.exit_threshold < 0:
            raise ValueError("exit_threshold must be non-negative")
        if self.horizons != (3, 6):
            raise ValueError("LoopWM-lite predicts exactly the 3 and 6 frame latents")
        if self.target_width < 1:
            raise ValueError("target_width must be positive")


class LiteAttention(nn.Module):
    """Multi-head attention with a rank bottleneck on every projection."""

    def __init__(self, q_dim: int, kv_dim: int, rank: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = rank // heads
        self.q = nn.Linear(q_dim, rank)
        self.k = nn.Linear(kv_dim, rank)
        self.v = nn.Linear(kv_dim, rank)
        self.out = nn.Linear(rank, q_dim)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        batch, length, _ = query.shape
        source = key_value.shape[1]
        view_q = (batch, length, self.heads, self.head_dim)
        view_kv = (batch, source, self.heads, self.head_dim)
        q = self.q(query).view(view_q).transpose(1, 2)
        k = self.k(key_value).view(view_kv).transpose(1, 2)
        v = self.v(key_value).view(view_kv).transpose(1, 2)
        merged = F.scaled_dot_product_attention(q, k, v)
        return self.out(merged.transpose(1, 2).reshape(batch, length, -1))


class PerceiverEncoder(nn.Module):
    """Compress a 50x14 window into 16 learned latent tokens via cross-attention."""

    def __init__(self, cfg: LiteConfig) -> None:
        super().__init__()
        self.input_proj = nn.Linear(cfg.in_features, cfg.width)
        self.queries = nn.Parameter(torch.empty(cfg.latent_tokens, cfg.width))
        self.cross = LiteAttention(cfg.width, cfg.width, cfg.attn_rank, cfg.heads)
        self.film = nn.Sequential(
            nn.Linear(cfg.cond_features, 2 * cfg.width),
            nn.SiLU(),
            nn.Linear(2 * cfg.width, 2 * cfg.width),
        )
        self.norm = nn.LayerNorm(cfg.width)
        nn.init.normal_(self.queries, std=0.02)

    def forward(self, window: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if window.ndim != 3 or window.shape[-1] != self.input_proj.in_features:
            raise ValueError("expected a window of shape (batch, time, in_features)")
        batch = window.shape[0]
        if cond.shape != (batch, self.film[0].in_features):
            raise ValueError("conditioning must have shape (batch, cond_features)")
        tokens = self.input_proj(window)
        queries = self.queries.unsqueeze(0).expand(batch, -1, -1)
        z = self.norm(queries + self.cross(queries, tokens))
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        return z * gamma.unsqueeze(1) + beta.unsqueeze(1)


class LiteBlock(nn.Module):
    """The single shared Transformer block; returns an increment, not a state."""

    def __init__(self, cfg: LiteConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.width)
        self.attn = LiteAttention(cfg.width, cfg.width, cfg.attn_rank, cfg.heads)
        self.norm2 = nn.LayerNorm(cfg.width)
        inner = 2 * cfg.attn_rank
        self.ff = nn.Sequential(
            nn.Linear(cfg.width, inner),
            nn.GELU(),
            nn.Linear(inner, cfg.width),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        increment = self.attn(self.norm1(z), self.norm1(z))
        return increment + self.ff(self.norm2(z + increment))


class LiteModel(nn.Module):
    """Window + conditioning -> corrections, with two-pass latent refinement."""

    def __init__(self, cfg: LiteConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or LiteConfig()
        c = self.cfg
        self.encoder = PerceiverEncoder(c)
        self.block = LiteBlock(c)  # ONE module, reused on every pass
        self.norm = nn.LayerNorm(c.width)
        self.latent_heads = nn.ModuleDict(
            {str(h): nn.Linear(c.width, c.width) for h in c.horizons}
        )
        self.head_corr = nn.Linear(c.width, 2)
        self.head_conf = nn.Linear(c.width, 1)
        self.target_proj: nn.Linear | None = None
        self.teacher: nn.Module | None = None

    def set_teacher(self, teacher: nn.Module) -> None:
        """Attach a frozen full LoopWM teacher as the source of latent targets."""
        width = teacher.cfg.width  # LoopWMConfig
        # Keep the privileged teacher out of the module tree: it must never be
        # optimised, exported or counted as part of the onboard model.
        object.__setattr__(self, "teacher", teacher.eval())
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.target_proj = nn.Linear(width, self.cfg.width).to(
            next(self.parameters()).device
        )

    def refine(
        self, window: torch.Tensor, cond: torch.Tensor, threshold: float | None = None
    ) -> tuple[torch.Tensor, RefineInfo]:
        """Refine the latent tokens; exit when the update norm is small."""
        z = self.encoder(window, cond)
        limit = self.cfg.exit_threshold if threshold is None else threshold
        if limit < 0:
            raise ValueError("threshold must be non-negative")
        updates: list[float] = []
        iters = 0
        for iteration in range(self.cfg.max_iters):
            increment = self.block(z)
            magnitude = float(increment.norm(dim=-1).mean().detach())
            updates.append(magnitude)
            z = z + increment
            iters = iteration + 1
            if magnitude < limit:
                break
        return self.norm(z), RefineInfo(iters=iters, updates=updates)

    def latent_predictions(self, pooled: torch.Tensor) -> dict[int, torch.Tensor]:
        """Predicted full-LoopWM latent summary at each horizon, shape (B, width)."""
        return {h: self.latent_heads[str(h)](pooled) for h in self.cfg.horizons}

    @torch.no_grad()
    def teacher_targets(self, window: torch.Tensor, cond: torch.Tensor) -> dict[int, torch.Tensor]:
        """Detached 3/6-frame latent targets from the full teacher."""
        if self.teacher is None or self.target_proj is None:
            raise ValueError("LoopWM-lite needs a full LoopWM teacher for latent targets")
        latent, _ = self.teacher.refine(window, cond)
        predictions = self.teacher.latent_predictions(latent)
        return {
            h: self.target_proj(prediction.mean(dim=1)).detach()
            for h, prediction in predictions.items()
            if h in self.cfg.horizons
        }

    def head_logits(self, pooled: torch.Tensor) -> torch.Tensor:
        """Raw [roll, pitch, confidence] logits of the pooled latent."""
        return torch.cat([self.head_corr(pooled), self.head_conf(pooled)], dim=-1)

    def head_output(self, raw: torch.Tensor) -> torch.Tensor:
        """Bounded corrections plus confidence from the raw head logits."""
        corrections = torch.tanh(raw[:, :2]) * self.cfg.g_max
        confidence = torch.sigmoid(raw[:, 2:3])
        return torch.cat([corrections, confidence], dim=-1)

    def head_from_latent(self, pooled: torch.Tensor) -> torch.Tensor:
        """Bounded corrections plus confidence from the pooled latent summary."""
        return self.head_output(self.head_logits(pooled))

    def forward(self, window: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        latent, _ = self.refine(window, cond)
        return self.head_from_latent(latent.mean(dim=1))


def analytical_mac(cfg: LiteConfig | None = None) -> int:
    """Exact analytic MAC count for one window inference (counting every pass).

    Each Linear(a, b) applied to n tokens costs n*a*b MACs; attention scores and
    weighted sums cost heads*q*k*(rank/heads) = q*k*rank. The teacher and its
    target projection are excluded from onboard cost, except the projection
    itself, which the verifier pays at training time only -- hence it is listed
    separately in the breakdown comment below and excluded from the reported
    inference number.
    """
    c = cfg or LiteConfig()
    t, l, d, r = c.window, c.latent_tokens, c.width, c.attn_rank
    encoder = t * c.in_features * d  # input_proj
    encoder += l * d * r + 2 * t * d * r  # cross q, k, v
    encoder += 2 * l * t * r  # scores + weighted sum
    encoder += l * r * d  # cross out
    encoder += c.cond_features * 2 * d + (2 * d) * (2 * d)  # FiLM

    inner = 2 * r
    block = l * d * r * 3 + 2 * l * l * r + l * r * d  # attention
    block += l * d * inner + l * inner * d  # bottleneck FFN
    heads = len(c.horizons) * d * d + d * (2 + 1)  # latent heads + output heads

    mac = encoder + c.max_iters * block + heads
    # With the defaults: encoder 142_336 + 2 * 69_632 + heads 8_384 = 289_984.
    return int(mac)
