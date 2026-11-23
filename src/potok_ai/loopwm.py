"""Privileged LoopWM teacher (SPEC_SMART.md, arXiv:2606.18208 idea).

One parameter-shared transformer block refines a latent state iteratively;
compute stops when the mean norm of the latent update falls below a threshold,
with a hard cap of ``max_iters``. The only onboard inputs are the 14-feature
window and the 6 conditioning numbers. The simulator's reference correction
(computed with true wind) is used **only as a loss target**; it is never fed to
the model. JEPA-style heads predict the teacher encoder's latents 3, 6 and 10
frames ahead -- no raw sensor reconstruction anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class LoopWMConfig:
    """Default sizes follow the smart teacher; width need not be a power of two."""

    in_features: int = 14  # 8 raw telemetry + 6 physics features
    cond_features: int = 6  # mass, area, cd0, static margin, thrust, gyro noise
    width: int = 128
    heads: int = 4
    ff_mult: int = 4
    max_iters: int = 6  # hard compute cap
    exit_threshold: float = 1e-3
    horizons: tuple[int, ...] = (3, 6, 10)
    out_features: int = 3  # g_roll, g_pitch, confidence
    g_max: float = 1.0
    latent_weight: float = 0.05
    ema_momentum: float = 0.999

    def __post_init__(self) -> None:
        if min(self.in_features, self.cond_features, self.width, self.heads, self.max_iters) < 1:
            raise ValueError("feature counts, width, heads and max_iters must be positive")
        if self.width % self.heads:
            raise ValueError("width must be divisible by the number of attention heads")
        if self.exit_threshold < 0:
            raise ValueError("exit_threshold must be non-negative")
        if not self.horizons or any(h < 1 for h in self.horizons):
            raise ValueError("latent horizons must be positive frame offsets")


class CausalLatentEncoder(nn.Module):
    """Small causal stem: (B, T, 14) -> per-frame latents (B, T, width)."""

    def __init__(self, cfg: LoopWMConfig) -> None:
        super().__init__()
        self.stem = nn.Conv1d(cfg.in_features, cfg.width, kernel_size=3)
        self.norm = nn.LayerNorm(cfg.width)

    def forward_features(self, window: torch.Tensor) -> torch.Tensor:
        """Return the per-frame latent sequence; also the JEPA target network API."""
        if window.ndim != 3:
            raise ValueError("expected a window of shape (batch, time, features)")
        x = F.pad(window.transpose(1, 2), (2, 0))  # left-pad only: frame t sees <= t
        return self.norm(self.stem(x).transpose(1, 2))

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        return self.forward_features(window)


class LoopBlock(nn.Module):
    """The single shared block. Returns the *increment* to the latent state."""

    def __init__(self, cfg: LoopWMConfig) -> None:
        super().__init__()
        self.heads = cfg.heads
        self.head_dim = cfg.width // cfg.heads
        self.norm1 = nn.LayerNorm(cfg.width)
        self.qkv = nn.Linear(cfg.width, 3 * cfg.width)
        self.proj = nn.Linear(cfg.width, cfg.width)
        self.norm2 = nn.LayerNorm(cfg.width)
        self.ff = nn.Sequential(
            nn.Linear(cfg.width, cfg.width * cfg.ff_mult),
            nn.GELU(),
            nn.Linear(cfg.width * cfg.ff_mult, cfg.width),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch, time, width = z.shape
        h = self.norm1(z)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        view = (batch, time, self.heads, self.head_dim)
        q, k, v = (t.view(view).transpose(1, 2) for t in (q, k, v))
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).reshape(batch, time, width)
        increment = self.proj(attn)
        return increment + self.ff(self.norm2(z + increment))


@dataclass
class RefineInfo:
    """Adaptive-compute trace returned alongside the refined latent."""

    iters: int
    updates: list[float] = field(default_factory=list)


class LoopWM(nn.Module):
    """Window + conditioning -> corrections, with iterative latent refinement.

    The privileged simulator target is deliberately absent from every forward
    signature: it enters only through the training loss. Latent targets come
    from an EMA copy of ``encoder`` (built by the training loop) and are
    detached, so the JEPA heads cannot reconstruct raw sensor values.
    """

    def __init__(self, cfg: LoopWMConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or LoopWMConfig()
        c = self.cfg
        self.encoder = CausalLatentEncoder(c)
        self.film = nn.Sequential(
            nn.Linear(c.cond_features, 2 * c.width),
            nn.SiLU(),
            nn.Linear(2 * c.width, 2 * c.width),
        )
        self.block = LoopBlock(c)  # ONE module, reused on every iteration
        self.norm = nn.LayerNorm(c.width)
        self.latent_heads = nn.ModuleDict({str(h): nn.Linear(c.width, c.width) for h in c.horizons})
        self.head = nn.Linear(c.width, c.out_features)

    def refine(
        self,
        window: torch.Tensor,
        cond: torch.Tensor | None = None,
        threshold: float | None = None,
    ) -> tuple[torch.Tensor, RefineInfo]:
        """Refine the encoder latent with the shared block; trace the exit."""
        if window.ndim != 3 or window.shape[-1] != self.cfg.in_features:
            raise ValueError(f"expected a window of shape (batch, time, {self.cfg.in_features})")
        if cond is None:
            cond = torch.zeros(
                window.shape[0], self.cfg.cond_features, device=window.device, dtype=window.dtype
            )
        if cond.shape != (window.shape[0], self.cfg.cond_features):
            raise ValueError(f"conditioning must have shape (batch, {self.cfg.cond_features})")
        limit = self.cfg.exit_threshold if threshold is None else threshold
        if limit < 0:
            raise ValueError("threshold must be non-negative")

        gamma, beta = self.film(cond).chunk(2, dim=-1)
        z = self.encoder(window) * gamma.unsqueeze(1) + beta.unsqueeze(1)
        updates: list[float] = []
        iters = 0
        for iteration in range(self.cfg.max_iters):
            increment = self.block(z)  # same weights on every pass
            magnitude = float(increment.norm(dim=-1).mean().detach())
            updates.append(magnitude)
            z = z + increment
            iters = iteration + 1
            if magnitude < limit:
                break
        return self.norm(z), RefineInfo(iters=iters, updates=updates)

    def latent_predictions(self, latent: torch.Tensor) -> dict[int, torch.Tensor]:
        """Predicted encoder latents at each horizon, one (B, T-h, width) per h."""
        return {h: self.latent_heads[str(h)](latent[:, :-h]) for h in self.cfg.horizons}

    def latent_prediction_loss(
        self,
        window: torch.Tensor,
        cond: torch.Tensor | None,
        target_encoder: nn.Module,
        latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """MSE of predicted future latents against the detached encoder target."""
        time = window.shape[1]
        if any(h >= time for h in self.cfg.horizons):
            raise ValueError("the window must be longer than every latent horizon")
        if latent is None:
            latent, _ = self.refine(window, cond)
        predictions = self.latent_predictions(latent)
        with torch.no_grad():
            targets = target_encoder(window).detach()
        return torch.stack(
            [F.mse_loss(predictions[h], targets[:, h:, :]) for h in self.cfg.horizons]
        ).mean()

    def head_logits(self, latent: torch.Tensor) -> torch.Tensor:
        """Raw [roll, pitch, confidence] logits of the final latent frame."""
        return self.head(latent[:, -1])

    def head_output(self, raw: torch.Tensor) -> torch.Tensor:
        """Bounded corrections plus confidence from the raw head logits."""
        corrections = torch.tanh(raw[:, :2]) * self.cfg.g_max
        confidence = torch.sigmoid(raw[:, 2:3])
        return torch.cat([corrections, confidence], dim=-1)

    def head_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Bounded corrections plus confidence from a refined latent."""
        return self.head_output(self.head_logits(latent))

    def forward(self, window: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Map the onboard window and conditioning to (g_roll, g_pitch, confidence)."""
        latent, _ = self.refine(window, cond)
        return self.head_from_latent(latent)
