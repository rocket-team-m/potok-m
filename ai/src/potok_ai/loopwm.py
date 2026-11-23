# SPDX-License-Identifier: GPL-3.0-or-later
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

import math
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F

# The latent targets are LayerNorm'd EMA-encoder outputs, so their batch std is
# normally ~1. Divide the JEPA regression by that std (floored here so a
# constant target batch cannot divide by zero): a constant predictor then scores
# ~1, the same convention as train_smart.normalized_correction_loss.
LATENT_SCALE_FLOOR = 1e-3


def _project_row_column_spectral(weight: torch.Tensor, bound: float) -> torch.Tensor:
    """Scale a square weight so its spectral norm cannot exceed ``bound``.

    sigma(W) <= sqrt(||W||_1 ||W||_inf), so bounding the largest absolute row
    and column sums by ``bound`` bounds the spectral norm too. A weight already
    inside the ball is returned unchanged.
    """
    rows = weight.abs().sum(dim=1).amax()
    cols = weight.abs().sum(dim=0).amax()
    scale = torch.maximum(rows, cols).clamp_min(bound) / bound
    return weight / scale


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

    # LoopWM §3.2 Eq. 5-7 / §3.3-3.4 Eq. 11-12. Both switches default off so the
    # shipped teacher's forward, checkpoints and A/B gates are unchanged.
    spectral_residual: bool = False
    spectral_bound: float = 1.0
    learned_exit: bool = False
    exit_gate_threshold: float = 0.5
    exit_entropy_alpha: float = 0.01

    def __post_init__(self) -> None:
        if min(self.in_features, self.cond_features, self.width, self.heads, self.max_iters) < 1:
            raise ValueError("feature counts, width, heads and max_iters must be positive")
        if self.width % self.heads:
            raise ValueError("width must be divisible by the number of attention heads")
        if not math.isfinite(self.exit_threshold) or self.exit_threshold < 0:
            raise ValueError("exit_threshold must be finite and non-negative")
        if not self.horizons or any(h < 1 for h in self.horizons):
            raise ValueError("latent horizons must be positive frame offsets")
        if self.spectral_residual and (
            not math.isfinite(self.spectral_bound) or self.spectral_bound <= 0
        ):
            raise ValueError("spectral_bound must be finite and positive")
        if not math.isfinite(self.exit_gate_threshold) or not 0 <= self.exit_gate_threshold <= 1:
            raise ValueError("exit_gate_threshold must lie in [0, 1]")
        if not math.isfinite(self.exit_entropy_alpha) or self.exit_entropy_alpha < 0:
            raise ValueError("exit_entropy_alpha must be finite and non-negative")


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
        self.spectral_bound = cfg.spectral_bound if cfg.spectral_residual else None
        if self.spectral_bound is not None:
            # Residual-branch retention (Eq. 5-7). Identity init makes the first
            # pass a no-op for bound >= 1; the projection keeps it contractive.
            self.retention = nn.Linear(cfg.width, cfg.width, bias=False)
            nn.init.eye_(self.retention.weight)

    def projected_residual_weight(self) -> torch.Tensor | None:
        """Effective (spectrally projected) residual weight; None when disabled."""
        if self.spectral_bound is None:
            return None
        return _project_row_column_spectral(self.retention.weight, self.spectral_bound)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch, time, width = z.shape
        h = self.norm1(z)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        view = (batch, time, self.heads, self.head_dim)
        q, k, v = (t.view(view).transpose(1, 2) for t in (q, k, v))
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).reshape(batch, time, width)
        increment = self.proj(attn)
        increment = increment + self.ff(self.norm2(z + increment))
        if self.spectral_bound is not None:
            increment = F.linear(increment, self.projected_residual_weight())
        return increment


@dataclass
class RefineInfo:
    """Adaptive-compute trace returned alongside the refined latent."""

    iters: int
    updates: list[float] = field(default_factory=list)
    gates: list[torch.Tensor] = field(default_factory=list)


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
        if c.learned_exit:
            self.exit_gate = nn.Linear(c.width, 1)

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
        if not math.isfinite(limit) or limit < 0:
            raise ValueError("threshold must be finite and non-negative")

        gamma, beta = self.film(cond).chunk(2, dim=-1)
        z = self.encoder(window) * gamma.unsqueeze(1) + beta.unsqueeze(1)
        updates: list[float] = []
        gates: list[torch.Tensor] = []
        iters = 0
        for iteration in range(self.cfg.max_iters):
            increment = self.block(z)  # same weights on every pass
            magnitude = float(increment.norm(dim=-1).mean().detach())
            updates.append(magnitude)
            z = z + increment
            iters = iteration + 1
            stop = magnitude < limit
            if self.cfg.learned_exit:
                gate = torch.sigmoid(self.exit_gate(z[:, -1]))  # (batch, 1)
                gates.append(gate)
                stop = stop or bool(gate.detach().mean() > self.cfg.exit_gate_threshold)
            if stop:
                break
        return self.norm(z), RefineInfo(iters=iters, updates=updates, gates=gates)

    def exit_entropy_loss(self, info: RefineInfo) -> torch.Tensor:
        """Eq. 11: ``-alpha * E sum_t H(g^t)``; minimising it resists exit collapse."""
        if not info.gates:
            return next(self.parameters()).new_zeros(())
        gates = torch.stack(info.gates).clamp(1e-6, 1.0 - 1e-6)
        entropy = -(gates * gates.log() + (1 - gates) * (1 - gates).log())
        return -self.cfg.exit_entropy_alpha * entropy.mean()

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
        """Normalised MSE of predicted future latents against the EMA target.

        The scale is one scalar: the std of the detached target latents over the
        whole batch (all horizons' time and width entries), floored at
        ``LATENT_SCALE_FLOOR``. Dividing both prediction and target by it makes
        the term independent of the encoder's latent magnitude; a constant
        predictor scores ~1, as in the correction loss.
        """
        time = window.shape[1]
        if any(h >= time for h in self.cfg.horizons):
            raise ValueError("the window must be longer than every latent horizon")
        if latent is None:
            latent, _ = self.refine(window, cond)
        predictions = self.latent_predictions(latent)
        with torch.no_grad():
            targets = target_encoder(window).detach()
            scale = targets.std(unbiased=False).clamp_min(LATENT_SCALE_FLOOR)
        return torch.stack(
            [
                F.mse_loss(predictions[h] / scale, targets[:, h:, :] / scale)
                for h in self.cfg.horizons
            ]
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
