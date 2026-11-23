"""Causal TCN + TSCT model for the Potok-M assistant.

Architecture (concept PDF, §4): a 50x8 sensor window passes through five
residual causal depthwise-separable blocks (dilation 1,2,4,8,16, kernel 3).
Channel mixing uses TSCT low-rank maps ``W ~= U diag(s) V^T`` whose factors
are ternary (b1.58) and whose activations are 4-bit with a Hadamard rotation
(BitNet v2). The head emits two bounded corrections and a confidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def sylvester_hadamard(n: int) -> torch.Tensor:
    """Orthonormal n x n Sylvester Hadamard matrix (n must be a power of two)."""
    if n < 1 or n & (n - 1):
        raise ValueError("Hadamard rotation requires a power-of-two width")
    h = torch.tensor([[1.0]])
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h / (n**0.5)


def fast_hadamard(x: torch.Tensor) -> torch.Tensor:
    """Normalised fast Walsh-Hadamard transform: O(d log d), no weight matrix.

    Applying it twice is the identity, so it doubles as its own inverse.
    """
    n = x.shape[-1]
    if n < 1 or n & (n - 1):
        raise ValueError("fast Hadamard transform requires a power-of-two width")
    prefix = x.shape[:-1]
    y = x
    stage = 1
    while stage < n:
        pairs = y.reshape(*prefix, n // (2 * stage), 2, stage)
        left, right = pairs[..., 0, :], pairs[..., 1, :]
        y = torch.stack((left + right, left - right), dim=-2).reshape(*prefix, n)
        stage *= 2
    return y / (n**0.5)


def ternary_codes(x: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the b1.58 codes in {-1, 0, 1} and their scale ``mean|x|``."""
    alpha = x.detach().abs().mean().clamp_min(eps)
    codes = torch.clamp(torch.round(x.detach() / alpha), -1.0, 1.0)
    return codes, alpha


def ternary_quantize(x: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure b1.58 forward: ``alpha * clip(round(x/alpha), -1, 1)`` and ``alpha``."""
    codes, alpha = ternary_codes(x, eps)
    return alpha * codes, alpha


class TernarySTE(nn.Module):
    """Ternary quantiser with straight-through annealing.

    In training the forward value is ``x + a * (q - x)`` with ``a`` ramping
    0 -> 1, while the gradient is the identity. In eval/export the forward is
    the pure ternary value, independent of the anneal setting.
    """

    def __init__(self, anneal: float = 1.0, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.register_buffer("anneal", torch.tensor(float(anneal)))

    def set_anneal(self, value: float) -> None:
        self.anneal.fill_(float(value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            quantized, _ = ternary_quantize(x, self.eps)
            return quantized
        alpha = x.detach().abs().mean().clamp_min(self.eps)
        codes = torch.clamp(torch.round(x.detach() / alpha), -1.0, 1.0)
        return x + self.anneal * (alpha * codes - x).detach()

    def extra_repr(self) -> str:
        return f"anneal={float(self.anneal):.3f}"


class Hadamard4Bit(nn.Module):
    """4-bit symmetric activations with a Hadamard pre-rotation (BitNet v2).

    Rotate by an orthonormal Hadamard matrix, quantise to levels -7..7 with a
    per-row scale, then rotate back. The gradient is straight-through.
    """

    LEVELS = 7

    def __init__(self, dim: int, enabled: bool = True) -> None:
        super().__init__()
        if dim < 1 or dim & (dim - 1):
            raise ValueError("Hadamard rotation requires a power-of-two width")
        self.enabled = enabled
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled or not self.training:
            return x
        y = fast_hadamard(x)
        scale = y.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / self.LEVELS
        codes = torch.clamp(torch.round(y / scale), -self.LEVELS, self.LEVELS)
        # FWHT is self-inverse; x + fwht(y_q - y) == fwht(y_q) with identity grad.
        return x + fast_hadamard(scale * codes - y).detach()


class TSCTLinear(nn.Module):
    """Low-rank linear map ``W ~= U diag(s) V^T`` with ternary factors."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        ternary: bool = True,
        act4bit: bool = False,
        fused: bool = True,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be positive")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.fused = fused
        self.U = nn.Parameter(torch.empty(in_features, rank))
        self.s = nn.Parameter(torch.empty(rank))
        self.V = nn.Parameter(torch.empty(out_features, rank))
        self.ternary = TernarySTE() if ternary else nn.Identity()
        self.act = Hadamard4Bit(in_features) if act4bit else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.U, std=self.rank**-0.5)
        nn.init.normal_(self.V, std=self.rank**-0.5)
        nn.init.ones_(self.s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(x)
        u = self.ternary(self.U)
        v = self.ternary(self.V)
        if self.fused:
            # Fold the two small GEMMs into the pointwise map: one launch.
            return F.linear(x, (u * self.s) @ v.T)
        return (x @ (u * self.s)) @ v.T

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, rank={self.rank}"


class CausalDSBlock(nn.Module):
    """Residual causal depthwise-separable block with a TSCT pointwise map."""

    def __init__(
        self,
        width: int,
        dilation: int,
        kernel: int,
        rank: int,
        ternary: bool,
        act4bit: bool,
    ) -> None:
        super().__init__()
        self.dw = nn.Conv1d(width, width, kernel, dilation=dilation, groups=width, padding=0)
        self.pw = TSCTLinear(width, width, rank, ternary=ternary, act4bit=act4bit)
        self.norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C). Left-pad only, so frame t never sees the future.
        pad = (self.dw.kernel_size[0] - 1) * self.dw.dilation[0]
        z = F.pad(x.transpose(1, 2), (pad, 0))
        z = self.norm(self.dw(z).transpose(1, 2))
        return x + self.pw(z)


@dataclass
class ModelConfig:
    in_features: int = 8
    width: int = 128
    rank: int = 64
    kernel: int = 3
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16)
    out_features: int = 3
    g_max: float = 1.0
    ternary: bool = True
    act4bit: bool = True


class PotokModel(nn.Module):
    """Window -> (pitch correction, yaw correction, confidence)."""

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig()
        c = self.cfg
        self.stem = nn.Conv1d(c.in_features, c.width, 1)
        self.blocks = nn.ModuleList(
            [CausalDSBlock(c.width, d, c.kernel, c.rank, c.ternary, c.act4bit) for d in c.dilations]
        )
        self.norm = nn.LayerNorm(c.width)
        self.head = nn.Linear(c.width, c.out_features)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the causal per-frame latent state, shape (B, T, width)."""
        z = self.stem(x.transpose(1, 2)).transpose(1, 2)
        for block in self.blocks:
            z = block(z)
        return self.norm(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a window (B, T, F) to bounded corrections and confidence (B, 3)."""
        if x.ndim != 3:
            raise ValueError("expected a window of shape (batch, time, features)")
        h = self.forward_features(x)[:, -1, :]
        raw = self.head(h)
        corrections = torch.tanh(raw[:, :2]) * self.cfg.g_max
        confidence = torch.sigmoid(raw[:, 2:3])
        return torch.cat([corrections, confidence], dim=-1)

    def set_quant_anneal(self, value: float) -> None:
        """Ramp the straight-through ternary quantisation (0 = fp, 1 = ternary)."""
        for module in self.modules():
            if isinstance(module, TernarySTE):
                module.set_anneal(value)

    def quantized_factors(
        self,
    ) -> dict[str, tuple[torch.Tensor, float, torch.Tensor, float, torch.Tensor]]:
        """Return exportable factors per TSCT map: (U codes, alpha_u, s, V codes, alpha_v)."""
        result = {}
        for name, module in self.named_modules():
            if isinstance(module, TSCTLinear):
                u, alpha_u = ternary_codes(module.U.detach())
                v, alpha_v = ternary_codes(module.V.detach())
                result[name] = (
                    u.to(torch.int8).cpu(),
                    float(alpha_u),
                    module.s.detach().cpu(),
                    v.to(torch.int8).cpu(),
                    float(alpha_v),
                )
        return result


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Smart teacher (SPEC_SMART.md): physics features + vehicle conditioning
# ---------------------------------------------------------------------------


@dataclass
class SmartConfig:
    """Big teacher: 14-frame features, 6 conditioning numbers, 4 layers, 2 loops."""

    in_features: int = 14  # 8 raw telemetry + 6 physics features
    cond_features: int = 6  # mass, area, cd0, static margin, thrust, gyro noise
    width: int = 128
    rank: int = 64
    kernel: int = 3
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    loops: int = 2
    out_features: int = 3  # g_roll, g_pitch, confidence
    g_max: float = 1.0
    jepa_weight: float = 0.05
    mtp_weight: float = 0.02
    mtp_k: int = 3
    ema_momentum: float = 0.999
    ternary: bool = True
    act4bit: bool = True


class SmartTeacher(nn.Module):
    """Windowed causal TCN with two weight-shared loop passes and AttnRes.

    Conditioning choice: **FiLM**. The six vehicle numbers are constant for a
    whole flight, so a small MLP maps them to a per-channel scale and bias on
    every frame of the stem output. A global-token concatenation was rejected:
    a token inserted at frame 0 would let the residual stack attenuate it away,
    while FiLM reaches every timestep for the cost of two width-sized vectors.
    The two passes reuse the *same* ``blocks`` module list (weight sharing);
    a learned query scores content keys of both loop outputs and a softmax
    allocates the mixture (AttnRes-style).
    """

    def __init__(self, cfg: SmartConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or SmartConfig()
        c = self.cfg
        if c.in_features < 1 or c.cond_features < 1 or c.loops < 1:
            raise ValueError("feature counts and loop count must be positive")
        if c.width < 1 or c.width & (c.width - 1):
            raise ValueError("teacher width must be a power of two for Hadamard activations")

        self.stem = nn.Conv1d(c.in_features, c.width, 1)
        self.film = nn.Sequential(
            nn.Linear(c.cond_features, 2 * c.width),
            nn.SiLU(),
            nn.Linear(2 * c.width, 2 * c.width),
        )
        self.blocks = nn.ModuleList(
            CausalDSBlock(c.width, d, c.kernel, c.rank, c.ternary, c.act4bit) for d in c.dilations
        )
        self.norm = nn.LayerNorm(c.width)
        self.key = nn.Linear(c.width, c.width, bias=False)
        self.query = nn.Parameter(torch.empty(c.width))
        self.mtp_head = nn.Linear(c.width, c.mtp_k * c.in_features)
        self.head = nn.Linear(c.width, c.out_features)
        self.last_alloc: torch.Tensor | None = None
        nn.init.normal_(self.query, std=c.width**-0.5)

    def forward_features(
        self, window: torch.Tensor, cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the allocated per-frame latent ``(B, T, width)``."""
        if window.ndim != 3 or window.shape[-1] != self.cfg.in_features:
            raise ValueError(f"expected a window of shape (batch, time, {self.cfg.in_features})")
        if cond is None:
            cond = torch.zeros(
                window.shape[0], self.cfg.cond_features, device=window.device, dtype=window.dtype
            )
        if cond.shape != (window.shape[0], self.cfg.cond_features):
            raise ValueError(f"conditioning must have shape (batch, {self.cfg.cond_features})")

        h = self.stem(window.transpose(1, 2)).transpose(1, 2)
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        h = h * gamma.unsqueeze(1) + beta.unsqueeze(1)

        outputs: list[torch.Tensor] = []
        z = h
        for _ in range(self.cfg.loops):
            for block in self.blocks:
                z = block(z)
            outputs.append(z)

        keys = torch.stack([self.key(output[:, -1]) for output in outputs], dim=1)
        scores = (keys * self.query.view(1, 1, -1)).sum(-1) / (self.cfg.width**0.5)
        weights = torch.softmax(scores, dim=-1)
        self.last_alloc = weights.detach()
        pooled = sum(weights[:, index].view(-1, 1, 1) * out for index, out in enumerate(outputs))
        return self.norm(pooled)

    def head_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Apply the readout to a latent from ``forward_features`` (no second pass)."""
        raw = self.head(latent[:, -1])
        corrections = torch.tanh(raw[:, :2]) * self.cfg.g_max
        confidence = torch.sigmoid(raw[:, 2:3])
        return torch.cat([corrections, confidence], dim=-1)

    def forward(self, window: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Map ``(B, T, 14)`` plus ``(B, 6)`` to bounded corrections and confidence."""
        return self.head_from_latent(self.forward_features(window, cond))
