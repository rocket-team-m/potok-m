# SPDX-License-Identifier: GPL-3.0-or-later
"""JEPA v2: one-shot, multi-horizon latent objective for the privileged teacher.

Deltas from :mod:`potok_ai.jepa`, all of them instruments and none of them
claimed to improve held-out quality without an A/B:

* the target encoder is an EMA copy of the student encoder, frozen and called
  under ``no_grad``; it is never a gradient path into the target;
* the student window gets a contiguous temporal mask *and* per-sensor dropout
  (a channel is dark for the whole window), both from a local seeded stream;
* one linear probe predicts the 3/6/10-frame targets in a single pass -- no
  recurrent rollout, no compounding error (Fast-JEPA's one-shot target);
* targets are symlog-transformed and 5-95 percentile normalised, with the symlog
  range floored at 1 (DreamerV3's robustness package, arXiv:2301.04104 Eq. 7-11);
* each horizon's normalised MSE has a free-bits floor, so the auxiliary term
  stops pushing once the error is small and cannot collapse the representation.
  This is a floor on normalised MSE, NOT DreamerV3's 1-nat KL floor: a constant
  predictor already scores ~1 here, so a 1.0 floor would make the term inert;
* KoLeo on the student's final latent spreads the representation.

No reconstruction of raw sensors, no reward, no actor. References are for the
mechanisms only: the paper's benchmark numbers say nothing about gust rejection
at our scale.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

DEFAULT_HORIZONS = (3, 6, 10)

__all__ = [
    "DEFAULT_HORIZONS",
    "JEPAV2",
    "EMATeacherV2",
    "JEPAV2Config",
    "JEPAV2Losses",
    "JEPAV2Predictor",
    "koleo_loss",
    "normalize_latents",
    "percentile_stats",
    "symlog",
]


def symlog(x: torch.Tensor) -> torch.Tensor:
    """``sign(x) * log1p(|x|)``: log-like for large values, linear near zero."""
    return x.sign() * torch.log1p(x.abs())


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or inf; feed finite features, do not patch with zero fill")


def percentile_stats(
    targets: torch.Tensor,
    *,
    low: float = 0.05,
    high: float = 0.95,
    floor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(location, scale)`` of symlog targets in 5-95 percentile space.

    The scale is floored at ``floor`` as in DreamerV3's return normalisation:
    a tight target range cannot turn a small error into a huge gradient. The
    statistics are taken over every element of ``targets`` (batch, time, width).
    """
    if targets.ndim < 1 or targets.numel() == 0:
        raise ValueError(f"targets must be a non-empty tensor, got shape {tuple(targets.shape)}")
    _require_finite("targets", targets)
    if not (0.0 <= low < high <= 1.0):
        raise ValueError(f"percentiles must satisfy 0 <= low < high <= 1, got ({low}, {high})")
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError(f"scale floor must be finite and positive, got {floor}")
    z = symlog(targets.detach().to(torch.float32).reshape(-1))
    lo, hi = torch.quantile(z, torch.tensor([low, high], device=z.device, dtype=z.dtype)).unbind(0)
    return lo, (hi - lo).clamp_min(floor)


def normalize_latents(
    targets: torch.Tensor,
    *,
    low: float = 0.05,
    high: float = 0.95,
    floor: float = 1.0,
) -> torch.Tensor:
    """Map detached latent targets into a bounded, scale-robust space."""
    lo, scale = percentile_stats(targets, low=low, high=high, floor=floor)
    return (symlog(targets.to(torch.float32)) - lo) / scale


def koleo_loss(features: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """``-mean log(min pairwise distance)`` over L2-normalised rows."""
    if features.ndim != 2:
        raise ValueError(f"KoLeo expects (batch, width) features, got {tuple(features.shape)}")
    _require_finite("KoLeo features", features)
    if features.shape[0] < 2:
        return features.sum() * 0.0
    z = F.normalize(features, dim=-1)
    distances = torch.cdist(z, z)
    distances = distances + torch.eye(z.shape[0], device=z.device, dtype=z.dtype) * 1e3
    nearest = distances.min(dim=1).values.clamp_min(eps)
    return -torch.log(nearest).mean()


@dataclass(frozen=True)
class JEPAV2Config:
    width: int
    horizons: tuple[int, ...] = DEFAULT_HORIZONS
    mask_ratio: float = 0.25
    sensor_dropout: float = 0.1
    free_bits: float = 0.05
    koleo_weight: float = 0.1
    momentum: float = 0.99
    percentile_low: float = 0.05
    percentile_high: float = 0.95
    scale_floor: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.width < 1:
            raise ValueError(f"width must be positive, got {self.width}")
        if not self.horizons or any(h < 1 for h in self.horizons):
            raise ValueError(f"horizons must be positive frame offsets, got {self.horizons}")
        if len(set(self.horizons)) != len(self.horizons):
            raise ValueError(f"horizons must be unique, got {self.horizons}")
        if not 0.0 < self.mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must be in (0, 1), got {self.mask_ratio}")
        if not 0.0 <= self.sensor_dropout < 1.0:
            raise ValueError(f"sensor_dropout must be in [0, 1), got {self.sensor_dropout}")
        if not math.isfinite(self.free_bits) or self.free_bits < 0.0:
            raise ValueError(f"free_bits must be finite and non-negative, got {self.free_bits}")
        if not math.isfinite(self.koleo_weight) or self.koleo_weight < 0.0:
            raise ValueError(f"koleo_weight must be finite and non-negative, got {self.koleo_weight}")
        if not 0.0 <= self.momentum <= 1.0:
            raise ValueError(f"momentum must be in [0, 1], got {self.momentum}")
        if not (0.0 <= self.percentile_low < self.percentile_high <= 1.0):
            raise ValueError(
                f"percentiles must satisfy 0 <= low < high <= 1, got "
                f"({self.percentile_low}, {self.percentile_high})"
            )
        if not math.isfinite(self.scale_floor) or self.scale_floor <= 0.0:
            raise ValueError(f"scale_floor must be finite and positive, got {self.scale_floor}")


class EMATeacherV2(nn.Module):
    """Frozen EMA copy of an encoder; parameters and buffers both tracked."""

    def __init__(self, encoder: nn.Module, momentum: float = 0.99) -> None:
        super().__init__()
        if not callable(getattr(encoder, "forward_features", None)):
            raise TypeError("encoder must implement forward_features(window, cond=None)")
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"momentum must be in [0, 1], got {momentum}")
        self.momentum = momentum
        self.net = deepcopy(encoder).eval()
        for parameter in self.net.parameters():
            parameter.requires_grad_(False)
        self.sync(encoder)

    def _pairs(self, student: nn.Module, kind: str) -> list[tuple[torch.Tensor, torch.Tensor]]:
        teacher_items = list(getattr(self.net, kind)())
        student_items = list(getattr(student, kind)())
        if len(teacher_items) != len(student_items):
            raise ValueError(
                f"EMA teacher/student {kind} count mismatch: {len(teacher_items)} vs "
                f"{len(student_items)}; construct the teacher from this encoder"
            )
        for teacher_item, student_item in zip(teacher_items, student_items):
            if teacher_item.shape != student_item.shape:
                raise ValueError(
                    f"EMA teacher/student {kind} shape mismatch: {tuple(teacher_item.shape)} vs "
                    f"{tuple(student_item.shape)}; construct the teacher from this encoder"
                )
        return teacher_items

    @torch.no_grad()
    def sync(self, student: nn.Module) -> None:
        for teacher_param, student_param in zip(
            self._pairs(student, "parameters"), list(student.parameters())
        ):
            teacher_param.copy_(student_param)
        for teacher_buffer, student_buffer in zip(
            self._pairs(student, "buffers"), list(student.buffers())
        ):
            teacher_buffer.copy_(student_buffer)

    @torch.no_grad()
    def update(self, student: nn.Module) -> None:
        decay = self.momentum
        for teacher_param, student_param in zip(
            self._pairs(student, "parameters"), list(student.parameters())
        ):
            teacher_param.mul_(decay).add_(student_param, alpha=1.0 - decay)
        # Buffers are state, not moving averages (normaliser stats, anneal): keep
        # them in lockstep or a running-stats encoder silently drifts from EMA.
        for teacher_buffer, student_buffer in zip(
            self._pairs(student, "buffers"), list(student.buffers())
        ):
            teacher_buffer.copy_(student_buffer)

    @torch.no_grad()
    def features(self, window: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        return _forward_features(self.net, window, cond).detach()


def _forward_features(encoder: nn.Module, window: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
    if cond is None:
        features = encoder.forward_features(window)
    else:
        features = encoder.forward_features(window, cond)
    batch, time, _ = window.shape
    if features.ndim != 3 or features.shape[:2] != (batch, time):
        raise ValueError(
            f"encoder.forward_features must return (batch, time, width)=({batch}, {time}, W), "
            f"got {tuple(features.shape)}"
        )
    return features


class JEPAV2Predictor(nn.Module):
    """One linear probe: latent at position t -> all horizons at position t+h."""

    def __init__(self, width: int, horizons: tuple[int, ...], seed: int) -> None:
        super().__init__()
        self.horizons = tuple(horizons)
        self.probe = nn.Linear(width, len(self.horizons) * width)
        generator = torch.Generator().manual_seed(int(seed))
        bound = width**-0.5
        nn.init.uniform_(self.probe.weight, -bound, bound, generator=generator)
        nn.init.zeros_(self.probe.bias)

    def forward(self, latent: torch.Tensor) -> dict[int, torch.Tensor]:
        batch, time, width = latent.shape
        output = self.probe(latent).view(batch, time, len(self.horizons), width)
        return dict(zip(self.horizons, output.unbind(dim=2)))


@dataclass
class JEPAV2Losses:
    jepa: torch.Tensor  # mean of the free-bits-floored per-horizon terms
    koleo: torch.Tensor
    total: torch.Tensor
    per_horizon: dict[int, torch.Tensor]  # floored, one scalar per horizon
    predictions: dict[int, torch.Tensor]  # (B, T-h, width), for shape gates


class JEPAV2(nn.Module):
    """JEPA v2 objective; step the student and predictor, then call ``update_teacher``."""

    def __init__(self, encoder: nn.Module, cfg: JEPAV2Config) -> None:
        super().__init__()
        if not isinstance(cfg, JEPAV2Config):
            raise TypeError(f"cfg must be a JEPAV2Config, got {type(cfg).__name__}")
        self.cfg = cfg
        self.teacher = EMATeacherV2(encoder, cfg.momentum)
        self.predictor = JEPAV2Predictor(cfg.width, cfg.horizons, cfg.seed)
        # Two disjoint named streams: seed+0 initialises the probe, seed+1 draws
        # masks/dropout. Neither touches torch's global RNG.
        self._rng = torch.Generator().manual_seed(int(cfg.seed) + 1)

    def reset_seed(self, seed: int | None = None) -> None:
        """Rewind the augmentation stream; probe weights are left alone."""
        value = self.cfg.seed if seed is None else seed
        self._rng = torch.Generator().manual_seed(int(value) + 1)

    def _augment(self, window: torch.Tensor) -> torch.Tensor:
        batch, time, features = window.shape
        span = max(1, int(time * self.cfg.mask_ratio))
        starts = torch.randint(0, time - span + 1, (batch,), generator=self._rng)
        positions = torch.arange(time).unsqueeze(0)
        mask = (positions >= starts[:, None]) & (positions < starts[:, None] + span)
        dark = torch.rand((batch, features), generator=self._rng) < self.cfg.sensor_dropout
        keep_time = (~mask).unsqueeze(-1).to(window.dtype).to(window.device)
        keep_sensor = (~dark).unsqueeze(1).to(window.dtype).to(window.device)
        return window * keep_time * keep_sensor

    def _validate(self, window: torch.Tensor, cond: torch.Tensor | None) -> None:
        if not isinstance(window, torch.Tensor):
            raise TypeError(f"window must be a tensor, got {type(window).__name__}")
        if window.ndim != 3:
            raise ValueError(f"window must be (batch, time, features), got {tuple(window.shape)}")
        batch, time, features = window.shape
        longest = max(self.cfg.horizons)
        if batch == 0 or features == 0 or time <= longest:
            raise ValueError(
                f"window {tuple(window.shape)} is malformed for horizons {self.cfg.horizons}: "
                f"need batch>0, features>0 and time>{longest}"
            )
        _require_finite("window", window)
        if cond is not None:
            if not isinstance(cond, torch.Tensor) or cond.ndim != 2 or cond.shape[0] != batch:
                raise ValueError(
                    f"cond must be (batch, cond_features) with batch={batch}, got "
                    f"{tuple(cond.shape) if isinstance(cond, torch.Tensor) else type(cond).__name__}"
                )
            _require_finite("cond", cond)

    def loss(
        self,
        student: nn.Module,
        window: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> JEPAV2Losses:
        """Compute the objective; call ``update_teacher(student)`` after the optimiser step."""
        self._validate(window, cond)
        _batch, time, _ = window.shape
        student_latent = _forward_features(student, self._augment(window), cond)
        if student_latent.shape[-1] != self.cfg.width:
            raise ValueError(
                f"encoder width {student_latent.shape[-1]} != cfg.width {self.cfg.width}; "
                "set JEPAV2Config.width to the encoder's forward_features width"
            )
        with torch.no_grad():
            target = normalize_latents(
                self.teacher.features(window, cond),
                low=self.cfg.percentile_low,
                high=self.cfg.percentile_high,
                floor=self.cfg.scale_floor,
            )
        if target.shape[-1] != self.cfg.width:
            raise ValueError(
                f"teacher width {target.shape[-1]} != cfg.width {self.cfg.width}; "
                "construct the objective from the same encoder passed to loss"
            )

        predictions = self.predictor(student_latent)
        per_horizon: dict[int, torch.Tensor] = {}
        for horizon in self.cfg.horizons:
            predicted = predictions[horizon][:, : time - horizon, :]
            desired = target[:, horizon:, :]
            if predicted.shape != desired.shape:
                raise ValueError(
                    f"horizon {horizon}: prediction {tuple(predicted.shape)} != "
                    f"target {tuple(desired.shape)}"
                )
            per_horizon[horizon] = F.mse_loss(predicted, desired).clamp_min(self.cfg.free_bits)

        jepa = torch.stack(list(per_horizon.values())).mean()
        koleo = koleo_loss(student_latent[:, -1, :])
        return JEPAV2Losses(jepa, koleo, jepa + self.cfg.koleo_weight * koleo, per_horizon, predictions)

    @torch.no_grad()
    def update_teacher(self, student: nn.Module) -> None:
        self.teacher.update(student)
