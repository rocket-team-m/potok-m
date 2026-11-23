"""JEPA pretraining with Fast-JEPA's one-shot target.

Fast-JEPA (arXiv:2611.25927) predicts the outcome of a complete candidate in
one pass instead of recursively rolling out a latent dynamics model. For a
50-frame sensor window this means a single linear probe maps the masked
student latents to the teacher latents of the whole (shifted) window -- no
iterative rollout, no candidate search, no planning machinery. KoLeo keeps the
latents from collapsing.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .model import PotokModel


class EMATeacher:
    """Exponential moving average of a student model, used as a frozen teacher."""

    def __init__(self, student: nn.Module, momentum: float = 0.999) -> None:
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("momentum must be in [0, 1]")
        self.momentum = momentum
        self.net = deepcopy(student).eval()
        for parameter in self.net.parameters():
            parameter.requires_grad_(False)
        self.sync(student)

    @torch.no_grad()
    def sync(self, student: nn.Module) -> None:
        for teacher_param, student_param in zip(self.net.parameters(), student.parameters()):
            teacher_param.copy_(student_param)
        for teacher_buffer, student_buffer in zip(self.net.buffers(), student.buffers()):
            teacher_buffer.copy_(student_buffer)

    @torch.no_grad()
    def update(self, student: nn.Module) -> None:
        decay = self.momentum
        for teacher_param, student_param in zip(self.net.parameters(), student.parameters()):
            teacher_param.mul_(decay).add_(student_param, alpha=1.0 - decay)

    @torch.no_grad()
    def __call__(self, window: torch.Tensor) -> torch.Tensor:
        return self.net.forward_features(window)


def mask_windows(window: torch.Tensor, mask_ratio: float = 0.25) -> torch.Tensor:
    """Boolean (B, T) mask selecting one contiguous span per sample."""
    batch, time = window.shape[:2]
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError("mask_ratio must be in (0, 1)")
    span = max(1, int(time * mask_ratio))
    starts = torch.rand(batch, device=window.device) * (time - span)
    positions = torch.arange(time, device=window.device).unsqueeze(0)
    return (positions >= starts.unsqueeze(1)) & (positions < starts.unsqueeze(1) + span)


class JEPAPredictor(nn.Module):
    """One linear probe: masked latents -> the complete target sequence, in one pass."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.probe = nn.Linear(width, width)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.probe(latent)


@dataclass
class JEPALosses:
    jepa: torch.Tensor
    koleo: torch.Tensor
    total: torch.Tensor


def koleo_loss(features: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """KoLeo: ``-mean log(min pairwise distance)`` over normalised features."""
    if features.shape[0] < 2:
        return features.sum() * 0.0
    z = F.normalize(features, dim=-1)
    distances = torch.cdist(z, z)
    distances = distances + torch.eye(z.shape[0], device=z.device) * 1e3
    nearest = distances.min(dim=1).values.clamp_min(eps)
    return -torch.log(nearest).mean()


def jepa_loss(
    student: PotokModel,
    teacher: EMATeacher,
    window: torch.Tensor,
    predictor: JEPAPredictor,
    mask_ratio: float = 0.25,
    koleo_weight: float = 0.1,
    shift: int = 3,
) -> JEPALosses:
    """Predict the teacher's full shifted latent sequence in a single forward."""
    if not 0 <= shift < window.shape[1]:
        raise ValueError("shift must be smaller than the window length")
    mask = mask_windows(window, mask_ratio)
    masked = torch.where(mask.unsqueeze(-1), torch.zeros_like(window), window)
    student_latent = student.forward_features(masked)
    with torch.no_grad():
        teacher_latent = teacher(window)

    source = student_latent[:, : window.shape[1] - shift, :]
    target = teacher_latent[:, shift:, :]
    jepa = F.mse_loss(predictor(source), target)
    koleo = koleo_loss(student_latent[:, -1, :])
    return JEPALosses(jepa, koleo, jepa + koleo_weight * koleo)
