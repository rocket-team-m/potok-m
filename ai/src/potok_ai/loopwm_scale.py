"""Scale-free views of LoopWM's latent loss and adaptive-exit threshold.

LoopWM's JEPA targets are LayerNorm'd EMA-encoder latents with batch std ~1,
and its exit threshold is defined for increments at that scale. These pure
helpers make the units explicit so both quantities stay meaningful when a
latent batch is multiplied by a common factor. ``loopwm.py`` is unchanged.

GPL-3.0-or-later
"""

from __future__ import annotations

import math

import torch

from .loopwm import LATENT_SCALE_FLOOR

__all__ = ["LATENT_SCALE_FLOOR", "normalized_latent_loss", "scaled_exit_threshold"]


def _latent_scale(latents: torch.Tensor) -> torch.Tensor:
    """Std of a latent batch, floored so a constant batch cannot divide by zero."""
    if latents.ndim < 2:
        raise ValueError("latents must have a token and a feature dimension")
    return latents.detach().std(unbiased=False).clamp_min(LATENT_SCALE_FLOOR)


def normalized_latent_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """MSE of future-latent predictions, measured in the targets' own std units.

    Both sides are divided by the detached target std, as in
    ``LoopWM.latent_prediction_loss``: a constant predictor scores ~1 and
    rescaling every latent by a common factor leaves the loss unchanged.
    Pass ``scale`` when several prediction heads share one normalization
    (LoopWM uses the std of all horizons' targets for every term).
    """
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    target = target.detach()
    units = _latent_scale(target if scale is None else scale)
    return ((prediction / units) - (target / units)).square().mean()


def scaled_exit_threshold(threshold: float, latent: torch.Tensor) -> torch.Tensor:
    """Convert a reference-scale exit threshold into a latent batch's raw units.

    ``LoopWM.refine`` compares the mean per-token increment norm with this
    number. Multiplying by the latent std preserves that decision when every
    latent -- and therefore every increment -- is rescaled by a common factor.
    """
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and non-negative")
    return latent.new_tensor(threshold) * _latent_scale(latent)
