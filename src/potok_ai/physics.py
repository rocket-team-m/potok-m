"""Physics reference for the residual-to-physics objective (SPEC_ENGINE_IDEAL.md §4).

The network never emits the absolute actuator command. It emits a bounded
correction ``Delta`` on top of a deterministic baseline ``b(x)`` built from the
same onboard features; the command that is flown is ``b(x) + Delta``. Training
supervises the *command*, so the network learns the residual ``target - b(x)``.

Baseline formula, per 14-feature frame (``alpha_deg`` = feature 8, ``beta_deg`` =
feature 9, lateral-acceleration residual ``a_lat`` = feature 11)::

    roll(t)  = clip( Kp * beta(t) / BETA0 + Kd * tanh(a_lat(t)), -1, 1 )
    pitch(t) = clip( Kp * alpha(t) / ALPHA0,                         -1, 1 )

A proportional term on the aerodynamic angles plus a bounded derivative term on
the lateral residual. No integral state is kept: the reference command already
closes the outer loop, and a frame-local baseline keeps training and deployment
identical. A window model evaluates the baseline on the window's final frame --
the same frame its head reads. The constants below are the single source of
truth for both the trainer and the flight controller.
"""

from __future__ import annotations

import torch

KP = 0.4
KD = 0.1
BETA0 = 10.0  # degrees, normalisation for the roll error
ALPHA0 = 10.0  # degrees, normalisation for the pitch error
BOUND = 1.0  # target corrections are tanh-bounded to [-1, 1]

_ALPHA = 8
_BETA = 9
_A_LAT = 11


def physics_baseline(features: torch.Tensor) -> torch.Tensor:
    """Return ``[roll, pitch]`` for a frame or a batch of frames.

    ``features`` has at least 12 columns; the 14-feature smart frame is the
    canonical input. The output shape is ``features.shape[:-1] + (2,)``.
    """
    if features.shape[-1] < 12:
        raise ValueError(f"the physics baseline needs 12 features, got {features.shape[-1]}")
    beta = features[..., _BETA]
    alpha = features[..., _ALPHA]
    a_lat = features[..., _A_LAT]
    roll = (KP * beta / BETA0 + KD * torch.tanh(a_lat)).clamp(-BOUND, BOUND)
    pitch = (KP * alpha / ALPHA0).clamp(-BOUND, BOUND)
    return torch.stack((roll, pitch), dim=-1)


def final_command(
    prediction: torch.Tensor, baseline: torch.Tensor, residual: bool = True
) -> torch.Tensor:
    """Map a network output to the command compared with the reference."""
    return baseline + prediction if residual else prediction
