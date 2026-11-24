# SPDX-License-Identifier: GPL-3.0-or-later
"""Privileged-teacher impact objective (AIOBJ, simulation-only training term).

The teacher's command is a two-channel fin acceleration demand in m/s^2,
``[roll, pitch]``. The classical baseline is unchanged; this module only adds a
measurement term to the privileged stage's loss. It reads privileged velocity
from the v3 truth sidecar (slots 3..5, m/s; ``sim/core/src/trajectory.rs:94-121``)
and two *measured* sensor features (``a_lat_resid`` and ``drag_est``, both m/s^2,
SPEC_SMART.md). Nothing here changes physics, the command envelope, a gate, or
any flight claim.

The loss is the mean over the batch of two dimensionless squared terms::

    lateral = ((a_lat_resid - u_lat) / g0)^2
    gamma   = atan2(vz_next, hypot(vx_next, vy_next))          # radians, SI dimensionless
    flight  = (drag_est / g0) * gamma^2
    L       = mean(lateral + flight)

``u_lat = command[..., 0]``; ``u_pitch = command[..., 1]``. World-frame command
mapping is ``[ax, ay] = [-u_pitch, u_lat]`` (sim/core/src/optimizer.rs:127-137),
so one frame at the 50 Hz validation rate gives
``vx_next = vx - u_pitch*dt``, ``vy_next = vy + u_lat*dt``, ``vz_next = vz``.
After FIXPUMP the normal force does no work (post-FIXPUMP dynamics), so lateral
steering is spent through drag: the flight-path angle is weighted by the
*measured* drag acceleration, not by a tuned coefficient.

Units: ``a_lat_resid``, ``drag_est``, ``u_lat``, ``u_pitch`` in m/s^2; ``g0`` in
m/s^2; velocity in m/s; ``dt`` in s; angles in radians (dimensionless in SI).
``g0`` and ``dt`` are physical/measurement constants, not loss weights; the two
squared terms are added with coefficient **1** each. No weight was fitted to a
closed-loop target, and no threshold or envelope is touched here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

G0_MPS2 = 9.80665  # standard gravity, m/s^2 (physical constant, not a weight)
FRAME_DT_S = 0.02  # validation v3 frame period at 50 Hz (header.frame_rate_hz)

# Smart-window physics columns (data.py / SPEC_SMART.md), all in the last frame.
A_LAT_RESID_COL = 11  # m/s^2
DRAG_EST_COL = 12  # m/s^2
SMART_FEATURES = 14

# v3 privileged truth slots, pinned to sim/core/src/trajectory.rs:94-121 and
# as_truth_array: vel_x/vel_y/vel_z are slots 3, 4, 5; slots 10..13 are omega/mass.
# The old 11..14 offset read body rate/mass as velocity (off-by-slot defect).
TRUTH_VEL_START = 3
TRUTH_VEL_END = 6  # exclusive: vx, vy, vz in m/s
TRUTH_DIM = 22

COMMAND_CHANNELS = 2
IMPACT_FORMULA = (
    "L = mean[((a_lat_resid - u_lat)/g0)^2 + (drag_est/g0)*gamma(u)^2]; "
    "gamma = atan2(vz_next, hypot(vx_next, vy_next)), "
    "vx_next = vx - u_pitch*dt, vy_next = vy + u_lat*dt, vz_next = vz"
)

__all__ = [
    "DRAG_EST_COL",
    "FRAME_DT_S",
    "G0_MPS2",
    "IMPACT_FORMULA",
    "TRUTH_DIM",
    "TRUTH_VEL_END",
    "TRUTH_VEL_START",
    "ImpactObjective",
    "ImpactObjectiveConfig",
    "ImpactTerms",
    "flight_path_angle_rad",
]


@dataclass(frozen=True)
class ImpactObjectiveConfig:
    """Physical constants of the objective; both fields carry units."""

    gravity_mps2: float = G0_MPS2
    frame_dt_s: float = FRAME_DT_S

    def __post_init__(self) -> None:
        if not math.isfinite(self.gravity_mps2) or self.gravity_mps2 <= 0.0:
            raise ValueError(f"gravity_mps2 must be finite and positive, got {self.gravity_mps2!r}")
        if not math.isfinite(self.frame_dt_s) or self.frame_dt_s <= 0.0:
            raise ValueError(f"frame_dt_s must be finite and positive, got {self.frame_dt_s!r}")


@dataclass(frozen=True)
class ImpactTerms:
    """Scalar loss plus its two dimensionless components for the metrics log."""

    loss: torch.Tensor
    lateral: torch.Tensor
    flight_path: torch.Tensor
    gamma_rad: torch.Tensor


def flight_path_angle_rad(velocity: torch.Tensor) -> torch.Tensor:
    """Flight-path angle ``gamma = atan2(vz, hypot(vx, vy))`` in radians.

    ``velocity`` is ``[..., 3]`` in m/s. A zero horizontal speed makes gamma
    undefined, so it is refused loudly instead of being regularised away.
    """
    if velocity.ndim < 1 or velocity.shape[-1] != 3:
        raise ValueError(f"velocity must have a 3-axis last dimension, got {tuple(velocity.shape)}")
    if not torch.isfinite(velocity).all():
        raise ValueError("privileged velocity must be finite (m/s)")
    vx, vy, vz = velocity.unbind(dim=-1)
    horizontal = torch.hypot(vx, vy)
    if not bool((horizontal > 0.0).all()):
        raise ValueError("horizontal speed is zero; flight-path angle is undefined")
    return torch.atan2(vz, horizontal)


class ImpactObjective:
    """Differentiable privileged term; no parameters, no learned weights."""

    def __init__(self, cfg: ImpactObjectiveConfig | None = None) -> None:
        self.cfg = cfg or ImpactObjectiveConfig()

    def loss(
        self,
        command: torch.Tensor,
        window: torch.Tensor,
        privileged: torch.Tensor,
    ) -> ImpactTerms:
        """Return the impact loss for one command batch.

        ``command`` is ``[B, 2]`` m/s^2, ``window`` is ``[B, T, 14]`` sensor
        frames, ``privileged`` is ``[B, 22]`` truth slots. Gradients flow to
        ``command`` only; the truth and sensor columns are constants.
        """
        if command.ndim != 2 or command.shape[-1] != COMMAND_CHANNELS:
            raise ValueError(f"command must have shape [B, 2], got {tuple(command.shape)}")
        if (
            window.ndim != 3
            or window.shape[-1] != SMART_FEATURES
            or window.shape[0] != command.shape[0]
        ):
            raise ValueError(
                f"window must have shape [B, T, {SMART_FEATURES}], got {tuple(window.shape)}"
            )
        if privileged.ndim != 2 or privileged.shape != (command.shape[0], TRUTH_DIM):
            raise ValueError(
                f"privileged truth must have shape [B, {TRUTH_DIM}], got {tuple(privileged.shape)}"
            )
        if not torch.isfinite(command).all() or not torch.isfinite(window).all():
            raise ValueError("command and sensor window must be finite")
        # Validate the velocity before any arithmetic so a zero-speed row cannot
        # silently produce an atan2(0, 0) gradient.
        flight_path_angle_rad(privileged[..., TRUTH_VEL_START:TRUTH_VEL_END])

        last = window[:, -1, :]
        a_lat = last[..., A_LAT_RESID_COL]  # m/s^2
        drag = last[..., DRAG_EST_COL]  # m/s^2
        u_lat, u_pitch = command.unbind(dim=-1)
        velocity = privileged[..., TRUTH_VEL_START:TRUTH_VEL_END]
        dt = self.cfg.frame_dt_s
        vx_next = velocity[..., 0] - u_pitch * dt
        vy_next = velocity[..., 1] + u_lat * dt
        vz_next = velocity[..., 2]  # normal force does no work after FIXPUMP
        horizontal_next = torch.hypot(vx_next, vy_next)
        if not bool((horizontal_next > 0.0).all()):
            raise ValueError("command drives horizontal speed to zero; gamma is undefined")
        gamma = torch.atan2(vz_next, horizontal_next)  # rad, SI dimensionless

        lateral_ratio = (a_lat - u_lat) / self.cfg.gravity_mps2
        drag_ratio = drag / self.cfg.gravity_mps2
        if not bool((drag_ratio >= 0.0).all()):
            raise ValueError("drag_est must be non-negative m/s^2 (cd0*q_dyn*area/mass)")
        lateral = lateral_ratio.square()
        flight_path = drag_ratio * gamma.square()
        return ImpactTerms(
            loss=(lateral + flight_path).mean(),
            lateral=lateral.mean(),
            flight_path=flight_path.mean(),
            gamma_rad=gamma.abs().detach().mean(),
        )
