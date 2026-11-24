# SPDX-License-Identifier: GPL-3.0-or-later
"""Privileged radial-braking reference for the teacher stages (BRK01).

A *reference*, not a learned model and not a flight controller. It reads the
true inertial state from the v3 truth sidecar (positions at slots 0..3,
velocities at slots 3..6, ``sim/core/src/trajectory.rs:94-121``) and returns the
fin acceleration demand ``[u_lat, u_pitch]`` in m/s^2, the same two channels the
dataset's ``g_accel`` label carries (``sim/core/src/optimizer.rs:127-137``).

Law. With radial offset ``r = hypot(x, y)`` and outward radial speed
``v_r = (x*vx + y*vy)/r``, the inward demand is the clipped sum of a velocity
term and a position term::

    a_r = clip( -(k_v * v_r + k_p * r), -a_cap, a_cap )

Every coefficient follows from the command cap ``a_cap`` and a declared
stopping-time horizon ``T`` by the stopping-distance identity ``d = v^2/(2a)``::

    v_ref = a_cap * T          # speed the cap can null in T seconds
    d_ref = v_ref^2/(2*a_cap)  # stopping distance at that speed
    k_v   = a_cap / v_ref = 1/T
    k_p   = a_cap / d_ref = 2/T^2

No coefficient is fitted to a trajectory, a gate or a closed-loop target. The
phase labels describe which term governs (BRAKE above ``v_ref`` outward,
CORRECT otherwise), but they do not change the arithmetic: clipping the same
sum already saturates at the cap.

The AI interlock is explicit and one-way. ``FlightPhase`` names the organizer's
phases: Pad, Boost and MotorArmed are BANNED (zero demand, reason ``boost_gate``);
Coast and Descent are the only ACTIVE phases, where the radial law runs. A
caller-supplied ``ai_allowed == False`` is a second, independent veto. The
predicate is a whitelist of active phases, never its inverse, so a mislabelled
clock cannot quietly turn a banned frame into a target. No command is invented
from an invalid state: non-finite, zero-radius and out-of-range rows return zero
with their own reason. All are counted by :meth:`BrakingTeacher.reason_counts`
so the training log shows which arm ran.

Scope: simulation-only training target. The student never sees privileged truth,
no flight gate or physics constant is changed here, and no deployment-lateral
improvement is claimed by this module alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

# v3 privileged slot layout, pinned to sim/core/src/trajectory.rs:94-121.
POS_SLICE = slice(0, 3)
VEL_SLICE = slice(3, 6)
PRIV_END = 22
V3_STRIDE = 26

# Hardware/measurement constants with their sources, not loss weights.
COMMAND_CAP_MPS2 = 10.0  # sim/core/src/control.rs:34 (MAX_COMMAND_MPS2)
FRAME_DT_S = 0.02  # sim/config/default.toml:9 (50 Hz validation frames)
BOOST_END_S = 9.95  # sim/core/tests/sim_tests.rs:295 (9.9 <= burn_end <= 10.0)

__all__ = [
    "ACTIVE_PHASES",
    "BOOST_END_S",
    "COMMAND_CAP_MPS2",
    "FORBIDDEN_PHASES",
    "FRAME_DT_S",
    "POS_SLICE",
    "PRIV_END",
    "V3_STRIDE",
    "VEL_SLICE",
    "BrakePhase",
    "BrakeReason",
    "BrakingConfig",
    "BrakingDecision",
    "BrakingTeacher",
    "FlightPhase",
    "TrueState",
    "flight_phase",
]


class BrakePhase(str, Enum):
    """Which term of the radial law governs a row."""

    BOOST = "boost"
    BRAKE = "brake"
    CORRECT = "correct"
    SETTLED = "settled"
    OUT_OF_RANGE = "out_of_range"


class FlightPhase(str, Enum):
    """Organizer flight phases. The AI ban is a whitelist, not a blacklist.

    Pad, Boost and MotorArmed precede burnout and are forbidden. The v3 truth
    row carries no motor-arm bit, so pre-burnout ticks are reported as ``BOOST``
    (motor armed throughout); a caller with an explicit arm signal can label
    those rows ``MOTOR_ARMED`` and the gate treats them identically. Coast and
    Descent are the only phases in which a reference command is produced.
    ``UNKNOWN`` is the DAgger interlock path: no clock is carried by a wire row,
    so the recorded health bit (``ai_allowed``) is the sole authority there.
    """

    PAD = "pad"
    BOOST = "boost"
    MOTOR_ARMED = "motor_armed"
    COAST = "coast"
    DESCENT = "descent"
    UNKNOWN = "unknown"


#: The organizer ban, written out so it cannot drift from the whitelist.
FORBIDDEN_PHASES = frozenset(
    {FlightPhase.PAD, FlightPhase.BOOST, FlightPhase.MOTOR_ARMED, FlightPhase.UNKNOWN}
)
ACTIVE_PHASES = frozenset({FlightPhase.COAST, FlightPhase.DESCENT})
if FORBIDDEN_PHASES & ACTIVE_PHASES or FORBIDDEN_PHASES | ACTIVE_PHASES != set(FlightPhase):
    raise AssertionError("flight phases must split exactly into banned and active")


def flight_phase(time_s: float, cfg: BrakingConfig, *, vz: float | None = None) -> FlightPhase:
    """Transcribe ``sim/core/src/dynamics.rs:735-755`` into a training label.

    Only the burnout boundary is used for the ban: ``t <= 0`` is Pad,
    ``0 < t <= boost_end_s`` is Boost (motor armed), and everything after
    burnout is Coast or Descent by the sign of vertical speed. No gate limit,
    threshold or physics constant is changed here.
    """
    if time_s <= 0.0:
        return FlightPhase.PAD
    if time_s <= cfg.boost_end_s:
        return FlightPhase.BOOST
    return FlightPhase.DESCENT if vz is not None and vz < 0.0 else FlightPhase.COAST


class BrakeReason(str, Enum):
    """One reason per row; the not-in-range cases never produce a demand."""

    OK = "ok"
    BOOST_GATE = "boost_gate"
    NON_FINITE = "non_finite"
    RADIUS_ZERO = "radius_zero"
    OUT_OF_RANGE = "out_of_range"
    CAP_CLAMPED = "cap_clamped"
    SETTLED = "settled"


@dataclass(frozen=True)
class BrakingConfig:
    """Typed constants. ``horizon_s`` is a declared design knob, not a fit."""

    cap_mps2: float = COMMAND_CAP_MPS2
    horizon_s: float = 2.0
    boost_end_s: float = BOOST_END_S
    frame_dt_s: float = FRAME_DT_S
    max_radius_m: float = 10_000.0
    max_speed_mps: float = 1_000.0
    deadband_m: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "cap_mps2",
            "horizon_s",
            "boost_end_s",
            "frame_dt_s",
            "max_radius_m",
            "max_speed_mps",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive, got {value!r}")
        if not math.isfinite(self.deadband_m) or self.deadband_m < 0.0:
            raise ValueError(f"deadband_m must be finite and non-negative, got {self.deadband_m!r}")

    @property
    def ref_speed_mps(self) -> float:
        """Speed the cap can null in one horizon: ``v = a*T``."""
        return self.cap_mps2 * self.horizon_s

    @property
    def stop_distance_m(self) -> float:
        """Stopping distance at the reference speed: ``v^2/(2a)``."""
        return self.ref_speed_mps**2 / (2.0 * self.cap_mps2)

    @property
    def velocity_gain_inv_s(self) -> float:
        """``a_cap / v_ref = 1/T``."""
        return self.cap_mps2 / self.ref_speed_mps

    @property
    def position_gain_inv_s2(self) -> float:
        """``a_cap / d_ref = 2/T^2``."""
        return self.cap_mps2 / self.stop_distance_m


@dataclass(frozen=True)
class TrueState:
    """True inertial state, batchable; seconds and metres, no label fields."""

    position: np.ndarray  # [..., 3] m: x, y, z
    velocity: np.ndarray  # [..., 3] m/s: vx, vy, vz
    time_s: np.ndarray | float | None = None  # optional [batch] flight time

    def __post_init__(self) -> None:
        position = np.asarray(self.position, dtype=np.float64)
        velocity = np.asarray(self.velocity, dtype=np.float64)
        if position.ndim < 1 or position.shape[-1] != 3:
            raise ValueError(f"position must have a 3-axis last dimension, got {position.shape}")
        if velocity.shape != position.shape:
            raise ValueError(f"velocity {velocity.shape} must match position {position.shape}")
        if position.ndim == 1:
            position, velocity = position[None, :], velocity[None, :]
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "velocity", velocity)
        if self.time_s is not None:
            time_s = np.asarray(self.time_s, dtype=np.float64)
            if time_s.ndim > 1 or (time_s.ndim == 1 and time_s.shape != (len(position),)):
                raise ValueError(f"time_s {time_s.shape} does not match state {position.shape}")
            object.__setattr__(self, "time_s", time_s)

    @classmethod
    def from_privileged(cls, snapshot: Any) -> TrueState:
        """Build from a v3 privileged row (22 slots) or a full 26-slot row.

        Only slots 0..6 are read; the four label slots of a 26-row are never an
        input to the demand. A wrong stride is refused loudly.
        """
        array = np.asarray(snapshot, dtype=np.float64)
        if array.ndim < 1 or array.shape[-1] not in (PRIV_END, V3_STRIDE):
            raise ValueError(
                f"a v3 privileged snapshot needs {PRIV_END} or {V3_STRIDE} slots, "
                f"got {array.shape}; regenerate with write_trajectory_v3"
            )
        return cls(position=array[..., POS_SLICE], velocity=array[..., VEL_SLICE])

    @property
    def batch(self) -> int:
        """Number of rows (one for an unbatched state)."""
        return int(self.position.shape[0])


@dataclass(frozen=True)
class BrakingDecision:
    """One fin demand per row plus its phase, reason and measured radial state."""

    command: np.ndarray  # [batch, 2] m/s^2, [u_lat, u_pitch]
    radial_demand: np.ndarray  # [batch] signed inward a_r, m/s^2
    radius: np.ndarray  # [batch] m
    radial_velocity: np.ndarray  # [batch] outward m/s
    phases: tuple[BrakePhase, ...]
    reasons: tuple[BrakeReason, ...]
    flight_phases: tuple[FlightPhase, ...]


class BrakingTeacher:
    """Stateless radial reference with a counted reason histogram."""

    def __init__(self, cfg: BrakingConfig | None = None) -> None:
        self.cfg = cfg or BrakingConfig()
        self.reset_reason_counts()

    def reset_reason_counts(self) -> None:
        self._reason_counts = {reason.value: 0 for reason in BrakeReason}

    def reason_counts(self) -> dict[str, int]:
        return dict(self._reason_counts)

    def demand(
        self,
        state: TrueState,
        ai_allowed: np.ndarray | None = None,
        *,
        flight_phases: tuple[FlightPhase, ...] | None = None,
    ) -> BrakingDecision:
        """Return the bounded fin demand for each row of ``state``.

        ``ai_allowed`` is the caller's interlock (e.g. the recorded health flag).
        A supplied ``state.time_s`` labels each row with an explicit
        :class:`FlightPhase`: only Coast and Descent receive a braking command,
        while Pad, Boost and MotorArmed return zero. An explicit ``flight_phases``
        (e.g. an organizer phase field) overrides the clock-derived label.
        Without either (the DAgger wire-row path) the phase is ``UNKNOWN`` and
        only the caller's interlock decides, so that path is never silently
        upgraded to active.
        """
        if not isinstance(state, TrueState):
            raise TypeError(f"state must be a TrueState, got {state!r}")
        position, velocity = state.position, state.velocity
        if position.ndim != 2 or position.shape != (position.shape[0], 3):
            raise ValueError(f"demand expects a [batch, 3] state, got {position.shape}")
        n = position.shape[0]
        if ai_allowed is not None:
            allowed = np.asarray(ai_allowed, dtype=bool)
            if allowed.shape != (n,):
                raise ValueError(f"ai_allowed {allowed.shape} must have shape ({n},)")
        else:
            allowed = np.ones(n, dtype=bool)
        if flight_phases is not None:
            flight = tuple(flight_phases)
            if len(flight) != n or not all(isinstance(p, FlightPhase) for p in flight):
                raise ValueError(f"flight_phases must be {n} FlightPhase values")
        elif state.time_s is not None:
            times = np.asarray(state.time_s, dtype=np.float64).reshape(n)
            flight = tuple(
                flight_phase(float(t), self.cfg, vz=float(velocity[i, 2]))
                for i, t in enumerate(times)
            )
        else:
            flight = (FlightPhase.UNKNOWN,) * n
        if flight_phases is not None or state.time_s is not None:
            # Whitelist: ACTIVE only. Never gate on the complement.
            allowed = allowed & np.asarray(
                [phase in ACTIVE_PHASES for phase in flight], dtype=bool
            )

        x, y = position[:, 0], position[:, 1]
        vx, vy = velocity[:, 0], velocity[:, 1]
        radius = np.hypot(x, y)
        radial_velocity = np.where(
            radius > 0.0, (x * vx + y * vy) / np.maximum(radius, 1e-12), 0.0
        )
        speed = np.linalg.norm(velocity, axis=-1)

        finite = np.isfinite(position).all(axis=-1) & np.isfinite(velocity).all(axis=-1)
        in_range = (
            finite
            & (radius <= self.cfg.max_radius_m)
            & (speed <= self.cfg.max_speed_mps)
            & (radius > 1e-6)
        )

        raw = -(
            self.cfg.velocity_gain_inv_s * radial_velocity
            + self.cfg.position_gain_inv_s2 * radius
        )
        radial_demand = np.clip(raw, -self.cfg.cap_mps2, self.cfg.cap_mps2)
        clipped = np.abs(raw) > self.cfg.cap_mps2

        ux = np.divide(x, radius, out=np.zeros(n), where=radius > 1e-6)
        uy = np.divide(y, radius, out=np.zeros(n), where=radius > 1e-6)
        world_x = radial_demand * ux
        world_y = radial_demand * uy
        # [ax, ay] = [-u_pitch, u_lat]  (sim/core/src/optimizer.rs:127-137)
        command = np.stack((world_y, -world_x), axis=-1)
        command = np.clip(command, -self.cfg.cap_mps2, self.cfg.cap_mps2)
        clipped |= np.abs(command).max(axis=-1) > self.cfg.cap_mps2 - 1e-12

        phases: list[BrakePhase] = []
        reasons: list[BrakeReason] = []
        for i in range(n):
            if not finite[i]:
                phase, reason = BrakePhase.OUT_OF_RANGE, BrakeReason.NON_FINITE
            elif not allowed[i]:
                phase, reason = BrakePhase.BOOST, BrakeReason.BOOST_GATE
            elif not in_range[i]:
                if radius[i] <= 1e-6:
                    phase, reason = BrakePhase.OUT_OF_RANGE, BrakeReason.RADIUS_ZERO
                else:
                    phase, reason = BrakePhase.OUT_OF_RANGE, BrakeReason.OUT_OF_RANGE
            elif radius[i] <= self.cfg.deadband_m and abs(radial_velocity[i]) <= 1e-6:
                phase, reason = BrakePhase.SETTLED, BrakeReason.SETTLED
            else:
                phase = (
                    BrakePhase.BRAKE
                    if radial_velocity[i] > self.cfg.ref_speed_mps
                    else BrakePhase.CORRECT
                )
                reason = BrakeReason.CAP_CLAMPED if clipped[i] else BrakeReason.OK
            if phase in (BrakePhase.BOOST, BrakePhase.OUT_OF_RANGE, BrakePhase.SETTLED):
                command[i] = 0.0
                radial_demand[i] = 0.0
            phases.append(phase)
            reasons.append(reason)
            self._reason_counts[reason.value] += 1
        if len(phases) != n or len(reasons) != n:
            raise AssertionError("one phase and one reason per state row are required")
        if len(flight) != n:
            raise AssertionError("one flight phase per state row is required")
        return BrakingDecision(
            command, radial_demand, radius, radial_velocity, tuple(phases), tuple(reasons), flight
        )
