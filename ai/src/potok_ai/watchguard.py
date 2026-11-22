# SPDX-License-Identifier: GPL-3.0-or-later
"""Pure AI output guard for the stabilisation assistant.

``guard(command, confidence, state, config)`` is a total, deterministic function
of its inputs: no clock, no global state, no mutation. Every rejection emits an
all-zero command and advances the miss/latch state; the accepted command becomes
the next rate-of-change reference. Checks run in a fixed order so the reason is
deterministic: latched -> non-finite -> bounds -> confidence -> rate.

Three consecutive misses latch the assistant off. The latch is sticky: a valid
sample after it does not clear it, only an explicit :func:`reset` does. That is
the flight rule "three misses and the AI is off until end of flight".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np


class GuardReason(str, Enum):
    """Why the guard accepted or rejected a sample."""

    OK = "ok"
    NON_FINITE = "non_finite"
    OUT_OF_BOUNDS = "out_of_bounds"
    LOW_CONFIDENCE = "low_confidence"
    RATE_LIMIT = "rate_limit"
    LATCHED = "latched"


@dataclass(frozen=True)
class GuardConfig:
    """Per-channel command bounds and per-tick rate limits.

    ``bounds[i] = (low, high)`` and ``max_rate[i]`` are in the command's own
    units, one entry per command channel. Bounds must be finite with
    ``low < high``; rate limits finite and non-negative.
    """

    bounds: tuple[tuple[float, float], ...]
    max_rate: tuple[float, ...]
    min_confidence: float = 0.5
    max_misses: int = 3

    def __post_init__(self) -> None:
        if not self.bounds or len(self.bounds) != len(self.max_rate):
            raise ValueError("bounds and max_rate need one entry per channel")
        for low, high in self.bounds:
            if not (math.isfinite(low) and math.isfinite(high) and low < high):
                raise ValueError(f"bounds need finite low < high, got {(low, high)!r}")
        for rate in self.max_rate:
            if not math.isfinite(rate) or rate < 0.0:
                raise ValueError(f"max_rate must be finite and non-negative, got {rate!r}")
        if not math.isfinite(self.min_confidence) or not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError(f"min_confidence must be in [0, 1], got {self.min_confidence!r}")
        if self.max_misses < 1:
            raise ValueError(f"max_misses must be >= 1, got {self.max_misses!r}")


@dataclass(frozen=True)
class GuardState:
    """Immutable carry-along state: consecutive misses, latch, last accepted command."""

    misses: int = 0
    latched: bool = False
    last_command: tuple[float, ...] | None = None


@dataclass(frozen=True)
class GuardDecision:
    """Result: command to send (zeros on reject), why, and the next state."""

    accepted: bool
    reason: GuardReason
    command: tuple[float, ...]
    state: GuardState


def reset() -> GuardState:
    """Return a pristine state - the only way to clear a latch (pure, no mutation)."""
    return GuardState()


def guard(command, confidence: float, state: GuardState, config: GuardConfig) -> GuardDecision:
    """Decide whether an AI command may reach the controller.

    Rejections emit zeros and leave ``last_command`` at the last accepted value,
    so a rejected spike cannot move the rate-of-change reference. The first
    sample has no reference and skips the rate check. Raises ValueError on a
    command that is not 1-D with exactly one value per configured channel.
    """
    values = np.asarray(command, dtype=np.float64)
    if values.ndim != 1 or values.size != len(config.bounds):
        raise ValueError(
            f"command must be 1-D with {len(config.bounds)} values, got shape {values.shape}"
        )
    zeros = (0.0,) * values.size

    if state.latched:
        return GuardDecision(False, GuardReason.LATCHED, zeros, state)

    def reject(reason: GuardReason) -> GuardDecision:
        misses = state.misses + 1
        return GuardDecision(
            False,
            reason,
            zeros,
            GuardState(
                misses=misses,
                latched=misses >= config.max_misses,
                last_command=state.last_command,
            ),
        )

    if not np.all(np.isfinite(values)) or not math.isfinite(confidence):
        return reject(GuardReason.NON_FINITE)
    if any(v < low or v > high for v, (low, high) in zip(values, config.bounds)):
        return reject(GuardReason.OUT_OF_BOUNDS)
    if confidence < config.min_confidence:
        return reject(GuardReason.LOW_CONFIDENCE)
    if state.last_command is not None and any(
        abs(float(v) - prev) > rate
        for v, prev, rate in zip(values, state.last_command, config.max_rate)
    ):
        return reject(GuardReason.RATE_LIMIT)

    accepted = tuple(float(v) for v in values)
    return GuardDecision(
        True,
        GuardReason.OK,
        accepted,
        GuardState(misses=0, latched=False, last_command=accepted),
    )
