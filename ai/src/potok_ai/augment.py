"""Deterministic adversarial augmentation of training windows.

Every transform is a pure function of ``(window, seed, config)``. The only RNG
is a local ``numpy.random.default_rng`` seeded from the explicit argument, so a
(seed, config) pair reproduces the same window on any process. No global RNG,
no mutation of the caller's array.

The default channel set is the canonical 8-channel raw telemetry layout from
``sim/config/default.toml``'s ``output.features``: gyro_x/y/z (rad/s), acc_x/y/z
(m/s^2), baro_h (m), pitot_q (Pa). The six derived physics features of a smart
window (indices 8..13) are left alone by default: they must stay consistent
with the raw telemetry they were computed from.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Absolute sensor biases from sim/config/default.toml's [sensor] block
# (gyro_bias, accel_bias, baro_bias, pitot_bias). Derived physics features get
# zero: there is no direct sensor tolerance to shift within.
BIAS_TOLERANCE: tuple[float, ...] = (
    0.002, 0.002, 0.002,  # gyro_x, gyro_y, gyro_z (rad/s)
    0.020, 0.020, 0.020,  # acc_x, acc_y, acc_z (m/s^2)
    0.400,  # baro_h (m)
    8.000,  # pitot_q (Pa)
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # alpha, beta, q_dyn, a_lat, drag, margin
)

# Training gusts span 5..20 m/s (SPEC_SMART.md, "Рандомизация вокруг Поток-М").
# 1.25x takes the 20 m/s ceiling to 25 m/s: deliberately above training range.
DEFAULT_GUST_GAIN = 1.25

_DROPOUT, _BIAS, _DELAY, _GUST = 1, 2, 3, 4


@dataclass(frozen=True)
class AugmentConfig:
    """All knobs are probabilities/ranges; every transform is off by default."""

    channels: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)
    dropout_prob: float = 0.0
    bias_prob: float = 0.0
    bias_fraction: float = 1.0  # fraction of the sensor tolerance, in [0, 1]
    delay_prob: float = 0.0
    delay_max_steps: int = 2  # frames of late arrival, per channel
    gust_prob: float = 0.0
    gust_gain: float = DEFAULT_GUST_GAIN  # documented multiplier above training range
    gust_shift_max: int = 2  # frames the transient moves, shared by all channels

    def __post_init__(self) -> None:
        for name in ("dropout_prob", "bias_prob", "delay_prob", "gust_prob", "bias_fraction"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.delay_max_steps < 0 or self.gust_shift_max < 0:
            raise ValueError("delay_max_steps and gust_shift_max must be non-negative")
        if self.gust_gain < 1.0:
            raise ValueError("gust_gain must be >= 1 (documented above-range multiplier)")
        if not self.channels:
            raise ValueError("channels must not be empty")
        if any(channel < 0 or channel >= len(BIAS_TOLERANCE) for channel in self.channels):
            raise ValueError(f"channels must be in 0..{len(BIAS_TOLERANCE) - 1}")


DEFAULT_CONFIG = AugmentConfig()


@dataclass(frozen=True)
class AugmentFlags:
    """Which transforms actually changed the window (a change, not a draw)."""

    dropout: bool = False
    bias: bool = False
    delay: bool = False
    gust: bool = False


@dataclass(frozen=True)
class AugmentResult:
    window: np.ndarray
    flags: AugmentFlags


def _prepare(window, config: AugmentConfig) -> np.ndarray:
    array = np.asarray(window)
    if array.ndim != 2 or min(array.shape) < 1:
        raise ValueError("window must be a non-empty 2-D (frames, channels) array")
    if max(config.channels) >= array.shape[1]:
        raise ValueError("config channel index out of range for the window")
    return np.array(array, dtype=np.float32, copy=True)


def _rng(seed: int, tag: int) -> np.random.Generator:
    return np.random.default_rng([int(seed), tag])


def _shift(values: np.ndarray, steps: int) -> np.ndarray:
    """Move a channel in time by ``steps`` (>0 later), filling with edge values."""
    if steps > 0:
        out = np.empty_like(values)
        out[:steps] = values[0]
        out[steps:] = values[:-steps]
        return out
    if steps < 0:
        out = np.empty_like(values)
        out[:steps] = values[-steps:]
        out[steps:] = values[-1]
        return out
    return values.copy()


def sensor_dropout(window, seed: int, config: AugmentConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Mask whole channels, holding their first reading (no out-of-range fill)."""
    out = _prepare(window, config)
    if config.dropout_prob <= 0.0:
        return out
    rng = _rng(seed, _DROPOUT)
    for channel in config.channels:
        if rng.random() < config.dropout_prob:
            out[:, channel] = out[0, channel]
    return out


def bias_shift(window, seed: int, config: AugmentConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Add a constant per-channel offset drawn within the sensor tolerance."""
    out = _prepare(window, config)
    if config.bias_prob <= 0.0:
        return out
    rng = _rng(seed, _BIAS)
    for channel in config.channels:
        if rng.random() < config.bias_prob:
            magnitude = BIAS_TOLERANCE[channel] * config.bias_fraction
            out[:, channel] += magnitude * rng.uniform(-1.0, 1.0)
    return out


def delay_jitter(window, seed: int, config: AugmentConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Delay selected channels by a random 1..delay_max_steps frames, edge-filled."""
    out = _prepare(window, config)
    if config.delay_prob <= 0.0 or config.delay_max_steps == 0:
        return out
    rng = _rng(seed, _DELAY)
    limit = min(config.delay_max_steps, out.shape[0] - 1)
    for channel in config.channels:
        if rng.random() < config.delay_prob and limit > 0:
            steps = int(rng.integers(1, limit + 1))
            out[:, channel] = _shift(out[:, channel], steps)
    return out


# ponytail: gain scales channel deviation, not physical gust m/s (gust truth is
# not in the telemetry); drive it from gust truth when that reaches the dataset.
def gust_shift(window, seed: int, config: AugmentConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Amplify and time-shift transients by the documented above-range multiplier."""
    out = _prepare(window, config)
    if config.gust_prob <= 0.0:
        return out
    rng = _rng(seed, _GUST)
    if rng.random() >= config.gust_prob:
        return out
    limit = min(config.gust_shift_max, out.shape[0] - 1)
    steps = int(rng.integers(-limit, limit + 1)) if limit > 0 else 0
    for channel in config.channels:
        baseline = float(out[:, channel].mean())
        detail = out[:, channel] - baseline
        out[:, channel] = baseline + config.gust_gain * _shift(detail, steps)
    return out


def augment(window, seed: int, config: AugmentConfig = DEFAULT_CONFIG) -> AugmentResult:
    """Apply dropout -> bias -> delay -> gust and report what actually changed."""
    original = _prepare(window, config)
    dropped = sensor_dropout(original, seed, config)
    biased = bias_shift(dropped, seed, config)
    delayed = delay_jitter(biased, seed, config)
    gusty = gust_shift(delayed, seed, config)
    flags = AugmentFlags(
        dropout=not np.array_equal(dropped, original),
        bias=not np.array_equal(biased, dropped),
        delay=not np.array_equal(delayed, biased),
        gust=not np.array_equal(gusty, delayed),
    )
    return AugmentResult(gusty, flags)
