"""Deterministic adversarial augmentation of training windows."""

import numpy as np
import pytest

from potok_ai.augment import (
    BIAS_TOLERANCE,
    AugmentConfig,
    AugmentFlags,
    augment,
    bias_shift,
    delay_jitter,
    gust_shift,
    sensor_dropout,
)

TRANSFORMS = (sensor_dropout, bias_shift, delay_jitter, gust_shift)


def _window(seed: int = 7, frames: int = 12) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, 1.0, (frames, 8)).astype(np.float32)


def _all_on(**overrides) -> AugmentConfig:
    knobs = {
        "dropout_prob": 0.5,
        "bias_prob": 0.5,
        "delay_prob": 0.5,
        "gust_prob": 0.5,
    }
    knobs.update(overrides)
    return AugmentConfig(**knobs)


def test_default_config_is_a_noop():
    window = _window()
    result = augment(window, seed=3)
    assert result.flags == AugmentFlags()
    np.testing.assert_array_equal(result.window, window)
    assert result.window.dtype == np.float32


def test_no_transform_mutates_the_input():
    original = _window()
    before = original.copy()
    for transform in TRANSFORMS:
        transform(original, 4, _all_on())
        np.testing.assert_array_equal(original, before)
    augment(original, 4, _all_on())
    np.testing.assert_array_equal(original, before)


def test_determinism_and_seed_sensitivity():
    window = _window()
    config = _all_on()
    first, repeat, other = augment(window, 11, config), augment(window, 11, config), augment(window, 12, config)
    np.testing.assert_array_equal(first.window, repeat.window)
    assert first.flags == repeat.flags
    assert not np.array_equal(first.window, other.window)
    for transform in TRANSFORMS:
        np.testing.assert_array_equal(transform(window, 9, config), transform(window, 9, config))


def test_flags_report_only_transforms_that_changed_the_window():
    window = _window()
    cases = (
        (AugmentConfig(dropout_prob=1.0), AugmentFlags(dropout=True)),
        (AugmentConfig(bias_prob=1.0), AugmentFlags(bias=True)),
        (AugmentConfig(delay_prob=1.0, delay_max_steps=2), AugmentFlags(delay=True)),
        (AugmentConfig(gust_prob=1.0, gust_gain=1.25, gust_shift_max=2), AugmentFlags(gust=True)),
    )
    for config, expected in cases:
        assert augment(window, 1, config).flags == expected


def test_finite_output_across_seeds_and_transforms():
    for seed in range(20):
        result = augment(_window(seed), seed, _all_on())
        assert np.isfinite(result.window).all()
        assert result.window.shape == (12, 8)


def test_bias_stays_within_sensor_tolerance_and_skips_physics_features():
    window = np.concatenate([_window(), np.ones((12, 6), np.float32)], axis=1)
    config = AugmentConfig(bias_prob=1.0)
    out = bias_shift(window, 5, config)
    delta = out[:, :8] - window[:, :8]
    for channel in range(8):
        assert np.abs(delta[:, channel]).max() <= BIAS_TOLERANCE[channel] + 1e-6
    np.testing.assert_array_equal(out[:, 8:], window[:, 8:])


def test_bias_fraction_bounds_the_shift():
    window = _window()
    config = AugmentConfig(bias_prob=1.0, bias_fraction=0.25)
    delta = np.abs(bias_shift(window, 5, config) - window)
    assert delta.max() <= 0.25 * max(BIAS_TOLERANCE[:8]) + 1e-6


def test_dropout_holds_an_in_range_value_per_channel():
    window = _window()
    config = AugmentConfig(channels=(0, 1), dropout_prob=1.0)
    out = sensor_dropout(window, 1, config)
    np.testing.assert_array_equal(out[:, 0], window[0, 0])
    np.testing.assert_array_equal(out[:, 1], window[0, 1])
    np.testing.assert_array_equal(out[:, 2:], window[:, 2:])


def test_delay_uses_edge_fill_and_never_wraps():
    window = np.arange(12, dtype=np.float32).reshape(12, 1)
    config = AugmentConfig(channels=(0,), delay_prob=1.0, delay_max_steps=2)
    out = delay_jitter(window, 1, config)
    np.testing.assert_array_equal(out[:2, 0], window[0, 0])
    assert set(np.unique(out[:, 0])).issubset(set(window[:, 0]))


def test_gust_gain_is_the_documented_multiplier():
    window = _window()
    config = AugmentConfig(channels=(0,), gust_prob=1.0, gust_gain=1.25, gust_shift_max=0)
    out = gust_shift(window, 1, config)
    baseline = window[:, 0].mean()
    np.testing.assert_allclose(out[:, 0], baseline + 1.25 * (window[:, 0] - baseline), rtol=1e-6)


def test_gust_shift_moves_the_transient():
    window = np.zeros((12, 8), np.float32)
    window[5, 0] = 1.0
    config = AugmentConfig(channels=(0,), gust_prob=1.0, gust_shift_max=2)
    peaks = {int(gust_shift(window, seed, config)[:, 0].argmax()) for seed in range(10)}
    assert peaks != {5}
    assert peaks <= {3, 4, 5, 6, 7}


def test_config_rejects_out_of_range_knobs():
    with pytest.raises(ValueError, match="dropout_prob"):
        AugmentConfig(dropout_prob=1.5)
    with pytest.raises(ValueError, match="gust_gain"):
        AugmentConfig(gust_gain=0.9)
    with pytest.raises(ValueError, match="channels"):
        AugmentConfig(channels=(99,))
    with pytest.raises(ValueError, match="non-empty"):
        augment(np.zeros((0, 8), np.float32), 1)
