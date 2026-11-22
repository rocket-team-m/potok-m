# SPDX-License-Identifier: GPL-3.0-or-later
"""AI06: hard gate for the deterministic augmentation bridge.

Pins four invariants of ``augment`` / ``augment_integration`` that a silent
regression could break without changing the public API:

1. held-out windows stay byte-identical to their source;
2. the per-window seed is a pure, cross-process-stable function of
   ``(base_seed, index)``, immune to global RNG state and call order;
3. ``AugmentFlags`` report real changes, not random draws;
4. finite input always produces finite output.

No source module is edited and no model, accuracy or speed claim is made here.
All fixtures are in-memory and deterministic.
"""

from __future__ import annotations

import hashlib
import os
import random
import subprocess
import sys
import warnings
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import potok_ai.augment as augment_module
import potok_ai.augment_integration as bridge
from potok_ai.augment import (
    AugmentConfig,
    AugmentFlags,
    augment,
    bias_shift,
    delay_jitter,
    gust_shift,
    sensor_dropout,
)
from potok_ai.augment_integration import (
    AugmentedDataset,
    augment_window,
    split_window_datasets,
    window_seed,
)
from potok_ai.data import SmartWindowDataset, make_smart_episode

_SEED_MASK = 0x7FFF_FFFF_FFFF_FFFF


def _max_config(channels: tuple[int, ...] = tuple(range(8))) -> AugmentConfig:
    return AugmentConfig(
        channels=channels,
        dropout_prob=1.0,
        bias_prob=1.0,
        bias_fraction=1.0,
        delay_prob=1.0,
        delay_max_steps=3,
        gust_prob=1.0,
        gust_gain=4.0,
        gust_shift_max=3,
    )


def _episodes() -> list:
    return [
        make_smart_episode(seed=100 + i, frames=48, scenario_id=i, held_out=i in (1, 3))
        for i in range(5)
    ]


# --- held-out is never augmented -------------------------------------------


def test_held_out_windows_stay_bit_identical_under_max_augmentation():
    episodes = _episodes()
    before = [(e.frames.copy(), e.targets.copy(), e.conditioning.copy()) for e in episodes]
    for base_seed in (0, 7, 2**31):
        for window, step in ((16, 4), (8, 3)):
            train, held = split_window_datasets(
                episodes, base_seed, _max_config(tuple(range(14))), window=window, step=step
            )
            plain = SmartWindowDataset([e for e in episodes if e.held_out], window=window, step=step)
            assert not isinstance(held, AugmentedDataset)
            assert len(train) > 0 and len(held) == len(plain) > 0
            for i in range(len(train)):
                train[i]
            for i in reversed(range(len(held))):
                for got, want in zip(held[i], plain[i]):
                    assert torch.equal(got, want)
                    assert torch.isfinite(got).all()
    for episode, (frames, targets, conditioning) in zip(episodes, before):
        np.testing.assert_array_equal(episode.frames, frames)
        np.testing.assert_array_equal(episode.targets, targets)
        np.testing.assert_array_equal(episode.conditioning, conditioning)


def test_held_out_path_never_reaches_the_augment_call(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("augment_window reached the held-out path")

    monkeypatch.setattr(bridge, "augment_window", forbidden)
    _train, held = split_window_datasets(_episodes(), 9, _max_config(), window=16, step=4)
    assert len(held) > 0
    for i in range(len(held)):
        held[i]


def test_no_path_writes_into_source_frames():
    episodes = _episodes()
    before = [e.frames.copy() for e in episodes]
    for episode in episodes:
        episode.frames.flags.writeable = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*not writable.*", category=UserWarning)
        for base_seed in (0, 5, 13):
            train, held = split_window_datasets(
                episodes, base_seed, _max_config(), window=16, step=4
            )
            for i in range(len(train)):
                train[i]
            for i in range(len(held)):
                held[i]
    for episode, snapshot in zip(episodes, before):
        np.testing.assert_array_equal(episode.frames, snapshot)


# --- deterministic per-window seed -----------------------------------------


def test_window_seed_matches_an_independent_reference():
    for base_seed in (0, 1, 99, 2**40, 2**62):
        for index in (0, 1, 7, 2**32):
            digest = hashlib.blake2b(
                f"{base_seed}:{index}".encode("ascii"), digest_size=8
            ).digest()
            expected = int.from_bytes(digest, "little") & _SEED_MASK
            assert window_seed(base_seed, index) == expected
            assert 0 <= window_seed(base_seed, index) < 2**63
    assert len({window_seed(7, i) for i in range(512)}) == 512
    for bad in (-1, -(2**40)):
        with pytest.raises(ValueError, match="non-negative"):
            window_seed(bad, 0)
        with pytest.raises(ValueError, match="non-negative"):
            window_seed(0, bad)


def test_per_window_augmentation_is_immune_to_global_rng_and_call_order():
    window = make_smart_episode(seed=5, frames=40).frames
    config = _max_config()
    baseline = augment_window(window, 7, 3, config)
    np_state, py_state, torch_state = (
        np.random.get_state(),
        random.getstate(),
        torch.get_rng_state(),
    )
    try:
        for noise in (0, 1, 999):
            np.random.seed(noise)
            random.seed(noise)
            torch.manual_seed(noise)
            for i in range(20):
                augment_window(window, 7, i, config)
            np.testing.assert_array_equal(augment_window(window, 7, 3, config), baseline)
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)
        torch.set_rng_state(torch_state)
    assert not np.array_equal(augment_window(window, 7, 4, config), baseline)
    assert not np.array_equal(augment_window(window, 8, 3, config), baseline)


_CHILD = """
import random
import sys

sys.path.insert(0, sys.argv[1])
import numpy as np
import torch

np.random.seed(int(sys.argv[2]))
random.seed(int(sys.argv[2]))
torch.manual_seed(int(sys.argv[2]))

from potok_ai.augment import AugmentConfig
from potok_ai.augment_integration import augment_window, window_seed

config = AugmentConfig(
    channels=(0, 1, 2, 3, 4, 5, 6, 7),
    dropout_prob=1.0,
    bias_prob=1.0,
    delay_prob=1.0,
    delay_max_steps=3,
    gust_prob=1.0,
    gust_gain=2.0,
    gust_shift_max=3,
)
window = np.arange(16 * 8, dtype=np.float32).reshape(16, 8) / 7.0
print(window_seed(7, 3))
print(augment_window(window, 7, 3, config).tobytes().hex())
"""


def test_window_seed_and_augmentation_are_cross_process_stable():
    src = str(Path(__file__).resolve().parents[1] / "src")
    outputs = []
    for hash_seed, noise in (("0", 1), ("1", 2), ("random", 3)):
        env = os.environ.copy()
        env["PYTHONPATH"] = src
        env["PYTHONHASHSEED"] = hash_seed
        proc = subprocess.run(
            [sys.executable, "-c", _CHILD, src, str(noise)],
            capture_output=True,
            check=True,
            env=env,
            text=True,
            timeout=120,
        )
        outputs.append(proc.stdout)
    assert len(set(outputs)) == 1
    assert outputs[0].strip().splitlines()[0].isdigit()


def test_wrapper_is_index_keyed_under_reversed_batching():
    base = SmartWindowDataset(_episodes(), window=16, step=4)
    config = _max_config()
    forward = AugmentedDataset(base, 42, config)
    expected = {i: forward[i][0].clone() for i in range(len(forward))}
    for i in range(len(base)):
        assert torch.equal(forward[i][1], base[i][1])
        assert torch.equal(forward[i][2], base[i][2])

    order = list(reversed(range(len(forward))))
    seen = []
    for augmented, _conditioning, _target in DataLoader(forward, batch_size=5, sampler=order):
        seen.extend(augmented)
    assert len(seen) == len(expected)
    for index, window in zip(order, seen):
        assert torch.equal(window, expected[index])


# --- fault flags ------------------------------------------------------------


def test_flags_report_changes_not_draws():
    window = np.full((12, 8), 3.0, np.float32)
    window[:, 1] = -2.0
    cases = (
        (AugmentConfig(channels=(0,), dropout_prob=1.0), "dropout"),
        (AugmentConfig(channels=(0,), bias_prob=1.0, bias_fraction=0.0), "bias"),
        (AugmentConfig(channels=(0,), delay_prob=1.0, delay_max_steps=0), "delay"),
        (AugmentConfig(channels=(0,), gust_prob=1.0, gust_gain=1.0, gust_shift_max=0), "gust"),
    )
    for config, name in cases:
        result = augment(window, 3, config)
        assert getattr(result.flags, name) is False
        np.testing.assert_array_equal(result.window, window)

    variable = make_smart_episode(seed=5, frames=16).frames
    positive = augment(
        variable, 3, AugmentConfig(channels=(0,), gust_prob=1.0, gust_gain=2.0, gust_shift_max=0)
    )
    assert positive.flags.gust is True


def test_flags_track_every_stage_for_many_seeds():
    window = make_smart_episode(seed=5, frames=24).frames
    config = _max_config()
    for seed in range(64):
        original = np.array(window, dtype=np.float32, copy=True)
        dropped = sensor_dropout(original, seed, config)
        biased = bias_shift(dropped, seed, config)
        delayed = delay_jitter(biased, seed, config)
        gusty = gust_shift(delayed, seed, config)
        result = augment(window, seed, config)
        assert result.flags == AugmentFlags(
            dropout=not np.array_equal(dropped, original),
            bias=not np.array_equal(biased, dropped),
            delay=not np.array_equal(delayed, biased),
            gust=not np.array_equal(gusty, delayed),
        )
        np.testing.assert_array_equal(result.window, gusty)
        np.testing.assert_array_equal(original, window)


def test_injected_faults_are_reflected_in_flags(monkeypatch):
    window = make_smart_episode(seed=5, frames=24).frames
    config = _max_config()

    def no_op(w, seed, cfg):
        return np.array(w, dtype=np.float32, copy=True)

    monkeypatch.setattr(augment_module, "bias_shift", no_op)
    assert augment(window, 3, config).flags.bias is False

    def corrupt(w, seed, cfg):
        out = np.array(w, dtype=np.float32, copy=True)
        out[:, 0] += 1.0
        return out

    monkeypatch.setattr(augment_module, "gust_shift", corrupt)
    result = augment(window, 3, config)
    assert result.flags.gust is True
    assert np.isfinite(result.window).all()


def test_flags_are_frozen_plain_booleans():
    result = augment(make_smart_episode(seed=5, frames=16).frames, 3, _max_config())
    assert all(type(value) is bool for value in vars(result.flags).values())
    with pytest.raises(FrozenInstanceError):
        result.flags.dropout = True


# --- no NaN -----------------------------------------------------------------


def test_finite_input_never_produces_nan_or_inf():
    for shape in ((1, 8), (2, 14), (24, 14), (64, 8)):
        channels = tuple(range(shape[1]))
        configs = (
            _max_config(channels),
            AugmentConfig(
                channels=channels,
                dropout_prob=1.0,
                bias_prob=1.0,
                bias_fraction=1.0,
                delay_prob=1.0,
                delay_max_steps=16,
                gust_prob=1.0,
                gust_gain=1e3,
                gust_shift_max=16,
            ),
        )
        for seed in range(64):
            window = np.random.default_rng(seed).normal(0.0, 10.0, shape).astype(np.float32)
            for config in configs:
                result = augment(window, seed, config)
                assert result.window.shape == shape
                assert result.window.dtype == np.float32
                assert np.isfinite(result.window).all()
