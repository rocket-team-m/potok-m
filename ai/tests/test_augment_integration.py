"""Augmentation bridge: per-window seed, held-out untouched, no global RNG."""

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from potok_ai.augment import AugmentConfig
from potok_ai.augment_integration import (
    AugmentedDataset,
    augment_window,
    split_window_datasets,
    window_seed,
)
from potok_ai.data import SmartWindowDataset, make_smart_episode


def _all_on() -> AugmentConfig:
    return AugmentConfig(dropout_prob=0.5, bias_prob=0.5, delay_prob=0.5, gust_prob=0.5)


def _episodes(held_out: tuple[int, ...] = (2,)) -> list:
    return [
        make_smart_episode(seed=100 + i, frames=40, scenario_id=i, held_out=i in held_out)
        for i in range(4)
    ]


def test_window_seed_is_pinned_and_separates_neighbours():
    # Fixed bytes across processes and Python versions -- never hash().
    assert window_seed(1, 0) == 8864653025108052306
    assert window_seed(7, 3) == 1196508733449348778
    assert len({window_seed(1, i) for i in range(32)}) == 32
    assert len({window_seed(s, 0) for s in range(32)}) == 32
    with pytest.raises(ValueError, match="non-negative"):
        window_seed(-1, 0)
    with pytest.raises(ValueError, match="non-negative"):
        window_seed(1, -1)


def test_augment_window_is_pure_deterministic_and_per_window():
    window = make_smart_episode(seed=5, frames=40).frames
    before = window.copy()
    config = _all_on()
    first = augment_window(window, 7, 3, config)
    np.testing.assert_array_equal(first, augment_window(window, 7, 3, config))
    assert not np.array_equal(first, augment_window(window, 7, 4, config))
    assert not np.array_equal(first, augment_window(window, 8, 3, config))
    np.testing.assert_array_equal(window, before)
    assert first.dtype == np.float32


def test_wrapper_is_deterministic_and_preserves_item_structure():
    base = SmartWindowDataset(_episodes(), window=16, step=4)
    config = _all_on()
    first = AugmentedDataset(base, 11, config)
    repeat = AugmentedDataset(base, 11, config)
    other = AugmentedDataset(base, 12, config)
    assert isinstance(first, Dataset) and len(first) == len(base)
    changed = 0
    for i in range(len(base)):
        window, condition, target = first[i]
        np.testing.assert_array_equal(window, repeat[i][0])
        np.testing.assert_array_equal(condition, base[i][1])
        np.testing.assert_array_equal(target, base[i][2])
        assert window.dtype == torch.float32 and window.shape == base[i][0].shape
        changed += not torch.equal(window, other[i][0])
    assert changed > 0


def test_wrapper_default_config_is_a_noop_and_never_mutates_base():
    base = SmartWindowDataset(_episodes(), window=16, step=4)
    plain = [base[i][0].clone() for i in range(len(base))]
    wrapped = AugmentedDataset(base, 11)
    for i in range(len(base)):
        np.testing.assert_array_equal(wrapped[i][0], plain[i])
        np.testing.assert_array_equal(base[i][0], plain[i])
    with pytest.raises(ValueError, match="base_seed"):
        AugmentedDataset(base, -1)


def test_held_out_is_never_augmented():
    episodes = _episodes(held_out=(1, 3))
    before = [episode.frames.copy() for episode in episodes]
    train, held_out = split_window_datasets(episodes, 7, _all_on(), window=16, step=4)
    assert len(train) > 0 and len(held_out) > 0
    for episode, frames in zip(episodes, before):
        np.testing.assert_array_equal(episode.frames, frames)

    held_plain = SmartWindowDataset(
        [episode for episode in episodes if episode.held_out], window=16, step=4
    )
    train_plain = SmartWindowDataset(
        [episode for episode in episodes if not episode.held_out], window=16, step=4
    )
    assert len(held_out) == len(held_plain)
    for i in range(len(held_out)):
        for got, expected in zip(held_out[i], held_plain[i]):
            np.testing.assert_array_equal(got, expected)
    changed = sum(not torch.equal(train[i][0], train_plain[i][0]) for i in range(len(train)))
    assert changed > 0


def test_augmentation_does_not_advance_any_global_rng():
    np.random.seed(1234)
    expected_np = np.random.random(4)
    torch.manual_seed(1234)
    expected_torch = torch.rand(4)

    np.random.seed(1234)
    torch.manual_seed(1234)
    window = make_smart_episode(seed=5, frames=40).frames
    for index in range(4):
        augment_window(window, 7, index, _all_on())
    np.testing.assert_array_equal(np.random.random(4), expected_np)
    np.testing.assert_array_equal(torch.rand(4), expected_torch)


def test_split_rejects_a_missing_side():
    with pytest.raises(ValueError, match="held-out"):
        split_window_datasets(_episodes(held_out=()), 7, window=16)
    with pytest.raises(ValueError, match="training"):
        split_window_datasets(_episodes(held_out=(0, 1, 2, 3)), 7, window=16)
