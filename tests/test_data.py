import numpy as np
import torch

from potok_ai.data import (
    make_synthetic_dataset,
    make_synthetic_episode,
    read_episode,
    write_episode,
)


def test_synthetic_dataset_is_deterministic():
    first = make_synthetic_dataset(n_episodes=2, frames=32, seed=3, window=8)
    second = make_synthetic_dataset(n_episodes=2, frames=32, seed=3, window=8)
    assert len(first) > 0
    for (x1, y1), (x2, y2) in zip(first, second):
        assert torch.equal(x1, x2)
        assert torch.equal(y1, y2)


def test_episode_binary_roundtrip(tmp_path):
    episode = make_synthetic_episode(0, frames=12)
    path = tmp_path / "flight.bin"
    write_episode(path, episode)
    loaded = read_episode(path)
    assert np.allclose(loaded.frames, episode.frames)
    assert loaded.targets is not None and loaded.targets.shape == (12, 2)
    assert np.allclose(loaded.targets, episode.targets)


def test_raw_float32_fallback(tmp_path):
    path = tmp_path / "raw.bin"
    path.write_bytes(np.arange(24, dtype="<f4").tobytes())
    episode = read_episode(path, n_features=8)
    assert episode.frames.shape == (3, 8)
    assert episode.targets is None
