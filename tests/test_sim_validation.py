"""Handcrafted POTOKSIM validation file, packed exactly as sim/README.md says."""

import struct

import numpy as np

from potok_ai.data import SIM_MAGIC, WindowDataset, load_episodes, read_validation


def _pack_validation(seed: int, count: int, frames: int, features: int) -> bytes:
    header = struct.pack(
        "<8sIQ4I3f32s",
        SIM_MAGIC,
        1,
        seed,
        count,
        frames,
        features,
        4,
        50.0,
        0.02,
        0.06,
        b"\0" * 32,
    )
    raw = bytearray(header)
    for episode in range(count):
        metadata = [episode + 1.0, 1.5, 2.0, 3.0, -1.0, 0.5, 4.0, 5.0, 6.0]
        raw += struct.pack("<Q9f", episode, *metadata)
        for frame in range(frames):
            raw += struct.pack(f"<{features}f", *[frame + k / 10.0 for k in range(features)])
        for frame in range(frames):
            raw += struct.pack("<4f", 0.01 * frame, -0.02 * frame, 0.9, float(frame % 2))
    return bytes(raw)


def test_reads_potoksim_validation_exactly(tmp_path):
    raw = _pack_validation(seed=42, count=2, frames=3, features=8)
    assert len(raw) == 80 + 2 * (44 + 3 * 8 * 4 + 3 * 4 * 4)
    path = tmp_path / "validation.bin"
    path.write_bytes(raw)

    data = read_validation(path)
    assert (data.header.seed, data.header.episode_count) == (42, 2)
    assert (data.header.frames_per_episode, data.header.feature_count) == (3, 8)
    assert data.header.target_count == 4
    assert data.header.frame_rate_hz == 50.0
    assert abs(data.header.dt - 0.02) < 1e-6
    assert abs(data.header.duration_s - 0.06) < 1e-6
    assert len(data.episodes) == 2

    episode = data.episodes[1]
    assert episode.index == 1
    assert np.array_equal(
        episode.metadata, np.array([2.0, 1.5, 2.0, 3.0, -1.0, 0.5, 4.0, 5.0, 6.0], np.float32)
    )
    assert episode.frames.shape == (3, 8)
    assert episode.targets is not None and episode.targets.shape == (3, 4)
    assert episode.frames[1, 0] == 1.0
    assert np.allclose(episode.targets[2], [0.02, -0.04, 0.9, 0.0], atol=1e-7)

    episodes = load_episodes(path)
    assert [ep.index for ep in episodes] == [0, 1]
    dataset = WindowDataset(episodes, window=2)
    window, target = dataset[0]
    assert window.shape == (2, 8) and target.shape == (2,)
    assert np.allclose(target.numpy(), [0.01, -0.02], atol=1e-7)


def test_rejects_corrupt_header(tmp_path):
    path = tmp_path / "bad.bin"
    path.write_bytes(b"POTOKSIM" + struct.pack("<I", 9) + b"\0" * 68)
    try:
        read_validation(path)
    except ValueError as error:
        assert "version 1" in str(error)
    else:
        raise AssertionError("a bad version must be refused")
