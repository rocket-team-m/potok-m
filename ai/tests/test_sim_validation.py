"""Handcrafted POTOKSIM validation file, packed exactly as sim/README.md says."""

import struct

import numpy as np

from potok_ai.data import (
    SIM_MAGIC,
    SIM_META_V2,
    SIM_META_V2_TRUTH,
    SIM_TARGETS,
    SIM_TARGETS_V3,
    SIM_V2_EPISODE,
    SIM_V2_TRUTH_EPISODE,
    SMART_FEATURES,
    SmartWindowDataset,
    WindowDataset,
    load_episodes,
    read_validation,
)


def _pack_validation(seed: int, count: int, frames: int, features: int) -> bytes:
    header = struct.pack(
        "<8sIQ4I3f32s",
        SIM_MAGIC, 1, seed, count, frames, features, 4,
        50.0, 0.02, 0.06, b"\0" * 32,
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
    assert np.array_equal(episode.metadata, np.array([2.0, 1.5, 2.0, 3.0, -1.0, 0.5, 4.0, 5.0, 6.0], np.float32))
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


def _pack_v3(version: int, targets: int, meta_count: int, frames: int = 4) -> bytes:
    """Pack one smart-layout POTOKSIM file with the given version/target width."""
    reserved = struct.pack("<I", meta_count) + b"\0" * 28
    header = struct.pack(
        "<8sIQ4I3f32s",
        SIM_MAGIC, version, 7, 1, frames, SMART_FEATURES, targets,
        50.0, 0.02, 0.06, reserved,
    )
    meta = [float(i) for i in range(9)] + [13.0, 0.0082, 0.35, 0.95, 3000.0, 0.01]
    meta += [16.0, 1.0]  # scenario 16, held-out
    if meta_count == SIM_META_V2_TRUTH:
        meta += [float(i) for i in range(22)]  # terminal truth record
    raw = bytearray(header)
    record = SIM_V2_TRUTH_EPISODE if meta_count == SIM_META_V2_TRUTH else SIM_V2_EPISODE
    raw += record.pack(0, *meta)
    raw += np.zeros((frames, SMART_FEATURES), np.float32).tobytes()
    # Legacy columns carry g_theta; v3 appends [g_accel_y, g_accel_x].
    rows = np.zeros((frames, targets), np.float32)
    rows[:, :SIM_TARGETS] = [0.01, -0.02, 0.9, 1.0]
    if targets == SIM_TARGETS_V3:
        rows[:, 4] = 2.0
        rows[:, 5] = 3.0
    raw += rows.tobytes()
    return bytes(raw)


def test_reads_v3_command_target_not_angle_target(tmp_path):
    path = tmp_path / "v3.bin"
    path.write_bytes(_pack_v3(3, SIM_TARGETS_V3, SIM_META_V2))
    data = read_validation(path)
    assert data.header.version == 3 and data.header.target_count == SIM_TARGETS_V3
    episode = data.episodes[0]
    assert episode.targets.shape == (4, 6)
    assert episode.g_accel is not None and episode.g_accel.shape == (4, 2)
    assert np.allclose(episode.g_accel[0], [2.0, 3.0])
    # g_theta stays a diagnostic; the fin command flips the X axis.
    assert np.allclose(episode.g_theta()[0], [0.01, -0.02])
    assert np.allclose(episode.command_target()[0], [2.0, -3.0])
    dataset = SmartWindowDataset([episode], window=2, limit=1)
    _, _, target = dataset[0]
    assert np.allclose(target.numpy(), [2.0, -3.0])


def test_v3_target_width_mismatch_is_loud(tmp_path):
    path = tmp_path / "bad.bin"
    path.write_bytes(_pack_v3(3, SIM_TARGETS, SIM_META_V2))
    try:
        read_validation(path)
    except ValueError as error:
        assert "version 3 needs 6 targets" in str(error)
    else:
        raise AssertionError("a v3 file with 4 targets must be refused")

    path.write_bytes(_pack_v3(2, SIM_TARGETS_V3, SIM_META_V2))
    try:
        read_validation(path)
    except ValueError as error:
        assert "version 2 needs 4 targets" in str(error)
    else:
        raise AssertionError("a v2 file with 6 targets must be refused")


def test_v3_truth_metadata_keeps_scenario_split_and_target(tmp_path):
    path = tmp_path / "truth.bin"
    path.write_bytes(_pack_v3(3, SIM_TARGETS_V3, SIM_META_V2_TRUTH))
    episode = read_validation(path).episodes[0]
    assert (episode.scenario, episode.scenario_id, episode.held_out) == ("scenario_16", 16, True)
    assert np.allclose(episode.command_target()[0], [2.0, -3.0])


def test_legacy_v2_target_is_diagnostic_g_theta(tmp_path):
    path = tmp_path / "v2.bin"
    path.write_bytes(_pack_v3(2, SIM_TARGETS, SIM_META_V2))
    episode = read_validation(path).episodes[0]
    assert episode.g_accel is None
    assert np.allclose(episode.command_target()[0], [0.01, -0.02])


def test_rejects_corrupt_header(tmp_path):
    path = tmp_path / "bad.bin"
    path.write_bytes(b"POTOKSIM" + struct.pack("<I", 9) + b"\0" * 68)
    try:
        read_validation(path)
    except ValueError as error:
        assert "version 1" in str(error)
    else:
        raise AssertionError("a bad version must be refused")
