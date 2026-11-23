"""Synthetic v1/v2/v3 POTOKSIM bytes for the trainer-facing target contract."""

import argparse
import struct

import numpy as np
import pytest
import torch

from potok_ai.data import (
    SIM_EPISODE,
    SIM_MAGIC,
    SIM_META_V2,
    SIM_META_V2_TRUTH,
    SIM_TARGETS,
    SIM_TARGETS_V3,
    SIM_V2_EPISODE,
    SIM_V2_TRUTH_EPISODE,
    SMART_FEATURES,
    SmartWindowDataset,
    read_validation,
)
from potok_ai.data_v3 import (
    authoritative_command,
    authoritative_target,
    diagnostic_target,
    read_targets,
    training_target,
)
from potok_ai.engine import _smart_frames
from potok_ai.train_smart import distil_student

FRAMES = 3


def _pack(version, *, targets, meta_count, frames=FRAMES, theta=(0.1, -0.2), accel=(2.0, 3.0)):
    """Pack one-episode validation bytes with distinct, version-correct columns."""
    features = SMART_FEATURES if version in (2, 3) else 8
    reserved = struct.pack("<I", meta_count) + b"\0" * 28 if meta_count else b"\0" * 32
    header = struct.pack(
        "<8sIQ4I3f32s",
        SIM_MAGIC, version, 11, 1, frames, features, targets,
        50.0, 0.02, 0.06, reserved,
    )
    meta = [float(i) + 1.0 for i in range(9)]
    if meta_count in (SIM_META_V2, SIM_META_V2_TRUTH):
        meta += [13.0, 0.0082, 0.35, 0.95, 3000.0, 0.01, 16.0, 1.0]
    if meta_count == SIM_META_V2_TRUTH:
        meta += [float(i) for i in range(22)]
    if version == 1:
        record = SIM_EPISODE
    elif meta_count == SIM_META_V2_TRUTH:
        record = SIM_V2_TRUTH_EPISODE
    else:
        record = SIM_V2_EPISODE
    raw = bytearray(header)
    raw += record.pack(0, *meta)
    for tick in range(frames):
        raw += struct.pack(f"<{features}f", *[float(tick)] * features)
    for tick in range(frames):
        row = [theta[0] + tick, theta[1] - tick, 0.9, 1.0]
        if targets == SIM_TARGETS_V3:
            row += [accel[0] + tick, accel[1] + tick]
        raw += struct.pack(f"<{targets}f", *row)
    return bytes(raw)


def test_v3_exposes_authoritative_command_and_diagnostic(tmp_path):
    path = tmp_path / "v3.bin"
    path.write_bytes(_pack(3, targets=SIM_TARGETS_V3, meta_count=SIM_META_V2))
    records = read_targets(path)
    assert len(records) == 1
    record = records[0]
    assert record.version == 3 and record.authoritative
    assert record.command.shape == (FRAMES, 2)
    assert np.allclose(record.command[0], [2.0, -3.0])
    assert np.allclose(record.command[2], [4.0, -5.0])
    assert np.allclose(record.g_theta[0], [0.1, -0.2])
    assert np.allclose(record.training_target, record.command)

    episode = read_validation(path).episodes[0]
    assert np.allclose(authoritative_command(episode), record.command)
    assert np.allclose(authoritative_target(episode), record.command)
    assert np.allclose(diagnostic_target(episode), record.g_theta)


@pytest.mark.parametrize(("version", "meta_count"), [(1, 0), (2, SIM_META_V2)])
def test_v1_v2_keep_diagnostic_but_no_authoritative_command(tmp_path, version, meta_count):
    path = tmp_path / f"v{version}.bin"
    path.write_bytes(_pack(version, targets=SIM_TARGETS, meta_count=meta_count))
    record = read_targets(path)[0]
    assert record.version == version
    assert not record.authoritative and record.command is None
    assert np.allclose(record.g_theta[0], [0.1, -0.2])
    with pytest.raises(ValueError, match="diagnostic"):
        _ = record.training_target

    episode = read_validation(path).episodes[0]
    assert authoritative_command(episode) is None
    with pytest.raises(ValueError, match="diagnostic"):
        authoritative_target(episode)
    assert np.allclose(diagnostic_target(episode), record.g_theta)


def test_command_is_independent_of_diagnostic_columns(tmp_path):
    common = {"targets": SIM_TARGETS_V3, "meta_count": SIM_META_V2}
    packs = {
        "plain": _pack(3, **common),
        "theta": _pack(3, theta=(9.0, -9.0), **common),
        "accel": _pack(3, accel=(7.0, 8.0), **common),
    }
    records = {}
    for name, raw in packs.items():
        path = tmp_path / f"{name}.bin"
        path.write_bytes(raw)
        records[name] = read_targets(path)[0]

    assert np.allclose(records["plain"].command, records["theta"].command)
    assert not np.allclose(records["plain"].command, records["accel"].command)
    assert np.allclose(records["plain"].g_theta, records["accel"].g_theta)
    assert not np.allclose(records["plain"].g_theta, records["theta"].g_theta)


def test_v3_truth_metadata_keeps_authoritative_target(tmp_path):
    path = tmp_path / "truth.bin"
    path.write_bytes(_pack(3, targets=SIM_TARGETS_V3, meta_count=SIM_META_V2_TRUTH))
    data = read_validation(path)
    record = read_targets(path)[0]
    assert (data.episodes[0].scenario_id, data.episodes[0].held_out) == (16, True)
    assert record.authoritative and np.allclose(record.command[0], [2.0, -3.0])


@pytest.mark.parametrize(
    ("version", "targets"), [(3, SIM_TARGETS), (2, SIM_TARGETS_V3)]
)
def test_version_target_width_mismatch_is_loud(tmp_path, version, targets):
    path = tmp_path / "bad.bin"
    path.write_bytes(_pack(version, targets=targets, meta_count=SIM_META_V2))
    with pytest.raises(ValueError, match="targets"):
        read_targets(path)


def _episode_at(tmp_path, version, targets, name):
    path = tmp_path / f"{name}.bin"
    path.write_bytes(_pack(version, targets=targets, meta_count=SIM_META_V2))
    return read_validation(path).episodes[0]


def test_training_target_refuses_v1_v2_unless_diagnostic_is_explicit(tmp_path):
    v3 = _episode_at(tmp_path, 3, SIM_TARGETS_V3, "v3")
    v2 = _episode_at(tmp_path, 2, SIM_TARGETS, "v2")
    assert np.allclose(training_target(v3)[0], [2.0, -3.0])
    assert np.allclose(authoritative_target(v3)[0], [2.0, -3.0])
    with pytest.raises(ValueError, match="v1/v2"):
        training_target(v2)
    with pytest.raises(ValueError, match="v1/v2"):
        authoritative_target(v2)
    assert np.allclose(training_target(v2, allow_diagnostic_target=True)[0], [0.1, -0.2])
    assert np.allclose(diagnostic_target(v2)[0], [0.1, -0.2])


def test_smart_window_dataset_refuses_v2_command_target(tmp_path):
    v2 = _episode_at(tmp_path, 2, SIM_TARGETS, "v2")
    with pytest.raises(ValueError, match="v1/v2"):
        SmartWindowDataset([v2], window=2, limit=1)
    dataset = SmartWindowDataset([v2], window=1, limit=1, allow_diagnostic_target=True)
    assert np.allclose(dataset[0][2].numpy(), [0.1, -0.2])


def test_engine_smart_frames_refuse_v2_command_target(tmp_path):
    v2 = _episode_at(tmp_path, 2, SIM_TARGETS, "v2")
    with pytest.raises(ValueError, match="v1/v2"):
        _smart_frames(v2)
    _, targets = _smart_frames(v2, allow_diagnostic_target=True)
    assert np.allclose(targets[0], [0.1, -0.2])


def test_distil_student_refuses_v2_command_target(tmp_path):
    v2 = _episode_at(tmp_path, 2, SIM_TARGETS, "v2")
    args = argparse.Namespace(seed=1, student_lr=1e-3, window=2, limit=None)
    with pytest.raises(ValueError, match="v1/v2"):
        distil_student(None, [v2], args, torch.device("cpu"))
