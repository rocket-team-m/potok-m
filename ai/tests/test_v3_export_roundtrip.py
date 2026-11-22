# SPDX-License-Identifier: GPL-3.0-or-later
"""Round-trip gates for the v3 command-target export and its AI-side loader.

The simulator's v3 target export is a POTOKSIM version-3 ``validation.bin``:
the four legacy columns plus ``g_accel_y``/``g_accel_x`` (m/s², world frame) on
every frame. These tests build that export deterministically in memory and load
it through the production readers (``data.read_validation``,
``data_v3.read_targets``). The digest is the integrity record a dataset manifest
carries; a final gate covers the SHA-trailered model export, the package's other
portable export format.
"""

from __future__ import annotations

import hashlib
import struct

import numpy as np
import pytest
import torch

from potok_ai.data import (
    SIM_EPISODE,
    SIM_MAGIC,
    SIM_META_V2,
    SIM_TARGETS,
    SIM_TARGETS_V3,
    SIM_V2_EPISODE,
    SIM_VERSION,
    SIM_VERSION_SMART,
    SIM_VERSION_V3,
    SMART_FEATURES,
    read_validation,
)
from potok_ai.data_v3 import (
    authoritative_command,
    authoritative_target,
    diagnostic_target,
    read_targets,
)
from potok_ai.export import export_model, load_export
from potok_ai.model import ModelConfig, PotokModel

FRAMES = 3
THETA = (0.1, -0.2)
ACCEL = (2.0, 3.0)


def _pack_target_export(version, *, targets, theta=THETA, accel=ACCEL):
    """Build one-episode POTOKSIM bytes with the target columns ``version`` writes."""
    smart = version in (SIM_VERSION_SMART, SIM_VERSION_V3)
    features = SMART_FEATURES if smart else 8
    meta_count = SIM_META_V2 if smart else 0
    reserved = struct.pack("<I", meta_count) + bytes(28) if smart else bytes(32)
    header = struct.pack(
        "<8sIQ4I3f32s",
        SIM_MAGIC, version, 11, 1, FRAMES, features, targets,
        50.0, 0.02, 0.06, reserved,
    )
    meta = [float(i) + 1.0 for i in range(9)]
    if smart:
        meta += [13.0, 0.0082, 0.35, 0.95, 3000.0, 0.01, 16.0, 1.0]
    record = SIM_V2_EPISODE if smart else SIM_EPISODE
    raw = bytearray(header)
    raw += record.pack(0, *meta)
    for tick in range(FRAMES):
        raw += struct.pack(f"<{features}f", *([float(tick)] * features))
    for tick in range(FRAMES):
        row = [theta[0] + tick, theta[1] - tick, 0.9, 1.0]
        if targets == SIM_TARGETS_V3:
            row += [accel[0] + tick, accel[1] + tick]
        raw += struct.pack(f"<{targets}f", *row)
    return bytes(raw)


def _load(path, raw):
    path.write_bytes(raw)
    return read_targets(path)[0]


def test_v3_target_export_bytes_and_digest_are_deterministic():
    first = _pack_target_export(SIM_VERSION_V3, targets=SIM_TARGETS_V3, accel=(1.25, -3.5))
    second = _pack_target_export(SIM_VERSION_V3, targets=SIM_TARGETS_V3, accel=(1.25, -3.5))
    assert first == second
    assert hashlib.sha256(first).digest() == hashlib.sha256(second).digest()
    assert len(hashlib.sha256(first).digest()) == 32


def test_v3_target_load_is_deterministic(tmp_path):
    raw = _pack_target_export(SIM_VERSION_V3, targets=SIM_TARGETS_V3)
    first = _load(tmp_path / "a.bin", raw)
    second = _load(tmp_path / "b.bin", raw)
    assert np.array_equal(first.command, second.command)
    assert np.array_equal(first.g_theta, second.g_theta)
    assert first.authoritative == second.authoritative


def test_export_digest_detects_tampered_target_bytes(tmp_path):
    raw = _pack_target_export(SIM_VERSION_V3, targets=SIM_TARGETS_V3)
    digest = hashlib.sha256(raw).digest()
    clean = _load(tmp_path / "clean.bin", raw)
    last = len(raw) - 4  # low byte of the last g_accel_x value
    tampered = raw[:last] + bytes([raw[last] ^ 0x01]) + raw[last + 1 :]
    changed = _load(tmp_path / "tampered.bin", tampered)
    assert hashlib.sha256(tampered).digest() != digest
    assert np.array_equal(clean.g_theta, changed.g_theta)
    assert not np.array_equal(clean.command, changed.command)


def test_g_accel_provenance_survives_the_roundtrip(tmp_path):
    path = tmp_path / "v3.bin"
    path.write_bytes(_pack_target_export(SIM_VERSION_V3, targets=SIM_TARGETS_V3))
    record = read_targets(path)[0]
    assert record.version == SIM_VERSION_V3 and record.authoritative
    assert np.allclose(record.command[0], [ACCEL[0], -ACCEL[1]])
    assert np.allclose(record.command[-1], [ACCEL[0] + FRAMES - 1, -(ACCEL[1] + FRAMES - 1)])
    assert np.allclose(record.g_theta[0], THETA)
    assert np.array_equal(record.training_target, record.command)

    episode = read_validation(path).episodes[0]
    assert np.array_equal(authoritative_command(episode), record.command)
    assert np.array_equal(authoritative_target(episode), record.command)
    assert np.array_equal(diagnostic_target(episode), record.g_theta)


def test_command_comes_from_g_accel_not_g_theta(tmp_path):
    common = {"targets": SIM_TARGETS_V3}
    packs = {
        "plain": _pack_target_export(SIM_VERSION_V3, **common),
        "theta": _pack_target_export(SIM_VERSION_V3, theta=(9.0, -9.0), **common),
        "accel": _pack_target_export(SIM_VERSION_V3, accel=(7.0, 8.0), **common),
    }
    records = {name: _load(tmp_path / f"{name}.bin", raw) for name, raw in packs.items()}
    assert np.array_equal(records["plain"].command, records["theta"].command)
    assert np.array_equal(records["plain"].g_theta, records["accel"].g_theta)
    assert not np.array_equal(records["plain"].command, records["accel"].command)
    assert not np.array_equal(records["plain"].g_theta, records["theta"].g_theta)


@pytest.mark.parametrize(
    ("version", "targets"),
    [(SIM_VERSION, SIM_TARGETS), (SIM_VERSION_SMART, SIM_TARGETS)],
)
def test_diagnostic_only_export_is_never_promoted(tmp_path, version, targets):
    path = tmp_path / f"v{version}.bin"
    path.write_bytes(_pack_target_export(version, targets=targets))
    record = read_targets(path)[0]
    assert record.version == version
    assert not record.authoritative and record.command is None
    with pytest.raises(ValueError, match="diagnostic"):
        _ = record.training_target

    episode = read_validation(path).episodes[0]
    assert authoritative_command(episode) is None
    with pytest.raises(ValueError, match="diagnostic"):
        authoritative_target(episode)
    assert np.array_equal(diagnostic_target(episode), record.g_theta)


@pytest.mark.parametrize(
    ("version", "targets"), [(SIM_VERSION_V3, SIM_TARGETS), (SIM_VERSION_SMART, SIM_TARGETS_V3)]
)
def test_version_target_width_mismatch_is_refused(tmp_path, version, targets):
    path = tmp_path / "bad.bin"
    path.write_bytes(_pack_target_export(version, targets=targets))
    with pytest.raises(ValueError, match="targets"):
        read_targets(path)


def test_export_file_sha_trailer_is_deterministic_and_verified(tmp_path):
    torch.manual_seed(23)
    model = PotokModel(ModelConfig(width=16, rank=8))
    first = export_model(model, tmp_path / "a.bin")
    second = export_model(model, tmp_path / "b.bin")
    raw = first.read_bytes()
    assert raw == second.read_bytes()
    payload, checksum = raw[:-32], raw[-32:]
    assert checksum == hashlib.sha256(payload).digest()
    assert load_export(first).checksum == checksum
    assert load_export(second).checksum == checksum

    tampered = bytearray(raw)
    tampered[len(payload) // 2] ^= 0xFF
    path = tmp_path / "tampered.bin"
    path.write_bytes(tampered)
    with pytest.raises(ValueError, match="checksum"):
        load_export(path)
