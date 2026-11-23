# SPDX-License-Identifier: GPL-3.0-or-later
"""Privilege-boundary gates for the POTOKTRJ v3 teacher feature builder."""

import struct

import numpy as np
import pytest
import torch

from potok_ai.privileged_teacher import (
    LABEL_DIM,
    PRIV_DIM,
    STUDENT_INPUT_DIM,
    TRJ_HEADER,
    TRJ_HEADER_SIZE,
    TRJ_MAGIC,
    TRJ_VERSION_V3,
    PrivilegedEncoder,
    PrivilegedFeatureBuilder,
    child_seed,
    read_truth_sidecar,
)

FRAMES = 4
EPISODES = (0, 7)


def _values(episode, tick, *, label_corrupt=False, nan=False):
    row = [1000 * episode + 100 * tick + slot for slot in range(26)]
    if label_corrupt:
        for slot in (22, 23, 24, 25):
            row[slot] = -12345.0
    if nan:
        row[0] = float("nan")
    return row


def _sidecar(
    tmp_path,
    *,
    episodes=EPISODES,
    frames=FRAMES,
    version=TRJ_VERSION_V3,
    stride=26,
    magic=TRJ_MAGIC,
    label_corrupt=False,
    nan=False,
    name="truth.trajectory",
):
    raw = bytearray(TRJ_HEADER.pack(magic, version, len(episodes), frames, stride))
    raw += b"\0" * (TRJ_HEADER_SIZE - len(raw))
    for episode in episodes:
        raw += struct.pack("<Q", episode)
        for tick in range(frames):
            raw += struct.pack("<26f", *_values(episode, tick, label_corrupt=label_corrupt, nan=nan))
    path = tmp_path / name
    path.write_bytes(bytes(raw))
    return path


def _builder(path, *, seed=1, z_dim=8):
    return PrivilegedFeatureBuilder(path, seed=seed, z_dim=z_dim)


def test_reader_splits_privileged_slots_from_labels(tmp_path):
    records = read_truth_sidecar(_sidecar(tmp_path))
    assert [r.episode_index for r in records] == list(EPISODES)
    for record in records:
        assert record.privileged.shape == (FRAMES, PRIV_DIM)
        assert record.labels.shape == (FRAMES, LABEL_DIM)
        assert record.privileged.dtype == record.labels.dtype == np.float32
        for tick in range(FRAMES):
            for slot in range(PRIV_DIM):
                assert record.privileged[tick, slot] == 1000 * record.episode_index + 100 * tick + slot
            for slot in range(LABEL_DIM):
                assert record.labels[tick, slot] == 1000 * record.episode_index + 100 * tick + 22 + slot


def test_reader_is_loud_on_missing_and_malformed_truth(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_truth_sidecar(tmp_path / "absent.trajectory")
    with pytest.raises(ValueError, match="bad truth magic"):
        read_truth_sidecar(_sidecar(tmp_path, magic=b"NOPE____", name="magic.trajectory"))
    with pytest.raises(ValueError, match="not v3"):
        read_truth_sidecar(_sidecar(tmp_path, version=1, name="v1.trajectory"))
    with pytest.raises(ValueError, match="not v3"):
        read_truth_sidecar(_sidecar(tmp_path, version=2, name="v2.trajectory"))
    with pytest.raises(ValueError, match="stride"):
        read_truth_sidecar(_sidecar(tmp_path, stride=24, name="stride.trajectory"))

    good = _sidecar(tmp_path)
    short = tmp_path / "short.trajectory"
    short.write_bytes(good.read_bytes()[:-8])
    with pytest.raises(ValueError, match="truncated"):
        read_truth_sidecar(short)

    with pytest.raises(ValueError, match="non-finite privileged truth"):
        read_truth_sidecar(_sidecar(tmp_path, nan=True, name="nan.trajectory"))


def test_z_priv_is_low_dimensional_and_blind_to_label_columns(tmp_path):
    path = _sidecar(tmp_path)
    builder = _builder(path, z_dim=6)
    z = builder.z_priv(7)
    assert z.shape == (FRAMES, 6)
    assert 6 < PRIV_DIM
    assert torch.isfinite(z).all()

    before = builder.z_priv(7)
    builder._record(7).labels[:] = 9999.0
    assert torch.equal(builder.z_priv(7), before)

    other = _sidecar(tmp_path, label_corrupt=True, name="other.trajectory")
    assert torch.allclose(_builder(other).z_priv(0), _builder(path).z_priv(0))


def test_seed_is_deterministic_and_local(tmp_path):
    path = _sidecar(tmp_path)
    before = torch.get_rng_state().clone()
    a = _builder(path, seed=11)
    after = torch.get_rng_state().clone()
    assert torch.equal(before, after), "constructing the builder advanced the global RNG"

    b = _builder(path, seed=11)
    assert torch.equal(a.z_priv(0), b.z_priv(0))
    assert torch.equal(a.encoder.state_dict()["w1"], b.encoder.state_dict()["w1"])

    c = _builder(path, seed=12)
    assert not torch.equal(a.z_priv(0), c.z_priv(0))
    assert child_seed(11, "privileged:encoder") == child_seed(11, "privileged:encoder")
    assert child_seed(11, "privileged:encoder") != child_seed(12, "privileged:encoder")


def test_student_inputs_are_never_contaminated(tmp_path):
    builder = _builder(_sidecar(tmp_path))
    student = torch.zeros(FRAMES, STUDENT_INPUT_DIM)
    snapshot = student.clone()
    builder.assert_student_clean(student)
    builder.z_priv(0)
    assert torch.equal(student, snapshot), "builder mutated the student tensor"

    for width in (PRIV_DIM, 26, STUDENT_INPUT_DIM + 1):
        with pytest.raises(ValueError, match="sensor features"):
            builder.assert_student_clean(torch.zeros(FRAMES, width))


def test_normalizer_is_checkpointable_and_uses_only_builder_truth(tmp_path):
    train = _sidecar(tmp_path, episodes=(0, 1), name="train.trajectory")
    heldout = _sidecar(tmp_path, episodes=(99,), name="heldout.trajectory")
    builder = _builder(train)
    state = builder.encoder.state_dict()
    assert "mean" in state and "std" in state
    assert state["mean"].shape == (PRIV_DIM,)

    train_only = np.concatenate([r.privileged for r in builder.records], axis=0).mean(axis=0)
    assert np.allclose(state["mean"].numpy(), train_only)
    all_records = builder.records + read_truth_sidecar(heldout)
    combined = np.concatenate([r.privileged for r in all_records], axis=0).mean(axis=0)
    assert not np.allclose(state["mean"].numpy(), combined), "normalizer saw held-out truth"


def test_encoder_rejects_wrong_width_and_bad_dimensions():
    mean = np.zeros(PRIV_DIM, dtype=np.float32)
    std = np.ones(PRIV_DIM, dtype=np.float32)
    with pytest.raises(ValueError, match="z_dim"):
        PrivilegedEncoder(seed=1, mean=mean, std=std, z_dim=PRIV_DIM)
    with pytest.raises(ValueError, match="shape"):
        PrivilegedEncoder(seed=1, mean=mean[:-1], std=std)
    encoder = PrivilegedEncoder(seed=1, mean=mean, std=std)
    with pytest.raises(ValueError, match="slots 0..21"):
        encoder(torch.zeros(3, 20))


def test_unknown_episode_is_loud(tmp_path):
    builder = _builder(_sidecar(tmp_path))
    with pytest.raises(ValueError, match="not in truth sidecar"):
        builder.z_priv(123)
    with pytest.raises(ValueError, match="not in truth sidecar"):
        builder.labels(123)
