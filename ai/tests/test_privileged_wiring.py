# SPDX-License-Identifier: GPL-3.0-or-later
"""Wiring gates: POTOKTRJ v3 truth must reach the privileged stage and DAgger."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
import torch

from potok_ai import engine
from potok_ai import train_teacher_v2 as ttv
from potok_ai.data import make_smart_episode
from potok_ai.privileged_teacher import (
    LABEL_DIM,
    PRIV_DIM,
    TRJ_HEADER,
    TRJ_HEADER_SIZE,
    TRJ_MAGIC,
    TRJ_VERSION_V3,
)

FRAMES = 24
EPISODE_IDS = tuple(100 + i for i in range(4))


def _episodes() -> list:
    return [
        make_smart_episode(seed=eid, frames=FRAMES, scenario_id=i)
        for i, eid in enumerate(EPISODE_IDS)
    ]


def _values(episode, tick: int, *, zero_priv: bool = False) -> list[float]:
    """One v3 tick: non-zero privileged truth + labels matching the episode."""
    privileged = (
        [0.0] * PRIV_DIM
        if zero_priv
        else [10.0 + slot + 0.1 * tick + 0.01 * int(episode.index) for slot in range(PRIV_DIM)]
    )
    theta = episode.targets[tick, :2]
    g_accel = episode.g_accel[tick]
    return [*privileged, float(theta[0]), float(theta[1]), float(g_accel[0]), float(g_accel[1])]


def _sidecar(tmp_path: Path, *, zero_priv: bool = False, name: str = "truth.trajectory") -> Path:
    by_id = {int(episode.index): episode for episode in _episodes()}
    raw = bytearray(
        TRJ_HEADER.pack(TRJ_MAGIC, TRJ_VERSION_V3, len(EPISODE_IDS), FRAMES, 26)
    )
    raw += b"\0" * (TRJ_HEADER_SIZE - len(raw))
    for episode_id in EPISODE_IDS:
        episode = by_id[episode_id]
        raw += struct.pack("<Q", episode_id)
        for tick in range(FRAMES):
            raw += struct.pack("<26f", *_values(episode, tick, zero_priv=zero_priv))
    path = tmp_path / name
    path.write_bytes(bytes(raw))
    return path


def _cfg(tmp_path: Path, **overrides) -> ttv.TeacherV2Config:
    kwargs = {
        "data": tmp_path / "data",
        "out": tmp_path / "out",
        "seed": 1,
        "epochs": 1,
        "batch": 2,
        "window": 12,
        "limit": 6,
        "width": 16,
        "heads": 2,
        "max_iters": 2,
    }
    kwargs.update(overrides)
    return ttv.TeacherV2Config(**kwargs)


def test_v3_truth_parts_pin_the_privilege_boundary() -> None:
    row = np.arange(26, dtype=np.float32)
    privileged, labels = ttv._v3_truth_parts(row, where="test")
    assert privileged.shape == (PRIV_DIM,) and labels.shape == (LABEL_DIM,)
    np.testing.assert_array_equal(privileged, row[:22])
    np.testing.assert_array_equal(labels, row[22:])
    batch_priv, batch_labels = ttv._v3_truth_parts(np.zeros((2, 26)), where="test")
    assert batch_priv.shape == (2, PRIV_DIM) and batch_labels.shape == (2, LABEL_DIM)
    with pytest.raises(ValueError, match="v3 truth snapshot"):
        ttv._v3_truth_parts(row[:24], where="test")
    with pytest.raises(ValueError, match="v3 truth snapshot"):
        ttv._v3_truth_parts(np.zeros((2, 25)), where="test")


def test_dagger_snapshot_carries_real_truth_and_label_blind_encoder(tmp_path: Path) -> None:
    sidecar = _sidecar(tmp_path)
    builder = ttv._build_privileged(_cfg(tmp_path, truth_sidecar=sidecar))
    assert builder is not None
    snapshot = ttv._privileged_snapshot(builder, EPISODE_IDS[0], 11, where="test")
    assert snapshot.shape == (26,)
    privileged, labels = ttv._v3_truth_parts(snapshot, where="test")
    assert np.any(privileged) and np.isfinite(privileged).all()  # never zeros(22)
    np.testing.assert_array_equal(privileged, builder.records[0].privileged[11])
    np.testing.assert_array_equal(labels, builder.records[0].labels[11])
    correction = ttv._expert_label(snapshot)
    np.testing.assert_allclose(correction, [labels[2], -labels[3]])

    before = builder.z_priv(EPISODE_IDS[0]).clone()
    builder.records[0].labels[:] = 9999.0
    assert torch.equal(builder.z_priv(EPISODE_IDS[0]), before), "label slots leaked into z_priv"


def test_zeroed_privileged_truth_fails_loudly(tmp_path: Path) -> None:
    sidecar = _sidecar(tmp_path, zero_priv=True, name="zero.trajectory")
    builder = ttv._build_privileged(_cfg(tmp_path, truth_sidecar=sidecar))
    assert builder is not None
    with pytest.raises(AssertionError, match="all zero"):
        ttv._privileged_snapshot(builder, EPISODE_IDS[0], 11, where="test")


def test_configured_sidecar_without_builder_fails_loudly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ttv, "_UpstreamPrivilegedBuilder", None)
    with pytest.raises(RuntimeError, match="unavailable"):
        ttv._build_privileged(_cfg(tmp_path, truth_sidecar=tmp_path / "truth.bin"))


def test_privileged_stage_refuses_a_configured_sidecar_with_no_truth_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ttv, "load_episodes", lambda _path: _episodes())
    monkeypatch.setattr(ttv, "_build_privileged", lambda _cfg: None)
    with pytest.raises(AssertionError, match="consumed 0 truth rows"):
        ttv.run(_cfg(tmp_path, stage="privileged", truth_sidecar=tmp_path / "truth.bin"))


def test_pipeline_wires_sidecar_into_privileged_stage_and_dagger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ttv, "load_episodes", lambda _path: _episodes())
    sidecar = _sidecar(tmp_path)
    metrics = ttv.run(_cfg(tmp_path, stage="all", truth_sidecar=sidecar))

    privileged = metrics.stages["privileged"]["details"]
    assert privileged["privileged_batches"] > 0
    assert privileged["privileged_source"] == "privileged_teacher"
    assert privileged["truth_sidecar"] == str(sidecar)
    assert privileged["eval_label_blind"] is True

    dagger = metrics.stages["dagger"]["details"]
    assert "truth_sidecar" in dagger["label_source"]
    assert dagger["privileged_fields_zeroed"] is False
    assert dagger["truth_sidecar"] == str(sidecar)
    assert dagger["eval_label_blind"] is True


def test_predict_never_reads_the_label(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    model = ttv._build_teacher(cfg, torch.device("cpu"))
    model.eval()
    window = torch.randn(2, 12, 14)
    cond = torch.randn(2, 6)
    target_a = torch.zeros(2, 2)
    target_b = torch.full((2, 2), 7.0)
    command_a, logit_a, error_a = ttv._predict(
        model, window, cond, target_a, cfg=cfg, student=False
    )
    command_b, logit_b, error_b = ttv._predict(
        model, window, cond, target_b, cfg=cfg, student=False
    )
    assert torch.equal(command_a, command_b)
    assert torch.equal(logit_a, logit_b)
    assert not torch.allclose(error_a, error_b), "eval error must still be scored on labels"


def test_engine_eval_guard_rejects_privileged_or_label_columns() -> None:
    engine.assert_eval_label_blind(torch.zeros(2, 3, 14), torch.zeros(2, 6))
    engine.assert_eval_label_blind(torch.zeros(1, 5, 14), torch.zeros(1, 6))
    with pytest.raises(ValueError, match="sensor features"):
        engine.assert_eval_label_blind(torch.zeros(2, 3, 22), torch.zeros(2, 6))
    with pytest.raises(ValueError, match="sensor features"):
        engine.assert_eval_label_blind(torch.zeros(2, 3, 26), torch.zeros(2, 6))
    with pytest.raises(ValueError, match="conditioning"):
        engine.assert_eval_label_blind(torch.zeros(2, 3, 14), torch.zeros(2, 22))
