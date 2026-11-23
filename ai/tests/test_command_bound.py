# SPDX-License-Identifier: GPL-3.0-or-later
"""AITRAINIDEAL B1 gate: teacher/student command bound covers the v3 target.

The pre-v3 student head emitted at most +/-1 m/s^2 against a target with RMS
7.70 and max 10.0 m/s^2. These gates pin the typed bound, the explicit
default-off legacy escape, and the loud refusal on data beyond the bound.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from potok_ai import train_teacher_v2 as ttv
from potok_ai.data import make_smart_episode
from potok_ai.gdn2_student import GDN2Student, GDN2StudentConfig
from potok_ai.residual_policy import (
    LEGACY_BOUND_MPS2,
    TARGET_MAX_MPS2,
    TARGET_RMS_MPS2,
    ResidualConfig,
    assert_command_bound_covers,
)
from potok_ai.student import CommandEnvelope


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


def _episodes(frames: int = 24, command: float | None = None) -> list:
    result = []
    for index in range(4):
        episode = make_smart_episode(seed=100 + index, frames=frames, scenario_id=index)
        if command is not None:
            g_accel = np.full((frames, 2), command, dtype=np.float32)
            episode = replace(episode, g_accel=g_accel)
        result.append(episode)
    return result


def test_default_student_bound_is_the_typed_envelope() -> None:
    cfg = GDN2StudentConfig()
    assert TARGET_RMS_MPS2 == 7.70
    assert cfg.g_max == TARGET_MAX_MPS2 == 10.0
    assert cfg.g_max == CommandEnvelope().accel
    assert cfg.legacy_bound is False

    model = GDN2Student(cfg).eval()
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.copy_(torch.tensor([9.0, 9.0, 0.0]))
        corrections, _ = model.step(torch.randn(14), None)
    assert float(corrections.abs().max()) > 1.0  # the legacy head could not do this
    assert float(corrections.abs().max()) <= cfg.g_max + 1e-6


def test_legacy_bound_is_explicit_and_default_off(tmp_path: Path) -> None:
    assert GDN2StudentConfig().legacy_bound is False
    assert GDN2StudentConfig(legacy_bound=True).g_max == LEGACY_BOUND_MPS2 == 1.0

    assert ttv._student_bound(_cfg(tmp_path)) == TARGET_MAX_MPS2
    assert ttv._student_bound(_cfg(tmp_path, legacy_student_bound=True)) == LEGACY_BOUND_MPS2
    default = ttv._build_student(_cfg(tmp_path), torch.device("cpu"))
    legacy = ttv._build_student(
        _cfg(tmp_path, legacy_student_bound=True), torch.device("cpu")
    )
    assert default.cfg.g_max == TARGET_MAX_MPS2  # type: ignore[attr-defined]
    assert legacy.cfg.g_max == LEGACY_BOUND_MPS2  # type: ignore[attr-defined]

    base = ["--data", "d", "--out", "o"]
    default_cfg = ttv.TeacherV2Config.from_args(ttv.build_parser().parse_args(base))
    assert default_cfg.legacy_student_bound is False
    opt_in = ttv.TeacherV2Config.from_args(
        ttv.build_parser().parse_args([*base, "--legacy-student-bound"])
    )
    assert opt_in.legacy_student_bound is True


def test_bound_assert_refuses_a_narrow_or_out_of_range_config() -> None:
    with pytest.raises(AssertionError, match="cannot represent the v3 target"):
        assert_command_bound_covers(LEGACY_BOUND_MPS2)
    with pytest.raises(AssertionError, match="below observed target max"):
        assert_command_bound_covers(TARGET_MAX_MPS2, target=torch.tensor([[12.0, 0.0]]))
    with pytest.raises(ValueError, match="finite and positive"):
        assert_command_bound_covers(float("nan"))
    assert_command_bound_covers(TARGET_MAX_MPS2, target=torch.tensor([[10.0, -7.7]]))
    assert_command_bound_covers(TARGET_MAX_MPS2, target=torch.tensor([[float("nan"), 0.0]]))


def test_teacher_residual_config_uses_the_envelope_and_refuses_narrow() -> None:
    assert ResidualConfig().envelope.accel == CommandEnvelope().accel == TARGET_MAX_MPS2
    with pytest.raises(AssertionError, match="teacher command envelope"):
        ResidualConfig(envelope=CommandEnvelope(LEGACY_BOUND_MPS2))


def test_distill_refuses_target_beyond_student_bound(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ttv, "load_episodes", lambda path: _episodes(command=12.0))
    with pytest.raises(AssertionError, match="below observed target max"):
        ttv.run(_cfg(tmp_path, stage="distill"))
    # The same data is allowed when the legacy head is explicitly requested.
    metrics = ttv.run(_cfg(tmp_path / "legacy", stage="distill", legacy_student_bound=True))
    details = metrics.stages["distill"]["details"]
    assert details["student_command_bound"] == LEGACY_BOUND_MPS2
    assert details["legacy_student_bound"] is True


def test_distill_records_the_default_bound(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ttv, "load_episodes", lambda path: _episodes())
    metrics = ttv.run(_cfg(tmp_path, stage="distill"))
    details = metrics.stages["distill"]["details"]
    assert details["student_command_bound"] == TARGET_MAX_MPS2
    assert details["legacy_student_bound"] is False
