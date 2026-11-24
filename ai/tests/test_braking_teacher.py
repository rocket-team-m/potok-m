# SPDX-License-Identifier: GPL-3.0-or-later
"""BRK01 gates: privileged radial-braking reference and its stage wiring."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from potok_ai import train_teacher_v2 as ttv
from potok_ai.braking_teacher import (
    BrakePhase,
    BrakeReason,
    BrakingConfig,
    BrakingTeacher,
    TrueState,
)
from potok_ai.data import make_smart_episode


def _state(x: float, vx: float, *, time_s: float = 12.0) -> TrueState:
    return TrueState(
        position=np.array([[x, 0.0, 1197.0]]),
        velocity=np.array([[vx, 0.0, 169.0]]),
        time_s=np.array([time_s]),
    )


def test_stopping_distance_coefficients_are_derived_not_fitted() -> None:
    cfg = BrakingConfig(cap_mps2=10.0, horizon_s=2.0)
    assert cfg.ref_speed_mps == pytest.approx(20.0)
    assert cfg.stop_distance_m == pytest.approx(20.0)  # v^2 / (2a)
    assert cfg.velocity_gain_inv_s == pytest.approx(1.0 / 2.0)
    assert cfg.position_gain_inv_s2 == pytest.approx(2.0 / 2.0**2)
    with pytest.raises(ValueError, match="horizon_s"):
        BrakingConfig(horizon_s=0.0)


def test_true_state_reads_velocity_slots_and_never_labels() -> None:
    row = np.zeros(26, dtype=np.float32)
    row[:6] = (755.0, 0.0, 1197.0, 155.0, 0.0, 169.0)
    row[22:] = 9999.0
    state = TrueState.from_privileged(row)
    np.testing.assert_allclose(state.position, [[755.0, 0.0, 1197.0]])
    np.testing.assert_allclose(state.velocity, [[155.0, 0.0, 169.0]])
    with pytest.raises(ValueError, match="slots"):
        TrueState.from_privileged(row[:24])


def test_brake_phase_nulls_outward_velocity_at_the_actuator_cap() -> None:
    decision = BrakingTeacher().demand(_state(755.0, 155.0))
    assert decision.phases == (BrakePhase.BRAKE,)
    assert decision.reasons == (BrakeReason.CAP_CLAMPED,)
    np.testing.assert_allclose(decision.radial_velocity, [155.0])
    np.testing.assert_allclose(decision.radial_demand, [-10.0])
    # [ax, ay] = [-u_pitch, u_lat], so an inward -X demand is u_pitch = +10.
    np.testing.assert_allclose(decision.command, [[0.0, 10.0]])


def test_correct_phase_pulls_position_back_after_velocity_is_nulled() -> None:
    decision = BrakingTeacher().demand(_state(10.0, 0.0))
    assert decision.phases == (BrakePhase.CORRECT,)
    assert decision.reasons == (BrakeReason.OK,)
    assert -10.0 < float(decision.radial_demand[0]) < 0.0
    assert float(decision.command[0, 1]) > 0.0


def test_settled_inside_deadband_has_zero_demand() -> None:
    decision = BrakingTeacher().demand(_state(0.5, 0.0))
    assert decision.phases == (BrakePhase.SETTLED,)
    assert decision.reasons == (BrakeReason.SETTLED,)
    np.testing.assert_allclose(decision.command, [[0.0, 0.0]])


def test_no_boost_from_time_gate_or_explicit_interlock() -> None:
    teacher = BrakingTeacher()
    boost = teacher.demand(_state(755.0, 155.0, time_s=9.95))
    assert boost.phases == (BrakePhase.BOOST,)
    assert boost.reasons == (BrakeReason.BOOST_GATE,)
    np.testing.assert_allclose(boost.command, [[0.0, 0.0]])
    vetoed = teacher.demand(_state(755.0, 155.0), ai_allowed=np.array([False]))
    assert vetoed.reasons == (BrakeReason.BOOST_GATE,)
    np.testing.assert_allclose(vetoed.command, [[0.0, 0.0]])


def test_not_in_range_rows_are_zero_with_explicit_reasons() -> None:
    teacher = BrakingTeacher()
    nan_state = _state(755.0, 155.0)
    nan_state.position[0, 0] = float("nan")
    nan = teacher.demand(nan_state)
    zero = teacher.demand(_state(0.0, 0.0))
    far = teacher.demand(_state(20_000.0, 155.0))
    assert nan.reasons == (BrakeReason.NON_FINITE,)
    assert zero.reasons == (BrakeReason.RADIUS_ZERO,)
    assert far.reasons == (BrakeReason.OUT_OF_RANGE,)
    for decision in (nan, zero, far):
        np.testing.assert_allclose(decision.command, [[0.0, 0.0]])
        assert decision.phases == (BrakePhase.OUT_OF_RANGE,)


def test_cap_holds_for_every_row_and_reasons_are_counted() -> None:
    teacher = BrakingTeacher()
    state = TrueState(
        position=np.array([[755.0, 0.0, 1.0], [10.0, 0.0, 1.0], [0.5, 0.0, 1.0], [755.0, 0.0, 1.0]]),
        velocity=np.array([[155.0, 0.0, 10.0], [0.0, 0.0, 10.0], [0.0, 0.0, 10.0], [155.0, 0.0, 10.0]]),
        time_s=np.array([12.0, 12.0, 12.0, 9.0]),
    )
    decision = teacher.demand(state)
    assert np.all(np.abs(decision.command) <= BrakingConfig().cap_mps2)
    assert len(decision.reasons) == 4
    assert sum(teacher.reason_counts().values()) == 4
    teacher.reset_reason_counts()
    assert sum(teacher.reason_counts().values()) == 0


def test_config_is_default_on_with_a_legacy_cli_escape() -> None:
    cfg = ttv.TeacherV2Config(data=Path("d"), out=Path("o"))
    assert cfg.braking_teacher is True
    parser = ttv.build_parser()
    legacy = ttv.TeacherV2Config.from_args(
        parser.parse_args(["--data", "d", "--out", "o", "--no-braking-teacher"])
    )
    explicit = ttv.TeacherV2Config.from_args(
        parser.parse_args(["--data", "d", "--out", "o", "--braking-teacher"])
    )
    assert legacy.braking_teacher is False
    assert explicit.braking_teacher is True
    assert ttv.config_hash(cfg) != ttv.config_hash(legacy)


class _TruthRecord:
    def __init__(self, index: int, frames: int) -> None:
        self.episode_index = index
        privileged = np.zeros((frames, 22), dtype=np.float32)
        privileged[:, 0] = 755.0
        privileged[:, 3] = 155.0
        # The sibling impact objective reads velocity at slots 11..13; keep
        # those finite and non-zero so the privileged stage can run.
        privileged[:, 11:14] = (100.0, 20.0, 50.0)
        privileged[:, 15] = 5.0
        self.privileged = privileged
        self.labels = np.tile(np.asarray([0.0, 0.0, 3.0, -4.0], np.float32), (frames, 1))


class _TruthBuilder:
    def __init__(self, episodes: list) -> None:
        self.encoder = SimpleNamespace(z_dim=3)
        self.records = [_TruthRecord(int(e.index), len(e.frames)) for e in episodes]

    def z_priv(self, index: int) -> torch.Tensor:
        record = next(r for r in self.records if r.episode_index == index)
        return torch.as_tensor(record.privileged[:, :3]).clone()


def _episodes() -> list:
    return [make_smart_episode(seed=100 + i, frames=24, scenario_id=i) for i in range(4)]


def _cfg(tmp_path: Path, **overrides) -> ttv.TeacherV2Config:
    kwargs = {
        "data": tmp_path / "data",
        "out": tmp_path / "out",
        "stage": "privileged",
        "seed": 1,
        "epochs": 1,
        "batch": 2,
        "window": 12,
        "truth_sidecar": tmp_path / "truth.bin",
    }
    kwargs.update(overrides)
    return ttv.TeacherV2Config(**kwargs)


def test_privileged_stage_uses_braking_target_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episodes = _episodes()
    monkeypatch.setattr(ttv, "load_episodes", lambda _path: episodes)
    monkeypatch.setattr(ttv, "_build_privileged", lambda cfg: _TruthBuilder(episodes))
    braking = ttv.run(_cfg(tmp_path, out=tmp_path / "braking"))
    legacy = ttv.run(
        _cfg(tmp_path, out=tmp_path / "legacy", braking_teacher=False)
    )
    active = braking.stages["privileged"]["details"]["braking"]
    old = legacy.stages["privileged"]["details"]["braking"]
    assert active["enabled"] is True
    assert active["source"] == "radial_braking_reference"
    assert sum(active["reason_counts"].values()) > 0
    assert old["enabled"] is False
    assert old["reason"] == "legacy_braking_teacher"


def test_dagger_stage_trains_on_braking_targets_not_recorded_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episodes = _episodes()
    monkeypatch.setattr(ttv, "load_episodes", lambda _path: episodes)
    monkeypatch.setattr(ttv, "_build_privileged", lambda cfg: _TruthBuilder(episodes))
    cfg = _cfg(tmp_path, out=tmp_path / "dagger", stage="dagger")
    fit, inner, held = ttv.partition_episodes(episodes, seed=cfg.seed)
    dataset = ttv._build_dagger_dataset(cfg, [*fit, *inner], held)
    np.testing.assert_array_equal(dataset.samples[0].correction, [3.0, 4.0])
    np.testing.assert_allclose(next(iter(dataset.braking_targets.values())), [0.0, 10.0])

    metrics = ttv.run(cfg)
    details = metrics.stages["dagger"]["details"]
    assert details["braking"]["enabled"] is True
    assert details["training_target"] == "radial_braking_reference"
    legacy = ttv.run(_cfg(tmp_path, out=tmp_path / "old", stage="dagger", braking_teacher=False))
    old = legacy.stages["dagger"]["details"]
    assert old["training_target"] == "recorded_label"
    assert old["braking"]["reason"] == "legacy_braking_teacher"
