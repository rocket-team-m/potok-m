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
    ACTIVE_PHASES,
    FORBIDDEN_PHASES,
    BrakePhase,
    BrakeReason,
    BrakingConfig,
    BrakingTeacher,
    FlightPhase,
    TrueState,
    flight_phase,
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


def test_phase_gate_is_zero_on_banned_phases_and_active_after_burnout() -> None:
    cfg = BrakingConfig()
    banned = (FlightPhase.PAD, FlightPhase.BOOST, FlightPhase.MOTOR_ARMED)
    active = (FlightPhase.COAST, FlightPhase.DESCENT)
    assert FORBIDDEN_PHASES >= set(banned)
    assert ACTIVE_PHASES == set(active)
    assert not (FORBIDDEN_PHASES & ACTIVE_PHASES)
    assert flight_phase(0.0, cfg) is FlightPhase.PAD
    assert flight_phase(5.0, cfg) is FlightPhase.BOOST  # motor armed
    assert flight_phase(12.0, cfg, vz=10.0) is FlightPhase.COAST
    assert flight_phase(12.0, cfg, vz=-10.0) is FlightPhase.DESCENT

    phases = (*banned, *active)
    state = TrueState(
        position=np.tile([755.0, 0.0, 1200.0], (len(phases), 1)),
        velocity=np.tile([155.0, 0.0, 10.0], (len(phases), 1)),
        time_s=np.full(len(phases), 12.0),
    )
    decision = BrakingTeacher().demand(state, flight_phases=phases)
    assert decision.flight_phases == phases
    np.testing.assert_allclose(decision.command[: len(banned)], 0.0)
    assert np.count_nonzero(np.any(decision.command[len(banned) :] != 0.0, axis=1)) == len(active)
    # The inverse gate would zero the active rows and light up the banned ones.
    inverted = BrakingTeacher().demand(state, flight_phases=tuple(reversed(phases)))
    np.testing.assert_allclose(inverted.command[len(active) :], 0.0)


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


class _BrakingTruthRecord:
    """600 frames (12 s): ticks 0..497 are boost, 498..599 are post-burnout."""

    def __init__(self, index: int, frames: int) -> None:
        self.episode_index = index
        privileged = np.zeros((frames, 22), dtype=np.float32)
        privileged[:, 0] = 755.0
        privileged[:, 3] = 155.0  # outward radial speed: cap-clamped when active
        privileged[:, 2] = 1200.0
        privileged[:, 5] = 10.0
        privileged[:, 11:14] = (100.0, 20.0, 50.0)
        privileged[:, 15] = 5.0
        self.privileged = privileged
        self.labels = np.tile(np.asarray([0.0, 0.0, 3.0, -4.0], np.float32), (frames, 1))


class _BrakingTruthBuilder:
    def __init__(self, episodes: list, frames: int) -> None:
        self.encoder = SimpleNamespace(z_dim=3)
        self.records = [_BrakingTruthRecord(int(e.index), frames) for e in episodes]

    def z_priv(self, index: int) -> torch.Tensor:
        record = next(r for r in self.records if r.episode_index == index)
        return torch.as_tensor(record.privileged[:, :3]).clone()


def test_privileged_target_is_active_on_post_burnout_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """BRK03 falsification: the old per-batch tick counter boost-gated 1451/1451."""
    frames, window, batch = 600, 12, 64
    episodes = [make_smart_episode(seed=100 + i, frames=frames, scenario_id=i) for i in range(4)]
    monkeypatch.setattr(ttv, "load_episodes", lambda _path: episodes)
    monkeypatch.setattr(
        ttv, "_build_privileged", lambda cfg: _BrakingTruthBuilder(episodes, frames)
    )
    cfg = _cfg(tmp_path, out=tmp_path / "post_burnout", window=window, batch=batch, limit=1200)
    fit, _inner, _held = ttv.partition_episodes(episodes, seed=cfg.seed)
    n_windows = sum(frames - window + 1 for _e in fit)  # 2 * 589 = 1178 <= limit
    burn_tick = int(BrakingConfig().boost_end_s / BrakingConfig().frame_dt_s)  # 497
    # Window n of an episode ends at tick window-1+n, so post-burnout n >= 487.
    post_burnout = sum(max(0, frames - window + 1 - (burn_tick + 2 - window)) for _e in fit)

    cursor = ttv._WindowTickCursor()
    ids = [int(e.index) for e in fit for _ in range(frames - window + 1)]
    ticks = cursor.ticks(ids, cfg, where="falsification")
    assert ticks[batch] == ticks[0] + batch  # no per-batch reset
    assert ticks[-1] == frames - 1
    target = ttv._braking_target_rows(
        _BrakingTruthBuilder(episodes, frames),
        ids,
        ticks,
        cfg,
        BrakingTeacher(),
        where="falsification",
    ).numpy()
    boost = ticks * BrakingConfig().frame_dt_s <= BrakingConfig().boost_end_s
    post = ~boost
    assert int(boost.sum()) + int(post.sum()) == n_windows
    assert int(post.sum()) == post_burnout
    np.testing.assert_allclose(target[boost], 0.0)
    assert np.count_nonzero(np.any(target[post] != 0.0, axis=1)) >= 0.8 * post.sum()

    metrics = ttv.run(cfg)
    counts = metrics.stages["privileged"]["details"]["braking"]["reason_counts"]
    active = counts.get("ok", 0) + counts.get("cap_clamped", 0)
    assert counts.get("boost_gate", 0) > 0  # banned frames are counted and zero
    assert active >= 0.8 * post_burnout  # BRK02's 0-active defect cannot recur
