# SPDX-License-Identifier: GPL-3.0-or-later
"""AIOBJ gates for the privileged impact objective (training-only, no flight claim)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from potok_ai import train_teacher_v2 as ttv
from potok_ai.data import make_smart_episode
from potok_ai.objective import (
    A_LAT_RESID_COL,
    DRAG_EST_COL,
    FRAME_DT_S,
    G0_MPS2,
    IMPACT_FORMULA,
    TRUTH_DIM,
    TRUTH_VEL_START,
    ImpactObjective,
    ImpactObjectiveConfig,
    flight_path_angle_rad,
)


def _batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    window = torch.zeros(2, 4, 14)
    window[:, -1, A_LAT_RESID_COL] = torch.tensor([2.0, -1.0])
    window[:, -1, DRAG_EST_COL] = torch.tensor([3.0, 6.0])
    truth = torch.zeros(2, TRUTH_DIM)
    truth[:, TRUTH_VEL_START : TRUTH_VEL_START + 3] = torch.tensor(
        [[100.0, 20.0, 50.0], [80.0, 0.0, 30.0]]
    )
    command = torch.zeros(2, 2)
    return command, window, truth


def test_flight_path_angle_units_and_zero_speed_refusal() -> None:
    velocity = torch.tensor([[3.0, 4.0, 0.0], [1e-6, 0.0, 1.0]])
    gamma = flight_path_angle_rad(velocity)
    assert gamma.shape == (2,)
    assert float(gamma[0]) == pytest.approx(0.0)
    assert float(gamma[1]) == pytest.approx(np.pi / 2, abs=1e-5)
    with pytest.raises(ValueError, match="horizontal speed is zero"):
        flight_path_angle_rad(torch.zeros(1, 3))
    with pytest.raises(ValueError, match="finite"):
        flight_path_angle_rad(torch.tensor([[float("nan"), 1.0, 0.0]]))
    with pytest.raises(ValueError, match="3-axis"):
        flight_path_angle_rad(torch.zeros(2, 2))


def test_impact_loss_has_explicit_unit_normalisation_and_unit_weights() -> None:
    command, window, truth = _batch()
    objective = ImpactObjective()
    terms = objective.loss(command, window, truth)
    g0 = G0_MPS2
    vx, vy, vz = truth[:, TRUTH_VEL_START : TRUTH_VEL_START + 3].unbind(-1)
    gamma = torch.atan2(vz, torch.hypot(vx + command[:, 1] * FRAME_DT_S, vy))
    lateral = (window[:, -1, A_LAT_RESID_COL] / g0).square()
    flight = (window[:, -1, DRAG_EST_COL] / g0) * gamma.square()
    assert torch.allclose(terms.loss, (lateral + flight).mean(), atol=1e-7)
    assert torch.allclose(terms.lateral, lateral.mean(), atol=1e-7)
    assert torch.allclose(terms.flight_path, flight.mean(), atol=1e-7)
    assert float(terms.gamma_rad) > 0.0
    assert "g0" in IMPACT_FORMULA and "drag_est" in IMPACT_FORMULA


def test_lateral_residual_is_cancelled_by_matching_command() -> None:
    command, window, truth = _batch()
    command[:, 0] = window[:, -1, A_LAT_RESID_COL]
    before = ImpactObjective().loss(torch.zeros_like(command), window, truth)
    after = ImpactObjective().loss(command, window, truth)
    assert float(after.lateral) == pytest.approx(0.0, abs=1e-12)
    assert float(after.lateral) < float(before.lateral)


def test_command_changes_gamma_and_loss_is_differentiable() -> None:
    command, window, truth = _batch()
    command = command.clone().requires_grad_(True)
    objective = ImpactObjective()
    zero = objective.loss(command, window, truth).loss
    steered = command.clone()
    steered[:, 0] += 50.0  # a large lateral demand rotates the next-step velocity
    moved = objective.loss(steered, window, truth).loss
    assert float(moved.detach()) != pytest.approx(float(zero.detach()))
    zero.backward()
    assert command.grad is not None and torch.isfinite(command.grad).all()
    assert float(command.grad.abs().sum()) > 0.0


def test_objective_reads_velocity_slot_not_body_rate_slot() -> None:
    """Privileged velocity is at v3 slots 3..5, not the old 11..14 offset.

    Permuting the velocity slot must move the objective; disturbing the slot the
    old code wrongly read (omega_y/omega_z/mass) must leave it untouched. This
    falsifies the off-by-slot defect that made AIOBJ read body rate as velocity.
    """
    body_rate_slice = slice(11, 14)  # omega_y, omega_z, mass_kg - the old wrong read

    def truth_with(velocity: list[float], body_rate: list[float]) -> torch.Tensor:
        row = torch.zeros(1, TRUTH_DIM)
        row[0, TRUTH_VEL_START : TRUTH_VEL_START + 3] = torch.tensor(velocity)
        row[0, body_rate_slice] = torch.tensor(body_rate)
        return row

    # command=0 and a_lat_resid=0 isolate gamma; drag>0 keeps the flight term live.
    window = torch.zeros(1, 2, 14)
    window[0, -1, DRAG_EST_COL] = 5.0
    command = torch.zeros(1, 2)
    objective = ImpactObjective()

    base_truth = truth_with([3.0, 4.0, 5.0], [10.0, 0.0, 1.0])
    base = float(objective.loss(command, window, base_truth).loss)
    assert base > 0.0

    # Permuting two velocity components changes hypot(vx,vy) and vz, so gamma moves.
    permuted = truth_with([3.0, 5.0, 4.0], [10.0, 0.0, 1.0])
    after_velocity_perm = float(objective.loss(command, window, permuted).loss)
    assert abs(after_velocity_perm - base) > 1e-3

    # Disturbing the body-rate/mass slot the old code read must NOT move the loss.
    perturbed = truth_with([3.0, 4.0, 5.0], [999.0, -999.0, 123.0])
    after_body_rate_perm = float(objective.loss(command, window, perturbed).loss)
    assert after_body_rate_perm == pytest.approx(base, abs=1e-12)


def test_impact_refuses_unphysical_or_misshaped_inputs() -> None:
    command, window, truth = _batch()
    objective = ImpactObjective()
    with pytest.raises(ValueError, match="shape"):
        objective.loss(command[:, :1], window, truth)
    with pytest.raises(ValueError, match="shape"):
        objective.loss(command, window[..., :13], truth)
    with pytest.raises(ValueError, match="shape"):
        objective.loss(command, window, truth[:, :21])
    bad_truth = truth.clone()
    bad_truth[0, TRUTH_VEL_START] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        objective.loss(command, window, bad_truth)
    negative_drag = window.clone()
    negative_drag[0, -1, DRAG_EST_COL] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        objective.loss(command, negative_drag, truth)
    with pytest.raises(ValueError, match="positive"):
        ImpactObjectiveConfig(gravity_mps2=0.0)


def test_supervised_loss_adds_impact_term_only_when_wired() -> None:
    cfg = ttv.TeacherV2Config(data=Path("d"), out=Path("o"))
    model = ttv._build_teacher(cfg, torch.device("cpu"))
    window = torch.randn(2, 12, 14)
    window[:, -1, DRAG_EST_COL] = 2.0  # cd0*q_dyn*area/mass is non-negative
    cond = torch.randn(2, 6)
    target = torch.randn(2, 2)
    truth = torch.zeros(2, TRUTH_DIM)
    truth[:, TRUTH_VEL_START : TRUTH_VEL_START + 3] = torch.tensor([100.0, 20.0, 50.0])
    plain = ttv._supervised_loss(model, (window, cond, target, [0, 1]), cfg)
    wired = ttv._supervised_loss(
        model,
        (window, cond, target, [0, 1]),
        cfg,
        impact=ImpactObjective(),
        privileged_truth=truth,
    )
    assert plain["impact"] == 0.0
    assert wired["impact"] > 0.0
    assert wired["loss"] > plain["loss"]
    with pytest.raises(ValueError, match="one privileged truth row"):
        ttv._supervised_loss(model, (window, cond, target, [0, 1]), cfg, impact=ImpactObjective())


class _TruthRecord:
    def __init__(self, index: int, frames: int) -> None:
        self.episode_index = index
        privileged = np.zeros((frames, TRUTH_DIM), dtype=np.float32)
        privileged[:, TRUTH_VEL_START : TRUTH_VEL_START + 3] = [100.0, 20.0, 50.0]
        self.privileged = privileged
        self.labels = np.zeros((frames, 4), dtype=np.float32)


class _TruthBuilder:
    def __init__(self, episodes: list) -> None:
        self.encoder = SimpleNamespace(z_dim=3)
        self.records = [_TruthRecord(int(e.index), len(e.frames)) for e in episodes]

    def z_priv(self, idx: int) -> torch.Tensor:
        record = next(r for r in self.records if r.episode_index == idx)
        return torch.as_tensor(record.privileged[:, TRUTH_VEL_START : TRUTH_VEL_START + 3]).clone()


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
        "limit": 6,
    }
    kwargs.update(overrides)
    return ttv.TeacherV2Config(**kwargs)


def test_privileged_stage_defaults_to_impact_and_legacy_is_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episodes = _episodes()
    monkeypatch.setattr(ttv, "load_episodes", lambda _path: episodes)
    monkeypatch.setattr(ttv, "_build_privileged", lambda cfg: _TruthBuilder(episodes))
    impact = ttv.run(_cfg(tmp_path, out=tmp_path / "impact"))
    legacy = ttv.run(_cfg(tmp_path, out=tmp_path / "legacy", legacy_objective=True))
    impact_details = impact.stages["privileged"]["details"]["objective"]
    legacy_details = legacy.stages["privileged"]["details"]["objective"]
    assert impact_details["source"] == "impact"
    assert impact_details["legacy"] is False
    assert impact_details["formula"] == IMPACT_FORMULA
    assert legacy_details["source"] == "legacy"
    assert legacy_details["reason"] == "legacy_objective"
    assert legacy_details["formula"] is None
    assert ttv.config_hash(_cfg(tmp_path)) != ttv.config_hash(_cfg(tmp_path, legacy_objective=True))


def test_no_truth_sidecar_records_falls_back_loudly_in_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episodes = _episodes()
    monkeypatch.setattr(ttv, "load_episodes", lambda _path: episodes)
    metrics = ttv.run(_cfg(tmp_path))
    objective = metrics.stages["privileged"]["details"]["objective"]
    assert objective["source"] == "legacy"
    assert objective["reason"] == "no_truth_sidecar_records"


def test_cli_flag_toggles_legacy_objective() -> None:
    base = ttv.build_parser().parse_args(["--data", "d", "--out", "o"])
    assert ttv.TeacherV2Config.from_args(base).legacy_objective is False
    flagged = ttv.build_parser().parse_args(["--data", "d", "--out", "o", "--legacy-objective"])
    assert ttv.TeacherV2Config.from_args(flagged).legacy_objective is True
