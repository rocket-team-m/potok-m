# SPDX-License-Identifier: GPL-3.0-or-later
"""SHIELD02 gates for the typed residual policy and its teacher-stage seam."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from potok_ai import train_teacher_v2 as ttv
from potok_ai.data import make_smart_episode
from potok_ai.physics import final_command, physics_baseline
from potok_ai.residual_policy import GateReason, ResidualConfig, ResidualDecision, ResidualPolicy
from potok_ai.student import CommandEnvelope


def _policy(gain: float = 0.0) -> ResidualPolicy:
    policy = ResidualPolicy()
    if gain:
        with torch.no_grad():
            policy.gain.fill_(gain)
    return policy


def _episodes() -> list:
    return [make_smart_episode(seed=100 + i, frames=24, scenario_id=i) for i in range(4)]


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


def test_zero_at_init_is_the_classical_controller() -> None:
    policy = _policy()
    assert float(policy.gain.detach()) == 0.0
    baseline = torch.tensor([[0.25, -0.5], [1.0, -1.0]])
    correction = torch.randn(2, 2, requires_grad=True)
    decision = policy(baseline, correction)
    assert isinstance(decision, ResidualDecision)
    assert torch.equal(decision.command, baseline)
    assert torch.equal(decision.correction, torch.zeros_like(baseline))
    assert decision.reasons == (GateReason.ZERO_GAIN, GateReason.ZERO_GAIN)
    decision.command.sum().backward()
    assert correction.grad is not None and torch.isfinite(correction.grad).all()


def test_superposition_and_gradients_flow() -> None:
    policy = _policy(gain=0.5)
    baseline = torch.zeros(1, 2)
    correction = torch.tensor([[0.6, -0.4]], requires_grad=True)
    decision = policy(baseline, correction)
    assert torch.allclose(decision.command, torch.tensor([[0.3, -0.2]]))
    assert decision.reasons == (GateReason.OK,)
    decision.command.sum().backward()
    assert float(policy.gain.grad) == pytest.approx(0.2)
    assert correction.grad is not None
    assert torch.allclose(correction.grad, torch.full((1, 2), 0.5))


def test_envelope_projects_command_and_names_clamp() -> None:
    policy = _policy(gain=1.0)
    baseline = torch.tensor([[2.0, 0.0]])
    correction = torch.tensor([[100.0, -100.0]])
    limit = CommandEnvelope().accel
    decision = policy(baseline, correction)
    assert torch.allclose(decision.command, torch.tensor([[limit, -limit]]))
    assert torch.allclose(decision.correction, torch.tensor([[limit - 2.0, -limit]]))
    assert decision.reasons == (GateReason.CLAMPED,)


def test_nonfinite_correction_falls_back_to_baseline() -> None:
    policy = _policy(gain=1.0)
    baseline = torch.tensor([[0.5, -0.5]])
    correction = torch.tensor([[float("nan"), float("inf")]])
    decision = policy(baseline, correction)
    assert torch.equal(decision.command, baseline)
    assert torch.equal(decision.correction, torch.zeros_like(baseline))
    assert decision.reasons == (GateReason.NON_FINITE,)


def test_disabled_is_a_pure_classical_fallback() -> None:
    policy = _policy(gain=1.0)
    baseline = torch.tensor([[0.5, -0.5]])
    decision = policy(baseline, torch.tensor([[9.0, -9.0]]), enabled=False)
    assert torch.equal(decision.command, baseline)
    assert torch.equal(decision.correction, torch.zeros_like(baseline))
    assert decision.reasons == (GateReason.DISABLED,)


def test_typed_config_and_shape_validation() -> None:
    with pytest.raises(TypeError, match="CommandEnvelope"):
        ResidualConfig(envelope=10.0)  # type: ignore[arg-type]
    assert ResidualConfig().envelope.accel == CommandEnvelope().accel
    policy = _policy(gain=1.0)
    with pytest.raises(ValueError, match="same shape"):
        policy(torch.zeros(2, 2), torch.zeros(3, 2))
    with pytest.raises(ValueError, match="two-channel"):
        policy(torch.zeros(2, 3), torch.zeros(2, 3))


def test_reason_histogram_counts_and_resets() -> None:
    policy = _policy()
    baseline = torch.zeros(1, 2)
    policy(baseline, torch.ones(1, 2))
    policy(baseline, torch.tensor([[float("nan"), 0.0]]))
    counts = policy.reason_counts()
    assert counts["zero_gain"] == 1 and counts["non_finite"] == 1
    policy.reset_reason_counts()
    assert policy.reason_counts() == {reason.value: 0 for reason in GateReason}


def test_teacher_default_on_and_legacy_flag(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    assert cfg.residual_policy is True
    model = ttv._build_teacher(cfg, torch.device("cpu"))
    assert isinstance(model.residual_policy, ResidualPolicy)
    window = torch.randn(2, 12, 14)
    correction = torch.randn(2, 2)
    assert torch.equal(
        ttv._command(model, window, correction, residual=cfg.residual_policy),
        physics_baseline(window[:, -1]),
    )
    legacy = ttv._build_teacher(_cfg(tmp_path, residual_policy=False), torch.device("cpu"))
    assert not hasattr(legacy, "residual_policy")
    assert torch.equal(
        ttv._command(legacy, window, correction, residual=False),
        final_command(correction, physics_baseline(window[:, -1]), residual=True),
    )


def test_stage_metrics_record_gate_reasons(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ttv, "load_episodes", lambda path: _episodes())
    cfg = _cfg(tmp_path, stage="sft")
    metrics = ttv.run(cfg)
    info = metrics.residual_policy
    assert info["enabled"] is True
    assert info["source"] == "residual_policy"
    assert sum(info["gate_reasons"].values()) > 0
    assert metrics.module_sources["residual_policy"] == "residual_policy"
    payload = json.loads((cfg.out / "metrics.json").read_text())
    assert payload["residual_policy"] == info
    legacy = ttv.run(_cfg(tmp_path / "legacy", stage="sft", residual_policy=False))
    assert legacy.residual_policy == {"enabled": False, "source": "legacy"}
    assert legacy.config_hash != metrics.config_hash


def test_parser_flag_keeps_legacy_teacher() -> None:
    base = ["--data", "d", "--out", "o"]
    default = ttv.TeacherV2Config.from_args(ttv.build_parser().parse_args(base))
    legacy = ttv.TeacherV2Config.from_args(
        ttv.build_parser().parse_args([*base, "--no-residual-policy"])
    )
    assert default.residual_policy is True
    assert legacy.residual_policy is False
