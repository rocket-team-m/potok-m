# SPDX-License-Identifier: GPL-3.0-or-later
"""Typed online shield in the RL/env training path (RES-SHIELD, SHIELD01).

The shield is a filter on the candidate command, not a reward change: every
veto falls back to the deterministic classical baseline with a typed reason
and is counted. These tests cover the pure verifier and its wiring into the
stateful student's training step.
"""

import json
import math

import pytest
import torch
from test_engine import _argv, _data

from potok_ai import engine
from potok_ai.data import group_by_scenario, load_episodes


def shield(mach_max=0.6, authority=10.0):
    return engine.OnlineShield(mach_max=mach_max, authority=authority)


def candidate():
    return torch.tensor([2.0, -1.0])


def baseline():
    return torch.tensor([0.25, -0.5])


def test_reasons_are_typed_and_counts_shape_is_fixed():
    assert [reason.value for reason in engine.ShieldReason] == [
        "mach_unavailable", "mach_limit", "authority", "phase_forbidden",
    ]
    counts = engine.ShieldCounts()
    assert counts.as_dict() == dict.fromkeys(counts.as_dict(), 0)
    assert counts.interventions == 0


def test_accepted_command_passes_through_bitwise():
    s = shield()
    c, b = candidate(), baseline()
    decision = s.verify(c, b, engine.ShieldSample(mach=0.5, phase=1))
    assert isinstance(decision, engine.ShieldDecision)
    assert decision.reason is None and not decision.intervened
    assert torch.equal(decision.command, c)
    assert s.counts.accepted == 1 and s.counts.interventions == 0


@pytest.mark.parametrize(
    "sample,command,expected",
    [
        (engine.ShieldSample(mach=0.9, phase=1), candidate(), engine.ShieldReason.MACH_LIMIT),
        (engine.ShieldSample(mach=0.1, phase=0), candidate(), engine.ShieldReason.PHASE_FORBIDDEN),
        (engine.ShieldSample(mach=math.nan, phase=1), candidate(), engine.ShieldReason.MACH_UNAVAILABLE),
        (engine.ShieldSample(mach=0.1, phase=3), torch.tensor([11.0, 0.0]), engine.ShieldReason.AUTHORITY),
        (engine.ShieldSample(mach=0.1, phase=2), torch.tensor([float("nan"), 0.0]), engine.ShieldReason.AUTHORITY),
    ],
)
def test_each_constraint_returns_baseline_with_a_reason_and_is_counted(
    sample, command, expected
):
    s = shield()
    c, b = command, baseline()
    decision = s.verify(c, b, sample)
    assert decision.reason is expected
    assert decision.intervened
    assert torch.equal(decision.command, b)
    assert not torch.equal(decision.command, c)
    assert s.counts.as_dict()["accepted"] == 0
    assert s.counts.as_dict()[expected.value] == 1
    assert s.counts.interventions == 1


def test_check_priority_is_fixed_and_mach_evidence_fails_safe():
    s = shield()
    # Every constraint violated at once: the first check wins deterministically.
    decision = s.verify(
        torch.tensor([float("inf"), 0.0]),
        baseline(),
        engine.ShieldSample(mach=float("inf"), phase=0),
    )
    assert decision.reason is engine.ShieldReason.MACH_UNAVAILABLE
    assert torch.equal(decision.command, baseline())


def test_shield_rejects_malformed_limits_and_mismatched_shapes():
    with pytest.raises(ValueError, match="Mach ceiling"):
        shield(mach_max=math.nan)
    with pytest.raises(ValueError, match="Mach ceiling"):
        shield(mach_max=-1.0)
    with pytest.raises(ValueError, match="authority"):
        shield(authority=0.0)
    with pytest.raises(ValueError, match="must match"):
        shield().verify(torch.zeros(2), torch.zeros(3), engine.ShieldSample(0.1, 1))


def test_limits_are_untouched_by_the_shield():
    s = shield(mach_max=0.6, authority=10.0)
    assert s.mach_max == 0.6
    assert s.authority == 10.0
    # The certified Mach ceiling is a config field, not a tuned reward gate.
    assert engine.OFFLINE_MACH == 0.0 and engine.OFFLINE_PHASE == 1


def _adapter_and_batch(tmp_path, monkeypatch):
    cfg = engine.TrainConfig.from_args(
        engine.build_parser().parse_args(
            _argv("lrs", _data(tmp_path), tmp_path / "shielded")
        )
    )
    adapter = engine.LRSAdapter()
    model = adapter.build(cfg, torch.device("cpu"))
    groups = group_by_scenario(load_episodes(cfg.data))
    held = engine._held_out(groups)
    episodes = [e for name, g in groups.items() if name not in held for e in g]
    batch = next(adapter.batches(episodes, cfg))
    return cfg, adapter, model, batch


def test_training_step_falls_back_to_baseline_and_counts_interventions(tmp_path, monkeypatch):
    cfg, adapter, model, batch = _adapter_and_batch(tmp_path, monkeypatch)

    seen = {}

    defSpy_candidate = engine.final_command

    def spy_final(prediction, base, residual=True):
        out = defSpy_candidate(prediction, base, residual)
        seen["current"] = (out.detach().clone(), base.detach().clone())
        return out

    real_mse = engine.F.mse_loss

    def spy_mse(command, reference, *args, **kwargs):
        if "command" not in seen:
            candidate_tick, baseline_tick = seen["current"]
            seen["candidate"] = candidate_tick
            seen["baseline"] = baseline_tick
            seen["command"] = command.detach().clone()
        return real_mse(command, reference, *args, **kwargs)

    monkeypatch.setattr(engine, "final_command", spy_final)
    monkeypatch.setattr(engine.F, "mse_loss", spy_mse)

    # Mach over the ceiling: every supervised tick must be vetoed.
    adapter.shield_sample = engine.ShieldSample(mach=0.99, phase=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss = adapter.train_step(model, batch, optimizer, cfg)

    assert math.isfinite(loss)
    assert "command" in seen and "candidate" in seen
    assert torch.equal(seen["command"], seen["baseline"])
    assert not torch.equal(seen["command"], seen["candidate"])
    counts = adapter.shield.counts
    assert counts.mach_limit > 0 and counts.accepted == 0
    assert counts.interventions == counts.mach_limit


def test_training_step_accepts_a_clean_sample(tmp_path, monkeypatch):
    cfg, adapter, model, batch = _adapter_and_batch(tmp_path, monkeypatch)
    adapter.shield_sample = engine.ShieldSample(mach=0.1, phase=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss = adapter.train_step(model, batch, optimizer, cfg)
    assert math.isfinite(loss)
    counts = adapter.shield.counts
    assert counts.accepted > 0 and counts.interventions == 0


def test_training_run_writes_the_shield_report(tmp_path):
    data, out = _data(tmp_path), tmp_path / "run"
    engine.main(_argv("lrs", data, out))
    report = json.loads((out / "shield.json").read_text())
    assert tuple(report) == tuple(engine.ShieldCounts().as_dict())
    assert sum(report.values()) > 0
    assert all(isinstance(value, int) and value >= 0 for value in report.values())
