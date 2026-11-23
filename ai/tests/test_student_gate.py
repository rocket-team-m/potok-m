# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression gate for the v3 command-envelope fix (MUON02 ranks 1-2).

The shipped LRS recipe fitted a target with RMS 7.70 / max 10.0 m/s^2 using a
head bounded below 1 m/s^2, and trained on two scenario-0 episodes. These
checks pin the typed envelope, its export roundtrip, the full-split recipe and
the loud failure when a target lies outside the reachable range. Gate
thresholds (ratio, ECE) are deliberately NOT changed here.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from potok_ai.data import group_by_scenario, load_episodes, make_smart_episode
from potok_ai.engine import LRSAdapter, TrainConfig
from potok_ai.export import export_student, load_student_export
from potok_ai.student import (
    FRAME_FEATURES,
    CommandEnvelope,
    StatefulTCN,
    StudentConfig,
    smart_student_config,
)

AI = Path(__file__).resolve().parents[1]
V3_DATA = AI / "out" / "v3_final" / "data"


def _config() -> TrainConfig:
    return TrainConfig(model="lrs", stage="train", data=V3_DATA, out=AI / "out")


def _saturate(model: StatefulTCN) -> None:
    with torch.no_grad():
        model.head_corr.weight.zero_()
        model.head_corr.bias.copy_(torch.tensor([100.0, -100.0]))


def test_command_envelope_is_typed_and_defaults_to_v3_limit() -> None:
    assert CommandEnvelope().accel == 10.0
    with pytest.raises(ValueError, match="finite and positive"):
        CommandEnvelope(0.0)
    with pytest.raises(ValueError, match="finite and positive"):
        CommandEnvelope(float("nan"))


def test_student_head_reaches_the_envelope_and_export_preserves_it(tmp_path: Path) -> None:
    envelope = CommandEnvelope()
    model = StatefulTCN(
        smart_student_config(seed=1, g_max=envelope.accel, command_ceiling=envelope.accel)
    ).eval()
    _saturate(model)
    with torch.no_grad():
        output, _ = model.step(np.zeros(20, np.float32))
    assert output[0] == pytest.approx(envelope.accel, abs=1e-4)
    assert output[1] == pytest.approx(-envelope.accel, abs=1e-4)

    # The legacy spec head stays +/-1; the bug was a hard ceiling, not slow fit.
    legacy = StatefulTCN(StudentConfig(seed=1)).eval()
    _saturate(legacy)
    with torch.no_grad():
        old_output, _ = legacy.step(np.zeros(FRAME_FEATURES, np.float32))
    assert old_output.abs().max() <= 1.0 + 1e-6

    path = export_student(model, tmp_path / "lrs.bin")
    exported = load_student_export(path)
    assert exported.g_max == pytest.approx(envelope.accel)
    reloaded = exported.to_model()
    with torch.no_grad():
        reloaded_output, _ = reloaded.step(np.zeros(20, np.float32))
    assert torch.allclose(reloaded_output, output, atol=1e-5)


def test_engine_build_uses_the_typed_envelope() -> None:
    cfg = _config()
    assert cfg.command_envelope.accel == 10.0
    model = LRSAdapter().build(cfg, torch.device("cpu"))
    assert model.cfg.g_max == 10.0
    assert model.cfg.command_ceiling == 10.0


def test_target_outside_envelope_fails_loudly() -> None:
    episode = make_smart_episode(seed=1, frames=12, scenario_id=0)
    episode.g_accel = np.full((12, 2), 20.0, np.float32)
    episode.targets = np.concatenate(
        [np.zeros((12, 4), np.float32), episode.g_accel], axis=1
    )
    with pytest.raises(ValueError, match="exceeds the command envelope"):
        list(LRSAdapter().batches([episode], _config()))


def test_v3_driver_trains_full_split_with_three_seeds() -> None:
    path = AI / "out" / "v3_final" / "driver.py"
    spec = importlib.util.spec_from_file_location("v3_final_driver", path)
    assert spec is not None and spec.loader is not None
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    lrs_flags = driver.PER_MODEL["lrs"]
    assert "--limit" not in lrs_flags  # the two-episode recipe is gone
    assert "--no-curriculum" in lrs_flags  # full split, shuffled every epoch
    assert driver.SEEDS == (1, 2, 3)
    command = driver.command("lrs", 1, AI / "out" / "tmp")
    assert "--limit" not in command
    assert "--no-curriculum" in command


@pytest.mark.skipif(not V3_DATA.exists(), reason="v3_final data not present")
def test_real_v3_targets_fit_the_envelope_and_held_out_is_disjoint() -> None:
    episodes = load_episodes(V3_DATA)
    groups = group_by_scenario(episodes)
    held = {name for name, group in groups.items() if any(e.held_out for e in group)}
    assert held == {f"scenario_{i:02d}" for i in (16, 17, 18, 19)}
    train = [e for e in episodes if not e.held_out]
    assert {e.scenario_id for e in train} == set(range(16))
    targets = np.concatenate([e.command_target() for e in train])
    rms = float(np.sqrt(np.mean(targets.astype(np.float64) ** 2)))
    assert np.isfinite(rms) and rms <= CommandEnvelope().accel
    assert float(np.abs(targets).max()) <= CommandEnvelope().accel + 1e-3
