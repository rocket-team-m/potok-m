"""SPEC_SMART.md: physics teacher, scenario holdout, and student distillation."""

import struct

import numpy as np
import pytest
import torch

from potok_ai.data import (
    SIM_HEADER,
    SIM_MAGIC,
    SIM_META_V2,
    SIM_TARGETS_V3,
    SIM_V2_EPISODE,
    SMART_FEATURES,
    group_by_scenario,
    make_smart_episode,
    read_validation,
)
from potok_ai.export import (
    MODEL_KIND_STUDENT,
    SMART_STUDENT_FEATURES,
    export_student,
    load_student_export,
)
from potok_ai.model import SmartConfig, SmartTeacher
from potok_ai.student import StatefulTCN, smart_student_config
from potok_ai.train_smart import (
    build_parser,
    evaluate,
    generalisation_ratio,
    normalized_correction_loss,
    run,
)


def _conditioning(offset: float) -> np.ndarray:
    return np.array([13.0 + offset, 0.0082, 0.35 + offset, 0.95, 3000.0, 0.01], np.float32)


def _episodes() -> list:
    episodes = []
    for scenario_id, held_out, count in [(0, False, 2), (1, False, 2), (16, True, 2)]:
        for index in range(count):
            episodes.append(
                make_smart_episode(
                    seed=100 + len(episodes),
                    frames=40,
                    scenario_id=scenario_id,
                    conditioning=_conditioning(index * 0.01),
                    held_out=held_out,
                )
            )
    return episodes


def _pack(episodes: list) -> bytes:
    first = episodes[0]
    header = struct.pack(
        "<8sIQ4I3f32s",
        SIM_MAGIC,
        3,
        42,
        len(episodes),
        first.frames.shape[0],
        SMART_FEATURES,
        SIM_TARGETS_V3,
        50.0,
        0.02,
        0.06,
        struct.pack("<I", SIM_META_V2) + b"\0" * 28,
    )
    assert len(header) == SIM_HEADER.size
    raw = bytearray(header)
    for episode in episodes:
        raw += SIM_V2_EPISODE.pack(
            episode.index,
            *episode.metadata,
            *episode.conditioning,
            episode.scenario_id,
            int(episode.held_out),
        )
        raw += np.asarray(episode.frames[:, :SMART_FEATURES], np.float32).tobytes()
        raw += np.asarray(episode.targets, np.float32).tobytes()
    return bytes(raw)


def _teacher() -> SmartTeacher:
    return SmartTeacher(
        SmartConfig(width=16, rank=8, dilations=(1, 2, 4, 8), loops=2)
    ).eval()


def test_smart_validation_reader_and_scenario_split(tmp_path):
    path = tmp_path / "validation.bin"
    path.write_bytes(_pack(_episodes()))
    data = read_validation(path)
    assert data.header.version == 3
    assert data.header.feature_count == SMART_FEATURES
    episodes = data.episodes
    assert [episode.frames.shape for episode in episodes] == [(40, 14)] * 6
    assert episodes[0].scenario == "scenario_00" and episodes[0].scenario_id == 0
    assert not episodes[0].held_out
    assert episodes[-1].scenario == "scenario_16" and episodes[-1].held_out
    assert np.allclose(episodes[1].conditioning, _conditioning(0.01))
    groups = group_by_scenario(episodes)
    assert sorted(groups) == ["scenario_00", "scenario_01", "scenario_16"]
    assert all(episode.held_out for episode in groups["scenario_16"])


def test_loop_forward_shape_and_attnres_allocation():
    torch.manual_seed(3)
    model = _teacher()
    window = torch.randn(2, 12, 14)
    cond = torch.randn(2, 6)
    with torch.no_grad():
        output = model(window, cond)
    assert output.shape == (2, 3)
    assert torch.isfinite(output).all()
    assert (output[:, :2].abs() <= 1.0 + 1e-6).all()
    assert ((output[:, 2] >= 0) & (output[:, 2] <= 1)).all()
    assert len(model.blocks) == 4
    allocation = model.last_alloc
    assert allocation is not None and allocation.shape == (2, 2)
    assert torch.allclose(allocation.sum(dim=-1), torch.ones(2))
    assert ((allocation >= 0) & (allocation <= 1)).all()


def test_conditioning_changes_output():
    torch.manual_seed(4)
    model = _teacher()
    window = torch.randn(2, 12, 14)
    with torch.no_grad():
        first = model(window, torch.zeros(2, 6))
        second = model(window, torch.ones(2, 6) * 3)
    assert not torch.allclose(first, second)


def test_normalized_loss_punishes_a_constant_output():
    torch.manual_seed(5)
    target = torch.randn(64, 2)
    constant = torch.zeros(64, 2)
    assert normalized_correction_loss(constant, target) == pytest.approx(1.0, rel=0.15)
    assert normalized_correction_loss(target, target) == 0.0


def test_generalisation_ratio_and_evaluate(tmp_path):
    ratio = generalisation_ratio(
        {"scenario_00": 1.0, "scenario_01": 1.2, "scenario_16": 1.32},
        {"scenario_16"},
    )
    assert ratio == pytest.approx(1.2)

    model = _teacher()
    groups = group_by_scenario(_episodes())
    args = build_parser().parse_args(
        ["--data", str(tmp_path / "unused.bin"), "--window", "12", "--batch", "2", "--eval-limit", "4"]
    )
    rmse = evaluate(model, groups, args, torch.device("cpu"))
    assert set(rmse) == set(groups)
    assert all(np.isfinite(value) and value >= 0 for value in rmse.values())


def test_smart_student_export_roundtrip(tmp_path):
    model = StatefulTCN(smart_student_config(seed=1, width=16, rank=8)).eval()
    condition = _conditioning(0.0)
    window = np.concatenate(
        [make_smart_episode(7, frames=8).frames[:, :SMART_FEATURES], np.tile(condition, (8, 1))],
        axis=1,
    )
    probe = np.concatenate([make_smart_episode(8, frames=1).frames[0, :14], condition])
    with torch.no_grad():
        model.warmup(window)
        expected = model.step(probe)

    path = export_student(model, tmp_path / "smart_student.bin")
    exported = load_student_export(path)
    assert exported.model_kind == MODEL_KIND_STUDENT
    assert exported.in_features == SMART_STUDENT_FEATURES == 20
    reloaded = exported.to_model()
    with torch.no_grad():
        reloaded.warmup(window)
        actual = reloaded.step(probe)
    assert torch.allclose(actual[0], expected[0], atol=1e-6)
    assert torch.allclose(actual[1], expected[1], atol=1e-6)


def test_tiny_cpu_training_step(tmp_path):
    data = tmp_path / "validation.bin"
    data.write_bytes(_pack(_episodes()))
    out = tmp_path / "smart.pt"
    args = build_parser().parse_args(
        [
            "--data", str(data),
            "--out", str(out),
            "--epochs", "1",
            "--seed", "1",
            "--window", "12",
            "--batch", "2",
            "--limit", "4",
            "--eval-limit", "4",
            "--width", "16",
            "--rank", "8",
        ]
    )
    result = run(args)
    assert set(result["rmse"]) == {"scenario_00", "scenario_01", "scenario_16"}
    assert result["held_out"] == ["scenario_16"]
    assert np.isfinite(result["ratio"]) and result["ratio"] > 0
    assert out.exists()
    student_pt = out.with_name("smart_student.pt")
    assert student_pt.exists() and result["export"].exists()
    stored = torch.load(student_pt, map_location="cpu", weights_only=True)
    assert stored["config"]["in_features"] == SMART_STUDENT_FEATURES
    assert stored["config"]["loops"] == 2
