"""Unified engine (SPEC_ENGINE.md): CLI, adapters, metrics and CPU determinism."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from test_loopwm import _episodes, _pack

from potok_ai import engine
from potok_ai.export import export_student
from potok_ai.loopwm import LoopWM, LoopWMConfig
from potok_ai.loopwm_lite import LiteConfig, LiteModel, analytical_mac
from potok_ai.student import StatefulTCN, smart_student_config


def _data(tmp_path: Path) -> Path:
    path = tmp_path / "validation.bin"
    path.write_bytes(_pack(_episodes()))
    return path


def _argv(model: str, data: Path, out: Path, *extra: str) -> list[str]:
    return [
        "--model", model,
        "--stage", "train",
        "--data", str(data),
        "--out", str(out),
        "--epochs", "1",
        "--seed", "1",
        "--device", "cpu",
        "--window", "12",
        "--batch", "2",
        "--limit", "4",
        "--eval-limit", "4",
        "--width", "16",
        "--rank", "8",
        "--heads", "4",
        "--lite-width", "16",
        "--attn-rank", "4",
        *extra,
    ]


def test_engine_cli_help():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run(
        [sys.executable, "-m", "potok_ai.engine", "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert result.returncode == 0, result.stderr
    for token in ("--model", "--stage", "--data", "--out", "--epochs", "--seed",
                  "--device", "{lrs,loopwm-lite,loopwm}", "{train,distill,eval}"):
        assert token in result.stdout


def test_adapter_registry_and_dispatch(tmp_path, monkeypatch):
    assert engine.ADAPTERS["lrs"] is engine.LRSAdapter
    assert engine.ADAPTERS["loopwm-lite"] is engine.LiteAdapter
    assert engine.ADAPTERS["loopwm"] is engine.LoopWMAdapter

    called = []

    def fake_run(args):
        called.append(args)
        config = LoopWMConfig(width=16, heads=4)
        torch.save(
            {"model_config": config.__dict__, "state_dict": LoopWM(config).state_dict()},
            args.out,
        )
        export_student(StatefulTCN(smart_student_config()), args.out.with_name(
            "loopwm_student.bin"
        ))
        return {
            "rmse": {"scenario_00": 0.5, "scenario_01": 0.6, "scenario_16": 0.7},
            "ratio": 0.7 / 0.55,
            "held_out": ["scenario_16"],
            "teacher": args.out,
            "export": args.out.with_name("loopwm_student.bin"),
        }

    monkeypatch.setattr(engine, "run_loopwm", fake_run)
    data, out = _data(tmp_path), tmp_path / "legacy"
    engine.main(_argv("loopwm", data, out))
    assert len(called) == 1 and called[0].out == out / "loopwm.pt"
    report = json.loads((out / "metrics.json").read_text())
    assert report["model"] == "loopwm"
    assert report["rmse_train"] == pytest.approx(0.55)
    assert report["rmse_held_out"] == pytest.approx(0.7)


def test_lite_shapes_adaptive_exit_and_shared_weights():
    torch.manual_seed(0)
    cfg = LiteConfig(window=12, width=16, attn_rank=4, heads=2, target_width=16)
    model = LiteModel(cfg)
    window, cond = torch.randn(2, 12, 14), torch.randn(2, 6)
    calls = []
    handle = model.block.register_forward_hook(lambda *_: calls.append(1))
    params = sum(p.numel() for p in model.block.parameters())
    latent, capped = model.refine(window, cond, threshold=0.0)
    handle.remove()
    assert len(calls) == cfg.max_iters == 2
    assert capped.iters == 2 and len(capped.updates) == 2
    assert latent.shape == (2, 16, 16)
    assert params == sum(p.numel() for p in model.block.parameters())
    _, early = model.refine(window, cond, threshold=1e9)
    assert early.iters == 1 and len(early.updates) == 1
    assert model(window, cond).shape == (2, 3)
    pooled = latent.mean(dim=1)
    predictions = model.latent_predictions(pooled)
    assert set(predictions) == {3, 6}
    assert all(value.shape == (2, 16) for value in predictions.values())

    teacher = LoopWM(LoopWMConfig(width=16, heads=4))
    model.set_teacher(teacher)
    targets = model.teacher_targets(window, cond)
    assert set(targets) == {3, 6} and targets[3].shape == (2, 16)
    assert "teacher" not in dict(model.named_modules())
    assert not any(p.requires_grad for p in teacher.parameters())


def test_lite_analytical_mac_is_documented_band():
    mac = analytical_mac()
    assert 200_000 <= mac <= 450_000
    assert mac == analytical_mac(LiteConfig())


def test_metrics_json_schema_and_table(capsys, tmp_path):
    report = engine.Metrics("lrs", 0.1, 0.2, 2.0, 100, 200, 300)
    path = engine.write_metrics(tmp_path / "nested", report)
    data = json.loads(path.read_text())
    assert tuple(data) == engine.METRIC_FIELDS
    assert data == report.as_dict()
    engine.print_table(report)
    printed = capsys.readouterr().out
    for field in engine.METRIC_FIELDS:
        assert field in printed


@pytest.mark.parametrize("model", engine.MODELS)
def test_tiny_cpu_end_to_end(tmp_path, model):
    data, out = _data(tmp_path), tmp_path / model
    engine.main(_argv(model, data, out))
    report = json.loads((out / "metrics.json").read_text())
    assert tuple(report) == engine.METRIC_FIELDS
    assert report["model"] == model
    assert report["params"] > 0 and report["mac"] > 0 and report["bytes"] > 0
    for key in ("rmse_train", "rmse_held_out", "ratio"):
        assert np.isfinite(report[key]) and report[key] > 0
    expected = {
        "lrs": out / "lrs.bin",
        "loopwm-lite": out / "lite.pt",
        "loopwm": out / "loopwm_student.bin",
    }[model]
    assert expected.exists()


def test_eval_stage_scores_an_existing_export(tmp_path):
    data, out = _data(tmp_path), tmp_path / "evaluated"
    engine.main(_argv("loopwm-lite", data, out))
    trained = json.loads((out / "metrics.json").read_text())
    argv = _argv("loopwm-lite", data, out)
    argv[argv.index("train")] = "eval"
    engine.main(argv)
    report = json.loads((out / "metrics.json").read_text())
    assert report["model"] == "loopwm-lite" and report["rmse_train"] > 0
    assert report["params"] == trained["params"]
    assert report["mac"] == trained["mac"]
    assert report["rmse_train"] == pytest.approx(trained["rmse_train"])


def test_deterministic_seeding(tmp_path):
    data = _data(tmp_path)
    states = []
    for index, seed in enumerate((7, 7, 9)):
        out = tmp_path / f"run{index}"
        argv = _argv("loopwm-lite", data, out, "--seed", str(seed))
        engine.main(argv)
        stored = torch.load(out / "lite.pt", map_location="cpu", weights_only=True)
        states.append(stored["state_dict"])
    assert all(torch.equal(states[0][key], states[1][key]) for key in states[0])
    assert any(not torch.equal(states[0][key], states[2][key]) for key in states[0])
