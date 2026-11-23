"""LoopWM privileged teacher: shared iterative refinement, latent JEPA, and CLI."""

import inspect
import struct
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from potok_ai.data import (
    SIM_HEADER,
    SIM_MAGIC,
    SIM_META_V2,
    SIM_TARGETS,
    SIM_V2_EPISODE,
    SMART_FEATURES,
    group_by_scenario,
    load_episodes,
    make_smart_episode,
)
from potok_ai.export import MODEL_KIND_STUDENT, SMART_STUDENT_FEATURES, load_student_export
from potok_ai.jepa import EMATeacher
from potok_ai.loopwm import LATENT_SCALE_FLOOR, LoopWM, LoopWMConfig
from potok_ai.train_loopwm import build_parser, evaluate, run
from potok_ai.train_smart import generalisation_ratio


def _config(**overrides) -> LoopWMConfig:
    settings = {"width": 16, "heads": 4, "max_iters": 6, "exit_threshold": 0.0}
    settings.update(overrides)
    return LoopWMConfig(**settings)


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
        2,
        42,
        len(episodes),
        first.frames.shape[0],
        SMART_FEATURES,
        SIM_TARGETS,
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


def test_shared_weight_iterative_refinement():
    torch.manual_seed(0)
    model = LoopWM(_config())
    window, cond = torch.randn(2, 12, 14), torch.randn(2, 6)
    hooks = []
    params_before = sum(p.numel() for p in model.block.parameters())

    def count(_module, _inputs, _output):
        hooks.append(1)

    handle = model.block.register_forward_hook(count)
    latent, info = model.refine(window, cond, threshold=0.0)
    handle.remove()
    assert len(hooks) == model.cfg.max_iters == 6
    assert info.iters == 6 and len(info.updates) == 6
    assert latent.shape == (2, 12, model.cfg.width)
    latent2, _ = model.refine(window, cond, threshold=0.0)
    assert torch.allclose(latent, latent2)
    assert params_before == sum(p.numel() for p in model.block.parameters())


def test_adaptive_exit_with_cap():
    torch.manual_seed(1)
    model = LoopWM(_config())
    window, cond = torch.randn(2, 12, 14), torch.randn(2, 6)
    _, capped = model.refine(window, cond, threshold=0.0)
    assert capped.iters == 6
    _, early = model.refine(window, cond, threshold=1e9)
    assert early.iters == 1 and len(early.updates) == 1
    short = LoopWM(_config(max_iters=3))
    _, trace = short.refine(window, cond, threshold=0.0)
    assert trace.iters == 3


def test_latent_targets_shapes():
    torch.manual_seed(2)
    model = LoopWM(_config())
    window, cond = torch.randn(2, 12, 14), torch.randn(2, 6)
    latent, _ = model.refine(window, cond)
    predictions = model.latent_predictions(latent)
    assert set(predictions) == {3, 6, 10}
    for horizon, prediction in predictions.items():
        assert prediction.shape == (2, 12 - horizon, model.cfg.width)
    target_encoder = EMATeacher(model.encoder, 0.999)
    assert target_encoder(window).shape == (2, 12, model.cfg.width)
    loss = model.latent_prediction_loss(window, cond, target_encoder, latent=latent)
    assert loss.ndim == 0 and torch.isfinite(loss)
    with pytest.raises(ValueError):
        model.latent_prediction_loss(window[:, :8], cond, target_encoder)


def test_conditioning_sensitivity():
    torch.manual_seed(3)
    model = LoopWM(_config()).eval()
    window = torch.randn(2, 12, 14)
    low, high = torch.zeros(2, 6), torch.ones(2, 6) * 3
    with torch.no_grad():
        latent_low, _ = model.refine(window, low)
        latent_high, _ = model.refine(window, high)
        output_low, output_high = model(window, low), model(window, high)
    assert not torch.allclose(latent_low, latent_high)
    assert not torch.allclose(output_low, output_high)
    assert output_low.shape == (2, 3)


def test_privileged_reference_is_supervision_only():
    parameters = inspect.signature(LoopWM.forward).parameters
    assert set(parameters) == {"self", "window", "cond"}
    model = LoopWM(_config()).eval()
    window, cond, reference = torch.randn(2, 12, 14), torch.randn(2, 6), torch.randn(2, 2)
    with pytest.raises(TypeError):
        model(window, cond, reference)
    prediction = model(window, cond)[:, :2]
    assert prediction.shape == reference.shape
    assert prediction.abs().le(1.0 + 1e-6).all()


def test_held_out_ratio():
    ratio = generalisation_ratio(
        {"scenario_00": 1.0, "scenario_01": 1.2, "scenario_16": 1.35}, {"scenario_16"}
    )
    assert ratio == pytest.approx(1.35 / 1.1)
    model = LoopWM(_config()).eval()
    args = build_parser().parse_args(
        ["--data", "unused", "--window", "12", "--batch", "2", "--eval-limit", "4"]
    )
    rmse = evaluate(model, group_by_scenario(_episodes()), args, torch.device("cpu"))
    assert set(rmse) == {"scenario_00", "scenario_01", "scenario_16"}
    assert all(np.isfinite(value) and value >= 0 for value in rmse.values())
    assert generalisation_ratio(rmse, {"scenario_16"}) > 0


def test_tiny_cpu_end_to_end(tmp_path):
    data = tmp_path / "validation.bin"
    data.write_bytes(_pack(_episodes()))
    out = tmp_path / "loopwm.pt"
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
            "--heads", "4",
        ]
    )
    result = run(args)
    assert set(result["rmse"]) == {"scenario_00", "scenario_01", "scenario_16"}
    assert result["held_out"] == ["scenario_16"]
    assert np.isfinite(result["ratio"]) and result["ratio"] > 0
    assert out.exists()
    student_pt = out.with_name("loopwm_student.pt")
    assert student_pt.exists() and result["export"].exists()
    stored = torch.load(out, map_location="cpu", weights_only=True)
    assert stored["model_config"]["max_iters"] == 6
    exported = load_student_export(result["export"])
    assert exported.model_kind == MODEL_KIND_STUDENT
    assert exported.in_features == SMART_STUDENT_FEATURES == 20


class _FixedEncoder(nn.Module):
    """Deterministic JEPA target network: broadcasts a fixed (B, T, width) batch."""

    def __init__(self, target: torch.Tensor) -> None:
        super().__init__()
        self.target = target

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        return self.target


def test_exit_threshold_is_typed_config_and_cli():
    args = build_parser().parse_args(
        ["--data", "unused", "--exit-threshold", "0.25", "--latent-weight", "0.2"]
    )
    assert isinstance(args.exit_threshold, float) and args.exit_threshold == 0.25
    assert isinstance(args.latent_weight, float) and args.latent_weight == 0.2
    with pytest.raises(ValueError):
        LoopWMConfig(exit_threshold=float("nan"))


def test_exit_count_uses_config_threshold_and_trace():
    torch.manual_seed(11)
    window, cond = torch.randn(2, 12, 14), torch.randn(2, 6)
    never = LoopWM(_config(exit_threshold=1e9))
    _, trace = never.refine(window, cond)  # no override: config is authoritative
    assert trace.iters == 1 and len(trace.updates) == 1
    always = LoopWM(_config(exit_threshold=0.0))
    _, trace = always.refine(window, cond)
    assert trace.iters == always.cfg.max_iters == 6
    assert len(trace.updates) == 6
    capped = LoopWM(_config(max_iters=3, exit_threshold=0.0))
    _, trace = capped.refine(window, cond)
    assert trace.iters == 3 and len(trace.updates) == 3
    with pytest.raises(ValueError):
        always.refine(window, cond, threshold=-1.0)


def test_latent_loss_normalized_by_batch_std():
    torch.manual_seed(12)
    width = _config().width
    window, cond = torch.randn(2, 12, 14), torch.randn(2, 6)
    model = LoopWM(_config())
    for head in model.latent_heads.values():
        with torch.no_grad():
            head.weight.zero_()
            head.bias.zero_()
    latent, _ = model.refine(window, cond)
    pattern = torch.linspace(-1.0, 1.0, width).view(1, 1, width).expand(2, 12, width).clone()
    assert pattern.std(unbiased=False) > LATENT_SCALE_FLOOR

    def loss_at(scale: float) -> torch.Tensor:
        return model.latent_prediction_loss(
            window, cond, _FixedEncoder(pattern * scale), latent=latent
        )

    expected = float(pattern.square().mean() / pattern.var(unbiased=False))
    assert loss_at(1.0).item() == pytest.approx(expected, rel=1e-5)
    assert loss_at(100.0).item() == pytest.approx(loss_at(1.0).item(), rel=1e-5)


def test_latent_loss_floor_keeps_gradients_finite():
    torch.manual_seed(13)
    window, cond = torch.randn(2, 12, 14), torch.randn(2, 6)
    model = LoopWM(_config())
    constant = _FixedEncoder(torch.full((2, 12, model.cfg.width), 2.0))
    loss = model.latent_prediction_loss(window, cond, constant)
    assert torch.isfinite(loss) and loss.ndim == 0
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)
    assert LATENT_SCALE_FLOOR == 1e-3


def test_training_finite_across_threshold_sweep():
    from potok_ai.train_loopwm import train_teacher

    data = Path(__file__).resolve().parents[1] / "out" / "run4" / "data"
    if not data.is_dir():
        pytest.skip("run4 held-out data is not available")
    for threshold in (1e-3, 1e-2, 1e-1, 1.0):
        args = build_parser().parse_args(
            [
                "--data",
                str(data),
                "--window",
                "12",
                "--batch",
                "4",
                "--limit",
                "8",
                "--epochs",
                "1",
                "--seed",
                "1",
                "--exit-threshold",
                str(threshold),
                "--latent-weight",
                "0.1",
            ]
        )
        episodes = [e for e in load_episodes(data) if not e.held_out]
        model = train_teacher(episodes, args, torch.device("cpu"))
        assert all(torch.isfinite(p).all() for p in model.parameters())
        latent, info = model.refine(torch.randn(2, 12, 14), torch.randn(2, 6))
        assert 1 <= info.iters <= 6 and torch.isfinite(latent).all()
