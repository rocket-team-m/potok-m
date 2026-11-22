"""Residual-to-physics objective: baseline algebra and command supervision."""

import pytest
import torch
from torch.optim import AdamW

from potok_ai.physics import ALPHA0, BETA0, KD, KP, final_command, physics_baseline
from potok_ai.train_smart import normalized_correction_loss


def _frame(alpha: float = 0.0, beta: float = 0.0, a_lat: float = 0.0) -> torch.Tensor:
    frame = torch.zeros(14)
    frame[8], frame[9], frame[11] = alpha, beta, a_lat
    return frame


def test_baseline_formula_and_clipping():
    baseline = physics_baseline(_frame(alpha=20.0, beta=10.0))
    assert baseline.shape == (2,)
    assert baseline[0] == pytest.approx(KP * 10.0 / BETA0)
    assert baseline[1] == pytest.approx(KP * 20.0 / ALPHA0)

    damped = physics_baseline(_frame(a_lat=5.0))
    assert damped[0] == pytest.approx(KD * torch.tanh(torch.tensor(5.0)).item())
    assert damped[1] == pytest.approx(0.0)

    clipped = physics_baseline(_frame(alpha=1e4, beta=1e4, a_lat=1e4))
    assert bool((clipped.abs() <= 1.0).all())
    assert physics_baseline(torch.stack([_frame(), _frame(alpha=10.0)])).shape == (2, 2)


def test_baseline_rejects_short_frames():
    with pytest.raises(ValueError, match="12 features"):
        physics_baseline(torch.zeros(8))


def test_zero_residual_reproduces_baseline():
    baseline = physics_baseline(torch.rand(4, 14))
    assert torch.equal(final_command(torch.zeros_like(baseline), baseline, True), baseline)
    delta = torch.rand(4, 2)
    assert torch.equal(final_command(delta, baseline, False), delta)


def test_residual_training_drives_command_to_target():
    generator = torch.Generator().manual_seed(0)
    frames = torch.rand(32, 14, generator=generator) * 4 - 2
    baseline = physics_baseline(frames)
    targets = torch.tanh(frames[:, 1:3] * 0.3)
    delta = torch.zeros_like(targets, requires_grad=True)
    optimizer = AdamW([delta], lr=5e-2)

    def loss_value() -> float:
        with torch.no_grad():
            command = final_command(delta, baseline, True)
            return float(normalized_correction_loss(command, targets))

    initial = loss_value()
    for _ in range(300):
        command = final_command(delta, baseline, True)
        loss = normalized_correction_loss(command, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss_value() < initial * 0.2
