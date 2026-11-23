# SPDX-License-Identifier: GPL-3.0-or-later
"""Gates for the JEPA v2 objective. CPU only, no GPU, no external fixtures."""

import pytest
import torch
from torch import nn

from potok_ai.jepa_v2 import (
    DEFAULT_HORIZONS,
    JEPAV2,
    JEPAV2Config,
    koleo_loss,
    normalize_latents,
    percentile_stats,
)

BATCH, TIME, FEATURES, WIDTH = 4, 16, 6, 8


class TinyEncoder(nn.Module):
    """Deterministic encoder with the forward_features signature used in-tree."""

    def __init__(self, seed: int = 1) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.layer = nn.Linear(FEATURES, WIDTH)
        nn.init.normal_(self.layer.weight, std=0.5, generator=generator)
        nn.init.zeros_(self.layer.bias)

    def forward_features(self, window: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.layer(window))


def make(seed: int = 0, **kwargs) -> tuple[TinyEncoder, JEPAV2]:
    student = TinyEncoder(seed=1)
    objective = JEPAV2(student, JEPAV2Config(width=WIDTH, seed=seed, **kwargs))
    return student, objective


def window() -> torch.Tensor:
    return torch.randn(BATCH, TIME, FEATURES, generator=torch.Generator().manual_seed(11))


def test_multi_horizon_shapes() -> None:
    student, objective = make()
    losses = objective.loss(student, window())
    assert set(losses.predictions) == set(DEFAULT_HORIZONS) == {3, 6, 10}
    for horizon in DEFAULT_HORIZONS:
        assert losses.predictions[horizon].shape == (BATCH, TIME, WIDTH)
        assert losses.per_horizon[horizon].ndim == 0
    assert set(losses.per_horizon) == set(DEFAULT_HORIZONS)
    assert losses.jepa.ndim == 0 and losses.koleo.ndim == 0 and losses.total.ndim == 0
    assert torch.isfinite(losses.total)


def test_ema_update_and_sync() -> None:
    student, objective = make(momentum=0.5)
    teacher_param = objective.teacher.net.layer.weight
    assert all(not parameter.requires_grad for parameter in objective.teacher.parameters())
    before = teacher_param.detach().clone()
    with torch.no_grad():
        student.layer.weight.add_(0.4)
    objective.update_teacher(student)
    assert torch.allclose(teacher_param.detach(), before + 0.2, atol=1e-6)
    objective.teacher.sync(student)
    assert torch.equal(teacher_param.detach(), student.layer.weight.detach())


def test_no_gradient_to_teacher() -> None:
    student, objective = make()
    objective.loss(student, window()).total.backward()
    assert all(p.grad is None for p in objective.teacher.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in objective.predictor.parameters())


def test_free_bits_floor() -> None:
    student, objective = make(free_bits=5.0, koleo_weight=0.0)
    losses = objective.loss(student, window())
    assert losses.jepa.item() == pytest.approx(5.0)
    assert all(value.item() == pytest.approx(5.0) for value in losses.per_horizon.values())
    assert losses.total.item() == pytest.approx(5.0)
    losses.total.backward()
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in objective.predictor.parameters())


def test_symlog_percentile_scale_invariance() -> None:
    # symlog is log-like away from zero, linear near it: test the log-like regime.
    signs = torch.where(torch.randn(BATCH, TIME, WIDTH, generator=torch.Generator().manual_seed(6)) > 0, 1.0, -1.0)
    targets = signs * (1000.0 + 100.0 * torch.randn(BATCH, TIME, WIDTH, generator=torch.Generator().manual_seed(7)))
    _, scale = percentile_stats(targets)
    assert scale.item() >= 1.0  # DreamerV3's floor, in symlog space
    assert torch.allclose(normalize_latents(targets), normalize_latents(targets * 4.0), atol=1e-2)
    tight = targets * 1e-6
    assert percentile_stats(tight)[1].item() == pytest.approx(1.0)


def test_seeded_augmentation_is_reproducible() -> None:
    a, objective_a = make(seed=3)
    b, objective_b = make(seed=3)
    _, objective_c = make(seed=4)
    first = objective_a.loss(a, window()).jepa
    assert torch.equal(first, objective_b.loss(b, window()).jepa)
    assert not torch.equal(first, objective_c.loss(TinyEncoder(seed=1), window()).jepa)
    objective_a.reset_seed()
    assert torch.equal(first, objective_a.loss(a, window()).jepa)


def test_koleo_is_finite() -> None:
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    loss = koleo_loss(features)
    assert loss.ndim == 0 and torch.isfinite(loss) and loss != 0.0
    assert koleo_loss(features[:1]).item() == 0.0


def test_malformed_batch_is_loud() -> None:
    _, objective = make()
    with pytest.raises(ValueError, match="batch, time, features"):
        objective.loss(TinyEncoder(), torch.randn(BATCH, TIME))
    with pytest.raises(ValueError, match="time>10"):
        objective.loss(TinyEncoder(), torch.randn(BATCH, 10, FEATURES))
    bad = window()
    bad[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or inf"):
        objective.loss(TinyEncoder(), bad)
    with pytest.raises(ValueError, match="cond must be"):
        objective.loss(TinyEncoder(), window(), torch.randn(2, 3))
    with pytest.raises(ValueError, match="horizons must be positive"):
        JEPAV2Config(width=WIDTH, horizons=(3, 0, 10))
