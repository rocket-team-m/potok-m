"""Scale invariance of LoopWM's normalized latent loss and exit threshold."""

import pytest
import torch
from torch import nn

from potok_ai.loopwm import LATENT_SCALE_FLOOR, LoopWM, LoopWMConfig
from potok_ai.loopwm_scale import normalized_latent_loss, scaled_exit_threshold


class _FixedEncoder(nn.Module):
    """Deterministic JEPA target network: broadcasts a fixed latent batch."""

    def __init__(self, target: torch.Tensor) -> None:
        super().__init__()
        self.target = target

    def forward(self, _window: torch.Tensor) -> torch.Tensor:
        return self.target


def test_normalized_latent_loss_is_scale_invariant():
    torch.manual_seed(0)
    target = torch.randn(2, 12, 16)
    target = target - target.mean()  # a constant predictor should then score ~1
    prediction = torch.randn(2, 12, 16)
    base = normalized_latent_loss(prediction, target)
    for factor in (1e-2, 1.0, 1e2, 1e4):
        scaled = normalized_latent_loss(prediction * factor, target * factor)
        assert scaled.item() == pytest.approx(base.item(), rel=1e-4)
    assert normalized_latent_loss(torch.zeros_like(target), target).item() == pytest.approx(
        1.0, rel=1e-4
    )


def test_normalized_latent_loss_floor_keeps_loss_and_grads_finite():
    prediction = torch.randn(2, 3, 4, requires_grad=True)
    target = torch.full((2, 3, 4), 2.0)
    loss = normalized_latent_loss(prediction, target)
    expected = (prediction.detach() - 2.0).square().mean() / LATENT_SCALE_FLOOR**2
    assert loss.item() == pytest.approx(expected.item(), rel=1e-5)
    loss.backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()


def test_normalized_latent_loss_shared_scale_matches_loopwm():
    torch.manual_seed(1)
    width, time, horizon = 16, 12, 3
    window, cond = torch.randn(2, time, 14), torch.randn(2, 6)
    pattern = torch.linspace(-1.0, 1.0, width).view(1, 1, width).expand(2, time, width)
    encoder = _FixedEncoder(pattern * 37.0)
    model = LoopWM(
        LoopWMConfig(width=width, heads=4, max_iters=1, exit_threshold=0.0, horizons=(horizon,))
    )
    for head in model.latent_heads.values():
        with torch.no_grad():
            head.weight.zero_()
            head.bias.zero_()

    expected = model.latent_prediction_loss(window, cond, encoder)
    prediction = model.latent_predictions(model.refine(window, cond)[0])[horizon]
    actual = normalized_latent_loss(
        prediction, encoder.target[:, horizon:, :], scale=encoder.target
    )
    assert actual.item() == pytest.approx(expected.item(), rel=1e-6)


def test_normalized_latent_loss_validates_inputs():
    with pytest.raises(ValueError, match="same shape"):
        normalized_latent_loss(torch.zeros(2, 3, 4), torch.zeros(2, 3, 5))
    with pytest.raises(ValueError, match="feature dimension"):
        normalized_latent_loss(torch.zeros(4), torch.zeros(4))


def test_scaled_exit_threshold_follows_latent_scale():
    torch.manual_seed(2)
    latent = torch.randn(2, 12, 16)
    base = scaled_exit_threshold(0.1, latent)
    for factor in (1e-2, 1.0, 1e2, 1e4):
        scaled = scaled_exit_threshold(0.1, latent * factor)
        assert scaled.item() == pytest.approx((base * factor).item(), rel=1e-5)
    constant = scaled_exit_threshold(0.1, torch.ones(2, 12, 16))
    assert constant.item() == pytest.approx(0.1 * LATENT_SCALE_FLOOR, rel=1e-6)


def test_exit_decision_is_scale_invariant():
    torch.manual_seed(3)
    latent, increment = torch.randn(2, 12, 16), torch.randn(2, 12, 16)
    for threshold in (0.01, 0.1, 1.0, 100.0):
        reference = float(increment.norm(dim=-1).mean() < scaled_exit_threshold(threshold, latent))
        for factor in (1e-2, 1e2):
            norm = (increment * factor).norm(dim=-1).mean()
            limit = scaled_exit_threshold(threshold, latent * factor)
            assert bool(norm < limit) == bool(reference)


@pytest.mark.parametrize("threshold", [-0.1, float("nan"), float("inf")])
def test_scaled_exit_threshold_rejects_invalid_thresholds(threshold):
    with pytest.raises(ValueError, match="finite and non-negative"):
        scaled_exit_threshold(threshold, torch.randn(2, 3, 4))
