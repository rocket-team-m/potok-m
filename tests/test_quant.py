import pytest
import torch

from potok_ai.model import Hadamard4Bit, TernarySTE, fast_hadamard, ternary_quantize


def test_ternary_codes_are_b1_58():
    weights = torch.tensor([[0.3, -0.2, 0.0, 1.0]])
    quantized, alpha = ternary_quantize(weights)
    codes = quantized / alpha
    assert torch.all(torch.isin(codes, torch.tensor([-1.0, 0.0, 1.0])))
    assert alpha > 0


def test_ternary_ste_gradient_flows():
    quantizer = TernarySTE(anneal=1.0).train()
    weights = torch.randn(8, 4, requires_grad=True)
    quantized = quantizer(weights)
    quantized.sum().backward()
    assert weights.grad is not None
    assert torch.isfinite(weights.grad).all()
    assert weights.grad.abs().sum() > 0


def test_anneal_interpolates_forward_value():
    quantizer = TernarySTE().train()
    weights = torch.tensor([[0.3, -0.2]])
    quantizer.set_anneal(0.0)
    assert torch.allclose(quantizer(weights), weights)
    quantizer.set_anneal(1.0)
    assert not torch.allclose(quantizer(weights), weights)


def test_hadamard4bit_shape_and_gradient():
    activation = Hadamard4Bit(16).train()
    x = torch.randn(3, 16, requires_grad=True)
    y = activation(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert torch.allclose(fast_hadamard(fast_hadamard(x)), x, atol=1e-5)


def test_hadamard_requires_power_of_two():
    with pytest.raises(ValueError):
        Hadamard4Bit(30)
