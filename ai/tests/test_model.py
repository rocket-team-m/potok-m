import torch

from potok_ai.model import ModelConfig, PotokModel, TSCTLinear, count_parameters


def small_config() -> ModelConfig:
    return ModelConfig(width=32, rank=16, in_features=8, out_features=3)


def test_forward_shape_and_bounded_outputs():
    torch.manual_seed(0)
    model = PotokModel(small_config())
    output = model(torch.randn(2, 16, 8))
    assert output.shape == (2, 3)
    assert torch.isfinite(output).all()
    assert (output[:, :2].abs() <= 1.0 + 1e-6).all()
    assert ((output[:, 2] >= 0.0) & (output[:, 2] <= 1.0)).all()


def test_architecture_layout():
    model = PotokModel(small_config())
    assert len(model.blocks) == 5
    assert [block.dw.dilation[0] for block in model.blocks] == [1, 2, 4, 8, 16]
    assert all(block.dw.kernel_size == (3,) for block in model.blocks)
    assert all(block.dw.groups == 32 for block in model.blocks)


def test_tsct_parameter_count():
    layer = TSCTLinear(32, 32, 16)
    assert count_parameters(layer) == 16 * (32 + 32 + 1)
    x = torch.randn(4, 32)
    assert layer(x).shape == (4, 32)


def test_seed_determinism():
    torch.manual_seed(7)
    first = PotokModel(small_config())
    torch.manual_seed(7)
    second = PotokModel(small_config())
    window = torch.randn(2, 16, 8)
    assert torch.equal(first(window), second(window))


def test_causal_prefix_is_unaffected_by_future_frames():
    model = PotokModel(small_config()).eval()
    window = torch.randn(1, 16, 8)
    changed = window.clone()
    changed[:, -1, :] += 10.0
    before = model.forward_features(window)
    after = model.forward_features(changed)
    assert torch.allclose(before[:, :-1], after[:, :-1], atol=1e-6)
    assert not torch.allclose(before[:, -1], after[:, -1])
