# SPDX-License-Identifier: GPL-3.0-or-later
"""B01 gates: exact MAC count, deterministic inference, causal state sweep."""

import pytest
import torch

from potok_ai.fast_model import FastModel, FastModelConfig, analytical_mac, mac_breakdown


def _config(**overrides):
    settings = {"window": 12, "width": 8, "rank": 2, "seed": 0}
    settings.update(overrides)
    return FastModelConfig(**settings)


def _runtime_mac(model: FastModel, window: torch.Tensor) -> dict[str, int]:
    """Count MACs actually executed, from hooks on every multiply-bearing op."""
    targets = {
        "stem": model.stem,
        "depthwise_conv": model.block.dw,
        "state_in": model.block.state_in,
        "state_out": model.block.state_out,
        "head": model.head,
    }
    counts = dict.fromkeys(targets, 0)

    def make_hook(name):
        def count(module, inputs, _output):
            value = inputs[0]
            if isinstance(module, torch.nn.Conv1d):
                out_len = value.shape[-1] - module.kernel_size[0] + 1
                per_call = (
                    value.shape[0]
                    * out_len
                    * module.out_channels
                    * (module.in_channels // module.groups)
                    * module.kernel_size[0]
                )
            else:
                tokens = value.numel() // value.shape[-1]
                per_call = tokens * module.in_features * module.out_features
            counts[name] += per_call

        return count

    handles = [target.register_forward_hook(make_hook(name)) for name, target in targets.items()]
    with torch.no_grad():
        model(window)
    for handle in handles:
        handle.remove()
    return counts


def test_mac_breakdown_is_exact_and_matches_runtime():
    for cfg in (
        FastModelConfig(),
        _config(loops=2),
        _config(loops=2, window=7, kernel=5, rank=3, in_features=4, width=16),
    ):
        model = FastModel(cfg).eval()
        window = torch.randn(2, cfg.window, cfg.in_features)
        assert _runtime_mac(model, window) == mac_breakdown(cfg, batch=2)
        assert sum(_runtime_mac(model, window).values()) == analytical_mac(cfg, batch=2)


def test_mac_totals_are_pinned_and_loops_scale_linearly():
    assert mac_breakdown() == {
        "stem": 22_400,
        "depthwise_conv": 4_800,
        "state_in": 12_800,
        "state_out": 12_800,
        "head": 96,
    }
    assert analytical_mac() == 52_896
    assert analytical_mac(FastModelConfig(loops=2)) == 83_296
    assert analytical_mac(FastModelConfig(loops=2)) - analytical_mac() == 30_400
    with pytest.raises(ValueError):
        mac_breakdown(batch=0)


def test_same_seed_is_bit_identical_and_global_rng_is_unused():
    window = torch.randn(2, 12, 14)
    torch.manual_seed(999)
    first = FastModel(_config()).eval()
    torch.manual_seed(1)
    torch.randn(10_000)
    second = FastModel(_config()).eval()

    assert all(
        torch.equal(a, b)
        for a, b in zip(first.state_dict().values(), second.state_dict().values())
    )
    with torch.no_grad():
        out_a, out_b = first(window), first(window)
        out_c = second(window)
    assert torch.equal(out_a, out_b)
    assert torch.equal(out_a, out_c)

    # Different seed must actually separate the candidates.
    with torch.no_grad():
        out_other = FastModel(_config(seed=7)).eval()(window)
    assert not torch.equal(out_a, out_other)


def test_eval_and_train_modes_agree():
    torch.manual_seed(3)
    model = FastModel(_config(loops=2))
    window = torch.randn(2, 12, 14)
    with torch.no_grad():
        train_out = model.train()(window)
        eval_out = model.eval()(window)
    assert torch.equal(train_out, eval_out)


def test_features_are_causal_for_both_loop_counts():
    torch.manual_seed(4)
    window = torch.randn(2, 12, 14)
    for loops in (1, 2):
        model = FastModel(_config(loops=loops)).eval()
        with torch.no_grad():
            baseline = model.forward_features(window)
        for frame in (0, 5, 11):
            changed = window.clone()
            changed[:, frame + 1 :, :] += 7.0
            with torch.no_grad():
                after = model.forward_features(changed)
            assert torch.equal(baseline[:, : frame + 1], after[:, : frame + 1])
        changed = window.clone()
        changed[:, 1:, :] += 7.0
        with torch.no_grad():
            after = model.forward_features(changed)
        assert not torch.allclose(baseline[:, 1:], after[:, 1:])


def test_output_shape_bounds_and_finiteness():
    torch.manual_seed(5)
    cfg = _config(g_max=0.25)
    model = FastModel(cfg).eval()
    with torch.no_grad():
        out = model(torch.randn(3, cfg.window, cfg.in_features))
    assert out.shape == (3, 3)
    assert torch.isfinite(out).all()
    assert out[:, :2].abs().le(cfg.g_max + 1e-6).all()
    assert ((out[:, 2] > 0) & (out[:, 2] < 1)).all()


def test_config_and_input_validation():
    with pytest.raises(ValueError):
        FastModelConfig(loops=0)
    with pytest.raises(ValueError):
        FastModelConfig(loops=3)
    with pytest.raises(ValueError):
        FastModelConfig(rank=9, width=8)
    with pytest.raises(ValueError):
        FastModelConfig(kernel=0)
    with pytest.raises(ValueError):
        FastModelConfig(window=0)
    with pytest.raises(ValueError):
        FastModelConfig(out_features=2)
    with pytest.raises(ValueError):
        FastModelConfig(decay=1.0)
    with pytest.raises(ValueError):
        FastModelConfig(g_max=0.0)

    model = FastModel(_config()).eval()
    with pytest.raises(ValueError):
        model(torch.randn(2, 12, 13))
    with pytest.raises(ValueError):
        model(torch.randn(2, 0, 14))
