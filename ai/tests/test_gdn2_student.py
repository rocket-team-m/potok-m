# SPDX-License-Identifier: GPL-3.0-or-later
"""POST07 gates: exact MAC accounting, causality, determinism and bounds.

This is a structural study of an experimental candidate. It makes no accuracy,
latency or "beats LRS-TCN" claim; such a claim needs a 3-seed A/B at a matched
budget. No GPU is used: every parameter and input here lives on CPU.
"""

import pytest
import torch

from potok_ai.gdn2_student import (
    COND_FEATURES,
    FRAME_FEATURES,
    GDN2Student,
    GDN2StudentConfig,
    analytical_mac,
    mac_breakdown,
)


def _rng(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _frames(seed: int, count: int, scale: float = 1.0) -> torch.Tensor:
    return scale * torch.randn(count, FRAME_FEATURES, generator=_rng(seed))


def _conds(seed: int, count: int, scale: float = 1.0) -> torch.Tensor:
    return scale * torch.randn(count, COND_FEATURES, generator=_rng(seed))


def _config(**overrides):
    settings = {"seed": 0}
    settings.update(overrides)
    return GDN2StudentConfig(**settings)


# --- exact MAC -------------------------------------------------------------


def _linear_mac(model: GDN2Student) -> dict[str, int]:
    """Count executed Linear MACs over one step via forward hooks."""
    targets = {"stem": model.stem, "film": model.film, "head": model.head}
    for index, block in enumerate(model.blocks):
        for name in ("q", "k", "v", "erase", "write", "beta", "out"):
            targets[f"block{index}.{name}"] = getattr(block, name)
    counts = dict.fromkeys(targets, 0)

    def make_hook(name: str):
        def count(module, inputs, _output):
            value = inputs[0]
            tokens = value.numel() // value.shape[-1]
            counts[name] += int(tokens * module.in_features * module.out_features)

        return count

    handles = [target.register_forward_hook(make_hook(name)) for name, target in targets.items()]
    try:
        with torch.no_grad():
            model.step(_frames(11, 1)[0], _conds(12, 1)[0])
    finally:
        for handle in handles:
            handle.remove()
    return counts


def test_mac_breakdown_is_pinned_and_matches_runtime():
    assert mac_breakdown() == {
        "stem": 672,
        "film": 576,
        "block_projections": 30_720,
        "block_beta": 384,
        "block_recurrence": 4_096,
        "block_out": 6_144,
        "head": 144,
    }
    assert analytical_mac() == 42_736 == sum(mac_breakdown().values())
    assert analytical_mac(_config(width=64)) == 55_616

    for cfg in (_config(), _config(width=64, n_blocks=1)):
        model = GDN2Student(cfg).eval()
        runtime = _linear_mac(model)
        breakdown = mac_breakdown(cfg)
        assert runtime["stem"] == breakdown["stem"]
        assert runtime["film"] == breakdown["film"]
        assert runtime["head"] == breakdown["head"]
        assert (
            sum(
                v
                for k, v in runtime.items()
                if k.split(".")[-1] in {"q", "k", "v", "erase", "write"}
            )
            == breakdown["block_projections"]
        )
        assert sum(v for k, v in runtime.items() if k.endswith(".beta")) == breakdown["block_beta"]
        assert sum(v for k, v in runtime.items() if k.endswith(".out")) == breakdown["block_out"]
        # The two einsums per head are not nn.Linear; their count is checked
        # against the state shape executed by the recurrence.
        _, _, state = model._tick(
            _frames(0, 1),
            _conds(0, 1).unsqueeze(0),
            [model.state[i].detach().unsqueeze(0) for i in range(cfg.n_blocks)],
        )
        assert state[0].shape[-2:] == (cfg.head_dim, cfg.head_dim)
        assert analytical_mac(cfg) == sum(runtime.values()) + breakdown["block_recurrence"]


def test_mac_scales_with_batch_and_blocks():
    assert mac_breakdown(batch=3) == {key: 3 * value for key, value in mac_breakdown().items()}
    assert analytical_mac(_config(n_blocks=1)) == 42_736 - 20_672
    with pytest.raises(ValueError):
        mac_breakdown(batch=0)


# --- causality -------------------------------------------------------------


def test_window_sweep_is_causal_in_frames_and_conditioning():
    model = GDN2Student(_config()).eval()
    window = 3.0 * torch.randn(2, 12, FRAME_FEATURES, generator=_rng(1))
    cond = torch.randn(2, 12, COND_FEATURES, generator=_rng(2))
    with torch.no_grad():
        baseline = model(window, cond)

    for tick in (0, 5, 10):
        changed = window.clone()
        changed[:, tick + 1 :] += 9.0
        with torch.no_grad():
            after = model(changed, cond)
        assert torch.equal(baseline[:, : tick + 1], after[:, : tick + 1])
        assert not torch.allclose(baseline[:, tick + 1 :], after[:, tick + 1 :])

    changed_cond = cond.clone()
    changed_cond[:, 7:] -= 5.0
    with torch.no_grad():
        after = model(window, changed_cond)
    assert torch.equal(baseline[:, :7], after[:, :7])
    assert not torch.allclose(baseline[:, 7:], after[:, 7:])


def test_step_sequence_equals_pure_sweep_and_reset_restores_it():
    model = GDN2Student(_config()).eval()
    window = _frames(3, 8)
    cond = _conds(4, 8)
    with torch.no_grad():
        swept = model(window, cond)[0]
        sequential = torch.stack(
            [torch.cat(model.step(frame, c)) for frame, c in zip(window, cond)]
        )
        assert torch.equal(swept, sequential)
        assert model.state.abs().sum() > 0
        probe, probe_cond = _frames(5, 1)[0], _conds(6, 1)[0]
        reference = GDN2Student(_config()).eval()
        with torch.no_grad():
            expected = torch.cat(reference.step(probe, probe_cond))
        model.reset_state()
        assert torch.equal(model.state, torch.zeros_like(model.state))
        with torch.no_grad():
            assert torch.equal(torch.cat(model.step(probe, probe_cond)), expected)


# --- determinism -----------------------------------------------------------


def test_same_seed_is_bit_identical_and_global_rng_is_unused():
    cfg = _config()
    torch.manual_seed(1)
    first = GDN2Student(cfg).eval()
    torch.manual_seed(999)
    torch.randn(10_000)
    second = GDN2Student(cfg).eval()
    assert all(
        torch.equal(a, b) for a, b in zip(first.state_dict().values(), second.state_dict().values())
    )

    window, cond = _frames(7, 10), _conds(8, 10)
    with torch.no_grad():
        out_a = first(window, cond)
        out_b = second(window, cond)
        out_c = first(window, cond)
        other = GDN2Student(_config(seed=7)).eval()(window, cond)
    assert torch.equal(out_a, out_b)
    assert torch.equal(out_a, out_c)
    assert not torch.equal(out_a, other)

    first.reset_state()
    second.reset_state()
    with torch.no_grad():
        for frame, c in zip(window, cond):
            a = torch.cat(first.step(frame, c))
            b = torch.cat(second.step(frame, c))
            assert torch.equal(a, b)


def test_eval_and_train_modes_agree():
    model = GDN2Student(_config())
    window, cond = _frames(9, 6), _conds(10, 6)
    with torch.no_grad():
        trained = model.train()(window, cond)
        evaluated = model.eval()(window, cond)
    assert torch.equal(trained, evaluated)


# --- bounds ----------------------------------------------------------------


def test_output_and_state_stay_bounded_over_1500_ticks():
    cfg = _config(g_max=0.25)
    model = GDN2Student(cfg).eval()
    window, cond = _frames(11, 1500, scale=40.0), _conds(12, 1500, scale=8.0)
    with torch.no_grad():
        for frame, c in zip(window, cond):
            corrections, confidence = model.step(frame, c)
            assert corrections.shape == (2,) and confidence.shape == (1,)
            assert torch.isfinite(corrections).all() and torch.isfinite(confidence).all()
            assert corrections.abs().max() <= cfg.g_max + 1e-6
            assert 0.0 <= confidence.min() and confidence.max() <= 1.0
        assert torch.isfinite(model.state).all()
        assert torch.equal(model.state, model.state)
        assert model.state.abs().max() <= 4.0


def test_shape_and_config_validation():
    with pytest.raises(ValueError):
        GDN2StudentConfig(width=47)
    with pytest.raises(ValueError):
        GDN2StudentConfig(width=65)
    with pytest.raises(ValueError):
        GDN2StudentConfig(decay=1.0)
    with pytest.raises(ValueError):
        GDN2StudentConfig(out_features=2)

    model = GDN2Student(_config()).eval()
    with pytest.raises(ValueError):
        model(torch.randn(2, 3, FRAME_FEATURES + 1))
    with pytest.raises(ValueError):
        model(torch.randn(2, 0, FRAME_FEATURES))
    with pytest.raises(ValueError):
        model.step(torch.randn(FRAME_FEATURES + 1))
    with pytest.raises(ValueError):
        model(_frames(0, 2), torch.randn(2, 3))
