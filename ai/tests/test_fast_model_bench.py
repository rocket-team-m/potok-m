# SPDX-License-Identifier: GPL-3.0-or-later
"""AI04: exact MAC accounting and a deterministic CPU p50/p99 micro-benchmark.

Scope is deliberately narrow. This file pins the exact multiply-accumulate count
of ``potok_ai.fast_model.FastModel`` (analytical breakdown, total, and
hook-counted runtime agreement) and records CPU inference latency as p50/p99
percentiles computed by a deterministic, pinned helper.

There is NO architecture, accuracy or latency claim here: no comparison to any
other model, no "faster than" statement, and no machine-specific wall-clock
threshold. Timings are checked only for structural invariants (positive,
finite, ordered, reproducible statistic). Inference runs on CPU only; no GPU
API is used.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from potok_ai.fast_model import FastModel, FastModelConfig, analytical_mac, mac_breakdown


def _rng(seed: int) -> torch.Generator:
    """Private RNG so fixtures never mutate the global torch RNG stream."""
    return torch.Generator().manual_seed(seed)


# --- exact MAC -----------------------------------------------------------

_DEFAULT_BREAKDOWN = {
    "stem": 22_400,
    "depthwise_conv": 4_800,
    "state_in": 12_800,
    "state_out": 12_800,
    "head": 96,
}


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

    def make_hook(name: str):
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
            counts[name] += int(per_call)

        return count

    handles = [target.register_forward_hook(make_hook(name)) for name, target in targets.items()]
    try:
        with torch.no_grad():
            model(window)
    finally:
        for handle in handles:
            handle.remove()
    return counts


def test_exact_mac_is_pinned_and_matches_runtime():
    assert mac_breakdown() == _DEFAULT_BREAKDOWN
    assert analytical_mac() == sum(_DEFAULT_BREAKDOWN.values()) == 52_896

    for cfg, batch in (
        (FastModelConfig(), 2),
        (FastModelConfig(loops=2, window=7, kernel=5, rank=3, in_features=4, width=16), 2),
    ):
        model = FastModel(cfg).eval()
        window = torch.randn(batch, cfg.window, cfg.in_features, generator=_rng(0))
        breakdown = mac_breakdown(cfg, batch=batch)
        assert analytical_mac(cfg, batch=batch) == sum(breakdown.values())
        assert _runtime_mac(model, window) == breakdown


# --- deterministic percentiles -------------------------------------------


def _percentile(samples: list[float], q: float) -> float:
    """Pinned linear-interpolation percentile (numpy ``method="linear"``).

    A pure function of its inputs: identical samples always yield an identical
    number, independent of the clock, the machine or the call order.
    """
    if not samples:
        raise ValueError("percentile needs at least one sample")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def test_percentiles_are_deterministic_and_pinned():
    samples = [float(v) for v in range(1, 10)]  # 1..9

    assert _percentile(samples, 0.0) == pytest.approx(1.0)
    assert _percentile(samples, 0.5) == pytest.approx(5.0)
    assert _percentile(samples, 0.99) == pytest.approx(8.92)
    assert _percentile(samples, 1.0) == pytest.approx(9.0)

    # Cross-check the helper against the reference implementation.
    assert _percentile(samples, 0.5) == pytest.approx(
        float(np.percentile(samples, 50, method="linear"))
    )
    assert _percentile(samples, 0.99) == pytest.approx(
        float(np.percentile(samples, 99, method="linear"))
    )

    # Order-independent and bit-identical across repeated calls.
    shuffled = [samples[i] for i in (3, 0, 7, 1, 8, 2, 5, 4, 6)]
    assert _percentile(shuffled, 0.99) == _percentile(samples, 0.99)
    assert _percentile(samples, 0.99) == _percentile(samples, 0.99)

    with pytest.raises(ValueError):
        _percentile([], 0.5)
    with pytest.raises(ValueError):
        _percentile(samples, 1.5)


# --- CPU-only p50/p99 benchmark ------------------------------------------


@pytest.fixture
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _latency_samples_ns(
    model: FastModel, window: torch.Tensor, warmup: int, iters: int
) -> list[float]:
    with torch.no_grad():
        for _ in range(warmup):
            model(window)
        samples: list[float] = []
        for _ in range(iters):
            started = time.perf_counter_ns()
            model(window)
            samples.append(float(time.perf_counter_ns() - started))
    return samples


def test_cpu_benchmark_reports_deterministic_p50_p99(single_thread):
    cfg = FastModelConfig(window=12, width=8, rank=2, seed=0)
    model = FastModel(cfg).eval()
    window = torch.randn(2, cfg.window, cfg.in_features, generator=_rng(1234))

    # No GPU is involved: both the input and every parameter live on CPU.
    assert window.device.type == "cpu"
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())

    warmup, iters = 3, 32
    samples = _latency_samples_ns(model, window, warmup=warmup, iters=iters)

    assert len(samples) == iters
    assert all(0.0 < sample < float("inf") for sample in samples)

    p50 = _percentile(samples, 0.5)
    p99 = _percentile(samples, 0.99)

    # The reported statistics are a deterministic function of the samples.
    assert p50 == _percentile(samples, 0.5)
    assert p99 == _percentile(samples, 0.99)
    assert 0.0 < p50 <= p99 < float("inf")

    # Inference itself is deterministic: same seed, same input, same output.
    with torch.no_grad():
        out_a = model(window)
        out_b = model(window)
    assert torch.equal(out_a, out_b)
