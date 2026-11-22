"""Confidence calibration on held-out logits (SPEC_ENGINE_IDEAL.md §5).

The confidence head's pre-sigmoid scalar is the calibration input. A sample is
"correct" when its command error (max absolute correction error) is within
``tol``. Temperature scaling is tried first with a bounded search; when a
saturated head makes it ineffective, a pool-adjacent-violators isotonic map is
fitted instead. ECE before and after is always the measured value on the held-out
set -- a failure to reach the 0.05 target is reported, never hidden. The
adaptive-compute threshold is a quantile of the calibrated confidence, not 0.5.

``confidence_target`` and ``confidence_spread_loss`` are the training-side
counterpart: the head is supervised toward a smooth correctness target rather
than a constant, and a small reward keeps its outputs spread on uncertain cells.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

DEFAULT_TOL = 0.1
DEFAULT_TARGET_ECE = 0.05
DEFAULT_MAX_LOOPS = 3
BINS = 10
MAX_SPREAD_WEIGHT = 0.05


@dataclass(frozen=True)
class CalibrationResult:
    threshold: float  # adaptive-compute threshold, from held-out data
    ece: float  # ECE after calibration
    raw_ece: float  # ECE of the uncalibrated sigmoid(logit)
    temperature_ece: float  # ECE after temperature alone
    coverage: float  # empirical correct fraction
    temperature: float
    method: str  # "temperature" or "isotonic"
    loops: int  # search passes, <= max_loops
    n: int
    target_ece: float = DEFAULT_TARGET_ECE

    def as_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "ece": self.ece,
            "raw_ece": self.raw_ece,
            "temperature_ece": self.temperature_ece,
            "coverage": self.coverage,
            "temperature": self.temperature,
            "method": self.method,
            "loops": self.loops,
            "n": self.n,
            "target_ece": self.target_ece,
        }


def correctness(error: np.ndarray, tol: float = DEFAULT_TOL) -> np.ndarray:
    """Binary empirical correctness from per-sample command errors."""
    return (np.asarray(error, dtype=np.float64).ravel() <= tol).astype(np.float64)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid of held-out logits."""
    z = np.clip(np.asarray(logits, dtype=np.float64).ravel(), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray,
                               bins: int = BINS) -> float:
    """Standard equal-width ECE between predicted confidence and correctness."""
    conf = np.asarray(confidence, dtype=np.float64).ravel()
    ok = np.asarray(correct, dtype=np.float64).ravel()
    if conf.shape != ok.shape or conf.size == 0:
        raise ValueError("confidence and correctness must be equal, non-empty vectors")
    if np.any((conf < 0.0) | (conf > 1.0)):
        raise ValueError("confidences must lie in [0, 1]")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        low, high = edges[index], edges[index + 1]
        inside = (conf >= low) & (conf < high if index < bins - 1 else conf <= high)
        if np.any(inside):
            total += inside.mean() * abs(conf[inside].mean() - ok[inside].mean())
    return float(total)


def isotonic_calibration(confidence: np.ndarray, correct: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators fit of empirical correctness on confidence.

    A constant head collapses to one pool and is mapped to its overall accuracy,
    which is exactly the case temperature scaling cannot repair.
    """
    conf = np.asarray(confidence, dtype=np.float64).ravel()
    ok = np.asarray(correct, dtype=np.float64).ravel()
    if conf.shape != ok.shape:
        raise ValueError("confidence and correctness must have the same length")
    order = np.argsort(conf, kind="stable")
    totals: list[float] = []
    counts: list[int] = []
    for target in ok[order]:
        totals.append(float(target))
        counts.append(1)
        while len(totals) > 1 and totals[-2] / counts[-2] > totals[-1] / counts[-1]:
            total, count = totals.pop(), counts.pop()
            totals[-1] += total
            counts[-1] += count
    fitted = np.empty(conf.size, dtype=np.float64)
    position = 0
    for total, count in zip(totals, counts):
        fitted[position : position + count] = total / count
        position += count
    result = np.empty(conf.size, dtype=np.float64)
    result[order] = fitted
    return result


def conformal_calibrate(logits: np.ndarray, error: np.ndarray,
                        tol: float = DEFAULT_TOL,
                        target_ece: float = DEFAULT_TARGET_ECE,
                        max_loops: int = DEFAULT_MAX_LOOPS,
                        bins: int = BINS) -> CalibrationResult:
    """Fit temperature (then isotonic if needed) and derive the threshold.

    The temperature search is bounded by ``max_loops`` passes. Isotonic is only
    consulted when temperature misses the ECE target; the lower-ECE of the two
    is reported with its method name. ECE is measured in-sample on the held-out
    scenarios, so ``n`` is part of the result for honesty about small samples.
    """
    if max_loops < 1:
        raise ValueError("max_loops must be at least one")
    z = np.asarray(logits, dtype=np.float64).ravel()
    ok = correctness(error, tol)
    if z.shape != ok.shape:
        raise ValueError("logits and error must have the same length")

    raw_conf = sigmoid(z)
    raw_ece = expected_calibration_error(raw_conf, ok, bins)
    grid = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0])
    temperature, best = 1.0, raw_ece
    loops = 0
    for loop in range(max_loops):
        loops = loop + 1
        scored = sorted(
            (expected_calibration_error(sigmoid(z / t), ok, bins), float(t)) for t in grid
        )
        score, candidate = scored[0]
        if score < best - 1e-12:
            best, temperature = score, candidate
        if best <= target_ece:
            break
        grid = np.linspace(max(0.1, candidate / 2.0), min(20.0, candidate * 2.0), 5)

    temperature_conf = sigmoid(z / max(temperature, 1e-6))
    temperature_ece = expected_calibration_error(temperature_conf, ok, bins)
    if temperature_ece <= target_ece:
        method, calibrated = "temperature", temperature_conf
    else:
        isotonic_conf = isotonic_calibration(raw_conf, ok)
        if expected_calibration_error(isotonic_conf, ok, bins) <= temperature_ece:
            method, calibrated = "isotonic", isotonic_conf
        else:
            method, calibrated = "temperature", temperature_conf

    return CalibrationResult(
        threshold=float(np.quantile(calibrated, target_ece)),
        ece=expected_calibration_error(calibrated, ok, bins),
        raw_ece=raw_ece,
        temperature_ece=temperature_ece,
        coverage=float(ok.mean()),
        temperature=float(temperature),
        method=method,
        loops=loops,
        n=int(z.size),
        target_ece=float(target_ece),
    )


def confidence_target(error: torch.Tensor, tol: float = DEFAULT_TOL) -> torch.Tensor:
    """Smooth correctness target: 1 at zero error, exp(-1) at the tolerance."""
    return torch.exp(-error.detach().to(torch.float32) / max(tol, 1e-6))


def confidence_spread_loss(confidence: torch.Tensor, error: torch.Tensor,
                           tol: float = DEFAULT_TOL,
                           weight: float = MAX_SPREAD_WEIGHT) -> torch.Tensor:
    """Reward confidence spread on uncertain cells, with a hard weight cap."""
    if not 0.0 <= weight <= MAX_SPREAD_WEIGHT:
        raise ValueError(f"spread weight must be in [0, {MAX_SPREAD_WEIGHT}]")
    uncertain = (error.detach() > tol).to(confidence.dtype).mean()
    return -weight * uncertain * confidence.var(unbiased=False)
