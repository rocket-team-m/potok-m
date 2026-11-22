"""Conformal-style confidence calibration (SPEC_ENGINE_IDEAL.md §5).

Confidence is calibrated on held-out scenarios only. A sample is "correct" when
its command error (max absolute correction error) is within ``tol``. The
calibration map is a temperature on the confidence logit, chosen by a bounded
search so that the 10-bin expected calibration error (ECE) reaches the 0.05
target. The adaptive-compute threshold is a split-conformal empirical quantile
of the calibrated confidence -- it is measured on held-out data, never the
hard-coded 0.5. The search makes at most ``max_loops`` passes; the ECE actually
achieved is always reported, whether or not the target was met.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_TOL = 0.1
DEFAULT_TARGET_ECE = 0.05
DEFAULT_MAX_LOOPS = 3
BINS = 10


@dataclass(frozen=True)
class CalibrationResult:
    threshold: float  # adaptive-compute threshold, from held-out data
    ece: float  # ECE after calibration
    raw_ece: float  # ECE before calibration
    coverage: float  # empirical correct fraction
    temperature: float
    loops: int  # search passes, <= max_loops
    n: int
    target_ece: float = DEFAULT_TARGET_ECE

    def as_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "ece": self.ece,
            "raw_ece": self.raw_ece,
            "coverage": self.coverage,
            "temperature": self.temperature,
            "loops": self.loops,
            "n": self.n,
            "target_ece": self.target_ece,
        }


def correctness(error: np.ndarray, tol: float = DEFAULT_TOL) -> np.ndarray:
    """Binary empirical correctness from per-sample command errors."""
    return (np.asarray(error, dtype=np.float64).ravel() <= tol).astype(np.float64)


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


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _temperature_scale(confidence: np.ndarray, temperature: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-_logit(confidence) / max(temperature, 1e-6)))


def conformal_calibrate(confidence: np.ndarray, error: np.ndarray,
                        tol: float = DEFAULT_TOL,
                        target_ece: float = DEFAULT_TARGET_ECE,
                        max_loops: int = DEFAULT_MAX_LOOPS,
                        bins: int = BINS) -> CalibrationResult:
    """Fit the temperature and derive the adaptive-compute threshold.

    The search is bounded: a coarse temperature grid, then a refinement around
    the best candidate, for at most ``max_loops`` passes. It stops early once
    the ECE target is met. The threshold is the ``target_ece`` quantile of the
    calibrated confidence, i.e. the bottom slice that gets extra refinement.
    """
    if max_loops < 1:
        raise ValueError("max_loops must be at least one")
    conf = np.asarray(confidence, dtype=np.float64).ravel()
    ok = correctness(error, tol)
    if conf.shape != ok.shape:
        raise ValueError("confidence and error must have the same length")

    raw_ece = expected_calibration_error(conf, ok, bins)
    grid = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0])
    temperature, best = 1.0, raw_ece
    loops = 0
    for loop in range(max_loops):
        loops = loop + 1
        scored = sorted(
            (expected_calibration_error(_temperature_scale(conf, t), ok, bins), float(t))
            for t in grid
        )
        score, candidate = scored[0]
        if score < best - 1e-12:
            best, temperature = score, candidate
        if best <= target_ece:
            break
        grid = np.linspace(max(0.1, candidate / 2.0), min(20.0, candidate * 2.0), 5)

    calibrated = _temperature_scale(conf, temperature)
    return CalibrationResult(
        threshold=float(np.quantile(calibrated, target_ece)),
        ece=expected_calibration_error(calibrated, ok, bins),
        raw_ece=raw_ece,
        coverage=float(ok.mean()),
        temperature=float(temperature),
        loops=loops,
        n=int(conf.size),
        target_ece=float(target_ece),
    )
