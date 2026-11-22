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
    # Isotonic step map: upper edges of raw-sigmoid-confidence intervals and the
    # calibrated value per interval. None when temperature scaling was selected.
    bins: tuple[float, ...] | None = None
    values: tuple[float, ...] | None = None

    def as_dict(self) -> dict:
        payload = {
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
        if self.method == "isotonic" and self.bins is not None and self.values is not None:
            # Top-level `bins` + `values` is the shape the edge reader accepts;
            # temperature files stay byte-for-byte backward compatible.
            payload["bins"] = list(self.bins)
            payload["values"] = list(self.values)
        return payload


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


def isotonic_step_map(confidence: np.ndarray, correct: np.ndarray
                      ) -> tuple[list[float], list[float]]:
    """Collapse the PAVA fit into upper edges and values an edge reader can apply.

    ``isotonic_calibration`` returns one fitted value per sample, constant inside
    each pooled block. Emit one bin per block with the block's largest raw
    confidence as its upper edge: the runtime looks up ``sigmoid(logit)``, maps
    ``p <= edge`` to that block's value, and holds the last value past the final
    edge. This is a pure re-encoding of the same fit, not a different algorithm.
    """
    conf = np.asarray(confidence, dtype=np.float64).ravel()
    ok = np.asarray(correct, dtype=np.float64).ravel()
    fitted = isotonic_calibration(conf, ok)
    order = np.argsort(conf, kind="stable")
    ordered_conf, ordered_fit = conf[order], fitted[order]
    bins: list[float] = []
    values: list[float] = []
    block_start = 0
    for i in range(1, ordered_conf.size + 1):
        if i == ordered_conf.size or ordered_fit[i] != ordered_fit[block_start]:
            edge = float(ordered_conf[i - 1])
            value = float(ordered_fit[block_start])
            if bins and edge <= bins[-1]:
                # Exact confidence ties split across two PAVA blocks cannot be
                # told apart by a step in p: keep the later (larger) fit.
                values[-1] = value
            else:
                bins.append(edge)
                values.append(value)
            block_start = i
    return bins, values


def apply_calibration(payload: dict, logits: np.ndarray) -> np.ndarray:
    """Apply a serialised calibration payload to pre-sigmoid logits.

    Mirrors the edge reader's ``apply_logit``: temperature uses ``sigmoid(z / T)``;
    isotonic looks the raw sigmoid up in the exported step map (first upper edge
    >= p, holding the final value past the last edge). An ``n+1`` edge array is
    accepted and its leading lower edge dropped, as on the edge side.
    """
    z = np.asarray(logits, dtype=np.float64).ravel()
    method = payload["method"]
    if method == "temperature":
        return sigmoid(z / float(payload["temperature"]))
    if method == "isotonic":
        bins = np.asarray(payload["bins"], dtype=np.float64)
        values = np.asarray(payload["values"], dtype=np.float64)
        if bins.size == values.size + 1:
            bins = bins[1:]
        if values.size == 0 or bins.size != values.size:
            raise ValueError("isotonic payload needs n upper edges and n values")
        index = np.minimum(np.searchsorted(bins, sigmoid(z), side="left"), values.size - 1)
        return values[index]
    raise ValueError(f"unknown calibration method {method!r}")


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
    iso_bins: tuple[float, ...] | None = None
    iso_values: tuple[float, ...] | None = None
    if temperature_ece <= target_ece:
        method, calibrated = "temperature", temperature_conf
    else:
        isotonic_conf = isotonic_calibration(raw_conf, ok)
        if expected_calibration_error(isotonic_conf, ok, bins) <= temperature_ece:
            method, calibrated = "isotonic", isotonic_conf
            edges, fitted = isotonic_step_map(raw_conf, ok)
            iso_bins, iso_values = tuple(edges), tuple(fitted)
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
        bins=iso_bins,
        values=iso_values,
    )


def confidence_target(
    error: torch.Tensor, tol: float = DEFAULT_TOL, scale: torch.Tensor | float | None = None
) -> torch.Tensor:
    """Smooth correctness target with a *data-driven* error scale.

    ``exp(-error/tol)`` is ~0 for every sample when the model's errors are much
    larger than ``tol`` (error 1.0 with tol 0.1 gives 4.5e-5). Regressing a
    sigmoid toward that value drives its logit to a constant negative number,
    so calibration sees a collapsed head and reports a trivially small ECE.
    Scaling by the batch's own mean error keeps a spread of targets at any
    error scale; calibration is then free to map the logits to empirical
    correctness. When the model already resolves errors below ``tol`` the
    original behaviour is unchanged (the scale is floored at ``tol``).
    """
    e = error.detach().to(torch.float32)
    if scale is None:
        scale = e.mean().clamp_min(tol)
    s = torch.as_tensor(scale, dtype=e.dtype).clamp_min(1e-6)
    return torch.exp(-e / s)


def confidence_spread_loss(confidence: torch.Tensor, error: torch.Tensor,
                           tol: float = DEFAULT_TOL,
                           weight: float = MAX_SPREAD_WEIGHT) -> torch.Tensor:
    """Reward confidence spread on uncertain cells, with a hard weight cap."""
    if not 0.0 <= weight <= MAX_SPREAD_WEIGHT:
        raise ValueError(f"spread weight must be in [0, {MAX_SPREAD_WEIGHT}]")
    uncertain = (error.detach() > tol).to(confidence.dtype).mean()
    return -weight * uncertain * confidence.var(unbiased=False)
