"""Trainer-facing bridge to split-conformal confidence (``conformal.py``).

The trainer fits on its calibration split and writes ``conformal.json`` beside
its other reports; a disjoint split is scored through the exported payload, the
same way the edge reader consumes it. All statistics stay in ``conformal.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .calibration import (
    DEFAULT_MAX_LOOPS,
    DEFAULT_TARGET_ECE,
    apply_calibration,
    conformal_calibrate,
    correctness,
    expected_calibration_error,
)
from .conformal import (
    DEFAULT_ALPHA,
    DEFAULT_TOL,
    ConformalResult,
    apply_confidence,
    fit_conformal,
)

CONFORMAL_FILE = "conformal.json"

# ECE provenance labels. A disjoint score is the only one allowed to certify a
# report; a stage-local diagnostic stays explicitly in-sample.
DISJOINT_ECE_SOURCE = "held-out"
IN_SAMPLE_ECE_SOURCE = "in-sample"


def frozen_step_map(payload: dict) -> tuple[list[float], list[float]]:
    """Validate and return the edge-facing two-bin step map of an export.

    Frozen shape: two strictly ascending upper edges in ``(0, 1]`` and binary
    levels ``(0, 1)``. The lower edge is inclusive -- the edge reader takes the
    first upper edge strictly greater than ``p`` -- so no runtime tuning is
    involved. A missing, malformed or non-finite map is a hard error.
    """
    result = ConformalResult.from_dict(payload)
    if result.bins is None or result.values is None:
        raise ValueError("conformal export is missing the frozen step map")
    bins, values = list(result.bins), list(result.values)
    if len(bins) != 2 or tuple(values) != (0.0, 1.0):
        raise ValueError(
            f"frozen step map needs two ascending edges and values (0, 1), "
            f"got {bins} / {values}"
        )
    if not all(np.isfinite(v) for v in (*bins, *values)):
        raise ValueError("frozen step map contains NaN or infinity")
    return bins, values


def conformal_export(
    calibration_error: np.ndarray,
    out: str | Path,
    *,
    alpha: float = DEFAULT_ALPHA,
    tol: float = DEFAULT_TOL,
    with_bin_map: bool = True,
) -> ConformalResult:
    """Fit on held-out command errors and write ``<out>/conformal.json``.

    ``with_bin_map`` emits the monotone step map an edge reader applies via
    ``apply_confidence``; it does not change the fit. Empty, non-finite or
    misconfigured inputs fail loudly, exactly as in ``fit_conformal``.
    """
    result = fit_conformal(calibration_error, alpha=alpha, tol=tol, with_bin_map=with_bin_map)
    payload = result.as_dict()
    if not all(np.isfinite(value) for value in payload.values()
               if isinstance(value, (int, float))):
        raise ValueError("refusing to export a non-finite conformal calibration")
    if with_bin_map:
        frozen_step_map(payload)  # LOUD on a missing or non-finite edge map
    destination = Path(out) / CONFORMAL_FILE
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(result.to_json())
    return result


def conformal_coverage(payload: dict, error: np.ndarray) -> float:
    """Empirical coverage of a disjoint split through an exported payload.

    Validation rejects a corrupt export or one from ``calibration.py`` (whose
    temperature/isotonic map has different semantics); scoring then follows the
    edge route, ``apply_confidence``.
    """
    ConformalResult.from_dict(payload)
    return float(apply_confidence(payload, error).mean())


@dataclass(frozen=True)
class SplitScore:
    """Honest disjoint-split score for the trainer's held-out predictions.

    ``ece_held_out`` is measured on the split that was NOT used for fitting;
    ``conformal_coverage`` scores the same split through the exported payload.
    ``alpha`` is the level actually used after the small-sample floor (the
    exported ``conformal.json`` records it too), so no fit is silently scored.
    """

    ece_held_out: float
    conformal_coverage: float
    alpha: float
    n_fit: int
    n_score: int


def split_confidence_score(
    logits: np.ndarray,
    error: np.ndarray,
    out: str | Path,
    *,
    seed: int,
    tol: float = DEFAULT_TOL,
    alpha: float = DEFAULT_ALPHA,
    target_ece: float = DEFAULT_TARGET_ECE,
    max_loops: int = DEFAULT_MAX_LOOPS,
) -> SplitScore:
    """Fit confidence on half the held-out predictions, score the other half.

    The deterministic seed-keyed permutation is the only randomness. The fit
    half feeds both the temperature/isotonic map (scored as ``ece_held_out``)
    and ``conformal_export`` (written to ``<out>/conformal.json`` and scored as
    ``conformal_coverage``). Nothing fitted here is ever scored on the same rows.
    A conformal quantile needs ``n_fit >= ceil(1/alpha) - 1``; when the fit half
    is smaller, ``alpha`` is raised to the smallest valid level and recorded in
    both the export and this result rather than failing or fitting in-sample.
    """
    z = np.asarray(logits, dtype=np.float64).ravel()
    err = np.asarray(error, dtype=np.float64).ravel()
    if z.size != err.size or z.size < 2:
        raise ValueError("disjoint confidence scoring needs at least two paired predictions")
    if not np.all(np.isfinite(z)) or not np.all(np.isfinite(err)):
        raise ValueError("held-out logits and errors must be finite")

    order = np.random.default_rng(seed).permutation(z.size)
    cut = z.size // 2
    fit, score = order[:cut], order[cut:]
    effective_alpha = max(float(alpha), 1.0 / (fit.size + 1))

    fitted = conformal_calibrate(
        z[fit], err[fit], tol=tol, target_ece=target_ece, max_loops=max_loops
    )
    applied = apply_calibration(fitted.as_dict(), z[score])
    ece_held_out = expected_calibration_error(applied, correctness(err[score], tol))

    conformal = conformal_export(err[fit], out, alpha=effective_alpha, tol=tol)
    coverage = conformal_coverage(conformal.as_dict(), err[score])
    return SplitScore(
        ece_held_out=float(ece_held_out),
        conformal_coverage=float(coverage),
        alpha=float(effective_alpha),
        n_fit=int(fit.size),
        n_score=int(score.size),
    )
