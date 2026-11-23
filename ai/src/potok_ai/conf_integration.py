"""Trainer-facing bridge to split-conformal confidence (``conformal.py``).

The trainer fits on its calibration split and writes ``conformal.json`` beside
its other reports; a disjoint split is scored through the exported payload, the
same way the edge reader consumes it. All statistics stay in ``conformal.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .conformal import (
    DEFAULT_ALPHA,
    DEFAULT_TOL,
    ConformalResult,
    apply_confidence,
    fit_conformal,
)

CONFORMAL_FILE = "conformal.json"


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
