"""Split-conformal confidence calibration (replaces in-sample isotonic).

A held-out *calibration* set of command errors ``e_i`` is normalised by a
documented tolerance ``tol``: ``s_i = |e_i| / tol``. A score of 1 means the
error is exactly at the tolerance; the sign of the error is ignored. The
conformal quantile uses the standard finite-sample correction: with ``n``
scores it is the ``k``-th smallest score (1-based) where
``k = ceil((n + 1) * (1 - alpha))``. A new prediction is *confident* iff its
normalised error is at most ``q_hat``. By exchangeability this covers a fresh
sample with probability at least ``1 - alpha``; equivalently the prediction
interval is ``center ± q_hat * tol``.

Nothing is fitted in-sample and no logits are consulted: the only input is the
held-out error, so a fit is a pure deterministic function of its data. Empty,
non-finite or misconfigured inputs raise loudly -- there is no silent fallback.

Edge export: the optional monotone bin map encodes the binary indicator as a
step map over ``p = 1 / (1 + s)`` (a confidence coordinate in ``(0, 1]`` that
decreases with error). ``bins`` are ascending upper edges and ``values`` are
0/1, so a reader that already applies ``first upper edge >= p`` needs no new
machinery (see ``apply_confidence`` and the usage note in the module report).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np

DEFAULT_TOL = 0.1
DEFAULT_ALPHA = 0.05
METHOD = "split-conformal"

# Keep the two step-map edges strictly increasing even at q_hat = 0 or q_hat huge.
_EDGE_EPS = 1e-12


def conformal_scores(error: np.ndarray, tol: float = DEFAULT_TOL) -> np.ndarray:
    """Normalised absolute errors ``|error| / tol``; LOUD on empty/NaN/inf/bad tol."""
    t = float(tol)
    if not np.isfinite(t) or t <= 0.0:
        raise ValueError(f"tol must be a finite positive number, got {tol!r}")
    raw = np.asarray(error, dtype=np.float64).ravel()
    if raw.size == 0:
        raise ValueError("calibration scores must be non-empty")
    scores = np.abs(raw) / t
    if not np.all(np.isfinite(scores)):
        raise ValueError("calibration scores contain NaN or infinity")
    return scores


def conformal_quantile(scores: np.ndarray, alpha: float = DEFAULT_ALPHA) -> float:
    """Finite-sample conformal quantile ``s_(ceil((n+1)(1-alpha)))``.

    Returns a value in score units (multiples of ``tol``). Requires enough
    calibration points for the order statistic to exist, else raises LOUDLY.
    """
    a = float(alpha)
    if not np.isfinite(a) or not 0.0 < a < 1.0:
        raise ValueError(f"alpha must lie strictly in (0, 1), got {alpha!r}")
    s = np.asarray(scores, dtype=np.float64).ravel()
    if s.size == 0:
        raise ValueError("calibration scores must be non-empty")
    if not np.all(np.isfinite(s)):
        raise ValueError("calibration scores contain NaN or infinity")
    n = s.size
    k = math.ceil((n + 1) * (1.0 - a))
    if k > n:
        raise ValueError(
            f"alpha={a} too small for n={n}: need n >= {math.ceil(1.0 / a) - 1}"
        )
    return float(np.sort(s)[k - 1])


def monotone_bin_map(q_hat: float) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Encode ``s <= q_hat`` as a monotone step map over ``p = 1/(1+s)``.

    ``confident iff p >= 1/(1+q_hat)``. Emits two strictly ascending edges and
    non-decreasing values ``(0, 1)``, matching the edge reader's ``n``-edge
    ``bins``/``values`` shape. Uses ``apply_confidence`` to evaluate it.
    """
    q = float(q_hat)
    if not np.isfinite(q) or q < 0.0:
        raise ValueError(f"q_hat must be a finite non-negative number, got {q_hat!r}")
    p_hat = 1.0 / (1.0 + q)
    p_hat = min(max(p_hat, _EDGE_EPS), 1.0 - _EDGE_EPS)
    return (p_hat, 1.0), (0.0, 1.0)


@dataclass(frozen=True)
class ConformalResult:
    """A frozen split-conformal calibration, ready to serialise or apply."""

    alpha: float
    tol: float
    q_hat: float
    n: int
    coverage: float
    bins: tuple[float, ...] | None = None
    values: tuple[float, ...] | None = None

    @property
    def half_width(self) -> float:
        """Prediction-interval half-width in the error's own units."""
        return self.q_hat * self.tol

    def is_confident(self, error: np.ndarray) -> np.ndarray:
        """Binary indicator: ``True`` where the error lies within ``q_hat``."""
        return conformal_scores(error, self.tol) <= self.q_hat

    def predict_interval(self, center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Conformal interval ``center ± q_hat * tol`` (never materialises 4D)."""
        c = np.asarray(center, dtype=np.float64)
        return c - self.half_width, c + self.half_width

    def as_dict(self) -> dict:
        payload = {
            "method": METHOD,
            "alpha": float(self.alpha),
            "tol": float(self.tol),
            "q_hat": float(self.q_hat),
            "n": int(self.n),
            "coverage": float(self.coverage),
            "half_width": float(self.half_width),
        }
        if self.bins is not None and self.values is not None:
            payload["bins"] = list(self.bins)
            payload["values"] = list(self.values)
        return payload

    def to_json(self) -> str:
        """Deterministic bytes: sorted keys, fixed indentation, trailing newline."""
        return json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, payload: dict) -> ConformalResult:
        if payload.get("method") != METHOD:
            raise ValueError(f"expected method {METHOD!r}, got {payload.get('method')!r}")
        alpha, tol = float(payload["alpha"]), float(payload["tol"])
        q_hat, n = float(payload["q_hat"]), int(payload["n"])
        if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must lie strictly in (0, 1), got {alpha!r}")
        if not np.isfinite(tol) or tol <= 0.0:
            raise ValueError(f"tol must be a finite positive number, got {tol!r}")
        if not np.isfinite(q_hat) or q_hat < 0.0:
            raise ValueError(f"q_hat must be a finite non-negative number, got {q_hat!r}")
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        bins, values = payload.get("bins"), payload.get("values")
        if (bins is None) != (values is None):
            raise ValueError("bins and values must be supplied together")
        bins_t = values_t = None
        if bins is not None:
            b = np.asarray(bins, dtype=np.float64)
            v = np.asarray(values, dtype=np.float64)
            if b.size == v.size + 1:  # tolerate the reader's leading lower edge
                b = b[1:]
            if v.size == 0 or b.size != v.size:
                raise ValueError("bin map needs n upper edges and n values")
            if b[0] <= 0.0 or b[-1] > 1.0 or np.any(np.diff(b) <= 0.0):
                raise ValueError("bin edges must be strictly increasing in (0, 1]")
            if np.any((v < 0.0) | (v > 1.0)) or np.any(np.diff(v) < 0.0):
                raise ValueError("bin values must be monotone in [0, 1]")
            bins_t, values_t = tuple(float(x) for x in b), tuple(float(x) for x in v)
        return cls(
            alpha=alpha, tol=tol, q_hat=q_hat, n=n,
            coverage=float(payload["coverage"]), bins=bins_t, values=values_t,
        )

    @classmethod
    def from_json(cls, text: str) -> ConformalResult:
        return cls.from_dict(json.loads(text))


def fit_conformal(
    calibration_error: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
    tol: float = DEFAULT_TOL,
    *,
    with_bin_map: bool = False,
) -> ConformalResult:
    """Fit split-conformal confidence on held-out errors.

    ``alpha`` is the miscoverage level (coverage target ``1 - alpha``) and
    ``tol`` is the documented error tolerance that normalises the scores.
    ``with_bin_map`` adds the monotone 2-bin edge export; it does not change
    the fit.
    """
    scores = conformal_scores(calibration_error, tol)
    q_hat = conformal_quantile(scores, alpha)
    bins = values = None
    if with_bin_map:
        edges, levels = monotone_bin_map(q_hat)
        bins, values = tuple(edges), tuple(levels)
    return ConformalResult(
        alpha=float(alpha),
        tol=float(tol),
        q_hat=q_hat,
        n=int(scores.size),
        coverage=float(np.mean(scores <= q_hat)),
        bins=bins,
        values=values,
    )


def apply_confidence(payload: dict, error: np.ndarray) -> np.ndarray:
    """Apply a serialised calibration to raw errors (the future edge path).

    With a bin map, maps ``p = 1/(1 + s)`` through the first upper edge ``>= p``
    and holds the last value past the end -- identical to the existing edge
    step-map reader. Without one, applies the exact ``s <= q_hat`` indicator.
    Both routes return a boolean confidence array.
    """
    s = conformal_scores(error, float(payload["tol"]))
    bins = payload.get("bins")
    if bins is None:
        return s <= float(payload["q_hat"])
    b = np.asarray(bins, dtype=np.float64)
    v = np.asarray(payload["values"], dtype=np.float64)
    if b.size == v.size + 1:
        b = b[1:]
    if v.size == 0 or b.size != v.size:
        raise ValueError("bin map needs n upper edges and n values")
    p = 1.0 / (1.0 + s)
    index = np.minimum(np.searchsorted(b, p, side="left"), v.size - 1)
    return v[index].astype(bool)
