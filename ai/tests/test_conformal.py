"""Split-conformal confidence: analytic fit, coverage, loud failures, JSON."""

import json
import math
from statistics import NormalDist

import numpy as np
import pytest

from potok_ai.conformal import (
    ConformalResult,
    apply_confidence,
    conformal_quantile,
    conformal_scores,
    fit_conformal,
    monotone_bin_map,
)


def test_scores_normalize_by_the_documented_tolerance():
    error = np.array([-0.2, 0.0, 0.1, 0.5])
    # Sign is ignored; a score of 1 means "error exactly at tol".
    assert conformal_scores(error, tol=0.5).tolist() == pytest.approx(
        [0.4, 0.0, 0.2, 1.0]
    )
    assert conformal_scores(np.array([0.3])).tolist() == pytest.approx([3.0])


def test_synthetic_gaussian_errors_match_the_analytic_quantile():
    """Half-normal errors: q_hat must land on the analytic conformal quantile."""
    sigma, tol, alpha, n = 1.0, 0.5, 0.1, 20_000
    rng = np.random.default_rng(11)
    errors = rng.normal(0.0, sigma, n)

    result = fit_conformal(errors, alpha=alpha, tol=tol)
    analytic = (sigma / tol) * NormalDist().inv_cdf(1.0 - alpha / 2.0)
    assert result.q_hat == pytest.approx(analytic, rel=0.03)
    assert result.n == n
    assert result.coverage == pytest.approx(1.0 - alpha, abs=0.02)
    assert result.half_width == pytest.approx(result.q_hat * tol)
    assert result.values is None and result.bins is None

    lo, hi = result.predict_interval(np.array([0.0, 10.0]))
    assert lo.tolist() == pytest.approx([-result.half_width, 10.0 - result.half_width])
    assert hi.tolist() == pytest.approx([result.half_width, 10.0 + result.half_width])


def test_coverage_near_one_minus_alpha_on_large_held_out_sample():
    """Fresh held-out errors are covered ~1-alpha of the time (conformal)."""
    alpha, tol = 0.05, 0.3
    rng = np.random.default_rng(2026)
    calibration = rng.normal(0.0, 1.0, 30_000)
    held_out = rng.normal(0.0, 1.0, 30_000)

    result = fit_conformal(calibration, alpha=alpha, tol=tol)
    covered = result.is_confident(held_out)
    assert covered.dtype == np.bool_
    assert covered.mean() == pytest.approx(1.0 - alpha, abs=0.01)
    assert covered.mean() >= 1.0 - alpha - 0.005  # finite-sample guarantee


def test_empty_nan_and_bad_config_are_loud():
    with pytest.raises(ValueError, match="non-empty"):
        fit_conformal(np.array([]))
    with pytest.raises(ValueError, match="non-empty"):
        conformal_scores(np.array([]))
    for bad in (np.array([np.nan]), np.array([1.0, np.inf]), np.array([-np.inf])):
        with pytest.raises(ValueError, match="NaN or infinity"):
            fit_conformal(bad)
        with pytest.raises(ValueError, match="NaN or infinity"):
            conformal_quantile(bad)
    with pytest.raises(ValueError, match="tol"):
        fit_conformal(np.array([0.1, 0.2]), tol=0.0)
    with pytest.raises(ValueError, match="tol"):
        fit_conformal(np.array([0.1, 0.2]), tol=float("nan"))
    with pytest.raises(ValueError, match="alpha"):
        fit_conformal(np.array([0.1, 0.2]), alpha=1.0)
    with pytest.raises(ValueError, match="too small for n"):
        fit_conformal(np.array([0.1, 0.2]), alpha=0.01)  # n=2 < ceil(1/0.01)-1


def test_monotone_bin_map_matches_the_binary_indicator():
    result = fit_conformal(np.abs(np.random.default_rng(3).normal(0, 1, 5000)),
                           alpha=0.1, tol=0.4, with_bin_map=True)
    assert result.bins is not None and result.values is not None
    assert result.bins == pytest.approx([1 / (1 + result.q_hat), 1.0])
    assert result.values == (0.0, 1.0)

    b, v = monotone_bin_map(result.q_hat)
    assert b[0] < b[1] and v[0] < v[1]  # strictly increasing / monotone

    probes = np.linspace(0.0, 5.0, 101)
    assert apply_confidence(result.as_dict(), probes).tolist() == (
        result.is_confident(probes).tolist()
    )


def test_bin_map_and_exact_indicator_agree_at_p_hat():
    """Canonical boundary: inclusive lower edge -- both routes confident at equality."""
    errors = np.full(19, 0.5)
    result = fit_conformal(errors, alpha=0.05, tol=0.2, with_bin_map=True)
    assert result.q_hat == pytest.approx(2.5)
    p_hat = 1.0 / (1.0 + result.q_hat)

    # Raw error whose normalised score is exactly q_hat -> p == p_hat.
    boundary = np.array([result.q_hat * result.tol])
    assert 1.0 / (1.0 + boundary[0] / result.tol) == p_hat
    for payload in (result.as_dict(), json.loads(result.to_json())):
        assert apply_confidence(payload, boundary).tolist() == [True]
        assert result.is_confident(boundary).tolist() == [True]

    # Both routes still agree immediately below and above the boundary.
    probes = np.array([0.4, 0.5, 0.6])  # scores 2.0, 2.5, 3.0
    expected = [True, True, False]
    assert apply_confidence(result.as_dict(), probes).tolist() == expected
    assert result.is_confident(probes).tolist() == expected


def test_json_round_trip_with_and_without_bin_map():
    errors = np.abs(np.random.default_rng(5).normal(0, 1, 2000))
    for with_map in (False, True):
        result = fit_conformal(errors, alpha=0.1, tol=0.25, with_bin_map=with_map)
        text = result.to_json()
        assert text.endswith("\n") and text == result.to_json()  # deterministic

        restored = ConformalResult.from_json(text)
        assert restored == result
        assert restored.as_dict() == json.loads(text)
        assert ("bins" in restored.as_dict()) is with_map

        # The serialised object is self-sufficient: no raw scores are persisted.
        payload = json.loads(text)
        probes = np.array([0.0, 0.1, 0.5, 1.0, 2.5])
        assert apply_confidence(payload, probes).tolist() == (
            result.is_confident(probes).tolist()
        )

    # The edge reader also accepts n+1 edges with a leading lower edge.
    payload = fit_conformal(errors, with_bin_map=True).as_dict()
    extended = dict(payload, bins=[0.0, *payload["bins"]])
    assert ConformalResult.from_dict(extended).bins == tuple(payload["bins"])
    probes = np.array([0.0, 0.5, 2.0])
    assert apply_confidence(extended, probes).tolist() == (
        apply_confidence(payload, probes).tolist()
    )


def test_non_finite_thresholds_are_loud():
    good = fit_conformal(np.array([0.1, 0.3, 0.7, 1.2]), alpha=0.5,
                         with_bin_map=True).as_dict()
    for bins in ([float("nan"), 1.0], [0.5, float("inf")]):
        with pytest.raises(ValueError, match="NaN or infinity"):
            ConformalResult.from_dict(dict(good, bins=bins))
        with pytest.raises(ValueError, match="NaN or infinity"):
            apply_confidence(dict(good, bins=bins), np.array([0.1, 0.3]))
    for bad in (dict(good, values=[0.0, float("nan")]),
                dict(good, bins=None, q_hat=float("nan"))):
        with pytest.raises(ValueError, match="NaN or infinity|finite non-negative"):
            apply_confidence(bad, np.array([0.1, 0.3]))


def test_json_rejects_corrupt_payloads():
    good = fit_conformal(np.array([0.1, 0.3, 0.7]), alpha=0.5,
                         with_bin_map=True).as_dict()
    with pytest.raises(ValueError, match="method"):
        ConformalResult.from_dict(dict(good, method="isotonic"))
    with pytest.raises(ValueError, match="together"):
        ConformalResult.from_dict({k: v for k, v in good.items() if k != "values"})
    with pytest.raises(ValueError, match="strictly increasing"):
        ConformalResult.from_dict(dict(good, bins=[0.9, 0.9]))
    with pytest.raises(ValueError, match="non-negative"):
        ConformalResult.from_dict(dict(good, q_hat=-1.0))
    with pytest.raises(KeyError):
        ConformalResult.from_dict({"method": "split-conformal"})
    assert math.isclose(good["q_hat"] * good["tol"], good["half_width"])
