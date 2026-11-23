# SPDX-License-Identifier: GPL-3.0-or-later
"""Hard gate for split-conformal coverage, export/import and refusal semantics.

Coverage is always measured on a *disjoint* split through the serialised payload
(the edge route). The ``coverage`` field in an export is inert metadata: a gate
that reported it as proof would certify the fit on its own calibration data.
Corrupt fit inputs and corrupt exports fail loudly.
"""

import json

import numpy as np
import pytest

from potok_ai.conf_integration import CONFORMAL_FILE, conformal_coverage, conformal_export
from potok_ai.conformal import ConformalResult, apply_confidence, fit_conformal

EXPECTED_KEYS = {
    "method",
    "alpha",
    "tol",
    "q_hat",
    "n",
    "coverage",
    "half_width",
    "bins",
    "values",
}


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_split_conformal_covers_disjoint_split_at_alpha(tmp_path, alpha):
    tol = 0.3
    rng = np.random.default_rng(2026)
    calibration = rng.normal(0.0, 1.0, 30_000)
    fresh = rng.normal(0.0, 1.0, 30_000)  # disjoint: never scored on the fit split

    result = conformal_export(calibration, tmp_path, alpha=alpha, tol=tol)
    payload = json.loads((tmp_path / CONFORMAL_FILE).read_text())

    assert set(payload) == EXPECTED_KEYS
    assert payload["method"] == "split-conformal"
    assert payload["alpha"] == pytest.approx(alpha)
    assert payload["n"] == calibration.size
    assert payload["half_width"] == pytest.approx(payload["q_hat"] * tol)

    covered = conformal_coverage(payload, fresh)
    assert covered == pytest.approx(1.0 - alpha, abs=0.01)
    assert covered >= 1.0 - alpha - 0.005  # finite-sample guarantee
    assert covered == pytest.approx(result.is_confident(fresh).mean(), abs=1e-12)


def test_gate_refuses_in_sample_self_certification(tmp_path):
    """In-sample coverage is 1.0 by construction; the gate scores a disjoint split."""
    tol, alpha = 1.0, 0.05
    calibration = np.full(19, 0.1)  # every score == q_hat -> 100% in-sample
    fresh = np.r_[np.zeros(95), np.full(5, 0.5)]  # 95 of 100 lie inside q_hat

    # No bin map: the exact s <= q_hat route, so the boundary is unambiguous.
    result = conformal_export(calibration, tmp_path, alpha=alpha, tol=tol, with_bin_map=False)
    payload = json.loads((tmp_path / CONFORMAL_FILE).read_text())

    assert result.coverage == 1.0
    assert payload["coverage"] == 1.0  # the number a self-certifying gate would report
    assert conformal_coverage(payload, calibration) == 1.0  # self-certification route
    assert conformal_coverage(payload, fresh) == 0.95  # honest disjoint route
    assert conformal_coverage(payload, fresh) < conformal_coverage(payload, calibration)

    # A forged coverage field is inert: the gate recomputes it from fresh data.
    for forged in (0.0, 1.0, float("nan")):
        assert conformal_coverage(dict(payload, coverage=forged), fresh) == 0.95


def test_json_round_trip_preserves_decisions_and_carries_no_raw_scores(tmp_path):
    errors = np.abs(np.random.default_rng(5).normal(0.0, 1.0, 2000))
    result = fit_conformal(errors, alpha=0.1, tol=0.25, with_bin_map=True)
    text = result.to_json()

    assert text.endswith("\n") and text == result.to_json()  # deterministic bytes
    assert text == json.dumps(result.as_dict(), indent=2, sort_keys=True) + "\n"
    restored = ConformalResult.from_json(text)
    assert restored == result
    assert restored.as_dict() == json.loads(text)
    assert all(not isinstance(v, list) or len(v) == 2 for v in result.as_dict().values())

    probes = np.array([0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5])
    expected = result.is_confident(probes).tolist()
    assert apply_confidence(json.loads(text), probes).tolist() == expected
    assert restored.is_confident(probes).tolist() == expected

    # The trainer's file path writes the same bytes and reloads to the same fit.
    conformal_export(errors, tmp_path, alpha=0.1, tol=0.25)
    reread = ConformalResult.from_json((tmp_path / CONFORMAL_FILE).read_text())
    assert reread == result
    assert reread.is_confident(probes).tolist() == expected


def _good_payload() -> dict:
    errors = np.array([0.1, 0.3, 0.7, 1.2, 2.0, 0.05, 0.9, 0.4, 0.6, 0.2])
    return fit_conformal(errors, alpha=0.2, tol=0.2, with_bin_map=True).as_dict()


@pytest.mark.parametrize(
    ("mutate", "exc", "match"),
    [
        (lambda p: p.update(method="isotonic"), ValueError, "method"),
        (lambda p: p.pop("alpha"), KeyError, "alpha"),
        (lambda p: p.update(alpha=0.0), ValueError, "alpha"),
        (lambda p: p.update(alpha=float("nan")), ValueError, "alpha"),
        (lambda p: p.update(tol=0.0), ValueError, "tol"),
        (lambda p: p.update(tol=float("inf")), ValueError, "tol"),
        (lambda p: p.update(q_hat=-1.0), ValueError, "non-negative"),
        (lambda p: p.update(q_hat=float("nan")), ValueError, "non-negative"),
        (lambda p: p.update(n=0), ValueError, "n must be at least 1"),
        (lambda p: p.pop("values"), ValueError, "together"),
        (lambda p: p.update(bins=[0.5, 0.5]), ValueError, "strictly increasing"),
        (lambda p: p.update(bins=[0.0, 1.5]), ValueError, "strictly increasing"),
        (lambda p: p.update(values=[0.0, 2.0]), ValueError, "monotone"),
        (lambda p: p.update(values=[1.0, 0.0]), ValueError, "monotone"),
    ],
)
def test_corrupt_export_is_loud(mutate, exc, match):
    payload = _good_payload()
    mutate(payload)
    with pytest.raises(exc, match=match):
        ConformalResult.from_dict(payload)
    with pytest.raises(exc, match=match):
        conformal_coverage(payload, np.array([0.1, 0.3]))


def test_malformed_json_is_loud():
    for text in ("", "{not-json", '{"method": "split-conformal"'):
        with pytest.raises(json.JSONDecodeError):
            ConformalResult.from_json(text)


def test_corrupt_fit_input_is_loud(tmp_path):
    good = np.array([0.1, 0.2, 0.4, 0.8, 1.6])
    with pytest.raises(ValueError, match="non-empty"):
        conformal_export(np.array([]), tmp_path)
    for bad in (np.array([np.nan]), np.array([1.0, np.inf]), np.array([-np.inf])):
        with pytest.raises(ValueError, match="NaN or infinity"):
            conformal_export(bad, tmp_path)
    with pytest.raises(ValueError, match="tol"):
        conformal_export(good, tmp_path, tol=0.0)
    with pytest.raises(ValueError, match="alpha"):
        conformal_export(good, tmp_path, alpha=1.0)
    with pytest.raises(ValueError, match="too small for n"):
        conformal_export(good, tmp_path, alpha=0.01)
