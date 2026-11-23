"""Trainer-facing split-conformal export: shape, determinism, coverage, refusals."""

import json

import numpy as np
import pytest

from potok_ai.conf_integration import CONFORMAL_FILE, conformal_coverage, conformal_export
from potok_ai.conformal import ConformalResult


def test_export_shape_is_deterministic_and_round_trips(tmp_path):
    errors = np.abs(np.random.default_rng(4).normal(0, 0.5, 2000))
    out = tmp_path / "run"
    result = conformal_export(errors, out, alpha=0.1, tol=0.2)

    path = out / CONFORMAL_FILE
    text = path.read_text()
    payload = json.loads(text)
    assert set(payload) == {
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
    assert payload["method"] == "split-conformal"
    assert payload["alpha"] == pytest.approx(0.1)
    assert payload["tol"] == pytest.approx(0.2)
    assert payload["n"] == errors.size
    assert payload["half_width"] == pytest.approx(payload["q_hat"] * payload["tol"])
    assert len(payload["bins"]) == len(payload["values"]) == 2
    assert payload["bins"][0] < payload["bins"][1] <= 1.0
    assert payload["values"] == [0.0, 1.0]

    conformal_export(errors, out, alpha=0.1, tol=0.2)
    assert path.read_text() == text  # deterministic bytes
    assert ConformalResult.from_json(text) == result


def test_export_without_bin_map_omits_the_map(tmp_path):
    errors = np.abs(np.random.default_rng(1).normal(0, 1, 100))
    result = conformal_export(errors, tmp_path, with_bin_map=False)
    payload = json.loads((tmp_path / CONFORMAL_FILE).read_text())
    assert result.bins is None and result.values is None
    assert "bins" not in payload and "values" not in payload


def test_exported_map_covers_a_disjoint_split(tmp_path):
    alpha, tol = 0.05, 0.3
    rng = np.random.default_rng(2026)
    calibration = rng.normal(0, 1, 30_000)
    fresh = rng.normal(0, 1, 30_000)
    result = conformal_export(calibration, tmp_path, alpha=alpha, tol=tol)
    payload = json.loads((tmp_path / CONFORMAL_FILE).read_text())

    covered = conformal_coverage(payload, fresh)
    assert covered == pytest.approx(1.0 - alpha, abs=0.01)
    assert covered >= 1.0 - alpha - 0.005
    assert conformal_coverage(payload, calibration) == pytest.approx(
        result.as_dict()["coverage"], abs=1e-3
    )


def test_exported_bin_map_matches_the_exact_indicator(tmp_path):
    errors = np.abs(np.random.default_rng(9).normal(0, 1, 5000))
    mapped = conformal_export(errors, tmp_path / "map", alpha=0.1, tol=0.4)
    exact = conformal_export(errors, tmp_path / "exact", alpha=0.1, tol=0.4, with_bin_map=False)
    probes = np.linspace(0.0, 4.0, 257)
    map_payload = json.loads((tmp_path / "map" / CONFORMAL_FILE).read_text())
    exact_payload = json.loads((tmp_path / "exact" / CONFORMAL_FILE).read_text())

    expected = mapped.is_confident(probes).mean()
    assert conformal_coverage(map_payload, probes) == pytest.approx(expected)
    assert conformal_coverage(exact_payload, probes) == pytest.approx(expected)
    assert expected == pytest.approx(exact.is_confident(probes).mean())


def test_export_and_coverage_reject_bad_inputs(tmp_path):
    errors = np.linspace(0.0, 1.0, 25)
    with pytest.raises(ValueError, match="non-empty"):
        conformal_export(np.array([]), tmp_path)

    conformal_export(errors, tmp_path)
    payload = json.loads((tmp_path / CONFORMAL_FILE).read_text())
    with pytest.raises(ValueError, match="method"):
        conformal_coverage({"method": "temperature", "temperature": 1.0}, errors)
    with pytest.raises(ValueError, match="non-empty"):
        conformal_coverage(payload, np.array([]))
    with pytest.raises(ValueError, match="NaN or infinity"):
        conformal_coverage(payload, np.array([np.nan]))
    with pytest.raises(ValueError, match="non-negative"):
        conformal_coverage(dict(payload, q_hat=-1.0), errors)


def test_export_and_coverage_refuse_non_finite_values(tmp_path):
    out = tmp_path / "run"
    with pytest.raises(ValueError, match="NaN or infinity"):
        conformal_export(np.array([0.1, np.nan, 0.4]), out)
    assert not (out / CONFORMAL_FILE).exists()

    errors = np.linspace(0.0, 1.0, 25)
    conformal_export(errors, out)
    payload = json.loads((out / CONFORMAL_FILE).read_text())
    assert all(np.isfinite(v) for v in payload.values() if isinstance(v, (int, float)))
    with pytest.raises(ValueError, match="NaN or infinity"):
        conformal_coverage(dict(payload, bins=[float("nan"), 1.0]), errors)
    with pytest.raises(ValueError, match="NaN or infinity"):
        conformal_coverage(payload, np.array([0.2, np.inf]))
