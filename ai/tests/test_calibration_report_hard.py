"""Hard gates for the calibration report artifact (GPL-3.0-or-later).

Four properties are pinned, all on synthetic data with fixed seeds so no
production artifact under ``out/`` is read:

* byte-determinism of the calibration JSON (direct and engine-written),
* ECE / coverage gates, including recomputation through the exported payload,
* the conformal, temperature and isotonic export routes,
* LOUD rejection of corrupt input at every public entry point.
"""

import json
import math

import numpy as np
import pytest

from potok_ai.calibration import (
    DEFAULT_TARGET_ECE,
    DEFAULT_TOL,
    apply_calibration,
    conformal_calibrate,
    correctness,
    expected_calibration_error,
    isotonic_calibration,
    sigmoid,
)
from potok_ai.conf_integration import conformal_coverage, conformal_export
from potok_ai.conformal import (
    METHOD as CONFORMAL_METHOD,
)
from potok_ai.conformal import (
    ConformalResult,
    apply_confidence,
    conformal_quantile,
    conformal_scores,
    fit_conformal,
)
from potok_ai.engine import TrainConfig, build_metrics, config_hash

REPORT_FIELDS = {
    "threshold", "ece", "raw_ece", "temperature_ece", "coverage", "temperature",
    "method", "loops", "n", "target_ece",
}


def _logit(probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _isotonic_fixture():
    """Four logit levels with mis-scaled correctness force a multi-bin PAVA map."""
    logits = np.repeat(np.array([-20.0, -4.0, 4.0, 20.0]), 40)
    correct = np.concatenate([
        np.r_[np.ones(count), np.zeros(40 - count)] for count in (12, 20, 28, 36)
    ])
    error = np.where(correct > 0, 0.01, 0.5)
    return logits, error, correct


def _temperature_fixture():
    rng = np.random.default_rng(7)
    probability = rng.uniform(0.05, 0.95, 600)
    correct = (rng.random(600) < probability).astype(float)
    logits = _logit(probability)
    error = np.where(correct > 0, 0.01, 0.5)
    return logits, error, correct


def test_calibration_report_json_is_byte_deterministic(tmp_path):
    """Same held-out data -> same bytes, both directly and through the engine."""
    logits, error, _ = _isotonic_fixture()
    first = conformal_calibrate(logits, error, tol=DEFAULT_TOL,
                                target_ece=DEFAULT_TARGET_ECE, max_loops=3)
    second = conformal_calibrate(logits, error, tol=DEFAULT_TOL,
                                 target_ece=DEFAULT_TARGET_ECE, max_loops=3)

    assert first.as_dict() == second.as_dict()
    encode = lambda result: json.dumps(result.as_dict(), sort_keys=True, indent=2) + "\n"
    assert encode(first) == encode(second)
    assert json.loads(encode(first)) == first.as_dict()

    # The engine adds model / seed / config_hash; the whole file must be stable.
    temperature_logits, temperature_error, _ = _temperature_fixture()
    blobs = []
    for name in ("run_a", "run_b"):
        out = tmp_path / name
        cfg = TrainConfig(model="lrs", stage="eval", data=tmp_path / "data", out=out,
                          calibrate=True, seed=42)
        metrics = build_metrics("lrs", cfg, {"a": 1.0, "b": 2.0}, {"b"},
                                100, 200, 300, temperature_logits, temperature_error)
        payload = json.loads((out / "calibration.json").read_text())
        assert REPORT_FIELDS <= payload.keys()
        assert payload["model"] == "lrs" and payload["seed"] == 42
        assert payload["config_hash"] == config_hash(cfg) == metrics.config_hash
        blobs.append((out / "calibration.json").read_bytes())

    assert blobs[0] == blobs[1]
    assert blobs[0].endswith(b"\n")


def test_calibration_report_ece_and_coverage_gates():
    """Reported gates must hold and survive a round-trip through the payload."""
    logits, error, correct = _isotonic_fixture()
    result = conformal_calibrate(logits, error, tol=DEFAULT_TOL,
                                 target_ece=DEFAULT_TARGET_ECE, max_loops=3)
    payload = result.as_dict()

    assert result.ece <= DEFAULT_TARGET_ECE
    assert result.ece <= result.raw_ece + 1e-12
    assert result.temperature_ece <= result.raw_ece + 1e-12
    assert result.coverage == pytest.approx(correctness(error, DEFAULT_TOL).mean())
    assert 0.0 <= result.coverage <= 1.0
    assert 0.0 <= result.threshold <= 1.0
    assert 1 <= result.loops <= 3
    assert result.n == logits.size
    assert all(math.isfinite(payload[key]) for key in REPORT_FIELDS - {"method"})

    applied = apply_calibration(payload, logits)
    assert applied.shape == logits.shape
    assert expected_calibration_error(applied, correct) == pytest.approx(result.ece, abs=1e-12)
    assert expected_calibration_error(applied, correct) <= DEFAULT_TARGET_ECE
    assert result.method == "isotonic"  # the saturated fixture cannot be repaired by T
    assert result.ece < 1e-9  # PAVA is exact on this discrete fixture


def test_temperature_export_report():
    """Temperature files stay lean and reproduce sigmoid(z / T) byte-for-byte."""
    logits, error, correct = _temperature_fixture()
    result = conformal_calibrate(logits, error, tol=DEFAULT_TOL,
                                 target_ece=DEFAULT_TARGET_ECE, max_loops=3)
    assert result.method == "temperature"

    payload = result.as_dict()
    assert "bins" not in payload and "values" not in payload
    assert json.loads(json.dumps(payload)) == payload

    applied = apply_calibration(payload, logits)
    assert applied == pytest.approx(sigmoid(logits / result.temperature), abs=1e-12)
    assert expected_calibration_error(applied, correct) == pytest.approx(result.ece, abs=1e-12)
    assert result.ece <= DEFAULT_TARGET_ECE


def test_isotonic_export_report():
    """The exported step map must reproduce the in-memory PAVA fit exactly."""
    logits, error, correct = _isotonic_fixture()
    result = conformal_calibrate(logits, error, tol=DEFAULT_TOL,
                                 target_ece=DEFAULT_TARGET_ECE, max_loops=3)
    assert result.method == "isotonic"
    assert result.bins is not None and result.values is not None

    payload = json.loads(json.dumps(result.as_dict()))
    bins = np.asarray(payload["bins"], dtype=float)
    values = np.asarray(payload["values"], dtype=float)
    assert bins.size == values.size == 4
    assert bins[0] > 0.0 and bins[-1] <= 1.0 and np.all(np.diff(bins) > 0.0)
    assert np.all((values >= 0.0) & (values <= 1.0)) and np.all(np.diff(values) >= 0.0)

    expected = isotonic_calibration(sigmoid(logits), correct)
    assert apply_calibration(payload, logits) == pytest.approx(expected, abs=1e-12)
    # The edge reader also accepts an n+1 edge array with a leading lower edge.
    extended = dict(payload, bins=[0.0, *payload["bins"]])
    assert apply_calibration(extended, logits) == pytest.approx(expected, abs=1e-12)


def test_conformal_export_report(tmp_path):
    """conformal.json is deterministic, round-trips, and covers a disjoint split."""
    rng = np.random.default_rng(11)
    calibration = rng.normal(0.0, 1.0, 20_000)
    held_out = rng.normal(0.0, 1.0, 20_000)
    alpha, tol = 0.1, 0.5

    result = fit_conformal(calibration, alpha=alpha, tol=tol, with_bin_map=True)
    assert result.q_hat == pytest.approx(
        conformal_quantile(conformal_scores(calibration, tol), alpha)
    )
    assert result.half_width == pytest.approx(result.q_hat * tol)
    assert result.n == calibration.size
    assert result.coverage == pytest.approx(1.0 - alpha, abs=0.01)

    text = result.to_json()
    assert text.endswith("\n") and text == result.to_json()
    assert ConformalResult.from_json(text) == result
    assert json.loads(text) == result.as_dict()
    assert result.bins == pytest.approx([1 / (1 + result.q_hat), 1.0])
    assert result.values == (0.0, 1.0)

    exported = conformal_export(calibration, tmp_path, alpha=alpha, tol=tol,
                                with_bin_map=True)
    assert exported == result
    payload = json.loads((tmp_path / "conformal.json").read_text())
    assert payload["method"] == CONFORMAL_METHOD
    assert payload["bins"] == list(result.bins)
    assert apply_confidence(payload, held_out).tolist() == exported.is_confident(held_out).tolist()

    covered = conformal_coverage(payload, held_out)
    assert covered == pytest.approx(1.0 - alpha, abs=0.01)
    assert covered >= 1.0 - alpha - 0.005  # finite-sample conformal guarantee


def test_build_metrics_refuses_nan_held_out_data(tmp_path):
    """A diverged held-out batch must be refused before calibration.json is written."""
    out = tmp_path / "out"
    cfg = TrainConfig(model="lrs", stage="eval", data=tmp_path / "data", out=out,
                      calibrate=True, seed=42)
    for logits, error in (
        (np.array([0.0, np.nan]), np.array([0.01, 0.5])),
        (np.array([0.0, 1.0]), np.array([0.01, np.nan])),
    ):
        with pytest.raises(ValueError, match="NaN or infinity"):
            build_metrics("lrs", cfg, {"a": 1.0}, {"a"}, 100, 200, 300, logits, error)
        assert not (out / "calibration.json").exists()


def test_corrupt_calibration_input_is_loud():
    """No silent fallback: malformed data, configs and payloads all raise."""
    with pytest.raises(ValueError, match="equal, non-empty"):
        expected_calibration_error(np.array([0.2, 0.8]), np.array([1.0]))
    with pytest.raises(ValueError, match="non-empty"):
        expected_calibration_error(np.array([]), np.array([]))
    with pytest.raises(ValueError, match="confidences must lie"):
        expected_calibration_error(np.array([1.2, 0.4]), np.array([1.0, 0.0]))

    with pytest.raises(ValueError, match="same length"):
        conformal_calibrate(np.array([0.1, 0.2]), np.array([0.0]), max_loops=3)
    with pytest.raises(ValueError, match="max_loops"):
        conformal_calibrate(np.array([0.1, 0.2]), np.array([0.0, 0.0]), max_loops=0)

    logits = np.array([-1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="unknown calibration method"):
        apply_calibration({"method": "bogus"}, logits)
    with pytest.raises(ValueError, match="isotonic payload needs"):
        apply_calibration({"method": "isotonic", "bins": [], "values": []}, logits)


def test_corrupt_conformal_report_is_loud():
    """The conformal reader validates every field, including a wrong-method file."""
    with pytest.raises(ValueError, match="non-empty"):
        fit_conformal(np.array([]))
    for bad in (np.array([np.nan]), np.array([1.0, np.inf]), np.array([-np.inf])):
        with pytest.raises(ValueError, match="NaN or infinity"):
            fit_conformal(bad)
    with pytest.raises(ValueError, match="tol"):
        fit_conformal(np.array([0.1, 0.2]), tol=0.0)
    with pytest.raises(ValueError, match="alpha"):
        fit_conformal(np.array([0.1, 0.2]), alpha=1.0)

    good = fit_conformal(np.array([0.1, 0.3, 0.7]), alpha=0.5,
                         with_bin_map=True).as_dict()
    with pytest.raises(ValueError, match="expected method"):
        ConformalResult.from_dict(dict(good, method="temperature"))
    with pytest.raises(ValueError, match="non-negative"):
        ConformalResult.from_dict(dict(good, q_hat=-1.0))
    with pytest.raises(ValueError, match="alpha"):
        ConformalResult.from_dict(dict(good, alpha=0.0))
    with pytest.raises(ValueError, match="strictly increasing"):
        ConformalResult.from_dict(dict(good, bins=[0.9, 0.9], values=[0.0, 1.0]))
    with pytest.raises(KeyError):
        ConformalResult.from_dict({"method": CONFORMAL_METHOD})

    # A calibration.py report has different semantics and must never be scored
    # as a conformal one.
    logits, error, _ = _temperature_fixture()
    temperature_payload = conformal_calibrate(
        logits, error, max_loops=3
    ).as_dict()
    with pytest.raises(ValueError, match="expected method"):
        conformal_coverage(temperature_payload, error)
