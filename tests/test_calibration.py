"""Held-out logit calibration, isotonic fallback and the spread regularizer."""

import json

import numpy as np
import pytest
import torch

from potok_ai.calibration import (
    MAX_SPREAD_WEIGHT,
    apply_calibration,
    confidence_spread_loss,
    confidence_target,
    conformal_calibrate,
    correctness,
    expected_calibration_error,
    isotonic_calibration,
    sigmoid,
)


def _logit(probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def test_sigmoid_round_trip():
    values = np.array([-4.0, 0.0, 3.0])
    assert sigmoid(values) == pytest.approx(np.array([0.01798621, 0.5, 0.95257413]))


def test_ece_perfect_and_biased():
    confidence = np.array([1.0, 1.0, 0.0, 0.0])
    correct = np.array([1.0, 1.0, 0.0, 0.0])
    assert expected_calibration_error(confidence, correct) == pytest.approx(0.0)
    biased = expected_calibration_error(np.array([0.95, 0.95, 0.05, 0.05]), correct)
    assert biased == pytest.approx(0.05)


def test_correctness_uses_the_error_tolerance():
    labels = correctness(np.array([0.0, 0.1, 0.11, 0.5]), tol=0.1)
    assert labels.tolist() == [1.0, 1.0, 0.0, 0.0]


def test_isotonic_maps_a_constant_head_to_its_accuracy():
    confidence = np.full(20, 0.5)
    correct = np.array([1.0] * 8 + [0.0] * 12)
    fitted = isotonic_calibration(confidence, correct)
    assert fitted.min() == pytest.approx(0.4) and fitted.max() == pytest.approx(0.4)


def test_temperature_meets_target_on_informative_logits():
    rng = np.random.default_rng(0)
    probability = rng.uniform(0.05, 0.95, 600)
    correct = rng.random(600) < probability
    logits = _logit(np.clip(probability + rng.normal(0, 0.02, 600), 0, 1))
    error = np.where(correct, 0.01, 0.5)

    result = conformal_calibrate(logits, error, tol=0.1, target_ece=0.05, max_loops=3)
    assert result.ece <= 0.05
    assert result.ece <= result.raw_ece + 1e-12
    assert result.method in {"temperature", "isotonic"}
    assert 1 <= result.loops <= 3
    assert 0.0 <= result.threshold <= 1.0
    assert result.n == 600
    assert result.temperature_ece <= result.raw_ece + 1e-12


def test_constant_saturated_logits_fall_back_to_isotonic():
    # Every sigmoid is exactly 0.5, so no temperature can move it. This is the
    # failure the constant-0.95 head used to cause; isotonic must still work.
    logits = np.zeros(60)
    error = np.array([0.01] * 48 + [0.5] * 12)
    result = conformal_calibrate(logits, error, tol=0.1, target_ece=0.05, max_loops=3)
    assert result.raw_ece == pytest.approx(0.3)
    assert result.method == "isotonic"
    assert result.ece <= 0.05
    assert result.loops <= 3


def test_isotonic_calibration_json_round_trip(tmp_path):
    """The exported map must reproduce the in-memory isotonic fit exactly.

    No logit array is persisted under `ai/out/`, so this uses a deterministic
    synthetic held-out distribution -- not a production measurement. Four logit
    levels with mis-scaled empirical correctness force the isotonic path and a
    multi-bin step map.
    """
    logits = np.repeat(np.array([-20.0, -4.0, 4.0, 20.0]), 40)
    correct = np.concatenate(
        [np.r_[np.ones(count), np.zeros(40 - count)] for count in (12, 20, 28, 36)]
    )
    error = np.where(correct > 0, 0.01, 0.5)

    result = conformal_calibrate(logits, error, tol=0.1, target_ece=0.05, max_loops=3)
    assert result.method == "isotonic"
    assert result.bins is not None and result.values is not None

    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(result.as_dict(), indent=2) + "\n")
    payload = json.loads(path.read_text())

    assert payload["method"] == "isotonic"
    assert payload["threshold"] == pytest.approx(result.threshold)
    bins = np.asarray(payload["bins"])
    values = np.asarray(payload["values"])
    assert bins.size == values.size == 4
    assert np.all(np.diff(bins) > 0) and bins[0] > 0.0 and bins[-1] <= 1.0
    assert np.all((values >= 0.0) & (values <= 1.0))

    expected = isotonic_calibration(sigmoid(logits), correct)
    assert apply_calibration(payload, logits) == pytest.approx(expected, abs=1e-12)

    # The edge reader also accepts n+1 edges (leading lower edge), so confirm
    # that shape round-trips to the same step map.
    extended = dict(payload, bins=[0.0, *payload["bins"]])
    assert apply_calibration(extended, logits) == pytest.approx(expected, abs=1e-12)


def test_temperature_calibration_json_stays_backward_compatible(tmp_path):
    """Temperature files keep their old keys and their sigmoid(z/T) behaviour."""
    rng = np.random.default_rng(7)
    probability = rng.uniform(0.05, 0.95, 600)
    correct = (rng.random(600) < probability).astype(float)
    logits = _logit(probability)
    error = np.where(correct > 0, 0.01, 0.5)

    result = conformal_calibrate(logits, error, tol=0.1, target_ece=0.05, max_loops=3)
    assert result.method == "temperature"

    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(result.as_dict(), indent=2) + "\n")
    payload = json.loads(path.read_text())

    assert "bins" not in payload and "values" not in payload
    assert payload["temperature"] == pytest.approx(result.temperature)
    assert apply_calibration(payload, logits) == pytest.approx(
        sigmoid(logits / result.temperature), abs=1e-12
    )


def test_calibration_rejects_bad_inputs():
    with pytest.raises(ValueError, match="same length"):
        conformal_calibrate(np.array([0.1, 0.2]), np.array([0.0]), max_loops=3)
    with pytest.raises(ValueError, match="max_loops"):
        conformal_calibrate(np.array([0.1, 0.2]), np.array([0.0, 0.0]), max_loops=0)


def test_confidence_target_tracks_uncertainty():
    assert confidence_target(torch.tensor(0.0)).item() == pytest.approx(1.0)
    assert confidence_target(torch.tensor(0.1)).item() == pytest.approx(np.exp(-1))
    # A single large error no longer collapses to ~0: the scale follows the data.
    assert confidence_target(torch.tensor(1.0, requires_grad=True)).item() == pytest.approx(
        np.exp(-1)
    )


def test_confidence_target_spreads_when_errors_exceed_tolerance():
    error = torch.tensor([0.01, 0.1, 1.0, 2.0])
    target = confidence_target(error)
    assert target.min() > 0.05
    assert target.max() > 0.7
    assert (target.max() - target.min()) > 0.4
    # An explicit scale keeps the original absolute-error semantics.
    fixed = confidence_target(error, scale=0.1)
    assert fixed[0].item() == pytest.approx(np.exp(-0.1))
    assert fixed[1].item() == pytest.approx(np.exp(-1))


def test_spread_reward_is_small_and_uncertainty_gated():
    confidence = torch.tensor([0.9, 0.2, 0.7])
    uncertain = torch.tensor([0.5, 0.5, 0.5])
    certain = torch.zeros_like(uncertain)
    reward = confidence_spread_loss(confidence, uncertain)
    assert reward < 0
    assert abs(reward) <= MAX_SPREAD_WEIGHT * confidence.var(unbiased=False).item() + 1e-9
    assert confidence_spread_loss(confidence, certain) == pytest.approx(0.0)
    assert confidence_spread_loss(confidence, uncertain, weight=0.0) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="spread weight"):
        confidence_spread_loss(confidence, uncertain, weight=0.1)
