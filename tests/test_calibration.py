"""Conformal-style confidence calibration on toy held-out distributions."""

import numpy as np
import pytest

from potok_ai.calibration import (
    conformal_calibrate,
    correctness,
    expected_calibration_error,
)


def test_ece_perfect_and_biased():
    confidence = np.array([1.0, 1.0, 0.0, 0.0])
    correct = np.array([1.0, 1.0, 0.0, 0.0])
    assert expected_calibration_error(confidence, correct) == pytest.approx(0.0)
    biased = expected_calibration_error(np.array([0.95, 0.95, 0.05, 0.05]), correct)
    assert biased == pytest.approx(0.05)


def test_correctness_uses_the_error_tolerance():
    labels = correctness(np.array([0.0, 0.1, 0.11, 0.5]), tol=0.1)
    assert labels.tolist() == [1.0, 1.0, 0.0, 0.0]


def test_conformal_toy_meets_the_ece_target():
    rng = np.random.default_rng(0)
    probability = rng.uniform(0.05, 0.95, 600)
    correct = rng.random(600) < probability
    confidence = np.clip(probability + rng.normal(0, 0.02, 600), 0, 1)
    error = np.where(correct, 0.01, 0.5)

    result = conformal_calibrate(confidence, error, tol=0.1, target_ece=0.05, max_loops=3)
    assert result.ece <= 0.05
    assert result.ece <= result.raw_ece + 1e-12
    assert 1 <= result.loops <= 3
    assert 0.0 <= result.threshold <= 1.0
    assert result.n == 600
    assert result.target_ece == 0.05
    assert result.coverage == pytest.approx(float(correct.mean()))


def test_hard_loop_cap_when_target_is_unreachable():
    # A constant 0.5 confidence cannot be moved by temperature scaling, so the
    # 0.05 ECE target is never met and the search must stop at its hard cap.
    confidence = np.full(64, 0.5)
    error = np.zeros(64)
    result = conformal_calibrate(confidence, error, tol=0.1, target_ece=0.05, max_loops=3)
    assert result.loops == 3
    assert result.ece > 0.05


def test_calibration_rejects_bad_inputs():
    with pytest.raises(ValueError, match="same length"):
        conformal_calibrate(np.array([0.1, 0.2]), np.array([0.0]), max_loops=3)
    with pytest.raises(ValueError, match="max_loops"):
        conformal_calibrate(np.array([0.1, 0.2]), np.array([0.0, 0.0]), max_loops=0)
