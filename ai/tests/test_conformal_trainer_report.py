"""Focused gates: in-sample ECE is labelled, split-conformal scores a disjoint half."""

import json

import numpy as np
import pytest

from potok_ai.calibration import conformal_calibrate
from potok_ai.conf_integration import split_confidence_score
from potok_ai.engine import TrainConfig, build_metrics

SEED = 0
TOL = 0.1


def _config(out, *, calibrate: bool, seed: int = SEED):
    return TrainConfig(
        model="lrs", stage="train", data=out.parent / "validation.bin", out=out,
        calibrate=calibrate, seed=seed, calib_tol=TOL,
    )


def _fixture(n: int = 8):
    """Saturated confidence; the seeded fit half is correct, the score half is not."""
    order = np.random.default_rng(SEED).permutation(n)
    fit = order[: n // 2]
    z = np.full(n, 10.0)
    err = np.full(n, 0.5)
    err[fit] = 0.001
    return z, err


def test_in_sample_legacy_field_is_labelled_and_disjoint_score_is_honest(tmp_path):
    z, err = _fixture()
    cfg = _config(tmp_path / "run", calibrate=True)
    metrics = build_metrics(
        "lrs", cfg, {"a": 0.5, "b": 0.6}, {"b"}, 10, 20, 30, z, err
    )

    # Legacy field keeps its old value; its label no longer claims held-out.
    assert metrics.ece_source == "in-sample"
    assert metrics.ece == pytest.approx(conformal_calibrate(z, err, tol=TOL).ece)
    # The score half is confident but always wrong: ECE ~ 1, not the fit's 0.
    assert metrics.ece_held_out == pytest.approx(1.0, abs=1e-6)
    assert 0.0 <= metrics.conformal_coverage <= 1.0

    calibration = json.loads((cfg.out / "calibration.json").read_text())
    assert calibration["ece_source"] == "in-sample"
    assert calibration["ece"] == pytest.approx(metrics.ece)
    assert calibration["n"] == z.size

    conformal = json.loads((cfg.out / "conformal.json").read_text())
    assert conformal["method"] == "split-conformal"
    assert conformal["n"] == z.size // 2  # fit half, not the scored set
    assert conformal["n"] < calibration["n"]


def test_disjoint_report_is_deterministic_and_seed_keyed(tmp_path):
    z, err = _fixture()
    runs = []
    for seed in (SEED, SEED, 3):
        out = tmp_path / f"run{seed}-{len(runs)}"
        cfg = _config(out, calibrate=True, seed=seed)
        metrics = build_metrics("lrs", cfg, {"a": 0.5, "b": 0.6}, {"b"}, 10, 20, 30, z, err)
        runs.append((metrics.as_dict(), (out / "conformal.json").read_bytes()))

    assert runs[0] == runs[1]  # same seed -> same split, report and export
    assert runs[0][1] != runs[2][1]  # a different seed changes the disjoint split


def test_without_calibrate_there_is_no_conformal_claim(tmp_path):
    z, err = _fixture()
    cfg = _config(tmp_path / "plain", calibrate=False)
    metrics = build_metrics("lrs", cfg, {"a": 0.5, "b": 0.6}, {"b"}, 10, 20, 30, z, err)
    assert metrics.ece_source == "in-sample"
    assert metrics.ece_held_out == 0.0 and metrics.conformal_coverage == 0.0
    assert not (cfg.out / "conformal.json").exists()
    assert not (cfg.out / "calibration.json").exists()


def test_tiny_fit_split_records_the_effective_alpha(tmp_path):
    score = split_confidence_score(
        np.array([0.0, 1.0]), np.array([0.01, 0.2]), tmp_path / "tiny", seed=1
    )
    payload = json.loads((tmp_path / "tiny" / "conformal.json").read_text())
    assert (score.n_fit, score.n_score) == (1, 1)
    assert score.alpha == pytest.approx(0.5)  # smallest valid conformal level for n=1
    assert payload["alpha"] == pytest.approx(score.alpha)


def test_disjoint_score_rejects_unusable_input(tmp_path):
    with pytest.raises(ValueError, match="at least two"):
        split_confidence_score(np.array([0.0]), np.array([0.1]), tmp_path, seed=1)
    with pytest.raises(ValueError, match="finite"):
        split_confidence_score(
            np.array([0.0, np.nan]), np.array([0.1, 0.2]), tmp_path, seed=1
        )
    with pytest.raises(ValueError, match="finite"):
        split_confidence_score(
            np.array([0.0, 1.0]), np.array([0.1, np.inf]), tmp_path, seed=1
        )


def test_engine_cli_writes_the_exported_payload(tmp_path):
    from test_engine import _argv, _data

    from potok_ai import engine

    data, out = _data(tmp_path), tmp_path / "cli"
    engine.main(_argv("loopwm-lite", data, out, "--calibrate"))
    report = json.loads((out / "metrics.json").read_text())
    assert report["ece_source"] == "in-sample"
    assert 0.0 <= report["ece_held_out"] <= 1.0
    payload = json.loads((out / "conformal.json").read_text())
    assert payload["method"] == "split-conformal"
    assert payload["n"] >= 1
