# SPDX-License-Identifier: GPL-3.0-or-later
"""AICAL gates: split-conformal confidence landed in the staged teacher-v2 trainer."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch
from test_train_teacher_v2 import _cfg, _episodes, _use_episodes

from potok_ai import train_teacher_v2 as ttv
from potok_ai.conf_integration import (
    CONFORMAL_FILE,
    DISJOINT_ECE_SOURCE,
    IN_SAMPLE_ECE_SOURCE,
    frozen_step_map,
    split_confidence_score,
)
from potok_ai.conformal import ConformalResult, apply_confidence
from potok_ai.teacher_gates import HONEST_ECE_SOURCES


def test_staged_trainer_reports_disjoint_confidence_and_export(tmp_path, monkeypatch) -> None:
    _use_episodes(monkeypatch, _episodes())
    cfg = _cfg(tmp_path / "run", stage="sft")
    metrics = ttv.run(cfg)
    report = json.loads((cfg.out / "metrics.json").read_text())

    assert metrics.ece_source == DISJOINT_ECE_SOURCE == "held-out"
    assert metrics.ece_source in HONEST_ECE_SOURCES
    assert report["ece_source"] == metrics.ece_source
    assert metrics.ece == pytest.approx(metrics.ece_held_out)
    assert report["ece"] == pytest.approx(report["ece_held_out"])
    assert all(
        math.isfinite(value)
        for value in (metrics.ece, metrics.ece_held_out, metrics.conformal_coverage)
    )
    assert 0.0 <= metrics.ece <= 1.0
    assert 0.0 <= metrics.conformal_coverage <= 1.0

    # Stage-local ECE remains a diagnostic and says so explicitly.
    assert report["stages"]
    assert all(stage["ece_source"] == IN_SAMPLE_ECE_SOURCE for stage in report["stages"].values())

    payload = json.loads((cfg.out / CONFORMAL_FILE).read_text())
    assert payload["method"] == "split-conformal"
    _fit, inner, _held = ttv.partition_episodes(
        _episodes(), seed=cfg.seed, n_held_out=cfg.n_inner_held_out
    )
    n_inner = sum(len(batch[0]) for batch in ttv._window_batches(inner, cfg))
    assert payload["n"] == n_inner // 2  # fit half, never the scored rows
    bins, values = frozen_step_map(payload)
    assert len(bins) == 2 and values == [0.0, 1.0]


def test_disjoint_score_is_deterministic_and_seed_keyed(tmp_path) -> None:
    # A saturated head whose seed-1 fit half is entirely correct: scoring any
    # other half must move the honest ECE, so a different seed is visible.
    n = 40
    logits = np.full(n, 10.0)
    errors = np.full(n, 0.5)
    errors[np.random.default_rng(1).permutation(n)[: n // 2]] = 0.001
    scores, texts = [], []
    for index, seed in enumerate((1, 1, 2)):
        out = tmp_path / f"run{index}"
        score = split_confidence_score(logits, errors, out, seed=seed, tol=0.1)
        scores.append(score)
        texts.append((out / CONFORMAL_FILE).read_bytes())
    assert scores[0] == scores[1]
    assert texts[0] == texts[1]
    assert (scores[0].n_fit, scores[0].n_score) == (20, 20)
    assert scores[0] != scores[2]  # a different seed draws a different split


def test_exported_step_map_is_frozen_and_edge_compatible(tmp_path) -> None:
    errors = np.abs(np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.4]))
    logits = np.zeros(errors.size)
    score = split_confidence_score(logits, errors, tmp_path / "a", seed=1, tol=0.1)
    payload = json.loads((tmp_path / "a" / CONFORMAL_FILE).read_text())
    bins, values = frozen_step_map(payload)
    q_hat, tol = payload["q_hat"], payload["tol"]

    assert bins == pytest.approx([1.0 / (1.0 + q_hat), 1.0])
    assert values == [0.0, 1.0]
    assert payload["half_width"] == pytest.approx(q_hat * tol)
    assert payload["n"] == score.n_fit

    # The edge (step-map) route and the exact conformal indicator agree,
    # including at the inclusive boundary s == q_hat.
    probes = np.array([0.0, 0.5 * q_hat * tol, q_hat * tol, 1.01 * q_hat * tol, 10.0])
    exact = ConformalResult.from_dict(payload)
    assert apply_confidence(payload, probes).tolist() == exact.is_confident(probes).tolist()

    split_confidence_score(logits, errors, tmp_path / "b", seed=1, tol=0.1)
    assert (tmp_path / "a" / CONFORMAL_FILE).read_bytes() == (
        tmp_path / "b" / CONFORMAL_FILE
    ).read_bytes()


def test_frozen_step_map_refuses_missing_or_malformed_maps() -> None:
    base = {
        "method": "split-conformal",
        "alpha": 0.5,
        "tol": 0.1,
        "q_hat": 1.0,
        "n": 10,
        "coverage": 1.0,
        "bins": [0.5, 1.0],
        "values": [0.0, 1.0],
    }
    assert frozen_step_map(base) == ([0.5, 1.0], [0.0, 1.0])
    with pytest.raises(ValueError, match="missing the frozen step map"):
        frozen_step_map({key: value for key, value in base.items() if key not in ("bins", "values")})
    with pytest.raises(ValueError, match="NaN or infinity"):
        frozen_step_map(dict(base, bins=[float("nan"), 1.0]))
    with pytest.raises(ValueError, match="frozen step map needs"):
        frozen_step_map(dict(base, values=[0.0, 0.5]))


def test_confidence_hook_refuses_non_finite_predictions(tmp_path, monkeypatch) -> None:
    cfg = _cfg(tmp_path / "nan", stage="sft")
    real_predict = ttv._predict

    def broken(model, window, cond, target, *, cfg, student):
        command, logit, error = real_predict(model, window, cond, target, cfg=cfg, student=student)
        return command, logit, torch.full_like(error, float("nan"))

    monkeypatch.setattr(ttv, "_predict", broken)
    with pytest.raises(ValueError, match="finite"):
        ttv._confidence_hook(
            ttv._build_teacher(cfg, torch.device("cpu")),
            _episodes()[:2],
            cfg,
            torch.device("cpu"),
            student=False,
        )
    assert not (cfg.out / CONFORMAL_FILE).exists()


def test_distill_pipeline_and_cli_write_the_export(tmp_path, monkeypatch) -> None:
    _use_episodes(monkeypatch, _episodes())
    cfg = _cfg(tmp_path / "all", stage="all")
    metrics = ttv.run(cfg)
    assert metrics.ece_source == DISJOINT_ECE_SOURCE
    assert (cfg.out / CONFORMAL_FILE).is_file()
    assert (cfg.out / "student.pt").is_file()

    cli_out = tmp_path / "cli"
    ttv.main(
        [
            "--data",
            str(tmp_path / "data"),
            "--out",
            str(cli_out),
            "--stage",
            "sft",
            "--seed",
            "1",
        ]
    )
    report = json.loads((cli_out / "metrics.json").read_text())
    assert report["ece_source"] == DISJOINT_ECE_SOURCE
    assert (cli_out / CONFORMAL_FILE).is_file()
