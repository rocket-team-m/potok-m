# SPDX-License-Identifier: GPL-3.0-or-later
"""ENGINE01 gates for the staged teacher-v2 CLI (CPU only, no full training)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from potok_ai import train_teacher_v2 as ttv
from potok_ai.data import make_smart_episode
from potok_ai.loopwm import LoopWMConfig


def _episodes(n: int = 4, frames: int = 24) -> list:
    return [make_smart_episode(seed=100 + i, frames=frames, scenario_id=i) for i in range(n)]


def _cfg(tmp_path: Path, **overrides) -> ttv.TeacherV2Config:
    kwargs = {
        "data": tmp_path / "data",
        "out": tmp_path / "out",
        "seed": 1,
        "epochs": 1,
        "batch": 2,
        "window": 12,
        "limit": 6,
        "width": 16,
        "heads": 2,
        "max_iters": 2,
    }
    kwargs.update(overrides)
    return ttv.TeacherV2Config(**kwargs)


def _use_episodes(monkeypatch, episodes: list) -> None:
    monkeypatch.setattr(ttv, "load_episodes", lambda path: episodes)


def _model(cfg: ttv.TeacherV2Config) -> torch.nn.Module:
    return ttv._build_teacher(cfg, torch.device("cpu"))


def _batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    return (
        torch.randn(2, 12, 14),
        torch.randn(2, 6),
        torch.randn(2, 2),
        [0, 1],
    )


def test_config_validation_and_seed_set(tmp_path: Path) -> None:
    base = _cfg(tmp_path)
    assert base.seed == 1
    assert len({ttv.config_hash(_cfg(tmp_path, seed=s)) for s in ttv.SEEDS}) == 3
    with pytest.raises(ValueError, match="seed"):
        _cfg(tmp_path, seed=7)
    with pytest.raises(ValueError, match="unknown stage"):
        _cfg(tmp_path, stage="nope")
    with pytest.raises(ValueError, match="window"):
        _cfg(tmp_path, window=10)
    with pytest.raises(ValueError, match="divisible"):
        _cfg(tmp_path, width=15, heads=2)
    with pytest.raises(ValueError, match="CPU"):
        _cfg(tmp_path, device="cuda")


def test_partition_disjoint_stable_and_refuses_eval() -> None:
    episodes = _episodes()
    fit, inner, held = ttv.partition_episodes(episodes, seed=1, n_held_out=2)
    assert len(held) == 2
    assert {e.scenario_id for e in inner} == set(held)
    assert not ({e.scenario_id for e in fit} & set(held))
    again = ttv.partition_episodes(episodes, seed=1, n_held_out=2)
    assert {e.scenario_id for e in again[1]} == set(held)
    contaminated = episodes + [make_smart_episode(9, frames=24, scenario_id=17)]
    with pytest.raises(ValueError, match="eval scenario"):
        ttv.partition_episodes(contaminated, seed=1)
    no_id = make_smart_episode(9, frames=24, scenario_id=5)
    no_id.scenario_id = None
    with pytest.raises(ValueError, match="scenario id"):
        ttv.partition_episodes([no_id], seed=1, n_held_out=2)


def test_default_exit_off_has_no_gate_or_entropy(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    model = _model(cfg)
    assert not hasattr(model, "exit_gate")
    result = ttv._supervised_loss(model, _batch(), cfg)
    assert result["entropy"] == 0.0
    assert not cfg.learned_exit


def test_entropy_term_flows_to_exit_gate_and_is_recorded(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg(tmp_path, stage="sft", learned_exit=True, exit_entropy_alpha=0.05)
    model = _model(cfg)
    assert hasattr(model, "exit_gate")
    result = ttv._supervised_loss(model, _batch(), cfg)
    result["loss"].backward()
    gradient = model.exit_gate.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0
    assert result["entropy"] < 0.0
    _use_episodes(monkeypatch, _episodes())
    metrics = ttv.run(cfg)
    assert metrics.entropy_enabled
    assert np.isfinite(metrics.entropy) and metrics.entropy < 0.0
    payload = json.loads((cfg.out / "metrics.json").read_text())
    assert payload["stages"]["sft"]["entropy"] == metrics.entropy
    assert payload["entropy_enabled"] is True


def test_entropy_is_reported_at_top_level_in_full_pipeline(tmp_path: Path, monkeypatch) -> None:
    _use_episodes(monkeypatch, _episodes())
    cfg = _cfg(tmp_path, stage="all", learned_exit=True)
    metrics = ttv.run(cfg)
    payload = json.loads((cfg.out / "metrics.json").read_text())
    assert metrics.entropy_enabled and metrics.entropy != 0.0
    assert payload["entropy"] == metrics.stages["dagger"]["entropy"]
    assert metrics.entropy == metrics.stages["dagger"]["entropy"]


@pytest.mark.parametrize("seed", ttv.SEEDS)
def test_three_seed_sft_gate_is_repeatable(tmp_path: Path, monkeypatch, seed: int) -> None:
    _use_episodes(monkeypatch, _episodes())
    cfg = _cfg(tmp_path, stage="sft", seed=seed)
    first = ttv.run(cfg)
    second = ttv.run(cfg)
    assert first.as_dict() == second.as_dict()
    assert first.seed == seed
    assert first.config_hash == ttv.config_hash(cfg)
    assert first.stages_completed == ("sft",)
    assert np.isfinite(first.ratio) and np.isfinite(first.ece)


def test_metrics_schema(tmp_path: Path, monkeypatch) -> None:
    _use_episodes(monkeypatch, _episodes())
    cfg = _cfg(tmp_path, stage="sft")
    metrics = ttv.run(cfg)
    payload = json.loads((cfg.out / "metrics.json").read_text())
    assert {
        "ratio",
        "ece",
        "entropy",
        "entropy_enabled",
        "reward_provenance",
        "config_hash",
        "seed",
        "module_sources",
        "stages",
        "stages_completed",
    } <= payload.keys()
    assert metrics.stage == "sft"
    assert payload["reward_provenance"] == {}


def test_privileged_stage_runs_jepa(tmp_path: Path, monkeypatch) -> None:
    _use_episodes(monkeypatch, _episodes())
    metrics = ttv.run(_cfg(tmp_path, stage="privileged"))
    assert metrics.stages_completed == ("sft", "privileged")
    assert metrics.stages["privileged"]["steps"] >= 1
    assert metrics.module_sources["jepa"] in {"jepa_v2", "local"}
    assert metrics.module_sources["privileged"] in {"privileged_teacher", "local"}


def test_local_jepa_fallback_interface() -> None:
    model = ttv._build_teacher(_cfg(Path("/tmp")), torch.device("cpu"))
    objective = ttv._LocalJEPAV2(model.encoder, 16, seed=7)
    losses = objective.loss(model.encoder, torch.randn(2, 12, 14), None)
    assert torch.isfinite(losses.total)
    losses.total.backward()
    objective.update_teacher(model.encoder)


def test_privileged_auxiliary_uses_builder(tmp_path: Path, monkeypatch) -> None:
    class FakeBuilder:
        encoder = SimpleNamespace(z_dim=3)

        def z_priv(self, idx: int) -> torch.Tensor:
            return torch.full((4, 3), float(idx) / 10.0)

    monkeypatch.setattr(ttv, "_build_privileged", lambda cfg: FakeBuilder())
    _use_episodes(monkeypatch, _episodes())
    metrics = ttv.run(_cfg(tmp_path, stage="privileged", truth_sidecar=tmp_path / "truth.bin"))
    assert metrics.stages["privileged"]["details"]["privileged_batches"] > 0


def test_dagger_dataset_provenance_and_fingerprint() -> None:
    cfg = _cfg(Path("/tmp"))
    episodes = _episodes()
    fit, inner, held = ttv.partition_episodes(episodes, seed=1, n_held_out=2)
    first = ttv._build_dagger_dataset(cfg, [*fit, *inner], held)
    second = ttv._build_dagger_dataset(cfg, [*fit, *inner], held)
    assert first.fingerprint() == second.fingerprint()
    fit_ids = {s.provenance.scenario_id for s in first.fit_samples()}
    held_ids = {s.provenance.scenario_id for s in first.held_out_samples()}
    assert fit_ids and held_ids and not (fit_ids & held_ids)
    sample = first.samples[0]
    assert sample.correction.shape == (2,)
    assert np.isfinite(sample.correction).all()
    assert np.allclose(sample.correction, sample.authoritative)
    assert ttv.DAGGER_SOURCE in {"dagger", "local"}


def test_dagger_stage_reports_dataset(tmp_path: Path, monkeypatch) -> None:
    _use_episodes(monkeypatch, _episodes())
    metrics = ttv.run(_cfg(tmp_path, stage="dagger"))
    details = metrics.stages["dagger"]["details"]
    assert metrics.stage == "dagger"
    assert details["dataset_fingerprint"]
    assert details["fit_samples"] > 0 and details["held_out_samples"] > 0
    assert details["label_source"].startswith("reference_optimizer")


def test_verifiable_reward_formula_gate_and_determinism() -> None:
    baseline = torch.zeros(2, 2)
    target = torch.full((2, 2), 0.5)
    candidate = torch.tensor([[0.2, 0.1], [-0.1, 0.0]])
    reward, parts = ttv.verifiable_reward(candidate, baseline, target, g_max=1.0)
    lateral = (baseline - target).abs().mean(dim=-1) - (candidate + baseline - target).abs().mean(
        dim=-1
    )
    expected = 2.0 * lateral - 0.5 * candidate.abs().mean(dim=-1)
    assert torch.allclose(reward, expected)
    assert bool(parts["valid"].all())
    bad = candidate.clone()
    bad[0, 0] = 2.0
    _reward_bad, parts_bad = ttv.verifiable_reward(bad, baseline, target, g_max=1.0)
    assert not bool(parts_bad["valid"][0])
    assert float(parts_bad["risk"][0]) == 1.0
    with pytest.raises(ValueError, match="NaN or infinity"):
        ttv.verifiable_reward(candidate * float("nan"), baseline, target, g_max=1.0)
    again, _ = ttv.verifiable_reward(candidate, baseline, target, g_max=1.0)
    assert torch.equal(reward, again)


def test_rlvr_stage_records_reward_provenance(tmp_path: Path, monkeypatch) -> None:
    _use_episodes(monkeypatch, _episodes())
    cfg = _cfg(tmp_path, stage="rlvr")
    metrics = ttv.run(cfg)
    provenance = metrics.reward_provenance
    assert provenance["seed"] == cfg.seed
    assert provenance["config_hash"] == metrics.config_hash
    assert provenance["samples"] > 0
    assert set(provenance["weights"]) == {"lateral", "effort", "risk"}
    assert "v3" in provenance["source"]
    assert "R =" in provenance["formula"]
    assert np.isfinite(provenance["total_mean"])
    details = metrics.stages["rlvr"]["details"]
    assert details["role"] == "checkpoint_selection"
    assert details["optimizer_steps"] == 0
    assert details["selected_checkpoint"] in {"sft", "privileged", "dagger"}
    saved = torch.load(cfg.out / "teacher_selected.pt", map_location="cpu", weights_only=True)
    assert saved["selected"] == details["selected_checkpoint"]
    assert saved["role"] == "checkpoint_selection"


def test_distill_stage_writes_student(tmp_path: Path, monkeypatch) -> None:
    _use_episodes(monkeypatch, _episodes())
    cfg = _cfg(tmp_path, stage="distill")
    metrics = ttv.run(cfg)
    assert metrics.stage == "distill"
    assert metrics.stages_completed == ttv.STAGE_ORDER
    assert "rlvr" not in metrics.stages_completed
    path = cfg.out / "student.pt"
    assert path.exists()
    saved = torch.load(path, map_location="cpu", weights_only=True)
    assert saved["seed"] == cfg.seed
    assert saved["config_hash"] == metrics.config_hash
    assert saved["source"] in {"gdn2_student", "local"}
    assert metrics.module_sources["student"] == saved["source"]


def test_default_pipeline_is_dagger_distillation_without_rlvr(tmp_path: Path, monkeypatch) -> None:
    _use_episodes(monkeypatch, _episodes())
    cfg = _cfg(tmp_path, stage="all")
    assert ttv.STAGE_ORDER == ("sft", "privileged", "dagger", "distill")
    metrics = ttv.run(cfg)
    assert metrics.stages_completed == ttv.STAGE_ORDER
    assert "rlvr" not in metrics.stages
    assert metrics.reward_provenance == {}
    assert not (cfg.out / "teacher_selected.pt").exists()


def test_rlvr_opt_in_selects_before_distill(tmp_path: Path, monkeypatch) -> None:
    _use_episodes(monkeypatch, _episodes())
    metrics = ttv.run(_cfg(tmp_path, stage="all", rlvr=True))
    assert metrics.stages_completed == ("sft", "privileged", "dagger", "rlvr", "distill")
    assert metrics.stages_completed.index("rlvr") < metrics.stages_completed.index("distill")
    details = metrics.stages["rlvr"]["details"]
    assert set(details["candidates_scored"]) == {"sft", "privileged", "dagger"}
    assert details["optimizer_steps"] == 0


def test_rlvr_selection_takes_no_policy_gradient(tmp_path: Path, monkeypatch) -> None:
    _use_episodes(monkeypatch, _episodes())
    real_build = ttv._build_optimizer
    real_reward = ttv.verifiable_reward
    builds = {"count": 0}

    def counting_build(parameters, cfg, lr):
        builds["count"] += 1
        return real_build(parameters, cfg, lr)

    def verifier_only(candidate, baseline, target, g_max):
        assert not torch.is_grad_enabled()
        return real_reward(candidate, baseline, target, g_max)

    monkeypatch.setattr(ttv, "_build_optimizer", counting_build)
    monkeypatch.setattr(ttv, "verifiable_reward", verifier_only)
    metrics = ttv.run(_cfg(tmp_path, stage="all", rlvr=True))
    assert builds["count"] == len([s for s in metrics.stages_completed if s != "rlvr"])


def test_eval_contamination_refused_before_training(tmp_path: Path, monkeypatch) -> None:
    episodes = _episodes() + [make_smart_episode(9, frames=24, scenario_id=17)]
    _use_episodes(monkeypatch, episodes)
    with pytest.raises(ValueError, match="eval scenario"):
        ttv.run(_cfg(tmp_path, stage="sft"))


def test_cli_parser_builds_typed_config() -> None:
    args = ttv.build_parser().parse_args(
        ["--data", "d", "--out", "o", "--seed", "2", "--stage", "rlvr", "--learned-exit"]
    )
    cfg = ttv.TeacherV2Config.from_args(args)
    assert cfg.seed == 2 and cfg.stage == "rlvr" and cfg.learned_exit
    assert cfg.data == Path("d") and cfg.out == Path("o")
    with pytest.raises(SystemExit):
        ttv.build_parser().parse_args(["--data", "d", "--out", "o", "--seed", "9"])
    opt_in = ttv.build_parser().parse_args(["--data", "d", "--out", "o", "--rlvr"])
    assert ttv.TeacherV2Config.from_args(opt_in).rlvr is True


def test_teacher_config_keeps_post05_defaults_off(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    base = ttv._teacher_config(cfg)
    assert base.learned_exit is False and base.spectral_residual is False
    enabled = ttv._teacher_config(
        _cfg(
            tmp_path,
            learned_exit=True,
            spectral_residual=True,
            exit_gate_threshold=0.7,
            exit_entropy_alpha=0.02,
        )
    )
    assert enabled.learned_exit and enabled.spectral_residual
    assert enabled.exit_gate_threshold == 0.7 and enabled.exit_entropy_alpha == 0.02
    assert isinstance(enabled, LoopWMConfig)
