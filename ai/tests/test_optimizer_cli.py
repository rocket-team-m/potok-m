# SPDX-License-Identifier: GPL-3.0-or-later
"""MUON04 gates: typed K3 flags reach both training CLIs' optimizers.

The K3 arms are name-based, so passing ``model.parameters()`` made them inert.
These gates pin that both CLIs now construct a typed ``K3Config`` and hand named
parameters to ``MuonPlus``. Defaults stay off and keep the historical optimizers.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.optim import AdamW

from potok_ai import train as train_cli
from potok_ai import train_teacher_v2 as ttv
from potok_ai.data import make_smart_episode
from potok_ai.optim import K3Config, MuonPlus


class _Attn(nn.Module):
    """Dense projections named the way K3Config detects them."""

    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(6, 8, bias=False)
        self.k_proj = nn.Linear(6, 8, bias=False)
        self.v_proj = nn.Linear(6, 8, bias=False)


def _train_args(*extra: str):
    return train_cli.build_parser().parse_args(
        ["--out", "model.pt", "--synthetic", "--epochs", "1", "--lr", "0.01", *extra]
    )


def _ttv_args(*extra: str):
    return ttv.build_parser().parse_args(["--data", "d", "--out", "o", *extra])


def test_train_parser_k3_defaults_off() -> None:
    args = _train_args()
    assert args.opt == "muon" and args.ns_steps == 5
    assert (args.k3_per_head, args.k3_qk_clip, args.k3_qk_clip_tau) == (0, False, 100.0)
    assert train_cli.k3_config(args) == K3Config()


def test_train_parser_k3_flags_build_typed_config() -> None:
    args = _train_args("--k3-per-head", "3", "--k3-qk-clip", "--k3-qk-clip-tau", "7.5")
    cfg = train_cli.k3_config(args)
    assert cfg.num_heads == 3 and cfg.qk_clip is True and cfg.qk_clip_tau == 7.5


def test_train_optimizer_gets_named_params_and_k3() -> None:
    model = _Attn()
    args = _train_args("--k3-per-head", "2", "--k3-qk-clip", "--k3-qk-clip-tau", "5")
    optimizer = train_cli._optimizer(list(model.named_parameters()), args)
    assert isinstance(optimizer, MuonPlus)
    assert optimizer.k3 == train_cli.k3_config(args)
    assert len(optimizer._names) == 3  # names reached the optimizer (no longer inert)
    assert len(optimizer._qk_pairs) == 1  # QK-Clip found the q/k pair


def test_train_default_optimizer_gate_unchanged() -> None:
    model = _Attn()
    muon = train_cli._optimizer(list(model.named_parameters()), _train_args())
    assert isinstance(muon, MuonPlus) and muon.k3 == K3Config()
    adam = train_cli._optimizer(list(model.named_parameters()), _train_args("--opt", "adamw"))
    assert isinstance(adam, AdamW)
    for group in adam.param_groups:
        assert all(isinstance(p, torch.Tensor) for p in group["params"])


def test_train_run_with_k3_per_head_stays_finite(tmp_path: Path) -> None:
    args = train_cli.build_parser().parse_args(
        [
            "--stage",
            "distill",
            "--epochs",
            "1",
            "--seed",
            "0",
            "--device",
            "cpu",
            "--out",
            str(tmp_path / "model.pt"),
            "--synthetic",
            "--batch",
            "2",
            "--seq-len",
            "16",
            "--limit",
            "4",
            "--width",
            "16",
            "--rank",
            "8",
            "--lr",
            "0.01",
            "--k3-per-head",
            "2",
        ]
    )
    assert math.isfinite(train_cli.run(args))
    assert (tmp_path / "model.pt").exists()


def test_teacher_v2_parser_k3_defaults_off() -> None:
    cfg = ttv.TeacherV2Config.from_args(_ttv_args())
    assert cfg.opt == "adamw"
    assert (cfg.k3_per_head, cfg.k3_qk_clip, cfg.k3_qk_clip_tau) == (0, False, 100.0)
    assert ttv.k3_config(cfg) == K3Config()


def test_teacher_v2_k3_flags_build_typed_config() -> None:
    cfg = ttv.TeacherV2Config.from_args(
        _ttv_args("--opt", "muon", "--k3-per-head", "2", "--k3-qk-clip", "--k3-qk-clip-tau", "9")
    )
    assert cfg.opt == "muon"
    assert (cfg.k3_per_head, cfg.k3_qk_clip, cfg.k3_qk_clip_tau) == (2, True, 9.0)
    assert ttv.k3_config(cfg) == K3Config(num_heads=2, qk_clip=True, qk_clip_tau=9.0)


def test_teacher_v2_k3_validation_fails_loudly() -> None:
    with pytest.raises(ValueError, match="num_heads"):
        ttv.TeacherV2Config.from_args(_ttv_args("--k3-qk-clip"))
    with pytest.raises(ValueError, match="tau"):
        ttv.TeacherV2Config.from_args(_ttv_args("--k3-qk-clip-tau", "0"))


def test_teacher_v2_optimizer_builder_defaults_and_k3() -> None:
    model = _Attn()
    default = ttv.TeacherV2Config.from_args(_ttv_args())
    adam = ttv._build_optimizer(list(model.named_parameters()), default, 0.01)
    assert isinstance(adam, AdamW)
    for group in adam.param_groups:
        assert all(isinstance(p, torch.Tensor) for p in group["params"])

    cfg = ttv.TeacherV2Config.from_args(
        _ttv_args("--opt", "muon", "--k3-per-head", "2", "--k3-qk-clip", "--k3-qk-clip-tau", "5")
    )
    muon = ttv._build_optimizer(list(model.named_parameters()), cfg, 0.01)
    assert isinstance(muon, MuonPlus)
    assert muon.k3 == ttv.k3_config(cfg)
    assert len(muon._names) == 3 and len(muon._qk_pairs) == 1


def test_teacher_v2_muon_stage_runs_with_k3_per_head(tmp_path: Path, monkeypatch) -> None:
    episodes = [make_smart_episode(seed=100 + i, frames=24, scenario_id=i) for i in range(4)]
    monkeypatch.setattr(ttv, "load_episodes", lambda _path: episodes)
    cfg = ttv.TeacherV2Config(
        data=tmp_path / "data",
        out=tmp_path / "out",
        seed=1,
        stage="sft",
        epochs=1,
        batch=2,
        window=12,
        limit=6,
        width=16,
        heads=2,
        max_iters=2,
        opt="muon",
        k3_per_head=2,
    )
    metrics = ttv.run(cfg)
    assert metrics.stages_completed == ("sft",)
    assert math.isfinite(metrics.rmse_train) and math.isfinite(metrics.rmse_held_out)
