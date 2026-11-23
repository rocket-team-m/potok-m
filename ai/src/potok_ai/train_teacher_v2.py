# SPDX-License-Identifier: GPL-3.0-or-later
"""Staged teacher-v2 trainer (ENGINE01).

One CLI, one typed config and one ``metrics.json`` for the five-stage teacher
pipeline:

    sft -> privileged (LoopWM + JEPA v2) -> dagger -> rlvr -> distill

``--stage X`` runs every stage through X (so a later stage never starts from a
fresh model). Determinism is structural: seeds 1/2/3 only, named child streams
for every auxiliary RNG, split by scenario with final-eval cells 16..19 refused
before any model sees them. Nothing here needs a GPU and no production training
is launched by importing or testing this module.

Siblings from POST03/04/05/JEPA01/POST07 are optional imports. Each one records
which arm ran in ``metrics.json``; a missing module falls back to a narrow local
implementation of exactly the interface used here, never to a silent no-op.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW

from .calibration import confidence_target, correctness, expected_calibration_error, sigmoid
from .data import Episode, SmartWindowDataset, load_episodes
from .loopwm import LoopWM, LoopWMConfig
from .physics import final_command, physics_baseline
from .train_smart import generalisation_ratio, normalized_correction_loss

# ---------------------------------------------------------------------------
# Optional siblings. A partial/unwritten sibling module must not brick the CLI.
# ---------------------------------------------------------------------------
try:
    from .jepa_v2 import JEPAV2 as _UpstreamJEPAV2
    from .jepa_v2 import JEPAV2Config as _UpstreamJEPAV2Config

    JEPA_SOURCE = "jepa_v2"
except (ImportError, SyntaxError):  # pragma: no cover - depends on sibling task
    _UpstreamJEPAV2 = None
    _UpstreamJEPAV2Config = None
    JEPA_SOURCE = "local"

try:
    from .dagger import RolloutState
    from .dagger import build_dagger_dataset as _upstream_build_dagger

    DAGGER_SOURCE = "dagger"
except (ImportError, SyntaxError):  # pragma: no cover - depends on sibling task
    _upstream_build_dagger = None

    @dataclass(frozen=True, eq=False)
    class RolloutState:  # type: ignore[no-redef]
        """Narrow stand-in for dagger.RolloutState."""

        observation: np.ndarray
        privileged: np.ndarray
        authoritative: np.ndarray | None = None

    DAGGER_SOURCE = "local"

try:
    from .privileged_teacher import PrivilegedFeatureBuilder as _UpstreamPrivilegedBuilder

    PRIVILEGED_SOURCE = "privileged_teacher"
except (ImportError, SyntaxError):  # pragma: no cover - depends on sibling task
    _UpstreamPrivilegedBuilder = None
    PRIVILEGED_SOURCE = "local"

try:
    from .gdn2_student import GDN2Student as _UpstreamStudent
    from .gdn2_student import GDN2StudentConfig as _UpstreamStudentConfig

    STUDENT_SOURCE = "gdn2_student"
except (ImportError, SyntaxError):  # pragma: no cover - depends on sibling task
    _UpstreamStudent = None
    _UpstreamStudentConfig = None
    STUDENT_SOURCE = "local"


class JepaObjective(Protocol):
    """The two calls the privileged stage needs from a JEPA objective."""

    def loss(self, student: nn.Module, window: torch.Tensor, cond: torch.Tensor | None) -> Any: ...

    def update_teacher(self, student: nn.Module) -> None: ...


class StudentLike(Protocol):
    """The one call the distillation stage needs from a student."""

    def forward(self, window: torch.Tensor, cond: torch.Tensor) -> torch.Tensor: ...


STAGE_ORDER = ("sft", "privileged", "dagger", "rlvr", "distill")
SEEDS = (1, 2, 3)
EVAL_SCENARIO_IDS = frozenset({16, 17, 18, 19})
JEPA_HORIZONS = (3, 6, 10)
REWARD_WEIGHTS = {"lateral": 2.0, "effort": 0.5, "risk": 1.0}
SOFT_LIMIT_FRACTION = 0.8
REWARD_FORMULA = (
    "R = w_lat*(mean|b-u| - mean|b+t-u|) - w_eff*mean|u| - w_risk*soft_violations; "
    "hard gate: finite u and |u| <= g_max rejects the candidate from the group baseline"
)


def child_seed(seed: int, name: str) -> int:
    """Named stream, stable across processes and versions."""
    digest = hashlib.sha256(f"{int(seed)}:engine01:{name}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


@dataclass(frozen=True)
class TeacherV2Config:
    """All knobs of the staged run; every invalid value fails loudly."""

    data: Path
    out: Path
    stage: str = "all"
    seed: int = 1
    device: str = "cpu"
    epochs: int = 1
    batch: int = 2
    window: int = 12
    limit: int | None = None
    lr: float = 2e-3
    student_lr: float = 1e-3
    width: int = 16
    heads: int = 2
    max_iters: int = 2
    learned_exit: bool = False
    exit_gate_threshold: float = 0.5
    exit_entropy_alpha: float = 0.01
    spectral_residual: bool = False
    spectral_bound: float = 1.0
    jepa_weight: float = 0.05
    entropy_weight: float = 1.0
    dagger_rounds: int = 1
    n_inner_held_out: int = 2
    rlvr_samples: int = 4
    rlvr_temperature: float = 0.05
    truth_sidecar: Path | None = None
    calib_tol: float = 0.15

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", Path(self.data))
        object.__setattr__(self, "out", Path(self.out))
        if self.truth_sidecar is not None:
            object.__setattr__(self, "truth_sidecar", Path(self.truth_sidecar))
        if self.stage not in (*STAGE_ORDER, "all"):
            raise ValueError(
                f"unknown stage {self.stage!r}; expected one of all, {', '.join(STAGE_ORDER)}"
            )
        if self.seed not in SEEDS:
            raise ValueError(f"seed must be one of {SEEDS}, got {self.seed}")
        if self.device != "cpu":
            raise ValueError(f"only the CPU path is supported, got device {self.device!r}")
        for name in (
            "epochs",
            "batch",
            "window",
            "width",
            "heads",
            "max_iters",
            "dagger_rounds",
            "n_inner_held_out",
            "rlvr_samples",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value}")
        if self.rlvr_samples < 2:
            raise ValueError("rlvr_samples must be at least 2 for a group baseline")
        if self.limit is not None and self.limit < 1:
            raise ValueError(f"limit must be positive or None, got {self.limit}")
        if self.width % self.heads:
            raise ValueError(f"width {self.width} must be divisible by heads {self.heads}")
        if self.window <= max(JEPA_HORIZONS):
            raise ValueError(f"window must exceed the longest JEPA horizon {max(JEPA_HORIZONS)}")
        for name in ("lr", "student_lr", "rlvr_temperature", "calib_tol", "spectral_bound"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive, got {value}")
        for name in ("jepa_weight", "entropy_weight", "exit_entropy_alpha"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative, got {value}")
        if not 0.0 <= self.exit_gate_threshold <= 1.0:
            raise ValueError(
                f"exit_gate_threshold must lie in [0, 1], got {self.exit_gate_threshold}"
            )
        if not 0.0 < self.rlvr_temperature < 1.0:
            raise ValueError(f"rlvr_temperature must lie in (0, 1), got {self.rlvr_temperature}")

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> TeacherV2Config:
        fields = {name: getattr(args, name) for name in cls.__dataclass_fields__}
        return cls(**fields)


def config_hash(cfg: TeacherV2Config) -> str:
    """Hash of the effective knobs; the output path is deliberately excluded."""
    fields = asdict(cfg)
    fields.pop("out", None)
    body = json.dumps(fields, sort_keys=True, default=str)
    return hashlib.sha256(body.encode()).hexdigest()[:16]


def seed_everything(seed: int) -> None:
    """CPU only, one thread; auxiliary RNGs use named child streams instead."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


@dataclass
class StageReport:
    stage: str
    steps: int
    rmse_train: float
    rmse_held_out: float
    ratio: float
    ece: float
    entropy: float = 0.0
    entropy_enabled: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TeacherV2Metrics:
    stage: str
    seed: int
    config_hash: str
    rmse_train: float
    rmse_held_out: float
    ratio: float
    ece: float
    steps: int
    entropy: float = 0.0
    entropy_enabled: bool = False
    reward_provenance: dict[str, Any] = field(default_factory=dict)
    module_sources: dict[str, str] = field(default_factory=dict)
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)
    stages_completed: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        result = self.__dict__.copy()
        result["stages_completed"] = list(self.stages_completed)
        return result


def write_metrics(out: Path, metrics: TeacherV2Metrics) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / "metrics.json"
    path.write_text(json.dumps(metrics.as_dict(), indent=2, sort_keys=True) + "\n")
    return path


def partition_episodes(
    episodes: list[Episode], *, seed: int, n_held_out: int = 2
) -> tuple[list[Episode], list[Episode], frozenset[int]]:
    """Disjoint fit/inner-held-out split by scenario; eval cells are refused."""
    if n_held_out < 1:
        raise ValueError(f"n_held_out must be positive, got {n_held_out}")
    train: list[Episode] = []
    for episode in episodes:
        sid = episode.scenario_id
        if episode.held_out or (sid is not None and sid in EVAL_SCENARIO_IDS):
            raise ValueError(
                f"eval scenario {sid} cannot enter teacher-v2 training; "
                "use training cells 0..15 only"
            )
        if sid is None:
            raise ValueError("every episode needs a numeric scenario id for the split")
        train.append(episode)
    ids = sorted({int(episode.scenario_id) for episode in train})
    if len(ids) <= n_held_out:
        raise ValueError(f"need at least {n_held_out + 1} training scenarios, got {len(ids)}")
    held = set(random.Random(f"{seed}:engine01:split").sample(ids, n_held_out))
    fit = [episode for episode in train if int(episode.scenario_id) not in held]
    inner = [episode for episode in train if int(episode.scenario_id) in held]
    if not fit or not inner or ({e.scenario_id for e in fit} & {e.scenario_id for e in inner}):
        raise AssertionError("fit/held-out partition is not disjoint and non-empty")
    return fit, inner, frozenset(held)


# ---------------------------------------------------------------------------
# Data helpers.
# ---------------------------------------------------------------------------
def _window_batches(
    episodes: list[Episode], cfg: TeacherV2Config
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int | None]]]:
    """Materialise tiny CPU windows once; return (window, cond, target, episode ids)."""
    samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, int | None]] = []
    for episode in episodes:
        dataset = SmartWindowDataset([episode], window=cfg.window, limit=cfg.limit)
        for window, cond, target in dataset.samples:
            samples.append((window, cond, target, episode.index))
    if cfg.limit is not None:
        samples = samples[: cfg.limit]
    if not samples:
        raise ValueError("no training windows: episodes must be longer than the window")
    windows = torch.stack([torch.as_tensor(s[0], dtype=torch.float32) for s in samples])
    conds = torch.stack([torch.as_tensor(s[1], dtype=torch.float32) for s in samples])
    targets = torch.stack([torch.as_tensor(s[2], dtype=torch.float32) for s in samples])
    ids = [s[3] for s in samples]
    batches = []
    for start in range(0, len(samples), cfg.batch):
        stop = start + cfg.batch
        batches.append(
            (windows[start:stop], conds[start:stop], targets[start:stop], ids[start:stop])
        )
    return batches


def _predict(
    model: nn.Module,
    window: torch.Tensor,
    cond: torch.Tensor,
    target: torch.Tensor,
    *,
    student: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (command, confidence logit, per-sample command error)."""
    if student:
        output = model.forward(window[:, :, :14], cond)  # type: ignore[operator]
        if output.ndim != 3 or output.shape[0] != window.shape[0]:
            raise AssertionError(f"student output has wrong shape {tuple(output.shape)}")
        correction, confidence = output[:, -1, :2], output[:, -1, 2:3]
        logit = torch.logit(confidence.clamp(1e-6, 1.0 - 1e-6)).squeeze(-1)
    else:
        latent, _ = model.refine(window, cond)  # type: ignore[attr-defined]
        raw = model.head_logits(latent)  # type: ignore[attr-defined]
        output = model.head_output(raw)  # type: ignore[attr-defined]
        correction, logit = output[:, :2], raw[:, 2]
    command = final_command(correction, physics_baseline(window[:, -1]), residual=True)
    if command.shape != target.shape:
        raise AssertionError(f"command {tuple(command.shape)} != target {tuple(target.shape)}")
    return command, logit, (command - target).abs().max(dim=1).values.detach()


def _evaluate(
    model: nn.Module,
    fit: list[Episode],
    inner: list[Episode],
    cfg: TeacherV2Config,
    device: torch.device,
    *,
    student: bool = False,
) -> tuple[float, float, float, float]:
    """Return train RMSE, held-out RMSE, generalisation ratio and inner ECE."""
    model.eval()
    held_ids = {int(e.scenario_id) for e in inner}
    rmse: dict[str, float] = {}
    logits: list[float] = []
    errors: list[float] = []
    with torch.no_grad():
        for sid in sorted({int(e.scenario_id) for e in (*fit, *inner)}):
            group = [e for e in (*fit, *inner) if int(e.scenario_id) == sid]
            squared, count = 0.0, 0
            for window, cond, target, _ in _window_batches(group, cfg):
                command, logit, error = _predict(
                    model, window.to(device), cond.to(device), target.to(device), student=student
                )
                squared += float((command - target.to(device)).square().sum())
                count += target.numel()
                if sid in held_ids:
                    logits.extend(logit.cpu().tolist())
                    errors.extend(error.cpu().tolist())
            if count == 0:
                raise ValueError(f"scenario {sid} has no scored windows")
            rmse[str(sid)] = (squared / count) ** 0.5
    if not logits or not errors:
        raise ValueError("inner-held-out split produced no predictions to calibrate")
    ratio = generalisation_ratio(rmse, {str(sid) for sid in held_ids})
    ece = expected_calibration_error(
        sigmoid(np.asarray(logits)), correctness(np.asarray(errors), cfg.calib_tol)
    )
    held_names = {str(sid) for sid in held_ids}
    train_values = [value for key, value in rmse.items() if key not in held_names]
    held_values = [rmse[name] for name in held_names if name in rmse]
    if not train_values or len(held_values) != len(held_names):
        raise ValueError("fit and inner-held-out scenarios must both contribute RMSE")
    return sum(train_values) / len(train_values), sum(held_values) / len(held_values), ratio, ece


# ---------------------------------------------------------------------------
# Models and the supervised / entropy objective shared by teacher stages.
# ---------------------------------------------------------------------------
def _teacher_config(cfg: TeacherV2Config) -> LoopWMConfig:
    return LoopWMConfig(
        width=cfg.width,
        heads=cfg.heads,
        max_iters=cfg.max_iters,
        learned_exit=cfg.learned_exit,
        exit_gate_threshold=cfg.exit_gate_threshold,
        exit_entropy_alpha=cfg.exit_entropy_alpha,
        spectral_residual=cfg.spectral_residual,
        spectral_bound=cfg.spectral_bound,
    )


def _build_teacher(cfg: TeacherV2Config, device: torch.device) -> LoopWM:
    return LoopWM(_teacher_config(cfg)).to(device)


def _supervised_loss(
    model: LoopWM,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int | None]],
    cfg: TeacherV2Config,
    *,
    jepa: JepaObjective | None = None,
    privileged_head: nn.Linear | None = None,
    privileged_target: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Command + confidence + optional JEPA/entropy/privileged auxiliary terms."""
    window, cond, target, _ = batch
    latent, info = model.refine(window, cond)
    raw = model.head_logits(latent)
    output = model.head_output(raw)
    command = final_command(output[:, :2], physics_baseline(window[:, -1]), residual=True)
    if command.shape != target.shape:
        raise AssertionError(f"command {tuple(command.shape)} != target {tuple(target.shape)}")
    error = (command - target).abs().max(dim=1).values.detach()
    main = normalized_correction_loss(command, target)
    conf = F.binary_cross_entropy_with_logits(raw[:, 2], confidence_target(error, cfg.calib_tol))
    entropy = model.exit_entropy_loss(info) if cfg.learned_exit else raw.new_zeros(())
    total = main + conf + cfg.entropy_weight * entropy
    if jepa is not None:
        total = total + cfg.jepa_weight * jepa.loss(model.encoder, window, None).total
    if privileged_head is not None:
        if privileged_target is None or privileged_target.shape[0] != window.shape[0]:
            raise ValueError("privileged target must be one row per window")
        total = total + F.mse_loss(privileged_head(latent[:, -1]), privileged_target)
    return {
        "loss": total,
        "main": float(main.detach()),
        "conf": float(conf.detach()),
        "entropy": float(entropy.detach()),
    }


# ---------------------------------------------------------------------------
# Narrow local fallbacks (only used when a sibling module is absent).
# ---------------------------------------------------------------------------
@dataclass
class _LocalJepaLosses:
    total: torch.Tensor


class _LocalJEPAV2(nn.Module):
    """One-shot shifted-latent probe; the interface the privileged stage uses."""

    def __init__(self, encoder: nn.Module, width: int, seed: int) -> None:
        super().__init__()
        self.teacher = _LocalEMATeacher(encoder)
        generator = torch.Generator().manual_seed(seed + 1)
        self.probe = nn.Linear(width, width)
        nn.init.normal_(self.probe.weight, std=width**-0.5, generator=generator)
        self._rng = generator

    @torch.no_grad()
    def update_teacher(self, student: nn.Module) -> None:
        for teacher_param, student_param in zip(
            self.teacher.net.parameters(), student.parameters()
        ):
            teacher_param.mul_(0.99).add_(student_param, alpha=0.01)

    def loss(
        self, student: nn.Module, window: torch.Tensor, cond: torch.Tensor | None
    ) -> _LocalJepaLosses:
        batch, time, _ = window.shape
        if time <= max(JEPA_HORIZONS):
            raise ValueError(f"window {tuple(window.shape)} is too short for JEPA")
        span = max(1, int(time * 0.25))
        starts = torch.randint(0, time - span + 1, (batch,), generator=self._rng)
        positions = torch.arange(time).unsqueeze(0)
        mask = (positions >= starts[:, None]) & (positions < starts[:, None] + span)
        visible = torch.where(mask.unsqueeze(-1), torch.zeros_like(window), window)
        source = student.forward_features(visible)[:, : time - 3]
        with torch.no_grad():
            target = self.teacher.net.forward_features(window)[:, 3:]
        return _LocalJepaLosses(F.mse_loss(self.probe(source), target))


class _LocalEMATeacher(nn.Module):
    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        from copy import deepcopy

        self.net = deepcopy(encoder).eval()
        for parameter in self.net.parameters():
            parameter.requires_grad_(False)


class _LocalStudent(nn.Module):
    """Tiny deterministic student when gdn2_student is unavailable."""

    def __init__(self, seed: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(20, 32), nn.Tanh(), nn.Linear(32, 3))

        def init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.1, generator=generator)
                nn.init.zeros_(module.bias)

        self.apply(init)

    def forward(self, window: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if window.ndim != 3 or window.shape[-1] != 14:
            raise ValueError(f"expected (batch, time, 14) frames, got {tuple(window.shape)}")
        conds = cond.unsqueeze(1).expand(-1, window.shape[1], -1)
        raw = self.net(torch.cat([window, conds], dim=-1))
        return torch.cat([torch.tanh(raw[..., :2]), torch.sigmoid(raw[..., 2:3])], dim=-1)


@dataclass(frozen=True)
class _LocalProvenance:
    sample_id: str
    seed: int
    scenario_id: int | None
    split: str
    expert: str
    label_source: str


@dataclass(eq=False)
class _LocalDaggerSample:
    observation: np.ndarray
    correction: np.ndarray
    authoritative: np.ndarray | None
    provenance: _LocalProvenance


class _LocalDaggerDataset:
    def __init__(self, samples: list[_LocalDaggerSample], fingerprint: str) -> None:
        self.samples = samples
        self._fingerprint = fingerprint

    def fit_samples(self) -> list[_LocalDaggerSample]:
        return [s for s in self.samples if s.provenance.split == "fit"]

    def held_out_samples(self) -> list[_LocalDaggerSample]:
        return [s for s in self.samples if s.provenance.split == "held_out"]

    def fingerprint(self) -> str:
        return self._fingerprint


# ---------------------------------------------------------------------------
# Stage implementations.
# ---------------------------------------------------------------------------
def _stage_sft(
    model: LoopWM,
    cfg: TeacherV2Config,
    fit: list[Episode],
    inner: list[Episode],
    device: torch.device,
) -> StageReport:
    optimizer = AdamW(model.parameters(), lr=cfg.lr)
    batches = _window_batches(fit, cfg)
    steps = 0
    entropy = 0.0
    model.train()
    for _ in range(cfg.epochs):
        for batch in batches:
            window, cond, target = (t.to(device) for t in batch[:3])
            result = _supervised_loss(model, (window, cond, target, batch[3]), cfg)
            optimizer.zero_grad(set_to_none=False)
            result["loss"].backward()
            optimizer.step()
            entropy = result["entropy"]
            steps += 1
    train, held, ratio, ece = _evaluate(model, fit, inner, cfg, device)
    return StageReport("sft", steps, train, held, ratio, ece, entropy, cfg.learned_exit)


def _build_jepa(encoder: nn.Module, cfg: TeacherV2Config) -> JepaObjective:
    if _UpstreamJEPAV2 is not None and _UpstreamJEPAV2Config is not None:
        jcfg = _UpstreamJEPAV2Config(
            width=cfg.width, horizons=JEPA_HORIZONS, seed=child_seed(cfg.seed, "jepa")
        )
        return _UpstreamJEPAV2(encoder, jcfg)
    return _LocalJEPAV2(encoder, cfg.width, child_seed(cfg.seed, "jepa"))


def _build_privileged(cfg: TeacherV2Config) -> Any | None:
    if cfg.truth_sidecar is None or _UpstreamPrivilegedBuilder is None:
        return None
    return _UpstreamPrivilegedBuilder(cfg.truth_sidecar, seed=child_seed(cfg.seed, "privileged"))


def _stage_privileged(
    model: LoopWM,
    cfg: TeacherV2Config,
    fit: list[Episode],
    inner: list[Episode],
    device: torch.device,
) -> StageReport:
    jepa = _build_jepa(model.encoder, cfg).to(device)
    builder = _build_privileged(cfg)
    head = (
        nn.Linear(cfg.width, int(builder.encoder.z_dim)).to(device)
        if builder is not None and hasattr(builder, "encoder")
        else None
    )
    parameters = list(model.parameters()) + list(jepa.parameters())  # type: ignore[attr-defined]
    if head is not None:
        parameters += list(head.parameters())
    optimizer = AdamW(parameters, lr=cfg.lr)
    batches = _window_batches(fit, cfg)
    steps, entropy, privileged_terms = 0, 0.0, 0
    model.train()
    jepa.train()  # type: ignore[attr-defined]
    for _ in range(cfg.epochs):
        for batch in batches:
            window, cond, target = (t.to(device) for t in batch[:3])
            z_priv = None
            if builder is not None and head is not None:
                z_priv = torch.stack(
                    [builder.z_priv(int(idx)).mean(dim=0) for idx in batch[3] if idx is not None]
                ).to(device)
                if z_priv.shape[0] != window.shape[0]:
                    raise ValueError("every privileged batch row needs a sidecar episode index")
                privileged_terms += 1
            result = _supervised_loss(
                model,
                (window, cond, target, batch[3]),
                cfg,
                jepa=jepa,
                privileged_head=head,
                privileged_target=z_priv,
            )
            optimizer.zero_grad(set_to_none=False)
            result["loss"].backward()
            optimizer.step()
            jepa.update_teacher(model.encoder)
            entropy = result["entropy"]
            steps += 1
    train, held, ratio, ece = _evaluate(model, fit, inner, cfg, device)
    details = {
        "jepa_source": JEPA_SOURCE,
        "privileged_source": PRIVILEGED_SOURCE,
        "privileged_batches": privileged_terms,
    }
    return StageReport(
        "privileged", steps, train, held, ratio, ece, entropy, cfg.learned_exit, details
    )


def _expert_label(privileged: np.ndarray) -> np.ndarray:
    """The v3 reference optimizer's answer lives in truth slots 22..25."""
    if privileged.shape[-1] < 26:
        raise ValueError(f"privileged snapshot needs 26 slots, got {privileged.shape}")
    return np.asarray([privileged[24], -privileged[25]], dtype=np.float32)


def _dagger_rollout(cfg: TeacherV2Config) -> Any:
    def rollout(episode: Episode, _rng: random.Random) -> list[RolloutState]:
        frames = np.asarray(episode.frames[:, :14], dtype=np.float32)
        if len(frames) < cfg.window:
            raise ValueError("episode shorter than the teacher window")
        cond = np.asarray(episode.conditioning, dtype=np.float32)
        target = episode.command_target()
        states: list[RolloutState] = []
        for start in range(0, len(frames) - cfg.window + 1, max(1, cfg.window // 2)):
            window = np.concatenate(
                [frames[start : start + cfg.window], np.tile(cond, (cfg.window, 1))], axis=1
            )
            tick = start + cfg.window - 1
            truth4 = np.zeros(4, dtype=np.float32)
            if episode.targets is not None:
                truth4[:2] = episode.targets[tick, :2]
            if episode.g_accel is not None:
                truth4[2:] = episode.g_accel[tick]
            else:
                truth4[2:] = (target[tick, 0], -target[tick, 1])
            # No truth sidecar: wind slots are a documented CPU fixture of zeros;
            # the correction still comes only from the reference optimizer slots.
            privileged = np.concatenate([np.zeros(22, dtype=np.float32), truth4])
            states.append(RolloutState(window, privileged, target[tick]))
        return states

    return rollout


def _build_dagger_dataset(
    cfg: TeacherV2Config, episodes: list[Episode], held_ids: frozenset[int]
) -> Any:
    if _upstream_build_dagger is not None:
        return _upstream_build_dagger(
            seed=cfg.seed,
            episodes=episodes,
            rollout=_dagger_rollout(cfg),
            teacher=_expert_label,
            rounds=cfg.dagger_rounds,
            rollouts_per_round=1,
        )
    # Local fallback: identical shapes/provenance, no extra dependency.
    samples: list[_LocalDaggerSample] = []
    rollout = _dagger_rollout(cfg)
    for episode in episodes:
        sid = int(episode.scenario_id)
        rng = random.Random(f"{cfg.seed}:dagger:episode:{episode.index}")
        for step, state in enumerate(rollout(episode, rng)):
            correction = _expert_label(state.privileged)
            if not np.isfinite(correction).all():
                raise ValueError(f"non-finite expert label at episode {sid} step {step}")
            samples.append(
                _LocalDaggerSample(
                    observation=state.observation,
                    correction=correction,
                    authoritative=state.authoritative,
                    provenance=_LocalProvenance(
                        sample_id=f"{cfg.seed}:dagger:{sid}:{step}",
                        seed=cfg.seed,
                        scenario_id=sid,
                        split="held_out" if sid in held_ids else "fit",
                        expert="reference_optimizer_v3",
                        label_source="expert",
                    ),
                )
            )
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(sample.provenance.sample_id.encode())
        digest.update(np.asarray(sample.observation, dtype=np.float32).tobytes())
        digest.update(sample.correction.tobytes())
    return _LocalDaggerDataset(samples, digest.hexdigest())


def _stage_dagger(
    model: LoopWM,
    cfg: TeacherV2Config,
    fit: list[Episode],
    inner: list[Episode],
    held_ids: frozenset[int],
    device: torch.device,
) -> StageReport:
    dataset = _build_dagger_dataset(cfg, [*fit, *inner], held_ids)
    rows = dataset.fit_samples()
    held_rows = dataset.held_out_samples()
    if not rows or not held_rows:
        raise ValueError("DAgger needs both fit and inner-held-out corrective samples")
    observations = torch.stack([torch.as_tensor(s.observation, dtype=torch.float32) for s in rows])
    corrections = torch.stack([torch.as_tensor(s.correction, dtype=torch.float32) for s in rows])
    optimizer = AdamW(model.parameters(), lr=cfg.lr)
    steps, entropy = 0, 0.0
    model.train()
    for _ in range(cfg.epochs):
        for start in range(0, len(rows), cfg.batch):
            chunk = observations[start : start + cfg.batch]
            window = chunk[:, :, :14].to(device)
            cond = chunk[:, 0, 14:].to(device)
            target = corrections[start : start + cfg.batch].to(device)
            result = _supervised_loss(model, (window, cond, target, [None] * len(target)), cfg)
            optimizer.zero_grad(set_to_none=False)
            result["loss"].backward()
            optimizer.step()
            entropy = result["entropy"]
            steps += 1
    train, held, ratio, ece = _evaluate(model, fit, inner, cfg, device)
    details = {
        "dagger_source": DAGGER_SOURCE,
        "dataset_fingerprint": dataset.fingerprint(),
        "fit_samples": len(rows),
        "held_out_samples": len(held_rows),
        "label_source": "reference_optimizer_v3:slots22..25",
    }
    return StageReport("dagger", steps, train, held, ratio, ece, entropy, cfg.learned_exit, details)


@dataclass(frozen=True)
class VerifiableReward:
    """All numbers needed to recompute the RLVR signal, plus where they came from."""

    seed: int
    config_hash: str
    source: str
    formula: str
    weights: dict[str, float]
    samples: int
    rejected: int
    lateral_gain: float
    effort: float
    risk: float
    total_mean: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def verifiable_reward(
    candidate: torch.Tensor, baseline: torch.Tensor, target: torch.Tensor, g_max: float
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Deterministic reward from values a verifier can recompute from v3 truth."""
    try:
        candidate, baseline, target = torch.broadcast_tensors(candidate, baseline, target)
    except RuntimeError as error:
        raise ValueError(f"candidate, baseline and target cannot broadcast: {error}") from error
    if candidate.ndim < 1 or candidate.shape[-1] == 0:
        raise ValueError(
            f"reward inputs must have a command dimension, got {tuple(candidate.shape)}"
        )
    if not torch.isfinite(candidate).all():
        raise ValueError("RLVR candidate contains NaN or infinity")
    rejected = (candidate.abs() > g_max).any(dim=-1)
    soft_violations = (candidate.abs() > SOFT_LIMIT_FRACTION * g_max).any(dim=-1)
    err_candidate = (candidate + baseline - target).abs().mean(dim=-1)
    err_baseline = (baseline - target).abs().mean(dim=-1)
    lateral = err_baseline - err_candidate
    effort = candidate.abs().mean(dim=-1)
    risk = soft_violations.to(candidate.dtype)
    total = REWARD_WEIGHTS["lateral"] * lateral - REWARD_WEIGHTS["effort"] * effort
    total = total - REWARD_WEIGHTS["risk"] * risk
    parts = {"valid": ~rejected, "lateral": lateral, "effort": effort, "risk": risk}
    return total, parts


def _stage_rlvr(
    model: LoopWM,
    cfg: TeacherV2Config,
    fit: list[Episode],
    inner: list[Episode],
    device: torch.device,
    cfg_hash: str,
) -> StageReport:
    optimizer = AdamW(model.parameters(), lr=cfg.lr)
    rng = torch.Generator().manual_seed(child_seed(cfg.seed, "rlvr"))
    batches = _window_batches(fit, cfg)
    steps, entropy = 0, 0.0
    totals: list[float] = []
    laterals: list[float] = []
    efforts: list[float] = []
    risks: list[float] = []
    valid_count, sample_count = 0, 0
    model.train()
    for _ in range(cfg.epochs):
        for batch in batches:
            window, cond, target = (t.to(device) for t in batch[:3])
            latent, info = model.refine(window, cond)
            raw = model.head_logits(latent)
            mean = model.head_output(raw)[:, :2]
            baseline = physics_baseline(window[:, -1])
            noise = torch.randn((cfg.rlvr_samples, *mean.shape), generator=rng, dtype=mean.dtype)
            candidate = mean.unsqueeze(0).detach() + cfg.rlvr_temperature * noise
            rewards, parts = verifiable_reward(
                candidate, baseline.unsqueeze(0), target.unsqueeze(0), float(model.cfg.g_max)
            )
            valid = parts["valid"]
            counts = valid.sum(dim=0).clamp_min(1)
            if not bool((counts > 0).all()):
                raise ValueError(
                    "every RLVR group needs at least one candidate that passes the gate"
                )
            masked = torch.where(valid, rewards, torch.zeros_like(rewards))
            group_mean = masked.sum(dim=0) / counts
            spread = torch.where(valid, (rewards - group_mean).square(), torch.zeros_like(rewards))
            group_std = torch.sqrt(spread.sum(dim=0) / counts)
            advantage = torch.where(
                valid, (rewards - group_mean) / (group_std + 1e-6), torch.zeros_like(rewards)
            )
            policy = -(advantage.unsqueeze(-1) * (candidate - mean.detach().unsqueeze(0))).mean()
            command = final_command(mean, baseline, residual=True)
            anchor = normalized_correction_loss(command, target)
            entropy = model.exit_entropy_loss(info) if cfg.learned_exit else raw.new_zeros(())
            loss = policy + anchor + cfg.entropy_weight * entropy
            optimizer.zero_grad(set_to_none=False)
            loss.backward()
            optimizer.step()
            steps += 1
            totals.extend(rewards.detach().reshape(-1).tolist())
            laterals.extend(parts["lateral"].detach().reshape(-1).tolist())
            efforts.extend(parts["effort"].detach().reshape(-1).tolist())
            risks.extend(parts["risk"].detach().reshape(-1).tolist())
            valid_count += int(valid.sum())
            sample_count += int(valid.numel())
    n = max(len(totals), 1)
    reward = VerifiableReward(
        seed=cfg.seed,
        config_hash=cfg_hash,
        source="v3:g_accel command target + frame-local physics baseline (no truth sidecar needed)",
        formula=REWARD_FORMULA,
        weights=dict(REWARD_WEIGHTS),
        samples=sample_count,
        rejected=sample_count - valid_count,
        lateral_gain=sum(laterals) / n,
        effort=sum(efforts) / n,
        risk=sum(risks) / n,
        total_mean=sum(totals) / n,
    )
    train, held, ratio, ece = _evaluate(model, fit, inner, cfg, device)
    details = {"reward": reward.as_dict(), "rlvr_samples_per_group": cfg.rlvr_samples}
    return StageReport(
        "rlvr",
        steps,
        train,
        held,
        ratio,
        ece,
        float(entropy.detach()),
        cfg.learned_exit,
        details,
    )


def _build_student(cfg: TeacherV2Config, device: torch.device) -> StudentLike:
    if _UpstreamStudent is not None and _UpstreamStudentConfig is not None:
        scfg = _UpstreamStudentConfig(seed=child_seed(cfg.seed, "student"))
        return _UpstreamStudent(scfg).to(device)
    return _LocalStudent(child_seed(cfg.seed, "student")).to(device)


def _stage_distill(
    teacher: LoopWM,
    cfg: TeacherV2Config,
    fit: list[Episode],
    inner: list[Episode],
    device: torch.device,
) -> StageReport:
    student = _build_student(cfg, device)
    optimizer = AdamW(student.parameters(), lr=cfg.student_lr)  # type: ignore[attr-defined]
    batches = _window_batches(fit, cfg)
    steps = 0
    teacher.eval()
    student.train()  # type: ignore[attr-defined]
    for _ in range(cfg.epochs):
        for batch in batches:
            window, cond, _target = (t.to(device) for t in batch[:3])
            with torch.no_grad():
                latent, _ = teacher.refine(window, cond)
                target_output = teacher.head_output(teacher.head_logits(latent))
            output = student.forward(window, cond)
            if output.shape[0] != window.shape[0] or output.shape[-1] != 3:
                raise AssertionError(f"student output has wrong shape {tuple(output.shape)}")
            last = output[:, -1]
            loss = F.mse_loss(last[:, :2], target_output[:, :2])
            loss = loss + F.mse_loss(last[:, 2:3], target_output[:, 2:3])
            optimizer.zero_grad(set_to_none=False)
            loss.backward()
            optimizer.step()
            steps += 1
    cfg.out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": "distill",
            "source": STUDENT_SOURCE,
            "seed": cfg.seed,
            "config_hash": config_hash(cfg),
            "state_dict": student.state_dict(),  # type: ignore[attr-defined]
        },
        cfg.out / "student.pt",
    )
    train, held, ratio, ece = _evaluate(student, fit, inner, cfg, device, student=True)
    details = {"student_source": STUDENT_SOURCE, "student_file": "student.pt"}
    return StageReport("distill", steps, train, held, ratio, ece, 0.0, False, details)


def run(cfg: TeacherV2Config) -> TeacherV2Metrics:
    """Execute every stage through ``cfg.stage`` and write one metrics.json."""
    seed_everything(cfg.seed)
    device = torch.device(cfg.device)
    fit, inner, held_ids = partition_episodes(
        load_episodes(cfg.data), seed=cfg.seed, n_held_out=cfg.n_inner_held_out
    )
    stages = STAGE_ORDER if cfg.stage == "all" else STAGE_ORDER[: STAGE_ORDER.index(cfg.stage) + 1]
    model = _build_teacher(cfg, device)
    digest = config_hash(cfg)
    reports: dict[str, StageReport] = {}
    reward: dict[str, Any] = {}
    for stage in stages:
        if stage == "privileged":
            report = _stage_privileged(model, cfg, fit, inner, device)
        elif stage == "dagger":
            report = _stage_dagger(model, cfg, fit, inner, held_ids, device)
        elif stage == "rlvr":
            report = _stage_rlvr(model, cfg, fit, inner, device, digest)
        elif stage == "distill":
            report = _stage_distill(model, cfg, fit, inner, device)
        else:
            report = _stage_sft(model, cfg, fit, inner, device)
        reports[stage] = report
        if "reward" in report.details:
            reward = report.details["reward"]
    last = reports[stages[-1]]
    # The student stage has no LoopWM entropy; report the last teacher stage's
    # value at the top level so --stage all still records the entropy term.
    entropy_enabled = any(report.entropy_enabled for report in reports.values())
    entropy = next(
        (report.entropy for report in reversed(list(reports.values())) if report.entropy_enabled),
        0.0,
    )
    metrics = TeacherV2Metrics(
        stage=last.stage,
        seed=cfg.seed,
        config_hash=digest,
        rmse_train=last.rmse_train,
        rmse_held_out=last.rmse_held_out,
        ratio=last.ratio,
        ece=last.ece,
        steps=last.steps,
        entropy=entropy,
        entropy_enabled=entropy_enabled,
        reward_provenance=reward,
        module_sources={
            "jepa": JEPA_SOURCE,
            "dagger": DAGGER_SOURCE,
            "privileged": PRIVILEGED_SOURCE,
            "student": STUDENT_SOURCE,
        },
        stages={name: report.as_dict() for name, report in reports.items()},
        stages_completed=tuple(stages),
    )
    write_metrics(cfg.out, metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="potok-train-teacher-v2", description=__doc__.splitlines()[0]
    )
    parser.add_argument(
        "--data", type=Path, required=True, help="directory of training validation files"
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="directory for metrics and the student"
    )
    parser.add_argument(
        "--stage",
        choices=(*STAGE_ORDER, "all"),
        default="all",
        help="run every stage through this one (default: all)",
    )
    parser.add_argument("--seed", type=int, choices=SEEDS, default=1)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--student-lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--max-iters", type=int, default=2)
    parser.add_argument("--learned-exit", action="store_true")
    parser.add_argument("--exit-gate-threshold", type=float, default=0.5)
    parser.add_argument("--exit-entropy-alpha", type=float, default=0.01)
    parser.add_argument("--spectral-residual", action="store_true")
    parser.add_argument("--spectral-bound", type=float, default=1.0)
    parser.add_argument("--jepa-weight", type=float, default=0.05)
    parser.add_argument("--entropy-weight", type=float, default=1.0)
    parser.add_argument("--dagger-rounds", type=int, default=1)
    parser.add_argument("--n-inner-held-out", type=int, default=2)
    parser.add_argument("--rlvr-samples", type=int, default=4)
    parser.add_argument("--rlvr-temperature", type=float, default=0.05)
    parser.add_argument("--truth-sidecar", type=Path, default=None)
    parser.add_argument("--calib-tol", type=float, default=0.15)
    return parser


def main(argv: list[str] | None = None) -> None:
    cfg = TeacherV2Config.from_args(build_parser().parse_args(argv))
    metrics = run(cfg)
    print(
        json.dumps(
            {
                "stage": metrics.stage,
                "seed": metrics.seed,
                "ratio": metrics.ratio,
                "ece": metrics.ece,
                "entropy": metrics.entropy,
                "entropy_enabled": metrics.entropy_enabled,
                "stages_completed": metrics.stages_completed,
                "out": str(cfg.out / "metrics.json"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
