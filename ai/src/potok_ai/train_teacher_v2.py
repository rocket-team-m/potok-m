# SPDX-License-Identifier: GPL-3.0-or-later
"""Staged teacher-v2 trainer (ENGINE01).

One CLI, one typed config and one ``metrics.json`` for the staged teacher
pipeline:

    sft -> privileged (LoopWM + JEPA v2) -> dagger -> distill

On-policy privileged-teacher distillation (DAgger/RMA-style) is the default;
RLVR is NOT (RES-RLVR: direct GRPO collapses on this low-dimensional continuous
controller). ``--rlvr`` or ``--stage rlvr`` opts into a final checkpoint-selection
stage: the deterministic verifier scores the teacher snapshots saved after each
teacher stage, the best one is restored before distillation, and no reward
gradient is ever taken. The reward formula, weights, hard gate and provenance
are unchanged; only its role moves from policy update to model selection.

Teacher stages compose the classical physics baseline with a bounded,
zero-initialised learned correction (SHIELD02 ``ResidualPolicy``); the raw
pre-SHIELD02 teacher stays available with ``--no-residual-policy``.

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
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW, Optimizer

from .braking_teacher import BrakingConfig, BrakingTeacher, TrueState
from .calibration import confidence_target, correctness, expected_calibration_error, sigmoid
from .conf_integration import DISJOINT_ECE_SOURCE, IN_SAMPLE_ECE_SOURCE, split_confidence_score
from .curriculum import STAGES as CURRICULUM_STAGES
from .curriculum import AdversarialCurriculum, refuse_eval_episode, refuse_eval_episodes
from .data import (
    SMART_COND_FEATURES,
    SMART_FEATURES,
    Episode,
    SmartWindowDataset,
    load_episodes,
)
from .loopwm import LoopWM, LoopWMConfig
from .objective import IMPACT_FORMULA, ImpactObjective
from .optim import K3Config, MuonPlus
from .physics import final_command, physics_baseline
from .residual_policy import (
    LEGACY_BOUND_MPS2,
    ResidualConfig,
    ResidualDecision,
    ResidualPolicy,
    assert_command_bound_covers,
)
from .student import CommandEnvelope
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
    from .dagger import RolloutState, on_policy_rollout, sample_sha256
    from .dagger import build_dagger_dataset as _upstream_build_dagger

    DAGGER_SOURCE = "dagger"
except (ImportError, SyntaxError):  # pragma: no cover - depends on sibling task
    _upstream_build_dagger = None

    def on_policy_rollout(**_kwargs):  # type: ignore[no-redef]
        raise RuntimeError(
            "on-policy DAgger needs potok_ai.dagger; the local fallback cannot roll the student"
        )

    def sample_sha256(*_args):  # type: ignore[no-redef]
        raise RuntimeError("on-policy DAgger needs potok_ai.dagger for SHA provenance")

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


STAGE_ORDER = ("sft", "privileged", "dagger", "distill")
RLVR_STAGE = "rlvr"
SEEDS = (1, 2, 3)
# POTOKTRJ v3 slot boundary, pinned to sim/core/src/trajectory.rs:93-120.
# 0..21 are privileged truth, 22..25 are the reference optimizer's labels.
# Local constants keep the trainer loud even when the sibling builder is absent.
TRUTH_PRIV_END = 22
TRUTH_LABEL_START = 22
TRUTH_V3_STRIDE = 26
EVAL_SCENARIO_IDS = frozenset({16, 17, 18, 19})
JEPA_HORIZONS = (3, 6, 10)
REWARD_WEIGHTS = {"lateral": 2.0, "effort": 0.5, "risk": 1.0}
SOFT_LIMIT_FRACTION = 0.8
REWARD_FORMULA = (
    "R = w_lat*(mean|b-u| - mean|b+t-u|) - w_eff*mean|u| - w_risk*soft_violations; "
    "hard gate: finite u and |u| <= g_max rejects the candidate from the group baseline"
)
# Stage-scoped adversarial context; only train=True window builders read it.
_ADV_CONTEXT: ContextVar[tuple[AdversarialCurriculum, str, dict[str, int]] | None] = ContextVar(
    "potok_adversarial", default=None
)


def adversarial_budget(stage: str, adv: AdversarialCurriculum) -> float:
    """Map a pipeline stage to the adversary's cap-bounded sampling budget.

    sft -> calm (clean), privileged -> gust, dagger -> shear, distill -> sensor_fault.
    RLVR and other non-training stages get no budget.
    """
    if stage in CURRICULUM_STAGES:
        return adv.intensity_for_stage(stage)
    if stage in STAGE_ORDER:
        return adv.intensity(STAGE_ORDER.index(stage))
    return 0.0


def child_seed(seed: int, name: str) -> int:
    """Named stream, stable across processes and versions."""
    digest = hashlib.sha256(f"{int(seed)}:engine01:{name}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def k3_config(cfg: TeacherV2Config) -> K3Config:
    """Typed K3 config from the staged CLI fields; defaults stay off."""
    return K3Config(
        num_heads=cfg.k3_per_head,
        qk_clip=cfg.k3_qk_clip,
        qk_clip_tau=cfg.k3_qk_clip_tau,
    )


@dataclass(frozen=True)
class TeacherV2Config:
    """All knobs of the staged run; every invalid value fails loudly."""

    data: Path
    out: Path
    stage: str = "all"
    adversarial: bool = False  # opt-in: bounded adversarial windows in training stages
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
    residual_policy: bool = True  # SHIELD02: bounded superposition in teacher stages
    # AIOBJ: the privileged impact objective is the default when truth is available.
    # --legacy-objective restores the supervised-only privileged stage.
    legacy_objective: bool = False
    # BRK01: privileged radial-braking reference is the default teacher target.
    # --no-braking-teacher restores the recorded reference-optimizer labels.
    braking_teacher: bool = True
    # Explicit opt-in to the pre-v3 +/-1 m/s^2 student head (AITRAINIDEAL B1).
    # Default off: the student must be able to represent the v3 target.
    legacy_student_bound: bool = False
    jepa_weight: float = 0.05
    entropy_weight: float = 1.0
    dagger_rounds: int = 1
    n_inner_held_out: int = 2
    rlvr: bool = False  # opt-in checkpoint selection, never a policy gradient
    rlvr_samples: int = 4
    rlvr_temperature: float = 0.05
    truth_sidecar: Path | None = None
    calib_tol: float = 0.15
    opt: str = "adamw"
    k3_per_head: int = 0
    k3_qk_clip: bool = False
    k3_qk_clip_tau: float = 100.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", Path(self.data))
        object.__setattr__(self, "out", Path(self.out))
        if self.truth_sidecar is not None:
            object.__setattr__(self, "truth_sidecar", Path(self.truth_sidecar))
        if self.stage not in (*STAGE_ORDER, RLVR_STAGE, "all"):
            raise ValueError(
                f"unknown stage {self.stage!r}; expected one of all, "
                f"{', '.join((*STAGE_ORDER, RLVR_STAGE))}"
            )
        if self.opt not in ("adamw", "muon"):
            raise ValueError(f"unknown optimizer {self.opt!r}; expected adamw or muon")
        k3_config(self)  # the frozen K3Config validates its own fields
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


def _build_optimizer(
    parameters: list[tuple[str, torch.Tensor]], cfg: TeacherV2Config, lr: float
) -> Optimizer:
    """AdamW keeps the historical gates; Muon+ is an explicit opt-in."""
    if cfg.opt == "adamw":
        return AdamW([p for _name, p in parameters], lr=lr)
    return MuonPlus(parameters, lr=lr, k3=k3_config(cfg))


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
    ece_source: str = IN_SAMPLE_ECE_SOURCE

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
    ece_source: str = DISJOINT_ECE_SOURCE
    ece_held_out: float = 0.0
    conformal_coverage: float = 0.0
    entropy: float = 0.0
    entropy_enabled: bool = False
    reward_provenance: dict[str, Any] = field(default_factory=dict)
    module_sources: dict[str, str] = field(default_factory=dict)
    residual_policy: dict[str, Any] = field(default_factory=dict)
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
    episodes: list[Episode], cfg: TeacherV2Config, *, train: bool = False
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int | None]]]:
    """Materialise tiny CPU windows once; return (window, cond, target, episode ids).

    Only ``train=True`` reads the stage-scoped adversarial context. Eval and
    verifier windows stay untouched, and an eval cell is refused before a single
    augmentation draw.
    """
    ctx = _ADV_CONTEXT.get() if train else None
    samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, int | None]] = []
    for episode in episodes:
        dataset = SmartWindowDataset([episode], window=cfg.window, limit=cfg.limit)
        for window_index, (window, cond, target) in enumerate(dataset.samples):
            if ctx is not None:
                adv, stage, stats = ctx
                refuse_eval_episode(episode)
                stats["offered"] += 1
                result = adv.perturb(
                    window,
                    tag=f"{stage}:{episode.index}:{window_index}",
                    intensity=adversarial_budget(stage, adv),
                )
                if result.won:
                    stats["won"] += 1
                window = result.window
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
    cfg: TeacherV2Config,
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
    command = _command(model, window, correction, residual=cfg.residual_policy and not student)
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
                if window.shape[-1] != SMART_FEATURES or cond.shape[-1] != SMART_COND_FEATURES:
                    raise AssertionError(
                        "eval forward must see only the 14 sensor + 6 conditioning columns; "
                        "privileged truth and labels are teacher-only"
                    )
                command, logit, error = _predict(
                    model,
                    window.to(device),
                    cond.to(device),
                    target.to(device),
                    cfg=cfg,
                    student=student,
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


def _confidence_hook(
    model: nn.Module,
    inner: list[Episode],
    cfg: TeacherV2Config,
    device: torch.device,
    *,
    student: bool,
):
    """Disjoint, seed-keyed split-conformal score on inner-held-out predictions.

    ``split_confidence_score`` fits on one half and scores the other, writes the
    edge-facing ``conformal.json`` with its frozen step map, and records the
    effective alpha when the fit half is too small for the requested level.
    Nothing here tunes a gate or claims a flight result: this is a measurement
    of the network whose metrics are being reported.
    """
    logits: list[float] = []
    errors: list[float] = []
    with torch.no_grad():
        for batch in _window_batches(inner, cfg):
            window, cond, target = (t.to(device) for t in batch[:3])
            _command, logit, error = _predict(model, window, cond, target, cfg=cfg, student=student)
            logits.extend(logit.tolist())
            errors.extend(error.tolist())
    if not logits or not np.isfinite(logits).all() or not np.isfinite(errors).all():
        raise ValueError("inner-held-out confidence predictions must be non-empty and finite")
    return split_confidence_score(
        np.asarray(logits, dtype=np.float64),
        np.asarray(errors, dtype=np.float64),
        cfg.out,
        seed=cfg.seed,
        tol=cfg.calib_tol,
    )


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
    model = LoopWM(_teacher_config(cfg)).to(device)
    if cfg.residual_policy:
        model.add_module("residual_policy", ResidualPolicy(ResidualConfig()).to(device))
    return model


def _residual_decision(
    model: nn.Module,
    window: torch.Tensor,
    correction: torch.Tensor,
    *,
    residual: bool,
) -> ResidualDecision | None:
    """Apply the gated superposition, or return None for the legacy path."""
    if not residual:
        return None
    policy = getattr(model, "residual_policy", None)
    if not isinstance(policy, ResidualPolicy):
        raise TypeError("residual_policy is enabled but the model has no ResidualPolicy module")
    return policy(physics_baseline(window[:, -1]), correction)


def _command(
    model: nn.Module,
    window: torch.Tensor,
    correction: torch.Tensor,
    *,
    residual: bool,
) -> torch.Tensor:
    """Classical baseline plus correction; the unchanged teacher path when disabled."""
    decision = _residual_decision(model, window, correction, residual=residual)
    if decision is None:
        return final_command(correction, physics_baseline(window[:, -1]), residual=True)
    return decision.command


def _supervised_loss(
    model: LoopWM,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int | None]],
    cfg: TeacherV2Config,
    *,
    jepa: JepaObjective | None = None,
    privileged_head: nn.Linear | None = None,
    privileged_target: torch.Tensor | None = None,
    impact: ImpactObjective | None = None,
    privileged_truth: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Command + confidence + optional JEPA/entropy/impact/privileged terms."""
    window, cond, target, _ = batch
    latent, info = model.refine(window, cond)
    raw = model.head_logits(latent)
    output = model.head_output(raw)
    command = _command(model, window, output[:, :2], residual=cfg.residual_policy)
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
    impact_terms = None
    if impact is not None:
        if privileged_truth is None or privileged_truth.shape[0] != window.shape[0]:
            raise ValueError("impact objective needs one privileged truth row per window")
        impact_terms = impact.loss(command, window, privileged_truth)
        total = total + impact_terms.loss
    return {
        "loss": total,
        "main": float(main.detach()),
        "conf": float(conf.detach()),
        "entropy": float(entropy.detach()),
        "impact": float(impact_terms.loss.detach()) if impact_terms is not None else 0.0,
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

    def __init__(self, seed: int, bound: float = CommandEnvelope().accel) -> None:
        super().__init__()
        self.bound = bound
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
        corrections = torch.tanh(raw[..., :2]) * self.bound
        return torch.cat([corrections, torch.sigmoid(raw[..., 2:3])], dim=-1)


@dataclass(frozen=True)
class _LocalProvenance:
    sample_id: str
    seed: int
    scenario_id: int | None
    split: str
    expert: str
    label_source: str
    rollout_sha256: str = ""


@dataclass(eq=False)
class _LocalDaggerSample:
    observation: np.ndarray
    correction: np.ndarray
    authoritative: np.ndarray | None
    provenance: _LocalProvenance
    action: np.ndarray | None = None
    drift: float | None = None


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
    optimizer = _build_optimizer(list(model.named_parameters()), cfg, cfg.lr)
    batches = _window_batches(fit, cfg, train=True)
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
    if cfg.truth_sidecar is None:
        return None
    if _UpstreamPrivilegedBuilder is None:
        raise RuntimeError(
            "truth_sidecar is configured but the POTOKTRJ v3 privileged builder is "
            "unavailable; a privileged teacher cannot run without reading the truth "
            "sidecar (no silent zero fill)"
        )
    return _UpstreamPrivilegedBuilder(cfg.truth_sidecar, seed=child_seed(cfg.seed, "privileged"))


def _v3_truth_parts(snapshot: np.ndarray, *, where: str) -> tuple[np.ndarray, np.ndarray]:
    """Split a 26-slot v3 snapshot at the privilege boundary, loudly.

    Slots 0..21 are the only fields a teacher may read; 22..25 are labels and
    belong to the loss only. The two halves are disjoint by construction.
    """
    arr = np.asarray(snapshot, dtype=np.float32)
    if arr.shape[-1] != TRUTH_V3_STRIDE:
        raise ValueError(
            f"{where}: a v3 truth snapshot needs {TRUTH_V3_STRIDE} slots, got {arr.shape}; "
            "regenerate with write_trajectory_v3"
        )
    privileged = arr[..., :TRUTH_PRIV_END]
    labels = arr[..., TRUTH_LABEL_START:]
    if (
        privileged.shape[-1] != TRUTH_PRIV_END
        or labels.shape[-1] != TRUTH_V3_STRIDE - TRUTH_PRIV_END
    ):
        raise AssertionError(f"{where}: privilege/label slots are not disjoint")
    return privileged, labels


def _privileged_snapshot(builder: Any, episode_index: int, tick: int, *, where: str) -> np.ndarray:
    """Assemble a v3 snapshot from the sidecar; zeroed truth is a loud failure.

    The privileged half comes only from the sidecar's truth slots, never from
    episode targets. IDEAL_DATA_RESEARCH R3: a teacher whose truth channel is
    zero is a contract violation, not a result.
    """
    record = next((r for r in builder.records if r.episode_index == int(episode_index)), None)
    if record is None:
        raise ValueError(f"{where}: episode {episode_index} is not in the truth sidecar")
    if not 0 <= tick < record.privileged.shape[0]:
        raise ValueError(
            f"{where}: tick {tick} is outside the sidecar's {record.privileged.shape[0]} frames"
        )
    privileged = np.asarray(record.privileged[tick], dtype=np.float32)
    labels = np.asarray(record.labels[tick], dtype=np.float32)
    snapshot = np.concatenate([privileged, labels])
    _v3_truth_parts(snapshot, where=where)  # pin the 22/26 boundary
    if not np.isfinite(privileged).all():
        raise ValueError(f"{where}: non-finite privileged truth at episode {episode_index}")
    if not np.any(privileged):
        raise AssertionError(
            f"{where}: privileged slots 0..21 are all zero at episode {episode_index} "
            f"tick {tick}; a privileged teacher cannot run on a zeroed truth channel"
        )
    return snapshot


def _privileged_rows(
    builder: Any, ids: list[int | None], cfg: TeacherV2Config, *, where: str
) -> torch.Tensor:
    """One 22-slot truth row per window, keyed by episode and window offset.

    ``_window_batches`` emits each episode's windows in order starting at tick
    0, so the n-th window of an episode ends at tick ``window - 1 + n``.
    """
    counters: dict[int, int] = {}
    rows: list[np.ndarray] = []
    for episode_index in ids:
        if episode_index is None:
            raise ValueError(f"{where}: the impact objective needs a numeric episode index")
        index = int(episode_index)
        offset = counters.get(index, 0)
        counters[index] = offset + 1
        tick = cfg.window - 1 + offset
        rows.append(_privileged_snapshot(builder, index, tick, where=where)[:TRUTH_PRIV_END])
    return torch.as_tensor(np.stack(rows), dtype=torch.float32)


def _window_ticks(
    ids: list[int | None], cfg: TeacherV2Config, *, where: str
) -> np.ndarray:
    """Flight ticks of the same window rows :func:`_privileged_rows` returns."""
    counters: dict[int, int] = {}
    ticks: list[int] = []
    for episode_index in ids:
        if episode_index is None:
            raise ValueError(f"{where}: a braking target needs a numeric episode index")
        index = int(episode_index)
        offset = counters.get(index, 0)
        counters[index] = offset + 1
        ticks.append(cfg.window - 1 + offset)
    return np.asarray(ticks, dtype=np.float64)


def _braking_target_rows(
    builder: Any,
    ids: list[int | None],
    cfg: TeacherV2Config,
    teacher: BrakingTeacher,
    *,
    where: str,
) -> torch.Tensor:
    """True-state radial-braking command for one batch of window rows.

    The authority gate comes from the flight tick (boost is forbidden), never
    from the reference label slots.
    """
    privileged = _privileged_rows(builder, ids, cfg, where=where)
    ticks = _window_ticks(ids, cfg, where=where)
    state = TrueState(
        position=privileged[:, :3].numpy(),
        velocity=privileged[:, 3:6].numpy(),
        time_s=ticks * BrakingConfig().frame_dt_s,
    )
    decision = teacher.demand(state)
    if not np.isfinite(decision.command).all():
        raise ValueError(f"{where}: braking reference produced a non-finite command")
    return torch.as_tensor(decision.command, dtype=torch.float32)


def _braking_details(cfg: TeacherV2Config, brake: BrakingTeacher | None) -> dict[str, Any]:
    """Counted provenance of the braking reference for metrics.json."""
    config = BrakingConfig()
    return {
        "enabled": brake is not None,
        "source": "radial_braking_reference" if brake is not None else "legacy_or_no_truth",
        "reason": (
            None
            if brake is not None
            else ("legacy_braking_teacher" if not cfg.braking_teacher else "no_truth_sidecar")
        ),
        "reason_counts": brake.reason_counts() if brake is not None else {},
        "cap_mps2": config.cap_mps2,
        "horizon_s": config.horizon_s,
        "stop_distance_m": config.stop_distance_m,
        "boost_end_s": config.boost_end_s,
    }


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
    parameters = list(model.named_parameters()) + [
        (f"jepa.{name}", p)
        for name, p in jepa.named_parameters()  # type: ignore[attr-defined]
    ]
    if head is not None:
        parameters += [(f"privileged_head.{name}", p) for name, p in head.named_parameters()]
    impact = None if cfg.legacy_objective else ImpactObjective()
    if impact is not None and not getattr(builder, "records", None):
        impact = None  # no truth sidecar records: the privilege cannot be read
    objective_source = "impact" if impact is not None else "legacy"
    optimizer = _build_optimizer(parameters, cfg, cfg.lr)
    batches = _window_batches(fit, cfg, train=True)
    brake = (
        BrakingTeacher()
        if cfg.braking_teacher and getattr(builder, "records", None)
        else None
    )
    if brake is not None:
        batches = [
            (window, cond, _braking_target_rows(builder, ids, cfg, brake, where="privileged"), ids)
            for window, cond, _recorded, ids in batches
        ]
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
                if not torch.isfinite(z_priv).all() or not bool(z_priv.abs().sum() > 0):
                    raise AssertionError(
                        "privileged latent is non-finite or all zero: the teacher never saw truth"
                    )
                privileged_terms += 1
            privileged_truth = (
                _privileged_rows(builder, batch[3], cfg, where="impact objective").to(device)
                if impact is not None
                else None
            )
            result = _supervised_loss(
                model,
                (window, cond, target, batch[3]),
                cfg,
                jepa=jepa,
                privileged_head=head,
                privileged_target=z_priv,
                impact=impact,
                privileged_truth=privileged_truth,
            )
            optimizer.zero_grad(set_to_none=False)
            result["loss"].backward()
            optimizer.step()
            jepa.update_teacher(model.encoder)
            entropy = result["entropy"]
            steps += 1
    train, held, ratio, ece = _evaluate(model, fit, inner, cfg, device)
    if cfg.truth_sidecar is not None and privileged_terms == 0:
        raise AssertionError(
            "truth_sidecar is configured but the privileged stage consumed 0 truth rows; "
            "the privileged teacher did not run (no silent label-only stage)"
        )
    details = {
        "jepa_source": JEPA_SOURCE,
        "privileged_source": "privileged_teacher" if builder is not None else "disabled",
        "truth_sidecar": str(cfg.truth_sidecar) if cfg.truth_sidecar is not None else None,
        "privileged_batches": privileged_terms,
        "braking": _braking_details(cfg, brake),
        "objective": {
            "source": objective_source,
            "legacy": cfg.legacy_objective,
            "formula": IMPACT_FORMULA if impact is not None else None,
            "reason": (
                None
                if impact is not None
                else ("legacy_objective" if cfg.legacy_objective else "no_truth_sidecar_records")
            ),
        },
        "eval_label_blind": True,
    }
    return StageReport(
        "privileged", steps, train, held, ratio, ece, entropy, cfg.learned_exit, details
    )


def _expert_label(privileged: np.ndarray) -> np.ndarray:
    """The v3 reference optimizer's answer lives in label slots 22..25 only."""
    labels = _v3_truth_parts(privileged, where="expert label")[1]
    return np.asarray([labels[2], -labels[3]], dtype=np.float32)


def _braking_command(teacher: BrakingTeacher, snapshot: np.ndarray) -> np.ndarray:
    """Radial demand for one visited snapshot; interlock from recorded health.

    The label slots are read only as a boolean health bit (the reference emits
    exactly zero during boost); their numeric values never enter the demand.
    """
    privileged, labels = _v3_truth_parts(snapshot, where="braking label")
    state = TrueState.from_privileged(privileged)
    recorded = np.asarray(labels, dtype=np.float64)
    healthy = np.any(recorded != 0.0, axis=-1 if recorded.ndim > 1 else 0)
    allowed = np.atleast_1d(np.asarray(healthy, dtype=bool))
    command = teacher.demand(state, ai_allowed=allowed).command.astype(np.float32)
    return command[0] if np.asarray(snapshot).ndim == 1 else command


def _student_rollout_action(student: nn.Module, device: torch.device):
    """One student command from sensor history only; no privileged field is read."""
    student.eval()

    def act(observation: np.ndarray, _rng: random.Random) -> np.ndarray:
        window = torch.as_tensor(observation[:, :14], dtype=torch.float32).unsqueeze(0).to(device)
        cond = torch.as_tensor(observation[-1, 14:20], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            if hasattr(student, "step"):
                correction, _confidence = student.step(window[0, -1], cond[0])
            else:
                output = student.forward(window, cond)
                correction = output[0, -1, :2]
        action = np.asarray(correction.detach().cpu(), dtype=np.float32).reshape(-1)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("student rollout action must be two finite commands")
        return action

    return act


def _student_world_step(frame: np.ndarray, action: np.ndarray, rng: random.Random) -> np.ndarray:
    """Advance the student-visible sensor frame from the student's own command.

    ponytail: the Rust simulator is not importable here, so only the onboard
    features drift; upgrade path is to replay its real dynamics step on
    (state, command) once a Python binding exists. Physics constants and the
    baseline+correction composition come from physics.py unchanged.
    """
    tensor = torch.as_tensor(frame, dtype=torch.float32)
    baseline = physics_baseline(tensor.unsqueeze(0))[0].numpy()
    command = final_command(torch.as_tensor(action), torch.as_tensor(baseline)).numpy()
    error = np.asarray(command - baseline, dtype=np.float32)
    nxt = np.asarray(frame, dtype=np.float32).copy()
    nxt[8] = np.clip(frame[8] + 0.4 * error[1] + 0.02 * rng.uniform(-1.0, 1.0), -30.0, 30.0)
    nxt[9] = np.clip(frame[9] + 0.4 * error[0] + 0.02 * rng.uniform(-1.0, 1.0), -30.0, 30.0)
    nxt[11] = np.clip(frame[11] + 0.2 * error[0], -10.0, 10.0)
    return nxt


def _dagger_rollout(
    cfg: TeacherV2Config,
    builder: Any | None = None,
    *,
    student: nn.Module | None = None,
    device: torch.device | None = None,
) -> Any:
    """Student-in-the-loop rollout; the teacher labels the states the student visits."""
    device = device or torch.device(cfg.device)
    policy = student if student is not None else _build_student(cfg, device)
    actor = _student_rollout_action(policy, device)

    def snapshot(episode: Episode, step: int, _rng: random.Random) -> np.ndarray:
        tick = cfg.window - 1 + step
        where = f"dagger episode {episode.index} tick {tick}"
        if builder is not None:
            if episode.index is None:
                raise ValueError(f"{where}: episode has no index; cannot locate its truth row")
            return _privileged_snapshot(builder, episode.index, tick, where=where)
        # No sidecar: the recorded command is a documented CPU fixture. The
        # zeroed privileged half is reported as such, never sold as a teacher.
        target = episode.command_target()
        truth4 = np.zeros(4, dtype=np.float32)
        if episode.targets is not None:
            truth4[:2] = episode.targets[tick, :2]
        if episode.g_accel is not None:
            truth4[2:] = episode.g_accel[tick]
        else:
            truth4[2:] = (target[tick, 0], -target[tick, 1])
        return np.concatenate([np.zeros(TRUTH_PRIV_END, dtype=np.float32), truth4])

    def authoritative(episode: Episode, tick: int, _frame: np.ndarray) -> np.ndarray:
        return np.asarray(episode.command_target()[tick], dtype=np.float32)

    def reset(episode: Episode) -> None:
        if hasattr(policy, "reset_state"):
            policy.reset_state()  # type: ignore[attr-defined]

    return on_policy_rollout(
        window=cfg.window,
        student_action=actor,
        world_step=_student_world_step,
        privileged_snapshot=snapshot,
        authoritative_target=authoritative,
        on_episode_start=reset,
    )


def _dagger_info(cfg: TeacherV2Config, dataset: Any, builder: Any | None) -> dict[str, Any]:
    """Provenance of the rollout mechanism, recorded in the DAgger stage details."""
    samples = dataset.samples
    induced = sum(1 for s in samples if s.drift is not None and s.drift > 1e-6)
    rollout_digest = hashlib.sha256()
    for sample in samples:
        rollout_digest.update(sample.provenance.rollout_sha256.encode())
    return {
        "rollout_source": f"on_policy_student:{STUDENT_SOURCE}",
        "privileged_source": (
            "v3_truth_sidecar" if builder is not None else "recorded_command_fixture"
        ),
        "privileged_fields_present": builder is not None,
        "student_induced_rows": induced,
        "total_rows": len(samples),
        "rollout_sha256": rollout_digest.hexdigest(),
    }


def _build_dagger_dataset(
    cfg: TeacherV2Config, episodes: list[Episode], held_ids: frozenset[int]
) -> Any:
    builder = _build_privileged(cfg)
    student = _build_student(cfg, torch.device(cfg.device))
    rollout = _dagger_rollout(cfg, builder, student=student)
    brake = (
        BrakingTeacher()
        if cfg.braking_teacher and getattr(builder, "records", None)
        else None
    )
    braking_targets: list[np.ndarray] = []

    def recorded_label(snapshot: np.ndarray) -> np.ndarray:
        # The historical ``correction`` field stays the recorded optimizer label
        # (its provenance contract is unchanged). The braking reference is a
        # parallel target stream used by the training stage below.
        if brake is not None:
            braking_targets.append(_braking_command(brake, snapshot))
        return _expert_label(snapshot)

    if _upstream_build_dagger is not None:
        dataset = _upstream_build_dagger(
            seed=cfg.seed,
            episodes=episodes,
            rollout=rollout,
            teacher=recorded_label,
            rounds=cfg.dagger_rounds,
            rollouts_per_round=1,
        )
    else:
        # Local fallback: identical shapes/provenance, no extra dependency.
        samples: list[_LocalDaggerSample] = []
        for episode in episodes:
            sid = int(episode.scenario_id)
            rng = random.Random(f"{cfg.seed}:dagger:{episode.index}")
            for step, state in enumerate(rollout(episode, rng)):
                correction = recorded_label(state.privileged)
                if not np.isfinite(correction).all():
                    raise ValueError(f"non-finite expert label at episode {sid} step {step}")
                sample_id = f"{cfg.seed}:dagger:{sid}:{step}"
                sha = sample_sha256(
                    sample_id,
                    state.observation,
                    state.privileged,
                    state.action,
                    state.authoritative,
                )
                drift = (
                    None
                    if state.recorded_frame is None
                    else float(np.linalg.norm(state.observation[-1, :14] - state.recorded_frame))
                )
                samples.append(
                    _LocalDaggerSample(
                        observation=state.observation,
                        correction=correction,
                        authoritative=state.authoritative,
                        provenance=_LocalProvenance(
                            sample_id=sample_id,
                            seed=cfg.seed,
                            scenario_id=sid,
                            split="held_out" if sid in held_ids else "fit",
                            expert="reference_optimizer_v3",
                            label_source="expert",
                            rollout_sha256=sha,
                        ),
                        action=state.action,
                        drift=drift,
                    )
                )
        digest = hashlib.sha256()
        for sample in samples:
            digest.update(sample.provenance.sample_id.encode())
            digest.update(np.asarray(sample.observation, dtype=np.float32).tobytes())
            digest.update(sample.correction.tobytes())
        dataset = _LocalDaggerDataset(samples, digest.hexdigest())
    if brake is not None:
        if len(braking_targets) != len(dataset.samples):
            raise AssertionError("every DAgger sample needs one braking target")
        dataset.braking_targets = {
            str(sample.provenance.sample_id): target
            for sample, target in zip(dataset.samples, braking_targets)
        }
    dataset.dagger_info = _dagger_info(cfg, dataset, builder)
    dataset.dagger_info["braking"] = _braking_details(cfg, brake)
    return dataset


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
    braking = getattr(dataset, "braking_targets", None)
    optimizer = _build_optimizer(list(model.named_parameters()), cfg, cfg.lr)
    steps, entropy = 0, 0.0
    model.train()
    for _ in range(cfg.epochs):
        for start in range(0, len(rows), cfg.batch):
            chunk = observations[start : start + cfg.batch]
            window = chunk[:, :, :14].to(device)
            cond = chunk[:, 0, 14:].to(device)
            target = corrections[start : start + cfg.batch].to(device)
            if braking is not None:
                target = torch.as_tensor(
                    np.stack([braking[str(s.provenance.sample_id)] for s in rows[start : start + cfg.batch]]),
                    dtype=torch.float32,
                ).to(device)
            result = _supervised_loss(model, (window, cond, target, [None] * len(target)), cfg)
            optimizer.zero_grad(set_to_none=False)
            result["loss"].backward()
            optimizer.step()
            entropy = result["entropy"]
            steps += 1
    train, held, ratio, ece = _evaluate(model, fit, inner, cfg, device)
    details_braking = dataset.dagger_info.get("braking", _braking_details(cfg, None))
    details = {
        "dagger_source": DAGGER_SOURCE,
        **getattr(dataset, "dagger_info", {}),
        "dataset_fingerprint": dataset.fingerprint(),
        "fit_samples": len(rows),
        "held_out_samples": len(held_rows),
        "label_source": (
            "reference_optimizer_v3:truth_sidecar_slots22..25"
            if cfg.truth_sidecar is not None
            else "reference_optimizer_v3:recorded_target(no_truth_sidecar)"
        ),
        "truth_sidecar": str(cfg.truth_sidecar) if cfg.truth_sidecar is not None else None,
        "privileged_fields_zeroed": cfg.truth_sidecar is None,
        "braking": details_braking,
        "training_target": "radial_braking_reference" if braking is not None else "recorded_label",
        "eval_label_blind": True,
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


def _verifier_score(
    model: LoopWM,
    cfg: TeacherV2Config,
    batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int | None]]],
    cfg_hash: str,
) -> tuple[float, VerifiableReward]:
    """Score one teacher snapshot with the verifier; no graph, no optimizer.

    The reward formula, weights, hard gate and provenance are the historical
    ones (REWARD_FORMULA / REWARD_WEIGHTS / verifiable_reward). Only the use of
    the result changes: it ranks snapshots, it is never back-propagated.
    """
    totals: list[float] = []
    laterals: list[float] = []
    efforts: list[float] = []
    risks: list[float] = []
    valid_count, sample_count = 0, 0
    generator = torch.Generator().manual_seed(child_seed(cfg.seed, RLVR_STAGE))
    model.eval()
    with torch.no_grad():
        for window, cond, target, _ in batches:
            latent, _ = model.refine(window, cond)
            raw = model.head_logits(latent)
            mean = model.head_output(raw)[:, :2]
            baseline = physics_baseline(window[:, -1])
            noise = torch.randn(
                (cfg.rlvr_samples, *mean.shape), generator=generator, dtype=mean.dtype
            )
            candidate = mean.unsqueeze(0) + cfg.rlvr_temperature * noise
            rewards, parts = verifiable_reward(
                candidate, baseline.unsqueeze(0), target.unsqueeze(0), float(model.cfg.g_max)
            )
            valid = parts["valid"]
            counts = valid.sum(dim=0).clamp_min(1)
            if not bool((counts > 0).all()):
                raise ValueError(
                    "every RLVR group needs at least one candidate that passes the gate"
                )
            totals.extend(rewards.reshape(-1).tolist())
            laterals.extend(parts["lateral"].reshape(-1).tolist())
            efforts.extend(parts["effort"].reshape(-1).tolist())
            risks.extend(parts["risk"].reshape(-1).tolist())
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
    return reward.total_mean, reward


def _stage_rlvr(
    model: LoopWM,
    cfg: TeacherV2Config,
    fit: list[Episode],
    inner: list[Episode],
    device: torch.device,
    cfg_hash: str,
    candidates: list[tuple[str, dict[str, torch.Tensor]]] | None = None,
) -> StageReport:
    """Select the best teacher snapshot by verifier reward; never update weights from it.

    RES-RLVR: a GRPO policy gradient collapses on this continuous controller, so
    the verifier is a selection/falsification instrument only. Candidates are the
    snapshots saved after each teacher stage. The winner is restored into ``model``
    before distillation. No optimizer is built, no loss is back-propagated.
    """
    if not candidates:
        state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        candidates = [(RLVR_STAGE, state)]
    batches = _window_batches(fit, cfg)
    scored: list[tuple[float, str, dict[str, torch.Tensor], VerifiableReward]] = []
    for name, state in candidates:
        model.load_state_dict(state)
        score, reward = _verifier_score(model, cfg, batches, cfg_hash)
        scored.append((score, name, state, reward))
    if not scored:
        raise ValueError("RLVR checkpoint selection needs at least one teacher snapshot")
    # max() keeps the first entry on ties -> deterministic across seeds/processes.
    best_score, best_name, best_state, best_reward = max(scored, key=lambda row: row[0])
    model.load_state_dict(best_state)
    cfg.out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": RLVR_STAGE,
            "role": "checkpoint_selection",
            "source": "v3 verifier (no policy gradient)",
            "seed": cfg.seed,
            "config_hash": cfg_hash,
            "selected": best_name,
            "score": best_score,
            "state_dict": best_state,
        },
        cfg.out / "teacher_selected.pt",
    )
    train, held, ratio, ece = _evaluate(model, fit, inner, cfg, device)
    details = {
        "reward": best_reward.as_dict(),
        "rlvr_samples_per_group": cfg.rlvr_samples,
        "role": "checkpoint_selection",
        "selected_checkpoint": best_name,
        "candidates_scored": [name for _, name, _, _ in scored],
        "optimizer_steps": 0,
    }
    return StageReport(RLVR_STAGE, 0, train, held, ratio, ece, 0.0, False, details)


def _student_bound(cfg: TeacherV2Config) -> float:
    """Typed command bound: v3 envelope by default, +/-1 only when opted in."""
    if cfg.legacy_student_bound:
        return LEGACY_BOUND_MPS2
    bound = CommandEnvelope().accel
    assert_command_bound_covers(bound, name="student command bound")
    return bound


def _build_student(cfg: TeacherV2Config, device: torch.device) -> StudentLike:
    bound = _student_bound(cfg)
    if _UpstreamStudent is not None and _UpstreamStudentConfig is not None:
        scfg = _UpstreamStudentConfig(
            seed=child_seed(cfg.seed, "student"),
            g_max=bound,
            legacy_bound=cfg.legacy_student_bound,
        )
        return _UpstreamStudent(scfg).to(device)
    return _LocalStudent(child_seed(cfg.seed, "student"), bound).to(device)


def _stage_distill(
    teacher: LoopWM,
    cfg: TeacherV2Config,
    fit: list[Episode],
    inner: list[Episode],
    device: torch.device,
) -> StageReport:
    student = _build_student(cfg, device)
    bound = _student_bound(cfg)
    optimizer = _build_optimizer(
        list(student.named_parameters()),
        cfg,
        cfg.student_lr,  # type: ignore[attr-defined]
    )
    batches = _window_batches(fit, cfg, train=True)
    if not cfg.legacy_student_bound:
        assert_command_bound_covers(
            bound,
            target=torch.cat([batch[2] for batch in batches]),
            name="student command bound",
        )
    steps = 0
    teacher.eval()
    student.train()  # type: ignore[attr-defined]
    for _ in range(cfg.epochs):
        for batch in batches:
            window, cond, _target = (t.to(device) for t in batch[:3])
            with torch.no_grad():
                latent, _ = teacher.refine(window, cond)
                target_output = teacher.head_output(teacher.head_logits(latent))
                decision = _residual_decision(
                    teacher, window, target_output[:, :2], residual=cfg.residual_policy
                )
                if decision is not None:
                    target_output = torch.cat([decision.correction, target_output[:, 2:3]], dim=-1)
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
    details = {
        "student_source": STUDENT_SOURCE,
        "student_file": "student.pt",
        "residual_policy_target": cfg.residual_policy,
        "student_command_bound": bound,
        "legacy_student_bound": cfg.legacy_student_bound,
    }
    return StageReport("distill", steps, train, held, ratio, ece, 0.0, False, details)


def run(cfg: TeacherV2Config) -> TeacherV2Metrics:
    """Execute every stage through ``cfg.stage`` and write one metrics.json."""
    seed_everything(cfg.seed)
    device = torch.device(cfg.device)
    fit, inner, held_ids = partition_episodes(
        load_episodes(cfg.data), seed=cfg.seed, n_held_out=cfg.n_inner_held_out
    )
    adv = AdversarialCurriculum(seed=cfg.seed) if cfg.adversarial else None
    if adv is not None:
        # partition_episodes already refuses eval cells; refuse again at the
        # adversarial seam so the two rules cannot drift apart.
        refuse_eval_episodes([*fit, *inner])
    adv_stats: dict[str, int] = {"offered": 0, "won": 0}
    if cfg.stage == "all":
        stages = list(STAGE_ORDER)
    elif cfg.stage == RLVR_STAGE:
        # Opt-in selection after the teacher stages, before any distillation.
        stages = [*STAGE_ORDER[:-1], RLVR_STAGE]
    else:
        stages = list(STAGE_ORDER[: STAGE_ORDER.index(cfg.stage) + 1])
    if cfg.rlvr and RLVR_STAGE not in stages:
        insert_at = stages.index("distill") if "distill" in stages else len(stages)
        stages.insert(insert_at, RLVR_STAGE)
    model = _build_teacher(cfg, device)
    digest = config_hash(cfg)
    reports: dict[str, StageReport] = {}
    reward: dict[str, Any] = {}
    snapshots: list[tuple[str, dict[str, torch.Tensor]]] = []
    for stage in stages:
        token = _ADV_CONTEXT.set((adv, stage, adv_stats)) if adv is not None else None
        try:
            if stage == "privileged":
                report = _stage_privileged(model, cfg, fit, inner, device)
            elif stage == "dagger":
                report = _stage_dagger(model, cfg, fit, inner, held_ids, device)
            elif stage == RLVR_STAGE:
                report = _stage_rlvr(model, cfg, fit, inner, device, digest, snapshots)
            elif stage == "distill":
                report = _stage_distill(model, cfg, fit, inner, device)
            else:
                report = _stage_sft(model, cfg, fit, inner, device)
        finally:
            if token is not None:
                _ADV_CONTEXT.reset(token)
        if adv is not None:
            report.details["adversarial"] = {
                "enabled": True,
                "stage": stage,
                "seed": adv.seed,
                "win_rate_cap": adv.config.win_rate_cap,
                "intensity": adversarial_budget(stage, adv),
                "offered": adv_stats["offered"],
                "won": adv_stats["won"],
            }
        reports[stage] = report
        if RLVR_STAGE in stages and stage in STAGE_ORDER[:-1]:
            state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            snapshots.append((stage, state))
        if "reward" in report.details:
            reward = report.details["reward"]
    last = reports[stages[-1]]
    # The student (and RLVR selection) stages have no LoopWM entropy; report the
    # last teacher stage's value so --stage all still records the entropy term.
    entropy_enabled = any(report.entropy_enabled for report in reports.values())
    entropy = next(
        (report.entropy for report in reversed(list(reports.values())) if report.entropy_enabled),
        0.0,
    )
    policy = getattr(model, "residual_policy", None)
    residual_info: dict[str, Any] = {
        "enabled": cfg.residual_policy,
        "source": "residual_policy" if cfg.residual_policy else "legacy",
    }
    if isinstance(policy, ResidualPolicy):
        residual_info["gain"] = float(policy.gain.detach())
        residual_info["gate_reasons"] = policy.reason_counts()

    # The distill stage trains a student it does not return; reload that saved
    # network so the disjoint score describes exactly the model `last` reports.
    if "distill" in reports:
        confidence_model = _build_student(cfg, device)
        stored = torch.load(cfg.out / "student.pt", map_location=device, weights_only=True)
        confidence_model.load_state_dict(stored["state_dict"])  # type: ignore[attr-defined]
        confidence_student = True
    else:
        confidence_model = model
        confidence_student = False
    confidence = _confidence_hook(confidence_model, inner, cfg, device, student=confidence_student)

    metrics = TeacherV2Metrics(
        stage=last.stage,
        seed=cfg.seed,
        config_hash=digest,
        rmse_train=last.rmse_train,
        rmse_held_out=last.rmse_held_out,
        ratio=last.ratio,
        ece=confidence.ece_held_out,
        ece_source=DISJOINT_ECE_SOURCE,
        ece_held_out=confidence.ece_held_out,
        conformal_coverage=confidence.conformal_coverage,
        steps=last.steps,
        entropy=entropy,
        entropy_enabled=entropy_enabled,
        reward_provenance=reward,
        module_sources={
            "jepa": JEPA_SOURCE,
            "dagger": DAGGER_SOURCE,
            "privileged": PRIVILEGED_SOURCE,
            "student": STUDENT_SOURCE,
            "residual_policy": residual_info["source"],
        },
        residual_policy=residual_info,
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
        choices=(*STAGE_ORDER, RLVR_STAGE, "all"),
        default="all",
        help=(
            "run every stage through this one (default: all; RLVR is opt-in via "
            "--rlvr or --stage rlvr)"
        ),
    )
    parser.add_argument(
        "--adversarial",
        action="store_true",
        help="opt into the bounded adversarial curriculum on training windows (default off)",
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
    parser.add_argument(
        "--legacy-objective",
        action="store_true",
        help=(
            "keep the supervised-only privileged objective; the default adds the "
            "privileged impact term when a v3 truth sidecar is available"
        ),
    )
    parser.add_argument(
        "--braking-teacher",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "use the privileged radial-braking reference as the privileged/DAgger "
            "target (default); --no-braking-teacher restores recorded v3 labels"
        ),
    )
    parser.add_argument(
        "--no-residual-policy",
        dest="residual_policy",
        action="store_false",
        help="keep the pre-SHIELD02 teacher: raw head correction on the physics baseline",
    )
    parser.add_argument(
        "--legacy-student-bound",
        action="store_true",
        help="explicitly restore the pre-v3 +/-1 m/s^2 student head (too small for v3)",
    )
    parser.add_argument("--jepa-weight", type=float, default=0.05)
    parser.add_argument("--entropy-weight", type=float, default=1.0)
    parser.add_argument("--dagger-rounds", type=int, default=1)
    parser.add_argument("--n-inner-held-out", type=int, default=2)
    parser.add_argument(
        "--rlvr",
        action="store_true",
        help="opt into RLVR verifier checkpoint selection (no policy gradient)",
    )
    parser.add_argument("--rlvr-samples", type=int, default=4)
    parser.add_argument("--rlvr-temperature", type=float, default=0.05)
    parser.add_argument("--truth-sidecar", type=Path, default=None)
    parser.add_argument("--calib-tol", type=float, default=0.15)
    parser.add_argument(
        "--opt", choices=("adamw", "muon"), default="adamw", help="optimizer for all stages"
    )
    parser.add_argument(
        "--k3-per-head", type=int, default=0, help="K3 per-head QKV NS heads (0 = off)"
    )
    parser.add_argument(
        "--k3-qk-clip",
        action="store_true",
        help="K3/K2 QK-Clip (needs --opt muon and named Q/K projections)",
    )
    parser.add_argument(
        "--k3-qk-clip-tau", type=float, default=100.0, help="QK-Clip tau (K2 default 100)"
    )
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
