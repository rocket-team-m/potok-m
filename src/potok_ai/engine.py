"""Unified training engine (SPEC_ENGINE.md v1).

One CLI, one TrainConfig and one metrics report for all three model families.
``lrs`` and ``loopwm-lite`` go through the shared adapter loop below; ``loopwm``
dispatches to the existing ``train_loopwm`` / ``train_smart`` pipelines, which
train the privileged teacher, distil the onboard student and export it. The
``distill`` stage uses the same pipeline: every family already learns from a
teacher (full LoopWM latents for lite, LoopWM outputs for lrs/loopwm), so there
is no second training code path to maintain. ``eval`` loads an export and only
scores it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW

from .calibration import (
    DEFAULT_MAX_LOOPS,
    DEFAULT_TARGET_ECE,
    DEFAULT_TOL,
    confidence_spread_loss,
    confidence_target,
    conformal_calibrate,
    correctness,
    expected_calibration_error,
    sigmoid,
)
from .curriculum import STAGES as CURRICULUM_STAGES
from .curriculum import Curriculum
from .data import (
    SMART_COND_FEATURES,
    SMART_FEATURES,
    SmartWindowDataset,
    group_by_scenario,
    load_episodes,
)
from .export import export_student, load_student_export
from .loopwm import LoopWM, LoopWMConfig
from .loopwm_lite import LiteConfig, LiteModel
from .loopwm_lite import analytical_mac as lite_analytical_mac
from .physics import final_command, physics_baseline
from .student import StatefulTCN, smart_student_config
from .train_loopwm import run as run_loopwm
from .train_smart import _batches, generalisation_ratio, normalized_correction_loss
from .train_smart import evaluate as window_evaluate

MODELS = ("lrs", "loopwm-lite", "loopwm")
STAGES = ("train", "distill", "eval")
METRIC_FIELDS = (
    "model",
    "rmse_train",
    "rmse_held_out",
    "ratio",
    "params",
    "mac",
    "bytes",
    "mae",
    "max_error",
    "ece",
    "seed",
    "config_hash",
)


@dataclass
class TrainConfig:
    """The single configuration shared by every adapter."""

    model: str
    stage: str
    data: Path
    out: Path
    epochs: int = 1
    seed: int = 1
    device: str = "cpu"
    window: int = 50
    batch: int = 8
    limit: int | None = None
    eval_limit: int | None = None
    lr: float = 2e-3
    student_lr: float = 1e-3
    width: int = 128  # full LoopWM width
    rank: int = 64
    heads: int = 4
    max_iters: int = 6
    exit_threshold: float = 1e-3
    latent_weight: float = 0.05
    target_weight: float = 0.1
    conf_weight: float = 1.0
    latent_tokens: int = 16
    lite_width: int = 64
    attn_rank: int = 8
    lite_iters: int = 2
    lite_threshold: float = 1e-2
    residual: bool = True
    curriculum: bool = True
    calibrate: bool = False
    ece_target: float = DEFAULT_TARGET_ECE
    calib_tol: float = DEFAULT_TOL

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> TrainConfig:
        fields = {name: getattr(args, name) for name in cls.__dataclass_fields__}
        return cls(**fields)

    def legacy_args(self, out: Path) -> argparse.Namespace:
        """Adapt to the existing train_smart / train_loopwm CLI helpers."""
        return argparse.Namespace(
            data=self.data,
            epochs=self.epochs,
            seed=self.seed,
            device=self.device,
            out=out,
            window=self.window,
            batch=self.batch,
            limit=self.limit,
            eval_limit=self.eval_limit,
            lr=self.lr,
            student_lr=self.student_lr,
            target_weight=self.target_weight,
            width=self.width,
            rank=self.rank,
            heads=self.heads,
            max_iters=self.max_iters,
            exit_threshold=self.exit_threshold,
            latent_weight=self.latent_weight,
            residual=self.residual,
            curriculum=self.curriculum,
        )


@dataclass
class Metrics:
    """The one report format written to ``out/metrics.json``.

    The seven SPEC_ENGINE fields are unchanged; the tail comes from the ideal
    loop's held-out report (SPEC_ENGINE_IDEAL.md §5).
    """

    model: str
    rmse_train: float
    rmse_held_out: float
    ratio: float
    params: int
    mac: int
    bytes: int
    mae: float = 0.0
    max_error: float = 0.0
    ece: float = 0.0
    seed: int = 1
    config_hash: str = ""

    def as_dict(self) -> dict:
        return {field: getattr(self, field) for field in METRIC_FIELDS}


def print_table(metrics: Metrics) -> None:
    """Print the single table SPEC_ENGINE.md asks for."""
    values = [str(metrics.as_dict()[field]) for field in METRIC_FIELDS]
    widths = [max(len(field), len(value)) for field, value in zip(METRIC_FIELDS, values)]
    print("  ".join(field.ljust(width) for field, width in zip(METRIC_FIELDS, widths)))
    print("  ".join("-" * width for width in widths))
    print("  ".join(value.ljust(width) for value, width in zip(values, widths)))


def write_metrics(out: Path, metrics: Metrics) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / "metrics.json"
    path.write_text(json.dumps(metrics.as_dict(), indent=2) + "\n")
    return path


def _held_out(groups: dict) -> set[str]:
    held = {name for name, episodes in groups.items() if any(e.held_out for e in episodes)}
    if not held or len(held) == len(groups):
        raise ValueError("validation data needs at least one held-out and one training scenario")
    return held


def config_hash(cfg: TrainConfig) -> str:
    """Short hash of the training config; the output directory is not a knob."""
    fields = asdict(cfg)
    fields.pop("out", None)
    body = json.dumps(fields, sort_keys=True, default=str)
    return hashlib.sha256(body.encode()).hexdigest()[:16]


def _split_metrics(
    model: str,
    rmse: dict,
    held_out: set[str],
    params: int,
    mac: int,
    size: int,
    *,
    mae: float = 0.0,
    max_error: float = 0.0,
    ece: float = 0.0,
    seed: int = 1,
    hash_value: str = "",
) -> Metrics:
    train = [value for name, value in rmse.items() if name not in held_out]
    held = [value for name, value in rmse.items() if name in held_out]
    return Metrics(
        model=model,
        rmse_train=sum(train) / len(train),
        rmse_held_out=sum(held) / len(held),
        ratio=generalisation_ratio(rmse, held_out),
        params=int(params),
        mac=int(mac),
        bytes=int(size),
        mae=float(mae),
        max_error=float(max_error),
        ece=float(ece),
        seed=int(seed),
        config_hash=hash_value,
    )


def build_metrics(
    model: str,
    cfg: TrainConfig,
    rmse: dict,
    held_out: set[str],
    params: int,
    mac: int,
    size: int,
    logits: np.ndarray,
    error: np.ndarray,
) -> Metrics:
    """Fold held-out confidence logits and command errors into the report."""
    z = np.asarray(logits, dtype=np.float64).ravel()
    err = np.asarray(error, dtype=np.float64).ravel()
    if z.size == 0 or err.size != z.size:
        raise ValueError("held-out predictions must be non-empty and paired")

    raw_ece = expected_calibration_error(sigmoid(z), correctness(err, cfg.calib_tol))
    calibrated_ece = raw_ece
    if cfg.calibrate:
        result = conformal_calibrate(
            z,
            err,
            tol=cfg.calib_tol,
            target_ece=cfg.ece_target,
            max_loops=DEFAULT_MAX_LOOPS,
        )
        calibrated_ece = result.ece
        payload = result.as_dict()
        payload.update(model=model, seed=cfg.seed, config_hash=config_hash(cfg))
        cfg.out.mkdir(parents=True, exist_ok=True)
        (cfg.out / "calibration.json").write_text(json.dumps(payload, indent=2) + "\n")

    return _split_metrics(
        model,
        rmse,
        held_out,
        params,
        mac,
        size,
        mae=float(np.abs(err).mean()),
        max_error=float(err.max()),
        ece=calibrated_ece,
        seed=cfg.seed,
        hash_value=config_hash(cfg),
    )


class ModelAdapter(ABC):
    """The five operations every family exposes, plus the shared run loop."""

    name: str

    @abstractmethod
    def build(self, cfg: TrainConfig, device: torch.device) -> nn.Module: ...

    @abstractmethod
    def batches(self, episodes: list, cfg: TrainConfig): ...

    @abstractmethod
    def train_step(self, model: nn.Module, batch, optimizer: AdamW, cfg: TrainConfig) -> float: ...

    @abstractmethod
    def evaluate(
        self, model: nn.Module, groups: dict, cfg: TrainConfig, device: torch.device
    ) -> dict[str, float]: ...

    @abstractmethod
    def export_path(self, cfg: TrainConfig) -> Path: ...

    @abstractmethod
    def export(self, model: nn.Module, cfg: TrainConfig) -> Path: ...

    @abstractmethod
    def load(self, cfg: TrainConfig, device: torch.device) -> nn.Module: ...

    @abstractmethod
    def mac(self, cfg: TrainConfig) -> int: ...

    @abstractmethod
    def held_out_predictions(
        self, model: nn.Module, groups: dict, cfg: TrainConfig, device: torch.device
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (pre-sigmoid confidence logits, command errors) on held-out data."""

    def count_params(self, model: nn.Module) -> int:
        return sum(parameter.numel() for parameter in model.parameters())

    @staticmethod
    def _curriculum(cfg: TrainConfig) -> Curriculum | None:
        if not cfg.curriculum:
            return None
        return Curriculum(
            seed=cfg.seed, epochs_per_stage=max(1, cfg.epochs // len(CURRICULUM_STAGES))
        )

    def run(self, cfg: TrainConfig) -> Metrics:
        """Train, evaluate, export and report -- shared by lrs and loopwm-lite."""
        seed_everything(cfg.seed)
        device = torch.device(cfg.device)
        groups = group_by_scenario(load_episodes(cfg.data))
        held_out = _held_out(groups)
        train_episodes = [
            episode for name, group in groups.items() if name not in held_out for episode in group
        ]
        curriculum = self._curriculum(cfg)
        model = self.build(cfg, device)
        optimizer = AdamW(model.parameters(), lr=cfg.lr)
        for epoch in range(cfg.epochs):
            chosen = curriculum.select(train_episodes, epoch) if curriculum else train_episodes
            if not chosen:
                continue
            for batch in self.batches(chosen, cfg):
                self.train_step(model, batch, optimizer, cfg)
        rmse = self.evaluate(model, groups, cfg, device)
        path = self.export(model, cfg)
        return self._report(model, cfg, groups, held_out, rmse, path, device)

    def evaluate_only(self, cfg: TrainConfig) -> Metrics:
        device = torch.device(cfg.device)
        groups = group_by_scenario(load_episodes(cfg.data))
        held_out = _held_out(groups)
        model = self.load(cfg, device)
        path = self.export_path(cfg)
        if not path.exists():
            raise ValueError(f"no export at {path}; run the train stage first")
        rmse = self.evaluate(model, groups, cfg, device)
        return self._report(model, cfg, groups, held_out, rmse, path, device)

    def _report(
        self,
        model: nn.Module,
        cfg: TrainConfig,
        groups: dict,
        held_out: set[str],
        rmse: dict,
        path: Path,
        device: torch.device,
    ) -> Metrics:
        logits, error = self.held_out_predictions(model, groups, cfg, device)
        return build_metrics(
            self.name,
            cfg,
            rmse,
            held_out,
            self.count_params(model),
            self.mac(cfg),
            path.stat().st_size,
            logits,
            error,
        )


def seed_everything(seed: int) -> None:
    """All families share this seeding rule; CPU threading is fixed to one."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


def _smart_frames(episode) -> tuple[np.ndarray, np.ndarray | None]:
    """A 20-value frame per tick: 14 features plus the 6 conditioning numbers."""
    frames = np.asarray(episode.frames[:, :SMART_FEATURES], dtype=np.float32)
    conditioning = np.asarray(episode.conditioning, dtype=np.float32)
    with_cond = np.concatenate([frames, np.tile(conditioning, (len(frames), 1))], axis=1)
    targets = None if episode.targets is None else np.asarray(episode.targets, dtype=np.float32)
    return with_cond, targets


def _window_held_out_predictions(
    model: nn.Module, groups: dict, cfg: TrainConfig, device: torch.device, pooled: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Confidence logits and command errors for lite / LoopWM window families."""
    model.eval()
    logits, errors = [], []
    with torch.no_grad():
        for name, episodes in groups.items():
            if not any(episode.held_out for episode in episodes):
                continue
            dataset = SmartWindowDataset(episodes, window=cfg.window, limit=cfg.eval_limit)
            if not dataset.samples:
                raise ValueError(f"scenario {name} has no evaluation windows")
            for windows, conditions, targets in _batches(dataset, cfg.batch):
                latent, _ = model.refine(windows, conditions)
                summary = latent.mean(dim=1) if pooled else latent
                raw = model.head_logits(summary)
                output = model.head_output(raw)
                baseline = physics_baseline(windows[:, -1, :])
                command = final_command(output[:, :2], baseline, cfg.residual)
                errors.extend((command - targets).abs().max(dim=1).values.tolist())
                logits.extend(raw[:, 2].tolist())
    return np.asarray(logits), np.asarray(errors)


class LRSAdapter(ModelAdapter):
    """Onboard student (StatefulTCN): one 20-value frame per tick."""

    name = "lrs"

    def build(self, cfg: TrainConfig, device: torch.device) -> StatefulTCN:
        return StatefulTCN(smart_student_config(seed=cfg.seed)).to(device)

    def batches(self, episodes: list, cfg: TrainConfig):
        data = [_smart_frames(episode) for episode in episodes]
        if cfg.limit is not None:
            data = data[: cfg.limit]
        yield from data

    def train_step(self, model: StatefulTCN, batch, optimizer: AdamW, cfg: TrainConfig) -> float:
        frames, targets = batch
        if targets is None:
            raise ValueError("lrs training needs reference corrections")
        device = next(model.parameters()).device
        model.reset_state()
        command_losses, logits, confidences, errors = [], [], [], []
        for tick, frame in enumerate(frames):
            corrections, confidence = model.step(frame)
            if tick + 1 < cfg.window:
                continue
            reference = torch.as_tensor(targets[tick, :2], dtype=torch.float32, device=device)
            features = torch.as_tensor(frame[:SMART_FEATURES], dtype=torch.float32, device=device)
            command = final_command(corrections, physics_baseline(features), cfg.residual)
            error = (command - reference).abs().max().detach()
            command_losses.append(F.mse_loss(command, reference))
            # Train the LOGIT, never its sigmoid: a near-zero target on the
            # probability drives the logit to -inf and collapses calibration.
            # The target scale is shared across the episode's ticks so a spread
            # survives even when every error is much larger than the tolerance.
            logits.append(model.confidence_logit())
            confidences.append(confidence)
            errors.append(error)
        if not command_losses:
            raise ValueError("no student updates: episodes must be longer than the window")
        error_stack = torch.stack(errors)
        conf_loss = F.binary_cross_entropy_with_logits(
            torch.stack(logits), confidence_target(error_stack, cfg.calib_tol)
        )
        loss = (
            torch.stack(command_losses).mean()
            + cfg.conf_weight * conf_loss
            + confidence_spread_loss(torch.stack(confidences), error_stack, cfg.calib_tol)
        )
        optimizer.zero_grad(set_to_none=False)
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    def evaluate(
        self, model: StatefulTCN, groups: dict, cfg: TrainConfig, device: torch.device
    ) -> dict[str, float]:
        result: dict[str, float] = {}
        model.eval()
        with torch.no_grad():
            for name, episodes in groups.items():
                squared, count = 0.0, 0
                for episode in episodes:
                    frames, targets = _smart_frames(episode)
                    if targets is None:
                        raise ValueError(f"scenario {name} has no reference corrections")
                    model.reset_state()
                    for tick, frame in enumerate(frames):
                        corrections, _ = model.step(frame)
                        if tick + 1 < cfg.window:
                            continue
                        reference = torch.as_tensor(
                            targets[tick, :2], dtype=torch.float32, device=device
                        )
                        features = torch.as_tensor(
                            frame[:SMART_FEATURES], dtype=torch.float32, device=device
                        )
                        command = final_command(
                            corrections, physics_baseline(features), cfg.residual
                        )
                        squared += float((command - reference).square().sum())
                        count += 2
                result[name] = (squared / count) ** 0.5
        return result

    def held_out_predictions(
        self, model: StatefulTCN, groups: dict, cfg: TrainConfig, device: torch.device
    ) -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        logits, errors = [], []
        with torch.no_grad():
            for name, episodes in groups.items():
                if not any(episode.held_out for episode in episodes):
                    continue
                for episode in episodes:
                    frames, targets = _smart_frames(episode)
                    if targets is None:
                        raise ValueError(f"scenario {name} has no reference corrections")
                    model.reset_state()
                    for tick, frame in enumerate(frames):
                        corrections, _ = model.step(frame)
                        if tick + 1 < cfg.window:
                            continue
                        reference = torch.as_tensor(
                            targets[tick, :2], dtype=torch.float32, device=device
                        )
                        features = torch.as_tensor(
                            frame[:SMART_FEATURES], dtype=torch.float32, device=device
                        )
                        command = final_command(
                            corrections, physics_baseline(features), cfg.residual
                        )
                        errors.append(float((command - reference).abs().max()))
                        logits.append(float(model.confidence_logit()))
        return np.asarray(logits), np.asarray(errors)

    def export_path(self, cfg: TrainConfig) -> Path:
        return cfg.out / "lrs.bin"

    def export(self, model: StatefulTCN, cfg: TrainConfig) -> Path:
        return export_student(model, self.export_path(cfg))

    def load(self, cfg: TrainConfig, device: torch.device) -> StatefulTCN:
        return load_student_export(self.export_path(cfg)).to_model().to(device)

    def mac(self, cfg: TrainConfig) -> int:
        c = smart_student_config()
        w, r = c.width, c.rank
        # stem + loops * blocks * (rec_h, rec_x, fused TSCT) + heads
        return int(c.in_features * w + c.loops * c.n_blocks * (3 * w * w + 2 * w * r) + w * (2 + 1))


class LiteAdapter(ModelAdapter):
    """LoopWM-lite: Perceiver encoder, two shared passes, latent distillation."""

    name = "loopwm-lite"

    def _teacher(self, cfg: TrainConfig, device: torch.device) -> LoopWM:
        teacher = LoopWM(
            LoopWMConfig(
                width=cfg.width,
                heads=cfg.heads,
                max_iters=cfg.max_iters,
                exit_threshold=cfg.exit_threshold,
            )
        )
        checkpoint = cfg.out / "loopwm.pt"
        if checkpoint.exists():
            stored = torch.load(checkpoint, map_location="cpu", weights_only=True)
            teacher.load_state_dict(stored["state_dict"])
        return teacher.to(device)

    def build(self, cfg: TrainConfig, device: torch.device) -> LiteModel:
        model = LiteModel(
            LiteConfig(
                window=cfg.window,
                latent_tokens=cfg.latent_tokens,
                width=cfg.lite_width,
                attn_rank=cfg.attn_rank,
                max_iters=cfg.lite_iters,
                exit_threshold=cfg.lite_threshold,
                target_width=cfg.width,
            )
        ).to(device)
        model.set_teacher(self._teacher(cfg, device))
        return model

    def batches(self, episodes: list, cfg: TrainConfig):
        dataset = SmartWindowDataset(episodes, window=cfg.window, limit=cfg.limit)
        if not dataset.samples:
            raise ValueError("no training windows in the non-held-out scenarios")
        yield from _batches(dataset, cfg.batch)

    def train_step(self, model: LiteModel, batch, optimizer: AdamW, cfg: TrainConfig) -> float:
        windows, conditions, targets = batch
        latent, _ = model.refine(windows, conditions)
        pooled = latent.mean(dim=1)
        raw = model.head_logits(pooled)
        prediction = model.head_output(raw)
        command = final_command(
            prediction[:, :2], physics_baseline(windows[:, -1, :]), cfg.residual
        )
        correction = normalized_correction_loss(command, targets)
        teacher_targets = model.teacher_targets(windows, conditions)
        predictions = model.latent_predictions(pooled)
        latent_loss = torch.stack(
            [F.mse_loss(predictions[h], teacher_targets[h]) for h in predictions]
        ).mean()
        error = (command - targets).abs().max(dim=1).values.detach()
        # Logit-domain BCE keeps a spread; sigmoid MSE collapses the logit.
        confidence = F.binary_cross_entropy_with_logits(
            raw[:, 2], confidence_target(error, cfg.calib_tol)
        )
        loss = (
            correction
            + cfg.latent_weight * latent_loss
            + cfg.conf_weight * confidence
            + confidence_spread_loss(prediction[:, 2], error, cfg.calib_tol)
        )
        optimizer.zero_grad(set_to_none=False)
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    def evaluate(
        self, model: LiteModel, groups: dict, cfg: TrainConfig, device: torch.device
    ) -> dict[str, float]:
        return window_evaluate(model, groups, cfg.legacy_args(cfg.out / "unused.pt"), device)

    def held_out_predictions(
        self, model: LiteModel, groups: dict, cfg: TrainConfig, device: torch.device
    ) -> tuple[np.ndarray, np.ndarray]:
        return _window_held_out_predictions(model, groups, cfg, device, pooled=True)

    def export_path(self, cfg: TrainConfig) -> Path:
        return cfg.out / "lite.pt"

    def export(self, model: LiteModel, cfg: TrainConfig) -> Path:
        path = self.export_path(cfg)
        torch.save(
            {
                "model_config": asdict(model.cfg),
                "state_dict": {
                    key: value
                    for key, value in model.state_dict().items()
                    if not key.startswith("target_proj.")
                },
                "seed": cfg.seed,
            },
            path,
        )
        return path

    def load(self, cfg: TrainConfig, device: torch.device) -> LiteModel:
        stored = torch.load(self.export_path(cfg), map_location="cpu", weights_only=True)
        model = LiteModel(LiteConfig(**stored["model_config"]))
        model.load_state_dict(stored["state_dict"], strict=False)
        return model.to(device)

    def count_params(self, model: LiteModel) -> int:
        # target_proj is a training-only teacher adapter; it is not exported.
        return sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if not name.startswith("target_proj.")
        )

    def mac(self, cfg: TrainConfig) -> int:
        return lite_analytical_mac(
            LiteConfig(
                window=cfg.window,
                latent_tokens=cfg.latent_tokens,
                width=cfg.lite_width,
                attn_rank=cfg.attn_rank,
                max_iters=cfg.lite_iters,
            )
        )


def loopwm_mac(cfg: TrainConfig) -> int:
    """Analytic MAC count for one full LoopWM window inference."""
    t, d, w, heads = cfg.window, cfg.width, cfg.max_iters, cfg.heads
    encoder = t * 14 * d + SMART_COND_FEATURES * 2 * d + (2 * d) * (2 * d)
    per_iter = t * d * 3 * d + 2 * heads * t * t * (d // heads)
    per_iter += t * d * d + 2 * t * d * 4 * d
    heads_cost = 3 * t * d * d + d * 3
    return int(encoder + w * per_iter + heads_cost)


class LoopWMAdapter(ModelAdapter):
    """Privileged LoopWM; train/distill dispatches to the existing scripts."""

    name = "loopwm"

    def build(self, cfg: TrainConfig, device: torch.device) -> LoopWM:
        return LoopWM(
            LoopWMConfig(
                width=cfg.width,
                heads=cfg.heads,
                max_iters=cfg.max_iters,
                exit_threshold=cfg.exit_threshold,
                latent_weight=cfg.latent_weight,
            )
        ).to(device)

    def batches(self, episodes: list, cfg: TrainConfig):
        dataset = SmartWindowDataset(episodes, window=cfg.window, limit=cfg.limit)
        if not dataset.samples:
            raise ValueError("no training windows in the non-held-out scenarios")
        yield from _batches(dataset, cfg.batch)

    def train_step(self, model: LoopWM, batch, optimizer: AdamW, cfg: TrainConfig) -> float:
        from .jepa import EMATeacher

        if not hasattr(self, "_target_encoder"):
            self._target_encoder = EMATeacher(model.encoder, model.cfg.ema_momentum)
        windows, conditions, targets = batch
        latent, _ = model.refine(windows, conditions)
        raw = model.head_logits(latent)
        output = model.head_output(raw)
        command = final_command(output[:, :2], physics_baseline(windows[:, -1, :]), cfg.residual)
        main = normalized_correction_loss(command, targets)
        latent_loss = model.latent_prediction_loss(
            windows, conditions, self._target_encoder, latent=latent
        )
        error = (command - targets).abs().max(dim=1).values.detach()
        confidence = F.mse_loss(raw[:, 2], confidence_target(error, cfg.calib_tol))
        loss = (
            main
            + cfg.latent_weight * latent_loss
            + 0.1 * confidence
            + confidence_spread_loss(raw[:, 2], error, cfg.calib_tol)
        )
        optimizer.zero_grad(set_to_none=False)
        loss.backward()
        optimizer.step()
        self._target_encoder.update(model.encoder)
        return float(loss.detach())

    def evaluate(
        self, model: LoopWM, groups: dict, cfg: TrainConfig, device: torch.device
    ) -> dict[str, float]:
        return window_evaluate(model, groups, cfg.legacy_args(cfg.out / "unused.pt"), device)

    def held_out_predictions(
        self, model: LoopWM, groups: dict, cfg: TrainConfig, device: torch.device
    ) -> tuple[np.ndarray, np.ndarray]:
        return _window_held_out_predictions(model, groups, cfg, device, pooled=False)

    def export_path(self, cfg: TrainConfig) -> Path:
        return cfg.out / "loopwm.pt"

    def export(self, model: LoopWM, cfg: TrainConfig) -> Path:
        path = self.export_path(cfg)
        torch.save(
            {
                "model_config": model.cfg.__dict__,
                "state_dict": model.state_dict(),
                "seed": cfg.seed,
            },
            path,
        )
        return path

    def load(self, cfg: TrainConfig, device: torch.device) -> LoopWM:
        stored = torch.load(self.export_path(cfg), map_location="cpu", weights_only=True)
        model = LoopWM(LoopWMConfig(**stored["model_config"]))
        model.load_state_dict(stored["state_dict"])
        return model.to(device)

    def mac(self, cfg: TrainConfig) -> int:
        return loopwm_mac(cfg)

    def run(self, cfg: TrainConfig) -> Metrics:
        """Dispatch to the existing train_loopwm (and its train_smart helpers)."""
        seed_everything(cfg.seed)
        result = run_loopwm(cfg.legacy_args(self.export_path(cfg)))
        device = torch.device(cfg.device)
        model = self.load(cfg, device)  # the trained teacher the legacy path saved
        groups = group_by_scenario(load_episodes(cfg.data))
        logits, error = self.held_out_predictions(model, groups, cfg, device)
        student_bin = self.export_path(cfg).with_name("loopwm_student.bin")
        return build_metrics(
            self.name,
            cfg,
            result["rmse"],
            set(result["held_out"]),
            self.count_params(model),
            self.mac(cfg),
            student_bin.stat().st_size if student_bin.exists() else 0,
            logits,
            error,
        )


ADAPTERS: dict[str, type[ModelAdapter]] = {
    "lrs": LRSAdapter,
    "loopwm-lite": LiteAdapter,
    "loopwm": LoopWMAdapter,
}


def run(cfg: TrainConfig) -> Metrics:
    """Dispatch one run by model name and stage."""
    if cfg.model not in ADAPTERS:
        raise ValueError(f"unknown model {cfg.model}; expected one of {MODELS}")
    if cfg.stage not in STAGES:
        raise ValueError(f"unknown stage {cfg.stage}; expected one of {STAGES}")
    if cfg.epochs < 1 or cfg.batch < 1 or cfg.window <= max(LiteConfig.horizons):
        raise ValueError("epochs and batch must be positive; window must exceed the horizons")
    cfg.out.mkdir(parents=True, exist_ok=True)
    adapter = ADAPTERS[cfg.model]()
    if cfg.stage == "eval":
        return adapter.evaluate_only(cfg)
    return adapter.run(cfg)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="potok-engine", description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--stage", choices=STAGES, default="train")
    parser.add_argument("--data", type=Path, required=True, help="v2 validation file or directory")
    parser.add_argument("--out", type=Path, required=True, help="directory for exports and metrics")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--student-lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=128, help="full LoopWM width")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--max-iters", type=int, default=6)
    parser.add_argument("--exit-threshold", type=float, default=1e-3)
    parser.add_argument("--latent-weight", type=float, default=0.05)
    parser.add_argument("--target-weight", type=float, default=0.1)
    parser.add_argument(
        "--conf-weight",
        type=float,
        default=1.0,
        help="weight of the logit-domain confidence loss (default: 1.0)",
    )
    parser.add_argument("--latent-tokens", type=int, default=16)
    parser.add_argument("--lite-width", type=int, default=64)
    parser.add_argument("--attn-rank", type=int, default=8)
    parser.add_argument("--lite-iters", type=int, default=2)
    parser.add_argument("--lite-threshold", type=float, default=1e-2)
    parser.add_argument(
        "--residual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="predict a correction on top of the physics baseline (default: on)",
    )
    parser.add_argument(
        "--curriculum",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="stage scenarios calm -> gust -> shear -> sensor_fault (default: on)",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="calibrate held-out confidence and write calibration.json",
    )
    parser.add_argument("--ece-target", type=float, default=DEFAULT_TARGET_ECE)
    parser.add_argument("--calib-tol", type=float, default=DEFAULT_TOL)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = TrainConfig.from_args(args)
    metrics = run(cfg)
    print_table(metrics)
    write_metrics(cfg.out, metrics)


if __name__ == "__main__":
    main()
