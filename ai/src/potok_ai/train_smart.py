"""Train the SPEC_SMART.md teacher on randomized Potok-M, then distil the student.

The split lives in the data: a v2 validation episode names its wind/gust
``scenario`` and carries a ``held_out`` flag. Training uses the other
scenarios; the acceptance number is the ratio of held-out RMSE to the mean
training-scenario RMSE (must be <= 1.25). No vehicle names are involved: every
episode is the same rocket inside its real tolerances.

The teacher uses a std-normalised correction loss plus JEPA (0.05) and MTP
(0.02); the small stateful student learns from the teacher with a small weight
on the simulation target and is exported as v2 model_kind=2.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import AdamW

from .data import SMART_FEATURES, SmartWindowDataset, group_by_scenario, load_episodes
from .data_v3 import training_target
from .export import export_student
from .jepa import JEPAPredictor, mask_windows
from .model import SmartConfig, SmartTeacher
from .physics import final_command, physics_baseline
from .student import StatefulTCN, smart_student_config

GENERALISATION_LIMIT = 1.25


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="potok-train-smart")
    parser.add_argument("--data", type=Path, required=True, help="validation file or directory")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    parser.add_argument("--out", type=Path, default=Path("smart.pt"))
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="cap windows / student updates")
    parser.add_argument("--eval-limit", type=int, default=None, help="cap windows per scenario")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--student-lr", type=float, default=1e-3)
    parser.add_argument("--target-weight", type=float, default=0.1)
    parser.add_argument(
        "--allow-diagnostic-target", action="store_true",
        help="fit angle-domain g_theta when data is v1/v2 (diagnostic training "
             "only; command-target training requires v3)",
    )
    parser.add_argument("--width", type=int, default=128, help="teacher width (spec: 128)")
    parser.add_argument("--rank", type=int, default=64, help="teacher TSCT rank (spec: 64)")
    return parser


def _batches(dataset, batch: int):
    if len(dataset) == 0:
        return
    samples = [dataset[index] for index in range(len(dataset))]
    windows = torch.stack([sample[0] for sample in samples])
    conditions = torch.stack([sample[1] for sample in samples])
    targets = torch.stack([sample[2] for sample in samples])
    for start in range(0, len(windows), batch):
        yield windows[start : start + batch], conditions[start : start + batch], targets[start : start + batch]


class _TeacherEMA:
    """Frozen EMA of the teacher, used as the JEPA target network."""

    def __init__(self, student: SmartTeacher, momentum: float) -> None:
        self.momentum = momentum
        self.net = deepcopy(student).eval().requires_grad_(False)
        self.sync(student)

    @torch.no_grad()
    def sync(self, student: SmartTeacher) -> None:
        for target, source in zip(self.net.parameters(), student.parameters()):
            target.copy_(source)
        for target, source in zip(self.net.buffers(), student.buffers()):
            target.copy_(source)

    @torch.no_grad()
    def update(self, student: SmartTeacher) -> None:
        for target, source in zip(self.net.parameters(), student.parameters()):
            target.mul_(self.momentum).add_(source, alpha=1.0 - self.momentum)

    @torch.no_grad()
    def __call__(self, window: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.net.forward_features(window, cond)


def normalized_correction_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE scaled by the target's per-column std; a constant output scores ~1."""
    scale = target.std(dim=0, unbiased=False).clamp_min(1e-3)
    return ((prediction - target) / scale).square().mean()


def smart_jepa_loss(
    model: SmartTeacher,
    ema: _TeacherEMA,
    window: torch.Tensor,
    cond: torch.Tensor,
    predictor: JEPAPredictor,
    mask_ratio: float = 0.25,
    shift: int = 3,
) -> torch.Tensor:
    """Predict the EMA teacher's full shifted latent from a masked latent, once."""
    if window.shape[1] <= shift:
        raise ValueError("the window must be longer than the JEPA shift")
    mask = mask_windows(window, mask_ratio)
    masked = torch.where(mask.unsqueeze(-1), torch.zeros_like(window), window)
    latent = model.forward_features(masked, cond)
    with torch.no_grad():
        target = ema(window, cond)
    return F.mse_loss(predictor(latent[:, :-shift]), target[:, shift:])


def mtp_loss(
    model: SmartTeacher,
    window: torch.Tensor,
    latent: torch.Tensor,
) -> torch.Tensor:
    """Predict the next K frames at each position from the teacher's latent."""
    horizon = model.cfg.mtp_k
    time = window.shape[1]
    if time <= horizon:
        raise ValueError("the window must be longer than the MTP horizon")
    source = latent[:, :-horizon]
    predicted = model.mtp_head(source).view(
        window.shape[0], time - horizon, horizon, model.cfg.in_features
    )
    target = torch.stack(
        [window[:, step + 1 : step + 1 + time - horizon] for step in range(horizon)], dim=2
    )
    mu = target.mean(dim=(0, 1, 2), keepdim=True)
    scale = target.std(dim=(0, 1, 2), unbiased=False, keepdim=True).clamp_min(1e-3)
    return F.mse_loss((predicted - mu) / scale, (target - mu) / scale)


def train_teacher(episodes, args: argparse.Namespace, device: torch.device) -> SmartTeacher:
    """Train the big loop teacher on every non-held-out scenario."""
    config = SmartConfig(width=args.width, rank=args.rank)
    model = SmartTeacher(config).to(device)
    ema = _TeacherEMA(model, config.ema_momentum)
    predictor = JEPAPredictor(config.width).to(device)
    optimizer = AdamW(list(model.parameters()) + list(predictor.parameters()), lr=args.lr)
    dataset = SmartWindowDataset(
        episodes,
        window=args.window,
        limit=args.limit,
        allow_diagnostic_target=getattr(args, "allow_diagnostic_target", False),
    )
    if not dataset.samples:
        raise ValueError("no training windows in the non-held-out scenarios")

    model.train()
    predictor.train()
    for _ in range(args.epochs):
        for windows, conditions, targets in _batches(dataset, args.batch):
            latent = model.forward_features(windows, conditions)
            prediction = model.head_from_latent(latent)[:, :2]
            main = normalized_correction_loss(prediction, targets)
            jepa = smart_jepa_loss(model, ema, windows, conditions, predictor)
            mtp = mtp_loss(model, windows, latent)
            loss = main + config.jepa_weight * jepa + config.mtp_weight * mtp
            optimizer.zero_grad(set_to_none=False)
            loss.backward()
            optimizer.step()
            ema.update(model)
    return model


def evaluate(
    model: SmartTeacher, groups: dict[str, list], args: argparse.Namespace, device: torch.device
) -> dict[str, float]:
    """Return raw-scale RMSE for the two corrections, per wind/gust scenario."""
    residual = getattr(args, "residual", False)
    result: dict[str, float] = {}
    model.eval()
    with torch.no_grad():
        for name, episodes in groups.items():
            dataset = SmartWindowDataset(
                episodes,
                window=args.window,
                limit=args.eval_limit,
                allow_diagnostic_target=getattr(args, "allow_diagnostic_target", False),
            )
            if not dataset.samples:
                raise ValueError(f"scenario {name} has no evaluation windows")
            squared, count = 0.0, 0
            for windows, conditions, targets in _batches(dataset, args.batch):
                output = model(windows, conditions)
                command = final_command(
                    output[:, :2], physics_baseline(windows[:, -1, :]), residual
                )
                squared += float((command - targets).square().sum())
                count += command.numel()
            result[name] = (squared / count) ** 0.5
    return result


def generalisation_ratio(rmse: dict[str, float], held_out: set[str]) -> float:
    """Mean held-out RMSE / mean training RMSE; <= 1.25 means generalising."""
    train = [value for name, value in rmse.items() if name not in held_out]
    held = [value for name, value in rmse.items() if name in held_out]
    if not train or not held:
        raise ValueError("both training and held-out scenarios are needed")
    return (sum(held) / len(held)) / (sum(train) / len(train))


def distil_student(
    teacher: SmartTeacher, episodes, args: argparse.Namespace, device: torch.device
) -> StatefulTCN:
    """Match the teacher at each window's last frame, with a small target weight."""
    student = StatefulTCN(smart_student_config(seed=args.seed)).to(device)
    optimizer = AdamW(student.parameters(), lr=args.student_lr)
    student.train()
    updates = 0
    for episode in episodes:
        if episode.conditioning is None or episode.targets is None:
            raise ValueError("smart distillation needs conditioning and reference targets")
        frames = np.asarray(episode.frames, dtype=np.float32)[:, :SMART_FEATURES]
        command = training_target(
            episode,
            allow_diagnostic_target=getattr(args, "allow_diagnostic_target", False),
        )
        condition = torch.as_tensor(episode.conditioning, dtype=torch.float32, device=device)
        student.reset_state()
        for tick in range(len(frames)):
            frame = np.concatenate(
                [frames[tick], np.asarray(episode.conditioning, dtype=np.float32)]
            ).astype(np.float32)
            corrections, confidence = student.step(frame)
            if tick + 1 < args.window:
                continue
            window = torch.as_tensor(
                frames[tick + 1 - args.window : tick + 1], dtype=torch.float32, device=device
            ).unsqueeze(0)
            with torch.no_grad():
                expected = teacher(window, condition.unsqueeze(0))[0, :2]
                reference = torch.as_tensor(command[tick], dtype=torch.float32, device=device)
                target_conf = torch.as_tensor(0.95, dtype=torch.float32, device=device)
            loss = (
                F.mse_loss(corrections, expected)
                + args.target_weight * F.mse_loss(corrections, reference)
                + 0.1 * F.mse_loss(confidence, target_conf)
            )
            optimizer.zero_grad(set_to_none=False)
            loss.backward()
            optimizer.step()
            updates += 1
            if args.limit is not None and updates >= args.limit:
                return student
    if updates == 0:
        raise ValueError("no student updates: episodes must be longer than the window")
    return student


def run(args: argparse.Namespace) -> dict:
    """Full pipeline: split by scenario, train, evaluate, distil, export."""
    if args.epochs < 1 or args.batch < 1 or args.window <= 3:
        raise ValueError("epochs and batch must be positive; window must exceed the aux horizons")
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device(args.device)

    episodes = load_episodes(args.data)
    groups = group_by_scenario(episodes)
    held_out = {name for name, group in groups.items() if any(episode.held_out for episode in group)}
    if not held_out or len(held_out) == len(groups):
        raise ValueError("validation data must mark at least one held-out and one training scenario")
    train_groups = {name: group for name, group in groups.items() if name not in held_out}
    train_episodes = [episode for group in train_groups.values() for episode in group]

    teacher = train_teacher(train_episodes, args, device)
    rmse = evaluate(teacher, groups, args, device)
    ratio = generalisation_ratio(rmse, held_out)

    student = distil_student(teacher, train_episodes, args, device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": teacher.cfg.__dict__,
            "state_dict": teacher.state_dict(),
            "scenario_rmse": rmse,
            "generalisation_ratio": ratio,
            "seed": args.seed,
        },
        args.out,
    )
    student_pt = args.out.with_name(args.out.stem + "_student.pt")
    torch.save(
        {"config": student.cfg.__dict__, "state_dict": student.state_dict(), "seed": args.seed},
        student_pt,
    )
    export_path = student_pt.with_suffix(".bin")
    export_student(student, export_path)
    return {
        "rmse": rmse,
        "ratio": ratio,
        "held_out": sorted(held_out),
        "teacher": args.out,
        "export": export_path,
    }


def main() -> None:
    args = build_parser().parse_args()
    result = run(args)
    held_out = set(result["held_out"])
    for name, value in sorted(result["rmse"].items()):
        split = "held-out" if name in held_out else "train"
        print(f"scenario={name} rmse={value:.6f} split={split}")
    print(
        f"ratio={result['ratio']:.4f} limit={GENERALISATION_LIMIT:.2f} "
        f"generalises={'yes' if result['ratio'] <= GENERALISATION_LIMIT else 'no'}"
    )
    print(f"teacher={result['teacher']} export={result['export']}")


if __name__ == "__main__":
    main()
