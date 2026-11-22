"""Distil the windowed teacher into the stateful student (CPU, SPEC_STUDENT.md).

The teacher scores each 50-frame window; the student sees one frame at a time
and is trained to match the teacher's correction at the window's last frame.
Its confidence target is the discrepancy |g_teacher - g_student|. State is
reset at episode boundaries and whenever the frame's phase code changes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.optim import AdamW

from .data import load_episodes
from .export import MAGIC, export_student, load_export
from .model import ModelConfig, PotokModel
from .student import FLAGS_INDEX, PHASE_INDEX, SENSOR_FEATURES, StatefulTCN, StudentConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="potok-distill-student")
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("student.pt"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--loops", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--conf-weight", type=float, default=1.0)
    parser.add_argument("--conf-temp", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def load_teacher(path: str | Path, device: torch.device) -> PotokModel:
    """Load a windowed teacher from a torch checkpoint or a v2 binary export."""
    file = Path(path)
    if file.read_bytes()[: len(MAGIC)] == MAGIC:
        model = load_export(file).to_model()
    else:
        try:
            checkpoint = torch.load(file, map_location="cpu", weights_only=False)
        except TypeError:  # torch < 2.6 has no weights_only keyword
            checkpoint = torch.load(file, map_location="cpu")
        model = PotokModel(ModelConfig(**checkpoint["model_config"]))
        model.load_state_dict(checkpoint["state_dict"])
    return model.eval().to(device).requires_grad_(False)


def phase_of(row) -> int:
    """Phase code carried in column 8, or 0 for sensor-only frames."""
    return int(row[PHASE_INDEX]) if len(row) > PHASE_INDEX else 0


def flags_valid(row) -> bool:
    """A missing flags word means the sensors are valid."""
    return bool(row[FLAGS_INDEX]) if len(row) > FLAGS_INDEX else True


def train_episode(
    student: StatefulTCN,
    teacher: PotokModel,
    frames,
    optimizer: AdamW,
    args: argparse.Namespace,
) -> tuple[int, int, torch.Tensor | None]:
    """Run one episode; return (windows trained, state resets, last loss)."""
    device = next(student.parameters()).device
    student.reset_state()
    resets = 1
    windows = 0
    last_loss = None
    previous_phase = None
    for tick, row in enumerate(frames):
        phase = phase_of(row)
        if previous_phase is not None and phase != previous_phase:
            student.reset_state()
            resets += 1
        previous_phase = phase

        corrections, confidence = student.step(row)
        if tick + 1 < args.window or not flags_valid(row):
            continue

        window = torch.as_tensor(
            frames[tick + 1 - args.window : tick + 1, :SENSOR_FEATURES],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        with torch.no_grad():
            expected = teacher(window)[0, :2]
            discrepancy = (expected - corrections.detach()).abs().mean()
            target = torch.exp(-discrepancy / args.conf_temp)

        imitation = F.mse_loss(corrections, expected)
        calibration = F.mse_loss(confidence, target)
        loss = imitation + args.conf_weight * calibration
        optimizer.zero_grad(set_to_none=False)
        loss.backward()
        optimizer.step()
        windows += 1
        last_loss = loss.detach()
        if args.limit is not None and windows >= args.limit:
            break
    return windows, resets, last_loss


def run(args: argparse.Namespace) -> float:
    """Train the student for ``args.epochs`` passes over ``args.data``."""
    if args.epochs < 1 or args.window < 1 or args.loops < 1:
        raise ValueError("epochs, window and loops must be positive")
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(1)

    teacher = load_teacher(args.teacher_checkpoint, device)
    student = StatefulTCN(
        StudentConfig(seed=args.seed, loops=args.loops)
    ).to(device)
    optimizer = AdamW(student.parameters(), lr=args.lr)
    episodes = load_episodes(args.data)

    student.train()
    final_loss = float("nan")
    total_windows = 0
    for _ in range(args.epochs):
        for episode in episodes:
            windows, _, last_loss = train_episode(
                student, teacher, episode.frames, optimizer, args
            )
            total_windows += windows
            if last_loss is not None:
                final_loss = float(last_loss)
            if args.limit is not None and total_windows >= args.limit:
                break
        if args.limit is not None and total_windows >= args.limit:
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": student.cfg.__dict__,
            "state_dict": student.state_dict(),
            "loss": final_loss,
            "seed": args.seed,
        },
        args.out,
    )
    export_path = (
        args.out.with_suffix(".bin")
        if args.out.suffix != ".bin"
        else args.out.with_name(args.out.stem + "_student.bin")
    )
    export_student(student, export_path)
    return final_loss


def main() -> None:
    args = build_parser().parse_args()
    loss = run(args)
    export_path = (
        args.out.with_suffix(".bin")
        if args.out.suffix != ".bin"
        else args.out.with_name(args.out.stem + "_student.bin")
    )
    print(
        f"stage=student epochs={args.epochs} loss={loss:.6f} "
        f"out={args.out} export={export_path}"
    )


if __name__ == "__main__":
    main()
