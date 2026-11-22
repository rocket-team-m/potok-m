"""Train the privileged LoopWM teacher, then distil the 20-input student.

The split comes from the v2 validation metadata: scenarios marked held-out are
never trained on, and the acceptance number is mean held-out RMSE / mean
training RMSE (limit 1.25). The simulator's reference corrections supervise the
teacher's correction head; the student only sees the teacher's outputs plus a
small weight on the same reference, and is exported as v2 model_kind=2.

LoopWM is an alternative to SmartTeacher: nothing here imports or changes
``train_smart``'s behaviour, and the two CLIs can coexist.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW

from .curriculum import STAGES, Curriculum
from .data import SmartWindowDataset, group_by_scenario, load_episodes
from .export import export_student
from .jepa import EMATeacher
from .loopwm import LoopWM, LoopWMConfig
from .physics import final_command, physics_baseline
from .train_smart import (
    GENERALISATION_LIMIT,
    _batches,
    distil_student,
    evaluate,
    generalisation_ratio,
    normalized_correction_loss,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="potok-train-loopwm")
    parser.add_argument("--data", type=Path, required=True, help="v2 validation file or directory")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    parser.add_argument("--out", type=Path, default=Path("loopwm.pt"))
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="cap windows / student updates")
    parser.add_argument("--eval-limit", type=int, default=None, help="cap windows per scenario")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--student-lr", type=float, default=1e-3)
    parser.add_argument("--target-weight", type=float, default=0.1)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--max-iters", type=int, default=6)
    parser.add_argument("--exit-threshold", type=float, default=1e-3)
    parser.add_argument("--latent-weight", type=float, default=0.05)
    return parser


def train_teacher(episodes, args: argparse.Namespace, device: torch.device) -> LoopWM:
    """Train the LoopWM teacher on every non-held-out scenario."""
    config = LoopWMConfig(
        width=args.width,
        heads=args.heads,
        max_iters=args.max_iters,
        exit_threshold=args.exit_threshold,
        latent_weight=args.latent_weight,
    )
    model = LoopWM(config).to(device)
    target_encoder = EMATeacher(model.encoder, config.ema_momentum)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    residual = getattr(args, "residual", False)
    curriculum = (
        Curriculum(seed=args.seed, epochs_per_stage=max(1, args.epochs // len(STAGES)))
        if getattr(args, "curriculum", False)
        else None
    )

    model.train()
    for epoch in range(args.epochs):
        chosen = curriculum.select(episodes, epoch) if curriculum else episodes
        dataset = SmartWindowDataset(chosen, window=args.window, limit=args.limit)
        if not dataset.samples:
            continue
        for windows, conditions, targets in _batches(dataset, args.batch):
            latent, _ = model.refine(windows, conditions)
            output = model.head_from_latent(latent)
            command = final_command(
                output[:, :2], physics_baseline(windows[:, -1, :]), residual
            )
            # `targets` are the privileged reference corrections (true wind).
            main = normalized_correction_loss(command, targets)
            latent_loss = model.latent_prediction_loss(
                windows, conditions, target_encoder, latent=latent
            )
            loss = main + config.latent_weight * latent_loss
            optimizer.zero_grad(set_to_none=False)
            loss.backward()
            optimizer.step()
            target_encoder.update(model.encoder)
    return model


def run(args: argparse.Namespace) -> dict:
    """Full pipeline: scenario split, train, evaluate, distil, export."""
    if args.epochs < 1 or args.batch < 1 or args.window <= max(LoopWMConfig().horizons):
        raise ValueError("epochs and batch must be positive; window must exceed the latent horizons")
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device(args.device)

    episodes = load_episodes(args.data)
    groups = group_by_scenario(episodes)
    held_out = {name for name, group in groups.items() if any(e.held_out for e in group)}
    if not held_out or len(held_out) == len(groups):
        raise ValueError("validation data must mark at least one held-out and one training scenario")
    train_episodes = [
        episode for name, group in groups.items() if name not in held_out for episode in group
    ]

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
