"""Training CLI.

Stage 1 (pretrain): masked Fast-JEPA prediction of a full shifted latent window.
Stage 2 (distill): imitate the reference corrections plus a jerk penalty.
Muon+ is the default optimizer; plain AdamW is an explicit comparison option.
Synthetic data is the default so the CPU smoke test needs no simulator files.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.optim import AdamW, Optimizer

from .data import WindowDataset, load_episodes, make_synthetic_dataset
from .jepa import EMATeacher, JEPAPredictor, jepa_loss
from .model import ModelConfig, PotokModel
from .optim import MuonPlus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="potok-train")
    parser.add_argument("--stage", choices=["pretrain", "distill"], default="pretrain")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--out", type=Path, default=Path("model.pt"))
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--ema-momentum", type=float, default=0.99)
    parser.add_argument("--jerk-weight", type=float, default=0.01)
    parser.add_argument("--opt", choices=["muon", "adamw"], default="muon")
    parser.add_argument("--ns-steps", type=int, default=5)
    return parser


def _dataset(args: argparse.Namespace) -> WindowDataset:
    if args.synthetic or args.data is None:
        return make_synthetic_dataset(window=args.seq_len, limit=args.limit)
    return WindowDataset(load_episodes(args.data), window=args.seq_len, limit=args.limit)


def _batches(dataset: WindowDataset, batch: int) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Materialise the tiny CPU dataset once; the hot loop only slices views."""
    if not dataset.samples:
        return
    windows = torch.stack([torch.as_tensor(sample[0]) for sample in dataset.samples])
    targets = torch.stack([torch.as_tensor(sample[1]) for sample in dataset.samples])
    for start in range(0, len(windows), batch):
        yield windows[start : start + batch], targets[start : start + batch]


def _optimizer(parameters, args: argparse.Namespace) -> Optimizer:
    if args.opt == "adamw":
        return AdamW(parameters, lr=args.lr)
    return MuonPlus(parameters, lr=args.lr, ns_steps=args.ns_steps)


def _save(model: PotokModel, args: argparse.Namespace, stage: str, loss: float) -> None:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": stage,
            "config": vars(args) | {"out": str(args.out)},
            "model_config": model.cfg.__dict__,
            "state_dict": model.state_dict(),
            "loss": loss,
        },
        args.out,
    )


def pretrain(args: argparse.Namespace) -> float:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    config = ModelConfig(width=args.width, rank=args.rank)
    model = PotokModel(config).to(device)
    teacher = EMATeacher(model, momentum=args.ema_momentum)
    predictor = JEPAPredictor(config.width).to(device)
    optimizer = _optimizer(list(model.parameters()) + list(predictor.parameters()), args)

    model.train()
    predictor.train()
    final_loss = float("nan")
    for _ in range(args.epochs):
        for window, _ in _batches(_dataset(args), args.batch):
            window = window.to(device)
            losses = jepa_loss(model, teacher, window, predictor)
            optimizer.zero_grad(set_to_none=False)
            losses.total.backward()
            optimizer.step()
            teacher.update(model)
            final_loss = float(losses.total.detach())
    _save(model, args, "pretrain", final_loss)
    return final_loss


def distill(args: argparse.Namespace) -> float:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = PotokModel(ModelConfig(width=args.width, rank=args.rank)).to(device)
    optimizer = _optimizer(model.parameters(), args)

    model.train()
    final_loss = float("nan")
    for _ in range(args.epochs):
        for window, reference in _batches(_dataset(args), args.batch):
            window, reference = window.to(device), reference.to(device)
            corrections = model(window)[:, :2]
            imitation = F.mse_loss(corrections, reference)
            per_frame = model.head(model.forward_features(window))[:, :, :2]
            delta = per_frame[:, 2:] - 2 * per_frame[:, 1:-1] + per_frame[:, :-2]
            jerk = delta.square().mean() if delta.numel() else corrections.sum() * 0.0
            loss = imitation + args.jerk_weight * jerk
            optimizer.zero_grad(set_to_none=False)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach())
    _save(model, args, "distill", final_loss)
    return final_loss


def run(args: argparse.Namespace) -> float:
    if args.epochs < 1 or args.batch < 1:
        raise ValueError("epochs and batch must be positive")
    if args.device == "cpu":
        torch.set_num_threads(1)
    return pretrain(args) if args.stage == "pretrain" else distill(args)


def main() -> None:
    args = build_parser().parse_args()
    loss = run(args)
    print(f"stage={args.stage} opt={args.opt} epochs={args.epochs} loss={loss:.6f} out={args.out}")


if __name__ == "__main__":
    main()
