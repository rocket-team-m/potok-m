"""CPU micro-benchmark: frames/s and parameter count.

    uv run python benchmark.py --stage train --steps 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import torch
from torch.optim import AdamW

from potok_ai.jepa import EMATeacher, JEPAPredictor, jepa_loss
from potok_ai.model import ModelConfig, PotokModel, count_parameters
from potok_ai.optim import MuonPlus


def main() -> None:
    parser = argparse.ArgumentParser(prog="potok-bench")
    parser.add_argument("--stage", choices=["forward", "train"], default="train")
    parser.add_argument("--opt", choices=["muon", "adamw"], default="muon")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(1)
    model = PotokModel(ModelConfig(width=args.width, rank=args.rank))
    window = torch.randn(args.batch, args.seq_len, 8)

    if args.stage == "train":
        teacher = EMATeacher(model, momentum=0.99)
        predictor = JEPAPredictor(args.width)
        parameters = list(model.parameters()) + list(predictor.parameters())
        optimizer = (
            AdamW(parameters, lr=0.01)
            if args.opt == "adamw"
            else MuonPlus(parameters, lr=0.01)
        )

        def step() -> None:
            loss = jepa_loss(model, teacher, window, predictor).total
            optimizer.zero_grad(set_to_none=False)
            loss.backward()
            optimizer.step()
            teacher.update(model)

    else:

        def step() -> None:
            model(window).sum().backward()

    for _ in range(3):
        step()
    started = time.perf_counter()
    for _ in range(args.steps):
        step()
    elapsed = time.perf_counter() - started

    frames = args.batch * args.seq_len * args.steps
    print(
        f"stage={args.stage} opt={args.opt} params={count_parameters(model)} "
        f"ms_per_step={1000 * elapsed / args.steps:.2f} frames_per_s={frames / elapsed:.0f}"
    )


if __name__ == "__main__":
    main()
