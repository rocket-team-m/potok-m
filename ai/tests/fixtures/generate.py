"""Regenerate the deterministic export fixtures: `uv run python tests/fixtures/generate.py`.

The model is deliberately untrained and small; only init and export are involved.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from potok_ai.export import export_model
from potok_ai.model import ModelConfig, PotokModel

SEED = 1234
HERE = Path(__file__).resolve().parent


def fixture_window() -> torch.Tensor:
    # No RNG: any reader can reconstruct the exact input window.
    return (torch.arange(50 * 8, dtype=torch.float32).reshape(1, 50, 8) - 200.0) / 100.0


def main() -> None:
    torch.manual_seed(SEED)
    model = PotokModel(ModelConfig(width=16, rank=8)).eval()
    window = fixture_window()
    expected = model(window).detach().numpy()
    export_model(model, HERE / "tiny_model.bin")
    np.save(HERE / "tiny_expected.npy", expected)
    print(f"wrote tiny_model.bin and tiny_expected.npy: output shape {expected.shape}")


if __name__ == "__main__":
    main()
