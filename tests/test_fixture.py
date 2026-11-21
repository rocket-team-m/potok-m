"""Cross-check fixture: an edge reader must produce the same three outputs."""

from pathlib import Path

import numpy as np
import torch

from potok_ai.export import load_export

FIXTURES = Path(__file__).parent / "fixtures"


def test_fixture_reproduces_expected_outputs_exactly() -> None:
    exported = load_export(FIXTURES / "tiny_model.bin")
    assert exported.version == 2
    assert exported.dilations == (1, 2, 4, 8, 16)
    model = exported.to_model()
    # Same fixed input as tests/fixtures/generate.py (no RNG involved).
    window = (torch.arange(50 * 8, dtype=torch.float32).reshape(1, 50, 8) - 200.0) / 100.0
    actual = model(window).detach().numpy()
    expected = np.load(FIXTURES / "tiny_expected.npy")
    assert np.array_equal(actual, expected)
