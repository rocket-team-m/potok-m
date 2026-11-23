import torch
from torch.nn import functional as F

from potok_ai.model import ModelConfig, PotokModel
from potok_ai.optim import MuonPlus, colrow, orthNS


def test_newton_schulz_compresses_the_spectrum():
    torch.manual_seed(0)
    matrix = torch.randn(16, 16)
    before = torch.linalg.svdvals(matrix)
    after = torch.linalg.svdvals(orthNS(matrix, steps=5))
    # The Jordan quintic flattens the spectrum toward 1; it is not an exact polar factor.
    assert after.min() > 0.5 and after.max() < 1.5
    assert (after.max() - after.min()) < (before.max() - before.min())


def test_muon_decreases_loss_on_cpu():
    torch.manual_seed(2)
    model = PotokModel(ModelConfig(width=16, rank=8))
    optimizer = MuonPlus(model.parameters(), lr=0.01, ns_steps=3)
    window = torch.randn(4, 16, 8)
    target = torch.randn(4, 3)

    def loss() -> torch.Tensor:
        return F.mse_loss(model(window), target)

    before = loss().item()
    for _ in range(10):
        optimizer.zero_grad()
        current = loss()
        current.backward()
        optimizer.step()
    assert loss().item() < before


def test_adamw_fallback_moves_vector_parameters():
    parameter = torch.nn.Parameter(torch.zeros(4))
    optimizer = MuonPlus([parameter], lr=0.01)
    parameter.grad = torch.ones(4)
    optimizer.step()
    assert torch.all(parameter != 0)


def test_colrow_is_finite_and_unit_scaled():
    torch.manual_seed(3)
    matrix = torch.randn(8, 4)
    normalized = colrow(matrix)
    assert normalized.shape == matrix.shape
    assert torch.isfinite(normalized).all()
    assert torch.allclose(normalized.norm(dim=-1), torch.ones(8), atol=1e-5)
