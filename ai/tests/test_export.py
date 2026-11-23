import numpy as np
import pytest
import torch

from potok_ai.export import export_model, load_export
from potok_ai.model import ModelConfig, PotokModel


def test_export_load_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = PotokModel(ModelConfig(width=16, rank=8))
    path = export_model(model, tmp_path / "model.bin")
    exported = load_export(path)

    assert exported.width == 16 and exported.rank == 8
    factors = model.quantized_factors()
    name = next(iter(factors))
    layer = exported.layers[name]
    expected_u, alpha_u, scales, expected_v, alpha_v = factors[name]
    assert np.array_equal(layer.u_codes.reshape(-1), expected_u.numpy().reshape(-1))
    assert np.array_equal(layer.scales, scales.numpy())
    assert np.array_equal(layer.v_codes.reshape(-1), expected_v.numpy().reshape(-1))
    assert np.isclose(layer.alpha_u, alpha_u, atol=1e-6)
    assert np.isclose(layer.alpha_v, alpha_v, atol=1e-6)
    assert np.all(np.isin(layer.u_codes, [-1, 0, 1]))
    assert any(value.tensor is not None for value in exported.layers.values())


def test_exported_model_reproduces_forward_exactly(tmp_path):
    torch.manual_seed(11)
    model = PotokModel(ModelConfig(width=16, rank=8)).eval()
    window = torch.arange(16 * 8, dtype=torch.float32).reshape(1, 16, 8) / 50.0
    expected = model(window)
    reloaded = load_export(export_model(model, tmp_path / "m.bin")).to_model()
    assert torch.equal(reloaded(window), expected)


def test_export_rejects_tampering(tmp_path):
    model = PotokModel(ModelConfig(width=16, rank=8))
    path = export_model(model, tmp_path / "model.bin")
    raw = bytearray(path.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="checksum"):
        load_export(path)
