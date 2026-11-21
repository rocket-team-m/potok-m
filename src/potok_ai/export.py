"""Portable binary export of a Potok-M model (format v2; see FORMAT.md).

All integers are little-endian. The header carries the full architecture so a
foreign reader never has to guess the causal layout. Ternary factors are stored
as codes plus their scale; the effective inference weight is ``codes * alpha``
and must NOT be quantised again.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .model import PotokModel

MAGIC = b"POTOKM1\0"
VERSION = 2
TSCT_LAYER = 1
FP32_LAYER = 2


@dataclass
class ExportedLayer:
    kind: int
    name: str
    in_features: int = 0
    out_features: int = 0
    rank: int = 0
    alpha_u: float = 0.0
    alpha_v: float = 0.0
    u_codes: np.ndarray = field(default_factory=lambda: np.empty(0, np.int8))
    scales: np.ndarray = field(default_factory=lambda: np.empty(0, np.float32))
    v_codes: np.ndarray = field(default_factory=lambda: np.empty(0, np.int8))
    tensor: np.ndarray | None = None


@dataclass
class ExportedModel:
    version: int
    in_features: int
    width: int
    out_features: int
    rank: int
    kernel: int
    g_max: float
    dilations: tuple[int, ...]
    flags: int
    layers: dict[str, ExportedLayer]
    checksum: bytes

    def to_model(self) -> PotokModel:
        """Rebuild an eval-mode model whose forward is bit-identical to the export.

        The exported U/V are the effective ternary weights; we load them into a
        model with ``ternary=False`` so the straight-through quantiser does not
        apply a second, different scale at inference.
        """
        from .model import ModelConfig

        config = ModelConfig(
            in_features=self.in_features,
            width=self.width,
            out_features=self.out_features,
            rank=self.rank,
            kernel=self.kernel,
            dilations=self.dilations,
            g_max=self.g_max,
            ternary=False,
            act4bit=bool(self.flags & 2),
        )
        model = PotokModel(config)
        state = model.state_dict()
        for name, layer in self.layers.items():
            if layer.kind == TSCT_LAYER:
                u = torch.from_numpy(layer.u_codes.astype(np.float32).reshape(layer.in_features, layer.rank))
                v = torch.from_numpy(layer.v_codes.astype(np.float32).reshape(layer.out_features, layer.rank))
                state[f"{name}.U"].copy_(u * layer.alpha_u)
                state[f"{name}.V"].copy_(v * layer.alpha_v)
                state[f"{name}.s"].copy_(torch.from_numpy(layer.scales.copy()))
            elif layer.tensor is not None:
                state[name].copy_(torch.from_numpy(layer.tensor.copy()))
        model.load_state_dict(state)
        return model.eval()


class _Writer:
    def __init__(self) -> None:
        self.data = bytearray()

    def raw(self, value: bytes) -> None:
        self.data.extend(value)

    def u8(self, value: int) -> None:
        self.raw(struct.pack("<B", value))

    def u32(self, value: int) -> None:
        self.raw(struct.pack("<I", value))

    def i32(self, value: int) -> None:
        self.raw(struct.pack("<i", value))

    def f32(self, value: float) -> None:
        self.raw(struct.pack("<f", value))

    def array(self, values, dtype: str, code: str) -> None:
        if hasattr(values, "detach"):
            values = values.detach().cpu().numpy()
        array = np.asarray(values, dtype=dtype).reshape(-1)
        self.raw(struct.pack(f"<{array.size}{code}", *array.tolist()))


class _Reader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.offset = 0

    def raw(self, count: int) -> bytes:
        chunk = self.data[self.offset : self.offset + count]
        if len(chunk) != count:
            raise ValueError("export file truncated")
        self.offset += count
        return chunk

    def u8(self) -> int:
        return self.raw(1)[0]

    def u32(self) -> int:
        return struct.unpack("<I", self.raw(4))[0]

    def i32(self) -> int:
        return struct.unpack("<i", self.raw(4))[0]

    def f32(self) -> float:
        return struct.unpack("<f", self.raw(4))[0]

    def array(self, count: int, dtype: str, code: str) -> np.ndarray:
        size = count * struct.calcsize(code)
        return np.frombuffer(self.raw(size), dtype=dtype).copy()


def export_model(model: PotokModel, path: str | Path) -> Path:
    """Write ``model`` to ``path`` in the documented binary format."""
    config = model.cfg
    writer = _Writer()
    writer.raw(MAGIC)
    writer.u32(VERSION)
    writer.u32(config.in_features)
    writer.u32(config.width)
    writer.u32(config.out_features)
    writer.u32(config.rank)
    writer.u32(config.kernel)
    writer.u32(len(config.dilations))
    writer.f32(config.g_max)
    for dilation in config.dilations:
        writer.u32(dilation)
    writer.u32((1 if config.ternary else 0) | (2 if config.act4bit else 0))

    factors = model.quantized_factors()
    owned = {
        name for name, parameter in model.named_parameters()
        if any(name.startswith(module + ".") for module in factors)
    }
    others = {
        name: parameter.detach().cpu().numpy()
        for name, parameter in model.named_parameters()
        if name not in owned
    }
    writer.u32(len(factors) + len(others))

    for name, (u, alpha_u, scales, v, alpha_v) in factors.items():
        in_features, rank = u.shape
        out_features = v.shape[0]
        encoded = name.encode("utf-8")
        writer.u8(TSCT_LAYER)
        writer.u32(len(encoded))
        writer.raw(encoded)
        writer.u32(in_features)
        writer.u32(out_features)
        writer.u32(rank)
        writer.f32(alpha_u)
        writer.f32(alpha_v)
        writer.array(u, np.int8, "b")
        writer.array(scales, np.float32, "f")
        writer.array(v, np.int8, "b")

    for name, tensor in others.items():
        encoded = name.encode("utf-8")
        writer.u8(FP32_LAYER)
        writer.u32(len(encoded))
        writer.raw(encoded)
        writer.u32(tensor.ndim)
        for dimension in tensor.shape:
            writer.i32(int(dimension))
        writer.array(tensor.astype(np.float32), np.float32, "f")

    payload = bytes(writer.data)
    Path(path).write_bytes(payload + hashlib.sha256(payload).digest())
    return Path(path)


def load_export(path: str | Path) -> ExportedModel:
    """Load and verify an exported model, raising on a bad magic or checksum."""
    data = Path(path).read_bytes()
    if len(data) < len(MAGIC) + 32 or data[: len(MAGIC)] != MAGIC:
        raise ValueError("not a Potok-M export")
    payload, checksum = data[:-32], data[-32:]
    if hashlib.sha256(payload).digest() != checksum:
        raise ValueError("export checksum mismatch")

    reader = _Reader(payload)
    reader.raw(len(MAGIC))
    version = reader.u32()
    if version != VERSION:
        raise ValueError(f"unsupported export version {version}")
    in_features, width, out_features, rank, kernel, n_blocks = (
        reader.u32() for _ in range(6)
    )
    g_max = reader.f32()
    dilations = tuple(reader.u32() for _ in range(n_blocks))
    flags = reader.u32()
    count = reader.u32()

    layers: dict[str, ExportedLayer] = {}
    for _ in range(count):
        kind = reader.u8()
        name = reader.raw(reader.u32()).decode("utf-8")
        if kind == TSCT_LAYER:
            layer_in, layer_out, layer_rank = reader.u32(), reader.u32(), reader.u32()
            alpha_u, alpha_v = reader.f32(), reader.f32()
            layers[name] = ExportedLayer(
                kind=kind,
                name=name,
                in_features=layer_in,
                out_features=layer_out,
                rank=layer_rank,
                alpha_u=alpha_u,
                alpha_v=alpha_v,
                u_codes=reader.array(layer_in * layer_rank, np.int8, "b"),
                scales=reader.array(layer_rank, np.float32, "f"),
                v_codes=reader.array(layer_out * layer_rank, np.int8, "b"),
            )
        elif kind == FP32_LAYER:
            ndim = reader.u32()
            shape = tuple(reader.i32() for _ in range(ndim))
            size = 1
            for dimension in shape:
                size *= dimension
            layers[name] = ExportedLayer(
                kind=kind, name=name, tensor=reader.array(size, np.float32, "f").reshape(shape)
            )
        else:
            raise ValueError(f"unknown export layer kind {kind}")
    if reader.offset != len(payload):
        raise ValueError("trailing bytes in export")
    return ExportedModel(
        version, in_features, width, out_features, rank, kernel, g_max, dilations, flags,
        layers, checksum,
    )
