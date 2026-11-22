"""Stateful student for speculative dispatch (SPEC_STUDENT.md, v1).

One frame per call: 8 sensor floats plus a phase code and a flags word. Two
recurrent blocks of width 64 keep one hidden state each between calls; their
weights are shared across the 1..3 refinement iterations of a single tick.
The quantisation primitives are reused from ``model.py`` so an export stores
the same ternary codes plus fp32 scales as the windowed teacher.

For SPEC_SMART.md the frame is 20 floats: 14 physics features followed by
6 vehicle-conditioning numbers (concatenated into the stem input). The export
header records the full width, so an edge reader sees 20 instead of 10.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .model import TSCTLinear, ternary_codes

SENSOR_FEATURES = 8
PHASE_INDEX = 8
FLAGS_INDEX = 9
FRAME_FEATURES = 10  # legacy: 8 sensors + phase + flags
PHYSICS_FEATURES = 14  # smart: 8 raw + 6 physics features
COND_FEATURES = 6  # mass, area, cd0, static margin, thrust, gyro noise
SMART_FRAME_FEATURES = PHYSICS_FEATURES + COND_FEATURES  # 20
MAX_LOOPS = 3


def smart_student_config(seed: int = 0, **overrides) -> StudentConfig:
    """SPEC_SMART.md small model: 2 blocks, width 64, 2 loops, decay 0.8."""
    settings = {
        "in_features": SMART_FRAME_FEATURES,
        "width": 64,
        "out_features": 2,
        "rank": 16,
        "n_blocks": 2,
        "loops": 2,
        "decay": 0.8,
        "seed": seed,
    }
    settings.update(overrides)
    return StudentConfig(**settings)


@dataclass
class StudentConfig:
    """Architecture of the stateful student. All fields are plain data."""

    in_features: int = FRAME_FEATURES  # 8 sensors + phase + flags
    width: int = 64
    out_features: int = 2
    rank: int = 16
    n_blocks: int = 2
    max_loops: int = MAX_LOOPS
    loops: int = MAX_LOOPS
    decay: float = 0.8
    g_max: float = 1.0
    ternary: bool = True
    act4bit: bool = True
    seed: int = 0


class _StateBlock(nn.Module):
    """One weight-shared recurrent block with a ternary TSCT projection."""

    def __init__(self, width: int, rank: int, ternary: bool, act4bit: bool) -> None:
        super().__init__()
        self.rec_h = nn.Linear(width, width, bias=True)
        self.rec_x = nn.Linear(width, width, bias=True)
        self.proj = TSCTLinear(
            width, width, rank, ternary=ternary, act4bit=act4bit
        )

    def forward(
        self, x: torch.Tensor, h_prev: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape != h_prev.shape:
            raise ValueError("block input and hidden state must have the same shape")
        candidate = torch.tanh(self.rec_x(x) + self.rec_h(h_prev))
        h = 0.8 * h_prev + 0.2 * candidate
        return self.proj(h), h


class StatefulTCN(nn.Module):
    """Frame-by-frame student with a persistent hidden state per block."""

    def __init__(self, cfg: StudentConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or StudentConfig()
        c = self.cfg
        if c.in_features < SENSOR_FEATURES:
            raise ValueError("a frame needs at least the 8 sensor features")
        if c.n_blocks < 1 or c.loops < 1 or c.loops > MAX_LOOPS:
            raise ValueError("n_blocks must be >=1 and loops must be in 1..3")

        self.stem = nn.Linear(c.in_features, c.width, bias=True)
        self.blocks = nn.ModuleList(
            _StateBlock(c.width, c.rank, c.ternary, c.act4bit)
            for _ in range(c.n_blocks)
        )
        self.norm = nn.LayerNorm(c.width)
        self.alloc_logits = nn.Parameter(torch.empty(MAX_LOOPS))
        self.head_corr = nn.Linear(c.width, c.out_features, bias=True)
        self.head_conf = nn.Linear(c.width, 1, bias=True)
        self.g_max = nn.Parameter(torch.empty(1))
        # Non-persistent: the exported initial state is always zeros.
        self.register_buffer("h", torch.zeros(c.n_blocks, c.width), persistent=False)
        self.last_conf_logit: torch.Tensor | None = None
        self._init_parameters()

    def _init_parameters(self) -> None:
        """Seed every parameter from a local generator; the global RNG is unused."""
        gen = torch.Generator().manual_seed(self.cfg.seed)
        width = self.cfg.width
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                if name == "g_max":
                    parameter.fill_(self.cfg.g_max)
                elif name == "alloc_logits":
                    parameter.zero_()
                elif name.endswith(".s"):
                    parameter.fill_(1.0)
                elif name.endswith(".bias"):
                    parameter.zero_()
                elif ".proj.U" in name or ".proj.V" in name:
                    nn.init.normal_(parameter, 0.0, self.cfg.rank**-0.5, generator=gen)
                else:
                    nn.init.normal_(parameter, 0.0, width**-0.5, generator=gen)

    def _as_frame(self, frame) -> torch.Tensor:
        tensor = torch.as_tensor(frame, dtype=torch.float32).reshape(-1)
        need = self.cfg.in_features
        if tensor.numel() > need:
            raise ValueError(f"frame has {tensor.numel()} values, expected <= {need}")
        if tensor.numel() < need:
            padding = torch.zeros(need - tensor.numel(), dtype=tensor.dtype)
            # A missing flags word means "valid sensor" for the spec's dispatch.
            if tensor.numel() <= FLAGS_INDEX < need:
                padding[FLAGS_INDEX - tensor.numel()] = 1.0
            tensor = torch.cat([tensor, padding])
        return tensor

    def reset_state(self) -> None:
        """Zero the hidden state of every block (episode / phase boundary)."""
        self.h.zero_()

    def step(self, frame) -> tuple[torch.Tensor, torch.Tensor]:
        """Consume one frame and return ``(corrections[2], confidence[])``.

        The previous state is detached on entry, so the autograd graph never
        spans ticks: memory stays bounded over arbitrarily long streams.
        """
        x = self._as_frame(frame)
        # Clone so the graph never holds a view of the buffer we overwrite below.
        state = [self.h[i].detach().clone() for i in range(self.cfg.n_blocks)]
        allocation = torch.softmax(self.alloc_logits[: self.cfg.loops], dim=0)

        source = self.stem(x)
        current = source  # direct current-frame path; the recurrent state alone
        # converges to a fixed point and washes out the frame that determines error.
        total = None
        for iteration in range(self.cfg.loops):
            for index, block in enumerate(self.blocks):
                source, state[index] = block(source, state[index])
            weighted = allocation[iteration] * (self.cfg.decay**iteration) * source
            total = weighted if total is None else total + weighted

        for index, value in enumerate(state):
            self.h[index] = value.detach()

        z = self.norm(current + 0.25 * (total + source))
        self.last_z = z
        bound = torch.tanh(self.g_max.abs())
        corrections = torch.tanh(self.head_corr(z)) * bound
        logit = self.head_conf(z).squeeze(-1)
        self.last_conf_logit = logit
        return corrections, torch.sigmoid(logit)

    def confidence_logit(self) -> torch.Tensor:
        """Pre-sigmoid confidence logit of the last tick, for calibration.

        The export already stores ``head_conf.weight`` / ``head_conf.bias`` as
        fp32 layer records, so the same scalar is reproducible from a binary
        export without any format change.
        """
        if self.last_conf_logit is None:
            raise ValueError("call step() before asking for a confidence logit")
        return self.last_conf_logit

    def warmup(self, window) -> None:
        """Initialise the state by replaying a window without building a graph."""
        frames = torch.as_tensor(window, dtype=torch.float32)
        if frames.ndim != 2:
            raise ValueError("warmup expects a window of shape (time, features)")
        with torch.no_grad():
            self.reset_state()
            for frame in frames:
                self.step(frame)

    def quantized_factors(self) -> dict[str, tuple[torch.Tensor, float, torch.Tensor, float, torch.Tensor]]:
        """Return exportable TSCT factors: (U codes, alpha_u, s, V codes, alpha_v)."""
        result = {}
        for name, module in self.named_modules():
            if isinstance(module, TSCTLinear):
                u, alpha_u = ternary_codes(module.U.detach())
                v, alpha_v = ternary_codes(module.V.detach())
                result[name] = (
                    u.to(torch.int8).cpu(),
                    float(alpha_u),
                    module.s.detach().cpu(),
                    v.to(torch.int8).cpu(),
                    float(alpha_v),
                )
        return result
