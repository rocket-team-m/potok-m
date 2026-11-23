# SPDX-License-Identifier: GPL-3.0-or-later
"""Trainer-facing privileged feature builder for POTOKTRJ v3 truth sidecars.

A v3 per-tick record is 26 f32 values (``sim/core/src/trajectory.rs``):

* slots 0..21 -- privileged truth: wind (0-2), gust (3-4), shear (5-7),
  position (8-10), velocity (11-13), quaternion (14-17), body rate
  (18-20), mass (21);
* slots 22..25 -- the reference optimizer's answer: ``g_theta_roll``,
  ``g_theta_pitch``, ``g_accel_y``, ``g_accel_x``.

The split happens before anything reaches a model. :func:`read_truth_sidecar`
returns the two halves as distinct arrays, so slots 22..25 can only ever be
labels. :class:`PrivilegedFeatureBuilder` compresses the 22 privileged floats
into a low-dimensional ``z_priv`` (8-d by default, RMA-style) for the
privileged teacher; the student tensor is checked and passed through
untouched, never extended with privileged or label columns.

Seeding is local: a top-level integer is mixed into named child streams and a
private ``torch.Generator`` initializes the encoder, so construction reads and
advances no global RNG. Different seeds give different projections; the same
seed gives bit-identical ones.

Missing or malformed truth is LOUD, never a zero fill: an absent sidecar, a
bad magic, a non-v3 version, a wrong stride, a truncated file or non-finite
privileged values all raise with the cause and the fix (regenerate with
``write_trajectory_v3``).
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .data import SMART_COND_FEATURES, SMART_FEATURES

# Binary layout, pinned to sim/core/src/trajectory.rs.
TRJ_MAGIC = b"POTOKTRJ"
TRJ_VERSION_V3 = 3
TRJ_HEADER_SIZE = 64
TRJ_HEADER = struct.Struct("<8sIIII")  # magic, version, episodes, frames, stride
TRJ_EPISODE_INDEX = struct.Struct("<Q")
TRJ_STRIDE_V3 = 26
TRJ_TICK = struct.Struct("<26f")

PRIV_START = 0
PRIV_END = 22  # exclusive: slots 22..25 are labels
LABEL_START = 22
LABEL_END = 26  # exclusive
PRIV_DIM = PRIV_END - PRIV_START  # 22
LABEL_DIM = LABEL_END - LABEL_START  # 4

TRUTH_SLOT_NAMES: tuple[str, ...] = (
    "pos_x", "pos_y", "pos_z",
    "vel_x", "vel_y", "vel_z",
    "quat_w", "quat_x", "quat_y", "quat_z",
    "omega_x", "omega_y", "omega_z",
    "mass_kg",
    "wind_x", "wind_y", "wind_z",
    "gust_start_s", "gust_amplitude",
    "shear_altitude_m", "shear_delta", "shear_thickness_m",
    "g_theta_roll", "g_theta_pitch", "g_accel_y", "g_accel_x",
)
PRIVILEGED_SLOT_NAMES = TRUTH_SLOT_NAMES[PRIV_START:PRIV_END]
LABEL_SLOT_NAMES = TRUTH_SLOT_NAMES[LABEL_START:LABEL_END]

STUDENT_INPUT_DIM = SMART_FEATURES + SMART_COND_FEATURES  # 14 sensor + 6 conditioning
DEFAULT_Z_DIM = 8  # RMA's latent width
DEFAULT_HIDDEN = 32
_EPS = 1e-6

__all__ = [
    "LABEL_DIM",
    "LABEL_SLOT_NAMES",
    "PRIVILEGED_SLOT_NAMES",
    "PRIV_DIM",
    "STUDENT_INPUT_DIM",
    "TRUTH_SLOT_NAMES",
    "PrivilegedEncoder",
    "PrivilegedFeatureBuilder",
    "TruthRecord",
    "child_seed",
    "read_truth_sidecar",
]


@dataclass(frozen=True)
class TruthRecord:
    """One episode's per-tick v3 truth, already split at the boundary.

    ``privileged`` is the only array a teacher may build features from.
    ``labels`` is the answer and is never used as an input to any forward.
    """

    episode_index: int
    privileged: np.ndarray  # (T, 22) f32, slots 0..21
    labels: np.ndarray  # (T, 4) f32, slots 22..25 -- labels only


def read_truth_sidecar(path: str | Path) -> list[TruthRecord]:
    """Read a POTOKTRJ v3 per-tick sidecar; raise on anything less than v3.

    The header version and stride are the authority. A v1/v2 sidecar has no
    ``g_accel`` labels and is refused instead of being zero-filled; a missing
    file is reported by :class:`FileNotFoundError` with the requested path.
    """
    sidecar = Path(path)
    raw = sidecar.read_bytes()
    if len(raw) < TRJ_HEADER_SIZE:
        raise ValueError(
            f"{sidecar}: truth sidecar header is {len(raw)} bytes, need {TRJ_HEADER_SIZE}; "
            "regenerate with write_trajectory_v3"
        )
    magic, version, episodes, frames, stride = TRJ_HEADER.unpack_from(raw)
    if magic != TRJ_MAGIC:
        raise ValueError(
            f"{sidecar}: bad truth magic {magic!r}, expected {TRJ_MAGIC!r}; "
            "point at a POTOKTRJ sidecar"
        )
    if version != TRJ_VERSION_V3:
        raise ValueError(
            f"{sidecar}: truth version {version} is not v3; v1/v2 have no g_accel "
            "labels and cannot back a privileged teacher -- regenerate with write_trajectory_v3"
        )
    if stride != TRJ_STRIDE_V3:
        raise ValueError(
            f"{sidecar}: truth stride {stride} does not match v3's {TRJ_STRIDE_V3}; "
            "regenerate with write_trajectory_v3"
        )
    if episodes == 0 or frames == 0:
        raise ValueError(f"{sidecar}: empty truth sidecar ({episodes} episodes x {frames} frames)")
    expected = TRJ_HEADER_SIZE + episodes * (TRJ_EPISODE_INDEX.size + frames * TRJ_TICK.size)
    if len(raw) != expected:
        raise ValueError(
            f"{sidecar}: truth sidecar is {len(raw)} bytes, expected {expected} "
            f"({episodes} episodes x {frames} ticks x {stride} f32); file truncated or corrupt"
        )

    records: list[TruthRecord] = []
    offset = TRJ_HEADER_SIZE
    for _ in range(episodes):
        (episode_index,) = TRJ_EPISODE_INDEX.unpack_from(raw, offset)
        offset += TRJ_EPISODE_INDEX.size
        ticks = np.frombuffer(raw, dtype="<f4", count=frames * stride, offset=offset)
        offset += frames * stride * np.dtype("<f4").itemsize
        grid = ticks.reshape(frames, stride)
        privileged = np.ascontiguousarray(grid[:, PRIV_START:PRIV_END], dtype=np.float32)
        labels = np.ascontiguousarray(grid[:, LABEL_START:LABEL_END], dtype=np.float32)
        if not np.isfinite(privileged).all():
            raise ValueError(
                f"{sidecar}: episode {episode_index} has non-finite privileged truth; "
                "regenerate the dataset rather than training on corrupt truth"
            )
        records.append(TruthRecord(episode_index, privileged, labels))
    return records


def child_seed(seed: int, name: str) -> int:
    """Deterministic named child stream, stable across processes and versions."""
    digest = hashlib.sha256(f"{int(seed)}:{name}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


class PrivilegedEncoder(torch.nn.Module):
    """22 -> ``z_dim`` projection with a locally seeded, checkpointable normalizer.

    Weights are created directly from a private generator: ``nn.Linear``'s
    constructor would advance the global RNG, which this module never touches.
    """

    def __init__(
        self,
        *,
        seed: int,
        mean: np.ndarray,
        std: np.ndarray,
        z_dim: int = DEFAULT_Z_DIM,
        hidden: int = DEFAULT_HIDDEN,
    ) -> None:
        super().__init__()
        if not 1 <= z_dim < PRIV_DIM:
            raise ValueError(f"z_dim must be in [1, {PRIV_DIM - 1}], got {z_dim}; z_priv is a bottleneck")
        if hidden < 1:
            raise ValueError(f"hidden must be positive, got {hidden}")
        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)
        if mean.shape != (PRIV_DIM,) or std.shape != (PRIV_DIM,):
            raise ValueError(f"mean/std must have shape ({PRIV_DIM},), got {mean.shape}/{std.shape}")

        generator = torch.Generator().manual_seed(int(seed))
        self.w1 = torch.nn.Parameter(torch.empty(hidden, PRIV_DIM).normal_(0.0, 0.1, generator=generator))
        self.b1 = torch.nn.Parameter(torch.zeros(hidden))
        self.w2 = torch.nn.Parameter(torch.empty(z_dim, hidden).normal_(0.0, 0.1, generator=generator))
        self.b2 = torch.nn.Parameter(torch.zeros(z_dim))
        self.register_buffer("mean", torch.from_numpy(mean.copy()))
        self.register_buffer("std", torch.from_numpy(std.copy()))
        self.z_dim = z_dim

    def forward(self, privileged: torch.Tensor) -> torch.Tensor:
        if privileged.ndim < 1 or privileged.shape[-1] != PRIV_DIM:
            raise ValueError(
                f"privileged input must have {PRIV_DIM} fields (slots 0..21), got shape "
                f"{tuple(privileged.shape)}; labels are never a feature"
            )
        x = (privileged.to(torch.float32) - self.mean) / self.std
        h = torch.tanh(x @ self.w1.T + self.b1)
        return h @ self.w2.T + self.b2


class PrivilegedFeatureBuilder:
    """Turns a v3 truth sidecar into teacher-only ``z_priv`` features.

    Normalization is fit on the sidecar handed to this builder -- the caller
    passes the train split's sidecar; held-out truth is never used to fit it.
    The encoder and its normalizer live in ``builder.encoder`` so a trainer can
    checkpoint them with the rest of the teacher.
    """

    def __init__(
        self,
        sidecar: str | Path,
        *,
        seed: int,
        z_dim: int = DEFAULT_Z_DIM,
        student_dim: int = STUDENT_INPUT_DIM,
    ) -> None:
        self.records = read_truth_sidecar(sidecar)
        self.seed = int(seed)
        self.z_dim = int(z_dim)
        self.student_dim = int(student_dim)
        privileged = np.concatenate([record.privileged for record in self.records], axis=0)
        mean = privileged.mean(axis=0, dtype=np.float64)
        std = privileged.std(axis=0, dtype=np.float64)
        # ponytail: constant channels stay unscaled instead of becoming NaN/inf.
        std = np.where(std < _EPS, 1.0, std)
        self.encoder = PrivilegedEncoder(
            seed=child_seed(self.seed, "privileged:encoder"),
            mean=mean,
            std=std,
            z_dim=self.z_dim,
        )

    def _record(self, episode_index: int) -> TruthRecord:
        for record in self.records:
            if record.episode_index == episode_index:
                return record
        available = [record.episode_index for record in self.records]
        raise ValueError(f"episode {episode_index} not in truth sidecar; available: {available}")

    def z_priv(self, episode_index: int) -> torch.Tensor:
        """Teacher-only latent ``(T, z_dim)``; never derived from label slots."""
        return self.encoder(torch.from_numpy(self._record(episode_index).privileged))

    def labels(self, episode_index: int) -> torch.Tensor:
        """Slots 22..25 ``(T, 4)`` for the loss only -- never a model input."""
        return torch.from_numpy(self._record(episode_index).labels.copy())

    def assert_student_clean(self, student_inputs: torch.Tensor) -> None:
        """Reject a student tensor grown with privileged fields or labels."""
        if student_inputs.ndim < 1 or student_inputs.shape[-1] != self.student_dim:
            raise ValueError(
                f"student inputs must carry exactly {self.student_dim} sensor features per tick, "
                f"got shape {tuple(student_inputs.shape)}; privileged truth and labels are "
                "teacher-only and must not be concatenated into the student"
            )
