"""Episode loaders for Potok-M training data.

Canonical input: the simulator's ``validation.bin`` (``POTOKSIM`` magic), exactly
as documented in ``sim/README.md`` and written by ``potok_sim::validation``.
The older ``POTOK1\\0\\0`` layout and a bare float32 array are still accepted as
fallbacks for local synthetic fixtures.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

MAGIC = b"POTOK1\0\0"
HEADER = struct.Struct("<8sIIII")
VERSION = 1

SIM_MAGIC = b"POTOKSIM"
SIM_VERSION = 1
SIM_TARGETS = 4  # g_roll, g_pitch, confidence, ai_healthy
# magic, version, seed, count, frames, features, targets, rate, dt, duration, reserved
SIM_HEADER = struct.Struct("<8sIQ4I3f32s")
SIM_EPISODE = struct.Struct("<Q9f")  # index + 9 metadata floats


@dataclass
class Episode:
    """A single simulated flight: sensor frames and optional reference targets."""

    frames: np.ndarray  # (T, F) float32
    targets: np.ndarray | None = None  # (T, K) float32, K = 4 for validation sets
    index: int | None = None
    metadata: np.ndarray | None = None  # (9,) f32: mass, thrust, pitch, yaw, gust...


@dataclass
class ValidationHeader:
    seed: int
    episode_count: int
    frames_per_episode: int
    feature_count: int
    target_count: int
    frame_rate_hz: float
    dt: float
    duration_s: float


@dataclass
class ValidationData:
    header: ValidationHeader
    episodes: list[Episode]


def read_validation(path: str | Path) -> ValidationData:
    """Parse ``validation.bin`` written by ``potok_sim::validation::write_validation``.

    The header fields and their offsets come from ``sim/README.md``; reading
    them sequentially yields 80 bytes (the README's "76 bytes" undercounts its
    own 32-byte reserved field). Per episode: u64 index, 9 metadata f32,
    frames x features f32, then frames x 4 target f32, all row-major.
    """
    raw = Path(path).read_bytes()
    if len(raw) < SIM_HEADER.size:
        raise ValueError("validation file is shorter than its header")
    magic, version, seed, count, frames, features, targets, rate, dt, duration, _ = (
        SIM_HEADER.unpack_from(raw, 0)
    )
    if magic != SIM_MAGIC or version != SIM_VERSION:
        raise ValueError("not a version 1 POTOKSIM validation file")
    if targets != SIM_TARGETS or frames < 1 or features < 1:
        raise ValueError("unsupported validation dimensions")

    header = ValidationHeader(seed, count, frames, features, targets, rate, dt, duration)
    offset = SIM_HEADER.size
    episodes: list[Episode] = []
    for _ in range(count):
        if len(raw) < offset + SIM_EPISODE.size:
            raise ValueError("validation file truncated at an episode header")
        index, *metadata = SIM_EPISODE.unpack_from(raw, offset)
        offset += SIM_EPISODE.size
        frame_bytes = frames * features * 4
        target_bytes = frames * targets * 4
        if len(raw) < offset + frame_bytes + target_bytes:
            raise ValueError("validation file truncated at episode data")
        frame_data = np.frombuffer(
            raw, dtype="<f4", count=frames * features, offset=offset
        ).reshape(frames, features).copy()
        offset += frame_bytes
        target_data = np.frombuffer(
            raw, dtype="<f4", count=frames * targets, offset=offset
        ).reshape(frames, targets).copy()
        offset += target_bytes
        episodes.append(
            Episode(frame_data, target_data, index=index, metadata=np.asarray(metadata, np.float32))
        )
    if offset != len(raw):
        raise ValueError("trailing bytes in validation file")
    return ValidationData(header, episodes)


def read_episode(path: str | Path, n_features: int = 8) -> Episode:
    raw = Path(path).read_bytes()
    if len(raw) >= HEADER.size and raw[: len(MAGIC)] == MAGIC:
        _magic, version, frames, features, stride = HEADER.unpack_from(raw, 0)
        if version != VERSION:
            raise ValueError(f"unsupported episode version {version}")
        if stride < features:
            raise ValueError("episode stride is smaller than its feature count")
        need = HEADER.size + frames * stride * 4
        if len(raw) < need:
            raise ValueError("episode file is shorter than its header claims")
        data = np.frombuffer(raw, dtype="<f4", count=frames * stride, offset=HEADER.size)
        grid = data.reshape(frames, stride).copy()
        targets = grid[:, features:].copy() if stride > features else None
        return Episode(grid[:, :features], targets)

    values = np.frombuffer(raw, dtype="<f4")
    if values.size == 0 or values.size % n_features:
        raise ValueError("raw episode is not a multiple of n_features float32 values")
    return Episode(values.reshape(-1, n_features).copy(), None)


def write_episode(path: str | Path, episode: Episode) -> None:
    """Write an episode in the documented binary format (also used by tests)."""
    frames, features = episode.frames.shape
    targets = episode.targets
    stride = features + (0 if targets is None else targets.shape[1])
    payload = episode.frames if targets is None else np.concatenate([episode.frames, targets], axis=1)
    header = HEADER.pack(MAGIC, VERSION, frames, features, stride)
    Path(path).write_bytes(header + np.asarray(payload, dtype="<f4").tobytes())


def load_episodes(path: str | Path, n_features: int = 8) -> list[Episode]:
    """Load a validation file, one episode file, or every file in a directory.

    ``POTOKSIM`` files use the canonical reader; anything else falls back to the
    older ``POTOK1`` / bare-float32 fixture reader.
    """
    source = Path(path)
    files = (
        sorted(p for p in source.iterdir() if p.is_file())
        if source.is_dir()
        else [source]
    )
    episodes: list[Episode] = []
    for file in files:
        with file.open("rb") as handle:
            magic = handle.read(len(SIM_MAGIC))
        if magic == SIM_MAGIC:
            episodes.extend(read_validation(file).episodes)
        else:
            episodes.append(read_episode(file, n_features))
    return episodes


def make_synthetic_episode(
    seed: int, frames: int = 64, n_features: int = 8, with_targets: bool = True
) -> Episode:
    """Deterministic smooth sensor window with a gust and reference corrections."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, (frames, n_features)).astype(np.float32)
    smooth = np.concatenate([noise[:1], np.cumsum(noise[1:] * 0.3, axis=0)], axis=0)
    gust = np.sin(np.linspace(0.0, np.pi, frames, dtype=np.float32))[:, None]
    smooth[:, 1] += 2.0 * gust[:, 0]
    targets = np.tanh(smooth[:, 1:3] * 0.2).astype(np.float32) if with_targets else None
    return Episode(smooth, targets)


class WindowDataset(Dataset):
    """Sliding windows of consecutive frames, paired with a reference correction."""

    def __init__(
        self,
        episodes: list[Episode],
        window: int = 50,
        step: int = 1,
        limit: int | None = None,
        target_cols: int = 2,
    ) -> None:
        if window < 1:
            raise ValueError("window must be positive")
        self.target_cols = target_cols
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []
        for episode in episodes:
            n = len(episode.frames) - window
            for i in range(0, n + 1, step):
                target = (
                    episode.targets[i + window - 1, :target_cols]
                    if episode.targets is not None
                    else np.zeros(target_cols, dtype=np.float32)
                )
                self.samples.append((episode.frames[i : i + window], target))
        if limit is not None:
            self.samples = self.samples[:limit]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        window, target = self.samples[index]
        return (
            torch.from_numpy(np.asarray(window, dtype=np.float32)),
            torch.from_numpy(np.asarray(target, dtype=np.float32)),
        )


def make_synthetic_dataset(
    n_episodes: int = 4,
    frames: int = 64,
    seed: int = 0,
    window: int = 50,
    step: int = 1,
    limit: int | None = None,
) -> WindowDataset:
    episodes = [make_synthetic_episode(seed + i, frames) for i in range(n_episodes)]
    return WindowDataset(episodes, window=window, step=step, limit=limit)
