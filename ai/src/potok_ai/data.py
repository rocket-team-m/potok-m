"""Episode loaders for Potok-M training data.

Canonical input: the simulator's ``validation.bin`` (``POTOKSIM`` magic), written
by ``potok_sim::validation``. Version 1 is the legacy 8-feature / 9-metadata
layout; version 2 adds conditioning, a wind/gust scenario id and a split flag;
version 3 keeps the v2 layout but writes six targets per frame, appending the
authoritative fin command demand ``g_accel_y, g_accel_x`` (m/s^2) to the four
legacy angle-domain columns. ``Episode.command_target()`` returns the fin
command ``[g_accel_y, -g_accel_x]`` for v3 and raises for v1/v2; ``Episode.g_theta()``
keeps the legacy angle target as a diagnostic. Fitting that angle target as a
command requires the explicit ``allow_diagnostic_target`` escape, never a silent
substitution. The version and target count are checked together, so a v3 file is
never silently read as v2. The older
``POTOK1\\0\\0`` layout and a bare float32 array remain as fixture fallbacks.
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
SIM_VERSION_SMART = 2
SIM_VERSION_V3 = 3
SIM_TARGETS = 4  # v1/v2: g_roll, g_pitch, confidence, ai_healthy
SIM_TARGETS_V3 = 6  # v3: the four legacy columns + g_accel_y, g_accel_x (m/s^2)
# Target column indices. Columns 0-1 are the diagnostic angle target g_theta;
# columns 4-5 are the authoritative fin command demand written by v3.
TARGET_G_ROLL = 0
TARGET_G_PITCH = 1
TARGET_CONFIDENCE = 2
TARGET_AI_HEALTHY = 3
TARGET_G_ACCEL_Y = 4
TARGET_G_ACCEL_X = 5
# magic, version, seed, count, frames, features, targets, rate, dt, duration, reserved
SIM_HEADER = struct.Struct("<8sIQ4I3f32s")
SIM_EPISODE = struct.Struct("<Q9f")  # index + 9 legacy metadata floats

# SPEC_SMART.md v2 / v3, written by potok_sim::validation: the same 80-byte
# header (meta_count in reserved bytes 48..52) and per episode 17 metadata
# floats: the legacy 9, then 6 conditioning numbers, scenario id and split.
# meta_count=39 appends a 22-float terminal truth record (v2/v3 truth files).
SIM_META_V1 = 9
SIM_META_V2 = 17
SIM_META_V2_TRUTH = 39
SIM_V2_EPISODE = struct.Struct("<Q17f")  # index + 17 metadata floats
SIM_V2_TRUTH_EPISODE = struct.Struct("<Q39f")  # + 22 truth floats
SMART_FEATURES = 14  # 8 raw telemetry + 6 physics features
SMART_COND_FEATURES = 6  # mass, area, cd0, static margin, thrust, gyro noise
SMART_SPLIT_TRAIN = 0
SMART_SPLIT_HELD_OUT = 1


@dataclass
class Episode:
    """A single simulated flight: sensor frames and optional reference targets."""

    frames: np.ndarray  # (T, F) float32
    targets: np.ndarray | None = None  # (T, K) float32, K = 4 for validation sets
    index: int | None = None
    metadata: np.ndarray | None = None  # (9,) f32 legacy, or (6,) conditioning
    scenario: str | None = None  # wind/gust group, e.g. "scenario_16"
    scenario_id: int | None = None  # numeric cell id from a v2 validation file
    conditioning: np.ndarray | None = None  # (6,) f32 vehicle numbers
    held_out: bool = False  # scenario reserved for the generalisation check
    source: str | None = None  # file stem, used as a scenario fallback for v1 files
    g_accel: np.ndarray | None = None  # (T,2) [g_accel_y, g_accel_x] m/s^2, v3 only

    def g_theta(self) -> np.ndarray:
        """Legacy angle-domain diagnostic target ``(T,2)`` [roll, pitch] (rad)."""
        if self.targets is None:
            raise ValueError("episode has no reference targets")
        return np.asarray(self.targets[:, TARGET_G_ROLL : TARGET_G_PITCH + 1], np.float32)

    def command_target(self, *, allow_diagnostic_target: bool = False) -> np.ndarray:
        """Authoritative fin command ``(T,2)`` in [roll, pitch] order.

        A v3 file carries ``g_accel``; the fin actuator consumes
        ``[g_accel_y, -g_accel_x]`` m/s^2 (sim ``fin_accel_command``). A v1/v2
        file has no acceleration column, so it has no command at all: fitting
        the diagnostic ``g_theta`` (rad) as a command is a loud error unless
        ``allow_diagnostic_target`` explicitly requests angle-domain diagnostic
        training.
        """
        if self.g_accel is None:
            if allow_diagnostic_target:
                return self.g_theta()
            raise ValueError(
                "v1/v2 episode has no authoritative g_accel command; "
                "command-target training requires v3 data (g_theta is "
                "diagnostic only). Pass allow_diagnostic_target=True only for "
                "explicit angle-domain diagnostic training"
            )
        return np.stack(
            (self.g_accel[:, 0], -self.g_accel[:, 1]), axis=1
        ).astype(np.float32)


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
    version: int = SIM_VERSION


@dataclass
class ValidationData:
    header: ValidationHeader
    episodes: list[Episode]


def read_validation(path: str | Path) -> ValidationData:
    """Parse ``validation.bin`` written by ``potok_sim::validation::write_validation``.

    The header fields and their offsets come from ``sim/README.md``; reading
    them sequentially yields 80 bytes (the README's "76 bytes" undercounts its
    own 32-byte reserved field). Version 1 carries 9 metadata floats and 8
    features; versions 2 and 3 (SPEC_SMART.md) carry a scenario name, 6
    conditioning numbers, a held-out flag and 14 physics features. Version 3
    writes six targets per frame (the four legacy columns plus the fin command
    demand ``g_accel_y, g_accel_x``). The version and target count are checked
    together: a v3 file is never silently reinterpreted as v2. Frames and
    targets are row-major in all versions.
    """
    raw = Path(path).read_bytes()
    if len(raw) < SIM_HEADER.size:
        raise ValueError("validation file is shorter than its header")
    magic, version, seed, count, frames, features, targets, rate, dt, duration, reserved = (
        SIM_HEADER.unpack_from(raw, 0)
    )
    if magic != SIM_MAGIC:
        raise ValueError("not a POTOKSIM validation file")
    if version not in (SIM_VERSION, SIM_VERSION_SMART, SIM_VERSION_V3):
        raise ValueError(
            f"unsupported validation version {version}, expected version 1, 2 or 3"
        )
    expected_targets = SIM_TARGETS_V3 if version == SIM_VERSION_V3 else SIM_TARGETS
    if targets != expected_targets:
        raise ValueError(
            f"validation version {version} needs {expected_targets} targets, "
            f"header says {targets}; refusing to reinterpret the file as another version"
        )
    if frames < 1 or features < 1:
        raise ValueError("unsupported validation dimensions")
    smart = version in (SIM_VERSION_SMART, SIM_VERSION_V3)
    record = SIM_EPISODE
    if smart:
        if features < SMART_FEATURES:
            raise ValueError(f"smart validation needs at least {SMART_FEATURES} features")
        meta_count = struct.unpack_from("<I", reserved, 0)[0]
        if meta_count == SIM_META_V2:
            record = SIM_V2_EPISODE
        elif meta_count == SIM_META_V2_TRUTH:
            record = SIM_V2_TRUTH_EPISODE
        else:
            raise ValueError(f"unsupported metadata count {meta_count}")

    header = ValidationHeader(
        seed, count, frames, features, targets, rate, dt, duration, version=version
    )
    offset = SIM_HEADER.size
    episodes: list[Episode] = []
    for _ in range(count):
        if len(raw) < offset + record.size:
            raise ValueError("validation file truncated at an episode header")
        fields = record.unpack_from(raw, offset)
        offset += record.size
        index, *meta = fields
        if smart:
            metadata = np.asarray(meta[:SIM_META_V1], np.float32)
            conditioning = np.asarray(meta[SIM_META_V1 : SIM_META_V1 + SMART_COND_FEATURES], np.float32)
            # Scenario/split are metadata slots 15/16; truth-extended v2/v3
            # files append 22 more floats after them.
            scenario_at = SIM_META_V1 + SMART_COND_FEATURES
            scenario_value, split = meta[scenario_at], meta[scenario_at + 1]
            if not float(scenario_value).is_integer() or not 0 <= scenario_value <= 65535:
                raise ValueError("bad scenario id")
            if split not in (SMART_SPLIT_TRAIN, SMART_SPLIT_HELD_OUT):
                raise ValueError("bad split flag")
            scenario_id = int(scenario_value)
            scenario = f"scenario_{scenario_id:02d}"
            held_out = split == SMART_SPLIT_HELD_OUT
        else:
            metadata = np.asarray(meta, np.float32)
            # v1 files have no conditioning; a zero vector says "nominal Potok-M".
            conditioning = np.zeros(SMART_COND_FEATURES, np.float32)
            scenario_id, scenario, held_out = None, None, False
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
        g_accel = (
            np.asarray(
                target_data[:, TARGET_G_ACCEL_Y : TARGET_G_ACCEL_X + 1], np.float32
            ).copy()
            if version == SIM_VERSION_V3
            else None
        )
        episodes.append(
            Episode(
                frame_data,
                target_data,
                index=index,
                metadata=metadata,
                scenario=scenario,
                scenario_id=scenario_id,
                conditioning=conditioning,
                held_out=held_out,
                g_accel=g_accel,
            )
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
        loaded = read_validation(file).episodes if magic == SIM_MAGIC else [read_episode(file, n_features)]
        for episode in loaded:
            episode.source = file.stem
        episodes.extend(loaded)
    return episodes


def group_by_scenario(episodes: list[Episode]) -> dict[str, list[Episode]]:
    """Group episodes by wind/gust scenario; v1 files fall back to the file name."""
    groups: dict[str, list[Episode]] = {}
    for episode in episodes:
        name = episode.scenario or episode.source or "default"
        groups.setdefault(name, []).append(episode)
    return groups


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


def make_smart_episode(
    seed: int,
    frames: int = 80,
    scenario_id: int = 0,
    conditioning: np.ndarray | None = None,
    held_out: bool = False,
) -> Episode:
    """Deterministic 14-feature Potok-M episode for tests and CLI smoke runs."""
    rng = np.random.default_rng(seed)
    raw = rng.normal(0.0, 1.0, (frames, 8)).astype(np.float32)
    raw[:, 1] += np.sin(np.linspace(0.0, np.pi, frames, dtype=np.float32)) * 2.0
    physics = np.stack(
        [
            raw[:, 2] * 3.0,  # alpha_deg
            raw[:, 1] * 1.5,  # beta_deg
            np.abs(raw[:, 0]) + 0.5,  # q_dyn_kpa
            raw[:, 3] * 0.4,  # a_lat_resid
            np.abs(raw[:, 0]) * 0.2,  # drag_est
            np.full(frames, 0.95, np.float32),  # static_margin_cal
        ],
        axis=1,
    ).astype(np.float32)
    frames_array = np.concatenate([raw, physics], axis=1)
    diagnostic = np.tanh(frames_array[:, 1:3] * 0.3).astype(np.float32)
    # v3 fixture: choose g_accel so the authoritative command equals the old
    # diagnostic target, keeping training numerics unchanged while exercising
    # the command-target path.
    g_accel = np.stack((diagnostic[:, 0], -diagnostic[:, 1]), axis=1).astype(np.float32)
    targets = np.concatenate(
        [diagnostic, np.ones((frames, 2), np.float32), g_accel], axis=1
    )
    if conditioning is None:
        conditioning = np.array([13.0, 0.0082, 0.35, 0.95, 3000.0, 0.01], np.float32)
    legacy_meta = np.array([13.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0], np.float32)
    return Episode(
        frames_array,
        targets,
        index=seed,
        metadata=legacy_meta,
        scenario=f"scenario_{scenario_id:02d}",
        scenario_id=scenario_id,
        conditioning=np.asarray(conditioning, np.float32).copy(),
        held_out=held_out,
        g_accel=g_accel,
    )


class SmartWindowDataset(Dataset):
    """Sliding windows of 14 features with conditioning and a correction target."""

    def __init__(
        self,
        episodes: list[Episode],
        window: int = 50,
        step: int = 1,
        limit: int | None = None,
        allow_diagnostic_target: bool = False,
    ) -> None:
        if window < 1:
            raise ValueError("window must be positive")
        self.samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for episode in episodes:
            if episode.frames.shape[1] < SMART_FEATURES:
                raise ValueError(
                    f"smart episodes need {SMART_FEATURES} features, got {episode.frames.shape[1]}"
                )
            if episode.conditioning is None:
                raise ValueError("smart episodes need a conditioning vector")
            # v3 targets are the fin command [g_accel_y, -g_accel_x]. A legacy
            # v1/v2 episode has no command: it is refused loudly unless the
            # caller opts into angle-domain diagnostic training.
            command = (
                episode.command_target(allow_diagnostic_target=allow_diagnostic_target)
                if episode.targets is not None
                else None
            )
            for start in range(0, len(episode.frames) - window + 1, step):
                target = (
                    command[start + window - 1]
                    if command is not None
                    else np.zeros(2, np.float32)
                )
                self.samples.append(
                    (
                        episode.frames[start : start + window, :SMART_FEATURES],
                        episode.conditioning,
                        target,
                    )
                )
        if limit is not None:
            self.samples = self.samples[:limit]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        window, conditioning, target = self.samples[index]
        return (
            torch.from_numpy(np.asarray(window, dtype=np.float32)),
            torch.from_numpy(np.asarray(conditioning, dtype=np.float32)),
            torch.from_numpy(np.asarray(target, dtype=np.float32)),
        )
