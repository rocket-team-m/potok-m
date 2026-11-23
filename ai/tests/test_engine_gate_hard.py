# SPDX-License-Identifier: GPL-3.0-or-later
"""Hard acceptance gate for the engine: 100 epochs, tiny v3 data, CPU only.

SPEC_ENGINE_IDEAL.md §4-5 fixes two numeric gates: held-out/train RMSE ratio
<= 1.25 and held-out ECE <= 0.05. This test drives the real CLI on a tiny but
version-correct v3 file, checks both gates, and proves the run is reproducible
and that the held-out scenario was never fed to training.
"""

import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from potok_ai import engine
from potok_ai.calibration import DEFAULT_TARGET_ECE
from potok_ai.data import (
    SIM_MAGIC,
    SIM_META_V2,
    SIM_TARGETS_V3,
    SIM_V2_EPISODE,
    SMART_FEATURES,
    make_smart_episode,
)
from potok_ai.train_smart import GENERALISATION_LIMIT

EPOCHS = 100
SEED = 1
WINDOW = 12
FRAMES = 24
HELD_OUT_ID = 16
TRAIN_IDS = (0, 1)


def _command(frames: np.ndarray) -> np.ndarray:
    """Authoritative fin demand: physics baseline plus a learnable residual."""
    beta, alpha, a_lat = frames[:, 9], frames[:, 8], frames[:, 11]
    roll = np.clip(0.4 * beta / 10.0 + 0.1 * np.tanh(a_lat), -1.0, 1.0)
    pitch = np.clip(0.4 * alpha / 10.0, -1.0, 1.0)
    return np.stack(
        (roll + 0.2 * np.tanh(frames[:, 1]), pitch + 0.15 * np.tanh(frames[:, 2])), axis=1
    ).astype(np.float32)


def _episode(seed: int, scenario_id: int, held_out: bool):
    episode = make_smart_episode(
        seed=seed, frames=FRAMES, scenario_id=scenario_id, held_out=held_out
    )
    command = _command(episode.frames)
    diagnostic = np.concatenate(
        [np.tanh(episode.frames[:, 1:3] * 0.3), np.ones((FRAMES, 2), np.float32)], axis=1
    )
    # v3: columns 4/5 are g_accel_y, g_accel_x; command_target() flips the x sign.
    episode.g_accel = np.stack((command[:, 0], -command[:, 1]), axis=1).astype(np.float32)
    episode.targets = np.concatenate([diagnostic, episode.g_accel], axis=1).astype(np.float32)
    return episode


def _pack(episodes: list) -> bytes:
    header = struct.pack(
        "<8sIQ4I3f32s",
        SIM_MAGIC,
        3,
        42,
        len(episodes),
        FRAMES,
        SMART_FEATURES,
        SIM_TARGETS_V3,
        50.0,
        0.02,
        0.06,
        struct.pack("<I", SIM_META_V2) + b"\0" * 28,
    )
    raw = bytearray(header)
    for episode in episodes:
        metadata = (
            episode.index,
            *episode.metadata,
            *episode.conditioning,
            episode.scenario_id,
            int(episode.held_out),
        )
        raw += SIM_V2_EPISODE.pack(*metadata)
        raw += np.asarray(episode.frames[:, :SMART_FEATURES], np.float32).tobytes()
        raw += np.asarray(episode.targets, np.float32).tobytes()
    return bytes(raw)


def test_engine_100_epoch_v3_hard_gate(tmp_path: Path, monkeypatch, capsys) -> None:
    episodes = [
        _episode(seed, scenario_id, held_out)
        for seed, (scenario_id, held_out) in enumerate(
            [
                (TRAIN_IDS[0], False),
                (TRAIN_IDS[0], False),
                (TRAIN_IDS[1], False),
                (TRAIN_IDS[1], False),
                (HELD_OUT_ID, True),
                (HELD_OUT_ID, True),
            ]
        )
    ]
    validation = tmp_path / "validation.bin"
    raw = _pack(episodes)
    validation.write_bytes(raw)
    before = hashlib.sha256(raw).hexdigest()

    trained: list[int] = []
    real_batches = engine.LRSAdapter.batches

    def batches(self, chosen, cfg):
        assert all(
            not episode.held_out and episode.scenario_id != HELD_OUT_ID for episode in chosen
        )
        trained.extend(episode.index for episode in chosen)
        yield from real_batches(self, chosen, cfg)

    monkeypatch.setattr(engine.LRSAdapter, "batches", batches)

    reports = []
    for run in range(2):
        out = tmp_path / f"run{run}"
        argv = [
            "--model",
            "lrs",
            "--stage",
            "train",
            "--data",
            str(validation),
            "--out",
            str(out),
            "--epochs",
            str(EPOCHS),
            "--seed",
            str(SEED),
            "--device",
            "cpu",
            "--window",
            str(WINDOW),
            "--batch",
            "2",
            "--limit",
            "4",
            "--eval-limit",
            "2",
            "--calibrate",
        ]
        engine.main(argv)
        capsys.readouterr()
        report = json.loads((out / "metrics.json").read_text())
        assert (out / "lrs.bin").stat().st_size == report["bytes"]
        reports.append(report)

    # Same seed, same data, same report: determinism at the full epoch budget.
    assert reports[0] == reports[1]
    report = reports[0]
    assert tuple(report) == engine.METRIC_FIELDS
    assert report["seed"] == SEED and len(report["config_hash"]) == 16

    # Held-out data stayed on disk, byte for byte, and never entered a batch.
    assert hashlib.sha256(validation.read_bytes()).hexdigest() == before
    assert set(trained) == {0, 1, 2, 3}
    assert len(trained) >= 2 * EPOCHS * len(TRAIN_IDS)

    # Hard gates from SPEC_ENGINE_IDEAL.md §4-5.
    assert np.isfinite(report["rmse_train"]) and report["rmse_train"] > 0
    assert np.isfinite(report["rmse_held_out"]) and report["rmse_held_out"] > 0
    assert report["ratio"] <= GENERALISATION_LIMIT
    assert 0.0 <= report["ece"] <= DEFAULT_TARGET_ECE

    calibration = json.loads((tmp_path / "run0" / "calibration.json").read_text())
    assert calibration["ece"] == report["ece"]
    assert calibration["config_hash"] == report["config_hash"]
    assert calibration["n"] > 0 and 0.0 <= calibration["threshold"] <= 1.0
