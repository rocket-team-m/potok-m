# SPDX-License-Identifier: GPL-3.0-or-later
"""On-policy DAgger gates: student-induced states, privileged labels, SHA provenance."""

from __future__ import annotations

import random
import struct
from pathlib import Path

import numpy as np
import pytest

from potok_ai import train_teacher_v2 as ttv
from potok_ai.dagger import (
    EVAL_SCENARIO_IDS,
    build_dagger_dataset,
    on_policy_rollout,
    sample_sha256,
)
from potok_ai.data import make_smart_episode

WINDOW = 12


def _train_episodes(n: int = 6, frames: int = 24) -> list:
    return [make_smart_episode(seed=100 + i, frames=frames, scenario_id=i) for i in range(n)]


def _action(scale: float):
    def act(observation: np.ndarray, _rng: random.Random) -> np.ndarray:
        return np.asarray([scale, -0.5 * scale], dtype=np.float32)

    return act


def _world(frame: np.ndarray, action: np.ndarray, _rng: random.Random) -> np.ndarray:
    nxt = np.asarray(frame, dtype=np.float32).copy()
    nxt[9] = np.clip(frame[9] + 2.0 * action[0], -30.0, 30.0)
    return nxt


def _privileged(episode, step: int, _rng: random.Random) -> np.ndarray:
    snapshot = np.zeros(26, dtype=np.float32)
    snapshot[:22] = np.arange(22, dtype=np.float32) + 1.0 + step
    snapshot[24:] = (3.0, -4.0)
    return snapshot


def _teacher(privileged: np.ndarray) -> np.ndarray:
    return np.asarray([privileged[24], -privileged[25]], dtype=np.float32)


def _rollout(**overrides):
    options = {
        "window": WINDOW,
        "student_action": _action(0.5),
        "world_step": _world,
        "privileged_snapshot": _privileged,
    }
    options.update(overrides)
    return on_policy_rollout(**options)


def _build_dataset(**overrides):
    options = {
        "seed": 7,
        "episodes": _train_episodes(),
        "rollout": _rollout(),
        "teacher": _teacher,
    }
    options.update(overrides)
    return build_dagger_dataset(**options)


def _cfg(tmp_path: Path, **overrides) -> ttv.TeacherV2Config:
    options = {
        "data": tmp_path / "data",
        "out": tmp_path / "out",
        "seed": 1,
        "epochs": 1,
        "window": WINDOW,
    }
    options.update(overrides)
    return ttv.TeacherV2Config(**options)


def _write_v3_sidecar(path: Path, episodes: list) -> None:
    frames = len(episodes[0].frames)
    header = struct.pack("<8sIIII", b"POTOKTRJ", 3, len(episodes), frames, 26)
    payload = bytearray(header.ljust(64, b"\0"))
    for episode in episodes:
        payload += struct.pack("<Q", int(episode.index))
        for tick in range(frames):
            privileged = np.zeros(22, dtype=np.float32)
            privileged[15] = 5.0  # wind_x: the teacher's truth channel is not zero
            labels = np.asarray([0.0, 0.0, 3.0, -4.0], dtype=np.float32)
            payload += np.concatenate([privileged, labels]).astype("<f4").tobytes()
    path.write_bytes(payload)


def test_student_actions_cause_the_visited_states() -> None:
    episode = _train_episodes(1)[0]
    states = _rollout()(episode, random.Random("on-policy"))
    assert len(states) == len(episode.frames) - WINDOW
    np.testing.assert_allclose(states[0].observation[:, :14], episode.frames[:WINDOW, :14])
    assert all(state.action is not None and state.action.shape == (2,) for state in states)
    # The student's command changes the next sensor frame, so the trace drifts
    # off the recorded one; a bigger command induces more drift than a zero one.
    assert not np.allclose(states[-1].observation[-1, :14], states[-1].recorded_frame)
    idle = _rollout(student_action=_action(0.0))(episode, random.Random("on-policy"))

    def mean_drift(trace) -> float:
        return float(
            np.mean([np.linalg.norm(s.observation[-1, :14] - s.recorded_frame) for s in trace[1:]])
        )

    assert mean_drift(states) > mean_drift(idle)


def test_teacher_sees_only_privileged_snapshot_at_visited_state() -> None:
    seen: list[np.ndarray] = []

    def teacher(privileged: np.ndarray) -> np.ndarray:
        seen.append(np.asarray(privileged, dtype=np.float32).copy())
        return np.asarray([7.0, 8.0], dtype=np.float32)

    data = _build_dataset(teacher=teacher)
    assert len(seen) == len(data.samples) > 0
    for sample, privileged in zip(data.samples, seen):
        assert privileged.shape == (26,)
        assert sample.observation.shape == (WINDOW, 20)
        # The 22 privileged truth slots are never part of the student's window.
        assert not np.array_equal(privileged[:14], sample.observation[0, :14])
        np.testing.assert_array_equal(sample.correction, np.asarray([7.0, 8.0], np.float32))


def test_same_seed_is_bit_identical_and_sha_commits_to_privileged() -> None:
    first = _build_dataset()
    second = _build_dataset()
    other_seed = _build_dataset(seed=8)
    assert first.fingerprint() == second.fingerprint()
    assert first.fingerprint() != other_seed.fingerprint()

    sample = first.samples[0]
    sha = sample.provenance.rollout_sha256
    assert len(sha) == 64 and set(sha) <= set("0123456789abcdef")
    assert sample.provenance.sample_id == (
        f"{sample.provenance.seed}:dagger:{sample.provenance.round}:rollout:"
        f"{sample.provenance.rollout}:episode:{sample.provenance.episode}:"
        f"step:{sample.provenance.step}"
    )

    def shifted(episode, step, rng):
        snapshot = _privileged(episode, step, rng).copy()
        snapshot[15] += 1.0
        return snapshot

    changed = _build_dataset(rollout=_rollout(privileged_snapshot=shifted))
    assert changed.samples[0].provenance.rollout_sha256 != sha
    assert changed.fingerprint() != first.fingerprint()


def test_sample_sha256_helper_is_pure_and_ordered() -> None:
    observation = np.zeros(3, dtype=np.float32)
    privileged = np.ones(2, dtype=np.float32)
    digest = sample_sha256("id", observation, privileged, None, None)
    assert digest == sample_sha256("id", observation, privileged, None, None)
    assert digest != sample_sha256("other", observation, privileged, None, None)
    assert digest != sample_sha256("id", privileged, privileged, None, None)


def test_eval_cells_are_refused_before_student_or_teacher_runs() -> None:
    calls = {"student": 0, "teacher": 0}

    def student(observation, rng):
        calls["student"] += 1
        return _action(0.5)(observation, rng)

    def teacher(privileged):
        calls["teacher"] += 1
        return _teacher(privileged)

    for sid in sorted(EVAL_SCENARIO_IDS):
        episode = make_smart_episode(99, frames=24, scenario_id=sid)
        with pytest.raises(ValueError, match="eval"):
            _build_dataset(episodes=_train_episodes() + [episode], teacher=teacher)
        with pytest.raises(ValueError, match="eval"):
            _rollout(student_action=student)(episode, random.Random("x"))
    flagged = make_smart_episode(98, frames=24, scenario_id=0, held_out=True)
    with pytest.raises(ValueError, match="eval"):
        _build_dataset(episodes=_train_episodes() + [flagged])
    assert calls == {"student": 0, "teacher": 0}


def test_trainer_dataset_is_on_policy_and_records_provenance() -> None:
    cfg = _cfg(Path("/tmp"))
    episodes = _train_episodes()
    fit, inner, held = ttv.partition_episodes(episodes, seed=cfg.seed)
    dataset = ttv._build_dagger_dataset(cfg, [*fit, *inner], held)
    info = dataset.dagger_info
    assert info["rollout_source"].startswith("on_policy_student")
    # Step 0 of each episode is the recorded seed prefix; every later visited
    # state is produced by the student's own command.
    assert info["student_induced_rows"] == len(dataset.samples) - len(episodes)
    assert info["student_induced_rows"] > 0
    assert info["privileged_fields_present"] is False
    assert len(info["rollout_sha256"]) == 64
    assert all(sample.action is not None and sample.drift is not None for sample in dataset.samples)
    assert all(np.isfinite(sample.correction).all() for sample in dataset.samples)
    np.testing.assert_allclose(dataset.samples[0].correction, dataset.samples[0].authoritative)


def test_trainer_refuses_eval_cells_in_dagger_rollout() -> None:
    cfg = _cfg(Path("/tmp"))
    episodes = _train_episodes() + [make_smart_episode(9, frames=24, scenario_id=17)]
    with pytest.raises(ValueError, match="eval"):
        ttv._build_dagger_dataset(cfg, episodes, frozenset())


def test_trainer_metrics_pin_rollout_provenance(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ttv, "load_episodes", lambda path: _train_episodes())
    cfg = _cfg(tmp_path, stage="dagger")
    first = ttv.run(cfg).stages["dagger"]["details"]
    second = ttv.run(cfg).stages["dagger"]["details"]
    assert first == second
    assert first["rollout_source"].startswith("on_policy_student")
    assert first["student_induced_rows"] > 0
    assert first["privileged_fields_present"] is False
    assert first["rollout_sha256"] == second["rollout_sha256"]


def test_trainer_uses_sidecar_privilege_at_visited_states(tmp_path: Path) -> None:
    episodes = _train_episodes()
    sidecar = tmp_path / "truth.bin"
    _write_v3_sidecar(sidecar, episodes)
    cfg = _cfg(tmp_path, truth_sidecar=sidecar)
    fit, inner, held = ttv.partition_episodes(episodes, seed=cfg.seed)
    dataset = ttv._build_dagger_dataset(cfg, [*fit, *inner], held)
    assert dataset.dagger_info["privileged_source"] == "v3_truth_sidecar"
    assert dataset.dagger_info["privileged_fields_present"] is True
    assert dataset.dagger_info["student_induced_rows"] > 0
    # The label is read from the truth sidecar at every student-visited tick.
    for sample in dataset.samples:
        np.testing.assert_array_equal(sample.correction, np.asarray([3.0, 4.0], np.float32))
