# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the DAgger corrective dataset builder (POST04)."""

from __future__ import annotations

import random

import numpy as np
import pytest

from potok_ai.dagger import (
    EVAL_SCENARIO_IDS,
    EXPERT_TEACHER,
    LABEL_SOURCE_EXPERT,
    SPLIT_FIT,
    SPLIT_HELD_OUT,
    RolloutState,
    build_dagger_dataset,
    is_eval_episode,
    split_scenarios,
)
from potok_ai.data import Episode, make_smart_episode


def _train_episodes(n: int = 6, frames: int = 6) -> list[Episode]:
    return [make_smart_episode(seed=100 + i, frames=frames, scenario_id=i) for i in range(n)]


def _rollout(episode: Episode, rng: random.Random) -> list[RolloutState]:
    """Student-induced states; all noise comes from the named child stream."""
    states = []
    for step in range(3):
        states.append(
            RolloutState(
                observation=np.array([rng.random(), step], dtype=np.float32),
                privileged=np.array([rng.random() * 4.0, -step], dtype=np.float32),
                authoritative=np.array([rng.random()], dtype=np.float32) if step == 0 else None,
            )
        )
    return states


def _teacher(privileged: np.ndarray) -> np.ndarray:
    return np.array([privileged[0] * 10.0, -privileged[1]], dtype=np.float32)


def _build(**overrides):
    options = {"seed": 7, "episodes": _train_episodes(), "rollout": _rollout, "teacher": _teacher}
    options.update(overrides)
    return build_dagger_dataset(**options)


def test_labels_come_from_teacher_not_student():
    constant = np.array([3.0, -5.0], dtype=np.float32)
    data = _build(teacher=lambda _privileged: constant.copy())
    assert data.samples
    for sample in data.samples:
        np.testing.assert_array_equal(sample.correction, constant)


def test_teacher_receives_privileged_snapshot_not_observation():
    seen: list[np.ndarray] = []

    def rollout(episode, rng):
        return [
            RolloutState(
                observation=np.array([0.0], dtype=np.float32),
                privileged=np.array([11.0, 22.0], dtype=np.float32),
            )
        ]

    def teacher(privileged):
        seen.append(np.asarray(privileged, dtype=np.float32).copy())
        return np.array([1.0, 2.0], dtype=np.float32)

    data = _build(rollout=rollout, teacher=teacher)
    assert len(seen) == len(data.samples) == len(_train_episodes())
    for privileged in seen:
        np.testing.assert_array_equal(privileged, np.array([11.0, 22.0], dtype=np.float32))


def test_same_seed_is_bit_identical():
    first = _build(rounds=2, rollouts_per_round=2)
    second = _build(rounds=2, rollouts_per_round=2)
    assert first.fingerprint() == second.fingerprint()
    assert [s.provenance.sample_id for s in first.samples] == [
        s.provenance.sample_id for s in second.samples
    ]


def test_different_seed_changes_fingerprint():
    assert _build(seed=7).fingerprint() != _build(seed=8).fingerprint()


def test_rounds_aggregate_with_unique_provenance():
    data = _build(rounds=3, rollouts_per_round=2)
    assert len(data.samples) == 3 * 2 * len(_train_episodes()) * 3
    ids = [s.provenance.sample_id for s in data.samples]
    assert len(ids) == len(set(ids))
    assert {s.provenance.round for s in data.samples} == {0, 1, 2}
    assert {s.provenance.rollout for s in data.samples} == {0, 1}
    for sample in data.samples:
        provenance = sample.provenance
        assert provenance.seed == 7
        assert provenance.expert == EXPERT_TEACHER
        assert provenance.label_source == LABEL_SOURCE_EXPERT
        assert provenance.split in (SPLIT_FIT, SPLIT_HELD_OUT)
        assert provenance.sample_id == (
            f"{provenance.seed}:dagger:{provenance.round}:rollout:{provenance.rollout}:"
            f"episode:{provenance.episode}:step:{provenance.step}"
        )


def test_held_out_split_is_disjoint_and_deterministic():
    fit, held = split_scenarios(7, range(6), n_held_out=2)
    assert fit & held == frozenset()
    assert fit | held == frozenset(range(6))
    assert len(held) == 2
    assert (fit, held) == split_scenarios(7, range(6), n_held_out=2)

    data = _build(n_held_out=2)
    assert data.fit_scenarios & data.held_out_scenarios == frozenset()
    fit_sids = {s.provenance.scenario_id for s in data.fit_samples()}
    held_sids = {s.provenance.scenario_id for s in data.held_out_samples()}
    assert fit_sids & held_sids == set()
    assert fit_sids == set(data.fit_scenarios)
    assert held_sids == set(data.held_out_scenarios)


def test_eval_episodes_are_refused_before_any_call():
    calls = {"rollout": 0, "teacher": 0}

    def rollout(episode, rng):
        calls["rollout"] += 1
        return _rollout(episode, rng)

    def teacher(privileged):
        calls["teacher"] += 1
        return _teacher(privileged)

    eval_episode = make_smart_episode(seed=99, frames=6, scenario_id=17, held_out=True)
    with pytest.raises(ValueError, match="eval"):
        _build(episodes=_train_episodes() + [eval_episode], rollout=rollout, teacher=teacher)
    assert calls == {"rollout": 0, "teacher": 0}


def test_eval_scenario_id_refused_even_without_flag():
    eval_episode = make_smart_episode(seed=99, frames=6, scenario_id=18, held_out=False)
    with pytest.raises(ValueError, match="eval"):
        _build(episodes=[eval_episode])


def test_is_eval_episode_helper():
    assert is_eval_episode(make_smart_episode(1, scenario_id=16))
    assert is_eval_episode(make_smart_episode(1, scenario_id=5, held_out=True))
    assert not is_eval_episode(make_smart_episode(1, scenario_id=5))


def test_no_eval_sample_in_aggregate():
    data = _build()
    assert data.samples
    assert all(s.provenance.scenario_id not in EVAL_SCENARIO_IDS for s in data.samples)


def test_authoritative_target_passthrough():
    data = _build()
    authoritative = [s for s in data.samples if s.authoritative is not None]
    assert authoritative
    assert all(s.provenance.step == 0 for s in authoritative)
    assert all(s.authoritative is None for s in data.samples if s.provenance.step != 0)


def test_split_refuses_eval_cells():
    with pytest.raises(ValueError, match="eval"):
        split_scenarios(7, [0, 1, 16, 17, 18], n_held_out=1)


def test_invalid_arguments_are_loud():
    with pytest.raises(ValueError, match="positive"):
        _build(rounds=0)
    with pytest.raises(ValueError, match="positive"):
        _build(rollouts_per_round=0)
    with pytest.raises(ValueError, match="no episodes"):
        _build(episodes=[])
    with pytest.raises(ValueError, match="n_held_out"):
        _build(n_held_out=0)
    with pytest.raises(ValueError, match="scenario id"):
        _build(episodes=[Episode(np.zeros((6, 14), dtype=np.float32))])
