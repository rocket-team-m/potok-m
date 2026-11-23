# SPDX-License-Identifier: GPL-3.0-or-later
"""Guard decisions, latch semantics, purity, and loud config/shape failures."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from potok_ai.watchguard import GuardConfig, GuardReason, GuardState, guard, reset

CFG = GuardConfig(bounds=((-1.0, 1.0), (-2.0, 2.0)), max_rate=(0.5, 0.25))


def test_first_sample_is_accepted_and_guard_is_pure():
    command = np.array([0.4, -0.3])
    state = GuardState()
    decision = guard(command, 0.9, state, CFG)

    assert decision == guard(command, 0.9, GuardState(), CFG)  # deterministic
    assert decision.accepted and decision.reason is GuardReason.OK
    assert decision.command == (0.4, -0.3) and isinstance(decision.command, tuple)
    assert decision.state == GuardState(misses=0, latched=False, last_command=(0.4, -0.3))
    assert decision.state is not state  # a new state, never a mutation
    assert command.tolist() == [0.4, -0.3]  # input untouched


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_command_is_rejected_with_zero_output(bad):
    state = GuardState(misses=1, last_command=(0.1, 0.0))
    decision = guard([bad, 0.2], 0.9, state, CFG)

    assert not decision.accepted and decision.reason is GuardReason.NON_FINITE
    assert decision.command == (0.0, 0.0)
    assert decision.state == GuardState(misses=2, last_command=(0.1, 0.0))


def test_non_finite_confidence_is_rejected():
    decision = guard([0.0, 0.0], float("inf"), GuardState(), CFG)
    assert decision.reason is GuardReason.NON_FINITE
    assert decision.command == (0.0, 0.0)


def test_bounds_are_inclusive_and_violations_reject():
    for command in ([1.1, 0.0], [-1.1, 0.0], [0.0, 2.1], [0.0, -2.1]):
        decision = guard(command, 0.9, GuardState(), CFG)
        assert not decision.accepted and decision.reason is GuardReason.OUT_OF_BOUNDS
        assert decision.command == (0.0, 0.0)
        assert decision.state.misses == 1
    assert guard([1.0, -2.0], 0.9, GuardState(), CFG).accepted


def test_confidence_threshold_is_inclusive():
    low = guard([0.2, 0.1], CFG.min_confidence - 0.01, GuardState(), CFG)
    assert low.reason is GuardReason.LOW_CONFIDENCE
    assert low.command == (0.0, 0.0)
    assert guard([0.2, 0.1], CFG.min_confidence, GuardState(), CFG).accepted


def test_rate_limit_is_inclusive_and_uses_last_accepted_command():
    first = guard([0.4, 0.2], 0.9, GuardState(), CFG)
    at_limit = guard([0.9, -0.05], 0.9, first.state, CFG)  # deltas 0.5 and -0.25
    assert at_limit.accepted and at_limit.reason is GuardReason.OK
    too_fast = guard([0.0, 0.0], 0.9, at_limit.state, CFG)  # delta -0.9 on channel 0
    assert too_fast.reason is GuardReason.RATE_LIMIT
    assert too_fast.command == (0.0, 0.0)
    assert too_fast.state.last_command == at_limit.command  # spike did not move reference
    assert too_fast.state.misses == 1


def test_three_consecutive_misses_latch_and_stick():
    state = guard([float("nan"), 0.0], 0.9, GuardState(), CFG).state
    state = guard([5.0, 0.0], 0.9, state, CFG).state
    assert state.misses == 2 and not state.latched

    third = guard([0.0, 0.0], 0.1, state, CFG)
    assert third.reason is GuardReason.LOW_CONFIDENCE
    assert third.state.misses == 3 and third.state.latched

    after = guard([0.2, 0.1], 0.99, third.state, CFG)  # valid, but too late
    assert not after.accepted and after.reason is GuardReason.LATCHED
    assert after.command == (0.0, 0.0)
    assert after.state == third.state
    assert guard([0.2, 0.1], 0.99, after.state, CFG).reason is GuardReason.LATCHED


def test_accept_resets_miss_counter_before_the_latch():
    state = guard([9.0, 0.0], 0.9, GuardState(), CFG).state
    state = guard([0.0, 0.0], 0.1, state, CFG).state
    assert state.misses == 2 and not state.latched

    good = guard([0.0, 0.0], 0.9, state, CFG)
    assert good.accepted and good.state.misses == 0 and not good.state.latched
    assert guard([9.0, 0.0], 0.9, good.state, CFG).state.misses == 1


def test_max_misses_one_latches_on_first_rejection():
    cfg = GuardConfig(bounds=((-1.0, 1.0),), max_rate=(0.5,), max_misses=1)
    decision = guard([2.0], 0.9, GuardState(), cfg)
    assert decision.state.latched and decision.reason is GuardReason.OUT_OF_BOUNDS


def test_reset_is_the_only_way_to_clear_the_latch():
    latched = GuardState(misses=3, latched=True, last_command=(0.4, 0.2))
    assert guard([0.2, 0.1], 0.9, latched, CFG).reason is GuardReason.LATCHED

    fresh = reset()
    assert fresh == GuardState()
    decision = guard([0.0, 0.0], 0.9, fresh, CFG)  # stale rate baseline is gone
    assert decision.accepted and not decision.state.latched


def test_bad_configs_are_loud():
    with pytest.raises(ValueError, match="per channel"):
        GuardConfig(bounds=(), max_rate=())
    with pytest.raises(ValueError, match="per channel"):
        GuardConfig(bounds=((-1.0, 1.0),), max_rate=())
    with pytest.raises(ValueError, match="low < high"):
        GuardConfig(bounds=((1.0, -1.0),), max_rate=(0.5,))
    with pytest.raises(ValueError, match="non-negative"):
        GuardConfig(bounds=((-1.0, 1.0),), max_rate=(-0.5,))
    with pytest.raises(ValueError, match="min_confidence"):
        GuardConfig(bounds=((-1.0, 1.0),), max_rate=(0.5,), min_confidence=1.5)
    with pytest.raises(ValueError, match="max_misses"):
        GuardConfig(bounds=((-1.0, 1.0),), max_rate=(0.5,), max_misses=0)


def test_bad_command_shape_is_loud():
    with pytest.raises(ValueError, match="1-D"):
        guard([[0.1, 0.2]], 0.9, GuardState(), CFG)
    with pytest.raises(ValueError, match="1-D"):
        guard([0.1], 0.9, GuardState(), CFG)


def test_decision_and_state_are_immutable():
    decision = guard([0.1, 0.2], 0.9, GuardState(), CFG)
    with pytest.raises(FrozenInstanceError):
        decision.state.misses = 9
    with pytest.raises(FrozenInstanceError):
        decision.accepted = False
