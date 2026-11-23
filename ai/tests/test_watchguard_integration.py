# SPDX-License-Identifier: GPL-3.0-or-later
"""Watchguard in the dispatch path: real model outputs, latch/reset, eval purity.

The student proposes a two-channel correction and a confidence every tick; the
guard is the only gate to the controller. An accepted, healthy decision forwards
the model command, everything else emits zeros. Screening outputs must not touch
the model, its input frames or the evaluation stream.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from potok_ai.data import make_smart_episode
from potok_ai.student import FRAME_FEATURES, StatefulTCN, StudentConfig
from potok_ai.watchguard import GuardConfig, GuardReason, GuardState, guard, reset

BOUNDS = ((-1.0, 1.0), (-1.0, 1.0))
RATE = (1.0, 1.0)  # at least the widest swing a bounded two-channel command can make


def _cfg(min_confidence: float = 0.0) -> GuardConfig:
    return GuardConfig(bounds=BOUNDS, max_rate=RATE, min_confidence=min_confidence)


def _student(seed: int = 7, features: int = FRAME_FEATURES) -> StatefulTCN:
    return StatefulTCN(
        StudentConfig(seed=seed, width=16, rank=8, in_features=features, loops=2)
    ).eval()


def _frames(seed: int, count: int, features: int = FRAME_FEATURES) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    frames = torch.randn(count, features, generator=gen)
    frames[:, :8] *= 0.5
    if features > 9:
        frames[:, 9] = 1.0
    return frames


def _dispatch(output, state, config, *, sensors_valid=True, ai_healthy=True):
    """One tick of the seam: screen a real output, then decide what to send."""
    corrections, confidence = output
    decision = guard(corrections.numpy(), float(confidence), state, config)
    send = decision.command if decision.accepted and sensors_valid and ai_healthy else (0.0, 0.0)
    return send, decision


def test_real_model_outputs_reach_the_controller_only_when_guard_accepts():
    model, permissive, strict = _student(), _cfg(), _cfg(min_confidence=0.99)
    with torch.no_grad():
        outputs = [model.step(frame) for frame in _frames(3, 6)]

    state = reset()
    for corrections, confidence in outputs:
        assert torch.isfinite(corrections).all() and torch.isfinite(confidence).all()
        sent, decision = _dispatch((corrections, confidence), state, permissive)
        assert decision.reason is GuardReason.OK
        assert sent == pytest.approx(tuple(corrections.tolist()))
        assert all(np.isfinite(sent)) and all(-1.0 <= value <= 1.0 for value in sent)
        state = decision.state
        assert state.misses == 0 and state.last_command == sent

    state = reset()
    for misses, (corrections, confidence) in enumerate(outputs[:2], start=1):
        assert float(confidence) < 0.99  # fixed fixture: real confidences stay near 0.5
        sent, decision = _dispatch((corrections, confidence), state, strict)
        assert decision.reason is GuardReason.LOW_CONFIDENCE
        assert sent == (0.0, 0.0) and all(np.isfinite(sent))
        state = decision.state
        assert state.misses == misses and not state.latched


def test_three_rejections_latch_dispatch_and_reset_clears_it():
    model, strict = _student(), _cfg(min_confidence=0.99)
    with torch.no_grad():
        corrections, _ = model.step(_frames(4, 1)[0])
    valid = (corrections, 1.0)  # real corrections, confidently proposed

    state = reset()
    for expected_misses in (1, 2, 3):
        sent, decision = _dispatch((corrections, 0.1), state, strict)
        assert decision.reason is GuardReason.LOW_CONFIDENCE
        assert sent == (0.0, 0.0)
        state = decision.state
        assert state.misses == expected_misses and state.latched == (expected_misses == 3)

    sent, decision = _dispatch(valid, state, strict)
    assert decision.reason is GuardReason.LATCHED and decision.state == state
    assert sent == (0.0, 0.0) and all(np.isfinite(sent))

    sent, decision = _dispatch(valid, reset(), strict)
    assert decision.reason is GuardReason.OK
    assert sent == pytest.approx(tuple(corrections.tolist()))
    assert not decision.state.latched and decision.state.misses == 0


@pytest.mark.parametrize("poison", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_model_output_is_vetoed_with_a_finite_zero_command(poison):
    model, config = _student(), _cfg()
    with torch.no_grad():
        corrections, confidence = model.step(_frames(5, 1)[0])

    bad = corrections.clone()
    bad[0] = poison
    sent, decision = _dispatch((bad, confidence), reset(), config)
    assert decision.reason is GuardReason.NON_FINITE
    assert sent == (0.0, 0.0) and all(np.isfinite(sent))
    assert decision.state.misses == 1 and decision.state.last_command is None

    sent, decision = _dispatch((corrections, poison), reset(), config)
    assert decision.reason is GuardReason.NON_FINITE
    assert sent == (0.0, 0.0) and all(np.isfinite(sent))
    assert np.isfinite(corrections.numpy()).all()  # the model output was not mutated


def test_eval_stream_and_model_state_survive_the_guard_path_untouched():
    episode = make_smart_episode(seed=42, frames=16, scenario_id=16, held_out=True)
    assert episode.held_out
    original = episode.frames.copy()
    width = original.shape[1]
    guarded, plain = _student(seed=11, features=width), _student(seed=11, features=width)
    config, state = _cfg(), reset()

    for frame in original:
        with torch.no_grad():
            proposed = guarded.step(frame)
            reference = plain.step(frame)
        sent, decision = _dispatch(proposed, state, config)
        assert decision.reason is GuardReason.OK
        assert sent == pytest.approx(tuple(proposed[0].tolist()))
        assert torch.equal(proposed[0], reference[0]) and torch.equal(proposed[1], reference[1])
        state = decision.state

    np.testing.assert_array_equal(episode.frames, original)
    assert torch.equal(guarded.h, plain.h)

    latched = GuardState(misses=3, latched=True)
    with torch.no_grad():
        proposed = guarded.step(original[0])
        reference = plain.step(original[0])
    sent, decision = _dispatch(proposed, latched, config)
    assert decision.reason is GuardReason.LATCHED and sent == (0.0, 0.0)
    assert torch.equal(proposed[0], reference[0]) and torch.equal(proposed[1], reference[1])
    np.testing.assert_array_equal(episode.frames, original)


def test_unhealthy_or_invalid_sensor_ticks_never_dispatch_a_command():
    model, config = _student(), _cfg()
    with torch.no_grad():
        output = model.step(_frames(6, 1)[0])

    for sensors_valid, ai_healthy in ((False, True), (True, False), (False, False)):
        sent, decision = _dispatch(
            output, reset(), config, sensors_valid=sensors_valid, ai_healthy=ai_healthy
        )
        assert decision.reason is GuardReason.OK  # the guard itself is healthy
        assert sent == (0.0, 0.0) and all(np.isfinite(sent))
