# SPDX-License-Identifier: GPL-3.0-or-later
"""AICURR gates: bounded adversarial curriculum and its staged-trainer hook."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from potok_ai import train_teacher_v2 as ttv
from potok_ai.curriculum import (
    DEFAULT_WIN_RATE_CAP,
    AdversarialCurriculum,
    AdversaryConfig,
    refuse_eval_episode,
    stream_seed,
)
from potok_ai.data import make_smart_episode


def _window() -> np.ndarray:
    """Sensor-only window: transient in ch0, ramp in ch1, zeros elsewhere."""
    frames = 12
    t = np.arange(frames, dtype=np.float32)
    window = np.zeros((frames, 14), dtype=np.float32)
    window[:, 0] = np.where((t >= 3) & (t <= 6), 4.0, 0.0)
    window[:, 1] = t
    return window


def _winning_tag(seed: int, intensity: float, prefix: str = "case") -> str:
    """Pick a tag whose named win stream falls inside the budget."""
    for index in range(1000):
        tag = f"{prefix}:{index}"
        draw = np.random.default_rng(stream_seed(seed, f"win:{tag}")).random()
        if draw < intensity:
            return tag
    raise AssertionError("no winning draw found")


def _episodes(count: int = 5) -> list:
    return [make_smart_episode(100 + i, frames=24, scenario_id=i) for i in range(count)]


def _cfg(tmp_path: Path, **overrides) -> ttv.TeacherV2Config:
    kwargs = {
        "data": tmp_path / "data",
        "out": tmp_path / "out",
        "stage": "distill",
        "seed": 1,
        "epochs": 1,
        "batch": 2,
        "window": 12,
        "limit": 6,
        "adversarial": True,
    }
    kwargs.update(overrides)
    return ttv.TeacherV2Config(**kwargs)


def test_seed_streams_are_named_deterministic_and_global_rng_free() -> None:
    assert stream_seed(1, "win:a") != stream_seed(1, "augment:a")
    assert stream_seed(1, "win:a") == stream_seed(1, "win:a")
    assert stream_seed(1, "win:a") != stream_seed(2, "win:a")

    adv = AdversarialCurriculum(seed=7)
    tag = _winning_tag(7, DEFAULT_WIN_RATE_CAP)
    first = adv.perturb(_window(), tag=tag, intensity=DEFAULT_WIN_RATE_CAP)
    np.random.seed(999)
    repeat = adv.perturb(_window(), tag=tag, intensity=DEFAULT_WIN_RATE_CAP)
    assert np.array_equal(first.window, repeat.window)
    other = AdversarialCurriculum(seed=8).perturb(
        _window(), tag=tag, intensity=DEFAULT_WIN_RATE_CAP
    )
    assert not np.array_equal(first.window, other.window)


def test_stage_ramp_and_hard_win_rate_cap() -> None:
    adv = AdversarialCurriculum(seed=1)
    assert adv.intensity_for_stage("calm") == 0.0
    assert adv.intensity_for_stage("gust") == pytest.approx(0.10)
    assert adv.intensity_for_stage("shear") == pytest.approx(0.20)
    assert adv.intensity_for_stage("sensor_fault") == pytest.approx(DEFAULT_WIN_RATE_CAP)
    assert adv.intensity(6) == adv.intensity_for_stage("sensor_fault")

    # A caller asking for a larger budget cannot breach the cap.
    result = adv.perturb(_window(), tag="capped", intensity=1.0)
    assert result.cap == DEFAULT_WIN_RATE_CAP
    assert result.intensity == DEFAULT_WIN_RATE_CAP

    won = sum(
        adv.perturb(_window(), tag=f"draw:{i}", intensity=DEFAULT_WIN_RATE_CAP).won
        for i in range(600)
    )
    assert 0.20 < won / 600 <= 0.36
    assert not any(adv.perturb(_window(), tag=f"calm:{i}").won for i in range(50))

    with pytest.raises(ValueError, match="win_rate_cap"):
        AdversaryConfig(win_rate_cap=0.0)
    with pytest.raises(ValueError, match="win_rate_cap"):
        AdversaryConfig(win_rate_cap=1.0)
    with pytest.raises(ValueError, match="gust_gain"):
        AdversaryConfig(gust_gain=0.9)


def test_gust_gain_is_above_the_training_range() -> None:
    config = AdversaryConfig(dropout_prob=0.0, delay_prob=0.0, gust_shift_max=0)
    assert config.gust_gain > 1.0  # 1.25x of the 20 m/s training ceiling
    adv = AdversarialCurriculum(seed=2, config=config)
    tag = _winning_tag(2, DEFAULT_WIN_RATE_CAP, "gust")
    window = _window()
    result = adv.perturb(window, tag=tag, intensity=DEFAULT_WIN_RATE_CAP)

    assert result.won and result.flags is not None
    assert result.flags.gust and not result.flags.dropout and not result.flags.delay
    baseline = float(window[:, 0].mean())
    before = float(np.abs(window[:, 0] - baseline).max())
    after = float(np.abs(result.window[:, 0] - baseline).max())
    assert after == pytest.approx(config.gust_gain * before)
    assert after > before


def test_delay_jitter_and_dropout_apply_only_on_a_win() -> None:
    delay = AdversarialCurriculum(
        seed=3, config=AdversaryConfig(delay_prob=1.0, dropout_prob=0.0, gust_gain=1.0)
    )
    tag = _winning_tag(3, DEFAULT_WIN_RATE_CAP, "delay")
    delayed = delay.perturb(_window(), tag=tag, intensity=DEFAULT_WIN_RATE_CAP)
    assert delayed.won and delayed.flags is not None and delayed.flags.delay
    assert not np.array_equal(delayed.window[:, 1], _window()[:, 1])

    dropout = AdversarialCurriculum(
        seed=4, config=AdversaryConfig(dropout_prob=1.0, delay_prob=0.0, gust_gain=1.0)
    )
    tag = _winning_tag(4, DEFAULT_WIN_RATE_CAP, "dropout")
    dropped = dropout.perturb(_window(), tag=tag, intensity=DEFAULT_WIN_RATE_CAP)
    assert dropped.won and dropped.flags is not None and dropped.flags.dropout
    assert np.unique(dropped.window[:, 1]).size == 1

    losing = [
        f"lose:{i}"
        for i in range(50)
        if not delay.perturb(_window(), tag=f"lose:{i}", intensity=DEFAULT_WIN_RATE_CAP).won
    ]
    assert losing
    untouched = delay.perturb(_window(), tag=losing[0], intensity=DEFAULT_WIN_RATE_CAP)
    assert not untouched.won and untouched.flags is None
    assert np.array_equal(untouched.window, _window())


@pytest.mark.parametrize("sid", [16, 17, 18, 19])
def test_eval_cells_are_refused_before_selection(sid: int) -> None:
    adv = AdversarialCurriculum(seed=1)
    episode = make_smart_episode(9, frames=24, scenario_id=sid)
    with pytest.raises(ValueError, match="eval cell"):
        refuse_eval_episode(episode)
    with pytest.raises(ValueError, match="eval cell"):
        adv.select([episode], epoch=3)


def test_held_out_flag_and_scenario_name_fallback_are_refused() -> None:
    adv = AdversarialCurriculum(seed=1)
    flagged = make_smart_episode(9, frames=24, scenario_id=5)
    flagged.held_out = True
    named = make_smart_episode(9, frames=24, scenario_id=0)
    named.scenario_id = None
    named.scenario = "scenario_18"
    for episode in (flagged, named):
        with pytest.raises(ValueError, match="eval cell"):
            refuse_eval_episode(episode)
    assert adv.select([make_smart_episode(1, frames=24, scenario_id=0)], epoch=0)


def test_trainer_stage_hook_reports_bounded_budget(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ttv, "load_episodes", lambda _path: _episodes())
    metrics = ttv.run(_cfg(tmp_path))

    ramps = {
        name: metrics.stages[name]["details"]["adversarial"]["intensity"]
        for name in ttv.STAGE_ORDER
    }
    assert ramps == {
        "sft": pytest.approx(0.0),
        "privileged": pytest.approx(0.10),
        "dagger": pytest.approx(0.20),
        "distill": pytest.approx(DEFAULT_WIN_RATE_CAP),
    }
    for name in ttv.STAGE_ORDER:
        detail = metrics.stages[name]["details"]["adversarial"]
        assert detail["win_rate_cap"] == DEFAULT_WIN_RATE_CAP
        assert 0 <= detail["won"] <= detail["offered"]
        assert detail["seed"] == 1

    payload = (tmp_path / "out" / "metrics.json").read_text()
    assert '"adversarial"' in payload
    assert metrics.config_hash != ttv.config_hash(_cfg(tmp_path, adversarial=False))


def test_eval_and_verifier_windows_are_never_augmented(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    episodes = _episodes()
    baseline = ttv._window_batches(episodes, cfg)

    adv = AdversarialCurriculum(seed=cfg.seed)
    stats = {"offered": 0, "won": 0}
    token = ttv._ADV_CONTEXT.set((adv, "distill", stats))
    try:
        eval_batches = ttv._window_batches(episodes, cfg)
        eval_offered = stats["offered"]
        train_batches = ttv._window_batches(episodes, cfg, train=True)
    finally:
        ttv._ADV_CONTEXT.reset(token)

    assert all(torch.equal(a[0], b[0]) for a, b in zip(baseline, eval_batches, strict=True))
    assert eval_offered == 0  # no eval window was even counted
    assert any(not torch.equal(a[0], b[0]) for a, b in zip(baseline, train_batches, strict=True))
    assert stats["offered"] > 0 and stats["won"] > 0


def test_adversarial_is_opt_in_from_the_cli(tmp_path: Path, monkeypatch) -> None:
    argv = ["--data", str(tmp_path / "data"), "--out", str(tmp_path / "out")]
    off = ttv.TeacherV2Config.from_args(ttv.build_parser().parse_args(argv))
    on = ttv.TeacherV2Config.from_args(ttv.build_parser().parse_args([*argv, "--adversarial"]))
    assert not off.adversarial and on.adversarial
    assert ttv.config_hash(off) != ttv.config_hash(on)

    monkeypatch.setattr(ttv, "load_episodes", lambda _path: _episodes())
    metrics = ttv.run(off)
    assert "adversarial" not in metrics.stages["sft"]["details"]
