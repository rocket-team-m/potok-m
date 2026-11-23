"""Deterministic, seeded scenario curriculum over the four difficulty stages."""

import pytest

from potok_ai.curriculum import STAGES, Curriculum, stage_of
from potok_ai.data import make_smart_episode


def _episodes(count: int = 12) -> list:
    return [
        make_smart_episode(seed=100 + index, frames=8, scenario_id=index % 4)
        for index in range(count)
    ]


def test_stage_names_and_scenario_mapping():
    assert STAGES == ("calm", "gust", "shear", "sensor_fault")
    assert [stage_of(episode) for episode in _episodes(4)] == list(STAGES)
    assert stage_of(make_smart_episode(1, frames=8, scenario_id=16)) == "calm"


def test_cumulative_eligibility_is_capped():
    curriculum = Curriculum(seed=1)
    assert curriculum.active(0) == STAGES[:1]
    assert curriculum.active(1) == STAGES[:2]
    assert curriculum.active(3) == STAGES
    assert curriculum.active(99) == STAGES
    assert curriculum.stage_at(2) == "shear"
    assert curriculum.stage_at(99) == "sensor_fault"
    with pytest.raises(ValueError, match="epochs_per_stage"):
        Curriculum(seed=1, epochs_per_stage=0)


def test_order_is_deterministic_and_seed_dependent():
    episodes = _episodes()
    first = [e.index for e in Curriculum(seed=7).select(episodes, 2)]
    repeat = [e.index for e in Curriculum(seed=7).select(episodes, 2)]
    other = [e.index for e in Curriculum(seed=9).select(episodes, 2)]
    assert first == repeat
    assert first != other
    assert all(
        stage_of(episode) in STAGES[:3] for episode in Curriculum(seed=7).select(episodes, 2)
    )


def test_inclusion_is_monotonic_and_pacing_is_configurable():
    episodes = _episodes()
    curriculum = Curriculum(seed=3)
    sets = [{episode.index for episode in curriculum.select(episodes, e)} for e in range(4)]
    assert sets[0] < sets[1] < sets[2] < sets[3]

    slow = Curriculum(seed=3, epochs_per_stage=2)
    assert slow.active(0) == slow.active(1) == STAGES[:1]
    assert slow.active(2) == STAGES[:2]
    assert slow.active(6) == STAGES
