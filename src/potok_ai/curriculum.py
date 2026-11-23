"""Deterministic scenario curriculum (SPEC_ENGINE_IDEAL.md §4).

Difficulty grows in four stages::

    calm -> gust -> shear -> sensor_fault

A scenario is assigned a stage from its numeric id (``id % 4``); the simulator
owns the id, so this is a stable placeholder mapping until the wind-cell
semantics land in the v2 metadata. At epoch ``e`` the eligible set is the union
of stages up to ``e // epochs_per_stage`` -- earlier material stays in the mix --
and the eligible episodes are shuffled by a seed local to ``(seed, epoch)``. No
global RNG is involved, so a run is reproducible from ``--seed`` alone.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

STAGES: tuple[str, ...] = ("calm", "gust", "shear", "sensor_fault")


def stage_of(episode) -> str:
    """Stage of one episode, derived from its scenario id or name."""
    sid = episode.scenario_id
    if sid is None:
        digits = "".join(ch for ch in (episode.scenario or "") if ch.isdigit())
        sid = int(digits) if digits else 0
    return STAGES[int(sid) % len(STAGES)]


@dataclass(frozen=True)
class Curriculum:
    """Seeded scheduler: stage eligibility plus a per-epoch episode order."""

    seed: int
    epochs_per_stage: int = 1
    stages: tuple[str, ...] = STAGES

    def __post_init__(self) -> None:
        if self.epochs_per_stage < 1:
            raise ValueError("epochs_per_stage must be positive")

    def active(self, epoch: int) -> tuple[str, ...]:
        """Stages eligible at ``epoch`` (cumulative, never shrinking)."""
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        index = min(epoch // self.epochs_per_stage, len(self.stages) - 1)
        return self.stages[: index + 1]

    def stage_at(self, epoch: int) -> str:
        return self.active(epoch)[-1]

    def select(self, episodes: list, epoch: int) -> list:
        """Eligible episodes in a seed- and epoch-deterministic order."""
        allowed = set(self.active(epoch))
        chosen = [episode for episode in episodes if stage_of(episode) in allowed]
        random.Random(f"{self.seed}:{epoch}").shuffle(chosen)
        return chosen
