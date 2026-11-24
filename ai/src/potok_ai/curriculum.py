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

import hashlib
import random
from dataclasses import dataclass, field

import numpy as np

from .augment import DEFAULT_GUST_GAIN, AugmentConfig, AugmentFlags, augment

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


# ---------------------------------------------------------------------------
# Bounded adversarial curriculum (WORLD09 #5 / IDEAL_GENERATOR S6 / WORLD05 A3).
# ---------------------------------------------------------------------------
# The adversary does not invent physics. It decides which fraction of TRAINING
# windows receive the existing sensor-domain transforms (gust gain above the
# training range, delay jitter, dropout) and derives every draw from a named
# child seed. A "win" means the adversary was allowed to perturb one window;
# the win rate is capped so the majority of the batch stays clean and the
# supervised task remains learnable. This is a sampling cap, NOT a measured RL
# win rate from rollouts and NOT a flight result.

EVAL_SCENARIO_IDS: frozenset[int] = frozenset({16, 17, 18, 19})
DEFAULT_WIN_RATE_CAP = 0.30


def stream_seed(seed: int, name: str) -> int:
    """Stable named child seed; no global RNG is read or reseeded."""
    digest = hashlib.sha256(f"{seed}:curriculum:adv:{name}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _scenario_id(episode) -> int | None:
    if episode.scenario_id is not None:
        return int(episode.scenario_id)
    digits = "".join(ch for ch in (episode.scenario or "") if ch.isdigit())
    return int(digits) if digits else None


def refuse_eval_episode(episode) -> None:
    """Refuse a final-eval cell (id 16..19 or held_out) before any training use."""
    sid = _scenario_id(episode)
    if episode.held_out or (sid is not None and sid in EVAL_SCENARIO_IDS):
        raise ValueError(
            f"eval cell {sid} cannot enter adversarial training; use training cells 0..15"
        )


def refuse_eval_episodes(episodes) -> None:
    for episode in episodes:
        refuse_eval_episode(episode)


@dataclass(frozen=True)
class AdversaryConfig:
    """Sampling budget and sensor-domain transforms; every default is documented."""

    win_rate_cap: float = DEFAULT_WIN_RATE_CAP
    gust_gain: float = DEFAULT_GUST_GAIN  # 1.25x: above the 5..20 m/s training range
    dropout_prob: float = 0.5
    delay_prob: float = 0.5
    delay_max_steps: int = 2
    gust_shift_max: int = 2

    def __post_init__(self) -> None:
        if not 0.0 < self.win_rate_cap < 1.0:
            raise ValueError(f"win_rate_cap must lie in (0, 1), got {self.win_rate_cap}")
        if self.gust_gain < 1.0:
            raise ValueError("gust_gain must be >= 1 (above-range multiplier)")
        for name in ("dropout_prob", "delay_prob"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.delay_max_steps < 0 or self.gust_shift_max < 0:
            raise ValueError("delay_max_steps and gust_shift_max must be non-negative")

    def augmentation(self) -> AugmentConfig:
        """Reuse augment.py's channel set: raw sensors only, derived physics untouched."""
        return AugmentConfig(
            dropout_prob=self.dropout_prob,
            delay_prob=self.delay_prob,
            gust_prob=1.0,
            gust_gain=self.gust_gain,
            delay_max_steps=self.delay_max_steps,
            gust_shift_max=self.gust_shift_max,
        )


@dataclass(frozen=True)
class AdversaryResult:
    window: np.ndarray
    won: bool
    intensity: float
    cap: float
    flags: AugmentFlags | None = None


@dataclass(frozen=True)
class AdversarialCurriculum:
    """Per-window adversary with a hard win-rate cap and named seed streams."""

    seed: int
    config: AdversaryConfig = field(default_factory=AdversaryConfig)
    epochs_per_stage: int = 1

    def __post_init__(self) -> None:
        if self.epochs_per_stage < 1:
            raise ValueError("epochs_per_stage must be positive")

    def _budget(self, stage_index: int) -> float:
        steps = len(STAGES) - 1
        return self.config.win_rate_cap * min(stage_index, steps) / steps

    def intensity(self, epoch: int) -> float:
        """Ramp 0 -> cap across the four cumulative stages; calm stage stays clean."""
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        return self._budget(epoch // self.epochs_per_stage)

    def intensity_for_stage(self, stage: str) -> float:
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
        return self._budget(STAGES.index(stage))

    def perturb(
        self,
        window,
        *,
        tag: str,
        epoch: int = 0,
        intensity: float | None = None,
    ) -> AdversaryResult:
        """Possibly perturb one sensor window; every draw is a named child stream."""
        budget = self.intensity(epoch) if intensity is None else float(intensity)
        # ponytail: the cap holds even if a caller asks for a larger budget; V_max is
        # not clamped here because the dataset carries no wind vector (WORLD05 A3).
        budget = min(self.config.win_rate_cap, max(0.0, budget))
        draw = np.random.default_rng(stream_seed(self.seed, f"win:{tag}")).random()
        if draw >= budget:
            return AdversaryResult(np.asarray(window), False, budget, self.config.win_rate_cap)
        augmented = augment(
            window,
            stream_seed(self.seed, f"augment:{tag}"),
            self.config.augmentation(),
        )
        return AdversaryResult(
            augmented.window, True, budget, self.config.win_rate_cap, augmented.flags
        )

    def select(self, episodes: list, epoch: int) -> list:
        """Stage-eligible episodes; eval cells are refused before any shuffling."""
        refuse_eval_episodes(episodes)
        return Curriculum(seed=self.seed, epochs_per_stage=self.epochs_per_stage).select(
            episodes, epoch
        )
