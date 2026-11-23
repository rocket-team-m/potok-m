# SPDX-License-Identifier: GPL-3.0-or-later
"""DAgger corrective dataset builder (WORLD02/03, R2 arXiv:1011.0686).

Pure data layer: no trainer, optimiser or torch training loop is touched. The
student rollout and the privileged teacher labeller are injected as callables;
this module orchestrates the DAgger mechanism around them:

  1. each round rolls the student out on TRAINING scenarios and keeps the states
     the student actually visited (on-policy / student-induced states);
  2. each visited state is labelled by the privileged teacher -- the expert --
     never by the student's own prediction;
  3. rounds are aggregated with per-sample expert-label provenance;
  4. a deterministic, disjoint fit / held-out partition of the training cells
     supports the covariate-shift diagnostic.

Two "held out" notions are deliberately separate:

  * EVAL cells (scenario ids 16..19) are the final scoring set and are REFUSED
    here -- no rollout, no teacher call, no sample. This is the no-eval-
    contamination rule.
  * the inner held-out split is a disjoint subset of the TRAINING cells (0..15),
    used only to measure loss on student-induced states; it never trains.

Determinism: every stochastic stream is a named child of the top-level seed
(``random.Random(f"{seed}:dagger:{i}:rollout:{j}:episode:{idx}")`` and
``random.Random(f"{seed}:dagger:split")``). No global RNG is read or reseeded.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .data import Episode

EVAL_SCENARIO_IDS: frozenset[int] = frozenset({16, 17, 18, 19})
EXPERT_TEACHER = "privileged_teacher"
LABEL_SOURCE_EXPERT = "expert"
SPLIT_FIT = "fit"
SPLIT_HELD_OUT = "held_out"


@dataclass(frozen=True, eq=False)
class RolloutState:
    """One state produced by a student rollout.

    ``observation`` is what the student saw (sensor history only); ``privileged``
    is the full-state / true-wind snapshot available only to the teacher. The
    labeller consumes ``privileged`` and it is never stored in the dataset, so
    no privilege leaks into student training. ``authoritative`` is the verifier
    target, kept for diagnostics only.
    """

    observation: np.ndarray
    privileged: np.ndarray
    authoritative: np.ndarray | None = None


@dataclass(frozen=True)
class Provenance:
    """Expert-label provenance attached to every aggregated sample."""

    sample_id: str
    seed: int
    round: int
    rollout: int
    step: int
    episode: int | None
    scenario_id: int | None
    split: str
    expert: str
    label_source: str


@dataclass(frozen=True, eq=False)
class DaggerSample:
    """One (student-induced observation, privileged-teacher correction) row."""

    observation: np.ndarray
    correction: np.ndarray
    authoritative: np.ndarray | None
    provenance: Provenance


type RolloutFn = Callable[[Episode, random.Random], Sequence[RolloutState]]
type TeacherFn = Callable[[np.ndarray], np.ndarray]


def _scenario_id(episode: Episode) -> int | None:
    """Numeric scenario id; v1 fixtures fall back to the name's digit suffix."""
    if episode.scenario_id is not None:
        return int(episode.scenario_id)
    digits = "".join(ch for ch in (episode.scenario or "") if ch.isdigit())
    return int(digits) if digits else None


def is_eval_episode(episode: Episode) -> bool:
    """True for a final-evaluation cell (scenario 16..19 or a held-out flag)."""
    sid = _scenario_id(episode)
    return bool(episode.held_out) or (sid is not None and sid in EVAL_SCENARIO_IDS)


def split_scenarios(
    seed: int, scenario_ids: Sequence[int], n_held_out: int = 2
) -> tuple[frozenset[int], frozenset[int]]:
    """Partition train scenario ids into disjoint (fit, held_out) sets.

    Refuses eval cells so a caller cannot accidentally reserve a scoring cell
    for its diagnostic split.
    """
    ids = sorted({int(s) for s in scenario_ids})
    bad = [s for s in ids if s in EVAL_SCENARIO_IDS]
    if bad:
        raise ValueError(f"eval scenario ids cannot enter a training split: {bad}")
    if not ids:
        raise ValueError("no training scenario ids to split")
    if n_held_out < 1 or n_held_out >= len(ids):
        raise ValueError(f"n_held_out must be in 1..{len(ids) - 1}, got {n_held_out}")
    held = random.Random(f"{seed}:dagger:split").sample(ids, n_held_out)
    held_set = frozenset(held)
    return frozenset(ids) - held_set, held_set


@dataclass(eq=False)
class DaggerDataset:
    """Aggregated corrective dataset plus its fit/held-out scenario partition."""

    seed: int
    samples: list[DaggerSample]
    fit_scenarios: frozenset[int]
    held_out_scenarios: frozenset[int]
    rounds: int
    rollouts_per_round: int

    def fit_samples(self) -> list[DaggerSample]:
        """Rows usable for student retraining (never from the held-out split)."""
        return [s for s in self.samples if s.provenance.split == SPLIT_FIT]

    def held_out_samples(self) -> list[DaggerSample]:
        """Disjoint diagnostic rows: loss here measures student-induced shift."""
        return [s for s in self.samples if s.provenance.split == SPLIT_HELD_OUT]

    def fingerprint(self) -> str:
        """Deterministic SHA-256 over ids and arrays; no timestamp or PID."""
        digest = hashlib.sha256()
        for sample in self.samples:
            digest.update(sample.provenance.sample_id.encode())
            digest.update(np.asarray(sample.observation, dtype=np.float32).tobytes())
            digest.update(np.asarray(sample.correction, dtype=np.float32).tobytes())
            if sample.authoritative is not None:
                digest.update(np.asarray(sample.authoritative, dtype=np.float32).tobytes())
        return digest.hexdigest()


def build_dagger_dataset(
    *,
    seed: int,
    episodes: Sequence[Episode],
    rollout: RolloutFn,
    teacher: TeacherFn,
    rounds: int = 1,
    rollouts_per_round: int = 1,
    n_held_out: int = 2,
    expert: str = EXPERT_TEACHER,
) -> DaggerDataset:
    """Aggregate privileged-teacher corrections on student-induced states.

    ``rollout(episode, rng)`` returns the states the student visited on one
    training episode; ``rng`` is a named child stream so caller-side rollout
    noise is deterministic. ``teacher(privileged)`` returns the expert label for
    one visited state; it is the sole source of ``correction``.

    Eval cells are refused before either callable runs, so no scoring scenario
    can leak into the aggregate.
    """
    if rounds < 1 or rollouts_per_round < 1:
        raise ValueError("rounds and rollouts_per_round must be positive")
    if not episodes:
        raise ValueError("no episodes supplied")

    contaminated = [e for e in episodes if is_eval_episode(e)]
    if contaminated:
        raise ValueError(
            f"{len(contaminated)} eval episode(s) in the DAgger rollout set; "
            "no eval contamination: use training cells 0..15 only"
        )
    scenario_ids = [sid for e in episodes if (sid := _scenario_id(e)) is not None]
    if len(scenario_ids) != len(episodes):
        raise ValueError("every rollout episode needs a scenario id")
    fit, held = split_scenarios(seed, scenario_ids, n_held_out)

    samples: list[DaggerSample] = []
    for i in range(rounds):
        for j in range(rollouts_per_round):
            for episode in episodes:
                sid = _scenario_id(episode)
                split = SPLIT_HELD_OUT if sid in held else SPLIT_FIT
                idx = episode.index
                rng = random.Random(f"{seed}:dagger:{i}:rollout:{j}:episode:{idx}")
                for step, state in enumerate(rollout(episode, rng)):
                    correction = np.asarray(teacher(state.privileged), dtype=np.float32)
                    if correction.size == 0 or not np.isfinite(correction).all():
                        raise ValueError(
                            f"non-finite or empty teacher label at {seed}:dagger:{i}:"
                            f"rollout:{j}:episode:{idx}:step:{step}"
                        )
                    sample_id = f"{seed}:dagger:{i}:rollout:{j}:episode:{idx}:step:{step}"
                    samples.append(
                        DaggerSample(
                            observation=np.asarray(state.observation, dtype=np.float32),
                            correction=correction,
                            authoritative=(
                                None
                                if state.authoritative is None
                                else np.asarray(state.authoritative, dtype=np.float32)
                            ),
                            provenance=Provenance(
                                sample_id=sample_id,
                                seed=seed,
                                round=i,
                                rollout=j,
                                step=step,
                                episode=idx,
                                scenario_id=sid,
                                split=split,
                                expert=expert,
                                label_source=LABEL_SOURCE_EXPERT,
                            ),
                        )
                    )

    return DaggerDataset(
        seed=seed,
        samples=samples,
        fit_scenarios=fit,
        held_out_scenarios=held,
        rounds=rounds,
        rollouts_per_round=rollouts_per_round,
    )
