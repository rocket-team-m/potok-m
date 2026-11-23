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

from .data import SMART_COND_FEATURES, SMART_FEATURES, Episode

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
    # Filled by on-policy rollouts only. ``action`` is the student's command at
    # this state; ``recorded_frame`` is the sensor frame the recorded trace had
    # at the same tick, kept ONLY to measure student-induced drift. Neither is
    # ever concatenated into an observation.
    action: np.ndarray | None = None
    recorded_frame: np.ndarray | None = None


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
    rollout_sha256: str = ""


@dataclass(frozen=True, eq=False)
class DaggerSample:
    """One (student-induced observation, privileged-teacher correction) row."""

    observation: np.ndarray
    correction: np.ndarray
    authoritative: np.ndarray | None
    provenance: Provenance
    action: np.ndarray | None = None
    drift: float | None = None


type RolloutFn = Callable[[Episode, random.Random], Sequence[RolloutState]]
type TeacherFn = Callable[[np.ndarray], np.ndarray]
type ActionFn = Callable[[np.ndarray, random.Random], np.ndarray]
type WorldStepFn = Callable[[np.ndarray, np.ndarray, random.Random], np.ndarray]
type PrivilegedFn = Callable[[Episode, int, random.Random], np.ndarray]
type TargetFn = Callable[[Episode, int, np.ndarray], np.ndarray | None]


def sample_sha256(
    sample_id: str,
    observation: np.ndarray,
    privileged: np.ndarray,
    action: np.ndarray | None,
    authoritative: np.ndarray | None,
) -> str:
    """SHA-256 over the named sample id and every array that produced it."""
    digest = hashlib.sha256()
    digest.update(sample_id.encode())
    for array in (observation, action, privileged, authoritative):
        if array is not None:
            digest.update(np.asarray(array, dtype=np.float32).tobytes())
    return digest.hexdigest()


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


def on_policy_rollout(
    *,
    window: int,
    student_action: ActionFn,
    world_step: WorldStepFn,
    privileged_snapshot: PrivilegedFn,
    authoritative_target: TargetFn | None = None,
    on_episode_start: Callable[[Episode], None] | None = None,
    horizon: int | None = None,
) -> RolloutFn:
    """Build a rollout whose visited states are caused by the student itself.

    A recorded episode supplies only the first ``window`` sensor frames. At each
    tick the student sees sensor history (14 features + 6 conditioning per
    frame) and emits an action; ``world_step`` advances the sensor frame from
    that action, so the next observation is student-induced and can drift off
    the recorded trace. The privileged snapshot for the visited tick comes from
    ``privileged_snapshot`` (truth sidecar in production) and is passed ONLY to
    the teacher labeler -- it never reaches ``student_action`` or the stored
    observation.

    All noise flows through the named child ``rng`` supplied by
    :func:`build_dagger_dataset`; no global RNG is read. Eval cells are refused
    here too, so a direct caller of the rollout cannot bypass the dataset gate.
    """
    if window < 1:
        raise ValueError(f"window must be positive, got {window}")
    if horizon is not None and horizon < 1:
        raise ValueError(f"horizon must be positive or None, got {horizon}")

    def rollout(episode: Episode, rng: random.Random) -> list[RolloutState]:
        if is_eval_episode(episode):
            raise ValueError(
                f"eval cell {episode.scenario_id} cannot enter an on-policy DAgger rollout"
            )
        frames = np.asarray(episode.frames, dtype=np.float32)
        if frames.ndim != 2 or frames.shape[1] < SMART_FEATURES:
            raise ValueError(f"expected (T, >= {SMART_FEATURES}) sensor frames, got {frames.shape}")
        cond = episode.conditioning
        if cond is None:
            raise ValueError("on-policy rollout needs episode conditioning (student input)")
        cond_array = np.asarray(cond, dtype=np.float32).reshape(-1)
        if cond_array.shape != (SMART_COND_FEATURES,):
            raise ValueError(
                f"conditioning must have {SMART_COND_FEATURES} values, got {cond_array.shape}"
            )
        available = len(frames) - window
        if available < 1:
            raise ValueError(f"episode of {len(frames)} frames is shorter than window {window}")
        if on_episode_start is not None:
            on_episode_start(episode)

        steps = available if horizon is None else min(horizon, available)
        history = frames[:window, :SMART_FEATURES].copy()
        states: list[RolloutState] = []
        for step in range(steps):
            observation = np.concatenate(
                [history, np.tile(cond_array, (window, 1))], axis=1
            ).astype(np.float32)
            action = np.asarray(student_action(observation, rng), dtype=np.float32).reshape(-1)
            if action.size != 2 or not np.isfinite(action).all():
                raise ValueError(f"student action at step {step} must be two finite commands")
            privileged = np.asarray(
                privileged_snapshot(episode, step, rng), dtype=np.float32
            ).reshape(-1)
            if privileged.size == 0 or not np.isfinite(privileged).all():
                raise ValueError(f"privileged snapshot at step {step} is empty or non-finite")
            tick = window - 1 + step
            target = (
                authoritative_target(episode, tick, history[-1])
                if authoritative_target is not None
                else None
            )
            states.append(
                RolloutState(
                    observation=observation,
                    privileged=privileged,
                    authoritative=(
                        None if target is None else np.asarray(target, dtype=np.float32)
                    ),
                    action=action.copy(),
                    recorded_frame=frames[tick, :SMART_FEATURES].copy(),
                )
            )
            next_frame = np.asarray(world_step(history[-1], action, rng), dtype=np.float32)
            if next_frame.shape != (SMART_FEATURES,) or not np.isfinite(next_frame).all():
                raise ValueError(f"world step at {step} must return {SMART_FEATURES} finite values")
            history = np.concatenate([history[1:], next_frame[None, :]], axis=0)
        return states

    return rollout


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
            digest.update(sample.provenance.rollout_sha256.encode())
            for array in (
                sample.observation,
                sample.action,
                sample.correction,
                sample.authoritative,
            ):
                if array is not None:
                    digest.update(np.asarray(array, dtype=np.float32).tobytes())
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
                    observation = np.asarray(state.observation, dtype=np.float32)
                    authoritative = (
                        None
                        if state.authoritative is None
                        else np.asarray(state.authoritative, dtype=np.float32)
                    )
                    action = (
                        None if state.action is None else np.asarray(state.action, dtype=np.float32)
                    )
                    sha = sample_sha256(
                        sample_id, observation, state.privileged, action, authoritative
                    )
                    drift = (
                        None
                        if state.recorded_frame is None
                        else float(
                            np.linalg.norm(
                                observation[-1, :SMART_FEATURES]
                                - np.asarray(state.recorded_frame, dtype=np.float32)
                            )
                        )
                    )
                    samples.append(
                        DaggerSample(
                            observation=observation,
                            correction=correction,
                            authoritative=authoritative,
                            action=action,
                            drift=drift,
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
                                rollout_sha256=sha,
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
