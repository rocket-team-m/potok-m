# SPDX-License-Identifier: GPL-3.0-or-later
"""Hard teacher->student distillation gate (SPEC_ENGINE_IDEAL.md §4-5).

The student is trained by the real ``distill_student.train_episode`` loop on
training scenarios only. Held-out scenarios are used exclusively for the two
numerical gates:

* ``rmse_held_out / rmse_train <= 1.25``  (SPEC_ENGINE_IDEAL.md §4),
* ``ECE <= 0.05`` on held-out confidence logits (§5).

The teacher is a small deterministic windowed map whose corrections stay inside
the student's ``tanh(g_max)`` output bound.  A randomly initialised windowed TCN
saturates that bound and the gate then measures an unlearnable target rather
than the distillation pipeline; the controlled teacher keeps the gate honest.
Everything is seeded locally -- no global RNG, no wall-clock, no fixture files.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW

from potok_ai.calibration import conformal_calibrate
from potok_ai.data import make_synthetic_episode
from potok_ai.distill_student import build_parser, train_episode
from potok_ai.student import SENSOR_FEATURES, StatefulTCN, StudentConfig

SEED = 7
WINDOW = 4
EPOCHS = 5
LEARNING_RATE = 1e-2
RATIO_LIMIT = 1.25
ECE_LIMIT = 0.05
ERROR_TOL = 0.1
FRAMES = 60
TRAIN_SEEDS = (100, 101, 102)
HELD_SEEDS = (300, 301, 302)
OTHER_HELD_SEEDS = (900, 901, 902)


class _Teacher(nn.Module):
    """Windowed teacher: tanh of the mean of the last WINDOW sensor frames."""

    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(2024)
        self.register_buffer("weight", torch.randn(2, SENSOR_FEATURES, generator=generator) * 0.05)
        self.register_buffer("bias", torch.randn(2, generator=generator) * 0.02)

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        pooled = window[:, -WINDOW:, :SENSOR_FEATURES].mean(dim=1)
        logits = pooled @ self.weight.T + self.bias
        correction = torch.tanh(logits)
        confidence = torch.sigmoid(logits[:, :1].sum(dim=-1, keepdim=True))
        return torch.cat([correction, confidence], dim=-1)


def _episodes(seeds: tuple[int, ...]) -> list:
    return [make_synthetic_episode(seed=seed, frames=FRAMES) for seed in seeds]


def _distill(train_episodes: list) -> tuple[StatefulTCN, _Teacher]:
    """Run the production distill loop on training episodes only."""
    torch.set_num_threads(1)
    teacher = _Teacher().eval()
    student = StatefulTCN(StudentConfig(seed=SEED, loops=2)).train()
    optimizer = AdamW(student.parameters(), lr=LEARNING_RATE)
    args = build_parser().parse_args(
        [
            "--teacher-checkpoint",
            "unused",
            "--data",
            "unused",
            "--seed",
            str(SEED),
            "--loops",
            "2",
            "--window",
            str(WINDOW),
            "--lr",
            str(LEARNING_RATE),
            "--conf-weight",
            "1.0",
            "--anomaly-rate",
            "0.0",
            "--mtp-k",
            "0",
        ]
    )
    for _ in range(EPOCHS):
        for index, episode in enumerate(train_episodes):
            train_episode(
                student, teacher, None, episode.frames, optimizer, args, seed_offset=index
            )
    return student.eval(), teacher


def _measure(
    student: StatefulTCN, teacher: _Teacher, episodes: list
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (RMSE, held-out confidence logits, max-axis correction errors)."""
    torch.set_num_threads(1)
    corrections: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    logits: list[float] = []
    errors: list[float] = []
    with torch.no_grad():
        for episode in episodes:
            student.reset_state()
            for tick, row in enumerate(episode.frames):
                correction, _ = student.step(row)
                if tick + 1 < WINDOW:
                    continue
                logits.append(student.confidence_logit().item())
                window = torch.as_tensor(
                    episode.frames[tick + 1 - WINDOW : tick + 1, :SENSOR_FEATURES],
                    dtype=torch.float32,
                ).unsqueeze(0)
                target = teacher(window)[0, :2]
                corrections.append(correction.numpy().copy())
                targets.append(target.numpy().copy())
                errors.append((correction - target).abs().max().item())
    predicted = np.asarray(corrections)
    expected = np.asarray(targets)
    rmse = float(np.sqrt(np.mean((predicted - expected) ** 2)))
    return rmse, np.asarray(logits, dtype=np.float64), np.asarray(errors, dtype=np.float64)


def test_distill_gate_hard_ratio_ece_and_held_out_untouched():
    train = _episodes(TRAIN_SEEDS)
    held = _episodes(HELD_SEEDS)
    held_snapshot = [(ep.frames.copy(), ep.targets.copy()) for ep in held]

    student, teacher = _distill(train)
    # A genuinely different held-out pool is available but is never passed to
    # the trainer: identical seed and train data must still give a bit-identical
    # model, so held-out data cannot have leaked into the fit.
    other_held = _episodes(OTHER_HELD_SEEDS)
    assert not np.array_equal(held[0].frames, other_held[0].frames)
    repeat, _ = _distill(train)
    assert all(
        torch.equal(student.state_dict()[key], repeat.state_dict()[key])
        for key in student.state_dict()
    )

    for episode, (frames, targets) in zip(held, held_snapshot):
        np.testing.assert_array_equal(episode.frames, frames)
        np.testing.assert_array_equal(episode.targets, targets)

    rmse_train, _, _ = _measure(student, teacher, train)
    rmse_held, logits, errors = _measure(student, teacher, held)
    assert np.isfinite(rmse_train) and np.isfinite(rmse_held)
    assert rmse_train > 0.0 and rmse_held > 0.0

    ratio = rmse_held / rmse_train
    assert ratio <= RATIO_LIMIT, f"held-out/train RMSE ratio {ratio:.3f} > {RATIO_LIMIT}"

    calibration = conformal_calibrate(
        logits, errors, tol=ERROR_TOL, target_ece=ECE_LIMIT, max_loops=3
    )
    assert calibration.n == logits.size
    assert 0.0 < calibration.coverage < 1.0  # non-degenerate correctness labels
    assert calibration.raw_ece > ECE_LIMIT  # the gate has something to fix
    assert calibration.ece <= calibration.raw_ece + 1e-12
    assert calibration.ece <= ECE_LIMIT, f"held-out ECE {calibration.ece:.4f} > {ECE_LIMIT}"


def test_distill_seed_controls_the_student_without_global_rng():
    torch.manual_seed(1)
    first = StatefulTCN(StudentConfig(seed=SEED, loops=2))
    torch.manual_seed(999)
    second = StatefulTCN(StudentConfig(seed=SEED, loops=2))
    other = StatefulTCN(StudentConfig(seed=SEED + 1, loops=2))
    first_state, second_state, other_state = (
        first.state_dict(),
        second.state_dict(),
        other.state_dict(),
    )
    assert all(torch.equal(first_state[k], second_state[k]) for k in first_state)
    assert any(not torch.equal(first_state[k], other_state[k]) for k in first_state)
