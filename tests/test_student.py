import numpy as np
import pytest
import torch

from potok_ai.data import make_synthetic_episode, write_episode
from potok_ai.distill_student import build_parser, phase_of, run, train_episode
from potok_ai.export import (
    MODEL_KIND_STUDENT,
    export_student,
    load_student_export,
)
from potok_ai.model import ModelConfig, PotokModel
from potok_ai.student import (
    FLAGS_INDEX,
    FRAME_FEATURES,
    PHASE_INDEX,
    SENSOR_FEATURES,
    StatefulTCN,
    StudentConfig,
)


def frames(seed: int, count: int, dim: int = FRAME_FEATURES) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(count, dim, generator=generator)
    values[:, :SENSOR_FEATURES] *= 4.0
    if dim > FLAGS_INDEX:
        values[:, FLAGS_INDEX] = 1.0
    return values


def test_reset_state_restores_fresh_behaviour():
    model = StatefulTCN(StudentConfig(seed=1, width=16, rank=8)).eval()
    frame = frames(0, 1)[0]
    with torch.no_grad():
        expected = model.step(frame)
        model.step(frames(2, 1)[0])
        assert model.h.abs().sum() > 0
        model.reset_state()
        assert torch.equal(model.h, torch.zeros_like(model.h))
        actual = model.step(frame)
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


def test_determinism_ignores_global_rng_state():
    torch.manual_seed(1)
    first = StatefulTCN(StudentConfig(seed=7, width=16, rank=8)).eval()
    torch.manual_seed(999)
    second = StatefulTCN(StudentConfig(seed=7, width=16, rank=8)).eval()
    with torch.no_grad():
        for frame in frames(3, 20):
            a_corr, a_conf = first.step(frame)
            b_corr, b_conf = second.step(frame)
            assert torch.equal(a_corr, b_corr)
            assert torch.equal(a_conf, b_conf)


def test_bounded_state_drift_over_1500_steps():
    model = StatefulTCN(StudentConfig(seed=4, width=32, rank=8)).eval()
    with torch.no_grad():
        for frame in frames(5, 1500):
            corrections, confidence = model.step(frame)
            assert torch.isfinite(corrections).all()
            assert torch.isfinite(confidence).all()
            assert corrections.abs().max() <= 1.0
            assert confidence.min() >= 0.0 and confidence.max() <= 1.0
            assert model.h.abs().max() <= 1.0 + 1e-6
        assert torch.equal(model.h, model.h)  # no NaN crept in


def test_export_load_roundtrip(tmp_path):
    model = StatefulTCN(StudentConfig(seed=5, width=32, rank=8)).eval()
    window = frames(6, 12)
    probe = frames(7, 1)[0]
    with torch.no_grad():
        model.warmup(window)
        expected = model.step(probe)

    path = export_student(model, tmp_path / "student.bin")
    exported = load_student_export(path)
    assert exported.model_kind == MODEL_KIND_STUDENT
    assert exported.version == 2
    assert exported.n_blocks == 2 and exported.loops == 3
    assert exported.width == 32 and exported.rank == 8

    reloaded = exported.to_model()
    with torch.no_grad():
        reloaded.warmup(window)
        actual = reloaded.step(probe)
    assert torch.allclose(actual[0], expected[0], atol=1e-6)
    assert torch.allclose(actual[1], expected[1], atol=1e-6)


def test_student_export_rejects_tampering_and_wrong_kind(tmp_path):
    from potok_ai.export import MAGIC, export_model

    model = StatefulTCN(StudentConfig(seed=8, width=16, rank=8))
    path = export_student(model, tmp_path / "s.bin")
    raw = bytearray(path.read_bytes())
    raw[-5] ^= 0xFF
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="checksum"):
        load_student_export(path)

    teacher_path = export_model(PotokModel(ModelConfig(width=16, rank=8)), tmp_path / "t.bin")
    assert teacher_path.read_bytes()[: len(MAGIC)] == MAGIC
    with pytest.raises(ValueError, match="model_kind"):
        load_student_export(teacher_path)


def test_tiny_cpu_distillation_step(tmp_path):
    torch.manual_seed(1234)
    teacher = PotokModel(ModelConfig(width=16, rank=8))
    checkpoint = tmp_path / "teacher.pt"
    torch.save(
        {"model_config": teacher.cfg.__dict__, "state_dict": teacher.state_dict()},
        checkpoint,
    )

    episode = make_synthetic_episode(seed=2, frames=60)
    data = tmp_path / "episode.bin"
    write_episode(data, episode)

    args = build_parser().parse_args(
        [
            "--teacher-checkpoint",
            str(checkpoint),
            "--data",
            str(data),
            "--out",
            str(tmp_path / "student.pt"),
            "--epochs",
            "1",
            "--seed",
            "1",
            "--loops",
            "2",
            "--limit",
            "3",
        ]
    )
    loss = run(args)
    out = tmp_path / "student.pt"
    assert out.exists() and out.with_suffix(".bin").exists()
    assert np.isfinite(loss)
    stored = torch.load(out, map_location="cpu", weights_only=True)
    assert stored["config"]["seed"] == 1
    assert stored["state_dict"]["alloc_logits"].shape == (3,)


def test_phase_change_resets_state(tmp_path):
    torch.manual_seed(1234)
    teacher = PotokModel(ModelConfig(width=16, rank=8))
    checkpoint = tmp_path / "teacher.pt"
    torch.save(
        {"model_config": teacher.cfg.__dict__, "state_dict": teacher.state_dict()},
        checkpoint,
    )
    episode = make_synthetic_episode(seed=3, frames=60)
    frames_with_phase = np.concatenate(
        [episode.frames, np.zeros((60, 2), dtype=np.float32)], axis=1
    )
    frames_with_phase[:30, PHASE_INDEX] = 0.0
    frames_with_phase[30:, PHASE_INDEX] = 1.0
    frames_with_phase[:, FLAGS_INDEX] = 1.0
    assert phase_of(frames_with_phase[0]) == 0
    assert phase_of(frames_with_phase[40]) == 1

    args = build_parser().parse_args(
        ["--teacher-checkpoint", str(checkpoint), "--data", str(tmp_path / "unused.bin")]
    )
    student = StatefulTCN(StudentConfig(seed=1, width=16, rank=8))
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    _, resets, _ = train_episode(student, teacher, None, frames_with_phase, optimizer, args)
    assert resets == 2  # episode start plus the single phase change
