"""Pre-sigmoid confidence logits are available in models and binary exports."""

import pytest
import torch

from potok_ai.data import make_smart_episode
from potok_ai.export import export_student, load_student_export
from potok_ai.loopwm import LoopWM, LoopWMConfig
from potok_ai.loopwm_lite import LiteConfig, LiteModel
from potok_ai.student import StatefulTCN, smart_student_config


def test_student_step_exposes_its_logit():
    model = StatefulTCN(smart_student_config())
    frame = make_smart_episode(1).frames[0]
    _, confidence = model.step(frame)
    assert torch.allclose(torch.sigmoid(model.confidence_logit()), confidence)
    with pytest.raises(ValueError, match="call step"):
        StatefulTCN(smart_student_config()).confidence_logit()


def test_export_carries_logit_weights_without_format_change(tmp_path):
    model = StatefulTCN(smart_student_config())
    path = export_student(model, tmp_path / "student.bin")
    exported = load_student_export(path)
    frame = make_smart_episode(1).frames[0]
    model.step(frame)
    z = model.last_z.detach().numpy()

    assert exported.confidence_logit(z) == pytest.approx(
        float(model.confidence_logit().detach()), abs=1e-6
    )
    restored = exported.to_model()
    _, confidence = restored.step(frame)
    assert torch.allclose(torch.sigmoid(restored.confidence_logit()), confidence)


def test_window_heads_split_logits_from_output():
    torch.manual_seed(0)
    window, cond = torch.randn(2, 12, 14), torch.randn(2, 6)

    teacher = LoopWM(LoopWMConfig(width=16, heads=4))
    latent, _ = teacher.refine(window, cond)
    raw = teacher.head_logits(latent)
    output = teacher.head_from_latent(latent)
    assert raw.shape == (2, 3)
    assert torch.allclose(output[:, 2], torch.sigmoid(raw[:, 2]))
    assert torch.allclose(output[:, :2], torch.tanh(raw[:, :2]) * teacher.cfg.g_max)

    lite = LiteModel(LiteConfig(window=12, width=16, attn_rank=4, heads=2, target_width=16))
    latent, _ = lite.refine(window, cond)
    pooled = latent.mean(dim=1)
    raw = lite.head_logits(pooled)
    output = lite.head_from_latent(pooled)
    assert raw.shape == (2, 3)
    assert torch.allclose(output[:, 2], torch.sigmoid(raw[:, 2]))
