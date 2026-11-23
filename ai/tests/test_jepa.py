import torch

from potok_ai.jepa import EMATeacher, JEPAPredictor, jepa_loss, koleo_loss, mask_windows
from potok_ai.model import ModelConfig, PotokModel


def small_model() -> PotokModel:
    return PotokModel(ModelConfig(width=16, rank=8, in_features=8))


def test_teacher_ema_update():
    torch.manual_seed(0)
    student = small_model()
    teacher = EMATeacher(student, momentum=0.5)
    student_param = next(student.parameters())
    teacher_param = next(teacher.net.parameters())
    before = teacher_param.detach().clone()
    with torch.no_grad():
        student_param.add_(1.0)
    teacher.update(student)
    assert torch.allclose(teacher_param.detach(), before + 0.5, atol=1e-6)


def test_mask_windows_selects_one_span_per_sample():
    mask = mask_windows(torch.zeros(4, 16, 8), mask_ratio=0.25)
    assert mask.shape == (4, 16)
    assert (mask.sum(dim=1) == 4).all()


def test_jepa_loss_and_gradient():
    torch.manual_seed(1)
    student = small_model()
    teacher = EMATeacher(student, momentum=0.9)
    predictor = JEPAPredictor(16)
    window = torch.randn(3, 16, 8)
    losses = jepa_loss(student, teacher, window, predictor)
    assert torch.isfinite(losses.total).all()
    assert losses.jepa.item() >= 0.0
    losses.total.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.parameters())


def test_koleo_is_finite():
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    loss = koleo_loss(features)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert koleo_loss(features[:1]) == 0.0
