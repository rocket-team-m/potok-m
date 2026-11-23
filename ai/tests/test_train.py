import math

from potok_ai.train import build_parser, run


def _args(stage: str, out):
    return build_parser().parse_args(
        [
            "--stage", stage,
            "--epochs", "1",
            "--seed", "0",
            "--device", "cpu",
            "--out", str(out),
            "--synthetic",
            "--batch", "2",
            "--seq-len", "16",
            "--limit", "4",
            "--width", "16",
            "--rank", "8",
            "--lr", "0.01",
        ]
    )


def test_tiny_cpu_pretrain(tmp_path):
    loss = run(_args("pretrain", tmp_path / "pretrain.pt"))
    assert math.isfinite(loss)
    assert (tmp_path / "pretrain.pt").exists()


def test_tiny_cpu_distill(tmp_path):
    loss = run(_args("distill", tmp_path / "distill.pt"))
    assert math.isfinite(loss)
    assert (tmp_path / "distill.pt").exists()
