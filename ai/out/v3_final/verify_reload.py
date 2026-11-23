"""Reload verification for the v3 exports (run after training finishes).

For each family: reload the export with the engine's own `eval` stage in a
separate output dir, compare the recomputed metrics to the training report,
and check the calibration JSON round-trip through apply_calibration.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
AI = HERE.parents[1]
TMP = Path("/tmp/opencode/v3-reload")
EXPORT_NAMES = {
    "lrs": ("lrs.bin",),
    "loopwm-lite": ("lite.pt",),
    "loopwm": ("loopwm.pt", "loopwm_student.bin"),
}
FIELDS = ("rmse_train", "rmse_held_out", "ratio", "mae", "max_error", "ece", "params", "mac", "bytes")


def main() -> None:
    sys.path.insert(0, str(AI / "src"))
    from potok_ai.calibration import apply_calibration

    ok = True
    TMP.mkdir(parents=True, exist_ok=True)
    for model, names in EXPORT_NAMES.items():
        src = HERE / "runs" / model / "s1"
        metrics = json.loads((src / "metrics.json").read_text())
        calib = json.loads((src / "calibration.json").read_text())
        missing = [name for name in names if not (src / name).exists()]
        if missing:
            print(f"FAIL {model}: missing exports {missing}")
            ok = False
            continue

        dst = TMP / model
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        for name in names:
            shutil.copy2(src / name, dst / name)

        env = dict(__import__("os").environ)
        env["PYTHONPATH"] = str(AI / "src")
        proc = subprocess.run(
            [str(AI / ".venv" / "bin" / "python"), "-m", "potok_ai.engine",
             "--model", model, "--stage", "eval",
             "--data", str(HERE / "data"), "--out", str(dst),
             "--window", "50", "--batch", "8",
             "--width", "64", "--rank", "32", "--heads", "4", "--max-iters", "2",
             "--lite-width", "64", "--attn-rank", "8", "--lite-iters", "2",
             "--latent-tokens", "16", "--residual", "--curriculum", "--calibrate",
             *(("--limit", "2", "--eval-limit", "1451", "--lr", "0.03") if model == "lrs"
               else ("--limit", "1451", "--eval-limit", "1451")),
             "--seed", "1", "--device", "cpu"],
            cwd=AI, env=env, capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print(f"FAIL {model}: eval reload rc={proc.returncode}\n{proc.stdout}\n{proc.stderr}")
            ok = False
            continue
        reloaded = json.loads((dst / "metrics.json").read_text())
        drift = {k: abs(reloaded[k] - metrics[k]) for k in FIELDS}
        worst = max(drift.values())
        # The v2 student export stores effective ternary codes, so a reloaded
        # LRS forward can differ slightly from the in-memory trained model.
        # Torch checkpoints (lite / loopwm) reload bit-exactly.
        tolerance = 0.02 if model == "lrs" else 1e-9
        match = worst <= tolerance
        gates = (reloaded["ratio"] <= 1.25 and reloaded["ece"] <= 0.05)
        print(f"{'PASS' if match and gates else 'FAIL'} {model}: reload drift max={worst:.3g} "
              f"(tol={tolerance:g}, gates after reload={'PASS' if gates else 'FAIL'}) "
              f"rmse_train {metrics['rmse_train']:.6f} -> {reloaded['rmse_train']:.6f}, "
              f"ratio {metrics['ratio']:.6f} -> {reloaded['ratio']:.6f}")
        ok &= match and gates

        # calibration.json must be self-consistent when applied to its own scale.
        z = np.linspace(-4.0, 4.0, 9)
        p = apply_calibration(calib, z)
        valid = np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1))
        print(f"{'PASS' if valid else 'FAIL'} {model}: calibration {calib['method']} "
              f"n={calib['n']} apply range=[{p.min():.4f},{p.max():.4f}]")
        ok &= bool(valid)

        # The confidence head logit must survive the binary export path.
        if model == "lrs":
            from potok_ai.export import load_student_export
            student = load_student_export(src / "lrs.bin").to_model()
            logit = student.confidence_logit()
            print(f"{'PASS' if np.isfinite(logit) else 'FAIL'} lrs: exported logit={logit:.6f}")
            ok &= bool(np.isfinite(logit))

    if not ok:
        raise SystemExit(1)
    print("ALL RELOAD CHECKS PASSED")


if __name__ == "__main__":
    main()
