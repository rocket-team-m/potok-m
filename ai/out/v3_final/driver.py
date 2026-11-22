"""100-epoch v3 training driver: exact sweep recipe, new g_accel data.

Identical flags to out/sweep/driver.py, only DATA/RUNS change. Residual and
curriculum are explicit (also defaults); --calibrate is always on. Each
(model, seed) gets a private --out so no teacher checkpoint leaks.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
AI = HERE.parents[1]
DATA = AI / "out" / "v3_final" / "data"
RUNS = HERE / "runs"
PYTHON = AI / ".venv" / "bin" / "python"

MODELS = ("lrs", "loopwm-lite", "loopwm")
SHARED = (
    "--stage", "train",
    "--window", "50", "--batch", "8",
    "--width", "64", "--rank", "32", "--heads", "4", "--max-iters", "2",
    "--lite-width", "64", "--attn-rank", "8", "--lite-iters", "2",
    "--latent-tokens", "16",
    "--residual", "--curriculum", "--calibrate",
)
PER_MODEL = {
    "lrs": ("--limit", "2", "--eval-limit", "1451", "--lr", "0.03"),
    "loopwm-lite": ("--limit", "1451", "--eval-limit", "1451"),
    "loopwm": ("--limit", "1451", "--eval-limit", "1451"),
}
FIELDS = (
    "model", "seed", "rmse_train", "rmse_held_out", "ratio", "ece",
    "raw_ece", "coverage", "n", "method", "config_hash", "elapsed_s", "status",
)


def run_dir(model: str, seed: int) -> Path:
    return RUNS / model / f"s{seed}"


def complete(path: Path, model: str, seed: int) -> bool:
    m, c = path / "metrics.json", path / "calibration.json"
    if not (m.exists() and c.exists()):
        return False
    try:
        metrics, calib = json.loads(m.read_text()), json.loads(c.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return metrics.get("model") == model and metrics.get("seed") == seed and calib.get("n", 0) > 0


def command(model: str, seed: int, out: Path) -> list[str]:
    return [
        str(PYTHON), "-m", "potok_ai.engine",
        "--model", model, "--data", str(DATA), "--out", str(out),
        "--epochs", "100", "--seed", str(seed), "--device", "cpu",
        *SHARED, *PER_MODEL[model],
    ]


def collect(model: str, seed: int, status: str, elapsed: float) -> dict:
    out = run_dir(model, seed)
    metrics = json.loads((out / "metrics.json").read_text())
    calib = json.loads((out / "calibration.json").read_text())
    return {
        "model": model, "seed": seed,
        "rmse_train": metrics["rmse_train"], "rmse_held_out": metrics["rmse_held_out"],
        "ratio": metrics["ratio"], "ece": metrics["ece"],
        "raw_ece": calib["raw_ece"], "coverage": calib["coverage"], "n": calib["n"],
        "method": calib["method"], "config_hash": metrics["config_hash"],
        "elapsed_s": round(elapsed, 1), "status": status,
    }


def run_one(model: str, seed: int) -> dict | None:
    out = run_dir(model, seed)
    out.mkdir(parents=True, exist_ok=True)
    if complete(out, model, seed):
        print(f"REUSE {model:12s} s{seed}", flush=True)
        return collect(model, seed, "reused", 0.0)
    started = time.perf_counter()
    log = out / "run.log"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(AI / "src")
    with log.open("w") as handle:
        proc = subprocess.run(command(model, seed, out), cwd=AI, env=env,
                              stdout=handle, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - started
    if proc.returncode != 0:
        print(f"FAIL  {model:12s} s{seed} rc={proc.returncode} log={log}", flush=True)
        return {"model": model, "seed": seed, "status": f"failed rc={proc.returncode}",
                "elapsed_s": round(elapsed, 1)}
    row = collect(model, seed, "ok", elapsed)
    print(f"DONE  {model:12s} s{seed} {elapsed:7.1f}s "
          f"held={row['rmse_held_out']:.6f} ratio={row['ratio']:.4f} ece={row['ece']:.4f}",
          flush=True)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    results = HERE / "results.csv"
    jobs = [(model, seed) for seed in args.seeds for model in MODELS]
    with results.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if handle.tell() == 0:
            writer.writeheader()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_one, model, seed) for model, seed in jobs]
            for future in as_completed(futures):
                row = future.result()
                if row is not None:
                    writer.writerow(row)
                    handle.flush()


if __name__ == "__main__":
    main()
