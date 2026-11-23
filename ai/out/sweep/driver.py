"""Honest epoch/seed sweep driver (run 4 data, unchanged pipeline).

Runs the exact `potok_ai.engine` CLI once per (model, epoch, seed) with the
same flags as run 4, varying only --epochs, --seed and --out.  Residual and
curriculum are passed explicitly (they are also the defaults) and --calibrate
is on for every run.  Each run gets a private output directory so the lite
teacher checkpoint cannot leak between runs.

Results are appended to results.csv after every completed job so a crash or a
timeout never loses the work already done.  Existing complete runs are reused.

    .venv/bin/python out/sweep/driver.py --workers 8
    .venv/bin/python out/sweep/driver.py --workers 8 --epochs 40 60 100 --seeds 1 2 3
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

SWEEP = Path(__file__).resolve().parent
AI = SWEEP.parents[1]
RUNS = SWEEP / "runs"
DATA = AI / "out" / "run4" / "data"
PYTHON = AI / ".venv" / "bin" / "python"
ENGINE = "potok_ai.engine"

MODELS = ("lrs", "loopwm-lite", "loopwm")

# Shared flags are byte-for-byte the run 4 command line (ai/BUILD_LOG.md:572-586).
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
    "model", "epoch", "seed", "rmse_train", "rmse_held_out", "ratio",
    "ece", "raw_ece", "coverage", "n", "method", "config_hash",
    "elapsed_s", "status",
)


def run_dir(model: str, epoch: int, seed: int) -> Path:
    return RUNS / model / f"e{epoch}_s{seed}"


def complete(path: Path, model: str, seed: int) -> bool:
    m, c = path / "metrics.json", path / "calibration.json"
    if not (m.exists() and c.exists()):
        return False
    try:
        metrics = json.loads(m.read_text())
        calib = json.loads(c.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return metrics.get("model") == model and metrics.get("seed") == seed and calib.get("n", 0) > 0


def command(model: str, epoch: int, seed: int, out: Path) -> list[str]:
    return [
        str(PYTHON), "-m", ENGINE,
        "--model", model, "--data", str(DATA), "--out", str(out),
        "--epochs", str(epoch), "--seed", str(seed), "--device", "cpu",
        *SHARED, *PER_MODEL[model],
    ]


def run_one(model: str, epoch: int, seed: int) -> dict:
    out = run_dir(model, epoch, seed)
    out.mkdir(parents=True, exist_ok=True)
    if complete(out, model, seed):
        row = collect(model, epoch, seed, "reused", 0.0)
        print(f"REUSE {model:12s} e{epoch:<4d} s{seed}", flush=True)
        return row
    started = time.perf_counter()
    log = out / "run.log"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(AI / "src")
    with log.open("w") as handle:
        proc = subprocess.run(command(model, epoch, seed, out), cwd=AI, env=env,
                              stdout=handle, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - started
    if proc.returncode != 0:
        print(f"FAIL  {model:12s} e{epoch:<4d} s{seed} rc={proc.returncode} log={log}", flush=True)
        return {"model": model, "epoch": epoch, "seed": seed, "status": f"failed rc={proc.returncode}",
                "elapsed_s": round(elapsed, 1)}
    row = collect(model, epoch, seed, "ok", elapsed)
    print(f"DONE  {model:12s} e{epoch:<4d} s{seed} {elapsed:7.1f}s "
          f"held={row['rmse_held_out']:.6f}", flush=True)
    return row


def collect(model: str, epoch: int, seed: int, status: str, elapsed: float) -> dict:
    out = run_dir(model, epoch, seed)
    metrics = json.loads((out / "metrics.json").read_text())
    calib = json.loads((out / "calibration.json").read_text())
    return {
        "model": model,
        "epoch": epoch,
        "seed": seed,
        "rmse_train": metrics["rmse_train"],
        "rmse_held_out": metrics["rmse_held_out"],
        "ratio": metrics["ratio"],
        "ece": metrics["ece"],
        "raw_ece": calib["raw_ece"],
        "coverage": calib["coverage"],
        "n": calib["n"],
        "method": calib["method"],
        "config_hash": metrics["config_hash"],
        "elapsed_s": round(elapsed, 1),
        "status": status,
    }


def write_csv(rows: dict[tuple[str, int, int], dict]) -> None:
    path = SWEEP / "results.csv"
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for key in sorted(rows):
            writer.writerow({field: rows[key].get(field, "") for field in FIELDS})
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, nargs="+", default=[40, 60, 100])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    args = parser.parse_args()

    jobs = [(model, epoch, seed)
            for model in args.models for epoch in args.epochs for seed in args.seeds]
    rows: dict[tuple[str, int, int], dict] = {}
    path = SWEEP / "results.csv"
    if path.exists():
        with path.open(newline="") as handle:
            for record in csv.DictReader(handle):
                key = (record["model"], int(record["epoch"]), int(record["seed"]))
                if record.get("status") in ("ok", "reused"):
                    rows[key] = record
    write_csv(rows)

    todo = [job for job in jobs if job not in rows]
    print(f"jobs={len(jobs)} already_done={len(jobs) - len(todo)} todo={len(todo)} "
          f"workers={args.workers}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, *job): job for job in todo}
        for future in as_completed(futures):
            job = futures[future]
            try:
                row = future.result()
            except Exception as exc:  # keep the sweep alive; log the failure
                row = {"model": job[0], "epoch": job[1], "seed": job[2],
                       "status": f"exception {type(exc).__name__}: {exc}"}
                print(f"ERROR {job} {exc}", flush=True)
            rows[job] = row
            write_csv(rows)

    ok = sum(1 for row in rows.values() if row.get("status") in ("ok", "reused"))
    print(f"DONE {ok}/{len(jobs)} successful; results={SWEEP / 'results.csv'}", flush=True)


if __name__ == "__main__":
    main()
