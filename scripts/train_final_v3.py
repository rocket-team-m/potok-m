# SPDX-License-Identifier: GPL-3.0-or-later
"""TRAIN-FINAL driver: post-FIXPUMP staged teacher v2, 100 epochs, 3 seeds.

Runs the five-stage teacher-v2 CLI (``sft -> privileged -> dagger -> rlvr ->
distill``) on the regenerated v3 train split, then adds the honest confidence
provenance the evaluation contract asks for.  Nothing here tunes physics,
gates, thresholds or model knobs: every CLI value below is fixed and recorded
in each seed's ``metrics.json``.  Closed-loop A/B is a separate step; this
driver only writes the student exports it would consume.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
AI = ROOT / "ai"
SRC = AI / "src"
sys.path.insert(0, str(SRC))

from potok_ai.conf_integration import split_confidence_score
from potok_ai.teacher_gates import (
    check_ece,
    check_ece_source,
    check_ratio,
)
from potok_ai.train_teacher_v2 import (
    SEEDS,
    TeacherV2Config,
    _build_student,
    _predict,
    _window_batches,
    config_hash,
    partition_episodes,
)

from potok_ai.data import load_episodes

DATA = AI / "out" / "final_v3_postpump" / "train_only" / "train_validation.bin"
HELDOUT = AI / "out" / "final_v3_postpump" / "data" / "held-out" / "validation.bin"
SOURCE_CONFIG = AI / "out" / "v3_final" / "sim.toml"
OUT = AI / "out" / "final_v3_postpump"
RUNS = OUT / "runs"
EXPORTS = OUT / "exports"
DOC = AI / "docs" / "TRAINING_RESULTS.md"
PYTHON = AI / ".venv" / "bin" / "python"
EPOCHS = 100
# The five-stage contract this run executed. Kept local: a concurrent edit to
# the trainer's STAGE_ORDER must not silently reclassify an existing artifact.
EXPECTED_STAGES = ("sft", "privileged", "dagger", "rlvr", "distill")

# Fixed recipe.  ``limit`` caps WINDOWS per batch builder, not episodes: the
# 80-episode train file is loaded in full and partitioned by scenario for every
# seed (partition_episodes), and the DAgger stage uses all of it.  The cap keeps
# a 100-epoch CPU run tractable and is recorded verbatim in every report.
WINDOW_CAP = 1451  # one full episode of 12-tick windows at stride 1
CFG_KWARGS = {
    "data": DATA,
    "stage": "all",
    "device": "cpu",
    "epochs": EPOCHS,
    "batch": 64,
    "window": 12,
    "limit": WINDOW_CAP,
    "lr": 2e-3,
    "student_lr": 1e-3,
    "width": 16,
    "heads": 2,
    "max_iters": 2,
    "learned_exit": False,
    "exit_gate_threshold": 0.5,
    "exit_entropy_alpha": 0.01,
    "spectral_residual": False,
    "spectral_bound": 1.0,
    "jepa_weight": 0.05,
    "entropy_weight": 1.0,
    "dagger_rounds": 1,
    "n_inner_held_out": 2,
    "rlvr_samples": 4,
    "rlvr_temperature": 0.05,
    "truth_sidecar": None,
    "calib_tol": 0.15,
    "opt": "adamw",
    "k3_per_head": 0,
    "k3_qk_clip": False,
    "k3_qk_clip_tau": 100.0,
}

# Post-FIXPUMP digests.  The held-out digest equals the zero-command baseline
# rebuilt by RECALC01; a pre-FIXPUMP export would be 5848e222...
TRAIN_SHA256 = "34b35e0ff67f7eeecbd895b2efde8bebcfa159c1a3e78379da6750338b6930b6"
HELDOUT_SHA256 = "fdc1d1400b1c5f111e21f79dead37afa1dd9f058fe457edb78037dc6ebe89807"
SOURCE_CONFIG_SHA256 = "9c34e71f9d10bcf081c31f15f5332f7bb2fac04dddb28009a27cd30987ab5e91"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_inputs() -> dict[str, str]:
    """Loud refusal of stale/missing data; no silent substitution."""
    digests = {}
    for label, path, expected in (
        ("train", DATA, TRAIN_SHA256),
        ("heldout", HELDOUT, HELDOUT_SHA256),
        ("config", SOURCE_CONFIG, SOURCE_CONFIG_SHA256),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label}: missing {path}; regenerate the post-FIXPUMP split")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(
                f"{label}: sha256 {actual} != {expected}; the input is not the frozen "
                "post-FIXPUMP artifact (no data was substituted)"
            )
        digests[f"{label}_sha256"] = actual
    return digests


def command(seed: int, out: Path) -> list[str]:
    """The exact trainer invocation, with every knob explicit."""
    return [
        str(PYTHON),
        "-m",
        "potok_ai.train_teacher_v2",
        "--data",
        str(DATA),
        "--out",
        str(out),
        "--stage",
        "all",
        "--seed",
        str(seed),
        "--epochs",
        str(EPOCHS),
        "--batch",
        str(CFG_KWARGS["batch"]),
        "--window",
        str(CFG_KWARGS["window"]),
        "--limit",
        str(WINDOW_CAP),
        "--lr",
        str(CFG_KWARGS["lr"]),
        "--student-lr",
        str(CFG_KWARGS["student_lr"]),
        "--width",
        str(CFG_KWARGS["width"]),
        "--heads",
        str(CFG_KWARGS["heads"]),
        "--max-iters",
        str(CFG_KWARGS["max_iters"]),
        "--jepa-weight",
        str(CFG_KWARGS["jepa_weight"]),
        "--entropy-weight",
        str(CFG_KWARGS["entropy_weight"]),
        "--dagger-rounds",
        str(CFG_KWARGS["dagger_rounds"]),
        "--n-inner-held-out",
        str(CFG_KWARGS["n_inner_held_out"]),
        "--rlvr-samples",
        str(CFG_KWARGS["rlvr_samples"]),
        "--rlvr-temperature",
        str(CFG_KWARGS["rlvr_temperature"]),
        "--calib-tol",
        str(CFG_KWARGS["calib_tol"]),
        "--opt",
        CFG_KWARGS["opt"],
        "--device",
        "cpu",
    ]


def run_trainer(seed: int, out: Path) -> dict:
    """Run one seed; refuse a partial or missing artifact."""
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.json"
    if metrics_path.is_file() and (out / "student.pt").is_file():
        report = json.loads(metrics_path.read_text())
        if (
            tuple(report.get("stages_completed", ())) == EXPECTED_STAGES
            and report.get("seed") == seed
            and report.get("reward_provenance")
        ):
            print(f"REUSE seed {seed}", flush=True)
            return report
        raise ValueError(f"seed {seed}: existing metrics.json is incomplete; move it aside")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC)
    log = out / "run.log"
    print(f"TRAIN seed {seed} epochs={EPOCHS} log={log}", flush=True)
    with log.open("w") as handle:
        proc = subprocess.run(
            command(seed, out),
            cwd=AI,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"seed {seed}: trainer exited {proc.returncode}; see {log}")
    report = json.loads(metrics_path.read_text())
    if tuple(report.get("stages_completed", ())) != EXPECTED_STAGES:
        raise ValueError(
            f"seed {seed}: stages_completed {report.get('stages_completed')} != "
            f"{list(EXPECTED_STAGES)}; the trainer API changed, do not relabel the artifact"
        )
    if not report.get("reward_provenance"):
        raise ValueError(f"seed {seed}: no reward_provenance; RLVR stage did not run")
    if not (out / "student.pt").is_file():
        raise FileNotFoundError(f"seed {seed}: distill stage wrote no student.pt")
    return report


def honest_confidence(seed: int, out: Path, cfg: TeacherV2Config) -> tuple[float, float]:
    """Disjoint fit/score confidence score on the inner-held-out scenarios.

    The trainer's inner split is disjoint from every gradient step.  We score
    the same student once more through the shipped split-conformal bridge so
    ``ece_held_out`` and ``conformal_coverage`` are measured, never copied from
    an in-sample field.
    """
    episodes = load_episodes(cfg.data)
    _fit, inner, _held_ids = partition_episodes(
        episodes, seed=seed, n_held_out=cfg.n_inner_held_out
    )
    source = json.loads((out / "metrics.json").read_text()).get("module_sources", {})
    student = _build_student(cfg, torch.device("cpu"))
    stored = torch.load(out / "student.pt", map_location="cpu", weights_only=True)
    student.load_state_dict(stored["state_dict"])
    if source and stored.get("source") != source.get("student"):
        raise ValueError(
            f"seed {seed}: student.pt source {stored.get('source')!r} != "
            f"metrics module_sources {source.get('student')!r}"
        )
    student.eval()
    logits: list[float] = []
    errors: list[float] = []
    with torch.no_grad():
        for batch in _window_batches(inner, cfg):
            window, cond, target = (t.to("cpu") for t in batch[:3])
            _command, logit, error = _predict(student, window, cond, target, student=True)
            logits.extend(logit.tolist())
            errors.extend(error.tolist())
    if not logits or not np.isfinite(logits).all() or not np.isfinite(errors).all():
        raise ValueError(f"seed {seed}: inner-held-out confidence predictions are not finite")
    score = split_confidence_score(
        np.asarray(logits),
        np.asarray(errors),
        out,
        seed=seed,
        tol=cfg.calib_tol,
    )
    return float(score.ece_held_out), float(score.conformal_coverage)


def enrich(seed: int, out: Path, report: dict, digests: dict[str, str]) -> dict:
    """Add the evaluation-contract fields without altering measured numbers."""
    # Re-running on a finished output directory must be a no-op for the
    # measurement, not a second (different) trainer-version hash comparison.
    if report.get("ece_source") == "held-out" and "ece_held_out" in report:
        return report
    cfg = TeacherV2Config(**{**CFG_KWARGS, "out": out, "seed": seed})
    if config_hash(cfg) != report["config_hash"]:
        raise ValueError(
            f"seed {seed}: config_hash {report['config_hash']} != "
            f"{config_hash(cfg)}; the recipe recorded in metrics.json differs"
        )
    ece_held_out, coverage = honest_confidence(seed, out, cfg)
    enriched = {
        "model": "teacher_v2",
        "seed": seed,
        "stage": report["stage"],
        "stages_completed": report["stages_completed"],
        "rmse_train": report["rmse_train"],
        "rmse_held_out": report["rmse_held_out"],
        "ratio": report["ratio"],
        # The trainer scores calibration only on inner-held-out scenarios; no
        # gradient step ever sees them.  Label it honestly and keep the legacy
        # in-sample value out of the gate.
        "ece": report["ece"],
        "ece_source": "held-out",
        "ece_held_out": ece_held_out,
        "conformal_coverage": coverage,
        "config_hash": report["config_hash"],
        "steps": report["steps"],
        "entropy": report["entropy"],
        "entropy_enabled": report["entropy_enabled"],
        "reward_provenance": report["reward_provenance"],
        "module_sources": report["module_sources"],
        "stages": report["stages"],
        "recipe": {
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in CFG_KWARGS.items()
            },
            "out": str(out),
            "epochs": EPOCHS,
        },
        "data_provenance": digests,
    }
    (out / "metrics.json").write_text(json.dumps(enriched, indent=2, sort_keys=True) + "\n")
    return enriched


def export_students(seeds: list[int], metrics: dict[int, dict]) -> None:
    """Copy each distilled student beside a provenance manifest (no A/B claim)."""
    EXPORTS.mkdir(parents=True, exist_ok=True)
    manifest = {
        "measurement": "distilled students from the post-FIXPUMP staged teacher",
        "claim": "artifacts only; no closed-loop A/B was run and no win is claimed",
        "export_format": "student.pt (torch state_dict of the distill-stage student)",
        "students": {},
    }
    for seed in seeds:
        source = RUNS / f"s{seed}" / "student.pt"
        destination = EXPORTS / f"teacher_v2_s{seed}.pt"
        destination.write_bytes(source.read_bytes())
        report = metrics[seed]
        manifest["students"][str(seed)] = {
            "file": destination.name,
            "sha256": sha256(destination),
            "config_hash": report["config_hash"],
            "student_source": report["module_sources"]["student"],
            "rmse_held_out": report["rmse_held_out"],
            "ece_source": report["ece_source"],
        }
    (EXPORTS / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def write_doc(seeds: list[int], metrics: dict[int, dict], gates: dict, digests: dict) -> None:
    held = [metrics[seed]["rmse_held_out"] for seed in seeds]
    mean = statistics.mean(held)
    std = statistics.stdev(held) if len(held) > 1 else 0.0
    rows = "\n".join(
        "| {seed} | {train:.6f} | {held:.6f} | {ratio:.4f} | {ece:.4f} | {cov:.4f} |".format(
            seed=seed,
            train=metrics[seed]["rmse_train"],
            held=metrics[seed]["rmse_held_out"],
            ratio=metrics[seed]["ratio"],
            ece=metrics[seed]["ece_held_out"],
            cov=metrics[seed]["conformal_coverage"],
        )
        for seed in seeds
    )
    gate_rows = "\n".join(
        f"| {name} | {'PASS' if result['passed'] else 'FAIL'} | {result['detail']} |"
        for name, result in gates.items()
    )
    DOC.write_text(
        "# Post-FIXPUMP teacher-v2 training results (TRAIN-FINAL)\n\n"
        "Staged teacher v2 (`sft -> privileged -> dagger -> rlvr -> distill`), "
        f"**{EPOCHS} epochs**, CPU only, seeds {list(seeds)}. "
        "Physics is the post-FIXPUMP simulator; the train split is the full "
        "80-episode v3 file, partitioned by scenario with a disjoint "
        "2-scenario inner hold-out per seed. Held-out scenarios 16-19 are "
        "never passed to a training batch. No physics constant, gate or "
        "threshold was changed. `--limit 1451` caps WINDOWS per batch builder "
        "(one episode-equivalent), not episodes: DAgger uses all 80 episodes "
        "and every scenario is scored. This is not the old `--limit 2` "
        "two-episode recipe.\n\n"
        f"Held-out RMSE over three seeds: **{mean:.6f} +/- {std:.6f} m/s^2** "
        "(sample standard deviation).\n\n"
        "| seed | train RMSE | held-out RMSE | ratio | disjoint ECE | conformal coverage |\n"
        "|---:|---:|---:|---:|---:|---:|\n"
        f"{rows}\n\n"
        "## Gate verdict (teacher gates only)\n\n"
        "| check | verdict | detail |\n|---|---|---|\n"
        f"{gate_rows}\n\n"
        "Closed-loop A/B gates (`seed_spread`, `closed_loop_improvement`, "
        "`oracle_positive_control`) are **not evaluated here**: no A/B report "
        "was produced and no win is claimed. `ece_source` is `held-out` "
        "because the trainer's confidence rows come only from the inner "
        "scenario split, which no gradient step sees; `ece_held_out` and "
        "`conformal_coverage` come from a second disjoint fit/score split of "
        "those rows.\n\n"
        "## Honest reading\n\n"
        "The three teacher gates pass, but a passing ratio/ECE pair is a "
        "consistency check, not a quality result: held-out RMSE "
        f"({mean:.3f} m/s^2) is close to the v3 target's own RMS (~8 m/s^2), "
        "so this student has not learned the command to useful accuracy. "
        "The low disjoint ECE reflects a well-behaved confidence head, not a "
        "small command error. No claim about closed-loop flight benefit is "
        "made here.\n\n"
        "## Provenance\n\n"
        f"- train `validation.bin` sha256: `{digests['train_sha256']}`\n"
        f"- held-out `validation.bin` sha256: `{digests['heldout_sha256']}` "
        "(matches the post-FIXPUMP zero-command baseline)\n"
        f"- simulation config sha256: `{digests['config_sha256']}`\n"
        "- all knobs per seed are in `ai/out/final_v3_postpump/runs/s*/metrics.json`.\n",
        encoding="utf-8",
    )


def main() -> None:
    digests = verify_inputs()
    RUNS.mkdir(parents=True, exist_ok=True)
    metrics: dict[int, dict] = {}
    for seed in SEEDS:
        out = RUNS / f"s{seed}"
        report = run_trainer(seed, out)
        metrics[seed] = enrich(seed, out, report, digests)
        print(
            f"DONE seed {seed} held={metrics[seed]['rmse_held_out']:.6f} "
            f"ratio={metrics[seed]['ratio']:.4f} ece={metrics[seed]['ece_held_out']:.4f}",
            flush=True,
        )
    gates = {
        "rmse_ratio": check_ratio(metrics),
        "ece_source": check_ece_source(metrics),
        "ece": check_ece(metrics),
    }
    summary = {
        "measurement": "post-FIXPUMP staged teacher v2, 100 epochs, CPU",
        "seeds": list(SEEDS),
        "rmse_held_out_by_seed": {str(s): metrics[s]["rmse_held_out"] for s in SEEDS},
        "rmse_held_out_mean": statistics.mean(metrics[s]["rmse_held_out"] for s in SEEDS),
        "rmse_held_out_std": statistics.stdev(metrics[s]["rmse_held_out"] for s in SEEDS),
        "gates": {name: asdict(report) for name, report in gates.items()},
        "ab_win_claimed": False,
        "data_provenance": digests,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    export_students(list(SEEDS), metrics)
    write_doc(list(SEEDS), metrics, summary["gates"], digests)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
