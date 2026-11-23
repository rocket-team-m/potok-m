# SPDX-License-Identifier: GPL-3.0-or-later
"""RETRAIN-ALLFIX driver: every landed AI root-cause fix, 3 seeds, 100 epochs.

Runs the four-stage default (``sft -> privileged -> dagger -> distill``) with
all fixes active and checked in the artifacts: the +/-10 m/s^2 student bound
(AIFIX1), the real POTOKTRJ v3 truth sidecar in the privileged stage (AIFIX2),
the on-policy student-in-the-loop DAgger rollout (AIFIX3), and the default-on
residual policy plus the online shield (SHIELD01/02). A regression of any of
them fails loudly instead of being relabelled.

Data, seeds, epochs and every numeric knob are unchanged from RETRAIN-DAGGER;
no physics, gate, threshold, weight or recipe value was tuned. The delta
against RETRAIN-DAGGER is descriptive only: no closed-loop A/B was run and no
win is claimed. The surrogate world step and simulator-emitted truth remain the
binding mechanisms; that is reported, not hidden.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import subprocess
import sys
from dataclasses import asdict
from dataclasses import fields as dataclass_fields
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
from potok_ai.train_teacher_v2 import (
    STAGE_ORDER as TRAINER_STAGE_ORDER,
)

from potok_ai.data import load_episodes
from potok_ai.engine import OnlineShield, ShieldSample, TrainConfig
from potok_ai.student import CommandEnvelope

DATA = AI / "out" / "final_v3_postpump" / "train_only" / "train_validation.bin"
HELDOUT = AI / "out" / "final_v3_postpump" / "data" / "held-out" / "validation.bin"
SOURCE_CONFIG = AI / "out" / "v3_final" / "sim.toml"
# Train-split truth only: the builder fits its normalizer on every record it is
# handed, so held-out truth must never be in this file (PRIVILEGED_TEACHER.md).
TRUTH_SIDECAR = AI / "out" / "final_v3_postpump" / "data" / "train" / "validation.trajectory"
OUT = AI / "out" / "final_v3_allfix"
RUNS = OUT / "runs"
EXPORTS = OUT / "exports"
DOC = AI / "docs" / "TRAINING_RESULTS.md"
PYTHON = AI / ".venv" / "bin" / "python"
EPOCHS = 100
# Pinned locally: a concurrent edit to the trainer's STAGE_ORDER must not
# silently reclassify an artifact.
EXPECTED_STAGES = ("sft", "privileged", "dagger", "distill")
# RETRAIN-DAGGER (ai/out/final_v3_postpump_dagger/summary.json): same data,
# seeds and knobs, but no truth sidecar, no on-policy rollout and the old
# bounded student. Mean and sample std, m/s^2.
BASELINE_MEAN = 7.902915971665115
BASELINE_STD = 0.49880831277850946
TARGET_RMS = 7.70  # ai/docs/TRAINING_RECIPE.md:10-13, not a tuned threshold.
# AIFIX1: the typed envelope is the only authority for the student head.
COMMAND_BOUND = CommandEnvelope().accel

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
    "residual_policy": True,  # SHIELD02 default; no --no-residual-policy.
    "legacy_student_bound": False,  # AIFIX1: default is the +/-10 envelope.
    "jepa_weight": 0.05,
    "entropy_weight": 1.0,
    "dagger_rounds": 1,
    "n_inner_held_out": 2,
    "rlvr_samples": 4,
    "rlvr_temperature": 0.05,
    "truth_sidecar": TRUTH_SIDECAR,  # AIFIX2: real v3 train-split truth.
    "calib_tol": 0.15,
    "opt": "adamw",
    "k3_per_head": 0,
    "k3_qk_clip": False,
    "k3_qk_clip_tau": 100.0,
}

# Post-FIXPUMP digests. The held-out digest equals the zero-command baseline
# rebuilt by RECALC01; the trajectory is the matching train-split v3 sidecar.
TRAIN_SHA256 = "34b35e0ff67f7eeecbd895b2efde8bebcfa159c1a3e78379da6750338b6930b6"
HELDOUT_SHA256 = "fdc1d1400b1c5f111e21f79dead37afa1dd9f058fe457edb78037dc6ebe89807"
SOURCE_CONFIG_SHA256 = "9c34e71f9d10bcf081c31f15f5332f7bb2fac04dddb28009a27cd30987ab5e91"
TRUTH_SIDECAR_SHA256 = "7f25bb69124638ea7542be99f43330fda1c5248513f1971f6bfd125f5f759cb8"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_inputs() -> dict[str, str]:
    """Loud refusal of stale data, a moved trainer order or a wrong sidecar."""
    if tuple(TRAINER_STAGE_ORDER) != EXPECTED_STAGES:
        raise ValueError(
            f"trainer STAGE_ORDER {tuple(TRAINER_STAGE_ORDER)} != pinned "
            f"{EXPECTED_STAGES}; retrain under a new output directory"
        )
    digests = {}
    for label, path, expected in (
        ("train", DATA, TRAIN_SHA256),
        ("heldout", HELDOUT, HELDOUT_SHA256),
        ("config", SOURCE_CONFIG, SOURCE_CONFIG_SHA256),
        ("truth_sidecar", TRUTH_SIDECAR, TRUTH_SIDECAR_SHA256),
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


def assert_allfix(report: dict, seed: int) -> None:
    """Every required fix must be visible in a trainer (or reused) report."""
    if tuple(report.get("stages_completed", ())) != EXPECTED_STAGES:
        raise ValueError(
            f"seed {seed}: stages_completed {report.get('stages_completed')} != "
            f"{list(EXPECTED_STAGES)}; do not relabel the artifact"
        )
    if report.get("reward_provenance") != {}:
        raise ValueError(f"seed {seed}: non-empty reward_provenance; RLVR was not opted into")
    stages = report.get("stages", {})
    for name in EXPECTED_STAGES:
        if name not in stages:
            raise ValueError(f"seed {seed}: stage {name} is missing from metrics.json")
    privileged = stages["privileged"]["details"]
    dagger = stages["dagger"]["details"]
    distill = stages["distill"]["details"]
    checks = (
        ("AIFIX2 privileged_batches > 0", privileged.get("privileged_batches", 0) > 0),
        (
            "AIFIX2 privileged_source",
            privileged.get("privileged_source") == "privileged_teacher",
        ),
        ("AIFIX2 train sidecar", privileged.get("truth_sidecar") == str(TRUTH_SIDECAR)),
        ("AIFIX2 label-blind eval", privileged.get("eval_label_blind") is True),
        (
            "AIFIX3 on-policy rollout",
            str(dagger.get("rollout_source", "")).startswith("on_policy_student:"),
        ),
        ("AIFIX3 real privilege", dagger.get("privileged_source") == "v3_truth_sidecar"),
        ("AIFIX3 fields present", dagger.get("privileged_fields_present") is True),
        ("AIFIX3 student-induced rows", dagger.get("student_induced_rows", 0) > 0),
        (
            "AIFIX3 rollout digest",
            len(str(dagger.get("rollout_sha256", ""))) == 64,
        ),
        (
            "AIFIX3 sidecar labels",
            dagger.get("label_source") == "reference_optimizer_v3:truth_sidecar_slots22..25",
        ),
        ("AIFIX1 bound >= 10", distill.get("student_command_bound", 0) >= COMMAND_BOUND),
        ("AIFIX1 legacy bound off", distill.get("legacy_student_bound") is False),
        ("SHIELD02 residual target", distill.get("residual_policy_target") is True),
        ("SHIELD02 residual enabled", report.get("residual_policy", {}).get("enabled") is True),
        (
            "SHIELD02 residual source",
            report.get("residual_policy", {}).get("source") == "residual_policy",
        ),
    )
    failed = [name for name, ok in checks if not ok]
    if failed:
        raise ValueError(f"seed {seed}: required landed fixes missing: {failed}")


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
        "--truth-sidecar",
        str(TRUTH_SIDECAR),
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
        assert_allfix(report, seed)
        print(f"REUSE seed {seed}", flush=True)
        return report
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
    assert_allfix(report, seed)
    if not (out / "student.pt").is_file():
        raise FileNotFoundError(f"seed {seed}: distill stage wrote no student.pt")
    return report


def shield_smoke() -> dict:
    """One counted OnlineShield verification; not a training intervention count.

    SHIELD01 lives in the engine's LRSAdapter. The staged teacher-v2 trainer is
    a supervised pipeline and does not invoke it, so the honest artifact is a
    counted check of the shipped module, not a fabricated training count.
    """
    mach_max = next(
        field.default for field in dataclass_fields(TrainConfig) if field.name == "shield_mach_max"
    )
    authority = CommandEnvelope().accel
    shield = OnlineShield(mach_max=mach_max, authority=authority)
    fallback = torch.zeros(2)
    cases = (
        ("in_envelope", torch.tensor([1.0, -1.0]), ShieldSample(mach=0.0, phase=1)),
        ("over_authority", torch.tensor([20.0, 0.0]), ShieldSample(mach=0.0, phase=1)),
        ("mach_limit", torch.zeros(2), ShieldSample(mach=mach_max + 0.1, phase=1)),
        ("forbidden_phase", torch.zeros(2), ShieldSample(mach=0.0, phase=0)),
        ("unverifiable_mach", torch.zeros(2), ShieldSample(mach=float("nan"), phase=1)),
    )
    rows = []
    for name, candidate, sample in cases:
        decision = shield.verify(candidate, fallback, sample)
        rows.append(
            {
                "case": name,
                "reason": None if decision.reason is None else decision.reason.value,
                "command": decision.command.tolist(),
            }
        )
    expected = {
        "accepted": 1,
        "mach_unavailable": 1,
        "mach_limit": 1,
        "authority": 1,
        "phase_forbidden": 1,
    }
    if shield.counts.as_dict() != expected:
        raise AssertionError(f"online shield counts {shield.counts.as_dict()} != {expected}")
    if any(row["command"] != [0.0, 0.0] for row in rows[1:]):
        raise AssertionError("online shield intervention did not apply the classical fallback")
    payload = {
        "measurement": "counted OnlineShield verification, not a training intervention count",
        "claim": "SHIELD01 module is present and every veto is counted; the staged "
        "teacher-v2 trainer does not invoke the shield",
        "mach_max": mach_max,
        "authority": authority,
        "counts": shield.counts.as_dict(),
        "cases": rows,
    }
    (OUT / "shield.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def honest_confidence(seed: int, out: Path, cfg: TeacherV2Config) -> tuple[float, float]:
    """Disjoint fit/score confidence score on the inner-held-out scenarios."""
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
            _command, logit, error = _predict(student, window, cond, target, cfg=cfg, student=True)
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
    """Add evaluation-contract and all-fix provenance without altering measurements."""
    cfg = TeacherV2Config(**{**CFG_KWARGS, "out": out, "seed": seed})
    if config_hash(cfg) != report["config_hash"]:
        raise ValueError(
            f"seed {seed}: config_hash {report['config_hash']} != "
            f"{config_hash(cfg)}; the recipe recorded in metrics.json differs"
        )
    if (
        report.get("ece_source") == "held-out"
        and "ece_held_out" in report
        and report.get("allfix_provenance") is True
    ):
        return report
    ece_held_out, coverage = honest_confidence(seed, out, cfg)
    stages = report["stages"]
    privileged = stages["privileged"]["details"]
    dagger = stages["dagger"]["details"]
    distill = stages["distill"]["details"]
    enriched = {
        "model": "teacher_v2",
        "seed": seed,
        "stage": report["stage"],
        "stages_completed": report["stages_completed"],
        "rmse_train": report["rmse_train"],
        "rmse_held_out": report["rmse_held_out"],
        "ratio": report["ratio"],
        "ece": report["ece"],
        "ece_source": "held-out",
        "ece_held_out": ece_held_out,
        "conformal_coverage": coverage,
        "config_hash": report["config_hash"],
        "steps": report["steps"],
        "entropy": report["entropy"],
        "entropy_enabled": report["entropy_enabled"],
        "reward_provenance": report["reward_provenance"],
        "residual_policy": report["residual_policy"],
        "module_sources": report["module_sources"],
        "stages": report["stages"],
        "allfix_provenance": True,
        "student_command_bound": distill["student_command_bound"],
        "legacy_student_bound": distill["legacy_student_bound"],
        "privileged_batches": privileged["privileged_batches"],
        "privileged_source": privileged["privileged_source"],
        "truth_sidecar": privileged["truth_sidecar"],
        "eval_label_blind": privileged["eval_label_blind"],
        "rollout_source": dagger["rollout_source"],
        "student_induced_rows": dagger["student_induced_rows"],
        "total_rows": dagger["total_rows"],
        "rollout_sha256": dagger["rollout_sha256"],
        "label_source": dagger["label_source"],
        "privileged_fields_present": dagger["privileged_fields_present"],
        "shield": "module verified in shield.json; not invoked by the staged trainer",
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
        "measurement": "distilled students from the post-FIXPUMP all-fix four-stage default",
        "claim": "artifacts only; no closed-loop A/B was run and no win is claimed",
        "stage_order": list(EXPECTED_STAGES),
        "fixes": {
            "command_bound": COMMAND_BOUND,
            "truth_sidecar": str(TRUTH_SIDECAR),
            "rollout": "on_policy_student",
            "residual_policy": "enabled by trainer default (SHIELD02)",
            "online_shield": "module verified in shield.json; not invoked by the trainer",
        },
        "rlvr": "not run (opt-in checkpoint selection only)",
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
            "student_command_bound": report["student_command_bound"],
            "rollout_source": report["rollout_source"],
            "privileged_batches": report["privileged_batches"],
        }
    (EXPORTS / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def write_doc(
    seeds: list[int], metrics: dict[int, dict], gates: dict, digests: dict, summary: dict
) -> None:
    """Write the results page: all fixes on, against the RETRAIN-DAGGER baseline."""
    held = [metrics[seed]["rmse_held_out"] for seed in seeds]
    mean = statistics.mean(held)
    std = statistics.stdev(held) if len(held) > 1 else 0.0
    delta = mean - BASELINE_MEAN
    within = abs(delta) <= BASELINE_STD
    spread_ratio = std / BASELINE_STD if BASELINE_STD else float("inf")
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
        "# Post-FIXPUMP teacher-v2 results (all AI root-cause fixes active)\n\n"
        f"The four-stage default (`{' -> '.join(EXPECTED_STAGES)}`) was run with every "
        "landed fix active and checked in each `metrics.json`: the +/-10 m/s² "
        "student bound (AIFIX1), the real train-split POTOKTRJ v3 truth sidecar "
        "in the privileged stage (AIFIX2), the on-policy student-in-the-loop "
        "DAgger rollout (AIFIX3), and the default-on bounded residual policy "
        "(SHIELD02). The online shield (SHIELD01) is verified and counted in "
        "`ai/out/final_v3_allfix/shield.json`; it is **not** invoked by the "
        "staged teacher-v2 trainer, so no training intervention count is "
        "claimed. RLVR remains opt-in checkpoint selection and was not run. "
        f"**{EPOCHS} epochs**, CPU only, seeds {list(seeds)}. Data, seeds, epochs "
        "and every recorded numeric knob are identical to RETRAIN-DAGGER; no "
        "physics constant, gate, threshold, weight or recipe value was tuned. "
        f"`--limit {WINDOW_CAP}` caps WINDOWS per batch builder, not episodes.\n\n"
        f"Held-out RMSE over three seeds: **{mean:.6f} +/- {std:.6f} m/s^2** "
        "(sample standard deviation).\n\n"
        "| seed | train RMSE | held-out RMSE | ratio | disjoint ECE | conformal coverage |\n"
        "|---:|---:|---:|---:|---:|---:|\n"
        f"{rows}\n\n"
        "## Comparison to RETRAIN-DAGGER (no win claimed)\n\n"
        "| run | fixes | held-out RMSE, mean +/- std (m/s^2) |\n"
        "|---|---|---:|\n"
        f"| this run | AIFIX1/2/3 + SHIELD01/02, DAgger default | "
        f"{mean:.6f} +/- {std:.6f} |\n"
        f"| RETRAIN-DAGGER | DAgger default, older bounded student, no sidecar, "
        f"recorded-window DAgger | {BASELINE_MEAN:.6f} +/- {BASELINE_STD:.6f} |\n\n"
        f"Difference of means: {delta:+.6f} m/s^2. The baseline's own sample std is "
        f"{BASELINE_STD:.6f} m/s^2, so the delta is "
        f"{'within' if within else 'outside'} one baseline seed spread"
        + (
            ". It does not separate from the baseline at the resolution this instrument can resolve"
            if within
            else ", but a single figure still does not constitute an A/B verdict"
        )
        + ". There is no closed-loop harness on these exports, no independent "
        "flight-truth anchor, no positive control and no unweakened gate "
        "evaluation. `ab_win_claimed: false` in `summary.json`. **No win is "
        "claimed until a seeded closed-loop A/B is run.**\n\n"
        "## Surrogate dynamics are the only binding mechanism — stated loudly\n\n"
        "The fixes make the wiring true; they do not add information about "
        "closed-loop flight. AIFIX3's `world_step` advances only the onboard "
        "sensor features through the unchanged `physics_baseline`/`final_command` "
        "plus a named noise stream — it is a surrogate, not the Rust simulator "
        "step on (state, command). The privileged sidecar is emitted by that "
        "same simulator, not a flight log, so the verifier and the generator "
        "share a source. The DAgger expert label is still the reference "
        "optimizer's recorded answer from sidecar slots 22..25, and the "
        "privileged stage still supervises an episode-pooled latent with MSE. "
        "With those two surrogate links the student-induced states are genuine "
        "bookkeeping facts but not flight dynamics. A held-out RMSE near the "
        f"target's own RMS (~{TARGET_RMS:g} m/s²) is the expected shape of that "
        "binding, not evidence that the fixes failed or that a controller "
        "improved. Honest artifacts are produced; the closed-loop A/B and an "
        "independent truth anchor remain owed.\n\n"
        "## Gate verdict (teacher gates only)\n\n"
        "| check | verdict | detail |\n|---|---|---|\n"
        f"{gate_rows}\n\n"
        "Closed-loop A/B gates (`seed_spread`, `closed_loop_improvement`, "
        "`oracle_positive_control`) are **not evaluated**: no A/B report, no "
        "flight-truth sidecar, no positive control. `ece_source` is `held-out` "
        "because confidence rows come only from the inner scenario split; "
        "`ece_held_out` and `conformal_coverage` are a second disjoint fit/score "
        "split of those rows.\n\n"
        "## Honest reading\n\n"
        "The three teacher gates are consistency checks, not a quality result. "
        f"Held-out RMSE ({mean:.3f} m/s²) is close to the v3 target's own RMS, so "
        "the student has not learned the command to useful accuracy. The low "
        "disjoint ECE reflects a well-behaved confidence head, not a small "
        "command error. Seed spread is "
        f"{std:.6f} m/s², {spread_ratio:.1f}x the baseline's {BASELINE_STD:.6f}; "
        "the on-policy surrogate rollout made three identical-config seeds far "
        "less reproducible, not more. The "
        f"{delta:+.6f} m/s² mean difference is not a separable effect. No claim "
        "about closed-loop flight benefit follows.\n\n"
        "## Provenance\n\n"
        f"- train `validation.bin` sha256: `{digests['train_sha256']}`\n"
        f"- held-out `validation.bin` sha256: `{digests['heldout_sha256']}`\n"
        f"- simulation config sha256: `{digests['config_sha256']}`\n"
        f"- train-split POTOKTRJ v3 sidecar sha256: "
        f"`{digests['truth_sidecar_sha256']}`\n"
        "- all knobs and per-fix evidence are in "
        "`ai/out/final_v3_allfix/runs/s*/metrics.json`; shield verification is "
        "in `ai/out/final_v3_allfix/shield.json`; exports and digests are in "
        "`ai/out/final_v3_allfix/exports/manifest.json`.\n"
        "- the earlier RETRAIN-DAGGER artifacts remain untouched in "
        "`ai/out/final_v3_postpump_dagger/`.\n",
        encoding="utf-8",
    )


def main() -> None:
    digests = verify_inputs()
    OUT.mkdir(parents=True, exist_ok=True)
    shield = shield_smoke()
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
    held_mean = statistics.mean(metrics[s]["rmse_held_out"] for s in SEEDS)
    delta = held_mean - BASELINE_MEAN
    summary = {
        "measurement": "post-FIXPUMP all-fix four-stage default, 100 epochs, CPU",
        "stage_order": list(EXPECTED_STAGES),
        "rlvr_used": False,
        "fixes": {
            "command_bound_mps2": COMMAND_BOUND,
            "truth_sidecar": str(TRUTH_SIDECAR),
            "truth_sidecar_sha256": digests["truth_sidecar_sha256"],
            "privileged_batches_by_seed": {str(s): metrics[s]["privileged_batches"] for s in SEEDS},
            "rollout_source": metrics[SEEDS[0]]["rollout_source"],
            "student_induced_rows_by_seed": {
                str(s): metrics[s]["student_induced_rows"] for s in SEEDS
            },
            "residual_policy": "enabled by trainer default (SHIELD02)",
            "online_shield": "module verified in shield.json; not invoked by the trainer",
        },
        "seeds": list(SEEDS),
        "rmse_held_out_by_seed": {str(s): metrics[s]["rmse_held_out"] for s in SEEDS},
        "rmse_held_out_mean": held_mean,
        "rmse_held_out_std": statistics.stdev(metrics[s]["rmse_held_out"] for s in SEEDS),
        "seed_spread_ratio_vs_baseline": (
            statistics.stdev(metrics[s]["rmse_held_out"] for s in SEEDS) / BASELINE_STD
        ),
        "gates": {name: asdict(report) for name, report in gates.items()},
        "ab_win_claimed": False,
        "surrogate_binding": {
            "claimed": True,
            "reason": "on-policy world_step and the POTOKTRJ truth sidecar are both "
            "simulator surrogates; expert labels are recorded answers. No "
            "closed-loop flight dynamics are measured by these fixes.",
            "delta_within_one_baseline_std": abs(delta) <= BASELINE_STD,
        },
        "comparison": {
            "baseline": "ai/out/final_v3_postpump_dagger (RETRAIN-DAGGER)",
            "baseline_rmse_held_out_mean": BASELINE_MEAN,
            "baseline_rmse_held_out_std": BASELINE_STD,
            "delta_mean": delta,
            "claim": "two measurements on the same data; no A/B win claimed until "
            "a seeded closed-loop A/B with unweakened gates is run",
        },
        "shield_verification": shield,
        "data_provenance": digests,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    export_students(list(SEEDS), metrics)
    write_doc(list(SEEDS), metrics, summary["gates"], digests, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
