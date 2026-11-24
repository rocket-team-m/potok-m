# SPDX-License-Identifier: GPL-3.0-or-later
"""RETRAIN-NIGHT driver: current teacher-v2 default, retrained on frozen data.

This runs the shipped default (``sft -> privileged -> dagger -> distill``) with
the train-split POTOKTRJ v3 truth sidecar. Two upgrades that landed after
RETRAIN-ALLFIX are active **because they are the trainer default**, not because
this driver tunes anything: AIOBJ's privileged impact objective (``--legacy-
objective`` is deliberately NOT passed) and AICAL's split-conformal confidence
(written by the trainer into each run directory). AICURR's adversarial
curriculum is opt-in (``--adversarial``); it is run as a separate arm only when
it can pass the unmodified teacher gates, otherwise its status is NOT RUN.

Data, seeds, epochs, batch size and every numeric knob are identical to
RETRAIN-ALLFIX. No physics constant, gate threshold, reward weight, command
envelope or confidence threshold was changed. The comparison to allfix is
descriptive: no closed-loop A/B was run and no win is claimed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
AI = ROOT / "ai"
SRC = AI / "src"
sys.path.insert(0, str(SRC))

from potok_ai.conf_integration import DISJOINT_ECE_SOURCE, frozen_step_map
from potok_ai.objective import IMPACT_FORMULA
from potok_ai.teacher_gates import check_ece, check_ece_source, check_ratio
from potok_ai.train_teacher_v2 import (
    SEEDS,
    TeacherV2Config,
    build_parser,
    config_hash,
)
from potok_ai.train_teacher_v2 import (
    STAGE_ORDER as TRAINER_STAGE_ORDER,
)

from potok_ai.curriculum import DEFAULT_WIN_RATE_CAP
from potok_ai.engine import OnlineShield, ShieldSample, TrainConfig
from potok_ai.student import CommandEnvelope

DATA = AI / "out" / "final_v3_postpump" / "train_only" / "train_validation.bin"
HELDOUT = AI / "out" / "final_v3_postpump" / "data" / "held-out" / "validation.bin"
SOURCE_CONFIG = AI / "out" / "v3_final" / "sim.toml"
# Train split only: the privileged builder fits its normalizer on every record
# handed to it, so held-out truth must never enter this file (AIFIX2).
TRUTH_SIDECAR = AI / "out" / "final_v3_postpump" / "data" / "train" / "validation.trajectory"
OUT = AI / "out" / "night_final"
RUNS = OUT / "runs"
EXPORTS = OUT / "exports"
ADV = OUT / "adversarial"
DOC = AI / "docs" / "TRAINING_RESULTS.md"
PYTHON = AI / ".venv" / "bin" / "python"
EPOCHS = 100
EXPECTED_STAGES = ("sft", "privileged", "dagger", "distill")
WINDOW_CAP = 1451  # one full episode of 12-tick windows at stride 1
COMMAND_BOUND = CommandEnvelope().accel  # typed envelope, not a tuned value
TARGET_RMS = 7.70  # ai/docs/TRAINING_RECIPE.md:10-13, documented target RMS

# Frozen post-FIXPUMP artifact digests (RETRAIN-ALLFIX / RETRAIN-DAGGER).
TRAIN_SHA256 = "34b35e0ff67f7eeecbd895b2efde8bebcfa159c1a3e78379da6750338b6930b6"
HELDOUT_SHA256 = "fdc1d1400b1c5f111e21f79dead37afa1dd9f058fe457edb78037dc6ebe89807"
SOURCE_CONFIG_SHA256 = "9c34e71f9d10bcf081c31f15f5332f7bb2fac04dddb28009a27cd30987ab5e91"
TRUTH_SIDECAR_SHA256 = "7f25bb69124638ea7542be99f43330fda1c5248513f1971f6bfd125f5f759cb8"

# Descriptive baseline only: ai/out/final_v3_allfix/summary.json (RETRAIN-ALLFIX).
ALLFIX_MEAN = 7.855551495797786
ALLFIX_STD = 2.6046004891820242

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
    "legacy_student_bound": False,  # AIFIX1: +/-10 envelope, not the old +/-1.
    "legacy_objective": False,  # AIOBJ: impact term ON when truth records exist.
    "adversarial": False,  # AICURR: opt-in arm only; default arm stays clean.
    "jepa_weight": 0.05,
    "entropy_weight": 1.0,
    "dagger_rounds": 1,
    "n_inner_held_out": 2,
    "rlvr_samples": 4,
    "rlvr_temperature": 0.05,
    "truth_sidecar": TRUTH_SIDECAR,  # AIFIX2: real train-split v3 truth.
    "calib_tol": 0.15,
    "opt": "adamw",
    "k3_per_head": 0,
    "k3_qk_clip": False,
    "k3_qk_clip_tau": 100.0,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_cli_config() -> None:
    """The exact emitted CLI must hash to the driver's recipe, both arms."""
    for adversarial in (False, True):
        argv = command(SEEDS[0], RUNS / f"s{SEEDS[0]}", adversarial=adversarial)
        cli_cfg = TeacherV2Config.from_args(build_parser().parse_args(argv[3:]))
        expected = TeacherV2Config(
            **{
                **CFG_KWARGS,
                "out": RUNS / f"s{SEEDS[0]}",
                "seed": SEEDS[0],
                "adversarial": adversarial,
            }
        )
        if config_hash(cli_cfg) != config_hash(expected):
            raise ValueError(
                f"CLI/config disagree for adversarial={adversarial}: "
                f"{config_hash(cli_cfg)} != {config_hash(expected)}"
            )


def verify_inputs() -> dict[str, str]:
    """Loud refusal of stale data, a moved stage order or a wrong sidecar."""
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
            raise FileNotFoundError(f"{label}: missing {path}; regenerate the frozen split")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(
                f"{label}: sha256 {actual} != {expected}; the input is not the frozen "
                "artifact (no data was substituted)"
            )
        digests[f"{label}_sha256"] = actual
    return digests


def assert_night(report: dict, seed: int, *, adversarial: bool) -> None:
    """Every landed mechanism and both fidelity arms must be visible, or fail."""
    where = "adversarial" if adversarial else "default"
    if tuple(report.get("stages_completed", ())) != EXPECTED_STAGES:
        raise ValueError(
            f"{where} seed {seed}: stages_completed {report.get('stages_completed')} != "
            f"{list(EXPECTED_STAGES)}; do not relabel the artifact"
        )
    if report.get("reward_provenance") != {}:
        raise ValueError(
            f"{where} seed {seed}: non-empty reward_provenance; RLVR was not opted into"
        )
    stages = report.get("stages", {})
    for name in EXPECTED_STAGES:
        if name not in stages:
            raise ValueError(f"{where} seed {seed}: stage {name} missing from metrics.json")
    privileged = stages["privileged"]["details"]
    dagger = stages["dagger"]["details"]
    distill = stages["distill"]["details"]
    objective = privileged.get("objective", {})
    checks = [
        ("AIFIX2 privileged_batches > 0", privileged.get("privileged_batches", 0) > 0),
        ("AIFIX2 privileged_source", privileged.get("privileged_source") == "privileged_teacher"),
        ("AIFIX2 train sidecar", privileged.get("truth_sidecar") == str(TRUTH_SIDECAR)),
        ("AIFIX2 label-blind eval", privileged.get("eval_label_blind") is True),
        ("AIOBJ impact default", objective.get("source") == "impact"),
        ("AIOBJ legacy off", objective.get("legacy") is False),
        ("AIOBJ no fallback reason", objective.get("reason") is None),
        ("AIOBJ formula recorded", objective.get("formula") == IMPACT_FORMULA),
        (
            "AIFIX3 on-policy rollout",
            str(dagger.get("rollout_source", "")).startswith("on_policy_student:"),
        ),
        ("AIFIX3 real privilege", dagger.get("privileged_source") == "v3_truth_sidecar"),
        ("AIFIX3 fields present", dagger.get("privileged_fields_present") is True),
        ("AIFIX3 student-induced rows", dagger.get("student_induced_rows", 0) > 0),
        ("AIFIX3 rollout digest", len(str(dagger.get("rollout_sha256", ""))) == 64),
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
        ("AICAL disjoint source", report.get("ece_source") == DISJOINT_ECE_SOURCE),
        ("AICAL ece equals held-out", report.get("ece") == report.get("ece_held_out")),
        (
            "AICAL finite held-out ece",
            isinstance(report.get("ece_held_out"), (int, float))
            and np.isfinite(report["ece_held_out"]),
        ),
        ("AICAL coverage in (0,1]", 0.0 < report.get("conformal_coverage", 0.0) <= 1.0),
    ]
    if adversarial:
        expected_intensity = {"sft": 0.0, "privileged": 0.1, "dagger": 0.2, "distill": 0.3}
        for name in EXPECTED_STAGES:
            info = stages[name]["details"].get("adversarial", {})
            checks += [
                (f"AICURR {name} enabled", info.get("enabled") is True),
                (f"AICURR {name} stage label", info.get("stage") == name),
                (f"AICURR {name} cap", info.get("win_rate_cap") == DEFAULT_WIN_RATE_CAP),
                (
                    f"AICURR {name} intensity",
                    math.isclose(
                        info.get("intensity", -1.0), expected_intensity[name], abs_tol=1e-9
                    ),
                ),
                (f"AICURR {name} offered", info.get("offered", 0) > 0),
                (f"AICURR {name} wins bounded", 0 <= info.get("won", -1) <= info.get("offered", 0)),
            ]
            if expected_intensity[name] == 0.0:
                checks.append((f"AICURR {name} calm clean", info.get("won") == 0))
    else:
        checks.append(
            (
                "AICURR default off",
                all("adversarial" not in stages[name]["details"] for name in EXPECTED_STAGES),
            )
        )
    failed = [name for name, ok in checks if not ok]
    if failed:
        raise ValueError(f"{where} seed {seed}: required mechanisms missing: {failed}")


def command(seed: int, out: Path, *, adversarial: bool) -> list[str]:
    """Exact trainer invocation; new upgrades enter only through shipped defaults."""
    cmd = [
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
    if adversarial:
        cmd.append("--adversarial")
    return cmd


def run_trainer(seed: int, out: Path, *, adversarial: bool) -> dict:
    """Run or reuse one seed; refuse a partial or mislabelled artifact."""
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.json"
    if metrics_path.is_file() and (out / "student.pt").is_file():
        report = json.loads(metrics_path.read_text())
        assert_night(report, seed, adversarial=adversarial)
        print(f"REUSE {'adv' if adversarial else 'default'} seed {seed}", flush=True)
        return report
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC)
    log = out / "run.log"
    print(
        f"TRAIN {'adv' if adversarial else 'default'} seed {seed} epochs={EPOCHS} log={log}",
        flush=True,
    )
    with log.open("w") as handle:
        proc = subprocess.run(
            command(seed, out, adversarial=adversarial),
            cwd=AI,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"seed {seed}: trainer exited {proc.returncode}; see {log}")
    report = json.loads(metrics_path.read_text())
    assert_night(report, seed, adversarial=adversarial)
    if not (out / "student.pt").is_file():
        raise FileNotFoundError(f"seed {seed}: distill stage wrote no student.pt")
    return report


def shield_smoke() -> dict:
    """One counted OnlineShield verification, not a training intervention count."""
    mach_max = next(
        field.default for field in fields(TrainConfig) if field.name == "shield_mach_max"
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
        raise AssertionError(f"shield counts {shield.counts.as_dict()} != {expected}")
    if any(row["command"] != [0.0, 0.0] for row in rows[1:]):
        raise AssertionError("shield intervention did not apply the classical fallback")
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


def enrich(seed: int, out: Path, report: dict, digests: dict, *, adversarial: bool) -> dict:
    """Add provenance only; the trainer's measurements are never recomputed."""
    cfg = TeacherV2Config(
        **{
            **CFG_KWARGS,
            "out": out,
            "seed": seed,
            "adversarial": adversarial,
        }
    )
    if config_hash(cfg) != report["config_hash"]:
        raise ValueError(
            f"seed {seed}: config_hash {report['config_hash']} != {config_hash(cfg)}; "
            "the recipe recorded in metrics.json differs from the driver recipe"
        )
    conformal = out / "conformal.json"
    if not conformal.is_file():
        raise FileNotFoundError(f"seed {seed}: AICAL wrote no conformal.json")
    frozen_step_map(json.loads(conformal.read_text()))  # LOUD on a bad edge map
    stages = report["stages"]
    privileged = stages["privileged"]["details"]
    dagger = stages["dagger"]["details"]
    distill = stages["distill"]["details"]
    enriched = dict(report)
    enriched.update(
        {
            # Top-level convenience mirrors of the stage evidence, so a reader
            # of summary/exports need not walk the stage tree (allfix layout).
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
            "night_provenance": True,
            "arm": "adversarial" if adversarial else "default",
            "confidence_source": "trainer AICAL hook (disjoint inner-held-out split)",
            "shield": "module verified in shield.json; not invoked by the staged trainer",
            "fixes": {
                "command_bound_mps2": COMMAND_BOUND,
                "truth_sidecar": str(TRUTH_SIDECAR),
                "rollout": "on_policy_student",
                "objective": "AIOBJ impact (default, truth records present)",
                "residual_policy": "enabled by trainer default (SHIELD02)",
                "online_shield": "module verified; not invoked by the trainer",
                "adversarial": "enabled via --adversarial" if adversarial else "not opted in",
            },
            "recipe": {
                **{k: str(v) if isinstance(v, Path) else v for k, v in CFG_KWARGS.items()},
                "adversarial": adversarial,
                "out": str(out),
                "epochs": EPOCHS,
            },
            "data_provenance": digests,
        }
    )
    metrics_path = out / "metrics.json"
    metrics_path.write_text(json.dumps(enriched, indent=2, sort_keys=True) + "\n")
    return enriched


def run_arm(adversarial: bool, digests: dict[str, str]) -> tuple[dict[int, dict], dict]:
    """Run three seeds of one arm in parallel, then gate the finished artifacts."""
    arm_dir = ADV if adversarial else OUT
    runs = arm_dir / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=len(SEEDS)) as pool:
        reports = list(
            pool.map(
                lambda seed: run_trainer(seed, runs / f"s{seed}", adversarial=adversarial),
                list(SEEDS),
            )
        )
    metrics = {
        seed: enrich(seed, runs / f"s{seed}", report, digests, adversarial=adversarial)
        for seed, report in zip(SEEDS, reports)
    }
    gates = {
        "rmse_ratio": check_ratio(metrics),
        "ece_source": check_ece_source(metrics),
        "ece": check_ece(metrics),
    }
    for name, gate in gates.items():
        print(
            f"GATE {'adv' if adversarial else 'default'} {name}: {'PASS' if gate.passed else 'FAIL'} {gate.detail}",
            flush=True,
        )
    return metrics, gates


def export_students(metrics: dict[int, dict]) -> None:
    """Copy each distilled student beside a provenance manifest (no A/B claim)."""
    EXPORTS.mkdir(parents=True, exist_ok=True)
    manifest = {
        "measurement": "distilled students from the RETRAIN-NIGHT default four-stage pipeline",
        "claim": "artifacts only; no closed-loop A/B was run and no win is claimed",
        "stage_order": list(EXPECTED_STAGES),
        "fixes": {
            "command_bound": COMMAND_BOUND,
            "truth_sidecar": str(TRUTH_SIDECAR),
            "rollout": "on_policy_student",
            "objective": "AIOBJ impact default (no --legacy-objective)",
            "confidence": "AICAL split-conformal, disjoint inner-held-out split",
            "residual_policy": "enabled by trainer default (SHIELD02)",
            "online_shield": "module verified in shield.json; not invoked by the trainer",
            "adversarial": "not opted in (see summary.json adversarial_arm)",
        },
        "rlvr": "not run (opt-in checkpoint selection only)",
        "export_format": "student.pt (torch state_dict of the distill-stage student)",
        "students": {},
    }
    for seed in SEEDS:
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
            "conformal_coverage": report["conformal_coverage"],
            "objective": report["stages"]["privileged"]["details"]["objective"],
            "student_command_bound": report["student_command_bound"],
            "rollout_source": report["rollout_source"],
            "privileged_batches": report["privileged_batches"],
        }
    (EXPORTS / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def write_doc(metrics: dict[int, dict], gates: dict, digests: dict, summary: dict) -> None:
    """Replace only the marked night-final section; earlier results stay intact."""
    held = [metrics[seed]["rmse_held_out"] for seed in SEEDS]
    mean = statistics.mean(held)
    std = statistics.stdev(held) if len(held) > 1 else 0.0
    delta = mean - ALLFIX_MEAN
    within = abs(delta) <= ALLFIX_STD
    rows = "\n".join(
        "| {seed} | {train:.6f} | {held:.6f} | {ratio:.4f} | {ece:.4f} | {cov:.4f} |".format(
            seed=seed,
            train=metrics[seed]["rmse_train"],
            held=metrics[seed]["rmse_held_out"],
            ratio=metrics[seed]["ratio"],
            ece=metrics[seed]["ece_held_out"],
            cov=metrics[seed]["conformal_coverage"],
        )
        for seed in SEEDS
    )
    gate_rows = "\n".join(
        f"| {name} | {'PASS' if result.passed else 'FAIL'} | {result.detail} |"
        for name, result in gates.items()
    )
    adv = summary["adversarial_arm"]
    adv_line = (
        f"**{adv['status']}** — {adv['reason']}"
        if adv["status"] == "NOT RUN"
        else (
            f"**{adv['status']}** — 3 seeds, 100 epochs, `--adversarial`; "
            f"teacher gates: {', '.join(n + '=' + ('PASS' if g['passed'] else 'FAIL') for n, g in adv['gates'].items())}. "
            "Descriptive second arm only; no win claimed."
        )
    )
    section = (
        "<!-- night-final:begin -->\n"
        "# Teacher-v2 results — RETRAIN-NIGHT (current)\n\n"
        f"The shipped four-stage default (`{' -> '.join(EXPECTED_STAGES)}`) was rerun "
        "on the frozen post-FIXPUMP split with the train-split v3 truth sidecar, "
        "**3 seeds x 100 epochs, CPU**, and no physics/gate/envelope/weight tuning. "
        "Two upgrades that landed after RETRAIN-ALLFIX are active through trainer "
        "defaults: the privileged **impact objective (AIOBJ)**, on because a truth "
        "sidecar with records is supplied and `--legacy-objective` is NOT passed, "
        "and the **split-conformal confidence (AICAL)** hook, whose disjoint ECE and "
        "coverage are read from the trainer's own `metrics.json` and `conformal.json`. "
        "AIFIX1/2/3, SHIELD01/02 and on-policy DAgger are inherited defaults and are "
        "re-checked per seed. RLVR stayed opt-in and was not run. "
        f"`--limit {WINDOW_CAP}` caps WINDOWS per batch builder, not episodes.\n\n"
        f"Held-out RMSE: **{mean:.6f} +/- {std:.6f} m/s^2** (sample std).\n\n"
        "| seed | train RMSE | held-out RMSE | ratio | disjoint ECE | conformal coverage |\n"
        "|---:|---:|---:|---:|---:|---:|\n"
        f"{rows}\n\n"
        "## Comparison to RETRAIN-ALLFIX — descriptive, no A/B win claim\n\n"
        "| run | held-out RMSE, mean +/- std (m/s^2) |\n"
        "|---|---:|\n"
        f"| RETRAIN-NIGHT (impact objective + conformal) | {mean:.6f} +/- {std:.6f} |\n"
        f"| RETRAIN-ALLFIX (all root-cause fixes, earlier code) | "
        f"{ALLFIX_MEAN:.6f} +/- {ALLFIX_STD:.6f} |\n\n"
        f"Difference of means: {delta:+.6f} m/s^2; the allfix sample std is "
        f"{ALLFIX_STD:.6f}, so the difference is "
        f"{'within' if within else 'outside'} one allfix seed spread. The two runs "
        "differ in the privileged objective and the confidence measurement at once, "
        "no arm isolates either, and there is no closed-loop harness, no independent "
        "flight-truth anchor, no positive control and no unweakened gate evaluation. "
        "`ab_win_claimed: false` in `summary.json`: **no win is claimed**.\n\n"
        "## Adversarial curriculum (AICURR)\n\n"
        f"{adv_line}\n\n"
        "## Gate verdict (teacher gates only)\n\n"
        "| check | verdict | detail |\n|---|---|---|\n"
        f"{gate_rows}\n\n"
        "Closed-loop A/B gates (`seed_spread`, `closed_loop_improvement`, "
        "`oracle_positive_control`) are **not evaluated**: no A/B report, no "
        "independent truth sidecar and no positive control. The ECE is a disjoint "
        "inner-held-out score; stage-local ECE values remain in-sample diagnostics.\n\n"
        "## Binding limitation — stated loudly\n\n"
        "AIFIX3's `world_step` advances only onboard sensor features through the "
        "unchanged `physics_baseline`/`final_command` plus named noise — a surrogate, "
        "not a Rust simulator step on (state, command). The v3 truth sidecar is emitted "
        "by that same simulator, so generator and verifier share a source; it is not "
        "an independent flight-truth anchor. The impact objective uses measured sensor "
        "columns and true velocity as an additional training term, not a new closed-loop "
        "measurement. A held-out RMSE near the target's own RMS "
        f"(~{TARGET_RMS:g} m/s²) is the expected shape of that binding.\n\n"
        "## Provenance\n\n"
        f"- train sha256: `{digests['train_sha256']}`; held-out sha256: "
        f"`{digests['heldout_sha256']}`\n"
        f"- sim config sha256: `{digests['config_sha256']}`; train-split truth sidecar "
        f"sha256: `{digests['truth_sidecar_sha256']}`\n"
        "- per-seed evidence: `ai/out/night_final/runs/s*/metrics.json` and "
        "`conformal.json`; shield: `ai/out/night_final/shield.json`; exports: "
        "`ai/out/night_final/exports/manifest.json`; full summary: "
        "`ai/out/night_final/summary.json`.\n"
        "<!-- night-final:end -->\n"
    )
    existing = DOC.read_text(encoding="utf-8") if DOC.is_file() else ""
    start, end = "<!-- night-final:begin -->", "<!-- night-final:end -->"
    if start in existing and end in existing:
        left = existing[: existing.index(start)]
        right = existing[existing.index(end) + len(end) :]
        DOC.write_text(left + section + right, encoding="utf-8")
    else:
        DOC.write_text(section + "\n---\n\n" + existing, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--adversarial-arm",
        action="store_true",
        help="also run the opt-in AICURR arm and gate it with the unmodified teacher gates",
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    verify_cli_config()
    digests = verify_inputs()
    shield = shield_smoke()

    metrics, gates = run_arm(False, digests)
    for seed in SEEDS:
        print(
            f"DONE default seed {seed} held={metrics[seed]['rmse_held_out']:.6f} "
            f"ratio={metrics[seed]['ratio']:.4f} ece={metrics[seed]['ece_held_out']:.4f}",
            flush=True,
        )
    export_students(metrics)

    held_mean = statistics.mean(metrics[s]["rmse_held_out"] for s in SEEDS)
    std = statistics.stdev(metrics[s]["rmse_held_out"] for s in SEEDS)
    delta = held_mean - ALLFIX_MEAN

    adv_path = OUT / "adversarial_summary.json"
    if args.adversarial_arm:
        adv_metrics, adv_gates = run_arm(True, digests)
        adv_report = {
            "status": "RUN"
            if all(g.passed for g in adv_gates.values())
            else "NOT RUN (gates failed)",
            "reason": "opt-in --adversarial, same 3 seeds/100 epochs, no gate or knob change",
            "seeds": list(SEEDS),
            "gates": {name: asdict(gate) for name, gate in adv_gates.items()},
            "rmse_held_out_by_seed": {str(s): adv_metrics[s]["rmse_held_out"] for s in SEEDS},
            "rmse_held_out_mean": statistics.mean(adv_metrics[s]["rmse_held_out"] for s in SEEDS),
            "win_rate_cap": DEFAULT_WIN_RATE_CAP,
            "ab_win_claimed": False,
            "claim": "separate opt-in arm; descriptive only, no win claim",
        }
        if not all(g.passed for g in adv_gates.values()):
            adv_report["reason"] += (
                "; teacher gates failed with no gate edit, so the arm is reported NOT RUN"
            )
        adv_path.write_text(json.dumps(adv_report, indent=2, sort_keys=True) + "\n")
    elif adv_path.is_file():
        adv_report = json.loads(adv_path.read_text())
    else:
        adv_report = {
            "status": "NOT RUN",
            "reason": "opt-in arm not launched; run with --adversarial-arm to gate it",
            "ab_win_claimed": False,
        }

    summary = {
        "measurement": "RETRAIN-NIGHT: current teacher-v2 default, impact objective + conformal, 3 seeds x 100 epochs, CPU",
        "stage_order": list(EXPECTED_STAGES),
        "rlvr_used": False,
        "seeds": list(SEEDS),
        "rmse_held_out_by_seed": {str(s): metrics[s]["rmse_held_out"] for s in SEEDS},
        "rmse_held_out_mean": held_mean,
        "rmse_held_out_std": std,
        "gates": {name: asdict(gate) for name, gate in gates.items()},
        "upgrades": {
            "AIOBJ": {
                "enabled": True,
                "source": metrics[SEEDS[0]]["stages"]["privileged"]["details"]["objective"][
                    "source"
                ],
                "formula": IMPACT_FORMULA,
                "flag": "default (--legacy-objective not passed)",
            },
            "AICAL": {
                "enabled": True,
                "ece_source": metrics[SEEDS[0]]["ece_source"],
                "conformal_files": [str(RUNS / f"s{s}" / "conformal.json") for s in SEEDS],
            },
            "AIFIX1_2_3_SHIELD01_02_DAgger": "inherited defaults, re-checked per seed in assert_night()",
        },
        "adversarial_arm": adv_report,
        "ab_win_claimed": False,
        "surrogate_binding": {
            "claimed": True,
            "reason": "on-policy world_step and the POTOKTRJ truth sidecar are simulator "
            "surrogates; expert labels are recorded answers. The impact objective is a "
            "training term, not an independent flight measurement.",
            "delta_within_one_allfix_std": abs(delta) <= ALLFIX_STD,
        },
        "comparison": {
            "baseline": "ai/out/final_v3_allfix (RETRAIN-ALLFIX)",
            "baseline_rmse_held_out_mean": ALLFIX_MEAN,
            "baseline_rmse_held_out_std": ALLFIX_STD,
            "delta_mean": delta,
            "claim": "descriptive comparison of two retrains; no A/B win claimed until a "
            "seeded closed-loop A/B with unweakened gates is run",
        },
        "shield_verification": shield,
        "data_provenance": digests,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_doc(metrics, gates, digests, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
