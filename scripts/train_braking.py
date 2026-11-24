#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""BRK02 retrain: the shipped teacher-v2 default with --braking-teacher.

Frozen RETRAIN-NIGHT recipe, seeds and epochs, on the same train-split data and
truth sidecar. The only change is the explicit ``--braking-teacher`` flag: the
privileged and DAgger targets come from the BRK01 radial reference instead of
the recorded optimizer labels. No physics constant, gate, command envelope,
confidence threshold or loss weight is changed here, and no win is claimed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AI = ROOT / "ai"
SRC = AI / "src"
sys.path.insert(0, str(SRC))

from potok_ai.conf_integration import DISJOINT_ECE_SOURCE, frozen_step_map
from potok_ai.objective import IMPACT_FORMULA, TRUTH_VEL_END, TRUTH_VEL_START
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

from potok_ai.student import CommandEnvelope

DATA = AI / "out" / "final_v3_postpump" / "train_only" / "train_validation.bin"
HELDOUT = AI / "out" / "final_v3_postpump" / "data" / "held-out" / "validation.bin"
SOURCE_CONFIG = AI / "out" / "v3_final" / "sim.toml"
TRUTH_SIDECAR = AI / "out" / "final_v3_postpump" / "data" / "train" / "validation.trajectory"
OUT = AI / "out" / "braking"
RUNS = OUT / "runs"
EXPORTS = OUT / "exports"
DOC = AI / "docs" / "TRAINING_RESULTS.md"
PYTHON = AI / ".venv" / "bin" / "python"
EPOCHS = 100
EXPECTED_STAGES = ("sft", "privileged", "dagger", "distill")
WINDOW_CAP = 1451
COMMAND_BOUND = CommandEnvelope().accel
TRAIN_SHA256 = "34b35e0ff67f7eeecbd895b2efde8bebcfa159c1a3e78379da6750338b6930b6"
HELDOUT_SHA256 = "fdc1d1400b1c5f111e21f79dead37afa1dd9f058fe457edb78037dc6ebe89807"
SOURCE_CONFIG_SHA256 = "9c34e71f9d10bcf081c31f15f5332f7bb2fac04dddb28009a27cd30987ab5e91"
TRUTH_SIDECAR_SHA256 = "7f25bb69124638ea7542be99f43330fda1c5248513f1971f6bfd125f5f759cb8"

# Identical to scripts/train_night_final.py::CFG_KWARGS; braking_teacher is the
# BRK01 flag, already the trainer default but passed explicitly in command().
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
    "residual_policy": True,
    "legacy_student_bound": False,
    "legacy_objective": False,
    "adversarial": False,
    "braking_teacher": True,
    "jepa_weight": 0.05,
    "entropy_weight": 1.0,
    "dagger_rounds": 1,
    "n_inner_held_out": 2,
    "rlvr_samples": 4,
    "rlvr_temperature": 0.05,
    "truth_sidecar": TRUTH_SIDECAR,
    "calib_tol": 0.15,
    "opt": "adamw",
    "k3_per_head": 0,
    "k3_qk_clip": False,
    "k3_qk_clip_tau": 100.0,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_cli_config() -> None:
    """The emitted CLI must hash to the driver recipe, including the brake flag."""
    argv = command(SEEDS[0], RUNS / f"s{SEEDS[0]}")
    cli_cfg = TeacherV2Config.from_args(build_parser().parse_args(argv[3:]))
    expected = TeacherV2Config(
        **{**CFG_KWARGS, "out": RUNS / f"s{SEEDS[0]}", "seed": SEEDS[0]}
    )
    if config_hash(cli_cfg) != config_hash(expected):
        raise ValueError(
            f"CLI/config disagree: {config_hash(cli_cfg)} != {config_hash(expected)}"
        )
    if not cli_cfg.braking_teacher:
        raise ValueError("the emitted CLI does not enable --braking-teacher")


def verify_inputs() -> dict[str, str]:
    """Loud refusal of stale data, a moved stage order or a wrong sidecar."""
    if tuple(TRAINER_STAGE_ORDER) != EXPECTED_STAGES:
        raise ValueError(
            f"trainer STAGE_ORDER {tuple(TRAINER_STAGE_ORDER)} != pinned {EXPECTED_STAGES}"
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
                f"{label}: sha256 {actual} != {expected}; the input is not the frozen artifact"
            )
        digests[f"{label}_sha256"] = actual
    for name, path in (
        ("trainer_source", SRC / "potok_ai" / "train_teacher_v2.py"),
        ("braking_teacher_source", SRC / "potok_ai" / "braking_teacher.py"),
        ("teacher_gates_source", SRC / "potok_ai" / "teacher_gates.py"),
        ("objective_source", SRC / "potok_ai" / "objective.py"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{name}: missing {path}")
        digests[f"{name}_sha256"] = sha256(path)
    # BRK01B: slots 11..14 read body rate/mass as velocity. A retrain under the
    # stale offset is INVALID; refuse it before any epoch is run.
    if (TRUTH_VEL_START, TRUTH_VEL_END) != (3, 6):
        raise ValueError(
            f"impact objective velocity slots {TRUTH_VEL_START}:{TRUTH_VEL_END} != 3:6 "
            "(BRK01B); a retrain under the old offset is INVALID"
        )
    digests["objective_version"] = (
        f"TRUTH_VEL_START={TRUTH_VEL_START},TRUTH_VEL_END={TRUTH_VEL_END}"
    )
    digests["objective_head"] = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return digests


def assert_braking(report: dict, seed: int) -> None:
    """Every inherited mechanism and the BRK01 reference must be visible per seed."""
    where = f"braking seed {seed}"
    if tuple(report.get("stages_completed", ())) != EXPECTED_STAGES:
        raise ValueError(f"{where}: stages_completed {report.get('stages_completed')} is not pinned")
    if report.get("reward_provenance") != {}:
        raise ValueError(f"{where}: non-empty reward_provenance; RLVR was not opted into")
    stages = report.get("stages", {})
    for name in EXPECTED_STAGES:
        if name not in stages:
            raise ValueError(f"{where}: stage {name} missing from metrics.json")
    privileged = stages["privileged"]["details"]
    dagger = stages["dagger"]["details"]
    distill = stages["distill"]["details"]
    objective = privileged.get("objective", {})
    for arm, details in (("privileged", privileged), ("dagger", dagger)):
        brake = details.get("braking", {})
        counts = brake.get("reason_counts", {})
        active = counts.get("ok", 0) + counts.get("cap_clamped", 0)
        checks = [
            (f"BRK01 {arm} enabled", brake.get("enabled") is True),
            (f"BRK01 {arm} source", brake.get("source") == "radial_braking_reference"),
            (f"BRK01 {arm} no fallback reason", brake.get("reason") is None),
            (f"BRK01 {arm} cap", brake.get("cap_mps2") == COMMAND_BOUND),
            (f"BRK01 {arm} counted reasons", isinstance(counts, dict) and bool(counts)),
        ]
        if arm == "dagger":
            # The deployed student is distilled from the DAgger target, so an
            # inert DAgger reference would make this arm a different experiment.
            checks.append((f"BRK01 {arm} active ticks counted", active > 0))
        elif active == 0:
            # BRK01's _window_ticks() maps a window batch whose episode indices
            # are unique to tick=window-1, so every privileged row is boost-gated.
            # Refuse silently-inert evidence, accept only the counted, named
            # defect, and report it in summary.json and the results doc.
            checks.append(("BRK01 privileged inert counted", counts.get("boost_gate", 0) > 0))
            print(
                f"COUNTED BRK01 seed {seed}: privileged braking target inert "
                f"({counts.get('boost_gate', 0)} boost-gated rows, 0 active); "
                "DAgger reference is active; see summary.json",
                flush=True,
            )
        failed = [name for name, ok in checks if not ok]
        if failed:
            raise ValueError(f"{where}: {failed}; braking facts: {brake!r}")
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
        (
            "AICURR default off",
            all("adversarial" not in stages[name]["details"] for name in EXPECTED_STAGES),
        ),
    ]
    failed = [name for name, ok in checks if not ok]
    if failed:
        raise ValueError(f"{where}: required mechanisms missing: {failed}")


def command(seed: int, out: Path) -> list[str]:
    """Exact trainer invocation: the frozen recipe plus the explicit brake flag."""
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
        "--braking-teacher",
    ]


def run_trainer(seed: int, out: Path) -> dict:
    """Run or reuse one seed; refuse a partial or mislabelled artifact."""
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.json"
    if metrics_path.is_file() and (out / "student.pt").is_file():
        report = json.loads(metrics_path.read_text())
        assert_braking(report, seed)
        cfg = TeacherV2Config(
            **{**CFG_KWARGS, "out": out, "seed": seed}
        )
        if config_hash(cfg) != report["config_hash"]:
            raise ValueError(f"seed {seed}: reused metrics config_hash disagrees with the recipe")
        print(f"REUSE braking seed {seed}", flush=True)
        return report
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC)
    log = out / "run.log"
    print(f"TRAIN braking seed {seed} epochs={EPOCHS} log={log}", flush=True)
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
    assert_braking(report, seed)
    if not (out / "student.pt").is_file():
        raise FileNotFoundError(f"seed {seed}: distill stage wrote no student.pt")
    return report


def enrich(seed: int, out: Path, report: dict, digests: dict) -> dict:
    """Add provenance only; the trainer's measurements are never recomputed."""
    cfg = TeacherV2Config(**{**CFG_KWARGS, "out": out, "seed": seed})
    if config_hash(cfg) != report["config_hash"]:
        raise ValueError(
            f"seed {seed}: config_hash {report['config_hash']} != {config_hash(cfg)}"
        )
    conformal = out / "conformal.json"
    if not conformal.is_file():
        raise FileNotFoundError(f"seed {seed}: AICAL wrote no conformal.json")
    frozen_step_map(json.loads(conformal.read_text()))
    stages = report["stages"]
    privileged = stages["privileged"]["details"]
    dagger = stages["dagger"]["details"]
    distill = stages["distill"]["details"]
    enriched = dict(report)
    enriched.update(
        {
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
            "braking_provenance": {
                "privileged": privileged["braking"],
                "dagger": dagger["braking"],
            },
            "braking_arm": True,
            "confidence_source": "trainer AICAL hook (disjoint inner-held-out split)",
            "shield": "module verified in RETRAIN-NIGHT; not invoked by the staged trainer",
            "fixes": {
                "command_bound_mps2": COMMAND_BOUND,
                "truth_sidecar": str(TRUTH_SIDECAR),
                "rollout": "on_policy_student",
                "objective": "AIOBJ impact (default, truth records present)",
                "residual_policy": "enabled by trainer default (SHIELD02)",
                "online_shield": "not invoked by the trainer",
                "braking_teacher": "explicit --braking-teacher",
                "adversarial": "not opted in",
            },
            "recipe": {
                **{k: str(v) if isinstance(v, Path) else v for k, v in CFG_KWARGS.items()},
                "out": str(out),
                "epochs": EPOCHS,
            },
            "data_provenance": digests,
        }
    )
    metrics_path = out / "metrics.json"
    metrics_path.write_text(json.dumps(enriched, indent=2, sort_keys=True) + "\n")
    return enriched


def run_arm(digests: dict[str, str]) -> tuple[dict[int, dict], dict]:
    """Run three seeds, then gate the finished artifacts."""
    RUNS.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=len(SEEDS)) as pool:
        reports = list(pool.map(lambda seed: run_trainer(seed, RUNS / f"s{seed}"), list(SEEDS)))
    metrics = {
        seed: enrich(seed, RUNS / f"s{seed}", report, digests)
        for seed, report in zip(SEEDS, reports)
    }
    gates = {
        "rmse_ratio": check_ratio(metrics),
        "ece_source": check_ece_source(metrics),
        "ece": check_ece(metrics),
    }
    for name, gate in gates.items():
        print(f"GATE braking {name}: {'PASS' if gate.passed else 'FAIL'} {gate.detail}", flush=True)
    return metrics, gates


def export_students(metrics: dict[int, dict]) -> None:
    """Copy each distilled student beside a provenance manifest (no A/B claim)."""
    EXPORTS.mkdir(parents=True, exist_ok=True)
    manifest = {
        "measurement": "distilled students from the BRK02 --braking-teacher four-stage retrain",
        "claim": "artifacts only; no win is claimed from these exports alone",
        "stage_order": list(EXPECTED_STAGES),
        "arm": "braking",
        "teacher": "BRK01 radial braking reference (privileged and DAgger targets)",
        "fixes": {
            "command_bound": COMMAND_BOUND,
            "truth_sidecar": str(TRUTH_SIDECAR),
            "rollout": "on_policy_student",
            "objective": "AIOBJ impact default (no --legacy-objective)",
            "confidence": "AICAL split-conformal, disjoint inner-held-out split",
            "residual_policy": "enabled by trainer default (SHIELD02)",
            "online_shield": "not invoked by the trainer",
            "adversarial": "not opted in",
        },
        "rlvr": "not run (opt-in checkpoint selection only)",
        "export_format": "student.pt (torch state_dict of the distill-stage student)",
        "students": {},
    }
    for seed in SEEDS:
        source = RUNS / f"s{seed}" / "student.pt"
        destination = EXPORTS / f"teacher_braking_s{seed}.pt"
        destination.write_bytes(source.read_bytes())
        report = metrics[seed]
        manifest["students"][str(seed)] = {
            "file": destination.name,
            "sha256": sha256(destination),
            "config_hash": report["config_hash"],
            "rmse_held_out": report["rmse_held_out"],
            "ece_source": report["ece_source"],
            "conformal_coverage": report["conformal_coverage"],
            "student_command_bound": report["student_command_bound"],
            "rollout_source": report["rollout_source"],
            "privileged_batches": report["privileged_batches"],
        }
    (EXPORTS / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def privileged_inert(metrics: dict[int, dict]) -> dict[str, bool]:
    """Which seeds have a 100% boost-gated (inert) privileged braking target."""
    result: dict[str, bool] = {}
    for seed in SEEDS:
        counts = metrics[seed]["braking_provenance"]["privileged"]["reason_counts"]
        active = counts.get("ok", 0) + counts.get("cap_clamped", 0)
        result[str(seed)] = active == 0 and counts.get("boost_gate", 0) > 0
    return result


def write_doc(metrics: dict[int, dict], gates: dict, digests: dict) -> None:
    """Replace only the marked braking section; earlier results stay intact."""
    held = [metrics[seed]["rmse_held_out"] for seed in SEEDS]
    mean = statistics.mean(held)
    std = statistics.stdev(held)
    inert = privileged_inert(metrics)
    inert_note = (
        "**BRK01 defect, counted and not tuned away:** the privileged braking target "
        f"is 100% boost-gated (zero demand) on seeds {', '.join(s for s, v in inert.items() if v)}. "
        "DAgger emits active braking commands; the distilled student is trained from "
        "that active target. This is not a clean end-to-end braking-teacher run and no "
        "win is claimed on the strength of it."
        if any(inert.values())
        else "All seeds have an active privileged braking target."
    )
    rows = "\n".join(
        "| {seed} | {train:.6f} | {held:.6f} | {ratio:.4f} | {ece:.4f} |".format(
            seed=seed,
            train=metrics[seed]["rmse_train"],
            held=metrics[seed]["rmse_held_out"],
            ratio=metrics[seed]["ratio"],
            ece=metrics[seed]["ece_held_out"],
        )
        for seed in SEEDS
    )
    gate_rows = "\n".join(
        f"| {name} | {'PASS' if result.passed else 'FAIL'} | {result.detail} |"
        for name, result in gates.items()
    )
    section = (
        "<!-- braking:begin -->\n"
        "# Braking-teacher results — BRK02 (explicit --braking-teacher)\n\n"
        "The shipped four-stage default (`sft -> privileged -> dagger -> distill`) was rerun "
        "on the frozen post-FIXPUMP split with the train-split v3 truth sidecar, "
        "**3 seeds x 100 epochs, CPU**, with the explicit `--braking-teacher` flag. "
        "The BRK01 radial reference replaces the recorded optimizer labels in the "
        "privileged and DAgger stages; every other recipe value is identical to "
        "RETRAIN-NIGHT. No physics constant, gate threshold, command envelope, "
        "confidence threshold or loss weight was tuned. `--limit 1451` caps WINDOWS "
        "per batch builder, not episodes. RLVR stayed opt-in and was not run.\n\n"
        f"Held-out RMSE: **{mean:.6f} +/- {std:.6f} m/s^2** (sample std).\n\n"
        "| seed | train RMSE | held-out RMSE | ratio | disjoint ECE |\n"
        "|---:|---:|---:|---:|---:|\n"
        f"{rows}\n\n"
        "## Gate verdict (teacher gates only)\n\n"
        "| check | verdict | detail |\n|---|---|---|\n"
        f"{gate_rows}\n\n"
        "Closed-loop gates are reported separately in `scripts/ab_report_braking.json/.md` "
        "with the **unweakened v6 gates**; no win is claimed here. The ECE is a disjoint "
        "inner-held-out score; stage-local ECE values remain in-sample diagnostics.\n\n"
        "## Known BRK01 limitation\n\n"
        f"{inert_note}\n\n"
        "## Binding limitation — stated loudly\n\n"
        "The BRK01 target is a privileged simulator reference. The on-policy "
        "`world_step` still advances surrogate sensor features, and the v3 truth "
        "sidecar is emitted by the same simulator, so generator and verifier share a "
        "source. This is not an independent flight-truth anchor and not a flight claim.\n\n"
        "## Provenance\n\n"
        f"- train sha256: `{digests['train_sha256']}`; held-out sha256: "
        f"`{digests['heldout_sha256']}`\n"
        f"- config sha256: `{digests['config_sha256']}`; truth sidecar sha256: "
        f"`{digests['truth_sidecar_sha256']}`\n"
        f"- trainer sha256: `{digests['trainer_source_sha256']}`; BRK01 reference sha256: "
        f"`{digests['braking_teacher_source_sha256']}`\n"
        f"- BRK01B objective sha256: `{digests['objective_source_sha256']}`; "
        f"version `{digests['objective_version']}` at `{digests['objective_head']}`\n"
        "- per-seed evidence: `ai/out/braking/runs/s*/metrics.json` and "
        "`conformal.json`; exports: `ai/out/braking/exports/manifest.json`; full "
        "summary: `ai/out/braking/summary.json`.\n"
        "<!-- braking:end -->\n"
    )
    existing = DOC.read_text(encoding="utf-8") if DOC.is_file() else ""
    marker_start, marker_end = "<!-- braking:begin -->", "<!-- braking:end -->"
    if marker_start in existing and marker_end in existing:
        left = existing[: existing.index(marker_start)]
        right = existing[existing.index(marker_end) + len(marker_end) :]
        DOC.write_text(left + section + right, encoding="utf-8")
    else:
        DOC.write_text(section + "\n---\n\n" + existing, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    verify_cli_config()
    digests = verify_inputs()

    metrics, gates = run_arm(digests)
    for seed in SEEDS:
        print(
            f"DONE braking seed {seed} held={metrics[seed]['rmse_held_out']:.6f} "
            f"ratio={metrics[seed]['ratio']:.4f} ece={metrics[seed]['ece_held_out']:.4f}",
            flush=True,
        )
    export_students(metrics)

    mean = statistics.mean(metrics[s]["rmse_held_out"] for s in SEEDS)
    summary = {
        "measurement": (
            "BRK02: BRK01 radial braking teacher, 3 seeds x 100 epochs CPU, "
            "frozen RETRAIN-NIGHT recipe plus explicit --braking-teacher"
        ),
        "stage_order": list(EXPECTED_STAGES),
        "arm": "braking",
        "rlvr_used": False,
        "seeds": list(SEEDS),
        "rmse_held_out_by_seed": {str(s): metrics[s]["rmse_held_out"] for s in SEEDS},
        "rmse_held_out_mean": mean,
        "rmse_held_out_std": statistics.stdev(metrics[s]["rmse_held_out"] for s in SEEDS),
        "gates": {name: asdict(gate) for name, gate in gates.items()},
        "braking": {
            "enabled": True,
            "source": "radial_braking_reference",
            "flag": "--braking-teacher",
            "cap_mps2": COMMAND_BOUND,
            "privileged_target_inert": privileged_inert(metrics),
            "limitation": (
                "BRK01 _window_ticks() boost-gates every privileged window; DAgger "
                "target is active. No gate, physics or envelope value was changed."
            ),
            "per_seed": {
                str(s): metrics[s]["braking_provenance"] for s in SEEDS
            },
        },
        "upgrades": {
            "braking_teacher": "explicit CLI flag; radial law from BRK01, no fitted coefficient",
            "AIOBJ": {
                "enabled": True,
                "source": metrics[SEEDS[0]]["stages"]["privileged"]["details"]["objective"]["source"],
                "formula": IMPACT_FORMULA,
                "flag": "default (--legacy-objective not passed)",
            },
            "AICAL": {
                "enabled": True,
                "ece_source": metrics[SEEDS[0]]["ece_source"],
                "conformal_files": [str(RUNS / f"s{s}" / "conformal.json") for s in SEEDS],
            },
            "AIFIX1_2_3_SHIELD01_02_DAgger": "inherited defaults, re-checked per seed",
        },
        "ab_win_claimed": False,
        "surrogate_binding": {
            "claimed": True,
            "reason": (
                "The braking target is a privileged simulator reference; the on-policy "
                "world_step and the POTOKTRJ sidecar are simulator surrogates, not flight logs."
            ),
        },
        "data_provenance": digests,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_doc(metrics, gates, digests)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
