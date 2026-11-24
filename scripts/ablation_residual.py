#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Deterministic residual-policy deployment ablation on the FINAL all-fix exports.

Same three seeds (1/2/3), same 20 held-out episodes (scenarios 16-19), same
frozen post-FIXPUMP physics and the same read-only v3 control seam as v5. Only
the *deployment composition* of the shipped distill export changes; no trainer,
physics constant, weight, threshold or gate is touched and no export is retrained.

Arms (all gated by the paired baseline's ``ai_healthy`` target and clipped at the
seam's hard +/-10 m/s^2 bound):

  residual_on    command = clip(physics_baseline + export_delta)  (shipped v5 arm)
  residual_off   command = clip(export_delta)   -- classical superposition removed
  classical_only command = clip(physics_baseline) -- learned correction removed;
                 this is also the literal ResidualPolicy(enabled=False) fallback
                 (ai/src/potok_ai/residual_policy.py:147-149)
  zero_command   no TSV at all (Simulator::run_episode, ControlMode::None) -- the
                 paired control the unweakened gates measure improvement against
  oracle_gtheta  run_episode_with_oracle / fin_accel_command -- positive control

Both readings of "residual OFF" therefore have exact numbers: the superposition-
off arm is ``residual_off``, and the module's disabled fallback is
``classical_only``. No win is claimed: every unweakened gate must pass or the
verdict is FAIL.

Usage:
    ai/.venv/bin/python scripts/ablation_residual.py
Work files live outside the repo (default /tmp/opencode/ablation_residual).
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import ab_v5 as base  # reuse the frozen harness helpers, not a second implementation

ROOT = base.ROOT
sys.path.insert(0, str(ROOT / "ai" / "src"))
from potok_ai import teacher_gates
from potok_ai.data import SMART_FEATURES, read_validation
from potok_ai.physics import physics_baseline

OUT_JSON = SCRIPTS / "ablation_residual.json"
OUT_MD = SCRIPTS / "ablation_residual.md"
WORK = Path("/tmp/opencode/ablation_residual")
ARMS = ("residual_on", "residual_off", "classical_only")
CONTROLS = ("zero_command", base.ORACLE)
ALL_ARMS = CONTROLS + ARMS
ARM_DEFINITIONS = {
    "residual_on": "clip(physics_baseline + export_delta); shipped all-fix v5 arm",
    "residual_off": "clip(export_delta); classical superposition removed",
    "classical_only": (
        "clip(physics_baseline); learned correction removed = ResidualPolicy(enabled=False)"
    ),
    "zero_command": "no TSV; Simulator::run_episode ControlMode::None; paired gate control",
    "oracle_gtheta": "run_episode_with_oracle / fin_accel_command; positive control",
}


def rel(path: Path) -> str:
    """Printable path for repo-relative or external --out-* locations."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def verify_inputs() -> tuple[dict, dict, dict, list[int], dict]:
    """Loud arm-identity checks. No substitute export, split or physics is allowed."""
    for path in (
        base.HELPER,
        base.CONFIG,
        base.HELDOUT,
        base.MANIFEST,
        base.EXPORT_MANIFEST,
        base.SUMMARY,
        base.SHIELD,
        base.QUALITY,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing {path.relative_to(ROOT)}")
    config = tomllib.loads(base.CONFIG.read_text())
    manifest = json.loads(base.MANIFEST.read_text())
    exp_manifest = json.loads(base.EXPORT_MANIFEST.read_text())
    summary = json.loads(base.SUMMARY.read_text())
    shield = json.loads(base.SHIELD.read_text())

    heldout = read_validation(base.HELDOUT)
    indices = [int(ep.index) for ep in heldout.episodes]
    frames = int(manifest["frames_per_episode"])
    if len(indices) != 20 or any(i % 20 not in (16, 17, 18, 19) for i in indices):
        raise ValueError(f"unexpected held-out episodes: {indices}")
    if any(ep.frames.shape[0] != frames for ep in heldout.episodes):
        raise ValueError("held-out frame count differs from manifest")
    if manifest.get("split") != "held-out" or base.sha256(base.HELDOUT) != next(
        f["sha256"] for f in manifest["files"] if f["name"] == "validation.bin"
    ):
        raise ValueError("held-out manifest split or file hash mismatch")
    if tuple(exp_manifest["stage_order"]) != base.STAGES:
        raise ValueError(f"exports are not the four-stage default: {exp_manifest['stage_order']}")
    if "no win is claimed" not in exp_manifest.get("claim", ""):
        raise ValueError("export manifest no longer disclaims a win")
    if summary.get("ab_win_claimed") is not False or summary.get("rlvr_used") is not False:
        raise ValueError("allfix summary claims a win or RLVR; arm identity changed")
    if tuple(summary.get("stage_order", ())) != base.STAGES:
        raise ValueError("allfix summary stage_order disagrees with the four-stage default")
    if exp_manifest.get("fixes", {}).get("command_bound") != base.BOUND_MPS2:
        raise ValueError("allfix export manifest no longer pins the +/-10 command bound")
    if shield.get("authority") != base.BOUND_MPS2:
        raise ValueError("allfix shield authority disagrees with the command bound")
    if summary.get("shield_verification", {}).get("counts", {}).get("accepted") != 1:
        raise ValueError("allfix shield counted accept case missing")
    frozen = summary.get("data_provenance", {})
    if frozen.get("heldout_sha256") != base.sha256(base.HELDOUT):
        raise ValueError("allfix summary heldout hash disagrees with the flown held-out set")
    if frozen.get("config_sha256") != base.sha256(base.CONFIG):
        raise ValueError("allfix summary config hash disagrees with the frozen sim config")

    metrics_by_seed: dict[str, dict] = {}
    for seed in base.SEEDS:
        export = base.EXPORTS / f"teacher_v2_s{seed}.pt"
        metrics_path = base.RUNS / f"s{seed}" / "metrics.json"
        declared = exp_manifest["students"].get(str(seed))
        if not export.is_file() or not metrics_path.is_file() or declared is None:
            raise FileNotFoundError(f"seed {seed}: missing export/metrics; no stand-in substituted")
        if base.sha256(export) != declared["sha256"]:
            raise ValueError(f"seed {seed}: export sha256 disagrees with export manifest")
        metrics = json.loads(metrics_path.read_text())
        if metrics.get("config_hash") != declared["config_hash"]:
            raise ValueError(f"seed {seed}: metrics config_hash disagrees with export manifest")
        if metrics.get("seed") != seed:
            raise ValueError(f"seed {seed}: metrics seed field mismatch")
        if declared.get("file") != export.name:
            raise ValueError(f"seed {seed}: export manifest names a different student file")
        if metrics.get("ece_source") != declared.get("ece_source"):
            raise ValueError(f"seed {seed}: metrics ece_source disagrees with export manifest")
        if metrics.get("rmse_held_out") != declared.get("rmse_held_out"):
            raise ValueError(f"seed {seed}: metrics held-out RMSE disagrees with export manifest")
        if metrics.get("allfix_provenance") is not True:
            raise ValueError(f"seed {seed}: metrics do not carry allfix_provenance")
        if metrics.get("student_command_bound") != base.BOUND_MPS2:
            raise ValueError(f"seed {seed}: student command bound is not the allfix +/-10")
        if metrics.get("legacy_student_bound") is not False:
            raise ValueError(f"seed {seed}: legacy student bound is still enabled")
        if metrics.get("residual_policy", {}).get("enabled") is not True:
            raise ValueError(f"seed {seed}: SHIELD02 residual policy is not enabled")
        stages = metrics.get("stages", {})
        if stages.get("distill", {}).get("details", {}).get("residual_policy_target") is not True:
            raise ValueError(f"seed {seed}: distill target is not the residual decision")
        if (
            stages.get("dagger", {}).get("details", {}).get("rollout_source")
            != "on_policy_student:gdn2_student"
        ):
            raise ValueError(f"seed {seed}: DAgger rollout is not the AIFIX3 on-policy student")
        if stages.get("privileged", {}).get("details", {}).get("privileged_batches") != 2300:
            raise ValueError(f"seed {seed}: AIFIX2 privileged batches are missing")
        if metrics.get("ece") is None or metrics.get("ece_held_out") is None:
            raise ValueError(f"seed {seed}: ECE fields missing; no gate value invented")
        metrics_by_seed[str(seed)] = metrics
    return config, manifest, summary, indices, metrics_by_seed


def make_commands(baseline, prediction: dict, arm: str, out_dir: Path) -> dict:
    """Write one TSV per episode; identical gate and clamp for every arm.

    This mirrors ab_v5.build_controls bit-for-bit for ``residual_on`` (including
    its 6-decimal text format), so the shipped v5 flight is a fresh cross-check.
    """
    raw_base = np.stack(
        [
            physics_baseline(torch.as_tensor(ep.frames[:, :SMART_FEATURES], dtype=torch.float32))
            .numpy()
            .astype(np.float64)
            for ep in baseline.episodes
        ]
    )
    delta = prediction["deltas"]
    if arm == "residual_on":
        raw = raw_base + delta
    elif arm == "residual_off":
        raw = delta
    elif arm == "classical_only":
        raw = raw_base
    else:
        raise ValueError(f"unknown arm {arm}")
    healthy = np.stack([np.asarray(ep.targets[:, 3], np.float64) for ep in baseline.episodes])
    if healthy.shape != raw.shape[:2]:
        raise ValueError(f"ai_healthy shape {healthy.shape} differs from command {raw.shape[:2]}")
    gate = healthy > 0.5
    command = np.clip(np.where(gate[:, :, None], raw, 0.0), -base.COMMAND_LIMIT, base.COMMAND_LIMIT)
    stats = {
        "rows": int(command.shape[0] * command.shape[1]),
        "authority_gate_open_ticks": int(gate.sum()),
        "authority_gate_closed_ticks": int((~gate).sum()),
        "interventions": int((np.max(np.abs(command), axis=2) > 1e-9).sum()),
        "clipped_ticks": int((np.abs(raw) > base.COMMAND_LIMIT).any(axis=2).sum()),
        "max_command_mps2": float(np.max(np.abs(command))),
        "mean_abs_correction_mps2": float(np.mean(np.abs(raw))),
        "mean_confidence": float(np.mean(prediction["confidences"])),
        "confidence_gate": "none (residual decision target is ungated; seam clips only)",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for e, ep in enumerate(baseline.episodes):
        with (out_dir / f"{ep.index}.tsv").open("w") as handle:
            handle.write("# frame_index g_roll_mps2 g_pitch_mps2\n")
            for tick, (roll, pitch) in enumerate(command[e]):
                handle.write(f"{tick} {roll:.6f} {pitch:.6f}\n")
    return stats


def fidelity(report: dict, tolerance: float = 1e-6) -> dict:
    """Fresh residual_on flights must reproduce the shipped v5 numbers exactly."""
    shipped_path = SCRIPTS / "ab_report_v5.json"
    out: dict = {"checked": False, "passed": False, "detail": ""}
    if not shipped_path.is_file():
        out["detail"] = "scripts/ab_report_v5.json missing; no comparison made"
        return out
    shipped = json.loads(shipped_path.read_text())
    out["checked"] = True
    problems: list[str] = []
    for seed in base.SEEDS:
        try:
            ref = shipped["seeds"][str(seed)]["teacher_v2"]["aggregate"]["assisted"]
        except (KeyError, TypeError) as exc:
            problems.append(f"seed {seed}: v5 reference missing ({exc})")
            continue
        got = report["seeds"][str(seed)]["residual_on"]["aggregate"]["assisted"]
        for metric in ("apogee_m", "lateral_rms_m", "deployment_lateral_m"):
            a, b = float(got[metric]), float(ref[metric])
            if abs(a - b) > tolerance * max(1.0, abs(b)):
                problems.append(f"seed {seed} {metric}: {a!r} != v5 {b!r}")
    out["passed"] = not problems
    out["detail"] = (
        "residual_on reproduces scripts/ab_report_v5.json assisted aggregates"
        if not problems
        else "; ".join(problems)
    )
    return out


def markdown(report: dict) -> str:
    out = [
        "# Residual-policy deployment ablation — unweakened gates, FINAL all-fix exports",
        "",
        (
            f"Verdict: **{report['verdict']['status']}** (exit {report['exit_code']}). "
            "Simulator measurement only; no win is claimed unless every gate passes."
        ),
        "",
        "## Arms (deployment composition only; no retraining, no tuning)",
        "",
        "| arm | command written to the frozen seam |",
        "|---|---|",
    ]
    for name in ALL_ARMS:
        out.append(f"| {name} | {ARM_DEFINITIONS[name]} |")
    out += [
        "",
        "## Model quality (exact values from allfix metrics.json)",
        "",
        "| seed | rmse_train | rmse_held_out | ratio | ece_held_out | coverage | residual_gain |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in report["seed_order"]:
        q = report["model_quality"][str(seed)]
        out.append(
            f"| {seed} | {q['rmse_train']!r} | {q['rmse_held_out']!r} | {q['ratio']!r} "
            f"| {q['ece_held_out']!r} | {q['conformal_coverage']!r} | {q['residual_gain']!r} |"
        )
    out += [
        "",
        "## Flight metrics (metres; baseline -> assisted, pooled over 20 held-out episodes)",
        "",
        "| seed | arm | deployment lateral | apogee | whole-trajectory lateral RMS |",
        "|---:|---|---:|---:|---:|",
    ]
    for seed in report["seed_order"]:
        for arm in ARMS:
            agg = report["seeds"][str(seed)][arm]["aggregate"]
            out.append(
                f"| {seed} | {arm} "
                f"| {agg['baseline']['deployment_lateral_m']:.6f} -> "
                f"{agg['assisted']['deployment_lateral_m']:.6f} "
                f"| {agg['baseline']['apogee_m']:.6f} -> {agg['assisted']['apogee_m']:.6f} "
                f"| {agg['baseline']['lateral_rms_m']:.6f} -> "
                f"{agg['assisted']['lateral_rms_m']:.6f} |"
            )
    out += ["", "## Improvement vs paired zero command (whole-trajectory RMS)", ""]
    out += ["| seed | " + " | ".join(ARMS) + " |", "|---:|" + "---:|" * len(ARMS)]
    for seed in report["seed_order"]:
        cells = []
        for arm in ARMS:
            agg = report["seeds"][str(seed)][arm]["aggregate"]
            lat = 100.0 * (
                agg["assisted"]["lateral_rms_m"] / agg["baseline"]["lateral_rms_m"] - 1.0
            )
            ap = 100.0 * (agg["assisted"]["apogee_m"] / agg["baseline"]["apogee_m"] - 1.0)
            cells.append(f"{lat:+.2f}% / {ap:+.2f}%")
        out.append(f"| {seed} | " + " | ".join(cells) + " |")
    out += [
        "",
        "## Gates (hard-coded, unweakened)",
        "",
        "| gate | result | detail |",
        "|---|---|---|",
    ]
    for row in report["gates"]:
        out.append(f"| {row['name']} | {'PASS' if row['passed'] else 'FAIL'} | {row['detail']} |")
    pc = report["positive_control"]
    out += ["", "## Positive control", "", f"- passed: **{pc['passed']}**"]
    for seed, row in pc["per_seed"].items():
        out.append(
            f"- seed {seed}: deployment lateral {row['deployment_delta_m']:+.6f} m, "
            f"apogee {row['apogee_delta_m']:+.6f} m, "
            f"actuator cap ticks {row['authority']['cap_hit_ticks']}/"
            f"{row['authority']['active_ticks']}"
        )
    fid = report["fidelity"]
    out += [
        "",
        "## Fidelity to the shipped v5 A/B",
        "",
        f"- checked: {fid['checked']}; passed: **{fid['passed']}**; {fid['detail']}",
        "",
        "## Provenance",
        "",
        "| field | value |",
        "|---|---|",
    ]
    for key, value in report["provenance"].items():
        text = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
        out.append(f"| {key} | `{text}` |")
    if report["blockers"]:
        out += ["", "## Blockers", ""]
        out += [f"- {b}" for b in report["blockers"]]
    out += ["", "## Honesty notes", ""]
    out += [f"- {n}" for n in report["honesty_notes"]]
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(base.SEEDS))
    ap.add_argument("--work-dir", type=Path, default=WORK)
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    ap.add_argument("--out-md", type=Path, default=OUT_MD)
    args = ap.parse_args()
    torch.set_num_threads(1)
    if sorted(args.seeds) != list(base.SEEDS):
        raise ValueError(f"pre-registered three-seed gate needs {list(base.SEEDS)}")

    config, manifest, summary, indices, metrics_by_seed = verify_inputs()
    sample_dt = 1.0 / config["run"]["frame_rate_hz"]
    seed_order = sorted(args.seeds)
    blockers: list[str] = []
    report_seeds: dict[str, dict] = {}
    baseline_shas: dict[str, str] = {}

    for seed in seed_order:
        work = args.work_dir / f"seed{seed}"
        baseline_dir = work / "zero_command"
        base.run_helper(seed, indices, baseline_dir)
        baseline = read_validation(baseline_dir / "validation.bin")
        baseline_track = base.read_track(baseline_dir / "validation.trajectory", sample_dt)
        if set(baseline_track) != set(indices):
            raise RuntimeError(f"seed {seed}: baseline episode set differs from held-out manifest")
        baseline_shas[str(seed)] = base.sha256(baseline_dir / "validation.trajectory")
        if seed == 1:
            frozen = next(
                f["sha256"] for f in manifest["files"] if f["name"] == "validation.trajectory"
            )
            if baseline_shas["1"] != frozen:
                raise RuntimeError(
                    "seed 1 zero-command flight is not bit-identical to the frozen "
                    "post-FIXPUMP trajectory: the helper is not the corrected binary"
                )
            if base.sha256(baseline_dir / "validation.bin") != base.sha256(base.HELDOUT):
                raise RuntimeError("seed 1 baseline validation.bin differs from frozen held-out")
        report_seeds[str(seed)] = {
            "zero_command": {
                "arm": ARM_DEFINITIONS["zero_command"],
                "track": baseline_track,
                "aggregate": base.summarise(baseline_track, baseline_track, None)["aggregate"],
                "sha256": {"assisted": baseline_shas[str(seed)]},
            }
        }

        oracle_dir = work / base.ORACLE
        base.run_helper(seed, indices, oracle_dir, oracle=True)
        oracle_track = base.read_track(oracle_dir / "validation.trajectory", sample_dt)
        if set(oracle_track) != set(baseline_track):
            raise RuntimeError(f"seed {seed}: oracle episode set differs")
        oracle_summary = base.summarise(baseline_track, oracle_track, None)
        oracle_summary["arm"] = ARM_DEFINITIONS[base.ORACLE]
        oracle_summary["authority"] = base.oracle_authority(oracle_dir / "validation.trajectory")
        oracle_summary["sha256"] = {
            "baseline": baseline_shas[str(seed)],
            "assisted": base.sha256(oracle_dir / "validation.trajectory"),
        }
        report_seeds[str(seed)][base.ORACLE] = oracle_summary

        export = base.EXPORTS / f"teacher_v2_s{seed}.pt"
        checkpoint = torch.load(export, map_location="cpu", weights_only=True)
        model = base.GDN2Student(base.GDN2StudentConfig(seed=seed))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        prediction = base.commands_for_export(model, baseline.episodes)
        for arm in ARMS:
            stats = make_commands(baseline, prediction, arm, work / arm / "commands")
            flown_dir = work / arm / "flown"
            base.run_helper(seed, indices, flown_dir, commands_dir=work / arm / "commands")
            flown_track = base.read_track(flown_dir / "validation.trajectory", sample_dt)
            if set(flown_track) != set(baseline_track):
                raise RuntimeError(f"seed {seed} {arm}: episode set differs")
            for idx in indices:
                if flown_track[idx]["ticks"] != baseline_track[idx]["ticks"]:
                    raise RuntimeError(f"seed {seed} {arm}: tick count changed at {idx}")
            summary_entry = base.summarise(baseline_track, flown_track, stats)
            summary_entry["arm"] = ARM_DEFINITIONS[arm]
            summary_entry["sha256"] = {
                "baseline": baseline_shas[str(seed)],
                "assisted": base.sha256(flown_dir / "validation.trajectory"),
            }
            summary_entry["export_sha256"] = base.sha256(export)
            report_seeds[str(seed)][arm] = summary_entry
        del model
        agg = report_seeds[str(seed)]["residual_on"]["aggregate"]
        print(
            f"seed {seed}: residual_on deployment lateral "
            f"{agg['baseline']['deployment_lateral_m']:.2f} -> "
            f"{agg['assisted']['deployment_lateral_m']:.2f} m; apogee "
            f"{agg['baseline']['apogee_m']:.2f} -> {agg['assisted']['apogee_m']:.2f} m",
            flush=True,
        )

    report: dict = {
        "measurement": "POTOKTRJ v3 flown-truth trajectory sidecar (stride 26), not surrogate",
        "sim_source": (
            "read-only scripts/ab_control_v3 on git "
            + base.git_head()
            + "; oracle uses run_episode_with_oracle / fin_accel_command"
        ),
        "supersedes": (
            "any win reading of the RETRAIN-ALLFIX surrogate RMSE comparison: it was never "
            "a closed-loop A/B and is void here; the unweakened pre-registered gates are "
            "restored verbatim"
        ),
        "held_out_indices": indices,
        "seed_order": seed_order,
        "students": [{"name": name, "definition": ARM_DEFINITIONS[name]} for name in ALL_ARMS],
        "verdict": {"status": "PENDING", "primary": "residual_on"},
        "seeds": report_seeds,
        "arms": ARM_DEFINITIONS,
        "model_quality": {
            str(seed): {
                "rmse_train": metrics_by_seed[str(seed)]["rmse_train"],
                "rmse_held_out": metrics_by_seed[str(seed)]["rmse_held_out"],
                "ratio": metrics_by_seed[str(seed)]["ratio"],
                "ece_held_out": metrics_by_seed[str(seed)]["ece_held_out"],
                "conformal_coverage": metrics_by_seed[str(seed)]["conformal_coverage"],
                "residual_gain": metrics_by_seed[str(seed)].get("residual_policy", {}).get("gain"),
            }
            for seed in seed_order
        },
    }
    report["fidelity"] = fidelity(report)
    if not report["fidelity"]["passed"]:
        blockers.append(
            "fidelity: residual_on does not reproduce v5: " + report["fidelity"]["detail"]
        )

    provenance = {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_head": base.git_head(),
        "config": str(base.CONFIG.relative_to(ROOT)),
        "config_sha256": base.sha256(base.CONFIG),
        "config_hash": base.sha256(base.CONFIG),
        "heldout": str(base.HELDOUT.relative_to(ROOT)),
        "heldout_sha256": base.sha256(base.HELDOUT),
        "heldout_manifest_sha256": base.sha256(base.MANIFEST),
        "quality_report_sha256": base.sha256(base.QUALITY),
        "allfix_root": str(base.ALLFIX.relative_to(ROOT)),
        "export_manifest_sha256": base.sha256(base.EXPORT_MANIFEST),
        "summary_sha256": base.sha256(base.SUMMARY),
        "shield_sha256": base.sha256(base.SHIELD),
        "exports_sha256": {
            f"teacher_v2_s{seed}.pt": base.sha256(base.EXPORTS / f"teacher_v2_s{seed}.pt")
            for seed in seed_order
        },
        "metrics_sha256": {
            f"s{seed}/metrics.json": base.sha256(base.RUNS / f"s{seed}" / "metrics.json")
            for seed in seed_order
        },
        "metrics_config_hash": {
            str(seed): metrics_by_seed[str(seed)]["config_hash"] for seed in metrics_by_seed
        },
        "ece_source": {
            str(seed): metrics_by_seed[str(seed)]["ece_source"] for seed in metrics_by_seed
        },
        "ece_held_out": {
            str(seed): metrics_by_seed[str(seed)]["ece_held_out"] for seed in metrics_by_seed
        },
        "conformal_coverage": {
            str(seed): metrics_by_seed[str(seed)]["conformal_coverage"] for seed in metrics_by_seed
        },
        "allfix_provenance": {
            str(seed): metrics_by_seed[str(seed)]["allfix_provenance"] for seed in metrics_by_seed
        },
        "residual_gain": {
            str(seed): metrics_by_seed[str(seed)].get("residual_policy", {}).get("gain")
            for seed in metrics_by_seed
        },
        "residual_policy_enabled": {
            str(seed): metrics_by_seed[str(seed)]["residual_policy"]["enabled"]
            for seed in metrics_by_seed
        },
        "stage_order": list(base.STAGES),
        "summary_ab_win_claimed": summary.get("ab_win_claimed"),
        "summary_rlvr_used": summary.get("rlvr_used"),
        "allfix_data_provenance": summary.get("data_provenance", {}),
        "student_command_bound": base.BOUND_MPS2,
        "legacy_student_bound": False,
        "arm_definitions": ARM_DEFINITIONS,
        "source_of_off_semantics": (
            "residual_policy.py:139-149 (gain*correction, enabled=False -> baseline.clone()); "
            "physics.py:53-56 final_command(residual=True) = baseline + prediction"
        ),
        "harness": "scripts/ablation_residual.py",
        "harness_sha256": base.sha256(Path(__file__)),
        "helper": str(base.HELPER_SRC.relative_to(ROOT)),
        "helper_sha256": base.sha256(base.HELPER_SRC),
        "helper_binary": str(base.HELPER.relative_to(ROOT)),
        "helper_binary_sha256": base.sha256(base.HELPER),
        "dynamics_sha256": base.sha256(base.DYNAMICS_SRC),
        "v5_harness_sha256": base.sha256(SCRIPTS / "ab_v5.py"),
        "v5_report_sha256": base.sha256(SCRIPTS / "ab_report_v5.json"),
        "baseline_trajectory_sha256": baseline_shas,
        "assisted_trajectory_sha256": {
            f"{seed}/{arm}": report_seeds[str(seed)][arm]["sha256"]["assisted"]
            for seed in seed_order
            for arm in ALL_ARMS
        },
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "sample_dt_s": sample_dt,
        "command_limit_mps2": base.COMMAND_LIMIT,
        "deployment_proxy": "first DESCENT tick (vz<0 and z>=5), radial distance from pad",
        "lateral_limit_m": base.LATERAL_LIMIT_M,
        "apogee_limit_m": base.APOGEE_LIMIT_M,
    }
    report["provenance"] = provenance

    # Verbatim EVAL01 suite on the primary arm; no threshold is copied or relaxed.
    evals: dict[str, bool] = {}
    for name, fn in (
        ("rmse_ratio", teacher_gates.check_ratio),
        ("ece_source", teacher_gates.check_ece_source),
        ("ece", teacher_gates.check_ece),
        ("provenance", teacher_gates.check_provenance),
    ):
        try:
            result = fn(report) if name == "provenance" else fn(metrics_by_seed)
        except teacher_gates.ProvenanceError as exc:
            blockers.append(f"EVAL01 {name}: {exc}")
            evals[name] = False
            continue
        evals[name] = result.passed
        if not result.passed:
            blockers.append(f"EVAL01 {name} failed: {result.detail}")

    gates: list[dict] = [
        {
            "name": f"eval01 {name}",
            "passed": bool(value),
            "detail": "verbatim EVAL01 gate" if value else "see blockers",
        }
        for name, value in evals.items()
    ]

    for arm in ARMS:
        view = {**report, "verdict": {**report["verdict"], "primary": arm}}
        for name, fn in (
            ("seed_spread", teacher_gates.check_seed_spread),
            ("improvement", teacher_gates.check_improvement),
        ):
            try:
                result = fn(view, student=arm)
            except teacher_gates.ProvenanceError as exc:
                blockers.append(f"EVAL01 {name} ({arm}): {exc}")
                gates.append(
                    {"name": f"eval01 {name} [{arm}]", "passed": False, "detail": str(exc)}
                )
                continue
            gates.append(
                {
                    "name": f"eval01 {name} [{arm}]",
                    "passed": result.passed,
                    "detail": result.detail,
                }
            )
            if not result.passed:
                blockers.append(f"EVAL01 {name} ({arm}) failed: {result.detail}")

    oracle_check = teacher_gates.check_oracle(report)
    gates.append(
        {
            "name": "eval01 oracle_positive_control",
            "passed": oracle_check.passed,
            "detail": oracle_check.detail,
        }
    )
    if not oracle_check.passed:
        blockers.append(f"EVAL01 oracle_positive_control failed: {oracle_check.detail}")

    positive_control: dict = {"per_seed": {}, "passed": False}
    for seed in seed_order:
        entry = report_seeds[str(seed)][base.ORACLE]
        agg = entry["aggregate"]
        dep_delta = agg["deployment_lateral_m_delta"]
        ap_delta = agg["apogee_m_delta"]
        ok = dep_delta < -base.ULP_M and ap_delta > base.ULP_M
        positive_control["per_seed"][str(seed)] = {
            "deployment_delta_m": dep_delta,
            "apogee_delta_m": ap_delta,
            "authority": entry["authority"],
            "improved_both": ok,
        }
    positive_control["passed"] = bool(positive_control["per_seed"]) and all(
        row["improved_both"] for row in positive_control["per_seed"].values()
    )
    if not positive_control["passed"]:
        blockers.append("oracle positive control did not improve both metrics on all seeds")
    gates.append(
        {
            "name": "oracle_deployment_and_apogee_positive_control",
            "passed": positive_control["passed"],
            "detail": "both metrics improve against paired zero command on every seed",
        }
    )

    for arm in ARMS:
        for seed in seed_order:
            agg = report_seeds[str(seed)][arm]["aggregate"]
            dep = agg["assisted"]["deployment_lateral_m"]
            ap = agg["assisted"]["apogee_m"]
            passed = dep <= base.LATERAL_LIMIT_M and ap >= base.APOGEE_LIMIT_M
            gates.append(
                {
                    "name": f"absolute [{arm}] seed {seed}",
                    "passed": passed,
                    "detail": (
                        f"deployment lateral {dep:.6f} m <= {base.LATERAL_LIMIT_M:g} m: "
                        f"{dep <= base.LATERAL_LIMIT_M}; apogee {ap:.6f} m >= "
                        f"{base.APOGEE_LIMIT_M:g} m: {ap >= base.APOGEE_LIMIT_M}"
                    ),
                }
            )
            if not passed:
                blockers.append(f"absolute gate failed: {arm} seed {seed}")

    passed = all(row["passed"] for row in gates) and not blockers
    report["positive_control"] = positive_control
    report["gates"] = gates
    report["blockers"] = blockers
    report["verdict"] = {
        "status": "PASS" if passed else "FAIL",
        "primary": "residual_on",
        "pass": passed,
        "win_claimed": passed,
        "gates_failed": sum(1 for row in gates if not row["passed"]),
        "fidelity": report["fidelity"]["passed"],
    }
    report["exit_code"] = 0 if passed else 1
    report["honesty_notes"] = [
        (
            "Simulator-only: the POTOKTRJ sidecar is emitted by the same simulator that flies the "
            "episodes; it is not an independent flight-truth anchor."
        ),
        (
            "No canopy/deployment event is scored; the proxy is the first DESCENT tick and is not a "
            "landing-dispersion measurement."
        ),
        (
            "Commands are generated from the zero-command flight's recorded frames and replayed by "
            "frame index; after a command moves the vehicle, later frames differ from that flight "
            "while the phase/gate schedule stays paired. Commands are emitted only where the paired "
            "baseline ai_healthy target is true, so boost-phase flight stays identical to control."
        ),
        (
            "No export was retrained. residual_off removes the classical baseline at deployment and "
            "applies the residual-trained delta as an absolute command; classical_only removes the "
            "learned delta and applies physics_baseline; both are diagnostic interpretations of "
            "'residual OFF' and neither is presented as a shipped policy."
        ),
        (
            "Seeds 2/3 use seed-matched exports on regenerated held-out flights; a missing export is "
            "a loud blocker, never a stand-in."
        ),
        (
            "Primary arm = ai/out/final_v3_allfix/exports/teacher_v2_s{1,2,3}.pt; the harness refuses "
            "to run unless manifest, summary, shield and per-seed metrics still carry the all-fix "
            "evidence. No gate, weight, physics constant or threshold was changed."
        ),
        (
            "The helper binary was not rebuilt in this shared checkout; its seed-1 zero-command "
            "flight is checked bit-for-bit against the frozen post-FIXPUMP held-out sidecar."
        ),
        (
            "Exit 1 is the unweakened gate verdict, not a crash; the JSON and Markdown reports are "
            "written before exit. No win is claimed unless every gate passes."
        ),
    ]

    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.out_md.write_text(markdown(report))
    print(f"wrote {rel(args.out_json)} and {rel(args.out_md)}")
    print(
        f"verdict: {report['verdict']['status']} (exit {report['exit_code']}), "
        f"gates failed={report['verdict']['gates_failed']}, blockers={len(blockers)}"
    )
    sys.exit(report["exit_code"])


if __name__ == "__main__":
    main()
