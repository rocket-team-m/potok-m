#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""BRK02 closed-loop A/B of the --braking-teacher student, under unweakened v6 gates.

Flies the 20 held-out scenarios (cells 16..19) through the read-only v3 control
seam on post-FIXPUMP dynamics, paired against the zero-command control, with
``run_episode_with_oracle`` as the positive control. The measurement helpers and
every gate call are imported from scripts/ab_v6.py; no threshold, bound, gain,
physics constant or harness step is relaxed or reimplemented here.

Usage:
    ai/.venv/bin/python scripts/ab_braking.py
Work files live outside the repo (default /tmp/opencode/ab_braking).
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

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT / "ai" / "src"))
sys.path.insert(0, str(SCRIPTS))

import ab_v6
from potok_ai.gdn2_student import GDN2Student, GDN2StudentConfig

from potok_ai.data import read_validation

CONFIG = ab_v6.CONFIG
HELDOUT = ab_v6.HELDOUT
MANIFEST = ab_v6.MANIFEST
QUALITY = ab_v6.QUALITY
BRAKING = ROOT / "ai" / "out" / "braking"
SUMMARY = BRAKING / "summary.json"
EXPORT_MANIFEST = BRAKING / "exports" / "manifest.json"
WORK = Path("/tmp/opencode/ab_braking")
OUT_JSON = SCRIPTS / "ab_report_braking.json"
OUT_MD = SCRIPTS / "ab_report_braking.md"

ARM = "braking"
SEEDS = ab_v6.SEEDS
ORACLE = ab_v6.ORACLE
STAGES = ab_v6.STAGES
BOUND_MPS2 = ab_v6.BOUND_MPS2


def sha256(path: Path) -> str:
    return ab_v6.sha256(path)


def export_path(seed: int) -> Path:
    return BRAKING / "exports" / f"teacher_braking_s{seed}.pt"


def metrics_path(seed: int) -> Path:
    return BRAKING / "runs" / f"s{seed}" / "metrics.json"


def check_identity(seed_order: list[int]) -> tuple[dict[str, dict], dict, list[str]]:
    """Arm identity and evidence; a mismatch is a loud refusal, never a stand-in."""
    summary = json.loads(SUMMARY.read_text())
    manifest = json.loads(EXPORT_MANIFEST.read_text())
    config_sha = sha256(CONFIG)
    heldout_sha = sha256(HELDOUT)
    dp = summary.get("data_provenance", {})
    if summary.get("ab_win_claimed") is not False:
        raise ValueError("braking summary claims an A/B win; exports are not artifacts")
    if summary.get("arm") != ARM or manifest.get("arm") != ARM:
        raise ValueError("braking summary/manifest arm identity is not the braking retrain")
    if summary.get("rlvr_used") is not False:
        raise ValueError("braking summary says RLVR was used; arm identity changed")
    if tuple(summary.get("stage_order", ())) != STAGES:
        raise ValueError("braking summary stage_order disagrees with the four-stage default")
    if tuple(manifest.get("stage_order", ())) != STAGES:
        raise ValueError("braking export manifest stage_order disagrees with the four-stage default")
    if dp.get("config_sha256") != config_sha or dp.get("heldout_sha256") != heldout_sha:
        raise ValueError("braking data provenance disagrees with the flown config/held-out")
    brake = summary.get("braking", {})
    if brake.get("enabled") is not True or brake.get("source") != "radial_braking_reference":
        raise ValueError("braking summary does not record the BRK01 reference as the target")
    if brake.get("flag") != "--braking-teacher" or brake.get("cap_mps2") != BOUND_MPS2:
        raise ValueError("braking summary does not pin the flag or the +/-10 command bound")
    if manifest.get("fixes", {}).get("command_bound") != BOUND_MPS2:
        raise ValueError("braking export manifest no longer pins the +/-10 command bound")
    if "no win is claimed" not in manifest.get("claim", ""):
        raise ValueError("braking export manifest no longer disclaims a win")

    blockers: list[str] = []
    metrics_by_seed: dict[str, dict] = {}
    evidence: dict = {
        "arm": ARM,
        "exports_sha256": {},
        "metrics_sha256": {},
        "metrics_config_hash": {},
        "ece_source": {},
        "ece_held_out": {},
        "conformal_coverage": {},
        "rmse_held_out": {},
        "braking_flags": {},
        "stage_order": list(STAGES),
        "command_bound": BOUND_MPS2,
        "summary_sha256": sha256(SUMMARY),
        "export_manifest_sha256": sha256(EXPORT_MANIFEST),
        "data_provenance": dp,
        "summary_braking": brake,
    }
    for seed in seed_order:
        export, metrics_file = export_path(seed), metrics_path(seed)
        if not export.is_file() or not metrics_file.is_file():
            blockers.append(f"{ARM} seed {seed}: missing export/metrics; no stand-in substituted")
            continue
        metrics = json.loads(metrics_file.read_text())
        export_sha = sha256(export)
        where = f"{ARM} seed {seed}"
        if metrics.get("seed") != seed or metrics.get("braking_arm") is not True:
            raise ValueError(f"{where}: metrics do not identify the braking arm/seed")
        if metrics.get("ece_source") != "held-out" or metrics.get("ece") != metrics.get(
            "ece_held_out"
        ):
            raise ValueError(f"{where}: ECE is not the disjoint held-out score")
        if not 0.0 < metrics.get("conformal_coverage", 0.0) <= 1.0:
            raise ValueError(f"{where}: conformal coverage outside (0, 1]")
        if metrics.get("student_command_bound") != BOUND_MPS2:
            raise ValueError(f"{where}: student command bound is not +/-10")
        if metrics.get("legacy_student_bound") is not False:
            raise ValueError(f"{where}: legacy student bound is still enabled")
        rp = metrics.get("residual_policy", {})
        if rp.get("enabled") is not True or rp.get("source") != "residual_policy":
            raise ValueError(f"{where}: SHIELD02 residual policy is not enabled")
        stages = metrics.get("stages", {})
        if any(stage not in stages for stage in STAGES):
            raise ValueError(f"{where}: a four-stage record is missing")
        privileged = stages["privileged"]["details"]
        dagger = stages["dagger"]["details"]
        distill = stages["distill"]["details"]
        objective = privileged.get("objective", {})
        if objective.get("source") != "impact" or objective.get("legacy") is not False:
            raise ValueError(f"{where}: AIOBJ impact objective is not the default")
        if privileged.get("privileged_batches", 0) <= 0 or privileged.get(
            "eval_label_blind"
        ) is not True:
            raise ValueError(f"{where}: AIFIX2 privileged evidence missing")
        if dagger.get("rollout_source") != "on_policy_student:gdn2_student":
            raise ValueError(f"{where}: DAgger rollout is not the on-policy student")
        if dagger.get("student_induced_rows", 0) <= 0:
            raise ValueError(f"{where}: DAgger on-policy evidence missing")
        if distill.get("residual_policy_target") is not True:
            raise ValueError(f"{where}: distill target is not the residual decision")
        if distill.get("student_source") != "gdn2_student":
            raise ValueError(f"{where}: distill student source changed")
        if metrics.get("data_provenance") != dp:
            raise ValueError(f"{where}: data provenance disagrees with the braking summary")
        if metrics.get("fixes", {}).get("braking_teacher") != "explicit --braking-teacher":
            raise ValueError(f"{where}: fixes do not record the explicit braking flag")
        for stage in STAGES:
            if "adversarial" in stages[stage]["details"]:
                raise ValueError(f"{where}: braking arm carries adversarial stage records")
        for name in ("privileged", "dagger"):
            arm_brake = stages[name]["details"].get("braking", {})
            if arm_brake.get("enabled") is not True or arm_brake.get(
                "source"
            ) != "radial_braking_reference":
                raise ValueError(f"{where}: {name} stage did not use the BRK01 reference")
            if not arm_brake.get("reason_counts"):
                raise ValueError(f"{where}: {name} stage has no counted braking reasons")
        declared = manifest["students"].get(str(seed))
        if declared is None or declared.get("file") != export.name:
            raise ValueError(f"{where}: export manifest has no matching student record")
        if export_sha != declared["sha256"]:
            raise ValueError(f"{where}: export sha256 disagrees with export manifest")
        if metrics.get("config_hash") != declared.get("config_hash"):
            raise ValueError(f"{where}: metrics config_hash disagrees with export manifest")
        if metrics.get("rmse_held_out") != declared.get("rmse_held_out"):
            raise ValueError(f"{where}: metrics RMSE disagrees with export manifest")
        if summary.get("rmse_held_out_by_seed", {}).get(str(seed)) != metrics.get("rmse_held_out"):
            raise ValueError(f"{where}: held-out RMSE disagrees with the braking summary")

        metrics_by_seed[str(seed)] = metrics
        evidence["exports_sha256"][str(seed)] = export_sha
        evidence["metrics_sha256"][str(seed)] = sha256(metrics_file)
        evidence["metrics_config_hash"][str(seed)] = metrics["config_hash"]
        evidence["ece_source"][str(seed)] = metrics["ece_source"]
        evidence["ece_held_out"][str(seed)] = metrics["ece_held_out"]
        evidence["conformal_coverage"][str(seed)] = metrics["conformal_coverage"]
        evidence["rmse_held_out"][str(seed)] = metrics["rmse_held_out"]
        evidence["braking_flags"][str(seed)] = metrics["braking_provenance"]
    return metrics_by_seed, evidence, blockers


def arm_report(indices: list[int], seed_order: list[int], seed_results: dict, evidence: dict, provenance: dict) -> dict:
    return {
        "measurement": "POTOKTRJ v3 flown-truth trajectory sidecar (stride 26), not surrogate",
        "sim_source": (
            "read-only scripts/ab_control_v3 on git "
            + provenance["git_head"]
            + "; oracle uses run_episode_with_oracle / fin_accel_command"
        ),
        "supersedes": (
            "any win reading of the RETRAIN-NIGHT server-side RMSE comparison or of an "
            "RMSE-only braking comparison: neither is a closed-loop A/B and both are void "
            "here; the unweakened pre-registered v6 gates are restored"
        ),
        "held_out_indices": indices,
        "seed_order": seed_order,
        "students": [
            {
                "name": ARM,
                "label": "BRK02 --braking-teacher four-stage distill student",
                "stage_order": list(STAGES),
                "exports_sha256": evidence["exports_sha256"],
            }
        ],
        "verdict": {"status": "PENDING", "primary": ARM},
        "provenance": {**provenance, "arm": ARM, **evidence},
        "seeds": {
            str(seed): {
                name: seed_results[str(seed)][name]
                for name in (ORACLE, ARM)
                if name in seed_results.get(str(seed), {})
            }
            for seed in seed_order
        },
    }


def markdown(report: dict) -> str:
    entry = report["arms"][ARM]
    verdict = entry["verdict"]
    out = [
        "# BRK02 closed-loop A/B — braking-teacher student under unweakened v6 gates",
        "",
        (
            f"Arm verdict: **{verdict['status']}** (exit {report['exit_code']}). "
            "The control is the paired zero-command flight; the oracle is the positive "
            "control. Every v6 gate is called verbatim from `scripts/ab_v6.py`. "
            "This is a post-FIXPUMP simulator measurement, not a flight claim."
        ),
        "",
        "| gate | result | detail |",
        "|---|---|---|",
    ]
    for row in entry["gates"]:
        out.append(f"| {row['name']} | {'PASS' if row['passed'] else 'FAIL'} | {row['detail']} |")
    out += [
        "",
        "| seed | arm | deployment lateral (m) | apogee (m) | whole-trajectory lateral RMS (m) |",
        "|---:|---|---:|---:|---:|",
    ]
    for seed in report["seed_order"]:
        for name in (ORACLE, ARM):
            model = entry["seeds"].get(str(seed), {}).get(name)
            if model is None:
                continue
            agg = model["aggregate"]
            out.append(
                f"| {seed} | {name} "
                f"| {agg['baseline']['deployment_lateral_m']:.2f} -> "
                f"{agg['assisted']['deployment_lateral_m']:.2f} "
                f"| {agg['baseline']['apogee_m']:.2f} -> {agg['assisted']['apogee_m']:.2f} "
                f"| {agg['baseline']['lateral_rms_m']:.2f} -> "
                f"{agg['assisted']['lateral_rms_m']:.2f} |"
            )
    pc = entry["positive_control"]
    out += ["", f"Positive control passed: **{pc['passed']}**", ""]
    for seed, row in pc["per_seed"].items():
        out.append(
            f"- seed {seed}: deployment lateral {row['deployment_delta_m']:+.2f} m, "
            f"apogee {row['apogee_delta_m']:+.2f} m, cap ticks "
            f"{row['authority']['cap_hit_ticks']}/{row['authority']['active_ticks']}"
        )
    if entry["blockers"]:
        out += ["", "Blockers:", ""] + [f"- {b}" for b in entry["blockers"]]
    out += [
        "",
        "## Verdict",
        "",
        (
            f"`pass`={verdict['pass']}, `win_claimed`={verdict['win_claimed']}, "
            f"gates failed={verdict['gates_failed']}. "
            + (
                "A win is claimed only because every unweakened gate passed."
                if verdict["pass"]
                else "A measured FAIL: no win is claimed and no gate was weakened."
            )
        ),
        "",
        "## Provenance",
        "",
        "| field | value |",
        "|---|---|",
    ]
    for key in (
        "git_head",
        "harness",
        "harness_sha256",
        "objective_source_sha256",
        "objective_version",
        "objective_head",
        "config_sha256",
        "heldout_sha256",
        "summary_sha256",
        "export_manifest_sha256",
        "exports_sha256",
        "command_limit_mps2",
        "deployment_proxy",
        "lateral_limit_m",
        "apogee_limit_m",
    ):
        value = report["provenance"].get(key)
        text = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
        out.append(f"| {key} | `{text}` |")
    out += ["", "## Honesty notes", ""] + [f"- {n}" for n in report["honesty_notes"]]
    return "\n".join(out) + "\n"


def write_doc(report: dict) -> None:
    """Record the exact verdict beside the training results (marked section)."""
    doc = ROOT / "ai" / "docs" / "TRAINING_RESULTS.md"
    entry = report["arms"][ARM]
    verdict = entry["verdict"]
    gate_rows = "\n".join(
        f"| {row['name']} | {'PASS' if row['passed'] else 'FAIL'} |"
        for row in entry["gates"]
    )
    blockers = "\n".join(f"- {b}" for b in entry["blockers"]) or "- none"
    section = (
        "<!-- braking-ab:begin -->\n"
        "# BRK02 closed-loop A/B — exact verdict\n\n"
        f"**{verdict['status']}** — {verdict['gates_failed']} gates failed, "
        f"`win_claimed={verdict['win_claimed']}`. "
        "Full evidence: `scripts/ab_report_braking.json` and `ab_report_braking.md`.\n\n"
        "| gate | result |\n|---|---|\n"
        f"{gate_rows}\n\n"
        f"Blockers:\n{blockers}\n\n"
        "No gate, threshold, physics value or command envelope was changed. "
        "Two counted limitations accompany this verdict: the BRK01 privileged "
        "braking target was 100% boost-gated (DAgger was active), and the helper "
        f"rebuilt from current HEAD shifted input frames by "
        f"{report['provenance']['baseline_frame_maxdiff_vs_frozen']:.6f} while the flown "
        "sidecar stayed bit-identical. Neither is a gate edit.\n"
        "<!-- braking-ab:end -->\n"
    )
    existing = doc.read_text(encoding="utf-8") if doc.is_file() else ""
    start, end = "<!-- braking-ab:begin -->", "<!-- braking-ab:end -->"
    if start in existing and end in existing:
        left = existing[: existing.index(start)]
        right = existing[existing.index(end) + len(end) :]
        doc.write_text(left + section + right, encoding="utf-8")
    else:
        doc.write_text(section + "\n---\n\n" + existing, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--work-dir", type=Path, default=WORK)
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    ap.add_argument("--out-md", type=Path, default=OUT_MD)
    args = ap.parse_args()
    torch.set_num_threads(1)
    for path in (
        ab_v6.HELPER,
        CONFIG,
        HELDOUT,
        MANIFEST,
        QUALITY,
        SUMMARY,
        EXPORT_MANIFEST,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing {path.relative_to(ROOT)}")

    config = tomllib.loads(CONFIG.read_text())
    sample_dt = 1.0 / config["run"]["frame_rate_hz"]
    heldout_manifest = json.loads(MANIFEST.read_text())
    heldout = read_validation(HELDOUT)
    indices = [int(ep.index) for ep in heldout.episodes]
    frames_per_episode = int(heldout_manifest["frames_per_episode"])
    if len(indices) != 20 or any(i % 20 not in (16, 17, 18, 19) for i in indices):
        raise ValueError(f"unexpected held-out indices: {indices}")
    if any(ep.frames.shape[0] != frames_per_episode for ep in heldout.episodes):
        raise ValueError("held-out frame count differs from manifest")
    if heldout_manifest.get("split") != "held-out" or sha256(HELDOUT) != next(
        f["sha256"] for f in heldout_manifest["files"] if f["name"] == "validation.bin"
    ):
        raise ValueError("held-out manifest split or file hash mismatch")

    seed_order = sorted(args.seeds)
    if seed_order != list(SEEDS):
        raise ValueError(f"pre-registered gate needs seeds {list(SEEDS)}, got {seed_order}")
    metrics_by_seed, evidence, blockers = check_identity(seed_order)

    seed_results: dict[str, dict] = {str(seed): {} for seed in seed_order}
    baseline_shas: dict[str, str] = {}
    frame_maxdiff_vs_frozen: float | None = None
    baseline_validation_sha1: str | None = None
    for seed in seed_order:
        work = args.work_dir / f"seed{seed}"
        baseline_dir = work / "baseline"
        printed = ab_v6.run_helper(seed, indices, baseline_dir)
        baseline = read_validation(baseline_dir / "validation.bin")
        baseline_track = ab_v6.read_track(baseline_dir / "validation.trajectory", sample_dt)
        if set(baseline_track) != set(indices):
            raise RuntimeError(f"seed {seed}: baseline episode set differs from held-out manifest")
        baseline_shas[str(seed)] = sha256(baseline_dir / "validation.trajectory")
        if seed == 1:
            frozen = next(
                f["sha256"]
                for f in heldout_manifest["files"]
                if f["name"] == "validation.trajectory"
            )
            if baseline_shas["1"] != frozen:
                raise RuntimeError(
                    "seed 1 baseline is not bit-identical to the frozen post-FIXPUMP trajectory"
                )
            # The helper was rebuilt from current HEAD; later simulator commits
            # changed some recorded sensor frames. The flown sidecar above is
            # still bit-identical to the frozen instrument. Conditioning and
            # labels must stay bit-identical, and the frame delta is counted,
            # not silently accepted as equivalent.
            baseline_validation_sha1 = sha256(baseline_dir / "validation.bin")
            frozen_frames = read_validation(HELDOUT)
            regen_frames = read_validation(baseline_dir / "validation.bin")
            if len(regen_frames.episodes) != len(frozen_frames.episodes):
                raise RuntimeError("seed 1 baseline episode count differs from frozen held-out")
            for regen, frozen in zip(regen_frames.episodes, frozen_frames.episodes):
                if regen.frames.shape != frozen.frames.shape:
                    raise RuntimeError("seed 1 baseline frame shape differs from frozen held-out")
                if not np.array_equal(regen.conditioning, frozen.conditioning) or not np.array_equal(
                    regen.targets, frozen.targets
                ):
                    raise RuntimeError("seed 1 baseline conditioning/labels differ from frozen")
            frame_maxdiff_vs_frozen = float(
                np.max(
                    np.abs(
                        np.stack([e.frames for e in regen_frames.episodes])
                        - np.stack([e.frames for e in frozen_frames.episodes])
                    )
                )
            )
            print(
                f"COUNTED seed 1 input-frame delta vs frozen: {frame_maxdiff_vs_frozen:.6f}; "
                "conditioning and labels bit-identical; flown sidecar bit-identical",
                flush=True,
            )
        for idx, values in printed.items():
            got = baseline_track[idx]
            if (
                abs(got["apogee_m"] - values["apogee_m"]) > 1e-3
                or abs(got["lateral_rms_m"] - values["lateral_rms_m"]) > 1e-3
            ):
                raise RuntimeError(f"seed {seed} episode {idx}: stdout/sidecar mismatch")

        oracle_dir = work / "oracle"
        ab_v6.run_helper(seed, indices, oracle_dir, oracle=True)
        oracle_track = ab_v6.read_track(oracle_dir / "validation.trajectory", sample_dt)
        if set(oracle_track) != set(baseline_track):
            raise RuntimeError(f"seed {seed}: oracle episode set differs")
        oracle_summary = ab_v6.summarise(baseline_track, oracle_track, None)
        oracle_summary["arm"] = "run_episode_with_oracle / fin_accel_command"
        oracle_summary["authority"] = ab_v6.oracle_authority(oracle_dir / "validation.trajectory")
        oracle_summary["sha256"] = {
            "baseline": baseline_shas[str(seed)],
            "assisted": sha256(oracle_dir / "validation.trajectory"),
        }
        seed_results[str(seed)][ORACLE] = oracle_summary

        if str(seed) not in metrics_by_seed:
            continue
        export = export_path(seed)
        checkpoint = torch.load(export, map_location="cpu", weights_only=True)
        model = GDN2Student(GDN2StudentConfig(seed=seed))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        prediction = ab_v6.commands_for_export(model, baseline.episodes)
        tsv_dir = work / ARM / "commands"
        control = ab_v6.build_controls(baseline, prediction, tsv_dir)
        flown_dir = work / ARM / "flown"
        ab_v6.run_helper(seed, indices, flown_dir, commands_dir=tsv_dir)
        flown_track = ab_v6.read_track(flown_dir / "validation.trajectory", sample_dt)
        if set(flown_track) != set(baseline_track):
            raise RuntimeError(f"seed {seed} {ARM}: episode set differs")
        for idx in indices:
            if flown_track[idx]["ticks"] != baseline_track[idx]["ticks"]:
                raise RuntimeError(f"seed {seed} {ARM}: tick count changed at {idx}")
        summary = ab_v6.summarise(baseline_track, flown_track, control)
        summary["arm"] = "BRK02 --braking-teacher four-stage distill student"
        summary["sha256"] = {
            "baseline": baseline_shas[str(seed)],
            "assisted": sha256(flown_dir / "validation.trajectory"),
        }
        summary["export_sha256"] = sha256(export)
        seed_results[str(seed)][ARM] = summary
        del model
        agg = summary["aggregate"]
        print(
            f"seed {seed} {ARM}: deployment lateral "
            f"{agg['baseline']['deployment_lateral_m']:.1f} -> "
            f"{agg['assisted']['deployment_lateral_m']:.1f} m; apogee "
            f"{agg['baseline']['apogee_m']:.1f} -> {agg['assisted']['apogee_m']:.1f} m",
            flush=True,
        )

    provenance = {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_head": ab_v6.git_head(),
        "config": str(CONFIG.relative_to(ROOT)),
        "config_sha256": sha256(CONFIG),
        "config_hash": sha256(CONFIG),
        "heldout": str(HELDOUT.relative_to(ROOT)),
        "heldout_sha256": sha256(HELDOUT),
        "heldout_manifest_sha256": sha256(MANIFEST),
        "quality_report_sha256": sha256(QUALITY),
        "braking_root": str(BRAKING.relative_to(ROOT)),
        "summary_sha256": sha256(SUMMARY),
        "export_manifest_sha256": sha256(EXPORT_MANIFEST),
        "harness": "scripts/ab_braking.py",
        "harness_sha256": sha256(Path(__file__)),
        "gate_source": "scripts/ab_v6.py (imported verbatim)",
        "gate_source_sha256": sha256(SCRIPTS / "ab_v6.py"),
        "teacher_gates_sha256": sha256(ROOT / "ai" / "src" / "potok_ai" / "teacher_gates.py"),
        "trainer_source_sha256": evidence["data_provenance"]["trainer_source_sha256"],
        "braking_teacher_source_sha256": evidence["data_provenance"][
            "braking_teacher_source_sha256"
        ],
        "objective_source_sha256": evidence["data_provenance"]["objective_source_sha256"],
        "objective_version": evidence["data_provenance"]["objective_version"],
        "objective_head": evidence["data_provenance"]["objective_head"],
        "helper": str(ab_v6.HELPER_SRC.relative_to(ROOT)),
        "helper_sha256": sha256(ab_v6.HELPER_SRC),
        "helper_binary": str(ab_v6.HELPER.relative_to(ROOT)),
        "helper_binary_sha256": sha256(ab_v6.HELPER),
        "dynamics_sha256": sha256(ab_v6.DYNAMICS_SRC),
        "dynamics_basis": (
            "post-FIXPUMP; seed-1 zero-command sidecar checked bit-for-bit against the frozen "
            "held-out trajectory"
        ),
        "baseline_trajectory_sha256": baseline_shas,
        "baseline_validation_sha256": baseline_validation_sha1,
        "baseline_frame_maxdiff_vs_frozen": frame_maxdiff_vs_frozen,
        "baseline_input_note": (
            "helper rebuilt from current HEAD; sensor frames differ slightly from the "
            "frozen dataset, while conditioning, labels and the flown sidecar are bit-identical"
        ),
        "arms": [ARM],
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "sample_dt_s": sample_dt,
        "command_limit_mps2": ab_v6.COMMAND_LIMIT,
        "deployment_proxy": "first DESCENT tick (vz<0 and z>=5), radial distance from pad",
        "lateral_limit_m": ab_v6.LATERAL_LIMIT_M,
        "apogee_limit_m": ab_v6.APOGEE_LIMIT_M,
    }

    arm_rep = arm_report(indices, seed_order, seed_results, evidence, provenance)
    arm_rep = ab_v6.evaluate_arm(arm_rep, metrics_by_seed, seed_order, blockers)
    all_blockers = [f"{ARM}: {b}" for b in arm_rep["blockers"]]
    passed = arm_rep["verdict"]["pass"] and not all_blockers
    report = {
        "measurement": "POTOKTRJ v3 flown-truth trajectory sidecar (stride 26), not surrogate",
        "sim_source": arm_rep["sim_source"],
        "supersedes": arm_rep["supersedes"],
        "held_out_indices": indices,
        "seed_order": seed_order,
        "students": arm_rep["students"],
        "arms": {ARM: arm_rep},
        "provenance": provenance,
        "verdict": {
            "status": "PASS" if passed else "FAIL",
            "pass": passed,
            "win_claimed": passed,
            "per_arm": {ARM: arm_rep["verdict"]["status"]},
        },
        "blockers": all_blockers,
        "exit_code": 0 if passed else 1,
        "honesty_notes": [
            (
                "Simulator-only: the POTOKTRJ sidecar is emitted by the same simulator that "
                "flies the episodes; it is not an independent flight truth anchor."
            ),
            (
                "No canopy/deployment event is scored; the proxy is the first DESCENT tick "
                "and is not a landing-dispersion measurement."
            ),
            (
                "The braking teacher is privileged simulator truth. This A/B measures the "
                "distilled student's closed-loop behaviour, not teacher fidelity or flight."
            ),
            (
                "Commands are generated from the baseline flight's recorded frames and "
                "replayed by frame index; they are emitted only where the baseline ai_healthy "
                "target is true, so boost-phase flight stays paired with the control."
            ),
            (
                "provenance.git_head is the checked-out commit at run time; BRK01 and BRK02 "
                "source digests are recorded separately so an uncommitted stack is visible."
            ),
            (
                "The helper was rebuilt from current HEAD; its seed-1 flown sidecar is "
                "bit-identical to the frozen instrument, while a small input-frame delta "
                "is counted in provenance. No gate threshold was changed."
            ),
            "No win is claimed unless every unweakened v6 gate passes; a FAIL is a measured miss.",
        ],
    }

    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.out_md.write_text(markdown(report))
    write_doc(report)
    print(f"wrote {args.out_json.relative_to(ROOT)} and {args.out_md.relative_to(ROOT)}")
    print(
        f"verdict {ARM}: {arm_rep['verdict']['status']} "
        f"(gates failed {arm_rep['verdict']['gates_failed']}, blockers {len(all_blockers)})"
    )
    print(f"overall: {report['verdict']['status']} (exit {report['exit_code']})")
    sys.exit(report["exit_code"])


if __name__ == "__main__":
    main()
