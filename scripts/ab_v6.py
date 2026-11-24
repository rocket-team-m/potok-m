#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pre-registered v6 closed-loop A/B: RETRAIN-NIGHT default and adversarial arms.

Flies the 20 held-out scenarios (16..19) through the read-only v3 control seam on
post-FIXPUMP dynamics. Zero command is the control; ``run_episode_with_oracle``
is the positive control. Two separate opt-in export sets are measured, each as
its own arm with its own verdict:

* ``night_default``      -- ``ai/out/night_final/exports/teacher_v2_s{1,2,3}.pt``
* ``night_adversarial``  -- ``ai/out/night_final/adversarial/runs/s{1,2,3}/student.pt``

Both are the four-stage distill (sft -> privileged -> dagger -> distill) students
of the current teacher-v2 default; the adversarial arm differs only by the
opt-in ``--adversarial`` curriculum. No export is substituted, no gate, physics,
weight or knob is changed here.

Gates are fixed here, never fitted and never weakened (AIPOST1/EVAL01):
deployment-proxy lateral RMS <= 1000 m and apogee >= 2000 m on every seed, ECE
with an explicit disjoint source, provenance from metrics and artifact hashes,
and the verbatim unweakened EVAL01 suite (-35% / +15%, 1.25, 0.05, 3x spread).
A win is claimed only if every gate of every arm passes.

Usage:
    ai/.venv/bin/python scripts/ab_v6.py
Work files live outside the repo (default /tmp/opencode/ab_v6).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT / "ai" / "src"))

from potok_ai.gdn2_student import GDN2Student, GDN2StudentConfig

from potok_ai import teacher_gates
from potok_ai.data import SMART_FEATURES, read_validation
from potok_ai.physics import physics_baseline

CONFIG = ROOT / "ai" / "out" / "v3_final" / "sim.toml"
HELD_OUT = ROOT / "ai" / "out" / "final_v3_postpump" / "data" / "held-out"
HELDOUT = HELD_OUT / "validation.bin"
MANIFEST = HELD_OUT / "manifest.json"
QUALITY = ROOT / "ai" / "out" / "final_v3_postpump" / "data" / "quality_report.json"
NIGHT = ROOT / "ai" / "out" / "night_final"
SUMMARY = NIGHT / "summary.json"
ADV_SUMMARY = NIGHT / "adversarial_summary.json"
SHIELD = NIGHT / "shield.json"
EXPORT_MANIFEST = NIGHT / "exports" / "manifest.json"
HELPER = SCRIPTS / "ab_control_v3" / "target" / "release" / "ab-control-v3"
HELPER_SRC = SCRIPTS / "ab_control_v3" / "src" / "main.rs"
DYNAMICS_SRC = ROOT / "sim" / "core" / "src" / "dynamics.rs"
WORK = Path("/tmp/opencode/ab_v6")
OUT_JSON = SCRIPTS / "ab_report_v6.json"
OUT_MD = SCRIPTS / "ab_report_v6.md"

SEEDS = (1, 2, 3)
ARMS = ("night_default", "night_adversarial")
ARM_LABELS = {
    "night_default": "RETRAIN-NIGHT default four-stage distill exports",
    "night_adversarial": "RETRAIN-NIGHT opt-in adversarial four-stage distill students",
}
ORACLE = "oracle_gtheta"
STAGES = ("sft", "privileged", "dagger", "distill")
BOUND_MPS2 = 10.0
LATERAL_LIMIT_M = 1000.0
APOGEE_LIMIT_M = 2000.0
COMMAND_LIMIT = 10.0
ULP_M = float(np.spacing(np.float32(600.0)))
SIDECAR_MAGIC = b"POTOKTRJ"
SIDECAR_STRIDE = 26
ADV_INTENSITY = {"sft": 0.0, "privileged": 0.1, "dagger": 0.2, "distill": 0.3}
WIN_RATE_CAP = 0.3


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def arm_export(arm: str, seed: int) -> Path:
    if arm == "night_default":
        return NIGHT / "exports" / f"teacher_v2_s{seed}.pt"
    if arm == "night_adversarial":
        return NIGHT / "adversarial" / "runs" / f"s{seed}" / "student.pt"
    raise ValueError(f"unknown arm {arm!r}")


def arm_metrics_path(arm: str, seed: int) -> Path:
    if arm == "night_default":
        return NIGHT / "runs" / f"s{seed}" / "metrics.json"
    if arm == "night_adversarial":
        return NIGHT / "adversarial" / "runs" / f"s{seed}" / "metrics.json"
    raise ValueError(f"unknown arm {arm!r}")


def run_helper(
    seed: int,
    indices: list[int],
    out_dir: Path,
    oracle: bool = False,
    commands_dir: Path | None = None,
) -> dict[int, dict]:
    if not HELPER.is_file():
        raise FileNotFoundError(
            f"missing {HELPER.relative_to(ROOT)}; build with: cargo build --release "
            "--manifest-path scripts/ab_control_v3/Cargo.toml "
            "--target-dir scripts/ab_control_v3/target"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(HELPER),
        "--config",
        str(CONFIG),
        "--seed",
        str(seed),
        "--indices",
        ",".join(map(str, indices)),
        "--out-dir",
        str(out_dir),
    ]
    if oracle:
        cmd.append("--oracle")
    if commands_dir is not None:
        cmd += ["--commands-dir", str(commands_dir)]
    proc = subprocess.run(cmd, check=False, cwd=ROOT / "sim", capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ab-control-v3 failed ({proc.returncode}): {proc.stderr}")
    printed: dict[int, dict] = {}
    for line in proc.stdout.splitlines():
        if not line.startswith("episode "):
            continue
        parts = dict(part.split("=", 1) for part in line.split() if "=" in part)
        printed[int(line.split()[1].rstrip(":"))] = {
            "scenario_id": int(parts["scenario"]),
            "apogee_m": float(parts["apogee"]),
            "lateral_rms_m": float(parts["lateral_rms"]),
        }
    if len(printed) != len(indices):
        raise RuntimeError(f"helper printed {len(printed)} episodes, expected {len(indices)}")
    return printed


def read_track(path: Path, sample_dt: float) -> dict[int, dict]:
    """Measure the sidecar directly; deployment proxy = first DESCENT tick.

    Phase rule is ``dynamics::phase_at``: DESCENT iff vz < 0 and z >= 5 after
    burnout (dynamics.rs:736-757). LATERAL01.md:116-127 names this tick as the
    simulator's deployment proxy (the repo has no independently scored canopy).
    """
    raw = path.read_bytes()
    if len(raw) < 64:
        raise ValueError(f"{path}: sidecar shorter than its 64-byte header")
    magic, version, count, frames, stride = struct.unpack_from("<8sIIII", raw, 0)
    if magic != SIDECAR_MAGIC or version != 3 or stride != SIDECAR_STRIDE:
        raise ValueError(
            f"{path}: expected POTOKTRJ v3 stride 26, got {magic!r} v{version}/{stride}"
        )
    off = 64
    result: dict[int, dict] = {}
    for _ in range(count):
        (index,) = struct.unpack_from("<Q", raw, off)
        off += 8
        values = np.frombuffer(raw, dtype="<f4", count=frames * stride, offset=off)
        off += frames * stride * 4
        rows = values.reshape(frames, stride).astype(np.float64)
        if not np.isfinite(rows).all():
            raise ValueError(f"{path}: non-finite flown truth at episode {index}")
        pos, vel = rows[:, :3], rows[:, 3:6]
        lateral = np.hypot(pos[:, 0], pos[:, 1])
        descent = np.flatnonzero((vel[:, 2] < 0.0) & (pos[:, 2] >= 5.0))
        if descent.size == 0:
            raise ValueError(f"{path}: episode {index} has no DESCENT tick, no deployment proxy")
        tick = int(descent[0])
        result[int(index)] = {
            "apogee_m": float(np.max(pos[:, 2])),
            "lateral_rms_m": float(np.sqrt(np.mean(lateral**2))),
            "deployment_lateral_m": float(lateral[tick]),
            "deployment_tick": tick,
            "deployment_time_s": tick * sample_dt,
            "deployment_altitude_m": float(pos[tick, 2]),
            "ticks": int(frames),
        }
    if off != len(raw):
        raise ValueError(f"{path}: trailing bytes in sidecar")
    return result


def oracle_authority(path: Path) -> dict:
    """Did the flown oracle exercise the +/-10 m/s^2 command actuator?"""
    raw = path.read_bytes()
    magic, version, count, frames, stride = struct.unpack_from("<8sIIII", raw, 0)
    if magic != SIDECAR_MAGIC or version != 3 or stride != SIDECAR_STRIDE:
        raise ValueError(f"{path}: expected POTOKTRJ v3 stride 26")
    off = 64
    active = cap_hits = 0
    max_command = 0.0
    for _ in range(count):
        (episode,) = struct.unpack_from("<Q", raw, off)
        off += 8
        values = np.frombuffer(raw, dtype="<f4", count=frames * stride, offset=off)
        off += frames * stride * 4
        demand = values.reshape(frames, stride).astype(np.float64)[:, 24:26]
        if not np.isfinite(demand).all():
            raise ValueError(f"{path}: non-finite command at episode {episode}")
        fin = np.stack((demand[:, 0], -demand[:, 1]), axis=1)
        mask = np.max(np.abs(fin), axis=1) > np.spacing(np.float32(COMMAND_LIMIT))
        max_command = max(max_command, float(np.max(np.abs(fin))))
        active += int(mask.sum())
        if mask.any():
            cap_hits += int(
                (
                    np.max(np.abs(fin[mask]), axis=1)
                    >= COMMAND_LIMIT - np.spacing(np.float32(COMMAND_LIMIT))
                ).sum()
            )
    if active == 0:
        raise ValueError(f"{path}: oracle never commanded the actuator")
    return {
        "active_ticks": active,
        "cap_hit_ticks": cap_hits,
        "cap_fraction": cap_hits / active,
        "max_command_mps2": max_command,
    }


@torch.no_grad()
def commands_for_export(model: torch.nn.Module, episodes: list) -> dict:
    """One stateful GDN2Student sweep per held-out episode, batched over 20.

    The distill target is the applied residual decision ``command - baseline``
    (train_teacher_v2.py:1071-1077), so deployment is ``baseline + correction``
    with no second confidence gate: SHIELD02's projection already happened in
    that target. The seam clips at its hard +/-10 m/s^2 bound.
    """
    frames = np.stack([np.asarray(ep.frames[:, :SMART_FEATURES], np.float32) for ep in episodes])
    conds = np.stack([np.asarray(ep.conditioning, np.float32) for ep in episodes])
    if conds.shape != (len(episodes), 6):
        raise ValueError(f"expected 6 conditioning values per episode, got {conds.shape}")
    out = model(torch.from_numpy(frames), torch.from_numpy(conds))
    if tuple(out.shape) != (len(episodes), frames.shape[1], 3):
        raise ValueError(f"export returned unexpected shape {tuple(out.shape)}")
    if not bool(torch.isfinite(out).all()):
        raise ValueError("export returned non-finite corrections")
    return {
        "deltas": out[..., :2].numpy().astype(np.float64),
        "confidences": out[..., 2].numpy().astype(np.float64),
    }


def build_controls(baseline, prediction: dict, out_dir: Path) -> dict:
    """Write one TSV per episode in the seam's [roll, pitch] m/s^2 order.

    The paired baseline's ``ai_healthy`` target (index 3) is the sim's own
    authority/sensor gate (firmware_dataset.rs:440): the AI may not command
    during boost or on invalid sensor frames. Outside that gate the command is
    exactly zero, so the boost phase stays bit-identical to the control arm.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    base = np.stack(
        [
            physics_baseline(torch.as_tensor(ep.frames[:, :SMART_FEATURES], dtype=torch.float32))
            .numpy()
            .astype(np.float64)
            for ep in baseline.episodes
        ]
    )
    healthy = np.stack([np.asarray(ep.targets[:, 3], np.float64) for ep in baseline.episodes])
    if healthy.shape != base.shape[:2]:
        raise ValueError(f"ai_healthy shape {healthy.shape} differs from command {base.shape[:2]}")
    gate = healthy > 0.5
    raw = np.where(gate[:, :, None], base + prediction["deltas"], 0.0)
    command = np.clip(raw, -COMMAND_LIMIT, COMMAND_LIMIT)
    stats = {
        "rows": int(command.shape[0] * command.shape[1]),
        "authority_gate_open_ticks": int(gate.sum()),
        "authority_gate_closed_ticks": int((~gate).sum()),
        "interventions": int((np.max(np.abs(command), axis=2) > 1e-9).sum()),
        "clipped_ticks": int((np.abs(raw) > COMMAND_LIMIT).any(axis=2).sum()),
        "max_command_mps2": float(np.max(np.abs(command))),
        "mean_abs_correction_mps2": float(np.mean(np.abs(prediction["deltas"]))),
        "mean_confidence": float(np.mean(prediction["confidences"])),
        "confidence_gate": "none (residual decision target is ungated; seam clips only)",
    }
    for e, ep in enumerate(baseline.episodes):
        with (out_dir / f"{ep.index}.tsv").open("w") as handle:
            handle.write("# frame_index g_roll_mps2 g_pitch_mps2\n")
            for tick, (roll, pitch) in enumerate(command[e]):
                handle.write(f"{tick} {roll:.6f} {pitch:.6f}\n")
    return stats


def summarise(b_track: dict, a_track: dict, control: dict | None) -> dict:
    scenarios = sorted({k % 20 for k in b_track})

    def rows_for(keys: list[int]) -> dict:
        row: dict = {"episodes": len(keys), "baseline": {}, "assisted": {}}
        for metric in ("apogee_m", "lateral_rms_m", "deployment_lateral_m"):
            # The two lateral fields are RMS quantities, so pool them over
            # episodes exactly as trajectory.rs:263-273 does for the control arm.
            if metric == "apogee_m":
                b = float(np.mean([b_track[k][metric] for k in keys]))
                a = float(np.mean([a_track[k][metric] for k in keys]))
            else:
                b = float(np.sqrt(np.mean([b_track[k][metric] ** 2 for k in keys])))
                a = float(np.sqrt(np.mean([a_track[k][metric] ** 2 for k in keys])))
            row["baseline"][metric] = b
            row["assisted"][metric] = a
            row[f"{metric}_delta"] = a - b
        return row

    aggregate = rows_for(sorted(b_track))
    if control:
        aggregate.update(control)
    return {
        "per_scenario": [
            {"scenario": f"scenario_{sc:02d}", **rows_for([k for k in b_track if k % 20 == sc])}
            for sc in scenarios
        ],
        "aggregate": aggregate,
    }


def check_identity(arm: str, seed_order: list[int]) -> tuple[dict[str, dict], dict, list[str]]:
    """Arm identity and evidence; a mismatch is a loud refusal, never a stand-in."""
    summary = json.loads(SUMMARY.read_text())
    adv_summary = json.loads(ADV_SUMMARY.read_text())
    shield = json.loads(SHIELD.read_text())
    manifest = json.loads(EXPORT_MANIFEST.read_text())
    config_sha = sha256(CONFIG)
    heldout_sha = sha256(HELDOUT)
    dp = summary.get("data_provenance", {})

    if summary.get("ab_win_claimed") is not False:
        raise ValueError("night summary claims an A/B win; exports are not artifacts")
    if summary.get("rlvr_used") is not False:
        raise ValueError("night summary says RLVR was used; arm identity changed")
    if tuple(summary.get("stage_order", ())) != STAGES:
        raise ValueError("night summary stage_order disagrees with the four-stage default")
    if dp.get("config_sha256") != config_sha or dp.get("heldout_sha256") != heldout_sha:
        raise ValueError("night summary data provenance disagrees with the flown config/held-out")
    if shield.get("authority") != BOUND_MPS2:
        raise ValueError("night shield authority disagrees with the command bound")
    if summary.get("shield_verification", {}).get("counts", {}).get("accepted") != 1:
        raise ValueError("night shield verification no longer has its counted accept case")
    if manifest.get("claim", "").find("no win is claimed") < 0:
        raise ValueError("export manifest no longer disclaims a win")
    if tuple(manifest.get("stage_order", ())) != STAGES:
        raise ValueError("export manifest stage_order disagrees with the four-stage default")
    if manifest.get("fixes", {}).get("command_bound") != BOUND_MPS2:
        raise ValueError("export manifest no longer pins the +/-10 command bound")
    if adv_summary.get("win_rate_cap") != WIN_RATE_CAP or adv_summary.get("status") != "RUN":
        raise ValueError("adversarial summary no longer records the opt-in RUN arm")
    adv_arm_summary = summary.get("adversarial_arm", {})
    if adv_arm_summary.get("status") != "RUN" or adv_arm_summary.get("ab_win_claimed") is not False:
        raise ValueError("night summary adversarial arm is not the descriptive opt-in RUN arm")
    formula = summary.get("upgrades", {}).get("AIOBJ", {}).get("formula")
    if not isinstance(formula, str) or not formula:
        raise ValueError("night summary does not record the AIOBJ impact formula")
    if summary.get("upgrades", {}).get("AIOBJ", {}).get("source") != "impact":
        raise ValueError("night summary AIOBJ source is not the impact objective")

    expected_arm = "default" if arm == "night_default" else "adversarial"
    expected_adv = arm == "night_adversarial"
    blockers: list[str] = []
    metrics_by_seed: dict[str, dict] = {}
    evidence: dict = {
        "arm": expected_arm,
        "recipe_adversarial": expected_adv,
        "exports_sha256": {},
        "metrics_sha256": {},
        "metrics_config_hash": {},
        "ece_source": {},
        "ece_held_out": {},
        "conformal_coverage": {},
        "residual_gain": {},
        "privileged_batches": {},
        "rmse_held_out": {},
        "stages_completed": list(STAGES),
        "command_bound": BOUND_MPS2,
        "legacy_student_bound": False,
        "stage_order": list(STAGES),
        "data_provenance": dp,
        "summary_sha256": sha256(SUMMARY),
        "adversarial_summary_sha256": sha256(ADV_SUMMARY),
        "shield_sha256": sha256(SHIELD),
        "export_manifest_sha256": sha256(EXPORT_MANIFEST),
        "objective_formula": formula,
    }
    if expected_adv:
        evidence["adversarial_intensity"] = ADV_INTENSITY
        evidence["win_rate_cap"] = WIN_RATE_CAP

    for seed in seed_order:
        export_path = arm_export(arm, seed)
        metrics_path = arm_metrics_path(arm, seed)
        if not export_path.is_file() or not metrics_path.is_file():
            blockers.append(f"{arm} seed {seed}: missing export/metrics; no stand-in substituted")
            continue
        metrics = json.loads(metrics_path.read_text())
        export_sha = sha256(export_path)
        where = f"{arm} seed {seed}"

        if metrics.get("arm") != expected_arm:
            raise ValueError(f"{where}: arm field {metrics.get('arm')!r} != {expected_arm!r}")
        if metrics.get("night_provenance") is not True:
            raise ValueError(f"{where}: metrics do not carry night_provenance")
        if metrics.get("seed") != seed:
            raise ValueError(f"{where}: metrics seed field mismatch")
        if tuple(metrics.get("stages_completed", ())) != STAGES:
            raise ValueError(f"{where}: stages_completed differs from the four-stage default")
        if metrics.get("reward_provenance") != {}:
            raise ValueError(f"{where}: non-empty reward_provenance; RLVR not opted into")
        if metrics.get("ece_source") != "held-out":
            raise ValueError(f"{where}: ece_source is not the disjoint held-out label")
        if metrics.get("ece") != metrics.get("ece_held_out"):
            raise ValueError(f"{where}: ece disagrees with the disjoint held-out ECE")
        if not 0.0 < metrics.get("conformal_coverage", 0.0) <= 1.0:
            raise ValueError(f"{where}: conformal coverage outside (0, 1]")
        if metrics.get("student_command_bound") != BOUND_MPS2:
            raise ValueError(f"{where}: student command bound is not +/-10")
        if metrics.get("legacy_student_bound") is not False:
            raise ValueError(f"{where}: legacy student bound is still enabled")
        rp = metrics.get("residual_policy", {})
        if rp.get("enabled") is not True or rp.get("source") != "residual_policy":
            raise ValueError(f"{where}: SHIELD02 residual policy is not enabled")
        if not isinstance(rp.get("gain"), (int, float)) or not math.isfinite(rp["gain"]):
            raise ValueError(f"{where}: residual gain is not finite")
        stages = metrics.get("stages", {})
        if any(stage not in stages for stage in STAGES):
            raise ValueError(f"{where}: a four-stage record is missing")
        privileged = stages["privileged"]["details"]
        dagger = stages["dagger"]["details"]
        distill = stages["distill"]["details"]
        objective = privileged.get("objective", {})
        if objective.get("source") != "impact" or objective.get("legacy") is not False:
            raise ValueError(f"{where}: AIOBJ impact objective is not the default")
        if objective.get("reason") is not None or objective.get("formula") != formula:
            raise ValueError(f"{where}: AIOBJ formula/reason disagrees with the night summary")
        if privileged.get("privileged_batches") != 2300:
            raise ValueError(f"{where}: AIFIX2 privileged batches are missing")
        if privileged.get("privileged_source") != "privileged_teacher":
            raise ValueError(f"{where}: privileged teacher source changed")
        if privileged.get("eval_label_blind") is not True:
            raise ValueError(f"{where}: privileged eval is not label-blind")
        if dagger.get("rollout_source") != "on_policy_student:gdn2_student":
            raise ValueError(f"{where}: DAgger rollout is not the AIFIX3 on-policy student")
        if dagger.get("privileged_source") != "v3_truth_sidecar":
            raise ValueError(f"{where}: DAgger privileged truth source changed")
        if dagger.get("privileged_fields_present") is not True:
            raise ValueError(f"{where}: DAgger privileged fields are not present")
        if (
            dagger.get("student_induced_rows", 0) <= 0
            or len(str(dagger.get("rollout_sha256", ""))) != 64
        ):
            raise ValueError(f"{where}: DAgger on-policy evidence missing")
        if distill.get("residual_policy_target") is not True:
            raise ValueError(f"{where}: distill target is not the residual decision")
        if distill.get("student_command_bound") != BOUND_MPS2:
            raise ValueError(f"{where}: distill student bound is not +/-10")
        if distill.get("legacy_student_bound") is not False:
            raise ValueError(f"{where}: distill legacy bound is still enabled")
        if distill.get("student_source") != "gdn2_student":
            raise ValueError(f"{where}: distill student source changed")
        if metrics.get("data_provenance") != dp:
            raise ValueError(f"{where}: data provenance disagrees with the night summary")
        if metrics.get("recipe", {}).get("adversarial") is not expected_adv:
            raise ValueError(f"{where}: recipe adversarial flag disagrees with the arm")
        fixes_adv = metrics.get("fixes", {}).get("adversarial")
        if expected_adv:
            if fixes_adv != "enabled via --adversarial":
                raise ValueError(f"{where}: fixes do not record the opt-in adversarial arm")
            for stage in STAGES:
                info = stages[stage]["details"].get("adversarial", {})
                if info.get("enabled") is not True or info.get("stage") != stage:
                    raise ValueError(f"{where}: AICURR {stage} not enabled/labelled")
                if info.get("win_rate_cap") != WIN_RATE_CAP:
                    raise ValueError(f"{where}: AICURR {stage} win-rate cap changed")
                if not math.isclose(
                    info.get("intensity", -1.0), ADV_INTENSITY[stage], abs_tol=1e-9
                ):
                    raise ValueError(f"{where}: AICURR {stage} intensity changed")
                if info.get("offered", 0) <= 0 or not 0 <= info.get("won", -1) <= info["offered"]:
                    raise ValueError(f"{where}: AICURR {stage} offered/won counts invalid")
                if ADV_INTENSITY[stage] == 0.0 and info.get("won") != 0:
                    raise ValueError(f"{where}: AICURR {stage} calm stage is not clean")
            seed_rmse = adv_summary.get("rmse_held_out_by_seed", {}).get(str(seed))
            seed_rmse2 = adv_arm_summary.get("rmse_held_out_by_seed", {}).get(str(seed))
            if seed_rmse != metrics.get("rmse_held_out") or seed_rmse2 != seed_rmse:
                raise ValueError(f"{where}: adversarial held-out RMSE disagrees with the summaries")
        else:
            if fixes_adv != "not opted in":
                raise ValueError(f"{where}: default arm records adversarial opt-in")
            if any("adversarial" in stages[s]["details"] for s in STAGES):
                raise ValueError(f"{where}: default arm carries adversarial stage records")
            declared = manifest["students"].get(str(seed))
            if declared is None:
                raise ValueError(f"{where}: export manifest has no student record")
            if export_sha != declared["sha256"]:
                raise ValueError(f"{where}: export sha256 disagrees with export manifest")
            if declared.get("file") != export_path.name:
                raise ValueError(f"{where}: export manifest names a different student file")
            if metrics.get("config_hash") != declared.get("config_hash"):
                raise ValueError(f"{where}: metrics config_hash disagrees with export manifest")
            if metrics.get("ece_source") != declared.get("ece_source"):
                raise ValueError(f"{where}: metrics ece_source disagrees with export manifest")
            if metrics.get("rmse_held_out") != declared.get("rmse_held_out"):
                raise ValueError(f"{where}: metrics held-out RMSE disagrees with export manifest")
            if summary.get("rmse_held_out_by_seed", {}).get(str(seed)) != metrics.get(
                "rmse_held_out"
            ):
                raise ValueError(f"{where}: held-out RMSE disagrees with the night summary")

        metrics_by_seed[str(seed)] = metrics
        evidence["exports_sha256"][f"seed{seed}"] = export_sha
        evidence["metrics_sha256"][f"seed{seed}"] = sha256(metrics_path)
        evidence["metrics_config_hash"][str(seed)] = metrics["config_hash"]
        evidence["ece_source"][str(seed)] = metrics["ece_source"]
        evidence["ece_held_out"][str(seed)] = metrics["ece_held_out"]
        evidence["conformal_coverage"][str(seed)] = metrics["conformal_coverage"]
        evidence["residual_gain"][str(seed)] = rp["gain"]
        evidence["privileged_batches"][str(seed)] = privileged["privileged_batches"]
        evidence["rmse_held_out"][str(seed)] = metrics["rmse_held_out"]

    return metrics_by_seed, evidence, blockers


def arm_report(
    arm: str,
    seed_order: list[int],
    indices: list[int],
    seed_results: dict,
    metrics_by_seed: dict,
    evidence: dict,
    provenance: dict,
) -> dict:
    report = {
        "measurement": "POTOKTRJ v3 flown-truth trajectory sidecar (stride 26), not surrogate",
        "sim_source": (
            "read-only scripts/ab_control_v3 on git "
            + provenance["git_head"]
            + "; oracle uses run_episode_with_oracle / fin_accel_command"
        ),
        "supersedes": (
            "any win reading of the RETRAIN-NIGHT server-side RMSE comparison: it was never "
            "a closed-loop A/B and is void here; the unweakened pre-registered gates are restored"
        ),
        "held_out_indices": indices,
        "seed_order": seed_order,
        "students": [
            {
                "name": arm,
                "family": f"gdn2_student ({ARM_LABELS[arm]}, +/-10 head, residual+shield)",
                "stage_order": list(STAGES),
                "exports_sha256": evidence["exports_sha256"],
            }
        ],
        "verdict": {"status": "PENDING", "primary": arm},
        "provenance": {**provenance, "arm": arm, **evidence},
        "seeds": {
            str(seed): {
                name: seed_results[str(seed)][name]
                for name in (ORACLE, arm)
                if name in seed_results.get(str(seed), {})
            }
            for seed in seed_order
        },
    }
    return report


def evaluate_arm(
    report: dict, metrics_by_seed: dict, seed_order: list[int], blockers: list[str]
) -> dict:
    arm = report["verdict"]["primary"]
    evals: dict[str, bool] = {}
    for name, fn in (
        ("rmse_ratio", teacher_gates.check_ratio),
        ("ece_source", teacher_gates.check_ece_source),
        ("ece", teacher_gates.check_ece),
    ):
        try:
            result = fn(metrics_by_seed)
        except teacher_gates.ProvenanceError as exc:
            blockers.append(f"EVAL01 {name}: {exc}")
            evals[name] = False
            continue
        evals[name] = result.passed
        if not result.passed:
            blockers.append(f"EVAL01 {name} failed: {result.detail}")

    try:
        result = teacher_gates.check_provenance(report)
        evals["provenance"] = result.passed
        if not result.passed:
            blockers.append(f"EVAL01 provenance failed: {result.detail}")
    except teacher_gates.ProvenanceError as exc:
        blockers.append(f"EVAL01 provenance: {exc}")
        evals["provenance"] = False

    try:
        spread = teacher_gates.check_seed_spread(report, student=arm)
        improvement = teacher_gates.check_improvement(report, student=arm)
        oracle_check = teacher_gates.check_oracle(report)
        evals["seed_spread"] = spread.passed
        evals["closed_loop_improvement"] = improvement.passed
        evals["oracle_positive_control"] = oracle_check.passed
        for result in (spread, improvement, oracle_check):
            if not result.passed:
                blockers.append(f"EVAL01 {result.name} failed: {result.detail}")
    except teacher_gates.ProvenanceError as exc:
        blockers.append(f"EVAL01 closed-loop gate: {exc}")
        for name in ("seed_spread", "closed_loop_improvement", "oracle_positive_control"):
            evals[name] = False

    positive_control: dict = {"per_seed": {}, "passed": False}
    for seed in seed_order:
        entry = report["seeds"].get(str(seed), {}).get(ORACLE)
        if entry is None:
            blockers.append(f"{arm} seed {seed}: oracle missing; positive control not measured")
            continue
        agg = entry["aggregate"]
        dep_delta = agg["deployment_lateral_m_delta"]
        ap_delta = agg["apogee_m_delta"]
        ok = dep_delta < -ULP_M and ap_delta > ULP_M
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
        blockers.append(f"{arm}: oracle positive control did not improve both metrics on all seeds")

    primary_gates: list[dict] = []
    for seed in seed_order:
        entry = report["seeds"].get(str(seed), {}).get(arm)
        if entry is None:
            primary_gates.append(
                {
                    "name": f"seed {seed} absolute gates",
                    "passed": False,
                    "detail": "no export measured; no stand-in substituted",
                }
            )
            continue
        agg = entry["aggregate"]
        dep = agg["assisted"]["deployment_lateral_m"]
        ap = agg["assisted"]["apogee_m"]
        passed = dep <= LATERAL_LIMIT_M and ap >= APOGEE_LIMIT_M
        primary_gates.append(
            {
                "name": f"seed {seed} absolute gates",
                "passed": passed,
                "detail": (
                    f"deployment lateral {dep:.2f} m <= {LATERAL_LIMIT_M:g} m: "
                    f"{dep <= LATERAL_LIMIT_M}; apogee {ap:.2f} m >= {APOGEE_LIMIT_M:g} m: "
                    f"{ap >= APOGEE_LIMIT_M}"
                ),
            }
        )

    gates = [
        {
            "name": f"eval01 {name}",
            "passed": bool(value),
            "detail": "verbatim EVAL01 gate" if value else "see blockers",
        }
        for name, value in evals.items()
    ]
    gates += primary_gates
    gates.append(
        {
            "name": "oracle_deployment_and_apogee_positive_control",
            "passed": positive_control["passed"],
            "detail": "both metrics improve against paired zero command on every seed",
        }
    )
    passed = all(row["passed"] for row in gates) and not blockers
    report["positive_control"] = positive_control
    report["gates"] = gates
    report["blockers"] = blockers
    report["verdict"] = {
        "status": "PASS" if passed else "FAIL",
        "primary": arm,
        "pass": passed,
        "win_claimed": passed,
        "absolute_gates": all(row["passed"] for row in primary_gates),
        "eval01_gates": evals,
        "oracle_positive_control": positive_control["passed"],
        "gates_failed": sum(1 for row in gates if not row["passed"]),
    }
    return report


def markdown(report: dict) -> str:
    verdict = report["verdict"]
    out = [
        "# v6 closed-loop A/B — unweakened gates, RETRAIN-NIGHT default vs adversarial",
        "",
        (
            f"Overall verdict: **{verdict['status']}** (exit {report['exit_code']}). "
            "A win is claimed only if every gate of every arm passes. "
            "This is a post-FIXPUMP simulator measurement, not a flight claim."
        ),
        "",
        "| arm | verdict | gates failed | blockers | win claimed |",
        "|---|---|---:|---:|---|",
    ]
    for arm in ARMS:
        entry = report["arms"][arm]["verdict"]
        out.append(
            f"| {arm} | {entry['status']} | {entry['gates_failed']} | "
            f"{len(report['arms'][arm]['blockers'])} | {entry['win_claimed']} |"
        )
    out += [
        "",
        "## Gates (hard-coded, unweakened AIPOST1/EVAL01)",
        "",
        (
            "Deployment proxy = first `DESCENT` tick (vz<0 and z>=5) in each flown POTOKTRJ v3 "
            "sidecar; the absolute gate uses the pooled RMS of its radial distance from the pad "
            "over the 20 held-out episodes (LATERAL01.md:116-127). Apogee is the max z over the "
            "whole 30 s trajectory. The repo has no independently scored canopy; nothing here is "
            "a landing-dispersion number."
        ),
        "",
    ]
    for arm in ARMS:
        entry = report["arms"][arm]
        out += [f"### {arm} — {ARM_LABELS[arm]}", "", "| gate | result | detail |", "|---|---|---|"]
        for row in entry["gates"]:
            out.append(
                f"| {row['name']} | {'PASS' if row['passed'] else 'FAIL'} | {row['detail']} |"
            )
        out += [
            "",
            "| seed | arm | deployment lateral | apogee | whole-trajectory lateral RMS |",
            "|---:|---|---:|---:|---:|",
        ]
        for seed in report["seed_order"]:
            arms = entry["seeds"].get(str(seed), {})
            for name in (ORACLE, arm):
                if name not in arms:
                    continue
                agg = arms[name]["aggregate"]
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
                f"apogee {row['apogee_delta_m']:+.2f} m, "
                f"actuator cap ticks {row['authority']['cap_hit_ticks']}/"
                f"{row['authority']['active_ticks']}"
            )
        if entry["blockers"]:
            out += ["", "Blockers:", ""] + [f"- {b}" for b in entry["blockers"]]
        out.append("")

    out += ["## Shared provenance", "", "| field | value |", "|---|---|"]
    for key, value in report["provenance"].items():
        text = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
        out.append(f"| {key} | `{text}` |")
    out += ["", "## Honesty notes", ""]
    out += [f"- {n}" for n in report["honesty_notes"]]
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--work-dir", type=Path, default=WORK)
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    ap.add_argument("--out-md", type=Path, default=OUT_MD)
    args = ap.parse_args()
    torch.set_num_threads(1)
    for path in (
        HELPER,
        CONFIG,
        HELDOUT,
        MANIFEST,
        EXPORT_MANIFEST,
        SUMMARY,
        ADV_SUMMARY,
        SHIELD,
        QUALITY,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing {path.relative_to(ROOT)}")

    config = tomllib.loads(CONFIG.read_text())
    sample_dt = 1.0 / config["run"]["frame_rate_hz"]
    manifest = json.loads(MANIFEST.read_text())
    heldout = read_validation(HELDOUT)
    indices = [int(ep.index) for ep in heldout.episodes]
    frames_per_episode = int(manifest["frames_per_episode"])
    if len(indices) != 20 or any(i % 20 not in (16, 17, 18, 19) for i in indices):
        raise ValueError(f"unexpected held-out indices: {indices}")
    if any(ep.frames.shape[0] != frames_per_episode for ep in heldout.episodes):
        raise ValueError("held-out frame count differs from manifest")
    if manifest.get("split") != "held-out" or sha256(HELDOUT) != next(
        f["sha256"] for f in manifest["files"] if f["name"] == "validation.bin"
    ):
        raise ValueError("held-out manifest split or file hash mismatch")

    seed_order = sorted(args.seeds)
    if seed_order != list(SEEDS):
        raise ValueError(f"pre-registered gate needs seeds {list(SEEDS)}, got {seed_order}")

    arm_evidence: dict[str, tuple[dict[str, dict], dict, list[str]]] = {}
    for arm in ARMS:
        arm_evidence[arm] = check_identity(arm, seed_order)

    seed_results: dict[str, dict] = {str(seed): {} for seed in seed_order}
    baseline_shas: dict[str, str] = {}
    for seed in seed_order:
        work = args.work_dir / f"seed{seed}"
        baseline_dir = work / "baseline"
        printed = run_helper(seed, indices, baseline_dir)
        baseline = read_validation(baseline_dir / "validation.bin")
        baseline_track = read_track(baseline_dir / "validation.trajectory", sample_dt)
        if set(baseline_track) != set(indices):
            raise RuntimeError(f"seed {seed}: baseline episode set differs from held-out manifest")
        baseline_shas[str(seed)] = sha256(baseline_dir / "validation.trajectory")
        if seed == 1:
            frozen = next(
                f["sha256"] for f in manifest["files"] if f["name"] == "validation.trajectory"
            )
            if baseline_shas["1"] != frozen:
                raise RuntimeError(
                    "seed 1 baseline is not bit-identical to the frozen post-FIXPUMP "
                    "trajectory: the helper is not the corrected-physics binary"
                )
            if sha256(baseline_dir / "validation.bin") != sha256(HELDOUT):
                raise RuntimeError("seed 1 baseline validation.bin differs from frozen held-out")
        for idx, values in printed.items():
            got = baseline_track[idx]
            if (
                abs(got["apogee_m"] - values["apogee_m"]) > 1e-3
                or abs(got["lateral_rms_m"] - values["lateral_rms_m"]) > 1e-3
            ):
                raise RuntimeError(f"seed {seed} episode {idx}: stdout/sidecar mismatch")

        oracle_dir = work / "oracle"
        run_helper(seed, indices, oracle_dir, oracle=True)
        oracle_track = read_track(oracle_dir / "validation.trajectory", sample_dt)
        if set(oracle_track) != set(baseline_track):
            raise RuntimeError(f"seed {seed}: oracle episode set differs")
        oracle_summary = summarise(baseline_track, oracle_track, None)
        oracle_summary["arm"] = "run_episode_with_oracle / fin_accel_command"
        oracle_summary["authority"] = oracle_authority(oracle_dir / "validation.trajectory")
        oracle_summary["sha256"] = {
            "baseline": baseline_shas[str(seed)],
            "assisted": sha256(oracle_dir / "validation.trajectory"),
        }
        seed_results[str(seed)][ORACLE] = oracle_summary

        for arm in ARMS:
            metrics_by_seed, _, arm_blockers = arm_evidence[arm]
            if str(seed) not in metrics_by_seed:
                continue
            export_path = arm_export(arm, seed)
            checkpoint = torch.load(export_path, map_location="cpu", weights_only=True)
            model = GDN2Student(GDN2StudentConfig(seed=seed))
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            prediction = commands_for_export(model, baseline.episodes)
            tsv_dir = work / arm / "commands"
            control = build_controls(baseline, prediction, tsv_dir)
            flown_dir = work / arm / "flown"
            run_helper(seed, indices, flown_dir, commands_dir=tsv_dir)
            flown_track = read_track(flown_dir / "validation.trajectory", sample_dt)
            if set(flown_track) != set(baseline_track):
                raise RuntimeError(f"seed {seed} {arm}: episode set differs")
            for idx in indices:
                if flown_track[idx]["ticks"] != baseline_track[idx]["ticks"]:
                    raise RuntimeError(f"seed {seed} {arm}: tick count changed at {idx}")
            summary = summarise(baseline_track, flown_track, control)
            summary["arm"] = ARM_LABELS[arm]
            summary["sha256"] = {
                "baseline": baseline_shas[str(seed)],
                "assisted": sha256(flown_dir / "validation.trajectory"),
            }
            summary["export_sha256"] = sha256(export_path)
            seed_results[str(seed)][arm] = summary
            del model
            agg = summary["aggregate"]
            print(
                f"seed {seed} {arm}: deployment lateral "
                f"{agg['baseline']['deployment_lateral_m']:.1f} -> "
                f"{agg['assisted']['deployment_lateral_m']:.1f} m; apogee "
                f"{agg['baseline']['apogee_m']:.1f} -> {agg['assisted']['apogee_m']:.1f} m",
                flush=True,
            )

    provenance = {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_head": git_head(),
        "config": str(CONFIG.relative_to(ROOT)),
        "config_sha256": sha256(CONFIG),
        "config_hash": sha256(CONFIG),
        "heldout": str(HELDOUT.relative_to(ROOT)),
        "heldout_sha256": sha256(HELDOUT),
        "heldout_manifest_sha256": sha256(MANIFEST),
        "quality_report_sha256": sha256(QUALITY),
        "night_root": str(NIGHT.relative_to(ROOT)),
        "summary_sha256": sha256(SUMMARY),
        "adversarial_summary_sha256": sha256(ADV_SUMMARY),
        "export_manifest_sha256": sha256(EXPORT_MANIFEST),
        "shield_sha256": sha256(SHIELD),
        "harness": "scripts/ab_v6.py",
        "harness_sha256": sha256(Path(__file__)),
        "helper": str(HELPER_SRC.relative_to(ROOT)),
        "helper_sha256": sha256(HELPER_SRC),
        "helper_binary": str(HELPER.relative_to(ROOT)),
        "helper_binary_sha256": sha256(HELPER),
        "dynamics_sha256": sha256(DYNAMICS_SRC),
        "dynamics_basis": (
            "post-FIXPUMP: aerodynamic normal force does no positive work (584b1dc); "
            "seed-1 zero-command sidecar is bit-identical to the frozen held-out trajectory"
        ),
        "baseline_trajectory_sha256": baseline_shas,
        "arms": list(ARMS),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "sample_dt_s": sample_dt,
        "command_limit_mps2": COMMAND_LIMIT,
        "deployment_proxy": "first DESCENT tick (vz<0 and z>=5), radial distance from pad",
        "lateral_limit_m": LATERAL_LIMIT_M,
        "apogee_limit_m": APOGEE_LIMIT_M,
    }

    arms_report: dict[str, dict] = {}
    all_blockers: list[str] = []
    for arm in ARMS:
        metrics_by_seed, evidence, arm_blockers = arm_evidence[arm]
        arm_rep = arm_report(
            arm, seed_order, indices, seed_results, metrics_by_seed, evidence, provenance
        )
        arms_report[arm] = evaluate_arm(arm_rep, metrics_by_seed, seed_order, arm_blockers)
        all_blockers += [f"{arm}: {b}" for b in arm_blockers]

    overall = all(arms_report[arm]["verdict"]["pass"] for arm in ARMS) and not all_blockers
    report = {
        "measurement": "POTOKTRJ v3 flown-truth trajectory sidecar (stride 26), not surrogate",
        "sim_source": (
            "read-only scripts/ab_control_v3 on git "
            + provenance["git_head"]
            + "; oracle uses run_episode_with_oracle / fin_accel_command"
        ),
        "supersedes": (
            "any win reading of the RETRAIN-NIGHT server-side RMSE comparison: it was never "
            "a closed-loop A/B and is void here; the unweakened pre-registered gates are restored"
        ),
        "held_out_indices": indices,
        "seed_order": seed_order,
        "students": [{"name": arm, "label": ARM_LABELS[arm]} for arm in ARMS],
        "arms": arms_report,
        "provenance": provenance,
        "verdict": {
            "status": "PASS" if overall else "FAIL",
            "pass": overall,
            "win_claimed": overall,
            "per_arm": {arm: arms_report[arm]["verdict"]["status"] for arm in ARMS},
        },
        "blockers": all_blockers,
        "exit_code": 0 if overall else 1,
        "honesty_notes": [
            (
                "Simulator-only: the POTOKTRJ sidecar is emitted by the same simulator that flies "
                "the episodes; it is not an independent flight truth anchor."
            ),
            (
                "No canopy/deployment event is scored; the proxy is the first DESCENT tick and is "
                "not a landing-dispersion measurement."
            ),
            (
                "Commands are generated from the baseline flight's recorded frames and replayed by "
                "frame index; after an assisted command moves the vehicle, later frames differ from "
                "the baseline flight while the phase/gate schedule stays paired. Commands are "
                "emitted only where the baseline ai_healthy target is true, so boost-phase flight "
                "stays identical to the zero-command arm."
            ),
            (
                "The distill target is the applied residual decision (baseline + gain*correction, "
                "box projected); deployment adds the export's correction to physics_baseline and "
                "clips only at the seam's hard +/-10 m/s^2 bound. No confidence gate, gain, weight "
                "or physics constant was introduced or tuned."
            ),
            (
                "Seeds 2/3 use seed-matched exports on regenerated held-out flights; a missing "
                "export is a blocker, never a stand-in."
            ),
            (
                "The two arms are separate opt-in retrains of the same four-stage teacher-v2 "
                "default; adversarial differs only by --adversarial. Each arm carries its own "
                "gates and verdict. No arm-to-arm winner is declared: both are compared to the "
                "paired zero-command control under the same unweakened gates."
            ),
            (
                "The helper binary was not rebuilt in this shared checkout; its seed-1 zero-command "
                "flight is checked bit-for-bit against the frozen post-FIXPUMP held-out sidecar."
            ),
            (
                "provenance.git_head is the commit that was checked out when the harness ran, "
                "before this harness and report were committed locally; no remote action was taken."
            ),
            "No win is claimed unless every gate of every arm passes. A FAIL is a measured miss, not a weaker gate.",
        ],
    }

    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.out_md.write_text(markdown(report))
    print(f"wrote {args.out_json.relative_to(ROOT)} and {args.out_md.relative_to(ROOT)}")
    for arm in ARMS:
        v = arms_report[arm]["verdict"]
        print(
            f"verdict {arm}: {v['status']} (gates failed {v['gates_failed']}, "
            f"blockers {len(arms_report[arm]['blockers'])})"
        )
    print(
        f"overall: {report['verdict']['status']} (exit {report['exit_code']}), "
        f"blockers={len(all_blockers)}"
    )
    sys.exit(report["exit_code"])


if __name__ == "__main__":
    main()
