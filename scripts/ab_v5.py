#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pre-registered v5 closed-loop A/B: FINAL all-fix four-stage exports.

Flies the 20 held-out scenarios (16..19) through the read-only v3 control seam on
post-FIXPUMP dynamics. Zero command is the control; ``run_episode_with_oracle``
is the positive control; the primary arm is the shipped distill export of the
FINAL all-fix four-stage (sft -> privileged -> dagger -> distill) teacher in
``ai/out/final_v3_allfix`` (AIFIX1 bound, AIFIX2 truth sidecar, AIFIX3 on-policy
DAgger, SHIELD01 module, SHIELD02 residual policy).

Gates are fixed here, never fitted and never weakened (AIPOST1):
deployment-proxy lateral RMS <= 1000 m and apogee >= 2000 m on every seed, ECE
with an explicit disjoint source, provenance from metrics and artifact hashes,
and the verbatim unweakened EVAL01 suite (-35% / +15%, 1.25, 0.05, 3x spread).
No physics, gate, weight or tuning knob is changed here. No win is claimed
unless every gate passes.

Usage:
    ai/.venv/bin/python scripts/ab_v5.py
Work files live outside the repo (default /tmp/opencode/ab_v5).
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
ALLFIX = ROOT / "ai" / "out" / "final_v3_allfix"
EXPORTS = ALLFIX / "exports"
EXPORT_MANIFEST = EXPORTS / "manifest.json"
RUNS = ALLFIX / "runs"
SUMMARY = ALLFIX / "summary.json"
SHIELD = ALLFIX / "shield.json"
HELPER = SCRIPTS / "ab_control_v3" / "target" / "release" / "ab-control-v3"
HELPER_SRC = SCRIPTS / "ab_control_v3" / "src" / "main.rs"
DYNAMICS_SRC = ROOT / "sim" / "core" / "src" / "dynamics.rs"
WORK = Path("/tmp/opencode/ab_v5")
OUT_JSON = SCRIPTS / "ab_report_v5.json"
OUT_MD = SCRIPTS / "ab_report_v5.md"

SEEDS = (1, 2, 3)
ARM = "teacher_v2"
ORACLE = "oracle_gtheta"
STAGES = ("sft", "privileged", "dagger", "distill")
BOUND_MPS2 = 10.0  # allfix AIFIX1 student command bound; evidence, not a tuning knob
LATERAL_LIMIT_M = 1000.0  # pre-registered: deployment-proxy lateral RMS ceiling
APOGEE_LIMIT_M = 2000.0  # pre-registered: apogee floor
COMMAND_LIMIT = 10.0  # sim/core/src/control.rs MAX_COMMAND_MPS2; not a tuning knob
ULP_M = float(np.spacing(np.float32(600.0)))
SIDECAR_MAGIC = b"POTOKTRJ"
SIDECAR_STRIDE = 26


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


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


def markdown(report: dict) -> str:
    out = [
        "# v5 closed-loop A/B — unweakened gates, FINAL all-fix four-stage exports",
        "",
        (
            f"Verdict: **{report['verdict']['status']}** (exit {report['exit_code']}). "
            "This is a post-FIXPUMP simulator measurement, not a flight claim. "
            "No win is claimed unless every gate passes."
        ),
        "",
        "## Gates (hard-coded, unweakened)",
        "",
        "| gate | result | detail |",
        "|---|---|---|",
    ]
    for row in report["gates"]:
        out.append(f"| {row['name']} | {'PASS' if row['passed'] else 'FAIL'} | {row['detail']} |")
    out += [
        "",
        (
            "Deployment proxy = first `DESCENT` tick (vz<0 and z>=5) in each flown POTOKTRJ v3 "
            "sidecar; the gate uses the pooled RMS of its radial distance from the pad over the "
            "20 held-out episodes (LATERAL01.md:116-127). Apogee is the max z over the whole "
            "30 s trajectory. The repo has no independently scored canopy; nothing here is a "
            "landing-dispersion number."
        ),
        "",
        "## Per-seed aggregate (metres)",
        "",
        "| seed | arm | deployment lateral | apogee | whole-trajectory lateral RMS |",
        "|---:|---|---:|---:|---:|",
    ]
    for seed in report["seed_order"]:
        arms = report["seeds"][str(seed)]
        for name in (ORACLE, ARM):
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
    out += ["", "## Positive control", ""]
    pc = report["positive_control"]
    out.append(f"- passed: **{pc['passed']}**")
    for seed, row in pc["per_seed"].items():
        out.append(
            f"- seed {seed}: deployment lateral {row['deployment_delta_m']:+.2f} m, "
            f"apogee {row['apogee_delta_m']:+.2f} m, "
            f"actuator cap ticks {row['authority']['cap_hit_ticks']}/"
            f"{row['authority']['active_ticks']}"
        )
    out += ["", "## Provenance", "", "| field | value |", "|---|---|"]
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
        SHIELD,
        QUALITY,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing {path.relative_to(ROOT)}")

    config = tomllib.loads(CONFIG.read_text())
    sample_dt = 1.0 / config["run"]["frame_rate_hz"]
    manifest = json.loads(MANIFEST.read_text())
    exports_manifest = json.loads(EXPORT_MANIFEST.read_text())
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
    if tuple(exports_manifest["stage_order"]) != STAGES:
        raise ValueError(
            f"exports are not the four-stage default: {exports_manifest['stage_order']}"
        )
    if exports_manifest.get("claim", "").find("no win is claimed") < 0:
        raise ValueError("export manifest no longer disclaims a win")

    summary_allfix = json.loads(SUMMARY.read_text())
    shield_allfix = json.loads(SHIELD.read_text())
    if summary_allfix.get("ab_win_claimed") is not False:
        raise ValueError("allfix summary claims an A/B win; refusing to treat exports as artifacts")
    if summary_allfix.get("rlvr_used") is not False:
        raise ValueError("allfix summary says RLVR was used; arm identity changed")
    if tuple(summary_allfix.get("stage_order", ())) != STAGES:
        raise ValueError("allfix summary stage_order disagrees with the four-stage default")
    if exports_manifest.get("fixes", {}).get("command_bound") != BOUND_MPS2:
        raise ValueError("allfix export manifest no longer pins the +/-10 command bound")
    if shield_allfix.get("authority") != BOUND_MPS2:
        raise ValueError("allfix shield authority disagrees with the command bound")
    if summary_allfix.get("shield_verification", {}).get("counts", {}).get("accepted") != 1:
        raise ValueError("allfix shield verification no longer has its counted accept case")
    frozen_dp = summary_allfix.get("data_provenance", {})
    if frozen_dp.get("heldout_sha256") != sha256(HELDOUT):
        raise ValueError("allfix summary heldout hash disagrees with the flown held-out set")
    if frozen_dp.get("config_sha256") != sha256(CONFIG):
        raise ValueError("allfix summary config hash disagrees with the frozen sim config")

    seed_order = sorted(args.seeds)
    if seed_order != list(SEEDS):
        raise ValueError(f"pre-registered gate needs seeds {list(SEEDS)}, got {seed_order}")
    blockers: list[str] = []
    report_seeds: dict[str, dict] = {}
    metrics_by_seed: dict[str, dict] = {}
    baseline_shas: dict[str, str] = {}

    for seed in seed_order:
        export_dir = EXPORTS / f"teacher_v2_s{seed}.pt"
        metrics_path = RUNS / f"s{seed}" / "metrics.json"
        declared = exports_manifest["students"].get(str(seed))
        if not export_dir.is_file() or not metrics_path.is_file() or declared is None:
            blockers.append(f"seed {seed}: missing export/metrics; no stand-in substituted")
            continue
        if sha256(export_dir) != declared["sha256"]:
            raise ValueError(f"seed {seed}: export sha256 disagrees with export manifest")
        metrics = json.loads(metrics_path.read_text())
        if metrics.get("config_hash") != declared["config_hash"]:
            raise ValueError(f"seed {seed}: metrics config_hash disagrees with export manifest")
        if metrics.get("seed") != seed:
            raise ValueError(f"seed {seed}: metrics seed field mismatch")
        if declared.get("file") != export_dir.name:
            raise ValueError(f"seed {seed}: export manifest names a different student file")
        if metrics.get("ece_source") != declared.get("ece_source"):
            raise ValueError(f"seed {seed}: metrics ece_source disagrees with export manifest")
        if metrics.get("rmse_held_out") != declared.get("rmse_held_out"):
            raise ValueError(f"seed {seed}: metrics held-out RMSE disagrees with export manifest")
        if metrics.get("allfix_provenance") is not True:
            raise ValueError(f"seed {seed}: metrics do not carry allfix_provenance")
        if metrics.get("student_command_bound") != BOUND_MPS2:
            raise ValueError(f"seed {seed}: student command bound is not the allfix +/-10")
        if metrics.get("legacy_student_bound") is not False:
            raise ValueError(f"seed {seed}: legacy student bound is still enabled")
        if metrics.get("residual_policy", {}).get("enabled") is not True:
            raise ValueError(f"seed {seed}: SHIELD02 residual policy is not enabled")
        distill_details = metrics.get("stages", {}).get("distill", {}).get("details", {})
        if distill_details.get("residual_policy_target") is not True:
            raise ValueError(f"seed {seed}: distill target is not the residual decision")
        dagger_details = metrics.get("stages", {}).get("dagger", {}).get("details", {})
        if dagger_details.get("rollout_source") != "on_policy_student:gdn2_student":
            raise ValueError(f"seed {seed}: DAgger rollout is not the AIFIX3 on-policy student")
        privileged_details = metrics.get("stages", {}).get("privileged", {}).get("details", {})
        if privileged_details.get("privileged_batches") != 2300:
            raise ValueError(f"seed {seed}: AIFIX2 privileged batches are missing")
        if metrics.get("ece") is None or metrics.get("ece_held_out") is None:
            raise ValueError(f"seed {seed}: ECE fields missing; no gate value invented")
        metrics_by_seed[str(seed)] = metrics

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

        report_seeds[str(seed)] = {}
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
        report_seeds[str(seed)][ORACLE] = oracle_summary

        export_path = EXPORTS / f"teacher_v2_s{seed}.pt"
        if not export_path.is_file() or str(seed) not in metrics_by_seed:
            continue
        checkpoint = torch.load(export_path, map_location="cpu", weights_only=True)
        model = GDN2Student(GDN2StudentConfig(seed=seed))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        prediction = commands_for_export(model, baseline.episodes)
        tsv_dir = work / ARM / "commands"
        control = build_controls(baseline, prediction, tsv_dir)
        flown_dir = work / ARM / "flown"
        run_helper(seed, indices, flown_dir, commands_dir=tsv_dir)
        flown_track = read_track(flown_dir / "validation.trajectory", sample_dt)
        if set(flown_track) != set(baseline_track):
            raise RuntimeError(f"seed {seed} {ARM}: episode set differs")
        for idx in indices:
            if flown_track[idx]["ticks"] != baseline_track[idx]["ticks"]:
                raise RuntimeError(f"seed {seed} {ARM}: tick count changed at {idx}")
        summary = summarise(baseline_track, flown_track, control)
        summary["arm"] = "FINAL all-fix four-stage distill export, residual command, seam-clipped"
        summary["sha256"] = {
            "baseline": baseline_shas[str(seed)],
            "assisted": sha256(flown_dir / "validation.trajectory"),
        }
        summary["export_sha256"] = sha256(export_path)
        report_seeds[str(seed)][ARM] = summary
        del model
        agg = summary["aggregate"]
        print(
            f"seed {seed}: deployment lateral "
            f"{agg['baseline']['deployment_lateral_m']:.1f} -> "
            f"{agg['assisted']['deployment_lateral_m']:.1f} m; apogee "
            f"{agg['baseline']['apogee_m']:.1f} -> {agg['assisted']['apogee_m']:.1f} m",
            flush=True,
        )

    # The EVAL01 suite is imported verbatim; no threshold was copied or relaxed.
    evals: dict[str, bool] = {}
    for name, fn in (
        ("rmse_ratio", teacher_gates.check_ratio),
        ("ece_source", teacher_gates.check_ece_source),
        ("ece", teacher_gates.check_ece),
    ):
        if not metrics_by_seed:
            blockers.append("no metrics for EVAL01 gate " + name)
            evals[name] = False
            continue
        try:
            result = fn(metrics_by_seed)
        except teacher_gates.ProvenanceError as exc:
            blockers.append(f"EVAL01 {name}: {exc}")
            evals[name] = False
            continue
        evals[name] = result.passed
        if not result.passed:
            blockers.append(f"EVAL01 {name} failed: {result.detail}")

    positive_control: dict = {"per_seed": {}, "passed": False}
    for seed in seed_order:
        if ORACLE not in report_seeds.get(str(seed), {}):
            blockers.append(f"seed {seed}: oracle missing; positive control not measured")
            continue
        entry = report_seeds[str(seed)][ORACLE]
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
        blockers.append("oracle positive control did not improve both metrics on all seeds")

    primary_gates: list[dict] = []
    for seed in seed_order:
        entry = report_seeds.get(str(seed), {}).get(ARM)
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
        "allfix_root": str(ALLFIX.relative_to(ROOT)),
        "exports_dir": str(EXPORTS.relative_to(ROOT)),
        "export_manifest_sha256": sha256(EXPORT_MANIFEST),
        "shield_sha256": sha256(SHIELD),
        "shield_json": str(SHIELD.relative_to(ROOT)),
        "exports_sha256": {
            f"teacher_v2_s{seed}.pt": sha256(EXPORTS / f"teacher_v2_s{seed}.pt")
            for seed in seed_order
            if (EXPORTS / f"teacher_v2_s{seed}.pt").is_file()
        },
        "metrics_sha256": {
            f"s{seed}/metrics.json": sha256(RUNS / f"s{seed}" / "metrics.json")
            for seed in seed_order
            if (RUNS / f"s{seed}" / "metrics.json").is_file()
        },
        "metrics_config_hash": {
            str(seed): metrics_by_seed[str(seed)]["config_hash"] for seed in metrics_by_seed
        },
        "residual_gain": {
            str(seed): metrics_by_seed[str(seed)].get("residual_policy", {}).get("gain")
            for seed in metrics_by_seed
        },
        "stage_order": list(STAGES),
        "summary_sha256": sha256(SUMMARY),
        "summary_json": str(SUMMARY.relative_to(ROOT)),
        "summary_ab_win_claimed": summary_allfix.get("ab_win_claimed"),
        "summary_rlvr_used": summary_allfix.get("rlvr_used"),
        "allfix_data_provenance": frozen_dp,
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
        "student_command_bound": BOUND_MPS2,
        "legacy_student_bound": False,
        "residual_policy_enabled": {
            str(seed): metrics_by_seed[str(seed)]["residual_policy"]["enabled"]
            for seed in metrics_by_seed
        },
        "privileged_batches": {
            str(seed): metrics_by_seed[str(seed)]["privileged_batches"] for seed in metrics_by_seed
        },
        "harness": "scripts/ab_v5.py",
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
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "sample_dt_s": sample_dt,
        "command_limit_mps2": COMMAND_LIMIT,
        "deployment_proxy": "first DESCENT tick (vz<0 and z>=5), radial distance from pad",
        "lateral_limit_m": LATERAL_LIMIT_M,
        "apogee_limit_m": APOGEE_LIMIT_M,
    }

    report = {
        "measurement": "POTOKTRJ v3 flown-truth trajectory sidecar (stride 26), not surrogate",
        "sim_source": (
            "read-only scripts/ab_control_v3 on git "
            + provenance["git_head"]
            + "; oracle uses run_episode_with_oracle / fin_accel_command"
        ),
        "supersedes": (
            "any win reading of the RETRAIN-ALLFIX summary's surrogate-only RMSE comparison: "
            "it was never a closed-loop A/B and is void here; the unweakened pre-registered "
            "gates are restored verbatim"
        ),
        "held_out_indices": indices,
        "seed_order": seed_order,
        "students": [
            {
                "name": ARM,
                "family": "gdn2_student (FINAL all-fix four-stage distill, +/-10 head, residual+shield)",
                "export": str((EXPORTS / "teacher_v2_s1.pt").relative_to(ROOT)),
                "stage_order": list(STAGES),
            }
        ],
        "verdict": {"status": "PENDING", "primary": ARM},
        "provenance": provenance,
        "seeds": report_seeds,
    }

    # Now that the provenance shape exists, run the EVAL01 provenance gate verbatim.
    try:
        result = teacher_gates.check_provenance(report)
        evals["provenance"] = result.passed
        if not result.passed:
            blockers.append(f"EVAL01 provenance failed: {result.detail}")
    except teacher_gates.ProvenanceError as exc:
        blockers.append(f"EVAL01 provenance: {exc}")
        evals["provenance"] = False

    ab_for_eval = {
        **report,
        "seeds": report_seeds,
        "verdict": {**report["verdict"], "primary": ARM},
    }
    try:
        spread = teacher_gates.check_seed_spread(ab_for_eval, student=ARM)
        improvement = teacher_gates.check_improvement(ab_for_eval, student=ARM)
        oracle_check = teacher_gates.check_oracle(ab_for_eval)
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
        "primary": ARM,
        "pass": passed,
        "win_claimed": passed,
        "absolute_gates": all(row["passed"] for row in primary_gates),
        "eval01_gates": evals,
        "oracle_positive_control": positive_control["passed"],
    }
    report["seeds"] = report_seeds
    report["exit_code"] = 0 if passed else 1
    report["honesty_notes"] = [
        (
            "Simulator-only: the POTOKTRJ sidecar is emitted by the same simulator that flies the "
            "episodes; it is not an independent flight truth anchor."
        ),
        (
            "No canopy/deployment event is scored; the proxy is the first DESCENT tick and is not "
            "a landing-dispersion measurement."
        ),
        (
            "Commands are generated from the baseline flight's recorded frames and replayed by "
            "frame index; after an assisted command moves the vehicle, later frames differ from "
            "the baseline flight while the phase/gate schedule stays paired. Commands are "
            "emitted only where the baseline ai_healthy target is true, so boost-phase flight "
            "stays identical to the zero-command arm."
        ),
        (
            "The distill target is the applied residual decision (baseline + gain*correction, box "
            "projected); deployment adds the export's correction to physics_baseline and clips "
            "only at the seam's hard +/-10 m/s^2 bound. No confidence gate, gain, weight or "
            "physics constant was introduced or tuned."
        ),
        (
            "Seeds 2/3 use the seed-matched exports on regenerated held-out flights; a missing "
            "export is a blocker, never a stand-in."
        ),
        (
            "Primary arm = ai/out/final_v3_allfix/exports/teacher_v2_s{1,2,3}.pt, the FINAL "
            "all-fix four-stage distill exports (AIFIX1/2/3 + SHIELD01/02). The harness refuses "
            "to run unless the export manifest, summary, shield and per-seed metrics still carry "
            "that evidence; no older export set is substituted."
        ),
        (
            "The all-fix fixes are wiring/provenance fixes validated on a simulator surrogate: "
            "world_step and the POTOKTRJ sidecar share a source. A pass here is a simulator "
            "measurement under the pre-registered gates, not an independent flight-truth claim."
        ),
        (
            "The helper binary was not rebuilt in this shared checkout; its seed-1 zero-command "
            "flight is checked bit-for-bit against the frozen post-FIXPUMP held-out sidecar."
        ),
        (
            "provenance.git_head is the commit that was checked out when the harness ran, before "
            "this harness and report were committed locally; no remote action was taken."
        ),
        "No win is claimed: every gate must pass. A FAIL is a measured miss, not a weaker gate.",
    ]

    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.out_md.write_text(markdown(report))
    print(f"wrote {args.out_json.relative_to(ROOT)} and {args.out_md.relative_to(ROOT)}")
    print(
        f"verdict: {report['verdict']['status']} (exit {report['exit_code']}), "
        f"gates failed={sum(1 for g in gates if not g['passed'])}, blockers={len(blockers)}"
    )
    sys.exit(report["exit_code"])


if __name__ == "__main__":
    main()
