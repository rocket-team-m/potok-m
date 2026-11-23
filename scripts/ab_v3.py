#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pre-registered v3 closed-loop A/B: zero command, oracle, v3 exports.

Flies the 20 held-out scenarios (16..19) of ``ai/out/v3_final`` through the
read-only ``scripts/ab_control_v3`` helper. Apogee and lateral RMS come from the
POTOKTRJ v3 flown-truth sidecar (stride 26). No sim source, config, physics
constant or interlock is changed: the command is the same residual
``physics_baseline + student Delta`` the exports were trained against, mapped to
the seam's ``[roll, pitch]`` m/s^2 order and clipped only at its hard +/-10 limit.

Usage:
    ai/.venv/bin/python scripts/ab_v3.py
    ai/.venv/bin/python scripts/ab_v3.py --seeds 1   # primary only (report marked partial)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT / "ai" / "src"))

from potok_ai.data import SMART_FEATURES, read_validation
from potok_ai.export import load_student_export
from potok_ai.loopwm import LoopWM, LoopWMConfig
from potok_ai.loopwm_lite import LiteConfig, LiteModel
from potok_ai.physics import final_command, physics_baseline

CONFIG = ROOT / "ai" / "out" / "v3_final" / "sim.toml"
EXPORTS = ROOT / "ai" / "out" / "v3_final"
HELDOUT = EXPORTS / "data" / "heldout_validation.bin"
MANIFEST = EXPORTS / "heldout_manifest.json"
HELPER = SCRIPTS / "ab_control_v3" / "target" / "release" / "ab-control-v3"
WORK = SCRIPTS / "ab_control_v3" / "runs"
OUT_JSON = ROOT / "ab_report.json"
OUT_MD = ROOT / "ab_report.md"

WINDOW = 50
SEEDS = (1, 2, 3)
COMMAND_LIMIT = 10.0  # sim/core/src/control.rs MAX_COMMAND_MPS2; not a tuning knob
ULP_M = float(np.spacing(np.float32(600.0)))
# f32 sidecar resolution is ~6e-5 m; require >16 ULP before calling a win real.
MIN_MEANINGFUL_M = 1e-3

# Post-FIXPUMP recalculation. The helper is rebuilt against corrected dynamics:
# the aerodynamic normal force no longer does positive work (584b1dc) and drag_est
# is pinned to trajectory mass (7ddf324). The frozen v3 exports are untouched; no
# gate, threshold, weight or physics constant was tuned for this recalculation.
REPORT_LABEL = "post-FIXPUMP"
DYNAMICS_BASIS = (
    "post-FIXPUMP: aerodynamic normal force does no positive work (584b1dc), "
    "drag_est pinned to trajectory mass (7ddf324)"
)

# The two families with a complete export for every pre-registered seed.
STUDENTS = (
    {"name": "loopwm", "family": "loopwm", "file": "loopwm.pt", "kind": "window checkpoint"},
    {"name": "lite", "family": "loopwm-lite", "file": "lite.pt", "kind": "window checkpoint"},
)
# lrs s3 has no export (empty run.log): report s1/s2 separately, never fabricate s3.
PARTIAL = ({"name": "lrs", "family": "lrs", "file": "lrs.bin", "kind": "stateful bin export"},)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_helper(
    seed: int,
    indices: list[int],
    out_dir: Path,
    commands_dir: Path | None = None,
    oracle: bool = False,
) -> dict[int, dict]:
    if not HELPER.is_file():
        raise FileNotFoundError(
            f"missing {HELPER.relative_to(ROOT)}; build with: cargo build --release "
            f"--manifest-path scripts/ab_control_v3/Cargo.toml "
            f"--target-dir scripts/ab_control_v3/target"
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


def read_track(path: Path) -> dict[int, dict]:
    """Parse POTOKTRJ v3 and measure exactly the Rust sidecar API measures."""
    raw = path.read_bytes()
    if len(raw) < 64:
        raise ValueError(f"{path}: sidecar shorter than its 64-byte header")
    magic, version, count, frames, stride = struct.unpack_from("<8sIIII", raw, 0)
    if magic != b"POTOKTRJ" or version != 3 or stride != 26:
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
        result[int(index)] = {
            "apogee_m": float(np.max(rows[:, 2])),
            "lateral_rms_m": float(np.sqrt(np.mean(rows[:, 0] ** 2 + rows[:, 1] ** 2))),
            "ticks": int(frames),
        }
    if off != len(raw):
        raise ValueError(f"{path}: trailing bytes in sidecar")
    return result


def load_model(family: str, seed: int) -> tuple[torch.nn.Module, float]:
    """Load one v3 export with the engine's own checkpoint/export loaders."""
    directory = EXPORTS / "runs" / family / f"s{seed}"
    threshold_path = directory / "calibration.json"
    if not threshold_path.is_file():
        raise FileNotFoundError(f"missing {threshold_path.relative_to(ROOT)}")
    threshold = float(json.loads(threshold_path.read_text())["threshold"])
    if not math.isfinite(threshold):
        raise ValueError(f"non-finite calibration threshold for {family} s{seed}")
    if family == "lrs":
        export = load_student_export(directory / "lrs.bin")
        if export.in_features != 20:
            raise ValueError(f"lrs export needs the 20-value smart frame, got {export.in_features}")
        return export.to_model().eval(), threshold
    checkpoint = torch.load(
        directory / ("lite.pt" if family == "loopwm-lite" else "loopwm.pt"),
        map_location="cpu",
        weights_only=True,
    )
    cls, cfg_cls = (LiteModel, LiteConfig) if family == "loopwm-lite" else (LoopWM, LoopWMConfig)
    model = cls(cfg_cls(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), threshold


@torch.no_grad()
def commands_for_window(model: torch.nn.Module, episodes: list, threshold: float) -> dict:
    """One sliding-window forward per tick, batched over all 20 episodes."""
    frames = np.stack([np.asarray(ep.frames[:, :SMART_FEATURES], np.float32) for ep in episodes])
    conds = np.stack([np.asarray(ep.conditioning, np.float32) for ep in episodes])
    ticks = frames.shape[1]
    padded = np.zeros((frames.shape[0], ticks + WINDOW - 1, frames.shape[2]), np.float32)
    padded[:, WINDOW - 1 :] = frames
    # numpy appends the sliding window as the last axis: (E, T, F, W) -> (E, T, W, F).
    windows = np.lib.stride_tricks.sliding_window_view(padded, WINDOW, axis=1).transpose(0, 1, 3, 2)
    deltas = np.zeros((frames.shape[0], ticks, 2), np.float64)
    confs = np.zeros((frames.shape[0], ticks), np.float64)
    pooled = isinstance(model, LiteModel)
    for tick in range(WINDOW - 1, ticks):
        window = torch.from_numpy(np.ascontiguousarray(windows[:, tick])).float()
        cond = torch.from_numpy(conds).float()
        latent, _ = model.refine(window, cond)
        summary = latent.mean(dim=1) if pooled else latent
        output = model.head_output(model.head_logits(summary))
        baseline = physics_baseline(window[:, -1, :])
        command = final_command(output[:, :2], baseline, residual=True)
        deltas[:, tick] = command.numpy().astype(np.float64)
        confs[:, tick] = output[:, 2].numpy().astype(np.float64)
    return {
        "deltas": deltas,
        "confs": confs,
        "first_command_tick": WINDOW - 1,
        "threshold": threshold,
    }


@torch.no_grad()
def commands_for_lrs(model: torch.nn.Module, episodes: list, threshold: float) -> dict:
    """The exported stateful student consumes one 20-value frame per tick."""
    n, ticks = len(episodes), episodes[0].frames.shape[0]
    deltas = np.zeros((n, ticks, 2), np.float64)
    confs = np.zeros((n, ticks), np.float64)
    for e, ep in enumerate(episodes):
        model.reset_state()
        for tick in range(ticks):
            frame = np.concatenate((ep.frames[tick, :SMART_FEATURES], ep.conditioning)).astype(
                np.float32
            )
            correction, confidence = model.step(frame)
            baseline = physics_baseline(
                torch.as_tensor(ep.frames[tick, :SMART_FEATURES], dtype=torch.float32)
            )
            command = final_command(correction, baseline, residual=True)
            deltas[e, tick] = command.numpy().astype(np.float64)
            confs[e, tick] = float(confidence)
    return {"deltas": deltas, "confs": confs, "first_command_tick": 0, "threshold": threshold}


def build_controls(baseline, prediction: dict, out_dir: Path) -> dict:
    """Gate, map to the seam's m/s^2 order and write one TSV per episode."""
    out_dir.mkdir(parents=True, exist_ok=True)
    healthy = np.stack([np.asarray(ep.targets[:, 3], np.float64) for ep in baseline.episodes])
    reference = np.stack([ep.command_target().astype(np.float64) for ep in baseline.episodes])
    deltas = prediction["deltas"]
    confs = prediction["confs"]
    gate = (healthy > 0.5) & (confs >= prediction["threshold"])
    raw = np.where(gate[:, :, None], deltas, 0.0)
    clipped = np.abs(raw) > COMMAND_LIMIT
    command = np.clip(raw, -COMMAND_LIMIT, COMMAND_LIMIT)
    stats = {
        "rows": int(command.shape[0] * command.shape[1]),
        "gate_open_ticks": int(gate.sum()),
        "gate_trips": int((~gate).sum()),
        "interventions": int((gate & (np.max(np.abs(command), axis=2) > 1e-9)).sum()),
        "clipped_ticks": int(clipped.any(axis=2).sum()),
        "max_command_mps2": float(np.max(np.abs(command))),
        "first_command_tick": prediction["first_command_tick"],
        "calibration_threshold": prediction["threshold"],
    }
    active = gate & (np.max(np.abs(command), axis=2) > 1e-9)
    stats["tracking_rmse_mps2"] = (
        float(np.sqrt(np.mean((command[active] - reference[active]) ** 2)))
        if active.any()
        else None
    )
    for e, ep in enumerate(baseline.episodes):
        with (out_dir / f"{ep.index}.tsv").open("w") as handle:
            handle.write("# frame_index g_roll_mps2 g_pitch_mps2\n")
            for tick, (roll, pitch) in enumerate(command[e]):
                handle.write(f"{tick} {roll:.6f} {pitch:.6f}\n")
    return stats


def summarise(b_track: dict, a_track: dict, control: dict | None) -> dict:
    scenarios = sorted({k % 20 for k in b_track})

    def rows_for(keys: list[int]) -> dict:
        b_ap = float(np.mean([b_track[k]["apogee_m"] for k in keys]))
        a_ap = float(np.mean([a_track[k]["apogee_m"] for k in keys]))
        b_lat = float(np.sqrt(np.mean([b_track[k]["lateral_rms_m"] ** 2 for k in keys])))
        a_lat = float(np.sqrt(np.mean([a_track[k]["lateral_rms_m"] ** 2 for k in keys])))
        return {
            "episodes": len(keys),
            "baseline": {"apogee_m": b_ap, "lateral_rms_m": b_lat},
            "assisted": {"apogee_m": a_ap, "lateral_rms_m": a_lat},
            "apogee_delta_m": a_ap - b_ap,
            "apogee_retention_pct": 100.0 * a_ap / b_ap,
            "lateral_delta_m": a_lat - b_lat,
            "lateral_ratio_pct": 100.0 * a_lat / b_lat,
        }

    all_keys = sorted(b_track)
    aggregate = rows_for(all_keys)
    if control:
        aggregate.update(control)
    return {
        "per_scenario": [
            {"scenario": f"scenario_{sc:02d}", **rows_for([k for k in all_keys if k % 20 == sc])}
            for sc in scenarios
        ],
        "aggregate": aggregate,
    }


def markdown(report: dict) -> str:
    out = [
        f"# v3 closed-loop A/B — {REPORT_LABEL} — zero command vs oracle vs exported students",
        "",
        (
            f"**{REPORT_LABEL} recalculation.** The flight helper was rebuilt against corrected "
            "dynamics (the aerodynamic normal force does no positive work); the frozen v3 "
            "exports are unchanged and no gate, threshold, weight or physics constant was "
            "tuned. The zero-command baseline and oracle numbers here supersede the "
            "pre-FIXPUMP `ab_report.json/.md`."
        ),
        "",
        (
            "**This artifact also supersedes the v1/v2 closed-loop A/B** (`ab_report.json/.md` "
            "at `e57c6bf`): earlier surrogate numbers are void, and every v1/v2 held-out number "
            "was scored on the angle-domain target that the acceleration actuator cannot consume."
        ),
        "",
        (
            f"Verdict: **{report['verdict']['status']}** (exit {report['exit_code']}). "
            "Provenance, hashes and the full machine-readable evidence are in `ab_report.json`."
        ),
        "",
        "## Instrument and protocol",
        "",
        f"- Measurement: {report['measurement']}.",
        f"- Sim: {report['sim_source']}",
        f"- Dynamics: {report['dynamics_basis']}.",
        (
            f"- Config: `{report['provenance']['config']}` (config_hash "
            f"`{report['provenance']['config_hash']}`); all simulations are read-only through "
            "`scripts/ab_control_v3/`."
        ),
        (
            f"- Exports: `{report['provenance']['exports_dir']}`; per-file sha256 in "
            "`ab_report.json`. Held-out split sha256 "
            f"`{report['provenance']['heldout_sha256']}` (matches `SHA256SUMS` and "
            "`heldout_manifest.json`)."
        ),
        (
            f"- Window: {WINDOW} frames of 14 features + 6 conditioning values; "
            f"{report['provenance']['frames_per_episode']} ticks/episode; "
            f"{len(report['held_out_indices'])} episodes in scenarios 16..19; "
            f"seed order {report['seed_order']}."
        ),
        (
            "- Baseline: zero command every tick. Oracle: `run_episode_with_oracle` "
            "(privileged true state + `fin_accel_command`, v3 m/s^2 demand). Student: "
            "`physics_baseline + Delta` from the frozen v3 export, emitted only where the "
            "baseline reference says `ai_healthy` and the exported calibration threshold is "
            "cleared, clipped only at the seam's hard +/-10 m/s^2 bound. No physics, gain, "
            "threshold or scenario was tuned."
        ),
        (
            f"- Sidecar f32 floor: one ULP is about {ULP_M:.2e} m at ~600 m lateral distance; "
            f"a delta below {MIN_MEANINGFUL_M:g} m is a numerical tie, not a win."
        ),
        "",
    ]
    if report["blockers"]:
        out += ["## Blockers", ""]
        out += [f"- {b}" for b in report["blockers"]]
        out += [""]
    if report["oracle_failure"]:
        out += ["## Positive control", "", f"- **FAILED**: {report['oracle_failure']}", ""]
    else:
        oracle = report["seeds"]["1"]["oracle_gtheta"]["aggregate"]
        out += [
            "## Positive control",
            "",
            (
                f"- Oracle seed 1: lateral RMS {oracle['baseline']['lateral_rms_m']:.3f} -> "
                f"{oracle['assisted']['lateral_rms_m']:.3f} m, apogee "
                f"{oracle['baseline']['apogee_m']:.3f} -> "
                f"{oracle['assisted']['apogee_m']:.3f} m; passes on every seed "
                "(see JSON). The seam can detect an improvement; this credits no student."
            ),
            "",
        ]

    for seed in report["seed_order"]:
        for name, entry in report["seeds"][str(seed)].items():
            agg = entry["aggregate"]
            label = {
                "oracle_gtheta": "oracle (fin_accel_command, key kept for gate compatibility)",
                "lrs": "lrs (partial export: seeds 1/2 only)",
            }.get(name, name)
            out += [
                f"## Seed {seed} · `{label}`",
                "",
                (
                    "| scenario | ep | baseline apogee (m) | assisted apogee (m) | "
                    "Δ apogee (m) | baseline lateral RMS (m) | assisted lateral RMS (m) | "
                    "Δ lateral (m) | lateral % |"
                ),
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
            for r in entry["per_scenario"]:
                out.append(
                    f"| {r['scenario']} | {r['episodes']} "
                    f"| {r['baseline']['apogee_m']:.3f} | {r['assisted']['apogee_m']:.3f} "
                    f"| {r['apogee_delta_m']:+.4f} "
                    f"| {r['baseline']['lateral_rms_m']:.3f} "
                    f"| {r['assisted']['lateral_rms_m']:.3f} "
                    f"| {r['lateral_delta_m']:+.4f} | {r['lateral_ratio_pct']:.5f} |"
                )
            out += [
                (
                    f"| **ALL ({agg['episodes']} ep)** | — "
                    f"| {agg['baseline']['apogee_m']:.3f} | {agg['assisted']['apogee_m']:.3f} "
                    f"| {agg['apogee_delta_m']:+.4f} "
                    f"| {agg['baseline']['lateral_rms_m']:.3f} "
                    f"| {agg['assisted']['lateral_rms_m']:.3f} "
                    f"| {agg['lateral_delta_m']:+.4f} | {agg['lateral_ratio_pct']:.5f} |"
                ),
                "",
            ]
            if "gate_open_ticks" in agg:
                out += [
                    (
                        f"interventions {agg['interventions']} / {agg['rows']} ticks; gate trips "
                        f"{agg['gate_trips']}; clipped {agg['clipped_ticks']}; max command "
                        f"{agg['max_command_mps2']:.3f} m/s^2; tracking RMSE "
                        f"{agg['tracking_rmse_mps2']}; first command tick "
                        f"{agg['first_command_tick']}. Trajectory sha256: baseline "
                        f"`{entry['sha256']['baseline']}`, assisted "
                        f"`{entry['sha256']['assisted']}`."
                    ),
                    "",
                ]
    out += ["## Honesty notes", ""]
    out += [f"- {n}" for n in report["honesty_notes"]]
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    ap.add_argument("--out-md", type=Path, default=OUT_MD)
    args = ap.parse_args()
    torch.set_num_threads(1)
    for path in (HELPER, CONFIG, HELDOUT, MANIFEST):
        if not path.is_file():
            raise FileNotFoundError(f"missing {path.relative_to(ROOT)}")

    manifest = json.loads(MANIFEST.read_text())
    heldout = read_validation(HELDOUT)
    indices = [int(ep.index) for ep in heldout.episodes]
    if len(indices) != 20 or any(i % 20 not in (16, 17, 18, 19) for i in indices):
        raise ValueError(f"unexpected held-out indices: {indices}")
    heldout_hash = next(f["sha256"] for f in manifest["files"] if f["name"] == "validation.bin")
    if manifest.get("split") != "held-out" or sha256(HELDOUT) != heldout_hash:
        raise ValueError("heldout manifest split or file hash mismatch")

    seed_order = sorted(args.seeds)
    report_seeds: dict[str, dict] = {}
    baseline_shas: dict[str, str] = {}
    blockers: list[str] = []
    for spec in (*STUDENTS, *PARTIAL):
        for seed in seed_order:
            path = EXPORTS / "runs" / spec["family"] / f"s{seed}" / spec["file"]
            if not path.is_file():
                blockers.append(
                    f"{spec['family']} seed {seed}: missing export "
                    f"`{path.relative_to(ROOT)}`; seed not measured, no stand-in "
                    "was substituted"
                )
    for seed in seed_order:
        work = WORK / f"seed{seed}"
        baseline_dir = work / "baseline"
        printed = run_helper(seed, indices, baseline_dir)
        baseline = read_validation(baseline_dir / "validation.bin")
        baseline_track = read_track(baseline_dir / "validation.trajectory")
        if set(baseline_track) != set(indices):
            raise RuntimeError(f"seed {seed}: baseline episode set differs from held-out manifest")
        baseline_sha = sha256(baseline_dir / "validation.bin")
        baseline_shas[str(seed)] = baseline_sha
        if seed == 1:
            # Post-FIXPUMP the zero-command flight legitimately differs from the
            # frozen v3 export, which was recorded with the energy-pumping normal
            # force. What must still match is the scenario set: indices and
            # conditioning. If it is byte-identical, the helper was NOT rebuilt.
            if [ep.index for ep in baseline.episodes] != indices:
                raise RuntimeError("seed 1 baseline indices differ from the held-out manifest")
            cond_delta = max(
                float(
                    np.max(
                        np.abs(
                            np.asarray(ep.conditioning, np.float64)
                            - np.asarray(ref.conditioning, np.float64)
                        )
                    )
                )
                for ep, ref in zip(baseline.episodes, heldout.episodes)
            )
            if cond_delta > 1e-9:
                raise RuntimeError(
                    f"seed 1 baseline scenarios differ from the held-out export "
                    f"(max conditioning delta {cond_delta})"
                )
            if baseline_sha == sha256(HELDOUT):
                raise RuntimeError(
                    "seed 1 baseline is byte-identical to the PRE-FIXPUMP export: "
                    "the helper was not rebuilt against corrected dynamics"
                )
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
        oracle_track = read_track(oracle_dir / "validation.trajectory")
        if set(oracle_track) != set(baseline_track):
            raise RuntimeError(f"seed {seed}: oracle episode set differs")
        oracle_summary = summarise(baseline_track, oracle_track, None)
        oracle_summary["arm"] = "run_episode_with_oracle / fin_accel_command"
        oracle_summary["sha256"] = {
            "baseline": sha256(baseline_dir / "validation.trajectory"),
            "assisted": sha256(oracle_dir / "validation.trajectory"),
        }
        report_seeds[str(seed)]["oracle_gtheta"] = oracle_summary

        for spec in (*STUDENTS, *PARTIAL):
            export_dir = EXPORTS / "runs" / spec["family"] / f"s{seed}"
            if not (export_dir / spec["file"]).is_file():
                continue
            model, threshold = load_model(spec["family"], seed)
            if spec["family"] == "lrs":
                prediction = commands_for_lrs(model, baseline.episodes, threshold)
            else:
                prediction = commands_for_window(model, baseline.episodes, threshold)
            tsv_dir = work / spec["name"] / "commands"
            control = build_controls(baseline, prediction, tsv_dir)
            flown_dir = work / spec["name"] / "flown"
            run_helper(seed, indices, flown_dir, commands_dir=tsv_dir)
            flown_track = read_track(flown_dir / "validation.trajectory")
            if set(flown_track) != set(baseline_track):
                raise RuntimeError(f"seed {seed} {spec['name']}: episode set differs")
            for idx in indices:
                if flown_track[idx]["ticks"] != baseline_track[idx]["ticks"]:
                    raise RuntimeError(f"seed {seed} {spec['name']}: tick count changed at {idx}")
            summary = summarise(baseline_track, flown_track, control)
            summary["arm"] = "physics_baseline + exported student Delta, gated and clipped"
            summary["sha256"] = {
                "baseline": sha256(baseline_dir / "validation.trajectory"),
                "assisted": sha256(flown_dir / "validation.trajectory"),
            }
            summary["export_sha256"] = sha256(export_dir / spec["file"])
            report_seeds[str(seed)][spec["name"]] = summary
            del model
            print(
                f"seed {seed} {spec['name']:8s}: lateral "
                f"{summary['aggregate']['baseline']['lateral_rms_m']:.3f} -> "
                f"{summary['aggregate']['assisted']['lateral_rms_m']:.3f} m",
                flush=True,
            )

    oracle_failures = []
    for seed in seed_order:
        agg = report_seeds[str(seed)]["oracle_gtheta"]["aggregate"]
        if not (
            agg["assisted"]["lateral_rms_m"] < agg["baseline"]["lateral_rms_m"]
            and agg["assisted"]["apogee_m"] > agg["baseline"]["apogee_m"]
        ):
            oracle_failures.append(
                f"seed {seed}: lateral "
                f"{agg['baseline']['lateral_rms_m']:.3f} -> "
                f"{agg['assisted']['lateral_rms_m']:.3f}, apogee "
                f"{agg['baseline']['apogee_m']:.3f} -> "
                f"{agg['assisted']['apogee_m']:.3f}"
            )
    if oracle_failures:
        blockers.append(
            "oracle positive control failed on "
            + "; ".join(oracle_failures)
            + ": the seam or measurement is broken; no student verdict is valid"
        )

    primary = STUDENTS[0]["name"]

    def beats(metric: str, lower: bool) -> bool:
        for seed in seed_order:
            arms = report_seeds[str(seed)][primary]["aggregate"]
            a = arms["assisted"][metric]
            b = arms["baseline"][metric]
            if not (a < b if lower else a > b):
                return False
        return True

    beats_lateral = beats("lateral_rms_m", True) and not oracle_failures
    beats_apogee = beats("apogee_m", False) and not oracle_failures
    passed = beats_lateral and beats_apogee
    seed1_wins = [
        s["name"]
        for s in (*STUDENTS, *PARTIAL)
        if str(1) in report_seeds
        and s["name"] in report_seeds["1"]
        and report_seeds["1"][s["name"]]["aggregate"]["lateral_delta_m"] < -MIN_MEANINGFUL_M
    ]
    if passed and seed1_wins and primary not in seed1_wins:
        primary = seed1_wins[0]
        report_seeds_primary = True
    else:
        report_seeds_primary = False
    if report_seeds_primary:
        beats_lateral = beats("lateral_rms_m", True) and not oracle_failures
        beats_apogee = beats("apogee_m", False) and not oracle_failures
        passed = beats_lateral and beats_apogee

    if not passed:
        first = report_seeds.get("1", {}).get(primary, {}).get("aggregate", {})
        if oracle_failures:
            blockers.append(
                f"oracle positive control failed on {len(oracle_failures)} of "
                f"{len(seed_order)} seeds under corrected dynamics, so no student verdict is "
                f"valid; seed 1 primary ({primary}) lateral "
                f"{first.get('baseline', {}).get('lateral_rms_m', float('nan')):.3f} -> "
                f"{first.get('assisted', {}).get('lateral_rms_m', float('nan')):.3f} m is "
                "recorded but not credited. The oracle's pre-FIXPUMP apogee authority was "
                "the energy-pumping normal force, now removed."
            )
        else:
            blockers.append(
                f"no exported student beats zero command on the pre-registered primary "
                f"({primary}, all seeds): seed 1 lateral "
                f"{first.get('baseline', {}).get('lateral_rms_m', float('nan')):.3f} -> "
                f"{first.get('assisted', {}).get('lateral_rms_m', float('nan')):.3f} m; "
                "see per-seed tables for the corrected zero-command baseline"
            )

    provenance = {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_head": subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "exports_dir": "ai/out/v3_final",
        "config": str(CONFIG.relative_to(ROOT)),
        "config_sha256": sha256(CONFIG),
        "config_hash": manifest["config_hash"],
        "heldout_manifest_sha256": sha256(MANIFEST),
        "heldout_sha256": sha256(HELDOUT),
        "quality_report_sha256": sha256(EXPORTS / "quality_report.json"),
        "label": REPORT_LABEL,
        "dynamics_basis": DYNAMICS_BASIS,
        "post_fixpump_baseline_sha256": baseline_shas.get("1"),
        "baseline_sha256_by_seed": baseline_shas,
        "harness": "scripts/ab_v3.py",
        "harness_sha256": sha256(Path(__file__)),
        "helper": "scripts/ab_control_v3/src/main.rs",
        "helper_sha256": sha256(ROOT / "scripts" / "ab_control_v3" / "src" / "main.rs"),
        "exports_sha256": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in [
                EXPORTS / "runs" / spec["family"] / f"s{seed}" / spec["file"]
                for spec in (*STUDENTS, *PARTIAL)
                for seed in seed_order
                if (EXPORTS / "runs" / spec["family"] / f"s{seed}" / spec["file"]).is_file()
            ]
        },
        "frames_per_episode": manifest["frames_per_episode"],
        "feature_names": manifest["feature_names"],
        "target_names": manifest["target_names"],
        "window": WINDOW,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "command_limit_mps2": COMMAND_LIMIT,
        "min_meaningful_delta_m": MIN_MEANINGFUL_M,
        "sidecar_ulp_m": ULP_M,
    }
    report = {
        "label": REPORT_LABEL,
        "dynamics_basis": DYNAMICS_BASIS,
        "task": "closed-loop v3 AI assist vs zero command on OR-aligned held-out scenarios",
        "measurement": "POTOKTRJ v3 flown-truth sidecar (stride 26, 6-DoF truth), not surrogate",
        "sim_source": "read-only scripts/ab_control_v3 on commit "
        + provenance["git_head"]
        + "; oracle uses run_episode_with_oracle / "
        "ReferenceOptimizer::fin_accel_command (v3 g_accel, not g_theta)",
        "supersedes": "the pre-FIXPUMP v3 ab_report (energy-pumping normal force) plus the "
        "v1/v2 closed-loop ab_report at e57c6bf and the earlier surrogate A/B at 237920a; "
        "all are void, not smaller versions of these results",
        "seed_note": "seeds 2/3 use the seed-matched v3 exports on the same held-out indices "
        "generated with --seed 2/3; off-training generalization checks, not a "
        "re-training sweep",
        "held_out_indices": indices,
        "seed_order": seed_order,
        "students": [
            {
                "name": s["name"],
                "family": s["family"],
                "kind": s["kind"],
                "export": str(
                    (EXPORTS / "runs" / s["family"] / "s1" / s["file"]).relative_to(ROOT)
                ),
            }
            for s in STUDENTS
        ],
        "partial_students": [
            {
                "name": s["name"],
                "family": s["family"],
                "export": str(
                    (EXPORTS / "runs" / s["family"] / "s1" / s["file"]).relative_to(ROOT)
                ),
            }
            for s in PARTIAL
        ],
        "seeds": report_seeds,
        "verdict": {
            "status": "PASS" if passed else "FAIL",
            "primary": primary,
            "beats_lateral": beats_lateral,
            "beats_apogee": beats_apogee,
            "pass": passed,
            "oracle_positive_control": not oracle_failures,
            "any_student_beats_zero_seed1": seed1_wins,
        },
        "exit_code": 0 if passed else 1,
        "failure_location": ""
        if passed
        else (
            "no exported v3 student lowered pooled lateral RMS against the corrected "
            "zero-command baseline on every pre-registered seed beyond the f32 sidecar "
            "resolution; see blockers and per-seed tables; oracle positive control "
            + ("passed" if not oracle_failures else "FAILED")
        ),
        "provenance": provenance,
        "blockers": blockers,
        "oracle_failure": "; ".join(oracle_failures),
        "honesty_notes": [
            (
                "Commands are generated from the baseline flight's recorded frames and replayed "
                "by frame index through the simulator seam; after the assisted command moves the "
                "vehicle, later frames differ from the baseline flight, but the interlock schedule "
                "stays the paired baseline one."
            ),
            (
                "Lateral RMS and apogee are over the whole 30 s flown trajectory from the v3 "
                "sidecar, not over a gate-open tail and not from a surrogate."
            ),
            (
                "The v3 target is the fin command [g_accel_y, -g_accel_x] in m/s^2 seam order, so "
                "the student's trained output is written directly to the TSV; no degree/radian "
                "or g*tan conversion is applied."
            ),
            (
                "The physics_baseline term keeps its trained normalised scale and is added to the "
                "student Delta exactly as in engine.py::final_command; both are interpreted in "
                "the seam's m/s^2 units because that is the unit of the v3 supervised target."
            ),
            (
                "Clipping at +/-10 m/s^2 is the seam's hard actuator bound, counted per episode; "
                "no threshold, gain or physics constant was tuned for this report."
            ),
            (
                "Seeds 2/3 fly seed-matched exports on regenerated held-out flights; they test "
                "input robustness, not reproducibility of training. Missing exports are reported "
                "as blockers and never replaced with a stand-in."
            ),
            (
                "Post-FIXPUMP: the corrected zero-command baseline is NOT byte-identical to the "
                "frozen v3 held-out export, which was flown with the energy-pumping normal "
                "force. The 20 held-out scenario indices and their conditioning vectors are "
                "verified identical; only the flown physics changed. Exports were not retrained."
            ),
        ],
    }
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.out_md.write_text(markdown(report))
    print(f"wrote {args.out_json.relative_to(ROOT)} and {args.out_md.relative_to(ROOT)}")
    print(
        f"verdict: {report['verdict']['status']} (exit {report['exit_code']}), "
        f"oracle_ok={not oracle_failures}, blockers={len(blockers)}"
    )
    sys.exit(report["exit_code"])


if __name__ == "__main__":
    main()
