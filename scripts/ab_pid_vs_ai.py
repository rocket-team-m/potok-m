#!/usr/bin/env python3
"""Closed-loop A/B: zero command vs PID + exported student correction.

Runs the OR-aligned held-out scenarios through potok-sim's command->dynamics
seam (`Simulator::run_episode_with_control`). Apogee and lateral RMS are read
from the flown-truth trajectory sidecar (POTOKTRJ v1), never from the old
gravity/tilt surrogate. No sim source or config is modified: a thin Rust driver
under scripts/ab_control/ calls the public API because the CLI's --control-file
applies one TSV to every episode, not one per episode.

Final run4 (full 30 s flight) report. The earlier 12 s closed-loop report and
the gravity/tilt surrogate report are SUPERSEDED.

Usage:
    ai/.venv/bin/python scripts/ab_pid_vs_ai.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT / "ai" / "src"))

from potok_ai.data import read_validation  # noqa: E402
from potok_ai.export import load_student_export  # noqa: E402
from potok_ai.loopwm_lite import LiteConfig, LiteModel  # noqa: E402
from potok_ai.physics import physics_baseline  # noqa: E402

RAD_PER_NORM = 0.12  # sim/core/src/optimizer.rs:41, g_theta clamp, same mapping as the old surrogate
DEG_PER_RAD = 180.0 / np.pi
MAX_DEG = 10.0  # sim/core/src/control.rs:28, hard actuator limit (LOUD if exceeded)
INTERVENTION_EPS = 1e-9
STUDENTS = ("loopwm", "lrs", "lite")
PRIMARY = "loopwm"

CONFIG = ROOT / "ai" / "out" / "run4" / "sim.toml"
MODEL_DIR = ROOT / "ai" / "out" / "run4"
DATA_DIR = MODEL_DIR / "data"
HELPER = SCRIPTS / "ab_control" / "target" / "release" / "ab-control"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class StudentArm:
    """Exported v2 stateful student, run with its trained forward (to_model())."""

    def __init__(self, export) -> None:
        self.model = export.to_model()

    def reset(self) -> None:
        self.model.reset_state()

    @torch.no_grad()
    def step(self, frame, active: bool = True) -> tuple[np.ndarray, float]:
        delta, conf = self.model.step(frame)
        return delta.detach().numpy(), float(conf)


class LiteArm:
    """Windowed lite export: last `window` frames, then pooled head output."""

    def __init__(self, model: LiteModel) -> None:
        self.model = model.eval()
        self.window = model.cfg.window
        self.frames: list[np.ndarray] = []

    def reset(self) -> None:
        self.frames = []

    @torch.no_grad()
    def step(self, frame, active: bool = True) -> tuple[np.ndarray, float]:
        self.frames.append(np.asarray(frame[:14], dtype=np.float32))
        if len(self.frames) < self.window or not active:
            return np.zeros(2, dtype=np.float64), 0.0
        window = torch.as_tensor(np.stack(self.frames[-self.window:])).unsqueeze(0)
        cond = torch.as_tensor(np.asarray(frame[14:20], dtype=np.float32)).unsqueeze(0)
        latent, _ = self.model.refine(window, cond)
        output = self.model.head_from_latent(latent.mean(1))[0]
        return output[:2].numpy().astype(np.float64), float(output[2])


def run_helper(seed: int, indices: list[int], out_dir: Path,
               commands_dir: Path | None) -> dict[int, tuple[float, float]]:
    if not HELPER.is_file():
        raise FileNotFoundError(
            f"missing {HELPER.relative_to(ROOT)}; build with: cargo build --release "
            f"--manifest-path scripts/ab_control/Cargo.toml "
            f"--target-dir scripts/ab_control/target")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(HELPER), "--config", str(CONFIG), "--seed", str(seed),
        "--indices", ",".join(map(str, indices)), "--out-dir", str(out_dir),
    ]
    if commands_dir is not None:
        cmd += ["--commands-dir", str(commands_dir)]
    proc = subprocess.run(cmd, cwd=ROOT / "sim", capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ab-control failed ({proc.returncode}): {proc.stderr}")
    printed = {}
    for line in proc.stdout.splitlines():
        if not line.startswith("episode "):
            continue
        parts = dict(part.split("=", 1) for part in line.split() if "=" in part)
        printed[int(line.split()[1].rstrip(":"))] = (
            float(parts["apogee"]), float(parts["lateral_rms"]))
    if len(printed) != len(indices):
        raise RuntimeError(f"helper printed {len(printed)} episodes, expected {len(indices)}")
    return printed


def read_track(path: Path) -> dict[int, dict]:
    """Parse POTOKTRJ v1 and measure exactly what the Rust sidecar API measures."""
    raw = path.read_bytes()
    if len(raw) < 64:
        raise ValueError(f"{path}: sidecar shorter than its 64-byte header")
    magic, version, count, frames, stride = struct.unpack_from("<8sIIII", raw, 0)
    if magic != b"POTOKTRJ" or version != 1 or stride != 6:
        raise ValueError(f"{path}: bad sidecar header {magic!r} v{version} stride={stride}")
    off = 64
    result = {}
    for _ in range(count):
        (index,) = struct.unpack_from("<Q", raw, off)
        off += 8
        values = np.frombuffer(raw, dtype="<f4", count=frames * 6, offset=off)
        off += frames * 6 * 4
        rows = values.reshape(frames, 6).astype(np.float64)
        result[index] = {
            "apogee_m": float(np.max(rows[:, 2])),
            "lateral_rms_m": float(np.sqrt(np.mean(rows[:, 0] ** 2 + rows[:, 1] ** 2))),
            "ticks": int(frames),
        }
    if off != len(raw):
        raise ValueError(f"{path}: trailing bytes in sidecar")
    return result


def student_specs() -> list[dict]:
    """Thresholds come from run4 calibration.json: the export format has no field."""
    files = {
        "loopwm": MODEL_DIR / "loopwm" / "loopwm_student.bin",
        "lrs": MODEL_DIR / "lrs" / "lrs.bin",
        "lite": MODEL_DIR / "lite" / "lite.pt",
    }
    specs = []
    for name in STUDENTS:
        path = files[name]
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}")
        cal_path = MODEL_DIR / name / "calibration.json"
        cal = json.loads(cal_path.read_text())
        if name == "lite":
            stored = torch.load(path, map_location="cpu", weights_only=True)
            model = LiteModel(LiteConfig(**stored["model_config"]))
            model.load_state_dict(stored["state_dict"], strict=False)
            spec = {"model": LiteArm(model), "g_max": float(model.cfg.g_max),
                    "kind": "windowed lite torch export"}
        else:
            export = load_student_export(path)
            if export.in_features != 20:
                raise ValueError(f"{name}: expected 20-input frame, got {export.in_features}")
            spec = {"model": StudentArm(export), "g_max": export.g_max,
                    "kind": "v2 stateful student binary export"}
        spec.update({
            "name": name, "bin": str(path.relative_to(ROOT)),
            "sha256": sha256(path), "threshold": float(cal["threshold"]),
            "calibration": str(cal_path.relative_to(ROOT)),
            "calibration_method": cal["method"],
        })
        specs.append(spec)
    return specs


def build_controls(baseline, spec: dict, out_dir: Path) -> dict:
    """One TSV per episode; interlock = ai_healthy AND calibrated confidence."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model = spec["model"]
    stats = {"rows": 0, "ai_healthy_ticks": 0, "gate_open_ticks": 0,
             "gate_trips": 0, "interventions": 0, "clipped_ticks": 0,
             "max_command_deg": 0.0, "tracking_sq": 0.0}
    for ep in baseline.episodes:
        features = np.asarray(ep.frames[:, :14], dtype=np.float32)
        cond = np.asarray(ep.conditioning, dtype=np.float32)
        targets = np.asarray(ep.targets, dtype=np.float64)
        model.reset()
        deltas, confs = [], []
        with torch.no_grad():
            for i in range(len(features)):
                frame = np.concatenate((features[i], cond)).astype(np.float32)
                delta, conf = model.step(frame, active=targets[i, 3] > 0.5)
                deltas.append(np.asarray(delta, dtype=np.float64))
                confs.append(float(conf))
        delta = np.asarray(deltas, dtype=np.float64)
        conf = np.asarray(confs, dtype=np.float64)
        b = physics_baseline(torch.as_tensor(features, dtype=torch.float32)).numpy().astype(np.float64)
        gate = (targets[:, 3] > 0.5) & (conf >= spec["threshold"])

        command = np.where(gate[:, None], b + delta, 0.0)
        deg = command * RAD_PER_NORM * DEG_PER_RAD
        clipped = np.abs(deg) > MAX_DEG
        deg = np.clip(deg, -MAX_DEG, MAX_DEG)
        with (out_dir / f"{ep.index}.tsv").open("w") as handle:
            handle.write("# frame_index g_roll_deg g_pitch_deg\n")
            for i, (roll, pitch) in enumerate(deg):
                handle.write(f"{i} {roll:.6f} {pitch:.6f}\n")

        active = gate & (np.max(np.abs(deg), axis=1) > INTERVENTION_EPS)
        stats["rows"] += len(deg)
        stats["ai_healthy_ticks"] += int((targets[:, 3] > 0.5).sum())
        stats["gate_open_ticks"] += int(gate.sum())
        stats["gate_trips"] += int((~gate).sum())
        stats["interventions"] += int(active.sum())
        stats["clipped_ticks"] += int(clipped.any(axis=1).sum())
        stats["max_command_deg"] = max(stats["max_command_deg"], float(np.max(np.abs(deg))))
        if gate.any():
            stats["tracking_sq"] += float(np.sum((command[gate] - targets[gate, :2]) ** 2))
    stats["tracking_rmse_norm"] = float(
        np.sqrt(stats["tracking_sq"] / max(stats["gate_open_ticks"], 1)))
    stats.pop("tracking_sq")
    return stats


def summarise(baseline_track: dict, assisted_track: dict, control: dict) -> dict:
    scenarios = sorted({k % 20 for k in baseline_track})
    rows = []
    for sc in scenarios:
        candidates = [k for k in sorted(baseline_track) if k % 20 == sc]
        b_ap = float(np.mean([baseline_track[k]["apogee_m"] for k in candidates]))
        a_ap = float(np.mean([assisted_track[k]["apogee_m"] for k in candidates]))
        b_lat = float(np.sqrt(np.mean([baseline_track[k]["lateral_rms_m"] ** 2
                                       for k in candidates])))
        a_lat = float(np.sqrt(np.mean([assisted_track[k]["lateral_rms_m"] ** 2
                                       for k in candidates])))
        rows.append({
            "scenario": f"scenario_{sc:02d}", "episodes": len(candidates),
            "baseline": {"apogee_m": b_ap, "lateral_rms_m": b_lat},
            "assisted": {"apogee_m": a_ap, "lateral_rms_m": a_lat},
            "apogee_delta_m": a_ap - b_ap,
            "apogee_retention_pct": 100.0 * a_ap / b_ap,
            "lateral_ratio_pct": 100.0 * a_lat / b_lat,
            "lateral_delta_m": a_lat - b_lat,
        })
    keys = sorted(baseline_track)
    b_ap = float(np.mean([baseline_track[k]["apogee_m"] for k in keys]))
    a_ap = float(np.mean([assisted_track[k]["apogee_m"] for k in keys]))
    b_lat = float(np.sqrt(np.mean([baseline_track[k]["lateral_rms_m"] ** 2 for k in keys])))
    a_lat = float(np.sqrt(np.mean([assisted_track[k]["lateral_rms_m"] ** 2 for k in keys])))
    aggregate = {
        "episodes": len(keys),
        "baseline": {"apogee_m": b_ap, "lateral_rms_m": b_lat},
        "assisted": {"apogee_m": a_ap, "lateral_rms_m": a_lat},
        "apogee_delta_m": a_ap - b_ap,
        "apogee_retention_pct": 100.0 * a_ap / b_ap,
        "lateral_ratio_pct": 100.0 * a_lat / b_lat,
        "lateral_delta_m": a_lat - b_lat,
        **control,
    }
    return {"per_scenario": rows, "aggregate": aggregate}


def fmt(v: float, nd: int = 3) -> str:
    return f"{v:.{nd}f}"


def markdown(report: dict) -> str:
    out = [
        "# PID (zero command) vs AI assistant — closed-loop OR-aligned A/B",
        "",
        "**FINAL run4 report (full 30 s flights). It SUPERSEDES both earlier "
        "reports**: the 12 s closed-loop A/B (commit `42df574`) and the original "
        "gravity/tilt surrogate A/B (commit `237920a`). Those files are void, not "
        "a smaller version of these results.",
        "",
        "Apogee and lateral RMS are now measured directly from the flown-truth "
        "`validation.trajectory` sidecar (POTOKTRJ v1, per-tick x/y/z/vx/vy/vz)",
        "produced by `potok-sim`'s command→dynamics seam "
        "(`Simulator::run_episode_with_control`, `sim/core/src/episode.rs:130`).",
        "No sim source or config was changed; `scripts/ab_control/` is a read-only "
        "driver because the CLI's `--control-file` applies one TSV to every episode.",
        "",
        "## Protocol",
        "",
        "- Seed 1: 20 OR-aligned held-out episodes (indices "
        "16..19,36..39,56..59,76..79,96..99), 4 scenarios, 1500 frames each, "
        "from `ai/out/run4/sim.toml` (30 s, 50 Hz). Baseline validation is "
        "byte-identical to `ai/out/run4/data/heldout_validation.bin`.",
        "- Seeds 2 and 3: same 20 indices/scenarios, generated with `--seed 2/3`; "
        "the exports were NOT retrained, so these are off-training generalization "
        "checks, not a seed sweep of the training procedure.",
        "- Baseline arm: no control file, zero command on every tick.",
        "- Assisted arm: per-tick `command = physics_baseline(frame) + model Delta`. "
        "`loopwm`/`lrs` use the exported v2 stateful students with their trained "
        "forward (`load_student_export(...).to_model()`). `lite` uses the exported "
        "windowed torch model over the last 50 frames (no command before tick 50). "
        "A command is emitted only where `ai_healthy` (AI_READY, no FAULT, phase "
        "interlock allows) is set AND the model's confidence clears its threshold.",
        "- Calibration: the v2 export binary has NO calibration field; thresholds "
        "are read from each run4 `calibration.json` (loopwm "
        f"{report['calibration']['loopwm']:.6g}, lrs {report['calibration']['lrs']:.6g}, "
        f"lite {report['calibration']['lite']:.6g}). Nothing is hardcoded.",
        f"- Unit mapping: normalised command × {RAD_PER_NORM} rad × 180/π = degrees "
        f"(the reference optimizer's ±0.12 rad clamp, same scale as the old "
        f"surrogate). Commands beyond the seam's ±{MAX_DEG:.0f}° actuator limit are "
        f"clipped and counted (`clipped_ticks`); no physics or gains were tuned.",
        "- `interventions` counts gate-open ticks with a non-zero applied command; "
        "`gate_trips` counts ticks with the interlock closed. Both are summed over "
        "episodes. `tracking_rmse_norm` is the assisted command's RMS error against "
        "g_theta on gate-open ticks (diagnostic only).",
        "- Exit status: 0 iff the primary assisted arm reduces lateral RMS on seed 1; "
        "nonzero otherwise. Apogee deltas are reported even when tiny or negative.",
        "",
    ]
    s1 = report["seeds"]["1"][PRIMARY]["aggregate"]
    verdict = report["verdict"]
    out += [
        "## Verdict (primary: loopwm student, seed 1)",
        "",
        f"- lateral RMS {fmt(s1['baseline']['lateral_rms_m'], 6)} m → "
        f"{fmt(s1['assisted']['lateral_rms_m'], 6)} m "
        f"({fmt(s1['lateral_delta_m'], 6)} m, {fmt(s1['lateral_ratio_pct'], 6)} % of baseline): "
        f"{'BEAT' if verdict['beats_lateral'] else 'DID NOT BEAT'} baseline.",
        f"- apogee {fmt(s1['baseline']['apogee_m'])} m → "
        f"{fmt(s1['assisted']['apogee_m'])} m "
        f"({fmt(s1['apogee_delta_m'], 4)} m, "
        f"{fmt(s1['apogee_retention_pct'], 5)} % retention).",
        f"- overall **{'PASS' if verdict['pass'] else 'FAIL'}** (exit {report['exit_code']}).",
        "",
    ]
    for seed in report["seed_order"]:
        for name in STUDENTS:
            entry = report["seeds"][str(seed)][name]
            agg = entry["aggregate"]
            out += [
                f"## Seed {seed} · student `{name}`"
                + (" (primary)" if name == PRIMARY and seed == 1 else "")
                + (" (off-training)" if seed != 1 else ""),
                "",
                "| scenario | ep | baseline apogee (m) | assisted apogee (m) | Δ apogee (m) | "
                "retention % | baseline lateral RMS (m) | assisted lateral RMS (m) | "
                "Δ lateral (m) | lateral % |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
            for r in entry["per_scenario"]:
                out.append(
                    f"| {r['scenario']} | {r['episodes']} "
                    f"| {fmt(r['baseline']['apogee_m'])} | {fmt(r['assisted']['apogee_m'])} "
                    f"| {fmt(r['apogee_delta_m'], 4)} | {fmt(r['apogee_retention_pct'], 5)} "
                    f"| {fmt(r['baseline']['lateral_rms_m'], 6)} "
                    f"| {fmt(r['assisted']['lateral_rms_m'], 6)} "
                    f"| {fmt(r['lateral_delta_m'], 6)} | {fmt(r['lateral_ratio_pct'], 6)} |")
            out += [
                f"| **ALL ({agg['episodes']} ep)** | — | {fmt(agg['baseline']['apogee_m'])} "
                f"| {fmt(agg['assisted']['apogee_m'])} | {fmt(agg['apogee_delta_m'], 4)} "
                f"| {fmt(agg['apogee_retention_pct'], 5)} "
                f"| {fmt(agg['baseline']['lateral_rms_m'], 6)} "
                f"| {fmt(agg['assisted']['lateral_rms_m'], 6)} "
                f"| {fmt(agg['lateral_delta_m'], 6)} | {fmt(agg['lateral_ratio_pct'], 6)} |",
                "",
                f"interventions {agg['interventions']} / {agg['rows']} ticks; "
                f"gate trips {agg['gate_trips']}; clipped ticks {agg['clipped_ticks']}; "
                f"max applied command {fmt(agg['max_command_deg'])}°; "
                f"g_theta tracking RMSE {fmt(agg['tracking_rmse_norm'], 4)}. "
                f"Trajectory sha256: baseline `{entry['sha256']['baseline']}`, "
                f"assisted `{entry['sha256']['assisted']}`.",
                "",
            ]
    out += [
        "## Honesty notes",
        "",
        "- Commands are replayed open-loop from the baseline run's recorded frames; "
        "the control TSV is indexed by frame, as the simulator seam requires. Once "
        "the assisted command moves the vehicle, later frames differ from the "
        "baseline run — that is the point of the closed-loop comparison, but the "
        "interlock schedule stays the paired baseline one.",
        "- Lateral RMS is over the whole 30 s flown trajectory, not just the "
        "AI-active tail; the earlier 12 s dataset's tiny active window no longer "
        "limits the measurement.",
        "- Seeds 2/3 use the seed-1-trained exports; they test input robustness, "
        "not training reproducibility. No seed, scenario or episode was dropped.",
        "- The calibration `isotonic` map itself is not exported in calibration.json; "
        "only its threshold is available, and it is applied to the raw confidence. "
        "This is an upper bound on how often the gate opens.",
        "- `clipped_ticks` is the seam's hard ±10° actuator bound, not a tuning knob.",
        "- Measurement floor: the sidecar stores f32 positions. One ULP is about "
        f"{float(np.spacing(np.float32(s1['baseline']['lateral_rms_m']))):.2e} m at the "
        f"measured lateral scale and {float(np.spacing(np.float32(s1['baseline']['apogee_m']))):.2e} "
        "m at the measured altitude. Compare the deltas above against that floor "
        "before calling any difference a win.",
    ]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--work-dir", type=Path, default=SCRIPTS / "ab_work")
    ap.add_argument("--out-json", type=Path, default=SCRIPTS / "ab_report.json")
    ap.add_argument("--out-md", type=Path, default=SCRIPTS / "ab_report.md")
    args = ap.parse_args()
    torch.set_num_threads(1)
    if not CONFIG.is_file():
        raise FileNotFoundError(f"missing {CONFIG}")

    specs = student_specs()
    seed_order = sorted(args.seeds)
    report_seeds = {}
    reference_path = DATA_DIR / "heldout_validation.bin"
    reference = read_validation(reference_path)
    indices = [int(e.index) for e in reference.episodes]
    if len(indices) != 20 or reference.header.frames_per_episode != 1500:
        raise ValueError(f"expected 20 x 1500-frame held-out episodes, got "
                         f"{len(indices)} x {reference.header.frames_per_episode}")
        raise ValueError(f"expected 20 held-out indices, got {len(indices)}")
    for seed in seed_order:
        work = args.work_dir / f"seed{seed}"
        baseline_dir = work / "baseline"
        printed = run_helper(seed, indices, baseline_dir, None)
        baseline = read_validation(baseline_dir / "validation.bin")
        baseline_track = read_track(baseline_dir / "validation.trajectory")
        if seed == 1 and sha256(baseline_dir / "validation.bin") != sha256(reference_path):
            raise RuntimeError("seed-1 baseline does not match run4 held-out data")
        for idx, (apogee, lateral) in printed.items():
            got = baseline_track[idx]
            if abs(got["apogee_m"] - apogee) > 1e-3 or abs(got["lateral_rms_m"] - lateral) > 1e-3:
                raise RuntimeError(f"seed {seed} episode {idx}: sidecar/stdout mismatch")
        if len(indices) != 20 or len(baseline_track) != 20:
            raise ValueError(f"seed {seed}: expected 20 held-out episodes, got {len(indices)}")
        report_seeds[str(seed)] = {}
        for spec in specs:
            name = spec["name"]
            tsv_dir = work / name / "commands"
            control = build_controls(baseline, spec, tsv_dir)
            assisted_dir = work / name / "flown"
            run_helper(seed, indices, assisted_dir, tsv_dir)
            assisted_track = read_track(assisted_dir / "validation.trajectory")
            if set(assisted_track) != set(baseline_track):
                raise RuntimeError(f"seed {seed} {name}: episode sets differ")
            for idx in indices:
                if assisted_track[idx]["ticks"] != baseline_track[idx]["ticks"]:
                    raise RuntimeError(f"seed {seed} {name}: tick count changed at {idx}")
            summary = summarise(baseline_track, assisted_track, control)
            summary["sha256"] = {
                "baseline": sha256(baseline_dir / "validation.trajectory"),
                "assisted": sha256(assisted_dir / "validation.trajectory"),
            }
            report_seeds[str(seed)][name] = summary

    primary_agg = report_seeds["1"][PRIMARY]["aggregate"]
    beats_lateral = primary_agg["assisted"]["lateral_rms_m"] < primary_agg["baseline"]["lateral_rms_m"]
    report = {
        "task": "closed-loop AI assist vs zero-command PID on OR-aligned held-out scenarios",
        "supersedes": "12 s closed-loop A/B (commit 42df574) and surrogate A/B "
                      "(commit 237920a); both void",
        "measurement": "POTOKTRJ v1 flown-truth sidecar, not surrogate",
        "sim_source": "commit c25eabb control seam, read-only via scripts/ab_control",
        "config": str(CONFIG.relative_to(ROOT)),
        "calibration": {s["name"]: s["threshold"] for s in specs},
        "calibration_source": "run4 <model>/calibration.json; the v2 export binary "
                              "has no calibration field, nothing hardcoded",
        "frames_per_episode": reference.header.frames_per_episode,
        "seed_order": seed_order,
        "seed_note": "seeds 2/3 use seed-1-trained exports: off-training generalization, "
                     "not a training seed sweep",
        "held_out_indices": indices,
        "conversion": {
            "rad_per_normalised_command": RAD_PER_NORM,
            "actuator_limit_deg": MAX_DEG,
            "interlock": "recorded ai_healthy AND calibrated confidence >= threshold",
        },
        "students": [{k: v for k, v in s.items() if k != "model"} for s in specs],
        "seeds": report_seeds,
        "verdict": {
            "primary": PRIMARY,
            "beats_lateral": beats_lateral,
            "beats_apogee": primary_agg["apogee_delta_m"] > 0.0,
            "pass": beats_lateral,
        },
        "exit_code": 0 if beats_lateral else 1,
    }
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.out_md.write_text(markdown(report))
    print(f"wrote {args.out_json.relative_to(ROOT)} and {args.out_md.relative_to(ROOT)}")
    print(f"verdict: {'PASS' if beats_lateral else 'FAIL'} "
          f"(seed1 {PRIMARY}: lateral {primary_agg['baseline']['lateral_rms_m']:.3f} -> "
          f"{primary_agg['assisted']['lateral_rms_m']:.3f} m, "
          f"apogee {primary_agg['apogee_delta_m']:+.4f} m)")
    sys.exit(report["exit_code"])


if __name__ == "__main__":
    main()
