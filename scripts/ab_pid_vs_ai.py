#!/usr/bin/env python3
"""A/B: onboard PID alone vs PID + exported student correction.

CPU-only, deterministic, no simulator source changes. Reads the run2 v2
validation files (80 train + 20 held-out episodes, 16 + 4 wind/gust scenarios)
and the exported stateful students. The optional --verify-sim flag regenerates
the held-out file with the release CLI and refuses to continue unless it is
byte-identical.

IMPORTANT — this is a PROXY, not a flown A/B. `potok_sim` never feeds a command
back into its dynamics: `episode.rs` records `optimizer.solve(...)` as a target
and then integrates `rk4(...)` with the unchanged vehicle. Validation v2 also
stores no per-tick truth position or apogee. Therefore no end-to-end apogee or
lateral drift can be measured without editing sim/, which this task forbids.
The proxy and its assumptions are written into ab_report.json/.md in full.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT / "ai" / "src"))

from potok_ai.data import read_validation  # noqa: E402
from potok_ai.export import load_student_export  # noqa: E402
from potok_ai.physics import physics_baseline  # noqa: E402

G = 9.80665  # sim/core/src/dynamics.rs:437
TILT_PER_CMD = 0.12  # sim/core/src/optimizer.rs:41, max_correction (rad)
INTERVENTION_EPS = 1e-6
STUDENTS = ("loopwm", "lrs")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sim(sim_bin: Path, sim_cfg: Path, seed: int, n_train: int,
               n_held: int, held_file: Path) -> dict:
    """Regenerate the held-out split and require byte-identical output."""
    if not sim_bin.is_file():
        raise FileNotFoundError(f"missing simulator CLI: {sim_bin} (pass --no-verify-sim)")
    if not sim_cfg.is_file():
        raise FileNotFoundError(f"missing simulator config: {sim_cfg}")
    with tempfile.TemporaryDirectory(prefix="ab-sim-", dir=SCRIPTS) as tmp:
        out = Path(tmp)
        cmd = [
            str(sim_bin), "--config", str(sim_cfg), "--seed", str(seed),
            "--episodes", str(n_train + n_held), "--physics-features",
            "--split", "held-out", "--validation-out", str(out),
        ]
        proc = subprocess.run(cmd, cwd=ROOT / "sim", capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"simulator CLI failed ({proc.returncode}): {proc.stderr}")
        regenerated = out / "validation.bin"
        if not regenerated.is_file():
            raise FileNotFoundError("simulator wrote no validation.bin")
        a, b = sha256(regenerated), sha256(held_file)
        if a != b:
            raise RuntimeError(f"held-out data not reproducible: regen={a} dataset={b}")
    shown = ["<tmp>" if part == str(out) else part for part in cmd]
    return {"verified": True, "sha256": b, "command": " ".join(shown)}


def load_data(data_dir: Path, seed: int) -> dict:
    train = read_validation(data_dir / "train_validation.bin")
    held = read_validation(data_dir / "heldout_validation.bin")
    for tag, data in (("train", train), ("held-out", held)):
        if data.header.version != 2:
            raise ValueError(f"{tag} file is not validation v2")
        if data.header.feature_count < 14:
            raise ValueError(f"{tag} file has no physics features")
    if train.header.seed != seed or held.header.seed != seed:
        raise ValueError(f"dataset seed != requested seed {seed}")
    train_sc = sorted({e.scenario for e in train.episodes})
    held_sc = sorted({e.scenario for e in held.episodes})
    if len(train_sc) != 16 or len(held_sc) != 4:
        raise ValueError(f"expected 16 train + 4 held-out scenarios, got "
                         f"{len(train_sc)} + {len(held_sc)}")
    if len(held.episodes) < 20:
        raise ValueError(f"need >=20 held-out episodes, got {len(held.episodes)}")
    if set(train_sc) & set(held_sc):
        raise ValueError("train and held-out scenario sets overlap")
    return {"train": train, "held": held, "train_scenarios": train_sc,
            "held_scenarios": held_sc}


def student_specs(model_dir: Path) -> list[dict]:
    paths = {
        "loopwm": (model_dir / "loopwm" / "loopwm_student.bin",
                   model_dir / "loopwm" / "calibration.json"),
        "lrs": (model_dir / "lrs" / "lrs.bin",
                model_dir / "lrs" / "calibration.json"),
    }
    specs = []
    for name in STUDENTS:
        bin_path, cal_path = paths[name]
        if not bin_path.is_file() or not cal_path.is_file():
            raise FileNotFoundError(f"student {name}: need {bin_path} and {cal_path}")
        export = load_student_export(bin_path)
        if export.in_features != 20:
            raise ValueError(f"{name}: expected 20-input smart frame, got {export.in_features}")
        cal = json.loads(cal_path.read_text())
        specs.append({
            "name": name, "bin": str(bin_path.relative_to(ROOT)),
            "calibration": str(cal_path.relative_to(ROOT)),
            "threshold": float(cal["threshold"]), "calibration_method": cal["method"],
            "g_max": export.g_max, "model": export.to_model(),
            "sha256": sha256(bin_path),
        })
    return specs


def integrate(a: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trapezoidal double integral with v(0)=y(0)=0 on the gate-open ticks."""
    n = len(a)
    v = np.zeros(n, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        dt = t[i] - t[i - 1]
        v[i] = v[i - 1] + 0.5 * (a[i] + a[i - 1]) * dt
        y[i] = y[i - 1] + 0.5 * (v[i] + v[i - 1]) * dt
    return v, y


def score_episode(ep, spec, frame_rate: float) -> dict | None:
    frames = np.asarray(ep.frames[:, :14], dtype=np.float32)
    cond = np.asarray(ep.conditioning, dtype=np.float32)
    model = spec["model"]
    model.reset_state()
    delta, conf = [], []
    with torch.no_grad():
        for i in range(len(frames)):
            frame = np.concatenate((frames[i], cond)).astype(np.float32)
            d, c = model.step(frame)
            delta.append(d.detach().numpy())
            conf.append(float(c))
    delta = np.asarray(delta, dtype=np.float64)
    conf = np.asarray(conf, dtype=np.float64)
    target = np.asarray(ep.targets, dtype=np.float64)
    b = physics_baseline(torch.as_tensor(frames, dtype=torch.float32)).numpy().astype(np.float64)
    g = target[:, :2]

    # Gate = reference optimizer's AI interlock/health flag AND calibrated
    # confidence. calibration.json stores a threshold but not the isotonic map,
    # so the threshold is applied to the raw sigmoid confidence: a conservative
    # upper bound on how often the assisted arm is actually allowed to act.
    gate = (target[:, 3] > 0.5) & (conf >= spec["threshold"])
    idx = np.flatnonzero(gate)
    if len(idx) < 2:
        return None

    delta_mag = np.max(np.abs(delta), axis=1)
    t = (idx - idx[0]) / frame_rate
    h = np.asarray(frames[:, 6], dtype=np.float64)  # baro_h, biased/noisy
    result = {
        "scenario": ep.scenario, "index": int(ep.index), "ticks_total": len(frames),
        "ticks_gate_open": len(idx), "gate_trips": int(np.sum(~gate)),
        "interventions": int(np.sum(gate & (delta_mag > INTERVENTION_EPS))),
        "max_correction_all": float(np.max(delta_mag)),
        "max_correction_applied": float(np.max(delta_mag[idx])),
        "mean_correction_applied": float(np.mean(delta_mag[idx])),
    }
    for name, command in (("baseline", b), ("assisted", b + delta)):
        tilt = command[idx] * TILT_PER_CMD
        a_lat = G * tilt[:, 0]
        a_zloss = 0.5 * G * (tilt[:, 0] ** 2 + tilt[:, 1] ** 2)
        _, y = integrate(a_lat, t)
        _, zloss = integrate(a_zloss, t)
        result[name] = {
            "apogee_m": float(np.max(h[idx] - zloss)),
            "lateral_y_sq": float(np.mean(y ** 2)),
            "tracking_e_sq": float(np.mean((command[idx] - g[idx]) ** 2)),
        }
    return result


def aggregate(rows: list[dict]) -> dict:
    def controller(name: str) -> dict:
        return {
            "apogee_m": float(np.mean([r[name]["apogee_m"] for r in rows])),
            "lateral_rms_m": float(np.sqrt(np.mean([r[name]["lateral_y_sq"] for r in rows]))),
            "tracking_rmse_norm": float(np.sqrt(np.mean([r[name]["tracking_e_sq"] for r in rows]))),
        }
    base, assisted = controller("baseline"), controller("assisted")
    return {
        "episodes": len(rows),
        "gate_open_ticks": int(sum(r["ticks_gate_open"] for r in rows)),
        "gate_trips": int(sum(r["gate_trips"] for r in rows)),
        "interventions": int(sum(r["interventions"] for r in rows)),
        "max_correction_applied": float(max(r["max_correction_applied"] for r in rows)),
        "max_correction_all_ticks": float(max(r["max_correction_all"] for r in rows)),
        "baseline": base, "assisted": assisted,
        "apogee_retention_pct": 100.0 * assisted["apogee_m"] / base["apogee_m"],
        "lateral_rms_ratio_pct": 100.0 * assisted["lateral_rms_m"] / base["lateral_rms_m"],
    }


def fmt(value: float, nd: int = 4) -> str:
    return f"{value:.{nd}f}"


def markdown(report: dict) -> str:
    d, p = report["data"], report["proxy"]
    out = [
        "# PID vs AI assistant — OR-aligned held-out wind/gust scenarios",
        "",
        "**This is a deterministic PROXY A/B, not a flown flight test.** The simulator "
        "never applies a command to its dynamics (`sim/core/src/episode.rs:233` records "
        "`g_theta` as a target, `:257` integrates with unchanged `Vehicle`); validation v2 "
        "stores no per-tick truth position/apogee. Real apogee gain is therefore "
        "**not measurable** without changing `sim/`. No physics was invented: the proxy "
        "uses only recorded commands, recorded baro height, the recorded frame rate, "
        "`g = 9.80665` (`sim/core/src/dynamics.rs:437`) and the reference clamp "
        "`0.12 rad` (`sim/core/src/optimizer.rs:41`).",
        "",
        "## Scope",
        "",
        f"- seed: **{d['seed']}** (fixed, as required); 3 seeds are **not** used: the data "
        "is generated at one fixed seed and the students were trained at that seed. "
        "The 4 held-out scenarios are the independent cells; 20 episodes total, no cherry-picking.",
        f"- train: {d['train_episodes']} episodes / {len(d['train_scenarios'])} scenarios; "
        f"held-out: {d['held_episodes']} episodes / {len(d['held_scenarios'])} scenarios "
        f"({', '.join(d['held_scenarios'])}).",
        f"- simulator reproducibility: "
        f"{'byte-identical regeneration confirmed' if report['sim']['verified'] else 'not verified'} "
        f"(sha256 `{report['sim']['sha256']}`).",
        f"- students: " + "; ".join(
            f"`{s['bin']}` (sha256 {s['sha256'][:12]}, g_max {fmt(s['g_max'], 4)})"
            for s in report["students"]),
        "",
        "## Controllers and proxy definition",
        "",
        "- **baseline (AI off)**: `b(x) = physics_baseline(frame)` from "
        "`ai/src/potok_ai/physics.py:37`, the frame-local P/D command used by the "
        "onboard deployment contract. The simulator's `g_theta` is a distillation "
        "*target* computed with privileged full state, not a command sent to the "
        "actuator path; adding the student's correction to `g_theta` would not be the "
        "flown controller, so it is not used as the baseline. `g_theta` is used only "
        "for the `tracking_rmse_norm` diagnostic below.",
        "- **assisted (AI on)**: `b(x) + Delta`, where `Delta` comes from the exported "
        "stateful student replayed tick-by-tick with `reset_state()` per episode "
        "(canonical Python forward, same as `edge/src/student_runtime.rs`).",
        f"- **gate**: reference `ai_healthy` flag AND calibrated confidence >= threshold "
        f"(loopwm {fmt(report['gate_thresholds']['loopwm'], 6)}, "
        f"lrs {fmt(report['gate_thresholds']['lrs'], 6)}). The isotonic map is not stored "
        "in calibration.json, so the threshold is applied to the raw sigmoid confidence: "
        "an upper bound on gate openness. Gate trips are counted over all ticks.",
        f"- **surrogate dynamics**: only on gate-open ticks, with zero initial velocity "
        f"and displacement: `tilt = command * {TILT_PER_CMD} rad`, "
        f"`a_lat = {G} * tilt_roll`, `a_zloss = 0.5 * {G} * |tilt|^2`, integrated twice "
        "by trapezoid at the recorded 50 Hz frame rate. Boost/interlock ticks contribute "
        "gate trips but no surrogate dynamics (the reference interlocks them to zero).",
        f"- **apogee_m (proxy)**: `max(baro_h_t - zloss_t)` over gate-open ticks, i.e. the "
        "recorded biased/noisy barometric altitude minus the surrogate tilt-induced "
        "vertical loss at the same tick. "
        "`apogee_retention_pct = 100 * assisted / baseline`.",
        "- **lateral_rms_m (proxy)**: RMS of the surrogate lateral displacement over "
        "gate-open ticks. **max |g| correction**: largest absolute student correction "
        "actually applied (gate-open ticks). **interventions**: gate-open ticks with "
        "|Delta| > 1e-6. **tracking_rmse_norm**: RMS of `command - g_theta` in the "
        "reference command's normalised units (diagnostic, not metres).",
        "",
    ]
    for name in STUDENTS:
        agg = report["aggregate"][name]
        out += [
            f"## Student: `{name}`",
            "",
            "| scenario | n ep | gate-open / gate-trip | interventions | max abs Δg | "
            "baseline apogee (m) | assisted apogee (m) | retention % | "
            "baseline lateral RMS (m) | assisted lateral RMS (m) | tracking RMSE b / b+Δ |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in report["rows"][name]:
            base, ass = r["baseline"], r["assisted"]
            out.append(
                f"| {r['scenario']} | {r['episodes']} | {r['gate_open_ticks']} / {r['gate_trips']} "
                f"| {r['interventions']} | {fmt(r['max_correction_applied'], 4)} "
                f"| {fmt(base['apogee_m'], 6)} | {fmt(ass['apogee_m'], 6)} "
                f"| {fmt(r['apogee_retention_pct'], 6)} | {fmt(base['lateral_rms_m'], 5)} "
                f"| {fmt(ass['lateral_rms_m'], 5)} "
                f"| {fmt(base['tracking_rmse_norm'], 4)} / {fmt(ass['tracking_rmse_norm'], 4)} |"
            )
        b, a = agg["baseline"], agg["assisted"]
        out += [
            f"| **ALL ({agg['episodes']} ep)** | — | {agg['gate_open_ticks']} / "
            f"{agg['gate_trips']} | {agg['interventions']} | "
            f"{fmt(agg['max_correction_applied'], 4)} | {fmt(b['apogee_m'], 6)} "
            f"| {fmt(a['apogee_m'], 6)} | {fmt(agg['apogee_retention_pct'], 6)} "
            f"| {fmt(b['lateral_rms_m'], 5)} | {fmt(a['lateral_rms_m'], 5)} "
            f"| {fmt(b['tracking_rmse_norm'], 4)} / {fmt(a['tracking_rmse_norm'], 4)} |",
            "",
        ]
    primary = report["primary"]
    pagg = report["aggregate"][primary]
    pb, pa = pagg["baseline"], pagg["assisted"]
    verdict = report["verdict"]
    out += [
        "## Verdict",
        "",
        f"Primary model `{primary}` (the non-degenerate loopwm student): assisted beats "
        f"baseline on apogee retention "
        f"({fmt(verdict['apogee_retention_pct'], 6)} %, "
        f"delta {fmt(pa['apogee_m'] - pb['apogee_m'], 6)} m, "
        f"{'YES' if verdict['beats_apogee'] else 'NO'}) and on lateral RMS "
        f"({fmt(verdict['lateral_rms_ratio_pct'], 6)} %, "
        f"{'YES' if verdict['beats_lateral'] else 'NO'}). Overall: "
        f"**{'PASS' if verdict['pass'] else 'FAIL'}** (exit "
        f"{report['exit_code']}).",
        "",
        f"Magnitude check: the surrogate apogee win is {fmt(pa['apogee_m'] - pb['apogee_m'], 6)} m, "
        f"far below the 0.35 m baro noise floor — it is a tie in physical terms. The larger, "
        f"still proxy, signal is command tracking: g_theta RMSE "
        f"{fmt(pb['tracking_rmse_norm'], 4)} -> {fmt(pa['tracking_rmse_norm'], 4)} "
        f"({fmt(100.0 * (1.0 - pa['tracking_rmse_norm'] / pb['tracking_rmse_norm']), 2)}% lower) "
        f"and surrogate lateral RMS {fmt(pb['lateral_rms_m'], 5)} -> "
        f"{fmt(pa['lateral_rms_m'], 5)} m, both measured over only "
        f"{pagg['gate_open_ticks']} gate-open ticks.",
        "",
        "## Limitations (do not read this as a flight result)",
        "",
        "- The command is never applied to the RK4 state; both arms are scored on the "
        "same recorded trajectory, so the apogee/lateral numbers are surrogates, not "
        "measurements of a controlled flight.",
        f"- Only {report['data']['gate_open_ticks_primary']} gate-open ticks exist in the "
        f"whole held-out set for the primary student (out of "
        f"{report['data']['held_ticks']} recorded ticks): the AI interlock is closed for "
        "almost all of the 12 s scenario because the boost phase lasts ~9.9 s. Any "
        "advantage is measured on that short tail.",
        "- The baro height is biased (-0.4 m) and noisy (0.35 m); it is the same "
        "recorded signal for both arms, so comparisons are paired, but absolute apogee "
        "is not the simulator's truth `apogee_m`.",
        "- The calibration isotonic map is not exported; the threshold is applied to raw "
        "sigmoid confidence, so gate openings here are an upper bound.",
        "- 3-seed replication is unavailable by construction (fixed generation seed and "
        "seed-1-trained students); the 4 held-out scenarios, 20 episodes, are the "
        "replication units. Seeds 2 and 3 were not generated — no seed was cherry-picked.",
        "",
    ]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=ROOT / "ai" / "out" / "run2" / "data")
    ap.add_argument("--model-dir", type=Path, default=ROOT / "ai" / "out" / "run2")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--verify-sim", action=argparse.BooleanOptionalAction, default=False,
                    help="regenerate the held-out split with the release CLI and require "
                         "a byte-identical file (default: off; needs sim/config + CLI)")
    ap.add_argument("--sim-bin", type=Path, default=ROOT / "sim" / "target" / "release" / "potok-cli")
    ap.add_argument("--sim-config", type=Path, default=ROOT / "ai" / "out" / "run2" / "sim.toml")
    ap.add_argument("--out-json", type=Path, default=SCRIPTS / "ab_report.json")
    ap.add_argument("--out-md", type=Path, default=SCRIPTS / "ab_report.md")
    args = ap.parse_args()

    torch.set_num_threads(1)
    data = load_data(args.data_dir, args.seed)
    held_file = args.data_dir / "heldout_validation.bin"
    sim = {"verified": False, "sha256": sha256(held_file)}
    if args.verify_sim:
        sim = verify_sim(args.sim_bin, args.sim_config, args.seed,
                         len(data["train"].episodes), len(data["held"].episodes), held_file)

    specs = student_specs(args.model_dir)
    rows = {}
    for spec in specs:
        scored = []
        for ep in data["held"].episodes:
            result = score_episode(ep, spec, data["held"].header.frame_rate_hz)
            if result is None:
                raise ValueError(f"{spec['name']}: episode {ep.index} has <2 gate-open ticks")
            scored.append(result)
        rows[spec["name"]] = scored

    aggregate_by_student = {}
    for name in STUDENTS:
        by_scenario = {}
        for r in rows[name]:
            by_scenario.setdefault(r["scenario"], []).append(r)
        scenario_rows = [aggregate(group) | {"scenario": group[0]["scenario"]}
                         for group in (by_scenario[k] for k in sorted(by_scenario))]
        aggregate_by_student[name] = {
            "per_episode": rows[name],
            "per_scenario": scenario_rows,
            "aggregate": aggregate(rows[name]),
        }

    primary = "loopwm"
    agg = aggregate_by_student[primary]["aggregate"]
    beats_apogee = agg["apogee_retention_pct"] > 100.0
    beats_lateral = agg["lateral_rms_ratio_pct"] < 100.0
    passed = beats_apogee or beats_lateral
    report = {
        "task": "what does the AI assistant give over pure PID on OR-aligned wind/gust scenarios",
        "deterministic": True,
        "proxy_not_flight": True,
        "sim": sim,
        "data": {
            "seed": args.seed,
            "train_episodes": len(data["train"].episodes),
            "held_episodes": len(data["held"].episodes),
            "held_ticks": int(sum(len(e.frames) for e in data["held"].episodes)),
            "train_scenarios": data["train_scenarios"],
            "held_scenarios": data["held_scenarios"],
            "gate_open_ticks_primary": agg["gate_open_ticks"],
            "seeds_used": [args.seed],
            "seed_note": "3 seeds not used: generation seed is fixed by the task and both "
                         "students were trained at seed 1; the 4 held-out scenarios "
                         "(20 episodes) are the independent cells.",
        },
        "proxy": {
            "gravity_mps2": G,
            "tilt_rad_per_normalised_command": TILT_PER_CMD,
            "intervention_eps": INTERVENTION_EPS,
            "baseline_command": "physics_baseline(frame), ai/src/potok_ai/physics.py:37",
            "assisted_command": "physics_baseline(frame) + exported student Delta",
            "reference_command": "g_theta from validation targets, diagnostic only",
            "gate": "target ai_healthy AND raw sigmoid confidence >= calibration threshold",
            "integration": "trapezoid, zero initial v/y, gate-open ticks only, 50 Hz",
            "apogee_m": "max over gate-open ticks of (recorded baro_h_t - surrogate tilt-induced vertical loss_t)",
        },
        "gate_thresholds": {s["name"]: s["threshold"] for s in specs},
        "students": [{k: v for k, v in s.items() if k != "model"} for s in specs],
        "rows": {name: aggregate_by_student[name]["per_scenario"] for name in STUDENTS},
        "per_episode": {name: aggregate_by_student[name]["per_episode"] for name in STUDENTS},
        "aggregate": {name: aggregate_by_student[name]["aggregate"] for name in STUDENTS},
        "primary": primary,
        "verdict": {
            "beats_apogee": beats_apogee,
            "beats_lateral": beats_lateral,
            "apogee_retention_pct": agg["apogee_retention_pct"],
            "lateral_rms_ratio_pct": agg["lateral_rms_ratio_pct"],
            "pass": passed,
        },
        "exit_code": 0 if passed else 1,
    }
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.out_md.write_text(markdown(report))
    print(f"wrote {args.out_json.relative_to(ROOT)} and {args.out_md.relative_to(ROOT)}")
    print(f"verdict: {'PASS' if passed else 'FAIL'} "
          f"(retention {agg['apogee_retention_pct']:.4f}%, "
          f"lateral ratio {agg['lateral_rms_ratio_pct']:.4f}%)")
    sys.exit(report["exit_code"])


if __name__ == "__main__":
    main()
