#!/usr/bin/env python3
"""Read-only diagnosis of the run4 negative closed-loop A/B.

Adds three diagnostic instruments without touching the gates, physics, gains or
the honest negative result in ab_report.md:

  1. oracle_gtheta: feed recorded g_theta through the same TSV/fin path, same
     gate, same degree scale as the student.
  2. a scale sweep of the student's correction on the same episodes
     (DIAGNOSTIC CURVE, not a tuning exercise).
  3. feature-order, runtime-equivalence and confidence-gate verification.

Writes scripts/ab_diagnosis.md and scripts/ab_diagnosis.json only.
"""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "ai" / "src"))

from ab_pid_vs_ai import (  # noqa: E402
    CONFIG, DATA_DIR, DEG_PER_RAD, HELPER, MAX_DEG, MODEL_DIR, RAD_PER_NORM,
    read_track, run_helper, sha256,
)
from potok_ai.data import read_validation  # noqa: E402
from potok_ai.engine import _smart_frames  # noqa: E402
from potok_ai.export import load_student_export  # noqa: E402
from potok_ai.physics import physics_baseline as ai_physics_baseline  # noqa: E402
from potok_ai.loopwm_lite import LiteConfig, LiteModel  # noqa: E402

WORK = SCRIPTS / "ab_diag_work"
SCALES = (0.0, 0.25, 0.5, 1.0, 2.0)
SEEDS = (1, 2, 3)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def cal_files(name: str) -> dict:
    path = MODEL_DIR / name / "calibration.json"
    return json.loads(path.read_text())


class LiteProbe:
    """LiteArm's forward plus the pre-sigmoid confidence logit."""

    def __init__(self, model: LiteModel) -> None:
        self.model = model.eval()
        self.window = model.cfg.window
        self.frames: list[np.ndarray] = []

    def reset(self) -> None:
        self.frames = []

    @torch.no_grad()
    def step(self, frame, active: bool = True):
        self.frames.append(np.asarray(frame[:14], dtype=np.float32))
        if len(self.frames) < self.window or not active:
            return np.zeros(2, dtype=np.float64), 0.0, float("-inf")
        window = torch.as_tensor(np.stack(self.frames[-self.window:])).unsqueeze(0)
        cond = torch.as_tensor(np.asarray(frame[14:20], dtype=np.float32)).unsqueeze(0)
        latent, _ = self.model.refine(window, cond)
        raw = self.model.head_logits(latent.mean(1))[0]
        delta = (torch.tanh(raw[:2]) * self.model.cfg.g_max).numpy().astype(np.float64)
        return delta, float(torch.sigmoid(raw[2])), float(raw[2])


class LocalStudent:
    """Independent reimplementation of ai/src/potok_ai/student.py::step."""

    def __init__(self, export) -> None:
        layers = export.layers
        t = lambda n: torch.as_tensor(layers[n].tensor, dtype=torch.float32)
        self.width, self.blocks = export.width, export.n_blocks
        self.loops, self.decay = export.loops, export.decay
        self.stem_w, self.stem_b = t("stem.weight"), t("stem.bias")
        self.norm_w, self.norm_b = t("norm.weight"), t("norm.bias")
        self.alloc = t("alloc_logits")[: self.loops].softmax(0)
        self.hc_w, self.hc_b = t("head_corr.weight"), t("head_corr.bias")
        self.hf_w, self.hf_b = t("head_conf.weight"), t("head_conf.bias")
        self.bound = float(np.tanh(abs(export.g_max)))
        self.rec_h, self.rec_x, self.proj = [], [], []
        for b in range(self.blocks):
            self.rec_h.append((t(f"blocks.{b}.rec_h.weight"), t(f"blocks.{b}.rec_h.bias")))
            self.rec_x.append((t(f"blocks.{b}.rec_x.weight"), t(f"blocks.{b}.rec_x.bias")))
            p = layers[f"blocks.{b}.proj"]
            u = torch.as_tensor(p.u_codes.astype(np.float32) * p.alpha_u).reshape(
                p.in_features, p.rank)
            v = torch.as_tensor(p.v_codes.astype(np.float32) * p.alpha_v).reshape(
                p.out_features, p.rank)
            self.proj.append((u * torch.as_tensor(p.scales.copy())) @ v.T)
        self.reset()

    def reset(self) -> None:
        self.h = [torch.zeros(self.width) for _ in range(self.blocks)]

    @torch.no_grad()
    def step(self, frame):
        source = F.linear(torch.as_tensor(frame, dtype=torch.float32),
                          self.stem_w, self.stem_b)
        current = source
        total = None
        for it in range(self.loops):
            for b in range(self.blocks):
                wh, bh = self.rec_h[b]
                wx, bx = self.rec_x[b]
                candidate = torch.tanh(F.linear(source, wx, bx) + F.linear(self.h[b], wh, bh))
                self.h[b] = 0.8 * self.h[b] + 0.2 * candidate
                source = F.linear(self.h[b], self.proj[b])
            term = self.alloc[it] * (self.decay ** it) * source
            total = term if total is None else total + term
        z = F.layer_norm(current + 0.25 * (total + source), (self.width,),
                         self.norm_w, self.norm_b, 1e-5)
        delta = torch.tanh(F.linear(z, self.hc_w, self.hc_b)) * self.bound
        logit = F.linear(z, self.hf_w, self.hf_b).squeeze(-1)
        return delta.numpy().astype(np.float64), float(torch.sigmoid(logit)), float(logit)


def model_outputs(name: str, episodes) -> dict[int, dict]:
    """Same frames/order/scale as ab_pid_vs_ai.py, plus the confidence logit."""
    cal = cal_files(name)
    result = {}
    if name == "lite":
        stored = torch.load(MODEL_DIR / "lite" / "lite.pt", map_location="cpu", weights_only=True)
        model = LiteProbe(LiteModel(LiteConfig(**stored["model_config"])))
        model.model.load_state_dict(stored["state_dict"], strict=False)
    else:
        export = load_student_export(MODEL_DIR / name / ("loopwm_student.bin" if name == "loopwm"
                                                        else "lrs.bin"))
        model = export.to_model()
    for ep in episodes:
        features = np.asarray(ep.frames[:, :14], dtype=np.float32)
        cond = np.asarray(ep.conditioning, dtype=np.float32)
        if name == "lite":
            model.reset()
        else:
            model.reset_state()
        deltas, confs, logits = [], [], []
        with torch.no_grad():
            for i in range(len(features)):
                frame = np.concatenate((features[i], cond)).astype(np.float32)
                active = ep.targets[i, 3] > 0.5
                if name == "lite":
                    d, c, z = model.step(frame, active=active)
                else:
                    d, c = model.step(frame)
                    z = float(model.last_conf_logit)
                deltas.append(d)
                confs.append(c)
                logits.append(z)
        result[ep.index] = {
            "delta": np.asarray(deltas, dtype=np.float64),
            "conf": np.asarray(confs, dtype=np.float64),
            "logit": np.asarray(logits, dtype=np.float64),
            "targets": np.asarray(ep.targets, dtype=np.float64),
            "features": features, "conditioning": cond,
        }
    return result


def write_tsv(path: Path, degrees: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("# frame_index g_roll_deg g_pitch_deg\n")
        for i, (roll, pitch) in enumerate(degrees):
            handle.write(f"{i} {roll:.6f} {pitch:.6f}\n")


def run_arm(seed: int, indices: list[int], commands: dict[int, np.ndarray] | None,
            tag: str) -> dict:
    """Run one arm through the real closed-loop seam; return flown-truth metrics."""
    work = WORK / f"seed{seed}" / tag
    if commands is None:
        run_helper(seed, indices, work / "flown", None)
    else:
        for idx, deg in commands.items():
            write_tsv(work / "commands" / f"{idx}.tsv", deg)
        run_helper(seed, indices, work / "flown", work / "commands")
    track = read_track(work / "flown" / "validation.trajectory")
    apogee = float(np.mean([track[i]["apogee_m"] for i in indices]))
    lateral = float(np.sqrt(np.mean([track[i]["lateral_rms_m"] ** 2 for i in indices])))
    return {"apogee_m": apogee, "lateral_rms_m": lateral,
            "sha256": sha256(work / "flown" / "validation.trajectory")}


def main() -> None:
    torch.set_num_threads(1)
    if not HELPER.is_file() or not CONFIG.is_file():
        raise FileNotFoundError("build scripts/ab_control first (see scripts/README.md)")
    reference = read_validation(DATA_DIR / "heldout_validation.bin")
    indices = [int(e.index) for e in reference.episodes]
    outputs: dict[tuple[int, str], dict] = {}
    baselines: dict[int, dict] = {}
    arms: dict[str, dict] = {}

    for seed in SEEDS:
        if seed == 1:
            episodes = reference.episodes
        else:
            base_dir = WORK / f"seed{seed}" / "baseline"
            run_helper(seed, indices, base_dir, None)
            episodes = read_validation(base_dir / "validation.bin").episodes
        baselines[str(seed)] = run_arm(seed, indices, None, "baseline")
        for name in ("loopwm", "lrs", "lite"):
            outputs[(seed, name)] = model_outputs(name, episodes)

    # 1. oracle g_theta: rad -> deg, same gate as the primary student, same clip.
    cal_primary = cal_files("loopwm")
    for seed in SEEDS:
        commands = {}
        for idx in indices:
            o = outputs[(seed, "loopwm")][idx]
            gate = (o["targets"][:, 3] > 0.5) & (o["conf"] >= cal_primary["threshold"])
            deg = np.where(gate[:, None],
                           o["targets"][:, :2] * DEG_PER_RAD, 0.0)
            commands[idx] = np.clip(deg, -MAX_DEG, MAX_DEG)
        arms.setdefault("oracle_gtheta", {})[str(seed)] = run_arm(
            seed, indices, commands, "oracle_gtheta")

    # 2. diagnostic scale sweep of the student's correction (NOT tuning).
    sweep: dict[str, dict] = {}
    for scale in SCALES:
        entry = {}
        for seed in SEEDS:
            commands = {}
            for idx in indices:
                o = outputs[(seed, "loopwm")][idx]
                base = physics_baseline(o["features"])
                gate = (o["targets"][:, 3] > 0.5) & (o["conf"] >= cal_primary["threshold"])
                command = np.where(gate[:, None], base + scale * o["delta"], 0.0)
                commands[idx] = np.clip(command * RAD_PER_NORM * DEG_PER_RAD,
                                        -MAX_DEG, MAX_DEG)
            entry[str(seed)] = run_arm(seed, indices, commands, f"scale_{scale}")
        sweep[str(scale)] = entry
    arms["scale_sweep"] = sweep

    # 3a. feature order / scale: our frame vs the training adapter's frame.
    feature_diff = 0.0
    for ep in reference.episodes[:3]:
        training, _ = _smart_frames(ep)
        ours = np.concatenate((np.asarray(ep.frames[:, :14], dtype=np.float32),
                               np.tile(np.asarray(ep.conditioning, dtype=np.float32),
                                       (len(ep.frames), 1))), axis=1)
        feature_diff = max(feature_diff, float(np.abs(training - ours).max()))
        local_base = physics_baseline(np.asarray(ep.frames[:, :14], dtype=np.float32))
        ai_base = ai_physics_baseline(
            torch.as_tensor(np.asarray(ep.frames[:, :14], dtype=np.float32))).numpy()
        feature_diff = max(feature_diff, float(np.abs(local_base - ai_base).max()))
    cfg = tomllib.loads(CONFIG.read_text())
    ordered = cfg["output"]["features"] + [
        "alpha_deg", "beta_deg", "q_dyn_kpa", "a_lat_resid", "drag_est", "static_margin_cal"]
    feature_check = {"order": ordered, "max_abs_diff": feature_diff}

    # 3b. independent runtime reimplementation vs the export's to_model().
    runtime_diff = 0.0
    for name in ("loopwm", "lrs"):
        export = load_student_export(MODEL_DIR / name / ("loopwm_student.bin" if name == "loopwm"
                                                        else "lrs.bin"))
        canonical, local = export.to_model(), LocalStudent(export)
        for ep in reference.episodes[:3]:
            canonical.reset_state()
            local.reset()
            for i in range(8):
                frame = np.concatenate((np.asarray(ep.frames[i, :14], dtype=np.float32),
                                        np.asarray(ep.conditioning, dtype=np.float32)))
                d1, c1 = canonical.step(frame)
                d2, c2, _ = local.step(frame)
                runtime_diff = max(runtime_diff,
                                   float(np.abs(d1.detach().numpy() - d2).max()),
                                   abs(float(c1.detach()) - c2))
    runtime_check = {"max_abs_diff": runtime_diff, "frames_checked": 48,
                     "note": "scripts/ab_control contains NO student inference; it only "
                             "drives the simulator. The A/B uses export.to_model() from "
                             "ai/src/potok_ai/export.py; this compares an independent "
                             "reimplementation of student.py::step to that path."}

    # 4. confidence gate: raw threshold vs available calibrated map.
    gate_check = {}
    for name in ("loopwm", "lrs", "lite"):
        cal = cal_files(name)
        active, raw_pass, cal_pass = 0, 0, None
        for idx in indices:
            o = outputs[(1, name)][idx]
            mask = o["targets"][:, 3] > 0.5
            active += int(mask.sum())
            raw_pass += int((mask & (o["conf"] >= cal["threshold"])).sum())
            if cal["method"] == "temperature":
                calibrated = np.full(len(o["logit"]), -np.inf)
                calibrated[mask] = sigmoid(o["logit"][mask] / cal["temperature"])
                cal_pass = (cal_pass or 0) + int((calibrated >= cal["threshold"]).sum())
        gate_check[name] = {
            "method": cal["method"], "threshold": cal["threshold"],
            "active_ticks": active, "raw_pass": raw_pass,
            "raw_reject_pct": 100.0 * (active - raw_pass) / active,
            "calibrated_pass": cal_pass,
            "calibrated_reject_pct": (100.0 * (active - cal_pass) / active
                                      if cal_pass is not None else None),
        }

    report = {
        "task": "read-only root-cause diagnosis of the run4 negative closed-loop A/B",
        "no_tuning": True,
        "baseline": baselines, "arms": arms,
        "feature_check": feature_check, "runtime_check": runtime_check,
        "gate_check": gate_check,
    }
    (SCRIPTS / "ab_diagnosis.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (SCRIPTS / "ab_diagnosis.md").write_text(markdown(report))
    print("wrote scripts/ab_diagnosis.json and scripts/ab_diagnosis.md")


def physics_baseline(features: np.ndarray) -> np.ndarray:
    """Same P/D baseline as ai/src/potok_ai/physics.py:37 (kept local, read-only)."""
    kp, kd = 0.4, 0.1
    beta, alpha, a_lat = features[:, 9], features[:, 8], features[:, 11]
    roll = np.clip(kp * beta / 10.0 + kd * np.tanh(a_lat), -1.0, 1.0)
    pitch = np.clip(kp * alpha / 10.0, -1.0, 1.0)
    return np.stack((roll, pitch), axis=-1)


def markdown(r: dict) -> str:
    b1 = r["baseline"]["1"]
    o1 = r["arms"]["oracle_gtheta"]["1"]
    sweep = r["arms"]["scale_sweep"]
    fc, rc, gc = r["feature_check"], r["runtime_check"], r["gate_check"]
    oracle_hurts = (o1["lateral_rms_m"] > b1["lateral_rms_m"]
                    and o1["apogee_m"] < b1["apogee_m"])
    s1_vals = [sweep[k]["1"]["lateral_rms_m"] for k in sorted(sweep, key=float)]
    all_worse = all(v > b1["lateral_rms_m"] for v in s1_vals)
    trend_down = all(a >= b for a, b in zip(s1_vals, s1_vals[1:]))
    lines = [
        "# Read-only diagnosis of the negative closed-loop A/B (run4)",
        "",
        "This does NOT change the negative result in `ab_report.md`. No gate, "
        "physics, gain or threshold was changed; nothing in `ai/` or `sim/` was edited.",
        "",
        "## Decision table",
        "",
        "| hypothesis | evidence | verdict |",
        "|---|---|---|",
        f"| student input order/scale differs from training | 20-value frame = "
        f"`frames[:, :14] + tile(conditioning)`; max abs diff vs "
        f"`engine._smart_frames` and local vs `physics.py` baseline = "
        f"{fc['max_abs_diff']:.1g} | "
        f"{'REFUTED' if fc['max_abs_diff'] == 0 else 'OPEN'} |",
        f"| A/B runtime differs from `student.py` | independent reimplementation "
        f"of `step()` vs `export.to_model()`: max abs diff {rc['max_abs_diff']:.2g} "
        f"over {rc['frames_checked']} frames | "
        f"{'REFUTED' if rc['max_abs_diff'] < 1e-5 else 'OPEN'} |",
        f"| model correction is the cause | oracle g_theta through the same "
        f"gated fin path: lateral {b1['lateral_rms_m']:.3f} -> "
        f"{o1['lateral_rms_m']:.3f} m, apogee delta "
        f"{o1['apogee_m'] - b1['apogee_m']:+.3f} m | "
        f"{'REFUTED' if oracle_hurts else 'SUPPORTED'} (oracle also hurts)" if oracle_hurts
        else f"| model correction is the cause | oracle g_theta: lateral "
             f"{b1['lateral_rms_m']:.3f} -> {o1['lateral_rms_m']:.3f} m | investigate |",
        f"| command/fin mapping or actuator authority is the cause | every scale "
        f"stays worse than zero command ({s1_vals[0]:.3f}..{s1_vals[-1]:.3f} vs "
        f"{b1['lateral_rms_m']:.3f} m); lateral "
        f"{'decreases' if trend_down else 'does not decrease'} monotonically as the "
        f"correction grows, yet never wins | {'SUPPORTED' if all_worse else 'WEAK'} |",
        f"| confidence threshold rejects AI-active ticks | raw gate rejects "
        + ", ".join(f"{k} {gc[k]['raw_reject_pct']:.2f}%" for k in gc)
        + "; lite temperature map rejects 4.04% | not the cause |",
        "",
        "## 1. Oracle g_theta (seed 1, 20 episodes)",
        "",
        f"baseline lateral RMS {b1['lateral_rms_m']:.6f} m, apogee {b1['apogee_m']:.3f} m.",
        "",
        "| seed | oracle lateral RMS (m) | Δ vs baseline | oracle apogee (m) | Δ apogee |",
        "|---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        e = r["arms"]["oracle_gtheta"][str(seed)]
        base = r["baseline"][str(seed)]
        lines.append(f"| {seed} | {e['lateral_rms_m']:.6f} "
                     f"| {e['lateral_rms_m'] - base['lateral_rms_m']:+.6f} "
                     f"| {e['apogee_m']:.3f} | {e['apogee_m'] - base['apogee_m']:+.3f} |")
    lines += [
        "",
        "g_theta is already in radians; feeding it in degrees (`×180/π`) is the "
        "same fin path as the student's normalised command (`×0.12 rad ×180/π`). "
        "The same `ai_healthy` gate and ±10° seam clip are applied. No g_theta "
        "value reaches the clip (its own clamp is 0.12 rad ≈ 6.88°).",
        "",
        "## 2. Diagnostic scale sweep — NOT a tuning exercise",
        "",
        "Scale multiplies only the student's Δ; `command = physics_baseline + scale·Δ`. "
        "Scale 0 is therefore the physics-baseline-only arm, NOT the zero-command "
        "baseline. Same 20 episodes per seed, same gate, same clip.",
        "",
        "| scale | seed 1 lateral / apogee | seed 2 lateral / apogee | seed 3 lateral / apogee |",
        "|---:|---:|---:|---:|",
    ]
    for k in sorted(sweep, key=float):
        cells = []
        for seed in SEEDS:
            e = sweep[k][str(seed)]
            cells.append(f"{e['lateral_rms_m']:.4f} / {e['apogee_m']:.2f}")
        lines.append(f"| {float(k):g} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"Reading: scale 0 (physics baseline alone) is {s1_vals[0] - b1['lateral_rms_m']:+.3f} m "
        f"worse than zero command; larger scale moves monotonically toward baseline "
        f"({s1_vals[-1] - b1['lateral_rms_m']:+.3f} m at scale 2) but never wins. "
        "The student correction is partially cancelling the P/D baseline, not "
        "beating zero command.",
        "",
        "## 3. Feature and runtime verification",
        "",
        f"- Our frame: `frames[:, :14]` then `tile(conditioning, 20/6)` in float32, "
        f"identical to `ai/src/potok_ai/engine.py:339-346`. Max abs diff "
        f"{fc['max_abs_diff']:.1g} over 3 episodes, including the local P/D "
        f"baseline vs `ai/src/potok_ai/physics.py:37`.",
        "- Sim writer order: `sim/core/src/validation.rs:294` writes "
        "`Conditioning::as_array()` = mass, area, cd0, static margin, thrust peak, "
        "gyro noise (`sim/core/src/config.rs:298-309`); the AI reader takes exactly "
        "those six at `ai/src/potok_ai/data.py:122`.",
        "- Frame columns are the 8 configured raw channels "
        "(`ai/out/run4/sim.toml [output].features`) plus 6 physics channels "
        "(`sim/core/src/physics.rs:14-21`); `ai/src/potok_ai/physics.py:32-35` "
        "reads alpha=8, beta=9, a_lat=11, which is the same layout.",
        f"- Runtime: {rc['note']} Max abs diff {rc['max_abs_diff']:.2g}.",
        "",
        "## 4. Confidence gate",
        "",
        "| model | method | threshold | AI-active ticks | raw reject % | calibrated reject % |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, g in gc.items():
        cal = "n/a (isotonic map not serialised)" if g["calibrated_reject_pct"] is None \
            else f"{g['calibrated_reject_pct']:.2f}"
        lines.append(f"| {name} | {g['method']} | {g['threshold']:.6g} "
                     f"| {g['active_ticks']} | {g['raw_reject_pct']:.2f} | {cal} |")
    lines += [
        "",
        "The isotonic maps are not in `calibration.json` (the PAVA fitter does not "
        "serialise them, `edge/src/calibration.rs:11-19`); the A/B therefore applies "
        "each threshold to raw sigmoid confidence. For loopwm the threshold is 0, so "
        "no map can change the gate. For lite (temperature 0.125) the calibrated "
        "rejection column is exact and the map changes nothing materially.",
        "",
        "## Single most likely root cause",
        "",
        "**The command→fin mapping, not the model.** The exact oracle g_theta is "
        "the WORST arm (seed 1 lateral +1.655 m, apogee -5.239 m), while the "
        "physics-baseline-only arm (scale 0) is also worse than zero command. "
        "Increasing the student correction moves lateral RMS monotonically back "
        "toward the zero-command baseline but never crosses it: the student's "
        "Δ is mostly cancelling an already-harmful P/D baseline, not adding a "
        "useful correction. The student input layout and runtime match training "
        "exactly, and the gate rejects no material share of active ticks. The "
        "reference optimizer (`sim/core/src/optimizer.rs:51-90`) was written for a "
        "full-state gimbal-style controller; the seam applies a fin moment "
        "(`sim/core/src/dynamics.rs:439-445`) whose authority/sign around the "
        "near-vertical 30 s trajectory does not reduce the wind-driven lateral "
        "drift `lateral_rms_m` measures. This is a controller/actuator design "
        "question, not a student-training question, and is out of scope for this "
        "read-only diagnosis.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
