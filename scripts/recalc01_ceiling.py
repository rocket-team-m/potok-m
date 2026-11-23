#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""RECALC01: derive post-FIXPUMP ideal-teacher ceilings from flown sidecars.

Read-only. Consumes the POTOKTRJ v3 baseline/oracle sidecars produced by
``scripts/ab_v3.py`` and the finished ``ab_report.json``. No simulation is
re-run, no config, gate, threshold, weight or physics constant is changed.

Prints a Markdown section with:
  * the corrected zero-command baseline,
  * the phase-gate frozen-tick RMS floor,
  * the kinematic freeze bound on lateral reduction,
  * the burnout-energy ceiling on apogee gain,
  * the actuator-cap audit on the oracle reference command,
  * the measured oracle / student deltas from ab_report.json.

Usage:
    ai/.venv/bin/python scripts/recalc01_ceiling.py
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "scripts" / "ab_control_v3" / "runs"
REPORT = ROOT / "ab_report.json"
G = 9.80665
DT = 0.02  # 50 Hz
TICKS = 1500
BURN_TICK = 498  # t = 9.96 s, just after the last thrust knot (9.95 s)
GATE_TICK = 500  # t = 10.00 s, gate-open reference used by WORLD04


def read_sidecar(path: Path) -> dict[int, dict]:
    raw = path.read_bytes()
    magic, version, count, frames, stride = struct.unpack_from("<8sIIII", raw, 0)
    if magic != b"POTOKTRJ" or version != 3 or stride != 26 or frames != TICKS:
        raise ValueError(f"{path}: unexpected sidecar header {magic!r} v{version} {frames}/{stride}")
    off = 64
    out: dict[int, dict] = {}
    for _ in range(count):
        (index,) = struct.unpack_from("<Q", raw, off)
        off += 8
        values = np.frombuffer(raw, dtype="<f4", count=frames * stride, offset=off)
        off += frames * stride * 4
        rows = values.reshape(frames, stride).astype(np.float64)
        out[int(index)] = {
            "pos": rows[:, 0:3],
            "vel": rows[:, 3:6],
            "cmd": rows[:, 24:26],
        }
    if off != len(raw):
        raise ValueError(f"{path}: trailing bytes")
    return out


def aggregate(tracks: dict[int, dict]) -> dict:
    apogee = float(np.mean([np.max(t["pos"][:, 2]) for t in tracks.values()]))
    lats = [float(np.sqrt(np.mean(t["pos"][:, 0] ** 2 + t["pos"][:, 1] ** 2))) for t in tracks.values()]
    return {"apogee": apogee, "lateral": float(np.sqrt(np.mean(np.square(lats))))}


def derive(seed: int) -> dict:
    base = read_sidecar(RUNS / f"seed{seed}" / "baseline" / "validation.trajectory")
    oracle = read_sidecar(RUNS / f"seed{seed}" / "oracle" / "validation.trajectory")
    n = len(base)
    floor_sq = []
    freeze_r = []
    bo_z, bo_v2 = [], []
    gate_r, gate_vr, gate_speed = [], [], []
    for ep in base:
        p, v = base[ep]["pos"], base[ep]["vel"]
        floor_sq.append(np.sum(p[:BURN_TICK, 0] ** 2 + p[:BURN_TICK, 1] ** 2) / TICKS)
        r0 = float(np.hypot(p[GATE_TICK, 0], p[GATE_TICK, 1]))
        gate_r.append(r0)
        gate_vr.append(float((p[GATE_TICK, 0] * v[GATE_TICK, 0] + p[GATE_TICK, 1] * v[GATE_TICK, 1]) / r0))
        gate_speed.append(float(np.linalg.norm(v[GATE_TICK])))
        freeze_r.append(r0)
        bo_z.append(float(p[BURN_TICK, 2]))
        bo_v2.append(float(np.dot(v[BURN_TICK], v[BURN_TICK])))
    floor = float(np.sqrt(np.mean(floor_sq)))
    r0 = float(np.mean(freeze_r))
    active_frac = (TICKS - BURN_TICK) / TICKS
    freeze = float(np.sqrt(floor**2 + r0**2 * active_frac))
    ceiling = float(np.mean(bo_z) + np.mean(bo_v2) / (2 * G))
    b = aggregate(base)
    cap_hits = 0
    cap_total = 0
    for ep in oracle:
        cmd = oracle[ep]["cmd"][BURN_TICK:]
        cap_total += cmd.shape[0]
        cap_hits += int(np.sum(np.max(np.abs(cmd), axis=1) >= 9.999))
    return {
        "seed": seed,
        "episodes": n,
        "baseline": b,
        "gate_floor": floor,
        "gate_open_r": r0,
        "gate_open_vr": float(np.mean(gate_vr)),
        "gate_open_speed": float(np.mean(gate_speed)),
        "freeze_bound": freeze,
        "burnout_z": float(np.mean(bo_z)),
        "burnout_speed": float(np.sqrt(np.mean(bo_v2))),
        "energy_ceiling": ceiling,
        "lateral_ceiling_delta": freeze - b["lateral"],
        "apogee_ceiling_delta": ceiling - b["apogee"],
        "oracle_cap_fraction": cap_hits / cap_total if cap_total else 0.0,
    }


def main() -> None:
    rows = [derive(s) for s in (1, 2, 3)]
    report = json.loads(REPORT.read_text())
    print("## Post-FIXPUMP theoretical teacher ceiling (RECALC01)\n")
    print(
        "Derived from the corrected zero-command sidecars; phase gate = AI forbidden on "
        f"`Pad`/`Boost`, last thrust knot 9.95 s → {TICKS - BURN_TICK} active ticks "
        f"({(TICKS - BURN_TICK) * DT:.2f} s). Command cap ±10 m/s²; burnout energy "
        f"ceiling `z_bo + |v_bo|²/2g` with g={G}.\n"
    )
    print("| seed | baseline apogee (m) | baseline lateral (m) | gate floor (m) | r₀ @10 s (m) | v_r @10 s (m/s) | freeze bound (m) | max lateral cut (m / %) | z_bo (m) | \\|v_bo\\| (m/s) | energy ceiling (m) | max apogee gain (m) | oracle cap-hit % |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        pct = 100 * r["lateral_ceiling_delta"] / r["baseline"]["lateral"]
        print(
            f"| {r['seed']} | {r['baseline']['apogee']:.1f} | {r['baseline']['lateral']:.1f} "
            f"| {r['gate_floor']:.1f} | {r['gate_open_r']:.1f} | {r['gate_open_vr']:.1f} "
            f"| {r['freeze_bound']:.1f} | {r['lateral_ceiling_delta']:.1f} / {pct:.1f} "
            f"| {r['burnout_z']:.1f} | {r['burnout_speed']:.1f} "
            f"| {r['energy_ceiling']:.1f} | {r['apogee_ceiling_delta']:+.1f} "
            f"| {100 * r['oracle_cap_fraction']:.1f} |"
        )
    print("\nMeasured deltas vs the corrected zero-command baseline (`ab_report.json`):\n")
    print("| seed | arm | Δ lateral (m / %) | Δ apogee (m / %) |")
    print("|---:|---|---:|---:|")
    for s in (1, 2, 3):
        for arm in ("oracle_gtheta", "loopwm", "lite", "lrs"):
            e = report["seeds"].get(str(s), {}).get(arm)
            if e is None:
                continue
            a = e["aggregate"]
            dl = a["lateral_delta_m"]
            da = a["apogee_delta_m"]
            print(
                f"| {s} | {arm} | {dl:+.1f} / {a['lateral_ratio_pct'] - 100:.2f} "
                f"| {da:+.1f} / {a['apogee_retention_pct'] - 100:.2f} |"
            )


if __name__ == "__main__":
    main()
