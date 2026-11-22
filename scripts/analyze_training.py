"""Training-report plots in the visual language of opencode.ai.

Reads ai/out/sweep/results.csv and scripts/ab_report.json, writes PNG+SVG
plots and a Markdown analysis into a chosen output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "ai" / "out" / "sweep" / "results.csv"
AB = ROOT / "scripts" / "ab_report.json"

DARK = {
    "bg": "#0c0c0c",
    "panel": "#141414",
    "grid": "#262626",
    "text": "#f5f5f5",
    "muted": "#8a8a8a",
    "train": "#6b8fff",
    "held": "#f5a524",
    "good": "#34d399",
    "bad": "#f87171",
    "oracle": "#a78bfa",
}
LIGHT = {
    "bg": "#fbfbfb",
    "panel": "#ffffff",
    "grid": "#e5e5e5",
    "text": "#111111",
    "muted": "#666666",
    "train": "#3457d5",
    "held": "#b45309",
    "good": "#047857",
    "bad": "#b91c1c",
    "oracle": "#6d28d9",
}


def set_style(theme: dict) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": theme["bg"],
            "axes.facecolor": theme["panel"],
            "savefig.facecolor": theme["bg"],
            "text.color": theme["text"],
            "axes.labelcolor": theme["text"],
            "axes.edgecolor": theme["grid"],
            "xtick.color": theme["muted"],
            "ytick.color": theme["muted"],
            "grid.color": theme["grid"],
            "grid.linewidth": 0.6,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "normal",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlepad": 10,
            "legend.frameon": False,
            "svg.fonttype": "none",
        }
    )


def save(fig, out: Path, name: str) -> None:
    for suffix in ("svg", "png"):
        fig.savefig(out / f"{name}.{suffix}", bbox_inches="tight", dpi=180)
    plt.close(fig)


def load_sweep(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("epoch", "seed"):
            r[k] = int(r[k])
        for k in ("rmse_train", "rmse_held_out", "ratio", "ece", "raw_ece", "coverage"):
            r[k] = float(r[k])
    return rows


def mean_std(vals: list[float]) -> tuple[float, float]:
    return (statistics.fmean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0)


def plot_rmse(rows: list[dict], out: Path, theme: dict) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    by = defaultdict(list)
    for r in rows:
        by[(r["model"], r["epoch"])].append(r)
    colors = {"train": theme["train"], "held_out": theme["held"]}
    for key, style in (("rmse_train", "train"), ("rmse_held_out", "held_out")):
        for model in sorted({r["model"] for r in rows}):
            epochs = sorted({r["epoch"] for r in rows if r["model"] == model})
            means, stds = [], []
            for ep in epochs:
                m, s = mean_std([r[key] for r in by[(model, ep)]])
                means.append(m)
                stds.append(s)
            ax.errorbar(
                epochs,
                means,
                yerr=stds,
                marker="o" if style == "held_out" else "s",
                linestyle="-" if style == "held_out" else "--",
                linewidth=1.6,
                capsize=3,
                alpha=1.0 if style == "held_out" else 0.65,
                color=colors[style],
                label=f"{model} · {'held-out' if style == 'held_out' else 'train'}",
            )
    ax.set_title("Training log — RMSE over epochs, 3 seeds")
    ax.set_xlabel("epoch")
    ax.set_ylabel("RMSE (normalised command units)")
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.grid(True, axis="y")
    ax.legend(loc="best", fontsize=8)
    save(fig, out, "rmse_curve")


def plot_ece(rows: list[dict], out: Path, theme: dict) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    by = defaultdict(list)
    for r in rows:
        by[(r["model"], r["epoch"])].append(r)
    for model in sorted({r["model"] for r in rows}):
        epochs = sorted({r["epoch"] for r in rows if r["model"] == model})
        raw = [mean_std([r["raw_ece"] for r in by[(model, ep)]])[0] for ep in epochs]
        cal = [mean_std([r["ece"] for r in by[(model, ep)]])[0] for ep in epochs]
        ax.plot(epochs, raw, marker="o", linewidth=1.6, color=theme["muted"], label=f"{model} · raw")
        ax.plot(epochs, cal, marker="s", linewidth=1.6, label=f"{model} · calibrated")
    ax.axhline(0.05, color=theme["bad"], linestyle=":", linewidth=1.2, label="ECE gate 0.05")
    ax.set_title("Confidence calibration — raw vs calibrated ECE")
    ax.set_xlabel("epoch")
    ax.set_ylabel("ECE")
    ax.grid(True, axis="y")
    ax.legend(loc="best", fontsize=8)
    save(fig, out, "ece_curve")


def plot_ab(report: dict, out: Path, theme: dict) -> None:
    seeds = report.get("seeds", {})
    arms = ["oracle_gtheta", "loopwm", "lrs", "lite"]
    data = {}
    for arm in arms:
        vals = []
        for seed in seeds.values():
            agg = seed.get(arm, {}).get("aggregate")
            if agg:
                vals.append(agg)
        if vals:
            data[arm] = vals
    if not data:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.2))
    x = list(data)
    baseline = data["oracle_gtheta"][0]["baseline"]
    lateral = [
        statistics.fmean(
            v["assisted"]["lateral_rms_m"] - v["baseline"]["lateral_rms_m"] for v in vals
        )
        for vals in data.values()
    ]
    apogee = [
        statistics.fmean(v["apogee_delta_m"] for v in vals) for vals in data.values()
    ]
    colors = [theme["oracle"] if a == "oracle_gtheta" else theme["train"] for a in x]
    axes[0].bar(x, lateral, color=colors, width=0.6)
    axes[0].axhline(0, color=theme["muted"], linewidth=0.8)
    axes[0].set_title("Closed-loop A/B — lateral RMS delta")
    axes[0].set_ylabel("assisted − baseline (m)")
    axes[1].bar(x, apogee, color=colors, width=0.6)
    axes[1].axhline(0, color=theme["muted"], linewidth=0.8)
    axes[1].set_title("Closed-loop A/B — apogee delta")
    axes[1].set_ylabel("assisted − baseline (m)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=18)
        ax.grid(True, axis="y")
    fig.suptitle(f"Baseline lateral {baseline['lateral_rms_m']:.1f} m · oracle is the positive control", fontsize=10, color=theme["muted"])
    save(fig, out, "ab_comparison")


def write_analysis(rows: list[dict], report: dict | None, out: Path) -> None:
    lines = [
        "# Training analysis",
        "",
        "Source: `ai/out/sweep/results.csv` and `scripts/ab_report.json`.",
        "Style: opencode.ai palette — near-black/near-white, Inter-like sans, mono indices, blue accent.",
        "",
        "## Epoch × model, mean ± std over seeds",
        "",
        "| model | epoch | held-out RMSE | ratio | calibrated ECE | coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    by = defaultdict(list)
    for r in rows:
        by[(r["model"], r["epoch"])].append(r)
    for (model, epoch), vals in sorted(by.items()):
        hm, hs = mean_std([v["rmse_held_out"] for v in vals])
        rm, rs = mean_std([v["ratio"] for v in vals])
        em, _ = mean_std([v["ece"] for v in vals])
        cm, _ = mean_std([v["coverage"] for v in vals])
        lines.append(f"| {model} | {epoch} | {hm:.4f} ± {hs:.4f} | {rm:.3f} ± {rs:.3f} | {em:.4f} | {cm:.3f} |")
    if report:
        lines += [
            "",
            "## Closed-loop A/B",
            "",
            f"Verdict: **{report.get('verdict', {})}**",
            "",
            "Oracle is the positive control. A student that does not beat zero command is not flight-ready.",
        ]
    (out / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", type=Path, default=SWEEP)
    ap.add_argument("--ab", type=Path, default=AB)
    ap.add_argument("--out", type=Path, default=ROOT / "scripts" / "analysis_out")
    ap.add_argument("--theme", choices=("dark", "light"), default="dark")
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    set_style(DARK if args.theme == "dark" else LIGHT)

    rows = load_sweep(args.sweep) if args.sweep.is_file() else []
    if not rows:
        raise SystemExit(f"no sweep data at {args.sweep}; run the sweep first")
    report = json.loads(args.ab.read_text(encoding="utf-8")) if args.ab.is_file() else None

    plot_rmse(rows, out, DARK if args.theme == "dark" else LIGHT)
    plot_ece(rows, out, DARK if args.theme == "dark" else LIGHT)
    if report:
        plot_ab(report, out, DARK if args.theme == "dark" else LIGHT)
    write_analysis(rows, report, out)
    print(f"wrote {out}: {len(rows)} sweep rows, ab={'yes' if report else 'no'}")


if __name__ == "__main__":
    main()
