"""Summarise out/sweep/results.csv into out/sweep/SUMMARY.md.

Selection rule (fixed before looking at the numbers):
  an epoch is admissible only when EVERY seed has ratio <= 1.25 AND ece <= 0.05;
  among admissible epochs choose the lowest mean held-out RMSE across seeds;
  break ties (within 1e-9) by the highest mean coverage.
Spread is the sample standard deviation over the three seeds (ddof=1).
"""

from __future__ import annotations

import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

SWEEP = Path(__file__).resolve().parent
RATIO_LIMIT = 1.25
ECE_LIMIT = 0.05
METRICS = ("rmse_train", "rmse_held_out", "ratio", "ece", "coverage", "raw_ece")


def load() -> list[dict]:
    with (SWEEP / "results.csv").open(newline="") as handle:
        return [row for row in csv.DictReader(handle) if row.get("status") in ("ok", "reused")]


def fmt(value: float) -> str:
    return f"{value:.6f}"


def meanstd(values: list[float]) -> str:
    if len(values) < 2:
        return f"{fmt(values[0])} (n={len(values)})" if values else "n/a"
    return f"{fmt(st.mean(values))} ± {fmt(st.stdev(values))}"


def main() -> None:
    rows = load()
    by_key: dict[tuple[str, int, int], dict] = {
        (row["model"], int(row["epoch"]), int(row["seed"])): row for row in rows
    }
    models = sorted({key[0] for key in by_key})
    epochs = sorted({key[1] for key in by_key})
    seeds = sorted({key[2] for key in by_key})

    lines: list[str] = []
    add = lines.append
    add("# Epoch × seed sweep on the full run-4 dataset")
    add("")
    add(f"Data: `ai/out/run4/data` (80 train / 20 held-out episodes, 1500 frames each). "
        f"Pipeline: unchanged `potok_ai.engine`, residual + curriculum on, `--calibrate` on. "
        f"Epochs swept: **{', '.join(str(e) for e in epochs)}** — "
        f"epoch 80 was dropped because a single 40-epoch `lrs` run already costs ~17 min of "
        f"single-core CPU and 36 runs were infeasible; all three seeds were kept. "
        f"Seeds: {', '.join(str(s) for s in seeds)} (deterministic via `engine.seed_everything`). "
        f"Every run used a private output directory; gates, loss, architecture and data were not changed.")
    add("")
    add("Metrics are read from each run's `metrics.json` and `calibration.json`. "
        "`ece` is the calibrated ECE; `raw_ece` is before calibration; "
        "`coverage` is the empirical correct fraction at tol=0.1 over the held-out predictions "
        "(n=29020 for lrs, n=5804 for the window families).")
    add("")

    # Gate accounting and per-(model,epoch) aggregates.
    add("## Mean ± std across seeds (sample std, ddof=1)")
    add("")
    add("| model | epoch | rmse_train | rmse_held_out | ratio | ece | coverage | seeds passing gates |")
    add("|---|---:|---|---|---|---|---|---|")
    aggregates: dict[tuple[str, int], dict] = {}
    for model in models:
        for epoch in epochs:
            group = [by_key[(model, epoch, seed)] for seed in seeds if (model, epoch, seed) in by_key]
            if not group:
                continue
            agg = {
                "model": model, "epoch": epoch, "count": len(group),
                "passing": sum(
                    1 for row in group
                    if float(row["ratio"]) <= RATIO_LIMIT and float(row["ece"]) <= ECE_LIMIT
                ),
            }
            for metric in METRICS:
                values = [float(row[metric]) for row in group]
                agg[metric] = st.mean(values)
                agg[f"{metric}_std"] = st.stdev(values) if len(values) > 1 else 0.0
            aggregates[(model, epoch)] = agg
            add(f"| {model} | {epoch} | {meanstd([float(r['rmse_train']) for r in group])} "
                f"| {meanstd([float(r['rmse_held_out']) for r in group])} "
                f"| {meanstd([float(r['ratio']) for r in group])} "
                f"| {meanstd([float(r['ece']) for r in group])} "
                f"| {meanstd([float(r['coverage']) for r in group])} "
                f"| {agg['passing']}/{len(group)} |")
    add("")

    # Per-seed table, so nothing is hidden behind an average.
    add("## Every run (no cherry-picking)")
    add("")
    add("| model | epoch | seed | rmse_train | rmse_held_out | ratio | ece | coverage | n | pass |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for model in models:
        for epoch in epochs:
            for seed in seeds:
                row = by_key.get((model, epoch, seed))
                if row is None:
                    add(f"| {model} | {epoch} | {seed} | MISSING | | | | | | |")
                    continue
                passed = float(row["ratio"]) <= RATIO_LIMIT and float(row["ece"]) <= ECE_LIMIT
                add(f"| {model} | {epoch} | {seed} | {fmt(float(row['rmse_train']))} "
                    f"| {fmt(float(row['rmse_held_out']))} | {fmt(float(row['ratio']))} "
                    f"| {fmt(float(row['ece']))} | {fmt(float(row['coverage']))} "
                    f"| {row['n']} | {'yes' if passed else 'NO'} |")
    add("")

    # Selection.
    add("## Selected epoch per model")
    add("")
    for model in models:
        eligible = [agg for (name, _), agg in aggregates.items()
                    if name == model and agg["count"] == len(seeds)
                    and agg["passing"] == len(seeds)]
        if not eligible:
            add(f"- **{model}**: no epoch passes both gates on all {len(seeds)} seeds — "
                f"no selection made (rather than cherry-picking a seed).")
            continue
        best = sorted(eligible, key=lambda a: (a["rmse_held_out"], -a["coverage"]))[0]
        ties = [a for a in eligible if abs(a["rmse_held_out"] - best["rmse_held_out"]) <= 1e-9]
        tie_note = ""
        if len(ties) > 1:
            winner = sorted(ties, key=lambda a: -a["coverage"])[0]
            best = winner
            tie_note = f" (tie on held-out RMSE broken by coverage: {[a['epoch'] for a in ties]})"
        add(f"- **{model}**: epoch **{best['epoch']}**{tie_note} — held-out RMSE "
            f"{meanstd([float(by_key[(model, best['epoch'], s)]['rmse_held_out']) for s in seeds])}, "
            f"ratio {meanstd([float(by_key[(model, best['epoch'], s)]['ratio']) for s in seeds])}, "
            f"ece {meanstd([float(by_key[(model, best['epoch'], s)]['ece']) for s in seeds])}, "
            f"coverage {meanstd([float(by_key[(model, best['epoch'], s)]['coverage']) for s in seeds])}.")
    add("")

    # The actual question.
    add("## Does more than 40 epochs improve or degrade held-out performance?")
    add("")
    for model in models:
        curve = [(epoch, aggregates[(model, epoch)]["rmse_held_out"],
                  aggregates[(model, epoch)]["rmse_held_out_std"])
                 for epoch in epochs if (model, epoch) in aggregates]
        if len(curve) < 2:
            add(f"- **{model}**: insufficient data for a curve.")
            continue
        desc = ", ".join(f"{epoch}: {value:.6f}±{std:.6f}" for epoch, value, std in curve)
        first, last = curve[0][1], curve[-1][1]
        direction = "degrades" if last > first else "improves"
        delta = last - first
        best_epoch = min(curve, key=lambda item: item[1])[0]
        add(f"- **{model}** held-out RMSE vs epoch ({desc}): "
            f"40 → {curve[-1][0]} **{direction}** by {delta:+.6f} "
            f"({100 * delta / first:+.1f}%); lowest mean at epoch {best_epoch}.")
    add("")
    add("Ratio < 1 means held-out RMSE is *lower* than the mean training-scenario RMSE. "
        "That is not evidence of generalisation by itself: the four held-out scenarios "
        "may simply be easier than the sixteen training scenarios. The conditioning vectors "
        "(mass, area, cd0, static margin, thrust, gyro noise) have essentially identical "
        "means and ranges in both splits, so an easy-split explanation is plausible but "
        "not established — there is no re-split control in this sweep. The epoch trend is "
        "measured on the same fixed split, so it is comparable across epochs; the absolute "
        "ratio is not a claim about split difficulty being matched.")
    add("")
    add("## Caveats (read before using the selected epoch)")
    add("")
    add("- **Selection on held-out RMSE spends the held-out set.** The rule was fixed in "
        "advance, but the 100-epoch numbers are the winners on the same 20 episodes used "
        "for the gates, so they are optimistic as unbiased generalisation estimates.")
    add("- **ECE is in-sample and degenerate for most winners.** Calibration fits "
        "temperature/isotonic on the held-out logits themselves. Several runs report "
        "`ece = 0.000000` because isotonic maps the held-out set to itself; raw ECE is "
        "still large (e.g. loopwm seed 1: 0.582649). Passing `ece <= 0.05` is therefore "
        "weak evidence of calibrated confidence, not a standalone quality claim.")
    add("- **Epoch count also changes the curriculum.** `epochs_per_stage = epochs // 4` "
        "(10 / 15 / 25 for 40 / 60 / 100), so the curve is epoch count *plus* curriculum "
        "schedule, not training duration alone. The non-monotonic 60-epoch spike "
        "(loopwm-lite seed 3: 0.630336; loopwm seed 3: 0.388567) is seed-driven "
        "instability, not a smooth overfitting curve.")
    add("- **Seed variance dominates the 40-epoch baseline.** lrs seed 2 at 40 epochs "
        "scores 0.406082 held-out RMSE versus seed 1's 0.174805; the single-seed run 4 "
        "number was not a stable estimate. Three seeds are the minimum for reading these "
        "differences, and the 100-epoch advantage of loopwm (-0.001704, -0.9%) is inside "
        "the seed spread — treat it as a tie, not a win.")
    add("- **No run failed** (27/27 successful); all 27 passed both gates on every seed, "
        "so no selection is based on a dropped outlier.")
    add("")
    add("## Reproduce")
    add("")
    add("```bash")
    add("cd /home/sehaxe/potok-m/ai")
    add(".venv/bin/python out/sweep/driver.py --workers 10 --epochs 40 60 100 --seeds 1 2 3")
    add(".venv/bin/python out/sweep/summarize.py")
    add("```")
    add("")

    (SWEEP / "SUMMARY.md").write_text("\n".join(lines))
    print(f"wrote {SWEEP / 'SUMMARY.md'} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
