"""Aggregate the v3 driver's results.csv into per-model mean +/- std and gates."""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIELDS = ("rmse_train", "rmse_held_out", "ratio", "ece", "raw_ece", "coverage")


def main() -> None:
    rows: dict[tuple[str, int], dict] = {}
    with (HERE / "results.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["status"] not in ("ok", "reused"):
                continue
            rows[(row["model"], int(row["seed"]))] = row
    by_model: dict[str, list[dict]] = defaultdict(list)
    for (model, seed), row in sorted(rows.items()):
        by_model[model].append(row)

    print("| model | seed | rmse_train | rmse_held_out | ratio | raw_ece | ece | coverage | n | method | gates |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for model, group in by_model.items():
        for row in group:
            gates = "PASS" if (float(row["ratio"]) <= 1.25 and float(row["ece"]) <= 0.05) else "FAIL"
            print(f"| {model} | {row['seed']} | {row['rmse_train']} | {row['rmse_held_out']} | "
                  f"{row['ratio']} | {row['raw_ece']} | {row['ece']} | {row['coverage']} | "
                  f"{row['n']} | {row['method']} | {gates} |")
    print()
    for model, group in by_model.items():
        cell = []
        for field in FIELDS:
            values = [float(row[field]) for row in group]
            mean = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            cell.append(f"{field}={mean:.6f} +/- {std:.6f}")
        passed = all(float(r["ratio"]) <= 1.25 and float(r["ece"]) <= 0.05 for r in group)
        print(f"{model}: {', '.join(cell)} | gates {'PASS' if passed else 'FAIL'} ({len(group)} seeds)")


if __name__ == "__main__":
    main()
