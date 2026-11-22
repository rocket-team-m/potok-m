#!/usr/bin/env python3
"""End-to-end smoke of the sim -> engine -> report chain (CPU only).

    python3 scripts/smoke_engine.py --dry-run   # print commands, run nothing
    python3 scripts/smoke_engine.py             # run, write report.json / report.md

Determinism: fixed seed (1), 20 episodes, a throwaway temp dir removed on exit.
No GPU work, no network, no edits outside this script's own directory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SIM_DIR = REPO / "sim"
AI_DIR = REPO / "ai"
SIM_CLI = SIM_DIR / "target" / "release" / "potok-cli"
DEFAULT_CONFIG = SIM_DIR / "config" / "default.toml"
ENGINE = AI_DIR / "src" / "potok_ai" / "engine.py"

SEED = 1
EPISODES = 20
SIM_DURATION_S = 8.0  # smoke-sized flights; full 30 s makes the stateful lrs path >50 s
MODELS = ("lrs", "loopwm-lite", "loopwm")
MODEL_LIMITS = {"lrs": 4, "loopwm-lite": 16, "loopwm": 16}
EVAL_LIMIT = 4
GATES = {"ratio": 1.25, "ece": 0.05}
REQUIRED = ("model", "rmse_train", "rmse_held_out", "ratio", "ece", "params", "mac", "bytes")


class LoudError(RuntimeError):
    """A failure that names the cause and the way out."""


@dataclass(frozen=True)
class Cmd:
    argv: tuple[str, ...]
    cwd: Path = REPO
    env: dict[str, str] = field(default_factory=dict)

    def render(self) -> str:
        env = " ".join(f"{k}={shlex.quote(v)}" for k, v in self.env.items())
        body = " ".join(filter(None, (env, shlex.join(self.argv))))
        return f"cd {shlex.quote(str(self.cwd))} && {body}"

    def run(self) -> None:
        try:
            subprocess.run(
                list(self.argv),
                cwd=self.cwd,
                env={**os.environ, **self.env},
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise LoudError(f"command failed ({exc}):\n  {self.render()}") from exc


def resolve_python() -> str:
    """Prefer the AI venv; fall back to the interpreter on PATH."""
    venv = AI_DIR / ".venv" / "bin" / "python"
    if venv.is_file():
        return str(venv)
    found = shutil.which("python3") or shutil.which("python")
    if found:
        return found
    raise LoudError(f"no Python found; install AI deps first: cd {AI_DIR} && uv sync")


def preflight() -> str:
    if not SIM_CLI.is_file():
        raise LoudError(
            f"simulator binary missing: {SIM_CLI}\n"
            f"build it: cd {SIM_DIR} && cargo build --release -p potok-cli"
        )
    if not DEFAULT_CONFIG.is_file():
        raise LoudError(f"simulator config missing: {DEFAULT_CONFIG} (check out sim/)")
    if not ENGINE.is_file():
        raise LoudError(f"training engine missing: {ENGINE} (check out ai/)")
    python = resolve_python()
    if not Path(python).is_file() and not shutil.which(python):
        raise LoudError(f"Python missing: {python}; install AI deps: cd {AI_DIR} && uv sync")
    return python


def build_commands(tmp: Path, python: str) -> tuple[list[Cmd], dict[str, Cmd], dict[str, Path]]:
    """Setup + one engine command per model. All paths are concrete."""
    config = tmp / "sim.toml"
    data = tmp / "data"
    cp = shutil.which("cp") or "/bin/cp"
    setup = [
        Cmd((cp, str(DEFAULT_CONFIG), str(config))),
        Cmd(("sed", "-i", f"s/^duration_s = .*/duration_s = {SIM_DURATION_S}/", str(config))),
        Cmd((shutil.which("mkdir") or "/bin/mkdir", "-p", str(data))),
    ]
    for split, name in (("train", "train_validation.bin"), ("held-out", "heldout_validation.bin")):
        gen = tmp / "gen" / split
        setup.append(Cmd((
            str(SIM_CLI), "--config", str(config), "--seed", str(SEED),
            "--episodes", str(EPISODES), "--physics-features", "--split", split,
            "--validation-out", str(gen),
        ), SIM_DIR))
        setup.append(Cmd((cp, str(gen / "validation.bin"), str(data / name))))

    out_dirs = {model: tmp / "out" / model for model in MODELS}
    engine_cmds: dict[str, Cmd] = {}
    for model in MODELS:
        engine_cmds[model] = Cmd((
            python, "-m", "potok_ai.engine",
            "--model", model, "--stage", "train",
            "--data", str(data), "--out", str(out_dirs[model]),
            "--epochs", "1", "--seed", str(SEED), "--device", "cpu",
            "--window", "50", "--batch", "8",
            "--limit", str(MODEL_LIMITS[model]), "--eval-limit", str(EVAL_LIMIT),
            "--width", "64", "--rank", "32", "--heads", "4", "--max-iters", "2",
            "--lite-width", "32", "--attn-rank", "8", "--lite-iters", "1",
            "--latent-tokens", "8", "--calibrate",
        ), AI_DIR, {"PYTHONPATH": str(AI_DIR / "src")})
    return setup, engine_cmds, out_dirs


def load_metrics(path: Path) -> dict:
    if not path.is_file():
        raise LoudError(f"no metrics at {path}; engine did not finish its run")
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise LoudError(f"unreadable metrics at {path}: {exc}") from exc
    missing = [key for key in REQUIRED if key not in data]
    if missing:
        raise LoudError(
            f"{path} is missing {', '.join(missing)}; add these fields to "
            f"potok_ai.engine.Metrics / METRIC_FIELDS (SPEC_ENGINE_IDEAL.md 5)"
        )
    for key in REQUIRED[1:]:
        value = data[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise LoudError(f"{path}: {key} is not a finite number ({value!r})")
    return data


def gate(data: dict) -> list[str]:
    reasons = []
    if data["ratio"] > GATES["ratio"]:
        reasons.append(f"ratio {data['ratio']:.6f} > {GATES['ratio']}")
    if data["ece"] > GATES["ece"]:
        reasons.append(f"ece {data['ece']:.6f} > {GATES['ece']}")
    return reasons


def markdown(results: list[dict]) -> str:
    header = ("model", "rmse_train", "rmse_held_out", "ratio", "ece",
              "params", "mac", "bytes", "pass")
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for row in results:
        values = (
            row["model"], f"{row['rmse_train']:.6f}", f"{row['rmse_held_out']:.6f}",
            f"{row['ratio']:.6f}", f"{row['ece']:.6f}", str(row["params"]),
            str(row["mac"]), str(row["bytes"]),
            "PASS" if row["pass"] else "FAIL (" + ", ".join(row["fail_reasons"]) + ")",
        )
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    lines.append(f"Gates: ratio <= {GATES['ratio']}, ece <= {GATES['ece']} (seed {SEED}).")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="print exact commands; run nothing")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent,
                        help="directory for report.json and report.md (default: scripts/)")
    args = parser.parse_args()

    python = preflight()
    tmp = Path(tempfile.gettempdir()) / "potok-smoke-preview"
    setup, engine_cmds, out_dirs = build_commands(tmp, python)

    if args.dry_run:
        print(f"# temp dir (created fresh on a real run): {tmp}")
        for cmd in setup:
            print(cmd.render())
        for model in MODELS:
            print(engine_cmds[model].render())
        return 0

    results: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="potok-smoke-") as scratch:
        tmp = Path(scratch)
        setup, engine_cmds, out_dirs = build_commands(tmp, python)
        for cmd in setup:
            cmd.run()
        config = tmp / "sim.toml"
        if config.read_text().find(f"duration_s = {SIM_DURATION_S}") < 0:
            raise LoudError(f"sim config patch failed: {config}; check {DEFAULT_CONFIG}")
        for model in MODELS:
            engine_cmds[model].run()
            data = load_metrics(out_dirs[model] / "metrics.json")
            reasons = gate(data)
            row = {key: data[key] for key in REQUIRED}
            row["pass"] = not reasons
            row["fail_reasons"] = reasons
            results.append(row)

    report = {
        "seed": SEED,
        "episodes": EPISODES,
        "gates": {"ratio_max": GATES["ratio"], "ece_max": GATES["ece"]},
        "results": results,
        "pass": all(row["pass"] for row in results),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out / "report.md").write_text(markdown(results))

    print()
    print(markdown(results), end="")
    print(f"report: {args.out / 'report.json'}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except LoudError as exc:
        print(f"smoke_engine: {exc}", file=sys.stderr)
        sys.exit(2)
