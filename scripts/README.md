# Engine smoke

One command, CPU only, no GPU/network: generate simulator data, run all three
model families through `potok_ai.engine`, and check the acceptance gates.

```bash
python3 scripts/smoke_engine.py --dry-run   # print the exact commands, run nothing
python3 scripts/smoke_engine.py             # run; writes report.json + report.md here
```

Prerequisites (the script fails loudly and names the command if either is missing):

```bash
cd sim && cargo build --release -p potok-cli
cd ai && uv sync
```

`--dry-run` needs both tools present too; it prints `cd … && …` lines with a
concrete throwaway path so the real commands can be copied verbatim.

What a real run does: copies `sim/config/default.toml` into a temp dir with
`duration_s = 8.0`, generates 20 physics-feature episodes twice with the
simulator's `--split train` / `--split held-out`, places both `validation.bin`
files in one flat data directory (the engine needs both sides), then runs
`python -m potok_ai.engine` with `PYTHONPATH=src` from `ai/` for `lrs`,
`loopwm-lite` and `loopwm`. Limits are per model (`--limit 4/16/16`,
`--eval-limit 4`, one epoch, width 64, `--calibrate`) so each finish in seconds
on CPU. Seed is fixed at 1; the temp dir is removed on every exit path.

`report.json` holds the seed, episode count, gate values, one row per model and
an overall `pass`. `report.md` is the table printed to stdout:

| column | source |
|---|---|
| `rmse_train`, `rmse_held_out`, `ratio`, `ece`, `params`, `mac`, `bytes` | that model's `metrics.json` |
| `pass` | `ratio <= 1.25` **and** `ece <= 0.05` |

Exit codes: `0` all gates pass, `1` at least one gate failed, `2` a tool is
missing or a tool call / metrics file failed (message names the fix). A missing
`ece` field is a loud error — no calibration number is ever fabricated.

With the smoke-sized limits above, the ratio gate passes for all three families
but the ECE gate does not (the head is trained toward confidence 0.95 while its
command error is ~1). That is an engine finding, not a harness defect; the
script reports it as FAIL and exits 1 until the engine closes it.

## PID vs AI assistant (`ab_pid_vs_ai.py`) — CLOSED-LOOP, supersedes the surrogate

The earlier surrogate A/B (commit `237920a`) is **void**: it measured a
tilt/gravity proxy on recorded commands, because the simulator had no
command→dynamics seam. `sim` now has one (`c25eabb`), and this script measures
apogee and lateral RMS from the flown-truth POTOKTRJ v1 sidecar instead.

```bash
cargo build --release --manifest-path scripts/ab_control/Cargo.toml \
  --target-dir scripts/ab_control/target
ai/.venv/bin/python scripts/ab_pid_vs_ai.py
```

`scripts/ab_control/` is a ~100-line Rust driver that only READS `sim/` through
its public API (`run_episode_with_control`). The CLI's `--control-file` applies
one TSV to every episode in a run, so per-episode control needs the API. No sim
source/config is touched; artifacts land in `scripts/ab_work/`.

Scope: 20 OR-aligned held-out episodes / 4 scenarios (indices
16..19,36..39,56..59,76..79,96..99), 12 s, 50 Hz. Baseline = zero command.
Assisted = `physics_baseline(frame) + student Delta` in degrees (×0.12 rad,
then ×180/π), emitted only where the interlock allows (401/12000 ticks on seed
1). Both exported students are scored (`loopwm` primary, `lrs` secondary).
The student forward is reimplemented locally from the trained/exported weights
because an uncommitted change in `ai/src/potok_ai/student.py` is not valid for
these exports; it reproduces the old report's tracking RMSE exactly.

Seeds 1, 2, 3: the simulator supports them, so all three are run. Seeds 2/3 use
the seed-1-trained students and are labelled off-training generalization, not
a training seed sweep.

Writes `ab_report.json` (per-scenario, aggregate, per-seed, sidecar sha256) and
`ab_report.md`. Exit `0` iff the primary assisted arm reduces lateral RMS on
seed 1, `1` otherwise; numbers are written before exiting either way.

Current result (seed 1, loopwm): lateral RMS 566.002412 → 566.002452 m
(+3.9e-5 m, 100.000007 % — below the f32 coordinate ULP at ~600 m, a tie);
apogee 1321.346 → 1321.343 m (-0.0028 m). All three seeds and both students
are essentially tied or slightly worse; the script exits 1. The reason is that
the interlock only opens in the last ~0.4 s of a 12 s flight, so the assist has
almost no time to act. Do not read this as the controller being generally
useless — it is untested over a long AI-active window.
