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

## PID vs AI assistant (`ab_pid_vs_ai.py`)

Concrete answer to "what does the AI give over pure PID on the same OR-aligned
wind/gust scenarios?" — CPU only, deterministic, no simulator source changes.

```bash
ai/.venv/bin/python scripts/ab_pid_vs_ai.py --verify-sim
```

`--verify-sim` regenerates the held-out split with
`sim/target/release/potok-cli --physics-features --seed 1` and refuses to run
unless it is byte-identical to `ai/out/run2/data/heldout_validation.bin`.
Without the flag the checked-in v2 files are used as-is. Writes
`ab_report.json` (per-episode, per-scenario and aggregate numbers, verdict,
proxy definition) and `ab_report.md` (the honest table). Exit `0` if the
primary assisted arm beats baseline on apogee retention **or** lateral RMS,
`1` otherwise; numbers are written before exiting either way.

Scope: 80 train / 20 held-out episodes, 16 / 4 scenarios, seed 1 fixed. Three
seeds are not used — generation and both students are seed 1; the 4 held-out
scenarios are the independent cells. Both exported students are scored
(`loopwm_student.bin` primary, `lrs.bin` secondary) on the same episode stream:
baseline = `physics_baseline(frame)` (AI off), assisted = baseline + exported
`Delta`. `g_theta` is a distillation target, not an actuator command, so it is
used only as a tracking-RMSE diagnostic.

**The result is a proxy, not a flown A/B.** `potok_sim` records `g_theta` but
never feeds any command back into RK4 (`sim/core/src/episode.rs:233,257`), and
validation v2 stores no per-tick truth position. Apogee/lateral numbers come
from a documented tilt/gravity surrogate applied to recorded commands over the
AI-allowed ticks only (401 of 12 000; the interlock is closed during the ~9.9 s
boost). Current aggregate (loopwm): apogee retention 100.000414 % (+0.0054 m,
below the 0.35 m baro noise — a tie), surrogate lateral RMS 0.05153 -> 0.00870
m (-83.1 %), `g_theta` tracking RMSE 0.9425 -> 0.7085 (-24.8 %). Do not read
this as measured flight performance; see the report's limitations.
