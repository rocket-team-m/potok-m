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

## PID vs AI assistant (`ab_pid_vs_ai.py`) — FINAL run4, closed-loop

This is the FINAL report. It SUPERSEDES the 12 s closed-loop report (commit
`42df574`) and the gravity/tilt surrogate report (commit `237920a`); both are
void. The simulator's command→dynamics seam (`c25eabb`) is used read-only, and
apogee/lateral RMS come from the flown-truth POTOKTRJ v1 sidecar.

```bash
cargo build --release --manifest-path scripts/ab_control/Cargo.toml \
  --target-dir scripts/ab_control/target
ai/.venv/bin/python scripts/ab_pid_vs_ai.py
```

`scripts/ab_control/` is a thin Rust driver reading `sim/` through its public
API (`run_episode_with_control`); the CLI's `--control-file` applies one TSV to
all episodes, so per-episode control needs the API. No sim source/config is
touched; artifacts land in `scripts/ab_work/`.

Scope: run4, 20 OR-aligned held-out episodes / 4 scenarios (indices
16..19,36..39,56..59,76..79,96..99), 1500 frames each, 30 s, 50 Hz, seed 1.
Baseline = zero command. Assisted = `physics_baseline(frame) + model Delta`,
mapped ×0.12 rad ×180/π to degrees and emitted only where `ai_healthy`
(AI_READY/no FAULT/phase interlock) is set and confidence clears the threshold.
Run4 exports: `loopwm_student.bin` (primary), `lrs.bin`, and the windowed
`lite.pt` (50-frame window). The v2 binary has NO calibration field — thresholds
are read from run4 `calibration.json` (loopwm 0, lrs 0.313924, lite 0.011412),
never hardcoded. AI-active ticks actually used: 18 404 on seed 1 (≈920/episode).

Seeds 1, 2, 3 are all run; seeds 2/3 reuse the seed-1 exports and are labelled
off-training generalization, not a training seed sweep.

Writes `ab_report.json` (per-scenario, aggregate, per-seed, sidecar sha256) and
`ab_report.md`. Exit `0` iff the primary assisted arm reduces lateral RMS on
seed 1, `1` otherwise; numbers are written before exiting either way.

FINAL result (seed 1, loopwm): lateral RMS 1528.366295 → 1528.725432 m
(+0.359 m, 100.023498 % of baseline), apogee 1734.977 → 1733.811 m (-1.167 m).
`lrs` (+0.067 m lateral, -0.198 m apogee) and `lite` (+0.161 m, -0.474 m) also
worsen. Seeds 2/3 agree in sign for all three arms. The script exits 1.
Differences are real above the f32 ULP (~1.2e-4 m at this scale) but small
(0.004–0.026 % of lateral RMS). No physics, gain or threshold was tuned to win.
The paired interlock schedule comes from the baseline run; commands are
replayed by frame index, so later states diverge — that is the closed loop.

## Root-cause diagnosis (`ab_diagnose.py`)

```bash
ai/.venv/bin/python scripts/ab_diagnose.py
```

Read-only follow-up: writes `ab_diagnosis.md/.json` (plus TSVs/trajectories in
`scripts/ab_diag_work/`). It adds an `oracle_gtheta` arm, a DIAGNOSTIC scale
sweep of the student correction (0/0.25/0.5/1/2), and verifies feature order,
runtime equivalence and the confidence gate. It never changes `ab_report.md`.
Finding: the exact oracle is worse than zero command on all three seeds, and
no scale wins, so the cause is the command→fin mapping / actuator authority,
not the student model or its inputs.
