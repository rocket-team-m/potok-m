# potok-train-tui

A standalone Rust monitor for Potok-M training runs, plus a deterministic SVG
report renderer. No Python code, no GPU, no network at runtime. The live panels
and the static reports share one palette: graphite base, muted cyan for train,
amber for held-out, red/green for gates. Sentence case, thin grid, no emoji.

## Build and test

From the repository root:

```sh
cargo test --manifest-path ai/tui/Cargo.toml
```

All six integration tests use `ratatui::TestBackend` and in-memory data. They
need no run directory.

## Live monitor

```sh
# default run: ai/out/final_100 (resolved from cwd, then from ai/tui/..)
cargo run --release --manifest-path ai/tui/Cargo.toml

# explicit run, starting on the flagship model
cargo run --release --manifest-path ai/tui/Cargo.toml -- \
  ai/out/final_100 --model loopwm

# synthetic but clearly labelled DEMO (no disk read)
cargo run --release --manifest-path ai/tui/Cargo.toml -- --demo
```

Keys: `q` quit, `tab` next model, `+`/`-` change the poll interval (250–5000 ms).
Mouse is not used.

The top bar shows model, seed, epoch and ETA. The large panel plots train and
held-out RMSE against epoch with a legend and current-value markers. The right
column shows the gates (`ratio <= 1.25`, `ece <= 0.05`, and a coverage gauge at
`>= 0.50` — a display target, not a project gate) and the other models. The
bottom panel shows provenance notes, tail of the logs and the key hints.

Missing run directories and malformed metrics fail loudly: the live view shows
an error panel and the CLI exits with status 2. Nothing is synthesized unless
`--demo` is given.

## Static SVG reports

```sh
# curves only
cargo run --release --manifest-path ai/tui/Cargo.toml -- \
  ai/out/final_100 --plot-only --model loopwm

# curves plus the A/B panel
cargo run --release --manifest-path ai/tui/Cargo.toml -- \
  ai/out/final_100 --plot-only --model loopwm \
  --ab-report scripts/ab_report.json

# write somewhere else
cargo run --release --manifest-path ai/tui/Cargo.toml -- \
  ai/out/final_100 --plot-only --plots-out /tmp/opencode/potok-plots \
  --ab-report scripts/ab_report.json
```

Default output is `<run>/plots/` (`demo-plots/` in demo mode). The command prints
one path per file:

```
ai/out/final_100/plots/rmse_curve.svg
ai/out/final_100/plots/ece_curve.svg
ai/out/final_100/plots/ab_comparison.svg
```

`ab_comparison.svg` is written only when an A/B report is found — either via
`--ab-report` or at `<run>/ab_report.json`. Without one, a note is printed to
stderr and the file is skipped. The report needs `seeds.<seed>.<model>` entries
with `aggregate.baseline` and `aggregate.assisted` arms carrying
`lateral_rms_m` and `apogee_m`, plus an `oracle_gtheta` entry for the oracle arm.

SVG bytes are deterministic: same run, model and A/B report produce identical
files. The tests assert this.

## Data sources

Read-only, discovered under the run directory:

- `<model>/metrics.json` (or `<model>.metrics.json`) — final RMSE, ratio, ECE, seed, params.
- `<model>/calibration.json` — calibrated/raw ECE, coverage, sample count, method.
- `logs/sweep_results.csv` (or `sweep_results.csv`) — per-epoch curves, filtered by model and `--seed`.
- `<model>/run.log`, `logs/*.log`, `REPORT.md` — tail shown in the activity panel.
- `history.json` (optional) — `{epoch, rmse_train, rmse_held_out, ...}` array.

If a run has final metrics but no epoch history, the reports show the final
checkpoint as a single point and the activity panel says so. Nothing is invented.

## Screenshots

Terminal capture of the live view (requires `tmux`):

```sh
tmux new-session -d -s potok-tui -x 120 -y 40
tmux send-keys -t potok-tui \
  'cargo run --release --manifest-path ai/tui/Cargo.toml -- ai/out/final_100 --model loopwm' Enter
sleep 3
tmux capture-pane -t potok-tui -p > /tmp/opencode/potok-tui.txt
tmux kill-session -t potok-tui
```

Rasterize a report (requires `rsvg-convert`):

```sh
rsvg-convert -w 1040 ai/out/final_100/plots/rmse_curve.svg \
  -o /tmp/opencode/rmse_curve.png
```
