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
