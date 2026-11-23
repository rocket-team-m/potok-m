# FIX-CRIT2 — v3 closed-loop A/B (DONE)

**Verdict: PASS (exit 0).** This file is the idempotency marker for task FIX-CRIT2.
If it exists with this verdict, do not re-run the A/B.

## What ran

- Harness: `scripts/ab_v3.py` + read-only Rust seam driver `scripts/ab_control_v3/`
  (built release, no sim source/config/physics/gate edits).
- Data: `ai/out/v3_final` exports, held-out scenarios 16..19, 20 episodes x 1500
  ticks, seed order 1/2/3. Seed-1 baseline `validation.bin` is byte-identical to
  `ai/out/v3_final/data/heldout_validation.bin` (sha256
  `5848e222604b92d659f5778e467bfb359b98d5a2634bdc1b1c2a8257459685ed`).
- Arms: zero command (baseline), `run_episode_with_oracle` /
  `fin_accel_command` (positive control), and the frozen `loopwm`, `loopwm-lite`
  and `lrs` v3 exports. All metrics are from the POTOKTRJ v3 flown-truth sidecar
  (stride 26), never a surrogate.
- Artifacts: `ab_report.json` / `ab_report.md`, with full per-file export hashes,
  config hash, seed order, window, sidecar sha256s and harness git HEAD
  (`cee6202c97825c0e66dc54283a9926417cd87075` at measurement time).

## Numbers (pooled, 20 episodes)

| arm | seed 1 lateral (m) | seed 1 apogee (m) |
|---|---:|---:|
| zero command | 1528.366 | 1734.977 |
| oracle | 1035.228 | 3971.197 |
| loopwm (primary) | 1502.180 (-26.187, 98.29%) | 1860.093 (+125.116) |
| loopwm-lite | 1502.180 (-26.186) | 1860.093 (+125.115) |
| lrs (seed 1 only) | 1502.529 (-25.838) | 1858.291 (+123.313) |

Primary beats zero command on both metrics on all three seeds; oracle positive
control passes on all three. f32 sidecar ULP ~6.1e-5 m, so a 26 m delta is not a
rounding artifact. Clipping: 0 ticks; max command 2.0 m/s^2 (head g_max=1 plus
the trained physics baseline), no tuning.

## Blocker / follow-up

- `ai/out/v3_final/runs/lrs/s3/lrs.bin` is absent (empty `run.log`), so lrs seed 3
  was not measured; no stand-in or fabricated number was substituted. loopwm and
  lite are complete on all 3 seeds and carry the verdict. Re-run lrs seed 3 only
  when that export exists, then regenerate this report.
- `ai/docs/MODEL_CARD.md:130-134` still says the v3 closed-loop comparison never
  ran; that claim is now stale. Left untouched here because another lane owns
  that file. Its owner should point it at `ab_report.json/.md`.
- `ruff check .` in `ai/` reports 7 pre-existing findings in `out/sweep/` and
  `out/v3_final/` drivers; `ruff format --check src tests` also reports 29
  pre-existing files needing reformat. Neither is in AI CI (which runs only
  `ruff check src tests` + `pytest`). The new harness is clean under both
  `ruff check` and `ruff format --check`; pytest 237 passed.

## Gates run

- `ai/.venv/bin/python -m pytest -q` — 237 passed.
- `ai/.venv/bin/ruff check src tests` — clean (CI scope); new harness
  `ruff check`/`format --check` — clean.
- `ai/.venv/bin/python tests/test_ab_report_gate_hard.py ../ab_report.json` —
  `ab report: all hard gates pass`.
