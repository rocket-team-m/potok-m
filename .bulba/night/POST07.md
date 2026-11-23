# POST07 — Stateful low-rank gated-delta recurrent student candidate

**Date:** 2026-11-23
**Repo:** `/home/sehaxe/potok-m`
**Scope:** two new files plus this report. No existing file was edited. No subagents.
No external CUDA, no GPU, no push.

## What was built

`ai/src/potok_ai/gdn2_student.py` implements `GDN2Student`, an experimental
frame-per-tick student following the WORLD07 §5.2 specification:

* Width 48 (allowed 48–64), 2 blocks, 4 heads × head_dim 16, one pass per tick.
* Each block keeps a fixed fp32 recurrent state `[4, 16, 16]` = 4 KB (8 KB for
  both blocks). Nothing else carries history: no window buffer, no convolution,
  no attention.
* Update = gated delta rule: `decay·(1−erase)·S` contracts the memory, a rank-1
  term subtracts the component along the normalised current key, and an
  independent channel-wise write gate commits `v` (tanh-bounded). The readout is
  a matrix-vector product against the updated state.
* 14 frame features + 6 conditioning numbers; conditioning enters through a
  bounded FiLM, then RMSNorm. Head = two tanh-bounded corrections and a sigmoid
  confidence, as in `SPEC_STUDENT.md:12-25`.
* `step(frame, cond)` is the stateful runtime API; `forward(window, cond)` is a
  pure per-tick sweep from zero state used for causality checks and never
  mutates the persistent state. `reset_state()` supports phase boundaries.
* All parameters come from a private `torch.Generator` seeded by `cfg.seed`; the
  global RNG is never read. State is detached on entry and stored detached, so
  the autograd graph never spans ticks.

**Exact MAC accounting** (`mac_breakdown` / `analytical_mac`, one MAC = one
multiply-accumulate inside an `nn.Linear` or a recurrent matrix-vector product;
biases, gates, norms, scalar decay and rank-1 outer products excluded):

| part | width 48 | width 64 |
|---|---:|---:|
| stem | 672 | 896 |
| FiLM | 576 | 768 |
| block projections (q/k/v/erase/write) | 30 720 | 40 960 |
| block beta | 384 | 512 |
| block recurrence (2 × heads × 16²) | 4 096 | 4 096 |
| block out | 6 144 | 8 192 |
| head | 144 | 192 |
| **total per tick** | **42 736** | **55 616** |

Both defaults sit inside the 30–70 kMAC/tick band and below the ~50 kMAC
Cortex-A72 target at width 48. The breakdown is checked against forward hooks
on every `nn.Linear`, with the two einsums counted from the executed state
shape; it is not a number copied from WORLD07's estimate.

## Tests

`ai/tests/test_gdn2_student.py` adds 8 gates:

1. **MAC** — pinned breakdown and totals (42 736 / 55 616), hook-counted runtime
   agreement for two configurations, state-shape-checked recurrence term,
   batch/blocks scaling, `batch=0` refusal.
2. **Causality** — perturbing frames or conditioning at `t+1…` leaves outputs at
   `≤t` bit-identical and changes later outputs; a `step` sequence from reset
   equals the pure window sweep; `reset_state()` restores fresh behaviour.
3. **Determinism** — same seed is bit-identical across two models despite a
   disturbed global RNG, repeated sweeps agree, a different seed separates,
   sequential `step` outputs agree tick by tick, train/eval modes agree.
4. **Bounds** — 1500 adversarial ticks (40× frames, 8× conditioning): finite
   corrections within ±`g_max`, confidence in [0,1], finite state with peak
   absolute entry 0.375 (measured) against a pinned ≤4.0 gate.
5. Config and input validation (width range, decay, head count, shape).

No accuracy, latency or "beats LRS-TCN" claim is made anywhere. WORLD07 §6
requires a 3-seed A/B with held-out RMSE / p99 / memory at a matched MAC budget
before this candidate can replace `StatefulTCN`; none was run here. No training
or distillation code was added.

## Verification

- `uv run python -m pytest tests/test_gdn2_student.py -v` → **8 passed**.
- `uv run python -m pytest -q` (whole AI suite) → **286 passed in 140.46 s**.
- `uv run ruff check` + `uv run ruff format --check` on the two files → clean.
- Environment: CPU-only torch 2.15.0+cpu, Python 3.13.15.
- Local commit on `main`; no push.
