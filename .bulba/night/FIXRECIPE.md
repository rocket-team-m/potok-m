# FIXRECIPE — v3 LRS recipe root-cause fix (MUON02)

**Date:** 2026-11-23
**Repo:** `/home/sehaxe/potok-m/ai` (workspace `/home/sehaxe/potok-m`)
**Scope:** `src/potok_ai/student.py`, `src/potok_ai/engine.py`,
`out/v3_final/driver.py`, `tests/test_student_gate.py` (new),
`docs/TRAINING_RECIPE.md` (new). No physics constants, no gate thresholds, no
held-out data touched. No subagents.

## Finding and fix

MUON02 ranks 1–2: the v3 target reaches ±10 m/s² (train RMS 7.70) while the
student emitted `tanh(head_corr) * tanh(|g_max|) < 1`, and the shipped driver
trained `--limit 2` on two scenario-0 episodes and evaluated on OOD scenarios
16–19. The recipe could not fit its target regardless of optimizer or epochs.

* `student.py`: added frozen, validated `CommandEnvelope(accel=10.0)` (the
  simulator's `MAX_COMMAND_MPS2` unit contract). `g_max` is now the command
  bound **in m/s²** and `step()` uses `g_max.abs().clamp(max=command_ceiling)`
  instead of `tanh(|g_max|)`. `command_ceiling` is a plain float config field so
  saved configs remain loadable with `torch.load(weights_only=True)`. Export
  layout is unchanged (`g_max` still f32).
* `engine.py`: `TrainConfig.command_envelope` is the typed default (10.0);
  `LRSAdapter.build` initializes the student bound from it; `LRSAdapter.batches`
  raises loudly when the target RMS exceeds the envelope. With the curriculum
  off, every epoch sees a deterministic `(seed, epoch)` shuffle of all 80 train
  episodes. `--episodes-per-scenario N` is a smoke-only stratified cap
  (default `None` = full split).
* `driver.py`: LRS arm dropped `--limit 2`, uses `--no-curriculum`, and
  `--seeds` defaults to `(1, 2, 3)`. Threshold/gate flags untouched.

## Verification

* `ruff check` on all four changed/new Python files: clean.
* Full suite: `.venv/bin/python -m pytest -q` → **347 passed** (was 345 before;
  +6 new gate tests, 2 pre-existing tests needed the pickle-safe config fix).
* `tests/test_student_gate.py`: 6 passed — typed default/validation, head
  reaches ±10 with export roundtrip, engine build uses the envelope, loud
  failure above envelope, driver full-split/3-seed recipe, real v3 target range
  and scenario-16–19 disjointness.
* Held-out files were read for evaluation only; no write. Physics constants
  and ratio/ECE thresholds were not modified.

## Smoke — exact metrics

Short smoke on real v3 data: **1 epoch, 1 episode per training scenario (16 of
80 train episodes), 3 seeds**, driver flags otherwise unchanged, `--calibrate`
on; outputs in `/tmp/opencode/fixrecipe_smoke/s{1,2,3}`.

| seed | rmse_train | rmse_held_out | ratio | mae | max_error | ece | ece_held_out | conformal_coverage | config_hash | elapsed_s |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 4.25193574357606 | 4.536926046596469 | 1.0670260136106198 | 5.04963066186435 | 7.345600128173828 | 0.0 | 0.0 | 0.951895244658856 | 4300ebe247b8f76c | 348.5 |
| 2 | 4.822887190832863 | 4.762302338308947 | 0.9874380531564012 | 4.4853500636670605 | 9.13156509399414 | 0.04023865245854303 | 0.04021895439431163 | 0.9501033769813921 | 871b48a32e64f1f1 | 362.0 |
| 3 | 4.534507253694213 | 4.503112796591868 | 0.9930765449593738 | 4.398813141139929 | 8.676769256591797 | 0.02404840218535036 | 0.0240174172391171 | 0.9493452791178497 | da2ac4a1150625f6 | 364.6 |

Shipped 100-epoch recipe for comparison (**not** the same budget or split —
shown only as the pre-fix reference, not an A/B): seed 1 rmse_train 7.3394,
rmse_held_out 7.6301, ratio 1.0396, max_error 10.007, ece ≈ 5e-18, coverage
0.174 (`ai/out/v3_final/results.csv`).

Reading honestly: after a *single* epoch on a *stratified subset* the errors
dropped to ~4.3–4.8 m/s² and the ratio stayed near 1; this shows the head can
now move against a ±10 target. It is **not** the full 100-epoch full-split
recipe, and **no A/B win is claimed**. Seed 1's ECE of exactly 0 is a collapsed
confidence head on this tiny budget; seeds 2–3 have non-degenerate held-out
ECE (0.040 / 0.024) and ~0.95 conformal coverage. All ratio (≤1.25) and ECE
(≤0.05) gates pass without threshold changes.

## Not done / owner follow-up

* No 100-epoch full-split production run and no closed-loop `scripts/ab_v3.py`
  run in this task (time budget); those are the next measurements.
* **Deployment seam (loud):** `src/student_runtime.rs:417` still computes
  `correction_bound = g_max.abs().tanh()`. A v3 export now carries `g_max ≈ 10`,
  so firmware would bound corrections at ≈1 m/s² — the same defect on the Rust
  side. The Python A/B harness uses the Python export and is correct. Aligning
  the firmware reader is outside this subagent's file scope; do not claim a
  flown win until it is fixed and re-verified.
* Committed locally on `main`; nothing pushed.
