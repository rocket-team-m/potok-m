<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
# v3 LRS training recipe

Authoritative recipe for the stateful student (`lrs`) on the v3 `g_accel`
dataset. This replaces the shipped `out/v3_final` recipe that trained 200
steps on two scenario-0 episodes and scored OOD ±10 m/s² held-out data.

## Why the old recipe could not work (MUON02)

* v3 target `[g_accel_y, -g_accel_x]` has train RMS **7.70**, max **10.0** m/s².
* The head emitted `tanh(head_corr(z)) * tanh(|g_max|)`. With `g_max = 1`, and
  for *any* finite `g_max`, the bound is **< 1** — the target was unreachable.
  Observed v3 LRS RMSE 7.339 train / 7.630 held-out (≈ the target RMS itself).
* `--limit 2` selected two easy scenario-0 episodes (|mean| 0.36 / 0.17),
  while evaluation used held-out scenarios 16–19 at ±10 m/s².

## What changed

1. **Typed output envelope.** `student.py:CommandEnvelope` is a frozen,
   validated dataclass whose `accel` field defaults to **10.0 m/s²**, the
   simulator's `MAX_COMMAND_MPS2` clamp. `TrainConfig.command_envelope` carries
   it into `LRSAdapter.build`; `StudentConfig.g_max` becomes the *bound in
   m/s²* (no `tanh` compression), capped by that ceiling. Export still writes
   `g_max` as an f32 header field, so the format is unchanged.
2. **Full train split, randomized.** The driver's `lrs` arm drops `--limit`
   and uses `--no-curriculum`: the staged curriculum is an `id % 4` placeholder
   and would restrict early epochs to a quarter of the scenarios. With it off,
   every epoch sees all 80 train episodes (scenarios 0–15) in a deterministic
   `(seed, epoch)` shuffle. `--episodes-per-scenario N` is a smoke-only cap
   that keeps N episodes from **every** training scenario (default `None` =
   full split).
3. **Three seeds.** The driver's `--seeds` defaults to `(1, 2, 3)`.
4. **Loud seam check.** `LRSAdapter.batches` refuses a target whose RMS exceeds
   the envelope instead of silently training an unfittable objective.

Gate thresholds are untouched: `ratio <= GENERALISATION_LIMIT` (1.25),
`ece <= DEFAULT_TARGET_ECE` (0.05), and the `--calibrate` split-conformal
fields (`ece_held_out`, `conformal_coverage`) keep their honest disjoint-fit
semantics. Held-out scenarios 16–19 are never passed to a training batch;
their files are not written by this recipe.

## Run

Production (full split, 3 seeds, 100 epochs):

```sh
cd ai
.venv/bin/python out/v3_final/driver.py            # seeds 1,2,3
```

Short smoke (1 epoch, 1 episode per scenario, 3 seeds):

```sh
cd ai
PYTHONPATH=src .venv/bin/python -m potok_ai.engine \
  --model lrs --stage train \
  --data out/v3_final/data --out /tmp/opencode/fixrecipe_smoke/s1 \
  --epochs 1 --seed 1 --device cpu \
  --episodes-per-scenario 1 --eval-limit 1451 --lr 0.03 \
  --no-curriculum --residual --calibrate
```

The regression gate is `tests/test_student_gate.py`; run with
`.venv/bin/python -m pytest -q tests/test_student_gate.py`.

## Deployment seam (out of scope, owner follow-up)

The Rust firmware reader `src/student_runtime.rs:417` still derives its bound
as `g_max.abs().tanh()`. A v3 export now carries `g_max ≈ 10`, so that reader
would bound corrections at ≈ 1 m/s² — the old defect, on the deployment side.
The Python A/B harness (`scripts/ab_v3.py`) uses the Python export and is
correct. Aligning the firmware reader to the typed envelope requires a format
or reader change and is **not** part of this fix; do not claim a flown A/B win
until it is done and re-verified.
