# Bounded adversarial curriculum (AICURR)

**Date:** 2026-11-24 · **Scope:** `src/potok_ai/curriculum.py`, the `--adversarial`
stage hook in `src/potok_ai/train_teacher_v2.py`, `tests/test_curriculum_adv.py`.
No physics constant, gate threshold, model weight or reward is changed. No flight
claim is made.

## Why

WORLD09 §1 row 5 and IDEAL_GENERATOR §2 (S6) call for a bounded adversary on
training material, with a **~30 % win-rate cap** and `V_max` clamp, and with
evaluation cells 16–19 never used for selection (WORLD05 A3, references A3/A6).
The staged curriculum previously assigned stages by `scenario_id % 4`; this
module adds the adversary on top of the existing cumulative stage scheduler.

## Design

A "win" means *the adversary was allowed to perturb one training window*. The
cap is therefore a **sampling cap**, not a measured win rate from a rollout.
Each window gets one draw from a named child RNG; on a win the existing sensor
domain transforms run, on a loss the window is returned unchanged. This keeps
the majority of every batch clean, which is the learnability guard.

| Knob | Default | Meaning |
|---|---|---|
| `win_rate_cap` | `0.30` | hard ceiling on the per-window perturbation probability |
| `gust_gain` | `1.25` | above-training-range amplitude multiplier (1.25 × 20 m/s = 25 m/s) |
| `dropout_prob` | `0.5` | chance a sensor channel is held at its first reading |
| `delay_prob` | `0.5` | chance a sensor channel arrives 1–2 frames late |
| `delay_max_steps`, `gust_shift_max` | `2` | frame limits of the two time transforms |

`gust_gain`, delay jitter and dropout are **not reimplemented**: they are the
existing `augment.py` transforms with its default 8 raw-sensor channel set, so
the six derived physics columns stay consistent with their telemetry. The gain
scales channel deviation, not a physical wind vector — the dataset has no wind
truth column, so no `V_max` clamp is claimed here (WORLD05's A3 guard is only
partially transferable).

Budget ramp across the cumulative stages:

| Stage | `calm` | `gust` | `shear` | `sensor_fault` |
|---|---|---|---|---|
| perturbation probability | 0 | 0.10 | 0.20 | 0.30 |

`AdversaryConfig` refuses a cap outside `(0, 1)`, a gain below 1, and invalid
probabilities. `AdversarialCurriculum.perturb` re-clamps any caller-supplied
intensity to the cap, so the budget cannot be bypassed.

## Determinism and held-out discipline

Every stochastic decision comes from `stream_seed(seed, name)` = the first 8
bytes of `sha256(f"{seed}:curriculum:adv:{name}")`. Separate `win:` and
`augment:` substreams mean changing the win budget does not move the transform
draws. No global RNG is read or reseeded.

`refuse_eval_episode` raises on `held_out=True`, on scenario ids 16–19, and on
the `scenario_18`-style name fallback. `AdversarialCurriculum.select` refuses
before shuffling, and the trainer refuses again at the seam and inside the
train-only window builder — before the first augmentation draw. Evaluation and
RLVR windows use `_window_batches(..., train=False)` and are bit-identical to
the unaugmented dataset.

## Trainer hook

`--adversarial` is **off by default** (a plain run keeps its old config hash and
metrics). When on, `run()` builds one `AdversarialCurriculum(seed=cfg.seed)`,
sets a stage-scoped context for the pipeline stage, and records in each stage's
`metrics.json` entry:

```
details.adversarial = {enabled, stage, seed, win_rate_cap, intensity, offered, won}
```

Pipeline stages map to the curriculum ramp as
`sft → calm (0)`, `privileged → gust (0.10)`, `dagger → shear (0.20)`,
`distill → sensor_fault (0.30)`. Only the training window builders of
`sft`, `privileged` and `distill` read the context; DAgger's on-policy dataset
and the RLVR verifier do their own thing and are not augmented. `offered/won`
make the cap visible without a profiler.

## Honest limits

- This is a bounded sampler, **not** a learned adversary and **not** a measured
  RL win rate. No rollout verifier feeds the win rate back into the budget.
- No `V_max` clamp or stage-specific turbulence σ bound: no wind vector exists
  in the training telemetry. Only the three sensor-domain transforms are used.
- Whether this improves held-out lateral RMS or apogee is **unmeasured**; no A/B
  was run. The gates establish interfaces and the refusal/determinism contract.
- No physics, gate threshold, reward weight or model weight was tuned.

## Verification

```
.venv/bin/ruff check src/potok_ai/curriculum.py src/potok_ai/train_teacher_v2.py tests/test_curriculum_adv.py
.venv/bin/python -m pytest -q tests/test_curriculum_adv.py tests/test_curriculum.py
```
