# RETRAIN-NIGHT — final retrain with impact objective and conformal confidence

**Date:** 2026-11-24 · **Repo:** `/home/sehaxe/potok-m` (branch `main`). Code/artifacts commit `4373ad8`; this report is committed separately. Local only, **nothing pushed**.
**Mode:** no subagents. No trainer, physics, gate, threshold, reward, envelope or confidence code was edited.

## Scope

- `scripts/train_night_final.py` (new) — the retrain driver.
- `ai/out/night_final/**` — `runs/s{1,2,3}/{metrics.json,conformal.json,run.log,student.pt}`,
  `adversarial/runs/s{1,2,3}/...`, `adversarial_summary.json`, `summary.json`,
  `shield.json`, `exports/{manifest.json,teacher_v2_s{1,2,3}.pt}`, and the
  1-epoch `smoke/s1/` preflight artifacts. Student `.pt` files stay `.gitignore`d,
  same policy as earlier retrains.
- `ai/docs/TRAINING_RESULTS.md` — new current section inserted between explicit
  `night-final` markers; the earlier allfix section is preserved below it.
- this report.

No file outside those four paths was staged. The pre-existing uncommitted work in
`sim/`, `ai/out/v3_final/`, `ai/tests/` and elsewhere was left untouched.

## What ran

The shipped four-stage default `sft -> privileged -> dagger -> distill` with the
train-split POTOKTRJ v3 truth sidecar, **3 seeds x 100 epochs, CPU**, batch 64,
window 12, `--limit 1451` (caps WINDOWS per builder, not episodes), the same
numeric recipe as RETRAIN-ALLFIX. No physics constant, gate, reward weight,
command envelope, confidence threshold or data file was changed.

Two upgrades that landed after allfix are active through shipped defaults, not
through new knobs:

- **AIOBJ** — the privileged impact objective. The driver deliberately does
  **not** pass `--legacy-objective`; every seed records
  `stages.privileged.details.objective = {source: impact, legacy: false,
  reason: null, formula: IMPACT_FORMULA}`.
- **AICAL** — split-conformal confidence. The trainer's own hook writes
  `conformal.json` per seed; the driver validates the frozen two-bin step map and
  reads `ece_source = "held-out"`, `ece == ece_held_out`, and
  `conformal_coverage` from `metrics.json` without recomputing them.

Inherited defaults re-checked per seed: AIFIX1 ±10 m/s² student bound, AIFIX2
real train-split truth (`privileged_batches = 2300`, `privileged_source =
privileged_teacher`), AIFIX3 on-policy DAgger (`student_induced_rows = 118960`,
SHA-proven rollout), SHIELD02 residual policy (`gain` ≈ −5 with 0 clamped rows),
SHIELD01 verified in `shield.json` (it is not invoked by the staged trainer),
DAgger default. RLVR stayed opt-in and was not run.

## Adversarial curriculum (AICURR)

**RUN — not NOT RUN.** The opt-in arm was enabled by adding only
`--adversarial`; no gate, threshold or recipe file was edited and all three
teacher gates pass on it. It is reported as a separate descriptive arm, not as
an A/B result: `ab_win_claimed: false` in both summaries.

| arm | seeds | held-out RMSE mean +/- std (m/s²) | teacher gates |
|---|---:|---:|---|
| default (impact + conformal) | 1,2,3 | **7.358037 +/- 2.147744** | rmse_ratio PASS, ece_source PASS, ece PASS |
| `--adversarial` opt-in | 1,2,3 | 6.966325 +/- 3.381028 | rmse_ratio PASS, ece_source PASS, ece PASS |

Stage-scoped adversary evidence (all seeds): `sft` intensity 0.0 / 0 wins,
`privileged` 0.1, `dagger` 0.2, `distill` 0.3, hard cap 0.30; ~10k wins in
privileged, ~40k in distill. Eval and DAgger's own dataset are never perturbed
by the seam. No V_max / per-stage σ clamp: the dataset has no wind vector.

Two driver-side preflight failures occurred and are reported, not hidden: the
first attempt crashed in `export_students()` on a top-level field the upgraded
trainer no longer emits (the field is mirrored from stage details now), and the
first adversarial gate crashed on an exact float comparison of the ramp
intensity (0.09999999999999999 vs 0.1; now `math.isclose`). Both were driver
assertion/export defects; the trainer, gates and data were unchanged, and the
finished artifacts were reused rather than retrained. A 1-epoch smoke under
`smoke/s1/` passed the same assertions before either full arm.

## Honest comparison to RETRAIN-ALLFIX — no win claim

| run | held-out RMSE, mean +/- std (m/s²) |
|---|---:|
| RETRAIN-NIGHT default | 7.358037 +/- 2.147744 |
| RETRAIN-ALLFIX | 7.855551 +/- 2.604600 |

Difference of means: **−0.497515 m/s²**, within one allfix sample std
(2.604600). The two runs change the privileged objective and the confidence
measurement together; no arm isolates either, and the three-seed spread is
large (seed 1 = 5.474856, seed 2 = 9.697134, seed 3 = 6.902121). There is no
closed-loop harness, no independent flight-truth anchor, no positive control
and no unweakened closed-loop gate evaluation. `ab_win_claimed: false`:
**no A/B win is claimed**, and neither is one for the adversarial arm.

## Provenance and gates

Frozen input digests verified by the driver before any step: train
`34b35e0f…30b6`, held-out `fdc1d14…8907`, sim config `9c34e71…5e91`, train-split
truth sidecar `7f25bb6…cb8`. Stage-order pin and CLI-vs-config hash self-check
pass for both arms. Per-seed `assert_night()` checks the fix evidence, impact
objective, conformal fields and (for the adversarial arm) the stage ramp and
cap; it ran on fresh and reused artifacts. All three teacher gates
(`rmse_ratio` ≤ 1.25, disjoint `ece_source`, `ece` ≤ 0.05 with coverage in
(0,1]) pass on both arms. Closed-loop gates (`seed_spread`,
`closed_loop_improvement`, `oracle_positive_control`) remain **not evaluated**.

## Binding limitation

AIFIX3's `world_step` advances only onboard sensor features through the
unchanged `physics_baseline`/`final_command` plus named noise — a surrogate,
not a Rust simulator step on (state, command). The POTOKTRJ sidecar is emitted
by that same simulator, so generator and verifier share a source; it is not an
independent flight-truth anchor. The impact objective is a training term using
measured sensor columns and true velocity, not a closed-loop measurement. A
held-out RMSE near the target's own RMS (~7.7 m/s²) is the expected shape of
that binding, not evidence of a controller improvement. No flight claim.

## Verification

- `ai/.venv/bin/ruff check scripts/train_night_final.py` — clean;
  `ruff format --check` — clean (also after the final fixes).
- `cd ai && .venv/bin/ruff check src tests` — clean.
- `cd ai && .venv/bin/python -m pytest -q` — **423 passed in 138.87 s**, exit 0.
- Driver `verify_inputs()` four sha256 pins, stage-order pin, CLI/config hash
  self-check (both arms), per-seed `assert_night()`, `frozen_step_map()` check
  of every `conformal.json`, and counted `OnlineShield` smoke (1 accept + 4
  veto reasons, fallback applied) all green.
- No push, no force, no history rewrite.
