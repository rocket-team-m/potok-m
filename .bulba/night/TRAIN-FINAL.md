# TRAIN-FINAL — post-FIXPUMP staged teacher v2, 100 epochs, 3 seeds

**Date:** 2026-11-24
**Repo:** `/home/sehaxe/potok-m` (work scope `/home/sehaxe/potok-m/ai`)
**Status:** done. Local commit only, nothing pushed. No subagents.
**Scope:** `scripts/train_final_v3.py` (new), `ai/out/final_v3_postpump/**`,
`ai/docs/TRAINING_RESULTS.md` (new), this report. No other file edited.

## Why this run exists

Everything that preceded FIXPUMP measured a simulator whose placeholder
aerodynamic normal force was **parallel** to the lateral relative wind and did
positive work (`FIXPUMP.md`). The corrected, energy-conserving dynamics change
the data: the pre-FIXPUMP held-out export hashes `5848e222…`, while the
post-FIXPUMP zero-command baseline rebuilt by `RECALC01.md` hashes
`fdc1d140…`. Training on the shipped `ai/out/v3_final/data` would have meant
training on obsolete physics. This run regenerates the data and retrains.

## Data (post-FIXPUMP, verified)

1. Rebuilt the simulator CLI from the current tree:
   `cargo build --release --manifest-path sim/Cargo.toml` (the pre-existing
   binary was dated 00:01, before the FIXPUMP commit).
2. Regenerated both splits with the frozen v3 config:
   `potok-cli generate-dataset --seed 1 --train-episodes 80
   --held-out-episodes 20 --out ai/out/final_v3_postpump/data
   --config ai/out/v3_final/sim.toml` (quality report: 14/14 checks passed).
3. The generated held-out `validation.bin` is **byte-identical** to the
   post-FIXPUMP baseline recorded in `RECALC01.md` (`fdc1d1400b1c…`), which
   proves the rebuilt binary and the frozen config reproduce the corrected
   physics. The driver refuses stale/missing inputs by sha256 before training.

| artifact | sha256 |
|---|---|
| train `validation.bin` (80 episodes, scenarios 0–15) | `34b35e0ff67f7eeecbd895b2efde8bebcfa159c1a3e78379da6750338b6930b6` |
| held-out `validation.bin` (20 episodes, scenarios 16–19) | `fdc1d1400b1c5f111e21f79dead37afa1dd9f058fe457edb78037dc6ebe89807` |
| frozen `sim.toml` | `9c34e71f9d10bcf081c31f15f5332f7bb2fac04dddb28009a27cd30987ab5e91` |

## Recipe (fixed before the run, no tuning)

Five stages in order: `sft -> privileged -> dagger -> rlvr -> distill`, using
`potok_ai.train_teacher_v2` with **100 epochs** on CPU, seeds **1, 2, 3**,
`--batch 64`, `--window 12`, `--limit 1451`, the authoritative v3 command
targets `[g_accel_y, -g_accel_x]` in m/s² (the ±10 m/s² simulator envelope is
reached through the frame-local physics baseline plus the bounded residual
correction), and all other knobs at their documented defaults. No physics
constant, gate or threshold was changed; no GPU was used.

`--limit` caps **windows per batch builder**, not episodes: all 80 training
episodes are loaded and partitioned by scenario for every seed (inner split
is two held-out training scenarios, seeds 7/9 for seed 1), DAgger uses all of
them, and held-out scenarios 16–19 are refused before any training batch. The
cap is recorded in every `metrics.json`. It is **not** the old `--limit 2`
two-episode recipe, but it does mean the gradient stages see roughly one
episode-equivalent of windows per builder; this is the honest trade that made
a 100-epoch CPU run finish in ~10 min/seed (an uncapped pass is ~10M window
updates per stage). All modules were the real siblings, not local fallbacks:
`jepa_v2`, `dagger`, `privileged_teacher`, `gdn2_student`.

## Measurement contract

The trainer writes `metrics.json` with `ratio`, `ece`, `reward_provenance`,
`module_sources` and per-stage reports. The driver adds the fields the
privileged-teacher gate requires:

- `ece_source = "held-out"` — the trainer's confidence rows come only from
  its inner 2-scenario split, which receives no gradient step;
- `ece_held_out` / `conformal_coverage` — recomputed by scoring those same
  rows through the shipped `split_confidence_score` bridge (disjoint
  fit/score halves, `conformal.json` written beside each run). No in-sample
  ECE is ever promoted to a gate field;
- `reward_provenance` is copied verbatim from the RLVR stage (source, formula,
  weights, seed, config hash, sample/reject counts, per-term means).

Three teacher gates are evaluated: `rmse_ratio`, `ece_source`, `ece`. The
closed-loop gates (`seed_spread`, `closed_loop_improvement`,
`oracle_positive_control`) need an A/B report and are out of scope.

## Results

Held-out RMSE (distill-stage student, inner scenario split), mean ± sample
standard deviation over three seeds: **8.039883 ± 0.186312 m/s²**.

| seed | train RMSE | held-out RMSE | ratio | trainer ECE | disjoint ECE | conformal coverage |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.719484 | 8.246161 | 1.0682 | 0.51785 | 0.0003 | 0.9311 |
| 2 | 7.372674 | 7.989650 | 1.0837 | — | 0.0028 | 0.9683 |
| 3 | 7.402216 | 7.883837 | 1.0651 | — | 0.0025 | 0.9353 |

The raw trainer ECE is ≈0.52 on seed 1 (≈0.6 at the RLVR stage); the
`ece_held_out` column is the **disjoint recalibrated** score, not a copy of
that number. Both are preserved in each `metrics.json` (`ece` vs
`ece_held_out`) so the distinction stays auditable.

RLVR reward provenance (identical formula and weights on all seeds; 580 400
candidate samples each, 273 648 / 291 866 / 303 797 rejected by the hard
`|u| ≤ g_max` gate): mean lateral gain 0.658 / 0.620 / 0.623, effort 0.827 /
0.849 / 0.858, risk 0.920 / 0.928 / 0.946, total −0.017 / −0.113 / −0.129.
Full dicts are in `runs/s{1,2,3}/metrics.json`.

**Gate verdict: all three teacher gates PASS** (`rmse_ratio` ≤ 1.25 on every
seed; `ece_source` is a disjoint label; disjoint ECE ≤ 0.05 with non-zero
conformal coverage). This is a consistency result, **not** a quality result:
held-out RMSE ≈ 8 m/s² is close to the v3 target's own RMS (≈8 m/s²), so the
exported student has not learned the command to useful accuracy. The low
disjoint ECE describes a well-behaved confidence head, not a small command
error. No closed-loop A/B was run and no win is claimed.

## Student exports

`ai/out/final_v3_postpump/exports/teacher_v2_s{1,2,3}.pt` are the
distill-stage student `state_dict`s (source `gdn2_student`), copied verbatim
from the run directories with a `manifest.json` that records sha256, config
hash, student source and held-out RMSE per seed. They are **artifacts for a
later A/B**, not evidence of one. The manifest states this explicitly.

## Verification

- `.venv/bin/python -m pytest -q` → **347 passed** (no test edited).
- `ruff check` / `ruff format --check` on `scripts/train_final_v3.py`: clean.
- Driver refuses a stale/missing input by sha256 before any training; it also
  asserts `config_hash`, `stages_completed`, student-source agreement and
  finite confidence predictions before writing a gate field.

## Honest limits / follow-up

- A concurrent edit to `ai/src/potok_ai/train_teacher_v2.py` landed after
  this run and made RLVR an opt-in checkpoint-selection stage instead of the
  fifth training stage. The artifacts here are from the five-stage trainer as
  it stood at run time (all 5 stages in every `metrics.json`, with RLVR policy
  reward provenance). The driver therefore pins the five-stage contract in a
  local `EXPECTED_STAGES` constant and refuses to relabel a mismatched
  artifact; do not expect a fresh `--stage all` run against the edited trainer
  to reproduce this reward provenance.
- The 1451-window cap means the gradient stages of a seed see one episode's
  windows, so the ~8 m/s² RMSE may be a budget artifact rather than an
  architectural ceiling. An uncapped or stratified full-window 100-epoch run
  is the next measurement if the quality question matters; no gate is relaxed
  to make that cheaper.
- The teacher's internal residual bound is `LoopWMConfig.g_max = 1.0` m/s²;
  the ±10 m/s² envelope is carried by the physics baseline. This is the
  trainer's existing seam, not something this task tuned.
- The post-FIXPUMP oracle is no longer a valid positive control (`RECALC01.md`):
  it fails its own apogee gate on 2 of 3 seeds. Any later A/B must first
  re-specify that control rather than reuse the pre-FIXPUMP one.
