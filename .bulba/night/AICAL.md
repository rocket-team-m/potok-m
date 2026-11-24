# AICAL — split-conformal confidence in the staged teacher-v2 trainer

**Date:** 2026-11-24 · **Repo:** `/home/sehaxe/potok-m/ai` · **Commit:** `f2b4365` (local only, no push)

## Scope

Only `src/potok_ai/conf_integration.py`, `src/potok_ai/train_teacher_v2.py` (confidence hook),
`tests/test_conf_integration_v2.py` (new) and `docs/CONFIDENCE.md` (new). No subagents.
No gate threshold, reward weight, simulator or flight code was touched.

## What landed

- **Trainer hook.** `train_teacher_v2.run()` now calls `_confidence_hook` once, after all
  requested stages. It collects inner-held-out confidence logits/errors through the existing
  `_predict` on 14+6 sensor/conditioning columns (eval cells 16..19 are still refused by
  `partition_episodes`), then calls `conf_integration.split_confidence_score`.
  For `--stage all` / `distill` the saved `student.pt` is reloaded and scored with the
  student path; otherwise the final teacher is scored. The score describes the network
  whose metrics are written next to it.
- **Disjoint split, deterministic.** A `default_rng(seed)` permutation is cut 50/50; the fit
  half feeds the calibration map and the conformal fit, the score half only ECE/coverage.
  Same seed → same split, scores and byte-identical `conformal.json`. Small fit halves raise
  the effective `alpha` to `1/(n_fit+1)` and record it in the export.
- **Finite checks / loud failures.** Hook refuses empty or non-finite predictions;
  `conformal_export` refuses non-finite payloads; `frozen_step_map` additionally refuses a
  missing, malformed or non-finite step map before the file is written.
- **Explicit `ece_source`.** Top-level `ece_source="held-out"` (an honest disjoint label in
  `teacher_gates.HONEST_ECE_SOURCES`), `ece == ece_held_out` (the disjoint score), and
  `conformal_coverage` in `metrics.json`. Each `stages.<name>.ece` remains a diagnostic and
  is explicitly labelled `ece_source="in-sample"`.
- **Frozen edge step map.** `conformal.json` is sorted-key JSON with two ascending upper
  edges in `(0,1]` and levels `(0,1)`; the inclusive-lower-edge boundary matches
  `apply_confidence` and the edge reader's step-map shape. The current edge binary still
  loads `calibration.json` — this is an exported companion artifact, not a wired runtime
  file. No edge wiring is claimed.

## Verification

- New focused gates: `tests/test_conf_integration_v2.py` — **6 passed** (disjoint report and
  export, seed determinism, frozen step map / edge-route equality, malformed-map refusals,
  non-finite prediction refusal, distill pipeline + CLI end-to-end).
- Full `pytest -q`: **402 passed** (237.65 s).
- Post-settle focused re-run of 5 related test modules: **52 passed**.
- `ruff check` on all four touched files: **clean**. (`ruff check .` still reports 11
  pre-existing errors in `out/sweep/` and `out/v3_final/` artifacts — untouched.)

## Caveats / coordination

- A concurrent worker edited `train_teacher_v2.py` during this task (adversarial curriculum
  context). My commit stages **only the AICAL hunks** of that file via `git update-index`;
  their in-progress edits remain unstaged in the worktree and are not attributed to this
  commit. Tests above were run against the worktree containing both sets of edits.
- No gate tuning and no flight claim: `teacher_gates.py` and EVAL01 limits are unchanged;
  this lands evidence production only. No A/B, closed-loop gain or deployment result.
