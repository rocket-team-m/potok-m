# RETRAIN-DAGGER — POSTPIVOT default on the post-FIXPUMP split

**Date:** 2026-11-24
**Repo:** `/home/sehaxe/potok-m` (main, HEAD `c0d02e4`)
**Status:** done. Local commit only, nothing pushed. No subagents.

## Scope

- `scripts/train_final_v3.py` — modified.
- `ai/out/final_v3_postpump_dagger/**` — new: `driver.log`, `summary.json`, `exports/manifest.json`, `runs/s{1,2,3}/{metrics.json,conformal.json,run.log}`.
- `ai/docs/TRAINING_RESULTS.md` — rewritten by the driver.
- this report.

No trainer, test, physics, gate, weight or old-artifact file was edited. `ai/out/final_v3_postpump/` is byte-for-byte untouched (`git status` clean for that path).

## Why this run exists

POSTPIVOT (`e3da20c`) changed the trainer default to `sft -> privileged -> dagger -> distill` and made RLVR an opt-in checkpoint-selection stage (no policy gradient). TRAIN-FINAL's driver still asserted the old five-stage contract and required non-empty `reward_provenance`, and its results page described a pipeline the default no longer runs. This task re-runs the new default on the same frozen post-FIXPUMP data and rewrites the driver and results page to match what actually ran.

## Driver changes (no recipe tuning)

- `EXPECTED_STAGES` pinned to the four-stage default; `verify_inputs()` refuses if the trainer's `STAGE_ORDER` moves again, instead of silently relabelling an artifact.
- Output retargeted to `ai/out/final_v3_postpump_dagger`; old five-stage artifacts left in place for the comparison.
- Dropped the "RLVR must have run" requirement: a default run must have `reward_provenance == {}`, and a non-empty block is refused loudly. `rlvr_used: false` is recorded in the summary and export manifest.
- Every numeric knob is unchanged from the consistency pass: 100 epochs, CPU, seeds 1/2/3, batch 64, window 12, `--limit 1451`, lr 2e-3, student-lr 1e-3, width 16, heads 2, max-iters 2, jepa-weight 0.05, entropy-weight 1.0, dagger-rounds 1, n-inner-held-out 2, calib-tol 0.15, adamw. `--limit` caps WINDOWS per batch builder, not episodes: the full 80-episode train file is loaded and partitioned by scenario and DAgger uses all of it; held-out scenarios 16-19 are refused before any batch.
- API drift handled: SHIELD02 (landed mid-task) added a required `cfg=` argument to `_predict` and enabled a residual policy by default. `honest_confidence()` now passes `cfg`; the refined metrics carry `residual_policy` provenance. Seed 1's training had already finished when the first driver pass failed at post-processing, so it was REUSEd (hash-checked), not retrained. Seeds 2 and 3 were trained fresh with the same trainer.

## Data (sha256 verified by the driver before any step)

| artifact | sha256 |
|---|---|
| train `validation.bin` (80 episodes, scenarios 0-15) | `34b35e0ff67f7eeecbd895b2efde8bebcfa159c1a3e78379da6750338b6930b6` |
| held-out `validation.bin` (scenarios 16-19) | `fdc1d1400b1c5f111e21f79dead37afa1dd9f058fe457edb78037dc6ebe89807` |
| frozen `sim.toml` | `9c34e71f9d10bcf081c31f15f5332f7bb2fac04dddb28009a27cd30987ab5e91` |

These are the same post-FIXPUMP digests TRAIN-FINAL used; no data was regenerated or substituted.

## Results

Held-out RMSE over three seeds: **7.902916 +/- 0.498808 m/s^2** (sample std).

| seed | train RMSE | held-out RMSE | ratio | disjoint ECE | conformal coverage |
|---:|---:|---:|---:|---:|---:|
| 1 | 7.137926 | 7.572275 | 1.0609 | 0.0259 | 0.9545 |
| 2 | 7.352789 | 7.659804 | 1.0418 | 0.0108 | 0.9614 |
| 3 | 7.911737 | 8.476669 | 1.0714 | 0.0028 | 0.9394 |

Config hashes: `7bb5bc69ca995bb6` / `852f14851e8c745c` / `aafe4d567d73f945`. Teacher gates `rmse_ratio`, `ece_source`, `ece` all PASS. Closed-loop gates (`seed_spread`, `closed_loop_improvement`, `oracle_positive_control`) are not evaluated: no A/B report, no flight-truth sidecar, no positive control.

## Comparison to the old consistency pass — no win claim

| run | stages | held-out RMSE mean +/- std (m/s^2) |
|---|---|---:|
| earlier pass (`ai/out/final_v3_postpump/`) | sft -> privileged -> dagger -> rlvr -> distill | 8.039883 +/- 0.186312 |
| this run | sft -> privileged -> dagger -> distill | 7.902916 +/- 0.498808 |

Difference of means: **-0.136967 m/s^2**. It is not evidence of improvement:

- the new seed spread (0.498808) is 2.7x the old one (0.186312), so the difference is smaller than one run's scatter; seed 3 alone is worse (8.476669) than every seed in the old pass;
- two changes separate the runs, not one: RLVR went from a fifth gradient stage to an unrun opt-in selector, and SHIELD02's bounded residual policy is enabled by default here (gain about -11.1, ~461k-591k clamped gate decisions per seed) but absent from the old run. The delta cannot be attributed to the DAgger ordering;
- `ab_win_claimed: false` in `summary.json`, and the export manifest states artifacts are not an A/B result.

If the intent is to isolate the stage-order effect alone, rerun under a fresh output directory with `--no-residual-policy`. That run was not performed; it is not the current default.

## Exports

`ai/out/final_v3_postpump_dagger/exports/teacher_v2_s{1,2,3}.pt` are copied verbatim from the distill stage with sha256, config hash, source and RMSE in `exports/manifest.json`. The repo's `.gitignore` excludes `ai/out/**/*.pt`, so the binaries exist on disk but only the manifests, metrics, logs and summary are committed — the same policy as TRAIN-FINAL.

## Verification

- `ruff check scripts/train_final_v3.py`: clean; `ruff format --check scripts/train_final_v3.py`: clean.
- `ruff check src tests` (from `ai/`): clean.
- Full pytest: **373 passed in 104.18s** (the count rose from 350 because the concurrent SHIELD02 and RL-shield commits added tests).
- Whole-tree `ruff format --check src tests` is **not** clean: 38 pre-existing files outside this task's scope would be reformatted. They were not touched; no whole-tree format claim is made.
- Startup guards: frozen-data sha256, trainer stage order, per-seed `config_hash`, empty reward provenance, finite confidence predictions.

## Honest limits

- 100 epochs with a 1451-window cap is the same budget as the old run, not an uncapped full-window pass; the ~8 m/s^2 RMSE remains close to the target's own RMS, so the student has not learned the command to useful accuracy.
- The two-run comparison is descriptive only; no quality or closed-loop flight claim follows from it.
