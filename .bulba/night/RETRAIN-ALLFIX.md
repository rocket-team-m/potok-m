# RETRAIN-ALLFIX — final retrain after every AI root-cause fix

**Date:** 2026-11-24
**Repo:** `/home/sehaxe/potok-m` (branch `main`). Code/artifacts commit `02cc2bd`; this report is committed separately. Local only, nothing pushed.
**Mode:** no subagents. No trainer, test, physics, gate, weight or reward file was edited.

## Scope

- `scripts/train_final_v3.py` — modified and is now the all-fix driver.
- `ai/out/final_v3_allfix/**` — new: `summary.json`, `shield.json`,
  `exports/{manifest.json}`, `runs/s{1,2,3}/{metrics.json,conformal.json,run.log}`.
  Student `.pt` binaries exist on disk but are `.gitignore`d (`/ai/out/**/*.pt`),
  same policy as TRAIN-FINAL and RETRAIN-DAGGER.
- `ai/docs/TRAINING_RESULTS.md` — rewritten by the driver.
- this report.

No file outside those four paths was staged. The pre-existing uncommitted work in
`sim/`, `ai/out/v3_final/`, `ai/tests/` and elsewhere was left untouched.

## Why this run exists

AITRAINIDEAL scored the shipped teacher-v2 training system 3/10: the ±1 m/s²
student head could not represent the ±10 target (B1), the privileged stage ran
with `truth_sidecar = null` and `privileged_batches = 0` (B2), DAgger rolled a
fixed window over recorded frames with 22 zeroed truth slots (B2), and the RMSE
numbers were produced by that system. Four night fixes landed afterwards —
AIFIX1 (command bound), AIFIX2 (real v3 truth sidecar), AIFIX3 (on-policy
DAgger), SHIELD01/02 (online shield, bounded residual policy). None of them
included a retrain. This task runs all of them together, on the same frozen
post-FIXPUMP data and the same numeric recipe as RETRAIN-DAGGER, and reports
what actually happened without claiming a win.

## What the driver now enforces

`assert_allfix()` runs on every seed (train or REUSE) and fails loudly if any
fix regressed. Evidence is read from the trainer's own `metrics.json`, not from
a flag in the driver:

| required state | artifact evidence (all three seeds) |
|---|---|
| DAgger default | `stages_completed = ["sft", "privileged", "dagger", "distill"]`; `reward_provenance = {}` (RLVR not run) |
| AIFIX1 ±10 bound | `stages.distill.details.student_command_bound = 10.0`, `legacy_student_bound = false` |
| AIFIX2 real truth | `stages.privileged.details.privileged_batches = 2300`, `privileged_source = "privileged_teacher"`, `truth_sidecar` = the train-split v3 file, `eval_label_blind = true` |
| AIFIX3 on-policy | `stages.dagger.details.rollout_source = "on_policy_student:gdn2_student"`, `privileged_source = "v3_truth_sidecar"`, `privileged_fields_present = true`, `student_induced_rows = 118960`, non-empty `rollout_sha256`, labels from `truth_sidecar_slots22..25` |
| SHIELD02 residual | `residual_policy.enabled = true`, `source = "residual_policy"`, `stages.distill.details.residual_policy_target = true` |
| SHIELD01 shield | `shield.json`: counted `OnlineShield.verify` on five cases (1 accept + 4 veto reasons), fallback reaches the command on every veto |

The driver passes `--truth-sidecar <train-split v3>` and does **not** pass
`--no-residual-policy` or `--legacy-student-bound`. A config-hash self-check
confirms the driver's `TeacherV2Config` and the exact CLI it emits hash
identically (`f5b5106b378aac4b` for seed 1).

The shield is **not** invoked by the staged teacher-v2 trainer: it lives in the
engine's `LRSAdapter`. `shield.json` is a counted verification of the shipped
module, not a fabricated training intervention count. The report and results
page say so explicitly.

## Data (sha256 verified by the driver before any step)

| artifact | sha256 |
|---|---|
| train `validation.bin` (80 episodes, scenarios 0-15) | `34b35e0ff67f7eeecbd895b2efde8bebcfa159c1a3e78379da6750338b6930b6` |
| held-out `validation.bin` (scenarios 16-19) | `fdc1d1400b1c5f111e21f79dead37afa1dd9f058fe457edb78037dc6ebe89807` |
| frozen `sim.toml` | `9c34e71f9d10bcf081c31f15f5332f7bb2fac04dddb28009a27cd30987ab5e91` |
| train-split POTOKTRJ v3 `validation.trajectory` | `7f25bb69124638ea7542be99f43330fda1c5248513f1971f6bfd125f5f759cb8` |

The sidecar is 80 episodes × 1500 ticks × 26 slots (v3, stride 26), its episode
indices are 0..79 and match the train `validation.bin` one-for-one, and slots
0..21 are non-zero on every episode. It is the **train split only**: the
privileged builder fits its normalizer on every record handed to it, so a
held-out sidecar would leak scaling truth (PRIVILEGED_TEACHER.md). No data was
regenerated or substituted.

## Recipe (unchanged from RETRAIN-DAGGER)

100 epochs, CPU, seeds 1/2/3, batch 64, window 12, `--limit 1451` (caps WINDOWS
per batch builder, not episodes), lr 2e-3, student-lr 1e-3, width 16, heads 2,
max-iters 2, jepa-weight 0.05, entropy-weight 1.0, dagger-rounds 1,
n-inner-held-out 2, calib-tol 0.15, adamw. The truth sidecar is the one new
input; the three binary fixes are defaults, not new knobs. Held-out scenarios
16-19 never enter a training batch. Three seeds were launched in parallel as
independent single-threaded CPU processes, then reloaded and re-enriched by the
driver (REUSE path, hash/stage/fix checked).

## Results

Held-out RMSE over three seeds: **7.855551 +/- 2.604600 m/s²** (sample std).

| seed | train RMSE | held-out RMSE | ratio | disjoint ECE | conformal coverage |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.658077 | 5.148904 | 0.9100 | 0.0117 | 0.9353 |
| 2 | 9.595021 | 10.344432 | 1.0781 | 0.0079 | 0.9545 |
| 3 | 8.446756 | 8.073319 | 0.9558 | 0.0108 | 0.9642 |

Config hashes: `f5b5106b378aac4b` / `7b5ad1b86cfcb3f3` / `717d2a5ae810d807`
(out path and seed are hashed; every recipe knob is shared). Teacher gates
`rmse_ratio`, `ece_source`, `ece` all PASS. Closed-loop gates (`seed_spread`,
`closed_loop_improvement`, `oracle_positive_control`) are not evaluated.

## Comparison to RETRAIN-DAGGER — no win claim

| run | held-out RMSE, mean +/- std (m/s²) |
|---|---:|
| RETRAIN-DAGGER (7.903 ± 0.499) | 7.902916 +/- 0.498808 |
| this run (all fixes) | 7.855551 +/- 2.604600 |

Difference of means: **-0.047364 m/s²**, i.e. **within one baseline sample
std** (0.498808). It is not evidence of improvement:

- the seed spread grew to 2.604600 from 0.498808 — **5.2x worse** reproducibility
  across identical-config seeds; seed 2 is 10.344432 (worse than every seed in
  the baseline), seed 1 is 5.148904. The mean is the average of a bimodal-looking
  triplet, not a stable effect.
- three changes separate the runs at once: the ±10 head replaces the ±1 head,
  the sidecar replaces zeroed privilege, and the rollout is now student-induced.
  No arm isolates any of them, so even a large delta could not be attributed.
- `ab_win_claimed: false` in `summary.json`; the export manifest states the
  artifacts are not an A/B result. No closed-loop harness, no flight-truth
  anchor, no positive control, no unweakened gate evaluation.

The command-bound mechanism demonstrably landed and removed a known symptom:
residual gain is -4.86 / -6.23 / -5.85 with **0 clamped rows**, against roughly
-11 with 461k-591k clamped rows per seed in RETRAIN-DAGGER. That did not become
a lower held-out RMSE. Mechanism verified, quality win not shown.

## Surrogate dynamics are the only binding — stated loudly

This is the headline result, not a footnote. AIFIX3's `world_step` advances
only three onboard sensor features through the unchanged
`physics_baseline`/`final_command` plus a named noise stream. It is a surrogate
for a Rust simulator step on (state, command), and no Python binding exists.
The POTOKTRJ v3 sidecar is emitted by that same simulator, so the verifier and
the generator share a source — it is not an independent flight-truth anchor.
The DAgger expert label is still the reference optimizer's recorded answer from
sidecar slots 22..25, and the privileged stage still supervises an
episode-pooled latent with MSE rather than a decision-making teacher measured
against its removal.

Therefore the fixes make the wiring true and the provenance checkable, but they
add no new information about closed-loop flight dynamics. The ~7.9 m/s² held-out
RMSE sits at the target's own RMS (~7.7) and the 5.2x seed-spread blowup is the
shape a surrogate-driven on-policy loop produces. Honest artifacts were
produced. The closed-loop 3-seed A/B with unweakened `teacher_gates` thresholds
and an independent truth anchor remains the gate that has not been run.

## Honest limits

- The online shield is verified and counted but not exercised by the training
  loop; its Mach/phase arms are inert on offline v3 data by explicit declaration
  (`OFFLINE_MACH = 0.0`, `OFFLINE_PHASE = 1`). No CBF/QP certificate or QP timing.
- Raw `ece` is still 0.35-0.36 while `ece_held_out` is 0.008-0.012; the ECE is a
  well-behaved confidence head on a disjoint inner split, not a small command
  error, and the map is not applied at the edge.
- `--limit 1451` is still a budget confound; SFT/privileged/distill ran 2,300
  steps each and DAgger 162,800 steps (on-policy rollout yields 119,040 rows,
  not the old 19,920). Same cap as the baseline, so comparable, but not an
  uncapped pass.
- 100 epochs was originally chosen on held-out RMSE; this run inherits that
  protocol and does not re-validate it.
- No physics constant, gate, threshold, weight or reward formula was changed;
  no push.

## Verification

- `ai/.venv/bin/ruff check scripts/train_final_v3.py` — clean;
  `ruff format --check` — clean.
- `ai/.venv/bin/ruff check src tests` (from `ai/`) — clean.
- `.venv/bin/python -m pytest -q` (from `ai/`) — **396 passed in 139.87 s**,
  exit 0.
- Config-hash self-check: driver config == exact emitted CLI config.
- Driver `verify_inputs()`: four sha256 pins, stage-order pin.
- Driver `assert_allfix()`: 15 per-seed assertions on the fixes above, applied
  to fresh and REUSEd artifacts.
- Driver `shield_smoke()`: 5 counted shield decisions with fallback equality.
- A 1-epoch trainer smoke with the real sidecar was run before the full job
  (privileged_batches 23, on-policy rollout, bound 10.0); no recipe change
  followed from it.
