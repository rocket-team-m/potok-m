# Night task POST04 — DAgger corrective dataset builder

**Status:** done. Local commit only, nothing pushed. No subagents.
**Source brief:** WORLD02 §2.6/§3 (DAgger adopt), WORLD03 §2.2/§2c/§3 (corrective-label rounds, seed protocol, no-eval-contamination).

## Scope

Created exactly two files in `ai/` plus this report:

- `ai/src/potok_ai/dagger.py` — `SPDX-License-Identifier: GPL-3.0-or-later`, pure data layer.
- `ai/tests/test_dagger.py` — same header, 13 tests.

No trainer, model, config, generator, workflow, existing source or existing test was edited. No file under `out/`, no simulator and no GPU was read. Other agents' uncommitted files were left untouched and unstaged.

## Requirement → implementation

| Requirement | Where |
|---|---|
| Roll out student-induced states | `RolloutFn(episode, rng)` is called once per (round, rollout, episode); only the returned `RolloutState` sequence is aggregated. The builder never substitutes teacher-distribution states. |
| Query privileged teacher for corrections | `teacher(state.privileged)` is the **sole** writer of `correction`. The privileged snapshot is consumed by the labeller and never stored in a sample, so no privilege leaks into student training. |
| Aggregate with held-out disjoint | `DaggerDataset` accumulates all rounds; `split_scenarios` yields disjoint `(fit, held_out)` scenario sets; `fit_samples()` / `held_out_samples()` are disjoint by construction and cover the aggregate exactly. |
| Deterministic seed | Per-(round, rollout, episode) `random.Random(f"{seed}:dagger:{i}:rollout:{j}:episode:{idx}")` and `random.Random(f"{seed}:dagger:split")`. No global RNG, no reseeding, no wall-clock/PID input. `fingerprint()` is a SHA-256 over sample ids + arrays. |
| Expert-label provenance | Every sample carries `Provenance(sample_id, seed, round, rollout, step, episode, scenario_id, split, expert, label_source)` with `expert="privileged_teacher"`, `label_source="expert"`. `sample_id` is unique and reconstructible from the fields. |
| No eval contamination | Eval cells (scenario ids 16..19, or `held_out` flag) are refused **before** rollout or teacher runs — asserted by a call counter in the tests. `split_scenarios` also refuses them, so a scoring cell can never become a diagnostic holdout. |
| Loud failures | Non-finite/empty teacher labels, missing scenario ids, invalid round/rollout/holdout counts, eval contamination and empty input all raise `ValueError` with the cause. No silent fallback. |

Two "held out" notions are kept separate on purpose and documented in the module docstring: the final **eval** cells 16..19 are forbidden everywhere here; the **inner** held-out split is a disjoint subset of training cells 0..15 used only to measure loss on student-induced states (DART's covariate-shift diagnostic), never to retrain.

## Design notes

- The student rollout and the privileged labeller are injected callables. That is the seam a no-trainer-edits data layer needs: WORLD03's round loop, privileged snapshot and authoritative target are all representable without importing or modifying the training code.
- `RolloutState` carries `(observation, privileged, authoritative)`; `observation` is the only student-visible field persisted. This is the direct guard against the privileged-input mismatch WORLD02 §2.6 calls out.
- Frozen dataclasses with `eq=False` for array-bearing rows; determinism is compared through `fingerprint()`, never through numpy `==`.
- The teacher's non-finite label check lives at the callable boundary (external input), not scattered through consumers.

## Verification

- `uv run python -m pytest tests/test_dagger.py -v` → **13 passed** (1.74 s).
- `uv run python -m pytest -q` (whole `ai` suite) → **278 passed in 178.54 s**.
- `uv run ruff check src/potok_ai/dagger.py tests/test_dagger.py` → clean.
- `uv run ruff format --check src/potok_ai/dagger.py tests/test_dagger.py` → clean.

## Deliberately out of scope (honest limits)

- **β mixing of expert/student decisions** (`β_i = I(i=1)` or `0.5^{i−1}`), NaN trajectory abandonment with COUNTED logging, and DART noise injection are rollout-policy concerns. They belong inside the injected `rollout` callable, not in an aggregate builder; nothing here prevents them, and no trainer was touched.
- **No from-scratch FTL retrain, no early-stop diagnostic and no export**: this task builds the dataset, it does not train or round-trip a student. `fit_samples()` is the hand-off point to the existing distillation path.
- **Fixtures are synthetic** (`make_smart_episode`); no real simulator episode or teacher checkpoint was scored, so no quality or flight claim is made. The gates here are structural (determinism, provenance, disjointness, contamination), not statistical.
- The fixed-6-round schedule in WORLD03 cannot make `β̄_N → 0`; if these rounds are later run, that finite-budget deviation must be reported rather than hidden. It is a property of the calling schedule, not of this module.
