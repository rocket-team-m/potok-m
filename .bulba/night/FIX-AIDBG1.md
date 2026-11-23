# FIX-AIDBG1 — command-target contract: v1/v2 refused loudly

**Date:** 2026-11-23 · **Repo:** `/home/sehaxe/potok-m` (`ai/`)
**Finding:** `.bulba/night/AI-DBG01.md` §1 (HIGH) · **Scope:** `ai/src/potok_ai/`
`data.py`, `data_v3.py`, `engine.py`, `train_smart.py` + tests. No subagents.

## Root cause

`Episode.command_target()` silently returned the diagnostic `g_theta` (rad)
whenever a v1/v2 episode had no `g_accel`. Every trainer routed through that
method (`SmartWindowDataset`, `engine._smart_frames`,
`train_smart.distil_student`), so a v1/v2 validation file trained the m/s²
command head on radians and reported held-out RMSE in radians. The documented
contract (`data_v3.py`, `README.md`) says v1/v2 must be refused loudly with no
substitution; nothing enforced it. v3 data was never affected.

## Fix

- `Episode.command_target(*, allow_diagnostic_target=False)` now **raises** a
  `ValueError` for a v1/v2 episode (names "v1/v2", the v3 requirement and the
  escape); with `g_accel` it returns the authoritative `[g_accel_y, -g_accel_x]`.
  The flag opts into angle-domain diagnostic training only. `g_theta()` stays
  the diagnostic accessor and the v1/v2 readers (`read_validation`,
  `authoritative_command`, `diagnostic_target`, `read_targets`) are unchanged.
- `data_v3.training_target(episode, *, allow_diagnostic_target=False)` is the
  single trainer-facing resolver, with `authoritative_target` delegating to it.
- All training target call sites now route through it with the flag:
  `SmartWindowDataset`, `engine._smart_frames`, `train_smart.train_teacher`,
  `train_smart.evaluate` and the shared `train_smart.distil_student`.
- `--allow-diagnostic-target` added to the `engine` and `train_smart` CLIs
  (`TrainConfig.allow_diagnostic_target`, forwarded via `legacy_args`).
  Default is false everywhere: command-target training requires v3 data.
- Fixtures: `make_smart_episode` emits v3 targets (`g_accel` chosen so the
  command equals the former g_theta — training numerics unchanged); the
  `test_loopwm` / `test_smart` packs were upgraded to six v3 target columns;
  the archived v2 sweep fixture is promoted in memory inside that test.

## Verification

- `.venv/bin/ruff check src tests` → **All checks passed**.
- `.venv/bin/pytest -q` → **253 passed** (includes 4 new contract tests in
  `test_data_v3.py`: resolver loud/escape, dataset refusal+escape, engine
  frames refusal+escape, `distil_student` refusal; the legacy assertion in
  `test_sim_validation.py` now checks the loud default and the explicit escape).
- The 100-epoch v3 hard gate (`test_engine_gate_hard.py`: RMSE ratio, ECE,
  determinism, held-out isolation) remains green.

## Limits / not done

- `train_loopwm.py` is out of scope: its own `train_teacher` has no flag and
  keeps the loud default, so this CLI exposes only refusal for v1/v2 data
  (the shared `distil_student` does honour the flag). No silent fallback
  remains on any path.
- The other AI-DBG01 items (NaN handling in ECE/calibration, in-sample ECE,
  dual package copies, checkpoint reuse, ternary export, JEPA RNG, doc drift)
  are untouched.
- Committed locally; no push.
