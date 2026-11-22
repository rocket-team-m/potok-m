# Task B03 — AI v3 target integration

## Scope

Touched only the assigned paths:

- `ai/src/potok_ai/data_v3.py` (new)
- `ai/tests/test_data_v3.py` (new)

No edits to `data.py`, `engine.py`, or any sibling agent's files.

## Implementation

Built the trainer-facing target contract on top of the canonical
`potok_ai.data.read_validation` reader (reuse, no new parser):

- **v3**: `g_accel_y` / `g_accel_x` are read from target columns 4–5 by the
  existing reader. The authoritative fin command is
  `[g_accel_y, -g_accel_x]` (m/s², trainer `[roll, pitch]` seam order), taken
  from `Episode.command_target()` so the transform has one source of truth.
- **Diagnostic**: `g_theta` (`g_roll`, `g_pitch`) stays available separately in
  all versions and is never promoted to a command.
- **v1/v2**: no acceleration column. `authoritative_command()` returns `None`;
  `authoritative_target()` and `EpisodeTargets.training_target` raise a loud
  `ValueError` naming the version and the upgrade path — no silent fallback
  from the angle-domain diagnostic.
- `read_targets(path)` returns per-episode `EpisodeTargets` records carrying
  `version`, `g_theta`, `command`, an `authoritative` flag and the loud
  `training_target` accessor. Version/target-width mismatches remain loud via
  the underlying reader.

## Tests

`tests/test_data_v3.py` packs synthetic v1/v2/v3 POTOKSIM bytes with `struct`
and covers: v3 command vs diagnostic values and signs; v1/v2 diagnostic-only
behaviour with loud failure; independence of command from `g_theta` columns;
truth-extended v3 (`meta_count=39`); and refusal of v3-with-4-targets /
v2-with-6-targets.

## Verification

| Check | Result |
|---|---|
| `uv run pytest -q tests/test_data_v3.py` | 7 passed |
| `uv run pytest -q` | 166 passed |
| `uv run ruff check src tests` | All checks passed |
| `git diff --check` on B03 paths | clean (worktree-wide check flags unrelated `ai/out/v3_final/` files owned by another lane) |

Committed locally on `main`; nothing pushed. Unrelated uncommitted files in
`ai/out/` and sibling agents' in-flight files were left untouched.
