# Task B05 — AI augmentation integration

**Status:** implemented, tested, committed locally (no push).
**Repo:** `/home/sehaxe/potok-m` (single git root; code under `ai/`).
**Assigned paths only:** `ai/src/potok_ai/augment_integration.py`, `ai/tests/test_augment_integration.py`, this report.

## What was built

A trainer-facing bridge over the existing pure `augment.py`:

- `window_seed(base_seed, index)` — stable 63-bit blake2b seed from `(base_seed, index)`;
  no global RNG, no `hash()` (which is process-randomised), negative inputs refused LOUDLY.
  Pinned literals in the test catch any future change to the derivation.
- `augment_window(window, base_seed, index, config)` — pure one-window helper.
- `AugmentedDataset(base, base_seed, config)` — wraps any window Dataset; transforms only
  element 0 of each item (the window), passing targets, conditioning and other fields
  through byte-identically. Restores the original tensor dtype/device; the underlying
  dataset is never mutated.
- `split_window_datasets(episodes, base_seed, config, ...)` — splits on the authoritative
  `Episode.held_out` flag, returns `(train_augmented, held_out_plain)`. Held-out episodes
  go directly into a plain `SmartWindowDataset`, so there is no code path that could
  augment evaluation data. A missing side, or a split with zero windows, is a loud error.

Only the input window is transformed; labels are never augmented, and the default
`AugmentConfig` is a verified no-op.

## Verification

- `pytest -q` (full AI suite): **145 passed**.
- New module: 7 tests covering pinned per-window seeds, neighbour separation, purity,
  determinism across re-created wrappers, per-window seed variation, default no-op,
  no base-dataset mutation, item structure preservation, held-out bitwise equality,
  no global NumPy/Torch RNG advance, and loud refusals.
- `ruff check` + `ruff format --check` on both assigned files: clean.
- `git diff --check` on assigned files: no whitespace errors.

Repo-wide `ruff check` is **not** clean for reasons outside this task: pre-existing issues
in `ai/out/sweep/*` and `ai/out/v3_final/*` experiment scripts, plus an unsorted import
block in another agent's in-progress `ai/tests/test_conf_integration.py`. Those files were
left untouched. The uncommitted `ai/out/v3_final/*` work also predates this task and was
not staged.

## Skipped / when to add

- No wiring into `engine.py` or a CLI flag: those paths are owned by other tasks and the
  task scope is the bridge module plus tests. Add a `--augment` flag in a separate task
  once the orchestrator integrates the Wave B bridges.
- No epoch/episode-aware seed: seeds are keyed to the sample index of the wrapped dataset;
  build it once and reuse it (documented in the module docstring).
