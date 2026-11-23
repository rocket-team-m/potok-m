# MUON04 — wire the typed K3 config into both training CLIs

**Status:** done. Local commit only, no push. No subagents. Repo `/home/sehaxe/potok-m/ai`.
**Date:** 2026-11-23. Follow-up to MUON03 (`5acddab`), which added `K3Config` but
left it unreachable from the CLIs.

## Scope — exactly four files

- `ai/src/potok_ai/train.py` (modified)
- `ai/src/potok_ai/train_teacher_v2.py` (modified)
- `ai/tests/test_optimizer_cli.py` (new, 10 gates, SPDX GPL-3.0-or-later)
- this report

No optimizer math, model, preset, workflow or artifact was touched. `optim.py`
and `docs/OPTIMIZER_NOTES.md` are unchanged; uncommitted files from parallel
tasks were left alone.

## What changed

1. **Both parsers get the same typed flags**: `--k3-per-head N` (int, default
   `0`), `--k3-qk-clip` (store_true, default false), `--k3-qk-clip-tau X`
   (float, default `100.0`, K2's number). Each CLI has one small `k3_config`
   helper that returns an immutable `K3Config`; `K3Config.__post_init__` remains
   the single validator (`τ<=0`, clip without heads).
2. **`train.py`**: `_optimizer` now accepts **named** parameters and always
   passes the typed config to `MuonPlus`. Both call sites use
   `named_parameters()` (predictor names prefixed `predictor.`). The AdamW arm
   strips names before construction, so its parameter list is unchanged.
   Default `muon` + default `K3Config()` reproduces the historical step exactly
   (no name-based arm fires, no QK pair is sought).
3. **`train_teacher_v2.py`**: this CLI had only AdamW, so the flags had nowhere
   to go. Added `--opt {adamw,muon}` with **default `adamw`**, a `_build_optimizer`
   helper used by all five stages, and named parameters everywhere (including
   `jepa.`/`privileged_head.` prefixes in the privileged stage). Only
   `--opt muon` constructs `MuonPlus` and supplies the typed config. The new
   knobs are part of `TeacherV2Config` and therefore of `config_hash`; all stage
   tests keep their defaults and their determinism.
4. **Defaults are off in both CLIs**, and no coefficient, step count, routing,
   LR or clip rule was altered.

## Caveats, stated rather than hidden

- On both shipped models the flags change **optimizer configuration**, not
  training math: `PotokModel` has `dw`/`pw.U|s|V` names and `LoopWM` has a fused
  `block.qkv.weight`; neither has `q_proj`/`k_proj`/`v_proj`. Per-head NS is
  therefore inert there (name detection finds nothing), by the same design that
  made the old CLI inert. QK-Clip on these models fails **loudly** at
  construction with `qk_clip needs named q_proj/k_proj parameters`. The gates
  use a tiny dense q/k/v module to prove that names now reach `MuonPlus` and
  that a q/k pair is found (`len(_names)==3`, `len(_qk_pairs)==1`). Exposing the
  arms on the fused `qkv` layout would require optimizer changes and is out of
  scope.
- In teacher-v2 the K3 flags need `--opt muon`; otherwise the historical AdamW
  stages run unchanged. This is intentional: the default preserves every gate.
- The K3 clip still uses MUON03's weight-space S_max proxy; no measured logit
  maxima are available to an optimizer step. Unchanged from MUON03.

## Verification (commands run in `ai/` with `uv run`)

| Command | Result |
|---|---|
| `python -m pytest tests/test_optimizer_cli.py tests/test_optim.py tests/test_optim_k3.py tests/test_train.py tests/test_train_teacher_v2.py -q` | 42 passed |
| `python -m pytest -q --deselect tests/test_smart.py::test_tiny_cpu_training_step --deselect tests/test_student.py::test_tiny_cpu_distillation_step` | 345 passed, 2 deselected |
| `python -m ruff check src/potok_ai/train.py src/potok_ai/train_teacher_v2.py tests/test_optimizer_cli.py` | All checks passed |
| `python -m ruff format --check` (same three files) | 3 files already formatted |

**Full-suite caveat (honest failure report):** plain `python -m pytest -q` was
**2 failed, 339 passed** at the time it ran. Both failures are
`torch.load(..., weights_only=True)` rejections of
`potok_ai.student.CommandEnvelope` in `tests/test_smart.py::test_tiny_cpu_training_step`
and `tests/test_student.py::test_tiny_cpu_distillation_step`. They reproduce in
isolation, involve neither file I edited, and were introduced by a parallel
agent's uncommitted change to `ai/src/potok_ai/student.py` (timestamp 23:32:56,
after MUON03's commit at 23:29:43). I did not touch that file; those two gates
are not green under this worktree and are not claimed as such.

The 10 new gates pin: parser defaults off in both CLIs; flags produce the
correct typed `K3Config`; named params and QK pairs reach `MuonPlus` in both
CLIs; the AdamW paths still receive plain tensors; invalid clip/τ fail loudly;
a tiny `--k3-per-head` train run and a `--opt muon` teacher stage both finish
with finite metrics.

All code is original. No push was performed.
