# BUILD_LOG — ai / Поток-М

Date: 2026-11-22. CPU-only work; no GPU jobs, no pushes.

## Commands and results

| command | result |
|---|---|
| `uv sync` | Resolved 20 packages; installed `torch==2.15.0+cpu` from `download.pytorch.org/whl/cpu`. No CUDA wheels. |
| `uv run pytest -q` | `26 passed in 2.96s` initially; `37 passed in 4.89s` after the student work |
| `uv run ruff check src tests` | `All checks passed!` |
| `uv run python benchmark.py --stage train --steps 10` | `params=87875 ms_per_step=68.02 frames_per_s=5881` (width 128, rank 64, batch 8, seq 50) |
| `uv run python benchmark.py --stage forward --steps 10` | `params=87875 ms_per_step=31.80 frames_per_s=12579` |
| `PYTHONPATH=src uv run python -m potok_ai.train --help` | CLI exposes both stages, `--opt`, `--ns-steps`, `--limit` |

Tests cover: forward shape/bounds, 5-block layout and dilations, TSCT param count,
seed determinism, causality, ternary codes and STE gradient flow, ternary anneal,
FWHT round-trip and 4-bit gradient, EMA teacher update, mask spans, JEPA loss and
gradients, KoLeo, NS spectrum compression, Muon+ loss decrease, AdamW fallback,
ColRow, episode binary round-trip and raw fallback, export/load and tamper
rejection, and one tiny CPU step in each stage.

## Decisions

- **torch 2.15.0** is the current PyPI latest (verified via PyPI JSON); "3.15" does
  not exist. Plain `uv sync` resolves CPU wheels through an explicit index. The
  `cu134` index is declared in `pyproject.toml`; for GPU, point the `torch` source
  at `pytorch-cu134`. No `--extra gpu`: uv requires an explicitly selected extra
  for conflicting PyTorch sources, which would break the required plain `uv sync`.
  CUDA wheels were never installed.
- **Muon+ is the default** (`--opt muon`): momentum `M = μM + (1−μ)G`, Jordan
  Newton–Schulz quintic `(3.4445, −4.775, 2.0315)`, ColRow = `norm_row(norm_col(X))`
  (Muon+ Eq. 7, `eps` inside the sqrt), scale `max(1, m/n)^0.5`, 5 NS steps.
  Coefficients come from `dormouse/docs/papers/muon-plus.md` (row D6: exact match
  to `NS_COEFFS` from Jordan's `muon.py`); the scale is the documented D3 deviation
  from the paper's `√(m/n)`. The `max(1, ...)` form and 1-D AdamW routing are the
  paper's own hybrid (`Muon+` §3.1), not a fallback optimizer. Plain AdamW exists
  only as `--opt adamw` for comparison. The PolarExpress cubic
  `(1.875, −1.25, 0.375)` was also found in that repo (burn-spectral TSCT
  retraction, row D5) and is exposed as `POLAR_EXPRESS_COEFFS` / `MuonPlus(coeffs=…)`;
  it is not the default because rows D5/D6 identify it as a different polynomial
  than Muon+'s.
- **Fast-JEPA (arXiv:2611.25927)**: one linear probe maps masked student latents to
  the teacher latents of the *complete* window shifted by 3 frames (≈0.2–0.4 s at
  50 Hz) in a single forward. No recursive rollout, no candidate search, no
  planning code. KoLeo on the final latent keeps representations from collapsing.
- **Fusion / flops**: each TSCT pointwise map folds `(U·s) @ Vᵀ` into one effective
  weight and a single `F.linear`, so the block launches one GEMM instead of two.
  The depthwise causal conv stays separate: PyTorch has no asymmetric causal
  padding, and folding it into the pointwise map would multiply parameters by the
  kernel width — that trade is not worth it at width 128 (add when profiling says
  conv launches dominate).
- **Hadamard activations** use a fast Walsh–Hadamard transform: `d·log₂d` butterfly
  work with no stored matrix (better than "precompute the matrix"), self-inverse,
  4-bit symmetric levels −7..7 with a per-row scale and straight-through gradient.
- **CPU smoke path**: the tiny dataset is materialised once and the inner loop only
  slices tensor views; gradients are zeroed in place (`set_to_none=False`) so
  buffers are reused; one CPU thread; `--limit` caps windows.
- **Export** (`export.py`): version header, layer descriptors, ternary int8 codes,
  scales, fp32 tensors, SHA256 trailer; the loader verifies the checksum and
  refuses tampered files. The format is documented in the module docstring.
- **`filter/`** is unfinished parallel work: left untouched and excluded from
  ruff via `pyproject.toml`.

## Canonical export format (v2) and cross-check fixture

- `ai/FORMAT.md` is the authority for the exact byte layout: header carries the
  full architecture (dilations, kernel, bounds), layer records carry ternary
  codes + scales or fp32 tensors, SHA256 covers all bytes except the 32-byte
  digest trailer. Version was bumped to 2 so a reader cannot mistake the old
  header for the self-describing one.
- Two export bugs were found while building the fixture: the stored U/V were
  `alpha*codes` cast to int8 instead of the codes themselves, and a layer loop
  shadowed the header's `in_features`/`out_features`. Both are fixed; a new test
  asserts the reloaded model reproduces the original forward bit-for-bit.
- Fixtures: `tests/fixtures/tiny_model.bin` (seed 1234, width 16, rank 8,
  untrained) and `tests/fixtures/tiny_expected.npy` for the fixed 50x8 window.
  `tests/fixtures/generate.py` regenerates them; `tests/test_fixture.py` reloads
  and asserts exact equality.
- **The `edge/` reader should read `ai/tests/fixtures/tiny_model.bin` and compare
  its three float32 outputs to `ai/tests/fixtures/tiny_expected.npy`** before
  attempting any larger model. No files outside `ai/` were modified.

## Simulator validation reader

- `data.py` now reads `POTOKSIM` `validation.bin` exactly per `sim/README.md`:
  80 bytes of header fields (magic, version, seed, counts, frame timing,
  reserved), then per-episode `u64` index + 9 metadata f32 + `frames × features`
  f32 + `frames × 4` target f32. `Episode` carries index and metadata; targets
  keep all four columns, and `WindowDataset` uses the first two (g_roll, g_pitch)
  for distillation.
- The README's "76 bytes" disagrees with its own field table (48 + 32 = 80) and
  with the Rust writer (80 bytes; its `header_size()` says 84). The Python reader
  follows the documented fields and real bytes; the disagreement is recorded in
  `ai/FORMAT.md` and needs an owner decision in `sim/` — no files there were
  touched.
- The old `POTOK1\0\0` and bare-float32 readers remain as fallbacks for synthetic
  fixtures. `tests/test_sim_validation.py` packs a handcrafted validation file
  and checks header, shapes, index, metadata, and target values; all existing
  tests stay green.

## Stateful student (SPEC_STUDENT.md v1)

- `src/potok_ai/student.py` — `StatefulTCN` (defaults: 2 blocks, width 64,
  rank 16, 10-value frame = 8 sensors + phase + flags), one frame per `step`,
  persistent `h` per block. Each block is `tanh(W_h h + W_x x)` followed by a
  ternary `TSCTLinear` reused from `model.py`. A tick applies the shared blocks
  1..3 times, mixing loop outputs with `0.8^i` and a learned softmax allocation
  (`alloc_logits`, AttnRes-style). Head: `g_max·tanh(linear(h))` (two bounded
  corrections, `g_max = tanh(raw)` with `raw` exported) and
  `sigmoid(linear_conf(h))`. `reset_state()`, `step(frame)`, `warmup(window)`;
  all parameters are seeded from a local `torch.Generator`, so global RNG state
  cannot affect the model. The incoming state is cloned before use: writing it
  back in place otherwise invalidates tensors saved by autograd (caught by the
  tiny distillation test).
- `src/potok_ai/distill_student.py` — CLI with `--teacher-checkpoint --data
  --out --epochs --seed --device cpu` (plus `--window --loops --lr
  --conf-weight --conf-temp --limit`). Loads the teacher from a torch checkpoint
  or a v2 export, scores every 50-frame window, and trains the student at the
  window's last frame: MSE to `g_teacher` plus MSE of confidence to
  `exp(-|g_teacher - g_student| / conf_temp)`. State resets at each episode and
  on a phase-code change (column 8 if present); invalid flags skip the update.
  Writes a torch checkpoint to `--out` and the binary export to `out.with_suffix(".bin")`.
- `export.py` — `export_student()` / `load_student_export()` and
  `ExportedStudent.to_model()`. Header: magic, version 2, `model_kind = 2`,
  then in_features/width/out_features/rank/n_blocks/n_loops/decay/g_max/flags/
  n_layers, the same layer records and SHA256 trailer as the teacher format.
  The teacher's existing header is untouched (kind 1 is still implied by the
  absence of a `model_kind` field — changing it would break the fixture and the
  `edge/` reader); student files are rejected by the teacher loader's kind
  check and vice versa. Exported U/V are effective ternary codes × alpha, so
  `to_model()` builds with `ternary=False` for a bit-comparable forward.
- New `tests/test_student.py`: state reset, seed determinism under two different
  global seeds, 1500-step bounded drift (|h| ≤ 1, finite bounded outputs),
  export/load round-trip plus tamper and wrong-kind rejection, one tiny CPU
  distillation step (checkpoint + export written), and phase-change reset.
- Verification: `uv run pytest -q` → `37 passed`; `uv run ruff check src tests`
  → clean. `PYTHONPATH=src uv run python -m potok_ai.distill_student --help`
  exposes the CLI. CPU only; no GPU jobs, no pushes, no files outside `ai/`.

## Skipped

- No GPU wheels, no A/B runs, no training beyond the tiny CPU smoke steps.
- The simulator's episode writer does not exist yet, so `data.py` defines a magic
  header format plus a tolerant raw-float32 fallback. Align the header with
  `sim/README.md` when that lands.
