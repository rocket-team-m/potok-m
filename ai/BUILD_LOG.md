# BUILD_LOG — ai / Поток-М

Date: 2026-11-22. CPU-only work; no GPU jobs, no pushes.

## Commands and results

| command | result |
|---|---|
| `uv sync` | Resolved 20 packages; installed `torch==2.15.0+cpu` from `download.pytorch.org/whl/cpu`. No CUDA wheels. |
| `uv run pytest -q` | `26 passed in 2.96s` initially; `37 passed in 4.89s` after the student work; `44 passed in 5.45s` after the smart work; `51 passed in 6.24s` after the LoopWM work; `61 passed in 10.32s` after the engine work |
| `uv run ruff check src tests` | `All checks passed!` (after the smart, LoopWM and engine work) |
| `PYTHONPATH=src uv run python -m potok_ai.train_smart --data <v2 file> --out /tmp/opencode/smart.pt --epochs 1 --window 12 --batch 2 --limit 4 --eval-limit 4 --width 16 --rank 8` | prints three scenario RMSE rows, `ratio=0.7034 limit=1.25 generalises=yes`, teacher + export paths; CPU only |
| `uv run python benchmark.py --stage train --steps 10` | `params=87875 ms_per_step=68.02 frames_per_s=5881` (width 128, rank 64, batch 8, seq 50) |
| `uv run python benchmark.py --stage forward --steps 10` | `params=87875 ms_per_step=31.80 frames_per_s=12579` |
| `PYTHONPATH=src uv run python -m potok_ai.train --help` | CLI exposes both stages, `--opt`, `--ns-steps`, `--limit` |
| `PYTHONPATH=src uv run python -m potok_ai.engine --help` | CLI exposes `--model {lrs,loopwm-lite,loopwm}`, `--stage {train,distill,eval}`, `--data`, `--out`, `--epochs`, `--seed`, `--device` |
| `PYTHONPATH=src uv run python -m potok_ai.engine --model loopwm-lite --stage train --data <v2 file> --out /tmp/opencode/engine_lite --epochs 1 --seed 3 --device cpu --window 12 --batch 2 --limit 4 --eval-limit 4 --width 16 --lite-width 64 --attn-rank 8` | printed the one metrics table: `rmse_train=0.7375 rmse_held_out=0.7218 ratio=0.9787 params=34819 mac=207296 bytes=150095`; the `eval` stage on the same `--out` reproduced the same RMSE, params and MAC |

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

## Smart teacher / scenario holdout (SPEC_SMART.md v1, commit 8894951)

- `model.py` adds `SmartConfig` / `SmartTeacher`: a 50x14 window through a
  causal TCN (4 layers, width 128, TSCT rank 64, dilations 1/2/4/8), FiLM
  conditioning from the 6 vehicle numbers, two **weight-shared** passes over
  the same block stack, and AttnRes-style softmax allocation over the two loop
  outputs with a learned query and content keys. Head = two bounded corrections
  plus confidence. Width must be a power of two for the existing Hadamard
  activation quantiser.
- **FiLM, not a global token**: the conditioning is constant over a flight, so
  an MLP to per-channel `(gamma, beta)` reaches every frame directly; a token
  at frame 0 could be attenuated by the residual stack. Documented on
  `SmartTeacher`.
- **Loss**: correction MSE divided by the target's per-column std (a constant
  predictor scores ~1, so it cannot win), plus JEPA 0.05 and MTP K=3 0.02 as
  spec constants in `SmartConfig`. JEPA reuses `mask_windows`/`JEPAPredictor`
  with a locally deep-copied EMA of the teacher; MTP predicts the next 3 frames
  at every in-window position from the shared latent (one teacher forward per
  step is reused for correction, JEPA source and MTP source).
- **Data / split**: `data.py` reads the v2 `POTOKSIM` file that the simulator
  owner's `write_validation` actually emits: 80-byte header with `meta_count=17`
  in reserved bytes 48..52, then per episode `u64 index` + 17 f32 (legacy 9,
  6 conditioning, scenario id, split flag), then 14 features and 4 targets.
  `group_by_scenario` groups by `"scenario_<id>"`; v1 (8 features, 9 metadata)
  files still read with a zero conditioning vector but cannot train the smart
  CLI, which needs 14 features and an explicit held-out scenario. No vehicle
  names or per-vehicle splits exist: all episodes are randomized Potok-M.
  The Python reader was aligned with the Rust writer's bytes after that writer
  landed; the invented interim layout was dropped.
- `train_smart.py` CLI: `--data DIR_OR_FILE --epochs --seed --device cpu --out`
  plus sizing/limit overrides. It prints RMSE for the two corrections per
  scenario and `ratio = mean(held-out RMSE) / mean(train RMSE)` with the 1.25
  acceptance limit. Distillation trains a `StatefulTCN` from a 20-value frame
  (14 physics features + 6 conditioning) against the teacher with a small
  (0.1) weight on the simulation target; it exports v2 `model_kind=2` via the
  unchanged binary layout. `export.py` and `FORMAT.md` document 10 (legacy)
  vs 20 (smart) `in_features`. Edge is a separate owner.
- `tests/test_smart.py` covers the v2 reader and scenario split, loop forward
  shape/bounds, AttnRes weights summing to one, conditioning changing the
  output, std-normalised loss, per-scenario RMSE and ratio, a 20-feature
  student export/load round-trip, and one tiny CPU end-to-end step.

## Privileged LoopWM teacher (SPEC_SMART.md, arXiv:2606.18208 idea)

- `src/potok_ai/loopwm.py` — `LoopWMConfig` / `LoopWM`: one causal latent encoder
  (3-tap left-padded conv, width 128 by default), FiLM conditioning from the six
  vehicle numbers, and **one** `LoopBlock` (pre-norm causal self-attention + FFN)
  reused on every refinement pass. `refine()` adds the block's *increment* to the
  latent and exits when `mean ||increment||` drops below `exit_threshold`, hard
  capped at `max_iters = 6`; `RefineInfo` reports the iteration count and update
  trace. The readout emits two bounded corrections plus confidence from the last
  frame. The privileged simulator correction (true wind) is never a forward
  input — it only enters the training loss. No raw-sensor reconstruction: JEPA
  heads predict the EMA encoder's latents 3/6/10 frames ahead, MSE weighted 0.05.
  `exit_threshold` is a tunable knob (default `1e-3` is conservative: a fresh
  width-128 model's update norms are ~3.7-4.1, so all six passes run; raise it
  on real data when trading passes for latency is wanted).
- `train_loopwm.py` CLI (`--data --epochs --seed --device cpu --out`, plus sizing
  overrides): splits by v2 scenario metadata, trains on the non-held-out cells,
  prints RMSE per scenario and the held-out/train ratio, distils the LoopWM
  outputs into the unchanged 20-input `StatefulTCN` (reusing
  `train_smart.distil_student`), and exports v2 `model_kind=2`. It reuses the
  smart pipeline's scenario/eval/ratio helpers; `SmartTeacher` and
  `train_smart.py` are untouched, so both CLIs run side by side.
- Tiny CPU CLI smoke (synthetic v2 file, width 16, 2 epochs) printed three
  scenario RMSE rows, `ratio=0.8566 limit=1.25 generalises=yes`, teacher and
  export paths. `tests/test_loopwm.py` adds shared-weight iterative refinement
  (forward hook counts 6 calls through the single block), adaptive exit and cap,
  latent target/prediction shapes, conditioning sensitivity, privileged-target
  isolation, held-out ratio, and a tiny end-to-end train/distil/export step.
- Verification: `uv run pytest -q` → `51 passed`; `uv run ruff check src tests`
  → clean. CPU only; no GPU jobs, no pushes, no files outside `ai/`.

## Unified training engine (SPEC_ENGINE.md v1, commit fd5d2dc)

- `src/potok_ai/engine.py` — one CLI (`python -m potok_ai.engine`), one
  `TrainConfig` and one metrics report. `--model lrs|loopwm-lite|loopwm`,
  `--stage train|distill|eval`, `--data FILE`, `--out DIR`, `--epochs`, `--seed`,
  `--device cpu` plus sizing/limit overrides. `out/metrics.json` carries exactly
  the seven SPEC fields: `model, rmse_train, rmse_held_out, ratio, params, mac,
  bytes`; the same values are printed as one table. `rmse_train` /
  `rmse_held_out` are means over the v2 `scenario_<id>` groups, `ratio` reuses
  `train_smart.generalisation_ratio`, `bytes` is the real export size on disk.
- **Dispatch**: `loopwm` calls the existing `train_loopwm.run` (and its
  `train_smart` helpers) unchanged; `lrs` and `loopwm-lite` go through the
  shared adapter loop. Every adapter implements `build`, `train_step`,
  `evaluate`, `export`, `load`, `mac`, `count_params`. `distill` is the same
  pipeline as `train` — all three families already learn from a teacher, so
  there is no second training path. `eval` loads an export and scores it
  without training.
- **LRS adapter**: wraps `StatefulTCN` / `smart_student_config` (20-value
  frames = 14 features + 6 conditioning), one optimizer step per tick after
  the window, export via `export_student` (`out/lrs.bin`), load via
  `load_student_export().to_model()`. `train_smart.py` and `train_loopwm.py`
  were not changed.
- `src/potok_ai/loopwm_lite.py` — Perceiver encoder (50x14 -> 16 tokens, width
  64) via rank-8 cross-attention; ONE parameter-shared `LiteBlock` applied at
  most twice with adaptive exit on the mean latent-update norm (hard cap 2,
  `exit_threshold` configurable); heads for the full LoopWM's 3/6-frame latents
  plus a bounded 2-value correction and confidence. No raw reconstruction:
  latent targets are detached outputs of the teacher's own latent heads, the
  reference correction is loss-only and never a forward input. The teacher is
  kept out of the module tree (`object.__setattr__`), so it is never optimised
  or exported; `target_proj` is training-only and excluded from both the
  export and the reported parameter count.
- **MAC**: `analytical_mac()` counts every projection, attention score/sum and
  FFN matmul for one window inference: default 50 frames, 16 tokens, width 64,
  rank 8, two passes = **289 984 MAC (~0.29 M)**, inside SPEC_ENGINE.md's
  0.2-0.4 M band. Full-width attention here would be ~0.8 M; the rank-bottleneck
  attention and FFN are the deliberate latency trade. `loopwm_mac()` gives the
  privileged teacher's analytic count with the same convention.
- **Determinism**: one `seed_everything()` for all families (torch + numpy +
  one CPU thread). `StatefulTCN` also seeds its own local generator from
  `--seed`. Tests run seeds 7/7/9 and assert bit-equal exports for repeats and
  different weights for a changed seed.
- `tests/test_engine.py` (10 new tests): CLI help via subprocess, adapter
  registry and `train_loopwm` dispatch (monkeypatched), lite shape/exit/shared
  weights/teacher isolation, analytic MAC band, JSON schema and table, one tiny
  CPU end-to-end per family, `eval` stage, deterministic seeding. Verification:
  `uv run pytest -q` -> `61 passed`; `uv run ruff check src tests` -> clean.

## Skipped

- No GPU wheels, no A/B runs, no training beyond the tiny CPU smoke steps.
  LoopWM-lite's latent teacher is randomly initialised unless a previous
  `loopwm` stage left `out/loopwm.pt` in the same `--out` directory; wiring a
  real trained teacher checkpoint and measuring the 0.3-0.6 ms Pi estimate
  need a simulator-produced corpus and are not done here.
  The sim owner still has to emit v2 `POTOKSIM` files with real physics
  features, conditioning and scenario flags before a real 50-frame smoke run
  can replace the synthetic one; `sim/`, `edge/` and `docs/` were not touched.
- The simulator's episode writer does not exist yet, so `data.py` defines a magic
  header format plus a tolerant raw-float32 fallback. Align the header with
  `sim/README.md` when that lands.
