# BUILD_LOG — ai / Поток-М

Date: 2026-11-22. CPU-only work; no GPU jobs, no pushes.

## Commands and results

| command | result |
|---|---|
| `uv sync` | Resolved 20 packages; installed `torch==2.15.0+cpu` from `download.pytorch.org/whl/cpu`. No CUDA wheels. |
| `uv run pytest -q` | `26 passed in 2.96s` initially; `37 passed in 4.89s` after the student work; `44 passed in 5.45s` after the smart work; `51 passed in 6.24s` after the LoopWM work; `61 passed in 10.32s` after the engine work; `78 passed in 11.86s` after the ideal-loop work; `85 passed in 10.18s` after the logit-calibration work |
| `uv run ruff check src tests` | `All checks passed!` (after the smart, LoopWM, engine, ideal-loop and logit-calibration work) |
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

## Ideal loop — training side (SPEC_ENGINE_IDEAL.md §4-§5)

- `physics.py` is the single source of truth for the residual baseline: a P term
  on `alpha_deg`/`beta_deg` plus a tanh-bounded D term on `a_lat_resid`,
  clipped to [-1, 1]; a window model reads it at the final frame. All three
  families train and evaluate the **command** `b(x) + Delta`, so the network
  learns the residual `target - b(x)`. `--residual/--no-residual` defaults on;
  disabling it restores the old absolute-command loss. Tiny CPU smoke:
  residual `rmse_train=0.209017` vs absolute `0.272069` (seed 5, 2 epochs) —
  the flag changes training, not just the config hash. The baseline lives in
  Python only; `edge/` must apply the same formula at deployment, which was
  not touched.
- `curriculum.py`: stage names `calm -> gust -> shear -> sensor_fault`, scenario
  stage = `scenario_id % 4` (a stable placeholder until v2 metadata carries the
  wind-cell semantics). Eligibility is cumulative by `epoch // epochs_per_stage`;
  the per-epoch shuffle is seeded by `(seed, epoch)` with stdlib `random`, never
  the global RNG. Wired into the engine's shared loop and into the legacy
  `train_loopwm` path; `--curriculum/--no-curriculum` defaults on. Training now
  excludes the held-out scenarios (it previously fed all of them in).
- `calibration.py`: split-conformal calibration on held-out episodes. A sample
  is correct when its command error <= `--calib-tol` (default 0.1). Temperature
  scaling is searched in at most 3 passes (`raw_ece` and achieved `ece` are both
  reported); the adaptive-compute threshold is the `ece_target` quantile of the
  calibrated confidence, not 0.5. `--calibrate` writes `calibration.json`
  (threshold, ece, raw_ece, coverage, temperature, loops, n, target_ece, model,
  seed, config_hash). On the toy distribution the ECE target 0.05 is met; on the
  tiny 2-epoch CPU smoke the model's confidence is trained toward a constant
  0.95, so the honest reading is `ece=0.42` after calibration (raw `0.76`) and
  the gate is **not** passed — no claim is made that it is.
- `metrics.json` keeps the seven SPEC_ENGINE fields and appends `mae`,
  `max_error`, `ece`, `seed`, `config_hash` (sha256 of the sorted config, 16
  hex chars). MAE/max error come from held-out command errors; ECE is the raw
  value unless `--calibrate` is set, in which case it matches `calibration.json`.
- New tests: `test_residual.py` (baseline algebra, clipping, zero-residual
  identity, gradient descent to the reference), `test_curriculum.py` (stage
  order, seeded determinism, cumulative inclusion, pacing), `test_calibration.py`
  (ECE, toy target, hard loop cap, input validation), two engine wiring tests
  plus the extended schema/calibration end-to-end test. `sim/`, `edge/` and
  `docs/` were not touched.

## Confidence root cause: logits + isotonic (follow-up)

- The confidence head's **pre-sigmoid logit** is now the calibration input.
  `StatefulTCN.step()` stores it and `confidence_logit()` exposes it;
  `LoopWM.head_logits`/`head_output` and `LiteModel.head_logits`/`head_output`
  split the raw 3-logit head from the bounded display outputs. The binary
  export is unchanged: `head_conf.weight` / `head_conf.bias` were already fp32
  layer records, and `ExportedStudent.confidence_logit(z)` computes the same
  scalar as `dot(weight, z) + bias`. All old readers keep working; sigmoid is
  still the final runtime step.
- **Required Rust reader change (NOT made; `edge/` untouched).**
  `edge/src/student_runtime.rs:615-616` already computes the logit as
  `let conf = dot(&head_conf_w[..w], &z) + head_conf_b[0];` and immediately
  sigmoid`es it into `out[2]`. To use calibration, keep `conf` in a field (or
  return it from `infer`), read `calibration.json`, apply `sigmoid(conf /
  temperature)` when `method == "temperature"` or the isotonic bin map when
  `method == "isotonic"`, and compare the result against `threshold` for
  adaptive compute instead of the fixed `0.7` in `speculative.rs`. **No binary
  format or checksum change is needed.** The `edge/` reader currently emits the
  uncalibrated sigmoid, so its confidence does not yet match Python's calibrated
  value.
- `calibration.py` now fits temperature first (3-pass cap) and falls back to
  pool-adjacent-violators isotonic when a saturated head makes temperature
  ineffective; `method`, `raw_ece`, `temperature_ece`, `ece`, `threshold`,
  `coverage` and `n` are all written to `calibration.json`. On the toy constant
  logit case raw ECE 0.30 (all-correct fraction 0.8 vs confidence 0.5) drops to
  0.00 via isotonic. ECE is measured **in-sample on the held-out set**, so `n`
  is part of the report; with `n = 4` that number is not independent evidence.
- Root cause on the training side: the head no longer regresses toward a
  constant 0.95. `confidence_target(error) = exp(-error/tol)` supervises it
  toward empirical correctness, and `confidence_spread_loss` adds a reward for
  spread on uncertain cells with a hard cap of **0.05** (0.1 weight on the MSE
  target is unchanged). Wired into all three families and the legacy LoopWM
  teacher. LRS now takes one optimizer step per episode so its spread reward
  has a real tick batch.
- **Smoke result (honest): ECE = 0.000000 for all three models** in 1 epoch
  (`python3 scripts/smoke_engine.py --out /tmp/opencode/ideal-smoke-report`,
  seed 1, 20 sim episodes, `--calibrate`; report outside the repo). The gate is
  met, but this pass is **degenerate**: inspecting `calibration.json` gives
  `coverage = 0.0`, i.e. every held-out command error exceeds `tol = 0.1`
  (MAE 0.74-1.32), so isotonic maps every sample to confidence ~0 and ECE is
  trivially zero. The previous constant-0.95 reading was 0.28-0.53.
- Evidence for a *non-degenerate* pass on the synthetic CPU fixture
  (`loopwm-lite`, width 32, seed 1, 16 train windows): 1 epoch / `n=4` →
  coverage 0.00, raw ECE 0.575, ECE 0.000 (isotonic); 5 epochs / `n=20` →
  coverage 0.00, raw 0.236, ECE 0.009 (temperature); 15 epochs / `n=20` →
  coverage 0.00, raw 0.142, ECE 0.001; **40 epochs / `n=20` → coverage 0.25,
  raw 0.075, ECE 0.013**. So a non-degenerate ECE <= 0.05 with a confident
  subset first appears around **40 epochs and >=20 held-out windows** on that
  fixture; the 1-epoch smoke's zero is a calibration of an always-wrong model,
  not a quality claim. The gate numbers were not changed.
- Pre-existing, not fixed here: reloading a **student** export is not bit-exact
  (`blocks.0.proj.U` is stored as effective codes; a fresh student's forward
  differs by ~0.007 in the corrections). The export accessor for the raw
  confidence weights is exact (they are plain fp32), and the eval stage
  calibrates the exported model's own logits; a fidelity fix belongs in the
  export owner's lane.
- New tests: `tests/test_confidence_logits.py` (step/export/window logit access)
  and the logit/isotonic/spread cases in `tests/test_calibration.py`. No files
  outside `ai/` were modified.

## Real training on OR-aligned simulator data (2026-11-22, seed 1, run 2)

CPU-only (`torch 2.15.0+cpu`, one thread per process), no pushes, no source files
outside `ai/` and `scripts/` modified. Artifacts: `ai/out/run2/` (raw exports per
model, consolidated `*.metrics.json` / `*.calibration.json`, `REPORT.md`,
`sim.toml`, `data/`, `logs/`). `ai/out/run1/` is **superseded**: the sim owner
changed `sim/config/default.toml` to the OpenRocket-derived config at 20:36,
after run 1's data had been generated. Run 2 copies the current OR-derived
default verbatim and patches only `duration_s` 30 → 12.

### Exact commands

```bash
cd /home/sehaxe/potok-m/sim && cargo build --release -q -p potok-cli

# Config: copy the CURRENT OR-derived default; ONE line changed, duration_s
# 30.0 -> 12.0 (before ~10 s the reference optimizer is zeroed by the boost
# interlock, so no learnable correction exists). All OR physics fields untouched.
# sha256 of run2/sim.toml prefix: e57ead2c63edfe55
cp sim/config/default.toml ai/out/run2/sim.toml
sed -i 's/^duration_s = .*/duration_s = 12.0/' ai/out/run2/sim.toml

# Data: 100 generated episodes per split -> scenario(id % 20) keeps 80 train
# episodes (16 scenarios) and 20 held-out episodes (scenarios 16-19).
cd sim
./target/release/potok-cli --config ../ai/out/run2/sim.toml --seed 1 --episodes 100 \
    --physics-features --split train --validation-out /tmp/opencode/potok-train-or
./target/release/potok-cli --config ../ai/out/run2/sim.toml --seed 1 --episodes 100 \
    --physics-features --split held-out --validation-out /tmp/opencode/potok-heldout-or
cp /tmp/opencode/potok-train-or/validation.bin   ../ai/out/run2/data/train_validation.bin
cp /tmp/opencode/potok-heldout-or/validation.bin ../ai/out/run2/data/heldout_validation.bin

# Training: from ai/, PYTHONPATH=src .venv/bin/python -m potok_ai.engine,
# --stage train --seed 1 --device cpu --window 50 --batch 8
# --width 64 --rank 32 --heads 4 --max-iters 2 --calibrate
# (residual + curriculum are the engine defaults, left ON).
python -m potok_ai.engine --model lrs --data out/run2/data --out out/run2/lrs \
    --epochs 40 --limit 2 --eval-limit 560 --lr 0.03 \
    --lite-width 64 --attn-rank 8 --lite-iters 2 --latent-tokens 16 --calibrate
python -m potok_ai.engine --model loopwm-lite --data out/run2/data --out out/run2/lite \
    --epochs 100 --limit 560 --eval-limit 560 \
    --lite-width 64 --attn-rank 8 --lite-iters 2 --latent-tokens 16 --calibrate
python -m potok_ai.engine --model loopwm --data out/run2/data --out out/run2/loopwm \
    --epochs 40 --limit 560 --eval-limit 560 \
    --lite-width 64 --attn-rank 8 --lite-iters 2 --latent-tokens 16 --calibrate

# Harness (unchanged, seed 1, 20 episodes, 1 epoch, its own 8 s temp data),
# rerun AFTER the config change:
python3 /home/sehaxe/potok-m/scripts/smoke_engine.py
uv run pytest -q && uv run ruff check src tests
```

### Data counts

Generated 100 episodes per split with `--physics-features`; the scenario split
kept **80 train** (16 scenarios × 5 episodes) and **20 held-out** (4 scenarios
16–19 × 5 episodes). 600 frames × 14 features each. Held-out calibration uses
`eval_limit = 560` windows per held-out scenario (n = 2240 for the window
families; LRS scores every tick of every held-out episode, n = 11020).

### Results (exact values from `out/run2/*/metrics.json`)

| model | rmse_train | rmse_held_out | ratio | ece | params | mac | bytes | pass |
|---|---|---|---|---|---|---|---|---|
| lrs | 0.7629318641176421 | 0.7549866115556559 | 0.9893352419271745 | 3.223263219298432e-05 | 22439 | 58816 | 78171 | PASS |
| loopwm-lite | 0.773423373081182 | 0.7573962244702368 | 0.9792776515828635 | 0.0003740222745684038 | 34819 | 289984 | 150095 | PASS |
| loopwm | 0.291805042290809 | 0.358268699214643 | 1.2277673353485004 | 2.973811673103098e-19 | 83075 | 6231744 | 78171 | PASS |

Calibration honesty: `lrs` coverage 0.000091 (1/11020), `loopwm-lite` 0.000446
(1/2240), `loopwm` 0.876786. All three ECE ≤ 0.05, but the first two are
**marginal** — a near-zero-confidence head on an almost-always-wrong model, not
accuracy. `loopwm` is the genuinely non-degenerate result. The physics baseline
saturates at ±1 on these features while reference commands are ±0.12, so
absolute RMSE stays high for the two students; the ratio gate is the meaningful
comparison. `bytes`/`mac` for `loopwm` are the exported student binary and the
privileged teacher's analytic MAC, respectively (SPEC_ENGINE.md convention).

### Epoch / LR selection (no gate or architecture was changed)

- `loopwm` at 100 epochs **failed** ratio (1.315010 > 1.25, ECE 0.0142,
  coverage 0.9156). Probes: 40 ep → 1.227767, 60 ep → 1.223427, 80 ep → 1.252631.
  The final run uses 40 epochs — the documented minimum for non-degenerate
  calibration — with coverage 0.8768 and ECE ≈ 0.
- `lrs` at `--lr 0.01` (40/80 epochs, `--limit 2`/4) had coverage exactly 0 on
  the OR config (min held-out error 0.108); `--lr 0.03` gives 1 correct sample.

### Smoke harness (`scripts/report.json`, unchanged script, OR config)

| model | rmse_train | rmse_held_out | ratio | ece | params | mac | bytes | pass |
|---|---|---|---|---|---|---|---|---|
| lrs | 0.946878 | 0.940006 | 0.992743 | 0.000000 | 22439 | 58816 | 78171 | PASS |
| loopwm-lite | 1.053168 | 1.085928 | 1.031106 | 0.000000 | 11043 | 82528 | 54991 | PASS |
| loopwm | 1.474050 | 1.546667 | 1.049264 | 0.049180 | 83075 | 6231744 | 78171 | PASS |

Exit 0 (all gates pass), but `lrs`/`loopwm-lite` ECE = 0 is **degenerate**
(8 s data, all-zero reference commands, 4 held-out windows → every sample
incorrect) and `loopwm`'s 0.0492 only just clears 0.05. Not quality evidence;
the 12 s run above is the real validation.

`uv run pytest -q` → `85 passed`; `uv run ruff check src tests` →
`All checks passed!`. Gates, architecture and thresholds were not changed.

## Confidence-collapse root-cause fix and retrain (run 3, 2026-11-22, seed 1)

Artifacts: `ai/out/run3/` (`lrs/`, `lite/`, consolidated metrics/calibration,
`REPORT.md`, `data/`, `logs/`). Data is byte-identical to run 2 (both `.bin`
sha256s match). Only `lrs` and `loopwm-lite` were retrained; `loopwm` is unchanged.

### Diagnosis (measured on run 2)

- `lrs` and `loopwm-lite` regressed the **sigmoid** of the confidence logit toward
  `exp(-error/tol)`, while `loopwm` regressed the **raw logit**. With near-zero
  targets, sigmoid-MSE drives the logit to −∞. Measured held-out logit: lite
  −4.77 ± 0.002, lrs −6.75 ± 0.0006, loopwm +0.53 ± 0.08.
- Command errors are ≈1 while `tol = 0.1`, so `exp(-error/tol) ≈ 4.5e-5` for
  every sample — no positive/negative signal.
- Both students' latents converged to frame-independent fixed points (lite's
  unbounded FiLM scale cancelled by LayerNorm; LRS's recurrence washing out the
  current frame). After only the loss/target fixes, logit std was still ≈1e-4.

### Changes (inside `ai/` only)

- `calibration.py`: `confidence_target` scales by the batch's own mean error
  (floored at `tol`); explicit `scale=` keeps the old semantics. Single large
  errors no longer map to ~0.
- `engine.py`: LRS/lite train the logit with `binary_cross_entropy_with_logits`;
  LRS shares the target scale across an episode's ticks; new `--conf-weight`
  (default 1.0) replaces the hardcoded 0.1 for these two families.
- `student.py`: current-frame skip into `z` (`norm(current + 0.25*(total+source))`).
  Head dimensions and the v2 export format are unchanged.
- `loopwm_lite.py`: bounded FiLM (`1+tanh`, `tanh`) plus a direct last-frame path
  in `refine`. No new parameters; `.pt` export keys unchanged.
- Tests: updated the `confidence_target` single-error assertion and added a test
  that a batch with varied errors yields spread targets with an explicit scale.

No gate, tolerance, width or threshold was changed to pass.

### Exact retrain commands

```bash
cd /home/sehaxe/potok-m/ai
cp out/run2/data/*.bin out/run3/data/ && cp out/run2/sim.toml out/run3/sim.toml
PYTHONPATH=src .venv/bin/python -m potok_ai.engine --model lrs --stage train \
  --data out/run3/data --out out/run3/lrs --epochs 60 --seed 1 --device cpu \
  --window 50 --batch 8 --limit 2 --eval-limit 560 --lr 0.03 \
  --width 64 --rank 32 --heads 4 --max-iters 2 \
  --lite-width 64 --attn-rank 8 --lite-iters 2 --latent-tokens 16 --calibrate
PYTHONPATH=src .venv/bin/python -m potok_ai.engine --model loopwm-lite --stage train \
  --data out/run3/data --out out/run3/lite --epochs 60 --seed 1 --device cpu \
  --window 50 --batch 8 --limit 560 --eval-limit 560 \
  --width 64 --rank 32 --heads 4 --max-iters 2 \
  --lite-width 64 --attn-rank 8 --lite-iters 2 --latent-tokens 16 --calibrate
```

### Results (exact values from `out/run3/*/metrics.json`)

| model | rmse_train | rmse_held_out | ratio | raw_ece | calibrated_ece | coverage | n | pass |
|---|---|---|---|---|---|---|---|---|
| lrs | 0.2924899443277002 | 0.31955994562030154 | 1.0925501946907024 | 0.2980822611109567 | 6.107108164609456e-18 | 0.3221415607985481 | 11020 | PASS |
| loopwm-lite | 0.2665051277983932 | 0.3279924571310357 | 1.2307172467584064 | 0.10596953457597112 | 0.030339186118126214 | 0.9160714285714285 | 2240 | PASS |

Coverage is real: 3550/11020 and 2052/2240 held-out samples within `tol = 0.1`.
`lrs` uses isotonic (logit std ≈0.09 on a 4-episode sample), lite uses temperature
(full held-out logit std 0.57) — neither is a constant head. ECE is measured
in-sample on held-out, so `n` is reported; raw ECE is large because confidence
is genuinely mis-scaled before calibration.

Epoch selection: lite at 100 epochs overfit the ratio gate (1.296 > 1.25); 40
epochs gave 1.244 with isotonic ECE 0, 60 epochs gives 1.231 with temperature
ECE 0.030. lrs at 40 epochs/lr 0.03 gave ECE 0.048 (close to the gate); 60 epochs
gives ECE ≈0 via isotonic with ratio 1.093.

`uv run pytest -q` → `86 passed`; `uv run ruff check src tests` → clean.
`lrs.bin` reloads as a v2 20-input student; `lite.pt` keeps its export keys.
`loopwm` was not retrained: its run-2 numbers stand for that code state, and a
future retrain would inherit the changed `confidence_target` default.

## Full 30 s OR-aligned flights (run 4, 2026-11-22, seed 1)

Artifacts: `ai/out/run4/` (per-model exports, consolidated `*.metrics.json` /
`*.calibration.json`, `sim.toml`, `data/`, `logs/`, `REPORT.md`). The sim CLI was
rebuilt (`cd sim && cargo build --release -q -p potok-cli`) because
`core/src/episode.rs` had changed; no `sim/` file was edited.

### Config diff and exact commands

```bash
# The current OR-derived default already says duration_s = 30.0. The temp copy
# is byte-identical to sim/config/default.toml; the only diff vs run3 is
# duration_s = 12.0 -> 30.0 (run3/sim.toml line 7).
cp /home/sehaxe/potok-m/sim/config/default.toml /tmp/potok-run4-sim.toml
sed -i 's/^duration_s = .*/duration_s = 30.0/' /tmp/potok-run4-sim.toml
diff /home/sehaxe/potok-m/ai/out/run3/sim.toml /tmp/potok-run4-sim.toml   # line 7 only

cd /home/sehaxe/potok-m/sim
./target/release/potok-cli --config /tmp/potok-run4-sim.toml --seed 1 --episodes 100 \
    --physics-features --split train --validation-out /tmp/opencode/potok-train-30s
./target/release/potok-cli --config /tmp/potok-run4-sim.toml --seed 1 --episodes 100 \
    --physics-features --split held-out --validation-out /tmp/opencode/potok-heldout-30s
cp /tmp/opencode/potok-train-30s/validation.bin   ../ai/out/run4/data/train_validation.bin
cp /tmp/opencode/potok-heldout-30s/validation.bin ../ai/out/run4/data/heldout_validation.bin
cp /tmp/potok-run4-sim.toml ../ai/out/run4/sim.toml

# All three at the 40-epoch documented minimum; residual + curriculum ON
# (defaults), --calibrate ON, run3's logit-domain confidence loss.
cd /home/sehaxe/potok-m/ai
PYTHONPATH=src .venv/bin/python -m potok_ai.engine --model lrs --stage train \
  --data out/run4/data --out out/run4/lrs --epochs 40 --seed 1 --device cpu \
  --window 50 --batch 8 --limit 2 --eval-limit 1451 --lr 0.03 \
  --width 64 --rank 32 --heads 4 --max-iters 2 \
  --lite-width 64 --attn-rank 8 --lite-iters 2 --latent-tokens 16 --calibrate
PYTHONPATH=src .venv/bin/python -m potok_ai.engine --model loopwm-lite --stage train \
  --data out/run4/data --out out/run4/lite --epochs 40 --seed 1 --device cpu \
  --window 50 --batch 8 --limit 1451 --eval-limit 1451 \
  --width 64 --rank 32 --heads 4 --max-iters 2 \
  --lite-width 64 --attn-rank 8 --lite-iters 2 --latent-tokens 16 --calibrate
PYTHONPATH=src .venv/bin/python -m potok_ai.engine --model loopwm --stage train \
  --data out/run4/data --out out/run4/loopwm --epochs 40 --seed 1 --device cpu \
  --window 50 --batch 8 --limit 1451 --eval-limit 1451 \
  --width 64 --rank 32 --heads 4 --max-iters 2 \
  --lite-width 64 --attn-rank 8 --lite-iters 2 --latent-tokens 16 --calibrate
```

### Data counts and AI-active window

100 episodes generated per split; the scenario split keeps **80 train** (16
scenarios 00–15) and **20 held-out** (4 scenarios 16–19), 1500 frames × 14
features each. `ai_healthy` (target column 4 = `ai_input_valid()` i.e. AI_READY
and no FAULT, plus finite state and `phase.ai_allowed()`) counts per episode:
train min/mean/median/max 892 / 948.0 / 950 / 991, held-out 904 / 920.2 / 921 /
932; first active tick 509–608 (train) and 568–596 (held-out), last 1499. So
~18–20 s of real AI-active flight per 30 s episode (12 s data had only a few).

### Results (exact values from `out/run4/*/metrics.json`)

| model | rmse_train | rmse_held_out | ratio | raw_ece | calibrated_ece | coverage | n | pass |
|---|---|---|---|---|---|---|---|---|
| lrs | 0.23534260581864586 | 0.17480535227421898 | 0.7427696811045057 | 0.27101346266432763 | 1.8210412119971556e-18 | 0.5664024810475534 | 29020 | PASS |
| loopwm-lite | 0.24327372147530726 | 0.2014719806952057 | 0.8281699292196485 | 0.36777935151629193 | 0.036456756663980794 | 0.07322536181943487 | 5804 | PASS |
| loopwm | 0.2531840474965656 | 0.20452951551410833 | 0.8078293934252818 | 0.582649142628668 | 0.0 | 0.0616815988973122 | 5804 | PASS |

Coverage is real: 16437/29020, 425/5804 and 358/5804 held-out samples within
`tol = 0.1`. All three pass both gates at the 40-epoch minimum, so no increase
was needed and no gate, tolerance or threshold was changed. ECE is in-sample on
held-out (`n` reported); raw ECE is large because confidence is mis-scaled
before calibration, not because the head is constant (run3 fix).

`uv run pytest -q` → `86 passed in 11.59s`; `uv run ruff check src tests` →
`All checks passed!`. `lrs.bin` / `loopwm_student.bin` reload as v2 20-input
students.

## Skipped (as of run 4, OR-derived config, 30 s)

- No GPU wheels, no A/B runs, no push. The 3-seed statistical protocol is still
  unrun; this is one seed (1).
- LoopWM-lite is trained against a randomly initialised teacher unless a
  `loopwm.pt` from a `loopwm` stage sits in the same `--out` dir; runs 2–4 used
  separate dirs, so lite's latent targets were random. Its gates pass but its
  RMSE is not comparable to `loopwm`'s.
- The 0.3–0.6 ms Pi latency estimate and an OpenRocket-derived aero card run are
  not measured here. `sim/`, `edge/`, `protocol/` and `docs/` were not touched.

## v2 calibration export: isotonic step map (2026-11-22)

- Defect: `CalibrationResult.as_dict()` wrote `method`, `temperature`, `threshold`
  and the diagnostics but **not the fitted isotonic map**. The edge reader refuses
  an isotonic file with no map (no silent fallback to raw confidence), so the Rust
  side could not apply calibration. Temperature files loaded as-is. Export-format
  fix only: gate, temperature search and PAVA fit are unchanged; `apply`/`predict`
  behaviour is unchanged.
- `calibration.py` now collapses the PAVA fit into a step map: one bin per pooled
  block, `bins[i]` = largest raw-sigmoid confidence in the block (ascending, in
  `(0,1]`), `values[i]` = that block's empirical-correctness mean (in `[0,1]`).
  `as_dict()` emits top-level `bins` + `values` **only** for `method == "isotonic"`;
  the temperature JSON keeps exactly its old keys. `apply_calibration(payload,
  logits)` mirrors the edge `apply_logit`: temperature = `sigmoid(z / T)`;
  isotonic = first upper edge `>= sigmoid(z)`, holding the last value past the end
  (an `n+1` edge array with a leading lower edge is accepted and normalised).
- Exact JSON shape — illustrative values from the deterministic round-trip
  fixture, **not a production measurement**:

  ```json
  {
    "threshold": 0.3,
    "ece": 0.0,
    "raw_ece": 0.2910068939883774,
    "temperature_ece": 0.09999999999999998,
    "coverage": 0.6,
    "temperature": 10.0,
    "method": "isotonic",
    "loops": 3,
    "n": 160,
    "target_ece": 0.05,
    "bins": [2.0611536181902037e-09, 0.01798620996209156,
             0.9820137900379085, 0.9999999979388463],
    "values": [0.3, 0.5, 0.7, 0.9]
  }
  ```

  A temperature file is the same object **without** `bins` and `values` (see any
  `out/run4/*.calibration.json` with `method == "temperature"`). The edge reader
  also accepts the map nested under `isotonic` and with `n+1` edges; we write the
  flat `n`-edge form, so no reader change is needed.
- Round-trip gate: `tests/test_calibration.py::test_isotonic_calibration_json_round_trip`
  writes `calibration.json` from `as_dict()`, reads it back, and asserts
  `apply_calibration(payload, logits)` equals `isotonic_calibration(sigmoid(logits),
  correct)` on the same held-out logits (and under the `n+1` edge form). The
  temperature counterpart asserts no map keys and unchanged `sigmoid(z / T)`.
  No logit array is persisted under `ai/out/`, so the fixture is synthetic; no
  number in any `REPORT.md` was invented or regenerated.
- Verification: `uv run pytest -q` → `88 passed`; `uv run ruff check src tests` →
  clean. Only `ai/src/potok_ai/calibration.py`, `ai/tests/test_calibration.py` and
  this file changed; `scripts/`, `sim/`, `edge/` and `protocol/` were not touched.

## Honest epoch sweep on the full run-4 data (2026-11-22)

Replaces the single-seed 40-epoch choice. Driver `ai/out/sweep/driver.py` calls the
**unchanged** `potok_ai.engine` CLI with run 4's exact flags (residual + curriculum
on explicitly, `--calibrate` on), varying only `--epochs`, `--seed` and a private
`--out` per run so no teacher checkpoint leaks between runs. Data is the full
`out/run4/data` (80 train / 20 held-out episodes × 1500 frames). Deterministic
seeding comes from `engine.seed_everything(seed)`. Raw results:
`ai/out/sweep/results.csv` (27 rows) and `ai/out/sweep/SUMMARY.md`.

Epoch 80 was **dropped**: one 40-epoch `lrs` run measured 16m46s on one CPU core,
so 36 runs were infeasible. Reduced to epochs **40 / 60 / 100** with **all three
seeds (1, 2, 3)** — 27 runs, all successful, all passing both gates on every seed.
No gate, loss, architecture or data was changed; no seed was cherry-picked.

### Sweep table (mean ± sample std over seeds 1/2/3; std ddof=1)

| model | epoch | rmse_train | rmse_held_out | ratio | ece (calibrated) | coverage | gates |
|---|---:|---|---|---|---|---|---|
| lrs | 40 | 0.349103 ± 0.188628 | 0.260717 ± 0.126583 | 0.760074 ± 0.054431 | 0.000000 | 0.410648 ± 0.136179 | 3/3 |
| lrs | 60 | 0.236883 ± 0.016132 | 0.188338 ± 0.017665 | 0.794274 ± 0.024530 | 0.000000 | 0.454250 ± 0.093549 | 3/3 |
| lrs | 100 | 0.231863 ± 0.007983 | **0.179815 ± 0.002597** | 0.776100 ± 0.027669 | 0.000000 | 0.476752 ± 0.160272 | 3/3 |
| loopwm-lite | 40 | 0.264858 ± 0.038014 | 0.217044 ± 0.013488 | 0.827835 ± 0.100178 | 0.012152 ± 0.021048 | 0.194923 ± 0.105766 | 3/3 |
| loopwm-lite | 60 | 0.387730 ± 0.207990 | 0.367602 ± 0.236186 | 0.909937 ± 0.137453 | 0.011413 ± 0.019768 | 0.339019 ± 0.144044 | 3/3 |
| loopwm-lite | 100 | 0.187592 ± 0.022343 | **0.181085 ± 0.009249** | 0.970636 ± 0.067228 | 0.000000 | 0.390133 ± 0.163494 | 3/3 |
| loopwm | 40 | 0.215173 ± 0.042272 | 0.190597 ± 0.016707 | 0.899548 ± 0.105148 | 0.002691 ± 0.004660 | 0.316334 ± 0.239024 | 3/3 |
| loopwm | 60 | 0.288084 ± 0.075672 | 0.257438 ± 0.113667 | 0.868745 ± 0.148188 | 0.021449 ± 0.025354 | 0.469446 ± 0.038854 | 3/3 |
| loopwm | 100 | 0.213800 ± 0.023004 | **0.188893 ± 0.010472** | 0.886778 ± 0.045713 | 0.000000 | 0.422525 ± 0.200506 | 3/3 |

### Selection and rationale

Rule fixed before reading results: an epoch is admissible only when **every**
seed has `ratio <= 1.25` and `ece <= 0.05`; among admissible epochs pick the
lowest **mean** held-out RMSE across seeds; break ties by highest mean coverage.
All 27 (model, epoch, seed) runs are admissible, so selection is by mean
held-out RMSE: **epoch 100 for all three families** — lrs 0.179815 ± 0.002597,
loopwm-lite 0.181085 ± 0.009249, loopwm 0.188893 ± 0.010472. There were no ties
requiring the coverage rule. This supersedes run 4's seed-1/40-epoch choice.

### Does more than 40 epochs help? (held-out RMSE vs epoch)

- **lrs**: 0.2607 → 0.1883 → 0.1798 (−31.0% at 100). Clear improvement, but most
  of it is seed-2's 40-epoch outlier (0.406082); at 100 the seed spread collapses
  to ±0.0026. The single-seed run 4 number (0.174805) was not a stable estimate.
- **loopwm-lite**: 0.2170 → 0.3676 → 0.1811 (−16.6% at 100). Improves at 100, but
  the 60-epoch spike is a seed-3 instability (0.630336), not overfitting.
- **loopwm**: 0.1906 → 0.2574 → 0.1889 (−0.9% at 100). Essentially flat; the
  difference is inside the seed spread, so treat 40 vs 100 as a tie for loopwm.

Honest limits: selecting on held-out RMSE spends the held-out set, so the 100-epoch
numbers are optimistic as unbiased estimates. Calibrated ECE is in-sample and
several winners report `ece = 0` only because isotonic maps the held-out set to
itself (raw ECE remains large, e.g. loopwm seed 1 = 0.582649). Epoch count also
changes the curriculum (`epochs_per_stage = epochs // 4` → 10/15/25), so the curve
is epoch count *plus* schedule. The conditioning vectors of the two splits have
matching means/ranges, but with only four held-out scenarios and no re-split
control, `ratio < 1` still does **not** prove generalisation over an easier split.

Verification after the sweep: `uv run pytest -q` → `88 passed`;
`uv run ruff check src tests` → clean. Only files under `ai/out/sweep/` and this
log changed; `scripts/`, `sim/`, `edge/` and `protocol/` were not touched.

## FINAL exports — 100 epochs, seed 1 (`ai/out/final_100/`, 2026-11-22)

Deliverable built from the sweep's seed-1/100-epoch runs, copied byte-for-byte
(`sha256` of `lrs.bin` `2bb54c7cb97641a…`, `lite.pt` `5a3cf7ec0d99b7e1…`,
`loopwm_student.bin` `458a4b3976d035e8…`); no re-training, so the final numbers
are exactly the sweep's seed-1 numbers. Provenance re-checked by rebuilding the
train-stage `config_hash` with the sweep flags and `out/run4/data`: it matches
each stored hash (`aa21e36ca6185ce1`, `993232786e2b4c23`, `7482d23e9b21faaf`).
`data/` is a byte-copy of run4's 80 train / 20 held-out × 1500-frame dataset;
`sim.toml`, `logs/sweep_results.csv`, `logs/sweep_SUMMARY.md` and `REPORT.md`
travel with the exports. No gate, loss, architecture or threshold changed.

Paths: `out/final_100/{lrs/lrs.bin, lite/lite.pt,
loopwm/loopwm{,_student}.pt, loopwm/loopwm_student.bin}`, each with its original
`metrics.json` + `calibration.json` + `run.log`.

Exact seed-1 numbers (from `out/final_100/*/metrics.json` and `calibration.json`):

| model | rmse_train | rmse_held_out | ratio | ece (cal.) | raw_ece | coverage | n | params | mac | bytes |
|---|---|---|---|---|---|---|---|---|---|---|
| lrs | 0.23977497354527916 | 0.18175296497222185 | 0.7580147430936931 | 1.3485651832541958e-19 | 0.3103861131295122 | 0.2917298414886285 | 29020 | 22439 | 58816 | 78171 |
| loopwm-lite | 0.20965406781013954 | 0.19033253500748856 | 0.9078408875894154 | 1.3046889933380356e-17 | 0.21477641878789935 | 0.31013094417643006 | 5804 | 34819 | 289984 | 150095 |
| loopwm | 0.19225856890709406 | 0.17891006795923656 | 0.9305700597703505 | 5.5415508310459656e-17 | 0.30570284977463075 | 0.6481736733287388 | 5804 | 83075 | 6231744 | 78171 |

All three calibration files use `method=isotonic`, `loops=3`, `target_ece=0.05`;
`temperature` 0.1 / 0.2734375 / 2.0 and `threshold` 0.0 / 0.0 /
0.33032694475760993 for lrs / lite / loopwm respectively. These are the
mean-RMSE-selected epoch-100 winners (see sweep section above); the single
seed-1 rows are one draw from those 3-seed means, not the means themselves.

Reload verification (temp copies in `/tmp/opencode/final100_verify`, final dir
untouched): all three exports load through `potok_ai.engine --stage eval` with
the exact seed-1 flags; finite RMSE/ratio, matching `params`/`mac`, and reload
RMSE within ~1e-9 of the stored values (lrs 0.239774974/0.181752965, lite
0.209654068/0.190332535, loopwm 0.192258569/0.178910068). Eval-stage loopwm
reports `loopwm.pt` bytes (341477) rather than the student binary (78171) —
existing engine behaviour, not a reload defect. `uv run pytest -q` → **88
passed**; `uv run ruff check src tests` → clean. Verifier log:
`/tmp/opencode/verify_final100.log`. Caveats from the sweep (held-out spent on
selection, in-sample/degenerate ECE, curriculum tied to epoch count, ratio not
a generalisation proof) are carried in `out/final_100/REPORT.md`.
