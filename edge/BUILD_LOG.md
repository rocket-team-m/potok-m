# BUILD_LOG — potok-edge

## 2026-11-22 — initial implementation

- **Decision: pure Rust, no burn.** The task allows burn+ndarray "if
  practical" and a pure-Rust fixed-size implementation otherwise. This model
  is fixed-shape (50x8 window, 5 blocks, small C/D), has no autograd and no
  training, and the hot loop must be one multiply-add per weight from
  pre-dequantized f32 buffers. Adding the burn/cubecl dependency tree would
  cost minutes of compile time per clean build, complicate Raspberry Pi
  cross-compilation, and put a generic tensor API between the code and the
  exact arithmetic order we need. No burn dependency was added, so no claim
  is made that burn was tried and failed on this toolchain — it was rejected
  by design. The public API (`Model` -> `Runtime` -> `ReplyFrame`) is the same
  either way.
- Dependencies: `sha2 0.10` for the file checksum and `serialport 4` for UART.
  CLI parsing is hand-rolled; no clap.
- The protocol crate under `protocol/` is owned by another agent and is not
  ready. The frame layout from `protocol/uart.yaml` is reimplemented locally
  in `src/frame.rs` behind the `FrameLayout` trait; the only production
  dependency on the concrete layout is one line in `src/bin/infer.rs`.
- Test fixture is generated in Rust (`src/fixture.rs`); no binary artifact is
  checked in. `make-fixture` writes the same bytes for manual testing.
- Protocol version mismatch disables the assistant on the *first* frame
  (`handle_request` -> `Runtime::disable_now`); the three-strike rule applies
  to CRC/seq/timeouts only.

## 2026-11-22 — flop/copy optimization pass

Changes, all inside `edge/`:

1. **Ternary dequantization already happens once at load** (`Model::from_bytes`
   reconstructs dense f32 `W1 = U1 diag(s1) V1^T`, `W2 = U2 diag(s2) V2^T` and
   the depthwise kernels). The hot loop never touches a code byte.
2. **Fused per-timestep pass.** The five blocks now run one pass over the 50
   timesteps: for timestep `t`, norm1 -> causal depthwise conv + SiLU +
   residual -> norm2 -> TSCT up + SiLU + down + residual. The causal conv only
   needs norm1 rows `t-K..=t`, already final, so the fusion is exact.
3. **Smaller preallocated scratch.** `norm2` shrank from `T*C` to `C` and `up`
   from `T*D` to `D`; they are consumed in the same timestep. Scratch went
   5,600 -> 3,312 bytes on the fixture; `x` and `norm1` stay `T*C` because the
   conv needs causal history. No buffer is allocated in `on_request`.
4. **Bounds-check elimination.** Every inner dot receives pre-sliced rows
   (`&w[i*C..(i+1)*C]`) and iterates `zip`/`dot`, so no per-element index
   arithmetic remains in the matmul-like loops.
5. **NEON on aarch64, scalar fallback elsewhere** (`src/simd.rs`). `dot()`
   dispatches to a `#[target_feature(enable = "neon")]` implementation with a
   runtime `is_aarch64_feature_detected!` guard; four lanes per `vmulq_f32`,
   ordered lane sum, scalar tail. Other targets use the strictly ordered
   `dot_scalar`.

### Verification

- `cargo test`: 17 tests pass, no warnings. `cargo check`: clean.
- In-crate control: `src/baseline.rs` is a deliberately naive implementation
  (separate passes, per-frame allocation, indexed scalar loops, no fusion).
  `tests/optimized_test.rs` compares raw outputs over 40 windows and
  `bench --compare` over 2,000 and 5,000 frames: **max |diff| = 0.0,
  bit-identical on every frame** on x86_64. NEON changes only the summation
  tree, so the aarch64 assertion is bounded error (`1e-4*(1+|ref|)`), not
  bit identity. `simd::tests::dot_fast_matches_scalar_within_bound` pins that
  bound; it runs on the Pi at test time.
- The NEON intrinsics were **compile-checked for aarch64** with
  `rustup run stable rustc --target aarch64-linux-android --crate-type lib`
  (rustc 1.99.0) — exit 0. They were **not executed**: this box is x86_64 and
  there is no Pi here. Do not claim measured Pi latency until `cargo test`
  and `bench --compare` are run on the target.
- The 2,000-frame output checksum stayed `0x61a80` before and after the
  optimization; the 5,000-frame checksum is `0xf4240`.

### Before / after (release, x86_64 dev box, 2000 frames unless noted)

| metric | before | after (optimized) | naive control, same binary |
|---|---|---|---|
| p50 latency | 137 us | 96-116 us across 4 runs | 112-125 us |
| p99 latency | 183 us | 149-220 us | 169-237 us |
| resident model+buffers | 14,028 B (printed 13 KiB) | 11,740 B (11 KiB) | allocates per frame |
| output fingerprint (2000 frames) | 0x61a80 | 0x61a80 | identical |

The p99 spread is host noise (shared dev box), not a signal: three consecutive
5,000-frame runs gave optimized p99 219.6 / 149.7 / 149.4 us and naive p99
168.9 / 236.7 / 172.3 us. The trustworthy deltas are the p50 (~137 -> ~96-116
us) and the memory (-16.3%). `bench --compare` measures both paths in the
same process, so the control numbers are directly comparable with the
optimized ones; the "before" column is the earlier unoptimized release
binary on the same fixture.

Command: `cargo run --release --bin bench -- --model MODEL.bin --frames 2000 --compare`

## Canonical v2 support (supersedes the earlier integration-risk note)

The earlier note warned that `ai/src/potok_ai/export.py` emits `POTOKM1\0`
named-layer files with dilations, LayerNorm-after-conv and a tanh head while
this crate only read self-contained `POTM` files. **RESOLVED**: `ai/FORMAT.md`
v2 is now the canonical spec, implemented in `src/format_v2.rs`, and
`Runtime::load` dispatches on the magic bytes. The old POTM reader stays only
for the Rust fixture and the naive baseline; it does not block v2 loading.

Two defects were found by the cross-check and fixed before it went green:

1. **TSCT orientation.** `F.linear(x, W)` is `x @ W^T` with `W = (U*s) @ V^T`,
   so `W[out, in] = U[out,k]*s[k]*V[in,k]`: the `U` index is the linear output
   and the `V` index is the input. The first attempt transposed them (0.10
   absolute output error).
2. **Fused convolution read a partially updated residual.** The dilated
   depthwise conv must read the block's *input* at earlier timesteps, not the
   residual stream after this block's writes to those timesteps. A preallocated
   `block_in: [T*C]` row cache fixes it (0.043 absolute error before).

### Fixture cross-check (x86_64, debug)

`ai/tests/fixtures/tiny_model.bin`, window
`(arange(400,f32).reshape(1,50,8)-200)/100`:

| output | Rust | `tiny_expected.npy` | abs err |
|---|---:|---:|---:|
| g_pitch | -0.42798725 | -0.42798716 | 8.9e-8 |
| g_yaw | 0.15875198 | 0.15875186 | 1.2e-7 |
| confidence | 0.52251510 | 0.52251510 | 0 |

Documented tolerance: **1e-6 absolute** (~8 ULP at this magnitude); measured
max error 1.2e-7, about one ULP. The differences come from fp32 accumulation
order between Rust and PyTorch's CPU kernels, not from different math.
`tests/v2_fixture_test.rs` parses the `.npy` header directly (no numpy
dependency), verifies the checksum refusal and checks that the service
runtime loads v2 and mutes on a forbidden phase.

`cargo test`: 19/19 pass, no warnings. `cargo check`: clean. All files read
from `ai/` were read-only; nothing outside `edge/` was modified.

## 2026-11-22 — stateful student and speculative dispatch

Added `src/student_runtime.rs` (v2 `model_kind=2` reader, dequantized TSCT
factors, per-frame state, warmup), `src/speculative.rs` (`SpeculativeRuntime`
holding the existing windowed teacher **and** the student), `src/bin/bench_spec.rs`
(p50/p99 for all three dispatch paths), and `tests/speculative_test.rs`
(12 tests) plus `tests/real_models_spec_test.rs` (3 tests, skips when the data
volume is not mounted). Wired through `src/lib.rs`, `src/runtime.rs`
(`infer_window`, `max_correction`), `src/uart.rs` (`handle_spec_request`) and
`src/frame.rs` (`PHASE_BURN` alias). No files outside `edge/` were modified.

### Format mismatch found against the real export — FIXED

The first reader was written from SPEC_STUDENT.md alone, using a 40-byte
header (`width, blocks, loops, decay, g_max, n_records`) and a private fixture.
The real `ai/src/potok_ai/export.py::export_student` file uses the canonical
v2 header extended with `model_kind`: **56 bytes** —
`in_features, width, out_features, rank, n_blocks, loops, decay, g_max, flags,
n_layers` — and 20 layer records (`blocks.{b}.proj` TSCT,
`blocks.{b}.rec_h/rec_x`, `stem`, `norm`, `alloc_logits`, `g_max`,
`head_corr`, `head_conf`). The per-frame input is **10** floats (8 sensors +
phase + flags), not 8. The Python forward also has a stem projection, a
LayerNorm before the head, learned `alloc_logits` softmax, **outer** loop over
iterations with **inner** block chain, and scales corrections by
`tanh(|g_max|)` from the `g_max` tensor. The reader and the in-crate fixture
writer were rewritten to this exact layout; `ai/` was not touched. TSCT
dequantization keeps the canonical square-map convention
(`W[out,in] = U[out,k]·alpha_u·s[k]·V[in,k]·alpha_v`) already used by the
teacher reader. LayerNorm uses the **biased** variance (divide by n): the first
attempt used n−1 and read 0.9 % low against the Python model. Real-export
cross-check on a fixed 10-float frame: Rust
`[0.07282478, -0.02531921, 0.9089091]` vs Python
`[0.07282476, -0.02531959, 0.90890908]` — max error **5.3e-7**, asserted at
1e-6 in `real_models_spec_test.rs`.

Two naming/threshold facts, recorded rather than silently changed:

- `protocol/uart.yaml` calls phase 0 **BURN**; the edge frame copy calls it
  `PHASE_IDLE`. Added `PHASE_BURN` as an alias of the same byte and used it in
  the phase gate; no protocol file was edited.
- Both model heads are `tanh`-bounded by `g_max`, so `|g_t - g_s| <= 2·g_max`
  and the specified `> 3·g_max` divergence trip is unreachable with shared
  bounds. Kept the literal 3.0 as the default `divergence_factor`, with a
  public knob to exercise the log+warmup path in tests. Separately, the legacy
  POTM teacher returns **raw** head logits through `infer_window`, so the
  dispatch clamps both models to the system bound before applying or
  comparing them (this also fixed an 80 ms/cycle warmup storm).

### Verification

- `cargo check --all-targets`: clean, no warnings.
- `cargo test`: **35/35 pass** (5 lib, 4 model, 1 optimized, 7 runtime,
  12 speculative, 3 real-export, 3 v2 fixture). `cargo fmt` applied.
- Acceptance tests cover model_kind dispatch, student determinism and state
  reset, warmup replay, BURN/motor-armed phase gate, low-confidence teacher
  call, 20-frame forced-check cadence and its reset, CRC/seq bypass,
  checksum refusal, 30 s (1500 steps) finite bounded state, divergence log,
  and a <10 ms worst-case cycle.
- Release benchmark, 2000 synthetic frames, `bench-spec` (x86_64 dev box):
  normal p50/p99 **69.8 / 93.1 us**, low-confidence **186.8 / 224.7 us**,
  forced-check **132.1 / 205.2 us**; worst cycle 0.835 ms incl. 0.61 ms UART
  wire, well under 10 ms.
- Release benchmark with the **real exports** (`distill.bin` teacher,
  `student.bin` student): normal **47.1 / 87.0 us**, low-confidence
  **330.5 / 350.5 us**, forced-check **236.4 / 332.9 us**; worst cycle
  **0.960 ms** / 10 ms. Debug-build integration run (600 frames) measured
  p99 0.84 / 6.56 / 6.92 ms — also under budget.
- At most one teacher call per cycle is enforced by construction (single
  `if low_conf || forced` block), a `debug_assert!` on the counter delta, and
  a benchmark assertion `teacher_calls <= frames`.

Skipped: no GPU jobs, no pushes, no `ai/` edits. Outcome of a real A/B of
student vs teacher quality is out of scope here — this change only wires and
times the dispatch.

## 2026-11-22 — Python student forward update (supersedes the cross-check above)

`ai/src/potok_ai/student.py` changed during training work; read-only, not
modified. `src/student_runtime.rs` now matches it exactly. The new real export
(`potok-m-data/models/student.bin`, 75 611 B, g_max 0.910741) loads and runs.

### Spec deviations (authoritative, replace the older SPEC_STUDENT.md reading)

1. **Recurrent state is leaky.** Python
   `_StateBlock.forward`: `candidate = tanh(rec_x(x) + rec_h(h_prev))`,
   `h = 0.8*h_prev + 0.2*candidate`, and the TSCT projection is applied to the
   **new** `h`. SPEC_STUDENT.md's `h = tanh(W_h h_prev + W_x x)` with no carry
   is obsolete. The 0.8/0.2 coefficients are literals in the Python forward,
   not the model's `decay` field.
2. **Final feature is `LayerNorm(total + source)`**, where `total` is the
   decayed/allocated sum of loop outputs and `source` is the last block's
   output of the last iteration (the stem vector is overwritten by the block
   chain). Normalizing `total` alone is obsolete.
3. **Head:** `corrections = tanh(|g_max|) * tanh(head_corr(z))`,
   `confidence = sigmoid(head_conf(z))`, with `g_max` taken from the exported
   tensor. This part already matched; kept unchanged.

The module doc in `src/student_runtime.rs` carries the same three deviations.
Dispatch wiring (one teacher call per cycle, 20-frame forced check, BURN gate,
CRC/seq bypass, divergence log) is unchanged.

### Cross-check and verification

Python goldens regenerated from the **new** export with the `ai` venv
(`load_student_export(...).to_model().step/warmup`), read-only. Rust errors
are at most **1e-8**, asserted at 1e-5 in `tests/real_models_spec_test.rs`:

| case | Rust `[g0, g1, confidence]` | Python reference |
|---|---|---|
| zero state, one frame | `[0.012839871, -0.110565000, 0.813980000]` | `[0.012839884, -0.110564999, 0.813979983]` |
| after 3-frame warmup | `[0.004873989, -0.107443390, 0.817990240]` | `[0.004873985, -0.107443415, 0.817990184]` |
| after 50-frame warmup | `[-0.002074861, -0.110135900, 0.818366600]` | `[-0.002074862, -0.110135905, 0.818366587]` |

The 5.3e-7 agreement recorded in the previous section was against the **old**
export and old forward; it is superseded by the table above.

- `cargo fmt`, `cargo check --all-targets`: clean.
- `cargo test`: **35/35 pass**, including the 30 s bounded-state test (the
  convex `0.8/0.2` update keeps `|h| <= 1`).
- Release bench with the new real exports, 2000 frames: normal p50/p99
  **47.4 / 67.3 us**, low-confidence **294.4 / 461.5 us**, forced-check
  **285.4 / 372.5 us**; worst cycle **1.071 ms** incl. 0.61 ms UART wire,
  under the 10 ms budget.

## 2026-11-22 — smart 20-input student export

`ai/FORMAT.md` and `ai/src/potok_ai/export.py` (read-only) confirm the binary
layout is unchanged: `model_kind=2`, 56-byte header, and `in_features` is the
full frame width — **10** for the legacy student (8 sensors + phase + flags),
**20** for the smart student (14 physics features followed by 6 conditioning
numbers, `mass_kg, area_m2, cd0, static_margin_cal, thrust_peak_n,
gyro_noise`). No new header/meta fields were added in the file itself; the
conditioning travels in the frame, not in a separate record.

Changes inside `edge/`:

- `src/student_runtime.rs`: added `PHYSICS_FEATURES = 14`, `COND_FEATURES = 6`,
  `SMART_FRAME_FEATURES = 20`, `LEGACY_STUDENT_FEATURES = 10` and a
  `smart_frame(physics, conditioning)` builder. The loader already read the
  width from the header (`stem.weight` is `[width, in_features]`), so a 20-input
  export needed no reader change; both widths are now named and documented.
  `step(&[f32])` keeps zero-padding shorter frames exactly like Python's
  `_as_frame`, so legacy 10-input exports still work.
- `src/speculative.rs`: `SpeculativeRuntime::set_conditioning([f32; 6])`
  supplies the six numbers appended to every smart frame. When the student
  reports 20 inputs, indices 8/9 are **not** overwritten with phase/flags
  (they are physics features there); the conditioning is written to 14..20.
  Legacy 10-input students keep the old phase/flags behaviour.
- New `tests/fixtures/smart_student.bin` (78 171 B) and
  `tests/fixtures/smart_student_expected.txt`, generated by a read-only script
  that called `ai`'s `StatefulTCN(smart_student_config(seed=4242))` and
  `export_student()` through the `uv` venv, then reloaded the file with
  `load_student_export().to_model()` to produce goldens. Nothing in `ai/`,
  `sim/` or `docs/` was written.
- New `tests/smart_student_test.rs`: width assertion, three Python
  cross-check cases (zero state, 3-frame and 50-frame warmup), and a dispatch
  run with the smart student plus conditioning.

### Verification

- Rust vs Python for the smart export: single
  `[0.01507757, 0.04972160, 0.50769144]`, warm3
  `[0.04553854, 0.05702014, 0.50885016]`, warm50
  `[0.01920378, 0.06179270, 0.49901032]`; **max |error| = 5.96e-8**, asserted
  at 1e-5.
- The 10-input student at `potok-m-data/models/student.bin` still loads and its
  Python goldens still pass (the earlier section's table is unchanged).
- `cargo fmt`, `cargo check --all-targets`: clean. `cargo test`: **38/38 pass**,
  including the BURN phase gate, CRC/seq bypass, one-teacher-per-cycle and
  worst-case-cycle tests. Dispatch, phase gate and CRC behaviour unchanged.

## 2026-11-22 — lowlat profile

Added `potok-protocol` as a path dependency (`../protocol/potok-protocol`,
read-only) and `src/lowlat.rs`: a `FrameLayout` adapter whose encode/decode go
through `potok_protocol::encode_lowlat_request` / `decode_lowlat_request` /
`encode_lowlat_reply`. Magic, CRC, phase and version are checked by the
generated codec; protocol errors map onto `FrameError` (a new `BadPhase` variant
carries the rejected byte). The 14 int16 features are dequantized with
`LOWLAT_REQUEST_FEATURES_SCALES`; reply corrections are quantized with the
generated reply scales. Pressure/temperature/battery_mv (indices 10/12/13) are
decoded but not fed to the model, whose input uses the 8 sensors + height +
pitot. `compute_us` is written in 100 us units, saturating at 255 (25.5 ms),
from the dispatch-only duration recorded through the new
`FrameLayout::note_compute` hook (a no-op for compat). No heap allocation in
either adapter method.

`FrameLayout` gained `request_len` / `reply_len` (with a default `wire_bytes`),
so `run_loop` no longer hardcodes 56/14-byte buffers and measures dispatch time
for the reply field. `uart::serve_with_layout` lets the `infer` binary pick a
layout; `serve` stays compat. `infer` now accepts `--profile compat|lowlat`
(default compat) and defaults the baud to **921600** for compat and
**2000000** for lowlat; `--baud` overrides either. The config profile is a CLI
choice only — no runtime auto-detection, since the magics differ and firmware
owns the selection.

`bench-spec` accepts `--profile compat|lowlat|both` (default both) and
`--baud N`, and prints bytes per cycle, wire time at the chosen baud and
p50/p99 of the full cycle (encode request -> decode -> dispatch -> encode
reply, CRC and quantize included).

### Verification

- `cargo check --all-targets`: clean, no warnings. `cargo fmt` applied.
- `cargo test`: **44/44 pass**. New `tests/lowlat_test.rs` (6 tests): round trip
  and dequantization, bad CRC/magic/version/phase, int16 saturation at
  `i16::MIN/MAX`, the generated golden request/reply bytes re-encoded
  byte-identically, BURN phase gate plus three-miss disable, and dispatch with
  both profiles against both the 10-input and 20-input student.
- No `protocol/`, `ai/`, `sim/` or `docs/` files were written; the dependency is
  a read-only path reference.
- Release bench, 2000 synthetic frames, default bauds, 2026-11-22:
  - compat: **70 B/cycle**, wire **607.6 us @ 921600**, full cycle p50/p99
    **167.3 / 273.3 us** (p99 + wire 880.9 us).
  - lowlat: **58 B/cycle**, wire **232.0 us @ 2000000**, full cycle p50/p99
    **168.9 / 260.0 us** (p99 + wire 492.0 us).
  Both stay under the 10 ms dispatch budget; lowlat saves ~17% wire bytes and
  ~2.6x wire time, with essentially the same CPU cost.
