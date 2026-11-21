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
