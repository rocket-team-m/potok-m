# potok-edge

CPU inference service for the Potok-M assist loop on Raspberry Pi. No CUDA,
no autograd, no training, no Python in the loop. Builds with plain
`cargo test` / `cargo check`.

## What it does

- Reads a frozen model binary (v1) into dense f32 buffers at load time.
- Runs a causal TCN: input projection, then **5 residual depthwise-separable
  blocks** with a TSCT low-rank feedforward, then a 3-way head
  (`g_pitch`, `g_yaw`, confidence).
- Keeps a 50-frame × 8-feature sliding window and **all scratch buffers**
  allocated once; a frame costs no heap allocation.
- Talks the local copy of `potok-uart` v1 over a serial port at 921600 baud:
  50 Hz request loop, 8 ms read timeout, CRC16 filtering, sequence check,
  forbidden-phase muting and a 3-miss latched disable.

## Model binary format

`Runtime::load(path)` picks the format from the magic bytes. The **canonical
format is v2** (`POTOKM1\0`), defined by `ai/FORMAT.md`; `src/format_v2.rs`
implements its self-describing header (width, rank, kernel, five dilations,
`g_max`, flags), named layer records, ternary codes + scales, SHA256 trailer,
and the matching forward: stem, dilated causal depthwise conv, LayerNorm
**after** the conv, fused TSCT pointwise with residual from the un-normalized
block input, final LayerNorm, `tanh(g_max)` corrections and sigmoid confidence.
Codes are used as stored (`alpha_u`/`alpha_v` and per-rank `s`); they are never
requantized. 4-bit FWHT and ternary STE are training-only and are not
implemented. The legacy POTM reader below is kept for the self-contained
Rust fixture and the in-crate naive baseline; it does not block v2 loading.

Cross-check: `tests/v2_fixture_test.rs` reads
`ai/tests/fixtures/tiny_model.bin` and `tiny_expected.npy` (read-only) and
compares the three float32 outputs with a documented 1e-6 bound (measured max
error 1.2e-7, about one ULP).

### Legacy POTM (v1)

All integers little-endian. File = header + layer descriptors + tensor table
+ payload + 32-byte SHA256 checksum. The checksum covers every byte before it.

### Header (48 bytes)

| offset | type | field |
|---|---|---|
| 0 | `[u8;4]` | magic `POTM` |
| 4 | u32 | format version = 1 |
| 8 | u32 | number of residual blocks (5 in production) |
| 12 | u32 | channels `C` |
| 16 | u32 | depthwise kernel `K` (causal) |
| 20 | u32 | TSCT rank `r` |
| 24 | u32 | window length `T` (50) |
| 28 | u32 | input features (8) |
| 32 | u32 | output features (3) |
| 36 | u32 | feedforward width `D` |
| 40 | u32 | tensor count = `5 + 10 * layers` |
| 44 | u32 | reserved, must be 0 |

### Layer descriptor (8 bytes, one per block)

| offset | type | field |
|---|---|---|
| 0 | u32 | base index into the tensor table, must be `5 + 10*l` |
| 4 | u32 | reserved |

### Tensor descriptor (20 bytes)

| offset | type | field |
|---|---|---|
| 0 | u8 | kind: 1 = f32, 2 = ternary |
| 1 | `[u8;3]` | reserved |
| 4 | u32 | rows |
| 8 | u32 | cols (0 = vector) |
| 12 | u32 | byte offset into payload |
| 16 | u32 | byte length |

Encodings:

- **f32**: `rows * max(cols,1)` little-endian floats.
- **ternary**: `rows` little-endian f32 scales, then `rows*cols` code bytes
  where `0 = -1`, `1 = 0`, `2 = +1`. Dequantized value is `code * scale[row]`.
  This is b1.58 with a per-row scale `alpha_w = mean|W|` (concept §4, eq. 4).

Global tensors, fixed indices:

| index | shape | content |
|---|---|---|
| 0 | `[C, 8]` f32 | input projection |
| 1 | `[C]` f32 | input bias |
| 2 | `[C]` f32 | final RMSNorm gain |
| 3 | `[3, C]` f32 | output head |
| 4 | `[3]` f32 | output bias |

Per-layer tensors, at `base .. base+9`:

| index | shape | content |
|---|---|---|
| +0 | `[C, K]` ternary | depthwise causal kernel |
| +1 | `[C]` f32 | depthwise bias |
| +2 | `[C]` f32 | RMSNorm gain 1 |
| +3 | `[C]` f32 | RMSNorm gain 2 |
| +4 | `[D, r]` ternary | TSCT `U1` |
| +5 | `[C, r]` ternary | TSCT `V1` |
| +6 | `[r]` f32 | TSCT `s1` |
| +7 | `[C, r]` ternary | TSCT `U2` |
| +8 | `[D, r]` ternary | TSCT `V2` |
| +9 | `[r]` f32 | TSCT `s2` |

The two low-rank matrices are reconstructed once at load time:
`W1 = U1 diag(s1) V1^T` (`[D,C]`) and `W2 = U2 diag(s2) V2^T` (`[C,D]`).
The forward pass then uses only dense f32 matrices.

Tensor payloads are contiguous and in table order; the reader refuses gaps
and trailing bytes. Any unknown version, bad shape, invalid ternary code or
checksum mismatch is a hard error.

### Forward pass

For each of the `T` window frames (oldest first):

1. `x_t = W_in * frame_t + b_in`.
2. Per block: RMSNorm over channels → depthwise causal conv (`K` taps, zero
   left padding) → SiLU → residual add → RMSNorm → `up = W1 * norm` → SiLU →
   `down = W2 * up` → residual add. This is the depthwise (spatial) plus
   pointwise (TSCT, channel-mixing) separable pair.
3. Final RMSNorm on the last timestep → `W_out * h + b_out`. Corrections are
   clamped to `±0.1` rad, confidence passed through a saturating range.

## Protocol (local copy)

The canonical frame description is `protocol/uart.yaml`, owned by the
`protocol/` agent. That crate is not ready yet, so `src/frame.rs` is an
isolated reimplementation behind the `FrameLayout` trait. Swapping in the
generated crate means writing one `impl FrameLayout for GeneratedLayout`
and changing one line in `src/bin/infer.rs`.

- Request (56 bytes): magic `0x504D`, seq u16, time_us u32, phase u8,
  version u8, flags u8, reserved u8, gyro/accel/mag `3*f32` each, height f32,
  pitot dp i16, CRC16 u16.
- Reply (14 bytes): magic `0x5250`, seq u16, `g_pitch` i16, `g_yaw` i16
  (scale 0.001), confidence u8, ai_healthy u8, version u16, CRC16 u16.
- CRC-16/CCITT-FALSE: poly `0x1021`, init `0xFFFF`, no reflection, xor-out 0,
  over all bytes except the CRC field.

Safety behaviour:

- **Phase gate** (default): `Idle`, `Armed`, `Fault` and any frame with the
  `MOTOR_ARMED` flag get exactly zero corrections, confidence 0,
  `ai_healthy = 1`. The current protocol reports coast/apogee/descent as
  `Flight`, so `Flight` is allowed only with `MOTOR_ARMED` clear. The gate is
  a `PhasePolicy` and can be tightened per deployment.
- **Errors**: bad magic, version, CRC, wrong length and sequence gaps or
  duplicates are dropped and counted. On a sequence error the counter
  resynchronises to the received sequence so one bad frame does not poison
  the stream. A fully valid frame clears the consecutive-miss counter.
- **Three consecutive misses** latch `ai_healthy = 0` and zero output until
  the process is restarted. No in-flight recovery.

## Usage

```bash
cargo test
cargo check
cargo run --bin make-fixture -- --out /tmp/opencode/potok-model.bin
cargo run --release --bin bench -- --model /tmp/opencode/potok-model.bin --frames 2000
cargo run --release --bin infer -- --model /opt/potok-ai/model.bin --device /dev/ttyS0
```

`sudo ./scripts/install.sh /path/to/model.bin` installs the binary and
systemd unit under `/opt/potok-ai/`. See `BUILD_LOG.md` for build decisions.

## Performance notes

- All weights are dequantized once at load; the hot loop touches only dense
  f32 rows. All buffers are preallocated in `Runtime`.
- The five blocks are fused into one pass over timesteps; `norm2` and the
  TSCT `up` vector are reused per timestep instead of being stored for the
  whole window.
- `src/simd.rs` uses ARM NEON `dot` on aarch64 with a runtime feature probe
  and an ordered scalar fallback elsewhere. `src/baseline.rs` is the naive
  control for the legacy POTM layout only; `cargo run --release --bin bench -- --model M --compare` prints
  p50/p99 in microseconds, memory footprint, both implementations' latencies
  and their output difference. `tests/optimized_test.rs` requires bit-identical
  output on scalar targets and bounded error on NEON. MEASURED ON Pi NOT YET
  DONE — see `BUILD_LOG.md`.
