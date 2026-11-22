# Potok-M UART protocol

Single source of truth: [`uart.yaml`](uart.yaml). Everything else in
[`generated/`](generated/) is derived from it.

## Wire format

Two profiles share the same CRC, phase enum and flag bits. All frames are
little-endian and fixed size. CRC is CRC-16/CCITT-FALSE (poly `0x1021`,
init `0xFFFF`, no reflection, no final XOR) over every byte before the
trailing 2-byte CRC field. `compat` is the default and is unchanged by the
addition of `lowlat`.

### compat profile

#### Request (AI → STM32), 56 bytes, magic `0x504D`

| offset | field | type | note |
|---:|---|---|---|
| 0 | magic | u16 | `0x504D` |
| 2 | seq | u16 | wraps at `u16::MAX` |
| 4 | time_us | u32 | monotonic microseconds |
| 8 | phase | u8 | enum, 0..=4 |
| 9 | version | u8 | expected `1` |
| 10 | flags | u8 | bitfield |
| 11 | reserved | u8 | `0` |
| 12 | gyro | f32[3] | x, y, z |
| 24 | accel | f32[3] | x, y, z |
| 36 | mag | f32[3] | x, y, z |
| 48 | height | f32 | metres |
| 52 | pitot_dp | i16 | differential pressure |
| 54 | crc16 | u16 | over bytes `0..54` |

`Phase`: `0` BURN, `1` COAST, `2` APOGEE, `3` DESCENT, `4` LANDED.
The numeric codes are the wire contract and are unchanged by the rename;
`uart.yaml` validation rejects any other phase vocabulary.

Flags (bit index): `0` GPS_FIX, `1` IMU_OK, `2` BARO_OK, `3` PITOT_OK,
`4` AI_READY, `5` CALIBRATED, `6` MOTOR_ARMED, `7` FAULT.

#### Reply (STM32 → AI), 14 bytes, magic `0x5250`

| offset | field | type | note |
|---:|---|---|---|
| 0 | magic | u16 | `0x5250` |
| 2 | seq | u16 | echoes request seq |
| 4 | g_pitch | i16 | × 0.001 rad |
| 6 | g_yaw | i16 | × 0.001 rad |
| 8 | confidence | u8 | 0..255 |
| 9 | ai_healthy | u8 | 0 / 1 |
| 10 | version | u16 | expected `1` |
| 12 | crc16 | u16 | over bytes `0..12` |

### lowlat profile (SPEC_PROTOCOL_LOWLAT.md)

#### Request, 42 bytes, magic `0x4C54`

| offset | field | type | note |
|---:|---|---|---|
| 0 | magic | u16 | `0x4C54` |
| 2 | seq | u16 | wraps at `u16::MAX` |
| 4 | time_us | u32 | monotonic microseconds |
| 8 | phase | u8 | same Phase enum as compat |
| 9 | version | u8 | expected `1` |
| 10 | flags | u8 | same flag bits as compat |
| 11 | reserved | u8 | `0` |
| 12 | features | i16[14] | scales below |
| 40 | crc16 | u16 | over bytes `0..40` |

Feature scales, in wire order (`LOWLAT_REQUEST_FEATURES_SCALES` /
`LOWLAT_REQUEST_FEATURES_NAMES`), are physical units per LSB:
`gyro_x/y/z` 1e-3 rad/s, `accel_x/y/z` 1e-3 m/s², `mag_x/y/z` 1e-4 T,
`height` 1e-2 m, `pressure` 1 Pa, `pitot_dp` 1 Pa, `temperature` 0.1 °C,
`battery_mv` 1 mV. `quantize_i16(value, scale)` saturates at `i16::MIN/MAX`.

#### Reply, 16 bytes, magic `0x4C52`

| offset | field | type | note |
|---:|---|---|---|
| 0 | magic | u16 | `0x4C52` |
| 2 | seq | u16 | echoes request seq |
| 4 | g_pitch | i16 | × 0.001 rad |
| 6 | g_yaw | i16 | × 0.001 rad |
| 8 | confidence | u8 | 0..255 |
| 9 | ai_healthy | u8 | 0 / 1 |
| 10 | version | u8 | expected `1` |
| 11 | compute_us | u8 | units of 100 µs |
| 12 | reserved | u8[2] | `0`, see note |
| 14 | crc16 | u16 | over bytes `0..14` |

The spec states a 16-byte reply but its field list sums to 14 bytes; the two
`reserved` bytes make the stated size exact. The reply magic `0x4C52` is an
implementation choice — the spec only fixes the request magic. Both are
declared explicitly in `uart.yaml` and validated.

## Regenerate

From this directory (Rust 1.100+, network needed once for `serde_yaml`):

```sh
cargo run -p potok-generator
```

This rewrites `generated/protocol.rs`, `generated/protocol.h`,
`generated/protocol.py`. Output is deterministic (no timestamps); generated
files are committed so consumers do not need the generator.

Check that committed files match the YAML without writing anything:

```sh
cargo run -p potok-generator -- --check
```

The generator uses `serde` with `deny_unknown_fields` on every schema node, so
a misspelled or extra key is a hard error. It also validates frame sizes
against the sum of field sizes, enum/flag references, CRC placement, expected
versions and the magic constants.

## Use from Rust

`potok-protocol` has no dependencies. It `include!`s the generated types, so
the generated file must exist in `generated/`.

```rust
use potok_protocol::*;

let mut frame = RequestFrame {
    seq: 42,
    time_us: 1_700_000_123,
    phase: Phase::APOGEE,
    version: REQUEST_VERSION,
    flags: flags::IMU_OK | flags::AI_READY,
    reserved: 0,
    gyro: [0.0; 3], accel: [0.0; 3], mag: [0.0; 3],
    height: 0.0, pitot_dp: 0,
};
let mut buf = [0u8; REQUEST_SIZE];
encode_request(&frame, &mut buf)?;          // no heap allocation
let decoded = decode_request(&buf)?;

let mut tracker = SeqTracker::default();
let seq = tracker.next_tx();
frame.seq = seq;
tracker.observe(seq)?;                       // wrap-safe, rejects duplicates
```

`encode_*` writes into a caller-provided slice (larger is fine); `decode_*`
borrows the input and returns a `Result<_, Error>`. `Error` covers
`BufferTooSmall`, `TooShort`, `BadMagic`, `BadCrc`, `BadPhase` and
`VersionMismatch`. Decoding checks magic → CRC → enum → version in that order.

`SeqTracker::observe` accepts forward jumps of at most `window` (default 64)
using wrapping arithmetic; `0x7fff` is the largest valid window.

For receive paths that only read a few fields, skip the owned struct entirely:

```rust
let view = RequestView::new(&buf)?;   // validates magic, CRC, phase, version
let seq = view.seq();                 // reads straight from the input slice
let gyro = view.gyro();
assert_eq!(view.as_bytes().as_ptr(), buf.as_ptr()); // no copy
let owned = view.to_owned();          // only if a RequestFrame is needed
```

`ReplyView` has the same shape. Both are `Copy` and borrow the frame for `'a`.

CRC uses a compile-time table built from the YAML polynomial.
`CRC_TABLE2` holds the two positional tables for slicing-by-4, which cuts the
serial dependency chain to two table loads per four bytes. The byte table
`CRC_TABLE` handles the trailing 1-3 bytes. `crc16()` is bit-identical to the
original bitwise implementation (pinned by `crc_table_matches_bitwise_reference`).

Angle helpers: `reply_g_pitch_rad(raw)`, `reply_g_pitch_from_rad(rad)` and the
`_g_yaw_` equivalents clamp to `i16`.

lowlat uses the same shapes with generated `LOWLAT_*` constants:

```rust
let low = LowlatRequestFrame {
    seq: 42, time_us: 1_700_000_123, phase: Phase::APOGEE,
    version: LOWLAT_REQUEST_VERSION, flags: flags::IMU_OK, reserved: 0,
    features: [0; 14],
};
let mut buf = [0u8; LOWLAT_REQUEST_SIZE];
encode_lowlat_request(&low, &mut buf)?;
let low_view = LowlatRequestView::new(&buf)?;  // zero-copy
let gyro_x = dequantize_i16(low_view.features()[0], LOWLAT_REQUEST_FEATURES_SCALES[0]);
let raw = quantize_i16(1.0, LOWLAT_REQUEST_FEATURES_SCALES[0]); // 1000, saturating
```

## Use from C

Include `generated/protocol.h` (C11, stdlib only). It is header-only with
`static inline` functions and uses explicit byte packing, not `memcpy` of a
struct, so struct padding never reaches the wire.

```c
#include "protocol.h"

uint8_t buf[POTOK_REQUEST_SIZE];
PotokRequest req = {0};
req.seq = 42;
req.phase = POTOK_PHASE_APOGEE;
req.version = POTOK_REQUEST_VERSION;
req.flags = POTOK_FLAG_IMU_OK;

if (potok_request_encode(&req, buf, sizeof buf) != POTOK_OK) { /* ... */ }

PotokRequest in;
int rc = potok_request_decode(buf, sizeof buf, &in);
/* POTOK_OK, POTOK_ERR_BUFFER, POTOK_ERR_MAGIC, POTOK_ERR_CRC,
   POTOK_ERR_PHASE, POTOK_ERR_VERSION */
```

Flags: `potok_flag_has`, `potok_flag_set`, `potok_flag_clear` with
`POTOK_FLAG_*` masks. Scaled angles: `potok_reply_g_pitch_scaled`,
`potok_reply_g_pitch_from_scaled` (and `_g_yaw_`).

lowlat: `PotokLowlatRequest` / `PotokLowlatReply` with
`potok_lowlat_request_encode` / `_decode`, `potok_lowlat_reply_encode` /
`_decode`, constants `POTOK_LOWLAT_REQUEST_SIZE` (42),
`POTOK_LOWLAT_REPLY_SIZE` (16) and the scale arrays
`POTOK_LOWLAT_REQUEST_FEATURES_SCALES` / `_NAMES`.

## Use from Python

Add `generated/` to `sys.path` (or copy `protocol.py` next to your code), then:

```python
import protocol as proto

req = proto.Request(
    seq=42, time_us=1_700_000_123, phase=proto.Phase.APOGEE,
    version=proto.REQUEST_VERSION,
    flags=proto.FLAG_IMU_OK, reserved=0,
    gyro=[0.0, 0.0, 0.0], accel=[0.0, 0.0, 0.0], mag=[0.0, 0.0, 0.0],
    height=0.0, pitot_dp=0,
)
buf = proto.encode_request(req)      # bytes, len == REQUEST_SIZE
decoded = proto.decode_request(buf)

rep = proto.decode_reply(reply_bytes)
pitch_rad = proto.reply_g_pitch_scaled(rep.g_pitch)
```

Decode raises `Truncated`, `BadMagic`, `BadCrc`, `BadPhase` or
`VersionMismatch` (all subclasses of `ProtocolError`).

lowlat: `proto.LowlatRequest(seq=..., time_us=..., phase=proto.Phase.APOGEE,
version=proto.LOWLAT_REQUEST_VERSION, flags=..., reserved=0, features=[0]*14)`
with `proto.encode_lowlat_request` / `proto.decode_lowlat_request`; reply uses
`proto.LowlatReply` with `compute_us` and a 2-byte `reserved` list.

## Benchmark

```sh
cargo run --release -p potok-protocol --example hotpath
```

Optional first argument sets iterations (default 1 000 000). Prints ns/frame
and MB/s of wire bytes for CRC (bitwise reference and slicing-by-4), encode,
decode and encode+decode+CRC for both frames of both profiles. `black_box`
prevents elision; five repetitions per case report the minimum duration.

## Verify

```sh
cargo check
cargo test
```

36 tests: the original 22 plus CRC table vs bitwise reference, zero-copy view
equivalence, committed golden fixtures re-encoding byte-identically, and the
lowlat profile suite (round trips, truncation, magic/CRC/phase/version errors,
flags, int16 saturation, golden bytes and profile independence).
