# BUILD_LOG — protocol/

Date: 2026-11-22. Scope: `protocol/` only. No GPU jobs, no git push.

## Commands and results

| command | result |
|---|---|
| `cargo run -p potok-generator` | OK — wrote `generated/protocol.rs`, `generated/protocol.h`, `generated/protocol.py` |
| `cargo check` (from `protocol/`) | OK — both workspace crates, no warnings |
| `cargo test` (from `protocol/`) | OK — 7 generator tests + 1 lib test + 14 integration tests + 0 doc tests, 0 failed |
| `cargo test -p potok-protocol` | OK — same 15 protocol tests |
| `cargo run -p potok-generator -- --check` | OK — `protocol/generated: up to date` |
| `sha256sum generated/*` → regenerate → `sha256sum` → `diff` | OK — byte-identical (deterministic output) |
| `cc -std=c11 -Wall -Wextra -Werror -Igenerated target/c_verify.c -o target/c_verify && ./target/c_verify` | OK — `crc123456789=29B1`, `C OK` |
| Python round-trip/error checks against `generated/protocol.py` | OK — `PY OK` |
| C vs Python encoded-frame hex comparison | OK — both produce `4d50...c67f` (request) and `5052...c549` (reply) |

## Notes on fixes made during verification

- Generator emitted duplicate C helpers for `u16`/`i16` (same prefix). Split
  signed/unsigned prefixes; C header now compiles with `-Werror`.
- Scaled-angle helpers were emitted as `..._G_PITCH_...`; now emitted as
  `..._g_pitch_...` in C and Python.
- `golden_wire_bytes_match_c_and_python` pins the exact request/reply bytes so
  the three languages cannot silently diverge.

## Files

- `uart.yaml` — frames, CRC, phase enum, flags, versions.
- `generator/` — serde/serde_yaml strict schema, validation, 3 emitters.
- `potok-protocol/` — dependency-free Rust library with CRC, codec,
  version/phase checks, seq tracking, flag and angle helpers.
- `generated/` — committed artifacts; regenerate after any YAML change.

## Hot-path optimization (follow-up)

Method: `cargo run --release -p potok-protocol --example hotpath`, 1 000 000
iterations, 5 repetitions, minimum duration reported, `black_box` on inputs
and outputs. MB/s = wire bytes per second (CRC: 54 B; one frame: size bytes;
encode+decode: twice the frame size). Other agents' processes kept the box at
load ~5 during measurement, so the final run also prints an in-process bitwise
CRC reference; that pair is immune to cross-run noise.

| case | before (ns / MB/s) | after (ns / MB/s) | gain |
|---|---|---|---|
| crc16 over 54 B | 180.9 / 298.5 | 87.2 / 618.9 | 2.07× |
| request encode | 159.2 / 351.7 | 75.3 / 743.9 | 2.11× |
| request decode | 161.6 / 346.5 | 81.2 / 690.0 | 1.99× |
| reply encode | 16.9 / 830.5 | 10.0 / 1406.2 | 1.69× |
| reply decode | 18.4 / 759.6 | 9.2 / 1518.2 | 2.00× |
| **request encode+decode+CRC** | **328.1 / 341.3** | **160.0 / 700.2** | **2.05×** |
| reply encode+decode+CRC | 35.2 / 796.2 | 23.3 / 1199.3 | 1.51× |

In the final run, the bitwise CRC reference measured 150.0 ns / 360.1 MB/s
against slicing-by-4's 87.2 ns / 618.9 MB/s (1.72× in the same process).

What changed:

- Compile-time `CRC_TABLE` (from the YAML polynomial) plus `CRC_TABLE2`
  positional tables. `crc16` consumes four bytes per loop with two dependent
  table loads (slicing-by-4) and finishes the trailing 1-3 bytes byte-wise.
  `crc_table_matches_bitwise_reference` compares all lengths 0..256 against
  the old bitwise algorithm.
- Generated Rust codec now writes/reads bytes directly; the `try_into().unwrap()`
  in every `get_*` and the slice `copy_from_slice` in every `put_*` are gone.
- Added `RequestView<'a>` / `ReplyView<'a>`: validated zero-copy views over the
  input slice, with accessors and `to_owned()`. `decode_*` keeps the direct
  owned path; there is no allocation in encode/decode.
- Golden proof: `committed_golden_fixtures_reencode_byte_identical` decodes
  the committed request/reply hex fixtures (owned and view), re-encodes and
  asserts byte equality; `golden_wire_bytes_match_c_and_python` still pins the
  same bytes across all three languages.

SIMD: the measured bulk win is slicing-by-4 (two independent table loads per
byte pair). Explicit SIMD intrinsics were not added: a 16-bit CRC over 54 bytes
would need gathers plus a polynomial combine, and no such candidate was
measured to beat the table path. Skipped deliberately, not assumed.

Verification after the change: `cargo check` clean; `cargo test` 25/25 green
(the original 22 plus 3 new); `--check` clean; regeneration byte-identical;
C `-Werror` build and Python round trip both reproduce the committed golden
request bytes.

## Phase vocabulary fix (integration-review follow-up)

The enum names in `uart.yaml` were misleading (Idle/Armed/Flight/Landing/Fault)
relative to the documented flight phases. Renamed to `BURN=0`, `COAST=1`,
`APOGEE=2`, `DESCENT=3`, `LANDED=4`. Numeric codes are unchanged, so the wire
format and the committed golden frames are unchanged.

- `uart.yaml`: five variant names replaced, values kept 0..4.
- `generator/src/model.rs`: added a domain invariant — an enum named `Phase`
  must have exactly these five variants with these codes; any other name/code
  set fails validation loudly. Added test `phase_uses_documented_names_and_codes`
  that also proves the old `Idle=0` vocabulary is rejected.
- Updated all Rust, C and Python call sites (`Phase::APOGEE`,
  `POTOK_PHASE_APOGEE`, `proto.Phase.APOGEE`) and README.
- Regenerated all three artifacts from the YAML.

| command | result |
|---|---|
| `cargo run -p potok-generator` | OK — `.rs`/`.h`/`.py` regenerated with new names |
| `cargo check` | OK — no warnings |
| `cargo test` | OK — 8 generator + 3 lib + 15 integration = 26 passed, 0 failed |
| `cargo run -p potok-generator -- --check` | OK — `protocol/generated: up to date` |
| `golden_wire_bytes_match_c_and_python` + `committed_golden_fixtures_reencode_byte_identical` | OK — golden hex unchanged; rename is wire-neutral |

Only `protocol/` files were touched.

## Lowlat profile (SPEC_PROTOCOL_LOWLAT.md, commit 9caeb1a)

Added as a **second profile**; compat 56/14 frames, their generated names,
golden bytes and tests are unchanged. `uart.yaml` now has
`protocol.profiles.compat` and `protocol.profiles.lowlat`, each with its own
versions and frames. Shared CRC, Phase enum and flag bits stay at protocol
level. Generator emits compat names exactly as before and prefixes lowlat
names (`LOWLAT_REQUEST_SIZE`, `LowlatRequestFrame`, `PotokLowlatRequest`,
`proto.LowlatRequest`, `potok_lowlat_request_encode`).

Lowlat request: 42 B, magic `0x4C54`, 14 int16 features with explicit scales
in YAML (generated as `LOWLAT_REQUEST_FEATURES_SCALES` / `_NAMES`).
Lowlat reply: 16 B, magic `0x4C52`, `compute_us` u8 in 100 µs units.

Two spec ambiguities resolved explicitly:

- The spec's reply field list sums to 14 B while it states 16 B. Two
  `reserved` bytes at offset 12 make the stated size exact; validation pins
  `lowlat_reply.size == 16`.
- The spec fixes only the request magic; `0x4C52` was chosen for the reply and
  is declared in YAML.

Validation additions: profiles `compat`/`lowlat` required; frame-name prefixes
unique across profiles; feature scale/name arrays must match the int16 array
length, be finite and positive, and use unique identifier names; lowlat must
have 14 scaled features and a 16-byte reply with `compute_us` u8.

| command | result |
|---|---|
| `cargo run -p potok-generator` | OK — all three artifacts regenerated |
| `cargo check` | OK — no warnings |
| `cargo test` | OK — 9 generator + 3 lib + 9 lowlat + 15 compat = 36 passed, 0 failed |
| `cargo run -p potok-generator -- --check` | OK — `protocol/generated: up to date` |
| regenerate → `sha256sum generated/*` → `diff` | OK — byte-identical output |
| `cc -std=c11 -Wall -Wextra -Werror` C harness, both profiles | OK — reproduces both golden frames; compat hex unchanged |
| Python `protocol.py` harness, both profiles | OK — reproduces both golden frames |

Golden frames (also pinned in `lowlat_tests.rs`, decoded and re-encoded
byte-identically, and reproduced by C and Python):

- request `544c34127bf15365020112000100feff03000400fbff06000700f8ff09000a000b00f4ff0d000e000a2b`
- reply `524c34127d0006ffc801012a0000fd1b`

Benchmark (`cargo run --release -p potok-protocol --example hotpath`, 1M
iterations, 5 reps, minimum; MB/s counts both encode and decode wire bytes):

| case | ns/frame | MB/s |
|---|---:|---:|
| compat request encode+decode+CRC | 155.4 | 720.6 |
| compat reply encode+decode+CRC | 22.8 | 1228.0 |
| lowlat request encode | 52.9 | 793.5 |
| lowlat request decode | 69.5 | 604.3 |
| lowlat reply encode | 12.7 | 1262.1 |
| lowlat reply decode | 17.6 | 907.6 |
| **lowlat request encode+decode+CRC** | **128.5** | **653.7** |
| lowlat reply encode+decode+CRC | 28.7 | 1116.9 |

Only `protocol/` files were touched.
