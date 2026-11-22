# Night task EDGE01 — lowlat performance hard gates

**Status: done.** Local commit only, nothing pushed. No subagents. No source edits, no GPU.

## Scope

Created exactly one file in `/home/sehaxe/potok-m-edge`:

- `tests/lowlat_perf_hard_tests.rs` — SPDX `GPL-3.0-or-later`, 7 tests.

Nothing else was staged or modified.

## What the file pins

1. **Pure percentile statistic.** Private `percentile(sorted, q)`, linear
   interpolation (numpy `method="linear"` convention; rank `q·(n−1)`), pinned on
   `1..=9` (`p50 = 5.0`, `p99 = 8.92`) and on a tied array (`p50 = 4.5`,
   `p99 = 8.86`). Asserted order-independent, bit-identical on repeated calls,
   and loud on empty input / out-of-range `q` (three `should_panic` gates).

2. **Deterministic full-cycle microbench, model-vs-overhead split.** Fixed-seed
   lowlat request bytes (pure index function, no RNG), 16 warmup + 128 timed
   frames: `decode_request` (magic, CRC, version, phase, dequantize) →
   `Runtime::on_request` → `encode_reply` (quantize, CRC), timed from four
   timestamps. Per-stage p50/p99 is reported for decode, model, encode, total
   and overhead (decode+encode). Two runs over the same workload must produce
   the same pinned output checksum `0x9bfa` in both debug and release.
   Assertions are structural: sample count, finiteness, `0 < p50 <= p99`,
   bit-identical recomputation, quantile dominance of total over each stage and
   of overhead over decode/encode, and a **documented 2× tolerance**
   (`SPLIT_TOLERANCE_FACTOR`) on reconstructing total from model+overhead.
   No machine-specific microsecond threshold is asserted anywhere.

3. **CRC microbench.** 32 `crc16` calls per sample over the lowlat
   CRC-covered prefix, with `black_box` barriers on input and accumulator.
   Without the barriers release mode folded the batch into one call (3 ns per
   reported call); with them it measures ~0.074 µs/call. Sum of accumulators
   pinned at `0x3ff180` across debug and release. A separate gate flips one bit
   and checks both CRC change and codec rejection.

## Verification

```
cd /home/sehaxe/potok-m-edge
rustfmt --edition 2021 --check tests/lowlat_perf_hard_tests.rs -> OK
cargo test --test lowlat_perf_hard_tests                       -> 7 passed (debug)
cargo test --release --test lowlat_perf_hard_tests             -> 7 passed
cargo test --release --all-targets                             -> 80 passed, 0 failed
cargo clippy --all-targets                                     -> no warnings in the new file
                                                                 (pre-existing warnings in src/)
```

Measured on this x86_64 host (informational only, not asserted in tests):
release full cycle p50/p99 ≈ 170/236 µs, model ≈ 169/235 µs,
decode ≈ 0.26/0.31 µs, encode ≈ 0.21/0.28 µs.

## Out-of-scope observation

`cargo test` (debug, whole suite) fails one pre-existing wall-clock gate in
`tests/real_models_spec_test.rs:142`: debug forced-check p99 11.21 ms against a
10 ms budget (release passes it). That file was not touched; the release gate
above is green.

## Notes / gaps

- No Raspberry Pi here, so no target p99 claim — the test only certifies that
  the measurement instrument is deterministic and structurally sound.
- The percentile convention is the interpolated one from the AI04 night task,
  not the nearest-rank rule in `src/bin/bench.rs`; the difference is documented
  at the function.
