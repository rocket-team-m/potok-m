# SIM08 — trajectory v3 hard tests (night task)

**Date:** 2026-11-23
**Repo:** `/home/sehaxe/potok-m/sim` (git root `/home/sehaxe/potok-m`)
**Scope:** created exactly one file, `sim/core/tests/trajectory_v3_hard_tests.rs` (454 lines after `rustfmt`). No source edits. No subagents. No push.

## What was delivered

A single integration-test file with an SPDX `GPL-3.0-or-later` header. It uses only the public `potok_sim` API, builds 50-tick synthetic episodes with a distinct finite sentinel per field index (plus `-0.0` to test bit patterns), writes fixtures under a per-test temp dir, and asserts on raw little-endian bytes — not on the serializer's own conversion helpers.

| Requirement | Test |
|---|---|
| v1/v2/v3 stride compatibility | `header_stride_and_file_size_are_version_strict` — magic, version, counts, stride field 6/24/26, zeroed 40-byte reserved header, exact file lengths, per-episode u64 index, all three named writers byte-equal to `write_trajectory_with_version`, and v1's six floats a slot-for-slot prefix of v2 and v3 |
| g_accel placement | `g_accel_occupies_float_slots_24_and_25_of_every_v3_tick` — raw f32 at offsets `24*4` / `25*4` per tick equals `g_accel[0]/[1]` by `to_bits()`, `g_theta` stays at 22/23, field-name table pins `g_accel_y`/`g_accel_x`, and the 24-float v2 prefix inside v3 is byte-identical to a v2 write of the same episode |
| round-trip | `v3_round_trip_is_bit_exact_across_all_26_fields` — all 26 fields bit-identical on every tick, struct equality, array conversion pair, `-0.0` sign preserved |
| gapped fields | `older_versions_gap_new_fields_explicitly_and_drop_them_on_write` — v1 read fills all v2/v3-only fields with explicit defaults (`gust_start_s = -1`, rest zero); v2 read leaves `g_accel = [0,0]`; struct-level `from_array`/`from_truth_array` agree; v1/v2 bytes do not depend on dropped fields while v2 bytes do depend on `g_theta` and v3 bytes on `g_accel` |
| header corruption | `bad_magic_version_or_stride_fail_loudly` — bad magic / unsupported version / version-stride mismatch / truncated record all fail loudly for each version |
| hash stability | `sidecar_sha256_is_stable_and_binds_v3_only_fields` — identical regeneration gives identical sha256 and matches `file_record`; moving `g_accel_y` or `g_accel_x` at first/last tick changes the v3 hash but not the v2 hash; the three versions hash distinctly |

## Verification (evidence)

- `cargo test -p potok-sim --test trajectory_v3_hard_tests` → **6 passed, 0 failed** (initial run: one assertion failed — my fixture zeroed `g_theta` as well as `g_accel`, but `g_theta` is a real v2 field; split the fixture into `no_accel` and `no_v3_or_v2_cmd` variants, re-ran green).
- `cargo test -p potok-sim --test trajectory_v3_hard_tests --test dataset_tests` → **16 passed, 0 failed** across 2 binaries.
- All 14 tracked `potok-sim` integration-test targets plus the new one, run explicitly by target name → **116 passed, 0 failed** (including `quality_v3_tests`, `manifest_hardening_tests`, `dataset_tests`, `wind_scenario_tests`).
- `rustfmt --edition 2021 --check sim/core/tests/trajectory_v3_hard_tests.rs` → clean.

**Unrelated, not touched:** bare `cargo test -p potok-sim` cannot compile two *untracked* WIP files owned by other agents — `core/tests/sensor_realism_tests.rs` (E0277 `f32`/`f64` compares at 320, 340, 341) and `core/tests/control_seam_hard_tests.rs` (E0061 `fin_accel_command` missing `&Vehicle` at 244, 257, 266). Selecting the tracked targets avoids them; per the "create only this file / no source edits" constraint they were left alone.

## Git

Local commit `d851cb5` — `test(sim): harden trajectory v3 sidecar contract (SIM08)`. No push.
