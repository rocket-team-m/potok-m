# SIM04 — manifest hardening tests (night task)

**Date:** 2026-11-23
**Repo:** `/home/sehaxe/potok-m/sim` (git root `/home/sehaxe/potok-m`)
**Scope:** created exactly one file, `sim/core/tests/manifest_hardening_tests.rs`. No source edits. No subagents. No push.

## What was delivered

A single integration-test file with an SPDX `GPL-3.0-or-later` header, 527 lines after `rustfmt`. It exercises the public API of `potok_sim` only and writes nothing outside a per-test temp dir.

| Requirement | Test |
|---|---|
| manifest schema v1 | `manifest_schema_is_v1_with_exactly_the_expected_shape` — pins `SCHEMA_VERSION == 1`, `QUALITY_SCHEMA_VERSION == 1`, the exact 13 top-level keys plus `bounds`/`files` sub-keys, and that a foreign version stays visible instead of being rewritten |
| config hash | `config_hash_binds_the_canonical_toml_and_moves_with_any_field` — hash equals sha256 of canonical TOML, 64 lowercase hex, stable across clones, changes on seed / wind / `run.dt` edits |
| seed | `manifest_records_the_seed_and_a_new_seed_moves_the_bytes` — recorded seed matches config; a new seed moves config hash *and* validation bytes; same inputs regenerate byte-identical manifests |
| split disjointness | `split_disjointness_is_a_loud_gate` — accepts disjoint train/held-out sets (`overlap=[]`), rejects self-compare, a one-episode contamination (names the offending scenario id), and an empty split |
| sha256 | `sha256_detects_single_byte_corruption_in_every_manifest_file` — known `"abc"` vector, per-file digest/length match, one flipped byte in each of the two files changes the digest while length stays equal, then restores |
| bounds | `bounds_cover_every_exported_channel_and_stay_finite` — names and order match config features / `TARGET_NAMES` / `TRUTH_V3_TICK_NAMES`, every range finite with `min <= max`, at least one non-degenerate range, JSON round trip |
| no timestamps | `manifest_bytes_carry_no_timestamps_or_process_artifacts` — forbidden token scan on serialized JSON, two independent writes byte-equal, `check_determinism` passes on manifest + both data files |
| loud corruption | `corruption_fails_loudly_at_every_boundary` — malformed JSON (string and file) names `manifest parse`; NaN feature, out-of-range target, truncated truth, too-few episodes, and byte mismatch each fail with a named gate |

## Verification (evidence)

- `cargo test -p potok-sim --test manifest_hardening_tests` → **8 passed, 0 failed** (4.02 s, after formatting).
- `rustfmt --edition 2021 --check sim/core/tests/manifest_hardening_tests.rs` → clean.
- All 12 tracked `potok-sim` test targets run individually → **all PASS**, including the new one.
- `cargo test -p potok-sim --lib` → 0 tests, ok.
- `cargo test -p potok-cli -p potok-or-parity` → 9 + 4 passed, 0 failed.

**Unrelated, not touched:** a bare `cargo test` on the whole workspace fails to compile `core/tests/wind_scenario_tests.rs` (E0689 ambiguous float, E0277 integer×float at lines 97, 98, 158). That file is **untracked and is another agent's work in progress**; per the "no source edits / create only this file" constraint it was left alone. All tracked targets plus the new file are green.

## Git

Local commit `c23be01` — `test(sim): harden dataset manifest contract (SIM04)`. No push.
