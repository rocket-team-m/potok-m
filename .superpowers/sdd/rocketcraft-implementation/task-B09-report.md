# B09 — Sim quality gate v3

**Status:** done. Local commit only, no push.

## Files

- `sim/core/tests/quality_v3_tests.rs` (new) — 7 tests.
- This report.

No other files touched. Production code (`quality.rs`, `manifest.rs`, `validation.rs`,
`trajectory.rs`, `optimizer.rs`) and the other agents' test files are unchanged.

## Coverage

- **g_accel bounds**: column 4/5 limits equal `±MAX_COMMAND_MPS2` while legacy
  columns keep their own bounds; generated v3 targets are finite, in range and
  equal to the per-tick `g_accel` in the trajectory sidecar; boundary values pass;
  out-of-range values on either column (and NaN) fail `target_bounds`/`finite`
  loudly with the split name and column name; out-of-range sidecar truth fails
  `truth_bounds` loudly.
- **Split disjointness**: train/held-out scenario-id sets are disjoint and the
  gate accepts them; identical sets, a set carrying one training id, and an
  empty side all fail loudly with `split_scenarios_disjoint` (the contaminated
  id appears in the message).
- **Manifest hash**: `config_hash == sha256_hex(canonical_config_bytes(cfg))`,
  stable across a config clone, changed by a seed change; each `FileRecord` hash
  equals `sha256_file` on disk; flipping one file byte changes the hash; the
  manifest JSON bytes and their SHA-256 are deterministic and equal the written
  file.
- **Deterministic regeneration**: both splits regenerated twice from the same
  config/seed/index sets; six artifacts (validation, sidecar, manifest × 2
  splits) compared byte-for-byte via `check_determinism`, which reports
  `6 files compared, mismatches=[]`; a deliberately different pair fails and
  names the file; a different seed changes both `config_hash` and the exported
  bytes, so the gate can fail.

Tests write to per-process temp directories under `std::env::temp_dir()` with
unique tags and only read the real embedded default config — no corpus, no GPU.

## Verification

- `cargo test -p potok-sim --test quality_v3_tests` → **7 passed, 0 failed**.
- `cargo test --workspace` → **93 passed, 0 failed, 0 ignored** across 18 binaries
  (includes the new file and the other agents' committed tests).
- `rustfmt --edition 2021 --check core/tests/quality_v3_tests.rs` → clean.
  Whole-workspace `cargo fmt --check` still reports pre-existing diffs in
  `core/src/aero.rs`, `core/tests/aero_card_tests.rs`,
  `core/tests/aero_parity_hard_tests.rs` and `core/tests/residual_tests.rs`;
  those files are owned by other tasks and were not touched here.
- `git diff --check` on the new test file → clean.

## Notes

- The gate is tested as implemented: `check_split_disjoint` compares scenario
  **ids** (the `Scenario::signature` wording in its doc comment is not what the
  function checks). The test asserts id disjointness, not signature equality.
- No physics, config, CLI or production code changes; no binaries, no push.
