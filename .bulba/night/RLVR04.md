# RLVR04 — firmware-adapter command stream dataset

**Date:** 2026-11-23
**Repo:** `/home/sehaxe/potok-m/sim` (git root `/home/sehaxe/potok-m`)
**Status:** done, committed locally, not pushed.
**Source commit:** `d4c3e762a1ddacbe83f55eb472eea123b2b42539` —
`feat(sim): firmware-adapter command stream dataset with per-tick provenance`.
No subagents were used.

## Deliverable

Exactly three new source files, plus this report. No file outside the assigned
paths was edited by this task; every new file carries
`SPDX-License-Identifier: GPL-3.0-or-later` and the workspace already declares
GPL-3.0-or-later.

| File | Role |
|---|---|
| `sim/core/src/firmware_dataset.rs` | Generation, provenance writer/reader, split manifest, reference arm. |
| `sim/core/tests/firmware_dataset_tests.rs` | Six gates; includes the CLI module by path so it is compiled and exercised. |
| `sim/cli/src/firmware_dataset.rs` | clap-free CLI helper: args, built-in adapter selection, quality gates. |

## Interface: RLVR01's trait is reused, not duplicated

RLVR01 is present in the shared tree as `sim/core/src/firmware_adapter.rs`
(`FirmwareAdapter::step(&mut self, &Frame) -> Command`, `reset(seed)`,
`on_phase(phase)`; `Command { ay, ax, ai_healthy }` in world `[ay, ax]` order).
`core/src/lib.rs` already declares `pub mod firmware_adapter` and, by the
integration task, `pub mod firmware_dataset`. This task therefore uses that
trait directly: no local look-alike trait was added.

Because RLVR01's trait has no name/source accessor, the caller supplies a
`CommandProvenanceMeta { adapter, source }`. `CommandProvider::Firmware` owns a
`Box<dyn FirmwareAdapter>`; `CommandProvider::OfflineReference` is the
byte-for-byte privileged control arm. The adapter is reset per episode with
`episode_seed(cfg.run.seed, index)` and receives `on_phase(ctx.phase)` before
every `step(ctx.frame)`. World `[ay, ax]` becomes the fin command `[ay, -ax]`,
validated by the existing `validate_command` guard; out-of-range/finite
violations are LOUD, never clamped.

## Output contract

Per split directory:

- `validation.bin` — standard validation **v3** (six targets; `g_accel_y`,
  `g_accel_x` authoritative), written by the standard writer. The four legacy
  diagnostic columns keep the simulator's values; only `g_accel` is replaced
  by the adapter's command. Truth metadata follows the caller's config, as in
  the standard generator.
- `validation.trajectory` — standard v3 privileged-truth sidecar (stride 26),
  with per-tick `g_accel` overwritten by the same command.
- `validation.command_provenance` — additive deterministic binary sidecar:
  64-byte header (`POTOKCMD`, version 1, counts, record size 88), then per
  episode a u64 index and frame x 88-byte little-endian records:
  `frame u32, source u8, ai_healthy u8, reserved u16, adapter name 64 B NUL-padded,
  fin_roll/fin_pitch/world_ay/world_ax f32`. The reader is strict: wrong magic,
  version, stride, source tag, frame index, name encoding or unbounded command
  is a LOUD `SimError::Dataset`.
- `manifest.json` — standard `DatasetManifest` (schema 1) hashing all three
  files, with the provenance sidecar as the third `FileRecord`. No timestamps,
  pids or global RNG; regeneration is byte-identical.

The CLI helper selects `OfflineReference`, RLVR01's `MockAdapter` or a
zero-command adapter; a caller-supplied `CommandProvider` plugs in through
`run(args, factory)`. Both splits are checked with `check_dataset` and
`check_split_disjoint` before the helper returns.

## Verification

- `cargo test --offline -p potok-sim --test firmware_dataset_tests`:
  **6 passed, 0 failed, 0 warnings**. Gates cover: offline-reference bytes
  equal the oracle's v3 bytes; scripted adapter replaces `g_accel` and changes
  the flown trajectory; provenance round-trip and four loud-corruption paths;
  v3 layout/truth/manifest plus four-file deterministic regeneration; CLI both
  splits with zero/mock/reference/plugged-in adapters; unbounded command and
  overlong adapter name refused loudly.
- Referenced source facts: RLVR01 trait at
  `sim/core/src/firmware_adapter.rs:36-56`, command order at `:16-29`;
  simulator fin seam at `sim/core/src/optimizer.rs:127-137`; v3 writers at
  `sim/core/src/validation.rs:268-370` and `sim/core/src/trajectory.rs:318-377`.

**Full-workspace caveat (honest, concurrent work):** `cargo test --offline
--workspace` reached an unrelated failure in
`core/tests/or_parity_extended_tests.rs::episode0_time_to_apogee_matches_or_fixture`
(`tick 1352 altitude ... inconsistent with apogee ...`). That test,
`config/default.toml`, `core/src/aero.rs` and `core/src/dynamics.rs` are
uncommitted edits by other agents in this shared checkout; the failure does
not execute any code added by this task and was left untouched as required by
the file-scope restriction.

## Integration note

`sim/cli/src/lib.rs` does not yet declare `pub mod firmware_dataset;`; wiring a
clap subcommand to `FirmwareDatasetArgs` belongs to the integration task, same
as the `core/src/lib.rs` declaration that another agent already added. The CLI
module's compile-correctness is nevertheless verified here through the test
crate's `#[path]` include.
