# SIM3D — deterministic 3D playback export from POTOKTRJ v3 (2026-11-23)

Scope: create only `sim/core/src/playback3d.rs`, `sim/cli/src/playback3d.rs`,
`sim/core/tests/playback3d_tests.rs`. No GUI, no physics changes, no subagents,
no push. License: `SPDX-License-Identifier: GPL-3.0-or-later` in all three.

## Delivered

- **`core/src/playback3d.rs`** — pure exporter. Reads a POTOKTRJ v3 sidecar via
  the existing `trajectory::read_trajectory`, validates all 26 f32 fields as
  finite, and emits a text **glTF 2.0** document: `asset.version = "2.0"`, one
  root node per flown tick with glTF `translation` `[x,y,z]` and `rotation`
  `[x,y,z,w]`, plus `extras.playback3d` (schema, 50 Hz tick rate, coordinate
  convention, per-episode markers) and per-tick `extras` (raw v3 pos/quat/vel/
  omega/mass, derived phase + code, rocket body axis, gust/shear fields).
  Public entry points: `document`, `to_json`, `write_file`, `export_file`,
  `playback3d_path`, `ExportInfo`, and all schema structs.
- **`cli/src/playback3d.rs`** — std-only CLI adapter (`parse`, `run_with`,
  `main_with`, `format_summary`) with an injected exporter, so it compiles both
  in `potok-cli` and in the core test harness without touching a manifest.
  Usage: `playback3d --trajectory <POTOKTRJ-v3> --out <FILE.json>`.
- **`core/tests/playback3d_tests.rs`** — 8 hard gates: deterministic bytes and
  glTF envelope, sim Z-up -> glTF Y-up transform and body axis, phase sequence
  from the mass plateau, gust/shear markers, strict v3 + loud failure on
  non-finite/zero-quaternion/empty input, byte-stable file round trip, CLI
  happy path and bad-flag refusal.

Design decisions:
- Coordinates: `(x,y,z)_sim -> (x,z,-y)_gltf`; node rotation is `q_map ⊗ q_sim`
  with `q_map = -90° about X`, so a node's local +Z stays the rocket axis. Raw
  sim-frame truth remains in `extras` for binary-sidecar consumers.
- Phase is reconstructed from flown truth (mass plateau = burnout), mirroring
  `dynamics::phase_at` and the workbench's `trajectory_overlay::phase_at`.
  Deployment is never invented; the sidecar has no such field.
- Gust marker uses the first tick declaring `gust_start_s >= 0`, its amplitude
  from that declaring tick, at `round(start × 50)`; shear marker is the first
  upward altitude crossing. Both are emitted once per episode and sorted.
- No clock, RNG, hash-map iteration or wall-clock metadata enters the artifact;
  serde_json pretty output is byte-identical across runs.

## Consumer note (honest gap)

The document is valid glTF 2.0 and any standard glTF loader reads its per-tick
node transforms. RocketCraft's current `rocketcraft-io` glTF adapter is
**triangle-surface-only** (`crates/io/src/gltf.rs`): it requires `meshes` with
POSITION accessors and drops `extras`, so it will not read this artifact as-is.
A thin reader in the workbench is needed to map `extras.playback3d` onto the
`trajectory_overlay` / viewport data model; that crate is outside this task's
scope and was not touched. No claim that the existing adapter consumes it.

## Verification

- `cargo test -p potok-sim --test playback3d_tests` — **8 passed, 0 failed**.
- `cargo test -p potok-sim --lib --test playback3d_tests --test trajectory_v3_hard_tests` —
  **14 passed, 0 failed** across the three targets.
- `rustfmt --edition 2021 --check` on all three files — clean.
- Real-data smoke (temporary test, removed after the run): exported
  `scripts/ab_control_v3/runs/seed1/baseline/validation.trajectory`
  (v3, 20 episodes × 1500 ticks) — `ExportInfo { episodes: 20, ticks: 30000,
  markers: 40, bytes: 38064383 }`, no non-finite rejection, ~5.8 s debug build.

Pre-existing failures elsewhere in the shared checkout, **not caused by this
task** (the three new files are not referenced by any manifest yet):
- `cargo test -p potok-sim` stops at `firmware_cosim_tests.rs:140`
  (`guard_decision_order_latch_and_pass_through`: `Some(SensorInvalid)` vs
  `Some(Fault)`). That test file is untracked, written by another night agent.
- `cargo test -p potok-cli` — 8 passed, 1 failed:
  `cli_generate_dataset_is_byte_identical_on_regeneration` at
  `cli/tests/cli_tests.rs:276` (train/validation.bin not byte-identical across
  seeds). Unrelated to export code; other agents have uncommitted edits in
  `sim/core/src/atmosphere.rs`, `sensor.rs`, `lib.rs`.

## Wiring left to the parent session

`core/src/lib.rs`: `pub mod playback3d;` (optionally re-export the schema types).
`cli/src/main.rs`: `mod playback3d;` plus a subcommand branch:

```rust
playback3d::main_with(|t, o| {
    potok_sim::playback3d::export_file(t, o)
        .map(|i| playback3d::Summary {
            episodes: i.episodes, ticks: i.ticks, markers: i.markers, bytes: i.bytes,
        })
        .map_err(|e| e.to_string())
});
```

Commit: local only, no push.
