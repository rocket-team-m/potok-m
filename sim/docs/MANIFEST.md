<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
# Dataset manifest and provenance binding

`manifest.json` is the v1 dataset manifest: schema version, config hash, seed,
split, scenario ids, counts, channel names, observed bounds and the sha256/size
of every data file. That wire shape is shared by the generator, the CLI and the
firmware dataset writer, so it stays frozen.

`manifest.provenance.json` is the companion document added by SIMHASH. It binds
a dataset to the exact stack that produced it:

| field | meaning |
|---|---|
| `schema_version` | provenance wire version (currently 1) |
| `manifest_hash` | sha256 of the canonical `manifest.json` text, including its trailing newline |
| `code_hash` | lowercase 40-char `git rev-parse HEAD` of the generating code |
| `protocol_schema_hash` | sha256 of `protocol/uart.yaml`, the canonical wire schema |
| `measurement_hashes` | `filename -> sha256` for thrust/Allan/servo data files found in the measurement directory; empty when none are present |
| `golden_episode_hash` | sha256 over the pinned seed-1337 episode-0 file records |

The path is derived, not configured: `manifest.json` → `manifest.provenance.json`
(`provenance_path`).

## Determinism

- Every value is computed from file bytes, the config or an explicit input. No
  timestamps, process ids, environment variables or global RNG enter the JSON.
- `measurement_hashes` is a `BTreeMap`, so filename order is independent of
  filesystem enumeration order; a `servo.rs` source file is not a measurement
  (only `.csv/.json/.toml/.txt/.dat` names containing `thrust`, `allan` or
  `servo` are bound).
- Golden encoding: records sorted by name, each as `name NUL byte_count NUL
  sha256 LF`. NUL separators make the encoding unambiguous.
- Serialization is `serde_json::to_string_pretty` plus one trailing newline.
  Two runs with the same inputs produce byte-identical files.

## Verification is LOUD

`verify_provenance_binding`, `verify_code_hash`, `verify_protocol_schema`,
`verify_measurement_inputs` and `verify_golden_episode` each return an error
naming the field, the recorded value, the observed value and the escape
("regenerate the dataset at the recorded inputs or update the binding"). No
field is clamped, defaulted or silently skipped. Absent measurement files are
recorded as an empty map — that is the documented "when present" case, not a
fallback; if the dataset later gains a measurement file, verification fails.

`quality::check_provenance` runs the four source-bound gates and returns one
`GateCheck` per gate. `quality::check_golden_episode` adds the golden pin. A
single flipped byte in a data file, a new commit, a protocol edit or a changed
RNG draw order breaks the corresponding gate.

## Compatibility

The binding accepts manifest schema versions 1, 2 and 3 (`v1` legacy
validation, `v2` privileged truth, `v3` per-tick truth). A foreign version is
rejected loudly rather than silently rewritten. New provenance fields are
`#[serde(default)]`, so a shorter older document still parses and is then
rejected if a required bound hash is missing — it is never treated as a pass.

The golden pin is the sha256 of seed-1337 episode-0 `validation.bin` plus its
`validation.trajectory` sidecar. It is asserted in `core/tests/manifest_hash_tests.rs`
against a frozen eight-feature export list, so an unrelated edit to
`config/default.toml` does not invalidate it. Changing the RNG draw order or
the exported bytes requires recording a new pin deliberately.

## Caller hand-off

`build_provenance` takes already-computed hashes; `git_head_hash`,
`protocol_schema_hash`, `measurement_hashes` and `golden_episode_hash` compute
them. `write_provenance` / `read_provenance` handle the file. The generator and
CLI writers are not modified by this task — adopting the companion document in
those call sites is the follow-up that makes every A/B number auditable.
