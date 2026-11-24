# Measurement ingest protocol

**Scope.** `tools/measurement/` is a LOUD ingest harness for system-identification
data. It turns three CSV dialects into the typed contracts that already live in
`core/src/identification.rs` (`ThrustCurve`, `ImuAllanPoints`,
`ServoStepResponse`), each bound to provenance that identifies the exact bytes
parsed. The recipe is `config/measurement.toml`.

**Non-claims.** No real measurement ships with this repository. Every fixture
under `tools/measurement/fixtures/` is labelled `SYNTHETIC` in its header and
in its filename, and `assert_shipped_recipe_is_synthetic()` refuses the shipped
bundle if that label ever changes. A source label is recorded verbatim, never
verified against hardware. This harness does not fit parameters, tune a
trajectory, edit a config by itself, or assert anything about a real motor,
IMU or servo. It is an instrument ready to receive operator data, not a
measurement of it.

## CSV dialects

All three share one grammar:

* `# key: value` lines are metadata. Each key appears at most once; a `# line`
  without a colon is free-form commentary and is ignored.
* One header row whose column names and order must match exactly.
* One numeric data row per line; no quoting, no blanks, no dropped rows. The
  physical line number is carried in every error.

| Kind (`measurement` key) | Required metadata | Exact columns | Typed output |
|---|---|---|---|
| `thrust_stand` | `source` | `t_s,thrust_n` | `ThrustCurve` |
| `imu_allan` | `source`, `channel` = `gyro`\|`accel` | `tau_s,sigma` | `ImuAllanPoints` |
| `servo_step` | `source`, `delay_s`, `rate_per_s`, `deadband` | `t_s,commanded,measured` | `ServoStepResponse` |

The `measurement` key is checked against the parser's own kind, and the recipe
re-checks the recorded kind, label and Allan channel. The servo's identified
parameters are part of the measurement record; fitting them from the trace
would be a second, unrecorded estimator and is deliberately not done here.

## LOUD contract

There is no zero-fill, clamp, drop or best-effort reinterpretation. Every
refusal names the file and the fix:

| Failure | Refusal names |
|---|---|
| Missing file | `NOT FOUND`, resolved path, fix the recipe path or supply the file |
| Missing/empty/duplicate metadata key | key, file, required header form |
| Metadata kind mismatch | declared kind, expected kind, file |
| Source label mismatch | declared label, expected label, file |
| Wrong header | found columns, expected columns, line |
| Short/long row, blank row, no rows | line, column count or the missing-data condition |
| Non-numeric / non-finite token | line, column name, index, raw token, fix |
| Physical violation (duplicate time, negative thrust, non-positive rate, …) | the core contract's message, naming field, value and fix |

Identification rules live in exactly one place: the core constructors. The
CSV layer only checks shape and finiteness, so a second rulebook cannot drift
away from the model's contracts. Out-of-domain interpolation is also LOUD and
inherited from those contracts.

## Provenance

`Provenance` carries:

* `schema_version` — currently 1; a different value is refused.
* `recipe_sha256` — SHA-256 of the exact recipe bytes used.
* `sources` — one `SourceRecord` per file in fixed order: `kind`, resolved
  `path`, verbatim `source_label`, SHA-256 of the exact bytes parsed, and the
  number of data rows consumed.
* `calibration_sha256` — SHA-256 of the canonical typed values. The canonical
  byte string is `potok-measurement/1`, then fixed-order lines for source
  identity, recipe hash, every thrust knot, every Allan point of both channels
  and every servo parameter/step row. `f64::to_bits` is used as a 16-hex field,
  so the digest is independent of locale and float formatting.

Re-ingesting the same bytes gives a byte-identical canonical string and digest.
Changing one measured value changes both the source hash and the calibration
digest. Hashing uses `potok_sim::manifest::sha256_hex`, the same helper the
dataset provenance uses.

## Typed outputs and seams

`ingest_recipe()` returns a `CalibrationBundle`:

* `sensor_calibration()` reads ARW as `min(σ·√τ)` and bias stability as
  `min σ` from each Allan curve — the same readings the core contract defines.
  `apply_to_config(&mut Config)` writes only `gyro_arw`,
  `gyro_bias_stability`, `accel_arw` and `accel_bias_stability`; no wind, aero,
  thrust, trajectory or scenario field is touched.
* `install_servo(Vehicle, dt)` calls `Vehicle::with_servo` with the ingested
  step response. `dt` is supplied by the caller: the harness does not invent a
  timestep. Without this call the vehicle keeps its ideal instantaneous fin.
* `thrust_curve()` returns the validated `(t, thrust)` curve. `Vehicle`'s
  thrust table also needs a mass-flow column; no mass-flow measurement is
  ingested, so this harness never fabricates that third field.

Applying any of these is an explicit caller action. No CLI command in this tree
writes a config or a dataset from a measurement.

## Running

From `tools/measurement/`:

```
cargo test
cargo clippy --all-targets -- -D warnings
cargo run --quiet --bin ingest          # JSON digest to stdout
cargo run --quiet --bin ingest -- --recipe PATH --root DIR
```

Paths in the recipe are relative to `--root` (default: the repo root), so the
caller's cwd cannot change which bytes are read. A refusal exits non-zero with
the diagnostic on stderr and no output file.

Observed on the shipped SYNTHETIC recipe (values are fixture identities, not
hardware results): `recipe_sha256`
`9c3a1c8a1e4fbbb253c531837cb76f8bfbf742e7c6f31c43f6c9b2e7254468c4`,
`calibration_sha256`
`8a4f1c4b728b5b20e65a97324366ac917085e42928a9229f967350ce90c257c5`,
source rows 9/5/5/7.

## Ceilings

* No real source data, no real flight log, no per-channel sigma for NEES.
* No online parameter fitting; the servo's delay/rate/deadband are input data.
* No single-axis/cross-axis Allan model beyond the four scalar readings; no
  temperature or run-to-run variance.
* No mass-flow measurement, so no complete `Vehicle::thrust_table` is produced.
* No automatic wiring into `generate-dataset`; the existing provenance
  companion hashes measurement files but does not identify or validate them.
* A label is provenance metadata, not authentication. Anyone can write
  `# source: SYNTHETIC`; the harness records that claim and never upgrades it.
