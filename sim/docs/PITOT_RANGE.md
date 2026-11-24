# PITOT_RANGE — typed pitot range / saturation model

**Date:** 2026-11-24 · **Scope:** `sim/core/src/sensor.rs` only. No config preset,
gate, threshold, physics constant, trainer, weight or lateral target was
changed. Default wire bytes and the 56-byte layout are unchanged.

**Tags:** **[M]** recorded measurement · **[T]** arithmetic from recorded
constants · **[E]** estimate, not a bound.

---

## 0. Why

AILIMIT (`.bulba/night/AILIMIT.md` §5) records that the shipped 30 kPa pitot
saturates over the 26–31 kPa gate-open dynamic pressure. `FLAG_FAULT` is set,
`AI_READY` is cleared, and the first AI command lands at **t = 10.82–11.88 s
(mean 11.25 s)** instead of the gate-open time 9.95 s. That dead window covers
the highest-authority part of the flight (fin authority falling ~62 → 40 m/s²)
and is bounded at **~1.3 s / ~88 m** of deployment radius under the report's
generous braking instrument [M/T/E — the 88 m inherits that instrument's
optimism].

The lever is **a wider-range instrument**, not a relaxed gate. Declaring AI
healthy on invalid pitot data would be gate weakening and is excluded. The
shipped sensor model had one boolean (`saturated`) shared with gyro/accel/mag/
baro and one silent `clamp(...).round() as i16` on the pitot wire field: a
reader could not tell *why* a pitot sample was rejected.

## 1. The typed model

All types live in `core/src/sensor.rs`:

| type | meaning |
|---|---|
| `PitotProfile` | `Production` (shipped instrument) or `ExtendedRange` (experimental, opt-in). |
| `EXTENDED_FULL_SCALE_PA` | 35 000 Pa full scale for the experimental instrument. |
| `PitotRangeFault` | Named cause: `NonFinite`, `FullScale`, `WireOverflow`, `WireUnderflow`. |
| `PitotRangeStatus` | `InRange` or `OutOfRange(PitotRangeFault)`. |
| `classify_pitot_range(dp_pa, full_scale_pa)` | Pure classifier; the named reason is the result. |
| `Sensor::with_pitot_profile(..)` | Opt into a profile; `Sensor::new` stays `Production`. |
| `Sensor::pitot_range()` | Named status of the most recent sample (per-frame, not sticky). |

Classification priority: **non-finite → instrument full scale → i16 wire
limits**. A reading above both bounds reports `FullScale` for the production
instrument (the physical limit binds first); a reading inside an extended
sensor range but past 32 767 Pa reports `WireOverflow`. This is what makes the
encoding clamp loud instead of silent: the clamp to the frozen i16 field is
still performed for wire compatibility, but the reason travels with the sample
and is observable through `Sensor::pitot_range()`.

The `Frame` itself is deliberately **not** changed: four other test files
construct it with explicit struct literals, and adding a field would break
them. The typed reason is exposed on `Sensor`, not on the wire frame.

## 2. Profiles

**`Production` is the default.** Full scale is `sensor.pitot_sat` from config
(30 000 Pa in `default.toml`), so the legacy path is bit-identical. Verified
against a golden wire encoding captured from the pre-model code (test
`production_profile_keeps_the_legacy_wire_bytes`): at qdyn = 31 kPa the frame
is `flags=0x8e` (FAULT set, AI_READY clear), `pitot_dp=31000`, CRC `0x9a807918`.

**`ExtendedRange` is experimental and opt-in.** It replaces full scale with
35 000 Pa and prints one COUNTED-style `eprintln!` at construction naming the
profile, its full scale, the wire ceiling and the production alternative. No
shipped preset selects it. Opt in explicitly:

```rust
use potok_sim::sensor::{PitotProfile, Sensor};
let sensor = Sensor::with_pitot_profile(&cfg, episode, PitotProfile::ExtendedRange);
```

Measured on a quiet rig at qdyn = 31 kPa [M]:

| profile | `pitot_range()` | `saturated` | `pitot_dp` | `AI_READY` |
|---|---|---|---|---|
| Production (30 kPa) | `OutOfRange(FullScale)` | true | 31000 | clear |
| ExtendedRange (35 kPa) | `InRange` | false | 31000 | set |

The two frames differ **only** in the flags byte and its CRC. Airspeed validity
(`|v_rel| > 3 m/s`) and `pitot_dp > 0` are unchanged; the extended profile
removes the dead window by modeling a wider instrument, not by changing who is
allowed to consume a sample.

## 3. Boundaries (honest limits)

- The wire field is i16 Pa, max **32 767 Pa**. The 35 kPa full scale is an
  *instrument* range; it is only representable on the wire up to 32.767 kPa.
  The dead window (qdyn ≤ 31 kPa) sits inside that, but a 34 kPa reading under
  the extended profile is reported as `WireOverflow`, FAULT-set — loud, never
  silently clamped (test `extended_range_wire_overflow_is_loud_not_silently_clamped`).
- `ExtendedRange` is **not wired into the flight harness**: `episode.rs` calls
  `Sensor::new` (production). Wiring it in is an explicit, owner-decided change
  outside this task's file scope and outside any lateral-target tuning.
- The ~88 m figure is an [E] from the report's generous bound, not a flight
  measurement; it is the motivation, not a claim this change delivers on its
  own. No lateral RMS, deployment radius or gate was re-measured or tuned here.
- The only behavior difference on the production path is that a non-finite
  pitot sample now sets `saturated` (`NonFinite`); the simulator guards
  non-finite state upstream, so this cannot change a healthy flight's bytes.

## 4. Verification

`core/tests/pitot_range_tests.rs` (5 tests): classifier reason table,
profile/default selection and full-scale arithmetic, production golden bytes,
extended profile removing the dead window with only flags+CRC moved, and loud
wire-overflow handling. `cargo test -p potok-sim`, `cargo fmt --check` and
`cargo clippy --workspace --all-targets` are green at commit time.
