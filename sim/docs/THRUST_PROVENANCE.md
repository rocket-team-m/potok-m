# Thrust-curve provenance

Status, 2026-11-24: the shipped curve is a **synthetic, rippled table with no
static-fire measurement behind it**. It is labelled `experimental` in
`config/default.toml` and is deliberately kept bit-for-bit. Nothing in this
document claims a measured thrust curve.

## Passport numbers (external source)

Transcribed from the motor passport chat PDFs into `config/default.toml` and
`SIM_FIDELITY.md`. These are the only external numbers in this document; no
PDF or static-fire record is stored in the repository.

| | RDK-2000 (stage 1) | RDK-1000 (stage 2) |
|---|---:|---:|
| Impulse | 2.2–2.4 kN·s | 1.0–1.2 kN·s |
| Average thrust | 390–400 N | 260–270 N |
| Peak thrust | 500–550 N | 550–600 N |
| Burn | 5–6 s | 3–3.5 s |
| Propellant | 1.88 kg | 1.0 kg |

Passport minimum recipe: `400 N × 5.5 s + 312.5 N × 3.2 s = 3.2 kN·s`.
Passport maximum total: `2.4 + 1.2 = 3.6 kN·s`. So the gate envelope is
`[3200, 3600] N·s` total, stage-1 peak `[500, 550] N`, stage-2 peak
`[550, 600] N` (constants in `core/src/config.rs`).

## What the shipped table actually is

Read-only trapezoidal arithmetic from the 200 knots at
`config/default.toml:thrust_profile` (first 0.000 s, last 9.95 s), split at
`stage_sep_time_s = 6.0`:

| Quantity | Table | Passport |
|---|---:|---:|
| Total impulse | 3627.8227 N·s | 3200–3600 N·s — **over by 27.8** |
| Stage-1 peak | 610.0 N at t = 1.9 s | 500–550 N — **over by 60** |
| Stage-2 peak | 590.0 N at t = 7.5 s | 550–600 N — inside |
| Stage-1 impulse | 2355.106 N·s | 2200–2400 N·s — inside |
| Stage-2 impulse | 1272.55 N·s | 1000–1200 N·s — over by 72.6 |
| Propellant (mass-flow integral) | 2.9128 kg | 1.88 + 1.0 = 2.88 kg — close |

The per-stage impulses omit the single 5.95→6.0 s interval (0.16675 N·s),
hence they sum to 3627.656 rather than 3627.8227 N·s. `docs/SIM_FIDELITY.md`
and `sim/README.md` said "average thrust and mass flow only; peaks and ripple
not modelled" — that describes an older config, not the shipped one. SIMIDEAL
(2026-11-24 §2) recorded the discrepancy.

## Contract

- `aero.thrust_source` is an explicit provenance label:
  - `"experimental"` (default) — the curve is **not** claimed to be within the
    passport envelope. No envelope check runs. This is the shipped case.
  - `"passport"` — the curve claims the envelope. `Config::validate()` then
    refuses it LOUDLY on total impulse or either stage peak outside the ranges,
    naming the field, the measured value and the `experimental` escape hatch.
- Structural checks (finite, non-negative, strictly increasing time) apply to
  **both** labels at the config boundary. No silent zero-fill or clamp.
- `#[serde(default)]` means a config written before this field exists still
  loads and is labelled `experimental`. `skip_serializing_if` means the
  canonical config bytes — and therefore `manifest.config_hash` and every
  generated file — are identical to the pre-label config when the label is
  `experimental`. Labelling `passport` does change the hash, as it should.
- The runtime interpolator (`dynamics.rs:thrust_at`) is unchanged: it still
  clamps inside the table and returns zero outside it. Fixing that is a
  separate change with its own trajectory impact.

## Decisions and ceilings

- **The knots were not retuned.** The 610 N peak and the 3627.8 N·s impulse
  drive the pinned OpenRocket parity gates (apogee/vmax) and the zero-command
  bit-identity gate. Editing them to fit the passport would be tuning
  trajectory outputs, which SIMIDEAL forbids as the fix for this finding.
  The honest label plus this document is the fix.
- **The gate's ceiling:** it checks total impulse and both stage peaks, not
  per-stage impulse or average thrust. Those are recorded above for an
  auditor; add per-stage windows if a `passport` claim ever needs that level
  of scrutiny. No measurement file exists to justify a tighter gate today.
- **Open, not touched here:** `config.rs::default_stage_sep_time()` is 5.5 s /
  0.92 kg while `default.toml` ships 6.0 s / 2.55 kg. The field value in the
  config wins at runtime; the mismatch is noted, not changed, because the
  fallback value participates in other configs.

Gate: `cargo test -p potok-sim --test thrust_curve_tests` (7 tests).
