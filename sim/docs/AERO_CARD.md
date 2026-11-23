# Aero card: provenance, missing channels, and the LOUD refusal

## What the card is

`config/aero_card.toml` is a read-only extraction from OpenRocket Simulation 9,
databranch `Маршевая ступень`, of the sample at
`/home/sehaxe/Downloads/Telegram Desktop/гугугага.ork` (ZIP → `rocket.ork`).
594 finite, non-degenerate rows were binned onto a 30-knot Mach axis and a
29-knot angle-of-attack axis; each node is the mean of the nearest observed
rows. Only four OR columns are used:

| Quantity | OR column | Card channel |
|---|---:|---|
| Mach | 50 | `grid.mach` |
| angle of attack | 15 (rad → deg) | `grid.alpha_deg` |
| axial drag coefficient | 35 | `coefficients.cd0` |
| pitching moment coefficient | 37 | `coefficients.cm_pitch` |

There is no CFD, no analytic stand-in and no tuning in those two grids. Nodes
with no measurement in their neighbourhood are `nan`; a bilinear stencil that
touches one is a LOUD error, never a clamp or a zero.

## Why `cna`, `pitch_damping` and `cm_delta` are absent

These are not alternative names for columns we already have — they are
derivatives of quantities the flight data never recorded:

- **Fin control authority (`cm_delta`)** is `∂Cm/∂δ`. It needs fin deflection
  `δ` to vary. The extraction has no deflection column, so `δ` is unobservable
  and the derivative cannot be identified. `cm_pitch` is the raw OR pitching
  moment, **not** a control derivative; borrowing it would be a different
  physical quantity under the same symbol.
- **Normal-force slope (`cna`)** is `∂Cn/∂α`. Only axial `Cd` and `Cm` were
  exported; there is no normal-force column. Even `∂Cm/∂α` cannot be converted
  to `∂Cn/∂α` without the CP/CG separation and reference length, which the
  source does not provide.
- **Pitch damping** is `∂Cm/∂(qL/2V)`. It needs a pitch-rate `q` signal; the
  source has none.

So the three channels cannot be derived from repo-contained OR data. Adding
numbers to the card would be invention, and choosing them to make a controller
win would be tuning to apogee — both are refused. The honest state is that the
real card **cannot fly as shipped**; it is an evidence artifact, not a flight
model.

## The refusal contract

`AeroCard::channel(ch)` returns an empty slice for an absent channel and
`AeroCard::interpolate` returns `SimError::AeroCard` naming the channel, the
fact that no value was invented, and the escape (`supply a measured grid in the
card, or select the placeholder path`). `AeroCard::coeffs` refuses as a whole —
there is no partial struct and no zero default. The integrator
(`core/src/dynamics.rs:476-502`) records the fault in `Vehicle::card_fault`
because an RK4 step cannot return early; the episode loop
(`core/src/episode.rs:521-527`) converts it to `SimError::AeroOutOfGrid` before
the step is committed. A flight using the real card therefore stops loudly;
it never flies on zero coefficients.

The one end-to-end flyable card is `config/aero_card_placeholder.toml`, which
is explicitly the legacy placeholder fixture and is not fitted to the OR data.
The default `config/default.toml` also uses the placeholder path and increments
the COUNTED `aero::placeholder_uses()` marker.

## Gate

`core/tests/aero_channels_tests.rs` proves there is no silent zero-fill at four
layers:

1. the TOML text assigns only `cd0` and `cm_pitch`, and keeps the provenance
   markers naming the source and the three unmeasured signals;
2. all three channels are empty and every in-grid clean-flight query is LOUD
   with the channel name and escape text;
3. `coeffs` never returns `Ok` on this card, at any clean flight point;
4. a full episode with this card returns an error, never a trajectory.

It also shows the failure mode by parsing a synthetic zero-filled card whose
`coeffs` *does* return zeros — so the gate flips the moment a zero grid is
pasted into the real card.

To make the real card flyable, supply measured grids for the three derivatives
from a source that actually observes deflection, normal force and pitch rate,
with its own provenance in the card header. Do not derive them from `cd0` or
`cm_pitch`, and do not pick values by apogee.
