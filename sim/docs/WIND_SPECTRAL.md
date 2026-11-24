# SIMWIND — von Kármán spectral wind (opt-in)

Status: **implemented, gated and wired into the live simulator as an opt-in**
(SIMWINDWIRE). The module at `core/src/wind_spectral.rs` is exported from
`core/src/lib.rs`, held in `Config.spectral` (`#[serde(default)]`), validated at
the config boundary, and branched into the episode's wind call site. The CLI
exposes `--spectral-wind` as an override. `enabled = false` by default: no
field is built and the shipped `wind::Wind` path stays bit-identical.
`config/default.toml` is unchanged.

## What it models

The shipped `Wind` field is three independent AR(1) (Dryden-style) filters
driven by a seeded `StdRng`. This module is a different construction:

1. **Spectral synthesis.** A box of side `8 L` (`L` = integral length) is
   tiled by a lattice of wavevectors `k = (2π/8L)·n`, `|n_i| ≤ 8`, `kL ≤ 11`.
   `mode_count` (default 1024) lattice points are drawn **uniformly** with the
   seeded RNG; the rare high-`k` modes stay present. Each drawn mode gets
   energy `a² ∝ E(k)/k²`, normalised so `Σ 2a² = 3σ²` for this realisation.
   The `k²` divisor is the shell multiplicity: a shell holds `O(k²)` lattice
   points, so shell energy `∝ k² · E(k)/k² = E(k)` per unit wavenumber.
2. **von Kármán spectrum.** `E(k) ∝ (kL)⁴ / (1 + (kL)²)^(17/6)`. For `kL ≫ 1`
   this reduces to `k^(-5/3)`, the Kolmogorov inertial slope. It is a
   consequence of the exponent, not a fitted number.
3. **Divergence-free projection.** Each mode carries two independent Gaussian
   3-vectors `p, q ~ N(0, I₃)`, projected as `v ← v − k̂(k̂·v)`. The field
   `u(x) = Σ a (p cos(k·x) + q sin(k·x))` then satisfies `k·p = k·q = 0`
   mode by mode, hence `∇·u = 0` in the continuum. No projection, no field:
   the divergence gate would fail by orders of magnitude.
4. **Normalisation.** Total variance of `M` projected modes with `p,q ~ N(0,I₃)`
   is `2 M a²`; setting it to `3 σ²` gives `σ²` per component. The finite
   sample mean is removed with a constant DC offset (which cannot
   reintroduce divergence), so the delivered series is exactly zero-mean.
5. **Taylor hypothesis.** The field is frozen and advected past the vehicle at
   `advection_speed_mps` (the airspeed scale, **not** the 2 m/s mean wind).
   Step `k` reads `u(dir · U · k·dt)`. One field is built per episode; `at`
   only indexes the precomputed slice and evaluates the analytic mean.

A **1-cosine gust** and a **tanh shear layer** are added along the mean-wind
direction, with the same analytic shapes as `wind.rs`:

- `g(t) = ½ A (1 − cos(2π (t−t₀)/D))` for `t₀ ≤ t ≤ t₀+D`, else `0`. `t₀` is
  drawn from an independent sub-seed in `[0.08, 0.63] · duration_s`.
- `s(h) = ½ Δ (1 + tanh((h−h_s)/thickness))`.
- mean: `v₁₀ (h/10)^exponent`.

## Determinism and opt-in

All randomness derives from `episode_seed(run.seed, episode ^ tag)` with two
independent SplitMix64 tags (`SPEC`, `GUST`), so adding or removing a knob
cannot shift the shipped `WIND` / `DOMA` / `SENS` draw order. The local
`episode_seed` is bit-identical to `potok_sim::math::episode_seed` and a gate
pins the equality.

`SpectralConfig::default().enabled == false`. In that state `new` builds an
inert all-zero field and `at` returns `[0; 3]` for every step — an accidental
construction cannot change a trajectory. `from_toml_str` accepts a bare table
or a `[spectral]` wrapper; every field is validated LOUD before use (field
name, bad value, fix).

## Wiring (SIMWINDWIRE, done)

1. `core/src/lib.rs`: `pub mod wind_spectral;` plus root re-exports of
   `SpectralConfig` / `WindSpectral`.
2. `core/src/config.rs`: `#[serde(default)] pub spectral: SpectralConfig` on
   `Config`, and `spectral.validate()` called from `Config::validate` even when
   disabled, so a malformed field is LOUD before the switch is flipped.
3. `core/src/episode.rs`: one `WindSpectral::new(&cfg.spectral, cfg.run.seed,
   episode, cfg.run.dt, cfg.run.duration_s)` per episode when enabled. At the
   wind call site `at(step, alt)` replaces `Wind::at`; `gust`, `shear` and the
   v_10 / gust / shear metadata follow the same model, so truth never reports
   the shipped field while the spectral arm is flown. Disabled keeps the exact
   `Wind` expression, including the `air.wind_factor` multiplier.
4. `cli/src/main.rs`: `--spectral-wind` sets `cfg.spectral.enabled = true`
   after config load, in both the headless path and `generate-dataset`.

Domain randomisation is deliberately NOT forwarded: the spectral arm uses its
own typed `mean_wind_mps` / `direction_deg` / gust / shear scalars, not the
domain-randomised `[wind]` scalars or `air.wind_factor`. This keeps the two
wind models cleanly separable for an A/B. No comparison between them has been
run, and nothing was fitted.

## Gates

`cargo test -p potok-sim --test wind_spectral_tests` (11 tests):

| gate | proves |
|---|---|
| `default_is_opt_in_and_disabled_field_is_inert` | default config changes nothing |
| `local_episode_seed_matches_the_canonical_splitmix64` | the local seed is not a second RNG dialect |
| `field_is_bitwise_reproducible_and_seed_sensitive` | same `(seed, episode)` → identical field; either changes it |
| `modes_are_solenoidal_and_field_divergence_vanishes` | projection is load-bearing; `∇·u ≈ 0` |
| `delivered_field_has_zero_mean_and_bounded_intensity` | DC removed; RMS is the configured σ |
| `all_values_are_finite_and_invalid_config_is_loud` | no NaN/Inf; bad fields are named and rejected |
| `psd_slope_is_reproducible_and_matches_von_karman_inertial_range` | realised mode energies binned over `3 ≤ kL ≤ 9` fit `−5/3 ± 0.4` and repeat |
| `one_cos_gust_onset_is_bounded_and_envelope_is_exact` | onset in bounds, zero at both ends, peak `A`, along-wind only |
| `shear_layer_is_a_centered_bounded_monotone_jump` | midpoint `Δ/2`, bounded `[0, Δ]`, monotone, along-wind only |
| `taylor_hypothesis_reads_the_frozen_field_at_the_advected_position` | delivered slice equals the analytic field at `dir·U·t` |
| `standalone_config_is_opt_in_validated_and_accepts_both_spellings` | opt-in TOML parsing and validation |

The PSD gate fits the radially binned energy spectral density of the
**realised modes** (`energy_spectrum(20)`), not a closed-form formula. A 1-D
periodogram along the advection line was tried and rejected: with a finite
lattice the along-line projection starves wavenumbers above `kL ≈ 8`, which
drives any such fit to −14 regardless of the code's correctness. The radial
spectrum has tens of modes per bin and is stable across seeds.

## Integration gates

`cargo test -p potok-sim --test wind_wiring_tests` (5 tests):

| gate | proves |
|---|---|
| `disabled_switch_is_bit_identical_to_the_default_program` | an aggressively non-default but disabled `[spectral]` block changes no flown value |
| `enabled_spectral_changes_wind_truth` | enabled mode actually changes the trajectory and last-tick wind truth |
| `enabled_wind_truth_is_deterministic_by_seed` | same `(seed, episode)` is bitwise equal; either input changes it |
| `enabled_metadata_describes_the_spectral_field` | gust/shear/v_10 metadata follows the flown spectral model |
| `malformed_spectral_config_is_loud_even_when_disabled` | config-boundary validation names the offending field |

## What is not claimed

- No parameter here was fitted to trajectory outputs, an A/B, or a BPB-style
  target. The defaults are structural placeholders for a reproducible field.
- No comparison to `wind::Wind`, no evidence that spectral turbulence improves
  any control or identification metric. That requires an A/B after wiring.
- The frozen-field / constant-advection model is a deliberate simplification:
  real airspeed changes during flight. The ceiling: replace `U` with the true
  relative wind when a time-varying advection gate exists.
- This is not a digital-twin input: no measured atmospheric spectrum ships
  with the repo, so `L`, σ and the gust/shear values remain assumptions.
