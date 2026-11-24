// SPDX-License-Identifier: GPL-3.0-or-later
//! SIMWIND — von Kármán spectral turbulence with Taylor advection.
//!
//! Physically distinct from the shipped [`crate::wind::Wind`] AR(1) field. A
//! stationary, solenoidal Gaussian field is synthesised once from seeded
//! Fourier modes with a 3-D von Kármán energy spectrum
//! `E(k) ∝ (kL)^4 / (1 + (kL)^2)^(17/6)` (inertial range `k^-5/3`). Each mode is
//! projected onto the plane perpendicular to its wavevector, so `∇·u = 0`
//! holds in the continuum and the field cannot produce spurious compressional
//! accel. Taylor's hypothesis freezes that field and advects it past the
//! vehicle at `advection_speed_mps`: step `k` reads `u(dir * U * k*dt)`. A
//! 1-cosine gust and a tanh shear layer are added along the mean-wind
//! direction, matching the analytic shapes used by the shipped field.
//!
//! Everything is a pure function of `(root_seed, episode, config)`. There is no
//! global RNG, no wall clock and no allocation in `at`. `enabled = false` (the
//! shipped default) makes `new` build an inert, all-zero field, so the default
//! wind path and every existing A/B stay bit-identical.
//!
//! Integration note for the crate owner: this file is intentionally NOT wired
//! into `lib.rs` by this task. Add `pub mod wind_spectral;`, hold
//! `SpectralConfig` in `Config` behind `#[serde(default)]`, call
//! `cfg.spectral.validate()` from `Config::validate`, and branch on
//! `spectral.enabled` at the wind call site (CLI flag `--spectral-wind`).
//! See `docs/WIND_SPECTRAL.md`.

use std::f64::consts::TAU;
use std::fmt;

use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};

/// Frozen-field box side in integral lengths. Sets the irreducible low-`k`
/// edge of the spectrum (`k_min L = 2π / DOMAIN_LENGTHS`).
pub const DOMAIN_LENGTHS: f64 = 8.0;
/// Largest lattice index per axis. With `DOMAIN_LENGTHS = 8` this reaches
/// `|k| L ≈ 10.9`, past the `kL ≤ 9` end of the fitted inertial band.
pub const LATTICE_K: i32 = 8;
/// Upper bound on retained `kL`, independent of the lattice cap.
pub const K_MAX_L: f64 = 11.0;
/// Sub-seed tags. Independent of the shipped `WIND` / `DOMA` / `SENS` streams.
const SPECTRAL_KEY: u64 = 0x5350_4543; // "SPEC"
const GUST_KEY: u64 = 0x4755_5354; // "GUST"

/// SplitMix64 finalizer. Bit-identical to `potok_sim::math::episode_seed`;
/// duplicated here so this module compiles standalone through `#[path]`.
/// `wind_spectral_tests` pins the equality.
#[inline]
pub fn episode_seed(root: u64, index: u64) -> u64 {
    let mut z = root.wrapping_add(index.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn norm3(a: [f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}

#[inline]
fn scale3(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// 3-D von Kármán energy-spectrum shape `E(k) ∝ (kL)^4/(1+(kL)^2)^(17/6)`.
/// The `k^-5/3` inertial slope is a consequence of the exponent, not a fitted
/// constant. Returns 0 for non-positive or non-finite `k_l`.
#[inline]
pub fn von_karman_shape(k_l: f64) -> f64 {
    if k_l <= 0.0 || !k_l.is_finite() {
        return 0.0;
    }
    let x2 = k_l * k_l;
    x2 * x2 / (1.0 + x2).powf(17.0 / 6.0)
}

/// Fourier amplitude of a lattice mode. A shell at `k` holds `O(k^2)`
/// lattice points, so sampling density `∝ E(k)/k^2` reproduces `E(k)` per
/// unit wavenumber instead of piling energy into the tail.
#[inline]
pub fn lattice_weight(k_l: f64) -> f64 {
    let shape = von_karman_shape(k_l);
    if k_l <= 0.0 {
        0.0
    } else {
        shape / (k_l * k_l)
    }
}

/// Remove the component of `v` parallel to `k`. Used on both Gaussian
/// amplitude vectors of every mode; the result is exactly solenoidal.
pub fn project_solenoidal(k: [f64; 3], v: [f64; 3]) -> [f64; 3] {
    let km = norm3(k);
    if km <= 0.0 || !km.is_finite() {
        return v;
    }
    let kh = scale3(k, 1.0 / km);
    let parallel = scale3(kh, dot3(kh, v));
    [v[0] - parallel[0], v[1] - parallel[1], v[2] - parallel[2]]
}

// --- Config -----------------------------------------------------------------

/// Opt-in spectral-wind parameters. Defaults are structural placeholders for
/// the gate, NOT values fitted to any trajectory or A/B target.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SpectralConfig {
    /// Master switch. `false` keeps the shipped `Wind` path and every A/B.
    pub enabled: bool,
    /// Per-component RMS of the turbulence, m/s. `0` is a valid quiet field.
    pub intensity_mps: f64,
    /// von Kármán integral length, m.
    pub integral_length_m: f64,
    /// Frozen-field advection speed (the vehicle's airspeed scale), m/s.
    pub advection_speed_mps: f64,
    /// Mean along-wind speed at 10 m, m/s.
    pub mean_wind_mps: f64,
    /// Mean wind direction, degrees from +X towards +Y.
    pub direction_deg: f64,
    /// Power-law exponent of the mean boundary-layer profile.
    pub shear_exponent: f64,
    /// Velocity jump across the discrete shear layer, m/s.
    pub shear_delta_mps: f64,
    /// Shear-layer centre altitude, m.
    pub shear_altitude_m: f64,
    /// Shear-layer tanh thickness, m.
    pub shear_thickness_m: f64,
    /// 1-cosine gust amplitude, m/s. `0` means no gust onset is drawn.
    pub gust_amplitude_mps: f64,
    /// 1-cosine gust duration, s.
    pub gust_duration_s: f64,
    /// Number of seeded Fourier modes. Cost is O(modes × steps) once per episode.
    pub mode_count: usize,
}

impl Default for SpectralConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            intensity_mps: 0.5,
            integral_length_m: 60.0,
            advection_speed_mps: 90.0,
            mean_wind_mps: 2.0,
            direction_deg: 25.0,
            shear_exponent: 0.22,
            shear_delta_mps: 0.0,
            shear_altitude_m: 150.0,
            shear_thickness_m: 12.0,
            gust_amplitude_mps: 0.0,
            gust_duration_s: 2.0,
            mode_count: 1024,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpectralError {
    pub field: &'static str,
    pub detail: String,
}

impl SpectralError {
    fn new(field: &'static str, detail: impl Into<String>) -> Self {
        Self {
            field,
            detail: detail.into(),
        }
    }
}

impl fmt::Display for SpectralError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "spectral.{}: {} (config rejected before use; fix the value or keep enabled = false)",
            self.field, self.detail
        )
    }
}

impl std::error::Error for SpectralError {}

#[derive(Deserialize)]
struct Wrapped {
    spectral: SpectralConfig,
}

impl SpectralConfig {
    /// Parse a standalone spectral config. Accepts either a bare TOML table or
    /// one wrapped in `[spectral]`, so the same text can be embedded in the
    /// main `Config` or kept as its own opt-in file.
    pub fn from_toml_str(s: &str) -> Result<Self, SpectralError> {
        match toml::from_str::<Self>(s) {
            Ok(cfg) => {
                cfg.validate()?;
                Ok(cfg)
            }
            Err(bare) => match toml::from_str::<Wrapped>(s) {
                Ok(w) => {
                    w.spectral.validate()?;
                    Ok(w.spectral)
                }
                Err(_) => Err(SpectralError::new(
                    "config",
                    format!("not a valid spectral config ({bare}); use named fields from SpectralConfig"),
                )),
            },
        }
    }

    pub fn validate(&self) -> Result<(), SpectralError> {
        for (field, v) in [
            ("intensity_mps", self.intensity_mps),
            ("integral_length_m", self.integral_length_m),
            ("advection_speed_mps", self.advection_speed_mps),
            ("mean_wind_mps", self.mean_wind_mps),
            ("direction_deg", self.direction_deg),
            ("shear_exponent", self.shear_exponent),
            ("shear_delta_mps", self.shear_delta_mps),
            ("shear_altitude_m", self.shear_altitude_m),
            ("shear_thickness_m", self.shear_thickness_m),
            ("gust_amplitude_mps", self.gust_amplitude_mps),
            ("gust_duration_s", self.gust_duration_s),
        ] {
            finite(field, v)?;
        }
        range("intensity_mps", self.intensity_mps, 0.0, 50.0)?;
        range("integral_length_m", self.integral_length_m, 1e-3, 1e5)?;
        range("advection_speed_mps", self.advection_speed_mps, 0.0, 1e4)?;
        range("mean_wind_mps", self.mean_wind_mps, 0.0, 200.0)?;
        range("direction_deg", self.direction_deg, -360.0, 360.0)?;
        range("shear_exponent", self.shear_exponent, 0.0, 2.0)?;
        range("shear_delta_mps", self.shear_delta_mps, 0.0, 100.0)?;
        range("shear_altitude_m", self.shear_altitude_m, 0.0, 1e5)?;
        range("shear_thickness_m", self.shear_thickness_m, 1e-3, 1e4)?;
        range("gust_amplitude_mps", self.gust_amplitude_mps, 0.0, 100.0)?;
        range("gust_duration_s", self.gust_duration_s, 1e-3, 1e4)?;
        if !(128..=4096).contains(&self.mode_count) {
            return Err(SpectralError::new(
                "mode_count",
                format!(
                    "{} outside [128, 4096]; set a mode count in that range",
                    self.mode_count
                ),
            ));
        }
        Ok(())
    }
}

fn finite(field: &'static str, v: f64) -> Result<(), SpectralError> {
    if v.is_finite() {
        Ok(())
    } else {
        Err(SpectralError::new(
            field,
            format!("{v} is not finite; supply a finite number"),
        ))
    }
}

fn range(field: &'static str, v: f64, lo: f64, hi: f64) -> Result<(), SpectralError> {
    if (lo..=hi).contains(&v) {
        Ok(())
    } else {
        Err(SpectralError::new(
            field,
            format!("{v} outside [{lo}, {hi}]; set a value in that range"),
        ))
    }
}

// --- Field ------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Mode {
    k: [f64; 3],
    p: [f64; 3],
    q: [f64; 3],
    amp: f64,
}

/// One seeded, divergence-free wind realization. `turb` is built once in
/// `new`; `at` only indexes it and evaluates the analytic mean profile.
#[derive(Debug, Clone)]
pub struct WindSpectral {
    cfg: SpectralConfig,
    dir: [f64; 2],
    modes: Vec<Mode>,
    turb: Vec<[f64; 3]>,
    dc: [f64; 3],
    gust: Option<(f64, f64)>,
    dt: f64,
}

impl WindSpectral {
    /// Build one episode's field. `root_seed` is `run.seed`; `episode` is the
    /// episode index. `dt`/`duration_s` come from `run` and set the sample grid.
    pub fn new(
        cfg: &SpectralConfig,
        root_seed: u64,
        episode: u64,
        dt: f64,
        duration_s: f64,
    ) -> Result<Self, SpectralError> {
        cfg.validate()?;
        finite("run.dt", dt)?;
        finite("run.duration_s", duration_s)?;
        if dt <= 0.0 || duration_s <= 0.0 {
            return Err(SpectralError::new(
                "run",
                "dt and duration_s must be positive",
            ));
        }
        let steps_f = duration_s / dt;
        let steps = steps_f.round() as usize;
        if steps == 0 || (steps_f - steps as f64).abs() > 1e-6 {
            return Err(SpectralError::new(
                "run.duration_s",
                format!("{duration_s} must be a positive integer multiple of dt {dt}"),
            ));
        }

        let theta = cfg.direction_deg.to_radians();
        let dir = [theta.cos(), theta.sin()];
        let advection = cfg.advection_speed_mps;
        let mut rng = StdRng::seed_from_u64(episode_seed(root_seed, episode ^ SPECTRAL_KEY));

        let modes = if cfg.enabled && cfg.intensity_mps > 0.0 {
            synthesize(&mut rng, cfg)
        } else {
            Vec::new()
        };

        let mut turb = Vec::with_capacity(steps);
        for step in 0..steps {
            turb.push(field(
                &modes,
                advection_position(dir, advection, step as f64 * dt),
            ));
        }

        // The analytic field has no DC term; the finite sample mean is removed
        // so the delivered series is exactly zero-mean. The offset is constant
        // and therefore cannot reintroduce divergence.
        let n = steps as f64;
        let mut mean = [0.0; 3];
        for t in &turb {
            mean[0] += t[0] / n;
            mean[1] += t[1] / n;
            mean[2] += t[2] / n;
        }
        let dc = [-mean[0], -mean[1], -mean[2]];
        for t in &mut turb {
            *t = add3(*t, dc);
        }

        let gust = if cfg.enabled && cfg.gust_amplitude_mps > 0.0 {
            let mut gust_rng = StdRng::seed_from_u64(episode_seed(root_seed, episode ^ GUST_KEY));
            let t0 = duration_s * (0.08 + 0.55 * gust_rng.gen::<f64>());
            Some((t0, cfg.gust_duration_s))
        } else {
            None
        };

        Ok(Self {
            cfg: cfg.clone(),
            dir,
            modes,
            turb,
            dc,
            gust,
            dt,
        })
    }

    /// Precomputed turbulent triplet at each integration step.
    pub fn turbulence(&self) -> &[[f64; 3]] {
        &self.turb
    }

    /// Step interval the field was built for.
    pub fn dt(&self) -> f64 {
        self.dt
    }

    /// Constant removed from the finite sample so its mean is exactly zero.
    pub fn dc_offset(&self) -> [f64; 3] {
        self.dc
    }

    /// Analytic field at an arbitrary world position, including the DC offset.
    pub fn raw_field(&self, x: [f64; 3]) -> [f64; 3] {
        add3(field(&self.modes, x), self.dc)
    }

    /// Radially binned 3-D energy spectral density `E(k)` of the realised
    /// modes: `(k·L, energy per unit k)`. This is the spectrum the field was
    /// built from, measured on the actual amplitudes, not the target formula.
    pub fn energy_spectrum(&self, bins: usize) -> Vec<(f64, f64)> {
        assert!(bins >= 4, "energy_spectrum needs at least four bins");
        let length = self.cfg.integral_length_m;
        let lo = (TAU / (DOMAIN_LENGTHS * length)).ln();
        let hi = K_MAX_L.ln() - length.ln();
        let mut energy = vec![0.0; bins];
        for m in &self.modes {
            let k = norm3(m.k);
            let b = (((k.ln() - lo) / (hi - lo)) * bins as f64) as usize;
            energy[b.min(bins - 1)] += 2.0 * m.amp * m.amp;
        }
        let width = (hi - lo) / bins as f64;
        energy
            .into_iter()
            .enumerate()
            .map(|(b, e)| {
                let k = (lo + width * (b as f64 + 0.5)).exp();
                (k * length, e / (width * k))
            })
            .collect()
    }

    /// Divergence of the analytic field by central differences. Non-zero only
    /// if the solenoidal projection regresses.
    pub fn divergence_at(&self, x: [f64; 3], eps: f64) -> f64 {
        assert!(
            eps > 0.0 && eps.is_finite(),
            "eps must be positive and finite"
        );
        let mut div = 0.0;
        for axis in 0..3 {
            let mut plus = x;
            let mut minus = x;
            plus[axis] += eps;
            minus[axis] -= eps;
            div += (self.raw_field(plus)[axis] - self.raw_field(minus)[axis]) / (2.0 * eps);
        }
        div
    }

    /// Along-wind 1-cosine gust at time `t` seconds, m/s.
    pub fn gust_at(&self, t: f64) -> f64 {
        let Some((t0, duration)) = self.gust else {
            return 0.0;
        };
        let u = (t - t0) / duration;
        if (0.0..=1.0).contains(&u) {
            0.5 * self.cfg.gust_amplitude_mps * (1.0 - (TAU * u).cos())
        } else {
            0.0
        }
    }

    /// Across-wind tanh shear jump at altitude `h`, m/s.
    pub fn shear_at(&self, h: f64) -> f64 {
        if !self.cfg.enabled {
            return 0.0;
        }
        let z = (h - self.cfg.shear_altitude_m) / self.cfg.shear_thickness_m;
        0.5 * self.cfg.shear_delta_mps * (1.0 + z.tanh())
    }

    pub fn gust_info(&self) -> Option<(f64, f64, f64)> {
        self.gust
            .map(|(t0, d)| (t0, d, self.cfg.gust_amplitude_mps))
    }

    /// Position sampled by Taylor's hypothesis at step `k`.
    pub fn advection_position(&self, step: usize) -> [f64; 3] {
        assert!(step < self.turb.len(), "step {step} outside the field");
        advection_position(
            self.dir,
            self.cfg.advection_speed_mps,
            step as f64 * self.dt,
        )
    }

    /// Total wind at step `step` and altitude `alt`, m/s, world frame.
    pub fn at(&self, step: usize, alt: f64) -> [f64; 3] {
        assert!(step < self.turb.len(), "step {step} outside the field");
        if !self.cfg.enabled {
            return [0.0; 3];
        }
        let h = alt.max(0.1);
        let mean = self.cfg.mean_wind_mps * (h / 10.0).powf(self.cfg.shear_exponent);
        let speed = mean + self.shear_at(h) + self.gust_at(step as f64 * self.dt);
        let t = self.turb[step];
        [self.dir[0] * speed + t[0], self.dir[1] * speed + t[1], t[2]]
    }
}

#[inline]
fn advection_position(dir: [f64; 2], speed: f64, t: f64) -> [f64; 3] {
    [dir[0] * speed * t, dir[1] * speed * t, 0.0]
}

/// Evaluate the seeded mode sum. Each term is `a (p cos(k·x) + q sin(k·x))`
/// with `p, q ⊥ k`, so every term and the whole sum are divergence-free.
fn field(modes: &[Mode], x: [f64; 3]) -> [f64; 3] {
    let mut out = [0.0; 3];
    for m in modes {
        let (s, c) = (dot3(m.k, x)).sin_cos();
        let a = m.amp * c;
        let b = m.amp * s;
        out[0] += m.p[0] * a + m.q[0] * b;
        out[1] += m.p[1] * a + m.q[1] * b;
        out[2] += m.p[2] * a + m.q[2] * b;
    }
    out
}

/// Box-Muller from the seeded generator; `1 - u` avoids `ln(0)`.
#[inline]
fn gaussian(rng: &mut StdRng) -> f64 {
    let u1 = 1.0 - rng.gen::<f64>();
    let u2 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
}

/// Uniform lattice sampling with spectrum-shaped amplitudes. A shell at `k`
/// holds `O(k^2)` lattice points, so mode energy `∝ E(k)/k^2 = w(k)` makes
/// shell energy `∝ E(k)`. Uniform sampling keeps the rare high-`k` modes
/// present; importance sampling with equal amplitudes starves them.
fn synthesize(rng: &mut StdRng, cfg: &SpectralConfig) -> Vec<Mode> {
    let length = cfg.integral_length_m;
    let domain = DOMAIN_LENGTHS * length;
    let fundamental = TAU / domain;
    let k = LATTICE_K;

    let mut wavevectors = Vec::new();
    let mut weights = Vec::with_capacity(((2 * k + 1).pow(3)) as usize);
    for i in -k..=k {
        for j in -k..=k {
            for l in -k..=k {
                if i == 0 && j == 0 && l == 0 {
                    continue;
                }
                let kk = [
                    i as f64 * fundamental,
                    j as f64 * fundamental,
                    l as f64 * fundamental,
                ];
                let k_l = norm3(kk) * length;
                if k_l > K_MAX_L {
                    continue;
                }
                wavevectors.push(kk);
                weights.push(lattice_weight(k_l));
            }
        }
    }
    assert!(
        !wavevectors.is_empty(),
        "spectral lattice produced no modes"
    );

    let picks: Vec<usize> = (0..cfg.mode_count)
        .map(|_| (rng.gen::<f64>() * wavevectors.len() as f64) as usize)
        .collect();
    // Total variance of projected modes with p, q ~ N(0, I3) is 2 Σ a².
    // Setting that to 3 σ² gives σ² per component for this realization.
    let sum_w: f64 = picks.iter().map(|&i| weights[i]).sum();
    assert!(sum_w > 0.0, "spectral weights must be positive");
    let scale = cfg.intensity_mps * (1.5 / sum_w).sqrt();
    let mut modes = Vec::with_capacity(cfg.mode_count);
    for i in picks {
        let kk = wavevectors[i];
        let unit = scale3(kk, 1.0 / norm3(kk));
        let p = project_solenoidal(unit, [gaussian(rng), gaussian(rng), gaussian(rng)]);
        let q = project_solenoidal(unit, [gaussian(rng), gaussian(rng), gaussian(rng)]);
        modes.push(Mode {
            k: kk,
            p,
            q,
            amp: scale * weights[i].sqrt(),
        });
    }
    modes
}
