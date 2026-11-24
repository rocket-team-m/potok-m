// SPDX-License-Identifier: GPL-3.0-or-later
//! SIMWIND gates for the opt-in von Kármán spectral wind field.
//!
//! The module is included by path so this suite compiles without touching
//! `lib.rs` (the crate owner wires `pub mod wind_spectral;` when integrating).
//! Gates: opt-in inertness, deterministic seed, solenoidal projection,
//! divergence-free field, zero mean, finite values, von Kármán inertial PSD
//! slope, 1-cos gust onset bounds, tanh shear, and Taylor advection.

#[path = "../src/wind_spectral.rs"]
mod wind_spectral;

use potok_sim::math::episode_seed as canonical_episode_seed;
use wind_spectral::*;

const SEED: u64 = 1337;
const EPISODE: u64 = 10;
const DT: f64 = 0.01;
const DURATION: f64 = 40.0;
const STEPS: usize = 4000;

fn enabled() -> SpectralConfig {
    SpectralConfig {
        enabled: true,
        ..SpectralConfig::default()
    }
}

fn field(cfg: &SpectralConfig) -> WindSpectral {
    WindSpectral::new(cfg, SEED, EPISODE, DT, DURATION).unwrap()
}

fn direction(cfg: &SpectralConfig) -> [f64; 2] {
    let theta = cfg.direction_deg.to_radians();
    [theta.cos(), theta.sin()]
}

#[test]
fn default_is_opt_in_and_disabled_field_is_inert() {
    let cfg = SpectralConfig::default();
    assert!(
        !cfg.enabled,
        "the shipped default must preserve existing A/Bs"
    );
    cfg.validate().unwrap();
    let w = field(&cfg);

    assert!(w.turbulence().iter().all(|t| *t == [0.0; 3]));
    assert_eq!(w.gust_info(), None);
    assert_eq!(w.raw_field([1.0, 2.0, 3.0]), [0.0; 3]);
    assert_eq!(w.dc_offset(), [0.0; 3]);
    for step in 0..STEPS {
        assert_eq!(w.at(step, 100.0), [0.0; 3], "step {step}");
    }
}

#[test]
fn local_episode_seed_matches_the_canonical_splitmix64() {
    for (root, index) in [(0, 0), (SEED, EPISODE), (u64::MAX, 7), (42, 999)] {
        assert_eq!(
            episode_seed(root, index),
            canonical_episode_seed(root, index),
            "root {root} index {index}"
        );
    }
}

#[test]
fn field_is_bitwise_reproducible_and_seed_sensitive() {
    let cfg = enabled();
    let a = field(&cfg);
    let b = field(&cfg);
    assert_eq!(a.turbulence(), b.turbulence());
    assert_eq!(a.gust_info(), b.gust_info());
    assert!(a.turbulence().iter().flatten().all(|v| v.is_finite()));

    let other_seed = WindSpectral::new(&cfg, SEED + 1, EPISODE, DT, DURATION).unwrap();
    let other_episode = WindSpectral::new(&cfg, SEED, EPISODE + 1, DT, DURATION).unwrap();
    assert_ne!(a.turbulence(), other_seed.turbulence());
    assert_ne!(a.turbulence(), other_episode.turbulence());
}

#[test]
fn modes_are_solenoidal_and_field_divergence_vanishes() {
    let k = [1.0, -2.0, 0.5];
    for v in [[1.0, 2.0, 3.0], [-0.3, 4.0, 1.1], [0.0, 0.0, 0.0]] {
        let p = project_solenoidal(k, v);
        assert!(
            (k[0] * p[0] + k[1] * p[1] + k[2] * p[2]).abs() < 1e-12,
            "projection left a parallel component: {p:?}"
        );
    }

    let w = field(&enabled());
    for i in 0..12 {
        let x = [
            i as f64 * 13.7 - 40.0,
            i as f64 * 7.3 - 20.0,
            i as f64 * 3.1,
        ];
        let div = w.divergence_at(x, 0.05);
        assert!(div.abs() < 1e-5, "position {x:?}: divergence {div}");
    }
    let magnitude = w
        .turbulence()
        .iter()
        .map(|t| (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt())
        .fold(0.0_f64, f64::max);
    assert!(magnitude > 0.1, "the field must not be empty: {magnitude}");
}

#[test]
fn delivered_field_has_zero_mean_and_bounded_intensity() {
    let cfg = enabled();
    let w = field(&cfg);
    let n = w.turbulence().len() as f64;
    for axis in 0..3 {
        let mean = w.turbulence().iter().map(|t| t[axis]).sum::<f64>() / n;
        assert!(mean.abs() < 1e-12, "axis {axis}: mean {mean}");
        let variance = w
            .turbulence()
            .iter()
            .map(|t| (t[axis] - mean).powi(2))
            .sum::<f64>()
            / n;
        let rms = variance.sqrt();
        assert!(
            (0.5 * cfg.intensity_mps..1.5 * cfg.intensity_mps).contains(&rms),
            "axis {axis}: RMS {rms} vs intensity {}",
            cfg.intensity_mps
        );
    }
    let dc = w.dc_offset();
    assert!(
        (dc[0] * dc[0] + dc[1] * dc[1] + dc[2] * dc[2]).sqrt() < cfg.intensity_mps,
        "the DC correction must stay small: {dc:?}"
    );
}

#[test]
fn all_values_are_finite_and_invalid_config_is_loud() {
    let cfg = enabled();
    let w = field(&cfg);
    for step in 0..STEPS {
        for alt in [-100.0, 0.0, 150.0, 5000.0] {
            let v = w.at(step, alt);
            assert!(
                v.iter().all(|x| x.is_finite()),
                "step {step} alt {alt}: {v:?}"
            );
        }
    }
    for i in 0..32 {
        let x = [i as f64 * 17.0 - 100.0, i as f64 * 3.0, i as f64 - 16.0];
        assert!(w.raw_field(x).iter().all(|v| v.is_finite()));
        assert!(w.gust_at(i as f64).is_finite());
        assert!(w.shear_at(i as f64 * 10.0).is_finite());
    }

    type Mutator = dyn Fn(&mut SpectralConfig);
    let cases: [(&str, &Mutator); 5] = [
        ("intensity_mps", &|c| c.intensity_mps = f64::NAN),
        ("integral_length_m", &|c| c.integral_length_m = 0.0),
        ("advection_speed_mps", &|c| c.advection_speed_mps = -1.0),
        ("gust_duration_s", &|c| c.gust_duration_s = 0.0),
        ("mode_count", &|c| c.mode_count = 1),
    ];
    for (name, mutate) in cases {
        let mut bad = enabled();
        mutate(&mut bad);
        let err = bad.validate().unwrap_err().to_string();
        assert!(err.contains(name), "{name} was not named in: {err}");
    }
}

/// Log-log slope of the realised 3-D energy spectrum over the inertial band.
fn inertial_slope(cfg: &SpectralConfig, seed: u64, episode: u64) -> f64 {
    let w = WindSpectral::new(cfg, seed, episode, DT, DURATION).unwrap();
    let band: Vec<(f64, f64)> = w
        .energy_spectrum(20)
        .into_iter()
        .filter(|(k_l, energy)| (3.0..=9.0).contains(k_l) && *energy > 0.0)
        .map(|(k_l, energy)| (k_l.ln(), energy.ln()))
        .collect();
    assert!(
        band.len() >= 8,
        "inertial band has only {} populated bins",
        band.len()
    );
    let xbar = band.iter().map(|(x, _)| x).sum::<f64>() / band.len() as f64;
    let ybar = band.iter().map(|(_, y)| y).sum::<f64>() / band.len() as f64;
    let num: f64 = band.iter().map(|(x, y)| (x - xbar) * (y - ybar)).sum();
    let den: f64 = band.iter().map(|(x, _)| (x - xbar).powi(2)).sum();
    num / den
}

#[test]
fn psd_slope_is_reproducible_and_matches_von_karman_inertial_range() {
    let cfg = enabled();
    let target = -5.0 / 3.0;
    let slope = inertial_slope(&cfg, SEED, EPISODE);
    assert!(
        (target - 0.4..=target + 0.4).contains(&slope),
        "inertial slope {slope} outside {target} ± 0.4"
    );
    assert_eq!(
        slope,
        inertial_slope(&cfg, SEED, EPISODE),
        "the measured slope must be reproducible"
    );
    let other = inertial_slope(&cfg, SEED + 1, EPISODE);
    assert!(
        (target - 0.4..=target + 0.4).contains(&other),
        "a different seed must still produce von Kármán turbulence, got {other}"
    );
}

#[test]
fn one_cos_gust_onset_is_bounded_and_envelope_is_exact() {
    let mut cfg = enabled();
    cfg.intensity_mps = 0.0;
    cfg.mean_wind_mps = 0.0;
    cfg.shear_delta_mps = 0.0;
    cfg.gust_amplitude_mps = 8.0;
    cfg.gust_duration_s = 2.0;
    let w = field(&cfg);

    let (t0, duration, amp) = w.gust_info().expect("a gust config must carry an onset");
    assert!((0.08 * DURATION..=0.63 * DURATION).contains(&t0), "t0 {t0}");
    assert_eq!((duration, amp), (2.0, 8.0));
    assert_eq!(w.gust_at(t0), 0.0);
    assert!(w.gust_at(t0 + duration).abs() < 1e-12);
    assert!((w.gust_at(t0 + 0.5 * duration) - amp).abs() < 1e-12);
    assert_eq!(w.gust_at(t0 - 0.1), 0.0);
    assert_eq!(w.gust_at(t0 + duration + 0.1), 0.0);

    let dir = direction(&cfg);
    let mut peak: f64 = 0.0;
    let mut active: usize = 0;
    for step in 0..STEPS {
        let g = w.gust_at(step as f64 * DT);
        assert!((0.0..=amp).contains(&g), "step {step}: gust {g}");
        peak = peak.max(g);
        if g > 0.0 {
            active += 1;
        }
        let v = w.at(step, 100.0);
        let along = v[0] * dir[0] + v[1] * dir[1];
        let across = -v[0] * dir[1] + v[1] * dir[0];
        assert!(
            across.abs() < 1e-12 && v[2].abs() < 1e-12,
            "step {step}: {v:?}"
        );
        assert!((along - g).abs() < 1e-12, "step {step}: {along} vs {g}");
    }
    assert!((peak - amp).abs() < 1e-3, "peak {peak} vs {amp}");
    let want_span = (duration / DT).round() as usize;
    assert!(
        active.abs_diff(want_span) <= 2,
        "active {active} vs {want_span}"
    );

    let mut quiet = cfg.clone();
    quiet.gust_amplitude_mps = 0.0;
    assert_eq!(field(&quiet).gust_info(), None);
}

#[test]
fn shear_layer_is_a_centered_bounded_monotone_jump() {
    let mut cfg = enabled();
    cfg.intensity_mps = 0.0;
    cfg.mean_wind_mps = 2.0;
    cfg.shear_delta_mps = 6.0;
    cfg.shear_altitude_m = 150.0;
    cfg.shear_thickness_m = 12.0;
    let w = field(&cfg);
    let mut quiet_cfg = cfg.clone();
    quiet_cfg.shear_delta_mps = 0.0;
    let quiet = field(&quiet_cfg);
    let dir = direction(&cfg);

    let want = |h: f64| 3.0 * (1.0 + ((h - 150.0) / 12.0).tanh());
    for h in [0.0, 100.0, 138.0, 150.0, 162.0, 200.0, 1000.0] {
        let got = w.shear_at(h);
        assert!((got - want(h)).abs() < 1e-12, "h {h}: {got} vs {}", want(h));
        assert!((0.0..=6.0).contains(&got));
        let v = w.at(0, h);
        let q = quiet.at(0, h);
        let along = (v[0] - q[0]) * dir[0] + (v[1] - q[1]) * dir[1];
        assert!((along - got).abs() < 1e-9, "h {h}: along {along} vs {got}");
        let across = -(v[0] - q[0]) * dir[1] + (v[1] - q[1]) * dir[0];
        assert!(across.abs() < 1e-12, "h {h}: shear left the wind direction");
    }
    assert_eq!(w.shear_at(150.0), 3.0);
    for i in 1..=80 {
        let lo = 50.0 + (i - 1) as f64 * 2.5;
        assert!(w.shear_at(lo + 2.5) > w.shear_at(lo), "must be monotone");
    }
}

#[test]
fn taylor_hypothesis_reads_the_frozen_field_at_the_advected_position() {
    let cfg = enabled();
    let w = field(&cfg);
    let dir = direction(&cfg);
    assert_eq!(w.dt(), DT);
    for step in [0, 1, 7, 100, 1000, STEPS - 1] {
        let x = w.advection_position(step);
        let t = step as f64 * DT;
        assert!((x[0] - dir[0] * cfg.advection_speed_mps * t).abs() < 1e-12);
        assert!((x[1] - dir[1] * cfg.advection_speed_mps * t).abs() < 1e-12);
        assert_eq!(x[2], 0.0);
        assert_eq!(
            w.raw_field(x),
            w.turbulence()[step],
            "step {step}: the delivered slice must be the frozen field"
        );
    }
    let mut faster = cfg.clone();
    faster.advection_speed_mps *= 2.0;
    assert_ne!(field(&faster).turbulence(), w.turbulence());
}

#[test]
fn standalone_config_is_opt_in_validated_and_accepts_both_spellings() {
    let bare = SpectralConfig::from_toml_str("enabled = true\nintensity_mps = 0.4\n").unwrap();
    assert!(bare.enabled);
    assert_eq!(bare.intensity_mps, 0.4);
    assert_eq!(bare.mode_count, SpectralConfig::default().mode_count);

    let wrapped = SpectralConfig::from_toml_str("[spectral]\nenabled = true\n").unwrap();
    assert!(wrapped.enabled);
    assert!(SpectralConfig::from_toml_str("bogus = 1\n").is_err());
}
