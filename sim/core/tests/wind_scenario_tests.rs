// SPDX-License-Identifier: GPL-3.0-or-later
//! Gates for the deterministic wind field and the scenario train/withheld split.
//!
//! The field is a pure function of `(run.seed, episode)`: mean power law, tanh
//! shear jump, 1-cosine gust envelope and three seeded AR(1) (Dryden-style)
//! turbulence slices are all built once in `Wind::new`. These tests pin that
//! contract without touching any source module.

use std::f64::consts::PI;

use potok_sim::{
    config::Range, is_held_out, math::episode_seed, scenario, wind::Wind, Config, Split,
    SCENARIO_COUNT,
};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn box_muller(rng: &mut StdRng) -> f64 {
    let u1 = 1.0 - rng.gen::<f64>();
    let u2 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

fn direction(cfg: &Config) -> [f64; 2] {
    let theta = cfg.wind.direction_deg.to_radians();
    [theta.cos(), theta.sin()]
}

#[test]
fn wind_field_is_bitwise_reproducible_for_a_seed_and_episode() {
    let cfg = Config::default();
    let episode = 10; // strongest gust cell: exercises every field at once
    let a = Wind::new(&cfg, episode);
    let b = Wind::new(&cfg, episode);

    assert_eq!(a.v_10(), b.v_10());
    assert_eq!(a.shear_delta(), b.shear_delta());
    assert_eq!(a.shear_altitude(), b.shear_altitude());
    assert_eq!(a.shear_thickness(), b.shear_thickness());
    assert_eq!(a.gust_info(), b.gust_info());
    assert_eq!(a.v_ref(), b.v_ref());
    for step in 0..cfg.steps() {
        assert_eq!(
            a.gust_active_step(step),
            b.gust_active_step(step),
            "gust flag at step {step}"
        );
    }
    for step in (0..cfg.steps()).step_by(37) {
        for alt in [0.0, 120.0, 1_000.0, 3_000.0] {
            for factor in [0.0, 0.5, 1.0, 1.7] {
                assert_eq!(
                    a.at(step, alt, factor),
                    b.at(step, alt, factor),
                    "step {step}, alt {alt}, factor {factor}"
                );
            }
        }
    }
}

#[test]
fn wind_field_changes_with_seed_and_episode() {
    let cfg = Config::default();
    let sample = |w: &Wind| -> Vec<[f64; 3]> { (0..64).map(|k| w.at(k, 200.0, 1.0)).collect() };
    let base = sample(&Wind::new(&cfg, 10));

    let mut other_seed = cfg.clone();
    other_seed.run.seed = cfg.run.seed.wrapping_add(1);
    assert_ne!(
        base,
        sample(&Wind::new(&other_seed, 10)),
        "seed must enter the field"
    );
    assert_ne!(
        base,
        sample(&Wind::new(&cfg, 11)),
        "episode must enter the field"
    );
}

#[test]
fn mean_profile_follows_power_law_direction_and_wind_factor() {
    let mut cfg = Config::default();
    cfg.wind.shear_delta_range = Range(0.0, 0.0);
    cfg.wind.gust_amplitude_range = Range(0.0, 0.0);
    cfg.wind.turbulence_sigma = 0.0;
    let episode = 1; // v10_u = 0.20, gust_u = 0
    let w = Wind::new(&cfg, episode);

    let v10 = cfg.wind.v_10_range.sample(scenario(episode).v10_u);
    let dir = direction(&cfg);
    assert_eq!(w.v_10(), v10);
    assert_eq!(w.gust_info(), None);
    assert_eq!(w.shear_delta(), 0.0);
    assert_eq!(w.v_ref(), v10.abs().max(2.0));

    let cases: [(f64, f64); 6] = [
        (10.0, 1.0),
        (100.0, 1.0),
        (1_000.0, 1.0),
        (100.0, 2.0),
        (100.0, -1.0),
        (-100.0, 1.0),
    ];
    for (alt, factor) in cases {
        let h = alt.max(0.1);
        let want = v10 * (h / 10.0).powf(cfg.wind.shear_exponent) * factor.max(0.0);
        let got = w.at(7, alt, factor);
        assert!(
            (got[0] - want * dir[0]).abs() < 1e-9 && (got[1] - want * dir[1]).abs() < 1e-9,
            "alt {alt}, factor {factor}: {got:?} vs speed {want}"
        );
        assert_eq!(got[2], 0.0);
    }
}

#[test]
fn shear_layer_is_a_centered_tanh_jump() {
    let mut cfg = Config::default();
    cfg.wind.v_10_range = Range(0.0, 0.0);
    cfg.wind.gust_amplitude_range = Range(0.0, 0.0);
    cfg.wind.turbulence_sigma = 0.0;
    let episode = 1; // shear_u = 0.25, shear_alt_u = 0.20
    let w = Wind::new(&cfg, episode);
    let sc = scenario(episode);

    let delta = cfg.wind.shear_delta_range.sample(sc.shear_u);
    let center = cfg.wind.shear_altitude_range.sample(sc.shear_alt_u);
    let thick = cfg.wind.shear_thickness;
    assert_eq!(w.shear_delta(), delta);
    assert_eq!(w.shear_altitude(), center);
    assert_eq!(w.shear_thickness(), thick);

    let dir = direction(&cfg);
    let speed = |alt: f64| {
        let v = w.at(0, alt, 1.0);
        v[0] * dir[0] + v[1] * dir[1]
    };
    let tanh_profile = |h: f64| 0.5 * delta * (1.0 + ((h - center) / thick).tanh());

    for h in [
        center - 200.0,
        center - 2.0 * thick,
        center - thick,
        center,
        center + thick,
        center + 2.0 * thick,
        center + 200.0,
    ] {
        assert!(
            (speed(h) - tanh_profile(h)).abs() < 1e-6,
            "h {h}: got {} want {}",
            speed(h),
            tanh_profile(h)
        );
    }
    assert!(speed(center - 200.0).abs() < 1e-6, "below the layer");
    assert!(
        (speed(center + 200.0) - delta).abs() < 1e-6,
        "above the layer"
    );
    assert!((speed(center) - 0.5 * delta).abs() < 1e-9, "layer midpoint");
    for d in [1.0, thick, 2.0 * thick, 50.0] {
        assert!(
            (speed(center - d) + speed(center + d) - delta).abs() < 1e-9,
            "tanh symmetry at ±{d}"
        );
    }
    for i in 1..=80 {
        let lo = center - 100.0 + (i - 1) as f64 * 2.5;
        assert!(speed(lo + 2.5) > speed(lo), "shear jump must be monotone");
    }
}

#[test]
fn shear_active_band_matches_twice_the_thickness() {
    let w = Wind::new(&Config::default(), 1);
    let center = w.shear_altitude();
    let band = 2.0 * w.shear_thickness();
    assert!(w.shear_active(center));
    assert!(w.shear_active(center - band));
    assert!(w.shear_active(center + band));
    assert!(!w.shear_active(center - band - 0.1));
    assert!(!w.shear_active(center + band + 0.1));
    assert!(!w.shear_active(0.0));
    assert!(!w.shear_active(10_000.0));
}

#[test]
fn one_cos_gust_envelope_is_zero_at_ends_and_full_at_midpoint() {
    let cfg = Config::default();
    let episode = 10; // gust_u = 1.0
    let w = Wind::new(&cfg, episode);
    let (t0, duration, amp) = w.gust_info().expect("a gust scenario must carry a gust");
    assert_eq!(
        amp,
        cfg.wind
            .gust_amplitude_range
            .sample(scenario(episode).gust_u)
    );
    assert_eq!(duration, cfg.wind.gust_duration);
    assert!(t0 >= 0.08 * cfg.run.duration_s);
    assert!(t0 <= 0.63 * cfg.run.duration_s);

    let mut quiet = cfg.clone();
    quiet.wind.gust_amplitude_range = Range(0.0, 0.0);
    let no_gust = Wind::new(&quiet, episode); // same RNG draw order, zero amplitude

    let dir = direction(&cfg);
    let mut peak = 0.0_f64;
    let mut active = Vec::new();
    for step in 0..cfg.steps() {
        let t = step as f64 * cfg.run.dt;
        let u = (t - t0) / duration;
        let inside = (0.0..=1.0).contains(&u);
        assert_eq!(w.gust_active_step(step), inside, "flag at step {step}");

        let gusty = w.at(step, 200.0, 1.0);
        let calm = no_gust.at(step, 200.0, 1.0);
        let (dx, dy, dz) = (gusty[0] - calm[0], gusty[1] - calm[1], gusty[2] - calm[2]);
        let along = dx * dir[0] + dy * dir[1];
        let across = -dx * dir[1] + dy * dir[0];
        assert!(
            across.abs() < 1e-9 && dz.abs() < 1e-9,
            "gust must stay along-wind"
        );
        let want = if inside {
            0.5 * amp * (1.0 - (2.0 * PI * u).cos())
        } else {
            0.0
        };
        assert!(
            (along - want).abs() < 1e-3,
            "step {step}, u {u}: got {along} want {want}"
        );
        peak = peak.max(along);
        if inside {
            active.push(step);
        }
    }
    assert!(
        (peak - amp).abs() < 1e-3,
        "1-cos peak {peak} vs amplitude {amp}"
    );
    let span = active.last().unwrap() - active.first().unwrap();
    let want_span = (duration / cfg.run.dt).round() as usize;
    assert!(
        span.abs_diff(want_span) <= 2,
        "active span {span} vs {want_span}"
    );
}

#[test]
fn non_gust_scenarios_never_arm_the_gust() {
    let cfg = Config::default();
    for episode in [0u64, 1, 3, 9, 14] {
        assert_eq!(scenario(episode).gust_u, 0.0);
        let w = Wind::new(&cfg, episode);
        assert_eq!(w.gust_info(), None);
        assert!((0..cfg.steps()).all(|k| !w.gust_active_step(k)));
    }
}

#[test]
fn turbulence_slices_are_bitwise_reproducible_and_seed_sensitive() {
    let mut cfg = Config::default();
    cfg.wind.v_10_range = Range(0.0, 0.0);
    cfg.wind.shear_delta_range = Range(0.0, 0.0);
    cfg.wind.gust_amplitude_range = Range(0.0, 0.0); // t0 is still drawn, as in every gust cell
    cfg.wind.turbulence_sigma = 0.8;
    let episode = 4; // gust_u = 0.6
    let sample = |w: &Wind| -> Vec<[f64; 3]> { (0..256).map(|k| w.at(k, 500.0, 1.0)).collect() };

    let a = Wind::new(&cfg, episode);
    let base = sample(&a);
    assert_eq!(base, sample(&Wind::new(&cfg, episode)));
    assert!(base.iter().all(|v| v.iter().all(|x| x.is_finite())));

    let mut other_seed = cfg.clone();
    other_seed.run.seed = cfg.run.seed.wrapping_add(1);
    assert_ne!(base, sample(&Wind::new(&other_seed, episode)));
    assert_ne!(base, sample(&Wind::new(&cfg, episode + 1)));
}

#[test]
fn turbulence_is_a_correlated_process_not_white_noise() {
    let mut cfg = Config::default();
    cfg.wind.v_10_range = Range(15.0, 15.0);
    cfg.wind.shear_delta_range = Range(0.0, 0.0);
    cfg.wind.gust_amplitude_range = Range(0.0, 0.0);
    cfg.wind.turbulence_sigma = 1.0;
    let w = Wind::new(&cfg, 0);
    let series: Vec<[f64; 3]> = (0..cfg.steps()).map(|k| w.at(k, 500.0, 1.0)).collect();

    for component in 0..3 {
        let mean = series.iter().map(|v| v[component]).sum::<f64>() / series.len() as f64;
        let centered: Vec<f64> = series.iter().map(|v| v[component] - mean).collect();
        let variance: f64 = centered.iter().map(|x| x * x).sum::<f64>() / centered.len() as f64;
        assert!(variance > 0.0, "component {component} must vary");
        let autocorr = |lag: usize| -> f64 {
            centered[..centered.len() - lag]
                .iter()
                .zip(&centered[lag..])
                .map(|(a, b)| a * b)
                .sum::<f64>()
                / (centered.len() - lag) as f64
                / variance
        };
        let lag1 = autocorr(1);
        let lag200 = autocorr(200);
        assert!(lag1 > 0.99, "component {component}: lag-1 {lag1}");
        assert!(
            lag1 > lag200,
            "component {component}: {lag1} vs lag-200 {lag200}"
        );
    }
}

#[test]
fn dryden_style_turbulence_matches_the_seeded_ar1_reference() {
    let mut cfg = Config::default();
    cfg.wind.v_10_range = Range(0.0, 0.0);
    cfg.wind.shear_delta_range = Range(0.0, 0.0);
    cfg.wind.gust_amplitude_range = Range(0.0, 0.0);
    cfg.wind.turbulence_sigma = 0.7;
    let episode = 0u64; // no gust draw before the turbulence init
    assert_eq!(scenario(episode).gust_u, 0.0);
    let w = Wind::new(&cfg, episode);
    assert_eq!(w.v_ref(), 2.0);

    let mut rng = StdRng::seed_from_u64(episode_seed(cfg.run.seed, episode ^ 0x5749_4E44));
    let scales = [1.0_f64, 0.8, 0.45];
    let mut gain = [0.0; 3];
    let mut decay = [0.0; 3];
    for i in 0..3 {
        let length = (cfg.wind.turbulence_scale * (0.6 + 0.5 * scales[i])).max(1.0);
        decay[i] = w.v_ref() / length;
        gain[i] = cfg.wind.turbulence_sigma * scales[i];
    }
    let mut x = [0.0; 3];
    for i in 0..3 {
        x[i] = box_muller(&mut rng) * gain[i];
    }
    for step in 0..128 {
        let got = w.at(step, 500.0, 1.0);
        for i in 0..3 {
            assert!(
                (got[i] - x[i]).abs() <= 1e-12,
                "step {step}, component {i}: {got:?} vs {x:?}"
            );
        }
        for i in 0..3 {
            let a = (-decay[i] * cfg.run.dt).exp();
            let b = gain[i] * (1.0 - a * a).max(0.0).sqrt();
            x[i] = a * x[i] + b * box_muller(&mut rng);
        }
    }
}

#[test]
fn withheld_split_boundary_is_index_16_and_cycles_with_the_cells() {
    assert_eq!(SCENARIO_COUNT, 20);
    for index in 0..SCENARIO_COUNT as u64 {
        let sc = scenario(index);
        assert_eq!(sc.id as u64, index);
        assert_eq!(
            sc.split,
            if index < 16 {
                Split::Train
            } else {
                Split::HeldOut
            }
        );
        assert_eq!(is_held_out(index), index >= 16);
        assert_eq!(scenario(index + SCENARIO_COUNT as u64), sc);
        assert_eq!(scenario(index + 3 * SCENARIO_COUNT as u64), sc);
    }
    assert_eq!(scenario(15).split, Split::Train);
    assert_eq!(scenario(16).split, Split::HeldOut);
}

#[test]
fn withheld_cells_are_unseen_extreme_combinations() {
    let train: Vec<_> = (0..16).map(scenario).collect();
    let held: Vec<_> = (16..20).map(scenario).collect();

    assert!(train
        .iter()
        .all(|s| s.split == Split::Train && s.lag_scale == 1.0));
    assert!(held
        .iter()
        .all(|s| s.split == Split::HeldOut && s.lag_scale > 1.0));
    for h in &held {
        assert!(
            !train.iter().any(|t| t.signature() == h.signature()),
            "withheld cell {} repeats a training scenario",
            h.id
        );
        assert!(
            h.v10_u >= 0.8 && h.gust_u >= 0.8 && h.shear_u >= 0.9,
            "withheld cell {} is not an extreme combination: {h:?}",
            h.id
        );
    }
    assert!(held.iter().any(|s| s.signature() == (1.0, 1.0, 1.0, 1.0)));

    // The withheld cell must actually drive the wind field to those extremes.
    let cfg = Config::default();
    let sc = scenario(16);
    let w = Wind::new(&cfg, 16);
    assert_eq!(w.v_10(), cfg.wind.v_10_range.sample(sc.v10_u));
    assert_eq!(
        w.shear_delta(),
        cfg.wind.shear_delta_range.sample(sc.shear_u)
    );
    assert_eq!(
        w.shear_altitude(),
        cfg.wind.shear_altitude_range.sample(sc.shear_alt_u)
    );
    assert_eq!(
        w.gust_info().unwrap().2,
        cfg.wind.gust_amplitude_range.sample(sc.gust_u)
    );

    for (split, code, label) in [
        (Split::Train, 0u8, "train"),
        (Split::HeldOut, 1, "held-out"),
    ] {
        assert_eq!(split.as_u8(), code);
        assert_eq!(Split::from_u8(code), Some(split));
        assert_eq!(split.label(), label);
    }
    assert_eq!(Split::from_u8(2), None);
}
