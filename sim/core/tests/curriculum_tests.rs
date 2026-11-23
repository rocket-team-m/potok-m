// SPDX-License-Identifier: GPL-3.0-or-later
//! WORLD05 curriculum gates for the four knobs that used to be fixed in every
//! training cell: sensor lag scale, turbulence sigma/length, 1-cos gust
//! duration and gust direction sign.
//!
//! These tests pin three things without touching the scenario cells, the wind
//! generator, the sensor or any file layout:
//!   1. the training grid actually varies every knob, per episode;
//!   2. each knob has its own substream seed, so retuning one cannot move the
//!      others or the legacy INIT/DOMA/WIND/SENS draws;
//!   3. held-out cells stay deterministic and disjoint from the training
//!      support on every magnitude knob (their lag OOD comes from the cell).
//! A red test here means a knob silently went back to being fixed, or the
//! train/held-out split leaked. No fallback is acceptable.

use std::collections::HashSet;

use potok_sim::{
    config::{
        episode_config, sample_curriculum, CurriculumKnobs, Range, DR_GUST_DIR, DR_GUST_DUR,
        DR_LAG, DR_TURB_SCALE, DR_TURB_SIGMA,
    },
    math::episode_seed,
    scenario, Config, Simulator, Split, Wind, SCENARIO_COUNT,
};

const INIT: u64 = 0x494E_4954;
const DOMA: u64 = 0x444F_4D41;
const WIND: u64 = 0x5749_4E44;
const SENS: u64 = 0x5345_4E53;

fn distinct(values: &[f64]) -> usize {
    values
        .iter()
        .map(|v| v.to_bits())
        .collect::<HashSet<_>>()
        .len()
}

fn within(v: f64, lo: f64, hi: f64) -> bool {
    v >= lo - 1e-12 && v <= hi + 1e-12
}

#[test]
fn training_grid_varies_every_previously_fixed_knob() {
    let cfg = Config::default();
    let dr = &cfg.curriculum;
    // Four cycles of the scenario grid: every training cell is revisited with
    // a different episode index, so a per-cell constant would be exposed.
    let train: Vec<(u64, CurriculumKnobs)> = (0..(4 * SCENARIO_COUNT as u64))
        .filter(|i| scenario(*i).split == Split::Train)
        .map(|i| (i, sample_curriculum(&cfg, i)))
        .collect();
    assert_eq!(train.len(), 64, "16 training cells over 4 cycles");

    let lags: Vec<f64> = train.iter().map(|(_, k)| k.lag_scale).collect();
    let sigmas: Vec<f64> = train.iter().map(|(_, k)| k.turbulence_sigma).collect();
    let scales: Vec<f64> = train.iter().map(|(_, k)| k.turbulence_scale).collect();
    let durations: Vec<f64> = train.iter().map(|(_, k)| k.gust_duration_s).collect();
    let signs: Vec<f64> = train.iter().map(|(_, k)| k.direction_sign).collect();

    assert!(distinct(&lags) >= 8, "lag_scale barely varies: {lags:?}");
    assert!(distinct(&sigmas) >= 8, "turbulence sigma barely varies");
    assert!(distinct(&scales) >= 8, "turbulence scale barely varies");
    assert!(distinct(&durations) >= 8, "gust duration barely varies");
    assert!(
        signs.iter().all(|s| *s == -1.0 || *s == 1.0),
        "direction sign must be exactly +/-1"
    );
    assert!(
        signs.contains(&-1.0) && signs.contains(&1.0),
        "both gust direction signs must appear in training"
    );
    let mirrored = signs.iter().filter(|s| **s == -1.0).count();
    assert!(
        mirrored > 0 && mirrored * 2 < signs.len(),
        "the shipped +25 deg nominal must stay the majority sign"
    );

    assert!(lags
        .iter()
        .all(|v| within(*v, dr.lag_scale_range.0, dr.lag_scale_range.1)));
    assert!(sigmas.iter().all(|v| {
        within(
            *v,
            cfg.wind.turbulence_sigma * dr.sigma_factor_range.0,
            cfg.wind.turbulence_sigma * dr.sigma_factor_range.1,
        )
    }));
    assert!(scales.iter().all(|v| {
        within(
            *v,
            cfg.wind.turbulence_scale * dr.scale_factor_range.0,
            cfg.wind.turbulence_scale * dr.scale_factor_range.1,
        )
    }));
    assert!(durations.iter().all(|v| {
        within(
            *v,
            cfg.wind.gust_duration * dr.duration_factor_range.0,
            cfg.wind.gust_duration * dr.duration_factor_range.1,
        )
    }));

    // Same scenario cell, different episode: the knobs must moved per draw.
    let cell = SCENARIO_COUNT as u64;
    let repeats: Vec<CurriculumKnobs> = (0..4)
        .map(|c| sample_curriculum(&cfg, 8 + c * cell))
        .collect();
    assert!(distinct(&repeats.iter().map(|k| k.lag_scale).collect::<Vec<_>>()) > 1);
    assert!(
        distinct(
            &repeats
                .iter()
                .map(|k| k.turbulence_sigma)
                .collect::<Vec<_>>()
        ) > 1
    );
    assert!(
        distinct(
            &repeats
                .iter()
                .map(|k| k.turbulence_scale)
                .collect::<Vec<_>>()
        ) > 1
    );
    assert!(
        distinct(
            &repeats
                .iter()
                .map(|k| k.gust_duration_s)
                .collect::<Vec<_>>()
        ) > 1
    );
}

#[test]
fn calm_replay_anchor_stays_at_shipped_nominal() {
    let cfg = Config::default();
    for index in [0u64, SCENARIO_COUNT as u64, 3 * SCENARIO_COUNT as u64] {
        let sc = scenario(index);
        assert_eq!(sc.split, Split::Train);
        assert_eq!(
            sample_curriculum(&cfg, index),
            CurriculumKnobs {
                lag_scale: 1.0,
                turbulence_sigma: cfg.wind.turbulence_sigma,
                turbulence_scale: cfg.wind.turbulence_scale,
                gust_duration_s: cfg.wind.gust_duration,
                direction_sign: 1.0,
            },
            "the calm replay anchor must not drift with the curriculum"
        );
    }
}

#[test]
fn knobs_are_deterministic_and_each_owns_its_substream() {
    let cfg = Config::default();

    let tags = [
        DR_LAG,
        DR_TURB_SIGMA,
        DR_TURB_SCALE,
        DR_GUST_DUR,
        DR_GUST_DIR,
        INIT,
        DOMA,
        WIND,
        SENS,
    ];
    assert_eq!(
        tags.iter().copied().collect::<HashSet<_>>().len(),
        tags.len(),
        "curriculum tags must not alias each other or the legacy substreams"
    );

    let a = sample_curriculum(&cfg, 8);
    assert_eq!(
        a,
        sample_curriculum(&cfg, 8),
        "same seed/index must be bitwise stable"
    );
    let mut other_seed = cfg.clone();
    other_seed.run.seed = cfg.run.seed.wrapping_add(1);
    assert_ne!(
        a,
        sample_curriculum(&other_seed, 8),
        "seed must move every draw"
    );

    // Moving one range must leave every other knob bit-identical: that is what
    // a per-knob substream buys. Range(1,1) makes the moved knob exact.
    let mut lag_cfg = cfg.clone();
    lag_cfg.curriculum.lag_scale_range = Range(1.0, 1.0);
    let lag_only = sample_curriculum(&lag_cfg, 8);
    assert_eq!(lag_only.lag_scale, 1.0);
    assert_eq!(lag_only.turbulence_sigma, a.turbulence_sigma);
    assert_eq!(lag_only.turbulence_scale, a.turbulence_scale);
    assert_eq!(lag_only.gust_duration_s, a.gust_duration_s);
    assert_eq!(lag_only.direction_sign, a.direction_sign);

    let mut sigma_cfg = cfg.clone();
    sigma_cfg.curriculum.sigma_factor_range = Range(1.0, 1.0);
    let sigma_only = sample_curriculum(&sigma_cfg, 8);
    assert_eq!(sigma_only.turbulence_sigma, cfg.wind.turbulence_sigma);
    assert_eq!(sigma_only.lag_scale, a.lag_scale);
    assert_eq!(sigma_only.turbulence_scale, a.turbulence_scale);
    assert_eq!(sigma_only.gust_duration_s, a.gust_duration_s);
    assert_eq!(sigma_only.direction_sign, a.direction_sign);

    // The tags are real SplitMix64 keys, not decorative constants.
    for tag in [
        DR_LAG,
        DR_TURB_SIGMA,
        DR_TURB_SCALE,
        DR_GUST_DUR,
        DR_GUST_DIR,
    ] {
        let mut keys = HashSet::new();
        for index in 0..64u64 {
            assert!(keys.insert(episode_seed(cfg.run.seed, index ^ tag)));
        }
        assert_eq!(keys.len(), 64);
    }
}

#[test]
fn held_out_knobs_are_ood_disjoint_and_deterministic() {
    let cfg = Config::default();
    let dr = &cfg.curriculum;
    let train: Vec<CurriculumKnobs> = (0..16u64).map(|i| sample_curriculum(&cfg, i)).collect();
    let held: Vec<(u64, CurriculumKnobs)> = (16..20u64)
        .map(|i| (i, sample_curriculum(&cfg, i)))
        .collect();

    let t_sigma_hi = train
        .iter()
        .map(|k| k.turbulence_sigma)
        .fold(f64::NEG_INFINITY, f64::max);
    let t_scale_lo = train
        .iter()
        .map(|k| k.turbulence_scale)
        .fold(f64::INFINITY, f64::min);
    let t_dur_hi = train
        .iter()
        .map(|k| k.gust_duration_s)
        .fold(f64::NEG_INFINITY, f64::max);
    let t_lag_hi = train
        .iter()
        .map(|k| k.lag_scale)
        .fold(f64::NEG_INFINITY, f64::max);
    let held_lag_floor = (16..20u64)
        .map(scenario)
        .map(|s| s.lag_scale)
        .fold(f64::INFINITY, f64::min);
    assert!(
        t_lag_hi < held_lag_floor,
        "training lag must stay below held-out cell lag"
    );

    let train_sigs: Vec<_> = (0..16u64).map(|i| scenario(i).signature()).collect();
    for (index, k) in &held {
        let sc = scenario(*index);
        assert_eq!(sc.split, Split::HeldOut);
        assert!(
            !train_sigs.contains(&sc.signature()),
            "held-out cell repeated a training scenario"
        );

        // The curriculum lag multiplier is 1 on held-out: the cell's own
        // lag_scale is the OOD twist, so the effective lag is still disjoint.
        assert_eq!(k.lag_scale, 1.0);

        assert!(
            k.turbulence_sigma > t_sigma_hi,
            "held-out sigma overlaps training"
        );
        assert!(within(
            k.turbulence_sigma,
            cfg.wind.turbulence_sigma * dr.heldout_sigma_factor_range.0,
            cfg.wind.turbulence_sigma * dr.heldout_sigma_factor_range.1,
        ));
        assert!(
            k.turbulence_scale < t_scale_lo,
            "held-out turbulence scale overlaps training"
        );
        assert!(within(
            k.turbulence_scale,
            cfg.wind.turbulence_scale * dr.heldout_scale_factor_range.0,
            cfg.wind.turbulence_scale * dr.heldout_scale_factor_range.1,
        ));
        assert!(
            k.gust_duration_s > t_dur_hi,
            "held-out gust duration overlaps training"
        );
        assert!(within(
            k.gust_duration_s,
            cfg.wind.gust_duration * dr.heldout_duration_factor_range.0,
            cfg.wind.gust_duration * dr.heldout_duration_factor_range.1,
        ));

        let (ep, applied) = episode_config(&cfg, *index);
        assert_eq!(applied, *k);
        assert_eq!(ep.wind.turbulence_sigma, k.turbulence_sigma);
        assert_eq!(ep.wind.gust_duration, k.gust_duration_s);
        // Held-out cells carry the OOD lag themselves; Sensor::new multiplies
        // this base by sc.lag_scale (1.5-2.0) on top of the 1.0 curriculum draw.
        assert_eq!(ep.sensor.lag_tau, cfg.sensor.lag_tau);
        assert!(sc.lag_scale >= 1.5);
    }

    // Cycling the scenario grid revisits the same held-out cell with a new
    // episode index: the draw moves, but it must stay in the OOD support.
    for cycle in 1..4u64 {
        for index in 16..20u64 {
            let i = index + cycle * SCENARIO_COUNT as u64;
            let k = sample_curriculum(&cfg, i);
            assert_eq!(
                k,
                sample_curriculum(&cfg, i),
                "held-out draw must be deterministic"
            );
            assert_eq!(k.lag_scale, 1.0);
            assert!(k.turbulence_sigma > t_sigma_hi);
            assert!(k.turbulence_scale < t_scale_lo);
            assert!(k.gust_duration_s > t_dur_hi);
        }
    }
}

#[test]
fn knobs_reach_the_wind_sensor_and_episode_meta() {
    let cfg = Config::default();
    let index = 8u64;
    let (ep, knobs) = episode_config(&cfg, index);

    assert_eq!(ep.wind.gust_duration, knobs.gust_duration_s);
    assert_eq!(ep.wind.turbulence_sigma, knobs.turbulence_sigma);
    assert_eq!(ep.wind.turbulence_scale, knobs.turbulence_scale);
    assert_eq!(
        ep.wind.direction_deg,
        cfg.wind.direction_deg * knobs.direction_sign
    );
    assert_eq!(ep.sensor.lag_tau, cfg.sensor.lag_tau * knobs.lag_scale);

    let wind = Wind::new(&ep, index);
    assert_eq!(
        wind.gust_info().expect("cell 8 has a gust").1,
        knobs.gust_duration_s,
        "the swept gradient must reach the 1-cos field"
    );

    // Freeze gust amplitude and turbulence, leaving only the mean wind, so the
    // direction sign is observable from the wind vector itself.
    let mut probe = ep.clone();
    probe.wind.gust_amplitude_range = Range(0.0, 0.0);
    probe.wind.turbulence_sigma = 0.0;
    let wind = Wind::new(&probe, index);
    let v = wind.at(5, 300.0, 1.0);
    let theta = probe.wind.direction_deg.to_radians();
    assert!(
        (v[1] / v[0] - theta.tan()).abs() < 1e-9,
        "wind direction {v:?} does not match {theta}"
    );

    let episode = Simulator::new(cfg).unwrap().run_episode(index).unwrap();
    assert_eq!(
        episode.meta.curriculum, knobs,
        "episode meta must report the applied knobs"
    );
    assert_eq!(episode.meta.gust_duration_s, knobs.gust_duration_s);
}

#[test]
fn overlapping_held_out_support_is_a_loud_config_error() {
    assert!(Config::default().validate().is_ok());

    let cfg = Config::default();
    let dr = &cfg.curriculum;

    let mut sigma = cfg.clone();
    sigma.curriculum.heldout_sigma_factor_range = dr.sigma_factor_range;
    assert!(
        sigma.validate().is_err(),
        "overlapping sigma support must be refused"
    );

    let mut scale = cfg.clone();
    scale.curriculum.heldout_scale_factor_range = dr.scale_factor_range;
    assert!(
        scale.validate().is_err(),
        "overlapping scale support must be refused"
    );

    let mut duration = cfg.clone();
    duration.curriculum.heldout_duration_factor_range = dr.duration_factor_range;
    assert!(
        duration.validate().is_err(),
        "overlapping duration support must be refused"
    );

    let mut lag = cfg.clone();
    lag.curriculum.lag_scale_range = Range(1.0, 1.5);
    assert!(
        lag.validate().is_err(),
        "training lag reaching the held-out floor must be refused"
    );
}
