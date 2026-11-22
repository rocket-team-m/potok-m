//! Smart-model simulator lane: physics features, Potok-M domain randomization,
//! withheld wind/gust scenarios and versioned validation metadata.

use potok_sim::{
    scenario::{scenario, Split, SCENARIO_COUNT},
    sensor::FEATURE_NAMES,
    validation::{
        read_validation, write_validation, write_validation_with_version, FILE_NAME, META_V1,
        META_V2, TARGET_COUNT, VERSION, VERSION_V1,
    },
    Conditioning, Config, Simulator, PHYSICS_FEATURES,
};

fn smart_config() -> Config {
    let mut cfg = Config::default();
    cfg.output
        .features
        .extend(PHYSICS_FEATURES.iter().map(|s| s.to_string()));
    cfg
}

fn sim(seed: u64) -> Simulator {
    let mut cfg = smart_config();
    cfg.run.seed = seed;
    Simulator::new(cfg).unwrap()
}

#[test]
fn physics_features_are_finite_and_bounded() {
    assert_eq!(&FEATURE_NAMES[FEATURE_NAMES.len() - 6..], &PHYSICS_FEATURES);
    let cfg = smart_config();
    assert_eq!(cfg.feature_count(), 14);

    for seed in [1, 42, 1337] {
        for index in [0u64, 8, 16, 18] {
            let e = sim(seed).run_episode(index).unwrap();
            assert_eq!(e.features.len(), e.frames.len() * 14);
            assert!(e.features.iter().all(|v| v.is_finite()));
            let mut nonzero = false;
            for f in &e.frames {
                let p = f.physics;
                assert!(p.iter().all(|v| v.is_finite()), "{p:?}");
                assert!((-180.0..=180.0).contains(&p[0]), "alpha {}", p[0]);
                assert!((-180.0..=180.0).contains(&p[1]), "beta {}", p[1]);
                assert!((0.0..100.0).contains(&p[2]), "qdyn {}", p[2]);
                assert!((0.0..1000.0).contains(&p[3]), "a_lat_resid {}", p[3]);
                assert!((0.0..1000.0).contains(&p[4]), "drag_est {}", p[4]);
                assert!((0.5..1.5).contains(&p[5]), "margin_cal {}", p[5]);
                nonzero |= p[0].abs() > 1e-6 || p[1].abs() > 1e-6;
            }
            assert!(
                nonzero,
                "physics angles must come from the state, not zeros"
            );

            // Relation between derived features, not an invented constant:
            // drag_est = cd0 * qdyn(Pa) * area / current mass. The mass falls
            // as propellant burns, so lift-off mass gives the lower bound.
            let c = e.meta.conditioning;
            for (i, f) in e.frames.iter().enumerate() {
                let at_liftoff = c.cd0 * f.physics[2] * 1000.0 * c.area_m2 / c.mass_kg;
                assert!(f.physics[4] >= at_liftoff * 0.99 - 1e-6);
                assert!(f.physics[4] <= at_liftoff * 1.4 + 1e-6);
                let off = i * 14;
                assert_eq!(f.physics[0], e.features[off + 8]);
                assert_eq!(f.physics[5], e.features[off + 13]);
            }
        }
    }
}

#[test]
fn domain_randomization_stays_within_tolerances() {
    let s = sim(4242);
    let mut cd0s = Vec::new();
    let mut margins = Vec::new();
    let mut winds = Vec::new();
    for i in 0..40u64 {
        let e = s.run_episode(i).unwrap();
        let c = e.meta.conditioning;
        assert!((12.5..=13.5).contains(&(c.mass_kg as f64)));
        assert!((0.94..=1.06).contains(&e.meta.thrust_scale));
        assert!((0.30..=0.40).contains(&(c.cd0 as f64)));
        assert!((0.85..=1.10).contains(&(c.static_margin_cal as f64)));
        assert!((0.0010..=0.0020).contains(&(c.gyro_noise as f64)));
        assert!((c.area_m2 - 0.00817).abs() < 1e-9);
        assert!((c.thrust_peak_n - 436.0).abs() < 1e-6);
        assert!((0.0..=15.0).contains(&e.meta.v_10));
        assert!((0.0..=8.0).contains(&e.meta.shear_delta));
        if e.meta.gust_amplitude > 0.0 {
            assert!((5.0..=20.0).contains(&e.meta.gust_amplitude));
        }
        cd0s.push(c.cd0 as f64);
        margins.push(c.static_margin_cal as f64);
        winds.push(e.meta.v_10);
    }
    for vals in [cd0s, margins, winds] {
        assert!(
            vals.iter().any(|v| *v != vals[0]),
            "randomization is constant"
        );
    }
}

#[test]
fn held_out_scenarios_are_distinct_and_reachable() {
    let cells: Vec<_> = (0..SCENARIO_COUNT as u64).map(scenario).collect();
    let train: Vec<_> = cells
        .iter()
        .filter(|s| s.split == Split::Train)
        .map(|s| s.signature())
        .collect();
    let held: Vec<_> = cells
        .iter()
        .filter(|s| s.split == Split::HeldOut)
        .map(|s| s.signature())
        .collect();
    assert_eq!(train.len(), 16);
    assert_eq!(held.len(), 4);
    for sig in &held {
        assert!(
            !train.contains(sig),
            "withheld signature {sig:?} seen in training"
        );
    }
    assert!(cells
        .iter()
        .filter(|s| s.split == Split::HeldOut)
        .all(|s| s.lag_scale > 1.0));
    assert!(cells
        .iter()
        .filter(|s| s.split == Split::Train)
        .all(|s| s.lag_scale == 1.0));

    let s = sim(7);
    let episodes: Vec<_> = (0..20u64).map(|i| s.run_episode(i).unwrap()).collect();
    assert_eq!(
        episodes
            .iter()
            .filter(|e| e.meta.split == Split::HeldOut)
            .count(),
        4
    );
    for e in &episodes {
        let sc = scenario(e.meta.index);
        assert_eq!(e.meta.scenario_id, sc.id);
        assert_eq!(e.meta.split, sc.split);
    }
}

#[test]
fn determinism_covers_physics_features() {
    let a = sim(99).run_episode(5).unwrap();
    let b = sim(99).run_episode(5).unwrap();
    assert_eq!(a.features, b.features);
    assert_eq!(a.meta.conditioning, b.meta.conditioning);
    assert_eq!(a.meta.scenario_id, b.meta.scenario_id);
    let c = sim(100).run_episode(5).unwrap();
    assert_ne!(a.features, c.features);
}

#[test]
fn nominal_trajectory_plausible_for_both_splits() {
    let s = sim(1337);
    for index in [0u64, 16] {
        let e = s.run_episode(index).unwrap();
        assert_eq!(e.frames.len(), 1500);
        assert_eq!(e.nan_frames, 0);
        assert!(
            (500.0..6000.0).contains(&e.apogee_m),
            "index {index}: {}",
            e.apogee_m
        );
        assert!(
            (100.0..400.0).contains(&e.max_speed_mps),
            "index {index}: {}",
            e.max_speed_mps
        );
    }
}

#[test]
fn validation_v2_roundtrip_carries_conditioning_and_split() {
    let dir = std::env::temp_dir().join(format!("potok-smart-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(FILE_NAME);
    let cfg = smart_config();
    let sim = Simulator::new(cfg.clone()).unwrap();
    let episodes: Vec<_> = (16..18u64).map(|i| sim.run_episode(i).unwrap()).collect();

    write_validation(&path, &cfg, &episodes).unwrap();
    let raw = std::fs::read(&path).unwrap();
    let per_episode =
        8 + META_V2 * 4 + cfg.frames_per_episode() * (cfg.feature_count() + TARGET_COUNT) * 4;
    assert_eq!(raw.len(), 80 + episodes.len() * per_episode);
    let rs = read_validation(&path).unwrap();
    assert_eq!(rs.version, VERSION);
    assert_eq!(rs.header.meta_count as usize, META_V2);
    assert_eq!(rs.header.feature_count as usize, 14);
    for (src, dst) in episodes.iter().zip(rs.episodes.iter()) {
        assert_eq!(dst.conditioning, src.meta.conditioning);
        assert_eq!(dst.scenario_id, src.meta.scenario_id);
        assert_eq!(dst.split, Split::HeldOut);
        assert_eq!(dst.features.ncols(), 14);
    }

    // Legacy 9-metadata files remain readable.
    write_validation_with_version(&path, &cfg, &episodes, VERSION_V1).unwrap();
    let v1 = read_validation(&path).unwrap();
    assert_eq!(v1.version, VERSION_V1);
    assert_eq!(v1.header.meta_count as usize, META_V1);
    assert!(v1
        .episodes
        .iter()
        .all(|e| e.split == Split::Train && e.scenario_id == 0));
    assert_eq!(
        v1.episodes[0].conditioning,
        Conditioning {
            mass_kg: episodes[0].meta.conditioning.mass_kg,
            area_m2: 0.00817,
            cd0: 0.35,
            static_margin_cal: (0.0765 / (2.0 * (0.00817_f64 / std::f64::consts::PI).sqrt()))
                as f32,
            thrust_peak_n: 436.0,
            gyro_noise: 0.0015,
        }
    );
    std::fs::remove_file(&path).unwrap();
}
