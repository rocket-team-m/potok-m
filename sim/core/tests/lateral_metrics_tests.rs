// SPDX-License-Identifier: GPL-3.0-or-later
//! CHUTE02 gates for the lateral-metric split.
//!
//! The legacy `lateral_rms_m` covers the whole flown trace. The two explicit
//! metrics split it on canopy deployment truth carried in a new optional
//! sidecar: fin-flight RMS is AI-influenceable, landing dispersion is wind
//! drift the AI cannot touch. Truth absence is a LOUD error, never a guess.

use std::path::PathBuf;

use potok_sim::{
    trajectory::{
        lateral_flight_contributions, lateral_rms_at_deployment_m, lateral_rms_m,
        landing_dispersion_m,
    },
    validation::{
        deployment_sidecar_path, read_deployment_truth, write_deployment_truth, DeploymentTruth,
        DEPLOYMENT_HEADER_SIZE, DEPLOYMENT_RECORD_SIZE, DEPLOYMENT_VERSION,
    },
    TrajectoryTick,
};

fn tick(r: f32, z: f32) -> TrajectoryTick {
    TrajectoryTick {
        pos: [r, 0.0, z],
        ..Default::default()
    }
}

fn truth(boost_end_tick: u32, deployment_tick: u32, landing_xy: [f32; 2]) -> DeploymentTruth {
    DeploymentTruth {
        index: 7,
        boost_end_tick,
        deployment_tick,
        landing_xy,
        wind_mean_xy: [4.0, -2.0],
    }
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("potok-chute02-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// LATERAL01 §2.1 decomposition of the shipped baseline: boost 1.89 % of the
/// fin-flight RMS², coast+descent (fin-active) 98.11 %. Pin both the direction
/// and the magnitude so a phase-gate regression cannot hide inside 1528 m.
#[test]
fn boost_contribution_is_small_and_fin_active_dominates() {
    let mut ticks = Vec::with_capacity(30_000);
    for _ in 0..20 {
        ticks.push(tick(0.0, 0.0)); // PAD
    }
    for _ in 0..9_880 {
        ticks.push(tick(366.51, 1_500.0)); // BOOST, AI forbidden
    }
    for _ in 0..10_078 {
        ticks.push(tick(1_507.93, 2_500.0)); // COAST, AI allowed
    }
    for _ in 0..10_022 {
        ticks.push(tick(2_138.54, 1_700.0)); // DESCENT, fins still active
    }
    let t = truth(9_899, 29_999, [2_500.0, 0.0]);
    let c = lateral_flight_contributions(&ticks, &t).unwrap();
    let total = c.boost_m2 + c.fin_active_m2;
    let boost_share = c.boost_m2 / total;
    let fin_share = c.fin_active_m2 / total;
    eprintln!(
        "at deployment {:.3} m, boost {:.4} %, fin-active {:.4} %",
        c.at_deployment_m(),
        boost_share * 100.0,
        fin_share * 100.0
    );
    assert!(boost_share < 0.05, "boost contributes too much: {boost_share}");
    assert!(
        fin_share > 0.95,
        "fin-active ticks must dominate: {fin_share}"
    );
    assert!(
        (c.at_deployment_m() - 1_528.37).abs() < 0.5,
        "expected LATERAL01 baseline RMS, got {}",
        c.at_deployment_m()
    );
    assert_eq!(lateral_rms_at_deployment_m(&ticks, Some(&t)).unwrap(), total.sqrt());
}

/// The fin-flight metric stops at deployment; the legacy metric does not.
#[test]
fn deployment_metric_is_a_prefix_of_the_legacy_metric() {
    let ticks = vec![
        tick(100.0, 1_000.0),
        tick(200.0, 900.0),
        tick(300.0, 800.0),
        tick(900.0, 700.0),
    ];
    let t = truth(0, 1, [300.0, 0.0]); // fin flight is ticks 0..=1
    let at = lateral_rms_at_deployment_m(&ticks, Some(&t)).unwrap();
    assert!((at - 158.113_883).abs() < 1e-3, "got {at}");
    assert!(lateral_rms_m(&ticks) > at, "legacy metric must still span all ticks");
}

/// Changing the AI's fin-flight trajectory moves the deployment metric but
/// cannot move landing dispersion: that is fixed by deployment and landing
/// truth alone. Here both arms deploy at (300, 0) and land at (420, -160).
#[test]
fn landing_dispersion_is_post_canopy_wind_drift_only() {
    let good = vec![tick(10.0, 900.0), tick(20.0, 800.0), tick(300.0, 700.0)];
    let bad = vec![tick(5_000.0, 900.0), tick(20.0, 800.0), tick(300.0, 700.0)];
    let t = truth(0, 2, [420.0, -160.0]);
    let d_good = landing_dispersion_m(&good, Some(&t)).unwrap();
    let d_bad = landing_dispersion_m(&bad, Some(&t)).unwrap();
    assert!((d_good - 200.0).abs() < 1e-6, "got {d_good}");
    assert_eq!(d_good, d_bad, "fin-flight changes must not move landing drift");
    let r_good = lateral_rms_at_deployment_m(&good, Some(&t)).unwrap();
    let r_bad = lateral_rms_at_deployment_m(&bad, Some(&t)).unwrap();
    assert!(r_bad > r_good + 1_000.0, "fin authority must move the deployment metric");
}

/// No canopy phase exists in the flown trace: asking for a split metric
/// without the sidecar must fail loudly and name the way to supply it.
#[test]
fn missing_or_malformed_truth_is_loud() {
    let ticks = vec![tick(10.0, 900.0), tick(20.0, 800.0), tick(300.0, 700.0)];
    let err = lateral_rms_at_deployment_m(&ticks, None).unwrap_err().to_string();
    assert!(err.contains("missing canopy deployment truth"), "{err}");
    assert!(err.contains("write_deployment_truth"), "{err}");
    let err = landing_dispersion_m(&ticks, None).unwrap_err().to_string();
    assert!(err.contains("missing canopy deployment truth"), "{err}");

    let beyond = truth(0, 999, [0.0, 0.0]);
    let err = lateral_rms_at_deployment_m(&ticks, Some(&beyond))
        .unwrap_err()
        .to_string();
    assert!(err.contains("outside"), "{err}");

    let inverted = truth(2, 2, [0.0, 0.0]);
    let err = lateral_flight_contributions(&ticks, &inverted)
        .unwrap_err()
        .to_string();
    assert!(err.contains("boost_end_tick"), "{err}");
}

/// The new block is versioned and byte-counted independently of v1/v2/v3 of
/// validation.bin and the trajectory sidecar, so those layouts cannot drift.
#[test]
fn deployment_sidecar_roundtrip_and_layout_are_pinned() {
    let dir = temp_dir("sidecar");
    let validation = dir.join("validation.bin");
    let path = deployment_sidecar_path(&validation);
    assert_eq!(path, dir.join("validation.deployment"));
    let records = [
        truth(1, 3, [10.0, 20.0]),
        DeploymentTruth {
            index: 8,
            wind_mean_xy: [-1.0, 2.0],
            ..truth(2, 4, [30.0, 40.0])
        },
    ];
    write_deployment_truth(&path, &records).unwrap();
    assert_eq!(
        std::fs::metadata(&path).unwrap().len() as usize,
        DEPLOYMENT_HEADER_SIZE + 2 * DEPLOYMENT_RECORD_SIZE
    );
    assert_eq!(read_deployment_truth(&path).unwrap(), records);

    let mut bytes = std::fs::read(&path).unwrap();
    bytes[8..12].copy_from_slice(&(DEPLOYMENT_VERSION + 9).to_le_bytes());
    let bad = dir.join("bad.deployment");
    std::fs::write(&bad, bytes).unwrap();
    let err = read_deployment_truth(&bad).unwrap_err().to_string();
    assert!(err.contains("unsupported deployment truth version"), "{err}");
}
