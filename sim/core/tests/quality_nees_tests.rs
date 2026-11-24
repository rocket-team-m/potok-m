// SPDX-License-Identifier: GPL-3.0-or-later
//! Quality-report NEES corridor gates: typed status, estimator discrimination.
//!
//! The only evidence scored here is the deterministic synthetic gate from
//! `flight_replay`. No real log is supplied, scored or claimed. The two checks
//! below pin that a calibrated estimator passes, a biased estimator changes
//! the report, and a real log is always reported as NOT MEASURED_FOR_REAL_LOG.

use potok_sim::{
    flight_replay::{
        validate_real_log_nees, FlightLog, FlightLogError, CHANNEL_COUNT, SCHEMA_VERSION,
    },
    quality::{
        check_dataset, real_log_nees_verdict, synthetic_biased_nees_verdict,
        synthetic_calibrated_nees_verdict, NeesCorridorVerdict,
    },
    read_trajectory, scenario, sidecar_path,
    validation::{read_validation, write_validation, ValidationSet, FILE_NAME},
    write_trajectory_v3, Config, Episode, Simulator, Split, TrajectorySet,
};

const SEED: u64 = 0xC0FFEE;
const COVERAGE: f64 = 0.95;

fn stub_log() -> FlightLog {
    FlightLog {
        schema_version: SCHEMA_VERSION,
        episode_index: 0,
        sample_rate_hz: 50,
        sigma: [1.0; CHANNEL_COUNT],
        samples: Vec::new(),
    }
}

fn split_fixture(seed: u64) -> (Config, ValidationSet, TrajectorySet) {
    let mut cfg = Config::default();
    cfg.run.seed = seed;
    let sim = Simulator::new(cfg.clone()).unwrap();
    let eps: Vec<Episode> = (0..)
        .filter(|&i| scenario(i).split == Split::Train)
        .take(1)
        .map(|i| sim.run_episode(i).unwrap())
        .collect();
    let dir = std::env::temp_dir().join(format!(
        "potok-nees-q-{seed}-{}-{}",
        std::process::id(),
        SEED
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let validation = dir.join(FILE_NAME);
    write_validation(&validation, &cfg, &eps).unwrap();
    let v = read_validation(&validation).unwrap();
    let sidecar = sidecar_path(&validation);
    write_trajectory_v3(&sidecar, &cfg, &eps).unwrap();
    let t = read_trajectory(&sidecar).unwrap();
    (cfg, v, t)
}

#[test]
fn synthetic_verdict_is_measured_synthetic_and_passes() {
    let verdict = synthetic_calibrated_nees_verdict("train", SEED, COVERAGE).unwrap();

    match &verdict {
        NeesCorridorVerdict::MeasuredSynthetic {
            ok,
            dof,
            coverage,
            empirical_coverage,
            detail,
        } => {
            assert!(*ok, "calibrated arm failed: {detail}");
            assert_eq!(*dof, CHANNEL_COUNT);
            assert_eq!(*coverage, COVERAGE);
            assert!(
                (0.94..=0.96).contains(empirical_coverage),
                "empirical coverage {empirical_coverage} outside the pinned band"
            );
            assert!(detail.contains("MEASURED_SYNTHETIC"), "{detail}");
            assert!(detail.contains("no real log"), "{detail}");
        }
        other => panic!("synthetic gate must report MEASURED_SYNTHETIC, got {other:?}"),
    }
    assert_eq!(verdict.status_str(), "MEASURED_SYNTHETIC");
    assert!(verdict.is_ok());
    let check = verdict.as_gate_check();
    assert_eq!(check.name, "nees_corridor_synthetic");
    assert!(check.ok);
}

#[test]
fn biased_estimator_changes_the_report() {
    let calibrated = synthetic_calibrated_nees_verdict("train", SEED, COVERAGE).unwrap();
    let biased = synthetic_biased_nees_verdict("train", SEED, COVERAGE).unwrap();

    assert!(calibrated.is_ok());
    assert!(
        !biased.is_ok(),
        "biased estimator passed the corridor: {biased:?}"
    );
    // The bias changes the estimator verdict, never the evidence class.
    assert_eq!(calibrated.status_str(), "MEASURED_SYNTHETIC");
    assert_eq!(biased.status_str(), "MEASURED_SYNTHETIC");
    let empirical = match biased {
        NeesCorridorVerdict::MeasuredSynthetic {
            empirical_coverage, ..
        } => empirical_coverage,
        ref other => panic!("biased verdict must be synthetic, got {other:?}"),
    };
    assert!(
        empirical < 0.05,
        "biased coverage {empirical} should be near zero"
    );

    // The serialized report is genuinely different, not a relabelled stub.
    let calibrated_json = serde_json::to_string(&calibrated).unwrap();
    let biased_json = serde_json::to_string(&biased).unwrap();
    assert_ne!(calibrated_json, biased_json);
    assert!(calibrated_json.contains("MEASURED_SYNTHETIC"));
    assert!(biased_json.contains("MEASURED_SYNTHETIC"));
    assert!(!biased_json.contains("NOT MEASURED_FOR_REAL_LOG"));
    assert_ne!(
        calibrated.as_gate_check().ok,
        biased.as_gate_check().ok,
        "the quality gate must flip when the estimator is biased"
    );
}

#[test]
fn real_log_verdict_is_not_measured_for_real_log() {
    let log = stub_log();
    // The guard itself is LOUD; the verdict is derived from it, not assigned.
    match validate_real_log_nees(&log) {
        Err(FlightLogError::NotMeasured(msg)) => assert!(msg.contains("NOT MEASURED"), "{msg}"),
        other => panic!("real-log guard must refuse loudly, got {other:?}"),
    }

    let verdict = real_log_nees_verdict("train", &log).unwrap();
    match &verdict {
        NeesCorridorVerdict::NotMeasuredForRealLog { detail } => {
            assert!(detail.contains("NOT MEASURED"), "{detail}");
            assert!(detail.contains("train"), "{detail}");
        }
        other => panic!("real log must be NOT MEASURED_FOR_REAL_LOG, got {other:?}"),
    }
    assert_eq!(verdict.status_str(), "NOT MEASURED_FOR_REAL_LOG");
    assert!(!verdict.is_ok());
    assert!(!verdict.as_gate_check().ok);
    assert_ne!(
        verdict,
        synthetic_calibrated_nees_verdict("train", SEED, COVERAGE).unwrap(),
        "synthetic and real-log statuses must never be conflated"
    );
}

#[test]
fn quality_report_carries_the_nees_corridor_gate() {
    let (cfg, v, t) = split_fixture(2026);
    let report = check_dataset("train", &cfg, &v, &t, 1).unwrap();
    let gate = report
        .checks
        .iter()
        .find(|c| c.name == "nees_corridor_synthetic")
        .expect("quality report must carry the NEES corridor verdict");
    assert!(gate.ok, "{}", gate.detail);
    assert!(
        gate.detail.contains("MEASURED_SYNTHETIC"),
        "{}",
        gate.detail
    );
    assert!(gate.detail.contains("no real log"), "{}", gate.detail);
    assert!(gate.detail.contains("chi2_8"), "{}", gate.detail);
    assert!(report.all_passed);

    // The report gate uses the calibrated arm; a biased estimator could not be
    // substituted without flipping the gate that is recorded here.
    assert!(!synthetic_biased_nees_verdict("train", 2026, COVERAGE)
        .unwrap()
        .is_ok());
}
