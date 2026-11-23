// SPDX-License-Identifier: GPL-3.0-or-later
//! NEES corridor gates — deterministic synthetic evidence, no real-data claim.
//!
//! The only evidence here is a 10^4-run Monte-Carlo check that the chi-square
//! corridor accepts a calibrated estimator and rejects a deliberately biased
//! one. It proves the instrument discriminates; it does not validate the
//! vehicle, the sensors or any real recording. The real-log path is gated
//! separately and is LOUD `NOT MEASURED` until a real log exists.
//!
//! The production module is included by path rather than via `lib.rs`, so this
//! lane edits only its assigned files. The two shim modules below supply the
//! `crate::episode` / `crate::sensor` paths the module uses when it is
//! installed as `potok_sim::flight_replay`.

pub mod episode {
    pub use potok_sim::episode::Episode;
}
pub mod sensor {
    pub use potok_sim::sensor::Frame;
}

#[path = "../src/flight_replay.rs"]
mod flight_replay;

use flight_replay::{
    synthetic_nees_corridor_gate, validate_real_log_nees, FlightLog, FlightLogError,
    CHANNEL_COUNT, SCHEMA_VERSION, SYNTHETIC_NEES_RUNS,
};

fn stub_log() -> FlightLog {
    FlightLog {
        schema_version: SCHEMA_VERSION,
        episode_index: 0,
        sample_rate_hz: 50,
        sigma: [1.0; CHANNEL_COUNT],
        samples: Vec::new(),
    }
}

#[test]
fn synthetic_gate_is_deterministic_with_a_pinned_budget() {
    let a = synthetic_nees_corridor_gate(0xC0FFEE, 0.95).unwrap();
    let b = synthetic_nees_corridor_gate(0xC0FFEE, 0.95).unwrap();
    assert_eq!(a, b, "gate depends on hidden process state");
    assert_eq!(a.runs, SYNTHETIC_NEES_RUNS);
    assert_eq!(a.runs, 10_000);
    assert_eq!(a.dof, CHANNEL_COUNT);
    assert_eq!(a.coverage, 0.95);
    assert!(a.good_inside <= a.runs && a.biased_inside <= a.runs);
}

#[test]
fn calibrated_estimator_nees_stays_inside_the_chi2_corridor() {
    // Eight standard-normal residuals have NEES ~ chi2_8; a central 95 %
    // corridor should cover about 95 % of 10^4 draws.
    for seed in [0xC0FFEE_u64, 0x5A17, 42] {
        let gate = synthetic_nees_corridor_gate(seed, 0.95).unwrap();
        let covered = gate.good_coverage();
        assert!(
            (0.94..=0.96).contains(&covered),
            "seed {seed}: calibrated coverage {covered} outside [0.94, 0.96]"
        );
    }
}

#[test]
fn biased_estimator_nees_fails_the_chi2_corridor() {
    // +2 sigma on all eight channels shifts NEES to a non-central chi2_8 with
    // non-centrality 32 (mean 40), far above the corridor's hi ~= 17.5.
    for seed in [0xC0FFEE_u64, 0x5A17, 42] {
        let gate = synthetic_nees_corridor_gate(seed, 0.95).unwrap();
        let covered = gate.biased_coverage();
        assert!(
            covered < 0.05,
            "seed {seed}: biased coverage {covered} should be near zero"
        );
        assert!(
            covered < gate.good_coverage() - 0.80,
            "seed {seed}: gate did not discriminate good from biased"
        );
    }
}

#[test]
fn real_log_corridor_gate_is_loud_not_measured() {
    match validate_real_log_nees(&stub_log()) {
        Err(FlightLogError::NotMeasured(msg)) => {
            assert!(msg.contains("NOT MEASURED"), "message: {msg}");
            assert!(msg.contains("per-channel sigma"), "escape not named: {msg}");
        }
        other => panic!("real-log gate must refuse loudly, got {other:?}"),
    }
}
