//! Hard parity gates for the aero card extracted from OpenRocket Simulation 9.
//!
//! Evidence (read-only): `config/aero_card.toml`, extracted from
//! `/home/sehaxe/Downloads/Telegram Desktop/гугугага.ork`, databranch
//! `Маршевая ступень` (OR columns: Mach 50, alpha 15, axial Cd 35, Cm 37).
//! `aero_card_tests.rs` holds the loose parity envelope; this file pins hard
//! ceilings and the no-simulation-on-gaps behaviour. The card is never edited.

use std::path::PathBuf;

use potok_sim::{config::AeroSource, AeroCard, AeroChannel, Config, SimError, Simulator};

fn card_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../config/aero_card.toml")
}

fn card() -> AeroCard {
    AeroCard::from_file(&card_path()).expect("the extracted OR card must load")
}

/// (mach, alpha_deg, OR Cd, pinned card Cd, OR Cm, pinned card Cm).
/// The card readings are pinned so a silent regeneration fails loudly.
const OR_SAMPLES: &[(f64, f64, f64, f64, f64, f64)] = &[
    (0.020000, 14.896902673, 0.583000, 0.577527814100, -1.420000, -1.460777596378),
    (0.179000, 0.401070457, 0.363000, 0.360083698610, 0.046000, 0.045330449244),
    (0.493000, 0.006904141, 0.402000, 0.390683060057, 0.002000, 0.001964599527),
    (0.733000, 0.014558858, 0.394000, 0.393330003311, 0.001000, 0.000916033655),
    (0.407000, 0.028974476, 0.319000, 0.319184870070, 0.002000, 0.003365958763),
    (0.215000, 0.114591559, 0.300000, 0.299889960517, 0.010000, 0.011993622878),
    (0.059000, 3.437746771, 0.335000, 0.352317429833, 0.217000, 0.175238900897),
];

const NODE_TOL: f64 = 1e-8;
// Hard ceilings on the 7 sampled flight rows. The card's own header reports
// wider global maxima (0.080478 Cd, 1.193743 Cm) over all 594 extraction rows;
// none of these samples is allowed to approach them.
const MAX_CD_ROW_ERROR: f64 = 0.018;
const MAX_CM_ROW_ERROR: f64 = 0.045;

#[test]
fn extracted_card_parity_is_within_hard_max_error_thresholds() {
    let card = card();
    assert_eq!(card.mach.len(), 30, "card Mach axis must not change silently");
    assert_eq!(card.alpha_deg.len(), 29, "card alpha axis must not change silently");

    let (mut worst_cd, mut worst_cm) = (0.0_f64, 0.0_f64);
    for &(mach, alpha, cd_or, cd_pinned, cm_or, cm_pinned) in OR_SAMPLES {
        let cd = card
            .interpolate(AeroChannel::Cd0, mach, alpha)
            .unwrap_or_else(|e| panic!("Cd0 at ({mach}, {alpha}): {e}"));
        let cm = card
            .interpolate(AeroChannel::CmPitch, mach, alpha)
            .unwrap_or_else(|e| panic!("CmPitch at ({mach}, {alpha}): {e}"));
        assert!((cd - cd_pinned).abs() <= NODE_TOL, "card Cd drift at ({mach}, {alpha})");
        assert!((cm - cm_pinned).abs() <= NODE_TOL, "card Cm drift at ({mach}, {alpha})");
        worst_cd = worst_cd.max((cd - cd_or).abs());
        worst_cm = worst_cm.max((cm - cm_or).abs());
    }
    eprintln!("hard parity: worst Cd error {worst_cd:.6} <= {MAX_CD_ROW_ERROR}, worst Cm error {worst_cm:.6} <= {MAX_CM_ROW_ERROR}");
    assert!(worst_cd <= MAX_CD_ROW_ERROR, "Cd max error {worst_cd} > {MAX_CD_ROW_ERROR}");
    assert!(worst_cm <= MAX_CM_ROW_ERROR, "Cm max error {worst_cm} > {MAX_CM_ROW_ERROR}");
}

#[test]
fn missing_channels_and_nan_gaps_are_loud() {
    let card = card();
    // The flight data measures Cd and Cm only; the three control channels are
    // absent, never zero-filled.
    for ch in [AeroChannel::Cna, AeroChannel::PitchDamping, AeroChannel::CmDelta] {
        assert!(card.channel(ch).is_empty(), "{ch:?} must be absent from the card");
        match card.interpolate(ch, 0.493, 0.006904141) {
            Err(SimError::AeroCard(msg)) => assert!(msg.contains("absent"), "{ch:?}: {msg}"),
            other => panic!("expected missing-channel error for {ch:?}, got {other:?}"),
        }
    }
    match card.coeffs(0.493, 0.006904141, 0.30) {
        Err(SimError::AeroCard(msg)) => assert!(msg.contains("absent"), "{msg}"),
        other => panic!("expected missing-channel error from coeffs, got {other:?}"),
    }
    // A `nan` node means "not measured"; a bilinear stencil touching it is loud.
    match card.interpolate(AeroChannel::Cd0, 0.021, 0.0007) {
        Err(SimError::AeroCard(msg)) => assert!(msg.contains("no measured data"), "{msg}"),
        other => panic!("expected missing-data error, got {other:?}"),
    }
    for (mach, alpha, axis) in [
        (0.0049, 1.0, "mach"),
        (0.818, 1.0, "mach"),
        (0.493, 0.0006, "alpha_deg"),
        (0.493, 91.1, "alpha_deg"),
        (f64::NAN, 1.0, "mach"),
    ] {
        match card.interpolate(AeroChannel::Cd0, mach, alpha) {
            Err(SimError::AeroOutOfGrid { axis: got, .. }) => assert_eq!(got, axis),
            other => panic!("expected AeroOutOfGrid at ({mach}, {alpha}), got {other:?}"),
        }
    }
}

#[test]
fn simulation_does_not_fly_through_card_gaps() {
    let mut cfg = Config::default();
    cfg.aero.source = AeroSource::Card;
    cfg.aero.card_path = Some(PathBuf::from("config/aero_card.toml"));
    let sim = Simulator::new(cfg).expect("card config must load");
    let err = sim.run_episode(0).unwrap_err();
    assert!(
        matches!(err, SimError::AeroOutOfGrid { .. } | SimError::AeroCard(_)),
        "a gapped card must stop the run loudly, got {err:?}"
    );
}
