//! Parity of the OpenRocket-derived coefficient card against the flight rows
//! it was extracted from. The .ork sample is read-only and not in the repo;
//! the sampled values below are copied from Simulation 9, databranch
//! "Маршевая ступень" (column indices: Mach 50, alpha 15, axial Cd 35,
//! pitch moment coefficient 37). `cd_card` / `cm_card` are the card's own
//! bilinear readings at those exact points.

use std::path::{Path, PathBuf};

use potok_sim::{
    aero::AeroChannel,
    AeroCard, SimError,
};

fn card_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../config/aero_card.toml")
}

// (mach, alpha_deg, OR cd_axial, card cd0, OR Cm, card cm_pitch)
const OR_SAMPLES: &[(f64, f64, f64, f64, f64, f64)] = &[
    (0.020000, 14.896902673, 0.583000, 0.577527814100, -1.420000, -1.460777596378),
    (0.179000, 0.401070457, 0.363000, 0.360083698610, 0.046000, 0.045330449244),
    (0.493000, 0.006904141, 0.402000, 0.390683060057, 0.002000, 0.001964599527),
    (0.733000, 0.014558858, 0.394000, 0.393330003311, 0.001000, 0.000916033655),
    (0.407000, 0.028974476, 0.319000, 0.319184870070, 0.002000, 0.003365958763),
    (0.215000, 0.114591559, 0.300000, 0.299889960517, 0.010000, 0.011993622878),
    (0.059000, 3.437746771, 0.335000, 0.352317429833, 0.217000, 0.175238900897),
];

#[test]
fn or_card_reproduces_sampled_or_flight_rows() {
    let card = AeroCard::from_file(&card_path()).unwrap();
    assert_eq!(card.mach.len(), 30);
    assert_eq!(card.alpha_deg.len(), 29);
    for &(mach, alpha, cd_or, cd_card, cm_or, cm_card) in OR_SAMPLES {
        let cd = card.interpolate(AeroChannel::Cd0, mach, alpha).unwrap();
        assert!((cd - cd_card).abs() < 1e-8, "cd card reading at ({mach}, {alpha})");
        assert!((cd - cd_or).abs() < 0.02, "cd parity at ({mach}, {alpha}): {cd} vs {cd_or}");
        let cm = card.interpolate(AeroChannel::CmPitch, mach, alpha).unwrap();
        assert!((cm - cm_card).abs() < 1e-8, "cm card reading at ({mach}, {alpha})");
        assert!((cm - cm_or).abs() < 0.05, "cm parity at ({mach}, {alpha}): {cm} vs {cm_or}");
    }
}

#[test]
fn or_card_out_of_envelope_and_missing_data_are_loud() {
    let card = AeroCard::from_file(&card_path()).unwrap();
    for (mach, alpha, axis) in [
        (0.0049, 1.0, "mach"),
        (0.818, 1.0, "mach"),
        (0.493, 0.0006, "alpha_deg"),
        (0.493, 91.1, "alpha_deg"),
        (f64::NAN, 1.0, "mach"),
    ] {
        match card.interpolate(AeroChannel::Cd0, mach, alpha) {
            Err(SimError::AeroOutOfGrid { axis: got, .. }) => assert_eq!(got, axis),
            other => panic!("expected AeroOutOfGrid for ({mach}, {alpha}), got {other:?}"),
        }
    }
    // The flight data has no cna / pitch damping / control derivative: those
    // channels are absent from the card and must never be zero-filled.
    for ch in [
        AeroChannel::Cna,
        AeroChannel::PitchDamping,
        AeroChannel::CmDelta,
    ] {
        match card.interpolate(ch, 0.493, 0.006904141) {
            Err(SimError::AeroCard(msg)) => assert!(msg.contains("absent"), "{ch:?}: {msg}"),
            other => panic!("expected missing-channel error for {ch:?}, got {other:?}"),
        }
    }
    // The same applies to the four-channel helper used by the integrator: no
    // silent zero coefficients just because the flight data lacks them.
    match card.coeffs(0.493, 0.006904141, 0.30) {
        Err(SimError::AeroCard(msg)) => assert!(msg.contains("absent"), "{msg}"),
        other => panic!("expected missing-channel error from coeffs, got {other:?}"),
    }
    // A nan node means "not measured"; touching it in a bilinear stencil is
    // loud, not a zero or a clamp.
    match card.interpolate(AeroChannel::Cd0, 0.021, 0.0007) {
        Err(SimError::AeroCard(msg)) => assert!(msg.contains("no measured data"), "{msg}"),
        other => panic!("expected missing-data error, got {other:?}"),
    }
}

#[test]
fn or_card_load_is_deterministic() {
    let from_file = AeroCard::from_file(&card_path()).unwrap();
    let text = std::fs::read_to_string(card_path()).unwrap();
    let from_str = AeroCard::from_toml_str(&text).unwrap();
    let again = AeroCard::from_file(&card_path()).unwrap();
    // Debug text compares NaN nodes by representation; derived PartialEq would
    // report NaN != NaN and is not the determinism property under test.
    assert_eq!(format!("{from_file:?}"), format!("{from_str:?}"));
    assert_eq!(format!("{from_file:?}"), format!("{again:?}"));
    assert_eq!(from_file.mach, from_str.mach);
    assert_eq!(from_file.alpha_deg, from_str.alpha_deg);
}
