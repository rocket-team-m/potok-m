// SPDX-License-Identifier: GPL-3.0-or-later
//! No-silent-zero-fill gate for the three channels the OR flight data lacks.
//!
//! `config/aero_card.toml` was extracted from OpenRocket Simulation 9, which
//! exports Mach (50), alpha (15), axial Cd (35) and pitch Cm (37) only. There
//! is no fin-deflection, normal-force or pitch-rate signal in that source, so
//! `cna`, `pitch_damping` and `cm_delta` cannot be derived from it. This file
//! gates the LOUD refusal at four layers: the TOML text, the parsed card, the
//! four-channel helper used by the integrator, and a full episode. If any
//! layer starts returning a zero instead of an error, one of these tests fails.

use std::path::{Path, PathBuf};

use potok_sim::{
    aero::ABSENT_CHANNEL_ESCAPE, config::AeroSource, AeroCard, AeroChannel, Config, SimError,
    Simulator,
};

const CARD: &str = "../config/aero_card.toml";
const MISSING: [AeroChannel; 3] = [
    AeroChannel::Cna,
    AeroChannel::PitchDamping,
    AeroChannel::CmDelta,
];
// Clean cd0 stencils in the extracted card (the same points the parity gates
// use), so an absent-channel refusal cannot be confused with a nan gap.
const FLIGHT_POINTS: &[(f64, f64)] = &[
    (0.020, 14.896902673),
    (0.179, 0.401070457),
    (0.493, 0.006904141),
    (0.733, 0.014558858),
    (0.407, 0.028974476),
    (0.215, 0.114591559),
    (0.059, 3.437746771),
];

/// A zero-filled card has channels present as keys with numeric grids. Its
/// `coeffs` succeeds, which is exactly the silent path this gate excludes for
/// the OR card.
const ZERO_FILLED: &str = r#"
baseline_cd0 = 0.30
[grid]
mach = [0.0, 1.0]
alpha_deg = [0.0, 10.0]
[coefficients]
cd0 = [[0.30, 0.30], [0.30, 0.30]]
cna = [[0.0, 0.0], [0.0, 0.0]]
pitch_damping = [[0.0, 0.0], [0.0, 0.0]]
cm_delta = [[0.0, 0.0], [0.0, 0.0]]
cm_pitch = [[0.0, 0.0], [0.0, 0.0]]
"#;

fn card_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(CARD)
}

fn card_text() -> String {
    std::fs::read_to_string(card_path()).expect("read the extracted OR card")
}

fn card() -> AeroCard {
    AeroCard::from_toml_str(&card_text()).expect("parse the extracted OR card")
}

/// Keys assigned in `[coefficients]`. Array rows and comments are ignored, so
/// a pasted zero grid cannot hide from the check below.
fn coefficient_keys(text: &str) -> Vec<&str> {
    let mut in_table = false;
    let mut keys = Vec::new();
    for line in text.lines() {
        let t = line.trim();
        if t.starts_with('[') && !t.contains(',') {
            in_table = t == "[coefficients]";
            continue;
        }
        let code = t.split('#').next().unwrap_or("").trim();
        if !in_table || code.is_empty() || code.starts_with('[') {
            continue;
        }
        if let Some((k, _)) = code.split_once('=') {
            keys.push(k.trim());
        }
    }
    keys
}

#[test]
fn or_card_toml_has_no_channel_that_could_be_zero_filled() {
    let text = card_text();
    assert_eq!(
        coefficient_keys(&text),
        vec!["cd0", "cm_pitch"],
        "the OR card must contain only the two channels its source measured"
    );
    for marker in [
        "Simulation 9",
        "Mach (50)",
        "alpha (15)",
        "axial Cd (35)",
        "Cm (37)",
        "no fin-deflection channel",
        "no normal-force channel",
        "no pitch-rate channel",
        "NOT a control derivative",
    ] {
        assert!(text.contains(marker), "provenance marker missing: {marker}");
    }
}

#[test]
fn absent_channels_refuse_loudly_at_every_clean_flight_point() {
    let c = card();
    assert_eq!(c.mach.len(), 30, "Mach axis changed silently");
    assert_eq!(c.alpha_deg.len(), 29, "alpha axis changed silently");
    for ch in MISSING {
        assert!(
            c.channel(ch).is_empty(),
            "{ch:?} has a grid that could be zero-filled"
        );
        for &(mach, alpha) in FLIGHT_POINTS {
            match c.interpolate(ch, mach, alpha) {
                Err(SimError::AeroCard(msg)) => {
                    for want in [
                        ch.name(),
                        "absent",
                        "no value was invented",
                        ABSENT_CHANNEL_ESCAPE,
                    ] {
                        assert!(
                            msg.contains(want),
                            "{ch:?} at ({mach}, {alpha}) omitted '{want}': {msg}"
                        );
                    }
                }
                other => {
                    panic!("{ch:?} at ({mach}, {alpha}) returned {other:?} instead of refusing")
                }
            }
        }
    }

    // The four-channel helper the integrator calls must refuse as a whole.
    // At every clean point: no Ok result exists that could carry zeros.
    for &(mach, alpha) in FLIGHT_POINTS {
        match c.coeffs(mach, alpha, c.baseline_cd0) {
            Err(SimError::AeroCard(msg)) => {
                assert!(msg.contains("cna"), "first absent channel not named: {msg}");
                assert!(
                    msg.contains(ABSENT_CHANNEL_ESCAPE),
                    "refusal did not name the escape: {msg}"
                );
            }
            other => panic!(
                "coeffs at ({mach}, {alpha}) returned {other:?}; a missing channel was zero-filled"
            ),
        }
    }
}

#[test]
fn a_zero_filled_card_is_the_silent_path_this_gate_excludes() {
    // Failure-mode witness: with channels present, the same helper happily
    // returns zeros. The two gates above flip the moment a zero grid is pasted
    // into the real card, which is why absence — not "present but zero" — is
    // the contract.
    assert_eq!(
        coefficient_keys(ZERO_FILLED),
        vec!["cd0", "cna", "pitch_damping", "cm_delta", "cm_pitch"]
    );
    let z = AeroCard::from_toml_str(ZERO_FILLED).expect("zero-filled card parses");
    let got = z
        .coeffs(0.5, 5.0, z.baseline_cd0)
        .expect("a present zero grid is silently accepted");
    assert_eq!(got.cna, 0.0);
    assert_eq!(got.pitch_damping, 0.0);
    assert_eq!(got.cm_delta, 0.0);

    let real = card();
    assert!(real.coeffs(0.493, 0.006904141, real.baseline_cd0).is_err());
}

#[test]
fn episode_never_silently_flies_the_real_card() {
    let mut cfg = Config::default();
    cfg.aero.source = AeroSource::Card;
    cfg.aero.card_path = Some(PathBuf::from("config/aero_card.toml"));
    let sim = Simulator::new(cfg).expect("card config loads");
    match sim.run_episode(0) {
        Err(SimError::AeroOutOfGrid { axis, lo, hi, .. }) => {
            // At step 0 the first query may be below the Mach grid; later it
            // is the absent-channel fault ("card"). Both are LOUD; neither is
            // a successful episode flown on zero coefficients.
            if axis == "card" {
                assert!(
                    lo.is_nan() && hi.is_nan(),
                    "an absent channel has no grid bounds"
                );
            }
        }
        other => panic!("the real card must stop the episode loudly, got {other:?}"),
    }
}
