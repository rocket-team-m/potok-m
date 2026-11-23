// SPDX-License-Identifier: GPL-3.0-or-later
//! SIMTHRUST — gates for the config thrust-table provenance label.
//!
//! The shipped 200-knot curve is a rippled synthetic table with no static-fire
//! measurement behind it. It is explicitly labelled `experimental` and its
//! knots are untouched, so trajectories and zero-command gates stay identical.
//! A `passport` label is a claim that must be checked, not trusted.

use potok_sim::{
    config::{
        ThrustSource, DEFAULT_TOML, PASSPORT_IMPULSE_NS, PASSPORT_STAGE1_PEAK_N,
        PASSPORT_STAGE2_PEAK_N,
    },
    manifest::canonical_config_bytes,
    Config, Result, SimError,
};

fn assert_loud<T: std::fmt::Debug>(result: Result<T>, needle: &str) {
    match result {
        Err(SimError::Config(msg)) => {
            assert!(msg.contains(needle), "error {msg:?} lacks {needle:?}");
        }
        other => panic!("expected a loud Config error containing {needle:?}, got {other:?}"),
    }
}

/// Trapezoidal impulse and the per-stage peak, split at stage separation.
fn measured(table: &[(f64, f64, f64)], sep: f64) -> (f64, f64, f64) {
    let impulse: f64 = table
        .windows(2)
        .map(|w| (w[1].0 - w[0].0) * (w[0].1 + w[1].1) * 0.5)
        .sum();
    let stage1 = table
        .iter()
        .filter(|k| k.0 < sep)
        .map(|k| k.1)
        .fold(0.0, f64::max);
    let stage2 = table
        .iter()
        .filter(|k| k.0 >= sep)
        .map(|k| k.1)
        .fold(0.0, f64::max);
    (impulse, stage1, stage2)
}

fn compliant_table() -> Vec<(f64, f64, f64)> {
    vec![
        (0.0, 0.0, 0.0),
        (1.0, 520.0, 0.35),
        (3.5, 520.0, 0.35),
        (5.5, 0.0, 0.0),
        (6.0, 0.0, 0.0),
        (6.2, 575.0, 0.31),
        (8.0, 575.0, 0.31),
        (8.2, 0.0, 0.0),
    ]
}

#[test]
fn shipped_curve_is_labelled_experimental_with_its_knots_unchanged() {
    let cfg = Config::default();
    assert_eq!(cfg.aero.thrust_source, ThrustSource::Experimental);
    assert!(DEFAULT_TOML.contains("thrust_source = \"experimental\""));
    assert_eq!(cfg.aero.thrust_profile.len(), 200);
    let (impulse, stage1, stage2) = measured(&cfg.aero.thrust_profile, cfg.aero.stage_sep_time_s);
    assert!((impulse - 3627.8227).abs() < 1e-3, "impulse {impulse}");
    assert_eq!(stage1, 610.0);
    assert_eq!(stage2, 590.0);
    // The label exists precisely because these two numbers are out of passport.
    assert!(impulse > PASSPORT_IMPULSE_NS.1);
    assert!(stage1 > PASSPORT_STAGE1_PEAK_N.1);
    assert!((PASSPORT_STAGE2_PEAK_N.0..=PASSPORT_STAGE2_PEAK_N.1).contains(&stage2));
}

#[test]
fn a_config_without_the_label_has_identical_canonical_bytes() {
    let old_text = DEFAULT_TOML.replace("thrust_source = \"experimental\"\n", "");
    let old = Config::from_toml_str(&old_text).expect("pre-label config must still load");
    assert_eq!(old.aero.thrust_source, ThrustSource::Experimental);
    assert_eq!(
        canonical_config_bytes(&old).unwrap(),
        canonical_config_bytes(&Config::default()).unwrap(),
    );
}

#[test]
fn a_false_passport_claim_on_the_shipped_curve_is_loud_and_names_the_escape() {
    let mut cfg = Config::default();
    cfg.aero.thrust_source = ThrustSource::Passport;
    assert_loud(cfg.validate(), "aero.thrust_source");
    let msg = cfg.validate().unwrap_err().to_string();
    for needle in ["passport", "3627.8", "610.0", "experimental"] {
        assert!(msg.contains(needle), "error {msg:?} lacks {needle:?}");
    }
}

#[test]
fn the_label_is_read_from_toml_at_the_config_boundary() {
    let text = DEFAULT_TOML.replacen(
        "thrust_source = \"experimental\"",
        "thrust_source = \"passport\"",
        1,
    );
    assert_loud(
        Config::from_toml_str(&text),
        "violates the RDK passport envelope",
    );
}

#[test]
fn a_passport_labelled_curve_inside_the_envelope_loads() {
    let mut cfg = Config::default();
    cfg.aero.thrust_source = ThrustSource::Passport;
    cfg.aero.thrust_profile = compliant_table();
    let (impulse, stage1, stage2) = measured(&cfg.aero.thrust_profile, cfg.aero.stage_sep_time_s);
    assert_eq!(impulse, 3230.0);
    assert_eq!(stage1, 520.0);
    assert_eq!(stage2, 575.0);
    cfg.validate().expect("in-envelope passport curve");
    let text = String::from_utf8(canonical_config_bytes(&cfg).unwrap()).unwrap();
    assert!(text.contains("thrust_source = \"passport\""));
}

#[test]
fn an_explicit_experimental_curve_may_sit_outside_the_envelope() {
    let mut cfg = Config::default();
    cfg.aero.thrust_source = ThrustSource::Experimental;
    cfg.aero.thrust_profile = vec![(0.0, 0.0, 0.0), (1.0, 900.0, 1.0), (2.0, 0.0, 0.0)];
    cfg.validate()
        .expect("experimental is the documented escape hatch");
}

#[test]
fn structural_violations_are_loud_for_every_source() {
    let cases = vec![
        (vec![(0.0, 1.0, 0.0), (1.0, f64::NAN, 0.0)], "non-finite"),
        (
            vec![(0.0, -1.0, 0.0), (1.0, 1.0, 0.0)],
            "negative component",
        ),
        (vec![(0.0, 1.0, 0.0), (0.0, 2.0, 0.0)], "strictly after"),
    ];
    for source in [ThrustSource::Experimental, ThrustSource::Passport] {
        for (table, needle) in &cases {
            let mut cfg = Config::default();
            cfg.aero.thrust_source = source;
            cfg.aero.thrust_profile = table.clone();
            assert_loud(cfg.validate(), needle);
        }
    }
}
