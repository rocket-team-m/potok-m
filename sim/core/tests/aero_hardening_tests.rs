// SPDX-License-Identifier: GPL-3.0-or-later
//! Hardening gates for the extracted OpenRocket Simulation 9 aero card.
//!
//! Evidence (read-only): `config/aero_card.toml`, extracted from Simulation 9,
//! databranch "Маршевая ступень" (OR columns Mach 50, alpha 15, axial Cd 35,
//! Cm 37). `aero_card_tests.rs` and `aero_parity_hard_tests.rs` hold the first
//! parity and gap gates. This file hardens all five properties the extracted
//! card relies on: parity thresholds, LOUD gaps, no out-of-grid, determinism,
//! and no fallback to the placeholder coefficient path. No source is edited.

use std::{
    path::{Path, PathBuf},
    sync::Mutex,
};

use potok_sim::{
    aero::placeholder_uses,
    config::AeroSource,
    AeroCard, AeroChannel, Config, SimError, Simulator, PLACEHOLDER_CM_DELTA,
};

const CARD: &str = "../config/aero_card.toml";
const PLACEHOLDER_CARD: &str = "../config/aero_card_placeholder.toml";
const MACH_LEN: usize = 30;
const ALPHA_LEN: usize = 29;
const NODE_TOL: f64 = 1e-8;
const NODE_READ_TOL: f64 = 1e-12;

// Hard ceilings on the 7 sampled flight rows. The card header reports wider
// global maxima over all 594 extraction rows; no sampled row may approach them.
const MAX_CD_ROW_ERROR: f64 = 0.018;
const MAX_CM_ROW_ERROR: f64 = 0.045;
const MAX_CD_MEAN_ERROR: f64 = 0.010;
const MAX_CM_MEAN_ERROR: f64 = 0.020;

// The placeholder marker is process-global: serialize the one test that
// observes it. Other test binaries are separate processes and cannot interfere.
static COUNTER_LOCK: Mutex<()> = Mutex::new(());

// (mach, alpha_deg, OR Cd, pinned card Cd, OR Cm, pinned card Cm).
const OR_SAMPLES: &[(f64, f64, f64, f64, f64, f64)] = &[
    (0.020000, 14.896902673, 0.583000, 0.577527814100, -1.420000, -1.460777596378),
    (0.179000, 0.401070457, 0.363000, 0.360083698610, 0.046000, 0.045330449244),
    (0.493000, 0.006904141, 0.402000, 0.390683060057, 0.002000, 0.001964599527),
    (0.733000, 0.014558858, 0.394000, 0.393330003311, 0.001000, 0.000916033655),
    (0.407000, 0.028974476, 0.319000, 0.319184870070, 0.002000, 0.003365958763),
    (0.215000, 0.114591559, 0.300000, 0.299889960517, 0.010000, 0.011993622878),
    (0.059000, 3.437746771, 0.335000, 0.352317429833, 0.217000, 0.175238900897),
];

// A fully populated card: no nan nodes, all four control channels present.
const FULL_CARD: &str = r#"
baseline_cd0 = 0.30
[grid]
mach = [0.0, 1.0]
alpha_deg = [0.0, 10.0]
[coefficients]
cd0 = [[0.77, 0.77], [0.77, 0.77]]
cna = [[9.0, 9.0], [9.0, 9.0]]
pitch_damping = [[7.0, 7.0], [7.0, 7.0]]
cm_delta = [[3.0, 3.0], [3.0, 3.0]]
cm_pitch = [[-1.25, -1.25], [-1.25, -1.25]]
"#;

fn card_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(name)
}

fn card() -> AeroCard {
    AeroCard::from_file(&card_path(CARD)).expect("the extracted OR card must load")
}

fn config_for(name: &str) -> Config {
    let mut cfg = Config::default();
    cfg.aero.source = AeroSource::Card;
    cfg.aero.card_path = Some(PathBuf::from(format!("config/{name}")));
    cfg
}

/// Lower corner of the bilinear stencil containing knot `i` (mirrors
/// `interp_index`: an exact interior knot uses the segment to its left).
fn stencil_lo(axis: &[f64], i: usize) -> usize {
    let last = axis.len() - 1;
    if i == last {
        last - 1
    } else if i == 0 {
        0
    } else {
        i - 1
    }
}

fn channels_bit_identical(a: &AeroCard, b: &AeroCard) -> bool {
    for ch in [
        AeroChannel::Cd0,
        AeroChannel::Cna,
        AeroChannel::PitchDamping,
        AeroChannel::CmDelta,
        AeroChannel::CmPitch,
    ] {
        let (x, y) = (a.channel(ch), b.channel(ch));
        if x.len() != y.len() {
            return false;
        }
        for (rx, ry) in x.iter().zip(y.iter()) {
            if rx.len() != ry.len() {
                return false;
            }
            if rx
                .iter()
                .zip(ry.iter())
                .any(|(vx, vy)| vx.to_bits() != vy.to_bits())
            {
                return false;
            }
        }
    }
    true
}

#[test]
fn or_card_parity_stays_inside_hard_per_row_and_mean_ceilings() {
    let card = card();
    assert_eq!(card.mach.len(), MACH_LEN, "Mach axis changed silently");
    assert_eq!(card.alpha_deg.len(), ALPHA_LEN, "alpha axis changed silently");

    let (mut worst_cd, mut worst_cm, mut sum_cd, mut sum_cm) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    for &(mach, alpha, cd_or, cd_pinned, cm_or, cm_pinned) in OR_SAMPLES {
        let cd = card
            .interpolate(AeroChannel::Cd0, mach, alpha)
            .unwrap_or_else(|e| panic!("Cd0 at ({mach}, {alpha}): {e}"));
        let cm = card
            .interpolate(AeroChannel::CmPitch, mach, alpha)
            .unwrap_or_else(|e| panic!("CmPitch at ({mach}, {alpha}): {e}"));
        assert!(cd.is_finite() && cm.is_finite(), "non-finite card reading");
        assert!(
            (cd - cd_pinned).abs() <= NODE_TOL,
            "card Cd drift at ({mach}, {alpha}): {cd} vs {cd_pinned}"
        );
        assert!(
            (cm - cm_pinned).abs() <= NODE_TOL,
            "card Cm drift at ({mach}, {alpha}): {cm} vs {cm_pinned}"
        );
        let (e_cd, e_cm) = ((cd - cd_or).abs(), (cm - cm_or).abs());
        worst_cd = worst_cd.max(e_cd);
        worst_cm = worst_cm.max(e_cm);
        sum_cd += e_cd;
        sum_cm += e_cm;
    }
    let n = OR_SAMPLES.len() as f64;
    eprintln!(
        "hard parity: Cd worst {worst_cd:.6}/{MAX_CD_ROW_ERROR}, mean {:.6}/{MAX_CD_MEAN_ERROR}; \
         Cm worst {worst_cm:.6}/{MAX_CM_ROW_ERROR}, mean {:.6}/{MAX_CM_MEAN_ERROR}",
        sum_cd / n,
        sum_cm / n
    );
    assert!(worst_cd <= MAX_CD_ROW_ERROR, "Cd row error {worst_cd}");
    assert!(worst_cm <= MAX_CM_ROW_ERROR, "Cm row error {worst_cm}");
    assert!(sum_cd / n <= MAX_CD_MEAN_ERROR, "Cd mean error {}", sum_cd / n);
    assert!(sum_cm / n <= MAX_CM_MEAN_ERROR, "Cm mean error {}", sum_cm / n);
}

#[test]
fn or_card_finite_nodes_read_back_exactly_and_gaps_are_never_invented() {
    let card = card();
    for ch in [AeroChannel::Cd0, AeroChannel::CmPitch] {
        let grid = card.channel(ch);
        let (mut clean, mut gapped) = (0, 0);
        for ai in 0..card.alpha_deg.len() {
            for mi in 0..card.mach.len() {
                let (m0, a0) = (
                    stencil_lo(&card.mach, mi),
                    stencil_lo(&card.alpha_deg, ai),
                );
                let corners = [
                    grid[a0][m0],
                    grid[a0][m0 + 1],
                    grid[a0 + 1][m0],
                    grid[a0 + 1][m0 + 1],
                ];
                let (mach, alpha) = (card.mach[mi], card.alpha_deg[ai]);
                if corners.iter().all(|v| v.is_finite()) {
                    let got = card.interpolate(ch, mach, alpha).unwrap();
                    let node = grid[ai][mi];
                    assert!(
                        (got - node).abs() <= NODE_READ_TOL,
                        "{ch:?} node (alpha={alpha}, mach={mach}): {got} vs {node}"
                    );
                    clean += 1;
                } else {
                    match card.interpolate(ch, mach, alpha) {
                        Err(SimError::AeroCard(msg)) => assert!(
                            msg.contains("no measured data"),
                            "{ch:?} gap at (alpha={alpha}, mach={mach}): {msg}"
                        ),
                        other => panic!("{ch:?} gap at (alpha={alpha}, mach={mach}): {other:?}"),
                    }
                    gapped += 1;
                }
            }
        }
        assert!(clean > 0 && gapped > 0, "{ch:?}: clean={clean} gapped={gapped}");
    }
}

#[test]
fn or_card_absent_channels_and_partial_nan_stencils_are_loud() {
    let card = card();
    // The flight data measures Cd and Cm only. The three control derivatives
    // are absent and must never be zero-filled or borrowed from the placeholder.
    for ch in [AeroChannel::Cna, AeroChannel::PitchDamping, AeroChannel::CmDelta] {
        assert!(card.channel(ch).is_empty(), "{ch:?} must be absent");
        match card.interpolate(ch, 0.493, 0.006904141) {
            Err(SimError::AeroCard(msg)) => assert!(msg.contains("absent"), "{ch:?}: {msg}"),
            other => panic!("expected absent-channel error for {ch:?}, got {other:?}"),
        }
    }
    match card.coeffs(0.493, 0.006904141, 0.30) {
        Err(SimError::AeroCard(msg)) => assert!(msg.contains("absent"), "{msg}"),
        other => panic!("coeffs must not fill absent channels, got {other:?}"),
    }

    // Every stencil below touches at least one nan corner. The last point is a
    // measured node (Cd=0.468) whose neighbour is nan: the card must still
    // refuse the whole stencil rather than quietly interpolate the good corner.
    for (mach, alpha) in [
        (0.021, 0.0007),
        (0.005, 1.0),
        (0.074, 70.0),
        (0.021, 5.0),
        (0.817, 90.0),
    ] {
        match card.interpolate(AeroChannel::Cd0, mach, alpha) {
            Err(SimError::AeroCard(msg)) => assert!(
                msg.contains("no measured data"),
                "({mach}, {alpha}): {msg}"
            ),
            other => panic!("nan stencil at ({mach}, {alpha}) was not loud: {other:?}"),
        }
    }
}

#[test]
fn or_card_out_of_grid_is_loud_with_exact_bounds_and_no_clamp() {
    let card = card();
    let (m_lo, m_hi) = (card.mach[0], *card.mach.last().unwrap());
    let (a_lo, a_hi) = (card.alpha_deg[0], *card.alpha_deg.last().unwrap());
    let (mid_m, mid_a) = (0.493, 0.006904141);

    for ch in [AeroChannel::Cd0, AeroChannel::CmPitch] {
        for (mach, alpha, axis, lo, hi) in [
            (m_lo - 1e-4, mid_a, "mach", m_lo, m_hi),
            (m_hi + 1e-3, mid_a, "mach", m_lo, m_hi),
            (mid_m, a_lo - 1e-5, "alpha_deg", a_lo, a_hi),
            (mid_m, a_hi + 0.1, "alpha_deg", a_lo, a_hi),
            (f64::NAN, mid_a, "mach", m_lo, m_hi),
            (f64::INFINITY, mid_a, "mach", m_lo, m_hi),
            (f64::NEG_INFINITY, mid_a, "mach", m_lo, m_hi),
            (mid_m, f64::NAN, "alpha_deg", a_lo, a_hi),
        ] {
            match card.interpolate(ch, mach, alpha) {
                Err(SimError::AeroOutOfGrid {
                    axis: got,
                    value,
                    lo: got_lo,
                    hi: got_hi,
                }) => {
                    assert_eq!(got, axis, "({mach}, {alpha})");
                    assert_eq!(got_lo.to_bits(), lo.to_bits(), "lo bound");
                    assert_eq!(got_hi.to_bits(), hi.to_bits(), "hi bound");
                    assert!(
                        value.is_nan() || value < lo || value > hi,
                        "reported value {value} is not outside [{lo}, {hi}]"
                    );
                }
                other => panic!("{ch:?} ({mach}, {alpha}) was clamped: {other:?}"),
            }
        }
    }
}

#[test]
fn fully_populated_card_covers_its_whole_axis_range_without_false_out_of_grid() {
    let card = AeroCard::from_toml_str(FULL_CARD).unwrap();
    for i in 0..=200 {
        let f = i as f64 / 200.0;
        let c = card.coeffs(f, 10.0 * f, 0.30).unwrap();
        assert!(
            [c.cd0, c.cna, c.pitch_damping, c.cm_delta]
                .iter()
                .all(|v| v.is_finite()),
            "in-grid query at {f} produced a non-finite coefficient"
        );
    }
    for (mach, alpha) in [(0.0, 0.0), (1.0, 10.0)] {
        card.coeffs(mach, alpha, 0.30)
            .unwrap_or_else(|e| panic!("exact boundary ({mach}, {alpha}) must be in grid: {e}"));
    }
    assert!(matches!(
        card.interpolate(AeroChannel::Cd0, -1e-12, 0.0),
        Err(SimError::AeroOutOfGrid { .. })
    ));
    assert!(matches!(
        card.interpolate(AeroChannel::Cd0, 1.0 + 1e-12, 0.0),
        Err(SimError::AeroOutOfGrid { .. })
    ));
}

#[test]
fn or_card_load_and_interpolation_are_bitwise_deterministic() {
    let text = std::fs::read_to_string(card_path(CARD)).unwrap();
    let a = card();
    let b = AeroCard::from_toml_str(&text).unwrap();
    let c = card();

    assert!(channels_bit_identical(&a, &b), "file vs string grids differ");
    assert!(channels_bit_identical(&a, &c), "two file loads differ");
    let axis_bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(axis_bits(&a.mach), axis_bits(&b.mach));
    assert_eq!(axis_bits(&a.alpha_deg), axis_bits(&b.alpha_deg));
    assert_eq!(a.baseline_cd0.to_bits(), b.baseline_cd0.to_bits());
    assert_eq!(format!("{a:?}"), format!("{b:?}"));

    for &(mach, alpha, ..) in OR_SAMPLES {
        let first = a.interpolate(AeroChannel::Cd0, mach, alpha).unwrap();
        assert_eq!(first.to_bits(), b.interpolate(AeroChannel::Cd0, mach, alpha).unwrap().to_bits());
        assert_eq!(first.to_bits(), c.interpolate(AeroChannel::Cd0, mach, alpha).unwrap().to_bits());
        assert_eq!(first.to_bits(), a.interpolate(AeroChannel::Cd0, mach, alpha).unwrap().to_bits());
        let first_cm = a.interpolate(AeroChannel::CmPitch, mach, alpha).unwrap();
        assert_eq!(first_cm.to_bits(), b.interpolate(AeroChannel::CmPitch, mach, alpha).unwrap().to_bits());
    }
}

#[test]
fn card_path_episodes_are_deterministic() {
    // The OR card is intentionally gapped, so end-to-end determinism is checked
    // on the fully populated card; the OR card's own determinism is pinned above.
    let mut cfg = config_for("aero_card_placeholder.toml");
    cfg.run.seed = 1337;
    let a = Simulator::new(cfg.clone()).unwrap().run_episode(0).unwrap();
    let b = Simulator::new(cfg).unwrap().run_episode(0).unwrap();
    assert_eq!(a.features, b.features);
    assert_eq!(a.apogee_m.to_bits(), b.apogee_m.to_bits());
    let frames = |e: &potok_sim::Episode| {
        e.frames.iter().map(|f| f.to_bytes()).collect::<Vec<_>>()
    };
    assert_eq!(frames(&a), frames(&b));
}

#[test]
fn card_mode_never_selects_or_marks_the_placeholder_path() {
    let _guard = COUNTER_LOCK.lock().unwrap();
    let base = placeholder_uses();
    let _or = Simulator::new(config_for("aero_card.toml")).unwrap();
    let _or_again = Simulator::new(config_for("aero_card.toml")).unwrap();
    let _full = Simulator::new(config_for("aero_card_placeholder.toml")).unwrap();
    assert_eq!(
        placeholder_uses(),
        base,
        "selecting a card must not touch the placeholder COUNTED marker"
    );
    let _placeholder = Simulator::new(Config::default()).unwrap();
    assert_eq!(
        placeholder_uses(),
        base + 1,
        "the placeholder path must be COUNTED exactly once"
    );
}

#[test]
fn card_coefficients_are_card_values_not_placeholder_formulas() {
    let card = AeroCard::from_toml_str(FULL_CARD).unwrap();
    let cfg = Config::default();
    let c = card.coeffs(0.5, 5.0, card.baseline_cd0).unwrap();
    assert_eq!(c.cd0.to_bits(), 0.77_f64.to_bits());
    assert_eq!(c.cna.to_bits(), 9.0_f64.to_bits());
    assert_eq!(c.pitch_damping.to_bits(), 7.0_f64.to_bits());
    assert_eq!(c.cm_delta.to_bits(), 3.0_f64.to_bits());

    let placeholder_cd = AeroCard::placeholder_cd(cfg.aero.cd0, &cfg.aero.cd_mach, 0.5);
    let card_cd = c.cd0;
    assert!(
        (card_cd - placeholder_cd).abs() > 0.1,
        "card Cd0 {card_cd} collapsed onto the placeholder table {placeholder_cd}"
    );
    assert_ne!(c.cna, cfg.aero.cna);
    assert_ne!(c.pitch_damping, cfg.aero.pitch_damping);
    assert_ne!(c.cm_delta, PLACEHOLDER_CM_DELTA);

    // The episode cd0 shift applies on top of the CARD value, not the
    // placeholder baseline table.
    let shifted = card.coeffs(0.5, 5.0, card.baseline_cd0 + 0.10).unwrap();
    assert!((shifted.cd0 - 0.87).abs() <= NODE_READ_TOL, "shifted cd0 {}", shifted.cd0);
    assert_eq!(shifted.cna.to_bits(), c.cna.to_bits());
    assert_eq!(shifted.pitch_damping.to_bits(), c.pitch_damping.to_bits());
    assert_eq!(shifted.cm_delta.to_bits(), c.cm_delta.to_bits());
}

#[test]
fn extracted_card_cannot_be_silently_replaced_by_the_placeholder_card() {
    let or = card();
    let ph = AeroCard::from_file(&card_path(PLACEHOLDER_CARD)).unwrap();
    assert_ne!(or.mach.len(), ph.mach.len(), "the two cards share a Mach axis");
    assert_ne!(or.alpha_deg.len(), ph.alpha_deg.len(), "the two cards share an alpha axis");

    // Real OR data: nan gaps and a measured Cm channel. The placeholder card
    // has neither, and does carry the three control derivatives the OR flight
    // data never measured.
    let nan_count = |c: &AeroCard, ch: AeroChannel| {
        c.channel(ch).iter().flatten().filter(|v| v.is_nan()).count()
    };
    assert!(nan_count(&or, AeroChannel::Cd0) > 0, "OR card has no gaps");
    assert_eq!(nan_count(&ph, AeroChannel::Cd0), 0);
    assert!(!or.channel(AeroChannel::CmPitch).is_empty());
    assert!(ph.channel(AeroChannel::CmPitch).is_empty());
    for ch in [AeroChannel::Cna, AeroChannel::PitchDamping, AeroChannel::CmDelta] {
        assert!(or.channel(ch).is_empty(), "{ch:?} is not in the OR flight data");
        assert!(!ph.channel(ch).is_empty(), "{ch:?} missing from placeholder card");
    }

    let (mach, alpha, _, or_cd, _, _) = OR_SAMPLES[1];
    let ph_cd = ph.interpolate(AeroChannel::Cd0, mach, alpha).unwrap();
    assert!(
        (or_cd - ph_cd).abs() > 0.01,
        "OR Cd {or_cd} matches the placeholder card {ph_cd} at a sampled row"
    );
}
