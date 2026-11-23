// SPDX-License-Identifier: GPL-3.0-or-later
//! Gates for the typed derivative slots added around the extracted OR card.
//!
//! `core/src/aero_derivatives.rs` is included by path because `lib.rs` and
//! `aero.rs` are out of scope for this task: the module is additive and the
//! read-only card is never edited. The tests pin three things: an absent slot
//! is LOUD and never zero-filled, the bilinear grid reading is deterministic
//! and bit-identical to the card's own interpolation, and the extracted card
//! still refuses to fly until all three derivatives are supplied.

#[path = "../src/aero_derivatives.rs"]
mod aero_derivatives;

use std::path::{Path, PathBuf};

// The module below is included by path, so its `crate::` paths resolve here.
pub use potok_sim::{aero, error, AeroCard, AeroCoeffs};

use aero_derivatives::{bilinear, Derivative, DerivativeGrid, DerivativeSlots, FlightCard};
use potok_sim::{config::AeroSource, AeroChannel, Config, SimError, Simulator};

const CARD: &str = "../config/aero_card.toml";
// Clean cd0 stencil used by the existing OR card gates.
const MACH: f64 = 0.493;
const ALPHA: f64 = 0.006904141;
const EPISODE_CD0: f64 = 0.30;

// (mach, alpha_deg, pinned card Cd) from OR Simulation 9, copied verbatim from
// `aero_card_tests.rs`. The exact readings make a silent interpolation drift
// fail before any tolerance would hide it.
const OR_SAMPLES: &[(f64, f64, f64)] = &[
    (0.020000, 14.896902673, 0.577527814100),
    (0.179000, 0.401070457, 0.360083698610),
    (0.493000, 0.006904141, 0.390683060057),
    (0.733000, 0.014558858, 0.393330003311),
    (0.407000, 0.028974476, 0.319184870070),
    (0.215000, 0.114591559, 0.299889960517),
    (0.059000, 3.437746771, 0.352317429833),
];

fn card_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(CARD)
}

fn card() -> AeroCard {
    AeroCard::from_file(&card_path()).expect("the extracted OR card must load")
}

fn constant_grid(
    derivative: Derivative,
    value: f64,
    mach: &[f64],
    alpha: &[f64],
) -> DerivativeGrid {
    DerivativeGrid::new(
        derivative,
        mach.to_vec(),
        alpha.to_vec(),
        vec![vec![value; mach.len()]; alpha.len()],
    )
    .expect("constant grid must validate")
}

fn supply_all(card: &AeroCard, values: [(Derivative, f64); 3]) -> FlightCard {
    let (m, a) = (card.mach.clone(), card.alpha_deg.clone());
    let mut flight = FlightCard::new(card.clone());
    for (d, v) in values {
        flight = flight
            .with_derivative(constant_grid(d, v, &m, &a))
            .expect("distinct derivative slots must accept one grid each");
    }
    flight
}

#[test]
fn typed_slots_are_loud_when_absent_and_never_zero_filled() {
    let slots = DerivativeSlots::empty();
    assert!(!slots.is_complete());
    assert_eq!(
        slots.missing_names(),
        vec!["cna", "pitch_damping", "cm_delta"]
    );

    // Debug output is the cheapest witness that every slot is `None`: a
    // numeric default would show Some(..) and could quietly become 0.0.
    let text = format!("{slots:?}");
    for field in ["cna: None", "pitch_damping: None", "cm_delta: None"] {
        assert!(
            text.contains(field),
            "slot {field} was replaced by a value: {text}"
        );
    }

    for d in Derivative::ALL {
        match slots.get(d) {
            Err(SimError::AeroCard(msg)) => {
                assert!(msg.contains(d.name()), "{d:?}: {msg}");
                assert!(msg.contains("absent"), "{d:?}: {msg}");
                assert!(msg.contains("no value was invented"), "{d:?}: {msg}");
            }
            other => panic!("expected a missing-slot error for {d:?}, got {other:?}"),
        }
        match slots.interpolate(d, MACH, ALPHA) {
            Err(SimError::AeroCard(msg)) => assert!(msg.contains(d.name()), "{d:?}: {msg}"),
            other => panic!("expected a missing-slot interpolation error for {d:?}, got {other:?}"),
        }
    }
    match slots.all_at(MACH, ALPHA) {
        Err(SimError::AeroCard(msg)) => assert!(msg.contains("cna"), "{msg}"),
        other => panic!("expected all_at to refuse an empty slot set, got {other:?}"),
    }

    // Supplying a slot twice is also loud: no overwrite, no last-write-wins.
    let g = constant_grid(Derivative::Cna, 9.0, &[0.0, 1.0], &[0.0, 10.0]);
    let once = slots.with(g.clone()).expect("first slot");
    match once.with(g) {
        Err(SimError::AeroCard(msg)) => {
            assert!(msg.contains("cna") && msg.contains("twice"), "{msg}")
        }
        other => panic!("expected duplicate-slot refusal, got {other:?}"),
    }
}

#[test]
fn bilinear_interpolation_is_deterministic_and_matches_the_extracted_card() {
    // Same stencil, same arithmetic and the same flat-row collapse as the
    // card's own `interpolate`: bit-for-bit equality on the real grid, not a
    // tolerance that would let two different algorithms agree.
    let card = card();
    let cd0 = card.channel(AeroChannel::Cd0);
    for &(mach, alpha, pinned) in OR_SAMPLES {
        let mine = bilinear("cd0", &card.mach, &card.alpha_deg, cd0, mach, alpha)
            .unwrap_or_else(|e| panic!("cd0 at ({mach}, {alpha}): {e}"));
        let theirs = card
            .interpolate(AeroChannel::Cd0, mach, alpha)
            .expect("card reading");
        assert_eq!(
            mine.to_bits(),
            theirs.to_bits(),
            "interpolation drift at ({mach}, {alpha})"
        );
        assert!(
            (mine - pinned).abs() < 1e-8,
            "card reading drift at ({mach}, {alpha})"
        );
        let again = bilinear("cd0", &card.mach, &card.alpha_deg, cd0, mach, alpha).unwrap();
        assert_eq!(
            mine.to_bits(),
            again.to_bits(),
            "a repeated query changed the answer"
        );
    }

    // A genuinely bilinear (non-flat) stencil has one exact answer:
    // v00=0, v10=4, v01=10, v11=20 at (mu=0.5, au=0.5) -> 8.5.
    let mach = [0.0_f64, 2.0];
    let alpha = [0.0_f64, 10.0];
    let values = [vec![0.0, 4.0], vec![10.0, 20.0]];
    assert_eq!(
        bilinear("x", &mach, &alpha, &values, 1.0, 5.0).unwrap(),
        8.5
    );
    for (m, a, want) in [
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 4.0),
        (0.0, 10.0, 10.0),
        (2.0, 10.0, 20.0),
    ] {
        assert_eq!(
            bilinear("x", &mach, &alpha, &values, m, a).unwrap(),
            want,
            "node ({m}, {a})"
        );
    }

    // Both failure classes stay LOUD, with the card's exact grid bounds.
    match bilinear("cd0", &card.mach, &card.alpha_deg, cd0, 0.0049, 1.0) {
        Err(SimError::AeroOutOfGrid {
            axis,
            lo,
            hi,
            value,
        }) => {
            assert_eq!(axis, "mach");
            assert_eq!(value, 0.0049);
            assert_eq!(lo, card.mach[0]);
            assert_eq!(hi, *card.mach.last().unwrap());
        }
        other => panic!("expected AeroOutOfGrid on mach, got {other:?}"),
    }
    match bilinear("cd0", &card.mach, &card.alpha_deg, cd0, 0.493, f64::NAN) {
        Err(SimError::AeroOutOfGrid { axis, .. }) => assert_eq!(axis, "alpha_deg"),
        other => panic!("expected AeroOutOfGrid on NaN alpha, got {other:?}"),
    }
    match bilinear("cd0", &card.mach, &card.alpha_deg, cd0, 0.021, 0.0007) {
        Err(SimError::AeroCard(msg)) => assert!(msg.contains("no measured data"), "{msg}"),
        other => panic!("expected a nan-stencil refusal, got {other:?}"),
    }

    // Grid construction validates shape and rejects infinities; NaN is kept
    // as a deliberate "not measured" marker and is only refused on query.
    assert!(DerivativeGrid::new(
        Derivative::Cna,
        vec![0.0, 1.0],
        vec![0.0, 10.0],
        vec![vec![1.0]]
    )
    .is_err());
    assert!(DerivativeGrid::new(
        Derivative::Cna,
        vec![0.0, 0.0],
        vec![0.0, 10.0],
        vec![vec![1.0, 1.0]; 2],
    )
    .is_err());
    assert!(DerivativeGrid::new(
        Derivative::Cna,
        vec![0.0, 1.0],
        vec![0.0, 10.0],
        vec![vec![1.0, f64::INFINITY]; 2],
    )
    .is_err());
    let with_gap = DerivativeGrid::new(
        Derivative::Cna,
        vec![0.0, 1.0],
        vec![0.0, 10.0],
        vec![vec![1.0, f64::NAN], vec![1.0, 1.0]],
    )
    .expect("NaN is the missing-data marker");
    match with_gap.interpolate(1.0, 0.0) {
        Err(SimError::AeroCard(msg)) => assert!(
            msg.contains("cna") && msg.contains("no measured data"),
            "{msg}"
        ),
        other => panic!("expected a missing-node refusal, got {other:?}"),
    }
}

#[test]
fn extracted_or_card_still_refuses_to_fly_until_derivatives_are_supplied() {
    let card = card();

    // 1. The extracted card itself still has no derivative channels.
    for d in Derivative::ALL {
        assert!(
            card.channel(d.channel()).is_empty(),
            "{d:?} was written into the read-only card"
        );
        match card.interpolate(d.channel(), MACH, ALPHA) {
            Err(SimError::AeroCard(msg)) => assert!(msg.contains("absent"), "{d:?}: {msg}"),
            other => panic!("expected an absent-channel error for {d:?}, got {other:?}"),
        }
    }

    // 2. The wrapped card refuses before derivatives arrive, naming all three.
    let bare = FlightCard::new(card.clone());
    assert!(!bare.is_ready());
    match bare.coeffs(MACH, ALPHA, EPISODE_CD0) {
        Err(SimError::AeroCard(msg)) => {
            assert!(
                msg.contains("not ready") && msg.contains("no derivative was invented"),
                "{msg}"
            );
            for d in Derivative::ALL {
                assert!(msg.contains(d.name()), "{d:?} missing from {msg}");
            }
        }
        other => panic!("expected an unready-card refusal, got {other:?}"),
    }

    // 3. A partial supply is still a refusal, and names only what is missing:
    //    one real grid never licenses a zero for its siblings.
    let partial = FlightCard::new(card.clone())
        .with_derivative(constant_grid(
            Derivative::Cna,
            9.0,
            &card.mach,
            &card.alpha_deg,
        ))
        .expect("cna slot");
    assert!(!partial.is_ready());
    match partial.coeffs(MACH, ALPHA, EPISODE_CD0) {
        Err(SimError::AeroCard(msg)) => {
            assert!(
                msg.contains("pitch_damping") && msg.contains("cm_delta"),
                "{msg}"
            );
            assert!(!msg.contains("cna"), "cna was supplied: {msg}");
        }
        other => panic!("expected a partial-supply refusal, got {other:?}"),
    }

    // 4. The simulator, still pointed at the untouched card, still refuses a
    //    full episode. Adding typed slots does not fly an unmeasured card.
    let mut cfg = Config::default();
    cfg.aero.source = AeroSource::Card;
    cfg.aero.card_path = Some(PathBuf::from("config/aero_card.toml"));
    let sim = Simulator::new(cfg).expect("card config must load");
    let err = sim.run_episode(0).unwrap_err();
    assert!(
        matches!(err, SimError::AeroOutOfGrid { .. } | SimError::AeroCard(_)),
        "the extracted card must still refuse to fly, got {err:?}"
    );

    // 5. Supplying all three distinct grids makes the wrapper ready and its
    //    reading deterministic; cd0 still comes from the card plus its shift.
    let ready = supply_all(
        &card,
        [
            (Derivative::Cna, 9.0),
            (Derivative::PitchDamping, 7.0),
            (Derivative::CmDelta, -3.0),
        ],
    );
    assert!(ready.is_ready());
    assert!(ready.derivatives().missing_names().is_empty());
    let got = ready.coeffs(MACH, ALPHA, EPISODE_CD0).expect("ready card");
    assert_eq!(got.cna, 9.0);
    assert_eq!(got.pitch_damping, 7.0);
    assert_eq!(got.cm_delta, -3.0);
    let want_cd0 = card.interpolate(AeroChannel::Cd0, MACH, ALPHA).unwrap()
        + (EPISODE_CD0 - card.baseline_cd0);
    assert_eq!(got.cd0.to_bits(), want_cd0.to_bits());
    assert_eq!(got, ready.coeffs(MACH, ALPHA, EPISODE_CD0).unwrap());

    // Supply order must not change the result.
    let reordered = supply_all(
        &card,
        [
            (Derivative::CmDelta, -3.0),
            (Derivative::PitchDamping, 7.0),
            (Derivative::Cna, 9.0),
        ],
    );
    assert_eq!(got, reordered.coeffs(MACH, ALPHA, EPISODE_CD0).unwrap());
    assert_eq!(format!("{ready:?}"), format!("{reordered:?}"));
}
