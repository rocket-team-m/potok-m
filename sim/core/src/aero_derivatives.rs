// SPDX-License-Identifier: GPL-3.0-or-later
//! Typed aerodynamic derivative slots for the extracted OpenRocket card.
//!
//! `config/aero_card.toml` measures axial drag and pitching moment only. It has
//! no `cna`, `pitch_damping` or `cm_delta`: those channels are absent and every
//! query for one is LOUD. Nothing in this module edits the card or `aero.rs`.
//!
//! The three missing channels get a typed home here. A slot is either a
//! measured `(Mach, alpha_deg)` grid or `None`; reading an empty slot is an
//! error naming the channel, never `0.0`. [`FlightCard`] pairs a read-only
//! [`AeroCard`] with those slots and refuses to produce coefficients until all
//! three have been supplied.

use crate::{
    aero::AeroChannel,
    error::{Result, SimError},
    AeroCard, AeroCoeffs,
};

/// The three derivatives the OR flight data cannot supply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Derivative {
    Cna,
    PitchDamping,
    CmDelta,
}

impl Derivative {
    /// Declaration order is the order used in error messages and in
    /// [`DerivativeSlots::missing_names`]; it never depends on map iteration.
    pub const ALL: [Self; 3] = [Self::Cna, Self::PitchDamping, Self::CmDelta];

    pub fn name(self) -> &'static str {
        match self {
            Self::Cna => "cna",
            Self::PitchDamping => "pitch_damping",
            Self::CmDelta => "cm_delta",
        }
    }

    pub fn channel(self) -> AeroChannel {
        match self {
            Self::Cna => AeroChannel::Cna,
            Self::PitchDamping => AeroChannel::PitchDamping,
            Self::CmDelta => AeroChannel::CmDelta,
        }
    }
}

/// The three values needed alongside the card's drag and pitching moment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DerivedCoeffs {
    pub cna: f64,
    pub pitch_damping: f64,
    pub cm_delta: f64,
}

/// One measured grid, laid out as `values[alpha][mach]`.
///
/// `nan` means "not measured": a stencil touching it is LOUD. Infinities are
/// rejected at construction because they are a defect, not a missing value.
#[derive(Debug, Clone)]
pub struct DerivativeGrid {
    derivative: Derivative,
    mach: Vec<f64>,
    alpha_deg: Vec<f64>,
    values: Vec<Vec<f64>>,
}

impl DerivativeGrid {
    pub fn new(
        derivative: Derivative,
        mach: Vec<f64>,
        alpha_deg: Vec<f64>,
        values: Vec<Vec<f64>>,
    ) -> Result<Self> {
        check_axis("mach", &mach, derivative)?;
        check_axis("alpha_deg", &alpha_deg, derivative)?;
        if values.len() != alpha_deg.len() {
            return Err(SimError::AeroCard(format!(
                "{} grid has {} alpha rows, expected {}",
                derivative.name(),
                values.len(),
                alpha_deg.len()
            )));
        }
        for (i, row) in values.iter().enumerate() {
            if row.len() != mach.len() {
                return Err(SimError::AeroCard(format!(
                    "{} grid row {i} has {} mach values, expected {}",
                    derivative.name(),
                    row.len(),
                    mach.len()
                )));
            }
            if row.iter().any(|v| v.is_infinite()) {
                return Err(SimError::AeroCard(format!(
                    "{} grid row {i} has an infinite value",
                    derivative.name()
                )));
            }
        }
        Ok(Self {
            derivative,
            mach,
            alpha_deg,
            values,
        })
    }

    pub fn derivative(&self) -> Derivative {
        self.derivative
    }

    pub fn interpolate(&self, mach: f64, alpha_deg: f64) -> Result<f64> {
        bilinear(
            self.derivative.name(),
            &self.mach,
            &self.alpha_deg,
            &self.values,
            mach,
            alpha_deg,
        )
    }
}

/// One slot per derivative. No slot has a numeric default: an unsupplied slot
/// stays `None` and every read carries an error naming the absent channel.
#[derive(Debug, Clone, Default)]
pub struct DerivativeSlots {
    cna: Option<DerivativeGrid>,
    pitch_damping: Option<DerivativeGrid>,
    cm_delta: Option<DerivativeGrid>,
}

impl DerivativeSlots {
    pub fn empty() -> Self {
        Self::default()
    }

    /// Supply one grid. A second grid for the same derivative is a LOUD error
    /// rather than an overwrite, so a card cannot silently change mid-run.
    pub fn with(mut self, grid: DerivativeGrid) -> Result<Self> {
        let slot = match grid.derivative() {
            Derivative::Cna => &mut self.cna,
            Derivative::PitchDamping => &mut self.pitch_damping,
            Derivative::CmDelta => &mut self.cm_delta,
        };
        if slot.is_some() {
            return Err(SimError::AeroCard(format!(
                "derivative {} was supplied twice",
                grid.derivative().name()
            )));
        }
        *slot = Some(grid);
        Ok(self)
    }

    pub fn get(&self, derivative: Derivative) -> Result<&DerivativeGrid> {
        let slot = match derivative {
            Derivative::Cna => &self.cna,
            Derivative::PitchDamping => &self.pitch_damping,
            Derivative::CmDelta => &self.cm_delta,
        };
        slot.as_ref().ok_or_else(|| {
            SimError::AeroCard(format!(
                "derivative {} is absent from this card; no value was invented",
                derivative.name()
            ))
        })
    }

    /// Absent channel names in declaration order. Empty when ready.
    pub fn missing_names(&self) -> Vec<&'static str> {
        Derivative::ALL
            .iter()
            .copied()
            .filter(|d| self.get(*d).is_err())
            .map(Derivative::name)
            .collect()
    }

    pub fn is_complete(&self) -> bool {
        Derivative::ALL.iter().all(|d| self.get(*d).is_ok())
    }

    pub fn interpolate(&self, derivative: Derivative, mach: f64, alpha_deg: f64) -> Result<f64> {
        self.get(derivative)?.interpolate(mach, alpha_deg)
    }

    /// All three derivatives at one flight point. Missing any one of them is
    /// LOUD; there is no partial struct and no zero fill.
    pub fn all_at(&self, mach: f64, alpha_deg: f64) -> Result<DerivedCoeffs> {
        Ok(DerivedCoeffs {
            cna: self.interpolate(Derivative::Cna, mach, alpha_deg)?,
            pitch_damping: self.interpolate(Derivative::PitchDamping, mach, alpha_deg)?,
            cm_delta: self.interpolate(Derivative::CmDelta, mach, alpha_deg)?,
        })
    }
}

/// A read-only measured card plus the three typed derivative slots. The card
/// itself is never modified: an episode driven by the untouched card still
/// refuses to fly, exactly as before these slots existed.
#[derive(Debug, Clone)]
pub struct FlightCard {
    base: AeroCard,
    derivatives: DerivativeSlots,
}

impl FlightCard {
    pub fn new(base: AeroCard) -> Self {
        Self {
            base,
            derivatives: DerivativeSlots::empty(),
        }
    }

    pub fn with_derivative(mut self, grid: DerivativeGrid) -> Result<Self> {
        self.derivatives = self.derivatives.with(grid)?;
        Ok(self)
    }

    pub fn base(&self) -> &AeroCard {
        &self.base
    }

    pub fn derivatives(&self) -> &DerivativeSlots {
        &self.derivatives
    }

    pub fn is_ready(&self) -> bool {
        self.derivatives.is_complete()
    }

    /// Full coefficients at one flight point. Readiness is checked first, so
    /// an unsupplied card is refused before any grid is queried and the error
    /// names every absent channel. The `cd0` shift matches `AeroCard::coeffs`.
    pub fn coeffs(&self, mach: f64, alpha_deg: f64, episode_cd0: f64) -> Result<AeroCoeffs> {
        if !self.is_ready() {
            return Err(SimError::AeroCard(format!(
                "flight card is not ready: {} absent; no derivative was invented",
                self.derivatives.missing_names().join(", ")
            )));
        }
        let d = self.derivatives.all_at(mach, alpha_deg)?;
        let cd0 = self.base.interpolate(AeroChannel::Cd0, mach, alpha_deg)?
            + (episode_cd0 - self.base.baseline_cd0);
        Ok(AeroCoeffs {
            cd0,
            cna: d.cna,
            pitch_damping: d.pitch_damping,
            cm_delta: d.cm_delta,
        })
    }
}

/// Deterministic bilinear interpolation over `values[alpha][mach]`.
///
/// The knot search is a linear scan in declaration order: no RNG, no map
/// iteration and no epsilon forgiveness. A grid node reads back exactly,
/// anything past either end is `AeroOutOfGrid`, and a non-finite corner is a
/// LOUD card error. Flat rows collapse to one linear axis, matching
/// `AeroCard::interpolate` bit for bit.
pub fn bilinear(
    channel: &str,
    mach_knots: &[f64],
    alpha_knots: &[f64],
    values: &[Vec<f64>],
    mach: f64,
    alpha_deg: f64,
) -> Result<f64> {
    if values.len() != alpha_knots.len() || values.iter().any(|r| r.len() != mach_knots.len()) {
        return Err(SimError::AeroCard(format!(
            "channel {channel} grid shape does not match its axes"
        )));
    }
    let (mi, mu) = interp_index("mach", mach_knots, mach)?;
    let (ai, au) = interp_index("alpha_deg", alpha_knots, alpha_deg)?;
    let v00 = values[ai][mi];
    let v10 = values[ai][mi + 1];
    let v01 = values[ai + 1][mi];
    let v11 = values[ai + 1][mi + 1];
    if !v00.is_finite() || !v10.is_finite() || !v01.is_finite() || !v11.is_finite() {
        return Err(SimError::AeroCard(format!(
            "channel {channel} has no measured data at mach={mach}, alpha_deg={alpha_deg}"
        )));
    }
    if v00 == v01 && v10 == v11 {
        return Ok(v00 + mu * (v10 - v00));
    }
    Ok((1.0 - au) * ((1.0 - mu) * v00 + mu * v10) + au * ((1.0 - mu) * v01 + mu * v11))
}

fn check_axis(name: &str, axis: &[f64], derivative: Derivative) -> Result<()> {
    if axis.len() < 2 {
        return Err(SimError::AeroCard(format!(
            "{} {name} needs at least two knots",
            derivative.name()
        )));
    }
    for w in axis.windows(2) {
        if !w[0].is_finite() || !w[1].is_finite() || w[1] <= w[0] {
            return Err(SimError::AeroCard(format!(
                "{} {name} must be finite and strictly increasing",
                derivative.name()
            )));
        }
    }
    Ok(())
}

/// Same segment + fraction rule as the card: no clamp, no tolerance.
fn interp_index(axis: &str, knots: &[f64], x: f64) -> Result<(usize, f64)> {
    let out_of_grid = || SimError::AeroOutOfGrid {
        axis: axis.into(),
        value: x,
        lo: knots[0],
        hi: knots[knots.len() - 1],
    };
    if !x.is_finite() || x < knots[0] || x > knots[knots.len() - 1] {
        return Err(out_of_grid());
    }
    let last = knots.len() - 1;
    if x == knots[last] {
        return Ok((last - 1, 1.0));
    }
    for i in 1..=last {
        if x <= knots[i] {
            return Ok((i - 1, (x - knots[i - 1]) / (knots[i] - knots[i - 1])));
        }
    }
    Ok((last - 1, 1.0))
}
