//! Aerodynamic coefficient card: 2-D grids over (Mach, angle of attack).
//!
//! The simulator has two sources for `cd0`, `cna`, `pitch_damping` and
//! `cm_delta`:
//!
//! * `placeholder` (default) — the analytic constants and the 1-D Mach drag
//!   table in `config/default.toml`; the legacy code path is byte-identical
//!   to the one before cards existed. Selecting it increments a COUNTED
//!   marker (`aero::placeholder_uses()`) and prints one line to stderr.
//! * `card` — a TOML file with one grid per coefficient; values are bilinearly
//!   interpolated and **any** query outside a grid is a LOUD `AeroOutOfGrid`
//!   error. Nothing is clamped and nothing falls back silently. A grid node that
//!   has no measurement is `nan` and any bilinear stencil touching it is also
//!   LOUD. Channels absent from the file (`cna`, `pitch_damping`, `cm_delta`
//!   in the OpenRocket-derived card) are LOUD on query, never zero-filled.
//!   The refusal carries its own provenance and escape; see `docs/AERO_CARD.md`.

use std::{
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
};

use serde::Deserialize;

use crate::{
    error::{Result, SimError},
    math::lerp_table,
};

static PLACEHOLDER_USES: AtomicU64 = AtomicU64::new(0);

/// Escape text carried by every absent-channel refusal. The OR flight data
/// behind `config/aero_card.toml` measured only Mach (col 50), alpha (col 15),
/// axial Cd (col 35) and pitch Cm (col 37); it has no fin-deflection,
/// normal-force or pitch-rate signal, so `cna`, `pitch_damping` and `cm_delta`
/// cannot be derived from it. A refusal names the escape, not just the hole.
pub const ABSENT_CHANNEL_ESCAPE: &str =
    "supply a measured grid in the card, or select the placeholder path";

/// COUNTED marker: number of times the placeholder coefficient path was
/// selected since process start. A reader of the run log sees it without a
/// profiler.
pub fn placeholder_uses() -> u64 {
    PLACEHOLDER_USES.load(Ordering::Relaxed)
}

/// Select the placeholder path and return the new count. Called once per
/// `Simulator::new`, which also prints the marker.
pub fn count_placeholder() -> u64 {
    PLACEHOLDER_USES.fetch_add(1, Ordering::Relaxed) + 1
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AeroChannel {
    Cd0,
    Cna,
    PitchDamping,
    CmDelta,
    /// Raw pitching-moment coefficient from OpenRocket flight data. Not a
    /// control derivative: `cm_delta` stays the fin authority channel.
    CmPitch,
}

impl AeroChannel {
    pub fn name(self) -> &'static str {
        match self {
            AeroChannel::Cd0 => "cd0",
            AeroChannel::Cna => "cna",
            AeroChannel::PitchDamping => "pitch_damping",
            AeroChannel::CmDelta => "cm_delta",
            AeroChannel::CmPitch => "cm_pitch",
        }
    }
}

/// Effective coefficients for one integration step. `cd0` is the **total**
/// axial drag coefficient (the per-episode `cd0` shift is already applied by
/// the caller); the other three are the raw grid values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AeroCoeffs {
    pub cd0: f64,
    pub cna: f64,
    pub pitch_damping: f64,
    pub cm_delta: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AeroCard {
    /// Mach axis, strictly increasing.
    pub mach: Vec<f64>,
    /// Angle-of-attack axis in degrees, strictly increasing.
    pub alpha_deg: Vec<f64>,
    /// `values[alpha][mach]` for each channel. `None` means the source data
    /// did not contain the channel; querying it is LOUD.
    cd0: Vec<Vec<f64>>,
    cna: Option<Vec<Vec<f64>>>,
    pitch_damping: Option<Vec<Vec<f64>>>,
    cm_delta: Option<Vec<Vec<f64>>>,
    cm_pitch: Option<Vec<Vec<f64>>>,
    /// Placeholder drag baseline. A per-episode `cd0` of this value means
    /// "use the card's drag values verbatim"; another value shifts the whole
    /// grid by the difference, matching the legacy `(cd0 - cd_base) + table`.
    pub baseline_cd0: f64,
}

#[derive(Debug, Deserialize)]
struct RawCard {
    baseline_cd0: f64,
    grid: RawGrid,
    coefficients: RawCoefficients,
}

#[derive(Debug, Deserialize)]
struct RawGrid {
    mach: Vec<f64>,
    alpha_deg: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct RawCoefficients {
    cd0: Vec<Vec<f64>>,
    #[serde(default)]
    cna: Option<Vec<Vec<f64>>>,
    #[serde(default)]
    pitch_damping: Option<Vec<Vec<f64>>>,
    #[serde(default)]
    cm_delta: Option<Vec<Vec<f64>>>,
    #[serde(default)]
    cm_pitch: Option<Vec<Vec<f64>>>,
}

impl AeroCard {
    pub fn from_toml_str(s: &str) -> Result<Self> {
        let raw: RawCard =
            toml::from_str(s).map_err(|e| SimError::AeroCard(format!("TOML parse error: {e}")))?;
        Self::from_raw(raw)
    }

    pub fn from_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| SimError::AeroCard(format!("{}: {e}", path.display())))?;
        Self::from_toml_str(&text)
            .map_err(|e| SimError::AeroCard(format!("{}: {e}", path.display())))
    }

    fn from_raw(raw: RawCard) -> Result<Self> {
        if !raw.baseline_cd0.is_finite() {
            return Err(SimError::AeroCard("baseline_cd0 is not finite".into()));
        }
        check_axis("grid.mach", &raw.grid.mach)?;
        check_axis("grid.alpha_deg", &raw.grid.alpha_deg)?;
        let rows = raw.grid.alpha_deg.len();
        let cols = raw.grid.mach.len();
        let channels = [
            (AeroChannel::Cd0, Some(&raw.coefficients.cd0)),
            (AeroChannel::Cna, raw.coefficients.cna.as_ref()),
            (
                AeroChannel::PitchDamping,
                raw.coefficients.pitch_damping.as_ref(),
            ),
            (AeroChannel::CmDelta, raw.coefficients.cm_delta.as_ref()),
            (AeroChannel::CmPitch, raw.coefficients.cm_pitch.as_ref()),
        ];
        for (ch, grid) in channels {
            let Some(grid) = grid else { continue };
            if grid.len() != rows {
                return Err(SimError::AeroCard(format!(
                    "{} has {} alpha rows, expected {rows}",
                    ch.name(),
                    grid.len()
                )));
            }
            for (i, row) in grid.iter().enumerate() {
                if row.len() != cols {
                    return Err(SimError::AeroCard(format!(
                        "{} row {i} has {} mach values, expected {cols}",
                        ch.name(),
                        row.len()
                    )));
                }
                // `nan` is the explicit "no measurement here" marker used by
                // the OpenRocket-derived card; infinities are always a defect.
                if row.iter().any(|v| v.is_infinite()) {
                    return Err(SimError::AeroCard(format!(
                        "{} row {i} has an infinite value",
                        ch.name()
                    )));
                }
            }
        }
        Ok(Self {
            mach: raw.grid.mach,
            alpha_deg: raw.grid.alpha_deg,
            cd0: raw.coefficients.cd0,
            cna: raw.coefficients.cna,
            pitch_damping: raw.coefficients.pitch_damping,
            cm_delta: raw.coefficients.cm_delta,
            cm_pitch: raw.coefficients.cm_pitch,
            baseline_cd0: raw.baseline_cd0,
        })
    }

    pub fn channel(&self, ch: AeroChannel) -> &[Vec<f64>] {
        match ch {
            AeroChannel::Cd0 => &self.cd0,
            AeroChannel::Cna => self.cna.as_deref().unwrap_or(&[]),
            AeroChannel::PitchDamping => self.pitch_damping.as_deref().unwrap_or(&[]),
            AeroChannel::CmDelta => self.cm_delta.as_deref().unwrap_or(&[]),
            AeroChannel::CmPitch => self.cm_pitch.as_deref().unwrap_or(&[]),
        }
    }

    /// Bilinear interpolation of one channel at `(mach, alpha_deg)`.
    ///
    /// A query outside either axis returns `SimError::AeroOutOfGrid`; there is
    /// no clamp and no fallback. A `nan` node in the bilinear stencil means the
    /// source data did not measure that corner and returns a LOUD card error.
    /// Grid nodes with data return the stored value exactly.
    pub fn interpolate(&self, ch: AeroChannel, mach: f64, alpha_deg: f64) -> Result<f64> {
        let grid = self.channel(ch);
        if grid.is_empty() {
            return Err(SimError::AeroCard(format!(
                "channel {} is absent from this card; no value was invented; {ABSENT_CHANNEL_ESCAPE}",
                ch.name()
            )));
        }
        let (mi, mu) = interp_index("mach", &self.mach, mach)?;
        let (ai, au) = interp_index("alpha_deg", &self.alpha_deg, alpha_deg)?;
        let v00 = grid[ai][mi];
        let v10 = grid[ai][mi + 1];
        let v01 = grid[ai + 1][mi];
        let v11 = grid[ai + 1][mi + 1];
        if !v00.is_finite() || !v10.is_finite() || !v01.is_finite() || !v11.is_finite() {
            return Err(SimError::AeroCard(format!(
                "channel {} has no measured data at mach={mach}, alpha_deg={alpha_deg}",
                ch.name()
            )));
        }
        // Flat rows are the common case (a card generated from the constant
        // placeholder coefficients). Collapsing to one linear axis keeps the
        // result bit-identical to `lerp_table` at those nodes.
        if v00 == v01 && v10 == v11 {
            return Ok(v00 + mu * (v10 - v00));
        }
        Ok((1.0 - au) * ((1.0 - mu) * v00 + mu * v10) + au * ((1.0 - mu) * v01 + mu * v11))
    }

    /// All four coefficients at one point. The `cd0` shift relative to
    /// `baseline_cd0` is applied here so the card can stand in for the legacy
    /// `cd = (cd0 - cd_base) + lerp_table(cd_mach, mach)` expression.
    pub fn coeffs(&self, mach: f64, alpha_deg: f64, episode_cd0: f64) -> Result<AeroCoeffs> {
        Ok(AeroCoeffs {
            cd0: self.interpolate(AeroChannel::Cd0, mach, alpha_deg)?
                + (episode_cd0 - self.baseline_cd0),
            cna: self.interpolate(AeroChannel::Cna, mach, alpha_deg)?,
            pitch_damping: self.interpolate(AeroChannel::PitchDamping, mach, alpha_deg)?,
            cm_delta: self.interpolate(AeroChannel::CmDelta, mach, alpha_deg)?,
        })
    }

    /// Placeholder coefficients: the exact legacy expressions, kept as the
    /// reference used to generate `config/aero_card.toml`.
    pub fn placeholder_cd(cfg_cd0: f64, table: &[(f64, f64)], mach: f64) -> f64 {
        let cd_base = table.first().map(|p| p.1).unwrap_or(cfg_cd0);
        (cfg_cd0 - cd_base) + lerp_table(table, mach)
    }
}

fn check_axis(name: &str, axis: &[f64]) -> Result<()> {
    if axis.len() < 2 {
        return Err(SimError::AeroCard(format!(
            "{name} needs at least two knots"
        )));
    }
    for w in axis.windows(2) {
        if !w[0].is_finite() || !w[1].is_finite() || w[1] <= w[0] {
            return Err(SimError::AeroCard(format!(
                "{name} must be finite and strictly increasing"
            )));
        }
    }
    Ok(())
}

/// Segment + fraction with a LOUD out-of-grid error. No clamping, no epsilon
/// forgiveness: a grid node matches exactly and anything past the ends is a
/// defect in the card or in the requested flight condition.
fn interp_index(axis: &str, knots: &[f64], x: f64) -> Result<(usize, f64)> {
    if !x.is_finite() {
        return Err(SimError::AeroOutOfGrid {
            axis: axis.into(),
            value: x,
            lo: knots[0],
            hi: knots[knots.len() - 1],
        });
    }
    if x < knots[0] || x > knots[knots.len() - 1] {
        return Err(SimError::AeroOutOfGrid {
            axis: axis.into(),
            value: x,
            lo: knots[0],
            hi: knots[knots.len() - 1],
        });
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
