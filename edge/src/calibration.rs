//! Confidence calibration, loaded once from `calibration.json` (never in the
//! hot loop).
//!
//! The JSON schema mirrors `ai/src/potok_ai/calibration.py::CalibrationResult`
//! (read-only): `method` (`"temperature"` or `"isotonic"`), `temperature`,
//! `threshold`, plus diagnostics we ignore (`raw_ece`, `temperature_ece`,
//! `ece`, `coverage`, `n`, `loops`, `target_ece`, `model`, `seed`,
//! `config_hash`). Unknown fields are tolerated so a future Python field does
//! not break the reader.
//!
//! `method == "isotonic"` additionally needs the fitted step map: ascending
//! `bins` (upper edges of raw-sigmoid-confidence intervals, in `[0,1]`) and
//! `values` (calibrated output per interval), either at the top level or
//! nested under an `isotonic` object. An `n+1` edge array (lower edge first,
//! e.g. `[0.0, 0.1, …, 1.0]`) is accepted and normalised to `n` upper edges.
//! The Python fitter produces this map with
//! pool-adjacent-violators but does not yet serialise it; a file that selects
//! isotonic without a map is **refused loudly** rather than silently returning
//! uncalibrated confidence (ADR-style LOUD rule: a fallback must be visible).
//!
//! [`Calibration::apply`] takes the raw sigmoid confidence in `[0,1]`; its
//! temperature branch inverts the sigmoid, so it equals Python's
//! `sigmoid(z / T)` only while the raw confidence is not saturated. The
//! runtime uses [`Calibration::apply_logit`] instead: it receives the
//! pre-sigmoid scalar and is exact even for a saturated head, the very case
//! the Python fitter falls back to isotonic for. Isotonic looks up the step
//! map by `sigmoid(logit)`.

use serde::Deserialize;
use std::fmt;
use std::fs;
use std::path::Path;

/// Used when no calibration file is present, so legacy deployments keep the
/// originally specified 0.7 dispatch threshold.
pub const FALLBACK_THRESHOLD: f32 = 0.7;
const LOGIT_EPS: f64 = 1e-12;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CalibrationError {
    Io(String),
    Parse(String),
    Schema(String),
}

impl fmt::Display for CalibrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(s) => write!(f, "calibration file I/O error: {s}"),
            Self::Parse(s) => write!(f, "calibration file is not valid JSON: {s}"),
            Self::Schema(s) => write!(f, "calibration file rejected: {s}"),
        }
    }
}

impl std::error::Error for CalibrationError {}

#[derive(Debug, Deserialize)]
struct IsotonicMapFile {
    #[serde(default, alias = "edges", alias = "x")]
    bins: Option<Vec<f64>>,
    #[serde(default, alias = "y")]
    values: Option<Vec<f64>>,
}

#[derive(Debug, Deserialize)]
struct CalibrationFile {
    method: String,
    #[serde(default)]
    temperature: Option<f64>,
    threshold: f64,
    #[serde(default, alias = "isotonic_edges", alias = "isotonic_x")]
    bins: Option<Vec<f64>>,
    #[serde(default, alias = "isotonic_values", alias = "isotonic_y")]
    values: Option<Vec<f64>>,
    #[serde(default)]
    isotonic: Option<IsotonicMapFile>,
}

#[derive(Debug, Clone)]
enum Method {
    Identity,
    Temperature { temperature: f64 },
    Isotonic { bins: Vec<f64>, values: Vec<f64> },
}

/// A loaded calibration map plus the adaptive-compute threshold derived from
/// the same held-out data. Construct with [`Calibration::identity`] when no
/// file is supplied.
#[derive(Debug, Clone)]
pub struct Calibration {
    method: Method,
    threshold: f32,
}

impl Calibration {
    /// Neutral fallback: confidence passes through unchanged, dispatch uses
    /// [`FALLBACK_THRESHOLD`].
    pub fn identity() -> Self {
        Self {
            method: Method::Identity,
            threshold: FALLBACK_THRESHOLD,
        }
    }

    /// Read and validate `calibration.json`. Any problem is a hard error with
    /// the cause in the message — never a silent fallback to identity.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, CalibrationError> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)
            .map_err(|e| CalibrationError::Io(format!("{}: {e}", path.display())))?;
        Self::from_json(&text)
    }

    /// Parse a calibration document. Public so tests can exercise the schema
    /// without touching the filesystem; production goes through `from_file`.
    pub fn from_json(json: &str) -> Result<Self, CalibrationError> {
        let raw: CalibrationFile =
            serde_json::from_str(json).map_err(|e| CalibrationError::Parse(e.to_string()))?;

        if !raw.threshold.is_finite() || !(0.0..=1.0).contains(&raw.threshold) {
            return Err(CalibrationError::Schema(format!(
                "threshold must be finite and in [0,1], got {}",
                raw.threshold
            )));
        }

        let method = match raw.method.as_str() {
            "temperature" => {
                let t = raw.temperature.ok_or_else(|| {
                    CalibrationError::Schema("method=temperature requires a temperature".into())
                })?;
                if !t.is_finite() || t <= 0.0 {
                    return Err(CalibrationError::Schema(format!(
                        "temperature must be finite and > 0, got {t}"
                    )));
                }
                Method::Temperature { temperature: t }
            }
            "isotonic" => {
                let (bins, values) = match raw.isotonic {
                    Some(m) => (m.bins, m.values),
                    None => (raw.bins, raw.values),
                };
                let bins = bins.ok_or_else(|| {
                    CalibrationError::Schema(
                        "method=isotonic requires bins (isotonic map not serialised by the Python fitter yet)".into(),
                    )
                })?;
                let values = values.ok_or_else(|| {
                    CalibrationError::Schema("method=isotonic requires values".into())
                })?;
                // Accept both `[e0..en]` edges (n+1) and upper edges (n).
                if values.is_empty()
                    || (bins.len() != values.len() && bins.len() != values.len() + 1)
                {
                    return Err(CalibrationError::Schema(format!(
                        "isotonic bins/values must be non-empty with n or n+1 bins ({} vs {})",
                        bins.len(),
                        values.len()
                    )));
                }
                let mut prev = -f64::INFINITY;
                for (i, b) in bins.iter().enumerate() {
                    if !b.is_finite() || !(0.0..=1.0).contains(b) || *b <= prev {
                        return Err(CalibrationError::Schema(format!(
                            "isotonic bins must be finite, ascending, and in [0,1] (bin {i} = {b})"
                        )));
                    }
                    prev = *b;
                }
                for (i, v) in values.iter().enumerate() {
                    if !v.is_finite() || !(0.0..=1.0).contains(v) {
                        return Err(CalibrationError::Schema(format!(
                            "isotonic values must be finite and in [0,1] (value {i} = {v})"
                        )));
                    }
                }
                let bins = if bins.len() == values.len() + 1 {
                    bins[1..].to_vec()
                } else {
                    bins
                };
                Method::Isotonic { bins, values }
            }
            other => {
                return Err(CalibrationError::Schema(format!(
                    "unknown method {other:?}; expected temperature or isotonic"
                )))
            }
        };

        Ok(Self {
            method,
            threshold: raw.threshold as f32,
        })
    }

    /// Calibrate a raw sigmoid confidence in `[0,1]`. Identity when no file
    /// was loaded. For a saturated head prefer [`Self::apply_logit`]: inverting
    /// a probability that already rounded to 0 or 1 destroys information.
    pub fn apply(&self, raw_confidence: f32) -> f32 {
        if !raw_confidence.is_finite() {
            return raw_confidence;
        }
        let p = (raw_confidence as f64).clamp(0.0, 1.0);
        match &self.method {
            Method::Identity => p as f32,
            Method::Temperature { temperature } => {
                let p = p.clamp(LOGIT_EPS, 1.0 - LOGIT_EPS);
                let logit = (p / (1.0 - p)).ln();
                sigmoid(logit / temperature) as f32
            }
            Method::Isotonic { .. } => self.isotonic_lookup(p) as f32,
        }
    }

    /// Calibrate from the head's pre-sigmoid scalar. This is the path the
    /// runtime uses and is exactly `ai/calibration.py`'s `sigmoid(z / T)`.
    pub fn apply_logit(&self, logit: f32) -> f32 {
        if !logit.is_finite() {
            return if logit.is_sign_negative() { 0.0 } else { 1.0 };
        }
        let z = logit as f64;
        match &self.method {
            Method::Identity => sigmoid(z) as f32,
            Method::Temperature { temperature } => sigmoid(z / temperature) as f32,
            Method::Isotonic { .. } => self.isotonic_lookup(sigmoid(z)) as f32,
        }
    }

    fn isotonic_lookup(&self, p: f64) -> f64 {
        match &self.method {
            Method::Isotonic { bins, values } => {
                // First bin whose upper edge is >= p; past the last edge, hold
                // the final value. Right-continuous PAVA step map.
                let index = bins.partition_point(|edge| *edge < p);
                values[index.min(values.len() - 1)]
            }
            _ => p,
        }
    }

    /// Adaptive-compute threshold from the calibration file, or 0.7 for the
    /// no-file fallback.
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    pub fn is_identity(&self) -> bool {
        matches!(self.method, Method::Identity)
    }

    /// One-line description for the LOUD startup log.
    pub fn description(&self) -> String {
        match &self.method {
            Method::Identity => "identity (no calibration.json)".into(),
            Method::Temperature { temperature } => format!("temperature T={temperature}"),
            Method::Isotonic { bins, .. } => format!("isotonic ({} bins)", bins.len()),
        }
    }
}

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_is_exact() {
        let c = Calibration::identity();
        assert!(c.is_identity());
        assert_eq!(c.apply(0.73f32), 0.73);
        assert_eq!(c.threshold(), FALLBACK_THRESHOLD);
    }

    #[test]
    fn temperature_matches_python_formula() {
        let c = Calibration::from_json(
            r#"{"method":"temperature","temperature":2.0,"threshold":0.55}"#,
        )
        .unwrap();
        let raw = 0.9f64;
        let logit = (raw / (1.0 - raw)).ln();
        let expected = sigmoid(logit / 2.0) as f32;
        assert!((c.apply(raw as f32) - expected).abs() < 1e-6);
        assert_eq!(c.threshold(), 0.55);
    }

    #[test]
    fn apply_logit_is_exact_for_a_saturated_head() {
        let json = r#"{"method":"temperature","temperature":5.0,"threshold":0.5}"#;
        let c = Calibration::from_json(json).unwrap();
        // f32 sigmoid(20) is 0.99999994: the logit is nearly lost. The
        // probability-facing path inverts that rounded value and drifts;
        // `apply_logit` still reproduces Python's sigmoid(20 / 5) exactly.
        let raw = sigmoid(20.0) as f32;
        assert!(raw > 0.999_999);
        let expected = sigmoid(4.0) as f32;
        assert!((c.apply_logit(20.0) - expected).abs() < 1e-7);
        assert!((c.apply(raw) - expected).abs() > 1e-3);
    }

    #[test]
    fn isotonic_step_map_and_clamping() {
        let c = Calibration::from_json(
            r#"{"method":"isotonic","threshold":0.6,
                "bins":[0.25,0.5,0.75],"values":[0.1,0.4,0.9]}"#,
        )
        .unwrap();
        assert_eq!(c.apply(0.1), 0.1);
        assert_eq!(c.apply(0.25), 0.1);
        assert_eq!(c.apply(0.26), 0.4);
        assert_eq!(c.apply(0.5), 0.4);
        assert_eq!(c.apply(0.9), 0.9);
    }

    #[test]
    fn nested_isotonic_map_is_accepted() {
        let c = Calibration::from_json(
            r#"{"method":"isotonic","threshold":0.6,
                "isotonic":{"bins":[0.5,1.0],"values":[0.2,0.8]}}"#,
        )
        .unwrap();
        assert_eq!(c.apply(0.4), 0.2);
        assert_eq!(c.apply(1.0), 0.8);
    }

    #[test]
    fn isotonic_edge_array_form() {
        let c = Calibration::from_json(
            r#"{"method":"isotonic","threshold":0.5,
                "bins":[0.0,0.25,0.5,1.0],"values":[0.1,0.4,0.9]}"#,
        )
        .unwrap();
        assert_eq!(c.apply(0.1), 0.1);
        assert_eq!(c.apply(0.3), 0.4);
        assert_eq!(c.apply(0.7), 0.9);
    }

    #[test]
    fn malformed_inputs_fail_loudly() {
        assert!(Calibration::from_json("{not json").is_err());
        assert!(Calibration::from_json(r#"{"method":"mystery","threshold":0.5}"#).is_err());
        assert!(Calibration::from_json(r#"{"method":"temperature","threshold":0.5}"#).is_err());
        assert!(Calibration::from_json(
            r#"{"method":"isotonic","threshold":0.5,"bins":[0.5],"values":[0.4,0.6]}"#
        )
        .is_err());
        assert!(Calibration::from_json(
            r#"{"method":"temperature","temperature":2,"threshold":9}"#
        )
        .is_err());
    }

    #[test]
    fn python_diagnostic_fields_are_ignored() {
        let c = Calibration::from_json(
            r#"{"threshold":0.62,"ece":0.04,"raw_ece":0.31,"temperature_ece":0.07,
                "coverage":0.8,"temperature":1.5,"method":"temperature","loops":2,
                "n":120,"target_ece":0.05,"model":"lite","seed":1,"config_hash":"abc"}"#,
        )
        .unwrap();
        assert_eq!(c.threshold(), 0.62);
        assert!(!c.is_identity());
        assert!(c.description().contains("temperature"));
    }
}
