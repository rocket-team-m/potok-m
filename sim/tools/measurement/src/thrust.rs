//! Static-fire thrust stand CSV -> validated [`ThrustCurve`].

use potok_sim::identification::{ThrustCurve, ThrustSample};

use crate::csv;
use crate::error::{MeasurementError, Result};
use crate::provenance::SourceRecord;

pub const KIND: &str = "thrust_stand";
pub const COLUMNS: &[&str] = &["t_s", "thrust_n"];

/// A thrust curve together with the identity of the file it came from.
#[derive(Debug, Clone, PartialEq)]
pub struct ThrustStandMeasurement {
    pub curve: ThrustCurve,
    pub provenance: SourceRecord,
}

/// Parse and validate a thrust stand CSV. Physical validation (finite,
/// non-negative, strictly increasing time, at least two knots) is delegated to
/// the crate's existing typed contract, so no second rulebook can drift.
pub fn ingest(text: &str, path: &str) -> Result<ThrustStandMeasurement> {
    let doc = csv::parse(text, path, COLUMNS)?;
    let declared = doc.meta_required("measurement")?;
    if declared != KIND {
        return Err(MeasurementError::Csv {
            path: path.to_string(),
            detail: format!(
                "metadata 'measurement' is '{declared}', expected '{KIND}'; point this recipe at the right file"
            ),
        });
    }
    let label = doc.meta_required("source")?.to_string();
    let samples: Vec<ThrustSample> = doc
        .rows
        .iter()
        .map(|r| ThrustSample {
            t_s: r.cells[0],
            thrust_n: r.cells[1],
        })
        .collect();
    let curve = ThrustCurve::new(samples)
        .map_err(|e| MeasurementError::Identification(e.to_string()))?;
    let provenance = SourceRecord::new(&doc, text, KIND, &label);
    Ok(ThrustStandMeasurement { curve, provenance })
}
