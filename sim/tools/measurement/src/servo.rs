//! Servo step-response CSV -> validated [`ServoStepResponse`].
//!
//! The identified parameters (delay, rate, deadband) are part of the
//! measurement record, carried as metadata beside the raw step rows. The
//! harness does not fit them from the trace: fitting would be a second,
//! unrecorded estimator and a place to invent data.

use potok_sim::identification::{ServoStep, ServoStepResponse};

use crate::csv;
use crate::error::{MeasurementError, Result};
use crate::provenance::SourceRecord;

pub const KIND: &str = "servo_step";
pub const COLUMNS: &[&str] = &["t_s", "commanded", "measured"];

/// A servo step response together with the identity of its source file.
#[derive(Debug, Clone, PartialEq)]
pub struct ServoStepMeasurement {
    pub response: ServoStepResponse,
    pub provenance: SourceRecord,
}

pub fn ingest(text: &str, path: &str) -> Result<ServoStepMeasurement> {
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
    let delay_s = doc.meta_f64("delay_s")?;
    let rate_per_s = doc.meta_f64("rate_per_s")?;
    let deadband = doc.meta_f64("deadband")?;
    let samples: Vec<ServoStep> = doc
        .rows
        .iter()
        .map(|r| ServoStep {
            t_s: r.cells[0],
            commanded: r.cells[1],
            measured: r.cells[2],
        })
        .collect();
    let response = ServoStepResponse::new(delay_s, rate_per_s, deadband, samples)
        .map_err(|e| crate::error::MeasurementError::Identification(e.to_string()))?;
    let provenance = SourceRecord::new(&doc, text, KIND, &label);
    Ok(ServoStepMeasurement { response, provenance })
}
