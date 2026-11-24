//! Errors from the measurement ingest boundary.
//!
//! Every variant names the file, row, column or field that failed and the
//! fix. There is no zero-fill, clamp or silent substitution anywhere in this
//! crate.

use thiserror::Error;

pub type Result<T> = std::result::Result<T, MeasurementError>;

#[derive(Debug, Error)]
pub enum MeasurementError {
    #[error("measurement IO: {0}")]
    Io(#[from] std::io::Error),
    #[error("measurement recipe {path}: {detail}")]
    Recipe { path: String, detail: String },
    #[error("measurement CSV {path}: {detail}")]
    Csv { path: String, detail: String },
    #[error("measurement identification input rejected: {0}")]
    Identification(String),
    #[error("measurement JSON: {0}")]
    Json(#[from] serde_json::Error),
}
