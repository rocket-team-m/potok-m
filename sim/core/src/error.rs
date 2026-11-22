//! Errors from configuration, IO and simulation.

use thiserror::Error;

pub type Result<T> = std::result::Result<T, SimError>;

#[derive(Debug, Error)]
pub enum SimError {
    #[error("configuration error: {0}")]
    Config(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("non-finite simulation state at step {step}: {what}")]
    NonFinite { step: usize, what: &'static str },
    #[error("invalid validation file: {0}")]
    InvalidValidation(String),
    #[error("aero card error: {0}")]
    AeroCard(String),
    #[error("aero card out of grid: {axis}={value} outside [{lo}, {hi}]")]
    AeroOutOfGrid {
        axis: String,
        value: f64,
        lo: f64,
        hi: f64,
    },
    #[error("control input error: {0}")]
    Control(String),
    #[error("trajectory sidecar error: {0}")]
    Trajectory(String),
}
