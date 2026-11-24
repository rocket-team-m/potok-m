//! LOUD ingest of system-identification measurements.
//!
//! Three CSV dialects (static-fire thrust stand, IMU Allan deviation, servo
//! step response) become the typed, already-validated contracts in
//! `potok_sim::identification`, each bound to a SHA-256 record of the exact
//! bytes parsed and the operator-supplied source label.
//!
//! Nothing here fits a model, tunes a trajectory, edits a config on its own,
//! or claims a hardware measurement. The only fixtures in this tree are
//! labelled `SYNTHETIC` and exist to self-test the instrument. See
//! `docs/MEASUREMENT_PROTOCOL.md`.

pub mod bundle;
pub mod csv;
pub mod error;
pub mod imu;
pub mod provenance;
pub mod servo;
pub mod thrust;

pub use bundle::{
    assert_shipped_recipe_is_synthetic, default_paths, ingest_recipe, CalibrationBundle,
    CalibrationReport, Recipe, SensorCalibration,
};
pub use error::{MeasurementError, Result};
pub use provenance::{Provenance, SourceRecord};
