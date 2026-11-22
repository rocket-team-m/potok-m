//! Deterministic 6-DoF simulator for the Potok-M rocket AI training data.
//!
//! All stochastic inputs derive from `config.run.seed` and the episode index.
//! There is no global RNG, no wall-clock dependence, and no host/GPU coupling.

pub mod aero;
pub mod atmosphere;
pub mod config;
pub mod control;
pub mod dynamics;
pub mod episode;
pub mod error;
pub mod manifest;
pub mod math;
pub mod optimizer;
pub mod physics;
pub mod quality;
pub mod scenario;
pub mod sensor;
pub mod trajectory;
pub mod validation;
pub mod wind;

pub use aero::{AeroCard, AeroChannel, AeroCoeffs};
pub use atmosphere::Atmosphere;
pub use config::{AeroSource, Conditioning, Config, DomainParams, OutputMode};
pub use control::{validate_command, ControlFile, MAX_CONTROL_DEG};
pub use dynamics::{Phase, State, Vehicle};
pub use episode::{Episode, EpisodeMeta, Episodes, Simulator};
pub use error::{Result, SimError};
pub use manifest::{config_hash, file_record, read_manifest, write_manifest, DatasetManifest};
pub use optimizer::{ReferenceOptimizer, Target};
pub use physics::{PhysicsFeatures, PHYSICS_FEATURES};
pub use scenario::{is_held_out, scenario, Scenario, Split, SCENARIO_COUNT};
pub use sensor::{FlagSet, Frame, Sensor};
pub use trajectory::{
    apogee_m, lateral_rms_m, read_trajectory, sidecar_path, write_trajectory, write_trajectory_v2,
    write_trajectory_with_version, EpisodeTrajectory, TrajectorySet, TrajectoryTick, TRUTH_STRIDE,
    TRUTH_TICK_NAMES, VERSION_V2,
};
pub use validation::{TruthRecord, META_V2_TRUTH, TRUTH_COUNT};
pub use wind::Wind;
