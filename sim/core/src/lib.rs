//! Deterministic 6-DoF simulator for the Potok-M rocket AI training data.
//!
//! All stochastic inputs derive from `config.run.seed` and the episode index.
//! There is no global RNG, no wall-clock dependence, and no host/GPU coupling.

pub mod aero;
pub mod aero_derivatives;
pub mod atmosphere;
pub mod config;
pub mod control;
pub mod dynamics;
pub mod episode;
pub mod error;
pub mod firmware_adapter;
pub mod firmware_dataset;
pub mod flight_replay;
pub mod identification;
pub mod manifest;
pub mod math;
pub mod optimizer;
pub mod parachute;
pub mod physics;
pub mod playback3d;
pub mod quality;
pub mod reward;
pub mod rollout;
pub mod scenario;
pub mod sensor;
pub mod trajectory;
pub mod validation;
pub mod wind;

pub use aero::{AeroCard, AeroChannel, AeroCoeffs};
pub use atmosphere::Atmosphere;
pub use config::{AeroSource, Conditioning, Config, DomainParams, OutputMode};
pub use control::{validate_command, ControlFile, MAX_COMMAND_MPS2, MAX_CONTROL_DEG};
pub use dynamics::{
    cm_delta_placeholder_uses, fin_clamp_uses, fin_no_authority_uses, solve_fin_deflection,
    ControlFault, Phase, State, Vehicle, MAX_FIN_DEFLECTION_RAD, MIN_CONTROL_QDYN_PA,
    PLACEHOLDER_CM_DELTA,
};
pub use episode::{ControlContext, Episode, EpisodeMeta, Episodes, Simulator};
pub use error::{Result, SimError};
pub use manifest::{config_hash, file_record, read_manifest, write_manifest, DatasetManifest};
pub use optimizer::{ReferenceOptimizer, Target};
pub use physics::{PhysicsFeatures, PHYSICS_FEATURES};
pub use scenario::{is_held_out, scenario, Scenario, Split, SCENARIO_COUNT};
pub use sensor::{FlagSet, Frame, Sensor};
pub use trajectory::{
    apogee_m, lateral_rms_m, read_trajectory, sidecar_path, write_trajectory, write_trajectory_v2,
    write_trajectory_v3, write_trajectory_with_version, EpisodeTrajectory, TrajectorySet,
    TrajectoryTick, TRUTH_STRIDE, TRUTH_TICK_NAMES, TRUTH_V3_STRIDE, TRUTH_V3_TICK_NAMES,
    VERSION_V2, VERSION_V3,
};
pub use validation::{TruthRecord, LEGACY_TARGET_COUNT, META_V2_TRUTH, TRUTH_COUNT};
pub use wind::Wind;
