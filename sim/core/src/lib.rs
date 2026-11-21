//! Deterministic 6-DoF simulator for the Potok-M rocket AI training data.
//!
//! All stochastic inputs derive from `config.run.seed` and the episode index.
//! There is no global RNG, no wall-clock dependence, and no host/GPU coupling.

pub mod atmosphere;
pub mod config;
pub mod dynamics;
pub mod episode;
pub mod error;
pub mod math;
pub mod optimizer;
pub mod sensor;
pub mod validation;
pub mod wind;

pub use atmosphere::Atmosphere;
pub use config::{Config, OutputMode};
pub use dynamics::{Phase, State, Vehicle};
pub use episode::{Episode, EpisodeMeta, Episodes, Simulator};
pub use error::{Result, SimError};
pub use optimizer::{ReferenceOptimizer, Target};
pub use sensor::{FlagSet, Frame, Sensor};
pub use wind::Wind;
