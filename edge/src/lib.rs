//! `potok-edge` — fixed-size CPU inference for the Potok-M assist loop.
//!
//! Public API:
//!   `Model::from_file` -> `Runtime::new` -> `Runtime::on_request` -> `ReplyFrame`
//!
//! See README.md for the model binary layout and the protocol mapping.

pub mod baseline;
pub mod fixture;
pub mod format_v2;
pub mod frame;
pub mod model;
pub mod runtime;
pub mod simd;
pub mod speculative;
pub mod student_runtime;
pub mod uart;

pub use format_v2::{ModelV2, V2Error};
pub use frame::{FrameError, FrameLayout, ReplyFrame, RequestFrame};
pub use model::{Model, ModelError};
pub use runtime::{Counters, PhasePolicy, Runtime};
pub use speculative::{DispatchSource, DivergenceEvent, SpecCounters, SpeculativeRuntime};
pub use student_runtime::{
    build_student_fixture, is_student_file, model_kind, StudentError, StudentFixtureConfig,
    StudentModel, StudentRuntime,
};
pub use uart::Transport;
