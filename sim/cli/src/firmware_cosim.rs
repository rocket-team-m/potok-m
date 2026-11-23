// SPDX-License-Identifier: GPL-3.0-or-later
//! Co-simulation seam: potok-sim frame -> guard decision -> sim command.
//!
//! This is a HOST PORT of the firmware/AI guard decisions, not firmware code
//! and not a hardware-in-the-loop bridge. It runs no firmware binary, claims
//! no board, no timing and no electrical fidelity. See `docs/FIRMWARE_BRIDGE.md`
//! for the pinned provenance and why a Cargo link to the firmware crate is not
//! feasible inside this seam.
//!
//! The decisions below were transcribed from the read-only checkout at
//! `/home/sehaxe/potok-m-firmware` at [`FIRMWARE_COMMIT`]. Every constant is one
//! already published there or in `protocol/uart.yaml`; nothing is fitted to
//! make an arm win. [`verify_firmware_baseline`] is the LOUD gate: if the
//! pinned commit or the decision-bearing firmware source markers move, it
//! names the cause and the escape instead of letting the port drift silently.
//!
//! The seam is typed: [`Controller`] is a compile-time Rust trait. No config
//! string selects a function, nothing is loaded dynamically and no process is
//! spawned on the decision path. A future bridge that really links
//! `potok_m_firmware` implements the same trait; [`FirmwareGuard`] here is the
//! host-port implementation used today.
//!
//! Units: the firmware command is a raw 0.001 rad count bounded by
//! `MAX_CORRECTION_RAW` (100); the sim consumes a desired lateral acceleration
//! in m/s^2 (`MAX_COMMAND_MPS2`). No rad->m/s^2 scale is invented: the port
//! carries the decisions (phase/health/flags gate, three-strike latch, bounds,
//! rate reference, explicit reset) in the sim's native command unit, with
//! `MAX_COMMAND_MPS2` as the code-veto bound -- the exact analogue of the
//! firmware bound. A full swing is allowed between accepted commands, so the
//! oracle passes through bit-for-bit; the firmware's slew/throttle constants
//! are deliberately not reproduced (they have no sim-unit equivalent).
//!
//! `lib.rs` belongs to another lane, so this file is included by path from the
//! tests, the same pattern the firmware SIL harness uses for its parent-owned
//! `lib.rs`.

use std::cell::RefCell;

use potok_sim::{
    optimizer::ReferenceOptimizer,
    sensor::{Frame, FLAG_MOTOR_ARMED},
    Episode, Phase, SimError, Simulator, MAX_COMMAND_MPS2,
};

/// Git commit of the read-only firmware checkout this port was transcribed
/// from. The baseline gate refuses a stale port when that checkout moves.
pub const FIRMWARE_COMMIT: &str = "87118e07c243a89db2c5247c0a7e3bac9a74a8cf";
/// Reference checkout; never edited by this crate and never executed.
pub const FIRMWARE_REPO: &str = "/home/sehaxe/potok-m-firmware";
/// Decision-bearing sources and line ranges read from that checkout.
pub const FIRMWARE_SOURCES: &[(&str, &str)] = &[
    ("firmware/src/lib.rs", "41-78: constants, Command, clamped"),
    (
        "firmware/src/interlocks.rs",
        "35-112,130-137,239-321: phase gate, limits, latch/reset",
    ),
    ("firmware/src/control.rs", "205-279: miss latch, on_reply, command veto"),
];

/// A baseline check failed. Every variant carries the cause; `Display` adds
/// the escape (re-derive the port, then bump the pin).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProvenanceError {
    Unavailable(String),
    CommitMismatch {
        expected: &'static str,
        found: String,
    },
    SourceMismatch {
        file: &'static str,
        marker: &'static str,
    },
}

impl std::fmt::Display for ProvenanceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unavailable(what) => write!(f, "firmware baseline unavailable: {what}"),
            Self::CommitMismatch { expected, found } => write!(
                f,
                "firmware HEAD is {found}, but this port was derived from {expected}"
            ),
            Self::SourceMismatch { file, marker } => write!(
                f,
                "firmware source {file} no longer contains `{marker}`"
            ),
        }?;
        write!(
            f,
            " -- re-derive the host port, then bump FIRMWARE_COMMIT in cli/src/firmware_cosim.rs"
        )
    }
}

impl std::error::Error for ProvenanceError {}

/// Constants and gates the host port was transcribed from. A changed marker
/// means the copy may no longer be faithful and the sim must say so loudly.
const BASELINE_MARKERS: &[(&str, &str)] = &[
    ("firmware/src/lib.rs", "pub const MAX_CONSECUTIVE_MISSES: u8 = 3;"),
    ("firmware/src/lib.rs", "pub const MAX_CORRECTION_RAW: i16 = 100;"),
    ("firmware/src/interlocks.rs", "max_command: 100,"),
    ("firmware/src/interlocks.rs", "max_delta: 10,"),
    ("firmware/src/interlocks.rs", "min_interval_us: 1_000,"),
    ("firmware/src/control.rs", "if reply.ai_healthy == 0 {"),
    (
        "firmware/src/control.rs",
        "self.misses >= crate::MAX_CONSECUTIVE_MISSES",
    ),
];
/// Priority order inside `Interlocks::preflight` (phase, motor, health, fault).
const PREFLIGHT_TOKENS: [&str; 5] = [
    "PHASE_UNKNOWN",
    "PHASE_BOOST",
    "MOTOR_BIT",
    "UNHEALTHY_BIT",
    "FAULT_BIT",
];

/// Pure baseline check. `source` maps a pinned relative path to its text; no
/// process is spawned and no code from the checkout is executed.
pub fn check_firmware_baseline<F>(commit: &str, mut source: F) -> Result<(), ProvenanceError>
where
    F: FnMut(&str) -> Result<String, String>,
{
    if commit != FIRMWARE_COMMIT {
        return Err(ProvenanceError::CommitMismatch {
            expected: FIRMWARE_COMMIT,
            found: commit.to_owned(),
        });
    }
    for &(path, marker) in BASELINE_MARKERS {
        let text = source(path).map_err(ProvenanceError::Unavailable)?;
        if !text.contains(marker) {
            return Err(ProvenanceError::SourceMismatch { file: path, marker });
        }
    }
    let interlocks = source("firmware/src/interlocks.rs").map_err(ProvenanceError::Unavailable)?;
    let preflight = interlocks.split("fn preflight(").nth(1).ok_or(
        ProvenanceError::SourceMismatch {
            file: "firmware/src/interlocks.rs",
            marker: "fn preflight(",
        },
    )?;
    let mut cursor = 0;
    for token in PREFLIGHT_TOKENS {
        let at = preflight[cursor..].find(token).ok_or(
            ProvenanceError::SourceMismatch {
                file: "firmware/src/interlocks.rs",
                marker: token,
            },
        )? + cursor;
        cursor = at + token.len();
    }
    Ok(())
}

/// Check the live reference checkout against the pin. A missing checkout or a
/// moved HEAD/marker is a hard error with the escape in its `Display` text.
pub fn verify_firmware_baseline() -> Result<(), ProvenanceError> {
    let head_path = format!("{FIRMWARE_REPO}/.git/HEAD");
    let head = std::fs::read_to_string(&head_path)
        .map_err(|e| ProvenanceError::Unavailable(format!("{head_path}: {e}")))?;
    let head = head.trim();
    let commit = match head.strip_prefix("ref: ") {
        Some(reference) => {
            let path = format!("{FIRMWARE_REPO}/.git/{reference}");
            std::fs::read_to_string(&path)
                .map_err(|e| ProvenanceError::Unavailable(format!("{path}: {e}")))?
                .trim()
                .to_owned()
        }
        None => head.to_owned(),
    };
    check_firmware_baseline(&commit, |path| {
        let full = format!("{FIRMWARE_REPO}/{path}");
        std::fs::read_to_string(&full).map_err(|e| format!("{full}: {e}"))
    })
}

/// The sim's sensor frame rate (`config/default.toml`), i.e. one guard decision
/// per 20 ms. The firmware loop itself runs at 500 Hz; this seam samples it at
/// the simulator's frame tick, which is the only control cadence potok-sim has.
pub const SIM_TICK_HZ: u32 = 50;
/// `firmware/src/lib.rs`: three consecutive rejected frames latch AI off.
pub const MAX_CONSECUTIVE_MISSES: u8 = 3;
/// `firmware/src/lib.rs`: 100 raw counts at 0.001 rad/count = 0.1 rad bound.
pub const MAX_CORRECTION_RAW: i16 = 100;
/// `protocol/generated/protocol.rs` reply scale, pinned here so the sim needs
/// no direct `potok-protocol` dependency. `seam_constants_match_the_read_only_sources`
/// pins the value; this port never invents a rad->m/s^2 scale.
pub const REPLY_SCALE: f64 = 0.001;
/// Full swing between two legal commands. A command inside the envelope is
/// therefore never slewed, so the oracle passes through bit-for-bit.
pub const MAX_DELTA: f64 = 2.0 * MAX_COMMAND_MPS2;

/// Command in the sim seam's native unit: desired lateral acceleration [roll,
/// pitch] in m/s^2. Zero is the only failsafe, exactly as in the firmware.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Command {
    pub roll: f64,
    pub pitch: f64,
}

impl Command {
    pub const ZERO: Self = Self {
        roll: 0.0,
        pitch: 0.0,
    };

    pub const fn is_zero(self) -> bool {
        self.roll == 0.0 && self.pitch == 0.0
    }

    pub const fn finite(self) -> bool {
        self.roll.is_finite() && self.pitch.is_finite()
    }

    pub const fn within(self, bound: f64) -> bool {
        self.roll.abs() <= bound && self.pitch.abs() <= bound
    }

    /// Code veto, mirroring `Command::clamped` in the firmware: the guard never
    /// lets an unbounded demand reach the actuator.
    pub fn clamped(self, bound: f64) -> Self {
        Self {
            roll: self.roll.clamp(-bound, bound),
            pitch: self.pitch.clamp(-bound, bound),
        }
    }

    pub fn pair(self) -> (f64, f64) {
        (self.roll, self.pitch)
    }
}

/// Why a command was refused. The order and the latching class are the
/// firmware's (`interlocks::Refusal`) plus the AI guard's reject reasons.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Veto {
    Latched,
    UnknownPhase(u8),
    Boost,
    MotorArmed,
    AiUnhealthy,
    Fault,
    SensorInvalid,
    Implausible,
    NonFinite,
    RateLimit,
}

impl Veto {
    /// Firmware `Refusal::latches` plus the AI guard's three-strike latch on
    /// non-finite / rate-limit rejections (reached via `miss` below).
    pub const fn latches(self) -> bool {
        matches!(
            self,
            Self::UnknownPhase(_) | Self::AiUnhealthy | Self::Fault | Self::Implausible
        )
    }
}

/// Result of one guard tick.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Decision {
    /// What may reach the dynamics. ZERO on every veto.
    pub command: Command,
    pub veto: Option<Veto>,
    pub misses: u8,
    pub latched: bool,
}

/// Which compile-time implementation made a decision. Observability only:
/// neither variant claims a board, a firmware binary or an HIL run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Implementation {
    /// The host-translated decisions in this file.
    HostPort,
    /// A typed impl outside this file (currently the tests; later a real bridge).
    External,
}

/// Typed adapter from a sim frame and a proposed command to a guard decision.
/// The implementation is chosen at compile time: no config string, dynamic
/// library or subprocess participates. A bridge that really links
/// `potok_m_firmware` implements this trait instead of [`FirmwareGuard`].
pub trait Controller {
    fn decide(
        &mut self,
        frame: &Frame,
        proposed: Command,
        ai_healthy: bool,
        board_fault: bool,
    ) -> Decision;
    /// Clear latch, misses and rate reference, as the firmware reset does.
    fn reset(&mut self);
    fn implementation(&self) -> Implementation;
}

/// The decision chain carried across ticks: link three-strike counter, AI
/// latch, rate reference and observability counters. No clock, RNG or I/O.
#[derive(Clone, Debug)]
pub struct FirmwareGuard {
    last: Command,
    misses: u8,
    latched: Option<Veto>,
    last_veto: Option<Veto>,
    ticks: u64,
    accepted: u64,
    vetoed: u64,
    resets: u64,
}

impl Default for FirmwareGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl FirmwareGuard {
    pub const fn new() -> Self {
        Self {
            last: Command::ZERO,
            misses: 0,
            latched: None,
            last_veto: None,
            ticks: 0,
            accepted: 0,
            vetoed: 0,
            resets: 0,
        }
    }

    /// One 50 Hz decision with a healthy, board-fault-free model reply. This
    /// is the sim path: the oracle command is finite and bounded, and the
    /// frame's own `ai_input_valid` is the per-tick sensor-validity veto.
    pub fn step(&mut self, frame: &Frame, proposed: Command) -> Decision {
        self.step_with(frame, proposed, true, false)
    }

    /// Decision with explicit reply health and board fault, the two inputs the
    /// firmware SIL injects separately. Kept on the same code path so a fault
    /// cannot take a different route from a nominal tick.
    pub fn step_with(
        &mut self,
        frame: &Frame,
        proposed: Command,
        ai_healthy: bool,
        board_fault: bool,
    ) -> Decision {
        self.ticks += 1;
        if let Some(reason) = self.latched {
            return self.blocked(reason);
        }

        let code = frame.phase.protocol_code();
        let phase = match Phase::from_protocol_code(code) {
            Some(phase) => phase,
            None => return self.tripped(Veto::UnknownPhase(code)),
        };
        // BURN and MOTOR_ARMED are transient: burnout re-enables guidance with
        // no reset. Health and plausibility faults latch until explicit reset.
        // `protocol_code() == 0` is BURN on the wire and Boost in the sim.
        if phase == Phase::Boost {
            return self.blocked(Veto::Boost);
        }
        if frame.flags & FLAG_MOTOR_ARMED != 0 {
            return self.blocked(Veto::MotorArmed);
        }
        if !ai_healthy {
            return self.tripped(Veto::AiUnhealthy);
        }
        if board_fault {
            return self.tripped(Veto::Fault);
        }
        // AI02's dispatch AND rule: an invalid sensor sample vetoes this tick
        // only. It must not latch -- `AI_READY` is itself phase/pitot-gated in
        // potok-sim, so a transient low-pitot tick must not kill the flight.
        if !frame.ai_input_valid() {
            return self.blocked(Veto::SensorInvalid);
        }
        if !proposed.finite() {
            return self.missed(Veto::NonFinite);
        }
        if !proposed.within(MAX_COMMAND_MPS2) {
            return self.tripped(Veto::Implausible);
        }
        if (proposed.roll - self.last.roll).abs() > MAX_DELTA
            || (proposed.pitch - self.last.pitch).abs() > MAX_DELTA
        {
            return self.missed(Veto::RateLimit);
        }

        let out = proposed.clamped(MAX_COMMAND_MPS2);
        self.misses = 0;
        self.last = out;
        self.accepted += 1;
        Decision {
            command: out,
            veto: None,
            misses: 0,
            latched: false,
        }
    }

    /// The only way out of a latch. Mirrors `Interlocks::reset` /
    /// `LinkState::reset`: clears the rate reference as well as the latch.
    pub fn reset(&mut self) {
        self.latched = None;
        self.last_veto = None;
        self.misses = 0;
        self.last = Command::ZERO;
        self.resets += 1;
    }

    pub const fn ticks(&self) -> u64 {
        self.ticks
    }
    pub const fn accepted(&self) -> u64 {
        self.accepted
    }
    pub const fn vetoed(&self) -> u64 {
        self.vetoed
    }
    pub const fn resets(&self) -> u64 {
        self.resets
    }
    pub const fn misses(&self) -> u8 {
        self.misses
    }
    pub const fn latched(&self) -> Option<Veto> {
        self.latched
    }
    pub const fn last_veto(&self) -> Option<Veto> {
        self.last_veto
    }
    pub const fn last_command(&self) -> Command {
        self.last
    }

    fn blocked(&mut self, reason: Veto) -> Decision {
        self.vetoed += 1;
        self.last_veto = Some(reason);
        Decision {
            command: Command::ZERO,
            veto: Some(reason),
            misses: self.misses,
            latched: self.latched.is_some(),
        }
    }

    fn tripped(&mut self, reason: Veto) -> Decision {
        debug_assert!(reason.latches(), "only latching reasons may trip");
        self.latched = Some(reason);
        self.blocked(reason)
    }

    fn missed(&mut self, reason: Veto) -> Decision {
        self.misses = self.misses.saturating_add(1);
        if self.misses >= MAX_CONSECUTIVE_MISSES {
            self.latched = Some(Veto::Latched);
        }
        self.blocked(reason)
    }
}

impl Controller for FirmwareGuard {
    fn decide(
        &mut self,
        frame: &Frame,
        proposed: Command,
        ai_healthy: bool,
        board_fault: bool,
    ) -> Decision {
        self.step_with(frame, proposed, ai_healthy, board_fault)
    }

    fn reset(&mut self) {
        FirmwareGuard::reset(self);
    }

    fn implementation(&self) -> Implementation {
        Implementation::HostPort
    }
}

/// One co-simulated episode: the flown truth plus the controller's own trace.
#[derive(Debug)]
pub struct GuardedRun<C = FirmwareGuard> {
    pub episode: Episode,
    pub guard: C,
}

/// Drive one episode through the seam: oracle command -> guard -> dynamics.
pub fn run_guarded(sim: &Simulator, index: u64) -> Result<GuardedRun<FirmwareGuard>, SimError> {
    run_guarded_with(sim, index, |command| command)
}

/// Same seam with a transform on the proposed command, so a wrong-sign A/B
/// arm crosses the identical guard and actuator path.
pub fn run_guarded_with<F>(
    sim: &Simulator,
    index: u64,
    transform: F,
) -> Result<GuardedRun<FirmwareGuard>, SimError>
where
    F: Fn(Command) -> Command,
{
    run_guarded_with_controller(sim, index, FirmwareGuard::new(), transform)
}

/// Drive one episode through a caller-chosen [`Controller`]. The type is fixed
/// at compile time; no value read from a config file can select or alter it.
pub fn run_guarded_with_controller<C, F>(
    sim: &Simulator,
    index: u64,
    controller: C,
    transform: F,
) -> Result<GuardedRun<C>, SimError>
where
    C: Controller,
    F: Fn(Command) -> Command,
{
    let opt = ReferenceOptimizer::new(sim.config());
    let guard = RefCell::new(controller);
    let episode = sim.run_episode_with_control_ctx(index, &|ctx| {
        let g = opt.fin_accel_command(ctx.state, ctx.wind, ctx.vehicle, ctx.phase, ctx.frame);
        let proposed = transform(Command {
            roll: g[0],
            pitch: g[1],
        });
        guard
            .borrow_mut()
            .decide(ctx.frame, proposed, true, false)
            .command
            .pair()
    })?;
    Ok(GuardedRun {
        episode,
        guard: guard.into_inner(),
    })
}
