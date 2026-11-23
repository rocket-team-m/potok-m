// SPDX-License-Identifier: GPL-3.0-or-later
//! Co-simulation seam: potok-sim frame -> firmware guard decision -> sim command.
//!
//! Read-only reference (never edited by this file):
//! `/home/sehaxe/potok-m-firmware/firmware/src/{lib,control,interlocks,phase,
//! watchdog,sil}.rs` and `/home/sehaxe/potok-m/ai/src/potok_ai/watchguard.py`.
//! This is a host port of those *decisions*, not of their hardware I/O. Every
//! constant below is one already published there or in `protocol/uart.yaml`;
//! nothing is fitted to make an arm win.
//!
//! The flight controller's command is an angular correction in raw 0.001 rad
//! counts, while the sim's control seam consumes a desired lateral acceleration
//! in m/s^2 (`MAX_COMMAND_MPS2`). No rad->m/s^2 scale is invented: the port
//! carries the decisions (phase/health/flags gate, three-strike latch, bounds,
//! rate reference, explicit reset) in the sim's native command unit, with
//! `MAX_COMMAND_MPS2` as the code-veto bound -- the exact analogue of the
//! firmware's `MAX_CORRECTION_RAW`.
//!
//! `lib.rs` belongs to another lane, so this file is included by path from
//! `core/tests/firmware_cosim_tests.rs`, the same pattern the firmware SIL
//! harness uses for its own parent-owned `lib.rs`.

use std::cell::RefCell;

use potok_protocol as pp;
use potok_sim::{
    optimizer::ReferenceOptimizer, sensor::Frame, Episode, SimError, Simulator, MAX_COMMAND_MPS2,
};

/// The sim's sensor frame rate (`config/default.toml`), i.e. one guard decision
/// per 20 ms. The firmware loop itself runs at 500 Hz; this seam samples it at
/// the simulator's frame tick, which is the only control cadence potok-sim has.
pub const SIM_TICK_HZ: u32 = 50;
/// `firmware/src/lib.rs`: three consecutive rejected frames latch AI off.
pub const MAX_CONSECUTIVE_MISSES: u8 = 3;
/// `firmware/src/lib.rs`: 100 raw counts at 0.001 rad/count = 0.1 rad bound.
pub const MAX_CORRECTION_RAW: i16 = 100;
/// `protocol/uart.yaml` reply scale. Pinned here so the port cannot drift from
/// the wire contract even though the sim works in m/s^2.
pub const REPLY_SCALE: f64 = pp::REPLY_G_PITCH_SCALE;
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

        let phase = match pp::Phase::try_from(frame.phase.protocol_code()) {
            Ok(phase) => phase,
            Err(code) => return self.tripped(Veto::UnknownPhase(code)),
        };
        // BURN and MOTOR_ARMED are transient: burnout re-enables guidance with
        // no reset. Health and plausibility faults latch until explicit reset.
        if phase == pp::Phase::BURN {
            return self.blocked(Veto::Boost);
        }
        if frame.flags & pp::flags::MOTOR_ARMED != 0 {
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

/// One co-simulated episode: the flown truth plus the guard's own trace.
#[derive(Debug)]
pub struct GuardedRun {
    pub episode: Episode,
    pub guard: FirmwareGuard,
}

/// Drive one episode through the seam: oracle command -> guard -> dynamics.
pub fn run_guarded(sim: &Simulator, index: u64) -> Result<GuardedRun, SimError> {
    run_guarded_with(sim, index, |command| command)
}

/// Same seam with a transform on the proposed command, so a wrong-sign A/B
/// arm crosses the identical guard and actuator path.
pub fn run_guarded_with<F>(
    sim: &Simulator,
    index: u64,
    transform: F,
) -> Result<GuardedRun, SimError>
where
    F: Fn(Command) -> Command,
{
    let opt = ReferenceOptimizer::new(sim.config());
    let guard = RefCell::new(FirmwareGuard::new());
    let episode = sim.run_episode_with_control_ctx(index, &|ctx| {
        let g = opt.fin_accel_command(ctx.state, ctx.wind, ctx.vehicle, ctx.phase, ctx.frame);
        let proposed = transform(Command {
            roll: g[0],
            pitch: g[1],
        });
        guard.borrow_mut().step(ctx.frame, proposed).command.pair()
    })?;
    Ok(GuardedRun {
        episode,
        guard: guard.into_inner(),
    })
}
