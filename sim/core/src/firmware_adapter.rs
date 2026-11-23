// SPDX-License-Identifier: GPL-3.0-or-later
//! Safe boundary between a simulator sensor frame and the command the AI hands
//! to the flight controller.
//!
//! Nothing here executes code from configuration: no string dispatch, no
//! dynamic loading, no process spawning, no function pointers built from data.
//! The implementation is a Rust type chosen at compile time; [`FirmwareAdapter::reset`]
//! accepts only a numeric seed.
//!
//! The host seam mirrors the host-testable part of `potok-m-firmware`:
//! `LinkState::on_reply` (`firmware/src/control.rs:148`), its `command` helper
//! (`:178`), `on_timeout` (`:123`) and `reset` (`:137`), plus `PhaseFsm`'s
//! phase gate (`firmware/src/phase_fsm.rs:108`). This crate has no Cargo
//! dependency on the firmware repo; a bridge crate implements [`HostControl`]
//! for the real `LinkState` and maps its raw i16 command to the sim's m/s²
//! demands.

use crate::{control::MAX_COMMAND_MPS2, dynamics::Phase, sensor::Frame};

/// One AI command tick: desired world-frame lateral accelerations in m/s², in
/// the same `[ay, ax]` order as `optimizer::Target::g_accel`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Command {
    /// World +Y lateral acceleration demand, m/s².
    pub ay: f32,
    /// World +X lateral acceleration demand, m/s².
    pub ax: f32,
    /// The AI producer's self-report, independent of frame-level faults.
    pub ai_healthy: bool,
}

impl Command {
    /// Safe output: no demand, AI reported healthy.
    pub const ZERO: Self = Self {
        ay: 0.0,
        ax: 0.0,
        ai_healthy: true,
    };

    pub const fn is_zero(self) -> bool {
        self.ay == 0.0 && self.ax == 0.0
    }

    /// A demand outside the sim's command box is refused, never clamped.
    pub fn checked(self) -> Option<Self> {
        let limit = MAX_COMMAND_MPS2 as f32;
        if !self.ay.is_finite()
            || !self.ax.is_finite()
            || self.ay.abs() > limit
            || self.ax.abs() > limit
        {
            return None;
        }
        Some(self)
    }
}

/// Produce the AI command for one sensor frame. Implementations own their
/// state: the firmware owns phase transitions, so callers tell the adapter the
/// active phase through [`FirmwareAdapter::on_phase`].
pub trait FirmwareAdapter {
    fn step(&mut self, frame: &Frame) -> Command;
    /// Restore deterministic initial state with a new numeric seed. Callers
    /// must re-apply the current phase with `on_phase` afterwards.
    fn reset(&mut self, seed: u64);
    fn on_phase(&mut self, phase: Phase);
}

/// SplitMix64 finalizer: a fixed integer mixer, not a configurable algorithm.
fn mix(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    x
}

/// Deterministic stand-in for the AI producer. Every demand is a pure function
/// of `(seed, tick, frame.seq, frame.time_us)`; no global RNG, no wall clock.
#[derive(Clone, Debug)]
pub struct MockAdapter {
    seed: u64,
    phase: Phase,
    tick: u64,
    healthy: bool,
}

impl MockAdapter {
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            phase: Phase::Pad,
            tick: 0,
            healthy: true,
        }
    }

    /// A producer that reports itself unhealthy on every tick.
    pub fn unhealthy(seed: u64) -> Self {
        Self {
            healthy: false,
            ..Self::new(seed)
        }
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn tick(&self) -> u64 {
        self.tick
    }

    pub fn phase(&self) -> Phase {
        self.phase
    }

    fn demand(&self, salt: u64, frame: &Frame) -> f32 {
        let frame_key =
            mix(frame.seq as u64 ^ (frame.time_us as u64).wrapping_mul(0x9e3779b97f4a7c15));
        let h = mix(self.seed ^ frame_key ^ mix(self.tick.wrapping_add(salt)));
        let unit = (h >> 40) as f64 / (1u64 << 24) as f64;
        ((2.0 * unit - 1.0) * MAX_COMMAND_MPS2) as f32
    }
}

impl FirmwareAdapter for MockAdapter {
    fn step(&mut self, frame: &Frame) -> Command {
        self.tick = self.tick.wrapping_add(1);
        let ai_healthy = self.healthy;
        if !ai_healthy || !self.phase.ai_allowed() || !frame.ai_input_valid() {
            return Command {
                ay: 0.0,
                ax: 0.0,
                ai_healthy,
            };
        }
        Command {
            ay: self.demand(0x9e37, frame),
            ax: self.demand(0xbf58, frame),
            ai_healthy,
        }
    }

    fn reset(&mut self, seed: u64) {
        self.seed = seed;
        self.tick = 0;
        self.phase = Phase::Pad;
    }

    fn on_phase(&mut self, phase: Phase) {
        self.phase = phase;
    }
}

/// Host-testable seam over the firmware link decision. A bridge crate implements
/// this for `potok_m_firmware::control::LinkState`: build the reply from
/// `frame` and `proposed`, drive `on_reply`/`command` (or `on_timeout` when no
/// reply exists), and return the command that survives the gates, in sim units.
/// A rejected, stale or unhealthy reply must produce `Command::ZERO`.
pub trait HostControl {
    fn decide(&mut self, frame: &Frame, proposed: Command) -> Command;
    /// Mirror `LinkState::reset`: clear misses, sequence and the AI latch.
    fn reset_link(&mut self);
}

/// A [`FirmwareAdapter`] whose decision passes through a [`HostControl`] bridge.
#[derive(Debug)]
pub struct HostAdapter<C: HostControl, P: FirmwareAdapter> {
    control: C,
    producer: P,
    seed: u64,
}

impl<C: HostControl, P: FirmwareAdapter> HostAdapter<C, P> {
    pub fn new(control: C, producer: P, seed: u64) -> Self {
        let mut producer = producer;
        producer.reset(seed);
        Self {
            control,
            producer,
            seed,
        }
    }

    pub fn control(&self) -> &C {
        &self.control
    }

    pub fn control_mut(&mut self) -> &mut C {
        &mut self.control
    }

    pub fn producer(&self) -> &P {
        &self.producer
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }
}

impl<C: HostControl, P: FirmwareAdapter> FirmwareAdapter for HostAdapter<C, P> {
    fn step(&mut self, frame: &Frame) -> Command {
        let proposed = self.producer.step(frame);
        self.control.decide(frame, proposed)
    }

    fn reset(&mut self, seed: u64) {
        self.seed = seed;
        self.producer.reset(seed);
        self.control.reset_link();
    }

    fn on_phase(&mut self, phase: Phase) {
        self.producer.on_phase(phase);
    }
}
