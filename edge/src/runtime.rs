//! Frame-to-frame safety state: sliding window, phase gate, sequence and
//! miss accounting. No I/O and no allocation happens per frame.

use crate::frame::{
    ReplyFrame, RequestFrame, FLAG_MOTOR_ARMED, PHASE_ARMED, PHASE_FAULT, PHASE_IDLE,
    CORRECTION_SCALE,
};
use crate::format_v2::ModelV2;
use crate::model::{Model, Scratch};
use std::collections::VecDeque;
use std::fs;
use std::path::Path;

pub const MAX_CORRECTION: f32 = 0.1; // radians; |g| <= Gmax (concept §2)
pub const MAX_CONSECUTIVE_MISSES: u32 = 3;

/// Policy for when the assist output is allowed.
///
/// The concept requires silence while the motor works. `potok-uart` v1 has no
/// dedicated BURN phase, so the gate combines the phase byte with the
/// `MOTOR_ARMED` flag: `Flight` is allowed only when the motor is not armed
/// (coast / apogee / descent all report as `Flight` in this protocol).
#[derive(Debug, Clone)]
pub struct PhasePolicy {
    pub forbidden_phases: [bool; 8],
    pub forbid_motor_armed: bool,
}

impl Default for PhasePolicy {
    fn default() -> Self {
        Self::conservative()
    }
}

impl PhasePolicy {
    pub fn conservative() -> Self {
        let mut forbidden_phases = [false; 8];
        forbidden_phases[PHASE_IDLE as usize] = true;
        forbidden_phases[PHASE_ARMED as usize] = true;
        forbidden_phases[PHASE_FAULT as usize] = true;
        Self { forbidden_phases, forbid_motor_armed: true }
    }

    pub fn is_forbidden(&self, phase: u8, flags: u8) -> bool {
        let p = self.forbidden_phases.get(phase as usize).copied().unwrap_or(true);
        p || (self.forbid_motor_armed && flags & FLAG_MOTOR_ARMED != 0)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Counters {
    pub frames_ok: u64,
    pub frames_muted: u64,
    pub crc_errors: u64,
    pub seq_errors: u64,
    pub timeouts: u64,
    pub disables: u64,
}

/// One of the two supported model formats, behind the same fixed window API.
enum Engine {
    PotM { model: Model, scratch: Scratch },
    V2(ModelV2),
}

impl Engine {
    fn window_len(&self) -> usize {
        match self {
            Self::PotM { model, .. } => model.window,
            Self::V2(m) => m.window,
        }
    }

    fn max_correction(&self) -> f32 {
        match self {
            Self::PotM { .. } => MAX_CORRECTION,
            Self::V2(m) => m.g_max,
        }
    }

    fn forward(&mut self, window: &[f32]) -> [f32; 3] {
        match self {
            Self::PotM { model, scratch } => {
                let (p, y, c) = model.forward(window, scratch);
                [p, y, c]
            }
            Self::V2(m) => m.forward(window),
        }
    }

    fn memory_bytes(&self) -> usize {
        match self {
            Self::PotM { model, scratch } => model.memory_bytes() + scratch.memory_bytes(),
            Self::V2(m) => m.memory_bytes(),
        }
    }

    fn potm(&self) -> Option<&Model> {
        match self {
            Self::PotM { model, .. } => Some(model),
            Self::V2(_) => None,
        }
    }
}

/// Owns the model engine, the preallocated window and the safety state.
pub struct Runtime {
    engine: Engine,
    window: VecDeque<[f32; 8]>,
    flat: Vec<f32>,
    policy: PhasePolicy,
    last_seq: Option<u16>,
    misses: u32,
    disabled: bool,
    last_raw: [f32; 3],
    pub counters: Counters,
}

impl Runtime {
    pub fn new(model: Model) -> Self {
        Self::with_policy(model, PhasePolicy::default())
    }

    pub fn with_policy(model: Model, policy: PhasePolicy) -> Self {
        Self::from_engine(
            Engine::PotM { scratch: Scratch::new(&model), model },
            policy,
        )
    }

    pub fn from_v2(model: ModelV2) -> Self {
        Self::with_policy_v2(model, PhasePolicy::default())
    }

    pub fn with_policy_v2(model: ModelV2, policy: PhasePolicy) -> Self {
        Self::from_engine(Engine::V2(model), policy)
    }

    /// Load either format by its magic bytes: `POTOKM1\0` selects the v2
    /// reader, everything else goes to the legacy POTM reader.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let bytes = fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        if bytes.starts_with(&crate::format_v2::MAGIC) {
            ModelV2::from_bytes(&bytes)
                .map(Self::from_v2)
                .map_err(|e| e.to_string())
        } else {
            Model::from_bytes(&bytes)
                .map(Self::new)
                .map_err(|e| e.to_string())
        }
    }

    fn from_engine(engine: Engine, policy: PhasePolicy) -> Self {
        let window_len = engine.window_len();
        let mut window = VecDeque::with_capacity(window_len);
        for _ in 0..window_len {
            window.push_back([0.0; 8]);
        }
        Self {
            engine,
            window,
            flat: vec![0.0; window_len * 8],
            policy,
            last_seq: None,
            misses: 0,
            disabled: false,
            last_raw: [0.0; 3],
            counters: Counters::default(),
        }
    }

    /// The legacy model, when this runtime holds one (used by the naive
    /// baseline benchmark). Returns `None` for v2 models.
    pub fn potm_model(&self) -> Option<&Model> {
        self.engine.potm()
    }

    pub fn is_v2(&self) -> bool {
        matches!(self.engine, Engine::V2(_))
    }

    pub fn is_disabled(&self) -> bool {
        self.disabled
    }

    pub fn misses(&self) -> u32 {
        self.misses
    }

    /// Raw head output `[g_pitch, g_yaw, confidence]` of the last inference.
    pub fn last_raw(&self) -> [f32; 3] {
        self.last_raw
    }

    /// Current 50x8 sliding window, oldest first, flattened row-major.
    pub fn last_window(&self) -> &[f32] {
        &self.flat
    }

    /// Total bytes owned by the dequantized weights and the reusable buffers.
    pub fn memory_bytes(&self) -> usize {
        self.engine.memory_bytes()
            + self.flat.len() * std::mem::size_of::<f32>()
            + self.window.capacity() * 8 * std::mem::size_of::<f32>()
    }

    /// Call when a frame arrived but failed CRC / magic / version / read timeout.
    /// Three consecutive failures latch the assistant off until restart.
    pub fn on_invalid_frame(&mut self) {
        if self.disabled {
            return;
        }
        self.misses += 1;
        self.disable_if_threshold();
    }

    pub fn on_timeout(&mut self) {
        self.counters.timeouts += 1;
        self.on_invalid_frame();
    }

    /// A protocol version mismatch is fatal immediately (no three-strike
    /// allowance): firmware and edge disagree on the wire format.
    pub fn disable_now(&mut self) {
        if !self.disabled {
            self.disabled = true;
            self.counters.disables += 1;
        }
    }

    /// Produce the reply for one valid request. CRC was already checked by the
    /// codec; sequence and phase are checked here.
    pub fn on_request(&mut self, req: &RequestFrame) -> ReplyFrame {
        if self.disabled {
            return ReplyFrame::muted(req.seq, false);
        }

        if let Some(prev) = self.last_seq {
            let expected = prev.wrapping_add(1);
            if req.seq != expected {
                // Broken or duplicated packet: ignore it and resync on the new
                // sequence so one bad frame does not poison every later one.
                self.counters.seq_errors += 1;
                self.misses += 1;
                self.last_seq = Some(req.seq);
                self.disable_if_threshold();
                return ReplyFrame::muted(req.seq, !self.disabled);
            }
        }
        self.last_seq = Some(req.seq);

        // Any fully valid frame clears the consecutive-miss counter.
        self.misses = 0;

        if self.policy.is_forbidden(req.phase, req.flags) {
            self.counters.frames_muted += 1;
            // Muted by policy, not a fault: stay healthy so the controller keeps
            // listening, but send exactly zero correction.
            return ReplyFrame::muted(req.seq, true);
        }

        let features = req.features();
        self.window.pop_front();
        self.window.push_back(features);
        for (i, frame) in self.window.iter().enumerate() {
            self.flat[i * 8..i * 8 + 8].copy_from_slice(frame);
        }
        let [pitch, yaw, conf] = self.engine.forward(&self.flat);
        self.last_raw = [pitch, yaw, conf];

        self.counters.frames_ok += 1;
        let bound = self.engine.max_correction();
        let clamp = |v: f32| -> i16 {
            let v = v.clamp(-bound, bound);
            (v / CORRECTION_SCALE).round().clamp(i16::MIN as f32, i16::MAX as f32) as i16
        };
        let confidence = conf.clamp(0.0, 1.0);
        ReplyFrame {
            seq: req.seq,
            g_pitch: clamp(pitch),
            g_yaw: clamp(yaw),
            confidence: (confidence * 255.0).round() as u8,
            ai_healthy: 1,
            version: 1,
        }
    }

    fn disable_if_threshold(&mut self) {
        if self.misses >= MAX_CONSECUTIVE_MISSES && !self.disabled {
            self.disabled = true;
            self.counters.disables += 1;
        }
    }
}
