//! Speculative dispatch: a stateful student on every frame, the windowed
//! teacher by exception (low confidence / invalid sensors, or every 20 frames).
//!
//! Rules from SPEC_STUDENT.md §"Правила dispatch":
//! 1. student every frame; 2. apply it when the CALIBRATED confidence meets
//! the threshold from `calibration.json` (0.7 if no file) and sensors are
//! valid; 3. otherwise call the teacher on the last 50 frames and apply that;
//! 4. every 20 frames a forced teacher check compares `|g_t - g_s|` and, above
//! `3*g_max`, logs an event and re-warms the student from the same window;
//! 5. BURN (phase 0) outputs zero on both paths; 6. bad CRC or sequence gaps
//! bypass both models. A cycle makes **at most one** teacher call — the
//! low-confidence and forced conditions share the same single call.

use crate::frame::{
    ReplyFrame, RequestFrame, CORRECTION_SCALE, FLAG_FAULT, FLAG_IMU_OK, PHASE_BURN,
};
use crate::runtime::{PhasePolicy, Runtime};
use crate::student_runtime::{
    StudentModel, StudentRuntime, COND_FEATURES, PHYSICS_FEATURES, SMART_FRAME_FEATURES, WINDOW,
};
use std::collections::VecDeque;
use std::fmt;
use std::fs;
use std::path::Path;

pub const DEFAULT_CONFIDENCE_THRESHOLD: f32 = crate::calibration::FALLBACK_THRESHOLD;
pub const DEFAULT_FORCED_CHECK_EVERY: u32 = 20;
pub const DEFAULT_DIVERGENCE_FACTOR: f32 = 3.0;
const MAX_EVENTS: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchSource {
    Student,
    TeacherLowConfidence,
    TeacherForced,
    TeacherBoth,
    Zero,
}

impl fmt::Display for DispatchSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Student => "student",
            Self::TeacherLowConfidence => "teacher/low-confidence",
            Self::TeacherForced => "teacher/forced-check",
            Self::TeacherBoth => "teacher/low-confidence+forced",
            Self::Zero => "zero",
        };
        write!(f, "{s}")
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DivergenceEvent {
    pub seq: u16,
    pub frame: u64,
    pub g_student: [f32; 2],
    pub g_teacher: [f32; 2],
    pub diff: f32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpecCounters {
    pub frames: u64,
    pub student_calls: u64,
    pub teacher_calls: u64,
    pub forced_checks: u64,
    pub low_confidence: u64,
    pub student_resets: u64,
    pub divergences: u64,
    pub phase_resets: u64,
    pub invalid_frames: u64,
    pub seq_errors: u64,
    pub muted: u64,
    pub zeroed: u64,
    pub nan_firewalls: u64,
    pub disables: u64,
}

/// Holds the existing windowed teacher and the stateful student side by side.
/// The teacher is called through `Runtime::infer_window`; sequence, phase and
/// miss accounting live here so one bad frame cannot double-count.
pub struct SpeculativeRuntime {
    teacher: Runtime,
    student: StudentRuntime,
    window: VecDeque<[f32; 8]>,
    flat: Vec<f32>,
    /// Last 50 student frames (`in_features` floats each: 10 legacy or 20
    /// smart); the teacher window holds only the 8 sensor features.
    student_flat: Vec<f32>,
    /// Vehicle-conditioning numbers appended to every smart frame.
    conditioning: [f32; COND_FEATURES],
    policy: PhasePolicy,
    last_seq: Option<u16>,
    last_phase: Option<u8>,
    misses: u32,
    disabled: bool,
    frame_no: u64,
    frames_since_check: u32,
    g_max: f32,
    pub confidence_threshold: f32,
    pub forced_check_every: u32,
    /// Multiplies `g_max` for the divergence trip point (default 3.0).
    /// Both heads emit `tanh*g_max`, so a literal 3*g_max is unreachable when
    /// they share a bound; lowering this is the test seam for the log+reset.
    pub divergence_factor: f32,
    pub counters: SpecCounters,
    events: Vec<DivergenceEvent>,
    pub last_source: DispatchSource,
    pub last_g: [f32; 2],
    pub last_confidence: f32,
}

impl SpeculativeRuntime {
    pub fn new(teacher: Runtime, student: StudentRuntime) -> Self {
        // The student owns the calibration map, so the dispatch threshold is
        // whatever the same held-out fit produced (0.7 with no file).
        let confidence_threshold = student.calibration().threshold();
        let g_max = teacher.max_correction().max(student.model().g_max);
        let student_features = student.model().in_features;
        let mut window = VecDeque::with_capacity(WINDOW);
        for _ in 0..WINDOW {
            window.push_back([0.0; 8]);
        }
        Self {
            teacher,
            student,
            window,
            flat: vec![0.0; WINDOW * 8],
            student_flat: vec![0.0; WINDOW * student_features],
            conditioning: [0.0; COND_FEATURES],
            policy: PhasePolicy::default(),
            last_seq: None,
            last_phase: None,
            misses: 0,
            disabled: false,
            frame_no: 0,
            frames_since_check: 0,
            g_max,
            confidence_threshold,
            forced_check_every: DEFAULT_FORCED_CHECK_EVERY,
            divergence_factor: DEFAULT_DIVERGENCE_FACTOR,
            counters: SpecCounters::default(),
            events: Vec::new(),
            last_source: DispatchSource::Zero,
            last_g: [0.0; 2],
            last_confidence: 0.0,
        }
    }

    /// Load both files, refusing to confuse the two model kinds.
    pub fn load(
        teacher_path: impl AsRef<Path>,
        student_path: impl AsRef<Path>,
    ) -> Result<Self, String> {
        let teacher_bytes =
            fs::read(teacher_path.as_ref()).map_err(|e| format!("cannot read teacher: {e}"))?;
        if crate::student_runtime::is_student_file(&teacher_bytes) {
            return Err("teacher path holds a model_kind=2 student file".to_owned());
        }
        let student_bytes =
            fs::read(student_path.as_ref()).map_err(|e| format!("cannot read student: {e}"))?;
        if !crate::student_runtime::is_student_file(&student_bytes) {
            return Err("student path is not a model_kind=2 file".to_owned());
        }
        let model = StudentModel::from_bytes(&student_bytes).map_err(|e| e.to_string())?;
        let teacher = Runtime::load(teacher_path)?;
        Ok(Self::new(teacher, StudentRuntime::new(model)))
    }

    /// Like [`Self::load`], but attach a `calibration.json`. The file selects
    /// both the confidence map used by the student and the dispatch threshold.
    pub fn load_with_calibration(
        teacher_path: impl AsRef<Path>,
        student_path: impl AsRef<Path>,
        calibration_path: impl AsRef<Path>,
    ) -> Result<Self, String> {
        let student_bytes =
            fs::read(student_path.as_ref()).map_err(|e| format!("cannot read student: {e}"))?;
        if !crate::student_runtime::is_student_file(&student_bytes) {
            return Err("student path is not a model_kind=2 file".to_owned());
        }
        let model = StudentModel::from_bytes(&student_bytes).map_err(|e| e.to_string())?;
        let student = StudentRuntime::with_calibration_file(model, calibration_path)
            .map_err(|e| e.to_string())?;
        let teacher = Runtime::load(teacher_path)?;
        Ok(Self::new(teacher, student))
    }

    pub fn with_policy(mut self, policy: PhasePolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Set the 6 conditioning values appended to each smart (20-input) frame.
    /// Ignored by a legacy 10-input student.
    pub fn set_conditioning(&mut self, conditioning: [f32; COND_FEATURES]) {
        self.conditioning = conditioning;
    }

    pub fn student(&self) -> &StudentRuntime {
        &self.student
    }

    pub fn events(&self) -> &[DivergenceEvent] {
        &self.events
    }

    pub fn window(&self) -> &[f32] {
        &self.flat
    }

    pub fn misses(&self) -> u32 {
        self.misses
    }

    pub fn is_disabled(&self) -> bool {
        self.disabled
    }

    pub fn g_max(&self) -> f32 {
        self.g_max
    }

    /// CRC failure, bad magic/version or read timeout: no model is called.
    pub fn on_invalid_frame(&mut self) {
        self.counters.invalid_frames += 1;
        self.misses += 1;
        self.last_source = DispatchSource::Zero;
        self.last_g = [0.0; 2];
        self.disable_if_threshold();
    }

    pub fn on_timeout(&mut self) {
        self.on_invalid_frame();
    }

    pub fn disable_now(&mut self) {
        if !self.disabled {
            self.disabled = true;
            self.counters.disables += 1;
        }
    }

    fn disable_if_threshold(&mut self) {
        if self.misses >= 3 && !self.disabled {
            self.disabled = true;
            self.counters.disables += 1;
        }
    }

    fn muted(&mut self, seq: u16, healthy: bool) -> ReplyFrame {
        self.counters.muted += 1;
        ReplyFrame::muted(seq, healthy)
    }

    /// Produce the reply for one already CRC-checked request. Sequence, phase
    /// and sensor flags are checked here; at most one teacher call follows.
    pub fn on_request(&mut self, req: &RequestFrame) -> ReplyFrame {
        if self.disabled {
            return self.muted(req.seq, false);
        }

        if let Some(prev) = self.last_seq {
            if req.seq != prev.wrapping_add(1) {
                self.counters.seq_errors += 1;
                self.counters.invalid_frames += 1;
                self.misses += 1;
                self.last_seq = Some(req.seq);
                self.disable_if_threshold();
                self.last_source = DispatchSource::Zero;
                self.last_g = [0.0; 2];
                return self.muted(req.seq, false);
            }
        }
        self.last_seq = Some(req.seq);
        self.misses = 0;

        // Rule 5: phase 0 is BURN in the canonical protocol (edge calls it
        // PHASE_IDLE); the motor-armed flag is the other burn indication.
        if req.phase == PHASE_BURN || self.policy.is_forbidden(req.phase, req.flags) {
            self.counters.zeroed += 1;
            self.last_source = DispatchSource::Zero;
            self.last_g = [0.0; 2];
            self.last_confidence = 0.0;
            return self.muted(req.seq, true);
        }

        // Acceptance test: re-warm on a phase change so old-phase hidden state
        // cannot leak into the new phase.
        if self.last_phase != Some(req.phase) {
            self.student.warmup(&self.student_flat);
            self.counters.phase_resets += 1;
            self.last_phase = Some(req.phase);
        }

        self.window.pop_front();
        self.window.push_back(req.features());
        for (i, frame) in self.window.iter().enumerate() {
            self.flat[i * 8..i * 8 + 8].copy_from_slice(frame);
        }
        // The student's frame carries the phase code and flags word too.
        let nf = self.student.model().in_features;
        self.student_flat.copy_within(nf..WINDOW * nf, 0);
        let last = &mut self.student_flat[(WINDOW - 1) * nf..WINDOW * nf];
        last[..8].copy_from_slice(&req.features());
        if nf == SMART_FRAME_FEATURES {
            // Smart frame: 14 physics features (phase/flags are not part of
            // this layout) followed by the 6 conditioning numbers.
            last[PHYSICS_FEATURES..SMART_FRAME_FEATURES].copy_from_slice(&self.conditioning);
        } else {
            if nf > crate::student_runtime::PHASE_INDEX {
                last[crate::student_runtime::PHASE_INDEX] = req.phase as f32;
            }
            if nf > crate::student_runtime::FLAGS_INDEX {
                last[crate::student_runtime::FLAGS_INDEX] = req.flags as f32;
            }
        }
        self.frame_no += 1;
        self.counters.frames += 1;
        let teacher_calls_before = self.counters.teacher_calls;

        // Rule 1: student runs on every valid frame.
        let [gp, gy, conf] = self
            .student
            .step(&self.student_flat[(WINDOW - 1) * nf..WINDOW * nf]);
        self.counters.student_calls += 1;
        self.last_confidence = conf;
        let g_s = [gp, gy];

        let sensors_valid = req.flags & FLAG_IMU_OK != 0 && req.flags & FLAG_FAULT == 0;
        let low_conf = conf < self.confidence_threshold || !sensors_valid;
        if low_conf {
            self.counters.low_confidence += 1;
        }

        self.frames_since_check += 1;
        let forced = self.frames_since_check >= self.forced_check_every.max(1);

        // Rules 3 + 4 share ONE teacher call per cycle.
        let mut g = g_s;
        let mut source = DispatchSource::Student;
        let mut applied_conf = conf;
        if low_conf || forced {
            let [tp_raw, ty_raw, tc] = self.teacher.infer_window(&self.flat);
            // The legacy POTM engine emits raw head logits here; the v2 head
            // is already tanh-bounded. Clamp both to the same system bound
            // before applying or comparing, or a raw logit trips the trip
            // point and the command can leave its limit.
            let g_t = [
                tp_raw.clamp(-self.g_max, self.g_max),
                ty_raw.clamp(-self.g_max, self.g_max),
            ];
            self.counters.teacher_calls += 1;
            let [tp, ty] = g_t;
            if forced {
                self.counters.forced_checks += 1;
            }
            let diff = (tp - gp).abs().max((ty - gy).abs());
            if diff > self.divergence_factor * self.g_max {
                self.counters.divergences += 1;
                self.counters.student_resets += 1;
                if self.events.len() == MAX_EVENTS {
                    self.events.remove(0);
                }
                self.events.push(DivergenceEvent {
                    seq: req.seq,
                    frame: self.frame_no,
                    g_student: g_s,
                    g_teacher: g_t,
                    diff,
                });
                self.student.warmup(&self.student_flat);
            }
            g = g_t;
            applied_conf = tc;
            source = match (low_conf, forced) {
                (true, true) => DispatchSource::TeacherBoth,
                (true, false) => DispatchSource::TeacherLowConfidence,
                (false, true) => DispatchSource::TeacherForced,
                (false, false) => unreachable!(),
            };
            // Rule 3 also resets the periodic-check counter.
            self.frames_since_check = 0;
        }
        debug_assert!(
            self.counters.teacher_calls - teacher_calls_before <= 1,
            "one cycle may call the teacher at most once"
        );

        // A divergence is logged and clamped; a NaN takes the zero path.
        if !g[0].is_finite() || !g[1].is_finite() {
            self.counters.nan_firewalls += 1;
            g = [0.0; 2];
            source = DispatchSource::Zero;
        }
        g[0] = g[0].clamp(-self.g_max, self.g_max);
        g[1] = g[1].clamp(-self.g_max, self.g_max);
        self.last_g = g;
        self.last_source = source;
        self.last_confidence = applied_conf;

        ReplyFrame {
            seq: req.seq,
            g_pitch: quantize(g[0], self.g_max),
            g_yaw: quantize(g[1], self.g_max),
            confidence: (applied_conf.clamp(0.0, 1.0) * 255.0).round() as u8,
            ai_healthy: 1,
            version: 1,
        }
    }
}

fn quantize(v: f32, bound: f32) -> i16 {
    (v.clamp(-bound, bound) / CORRECTION_SCALE)
        .round()
        .clamp(i16::MIN as f32, i16::MAX as f32) as i16
}
