//! Reference optimizer: full-state correction target for distillation.
//!
//! Approximation: an infinite-horizon LQR-style feedback on (lateral
//! position, lateral velocity, attitude, body rates) with hand-tuned gains
//! instead of a solved Riccati equation or an online MPC. It sees the true
//! state and true wind, so it is an upper bound on what the sensor-only model
//! can achieve. The two outputs are small pitch/roll set-point corrections
//! `g_theta`, clamped to `max_correction` rad. During the powered segment the
//! corrections are zero regardless of the state (AI interlock). The exported
//! command target is `g_accel = [ay, ax]`: the clamped world-frame lateral
//! acceleration demand, in the `[Y, X]` field order used by the dataset, from
//! which `fin_accel_command` derives the fin command `[ay, -ax]`.

/// Clamp on both reference-correction axes, radians. Shared with the
/// dataset quality gate so a target can never silently exceed it.
pub const MAX_CORRECTION: f64 = 0.12;

use crate::{
    config::Config,
    control::MAX_COMMAND_MPS2,
    dynamics::{Phase, State, Vehicle},
    math::{clamp, quat_to_matrix, Vec3},
    sensor::Frame,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Target {
    /// [roll, pitch] correction, rad. Diagnostic/legacy target only: the fin
    /// actuator consumes `g_accel`, not an angle.
    pub g_theta: [f32; 2],
    pub confidence: f32,
    pub ai_healthy: bool,
    /// Clamped world-frame desired lateral acceleration `[ay, ax]` in m/s^2.
    /// The fin command on the same tick is `[g_accel[0], -g_accel[1]]`
    /// (see `fin_accel_command`); zero while the AI interlock is active.
    pub g_accel: [f32; 2],
}

pub struct ReferenceOptimizer {
    kp_pos: f64,
    kd_vel: f64,
    kp_att: f64,
    kd_rate: f64,
    max_correction: f64,
}

impl Default for ReferenceOptimizer {
    fn default() -> Self {
        Self {
            kp_pos: 0.04,
            kd_vel: 0.55,
            kp_att: 1.8,
            kd_rate: 0.35,
            max_correction: MAX_CORRECTION,
        }
    }
}

impl ReferenceOptimizer {
    pub fn new(_cfg: &Config) -> Self {
        Self::default()
    }

    /// Desired world-frame lateral acceleration `(ax, ay)` in m/s^2 before the
    /// attitude/gimbal conversion. Shared by `solve` (which turns it into a
    /// `g_theta` target) and by `fin_accel_command` (the physical fin demand).
    fn lateral_accel_demand(
        &self,
        s: &State,
        wind: Vec3,
        phase: Phase,
        frame: &Frame,
    ) -> Option<[f64; 2]> {
        let pos = s.pos();
        let vel = s.vel();
        let v_rel = [vel[0] - wind[0], vel[1] - wind[1], vel[2] - wind[2]];
        let healthy = frame.ai_input_valid()
            && pos
                .iter()
                .chain(vel.iter())
                .chain(v_rel.iter())
                .all(|x| x.is_finite());
        if !healthy || !phase.ai_allowed() {
            return None;
        }
        Some([
            -self.kd_vel * vel[0] - self.kp_pos * pos[0],
            -self.kd_vel * vel[1] - self.kp_pos * pos[1],
        ])
    }

    /// Clamped world-frame desired lateral acceleration `[ax, ay]` in m/s^2,
    /// or `[0, 0]` when the AI interlock is active or the state is unhealthy.
    /// This is the authoritative training target, exported to the dataset as
    /// `[g_accel_y, g_accel_x] = [ay, ax]` (Y first to match the fin seam's
    /// roll-then-pitch order). It shares the raw LQR demand with `solve`.
    pub fn world_accel_command(
        &self,
        s: &State,
        wind: Vec3,
        phase: Phase,
        frame: &Frame,
    ) -> [f64; 2] {
        match self.lateral_accel_demand(s, wind, phase, frame) {
            Some([ax, ay]) => [
                clamp(ax, -MAX_COMMAND_MPS2, MAX_COMMAND_MPS2),
                clamp(ay, -MAX_COMMAND_MPS2, MAX_COMMAND_MPS2),
            ],
            None => [0.0, 0.0],
        }
    }

    /// Physical fin command in the control seam's `[roll, pitch]` order, in
    /// m/s^2 of desired lateral acceleration (NOT radians or fin degrees).
    /// Exactly `[g_accel[0], -g_accel[1]]` of the exported target pair.
    ///
    /// Command convention, verified against the flown trajectory: a positive
    /// `[roll, pitch]` value rotates the nose so the lateral aerodynamic
    /// response is **opposite** to the naive right-hand-rule body-axis reading
    /// (the sim's normal force is built from the relative wind in body frame).
    /// The seam's first (roll) axis carries the +Y demand and the second
    /// (pitch) axis its negative; the interlock/health gate matches `solve`.
    ///
    /// The raw LQR demand can exceed the actuator's acceleration envelope late
    /// in the flight; the controller saturates at `MAX_COMMAND_MPS2` itself.
    /// This is the controller's own saturation, not a silent seam fallback.
    pub fn fin_accel_command(
        &self,
        s: &State,
        wind: Vec3,
        _v: &Vehicle,
        phase: Phase,
        frame: &Frame,
    ) -> [f64; 2] {
        let [ax, ay] = self.world_accel_command(s, wind, phase, frame);
        [ay, -ax]
    }

    pub fn solve(&self, s: &State, wind: Vec3, v: &Vehicle, phase: Phase, frame: &Frame) -> Target {
        let [ax_des, ay_des] = match self.lateral_accel_demand(s, wind, phase, frame) {
            Some(g) => g,
            None => {
                return Target {
                    g_theta: [0.0, 0.0],
                    confidence: 0.0,
                    ai_healthy: false,
                    g_accel: [0.0, 0.0],
                }
            }
        };
        let axial = ((v.max_thrust() / s.mass().max(v.dry_mass)) + 9.80665).max(2.0);
        let pitch_des = (ax_des / axial).atan();
        let roll_des = -(ay_des / axial).atan();

        let [ax, ay] = self.world_accel_command(s, wind, phase, frame);
        let r = quat_to_matrix(s.quat());
        let pitch = r[0][2].atan2(r[2][2]);
        let roll = (-r[1][2]).atan2(r[2][2]);
        let w = s.omega();
        let g_roll = self.kp_att * wrap(roll_des - roll) - self.kd_rate * w[0];
        let g_pitch = self.kp_att * wrap(pitch_des - pitch) - self.kd_rate * w[1];
        let mag = g_roll.hypot(g_pitch);
        let confidence = clamp(1.0 - mag / (2.0 * self.max_correction), 0.0, 1.0) as f32;
        Target {
            g_theta: [
                clamp(g_roll, -self.max_correction, self.max_correction) as f32,
                clamp(g_pitch, -self.max_correction, self.max_correction) as f32,
            ],
            confidence,
            ai_healthy: true,
            // Dataset field order is [g_accel_y, g_accel_x] = [ay, ax].
            g_accel: [ay as f32, ax as f32],
        }
    }
}

fn wrap(a: f64) -> f64 {
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut x = a % two_pi;
    if x > std::f64::consts::PI {
        x -= two_pi;
    }
    if x < -std::f64::consts::PI {
        x += two_pi;
    }
    x
}
