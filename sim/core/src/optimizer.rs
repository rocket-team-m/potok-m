//! Reference optimizer: full-state correction target for distillation.
//!
//! Approximation: an infinite-horizon LQR-style feedback on (lateral
//! position, lateral velocity, attitude, body rates) with hand-tuned gains
//! instead of a solved Riccati equation or an online MPC. It sees the true
//! state and true wind, so it is an upper bound on what the sensor-only model
//! can achieve. The two outputs are small pitch/roll set-point corrections
//! `g_theta`, clamped to `max_correction` rad. During the powered segment the
//! corrections are zero regardless of the state (AI interlock).

use crate::{
    config::Config,
    dynamics::{Phase, State, Vehicle},
    math::{clamp, quat_to_matrix, Vec3},
    sensor::Frame,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Target {
    /// [roll, pitch] correction, rad.
    pub g_theta: [f32; 2],
    pub confidence: f32,
    pub ai_healthy: bool,
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
            max_correction: 0.12,
        }
    }
}

impl ReferenceOptimizer {
    pub fn new(_cfg: &Config) -> Self {
        Self::default()
    }

    pub fn solve(&self, s: &State, wind: Vec3, v: &Vehicle, phase: Phase, frame: &Frame) -> Target {
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
            return Target {
                g_theta: [0.0, 0.0],
                confidence: 0.0,
                ai_healthy: false,
            };
        }
        let axial = ((v.max_thrust() / s.mass().max(v.dry_mass)) + 9.80665).max(2.0);
        let ax_des = -self.kd_vel * vel[0] - self.kp_pos * pos[0];
        let ay_des = -self.kd_vel * vel[1] - self.kp_pos * pos[1];
        let pitch_des = (ax_des / axial).atan();
        let roll_des = -(ay_des / axial).atan();

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
