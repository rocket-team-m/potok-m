//! Six physics features derived from the RK4 state, the atmosphere and the
//! measured specific force. They are appended to the eight raw telemetry
//! channels to form the 14-value frame vector. No fitted constants: all
//! numbers come from the current state, the vehicle card, or `Atmosphere::at`.

use crate::{
    atmosphere::Air,
    dynamics::{State, Vehicle},
    math::{mat_t_vec, norm, quat_to_matrix, scale, sub},
};

/// Order is load-bearing: the last six entries of `sensor::FEATURE_NAMES`
/// must match this array (asserted in the physics test).
pub const PHYSICS_FEATURES: [&str; 6] = [
    "alpha_deg",
    "beta_deg",
    "q_dyn_kpa",
    "a_lat_resid",
    "drag_est",
    "static_margin_cal",
];

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhysicsFeatures {
    pub alpha_deg: f32,
    pub beta_deg: f32,
    pub q_dyn_kpa: f32,
    pub a_lat_resid: f32,
    pub drag_est: f32,
    pub static_margin_cal: f32,
}

impl PhysicsFeatures {
    pub fn as_array(self) -> [f32; 6] {
        [
            self.alpha_deg,
            self.beta_deg,
            self.q_dyn_kpa,
            self.a_lat_resid,
            self.drag_est,
            self.static_margin_cal,
        ]
    }

    /// `measured_accel` is the body-frame specific force from the sensor model;
    /// it is the RK4 acceleration plus gravity tilt, bias, lag and noise.
    pub fn from_state(
        s: &State,
        wind: [f64; 3],
        air: Air,
        v: &Vehicle,
        measured_accel: [f32; 3],
    ) -> Self {
        let r = quat_to_matrix(s.quat());
        let v_rel = sub(s.vel(), wind);
        let speed = norm(v_rel);
        let vb = mat_t_vec(r, v_rel);
        let qdyn = 0.5 * air.density * speed * speed;
        let mass = s.mass().max(v.dry_mass);
        let area = v.aero.area_m2;
        let az = speed.max(1.0);
        let cna = v.effective_cna();

        // SPEC_SMART.md writes atan2(w_body_z, v_body_x): w is the body Z
        // angular rate from the RK4 state. Kept literally; the classical
        // aerodynamic AoA would use -v_body_z instead.
        let alpha_deg = s.omega()[2].atan2(vb[0]).to_degrees();
        let beta_deg = vb[1].atan2(vb[0]).to_degrees();
        let q_dyn_kpa = qdyn / 1000.0;

        // Lateral specific force minus the base aero model's lateral estimate
        // (normal force + mach-table drag), in m/s^2.
        let normal = [
            qdyn * area * cna * (vb[0] / az),
            qdyn * area * cna * (vb[1] / az),
        ];
        let mach = speed / air.speed_of_sound.max(1.0);
        let cd = v.effective_cd(mach);
        let vhat = if speed > 1e-6 {
            scale(vb, 1.0 / speed)
        } else {
            [0.0; 3]
        };
        let drag_body = scale(vhat, -qdyn * area * cd);
        let a_lat_resid = (f64::from(measured_accel[0]) - (normal[0] + drag_body[0]) / mass)
            .hypot(f64::from(measured_accel[1]) - (normal[1] + drag_body[1]) / mass);

        // cd0 * q_dyn * area / mass, m/s^2.
        let drag_est = v.aero.cd0 * qdyn * area / mass;
        let diameter = 2.0 * (area / std::f64::consts::PI).sqrt();
        let static_margin_cal = v.aero.static_margin_m / diameter.max(1e-9);

        Self {
            alpha_deg: alpha_deg as f32,
            beta_deg: beta_deg as f32,
            q_dyn_kpa: q_dyn_kpa as f32,
            a_lat_resid: a_lat_resid as f32,
            drag_est: drag_est as f32,
            static_margin_cal: static_margin_cal as f32,
        }
    }
}
