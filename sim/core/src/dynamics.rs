//! 6-DoF rigid-body dynamics with variable mass and RK4 integration.
//!
//! The RK4 state is stored struct-of-arrays (`State { pos, vel, quat, omega,
//! mass }`): each component group is its own contiguous fixed-size array, so
//! the integrator never touches a flat 14-element index space. Body +Z is the
//! rocket axis; world +Z is up. Inertia is diagonal and scales with mass.

use crate::{
    atmosphere::Atmosphere,
    config::{AeroConfig, Config},
    math::{
        add, clamp, cross, dot, lerp_table, mat_t_vec, mat_vec, norm, quat_normalize, quat_rate,
        quat_to_matrix, scale, sub, Quat, Vec3,
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Pad,
    Boost,
    Coast,
    Apogee,
    Descent,
    Landed,
}

impl Phase {
    pub fn code(self) -> u8 {
        match self {
            Phase::Pad => 0,
            Phase::Boost => 1,
            Phase::Coast => 2,
            Phase::Apogee => 3,
            Phase::Descent => 4,
            Phase::Landed => 5,
        }
    }
    /// AI corrections are forbidden on the active (powered) segment.
    pub fn ai_allowed(self) -> bool {
        !matches!(self, Phase::Pad | Phase::Boost)
    }

    /// Numeric wire phase from `protocol/uart.yaml` (0..=4). Pad and Boost
    /// both map to 0 (BURN/Idle), matching the integration contract.
    pub fn protocol_code(self) -> u8 {
        match self {
            Phase::Pad | Phase::Boost => 0,
            Phase::Coast => 1,
            Phase::Apogee => 2,
            Phase::Descent => 3,
            Phase::Landed => 4,
        }
    }

    /// Inverse of `protocol_code`; 0 is treated as Boost (powered flight).
    pub fn from_protocol_code(v: u8) -> Option<Self> {
        Some(match v {
            0 => Phase::Boost,
            1 => Phase::Coast,
            2 => Phase::Apogee,
            3 => Phase::Descent,
            4 => Phase::Landed,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Vehicle {
    pub dry_mass: f64,
    pub thrust_scale: f64,
    /// Unit vector of the thrust axis in body frame; normalized once here.
    pub thrust_dir: Vec3,
    pub aero: AeroConfig,
    pub cd_table: Vec<(f64, f64)>,
    pub thrust_table: Vec<(f64, f64, f64)>,
}

impl Vehicle {
    pub fn from_config(cfg: &Config, mass: f64, thrust_scale: f64, gimbal_deg: [f64; 2]) -> Self {
        let raw = [gimbal_deg[0].to_radians(), gimbal_deg[1].to_radians(), 1.0];
        let n = norm(raw);
        let thrust_dir = if n < 1e-9 {
            [0.0, 0.0, 1.0]
        } else {
            scale(raw, 1.0 / n)
        };
        Self {
            dry_mass: mass * 0.65,
            thrust_scale,
            thrust_dir,
            aero: cfg.aero.clone(),
            cd_table: cfg.aero.cd_mach.clone(),
            thrust_table: cfg.aero.thrust_profile.clone(),
        }
    }

    pub fn burn_end(&self) -> f64 {
        self.thrust_table
            .iter()
            .rev()
            .find(|(_, t, _)| *t > 0.0)
            .map(|(t, _, _)| *t)
            .unwrap_or(0.0)
    }

    pub fn max_thrust(&self) -> f64 {
        self.thrust_table
            .iter()
            .map(|p| p.1)
            .fold(0.0_f64, f64::max)
            * self.thrust_scale
    }

    /// Linear interpolation of (time, thrust, mass-flow) knots.
    #[inline]
    pub fn thrust_at(&self, t: f64) -> f64 {
        let mut thrust = 0.0;
        for i in 1..self.thrust_table.len() {
            let (t0, f0, _) = self.thrust_table[i - 1];
            let (t1, f1, _) = self.thrust_table[i];
            if t <= t1 {
                let u = clamp((t - t0) / (t1 - t0).max(1e-9), 0.0, 1.0);
                thrust = f0 + u * (f1 - f0);
                break;
            }
        }
        thrust * self.thrust_scale
    }

    #[inline]
    fn mdot_at(&self, t: f64) -> f64 {
        let mut mdot = 0.0;
        for i in 1..self.thrust_table.len() {
            let (t0, _, m0) = self.thrust_table[i - 1];
            let (t1, _, m1) = self.thrust_table[i];
            if t <= t1 {
                let u = clamp((t - t0) / (t1 - t0).max(1e-9), 0.0, 1.0);
                mdot = m0 + u * (m1 - m0);
                break;
            }
        }
        mdot * self.thrust_scale
    }
}

#[derive(Debug, Clone, Copy)]
pub struct State {
    pub pos: [f64; 3],
    pub vel: [f64; 3],
    pub quat: [f64; 4],
    pub omega: [f64; 3],
    pub mass: [f64; 1],
}

impl State {
    pub fn zeros() -> Self {
        Self {
            pos: [0.0; 3],
            vel: [0.0; 3],
            quat: [0.0; 4],
            omega: [0.0; 3],
            mass: [0.0; 1],
        }
    }

    pub fn new(mass: f64, pitch: f64, yaw: f64) -> Self {
        let cy = (yaw * 0.5).cos();
        let sy = (yaw * 0.5).sin();
        let cp = (pitch * 0.5).cos();
        let sp = (pitch * 0.5).sin();
        let q = quat_normalize([cy * cp, -sy * sp, sy * cp, cy * sp]);
        Self {
            pos: [0.0; 3],
            vel: [0.0; 3],
            quat: q,
            omega: [0.0; 3],
            mass: [mass],
        }
    }

    #[inline]
    pub fn pos(&self) -> Vec3 {
        self.pos
    }
    #[inline]
    pub fn vel(&self) -> Vec3 {
        self.vel
    }
    #[inline]
    pub fn quat(&self) -> Quat {
        self.quat
    }
    #[inline]
    pub fn omega(&self) -> Vec3 {
        self.omega
    }
    #[inline]
    pub fn mass(&self) -> f64 {
        self.mass[0]
    }

    pub fn is_finite(&self) -> bool {
        self.pos
            .iter()
            .chain(self.vel.iter())
            .chain(self.quat.iter())
            .chain(self.omega.iter())
            .chain(self.mass.iter())
            .all(|x| x.is_finite())
    }
}

/// Right-hand side. `wind` is frozen for the four RK4 stages of a step.
pub fn derivative(t: f64, s: &State, wind: Vec3, atm: &Atmosphere, v: &Vehicle) -> State {
    let mut d = State::zeros();
    let q = quat_normalize(s.quat);
    let r = quat_to_matrix(q);
    let air = atm.at(s.pos[2]);
    let v_rel = sub(s.vel, wind);
    let speed = norm(v_rel);
    let vb = mat_t_vec(r, v_rel);
    let mach = speed / air.speed_of_sound.max(1.0);

    let cd_base = v.cd_table.first().map(|p| p.1).unwrap_or(v.aero.cd0);
    let cd = (v.aero.cd0 - cd_base) + lerp_table(&v.cd_table, mach);
    let qdyn = 0.5 * air.density * speed * speed;
    let vhat = if speed > 1e-6 {
        scale(vb, 1.0 / speed)
    } else {
        [0.0, 0.0, 0.0]
    };
    let drag = scale(vhat, -qdyn * v.aero.area_m2 * cd);
    // Bounded small-angle proxy: alpha ~ lateral speed / total speed.
    let az = speed.max(1.0);
    // Normal force pushes the nose toward the relative wind. With the CP
    // ahead of the CG (positive static_margin) this is restoring.
    let normal = [
        qdyn * v.aero.area_m2 * v.aero.cna * (vb[0] / az),
        qdyn * v.aero.area_m2 * v.aero.cna * (vb[1] / az),
        0.0,
    ];
    let thrust = scale(v.thrust_dir, v.thrust_at(t));
    let body_force = add(add(drag, normal), thrust);

    let w = s.omega;
    let moment_aero = cross([0.0, 0.0, v.aero.static_margin_m], normal);
    let r_body = 0.15_f64;
    let damp_scale = qdyn * v.aero.area_m2 * r_body * 0.01;
    let moment = [
        moment_aero[0] - v.aero.pitch_damping * damp_scale * w[0],
        moment_aero[1] - v.aero.pitch_damping * damp_scale * w[1],
        -v.aero.roll_damping * damp_scale * w[2],
    ];

    let mass = s.mass[0].max(v.dry_mass);
    // Launch-rail constraint: before liftoff the rail carries the weight, so
    // the inertial acceleration is exactly zero and the accelerometer reads 1 g.
    let thrust_world_z = mat_vec(r, thrust)[2];
    let on_pad =
        s.pos[2] <= 1e-9 && s.vel[2] <= 0.0 && t < v.burn_end() && thrust_world_z < mass * 9.80665;
    let i_xx = 0.1 * mass * r_body * r_body;
    let i_zz = 0.02 * mass * r_body * r_body;
    let iw = [i_xx * w[0], i_xx * w[1], i_zz * w[2]];
    let gyro_term = cross(w, iw);
    let wdot = [
        (moment[0] - gyro_term[0]) / i_xx,
        (moment[1] - gyro_term[1]) / i_xx,
        (moment[2] - gyro_term[2]) / i_zz,
    ];

    let gravity = [0.0, 0.0, -9.80665];
    let force_world = add(scale(gravity, mass), mat_vec(r, body_force));
    // SoA convention: d.pos is the velocity, d.vel is the acceleration.
    d.pos = s.vel;
    d.vel = scale(force_world, 1.0 / mass);
    d.quat = quat_rate(q, w);
    d.omega = wdot;
    d.mass[0] = if mass <= v.dry_mass {
        0.0
    } else {
        -v.mdot_at(t)
    };
    if on_pad {
        d.pos = [0.0; 3];
        d.vel = [0.0; 3];
        d.omega = [0.0; 3];
    }
    d
}

#[inline]
fn state_add(s: &State, d: &State, h: f64) -> State {
    let mut o = State::zeros();
    for i in 0..3 {
        o.pos[i] = s.pos[i] + h * d.pos[i];
        o.vel[i] = s.vel[i] + h * d.vel[i];
        o.omega[i] = s.omega[i] + h * d.omega[i];
    }
    for i in 0..4 {
        o.quat[i] = s.quat[i] + h * d.quat[i];
    }
    o.mass[0] = s.mass[0] + h * d.mass[0];
    o
}

/// One classical RK4 step; wind is constant across the four sub-stages.
pub fn rk4(t: f64, s: &State, dt: f64, wind: Vec3, atm: &Atmosphere, v: &Vehicle) -> State {
    let k1 = derivative(t, s, wind, atm, v);
    let k2 = derivative(t + 0.5 * dt, &state_add(s, &k1, 0.5 * dt), wind, atm, v);
    let k3 = derivative(t + 0.5 * dt, &state_add(s, &k2, 0.5 * dt), wind, atm, v);
    let k4 = derivative(t + dt, &state_add(s, &k3, dt), wind, atm, v);
    let h6 = dt / 6.0;
    let mut o = State::zeros();
    for i in 0..3 {
        o.pos[i] = s.pos[i] + h6 * (k1.pos[i] + 2.0 * k2.pos[i] + 2.0 * k3.pos[i] + k4.pos[i]);
        o.vel[i] = s.vel[i] + h6 * (k1.vel[i] + 2.0 * k2.vel[i] + 2.0 * k3.vel[i] + k4.vel[i]);
        o.omega[i] =
            s.omega[i] + h6 * (k1.omega[i] + 2.0 * k2.omega[i] + 2.0 * k3.omega[i] + k4.omega[i]);
    }
    for i in 0..4 {
        o.quat[i] =
            s.quat[i] + h6 * (k1.quat[i] + 2.0 * k2.quat[i] + 2.0 * k3.quat[i] + k4.quat[i]);
    }
    o.mass[0] = (s.mass[0] + h6 * (k1.mass[0] + 2.0 * k2.mass[0] + 2.0 * k3.mass[0] + k4.mass[0]))
        .max(v.dry_mass);
    o.quat = quat_normalize(o.quat);
    if o.pos[2] <= 0.0 {
        o.pos[2] = 0.0;
        o.vel = [0.0; 3];
        o.omega = [0.0; 3];
    }
    o
}

pub fn phase_at(t: f64, s: &State, v: &Vehicle) -> Phase {
    if s.pos[2] <= 0.0 && t > 0.5 {
        return Phase::Landed;
    }
    if t <= 0.0 {
        return Phase::Pad;
    }
    if t <= v.burn_end() {
        return Phase::Boost;
    }
    if s.vel[2] < 0.0 {
        if s.pos[2] < 5.0 {
            Phase::Landed
        } else {
            Phase::Descent
        }
    } else if dot(s.vel, s.vel) < 4.0 && s.pos[2] > 10.0 {
        Phase::Apogee
    } else {
        Phase::Coast
    }
}
