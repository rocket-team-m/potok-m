//! 6-DoF rigid-body dynamics with variable mass and RK4 integration.
//!
//! The RK4 state is stored struct-of-arrays (`State { pos, vel, quat, omega,
//! mass }`): each component group is its own contiguous fixed-size array, so
//! the integrator never touches a flat 14-element index space. Body +Z is the
//! rocket axis; world +Z is up. Inertia is diagonal and scales with mass.

use std::cell::Cell;

use crate::{
    aero::{AeroCard, AeroCoeffs},
    atmosphere::Atmosphere,
    config::{AeroConfig, Config},
    error::SimError,
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

/// Placeholder fin control authority per radian of commanded deflection.
/// This is a new physical assumption for the control seam, not a tuned knob:
/// it is the same order as the other placeholder aero coefficients and is
/// applied symmetrically to both axes. It is only read when a command is
/// non-zero, so the legacy (zero-command) path is bit-for-bit unchanged.
pub const FIN_CONTROL_AUTHORITY: f64 = 0.02;

#[derive(Debug, Clone)]
pub struct Vehicle {
    pub dry_mass: f64,
    /// Dry mass jettisoned with the first stage.
    pub stage_sep_mass: f64,
    pub stage_sep_time: f64,
    pub thrust_scale: f64,
    /// Unit vector of the thrust axis in body frame; normalized once here.
    pub thrust_dir: Vec3,
    pub aero: AeroConfig,
    pub cd_table: Vec<(f64, f64)>,
    pub thrust_table: Vec<(f64, f64, f64)>,
    /// Set only in card mode. Coefficients are evaluated at every RK4
    /// sub-stage, not frozen: the legacy Mach table varies within a step.
    pub aero_card: Option<AeroCard>,
    /// Coefficients at the current tick, used by the exported physics
    /// features. The integrator itself queries the card per sub-stage.
    pub card_coeffs: AeroCoeffs,
    /// An out-of-grid query inside `derivative` is recorded here and returned
    /// as a `SimError::AeroOutOfGrid` by the episode loop. No silent clamp.
    pub card_fault: Cell<Option<CardFault>>,
    /// Per-step fin command `[roll, pitch]` in body-frame degrees. Set by the
    /// episode loop from the control closure; zero on the legacy path.
    pub control_deg: [f64; 2],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CardFault {
    pub axis: &'static str,
    pub value: f64,
    pub lo: f64,
    pub hi: f64,
}

fn fault_axis(axis: &str) -> &'static str {
    match axis {
        "mach" => "mach",
        "alpha_deg" => "alpha_deg",
        _ => "card",
    }
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
        // Dry mass follows from the propellant implied by the thrust table,
        // not from an arbitrary fraction of lift-off mass.
        let propellant = propellant_mass(&cfg.aero.thrust_profile) * thrust_scale;
        let dry_mass = (mass - propellant).max(mass * 0.5);
        Self {
            dry_mass,
            stage_sep_mass: cfg.aero.stage_sep_mass_kg,
            stage_sep_time: cfg.aero.stage_sep_time_s,
            thrust_scale,
            thrust_dir,
            aero: cfg.aero.clone(),
            cd_table: cfg.aero.cd_mach.clone(),
            thrust_table: cfg.aero.thrust_profile.clone(),
            aero_card: None,
            card_coeffs: AeroCoeffs {
                cd0: cfg.aero.cd0,
                cna: cfg.aero.cna,
                pitch_damping: cfg.aero.pitch_damping,
                cm_delta: 0.0,
            },
            card_fault: Cell::new(None),
            control_deg: [0.0; 2],
        }
    }

    /// Total axial drag coefficient. The placeholder branch is the original
    /// expression, unchanged bit-for-bit; the card branch reads the cached
    /// per-step coefficients already shifted by the episode's `cd0`.
    #[inline]
    pub fn effective_cd(&self, mach: f64) -> f64 {
        if self.aero_card.is_some() {
            self.card_coeffs.cd0
        } else {
            let cd_base = self.cd_table.first().map(|p| p.1).unwrap_or(self.aero.cd0);
            (self.aero.cd0 - cd_base) + lerp_table(&self.cd_table, mach)
        }
    }

    #[inline]
    pub fn effective_cna(&self) -> f64 {
        if self.aero_card.is_some() {
            self.card_coeffs.cna
        } else {
            self.aero.cna
        }
    }

    /// Damping multiplier before the `x 10` and the dynamic-pressure scaling
    /// in `derivative`.
    #[inline]
    pub fn effective_pitch_damping(&self) -> f64 {
        if self.aero_card.is_some() {
            self.card_coeffs.pitch_damping
        } else {
            self.aero.pitch_damping
        }
    }

    /// Gimbal control-moment coefficient; zero in placeholder mode.
    #[inline]
    pub fn effective_cm_delta(&self) -> f64 {
        if self.aero_card.is_some() {
            self.card_coeffs.cm_delta
        } else {
            0.0
        }
    }

    /// Propellant implied by the (time, thrust, mass-flow) table, kg.
    pub fn propellant_mass(&self) -> f64 {
        let mut total = 0.0;
        for w in self.thrust_table.windows(2) {
            total += w[0].2 * (w[1].0 - w[0].0);
        }
        total * self.thrust_scale
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

/// Integrate the mass-flow column of a thrust table over time.
fn propellant_mass(table: &[(f64, f64, f64)]) -> f64 {
    let mut total = 0.0;
    for w in table.windows(2) {
        total += w[0].2 * (w[1].0 - w[0].0);
    }
    total.max(0.0)
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

    // Card coefficients are evaluated at the sub-stage's own (Mach, alpha),
    // matching the legacy table's per-stage lookup. The placeholder branch
    // below is the original expression, unchanged bit-for-bit.
    let alpha_deg = s.omega()[2].atan2(vb[0]).to_degrees();
    let (cd, cna, pitch_k, cm_delta) = if let Some(card) = &v.aero_card {
        match card.coeffs(mach, alpha_deg, v.aero.cd0) {
            Ok(c) => (c.cd0, c.cna, c.pitch_damping, c.cm_delta),
            Err(SimError::AeroOutOfGrid {
                axis,
                value,
                lo,
                hi,
            }) => {
                v.card_fault.set(Some(CardFault {
                    axis: fault_axis(&axis),
                    value,
                    lo,
                    hi,
                }));
                (0.0, 0.0, 0.0, 0.0)
            }
            Err(_) => {
                v.card_fault.set(Some(CardFault {
                    axis: "card",
                    value: mach,
                    lo: f64::NAN,
                    hi: f64::NAN,
                }));
                (0.0, 0.0, 0.0, 0.0)
            }
        }
    } else {
        (
            v.effective_cd(mach),
            v.effective_cna(),
            v.effective_pitch_damping(),
            0.0,
        )
    };
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
        qdyn * v.aero.area_m2 * cna * (vb[0] / az),
        qdyn * v.aero.area_m2 * cna * (vb[1] / az),
        0.0,
    ];
    let thrust = scale(v.thrust_dir, v.thrust_at(t));
    let body_force = add(add(drag, normal), thrust);

    let w = s.omega;
    let moment_aero = cross([0.0, 0.0, v.aero.static_margin_m], normal);
    let r_body = 0.15_f64;
    let sep_done = if t >= v.stage_sep_time {
        v.stage_sep_mass
    } else {
        0.0
    };
    let mass_min = v.dry_mass - sep_done;
    let mass = s.mass[0].max(mass_min);
    let i_xx = 0.1 * mass * r_body * r_body;
    let i_zz = 0.02 * mass * r_body * r_body;
    // PLACEHOLDER pitch/roll damping (not from the real aero model). Scaled
    // with dynamic pressure and capped so the explicit RK4 step stays stable
    // at dt <= 0.01 s (angular decay rate <= 100 1/s).
    let damp_raw = qdyn * v.aero.area_m2 * v.aero.static_margin_m;
    let damp_cap = 100.0 * i_xx;
    let pitch_damp = (10.0 * pitch_k * damp_raw).min(damp_cap);
    let roll_damp = (10.0 * v.aero.roll_damping * damp_raw).min(0.5 * damp_cap);
    let mut moment = [
        moment_aero[0] - pitch_damp * w[0],
        moment_aero[1] - pitch_damp * w[1],
        -roll_damp * w[2],
    ];
    // `cm_delta` is the gimbal control-moment coefficient. The generated
    // example card sets it to zero, so a card run reproduces the placeholder
    // exactly; a real card adds a finite control moment here.
    if cm_delta != 0.0 {
        let diameter = 2.0 * (v.aero.area_m2 / std::f64::consts::PI).sqrt();
        let cm = cm_delta * qdyn * v.aero.area_m2 * diameter;
        moment[0] += cm * v.thrust_dir[0];
        moment[1] += cm * v.thrust_dir[1];
    }
    // Commanded fin moment: the actual control->dynamics seam. Scaled with
    // dynamic pressure like the other aero moments, so it acts on the coast
    // and descent segments and dies with the atmosphere. Body-X moment is
    // roll, body-Y moment is pitch (see `roll`/`pitch` in optimizer.rs).
    // Gated on a non-zero command to keep the zero path exactly identical.
    if v.control_deg[0] != 0.0 || v.control_deg[1] != 0.0 {
        let diameter = 2.0 * (v.aero.area_m2 / std::f64::consts::PI).sqrt();
        let fin_scale =
            FIN_CONTROL_AUTHORITY * qdyn * v.aero.area_m2 * diameter * std::f64::consts::PI / 180.0;
        moment[0] += fin_scale * v.control_deg[0];
        moment[1] += fin_scale * v.control_deg[1];
    }

    // Launch-rail constraint: before liftoff the rail carries the weight, so
    // the inertial acceleration is exactly zero and the accelerometer reads 1 g.
    let thrust_world_z = mat_vec(r, thrust)[2];
    let on_pad =
        s.pos[2] <= 1e-9 && s.vel[2] <= 0.0 && t < v.burn_end() && thrust_world_z < mass * 9.80665;
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
    d.mass[0] = if mass <= mass_min { 0.0 } else { -v.mdot_at(t) };
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
    let sep_mass = if t < v.stage_sep_time && t + dt >= v.stage_sep_time {
        v.stage_sep_mass
    } else {
        0.0
    };
    let sep_done = if t + dt >= v.stage_sep_time {
        v.stage_sep_mass
    } else {
        0.0
    };
    o.mass[0] = (s.mass[0] + h6 * (k1.mass[0] + 2.0 * k2.mass[0] + 2.0 * k3.mass[0] + k4.mass[0])
        - sep_mass)
        .max(v.dry_mass - sep_done);
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
