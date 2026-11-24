//! Sensor model and canonical 56-byte Potok-M UART request frame.
//!
//! Wire encoding comes from `potok-protocol`, which includes the generated
//! layout from `protocol/uart.yaml`; this module only adapts simulator values
//! into that layout. Specific force follows
//! `f_b = R(q)^T (a_world - g_world) + b_a + v`, so gravity leaks between body
//! axes when the vehicle tilts. All noise uses a per-episode seeded `StdRng`.

use std::collections::VecDeque;

use potok_protocol as proto;
use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::{
    atmosphere::Atmosphere,
    config::{Config, SensorConfig},
    dynamics::{derivative, Phase, State, Vehicle},
    error::{Result, SimError},
    math::{clamp, episode_seed, mat_t_vec, norm, quat_to_matrix, Vec3},
};

pub const FRAME_SIZE: usize = proto::REQUEST_SIZE;

/// Canonical flag bits (from `uart.yaml`), re-exported for simulator callers.
pub const FLAG_IMU_OK: u8 = proto::flags::IMU_OK;
pub const FLAG_BARO_OK: u8 = proto::flags::BARO_OK;
pub const FLAG_PITOT_OK: u8 = proto::flags::PITOT_OK;
pub const FLAG_AI_READY: u8 = proto::flags::AI_READY;
pub const FLAG_MOTOR_ARMED: u8 = proto::flags::MOTOR_ARMED;
pub const FLAG_FAULT: u8 = proto::flags::FAULT;

#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    pub seq: u16,
    pub time_us: u32,
    /// Simulator phase; the wire phase is `phase.protocol_code()`.
    pub phase: Phase,
    pub gyro: [f32; 3],
    pub accel: [f32; 3],
    pub mag: [f32; 3],
    /// Metres; wire field `height`.
    pub height: f32,
    /// Differential pressure in Pa; wire field `pitot_dp` (i16).
    pub pitot_dp: i16,
    /// Canonical protocol flags.
    pub flags: u8,
    pub crc: u16,
    // Sim-only context: these have no bit in the canonical flag byte.
    pub gust: bool,
    pub shear: bool,
    pub saturated: bool,
    /// True when a seeded spike was injected into this frame; also sets the
    /// canonical FAULT bit so the AI interlock rejects the sample.
    pub spike: bool,
    pub pitot_valid: bool,
    /// Six physics features from the RK4 state; no wire representation.
    pub physics: [f32; 6],
}

impl Frame {
    /// AI may consume a frame only with AI_READY set and no fault.
    #[inline]
    pub fn ai_input_valid(&self) -> bool {
        proto::flag_has(self.flags, FLAG_AI_READY) && !proto::flag_has(self.flags, FLAG_FAULT)
    }

    pub fn flags(&self) -> FlagSet {
        FlagSet(self.flags)
    }

    pub fn to_wire(&self) -> proto::RequestFrame {
        proto::RequestFrame {
            seq: self.seq,
            time_us: self.time_us,
            phase: proto::Phase::try_from(self.phase.protocol_code())
                .expect("protocol_code is 0..=4"),
            version: proto::REQUEST_VERSION,
            flags: self.flags,
            reserved: 0,
            gyro: self.gyro,
            accel: self.accel,
            mag: self.mag,
            height: self.height,
            pitot_dp: self.pitot_dp,
        }
    }

    /// Serialize into a caller-provided stack buffer: no heap in the hot loop.
    #[inline]
    pub fn write_bytes(&self, b: &mut [u8; FRAME_SIZE]) {
        proto::encode_request(&self.to_wire(), b).expect("56-byte request buffer");
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = [0u8; FRAME_SIZE];
        self.write_bytes(&mut b);
        b.to_vec()
    }

    #[inline]
    pub fn compute_crc(&self) -> u16 {
        let mut b = [0u8; FRAME_SIZE];
        self.write_bytes(&mut b);
        proto::crc16(&b[..proto::REQUEST_CRC_OFF])
    }

    pub fn from_bytes(b: &[u8]) -> Result<Self> {
        let wire =
            proto::decode_request(b).map_err(|e| SimError::InvalidValidation(format!("{e}")))?;
        let phase = Phase::from_protocol_code(u8::from(wire.phase))
            .ok_or_else(|| SimError::InvalidValidation("unknown wire phase".into()))?;
        let mut f = Self {
            seq: wire.seq,
            time_us: wire.time_us,
            phase,
            gyro: wire.gyro,
            accel: wire.accel,
            mag: wire.mag,
            height: wire.height,
            pitot_dp: wire.pitot_dp,
            flags: wire.flags,
            crc: proto::crc16(&b[..proto::REQUEST_CRC_OFF]),
            gust: false,
            shear: false,
            saturated: proto::flag_has(wire.flags, FLAG_FAULT),
            spike: false,
            pitot_valid: proto::flag_has(wire.flags, FLAG_PITOT_OK),
            physics: [0.0; 6],
        };
        f.crc = f.compute_crc();
        Ok(f)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FlagSet(u8);

impl FlagSet {
    pub fn has(self, flag: u8) -> bool {
        proto::flag_has(self.0, flag)
    }
    pub fn bits(self) -> u8 {
        self.0
    }
}

/// Pure gating rule used by the AI-side adapter and covered by a unit test.
pub fn gating(phase: Phase, sensor_ok: bool, pitot_valid: bool, saturated: bool) -> bool {
    phase.ai_allowed() && sensor_ok && pitot_valid && !saturated
}

/// Build the canonical flag byte. Gust/shear have no protocol bit and stay
/// simulator-only context on `Frame`.
pub fn wire_flags(phase: Phase, sensor_ok: bool, pitot_valid: bool, saturated: bool) -> u8 {
    let mut flags = 0u8;
    if sensor_ok {
        flags |= FLAG_IMU_OK | FLAG_BARO_OK;
    }
    if pitot_valid {
        flags |= FLAG_PITOT_OK;
    }
    if gating(phase, sensor_ok, pitot_valid, saturated) {
        flags |= FLAG_AI_READY;
    }
    if saturated || !sensor_ok {
        flags |= FLAG_FAULT;
    }
    if phase == Phase::Boost {
        flags |= FLAG_MOTOR_ARMED;
    }
    flags
}

/// One frame's worth of delayed measurements.
#[derive(Debug, Clone, Copy)]
struct Delayed {
    gyro: Vec3,
    accel: Vec3,
    mag: Vec3,
    baro: f64,
    pitot: f64,
    spike: bool,
}

/// Per-frame inputs to [`Sensor::measure`]. Grouped so the sampling entry
/// point stays a one-argument method instead of a flat argument list.
pub struct SampleContext<'a> {
    pub seq: u16,
    pub t: f64,
    pub state: &'a State,
    pub vehicle: &'a Vehicle,
    pub atm: &'a Atmosphere,
    pub wind: Vec3,
    pub phase: Phase,
    pub gust: bool,
    pub shear: bool,
}

pub struct Sensor {
    cfg: SensorConfig,
    /// Measurement sample period (1 / frame_rate_hz), not the RK4 step dt.
    /// `measure()` is called once per frame, so the lag filter and the Allan
    /// noise model must integrate on this clock, not on `run.dt`.
    sample_dt: f64,
    sqrt_sample_dt: f64,
    gyro_lag: Vec3,
    accel_lag: Vec3,
    baro_lag: f64,
    pitot_lag: f64,
    bias_scale: [f64; 4],
    noise_scale: [f64; 4],
    mag_bias: Vec3,
    mag_world: Vec3,
    rng: StdRng,
    /// Allan bias random walks. Only advanced when the corresponding
    /// stability knob is positive, so legacy RNG draw order is unchanged.
    gyro_rw_bias: Vec3,
    accel_rw_bias: Vec3,
    allan_gyro: bool,
    allan_accel: bool,
    /// Frame-delay line. Capacity 0 returns the current measurement.
    latency: usize,
    delayed: VecDeque<Delayed>,
}

impl Sensor {
    pub fn new(cfg: &Config, episode: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(episode_seed(cfg.run.seed, episode ^ 0x53454E53));
        let mut bias_scale = [0.0; 4];
        let mut noise_scale = [0.0; 4];
        for i in 0..4 {
            bias_scale[i] = 0.8 + 0.4 * rng.gen::<f64>();
            noise_scale[i] = 0.85 + 0.3 * rng.gen::<f64>();
        }
        let domain = crate::config::sample_domain(cfg, episode);
        let lag_scale = crate::scenario::scenario(episode).lag_scale;
        let s = &cfg.sensor;
        let mag_world = s.mag_world;
        let mag_bias = [
            s.mag_bias[0] * bias_scale[0],
            s.mag_bias[1] * bias_scale[1],
            s.mag_bias[2] * bias_scale[2],
        ];
        let mut sensor_cfg = s.clone();
        sensor_cfg.gyro_noise = domain.gyro_noise;
        sensor_cfg.accel_noise = domain.accel_noise;
        sensor_cfg.lag_tau *= lag_scale;
        let allan_gyro = sensor_cfg.gyro_arw > 0.0 || sensor_cfg.gyro_bias_stability > 0.0;
        let allan_accel = sensor_cfg.accel_arw > 0.0 || sensor_cfg.accel_bias_stability > 0.0;
        let latency = sensor_cfg.latency_steps.min(1024);
        // The sensor is sampled once per transmitted frame, not once per RK4
        // integration step. Using run.dt here made the lag time constant 10x
        // too large and the ARW noise sqrt(10)x too high at 50 Hz.
        let sample_dt = 1.0 / cfg.run.frame_rate_hz;
        Self {
            cfg: sensor_cfg,
            sample_dt,
            sqrt_sample_dt: sample_dt.sqrt(),
            gyro_lag: [0.0; 3],
            accel_lag: [0.0; 3],
            baro_lag: 0.0,
            pitot_lag: 0.0,
            bias_scale,
            noise_scale,
            mag_bias,
            mag_world,
            rng,
            gyro_rw_bias: [0.0; 3],
            accel_rw_bias: [0.0; 3],
            allan_gyro,
            allan_accel,
            latency,
            delayed: VecDeque::with_capacity(latency + 1),
        }
    }

    pub fn measure(&mut self, ctx: SampleContext<'_>) -> Frame {
        let SampleContext {
            seq,
            t,
            state: s,
            vehicle: v,
            atm,
            wind,
            phase,
            gust,
            shear,
        } = ctx;
        let r = quat_to_matrix(s.quat());
        // `derivative.vel` is the inertial acceleration (SoA convention).
        let accel_world = derivative(t, s, wind, atm, v).vel;
        let spec_world = [accel_world[0], accel_world[1], accel_world[2] + 9.80665];
        let spec_body = mat_t_vec(r, spec_world);
        let mag_body = mat_t_vec(r, self.mag_world);
        let v_rel = crate::math::sub(s.vel(), wind);
        let air = atm.at(s.pos()[2]);
        let pitot_true = 0.5 * air.density * norm(v_rel) * norm(v_rel);

        let a = (-self.sample_dt / self.cfg.lag_tau.max(1e-6)).exp();
        let gq = if self.cfg.gyro_quant_step > 0.0 {
            self.cfg.gyro_quant_step
        } else {
            self.cfg.quant_step
        };
        let aq = if self.cfg.accel_quant_step > 0.0 {
            self.cfg.accel_quant_step
        } else {
            self.cfg.quant_step
        };
        let mut gyro = [0.0; 3];
        let mut acc = [0.0; 3];
        let mut mag = [0.0; 3];
        for i in 0..3 {
            // Allan bias stability is a seeded random walk on top of the
            // legacy constant bias. The legacy branch draws exactly two
            // gaussians per axis, in the original order.
            let g_bias = if self.allan_gyro {
                let dw = self.gauss();
                self.gyro_rw_bias[i] += self.cfg.gyro_bias_stability * self.sqrt_sample_dt * dw;
                self.cfg.gyro_bias[i] * self.bias_scale[i] + self.gyro_rw_bias[i]
            } else {
                self.cfg.gyro_bias[i] * self.bias_scale[i]
            };
            let g_std = if self.allan_gyro {
                self.cfg.gyro_arw / self.sqrt_sample_dt
            } else {
                self.cfg.gyro_noise * self.noise_scale[0]
            };
            let g_raw = s.omega()[i] + g_bias + self.gauss() * g_std;
            let a_bias = if self.allan_accel {
                let dw = self.gauss();
                self.accel_rw_bias[i] += self.cfg.accel_bias_stability * self.sqrt_sample_dt * dw;
                self.cfg.accel_bias[i] * self.bias_scale[i + 1] + self.accel_rw_bias[i]
            } else {
                self.cfg.accel_bias[i] * self.bias_scale[i + 1]
            };
            let a_std = if self.allan_accel {
                self.cfg.accel_arw / self.sqrt_sample_dt
            } else {
                self.cfg.accel_noise * self.noise_scale[1]
            };
            let a_raw = spec_body[i] + a_bias + self.gauss() * a_std;
            self.gyro_lag[i] = a * self.gyro_lag[i] + (1.0 - a) * g_raw;
            self.accel_lag[i] = a * self.accel_lag[i] + (1.0 - a) * a_raw;
            gyro[i] = quant(self.gyro_lag[i], gq);
            acc[i] = quant(self.accel_lag[i], aq);
            mag[i] = quant(
                mag_body[i] + self.mag_bias[i] + self.gauss() * self.cfg.mag_noise,
                self.cfg.quant_step,
            );
        }
        let baro_raw = s.pos()[2]
            + self.cfg.baro_bias * self.bias_scale[2]
            + self.gauss() * self.cfg.baro_noise * self.noise_scale[2];
        self.baro_lag = a * self.baro_lag + (1.0 - a) * baro_raw;
        let baro = quant(self.baro_lag, self.cfg.quant_step);
        let pitot_raw = pitot_true
            + self.cfg.pitot_bias * self.bias_scale[3]
            + self.gauss() * self.cfg.pitot_noise * self.noise_scale[3];
        self.pitot_lag = a * self.pitot_lag + (1.0 - a) * pitot_raw;
        let pitot = quant(self.pitot_lag, self.cfg.quant_step);

        // Whole-frame latency: capacity 0 returns this frame's samples, so
        // the legacy path is untouched.
        self.delayed.push_back(Delayed {
            gyro,
            accel: acc,
            mag,
            baro,
            pitot,
            spike: false,
        });
        let mut out = if self.delayed.len() > self.latency {
            self.delayed.pop_front().unwrap()
        } else {
            *self.delayed.front().unwrap()
        };

        // Seeded fault injection. All draws happen only when enabled.
        if self.cfg.spike_probability > 0.0 && self.rng.gen::<f64>() < self.cfg.spike_probability {
            let axis = self.rng.gen_range(0..3);
            let sign = if self.rng.gen::<bool>() { 1.0 } else { -1.0 };
            out.gyro[axis] += sign * self.cfg.spike_amplitude;
            out.spike = true;
        }

        let mut saturated = false;
        for i in 0..3 {
            if out.gyro[i].abs() > self.cfg.gyro_sat {
                out.gyro[i] = clamp(out.gyro[i], -self.cfg.gyro_sat, self.cfg.gyro_sat);
                saturated = true;
            }
            if out.accel[i].abs() > self.cfg.accel_sat {
                out.accel[i] = clamp(out.accel[i], -self.cfg.accel_sat, self.cfg.accel_sat);
                saturated = true;
            }
            if self.cfg.mag_sat > 0.0 && out.mag[i].abs() > self.cfg.mag_sat {
                out.mag[i] = clamp(out.mag[i], -self.cfg.mag_sat, self.cfg.mag_sat);
                saturated = true;
            }
        }
        if self.cfg.baro_sat > 0.0 && out.baro.abs() > self.cfg.baro_sat {
            out.baro = clamp(out.baro, -self.cfg.baro_sat, self.cfg.baro_sat);
            saturated = true;
        }
        let pitot = out.pitot;
        if pitot.abs() > self.cfg.pitot_sat || pitot > i16::MAX as f64 || pitot < i16::MIN as f64 {
            saturated = true;
        }
        let pitot_dp = clamp(pitot, i16::MIN as f64, i16::MAX as f64).round() as i16;
        let pitot_valid = norm(v_rel) > 3.0 && pitot_dp > 0;
        if out.spike {
            saturated = true;
        }
        let gyro = out.gyro;
        let acc = out.accel;
        let mag = out.mag;
        let baro = out.baro;
        let sensor_ok = gyro
            .iter()
            .chain(acc.iter())
            .chain(mag.iter())
            .chain([&baro])
            .all(|x| x.is_finite());
        let flags = wire_flags(phase, sensor_ok && !out.spike, pitot_valid, saturated);
        let mut f = Frame {
            seq,
            time_us: (t * 1_000_000.0).round() as u32,
            phase,
            gyro: cast3(gyro),
            accel: cast3(acc),
            mag: cast3(mag),
            height: baro as f32,
            pitot_dp,
            flags,
            crc: 0,
            gust,
            shear,
            saturated,
            spike: out.spike,
            pitot_valid,
            physics: [0.0; 6],
        };
        f.crc = f.compute_crc();
        f
    }

    fn gauss(&mut self) -> f64 {
        use std::f64::consts::PI;
        let u1 = 1.0 - self.rng.gen::<f64>();
        let u2 = self.rng.gen::<f64>();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

fn quant(v: f64, step: f64) -> f64 {
    if step <= 0.0 {
        v
    } else {
        (v / step).round() * step
    }
}

fn cast3(v: Vec3) -> [f32; 3] {
    [v[0] as f32, v[1] as f32, v[2] as f32]
}

/// Names accepted in `output.features`. `pitot_q` is kept as an alias for the
/// integer `pitot_dp` wire field (Pa).
pub const FEATURE_NAMES: &[&str] = &[
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "acc_x",
    "acc_y",
    "acc_z",
    "mag_x",
    "mag_y",
    "mag_z",
    "baro_h",
    "pitot_q",
    "pitot_dp",
    "alpha_deg",
    "beta_deg",
    "q_dyn_kpa",
    "a_lat_resid",
    "drag_est",
    "static_margin_cal",
];

/// Feature lookup by the names used in `output.features`.
pub fn feature(frame: &Frame, name: &str) -> Option<f32> {
    Some(match name {
        "gyro_x" => frame.gyro[0],
        "gyro_y" => frame.gyro[1],
        "gyro_z" => frame.gyro[2],
        "acc_x" => frame.accel[0],
        "acc_y" => frame.accel[1],
        "acc_z" => frame.accel[2],
        "mag_x" => frame.mag[0],
        "mag_y" => frame.mag[1],
        "mag_z" => frame.mag[2],
        "baro_h" => frame.height,
        "pitot_q" | "pitot_dp" => frame.pitot_dp as f32,
        "alpha_deg" => frame.physics[0],
        "beta_deg" => frame.physics[1],
        "q_dyn_kpa" => frame.physics[2],
        "a_lat_resid" => frame.physics[3],
        "drag_est" => frame.physics[4],
        "static_margin_cal" => frame.physics[5],
        _ => return None,
    })
}
