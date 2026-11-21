//! TOML configuration for the simulator. All random inputs derive from `run.seed`.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{Result, SimError};

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
pub struct Range(pub f64, pub f64);

impl Range {
    pub fn sample(self, u: f64) -> f64 {
        self.0 + (self.1 - self.0) * u.clamp(0.0, 1.0)
    }

    pub fn contains(self, v: f64) -> bool {
        v >= self.0 && v <= self.1
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RunConfig {
    pub seed: u64,
    pub duration_s: f64,
    pub dt: f64,
    pub frame_rate_hz: f64,
    /// Number of Monte-Carlo episodes produced by `Episodes::default_count()`.
    #[serde(default = "default_episodes")]
    pub episodes: usize,
    /// Fast path: altitude-grid lookup for the mean wind field. `false` uses
    /// the exact analytic profile as a reference (slower, tiny numeric delta).
    #[serde(default = "default_true")]
    pub fast_field: bool,
}

fn default_true() -> bool {
    true
}

fn default_episodes() -> usize {
    1
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SimConfig {
    /// CSV boundary profile: `altitude_m,temperature_offset_k,wind_factor`.
    pub boundary_profile: PathBuf,
    /// Vertical resolution used when resampling the boundary profile.
    pub altitude_grid_m: f64,
    /// Run the sensor+optimizer path and materialize frames. Off = dynamics only.
    #[serde(default = "default_true")]
    pub sensors: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WindConfig {
    /// Wind speed at 10 m, m/s.
    pub v_10: f64,
    /// Power-law exponent for the mean boundary-layer profile.
    pub shear_exponent: f64,
    /// Velocity jump across the discrete shear layer, m/s.
    pub shear_delta: f64,
    /// Transition thickness of the shear layer, m.
    pub shear_thickness: f64,
    /// Altitude of the shear layer, m.
    #[serde(default = "default_shear_altitude")]
    pub shear_altitude: f64,
    /// Mean wind direction in degrees from world +X towards +Y.
    #[serde(default)]
    pub direction_deg: f64,
    #[serde(default)]
    pub gust_probability: f64,
    #[serde(default)]
    pub gust_amplitude: f64,
    #[serde(default = "default_gust_duration")]
    pub gust_duration: f64,
    #[serde(default)]
    pub turbulence_sigma: f64,
    #[serde(default = "default_turbulence_scale")]
    pub turbulence_scale: f64,
}

fn default_shear_altitude() -> f64 {
    150.0
}
fn default_gust_duration() -> f64 {
    2.0
}
fn default_turbulence_scale() -> f64 {
    60.0
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InitialConfig {
    /// Total lift-off mass, kg.
    pub mass_kg: Range,
    /// Axial CG offset from the geometric reference, m.
    pub cg_offset_m: Range,
    /// Multiplier on the thrust table.
    pub thrust_scale: Range,
    /// Initial tilt angles, degrees (randomized to excite lateral dynamics).
    pub pitch_deg: Range,
    pub yaw_deg: Range,
    /// Small thrust gimbal offset, degrees.
    pub gimbal_deg: Range,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SensorConfig {
    pub gyro_noise: f64,
    pub gyro_bias: [f64; 3],
    pub accel_noise: f64,
    pub accel_bias: [f64; 3],
    pub mag_noise: f64,
    pub mag_bias: [f64; 3],
    /// World magnetic field (gauss), inclined so tilt is observable.
    pub mag_world: [f64; 3],
    pub baro_noise: f64,
    pub baro_bias: f64,
    pub pitot_noise: f64,
    pub pitot_bias: f64,
    /// First-order lag time constant, s (applied to gyro/accel/baro/pitot).
    pub lag_tau: f64,
    /// Quantization step (rad/s for gyro, m/s^2 for accel, m for baro, Pa for pitot).
    pub quant_step: f64,
    pub gyro_sat: f64,
    pub accel_sat: f64,
    pub pitot_sat: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AeroConfig {
    pub area_m2: f64,
    pub cd0: f64,
    /// Normal-force slope per radian of angle of attack.
    pub cna: f64,
    /// Distance from CG to CP along body +Z; positive is statically stable.
    pub static_margin_m: f64,
    pub pitch_damping: f64,
    pub roll_damping: f64,
    /// Table of (Mach, Cd) pairs, linearly interpolated and clamped at the ends.
    pub cd_mach: Vec<(f64, f64)>,
    /// Table of (time_s, thrust_N, mass_flow_kg_s) knots.
    pub thrust_profile: Vec<(f64, f64, f64)>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum OutputMode {
    Stream,
    Validation,
    Both,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OutputConfig {
    pub mode: OutputMode,
    /// Training features exported to the validation file.
    pub features: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub run: RunConfig,
    pub sim: SimConfig,
    pub wind: WindConfig,
    pub initial: InitialConfig,
    pub sensor: SensorConfig,
    pub aero: AeroConfig,
    pub output: OutputConfig,
}

impl Config {
    pub fn from_toml_str(s: &str) -> Result<Self> {
        let cfg: Config = toml::from_str(s).map_err(|e| SimError::Config(e.to_string()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn from_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| SimError::Config(format!("{}: {e}", path.display())))?;
        Self::from_toml_str(&text)
    }

    pub fn validate(&self) -> Result<()> {
        let r = &self.run;
        check("run.dt", r.dt, 1e-6, 1.0)?;
        check("run.duration_s", r.duration_s, 1e-3, 1e5)?;
        check("run.frame_rate_hz", r.frame_rate_hz, 1e-3, 1e5)?;
        let n_steps = r.duration_s / r.dt;
        if (n_steps - n_steps.round()).abs() > 1e-6 {
            return Err(SimError::Config(
                "duration_s must be an integer multiple of dt".into(),
            ));
        }
        let steps_per_frame = 1.0 / (r.dt * r.frame_rate_hz);
        if (steps_per_frame - steps_per_frame.round()).abs() > 1e-6 {
            return Err(SimError::Config(
                "frame_rate_hz must divide 1/dt exactly".into(),
            ));
        }
        check("sim.altitude_grid_m", self.sim.altitude_grid_m, 0.1, 1e4)?;
        check("wind.v_10", self.wind.v_10, 0.0, 200.0)?;
        check("wind.shear_exponent", self.wind.shear_exponent, 0.0, 2.0)?;
        check(
            "wind.gust_probability",
            self.wind.gust_probability,
            0.0,
            1.0,
        )?;
        check("wind.gust_duration", self.wind.gust_duration, 1e-3, 1e4)?;
        check(
            "wind.turbulence_scale",
            self.wind.turbulence_scale,
            1e-3,
            1e6,
        )?;
        check("aero.area_m2", self.aero.area_m2, 1e-6, 1e3)?;
        if self.aero.cd_mach.len() < 2 || self.aero.thrust_profile.len() < 2 {
            return Err(SimError::Config(
                "aero tables need at least two knots".into(),
            ));
        }
        Ok(())
    }

    pub fn steps(&self) -> usize {
        (self.run.duration_s / self.run.dt).round() as usize
    }

    pub fn steps_per_frame(&self) -> usize {
        (1.0 / (self.run.dt * self.run.frame_rate_hz)).round() as usize
    }

    pub fn frames_per_episode(&self) -> usize {
        self.steps() / self.steps_per_frame()
    }

    pub fn feature_count(&self) -> usize {
        self.output.features.len()
    }
}

fn check(name: &str, v: f64, lo: f64, hi: f64) -> Result<()> {
    if !v.is_finite() || v < lo || v > hi {
        Err(SimError::Config(format!(
            "{name} = {v} outside [{lo}, {hi}]"
        )))
    } else {
        Ok(())
    }
}

/// Built-in configuration; equivalent to `config/default.toml`.
pub const DEFAULT_TOML: &str = include_str!("../../config/default.toml");

impl Default for Config {
    fn default() -> Self {
        Self::from_toml_str(DEFAULT_TOML).expect("embedded default.toml must be valid")
    }
}
