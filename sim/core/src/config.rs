//! TOML configuration for the simulator. All random inputs derive from `run.seed`.

use std::path::{Path, PathBuf};

use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::{
    error::{Result, SimError},
    math::episode_seed,
};

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
    /// Domain-randomization envelope for the mean wind at 10 m, m/s.
    /// Scenario cells pick a fraction of this range; `default.toml` omits it.
    #[serde(default = "default_v10_range")]
    pub v_10_range: Range,
    /// Domain-randomization envelope for gust amplitude, m/s.
    #[serde(default = "default_gust_amplitude_range")]
    pub gust_amplitude_range: Range,
    /// Domain-randomization envelope for the shear-layer velocity jump, m/s.
    #[serde(default = "default_shear_delta_range")]
    pub shear_delta_range: Range,
    /// Domain-randomization envelope for the shear-layer altitude, m.
    #[serde(default = "default_shear_altitude_range")]
    pub shear_altitude_range: Range,
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

fn default_v10_range() -> Range {
    Range(0.0, 15.0)
}

fn default_gust_amplitude_range() -> Range {
    Range(5.0, 20.0)
}

fn default_shear_delta_range() -> Range {
    Range(0.0, 8.0)
}

fn default_shear_altitude_range() -> Range {
    Range(50.0, 400.0)
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
    /// Measured-instrument tolerance for gyro noise, rad/s.
    #[serde(default = "default_gyro_noise_range")]
    pub gyro_noise_range: Range,
    /// Measured-instrument tolerance for accelerometer noise, m/s^2.
    #[serde(default = "default_accel_noise_range")]
    pub accel_noise_range: Range,

    // --- Allan-variance instrument model ---------------------------------
    // All default to 0 (off), which keeps the legacy white-noise/bias model
    // and its RNG draw order untouched. A positive ARW replaces the discrete
    // noise std with `arw / sqrt(dt)`; a positive bias stability adds a seeded
    // random walk to the constant bias.
    /// Gyro angle random walk, rad/s/sqrt(Hz) (rad/s at 1 s integration).
    #[serde(default)]
    pub gyro_arw: f64,
    /// Gyro bias stability, rad/s.
    #[serde(default)]
    pub gyro_bias_stability: f64,
    /// Accelerometer velocity random walk, m/s^2/sqrt(Hz).
    #[serde(default)]
    pub accel_arw: f64,
    /// Accelerometer bias stability, m/s^2.
    #[serde(default)]
    pub accel_bias_stability: f64,

    // --- Non-ideal instrument effects ------------------------------------
    /// Whole-frame measurement delay applied to gyro/accel/baro/pitot.
    #[serde(default)]
    pub latency_steps: usize,
    /// Per-channel quantization; 0 means "use `quant_step`".
    #[serde(default)]
    pub gyro_quant_step: f64,
    #[serde(default)]
    pub accel_quant_step: f64,
    /// Additional saturation limits; 0 means "no limit".
    #[serde(default)]
    pub mag_sat: f64,
    #[serde(default)]
    pub baro_sat: f64,
    /// Per-frame probability of injecting a measurement spike (0 = off).
    #[serde(default)]
    pub spike_probability: f64,
    /// Spike magnitude added to one gyro/accel axis.
    #[serde(default)]
    pub spike_amplitude: f64,
}

fn default_gyro_noise_range() -> Range {
    Range(0.0010, 0.0020)
}

fn default_accel_noise_range() -> Range {
    Range(0.008, 0.014)
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum AeroSource {
    /// Legacy analytic coefficients (default). Prints a COUNTED marker.
    Placeholder,
    /// Bilinear interpolation of a 2-D (Mach, alpha) TOML coefficient card.
    Card,
}

impl Default for AeroSource {
    fn default() -> Self {
        Self::Placeholder
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AeroConfig {
    pub area_m2: f64,
    /// Coefficient source; `placeholder` is the default and keeps old configs
    /// byte-identical.
    #[serde(default)]
    pub source: AeroSource,
    /// Card path when `source = "card"`; resolved relative to CWD, then to
    /// the workspace `config/` directory (same rule as `boundary_profile`).
    #[serde(default)]
    pub card_path: Option<PathBuf>,

    pub cd0: f64,
    /// Manufacturing tolerance on cd0; sampled per episode.
    #[serde(default = "default_cd0_range")]
    pub cd0_range: Range,
    /// Normal-force slope per radian of angle of attack.
    pub cna: f64,
    /// Distance from CG to CP along body +Z; positive is statically stable.
    pub static_margin_m: f64,
    /// Static-margin tolerance in calibers; sampled per episode and converted
    /// to metres with the frontal diameter.
    #[serde(default = "default_margin_cal_range")]
    pub static_margin_cal_range: Range,
    pub pitch_damping: f64,
    pub roll_damping: f64,
    /// Table of (Mach, Cd) pairs, linearly interpolated and clamped at the ends.
    pub cd_mach: Vec<(f64, f64)>,
    /// Table of (time_s, thrust_N, mass_flow_kg_s) knots.
    pub thrust_profile: Vec<(f64, f64, f64)>,
    /// First-stage jettison time and dry mass removed at separation.
    /// Defaults come from the RDK-2000 passport (2.8 - 1.88 kg casing).
    #[serde(default = "default_stage_sep_time")]
    pub stage_sep_time_s: f64,
    #[serde(default = "default_stage_sep_mass")]
    pub stage_sep_mass_kg: f64,
}

fn default_stage_sep_time() -> f64 {
    5.5
}
fn default_stage_sep_mass() -> f64 {
    0.92
}

fn default_cd0_range() -> Range {
    Range(0.30, 0.40)
}

fn default_margin_cal_range() -> Range {
    Range(0.85, 1.10)
}

/// The six vehicle numbers supplied to the model as conditioning. They are
/// written to validation metadata alongside the 14 per-frame features.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Conditioning {
    pub mass_kg: f32,
    pub area_m2: f32,
    pub cd0: f32,
    pub static_margin_cal: f32,
    pub thrust_peak_n: f32,
    pub gyro_noise: f32,
}

impl Conditioning {
    pub const COUNT: usize = 6;

    pub fn as_array(self) -> [f32; Self::COUNT] {
        [
            self.mass_kg,
            self.area_m2,
            self.cd0,
            self.static_margin_cal,
            self.thrust_peak_n,
            self.gyro_noise,
        ]
    }

    pub fn from_array(a: [f32; Self::COUNT]) -> Self {
        Self {
            mass_kg: a[0],
            area_m2: a[1],
            cd0: a[2],
            static_margin_cal: a[3],
            thrust_peak_n: a[4],
            gyro_noise: a[5],
        }
    }
}

/// Per-episode airframe/sensor tolerances drawn from `Config` ranges.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DomainParams {
    pub cd0: f64,
    pub static_margin_cal: f64,
    pub gyro_noise: f64,
    pub accel_noise: f64,
}

/// Independent sub-seed so adding a randomization knob cannot shift the
/// legacy `INIT` / `SENS` / `WIND` draw order.
pub fn sample_domain(cfg: &Config, index: u64) -> DomainParams {
    let mut rng = StdRng::seed_from_u64(episode_seed(cfg.run.seed, index ^ 0x444F_4D41));
    DomainParams {
        cd0: cfg.aero.cd0_range.sample(rng.gen()),
        static_margin_cal: cfg.aero.static_margin_cal_range.sample(rng.gen()),
        gyro_noise: cfg.sensor.gyro_noise_range.sample(rng.gen()),
        accel_noise: cfg.sensor.accel_noise_range.sample(rng.gen()),
    }
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
    /// Write the truth-extended v2 metadata (39 floats per episode instead
    /// of 17). Off by default: classic v2 and v1 files stay the default.
    #[serde(default)]
    pub truth_metadata: bool,
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
        check_range("aero.cd0_range", self.aero.cd0_range, 0.0, 2.0)?;
        check_range(
            "aero.static_margin_cal_range",
            self.aero.static_margin_cal_range,
            0.0,
            10.0,
        )?;
        check_range("wind.v_10_range", self.wind.v_10_range, 0.0, 100.0)?;
        check_range(
            "wind.gust_amplitude_range",
            self.wind.gust_amplitude_range,
            0.0,
            100.0,
        )?;
        check_range(
            "wind.shear_delta_range",
            self.wind.shear_delta_range,
            0.0,
            100.0,
        )?;
        check_range(
            "wind.shear_altitude_range",
            self.wind.shear_altitude_range,
            0.0,
            1e5,
        )?;
        check_range(
            "sensor.gyro_noise_range",
            self.sensor.gyro_noise_range,
            0.0,
            1.0,
        )?;
        check_range(
            "sensor.accel_noise_range",
            self.sensor.accel_noise_range,
            0.0,
            100.0,
        )?;
        if self.aero.cd_mach.len() < 2 || self.aero.thrust_profile.len() < 2 {
            return Err(SimError::Config(
                "aero tables need at least two knots".into(),
            ));
        }
        if self.aero.source == AeroSource::Card && self.aero.card_path.is_none() {
            return Err(SimError::Config(
                "aero.source = \"card\" needs aero.card_path".into(),
            ));
        }
        let s = &self.sensor;
        check("sensor.gyro_arw", s.gyro_arw, 0.0, 100.0)?;
        check(
            "sensor.gyro_bias_stability",
            s.gyro_bias_stability,
            0.0,
            100.0,
        )?;
        check("sensor.accel_arw", s.accel_arw, 0.0, 1000.0)?;
        check(
            "sensor.accel_bias_stability",
            s.accel_bias_stability,
            0.0,
            1000.0,
        )?;
        check("sensor.gyro_quant_step", s.gyro_quant_step, 0.0, 1.0)?;
        check("sensor.accel_quant_step", s.accel_quant_step, 0.0, 100.0)?;
        check("sensor.mag_sat", s.mag_sat, 0.0, 1e6)?;
        check("sensor.baro_sat", s.baro_sat, 0.0, 1e9)?;
        check("sensor.spike_probability", s.spike_probability, 0.0, 1.0)?;
        check("sensor.spike_amplitude", s.spike_amplitude, 0.0, 1e6)?;
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

fn check_range(name: &str, r: Range, lo: f64, hi: f64) -> Result<()> {
    check(name, r.0, lo, hi)?;
    check(name, r.1, lo, hi)?;
    if r.1 < r.0 {
        return Err(SimError::Config(format!("{name} has hi < lo")));
    }
    Ok(())
}

/// Built-in configuration; equivalent to `config/default.toml`.
pub const DEFAULT_TOML: &str = include_str!("../../config/default.toml");

impl Default for Config {
    fn default() -> Self {
        Self::from_toml_str(DEFAULT_TOML).expect("embedded default.toml must be valid")
    }
}
