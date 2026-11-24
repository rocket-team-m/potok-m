//! TOML configuration for the simulator. All random inputs derive from `run.seed`.

use std::path::{Path, PathBuf};

use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::{
    error::{Result, SimError},
    math::episode_seed,
    parachute::ParachuteConfig,
    scenario::{scenario, Split, SCENARIO_COUNT},
    wind_spectral::SpectralConfig,
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

/// The shipped `config/default.toml` spells the apogee trigger
/// `time_since_apogee`, while `DeployTrigger` serializes that variant as
/// `time_since_apogee_s`. Normalize the legacy spelling at the config boundary
/// so the opt-in field can be parsed without changing either shared file. New
/// configs should use the serde spelling; both are accepted here.
fn normalize_trigger_kind(value: &mut toml::Value) {
    if let toml::Value::Table(table) = value {
        if let Some(toml::Value::String(kind)) = table.get_mut("kind") {
            match kind.as_str() {
                "time_since_apogee" => *kind = "time_since_apogee_s".into(),
                "time_since_launch" => *kind = "time_since_launch_s".into(),
                _ => {}
            }
        }
    }
}

fn deserialize_parachute<'de, D>(deserializer: D) -> std::result::Result<ParachuteConfig, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let mut value = toml::Value::deserialize(deserializer)?;
    for stage in ["drogue", "main"] {
        if let Some(trigger) = value.get_mut(stage).and_then(|s| s.get_mut("trigger")) {
            normalize_trigger_kind(trigger);
        }
    }
    ParachuteConfig::deserialize(value).map_err(serde::de::Error::custom)
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

/// Which pitot differential-pressure instrument the simulator models.
/// `production` is the shipped 30 kPa sensor and keeps every golden byte; the
/// `extended` profile is experimental and uses the wider-range instrument.
/// The sensor-side `PitotProfile` is selected from this at episode construction.
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum PitotProfileConfig {
    /// Shipped instrument (default). No config or wire byte changes.
    #[default]
    Production,
    /// Experimental wider-range instrument, opt-in only.
    Extended,
}

/// Omit the default from canonical config serialization so an existing config
/// keeps the same `config_hash` and the same generated bytes.
fn is_production_pitot(profile: &PitotProfileConfig) -> bool {
    *profile == PitotProfileConfig::Production
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SensorConfig {
    /// Pitot instrument range profile. `production` (default) is the legacy
    /// instrument; `extended` is experimental, widens the full scale so the
    /// 26-31 kPa gate-open window stays in range, and changes no gate.
    #[serde(default, skip_serializing_if = "is_production_pitot")]
    pub pitot_profile: PitotProfileConfig,
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

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum AeroSource {
    /// Legacy analytic coefficients (default). Prints a COUNTED marker.
    #[default]
    Placeholder,
    /// Bilinear interpolation of a 2-D (Mach, alpha) TOML coefficient card.
    Card,
}

/// Provenance label for `aero.thrust_profile`. Neither variant claims a
/// static-fire measurement: no such record ships with the crate.
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ThrustSource {
    /// The curve is explicitly NOT claimed to be passport-compliant.
    #[default]
    Experimental,
    /// The curve claims the RDK passport envelope; validated LOUD on load.
    Passport,
}

/// Skipped in canonical config serialization so a pre-existing config without
/// this field keeps the same hash (and therefore the same generated bytes).
fn is_experimental(source: &ThrustSource) -> bool {
    *source == ThrustSource::Experimental
}

/// Passport total impulse window, N·s: 2.2 + 1.0 minimum, 2.4 + 1.2 maximum.
pub const PASSPORT_IMPULSE_NS: (f64, f64) = (3200.0, 3600.0);
/// RDK-2000 (stage 1) passport peak thrust window, N.
pub const PASSPORT_STAGE1_PEAK_N: (f64, f64) = (500.0, 550.0);
/// RDK-1000 (stage 2) passport peak thrust window, N.
pub const PASSPORT_STAGE2_PEAK_N: (f64, f64) = (550.0, 600.0);

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
    /// Distance from the CG to the tail-fin aerodynamic centre, metres. This is
    /// the moment arm `l` in `M_required = m * g_lat * l`: a lateral
    /// acceleration demand at the CG has to be produced by a fin moment about
    /// the CG. Default 0.50 m ~= 4.2 calibers for D = 0.118 m (derived from the
    /// 0.011 m^2 reference area), consistent with fins at the tail of a ~1.8 m
    /// airframe whose CG sits near mid-body. It is a placeholder from vehicle
    /// geometry, not a value from a wind-tunnel or CFD control derivative.
    /// The shipped value is 0.25 m ~= 2.1 calibers: a tail-fin arm on a short
    /// airframe whose CG sits near mid-body. A value near a caliber keeps the
    /// commanded moment (m*g*l) within the fin authority across the envelope;
    /// a full ~4-caliber arm would spin the airframe at this mass and inertia.
    #[serde(default = "default_control_moment_arm_m")]
    pub control_moment_arm_m: f64,
    /// Static-margin tolerance in calibers; sampled per episode and converted
    /// to metres with the frontal diameter.
    #[serde(default = "default_margin_cal_range")]
    pub static_margin_cal_range: Range,
    pub pitch_damping: f64,
    pub roll_damping: f64,
    /// Table of (Mach, Cd) pairs, linearly interpolated and clamped at the ends.
    pub cd_mach: Vec<(f64, f64)>,
    /// Provenance label for `thrust_profile`. The default keeps old configs
    /// loadable; `passport` is validated LOUD against the RDK envelope.
    #[serde(default, skip_serializing_if = "is_experimental")]
    pub thrust_source: ThrustSource,
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

fn default_control_moment_arm_m() -> f64 {
    0.25
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

// --- WORLD05 curriculum -------------------------------------------------
// Independent SplitMix64 keys for the four previously-fixed knobs. Each is
// independent of INIT/DOMA/WIND/SENS, so retuning one knob cannot shift the
// draw of another or disturb the legacy RNG order.
pub const DR_LAG: u64 = 0x4C41_474B; // "LAGK"
pub const DR_TURB_SIGMA: u64 = 0x5453_4947; // "TSIG"
pub const DR_TURB_SCALE: u64 = 0x544C_454E; // "TLEN"
pub const DR_GUST_DUR: u64 = 0x4744_5552; // "GDUR"
pub const DR_GUST_DIR: u64 = 0x4744_4952; // "GDIR"

/// Domain-randomization support for the four knobs that used to be fixed in
/// every training cell. The factor ranges multiply the `[wind]` scalars, so
/// zero turbulence stays off and the config stays the source of scale. The
/// held-out ranges must be disjoint from the training ranges: `validate()`
/// refuses an overlapping config rather than silently degrading to a shared
/// support. Held-out lag is carried by `scenario.lag_scale` (1.5-2.0).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CurriculumConfig {
    #[serde(default = "default_lag_scale_range")]
    pub lag_scale_range: Range,
    #[serde(default = "default_sigma_factor_range")]
    pub sigma_factor_range: Range,
    #[serde(default = "default_scale_factor_range")]
    pub scale_factor_range: Range,
    #[serde(default = "default_duration_factor_range")]
    pub duration_factor_range: Range,
    #[serde(default = "default_heldout_sigma_factor_range")]
    pub heldout_sigma_factor_range: Range,
    #[serde(default = "default_heldout_scale_factor_range")]
    pub heldout_scale_factor_range: Range,
    #[serde(default = "default_heldout_duration_factor_range")]
    pub heldout_duration_factor_range: Range,
}

fn default_lag_scale_range() -> Range {
    Range(1.0, 1.4)
}
fn default_sigma_factor_range() -> Range {
    Range(0.5, 1.5)
}
fn default_scale_factor_range() -> Range {
    Range(2.0 / 3.0, 1.5)
}
fn default_duration_factor_range() -> Range {
    Range(0.5, 1.0)
}
fn default_heldout_sigma_factor_range() -> Range {
    Range(2.0, 2.5)
}
fn default_heldout_scale_factor_range() -> Range {
    Range(0.4, 0.6)
}
fn default_heldout_duration_factor_range() -> Range {
    Range(1.75, 2.25)
}

impl Default for CurriculumConfig {
    fn default() -> Self {
        Self {
            lag_scale_range: default_lag_scale_range(),
            sigma_factor_range: default_sigma_factor_range(),
            scale_factor_range: default_scale_factor_range(),
            duration_factor_range: default_duration_factor_range(),
            heldout_sigma_factor_range: default_heldout_sigma_factor_range(),
            heldout_scale_factor_range: default_heldout_scale_factor_range(),
            heldout_duration_factor_range: default_heldout_duration_factor_range(),
        }
    }
}

/// The per-episode curriculum draws, carried into `EpisodeMeta` so a reader
/// can see which arm actually ran. `lag_scale` is the curriculum multiplier;
/// `Sensor::new` multiplies it again by the scenario's own `lag_scale`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurriculumKnobs {
    pub lag_scale: f64,
    pub turbulence_sigma: f64,
    pub turbulence_scale: f64,
    pub gust_duration_s: f64,
    /// +1.0 or -1.0 applied to `wind.direction_deg`.
    pub direction_sign: f64,
}

#[inline]
fn dr_draw(cfg: &Config, index: u64, tag: u64) -> f64 {
    let mut rng = StdRng::seed_from_u64(episode_seed(cfg.run.seed, index ^ tag));
    rng.gen::<f64>()
}

/// Draw the curriculum knobs for one episode. Training cells use the training
/// ranges; held-out cells use the disjoint OOD ranges (and leave the
/// curriculum lag multiplier at 1.0 because their scenario cell is already
/// OOD). Every knob is always sampled - there is no fixed-value fallback.
pub fn sample_curriculum(cfg: &Config, index: u64) -> CurriculumKnobs {
    let dr = &cfg.curriculum;
    let sc = scenario(index);
    // Cell 0 is the S0 calm replay anchor: no mean wind, gust or shear. It is
    // deliberately left at the shipped nominal so the zero-command gate and
    // the golden episode pin stay a fixed reference for every other arm.
    if sc.split == Split::Train && sc.v10_u == 0.0 && sc.gust_u == 0.0 && sc.shear_u == 0.0 {
        return CurriculumKnobs {
            lag_scale: 1.0,
            turbulence_sigma: cfg.wind.turbulence_sigma,
            turbulence_scale: cfg.wind.turbulence_scale,
            gust_duration_s: cfg.wind.gust_duration,
            direction_sign: 1.0,
        };
    }
    let held_out = sc.split == Split::HeldOut;
    let lag_scale = if held_out {
        1.0
    } else {
        dr.lag_scale_range.sample(dr_draw(cfg, index, DR_LAG))
    };
    let (sigma, scale, duration) = if held_out {
        (
            &dr.heldout_sigma_factor_range,
            &dr.heldout_scale_factor_range,
            &dr.heldout_duration_factor_range,
        )
    } else {
        (
            &dr.sigma_factor_range,
            &dr.scale_factor_range,
            &dr.duration_factor_range,
        )
    };
    CurriculumKnobs {
        lag_scale,
        turbulence_sigma: cfg.wind.turbulence_sigma
            * sigma.sample(dr_draw(cfg, index, DR_TURB_SIGMA)),
        turbulence_scale: cfg.wind.turbulence_scale
            * scale.sample(dr_draw(cfg, index, DR_TURB_SCALE)),
        gust_duration_s: cfg.wind.gust_duration * duration.sample(dr_draw(cfg, index, DR_GUST_DUR)),
        // The shipped +25 deg launch-day direction stays the nominal case;
        // the mirrored sign is a coverage arm at 25% of draws. Both signs are
        // present in training, so a -25 deg gust is no longer unseen.
        direction_sign: if dr_draw(cfg, index, DR_GUST_DIR) < 0.25 {
            -1.0
        } else {
            1.0
        },
    }
}

/// Per-episode config with the curriculum knobs applied to the scalars read
/// by `Wind::new` and `Sensor::new`. The base config is left untouched.
pub fn episode_config(cfg: &Config, index: u64) -> (Config, CurriculumKnobs) {
    let knobs = sample_curriculum(cfg, index);
    let mut ep = cfg.clone();
    ep.wind.turbulence_sigma = knobs.turbulence_sigma;
    ep.wind.turbulence_scale = knobs.turbulence_scale;
    ep.wind.gust_duration = knobs.gust_duration_s;
    ep.wind.direction_deg *= knobs.direction_sign;
    ep.sensor.lag_tau *= knobs.lag_scale;
    (ep, knobs)
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
    /// WORLD05 domain randomization. `#[serde(default)]` keeps older config
    /// files loadable; the defaults are the documented [curriculum] values,
    /// not a runtime degradation.
    #[serde(default = "CurriculumConfig::default")]
    pub curriculum: CurriculumConfig,
    /// CHUTE03: opt-in classical recovery canopy. `enabled = false` by default,
    /// so every existing trajectory, golden hash and A/B stays bit-identical.
    #[serde(default = "ParachuteConfig::default")]
    #[serde(deserialize_with = "deserialize_parachute")]
    pub parachute: ParachuteConfig,
    /// SIMWIND: opt-in von Kármán spectral wind field. `enabled = false` by
    /// default, so every existing trajectory, golden hash and A/B stays
    /// bit-identical.
    #[serde(default)]
    pub spectral: SpectralConfig,
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
        check(
            "aero.control_moment_arm_m",
            self.aero.control_moment_arm_m,
            0.01,
            10.0,
        )?;
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
        self.validate_thrust()?;
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

        let d = &self.curriculum;
        check_range("curriculum.lag_scale_range", d.lag_scale_range, 1.0, 100.0)?;
        check_range(
            "curriculum.sigma_factor_range",
            d.sigma_factor_range,
            1e-6,
            100.0,
        )?;
        check_range(
            "curriculum.scale_factor_range",
            d.scale_factor_range,
            1e-6,
            100.0,
        )?;
        check_range(
            "curriculum.duration_factor_range",
            d.duration_factor_range,
            1e-6,
            100.0,
        )?;
        check_range(
            "curriculum.heldout_sigma_factor_range",
            d.heldout_sigma_factor_range,
            1e-6,
            100.0,
        )?;
        check_range(
            "curriculum.heldout_scale_factor_range",
            d.heldout_scale_factor_range,
            1e-6,
            100.0,
        )?;
        check_range(
            "curriculum.heldout_duration_factor_range",
            d.heldout_duration_factor_range,
            1e-6,
            100.0,
        )?;
        disjoint_above(
            "curriculum.heldout_sigma_factor_range",
            d.sigma_factor_range,
            d.heldout_sigma_factor_range,
        )?;
        disjoint_below(
            "curriculum.heldout_scale_factor_range",
            d.scale_factor_range,
            d.heldout_scale_factor_range,
        )?;
        disjoint_above(
            "curriculum.heldout_duration_factor_range",
            d.duration_factor_range,
            d.heldout_duration_factor_range,
        )?;
        let held_lag_floor = (0..SCENARIO_COUNT as u64)
            .map(scenario)
            .filter(|s| s.split == Split::HeldOut)
            .map(|s| s.lag_scale)
            .fold(f64::INFINITY, f64::min);
        if d.lag_scale_range.1 >= held_lag_floor {
            return Err(SimError::Config(format!(
                "curriculum.lag_scale_range upper {} must stay below the held-out scenario lag floor {}",
                d.lag_scale_range.1, held_lag_floor
            )));
        }
        // CHUTE03: validate even when disabled, so a malformed canopy config
        // cannot lie dormant waiting for someone to flip the switch.
        self.parachute
            .validate()
            .map_err(|e| SimError::Config(e.to_string()))?;
        // SIMWIND: validate even when disabled, so a malformed spectral config
        // cannot lie dormant waiting for someone to flip the switch.
        self.spectral
            .validate()
            .map_err(|e| SimError::Config(e.to_string()))?;
        Ok(())
    }

    /// Validate the thrust table at the config boundary. Structural checks
    /// apply to every source; the passport envelope is enforced only when the
    /// config claims it, with `experimental` as the explicit escape hatch.
    fn validate_thrust(&self) -> Result<()> {
        let table = &self.aero.thrust_profile;
        for (i, &(t, thrust, mdot)) in table.iter().enumerate() {
            if !t.is_finite() || !thrust.is_finite() || !mdot.is_finite() {
                return Err(SimError::Config(format!(
                    "aero.thrust_profile[{i}] has a non-finite (t, thrust, mass_flow) triple; supply finite knots"
                )));
            }
            if t < 0.0 || thrust < 0.0 || mdot < 0.0 {
                return Err(SimError::Config(format!(
                    "aero.thrust_profile[{i}] = ({t}, {thrust}, {mdot}) has a negative component; time, thrust and mass flow cannot be negative"
                )));
            }
            if i > 0 && t <= table[i - 1].0 {
                return Err(SimError::Config(format!(
                    "aero.thrust_profile[{i}] t={t} is not strictly after knot {} t={}; sort and deduplicate the knots",
                    i - 1,
                    table[i - 1].0
                )));
            }
        }
        if self.aero.thrust_source == ThrustSource::Experimental {
            return Ok(());
        }
        let impulse: f64 = table
            .windows(2)
            .map(|w| (w[1].0 - w[0].0) * (w[0].1 + w[1].1) * 0.5)
            .sum();
        let sep = self.aero.stage_sep_time_s;
        let stage1_peak = table
            .iter()
            .filter(|k| k.0 < sep)
            .map(|k| k.1)
            .fold(0.0, f64::max);
        let stage2_peak = table
            .iter()
            .filter(|k| k.0 >= sep)
            .map(|k| k.1)
            .fold(0.0, f64::max);
        let mut violations = Vec::new();
        let (lo, hi) = PASSPORT_IMPULSE_NS;
        if !(lo..=hi).contains(&impulse) {
            violations.push(format!(
                "total impulse {impulse:.1} Ns outside passport [{lo:.0}, {hi:.0}] Ns"
            ));
        }
        for (name, peak, (lo, hi)) in [
            ("stage 1 (RDK-2000)", stage1_peak, PASSPORT_STAGE1_PEAK_N),
            ("stage 2 (RDK-1000)", stage2_peak, PASSPORT_STAGE2_PEAK_N),
        ] {
            if !(lo..=hi).contains(&peak) {
                violations.push(format!(
                    "{name} peak {peak:.1} N outside passport [{lo:.0}, {hi:.0}] N"
                ));
            }
        }
        if !violations.is_empty() {
            return Err(SimError::Config(format!(
                "aero.thrust_source = \"passport\" but aero.thrust_profile violates the RDK passport envelope: {}; supply a passport-compliant table or set aero.thrust_source = \"experimental\" to label this curve explicitly",
                violations.join("; ")
            )));
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

fn disjoint_above(name: &str, train: Range, ood: Range) -> Result<()> {
    if ood.0 <= train.1 {
        Err(SimError::Config(format!(
            "{name} held-out lower {} overlaps training upper {}; held-out must be above training",
            ood.0, train.1
        )))
    } else {
        Ok(())
    }
}

fn disjoint_below(name: &str, train: Range, ood: Range) -> Result<()> {
    if ood.1 >= train.0 {
        Err(SimError::Config(format!(
            "{name} held-out upper {} overlaps training lower {}; held-out must be below training",
            ood.1, train.0
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
