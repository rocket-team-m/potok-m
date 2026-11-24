//! Recipe-driven ingest and the calibrated value objects it produces.
//!
//! The harness reads exactly the files named by `config/measurement.toml`,
//! validates every row through the crate's existing typed contracts, and
//! returns a bundle whose provenance binds both the source bytes and the
//! derived numbers. It never edits a config, a trajectory or a scenario on its
//! own: applying calibration is an explicit caller action.

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use potok_sim::config::Config;
use potok_sim::dynamics::Vehicle;
use potok_sim::identification::ThrustCurve;
use serde::{Deserialize, Serialize};

use crate::error::{MeasurementError, Result};
use crate::imu::{self, ImuAllanMeasurement};
use crate::provenance::{hash_bytes, require_sha256, Provenance, SourceRecord};
use crate::servo::{self, ServoStepMeasurement};
use crate::thrust::{self, ThrustStandMeasurement};

pub const RECIPE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Deserialize)]
pub struct Recipe {
    pub schema_version: u32,
    pub expected_source_label: String,
    pub sources: Sources,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Sources {
    pub thrust_stand: SourceSpec,
    pub imu_allan_gyro: SourceSpec,
    pub imu_allan_accel: SourceSpec,
    pub servo_step: SourceSpec,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SourceSpec {
    /// Repo-root-relative path to the CSV.
    pub path: String,
    /// Expected `measurement` metadata key.
    pub measurement: String,
    /// Expected `channel` metadata key for Allan files.
    #[serde(default)]
    pub channel: Option<String>,
}

/// Allan-derived sensor calibration. These four values are *read* from the
/// measured curves; none is a hand-tuned constant.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SensorCalibration {
    pub gyro_arw: f64,
    pub gyro_bias_stability: f64,
    pub accel_arw: f64,
    pub accel_bias_stability: f64,
}

/// All typed calibration inputs and their provenance.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationBundle {
    pub thrust: ThrustStandMeasurement,
    pub imu_gyro: ImuAllanMeasurement,
    pub imu_accel: ImuAllanMeasurement,
    pub servo: ServoStepMeasurement,
    pub provenance: Provenance,
}

/// JSON-safe digest of a bundle. Core identification types are not serialised
/// here; the digest carries provenance, derived values and counts only.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CalibrationReport {
    pub provenance: Provenance,
    pub sensor: SensorCalibration,
    pub thrust_knots: usize,
    pub thrust_window_s: [f64; 2],
    pub gyro_allan_points: usize,
    pub accel_allan_points: usize,
    pub servo_delay_s: f64,
    pub servo_rate_per_s: f64,
    pub servo_deadband: f64,
    pub servo_step_rows: usize,
}

fn read_text(path: &Path, what: &str, is_recipe: bool) -> Result<String> {
    fs::read_to_string(path).map_err(|e| {
        let detail = if e.kind() == std::io::ErrorKind::NotFound {
            format!("NOT FOUND: {what} at {}; supply the file or fix the recipe path", path.display())
        } else {
            format!("cannot read {what} at {}: {e}; check permissions and encoding", path.display())
        };
        if is_recipe {
            MeasurementError::Recipe {
                path: path.display().to_string(),
                detail,
            }
        } else {
            MeasurementError::Csv {
                path: path.display().to_string(),
                detail,
            }
        }
    })
}

fn check_source(record: &SourceRecord, spec: &SourceSpec, expected_label: &str) -> Result<()> {
    if record.kind != spec.measurement {
        return Err(MeasurementError::Recipe {
            path: spec.path.clone(),
            detail: format!(
                "source declares kind '{}' but recipe expects '{}'; point the recipe at the right file",
                record.kind, spec.measurement
            ),
        });
    }
    if record.source_label != expected_label {
        return Err(MeasurementError::Recipe {
            path: spec.path.clone(),
            detail: format!(
                "source label '{}' does not match expected '{}'; a measurement label is not a guess, fix the '# source:' header or the recipe",
                record.source_label, expected_label
            ),
        });
    }
    Ok(())
}

impl Recipe {
    pub fn from_str(text: &str, path: &Path) -> Result<Self> {
        let recipe: Recipe = toml::from_str(text).map_err(|e| MeasurementError::Recipe {
            path: path.display().to_string(),
            detail: format!("invalid TOML: {e}; fix config/measurement.toml"),
        })?;
        if recipe.schema_version != RECIPE_SCHEMA_VERSION {
            return Err(MeasurementError::Recipe {
                path: path.display().to_string(),
                detail: format!(
                    "unsupported schema_version {}; this harness understands {}",
                    recipe.schema_version, RECIPE_SCHEMA_VERSION
                ),
            });
        }
        if recipe.expected_source_label.trim().is_empty() {
            return Err(MeasurementError::Recipe {
                path: path.display().to_string(),
                detail: "expected_source_label is empty; set it to the label the sources actually declare".into(),
            });
        }
        Ok(recipe)
    }

    pub fn from_file(path: &Path) -> Result<Self> {
        let text = read_text(path, "measurement recipe", true)?;
        Self::from_str(&text, path)
    }
}

/// Ingest every source named by the recipe. `root` anchors relative paths so
/// the caller's working directory cannot change which bytes are hashed.
pub fn ingest_recipe(recipe_path: &Path, root: &Path) -> Result<CalibrationBundle> {
    let recipe_text = read_text(recipe_path, "measurement recipe", true)?;
    let recipe = Recipe::from_str(&recipe_text, recipe_path)?;
    let recipe_sha256 = hash_bytes(recipe_text.as_bytes());
    require_sha256("recipe_sha256", &recipe_sha256)?;

    let expected = recipe.expected_source_label.as_str();
    let load = |spec: &SourceSpec, kind: &str| -> Result<(String, String)> {
        let path = root.join(&spec.path);
        let path = path.as_path();
        let text = read_text(path, kind, false)?;
        Ok((text, path.display().to_string()))
    };

    let spec = &recipe.sources.thrust_stand;
    let (text, path) = load(spec, "thrust stand CSV")?;
    let thrust = thrust::ingest(&text, &path)?;
    check_source(&thrust.provenance, spec, expected)?;

    let spec = &recipe.sources.imu_allan_gyro;
    let (text, path) = load(spec, "IMU Allan CSV (gyro)")?;
    let imu_gyro = imu::ingest(&text, &path)?;
    check_source(&imu_gyro.provenance, spec, expected)?;
    let want = spec.channel.as_deref().unwrap_or("gyro");
    if imu_gyro.points.channel.name() != want {
        return Err(MeasurementError::Recipe {
            path: spec.path.clone(),
            detail: format!(
                "recipe expects channel '{want}' but the CSV declares '{}'",
                imu_gyro.points.channel.name()
            ),
        });
    }

    let spec = &recipe.sources.imu_allan_accel;
    let (text, path) = load(spec, "IMU Allan CSV (accel)")?;
    let imu_accel = imu::ingest(&text, &path)?;
    check_source(&imu_accel.provenance, spec, expected)?;
    let want = spec.channel.as_deref().unwrap_or("accel");
    if imu_accel.points.channel.name() != want {
        return Err(MeasurementError::Recipe {
            path: spec.path.clone(),
            detail: format!(
                "recipe expects channel '{want}' but the CSV declares '{}'",
                imu_accel.points.channel.name()
            ),
        });
    }

    let spec = &recipe.sources.servo_step;
    let (text, path) = load(spec, "servo step CSV")?;
    let servo = servo::ingest(&text, &path)?;
    check_source(&servo.provenance, spec, expected)?;

    let provenance = Provenance {
        schema_version: recipe.schema_version,
        recipe_sha256,
        sources: vec![
            thrust.provenance.clone(),
            imu_gyro.provenance.clone(),
            imu_accel.provenance.clone(),
            servo.provenance.clone(),
        ],
        calibration_sha256: String::new(),
    };
    let mut bundle = CalibrationBundle {
        thrust,
        imu_gyro,
        imu_accel,
        servo,
        provenance,
    };
    let digest = hash_bytes(&bundle.canonical_bytes());
    bundle.provenance.calibration_sha256 = digest;
    Ok(bundle)
}

/// Default repo-root-relative recipe location, fixed at compile time.
pub const DEFAULT_RECIPE: &str = "config/measurement.toml";
pub const DEFAULT_ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");

pub fn default_paths() -> (PathBuf, PathBuf) {
    let raw = PathBuf::from(DEFAULT_ROOT);
    let root = fs::canonicalize(&raw).unwrap_or(raw);
    (root.join(DEFAULT_RECIPE), root)
}

/// Self-test guard: the tree ships SYNTHETIC fixtures only. A non-synthetic
/// label in the shipped recipe is refused LOUD rather than presented as a
/// hardware measurement. Operator-supplied data outside this tree is not
/// validated by this function.
pub fn assert_shipped_recipe_is_synthetic(bundle: &CalibrationBundle) -> Result<()> {
    for src in &bundle.provenance.sources {
        if !src.is_synthetic() {
            return Err(MeasurementError::Recipe {
                path: src.path.clone(),
                detail: format!(
                    "NOT MEASURED: shipped fixture declares source label '{}', not SYNTHETIC; this tree contains no hardware measurement and does not verify operator labels",
                    src.source_label
                ),
            });
        }
    }
    Ok(())
}

impl CalibrationBundle {
    /// Values read from the Allan curves, in the units `SensorConfig` uses.
    pub fn sensor_calibration(&self) -> SensorCalibration {
        SensorCalibration {
            gyro_arw: self.imu_gyro.points.arw(),
            gyro_bias_stability: self.imu_gyro.points.bias_stability(),
            accel_arw: self.imu_accel.points.arw(),
            accel_bias_stability: self.imu_accel.points.bias_stability(),
        }
    }

    /// Apply only the four Allan-derived sensor fields. No trajectory,
    /// scenario, wind, aero or thrust field is touched here.
    pub fn apply_to_config(&self, cfg: &mut Config) {
        let cal = self.sensor_calibration();
        cfg.sensor.gyro_arw = cal.gyro_arw;
        cfg.sensor.gyro_bias_stability = cal.gyro_bias_stability;
        cfg.sensor.accel_arw = cal.accel_arw;
        cfg.sensor.accel_bias_stability = cal.accel_bias_stability;
    }

    /// Install the ingested servo on both fin channels of an existing vehicle.
    /// The caller supplies `dt`; the harness cannot invent a timestep.
    pub fn install_servo(&self, vehicle: Vehicle, dt: f64) -> Result<Vehicle> {
        vehicle
            .with_servo(self.servo.response.clone(), dt)
            .map_err(|e| MeasurementError::Identification(e.to_string()))
    }

    /// The validated thrust curve only. Mass flow is not part of the
    /// measurement, so this harness never fabricates the third field of
    /// `Vehicle::thrust_table`.
    pub fn thrust_curve(&self) -> &ThrustCurve {
        &self.thrust.curve
    }

    /// Canonical byte string over every typed value and source identity.
    /// `f64::to_bits` keeps the digest platform- and locale-independent.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = String::from("potok-measurement/1\n");
        for src in &self.provenance.sources {
            let _ = writeln!(
                out,
                "source|{}|{}|{}|{}|{}",
                src.kind, src.path, src.source_label, src.sha256, src.rows
            );
        }
        let _ = writeln!(out, "recipe_sha256|{}", self.provenance.recipe_sha256);
        for (i, s) in self.thrust.curve.samples().iter().enumerate() {
            let _ = writeln!(out, "thrust[{i}].t_s|{:016x}", s.t_s.to_bits());
            let _ = writeln!(out, "thrust[{i}].thrust_n|{:016x}", s.thrust_n.to_bits());
        }
        for (name, series) in [
            ("gyro", &self.imu_gyro.points),
            ("accel", &self.imu_accel.points),
        ] {
            let _ = writeln!(out, "{name}.channel|{}", series.channel.name());
            for (i, p) in series.points().iter().enumerate() {
                let _ = writeln!(out, "{name}[{i}].tau_s|{:016x}", p.tau_s.to_bits());
                let _ = writeln!(out, "{name}[{i}].sigma|{:016x}", p.sigma.to_bits());
            }
        }
        let r = &self.servo.response;
        let _ = writeln!(out, "servo.delay_s|{:016x}", r.delay_s.to_bits());
        let _ = writeln!(out, "servo.rate_per_s|{:016x}", r.rate_per_s.to_bits());
        let _ = writeln!(out, "servo.deadband|{:016x}", r.deadband.to_bits());
        for (i, s) in r.samples().iter().enumerate() {
            let _ = writeln!(out, "servo[{i}].t_s|{:016x}", s.t_s.to_bits());
            let _ = writeln!(out, "servo[{i}].commanded|{:016x}", s.commanded.to_bits());
            let _ = writeln!(out, "servo[{i}].measured|{:016x}", s.measured.to_bits());
        }
        out.into_bytes()
    }

    pub fn report(&self) -> CalibrationReport {
        let first = self.thrust.curve.samples().first().expect("validated curve has >= 2 knots");
        let last = self.thrust.curve.samples().last().expect("validated curve has >= 2 knots");
        CalibrationReport {
            provenance: self.provenance.clone(),
            sensor: self.sensor_calibration(),
            thrust_knots: self.thrust.curve.samples().len(),
            thrust_window_s: [first.t_s, last.t_s],
            gyro_allan_points: self.imu_gyro.points.points().len(),
            accel_allan_points: self.imu_accel.points.points().len(),
            servo_delay_s: self.servo.response.delay_s,
            servo_rate_per_s: self.servo.response.rate_per_s,
            servo_deadband: self.servo.response.deadband,
            servo_step_rows: self.servo.response.samples().len(),
        }
    }

    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(&self.report())?)
    }
}
