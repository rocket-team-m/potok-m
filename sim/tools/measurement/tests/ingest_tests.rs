//! Gates for the measurement ingest instrument. Every fixture used here is
//! either a synthetic self-test fixture from the tree or a mutated in-memory
//! copy of one; no hardware measurement is asserted anywhere.

use std::fs;
use std::path::{Path, PathBuf};
use std::process;

use potok_measurement::bundle::{
    assert_shipped_recipe_is_synthetic, default_paths, ingest_recipe, Recipe,
};
use potok_measurement::{imu, servo, thrust};
use potok_sim::config::Config;
use potok_sim::dynamics::Vehicle;

fn repo_root() -> PathBuf {
    let (recipe, root) = default_paths();
    assert!(recipe.is_file(), "default recipe missing at {}", recipe.display());
    root
}

fn ingest_shipped() -> potok_measurement::CalibrationBundle {
    let (recipe, root) = default_paths();
    ingest_recipe(&recipe, &root).expect("shipped SYNTHETIC recipe must ingest")
}

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("potok-meas-{}-{name}", process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn write(dir: &Path, name: &str, text: &str) -> PathBuf {
    let path = dir.join(name);
    fs::write(&path, text).unwrap();
    path
}

fn recipe_text(thrust: &Path, gyro: &Path, accel: &Path, servo: &Path, label: &str) -> String {
    format!(
        "schema_version = 1\n\
         expected_source_label = \"{label}\"\n\
         [sources.thrust_stand]\n\
         path = \"{}\"\n\
         measurement = \"thrust_stand\"\n\
         [sources.imu_allan_gyro]\n\
         path = \"{}\"\n\
         measurement = \"imu_allan\"\n\
         channel = \"gyro\"\n\
         [sources.imu_allan_accel]\n\
         path = \"{}\"\n\
         measurement = \"imu_allan\"\n\
         channel = \"accel\"\n\
         [sources.servo_step]\n\
         path = \"{}\"\n\
         measurement = \"servo_step\"\n",
        thrust.display(),
        gyro.display(),
        accel.display(),
        servo.display()
    )
}

fn ingest_with_sources(
    name: &str,
    sources: [PathBuf; 4],
    label: &str,
) -> Result<potok_measurement::CalibrationBundle, potok_measurement::MeasurementError> {
    let dir = temp_dir(name);
    let recipe = write(&dir, "measurement.toml", &recipe_text(&sources[0], &sources[1], &sources[2], &sources[3], label));
    ingest_recipe(&recipe, Path::new("/"))
}

fn shipped_sources() -> [PathBuf; 4] {
    let root = repo_root();
    [
        root.join("tools/measurement/fixtures/thrust_synthetic.csv"),
        root.join("tools/measurement/fixtures/imu_allan_gyro_synthetic.csv"),
        root.join("tools/measurement/fixtures/imu_allan_accel_synthetic.csv"),
        root.join("tools/measurement/fixtures/servo_step_synthetic.csv"),
    ]
}

#[test]
fn shipped_synthetic_recipe_ingests_all_four_sources() {
    let bundle = ingest_shipped();
    assert_shipped_recipe_is_synthetic(&bundle).unwrap();
    assert!(bundle.provenance.all_synthetic());
    assert_eq!(bundle.provenance.sources.len(), 4);
    assert_eq!(bundle.thrust.provenance.rows, 9);
    assert_eq!(bundle.imu_gyro.provenance.rows, 5);
    assert_eq!(bundle.imu_accel.provenance.rows, 5);
    assert_eq!(bundle.servo.provenance.rows, 7);
    assert_eq!(bundle.thrust.provenance.kind, "thrust_stand");
    assert_eq!(bundle.imu_gyro.provenance.kind, "imu_allan");
    assert_eq!(bundle.servo.provenance.kind, "servo_step");
    for source in &bundle.provenance.sources {
        assert_eq!(source.sha256.len(), 64);
        assert_eq!(source.source_label, "SYNTHETIC");
    }
    assert_eq!(bundle.provenance.recipe_sha256.len(), 64);
    assert_eq!(bundle.provenance.calibration_sha256.len(), 64);

    // Typed values come from the core contracts, not from a second estimator.
    assert_eq!(bundle.thrust_curve().samples().len(), 9);
    assert_eq!(bundle.thrust_curve().thrust_at(0.75).unwrap(), 492.5);
    assert!(bundle.thrust_curve().thrust_at(20.0).is_err());
    let cal = bundle.sensor_calibration();
    assert!(cal.gyro_arw > 0.0 && cal.gyro_bias_stability > 0.0);
    assert!(cal.accel_arw > 0.0 && cal.accel_bias_stability > 0.0);
    assert_eq!(cal.gyro_arw, bundle.imu_gyro.points.arw());
    assert_eq!(cal.accel_bias_stability, bundle.imu_accel.points.bias_stability());
    assert_eq!(bundle.servo.response.delay_s, 0.02);
    assert_eq!(bundle.servo.response.rate_per_s, 120.0);
    assert_eq!(bundle.servo.response.deadband, 0.15);
}

#[test]
fn provenance_binds_exact_bytes_and_is_byte_stable() {
    let a = ingest_shipped();
    let b = ingest_shipped();
    assert_eq!(a.canonical_bytes(), b.canonical_bytes());
    assert_eq!(a.provenance.calibration_sha256, b.provenance.calibration_sha256);

    let root = repo_root();
    let expected = potok_measurement::provenance::hash_bytes(
        fs::read(root.join("tools/measurement/fixtures/thrust_synthetic.csv")).unwrap().as_slice(),
    );
    assert_eq!(a.thrust.provenance.sha256, expected);

    let mutated = fs::read_to_string(root.join("tools/measurement/fixtures/thrust_synthetic.csv"))
        .unwrap()
        .replace("480.0", "481.0");
    let dir = temp_dir("hash-mutation");
    write(&dir, "thrust.csv", &mutated);
    let mut sources = shipped_sources();
    sources[0] = dir.join("thrust.csv");
    let changed = ingest_with_sources("hash-mutation-bundle", sources, "SYNTHETIC").unwrap();
    assert_ne!(changed.thrust.provenance.sha256, a.thrust.provenance.sha256);
    assert_ne!(changed.provenance.calibration_sha256, a.provenance.calibration_sha256);
}

#[test]
fn missing_source_file_is_loud() {
    let root = repo_root();
    let recipe_text = fs::read_to_string(root.join("config/measurement.toml")).unwrap();
    let dir = temp_dir("missing-file");
    let recipe = write(&dir, "measurement.toml", &recipe_text.replace("thrust_synthetic.csv", "missing_thrust.csv"));
    let err = ingest_recipe(&recipe, &root).unwrap_err().to_string();
    assert!(err.contains("NOT FOUND"), "{err}");
    assert!(err.contains("missing_thrust.csv"), "{err}");
}

#[test]
fn missing_metadata_is_loud() {
    let root = repo_root();
    let original = fs::read_to_string(root.join("tools/measurement/fixtures/imu_allan_gyro_synthetic.csv")).unwrap();
    let stripped: String = original.lines().filter(|l| !l.starts_with("# source:")).collect::<Vec<_>>().join("\n");
    let err = imu::ingest(&stripped, "gyro.csv").unwrap_err().to_string();
    assert!(err.contains("source"), "{err}");
    assert!(err.contains("gyro.csv"), "{err}");

    let stripped: String = original.lines().filter(|l| !l.starts_with("# channel:")).collect::<Vec<_>>().join("\n");
    let err = imu::ingest(&stripped, "gyro.csv").unwrap_err().to_string();
    assert!(err.contains("channel"), "{err}");
}

#[test]
fn source_label_mismatch_is_loud() {
    let root = repo_root();
    let original = fs::read_to_string(root.join("tools/measurement/fixtures/imu_allan_gyro_synthetic.csv")).unwrap();
    let mut sources = shipped_sources();
    sources[1] = write(&temp_dir("label-mismatch"), "gyro.csv", &original.replace("SYNTHETIC", "REAL_HARDWARE"));
    let err = ingest_with_sources("label-mismatch-bundle", sources, "SYNTHETIC").unwrap_err().to_string();
    assert!(err.contains("REAL_HARDWARE"), "{err}");
    assert!(err.contains("SYNTHETIC"), "{err}");
}

#[test]
fn corrupt_and_non_finite_cells_are_loud() {
    let root = repo_root();
    let original = fs::read_to_string(root.join("tools/measurement/fixtures/thrust_synthetic.csv")).unwrap();
    let bad = original.replace("0.5,480.0", "0.5,abc");
    let err = thrust::ingest(&bad, "thrust.csv").unwrap_err().to_string();
    assert!(err.contains("thrust.csv"), "{err}");
    assert!(err.contains("line 8"), "{err}");
    assert!(err.contains("abc"), "{err}");

    let nan = original.replace("0.5,480.0", "0.5,NaN");
    let err = thrust::ingest(&nan, "thrust.csv").unwrap_err().to_string();
    assert!(err.contains("not finite"), "{err}");
}

#[test]
fn missing_column_and_blank_row_are_loud() {
    let root = repo_root();
    let original = fs::read_to_string(root.join("tools/measurement/fixtures/thrust_synthetic.csv")).unwrap();
    let bad_header = original.replace("t_s,thrust_n", "t_s,force_n");
    let err = thrust::ingest(&bad_header, "thrust.csv").unwrap_err().to_string();
    assert!(err.contains("thrust_n"), "{err}");

    let blank = original.replace("2.0,500.0\n", "2.0,500.0\n\n");
    let err = thrust::ingest(&blank, "thrust.csv").unwrap_err().to_string();
    assert!(err.contains("empty data row"), "{err}");

    let no_rows = original.lines().take_while(|l| !l.starts_with("0.0,")).collect::<Vec<_>>().join("\n");
    let err = thrust::ingest(&no_rows, "thrust.csv").unwrap_err().to_string();
    assert!(err.contains("no data rows"), "{err}");
}

#[test]
fn identification_validation_propagates_loud() {
    let root = repo_root();
    let original = fs::read_to_string(root.join("tools/measurement/fixtures/thrust_synthetic.csv")).unwrap();
    let duplicate_time = original.replace("1.0,505.0", "0.5,505.0");
    let err = thrust::ingest(&duplicate_time, "thrust.csv").unwrap_err().to_string();
    assert!(err.contains("not strictly after"), "{err}");

    let negative = original.replace("0.5,480.0", "0.5,-480.0");
    let err = thrust::ingest(&negative, "thrust.csv").unwrap_err().to_string();
    assert!(err.contains("negative"), "{err}");
}

#[test]
fn wrong_kind_and_wrong_channel_are_loud() {
    let root = repo_root();
    let original = fs::read_to_string(root.join("tools/measurement/fixtures/thrust_synthetic.csv")).unwrap();
    let err = thrust::ingest(&original.replace("measurement: thrust_stand", "measurement: imu_allan"), "thrust.csv")
        .unwrap_err()
        .to_string();
    assert!(err.contains("imu_allan"), "{err}");
    assert!(err.contains("thrust_stand"), "{err}");

    let gyro = fs::read_to_string(root.join("tools/measurement/fixtures/imu_allan_gyro_synthetic.csv")).unwrap();
    let mut sources = shipped_sources();
    sources[1] = write(&temp_dir("wrong-channel"), "gyro.csv", &gyro.replace("channel: gyro", "channel: accel"));
    let err = ingest_with_sources("wrong-channel-bundle", sources, "SYNTHETIC").unwrap_err().to_string();
    assert!(err.contains("channel"), "{err}");
}

#[test]
fn servo_parameter_refusal_and_vehicle_install() {
    let root = repo_root();
    let original = fs::read_to_string(root.join("tools/measurement/fixtures/servo_step_synthetic.csv")).unwrap();
    let err = servo::ingest(&original.replace("rate_per_s: 120.0", "rate_per_s: 0.0"), "servo.csv")
        .unwrap_err()
        .to_string();
    assert!(err.contains("rate_per_s"), "{err}");

    let bundle = ingest_shipped();
    let cfg = Config::default();
    let vehicle = bundle
        .install_servo(Vehicle::from_config(&cfg, 11.0, 1.0, [0.0, 0.0]), 0.01)
        .expect("validated synthetic servo installs on a vehicle");
    let servo = vehicle.servo.as_ref().expect("servo installed");
    assert_eq!(servo[0].borrow().delay_steps(), 2);
    assert_eq!(servo[1].borrow().delay_steps(), 2);
}

#[test]
fn recipe_schema_and_default_loader_are_loud_on_bad_input() {
    assert!(Recipe::from_str("schema_version = 2\n", Path::new("recipe.toml")).is_err());
    let root = repo_root();
    let recipe = fs::read_to_string(root.join("config/measurement.toml")).unwrap();
    assert!(Recipe::from_str(&recipe, Path::new("recipe.toml")).is_ok());
    let empty_label = recipe.replace("expected_source_label = \"SYNTHETIC\"", "expected_source_label = \"\"");
    assert!(Recipe::from_str(&empty_label, Path::new("recipe.toml")).is_err());
}

#[test]
fn apply_to_config_touches_only_allan_sensor_fields() {
    let bundle = ingest_shipped();
    let mut cfg = Config::default();
    let run_before = format!("{:?}", cfg.run);
    let aero_before = format!("{:?}", cfg.aero);
    let wind_before = format!("{:?}", cfg.wind);
    bundle.apply_to_config(&mut cfg);
    let cal = bundle.sensor_calibration();
    assert_eq!(cfg.sensor.gyro_arw, cal.gyro_arw);
    assert_eq!(cfg.sensor.gyro_bias_stability, cal.gyro_bias_stability);
    assert_eq!(cfg.sensor.accel_arw, cal.accel_arw);
    assert_eq!(cfg.sensor.accel_bias_stability, cal.accel_bias_stability);
    assert_eq!(format!("{:?}", cfg.run), run_before);
    assert_eq!(format!("{:?}", cfg.aero), aero_before);
    assert_eq!(format!("{:?}", cfg.wind), wind_before);
    cfg.validate().expect("Allan-derived values must be valid config");
}
