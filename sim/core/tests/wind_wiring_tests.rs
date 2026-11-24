// SPDX-License-Identifier: GPL-3.0-or-later
//! SIMWINDWIRE integration gates: the opt-in spectral wind field in the live
//! episode loop. Disabled must stay bit-identical; enabled must change wind
//! truth, be reproducible from (seed, episode) and report its own metadata.
//! No trajectory target or A/B number is fitted here.

use potok_sim::{config::DEFAULT_TOML, trajectory::TrajectoryTick, Config, Episode, Simulator};

fn base_cfg() -> Config {
    let mut cfg = Config::default();
    cfg.run.duration_s = 2.0; // 1000 steps at dt = 0.002
    cfg.sim.sensors = false;
    cfg.spectral.mode_count = 128;
    assert!(!cfg.spectral.enabled, "the shipped default must be opt-in");
    cfg
}

fn spectral_cfg() -> Config {
    let mut cfg = base_cfg();
    cfg.spectral.enabled = true;
    cfg
}

fn run(cfg: &Config, episode: u64) -> Episode {
    Simulator::new(cfg.clone())
        .expect("config must load")
        .run_episode(episode)
        .expect("episode must complete")
}

fn wind_bits(ticks: &[TrajectoryTick]) -> Vec<u32> {
    ticks
        .iter()
        .flat_map(|t| t.wind.iter().map(|v| v.to_bits()))
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

/// Zero-command bit-identity: an aggressively non-default but DISABLED
/// spectral block has no effect at all. No field is built and the shipped
/// wind path is the whole program.
#[test]
fn disabled_switch_is_bit_identical_to_the_default_program() {
    let off = run(&base_cfg(), 0);
    let mut inert = base_cfg();
    inert.spectral.intensity_mps = 50.0;
    inert.spectral.gust_amplitude_mps = 100.0;
    inert.spectral.mean_wind_mps = 200.0;
    inert.spectral.shear_delta_mps = 100.0;
    let inert = run(&inert, 0);

    assert_eq!(wind_bits(&off.trajectory), wind_bits(&inert.trajectory));
    assert_eq!(off.apogee_m.to_bits(), inert.apogee_m.to_bits());
    assert_eq!(off.max_speed_mps.to_bits(), inert.max_speed_mps.to_bits());
    assert!(off.truth.is_some() && inert.truth.is_some());
}

/// The opt-in arm must actually fly a different wind field, not merely carry
/// an inert config field. Compared on the same seed, episode and step grid.
#[test]
fn enabled_spectral_changes_wind_truth() {
    let off = run(&base_cfg(), 0);
    let on = run(&spectral_cfg(), 0);
    let a: Vec<f32> = off.trajectory.iter().flat_map(|t| t.wind).collect();
    let b: Vec<f32> = on.trajectory.iter().flat_map(|t| t.wind).collect();
    assert!(
        max_abs_diff(&a, &b) > 0.05,
        "enabled spectral wind must differ from the shipped AR(1) field"
    );
    let ta = off.truth.as_ref().unwrap();
    let tb = on.truth.as_ref().unwrap();
    assert_ne!(ta.wind, tb.wind, "the last-tick truth must change too");
}

/// Determinism by seed: same (seed, episode) is bitwise identical, changing
/// either one changes the delivered field. This is the A/B precondition.
#[test]
fn enabled_wind_truth_is_deterministic_by_seed() {
    let a = run(&spectral_cfg(), 7);
    let b = run(&spectral_cfg(), 7);
    assert_eq!(wind_bits(&a.trajectory), wind_bits(&b.trajectory));

    let mut other_seed = spectral_cfg();
    other_seed.run.seed += 1;
    let c = run(&other_seed, 7);
    assert_ne!(wind_bits(&a.trajectory), wind_bits(&c.trajectory));

    let other_episode = run(&spectral_cfg(), 8);
    assert_ne!(
        wind_bits(&a.trajectory),
        wind_bits(&other_episode.trajectory)
    );
}

/// Truth metadata must describe the wind model that was actually flown: with
/// the spectral arm on, gust/shear/v_10 come from its own typed config.
#[test]
fn enabled_metadata_describes_the_spectral_field() {
    let mut cfg = spectral_cfg();
    cfg.spectral.direction_deg = 0.0;
    cfg.spectral.mean_wind_mps = 3.5;
    cfg.spectral.shear_delta_mps = 4.0;
    cfg.spectral.shear_altitude_m = 5.0;
    cfg.spectral.shear_thickness_m = 2.0;
    cfg.spectral.gust_amplitude_mps = 1.5;
    let ep = run(&cfg, 0);

    assert_eq!(ep.meta.v_10, 3.5);
    assert_eq!(ep.meta.shear_delta, 4.0);
    assert!(ep.meta.gust_start_s.is_some());
    let tick = &ep.trajectory[0];
    assert_eq!(tick.shear_altitude_m, 5.0);
    assert_eq!(tick.shear_delta, 4.0);
    assert_eq!(tick.shear_thickness_m, 2.0);
    assert_eq!(tick.gust_amplitude, 1.5);
}

/// Config-boundary gate: malformed spectral values are LOUD even when the
/// switch is off, and a valid `[spectral]` table is a typed opt-in.
#[test]
fn malformed_spectral_config_is_loud_even_when_disabled() {
    let bad_on = format!("{DEFAULT_TOML}\n[spectral]\nenabled = true\nmode_count = 7\n");
    let err = Config::from_toml_str(&bad_on).unwrap_err().to_string();
    assert!(err.contains("spectral.mode_count"), "{err}");

    let bad_off = format!("{DEFAULT_TOML}\n[spectral]\nmode_count = 7\n");
    let err = Config::from_toml_str(&bad_off).unwrap_err().to_string();
    assert!(err.contains("spectral.mode_count"), "{err}");

    let ok = format!("{DEFAULT_TOML}\n[spectral]\nenabled = true\nmode_count = 128\n");
    let cfg = Config::from_toml_str(&ok).expect("valid opt-in table");
    assert!(cfg.spectral.enabled);
}
