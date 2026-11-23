//! Equivalence and continuity gates for the fast vs exact atmosphere paths.
//!
//! The documented tolerance (README.md / BUILD_LOG.md) is a relative feature
//! deviation below 1e-2; we apply the same `|a-b| / (1+|b|)` measure directly
//! to every air property.

use std::path::Path;

use potok_sim::atmosphere::{Air, Atmosphere};
use potok_sim::config::Config;

const EQUIV_TOL: f64 = 1e-2;

fn profile_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../config/boundary_profile.csv")
}

fn pair() -> (Atmosphere, Atmosphere) {
    let grid = Config::default().sim.altitude_grid_m;
    let fast = Atmosphere::load_with_fast(&profile_path(), grid, true).unwrap();
    let exact = Atmosphere::load_with_fast(&profile_path(), grid, false).unwrap();
    (fast, exact)
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / (1.0 + b.abs())
}

fn assert_equiv(f: &Air, e: &Air, alt: f64) {
    for (name, a, b) in [
        ("temperature", f.temperature, e.temperature),
        ("pressure", f.pressure, e.pressure),
        ("density", f.density, e.density),
        ("speed_of_sound", f.speed_of_sound, e.speed_of_sound),
        ("wind_factor", f.wind_factor, e.wind_factor),
    ] {
        let r = rel(a, b);
        assert!(
            r < EQUIV_TOL,
            "{name} diverges at {alt} m: fast={a} exact={b} rel={r}"
        );
    }
}

#[test]
fn fast_matches_exact_at_operational_altitudes() {
    let (fast, exact) = pair();
    for &alt in &[5_000.0, 6_000.0, 10_000.0, 20_000.0] {
        assert_equiv(&fast.at(alt), &exact.at(alt), alt);
    }
}

#[test]
fn fast_matches_exact_above_the_table() {
    // The old fast path froze at the grid top while exact continued, so the
    // two diverged above 20 km. Both must now use the same continuation.
    let (fast, exact) = pair();
    for &alt in &[20_000.0, 25_000.0, 30_000.0] {
        assert_equiv(&fast.at(alt), &exact.at(alt), alt);
    }
}

#[test]
fn fast_does_not_freeze_above_the_profile_top() {
    // PIN the reported bug: at 6 km the frozen fast row gave rho ~= 0.760
    // while the exact continuation gave ~= 0.660 (15% density error).
    let (fast, _) = pair();
    let d5 = fast.at(5_000.0).density;
    let d6 = fast.at(6_000.0).density;
    assert!(d6 < d5, "fast density froze at grid top: {d5} -> {d6}");
    assert!(d6 < 0.72, "fast density at 6 km looks frozen: {d6}");
}

#[test]
fn pressure_law_above_20km_matches_linear_temperature() {
    // The 20-32 km layer has T = 216.65 + 0.001*(h-20000): pressure must use
    // the matching power law with that varying T, not an isothermal exp().
    let (_, exact) = pair();
    let p20 = exact.at(20_000.0).pressure;
    let r = 9.80665 / (287.05287 * 0.001);
    for &alt in &[25_000.0, 30_000.0, 32_000.0] {
        let a = exact.at(alt);
        let t_base = 216.65 + 0.001 * (alt - 20_000.0);
        let expected_p = p20 * (216.65 / t_base).powf(r);
        assert!(
            rel(a.pressure, expected_p) < 1e-9,
            "pressure law at {alt} m: got {} expected {}",
            a.pressure,
            expected_p
        );
        // Density must be consistent with the pressure AND the varying T.
        let expected_rho = expected_p / (287.05287 * a.temperature);
        assert!(
            rel(a.density, expected_rho) < 1e-9,
            "density at {alt} m: got {} expected {}",
            a.density,
            expected_rho
        );
        // Temperature rises with altitude in this layer, so the lapse-rate
        // power law decays pressure SLOWER than a frozen-216.65 isothermal
        // exp() (the old inconsistent formula). Assert the two really differ.
        let isothermal = p20 * (-9.80665 * (alt - 20_000.0) / (287.05287 * 216.65)).exp();
        assert!(
            a.pressure > isothermal,
            "pressure at {alt} m did not use the lapse-rate law: {} <= isothermal {}",
            a.pressure,
            isothermal
        );
    }
}
