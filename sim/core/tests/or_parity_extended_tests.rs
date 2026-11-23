// SPDX-License-Identifier: GPL-3.0-or-later
//! Extended scalar parity of the deterministic simulator against OpenRocket
//! Simulation 9: apogee, maximum speed and time-to-apogee.
//!
//! Evidence (read-only): `../tests/fixtures/or_sim9.csv`, sampled from
//! `/home/sehaxe/Downloads/Telegram Desktop/гугугага.ork`
//! (ZIP -> `rocket.ork`, `<simulations><simulation><name>Simulation 9</name>`,
//! databranch `Маршевая ступень`, `optimumAltitude = 3127.362628772149`,
//! `timeToOptimumAltitude = 26.9`). The fixture's final row is that 26.9 s
//! apogee datapoint; the vmax anchor is the largest total-speed value over all
//! 55 rows, not just the 0.5 s sample grid.
//!
//! This file EXTENDS `tests/or_parity_tests.rs`; it does not replace or weaken
//! it. It flies the same flight (`Config::default()`, episode 0, seed 1337) and
//! uses the same 2 % scalar limits. No coefficient, config or source value is
//! changed here: the thresholds document the current model's measured gap to
//! OR, they are not targets used to tune the model toward OR.
//!
//! `Frame` has no speed field (`pitot_dp` is a noisy saturating i16 dynamic
//! pressure), so per-instant speed is not compared; speed parity uses only
//! `Episode::max_speed_mps`, exactly as the existing oracle does.
//! Time-to-apogee is read from the un-noised 50 Hz `Episode::trajectory`
//! (index of maximum z divided by `run.frame_rate_hz`), not from the noisy
//! barometric `Frame::height`.
//!
//! Thresholds (documented regression gates, not fidelity claims):
//!   * apogee          2 % relative — inherited from the existing oracle.
//!   * vmax            2 % relative — inherited from the existing oracle.
//!   * time-to-apogee  0.25 s absolute — measured sim bias is 0.14 s (tick
//!     1352 = 27.04 s vs OR 26.9 s); 0.25 s is ~1.8x that bias, 2.5x OR's own
//!     0.1 s reporting granularity and 12.5x our 0.02 s trajectory tick
//!     (0.93 % of the 26.9 s baseline). Absolute, because a relative gate on
//!     a 26.9 s baseline would hide a drift of the same physical size.

use potok_sim::{Config, Episode, Simulator};

const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../tests/fixtures/or_sim9.csv");

// Cross-checks for the values parsed out of the fixture below.
const OR_APOGEE_M: f64 = 3127.363;
const OR_VMAX_MPS: f64 = 274.278;
const OR_T_APOGEE_S: f64 = 26.9;

const APOGEE_REL_TOL: f64 = 0.02;
const VMAX_REL_TOL: f64 = 0.02;
const T_APOGEE_ABS_TOL_S: f64 = 0.25;

struct OrRow {
    time_s: f64,
    altitude_m: f64,
    speed_mps: f64,
}

fn load_or_table() -> Vec<OrRow> {
    let text = std::fs::read_to_string(FIXTURE).expect("or_sim9.csv fixture");
    let mut rows = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("time_s,") {
            continue;
        }
        let c: Vec<f64> = line
            .split(',')
            .map(|f| f.trim().parse().expect("numeric CSV field"))
            .collect();
        assert_eq!(c.len(), 8, "one row must have 8 columns: {line}");
        assert!(
            c.iter().all(|v| v.is_finite()),
            "non-finite OR sample: {line}"
        );
        rows.push(OrRow {
            time_s: c[0],
            altitude_m: c[1],
            speed_mps: c[2],
        });
    }
    rows
}

fn rel(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs() / expected.abs()
}

fn episode0() -> (Config, Episode) {
    let cfg = Config::default();
    let sim = Simulator::new(cfg.clone()).expect("Config::default() must load");
    let e = sim.run_episode(0).expect("episode 0 must run");
    (cfg, e)
}

/// Time of the maximum flown altitude, in seconds, from the 50 Hz truth tick.
fn time_to_apogee_s(cfg: &Config, e: &Episode) -> f64 {
    assert!(!e.trajectory.is_empty(), "episode has no trajectory ticks");
    let mut best = 0usize;
    for (i, tick) in e.trajectory.iter().enumerate() {
        assert!(tick.pos[2].is_finite(), "non-finite altitude at tick {i}");
        if tick.pos[2] > e.trajectory[best].pos[2] {
            best = i;
        }
    }
    // The tick must actually sit on the apogee this episode reports: the
    // trajectory is a 50 Hz subsample of the step-sampled apogee in `Episode`.
    let tick_apogee = e.trajectory[best].pos[2] as f64;
    assert!(
        tick_apogee <= e.apogee_m + 1e-6 && rel(e.apogee_m, tick_apogee) < 0.01,
        "tick {best} altitude {tick_apogee:.3} m inconsistent with apogee {:.3} m",
        e.apogee_m
    );
    let tick_dt = 1.0 / cfg.run.frame_rate_hz;
    if let Some(frame) = e.frames.get(best) {
        assert_eq!(
            frame.time_us,
            (best as f64 * tick_dt * 1e6).round() as u32,
            "frame/trajectory tick clocks disagree at apogee tick {best}"
        );
    }
    best as f64 * tick_dt
}

#[test]
fn or_fixture_scalar_anchors() {
    let rows = load_or_table();
    assert_eq!(
        rows.len(),
        55,
        "54 half-second samples + the 26.9 s apogee row"
    );
    assert!(
        rows.windows(2).all(|w| w[1].time_s > w[0].time_s),
        "OR sample times must rise"
    );
    let last = rows.last().expect("non-empty fixture");
    let vmax = rows.iter().map(|r| r.speed_mps).fold(0.0_f64, f64::max);
    assert!((last.time_s - OR_T_APOGEE_S).abs() < 1e-9);
    assert!((last.altitude_m - OR_APOGEE_M).abs() < 0.01);
    assert!((vmax - OR_VMAX_MPS).abs() < 0.01);
    eprintln!(
        "OR anchors from fixture: apogee={:.3} m at {:.1} s, vmax={:.3} m/s",
        last.altitude_m, last.time_s, vmax
    );
}

#[test]
fn episode0_apogee_matches_or_fixture() {
    let (_, e) = episode0();
    let err = rel(e.apogee_m, OR_APOGEE_M);
    eprintln!(
        "apogee: OR {OR_APOGEE_M:.3} m, sim {:.3} m, err {:.2}% (limit {:.0}%)",
        e.apogee_m,
        err * 100.0,
        APOGEE_REL_TOL * 100.0
    );
    assert!(
        err <= APOGEE_REL_TOL,
        "apogee err {:.2}% > {:.0}%",
        err * 100.0,
        APOGEE_REL_TOL * 100.0
    );
}

#[test]
fn episode0_vmax_matches_or_fixture() {
    let (_, e) = episode0();
    let err = rel(e.max_speed_mps, OR_VMAX_MPS);
    eprintln!(
        "vmax: OR {OR_VMAX_MPS:.3} m/s, sim {:.3} m/s, err {:.2}% (limit {:.0}%)",
        e.max_speed_mps,
        err * 100.0,
        VMAX_REL_TOL * 100.0
    );
    assert!(
        err <= VMAX_REL_TOL,
        "vmax err {:.2}% > {:.0}%",
        err * 100.0,
        VMAX_REL_TOL * 100.0
    );
}

#[test]
fn episode0_time_to_apogee_matches_or_fixture() {
    let (cfg, e) = episode0();
    let t = time_to_apogee_s(&cfg, &e);
    let err = (t - OR_T_APOGEE_S).abs();
    eprintln!(
        "time-to-apogee: OR {OR_T_APOGEE_S:.2} s, sim {t:.2} s (tick {}), abs err {err:.2} s (limit {T_APOGEE_ABS_TOL_S:.2} s)",
        (t * cfg.run.frame_rate_hz).round() as usize
    );
    assert!(
        err <= T_APOGEE_ABS_TOL_S,
        "time-to-apogee err {err:.2} s > {T_APOGEE_ABS_TOL_S:.2} s"
    );
}
