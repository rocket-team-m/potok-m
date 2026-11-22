//! Parity oracle for the deterministic simulator against OpenRocket Simulation 9.
//!
//! Source: `/home/sehaxe/Downloads/Telegram Desktop/гугугага.ork`
//! (ZIP -> `rocket.ork` -> `<simulations><simulation><name>Simulation 9</name>`,
//! databranch `Маршевая ступень`). The sampled table is `fixtures/or_sim9.csv`.
//!
//! `Frame` has no speed field: `pitot_dp` is a noisy, saturating i16 dynamic
//! pressure, not a velocity. Per-instant speed is therefore NOT compared
//! against the OR table; only `Episode.max_speed_mps` and `Episode.apogee_m`
//! are, plus `Frame.height` for altitude. This limitation is reported by
//! `episode0_parity_tolerances` on every run.

use potok_sim::{config::sample_domain, math::lerp_table, Config, Episode, Simulator};

const OR_FILE: &str = "/home/sehaxe/Downloads/Telegram Desktop/гугугага.ork";
const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../tests/fixtures/or_sim9.csv");
const AREA_M2: f64 = 0.011;
const FRAME_DT_S: f64 = 0.02;
const SPEED_TIMES: [f64; 6] = [5.0, 10.0, 15.0, 20.0, 25.0, 26.9];
const DRAG_TIMES: [f64; 4] = [5.0, 10.0, 15.0, 20.0];
// Highest speed over ALL OR datapoints, not just the 0.5 s sample grid.
const OR_VMAX_MPS: f64 = 274.278;
const OR_APOGEE_M: f64 = 3127.363;

#[derive(Debug, Clone, Copy)]
struct OrRow {
    time_s: f64,
    altitude_m: f64,
    speed_mps: f64,
    mass_kg: f64,
    drag_n: f64,
    mach: f64,
    cd_axial: f64,
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
        assert!(c.iter().all(|v| v.is_finite()), "non-finite OR sample: {line}");
        rows.push(OrRow {
            time_s: c[0],
            altitude_m: c[1],
            speed_mps: c[2],
            mass_kg: c[3],
            drag_n: c[5],
            mach: c[6],
            cd_axial: c[7],
        });
    }
    rows
}

fn row_at(rows: &[OrRow], t: f64) -> OrRow {
    *rows
        .iter()
        .find(|r| (r.time_s - t).abs() < 1e-9)
        .unwrap_or_else(|| panic!("no OR row at t={t}"))
}

fn altitude_at(e: &Episode, t: f64) -> f64 {
    let i = (t / FRAME_DT_S).round() as usize;
    assert!(i < e.frames.len(), "frame index {i} out of range at t={t}");
    assert_eq!(e.frames[i].time_us, (i * 20_000) as u32, "frame time mismatch");
    e.frames[i].height as f64
}

fn rel(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs() / expected.abs()
}

fn episode0() -> (Simulator, Episode) {
    let sim = Simulator::new(Config::default()).expect("Config::default() must load");
    let e = sim.run_episode(0).expect("episode 0 must run");
    (sim, e)
}

#[test]
fn fixture_covers_or_sim9() {
    let rows = load_or_table();
    assert_eq!(rows.len(), 55, "54 half-second samples + the 26.9 s apogee row");
    assert!(rows.windows(2).all(|w| w[1].time_s > w[0].time_s), "times must rise");
    let apogee = row_at(&rows, 26.9);
    assert!((apogee.altitude_m - OR_APOGEE_M).abs() < 0.01);
    assert!((apogee.mass_kg - 5.475).abs() < 0.01);
    for t in SPEED_TIMES.iter().chain(DRAG_TIMES.iter()) {
        row_at(&rows, *t);
    }
    eprintln!("OR fixture OK: {} rows, source {OR_FILE}", rows.len());
}

#[test]
fn drag_law_with_or_cd_axial_reproduces_or_drag() {
    let (sim, _) = episode0();
    let rows = load_or_table();
    for t in DRAG_TIMES {
        let r = row_at(&rows, t);
        let rho = sim.atmosphere().at(r.altitude_m).density;
        let f = 0.5 * rho * r.speed_mps * r.speed_mps * AREA_M2 * r.cd_axial;
        let err = rel(f, r.drag_n);
        eprintln!(
            "t={t:4.1} rho={rho:.4} v={:7.3} cd_axial={:.4} F_sim={:8.3} F_OR={:8.3} err={:5.2}%",
            r.speed_mps, r.cd_axial, f, r.drag_n, err * 100.0
        );
        assert!(err <= 0.05, "drag law with OR cd_axial missed OR drag at t={t}: {:.2}%", err * 100.0);
    }
}

#[test]
fn sim_cd_table_vs_or_cd_axial() {
    // Diagnostic, not a gate: the task is to REPORT a Cd mismatch, not tune it.
    let cfg = Config::from_toml_str(include_str!("../config/or_sim9.toml")).expect("OR config");
    let cd0_ep0 = sample_domain(&cfg, 0).cd0;
    let rows = load_or_table();
    let cd_base = cfg.aero.cd_mach.first().unwrap().1;
    eprintln!("sim cd_mach = {:?}; cd0_range = {:?}", cfg.aero.cd_mach, cfg.aero.cd0_range);
    eprintln!("episode 0 sampled cd0 = {cd0_ep0:.4}");
    for t in DRAG_TIMES {
        let r = row_at(&rows, t);
        let cd_nom = (cfg.aero.cd0 - cd_base) + lerp_table(&cfg.aero.cd_mach, r.mach);
        let cd_ep0 = (cd0_ep0 - cd_base) + lerp_table(&cfg.aero.cd_mach, r.mach);
        let d_nom = (cd_nom - r.cd_axial) * 100.0 / r.cd_axial;
        let d_ep0 = (cd_ep0 - r.cd_axial) * 100.0 / r.cd_axial;
        eprintln!(
            "t={t:4.1} mach={:.3} OR cd_axial={:.4} sim_cd(cd0=0.40)={:.4} ({:+.1}%) sim_cd(ep0)={:.4} ({:+.1}%)",
            r.mach, r.cd_axial, cd_nom, d_nom, cd_ep0, d_ep0
        );
    }
}

#[test]
fn episode0_parity_tolerances() {
    let rows = load_or_table();
    let (_, e) = episode0();
    eprintln!(
        "OR source: {OR_FILE}\nsim episode 0: apogee={:.3} m, vmax={:.3} m/s, mass0={:.3} kg",
        e.apogee_m, e.max_speed_mps, e.meta.mass_kg
    );
    eprintln!("Frame has no speed field; per-instant speed uses Episode vmax only.");
    eprintln!("  t(s)   OR_alt(m)  sim_alt(m)   alt_err | OR_v(m/s)  sim_vmax");
    for t in SPEED_TIMES {
        let or = row_at(&rows, t);
        let alt = altitude_at(&e, t);
        eprintln!(
            "{t:6.1} {:10.3} {alt:11.3} {:9.2}% | {:9.3}     (no per-frame speed)",
            or.altitude_m, rel(alt, or.altitude_m) * 100.0, or.speed_mps
        );
    }
    let ap_err = rel(e.apogee_m, OR_APOGEE_M) * 100.0;
    let v_err = rel(e.max_speed_mps, OR_VMAX_MPS) * 100.0;
    eprintln!("apogee: OR {OR_APOGEE_M:.3} m, sim {:.3} m, err {ap_err:.2}% (limit 2%)", e.apogee_m);
    eprintln!("vmax:   OR {OR_VMAX_MPS:.3} m/s, sim {:.3} m/s, err {v_err:.2}% (limit 2%)", e.max_speed_mps);

    let mut worst_alt = 0.0_f64;
    let mut worst_t = 0.0_f64;
    for t in SPEED_TIMES {
        let err = rel(altitude_at(&e, t), row_at(&rows, t).altitude_m);
        if err > worst_alt {
            worst_alt = err;
            worst_t = t;
        }
        assert!(err <= 0.05, "altitude at t={t}: err {:.2}% > 5%", err * 100.0);
    }
    eprintln!("worst altitude error: {:.2}% at t={worst_t}", worst_alt * 100.0);
    for t in [5.0_f64, 10.0] {
        let i = (t / 0.02).round() as usize;
        if i > 0 && i + 1 < e.frames.len() {
            let vh = (e.frames[i + 1].height - e.frames[i - 1].height) / 0.04;
            eprintln!("t={t} sim vh={vh:.1} h={:.1}", e.frames[i].height);
        }
    }
    assert!(rel(e.apogee_m, OR_APOGEE_M) <= 0.02, "apogee err {ap_err:.2}% > 2%");
    assert!(rel(e.max_speed_mps, OR_VMAX_MPS) <= 0.02, "vmax err {v_err:.2}% > 2%");
}

