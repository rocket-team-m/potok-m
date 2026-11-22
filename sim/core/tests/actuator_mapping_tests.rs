//! Physical command -> fin actuator mapping. The command is a lateral
//! acceleration demand (m/s^2); `dynamics::solve_fin_deflection` turns it into
//! a fin deflection by a moment balance. These tests measure the flown truth,
//! not a surrogate, and keep the zero-command path bit-identical.

use std::path::PathBuf;

use potok_sim::{
    fin_clamp_uses, fin_no_authority_uses, lateral_rms_m, solve_fin_deflection, AeroCard,
    AeroSource, Config, ControlFault, SimError, Simulator, MAX_FIN_DEFLECTION_RAD,
    MIN_CONTROL_QDYN_PA, PLACEHOLDER_CM_DELTA,
};

const HELD_OUT: [u64; 4] = [16, 17, 18, 19];

fn sim(seed: u64) -> Simulator {
    let mut cfg = Config::default();
    cfg.run.seed = seed;
    Simulator::new(cfg).expect("default config")
}

fn pooled_lateral(rms: &[f64]) -> f64 {
    (rms.iter().map(|v| v * v).sum::<f64>() / rms.len() as f64).sqrt()
}

#[test]
fn moment_balance_solves_deflection_and_clamps() {
    let area = Config::default().aero.area_m2;
    let arm = Config::default().aero.control_moment_arm_m;
    let mass = 11.0;
    let qdyn = 10_000.0;

    // Exact inversion: applying the solved deflection reproduces M_req.
    let g = 1.0;
    let delta = solve_fin_deflection(g, mass, qdyn, area, PLACEHOLDER_CM_DELTA, arm).unwrap();
    let diameter = 2.0 * (area / std::f64::consts::PI).sqrt();
    let applied = qdyn * area * diameter * PLACEHOLDER_CM_DELTA * delta;
    let required = mass * g * arm;
    assert!((applied - required).abs() < 1e-9, "{applied} vs {required}");
    assert!(delta > 0.0 && delta < MAX_FIN_DEFLECTION_RAD);

    // Twice the demand needs twice the deflection (linear until the stop).
    let delta2 =
        solve_fin_deflection(2.0 * g, mass, qdyn, area, PLACEHOLDER_CM_DELTA, arm).unwrap();
    assert!((delta2 - 2.0 * delta).abs() < 1e-9);

    // Out-of-range deflections clamp to the mechanical stop, and the clamp is
    // COUNTED rather than silent.
    let before = fin_clamp_uses();
    let big = solve_fin_deflection(50.0, mass, qdyn, area, PLACEHOLDER_CM_DELTA, arm).unwrap();
    assert_eq!(big, MAX_FIN_DEFLECTION_RAD);
    assert!(fin_clamp_uses() > before, "clamp must be COUNTED");
    let neg = solve_fin_deflection(-50.0, mass, qdyn, area, PLACEHOLDER_CM_DELTA, arm).unwrap();
    assert_eq!(neg, -MAX_FIN_DEFLECTION_RAD);

    // Zero command leaves authority untouched, no marker.
    assert_eq!(
        solve_fin_deflection(0.0, mass, qdyn, area, PLACEHOLDER_CM_DELTA, arm).unwrap(),
        0.0
    );
}

#[test]
fn thin_air_drops_command_to_safe_zero_with_a_counter() {
    let area = Config::default().aero.area_m2;
    let arm = Config::default().aero.control_moment_arm_m;
    let before = fin_no_authority_uses();
    let delta = solve_fin_deflection(
        2.0,
        11.0,
        MIN_CONTROL_QDYN_PA * 0.5,
        area,
        PLACEHOLDER_CM_DELTA,
        arm,
    )
    .unwrap();
    assert_eq!(delta, 0.0, "no authority -> safe zero, not a blow-up");
    assert!(fin_no_authority_uses() > before, "drop must be COUNTED");
}

#[test]
fn unknown_authority_is_loud_even_for_a_small_command() {
    let area = Config::default().aero.area_m2;
    let arm = Config::default().aero.control_moment_arm_m;
    for cm in [0.0, -1.0, f64::NAN] {
        match solve_fin_deflection(0.5, 11.0, 10_000.0, area, cm, arm) {
            Err(ControlFault {
                reason: "cm_delta",
                value,
            }) => {
                assert!(value.is_nan() || value == cm, "value {value} vs {cm}");
            }
            other => panic!("expected LOUD cm_delta fault, got {other:?}"),
        }
    }
    // And through the whole episode: a card with zero cm_delta + a command is
    // a LOUD error, not a silent zero moment.
    let dir = std::env::temp_dir().join(format!("potok-auth-{}-.toml", std::process::id()));
    let rows = ["[0.0, 0.0]", "[0.0, 0.0]"].join(", ");
    let card = format!(
        "baseline_cd0 = 0.0\n[grid]\nmach = [0.0, 3.0]\nalpha_deg = [-180.0, 180.0]\n\
         [coefficients]\ncd0 = [{rows}]\ncna = [{rows}]\n\
         pitch_damping = [{rows}]\ncm_delta = [{rows}]\n"
    );
    std::fs::write(&dir, card).unwrap();
    let _parsed = AeroCard::from_file(&dir).unwrap();
    let mut cfg = Config::default();
    cfg.aero.source = AeroSource::Card;
    cfg.aero.card_path = Some(PathBuf::from(&dir));
    let sim = Simulator::new(cfg).expect("zero-authority card config");
    assert!(matches!(
        sim.run_episode_with_control(0, &|_f| (1.0, 0.0)),
        Err(SimError::Control(_))
    ));
    std::fs::remove_file(&dir).unwrap();
}

#[test]
fn zero_command_stays_bit_identical_under_the_new_mapping() {
    let sim = sim(1337);
    let legacy = sim.run_episode(0).unwrap();
    let zero = sim.run_episode_with_control(0, &|_| (0.0, 0.0)).unwrap();
    assert_eq!(zero.trajectory, legacy.trajectory);
    assert_eq!(zero.apogee_m.to_bits(), legacy.apogee_m.to_bits());
    assert_eq!(zero.features, legacy.features);
    assert_eq!(zero.targets, legacy.targets);
}

#[test]
fn oracle_accel_command_improves_lateral_rms_on_held_out() {
    let mut report = String::new();
    let mut improved = true;
    for seed in [1u64, 2, 3] {
        let sim = sim(seed);
        let base: Vec<f64> = HELD_OUT
            .iter()
            .map(|&i| lateral_rms_m(&sim.run_episode(i).unwrap().trajectory))
            .collect();
        let oracle: Vec<f64> = HELD_OUT
            .iter()
            .map(|&i| lateral_rms_m(&sim.run_episode_with_oracle(i).unwrap().trajectory))
            .collect();
        let b = pooled_lateral(&base);
        let o = pooled_lateral(&oracle);
        report.push_str(&format!(
            "seed {seed}: baseline {b:.6} m -> oracle {o:.6} m ({:+.6} m)\n",
            o - b
        ));
        improved &= o < b;
    }
    eprintln!("\n{report}");
    assert!(
        improved,
        "physical oracle command must improve lateral RMS on the held-out set:\n{report}"
    );
}

#[test]
fn wrong_sign_command_worsens_lateral_rms_on_held_out() {
    let sim = sim(1);
    let base = pooled_lateral(
        &HELD_OUT
            .iter()
            .map(|&i| lateral_rms_m(&sim.run_episode(i).unwrap().trajectory))
            .collect::<Vec<_>>(),
    );
    let wrong = pooled_lateral(
        &HELD_OUT
            .iter()
            .map(|&i| {
                lateral_rms_m(
                    &sim.run_episode_with_control_ctx(i, &|ctx| {
                        let g = potok_sim::ReferenceOptimizer::new(sim.config()).fin_accel_command(
                            ctx.state,
                            ctx.wind,
                            ctx.vehicle,
                            ctx.phase,
                            ctx.frame,
                        );
                        (-g[0], -g[1])
                    })
                    .unwrap()
                    .trajectory,
                )
            })
            .collect::<Vec<_>>(),
    );
    eprintln!("wrong-sign lateral RMS {base:.6} -> {wrong:.6} m");
    assert!(wrong > base, "flipping the sign must make control hurt");
}

#[test]
fn v1_and_v2_sidecars_still_round_trip_controlled_flight() {
    use potok_sim::{
        read_trajectory, sidecar_path, write_trajectory, write_trajectory_v2, TrajectoryTick,
    };
    let sim = sim(42);
    let cfg = Config::default();
    let eps: Vec<_> = HELD_OUT
        .iter()
        .map(|&i| sim.run_episode_with_oracle(i).unwrap())
        .collect();
    let validation = std::env::temp_dir().join(format!("potok-act-{}-.bin", std::process::id()));
    let sidecar = sidecar_path(&validation);

    write_trajectory(&sidecar, &cfg, &eps).unwrap();
    let v1 = read_trajectory(&sidecar).unwrap();
    assert_eq!(v1.version, 1);
    for (got, want) in v1.episodes.iter().zip(eps.iter()) {
        assert!(got
            .ticks
            .iter()
            .map(|t: &TrajectoryTick| t.as_array())
            .eq(want.trajectory.iter().map(|t| t.as_array())));
    }
    assert_eq!(
        std::fs::metadata(&sidecar).unwrap().len() as usize,
        64 + eps.len() * (8 + cfg.frames_per_episode() * 6 * 4)
    );

    write_trajectory_v2(&sidecar, &cfg, &eps).unwrap();
    let v2 = read_trajectory(&sidecar).unwrap();
    assert_eq!(v2.version, 2);
    for (got, want) in v2.episodes.iter().zip(eps.iter()) {
        assert!(got
            .ticks
            .iter()
            .map(|t| t.g_theta)
            .eq(want.trajectory.iter().map(|t| t.g_theta)));
        assert_eq!(got.lateral_rms_m(), lateral_rms_m(&want.trajectory));
    }
    std::fs::remove_file(&sidecar).unwrap();
}

#[test]
fn placeholder_cm_delta_is_counted() {
    use potok_sim::cm_delta_placeholder_uses;
    let before = cm_delta_placeholder_uses();
    let _sim = sim(7);
    assert!(cm_delta_placeholder_uses() > before);
    assert_eq!(PLACEHOLDER_CM_DELTA, 5.0);
}
