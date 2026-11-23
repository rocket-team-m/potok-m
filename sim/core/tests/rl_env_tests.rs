// SPDX-License-Identifier: GPL-3.0-or-later
//! Hard tests for the stepwise RL environment.
//!
//! `rl_env.rs` is compiled straight from its source file so no file other than
//! the two deliverables is touched. Root re-exports make the module's
//! `crate::…` paths resolve here exactly as they will once a parent adds
//! `pub mod rl_env;` to `lib.rs`.

pub use potok_sim::{config, control, dynamics, episode, error, math, physics, reward, sensor};

#[path = "../src/rl_env.rs"]
mod rl_env;

use config::Config;
use reward::{
    FlightSample, GateLimits, GateObservation, GateViolation, RewardWeights, TruthSidecar,
};
use rl_env::{
    Action, Observation, RlEnv, Shaping, ACTION_DIM, CONTEXT_DIM, OBS_DIM, STATE_DIM, STATE_NAMES,
};

const WEIGHTS: RewardWeights = RewardWeights {
    lateral: 2.0,
    apogee: 0.5,
    effort: 3.0,
    risk: 7.0,
};

fn short_cfg() -> Config {
    let mut cfg = Config::default();
    cfg.run.duration_s = 2.0; // 100 frames at 50 Hz: terminal without landing
    cfg.validate().unwrap();
    cfg
}

fn permissive_truth(cfg: &Config) -> TruthSidecar {
    let dt = cfg.steps_per_frame() as f64 * cfg.run.dt;
    let reference = (0..cfg.frames_per_episode())
        .map(|i| FlightSample {
            t_s: i as f64 * dt,
            pos_m: [10.0, 0.0, 100.0],
            ctrl_mps2: 0.0,
            safety_violation: false,
        })
        .collect();
    TruthSidecar {
        reference,
        limits: GateLimits {
            mach_max: 10.0,
            safety_factor_min: 0.0,
            static_margin_cal_min: 0.0,
        },
        observed: GateObservation {
            mach_peak: 0.5,
            safety_factor_min: 5.0,
            static_margin_cal_min: 1.0,
        },
    }
}

fn env(cfg: Config, truth: TruthSidecar) -> RlEnv {
    RlEnv::new(cfg, 0, truth, WEIGHTS, Shaping::default()).unwrap()
}

fn run_to_terminal(e: &mut RlEnv, action: Action) -> rl_env::Step {
    let (_, _) = e.reset().unwrap();
    for _ in 0..e.config().frames_per_episode() + 4 {
        let s = e.step(action).unwrap();
        if s.done {
            return s;
        }
    }
    panic!("episode did not terminate");
}

fn same_obs(a: &Observation, b: &Observation) -> bool {
    a.tick == b.tick
        && a.phase == b.phase
        && a.ai_input_valid == b.ai_input_valid
        && a.features
            .iter()
            .zip(b.features.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits())
        && a.conditioning
            .iter()
            .zip(b.conditioning.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits())
}

fn finite_obs(o: &Observation) -> bool {
    o.as_array().iter().all(|v| v.is_finite())
}

#[test]
fn observation_layout_and_axis_contract() {
    assert_eq!(STATE_DIM, 14);
    assert_eq!(CONTEXT_DIM, 6);
    assert_eq!(OBS_DIM, 20);
    assert_eq!(ACTION_DIM, 2);
    assert_eq!(&STATE_NAMES[8..], &physics::PHYSICS_FEATURES);
    assert_eq!(STATE_NAMES[0], "gyro_x");
    assert_eq!(STATE_NAMES[7], "pitot_q");

    let a = Action { ay: 2.5, ax: -1.25 };
    assert_eq!(a.fin_command(), [2.5, 1.25]);
    assert_eq!(Action::ZERO.fin_command(), [0.0, 0.0]);
    assert!(a.checked().is_ok());
    assert!(Action {
        ay: 10.0,
        ax: -10.0
    }
    .checked()
    .is_ok());
    assert!(Action { ay: 10.1, ax: 0.0 }.checked().is_err());
    assert!(Action {
        ay: 0.0,
        ax: f32::NAN
    }
    .checked()
    .is_err());
    assert!(Action {
        ay: f32::INFINITY,
        ax: 0.0
    }
    .checked()
    .is_err());
}

#[test]
fn reset_returns_a_finite_twenty_value_state() {
    let cfg = short_cfg();
    let mut e = env(cfg, permissive_truth(&short_cfg()));
    let (obs, info) = e.reset().unwrap();
    assert_eq!(obs.features.len(), STATE_DIM);
    assert_eq!(obs.conditioning.len(), CONTEXT_DIM);
    assert_eq!(obs.as_array().len(), OBS_DIM);
    assert!(finite_obs(&obs));
    assert_eq!(obs.conditioning, e.conditioning().as_array());
    assert_eq!(info.seed, e.seed());
    assert_eq!(info.episode_index, 0);
    assert_eq!(info.tick, obs.tick);
    assert!(info.truth.pos_m.iter().all(|v| v.is_finite()));
    assert!(!e.is_done());
}

#[test]
fn conditioning_matches_the_episode_meta_it_wraps() {
    let cfg = short_cfg();
    let expected = episode::Simulator::new(cfg.clone())
        .unwrap()
        .run_episode(0)
        .unwrap()
        .meta
        .conditioning;
    let e = env(cfg, permissive_truth(&short_cfg()));
    assert_eq!(e.conditioning(), expected);
}

#[test]
fn stepping_is_deterministic_and_actions_move_the_state() {
    let mut a = env(short_cfg(), permissive_truth(&short_cfg()));
    let mut b = env(short_cfg(), permissive_truth(&short_cfg()));
    let mut z = env(short_cfg(), permissive_truth(&short_cfg()));
    let (oa, ia) = a.reset().unwrap();
    let (ob, _) = b.reset().unwrap();
    let (oz, iz) = z.reset().unwrap();
    assert!(same_obs(&oa, &ob));
    assert!(same_obs(&oa, &oz));

    let cmd = Action { ay: 4.0, ax: -3.0 };
    let mut pos_a = ia.truth.pos_m;
    let mut pos_z = iz.truth.pos_m;
    for _ in 0..50 {
        let sa = a.step(cmd).unwrap();
        let sb = b.step(cmd).unwrap();
        let sz = z.step(Action::ZERO).unwrap();
        assert!(same_obs(&sa.obs, &sb.obs));
        assert_eq!(sa.reward.to_bits(), sb.reward.to_bits());
        assert!(finite_obs(&sa.obs));
        assert!(!sa.done && !sb.done && !sz.done);
        pos_a = sa.info.truth.pos_m;
        pos_z = sz.info.truth.pos_m;
    }
    let drift = pos_a
        .iter()
        .zip(pos_z.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max);
    assert!(
        drift > 1e-9,
        "bounded actions must change the flown trajectory"
    );
}

#[test]
fn reset_rewinds_to_the_same_observation_and_reward() {
    let mut e = env(short_cfg(), permissive_truth(&short_cfg()));
    let (first, _) = e.reset().unwrap();
    let r1 = e.step(Action { ay: 1.0, ax: 2.0 }).unwrap();
    let r2 = e.step(Action { ay: -1.0, ax: 0.5 }).unwrap();
    let (again, _) = e.reset().unwrap();
    assert!(same_obs(&first, &again));
    let s1 = e.step(Action { ay: 1.0, ax: 2.0 }).unwrap();
    let s2 = e.step(Action { ay: -1.0, ax: 0.5 }).unwrap();
    assert!(same_obs(&r1.obs, &s1.obs));
    assert!(same_obs(&r2.obs, &s2.obs));
    assert_eq!(r1.reward.to_bits(), s1.reward.to_bits());
    assert_eq!(r2.reward.to_bits(), s2.reward.to_bits());
}

#[test]
fn potential_bonus_telescopes_at_gamma_one() {
    let s = Shaping {
        gamma: 1.0,
        lateral: 0.0,
        apogee: 1.0,
        alpha_deg: 5.0,
    };
    let point = |z: f64, _alpha: f64| rl_env::TickTruth {
        t_s: 0.0,
        pos_m: [3.0, 4.0, z],
        vel_mps: [0.0; 3],
        quat: [1.0, 0.0, 0.0, 0.0],
        omega_radps: [0.0; 3],
        mass_kg: 11.0,
        wind_mps: [0.0; 3],
    };
    let p0 = s.potential(&point(10.0, 0.0), 0.0);
    let p1 = s.potential(&point(50.0, 8.0), 2.0);
    let p2 = s.potential(&point(120.0, -7.0), 1.0);
    let sum = s.bonus(p0, p1) + s.bonus(p1, p2);
    assert_eq!(sum.to_bits(), s.bonus(p0, p2).to_bits());
}

#[test]
fn terminal_verifier_and_info_carry_truth_gates_and_seed() {
    let cfg = short_cfg();
    // Zero shaping weights make the terminal reward exactly the verifier total.
    let no_shaping = Shaping {
        gamma: 0.99,
        lateral: 0.0,
        apogee: 0.0,
        alpha_deg: 0.0,
    };
    let mut e = RlEnv::new(cfg.clone(), 0, permissive_truth(&cfg), WEIGHTS, no_shaping).unwrap();
    let final_step = run_to_terminal(&mut e, Action::ZERO);
    assert!(final_step.done);
    assert!(e.is_done());
    assert_eq!(final_step.info.seed, 1337);
    assert_eq!(final_step.info.episode_index, 0);
    assert!(final_step.info.truth.pos_m.iter().all(|v| v.is_finite()));
    assert_eq!(final_step.info.gates.limits.mach_max, 10.0);
    assert_eq!(final_step.info.gates.rejection, None);
    assert!(final_step.info.verify_error.is_none());
    let verifier = final_step
        .info
        .verifier
        .expect("verifier scored the flight");
    assert!(verifier.check(WEIGHTS));
    assert_eq!(final_step.reward.to_bits(), verifier.total.to_bits());
    assert_eq!(e.total_reward().to_bits(), final_step.reward.to_bits());
}

#[test]
fn gate_rejection_is_reported_without_an_invented_scalar() {
    let cfg = short_cfg();
    let mut truth = permissive_truth(&cfg);
    truth.observed.mach_peak = 11.0;
    let mut e = env(cfg, truth);
    let final_step = run_to_terminal(&mut e, Action::ZERO);
    assert!(final_step.done);
    assert!(final_step.info.verifier.is_none());
    let rejection = final_step.info.gates.rejection.expect("gate named");
    assert!(matches!(rejection, GateViolation::Mach { .. }));
    assert!(final_step
        .info
        .verify_error
        .as_ref()
        .expect("named error")
        .contains("gate rejected"));
    assert!(final_step.reward.is_finite());
}

#[test]
fn out_of_box_action_never_reaches_the_simulator() {
    let cfg = short_cfg();
    let mut e = env(cfg, permissive_truth(&short_cfg()));
    e.reset().unwrap();
    assert!(e.step(Action { ay: 11.0, ax: 0.0 }).is_err());
    assert!(!e.is_done());
    // The channel is still intact: a legal action continues the episode.
    assert!(e.step(Action::ZERO).is_ok());
}

#[test]
fn stepping_after_terminal_is_loud() {
    let cfg = short_cfg();
    let mut e = env(cfg.clone(), permissive_truth(&cfg));
    run_to_terminal(&mut e, Action::ZERO);
    let err = e.step(Action::ZERO).unwrap_err();
    assert!(err.to_string().contains("step after terminal"));
}
