// SPDX-License-Identifier: GPL-3.0-or-later
//! Sensor realism gates, driven through `Sensor::measure` with a controlled
//! static vehicle. Every randomized case is pinned to a fixed seed and no
//! source module is edited.
//!
//! Coverage: Allan angle/velocity random walk, bias drift, per-channel
//! quantization, whole-frame latency, saturation, fault-flag gating (including
//! a stuck-high spike fault), and tilt-dependent gravity projection.

use std::f64::consts::FRAC_PI_2;

use potok_sim::{
    atmosphere::Atmosphere,
    config::{Config, Range, SensorConfig},
    dynamics::{Phase, State, Vehicle},
    math::{mat_t_vec, quat_to_matrix},
    sensor::{
        gating, wire_flags, Frame, Sensor, FLAG_AI_READY, FLAG_BARO_OK, FLAG_FAULT, FLAG_IMU_OK,
        FLAG_MOTOR_ARMED, FLAG_PITOT_OK,
    },
};

const G: f64 = 9.80665;
/// Sensor sample period: 1 / frame_rate_hz at the shipped 50 Hz frame rate.
/// `Sensor::measure` is called once per frame, not once per 2 ms RK4 step.
const SAMPLE_DT: f64 = 0.02;

struct Rig {
    cfg: Config,
    atm: Atmosphere,
    vehicle: Vehicle,
}

/// Zero noise, zero bias, instant lag, no delay, no limits, no spikes. Domain
/// randomization overrides the raw `gyro_noise`/`accel_noise` in `Sensor::new`,
/// so the *ranges* must be zeroed too, or the rig is not actually quiet.
fn quiet_config(seed: u64) -> Config {
    let mut cfg = Config::default();
    cfg.run.seed = seed;
    let s = &mut cfg.sensor;
    s.gyro_noise_range = Range(0.0, 0.0);
    s.accel_noise_range = Range(0.0, 0.0);
    s.gyro_noise = 0.0;
    s.accel_noise = 0.0;
    s.mag_noise = 0.0;
    s.baro_noise = 0.0;
    s.pitot_noise = 0.0;
    s.gyro_bias = [0.0; 3];
    s.accel_bias = [0.0; 3];
    s.mag_bias = [0.0; 3];
    s.baro_bias = 0.0;
    s.pitot_bias = 0.0;
    s.quant_step = 0.0;
    s.gyro_quant_step = 0.0;
    s.accel_quant_step = 0.0;
    s.lag_tau = 1e-6; // exp(-dt/tau) underflows to 0: raw sample each frame
    s.latency_steps = 0;
    s.gyro_sat = 1e6;
    s.accel_sat = 1e6;
    s.pitot_sat = 1e9;
    s.mag_sat = 0.0;
    s.baro_sat = 0.0;
    s.spike_probability = 0.0;
    s.spike_amplitude = 0.0;
    cfg
}

fn rig_from(cfg: Config) -> Rig {
    let profile =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../config/boundary_profile.csv");
    let atm =
        Atmosphere::load_with_fast(&profile, cfg.sim.altitude_grid_m, cfg.run.fast_field).unwrap();
    // Zero thrust keeps the rail constraint active at the origin: the
    // accelerometer then reads the pure gravity projection, with no thrust or
    // aerodynamic force mixed in.
    let vehicle = Vehicle::from_config(&cfg, 11.0, 0.0, [0.0, 0.0]);
    Rig { cfg, atm, vehicle }
}

fn rig(seed: u64, setup: impl FnOnce(&mut SensorConfig)) -> Rig {
    let mut cfg = quiet_config(seed);
    setup(&mut cfg.sensor);
    rig_from(cfg)
}

/// Production sensor config (noise on), for seed-determinism checks.
fn noisy_rig(seed: u64) -> Rig {
    let mut cfg = Config::default();
    cfg.run.seed = seed;
    rig_from(cfg)
}

impl Rig {
    fn sensor(&self, episode: u64) -> Sensor {
        Sensor::new(&self.cfg, episode)
    }

    fn frame(&self, sensor: &mut Sensor, k: usize, st: &State, phase: Phase) -> Frame {
        sensor.measure(
            k as u16,
            k as f64 * SAMPLE_DT,
            st,
            &self.vehicle,
            &self.atm,
            [0.0; 3],
            phase,
            false,
            false,
        )
    }
}

/// Static on-rail state; `omega`/`quat`/`pos` are overridden per test.
fn state() -> State {
    State::new(11.0, 0.0, 0.0)
}

/// Rotation about world +X (axis-angle), chosen so the gravity projection has
/// simple analytic components: body-y = g*sin(t), body-z = g*cos(t).
fn tilt_x(angle: f64) -> [f64; 4] {
    [cos_half(angle), sin_half(angle), 0.0, 0.0]
}

fn tilt_y(angle: f64) -> [f64; 4] {
    [cos_half(angle), 0.0, sin_half(angle), 0.0]
}

fn cos_half(a: f64) -> f64 {
    (a * 0.5).cos()
}
fn sin_half(a: f64) -> f64 {
    (a * 0.5).sin()
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn stddev(v: &[f64]) -> f64 {
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
}

fn multiples_of(v: f64, step: f64) -> bool {
    let q = v / step;
    (q - q.round()).abs() < 1e-4
}

// ---------------------------------------------------------------------------
// Deterministic seeds
// ---------------------------------------------------------------------------

#[test]
fn same_seed_and_episode_reproduce_every_frame_byte() {
    let noisy = |r: &Rig| -> Vec<Frame> {
        let mut s = r.sensor(0);
        let st = state();
        (0..24)
            .map(|k| r.frame(&mut s, k, &st, Phase::Coast))
            .collect()
    };
    let fa = noisy(&noisy_rig(4242));
    let fb = noisy(&noisy_rig(4242));
    assert_eq!(fa.len(), 24);
    for (k, (x, y)) in fa.iter().zip(fb.iter()).enumerate() {
        assert_eq!(x, y, "frame {k}");
        assert_eq!(x.to_bytes(), y.to_bytes(), "wire bytes frame {k}");
        assert_eq!(x.compute_crc(), y.compute_crc());
    }

    // Seed and episode both feed the sensor sub-seed; both must move output.
    let other = noisy(&noisy_rig(4243));
    assert_ne!(
        fa[0].gyro, other[0].gyro,
        "seed must change noise realization"
    );

    let biased = rig(4242, |s| s.gyro_bias = [0.01, -0.02, 0.03]);
    let st = state();
    let mut s0 = biased.sensor(0);
    let mut s1 = biased.sensor(1);
    let f0 = biased.frame(&mut s0, 0, &st, Phase::Coast);
    let f1 = biased.frame(&mut s1, 0, &st, Phase::Coast);
    assert_ne!(f0.gyro, f1.gyro, "episode must change bias/noise scales");
}

// ---------------------------------------------------------------------------
// Allan noise: angle/velocity random walk and bias stability
// ---------------------------------------------------------------------------

#[test]
fn allan_arw_sets_the_discrete_noise_std() {
    // arw / sqrt(sample_dt) is the per-sample standard deviation for a noise
    // density quoted per root-Hz. The rig's lag is instant, so the measurement
    // is the raw draw; quant is off. Using the 2 ms RK4 step here instead of
    // the 20 ms frame period would inflate the spread by sqrt(10) = 3.16x.
    let r = rig(11, |s| {
        s.gyro_arw = 0.02;
        s.accel_arw = 0.08;
    });
    let mut sensor = r.sensor(0);
    let st = state();
    let (mut g, mut a) = (Vec::new(), Vec::new());
    for k in 0..600 {
        let f = r.frame(&mut sensor, k, &st, Phase::Coast);
        g.push(f.gyro[0] as f64);
        a.push(f.accel[0] as f64);
    }
    let gyro_want = 0.02 / SAMPLE_DT.sqrt();
    let accel_want = 0.08 / SAMPLE_DT.sqrt();
    assert!((stddev(&g) / gyro_want - 1.0).abs() < 0.25, "gyro std");
    assert!((stddev(&a) / accel_want - 1.0).abs() < 0.25, "accel std");
    assert!(mean(&g).abs() < 0.15 * gyro_want, "gyro noise has a mean");

    // Doubling the density doubles the spread; this rules out an ARW knob
    // that only changes the constant bias or the RNG draw order.
    let r2 = rig(11, |s| s.gyro_arw = 0.04);
    let mut s2 = r2.sensor(0);
    let g2: Vec<f64> = (0..600)
        .map(|k| r2.frame(&mut s2, k, &st, Phase::Coast).gyro[0] as f64)
        .collect();
    let ratio = stddev(&g2) / stddev(&g);
    assert!((1.6..2.4).contains(&ratio), "arw scaling ratio {ratio}");
}

#[test]
fn allan_bias_stability_drifts_while_zero_stability_is_exactly_flat() {
    let st = state();

    let flat = rig(11, |_| {});
    let mut fs = flat.sensor(0);
    let flat_g: Vec<f64> = (0..600)
        .map(|k| flat.frame(&mut fs, k, &st, Phase::Coast).gyro[0] as f64)
        .collect();
    assert!(flat_g.iter().all(|v| *v == 0.0), "quiet gyro is not flat");

    let walk = rig(11, |s| {
        s.gyro_bias_stability = 0.05;
        s.accel_bias_stability = 0.2;
    });
    let mut ws = walk.sensor(0);
    let (mut g, mut a) = (Vec::new(), Vec::new());
    for k in 0..600 {
        let f = walk.frame(&mut ws, k, &st, Phase::Coast);
        g.push(f.gyro[0] as f64);
        a.push(f.accel[0] as f64);
    }
    assert!(
        stddev(&g) > 0.01,
        "gyro bias walk too small: {}",
        stddev(&g)
    );
    assert!(
        stddev(&a) > 0.02,
        "accel bias walk too small: {}",
        stddev(&a)
    );

    let head = mean(&g[..100]);
    let tail = mean(&g[500..]);
    assert!((tail - head).abs() > 1e-6, "bias walk never drifted");
}

// ---------------------------------------------------------------------------
// First-order lag time base
// ---------------------------------------------------------------------------

#[test]
fn lag_filter_integrates_on_the_frame_period_not_the_rk4_step() {
    // A unit step into a first-order lag gives y_0 = 1 - exp(-T/tau). With
    // tau = 0.120 s sampled at 50 Hz (T = 0.02 s) that is 0.153518, i.e. a
    // physical time constant of exactly tau. Using the 2 ms integration dt
    // instead produced y_0 = 0.016529, an effective tau of 1.2 s -- 10x the
    // configured constant at the same call rate.
    const TAU: f64 = 0.120;
    let first_sample = |frame_rate_hz: f64| -> f64 {
        let mut cfg = quiet_config(0);
        cfg.run.frame_rate_hz = frame_rate_hz;
        cfg.sensor.lag_tau = TAU;
        let r = rig_from(cfg);
        let mut sensor = r.sensor(0);
        let mut st = state();
        st.omega = [1.0, 0.0, 0.0];
        r.frame(&mut sensor, 0, &st, Phase::Coast).gyro[0] as f64
    };

    let y50 = first_sample(50.0);
    let want50 = 1.0 - (-SAMPLE_DT / TAU).exp();
    assert!(
        (y50 - want50).abs() < 1e-6,
        "gyro lag at 50 Hz: {y50} vs {want50}"
    );
    let tau_hat = -SAMPLE_DT / (1.0 - y50).ln();
    assert!((tau_hat - TAU).abs() < 1e-3, "effective tau {tau_hat}");

    // At 100 Hz the sample period halves; the lag coefficient must follow
    // frame_rate_hz, not a hardcoded 0.02 and not the fixed 0.002 RK4 dt.
    let y100 = first_sample(100.0);
    let want100 = 1.0 - (-(SAMPLE_DT / 2.0) / TAU).exp();
    assert!(
        (y100 - want100).abs() < 1e-6,
        "gyro lag at 100 Hz: {y100} vs {want100}"
    );
    assert!(y100 < y50, "faster sampling must lag less per sample");
}

// ---------------------------------------------------------------------------
// Quantization
// ---------------------------------------------------------------------------

#[test]
fn quantization_uses_per_channel_steps_and_zero_step_disables_it() {
    let mut st = state();
    st.omega = [0.31, -0.17, 0.09];

    let r = rig(7, |s| {
        s.gyro_bias = [0.0137, -0.0073, 0.0211];
        s.baro_bias = 0.1234;
        s.quant_step = 0.05;
        s.gyro_quant_step = 0.01;
        s.accel_quant_step = 0.2;
    });
    let mut sensor = r.sensor(0);
    let mut frames = Vec::new();
    for k in 0..8 {
        let f = r.frame(&mut sensor, k, &st, Phase::Coast);
        for v in f.gyro {
            assert!(multiples_of(v as f64, 0.01), "gyro {v} not on 0.01 grid");
        }
        for v in f.accel {
            assert!(multiples_of(v as f64, 0.2), "accel {v} not on 0.2 grid");
        }
        for v in f.mag {
            assert!(multiples_of(v as f64, 0.05), "mag {v} not on 0.05 grid");
        }
        assert!(multiples_of(f.height as f64, 0.05), "height {}", f.height);
        frames.push(f);
    }
    assert!(
        frames
            .iter()
            .any(|f| f.gyro.iter().any(|v| !multiples_of(*v as f64, 0.2))),
        "gyro must use its own step, not quant_step"
    );

    // gyro_quant_step = 0 falls back to quant_step (0.05), changing the output.
    let fallback = rig(7, |s| {
        s.gyro_bias = [0.0137, -0.0073, 0.0211];
        s.baro_bias = 0.1234;
        s.quant_step = 0.05;
        s.accel_quant_step = 0.2;
    });
    let mut fs = fallback.sensor(0);
    let ff = fallback.frame(&mut fs, 0, &st, Phase::Coast);
    assert_ne!(frames[0].gyro, ff.gyro, "gyro step fallback did not apply");

    // step = 0 everywhere leaves the raw value (no rounding grid at all).
    let raw = rig(7, |s| s.gyro_bias = [0.0137, -0.0073, 0.0211]);
    let mut rs = raw.sensor(0);
    let rf = raw.frame(&mut rs, 0, &st, Phase::Coast);
    assert!(!multiples_of(rf.gyro[0] as f64, 0.05), "quant not disabled");
}

// ---------------------------------------------------------------------------
// Whole-frame delay
// ---------------------------------------------------------------------------

#[test]
fn latency_steps_shifts_gyro_and_mag_by_exactly_the_configured_frames() {
    const L: usize = 3;
    let r = rig(5, |s| s.latency_steps = L);
    let mut sensor = r.sensor(0);
    let mut st = state();
    let before = [0.5_f32, 0.0, 0.0];
    let after = [2.5_f32, 0.0, 0.0];
    let q0 = st.quat;
    let q1 = tilt_x(FRAC_PI_2);

    for k in 0..14 {
        let omega = if k < 4 { before } else { after };
        st.omega = omega.map(|v| v as f64);
        st.quat = if k < 4 { q0 } else { q1 };
        let f = r.frame(&mut sensor, k, &st, Phase::Coast);
        // Ring buffer emits sample max(0, k - L); the change is visible at
        // k = 4 + L, not at k = 4.
        let (want_gyro, want_quat) = if k < 4 + L { (before, q0) } else { (after, q1) };
        assert_eq!(f.gyro, want_gyro, "gyro delay at frame {k}");
        let want_mag = mat_t_vec(quat_to_matrix(want_quat), r.cfg.sensor.mag_world);
        for (i, (got, want)) in f.mag.iter().zip(want_mag.iter()).enumerate() {
            assert!(
                (*got as f64 - *want).abs() < 1e-4,
                "mag delay at frame {k}, axis {i}: {} vs {}",
                got,
                want
            );
        }
    }

    // latency 0 returns the current sample, with no warm-up plateau.
    let immediate = rig(5, |_| {});
    let mut s0 = immediate.sensor(0);
    let mut st = state();
    st.omega = before.map(|v| v as f64);
    let a = immediate.frame(&mut s0, 0, &st, Phase::Coast);
    st.omega = after.map(|v| v as f64);
    let b = immediate.frame(&mut s0, 1, &st, Phase::Coast);
    assert_eq!(a.gyro, before);
    assert_eq!(b.gyro, after);
}

// ---------------------------------------------------------------------------
// Saturation and fault flags
// ---------------------------------------------------------------------------

#[test]
fn saturation_clamps_each_channel_and_raises_fault() {
    let r = rig(13, |s| {
        s.gyro_sat = 0.5;
        s.accel_sat = 2.0;
        s.mag_sat = 0.1;
        s.baro_bias = 5.0;
        s.baro_sat = 0.25;
        s.pitot_sat = 100.0;
    });
    let mut sensor = r.sensor(0);
    let mut st = state();
    st.omega = [3.0, -3.0, 0.0];
    st.vel = [100.0, 0.0, 0.0]; // pitot dynamic pressure above pitot_sat
    let f = r.frame(&mut sensor, 0, &st, Phase::Coast);

    assert!(f.saturated, "saturation not reported");
    assert!(f.gyro.iter().all(|v| v.abs() <= 0.5 + 1e-6), "{:?}", f.gyro);
    assert!(
        f.accel.iter().all(|v| v.abs() <= 2.0 + 1e-6),
        "{:?}",
        f.accel
    );
    assert!(f.mag.iter().all(|v| v.abs() <= 0.1 + 1e-6), "{:?}", f.mag);
    assert!(f.height.abs() <= 0.25 + 1e-6, "{}", f.height);
    // pitot_sat raises the flag but does not clamp the wire i16; the wire
    // value is clamped to the i16 range instead.
    assert!((f.pitot_dp as f64) > 100.0, "pitot {}", f.pitot_dp);
    assert!((f.pitot_dp as f64) <= i16::MAX as f64);
    assert_ne!(f.flags & FLAG_FAULT, 0, "fault bit low on saturation");
    assert_eq!(f.flags & FLAG_AI_READY, 0, "AI_READY high on saturation");
    assert!(!f.ai_input_valid());

    // Faults are per-frame, not sticky: the same static state on a rig with
    // nominal limits produces a clean frame.
    let nominal = rig(13, |_| {});
    let mut ns = nominal.sensor(0);
    let clean = nominal.frame(&mut ns, 1, &state(), Phase::Coast);
    assert!(!clean.saturated);
    assert_eq!(clean.flags & FLAG_FAULT, 0);
}

#[test]
fn wire_flags_and_gating_cover_healthy_and_fault_cases() {
    let healthy = wire_flags(Phase::Coast, true, true, false);
    let want = FLAG_IMU_OK | FLAG_BARO_OK | FLAG_PITOT_OK | FLAG_AI_READY;
    assert_eq!(healthy, want);
    assert_eq!(healthy & FLAG_FAULT, 0, "FAULT bit low when healthy");

    let sensor_bad = wire_flags(Phase::Coast, false, true, false);
    assert_eq!(sensor_bad & (FLAG_IMU_OK | FLAG_BARO_OK | FLAG_AI_READY), 0);
    assert_ne!(sensor_bad & FLAG_FAULT, 0);

    let no_pitot = wire_flags(Phase::Coast, true, false, false);
    assert_eq!(no_pitot & FLAG_PITOT_OK, 0);
    assert_eq!(no_pitot & FLAG_AI_READY, 0);
    assert_eq!(no_pitot & FLAG_FAULT, 0);

    let sat = wire_flags(Phase::Coast, true, true, true);
    assert_ne!(sat & FLAG_FAULT, 0);
    assert_eq!(sat & FLAG_AI_READY, 0);

    let boost = wire_flags(Phase::Boost, true, true, false);
    assert_ne!(boost & FLAG_MOTOR_ARMED, 0);
    assert_eq!(boost & FLAG_AI_READY, 0);
    assert_eq!(wire_flags(Phase::Pad, true, true, false) & FLAG_AI_READY, 0);

    for phase in [
        Phase::Pad,
        Phase::Boost,
        Phase::Coast,
        Phase::Apogee,
        Phase::Descent,
        Phase::Landed,
    ] {
        for ok in [false, true] {
            for pitot in [false, true] {
                for saturated in [false, true] {
                    let flags = wire_flags(phase, ok, pitot, saturated);
                    let ai = flags & FLAG_AI_READY != 0;
                    assert_eq!(ai, gating(phase, ok, pitot, saturated));
                    assert_eq!(ai, ok && pitot && !saturated && phase.ai_allowed());
                }
            }
        }
    }
}

#[test]
fn seeded_spikes_latch_fault_high_while_they_keep_firing() {
    let st = state();
    let stuck = rig(21, |s| {
        s.spike_probability = 1.0;
        s.spike_amplitude = 2.0;
    });
    let mut sensor = stuck.sensor(0);
    for k in 0..20 {
        let f = stuck.frame(&mut sensor, k, &st, Phase::Coast);
        assert!(f.spike, "spike did not fire at frame {k}");
        assert!(f.saturated, "spike frame must be saturated");
        assert_ne!(f.flags & FLAG_FAULT, 0, "fault not high at frame {k}");
        assert_eq!(f.flags & FLAG_AI_READY, 0);
        assert!(!f.ai_input_valid());
        let moved: Vec<f64> = f
            .gyro
            .iter()
            .zip([0.0_f32; 3].iter())
            .map(|(a, b)| (a - b) as f64)
            .filter(|d| d.abs() > 1e-6)
            .collect();
        assert_eq!(moved.len(), 1, "exactly one gyro axis is spiked");
        assert!((moved[0].abs() - 2.0).abs() < 1e-4, "spike amplitude");
    }

    let none = rig(21, |_| {});
    let mut ns = none.sensor(0);
    for k in 0..20 {
        let f = none.frame(&mut ns, k, &st, Phase::Coast);
        assert!(!f.spike);
        assert_eq!(f.flags & FLAG_FAULT, 0);
    }

    // Intermittent faults are not sticky: a clean frame after a spike has the
    // fault bit low again.
    let intermittent = rig(29, |s| {
        s.spike_probability = 0.5;
        s.spike_amplitude = 1.0;
    });
    let mut is = intermittent.sensor(0);
    let mut saw_spike = false;
    let mut saw_clean_after = false;
    for k in 0..120 {
        let f = intermittent.frame(&mut is, k, &st, Phase::Coast);
        if f.spike {
            saw_spike = true;
            assert_ne!(f.flags & FLAG_FAULT, 0);
        } else if saw_spike {
            saw_clean_after = true;
            assert_eq!(f.flags & FLAG_FAULT, 0, "fault stuck high after a spike");
        }
    }
    assert!(
        saw_spike && saw_clean_after,
        "intermittent injection did not vary"
    );
}

// ---------------------------------------------------------------------------
// Tilt-dependent gravity projection
// ---------------------------------------------------------------------------

#[test]
fn accelerometer_reads_tilted_gravity_and_zero_in_free_fall() {
    let r = rig(3, |_| {});
    let mut sensor = r.sensor(0);
    let mut st = state();

    let check = |st: &State, s: &mut Sensor, r: &Rig, want: [f64; 3]| {
        let f = r.frame(s, 0, st, Phase::Coast);
        for (i, (got, want)) in f.accel.iter().zip(want.iter()).enumerate() {
            assert!(
                (*got as f64 - *want).abs() < 1e-3,
                "accel[{i}] {} vs {}",
                got,
                want
            );
        }
        f
    };

    // Upright on the rail: the accelerometer reads 1 g on body +Z.
    check(&st, &mut sensor, &r, [0.0, 0.0, G]);

    // Tilt about body/world X: gravity leaks into body Y with sin(theta).
    st.quat = tilt_x(30.0_f64.to_radians());
    check(
        &st,
        &mut sensor,
        &r,
        [0.0, G * 0.5, G * 30.0_f64.to_radians().cos()],
    );
    st.quat = tilt_x(-30.0_f64.to_radians());
    check(
        &st,
        &mut sensor,
        &r,
        [0.0, -G * 0.5, G * 30.0_f64.to_radians().cos()],
    );

    // Tilt about Y by 90 degrees puts the whole gravity vector on body -X
    // (R^T maps world +Z there for a positive Y rotation).
    st.quat = tilt_y(FRAC_PI_2);
    check(&st, &mut sensor, &r, [-G, 0.0, 0.0]);

    // Magnitude is invariant under tilt.
    let f = r.frame(&mut sensor, 0, &st, Phase::Coast);
    let mag =
        (f.accel[0] as f64).powi(2) + (f.accel[1] as f64).powi(2) + (f.accel[2] as f64).powi(2);
    assert!((mag.sqrt() - G).abs() < 1e-3, "|a| = {}", mag.sqrt());

    // In free fall there is no specific force, whatever the tilt.
    st.pos[2] = 100.0;
    let ff = r.frame(&mut sensor, 0, &st, Phase::Coast);
    assert!(
        ff.accel.iter().all(|v| v.abs() < 1e-3),
        "free-fall accel {:?}",
        ff.accel
    );

    // The magnetometer sees the same body-frame rotation of a world vector.
    let mw = r.cfg.sensor.mag_world;
    let theta = 30.0_f64.to_radians();
    st.quat = tilt_x(theta);
    st.pos[2] = 0.0;
    let mf = r.frame(&mut sensor, 0, &st, Phase::Coast);
    let want = [
        mw[0],
        mw[1] * theta.cos() + mw[2] * theta.sin(),
        -mw[1] * theta.sin() + mw[2] * theta.cos(),
    ];
    for (i, (got, want)) in mf.mag.iter().zip(want.iter()).enumerate() {
        assert!(
            (*got as f64 - *want).abs() < 1e-4,
            "mag[{i}] {} vs {}",
            got,
            want
        );
    }
}
