//! Streaming episode iterator and in-memory episode representation.

use std::path::Path;

use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;

use crate::{
    atmosphere::Atmosphere,
    config::{sample_domain, Conditioning, Config, Range},
    dynamics::{phase_at, rk4, State, Vehicle},
    error::{Result, SimError},
    math::episode_seed,
    optimizer::{ReferenceOptimizer, Target},
    physics::PhysicsFeatures,
    scenario::{scenario, Split},
    sensor::{feature, Frame, Sensor, FEATURE_NAMES},
    wind::Wind,
};

#[derive(Debug, Clone)]
pub struct EpisodeMeta {
    pub index: u64,
    pub seed: u64,
    pub mass_kg: f64,
    pub thrust_scale: f64,
    pub initial_pitch_deg: f64,
    pub initial_yaw_deg: f64,
    pub gimbal_deg: [f64; 2],
    pub gust_start_s: Option<f64>,
    pub gust_duration_s: f64,
    pub gust_amplitude: f64,
    pub v_10: f64,
    pub shear_delta: f64,
    pub scenario_id: u16,
    pub split: Split,
    /// Six vehicle numbers fed to the model as conditioning.
    pub conditioning: Conditioning,
}

#[derive(Debug, Clone)]
pub struct Episode {
    pub meta: EpisodeMeta,
    pub frames: Vec<Frame>,
    /// Number of 50 Hz sampling instants reached, even with sensors disabled.
    pub frames_generated: usize,
    pub targets: Vec<Target>,
    /// Row-major `frames x feature_count` matrix of f32 values.
    pub features: Vec<f32>,
    pub phase_counts: [usize; 6],
    pub invalid_frames: usize,
    pub nan_frames: usize,
    /// Truth telemetry from the integrated state (not the noisy baro frame).
    pub apogee_m: f64,
    pub max_speed_mps: f64,
}

impl Episode {
    pub fn feature_count(&self) -> usize {
        if self.frames.is_empty() {
            0
        } else {
            self.features.len() / self.frames.len()
        }
    }
}

pub struct Simulator {
    cfg: Config,
    atm: Atmosphere,
}

impl Simulator {
    pub fn new(cfg: Config) -> Result<Self> {
        for f in &cfg.output.features {
            if !FEATURE_NAMES.contains(&f.as_str()) {
                return Err(SimError::Config(format!("unknown output feature {f:?}")));
            }
        }
        let profile = resolve_profile(&cfg.sim.boundary_profile);
        let atm =
            Atmosphere::load_with_fast(&profile, cfg.sim.altitude_grid_m, cfg.run.fast_field)?;
        Ok(Self { cfg, atm })
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }
    pub fn atmosphere(&self) -> &Atmosphere {
        &self.atm
    }

    pub fn run_episode(&self, index: u64) -> Result<Episode> {
        let cfg = &self.cfg;
        let mut rng = StdRng::seed_from_u64(episode_seed(cfg.run.seed, index ^ 0x494E4954));
        let s_range: Range = cfg.initial.mass_kg;
        let mass = s_range.sample(rng.gen());
        let thrust_scale = cfg.initial.thrust_scale.sample(rng.gen());
        let pitch_deg = cfg.initial.pitch_deg.sample(rng.gen());
        let yaw_deg = cfg.initial.yaw_deg.sample(rng.gen());
        let gimbal_deg = [
            cfg.initial.gimbal_deg.sample(rng.gen()),
            cfg.initial.gimbal_deg.sample(rng.gen()),
        ];
        let cg_offset = cfg.initial.cg_offset_m.sample(rng.gen());

        // Tolerances of the one real vehicle. Mass/thrust/CG already come from
        // the `INIT` draw above; cd0, caliber margin and sensor noise use an
        // independent sub-seed so the legacy draw order does not move.
        let domain = sample_domain(cfg, index);
        let sc = scenario(index);
        let diameter = 2.0 * (cfg.aero.area_m2 / std::f64::consts::PI).sqrt();
        let mut vehicle = Vehicle::from_config(cfg, mass, thrust_scale, gimbal_deg);
        vehicle.aero.cd0 = domain.cd0;
        vehicle.aero.static_margin_m = domain.static_margin_cal * diameter + cg_offset;
        let thrust_peak_n = cfg
            .aero
            .thrust_profile
            .iter()
            .map(|p| p.1)
            .fold(0.0_f64, f64::max);
        let conditioning = Conditioning {
            mass_kg: mass as f32,
            area_m2: cfg.aero.area_m2 as f32,
            cd0: domain.cd0 as f32,
            static_margin_cal: domain.static_margin_cal as f32,
            thrust_peak_n: thrust_peak_n as f32,
            gyro_noise: domain.gyro_noise as f32,
        };
        let mut state = State::new(mass, pitch_deg.to_radians(), yaw_deg.to_radians());
        let wind = Wind::new(cfg, index);
        let mut sensor = Sensor::new(cfg, index);
        let optimizer = ReferenceOptimizer::new(cfg);

        let steps = cfg.steps();
        let spf = cfg.steps_per_frame();
        let nf = cfg.frames_per_episode();
        let nfeat = cfg.feature_count();
        let mut frames = Vec::with_capacity(if cfg.sim.sensors { nf } else { 0 });
        let mut frames_generated = 0;
        let mut targets = Vec::with_capacity(if cfg.sim.sensors { nf } else { 0 });
        let mut features = Vec::with_capacity(nf * nfeat);
        let mut phase_counts = [0usize; 6];
        let mut invalid_frames = 0;
        let mut nan_frames = 0;
        let mut apogee_m: f64 = 0.0;
        let mut max_speed_mps: f64 = 0.0;
        let gust_info = wind.gust_info();

        for step in 0..steps {
            let t = step as f64 * cfg.run.dt;
            let air = self.atm.at(state.pos()[2]);
            let w = wind.at(step, state.pos()[2], air.wind_factor);
            let phase = phase_at(t, &state, &vehicle);

            if step % spf == 0 {
                frames_generated += 1;
                if cfg.sim.sensors {
                    let mut frame = sensor.measure(
                        (step / spf) as u16,
                        t,
                        &state,
                        &vehicle,
                        &self.atm,
                        w,
                        phase,
                        wind.gust_active_step(step),
                        wind.shear_active(state.pos()[2]),
                    );
                    frame.physics =
                        PhysicsFeatures::from_state(&state, w, air, &vehicle, frame.accel)
                            .as_array();
                    let target = optimizer.solve(&state, w, &vehicle, phase, &frame);
                    if !frame.ai_input_valid() {
                        invalid_frames += 1;
                    }
                    if frame
                        .gyro
                        .iter()
                        .chain(frame.accel.iter())
                        .chain(frame.mag.iter())
                        .chain([&frame.height])
                        .chain([&(frame.pitot_dp as f32)])
                        .any(|v| !v.is_finite())
                    {
                        nan_frames += 1;
                    }
                    phase_counts[phase.code() as usize] += 1;
                    for name in &cfg.output.features {
                        features.push(feature(&frame, name).unwrap_or(0.0));
                    }
                    frames.push(frame);
                    targets.push(target);
                }
            }

            state = rk4(t, &state, cfg.run.dt, w, &self.atm, &vehicle);
            apogee_m = apogee_m.max(state.pos[2]);
            max_speed_mps = max_speed_mps.max(crate::math::norm(state.vel));
            if !state.is_finite() {
                return Err(SimError::NonFinite {
                    step,
                    what: "state",
                });
            }
        }

        let meta = EpisodeMeta {
            index,
            seed: cfg.run.seed,
            mass_kg: mass,
            thrust_scale,
            initial_pitch_deg: pitch_deg,
            initial_yaw_deg: yaw_deg,
            gimbal_deg,
            gust_start_s: gust_info.map(|g| g.0),
            gust_duration_s: gust_info.map(|g| g.1).unwrap_or(0.0),
            gust_amplitude: gust_info.map(|g| g.2).unwrap_or(0.0),
            v_10: wind.v_10(),
            shear_delta: wind.shear_delta(),
            scenario_id: sc.id,
            split: sc.split,
            conditioning,
        };
        Ok(Episode {
            meta,
            frames,
            frames_generated,
            targets,
            features,
            phase_counts,
            invalid_frames,
            nan_frames,
            apogee_m,
            max_speed_mps,
        })
    }

    pub fn episodes(&self, count: usize) -> Episodes<'_> {
        Episodes {
            sim: self,
            next: 0,
            remaining: count,
        }
    }

    /// Batched generation for validation sets. Episodes are independent
    /// functions of (seed, index), so parallelism cannot change the result.
    pub fn run_episodes_parallel(&self, count: usize) -> Vec<Result<Episode>> {
        (0..count as u64)
            .into_par_iter()
            .map(|i| self.run_episode(i))
            .collect()
    }
}

/// Resolve a config-relative profile against the CWD, then against the
/// workspace `config/` directory baked into the binary at build time.
fn resolve_profile(p: &Path) -> std::path::PathBuf {
    if p.is_absolute() || p.exists() {
        return p.to_path_buf();
    }
    let embedded = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../config")
        .join(p.file_name().unwrap_or_default());
    if embedded.exists() {
        embedded
    } else {
        p.to_path_buf()
    }
}

pub struct Episodes<'a> {
    sim: &'a Simulator,
    next: u64,
    remaining: usize,
}

impl<'a> Iterator for Episodes<'a> {
    type Item = Result<Episode>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        let i = self.next;
        self.next += 1;
        Some(self.sim.run_episode(i))
    }
}
