//! Streaming episode iterator and in-memory episode representation.

use std::path::Path;

use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;

use crate::{
    aero::{count_placeholder, AeroCard},
    atmosphere::Atmosphere,
    config::{sample_domain, AeroSource, Conditioning, Config, Range},
    control::validate_command,
    dynamics::{phase_at, rk4, State, Vehicle},
    error::{Result, SimError},
    math::{episode_seed, mat_t_vec, norm, quat_to_matrix, sub},
    optimizer::{ReferenceOptimizer, Target},
    physics::PhysicsFeatures,
    scenario::{scenario, Split},
    sensor::{feature, Frame, Sensor, FEATURE_NAMES},
    trajectory::TrajectoryTick,
    validation::TruthRecord,
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
    /// Ground truth at the last 50 Hz tick: true wind, gust, shear layer and
    /// the full 6-DoF state. Written to truth-extended v2 metadata.
    pub truth: Option<TruthRecord>,
    /// Flown truth at every 50 Hz tick: world-frame x/y/z and vx/vy/vz.
    /// Written to the trajectory sidecar; apogee and lateral drift are
    /// measured from this, not from a surrogate.
    pub trajectory: Vec<TrajectoryTick>,
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
    card: Option<AeroCard>,
}

impl Simulator {
    pub fn new(cfg: Config) -> Result<Self> {
        for f in &cfg.output.features {
            if !FEATURE_NAMES.contains(&f.as_str()) {
                return Err(SimError::Config(format!("unknown output feature {f:?}")));
            }
        }
        let profile = resolve_config_path(&cfg.sim.boundary_profile);
        let atm =
            Atmosphere::load_with_fast(&profile, cfg.sim.altitude_grid_m, cfg.run.fast_field)?;
        let card = match cfg.aero.source {
            AeroSource::Placeholder => {
                let n = count_placeholder();
                eprintln!(
                    "[potok-sim] aero=placeholder count={n} (COUNTED: set [aero] source=\"card\" to use a coefficient card)"
                );
                None
            }
            AeroSource::Card => {
                let path = cfg.aero.card_path.as_ref().ok_or_else(|| {
                    SimError::Config("aero.source=card needs aero.card_path".into())
                })?;
                Some(AeroCard::from_file(&resolve_config_path(path))?)
            }
        };
        Ok(Self { cfg, atm, card })
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }
    pub fn atmosphere(&self) -> &Atmosphere {
        &self.atm
    }

    /// Legacy path: the controller is absent, so no command reaches the
    /// dynamics and the trajectory is bit-for-bit the historical one.
    pub fn run_episode(&self, index: u64) -> Result<Episode> {
        self.run_episode_impl(index, None)
    }

    /// Drive the vehicle with a per-frame command. `control` is called once
    /// per 50 Hz tick with the frame just measured; its two degrees are held
    /// for the `steps_per_frame` integration steps up to the next tick.
    /// Commands outside +/- `MAX_CONTROL_DEG` (or non-finite) are LOUD errors.
    pub fn run_episode_with_control(
        &self,
        index: u64,
        control: &dyn Fn(&Frame) -> (f64, f64),
    ) -> Result<Episode> {
        self.run_episode_impl(index, Some(control))
    }

    fn run_episode_impl(
        &self,
        index: u64,
        control: Option<&dyn Fn(&Frame) -> (f64, f64)>,
    ) -> Result<Episode> {
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
        if let Some(card) = &self.card {
            vehicle.aero_card = Some(card.clone());
        }
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
        let mut truth: Option<TruthRecord> = None;
        let mut trajectory = Vec::with_capacity(nf);
        // Sampled at the frame tick, held across the step's RK4 sub-steps.
        let mut command_deg = [0.0_f64; 2];
        let gust_info = wind.gust_info();

        for step in 0..steps {
            let t = step as f64 * cfg.run.dt;
            let air = self.atm.at(state.pos()[2]);
            let w = wind.at(step, state.pos()[2], air.wind_factor);
            let phase = phase_at(t, &state, &vehicle);

            // Tick coefficients for the exported physics features. The
            // integrator queries the card at each RK4 sub-stage; both queries
            // are fallible, so out-of-grid is LOUD, never clamped.
            if vehicle.aero_card.is_some() {
                let v_rel = sub(state.vel(), w);
                let mach = norm(v_rel) / air.speed_of_sound.max(1.0);
                let vb = mat_t_vec(quat_to_matrix(state.quat()), v_rel);
                let alpha_deg = state.omega()[2].atan2(vb[0]).to_degrees();
                let coeffs = vehicle.aero_card.as_ref().unwrap().coeffs(
                    mach,
                    alpha_deg,
                    vehicle.aero.cd0,
                )?;
                vehicle.card_coeffs = coeffs;
            }

            if step % spf == 0 {
                truth = Some(TruthRecord {
                    wind: [w[0] as f32, w[1] as f32, w[2] as f32],
                    gust_start_s: gust_info.map(|g| g.0).unwrap_or(-1.0) as f32,
                    gust_amplitude: gust_info.map(|g| g.2).unwrap_or(0.0) as f32,
                    shear_altitude_m: wind.shear_altitude() as f32,
                    shear_delta: wind.shear_delta() as f32,
                    shear_thickness_m: wind.shear_thickness() as f32,
                    pos: cast3(state.pos()),
                    vel: cast3(state.vel()),
                    quat: [
                        state.quat()[0] as f32,
                        state.quat()[1] as f32,
                        state.quat()[2] as f32,
                        state.quat()[3] as f32,
                    ],
                    omega: cast3(state.omega()),
                    mass_kg: state.mass() as f32,
                });
                trajectory.push(TrajectoryTick {
                    pos: cast3(state.pos()),
                    vel: cast3(state.vel()),
                });

                // The controller sees the frame at this tick. With sensors off
                // a temporary frame is still built for it, but nothing is
                // exported and the sensor RNG has no observable consumer.
                let mut control_frame = None;
                if cfg.sim.sensors || control.is_some() {
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
                    if cfg.sim.sensors {
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
                    } else {
                        control_frame = Some(frame);
                    }
                    if let Some(control) = control {
                        let seen = control_frame
                            .as_ref()
                            .or_else(|| frames.last())
                            .expect("frame at tick");
                        let (roll, pitch) = control(seen);
                        let (roll, pitch) = validate_command(roll, pitch)?;
                        command_deg = [roll, pitch];
                    }
                }

                frames_generated += 1;
            }

            vehicle.control_deg = command_deg;
            state = rk4(t, &state, cfg.run.dt, w, &self.atm, &vehicle);
            if let Some(f) = vehicle.card_fault.take() {
                return Err(SimError::AeroOutOfGrid {
                    axis: f.axis.to_string(),
                    value: f.value,
                    lo: f.lo,
                    hi: f.hi,
                });
            }
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
            truth,
            trajectory,
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

fn cast3(v: [f64; 3]) -> [f32; 3] {
    [v[0] as f32, v[1] as f32, v[2] as f32]
}

/// Resolve a config-relative path against the CWD, then against the
/// workspace `config/` directory baked into the binary at build time.
fn resolve_config_path(p: &Path) -> std::path::PathBuf {
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
