//! Deterministic wind field: power-law mean, discrete shear layer, 1-cosine
//! gust, and a Dryden-like correlated turbulence field.
//!
//! All field slices are generated ONCE per episode: the altitude-indexed mean
//! wind, the per-step gust contribution, and the per-step turbulence triplet.
//! The hot loop only interpolates/copies those slices. No global RNG and no
//! allocation per step. The `run.fast_field` config flag selects the lookup
//! path; `false` keeps the exact analytic profile as a reference.
//!
//! Turbulence uses three first-order shaping filters driven by a per-episode
//! `StdRng`: `x_{k+1} = a*x_k + sigma*sqrt(1-a^2)*w_k`, `a = exp(-dt*v_ref/L)`.

use std::f64::consts::PI;

use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::{
    config::{Config, WindConfig},
    math::{episode_seed, Vec3},
};

#[derive(Debug, Clone)]
pub struct Wind {
    cfg: WindConfig,
    dir: Vec3,
    gust_field: Vec<f64>,
    gust_active: Vec<bool>,
    turb_field: Vec<Vec3>,
    gust: Option<(f64, f64)>,
    v_ref: f64,
}

impl Wind {
    pub fn new(cfg: &Config, episode: u64) -> Self {
        let w = &cfg.wind;
        let mut rng = StdRng::seed_from_u64(episode_seed(cfg.run.seed, episode ^ 0x57494E44));
        let theta = w.direction_deg.to_radians();
        let dir = [theta.cos(), theta.sin(), 0.0];
        let v_ref = w.v_10.abs().max(2.0);
        let gust = if rng.gen::<f64>() < w.gust_probability {
            let t0 = cfg.run.duration_s * (0.08 + 0.55 * rng.gen::<f64>());
            Some((t0, w.gust_duration))
        } else {
            None
        };

        let steps = cfg.steps();
        let mut gust_field = Vec::with_capacity(steps);
        let mut gust_active = Vec::with_capacity(steps);
        for k in 0..steps {
            let t = k as f64 * cfg.run.dt;
            let mut g = 0.0;
            let mut on = false;
            if let Some((t0, d)) = gust {
                let u = (t - t0) / d.max(1e-3);
                if (0.0..=1.0).contains(&u) {
                    g = 0.5 * w.gust_amplitude * (1.0 - (2.0 * PI * u).cos());
                    on = true;
                }
            }
            gust_field.push(g);
            gust_active.push(on);
        }

        // Turbulence: DRAW ORDER is identical to the original streaming model.
        let scales = [1.0_f64, 0.8, 0.45];
        let mut decay = [0.0; 3];
        let mut gain = [0.0; 3];
        for i in 0..3 {
            let length = (w.turbulence_scale * (0.6 + 0.5 * scales[i])).max(1.0);
            decay[i] = v_ref / length;
            gain[i] = w.turbulence_sigma * scales[i];
        }
        let mut turb_field = Vec::with_capacity(steps);
        let mut x = [0.0; 3];
        for i in 0..3 {
            x[i] = gaussian(&mut rng) * gain[i];
        }
        for k in 0..steps {
            turb_field.push(x);
            if k + 1 < steps {
                for i in 0..3 {
                    let a = (-decay[i] * cfg.run.dt).exp();
                    let b = gain[i] * (1.0 - a * a).max(0.0).sqrt();
                    x[i] = a * x[i] + b * gaussian(&mut rng);
                }
            }
        }

        Self {
            cfg: w.clone(),
            dir,
            gust_field,
            gust_active,
            turb_field,
            gust,
            v_ref,
        }
    }

    /// Total wind at integration step `step`. The gust and turbulence slices
    /// are precomputed; the (cheap) analytic mean profile is evaluated here.
    #[inline]
    pub fn at(&self, step: usize, alt: f64, wind_factor: f64) -> Vec3 {
        let h = alt.max(0.1);
        let speed =
            mean_smooth(&self.cfg, h, wind_factor) + shear_at(&self.cfg, h) + self.gust_field[step];
        let t = self.turb_field[step];
        [self.dir[0] * speed + t[0], self.dir[1] * speed + t[1], t[2]]
    }

    #[inline]
    pub fn gust_active_step(&self, step: usize) -> bool {
        self.gust_active[step]
    }

    #[inline]
    pub fn shear_active(&self, h: f64) -> bool {
        (h - self.cfg.shear_altitude).abs() <= 2.0 * self.cfg.shear_thickness
    }

    pub fn gust_info(&self) -> Option<(f64, f64, f64)> {
        self.gust.map(|(t0, d)| (t0, d, self.cfg.gust_amplitude))
    }

    pub fn v_ref(&self) -> f64 {
        self.v_ref
    }
}

#[inline]
fn mean_smooth(w: &WindConfig, h: f64, wind_factor: f64) -> f64 {
    w.v_10 * (h / 10.0).powf(w.shear_exponent) * wind_factor.max(0.0)
}

#[inline]
fn shear_at(w: &WindConfig, h: f64) -> f64 {
    let s = 0.5 * (1.0 + ((h - w.shear_altitude) / w.shear_thickness.max(1e-3)).tanh());
    w.shear_delta * s
}

/// Box-Muller from the seeded generator; `1 - u` avoids `ln(0)`.
#[inline]
fn gaussian(rng: &mut StdRng) -> f64 {
    let u1 = 1.0 - rng.gen::<f64>();
    let u2 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}
