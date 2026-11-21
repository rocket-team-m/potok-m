//! US Standard Atmosphere (1976) for the lower 20 km, plus a resampled
//! boundary profile used to perturb temperature and mean wind with altitude.
//!
//! `fast_field = true` uses a tabulated/interpolated atmosphere (built once
//! per simulator); `false` evaluates the analytic equations at every call.
//! Both are covered by an equivalence test.

use std::path::Path;

use ndarray::Array2;

use crate::error::{Result, SimError};

const G: f64 = 9.80665;
const R: f64 = 287.05287;
const GAMMA: f64 = 1.4;

#[derive(Debug, Clone)]
pub struct Atmosphere {
    /// Rows: altitude, temperature offset [K], wind factor [-].
    profile: Array2<f64>,
    /// Rows: altitude, temperature, pressure, density, speed of sound, wind factor.
    field: Array2<f64>,
    step: f64,
    fast: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct Air {
    pub temperature: f64,
    pub pressure: f64,
    pub density: f64,
    pub speed_of_sound: f64,
    pub wind_factor: f64,
}

impl Atmosphere {
    pub fn load(path: &Path, grid_m: f64) -> Result<Self> {
        Self::load_with_fast(path, grid_m, false)
    }

    pub fn load_with_fast(path: &Path, grid_m: f64, fast: bool) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let mut rows: Vec<(f64, f64, f64)> = Vec::new();
        for (i, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if i == 0 && line.starts_with("altitude_m") {
                continue;
            }
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() != 3 {
                return Err(SimError::Config(format!(
                    "{}:{} expected altitude,temp_offset,wind_factor",
                    path.display(),
                    i + 1
                )));
            }
            let mut p = [0.0; 3];
            for (j, s) in parts.iter().enumerate() {
                p[j] = s.trim().parse::<f64>().map_err(|_| {
                    SimError::Config(format!("{}:{} bad number {:?}", path.display(), i + 1, s))
                })?;
            }
            rows.push((p[0], p[1], p[2]));
        }
        rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        if rows.len() < 2 {
            return Err(SimError::Config(
                "boundary profile needs at least two rows".into(),
            ));
        }
        let hi = rows.last().unwrap().0;
        let n = (hi / grid_m).ceil() as usize + 1;
        let step = grid_m;
        let mut profile = Array2::<f64>::zeros((n, 3));
        for k in 0..n {
            let h = k as f64 * grid_m;
            let mut j = 1;
            while j < rows.len() - 1 && rows[j].0 < h {
                j += 1;
            }
            let (h0, t0, w0) = rows[j - 1];
            let (h1, t1, w1) = rows[j];
            let f = if h <= h0 {
                0.0
            } else if h >= h1 {
                1.0
            } else {
                (h - h0) / (h1 - h0)
            };
            profile[[k, 0]] = h;
            profile[[k, 1]] = t0 + f * (t1 - t0);
            profile[[k, 2]] = w0 + f * (w1 - w0);
        }
        let mut field = Array2::<f64>::zeros((n, 6));
        for k in 0..n {
            let h = k as f64 * grid_m;
            let a = analytic_at(h, profile[[k, 1]], profile[[k, 2]]);
            field[[k, 0]] = h;
            field[[k, 1]] = a.temperature;
            field[[k, 2]] = a.pressure;
            field[[k, 3]] = a.density;
            field[[k, 4]] = a.speed_of_sound;
            field[[k, 5]] = a.wind_factor;
        }
        Ok(Self {
            profile,
            field,
            step,
            fast,
        })
    }

    /// Standard atmosphere at geometric altitude h (m).
    #[inline]
    pub fn at(&self, h: f64) -> Air {
        let h = h.max(0.0);
        if self.fast {
            self.interp(h)
        } else {
            self.analytic(h)
        }
    }

    #[inline]
    fn interp(&self, h: f64) -> Air {
        let n = self.field.nrows();
        let x = h / self.step;
        let i = (x as usize).min(n - 1);
        if i + 1 >= n {
            return row_air(&self.field, n - 1);
        }
        let f = x - i as f64;
        let mut out = row_air(&self.field, i);
        let next = row_air(&self.field, i + 1);
        out.temperature += f * (next.temperature - out.temperature);
        out.pressure += f * (next.pressure - out.pressure);
        out.density += f * (next.density - out.density);
        out.speed_of_sound += f * (next.speed_of_sound - out.speed_of_sound);
        out.wind_factor += f * (next.wind_factor - out.wind_factor);
        out
    }

    #[inline]
    fn analytic(&self, h: f64) -> Air {
        let n = self.profile.nrows();
        let i = (h / self.step) as usize;
        if i + 1 >= n {
            return analytic_at(h, 0.0, 1.0);
        }
        let f = (h - self.profile[[i, 0]]) / self.step;
        let dt = self.profile[[i, 1]] + f * (self.profile[[i + 1, 1]] - self.profile[[i, 1]]);
        let wf = self.profile[[i, 2]] + f * (self.profile[[i + 1, 2]] - self.profile[[i, 2]]);
        analytic_at(h, dt, wf.max(0.0))
    }
}

#[inline]
fn row_air(a: &Array2<f64>, i: usize) -> Air {
    Air {
        temperature: a[[i, 1]],
        pressure: a[[i, 2]],
        density: a[[i, 3]],
        speed_of_sound: a[[i, 4]],
        wind_factor: a[[i, 5]],
    }
}

fn analytic_at(h: f64, dt: f64, wf: f64) -> Air {
    let h = h.max(0.0);
    let (t_base, p_base) = if h < 11_000.0 {
        let t = 288.15 - 0.0065 * h;
        (t, 101_325.0 * (t / 288.15).powf(G / (R * 0.0065)))
    } else if h < 20_000.0 {
        let t = 216.65;
        (t, 22_632.0 * (-G * (h - 11_000.0) / (R * t)).exp())
    } else {
        let t = 216.65 + 0.001 * (h - 20_000.0);
        (t, 5_474.89 * (-G * (h - 20_000.0) / (R * 216.65)).exp())
    };
    let t = t_base + dt;
    Air {
        temperature: t,
        pressure: p_base,
        density: p_base / (R * t),
        speed_of_sound: (GAMMA * R * t).sqrt(),
        wind_factor: wf.max(0.0),
    }
}
