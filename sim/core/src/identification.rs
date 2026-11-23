// SPDX-License-Identifier: GPL-3.0-or-later
//! Typed input contracts for system identification.
//!
//! These are the measurement-side inputs an identification run consumes to
//! calibrate the simulator: a static-fire thrust curve, IMU Allan-deviation
//! points and a servo step response. No measurement data ships with the crate;
//! callers construct every contract from their own fixtures.
//!
//! Two rules govern this module:
//!
//! * **LOUD validation at the boundary.** A constructor refuses empty,
//!   non-finite, non-monotone or physically impossible input with a
//!   [`SimError::Config`] that names the field, the offending value and the
//!   fix. Nothing is zero-filled, clamped or silently repaired.
//! * **Deterministic interpolation.** Every reading is a pure function of the
//!   validated samples — no RNG, no clock, no hidden fallback. Queries outside
//!   the measured domain are LOUD, never extrapolated.

use std::collections::VecDeque;

use crate::error::{Result, SimError};

fn invalid(message: String) -> SimError {
    SimError::Config(format!("identification input: {message}"))
}

fn finite(label: &str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(invalid(format!(
            "{label} must be finite, got {value}; replace the non-finite measurement"
        )))
    }
}

// --- Static-fire thrust curve ------------------------------------------------

/// One (time, thrust) knot of a static-fire measurement. `t_s` is seconds from
/// ignition; `thrust_n` is newtons, non-negative.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThrustSample {
    pub t_s: f64,
    pub thrust_n: f64,
}

/// Validated static-fire thrust curve. At least two strictly time-ordered
/// knots are required, so the curve can always be interpolated.
#[derive(Debug, Clone, PartialEq)]
pub struct ThrustCurve {
    samples: Vec<ThrustSample>,
}

impl ThrustCurve {
    pub fn new(samples: Vec<ThrustSample>) -> Result<Self> {
        if samples.len() < 2 {
            return Err(invalid(format!(
                "static-fire thrust curve needs >= 2 knots to interpolate, got {}; add a second (t_s, thrust_n) sample",
                samples.len()
            )));
        }
        for (i, s) in samples.iter().enumerate() {
            finite(&format!("static-fire thrust sample {i} t_s"), s.t_s)?;
            finite(
                &format!("static-fire thrust sample {i} thrust_n"),
                s.thrust_n,
            )?;
            if s.t_s < 0.0 {
                return Err(invalid(format!(
                    "static-fire thrust sample {i} t_s={} is negative; time is measured from ignition",
                    s.t_s
                )));
            }
            if s.thrust_n < 0.0 {
                return Err(invalid(format!(
                    "static-fire thrust sample {i} thrust_n={} is negative; thrust magnitude cannot be negative",
                    s.thrust_n
                )));
            }
            if i > 0 && s.t_s <= samples[i - 1].t_s {
                return Err(invalid(format!(
                    "static-fire thrust sample {i} t_s={} is not strictly after sample {} t_s={}; sort and deduplicate the knots",
                    s.t_s,
                    i - 1,
                    samples[i - 1].t_s
                )));
            }
        }
        Ok(Self { samples })
    }

    pub fn samples(&self) -> &[ThrustSample] {
        &self.samples
    }

    /// Deterministic piecewise-linear thrust in newtons at `t_s`.
    ///
    /// Exact knots return the stored sample bit-for-bit; a query outside the
    /// measured window is LOUD — there is no clamp and no extrapolation.
    pub fn thrust_at(&self, t_s: f64) -> Result<f64> {
        finite("thrust query t_s", t_s)?;
        let first = self.samples[0];
        let last = self.samples[self.samples.len() - 1];
        if t_s < first.t_s || t_s > last.t_s {
            return Err(invalid(format!(
                "thrust query t_s={t_s} outside measured [{}, {}] s; supply a time inside the static-fire window (no extrapolation)",
                first.t_s, last.t_s
            )));
        }
        for pair in self.samples.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            if t_s <= b.t_s {
                if t_s == a.t_s {
                    return Ok(a.thrust_n);
                }
                if t_s == b.t_s {
                    return Ok(b.thrust_n);
                }
                let u = (t_s - a.t_s) / (b.t_s - a.t_s);
                return Ok(a.thrust_n + u * (b.thrust_n - a.thrust_n));
            }
        }
        // Unreachable: the domain check above admits only t_s <= last.t_s.
        Err(invalid(format!(
            "thrust query t_s={t_s} has no bracketing interval"
        )))
    }
}

// --- IMU Allan points --------------------------------------------------------

/// One Allan-deviation point: σ(τ) at cluster time `τ` (seconds). `sigma` is
/// the Allan *deviation*, not the variance, in the channel's native units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AllanPoint {
    pub tau_s: f64,
    pub sigma: f64,
}

/// Which IMU channel an Allan series was measured on. Keeps gyro and accel
/// points from being mixed in one contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImuChannel {
    Gyro,
    Accel,
}

impl ImuChannel {
    pub fn name(self) -> &'static str {
        match self {
            ImuChannel::Gyro => "gyro",
            ImuChannel::Accel => "accel",
        }
    }
}

/// Validated Allan-deviation series for one IMU channel, with the ARW and bias
/// stability read from it deterministically.
#[derive(Debug, Clone, PartialEq)]
pub struct ImuAllanPoints {
    pub channel: ImuChannel,
    points: Vec<AllanPoint>,
}

impl ImuAllanPoints {
    pub fn new(channel: ImuChannel, points: Vec<AllanPoint>) -> Result<Self> {
        if points.len() < 2 {
            return Err(invalid(format!(
                "{} Allan series needs >= 2 points, got {}; add another (tau_s, sigma) point",
                channel.name(),
                points.len()
            )));
        }
        for (i, p) in points.iter().enumerate() {
            finite(
                &format!("{} Allan point {i} tau_s", channel.name()),
                p.tau_s,
            )?;
            finite(
                &format!("{} Allan point {i} sigma", channel.name()),
                p.sigma,
            )?;
            if p.tau_s <= 0.0 {
                return Err(invalid(format!(
                    "{} Allan point {i} tau_s={} must be > 0; log-log interpolation needs a positive cluster time",
                    channel.name(),
                    p.tau_s
                )));
            }
            if p.sigma <= 0.0 {
                return Err(invalid(format!(
                    "{} Allan point {i} sigma={} must be > 0; a non-positive deviation is not a measurement",
                    channel.name(),
                    p.sigma
                )));
            }
            if i > 0 && p.tau_s <= points[i - 1].tau_s {
                return Err(invalid(format!(
                    "{} Allan point {i} tau_s={} is not strictly after point {} tau_s={}; sort and deduplicate the points",
                    channel.name(),
                    p.tau_s,
                    i - 1,
                    points[i - 1].tau_s
                )));
            }
        }
        Ok(Self { channel, points })
    }

    pub fn points(&self) -> &[AllanPoint] {
        &self.points
    }

    /// ARW estimate: `min(σ(τ) · √τ)` over the measured points. On the pure
    /// τ^-1/2 branch this is exactly the ARW coefficient; the minimum makes
    /// the reading insensitive to a bias floor at longer cluster times.
    pub fn arw(&self) -> f64 {
        self.points
            .iter()
            .map(|p| p.sigma * p.tau_s.sqrt())
            .fold(f64::INFINITY, f64::min)
    }

    /// Bias stability: the floor of the Allan curve, `min σ(τ)`.
    pub fn bias_stability(&self) -> f64 {
        self.points
            .iter()
            .map(|p| p.sigma)
            .fold(f64::INFINITY, f64::min)
    }

    /// Deterministic log-log interpolation of σ at `tau_s`, the standard
    /// Allan-plot reading. Exact knots return the stored σ bit-for-bit; a
    /// query outside the measured cluster times is LOUD.
    pub fn sigma_at(&self, tau_s: f64) -> Result<f64> {
        finite("Allan query tau_s", tau_s)?;
        if tau_s <= 0.0 {
            return Err(invalid(format!(
                "Allan query tau_s={tau_s} must be > 0 for log-log interpolation"
            )));
        }
        let first = self.points[0];
        let last = self.points[self.points.len() - 1];
        if tau_s < first.tau_s || tau_s > last.tau_s {
            return Err(invalid(format!(
                "Allan query tau_s={tau_s} outside measured [{}, {}] s; supply a cluster time inside the curve (no extrapolation)",
                first.tau_s, last.tau_s
            )));
        }
        for pair in self.points.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            if tau_s <= b.tau_s {
                if tau_s == a.tau_s {
                    return Ok(a.sigma);
                }
                if tau_s == b.tau_s {
                    return Ok(b.sigma);
                }
                let u = (tau_s.ln() - a.tau_s.ln()) / (b.tau_s.ln() - a.tau_s.ln());
                return Ok((a.sigma.ln() + u * (b.sigma.ln() - a.sigma.ln())).exp());
            }
        }
        // Unreachable: the domain check above admits only tau_s <= last.tau_s.
        Err(invalid(format!(
            "Allan query tau_s={tau_s} has no bracketing interval"
        )))
    }
}

// --- Servo step response -----------------------------------------------------

/// One row of a servo step test. `commanded` and `measured` are in the same
/// output units (e.g. fin deflection in degrees).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ServoStep {
    pub t_s: f64,
    pub commanded: f64,
    pub measured: f64,
}

/// Validated servo identification: pure delay, slew rate and input deadband,
/// together with the step rows they were read from.
#[derive(Debug, Clone, PartialEq)]
pub struct ServoStepResponse {
    /// Pure transport delay before any motion starts, seconds (>= 0).
    pub delay_s: f64,
    /// Maximum slew rate, output units per second (> 0).
    pub rate_per_s: f64,
    /// Input deadband: `|command| <= deadband` produces no motion (>= 0).
    pub deadband: f64,
    samples: Vec<ServoStep>,
}

impl ServoStepResponse {
    pub fn new(
        delay_s: f64,
        rate_per_s: f64,
        deadband: f64,
        samples: Vec<ServoStep>,
    ) -> Result<Self> {
        finite("servo delay_s", delay_s)?;
        finite("servo rate_per_s", rate_per_s)?;
        finite("servo deadband", deadband)?;
        if delay_s < 0.0 {
            return Err(invalid(format!(
                "servo delay_s={delay_s} is negative; a response cannot precede its command"
            )));
        }
        if rate_per_s <= 0.0 {
            return Err(invalid(format!(
                "servo rate_per_s={rate_per_s} must be > 0; a stuck servo has no identifiable rate"
            )));
        }
        if deadband < 0.0 {
            return Err(invalid(format!(
                "servo deadband={deadband} is negative; a deadband is a non-negative width"
            )));
        }
        for (i, s) in samples.iter().enumerate() {
            finite(&format!("servo step {i} t_s"), s.t_s)?;
            finite(&format!("servo step {i} commanded"), s.commanded)?;
            finite(&format!("servo step {i} measured"), s.measured)?;
            if i > 0 && s.t_s <= samples[i - 1].t_s {
                return Err(invalid(format!(
                    "servo step {i} t_s={} is not strictly after step {} t_s={}; sort and deduplicate the step rows",
                    s.t_s,
                    i - 1,
                    samples[i - 1].t_s
                )));
            }
        }
        if !samples.is_empty() && !samples.iter().any(|s| s.commanded.abs() > deadband) {
            return Err(invalid(format!(
                "servo step test never commands outside the deadband={deadband}; raise the step amplitude or the rate is not identifiable"
            )));
        }
        Ok(Self {
            delay_s,
            rate_per_s,
            deadband,
            samples,
        })
    }

    pub fn samples(&self) -> &[ServoStep] {
        &self.samples
    }

    /// Command after the dead zone is removed: zero inside the band and the
    /// band width subtracted outside it (continuous, slope one).
    pub fn effective_command(&self, command: f64) -> Result<f64> {
        finite("servo command", command)?;
        if command.abs() <= self.deadband {
            return Ok(0.0);
        }
        Ok(command - command.signum() * self.deadband)
    }

    /// Deterministic predicted output at `t_s` for a step applied at t=0:
    /// deadband, then pure delay, then a rate-limited ramp to the target.
    pub fn response_at(&self, command: f64, t_s: f64) -> Result<f64> {
        finite("servo response time t_s", t_s)?;
        if t_s < 0.0 {
            return Err(invalid(format!(
                "servo response time t_s={t_s} is negative; the step is applied at t=0"
            )));
        }
        let target = self.effective_command(command)?;
        let active = t_s - self.delay_s;
        if active <= 0.0 {
            return Ok(0.0);
        }
        let magnitude = (self.rate_per_s * active).min(target.abs());
        Ok(target.signum() * magnitude)
    }

    /// Time from step onset until the predicted output reaches its target.
    pub fn settle_time_s(&self, command: f64) -> Result<f64> {
        let target = self.effective_command(command)?;
        Ok(self.delay_s + target.abs() / self.rate_per_s)
    }
}

// --- Stateful servo actuator -------------------------------------------------

/// A stateful actuator driven by a validated [`ServoStepResponse`].
///
/// `response_at` describes a *single* step applied at t=0; actuation is a
/// stream of commands. One `step` advances the actuator by one `dt` tick: the
/// new command enters a pure-delay line, the deadband is removed from the
/// delayed command, and the output slews toward it at the identified rate.
/// With a constant command held from t=0 the output equals `response_at` at
/// every tick (up to the delay being quantised to whole ticks), which is what
/// the wiring gate asserts. No parameter is fitted to a controller outcome.
#[derive(Debug, Clone)]
pub struct ServoActuator {
    response: ServoStepResponse,
    dt: f64,
    delay_steps: usize,
    queue: VecDeque<f64>,
    current: f64,
}

impl ServoActuator {
    /// Build an actuator from a validated step response and a positive `dt`.
    /// Invalid input is LOUD: it names the field, the value and the fix.
    pub fn new(response: ServoStepResponse, dt: f64) -> Result<Self> {
        finite("servo actuator dt", dt)?;
        if dt <= 0.0 {
            return Err(invalid(format!(
                "servo actuator dt={dt} must be > 0; the delay line is quantised to whole dt ticks"
            )));
        }
        let delay_steps = (response.delay_s / dt).ceil() as usize;
        Ok(Self {
            response,
            dt,
            delay_steps,
            queue: VecDeque::new(),
            current: 0.0,
        })
    }

    /// Push one new command and advance by one `dt` tick, returning the output.
    /// A non-finite command is LOUD (via `effective_command`).
    pub fn step(&mut self, command: f64) -> Result<f64> {
        // Checked on the arriving command, not on the delayed one: a non-finite
        // command must be LOUD now rather than after it drains through the
        // delay line.
        finite("servo command", command)?;
        self.queue.push_back(command);
        let delayed = if self.queue.len() > self.delay_steps {
            self.queue.pop_front().expect("delay line non-empty")
        } else {
            0.0
        };
        let target = self.response.effective_command(delayed)?;
        let max_delta = self.response.rate_per_s * self.dt;
        self.current += (target - self.current).clamp(-max_delta, max_delta);
        Ok(self.current)
    }

    /// Current output in the response's command units.
    pub fn output(&self) -> f64 {
        self.current
    }

    pub fn response(&self) -> &ServoStepResponse {
        &self.response
    }

    pub fn dt(&self) -> f64 {
        self.dt
    }

    /// Pure delay quantised to whole `dt` ticks (`ceil` of delay_s / dt).
    pub fn delay_steps(&self) -> usize {
        self.delay_steps
    }
}
