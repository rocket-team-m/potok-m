// SPDX-License-Identifier: GPL-3.0-or-later
//! Deterministic two-stage recovery canopy (CHUTE01).
//!
//! A self-contained, opt-in model of the *classical* recovery chain: a spring
//! ejected drogue, then a spring-ejected main, with one-shot latches. There is
//! no pyro channel and no AI authority anywhere under canopy — the step API
//! takes no command, and [`AI_AUTHORITY_UNDER_CANOPY`] is false. Recovery stays
//! outside the control loop, as `docs/src/content/docs/hardware/recovery.md`
//! requires.
//!
//! The model is a pure function of `ParachuteConfig`, the flight context and
//! `dt`: no RNG, no clock, no global state. `enabled = false` (the shipped
//! default) makes `step` a no-op, so the fin-flight trajectory and every
//! existing A/B remain bit-identical. When enabled the phase and canopy state
//! are emitted through [`ParachuteState::truth_floats`] /
//! [`CHUTE_TRUTH_NAMES`], and the landing point is recorded through
//! [`LandingDispersion::manifest_named_floats`] / [`CHUTE_MANIFEST_FIELDS`]
//! — separately from the fin-flight `trajectory::lateral_rms_m`.
//!
//! Physics: each stage relaxes the signed vertical velocity toward a target
//! descent rate with a first-order inflation transient, and relaxes horizontal
//! velocity toward the *true* wind with a coupling time constant. Position is
//! the integral of those velocities. One launch-to-landing pass is idempotent:
//! a stage never deploys twice and `Landed` fires exactly once.
//!
//! Integration note for the crate owner: this file is intentionally not wired
//! into `lib.rs` by this task. Add `pub mod parachute;` there, hold
//! `ParachuteConfig` in `Config` with `#[serde(default)]`, and call
//! `cfg.parachute.validate().map_err(|e| SimError::Config(e.to_string()))?`
//! from `Config::validate`. See `docs/PARACHUTE.md`.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Under canopy the AI has no command path at all. Not a config knob: recovery
/// is classical hardware logic and must not become controllable.
pub const AI_AUTHORITY_UNDER_CANOPY: bool = false;

/// f32 fields appended to the privileged truth tick when the canopy is on.
/// Names are single-sourced here so the writer and the manifest cannot drift.
pub const CHUTE_TRUTH_NAMES: [&str; 8] = [
    "chute_phase",
    "chute_inflation",
    "chute_drogue_t_s",
    "chute_main_t_s",
    "chute_vz_mps",
    "chute_wind_x",
    "chute_wind_y",
    "chute_landed",
];

/// Manifest fields for canopy landing dispersion. These are *not* a component
/// of `lateral_rms_m`: the fin-flight RMS stays over pre-deployment ticks and
/// `chute_landing_dispersion_m` is the canopy-only deploy-to-landing offset.
pub const CHUTE_MANIFEST_FIELDS: [&str; 9] = [
    "chute_enabled",
    "chute_drogue_t_s",
    "chute_main_t_s",
    "chute_landing_t_s",
    "chute_deploy_altitude_m",
    "chute_deploy_x_m",
    "chute_deploy_y_m",
    "chute_landing_dispersion_m",
    "chute_landing_radius_m",
];

// --- Config -----------------------------------------------------------------

/// Typed deployment condition. `kind` is part of the wire/config contract;
/// never add a variant that maps onto an existing numeric code.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DeployTrigger {
    /// Fire when altitude at or below `value` metres AGL.
    AltitudeM { value: f64 },
    /// Fire `value` seconds after the apogee time tracked by the flight.
    TimeSinceApogeeS { value: f64 },
    /// Fire `value` seconds after launch.
    TimeSinceLaunchS { value: f64 },
}

impl DeployTrigger {
    pub fn value(self) -> f64 {
        match self {
            Self::AltitudeM { value }
            | Self::TimeSinceApogeeS { value }
            | Self::TimeSinceLaunchS { value } => value,
        }
    }

    pub fn needs_apogee(self) -> bool {
        matches!(self, Self::TimeSinceApogeeS { .. })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct StageConfig {
    pub trigger: DeployTrigger,
    /// Positive magnitude of the settled descent rate, m/s.
    pub target_descent_mps: f64,
    /// First-order time constant for inflation and descent relaxation, s.
    pub inflation_s: f64,
}

impl Default for StageConfig {
    fn default() -> Self {
        Self {
            trigger: DeployTrigger::TimeSinceApogeeS { value: 2.0 },
            target_descent_mps: 25.0,
            inflation_s: 1.2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ParachuteConfig {
    /// Opt-in. `false` keeps every existing trajectory and A/B unchanged.
    pub enabled: bool,
    /// A trigger only fires while the vehicle is actually descending.
    pub require_descent_confirm: bool,
    /// Time constant with which the canopy follows the true horizontal wind, s.
    pub wind_coupling_s: f64,
    /// Altitude at which the landing event fires, m AGL.
    pub landing_altitude_m: f64,
    pub drogue: StageConfig,
    pub main: StageConfig,
}

impl Default for ParachuteConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            require_descent_confirm: true,
            wind_coupling_s: 3.0,
            landing_altitude_m: 0.0,
            drogue: StageConfig::default(),
            main: StageConfig {
                trigger: DeployTrigger::AltitudeM { value: 1500.0 },
                target_descent_mps: 6.5,
                inflation_s: 2.0,
            },
        }
    }
}

/// LOUD configuration error: names the field, the offending value/rule and the
/// escape. No silent clamp, no degraded canopy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChuteError {
    pub field: &'static str,
    pub detail: String,
}

impl ChuteError {
    fn new(field: &'static str, detail: impl Into<String>) -> Self {
        Self {
            field,
            detail: detail.into(),
        }
    }
}

impl fmt::Display for ChuteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "parachute.{}: {} (config rejected before flight; fix the value or keep enabled = false)",
            self.field, self.detail
        )
    }
}

impl std::error::Error for ChuteError {}

fn finite(field: &'static str, v: f64) -> Result<(), ChuteError> {
    if v.is_finite() {
        Ok(())
    } else {
        Err(ChuteError::new(
            field,
            format!("{v} is not finite; supply a finite number"),
        ))
    }
}

fn range(field: &'static str, v: f64, lo: f64, hi: f64) -> Result<(), ChuteError> {
    finite(field, v)?;
    if (lo..=hi).contains(&v) {
        Ok(())
    } else {
        Err(ChuteError::new(
            field,
            format!("{v} outside [{lo}, {hi}]; set a value in that range"),
        ))
    }
}

fn validate_stage(prefix: &'static str, s: &StageConfig) -> Result<(), ChuteError> {
    let trigger_field: &'static str = match prefix {
        "drogue" => "drogue.trigger",
        _ => "main.trigger",
    };
    let descent_field: &'static str = match prefix {
        "drogue" => "drogue.target_descent_mps",
        _ => "main.target_descent_mps",
    };
    let inflation_field: &'static str = match prefix {
        "drogue" => "drogue.inflation_s",
        _ => "main.inflation_s",
    };
    finite(trigger_field, s.trigger.value())?;
    if s.trigger.value() <= 0.0 || s.trigger.value() > 1e5 {
        return Err(ChuteError::new(
            trigger_field,
            format!(
                "{} must be in (0, 100000]; set a positive trigger value",
                s.trigger.value()
            ),
        ));
    }
    range(descent_field, s.target_descent_mps, 0.5, 100.0)?;
    range(inflation_field, s.inflation_s, 1e-3, 60.0)?;
    Ok(())
}

impl ParachuteConfig {
    /// Called even when `enabled = false`: a malformed config must never lie
    /// dormant waiting for someone to switch the feature on.
    pub fn validate(&self) -> Result<(), ChuteError> {
        range("wind_coupling_s", self.wind_coupling_s, 1e-3, 60.0)?;
        finite("landing_altitude_m", self.landing_altitude_m)?;
        if self.landing_altitude_m < 0.0 || self.landing_altitude_m > 1e5 {
            return Err(ChuteError::new(
                "landing_altitude_m",
                format!(
                    "{} outside [0, 100000]; set a non-negative AGL altitude",
                    self.landing_altitude_m
                ),
            ));
        }
        validate_stage("drogue", &self.drogue)?;
        validate_stage("main", &self.main)?;

        if self.main.target_descent_mps >= self.drogue.target_descent_mps {
            return Err(ChuteError::new(
                "main.target_descent_mps",
                format!(
                    "{} must be slower than drogue {}; a two-stage canopy slows down at main",
                    self.main.target_descent_mps, self.drogue.target_descent_mps
                ),
            ));
        }
        match (self.drogue.trigger, self.main.trigger) {
            (DeployTrigger::AltitudeM { value: d }, DeployTrigger::AltitudeM { value: m }) => {
                if m >= d {
                    return Err(ChuteError::new(
                        "main.trigger",
                        format!(
                            "main altitude {m} must be BELOW drogue altitude {d}; main cannot precede drogue"
                        ),
                    ));
                }
            }
            (
                DeployTrigger::TimeSinceLaunchS { value: d },
                DeployTrigger::TimeSinceLaunchS { value: m },
            ) => {
                if m <= d {
                    return Err(ChuteError::new(
                        "main.trigger",
                        format!(
                            "main time {m} must be AFTER drogue time {d}; main cannot precede drogue"
                        ),
                    ));
                }
            }
            (
                DeployTrigger::TimeSinceApogeeS { value: d },
                DeployTrigger::TimeSinceApogeeS { value: m },
            ) => {
                if m <= d {
                    return Err(ChuteError::new(
                        "main.trigger",
                        format!(
                            "main offset {m} must be AFTER drogue offset {d}; main cannot precede drogue"
                        ),
                    ));
                }
            }
            _ => {}
        }
        Ok(())
    }
}

// --- Runtime state ----------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChutePhase {
    Stowed = 0,
    DrogueInflating = 1,
    Drogue = 2,
    MainInflating = 3,
    Main = 4,
    Landed = 5,
}

impl ChutePhase {
    pub fn code(self) -> u8 {
        self as u8
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stowed => "stowed",
            Self::DrogueInflating => "drogue_inflating",
            Self::Drogue => "drogue",
            Self::MainInflating => "main_inflating",
            Self::Main => "main",
            Self::Landed => "landed",
        }
    }

    pub fn under_canopy(self) -> bool {
        matches!(
            self,
            Self::DrogueInflating | Self::Drogue | Self::MainInflating | Self::Main
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParachuteState {
    pub phase: ChutePhase,
    pub t_s: f64,
    pub altitude_m: f64,
    /// Signed vertical velocity, m/s (negative = descending).
    pub vz_mps: f64,
    /// Pad-relative horizontal position, m.
    pub pos_m: [f64; 2],
    /// Ground-frame horizontal velocity, m/s.
    pub vel_xy_mps: [f64; 2],
    /// Inflation progress of the active stage, 0..1.
    pub inflation: f64,
    pub drogue_t_s: Option<f64>,
    pub main_t_s: Option<f64>,
}

impl ParachuteState {
    /// f32 values in [`CHUTE_TRUTH_NAMES`] order. `-1.0` means "not deployed".
    pub fn truth_floats(&self, true_wind_xy: [f64; 2]) -> [f32; 8] {
        [
            self.phase.code() as f32,
            self.inflation as f32,
            self.drogue_t_s.unwrap_or(-1.0) as f32,
            self.main_t_s.unwrap_or(-1.0) as f32,
            self.vz_mps as f32,
            true_wind_xy[0] as f32,
            true_wind_xy[1] as f32,
            if self.phase == ChutePhase::Landed {
                1.0
            } else {
                0.0
            },
        ]
    }
}

/// Recorded once at touchdown. `lateral_offset_m` is canopy drift only; it is
/// deliberately a different metric from the fin-flight `lateral_rms_m`.
#[derive(Debug, Clone, PartialEq)]
pub struct LandingDispersion {
    pub deploy_pos_m: [f64; 2],
    pub deploy_altitude_m: f64,
    pub landing_pos_m: [f64; 2],
    pub landing_t_s: f64,
    pub drogue_t_s: f64,
    pub main_t_s: f64,
    pub settled_descent_mps: f64,
}

impl LandingDispersion {
    /// True-wind canopy drift from deployment to landing, metres.
    pub fn lateral_offset_m(&self) -> f64 {
        let dx = self.landing_pos_m[0] - self.deploy_pos_m[0];
        let dy = self.landing_pos_m[1] - self.deploy_pos_m[1];
        dx.hypot(dy)
    }

    /// Total pad-relative landing miss, metres (fin flight + canopy).
    pub fn radial_from_pad_m(&self) -> f64 {
        self.landing_pos_m[0].hypot(self.landing_pos_m[1])
    }

    /// Named f64 values in [`CHUTE_MANIFEST_FIELDS`] order.
    pub fn manifest_named_floats(&self, enabled: bool) -> [(&'static str, f64); 9] {
        [
            ("chute_enabled", if enabled { 1.0 } else { 0.0 }),
            ("chute_drogue_t_s", self.drogue_t_s),
            ("chute_main_t_s", self.main_t_s),
            ("chute_landing_t_s", self.landing_t_s),
            ("chute_deploy_altitude_m", self.deploy_altitude_m),
            ("chute_deploy_x_m", self.deploy_pos_m[0]),
            ("chute_deploy_y_m", self.deploy_pos_m[1]),
            (
                "chute_landing_dispersion_m",
                self.lateral_offset_m(),
            ),
            ("chute_landing_radius_m", self.radial_from_pad_m()),
        ]
    }
}

/// Flight values the canopy needs for one step. There is intentionally no
/// command field: nothing outside the classical recovery chain can steer it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlightCtx {
    pub t_s: f64,
    pub altitude_m: f64,
    pub vz_mps: f64,
    pub pos_m: [f64; 2],
    pub vel_xy_mps: [f64; 2],
    /// True wind at the current altitude, world frame (m/s).
    pub true_wind_xy: [f64; 2],
    /// None before apogee is detected; a time-since-apogee trigger simply waits.
    pub t_apogee_s: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct Parachute {
    cfg: ParachuteConfig,
    st: ParachuteState,
    landing: Option<LandingDispersion>,
    deploy_pos: [f64; 2],
    deploy_alt: f64,
}

impl Parachute {
    pub fn new(cfg: ParachuteConfig) -> Result<Self, ChuteError> {
        cfg.validate()?;
        Ok(Self {
            cfg,
            st: ParachuteState {
                phase: ChutePhase::Stowed,
                t_s: 0.0,
                altitude_m: 0.0,
                vz_mps: 0.0,
                pos_m: [0.0; 2],
                vel_xy_mps: [0.0; 2],
                inflation: 0.0,
                drogue_t_s: None,
                main_t_s: None,
            },
            landing: None,
            deploy_pos: [0.0; 2],
            deploy_alt: 0.0,
        })
    }

    pub fn config(&self) -> &ParachuteConfig {
        &self.cfg
    }

    pub fn state(&self) -> &ParachuteState {
        &self.st
    }

    pub fn landing(&self) -> Option<&LandingDispersion> {
        self.landing.as_ref()
    }

    /// Start a fresh episode: stowed, no latches, no landing record.
    pub fn reset(&mut self) {
        self.st = ParachuteState {
            phase: ChutePhase::Stowed,
            t_s: 0.0,
            altitude_m: 0.0,
            vz_mps: 0.0,
            pos_m: [0.0; 2],
            vel_xy_mps: [0.0; 2],
            inflation: 0.0,
            drogue_t_s: None,
            main_t_s: None,
        };
        self.landing = None;
        self.deploy_pos = [0.0; 2];
        self.deploy_alt = 0.0;
    }

    /// Advance one step. `dt` must be positive and every context value finite;
    /// malformed input is a LOUD error, never a clamped canopy.
    pub fn step(&mut self, dt: f64, ctx: FlightCtx) -> Result<(), ChuteError> {
        finite("dt", dt)?;
        if dt <= 0.0 {
            return Err(ChuteError::new("dt", format!("{dt} must be > 0")));
        }
        for (name, v) in [
            ("t_s", ctx.t_s),
            ("altitude_m", ctx.altitude_m),
            ("vz_mps", ctx.vz_mps),
            ("pos_m.x", ctx.pos_m[0]),
            ("pos_m.y", ctx.pos_m[1]),
            ("vel_xy_mps.x", ctx.vel_xy_mps[0]),
            ("vel_xy_mps.y", ctx.vel_xy_mps[1]),
            ("true_wind_xy.x", ctx.true_wind_xy[0]),
            ("true_wind_xy.y", ctx.true_wind_xy[1]),
        ] {
            finite(name, v)?;
        }
        if !self.cfg.enabled || self.st.phase == ChutePhase::Landed {
            return Ok(()); // disabled = inert; landed = one-shot latch
        }

        if self.st.phase == ChutePhase::Stowed {
            if !self.trigger_met(self.cfg.drogue.trigger, &ctx)
                || (self.cfg.require_descent_confirm && ctx.vz_mps >= 0.0)
            {
                return Ok(());
            }
            // Spring ejection, one-shot latch. No pyro, no command path.
            self.st.phase = ChutePhase::DrogueInflating;
            self.st.drogue_t_s = Some(ctx.t_s);
            self.st.inflation = 0.0;
            self.st.t_s = ctx.t_s;
            self.st.altitude_m = ctx.altitude_m;
            self.st.vz_mps = ctx.vz_mps.min(0.0);
            self.st.pos_m = ctx.pos_m;
            self.st.vel_xy_mps = ctx.vel_xy_mps;
            self.deploy_pos = ctx.pos_m;
            self.deploy_alt = ctx.altitude_m;
            return Ok(());
        }

        let (stage, settled, stable) = match self.st.phase {
            ChutePhase::MainInflating | ChutePhase::Main => (
                self.cfg.main,
                self.cfg.main.target_descent_mps,
                ChutePhase::Main,
            ),
            _ => (
                self.cfg.drogue,
                self.cfg.drogue.target_descent_mps,
                ChutePhase::Drogue,
            ),
        };

        // Main only considered after drogue latched; a mixed trigger pair can
        // therefore never invert the stages even if its numbers overlap.
        if self.st.main_t_s.is_none()
            && self.trigger_met(self.cfg.main.trigger, &ctx)
            && (!self.cfg.require_descent_confirm || self.st.vz_mps < 0.0)
        {
            self.st.phase = ChutePhase::MainInflating;
            self.st.main_t_s = Some(ctx.t_s);
            self.st.inflation = 0.0;
        }

        let alpha = 1.0 - (-dt / stage.inflation_s).exp();
        let alpha_w = 1.0 - (-dt / self.cfg.wind_coupling_s).exp();
        let target_vz = -settled;
        let vz = self.st.vz_mps + (target_vz - self.st.vz_mps) * alpha;
        let vx = self.st.vel_xy_mps[0]
            + (ctx.true_wind_xy[0] - self.st.vel_xy_mps[0]) * alpha_w;
        let vy = self.st.vel_xy_mps[1]
            + (ctx.true_wind_xy[1] - self.st.vel_xy_mps[1]) * alpha_w;
        self.st.inflation = (self.st.inflation + (1.0 - self.st.inflation) * alpha).min(1.0);
        if self.st.inflation >= 0.999 {
            self.st.phase = stable;
        }

        let mut alt = self.st.altitude_m + vz * dt;
        self.st.pos_m[0] += vx * dt;
        self.st.pos_m[1] += vy * dt;
        self.st.vel_xy_mps = [vx, vy];
        self.st.t_s = ctx.t_s;

        if alt <= self.cfg.landing_altitude_m {
            alt = self.cfg.landing_altitude_m;
            self.st.phase = ChutePhase::Landed;
            self.st.vz_mps = 0.0;
            self.st.vel_xy_mps = [0.0; 2];
            self.landing = Some(LandingDispersion {
                deploy_pos_m: self.deploy_pos,
                deploy_altitude_m: self.deploy_alt,
                landing_pos_m: self.st.pos_m,
                landing_t_s: ctx.t_s,
                drogue_t_s: self.st.drogue_t_s.unwrap_or(ctx.t_s),
                main_t_s: self.st.main_t_s.unwrap_or(ctx.t_s),
                settled_descent_mps: settled,
            });
        } else {
            self.st.vz_mps = vz;
        }
        self.st.altitude_m = alt;
        Ok(())
    }

    fn trigger_met(&self, trigger: DeployTrigger, ctx: &FlightCtx) -> bool {
        match trigger {
            DeployTrigger::AltitudeM { value } => ctx.altitude_m <= value,
            DeployTrigger::TimeSinceLaunchS { value } => ctx.t_s >= value,
            DeployTrigger::TimeSinceApogeeS { value } => ctx
                .t_apogee_s
                .map(|t_ap| ctx.t_s - t_ap >= value)
                .unwrap_or(false),
        }
    }
}
