// SPDX-License-Identifier: GPL-3.0-or-later
//! Flight-log replay seam for real telemetry: proposed schema, strict import,
//! timestamp/phase alignment, residuals against a simulated episode, and
//! NEES / chi-square corridor helpers.
//!
//! STATUS: **no real flight log has been supplied or examined.** The schema,
//! import, alignment and statistics code is exercised by synthetic fixtures,
//! and [`synthetic_nees_corridor_gate`] is a deterministic 10^4-run check that
//! the NEES corridor discriminates a calibrated estimator from a biased one.
//! That validates the instrument only. [`validate_real_log_nees`] is the real
//! data entry point and is LOUD `NOT MEASURED`: nothing here validates the
//! vehicle, the sensors, the simulator or any real recording.
//!
//! Schema v1 has one canonical in-memory form ([`FlightLog`]) and two strict
//! encodings:
//!
//! * JSON — the serialized [`FlightLog`] struct, field order fixed by serde.
//! * CSV — four `# key=value` metadata lines (`schema_version`,
//!   `episode_index`, `sample_rate_hz`, `sigma` — eight comma-separated values
//!   in [`CHANNEL_NAMES`] order), then the fixed [`CSV_HEADER`] row, then one
//!   row per sample in that column order. No quoting, no embedded commas;
//!   plain decimal or scientific notation only.
//!
//! `time_us` is a strictly increasing `u64` timestamp, `phase` is the
//! canonical wire code `0..=4` (Pad and Boost both map to 0), and `sigma` holds
//! measurement 1-sigma values in [`CHANNEL_UNITS`]. Import never reorders,
//! fills, clamps or guesses: every deviation is a loud error.

use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{episode::Episode, sensor::Frame};

/// Errors from the replay seam. Kept local so `lib.rs` needs no new import.
#[derive(Debug, Error)]
pub enum FlightLogError {
    #[error("unsupported flight-log schema version {got} (expected {want})")]
    UnsupportedVersion { got: u32, want: u32 },
    #[error("malformed flight-log JSON: {0}")]
    Json(String),
    #[error("malformed flight-log CSV: {0}")]
    Csv(String),
    #[error("invalid flight log: {0}")]
    Invalid(String),
    #[error("flight-log alignment error: {0}")]
    Alignment(String),
    #[error("flight-log statistic error: {0}")]
    Stats(String),
    #[error("real-data NEES corridor is {0}")]
    NotMeasured(String),
}

pub type Result<T> = std::result::Result<T, FlightLogError>;

pub const SCHEMA_VERSION: u32 = 1;
pub const CHANNEL_COUNT: usize = 8;

/// Residual/NEES channel order: the eight raw telemetry channels of a
/// canonical request frame, in wire order.
pub const CHANNEL_NAMES: [&str; CHANNEL_COUNT] = [
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "accel_x",
    "accel_y",
    "accel_z",
    "height_m",
    "pitot_dp_pa",
];

pub const CHANNEL_UNITS: [&str; CHANNEL_COUNT] = [
    "rad/s", "rad/s", "rad/s", "m/s^2", "m/s^2", "m/s^2", "m", "Pa",
];

/// All channels, in [`CHANNEL_NAMES`] order; the default NEES channel set.
pub const ALL_CHANNELS: [usize; CHANNEL_COUNT] = [0, 1, 2, 3, 4, 5, 6, 7];

pub const CSV_HEADER: &str =
    "time_us,phase,gyro_x,gyro_y,gyro_z,accel_x,accel_y,accel_z,height_m,pitot_dp_pa";

/// Half a 50 Hz tick: the default largest clock offset still treated as the
/// same sampling instant.
pub const HALF_TICK_US: u64 = 10_000;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Observation {
    /// Body angular rate, rad/s.
    pub gyro: [f64; 3],
    /// Body specific force, m/s^2.
    pub accel: [f64; 3],
    /// Barometric height, m.
    pub height_m: f64,
    /// Pitot differential pressure, Pa.
    pub pitot_dp_pa: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlightSample {
    pub time_us: u64,
    /// Canonical wire phase code, 0..=4.
    pub phase: u8,
    pub obs: Observation,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlightLog {
    pub schema_version: u32,
    pub episode_index: u64,
    pub sample_rate_hz: u64,
    /// Measurement 1-sigma per channel, in [`CHANNEL_NAMES`] order.
    pub sigma: [f64; CHANNEL_COUNT],
    pub samples: Vec<FlightSample>,
}

/// All import paths funnel through here. A log that fails validation is never
/// partially admitted: the caller gets no `FlightLog` at all.
pub fn validate(log: &FlightLog) -> Result<()> {
    if log.schema_version != SCHEMA_VERSION {
        return Err(FlightLogError::UnsupportedVersion {
            got: log.schema_version,
            want: SCHEMA_VERSION,
        });
    }
    if log.sample_rate_hz == 0 {
        return Err(FlightLogError::Invalid("sample_rate_hz must be > 0".into()));
    }
    for (i, &s) in log.sigma.iter().enumerate() {
        if !s.is_finite() || s <= 0.0 {
            return Err(FlightLogError::Invalid(format!(
                "sigma[{i}] ({}) is not finite and positive",
                CHANNEL_NAMES[i]
            )));
        }
    }
    if log.samples.is_empty() {
        return Err(FlightLogError::Invalid("samples must not be empty".into()));
    }
    let mut prev: Option<u64> = None;
    for (i, sample) in log.samples.iter().enumerate() {
        if sample.time_us > i64::MAX as u64 {
            return Err(FlightLogError::Invalid(format!(
                "sample[{i}].time_us exceeds the signed-microsecond range"
            )));
        }
        if let Some(p) = prev {
            if sample.time_us <= p {
                return Err(FlightLogError::Invalid(format!(
                    "sample[{i}].time_us {} is not after {p}",
                    sample.time_us
                )));
            }
        }
        prev = Some(sample.time_us);
        if sample.phase > 4 {
            return Err(FlightLogError::Invalid(format!(
                "sample[{i}].phase {} is outside 0..=4",
                sample.phase
            )));
        }
        let obs = &sample.obs;
        for &v in obs
            .gyro
            .iter()
            .chain(obs.accel.iter())
            .chain([obs.height_m, obs.pitot_dp_pa].iter())
        {
            if !v.is_finite() {
                return Err(FlightLogError::Invalid(format!(
                    "sample[{i}] carries a non-finite measurement"
                )));
            }
        }
    }
    Ok(())
}

pub fn from_json(text: &str) -> Result<FlightLog> {
    let log: FlightLog =
        serde_json::from_str(text).map_err(|e| FlightLogError::Json(e.to_string()))?;
    validate(&log)?;
    Ok(log)
}

pub fn to_json(log: &FlightLog) -> Result<String> {
    validate(log)?;
    let mut out =
        serde_json::to_string_pretty(log).map_err(|e| FlightLogError::Json(e.to_string()))?;
    out.push('\n');
    Ok(out)
}

fn set_once<T>(slot: &mut Option<T>, value: T, key: &str, line: usize) -> Result<()> {
    if slot.is_some() {
        return Err(FlightLogError::Csv(format!(
            "line {line}: duplicate metadata key {key}"
        )));
    }
    *slot = Some(value);
    Ok(())
}

fn parse_field<T: std::str::FromStr>(s: &str, key: &str, line: usize) -> Result<T> {
    s.parse::<T>()
        .map_err(|_| FlightLogError::Csv(format!("line {line}: bad {key} value {s:?}")))
}

pub fn from_csv(text: &str) -> Result<FlightLog> {
    let mut schema_version = None;
    let mut episode_index = None;
    let mut sample_rate_hz = None;
    let mut sigma = None;
    let mut header_seen = false;
    let mut samples = Vec::new();

    for (lineno, raw) in text.lines().enumerate() {
        let line = raw.trim_end_matches('\r');
        if line.is_empty() {
            continue;
        }
        let line_no = lineno + 1;
        if !header_seen {
            if let Some(rest) = line.strip_prefix('#') {
                let (key, value) = rest.trim_start().split_once('=').ok_or_else(|| {
                    FlightLogError::Csv(format!("line {line_no}: metadata needs key=value"))
                })?;
                match key {
                    "schema_version" => set_once(
                        &mut schema_version,
                        parse_field::<u32>(value, key, line_no)?,
                        key,
                        line_no,
                    )?,
                    "episode_index" => set_once(
                        &mut episode_index,
                        parse_field::<u64>(value, key, line_no)?,
                        key,
                        line_no,
                    )?,
                    "sample_rate_hz" => set_once(
                        &mut sample_rate_hz,
                        parse_field::<u64>(value, key, line_no)?,
                        key,
                        line_no,
                    )?,
                    "sigma" => {
                        let parts: Vec<&str> = value.split(',').collect();
                        if parts.len() != CHANNEL_COUNT {
                            return Err(FlightLogError::Csv(format!(
                                "line {line_no}: sigma needs {CHANNEL_COUNT} values, got {}",
                                parts.len()
                            )));
                        }
                        let mut values = [0.0; CHANNEL_COUNT];
                        for (i, part) in parts.iter().enumerate() {
                            values[i] = parse_field::<f64>(part, "sigma", line_no)?;
                        }
                        set_once(&mut sigma, values, key, line_no)?;
                    }
                    _ => {
                        return Err(FlightLogError::Csv(format!(
                            "line {line_no}: unknown metadata key {key:?}"
                        )))
                    }
                }
                continue;
            }
            if line != CSV_HEADER {
                return Err(FlightLogError::Csv(format!(
                    "line {line_no}: expected CSV header {CSV_HEADER:?}"
                )));
            }
            header_seen = true;
            continue;
        }

        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() != 2 + CHANNEL_COUNT {
            return Err(FlightLogError::Csv(format!(
                "line {line_no}: expected {} columns, got {}",
                2 + CHANNEL_COUNT,
                cols.len()
            )));
        }
        let obs = Observation {
            gyro: [
                parse_field(cols[2], "gyro_x", line_no)?,
                parse_field(cols[3], "gyro_y", line_no)?,
                parse_field(cols[4], "gyro_z", line_no)?,
            ],
            accel: [
                parse_field(cols[5], "accel_x", line_no)?,
                parse_field(cols[6], "accel_y", line_no)?,
                parse_field(cols[7], "accel_z", line_no)?,
            ],
            height_m: parse_field(cols[8], "height_m", line_no)?,
            pitot_dp_pa: parse_field(cols[9], "pitot_dp_pa", line_no)?,
        };
        samples.push(FlightSample {
            time_us: parse_field(cols[0], "time_us", line_no)?,
            phase: parse_field(cols[1], "phase", line_no)?,
            obs,
        });
    }

    if !header_seen {
        return Err(FlightLogError::Csv("missing CSV header".into()));
    }
    let log = FlightLog {
        schema_version: schema_version
            .ok_or_else(|| FlightLogError::Csv("missing schema_version".into()))?,
        episode_index: episode_index
            .ok_or_else(|| FlightLogError::Csv("missing episode_index".into()))?,
        sample_rate_hz: sample_rate_hz
            .ok_or_else(|| FlightLogError::Csv("missing sample_rate_hz".into()))?,
        sigma: sigma.ok_or_else(|| FlightLogError::Csv("missing sigma".into()))?,
        samples,
    };
    validate(&log)?;
    Ok(log)
}

pub fn to_csv(log: &FlightLog) -> Result<String> {
    validate(log)?;
    let mut out = String::new();
    out.push_str(&format!("# schema_version={}\n", log.schema_version));
    out.push_str(&format!("# episode_index={}\n", log.episode_index));
    out.push_str(&format!("# sample_rate_hz={}\n", log.sample_rate_hz));
    let sigmas: Vec<String> = log.sigma.iter().map(|s| s.to_string()).collect();
    out.push_str(&format!("# sigma={}\n", sigmas.join(",")));
    out.push_str(CSV_HEADER);
    out.push('\n');
    for sample in &log.samples {
        let o = &sample.obs;
        out.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{}\n",
            sample.time_us,
            sample.phase,
            o.gyro[0],
            o.gyro[1],
            o.gyro[2],
            o.accel[0],
            o.accel[1],
            o.accel[2],
            o.height_m,
            o.pitot_dp_pa
        ));
    }
    Ok(out)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AlignConfig {
    /// Largest absolute timestamp offset still treated as the same instant.
    pub tolerance_us: u64,
    /// Drop a candidate whose wire phase differs from the log's phase code.
    pub require_phase_match: bool,
}

impl Default for AlignConfig {
    fn default() -> Self {
        Self {
            tolerance_us: HALF_TICK_US,
            require_phase_match: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Alignment {
    pub sample_index: usize,
    pub frame_index: usize,
    /// `sample.time_us - frame.time_us`.
    pub delta_us: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AlignmentReport {
    pub pairs: Vec<Alignment>,
    pub unmatched_samples: usize,
}

/// Monotone nearest-neighbour matching. Every simulated frame is consumed at
/// most once, ties resolve to the smaller frame index, and no candidate is
/// considered outside `tolerance_us`. The result is a pure function of the log
/// and episode; equal inputs give equal pairs in equal order.
pub fn align(log: &FlightLog, episode: &Episode, cfg: AlignConfig) -> Result<AlignmentReport> {
    validate(log)?;
    if cfg.tolerance_us > i64::MAX as u64 {
        return Err(FlightLogError::Alignment(
            "tolerance_us exceeds the signed-microsecond range".into(),
        ));
    }
    let frames = &episode.frames;
    let tolerance = cfg.tolerance_us as i64;
    let mut prev_t = None;
    for (i, frame) in frames.iter().enumerate() {
        let t = frame.time_us as u64;
        if prev_t.is_some_and(|p| t <= p) {
            return Err(FlightLogError::Alignment(format!(
                "simulated frames are not strictly time-ordered at index {i}"
            )));
        }
        prev_t = Some(t);
    }

    let mut pairs = Vec::new();
    let mut unmatched_samples = 0;
    let mut first = 0;
    for (sample_index, sample) in log.samples.iter().enumerate() {
        let t = sample.time_us as i64;
        let lo = t.saturating_sub(tolerance);
        let hi = t.saturating_add(tolerance);
        while first < frames.len() && (frames[first].time_us as i64) < lo {
            first += 1;
        }
        let mut best: Option<(usize, i64)> = None;
        let mut j = first;
        while j < frames.len() && (frames[j].time_us as i64) <= hi {
            let delta = t - frames[j].time_us as i64;
            // Wire phase is the alignment key; `Phase::code()` is the
            // simulator-internal six-way code and is not the wire contract.
            let phase_ok =
                !cfg.require_phase_match || sample.phase == frames[j].phase.protocol_code();
            let better = match best {
                None => true,
                Some((bj, bdelta)) => (delta.abs(), j) < (bdelta.abs(), bj),
            };
            if better && phase_ok {
                best = Some((j, delta));
            }
            j += 1;
        }
        match best {
            Some((frame_index, delta_us)) => {
                pairs.push(Alignment {
                    sample_index,
                    frame_index,
                    delta_us,
                });
                first = frame_index + 1;
            }
            None => unmatched_samples += 1,
        }
    }
    Ok(AlignmentReport {
        pairs,
        unmatched_samples,
    })
}

/// Measured minus simulated, in [`CHANNEL_NAMES`] order.
pub fn residual(frame: &Frame, sample: &FlightSample) -> [f64; CHANNEL_COUNT] {
    let o = &sample.obs;
    [
        o.gyro[0] - frame.gyro[0] as f64,
        o.gyro[1] - frame.gyro[1] as f64,
        o.gyro[2] - frame.gyro[2] as f64,
        o.accel[0] - frame.accel[0] as f64,
        o.accel[1] - frame.accel[1] as f64,
        o.accel[2] - frame.accel[2] as f64,
        o.height_m - frame.height as f64,
        o.pitot_dp_pa - frame.pitot_dp as f64,
    ]
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ChannelStats {
    pub n: usize,
    pub mean: f64,
    pub rms: f64,
    pub max_abs: f64,
}

pub fn residual_stats(
    log: &FlightLog,
    episode: &Episode,
    report: &AlignmentReport,
) -> Result<[ChannelStats; CHANNEL_COUNT]> {
    validate(log)?;
    let mut sums = [0.0_f64; CHANNEL_COUNT];
    let mut squares = [0.0_f64; CHANNEL_COUNT];
    let mut stats = [ChannelStats::default(); CHANNEL_COUNT];
    for pair in &report.pairs {
        let sample = log.samples.get(pair.sample_index).ok_or_else(|| {
            FlightLogError::Stats(format!(
                "alignment sample index {} out of range",
                pair.sample_index
            ))
        })?;
        let frame = episode.frames.get(pair.frame_index).ok_or_else(|| {
            FlightLogError::Stats(format!(
                "alignment frame index {} out of range",
                pair.frame_index
            ))
        })?;
        let r = residual(frame, sample);
        for c in 0..CHANNEL_COUNT {
            stats[c].n += 1;
            sums[c] += r[c];
            squares[c] += r[c] * r[c];
            stats[c].max_abs = stats[c].max_abs.max(r[c].abs());
        }
    }
    for c in 0..CHANNEL_COUNT {
        if stats[c].n == 0 {
            continue;
        }
        stats[c].mean = sums[c] / stats[c].n as f64;
        stats[c].rms = (squares[c] / stats[c].n as f64).sqrt();
    }
    Ok(stats)
}

/// Normalized error squared for a diagonal measurement covariance:
/// `sum((residual[c] / sigma[c])^2)` over `channels`, which is chi-square with
/// `channels.len()` degrees of freedom for zero-mean Gaussian residuals.
///
/// ponytail: diagonal covariance only; use a full matrix and a Cholesky solve
/// if cross-axis correlation ever has to be represented.
pub fn nees(
    residual: &[f64; CHANNEL_COUNT],
    sigma: &[f64; CHANNEL_COUNT],
    channels: &[usize],
) -> Result<f64> {
    if channels.is_empty() {
        return Err(FlightLogError::Stats(
            "NEES needs at least one channel".into(),
        ));
    }
    let mut seen = [false; CHANNEL_COUNT];
    for &c in channels {
        if c >= CHANNEL_COUNT {
            return Err(FlightLogError::Stats(format!(
                "channel index {c} out of range"
            )));
        }
        if seen[c] {
            return Err(FlightLogError::Stats(format!(
                "duplicate channel index {c}"
            )));
        }
        seen[c] = true;
    }
    let mut sum = 0.0;
    for &c in channels {
        let r = residual[c];
        let s = sigma[c];
        if !r.is_finite() || !s.is_finite() || s <= 0.0 {
            return Err(FlightLogError::Stats(format!(
                "non-finite residual or non-positive sigma for {}",
                CHANNEL_NAMES[c]
            )));
        }
        let z = r / s;
        sum += z * z;
    }
    if !sum.is_finite() {
        return Err(FlightLogError::Stats("NEES overflowed".into()));
    }
    Ok(sum)
}

fn ln_gamma(x: f64) -> f64 {
    const COF: [f64; 6] = [
        76.180_091_729_471_46,
        -86.505_320_329_416_77,
        24.014_098_240_830_91,
        -1.231_739_572_450_155,
        0.001_208_650_973_866_179,
        -0.000_005_395_239_384_953,
    ];
    let mut tmp = x + 5.5;
    tmp -= (x + 0.5) * tmp.ln();
    let mut ser = 1.000_000_000_190_015;
    let mut y = x;
    for &c in &COF {
        y += 1.0;
        ser += c / y;
    }
    -tmp + (2.506_628_274_631_000_5 * ser / x).ln()
}

/// Regularized upper incomplete gamma `Q(a, x)`; series below `x < a + 1`,
/// continued fraction above. No external statistics dependency.
fn gamma_q(a: f64, x: f64) -> f64 {
    const EPS: f64 = 3e-14;
    const FPMIN: f64 = 1e-300;
    if x <= 0.0 {
        return 1.0;
    }
    let gln = ln_gamma(a);
    let scale = (-x + a * x.ln() - gln).exp();
    if x < a + 1.0 {
        let mut ap = a;
        let mut sum = 1.0 / a;
        let mut del = sum;
        for _ in 0..1000 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * EPS {
                break;
            }
        }
        1.0 - sum * scale
    } else {
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / FPMIN;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..=1000 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < FPMIN {
                d = FPMIN;
            }
            c = b + an / c;
            if c.abs() < FPMIN {
                c = FPMIN;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() <= EPS {
                break;
            }
        }
        scale * h
    }
}

/// Chi-square survival function `P(X > x)` for `dof` degrees of freedom.
pub fn chi2_sf(x: f64, dof: usize) -> Result<f64> {
    if dof == 0 {
        return Err(FlightLogError::Stats("chi-square dof must be > 0".into()));
    }
    if x.is_nan() {
        return Err(FlightLogError::Stats("chi-square x is NaN".into()));
    }
    if x <= 0.0 {
        return Ok(1.0);
    }
    if x.is_infinite() {
        return Ok(0.0);
    }
    Ok(gamma_q(0.5 * dof as f64, 0.5 * x))
}

/// Lower-tail chi-square quantile, inverted from [`chi2_sf`] by bisection.
pub fn chi2_quantile(p: f64, dof: usize) -> Result<f64> {
    if dof == 0 {
        return Err(FlightLogError::Stats("chi-square dof must be > 0".into()));
    }
    if !p.is_finite() || p <= 0.0 || p >= 1.0 {
        return Err(FlightLogError::Stats(
            "chi-square p must be in (0, 1)".into(),
        ));
    }
    let target = 1.0 - p;
    let mut lo = 0.0;
    let mut hi = 1.0;
    let mut grow = 0;
    while chi2_sf(hi, dof)? > target {
        hi *= 2.0;
        grow += 1;
        if grow > 128 {
            return Err(FlightLogError::Stats(
                "chi-square bracket did not close".into(),
            ));
        }
    }
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if chi2_sf(mid, dof)? > target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(0.5 * (lo + hi))
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Corridor {
    pub dof: usize,
    pub coverage: f64,
    pub lo: f64,
    pub hi: f64,
}

impl Corridor {
    pub fn contains(self, value: f64) -> bool {
        self.lo <= value && value <= self.hi
    }
}

/// Central chi-square acceptance interval: `[Q((1-c)/2), Q((1+c)/2)]`.
pub fn nees_corridor(dof: usize, coverage: f64) -> Result<Corridor> {
    if dof == 0 {
        return Err(FlightLogError::Stats("corridor dof must be > 0".into()));
    }
    if !coverage.is_finite() || coverage <= 0.0 || coverage >= 1.0 {
        return Err(FlightLogError::Stats(
            "corridor coverage must be in (0, 1)".into(),
        ));
    }
    let tail = 0.5 * (1.0 - coverage);
    Ok(Corridor {
        dof,
        coverage,
        lo: chi2_quantile(tail, dof)?,
        hi: chi2_quantile(1.0 - tail, dof)?,
    })
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CorridorReport {
    pub total: usize,
    pub inside: usize,
    pub empirical_coverage: f64,
}

pub fn corridor_report(values: &[f64], corridor: Corridor) -> Result<CorridorReport> {
    if values.is_empty() {
        return Err(FlightLogError::Stats(
            "corridor needs at least one value".into(),
        ));
    }
    let mut inside = 0;
    for &v in values {
        if !v.is_finite() {
            return Err(FlightLogError::Stats("corridor value is not finite".into()));
        }
        if corridor.contains(v) {
            inside += 1;
        }
    }
    Ok(CorridorReport {
        total: values.len(),
        inside,
        empirical_coverage: inside as f64 / values.len() as f64,
    })
}

/// Monte-Carlo runs in the synthetic NEES corridor gate. Pinned so a change
/// to the test budget has to be a deliberate edit.
pub const SYNTHETIC_NEES_RUNS: usize = 10_000;

/// Per-channel bias of the deliberately miscalibrated estimator, in units of
/// the declared sigma. Eight channels at +2 sigma give non-centrality 32, so
/// NEES centres on 40 instead of 8 and sits far above the 95 % corridor (hi
/// ~= 17.5).
const BIASED_ESTIMATOR_BIAS_SIGMA: f64 = 2.0;

/// Outcome of [`synthetic_nees_corridor_gate`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NeesCorridorGate {
    pub runs: usize,
    pub dof: usize,
    pub coverage: f64,
    pub corridor: Corridor,
    /// Runs of the zero-mean (calibrated) estimator inside the corridor.
    pub good_inside: usize,
    /// Runs of the deliberately biased estimator inside the corridor.
    pub biased_inside: usize,
}

impl NeesCorridorGate {
    pub fn good_coverage(&self) -> f64 {
        self.good_inside as f64 / self.runs as f64
    }

    pub fn biased_coverage(&self) -> f64 {
        self.biased_inside as f64 / self.runs as f64
    }
}

/// Deterministic synthetic validation of the NEES corridor: 10^4 paired
/// Monte-Carlo runs. Each run draws eight independent standard-normal
/// residuals. The calibrated estimator uses them unchanged (NEES ~ chi2_8);
/// the biased estimator adds +2 sigma per channel (NEES ~ non-central chi2_8).
/// The pair shares its draws, so the comparison is paired and lower-variance.
/// Pure function of `seed`: no wall clock and no global RNG.
///
/// ponytail: diagonal covariance and Gaussian residuals only; a full matrix
/// and a Cholesky solve would be needed for cross-axis correlation.
/// This checks the instrument, not the vehicle — see [`validate_real_log_nees`].
pub fn synthetic_nees_corridor_gate(seed: u64, coverage: f64) -> Result<NeesCorridorGate> {
    let corridor = nees_corridor(CHANNEL_COUNT, coverage)?;
    let sigma = [1.0; CHANNEL_COUNT];
    let mut rng = StdRng::seed_from_u64(seed);
    let mut good_inside = 0usize;
    let mut biased_inside = 0usize;
    for _ in 0..SYNTHETIC_NEES_RUNS {
        let mut residual = [0.0_f64; CHANNEL_COUNT];
        for pair in residual.as_chunks_mut::<2>().0 {
            let u = rng.gen::<f64>().max(1e-12);
            let v = rng.gen::<f64>();
            let mag = (-2.0 * u.ln()).sqrt();
            let angle = std::f64::consts::TAU * v;
            pair[0] = mag * angle.cos();
            pair[1] = mag * angle.sin();
        }
        if corridor.contains(nees(&residual, &sigma, &ALL_CHANNELS)?) {
            good_inside += 1;
        }
        for c in 0..CHANNEL_COUNT {
            residual[c] += BIASED_ESTIMATOR_BIAS_SIGMA * sigma[c];
        }
        if corridor.contains(nees(&residual, &sigma, &ALL_CHANNELS)?) {
            biased_inside += 1;
        }
    }
    Ok(NeesCorridorGate {
        runs: SYNTHETIC_NEES_RUNS,
        dof: CHANNEL_COUNT,
        coverage,
        corridor,
        good_inside,
        biased_inside,
    })
}

/// Real-flight-data NEES corridor gate — LOUD by design and never a pass.
/// No real flight log with per-channel sigma ships with this repository and
/// this path is not implemented, so real-data coverage is `NOT MEASURED`.
/// The error names the gap and the escape; it is not a silent fallback and a
/// synthetic fixture must never be substituted for a measurement here.
pub fn validate_real_log_nees(_log: &FlightLog) -> Result<CorridorReport> {
    Err(FlightLogError::NotMeasured(
        "NOT MEASURED: the real-flight-data NEES corridor has not been evaluated. No real flight log with per-channel sigma has been supplied and this path is not implemented. Supply a real log and implement this function before making any flight-validation claim; synthetic_nees_corridor_gate validates the instrument only, not the vehicle."
            .into(),
    ))
}
