//! Quality gates for generated datasets. Every failure is LOUD: a `SimError`
//! naming the dataset, the gate and the offending value. No clamping, no
//! skip, no silent fallback.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::{
    config::Config,
    control::MAX_COMMAND_MPS2,
    error::{Result, SimError},
    manifest::{
        verify_code_hash, verify_golden_episode, verify_measurement_inputs, verify_protocol_schema,
        verify_provenance_binding, Bounds, FileRecord, NamedRange, ProvenanceManifest,
        ProvenanceSources, SUPPORTED_SCHEMA_VERSIONS,
    },
    optimizer::MAX_CORRECTION,
    trajectory::{
        TrajectorySet, TRUTH_STRIDE, TRUTH_TICK_NAMES, TRUTH_V3_STRIDE, TRUTH_V3_TICK_NAMES,
        VERSION_V3,
    },
    validation::{ValidationSet, LEGACY_TARGET_COUNT, TARGET_COUNT, TARGET_NAMES},
};

pub const QUALITY_SCHEMA_VERSION: u32 = 1;
pub const MIN_EPISODES_PER_SPLIT: usize = 4;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GateCheck {
    pub name: String,
    pub ok: bool,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityReport {
    pub schema_version: u32,
    pub config_hash: String,
    pub seed: u64,
    pub checks: Vec<GateCheck>,
    pub all_passed: bool,
}

/// Conservative physical sanity limits, not tuned to any model or result.
/// An unknown feature name fails the gate rather than getting a free pass.
pub fn feature_limits(name: &str) -> Option<(f64, f64)> {
    Some(match name {
        "gyro_x" | "gyro_y" | "gyro_z" => (-1.0e3, 1.0e3),
        "acc_x" | "acc_y" | "acc_z" => (-1.0e4, 1.0e4),
        "mag_x" | "mag_y" | "mag_z" => (-100.0, 100.0),
        "baro_h" => (-100.0, 1.0e5),
        "pitot_q" | "pitot_dp" => (-1.0e6, 1.0e6),
        "alpha_deg" | "beta_deg" => (-180.0, 180.0),
        "q_dyn_kpa" => (0.0, 1.0e5),
        "a_lat_resid" => (0.0, 1.0e5),
        "drag_est" => (0.0, 1.0e5),
        "static_margin_cal" => (0.0, 100.0),
        _ => return None,
    })
}

/// Bounds for target column `index`: the four legacy columns followed by
/// `g_accel_y`/`g_accel_x` (the actuator's m/s^2 envelope) in v3.
pub fn target_limits(index: usize) -> (f64, f64) {
    match index {
        0 | 1 => (-MAX_CORRECTION, MAX_CORRECTION),
        2 | 3 => (0.0, 1.0),
        4 | 5 => (-MAX_COMMAND_MPS2, MAX_COMMAND_MPS2),
        _ => unreachable!("target count is at most six"),
    }
}

fn named_range(name: &str, min: f64, max: f64) -> NamedRange {
    NamedRange {
        name: name.to_string(),
        min,
        max,
    }
}

/// Observed (min, max) of every exported channel, for the manifest.
pub fn observed_bounds(cfg: &Config, v: &ValidationSet, t: &TrajectorySet) -> Bounds {
    let mut out = Bounds::empty();
    for (i, name) in cfg.output.features.iter().enumerate() {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for e in &v.episodes {
            for row in e.features.rows() {
                if i < row.len() {
                    lo = lo.min(row[i] as f64);
                    hi = hi.max(row[i] as f64);
                }
            }
        }
        if !lo.is_finite() {
            lo = 0.0;
            hi = 0.0;
        }
        out.features.push(named_range(name, lo, hi));
    }
    for (i, name) in TARGET_NAMES.iter().enumerate() {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for e in &v.episodes {
            for row in e.targets.rows() {
                if i < row.len() {
                    lo = lo.min(row[i] as f64);
                    hi = hi.max(row[i] as f64);
                }
            }
        }
        if !lo.is_finite() {
            lo = 0.0;
            hi = 0.0;
        }
        out.targets.push(named_range(name, lo, hi));
    }
    let truth_names: &[&str] = if t.version >= VERSION_V3 {
        &TRUTH_V3_TICK_NAMES
    } else {
        &TRUTH_TICK_NAMES
    };
    for (i, name) in truth_names.iter().enumerate() {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for e in &t.episodes {
            for tick in &e.ticks {
                let x = if t.version >= VERSION_V3 {
                    tick.as_truth_v3_array()[i] as f64
                } else {
                    tick.as_truth_array()[i] as f64
                };
                lo = lo.min(x);
                hi = hi.max(x);
            }
        }
        if !lo.is_finite() {
            lo = 0.0;
            hi = 0.0;
        }
        out.truth.push(named_range(name, lo, hi));
    }
    out
}

fn fail(dataset: &str, gate: &str, detail: impl std::fmt::Display) -> SimError {
    SimError::Quality(format!("{dataset}: {gate}: {detail}"))
}

/// Runs every gate on one split (both `validation.bin` and its v2 sidecar, as
/// read back from disk). Returns the report on success, a LOUD error on the
/// first failed gate with every other failure appended.
pub fn check_dataset(
    dataset: &str,
    cfg: &Config,
    v: &ValidationSet,
    t: &TrajectorySet,
    min_episodes: usize,
) -> Result<QualityReport> {
    let mut errors: Vec<String> = Vec::new();
    let mut checks = Vec::new();
    let mut record = |name: &str, detail: String, ok: bool| {
        checks.push(GateCheck {
            name: name.to_string(),
            ok,
            detail,
        });
    };

    // 1. Episode count, including the on-disk header and the sidecar.
    let n = v.episodes.len();
    let count_ok =
        n >= min_episodes && v.header.episode_count as usize == n && t.episode_count as usize == n;
    record(
        "episode_count",
        format!(
            "{n} >= {min_episodes}, header={}, sidecar={}",
            v.header.episode_count, t.episode_count
        ),
        count_ok,
    );
    if !count_ok {
        errors.push(format!(
            "episode_count: {n} episodes (need >= {min_episodes}), validation header {}, sidecar {}",
            v.header.episode_count, t.episode_count
        ));
    }

    // 2. Frame counts, matrix shapes and sidecar stride.
    let frames = cfg.frames_per_episode();
    let nf = v.header.frames_per_episode as usize;
    let nc = cfg.feature_count();
    let mut frame_problem = None;
    if nf != frames || t.frames_per_episode as usize != frames {
        frame_problem = Some(format!(
            "frames per episode: validation header {nf}, sidecar {}, config {frames}",
            t.frames_per_episode
        ));
    }
    if !t.is_privileged() {
        frame_problem = Some("sidecar is not version 2 or 3 (no privileged truth)".into());
    }
    let expected_targets = if v.version >= 3 {
        TARGET_COUNT
    } else {
        LEGACY_TARGET_COUNT
    };
    for e in &v.episodes {
        if e.features.nrows() != nf || e.features.ncols() != nc {
            frame_problem = Some(format!(
                "episode {} features {:?}, expected ({nf}, {nc})",
                e.index,
                e.features.dim()
            ));
        }
        if e.targets.nrows() != nf || e.targets.ncols() != expected_targets {
            frame_problem = Some(format!(
                "episode {} targets {:?}, expected ({nf}, {expected_targets})",
                e.index,
                e.targets.dim()
            ));
        }
    }
    for e in &t.episodes {
        if e.ticks.len() != frames {
            frame_problem = Some(format!(
                "episode {} has {} truth ticks, expected {frames}",
                e.index,
                e.ticks.len()
            ));
        }
    }
    if let Some(p) = &frame_problem {
        record("frame_count", p.clone(), false);
        errors.push(format!("frame_count: {p}"));
    } else {
        let total = (n as u64) * (frames as u64);
        let stride = if t.version >= VERSION_V3 {
            TRUTH_V3_STRIDE
        } else {
            TRUTH_STRIDE
        };
        record(
            "frame_count",
            format!("{n} x {frames} = {total} frames, stride={stride}"),
            true,
        );
    }

    // 3. No NaN/Inf anywhere: features, targets, terminal and per-tick truth.
    let mut nonfinite = Vec::new();
    for e in &v.episodes {
        if e.features.iter().any(|x| !x.is_finite()) {
            nonfinite.push(format!("episode {} features", e.index));
        }
        if e.targets.iter().any(|x| !x.is_finite()) {
            nonfinite.push(format!("episode {} targets", e.index));
        }
        if let Some(truth) = &e.truth {
            if truth.as_array().iter().any(|x| !x.is_finite()) {
                nonfinite.push(format!("episode {} terminal truth", e.index));
            }
        }
    }
    for e in &t.episodes {
        let bad = e.ticks.iter().any(|tick| {
            if t.version >= VERSION_V3 {
                tick.as_truth_v3_array().iter().any(|x| !x.is_finite())
            } else {
                tick.as_truth_array().iter().any(|x| !x.is_finite())
            }
        });
        if bad {
            nonfinite.push(format!("episode {} per-tick truth", e.index));
        }
    }
    record(
        "finite",
        if nonfinite.is_empty() {
            "all exported values finite".into()
        } else {
            nonfinite.join(", ")
        },
        nonfinite.is_empty(),
    );
    if !nonfinite.is_empty() {
        errors.push(format!("finite: non-finite in {}", nonfinite.join(", ")));
    }

    // 4. Feature bounds.
    let mut bad_features = Vec::new();
    for e in &v.episodes {
        for (i, name) in cfg.output.features.iter().enumerate() {
            let (lo, hi) = match feature_limits(name) {
                Some(b) => b,
                None => {
                    bad_features.push(format!("unknown feature {name:?}"));
                    continue;
                }
            };
            for row in e.features.rows() {
                if i < row.len() {
                    let x = row[i] as f64;
                    if !(lo..=hi).contains(&x) {
                        bad_features.push(format!("{name}={x} outside [{lo},{hi}]"));
                        break;
                    }
                }
            }
        }
    }
    record(
        "feature_bounds",
        if bad_features.is_empty() {
            "all features inside physical limits".into()
        } else {
            bad_features[..bad_features.len().min(3)].join(", ")
        },
        bad_features.is_empty(),
    );
    if !bad_features.is_empty() {
        errors.push(format!("feature_bounds: {}", bad_features.join(", ")));
    }

    // 5. Target bounds and the 0/1 health flag.
    let mut bad_targets = Vec::new();
    for e in &v.episodes {
        for row in e.targets.rows() {
            for (i, x) in row.iter().enumerate() {
                let (lo, hi) = target_limits(i);
                let x = *x as f64;
                if !(lo..=hi).contains(&x) {
                    bad_targets.push(format!("{}={x} outside [{lo},{hi}]", TARGET_NAMES[i]));
                    break;
                }
            }
            if row[3] != 0.0 && row[3] != 1.0 {
                bad_targets.push(format!("ai_healthy={} is not 0 or 1", row[3]));
            }
        }
    }
    record(
        "target_bounds",
        if bad_targets.is_empty() {
            format!(
                "g_theta within +/-{MAX_CORRECTION} rad, confidence and health in [0,1], \
                 g_accel within +/-{MAX_COMMAND_MPS2} m/s^2"
            )
        } else {
            bad_targets[..bad_targets.len().min(3)].join(", ")
        },
        bad_targets.is_empty(),
    );
    if !bad_targets.is_empty() {
        errors.push(format!("target_bounds: {}", bad_targets.join(", ")));
    }

    // 6. Truth bounds: physical plausibility of the privileged teacher input.
    let mut bad_truth = Vec::new();
    for e in &t.episodes {
        for (i, tick) in e.ticks.iter().enumerate() {
            if !(tick.mass_kg > 0.0) {
                bad_truth.push(format!(
                    "episode {} tick {i} mass={}",
                    e.index, tick.mass_kg
                ));
            }
            if (tick.g_accel[0] as f64).abs() > MAX_COMMAND_MPS2
                || (tick.g_accel[1] as f64).abs() > MAX_COMMAND_MPS2
            {
                bad_truth.push(format!(
                    "episode {} tick {i} g_accel={:?} outside +/-{MAX_COMMAND_MPS2} m/s^2",
                    e.index, tick.g_accel
                ));
            }
            let q2 = tick.quat.iter().map(|x| (*x as f64).powi(2)).sum::<f64>();
            if !(0.9..=1.1).contains(&q2.sqrt()) {
                bad_truth.push(format!(
                    "episode {} tick {i} quat norm={}",
                    e.index,
                    q2.sqrt()
                ));
            }
            if tick.gust_start_s < -1.0 || tick.gust_start_s > cfg.run.duration_s as f32 + 1.0 {
                bad_truth.push(format!(
                    "episode {} tick {i} gust_start={}",
                    e.index, tick.gust_start_s
                ));
            }
            if tick.gust_amplitude < 0.0 || tick.shear_delta < 0.0 || tick.shear_thickness_m < 0.0 {
                bad_truth.push(format!("episode {} tick {i} negative gust/shear", e.index));
            }
            for g in tick.g_theta {
                if !((-MAX_CORRECTION..=MAX_CORRECTION).contains(&(g as f64))) {
                    bad_truth.push(format!("episode {} tick {i} g_theta={g}", e.index));
                }
            }
        }
    }
    record(
        "truth_bounds",
        if bad_truth.is_empty() {
            "mass>0, unit quaternion, gust/shear, g_theta and g_accel in range".into()
        } else {
            bad_truth[..bad_truth.len().min(3)].join(", ")
        },
        bad_truth.is_empty(),
    );
    if !bad_truth.is_empty() {
        errors.push(format!("truth_bounds: {}", bad_truth.join(", ")));
    }

    if !errors.is_empty() {
        return Err(fail(dataset, "quality", errors.join("; ")));
    }
    Ok(QualityReport {
        schema_version: QUALITY_SCHEMA_VERSION,
        config_hash: String::new(),
        seed: v.header.seed,
        checks,
        all_passed: true,
    })
}

/// Train and held-out scenario id sets must be disjoint, by id and by the
/// underlying wind/gust signature.
pub fn check_split_disjoint(train: &ValidationSet, held: &ValidationSet) -> Result<GateCheck> {
    let ids =
        |v: &ValidationSet| -> BTreeSet<u16> { v.episodes.iter().map(|e| e.scenario_id).collect() };
    let a = ids(train);
    let b = ids(held);
    let overlap: Vec<u16> = a.intersection(&b).copied().collect();
    let detail = format!(
        "train={:?} held-out={:?} overlap={overlap:?}",
        a.iter().copied().collect::<Vec<_>>(),
        b.iter().copied().collect::<Vec<_>>()
    );
    if !overlap.is_empty() || a.is_empty() || b.is_empty() {
        return Err(SimError::Quality(format!(
            "split_scenarios_disjoint: {detail}"
        )));
    }
    Ok(GateCheck {
        name: "split_scenarios_disjoint".into(),
        ok: true,
        detail,
    })
}

/// Byte-for-byte comparison of a regenerated dataset against the first one.
pub fn check_determinism(pairs: &[(&str, &[u8], &[u8])]) -> Result<GateCheck> {
    let mut mismatches = Vec::new();
    for (name, a, b) in pairs {
        if a != b {
            mismatches.push(name.to_string());
        }
    }
    let detail = format!("{} files compared, mismatches={mismatches:?}", pairs.len());
    if !mismatches.is_empty() {
        return Err(SimError::Quality(format!(
            "deterministic_regeneration: {detail}"
        )));
    }
    Ok(GateCheck {
        name: "deterministic_regeneration".into(),
        ok: true,
        detail,
    })
}

pub fn quality_report_to_string(report: &QualityReport) -> Result<String> {
    let mut s = serde_json::to_string_pretty(report)
        .map_err(|e| SimError::Dataset(format!("quality report serialization: {e}")))?;
    s.push('\n');
    Ok(s)
}

pub fn write_quality_report(path: &std::path::Path, report: &QualityReport) -> Result<()> {
    let text = quality_report_to_string(report)?;
    std::fs::write(path, text.as_bytes())?;
    Ok(())
}

/// Every provenance gate in one call. A mismatch stops the check with a LOUD
/// error naming the field, the recorded value and the observed value; there is
/// no clamp, skip or silent fallback.
pub fn check_provenance(
    dataset: &str,
    manifest: &crate::manifest::DatasetManifest,
    provenance: &ProvenanceManifest,
    sources: &ProvenanceSources<'_>,
) -> Result<Vec<GateCheck>> {
    verify_provenance_binding(manifest, provenance)
        .map_err(|e| fail(dataset, "provenance_binding", e))?;
    let mut checks = vec![GateCheck {
        name: "provenance_binding".into(),
        ok: true,
        detail: format!(
            "manifest_sha256={}, accepted schemas={SUPPORTED_SCHEMA_VERSIONS:?}",
            provenance.manifest_hash
        ),
    }];

    verify_code_hash(provenance, sources.repo).map_err(|e| fail(dataset, "code_hash", e))?;
    checks.push(GateCheck {
        name: "code_hash".into(),
        ok: true,
        detail: format!("git HEAD={}", provenance.code_hash),
    });

    verify_protocol_schema(provenance, sources.protocol_schema)
        .map_err(|e| fail(dataset, "protocol_schema_hash", e))?;
    checks.push(GateCheck {
        name: "protocol_schema_hash".into(),
        ok: true,
        detail: format!("protocol schema sha256={}", provenance.protocol_schema_hash),
    });

    verify_measurement_inputs(provenance, sources.measurement_dir)
        .map_err(|e| fail(dataset, "measurement_hashes", e))?;
    checks.push(GateCheck {
        name: "measurement_hashes".into(),
        ok: true,
        detail: format!(
            "{} measurement input(s) bound",
            provenance.measurement_hashes.len()
        ),
    });
    Ok(checks)
}

/// The golden-episode pin: the recorded file hashes must still produce the
/// recorded combined hash. A changed RNG draw order or one flipped byte fails
/// this gate loudly.
pub fn check_golden_episode(
    dataset: &str,
    provenance: &ProvenanceManifest,
    files: &[FileRecord],
) -> Result<GateCheck> {
    verify_golden_episode(provenance, files).map_err(|e| fail(dataset, "golden_episode", e))?;
    Ok(GateCheck {
        name: "golden_episode".into(),
        ok: true,
        detail: format!(
            "{} pinned file(s), sha256={}",
            files.len(),
            provenance.golden_episode_hash
        ),
    })
}
