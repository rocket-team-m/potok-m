// SPDX-License-Identifier: GPL-3.0-or-later
//! SIMLOG — gates for the proposed flight-log replay seam.
//!
//! No real flight log exists. Every fixture here is synthetic: a simulated
//! episode's frames, perturbed by known constants or seeded Gaussian draws.
//! A green run proves the schema/import/alignment/statistics code is coherent;
//! it is **not** validation of the vehicle, the sensors or a real recording.
//!
//! The production module is included by path rather than via `lib.rs`, so this
//! lane edits only its two assigned files. The two shim modules below supply
//! the `crate::episode` / `crate::sensor` paths the module uses when it is
//! installed as `potok_sim::flight_replay`.

pub mod episode {
    pub use potok_sim::episode::Episode;
}
pub mod sensor {
    pub use potok_sim::sensor::Frame;
}

#[path = "../src/flight_replay.rs"]
mod flight_replay;

use flight_replay::{
    align, chi2_quantile, chi2_sf, corridor_report, from_csv, from_json, nees, nees_corridor,
    residual, residual_stats, to_csv, to_json, validate, AlignConfig, Alignment, AlignmentReport,
    CorridorReport, FlightLog, FlightLogError, FlightSample, Observation, ALL_CHANNELS,
    CHANNEL_COUNT, CHANNEL_NAMES, CHANNEL_UNITS, CSV_HEADER, HALF_TICK_US, SCHEMA_VERSION,
};
use potok_sim::{Config, Episode, Frame, Phase, Simulator};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn short_episode() -> Episode {
    let mut cfg = Config::default();
    cfg.run.seed = 1337;
    cfg.run.duration_s = 1.0;
    Simulator::new(cfg)
        .expect("default config must load")
        .run_episode(0)
        .expect("episode must complete")
}

fn sigma() -> [f64; CHANNEL_COUNT] {
    [0.0015, 0.0015, 0.0015, 0.011, 0.011, 0.011, 0.5, 200.0]
}

fn obs_of(frame: &Frame, offset: [f64; CHANNEL_COUNT]) -> Observation {
    Observation {
        gyro: [
            frame.gyro[0] as f64 + offset[0],
            frame.gyro[1] as f64 + offset[1],
            frame.gyro[2] as f64 + offset[2],
        ],
        accel: [
            frame.accel[0] as f64 + offset[3],
            frame.accel[1] as f64 + offset[4],
            frame.accel[2] as f64 + offset[5],
        ],
        height_m: frame.height as f64 + offset[6],
        pitot_dp_pa: frame.pitot_dp as f64 + offset[7],
    }
}

fn synthetic_log(ep: &Episode, offset: [f64; CHANNEL_COUNT]) -> FlightLog {
    FlightLog {
        schema_version: SCHEMA_VERSION,
        episode_index: ep.meta.index,
        sample_rate_hz: 50,
        sigma: sigma(),
        samples: ep
            .frames
            .iter()
            .map(|f| FlightSample {
                time_us: f.time_us as u64,
                phase: f.phase.protocol_code(),
                obs: obs_of(f, offset),
            })
            .collect(),
    }
}

fn replace_first_field(line: &str, value: &str) -> String {
    let comma = line.find(',').expect("CSV row has a comma");
    format!("{value}{}", &line[comma..])
}

#[test]
fn schema_constants_pin_the_wire_contract() {
    assert_eq!(SCHEMA_VERSION, 1);
    assert_eq!(
        CHANNEL_NAMES,
        [
            "gyro_x",
            "gyro_y",
            "gyro_z",
            "accel_x",
            "accel_y",
            "accel_z",
            "height_m",
            "pitot_dp_pa"
        ]
    );
    assert_eq!(CHANNEL_UNITS[0], "rad/s");
    assert_eq!(CHANNEL_UNITS[3], "m/s^2");
    assert_eq!(CHANNEL_UNITS[6], "m");
    assert_eq!(CHANNEL_UNITS[7], "Pa");
    assert_eq!(ALL_CHANNELS, [0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(
        CSV_HEADER,
        "time_us,phase,gyro_x,gyro_y,gyro_z,accel_x,accel_y,accel_z,height_m,pitot_dp_pa"
    );
    assert_eq!(HALF_TICK_US, 10_000);

    // Alignment keys are the canonical wire codes, not the six-way simulator
    // phase: Pad and Boost both collapse to 0.
    assert_eq!(Phase::Pad.protocol_code(), 0);
    assert_eq!(Phase::Boost.protocol_code(), 0);
    assert_eq!(Phase::Coast.protocol_code(), 1);
    assert_eq!(Phase::Apogee.protocol_code(), 2);
    assert_eq!(Phase::Descent.protocol_code(), 3);
    assert_eq!(Phase::Landed.protocol_code(), 4);
}

#[test]
fn json_round_trip_is_deterministic_and_strict() {
    let ep = short_episode();
    let log = synthetic_log(&ep, [0.0; CHANNEL_COUNT]);
    let a = to_json(&log).unwrap();
    assert_eq!(a, to_json(&log).unwrap(), "JSON depends on process state");
    assert!(a.contains("\"schema_version\": 1"));
    assert_eq!(from_json(&a).unwrap(), log);

    let bad = a.replace(
        "\"schema_version\": 1",
        "\"schema_version\": 1,\"extra\": 0",
    );
    assert!(from_json(&bad).is_err(), "unknown field was admitted");
    let bad = a.replace("\"schema_version\": 1", "\"schema_version\": 2");
    assert!(matches!(
        from_json(&bad),
        Err(FlightLogError::UnsupportedVersion { got: 2, want: 1 })
    ));

    let mut bad = log.clone();
    bad.sigma[0] = -1.0;
    assert!(validate(&bad).is_err());
    assert!(to_json(&bad).is_err());
    let mut bad = log.clone();
    bad.samples[1].time_us = bad.samples[0].time_us;
    assert!(validate(&bad).is_err(), "duplicate timestamp admitted");
    let mut bad = log.clone();
    bad.samples[0].phase = 5;
    assert!(validate(&bad).is_err(), "out-of-range phase admitted");
    let mut bad = log.clone();
    bad.samples[0].obs.gyro[1] = f64::NAN;
    assert!(validate(&bad).is_err(), "NaN measurement admitted");
    let mut bad = log;
    bad.samples.clear();
    assert!(validate(&bad).is_err(), "empty log admitted");
}

#[test]
fn csv_round_trip_matches_json_and_is_strict() {
    let ep = short_episode();
    let log = synthetic_log(&ep, [0.0; CHANNEL_COUNT]);
    let csv = to_csv(&log).unwrap();
    assert_eq!(csv, to_csv(&log).unwrap(), "CSV depends on process state");
    assert_eq!(
        csv.lines().count(),
        5 + log.samples.len(),
        "4 meta + header + rows (lines() drops the trailing empty line)"
    );
    let parsed = from_csv(&csv).unwrap();
    assert_eq!(parsed, log);
    assert_eq!(parsed, from_json(&to_json(&log).unwrap()).unwrap());

    let mut lines: Vec<String> = csv.lines().map(str::to_string).collect();
    lines[1] = lines[1].replacen("episode_index=", "sample_rate_hz=", 1);
    assert!(
        from_csv(&lines.join("\n")).is_err(),
        "duplicate metadata key"
    );

    let mut lines: Vec<String> = csv.lines().map(str::to_string).collect();
    lines[4] = "time_us,phase".to_string();
    assert!(from_csv(&lines.join("\n")).is_err(), "bad CSV header");

    let mut lines: Vec<String> = csv.lines().map(str::to_string).collect();
    lines[5].push_str(",0");
    assert!(from_csv(&lines.join("\n")).is_err(), "wrong column count");

    let mut lines: Vec<String> = csv.lines().map(str::to_string).collect();
    let t5 = lines[5].split(',').next().unwrap().to_string();
    let t6 = lines[6].split(',').next().unwrap().to_string();
    lines[5] = replace_first_field(&lines[5], &t6);
    lines[6] = replace_first_field(&lines[6], &t5);
    assert!(
        from_csv(&lines.join("\n")).is_err(),
        "non-monotonic timestamps"
    );

    let mut lines: Vec<String> = csv.lines().map(str::to_string).collect();
    lines[5] = replace_first_field(&lines[5], "not-a-number");
    assert!(from_csv(&lines.join("\n")).is_err(), "non-numeric field");

    let mut lines: Vec<String> = csv.lines().map(str::to_string).collect();
    lines.remove(0);
    assert!(
        from_csv(&lines.join("\n")).is_err(),
        "missing schema_version"
    );
}

#[test]
fn alignment_uses_time_and_wire_phase_without_reusing_frames() {
    let ep = short_episode();
    let base = ep.frames[0].clone();
    let frames: Vec<Frame> = (0..20u32)
        .map(|i| {
            let mut f = base.clone();
            f.time_us = i * 20_000;
            f.phase = if i < 5 {
                Phase::Boost
            } else if i < 12 {
                Phase::Coast
            } else {
                Phase::Descent
            };
            f.gyro[0] = i as f32;
            f
        })
        .collect();
    let mut sim = ep.clone();
    sim.frames = frames;

    let zero = Observation {
        gyro: [0.0; 3],
        accel: [0.0; 3],
        height_m: 0.0,
        pitot_dp_pa: 0.0,
    };
    let samples = vec![
        FlightSample {
            time_us: 0,
            phase: 0,
            obs: zero,
        },
        FlightSample {
            time_us: 103_000,
            phase: 1,
            obs: zero,
        },
        FlightSample {
            time_us: 200_000,
            phase: 3,
            obs: zero,
        },
        FlightSample {
            time_us: 255_000,
            phase: 4,
            obs: zero,
        },
    ];
    let log = FlightLog {
        samples,
        ..synthetic_log(&ep, [0.0; CHANNEL_COUNT])
    };

    let report = align(&log, &sim, AlignConfig::default()).unwrap();
    assert_eq!(
        report,
        AlignmentReport {
            pairs: vec![
                Alignment {
                    sample_index: 0,
                    frame_index: 0,
                    delta_us: 0
                },
                Alignment {
                    sample_index: 1,
                    frame_index: 5,
                    delta_us: 3_000
                },
            ],
            unmatched_samples: 2
        }
    );
    assert_eq!(align(&log, &sim, AlignConfig::default()).unwrap(), report);

    let no_phase = AlignConfig {
        require_phase_match: false,
        ..AlignConfig::default()
    };
    let relaxed = align(&log, &sim, no_phase).unwrap();
    assert_eq!(relaxed.pairs.len(), 4);
    assert_eq!(relaxed.pairs[2].frame_index, 10);
    // The last sample is 15 ms past frame 12 but only 5 ms before frame 13;
    // with the phase gate off it matches frame 13 instead.
    assert_eq!(relaxed.pairs[3].frame_index, 13);
    assert_eq!(relaxed.pairs[3].delta_us, -5_000);
    assert_eq!(relaxed.unmatched_samples, 0);

    let exact = AlignConfig {
        tolerance_us: 0,
        ..AlignConfig::default()
    };
    let exact_report = align(&log, &sim, exact).unwrap();
    assert_eq!(exact_report.pairs.len(), 1);
    assert_eq!(exact_report.unmatched_samples, 3);

    let two = FlightLog {
        samples: vec![
            FlightSample {
                time_us: 40_000,
                phase: 0,
                obs: zero,
            },
            FlightSample {
                time_us: 42_000,
                phase: 0,
                obs: zero,
            },
        ],
        ..log.clone()
    };
    let no_reuse = align(&two, &sim, AlignConfig::default()).unwrap();
    assert_eq!(no_reuse.pairs.len(), 1, "one frame was matched twice");
    assert_eq!(no_reuse.unmatched_samples, 1);

    let mut broken = sim.clone();
    broken.frames.swap(0, 1);
    assert!(align(&log, &broken, AlignConfig::default()).is_err());
    let mut bad_log = log.clone();
    bad_log.samples[0].time_us = u64::MAX;
    assert!(align(&bad_log, &sim, AlignConfig::default()).is_err());
    let huge_tolerance = AlignConfig {
        tolerance_us: u64::MAX,
        ..AlignConfig::default()
    };
    assert!(align(&log, &sim, huge_tolerance).is_err());
}

#[test]
fn residual_stats_recover_the_injected_constant_offset() {
    let ep = short_episode();
    let offset = [1e-4, -2e-4, 3e-4, 0.01, -0.02, 0.03, 0.25, -5.0];
    let log = synthetic_log(&ep, offset);
    let report = align(&log, &ep, AlignConfig::default()).unwrap();
    assert_eq!(report.pairs.len(), ep.frames.len());
    assert_eq!(report.unmatched_samples, 0);

    let stats = residual_stats(&log, &ep, &report).unwrap();
    for c in 0..CHANNEL_COUNT {
        assert_eq!(stats[c].n, ep.frames.len(), "channel {}", CHANNEL_NAMES[c]);
        assert!(
            (stats[c].mean - offset[c]).abs() < 1e-9,
            "{} mean {} != {}",
            CHANNEL_NAMES[c],
            stats[c].mean,
            offset[c]
        );
        assert!((stats[c].rms - offset[c].abs()).abs() < 1e-9);
        assert!((stats[c].max_abs - offset[c].abs()).abs() < 1e-9);
    }
    let first = residual(&ep.frames[report.pairs[0].frame_index], &log.samples[0]);
    for c in 0..CHANNEL_COUNT {
        assert!((first[c] - offset[c]).abs() < 1e-9);
    }

    let bogus = Alignment {
        sample_index: log.samples.len(),
        frame_index: 0,
        delta_us: 0,
    };
    let bad = AlignmentReport {
        pairs: vec![bogus],
        unmatched_samples: 0,
    };
    assert!(residual_stats(&log, &ep, &bad).is_err());
}

#[test]
fn nees_is_the_diagonal_mahalanobis_sum_and_rejects_bad_inputs() {
    let r = [1.0, 2.0, 0.0, -1.0, 0.0, 0.0, 3.0, -2.0];
    let s = [1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 3.0, 2.0];
    assert_eq!(nees(&r, &s, &ALL_CHANNELS).unwrap(), 5.0);
    assert_eq!(nees(&r, &s, &[0, 1, 6]).unwrap(), 3.0);
    assert!(nees(&r, &s, &[]).is_err());
    assert!(nees(&r, &s, &[0, 0]).is_err());
    assert!(nees(&r, &s, &[CHANNEL_COUNT]).is_err());
    let mut bad = s;
    bad[2] = 0.0;
    assert!(nees(&r, &bad, &ALL_CHANNELS).is_err());
    let mut bad = r;
    bad[0] = f64::NAN;
    assert!(nees(&bad, &s, &ALL_CHANNELS).is_err());
}

#[test]
fn chi2_helpers_match_published_quantiles() {
    for (p, dof, want) in [
        (0.95_f64, 8_usize, 15.5073_f64),
        (0.05, 8, 2.7326),
        (0.50, 1, 0.45494),
        (0.95, 1, 3.84146),
        (0.99, 4, 13.2767),
        (0.975, 2, 7.37776),
    ] {
        let got = chi2_quantile(p, dof).unwrap();
        assert!(
            (got - want).abs() < 2e-3,
            "p={p} dof={dof}: {got} vs {want}"
        );
        let sf = chi2_sf(got, dof).unwrap();
        assert!((sf - (1.0 - p)).abs() < 1e-6, "p={p} dof={dof}: sf={sf}");
    }
    let corridor = nees_corridor(8, 0.95).unwrap();
    assert_eq!(corridor.dof, 8);
    assert_eq!(corridor.coverage, 0.95);
    // A central 95 % corridor leaves 2.5 % in each tail, not 5 %.
    assert!((corridor.lo - 2.1797).abs() < 2e-3);
    assert!((corridor.hi - 17.5345).abs() < 2e-3);
    assert!(!corridor.contains(2.0));
    assert!(corridor.contains(8.0));
    assert!(!corridor.contains(18.0));

    assert_eq!(chi2_sf(0.0, 8).unwrap(), 1.0);
    assert_eq!(chi2_sf(f64::INFINITY, 8).unwrap(), 0.0);
    assert!(chi2_sf(1.0, 0).is_err());
    assert!(chi2_sf(f64::NAN, 8).is_err());
    assert!(chi2_quantile(0.0, 8).is_err());
    assert!(chi2_quantile(1.0, 8).is_err());
    assert!(nees_corridor(8, 1.0).is_err());
    assert!(corridor_report(&[], corridor).is_err());
    assert!(corridor_report(&[f64::NAN], corridor).is_err());
}

#[test]
fn corridor_covers_seeded_gaussian_nees_at_its_nominal_rate() {
    // A statistical self-check of the χ² helper, not a flight-data claim:
    // eight independent standard-normal residuals have NEES ~ χ²_8.
    let corridor = nees_corridor(CHANNEL_COUNT, 0.95).unwrap();
    let mut rng = StdRng::seed_from_u64(0x5A17);
    let mut inside = 0;
    let n = 2_000;
    for _ in 0..n {
        let mut r = [0.0_f64; CHANNEL_COUNT];
        for pair in r.as_chunks_mut::<2>().0 {
            let u1 = rng.gen::<f64>().max(1e-12);
            let u2 = rng.gen::<f64>();
            let mag = (-2.0 * u1.ln()).sqrt();
            let angle = std::f64::consts::TAU * u2;
            pair[0] = mag * angle.cos();
            pair[1] = mag * angle.sin();
        }
        if corridor.contains(nees(&r, &[1.0; CHANNEL_COUNT], &ALL_CHANNELS).unwrap()) {
            inside += 1;
        }
    }
    let report = corridor_report(&[corridor.lo, corridor.hi], corridor).unwrap();
    assert_eq!(
        report,
        CorridorReport {
            total: 2,
            inside: 2,
            empirical_coverage: 1.0
        }
    );
    let empirical = inside as f64 / n as f64;
    assert!(
        (0.93..=0.97).contains(&empirical),
        "synthetic Gaussian coverage {empirical} outside [0.93, 0.97]"
    );
}

#[test]
fn end_to_end_synthetic_replay_through_both_encodings() {
    // Offsets equal the declared sigmas, so the synthetic NEES is 8 on every
    // aligned tick. This is a fixture with a known answer, not a fidelity test.
    let ep = short_episode();
    let log = synthetic_log(&ep, sigma());
    let json = to_json(&log).unwrap();
    let csv = to_csv(&log).unwrap();
    let imported = [from_json(&json).unwrap(), from_csv(&csv).unwrap()];
    assert_eq!(imported[0], imported[1]);
    assert_eq!(imported[0], log);

    for decoded in &imported {
        let report = align(decoded, &ep, AlignConfig::default()).unwrap();
        assert_eq!(report.pairs.len(), ep.frames.len());
        assert_eq!(report.unmatched_samples, 0);
        let stats = residual_stats(decoded, &ep, &report).unwrap();
        for c in 0..CHANNEL_COUNT {
            assert!((stats[c].mean - decoded.sigma[c]).abs() < 1e-9);
        }
        let values: Vec<f64> = report
            .pairs
            .iter()
            .map(|pair| {
                nees(
                    &residual(
                        &ep.frames[pair.frame_index],
                        &decoded.samples[pair.sample_index],
                    ),
                    &decoded.sigma,
                    &ALL_CHANNELS,
                )
                .unwrap()
            })
            .collect();
        assert!(values.iter().all(|v| (v - 8.0).abs() < 1e-6));
        let corridor = nees_corridor(CHANNEL_COUNT, 0.95).unwrap();
        let covered = corridor_report(&values, corridor).unwrap();
        assert_eq!(covered.total, values.len());
        assert_eq!(
            covered.inside, covered.total,
            "synthetic NEES=8 must be inside"
        );
    }
}
