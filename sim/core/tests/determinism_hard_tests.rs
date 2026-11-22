// SPDX-License-Identifier: GPL-3.0-or-later
//! SIM05 — hard determinism gates for seeded episode regeneration.
//!
//! The simulator promises four things and this file pins all four without
//! touching a source module:
//!   1. same (seed, config, index) regenerates every field bit-for-bit,
//!      including across two independently constructed `Simulator`s;
//!   2. the four named substreams (`INIT`, `DOMA`, `WIND`, `SENS`) are distinct
//!      SplitMix64 keys and each one moves when `run.seed` moves;
//!   3. no global RNG exists — draining `thread_rng`, changing call order,
//!      interleaving seeds and running under rayon cannot move a single bit;
//!   4. the on-disk validation file and its v1/v2/v3 truth sidecars are
//!      byte-identical when regenerated from the same seed/config.
//!
//! A red test here means the determinism contract regressed, not that the
//! data changed legitimately. No compatibility fallback is acceptable.

use std::{
    collections::HashSet,
    path::{Path, PathBuf},
};

use potok_sim::{
    config::{sample_domain, Range},
    manifest::sha256_hex,
    math::episode_seed,
    quality::check_determinism,
    validation::write_validation,
    Config, Episode, Simulator, Wind,
};
use rand::Rng;

/// Domain-separation constants from the seeded call sites: `INIT` in
/// `episode.rs`, `DOMA` in `config.rs`, `WIND` in `wind.rs`, `SENS` in
/// `sensor.rs`. A collision between them would alias two substreams.
const INIT: u64 = 0x494E_4954;
const DOMA: u64 = 0x444F_4D41;
const WIND: u64 = 0x5749_4E44;
const SENS: u64 = 0x5345_4E53;
const SUBSTREAMS: [(&str, u64); 4] = [
    ("INIT", INIT),
    ("DOMA", DOMA),
    ("WIND", WIND),
    ("SENS", SENS),
];

fn cfg(seed: u64) -> Config {
    let mut c = Config::default();
    c.run.seed = seed;
    c
}

fn run(seed: u64, index: u64) -> Episode {
    Simulator::new(cfg(seed))
        .expect("default config must load")
        .run_episode(index)
        .expect("episode must complete")
}

fn push_f32(out: &mut Vec<u8>, v: f32) {
    out.extend_from_slice(&v.to_bits().to_le_bytes());
}
fn push_f64(out: &mut Vec<u8>, v: f64) {
    out.extend_from_slice(&v.to_bits().to_le_bytes());
}
fn push_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn push_f32s(out: &mut Vec<u8>, values: &[f32]) {
    for &v in values {
        push_f32(out, v);
    }
}
fn push_opt_f64(out: &mut Vec<u8>, v: Option<f64>) {
    match v {
        Some(x) => {
            out.push(1);
            push_f64(out, x);
        }
        None => out.push(0),
    }
}

/// Every observable of an episode, reduced to raw little-endian bits. `to_bits`
/// keeps `-0.0` distinct from `+0.0` and NaNs distinct from each other; `==`
/// on floats would not. Wire bytes are included verbatim, plus the sim-only
/// frame context that the 56-byte UART frame does not carry.
fn episode_bits(e: &Episode) -> Vec<u8> {
    let mut out = Vec::new();
    push_u64(&mut out, e.meta.index);
    push_u64(&mut out, e.meta.seed);
    push_f64(&mut out, e.meta.mass_kg);
    push_f64(&mut out, e.meta.thrust_scale);
    push_f64(&mut out, e.meta.initial_pitch_deg);
    push_f64(&mut out, e.meta.initial_yaw_deg);
    for v in e.meta.gimbal_deg {
        push_f64(&mut out, v);
    }
    push_opt_f64(&mut out, e.meta.gust_start_s);
    push_f64(&mut out, e.meta.gust_duration_s);
    push_f64(&mut out, e.meta.gust_amplitude);
    push_f64(&mut out, e.meta.v_10);
    push_f64(&mut out, e.meta.shear_delta);
    push_u64(&mut out, e.meta.scenario_id as u64);
    out.push(e.meta.split.as_u8());
    push_f32s(&mut out, &e.meta.conditioning.as_array());

    push_u64(&mut out, e.frames_generated as u64);
    for f in &e.frames {
        out.extend_from_slice(&f.to_bytes());
        push_u64(&mut out, f.crc as u64);
        push_f32s(&mut out, &f.physics);
        for flag in [f.gust, f.shear, f.saturated, f.spike, f.pitot_valid] {
            out.push(flag as u8);
        }
    }
    for t in &e.targets {
        push_f32s(&mut out, &t.g_theta);
        push_f32(&mut out, t.confidence);
        out.push(t.ai_healthy as u8);
        push_f32s(&mut out, &t.g_accel);
    }
    for &v in &e.features {
        push_f32(&mut out, v);
    }
    for tick in &e.trajectory {
        push_f32s(&mut out, &tick.as_truth_v3_array());
    }
    for &c in &e.phase_counts {
        push_u64(&mut out, c as u64);
    }
    push_u64(&mut out, e.invalid_frames as u64);
    push_u64(&mut out, e.nan_frames as u64);
    push_f64(&mut out, e.apogee_m);
    push_f64(&mut out, e.max_speed_mps);
    match e.truth {
        Some(t) => {
            out.push(1);
            push_f32s(&mut out, &t.as_array());
        }
        None => out.push(0),
    }
    out
}

fn meta_bits(e: &Episode) -> Vec<u8> {
    let mut out = Vec::new();
    push_u64(&mut out, e.meta.index); // seed deliberately excluded
    push_f64(&mut out, e.meta.mass_kg);
    push_f64(&mut out, e.meta.thrust_scale);
    push_u64(
        &mut out,
        e.meta.gust_start_s.map(f64::to_bits).unwrap_or(u64::MAX),
    );
    push_f64(&mut out, e.meta.v_10);
    push_f64(&mut out, e.meta.shear_delta);
    push_f32s(&mut out, &e.meta.conditioning.as_array());
    out
}

fn assert_bits_same(label: &str, a: &[u8], b: &[u8]) {
    assert_eq!(a.len(), b.len(), "{label}: fingerprint length changed");
    if let Some((i, (x, y))) = a
        .iter()
        .zip(b.iter())
        .enumerate()
        .find(|(_, (x, y))| x != y)
    {
        panic!(
            "{label}: first bit-level divergence at byte {i}: {x:#04x} vs {y:#04x} \
             (sha256 {} vs {})",
            sha256_hex(a),
            sha256_hex(b)
        );
    }
}

fn assert_bytes_same(label: &str, a: &[u8], b: &[u8]) {
    assert_eq!(a.len(), b.len(), "{label}: file length changed");
    if let Some((i, (x, y))) = a
        .iter()
        .zip(b.iter())
        .enumerate()
        .find(|(_, (x, y))| x != y)
    {
        panic!(
            "{label}: first byte divergence at {i}: {x:#04x} vs {y:#04x} \
             (sha256 {} vs {})",
            sha256_hex(a),
            sha256_hex(b)
        );
    }
}

fn temp(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("potok-sim05-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");
    dir
}

/// Collapse every config-driven envelope so only the SENS substream can move
/// the generated frames. `Wind` noise and gusts are disabled as well.
fn frozen_cfg(seed: u64) -> Config {
    let mut c = cfg(seed);
    c.initial.mass_kg = Range(11.0, 11.0);
    c.initial.cg_offset_m = Range(0.0, 0.0);
    c.initial.thrust_scale = Range(1.0, 1.0);
    c.initial.pitch_deg = Range(0.0, 0.0);
    c.initial.yaw_deg = Range(0.0, 0.0);
    c.initial.gimbal_deg = Range(0.0, 0.0);
    c.aero.cd0_range = Range(0.30, 0.30);
    c.aero.static_margin_cal_range = Range(0.65, 0.65);
    c.sensor.gyro_noise_range = Range(0.0015, 0.0015);
    c.sensor.accel_noise_range = Range(0.011, 0.011);
    c.wind.v_10_range = Range(2.0, 2.0);
    c.wind.gust_amplitude_range = Range(0.0, 0.0);
    c.wind.shear_delta_range = Range(0.0, 0.0);
    c.wind.shear_altitude_range = Range(150.0, 150.0);
    c.wind.turbulence_sigma = 0.0;
    c
}

#[test]
fn same_seed_config_regenerates_every_field_bit_identically() {
    for seed in [0u64, 1, 7, 1337] {
        for index in [0u64, 3, 15] {
            let first = episode_bits(&run(seed, index));
            let second = episode_bits(&run(seed, index));
            assert!(!first.is_empty(), "fingerprint must not be vacuous");
            assert_bits_same(&format!("seed={seed} index={index}"), &first, &second);
        }
    }
    // The streaming iterator must land on the same episode as a direct call.
    let sim = Simulator::new(cfg(7)).unwrap();
    let streamed: Vec<_> = sim.episodes(4).map(|e| episode_bits(&e.unwrap())).collect();
    assert_eq!(streamed.len(), 4);
    for (index, bits) in streamed.iter().enumerate() {
        assert_bits_same(
            &format!("iterator seed=7 index={index}"),
            bits,
            &episode_bits(&run(7, index as u64)),
        );
    }
}

#[test]
fn substream_keys_are_seed_index_and_domain_separated() {
    let seeds = [0u64, 1, 7, 1337, u64::MAX];
    let indices = [0u64, 1, 3, 15, 16];

    for &seed in &seeds {
        for &index in &indices {
            let keys: Vec<u64> = SUBSTREAMS
                .iter()
                .map(|&(_, tag)| episode_seed(seed, index ^ tag))
                .collect();
            for i in 0..keys.len() {
                for j in (i + 1)..keys.len() {
                    assert_ne!(
                        keys[i], keys[j],
                        "{} and {} alias at seed={seed} index={index}",
                        SUBSTREAMS[i].0, SUBSTREAMS[j].0
                    );
                }
            }
        }
    }

    for &(name, tag) in &SUBSTREAMS {
        let mut by_seed = HashSet::new();
        for &seed in &seeds {
            let key = episode_seed(seed, tag);
            assert!(by_seed.insert(key), "{name}: seed collision on {key}");
        }
        let mut by_index = HashSet::new();
        for index in 0..64u64 {
            let key = episode_seed(42, index ^ tag);
            assert!(by_index.insert(key), "{name}: index collision on {key}");
        }
    }
}

#[test]
fn different_seeds_move_every_observable_substream() {
    let a = cfg(101);
    let b = cfg(102);

    // INIT: first StdRng draw in episode.rs is the lift-off mass.
    let (ea, eb) = (run(a.run.seed, 0), run(b.run.seed, 0));
    assert_ne!(
        ea.meta.mass_kg, eb.meta.mass_kg,
        "INIT did not move with the seed"
    );

    // DOMA: independent tolerance sub-seed in sample_domain.
    let (da, db) = (sample_domain(&a, 0), sample_domain(&b, 0));
    assert_ne!(da.cd0, db.cd0, "DOMA did not move with the seed");
    assert_ne!(
        ea.meta.conditioning.cd0, eb.meta.conditioning.cd0,
        "DOMA did not reach the exported conditioning"
    );

    // WIND: gust onset (seeded in wind.rs) and the turbulence field both move.
    let (wa, wb) = (Wind::new(&a, 15), Wind::new(&b, 15));
    let (ga, gb) = (wa.gust_info().unwrap().0, wb.gust_info().unwrap().0);
    assert_ne!(ga, gb, "WIND gust onset did not move with the seed");
    assert_ne!(
        Wind::new(&a, 0).at(100, 200.0, 1.0),
        Wind::new(&b, 0).at(100, 200.0, 1.0),
        "WIND turbulence did not move with the seed"
    );

    // SENS: with every config envelope frozen and wind noise off, only the
    // seeded sensor bias/noise scales can still move the frames.
    let (sa, sb) = (frozen_cfg(101), frozen_cfg(102));
    let fa = Simulator::new(sa).unwrap().run_episode(0).unwrap();
    let fb = Simulator::new(sb).unwrap().run_episode(0).unwrap();
    assert_bits_same(
        "frozen meta must not depend on the seed",
        &meta_bits(&fa),
        &meta_bits(&fb),
    );
    let g_a: Vec<u8> = fa.frames[100]
        .gyro
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect();
    let g_b: Vec<u8> = fb.frames[100]
        .gyro
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect();
    assert_ne!(g_a, g_b, "SENS did not move the gyro noise");
}

#[test]
fn global_rng_state_and_call_order_cannot_change_episodes() {
    let baseline = episode_bits(&run(7, 0));

    let mut global = rand::thread_rng();
    for _ in 0..10_000 {
        let _: u64 = global.gen();
        let _: f64 = global.gen();
    }
    assert_bits_same(
        "draining the thread RNG changed an episode",
        &baseline,
        &episode_bits(&run(7, 0)),
    );

    let sim = Simulator::new(cfg(9)).unwrap();
    let forward: Vec<(u64, Vec<u8>)> = [4u64, 1, 7, 0]
        .iter()
        .map(|&i| (i, episode_bits(&sim.run_episode(i).unwrap())))
        .collect();
    let reverse: Vec<(u64, Vec<u8>)> = [0u64, 7, 1, 4]
        .iter()
        .map(|&i| (i, episode_bits(&sim.run_episode(i).unwrap())))
        .collect();
    for (index, bits) in &forward {
        let same = reverse.iter().find(|(i, _)| i == index).unwrap();
        assert_bits_same(&format!("call order index={index}"), bits, &same.1);
    }

    let one = Simulator::new(cfg(11)).unwrap();
    let first = episode_bits(&one.run_episode(2).unwrap());
    let _ = Simulator::new(cfg(99)).unwrap().run_episode(2).unwrap();
    let again = episode_bits(&one.run_episode(2).unwrap());
    assert_bits_same(
        "interleaving another seed changed an episode",
        &first,
        &again,
    );

    let sim = Simulator::new(cfg(13)).unwrap();
    let serial: Vec<_> = (0..8u64)
        .map(|i| episode_bits(&sim.run_episode(i).unwrap()))
        .collect();
    let parallel: Vec<_> = sim
        .run_episodes_parallel(8)
        .into_iter()
        .map(|e| episode_bits(&e.unwrap()))
        .collect();
    for i in 0..serial.len() {
        assert_bits_same(&format!("rayon index={i}"), &serial[i], &parallel[i]);
    }
}

fn write_split(dir: &Path, cfg: &Config, ids: &[u64]) {
    std::fs::create_dir_all(dir).unwrap();
    let sim = Simulator::new(cfg.clone()).unwrap();
    let episodes: Vec<_> = ids.iter().map(|&i| sim.run_episode(i).unwrap()).collect();
    let validation = dir.join("validation.bin");
    write_validation(&validation, cfg, &episodes).unwrap();
    potok_sim::trajectory::write_trajectory(&dir.join("v1.trajectory"), cfg, &episodes).unwrap();
    potok_sim::trajectory::write_trajectory_v2(&dir.join("v2.trajectory"), cfg, &episodes).unwrap();
    potok_sim::trajectory::write_trajectory_v3(
        &potok_sim::sidecar_path(&validation),
        cfg,
        &episodes,
    )
    .unwrap();
}

#[test]
fn regenerated_sidecars_are_byte_identical() {
    let cfg = cfg(4242);
    let ids = [0u64, 3, 15];
    let (a, b, c) = (temp("a"), temp("b"), temp("negative"));
    write_split(&a, &cfg, &ids);
    write_split(&b, &cfg, &ids);

    let names = [
        "validation.bin",
        "v1.trajectory",
        "v2.trajectory",
        "validation.trajectory",
    ];
    let mut pairs: Vec<(&str, Vec<u8>, Vec<u8>)> = Vec::new();
    for name in names {
        let left = std::fs::read(a.join(name)).unwrap();
        let right = std::fs::read(b.join(name)).unwrap();
        assert_bytes_same(name, &left, &right);
        pairs.push((name, left, right));
    }
    let refs: Vec<_> = pairs
        .iter()
        .map(|(name, left, right)| (*name, left.as_slice(), right.as_slice()))
        .collect();
    let gate = check_determinism(&refs).unwrap();
    assert!(gate.ok);
    assert!(
        gate.detail.contains("4 files compared, mismatches=[]"),
        "{}",
        gate.detail
    );

    // The equality above is only a gate if different seed actually moves the
    // sidecar bytes; otherwise a constant writer would pass it.
    let mut other = cfg.clone();
    other.run.seed += 1;
    write_split(&c, &other, &ids);
    let base = std::fs::read(a.join("validation.trajectory")).unwrap();
    let changed = std::fs::read(c.join("validation.trajectory")).unwrap();
    assert_ne!(
        base, changed,
        "a different seed must change the sidecar bytes"
    );

    for dir in [&a, &b, &c] {
        let _ = std::fs::remove_dir_all(dir);
    }
}
