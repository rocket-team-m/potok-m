// SPDX-License-Identifier: GPL-3.0-or-later
//! Hard compatibility gates for the versioned trajectory sidecar.
//!
//! These tests pin the on-disk contract of `write_trajectory{,_v2,_v3}` and
//! `read_trajectory`: header/stride mapping for v1/v2/v3, the exact byte slot
//! of v3's `g_accel`, bit-exact round trips, explicit values for fields absent
//! from an older version, and sha256 stability of the emitted bytes.
//! They read/write fixtures only; no production source is changed here.

use std::path::PathBuf;

use potok_sim::{
    manifest::sha256_hex,
    trajectory::{
        read_trajectory, write_trajectory, write_trajectory_v2, write_trajectory_v3,
        write_trajectory_with_version, HEADER_SIZE, MAGIC, STRIDE, TRUTH_STRIDE, TRUTH_TICK_NAMES,
        TRUTH_V3_STRIDE, TRUTH_V3_TICK_NAMES, VERSION,
    },
    Config, Episode, Simulator, TrajectoryTick, VERSION_V2, VERSION_V3,
};

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("potok-traj-v3-hard-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// 1 s / 0.002 dt / 10 steps per frame = 50 ticks: enough to exercise every
/// record boundary without making byte dumps large.
fn small_cfg() -> Config {
    let mut cfg = Config::default();
    cfg.run.duration_s = 1.0;
    cfg
}

/// Distinct finite sentinel per field index (0..26), tieing a byte slot to a
/// named field without depending on the serializer under test.
fn sentinel_tick(i: usize) -> TrajectoryTick {
    let v = |k: usize| (i as f32) * 0.5 + (k as f32) * 0.125 + 0.25;
    let mut tick = TrajectoryTick {
        pos: [v(0), v(1), v(2)],
        vel: [v(3), v(4), v(5)],
        quat: [v(6), v(7), v(8), v(9)],
        omega: [v(10), v(11), v(12)],
        mass_kg: v(13),
        wind: [v(14), v(15), v(16)],
        gust_start_s: v(17),
        gust_amplitude: v(18),
        shear_altitude_m: v(19),
        shear_delta: v(20),
        shear_thickness_m: v(21),
        g_theta: [v(22), v(23)],
        g_accel: [v(24), v(25)],
    };
    if i == 0 {
        // -0.0 compares equal to 0.0 but has a different bit pattern.
        tick.g_accel[0] = -0.0;
    }
    tick
}

fn synthetic_episode(cfg: &Config, index: u64) -> Episode {
    let sim = Simulator::new(cfg.clone()).unwrap();
    let mut ep = sim.run_episode(index).unwrap();
    ep.meta.index = index;
    ep.trajectory = (0..cfg.frames_per_episode()).map(sentinel_tick).collect();
    ep
}

fn stride_of(version: u32) -> usize {
    match version {
        VERSION => STRIDE,
        VERSION_V2 => TRUTH_STRIDE,
        VERSION_V3 => TRUTH_V3_STRIDE,
        other => panic!("unexpected version {other}"),
    }
}

fn u32_at(bytes: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap())
}

fn tick_start(version: u32, tick: usize) -> usize {
    HEADER_SIZE + 8 + tick * stride_of(version) * 4
}

fn f32_at(bytes: &[u8], start: usize, field: usize) -> f32 {
    let off = start + field * 4;
    f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap())
}

#[test]
fn header_stride_and_file_size_are_version_strict() {
    let dir = temp_dir("header");
    let cfg = small_cfg();
    let eps: Vec<Episode> = [11u64, 22]
        .into_iter()
        .map(|i| synthetic_episode(&cfg, i))
        .collect();
    let frames = cfg.frames_per_episode();

    for (version, stride) in [
        (VERSION, STRIDE),
        (VERSION_V2, TRUTH_STRIDE),
        (VERSION_V3, TRUTH_V3_STRIDE),
    ] {
        let path = dir.join(format!("v{version}.trajectory"));
        write_trajectory_with_version(&path, &cfg, &eps, version).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[0..8], MAGIC, "magic must be stable for v{version}");
        assert_eq!(u32_at(&bytes, 8), version);
        assert_eq!(u32_at(&bytes, 12), eps.len() as u32);
        assert_eq!(u32_at(&bytes, 16), frames as u32);
        assert_eq!(u32_at(&bytes, 20), stride as u32, "v{version} stride field");
        assert!(
            bytes[24..HEADER_SIZE].iter().all(|b| *b == 0),
            "v{version} reserved header bytes must be zero"
        );
        assert_eq!(
            bytes.len(),
            HEADER_SIZE + eps.len() * (8 + frames * stride * 4),
            "v{version} file length must match its stride"
        );
        for (e, ep) in eps.iter().enumerate() {
            let rec = HEADER_SIZE + e * (8 + frames * stride * 4);
            assert_eq!(
                u64::from_le_bytes(bytes[rec..rec + 8].try_into().unwrap()),
                ep.meta.index
            );
        }
        let set = read_trajectory(&path).unwrap();
        assert_eq!(set.version, version);
        assert_eq!(set.episode_count, eps.len() as u32);
        assert_eq!(set.frames_per_episode, frames as u32);
        assert_eq!(set.episodes.len(), eps.len());
        for (got, src) in set.episodes.iter().zip(eps.iter()) {
            assert_eq!(got.ticks.len(), frames);
            assert_eq!(got.ticks[0].pos, src.trajectory[0].pos);
            assert_eq!(got.ticks[frames - 1].vel, src.trajectory[frames - 1].vel);
        }
    }

    // The three named writers must emit exactly the bytes their header claims.
    for (version, named) in [
        (VERSION, "named-v1.trajectory"),
        (VERSION_V2, "named-v2.trajectory"),
        (VERSION_V3, "named-v3.trajectory"),
    ] {
        let path = dir.join(named);
        match version {
            VERSION => write_trajectory(&path, &cfg, &eps).unwrap(),
            VERSION_V2 => write_trajectory_v2(&path, &cfg, &eps).unwrap(),
            VERSION_V3 => write_trajectory_v3(&path, &cfg, &eps).unwrap(),
            other => panic!("unexpected version {other}"),
        }
        assert_eq!(
            std::fs::read(&path).unwrap(),
            std::fs::read(dir.join(format!("v{version}.trajectory"))).unwrap(),
            "named writer for v{version} diverged from write_trajectory_with_version"
        );
    }

    // The six v1 floats are a strict, slot-for-slot prefix of v2 and v3.
    let b1 = std::fs::read(dir.join("v1.trajectory")).unwrap();
    let b2 = std::fs::read(dir.join("v2.trajectory")).unwrap();
    let b3 = std::fs::read(dir.join("v3.trajectory")).unwrap();
    for tick in 0..frames {
        let s1 = tick_start(VERSION, tick);
        let s2 = tick_start(VERSION_V2, tick);
        let s3 = tick_start(VERSION_V3, tick);
        for k in 0..STRIDE {
            assert_eq!(f32_at(&b1, s1, k).to_bits(), f32_at(&b2, s2, k).to_bits());
            assert_eq!(f32_at(&b1, s1, k).to_bits(), f32_at(&b3, s3, k).to_bits());
        }
    }
}

#[test]
fn g_accel_occupies_float_slots_24_and_25_of_every_v3_tick() {
    let dir = temp_dir("gaccel-slot");
    let cfg = small_cfg();
    let ep = synthetic_episode(&cfg, 7);
    let v3 = dir.join("v3.trajectory");
    let v2 = dir.join("v2.trajectory");
    write_trajectory_v3(&v3, &cfg, std::slice::from_ref(&ep)).unwrap();
    write_trajectory_v2(&v2, &cfg, std::slice::from_ref(&ep)).unwrap();
    let b3 = std::fs::read(&v3).unwrap();
    let b2 = std::fs::read(&v2).unwrap();

    assert_eq!(TRUTH_V3_STRIDE, TRUTH_STRIDE + 2);
    assert_eq!(&TRUTH_V3_TICK_NAMES[..TRUTH_STRIDE], &TRUTH_TICK_NAMES);
    assert_eq!(TRUTH_V3_TICK_NAMES[24], "g_accel_y");
    assert_eq!(TRUTH_V3_TICK_NAMES[25], "g_accel_x");
    assert_eq!(TRUTH_V3_TICK_NAMES.len(), TRUTH_V3_STRIDE);

    for (i, src) in ep.trajectory.iter().enumerate() {
        let s3 = tick_start(VERSION_V3, i);
        let s2 = tick_start(VERSION_V2, i);
        assert_eq!(
            f32_at(&b3, s3, 24).to_bits(),
            src.g_accel[0].to_bits(),
            "tick {i}: g_accel_y must be the first appended f32"
        );
        assert_eq!(
            f32_at(&b3, s3, 25).to_bits(),
            src.g_accel[1].to_bits(),
            "tick {i}: g_accel_x must be the second appended f32"
        );
        // g_theta stays at 22/23: a swap of the two pairs would be caught.
        assert_eq!(f32_at(&b3, s3, 22).to_bits(), src.g_theta[0].to_bits());
        assert_eq!(f32_at(&b3, s3, 23).to_bits(), src.g_theta[1].to_bits());
        // Appending must not disturb the v2 record in front of it.
        assert_eq!(
            &b3[s3..s3 + TRUTH_STRIDE * 4],
            &b2[s2..s2 + TRUTH_STRIDE * 4],
            "tick {i}: v2 prefix inside v3 diverges"
        );
    }
    assert!(
        ep.trajectory.iter().any(|t| t.g_accel != t.g_theta),
        "fixture must distinguish g_accel from g_theta for slot checks to bite"
    );
}

#[test]
fn v3_round_trip_is_bit_exact_across_all_26_fields() {
    let dir = temp_dir("roundtrip");
    let cfg = small_cfg();
    let ep = synthetic_episode(&cfg, 9);
    let path = dir.join("rt.trajectory");
    write_trajectory_v3(&path, &cfg, std::slice::from_ref(&ep)).unwrap();
    let set = read_trajectory(&path).unwrap();
    assert_eq!(set.version, VERSION_V3);
    assert_eq!(set.episodes.len(), 1);
    let got = &set.episodes[0].ticks;
    assert_eq!(got.len(), ep.trajectory.len());

    for (i, (g, s)) in got.iter().zip(ep.trajectory.iter()).enumerate() {
        let ga = g.as_truth_v3_array();
        let sa = s.as_truth_v3_array();
        for k in 0..TRUTH_V3_STRIDE {
            assert_eq!(
                ga[k].to_bits(),
                sa[k].to_bits(),
                "tick {i} field {k} ({})",
                TRUTH_V3_TICK_NAMES[k]
            );
        }
        assert_eq!(*g, *s, "struct round trip at tick {i}");
        assert_eq!(
            TrajectoryTick::from_truth_v3_array(sa).as_truth_v3_array(),
            sa,
            "array conversion pair at tick {i}"
        );
    }
    assert_eq!(got[0].g_accel[0].to_bits(), (-0.0f32).to_bits());
}

#[test]
fn older_versions_gap_new_fields_explicitly_and_drop_them_on_write() {
    let dir = temp_dir("gaps");
    let cfg = small_cfg();
    let ep = synthetic_episode(&cfg, 3);
    let mut no_accel = ep.clone();
    for tick in &mut no_accel.trajectory {
        tick.g_accel = [0.0; 2];
    }
    let mut no_v3_or_v2_cmd = no_accel.clone();
    for tick in &mut no_v3_or_v2_cmd.trajectory {
        tick.g_theta = [0.0; 2];
    }

    // v1: six floats on disk, every privileged field an explicit default.
    let v1 = dir.join("g1.trajectory");
    write_trajectory(&v1, &cfg, std::slice::from_ref(&ep)).unwrap();
    let b1 = std::fs::read(&v1).unwrap();
    let s1 = read_trajectory(&v1).unwrap();
    let d = TrajectoryTick::default();
    for (i, (g, src)) in s1.episodes[0]
        .ticks
        .iter()
        .zip(ep.trajectory.iter())
        .enumerate()
    {
        assert_eq!(
            f32_at(&b1, tick_start(VERSION, i), 0).to_bits(),
            src.pos[0].to_bits()
        );
        assert_eq!(g.as_array(), src.as_array());
        assert_eq!(*g, TrajectoryTick::from_array(src.as_array()));
        assert_eq!(g.quat, d.quat);
        assert_eq!(g.omega, d.omega);
        assert_eq!(g.mass_kg, d.mass_kg);
        assert_eq!(g.wind, d.wind);
        assert_eq!(g.gust_start_s, -1.0);
        assert_eq!(g.gust_amplitude, 0.0);
        assert_eq!(g.shear_altitude_m, 0.0);
        assert_eq!(g.shear_delta, 0.0);
        assert_eq!(g.shear_thickness_m, 0.0);
        assert_eq!(g.g_theta, [0.0, 0.0]);
        assert_eq!(g.g_accel, [0.0, 0.0]);
    }

    // v2: 24 floats on disk, the v3-only pair is a zero gap on read.
    let v2 = dir.join("g2.trajectory");
    write_trajectory_v2(&v2, &cfg, std::slice::from_ref(&ep)).unwrap();
    let s2 = read_trajectory(&v2).unwrap();
    for (g, src) in s2.episodes[0].ticks.iter().zip(ep.trajectory.iter()) {
        assert_eq!(g.as_truth_array(), src.as_truth_array());
        assert_eq!(*g, TrajectoryTick::from_truth_array(src.as_truth_array()));
        assert_eq!(g.g_accel, [0.0, 0.0]);
    }
    assert_eq!(
        TrajectoryTick::from_truth_array(ep.trajectory[0].as_truth_array()).g_accel,
        [0.0; 2]
    );

    // Truncation is clean: v1 drops both command pairs, v2 drops only g_accel.
    let v1s = dir.join("g1-stripped.trajectory");
    write_trajectory(&v1s, &cfg, std::slice::from_ref(&no_v3_or_v2_cmd)).unwrap();
    assert_eq!(std::fs::read(&v1).unwrap(), std::fs::read(&v1s).unwrap());
    let v2s = dir.join("g2-stripped.trajectory");
    write_trajectory_v2(&v2s, &cfg, std::slice::from_ref(&no_accel)).unwrap();
    assert_eq!(std::fs::read(&v2).unwrap(), std::fs::read(&v2s).unwrap());
    let v2_no_cmd = dir.join("g2-no-cmd.trajectory");
    write_trajectory_v2(&v2_no_cmd, &cfg, std::slice::from_ref(&no_v3_or_v2_cmd)).unwrap();
    assert_ne!(
        std::fs::read(&v2).unwrap(),
        std::fs::read(&v2_no_cmd).unwrap()
    );

    // v3 bytes must depend on g_accel, or the writer is a no-op.
    let v3 = dir.join("g3.trajectory");
    write_trajectory_v3(&v3, &cfg, std::slice::from_ref(&ep)).unwrap();
    let v3s = dir.join("g3-stripped.trajectory");
    write_trajectory_v3(&v3s, &cfg, std::slice::from_ref(&no_accel)).unwrap();
    assert_ne!(std::fs::read(&v3).unwrap(), std::fs::read(&v3s).unwrap());
}

#[test]
fn bad_magic_version_or_stride_fail_loudly() {
    let dir = temp_dir("corrupt");
    let cfg = small_cfg();
    let ep = synthetic_episode(&cfg, 1);
    let bad = dir.join("bad.trajectory");

    for (version, stride) in [
        (VERSION, STRIDE),
        (VERSION_V2, TRUTH_STRIDE),
        (VERSION_V3, TRUTH_V3_STRIDE),
    ] {
        let good = dir.join(format!("good-{version}.trajectory"));
        write_trajectory_with_version(&good, &cfg, std::slice::from_ref(&ep), version).unwrap();
        let base = std::fs::read(&good).unwrap();

        let mut bytes = base.clone();
        bytes[8..12].copy_from_slice(&9u32.to_le_bytes());
        std::fs::write(&bad, &bytes).unwrap();
        assert!(
            read_trajectory(&bad)
                .unwrap_err()
                .to_string()
                .contains("version"),
            "v{version}: unsupported version must be loud"
        );

        let mut bytes = base.clone();
        bytes[20..24].copy_from_slice(&((stride + 1) as u32).to_le_bytes());
        std::fs::write(&bad, &bytes).unwrap();
        assert!(
            read_trajectory(&bad)
                .unwrap_err()
                .to_string()
                .contains("stride"),
            "v{version}: version/stride disagreement must be loud"
        );

        let mut bytes = base.clone();
        bytes[0] ^= 0xff;
        std::fs::write(&bad, &bytes).unwrap();
        assert!(
            read_trajectory(&bad)
                .unwrap_err()
                .to_string()
                .contains("magic"),
            "v{version}: bad magic must be loud"
        );
    }

    // A truncated record is an I/O failure, never a short silent read.
    let good = dir.join("good-3.trajectory");
    let mut bytes = std::fs::read(&good).unwrap();
    bytes.pop();
    std::fs::write(&bad, &bytes).unwrap();
    assert!(read_trajectory(&bad).is_err());
}

#[test]
fn sidecar_sha256_is_stable_and_binds_v3_only_fields() {
    let dir = temp_dir("hash");
    let cfg = small_cfg();
    let ep = synthetic_episode(&cfg, 5);
    let a = dir.join("a.trajectory");
    let b = dir.join("b.trajectory");
    write_trajectory_v3(&a, &cfg, std::slice::from_ref(&ep)).unwrap();
    write_trajectory_v3(&b, &cfg, std::slice::from_ref(&ep)).unwrap();
    let ha = sha256_hex(&std::fs::read(&a).unwrap());
    let hb = sha256_hex(&std::fs::read(&b).unwrap());
    assert_eq!(ha, hb, "identical regeneration must hash identically");
    assert_eq!(
        potok_sim::file_record("validation.trajectory", &a)
            .unwrap()
            .sha256,
        ha,
        "manifest file hash must bind the sidecar bytes"
    );

    // Moving one v3-only float, at the first or last tick, moves the hash.
    let mut changed_y = ep.clone();
    changed_y.trajectory[0].g_accel[0] += 0.5;
    let cy = dir.join("cy.trajectory");
    write_trajectory_v3(&cy, &cfg, std::slice::from_ref(&changed_y)).unwrap();
    assert_ne!(ha, sha256_hex(&std::fs::read(&cy).unwrap()));

    let last = ep.trajectory.len() - 1;
    let mut changed_x = ep.clone();
    changed_x.trajectory[last].g_accel[1] += 0.5;
    let cx = dir.join("cx.trajectory");
    write_trajectory_v3(&cx, &cfg, std::slice::from_ref(&changed_x)).unwrap();
    assert_ne!(ha, sha256_hex(&std::fs::read(&cx).unwrap()));

    // v2 truncates both fields: its hash cannot move with them.
    let v2a = dir.join("v2a.trajectory");
    let v2b = dir.join("v2b.trajectory");
    write_trajectory_v2(&v2a, &cfg, std::slice::from_ref(&ep)).unwrap();
    write_trajectory_v2(&v2b, &cfg, std::slice::from_ref(&changed_y)).unwrap();
    assert_eq!(
        sha256_hex(&std::fs::read(&v2a).unwrap()),
        sha256_hex(&std::fs::read(&v2b).unwrap())
    );

    // One episode, three versions: three distinct byte streams and hashes.
    let v1 = dir.join("v1.trajectory");
    write_trajectory(&v1, &cfg, std::slice::from_ref(&ep)).unwrap();
    let v3 = dir.join("v3.trajectory");
    write_trajectory_v3(&v3, &cfg, std::slice::from_ref(&ep)).unwrap();
    let h1 = sha256_hex(&std::fs::read(&v1).unwrap());
    let h2 = sha256_hex(&std::fs::read(&v2a).unwrap());
    let h3 = sha256_hex(&std::fs::read(&v3).unwrap());
    assert_ne!(h1, h2);
    assert_ne!(h2, h3);
    assert_ne!(h1, h3);
}
