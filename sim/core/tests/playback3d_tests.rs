// SPDX-License-Identifier: GPL-3.0-or-later
//! Hard gates for the deterministic POTOKTRJ v3 -> glTF playback artifact.
//!
//! The two new source modules are included by path: this task owns no manifest,
//! and the re-exports below give the core module its `crate::{error,trajectory}`
//! paths in the test crate. No other source is touched.

pub use potok_sim::{error, trajectory};

#[path = "../../cli/src/playback3d.rs"]
pub mod cli_playback3d;
#[path = "../src/playback3d.rs"]
pub mod playback3d;

use std::path::{Path, PathBuf};

use potok_sim::trajectory::{
    EpisodeTrajectory, TrajectorySet, TrajectoryTick, HEADER_SIZE, TRUTH_V3_STRIDE, VERSION_V2,
    VERSION_V3,
};

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("potok-playback3d-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn tick(pos: [f32; 3], vel: [f32; 3], mass_kg: f32) -> TrajectoryTick {
    TrajectoryTick {
        pos,
        vel,
        quat: [1.0, 0.0, 0.0, 0.0],
        mass_kg,
        gust_start_s: -1.0,
        ..TrajectoryTick::default()
    }
}

fn set(ticks: Vec<TrajectoryTick>) -> TrajectorySet {
    TrajectorySet {
        version: VERSION_V3,
        episode_count: 1,
        frames_per_episode: ticks.len() as u32,
        episodes: vec![EpisodeTrajectory { index: 7, ticks }],
    }
}

fn assert_close(got: [f64; 3], want: [f64; 3]) {
    for (g, w) in got.iter().zip(want.iter()) {
        assert!(
            (g - w).abs() < 1e-6,
            "got {got:?}, want {want:?} (component delta {} > 1e-6)",
            (g - w).abs()
        );
    }
}

/// Independent reader: raw little-endian v3 fixture, no writer under test.
fn write_v3(path: &Path, source: &TrajectorySet) {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"POTOKTRJ");
    bytes.extend_from_slice(&VERSION_V3.to_le_bytes());
    bytes.extend_from_slice(&source.episode_count.to_le_bytes());
    bytes.extend_from_slice(&source.frames_per_episode.to_le_bytes());
    bytes.extend_from_slice(&(TRUTH_V3_STRIDE as u32).to_le_bytes());
    bytes.extend_from_slice(&[0u8; HEADER_SIZE - 24]);
    for ep in &source.episodes {
        bytes.extend_from_slice(&ep.index.to_le_bytes());
        for t in &ep.ticks {
            for value in t.as_truth_v3_array() {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
    }
    std::fs::write(path, bytes).unwrap();
}

#[test]
fn document_is_gltf20_and_byte_deterministic() {
    let source = set(vec![
        tick([0.0, 0.0, 0.0], [0.0, 0.0, 10.0], 10.0),
        tick([1.0, 2.0, 3.0], [1.0, 0.0, 9.0], 9.0),
    ]);
    let a = playback3d::to_json(&source).unwrap();
    let b = playback3d::to_json(&source).unwrap();
    assert_eq!(
        a, b,
        "no clock, RNG or map iteration may enter the artifact"
    );

    let v: serde_json::Value = serde_json::from_str(&a).unwrap();
    assert_eq!(v["asset"]["version"], "2.0");
    assert_eq!(v["asset"]["generator"], "potok-sim playback3d");
    assert_eq!(v["scene"], 0);
    assert_eq!(v["scenes"][0]["nodes"].as_array().unwrap().len(), 2);
    assert_eq!(v["nodes"].as_array().unwrap().len(), 2);
    assert_eq!(v["nodes"][0]["name"], "ep7/t0");
    assert_eq!(v["nodes"][1]["name"], "ep7/t1");
    assert_eq!(v["extras"]["schema"], playback3d::SCHEMA_VERSION);
    assert_eq!(v["extras"]["source"], "POTOKTRJ v3");
    assert_eq!(v["extras"]["tick_hz"], 50);
    assert_eq!(v["extras"]["episodes"][0]["episode_index"], 7);
    assert_eq!(v["extras"]["episodes"][0]["node_start"], 0);
    assert_eq!(v["extras"]["episodes"][0]["tick_count"], 2);
}

#[test]
fn node_transforms_use_gltf_y_up_and_keep_the_sim_body_axis() {
    let mut t = tick([1.0, 2.0, 3.0], [0.0; 3], 10.0);
    let c = std::f64::consts::FRAC_PI_4.cos() as f32;
    t.quat = [c, 0.0, 0.0, c]; // +90 degrees about sim Z
    let doc = playback3d::document(&set(vec![t])).unwrap();
    let node = &doc.nodes[0];

    assert_eq!(node.translation, [1.0, 3.0, -2.0], "x,z,-y mapping");
    assert_eq!(node.extras.pos_sim, [1.0, 2.0, 3.0], "raw truth preserved");
    assert_close(node.extras.body_axis_sim, [0.0, 0.0, 1.0]);

    // A spin about sim Z leaves the sim body axis at (0,0,1). Rotate local +Z
    // by the node's glTF [x,y,z,w] rotation: it must land on the mapped axis
    // (0,1,0), proving the node transform carries the coordinate change.
    let [x, y, z, w] = node.rotation;
    let axis = [
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        1.0 - 2.0 * (x * x + y * y),
    ];
    assert_close(axis, [0.0, 1.0, 0.0]);
}

#[test]
fn phases_follow_the_mass_plateau_and_flown_state() {
    let ticks = vec![
        tick([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 10.0), // PAD
        tick([0.0, 0.0, 2.0], [0.0, 0.0, 20.0], 9.0), // BOOST
        tick([0.0, 0.0, 5.0], [0.0, 0.0, 25.0], 8.0), // BOOST (last burn tick)
        tick([0.0, 0.0, 12.0], [1.0, 0.0, 8.0], 8.0), // COAST
        tick([0.0, 0.0, 15.0], [0.5, 0.0, 0.0], 8.0), // APOGEE
        tick([0.0, 0.0, 8.0], [0.0, 0.0, -9.0], 8.0), // DESCENT
        tick([0.0, 0.0, 0.0], [0.0, 0.0, -9.0], 8.0), // LANDED
    ];
    let doc = playback3d::document(&set(ticks)).unwrap();
    let got: Vec<(&str, u8)> = doc
        .nodes
        .iter()
        .map(|n| (n.extras.phase, n.extras.phase_code))
        .collect();
    assert_eq!(
        got,
        vec![
            ("PAD", 0),
            ("BOOST", 1),
            ("BOOST", 1),
            ("COAST", 2),
            ("APOGEE", 3),
            ("DESCENT", 4),
            ("LANDED", 5),
        ]
    );
}

#[test]
fn gust_and_shear_markers_are_emitted_once_at_the_recorded_tick() {
    let mut t0 = tick([0.0, 0.0, 0.0], [0.0; 3], 10.0);
    t0.gust_start_s = 0.04; // tick 2 at 50 Hz
    t0.gust_amplitude = 2.5;
    t0.shear_altitude_m = 5.0;
    let ticks = vec![
        t0,
        tick([0.0, 0.0, 2.0], [0.0; 3], 9.0),
        tick([1.0, 0.0, 6.0], [0.0; 3], 8.0),
        tick([2.0, 0.0, 9.0], [0.0; 3], 8.0),
    ];
    let doc = playback3d::document(&set(ticks)).unwrap();
    let markers = &doc.extras.episodes[0].markers;
    assert_eq!(markers.len(), 2);
    assert_eq!(markers[0].kind, "gust");
    assert_eq!(markers[0].tick, 2);
    assert_eq!(markers[0].time_s, 0.04);
    assert_eq!(markers[0].value, 2.5);
    assert_eq!(markers[1].kind, "shear");
    assert_eq!(markers[1].tick, 2);
    assert_eq!(markers[1].value, 5.0);

    // The gust is armed at every tick from the v3 field, never inferred.
    assert!(doc.nodes[0].extras.gust_armed);
    assert_eq!(doc.nodes[0].extras.gust_amplitude, 2.5);
}

#[test]
fn wrong_version_and_non_finite_truth_fail_loudly() {
    let mut old = set(vec![tick([0.0; 3], [0.0; 3], 10.0)]);
    old.version = VERSION_V2;
    let err = playback3d::document(&old).unwrap_err().to_string();
    assert!(err.contains("requires POTOKTRJ v3"), "{err}");

    let mut bad = set(vec![tick([0.0; 3], [0.0; 3], 10.0)]);
    bad.episodes[0].ticks[0].pos[1] = f32::NAN;
    let err = playback3d::document(&bad).unwrap_err().to_string();
    assert!(err.contains("non-finite pos_y"), "{err}");
    assert!(err.contains("episode 7 tick 0"), "{err}");

    let mut zero_q = set(vec![tick([0.0; 3], [0.0; 3], 10.0)]);
    zero_q.episodes[0].ticks[0].quat = [0.0; 4];
    let err = playback3d::document(&zero_q).unwrap_err().to_string();
    assert!(err.contains("zero quaternion"), "{err}");

    let mut empty = set(vec![tick([0.0; 3], [0.0; 3], 10.0)]);
    empty.episodes[0].ticks.clear();
    assert!(playback3d::document(&empty).is_err());
}

#[test]
fn export_file_roundtrip_is_byte_stable() {
    let dir = temp_dir("roundtrip");
    let source = set(vec![
        tick([0.0, 0.0, 0.0], [0.0, 0.0, 5.0], 10.0),
        tick([1.0, 1.0, 2.0], [2.0, 0.0, 4.0], 9.0),
    ]);
    let traj = dir.join("validation.trajectory");
    write_v3(&traj, &source);

    let a = dir.join("a.json");
    let b = dir.join("b.json");
    let ia = playback3d::export_file(&traj, &a).unwrap();
    let ib = playback3d::export_file(&traj, &b).unwrap();
    assert_eq!(ia, ib);
    assert_eq!(ia.episodes, 1);
    assert_eq!(ia.ticks, 2);
    assert_eq!(ia.markers, 0);
    assert_eq!(std::fs::read(&a).unwrap(), std::fs::read(&b).unwrap());
    assert_eq!(
        playback3d::playback3d_path(&dir.join("validation.bin")),
        dir.join("validation.playback3d.json")
    );
    assert_eq!(
        ia.bytes,
        std::fs::read(&a).unwrap().len(),
        "reported byte count is the file length"
    );
}

#[test]
fn cli_adapter_parses_flags_and_runs_with_an_injected_exporter() {
    let dir = temp_dir("cli");
    let source = set(vec![tick([0.0; 3], [0.0; 3], 10.0)]);
    let traj = dir.join("v.trajectory");
    let out = dir.join("v.playback3d.json");
    write_v3(&traj, &source);

    let args = [
        "playback3d".to_owned(),
        "--trajectory".to_owned(),
        traj.display().to_string(),
        "--out".to_owned(),
        out.display().to_string(),
    ];
    let summary = cli_playback3d::run_with(args, |t, o| {
        assert_eq!(t, traj.as_path());
        assert_eq!(o, out.as_path());
        playback3d::export_file(t, o)
            .map(|i| cli_playback3d::Summary {
                episodes: i.episodes,
                ticks: i.ticks,
                markers: i.markers,
                bytes: i.bytes,
            })
            .map_err(|e| e.to_string())
    })
    .unwrap();
    assert_eq!(summary.episodes, 1);
    assert_eq!(summary.ticks, 1);
    assert_eq!(summary.markers, 0);
    assert!(out.exists());
    assert!(cli_playback3d::format_summary(&summary).starts_with("playback3d: episodes=1"));
}

#[test]
fn cli_adapter_rejects_bad_flags_without_touching_the_exporter() {
    let bad = ["playback3d".to_owned(), "--trajectory".to_owned()];
    let err = cli_playback3d::run_with(bad, |_, _| -> Result<cli_playback3d::Summary, String> {
        panic!("exporter must not be called on a usage error")
    })
    .unwrap_err();
    assert!(matches!(err, cli_playback3d::CliError::Usage(_)));
    assert!(err.to_string().contains("--trajectory needs a value"));

    let missing = ["--out".to_owned(), "x.json".to_owned()];
    let err = cli_playback3d::parse(missing).unwrap_err();
    assert!(err.to_string().contains("--trajectory is required"));
}
