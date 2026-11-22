//! Command->dynamics seam: honest tests for the control input, its LOUD
//! validation, and the flown-truth sidecar used to measure apogee/drift.

use std::path::PathBuf;

use potok_sim::control::{validate_command, ControlFile, MAX_CONTROL_DEG};
use potok_sim::{
    apogee_m, lateral_rms_m, read_trajectory, sidecar_path, write_trajectory, Config, SimError,
    Simulator,
};

fn sim() -> Simulator {
    Simulator::new(Config::default()).expect("default config")
}

fn temp(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!(
        "potok-control-{}-{tag}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = std::fs::remove_file(&p);
    p
}

#[test]
fn zero_control_is_bit_identical_to_legacy() {
    let sim = sim();
    let legacy = sim.run_episode(0).expect("legacy episode");
    let zero = sim
        .run_episode_with_control(0, &|_f| (0.0, 0.0))
        .expect("zero-control episode");
    assert_eq!(zero.frames, legacy.frames, "frame bytes moved");
    assert_eq!(zero.features, legacy.features, "feature vector moved");
    assert_eq!(zero.targets, legacy.targets, "targets moved");
    assert_eq!(zero.truth, legacy.truth, "terminal truth moved");
    assert_eq!(zero.trajectory, legacy.trajectory, "flown truth moved");
    assert_eq!(zero.apogee_m.to_bits(), legacy.apogee_m.to_bits());
    assert_eq!(zero.max_speed_mps.to_bits(), legacy.max_speed_mps.to_bits());
    assert_eq!(zero.frames_generated, legacy.frames_generated);
    assert_eq!(zero.phase_counts, legacy.phase_counts);
    assert_eq!(zero.invalid_frames, legacy.invalid_frames);
    assert_eq!(zero.nan_frames, legacy.nan_frames);
}

#[test]
fn constant_control_moves_the_flown_trajectory() {
    let sim = sim();
    let base = sim.run_episode(0).expect("legacy episode");
    let flown = sim
        .run_episode_with_control(0, &|_f| (MAX_CONTROL_DEG, 0.0))
        .expect("controlled episode");

    let base_apogee = apogee_m(&base.trajectory);
    let flown_apogee = apogee_m(&flown.trajectory);
    let base_lateral = lateral_rms_m(&base.trajectory);
    let flown_lateral = lateral_rms_m(&flown.trajectory);
    let lateral_delta = (flown_lateral - base_lateral).abs();
    let apogee_delta = (flown_apogee - base_apogee).abs();
    eprintln!(
        "control effect: apogee {base_apogee:.3} -> {flown_apogee:.3} m (d={apogee_delta:.3}); \
         lateral rms {base_lateral:.3} -> {flown_lateral:.3} m (d={lateral_delta:.3})"
    );
    assert!(
        apogee_delta > 1.0 || lateral_delta > 1.0,
        "a bounded constant command must move the vehicle: apogee d={apogee_delta}, lateral d={lateral_delta}"
    );
    assert_eq!(flown.trajectory.len(), base.trajectory.len());
}

#[test]
fn opposite_commands_move_the_vehicle_in_opposite_directions() {
    let sim = sim();
    let pos = sim
        .run_episode_with_control(0, &|_f| (0.0, MAX_CONTROL_DEG))
        .expect("+pitch");
    let neg = sim
        .run_episode_with_control(0, &|_f| (0.0, -MAX_CONTROL_DEG))
        .expect("-pitch");
    let x_pos = pos.trajectory.last().unwrap().pos[0] as f64;
    let x_neg = neg.trajectory.last().unwrap().pos[0] as f64;
    eprintln!("final x: +pitch {x_pos:.3} m, -pitch {x_neg:.3} m");
    assert!(
        (x_pos - x_neg).abs() > 1.0,
        "sign of the command must be preserved through the dynamics"
    );
}

#[test]
fn command_limits_are_loud() {
    assert_eq!(
        validate_command(MAX_CONTROL_DEG, -MAX_CONTROL_DEG).unwrap(),
        (MAX_CONTROL_DEG, -MAX_CONTROL_DEG)
    );
    for (r, p) in [
        (MAX_CONTROL_DEG + 0.1, 0.0),
        (0.0, -(MAX_CONTROL_DEG + 0.1)),
        (f64::NAN, 0.0),
        (0.0, f64::INFINITY),
    ] {
        assert!(
            matches!(validate_command(r, p), Err(SimError::Control(_))),
            "command ({r}, {p}) must be refused"
        );
    }
    let sim = sim();
    assert!(matches!(
        sim.run_episode_with_control(0, &|_f| (11.0, 0.0)),
        Err(SimError::Control(_))
    ));
}

#[test]
fn control_file_validation_is_loud_on_bad_index_length_and_range() {
    let n = 4;
    let good = "frame_index g_roll g_pitch\n0 0 0\n1 1 0\n2 0 -1\n3 0 0\n";
    let table = ControlFile::from_str(good, n).expect("valid table");
    assert_eq!(table.len(), n);
    assert_eq!(table.at(2).unwrap(), (0.0, -1.0));
    assert!(matches!(table.at(n), Err(SimError::Control(_))));

    let bad_cases = [
        ("missing final row", "0 0 0\n1 0 0\n2 0 0\n"),
        ("extra row", "0 0 0\n1 0 0\n2 0 0\n3 0 0\n4 0 0\n"),
        ("duplicate index", "0 0 0\n1 0 0\n1 0 0\n3 0 0\n"),
        ("gap in indices", "0 0 0\n1 0 0\n3 0 0\n"),
        ("index out of range", "0 0 0\n1 0 0\n2 0 0\n4 0 0\n"),
        ("index not a number", "zero 0 0\n1 0 0\n2 0 0\n3 0 0\n"),
        ("wrong field count", "0 0\n1 0 0\n2 0 0\n3 0 0\n"),
        ("command out of range", "0 0 0\n1 11 0\n2 0 0\n3 0 0\n"),
        ("non-finite command", "0 0 0\n1 nan 0\n2 0 0\n3 0 0\n"),
        ("command not a number", "0 0 0\n1 x 0\n2 0 0\n3 0 0\n"),
    ];
    for (name, text) in bad_cases {
        let err = ControlFile::from_str(text, n).expect_err(name);
        assert!(matches!(err, SimError::Control(_)), "{name}: {err}");
        eprintln!("{name}: {err}");
    }
}

#[test]
fn control_file_drives_the_same_trajectory_as_a_constant_closure() {
    let sim = sim();
    let frames = Config::default().frames_per_episode();
    let file = temp("commands.tsv");
    let mut text = String::from("# frame_index g_roll g_pitch\nframe_index g_roll g_pitch\n");
    for i in 0..frames {
        text.push_str(&format!("{i} 2.5 -1.0\n"));
    }
    std::fs::write(&file, &text).unwrap();
    let table = ControlFile::from_path(&file, frames).expect("control file");
    let from_file = sim
        .run_episode_with_control(0, &|f| table.for_frame(f).unwrap())
        .expect("file-driven episode");
    let from_closure = sim
        .run_episode_with_control(0, &|_f| (2.5, -1.0))
        .expect("closure-driven episode");
    assert_eq!(from_file.trajectory, from_closure.trajectory);
    assert_eq!(
        from_file.apogee_m.to_bits(),
        from_closure.apogee_m.to_bits()
    );
    std::fs::remove_file(&file).unwrap();
}

#[test]
fn trajectory_sidecar_roundtrip_carries_flown_truth() {
    let sim = sim();
    let cfg = Config::default();
    let eps: Vec<_> = (0..2)
        .map(|i| {
            sim.run_episode_with_control(i, &|_f| (3.0, -2.0))
                .expect("episode")
        })
        .collect();
    let validation = temp("validation.bin");
    let sidecar = sidecar_path(&validation);
    assert_eq!(sidecar, validation.with_extension("trajectory"));
    write_trajectory(&sidecar, &cfg, &eps).expect("write sidecar");

    let set = read_trajectory(&sidecar).expect("read sidecar");
    assert_eq!(set.episodes.len(), 2);
    assert_eq!(set.frames_per_episode as usize, cfg.frames_per_episode());
    for (got, want) in set.episodes.iter().zip(eps.iter()) {
        assert_eq!(got.index, want.meta.index);
        assert_eq!(got.ticks, want.trajectory);
        assert_eq!(got.apogee_m(), apogee_m(&want.trajectory));
        assert_eq!(got.lateral_rms_m(), lateral_rms_m(&want.trajectory));
    }
    assert_eq!(
        std::fs::metadata(&sidecar).unwrap().len() as usize,
        64 + 2 * (8 + cfg.frames_per_episode() * 6 * 4),
        "sidecar length is a direct function of frames x stride"
    );
    std::fs::remove_file(&sidecar).unwrap();
}
