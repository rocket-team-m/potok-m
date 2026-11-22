use std::{path::PathBuf, process::Command};

use potok_cli::tui::{render, DashboardState};
use potok_sim::{
    manifest::read_manifest, read_trajectory, scenario::Split, sidecar_path,
    validation::read_validation, Config,
};
use ratatui::{backend::TestBackend, Terminal};

fn run_cli(tag: &str, extra: &[&str]) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("potok-cli-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let out = Command::new(env!("CARGO_BIN_EXE_potok-cli"))
        .args(["--seed", "7", "--episodes", "20", "--validation-out"])
        .arg(&dir)
        .args(extra)
        .output()
        .expect("run potok-cli");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    dir
}

#[test]
fn cli_split_writes_only_withheld_scenarios() {
    let dir = run_cli("held", &["--physics-features", "--split", "held-out"]);
    let rs = read_validation(&dir.join("validation.bin")).unwrap();
    assert_eq!(rs.header.feature_count as usize, 14);
    assert_eq!(rs.episodes.len(), 4);
    assert!(rs.episodes.iter().all(|e| e.split == Split::HeldOut));
    assert_eq!(
        rs.episodes
            .iter()
            .map(|e| e.scenario_id)
            .collect::<Vec<_>>(),
        vec![16, 17, 18, 19]
    );

    let dir = run_cli("train", &["--physics-features", "--split", "train"]);
    let rs = read_validation(&dir.join("validation.bin")).unwrap();
    assert_eq!(rs.episodes.len(), 16);
    assert!(rs.episodes.iter().all(|e| e.split == Split::Train));
}

#[test]
fn cli_without_new_flags_keeps_the_eight_channel_default() {
    let dir = run_cli("legacy", &[]);
    let rs = read_validation(&dir.join("validation.bin")).unwrap();
    assert_eq!(rs.header.feature_count as usize, 8);
    assert_eq!(rs.episodes.len(), 20);
}

#[test]
fn cli_control_file_drives_dynamics_and_writes_truth_sidecar() {
    let cfg = Config::default();
    let dir = std::env::temp_dir().join(format!(
        "potok-cli-control-{}-{tag}",
        std::process::id(),
        tag = "tsv"
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let tsv = dir.join("commands.tsv");
    let mut text = String::from("frame_index g_roll g_pitch\n");
    for i in 0..cfg.frames_per_episode() {
        text.push_str(&format!("{i} 3.0 -1.5\n"));
    }
    std::fs::write(&tsv, &text).unwrap();
    let out = dir.join("validation");
    let proc = Command::new(env!("CARGO_BIN_EXE_potok-cli"))
        .args(["--seed", "7", "--episodes", "2", "--validation-out"])
        .arg(&out)
        .arg("--control-file")
        .arg(&tsv)
        .output()
        .expect("run potok-cli with control file");
    assert!(
        proc.status.success(),
        "{}",
        String::from_utf8_lossy(&proc.stderr)
    );
    let stdout = String::from_utf8_lossy(&proc.stdout);
    assert!(stdout.contains("control: rows=1500"), "{stdout}");
    assert!(stdout.contains("apogee="), "{stdout}");
    assert!(stdout.contains("lateral_rms="), "{stdout}");
    let rs = read_validation(&out.join("validation.bin")).unwrap();
    assert_eq!(rs.episodes.len(), 2);
    let traj = read_trajectory(&sidecar_path(&out.join("validation.bin"))).unwrap();
    assert_eq!(traj.episodes.len(), 2);
    assert_eq!(
        traj.episodes[0].ticks.len(),
        cfg.frames_per_episode(),
        "flown truth must be written for every tick"
    );
    assert!(traj.episodes[0].apogee_m() > 0.0);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn cli_control_file_errors_are_loud() {
    let dir = std::env::temp_dir().join(format!("potok-cli-badctl-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let tsv = dir.join("bad.tsv");
    std::fs::write(&tsv, "0 0 0\n1 99 0\n").unwrap();
    let proc = Command::new(env!("CARGO_BIN_EXE_potok-cli"))
        .args(["--seed", "7", "--episodes", "1", "--control-file"])
        .arg(&tsv)
        .output()
        .expect("run potok-cli with bad control file");
    assert!(!proc.status.success(), "bad control file must fail");
    let stderr = String::from_utf8_lossy(&proc.stderr);
    assert!(stderr.contains("control input error"), "{stderr}");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn dashboard_renders_on_test_backend_without_tty() {
    let cfg = Config::default();
    let mut state = DashboardState::from_config(&cfg, 3);
    state.config_path = "config/default.toml".into();
    let backend = TestBackend::new(100, 40);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|f| render(f, &state)).unwrap();
    let buf = terminal.backend().buffer();
    let text: String = buf.content.iter().map(|c| c.symbol()).collect();
    assert!(text.contains("POTOK-M"), "rendered dashboard missing title");
    assert!(
        text.contains("Progress"),
        "rendered dashboard missing progress panel"
    );
    assert!(
        text.contains("Wind"),
        "rendered dashboard missing wind panel"
    );
}

#[test]
fn dashboard_state_tracks_episodes() {
    let cfg = Config::default();
    let sim = potok_sim::Simulator::new(cfg.clone()).unwrap();
    let mut state = DashboardState::from_config(&cfg, 1);
    let e = sim.run_episode(0).unwrap();
    state.on_episode(&e);
    assert_eq!(state.episode, 1);
    assert_eq!(state.frames, e.frames.len());
    assert!(state.summary_lines()[0].contains("1/1"));
}

fn dataset_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("potok-dataset-cli-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn generate(out: &PathBuf, extra: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_potok-cli"))
        .arg("generate-dataset")
        .args([
            "--seed",
            "42",
            "--train-episodes",
            "8",
            "--held-out-episodes",
            "4",
        ])
        .arg("--out")
        .arg(out)
        .args(extra)
        .output()
        .expect("run generate-dataset")
}

#[test]
fn cli_generate_dataset_writes_both_splits_manifest_and_report() {
    let dir = dataset_dir("e2e");
    let out = generate(&dir, &[]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("generate-dataset: seed=42"), "{stdout}");
    assert!(stdout.contains("all_passed=true"), "{stdout}");
    assert!(!dir.join(".determinism-check").exists());

    for (label, expected) in [("train", 8usize), ("held-out", 4)] {
        let split = dir.join(label);
        let validation = split.join("validation.bin");
        let sidecar = sidecar_path(&validation);
        let manifest = read_manifest(&split.join("manifest.json")).unwrap();
        let rs = read_validation(&validation).unwrap();
        let traj = read_trajectory(&sidecar).unwrap();
        assert_eq!(manifest.schema_version, 1);
        assert_eq!(manifest.seed, 42);
        assert_eq!(manifest.split, label);
        assert_eq!(manifest.episode_count, expected as u32);
        assert_eq!(rs.episodes.len(), expected);
        assert_eq!(traj.episodes.len(), expected);
        assert!(traj.is_v3());
        assert_eq!(rs.header.target_count, 6);
        assert!(rs.episodes[0].targets.ncols() == 6);
        assert_eq!(manifest.feature_names.len(), 14, "canonical model frame");
        assert_eq!(
            manifest.target_names,
            vec![
                "g_roll",
                "g_pitch",
                "confidence",
                "ai_healthy",
                "g_accel_y",
                "g_accel_x"
            ]
        );
        assert_eq!(manifest.truth_names.len(), 26);
        assert_eq!(manifest.truth_names[24].as_str(), "g_accel_y");
        assert_eq!(manifest.truth_names[25].as_str(), "g_accel_x");
        assert_eq!(manifest.files.len(), 2);
        assert_eq!(manifest.files[0].name, "validation.bin");
        assert_eq!(manifest.files[1].name, "validation.trajectory");
        assert!(manifest.bounds.truth.len() == 26);
        assert!(manifest
            .bounds
            .targets
            .iter()
            .any(|b| b.name == "g_accel_x"));
        // The authoritative command target is emitted in both files and agrees
        // per tick: validation [g_accel_y, g_accel_x] == sidecar g_accel.
        for (ep, tr) in rs.episodes.iter().zip(traj.episodes.iter()) {
            for i in 0..ep.targets.nrows() {
                assert_eq!(ep.targets[[i, 4]], tr.ticks[i].g_accel[0]);
                assert_eq!(ep.targets[[i, 5]], tr.ticks[i].g_accel[1]);
            }
        }
        for rec in &manifest.files {
            let path = split.join(&rec.name);
            assert_eq!(rec.sha256, potok_sim::manifest::sha256_file(&path).unwrap());
            assert_eq!(rec.bytes, std::fs::metadata(&path).unwrap().len());
        }
        assert!(traj.episodes[0].ticks.iter().any(|t| t.quat[0] != 0.0));
    }

    let train = read_manifest(&dir.join("train/manifest.json")).unwrap();
    let held = read_manifest(&dir.join("held-out/manifest.json")).unwrap();
    assert!(train
        .scenario_ids
        .iter()
        .all(|id| !held.scenario_ids.contains(id)));
    assert_eq!(train.config_hash, held.config_hash);
    let report = std::fs::read_to_string(dir.join("quality_report.json")).unwrap();
    assert!(report.contains("\"all_passed\": true"));
    assert!(report.contains("deterministic_regeneration"));
    assert!(report.contains("split_scenarios_disjoint"));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn cli_generate_dataset_is_byte_identical_on_regeneration() {
    let a = dataset_dir("det-a");
    let b = dataset_dir("det-b");
    generate(&a, &[]);
    generate(&b, &[]);
    for name in [
        "train/validation.bin",
        "train/validation.trajectory",
        "train/manifest.json",
        "held-out/validation.bin",
        "held-out/validation.trajectory",
        "held-out/manifest.json",
        "quality_report.json",
    ] {
        assert_eq!(
            std::fs::read(a.join(name)).unwrap(),
            std::fs::read(b.join(name)).unwrap(),
            "{name} must be byte-identical across seeds"
        );
    }
    std::fs::remove_dir_all(&a).unwrap();
    std::fs::remove_dir_all(&b).unwrap();
}

#[test]
fn cli_generate_dataset_gate_failure_is_loud() {
    let dir = dataset_dir("gate-fail");
    let out = generate(&dir, &["--min-episodes-per-split", "9"]);
    assert!(!out.status.success(), "too few episodes must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("dataset quality gate failed"), "{stderr}");
    assert!(stderr.contains("episode_count"), "{stderr}");
    std::fs::remove_dir_all(&dir).unwrap();
}
