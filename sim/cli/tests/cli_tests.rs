use std::{path::PathBuf, process::Command};

use potok_cli::tui::{render, DashboardState};
use potok_sim::{scenario::Split, validation::read_validation, Config};
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
        rs.episodes.iter().map(|e| e.scenario_id).collect::<Vec<_>>(),
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
