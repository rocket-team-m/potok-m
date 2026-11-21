use potok_cli::tui::{render, DashboardState};
use potok_sim::Config;
use ratatui::{backend::TestBackend, Terminal};

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
