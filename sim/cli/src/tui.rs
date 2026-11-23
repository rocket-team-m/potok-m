//! Minimal ratatui dashboard. Rendering is a pure function of `DashboardState`,
//! so tests can draw it on a `TestBackend` without allocating a TTY.

use std::{io::stdout, time::Duration};

use crossterm::event::{self, Event, KeyCode};
use potok_sim::{episode::Episode, Config, Simulator};
use ratatui::{
    backend::Backend,
    layout::{Constraint, Layout},
    style::{Color, Modifier, Style},
    widgets::{Block, Gauge, Paragraph},
    Frame, Terminal,
};

pub struct DashboardState {
    pub config_path: String,
    pub seed: u64,
    pub episodes_total: usize,
    pub episode: usize,
    pub frames: usize,
    pub frames_total: usize,
    pub wind_v10: f64,
    pub shear_delta: f64,
    pub gust_probability: f64,
    pub gust_start: Option<f64>,
    pub gust_amplitude: f64,
    pub turbulence_sigma: f64,
    pub errors: usize,
    pub last_error: String,
    pub running: bool,
}

impl DashboardState {
    pub fn from_config(cfg: &Config, episodes_total: usize) -> Self {
        Self {
            config_path: String::new(),
            seed: cfg.run.seed,
            episodes_total,
            episode: 0,
            frames: 0,
            frames_total: cfg.frames_per_episode() * episodes_total,
            wind_v10: cfg.wind.v_10,
            shear_delta: cfg.wind.shear_delta,
            gust_probability: cfg.wind.gust_probability,
            gust_start: None,
            gust_amplitude: cfg.wind.gust_amplitude,
            turbulence_sigma: cfg.wind.turbulence_sigma,
            errors: 0,
            last_error: String::new(),
            running: true,
        }
    }

    pub fn on_episode(&mut self, episode: &Episode) {
        self.episode = episode.meta.index as usize + 1;
        self.frames += episode.frames.len();
        self.gust_start = episode.meta.gust_start_s;
        self.gust_amplitude = episode.meta.gust_amplitude;
        self.errors += episode.nan_frames;
    }

    pub fn summary_lines(&self) -> Vec<String> {
        vec![
            format!("Episode   {}/{}", self.episode, self.episodes_total),
            format!("Frames    {}/{}", self.frames, self.frames_total),
            format!("Seed      {}", self.seed),
            format!("Errors    {} {}", self.errors, self.last_error),
        ]
    }
}

pub fn render(f: &mut Frame, s: &DashboardState) {
    let chunks = Layout::vertical([
        Constraint::Length(3),
        Constraint::Length(9),
        Constraint::Length(7),
        Constraint::Min(4),
    ])
    .split(f.area());
    f.render_widget(
        Paragraph::new("POTOK-M  deterministic simulator")
            .style(
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )
            .block(Block::bordered().title("Status")),
        chunks[0],
    );
    let summary = s.summary_lines().join("\n");
    f.render_widget(
        Paragraph::new(summary).block(Block::bordered().title("Progress")),
        chunks[1],
    );
    let ratio = if s.frames_total == 0 {
        0.0
    } else {
        (s.frames as f64 / s.frames_total as f64).min(1.0)
    };
    f.render_widget(
        Gauge::default()
            .ratio(ratio)
            .label("generation")
            .block(Block::bordered().title("Completion")),
        chunks[2],
    );
    let wind = format!(
        "v_10        {:.2} m/s\nshear_delta {:.2} m/s\ngust p      {:.2}\ngust start  {}\ngust amp    {:.2} m/s\nturb sigma  {:.2} m/s\nconfig      {}",
        s.wind_v10, s.shear_delta, s.gust_probability,
        s.gust_start.map(|v| format!("{v:.2} s")).unwrap_or_else(|| "none".into()),
        s.gust_amplitude, s.turbulence_sigma, s.config_path,
    );
    f.render_widget(
        Paragraph::new(wind).block(Block::bordered().title("Wind / gust")),
        chunks[3],
    );
}

/// Run the dashboard on a real TTY. `q` or `Esc` requests an early stop.
pub fn run<B: Backend>(
    terminal: &mut Terminal<B>,
    sim: &Simulator,
    count: usize,
    config_path: &str,
) -> Result<Vec<Episode>, String> {
    let mut episodes = Vec::new();
    let mut state = DashboardState::from_config(sim.config(), count);
    state.config_path = config_path.to_string();
    terminal
        .draw(|f| render(f, &state))
        .map_err(|e| e.to_string())?;
    for result in sim.episodes(count) {
        if !state.running {
            break;
        }
        match result {
            Ok(episode) => {
                state.on_episode(&episode);
                episodes.push(episode);
            }
            Err(e) => {
                state.errors += 1;
                state.last_error = e.to_string();
            }
        }
        terminal
            .draw(|f| render(f, &state))
            .map_err(|e| e.to_string())?;
        if event::poll(Duration::from_millis(0)).map_err(|e| e.to_string())? {
            if let Event::Key(k) = event::read().map_err(|e| e.to_string())? {
                if matches!(k.code, KeyCode::Char('q') | KeyCode::Esc) {
                    state.running = false;
                }
            }
        }
    }
    Ok(episodes)
}

/// Convenience wrapper used by the CLI binary.
pub fn run_interactive(
    sim: &Simulator,
    count: usize,
    config_path: &str,
) -> Result<Vec<Episode>, String> {
    crossterm::terminal::enable_raw_mode().map_err(|e| e.to_string())?;
    let mut terminal = ratatui::Terminal::new(ratatui::backend::CrosstermBackend::new(stdout()))
        .map_err(|e| e.to_string())?;
    let result = run(&mut terminal, sim, count, config_path);
    let _ = crossterm::terminal::disable_raw_mode();
    result
}
