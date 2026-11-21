//! potok-cli: headless episode runner and ratatui dashboard.

use std::{fs, path::PathBuf};

use clap::Parser;
use potok_cli::tui;
use thiserror::Error;

use potok_sim::{
    episode::Episode,
    validation::{write_validation, FILE_NAME},
    Config, OutputMode, Simulator,
};

#[derive(Debug, Parser)]
#[command(name = "potok-cli", version, about = "Deterministic Potok-M simulator")]
struct Args {
    /// TOML config file; built-in defaults are used when omitted.
    #[arg(long, value_name = "PATH")]
    config: Option<PathBuf>,
    /// Override run.seed from the config.
    #[arg(long, value_name = "N")]
    seed: Option<u64>,
    /// Number of episodes to generate.
    #[arg(long, default_value_t = 1, value_name = "N")]
    episodes: usize,
    /// Directory for the validation binary file.
    #[arg(long, value_name = "DIR")]
    validation_out: Option<PathBuf>,
    /// Show the ratatui dashboard instead of a headless summary.
    #[arg(long)]
    tui: bool,
    /// Generate episodes serially instead of rayon-batching for validation.
    #[arg(long)]
    serial: bool,
}

#[derive(Debug, Error)]
enum CliError {
    #[error(transparent)]
    Sim(#[from] potok_sim::SimError),
    #[error("terminal error: {0}")]
    Terminal(String),
    #[error("no episodes requested")]
    NoEpisodes,
}

fn main() {
    if let Err(e) = run(Args::parse()) {
        eprintln!("potok-cli: {e}");
        std::process::exit(1);
    }
}

fn run(args: Args) -> Result<(), CliError> {
    if args.episodes == 0 {
        return Err(CliError::NoEpisodes);
    }
    let mut cfg = match &args.config {
        Some(p) => Config::from_file(p)?,
        None => Config::default(),
    };
    if let Some(seed) = args.seed {
        cfg.run.seed = seed;
    }
    cfg.run.episodes = args.episodes;
    let config_path = args
        .config
        .as_ref()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "<built-in defaults>".into());
    let sim = Simulator::new(cfg.clone())?;

    let wants_validation = args.validation_out.is_some()
        || matches!(cfg.output.mode, OutputMode::Validation | OutputMode::Both);
    let episodes = if args.tui {
        tui::run_interactive(&sim, args.episodes, &config_path).map_err(CliError::Terminal)?
    } else if !args.serial && wants_validation {
        parallel(&sim, args.episodes)
    } else {
        headless(&sim, &cfg, &config_path, args.episodes)
    };
    write_if_requested(&args, &cfg, &episodes)
}

fn headless(sim: &Simulator, cfg: &Config, config_path: &str, count: usize) -> Vec<Episode> {
    let mut total_frames = 0usize;
    let mut total_errors = 0usize;
    let mut episodes = Vec::new();
    println!(
        "seed={} episodes={} duration={}s dt={}s fps={}",
        cfg.run.seed, count, cfg.run.duration_s, cfg.run.dt, cfg.run.frame_rate_hz
    );
    println!("config={config_path} mode={:?}", cfg.output.mode);
    for result in sim.episodes(count) {
        match result {
            Ok(e) => {
                println!(
                    "episode {:>4}: frames={} mass={:.2}kg thrust_scale={:.3} gust={} phase_counts={:?}",
                    e.meta.index, e.frames.len(), e.meta.mass_kg, e.meta.thrust_scale,
                    e.meta.gust_start_s.map(|t| format!("{t:.1}s")).unwrap_or_else(|| "none".into()),
                    e.phase_counts,
                );
                total_frames += e.frames.len();
                total_errors += e.nan_frames;
                episodes.push(e);
            }
            Err(e) => {
                total_errors += 1;
                eprintln!("episode error: {e}");
            }
        }
    }
    println!("total: frames={total_frames} errors={total_errors}");
    episodes
}

fn parallel(sim: &Simulator, count: usize) -> Vec<Episode> {
    let mut total_frames = 0usize;
    let mut errors = 0usize;
    let mut out = Vec::new();
    for r in sim.run_episodes_parallel(count) {
        match r {
            Ok(e) => {
                total_frames += e.frames_generated;
                out.push(e);
            }
            Err(e) => {
                errors += 1;
                eprintln!("episode error: {e}");
            }
        }
    }
    println!(
        "seed={} episodes={} mode=parallel",
        sim.config().run.seed,
        count
    );
    println!("total: frames={total_frames} errors={errors}");
    out
}

fn write_if_requested(args: &Args, cfg: &Config, episodes: &[Episode]) -> Result<(), CliError> {
    if episodes.is_empty() {
        return Ok(());
    }
    let dir = match &args.validation_out {
        Some(d) => Some(d.clone()),
        None if cfg.output.mode == OutputMode::Validation
            || cfg.output.mode == OutputMode::Both =>
        {
            Some(PathBuf::from("validation"))
        }
        None => None,
    };
    if let Some(dir) = dir {
        fs::create_dir_all(&dir).map_err(|e| potok_sim::SimError::Io(e))?;
        let out = dir.join(FILE_NAME);
        write_validation(&out, cfg, episodes)?;
        println!("validation: {}", out.display());
    }
    Ok(())
}
