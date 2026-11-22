//! potok-cli: headless episode runner and ratatui dashboard.

use std::{fs, path::PathBuf};

use clap::Parser;
use potok_cli::tui;
use thiserror::Error;

use potok_sim::{
    control::ControlFile,
    episode::Episode,
    scenario::{scenario, Split},
    trajectory::{apogee_m, lateral_rms_m, sidecar_path, write_trajectory},
    validation::{write_validation, FILE_NAME},
    Config, OutputMode, Simulator, PHYSICS_FEATURES,
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
    /// Keep only one side of the withheld-scenario split in the validation file.
    #[arg(long, value_enum, default_value_t = SplitArg::All, value_name = "SPLIT")]
    split: SplitArg,
    /// Append the six physics features to the eight default channels (14 total).
    #[arg(long)]
    physics_features: bool,
    /// TSV control file: frame_index, g_roll_deg, g_pitch_deg (one row per tick).
    /// Commands reach the vehicle dynamics through the fin-control seam.
    #[arg(long, value_name = "PATH")]
    control_file: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum SplitArg {
    All,
    Train,
    HeldOut,
}

#[derive(Debug, Error)]
enum CliError {
    #[error(transparent)]
    Sim(#[from] potok_sim::SimError),
    #[error("terminal error: {0}")]
    Terminal(String),
    #[error("no episodes requested")]
    NoEpisodes,
    #[error("--control-file requires headless generation (drop --tui)")]
    ControlTui,
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
    if args.physics_features {
        cfg.output
            .features
            .extend(PHYSICS_FEATURES.iter().map(|s| s.to_string()));
    }
    cfg.run.episodes = args.episodes;
    let config_path = args
        .config
        .as_ref()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "<built-in defaults>".into());
    let sim = Simulator::new(cfg.clone())?;
    let control = match &args.control_file {
        Some(p) => Some(ControlFile::from_path(p, cfg.frames_per_episode())?),
        None => None,
    };
    if args.tui && control.is_some() {
        return Err(CliError::ControlTui);
    }

    let wants_validation = args.validation_out.is_some()
        || matches!(cfg.output.mode, OutputMode::Validation | OutputMode::Both);
    let episodes = if let Some(control) = &control {
        headless_with_control(&sim, &cfg, &config_path, args.episodes, control)
    } else if args.tui {
        tui::run_interactive(&sim, args.episodes, &config_path).map_err(CliError::Terminal)?
    } else if !args.serial && wants_validation {
        parallel(&sim, args.episodes)
    } else {
        headless(&sim, &cfg, &config_path, args.episodes)
    };
    let episodes: Vec<Episode> = episodes
        .into_iter()
        .filter(|e| match args.split {
            SplitArg::All => true,
            SplitArg::Train => scenario(e.meta.index).split == Split::Train,
            SplitArg::HeldOut => scenario(e.meta.index).split == Split::HeldOut,
        })
        .collect();
    println!(
        "split={} kept={}/{}",
        args.split.as_str(),
        episodes.len(),
        args.episodes
    );
    write_if_requested(&args, &cfg, &episodes)
}

impl SplitArg {
    fn as_str(self) -> &'static str {
        match self {
            SplitArg::All => "all",
            SplitArg::Train => "train",
            SplitArg::HeldOut => "held-out",
        }
    }
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

/// Control runs are serial: the command table is indexed by frame sequence,
/// and one episode at a time keeps the flown truth and the A/B comparison
/// trivially reproducible. No parallel/control race is possible.
fn headless_with_control(
    sim: &Simulator,
    cfg: &Config,
    config_path: &str,
    count: usize,
    control: &ControlFile,
) -> Vec<Episode> {
    let mut episodes = Vec::new();
    let mut errors = 0usize;
    println!(
        "control: rows={} mode=serial seed={} episodes={} config={config_path}",
        control.len(),
        cfg.run.seed,
        count
    );
    for i in 0..count as u64 {
        match sim.run_episode_with_control(i, &|f| {
            control
                .for_frame(f)
                .expect("control file covers every frame")
        }) {
            Ok(e) => {
                println!(
                    "episode {:>4}: frames={} apogee={:.1}m lateral_rms={:.1}m max_speed={:.1}m/s",
                    e.meta.index,
                    e.frames.len(),
                    apogee_m(&e.trajectory),
                    lateral_rms_m(&e.trajectory),
                    e.max_speed_mps,
                );
                episodes.push(e);
            }
            Err(e) => {
                errors += 1;
                eprintln!("episode error: {e}");
            }
        }
    }
    println!("total: episodes={} errors={errors}", episodes.len());
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
        fs::create_dir_all(&dir).map_err(potok_sim::SimError::Io)?;
        let out = dir.join(FILE_NAME);
        write_validation(&out, cfg, episodes)?;
        println!("validation: {}", out.display());
        let traj = sidecar_path(&out);
        write_trajectory(&traj, cfg, episodes)?;
        println!("trajectory: {}", traj.display());
    }
    Ok(())
}
