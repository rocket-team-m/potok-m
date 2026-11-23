//! potok-cli: headless episode runner and ratatui dashboard.

use std::{fs, path::PathBuf};

use clap::{Parser, Subcommand};
use potok_cli::tui;
use thiserror::Error;

use potok_sim::{
    config_hash,
    control::ControlFile,
    episode::Episode,
    file_record,
    manifest::{Bounds, DatasetManifest},
    quality::{
        check_dataset, check_determinism, check_split_disjoint, observed_bounds, QualityReport,
        QUALITY_SCHEMA_VERSION,
    },
    scenario::{scenario, Split},
    trajectory::{
        apogee_m, lateral_rms_m, read_trajectory, sidecar_path, write_trajectory,
        write_trajectory_v3,
    },
    validation::{read_validation, write_validation, FILE_NAME},
    write_manifest, Config, OutputMode, Simulator, PHYSICS_FEATURES,
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
    /// Whole-dataset generation (train + held-out + manifest + quality report).
    #[command(subcommand)]
    command: Option<DatasetCommand>,
}

#[derive(Debug, Clone, Subcommand)]
enum DatasetCommand {
    /// Generate train + held-out validation sets, v2 truth sidecars, manifests
    /// and a quality report in one deterministic invocation.
    #[command(name = "generate-dataset")]
    GenerateDataset {
        /// Explicit seed for both splits (required).
        #[arg(long, value_name = "N")]
        seed: u64,
        /// Number of train-split episodes.
        #[arg(long, value_name = "N")]
        train_episodes: usize,
        /// Number of held-out-split episodes.
        #[arg(long, value_name = "N")]
        held_out_episodes: usize,
        /// Output directory (created if absent).
        #[arg(long, value_name = "DIR")]
        out: PathBuf,
        /// Optional TOML config; built-in defaults when omitted.
        #[arg(long, value_name = "PATH")]
        config: Option<PathBuf>,
        /// Minimum episodes per split; failure is LOUD.
        #[arg(long, default_value_t = 4, value_name = "N")]
        min_episodes_per_split: usize,
        /// Optional TSV command file applied to every episode.
        #[arg(long, value_name = "PATH")]
        control_file: Option<PathBuf>,
    },
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
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
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
    if let Some(DatasetCommand::GenerateDataset {
        seed,
        train_episodes,
        held_out_episodes,
        out,
        config,
        min_episodes_per_split,
        control_file,
    }) = &args.command
    {
        return generate_dataset(
            *seed,
            *train_episodes,
            *held_out_episodes,
            out,
            config.as_deref(),
            *min_episodes_per_split,
            control_file.as_deref(),
        );
    }
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

/// Whole-dataset generation: one invocation writes both splits, their v2
/// privileged-truth sidecars, one manifest per split and a quality report.
/// All outputs are functions of (seed, config, counts) only.
fn generate_dataset(
    seed: u64,
    train_episodes: usize,
    held_out_episodes: usize,
    out: &PathBuf,
    config: Option<&std::path::Path>,
    min_episodes_per_split: usize,
    control_file: Option<&std::path::Path>,
) -> Result<(), CliError> {
    if train_episodes == 0 || held_out_episodes == 0 {
        return Err(CliError::NoEpisodes);
    }
    let mut cfg = match config {
        Some(p) => Config::from_file(p)?,
        None => Config::default(),
    };
    cfg.run.seed = seed;
    // The dataset's model frame is the canonical 14 channels: eight raw plus
    // six physics features. Appending is idempotent for custom configs.
    for name in PHYSICS_FEATURES {
        if !cfg.output.features.iter().any(|f| f == name) {
            cfg.output.features.push((*name).to_string());
        }
    }
    let config_hash = config_hash(&cfg)?;
    let sim = Simulator::new(cfg.clone())?;
    let control = match control_file {
        Some(p) => Some(ControlFile::from_path(p, cfg.frames_per_episode())?),
        None => None,
    };

    let train_idx = indices_for(Split::Train, train_episodes);
    let held_idx = indices_for(Split::HeldOut, held_out_episodes);
    let train = generate_episodes(&sim, control.as_ref(), &train_idx)?;
    let held = generate_episodes(&sim, control.as_ref(), &held_idx)?;

    fs::create_dir_all(out)?;
    let train_dir = out.join(Split::Train.label());
    let held_dir = out.join(Split::HeldOut.label());
    write_split(&train_dir, &cfg, &train)?;
    write_split(&held_dir, &cfg, &held)?;

    let (tv, tt) = read_split(&train_dir)?;
    let (hv, ht) = read_split(&held_dir)?;
    let mut report = QualityReport {
        schema_version: QUALITY_SCHEMA_VERSION,
        config_hash: config_hash.clone(),
        seed,
        checks: Vec::new(),
        all_passed: true,
    };
    report
        .checks
        .extend(check_dataset("train", &cfg, &tv, &tt, min_episodes_per_split)?.checks);
    report
        .checks
        .extend(check_dataset("held-out", &cfg, &hv, &ht, min_episodes_per_split)?.checks);
    report.checks.push(check_split_disjoint(&tv, &hv)?);

    // Regenerate from scratch and compare the actual bytes, not just the
    // in-memory tensors. The check directory is removed on success.
    let check_dir = out.join(".determinism-check");
    let _ = fs::remove_dir_all(&check_dir);
    let again_train_dir = check_dir.join(Split::Train.label());
    let again_held_dir = check_dir.join(Split::HeldOut.label());
    write_split(
        &again_train_dir,
        &cfg,
        &generate_episodes(&sim, control.as_ref(), &train_idx)?,
    )?;
    write_split(
        &again_held_dir,
        &cfg,
        &generate_episodes(&sim, control.as_ref(), &held_idx)?,
    )?;
    let mut pairs: Vec<(String, Vec<u8>, Vec<u8>)> = Vec::new();
    for (label, dir, again) in [
        (Split::Train.label(), &train_dir, &again_train_dir),
        (Split::HeldOut.label(), &held_dir, &again_held_dir),
    ] {
        for name in [FILE_NAME, "validation.trajectory"] {
            pairs.push((
                format!("{label}/{name}"),
                fs::read(dir.join(name))?,
                fs::read(again.join(name))?,
            ));
        }
    }
    let pair_refs: Vec<(&str, &[u8], &[u8])> = pairs
        .iter()
        .map(|(n, a, b)| (n.as_str(), a.as_slice(), b.as_slice()))
        .collect();
    report.checks.push(check_determinism(&pair_refs)?);
    fs::remove_dir_all(&check_dir)?;

    let bounds_train: Bounds = observed_bounds(&cfg, &tv, &tt);
    let bounds_held: Bounds = observed_bounds(&cfg, &hv, &ht);
    write_manifest(
        &train_dir.join("manifest.json"),
        &build_manifest(&cfg, seed, Split::Train, &tv, &tt, &train_dir, bounds_train)?,
    )?;
    write_manifest(
        &held_dir.join("manifest.json"),
        &build_manifest(&cfg, seed, Split::HeldOut, &hv, &ht, &held_dir, bounds_held)?,
    )?;
    report.all_passed = report.checks.iter().all(|c| c.ok);
    fs::write(
        out.join("quality_report.json"),
        potok_sim::quality::quality_report_to_string(&report)?,
    )?;

    println!(
        "generate-dataset: seed={seed} train={train_episodes} held-out={held_out_episodes} out={}",
        out.display()
    );
    println!(
        "manifest: {} / {}",
        train_dir.join("manifest.json").display(),
        held_dir.join("manifest.json").display()
    );
    println!(
        "quality_report: {} ({} checks, all_passed={})",
        out.join("quality_report.json").display(),
        report.checks.len(),
        report.all_passed
    );
    Ok(())
}

/// Episode indices whose scenario belongs to `split`, in ascending order.
fn indices_for(split: Split, count: usize) -> Vec<u64> {
    (0u64..)
        .filter(|i| scenario(*i).split == split)
        .take(count)
        .collect()
}

fn generate_episodes(
    sim: &Simulator,
    control: Option<&ControlFile>,
    indices: &[u64],
) -> Result<Vec<Episode>, CliError> {
    let mut out = Vec::with_capacity(indices.len());
    for &i in indices {
        let e = match control {
            Some(c) => sim.run_episode_with_control(i, &|f| {
                c.for_frame(f).expect("control file covers every frame")
            })?,
            None => sim.run_episode(i)?,
        };
        out.push(e);
    }
    Ok(out)
}

fn write_split(dir: &PathBuf, cfg: &Config, episodes: &[Episode]) -> Result<(), CliError> {
    fs::create_dir_all(dir)?;
    let validation = dir.join(FILE_NAME);
    write_validation(&validation, cfg, episodes)?;
    // The generated dataset carries the authoritative command target on both
    // sides: six validation targets and the v3 per-tick g_accel sidecar.
    write_trajectory_v3(&sidecar_path(&validation), cfg, episodes)?;
    Ok(())
}

fn read_split(
    dir: &PathBuf,
) -> Result<
    (
        potok_sim::validation::ValidationSet,
        potok_sim::TrajectorySet,
    ),
    CliError,
> {
    let validation = dir.join(FILE_NAME);
    Ok((
        read_validation(&validation)?,
        read_trajectory(&sidecar_path(&validation))?,
    ))
}

fn build_manifest(
    cfg: &Config,
    seed: u64,
    split: Split,
    v: &potok_sim::validation::ValidationSet,
    _t: &potok_sim::TrajectorySet,
    dir: &PathBuf,
    bounds: Bounds,
) -> Result<DatasetManifest, CliError> {
    let mut scenario_ids: Vec<u16> = v.episodes.iter().map(|e| e.scenario_id).collect();
    scenario_ids.sort_unstable();
    scenario_ids.dedup();
    let validation = dir.join(FILE_NAME);
    let sidecar = sidecar_path(&validation);
    let manifest = DatasetManifest {
        schema_version: potok_sim::manifest::SCHEMA_VERSION,
        config_hash: config_hash(cfg)?,
        seed,
        split: split.label().to_string(),
        frame_count: (v.episodes.len() as u64) * (v.header.frames_per_episode as u64),
        episode_count: v.episodes.len() as u32,
        frames_per_episode: v.header.frames_per_episode,
        scenario_ids,
        feature_names: cfg.output.features.clone(),
        target_names: potok_sim::manifest::default_target_names(),
        truth_names: potok_sim::manifest::default_truth_names(),
        bounds,
        files: vec![
            file_record(FILE_NAME, &validation)?,
            file_record("validation.trajectory", &sidecar)?,
        ],
    };
    Ok(manifest)
}
