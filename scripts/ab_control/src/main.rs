//! Thin driver for the potok-sim command->dynamics seam.
//!
//! Lives under scripts/ and only READS sim/ through its public API. The CLI's
//! --control-file applies one TSV to every episode in a run, so per-episode
//! controls need this driver: it calls `run_episode_with_control(index, ...)`.
//! Writes validation.bin plus the flown-truth trajectory sidecar.

use std::{error::Error, fs, path::PathBuf};

use potok_sim::{
    apogee_m, lateral_rms_m, sidecar_path, write_trajectory, Config, ControlFile,
    PHYSICS_FEATURES, Simulator,
};
use potok_sim::validation::write_validation;

struct Args {
    config: PathBuf,
    seed: u64,
    out_dir: PathBuf,
    indices: Vec<u64>,
    commands_dir: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut config = None;
    let mut seed = None;
    let mut out_dir = None;
    let mut indices = None;
    let mut commands_dir = None;
    let mut i = 0;
    while i < raw.len() {
        let key = raw[i].as_str();
        i += 1;
        let value = raw.get(i).ok_or_else(|| format!("{key} needs a value"))?.clone();
        i += 1;
        match key {
            "--config" => config = Some(PathBuf::from(value)),
            "--seed" => seed = Some(value.parse::<u64>().map_err(|e| e.to_string())?),
            "--out-dir" => out_dir = Some(PathBuf::from(value)),
            "--indices" => {
                indices = Some(
                    value
                        .split(',')
                        .filter(|s| !s.is_empty())
                        .map(|s| s.parse::<u64>().map_err(|e| e.to_string()))
                        .collect::<Result<Vec<_>, _>>()?,
                )
            }
            "--commands-dir" => commands_dir = Some(PathBuf::from(value)),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(Args {
        config: config.ok_or("--config is required")?,
        seed: seed.ok_or("--seed is required")?,
        out_dir: out_dir.ok_or("--out-dir is required")?,
        indices: indices.ok_or("--indices is required")?,
        commands_dir,
    })
}

fn run(args: Args) -> Result<(), Box<dyn Error>> {
    let mut cfg = Config::from_file(&args.config)?;
    cfg.run.seed = args.seed;
    cfg.output
        .features
        .extend(PHYSICS_FEATURES.iter().map(|s| s.to_string()));
    let frames = cfg.frames_per_episode();
    let sim = Simulator::new(cfg.clone())?;
    fs::create_dir_all(&args.out_dir)?;

    let mut episodes = Vec::with_capacity(args.indices.len());
    for index in &args.indices {
        let episode = match &args.commands_dir {
            None => sim.run_episode(*index)?,
            Some(dir) => {
                let path = dir.join(format!("{index}.tsv"));
                let table = ControlFile::from_path(&path, frames)?;
                sim.run_episode_with_control(*index, &|f| table.for_frame(f).unwrap())?
            }
        };
        println!(
            "episode {}: scenario={} apogee={:.6} lateral_rms={:.6}",
            episode.meta.index,
            episode.meta.scenario_id,
            apogee_m(&episode.trajectory),
            lateral_rms_m(&episode.trajectory)
        );
        episodes.push(episode);
    }

    let validation = args.out_dir.join("validation.bin");
    write_validation(&validation, &cfg, &episodes)?;
    write_trajectory(&sidecar_path(&validation), &cfg, &episodes)?;
    println!("validation: {}", validation.display());
    println!("trajectory: {}", sidecar_path(&validation).display());
    Ok(())
}

fn main() {
    if let Err(e) = parse_args().and_then(|a| run(a).map_err(|e| e.to_string())) {
        eprintln!("ab-control: {e}");
        std::process::exit(1);
    }
}
