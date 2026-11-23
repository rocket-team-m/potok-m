// SPDX-License-Identifier: GPL-3.0-or-later
//! Headless command adapter for the POTOKTRJ v3 -> glTF playback export.
//!
//! Std-only and side-effect free except for the injected exporter. `potok-cli`
//! wires in `potok_sim::playback3d::export_file`; the core test harness supplies
//! the same function to exercise this file without changing any manifest.
//! No physics, no GUI, no global state.

use std::{
    fmt,
    path::{Path, PathBuf},
};

pub const USAGE: &str = "playback3d --trajectory <POTOKTRJ-v3> --out <FILE.json>";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Args {
    pub trajectory: PathBuf,
    pub out: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Summary {
    pub episodes: usize,
    pub ticks: usize,
    pub markers: usize,
    pub bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CliError {
    Usage(String),
    Export(String),
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Usage(msg) => write!(f, "playback3d usage error: {msg}"),
            Self::Export(msg) => write!(f, "playback3d export failed: {msg}"),
        }
    }
}

impl std::error::Error for CliError {}

/// Parse the `playback3d` invocation. The command token itself is optional so
/// both `potok-cli playback3d --trajectory …` and `--trajectory …` work.
pub fn parse<I>(args: I) -> Result<Args, CliError>
where
    I: IntoIterator<Item = String>,
{
    let mut trajectory = None;
    let mut out = None;
    let mut it = args.into_iter();
    while let Some(arg) = it.next() {
        if arg == "playback3d" {
            continue;
        }
        let value = it
            .next()
            .ok_or_else(|| CliError::Usage(format!("{arg} needs a value; {USAGE}")))?;
        match arg.as_str() {
            "--trajectory" if trajectory.is_none() => trajectory = Some(PathBuf::from(value)),
            "--out" if out.is_none() => out = Some(PathBuf::from(value)),
            "--trajectory" => {
                return Err(CliError::Usage(format!("duplicate --trajectory; {USAGE}")))
            }
            "--out" => return Err(CliError::Usage(format!("duplicate --out; {USAGE}"))),
            other => {
                return Err(CliError::Usage(format!(
                    "unknown argument {other:?}; {USAGE}"
                )))
            }
        }
    }
    Ok(Args {
        trajectory: trajectory
            .ok_or_else(|| CliError::Usage(format!("--trajectory is required; {USAGE}")))?,
        out: out.ok_or_else(|| CliError::Usage(format!("--out is required; {USAGE}")))?,
    })
}

/// One-line, deterministic result for logs and tests.
pub fn format_summary(s: &Summary) -> String {
    format!(
        "playback3d: episodes={} ticks={} markers={} bytes={}",
        s.episodes, s.ticks, s.markers, s.bytes
    )
}

/// Parse and export through an injected `fn(trajectory, out) -> Summary`.
pub fn run_with<I, F>(args: I, export: F) -> Result<Summary, CliError>
where
    I: IntoIterator<Item = String>,
    F: FnOnce(&Path, &Path) -> Result<Summary, String>,
{
    let args = parse(args)?;
    export(&args.trajectory, &args.out).map_err(CliError::Export)
}

/// Process entry point: the caller supplies the core exporter, e.g.
/// `playback3d::main_with(|t, o| core_export(t, o).map_err(|e| e.to_string()))`.
pub fn main_with<F>(export: F) -> !
where
    F: FnOnce(&Path, &Path) -> Result<Summary, String>,
{
    match run_with(std::env::args().skip(1), export) {
        Ok(summary) => {
            println!("{}", format_summary(&summary));
            std::process::exit(0);
        }
        Err(e) => {
            eprintln!("potok-cli: {e}");
            std::process::exit(1);
        }
    }
}
