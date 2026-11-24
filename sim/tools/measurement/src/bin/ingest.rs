//! Run the default (SYNTHETIC) ingest and print its provenance digest as JSON.
//!
//! A corrupt or missing source exits non-zero with the offending path, row,
//! column and fix on stderr. No dataset or config file is written.

use std::env;
use std::path::PathBuf;
use std::process::exit;

use potok_measurement::bundle::{default_paths, ingest_recipe};

fn main() {
    let mut args = env::args_os().skip(1);
    let mut recipe = None;
    let mut root = None;
    while let Some(arg) = args.next() {
        match arg.to_str() {
            Some("--recipe") => match args.next() {
                Some(value) => recipe = Some(PathBuf::from(value)),
                None => fail("--recipe needs a path"),
            },
            Some("--root") => match args.next() {
                Some(value) => root = Some(PathBuf::from(value)),
                None => fail("--root needs a directory"),
            },
            Some("--help") => {
                println!("usage: ingest [--recipe PATH] [--root DIR]");
                println!("defaults to <repo>/config/measurement.toml and <repo>");
                return;
            }
            other => fail(&format!("unknown argument {other:?}; use --help")),
        }
    }

    let (default_recipe, default_root) = default_paths();
    let recipe = recipe.unwrap_or(default_recipe);
    let root = root.unwrap_or(default_root);
    match ingest_recipe(&recipe, &root) {
        Ok(bundle) => match bundle.to_json() {
            Ok(json) => println!("{json}"),
            Err(e) => fail(&e.to_string()),
        },
        Err(e) => fail(&e.to_string()),
    }
}

fn fail(message: &str) -> ! {
    eprintln!("{message}");
    exit(1);
}
