//! Write a canonical handcrafted model file for manual/hardware testing.

use potok_edge::fixture::{build_fixture, FixtureConfig};
use std::env;
use std::fs;

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut out = String::new();
    let mut seed = 1u64;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--out" if i + 1 < args.len() => {
                out = args[i + 1].clone();
                i += 2;
            }
            "--seed" if i + 1 < args.len() => {
                seed = args[i + 1].parse().unwrap_or_else(|_| {
                    eprintln!("invalid seed");
                    std::process::exit(2);
                });
                i += 2;
            }
            other => {
                eprintln!("unknown argument: {other}");
                eprintln!("usage: make-fixture --out MODEL.bin [--seed 1]");
                std::process::exit(2);
            }
        }
    }
    if out.is_empty() {
        eprintln!("usage: make-fixture --out MODEL.bin [--seed 1]");
        std::process::exit(2);
    }
    let bytes = build_fixture(FixtureConfig::default(), seed);
    fs::write(&out, &bytes).unwrap_or_else(|e| {
        eprintln!("write failed: {e}");
        std::process::exit(1);
    });
    println!("wrote {} bytes to {out}", bytes.len());
}
