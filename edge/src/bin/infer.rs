use potok_edge::frame::PotokUartV1;
use potok_edge::lowlat::LowlatLayout;
use potok_edge::runtime::Runtime;
use potok_edge::uart::{serve_with_layout, DEFAULT_BAUD};
use std::env;

const LOWLAT_BAUD: u32 = 2_000_000;

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut model_path = String::new();
    let mut device = String::new();
    let mut profile = String::from("compat");
    let mut baud: Option<u32> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" if i + 1 < args.len() => {
                model_path = args[i + 1].clone();
                i += 2;
            }
            "--device" if i + 1 < args.len() => {
                device = args[i + 1].clone();
                i += 2;
            }
            "--baud" if i + 1 < args.len() => {
                baud = Some(args[i + 1].parse().unwrap_or_else(|_| {
                    eprintln!("invalid baud rate");
                    std::process::exit(2);
                }));
                i += 2;
            }
            "--profile" if i + 1 < args.len() => {
                profile = args[i + 1].clone();
                i += 2;
            }
            other => {
                eprintln!("unknown argument: {other}");
                eprintln!("usage: infer --model MODEL.bin --device /dev/ttyS0 [--profile compat|lowlat] [--baud N]");
                std::process::exit(2);
            }
        }
    }
    if model_path.is_empty() || device.is_empty() {
        eprintln!("usage: infer --model MODEL.bin --device /dev/ttyS0 [--profile compat|lowlat] [--baud N]");
        std::process::exit(2);
    }
    let baud = baud.unwrap_or(match profile.as_str() {
        "compat" => DEFAULT_BAUD,
        "lowlat" => LOWLAT_BAUD,
        other => {
            eprintln!("unknown profile {other}; expected compat or lowlat");
            std::process::exit(2);
        }
    });
    let runtime = Runtime::load(&model_path).unwrap_or_else(|e| {
        eprintln!("model load failed: {e}");
        std::process::exit(1);
    });
    eprintln!(
        "potok-edge: {} profile, {} bytes resident, 50-frame window, baud {baud}",
        profile,
        runtime.memory_bytes()
    );
    let served = match profile.as_str() {
        "compat" => serve_with_layout(PotokUartV1, runtime, &device, baud),
        _ => serve_with_layout(LowlatLayout::new(), runtime, &device, baud),
    };
    if let Err(e) = served {
        eprintln!("potok-edge: {e}");
        std::process::exit(1);
    }
}
