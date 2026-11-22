use potok_edge::runtime::Runtime;
use potok_edge::uart::{serve, DEFAULT_BAUD};
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut model_path = String::new();
    let mut device = String::new();
    let mut baud = DEFAULT_BAUD;
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
                baud = args[i + 1].parse().unwrap_or_else(|_| {
                    eprintln!("invalid baud rate");
                    std::process::exit(2);
                });
                i += 2;
            }
            other => {
                eprintln!("unknown argument: {other}");
                eprintln!("usage: infer --model MODEL.bin --device /dev/ttyS0 [--baud 921600]");
                std::process::exit(2);
            }
        }
    }
    if model_path.is_empty() || device.is_empty() {
        eprintln!("usage: infer --model MODEL.bin --device /dev/ttyS0 [--baud 921600]");
        std::process::exit(2);
    }
    let runtime = Runtime::load(&model_path).unwrap_or_else(|e| {
        eprintln!("model load failed: {e}");
        std::process::exit(1);
    });
    eprintln!(
        "potok-edge: {}, {} bytes resident, 50-frame window",
        if runtime.is_v2() {
            "v2 export"
        } else {
            "legacy POTM"
        },
        runtime.memory_bytes()
    );
    if let Err(e) = serve(runtime, &device, baud) {
        eprintln!("potok-edge: {e}");
        std::process::exit(1);
    }
}
