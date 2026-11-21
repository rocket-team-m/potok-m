mod gen;
mod model;

use std::path::PathBuf;

fn die(msg: impl std::fmt::Display) -> ! {
    eprintln!("potok-generator: error: {msg}");
    std::process::exit(2);
}

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("generator crate must live directly under protocol/")
        .to_path_buf();
    let yaml = root.join("uart.yaml");
    let out = root.join("generated");

    let mut check_only = false;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--check" => check_only = true,
            other => die(format!("unknown argument {other:?} (expected --check)")),
        }
    }

    let text = std::fs::read_to_string(&yaml)
        .unwrap_or_else(|e| die(format!("cannot read {}: {e}", yaml.display())));
    let spec = model::parse(&text).unwrap_or_else(|e| die(format!("{}: {e}", yaml.display())));
    model::validate(&spec)
        .unwrap_or_else(|e| die(format!("{}: invalid spec: {e}", yaml.display())));

    let files = [
        ("protocol.rs", gen::gen_rust(&spec)),
        ("protocol.h", gen::gen_c(&spec)),
        ("protocol.py", gen::gen_python(&spec)),
    ];

    if check_only {
        for (name, content) in &files {
            let path = out.join(name);
            let disk = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| die(format!("cannot read {}: {e}", path.display())));
            if disk != *content {
                die(format!(
                    "{} is stale — rerun `cargo run -p potok-generator`",
                    path.display()
                ));
            }
        }
        println!("protocol/generated: up to date");
        return;
    }

    std::fs::create_dir_all(&out)
        .unwrap_or_else(|e| die(format!("cannot create {}: {e}", out.display())));
    for (name, content) in &files {
        let path = out.join(name);
        std::fs::write(&path, content)
            .unwrap_or_else(|e| die(format!("cannot write {}: {e}", path.display())));
        println!("wrote {}", path.display());
    }
}

#[cfg(test)]
mod cli_tests {
    use super::*;

    #[test]
    fn generated_outputs_are_stable_and_differ_by_language() {
        let text = std::fs::read_to_string(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .join("uart.yaml"),
        )
        .unwrap();
        let spec = model::parse(&text).unwrap();
        model::validate(&spec).unwrap();
        let a = gen::gen_rust(&spec);
        let b = gen::gen_rust(&spec);
        assert_eq!(a, b);
        assert!(a.contains("REQUEST_SIZE: usize = 56"));
        assert!(a.contains("REPLY_SIZE: usize = 14"));
        assert!(a.contains("CRC_POLYNOMIAL: u16 = 0x1021"));
        assert!(a.contains("pub enum Phase"));
        assert!(gen::gen_c(&spec).contains("#define POTOK_REQUEST_SIZE 56u"));
        assert!(gen::gen_python(&spec).contains("REQUEST_SIZE = 56"));
    }
}
