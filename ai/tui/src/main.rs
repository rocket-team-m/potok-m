use potok_train_tui::{data, plots, ui};

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use std::path::{Path, PathBuf};
use std::time::Duration;

const HELP: &str = "\
potok-train-tui - training-run monitor and deterministic SVG reports

Usage:
  potok-train-tui [run-dir] [options]

Options:
  --demo               use deterministic synthetic metrics (labelled DEMO)
  --plot-only          render SVG reports into --plots-out and exit
  --plots-out DIR      SVG output directory (default: <run>/plots)
  --ab-report PATH     A/B JSON report (default: <run>/ab_report.json)
  --refresh MS         poll interval in ms (default 1000)
  --seed N             seed to select from per-epoch data (default 1)
  --model NAME         initial model name
  --help               show this help

Keys: q quit, tab next model, +/- refresh interval.
";

struct Cli {
    run: PathBuf,
    demo: bool,
    plot_only: bool,
    plots_out: Option<PathBuf>,
    ab_report: Option<PathBuf>,
    refresh_ms: u64,
    seed: u32,
    model: Option<String>,
}

fn parse_num<T: std::str::FromStr>(value: &str, flag: &str) -> Result<T, String> {
    value
        .parse::<T>()
        .map_err(|_| format!("invalid value '{value}' for {flag}"))
}

fn default_run() -> PathBuf {
    let local = PathBuf::from("ai/out/final_100");
    if local.is_dir() {
        return local;
    }
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../out/final_100")
}

fn parse_args() -> Result<Cli, String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print!("{HELP}");
        std::process::exit(0);
    }
    let mut cli = Cli {
        run: default_run(),
        demo: false,
        plot_only: false,
        plots_out: None,
        ab_report: None,
        refresh_ms: 1000,
        seed: 1,
        model: None,
    };
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        match arg {
            "--demo" => cli.demo = true,
            "--plot-only" => cli.plot_only = true,
            "--plots-out" | "--ab-report" | "--refresh" | "--seed" | "--model" => {
                i += 1;
                let value = args.get(i).ok_or_else(|| format!("{arg} needs a value"))?;
                match arg {
                    "--plots-out" => cli.plots_out = Some(PathBuf::from(value)),
                    "--ab-report" => cli.ab_report = Some(PathBuf::from(value)),
                    "--refresh" => cli.refresh_ms = parse_num(value, arg)?,
                    "--seed" => cli.seed = parse_num(value, arg)?,
                    "--model" => cli.model = Some(value.clone()),
                    _ => unreachable!(),
                }
            }
            _ if arg.starts_with('-') => {
                return Err(format!("unknown option '{arg}'\n{HELP}"));
            }
            _ => cli.run = PathBuf::from(arg),
        }
        i += 1;
    }
    if cli.refresh_ms == 0 {
        return Err("--refresh must be greater than zero".to_string());
    }
    Ok(cli)
}

fn resolve_model(run: &data::RunData, name: Option<&str>) -> Result<usize, String> {
    match name {
        None => Ok(0),
        Some(name) => run
            .models
            .iter()
            .position(|m| m.name == name)
            .ok_or_else(|| {
                format!(
                    "model '{name}' not in run; available: {:?}",
                    run.models.iter().map(|m| m.name.as_str()).collect::<Vec<_>>()
                )
            }),
    }
}

fn load(cli: &Cli) -> Result<data::RunData, String> {
    if cli.demo {
        Ok(data::demo_run(cli.seed))
    } else {
        data::load_run(&cli.run, cli.seed).map_err(|e| e.0)
    }
}

fn plot_mode(cli: &Cli) -> Result<(), String> {
    let run = load(cli)?;
    let idx = resolve_model(&run, cli.model.as_deref())?;
    let model_name = run.models[idx].name.clone();

    let ab_source = cli.ab_report.clone().or_else(|| {
        let auto = run.path.join("ab_report.json");
        auto.is_file().then_some(auto)
    });
    let ab = match ab_source {
        Some(path) => {
            let report = plots::load_ab(&path)?;
            Some((report, cli.seed.to_string(), model_name.clone()))
        }
        None => {
            eprintln!("note: no ab_report.json found; skipping ab_comparison.svg");
            None
        }
    };
    let ab_ref = ab.as_ref().map(|(r, s, m)| (r, s.as_str(), m.as_str()));
    let rendered = plots::render_plots(&run, idx, ab_ref)?;
    let out = cli.plots_out.clone().unwrap_or_else(|| {
        if run.demo {
            PathBuf::from("demo-plots")
        } else {
            run.path.join("plots")
        }
    });
    let paths = plots::write_plots(&out, &rendered)?;
    for path in paths {
        println!("{}", path.display());
    }
    Ok(())
}

fn live_mode(cli: &Cli) -> Result<(), String> {
    let run = load(cli)?;
    let mut app = ui::App::new(run);
    app.model_idx = resolve_model(app.run.as_ref().unwrap(), cli.model.as_deref())?;
    app.refresh_ms = cli.refresh_ms;

    let mut terminal = ratatui::try_init().map_err(|e| format!("cannot init terminal: {e}"))?;
    let result = event_loop(&mut terminal, cli, &mut app);
    if let Err(e) = ratatui::try_restore() {
        eprintln!("warning: cannot restore terminal: {e}");
    }
    result
}

fn event_loop(
    terminal: &mut ratatui::DefaultTerminal,
    cli: &Cli,
    app: &mut ui::App,
) -> Result<(), String> {
    loop {
        terminal
            .draw(|frame| ui::render(frame, app))
            .map_err(|e| format!("cannot draw: {e}"))?;
        if event::poll(Duration::from_millis(app.refresh_ms))
            .map_err(|e| format!("cannot poll events: {e}"))?
        {
            if let Event::Key(key) = event::read().map_err(|e| format!("cannot read event: {e}"))? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
                        KeyCode::Tab => app.next_model(),
                        KeyCode::Char('+') | KeyCode::Char('=') => app.adjust_refresh(250),
                        KeyCode::Char('-') | KeyCode::Char('_') => app.adjust_refresh(-250),
                        _ => {}
                    }
                }
            }
        } else {
            app.tick += 1;
            if !cli.demo {
                // Live-tail: re-read the run on each idle poll. Errors are shown in the TUI.
                match load(cli) {
                    Ok(fresh) => {
                        let selected = app.current().map(|m| m.name.clone());
                        app.run = Some(fresh);
                        if let Some(name) = selected {
                            if let Ok(idx) = resolve_model(app.run.as_ref().unwrap(), Some(&name)) {
                                app.model_idx = idx;
                            }
                        }
                        app.error = None;
                    }
                    Err(message) => app.error = Some(message),
                }
            }
        }
    }
}

fn main() {
    let cli = match parse_args() {
        Ok(cli) => cli,
        Err(message) => {
            eprintln!("error: {message}");
            std::process::exit(2);
        }
    };
    let result = if cli.plot_only { plot_mode(&cli) } else { live_mode(&cli) };
    if let Err(message) = result {
        eprintln!("error: {message}");
        std::process::exit(2);
    }
}
