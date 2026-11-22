use potok_train_tui::{data, palette, plots, ui};
use ratatui::{backend::TestBackend, style::Color, Terminal};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_dir(tag: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "potok_tui_{tag}_{}_{}",
        std::process::id(),
        n
    ));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).unwrap();
    path
}

struct Snapshot {
    text: String,
    colors: Vec<Vec<Color>>,
}

fn snapshot(app: &ui::App) -> Snapshot {
    let mut terminal = Terminal::new(TestBackend::new(120, 40)).unwrap();
    terminal.draw(|frame| ui::render(frame, app)).unwrap();
    let buffer = terminal.backend().buffer();
    let mut text = String::new();
    let mut colors = Vec::new();
    for y in 0..buffer.area.height {
        let row: Vec<Color> = (0..buffer.area.width)
            .map(|x| {
                let cell = buffer.cell((x, y)).unwrap();
                text.push_str(cell.symbol());
                cell.fg
            })
            .collect();
        colors.push(row);
        text.push('\n');
    }
    Snapshot { text, colors }
}

fn color_of(snap: &Snapshot, word: &str) -> Option<Color> {
    for (y, line) in snap.text.lines().enumerate() {
        if let Some(col) = line.find(word) {
            let cell_x = line[..col].chars().count();
            return snap.colors.get(y)?.get(cell_x).copied();
        }
    }
    None
}
fn expected(rgb: palette::Rgb) -> Color {
    Color::Rgb(rgb.0, rgb.1, rgb.2)
}

#[test]
fn demo_layout_renders_with_labels_and_hints() {
    let app = ui::App::new(data::demo_run(1));
    let snap = snapshot(&app);
    for label in [
        "DEMO",
        "potok / train monitor",
        "RMSE by epoch",
        "gates",
        "activity",
        "train",
        "held-out",
        "ratio",
        "ece",
        "coverage",
        "q quit",
        "tab next model",
        "refresh",
    ] {
        assert!(snap.text.contains(label), "missing '{label}' in:\n{}", snap.text);
    }
    assert_eq!(snap.text.lines().count(), 40);
    assert!(snap.text.lines().all(|line| line.chars().count() == 120));
}

#[test]
fn gate_colors_are_red_for_failure_and_green_for_pass() {
    let mut run = data::demo_run(1);
    run.models[0].ratio = 1.9;
    run.models[0].ece = 0.20;
    run.models[0].coverage = 0.10;
    let failed = snapshot(&ui::App::new(run));
    assert!(failed.text.contains("FAIL"));
    assert_eq!(color_of(&failed, "FAIL"), Some(expected(palette::FAIL)));
    assert!(!failed.text.contains("PASS"));

    let app = ui::App::new(data::demo_run(1));
    let passed = snapshot(&app);
    assert!(passed.text.contains("PASS"));
    assert_eq!(color_of(&passed, "PASS"), Some(expected(palette::PASS)));
    assert!(!passed.text.contains("FAIL"));
}

#[test]
fn missing_run_fails_loudly_and_renders_error() {
    let missing = Path::new("/nonexistent/potok-tui-run");
    let err = data::load_run(missing, 1).unwrap_err().0;
    assert!(err.contains("run directory not found"), "{err}");
    assert!(err.contains("--demo"), "{err}");

    let empty = temp_dir("empty");
    let err = data::load_run(&empty, 1).unwrap_err().0;
    assert!(err.contains("no metrics.json found"), "{err}");

    let snap = snapshot(&ui::App::failed(err.clone()));
    assert!(snap.text.contains("cannot display a run"));
    assert!(snap.text.contains("no metrics.json found"));
}

#[test]
fn sweep_csv_drives_the_series_and_source_label() {
    let dir = temp_dir("sweep");
    std::fs::create_dir_all(dir.join("loopwm")).unwrap();
    std::fs::create_dir_all(dir.join("logs")).unwrap();
    std::fs::write(
        dir.join("loopwm/metrics.json"),
        r#"{"model":"loopwm","rmse_train":0.19,"rmse_held_out":0.17,"ratio":0.9,"ece":0.01,"seed":1,"params":83075}"#,
    )
    .unwrap();
    std::fs::write(
        dir.join("logs/sweep_results.csv"),
        "model,epoch,seed,rmse_train,rmse_held_out,ratio,ece,raw_ece,coverage,n\n\
         loopwm,40,1,0.253,0.204,0.808,0.049,0.582,0.566,5804\n\
         loopwm,60,1,0.249,0.187,0.751,0.049,0.178,0.470,5804\n\
         loopwm,100,1,0.192,0.179,0.931,0.001,0.305,0.648,5804\n",
    )
    .unwrap();
    let run = data::load_run(&dir, 1).unwrap();
    assert_eq!(run.models.len(), 1);
    assert_eq!(run.models[0].series.len(), 3);
    assert_eq!(run.models[0].history_source, "sweep_results.csv");
    let rendered = plots::render_plots(&run, 0, None).unwrap();
    assert!(rendered.rmse_svg.contains("RMSE by epoch"));
    assert!(rendered.rmse_svg.contains("source:"));
}

#[test]
fn svg_reports_have_titles_series_labels_and_stable_bytes() {
    let run = data::demo_run(1);
    let plots = plots::render_plots(&run, 0, None).unwrap();

    assert!(plots.rmse_svg.starts_with("<svg"));
    assert!(plots.rmse_svg.contains("RMSE by epoch"));
    assert!(plots.rmse_svg.contains("train"));
    assert!(plots.rmse_svg.contains("held-out"));
    assert!(plots.ece_svg.contains("Calibration error by epoch"));
    assert!(plots.ece_svg.contains("gate 0.05"));
    assert!(plots.ece_svg.contains("raw ece"));
    assert!(plots.ab_svg.is_none());

    let a = temp_dir("svg_a");
    let b = temp_dir("svg_b");
    let written_a = plots::write_plots(&a, &plots).unwrap();
    let written_b = plots::write_plots(&b, &plots).unwrap();
    assert_eq!(written_a.len(), 2);
    for (left, right) in written_a.iter().zip(written_b.iter()) {
        assert!(std::fs::metadata(left).unwrap().len() > 1000);
        assert_eq!(
            std::fs::read(left).unwrap(),
            std::fs::read(right).unwrap(),
            "{} differs between runs",
            left.display()
        );
    }
}

#[test]
fn ab_report_renders_three_arms_and_fails_loudly_on_missing_seed() {
    let json = r#"{
      "seeds": {
        "1": {
          "loopwm": {
            "aggregate": {
              "baseline": {"lateral_rms_m": 1528.0, "apogee_m": 1735.0},
              "assisted": {"lateral_rms_m": 1534.0, "apogee_m": 1717.0}
            }
          },
          "oracle_gtheta": {
            "aggregate": {
              "baseline": {"lateral_rms_m": 1528.0, "apogee_m": 1735.0},
              "assisted": {"lateral_rms_m": 1035.0, "apogee_m": 3971.0}
            }
          }
        }
      }
    }"#;
    let report: plots::AbReport = serde_json::from_str(json).unwrap();
    let run = data::demo_run(1);
    let rendered = plots::render_plots(&run, 0, Some((&report, "1", "loopwm"))).unwrap();
    let svg = rendered.ab_svg.expect("A/B SVG should be present");
    for label in [
        "A/B comparison",
        "Lateral RMS (m)",
        "Apogee (m)",
        "baseline",
        "assisted",
        "oracle",
    ] {
        assert!(svg.contains(label), "missing '{label}' in A/B SVG");
    }

    let err = plots::ab_bars(&report, "99", "loopwm").unwrap_err();
    assert!(err.contains("no seed 99"), "{err}");
    let err = plots::ab_bars(&report, "1", "missing").unwrap_err();
    assert!(err.contains("no model 'missing'"), "{err}");
}
