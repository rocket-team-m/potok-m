use crate::data::{RunData, ECE_LIMIT};
use crate::palette::{self, Rgb};
use plotters::coord::Shift;
use plotters::prelude::*;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
pub struct AbReport {
    pub seeds: BTreeMap<String, BTreeMap<String, AbModel>>,
    #[serde(default)]
    pub measurement: String,
}

#[derive(Deserialize)]
pub struct AbModel {
    pub aggregate: AbAggregate,
}

#[derive(Deserialize)]
pub struct AbAggregate {
    pub baseline: AbArm,
    pub assisted: AbArm,
}

#[derive(Deserialize)]
pub struct AbArm {
    pub lateral_rms_m: f64,
    pub apogee_m: f64,
}

#[derive(Clone, Debug)]
pub struct AbBar {
    pub label: &'static str,
    pub lateral: f64,
    pub apogee: f64,
    pub color: Rgb,
}

pub fn load_ab(path: &Path) -> Result<AbReport, String> {
    let text = fs::read_to_string(path)
        .map_err(|e| format!("cannot read A/B report {}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("invalid A/B report {}: {e}", path.display()))
}

/// baseline / assisted / oracle for one seed and one student model.
pub fn ab_bars(report: &AbReport, seed: &str, model: &str) -> Result<Vec<AbBar>, String> {
    let arms = report.seeds.get(seed).ok_or_else(|| {
        format!(
            "A/B report has no seed {seed}; available: {:?}",
            report.seeds.keys().collect::<Vec<_>>()
        )
    })?;
    let student = arms.get(model).ok_or_else(|| {
        format!(
            "A/B report seed {seed} has no model '{model}'; available: {:?}",
            arms.keys().collect::<Vec<_>>()
        )
    })?;
    let oracle = arms
        .get("oracle_gtheta")
        .ok_or_else(|| format!("A/B report seed {seed} has no 'oracle_gtheta' arm"))?;
    let check = |v: f64, what: &str| {
        if v.is_finite() {
            Ok(v)
        } else {
            Err(format!("A/B report {seed}/{model} has non-finite {what}"))
        }
    };
    Ok(vec![
        AbBar {
            label: "baseline",
            lateral: check(
                student.aggregate.baseline.lateral_rms_m,
                "baseline lateral_rms_m",
            )?,
            apogee: check(student.aggregate.baseline.apogee_m, "baseline apogee_m")?,
            color: palette::MUTED,
        },
        AbBar {
            label: "assisted",
            lateral: check(
                student.aggregate.assisted.lateral_rms_m,
                "assisted lateral_rms_m",
            )?,
            apogee: check(student.aggregate.assisted.apogee_m, "assisted apogee_m")?,
            color: palette::TRAIN,
        },
        AbBar {
            label: "oracle",
            lateral: check(
                oracle.aggregate.assisted.lateral_rms_m,
                "oracle lateral_rms_m",
            )?,
            apogee: check(oracle.aggregate.assisted.apogee_m, "oracle apogee_m")?,
            color: palette::HELD,
        },
    ])
}

pub struct Plots {
    pub rmse_svg: String,
    pub ece_svg: String,
    pub ab_svg: Option<String>,
}

fn rgb(c: Rgb) -> RGBColor {
    RGBColor(c.0, c.1, c.2)
}

fn shape(c: Rgb, width: u32) -> ShapeStyle {
    ShapeStyle {
        color: rgb(c).into(),
        filled: false,
        stroke_width: width,
    }
}

fn solid(c: Rgb) -> ShapeStyle {
    ShapeStyle {
        color: rgb(c).into(),
        filled: true,
        stroke_width: 1,
    }
}

fn text(size: f64, c: Rgb) -> TextStyle<'static> {
    FontDesc::new(FontFamily::Monospace, size, FontStyle::Normal).color(&rgb(c))
}

fn range(lo: f64, hi: f64, pad: f64) -> (f64, f64) {
    if !lo.is_finite() || !hi.is_finite() || (hi - lo).abs() < f64::EPSILON {
        (lo - pad, lo + pad)
    } else {
        (lo - pad * (hi - lo), hi + pad * (hi - lo))
    }
}

fn footer(
    root: &DrawingArea<SVGBackend<'_>, Shift>,
    height: u32,
    lines: &[String],
) -> Result<(), String> {
    let mut y = height as i32 - 14;
    for line in lines.iter().rev() {
        root.draw(&Text::new(
            line.clone(),
            (20, y),
            text(11.0, palette::MUTED),
        ))
        .map_err(|e| format!("cannot draw footer: {e}"))?;
        y -= 16;
    }
    Ok(())
}

fn curve_svg(
    title: &str,
    subtitle: &str,
    footer_lines: &[String],
    epochs: &[u32],
    series_a: &[f64],
    series_b: &[f64],
    label_a: &str,
    label_b: &str,
    color_a: Rgb,
    color_b: Rgb,
    threshold: Option<(f64, &str)>,
    y_fmt: fn(f64) -> String,
) -> Result<String, String> {
    let (w, h) = (1040u32, 620u32);
    let mut buf = String::new();
    {
        let root = SVGBackend::with_string(&mut buf, (w, h)).into_drawing_area();
        root.fill(&rgb(palette::BASE))
            .map_err(|e| format!("cannot fill report background: {e}"))?;
        root.draw(&Text::new(
            title.to_string(),
            (20, 12),
            text(17.0, palette::TEXT),
        ))
        .map_err(|e| format!("cannot draw report title: {e}"))?;
        root.draw(&Text::new(
            subtitle.to_string(),
            (20, 36),
            text(11.0, palette::MUTED),
        ))
        .map_err(|e| format!("cannot draw report subtitle: {e}"))?;

        let xs: Vec<f64> = epochs.iter().map(|e| *e as f64).collect();
        let x0 = xs.iter().copied().fold(f64::INFINITY, f64::min);
        let x1 = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let (xmin, xmax) = range(x0, x1, 0.04);
        let all: Vec<f64> = series_a.iter().chain(series_b.iter()).copied().collect();
        let mut y0 = all.iter().copied().fold(f64::INFINITY, f64::min);
        let mut y1 = all.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if let Some((t, _)) = threshold {
            y0 = y0.min(t);
            y1 = y1.max(t);
        }
        let (ymin, mut ymax) = range(y0, y1, 0.08);
        if ymax <= ymin {
            ymax = ymin + 1.0;
        }

        let mut chart = ChartBuilder::on(&root)
            .margin_left(12)
            .margin_right(20)
            .margin_top(62)
            .margin_bottom(52)
            .x_label_area_size(40)
            .y_label_area_size(66)
            .build_cartesian_2d(xmin..xmax, ymin..ymax)
            .map_err(|e| format!("cannot build chart: {e}"))?;

        chart
            .configure_mesh()
            .x_labels(9)
            .y_labels(7)
            .x_desc("epoch")
            .label_style(text(11.0, palette::MUTED))
            .axis_desc_style(text(11.0, palette::MUTED))
            .axis_style(shape(palette::GRID, 1))
            .light_line_style(shape(palette::GRID, 1))
            .bold_line_style(shape(palette::GRID, 1))
            .y_label_formatter(&|v: &f64| y_fmt(*v))
            .draw()
            .map_err(|e| format!("cannot draw chart grid: {e}"))?;

        let style_a = shape(color_a, 2);
        let style_b = shape(color_b, 2);
        let pts_a: Vec<(f64, f64)> = xs.iter().copied().zip(series_a.iter().copied()).collect();
        let pts_b: Vec<(f64, f64)> = xs.iter().copied().zip(series_b.iter().copied()).collect();

        chart
            .draw_series(LineSeries::new(pts_a.iter().copied(), style_a))
            .map_err(|e| format!("cannot draw {label_a}: {e}"))?
            .label(label_a)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 18, y)], style_a));

        chart
            .draw_series(LineSeries::new(pts_b.iter().copied(), style_b))
            .map_err(|e| format!("cannot draw {label_b}: {e}"))?
            .label(label_b)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 18, y)], style_b));

        if let Some((t, label)) = threshold {
            let ts = shape(palette::FAIL, 1);
            chart
                .draw_series(LineSeries::new(vec![(xmin, t), (xmax, t)], ts))
                .map_err(|e| format!("cannot draw gate line: {e}"))?
                .label(label)
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 18, y)], ts));
        }

        // Current-value markers: one filled dot at the last observation of each series.
        if let (Some((x, y)), Some((xb, yb))) = (pts_a.last(), pts_b.last()) {
            chart
                .draw_series(std::iter::once(Circle::new((*x, *y), 4, solid(color_a))))
                .map_err(|e| format!("cannot draw current {label_a} marker: {e}"))?;
            chart
                .draw_series(std::iter::once(Circle::new((*xb, *yb), 4, solid(color_b))))
                .map_err(|e| format!("cannot draw current {label_b} marker: {e}"))?;
        }

        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::UpperRight)
            .background_style(solid(palette::PANEL))
            .border_style(shape(palette::GRID, 1))
            .label_font(text(12.0, palette::TEXT))
            .draw()
            .map_err(|e| format!("cannot draw legend: {e}"))?;

        footer(&root, h, footer_lines)?;
    }
    Ok(buf)
}

pub fn render_plots(
    run: &RunData,
    model_idx: usize,
    ab: Option<(&AbReport, &str, &str)>,
) -> Result<Plots, String> {
    let model = run.models.get(model_idx).ok_or_else(|| {
        format!(
            "no model at index {model_idx}; run has {} models",
            run.models.len()
        )
    })?;
    if model.series.is_empty() {
        return Err(format!("model '{}' has no epochs to plot", model.name));
    }

    let epochs: Vec<u32> = model.series.iter().map(|p| p.epoch).collect();
    let train: Vec<f64> = model.series.iter().map(|p| p.rmse_train).collect();
    let held: Vec<f64> = model.series.iter().map(|p| p.rmse_held_out).collect();
    let ece: Vec<f64> = model.series.iter().map(|p| p.ece).collect();
    let raw_ece: Vec<f64> = model.series.iter().map(|p| p.raw_ece).collect();

    let source = format!(
        "source: {} - {} - seed {} - {} epochs",
        run.label,
        model.name,
        model.seed,
        epochs.len()
    );
    let provenance = if run.demo {
        "DEMO: synthetic series, no run data was read".to_string()
    } else {
        format!("history: {}", model.history_source)
    };

    let rmse_svg = curve_svg(
        "RMSE by epoch",
        &format!("{} - train vs held-out", model.name),
        &[source.clone(), provenance.clone()],
        &epochs,
        &train,
        &held,
        "train",
        "held-out",
        palette::TRAIN,
        palette::HELD,
        None,
        |v| format!("{v:.3}"),
    )?;

    let ece_svg = curve_svg(
        "Calibration error by epoch",
        &format!("{} - calibrated vs raw ECE", model.name),
        &[source, provenance],
        &epochs,
        &ece,
        &raw_ece,
        "ece",
        "raw ece",
        palette::TRAIN,
        palette::HELD,
        Some((ECE_LIMIT, "gate 0.05")),
        |v| format!("{v:.3}"),
    )?;

    let ab_svg = match ab {
        Some((report, seed, ab_model)) => {
            let bars = ab_bars(report, seed, ab_model)?;
            Some(ab_svg(&bars, seed, ab_model)?)
        }
        None => None,
    };

    Ok(Plots {
        rmse_svg,
        ece_svg,
        ab_svg,
    })
}

fn ab_svg(bars: &[AbBar], seed: &str, model: &str) -> Result<String, String> {
    let (w, h) = (1040u32, 620u32);
    let mut buf = String::new();
    {
        let root = SVGBackend::with_string(&mut buf, (w, h)).into_drawing_area();
        root.fill(&rgb(palette::BASE))
            .map_err(|e| format!("cannot fill A/B background: {e}"))?;
        root.draw(&Text::new(
            "A/B comparison".to_string(),
            (20, 12),
            text(17.0, palette::TEXT),
        ))
        .map_err(|e| format!("cannot draw A/B title: {e}"))?;
        root.draw(&Text::new(
            format!("seed {seed} - {model} - flown-truth trajectory sidecar"),
            (20, 36),
            text(11.0, palette::MUTED),
        ))
        .map_err(|e| format!("cannot draw A/B subtitle: {e}"))?;

        let (left, right) = root.split_horizontally(w as i32 / 2);
        ab_panel(
            &left,
            "Lateral RMS (m)",
            bars.iter().map(|b| b.lateral).collect(),
            bars,
            |v| format!("{v:.0}"),
        )?;
        ab_panel(
            &right,
            "Apogee (m)",
            bars.iter().map(|b| b.apogee).collect(),
            bars,
            |v| format!("{v:.0}"),
        )?;

        footer(
            &root,
            h,
            &["arms: baseline / assisted / oracle".to_string()],
        )?;
    }
    Ok(buf)
}

fn ab_panel(
    area: &DrawingArea<SVGBackend<'_>, Shift>,
    title: &str,
    values: Vec<f64>,
    bars: &[AbBar],
    y_fmt: fn(f64) -> String,
) -> Result<(), String> {
    let ymax = values.iter().copied().fold(0.0f64, f64::max) * 1.12;
    let ymax = if ymax.is_finite() && ymax > 0.0 {
        ymax
    } else {
        1.0
    };
    let mut chart = ChartBuilder::on(area)
        .caption(title, text(13.0, palette::TEXT))
        .margin_left(8)
        .margin_right(12)
        .margin_top(62)
        .margin_bottom(52)
        .x_label_area_size(46)
        .y_label_area_size(56)
        .build_cartesian_2d(-0.5f64..bars.len() as f64 - 0.5, 0.0f64..ymax)
        .map_err(|e| format!("cannot build A/B panel: {e}"))?;
    chart
        .configure_mesh()
        .x_labels(bars.len())
        .y_labels(6)
        .x_desc("arm")
        .label_style(text(10.0, palette::MUTED))
        .axis_desc_style(text(10.0, palette::MUTED))
        .axis_style(shape(palette::GRID, 1))
        .light_line_style(shape(palette::GRID, 1))
        .bold_line_style(shape(palette::GRID, 1))
        .x_label_formatter(&|x| {
            let i = (*x).round().max(0.0) as usize;
            bars.get(i).map(|b| b.label.to_string()).unwrap_or_default()
        })
        .y_label_formatter(&|v: &f64| y_fmt(*v))
        .draw()
        .map_err(|e| format!("cannot draw A/B grid: {e}"))?;

    for (i, (bar, value)) in bars.iter().zip(values.iter()).enumerate() {
        let style = solid(bar.color);
        let x0 = i as f64 + 0.22;
        let x1 = i as f64 + 0.78;
        chart
            .draw_series(std::iter::once(Rectangle::new(
                [(x0, 0.0), (x1, *value)],
                style,
            )))
            .map_err(|e| format!("cannot draw A/B bar {}: {e}", bar.label))?
            .label(bar.label)
            .legend(move |(x, y)| Rectangle::new([(x, y - 5), (x + 12, y + 5)], style));
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .background_style(solid(palette::PANEL))
        .border_style(shape(palette::GRID, 1))
        .label_font(text(11.0, palette::TEXT))
        .draw()
        .map_err(|e| format!("cannot draw A/B legend: {e}"))?;
    Ok(())
}

pub fn write_plots(out: &Path, plots: &Plots) -> Result<Vec<PathBuf>, String> {
    fs::create_dir_all(out).map_err(|e| format!("cannot create {}: {e}", out.display()))?;
    let mut written = vec![
        (out.join("rmse_curve.svg"), &plots.rmse_svg),
        (out.join("ece_curve.svg"), &plots.ece_svg),
    ];
    if let Some(svg) = &plots.ab_svg {
        written.push((out.join("ab_comparison.svg"), svg));
    }
    let mut paths = Vec::new();
    for (path, svg) in written {
        fs::write(&path, svg).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
        paths.push(path);
    }
    Ok(paths)
}
