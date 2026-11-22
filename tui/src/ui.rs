use crate::data::{ModelData, RunData, COVERAGE_TARGET, ECE_LIMIT, RATIO_LIMIT};
use crate::palette;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Axis, Block, BorderType, Borders, Chart, Dataset, GraphType, LegendPosition, Paragraph,
};
use ratatui::Frame;

fn c(rgb: palette::Rgb) -> Color {
    Color::Rgb(rgb.0, rgb.1, rgb.2)
}

pub struct App {
    pub run: Option<RunData>,
    pub error: Option<String>,
    pub model_idx: usize,
    pub refresh_ms: u64,
    pub tick: u64,
}

impl App {
    pub fn new(run: RunData) -> Self {
        Self {
            run: Some(run),
            error: None,
            model_idx: 0,
            refresh_ms: 1000,
            tick: 0,
        }
    }

    pub fn failed(message: String) -> Self {
        Self {
            run: None,
            error: Some(message),
            model_idx: 0,
            refresh_ms: 1000,
            tick: 0,
        }
    }

    pub fn current(&self) -> Option<&ModelData> {
        self.run.as_ref()?.models.get(self.model_idx)
    }

    pub fn next_model(&mut self) {
        if let Some(run) = &self.run {
            if !run.models.is_empty() {
                self.model_idx = (self.model_idx + 1) % run.models.len();
            }
        }
    }

    pub fn adjust_refresh(&mut self, delta: i64) {
        let next = self.refresh_ms as i64 + delta;
        self.refresh_ms = next.clamp(250, 5000) as u64;
    }
}

pub fn render(frame: &mut Frame, app: &App) {
    let area = frame.area();
    frame.render_widget(Block::default().style(Style::default().bg(c(palette::BASE)).fg(c(palette::TEXT))), area);

    if let Some(error) = &app.error {
        render_error(frame, area, error);
        return;
    }
    let run = match &app.run {
        Some(run) => run,
        None => {
            render_error(frame, area, "no run data; pass a run directory or --demo");
            return;
        }
    };
    if run.models.is_empty() {
        render_error(frame, area, "run has no model metrics");
        return;
    }

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(10),
            Constraint::Length(9),
        ])
        .split(area);

    render_status(frame, rows[0], app, run);

    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(71), Constraint::Percentage(29)])
        .split(rows[1]);
    render_chart(frame, cols[0], app, run);
    render_gates(frame, cols[1], app, run);
    render_activity(frame, rows[2], app, run);
}

fn panel(title: &str) -> Block<'_> {
    Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Plain)
        .border_style(Style::default().fg(c(palette::GRID)))
        .title(Span::styled(
            format!(" {title} "),
            Style::default().fg(c(palette::MUTED)).add_modifier(Modifier::BOLD),
        ))
        .style(Style::default().bg(c(palette::BASE)).fg(c(palette::TEXT)))
}

fn render_error(frame: &mut Frame, area: Rect, message: &str) {
    let block = panel("monitor error").border_style(Style::default().fg(c(palette::FAIL)));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let lines = vec![
        Line::from(Span::styled(
            "cannot display a run",
            Style::default().fg(c(palette::FAIL)).add_modifier(Modifier::BOLD),
        )),
        Line::from(message.to_string()),
        Line::from(Span::styled(
            "q quit   pass a run directory or --demo to continue",
            Style::default().fg(c(palette::MUTED)),
        )),
    ];
    frame.render_widget(Paragraph::new(lines), inner);
}

fn render_status(frame: &mut Frame, area: Rect, app: &App, run: &RunData) {
    let model = app.current().unwrap_or(&run.models[0]);
    let epoch = model.series.last().map(|p| p.epoch).unwrap_or(0);
    let eta = if run.demo {
        "00:34".to_string()
    } else {
        "n/a".to_string()
    };
    let mut spans = vec![
        Span::styled(
            if run.demo { "DEMO" } else { "RUN" },
            Style::default()
                .fg(if run.demo { c(palette::HELD) } else { c(palette::PASS) })
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("  ", Style::default()),
        Span::styled("model ", Style::default().fg(c(palette::MUTED))),
        Span::styled(model.name.clone(), Style::default().fg(c(palette::TRAIN))),
        Span::styled("   seed ", Style::default().fg(c(palette::MUTED))),
        Span::styled(model.seed.to_string(), Style::default()),
        Span::styled("   epoch ", Style::default().fg(c(palette::MUTED))),
        Span::styled(epoch.to_string(), Style::default()),
        Span::styled("   eta ", Style::default().fg(c(palette::MUTED))),
        Span::styled(eta, Style::default()),
        Span::styled("   refresh ", Style::default().fg(c(palette::MUTED))),
        Span::styled(format!("{}ms", app.refresh_ms), Style::default()),
    ];
    if run.models.len() > 1 {
        spans.push(Span::styled(
            format!("   view {}/{}", app.model_idx + 1, run.models.len()),
            Style::default().fg(c(palette::MUTED)),
        ));
    }
    let block = panel("potok / train monitor");
    let inner = block.inner(area);
    frame.render_widget(block, area);
    frame.render_widget(Paragraph::new(Line::from(spans)), inner);
}

fn float_bounds(values: impl Iterator<Item = f64>) -> (f64, f64, bool) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let mut any = false;
    for v in values {
        if v.is_finite() {
            any = true;
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    if !any {
        return (0.0, 1.0, false);
    }
    if (hi - lo).abs() < f64::EPSILON {
        (lo - 0.05, hi + 0.05, true)
    } else {
        let pad = (hi - lo) * 0.08;
        (lo - pad, hi + pad, true)
    }
}

fn render_chart(frame: &mut Frame, area: Rect, app: &App, run: &RunData) {
    let model = app.current().unwrap_or(&run.models[0]);
    let train: Vec<(f64, f64)> = model
        .series
        .iter()
        .map(|p| (p.epoch as f64, p.rmse_train))
        .collect();
    let held: Vec<(f64, f64)> = model
        .series
        .iter()
        .map(|p| (p.epoch as f64, p.rmse_held_out))
        .collect();
    let (ymin, ymax, has_data) = float_bounds(
        train
            .iter()
            .chain(held.iter())
            .map(|(_, y)| *y),
    );
    let xhi = model.series.last().map(|p| p.epoch as f64).unwrap_or(1.0);
    let xlo = model.series.first().map(|p| p.epoch as f64).unwrap_or(0.0);
    let (xmin, xmax) = if xhi > xlo { (xlo, xhi) } else { ((xlo - 1.0).max(0.0), xhi + 1.0) };

    let mut datasets = vec![
        Dataset::default()
            .name("train")
            .data(&train)
            .marker(ratatui::symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(c(palette::TRAIN))),
        Dataset::default()
            .name("held-out")
            .data(&held)
            .marker(ratatui::symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(c(palette::HELD))),
    ];
    // Current-value markers as a third scatter series.
    let markers: Vec<(f64, f64)> = match (train.last(), held.last()) {
        (Some(a), Some(b)) => vec![*a, *b],
        _ => vec![],
    };
    if has_data && !markers.is_empty() {
        datasets.push(
            Dataset::default()
                .name("current")
                .data(&markers)
                .marker(ratatui::symbols::Marker::Dot)
                .graph_type(GraphType::Scatter)
                .style(Style::default().fg(c(palette::TEXT))),
        );
    }

    let xlabels: Vec<String> = (0..=4)
        .map(|i| {
            let v = xmin + (xmax - xmin) * i as f64 / 4.0;
            format!("{}", v.round() as u64)
        })
        .collect();
    let ylabels: Vec<String> = (0..=4)
        .map(|i| format!("{:.3}", ymin + (ymax - ymin) * i as f64 / 4.0))
        .collect();

    let chart_title = format!("RMSE by epoch - {}", model.name);
    let chart = Chart::new(datasets)
        .block(panel(&chart_title))
        .x_axis(
            Axis::default()
                .title("epoch")
                .style(Style::default().fg(c(palette::GRID)))
                .bounds([xmin, xmax])
                .labels(xlabels),
        )
        .y_axis(
            Axis::default()
                .title("rmse")
                .style(Style::default().fg(c(palette::GRID)))
                .bounds([ymin, ymax])
                .labels(ylabels),
        )
        .legend_position(Some(LegendPosition::TopRight))
        .hidden_legend_constraints((Constraint::Min(0), Constraint::Min(0)))
        .style(Style::default().bg(c(palette::BASE)));
    frame.render_widget(chart, area);
}

fn gate_line(label: &str, value: f64, op: &str, limit: f64, ok: bool) -> Line<'static> {
    let (verdict, color) = if ok { ("PASS", palette::PASS) } else { ("FAIL", palette::FAIL) };
    Line::from(vec![
        Span::styled(format!("{label:<9}"), Style::default().fg(c(palette::MUTED))),
        Span::styled(format!("{value:.3} {op} {limit:.2}"), Style::default()),
        Span::styled("  ", Style::default()),
        Span::styled(
            verdict,
            Style::default().fg(c(color)).add_modifier(Modifier::BOLD),
        ),
    ])
}

fn render_gates(frame: &mut Frame, area: Rect, app: &App, run: &RunData) {
    let model = app.current().unwrap_or(&run.models[0]);
    let mut lines = vec![
        gate_line("ratio", model.ratio, "<=", RATIO_LIMIT, model.ratio_ok()),
        gate_line("ece", model.ece, "<=", ECE_LIMIT, model.ece_ok()),
        gate_line("coverage", model.coverage, ">=", COVERAGE_TARGET, model.coverage_ok()),
        Line::from(""),
        Line::from(vec![
            Span::styled("raw ece   ", Style::default().fg(c(palette::MUTED))),
            Span::styled(format!("{:.3}", model.raw_ece), Style::default()),
        ]),
        Line::from(vec![
            Span::styled("method    ", Style::default().fg(c(palette::MUTED))),
            Span::styled(model.method.clone(), Style::default()),
        ]),
        Line::from(vec![
            Span::styled("samples   ", Style::default().fg(c(palette::MUTED))),
            Span::styled(model.n.to_string(), Style::default()),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "models (tab)",
            Style::default().fg(c(palette::MUTED)).add_modifier(Modifier::BOLD),
        )),
    ];
    for (i, m) in run.models.iter().enumerate() {
        let active = i == app.model_idx;
        lines.push(Line::from(Span::styled(
            format!("{} {}", if active { ">" } else { " " }, m.name),
            Style::default().fg(if active { c(palette::TRAIN) } else { c(palette::MUTED) }),
        )));
    }
    let block = panel("gates");
    let inner = block.inner(area);
    frame.render_widget(block, area);
    frame.render_widget(Paragraph::new(lines), inner);
}

fn render_activity(frame: &mut Frame, area: Rect, app: &App, run: &RunData) {
    let mut lines: Vec<Line<'static>> = Vec::new();
    for note in &run.notes {
        lines.push(Line::from(Span::styled(
            note.clone(),
            Style::default().fg(c(palette::HELD)),
        )));
    }
    for log in run.logs.iter().rev().take(3) {
        lines.push(Line::from(Span::styled(
            log.clone(),
            Style::default().fg(c(palette::MUTED)),
        )));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled("q ", Style::default().fg(c(palette::TRAIN))),
        Span::styled("quit    ", Style::default().fg(c(palette::MUTED))),
        Span::styled("tab ", Style::default().fg(c(palette::TRAIN))),
        Span::styled("next model    ", Style::default().fg(c(palette::MUTED))),
        Span::styled("+/- ", Style::default().fg(c(palette::TRAIN))),
        Span::styled(
            format!("refresh {}ms", app.refresh_ms),
            Style::default().fg(c(palette::MUTED)),
        ),
    ]));
    let block = panel("activity");
    let inner = block.inner(area);
    frame.render_widget(block, area);
    frame.render_widget(Paragraph::new(lines), inner);
}
