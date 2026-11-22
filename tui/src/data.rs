use serde::Deserialize;
use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

pub const RATIO_LIMIT: f64 = 1.25;
pub const ECE_LIMIT: f64 = 0.05;
/// Display target for the coverage gauge. The project gates are ratio and ECE only.
pub const COVERAGE_TARGET: f64 = 0.50;

#[derive(Debug)]
pub struct LoadError(pub String);

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for LoadError {}

#[derive(Clone, Debug)]
pub struct EpochPoint {
    pub epoch: u32,
    pub rmse_train: f64,
    pub rmse_held_out: f64,
    pub ece: f64,
    pub raw_ece: f64,
    pub coverage: f64,
    pub elapsed_s: f64,
}

#[derive(Clone, Debug)]
pub struct ModelData {
    pub name: String,
    pub seed: u32,
    pub rmse_train: f64,
    pub rmse_held_out: f64,
    pub ratio: f64,
    pub ece: f64,
    pub raw_ece: f64,
    pub coverage: f64,
    pub n: u64,
    pub params: u64,
    pub method: String,
    pub series: Vec<EpochPoint>,
    pub history_source: String,
}

impl ModelData {
    pub fn ratio_ok(&self) -> bool {
        self.ratio <= RATIO_LIMIT
    }
    pub fn ece_ok(&self) -> bool {
        self.ece <= ECE_LIMIT
    }
    pub fn coverage_ok(&self) -> bool {
        self.coverage >= COVERAGE_TARGET
    }
}

#[derive(Clone, Debug)]
pub struct RunData {
    pub path: PathBuf,
    pub label: String,
    pub demo: bool,
    pub models: Vec<ModelData>,
    pub logs: Vec<String>,
    pub notes: Vec<String>,
    pub seed: u32,
}

#[derive(Deserialize)]
struct MetricsJson {
    #[serde(default)]
    model: Option<String>,
    rmse_train: f64,
    rmse_held_out: f64,
    #[serde(default)]
    ratio: Option<f64>,
    #[serde(default)]
    ece: Option<f64>,
    #[serde(default)]
    params: Option<u64>,
}

#[derive(Deserialize)]
struct CalibrationJson {
    #[serde(default)]
    ece: Option<f64>,
    #[serde(default)]
    raw_ece: Option<f64>,
    #[serde(default)]
    coverage: Option<f64>,
    #[serde(default)]
    n: Option<u64>,
    #[serde(default)]
    method: Option<String>,
}

#[derive(Deserialize)]
struct HistoryPoint {
    epoch: u32,
    rmse_train: f64,
    rmse_held_out: f64,
    #[serde(default)]
    ece: f64,
    #[serde(default)]
    raw_ece: f64,
    #[serde(default)]
    coverage: f64,
    #[serde(default)]
    elapsed_s: f64,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum HistoryFile {
    Array(Vec<HistoryPoint>),
    Object { history: Vec<HistoryPoint> },
}

impl HistoryFile {
    fn into_points(self) -> Vec<HistoryPoint> {
        match self {
            Self::Array(v) => v,
            Self::Object { history } => history,
        }
    }
}

struct CsvTable {
    headers: Vec<String>,
    lookup: BTreeMap<String, usize>,
    rows: Vec<Vec<String>>,
}

impl CsvTable {
    fn parse(path: &Path) -> Result<Self, LoadError> {
        let text = fs::read_to_string(path)
            .map_err(|e| LoadError(format!("cannot read {}: {e}", path.display())))?;
        let mut lines = text.lines().filter(|l| !l.trim().is_empty());
        let header = lines.next().ok_or_else(|| {
            LoadError(format!(
                "{} is empty; expected a CSV header",
                path.display()
            ))
        })?;
        let headers: Vec<String> = header
            .split(',')
            .map(|s| s.trim().trim_start_matches('\u{feff}').to_string())
            .collect();
        let lookup = headers
            .iter()
            .enumerate()
            .map(|(i, h)| (h.clone(), i))
            .collect();
        let rows = lines
            .map(|l| l.split(',').map(|s| s.trim().to_string()).collect())
            .collect();
        Ok(Self {
            headers,
            lookup,
            rows,
        })
    }

    fn has(&self, name: &str) -> bool {
        self.lookup.contains_key(name)
    }

    fn cell<'a>(&'a self, row: &'a [String], name: &str) -> Option<&'a str> {
        self.lookup
            .get(name)
            .and_then(|&i| row.get(i))
            .map(String::as_str)
    }

    fn num(&self, row: &[String], name: &str) -> Result<f64, LoadError> {
        let raw = self.cell(row, name).ok_or_else(|| {
            LoadError(format!("CSV column '{name}' missing in {:?}", self.headers))
        })?;
        raw.parse::<f64>()
            .map_err(|_| LoadError(format!("invalid number '{raw}' in CSV column '{name}'")))
    }

    fn uint(&self, row: &[String], name: &str) -> Result<u32, LoadError> {
        let v = self.num(row, name)?;
        if v < 0.0 || v.fract() != 0.0 || v > u32::MAX as f64 {
            return Err(LoadError(format!(
                "invalid unsigned integer in column '{name}': {v}"
            )));
        }
        Ok(v as u32)
    }
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, LoadError> {
    let text = fs::read_to_string(path)
        .map_err(|e| LoadError(format!("cannot read {}: {e}", path.display())))?;
    serde_json::from_str(&text)
        .map_err(|e| LoadError(format!("invalid JSON in {}: {e}", path.display())))
}

fn check_finite(v: f64, what: &str, path: &Path) -> Result<f64, LoadError> {
    if !v.is_finite() {
        return Err(LoadError(format!(
            "non-finite {what} in {}",
            path.display()
        )));
    }
    Ok(v)
}

/// Find the per-epoch series CSV in a run directory.
fn find_sweep(run: &Path) -> Option<PathBuf> {
    for rel in [
        "logs/sweep_results.csv",
        "sweep_results.csv",
        "logs/results.csv",
        "results.csv",
    ] {
        let p = run.join(rel);
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

fn history_file_for(run: &Path, model: &str) -> Option<PathBuf> {
    for rel in [
        format!("{model}/history.json"),
        format!("{model}.history.json"),
        "history.json".to_string(),
    ] {
        let p = run.join(rel);
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

fn read_history(path: &Path) -> Result<Vec<EpochPoint>, LoadError> {
    let file: HistoryFile = read_json(path)?;
    let mut pts: Vec<EpochPoint> = file
        .into_points()
        .into_iter()
        .map(|p| EpochPoint {
            epoch: p.epoch,
            rmse_train: p.rmse_train,
            rmse_held_out: p.rmse_held_out,
            ece: p.ece,
            raw_ece: p.raw_ece,
            coverage: p.coverage,
            elapsed_s: p.elapsed_s,
        })
        .collect();
    pts.sort_by_key(|p| p.epoch);
    Ok(pts)
}

fn sweep_series(
    table: &CsvTable,
    model: &str,
    seed: u32,
    path: &Path,
) -> Result<Vec<EpochPoint>, LoadError> {
    for name in ["model", "epoch", "rmse_train", "rmse_held_out"] {
        if !table.has(name) {
            return Err(LoadError(format!(
                "{} is missing required column '{name}'",
                path.display()
            )));
        }
    }
    let mut points = Vec::new();
    for row in &table.rows {
        if table.cell(row, "model") != Some(model) {
            continue;
        }
        let row_seed = if table.has("seed") {
            table.uint(row, "seed")?
        } else {
            1
        };
        if row_seed != seed {
            continue;
        }
        let epoch = table.uint(row, "epoch")?;
        let get = |name: &str, default: f64| -> Result<f64, LoadError> {
            if table.has(name) {
                table.num(row, name)
            } else {
                Ok(default)
            }
        };
        points.push(EpochPoint {
            epoch,
            rmse_train: check_finite(table.num(row, "rmse_train")?, "rmse_train", path)?,
            rmse_held_out: check_finite(table.num(row, "rmse_held_out")?, "rmse_held_out", path)?,
            ece: get("ece", 0.0)?,
            raw_ece: get("raw_ece", 0.0)?,
            coverage: get("coverage", 0.0)?,
            elapsed_s: get("elapsed_s", 0.0)?,
        });
    }
    points.sort_by_key(|p| p.epoch);
    Ok(points)
}

fn collect_logs(run: &Path, model_names: &[String]) -> Vec<String> {
    let mut lines = Vec::new();
    let mut push_tail = |path: &Path, n: usize| {
        if let Ok(text) = fs::read_to_string(path) {
            for line in text.lines().rev().take(n) {
                let s = line.trim_end();
                if !s.is_empty() && s != "EXIT=0" {
                    lines.push(s.to_string());
                }
            }
        }
    };
    for name in model_names {
        push_tail(&run.join(name).join("run.log"), 2);
    }
    if let Ok(rd) = fs::read_dir(run.join("logs")) {
        let mut paths: Vec<PathBuf> = rd.filter_map(Result::ok).map(|e| e.path()).collect();
        paths.sort();
        for p in paths {
            if p.extension().and_then(|s| s.to_str()) == Some("log") {
                push_tail(&p, 2);
            }
        }
    }
    push_tail(&run.join("REPORT.md"), 4);
    lines.truncate(12);
    lines
}

/// Load a run directory. Missing run, missing metrics, and malformed data are loud errors.
pub fn load_run(run: &Path, seed: u32) -> Result<RunData, LoadError> {
    if !run.is_dir() {
        return Err(LoadError(format!(
            "run directory not found: {}. Pass --demo for synthetic data.",
            run.display()
        )));
    }

    let mut files: Vec<(String, PathBuf)> = Vec::new();
    if let Ok(rd) = fs::read_dir(run) {
        for entry in rd.filter_map(Result::ok) {
            let p = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();
            if p.is_dir() {
                let m = p.join("metrics.json");
                if m.is_file() {
                    files.push((name, m));
                }
            } else if let Some(prefix) = name.strip_suffix(".metrics.json") {
                files.push((prefix.to_string(), p));
            }
        }
    }
    let root_metrics = run.join("metrics.json");
    if root_metrics.is_file() && !files.iter().any(|(_, p)| *p == root_metrics) {
        files.push((
            run.file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
            root_metrics,
        ));
    }
    files.sort_by(|a, b| a.0.cmp(&b.0));
    files.dedup_by(|a, b| a.1 == b.1);

    if files.is_empty() {
        return Err(LoadError(format!(
            "no metrics.json found under {}. Point at a run directory with <model>/metrics.json or <model>.metrics.json, or pass --demo.",
            run.display()
        )));
    }

    let sweep_path = find_sweep(run);
    let sweep = match &sweep_path {
        Some(p) => Some(CsvTable::parse(p)?),
        None => None,
    };

    let mut models = Vec::new();
    let mut notes = Vec::new();
    for (dir_name, path) in &files {
        let m: MetricsJson = read_json(path)?;
        let name = m.model.clone().unwrap_or_else(|| dir_name.clone());
        let rmse_train = check_finite(m.rmse_train, "rmse_train", path)?;
        let rmse_held_out = check_finite(m.rmse_held_out, "rmse_held_out", path)?;
        let ratio = m.ratio.unwrap_or_else(|| {
            if rmse_train != 0.0 {
                rmse_held_out / rmse_train
            } else {
                f64::NAN
            }
        });
        let ratio = check_finite(ratio, "ratio", path)?;
        let mut ece = m.ece.unwrap_or(0.0);
        let mut raw_ece = 0.0;
        let mut coverage = 0.0;
        let mut n = 0;
        let mut method = String::from("-");
        let calib_path = path.with_file_name("calibration.json");
        if calib_path.is_file() {
            let c: CalibrationJson = read_json(&calib_path)?;
            if let Some(v) = c.ece {
                ece = v;
            }
            raw_ece = c.raw_ece.unwrap_or(0.0);
            coverage = c.coverage.unwrap_or(0.0);
            n = c.n.unwrap_or(0);
            method = c.method.unwrap_or_else(|| "-".to_string());
        }

        let (series, source) = if let Some(table) = &sweep {
            let pts = sweep_series(table, &name, seed, sweep_path.as_ref().unwrap())?;
            if !pts.is_empty() {
                (pts, "sweep_results.csv".to_string())
            } else {
                fallback_series(history_file_for(run, &name), &name, path, &mut notes)?
            }
        } else {
            fallback_series(history_file_for(run, &name), &name, path, &mut notes)?
        };

        if let Some(last) = series.last() {
            // The final CSV row is the authoritative snapshot when a series exists.
            let mut model_data = ModelData {
                name,
                seed,
                rmse_train: last.rmse_train,
                rmse_held_out: last.rmse_held_out,
                ratio: if last.rmse_train != 0.0 {
                    last.rmse_held_out / last.rmse_train
                } else {
                    ratio
                },
                ece: if last.ece > 0.0 { last.ece } else { ece },
                raw_ece: if last.raw_ece > 0.0 {
                    last.raw_ece
                } else {
                    raw_ece
                },
                coverage: if last.coverage > 0.0 {
                    last.coverage
                } else {
                    coverage
                },
                n,
                params: m.params.unwrap_or(0),
                method,
                series,
                history_source: source,
            };
            if model_data.ece.is_nan() {
                model_data.ece = ece;
            }
            models.push(model_data);
        }
    }

    if models.is_empty() {
        return Err(LoadError(format!(
            "no model metrics for seed {seed} under {}",
            run.display()
        )));
    }

    let model_names: Vec<String> = models.iter().map(|m| m.name.clone()).collect();
    let logs = collect_logs(run, &model_names);
    let label = run
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| run.display().to_string());

    Ok(RunData {
        path: run.to_path_buf(),
        label,
        demo: false,
        models,
        logs,
        notes,
        seed,
    })
}

fn fallback_series(
    history: Option<PathBuf>,
    _name: &str,
    metrics_path: &Path,
    notes: &mut Vec<String>,
) -> Result<(Vec<EpochPoint>, String), LoadError> {
    if let Some(p) = history {
        let pts = read_history(&p)?;
        if !pts.is_empty() {
            return Ok((pts, "history.json".to_string()));
        }
    }
    let m: MetricsJson = read_json(metrics_path)?;
    notes.push(format!(
        "{}: no epoch history; plotting the final checkpoint only",
        m.model.as_deref().unwrap_or("model")
    ));
    Ok((
        vec![EpochPoint {
            epoch: 1,
            rmse_train: m.rmse_train,
            rmse_held_out: m.rmse_held_out,
            ece: m.ece.unwrap_or(0.0),
            raw_ece: 0.0,
            coverage: 0.0,
            elapsed_s: 0.0,
        }],
        "checkpoint".to_string(),
    ))
}

/// Deterministic synthetic run. No RNG, no time, no environment input.
pub fn demo_run(seed: u32) -> RunData {
    let names = ["loopwm", "loopwm-lite", "lrs"];
    let models = names
        .iter()
        .enumerate()
        .map(|(idx, name)| demo_model(name, idx as f64, seed))
        .collect();
    RunData {
        path: PathBuf::from("demo"),
        label: "DEMO".to_string(),
        demo: true,
        models,
        logs: vec![
            "demo: 100 epochs, seed 1, no network required".to_string(),
            "demo: series generated from a deterministic formula".to_string(),
            "demo: gates use the same thresholds as a real run".to_string(),
        ],
        notes: vec!["DEMO: synthetic metrics; nothing was read from disk".to_string()],
        seed,
    }
}

fn demo_model(name: &str, k: f64, seed: u32) -> ModelData {
    let npts = 60;
    let series: Vec<EpochPoint> = (0..npts)
        .map(|i| {
            let t = i as f64 / (npts - 1) as f64;
            let phase = k * 0.8 + seed as f64 * 0.01;
            EpochPoint {
                epoch: 1 + i * 2,
                rmse_train: 0.44 - 0.25 * (1.0 - (-3.4 * t).exp())
                    + 0.006 * (i as f64 * 0.7 + phase).sin(),
                rmse_held_out: 0.42 - 0.22 * (1.0 - (-2.6 * t).exp())
                    + 0.008 * (i as f64 * 0.5 + phase * 2.0).sin(),
                ece: 0.28 * (-3.2 * t).exp() + 0.008 + 0.002 * (i as f64 * 0.9).sin(),
                raw_ece: 0.36 - 0.09 * t,
                coverage: 0.22 + 0.48 * (1.0 - (-2.4 * t).exp()),
                elapsed_s: i as f64 * 6.4,
            }
        })
        .collect();
    let last = series.last().unwrap();
    let ratio = last.rmse_held_out / last.rmse_train;
    ModelData {
        name: name.to_string(),
        seed,
        rmse_train: last.rmse_train,
        rmse_held_out: last.rmse_held_out,
        ratio,
        ece: last.ece,
        raw_ece: last.raw_ece,
        coverage: last.coverage,
        n: 5804,
        params: match name {
            "loopwm" => 83_075,
            "loopwm-lite" => 34_819,
            _ => 22_439,
        },
        method: "isotonic".to_string(),
        series,
        history_source: "demo".to_string(),
    }
}
