//! Strict numeric CSV reader for measurement files.
//!
//! The format is deliberately small and fully specified by
//! `docs/MEASUREMENT_PROTOCOL.md`:
//!
//! * `# key: value` metadata lines (anywhere), each key at most once;
//! * one exact column-name row;
//! * one numeric value per column per data row.
//!
//! Quoted fields, escaped commas and thousands separators are not supported:
//! a wrong column count is a LOUD refusal, never a best-effort reinterpretation.
//! `NaN`/`inf` tokens parse as `f64` in Rust and are explicitly refused.

use std::collections::BTreeMap;

use crate::error::{MeasurementError, Result};

/// One parsed numeric data row with its physical line number for diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    pub line: usize,
    pub cells: Vec<f64>,
}

/// A parsed measurement CSV: metadata, exact header and numeric rows.
#[derive(Debug, Clone, PartialEq)]
pub struct Document {
    pub path: String,
    pub meta: BTreeMap<String, String>,
    pub columns: Vec<String>,
    pub rows: Vec<Row>,
}

impl Document {
    fn err(&self, detail: impl Into<String>) -> MeasurementError {
        MeasurementError::Csv {
            path: self.path.clone(),
            detail: detail.into(),
        }
    }

    /// Read a required metadata key. A missing or empty value is LOUD.
    pub fn meta_required(&self, key: &str) -> Result<&str> {
        match self.meta.get(key) {
            Some(v) if !v.is_empty() => Ok(v),
            Some(_) => Err(self.err(format!(
                "metadata key '{key}' is empty; supply a non-empty '# {key}: <value>' header"
            ))),
            None => Err(self.err(format!(
                "missing required metadata key '{key}'; add '# {key}: <value>' before the column row"
            ))),
        }
    }

    /// Read a required finite metadata number.
    pub fn meta_f64(&self, key: &str) -> Result<f64> {
        let raw = self.meta_required(key)?;
        let value = raw.parse::<f64>().map_err(|e| {
            self.err(format!(
                "metadata '{key}' value '{raw}' is not a number ({e}); supply a finite decimal value"
            ))
        })?;
        if !value.is_finite() {
            return Err(self.err(format!(
                "metadata '{key}' value '{raw}' is not finite; supply a finite decimal value"
            )));
        }
        Ok(value)
    }
}

/// Parse `text` into a strict numeric document whose header is exactly
/// `columns`. No row is dropped, repaired or defaulted.
pub fn parse(text: &str, path: &str, columns: &[&str]) -> Result<Document> {
    let mut doc = Document {
        path: path.to_string(),
        meta: BTreeMap::new(),
        columns: Vec::new(),
        rows: Vec::new(),
    };
    let text = text.trim_start_matches('\u{feff}');
    let mut header_seen = false;

    for (index, raw_line) in text.lines().enumerate() {
        let line = index + 1;
        let trimmed = raw_line.trim();
        if let Some(comment) = trimmed.strip_prefix('#') {
            let comment = comment.trim();
            // A '# line' without a colon is free-form commentary, not a
            // metadata key. Only 'key: value' lines are bound as metadata.
            let Some((key, value)) = comment.split_once(':') else {
                continue;
            };
            let key = key.trim();
            let value = value.trim();
            if key.is_empty() {
                return Err(doc.err(format!("line {line}: empty metadata key")));
            }
            if doc.meta.insert(key.to_string(), value.to_string()).is_some() {
                return Err(doc.err(format!(
                    "line {line}: duplicate metadata key '{key}'; keep exactly one"
                )));
            }
            continue;
        }
        if trimmed.is_empty() {
            if header_seen {
                return Err(doc.err(format!(
                    "line {line}: empty data row; remove the blank line or supply all {} columns",
                    columns.len()
                )));
            }
            continue;
        }
        if !header_seen {
            let found: Vec<String> = trimmed.split(',').map(|c| c.trim().to_string()).collect();
            let expected: Vec<String> = columns.iter().map(|c| (*c).to_string()).collect();
            if found != expected {
                return Err(doc.err(format!(
                    "line {line}: header {found:?} does not match expected {expected:?}; fix the column names and order"
                )));
            }
            doc.columns = found;
            header_seen = true;
            continue;
        }

        let cells: Vec<&str> = trimmed.split(',').map(str::trim).collect();
        if cells.len() != columns.len() {
            return Err(doc.err(format!(
                "line {line}: row has {} columns, expected {}; missing or corrupt row",
                cells.len(),
                columns.len()
            )));
        }
        let mut values = Vec::with_capacity(cells.len());
        for (col, token) in cells.iter().enumerate() {
            let value = token.parse::<f64>().map_err(|e| {
                doc.err(format!(
                    "line {line}, column '{}' (index {col}): token '{token}' is not a number ({e})",
                    columns[col]
                ))
            })?;
            if !value.is_finite() {
                return Err(doc.err(format!(
                    "line {line}, column '{}' (index {col}): token '{token}' is not finite; replace the corrupt value",
                    columns[col]
                )));
            }
            values.push(value);
        }
        doc.rows.push(Row { line, cells: values });
    }

    if !header_seen {
        return Err(doc.err(format!(
            "missing column row; expected '{}' with {} data columns",
            columns.join(","),
            columns.len()
        )));
    }
    if doc.rows.is_empty() {
        return Err(doc.err("measurement contains no data rows; supply at least one row"));
    }
    Ok(doc)
}
