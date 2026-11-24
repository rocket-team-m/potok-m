//! Provenance records for ingested measurements.
//!
//! A hash is only a measurement identity if it binds the exact bytes that were
//! parsed and the label the operator supplied. This module records both. The
//! label is carried verbatim: the only label shipped in this tree is
//! `SYNTHETIC`, and recording a label is not a claim that it is true.

use serde::{Deserialize, Serialize};

use crate::csv::Document;
use crate::error::{MeasurementError, Result};

/// SHA-256 of exact bytes, reusing the crate that owns the dataset provenance
/// hash so both instruments agree on canonicalisation.
pub fn hash_bytes(bytes: &[u8]) -> String {
    potok_sim::manifest::sha256_hex(bytes)
}

/// Require a lowercase 64-hex SHA-256, LOUD otherwise.
pub fn require_sha256(field: &str, value: &str) -> Result<()> {
    if value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    {
        Ok(())
    } else {
        Err(MeasurementError::Recipe {
            path: field.to_string(),
            detail: format!(
                "'{value}' is not a lowercase 64-char SHA-256; recompute it from the source bytes"
            ),
        })
    }
}

/// Identity of one ingested source file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceRecord {
    /// Schema kind, e.g. `thrust_stand`.
    pub kind: String,
    /// Path as resolved from the recipe root.
    pub path: String,
    /// Operator-supplied label verbatim; `SYNTHETIC` for bundled fixtures.
    pub source_label: String,
    /// SHA-256 of the exact bytes parsed.
    pub sha256: String,
    /// Number of numeric data rows consumed.
    pub rows: usize,
}

impl SourceRecord {
    /// Build a record from an already-parsed document. The hash is taken from
    /// the bytes the caller parsed, not from a later re-read of the path.
    pub fn new(doc: &Document, text: &str, kind: &str, source_label: &str) -> Self {
        Self {
            kind: kind.to_string(),
            path: doc.path.clone(),
            source_label: source_label.to_string(),
            sha256: hash_bytes(text.as_bytes()),
            rows: doc.rows.len(),
        }
    }

    pub fn is_synthetic(&self) -> bool {
        self.source_label == "SYNTHETIC"
    }
}

/// Provenance for one complete calibration ingest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    pub schema_version: u32,
    /// SHA-256 of the exact recipe bytes used to resolve sources.
    pub recipe_sha256: String,
    /// One record per source, in fixed order (never filesystem order).
    pub sources: Vec<SourceRecord>,
    /// SHA-256 of the canonical typed calibration values.
    pub calibration_sha256: String,
}

impl Provenance {
    pub fn all_synthetic(&self) -> bool {
        self.sources.iter().all(SourceRecord::is_synthetic)
    }
}
