//! Offline command source: bounded per-frame gimbal/fin corrections.
//!
//! A control file is whitespace-separated TSV with one row per 50 Hz tick:
//!
//! ```text
//! # frame_index g_roll_deg g_pitch_deg
//! frame_index  g_roll  g_pitch
//! 0            0.0     0.0
//! 1            2.5    -1.0
//! ...
//! ```
//!
//! `#` comments, blank lines and one optional `frame_index ...` header are
//! ignored. Indices must be exactly `0..frames_per_episode` in order: a
//! missing row, an extra row, a gap, a duplicate or an out-of-range command is
//! a LOUD `SimError::Control`, never a clamp or a skip.

use std::{fs, path::Path};

use crate::{
    error::{Result, SimError},
    sensor::Frame,
};

/// Symmetric command limit in degrees. Covers the reference optimizer's
/// `max_correction = 0.12 rad` (~6.88 deg) with margin; anything larger is
/// refused loudly instead of being silently clipped.
pub const MAX_CONTROL_DEG: f64 = 10.0;
pub const CONTROL_FIELDS: usize = 3;

/// Check one command pair. Both axes are finite and within
/// `[-MAX_CONTROL_DEG, MAX_CONTROL_DEG]`.
pub fn validate_command(roll_deg: f64, pitch_deg: f64) -> Result<(f64, f64)> {
    for (name, v) in [("g_roll", roll_deg), ("g_pitch", pitch_deg)] {
        if !v.is_finite() || v.abs() > MAX_CONTROL_DEG {
            return Err(SimError::Control(format!(
                "{name}={v} outside [-{MAX_CONTROL_DEG}, {MAX_CONTROL_DEG}] degrees"
            )));
        }
    }
    Ok((roll_deg, pitch_deg))
}

#[derive(Debug, Clone, PartialEq)]
pub struct ControlFile {
    commands: Vec<(f64, f64)>,
}

impl ControlFile {
    pub fn from_path(path: &Path, expected_frames: usize) -> Result<Self> {
        let text = fs::read_to_string(path)
            .map_err(|e| SimError::Control(format!("{}: {e}", path.display())))?;
        Self::from_str(&text, expected_frames)
    }

    pub fn from_str(text: &str, expected_frames: usize) -> Result<Self> {
        // Frame.seq is a u16; a longer episode would wrap the command index.
        if expected_frames > u16::MAX as usize + 1 {
            return Err(SimError::Control(format!(
                "frames_per_episode {expected_frames} exceeds the u16 frame sequence range"
            )));
        }
        let mut commands = Vec::with_capacity(expected_frames);
        for (line_no, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let f: Vec<&str> = line.split_whitespace().collect();
            if f.len() != CONTROL_FIELDS {
                return Err(SimError::Control(format!(
                    "line {}: expected {CONTROL_FIELDS} fields (frame_index g_roll g_pitch), got {}",
                    line_no + 1,
                    f.len()
                )));
            }
            if commands.is_empty() && f[0].eq_ignore_ascii_case("frame_index") {
                continue;
            }
            let idx: usize = f[0].parse().map_err(|e| {
                SimError::Control(format!(
                    "line {}: bad frame index {:?}: {e}",
                    line_no + 1,
                    f[0]
                ))
            })?;
            if idx != commands.len() {
                return Err(SimError::Control(format!(
                    "line {}: frame index {idx}, expected {} (indices must be sequential 0..{expected_frames})",
                    line_no + 1,
                    commands.len()
                )));
            }
            if idx >= expected_frames {
                return Err(SimError::Control(format!(
                    "line {}: frame index {idx} >= frames_per_episode {expected_frames}",
                    line_no + 1
                )));
            }
            let roll = parse_deg(f[1], "g_roll", line_no + 1)?;
            let pitch = parse_deg(f[2], "g_pitch", line_no + 1)?;
            commands.push((roll, pitch));
        }
        if commands.len() != expected_frames {
            return Err(SimError::Control(format!(
                "control file has {} data rows, expected {expected_frames}",
                commands.len()
            )));
        }
        Ok(Self { commands })
    }

    pub fn len(&self) -> usize {
        self.commands.len()
    }

    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    pub fn at(&self, frame_index: usize) -> Result<(f64, f64)> {
        self.commands.get(frame_index).copied().ok_or_else(|| {
            SimError::Control(format!(
                "frame index {frame_index} outside 0..{}",
                self.commands.len()
            ))
        })
    }

    pub fn for_frame(&self, frame: &Frame) -> Result<(f64, f64)> {
        self.at(frame.seq as usize)
    }
}

fn parse_deg(s: &str, name: &str, line: usize) -> Result<f64> {
    let v: f64 = s
        .parse()
        .map_err(|e| SimError::Control(format!("line {line}: bad {name} value {s:?}: {e}")))?;
    let (roll, pitch) = if name == "g_roll" { (v, 0.0) } else { (0.0, v) };
    validate_command(roll, pitch)?;
    Ok(v)
}
