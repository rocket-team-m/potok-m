//! CLI helper for RLVR01-adapter dataset generation.
//!
//! Deliberately free of `clap` types: the argument struct is a plain value so
//! generation and the quality gates can be exercised from the core test suite
//! without a binary. Wire it into `potok-cli` by declaring
//! `pub mod firmware_dataset;` and forwarding a subcommand's fields into
//! [`FirmwareDatasetArgs`], then calling [`run_with_builtin`] or [`run`].
//!
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::path::{Path, PathBuf};

use potok_sim::{
    firmware_adapter::{Command, FirmwareAdapter, MockAdapter},
    firmware_dataset::{
        generate_dataset, indices_for, CommandProvider, CommandProvenanceMeta,
        FirmwareDatasetOutput, FirmwareDatasetSpec,
    },
    quality::{check_dataset, check_split_disjoint},
    read_trajectory,
    validation::read_validation,
    Config, Phase, Result, SimError, Split, PHYSICS_FEATURES,
};

pub use potok_sim::firmware_dataset::PROVENANCE_FILE_NAME as PROVENANCE_SIDECAR;

/// Path of the provenance sidecar inside a split directory.
pub fn provenance_path(dir: &Path) -> PathBuf {
    dir.join(PROVENANCE_SIDECAR)
}

#[derive(Debug, Clone)]
pub struct FirmwareDatasetArgs {
    pub seed: u64,
    pub train_episodes: usize,
    pub held_out_episodes: usize,
    pub out: PathBuf,
    pub config: Option<PathBuf>,
    pub min_episodes_per_split: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FirmwareCliOutput {
    pub train: FirmwareDatasetOutput,
    pub held_out: FirmwareDatasetOutput,
}

/// Built-in command sources selectable by a CLI flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdapterKind {
    /// Byte-for-byte control arm: the privileged offline reference controller.
    OfflineReference,
    /// RLVR01's deterministic sensor-only producer.
    Mock,
    /// Zero demand on every tick (seam proof and smoke arm).
    Zero,
}

/// A zero-command RLVR01 adapter. If the generated v3 `g_accel` is not all
/// zero, the command stream did not come from the adapter.
pub struct ZeroAdapter;

impl FirmwareAdapter for ZeroAdapter {
    fn step(&mut self, _frame: &potok_sim::Frame) -> Command {
        Command::ZERO
    }

    fn reset(&mut self, _seed: u64) {}

    fn on_phase(&mut self, _phase: Phase) {}
}

impl AdapterKind {
    /// Each split gets a fresh adapter so the per-episode reset is clean.
    pub fn provider(self) -> CommandProvider {
        match self {
            Self::OfflineReference => CommandProvider::OfflineReference,
            Self::Mock => CommandProvider::Firmware {
                adapter: Box::new(MockAdapter::new(0)),
                meta: CommandProvenanceMeta::firmware("firmware_mock"),
            },
            Self::Zero => CommandProvider::Firmware {
                adapter: Box::new(ZeroAdapter),
                meta: CommandProvenanceMeta::firmware("firmware_zero"),
            },
        }
    }
}

/// Generate both splits with one provider factory (called once per split) and
/// run the standard quality gates on the written bytes. Every failure is LOUD.
pub fn run<F>(args: &FirmwareDatasetArgs, mut factory: F) -> Result<FirmwareCliOutput>
where
    F: FnMut(Split) -> CommandProvider,
{
    if args.train_episodes == 0 || args.held_out_episodes == 0 {
        return Err(SimError::Dataset(
            "firmware-dataset needs a nonzero train and held-out episode count".into(),
        ));
    }
    let mut cfg = match &args.config {
        Some(p) => Config::from_file(p)?,
        None => Config::default(),
    };
    cfg.run.seed = args.seed;
    // Same canonical 14-channel model frame as the standard dataset generator.
    for name in PHYSICS_FEATURES {
        if !cfg.output.features.iter().any(|f| f == name) {
            cfg.output.features.push((*name).to_string());
        }
    }
    let train_idx = indices_for(Split::Train, args.train_episodes);
    let held_idx = indices_for(Split::HeldOut, args.held_out_episodes);
    if train_idx.len() < args.min_episodes_per_split || held_idx.len() < args.min_episodes_per_split {
        return Err(SimError::Dataset(format!(
            "need {} episodes per split; got train={} held-out={}",
            args.min_episodes_per_split,
            train_idx.len(),
            held_idx.len()
        )));
    }

    let train_dir = args.out.join(Split::Train.label());
    let held_dir = args.out.join(Split::HeldOut.label());
    let train = generate_dataset(FirmwareDatasetSpec {
        cfg: &cfg,
        provider: factory(Split::Train),
        indices: &train_idx,
        split: Split::Train,
        out_dir: &train_dir,
    })?;
    let held_out = generate_dataset(FirmwareDatasetSpec {
        cfg: &cfg,
        provider: factory(Split::HeldOut),
        indices: &held_idx,
        split: Split::HeldOut,
        out_dir: &held_dir,
    })?;

    check_dataset(
        Split::Train.label(),
        &cfg,
        &read_validation(&train.validation)?,
        &read_trajectory(&train.trajectory)?,
        args.min_episodes_per_split,
    )?;
    check_dataset(
        Split::HeldOut.label(),
        &cfg,
        &read_validation(&held_out.validation)?,
        &read_trajectory(&held_out.trajectory)?,
        args.min_episodes_per_split,
    )?;
    check_split_disjoint(
        &read_validation(&train.validation)?,
        &read_validation(&held_out.validation)?,
    )?;
    Ok(FirmwareCliOutput { train, held_out })
}

/// Convenience wrapper for a CLI flag that selects one of the built-ins.
pub fn run_with_builtin(
    args: &FirmwareDatasetArgs,
    kind: AdapterKind,
) -> Result<FirmwareCliOutput> {
    run(args, |_| kind.provider())
}
