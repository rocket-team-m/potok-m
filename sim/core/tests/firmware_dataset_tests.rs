//! Firmware-adapter command source: v3 layout, per-tick provenance, loud
//! failures and deterministic regeneration.
//!
//! The CLI module is included by path because its `lib.rs` wiring is owned by
//! the integration task; its absolute `potok_sim::` paths resolve against the
//! real library, which already declares `firmware_dataset`.
//!
//! SPDX-License-Identifier: GPL-3.0-or-later

#[path = "../../cli/src/firmware_dataset.rs"]
mod cli_firmware_dataset;

use std::{fs, path::PathBuf};

use potok_sim::{
    firmware_adapter::{Command, FirmwareAdapter},
    firmware_dataset::{
        generate_dataset, generate_episode, generate_reference_episode, indices_for,
        read_provenance, write_provenance, CommandProvider, CommandProvenanceMeta, CommandSource,
        EpisodeProvenance, FirmwareDatasetSpec, ProvenanceSet, RECORD_SIZE,
    },
    manifest::read_manifest,
    quality::check_dataset,
    read_trajectory,
    trajectory::{sidecar_path, write_trajectory_v3, VERSION_V3},
    validation::{read_validation, write_validation, TARGET_NAMES, VERSION as VALIDATION_V3},
    Config, Frame, Phase, Simulator, Split,
};

use cli_firmware_dataset::{run, run_with_builtin, AdapterKind, FirmwareDatasetArgs};

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("potok-fwds-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// A deterministic RLVR01 adapter whose command is constant outside the AI
/// interlock. Its name is owned so overlong-name tests are possible.
struct ScriptedAdapter {
    command: [f32; 2],
    phase: Phase,
}

impl ScriptedAdapter {
    fn new(command: [f32; 2]) -> Self {
        Self {
            command,
            phase: Phase::Pad,
        }
    }
}

impl FirmwareAdapter for ScriptedAdapter {
    fn step(&mut self, frame: &Frame) -> Command {
        if self.phase.ai_allowed() && frame.ai_input_valid() {
            Command {
                ay: self.command[0],
                ax: self.command[1],
                ai_healthy: true,
            }
        } else {
            Command::ZERO
        }
    }

    fn reset(&mut self, _seed: u64) {
        self.phase = Phase::Pad;
    }

    fn on_phase(&mut self, phase: Phase) {
        self.phase = phase;
    }
}

#[test]
fn offline_reference_arm_reproduces_standard_v3_bytes() {
    let mut cfg = Config::default();
    cfg.run.seed = 424_242;
    let sim = Simulator::new(cfg.clone()).unwrap();
    let (episode, samples) = generate_reference_episode(&sim, 7).unwrap();

    // The same fin command through the simulator's own oracle seam.
    let reference = sim.run_episode_with_oracle(7).unwrap();
    assert_eq!(episode.trajectory, reference.trajectory);
    assert_eq!(episode.targets, reference.targets);

    let dir = temp_dir("offline-bytes");
    let generated = dir.join("generated.bin");
    let ref_validation = dir.join("reference.bin");
    write_validation(&generated, &cfg, &[episode.clone()]).unwrap();
    write_validation(&ref_validation, &cfg, &[reference.clone()]).unwrap();
    let generated_t = sidecar_path(&generated);
    let reference_t = sidecar_path(&ref_validation);
    write_trajectory_v3(&generated_t, &cfg, &[episode]).unwrap();
    write_trajectory_v3(&reference_t, &cfg, &[reference]).unwrap();
    assert_eq!(fs::read(&generated).unwrap(), fs::read(&ref_validation).unwrap());
    assert_eq!(fs::read(&generated_t).unwrap(), fs::read(&reference_t).unwrap());

    let frames = cfg.frames_per_episode();
    assert_eq!(samples.len(), frames);
    assert!(samples
        .iter()
        .all(|s| s.source == CommandSource::OfflineOptimizer));
    assert!(samples.iter().all(|s| s.adapter == "offline_optimizer"));
    assert!(samples
        .iter()
        .any(|s| s.world_ay.abs() > 1e-6 || s.world_ax.abs() > 1e-6));

    let v = read_validation(&generated).unwrap();
    for (s, row) in samples.iter().zip(v.episodes[0].targets.rows()) {
        assert_eq!(s.world_ay, row[4]);
        assert_eq!(s.world_ax, row[5]);
    }
}

#[test]
fn firmware_command_stream_replaces_the_offline_target_and_reaches_dynamics() {
    let cfg = Config::default();
    let sim = Simulator::new(cfg.clone()).unwrap();
    let mut adapter = ScriptedAdapter::new([2.0, -1.0]);
    let meta = CommandProvenanceMeta::firmware("scripted_v1");
    let (episode, samples) = generate_episode(&sim, 3, &mut adapter, &meta).unwrap();
    let oracle = sim.run_episode_with_oracle(3).unwrap();
    assert_ne!(
        episode.trajectory, oracle.trajectory,
        "the firmware command must reach the fin actuator"
    );

    let dir = temp_dir("scripted");
    let validation = dir.join("validation.bin");
    write_validation(&validation, &cfg, &[episode.clone()]).unwrap();
    let sidecar = sidecar_path(&validation);
    write_trajectory_v3(&sidecar, &cfg, &[episode]).unwrap();
    let v = read_validation(&validation).unwrap();
    let t = read_trajectory(&sidecar).unwrap();

    assert_eq!(v.version, VALIDATION_V3);
    assert_eq!(v.header.target_count as usize, TARGET_NAMES.len());
    for (i, (s, row)) in samples.iter().zip(v.episodes[0].targets.rows()).enumerate() {
        assert_eq!(s.source, CommandSource::Firmware);
        assert_eq!(s.adapter, "scripted_v1");
        assert_eq!(s.world_ay, row[4], "tick {i}: v3 target ay");
        assert_eq!(s.world_ax, row[5], "tick {i}: v3 target ax");
        assert_eq!(row[4], t.episodes[0].ticks[i].g_accel[0]);
        assert_eq!(row[5], t.episodes[0].ticks[i].g_accel[1]);
        assert_eq!((row[4], row[5]), (s.fin_roll, -s.fin_pitch));
        if s.ai_healthy {
            assert_eq!(row[4], 2.0);
            assert_eq!(row[5], -1.0); // command [ay, ax] = [2.0, -1.0]; fin [ay, -ax]
        } else {
            assert_eq!(row[4], 0.0);
            assert_eq!(row[5], 0.0);
        }
    }
    assert!(samples.iter().any(|s| s.ai_healthy));

    check_dataset("train", &cfg, &v, &t, 1).unwrap();
}

#[test]
fn provenance_roundtrip_and_loud_corruption() {
    let cfg = Config::default();
    let sim = Simulator::new(cfg.clone()).unwrap();
    let mut adapter = ScriptedAdapter::new([1.5, 0.5]);
    let meta = CommandProvenanceMeta::firmware("prov_v2");
    let (_, samples) = generate_episode(&sim, 1, &mut adapter, &meta).unwrap();
    let set = ProvenanceSet {
        frames_per_episode: cfg.frames_per_episode() as u32,
        episodes: vec![EpisodeProvenance {
            index: 1,
            samples,
        }],
    };
    let dir = temp_dir("prov");
    let path = dir.join("validation.command_provenance");
    write_provenance(&path, &set).unwrap();
    assert_eq!(read_provenance(&path).unwrap(), set);
    assert_eq!(
        fs::metadata(&path).unwrap().len() as usize,
        64 + 8 + cfg.frames_per_episode() * RECORD_SIZE
    );

    let mut bytes = fs::read(&path).unwrap();
    bytes[0] = b'X';
    fs::write(&path, &bytes).unwrap();
    assert!(read_provenance(&path)
        .unwrap_err()
        .to_string()
        .contains("magic"));

    let mut bytes = fs::read(&path).unwrap();
    bytes[0] = b'P';
    bytes[8..12].copy_from_slice(&9u32.to_le_bytes());
    fs::write(&path, &bytes).unwrap();
    assert!(read_provenance(&path)
        .unwrap_err()
        .to_string()
        .contains("version"));

    // First tick record starts after the 64-byte header and u64 episode index.
    let mut bytes = fs::read(&path).unwrap();
    bytes[8..12].copy_from_slice(&1u32.to_le_bytes()); // restore version first
    bytes[64 + 8 + 4] = 9;
    fs::write(&path, &bytes).unwrap();
    assert!(read_provenance(&path)
        .unwrap_err()
        .to_string()
        .contains("source"));
}

#[test]
fn firmware_split_preserves_v3_layout_manifest_truth_and_is_deterministic() {
    let mut cfg = Config::default();
    cfg.run.seed = 9_001;
    cfg.output.truth_metadata = true;
    let indices = indices_for(Split::Train, 3);
    assert_eq!(indices.len(), 3);
    let dir = temp_dir("split-a");
    let out = generate_dataset(FirmwareDatasetSpec {
        cfg: &cfg,
        provider: CommandProvider::Firmware {
            adapter: Box::new(ScriptedAdapter::new([2.0, -1.0])),
            meta: CommandProvenanceMeta::firmware("scripted_v1"),
        },
        indices: &indices,
        split: Split::Train,
        out_dir: &dir,
    })
    .unwrap();

    let v = read_validation(&out.validation).unwrap();
    assert_eq!(v.version, VALIDATION_V3);
    assert_eq!(v.header.target_count as usize, TARGET_NAMES.len());
    assert_eq!(v.header.meta_count as usize, 39);
    assert_eq!(v.episodes.len(), 3);
    assert!(v.episodes.iter().all(|e| e.truth.is_some()));

    let t = read_trajectory(&out.trajectory).unwrap();
    assert!(t.is_v3() && t.is_privileged());
    assert_eq!(t.version, VERSION_V3);
    assert!(t.episodes[0]
        .ticks
        .iter()
        .any(|tick| tick.quat.iter().any(|q| *q != 0.0)));

    let m = read_manifest(&out.manifest).unwrap();
    assert_eq!(m.schema_version, 1);
    assert_eq!(m.split, "train");
    assert_eq!(m.seed, cfg.run.seed);
    assert_eq!(m.episode_count, 3);
    assert_eq!(m.files.len(), 3);
    assert_eq!(m.files[0].name, "validation.bin");
    assert_eq!(m.files[1].name, "validation.trajectory");
    assert_eq!(m.files[2].name, "validation.command_provenance");
    assert_eq!(
        m.target_names,
        TARGET_NAMES.iter().map(|s| s.to_string()).collect::<Vec<_>>()
    );
    check_dataset("train", &cfg, &v, &t, 3).unwrap();

    assert_eq!(
        out.provenance,
        cli_firmware_dataset::provenance_path(&dir)
    );
    let p = read_provenance(&out.provenance).unwrap();
    assert_eq!(p.episodes.len(), 3);
    assert_eq!(p.tick_count(), 3 * cfg.frames_per_episode() as u64);
    assert!(p.episodes.iter().all(|e| e.samples.iter().all(|s| {
        s.source == CommandSource::Firmware && s.adapter == "scripted_v1"
    })));

    let dir2 = temp_dir("split-b");
    let out2 = generate_dataset(FirmwareDatasetSpec {
        cfg: &cfg,
        provider: CommandProvider::Firmware {
            adapter: Box::new(ScriptedAdapter::new([2.0, -1.0])),
            meta: CommandProvenanceMeta::firmware("scripted_v1"),
        },
        indices: &indices,
        split: Split::Train,
        out_dir: &dir2,
    })
    .unwrap();
    for (a, b) in [
        (&out.validation, &out2.validation),
        (&out.trajectory, &out2.trajectory),
        (&out.provenance, &out2.provenance),
        (&out.manifest, &out2.manifest),
    ] {
        assert_eq!(fs::read(a).unwrap(), fs::read(b).unwrap(), "{}", a.display());
    }
}

#[test]
fn cli_helper_generates_both_splits_with_a_pluggable_adapter() {
    let args = FirmwareDatasetArgs {
        seed: 5,
        train_episodes: 3,
        held_out_episodes: 3,
        out: temp_dir("cli"),
        config: None,
        min_episodes_per_split: 3,
    };
    let out = run_with_builtin(&args, AdapterKind::Zero).unwrap();
    assert_eq!(out.train.validation.file_name().unwrap(), "validation.bin");
    for split in [&out.train, &out.held_out] {
        let v = read_validation(&split.validation).unwrap();
        assert!(v.episodes.iter().all(|e| {
            e.targets.rows().into_iter().all(|r| r[4] == 0.0 && r[5] == 0.0)
        }));
        let p = read_provenance(&split.provenance).unwrap();
        assert!(p.episodes.iter().all(|e| e.samples.iter().all(|s| {
            s.adapter == "firmware_zero" && s.source == CommandSource::Firmware
        })));
    }
    assert_eq!(read_manifest(&out.train.manifest).unwrap().split, "train");
    assert_eq!(read_manifest(&out.held_out.manifest).unwrap().split, "held-out");

    // RLVR01's MockAdapter is pluggable through the same CLI entry point.
    let mock = run_with_builtin(
        &FirmwareDatasetArgs {
            out: temp_dir("cli-mock"),
            ..args.clone()
        },
        AdapterKind::Mock,
    )
    .unwrap();
    let p = read_provenance(&mock.train.provenance).unwrap();
    assert!(p.episodes.iter().all(|e| e.samples.iter().all(|s| s.adapter == "firmware_mock")));

    let reference = run_with_builtin(
        &FirmwareDatasetArgs {
            out: temp_dir("cli-reference"),
            ..args.clone()
        },
        AdapterKind::OfflineReference,
    )
    .unwrap();
    let p = read_provenance(&reference.train.provenance).unwrap();
    assert!(p.episodes.iter().all(|e| e.samples.iter().all(|s| {
        s.source == CommandSource::OfflineOptimizer
    })));

    // A caller-supplied adapter plugs in through `run` with a provider factory.
    let plugged = run(
        &FirmwareDatasetArgs {
            out: temp_dir("cli-plug"),
            ..args.clone()
        },
        |_| CommandProvider::Firmware {
            adapter: Box::new(ScriptedAdapter::new([1.0, -0.5])),
            meta: CommandProvenanceMeta::firmware("plugged_in"),
        },
    )
    .unwrap();
    let p = read_provenance(&plugged.train.provenance).unwrap();
    assert!(p.episodes.iter().all(|e| e.samples.iter().all(|s| s.adapter == "plugged_in")));
}

#[test]
fn invalid_command_and_overlong_name_fail_loudly() {
    let cfg = Config::default();
    let sim = Simulator::new(cfg.clone()).unwrap();
    let mut bad = ScriptedAdapter::new([1e9, 0.0]);
    let bad_meta = CommandProvenanceMeta::firmware("bad");
    let err = generate_episode(&sim, 0, &mut bad, &bad_meta)
        .unwrap_err()
        .to_string();
    assert!(err.contains("command") || err.contains("control"), "{err}");

    let long = CommandProvenanceMeta::firmware("x".repeat(64));
    let mut ok = ScriptedAdapter::new([0.0, 0.0]);
    let err = generate_episode(&sim, 0, &mut ok, &long)
        .unwrap_err()
        .to_string();
    assert!(err.contains("adapter name"), "{err}");
}
