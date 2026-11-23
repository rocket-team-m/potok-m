// SPDX-License-Identifier: GPL-3.0-or-later
//! CHUTE03 integration gates: the canopy model wired into the episode loop.
//!
//! A disabled canopy must leave the fin-flight program bit-identical and emit
//! nothing. An enabled canopy must deploy drogue then main, advect its landing
//! point with the true wind, and supply the POTOKDPL / manifest truth that
//! keeps landing dispersion separate from fin-flight lateral RMS. v1/v2/v3 of
//! `validation.bin` are byte-invariant under the new config field.

use std::path::PathBuf;

use potok_sim::{
    manifest::{
        canonical_config_bytes, chute_manifest_bounds, chute_manifest_path, config_hash,
        read_chute_manifest, write_chute_manifest, NamedRange,
    },
    parachute::{ChutePhase, DeployTrigger, CHUTE_MANIFEST_FIELDS, CHUTE_TRUTH_NAMES},
    trajectory::{landing_dispersion_m, lateral_rms_at_deployment_m, lateral_rms_m},
    validation::{
        deployment_sidecar_path, deployment_truth_from_episodes, read_deployment_truth,
        write_deployment_truth, write_validation_with_version, VERSION_V1, VERSION_V2,
    },
    Config, Episode, Simulator, TrajectoryTick,
};

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("potok-chute03-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn disabled_cfg() -> Config {
    let mut cfg = Config::default();
    cfg.run.duration_s = 5.0;
    cfg.sim.sensors = false;
    assert!(!cfg.parachute.enabled, "the shipped default must be opt-in");
    cfg
}

/// Enabled, but the drogue trigger cannot fire inside a 5 s run. Every fin
/// flight value must therefore be bit-identical to the disabled config.
fn enabled_never_deploy_cfg() -> Config {
    let mut cfg = disabled_cfg();
    cfg.parachute.enabled = true;
    cfg.parachute.drogue.trigger = DeployTrigger::TimeSinceLaunchS { value: 1e5 };
    cfg
}

fn canopy_cfg() -> Config {
    let mut cfg = Config::default();
    // ~230 s of main descent from 1500 m; 360 s guarantees a touchdown.
    cfg.run.duration_s = 360.0;
    cfg.sim.sensors = false;
    cfg.parachute.enabled = true;
    cfg
}

fn run(cfg: &Config) -> Episode {
    Simulator::new(cfg.clone())
        .expect("config must load")
        .run_episode(0)
        .expect("episode must complete")
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn tick_bits(t: &TrajectoryTick) -> Vec<u32> {
    [
        bits(&t.pos),
        bits(&t.vel),
        bits(&t.quat),
        bits(&t.omega),
        bits(&t.wind),
    ]
    .concat()
}

/// Zero-command bit-identity: switching the feature on without ever deploying
/// changes no flown value, and a disabled run emits no canopy truth at all.
#[test]
fn disabled_run_is_bit_identical_and_emits_nothing() {
    let off = run(&disabled_cfg());
    let on = run(&enabled_never_deploy_cfg());

    assert!(off.chute.is_none());
    assert!(
        on.chute.is_none(),
        "a canopy that never fires must emit no truth"
    );
    assert_eq!(off.trajectory.len(), on.trajectory.len());
    for (a, b) in off.trajectory.iter().zip(on.trajectory.iter()) {
        assert_eq!(tick_bits(a), tick_bits(b), "fin-flight tick diverged");
    }
    assert_eq!(off.apogee_m.to_bits(), on.apogee_m.to_bits());
    assert_eq!(off.max_speed_mps.to_bits(), on.max_speed_mps.to_bits());
    assert_eq!(off.frames_generated, on.frames_generated);

    assert!(
        deployment_truth_from_episodes(&disabled_cfg(), &[off])
            .unwrap()
            .is_empty(),
        "a disabled canopy must not produce POTOKDPL records"
    );
    assert!(
        chute_manifest_bounds(&disabled_cfg(), &[run(&disabled_cfg())])
            .unwrap()
            .is_empty(),
        "a disabled canopy must not emit manifest fields"
    );
}

/// The two-stage sequence, one-shot latches, true-wind advection and the final
/// truth row are all carried on the episode for the emitters.
#[test]
fn enabled_episode_deploys_drogue_then_main_and_lands() {
    let cfg = canopy_cfg();
    let ep = run(&cfg);
    let chute = ep.chute.as_ref().expect("a long enough run must land");

    assert!(
        chute.deploy_tick > chute.boost_end_tick,
        "boost must precede deploy"
    );
    let land = &chute.landing;
    assert!(land.drogue_t_s > 0.0);
    assert!(land.main_t_s > land.drogue_t_s, "main must follow drogue");
    assert!(land.landing_t_s > land.main_t_s, "landing must follow main");
    assert!(
        land.deploy_altitude_m > 1500.0,
        "drogue deployed above the main gate"
    );
    assert_eq!(
        land.settled_descent_mps,
        cfg.parachute.main.target_descent_mps
    );

    assert!(
        chute.landing_dispersion_m() > 1.0,
        "true wind must advect the canopy"
    );
    assert!(chute.landing.radial_from_pad_m().is_finite());
    assert!(chute.landing.radial_from_pad_m() >= chute.landing_dispersion_m() - 1e-9);

    assert_eq!(chute.truth_last.len(), CHUTE_TRUTH_NAMES.len());
    assert_eq!(chute.truth_last[0], ChutePhase::Landed.code() as f32);
    assert_eq!(chute.truth_last[7], 1.0, "landed flag must be set");
    assert!(chute.truth_last[2] > 0.0 && chute.truth_last[3] > chute.truth_last[2]);

    // The trajectory now spans the canopy descent too.
    let last = ep.trajectory.last().unwrap();
    assert!(last.pos[2] <= cfg.parachute.landing_altitude_m as f32 + 1e-6);
}

/// POTOKDPL records round-trip and feed the two split metrics, which measure
/// different things: fin-flight RMS is AI-influenceable, landing dispersion is
/// post-canopy wind drift the AI cannot touch.
#[test]
fn potokdpl_records_drive_separate_landing_and_deployment_metrics() {
    let cfg = canopy_cfg();
    let ep = run(&cfg);
    let records = deployment_truth_from_episodes(&cfg, std::slice::from_ref(&ep)).unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].index, ep.meta.index);

    let dir = temp_dir("sidecar");
    let validation = dir.join("validation.bin");
    let sidecar = deployment_sidecar_path(&validation);
    write_deployment_truth(&sidecar, &records).unwrap();
    assert_eq!(read_deployment_truth(&sidecar).unwrap(), records);

    let truth = &records[0];
    let dispersion = landing_dispersion_m(&ep.trajectory, Some(truth)).unwrap();
    let at_deploy = lateral_rms_at_deployment_m(&ep.trajectory, Some(truth)).unwrap();
    let legacy = lateral_rms_m(&ep.trajectory);

    let chute = ep.chute.as_ref().unwrap();
    assert!(
        (dispersion - chute.landing_dispersion_m()).abs() < 40.0,
        "sidecar dispersion {dispersion} disagrees with the model's {}",
        chute.landing_dispersion_m()
    );
    assert!(
        (dispersion - at_deploy).abs() > 1e-6,
        "the two metrics must be reported separately, not aliased ({dispersion} vs {at_deploy})"
    );
    assert!(legacy > 0.0 && at_deploy > 0.0);

    // Fin-flight changes move the deployment metric but cannot move the
    // canopy-only landing dispersion: that is the point of the split.
    let mut perturbed = ep.trajectory.clone();
    perturbed[0].pos = [9_000.0, -4_000.0, 0.0];
    assert_eq!(
        landing_dispersion_m(&perturbed, Some(truth)).unwrap(),
        dispersion,
        "landing dispersion must ignore fin-flight truth"
    );
    assert!(
        lateral_rms_at_deployment_m(&perturbed, Some(truth)).unwrap() > at_deploy + 100.0,
        "fin-flight changes must move the deployment metric"
    );
    assert!(
        landing_dispersion_m(&ep.trajectory, None).is_err(),
        "a missing sidecar must remain a LOUD error"
    );
}

/// The nine manifest fields exist only when the canopy is enabled, and the
/// sidecar is absent otherwise (no zero-filled block).
#[test]
fn manifest_fields_are_emitted_only_when_enabled() {
    let off_cfg = disabled_cfg();
    let off = run(&off_cfg);
    let off_bounds = chute_manifest_bounds(&off_cfg, std::slice::from_ref(&off)).unwrap();
    assert!(off_bounds.is_empty());
    let off_path = chute_manifest_path(&temp_dir("off").join("manifest.json"));
    write_chute_manifest(&off_path, &off_bounds).unwrap();
    assert!(!off_path.exists(), "no sidecar at all when disabled");

    let cfg = canopy_cfg();
    let ep = run(&cfg);
    let bounds = chute_manifest_bounds(&cfg, std::slice::from_ref(&ep)).unwrap();
    assert_eq!(bounds.len(), CHUTE_MANIFEST_FIELDS.len());
    for (range, name) in bounds.iter().zip(CHUTE_MANIFEST_FIELDS.iter()) {
        assert_eq!(range.name, *name);
        assert_eq!(
            range.min, range.max,
            "a single episode has one value per field"
        );
    }
    assert_eq!(bounds[0].min, 1.0, "chute_enabled");
    assert!(bounds[7].min > 0.0, "chute_landing_dispersion_m");

    let dir = temp_dir("on");
    let path = chute_manifest_path(&dir.join("manifest.json"));
    write_chute_manifest(&path, &bounds).unwrap();
    let back: Vec<NamedRange> = read_chute_manifest(&path).unwrap();
    assert_eq!(back, bounds);
}

/// The canonical TOML written into the config hash uses the serde spelling of
/// the trigger kinds and must parse back to the same opt-in field.
#[test]
fn canonical_config_round_trips_with_the_opt_in_field() {
    let cfg = canopy_cfg();
    let text = String::from_utf8(canonical_config_bytes(&cfg).unwrap()).unwrap();
    let back = Config::from_toml_str(&text).unwrap();
    assert!(back.parachute.enabled);
    assert_eq!(back.parachute.drogue.trigger, cfg.parachute.drogue.trigger);
    assert_eq!(back.parachute.main.trigger, cfg.parachute.main.trigger);
    assert_eq!(config_hash(&back).unwrap(), config_hash(&cfg).unwrap());
}

/// Adding the config field and wiring the canopy must not perturb the bytes of
/// the v1/v2/v3 validation layouts: same flown trajectory -> same file.
#[test]
fn validation_v1_v2_v3_layouts_are_byte_invariant() {
    let mut off_cfg = disabled_cfg();
    let mut on_cfg = enabled_never_deploy_cfg();
    off_cfg.sim.sensors = true;
    on_cfg.sim.sensors = true;
    let off = run(&off_cfg);
    let on = run(&on_cfg);
    let dir = temp_dir("layout");

    for (version, meta, targets) in [
        (VERSION_V1, 9usize, 4usize),
        (VERSION_V2, 17, 4),
        (3u32, 17, 6),
    ] {
        let a = dir.join(format!("v{version}-off.bin"));
        let b = dir.join(format!("v{version}-on.bin"));
        write_validation_with_version(&a, &off_cfg, std::slice::from_ref(&off), version).unwrap();
        write_validation_with_version(&b, &on_cfg, std::slice::from_ref(&on), version).unwrap();
        let ba = std::fs::read(&a).unwrap();
        let bb = std::fs::read(&b).unwrap();
        assert_eq!(
            ba, bb,
            "v{version} validation bytes changed with the canopy config"
        );

        let frames = off.frames.len();
        let expected =
            80 + 8 + meta * 4 + frames * off_cfg.feature_count() * 4 + frames * targets * 4;
        assert_eq!(ba.len(), expected, "v{version} stride changed");
    }
}
