// SPDX-License-Identifier: GPL-3.0-or-later
//! SIMPITOTWIRE — the typed `pitot_profile` config flag reaches `Sensor`.
//!
//! The default (`production`) arm keeps every golden byte and the same config
//! hash. The opt-in `extended` arm widens the modeled instrument so the
//! 26-31 kPa gate-open window stays in range, without touching a gate, a
//! threshold or the RNG draw order.

use potok_protocol::{REQUEST_CRC_OFF, REQUEST_OFF_FLAGS};
use potok_sim::{
    config::{PitotProfileConfig, Range, DEFAULT_TOML},
    manifest::{config_hash, sha256_hex},
    sensor::{FLAG_AI_READY, FLAG_FAULT},
    Config, Simulator,
};

fn with_profile(toml_text: &str, value: &str) -> String {
    let needle = "[sensor]\n";
    assert!(
        toml_text.contains(needle),
        "fixture needs a [sensor] section"
    );
    toml_text.replacen(
        needle,
        &format!("[sensor]\npitot_profile = \"{value}\"\n"),
        1,
    )
}

/// Zero-noise, zero-bias fixture: the only thing that can saturate a frame is
/// the pitot instrument under test. Sensor lag and all other saturations are
/// pushed out of the way; the truth dynamics are untouched.
fn quiet_config(seed: u64, profile: PitotProfileConfig) -> Config {
    let mut cfg = Config::default();
    cfg.run.seed = seed;
    let s = &mut cfg.sensor;
    s.pitot_profile = profile;
    s.gyro_noise_range = Range(0.0, 0.0);
    s.accel_noise_range = Range(0.0, 0.0);
    s.gyro_noise = 0.0;
    s.accel_noise = 0.0;
    s.mag_noise = 0.0;
    s.baro_noise = 0.0;
    s.pitot_noise = 0.0;
    s.gyro_bias = [0.0; 3];
    s.accel_bias = [0.0; 3];
    s.mag_bias = [0.0; 3];
    s.baro_bias = 0.0;
    s.pitot_bias = 0.0;
    s.quant_step = 0.0;
    s.gyro_quant_step = 0.0;
    s.accel_quant_step = 0.0;
    s.lag_tau = 1e-6;
    s.latency_steps = 0;
    s.gyro_sat = 1e6;
    s.accel_sat = 1e6;
    s.mag_sat = 0.0;
    s.baro_sat = 0.0;
    s.pitot_sat = 30_000.0;
    s.spike_probability = 0.0;
    s.spike_amplitude = 0.0;
    cfg
}

fn frame_bytes(frames: &[potok_sim::Frame]) -> Vec<u8> {
    let mut out = Vec::with_capacity(frames.len() * potok_sim::sensor::FRAME_SIZE);
    for f in frames {
        out.extend_from_slice(&f.to_bytes());
    }
    out
}

// ---------------------------------------------------------------------------
// Config boundary: typed flag, loud on an unknown value, invisible on default
// ---------------------------------------------------------------------------

#[test]
fn config_flag_is_typed_and_default_keeps_canonical_bytes() {
    let cfg = Config::default();
    assert_eq!(cfg.sensor.pitot_profile, PitotProfileConfig::Production);

    // The flag is skipped when it holds the default, so a config written
    // before the flag existed serializes to exactly the same bytes.
    let canonical = toml::to_string(&cfg).expect("config serializes");
    assert!(
        !canonical.contains("pitot_profile"),
        "default canonical config gained a new field"
    );
    let absent = Config::from_toml_str(DEFAULT_TOML).expect("default.toml loads");
    assert_eq!(config_hash(&absent).unwrap(), config_hash(&cfg).unwrap());

    // An explicit `production` is also byte-identical to an absent flag.
    let explicit = Config::from_toml_str(&with_profile(DEFAULT_TOML, "production")).unwrap();
    assert_eq!(
        explicit.sensor.pitot_profile,
        PitotProfileConfig::Production
    );
    assert_eq!(config_hash(&explicit).unwrap(), config_hash(&cfg).unwrap());

    // `extended` is a distinct, non-default config and must be visible in the
    // provenance hash so a dataset built with it cannot masquerade as default.
    let extended = Config::from_toml_str(&with_profile(DEFAULT_TOML, "extended")).unwrap();
    assert_eq!(extended.sensor.pitot_profile, PitotProfileConfig::Extended);
    assert!(toml::to_string(&extended)
        .unwrap()
        .contains("pitot_profile = \"extended\""));
    assert_ne!(config_hash(&extended).unwrap(), config_hash(&cfg).unwrap());

    // Unknown spelling is a loud config error, never a silent fallback.
    let bad = Config::from_toml_str(&with_profile(DEFAULT_TOML, "bogus"));
    assert!(bad.is_err(), "unknown pitot_profile must be refused loudly");
}

// ---------------------------------------------------------------------------
// Episode wiring: the flag changes the instrument, not the gates or the bytes
// outside the dead window
// ---------------------------------------------------------------------------

#[test]
fn episode_uses_the_configured_pitot_profile() {
    let prod_cfg = quiet_config(7, PitotProfileConfig::Production);
    let ext_cfg = quiet_config(7, PitotProfileConfig::Extended);
    let prod = Simulator::new(prod_cfg).unwrap().run_episode(0).unwrap();
    let ext = Simulator::new(ext_cfg).unwrap().run_episode(0).unwrap();
    assert_eq!(prod.frames.len(), ext.frames.len());
    assert!(!prod.frames.is_empty());

    let mut dead_window = 0usize;
    let mut ai_unblocked = 0usize;
    let mut wire_ceiling = 0usize;
    for (i, (pf, ef)) in prod.frames.iter().zip(&ext.frames).enumerate() {
        assert_eq!(
            pf.pitot_dp, ef.pitot_dp,
            "frame {i}: the instrument must not change the wire value"
        );
        if pf.pitot_dp <= 30_000 || pf.pitot_dp == i16::MAX {
            // Below the production full scale, or past the frozen i16 wire
            // ceiling (boost peak): both instruments reject the sample and
            // the wire bytes cannot move.
            if pf.pitot_dp == i16::MAX {
                wire_ceiling += 1;
            }
            assert_eq!(
                pf.to_bytes(),
                ef.to_bytes(),
                "frame {i}: outside the dead window the bytes must match"
            );
            continue;
        }
        dead_window += 1;
        // Production: saturated over the 26-31 kPa gate-open window.
        assert!(pf.saturated, "frame {i}: production must flag saturation");
        assert_ne!(pf.flags & FLAG_FAULT, 0, "frame {i}: production FAULT");
        assert_eq!(
            pf.flags & FLAG_AI_READY,
            0,
            "frame {i}: production AI_READY"
        );

        // Extended: same measurement, wider instrument, no gate change.
        assert!(
            !ef.saturated,
            "frame {i}: dp={} phase={:?}: extended instrument must stay in range",
            ef.pitot_dp, ef.phase
        );
        assert_eq!(ef.flags & FLAG_FAULT, 0, "frame {i}: extended FAULT");
        if pf.phase.ai_allowed() {
            ai_unblocked += 1;
            assert_ne!(ef.flags & FLAG_AI_READY, 0, "frame {i}: extended AI_READY");
            assert!(ef.ai_input_valid(), "frame {i}: extended AI input valid");
        }
        let pb = pf.to_bytes();
        let eb = ef.to_bytes();
        let changed: Vec<usize> = (0..pb.len()).filter(|&j| pb[j] != eb[j]).collect();
        assert!(
            changed.iter().all(|&j| {
                j == REQUEST_OFF_FLAGS || j == REQUEST_CRC_OFF || j == REQUEST_CRC_OFF + 1
            }),
            "frame {i}: unexpected wire byte changes at {changed:?}"
        );
    }
    assert!(
        dead_window > 0,
        "fixture never entered the 30 kPa dead window"
    );
    assert!(
        ai_unblocked > 0,
        "fixture never had a gate-open dead-window frame"
    );
    assert!(wire_ceiling > 0, "fixture never hit the i16 wire ceiling");
}

// ---------------------------------------------------------------------------
// Production golden and extended determinism
// ---------------------------------------------------------------------------

#[test]
fn default_episode_keeps_the_production_golden_bytes() {
    let cfg = Config::default();
    let sim = Simulator::new(cfg).unwrap();
    let ep = sim.run_episode(0).unwrap();
    assert_eq!(
        sha256_hex(&frame_bytes(&ep.frames)),
        "c0dec040b5b8ca4ac2709bc4ee74e7f42ffbf654ebc59682cf517f02ef7b547b",
        "default episode frame bytes changed; record the new pin deliberately"
    );
}

#[test]
fn extended_profile_episode_is_deterministic() {
    let cfg = quiet_config(7, PitotProfileConfig::Extended);
    let a = Simulator::new(cfg.clone()).unwrap().run_episode(0).unwrap();
    let b = Simulator::new(cfg).unwrap().run_episode(0).unwrap();
    assert_eq!(frame_bytes(&a.frames), frame_bytes(&b.frames));
    assert_eq!(a.features, b.features);
}
