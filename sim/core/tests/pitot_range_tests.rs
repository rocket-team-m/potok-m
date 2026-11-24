// SPDX-License-Identifier: GPL-3.0-or-later
//! Typed pitot range / saturation model: range classification, loud reasons,
//! and the explicit experimental extended-range profile.
//!
//! The default (production) profile is byte-for-byte the legacy path: a
//! production frame with a saturated pitot is pinned to a golden wire
//! encoding captured before the model was introduced.

use std::path::Path;

use potok_protocol::{REQUEST_CRC_OFF, REQUEST_OFF_FLAGS};
use potok_sim::{
    atmosphere::Atmosphere,
    config::{Config, Range},
    dynamics::{Phase, State, Vehicle},
    sensor::{
        classify_pitot_range, Frame, PitotProfile, PitotRangeFault, PitotRangeStatus,
        SampleContext, Sensor, EXTENDED_FULL_SCALE_PA, FLAG_AI_READY, FLAG_FAULT, FLAG_PITOT_OK,
    },
};

use PitotRangeFault::{FullScale, NonFinite, WireOverflow, WireUnderflow};
use PitotRangeStatus::{InRange, OutOfRange};

/// Wire bytes of a quiet frame at qdyn = 31 kPa with the production 30 kPa
/// pitot: FAULT set, AI_READY clear, pitot_dp = 31000 (no clamp). Captured
/// from the pre-model code; the production profile must keep producing it.
const GOLDEN_SATURATED_PROD: &str =
    "4d5000000000000001018e0000000000000000000000000000000000000000000ae81c41ae47613e0ad7a33d5c8fc23e000000001879809a";

fn quiet_config(seed: u64) -> Config {
    let mut cfg = Config::default();
    cfg.run.seed = seed;
    let s = &mut cfg.sensor;
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

struct Rig {
    cfg: Config,
    atm: Atmosphere,
    vehicle: Vehicle,
}

fn rig(cfg: Config) -> Rig {
    let profile = Path::new(env!("CARGO_MANIFEST_DIR")).join("../config/boundary_profile.csv");
    let atm =
        Atmosphere::load_with_fast(&profile, cfg.sim.altitude_grid_m, cfg.run.fast_field).unwrap();
    let vehicle = Vehicle::from_config(&cfg, 11.0, 0.0, [0.0, 0.0]);
    Rig { cfg, atm, vehicle }
}

/// Static state whose pitot dynamic pressure is `qdyn` Pa (wind is zero in
/// `measure`, sea-level density is exactly 1.225 in the boundary profile).
fn state_for_qdyn(r: &Rig, qdyn: f64) -> State {
    let rho = r.atm.at(0.0).density;
    let v = (2.0 * qdyn / rho).sqrt();
    let mut st = State::new(11.0, 0.0, 0.0);
    st.vel = [v, 0.0, 0.0];
    st
}

fn measure(s: &mut Sensor, r: &Rig, st: &State) -> Frame {
    s.measure(SampleContext {
        seq: 0,
        t: 0.0,
        state: st,
        vehicle: &r.vehicle,
        atm: &r.atm,
        wind: [0.0; 3],
        phase: Phase::Coast,
        gust: false,
        shear: false,
    })
}

fn hex(f: &Frame) -> String {
    f.to_bytes().iter().map(|b| format!("{b:02x}")).collect()
}

// ---------------------------------------------------------------------------
// Pure classifier: every out-of-range cause is a named reason, never a clamp
// ---------------------------------------------------------------------------

#[test]
fn classifier_names_every_out_of_range_reason() {
    let fs = 30_000.0;
    assert_eq!(classify_pitot_range(0.0, fs), InRange);
    assert_eq!(classify_pitot_range(29_999.0, fs), InRange);
    assert_eq!(classify_pitot_range(-29_999.0, fs), InRange);

    let full = classify_pitot_range(31_000.0, fs);
    assert_eq!(full, OutOfRange(FullScale));
    assert_eq!(full.fault(), Some(FullScale));
    assert!(full.is_out_of_range());
    assert_eq!(InRange.fault(), None);
    assert!(!InRange.is_out_of_range());
    assert_eq!(classify_pitot_range(-31_000.0, fs), OutOfRange(FullScale));

    for nan in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(classify_pitot_range(nan, fs), OutOfRange(NonFinite));
    }

    // Inside the extended sensor full scale but past the i16 wire field: a
    // distinct reason from instrument saturation.
    assert_eq!(
        classify_pitot_range(33_000.0, EXTENDED_FULL_SCALE_PA),
        OutOfRange(WireOverflow)
    );
    assert_eq!(
        classify_pitot_range(-33_000.0, EXTENDED_FULL_SCALE_PA),
        OutOfRange(WireUnderflow)
    );
    assert_eq!(
        classify_pitot_range(31_000.0, EXTENDED_FULL_SCALE_PA),
        InRange
    );

    assert_eq!(FullScale.to_string(), "pitot beyond sensor full scale");
    assert_eq!(WireOverflow.to_string(), "pitot above i16 wire range");
    assert_eq!(WireUnderflow.to_string(), "pitot below i16 wire range");
    assert_eq!(NonFinite.to_string(), "non-finite pitot sample");
}

#[test]
fn profile_full_scale_and_default_selection() {
    assert_eq!(PitotProfile::Production.full_scale_pa(30_000.0), 30_000.0);
    assert_eq!(
        PitotProfile::ExtendedRange.full_scale_pa(30_000.0),
        EXTENDED_FULL_SCALE_PA
    );
    // The experimental full scale covers the 26-31 kPa gate-open qdyn that
    // saturates the shipped 30 kPa instrument.
    assert!(EXTENDED_FULL_SCALE_PA > 31_000.0);
    assert!(!PitotProfile::Production.is_experimental());
    assert!(PitotProfile::ExtendedRange.is_experimental());

    let r = rig(quiet_config(7));
    let s = Sensor::new(&r.cfg, 0);
    assert_eq!(s.pitot_profile(), PitotProfile::Production);
    assert_eq!(s.pitot_range(), InRange);
}

// ---------------------------------------------------------------------------
// Default bytes/layout unchanged
// ---------------------------------------------------------------------------

#[test]
fn production_profile_keeps_the_legacy_wire_bytes() {
    let r = rig(quiet_config(7));
    let st = state_for_qdyn(&r, 31_000.0);
    let mut s = Sensor::new(&r.cfg, 0);
    let f = measure(&mut s, &r, &st);

    assert_eq!(f.pitot_dp, 31_000);
    assert_eq!(s.pitot_range(), OutOfRange(FullScale));
    assert!(f.saturated);
    assert_ne!(f.flags & FLAG_FAULT, 0);
    assert_eq!(f.flags & FLAG_AI_READY, 0);
    assert_eq!(hex(&f), GOLDEN_SATURATED_PROD);

    // A nominal frame stays healthy and keeps its golden flag byte.
    let st_ok = state_for_qdyn(&r, 24_000.0);
    let f_ok = measure(&mut s, &r, &st_ok);
    assert_eq!(f_ok.pitot_dp, 24_000);
    assert_eq!(s.pitot_range(), InRange);
    assert!(!f_ok.saturated);
    assert_eq!(f_ok.flags & FLAG_AI_READY, FLAG_AI_READY);
    assert_eq!(f_ok.flags & FLAG_FAULT, 0);
}

// ---------------------------------------------------------------------------
// The experimental profile removes the pitot dead window without weakening a
// gate: it changes the modeled instrument, not the validity or AI rules.
// ---------------------------------------------------------------------------

#[test]
fn extended_range_removes_the_pitot_dead_window() {
    let r = rig(quiet_config(7));
    let st = state_for_qdyn(&r, 31_000.0);

    let mut prod = Sensor::new(&r.cfg, 0);
    let pf = measure(&mut prod, &r, &st);

    let mut ext = Sensor::with_pitot_profile(&r.cfg, 0, PitotProfile::ExtendedRange);
    assert_eq!(ext.pitot_profile(), PitotProfile::ExtendedRange);
    let ef = measure(&mut ext, &r, &st);

    assert_eq!(prod.pitot_range(), OutOfRange(FullScale));
    assert_eq!(ext.pitot_range(), InRange);

    // Same wire value: no clamp, no changed layout. Only the health/AI flag
    // (and its CRC) move.
    assert_eq!(pf.pitot_dp, ef.pitot_dp);
    let pb = pf.to_bytes();
    let eb = ef.to_bytes();
    let changed: Vec<usize> = (0..pb.len()).filter(|&i| pb[i] != eb[i]).collect();
    assert!(
        changed.iter().all(|&i| {
            i == REQUEST_OFF_FLAGS || i == REQUEST_CRC_OFF || i == REQUEST_CRC_OFF + 1
        }),
        "unexpected wire byte changes at {changed:?}"
    );
    assert!(changed.contains(&REQUEST_OFF_FLAGS));

    // Production: FAULT, no AI. Extended: healthy, valid pitot, AI ready.
    assert_ne!(pf.flags & FLAG_FAULT, 0);
    assert_eq!(pf.flags & FLAG_AI_READY, 0);
    assert!(!pf.ai_input_valid());
    assert_eq!(ef.flags & FLAG_FAULT, 0);
    assert_ne!(ef.flags & FLAG_PITOT_OK, 0);
    assert_ne!(ef.flags & FLAG_AI_READY, 0);
    assert!(!ef.saturated);
    assert!(ef.ai_input_valid());
}

#[test]
fn extended_range_wire_overflow_is_loud_not_silently_clamped() {
    let r = rig(quiet_config(7));
    // 33.5 kPa is inside the 35 kPa extended sensor full scale but above the
    // i16 wire ceiling, so the clamp must carry a named reason.
    let st = state_for_qdyn(&r, 33_500.0);
    let mut s = Sensor::with_pitot_profile(&r.cfg, 0, PitotProfile::ExtendedRange);
    let f = measure(&mut s, &r, &st);

    assert_eq!(s.pitot_range(), OutOfRange(WireOverflow));
    assert_eq!(f.pitot_dp, i16::MAX);
    assert!(f.saturated);
    assert_ne!(f.flags & FLAG_FAULT, 0);
    assert_eq!(f.flags & FLAG_AI_READY, 0);
    assert!(!f.ai_input_valid());

    // The reason is per-frame, not sticky.
    let st_ok = state_for_qdyn(&r, 24_000.0);
    let f_ok = measure(&mut s, &r, &st_ok);
    assert_eq!(s.pitot_range(), InRange);
    assert_eq!(f_ok.flags & FLAG_FAULT, 0);
    assert!(f_ok.ai_input_valid());
}
