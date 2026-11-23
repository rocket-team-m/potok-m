// SPDX-License-Identifier: GPL-3.0-or-later
//! Gates on the safe frame -> AI command adapter seam.

use potok_sim::dynamics::Phase;
use potok_sim::firmware_adapter::{
    Command, FirmwareAdapter, HostAdapter, HostControl, MockAdapter,
};
use potok_sim::sensor::{Frame, FLAG_AI_READY, FLAG_FAULT};
use potok_sim::MAX_COMMAND_MPS2;

fn frame(seq: u16, time_us: u32, phase: Phase) -> Frame {
    Frame {
        seq,
        time_us,
        phase,
        gyro: [0.01, -0.02, 0.03],
        accel: [0.1, 0.2, 9.8],
        mag: [1.0, 0.0, 0.0],
        height: 123.5,
        pitot_dp: 7,
        flags: FLAG_AI_READY,
        crc: 0,
        gust: false,
        shear: false,
        saturated: false,
        spike: false,
        pitot_valid: true,
        physics: [0.0; 6],
    }
}

const FRAMES: [(u16, u32); 5] = [(0, 0), (1, 20000), (2, 40000), (3, 60000), (4, 80000)];

fn stream(adapter: &mut impl FirmwareAdapter, phase: Phase) -> Vec<Command> {
    adapter.on_phase(phase);
    FRAMES
        .iter()
        .map(|&(seq, t)| adapter.step(&frame(seq, t, phase)))
        .collect()
}

#[test]
fn command_zero_and_bounds_are_explicit() {
    assert!(Command::ZERO.is_zero());
    assert!(Command::ZERO.ai_healthy);
    let limit = MAX_COMMAND_MPS2 as f32;
    assert!(Command {
        ay: limit,
        ax: -limit,
        ai_healthy: true
    }
    .checked()
    .is_some());
    for bad in [
        Command {
            ay: f32::NAN,
            ax: 0.0,
            ai_healthy: true,
        },
        Command {
            ay: 0.0,
            ax: f32::INFINITY,
            ai_healthy: true,
        },
        Command {
            ay: limit + 0.1,
            ax: 0.0,
            ai_healthy: true,
        },
    ] {
        assert!(
            bad.checked().is_none(),
            "out-of-box command accepted: {bad:?}"
        );
    }
}

#[test]
fn mock_is_deterministic_for_same_seed_and_inputs() {
    let a = stream(&mut MockAdapter::new(7), Phase::Coast);
    let b = stream(&mut MockAdapter::new(7), Phase::Coast);
    assert_eq!(a, b);
    assert!(
        a.iter().any(|c| !c.is_zero()),
        "mock never produced a demand"
    );
    assert!(a.iter().all(|c| c.checked().is_some()));
}

#[test]
fn different_seeds_diverge() {
    let a = stream(&mut MockAdapter::new(7), Phase::Coast);
    let b = stream(&mut MockAdapter::new(8), Phase::Coast);
    assert_ne!(a, b);
}

#[test]
fn reset_restores_the_stream_and_reseeds() {
    let mut adapter = MockAdapter::new(7);
    let first = stream(&mut adapter, Phase::Coast);

    adapter.reset(7);
    let replay = stream(&mut adapter, Phase::Coast);
    assert_eq!(first, replay);
    assert_eq!(adapter.tick(), FRAMES.len() as u64);

    adapter.reset(9);
    let changed = stream(&mut adapter, Phase::Coast);
    assert_ne!(first, changed);
}

#[test]
fn phase_gate_zeros_powered_flight_only() {
    for phase in [Phase::Pad, Phase::Boost] {
        let out = stream(&mut MockAdapter::new(3), phase);
        assert!(
            out.iter().all(|c| c.is_zero()),
            "{phase:?} must not allow a command"
        );
    }
    for phase in [Phase::Coast, Phase::Apogee, Phase::Descent, Phase::Landed] {
        let out = stream(&mut MockAdapter::new(3), phase);
        assert!(
            out.iter().any(|c| !c.is_zero()),
            "{phase:?} must allow a command"
        );
    }
}

#[test]
fn on_phase_is_a_live_gate() {
    let mut adapter = MockAdapter::new(11);
    adapter.on_phase(Phase::Pad);
    assert!(adapter.step(&frame(0, 0, Phase::Pad)).is_zero());
    adapter.on_phase(Phase::Coast);
    assert!(!adapter.step(&frame(0, 0, Phase::Coast)).is_zero());
    adapter.on_phase(Phase::Boost);
    assert!(adapter.step(&frame(0, 0, Phase::Boost)).is_zero());
}

#[test]
fn health_and_frame_faults_zero_the_demand() {
    let mut sick = MockAdapter::unhealthy(5);
    let out = stream(&mut sick, Phase::Coast);
    assert!(out.iter().all(|c| c.is_zero() && !c.ai_healthy));

    let mut good = MockAdapter::new(5);
    good.on_phase(Phase::Coast);
    let mut bad_frame = frame(0, 0, Phase::Coast);
    bad_frame.flags |= FLAG_FAULT;
    let out = good.step(&bad_frame);
    assert!(out.is_zero() && out.ai_healthy);

    bad_frame.flags = 0;
    let out = good.step(&bad_frame);
    assert!(out.is_zero() && out.ai_healthy);
}

#[derive(Default)]
struct RecordingControl {
    decisions: Vec<(u16, Command)>,
    resets: u32,
    veto: bool,
}

impl HostControl for RecordingControl {
    fn decide(&mut self, frame: &Frame, proposed: Command) -> Command {
        self.decisions.push((frame.seq, proposed));
        if self.veto || !proposed.ai_healthy {
            Command {
                ay: 0.0,
                ax: 0.0,
                ai_healthy: proposed.ai_healthy,
            }
        } else {
            proposed
        }
    }

    fn reset_link(&mut self) {
        self.resets += 1;
    }
}

#[test]
fn host_adapter_delegates_vetoes_and_resets_the_link() {
    let mut adapter = HostAdapter::new(RecordingControl::default(), MockAdapter::new(7), 7);
    let expected = stream(&mut MockAdapter::new(7), Phase::Coast);

    adapter.on_phase(Phase::Coast);
    let got = stream(&mut adapter, Phase::Coast);
    assert_eq!(got, expected);
    let seen: Vec<u16> = adapter
        .control()
        .decisions
        .iter()
        .map(|(s, _)| *s)
        .collect();
    assert_eq!(seen, FRAMES.iter().map(|f| f.0).collect::<Vec<_>>());

    adapter.control_mut().veto = true;
    let vetoed = adapter.step(&frame(0, 0, Phase::Coast));
    assert!(vetoed.is_zero() && vetoed.ai_healthy);

    adapter.control_mut().veto = false;
    adapter.reset(7);
    assert_eq!(adapter.control().resets, 1);
    assert_eq!(adapter.seed(), 7);
    let replay = stream(&mut adapter, Phase::Coast);
    assert_eq!(replay, expected);
}

#[test]
fn host_adapter_forwards_phase_to_the_producer() {
    let mut adapter = HostAdapter::new(RecordingControl::default(), MockAdapter::new(2), 2);
    adapter.on_phase(Phase::Pad);
    assert!(adapter.step(&frame(0, 0, Phase::Pad)).is_zero());
    adapter.on_phase(Phase::Descent);
    assert!(!adapter.step(&frame(0, 0, Phase::Descent)).is_zero());
    assert_eq!(adapter.producer().phase(), Phase::Descent);
}
