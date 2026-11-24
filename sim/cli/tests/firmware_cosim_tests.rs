// SPDX-License-Identifier: GPL-3.0-or-later
//! Gates on the typed co-simulation adapter and its pinned firmware provenance.
//!
//! `firmware_cosim.rs` lives in the CLI crate's source, but `lib.rs` belongs to
//! another lane, so it is included by path here -- the same pattern the
//! firmware SIL harness uses for its parent-owned `lib.rs`. The seam is a host
//! port of guard decisions, not a firmware bridge and not a hardware claim.

#[path = "../src/firmware_cosim.rs"]
mod firmware_cosim;

use std::collections::BTreeMap;

use firmware_cosim::{
    check_firmware_baseline, run_guarded_with_controller, verify_firmware_baseline, Command,
    Controller, Decision, FirmwareGuard, Implementation, ProvenanceError, FIRMWARE_COMMIT,
    FIRMWARE_REPO, FIRMWARE_SOURCES,
};
use potok_sim::{Config, Frame, Simulator};

/// The stale pin from SIMCANOPY. The gate must refuse it: this is the
/// falsification proof that the refreshed pin is load-bearing.
const STALE_FIRMWARE_COMMIT: &str = "87118e07c243a89db2c5247c0a7e3bac9a74a8cf";

fn sim(seed: u64) -> Simulator {
    let mut cfg = Config::default();
    cfg.run.seed = seed;
    Simulator::new(cfg).expect("default config")
}

/// A controller whose command is constant. Ticks prove the seam called it
/// rather than falling back to the host port.
struct FixedController {
    command: Command,
    ticks: usize,
    resets: usize,
}

impl Controller for FixedController {
    fn decide(
        &mut self,
        _frame: &Frame,
        _proposed: Command,
        _ai_healthy: bool,
        _board_fault: bool,
    ) -> Decision {
        self.ticks += 1;
        Decision {
            command: self.command,
            veto: None,
            misses: 0,
            latched: false,
        }
    }

    fn reset(&mut self) {
        self.resets += 1;
    }

    fn implementation(&self) -> Implementation {
        Implementation::External
    }
}

#[test]
fn controller_is_a_compile_time_adapter_that_reaches_the_dynamics() {
    let sim = sim(1);
    let legacy = sim.run_episode(2).expect("legacy episode");
    let controller = FixedController {
        command: Command {
            roll: 1.5,
            pitch: -0.5,
        },
        ticks: 0,
        resets: 0,
    };
    let run = run_guarded_with_controller(&sim, 2, controller, |command| command)
        .expect("guarded episode with external controller");
    assert_eq!(run.guard.ticks, run.episode.frames_generated);
    assert!(run.guard.ticks > 0);
    assert_eq!(run.guard.implementation(), Implementation::External);
    assert_eq!(run.guard.resets, 0, "reset is an explicit typed call");
    assert_ne!(
        run.episode.frames, legacy.frames,
        "the typed controller must reach the dynamics"
    );
}

#[test]
fn host_port_identifies_itself_as_a_port_not_firmware() {
    assert_eq!(
        FirmwareGuard::new().implementation(),
        Implementation::HostPort
    );
    assert_eq!(FIRMWARE_COMMIT.len(), 40);
    assert!(FIRMWARE_SOURCES
        .iter()
        .any(|(path, _)| path.ends_with("interlocks.rs")));
}

fn pinned_sources() -> BTreeMap<String, String> {
    FIRMWARE_SOURCES
        .iter()
        .map(|(path, _)| {
            let full = format!("{FIRMWARE_REPO}/{path}");
            let text = std::fs::read_to_string(&full).unwrap_or_else(|e| {
                panic!("firmware baseline unavailable at {full}: {e} -- clone the pinned checkout")
            });
            ((*path).to_owned(), text)
        })
        .collect()
}

fn check_with(files: &BTreeMap<String, String>, commit: &str) -> Result<(), ProvenanceError> {
    check_firmware_baseline(commit, |path| {
        files
            .get(path)
            .cloned()
            .ok_or_else(|| format!("no pinned source for {path}"))
    })
}

#[test]
fn pinned_firmware_baseline_is_green_and_mismatches_are_loud() {
    // The live checkout must still be the pin; otherwise this gate names why
    // and tells the reader what to do about it.
    verify_firmware_baseline().unwrap_or_else(|e| panic!("firmware provenance gate: {e}"));

    let files = pinned_sources();
    check_with(&files, FIRMWARE_COMMIT).expect("pinned sources match the port");

    // Falsification: the exact stale hash from SIMCANOPY must be refused even
    // with otherwise-valid pinned sources. If it is accepted, the pin is dead.
    let wrong_head = check_with(&files, STALE_FIRMWARE_COMMIT).unwrap_err();
    assert!(matches!(wrong_head, ProvenanceError::CommitMismatch { .. }));
    assert!(
        wrong_head.to_string().contains("bump FIRMWARE_COMMIT"),
        "{wrong_head}"
    );

    let key = "firmware/src/interlocks.rs";
    let mut moved = files.clone();
    let text = moved
        .get(key)
        .expect("interlocks source")
        .replace("max_delta: 10,", "max_delta: 11,");
    moved.insert(key.to_owned(), text);
    let moved_logic = check_with(&moved, FIRMWARE_COMMIT).unwrap_err();
    assert!(
        matches!(
            moved_logic,
            ProvenanceError::SourceMismatch {
                marker: "max_delta: 10,",
                ..
            }
        ),
        "{moved_logic:?}"
    );
    assert!(
        moved_logic.to_string().contains("re-derive the host port"),
        "{moved_logic}"
    );

    let mut reordered = files;
    let text = reordered
        .get(key)
        .expect("interlocks source")
        .replace("fn preflight(", "fn old_preflight(");
    reordered.insert(key.to_owned(), text);
    assert!(matches!(
        check_with(&reordered, FIRMWARE_COMMIT),
        Err(ProvenanceError::SourceMismatch {
            marker: "fn preflight(",
            ..
        })
    ));
}
