<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
# Firmware bridge status (SIMIDEAL finding 8)

**Status: no real firmware bridge.** `cli/src/firmware_cosim.rs` is a host-side
port of the guard *decisions* in the read-only `potok-m-firmware` checkout. It
does not compile firmware code, run a firmware binary, talk to a board, or
claim any timing, electrical or HIL fidelity. Nothing in this seam may be
quoted as evidence that flight hardware was exercised.

This document records the pin, the boundary, and the one loud gate that keeps
the port from drifting. It closes the *provenance* part of the SIMIDEAL finding;
the *real bridge* part stays open until a Cargo-linked implementation of
`Controller` exists (see the last section).

## Pinned provenance

| item | value |
|---|---|
| reference checkout | `/home/sehaxe/potok-m-firmware` (read-only) |
| pinned commit | `87118e07c243a89db2c5247c0a7e3bac9a74a8cf` |
| decision sources | `firmware/src/lib.rs:41-78`, `firmware/src/interlocks.rs:35-112,130-137,239-321`, `firmware/src/control.rs:205-279` |
| owner in sim | `cli/src/firmware_cosim.rs` (`FIRMWARE_COMMIT`, `FIRMWARE_SOURCES`) |

The port carries: the phase/health/flags gate and its priority order, the
three-strike miss latch with explicit reset, the plausibility bound, the rate
reference, and the zero-command veto. Constants (`MAX_CONSECUTIVE_MISSES = 3`,
`MAX_CORRECTION_RAW = 100`, reply scale `0.001`) are transcribed, not fitted.

Deliberately **not** ported, because there is no sim-unit equivalent: the
firmware's raw-step slew (`max_delta = 10`) and throttle (`min_interval_us =
1000`), the Mach/pitot shield, sequence/CRC/version wire checks, deadman and
watchdog timing, and every board I/O path. The sim allows a full command swing
between ticks so the oracle passes through bit-for-bit; that is an explicit
adaptation, not a firmware behaviour claim. The command unit changes from raw
0.001-rad counts (firmware) to desired lateral acceleration in m/s² (sim),
bounded by `MAX_COMMAND_MPS2` as the analogue of `MAX_CORRECTION_RAW`. No
rad→m/s² scale is invented.

## Typed adapter boundary

`Controller` is a compile-time Rust trait (`decide`, `reset`, `implementation`).
`run_guarded_with_controller` takes a `C: Controller` value; `run_guarded` uses
`FirmwareGuard`, the host-port implementation. There is no string dispatch, no
dynamic loading, no subprocess and no function-pointer-from-config on the
decision path. A config value cannot select or alter the controller. The
`Implementation` marker makes the provenance visible at runtime:
`HostPort` today, `External` for a future typed impl.

## Why the firmware crate is not a dependency here

A real link is not feasible inside this seam as it stands:

- the firmware crate is `#![no_std]` and pins `potok-protocol` to a **git
  rev**, while potok-sim uses the local `../../protocol/potok-protocol` path
  crate. Cargo treats those as distinct sources, so a `Frame`/`Phase` built by
  the sim is not the type the firmware's `LinkState` expects;
- hardware dependencies sit behind `cfg(target_os = "none")`, and the board
  boundary (`Board`, `FlightBoard`) is a peripheral trait the sim does not
  implement;
- adding the dependency means editing `cli/Cargo.toml` and the shared protocol
  layout, which are other lanes' files; this task is confined to the seam,
  its tests and this document;
- the raw-count ↔ m/s² conversion has no published source, so wiring it would
  be a new physical claim, not a transcription.

## LOUD mismatch gate

`verify_firmware_baseline()` resolves the checkout's git HEAD, requires it to
equal `FIRMWARE_COMMIT`, and checks the decision-bearing markers (constants,
health/miss latch, interlock limits, `preflight` refusal order) in the pinned
source files. A missing checkout, a moved HEAD, a changed marker or a changed
`preflight` order returns `ProvenanceError` whose message names the cause and
the escape: *re-derive the host port, then bump `FIRMWARE_COMMIT` in
`cli/src/firmware_cosim.rs`*. Nothing is defaulted or skipped.

The gate is exercised by `pinned_firmware_baseline_is_green_and_mismatches_are_loud`
in `cli/tests/firmware_cosim_tests.rs`, which also injects a wrong commit and
a changed gate constant to prove the failure is loud. The older gate set in
`core/tests/firmware_cosim_tests.rs` still pins constants, decision order,
zero-command bit identity, once-per-frame cadence and the oracle positive
control on both host-port and transform paths.

## What would make it a real bridge

1. A Cargo-linked `Controller` implementation that owns a firmware
   `LinkState`/`Interlocks` instance and maps sim frames to typed firmware
   inputs (phase code, flags, health, raw bounded command) without a wire
   codec or a string protocol.
2. A published, unit-tested raw-count ↔ m/s² conversion, or a sim control seam
   that consumes the firmware's native raw command directly.
3. A board-free HIL harness with measured latency, plus a gate that compares
   firmware decisions against this host port tick-for-tick. Until that gate
   exists, the port is validated only against the pinned source text.

None of these is present. Do not describe the current seam as firmware-in-the-
loop, validated against hardware, or a substitute for a flight log.
