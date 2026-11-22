# Firmware plan: STM32H743, Rust and Embassy

The owner switched the STM32 firmware from C to Rust with Embassy. This file records the plan. No firmware code was written, and nothing was created under `firmware/`.

## Target and build

- MCU: STM32H743, Cortex-M7. Target `thumbv7em-none-eabihf`.
- Embassy through `embassy-stm32`, with one hardware timer enabled as the time driver. `defmt` for logging over RTT, `probe-rs` for flash and debug.
- `panic = "abort"` in the release profile. The panic handler logs and resets the MCU. No unwinding.
- No allocator. If a dependency pulls in `alloc`, replace the dependency.

## Shared protocol crate

- Both the Pi and the firmware use `protocol/potok-protocol`. The generated file `protocol/generated/protocol.rs` stays the single source of truth, with `uart.yaml` upstream.
- The crate needs `#![no_std]` by default and a `std` feature for host tests and Pi tooling. Gate `impl std::error::Error for Error` at `potok-protocol/src/lib.rs:56` behind that feature. The generated file has no `std` references, so nothing else is expected to change.
- Keep the generated types as the wire types. Both profiles build from the same crate with a Cargo feature, default `compat`, optional `lowlat`. A build with both enabled is a compile error.
- Reuse `encode_request`, `encode_lowlat_request`, `decode_reply`, `decode_lowlat_reply`, `quantize_i16`, `crc16` and the scale array from the generated file. Do not reimplement packing or CRC in firmware code.
- One direction matters here because the protocol README labels it backwards at `protocol/README.md:16` and `:41`. The STM32 owns the sensors and sends requests, 56 bytes compat or 42 lowlat. The Pi runs the model and sends replies, 14 compat or 16 lowlat, carrying pitch, yaw, confidence and health. Fix those two README labels in the protocol owner's lane. The firmware follows the field semantics, not those arrows.

## Buffers and tasks

- Static buffers only. Frame buffers live in the UART task or in `StaticCell`. Request buffers are `REQUEST_SIZE` and `LOWLAT_REQUEST_SIZE`. Reply buffers are `REPLY_SIZE` and `LOWLAT_REPLY_SIZE`. No `Vec`, no heap, no `Box`.
- Split the link into two Embassy tasks. The UART task owns DMA and framing. The control task owns link state: last request seq, consecutive misses, AI latch, current phase and the accepted and rejected counters.
- Pass decoded replies between tasks through `embassy_sync` with a fixed-depth channel over `CriticalSectionRawMutex`. No unbounded queues.

## UART and DMA

- One UART at 2 Mbit/s, 8N1, DMA for both directions. TX sends the request from a static buffer after the write completes. RX receives the reply frame.
- Double buffer RX with two static buffers. Receive into buffer A with DMA while the task validates buffer B, then swap. If the HAL's idle-line receive re-arms with a gap that loses frames at full rate, move to a manual circular DMA setup with two completion callbacks. Decide this from a logic-analyzer capture, not from assumption.
- Frame on the reply magic and the fixed frame size. Validate magic, CRC, phase, version and seq with the protocol crate. A bad frame updates the rejected counter and the miss counter. Noise without magic is discarded, not counted as a miss.
- Seq check: a reply seq must match the seq of the most recent request, wrapping at `u16::MAX`. Stale and duplicate replies are rejected. The protocol documents no retries, so nothing is resent.

## Safety behavior

- Phase and flags are the generated enums and constants, not raw integers. Convert only inside encode and decode. The flight state machine on the STM32 owns the phase.
- BURN always produces a zero pitch and yaw command, whatever the reply says. Zero is also the command on a stale reply, a rejected frame, or a latched AI.
- Three consecutive rejected frames latch AI off. Only an explicit reset from the firmware clears the latch. A valid frame after the latch does not clear it. Export accepted, rejected and latch events through `defmt` and telemetry.
- Feed the independent watchdog from the control task only after it has run a full cycle with a valid current phase. A dead or silent Pi makes guidance stale and forces zero output, then the deadman reset fires if the control loop stops. Do not feed the watchdog from the UART callback.

## Hardware tests

These need the real board and the real Pi. A simulator does not cover them.

- Logic analyzer at 2 Mbit/s: frame lengths 56 plus 14 and 42 plus 16, measured wire time against the 0.25 ms budget.
- Back-to-back frames at full rate for at least an hour with no byte loss. Include idle-line gaps and Pi scheduling jitter.
- Fault injection on the wire: corrupt magic, CRC, version and seq. Check rejection counters and the three-miss latch each time.
- BURN gate with servos connected. Confirm no actuation.
- Unplug the UART and kill the Pi. Confirm zero command, stale detection and watchdog reset, and that the reset reason is logged.
- Cross-language interop: Pi client against firmware on both profiles using the generated golden frames.
- Cold flash, brownout, RAM and flash headroom.

## Owner responsibilities

The firmware owner handles Embassy project setup, Cargo and linker configuration, pin and DMA stream selection, the time-driver timer, exact dependency versions and API updates. They own the flight state machine that sets phase, the servo actuation path, the latch reset policy, the watchdog policy and all hardware test runs. The protocol owner fixes the two README arrow labels.
