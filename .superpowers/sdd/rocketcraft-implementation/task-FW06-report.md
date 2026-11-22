# Task FW06 — deterministic PWM buzzer scheduler

**Status:** implemented, tested, committed locally (no push).
**Repo:** `/home/sehaxe/potok-m-firmware`.
**Assigned paths only:** `firmware/src/buzzer.rs`, `firmware/tests/buzzer_tests.rs`, this report. No other file was created or edited.

## What was built

A pure, deterministic PWM buzzer scheduler in `firmware/src/buzzer.rs`:

- `Buzzer::tick(Phase) -> Output` is the ISR entry point. It advances one
  scheduler tick (`TICK_MS = 10`) and returns the PWM value to apply now.
  It uses only integer state (`u16` note index/elapsed, `u16` Hz, `u8` duty):
  no floats, no allocation, no wall clock, no hardware access. The module
  contains no `f32`/`f64` token at all — gated by a test.
- **Priority mapping** `mode_for_phase`: `BURN -> Silence` (safety override),
  `DESCENT | LANDED -> Beacon` (priority locator after parachute deployment),
  `COAST | APOGEE -> Melody` (idle only). The beacon preempts the melody
  whenever the phase enters a deployed state; a mode change restarts the
  pattern from its first note so a repeated phase report does not glitch it.
- **Note table**: integer equal-temperament frequencies C4..C6 (A4 = 440 Hz),
  plus `Note::tone` / `Note::rest`. `BEACON` is three piercing C6 beeps with a
  long gap (32 ticks); `MELODY` is a 21-note arpeggio cycle ending on a rest
  (60 ticks). Both loop exactly and deterministically.
- `Output { freq_hz, duty }` carries **no pin number**: the board adapter owns
  the PWM channel. `SILENT` is `{ freq_hz: 0, duty: 0 }`.

The test file includes the module via `#[path = "../src/buzzer.rs"] mod buzzer;`
because task scope forbids editing `lib.rs` to add `pub mod buzzer;`. The module
itself is `no_std`-clean (only `core`-compatible constructs and the `no_std`
`potok_protocol::Phase`), so it can be wired into `lib.rs` by the integrator
without changes. On-target wiring is therefore deliberately left to the owner
of `lib.rs`.

## Verification

- `cargo test -p potok-m-firmware --test buzzer_tests`: **9 passed, 0 failed**.
  Gates cover BURN silence, full phase priority mapping, beacon preemption and
  restart, exact melody/beacon cycle repetition, determinism across two fresh
  instances, integer note table, and the no-float ISR gate.
- `rustfmt --edition 2021 --check` on both files: clean.
- `cargo clippy -p potok-m-firmware --test buzzer_tests -- -D warnings`: clean.
- `git diff --check` on staged files: clean.

Repo-wide `cargo test -p potok-m-firmware` is **not** green for reasons outside
this task: `firmware/tests/calibration_table_tests.rs::interpolation_is_linear_and_exact_at_known_fractions`
fails in another agent's in-progress file (expected `[0, -20, -40]`, got
`[0, 0, 0]`). That file was left untouched. Other agents' untracked files in
`firmware/src/` and `firmware/tests/` were also left untouched.

## Skipped / when to add

- No `pub mod buzzer;` in `lib.rs` and no `main.rs` ISR wiring: both are owned
  by other files and the task scope is the scheduler plus its host gates. Add
  the module declaration and call `tick()` from the 10 ms timer ISR when the
  board adapter / PWM channel is assigned by `TODO(schematic)`.
- No `MOTOR_ARMED` override: the requirement names only silence on `BURN`.
  Add it if the interlock owner wants a second independent mute source.
