# Task FW09 — software watchdog / deadman feeding policy

**Status:** implemented, host-tested, committed locally (no push).
**Repo:** `/home/sehaxe/potok-m-firmware`, commit `7bd3831` on `main`.
**Assigned paths only:** `firmware/src/watchdog.rs`, `firmware/tests/watchdog_tests.rs`, this report.
**License:** GPL-3.0-or-later; SPDX identifier on both source files.

## Note on "read the existing watchdog module, do not edit it"

`firmware/src/watchdog.rs` already existed as the skeleton (`may_feed`, `Watchdog`
trait, `WATCHDOG_TIMEOUT_US`, IWDG setup). The task also names that path as the
file to create. Resolution: the change is **purely additive** — 199 insertions,
zero deletions, zero modified lines in that file. The original public API and
the `#[cfg(target_os = "none")]` hardware section are byte-for-byte unchanged,
and no other existing file was touched.

## What was built

`firmware/src/watchdog.rs` (hardware-free policy, `no_std`, no allocation):

- `SoftwareDeadman` — watches the **control loop's own cadence**, not the AI
  link. `note_cycle(now_us)` treats a gap greater than
  `CONTROL_PERIOD_US (2_000) + DEADLINE_SLACK_US (250)` as a missed deadline;
  `DEADLINE_LIMIT (3)` consecutive misses trip it. An on-time cycle
  de-escalates to `Healthy` immediately (same recovery rule as the existing
  link `Deadman::note_valid`). Microsecond rollover uses wrapping subtraction.
- `Severity { Healthy, Degraded, Tripped }` — the missed-deadline escalation
  ladder. Feeding is withheld only at `Tripped`; `Degraded` is observable.
- `WatchdogPolicy` — the feeding policy. `on_control_cycle(now_us,
  cycle_complete, phase_valid)` is the **only** feed-arming path; there is no
  UART path to a feed. `feed_decision()`/`may_feed()` require both a complete
  cycle and a valid STM32-owned phase, and a non-tripped deadman.
- **Independent AI latch**: `latch_ai()` / `reset_ai_latch()` and
  `reset_deadman()` are separate controls. Clearing the AI latch never
  disarms the deadline monitor; resetting the monitor never clears the AI
  latch. A latched AI does **not** withhold feeding (the loop must stay alive
  to permit explicit reset) but does force `must_zero_output()`.

`firmware/tests/watchdog_tests.rs` — 7 host tests: pinned cadence constants and
legacy gate truth table, feed arming by the control cycle only, real `step()`
feeding via `MockBoard`, escalation to trip and de-escalation, μs wraparound,
independent resets in both directions, and latched-AI-still-feeds.

## Verification

- `cargo test --offline --test watchdog_tests` — **7 passed, 0 failed**.
- `cargo test --offline --test safety` — **10 passed, 0 failed** (legacy API intact).
- `cargo check --offline` — clean; `rustfmt --check` on both files — clean;
  `git diff --check` — clean.

**Full-suite caveat (honest):** bare `cargo test --offline` is currently **not
green** because `firmware/tests/interlocks_tests.rs::rate_limit_slews_and_throttles_without_a_jump`
fails (`left: 12, right: 11`) in another agent's untracked file, outside this
task and untouched by it. Earlier attitude-test failures were fixed by their
owner during this task. Cargo stops at that failing binary, so my two relevant
binaries were also run directly (above) and both pass.

## Skipped / when to add

- No wiring into `control::step` or `main.rs`: those files belong to other
  owners. `step` still uses the legacy `may_feed`. Add the `WatchdogPolicy`
  call when the control owner integrates it; it needs `now_us`, cycle-complete
  and phase-valid inputs passed at the end of a tick.
- No hardware changes, by design. The existing IWDG setup is preserved but the
  new policy never calls it.
- The AI latch is a policy flag driven by explicit `latch_ai()` /
  `reset_ai_latch()` calls; connecting those to `LinkState`'s latch is the
  control owner's integration task.
