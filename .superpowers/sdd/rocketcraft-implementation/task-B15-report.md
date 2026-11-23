# B15 — Edge speculative dispatch (v3) — implementation report

## Scope

Repo: `/home/sehaxe/potok-m-edge`. Files owned by this task:

- `src/speculative_v3.rs` (new)
- `tests/speculative_v3_tests.rs` (new)
- `src/lib.rs` — one line, `pub mod speculative_v3;` (shared file, no other task owns it)

Both new files carry `SPDX-License-Identifier: GPL-3.0-or-later`. No model binary,
no other repo, and no sibling agent's file was modified. Committed locally as
`286bd58`; nothing pushed.

## Implementation

`src/speculative_v3.rs` has two layers.

1. **Pure watchguard** — Rust mirror of `potok_ai.watchguard` (B02), in the
   authoritative v3 command units (m/s², `[g_accel_y, -g_accel_x]`):
   `GuardConfig::new` validates bounds/rate/min_confidence/max_misses loudly;
   `guard(command, confidence, state, &config)` is total and pure with a fixed
   rejection order `latched → non-finite → bounds → confidence → rate`;
   rejections emit zeros and do not move the rate reference; three consecutive
   misses latch, the latch is sticky, and `reset()` is the only clear path.
   `GuardReason` is `ok | non_finite | out_of_bounds | low_confidence |
   rate_limit | latched`.

2. **Dispatch** — `V3Dispatch<S, T>` is generic over `V3Predictor` so it is
   fixture-testable and does not depend on B14's still-in-flight `model_v3.rs`.
   The student runs on every valid cycle; low confidence / invalid sensors and
   the periodic forced check share **one** teacher call site
   (`src/speculative_v3.rs:454`), so low-confidence+forced cycles call the
   teacher once; a `debug_assert` enforces the cap and `V3Counters` exposes it.
   A divergence compares the two heads and calls `student.reset()`. The chosen
   command then passes through `guard` before the `V3Reply` is emitted. A frame
   of the wrong width or carrying a non-finite value bypasses both models and
   the guard. Note: `V3Predictor::reset` has a no-op default for closures;
   stateful implementations must override it — the counter records calls, not
   the effect.

`V3Counters` tracks cycles, invalid cycles, student/teacher calls, low-confidence
and forced triggers, simultaneous triggers, divergences, resets, guard
accepted/rejected, and a per-reason rejection breakdown.

## Interface (for integrators)

```rust
use potok_edge::speculative_v3::{
    guard, reset, DispatchConfig, GuardConfig, V3Dispatch, V3Predictor, V3Source,
};

let mut rt = V3Dispatch::new(student, teacher, DispatchConfig::default())?;
let reply = rt.on_cycle(seq, &sensor_frame, sensors_valid);
// reply: V3Reply { seq, command: [f32; 2], confidence, source, reason, accepted }
```

`DispatchConfig { frame_len, confidence_threshold, forced_check_every,
divergence_factor, guards }`; defaults are 8 / 0.7 / 20 / 3.0 / example guard
bounds ±20 m/s² (the caller supplies calibrated limits). The teacher receives
the last `WINDOW` (50) frames. `on_cycle` never returns a non-zero command with
`accepted == false`.

## Verification

| Command | Result |
|---|---|
| `cargo test --test speculative_v3_tests` | 12 passed, 0 failed |
| `cargo fmt --check` | clean |
| `git diff --check` on staged B15 paths | clean |
| `cargo test` (whole crate) | **not green** — 3 failures in `tests/model_v3_tests.rs`, owned by B14; B15 targets all pass. Do not attribute those failures to this task. |

The 12 tests use deterministic fixture predictors (fixed sequences plus an LCG
replay): guard reasons/purity/reference behaviour, sticky latch and reset, one
teacher call under simultaneous triggers, cadence, low-confidence and invalid
sensor routing, guard screening of the teacher path, divergence reset, latch
integration, malformed-frame bypass, loud config refusal, and bitwise
reproducibility of a 200-cycle replay.

## Shared-file note

`src/lib.rs` also contains B14's uncommitted `pub mod model_v3;` /
`pub use model_v3::{V3Error, V3Model};` lines. Only the `speculative_v3` line was
staged (via `git update-index --cacheinfo`), so B14's in-flight edit remains
uncommitted in the working tree and unclaimed by this commit.
