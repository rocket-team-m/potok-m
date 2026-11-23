# B02 — AI watchdog/guard — implementation report

## Scope

Repo: `/home/sehaxe/potok-m/ai` (git root: `/home/sehaxe/potok-m`). Only the assigned
paths were written:

- `ai/src/potok_ai/watchguard.py`
- `ai/tests/test_watchguard.py`
- this report (`.superpowers/sdd/rocketcraft-implementation/task-B02-report.md`)

Both source files carry `SPDX-License-Identifier: GPL-3.0-or-later`. The repo moved to
GPL-3.0-or-later in `29bea98`; `ai/README.md` still says MIT, which is stale but not a
file this task owns.

## Interface (for B13/B15 integrators)

```python
from potok_ai.watchguard import GuardConfig, GuardReason, GuardState, guard, reset

decision = guard(command, confidence, state, config)
# decision: GuardDecision(accepted, reason, command, state)
```

- `command`: 1-D array-like, exactly one value per configured channel.
- `confidence`: finite scalar.
- `state`: `GuardState(misses=0, latched=False, last_command=None)` — carry it between
  ticks; `reset()` returns a pristine state and is the only way to clear a latch.
- `config`: `GuardConfig(bounds=((low, high), ...), max_rate=(rate, ...),
  min_confidence=0.5, max_misses=3)` — per-channel units, validated at construction.
- `decision.command` is a tuple: the accepted command, or all zeros on any rejection.
  Nothing is written back into the caller's array or state.

## Decisions

Checks run in a fixed order, so the reason is deterministic:
`latched → non-finite → bounds → confidence → rate-of-change`.

A rejection emits zeros and leaves `last_command` at the last accepted value, so a
spike cannot move the rate reference. Thresholds are inclusive. The first sample has
no reference and skips the rate check. Consecutive misses reset on an accepted
sample; reaching `max_misses` (default 3) latches the assistant off, and a valid
sample after the latch does **not** clear it — that matches the flight rule in
`FIRMWARE_RUST_PLAN.md:37` and `docs/src/content/docs/ai/concept.md:26`.

State, config and result are frozen dataclasses; the guard reads no clock, has no
global state and mutates no argument. Bad command shape and bad config raise
`ValueError` loudly rather than degrading.

Deliberate limits: the protocol's `ai_healthy` flag is not an input here (it is not in
B02's check list) — the integration layer must AND it with `decision.accepted`. Add it
as a `GuardReason` if the integration asks for it.

## Verification

| Command | Result |
|---|---|
| `uv run ruff check src/potok_ai/watchguard.py tests/test_watchguard.py` | `All checks passed!` |
| `uv run ruff check src tests` | `All checks passed!` |
| `uv run pytest -q tests/test_watchguard.py` | `15 passed` |
| `uv run pytest -q` | `145 passed in 13.54s` |
| `git diff --cached --check` | clean (no output) |

The 15 new tests cover: acceptance + purity, non-finite command (nan/±inf) and
confidence, both bounds sides and inclusive edges, confidence threshold, inclusive
rate limit and last-accepted reference, three-miss latch and stickiness, miss reset
before the latch, `max_misses=1`, `reset()`, config/shape failures, and immutability.

## Commit

Local commit only; nothing pushed. Subject: `feat(ai): pure watchguard with miss latch`.
Unrelated work in `ai/out/` and other agents' in-progress files was left untouched and
unstaged.
