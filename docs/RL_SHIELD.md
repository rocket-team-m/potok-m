# RL_SHIELD — typed online shield on the RL/env training path

**Status:** implemented and test-covered in `sim/core/src/rl_env.rs` and
`ai/src/potok_ai/engine.py`. This is an **in-simulator training instrument**.
It makes no flight claim and implements no control-barrier certificate.

## Why

`RES-SHIELD.md` §5.1 is the evidence base: putting the safety constraint inside
the action loop, at training **and** deployment, is the only method in that
survey that improves safety, sample efficiency and final return at once
(violating training steps 0.22 % vs 82.8 %; 80 % of final return 7× faster).
Hard gates in the terminal verifier are post-hoc rejection — the violation has
already happened. A shield turns the constraint into a veto on the *applied*
command. The gates stay in the verifier as the audit; the shield does not
replace or retune them.

This implementation is deliberately the smallest typed version of that idea:
a candidate action is verified against three runtime constraints and, on any
violation, replaced by a safe fallback **with a named reason and a count**.
No QP, no CBF, no dynamics model, no new gate value.

## Where it sits

```
training loop
  candidate action / command
      │
      ▼
  Shield::verify / OnlineShield.verify  ── verify(Mach, phase, authority)
      │                                     record decision (always)
      ├── accept ──► candidate reaches simulator / supervision
      └── veto   ──► safe fallback reaches simulator / supervision
                        Rust: Action::ZERO
                        Python: deterministic classical baseline
```

Rust: `RlEnv::step_shielded(candidate, mach)` is the shielded entry point. The
raw `RlEnv::step(action)` is unchanged for bisect and legacy callers; it still
refuses an out-of-box demand loudly. Python: `LRSAdapter.train_step` verifies
the student's candidate command before it is used as the supervised command.
Evaluation and held-out prediction paths are deliberately **not** shielded —
they must measure the model, not the filter.

## Constraints and their sources

| Constraint | Source | Never invented here |
|---|---|---|
| Mach ceiling | Rust: `GateLimits.mach_max` from the truth sidecar. Python: `TrainConfig.shield_mach_max` (default 0.6, the documented certified ceiling). | No aerodynamics in the shield; `mach` is caller-derived. |
| Command authority | Rust: `control::MAX_COMMAND_MPS2`. Python: `CommandEnvelope.accel`. | No clipping: an out-of-box demand is refused, not trimmed. |
| Flight phase | Rust: `Observation.phase` from the environment's own last observation. Python: `ShieldSample.phase` (wire code 0 = Pad/Boost). | `Phase::ai_allowed()` is the existing rule; no new phase policy. |

Non-finite Mach is an explicit `MachUnavailable` veto (fail safe), never a
silently-passing zero. A malformed Mach ceiling is a loud constructor error.

Check priority is fixed for reproducible reasons:
`unverifiable Mach → Mach ceiling → forbidden phase → authority box → accept`.

## Rust API (`rl_env.rs`)

- `ShieldReason` — `MachUnavailable{observed}`, `MachLimit{observed,limit}`,
  `Authority{ay,ax,limit}`, `PhaseForbidden{phase}`; `Display` names each.
- `ShieldSample { mach, phase }` — runtime values the shield verifies against.
- `ShieldDecision { action, reason }` — `intervened()` is true iff vetoed.
- `ShieldCounts` — one slot per reason plus `accepted`; `interventions()` and
  `total()`.
- `Shield::new(GateLimits) -> Result<Self>` — validates the ceiling.
  `verify(&mut self, Action, ShieldSample) -> ShieldDecision` both decides and
  records, so an intervention cannot happen uncounted. `reset_counts()` is
  explicit; counts otherwise survive `RlEnv::reset` (they describe a run).
- `RlEnv::step_shielded(Action, mach)` — shielded step. Counters via
  `shield_counts()`, last veto via `last_shield_decision()`.

## Python API (`engine.py`)

- `ShieldReason(str, Enum)`, `ShieldSample(mach, phase)`,
  `ShieldDecision(command, reason)`, `ShieldCounts` (with `as_dict()`).
- `OnlineShield(mach_max=, authority=).verify(candidate, fallback, sample)`.
  The fallback is the deterministic classical `physics_baseline` (the residual
  is suppressed), matching the "keep the classical controller as fallback"
  recommendation in `RES-SHIELD.md` §6. `verify` is the only exit and records
  every decision.
- `LRSAdapter.build` constructs the shield; `train_step` uses it. A training
  run writes `out/shield.json` with the reason counts. `OFFLINE_MACH` and
  `OFFLINE_PHASE` are the explicit no-evidence sample used by default because
  offline v3 episodes carry no Mach or phase column; a caller with real
  evidence supplies its own `ShieldSample` (as the tests do).

## Verification

```sh
cd sim && cargo test -p potok-sim --test rl_shield_tests --test rl_env_tests
cd ai && .venv/bin/python -m pytest tests/test_shield_training.py -q
```

See `.bulba/night/SHIELD01.md` for the run results and scope.

## Limits / owed

- This is a typed veto, **not** an MPSF/CBF-QP. It does not compute a certified
  safe set, does not prove forward invariance, and has no embedded cost
  measurement. `RES-SHIELD.md` §5.1's sample-efficiency numbers are motivation,
  not a measurement of this code.
- No A/B run was performed; there is no claim that the shield changed BPB,
  return, or violation rate. Doing that requires the equalized protocol in
  `RES-SHIELD.md` §2 (fixed seeds/steps/scenario split/observation access).
- The Mach and phase arms are exercised with caller-supplied samples in tests.
  Offline v3 data has neither column, so the default training sample is inert
  on those two arms by explicit declaration; the authority arm is always live.
- There is no real flight log and no flight claim anywhere in this module.
