<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
# Typed residual policy (SHIELD02)

Offline, supervised composition rule used by the teacher stages of
`potok-train-teacher-v2`. It is **not** a flight controller, a certified safety
filter, or a claim about real flight.

## Contract

```
command = clamp(baseline + gain * learned_correction, -envelope.accel, +envelope.accel)
```

* `baseline` is the existing classical controller, `physics.physics_baseline`
  (unchanged constants `KP/KD/BETA0/ALPHA0`, no tuning in this work).
* `learned_correction` is the teacher head's two-channel output.
* `gain` is a scalar `nn.Parameter` initialised to **exactly zero**: an
  untrained policy returns the classical baseline bit-for-bit (the jump-start
  property), and gradients reach both `gain` and the correction's producer
  once training moves it off zero.
* `envelope` is the existing frozen `student.CommandEnvelope` (`accel = 10.0
  m/s²`, the simulator's `MAX_COMMAND_MPS2` unit contract). The applied
  correction is the part of the demand that fits inside the envelope given the
  baseline; the final command is never outside it.

`ResidualDecision` carries `command`, the actually applied `correction`, and
one `GateReason` per command row. Gate checks run in a fixed order and the
first matching reason wins:

| Reason | Meaning | Result |
|---|---|---|
| `disabled` | caller passed `enabled=False` | classical baseline unchanged |
| `non_finite` | NaN/inf in the scaled correction or baseline | correction dropped, baseline stands |
| `clamped` | projection at the envelope changed the command | bounded correction applied |
| `zero_gain` | gain is still zero at init | correction is exactly zero |
| `ok` | in-envelope, non-zero correction | applied unchanged |

A histogram of observed reasons is kept as a plain counter and exported in
`metrics.json` under `residual_policy.gate_reasons` (reset with
`ResidualPolicy.reset_reason_counts()`).

## Research provenance

RES-SHIELD (`.bulba/night/RES-SHIELD.md`, 2026-11-24), §5.2 and §6:

* Residual superposition over a classical controller improves sample
  efficiency and final return and is the most deployable small-student form
  (Johannink et al., arXiv:1812.03201; residual visual servo,
  arXiv:2611.19318).
* Jump-starting from the classical guide raised simulated rocket landing
  success from 8 % to 97.4 % over 10⁶ initial conditions
  (Jiang et al., arXiv:2407.15083).
* The safety-filter evidence (arXiv:2410.11671) motivates putting the bound in
  the action loop; the box projection here is the smallest such mechanism and
  gives **no** CBF/CQP certificate.

The cited results are motivations for the architecture, not measurements of
this repo. No A/B against the legacy teacher has been run; no held-out flight
result is implied.

## Configuration and stage integration

`TeacherV2Config.residual_policy` defaults to **True** (the teacher stages).
The policy is attached as a `LoopWM` submodule, so its `gain` is trained by the
stage optimizer and travels with teacher checkpoints/snapshots. It is applied
to the SFT/privileged/DAgger command, to the teacher's distillation target,
and to the evaluation command; the student's own output already carries the
internalised residual and is not gated twice. The RLVR verifier
(`REWARD_FORMULA`, weights, hard gate) is **unchanged**.

`--no-residual-policy` keeps the pre-SHIELD02 teacher exactly:
`final_command(raw_head_correction, physics_baseline, residual=True)` with no
policy module and no gate reasons. Both paths are recorded in `metrics.json`
(`module_sources.residual_policy`, `residual_policy.source`), and the flag is
part of `config_hash`.

Default-on was kept only after the full gate suite stayed green: `pytest` and
ruff on the four changed files. If a future change breaks a gate, flip the
default back to `False` and record why here rather than weakening a gate.

## Explicitly out of scope

* No physics/baseline constant or envelope value was tuned.
* No CBF/QP shield, RAJS horizon annealing, or deployment hook.
* No flight claim: everything runs on CPU fixtures and simulator-derived
  supervised targets. "Safety" here means a deterministic box projection with
  a counted reason, not a certified safe set.
