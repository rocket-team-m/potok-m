# PRIVILEGED_TEACHER — wiring the v3 truth sidecar into the teacher stages

SPDX-License-Identifier: GPL-3.0-or-later

This document describes the trainer-side privilege boundary after AIFIX2. It is a
data-contract note, not a flight claim and not a physics, gate or weight change.

## The v3 record

`POTOKTRJ` v3 is a 64-byte header plus one `u64` episode index and `frames × 26`
`f32` per episode (`sim/core/src/trajectory.rs:8-30,93-120`). Each tick splits at
a fixed boundary:

| slots | role | consumer |
|---|---|---|
| 0..21 | privileged truth (position, velocity, quaternion, body rate, mass, wind, gust, shear) | teacher only, via `z_priv` |
| 22..25 | reference optimizer's answer (`g_theta_roll`, `g_theta_pitch`, `g_accel_y`, `g_accel_x`) | loss only; authoritative command is `[24], -[25]` |

`read_truth_sidecar()` in `src/potok_ai/privileged_teacher.py` returns the two
halves as separate arrays. No label slot can enter a feature array by accident.
The split is pinned again in `train_teacher_v2.py` as `TRUTH_PRIV_END = 22`,
`TRUTH_LABEL_START = 22`, `TRUTH_V3_STRIDE = 26`.

## What was broken

Before AIFIX2 the default runs had `truth_sidecar = null`, so
`_build_privileged()` returned `None`, `_stage_privileged()` counted zero
privileged batches and still reported `privileged_source = "privileged_teacher"`.
The DAgger rollout built `privileged = concatenate(zeros(22), truth4)` from the
recorded target: wind, gust, shear and state were all zero and the "expert"
label was the same command the SFT stage already saw. There was no privileged
teacher and no covariate-shift correction (AITRAINIDEAL B2).

## What is wired now

`--truth-sidecar <POTOKTRJ v3 file>` is consumed in two places.

1. **Privileged stage.** `_build_privileged()` raises if a sidecar is configured
   but the builder module is unavailable. The stage builds the seeded
   `PrivilegedFeatureBuilder`, fits the normalizer on that sidecar, and supervises
   a small head on the episode-pooled `z_priv` from slots 0..21. If a sidecar is
   configured and no truth row is consumed, the stage raises; details report the
   real `privileged_source`, the sidecar path and `privileged_batches`.
2. **DAgger corrections.** The same builder is passed into `_dagger_rollout()`.
   Each visited state gets the sidecar's real 0..21 truth at the matching episode
   index and tick. `_expert_label()` reads only slots 22..25. The privileged half
   is never taken from episode targets. Without a sidecar the zeroed half is
   reported as `privileged_fields_zeroed = true` and the source is labelled
   `recorded_target(no_truth_sidecar)` — it is no longer sold as a privileged
   teacher.

### Loud failures

- sidecar configured but builder missing: `RuntimeError`;
- sidecar configured but no privileged batch consumed: `AssertionError`;
- snapshot width other than 26: `ValueError`;
- privileged slots 0..21 all zero or non-finite: `AssertionError` / `ValueError`;
- episode index or tick absent from the sidecar: `ValueError`;
- eval window wider than 14 sensor features or conditioning wider than 6:
  `ValueError` in `engine.assert_eval_label_blind()` and `AssertionError` in
  `train_teacher_v2._evaluate()`.

### Eval stays label-blind

The held-out forward receives only the 14 sensor columns and 6 conditioning
values; labels are used after the forward to score error. A guard rejects any
eval tensor grown with privileged or label columns, and a gate asserts that
changing the label leaves the predicted command and confidence logit
bit-identical. No eval scenario (16..19) can enter either stage.

## Honest limits

- No retrain was run and no held-out number changed. Passing a sidecar is a
  recipe change; the shipped default still has `truth_sidecar = null` and still
  executes the zeroed path, now labelled honestly.
- The privileged stage still supervises a pooled `z_priv` latent with MSE from a
  freshly seeded encoder. It is a privileged *target*, not a decision-making
  teacher whose action could be compared with its removal.
- DAgger still rolls a deterministic sliding window over recorded frames and the
  expert is the reference optimizer's answer in the label slots. This is not an
  on-policy student rollout and does not by itself establish covariate-shift
  correction. The R3 falsification (zero the privilege and measure the drop) is
  now possible, not performed.
- The normalizer is fit on every record in the sidecar handed to the builder;
  pass a train-split sidecar so held-out truth is not used for scaling.
- Nothing here claims a flown-truth anchor, an independent verifier, a gate
  result, a flight log, or a correction of the student's `g_max` bound (B1).
