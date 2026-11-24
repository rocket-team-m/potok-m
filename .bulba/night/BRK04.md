# BRK04 — final retrain and closed-loop A/B after BRK03

**Date:** 2026-11-24 · **Repo:** `/home/sehaxe/potok-m` · **Base commit:** `6f44a25` (BRK03, local, no push).
**Scope:** exactly six work products — `scripts/train_braking_v2.py` (new),
`ai/out/braking_v2/**`, `scripts/ab_braking_v2.py` (new),
`scripts/ab_report_braking_v2.json/.md` (new), `ai/docs/TRAINING_RESULTS.md`
(two new marked sections that leave all earlier sections intact), and this report.
BRK01/BRK01B/BRK03 source was read-only. No physics constant, gate threshold,
command envelope, confidence threshold or loss weight was changed. No push,
no subagents, no flight.

## Verdict (exact)

**Closed-loop A/B: FAIL — 6 of 11 gates failed, 3 blockers, `win_claimed=false`, exit 1.**
The same verdict stands for the v6 `night_default` and `night_adversarial` arms
(`ABV6.md`) and BRK02's braking arm; BRK03's target fix does not change it.

| gate | result |
|---|---|
| eval01 rmse_ratio | PASS |
| eval01 ece_source | PASS |
| eval01 ece | PASS |
| eval01 provenance | PASS |
| eval01 seed_spread | PASS |
| eval01 closed_loop_improvement | FAIL |
| eval01 oracle_positive_control | FAIL |
| seed 1 absolute gates | FAIL |
| seed 2 absolute gates | FAIL |
| seed 3 absolute gates | FAIL |
| oracle_deployment_and_apogee_positive_control | FAIL |

Blockers, verbatim:

- `EVAL01 closed_loop_improvement failed: seed 1 lateral -13.17% is above -35%; seed 2 lateral -10.99% is above -35%; seed 3 lateral -11.18% is above -35%; seed 3 apogee 14.86% is below 15%`
- `EVAL01 oracle_positive_control failed: seed 2 oracle did not raise apogee; seed 3 oracle did not raise apogee`
- `braking_v2: oracle positive control did not improve both metrics on all seeds`

Flight numbers (20 held-out episodes, pooled; deployment proxy = first DESCENT
tick, zero-command control):

| seed | deployment lateral (m) | apogee (m) | whole-trajectory lateral RMS (m) |
|---:|---:|---:|---:|
| 1 | 1690.76 → 1503.98 | 1973.74 → 2294.88 | 1309.41 → 1136.92 |
| 2 | 1688.42 → 1539.51 | 1972.86 → 2270.66 | 1306.40 → 1162.86 |
| 3 | 1689.19 → 1521.34 | 1983.00 → 2277.60 | 1306.38 → 1160.33 |

Absolute gate: deployment lateral ≤ 1000 m on every seed — all three FAIL by
504–540 m; apogee ≥ 2000 m — PASS. Positive control: seed 1 −40.48 m / +11.69 m
(both improved), seeds 2/3 −37.79/−35.99 m but apogee −2.15/−7.60 m (cap ticks
16661/16661, 16584/16584, 16531/16531). The oracle saturates ±10 m/s² on every
active tick and still fails the two-metric control, exactly as in `AILIMIT.md`.

## Retrain

3 seeds × 100 epochs, CPU, frozen RETRAIN-NIGHT recipe, explicit
`--braking-teacher`, corrected objective slots (`TRUTH_VEL_START=3`,
`TRUTH_VEL_END=6`) and BRK03's persistent window tick cursor, ±10 m/s² head,
no tuning. Held-out RMSE: **8.391135 ± 1.292430 m/s²** (seed 1/2/3:
6.903207 / 9.234743 / 9.035453). Teacher-side gates PASS: rmse_ratio,
ece_source, ece. ECE values used by the gates are the disjoint held-out score;
stage-local ECE remains an in-sample diagnostic.

**Per-seed target histogram (the BRK03 check, boost_gate + active rows + cap
bucket):**

| seed | privileged boost_gate | privileged ok | privileged cap_clamped | privileged active | DAgger cap_clamped |
|---:|---:|---:|---:|---:|---:|
| 1 | 487 | 964 | 0 | 964 | 72727 |
| 2 | 487 | 964 | 0 | 964 | 72727 |
| 3 | 487 | 964 | 0 | 964 | 72727 |

BRK02's privileged arm was 100% boost-gated with **0 active rows**; BRK04 has
**964 active rows per seed** — the inverted gate is fixed in the real retrain,
not only in the BRK03 fixture. The cap bucket is counted but **empty on the
privileged arm**: real v3 windows stay inside ±10 m/s², unlike BRK03's
synthetic high-speed fixture (204/204 cap-clamped). DAgger still saturates
(72727 cap-clamped rows). This is recorded loudly in `summary.json`, the results
doc and the A/B report; no coefficient, envelope or gate was tuned to produce
cap usage. The driver refuses a histogram that has no boost_gate bucket, no
active rows, or no cap bucket at all.

The first retrain attempt was killed by the harness 30-minute wait cap with no
artifacts retained; the shipped run is from HEAD `6f44a25` with the BRK03 fix.

## Counted limitations (not gate edits)

1. Helper input-frame delta: rebuilt from current HEAD (source `d46e38e5…`),
   seed-1 zero-command sidecar bit-identical to the frozen POTOKTRJ trajectory,
   conditioning and labels bit-identical, recorded sensor frames differ by max
   **0.120691**. The delta is counted in provenance; every EVAL01/v6 gate is
   called verbatim from `scripts/ab_v6.py`.
2. Simulator-only binding: the sidecar is emitted by the same simulator that
   flies the episodes, `world_step` is a surrogate, and no scored canopy exists.
   The deployment proxy is not a landing-dispersion measurement.
3. The braking target is privileged simulator truth. This A/B measures the
   distilled student's closed-loop behaviour, not teacher fidelity or flight.

## Provenance

- train sha256 `34b35e0ff67f7eeecbd895b2efde8bebcfa159c1a3e78379da6750338b6930b6`;
  held-out sha256 `fdc1d1400b1c5f111e21f79dead37afa1dd9f058fe457edb78037dc6ebe89807`;
  config sha256 `9c34e71f9d10bcf081c31f15f5332f7bb2fac04dddb28009a27cd30987ab5e91`;
  truth sidecar sha256 `7f25bb69124638ea7542be99f43330fda1c5248513f1971f6bfd125f5f759cb8`.
- trainer sha256 `eda3100f04873ff018a05151f39c7d230c22234f058365e7c15dcdef475535ef`;
  BRK03 reference sha256 `a1bd96135bda602a410e3acc05ab8916896eca0318b069861da6b0c90200faf9`.
- objective sha256 `64400853113e70b60868add49d63b0f313a6c84b6d0a4801ebddee47be628411`;
  version `TRUTH_VEL_START=3,TRUTH_VEL_END=6` at `6f44a25`.
- BRK03 version `BRK03 one-way ACTIVE_PHASES whitelist + persistent per-episode window tick cursor`.

## Verification

- `ruff check scripts/train_braking_v2.py scripts/ab_braking_v2.py` — All checks passed.
- `ai/.venv/bin/python -m pytest -q` in `ai/` — **437 passed in 114.01s**.
- `ai/.venv/bin/python -m pytest -q` at repo root — **88 passed in 9.00s**.
  No test was skipped, weakened or deleted.
- The A/B harness exits 1 by design on FAIL; that is the recorded verdict, not
  a harness error.

## Files

- `scripts/train_braking_v2.py` — frozen-recipe retrain driver with the BRK03
  version guard, per-seed histogram check and provenance.
- `scripts/ab_braking_v2.py` — single-arm closed-loop A/B reusing `ab_v6.py`
  helpers and gates verbatim; refuses an inert BRK02-style target.
- `ai/out/braking_v2/{summary.json,runs/s*/,exports/}` — measurements and exports.
- `scripts/ab_report_braking_v2.json`, `scripts/ab_report_braking_v2.md` — full verdict.
- `ai/docs/TRAINING_RESULTS.md` — two new marked sections (`braking-v2`,
  `braking-v2-ab`); all earlier result sections left intact.

**Report convention:** like BRK02, this report is committed with the work
products; no push was made.
