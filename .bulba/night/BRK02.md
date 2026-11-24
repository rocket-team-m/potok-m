# BRK02 — retrain and closed-loop A/B with the BRK01 braking teacher

**Date:** 2026-11-24 · **Scope:** BRK02 files only (`scripts/train_braking.py`,
`scripts/ab_braking.py`, `ai/out/braking/**`, `ai/docs/TRAINING_RESULTS.md`,
`scripts/ab_report_braking.json/.md`, this report). BRK01/BRK01B source was
treated as read-only. No physics constant, gate threshold, command envelope,
confidence threshold or loss weight was changed. No push, no subagents.

## Verdict (exact)

**Closed-loop A/B: FAIL — 6 of 11 gates failed, 3 blockers, `win_claimed=false`,
exit 1.** No win claim. Both the v6 `night_default` and `night_adversarial`
arms had already failed under the same unweakened gates (`ABV6.md`); BRK02 does
not change that verdict.

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

- `EVAL01 closed_loop_improvement failed: seed 1 lateral -14.32% is above -35%; seed 2 lateral -11.27% is above -35%; seed 3 lateral -10.88% is above -35%; seed 3 apogee 14.93% is below 15%`
- `EVAL01 oracle_positive_control failed: seed 2 oracle did not raise apogee; seed 3 oracle did not raise apogee`
- `braking: oracle positive control did not improve both metrics on all seeds`

Flight numbers (20 held-out episodes, pooled; deployment proxy = first DESCENT
tick, zero-command control):

| seed | deployment lateral (m) | apogee (m) | whole-trajectory lateral RMS (m) |
|---:|---:|---:|---:|
| 1 | 1690.76 → 1476.71 | 1973.74 → 2339.45 | 1309.41 → 1121.87 |
| 2 | 1688.42 → 1544.14 | 1972.86 → 2282.45 | 1306.40 → 1159.11 |
| 3 | 1689.19 → 1536.59 | 1983.00 → 2279.10 | 1306.38 → 1164.23 |

Absolute gate: deployment lateral ≤ 1000 m on every seed — all three FAIL by
477–544 m; apogee ≥ 2000 m — PASS. Positive control: seed 1 deployment −40.48 m
and apogee +11.69 m (both improved); seeds 2/3 deployment −37.79/−35.99 m but
apogee −2.15/−7.60 m — the oracle saturates ±10 m/s² on every active tick yet
fails the two-metric positive control, exactly as in `AILIMIT.md`.

## Retrain

3 seeds × 100 epochs, CPU, frozen RETRAIN-NIGHT recipe, explicit
`--braking-teacher`, ±10 m/s² head, no tuning. Held-out RMSE:
**8.242817 ± 1.822270 m/s²** (seed 1/2/3: 6.142736 / 9.179225 / 9.406489).
Teacher-side gates PASS: rmse_ratio, ece_source, ece. In-sample ECE values are
diagnostics; every ECE used by the gates is the disjoint held-out score.

The first retrain was started before BRK01B landed and was **killed and its
partial artifacts deleted (marked INVALID)**; the shipped run is from HEAD
`f825a50`, with the corrected objective. BRK01B provenance is pinned in
`ai/out/braking/summary.json`:

- objective sha256 `64400853113e70b60868add49d63b0f313a6c84b6d0a4801ebddee47be628411`
- version `TRUTH_VEL_START=3,TRUTH_VEL_END=6`, commit `f825a50912123989d2bc29103fbb649d76cd9e98`
- the driver refuses any other slot pair before an epoch runs.
- trainer sha256 `244010ff7b81ab5ed1a8a86739866e29e1353d4ad1db2897820eb9695267577f`;
  BRK01 reference `68433038e6627ad444ba19225037a04fda752628b0e9a4d03244b1fe2ebe884a`.

## Counted limitations (not gate edits)

1. **BRK01 privileged target is inert.** All three seeds record
   `privileged.reason_counts.boost_gate = 1451`, with 0 `ok`/`cap_clamped`:
   `_window_ticks()` maps a window batch with unique episode indices to
   tick = `window-1`, so every privileged row is before `boost_end_s = 9.95`
   and receives zero demand. The DAgger reference *is* active
   (`boost_gate=42589`, `cap_clamped=72727`, `ok=3724` per seed) and is what the
   distilled student is trained on. This is not a clean end-to-end
   braking-teacher run; it is a counted, recorded defect in BRK01's file, which
   was outside this task's edit scope. No gate, threshold or trainer file was
   patched to hide it.
2. **Rebuilt helper input-frame delta.** The helper was rebuilt from current
   HEAD (source `d46e38e5…`), after later simulator commits. Seed-1 zero-command
   sidecar is bit-identical to the frozen POTOKTRJ trajectory, and conditioning
   and labels are bit-identical to the frozen held-out, but recorded sensor
   frames differ by max **0.120691** (validation.bin sha
   `779069d27e9292eb20c3c404beeb6d25807728b4419628c1ddc79af5f54fd8f7` vs frozen
   `fdc1d1400b1c5f111e21f79dead37afa1dd9f058fe457edb78037dc6ebe89807`). The delta
   is counted in provenance; the flown-truth identity gate remains strict and
   every EVAL01/v6 gate is called verbatim from `scripts/ab_v6.py`.
3. Simulator-only binding: the sidecar is emitted by the same simulator that
   flies the episodes, the `world_step` is a surrogate, and there is no scored
   canopy. The deployment proxy is not a landing-dispersion measurement.

## Files

- `scripts/train_braking.py` — frozen-recipe retrain driver with explicit flag,
  objective-version refusal, per-seed BRK01 evidence checks and provenance.
- `scripts/ab_braking.py` — single-arm closed-loop A/B reusing `ab_v6.py` helpers
  and gates verbatim; writes the JSON/MD reports and the doc verdict section.
- `ai/out/braking/{summary.json,runs/s*/,exports/}` — measurements and exports.
- `scripts/ab_report_braking.json`, `scripts/ab_report_braking.md` — full verdict.
- `ai/docs/TRAINING_RESULTS.md` — two marked sections (`braking`, `braking-ab`);
  all earlier result sections left intact.

## Verification

- `ruff check scripts/train_braking.py scripts/ab_braking.py` — clean.
- `ai/.venv/bin/python -m pytest -q` in `ai/` — 435 passed.
- `ai/.venv/bin/python -m pytest -q` at repo root — 88 passed.
- A/B harness exits 1 by design on FAIL; that is the recorded verdict, not a
  harness error.
