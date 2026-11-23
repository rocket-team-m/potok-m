# ABFIX — F1 positive-control detector for the post-FIXPUMP A/B

**Date:** 2026-11-23. **Repo:** `/home/sehaxe/potok-m`. **Scope:** `scripts/ab_v3.py`,
`ab_report.json`, `ab_report.md`, this report. No sim source, config, gate, threshold,
weight, export or flight number was changed. No subagents. GPL-3.0-or-later throughout.

---

## 1. Why this fix

ORCLR (`ORCLR.md` §0–§4) found that the post-FIXPUMP oracle's apogee delta is near zero
and sign-mixed: **+11.692 / −2.151 / −7.601 m** on seeds 1/2/3, with both signs inside
each seed's four scenarios (e.g. seed 3: +21.37, −16.33, −8.53, −26.92 m). At ~1973 m
apogee this is 0.1–0.4 %, real but tiny; a uniform sign error would move every scenario
the same way. The same oracle lowers lateral RMS on **all three seeds** and saturates
the ±10 m/s² command actuator, so the sign/allocation and the authority are intact.

The old detector (`scripts/ab_v3.py` pre-fix) required `lateral ↓ AND apogee ↑` on every
seed. It passed 1/3, declared the seam broken and voided every student verdict. That
requirement tests a side effect the corrected dynamics no longer produce; it is an
invalid detector, not a measurement of a broken instrument. F1 in `ORCLR.md` §5 is the
smallest fix: redefine the control, keep apogee as a diagnostic.

## 2. What changed

- `scripts/ab_v3.py:65` — `POSITIVE_CONTROL_DEFINITION`: post-FIXPUMP F1 = lateral
  improvement + correct sign + authority use; apogee delta diagnostic only.
- `scripts/ab_v3.py:77` — `AUTHORITY_MAJORITY = 0.5`: a majority of active ticks at the
  hard cap is structural evidence the actuator was exercised. Not fitted to student
  deltas; the measured value is ~100 % on this denominator (§3).
- `scripts/ab_v3.py:179` — `oracle_authority()`: reads the flown oracle's POTOKTRJ v3
  sidecar fields 24–25 (`g_accel_y`, `g_accel_x`), maps the seam order `[ay, -ax]`,
  counts active (nonzero-command) ticks, cap hits, peak command and pitch-axis sign.
  Counts are stored so any reader can recompute another denominator.
- `scripts/ab_v3.py:687` — credibility criteria, per seed:
  - `lateral_improvement`: pooled Δ lateral ≤ −1 m (`MIN_MEANINGFUL_M`, the existing
    f32-sidecar noise floor, unchanged);
  - `sign_correct`: every one of the 12 per-scenario Δ lateral values is negative,
    with the pinned allocation gates as provenance;
  - `authority_use`: cap hits > 0 and ≥ 50 % of active ticks, finite max command.
- `scripts/ab_v3.py:791` — `credit_status()`: states which student arms the now-credible
  control credits. **The student gate itself is unchanged**: lower lateral RMS and
  higher apogee than zero command on every measured seed. Missing exports stay partial.
- `scripts/ab_v3.py:374` — Markdown report: credible-control table, apogee labelled
  diagnostic, and a "Creditable student verdicts" table.

Sign/allocation provenance (external gates, unchanged):
`sim/core/tests/actuator_mapping_tests.rs:206-238`,
`sim/core/tests/control_seam_hard_tests.rs:62-115`,
`sim/core/tests/firmware_dataset_tests.rs:154`.

## 3. Evidence — every post-FIXPUMP number preserved

From the regenerated `ab_report.json` / `ab_report.md` (20 held-out episodes per seed):

| seed | baseline lat / apogee (m) | oracle lat / apogee (m) | Δ lateral (m) | Δ apogee (m, diagnostic) | cap ticks / active ticks | scenario Δ lateral all < 0 |
|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 1309.409 / 1973.743 | 1292.097 / 1985.435 | −17.312 | +11.692 | 16 661 / 16 661 | yes (4/4) |
| 2 | 1306.397 / 1972.859 | 1295.894 / 1970.708 | −10.503 | −2.151 | 16 584 / 16 584 | yes (4/4) |
| 3 | 1306.380 / 1983.003 | 1297.999 / 1975.402 | −8.382 | −7.601 | 16 531 / 16 531 | yes (4/4) |

Peak command is 10.0 m/s² on every seed; pitch-axis command is negative on 100 % of
active ticks. The three criteria and `credible` are all `true` in the JSON.

**Denominator note (no cherry-picking):** the 100 % cap fraction here is over the
oracle's *health-gated active ticks* (16 531–16 661). ORCLR's 93.5 % is the recorded
reference demand over the wider 1002-tick phase-gate window; RECALC01 §3's 82.5–82.8 %
is the flown oracle over that same wider window, which includes gate-open-but-unhealthy
ticks with zero command. All three are the same actuator, different denominators; raw
counts are in `positive_control.per_seed.<seed>.authority` for recomputation.

## 4. Verdict and which student verdicts are now creditable

`verdict.status = PASS`, `exit_code = 0`, `oracle_positive_control = true`, primary
`loopwm`. The earlier apogee-gated FAIL verdict is **superseded**; pre-FIXPUMP PASS
numbers remain retracted, and the hard ≈51 % lateral / +1940 m apogee bounds remain
unattainable bounds, not predictions.

| arm | status | seeds | Δ lateral (m) | Δ apogee (m) |
|---|---|---|---:|---:|
| `loopwm` | **CREDITED (3/3 seeds)** | 1 / 2 / 3 | −42.042 / −41.248 / −42.944 | +104.041 / +94.927 / +100.102 |
| `lite` | **CREDITED (3/3 seeds)** | 1 / 2 / 3 | −42.039 / −36.130 / −42.910 | +104.046 / +92.454 / +100.579 |
| `lrs` | **PARTIAL (2/3 seeds)** | 1 / 2 (s3 export missing, no stand-in) | −152.145 / −141.004 | +330.935 / +296.717 |

The oracle being a credible control only says the seam can detect an improvement; it
does not make the oracle a good teacher — the students beat it on both metrics. F2
(repair the oracle control law) remains open and was not attempted.

The only remaining blocker is the missing `lrs` seed-3 export; it is partial, never a
fabricated third seed.

## 5. Verification

- `ai/.venv/bin/python -m py_compile scripts/ab_v3.py` → exit 0.
- `ai/.venv/bin/python -m ruff check scripts/ab_v3.py` → `All checks passed!`.
- `ai/.venv/bin/python scripts/ab_v3.py` → reran all 3 seeds and all arms; exit 0,
  `verdict: PASS`, `oracle_ok=True`, `blockers=1` (the known lrs s3 export).
- Compared against the pre-change report: **every** prior `seeds.*` numeric field is
  identical; the only added key is `oracle.authority`. Held-out indices, dynamics
  basis, config hash, baseline hashes and all eight export sha256 values unchanged.

Runnable check of the regenerated artifact:

```sh
cd /home/sehaxe/potok-m && ai/.venv/bin/python - <<'PY'
import json
r = json.load(open("ab_report.json"))
assert r["verdict"]["status"] == "PASS" and r["exit_code"] == 0
assert r["positive_control"]["credible"] is True
assert all(r["positive_control"]["criteria"].values())
expected = {"1": (-17.312, 11.692), "2": (-10.503, -2.151), "3": (-8.382, -7.601)}
for seed, (lat, ap) in expected.items():
    pc = r["positive_control"]["per_seed"][seed]
    assert abs(pc["lateral_delta_m"] - lat) < 1e-3
    assert abs(pc["apogee_delta_m_diagnostic"] - ap) < 1e-3
    assert pc["authority"]["cap_fraction"] >= 0.5
    assert all(x < 0 for x in pc["scenario_lateral_deltas_m"])
credit = r["verdict"]["creditable_students"]
assert credit["loopwm"]["status"] == "CREDITED (3/3 seeds)"
assert credit["lite"]["status"] == "CREDITED (3/3 seeds)"
assert credit["lrs"]["status"] == "PARTIAL (2/3 seeds; no stand-in)"
print("ABFIX CHECK OK")
PY
```

## 6. Residual risk / out of scope

Concurrent commits (`5af2a30`, `0c2196b` and follow-ups) updated documentation to the
*old* verdict "FAIL, no student credited". They are authoritative for nothing once
`ab_report.json/.md` is regenerated, but they now contradict it and need their owner's
update: `ai/README.md:7,104,111,122`, `ai/docs/MODEL_CARD.md:134,169`,
`docs/src/content/docs/project/run.md:135`, `site/index.html:211,232,241,276,332`,
`site/fragments/status.html:3`, and the SVG `docs/src/assets/ab-results-v3.svg`.
Not edited here — outside the assigned file list.

`docs/GROUND_STATION.md:463` says "A/B не проводился" about *ground-station commands*,
not the student A/B; no edit needed.

## 7. Files changed by this task

- `scripts/ab_v3.py` — F1 detector definition, authority reader, credibility and
  student-credit fields, honest Markdown. GPL header kept.
- `ab_report.json`, `ab_report.md` — regenerated by the unmodified flight inputs;
  all post-FIXPUMP numbers retained.
- `.bulba/night/ABFIX.md` — this report.
