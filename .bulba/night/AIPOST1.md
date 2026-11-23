# AIPOST1 — 3-seed closed-loop A/B on the CURRENT four-stage residual+shield exports

**Date:** 2026-11-24 · **Repo:** `/home/sehaxe/potok-m` (HEAD at run time `ec1bbb16405bec268551db979f0fea262ff8e771`, pre-commit).
**Status:** done. Local commit only, no push. No subagents. No physics, gate, weight, threshold, trainer or sim source was edited.

## Scope

Touched exactly three new files plus this report:

- `scripts/ab_v4.py` — closed-loop harness (sha256 `0d4fb47416604883ce64136684910f54e121e34cbedd5ea2f65638ca4da17d5d`).
- `scripts/ab_report_v4.json` / `scripts/ab_report_v4.md` — generated evidence and human-readable report.
- this report.

Flight work files were written outside the repo, under `/tmp/opencode/ab_v4/seed{1,2,3}/`.

This closes AITRAINIDEAL **B3** ("no 3-seed closed-loop A/B on the new default") with a measurement, not with a win. B5's reading is confirmed: the unweakened gates fail.

## What the harness does

- **Arms:** paired zero command (control), `run_episode_with_oracle` / `fin_accel_command` (positive control), and the shipped `ai/out/final_v3_postpump_dagger/exports/teacher_v2_s{1,2,3}.pt` distill exports of the four-stage default (`sft -> privileged -> dagger -> distill`, SHIELD02 residual policy on; verified against `exports/manifest.json`).
- **Physics:** the existing read-only `scripts/ab_control_v3` helper, not rebuilt. Its seed-1 zero-command sidecar is **bit-identical** to the frozen post-FIXPUMP held-out trajectory (`9b1d680d…`, `manifest.json:312-315`), which is the loud check that the corrected dynamics are in the binary.
- **Deployment composition:** `GDN2Student` consumes one 14-feature frame per tick with its own recurrent state; command = `physics_baseline + export correction`, clipped only at the seam's ±10 m/s² bound. The distill target is the applied residual decision (`train_teacher_v2.py:1068-1077`), so no second confidence gate was invented. Commands are emitted only where the paired baseline's `ai_healthy` target (index 3) is true — the sim's own authority/sensor gate (`firmware_dataset.rs:440`) — so boost-phase flight stays identical to the control.
- **Gates, hard-coded and unweakened:** deployment-proxy lateral RMS ≤ 1000 m and apogee ≥ 2000 m on **every** seed; ECE source must be explicitly disjoint; provenance from metrics/artifact SHA-256 digests; plus the verbatim `teacher_gates.py` suite (`rmse_ratio`, `ece_source`, `ece`, `provenance`, `seed_spread`, `closed_loop_improvement`, `oracle_positive_control`) with its −35 % / +15 %, 1.25, 0.05, 3×-spread limits untouched. Deployment proxy = first `DESCENT` tick (`vz<0 ∧ z≥5`, `dynamics.rs:736-757`), radial distance from the pad, pooled RMS over 20 episodes; this is the proxy named in `LATERAL01.md:116-127`.
- **Provenance:** config/held-out/manifest/quality/export/metrics/config-snapshot digests, harness + helper source + helper binary + dynamics hashes, residual gains, stage order, per-seed trajectory hashes, git HEAD. All in `ab_report_v4.json:138+`.

## Run

```
cd /home/sehaxe/potok-m
ai/.venv/bin/python scripts/ab_v4.py
```

CPU-feasible: ~16 s wall for all 15 helper flights, three stateful model sweeps and the gates. Exit code **1** (harness gate FAIL is the designed signal, not a crash). `ruff format --check` and `ruff check` on `scripts/ab_v4.py`: clean.

## Measured result (20 held-out episodes per seed, post-FIXPUMP)

Metres. `dep` = deployment-proxy lateral RMS; `ap` = mean apogee; `lat` = whole-trajectory lateral RMS.

| seed | arm | dep | ap | lat |
|---:|---|---:|---:|---:|
| 1 | zero command | 1690.76 | 1973.74 | 1309.41 |
| 1 | oracle | 1650.28 | 1985.44 | 1292.10 |
| 1 | teacher_v2 | **1611.99** | **2137.89** | 1236.59 |
| 2 | zero command | 1688.42 | 1972.86 | 1306.40 |
| 2 | oracle | 1650.63 | 1970.71 | 1295.89 |
| 2 | teacher_v2 | **1632.11** | **1974.90** | 1294.59 |
| 3 | zero command | 1689.19 | 1983.00 | 1306.38 |
| 3 | oracle | 1653.20 | 1975.40 | 1298.00 |
| 3 | teacher_v2 | **1642.11** | **1965.68** | 1301.23 |

The export does move the vehicle: ~−79 / −56 / −47 m deployment-proxy lateral, and whole-trajectory lateral −5.6 % / −0.9 % / −0.4 %. Post-FIXPUMP the simplest lateral gain is ~1–5 %, not the pre-pump ~32 %.

## Gate verdicts — 7 FAIL, 4 PASS

PASS: `rmse_ratio`, `ece_source`, `ece`, `provenance` (all verbatim EVAL01 gates).
FAIL:

1. `seed 1 absolute gates` — lateral 1611.99 m > 1000 m (apogee 2137.89 m ≥ 2000 m passes).
2. `seed 2 absolute gates` — lateral 1632.11 m > 1000 m; apogee 1974.90 m < 2000 m.
3. `seed 3 absolute gates` — lateral 1642.11 m > 1000 m; apogee 1965.68 m < 2000 m.
4. `seed_spread` — lateral gain 2.37 % does not beat 3× spread 5.18 %; apogee gain 2.52 % vs 3× spread 9.19 %.
5. `closed_loop_improvement` — every seed is far above −35 % lateral and below +15 % apogee.
6. `oracle_positive_control` (EVAL01) — seed 2/3 oracle apogee goes **down** (−2.15 / −7.60 m).
7. `oracle_deployment_and_apogee_positive_control` — same failure on the deployment-proxy metric: lateral improves −40.48 / −37.79 / −35.99 m on all three seeds, but apogee is +11.69 / −2.15 / −7.60 m. The oracle exercised the actuator on every active tick (cap fraction 1.0), so this is not an inert control.

**No win is claimed.** The exported students do not meet either pre-registered flight gate on any seed, and the oracle no longer has post-FIXPUMP apogee authority (`FIXPUMP.md`: the apparent pre-pump authority was the energy-pumping normal force, now removed). The positive-control failure means the instrument cannot currently certify an apogee-improving assist; it does not mean the simulator is broken — the lateral sign and actuator use are intact.

## Reading and honest limits

- The measured miss is consistent with AITRAINIDEAL **B1**: the distill student inherits `g_max = 1.0` while the residual gain (recorded in the report: −11.18 / −11.22 / −11.12) saturates the command envelope on essentially every allowed tick. It was trained against a target it cannot represent; this harness does not attempt to fix that.
- Simulator-only: POTOKTRJ is emitted by the same simulator that flies the episodes. No canopy/deployment event is scored — the proxy is a fin-flight tick, not landing dispersion.
- Commands are replayed by frame index from the paired baseline flight; later frames diverge after an assisted command moves the vehicle, while the interlock schedule stays paired.
- Seeds 2/3 use seed-matched exports on regenerated held-out flights: input robustness, not training reproducibility.
- The harness generator never rebuilt or edited sim sources in this shared checkout; `sim/core/src/aero.rs` and four aero tests remain dirty from another worker and were left alone. The frozen-baseline bit-identity check above is the guard against accidentally measuring that in-flight source.

## Not done (needs owner decision, not a silent relaxation)

- Isolate residual-on/off and stage-order arms (AITRAINIDEAL item 3), and fix the student command envelope / privileged channel before re-running (item 1). Any re-run must reuse these gates verbatim.
- No gate was weakened to make the current exports pass; none should be to make a future one pass without a recorded decision.

## Verification

- `ai/.venv/bin/python scripts/ab_v4.py` → exit 1, verdict FAIL, wrote both report files.
- `sha256sum scripts/ab_v4.py` equals `provenance.harness_sha256` in `ab_report_v4.json` after the final run.
- `ruff check scripts/ab_v4.py` → All checks passed; `ruff format --check scripts/ab_v4.py` → 1 file already formatted.
- No test was skipped or deleted; no source outside the three assigned files and this report was modified by this task.
