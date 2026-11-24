# ABV5 — closed-loop A/B v5 on the FINAL all-fix exports

**Date:** 2026-11-24 · **Repo:** `/home/sehaxe/potok-m` (HEAD at run time `58421e6518a64537db51549dbc9a87c11936d807`, pre-commit).
**Status:** done. Local commit only, no push. No subagents. No physics, gate, weight, threshold, trainer or sim source was edited.

## Scope

Touched exactly four files:

- `scripts/ab_v5.py` — closed-loop harness (sha256 `98cc203a7452a5bc58a043fae64cfc7665da100ce3959a5b558827ddfb079e90`).
- `scripts/ab_report_v5.json` / `scripts/ab_report_v5.md` — generated evidence and human-readable report.
- this report.

Flight work files were written outside the repo, under `/tmp/opencode/ab_v5/seed{1,2,3}/`.

This runs the A/B that RETRAIN-ALLFIX and AIPOST1 left open: the FINAL all-fix distill exports, on frozen post-FIXPUMP data, with the unweakened AIPOST1 gates. The answer is a measurement, not a win.

## What the harness does

- **Arms:** paired zero command (control), `run_episode_with_oracle` / `fin_accel_command` (positive control), and the shipped `ai/out/final_v3_allfix/exports/teacher_v2_s{1,2,3}.pt` distill exports of the FINAL four-stage default (`sft -> privileged -> dagger -> distill`).
- **Arm identity is enforced before flight:** export SHA-256 vs `exports/manifest.json`; per-seed `config_hash`, `ece_source`, held-out RMSE vs the manifest; `allfix_provenance = true`; `student_command_bound = 10.0`, `legacy_student_bound = false`; `residual_policy.enabled = true` and a residual-policy distill target; AIFIX3 on-policy DAgger rollout (`on_policy_student:gdn2_student`); AIFIX2 `privileged_batches = 2300`; summary `ab_win_claimed = false`, `rlvr_used = false`; shield authority/veto counts; and the summary's held-out/config hashes vs the flown files. A mismatch is a loud refusal, not a substituted export.
- **Physics:** the existing read-only `scripts/ab_control_v3` helper, not rebuilt. Its seed-1 zero-command sidecar is **bit-identical** to the frozen post-FIXPUMP held-out trajectory (`9b1d680d…`, `manifest.json:312-315`), the loud check that the corrected dynamics are in the binary.
- **Deployment composition:** `GDN2Student` consumes one 14-feature frame per tick with its own recurrent state; command = `physics_baseline + export correction`, clipped only at the seam's ±10 m/s² bound. The distill target is the applied residual decision, so no second confidence gate was invented. Commands are emitted only where the paired baseline's `ai_healthy` target (index 3) is true — the sim's own authority/sensor gate — so boost-phase flight stays identical to the control.
- **Gates, hard-coded and unweakened (AIPOST1):** deployment-proxy lateral RMS ≤ 1000 m and apogee ≥ 2000 m on **every** seed; ECE with an explicit disjoint source; provenance from metrics/artifact SHA-256 digests; plus the verbatim `teacher_gates.py` suite (`rmse_ratio` 1.25, `ece_source`, `ece` 0.05, `provenance`, `seed_spread` 3×, `closed_loop_improvement` −35 %/+15 %, `oracle_positive_control`). Deployment proxy = first `DESCENT` tick (`vz<0 ∧ z≥5`), radial distance from the pad, pooled RMS over 20 held-out episodes (scenarios 16–19); this is the proxy named in `LATERAL01.md:116-127`.
- **Provenance:** config/held-out/manifest/quality/export/metrics/config-snapshot digests, allfix summary + shield digests, disjoint ECE/source, coverage, residual gains, stage order, per-seed trajectory hashes, harness + helper source + helper binary + dynamics hashes, git HEAD. All in `ab_report_v5.json:41+`.

## Run

```
cd /home/sehaxe/potok-m
ai/.venv/bin/python scripts/ab_v5.py
```

CPU-feasible: ~16 s wall for 15 helper flights, three stateful model sweeps and the gates. Exit code **1** (harness gate FAIL is the designed signal, not a crash). `ruff check` and `ruff format --check` on `scripts/ab_v5.py`: clean.

## Measured result (20 held-out episodes per seed, post-FIXPUMP)

Metres. `dep` = deployment-proxy lateral RMS; `ap` = mean apogee; `lat` = whole-trajectory lateral RMS.

| seed | arm | dep | ap | lat |
|---:|---|---:|---:|---:|
| 1 | zero command | 1690.76 | 1973.74 | 1309.41 |
| 1 | oracle | 1650.28 | 1985.44 | 1292.10 |
| 1 | teacher_v2 | **1473.06** | **2334.55** | **1138.45** |
| 2 | zero command | 1688.42 | 1972.86 | 1306.40 |
| 2 | oracle | 1650.63 | 1970.71 | 1295.89 |
| 2 | teacher_v2 | **1558.44** | **2251.14** | **1179.44** |
| 3 | zero command | 1689.19 | 1983.00 | 1306.38 |
| 3 | oracle | 1653.20 | 1975.40 | 1298.00 |
| 3 | teacher_v2 | **1496.79** | **2305.87** | **1145.63** |

Closed-loop change vs the paired zero-command arm (whole-trajectory RMS): lateral **−13.06 % / −9.72 % / −12.31 %**, apogee **+18.28 % / +14.11 % / +16.28 %**. Deployment-proxy lateral changes −12.88 % / −7.70 % / −11.39 %. All three seeds clear 2000 m apogee; none reaches 1000 m deployment lateral.

## Gate verdict — 6 FAIL, 5 PASS

PASS (all verbatim EVAL01 gates): `rmse_ratio`, `ece_source`, `ece`, `provenance`, and — new relative to v4 — **`seed_spread`** (the closed-loop gains now beat 3× the three-seed spread).

FAIL:

1. `seed 1 absolute gates` — deployment lateral 1473.06 m > 1000 m (apogee 2334.55 m ≥ 2000 m passes).
2. `seed 2 absolute gates` — deployment lateral 1558.44 m > 1000 m (apogee 2251.14 m passes).
3. `seed 3 absolute gates` — deployment lateral 1496.79 m > 1000 m (apogee 2305.87 m passes).
4. `closed_loop_improvement` — lateral −13.06 % / −9.72 % / −12.31 % is far above −35 %; seed 2 apogee +14.11 % is below +15 %.
5. `oracle_positive_control` (EVAL01) — the oracle does not raise apogee on seeds 2 (−2.15 m) and 3 (−7.60 m).
6. `oracle_deployment_and_apogee_positive_control` — same failure on the deployment-proxy metric: oracle lateral improves −40.48 / −37.79 / −35.99 m on all seeds, but apogee is +11.69 / −2.15 / −7.60 m. The oracle exercised the actuator on every active tick (cap fraction 1.0), so this is not an inert control.

**No win is claimed.** Every gate must pass; six do not. The final all-fix export is materially closer than the v4 arm (lateral gain ~10–13 % vs ~0.4–5.6 %, apogee clears 2000 m on all seeds, seed spread gate now passes), but it still misses the pre-registered closure gate by a wide margin, and the instrument's own positive control fails the apogee half on two of three seeds. A FAIL is a measured miss, not a weaker gate.

## Reading and honest limits

- The apogee failure of the control is the same post-FIXPUMP finding as AIPOST1: removing the energy-pumping aerodynamic normal force removed the oracle's apparent apogee authority (`FIXPUMP.md`). The student does raise apogee on all three seeds, but the fixed positive-control gate is not satisfied, so this harness cannot certify an apogee-improving assist. No inference about the mechanism is drawn from that asymmetry here.
- All-fix evidence is intact and was checked before flight: `ece_source = "held-out"` on all three seeds, with disjoint ECE 0.0117 / 0.0079 / 0.0108 and coverage 0.935 / 0.955 / 0.964; residual gains −4.86 / −6.23 / −5.85 with the ±10 head (0 clamped rows per RETRAIN-ALLFIX). The ECE and provenance gates pass on the allfix metrics, not on numbers copied from the older run.
- Simulator-only: POTOKTRJ is emitted by the same simulator that flies the episodes. No canopy/deployment event is scored — the proxy is a fin-flight tick, not landing dispersion.
- Commands are replayed by frame index from the paired baseline flight; later frames diverge after an assisted command moves the vehicle, while the interlock schedule stays paired.
- Seeds 2/3 use seed-matched exports on regenerated held-out flights: input robustness, not training reproducibility.
- The helper binary was not rebuilt in this shared checkout; the frozen-baseline bit-identity check above is the guard against accidentally measuring the in-flight simulator source left dirty by another worker (`sim/core/tests/aero*_tests.rs`, `ai/out/v3_final/...`).
- The all-fix changes are wiring/provenance fixes validated on a simulator surrogate (`world_step` and the sidecar share a source). This run adds the missing closed-loop measurement; it does not add an independent flight-truth anchor.

## Not done (needs owner decision, not a silent relaxation)

- No gate was weakened to let the all-fix exports pass, and none should be to let a future one pass without a recorded decision.
- The open items are unchanged in kind: the deployment-lateral gap to 1000 m, the −35 %/+15 % target, the seed-2 apogee threshold, and the oracle's lost apogee authority. Isolating residual-on/off and stage-order arms remains a separate experiment; this harness deliberately runs the shipped FINAL export only.

## Verification

- `ai/.venv/bin/python scripts/ab_v5.py` → exit 1, verdict FAIL, 6 gates failed, 3 blockers, wrote both report files.
- `sha256sum scripts/ab_v5.py` equals `provenance.harness_sha256` in `ab_report_v5.json` after the final run (`98cc203a…`).
- `ruff check scripts/ab_v5.py` → All checks passed; `ruff format --check scripts/ab_v5.py` → 1 file already formatted.
- No test was skipped or deleted; no source outside the four assigned files was modified by this task. Git HEAD in the report is the pre-commit checkout; no remote action was taken.
