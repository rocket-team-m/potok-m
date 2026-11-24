# ABV6 — final closed-loop A/B v6 on the RETRAIN-NIGHT exports

**Date:** 2026-11-24 · **Repo:** `/home/sehaxe/potok-m` (HEAD at run time `0fe6a4e3bd98db9af49b8d91dbc01604760162c4`, pre-commit).
**Status:** done. Local commit only, no push. No subagents. No gate, physics, weight, threshold, trainer or sim source was edited.

## Scope

Touched exactly four files:

- `scripts/ab_v6.py` — two-arm closed-loop harness (sha256 `2da97169fc0f355c2d655f56985e963b1a578eee64bcf715b9fc8a2c2acc2463`).
- `scripts/ab_report_v6.json` / `scripts/ab_report_v6.md` — generated evidence and human-readable report.
- this report.

Flight work files were written outside the repo, under `/tmp/opencode/ab_v6/seed{1,2,3}/`.

This closes the last open item from RETRAIN-NIGHT: a seeded closed-loop A/B on the `night_final` exports with the unweakened pre-registered gates. The answer is a measurement, not a win.

## What the harness does

- **Two arms, each with its own gates and verdict:**
  - `night_default` — `ai/out/night_final/exports/teacher_v2_s{1,2,3}.pt`, the shipped four-stage default distill exports.
  - `night_adversarial` — `ai/out/night_final/adversarial/runs/s{1,2,3}/student.pt`, the opt-in `--adversarial` curriculum retrain (same 3 seeds / 100 epochs, no gate or knob change).
  - Both are the four-stage distill students (`sft -> privileged -> dagger -> distill`) of the current teacher-v2 default, residual policy on, ±10 m/s² head. They are compared to the **paired zero-command control**, never to each other; no arm-to-arm winner is declared.
- **Arm identity is enforced before flight:** default export SHA-256 vs `exports/manifest.json`, per-seed `config_hash`/`ece_source`/held-out RMSE vs that manifest, `allfix`-class evidence (`night_provenance`, AIFIX1 bound, AIFIX2 privileged batches, AIFIX3 on-policy DAgger, SHIELD02 residual target, AICAL disjoint ECE, AIOBJ impact objective), summaries disclaiming a win and RLVR, and — for the adversarial arm — per-stage AICURR enabled/intensity/cap/offered/won records against `adversarial_summary.json`. Mismatches raise a loud refusal; missing files are blockers, never stand-ins.
- **Physics:** the existing read-only `scripts/ab_control_v3` helper, not rebuilt. Its seed-1 zero-command sidecar is **bit-identical** to the frozen post-FIXPUMP held-out trajectory (`9b1d680d…`), the loud check that the corrected dynamics are in the binary.
- **Deployment composition:** `GDN2Student` consumes one 14-feature frame per tick with its own recurrent state; command = `physics_baseline + export correction`, clipped only at the seam's ±10 m/s² bound. The distill target is the applied residual decision, so no second confidence gate was invented. Commands are emitted only where the paired baseline's `ai_healthy` target (index 3) is true — the sim's own authority/sensor gate — so boost-phase flight stays identical to the control.
- **Gates, hard-coded and unweakened (AIPOST1 / EVAL01):** deployment-proxy lateral RMS ≤ 1000 m and apogee ≥ 2000 m on every seed; ECE with an explicit disjoint source; provenance from metrics/artifact SHA-256 digests; plus the verbatim `teacher_gates.py` suite (`rmse_ratio` 1.25, `ece_source`, `ece` 0.05, `provenance`, `seed_spread` 3×, `closed_loop_improvement` −35 % / +15 %, `oracle_positive_control`), plus the oracle lateral-and-apogee deployment-positive-control check. Deployment proxy = first `DESCENT` tick (`vz<0 ∧ z≥5`), pooled radial RMS over 20 held-out episodes (scenarios 16–19).
- **Provenance:** config/held-out/manifest/quality/export/metrics/config-snapshot digests, both summaries, shield, disjoint ECE/source/coverage, residual gains, stage order, per-seed baseline and assisted trajectory hashes, harness + helper source + helper binary + dynamics hashes, and git HEAD.

## Run

```
cd /home/sehaxe/potok-m
ai/.venv/bin/python scripts/ab_v6.py
```

CPU-feasible: ~18 s wall for 12 helper flights (shared zero-command control and oracle, plus one assisted flight per arm per seed), six stateful model sweeps and both gate suites. Exit code **1** (the harness gate FAIL is the designed signal, not a crash). `ruff check` and `ruff format --check` on `scripts/ab_v6.py`: clean.

## Measured result (20 held-out episodes per seed, post-FIXPUMP)

Metres. `dep` = deployment-proxy lateral RMS; `ap` = mean apogee; `lat` = whole-trajectory lateral RMS.

| seed | arm | dep | ap | lat |
|---:|---|---:|---:|---:|
| 1 | zero command | 1690.76 | 1973.74 | 1309.41 |
| 1 | oracle | 1650.28 | 1985.44 | 1292.10 |
| 1 | night_default | 1462.98 | 2366.25 | 1118.55 |
| 1 | night_adversarial | 1458.08 | 2347.99 | 1132.33 |
| 2 | zero command | 1688.42 | 1972.86 | 1306.40 |
| 2 | oracle | 1650.63 | 1970.71 | 1295.89 |
| 2 | night_default | 1560.57 | 2213.94 | 1182.50 |
| 2 | night_adversarial | 1553.58 | 2252.12 | 1179.59 |
| 3 | zero command | 1689.19 | 1983.00 | 1306.38 |
| 3 | oracle | 1653.20 | 1975.40 | 1298.00 |
| 3 | night_default | 1454.26 | 2327.48 | 1129.39 |
| 3 | night_adversarial | 1456.54 | 2308.26 | 1142.84 |

Whole-trajectory lateral change vs the paired zero-command arm: night_default **−14.58 % / −9.48 % / −13.55 %**, night_adversarial **−13.52 % / −9.71 % / −12.52 %**. Apogee change: night_default **+19.89 % / +12.22 % / +17.37 %**, night_adversarial **+18.96 % / +14.16 % / +16.40 %**. Both arms clear 2000 m apogee on all three seeds; neither reaches 1000 m deployment-proxy lateral.

## Exact verdict per arm

**Overall: FAIL (exit 1).** Both arms FAIL; `win_claimed = false` for each and for the run. A win requires every gate of every arm to pass.

### `night_default` — FAIL, 4 PASS / 7 FAIL

PASS: `rmse_ratio`, `ece_source`, `ece`, `provenance` (all verbatim EVAL01 gates).

FAIL (7):

1. `eval01 seed_spread` — lateral gain 12.54 % does not beat 3× spread 5.09 %; apogee gain 16.49 % does not beat 3× spread 7.67 %.
2. `eval01 closed_loop_improvement` — seed 1 lateral −14.58 %, seed 2 lateral −9.48 %, seed 3 lateral −13.55 % all above −35 %; seed 2 apogee +12.22 % below +15 %.
3. `eval01 oracle_positive_control` — the oracle does not raise apogee on seeds 2 (−2.15 m) and 3 (−7.60 m).
4. `seed 1 absolute gates` — deployment lateral 1462.98 m > 1000 m (apogee 2366.25 m passes).
5. `seed 2 absolute gates` — deployment lateral 1560.57 m > 1000 m (apogee 2213.94 m passes).
6. `seed 3 absolute gates` — deployment lateral 1454.26 m > 1000 m (apogee 2327.48 m passes).
7. `oracle_deployment_and_apogee_positive_control` — oracle lateral improves on all seeds (−40.48 / −37.79 / −35.99 m), but apogee is +11.69 / −2.15 / −7.60 m. The oracle exercised the actuator on every active tick (cap fraction 1.0), so this is not an inert control.

### `night_adversarial` — FAIL, 5 PASS / 6 FAIL

PASS: `rmse_ratio`, `ece_source`, `ece`, `provenance`, and `seed_spread` (the closed-loop gains here beat 3× the three-seed spread).

FAIL (6):

1. `eval01 closed_loop_improvement` — seed 1 lateral −13.52 %, seed 2 lateral −9.71 %, seed 3 lateral −12.52 % all above −35 %; seed 2 apogee +14.16 % below +15 %.
2. `eval01 oracle_positive_control` — the oracle does not raise apogee on seeds 2 (−2.15 m) and 3 (−7.60 m).
3. `seed 1 absolute gates` — deployment lateral 1458.08 m > 1000 m (apogee 2347.99 m passes).
4. `seed 2 absolute gates` — deployment lateral 1553.58 m > 1000 m (apogee 2252.12 m passes).
5. `seed 3 absolute gates` — deployment lateral 1456.54 m > 1000 m (apogee 2308.26 m passes).
6. `oracle_deployment_and_apogee_positive_control` — same oracle apogee failure as above on seeds 2/3.

**No win is claimed for either arm.** The adversarial arm is materially closer on one gate (`seed_spread` passes, unlike the default arm) and has slightly better seed-2 apogee, but both arms miss the −35 % / +15 % and 1000 m closure gates by a wide margin, and the instrument's own positive control fails the apogee half on two of three seeds for both arms. A FAIL is a measured miss, not a weaker gate.

## Reading and honest limits

- The apogee failure of the positive control is the same post-FIXPUMP finding as AIPOST1/ABV5: removing the energy-pumping aerodynamic normal force removed the oracle's apparent apogee authority (`FIXPUMP.md`). Both students do raise apogee on all three seeds, but the fixed positive-control gate is not satisfied, so this harness cannot certify an apogee-improving assist. No mechanism is inferred from that asymmetry here.
- The night_final server-side evidence is intact and was checked before flight: disjoint `ece_source = "held-out"` on all seeds (default ECE 0.0231 / 0.0091 / 0.0214; adversarial 0.0198 / 0.0096 / 0.0244), residual gains −5.15 / −6.18 / −5.93 and −5.10 / −6.34 / −6.15, ±10 head, four-stage order, on-policy DAgger and 2300 privileged batches. The ECE and provenance gates pass on these metrics, not on copied numbers.
- Simulator-only: POTOKTRJ is emitted by the same simulator that flies the episodes. No canopy/deployment event is scored — the proxy is a fin-flight tick, not landing dispersion.
- Commands are replayed by frame index from the paired baseline flight; later frames diverge after an assisted command moves the vehicle, while the interlock schedule stays paired.
- Seeds 2/3 use seed-matched exports on regenerated held-out flights: input robustness, not training reproducibility.
- The helper binary was not rebuilt in this shared checkout; the frozen-baseline bit-identity check above is the guard against accidentally measuring the in-flight simulator source left dirty by another worker.
- The two arms are separate retrains; the harness does not claim that one beats the other. Each is scored only against the shared zero-command control under identical unweakened gates.

## Not done (needs owner decision, not a silent relaxation)

- No gate was weakened to let either arm pass, and none should be to let a future one pass without a recorded decision.
- The open items are unchanged in kind from ABV5: the deployment-lateral gap to 1000 m, the −35 % / +15 % target, the seed-2 apogee threshold, and the oracle's lost apogee authority. Isolating residual-on/off and stage-order arms remains a separate experiment.
- The instrument's positive-control failure is a calibration defect in the post-FIXPUMP measurement setup, not evidence that the simulator is broken; repairing it is outside this task's no-tuning scope.

## Verification

- `ai/.venv/bin/python scripts/ab_v6.py` → exit 1, `overall FAIL`, wrote both report files; per-arm verdicts `night_default FAIL` (7 gates failed), `night_adversarial FAIL` (6 gates failed).
- `sha256sum scripts/ab_v6.py` equals `provenance.harness_sha256` in `ab_report_v6.json` after the final run (`2da97169…`).
- `ruff check scripts/ab_v6.py` → All checks passed; `ruff format --check scripts/ab_v6.py` → 1 file already formatted.
- An initial field-name collision (`config_hash` per-seed map shadowing the 64-hex provenance digest) made the provenance gate raise a type error instead of evaluating evidence. The field was renamed `metrics_config_hash` and the harness re-run; both arms now report `eval01 provenance PASS`. No threshold, physics, weight or gate was changed by that fix.
- No test was skipped or deleted; no source outside the four assigned files was modified by this task. Git HEAD in the report is the pre-commit checkout; no remote action was taken.
