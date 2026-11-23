# RECALC01 — post-FIXPUMP teacher-ceiling recalculation and closed-loop A/B rerun

**Date:** 2026-11-23
**Repo:** `/home/sehaxe/potok-m`, final HEAD `5acddab285c48964e8a4d52ed36cdfcef83681b8` (contains FIXPUMP `584b1dc` and drag pin `7ddf324`). The helper was first built at `0e06e24`, then rebuilt at `5acddab` after a concurrent config commit; baseline, oracle, loopwm and lite numbers were identical across both runs.
**Scope:** read-only analysis + A/B scripts and report files only. No physics, config, gate, threshold, weight or test file was edited. No subagents.

---

## 1. Why this recalculation exists

FIXPUMP (`584b1dc`) projected the placeholder aerodynamic normal force onto the plane perpendicular to the relative wind. Before the fix that force was **parallel** to the lateral relative wind and did positive work (`Ẇ_rel = qdyn·A·cna·(v_b0²+v_b1²)/a_z ≥ 0`), so every nonzero angle of attack injected energy. WORLD04's teacher-ceiling estimates were built on measurements taken with that pump active. They are therefore void as statements about the corrected dynamics:

| WORLD04 claim (pre-FIXPUMP) | post-FIXPUMP status |
|---|---|
| oracle reaches apogee 3971–4000 m, +2236 m over baseline | **retracted**: oracle now 1971–1985 m, **+12 m / −2 m / −8 m** over the three seeds |
| realistic lateral ceiling ≈35–40 % cut (RMS 920–990 m) | **retracted**: oracle now cuts **1.32 % / 0.80 % / 0.64 %**; the pump was the source of its apparent authority |
| oracle is effectively the ideal teacher for this hardware | **retracted**: the oracle now **fails its own positive control on 2 of 3 seeds** (apogee decreases) and is beaten by the frozen students on both metrics |
| apogee ceiling +2400…+2700 m under shipped aero | **retracted**: that was the energy-pump ceiling, not a guidance ceiling |

The helper was rebuilt from current HEAD (`cargo build --release --manifest-path scripts/ab_control_v3/Cargo.toml --target-dir scripts/ab_control_v3/target`, exit 0) and the A/B rerun with all 3 seeds against the **v3 exports as they stood at run time** (no export was edited by this task; one `lrs` export was rewritten by another agent mid-session — see §4). This task made no physics, gate, weight or threshold change.

---

## 2. Corrected zero-command baseline

From the regenerated seed{1,2,3} baseline sidecars (POTOKTRJ v3, stride 26, 1500 ticks at 50 Hz, 20 held-out episodes in scenarios 16–19):

| seed | baseline apogee (m) | baseline lateral RMS (m) | pre-FIXPUMP baseline (m) |
|---:|---:|---:|---:|
| 1 | 1973.7 | 1309.4 | 1735.0 / 1528.4 |
| 2 | 1972.9 | 1306.4 | 1728.3 / 1531.0 |
| 3 | 1983.0 | 1306.4 | 1745.4 / 1528.2 |

Both the baseline apogee (+~240 m) and lateral RMS (−~220 m) moved: the pump was inflating lateral spread and suppressing coast apogee even with zero command. The 20 scenario indices and their 6 conditioning values are verified **identical** to the frozen export (max conditioning delta 0.0); only the flown physics changed. The baseline `validation.bin` is correspondingly **not** byte-identical to the pre-FIXPUMP export (seed 1: `fdc1d140…` vs `5848e222…`), so the stale byte-identity assertion in `scripts/ab_v3.py` was replaced with an explicit scenario-identity check plus a guard that fails if the helper is *not* rebuilt.

---

## 3. Recomputed theoretical teacher ceiling

Derived by `scripts/recalc01_ceiling.py` (read-only). Constraints: phase gate forbids AI on `Pad`/`Boost`, last thrust knot 9.95 s → **1002 active ticks (20.04 s)**; command cap **±10 m/s² per axis**; burnout energy ceiling `z_bo + |v_bo|²/(2g)`, `g = 9.80665`.

| seed | gate floor (m) | r₀ @10 s (m) | v_r @10 s (m/s) | freeze bound (m) | max lateral cut (m / %) | z_bo (m) | \|v_bo\| (m/s) | energy ceiling (m) | max apogee gain (m) | oracle cap-hit % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 175.4 | 755.3 | 154.7 | 641.8 | −667.6 / −51.0 | 1190.2 | 231.5 | 3923.0 | +1949.2 | 83.1 |
| 2 | 173.6 | 748.3 | 154.0 | 635.8 | −670.6 / −51.3 | 1185.4 | 230.9 | 3903.0 | +1930.1 | 82.8 |
| 3 | 175.2 | 753.6 | 154.1 | 640.4 | −666.0 / −51.0 | 1194.3 | 231.4 | 3925.3 | +1942.3 | 82.5 |

**How to read these bounds (they are loose, not predictions):**

- **Phase gate** freezes the first 498 ticks: RMS floor **≈174–175 m**. No controller can go below it.
- **Kinematic freeze bound ≈640 m (≈51 % cut)** assumes infinite lateral acceleration to null `v_r ≈ 154 m/s` at gate-open and hold the vehicle there — unphysical, exactly as in WORLD04. It is a hard *impossibility* bound, **not** a recommendation. Achieving anything near it needs a larger cap or an earlier gate (architecture change).
- **Actuator cap**: the oracle reference command now sits at ±10 m/s² on **~83 % of active ticks** (pre-FIXPUMP 36 %). Saturating the cap harder than before buys almost nothing, because with energy-conserving aero the fins cannot sustain dynamic pressure by pumping energy. The cap remains the binding *command* actuator; fin mechanical stop is not the limiting item.
- **Burnout energy ceiling ≈3903–3925 m** gives a hard apogee gain bound of **≈+1930…+1949 m** from the corrected baseline. It is very loose: the vehicle carries large lateral velocity and drag dissipates energy, so a ballistic-all-speed bound cannot be approached by fin guidance.

**Realistic teacher ceiling under corrected dynamics:** guidance gains are **modest**. The privileged oracle achieves only **−8…−17 m lateral (0.6–1.3 %)** and **−8…+12 m apogee (≈0 %)**, and fails its own positive control on seeds 2/3. Frozen students do better: `loopwm`/`lite` reach **−36…−43 m lateral (2.8–3.3 %)** and **+92…+104 m apogee (4.7–5.3 %)**; the current `lrs` export reaches **−141…−152 m (≈11 %)** and **+297…+331 m (≈15 %)**. So the honest post-FIXPUMP statement is: **~1–11 % lateral and ~0–17 % apogee are the realized guidance-scale gains; the ~51 % / +1940 m numbers are hard physical bounds, not attainable teacher performance.** No gain here is large relative to those bounds. Also note the oracle is no longer the ideal teacher here — it is beaten on both metrics by the students under corrected physics.

---

## 4. Closed-loop A/B rerun (3 seeds, current v3 exports)

`ab_report.json` / `ab_report.md` were regenerated with `scripts/ab_v3.py` (post-FIXPUMP label embedded in both). Verdict: **FAIL (exit 1)**. Deltas vs the corrected zero-command baseline:

| seed | arm | Δ lateral (m / %) | Δ apogee (m / %) |
|---:|---|---:|---:|
| 1 | oracle | −17.3 / −1.32 | +11.7 / +0.59 |
| 1 | loopwm / lite | −42.0 / −3.21 | +104.0 / +5.27 |
| 1 | lrs | −152.1 / −11.62 | +330.9 / +16.77 |
| 2 | oracle | −10.5 / −0.80 | **−2.2 / −0.11** |
| 2 | loopwm / lite | −41.2 / −36.1 | +94.9 / +92.5 |
| 2 | lrs | −141.0 / −10.79 | +296.7 / +15.04 |
| 3 | oracle | −8.4 / −0.64 | **−7.6 / −0.38** |
| 3 | loopwm / lite | −42.9 / −42.9 | +100.1 / +100.6 |

**Blockers (unchanged from the instrument, reported honestly):**

1. `lrs` seed 3 export is missing; no stand-in was substituted.
2. **Oracle positive control FAILED on seeds 2 and 3** (apogee decreases). The instrument therefore declares no student verdict valid, and the pre-registered gate returns FAIL.
3. The primary (`loopwm`) numerically lowers lateral RMS on all three seeds, but is **not credited** because the positive control failed. Its seed-1 lateral goes 1309.4 → 1267.4 m and apogee 1973.7 → 2077.8 m (≈3 % / ≈5 %).

The `loopwm`/`lite` students emit at most **2.0 m/s²** of the ±10 cap (0 clipped ticks); the current `lrs` export reaches 3.8 m/s² (0 clipped ticks). Neither uses full authority, so this is not a story about saturating the actuator. No gate, threshold, weight, physics constant or scenario was tuned to produce these numbers.

**Concurrency note.** A separate agent rewrote `ai/out/v3_final/runs/lrs/s1/lrs.bin` during this session (mtime 00:40; sha256 `99df782a78f9a6fa…`). The first A/B pass used the older export (lrs ≈ −41 m / +102 m); the final `ab_report.json` and the numbers above use the current export (lrs −152 m / +331 m on seed 1). All export hashes in the report are authoritative for the numbers quoted. No export was edited by this task.

---

## 5. Bottom line — stated plainly

- After FIXPUMP the teacher ceiling collapses. The oracle's former +2236 m apogee and −493 m lateral gains were **mostly the energy pump**, not control authority.
- The gains that remain are **modest, not large**: oracle ≈0 % (apogee negative on 2 of 3 seeds); `loopwm`/`lite` ≈3 % lateral / ≈5 % apogee; the current `lrs` export reaches ≈11 % lateral / ≈15 % apogee. Even the best is ≈1/4 of the lateral hard bound and ≈1/12 of the apogee hard bound.
- The old WORLD04 "35–40 % lateral / +2400…+2700 m apogee" estimates are **retracted**; the new hard bounds are ≈51 % lateral and ≈+1940 m apogee, both physically unattainable by this actuator under energy-conserving aero.
- The A/B verdict is **FAIL** because the oracle positive control fails on seeds 2/3. That is a property of the corrected physics, not a reason to tune the gate or the exports.

---

## 6. Files changed by this task

- `scripts/ab_v3.py` — post-FIXPUMP label/fields; stale pre-FIXPUMP byte-identity assertion replaced with scenario-identity check; honest blocker/supersedes text. GPL header preserved; no gate/threshold/physics touched.
- `scripts/recalc01_ceiling.py` — new read-only derivation script (GPL-3.0-or-later header).
- `ab_report.json`, `ab_report.md` — regenerated with post-FIXPUMP label and honest deltas.
- `.bulba/night/RECALC01.md` — this report.

No Rust, physics, config, test or export file was modified. GPL licensing unchanged.
