# Ideal data-generating process for a robust gust-assist model

**Status:** research note; recommendations, not shipped changes.
**Scope:** `potok-sim` validation v3 (`POTOKSIM`, `version = 3`) and the trainer in
`ai/src/potok_ai/`. Every external claim below names a source that was checked
against the primary record (arXiv metadata/abstract, DOI registration, or the
eCFR text). Claims that are our inference are marked as such. No source ID in
this document was invented.

SPDX-License-Identifier: GPL-3.0-or-later

---

## 1. Bottom line

The evidence supports a **four-part generative process**, in this order of
payoff:

1. **Two-track disturbance envelope** — a deterministic 1-cosine gust family
   for transients *plus* continuous von Kármán/Dryden-type turbulence for the
   stationary background, matched on mean speed, turbulence intensity and
   integral scale. Both tracks are required by the airworthiness standard and
   by every gust-response study checked; neither alone covers the other.
2. **Scenario-cell domain randomization** over wind, vehicle and sensor
   parameters, covering the full deployment envelope, with a *coupled* severity
   axis for compound evaluation **and** isolated one-knob sweeps for
   attribution.
3. **Privileged teacher → sensor-only student** (RMA line): the truth record is
   valid only in the loss/latent target; the student sees the 14 physical
   features, the 6 conditioning numbers and a short history window.
4. **Cumulative curriculum with a calm replay share**, and a **held-out split
   by scenario cell**, reserved strictly for the final report; model selection
   by leave-one-scenario-out on training cells, never on the withheld cells.

A negative result is equally important: in the only cross-fidelity controlled
study found, the *structural fidelity* of the wind field (coherent structures,
LES) was at most a third-order variable, while actuator/sensor identification
and envelope coverage dominated. **Do not fund a high-fidelity turbulence
upgrade as a robustness fix**; fund it as an estimator benchmark. [_Huang 2026_]

---

## 2. Evidence base (verified)

| # | Source | Verification | What it establishes |
|---|---|---|---|
| S1 | von Kármán, T., "Progress in the Statistical Theory of Turbulence", *PNAS* 34(11), 530–539 (1948), DOI `10.1073/pnas.34.11.530` | DOI resolved | Origin of the continuous-turbulence spectrum; physically correct −5/3 inertial-range rolloff (as stated in S5). |
| S2 | Dryden, H. L., "A Review of the Statistical Theory of Turbulence", *Quarterly of Applied Mathematics* 1(1), 7–42 (1943), DOI `10.1090/qam/8209` | Listed in S5 references | Rational-filter realization of continuous turbulence; the model family closest to `wind.rs`'s AR(1) shaping filters. |
| S3 | 14 CFR 25.341 "Gust and turbulence loads" (FAA; eCFR current through 2026-11-19) | eCFR full text read | §(a) discrete gust: 1-cosine shape, sweep gust gradient 30–350 ft; §(b) continuous turbulence: PSD with turbulence scale L = 2500 ft. Two-track requirement is regulatory, not stylistic. |
| S4 | MIL-F-8785C, "Flying Qualities of Piloted Airplanes" (1980) | Listed in S5 references | Source of the 1-cosine definition and low-altitude scale/intensity prescriptions used by S5. |
| S5 | X. Huang, "Statistical Turbulence and High-Fidelity Disturbance Fields for Quadrotor Flight Control", **arXiv:2610.06874** (2026) | abstract + §§II, V, VI, references read | Five fidelity levels L0–L4; **full cross-fidelity train×test matrix**; generator verification tolerances; 1-cos domain randomization ranked first in 30 train/test cells; LES/coherent structures bought no measurable robustness; sim-to-real gap lives in vehicle/sensor identification. |
| S6 | A. Chowdhury, P. Joshi, "Curriculum-Based Deep RL for Robust Autonomous Rocket Landing under Atmospheric Uncertainty", **arXiv:2610.38636** (2026) | abstract + §§4.2, 7.2 read | Rocket landing with power-law shear, AR(1) correlated turbulence, finite-duration one-cosine gusts, coupled severity ℓ; weather curriculum `p(ℓ|c)=0.15 δ₀ + 0.85 U(0,c)` with a linear cap ramp over the first half of each stage; explicitly requires held-out evaluation and isolated-disturbance ablations. |
| S7 | A. A. Tasim, W. Sun, "Wind-Aware RL Control of a Small Quadrotor Using Learned Onboard Wind Estimation in Simulated Atmospheric Turbulence", **arXiv:2607.01528** (2026) | abstract + §§4.2, 6.1.1 read | von Kármán spectrum + power-law shear/veer; Latin-hypercube regime sampling with a **disjoint evaluation seed**; frozen wind estimator as the only privileged channel; 48 % tracking-error reduction vs wind-blind PD; graceful degradation on OOD winds outside the training range. |
| S8 | A. Kumar, Z. Fu, D. Pathak, J. Malik, "RMA: Rapid Motor Adaptation for Legged Robots", **arXiv:2107.04034** (2021) | abstract + §III-B read | Privileged configuration `e_t → z_t = μ(e_t)` available in simulation only; base policy trained with `z_t`; adaptation module `φ` estimates `ẑ_t` from a 50-step state/action history (0.5 s at 100 Hz), trained by supervised MSE; **on-policy adaptation data** needed for off-expert trajectories. |
| S9 | J. Tobin et al., "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World", **arXiv:1703.06907** (2017) | abstract verified | Foundational domain-randomization result: sufficient simulator variability makes the real world one more variation. |
| S10 | A. Molchanov et al., "Sim-to-(Multi)-Real: Transfer of Low-Level Robust Control Policies to Multiple Quadrotors", **arXiv:1903.04628** (2019) | abstract verified | Low-level control policies trained in simulation transfer zero-shot to multiple airframes; noise modeling and domain randomization are decisive knobs. |
| S11 | J. Chen et al., "What Matters in Learning A Zero-Shot Sim-to-Real RL Policy for Quadrotor Control? A Comprehensive Study", **arXiv:2412.11764** (2024) | abstract verified | System-identification quality, not the learning algorithm alone, governs zero-shot transfer (cited as decisive also in S5 §VI). |
| S12 | P. Kunapuli et al., "Leveling the Playing Field: Carefully Comparing Classical and Learned Controllers for Quadrotor Trajectory Tracking", **arXiv:2506.17832** (2025) | abstract verified | Honest classical-vs-learned comparison requires equalized conditions; S5's tuned geometric baseline matched or beat learned policies at high wind. |
| S13 | I. Gulrajani, D. Lopez-Paz, "In Search of Lost Domain Generalization" (DomainBed), **arXiv:2007.01434** (2020) | abstract + §3 read | Model selection is part of the learning problem; leave-one-domain-out CV and training-domain validation are legitimate; test-domain ("oracle") selection is not. |
| S14 | P. W. Koh et al., "WILDS: A Benchmark of in-the-Wild Distribution Shifts", **arXiv:2012.07421** (2020) | abstract verified | Distribution shifts are routinely under-represented in benchmarks; evaluation must be designed for the shift, not inherited from IID splits. |
| S15 | M. Arjovsky et al., "Invariant Risk Minimization", **arXiv:1907.02893** (2019) | abstract verified | Learning representations whose optimum is invariant across training distributions is a route to OOD generalization. |
| S16 | Y. Bengio, J. Louradour, R. Collobert, J. Weston, "Curriculum learning", ICML 2009, pp. 41–48, DOI `10.1145/1553374.1553380` | OpenAlex record `W2296073425` | Canonical formulation: order examples from easy to hard for generalization and convergence. |
| S17 | X. Wang, Y. Chen, W. Zhu, "A Survey on Curriculum Learning", **arXiv:2010.13166** (2020) | abstract verified | Curriculum needs both an easy→hard ranking and a pacing function. |
| S18 | S. Narvekar et al., "Curriculum Learning for Reinforcement Learning Domains: A Framework and Survey", **arXiv:2003.04960** (2020) | abstract verified | Task/sample sequencing framework for RL; curricula are a transfer mechanism, not a free win. |
| S19 | R. Portelas et al., "Automatic Curriculum Learning For Deep RL: A Short Survey", **arXiv:2003.04664** (2020) | abstract verified | ACL shapes the training distribution toward the learner's current capacity. |
| S20 | D. Cavaliere, N. Fezans, "Toward Automated Gust Load Alleviation Control Design via Discrete Gust Impulse Filters", *JGCD* 47(4), 697–710 (2024), DOI `10.2514/1.G007762` | Listed in S5 references | Gust-load-alleviation design is built on discrete gust impulse filters — the control-side counterpart of the 1-cos track. |
| S21 | E. Fournier et al., "Prediction of Preliminary Maximum Wing Bending Moments under Discrete Gust", **arXiv:1811.06534** (2018) | abstract verified | Discrete-gust envelope screening is an established load-case methodology. |
| S22 | F. Huvelin et al., "On the validation and use of high-fidelity numerical simulations for gust response analysis", **arXiv:1902.08678** (2019) | abstract verified | CS-25/FAR-25 gust cases need validation against the certification model before high-fidelity tools are trusted. |

Local contracts used for the field mapping: `sim/core/src/validation.rs`,
`sim/core/src/wind.rs`, `sim/core/src/scenario.rs`, `sim/core/src/config.rs`,
`sim/config/default.toml`, `ai/src/potok_ai/data.py`, `ai/src/potok_ai/data_v3.py`,
`ai/src/potok_ai/curriculum.py`, `SPEC_ENGINE_IDEAL.md`, `SPEC_SMART.md`,
`README.md` (all in `/home/sehaxe/potok-m`).

---

## 3. potok-sim v3 field map (the surface every recommendation must use)

**Header (80 bytes):** `magic=POTOKSIM`, `version=3`, `seed:u64`,
`episode_count`, `frames_per_episode`, `feature_count`, `target_count=6`,
`frame_rate_hz=50`, `dt=0.002`, `duration_s=30`, `meta_count` at bytes 48..52
(17 classic, 39 with truth).

**Per episode, metadata (`meta_count` f32 after `u64 index`):**

| Slots | Field | Role for the data process |
|---|---|---|
| 0..8 | `mass_kg, thrust_scale, pitch, yaw, gust_start_s, gust_duration_s, gust_amplitude, v_10, shear_delta` | legacy envelope; `gust_start_s=-1` means no gust |
| 9..14 | `conditioning = mass_kg, area_m2, cd0, static_margin_cal, thrust_peak_n, gyro_noise` | shared vehicle numbers; available to the smart student (20 in_features = 14+6) |
| 15 | `scenario_id` (u16 as f32, 0..19) | cell identity; 0–15 train, 16–19 held-out |
| 16 | `split` (0 train / 1 held-out) | authoritative split flag |
| 17..38 | `TruthRecord`: `wind[3], gust_start_s, gust_amplitude, shear_altitude_m, shear_delta, shear_thickness_m, pos[3], vel[3], quat[4], omega[3], mass_kg` | **privileged** teacher information; present only when `meta_count=39` |

**Per frame:** `feature_count=14` = 8 raw telemetry + `alpha_deg, beta_deg,
q_dyn_kpa, a_lat_resid, drag_est, static_margin_cal`; **6 targets** =
`g_roll, g_pitch, confidence, ai_healthy, g_accel_y, g_accel_x`. The
authoritative fin command is `[g_accel_y, -g_accel_x]` m/s² in `[roll, pitch]`
order; `g_roll/g_pitch` are angle-domain diagnostics only.

**Scenario cell (scenario.rs):** `v10_u, gust_u, shear_u, shear_alt_u` are
fractions of the configured `wind.*_range`; `gust_u==0` means no gust;
`lag_scale` multiplies `sensor.lag_tau`. Held-out cells 16–19 combine all knobs
at their extremes plus `lag_scale` 1.5–2.0 — an intentional compositional OOD
design. Signature = the four fractions.

**Wind generator (wind.rs):** mean `v_10 (h/10)^α · wind_factor`; shear
`shear_delta · ½(1+tanh((h−shear_altitude)/shear_thickness))`; 1-cos gust
`g = ½·gust_amplitude·(1−cos 2πu)`, `u=(t−t0)/duration`, `t0 ∈ [0.08,0.63]·T`;
turbulence = three AR(1) shaping filters
`x_{k+1}=a x_k + σ√(1−a²) w_k`, `a = exp(−dt·v_ref/L)`, scales `[1, 0.8, 0.45]`,
`L = turbulence_scale·(0.6+0.5·scale_i)`, gain `turbulence_sigma·scale_i`;
one `StdRng` per episode seeded from `(run.seed, episode)` — no global RNG.

**Sensor knobs (config/default.toml):** `gyro_noise/bias`, `accel_noise/bias`,
`mag_*`, `baro_*`, `pitot_*`, `lag_tau`, `quant_step`, saturations,
`latency_steps`, `spike_probability/amplitude`, `gyro_arw`,
`gyro_bias_stability` (all non-ideal extras are 0 by default).

---

## 4. Recommendations, each tied to v3 fields

### R1 — Keep the 1-cos gust as the deterministic transient probe; randomize its gradient, not only its amplitude

**Evidence:** S3 §(a) requires the 1-cos shape and a sweep of gust gradient;
S4/S5 define the same family; S5's L1 (1-cos domain randomization) ranked first
in every test column across two platforms; S20/S21 use discrete gusts for
alleviation and envelope screening. S5 warns that 1-cos training alone does not
cover continuous turbulence — keep S2/R2 alongside it.

**Tie to v3:** `meta[5]=gust_start_s` (onset), `meta[6]=gust_duration_s`
(duration), `meta[7]=gust_amplitude`, `TruthRecord.gust_start_s/gust_amplitude`,
`scenario.gust_u`, `wind.gust_duration`, and the authoritative target columns
4–5 (`g_accel_y, g_accel_x`).

**Do:** derive and log `gust_gradient_m ≈ v_ref · gust_duration_s` and sweep it;
today `wind.rs` samples the amplitude fraction and the onset but takes
`gust_duration` from config (default 2.0 s) — gradient coverage is a real gap.
Never train on `g_roll/g_pitch` as a substitute command (README contract).
Suite: gust_u ∈ {0.45, 0.60, 0.75, 0.90, 1.00} across several gradients and
onsets in the first 60 % of the 30 s flight, both signs of lateral direction.

### R2 — Add a von Kármán-shaped continuous-turbulence rung; verify its spectrum; do not buy LES

**Evidence:** S1 (−5/3 inertial range), S2 (rational filters), S3 §(b) (PSD
track), S5 (L2 von Kármán/Dryden; generator verification: variance within
0.3 %, log-slope −1.64…−1.66 vs −5/3, shear within 0.6 %; LES/coherent levels
gave no measurable robustness gain), S7 (von Kármán + power-law shear/veer,
disjoint evaluation seed, graceful OOD decay).

**Tie to v3:** `wind.turbulence_sigma`, `wind.turbulence_scale`, `wind.v_10`,
`meta[8]=shear_delta`, `TruthRecord.wind[3]` / `shear_altitude_m` /
`shear_thickness_m`, and the observable turbulence signatures
`a_lat_resid`, `q_dyn_kpa`, `alpha_deg`/`beta_deg` in the 14 features.

**Do:** keep the present Dryden-like AR(1) field as one rung; add a von
Kármán-spectrum rung with *matched* mean wind, TI = σ/U and integral scale
`L = turbulence_scale` so only the spectral shape changes (S5's comparability
rule). Gate the generator before training: check the −5/3 slope, the variance
against `(TI·U)²`, the shear profile and per-episode seed reproducibility.
Use a disjoint sampling seed for the held-out split (S7). Fund von Kármán as
an OOD stress/estimator benchmark, **not** as an expected robustness win.

### R3 — Privileged teacher, but make the privilege explicit and the student history real

**Evidence:** S8: privileged `e_t → z_t = μ(e_t)` is used by the teacher; the
student estimates `ẑ_t` from a state/action history (`k=50`), and adaptation
data must be collected **on-policy**, otherwise the student only sees expert
trajectories. S7 shows a frozen estimator feeding a controller as the sole
privileged channel (48 % RMSE reduction, perception share growing with wind).
The repository already states the teacher sees true wind/state only in the
loss (README, "LoopWM").

**Tie to v3:** privileged vector = `TruthRecord` slots 17..38 (`wind[3]`,
`gust_start_s`, `gust_amplitude`, `shear_*`, `pos/vel/quat/omega/mass_kg`);
shared = `conditioning` slots 9..14; student-visible = 14 frame features +
6 conditioning + a recent window (the trainer's teacher window is 50 frames
= 1 s at 50 Hz, comparable in spirit to RMA's history).

**Do:** define an explicit latent target `z_t` (encoded true wind/shear) and
train the history-based adapter against it with MSE, on student rollouts, not
only on teacher-perfect traces. Keep `TruthRecord` out of every student input
path and out of `manifest.json` consumers that run on the edge. Falsification:
zero out the privileged channel and confirm the student still runs and its
held-out score drops; a teacher that only works with truth at inference is a
contract violation, not a result.

### R4 — Domain randomization: cover the envelope, couple severity for evaluation, sweep one knob for attribution

**Evidence:** S9/S10 foundational sim-to-real results; S11 and S5 §VI identify
actuator/sensor identification and envelope coverage as the dominant gap;
S6 §7.2 couples perturbations through one severity level and explicitly notes
that this makes marginal effects statistically dependent and that
isolated-disturbance ablations remain necessary.

**Tie to v3:** cell fractions `v10_u, gust_u, shear_u, shear_alt_u, lag_scale`;
conditioning `mass_kg, cd0, static_margin_cal, thrust_peak_n, gyro_noise`;
sensor `lag_tau, latency_steps, spike_probability/amplitude`, saturation and
quantization knobs.

**Do:** keep the 16 training cells as the coupled envelope; add a severity cap
variable that scales all four fractions together for compound evaluation;
add one-knob sweeps around a reference cell (wind only, gust only, shear only,
sensor lag only) so a failure can be attributed. Spend the identification
budget on `gyro_noise`, `lag_tau`, `thrust_peak_n`, `cd0` and actuator mapping,
not on extra flow fidelity. All generators must share the same relative-wind
→ aerodynamics conversion (S5 comparability rule), i.e. do not let the
fidelity flag co-vary with the vehicle model.

### R5 — Curriculum: cumulative stages are right; the stage assignment and pacing are placeholders

**Evidence:** S16 (ordering easy→hard), S17 (ranking + pacing), S18/S19
(curricula as transfer/support shaping, no guaranteed win), S6 §7.2 (15 %
calm point mass retained while `U(0,c)` widens; `c` ramps linearly over the
first half of each stage).

**Tie to v3:** `curriculum.py::STAGES` and `stage_of()` currently use
`scenario_id % 4`; `scenario_id`, `gust_u`, `shear_u`, `lag_scale`, `split` are
the authoritative fields.

**Do:** derive the stage from the scenario *signature*, not `id % 4` (the file
itself calls the mapping a placeholder): `gust_u==0 → calm`, `gust_u>0 →
gust`, `shear_u>0 → shear`, `lag_scale>1 → sensor_fault`. Keep cumulative
eligibility (already implemented). Add the severity cap ramp and a fixed
calm replay share so early competence is not forgotten. Never admit
`split==1` cells to any stage. Report the repo gate
`rmse_held_out / rmse_train <= 1.25` (`SPEC_ENGINE_IDEAL.md`) and state that a
wider support does not imply monotonic improvement (S6).

### R6 — Held-out by scenario cell, reserved for the final report; model selection elsewhere

**Evidence:** S13 (model selection is part of the problem; leave-one-domain-out
CV; test-domain oracle invalid), S14 (evaluate the shift you expect), S15
(invariance across training domains), S7 (disjoint evaluation seed; OOD range
beyond training), S5 (5 seeds × 10 episodes per cell; claims graded against
seed spread, not a single run).

**Tie to v3:** `scenario_id` 0–19, `split` 0/1, `manifest.json` (config hash,
seed, split, scenarios, bounds, sha256), `TruthRecord` for post-hoc OOD
analysis, `conditioning` for vehicle stratification.

**Do:** treat cells 16–19 (all extremes + extra `lag_scale`) as a true withheld
domain: no tuning, no early stopping, no confidence calibration on them.
Choose hyperparameters and checkpoints by leave-one-scenario-out over cells
0–15. Run ≥3 seeds per arm (repo rule) and ≥5 seeds × 10 episodes per reported
cell; print the eval window (`eval_batches × batch × seq_len` bytes) with every
number. Add an explicit OOD ladder outside the training envelope and report
the degradation curve, as S7 does. This closes the documented repository
limitation "confidence calibrated in-sample on held-out; needed a separate
re-split".

### R7 — Gate the generator before gating the model

**Evidence:** S5 §V.1's four-link data-rigor chain: generator vs reference,
vehicle model, cross-level comparability, assessment statistics; S22
(validate high-fidelity gust simulations against the certification model).

**Tie to v3:** `meta[5..7]` (gust onset/duration/amplitude),
`TruthRecord.wind/gust/shear`, `header.seed`, `manifest.json` config hash and
sha256.

**Do:** assert the realized 1-cos shape equals
`½ A (1−cos 2πu)` on `[0,1]`; turbulence variance equals `(TI·U)²` and the
inertial-range log-slope is ≈−5/3; the tanh shear profile matches within
tolerance; the same seed reproduces the same field; `split==1` episodes are
bit-identical under augmentation (already covered by the night gate
`tests/test_augment_gate_hard.py`). A model A/B run on a generator that fails
these gates measures the generator, not the model.

---

## 5. Concrete level plan (mapping S5's ladder onto potok-sim v3)

| Level | Generator | v3 settings | Purpose |
|---|---|---|---|
| L0 | no wind | `v10_u=0, gust_u=0, shear_u=0`, all stochastic severity off | zero-command / nominal reference row |
| L1 | 1-cos gusts | `gust_u>0`, sweep `gust_duration_s` and onset, amplitude from `gust_amplitude_range` | primary domain-randomized workhorse; transient rejection |
| L2 | statistical turbulence | `turbulence_sigma/scale`, Dryden-like AR(1) **and** von Kármán-shaped rung, matched `TI` and `L` | continuous track; certification §(b) counterpart |
| L3 | coherent structures (vortices/gust fronts) | not representable in v3 fields today | **not recommended** for gust-assist robustness; estimator benchmark only |
| L4 | LES fields | not representable in v3 fields today | **not recommended**; S5 found no measurable robustness gain |

Match levels on dimensionless mean speed, TI and `L/d`; keep one relative-wind
→ aerodynamics code path across levels. L3/L4 would require new truth fields
and a new generator — that is a simulator change, not a data recipe, and the
evidence does not justify it for this controller.

---

## 6. What the evidence does *not* support

- **"More realistic turbulence makes the policy robust."** S5 found structural
  fidelity at most third-order and LES/coherent levels bought nothing
  measurable over 1-cos randomization. The residual sim-to-real gap was in
  actuator/sensor models (S5 §VI, S11).
- **"RMA transfers to rockets as-is."** S8 is legged robots; the latent-target
  and on-policy adaptation recipe is a hypothesis here and must be A/B'd.
- **"A curriculum guarantees improvement."** S6 says the schedule controls the
  support of the training distribution, not monotonic performance; S17–S19
  treat curricula as a shaping tool.
- **"Coupled severity gives per-disturbance conclusions."** It does not; S6
  says the perturbations become statistically dependent and asks for isolated
  ablations.
- **"Any current number is flight evidence."** The repository has no flight
  log, the aero coefficients are placeholders by default, ARW/bias-stability
  sensor effects are off by default, and the closed-loop A/B is red against
  zero command (`README.md`). All of the above is a design for *simulation*
  data only.

---

## 7. License and provenance of this note

This document is part of Potok-M and is distributed under
`GPL-3.0-or-later` (ADR-005, `LICENSE`). External sources are cited by their
verified arXiv ID, DOI or regulation section; no other provenance is asserted.
