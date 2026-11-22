# BUILD_LOG — potok-sim

Working dir: `/home/sehaxe/potok-m/sim`. All commands use `--offline`
(the registry cache is populated). No GPU jobs, no push, no files outside
`sim/`.

## Commands and results

```
cargo check --workspace --offline     -> Finished, no warnings
cargo test  --workspace --offline     -> 13 passed, 0 failed
cargo run -p potok-cli --offline -- --help  -> option list printed
cargo run -p potok-cli --offline -- --seed 42 --episodes 2 --validation-out .smoke-out
    -> seed=42 episodes=2 mode=parallel / total: frames=3000 errors=0
       validation: .smoke-out/validation.bin (144168 bytes)
cargo bench -p potok-sim --bench throughput --offline -> numbers below
```

## Baseline (before the optimization pass)

`cargo bench -p potok-sim --bench throughput --offline`, 12 episodes per arm,
one warm-up, seed 1337, 1500 frames per episode:

```
sensors=true  episodes/s=   69.67 frames/s=    104499  (12 episodes, 18000 frames, 0.172s)
sensors=false episodes/s=   83.32 frames/s=    124980  (12 episodes, 18000 frames, 0.144s)
```

Caveat: this baseline was taken before the SoA refactor exposed and fixed an
RK4 derivative-index bug (position/velocity derivative slots were swapped).
The per-step work was the same class of algebra, so the throughput is a fair
order-of-magnitude baseline, but the resulting trajectories were not valid
flights; the fix is part of this pass.

## After optimization

```
sensors=true  fast_field=true  episodes/s=   96.35 frames/s=    144521  (12 episodes, 18000 frames, 0.125s)
sensors=false fast_field=true  episodes/s=  124.19 frames/s=    186283  (12 episodes, 18000 frames, 0.097s)
sensors=true  fast_field=false episodes/s=   84.70 frames/s=    127045  (12 episodes, 18000 frames, 0.142s)
```

Change vs baseline: **+38% episodes/s with sensors, +49% without**.

What changed:

- Wind/gust/turbulence field slices are generated once per episode; the hot
  loop copies precomputed gust and turbulence values instead of advancing
  filters and evaluating a cosine per step. RNG draw order is unchanged, so
  seeded output is identical to the streaming generator.
- State is struct-of-arrays (`pos/vel/quat/omega/mass` separate fixed arrays);
  RK4 combines arrays directly. Vehicle thrust direction is normalized once.
- CRC is computed into a 59-byte stack buffer; no heap allocation per frame.
- Atmosphere lookup table (`run.fast_field`, default on) replaces per-call
  analytic `powf`/`exp`; the exact path is kept behind `fast_field = false`.
  `fast_field_matches_exact_analytic_profile` bounds the exported-feature
  relative deviation between the two paths (<1e-2) and re-checks determinism.
- Validation batches use `Simulator::run_episodes_parallel` (rayon); the CLI
  uses it by default when writing a validation set (`--serial` to disable).
  `parallel_episodes_match_serial` asserts bit-identical features.
- `sim.sensors = false` runs the dynamics-only path for the benchmark.

Determinism tests (`same_seed_twice_is_bit_identical`, `different_seeds_differ`,
`parallel_episodes_match_serial`, `fast_field_matches_exact_analytic_profile`)
are green after the pass.

## Canonical protocol integration (56-byte request)

`core` now depends on `potok-protocol` by path and encodes every frame through
`potok_protocol::encode_request`; no layout is duplicated in `sim/`. Simulator
phases map to wire codes 0..=4 (pad/boost→0, coast→1, apogee→2, descent→3,
landed→4). The canonical crate calls those variants Idle/Armed/Flight/
Landing/Fault; the numeric codes are the contract. Flags are the canonical
IMU_OK/BARO_OK/PITOT_OK/AI_READY/MOTOR_ARMED/FAULT bits; gust and shear stay
simulator-only. Pitot is now `i16` Pa on the wire. Also fixed in this pass:
the sensor model read the velocity slot of the RK4 derivative for specific
force; it now reads the acceleration slot.

```
cargo check --workspace --offline    -> clean
cargo test  --workspace --offline    -> 15 passed, 0 failed
  (13 sim + 2 CLI; includes canonical_golden_request_is_byte_identical
   and protocol_phase_codes_match_uart_yaml)
cargo bench -p potok-sim --bench throughput --offline:
  sensors=true  fast_field=true  episodes/s=  107.58 frames/s=  161366
  sensors=false fast_field=true  episodes/s=  131.88 frames/s=  197825
  sensors=true  fast_field=false episodes/s=   92.97 frames/s=  139459
```

Throughput vs the previous 96.35 / 124.19 / 84.70 ep/s: no regression; the
table-driven CRC from `potok-protocol` is slightly faster than the old bitwise
CRC. Determinism, frame-rate/count, no-NaN, gating and parallel-vs-serial tests
remain green.

## Validation header size: final = 80 bytes

The three claims disagreed (README 76, field sum 80, `header_size()` 84).
`validation::HEADER_SIZE = 80` is now the single authority, with a compile-time
assertion that the field sizes sum to it. Both `Header::write_to` and
`read_from` use one `[u8; HEADER_SIZE]` buffer, and `Header::header_size()`
returns the constant. `sim/README.md` says 80 bytes and shows the offsets.

Verification (`cargo test --workspace --offline`, 15/15 green) now asserts:
`HEADER_SIZE == 80`, `Header::header_size() == HEADER_SIZE`, magic at 0..8,
reserved bytes 48..80 zero, and exact file length =
`80 + episodes * (8 + 9*4 + frames*(features+4)*4)`.

Cross-language read check (file generated inside `sim/`, then deleted):
```
cargo run -p potok-cli --offline -- --seed 7 --episodes 2 --validation-out .verify-out
  -> .verify-out/validation.bin, 144168 bytes
ai/.venv/bin/python - <<... import ai/src/potok_ai/data.py; read_validation(...) ...
  -> SIM_HEADER.size = 80
  -> seed 7 episodes 2 frames 1500 features 8 targets 4
  -> episode shapes (1500, 8) (1500, 4)
  -> OK: ai/data.py reads sim validation.bin
```
`data.py` already declares `struct.Struct("<8sIQ4I3f32s")`, i.e. 80 bytes; no
change outside `sim/` was needed.

## Real vehicle TTX defaults

`config/default.toml` now uses the user-stated Potok-M values: mass
12.5–13.5 kg, area 0.00817 m² (D=0.102 m), static margin 0.0765 m (0.75 cal),
accel noise 0.011 m/s² (DELTA 1064 µg), gyro noise 0.0015 rad/s. Source table
and uncertainty notes are in `sim/README.md`.

Two-stage thrust uses the **passport minimum 3.2 kNs total**: 2.2 kNs / 5.5 s /
1.88 kg → 400 N at 0.342 kg/s (RDK-2000) and 1.0 kNs / 3.2 s / 1.0 kg →
312.5 N at 0.3125 kg/s (RDK-1000), both below the passport peak thrusts. At
OpenRocket-derived mass (10.4–11.4 kg), area 0.011 m², thrust curve and Cd table
the nominal run reaches ~276 m/s and ~3116 m apogee, within 1% of OR Sim 9. The gap is documented in the config and README
rather than hidden. Aerodynamic coefficients remain explicitly marked placeholders.

Also derived from the data rather than hardcoded: `Vehicle::dry_mass` is now
lift-off mass minus the table's propellant (~2.88 kg), and angular damping was
recalibrated (capped placeholder) so the vehicle does not weathercock ~60° off
vertical in the first seconds.

```
cargo check --workspace --offline -> clean
cargo test  --workspace --offline -> 17 passed, 0 failed
  new: defaults_match_real_vehicle_ttx, nominal_trajectory_is_plausible
nominal seed 1337: apogee 2359.9 m, max speed 202.7 m/s
cargo bench -p potok-sim --bench throughput --offline:
  sensors=true  fast_field=true  episodes/s=  115.18 frames/s=  172766
  sensors=false fast_field=true  episodes/s=  146.68 frames/s=  220019
  sensors=true  fast_field=false episodes/s=  102.19 frames/s=  153278
```

## Smart-model simulator lane (spec 8894951, Potok-M domain randomization)

Scope change from the owner: no five rocket types. One airframe, randomized
within tolerances, with a held-out **wind/gust scenario** split.

```
cargo check --workspace --offline   -> Finished, no warnings
cargo test  --workspace --offline   -> 25 passed, 0 failed
   (15 sim + 6 smart-sim + 4 CLI; all old tests still green)
cargo run -p potok-cli --offline -- --help -> --split and --physics-features listed
nominal seed 1337, episode 0: apogee 2564.2 m, max speed 204.4 m/s
```

What changed (all inside `sim/`):

- `core/src/config.rs`: new serde-default ranges for `cd0` (0.30–0.40),
  static margin (0.85–1.10 cal), gyro noise (0.0010–0.0020 rad/s), accel
  noise (0.008–0.014 m/s²), wind 0–15 m/s, gust 5–20 m/s, shear 0–8 m/s at
  50–400 m. `config/default.toml` was NOT edited; old configs still parse.
  Added `Conditioning` (6 numbers) and `sample_domain` on an independent
  `DOMA` sub-seed so the legacy INIT/SENS/WIND draw order is unchanged.
- `core/src/scenario.rs` (new): 20 deterministic cells, 16 train + 4 withheld.
  Withheld ids 16–19 combine all envelope extremes with lag ×1.5–2.0; no
  withheld signature occurs in training. `episode_index % 20` selects the cell.
- `core/src/physics.rs` (new): six features computed from the RK4 state,
  atmosphere and measured body specific force. `alpha_deg` follows the SPEC
  formula literally (`atan2(omega_body_z, v_body_x)`); the README notes the
  classical AoA alternative. Appended to `FEATURE_NAMES` and `Frame.physics`
  (wire layout untouched). Frame vector = 8 raw + 6 physics = 14.
- `core/src/wind.rs` / `sensor.rs`: scenario cells set v10/gust/shear and lag;
  sampled noise ranges replace the fixed gyro/accel noise.
- `core/src/validation.rs`: version 2 writes 17 metadata f32 (legacy 9 +
  6 conditioning + scenario id + split flag) and records `meta_count` in
  header bytes 48..52. Reader accepts v1 (9 metadata) and v2; legacy v1
  conditioning is reconstructed from defaults. Header stays 80 bytes.
- `cli/src/main.rs`: `--split all|train|held-out` and `--physics-features`;
  old flag combinations keep working but now emit v2 files.

Caveat for the AI lane: `validation.bin` is now version 2. `ai/src/potok_ai/data.py`
still reads 9 metadata, so it must be updated by its owner before consuming
new files; no files outside `sim/` were touched here.

## OpenRocket alignment (22.11.2026)

Max speed TTX: M=0.5–0.6 ±; OpenRocket Simulation 9 reaches 274 m/s at M=0.82, so the two statements are not yet reconciled.
At the nominal seed the simulator reaches ~276 m/s
and ~3116 m apogee, matching OpenRocket Simulation 9.

## Ideal loop — physics side (SPEC_ENGINE_IDEAL.md)

All changes inside `sim/`; no GPU jobs, no push.

```
cargo check --workspace --offline -> Finished, no warnings
cargo test  --workspace --offline -> 32 passed, 0 failed
  (4 CLI + 15 sim + 6 smart-sim + 7 new ideal-loop; all old tests green)
```

What changed:

- `core/src/aero.rs` (new): 2-D (Mach, alpha) coefficient card reader with
  bilinear interpolation. Out-of-grid is a LOUD `SimError::AeroOutOfGrid`
  (direct call and runtime propagation through `run_episode`), never a clamp.
  Selecting the placeholder path increments a COUNTED marker and prints a
  stderr line. `config/aero_card.toml` is generated from the current
  placeholder coefficients; selecting it reproduces the legacy trajectory
  bit-for-bit (frame bytes, features and apogee all equal).
- `core/src/config.rs`: `[aero] source = "placeholder"|"card"` +
  `card_path`; Allan ARW/bias-stability, per-channel quantization, frame
  latency, extra saturation and seeded spike injection under `[sensor]`;
  `output.truth_metadata`. Every new field serde-defaults to the legacy value.
- `core/src/dynamics.rs` / `physics.rs`: card coefficients are evaluated at
  each RK4 sub-stage (the legacy Mach table varies within a step); the
  placeholder branch is the original expression, unchanged. `cm_delta` adds a
  gimbal control moment, zero in the example card.
- `core/src/validation.rs`: v2 metadata may carry 39 floats instead of 17 —
  the classic 17 plus true wind, gust start/amplitude, shear layer and the
  full 6-DoF state at the last frame tick. `meta_count` is the authority;
  `read_validation` accepts v1 (9), classic v2 (17) and truth v2 (39).
  `sim/README.md` documents the field layout.
- `core/src/sensor.rs`: Allan white-noise density and bias random walk,
  per-channel quantization, whole-frame latency, mag/baro saturation and
  seeded spikes (FAULT bit + AI gating), all off by default so the original
  RNG draw order and test vectors are untouched.
- `core/tests/ideal_loop_tests.rs` (new, 7 tests): grid nodes + bilinear
  midpoint, out-of-grid error (including a run-time narrow card), generated
  card == placeholder and marker counting, truth round-trip with v1/v2 legacy
  readers, seeded determinism with faults, fault gating, Allan effect.

Skipped: CFD convergence grid (needs an external solver), bicubic (bilinear
meets the spec and is exact at nodes), and AI/edge reader changes (outside
`sim/`).

## OpenRocket Simulation 9 parity oracle (22.11.2026)

Exact source: `/home/sehaxe/Downloads/Telegram Desktop/гугугага.ork`
(ZIP → `rocket.ork`; `<simulations><simulation><name>Simulation 9</name>`,
databranch `Маршевая ступень`, `optimumAltitude=3127.362628772149`,
`timeToOptimumAltitude=26.9`). The fixture
`tests/fixtures/or_sim9.csv` holds 55 rows on the 0.5 s grid (nearest OR
datapoint, |Δt| ≤ 0.015 s; final row is the exact 26.9 s apogee point) with
the eight requested columns. Source column indices in the OR `types` list:
time 0, altitude 1, total speed 4, mass 21, thrust 29, drag 31, Mach 50,
axial Cd 35.

Wiring: `sim/tests/` is now a small workspace member (`potok-or-parity`) so
`sim/tests/or_parity_tests.rs` actually runs under `cargo test --offline`;
a bare `sim/tests/*.rs` would be invisible to a virtual workspace.

### Drag law — PASS (uses OR's own cd_axial, A = 0.011)

`F = 0.5·rho·v²·A·cd_axial`, with rho from our atmosphere at the OR altitude:

| t (s) | rho (kg/m³) | v (m/s) | cd_axial | F_calc (N) | F_OR (N) | err |
|---|---|---|---|---|---|---|
| 5  | 1.1850 | 166.805 | 0.402 | 72.898  | 72.169  | 1.01% |
| 10 | 1.0768 | 245.267 | 0.394 | 140.372 | 137.618 | 2.00% |
| 15 | 0.9873 | 134.625 | 0.319 | 31.394  | 30.702  | 2.26% |
| 20 | 0.9399 | 70.364  | 0.300 | 7.678   | 7.504   | 2.32% |

OR's `drag_N` is computed with **cd_axial** (index 35), not the total Cd
(index 32, 0.16–0.24 at those rows): solving the drag law for Cd gives
0.400 / 0.392 / 0.318 / 0.300, matching cd_axial to ~0.5%.

### Our Cd table vs OR cd_axial — DISCREPANCY, not tuned

`sim_cd = (cd0 − 0.30) + lerp(cd_mach, mach)`; episode 0 samples
`cd0 = 0.3204`, nominal is `cd0 = 0.40`.

| t (s) | Mach | OR cd_axial | sim cd0=0.40 | sim cd0=0.3204 (ep 0) |
|---|---|---|---|---|
| 5  | 0.493 | 0.402 | 0.4417 (+9.9%)  | 0.3621 (−9.9%) |
| 10 | 0.733 | 0.394 | 0.4917 (+24.8%) | 0.4120 (+4.6%) |
| 15 | 0.407 | 0.319 | 0.4216 (+32.2%) | 0.3420 (+7.2%) |
| 20 | 0.215 | 0.300 | 0.4108 (+36.9%) | 0.3311 (+10.4%) |

The placeholder `cd_mach` table does not reproduce OR's axial-Cd curve;
at nominal cd0 it is 10–37% high. This is reported, not tuned.

### Episode 0 parity — FAIL (the "<1% OR match" claim does not hold here)

`Config::default()`, seed 1337, episode 0; altitude from `Frame.height`
(no speed field in `Frame`, so per-instant speed is NOT compared — only
`Episode.max_speed_mps` / `apogee_m`; `pitot_dp` is a saturating i16
dynamic pressure, not a velocity):

| t (s) | OR alt (m) | sim alt (m) | err |
|---|---|---|---|
| 5   | 403.240 | 365.000 | 9.48% |
| 10  | 1476.598 | 1398.012 | 5.32% |
| 15  | 2388.280 | 2277.347 | 4.64% |
| 20  | 2891.346 | 2741.033 | 5.20% |
| 25  | 3110.085 | 2921.056 | 6.08% |
| 26.9 | 3127.363 | 2923.009 | 6.53% |

apogee: OR 3127.363 m vs sim 2927.149 m → **6.40%** (limit 2%).
vmax: OR 274.278 m/s (all OR datapoints) vs sim 267.643 m/s → **2.42%**
(limit 2%). Five of six altitude points and both scalar metrics exceed
their limits; `episode0_parity_tolerances` fails deliberately so the
oracle cannot pass while the trajectory is off. The "~276 m/s / ~3116 m,
within 1%" note above was not reproducible on this code state.

`cargo test --offline --workspace --no-fail-fast`: 36 passed, 1 failed
(the single parity gate above). All 33 pre-existing tests (4 CLI + 16 sim
+ 6 smart-sim + 7 ideal-loop) remain green; the 3 other new tests
(fixture sanity, drag law, Cd diagnostic) pass.
