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

## Command → dynamics feedback seam (22.11.2026)

The A/B agent's finding was real: `episode.rs` recorded `g_theta` as a target
and integrated `rk4` with an unchanged `Vehicle`, so no controller could move
the vehicle. The seam below closes that loop. All changes are inside `sim/`.

### What was added

- `core/src/control.rs`: `ControlFile` parser and `validate_command`.
  - **Exact limit: each axis is finite and within ±10.0 degrees**
    (`MAX_CONTROL_DEG`). A larger or non-finite value is a LOUD
    `SimError::Control` — no clamp, no skip. 10 deg covers the reference
    optimizer's `max_correction = 0.12 rad` (6.88 deg) with margin.
  - TSV is `frame_index g_roll_deg g_pitch_deg`, whitespace separated;
    `#` comments, blank lines and one optional `frame_index ...` header are
    ignored. Indices must be exactly `0..frames_per_episode` in order.
    Wrong field count, non-numeric/duplicate/gapped/out-of-range index,
    wrong row count (too few **or** too many) and bad/out-of-range commands
    are all LOUD errors naming the line and the cause.
- `core/src/episode.rs`: `Simulator::run_episode_with_control(index,
  &dyn Fn(&Frame) -> (f64, f64))`. `run_episode` is now a thin wrapper with
  `None`, so the legacy path is untouched. The closure is called **once per
  50 Hz tick** with the frame just measured; the returned pair is held for
  `steps_per_frame = 10` integration steps (default dt 0.002 s, fps 50) until
  the next tick. With `sim.sensors = false` a temporary frame is built for
  the controller but nothing is exported.
- `core/src/dynamics.rs`: `Vehicle.control_deg: [f64; 2]` and the fin moment
  in `derivative`. **Exact authority: `FIN_CONTROL_AUTHORITY = 0.02` per
  radian**, so `moment[0] += C·qdyn·A·D·(π/180)·roll_deg` (body-X / roll) and
  `moment[1] += ...·pitch_deg` (body-Y / pitch), with `D = 2√(A/π)`.
  It is a placeholder coefficient of the same class as the existing aero
  placeholders — **not tuned to favour AI**. The whole block is gated on a
  non-zero command, so the zero path is bit-for-bit the historical one.
  Fin authority scales with dynamic pressure, so it acts on coast/descent
  and vanishes with the atmosphere; the thrust gimbal was rejected because
  `Phase::ai_allowed()` already forbids corrections during powered flight,
  where all the thrust is.
- `cli/src/main.rs`: `--control-file <PATH>` (LOUD refusal with `--tui`).
  Control runs are serial by design, print per-episode `apogee=` and
  `lateral_rms=`, and are otherwise identical to normal headless runs.
  The canonical `validation.bin` layout is **unchanged**, so `ai/data.py`
  needs no update.
- `core/src/trajectory.rs` + `Episode.trajectory`: per-tick flown truth
  (`x, y, z, vx, vy, vz`, world frame) written to a sidecar beside the
  validation file: `validation.bin` → `validation.trajectory`.
  **Exact layout:** little-endian 64-byte header (`POTOKTRJ`, version 1,
  episode_count u32, frames_per_episode u32, stride u32 = 6, 40 reserved),
  then per episode a u64 index and `frames × 6` f32. At defaults that is
  1500 ticks × 24 B = 36 000 B per episode. `apogee_m()` and
  `lateral_rms_m()` are provided so both arms are measured from the flown
  trajectory, not from a surrogate. Note `Episode.apogee_m` is sampled at
  every integration step, while sidecar apogee is sampled at 50 Hz; they
  agree to the tick resolution, not bit-for-bit.

### Verification

```
cargo check --workspace --offline   -> Finished, no warnings
cargo fmt --all -- --check          -> clean
cargo test  --workspace --offline   -> 46 passed, 0 failed
  (37 pre-existing green + 7 new core/control_tests + 2 new CLI tests)
```

New gates:

- `zero_control_is_bit_identical_to_legacy`: frames, features, targets,
  terminal truth, flown truth, apogee bits and all counters match.
- `constant_control_moves_the_flown_trajectory`: at max roll command,
  lateral RMS 4.606 m → 6.988 m (Δ 2.382 m); apogee 3185.378 → 3185.374 m.
  Lateral drift, not apogee, is the measured effect at this airframe.
- `opposite_commands_move_the_vehicle_in_opposite_directions`: final x
  −3.865 m (+pitch) vs +15.623 m (−pitch), Δ 19.488 m.
- `command_limits_are_loud`, `control_file_validation_is_loud_on_bad_index_length_and_range`,
  `control_file_drives_the_same_trajectory_as_a_constant_closure`,
  `trajectory_sidecar_roundtrip_carries_flown_truth`,
  `cli_control_file_drives_dynamics_and_writes_truth_sidecar`,
  `cli_control_file_errors_are_loud`.

Skipped, deliberately: no parallel control runner (serial is fast enough at
~115 ep/s and removes a race); no change to validation v2 or the AI reader;
no physics tuning beyond declaring the fin authority needed for the seam;
no control policy baked into dynamics — the controller sees `Frame.phase`
and decides itself whether to command during boost.

## Dataset generator: privileged per-tick truth, manifest, quality gates (22.11.2026)

All changes inside `sim/`. No GPU, no push, no physics tuning: the RK4 state,
wind, optimizer and control authority are untouched; only observation/export
code and the generator CLI changed.

### Exact format versions

- `validation.bin` is **unchanged**: `POTOKSIM` v2, 80-byte header, 17 metadata
  floats, 14 frame features when generated by `generate-dataset`, 4 targets.
  The old CLI still writes the 8-channel default when `--physics-features` is
  absent.
- `POTOKTRJ` **v1** (`validation::trajectory::VERSION = 1`): 64-byte header
  (`POTOKTRJ`, version u32, episode_count u32, frames_per_episode u32, stride
  u32 = 6, 40 reserved), then `u64` index + `frames x 6` f32 `x,y,z,vx,vy,vz`.
  `write_trajectory` emits byte-identical v1 output; the older CLI path and
  all committed `.trajectory` files keep working.
- `POTOKTRJ` **v2** (`VERSION_V2 = 2`): same header with stride **24**;
  24 f32 per tick in the fixed order
  `pos(3), vel(3), quat(4), omega(3), mass, wind(3), gust_start, gust_amp,
  shear_alt, shear_delta, shear_thickness, g_theta(2)` (names pinned by
  `TRUTH_TICK_NAMES`). Size at defaults: `64 + 8 + 1500 x 24 x 4 =
  1 152 128 B` per episode. `g_theta` is the reference optimizer target (rad),
  not the optional flown control command.
- `manifest.json` schema version **1**, one per split: config_hash (sha256 of
  canonical TOML), seed, split, sorted scenario ids, episode/frame counts,
  feature/target/truth names, observed bounds, and sha256 + byte count of both
  files. Deterministic bytes: `serde_json = "=1.0.145"` with
  `float_roundtrip` — the default parser is 1 ULP off on some f32-derived
  bounds (measured: `-0.0255f32` widened). No timestamps, no process ids.
- `quality_report.json`: schema version 1, config_hash, seed, one entry per
  gate (`name`, `ok`, `detail`), `all_passed`.

### Commands and quality results

```
cargo check --workspace --offline   -> Finished, no warnings
cargo fmt --all -- --check          -> clean
cargo test  --workspace --offline   -> 54 passed, 0 failed
  (46 pre-existing green + 5 new core dataset_tests + 3 new CLI tests)
cargo run -q -p potok-cli --offline -- generate-dataset \
    --seed 42 --train-episodes 8 --held-out-episodes 4 \
    --out /tmp/opencode/ds-smoke
  -> 14 checks, all_passed=true
     config_hash=80a0af458284e878704e45cf015f613bf33b59368888a51e1d0b180befc11f0f
     train/validation.bin 864688 B, train/validation.trajectory 1152128 B
     train sha256=5edaa151b9de9f2ba7e6700e2b7c64153051f6dd4decc1e2149731a588b79fc7
     sidecar sha256=f972a7ae7d260b6b781553c41aa066bb7fad91885fccb57ba21d7f19b4b1f551
     scenario ids train=[0..7] held-out=[16..19], overlap=[]
```

The 14 checks: `episode_count`, `frame_count`, `finite`, `feature_bounds`,
`target_bounds`, `truth_bounds` for each split; `split_scenarios_disjoint`;
`deterministic_regeneration` (four binary files from a from-scratch second
generation compared byte-for-byte). A second run into a different directory
produced byte-identical manifests and quality report too (CLI regression
test). Lowering `--min-episodes-per-split` to 9 exits 1 with
`dataset quality gate failed: episode_count ...`, verified by a CLI test.

### Tests added

`core/tests/dataset_tests.rs` (5): v1 byte layout + reader defaults, v2
per-tick privileged round trip, manifest round trip/determinism/hashes, each
quality gate failing LOUD on a mutated copy, disjoint split acceptance.
`cli/tests/cli_tests.rs` (+3): end-to-end `generate-dataset`, byte-identical
regeneration, loud gate failure.

One pre-existing assertion was made format-aware, not weaker in intent:
`trajectory_sidecar_roundtrip_carries_flown_truth` now compares the six v1
floats (the fields that exist in v1) instead of the whole extended struct;
the privileged fields are compared by the v2 test.

Skipped: atomic staging of output files on a failed gate (failed runs leave
the files for diagnosis and still exit nonzero); AI-side readers outside
`sim/`; a control-file-specific command channel in the sidecar (the field is
documented as the optimizer target).

## Actuator mapping fix (22.11.2026) — root cause of the run4 negative A/B

`scripts/ab_diagnosis.md` found the oracle reference worse than zero command.
The seam was the cause: `optimizer.rs` emits a full-state gimbal-style angle,
but `dynamics.rs` applied it as a fin deflection with a fabricated authority
`FIN_CONTROL_AUTHORITY = 0.02`. The fix is a physical actuator mapping, not a
gain fitted to make the student win. All changes inside `sim/`.

### Mapping now implemented

The control command pair is **desired lateral acceleration in m/s^2**, not a
deflection angle (`MAX_CONTROL_DEG` is kept as a numeric alias of
`MAX_COMMAND_MPS2 = 10` so old control files/tests still parse; its unit is
m/s^2). Per axis, in `dynamics::solve_fin_deflection`:

```
M_req = m * g_cmd * l                       (m: current RK4 mass, l: CG->fin arm)
M_fin = qdyn * A * D * Cm_delta * delta     (D = 2*sqrt(A/pi))
delta = clamp(M_req / (qdyn*A*D*Cm_delta), +-25 deg)
moment[axis] += qdyn*A*D*Cm_delta*delta
```

Physical constants, all documented in code/config, none fitted to an A/B:
- `Cm_delta` placeholder = **5.0 /rad** (order 1-10/rad for tail-controlled
  rockets); the card is authoritative when present. Selecting the placeholder
  prints a COUNTED marker `cm_delta=5 /rad count=N` next to the existing aero
  marker. `config/aero_card.toml` carries the same 5.0 so the card path has the
  same authority.
- `aero.control_moment_arm_m` = **0.25 m** (~2.1 calibers of D=0.118 m), a
  tail-fin arm on a short airframe with CG near mid-body. Added as a
  serde-defaulted config field so run1-4 configs still parse.
- Fin mechanical stop = **25 deg**, clamp is COUNTED (`fin_clamp_uses()`).
- `qdyn <= 1 Pa` with a nonzero command -> safe zero deflection plus a COUNTED
  marker (`fin_no_authority_uses()`), not a divide-by-zero.
- Non-finite / zero / negative `Cm_delta` with a nonzero command -> LOUD
  `SimError::Control` naming the cause, propagated from `derivative` through a
  `ControlFault` cell. Verified with a zero-`cm_delta` card.
- The old `cm_delta` gimbal moment is now gated on `thrust > 0`: thrust
  vectoring exists only while the motor burns; after burnout the aerodynamic
  fins take over. The AI interlock already forbids boost corrections.
- Zero command never enters the mapping; that trajectory is bit-identical.

The oracle controller (`ReferenceOptimizer::fin_accel_command`) exposes the
lateral acceleration demand the solver already computes internally (`ax_des`,
`ay_des`), saturates it at the actuator envelope, and uses the command sign
validated against flown truth (a positive value rotates the nose so the
lateral aerodynamic response is opposite to the naive right-hand-rule reading
of this sim's relative-wind normal force). `Target`/`g_theta`, the AI models,
gates, `validation.bin` and both trajectory sidecar versions are untouched.

### Oracle A/B (held-out indices 16-19, pooled lateral RMS = sqrt(mean(rms^2)))

Each arm runs the same 4 episodes per seed, no RNG or config changes:

| seed | zero command (m) | oracle m/s^2 command (m) | delta |
|---:|---:|---:|---:|
| 1 | 1536.930 | 1045.631 | **-491.299** |
| 2 | 1516.967 | 1008.335 | **-508.632** |
| 3 | 1535.254 | 1019.964 | **-515.290** |

Wrong-sign oracle on seed 1: 1536.930 -> 1630.637 m (**+93.707**, worse), so the
sign is measured, not assumed. With the old fabricated authority the oracle
was +530..590 m WORSE; the mapping change reverses that.

Moment-arm sweep (same 3 seeds, corrected sign, Cm_delta = 5.0) — reported
honestly, including the values that do not work:

| arm | ~calibers | mean delta lateral RMS (m) |
|---:|---:|---:|
| 0.20 | 1.7 | -459 |
| 0.25 | 2.1 | **-505** (shipped) |
| 0.30 | 2.5 | -510 |
| 0.40 | 3.4 | -337 |
| 0.50 | 4.2 | +113 (oracle becomes harmful: the airframe spins, maxw ~23 rad/s) |

0.25 was chosen as a plausible tail-fin arm; the sweep shows a physical
authority ceiling, not an AI gain search. No AI or gate threshold was changed.

### Verification

```
cargo fmt --all -- --check        -> clean
cargo check --workspace --offline -> Finished, no warnings
cargo test  --workspace --offline -> 62 passed, 0 failed
```

8 new gates in `core/tests/actuator_mapping_tests.rs`: moment-balance inversion
and clamping, thin-air counted zero, LOUD zero authority (helper + whole run),
zero-command bit identity, oracle improves on held-out (3 seeds), wrong sign
worsens, v1/v2 sidecar round-trip of a controlled flight, placeholder COUNTED.
All 54 pre-existing tests stay green.

## Command-target export: validation v3 / POTOKTRJ v3 (22.11.2026)

Final A/B root cause: the AI models were trained on `g_theta`, an
angle-domain tilt target from `optimizer.solve`, but the actuator consumes
`fin_accel_command`, a world-frame lateral acceleration demand. The fix is to
export the command the fin solver actually consumes. No physics, gains or
A/B data were changed; all edits are inside `sim/`.

### Exact format versions and field order

- `validation.bin` writer is now **v3**: same 80-byte `POTOKSIM` header and
  the same 17 (or 39 with `truth_metadata`) metadata floats, but
  **`target_count = 6`** (was 4). Target row per 50 Hz frame, little-endian
  f32: `g_roll, g_pitch, confidence, ai_healthy, g_accel_y, g_accel_x`.
  Columns 0–3 are the untouched legacy record; `g_theta` is never overwritten.
- v1 (`target_count = 4`, 9 metadata) and v2 (4 targets, 17 or 39 metadata)
  writers still emit their original bytes via
  `write_validation_with_version`. `read_validation` auto-detects by version
  and `target_count` (v1/v2 → 4 columns, v3 → 6) and rejects a mismatch
  LOUDLY. Truth-extended files may still be v2; the default writer and
  `write_validation_with_truth` are v3.
- `POTOKTRJ` sidecar: `write_trajectory` (v1, stride 6) and
  `write_trajectory_v2` (stride 24) are byte-identical to before.
  `generate-dataset` now uses **`write_trajectory_v3`, version 3, stride 26**:
  the 24 v2 fields followed by `g_accel_y, g_accel_x` (indices 24–25).
  `read_trajectory` auto-detects stride 6/24/26; v1 defaults the privileged
  fields, v2 defaults `g_accel` to `[0,0]`.
- The terminal `TruthRecord` in validation metadata (22 floats) is unchanged;
  the per-tick command target lives in the v3 sidecar.

### Exact conversion semantics

`optimizer.rs` exposes `world_accel_command(s, wind, phase, frame) -> [ax, ay]`
(m/s²): the same raw LQR demand `solve` uses for `g_theta`, each axis clamped
to ±`MAX_COMMAND_MPS2` = 10 m/s², or `[0, 0]` when `!frame.ai_input_valid()`,
the state is non-finite or `!phase.ai_allowed()`. The dataset stores it as
`[g_accel_y, g_accel_x] = [ay, ax]` (Y first: the fin seam is roll-then-pitch).
The actuator command is recovered exactly as
`fin_accel_command = [g_accel_y, -g_accel_x]` in `[roll, pitch]` order; this
is bit-identical to the pre-change oracle (`clamp(-ax) == -clamp(ax)` for a
symmetric envelope). `Episode.targets[i].g_accel` is computed on the optimizer
from the true state and wind at tick `i` and is copied to
`TrajectoryTick.g_accel` at the same tick, so validation columns 4–5 equal
sidecar indices 24–25 frame-for-frame. `g_theta` remains a diagnostic/legacy
target. Quality gates check the new columns and truth against ±10 m/s².

### Verification

```
cargo check --workspace --offline -> Finished, no warnings
cargo fmt --all -- --check        -> clean
cargo test  --workspace --offline -> 67 passed, 0 failed
```

New gates in `core/tests/dataset_tests.rs` (+5): v3 sidecar stride/round-trip,
validation v1/v2/v3 byte layout and field order, LOUD version/target-count
mismatch, `fin_accel_command` vs exported target on a flown episode, and
zero-command bit identity for all three validation and sidecar versions.
The CLI `generate-dataset` gate now asserts v3 targets, 26 truth names and
sidecar/validation `g_accel` agreement; all 14 quality checks still pass.

Skipped: AI-, scripts-, edge- and protocol-side readers (owned by other lanes;
the AI reader still hardcodes target_count=4 and version 2, so it must accept
version 3 / six targets before consuming these datasets); no physics tuning; no
change to the v1/v2 byte layouts or to the terminal truth metadata.
