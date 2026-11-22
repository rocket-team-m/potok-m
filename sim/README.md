# potok-sim

Deterministic Rust simulator for the Potok-M rocket AI training data:
6-DoF variable-mass rigid body, RK4, standard atmosphere, wind (power-law
mean + discrete shear + 1-cosine gust + Dryden-like correlated turbulence),
canonical 50 Hz UART request frames, and a full-state reference optimizer.

Workspace members: `core` (`potok-sim`) and `cli` (`potok-cli`). The wire
layout is **not** defined here: `core` depends on `potok-protocol`
(`../../protocol/potok-protocol`), generated from `protocol/uart.yaml` and
already used by `edge/`.

```bash
cargo test --workspace
cargo check --workspace
cargo run -p potok-cli -- --help
cargo run -p potok-cli -- --config config/default.toml --seed 1 --episodes 4 \
    --validation-out out/
cargo run -p potok-cli -- --episodes 20 --physics-features --split train \
    --validation-out train/
cargo run -p potok-cli -- --episodes 20 --physics-features --split held-out \
    --validation-out heldout/
cargo bench -p potok-sim --bench throughput   # episodes/s, frames/s
```

`config/default.toml` is unchanged (8 output channels). `--physics-features`
appends the six derived channels to make the 14-value model frame; `--split`
keeps one side of the withheld wind/gust split in the validation file. Both
flags are optional, so every older command line keeps working.

## Source of default numbers

| value | default | source |
|---|---|---|
| lift-off mass | 12.5–13.5 kg | user-stated TTX (~13 kg) |
| diameter / frontal area | 0.102 m / 0.00817 m² | user-stated TTX |
| static margin | 0.0976 m = 0.957 cal | user-stated TTX |
| motor impulse | 2.4 + 1.2 kNs (upper passport) | RDK-2000 / RDK-1000 passports, chat PDFs (2.2–2.4 / 1.0–1.2 kNs) |
| thrust / mass flow | 436 N @ 0.342 kg/s for 5.5 s; 375 N @ 0.3125 kg/s for 3.2 s | propellant masses and burn times from the passports; mass flow = propellant / burn. Upper passport impulses are used because the quoted *average* thrusts (390–400 / 260–270 N) give only ~160 m/s / ~1.4 km, short of the TTX target |
| target speed / apogee | ~240 m/s / ~3 km | user-stated TTX; this model reaches ~203 m/s / ~2.4 km at nominal seed (see uncertainty below) |
| gyro noise | 0.0015 rad/s | previously validated simulator value |
| accel noise | 0.011 m/s² (1064 µg) | DELTA report |
| wind (v_10, shear, gust, turbulence) | 3 m/s, 2 m/s, p=0.2, σ=0.6 m/s | launch-day assumption, not passport data; reduced from earlier 8/6 m/s so the vehicle stays near vertical |

**Placeholder / uncertainty.** `cd0`, `cna`, `pitch_damping`, `roll_damping`, the
`cd_mach` table and the ad-hoc angular damping model in `dynamics.rs` are
**not** from the real Potok-M aerodynamic model. They are engineering
placeholders, calibrated only so the nominal trajectory stays in the TTX
ballpark. Their uncertainty is unquantified; a real OpenRocket/Ansys
aerodynamic card (as referenced in the project docs) should replace them.
Motor peaks (500–550 / 550–600 N) and thrust ripple are not modelled — only
average thrust and mass flow. The remaining gap between simulated (~203 m/s,
~2.4 km) and stated TTX (~240 m/s, ~3 km) is dominated by these placeholder
aerodynamics and by the impulse choice within the passport ranges.

## Domain randomization (Potok-M only)

Per `SPEC_SMART.md` (commit `8894951`) there are **no other rocket types**.
Every episode is the same airframe with tolerances drawn from independent
seeded sub-streams; `config/default.toml` itself is untouched (the ranges
below are serde defaults in `core/src/config.rs` and can be overridden by a
custom config).

| quantity | range | source |
|---|---|---|
| lift-off mass | 12.5–13.5 kg | existing `initial.mass_kg` |
| thrust scale | 0.94–1.06 | existing `initial.thrust_scale` |
| CG offset | ±0.02 m | existing `initial.cg_offset_m`, added to the sampled margin |
| `cd0` | 0.30–0.40 | `aero.cd0_range`, new; shifts the whole mach table |
| static margin | 0.85–1.10 cal | `aero.static_margin_cal_range`, new; converted with D = 2√(A/π) = 0.102 m |
| gyro noise | 0.0010–0.0020 rad/s | `sensor.gyro_noise_range`, tolerance around the 0.0015 validated value |
| accel noise | 0.008–0.014 m/s² | `sensor.accel_noise_range`, tolerance around the DELTA 1064 µg value |
| wind at 10 m | 0–15 m/s | `wind.v_10_range`, new |
| gust amplitude | 5–20 m/s | `wind.gust_amplitude_range`, new |
| shear jump / altitude | 0–8 m/s / 50–400 m | `wind.shear_delta_range`, `wind.shear_altitude_range`, new |
| sensor lag | ×1.0 train, ×1.5–2.0 withheld | scenario cell, on top of `sensor.lag_tau` |

Mass, thrust and CG use the legacy `INIT` draw; `cd0`, margin and sensor noise
use an independent `DOMA` sub-seed. Scenario cells replace the old
`wind.gust_probability` draw with a fixed wind/gust/shear combination, so
generalization is measured across scenarios rather than random luck.

## Physics features and the 14-value frame

The raw eight channels (`gyro_x/y/z`, `acc_x/y/z`, `baro_h`, `pitot_q`) are
unchanged. Six features are computed in `core/src/physics.rs` from the RK4
state, the atmosphere and the sensor's body-frame specific force:

| name | formula |
|---|---|
| `alpha_deg` | `atan2(omega_body_z, v_body_x)` in degrees. This is the literal SPEC formula; the classical aerodynamic AoA would use `−v_body_z`. |
| `beta_deg` | `atan2(v_body_y, v_body_x)` in degrees |
| `q_dyn_kpa` | `0.5 · rho · ‖v − w‖² / 1000`, `rho` from `Atmosphere::at` |
| `a_lat_resid` | `hypot(a_meas_x − (N_x + D_x)/m, a_meas_y − (N_y + D_y)/m)`; `N = q_dyn·A·cna·(v_body_i/az)`, `D` is the mach-table drag vector, `az = max(‖v−w‖, 1)` |
| `drag_est` | `cd0 · q_dyn · A / m_current` (current mass falls during burn) |
| `static_margin_cal` | `static_margin_m / (2√(A/π))` |

`v_body = R(q)ᵀ(v_world − w_world)`. The first six names live in
`physics::PHYSICS_FEATURES`; `sensor::FEATURE_NAMES` ends with them.

The six **conditioning** numbers are separate: `mass_kg`, `area_m2`, `cd0`,
`static_margin_cal`, `thrust_peak_n`, `gyro_noise`. They are written once per
episode in validation metadata (not into the frame matrix), so a caller can
broadcast them over the time axis. SPEC's "8 + 6 + 6 = 14" wording is
inconsistent; the frame is 8 raw + 6 physics = 14, and conditioning is
metadata.

## Withheld wind/gust scenarios

`core/src/scenario.rs` defines 20 deterministic cells: 16 training and 4
withheld (ids 16–19). A cell fixes fractions of the wind/gust/shear envelopes
plus a sensor-lag multiplier. Withheld cells combine envelope extremes that
appear in **no** training cell (all four knobs at maximum, lag 1.5–2.0×), so
the train/held-out boundary is a wind/gust scenario boundary, not a vehicle
type. `episode_index % 20` selects the cell; `EpisodeMeta.scenario_id`,
`.split`, `.v_10`, `.gust_amplitude` and `.shear_delta` record what ran.

Use `--split train` / `--split held-out` with `--episodes 20` (or a multiple)
to write the two validation sets. The criterion in the spec is held-out RMSE
≤ 1.25 × training RMSE.

## Determinism

Every random quantity derives from `run.seed` and the episode index through
`SplitMix64` sub-seeding into a local `StdRng`. No global RNG, no wall-clock,
no thread-count dependence: `sim.episodes(n)` and
`sim.run_episodes_parallel(n)` (rayon) produce identical episodes. Turbulence
filters are pre-generated in a fixed draw order, once per episode.

`run.fast_field = true` uses a precomputed altitude table for the atmosphere;
`false` evaluates the analytic equations. The two paths are numerically close
and a test (`fast_field_matches_exact_analytic_profile`) bounds the relative
deviation of every exported feature. Set `false` for reference runs.

`sim.sensors = false` runs dynamics only (no sensor frames) for throughput
measurement.

## Canonical UART request frame (50 Hz)

56 bytes little-endian, encoded by `potok_protocol::encode_request`:

| offset | size | field |
|---|---|---|
| 0 | 2 | magic `0x504D` |
| 2 | 2 | sequence `u16` (wraps) |
| 4 | 4 | `time_us` `u32` |
| 8 | 1 | phase `u8`, 0..=4 |
| 9 | 1 | version `u8` = 1 |
| 10 | 1 | flags `u8` |
| 11 | 1 | reserved = 0 |
| 12 | 12 | gyro xyz `f32`, rad/s |
| 24 | 12 | specific force xyz `f32`, m/s² (body frame, includes gravity tilt) |
| 36 | 12 | magnetometer xyz `f32`, gauss |
| 48 | 4 | height `f32`, m |
| 52 | 2 | `pitot_dp` `i16`, Pa |
| 54 | 2 | CRC-16/CCITT-FALSE (poly `0x1021`, init `0xFFFF`) over bytes 0..54 |

Simulator-to-wire phase map: `Pad`/`Boost` → 0 (BURN/Idle), `Coast` → 1
(COAST/Armed), `Apogee` → 2 (APOGEE/Flight), `Descent` → 3 (DESCENT/Landing),
`Landed` → 4 (LANDED/Fault). The canonical crate names the numeric variants
`Idle/Armed/Flight/Landing/Fault`; the numeric values are the integration
contract and match the BURN..LANDED names in the task. The inverse mapping in
`Phase::from_protocol_code` treats 0 as `Boost`.

Flags are the canonical bits: IMU_OK, BARO_OK, PITOT_OK, AI_READY (cleared on
pad/boost), MOTOR_ARMED (boost), FAULT (saturation/non-finite). Gust, shear and
CRC are kept as simulator-side context and do not consume flag bits.
`Frame::ai_input_valid()` requires AI_READY and no FAULT. The test
`canonical_golden_request_is_byte_identical` re-encodes the committed golden
request fixture from `protocol/potok-protocol/tests` byte for byte.

## Validation binary format

`--validation-out DIR` writes `DIR/validation.bin`. All integers are
little-endian; all arrays are `f32`, row-major.

Header (80 bytes = `potok_sim::validation::HEADER_SIZE`; the table below is
its exact byte layout, and `ai/src/potok_ai/data.py` uses the same 80-byte
`<8sIQ4I3f32s` struct):

| offset | size | field |
|---|---|---|
| 0 | 8 | magic `POTOKSIM` |
| 8 | 4 | version `u32` = 2 (1 = legacy 9-metadata file) |
| 12 | 8 | seed `u64` |
| 20 | 4 | episode count `u32` |
| 24 | 4 | frames per episode `u32` |
| 28 | 4 | feature count `u32` |
| 32 | 4 | target count `u32` = 4 |
| 36 | 4 | frame rate `f32` |
| 40 | 4 | dt `f32` |
| 44 | 4 | duration `f32` |
| 48 | 4 | metadata count `u32` = 17 in v2, zero in v1 |
| 52 | 28 | reserved (zero) |

The writer emits **version 2** (`validation::VERSION = 2`). Per episode:
`u64` index, then 17 metadata `f32`:

| index | field |
|---|---|
| 0–8 | legacy: mass, thrust_scale, pitch, yaw, gust_start [−1 = none], gust_duration, gust_amplitude, v_10, shear_delta |
| 9–14 | conditioning: mass_kg, area_m2, cd0, static_margin_cal, thrust_peak_n, gyro_noise |
| 15 | scenario id `u16` as `f32` |
| 16 | split flag: 0 = train, 1 = held-out |

Then `frames × features` f32, then `frames × 4` f32 targets (g_roll, g_pitch,
confidence, ai_healthy as 0/1). A consumer concatenates the conditioning
vector to every frame to form the model input.

**Backward compatibility:** `write_validation_with_version(..., VERSION_V1)`
writes the original 9-metadata layout with version 1 and a zeroed
`meta_count`. `read_validation` accepts both versions; for v1 it reconstructs
nominal Potok-M conditioning (scenario 0, train) from the defaults. Both
paths are covered by round-trip tests.

`potok_sim::validation::read_validation` parses the format; a round-trip test
covers it. Note that an AI-side reader must use the v2 layout when consuming
files produced by the new writer.

## Reference optimizer

`g_theta = [roll, pitch]` is an LQR-style full-state feedback (lateral
position/velocity error → desired tilt, minus body-rate damping), clamped to
±0.12 rad, with an infinite-horizon approximation rather than a solved
Riccati equation or online MPC. It sees true state and wind, so it is an upper
bound on the sensor-only model. It is zeroed on pad/boost by the AI interlock.
