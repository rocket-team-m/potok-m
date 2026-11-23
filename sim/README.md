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
cargo run -p potok-cli -- generate-dataset --seed 42 \
    --train-episodes 100 --held-out-episodes 20 --out dataset/
cargo bench -p potok-sim --bench throughput   # episodes/s, frames/s
```

`generate-dataset` is the single-command dataset generator: it writes both
splits, their v2 truth sidecars, a `manifest.json` next to each
`validation.bin`, and `quality_report.json` at the output root. All old CLI
flags keep working and still write the v1 sidecar.

`config/default.toml` is unchanged (8 output channels). `--physics-features`
appends the six derived channels to make the 14-value model frame; `--split`
keeps one side of the withheld wind/gust split in the validation file. Both
flags are optional, so every older command line keeps working.

## Source of default numbers

| value | default | source |
|---|---|---|
| lift-off mass | 10.4–11.4 kg | OpenRocket Simulation 9 |
| diameter / frontal area | 0.102 m / 0.00817 m² | user-stated TTX |
| static margin | 0.0765 m ≈ 0.65 cal of the 118 mm outer diameter | OpenRocket-derived |
| motor impulse | 2.4 + 1.2 kNs (upper passport) | RDK-2000 / RDK-1000 passports, chat PDFs (2.2–2.4 / 1.0–1.2 kNs) |
| thrust / mass flow | 400 N @ 0.342 kg/s for 5.5 s; 312.5 N @ 0.3125 kg/s for 3.2 s | passport minimum 2.2 + 1.0 kNs; propellant masses from the passports |
| target speed / apogee | M=0,5–0,6 (±) / ~3 km | current TTX; this model reaches ~276 m/s / ~3.12 km at nominal seed, matching OpenRocket within 1% |
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
average thrust and mass flow. With the OpenRocket-derived inputs the nominal simulation reaches ~276 m/s
and ~3.12 km, within 1% of OpenRocket Simulation 9.

## Domain randomization (Potok-M only)

Per `SPEC_SMART.md` (commit `8894951`) there are **no other rocket types**.
Every episode is the same airframe with tolerances drawn from independent
seeded sub-streams; `config/default.toml` itself is untouched (the ranges
below are serde defaults in `core/src/config.rs` and can be overridden by a
custom config).

| quantity | range | source |
|---|---|---|
| lift-off mass | 10.4–11.4 kg | existing `initial.mass_kg` |
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
| 8 | 4 | version `u32` = 3 (2 = legacy 4-target file, 1 = legacy 9-metadata file) |
| 12 | 8 | seed `u64` |
| 20 | 4 | episode count `u32` |
| 24 | 4 | frames per episode `u32` |
| 28 | 4 | feature count `u32` |
| 32 | 4 | target count `u32` = 6 (4 in v1/v2) |
| 36 | 4 | frame rate `f32` |
| 40 | 4 | dt `f32` |
| 44 | 4 | duration `f32` |
| 48 | 4 | metadata count `u32` = 17 in v2, zero in v1 |
| 52 | 28 | reserved (zero) |

The writer emits **version 3** (`validation::VERSION = 3`). Per episode:
`u64` index, then 17 metadata `f32`:

| index | field |
|---|---|
| 0–8 | legacy: mass, thrust_scale, pitch, yaw, gust_start [−1 = none], gust_duration, gust_amplitude, v_10, shear_delta |
| 9–14 | conditioning: mass_kg, area_m2, cd0, static_margin_cal, thrust_peak_n, gyro_noise |
| 15 | scenario id `u16` as `f32` |
| 16 | split flag: 0 = train, 1 = held-out |

Then `frames × features` f32, then `frames × 6` f32 targets in this exact
order (one row per 50 Hz frame):

| index | field | units / meaning |
|---|---|---|
| 0 | `g_roll` | rad, legacy angle-domain diagnostic target |
| 1 | `g_pitch` | rad, legacy angle-domain diagnostic target |
| 2 | `confidence` | [0,1] |
| 3 | `ai_healthy` | 0/1 |
| 4 | `g_accel_y` | m/s², clamped world **+Y** lateral acceleration demand |
| 5 | `g_accel_x` | m/s², clamped world **+X** lateral acceleration demand |

Columns 0–3 are the unchanged v1/v2 target record; `g_theta` is **never
overwritten** by the new fields. Columns 4–5 come from
`ReferenceOptimizer::world_accel_command` on the true 6-DoF state and true
wind at that tick, and are zero while the AI interlock is active. The command
the fin actuator actually consumes is
`fin_accel_command = [g_accel_y, -g_accel_x]` in the seam's `[roll, pitch]`
order, clamped to ±10 m/s² (`MAX_COMMAND_MPS2`). Y is written first to match
that seam order. This is the authoritative target for AI training; `g_theta`
is kept only as a diagnostic and for legacy readers. A consumer concatenates
the conditioning vector to every frame to form the model input.

**Backward compatibility:** `write_validation_with_version(...)` with
`VERSION_V1` or `VERSION_V2` writes the original 4-target layout (9 metadata
in v1, 17 in v2) with byte-identical target rows; `read_validation`
auto-detects the row width from the header version and `target_count`
(v1/v2 → 4, v3 → 6) and rejects a mismatch loudly. For v1 it reconstructs
nominal Potok-M conditioning (scenario 0, train) from the defaults. All three
paths are covered by round-trip tests. The AI reader in `ai/` is owned by
another lane: until it accepts version 3 with six targets, it cannot consume
files from the current writer.

#### Truth-extended v2 metadata

`output.truth_metadata = true` (or `write_validation_with_truth`) writes
**version 3** with `meta_count = 39`: the classic 17 floats plus a 22-float
truth record captured at the episode's last 50 Hz tick. The header
`meta_count` is the authority; `read_validation` accepts 9 (v1), 17 (classic
v2/v3) and 39 (truth-extended v2/v3), so old files and old readers keep
working. `ValidationEpisode.truth` is `None` for the first two. This terminal
record is unchanged (no `g_accel`); the per-tick command target lives in the
v3 sidecar below.

| index (after the 17) | field | units |
|---|---|---|
| 0–2 | true wind vector at the tick (world X/Y/Z) | m/s |
| 3 | gust start (`-1` = none) | s |
| 4 | gust amplitude | m/s |
| 5 | shear-layer altitude | m |
| 6 | shear-layer velocity jump | m/s |
| 7 | shear-layer thickness | m |
| 8–10 | true position (world X/Y/Z) | m |
| 11–13 | true velocity (world X/Y/Z) | m/s |
| 14–17 | true quaternion `(w, x, y, z)` | — |
| 18–20 | true body angular rate (X/Y/Z) | rad/s |
| 21 | true mass at the tick | kg |

The terminal truth record above is per-episode (last tick only). For a
privileged LoopWM teacher, use the per-tick sidecar below.

#### Per-tick privileged truth sidecar (POTOKTRJ v2)

`write_trajectory` still writes **version 1**: 64-byte header (`POTOKTRJ`,
`version=1`, episode_count u32, frames_per_episode u32, `stride=6`, 40 reserved)
and `frames x 6` f32 per episode (`x, y, z, vx, vy, vz`). Every old reader and
binary file keeps working.

`write_trajectory_v2` still writes **version 2** with the same 64-byte header
and `stride=24` (unchanged bytes). The writer used by `generate-dataset` is
`write_trajectory_v3`, **version 3, `stride=26`**: the 24 v2 fields plus two
appended floats per tick. The v2 field order is:

| index | field | units |
|---|---|---|
| 0–2 | world position x/y/z | m |
| 3–5 | world velocity x/y/z | m/s |
| 6–9 | quaternion `(w, x, y, z)` | — |
| 10–12 | body angular rate x/y/z | rad/s |
| 13 | mass | kg |
| 14–16 | true wind x/y/z (world frame) | m/s |
| 17 | gust start (`-1` = none) | s |
| 18 | gust amplitude | m/s |
| 19 | shear-layer altitude | m |
| 20 | shear velocity jump | m/s |
| 21 | shear-layer thickness | m |
| 22–23 | reference optimizer `g_theta` `[roll, pitch]` | rad |
| 24 | `g_accel_y` (v3 only) | m/s² |
| 25 | `g_accel_x` (v3 only) | m/s² |

Indices 24–25 equal validation target columns 4–5 on the same tick, and the
fin command is again `[24, -25]`. `read_trajectory` accepts all three versions:
the header version and stride are the authority (6 / 24 / 26), and a v1 read
leaves the privileged fields at explicit defaults (`gust_start_s = -1`,
everything else zero), never at guessed values. A v2 read leaves `g_accel` at
`[0, 0]`. The order is pinned by `trajectory::TRUTH_TICK_NAMES` and
`TRUTH_V3_TICK_NAMES`. `g_theta` and `g_accel` are the reference optimizer's
targets; a `--control-file` run flies its own command and is not silently
substituted.

## Dataset manifest and quality gates

`generate-dataset --seed S --train-episodes M --held-out-episodes K --out DIR`
writes:

```
DIR/train/validation.bin                (v3, 17 metadata floats, 6 targets)
DIR/train/validation.trajectory         (POTOKTRJ v3, stride 26)
DIR/train/manifest.json
DIR/held-out/validation.bin
DIR/held-out/validation.trajectory
DIR/held-out/manifest.json
DIR/quality_report.json
```

Each manifest is deterministic JSON (schema version 1): effective config hash
(sha256 of canonical TOML), seed, split, sorted scenario ids, episode/frame
counts, feature/target/truth names, observed min/max bounds, and `sha256` +
byte count of `validation.bin` and its sidecar. No timestamps or process ids.
The model frame is the canonical 14 channels (eight raw + six physics).

Before writing the manifests the generator re-reads the files it wrote and
runs LOUD gates: finite values everywhere, disjoint train/held-out scenario
sets, feature limits, `g_theta` within +/-0.12 rad, `g_accel_*` within
+/-10 m/s², confidence/health in `[0,1]`, plausible truth (mass > 0, unit
quaternion, gust/shear in range),
consistent frame counts, at least `--min-episodes-per-split` (default 4), and
a full second generation whose four binary files must be byte-identical.
Any failure exits nonzero with the gate name and offending value; nothing is
clamped or skipped. `quality_report.json` records every passing check.

## Aero coefficient card

`[aero] source = "placeholder"` (default) keeps the legacy analytic model and
prints a **COUNTED** marker line to stderr on each `Simulator::new`
(`aero::placeholder_uses()` is the counter). `source = "card"` loads the 2-D
grid named by `card_path`, resolved like `boundary_profile`.
`config/aero_card.toml` is a documented example generated from the
placeholder coefficients: selecting it reproduces the current trajectory
bit-for-bit (covered by `generated_card_reproduces_placeholder_and_placeholder_is_counted`).

Each channel (`cd0`, `cna`, `pitch_damping`, `cm_delta`) is a rectangular grid
of `values[alpha][mach]` rows. Interpolation is bilinear and coefficients are
evaluated at each RK4 sub-stage. Any query outside a grid axis is a **LOUD**
`SimError::AeroOutOfGrid` (returned by `interpolate`, and propagated out of
`run_episode`); nothing is clamped. `cm_delta` adds a gimbal control moment
proportional to `q_dyn·A·D`; the example card sets it to zero.

## Sensor realism knobs

All new `[sensor]` knobs default to **off/zero**, so old configs and tests are
unchanged. `gyro_arw` / `accel_arw` are Allan angle/velocity random-walk
densities (discrete std = ARW / √dt); `gyro_bias_stability` /
`accel_bias_stability` drive a seeded random walk on the constant bias.
`latency_steps` delays whole frames; `gyro_quant_step` / `accel_quant_step`
override `quant_step` per channel; `mag_sat` / `baro_sat` add optional limits;
`spike_probability` / `spike_amplitude` inject a seeded measurement spike that
sets the canonical FAULT bit and gates the AI. Every draw comes from the
episode's `SENS` sub-seed, so runs stay reproducible.

`potok_sim::validation::read_validation` parses the format; a round-trip test
covers it. Note that an AI-side reader must use the v2 layout when consuming
files produced by the new writer.

## Reference optimizer

`g_theta = [roll, pitch]` is an LQR-style full-state feedback (lateral
position/velocity error → desired tilt, minus body-rate damping), clamped to
±0.12 rad, with an infinite-horizon approximation rather than a solved
Riccati equation or online MPC. It sees true state and wind, so it is an upper
bound on the sensor-only model. It is zeroed on pad/boost by the AI interlock.

The exported command target is **not** `g_theta`. `ReferenceOptimizer`
computes the same raw LQR demand `(ax, ay)` and clamps it to the actuator
envelope as `[ax, ay]` (`world_accel_command`); the dataset writes it as
`[g_accel_y, g_accel_x] = [ay, ax]`. The fin solver consumes
`fin_accel_command = [g_accel_y, -g_accel_x]` m/s² in `[roll, pitch]` order
and converts it to a fin deflection by a moment balance. Training on
`g_theta` instead of this pair was the cause of the negative closed-loop A/B;
`g_theta` remains as a diagnostic/legacy column only.
