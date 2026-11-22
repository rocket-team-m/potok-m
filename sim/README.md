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
cargo bench -p potok-sim --bench throughput   # episodes/s, frames/s
```

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
| 8 | 4 | version `u32` = 1 |
| 12 | 8 | seed `u64` |
| 20 | 4 | episode count `u32` |
| 24 | 4 | frames per episode `u32` |
| 28 | 4 | feature count `u32` |
| 32 | 4 | target count `u32` = 4 |
| 36 | 4 | frame rate `f32` |
| 40 | 4 | dt `f32` |
| 44 | 4 | duration `f32` |
| 48 | 32 | reserved (zero) |

Per episode: `u64` index, then 9 metadata `f32` (mass, thrust_scale, pitch,
yaw, gust_start [−1 = none], gust_duration, gust_amplitude, v_10,
shear_delta), then `frames × features` f32, then `frames × 4` f32 targets
(g_roll, g_pitch, confidence, ai_healthy as 0/1).

`potok_sim::validation::read_validation` parses the format; a round-trip test
covers it.

## Reference optimizer

`g_theta = [roll, pitch]` is an LQR-style full-state feedback (lateral
position/velocity error → desired tilt, minus body-rate damping), clamped to
±0.12 rad, with an infinite-horizon approximation rather than a solved
Riccati equation or online MPC. It sees true state and wind, so it is an upper
bound on the sensor-only model. It is zeroed on pad/boost by the AI interlock.
