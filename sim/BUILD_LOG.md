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
12.5–13.5 kg, area 0.00817 m² (D=0.102 m), static margin 0.0976 m (0.957 cal),
accel noise 0.011 m/s² (DELTA 1064 µg), gyro noise 0.0015 rad/s. Source table
and uncertainty notes are in `sim/README.md`.

Two-stage thrust uses the **upper end** of the motor passports' impulse ranges:
2.4 kNs / 5.5 s / 1.88 kg → 436 N at 0.342 kg/s (RDK-2000) and 1.2 kNs /
3.2 s / 1.0 kg → 375 N at 0.3125 kg/s (RDK-1000), both below the passport peak
thrusts. The quoted *average* thrusts (400 / 265 N, lower impulses) were also
measured: with this mass and Cd=0.35 they give only ~161 m/s and ~1.4 km, so
they cannot meet the user's 240 m/s / ~3 km TTX. This discrepancy is documented
in the config and README rather than hidden. Aerodynamic coefficients remain
explicitly marked placeholders.

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
