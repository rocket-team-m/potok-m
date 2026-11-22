# Potok-M model export format (version 2)

`src/potok_ai/export.py` is the canonical writer; training produces the weights,
so every other reader (including `edge/`) must follow this document exactly.

All integers and floats are **little-endian**. Floats are IEEE-754 binary32;
`int8` is a signed byte; strings are UTF-8 without a NUL terminator. There is no
padding between fields.

## File layout

```
[ header ][ layer record ] * n_layers [ sha256: 32 raw bytes ]
```

The checksum covers **every byte from offset 0 up to but not including the last
32 bytes**. It is the raw 32-byte SHA-256 digest, not a hex string. A reader must
recompute it and refuse the file on mismatch.

### Header

| offset | type | field | value |
|---:|---|---|---|
| 0 | 8 bytes | magic | `POTOKM1\0` = `50 4F 54 4F 4B 4D 31 00` |
| 8 | u32 | version | `2` |
| 12 | u32 | `in_features` | 8 in the teacher fixture; 10 for a legacy student (8 sensors + phase + flags); 20 for a smart student (14 physics features + 6 conditioning numbers) |
| 16 | u32 | `width` | channel width of the TCN |
| 20 | u32 | `out_features` | 3 (pitch, yaw, confidence) |
| 24 | u32 | `rank` | TSCT rank |
| 28 | u32 | `kernel` | depthwise kernel, 3 in the shipped fixtures |
| 32 | u32 | `n_blocks` | number of TCN blocks |
| 36 | f32 | `g_max` | correction bound applied with `tanh` |
| 40 + 4·i | u32 | `dilations[i]` | `i = 0 … n_blocks-1`, normally 1,2,4,8,16 |
| 40 + 4·n_blocks | u32 | `flags` | bit 0 = ternary factors, bit 1 = 4-bit activations |
| 44 + 4·n_blocks | u32 | `n_layers` | number of layer records |

Header size is `48 + 4·n_blocks` bytes (68 for the 5-block fixtures); the first
layer record starts immediately after it.

### Layer records

Each record begins:

| type | field | meaning |
|---|---|---|
| u8 | `kind` | `1` = TSCT low-rank map, `2` = fp32 tensor |
| u32 | `name_len` | byte length of the name |
| bytes | `name` | `name_len` UTF-8 bytes, no terminator |

**kind 1 (TSCT)** stores the factors already in ternary form:

| type | field | count |
|---|---|---|
| u32 | `in_features` | 1 |
| u32 | `out_features` | 1 |
| u32 | `rank` | 1 |
| f32 | `alpha_u` | 1 |
| f32 | `alpha_v` | 1 |
| int8 | `u_codes` | `in_features·rank`, values −1/0/+1 |
| f32 | `s` | `rank` |
| int8 | `v_codes` | `out_features·rank`, values −1/0/+1 |

The effective factors are `U = u_codes·alpha_u` and `V = v_codes·alpha_v` in
float32. **Do not quantise them again** — the codes and scales are already the
final inference weights. A `PotokModel` rebuilt from a file uses
`ternary=False` for exactly this reason.

**kind 2 (fp32 tensor)** stores a dense parameter:

| type | field | count |
|---|---|---|
| u32 | `ndim` | 1 |
| i32 | `shape[i]` | `ndim` values, row-major (C order) |
| f32 | `data` | `prod(shape)` values, row-major |

### Names and order

TSCT records come first, in module order, then all fp32 tensors, in parameter
registration order. For a 5-block model (`n_blocks=5`) the exact sequence is:

1. `blocks.0.pw`, `blocks.1.pw`, `blocks.2.pw`, `blocks.3.pw`, `blocks.4.pw` (kind 1)
2. `stem.weight` `(width, in_features, 1)`, `stem.bias` `(width,)`
3. for each block `i`: `blocks.i.dw.weight` `(width, 1, kernel)`,
   `blocks.i.dw.bias` `(width,)`, `blocks.i.norm.weight` `(width,)`,
   `blocks.i.norm.bias` `(width,)`
4. `norm.weight` `(width,)`, `norm.bias` `(width,)`
5. `head.weight` `(out_features, width)`, `head.bias` `(out_features,)`

Only parameters are stored. Buffers (the ternary `anneal` scalar) and the
Hadamard/FWHT state are intentionally absent: both are training-only.

## Inference path the weights belong to

Let `x` be `(B,T,F)` float32. For each block with dilation `d` and kernel `k`:

1. `stem`: `Conv1d(in_features, width, 1)` on `(B,F,T)`, then back to `(B,T,width)`.
2. Left-pad the time axis with `(k−1)·d` zeros, run a **depthwise** `Conv1d`
   (`groups=width`, kernel `k`, dilation `d`, no built-in padding), giving `(B,T,width)`.
3. `LayerNorm` over the last dimension of that convolution output.
4. Pointwise TSCT: `W_eff = (U·s) @ Vᵀ`, then `y = x_block @ W_effᵀ` using the
   *un-normed* residual input. The block output is `x_block + y`.
5. After all blocks: final `LayerNorm` over the last dimension, take frame `T−1`.
6. `head`: `raw = h @ head.weightᵀ + head.bias`; output
   `[tanh(raw[0])·g_max, tanh(raw[1])·g_max, sigmoid(raw[2])]`.

The 4-bit Hadamard/FWHT activation quantiser is active **only in training**; in
eval/export it is the identity, so an inference reader does not implement it.
The same is true of ternary straight-through annealing: the file already holds
the quantised factors.

## Cross-check fixture

- `tests/fixtures/tiny_model.bin` — untrained, `seed=1234`, `width=16`, `rank=8`,
  default 5 dilations, `in_features=8`, `out_features=3`.
- `tests/fixtures/tiny_expected.npy` — shape `(1,3)`, outputs of that model in
  eval mode for the fixed window
  `(arange(400, dtype=float32).reshape(1,50,8) − 200) / 100`.
- `tests/fixtures/generate.py` regenerates both deterministically.
- `tests/test_fixture.py` reloads the file with `load_export(...).to_model()` and
  asserts `np.array_equal` against the `.npy`.

An edge reader should read **this fixture** and compare its three float32 outputs
to `tiny_expected.npy` before trying any larger model.

## Simulator validation input (`validation.bin`)

`src/potok_ai/data.py` reads the canonical training data written by
`potok_sim::validation::write_validation`. Little-endian throughout; all arrays
are float32, row-major. The field list is from `sim/README.md` §"Validation
binary format" (which is also what the Rust writer emits):

| offset | type | field |
|---:|---|---|
| 0 | 8 bytes | magic `POTOKSIM` |
| 8 | u32 | version = 1 |
| 12 | u64 | seed |
| 20 | u32 | episode count |
| 24 | u32 | frames per episode |
| 28 | u32 | feature count |
| 32 | u32 | target count = 4 |
| 36 | f32 | frame rate |
| 40 | f32 | dt |
| 44 | f32 | duration |
| 48 | 32 bytes | reserved (zero) |

Reading these fields sequentially consumes **80 bytes**, not the 76 stated in
the README: 48 bytes of fields plus a 32-byte reserved field. The Rust
`Header::write_to` also writes 80 bytes; `Header::header_size()` reports 84,
which is a third, inconsistent number. This reader follows the field table and
the actual bytes, and does not touch `sim/` to reconcile the docs.

Per episode: `u64` index, 9 metadata f32 (mass, thrust scale, pitch, yaw,
gust start, gust duration, gust amplitude, v_10, shear delta), then
`frames × features` f32, then `frames × 4` f32 targets in the order
`(g_roll, g_pitch, confidence, ai_healthy)`. `Episode.targets` keeps all four
columns; `WindowDataset` uses the first two (g_roll, g_pitch) as the
distillation reference.

`load_episodes` dispatches on the magic: `POTOKSIM` uses the canonical reader;
anything else falls back to the older `POTOK1\0\0` fixture format or a bare
float32 array. `tests/test_sim_validation.py` packs a tiny validation file with
Python `struct` and checks shapes, metadata, and target values.

## Smart validation input (`POTOKSIM` version 2, SPEC_SMART.md)

This is the format actually written by `potok_sim::validation` (commit with
the domain-randomization work). The header is still 80 bytes, with `version = 2`
and bytes 48..52 (the first four reserved bytes) holding `meta_count = 17` as a
u32; bytes 52..80 stay zero. `feature_count` must be at least 14 and the first
14 features are used. Every episode is one randomized Potok-M flight inside
the real tolerances (mass 14-15 kg, thrust 0.94-1.06 of nominal, CG
+/-0.02 m, Cd0 0.30-0.40, static margin 0.85-1.10 cal, sensor noise, wind
0-15 m/s, gusts 5-20 m/s and shear layers). There are no other vehicle types.
Per episode the layout is `u64 index` followed by **17 f32 metadata**:

| index | type | field | meaning |
|---:|---|---|---|
| 0..8 | 9 f32 | legacy meta | mass, thrust scale, pitch, yaw, gust start/duration/amplitude, v_10, shear delta |
| 9..14 | 6 f32 | `conditioning` | `mass_kg, area_m2, cd0, static_margin_cal, thrust_peak_n, gyro_noise` (the order of Rust `Conditioning::as_array`) |
| 15 | f32 | `scenario_id` | integer wind/gust cell id (0..19 in `sim/core/src/scenario.rs`) |
| 16 | f32 | `split` | 0 = train, 1 = held-out scenario |

Then `frames x feature_count` f32 features (8 raw channels plus `alpha_deg,
beta_deg, q_dyn_kpa, a_lat_resid, drag_est, static_margin_cal`) and `frames x 4`
f32 targets as in v1. `read_validation` exposes these as `Episode.metadata`
(legacy 9), `Episode.conditioning` (6), `Episode.scenario_id` and
`Episode.scenario` (`"scenario_16"`), and `Episode.held_out`. A v1 file has no
conditioning (a zero vector is used), scenario id or split flag and cannot be
used for the smart CLI. `group_by_scenario` groups by `scenario` string. The
holdout is always by scenario, never by vehicle.

`train_smart.py` reads the split straight from these flags. The teacher sees a
`(B, 50, 14)` window plus a `(B, 6)` conditioning batch; the exported smart
student header carries `in_features = 20` (14 physics features followed by the
6 conditioning values in every per-frame input). Edge is synced separately.
