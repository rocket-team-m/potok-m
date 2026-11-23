# SIM07 — command→dynamics control seam hard tests (night task)

**Date:** 2026-11-23
**Repo:** `/home/sehaxe/potok-m/sim` (git root `/home/sehaxe/potok-m`)
**Scope:** created exactly one source file, `sim/core/tests/control_seam_hard_tests.rs` (402 lines after `rustfmt`). No source edits. No subagents. No push.

## What was delivered

A single integration-test file with an SPDX `GPL-3.0-or-later` header, using only the public `potok_sim` API. Seven hard tests pin the command seam end to end — from the LQR command, through the moment balance in `dynamics::derivative`, to flown lateral RMS.

| Requirement | Test |
|---|---|
| command→dynamics mapping | `command_maps_to_the_documented_fin_moment_in_dynamics` — sets `Vehicle::control_accel` and compares the `omega` derivative against an independently recomputed `authority * solve_fin_deflection(...) / i_xx`; also pins that the command changes **no** translational derivative (`vel`, `pos`, `mass` bit-equal), that `authority * delta == mass * g * arm`, that the sign is preserved per axis, and that doubling the demand doubles angular acceleration |
| zero-command bit identity | `zero_command_is_bit_identical_to_legacy_on_every_exported_field` — seeds 1/7/1337, three entry points (`run_episode_with_control`, the same with `-0.0`, `run_episode_with_control_ctx`) versus `run_episode`: every exported field compared (`frames`, `features`, `targets`, `truth`, `trajectory`, `to_bits()` of `apogee_m`/`max_speed_mps`, `frames_generated`, `phase_counts`, `invalid_frames`, `nan_frames`). Negative zero is covered deliberately — the seam gates on `!= 0.0`, not on the sign bit |
| acceleration units | `command_units_are_lateral_mps2_and_scale_with_mass` — `MAX_COMMAND_MPS2 == 10.0` (~1 g), `MAX_CONTROL_DEG` is a numeric alias, `MAX_FIN_DEFLECTION_RAD == 25°` in radians; for g ∈ {0.25, 1, 3, 9.80665} m/s² the applied aerodynamic moment equals `mass * g * arm` (the command carries a mass factor, so it is acceleration, not an angle) and doubling mass doubles the deflection at fixed g. `validate_command` accepts ±10 exactly and refuses 10+1e-9 |
| fin clamp | `fin_deflection_saturates_at_the_mechanical_stop_and_counts` — monotone sweep, exact bit-level `±MAX_FIN_DEFLECTION_RAD` saturation that is stable under further demand, a just-inside case that does **not** clamp, clamp counter `>= before + 3` (COUNTED, not silent), thin-air zero with its own counter, and a LOUD `cm_delta` fault for zero authority. Global-counter assertions use `>=` because integration tests share one process |
| phase gating | `phase_and_health_interlocks_zero_the_command_independently` — a healthy frame (`FLAG_AI_READY`, FAULT clear) still yields exactly `[0,0]` on Pad and Boost, while Coast/Apogee/Descent/Landed command nonzero; pins the `[ay, -ax]` fin ordering and the LQR demand (`[-8.6, 6.55]` for the fixture state). The health gate is isolated by clearing AI_READY and by setting FAULT in Coast. `powered_ticks_export_zero_accel_command` checks the flown truth: every tick with `t <= burn_end()` exports `g_accel == [0,0]` and coast ticks carry a live command |
| wrong sign worsens | `wrong_sign_command_worsens_held_out_lateral_rms` — for seeds 1/2/3, pooled lateral RMS over held-out episodes `[16,17,18,19]` satisfies `wrong > base > oracle` strictly |

## Verification (evidence)

- `cargo test -p potok-sim --test control_seam_hard_tests` → **7 passed, 0 failed**.
- `cargo test -p potok-sim` (full suite, 21 binaries incl. doc-tests) → **141 passed, 0 failed**, every `test result:` line `ok`.
- `rustfmt --edition 2021 --check sim/core/tests/control_seam_hard_tests.rs` → clean.
- Measured held-out lateral RMS (pooled over 4 episodes, reported by the test):

  | seed | wrong sign | baseline | oracle |
  |---|---|---|---|
  | 1 | 1630.636618 m | 1536.929739 m | 1045.630967 m |
  | 2 | 1665.948330 m | 1516.966676 m | 1008.334644 m |
  | 3 | 1691.078011 m | 1535.253977 m | 1019.963779 m |

Two defects were found and fixed during development, both in the new file only: `fin_accel_command` needs the `&Vehicle` argument (3 call sites), and the doubled-demand linearity probe had to use smaller demands (0.25 / -0.5 m/s²) because the original 1 / -2 doubles crossed the fin stop — that was a test-fixture error, not a source bug.

## Git

Local commit `20f4d8d` — `test(sim): hard gates for command->dynamics control seam (SIM07)`. No push.
