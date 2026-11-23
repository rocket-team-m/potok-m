# Night task DOCS01 — ADR page

**Status: done.** Local commit on `main` (see `git log -1 --grep DOCS01`). No push. No subagents.

## Scope

Edited exactly one documentation file:

- `docs/src/content/docs/process/decisions.md`

Plus this report. No other source or doc file was touched. Uncommitted `ai/out/v3_final/…`, `sim/core/tests/` and other agents' report files were left untouched and unstaged. No chat/LLM provenance was introduced in the ADR text.

## What changed

The page previously carried only a template and three table rows. It now has the full ADR set with Context / Options / Decision / Consequences bodies.

- Existing entries kept: ADR-001 (already `отменён ADR-002`), ADR-002 and ADR-003 are now marked superseded by the new detailed entries (`отменён ADR-004`, `отменён ADR-005`).
- ADR-004 — canonical per-domain repositories; `potok-m` is the docs/ADR hub, code lives in `potok-m-ai`, `-sim`, `-protocol`, `-firmware`, `-edge`, `-3d` and `potok-m-workbench`.
- ADR-005 — GPL license statement: own code `GPL-3.0-or-later`, proprietary use via a separate commercial license (`LICENSE-COMMERCIAL.md`), third-party licenses preserved (`Apache-2.0` solidcore, `MIT OR Apache-2.0` SolidCraft-derived, `OFL-1.1` Inter, OpenTUI Mono), `NOTICE`/`THIRD-PARTY-LICENSES` shipped, internal use without distribution not covered, SPDX headers and manifests.
- ADR-006 — validation v3 target: six targets per frame, authoritative fin command `[g_accel_y, −g_accel_x]` (m/s², world frame), `g_theta` diagnostic only, loud refusal for v1/v2 without `g_accel` (no angle-for-command substitution).
- ADR-007 — RocketCraft (`potok-m-workbench`) as a separate private workbench owning CAD → mesh → CFD/FEA → aero-card → trajectory → report; the aero card, not the CAD sources, is the boundary consumed by the simulator.
- ADR-008 — OpenRocket Simulation 9 as the simulator's parity oracle: fixture `or_sim9.csv` (55 rows, 0.5 s grid), config `or_sim9.toml`, workspace member `potok-or-parity`, gate `cargo test --workspace -p potok-or-parity` with apogee/Vmax 2%, altitude 5%, drag 5% limits; coefficient fitting to the number forbidden; `pitot_dp` limitation stated.
- ADR-009 — SIL gate: `.github/workflows/sil.yml` in `potok-m-firmware`, four named steps (host suite, wire faults, timing budget, `no_std` thumb check), deterministic 500 Hz real-`step()` harness, `SIL_FRAME_FLOOD` 856 cycles vs 24 000, `assert_tick_budget()`; green SIL necessary but not sufficient, HIL remains final verification.

Facts were taken from the repo's own sources (`goals.md`, `ai/concept.md`, `ai/training.md`, `ai/telemetry.md`, `roadmap.md`, `sim/BUILD_LOG.md`, `SPEC_ENGINE_IDEAL.md`, `sim/tests/or_parity_tests.rs`, `LICENSE`, and the B18/B19/SIL04/SIL05 reports) — no invented numbers and no external references claimed.

## Verification

```
cd /home/sehaxe/potok-m/docs && bun run build
  -> ✓ Completed in 3.69s; 19 page(s) built
grep built HTML for internal links
  -> /potok-m/project/goals/, /potok-m/ai/concept/, /potok-m/ai/training/,
     /potok-m/ai/telemetry/ all present; no leftover relative *.md links
git diff --check -- docs/src/content/docs/process/decisions.md
  -> clean
```

Build warnings (empty `i18n` collection, missing 404 entry, >500 kB chunk) are pre-existing and unrelated.

## Notes

- Line 6 still reads `Раздел в работе.` — left as-is; the ADR set is complete but the process section as a whole is not claimed finished.
- ADR-007 states the private status of the workbench as the decision; the B18 report noted a conflicting "open" wording in the workbench README. The monorepo only links to the canonical repo, so no stale claim about the workbench's code being public was added to the site.
- The parity/oracle and SIL numbers are transcribed from this repo's build logs and reports; they are not independent measurements made in this task.
