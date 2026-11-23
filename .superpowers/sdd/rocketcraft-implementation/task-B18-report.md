# Task B18 — Docs site update

**Status:** done. Committed locally (`docs: document multi-repo layout, v3 target, GPL and RocketCraft`); nothing pushed. No subagents used.

## Scope

Touched exactly the assigned paths under `docs/src/content/docs/`:

- `ai/concept.md`
- `ai/telemetry.md`
- `ai/training.md`
- `project/goals.md`
- `project/roadmap.md`
- `project/roles.md`

No other files edited. `astro.config.mjs`, other docs sections and sibling
agents' files were left untouched. No chat/LLM provenance was introduced or
left in these files.

## What changed

- **Multi-repo structure** — `project/goals.md` gained a canonical-repository
  table (potok-m as docs/ADR hub; ai, sim, protocol, firmware, edge, 3d, and
  RocketCraft/workbench as code homes) plus the rule that code changes go to the
  canonical repo. `project/roles.md` maps each role to its repo;
  `project/roadmap.md` states that milestones are cross-repo.
- **v3 target** — `ai/concept.md`, `ai/training.md` and `ai/telemetry.md` now
  state the v3 contract: six targets per frame, authoritative fin command
  `[g_accel_y, −g_accel_x]` (m/s²), `g_theta` (`g_roll`, `g_pitch`) kept as
  diagnostic only, and a loud refusal for v1/v2 files without `g_accel` — no
  silent angle-for-command substitution. `project/roadmap.md` gained a v3 row.
  Wording matches `ai/src/potok_ai/data.py` / `data_v3.py` and the B03 report.
- **GPL license** — `project/goals.md` gained a license section: own code
  `GPL-3.0-or-later`, separate commercial license (`LICENSE-COMMERCIAL.md`),
  third-party licenses preserved (Apache-2.0 solidcore, MIT OR Apache-2.0
  SolidCraft-derived, OFL-1.1 Inter, OpenTUI Mono), NOTICE/THIRD-PARTY-LICENSES
  shipped. Repeated briefly in `project/roles.md` and the three AI pages.
- **RocketCraft links** — linked in the repo table and in "Где живёт код" /
  "Где обучаем" sections of all three AI pages, described as the desktop
  CAD → mesh → CFD/FEA → aero-card → trajectory → report workbench that
  supplies the aero card consumed by the simulator.
- Internal cross-links use the site's base-prefixed slug `/potok-m/project/goals/`;
  a first pass with relative `../project/goals.md` was caught in the built HTML
  as unresolved and replaced.

## Verification

```
cd /home/sehaxe/potok-m/docs && bun run build
  -> ✓ Complete! 19 page(s) built (baseline before edits: also green)
grep built HTML for internal links
  -> href="/potok-m/project/goals/" present; no leftover relative *.md links
git diff --check -- docs/src/content/docs/ai docs/src/content/docs/project
  -> clean
  6 files changed, 73 insertions(+), 4 deletions(-) (final)
```

## Notes / gaps

- `process/decisions.md` (outside B18 scope) still carries ADR-001
  "one repo with folders instead of many repos", which the new multi-repo docs
  supersede. Owner/process lane should update or mark it superseded.
- The workbench README calls RocketCraft "open" while the implementation plan
  calls it private; the docs link the canonical repo without asserting either
  status, so no claim is made that could go stale.
- Unrelated uncommitted work in `ai/out/v3_final/`, `sim/core/tests/` and other
  reports was left untouched and not staged.
