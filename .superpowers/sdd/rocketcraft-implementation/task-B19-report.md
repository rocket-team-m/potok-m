# Task B19 — CI hardening

**Status:** implemented; one inherited (pre-existing) blocker on the binary-size check is flagged below.
**Commit:** local only, no push. See commit message `ci: license gate, binary size guard and link checks`.

## Scope

Changed exactly one workflow file: `.github/workflows/guard.yml`.
No other repo's workflows were touched. No subagents were used. This report is the
only other file added.

## What changed

`guard.yml` now runs on every push and PR with three jobs and least-privilege
`permissions: contents: read`:

1. **`binary-size-guard`** (was `heavy-files`) — keeps calling the existing
   `.github/scripts/guard.sh`, which rejects binary extensions and any tracked
   file over 2 MB. Renamed for clarity; no logic change.
2. **`license-gate`** — own-code license gate:
   - `LICENSE` must be the GNU GPL, version 3;
   - `LICENSE-COMMERCIAL.md` must exist and name `GPL-3.0-or-later`;
   - every tracked `SPDX-License-Identifier` in source/manifests must be
     `GPL-3.0-or-later` (headers are optional; an incompatible declaration is
     a loud `::error`);
   - every `license =` / `"license":` field in tracked TOML/JSON manifests must
     be `GPL-3.0-or-later`.
   All failures are hard failures with the offending file and line, per the
   repo's loud-failure rule; no silent fallback.
3. **`link-check`** — local Markdown link gate over all tracked `.md`/`.mdx`:
   - hard-fails broken links to pages, directories and Astro site routes
     (`/potok-m/...` is mapped to `docs/src/content/docs/...`, with `.md`,
     `.mdx` and `index.md` fallbacks);
   - ignores external URLs, mailto/tel/data, pure anchors and media/PDF asset
     links, which are served from `docs/public/` or generated at build time.

`ai.yml` and `docs.yml` are unchanged: `guard.yml` runs on every push/PR, so the
new gates cover the whole repo without duplicating jobs.

## Verification

```
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/guard.yml'))"
  -> OK, jobs = binary-size-guard, license-gate, link-check

bash /tmp/opencode/b19-license-gate.sh          -> exit 0
  "OK: own code is GPL-3.0-or-later, commercial option documented."
bash /tmp/opencode/b19-link-check.sh            -> exit 0
  "checked 7 local navigation links, skipped 2 asset links"

Negative tests in an isolated temp git repo (not the working tree):
  - SPDX `MIT` in a .py              -> exit 1, ::error names file:line
  - manifest `license = "MIT"`       -> exit 1, ::error names file:line
  - non-GPL LICENSE text             -> exit 1, two ::error lines
  - `[bad](no/such/page.md)`         -> exit 1, ::error names file and target
  - `[bad](asset.svg)` only          -> exit 0, counted as skipped asset

git diff --check -- .github/workflows/guard.yml -> OK
```

The gate scripts were extracted verbatim from the YAML and executed locally so
the result is the same code CI will run. `git grep -H` is used instead of
`--with-filename` for portability; local verification caught that the latter is
rejected by this git build and that the first draft had therefore been a gate
that always passed.

## Inherited blocker — binary-size guard is already red on `main`

Running `.github/scripts/guard.sh` locally fails on four **already-tracked**
files added by commit `29bea98`:

```
2401600   ai/out/v3_data/held-out/validation.bin
3120224   ai/out/v3_data/held-out/validation.trajectory
9606160   ai/out/v3_data/train/validation.bin
12480704  ai/out/v3_data/train/validation.trajectory
```

These are real git blobs (not LFS pointers) over the repo's documented 2 MB
limit, so both `guard.yml`'s binary-size guard and the pre-existing `guard`
job in `ai.yml` fail on `main` independently of this task. Removing or
ignoring them is outside B19's assigned paths (`.github/workflows/*.yml`) and
is a destructive action, so I did not do it. Remediation for the owner of
`ai/out`: move the four artifacts to Releases, `git rm --cached` them and add
matching ignore rules. Until then, "workflows green" is not achievable for the
binary-size jobs without either that cleanup or weakening the gate, which I
refused to do.

## Notes

- Own code in this repo that carries an SPDX header declares
  `GPL-3.0-or-later`; the three manifests with a `license` field
  (`ai/tui/Cargo.toml`, `protocol/Cargo.toml`, `sim/Cargo.toml`) all declare
  `GPL-3.0-or-later`. The gate therefore passes on the current tree.
- Third-party license files (`NOTICE`, `THIRD-PARTY-LICENSES`) belong to the
  private workbench repo (task A01), not to this repo; no such check was added.
- Two pre-existing asset links are reported as skipped, not fixed:
  `ai/README.md -> docs/banner.svg` (file absent) and
  `docs/src/content/docs/ai/concept.md -> ../../pdf/potok-m-ai-concept.pdf`
  (asset lives in `docs/public/pdf/`). Both are in other agents' paths; the
  docs build, not the source-tree check, is authoritative for site assets.
