# SITE-RET — post-FIXPUMP site and SVG retraction

**Date:** 2026-11-23
**Repo:** `/home/sehaxe/potok-m`. Scope: `docs/src/assets/ab-results-v3.svg`,
`site/index.html`, `site/fragments/status.html`, `site/fragments/overview.html`,
`docs/src/content/docs/project/*.md*` where a stale A/B claim was present.
No subagents. No physics, config, gate, threshold, export or test touched.
GPL SPDX header on the SVG preserved; no chat provenance added. Local commit only, no push.

## Authority used

No `ABFIX.md` exists in `.bulba/night/`, so no positive-control redefinition applies.
Authority = `ab_report.md` / `ab_report.json` (post-FIXPUMP), `RECALC01.md`, `DOCS-RET.md`.

- Verdict **FAIL**, `exit_code = 1`; oracle positive control fails on seeds 2/3
  (apogee 1972.859 → 1970.708 m, 1983.003 → 1975.402 m), so no student result is credited.
- Zero-command baseline: lateral RMS 1306–1309 m, apogee 1973–1983 m
  (per seed 1309.409/1973.743, 1306.397/1972.859, 1306.380/1983.003).
- `loopwm` seed 1: lateral 1309.409 → 1267.367 m (−42.042 m, 96.789 %),
  apogee 1973.743 → 2077.784 m (+104.041 m) — recorded, not credited.
- `lrs` seed 3 export missing, no stand-in; `н/д` in the figure is a missing export, not a zero.
- Old PASS, −26.19 m / +125.12 m, and the 1528/1735 m and 3971 m bars are retracted.

## Files changed

- `docs/src/assets/ab-results-v3.svg` — regenerated from the post-FIXPUMP report:
  new axis ranges (lateral 1100–1400 m, apogee 1900–2400 m), all 5 arms × 3 seeds
  replaced with current values, explicit `ВЕРДИКТ: FAIL` badge and a failure line
  naming seeds 2/3; footnotes now say the old bars and PASS are retracted.
- `site/index.html` — strip stat is now `FAIL · oracle TC 2/3`; telemetry rail and
  meta use seed-1 baseline 1309.409 m vs loopwm 1267.367 m (−42.042 m, 96.789 %),
  apogee +104.041 m, with an explicit "не засчитано"; both per-scenario charts were
  replotted from the post-FIXPUMP seed-1 scenario rows and their aria-labels updated;
  source line and the AI closed-loop preview card now show FAIL, not PASS.
- `site/fragments/status.html` — the `AI · v3` row is `A/B FAIL · post-FIXPUMP`;
  states the positive-control failure, the 1306–1309 / 1973–1983 m baseline and that
  the seed-1 loopwm gain is recorded but not credited.
- `docs/src/content/docs/project/run.md` — the one stale bullet ("A/B пройден",
  −25–26 m / ~120 m) rewritten as a post-FIXPUMP FAIL with the same honest numbers.

## Checked, no edit needed

- `site/fragments/overview.html` — carries no A/B verdict, baseline or ceiling number.
- Other `docs/src/content/docs/project/*.md*` (`architecture`, `goals`, `licenses`,
  `roadmap`, `roles`) — no post-FIXPUMP or stale A/B number.

## Verification

- `cd docs && bun run build` → exit 0, 32 page(s) built; only the pre-existing
  i18n/404/chunk-size warnings. The built asset contains the FAIL badge.
- Guard link-check (Python from `.github/workflows/guard.yml:85-141`, run in repo root)
  → `checked 52 local navigation links, skipped 9 asset links`, 0 broken, exit 0.
- `python3 -c "xml.etree.ElementTree.parse(...)"` on the SVG → OK.
- `git diff --check` on the in-scope files → clean.

## Residual risk outside scope

`docs/src/content/docs/ai/gates.md:38-41` still says the embedded SVG "собран до
FIXPUMP и подписывает старый вердикт PASS — он устарел". That caption was true before
this task and is now wrong in the other direction: the figure says FAIL. That file is
in `docs/src/content/docs/ai/`, outside this assignment, so it was not edited; its
one-line caption needs a follow-up from its owner.

## Commit

Local commit only; no push.
