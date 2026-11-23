# DOCS-ABFIX — final docs sync after ABFIX F1

**Date:** 2026-11-23. **Repo:** `/home/sehaxe/potok-m`. **Scope:** exactly three
files — `docs/src/content/docs/ai/gates.md`, `ai/README.md`,
`ai/docs/MODEL_CARD.md` — plus this report. No sim source, config, gate,
threshold, weight, export or flight number was changed. No subagents.
GPL-3.0-or-later throughout.

## 1. What was stale

Concurrent commits (`5af2a30`, `0c2196b` and follow-ups, and the `docs(site)`
sync `57ad689`) left three documents on the superseded verdict: `FAIL`,
`exit_code = 1`, "positive control fails on 2/3 seeds", "no student credited",
and a figure caption saying the SVG still carried the pre-FIXPUMP PASS. All
three are now on the ABFIX F1 verdict.

## 2. Verdict now stated (source: regenerated `ab_report.json` / `ab_report.md`)

- `verdict.status = PASS`, `exit_code = 0`, `positive_control.credible = true`,
  primary arm `loopwm`.
- F1 positive control: lateral improved on all three seeds
  (−17.312 / −10.503 / −8.382 m), all 12 per-scenario lateral deltas negative,
  actuator exercised at the hard cap (16 661 / 16 584 / 16 531 cap ticks out
  of the same number of active ticks, peak command 10.0 m/s²). Apogee delta
  (+11.692 / −2.151 / −7.601 m) is labelled a **diagnostic only**, not a pass
  criterion.
- Post-FIXPUMP zero-command baseline: lateral 1306.397–1309.409 m, apogee
  1972.859–1983.003 m.
- Crediting: `loopwm` CREDITED 3/3, `lite` CREDITED 3/3, `lrs` PARTIAL 2/3
  (seed-3 export missing, no stand-in, never a fabricated third seed).
- PASS is about lateral guidance only; the win is modest — roughly 42 m off
  ~1309 m lateral (≈3%). Not flight qualification; the AI remains an observer.
- Explicitly retracted: pre-FIXPUMP +2 km apogee (≈3971 m), the 1528/1735 m
  bars, the derived 35–40 % / +2400…+2700 m ceilings, and the old PASS verdict.
  The ≈51 % lateral / +1940 m hard bounds remain unattainable bounds, not
  predictions. F2 (repair the oracle control law) remains open.

## 3. Edits by file

- `docs/src/content/docs/ai/gates.md` — verdict admonition (warning → note,
  F1 PASS / exit 0), figure caption (SVG now rebuilt under F1), removed the
  bold "two control failures" marks on the oracle apogee rows, rewrote the
  post-table interpretation (F1 criteria, apogee diagnostic, credit table,
  ~3 % lateral win), extended the historical retraction with the old PASS and
  +2 km, rewrote blocker item 3, added `ABFIX.md` to the source list.
- `ai/README.md` — status block (line 7), v3 data-contract note (line 73),
  nightly-gates paragraph (line 104), report list (line 111), limitations
  bullet (line 122).
- `ai/docs/MODEL_CARD.md` — intro (lines 6–9), the "Честный статус A/B" v3
  post-FIXPUMP bullet (lines 132–147), historical retraction (lines 153–160)
  and the parity/qualification bullet (lines 171–174).
- GPL license sections left intact in both AI documents; no chat, agent or
  session provenance introduced anywhere.

## 4. Verification

- `cd docs && npm run build` → 32 pages built, `[build] Complete!`, exit 0.
- CI local-markdown-link check (logic from `.github/workflows/guard.yml:85-141`)
  → `checked 52 local navigation links, skipped 9 asset links`, exit 0.
- Stale-verdict scan over the three files: no `FAIL`, `exit 1`, `красн`,
  `провален` or `не кредиту` remains, except the correct historical reference
  to the superseded v1/v2 artifact (`gates.md:103`, `e57c6bf`).
- Numbers cross-checked by reading `ab_report.json` directly: status, exit
  code, the three criteria, per-seed lateral/apogee deltas, cap fractions,
  all 12 scenario signs and all three credit statuses match the text.
- Safety caveats retained: no real flight log, `SIM_FIDELITY.md` criteria not
  met, model not admitted to control.

## 5. Landing

Committed locally on `main` in the same commit as the three doc edits. Before
pushing, `origin/main` (`e7a2d51`) was confirmed to be an ancestor of the local
tip (`57ad689`), so the push is a fast-forward.

## 6. Out of scope

ABFIX §6 named other files (`docs/src/content/docs/project/run.md`,
`site/index.html`, `site/fragments/status.html`,
`docs/src/assets/ab-results-v3.svg`) already synced to F1 PASS by `57ad689`;
none was touched here. `docs/GROUND_STATION.md:463` says "A/B не проводился"
about ground-station commands, not the student A/B, so no edit was needed.
