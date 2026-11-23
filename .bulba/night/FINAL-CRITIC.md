# FINAL-CRITIC — remaining risks and false claims (2026-11-23)

**Role:** read-only final critic. **Sources:** all `.bulba/night/*.md` reports, all
`.superpowers/sdd/rocketcraft-implementation/task-*.md` reports, and the current
worktrees (`potok-m`, `-sim`, `-ai`, `-protocol`, `-firmware`, `-edge`, `-workbench`).
No source, config or workflow file was edited by this task.

**Method note:** every claim below was checked against the tree at the time of
writing, not taken from a report alone. Several report findings are already
fixed in the current tree; those are listed at the end as *resolved*, not ranked.
The ranks are by consequence if the claim is believed and the gap is not closed.

## Top 10

### 1. CRITICAL — the safety layer is not in the flight path
All 15+ firmware safety modules are declared in the library and tested through
`#[path]` includes, but the shipping control loop bypasses them.

- Evidence: `potok-m-firmware/firmware/src/lib.rs:11-34` declares `interlocks`,
  `deployment`, `phase_fsm`, `watchdog`, `sensor_voting`, `control_pid`,
  `attitude`, `separation`, `search_beacon`, `self_test`.
  `firmware/src/control.rs:228-255` applies `reply_outcome.command` directly via
  `board.apply_command()` and feeds the watchdog with legacy
  `watchdog::may_feed(...)`; no `Interlocks::evaluate`, no `WatchdogPolicy`.
  `firmware/src/main.rs:113,125` wires only `step`, `PhaseFsm`, `LinkState`;
  `main.rs:90-98` leaves phase events and `apply_command` as stubs.
  The interlock-in-actuation-path arrangement exists **only** in the SIL harness
  (`SIL01-report.md:20`, `Guarded<B>`), not in `main.rs`.
- Risk: 216 green firmware tests (VERIFY-FW.md:27) measure modules the binary
  never calls. A link-accepted command reaches the board without the independent
  interlock, deployment or policy gates that the reports treat as delivered.
- Fix: route every `board.apply_command` in `control::step` through
  `Interlocks::evaluate` + `WatchdogPolicy::on_control_cycle`, and have
  `main.rs` consume deployment/separation/sensor-voting outputs; add a gate that
  drives the real `step()` and asserts a vetoed command never reaches the board.

### 2. CRITICAL — no AI model has beaten zero command; the v3 closed-loop A/B does not exist
- Evidence: `/home/sehaxe/potok-m/ab_report.json` — `exit_code: 1`,
  `verdict.pass: false`, `measurement: "POTOKTRJ v1 flown-truth sidecar, not surrogate"`.
  `ai/docs/MODEL_CARD.md:121-134` states the g_theta students "не обыгрывают zero
  command", the oracle win validates the actuator seam "не студента", and the
  v3 zero/oracle/student closed-loop comparison "не запускалось и не
  записывалось". All AI and SIM night reports are contract gates, not a positive A/B.
- Risk: the core value proposition is unproven, and the only recorded A/B
  artifact is scored on v1 targets while v3 is declared authoritative.
- Fix: run the pre-registered 3-seed v3 closed-loop A/B on the v3 sidecar and
  commit the resulting artifact with its window and seed order; until it passes,
  keep the AI in observer/advisor mode in every user-facing claim.

### 3. CRITICAL — three current documents disagree on OpenRocket parity
- Evidence: `SIM06.md:31-45` — apogee 3127.363 → 3185.378 m (**1.86 %**, under a
  2 % gate), vmax 1.45 %, time-to-apogee 0.14 s; thresholds picked after
  measuring, explicitly "regression gates … not fidelity claims".
  `ai/docs/MODEL_CARD.md:135-138` — "Parity с OpenRocket … красная: апогей 6,40 %
  при лимите 2 %, vmax 2,42 %", `SIM_FIDELITY.md` admission unmet.
  `docs/src/content/docs/process/decisions.md:134-151` (ADR-008) — worst altitude
  4.95 % at a 5 % limit, drag 1–2.3 % at 5 %, M=0.5–0.6 vs OR M=0.82 open.
- Risk: the simulator is the training oracle for every AI result; with three
  incompatible fidelity verdicts, no reader can tell whether the training data
  is validated. A 1.86 % "pass" and a 6.40 % "fail" cannot both describe the same gate.
- Fix: publish one parity run pinned to fixture + config + commit, state which
  scenario/config each of the three numbers belongs to, and make the docs quote
  that single result; leave the M contradiction explicitly open.

### 4. HIGH — the v3 edge inference path is unreachable and its format has no writer
- Evidence: `potok-m-edge/src/runtime.rs:149-163` dispatches only
  `POTOKM1\0` → v2 and everything else → legacy; no `POTOKM3\0` /
  `V3Model` branch. `src/lib.rs:9,15,19,27` exports `calibration_v3`,
  `model_v3`, `speculative_v3`, but no binary consumes them (`bin/infer.rs:56`,
  `bin/bench.rs:56` use `Runtime::load`). `B14-report.md:35-40`: the v3 model
  layout was *defined by the reader* and is "the one open interface risk".
  On the AI side the only v3 export code is `ai/tests/test_v3_export_roundtrip.py:46-51`
  — a test-local byte packer, not a production writer (AI01.md:46-51).
- Risk: reader, guard, dispatch and calibration for v3 are fully tested against
  in-memory fixtures, yet no real model can traverse the path end to end.
- Fix: add a `POTOKM3\0` branch in `Runtime::load` that builds
  `V3Model` + `V3Dispatch` + `Calibration`, implement the matching writer in
  `ai/`, and add one export → load → dispatch round-trip test.

### 5. HIGH — no board has ever been brought up; all hardware gates are blank
- Evidence: `potok-m-firmware/firmware/src/main.rs:34,45,55-61,90-98` —
  UART RX/TX DMA, board pins and phase events are `todo!()`/`TODO(schematic)`;
  `VERIFY-FW.md:53-58` records `hw::build_board()` as `todo!()` and SIL as not a
  substitute for HIL; `FW16.md:14-17` makes the checklist blank by design;
  `SIL05.md:48-50` says green SIL is necessary but not sufficient.
- Risk: any "flight-ready", "<10 ms p99" or "timing budget" reading is a model,
  not hardware evidence. FW10.md:83 and SIL04.md:40 say so themselves.
- Fix: complete schematic/BOM-bound pin map and the board adapters, then execute
  the 12 signed gates in `docs/HIL_CHECKLIST.md`; do not infer flight status from SIL.

### 6. HIGH — night reports are snapshots presented as current truth; several already contradict the tree
- Evidence: `VERIFY-MONO.md:56-67` and `DOCS02.md:50` report a one-repo claim in
  `index.mdx` and a 3-repo README table; current
  `docs/src/content/docs/index.mdx:43-49` and `README.md:13-24` already state the
  multi-repo model with all 7 repos. `AI10.md:55-60` says no `ab_report.json`
  exists in the tree; it exists and is committed. `VERIFY-PROTO.md:43-49` says no
  C golden harness is committed; standalone protocol HEAD now has `90a718f
  ci(protocol): commit self-contained C golden harness`. `VERIFY-FW.md:16-18`
  records 0 unpushed commits at `e39b131`; HEAD is now `349c96a`, 1 ahead.
- Risk: a reader acting on these reports re-fixes solved issues and trusts
  "all gates green" at commits that are no longer HEAD.
- Fix: stamp every night report with the repo HEAD it verified, and re-run the
  `VERIFY-*` gates at current HEADs before any of their claims are reused.

### 7. HIGH — protocol has two divergent homes and two golden-harness implementations
- Evidence: `PROTO01.md:8-15` added `.github/workflows/cross-language.yml` and
  `scripts/check_golden.sh` to the **monorepo** because
  `/home/sehaxe/potok-m-protocol` was believed absent; that path now exists.
  The canonical repo carries `stm32/tests/check_golden.sh` (commit `90a718f`)
  and its own generator drift gate (VERIFY-PROTO.md:14). Histories are unrelated:
  standalone `fbf6a46 → 90a718f`, monorepo `protocol/` `7f3adcb → 0a3dece`.
- Risk: golden frames, generated bindings and CI can drift between the two
  copies with no single authority; a fix landed in one is invisible to the other.
- Fix: declare one canonical protocol repo, make the other a thin reference or
  delete it, and run the same golden + generator-drift gate in that one repo.

### 8. MEDIUM-HIGH — conformal calibration gives opposite decisions at the boundary
- Evidence: `AI07.md:45-53` and `B16-report.md:62-66` — the bin-map route uses
  `searchsorted(..., side="left")`, so at `p == p_hat` it returns 0, while the
  no-map exact route `s <= q_hat` returns 1. AI07 sidesteps this by exporting
  with `with_bin_map=False`; the covered test never exercises the boundary
  through the path the edge actually uses.
- Risk: the same calibrated model can be accepted or rejected at the exact
  threshold depending on which route a consumer picks.
- Fix: make both routes inclusive at equality (`side="right"` or add equality to
  the exact route) in Python and Rust, then add one boundary test run with
  `with_bin_map=True`.

### 9. MEDIUM-HIGH — hard gates pin test-local fixtures rather than production paths
- Evidence: `AI01.md:46-51` (test-local byte packer for v3 target export),
  `AI02.md:58-62` (`_dispatch()` re-implemented in the test), `B03.md`,
  `B15-report.md:29-35` (fixture predictors), `B14-report.md:35-40` (reader
  defines the format), `SIM06.md:47-50` (thresholds fitted after measurement).
- Risk: a contract gate can stay green while the production writer/dispatcher
  diverges; the v3 format gap in item 4 is the realized form of this risk.
- Fix: for each contract, point at least one gate at the production writer or
  dispatcher (not a reimplementation), starting with an export → load → dispatch
  test for v3.

### 10. MEDIUM — RocketCraft visual verification is permanently red and the font fix changed runtime behavior
- Evidence: `FIX-WB.md:38-41` — `scripts/check_visual.sh` is red by design because
  the 12 matrix screenshots are not committed and no capture entry point exists;
  `FIX-WB.md:23` removed the bundled OFL fonts and `register_fonts()`, so GPUI
  falls back to system fonts when Inter / JetBrains Mono are not installed;
  `FIX-WB.md:42-43` notes the debug-build timing failure left open in edge.
- Risk: UI changes in the workbench cannot be regression-checked, and the
  "pinned look" is now host-dependent — a visual claim cannot be reproduced.
- Fix: add a single capture entry point that emits the 12 matrix screenshots to
  CI artifacts, and pin the font names with a runtime assertion that reports
  which family actually resolved.

## Resolved during the review (do not re-report as open)

- `index.mdx` one-repo claim and the 3-repo README table — fixed at current HEAD
  (VERIFY-MONO.md:56-67 vs `index.mdx:43-49`, `README.md:13-24`).
- Missing `ab_report.json` (AI10.md:55-60) — file now exists, exit_code 1.
- Missing committed C golden harness (VERIFY-PROTO.md:43-49) — now `90a718f`.
- Binary-size guard red on 4 tracked >2 MB files (B19-report.md:67-87) — current
  VERIFY-MONO.md:20-21 reports the guard green, 0 tracked files >2 MB.
- All firmware modules "not wired into lib.rs" (FW01-FW15, SIL01-SIL04) — now
  declared at `firmware/src/lib.rs:11-34`; the remaining issue is item 1 above
  (declared but not called from `step()`), not missing declarations.

## Verification limits

No build, test or hardware run was performed by this critic. Findings 1, 4, 5,
6 and 7 were confirmed by reading current source/workflow files; findings 2, 3,
8, 9 and 10 rest on the reports plus the current artifacts named above.
