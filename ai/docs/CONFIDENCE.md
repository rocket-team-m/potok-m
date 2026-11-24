# Split-conformal confidence in the staged trainer (AICAL)

Scope: `conf_integration.py`, `train_teacher_v2.py`, `tests/test_conf_integration_v2.py`.
This lands a **measurement instrument** in the staged teacher-v2 pipeline. It changes
no gate limit, no reward weight, and makes no closed-loop or flight claim.

## Where the number comes from

`train_teacher_v2.run()` scores the inner-held-out scenarios (the same scenarios
`partition_episodes` withholds from training; eval cells 16..19 are refused before
any model sees them). The prediction rows are the confidence logit and per-sample
command error produced by `_predict` on 14-sensor windows plus 6 conditioning
values. For `--stage all` / `--stage distill` the saved `student.pt` is reloaded
and scored with the student path; otherwise the final teacher is scored. The
score therefore describes the network whose metrics are written next to it.

`_confidence_hook` hands those rows to `conf_integration.split_confidence_score`:

- **Disjoint split.** A permutation of the rows is cut in half. The fit half
  feeds both the temperature/isotonic map and the conformal fit; the score half
  is used only for ECE and coverage. No fitted row is ever scored.
- **Deterministic.** The only randomness is `numpy.random.default_rng(seed)`
  with the run's typed `--seed` (1/2/3). Same seed, rows and config give the same
  split, the same scores and byte-identical `conformal.json`.
- **Small-sample honesty.** A conformal quantile needs `n_fit >= ceil(1/alpha)-1`.
  If the fit half is smaller, `alpha` is raised to `1/(n_fit+1)` and the effective
  value is recorded in both `conformal.json` and the split result. No in-sample
  fallback is substituted.
- **Finite checks.** The hook refuses empty or non-finite logits/errors;
  `fit_conformal` refuses non-finite scores and invalid `alpha`/`tol`;
  `conformal_export` refuses a non-finite payload and validates the frozen step
  map before writing. A refusal leaves no partial export behind.

## Reported fields (`out/metrics.json`)

| field | meaning |
|---|---|
| `ece_source` | explicit provenance of the top-level ECE; always `held-out` (a disjoint source) for this trainer |
| `ece` | the disjoint held-out ECE; identical to `ece_held_out` |
| `ece_held_out` | ECE measured on the score half after the fit-half calibration map |
| `conformal_coverage` | fraction of the score half covered by the exported conformal interval |
| `stages.<name>.ece_source` | `in-sample` — each stage's `ece` is a stage-local diagnostic, not a certified score |

A reader must use `ece_source` plus `ece_held_out` / `conformal_coverage`. The
per-stage `ece` and any legacy `in-sample` number cannot self-certify.

## Exported artifact (`out/conformal.json`)

Written once at the end of `run()` by `conformal_export`. Serialisation is
sorted-key, two-space JSON with a trailing newline, so repeated runs are
byte-identical. Fields:

```json
{
  "method": "split-conformal",
  "alpha": 0.05,
  "tol": 0.15,
  "q_hat": 1.23,
  "n": 2500,
  "coverage": 0.95,
  "half_width": 0.1845,
  "bins": [0.4484304932735426, 1.0],
  "values": [0.0, 1.0]
}
```

`n` is the number of **fit** rows, never the scored half. The interval is
`center ± q_hat * tol` (`half_width`).

### Frozen step map

`frozen_step_map(payload)` validates the edge-facing part of the export:

- exactly two strictly ascending upper edges in `(0, 1]`;
- binary levels `(0, 1)`;
- all edges and values finite.

The boundary is the **inclusive lower edge**: with `p = 1/(1+s)` and
`p_hat = 1/(1+q_hat)`, a prediction is confident iff `p >= p_hat`. The reader
takes the first upper edge strictly greater than `p` and holds the last value
past the end; `apply_confidence` implements this route and agrees with the exact
`s <= q_hat` indicator at equality. The shape (`n` upper edges, `n` values,
monotone levels, `n+1` edges tolerated with the leading lower edge dropped)
matches the step-map contract used by the edge confidence reader in
`src/calibration.rs`. The map is fixed by the fit and is never tuned at runtime.

The edge runtime still loads `calibration.json`; `conformal.json` is the exported
companion artifact for the conformal payload, not a file the current edge binary
loads. No edge-side wiring is claimed here.

## What is deliberately absent

- No gate threshold, reward weight or acceptance limit was changed.
- No A/B, closed-loop result, simulator gain or flight result is claimed.
- `teacher_gates.py` and the EVAL01 limits are untouched; this document describes
  evidence production only.
