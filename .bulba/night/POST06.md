# POST06 — TCN receptive-field arithmetic in SPEC_SMART.md

**Date:** 2026-11-23
**Repo:** `/home/sehaxe/potok-m`
**Scope:** documentation only. No code, config, fixture or test edited. No subagents. No push.

## Finding

`SPEC_SMART.md:34-35` put a 50-frame teacher window next to a 4-layer causal TCN
(kernel `k=3`, standard dilations `2^(l−1)` = 1/2/4/8). That juxtaposition asserts
the stack covers the window. It does not. For a causal TCN:

```
RF = 1 + (k−1)(2^L − 1)
k=3, L=4  →  RF = 1 + 2·15  = 31 frames   (not 50)
k=3, L=5  →  RF = 1 + 2·31  = 63 frames
k=5, L=4  →  RF = 1 + 4·15  = 61 frames
```

So the specified 4-layer stack sees 31 frames of the 50-frame input. Two
weight-shared loop passes refine the latent but do not widen the convolution's
receptive field: each pass sees the same 31 frames.

## Change

`SPEC_SMART.md:35-40` now states the formula and the 31-frame result explicitly,
says the stack does not cover the window, and lists the two minimal ways to reach
50 frames (`L=5` → 63, `k=5` → 61) or replacing the mixer. The note on the loop
passes records that they are not a substitute for receptive field. Model names
(`LoopWM`, `LoopWM-lite`, `LRS-TCN`, `RMS-TCN`, `StatefulTCN`) are unchanged; the
4-layer draft spec is kept and flagged, not silently replaced with a new
architecture.

## Files checked, no change needed

- `SPEC_STUDENT.md` — "окно 50" at line 33 is the input size / MAC estimate for
  the teacher, not a TCN coverage claim; dispatch at lines 52/55 uses the last
  50 frames as input.
- `ai/docs/MODEL_CARD.md` — teacher is described as a LoopBlock with causal
  self-attention + FFN (line 17), which sees the full window; the student's
  `StatefulTCN` uses dilations 1/2/4/8/16 (line 15), not the 4-layer draft stack.
- `ai/README.md` — line 40 says only the teacher sees a 50-frame window; line 42
  is a generic TCN description. Neither asserts 4-layer coverage.

No derived claim about the 4-layer TCN's coverage appears in those three files.

## Verification

- Arithmetic re-derived by hand from the TCN receptive-field formula (above).
- Only `SPEC_SMART.md` and this report were touched; no code edits, no chat
  provenance added to any document.
- Local commit on `main`; no push.
