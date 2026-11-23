# OTA01 — secure firmware/config update design (design doc only)

**Date:** 2026-11-23 · **Repo:** `/home/sehaxe/potok-m` @ `main`. **No code, no hardware, no network, no subagents, no push.**

## Scope honored

Two files only:

- `docs/OTA_SECURITY.md` — the new design document.
- `.bulba/night/OTA01.md` — this report.

No source, config, fixture, test or other document touched. The working tree already contained unrelated modifications and untracked files before this task; none of them were staged or changed.

## What the design specifies

`docs/OTA_SECURITY.md` defines a firmware/config update path that reuses the existing `GROUND_STATION.md` wire rules (CRC-16/CCITT-FALSE, HMAC-SHA256 8-byte tag, seq u16/window 64, op_id whitelist, PREFLIGHT/IN_FLIGHT/LANDED gate) instead of inventing a second protocol.

1. **Channels:** full image is wired USB-UART only; LoRa carries manifest/status/ACK and bounded config patches. Three separate anchors — Ed25519 public key in the bootloader, per-flight HMAC session key for commands, 8-byte board hardware ID for image binding.
2. **Manifest:** magic/format, payload kind, monotonic u64 version, hardware ID, target slot, size, SHA-256, 200 × CRC-16 per 1 KiB chunk, signer key id, Ed25519 signature. Total **528 B**. Signature covers every acted-on field; CRC table is a torn-write check, not a cryptographic one. Order: magic → hwid → version → size → signature → SHA-256 → chunk CRCs, fail closed and COUNTED.
3. **Bootloader:** two slots A/B, redundant seq+CRC state records, states `STEADY → PENDING → ARM → TRIAL(3 attempts) → COMMITTED | ROLLBACK`. Swap is one atomic state-record flip, never an image copy; the known-good slot is never erased while a candidate is unproven. Committed-version counter gives software anti-rollback.
4. **Radio whitelist:** five fixed command ids (`0x40`..`0x44`) under the unchanged verify pipeline; no firmware bytes over radio, no write-byte/jump/execute command, config patches schema-validated into an inactive bank.
5. **No arbitrary code execution:** write-protected verifier, signed boot, no raw flash/execute commands, debug-lock and MPU noted as hardware-dependent and unverified.
6. **Self-test checklist:** bootloader-level and trial-boot checks plus HIL power-cut, rollback, downgrade, cross-device and tamper tests — all marked `NOT RUN`.
7. **Threat model:** T1–T12 table plus an explicit list of what is **not** protected without hardware secure boot — bootloader replacement, hardware anti-rollback, key confidentiality, glitching, image confidentiality, buggy root of trust.
8. **Bandwidth:** 200 KiB = 1,638,400 bits. LoRa 5–10 kbit/s → 5.5–2.7 min ideal, ~9.1–3.9 min useful; 1 % duty cycle → 4.5–9.1 h; existing ≤1 Hz `CMD` cadence → ~2.4 h (8,533 commands). USB-UART 115,200 Bd → 17.8 s gross; 921,600 Bd → 2.22 s gross, i.e. 184×/92× faster than LoRa. Manifest 528 B → 0.84 s ideal at 5 kbit/s. Conclusion: image wired only, radio for manifest/status/patches only.

All numbers are labelled `SPEC`, `CALCULATED` or `NOT RUN`; no measurement is presented as one.

## Verification

- `git diff --check` on the two paths — clean.
- `git status --short` before and after — only the two intended paths changed by this task.
- Both files carry the `GPL-3.0-or-later` statement; no source code was added.
- Local commit on `main`; nothing pushed.

## Honest limits

No board, HIL or signing tool exists yet. Ed25519 verify cost on the MCU, real LoRa throughput/PER, duty-cycle region, debug-lock level and hardware anti-rollback availability are all open items (§9 of the design). The document is a contract to implement and test, not evidence that the path works.
