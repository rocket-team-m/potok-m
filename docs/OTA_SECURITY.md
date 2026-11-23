# OTA security design — Potok-M

**Status:** design only. No bootloader, no command handler, no signing tool is implemented by this document.
**Date:** 2026-11-23. **License:** design document; any implementation derived from it is `GPL-3.0-or-later`, third-party components keep their own licenses.
**Claim labels:** `SPEC` = contract this design defines · `CALCULATED` = arithmetic from stated inputs · `NOT RUN` = no hardware, no HIL, no measurement.

This document complements the Russian-language `docs/GROUND_STATION.md`, which is the authority for the existing six frame types, the CRC/HMAC/seq rules and the phase gate. It adds a firmware/config update path without changing any of those wire definitions.

---

## 0. Scope and hard boundary

1. **Bulk image transport is wired USB-UART only.** A full firmware image (200 KB in the budget below) is never sent over LoRa. LoRa carries only: the manifest (or its hash), status/ACK, and bounded configuration patches. §8 shows why this is not a preference but arithmetic.
2. **The update path is receive → verify → write inactive slot → trial boot → confirm.** There is no "run this bytes" operation anywhere.
3. **This is firmware-level trust, not hardware secure boot.** The root of trust is an Ed25519 public key in a write-protected bootloader region. §7 states exactly what an attacker with physical/debug access can still do.
4. No pyro, servo, motor-arm, phase-write, AI-gain or log-erase function is reachable from the update path. This is the same boundary as `GROUND_STATION.md` §3.6/§3.7, not a new one.

---

## 1. Channels and trust anchors

| Channel | Direction | Carries | Gate |
| --- | --- | --- | --- |
| USB-UART (wired) | host → board | full signed firmware image, full signed config blob | physical presence; verified by bootloader |
| LoRa 915 MHz | ground → board | signed manifest/hash, status request, bounded config patch, trial confirm | `PREFLIGHT` / `LANDED` only; HMAC + seq + CRC + whitelist |
| LoRa 915 MHz | board → ground | `STA` / `ACK` with slot, version, counters, result | existing telemetry budget (`GROUND_STATION.md` §2.3) |
| Bootloader (on chip) | — | verification, slot write, state records, swap | executes from write-protected region |

Three anchors, deliberately separate in lifetime and purpose:

- **Ed25519 public key** (`SPEC`, embedded in bootloader, read-only): authenticity of a manifest and its payload. The private signing key is offline and never present on the vehicle or the ground station. A key-id field allows rotation across a small set of embedded public keys.
- **HMAC-SHA256 session key** (`SPEC`, 32 B, `HKDF-SHA256(PSK, salt=flight_id || node_uid, info="potok-ground-v1")`): authenticity of radio commands, already specified in `GROUND_STATION.md` §3. Short-lived, per flight, symmetric. It does **not** sign firmware.
- **Board hardware ID / node UID** (`SPEC`, 8 B): binds a manifest to one airframe. An image signed for a different mass/CG/servo configuration cannot be flashed here.

The anchors do not overlap: a stolen radio PSK does not let an attacker sign firmware, and a stolen signing key does not let an attacker inject a command (they still need the HMAC key and a valid seq).

---

## 2. Manifest

### 2.1 Fields

Binary, little-endian, canonical fixed-width encoding so the signed bytes have exactly one representation. `SPEC`.

| Field | Width | Rule |
| --- | ---: | --- |
| magic + format version | 4 + 2 B | unknown version → loud refuse, no fallback |
| payload kind | 1 B | 0 = firmware image, 1 = config blob, 2 = bounded config patch |
| version | u64 | monotonic; must be strictly greater than committed version (§3.4) |
| hardware ID | 8 B | must equal the board UID; mismatch → loud refuse |
| target slot | 1 B | 0 = A, 1 = B — the slot to write. The *other* slot is the rollback slot |
| payload size | u32 | exact byte count; must fit the slot and match the received byte count |
| SHA-256 | 32 B | digest of the whole payload, computed over the payload only |
| chunk CRC table | 200 × 2 B | CRC-16/CCITT-FALSE (poly `0x1021`, init `0xFFFF`) per 1 KiB chunk; 204,800 B / 1,024 = 200 chunks |
| signer key id | 4 B | identifies which embedded Ed25519 public key verifies |
| Ed25519 signature | 64 B | over every field above, in this order |

Manifest size with 1 KiB chunks: 4+2+1+8+8+1+4+32+400+4+64 = **528 B** (`CALCULATED`). At 5 kbit/s that is 528×8/5,000 = **0.84 s** of ideal airtime, ~1.3 s with ACK/gap overhead — small enough to announce over LoRa occasionally. A 256 B chunk size would give 800 CRCs and a ~1.7 KB manifest; it improves resume granularity but triples radio cost, so 1 KiB is the chosen rung.

### 2.2 What the signature must cover

Every field the bootloader acts on: version, hardware ID, target slot, payload size, SHA-256, the full chunk-CRC table, payload kind and key id. If any were left unsigned — especially target slot or version — an attacker could flip it without invalidating the signature. The CRC table is *not* a cryptographic check; it is a torn-write/resume check that the signature happens to protect from substitution.

The signature does **not** cover the payload bytes directly; SHA-256 does, and the digest is signed. This keeps the signed object small.

### 2.3 Verification order

Same shape as `GROUND_STATION.md` §3.3 — fail closed, COUNTED, loud reason:

1. magic + format version → 2. hardware ID equality → 3. version monotonicity vs committed state → 4. size ≤ slot capacity and = received length → 5. Ed25519 verify (constant-time compare) → 6. SHA-256 of assembled payload → 7. per-chunk CRC after write and after read-back.

Any mismatch: refuse the whole update, leave the active slot untouched, increment a named counter, emit a reason string. There is no "signature bad but CRC good, continue" path. Ed25519 verification on this MCU is **NOT RUN** — the bootloader reserves one-time verify budget; measure it at the HIL gate before claiming feasibility.

---

## 3. Bootloader: A/B slots and power-safe swap

### 3.1 Layout

Two equally sized application slots A and B plus a small state area and a write-protected bootloader region. The bootloader is the only code that writes application flash or state records. Vector table and reset handler live at the start of each slot so either slot can be booted by a base-address remap (`SPEC`).

### 3.2 State records

Two redundant records in the state area, each carrying `seq u32`, state, target slot, boot-attempt count, committed version and a CRC. On reset the bootloader selects the highest-`seq` record with a valid CRC. If neither is valid it boots slot A **only after full verification**; if A also fails, it stays in the bootloader and signals `FAIL` — it never guesses a slot (`SPEC`).

### 3.3 State machine and power safety

| State | Meaning | Entry condition |
| --- | --- | --- |
| `STEADY(slot S, version V)` | S active and known-good; other slot is rollback | reset / successful commit |
| `PENDING(candidate C)` | C fully written, SHA-256 and all 200 CRCs pass | end of verified wired transfer |
| `ARM(C)` | next boot will try C | operator authorises trial |
| `TRIAL(C, attempts=3)` | C booted, awaiting health evidence | bootloader jumped to C |
| `COMMITTED(C, Vc)` | C active and known-good; S becomes rollback slot | health evidence + confirm |
| `ROLLBACK(S)` | return to previous known-good | health timeout / crash / explicit rollback |

Power-safe rules (`SPEC`, the whole point of A/B):

1. **Never erase or overwrite the active known-good slot** until a candidate is fully written, fully verified and a `PENDING` record is committed.
2. **The swap is one atomic state-record flip**, not a copy of images. There is no interval with two half-written slots. This is why the design defines no image-copy routine at all.
3. A candidate becomes bootable only after `PENDING` + `ARM`; a power cut before that leaves `STEADY(S)` intact. A cut after it leaves a *verified* candidate — booting it or retrying are both safe.
4. Trial boot has a boot-attempt counter and a hardware watchdog armed before the jump. The application must write a `HEALTHY` marker after its self-test (§6). Crash, hang, or three exhausted attempts → automatic `ROLLBACK(S)`.
5. The old slot is never auto-erased. It stays as the rollback slot until a later successful update promotes another slot; thus at least one verified image is always present.

### 3.4 Anti-rollback

Ed25519 proves integrity, not freshness: a correctly signed old image is still valid. The bootloader therefore keeps a **committed-version counter** in the redundant state records and refuses any manifest with version ≤ that counter. A "factory reset" that lowers it requires physical presence and signed authorisation, and is itself logged (`SPEC`). The counter is software state — see §7 for what that means against physical access.

---

## 4. Radio command whitelist (HMAC + seq + CRC)

Reuse the existing `CMD` frame and check order from `GROUND_STATION.md` §2.2/§3. The verification pipeline is unchanged: `type → CRC-16 → HMAC-SHA256 (8-byte tag, constant-time) → seq (u16, window 64, replay reject) → phase gate → whitelist → dispatch`. Any failure is a COUNTED refusal with reason and log line (`GROUND_STATION.md` §3.3).

OTA adds a small, fixed set of command ids; there is no generic "flash write" command (`SPEC`):

| cmd id | name | phase | effect |
| --- | --- | --- | --- |
| `0x40` | `OTA_MANIFEST` | PREFLIGHT, LANDED | accept/announce manifest or its hash; read-only until verified |
| `0x41` | `OTA_STATUS` | PREFLIGHT, LANDED | report active slot, version, state, counters |
| `0x42` | `OTA_PATCH_APPLY` | PREFLIGHT only | apply a schema-validated config patch ≤ 256 B |
| `0x43` | `OTA_CONFIRM` | PREFLIGHT, LANDED | confirm a trial boot; authenticated, wired preferred |
| `0x44` | `OTA_ROLLBACK` | PREFLIGHT, LANDED | force return to rollback slot |

Rules:

- **Everything not in the whitelist is a loud reject**, never "ignored silently". `GROUND_STATION.md`'s forbidden set (actuators, pyro, motor arm, Phase write, AI gain, log erase) stays forbidden.
- **No radio command carries firmware bytes.** `OTA_PATCH_APPLY` is config only, size-bounded, and requires a signed manifest whose SHA-256 covers the patch. The patch is schema-validated and applied to an inactive config bank with its own CRC, so a rejected patch leaves no partial state.
- Radio `OTA_CONFIRM` is weaker than a wired confirmation because it is remote; both require the phase gate and HMAC. Use wired for production commits; radio confirm is for post-landing recovery.
- All OTA commands stay within the ≤1 Hz `CMD` budget and the ~52 % airtime occupancy from `GROUND_STATION.md` §2.3. Bulk announcement uses a dedicated preflight window, never during flight.

---

## 5. No arbitrary code execution

1. No command writes to bootloader, vector table or executable flash except through the verified receive → verify → slot-write path (§2.3, §3).
2. No "write byte at address", "jump to address", "load and run" or raw flash-write command exists — there is nothing to whitelist for them.
3. Config patches are interpreted against a fixed schema with range checks; unknown keys and out-of-range values are loud refuses. They target non-executable config banks.
4. The bootloader's public key and verification code are in a write-protected region, so an application — even a compromised one — cannot rewrite the verifier.
5. The image is booted only after Ed25519 verification. Debug-port locking (RDP) and MPU non-executable data regions are recommended if the MCU provides them; both are **NOT RUN / unverified** and are hardware-dependent, not assumed.
6. This is defence against remote and USB-host attackers and against buggy images. It is not defence against someone with physical debug access (§7).

---

## 6. Post-update self-test checklist

Run before `COMMITTED`; all results are SPEC and **NOT RUN** until a board exists. Critical checks run in the bootloader before handing control; the rest run in the trial boot. Status is reported through `STA`/`ACK` and the append-only command log — no secrets in logs.

**Bootloader level (before jump):**

- [ ] State records: highest valid `seq`, CRC good, active slot identified.
- [ ] Candidate slot SHA-256 re-read from flash matches manifest.
- [ ] Version > committed counter; hardware ID equals board UID; payload kind expected.
- [ ] Reset vector and initial stack pointer inside the candidate slot / SRAM bounds.
- [ ] Config banks validate (schema, bounds, CRC).
- [ ] Power-cut recovery: interrupt at write, `PENDING`, `ARM`, `TRIAL`, `COMMIT` and confirm recovery to a valid slot at each.

**Trial boot (application):**

- [ ] Clock/PLL and peripheral init complete; watchdog configured and fed.
- [ ] Telemetry golden frame: encode/decode one `TLM`, CRC matches known vector (`GROUND_STATION.md` §2.2).
- [ ] Wired UART echo/command round-trip if cable present.
- [ ] Radio refusal test on the bench, pyro disconnected: bad CRC, bad HMAC, stale seq, unknown cmd id, and any command in `IN_FLIGHT` are each refused with the correct reason counter.
- [ ] Interlock check: actuator/pyro/motor-arm commands locked by the physical switch; no command moves a servo in this test. Use existing `0x12 SELF_TEST` semantics.
- [ ] Health marker write path and boot-attempt counter work; simulate one crash to confirm rollback.
- [ ] Sensor flags report real status or explicit `NOT READY`, never fabricated values.
- [ ] Bundled model/config hash matches the manifest side field.
- [ ] Downgrade test: correctly signed old-version manifest → refused.
- [ ] Cross-device test: valid image with wrong hardware ID → refused.
- [ ] Tamper tests: flip signature, SHA-256, a chunk CRC, version, target slot → all refused; recompute CRC without re-signing → still refused.
- [ ] Airtime check: OTA commands stay within the §4 budget and never displace telemetry.

**Before flight:** HIL power-cut and rollback tests signed off, radio region/duty-cycle chosen, signing key provenance recorded. Until then the whole checklist is `NOT RUN`.

---

## 7. Threat model and what is NOT protected

`CALCULATED`/`SPEC` analysis; no attack was executed.

| # | Adversary / event | Capability | Protected? | Mechanism | Residual |
| --- | --- | --- | --- | --- | --- |
| T1 | Remote radio attacker | transmit on 915 MHz | yes | HMAC + seq window + whitelist + phase gate; no image over radio | jamming not addressed; leaked PSK forges commands until rotation |
| T2 | Passive eavesdropper | listen only | yes for secrets | no key on wire; signature/digests are public | config patches may reveal settings (not secret) |
| T3 | Replay attacker | record and resend | yes | version monotonicity (image), seq window 64 (commands) | — |
| T4 | Malicious USB host | arbitrary bytes over UART | yes | Ed25519 + hardware ID; host cannot sign | can still send junk → verification DoS, COUNTED |
| T5 | Corrupt transfer / power loss | bit flips, cut power | yes | SHA-256, 200 chunk CRCs, A/B state machine, trial counter | — |
| T6 | Buggy (not malicious) image | crashes after boot | yes | trial boot + boot attempts + rollback | — |
| T7 | Downgrade with signed old image | valid old signature | software-only | committed-version counter (§3.4) | rewritable with physical access |
| T8 | Physical attacker with SWD/JTAG | full flash/RAM read-write | **no** | none in this design | can replace bootloader/key, rewind counter, extract PSK |
| T9 | Fault/glitch injection | voltage, clock, EM | **no** | none | can glitch a verify into accepting bad input |
| T10 | Signing-key compromise | offline private key | no on-device help | key id allows rotation | operational: keep key offline, rotate via signed bootloader |
| T11 | Supply-chain / build compromise | alter source or toolchain | out of scope | reproducible builds needed | — |
| T12 | Slot exhaustion / boot-counter DoS | repeated bad updates | detected | loud COUNTED refusals | operator recovers physically |

### Explicitly NOT protected without hardware secure boot

This design provides **integrity and authenticity against remote attackers, a malicious USB host, replay, accidents and buggy images**. It does **not** provide a hardware root of trust. Specifically not protected:

- **Bootloader replacement.** The Ed25519 public key and verifier live in flash. With physical/debug access an attacker overwrites both. This is firmware-level trust by construction, not measured boot from an immutable ROM.
- **Hardware anti-rollback.** The committed-version counter is software state in flash; physical access can rewind it, so T7 protection is conditional on debug ports being locked.
- **Key confidentiality.** The HMAC PSK and derived session keys reside in flash/RAM and are extractable with physical access. Secure element or MCU read-out protection is required.
- **Fault injection / glitching.** Software comparisons can be skipped by voltage or clock glitches; no hardware glitch detection is assumed.
- **Confidentiality of the image.** Not a goal; own code is GPL and the image is not secret.
- **Buggy bootloader.** The root of trust is trusted by fiat; its correctness needs its own review and test gate.

Closing these requires hardware support — immutable boot ROM or one-time-programmable key/anti-rollback fuses, read-out protection, and a secure element — plus HIL verification. None of that is claimed here.

---

## 8. Bandwidth arithmetic: 200 KB over LoRa vs serial

Inputs (`CALCULATED`): payload = 200 KiB = 204,800 B = 1,638,400 bits. LoRa link rate 5–10 kbit/s as budgeted. USB-UART 8N1 = 10 bit per byte. "Useful" assumes ~60–70 % after framing, ACK, inter-frame gaps and retransmissions; no measurement.

| Link | Gross | Useful | 200 KiB time |
| --- | ---: | ---: | ---: |
| LoRa, 5 kbit/s | 5,000 bit/s | ~3,000 bit/s | 1,638,400 / 5,000 = 327.7 s ≈ **5.5 min** ideal; ~9.1 min useful |
| LoRa, 10 kbit/s | 10,000 bit/s | ~7,000 bit/s | 163.8 s ≈ **2.7 min** ideal; ~3.9 min useful |
| LoRa with 1 % duty cycle | 50–100 bit/s effective | — | 1,638,400 / 50 = 32,768 s ≈ **9.1 h**; /100 = **4.5 h** |
| LoRa via existing `CMD` cadence (24 B, ≤1 Hz) | 24 B/cmd | — | 204,800 / 24 = 8,533 commands ≈ **2.4 h** at 1 Hz |
| USB-UART 115,200 Bd (8N1) | 11,520 B/s | ~9,200 B/s | 204,800 / 11,520 = **17.8 s** gross; ~22 s useful |
| USB-UART 921,600 Bd (8N1) | 92,160 B/s | ~74,000 B/s | 204,800 / 92,160 = **2.22 s** gross; ~2.8 s useful |

Ratios: 921,600 Bd is **184×** 5 kbit/s and **92×** 10 kbit/s; even 115,200 Bd is 23× / 11.5×. The wired link also does not consume the flight telemetry band or a duty-cycle budget.

Small-config case, for contrast: a 256 B patch is 2,048 bits → 0.41 s ideal at 5 kbit/s, ~0.6 s with overhead; a 528 B manifest → 0.84 s ideal, ~1.3 s with overhead. Both fit an occasional PREFLIGHT/LANDED window; neither justifies putting the image on the radio.

**Conclusion:** full image = wired only. LoRa = manifest, status, ACK and ≤256 B config patches. This is a protocol boundary, not a speed optimisation.

---

## 9. Open items before any hardware gate

All are `NOT RUN`:

1. Measure Ed25519 verify cost on the target MCU; confirm it fits the one-time bootloader budget.
2. Measure real LoRa throughput, PER and duty-cycle limit for the chosen region; replace §8's assumed rates with readings.
3. Select and review a `no_std` Ed25519 implementation and pin its provenance (no external-reference-free "verified" claim).
4. HIL power-cut matrix from §6 and the rollback test.
5. Decide debug-port lock / RDP level and whether hardware anti-rollback fuses exist on the chosen part; until then §7's T7/T8 remain open.
6. Define the schema and signed factory-reset token for the monotonic version counter.

---

Собственный код, реализующий этот дизайн, — `GPL-3.0-or-later`; подробности — в `LICENSE` и `docs/src/content/docs/project/licenses/`.
