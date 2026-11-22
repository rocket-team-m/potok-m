# Night task PROTO01 — cross-language golden CI

**Status: done.** Local commit on `main` (see `git log -1 --grep PROTO01`). No push. No subagents, no GPU.

## Scope

Created exactly two code files plus this report:

- `.github/workflows/cross-language.yml`
- `scripts/check_golden.sh`
- `.bulba/night/PROTO01.md`

No existing source file was edited. No dependency, manifest or Cargo change.

**Path note:** the task named repo `/home/sehaxe/potok-m-protocol`, which does not exist on this box. The protocol project lives inside the monorepo at `/home/sehaxe/potok-m/protocol` (git root `/home/sehaxe/potok-m`), so the two target paths were created relative to that git root. The C harness is built from `protocol/stm32/protocol_ref.{c,h}`, as specified.

## What was added

`scripts/check_golden.sh` pins the four committed golden frames once (request 56 B, reply 14 B, lowlat request 42 B, lowlat reply 16 B) and offers six modes: `all`, `self-test`, `license`, `yaml`, `c`, `python`, `rust`.

- **C** — writes a harness to a temp dir, compiles it with `cc -std=c11 -Wall -Wextra -Werror` against `protocol/stm32/protocol_ref.c` + its header (no new C file in the repo), runs it and compares all four emitted frame hexes to the pinned golden bytes. Float fields are set by exact IEEE bit patterns so the check does not depend on float-literal rounding.
- **Python** — imports the generated `protocol/generated/protocol.py` codec, re-encodes the same four sample frames and compares the emitted hexes.
- **Rust** — runs the three existing golden tests in `potok-protocol` by exact name (`golden_wire_bytes_match_c_and_python`, `committed_golden_fixtures_reencode_byte_identical`, `lowlat_golden_bytes_match_c_and_python`).
- **self-test** — checks `bash -n` syntax, golden fixture lengths/hex validity, and a negative control proving the comparator rejects a corrupted frame.
- **license** — requires an `SPDX-License-Identifier: GPL-3.0-or-later` first-line header in both owned files, rejects any other SPDX identifier in them, and rejects imported-code markers. No foreign/third-party code was copied in; the C harness is original and built only from this repo's reference source.
- **yaml** — parses every `.github/workflows/*.yml` with PyYAML and asserts a `jobs` mapping.

Both new files carry `SPDX-License-Identifier: GPL-3.0-or-later` as their first line (after the shebang in the shell script). The workflow has one visible step per mode, `permissions: contents: read`, `actions/checkout@v5` and `dtolnay/rust-toolchain@stable`, matching the repo's existing workflow style.

## Verification

- `bash scripts/check_golden.sh` (all modes) → exit 0: 16 byte comparisons green, 3 Rust golden tests green, all 4 workflows parse.
- Each standalone mode (`self-test`, `license`, `yaml`, `c`, `python`, `rust`) → exit 0.
- Negative control: copy of the script with `GOLD_REP` corrupted in one byte → `FAIL C reply golden bytes`, `FAIL Python reply golden bytes`, `check_golden all: FAILED (2)`, exit 2. The gate is not a no-op.
- Baseline before the change: `cargo test -p potok-protocol golden` → 3 passed.

## Not done / caveats

- No push, per instruction.
- The workflow was not run on GitHub; only local validation of every step was performed. `actionlint` is not installed on this box.
- The `yaml` mode installs PyYAML with pip only when `import yaml` fails; on the GitHub runner it is expected to be present already.
- Unrelated modified/untracked files from other night agents were left untouched; only the paths above were staged.
