#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Cross-language golden check for the Potok-M UART wire format.
#
# The four committed golden frames are pinned here once. The STM32 C reference,
# the generated Python codec and the Rust library are each built/evaluated from
# their own source and compared against those bytes, so a silent divergence in
# any one language fails the check.
#
# Modes (default: all): all | self-test | license | yaml | c | python | rust
set -euo pipefail

SELF_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$SELF_DIR/.." && pwd)
WORKFLOW="$ROOT/.github/workflows/cross-language.yml"

GOLD_REQ="4d5034127bf1536502011200cdcccc3dcdcc4cbe33335340c3f51c41000000000000c0bf00004841000050c0000030420080f742bffec67f"
GOLD_REP="505234127d0006ffc8010100c549"
GOLD_LOWREQ="544c34127bf15365020112000100feff03000400fbff06000700f8ff09000a000b00f4ff0d000e000a2b"
GOLD_LOWREP="524c34127d0006ffc801012a0000fd1b"

fail=0
tmp_dir=""

cleanup() {
    if [ -n "$tmp_dir" ]; then
        rm -rf -- "$tmp_dir"
    fi
}
trap cleanup EXIT

scratch() {
    if [ -z "$tmp_dir" ]; then
        tmp_dir=$(mktemp -d)
    fi
    printf '%s' "$tmp_dir"
}

bad() {
    printf 'FAIL %s\n' "$*" >&2
    fail=$((fail + 1))
}

# Returns 1 on mismatch so the self-test can exercise it; callers guard with
# `|| true` because `set -e` is on and `bad` already records the failure.
expect_hex() {
    local label=$1 got=$2 want=$3
    if [ "$got" = "$want" ]; then
        printf 'ok   %s\n' "$label"
        return 0
    fi
    printf 'FAIL %s\n  got  %s\n  want %s\n' "$label" "$got" "$want" >&2
    fail=$((fail + 1))
    return 1
}

self_test() {
    if ! bash -n "$SELF_DIR/check_golden.sh"; then
        bad "self-test: shell syntax invalid"
    fi
    local name length hexval
    for spec in "REQ:56:$GOLD_REQ" "REP:14:$GOLD_REP" \
                "LOWREQ:42:$GOLD_LOWREQ" "LOWREP:16:$GOLD_LOWREP"; do
        name=${spec%%:*}
        length=${spec#*:}; length=${length%%:*}
        hexval=${spec#*:}; hexval=${hexval#*:}
        if [ "${#hexval}" -ne "$((length * 2))" ]; then
            bad "self-test: $name fixture has wrong byte length"
            continue
        fi
        if [[ ! $hexval =~ ^[0-9a-f]+$ ]]; then
            bad "self-test: $name fixture is not lowercase hex"
            continue
        fi
        printf 'ok   self-test %s fixture (%d bytes)\n' "$name" "$length"
    done

    # Negative control: the comparator must actually reject a corrupted frame.
    local corrupt="${GOLD_REQ:0:${#GOLD_REQ}-2}00"
    if [ "$corrupt" = "$GOLD_REQ" ]; then
        corrupt="${GOLD_REQ:0:${#GOLD_REQ}-2}ff"
    fi
    if (expect_hex "self-test corruption" "$corrupt" "$GOLD_REQ" >/dev/null 2>&1); then
        bad "self-test: comparator accepted a corrupted frame"
    else
        printf 'ok   self-test detects a golden mismatch\n'
    fi
}

license() {
    local file line spdx='SPDX-License''-Identifier'
    for file in "$SELF_DIR/check_golden.sh" "$WORKFLOW"; do
        if [ ! -f "$file" ]; then
            bad "missing file: $file"
            continue
        fi
        if ! grep -qxF '# SPDX-License-Identifier: GPL-3.0-or-later' "$file"; then
            bad "missing GPL SPDX header: $file"
        fi
        # Any SPDX identifier present must be this project's own license; no
        # imported code with a different license is carried by these files.
        while IFS= read -r line; do
            case "$line" in
                *"SPDX-License-Identifier: GPL-3.0-or-later"*) ;;
                *) bad "foreign license declaration in $file: $line" ;;
            esac
        done < <(grep -n "$spdx" "$file" || true)
        if grep -q '^[[:space:]]*//[[:space:]]*Copied from\|^[[:space:]]*#[[:space:]]*Copied from' "$file"; then
            bad "imported code marker in $file"
        fi
    done
    if [ "$fail" -eq 0 ]; then
        printf 'ok   GPL headers, no foreign code in owned CI files\n'
    fi
}

yaml() {
    local file
    if ! python3 -c 'import yaml' >/dev/null 2>&1; then
        if ! python3 -m pip install --quiet pyyaml; then
            bad "cannot load a YAML parser"
            return 0
        fi
    fi
    for file in "$ROOT"/.github/workflows/*.yml; do
        if [ ! -f "$file" ]; then
            bad "no workflow files found"
            continue
        fi
        if ! python3 - "$file" <<'PY'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as fh:
    doc = yaml.safe_load(fh)
if not isinstance(doc, dict) or not isinstance(doc.get("jobs"), dict):
    sys.exit("workflow must be a mapping with a jobs mapping")
PY
        then
            bad "invalid workflow YAML: $file"
        else
            printf 'ok   YAML parses: %s\n' "${file#"$ROOT"/}"
        fi
    done
}

write_c_harness() {
    cat > "$1" <<'CEOF'
/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "protocol_ref.h"

#include <stdio.h>
#include <string.h>

static void set_f32_bits(float *dst, uint32_t bits)
{
    memcpy(dst, &bits, sizeof(bits));
}

static void emit(const char *name, const uint8_t *buf, size_t n)
{
    printf("%s=", name);
    for (size_t i = 0; i < n; ++i) {
        printf("%02x", (unsigned)buf[i]);
    }
    putchar('\n');
}

int main(void)
{
    uint8_t buf[64];
    PotokRefRequest req;
    PotokRefReply rep;
    PotokRefLowlatRequest lreq;
    PotokRefLowlatReply lrep;

    memset(&req, 0, sizeof(req));
    req.seq = 0x1234u;
    req.time_us = 1700000123u;
    req.phase = POTOK_REF_PHASE_APOGEE;
    req.version = 1u;
    req.flags = 0x12u;
    req.reserved = 0u;
    set_f32_bits(&req.gyro[0], 0x3dcccccdu);
    set_f32_bits(&req.gyro[1], 0xbe4ccccdu);
    set_f32_bits(&req.gyro[2], 0x40533333u);
    set_f32_bits(&req.accel[0], 0x411cf5c3u);
    set_f32_bits(&req.accel[1], 0x00000000u);
    set_f32_bits(&req.accel[2], 0xbfc00000u);
    set_f32_bits(&req.mag[0], 0x41480000u);
    set_f32_bits(&req.mag[1], 0xc0500000u);
    set_f32_bits(&req.mag[2], 0x42300000u);
    set_f32_bits(&req.height, 0x42f78000u);
    req.pitot_dp = (int16_t)-321;

    memset(&rep, 0, sizeof(rep));
    rep.seq = 0x1234u;
    rep.g_pitch = 125;
    rep.g_yaw = -250;
    rep.confidence = 200u;
    rep.ai_healthy = 1u;
    rep.version = 1u;

    memset(&lreq, 0, sizeof(lreq));
    lreq.seq = 0x1234u;
    lreq.time_us = 1700000123u;
    lreq.phase = POTOK_REF_PHASE_APOGEE;
    lreq.version = 1u;
    lreq.flags = 0x12u;
    lreq.reserved = 0u;
    const int16_t features[14] = {1, -2, 3, 4, -5, 6, 7, -8, 9, 10, 11, -12, 13, 14};
    memcpy(lreq.features, features, sizeof(features));

    memset(&lrep, 0, sizeof(lrep));
    lrep.seq = 0x1234u;
    lrep.g_pitch = 125;
    lrep.g_yaw = -250;
    lrep.confidence = 200u;
    lrep.ai_healthy = 1u;
    lrep.version = 1u;
    lrep.compute_us = 42u;
    lrep.reserved[0] = 0u;
    lrep.reserved[1] = 0u;

    if (potok_ref_request_encode(&req, buf, sizeof(buf)) != POTOK_REF_OK) return 2;
    emit("REQ", buf, POTOK_REF_REQUEST_SIZE);
    if (potok_ref_reply_encode(&rep, buf, sizeof(buf)) != POTOK_REF_OK) return 2;
    emit("REP", buf, POTOK_REF_REPLY_SIZE);
    if (potok_ref_lowlat_request_encode(&lreq, buf, sizeof(buf)) != POTOK_REF_OK) return 2;
    emit("LOWREQ", buf, POTOK_REF_LOWLAT_REQUEST_SIZE);
    if (potok_ref_lowlat_reply_encode(&lrep, buf, sizeof(buf)) != POTOK_REF_OK) return 2;
    emit("LOWREP", buf, POTOK_REF_LOWLAT_REPLY_SIZE);
    return 0;
}
CEOF
}

check_c() {
    local ref="$ROOT/protocol/stm32"
    if [ ! -f "$ref/protocol_ref.c" ] || [ ! -f "$ref/protocol_ref.h" ]; then
        bad "missing STM32 reference sources under $ref"
        return 0
    fi
    local dir
    dir=$(scratch)
    write_c_harness "$dir/golden_harness.c"
    if ! cc -std=c11 -Wall -Wextra -Werror -I"$ref" \
        "$ref/protocol_ref.c" "$dir/golden_harness.c" -o "$dir/golden_harness"; then
        bad "C harness failed to build from stm32/protocol_ref"
        return 0
    fi
    local out
    if ! out=$("$dir/golden_harness"); then
        bad "C harness exited non-zero"
        return 0
    fi
    local c_req="" c_rep="" c_lowreq="" c_lowrep="" key value
    while IFS='=' read -r key value; do
        case "$key" in
            REQ) c_req=$value ;;
            REP) c_rep=$value ;;
            LOWREQ) c_lowreq=$value ;;
            LOWREP) c_lowrep=$value ;;
        esac
    done <<< "$out"
    expect_hex "C request golden bytes" "$c_req" "$GOLD_REQ" || true
    expect_hex "C reply golden bytes" "$c_rep" "$GOLD_REP" || true
    expect_hex "C lowlat request golden bytes" "$c_lowreq" "$GOLD_LOWREQ" || true
    expect_hex "C lowlat reply golden bytes" "$c_lowrep" "$GOLD_LOWREP" || true
}

write_py_harness() {
    cat > "$1" <<'PYEOF'
# SPDX-License-Identifier: GPL-3.0-or-later
import protocol as p

req = p.Request(
    seq=0x1234, time_us=1_700_000_123, phase=p.Phase.APOGEE,
    version=1, flags=0x12, reserved=0,
    gyro=[0.1, -0.2, 3.3], accel=[9.81, 0.0, -1.5],
    mag=[12.5, -3.25, 44.0], height=123.75, pitot_dp=-321,
)
rep = p.Reply(seq=0x1234, g_pitch=125, g_yaw=-250,
              confidence=200, ai_healthy=1, version=1)
lreq = p.LowlatRequest(
    seq=0x1234, time_us=1_700_000_123, phase=p.Phase.APOGEE,
    version=1, flags=0x12, reserved=0,
    features=[1, -2, 3, 4, -5, 6, 7, -8, 9, 10, 11, -12, 13, 14],
)
lrep = p.LowlatReply(
    seq=0x1234, g_pitch=125, g_yaw=-250, confidence=200,
    ai_healthy=1, version=1, compute_us=42, reserved=[0, 0],
)

for name, data in (
    ("REQ", p.encode_request(req)),
    ("REP", p.encode_reply(rep)),
    ("LOWREQ", p.encode_lowlat_request(lreq)),
    ("LOWREP", p.encode_lowlat_reply(lrep)),
):
    print(f"{name}={data.hex()}")
PYEOF
}

check_python() {
    local gen="$ROOT/protocol/generated"
    if [ ! -f "$gen/protocol.py" ]; then
        bad "missing generated Python codec: $gen/protocol.py"
        return 0
    fi
    local dir
    dir=$(scratch)
    write_py_harness "$dir/golden_harness.py"
    local out
    if ! out=$(PYTHONPATH="$gen" python3 "$dir/golden_harness.py"); then
        bad "Python golden harness failed"
        return 0
    fi
    local py_req="" py_rep="" py_lowreq="" py_lowrep="" key value
    while IFS='=' read -r key value; do
        case "$key" in
            REQ) py_req=$value ;;
            REP) py_rep=$value ;;
            LOWREQ) py_lowreq=$value ;;
            LOWREP) py_lowrep=$value ;;
        esac
    done <<< "$out"
    expect_hex "Python request golden bytes" "$py_req" "$GOLD_REQ" || true
    expect_hex "Python reply golden bytes" "$py_rep" "$GOLD_REP" || true
    expect_hex "Python lowlat request golden bytes" "$py_lowreq" "$GOLD_LOWREQ" || true
    expect_hex "Python lowlat reply golden bytes" "$py_lowrep" "$GOLD_LOWREP" || true
}

check_rust() {
    local dir="$ROOT/protocol"
    if [ ! -f "$dir/Cargo.toml" ]; then
        bad "missing protocol workspace: $dir/Cargo.toml"
        return 0
    fi
    local spec target test
    for spec in \
        "protocol_tests:golden_wire_bytes_match_c_and_python" \
        "protocol_tests:committed_golden_fixtures_reencode_byte_identical" \
        "lowlat_tests:lowlat_golden_bytes_match_c_and_python"; do
        target=${spec%%:*}
        test=${spec#*:}
        if (cd "$dir" && cargo test -q -p potok-protocol --test "$target" -- --exact "$test"); then
            printf 'ok   Rust golden test %s\n' "$test"
        else
            bad "Rust golden test failed: $test"
        fi
    done
}

mode=${1:-all}
case "$mode" in
    all)
        self_test
        license
        yaml
        check_c
        check_python
        check_rust
        ;;
    self-test) self_test ;;
    license) license ;;
    yaml) yaml ;;
    c) check_c ;;
    python) check_python ;;
    rust) check_rust ;;
    *)
        printf 'usage: %s [all|self-test|license|yaml|c|python|rust]\n' "$0" >&2
        exit 2
        ;;
esac

printf 'check_golden %s: ' "$mode"
if [ "$fail" -eq 0 ]; then
    printf 'OK\n'
else
    printf 'FAILED (%d)\n' "$fail"
fi
exit "$fail"
