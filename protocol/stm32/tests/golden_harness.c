/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * Self-contained cross-language golden harness for the Potok-M UART wire
 * format (standalone potok-m-protocol repository).
 *
 * Reuses the generated consumer codec in ../../generated/protocol.h; no
 * imported code and no external harness. Each of the four committed golden
 * frames is checked twice: built from fields and encoded, then decoded from
 * the pinned bytes and re-encoded. CRC-16/CCITT-FALSE is pinned too.
 *
 * Build and run from the repository root:
 *   cc -std=c11 -Wall -Wextra -Werror \
 *     stm32/tests/golden_harness.c -o /tmp/opencode/golden_harness && \
 *     /tmp/opencode/golden_harness
 */
#include "../../generated/protocol.h"

#include <stdio.h>
#include <string.h>

static const char GOLD_REQ[] =
    "4d5034127bf1536502011200cdcccc3dcdcc4cbe33335340c3f51c41000000000000c0bf00004841000050c0000030420080f742bffec67f";
static const char GOLD_REP[] = "505234127d0006ffc8010100c549";
static const char GOLD_LOWREQ[] =
    "544c34127bf15365020112000100feff03000400fbff06000700f8ff09000a000b00f4ff0d000e000a2b";
static const char GOLD_LOWREP[] = "524c34127d0006ffc801012a0000fd1b";

static int failures;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);              \
            ++failures;                                                        \
        }                                                                      \
    } while (0)

static void set_f32_bits(float *dst, uint32_t bits)
{
    memcpy(dst, &bits, sizeof(bits));
}

static int hex_nibble(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
}

static int parse_golden(const char *hex, uint8_t *out, size_t cap, size_t *len)
{
    size_t n = strlen(hex);
    if ((n & 1u) != 0u || n / 2u > cap) return -1;
    for (size_t i = 0u; i < n / 2u; ++i) {
        int hi = hex_nibble(hex[2u * i]);
        int lo = hex_nibble(hex[2u * i + 1u]);
        if (hi < 0 || lo < 0) return -1;
        out[i] = (uint8_t)(((unsigned)hi << 4) | (unsigned)lo);
    }
    *len = n / 2u;
    return 0;
}

/* ------------------------------- fixtures -------------------------------- */

static void fill_request(PotokRequest *r)
{
    memset(r, 0, sizeof(*r));
    r->seq = 0x1234u;
    r->time_us = 1700000123u;
    r->phase = POTOK_PHASE_APOGEE;
    r->version = POTOK_REQUEST_VERSION;
    r->flags = 0x12u;
    set_f32_bits(&r->gyro[0], 0x3dcccccdu);
    set_f32_bits(&r->gyro[1], 0xbe4ccccdu);
    set_f32_bits(&r->gyro[2], 0x40533333u);
    set_f32_bits(&r->accel[0], 0x411cf5c3u);
    set_f32_bits(&r->accel[2], 0xbfc00000u);
    set_f32_bits(&r->mag[0], 0x41480000u);
    set_f32_bits(&r->mag[1], 0xc0500000u);
    set_f32_bits(&r->mag[2], 0x42300000u);
    set_f32_bits(&r->height, 0x42f78000u);
    r->pitot_dp = (int16_t)-321;
}

static void fill_reply(PotokReply *r)
{
    memset(r, 0, sizeof(*r));
    r->seq = 0x1234u;
    r->g_pitch = 125;
    r->g_yaw = -250;
    r->confidence = 200u;
    r->ai_healthy = 1u;
    r->version = POTOK_REPLY_VERSION;
}

static void fill_lowlat_request(PotokLowlatRequest *r)
{
    static const int16_t features[14] = {
        1, -2, 3, 4, -5, 6, 7, -8, 9, 10, 11, -12, 13, 14
    };
    memset(r, 0, sizeof(*r));
    r->seq = 0x1234u;
    r->time_us = 1700000123u;
    r->phase = POTOK_PHASE_APOGEE;
    r->version = POTOK_LOWLAT_REQUEST_VERSION;
    r->flags = 0x12u;
    memcpy(r->features, features, sizeof(features));
}

static void fill_lowlat_reply(PotokLowlatReply *r)
{
    memset(r, 0, sizeof(*r));
    r->seq = 0x1234u;
    r->g_pitch = 125;
    r->g_yaw = -250;
    r->confidence = 200u;
    r->ai_healthy = 1u;
    r->version = POTOK_LOWLAT_REPLY_VERSION;
    r->compute_us = 42u;
}

/* --------------------------------- tests --------------------------------- */

static void test_crc(void)
{
    static const uint8_t check[] = "123456789";
    CHECK(potok_crc16(check, 9u) == 0x29B1u);
    CHECK(potok_crc16(check, 0u) == 0xFFFFu);
    /* Pin the generated table to the spec polynomial (0x1021, CCITT-FALSE). */
    CHECK(POTOK_CRC16_TABLE[0] == 0x0000u);
    CHECK(POTOK_CRC16_TABLE[1] == 0x1021u);
    CHECK(POTOK_CRC16_TABLE[255] == 0x1EF0u);
}

static void test_encode(void)
{
    uint8_t want[64], got[64];
    size_t n = 0u;
    PotokRequest req;
    PotokReply rep;
    PotokLowlatRequest lreq;
    PotokLowlatReply lrep;

    fill_request(&req);
    CHECK(parse_golden(GOLD_REQ, want, sizeof(want), &n) == 0);
    CHECK(n == POTOK_REQUEST_SIZE);
    CHECK(potok_request_encode(&req, got, sizeof(got)) == POTOK_OK);
    CHECK(memcmp(got, want, POTOK_REQUEST_SIZE) == 0);

    fill_reply(&rep);
    CHECK(parse_golden(GOLD_REP, want, sizeof(want), &n) == 0);
    CHECK(n == POTOK_REPLY_SIZE);
    CHECK(potok_reply_encode(&rep, got, sizeof(got)) == POTOK_OK);
    CHECK(memcmp(got, want, POTOK_REPLY_SIZE) == 0);

    fill_lowlat_request(&lreq);
    CHECK(parse_golden(GOLD_LOWREQ, want, sizeof(want), &n) == 0);
    CHECK(n == POTOK_LOWLAT_REQUEST_SIZE);
    CHECK(potok_lowlat_request_encode(&lreq, got, sizeof(got)) == POTOK_OK);
    CHECK(memcmp(got, want, POTOK_LOWLAT_REQUEST_SIZE) == 0);

    fill_lowlat_reply(&lrep);
    CHECK(parse_golden(GOLD_LOWREP, want, sizeof(want), &n) == 0);
    CHECK(n == POTOK_LOWLAT_REPLY_SIZE);
    CHECK(potok_lowlat_reply_encode(&lrep, got, sizeof(got)) == POTOK_OK);
    CHECK(memcmp(got, want, POTOK_LOWLAT_REPLY_SIZE) == 0);
}

static void roundtrip(const uint8_t *golden, size_t n, int kind)
{
    uint8_t out[64];
    int rc = -1;

    switch (kind) {
    case 0: {
        PotokRequest f;
        rc = potok_request_decode(golden, n, &f);
        if (rc == POTOK_OK) rc = potok_request_encode(&f, out, n);
        break;
    }
    case 1: {
        PotokReply f;
        rc = potok_reply_decode(golden, n, &f);
        if (rc == POTOK_OK) rc = potok_reply_encode(&f, out, n);
        break;
    }
    case 2: {
        PotokLowlatRequest f;
        rc = potok_lowlat_request_decode(golden, n, &f);
        if (rc == POTOK_OK) rc = potok_lowlat_request_encode(&f, out, n);
        break;
    }
    case 3: {
        PotokLowlatReply f;
        rc = potok_lowlat_reply_decode(golden, n, &f);
        if (rc == POTOK_OK) rc = potok_lowlat_reply_encode(&f, out, n);
        break;
    }
    default:
        return;
    }
    CHECK(rc == POTOK_OK);
    if (rc == POTOK_OK) CHECK(memcmp(out, golden, n) == 0);
}

static void test_roundtrip(void)
{
    uint8_t want[64];
    size_t n = 0u;

    CHECK(parse_golden(GOLD_REQ, want, sizeof(want), &n) == 0);
    roundtrip(want, n, 0);
    CHECK(parse_golden(GOLD_REP, want, sizeof(want), &n) == 0);
    roundtrip(want, n, 1);
    CHECK(parse_golden(GOLD_LOWREQ, want, sizeof(want), &n) == 0);
    roundtrip(want, n, 2);
    CHECK(parse_golden(GOLD_LOWREP, want, sizeof(want), &n) == 0);
    roundtrip(want, n, 3);
}

int main(void)
{
    /* Consume the generated header's static tables: no unused-object warning. */
    CHECK(POTOK_LOWLAT_REQUEST_FEATURES_SCALES[0] == 0.001);
    CHECK(strcmp(POTOK_LOWLAT_REQUEST_FEATURES_NAMES[13], "battery_mv") == 0);

    test_crc();
    test_encode();
    test_roundtrip();

    if (failures != 0) {
        printf("C golden: FAILED (%d)\n", failures);
        return 1;
    }
    printf("C golden: OK\n");
    return 0;
}
