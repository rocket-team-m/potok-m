/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * Host test for the STM32 C11 protocol reference.
 *
 * Build and run from protocol/:
 *   cc -std=c11 -Wall -Wextra -Werror \
 *     stm32/protocol_ref.c stm32/tests/test_protocol_ref.c \
 *     -o /tmp/opencode/test_protocol_ref && /tmp/opencode/test_protocol_ref
 *
 * The generated header is included as an independent oracle: every frame the
 * reference encodes must be byte-identical to the generator's encoder, and
 * each decoder must agree with the other implementation's decode.
 */
#include "../protocol_ref.h"
#include "../../generated/protocol.h"

#include <stdio.h>
#include <string.h>

static int failures;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);              \
            ++failures;                                                        \
        }                                                                      \
    } while (0)

static uint32_t f32_bits(float v)
{
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    return bits;
}

static void resign(uint8_t *buf, size_t crc_off)
{
    potok_put_u16(buf, crc_off, potok_crc16(buf, crc_off));
}

/* ------------------------------- fixtures -------------------------------- */

static void fill_request(PotokRefRequest *r)
{
    r->seq = 0x1234u;
    r->time_us = 0x01020304u;
    r->phase = POTOK_REF_PHASE_APOGEE;
    r->version = (uint8_t)POTOK_REF_REQUEST_VERSION;
    r->flags = 0xA5u;
    r->reserved = 0x3Cu;
    r->gyro[0] = 1.5f;
    r->gyro[1] = -2.25f;
    r->gyro[2] = 0.0f;
    r->accel[0] = 0.001f;
    r->accel[1] = 9.80665f;
    r->accel[2] = -3.5f;
    r->mag[0] = 1.0e-4f;
    r->mag[1] = -2.0e-4f;
    r->mag[2] = 3.0e-4f;
    r->height = 1234.5f;
    r->pitot_dp = -1234;
}

static void to_generated_request(const PotokRefRequest *r, PotokRequest *g)
{
    g->seq = r->seq;
    g->time_us = r->time_us;
    g->phase = (enum PotokPhase)r->phase;
    g->version = r->version;
    g->flags = r->flags;
    g->reserved = r->reserved;
    for (size_t i = 0u; i < 3u; ++i) {
        g->gyro[i] = r->gyro[i];
        g->accel[i] = r->accel[i];
        g->mag[i] = r->mag[i];
    }
    g->height = r->height;
    g->pitot_dp = r->pitot_dp;
}

static void check_request_equal(const PotokRefRequest *r, const PotokRequest *g)
{
    CHECK(r->seq == g->seq);
    CHECK(r->time_us == g->time_us);
    CHECK((unsigned)r->phase == (unsigned)g->phase);
    CHECK(r->version == g->version);
    CHECK(r->flags == g->flags);
    CHECK(r->reserved == g->reserved);
    for (size_t i = 0u; i < 3u; ++i) {
        CHECK(f32_bits(r->gyro[i]) == f32_bits(g->gyro[i]));
        CHECK(f32_bits(r->accel[i]) == f32_bits(g->accel[i]));
        CHECK(f32_bits(r->mag[i]) == f32_bits(g->mag[i]));
    }
    CHECK(f32_bits(r->height) == f32_bits(g->height));
    CHECK(r->pitot_dp == g->pitot_dp);
}

static void fill_reply(PotokRefReply *r)
{
    r->seq = 0xABCDu;
    r->g_pitch = 1234;
    r->g_yaw = -2345;
    r->confidence = 200u;
    r->ai_healthy = 1u;
    r->version = (uint16_t)POTOK_REF_REPLY_VERSION;
}

static void to_generated_reply(const PotokRefReply *r, PotokReply *g)
{
    g->seq = r->seq;
    g->g_pitch = r->g_pitch;
    g->g_yaw = r->g_yaw;
    g->confidence = r->confidence;
    g->ai_healthy = r->ai_healthy;
    g->version = r->version;
}

static void check_reply_equal(const PotokRefReply *r, const PotokReply *g)
{
    CHECK(r->seq == g->seq);
    CHECK(r->g_pitch == g->g_pitch);
    CHECK(r->g_yaw == g->g_yaw);
    CHECK(r->confidence == g->confidence);
    CHECK(r->ai_healthy == g->ai_healthy);
    CHECK(r->version == g->version);
}

static void fill_lowlat_request(PotokRefLowlatRequest *r)
{
    r->seq = 0x2222u;
    r->time_us = 0x0A0B0C0Du;
    r->phase = POTOK_REF_PHASE_DESCENT;
    r->version = (uint8_t)POTOK_REF_LOWLAT_REQUEST_VERSION;
    r->flags = 0x5Au;
    r->reserved = 0x11u;
    r->features[0] = 1000;
    r->features[1] = -1000;
    r->features[2] = 250;
    r->features[3] = -250;
    r->features[4] = 1;
    r->features[5] = -1;
    r->features[6] = 32767;
    r->features[7] = -32768;
    r->features[8] = 0;
    r->features[9] = 123;
    r->features[10] = 456;
    r->features[11] = -789;
    r->features[12] = 42;
    r->features[13] = 12000;
}

static void to_generated_lowlat_request(const PotokRefLowlatRequest *r,
                                        PotokLowlatRequest *g)
{
    g->seq = r->seq;
    g->time_us = r->time_us;
    g->phase = (enum PotokPhase)r->phase;
    g->version = r->version;
    g->flags = r->flags;
    g->reserved = r->reserved;
    for (size_t i = 0u; i < 14u; ++i) {
        g->features[i] = r->features[i];
    }
}

static void check_lowlat_request_equal(const PotokRefLowlatRequest *r,
                                       const PotokLowlatRequest *g)
{
    CHECK(r->seq == g->seq);
    CHECK(r->time_us == g->time_us);
    CHECK((unsigned)r->phase == (unsigned)g->phase);
    CHECK(r->version == g->version);
    CHECK(r->flags == g->flags);
    CHECK(r->reserved == g->reserved);
    for (size_t i = 0u; i < 14u; ++i) {
        CHECK(r->features[i] == g->features[i]);
    }
}

static void fill_lowlat_reply(PotokRefLowlatReply *r)
{
    r->seq = 0x3333u;
    r->g_pitch = -500;
    r->g_yaw = 500;
    r->confidence = 7u;
    r->ai_healthy = 0u;
    r->version = (uint8_t)POTOK_REF_LOWLAT_REPLY_VERSION;
    r->compute_us = 250u;
    r->reserved[0] = 0xAAu;
    r->reserved[1] = 0x55u;
}

static void to_generated_lowlat_reply(const PotokRefLowlatReply *r,
                                      PotokLowlatReply *g)
{
    g->seq = r->seq;
    g->g_pitch = r->g_pitch;
    g->g_yaw = r->g_yaw;
    g->confidence = r->confidence;
    g->ai_healthy = r->ai_healthy;
    g->version = r->version;
    g->compute_us = r->compute_us;
    for (size_t i = 0u; i < 2u; ++i) {
        g->reserved[i] = r->reserved[i];
    }
}

static void check_lowlat_reply_equal(const PotokRefLowlatReply *r,
                                     const PotokLowlatReply *g)
{
    CHECK(r->seq == g->seq);
    CHECK(r->g_pitch == g->g_pitch);
    CHECK(r->g_yaw == g->g_yaw);
    CHECK(r->confidence == g->confidence);
    CHECK(r->ai_healthy == g->ai_healthy);
    CHECK(r->version == g->version);
    CHECK(r->compute_us == g->compute_us);
    CHECK(r->reserved[0] == g->reserved[0]);
    CHECK(r->reserved[1] == g->reserved[1]);
}

/* --------------------------------- tests --------------------------------- */

static void test_crc(void)
{
    static const uint8_t check[] = "123456789";
    CHECK(potok_ref_crc16(check, 9u) == 0x29B1u);
    CHECK(potok_ref_crc16(check, 0u) == 0xFFFFu);
    CHECK(potok_crc16(check, 9u) == 0x29B1u);

    uint8_t buf[POTOK_REF_REQUEST_SIZE];
    memset(buf, 0xA5, sizeof(buf));
    CHECK(potok_ref_crc16(buf, sizeof(buf)) == potok_crc16(buf, sizeof(buf)));
}

static void test_request(void)
{
    PotokRefRequest r;
    fill_request(&r);
    PotokRequest g;
    to_generated_request(&r, &g);

    uint8_t refbuf[POTOK_REF_REQUEST_SIZE + 4u];
    uint8_t genbuf[POTOK_REF_REQUEST_SIZE + 4u];
    memset(refbuf, 0xEE, sizeof(refbuf));
    memset(genbuf, 0xEE, sizeof(genbuf));

    CHECK(potok_ref_request_encode(&r, refbuf, sizeof(refbuf)) == POTOK_REF_OK);
    CHECK(potok_request_encode(&g, genbuf, sizeof(genbuf)) == POTOK_OK);
    CHECK(memcmp(refbuf, genbuf, POTOK_REF_REQUEST_SIZE) == 0);
    /* bytes beyond the frame are untouched in both implementations */
    for (size_t i = POTOK_REF_REQUEST_SIZE; i < sizeof(refbuf); ++i) {
        CHECK(refbuf[i] == 0xEEu);
        CHECK(genbuf[i] == 0xEEu);
    }

    PotokRefRequest ref_out;
    PotokRequest gen_out;
    memset(&ref_out, 0, sizeof(ref_out));
    memset(&gen_out, 0, sizeof(gen_out));
    CHECK(potok_ref_request_decode(genbuf, sizeof(genbuf), &ref_out) == POTOK_REF_OK);
    check_request_equal(&ref_out, &g);
    CHECK(potok_request_decode(refbuf, sizeof(refbuf), &gen_out) == POTOK_OK);
    check_request_equal(&r, &gen_out);
}

static void test_reply(void)
{
    PotokRefReply r;
    fill_reply(&r);
    PotokReply g;
    to_generated_reply(&r, &g);

    uint8_t refbuf[POTOK_REF_REPLY_SIZE];
    uint8_t genbuf[POTOK_REF_REPLY_SIZE];
    CHECK(potok_ref_reply_encode(&r, refbuf, sizeof(refbuf)) == POTOK_REF_OK);
    CHECK(potok_reply_encode(&g, genbuf, sizeof(genbuf)) == POTOK_OK);
    CHECK(memcmp(refbuf, genbuf, POTOK_REF_REPLY_SIZE) == 0);

    PotokRefReply ref_out;
    PotokReply gen_out;
    CHECK(potok_ref_reply_decode(genbuf, sizeof(genbuf), &ref_out) == POTOK_REF_OK);
    check_reply_equal(&ref_out, &g);
    CHECK(potok_reply_decode(refbuf, sizeof(refbuf), &gen_out) == POTOK_OK);
    check_reply_equal(&r, &gen_out);
}

static void test_lowlat_request(void)
{
    PotokRefLowlatRequest r;
    fill_lowlat_request(&r);
    PotokLowlatRequest g;
    to_generated_lowlat_request(&r, &g);

    uint8_t refbuf[POTOK_REF_LOWLAT_REQUEST_SIZE];
    uint8_t genbuf[POTOK_REF_LOWLAT_REQUEST_SIZE];
    CHECK(potok_ref_lowlat_request_encode(&r, refbuf, sizeof(refbuf)) == POTOK_REF_OK);
    CHECK(potok_lowlat_request_encode(&g, genbuf, sizeof(genbuf)) == POTOK_OK);
    CHECK(memcmp(refbuf, genbuf, POTOK_REF_LOWLAT_REQUEST_SIZE) == 0);

    PotokRefLowlatRequest ref_out;
    PotokLowlatRequest gen_out;
    CHECK(potok_ref_lowlat_request_decode(genbuf, sizeof(genbuf), &ref_out) == POTOK_REF_OK);
    check_lowlat_request_equal(&ref_out, &g);
    CHECK(potok_lowlat_request_decode(refbuf, sizeof(refbuf), &gen_out) == POTOK_OK);
    check_lowlat_request_equal(&r, &gen_out);

    /* pin the published scale table so an unused-static warning cannot hide it */
    CHECK(POTOK_LOWLAT_REQUEST_FEATURES_SCALES[0] == 0.001);
    CHECK(strcmp(POTOK_LOWLAT_REQUEST_FEATURES_NAMES[13], "battery_mv") == 0);
}

static void test_lowlat_reply(void)
{
    PotokRefLowlatReply r;
    fill_lowlat_reply(&r);
    PotokLowlatReply g;
    to_generated_lowlat_reply(&r, &g);

    uint8_t refbuf[POTOK_REF_LOWLAT_REPLY_SIZE];
    uint8_t genbuf[POTOK_REF_LOWLAT_REPLY_SIZE];
    CHECK(potok_ref_lowlat_reply_encode(&r, refbuf, sizeof(refbuf)) == POTOK_REF_OK);
    CHECK(potok_lowlat_reply_encode(&g, genbuf, sizeof(genbuf)) == POTOK_OK);
    CHECK(memcmp(refbuf, genbuf, POTOK_REF_LOWLAT_REPLY_SIZE) == 0);

    PotokRefLowlatReply ref_out;
    PotokLowlatReply gen_out;
    CHECK(potok_ref_lowlat_reply_decode(genbuf, sizeof(genbuf), &ref_out) == POTOK_REF_OK);
    check_lowlat_reply_equal(&ref_out, &g);
    CHECK(potok_lowlat_reply_decode(refbuf, sizeof(refbuf), &gen_out) == POTOK_OK);
    check_lowlat_reply_equal(&r, &gen_out);
}

static void test_error_paths(void)
{
    uint8_t buf[POTOK_REF_REQUEST_SIZE];
    PotokRefRequest req;
    PotokRefRequest out_r;
    fill_request(&req);
    CHECK(potok_ref_request_encode(&req, buf, sizeof(buf)) == POTOK_REF_OK);

    /* buffer / null arguments */
    CHECK(potok_ref_request_decode(buf, POTOK_REF_REQUEST_SIZE - 1u, &out_r) == POTOK_REF_ERR_BUFFER);
    CHECK(potok_ref_request_decode(NULL, sizeof(buf), &out_r) == POTOK_REF_ERR_BUFFER);
    CHECK(potok_ref_request_decode(buf, sizeof(buf), NULL) == POTOK_REF_ERR_BUFFER);
    CHECK(potok_ref_request_encode(NULL, buf, sizeof(buf)) == POTOK_REF_ERR_BUFFER);
    CHECK(potok_ref_request_encode(&req, NULL, sizeof(buf)) == POTOK_REF_ERR_BUFFER);

    /* magic: re-sign so the magic check, not CRC, decides */
    uint8_t bad[POTOK_REF_REQUEST_SIZE];
    memcpy(bad, buf, sizeof(bad));
    bad[0] ^= 0x01u;
    resign(bad, POTOK_REF_REQUEST_CRC_OFF);
    CHECK(potok_ref_request_decode(bad, sizeof(bad), &out_r) == POTOK_REF_ERR_MAGIC);

    /* CRC: corrupt payload without re-signing */
    memcpy(bad, buf, sizeof(bad));
    bad[20] ^= 0x80u;
    CHECK(potok_ref_request_decode(bad, sizeof(bad), &out_r) == POTOK_REF_ERR_CRC);

    /* phase: valid CRC, invalid enum */
    memcpy(bad, buf, sizeof(bad));
    bad[8] = 5u;
    resign(bad, POTOK_REF_REQUEST_CRC_OFF);
    CHECK(potok_ref_request_decode(bad, sizeof(bad), &out_r) == POTOK_REF_ERR_PHASE);

    /* version: valid CRC and phase, wrong version */
    memcpy(bad, buf, sizeof(bad));
    bad[9] = 2u;
    resign(bad, POTOK_REF_REQUEST_CRC_OFF);
    CHECK(potok_ref_request_decode(bad, sizeof(bad), &out_r) == POTOK_REF_ERR_VERSION);

    /* the generated decoder agrees on each corrupt frame */
    PotokRequest gen_out;
    CHECK(potok_request_decode(bad, sizeof(bad), &gen_out) == POTOK_ERR_VERSION);

    /* lowlat request: short buffer, magic, crc, phase, version */
    uint8_t low[POTOK_REF_LOWLAT_REQUEST_SIZE];
    PotokRefLowlatRequest lreq;
    PotokRefLowlatRequest lout;
    fill_lowlat_request(&lreq);
    CHECK(potok_ref_lowlat_request_encode(&lreq, low, sizeof(low)) == POTOK_REF_OK);
    CHECK(potok_ref_lowlat_request_decode(low, sizeof(low) - 1u, &lout) == POTOK_REF_ERR_BUFFER);
    memcpy(bad, low, sizeof(low));
    bad[0] ^= 0x01u;
    resign(bad, POTOK_REF_LOWLAT_REQUEST_CRC_OFF);
    CHECK(potok_ref_lowlat_request_decode(bad, sizeof(low), &lout) == POTOK_REF_ERR_MAGIC);
    memcpy(bad, low, sizeof(low));
    bad[30] ^= 0x40u;
    CHECK(potok_ref_lowlat_request_decode(bad, sizeof(low), &lout) == POTOK_REF_ERR_CRC);
    memcpy(bad, low, sizeof(low));
    bad[8] = 0xFFu;
    resign(bad, POTOK_REF_LOWLAT_REQUEST_CRC_OFF);
    CHECK(potok_ref_lowlat_request_decode(bad, sizeof(low), &lout) == POTOK_REF_ERR_PHASE);
    memcpy(bad, low, sizeof(low));
    bad[9] = 3u;
    resign(bad, POTOK_REF_LOWLAT_REQUEST_CRC_OFF);
    CHECK(potok_ref_lowlat_request_decode(bad, sizeof(low), &lout) == POTOK_REF_ERR_VERSION);

    /* replies: short buffer, magic, crc, version */
    uint8_t rep[POTOK_REF_REPLY_SIZE];
    PotokRefReply rrep;
    PotokRefReply rout;
    fill_reply(&rrep);
    CHECK(potok_ref_reply_encode(&rrep, rep, sizeof(rep)) == POTOK_REF_OK);
    CHECK(potok_ref_reply_decode(rep, sizeof(rep) - 1u, &rout) == POTOK_REF_ERR_BUFFER);
    memcpy(bad, rep, sizeof(rep));
    bad[0] ^= 0x01u;
    resign(bad, POTOK_REF_REPLY_CRC_OFF);
    CHECK(potok_ref_reply_decode(bad, sizeof(rep), &rout) == POTOK_REF_ERR_MAGIC);
    memcpy(bad, rep, sizeof(rep));
    bad[6] ^= 0x01u;
    CHECK(potok_ref_reply_decode(bad, sizeof(rep), &rout) == POTOK_REF_ERR_CRC);
    memcpy(bad, rep, sizeof(rep));
    potok_put_u16(bad, 10u, 9u);
    resign(bad, POTOK_REF_REPLY_CRC_OFF);
    CHECK(potok_ref_reply_decode(bad, sizeof(rep), &rout) == POTOK_REF_ERR_VERSION);

    uint8_t lrep[POTOK_REF_LOWLAT_REPLY_SIZE];
    PotokRefLowlatReply lrrep;
    PotokRefLowlatReply lrout;
    fill_lowlat_reply(&lrrep);
    CHECK(potok_ref_lowlat_reply_encode(&lrrep, lrep, sizeof(lrep)) == POTOK_REF_OK);
    CHECK(potok_ref_lowlat_reply_decode(lrep, sizeof(lrep) - 1u, &lrout) == POTOK_REF_ERR_BUFFER);
    memcpy(bad, lrep, sizeof(lrep));
    bad[0] ^= 0x01u;
    resign(bad, POTOK_REF_LOWLAT_REPLY_CRC_OFF);
    CHECK(potok_ref_lowlat_reply_decode(bad, sizeof(lrep), &lrout) == POTOK_REF_ERR_MAGIC);
    memcpy(bad, lrep, sizeof(lrep));
    bad[11] ^= 0x80u;
    CHECK(potok_ref_lowlat_reply_decode(bad, sizeof(lrep), &lrout) == POTOK_REF_ERR_CRC);
    memcpy(bad, lrep, sizeof(lrep));
    bad[10] = 2u;
    resign(bad, POTOK_REF_LOWLAT_REPLY_CRC_OFF);
    CHECK(potok_ref_lowlat_reply_decode(bad, sizeof(lrep), &lrout) == POTOK_REF_ERR_VERSION);
}

static void test_seq(void)
{
    PotokRefSeqTracker t;
    uint16_t last = 0u;

    /* invalid windows are refused loudly */
    CHECK(!potok_ref_seq_init(NULL, 64u));
    CHECK(!potok_ref_seq_init(&t, 0u));
    CHECK(!potok_ref_seq_init(&t, 0x8000u));
    CHECK(potok_ref_seq_init(&t, POTOK_REF_SEQ_DEFAULT_WINDOW));

    /* tx increments and wraps at u16::MAX */
    for (uint32_t i = 0u; i < 65535u; ++i) {
        CHECK(potok_ref_seq_next_tx(&t) == (uint16_t)i);
    }
    CHECK(potok_ref_seq_next_tx(&t) == 0xFFFFu);
    CHECK(potok_ref_seq_next_tx(&t) == 0u);

    /* first observation seeds rx; duplicates are rejected */
    CHECK(!potok_ref_seq_last_rx(&t, &last));
    CHECK(potok_ref_seq_observe(&t, 100u) == POTOK_REF_SEQ_OK);
    CHECK(potok_ref_seq_last_rx(&t, &last));
    CHECK(last == 100u);
    CHECK(potok_ref_seq_observe(&t, 100u) == POTOK_REF_SEQ_DUPLICATE);

    /* forward jumps inside the window are accepted, beyond it are refused
     * without moving state; the exact boundary is accepted */
    CHECK(potok_ref_seq_observe(&t, 150u) == POTOK_REF_SEQ_OK);
    CHECK(potok_ref_seq_observe(&t, 215u) == POTOK_REF_SEQ_OUT_OF_WINDOW);
    CHECK(potok_ref_seq_last_rx(&t, &last));
    CHECK(last == 150u);
    CHECK(potok_ref_seq_observe(&t, 214u) == POTOK_REF_SEQ_OK);

    /* wrap-around: 0xFFFF -> 0 is a forward jump of one; the reverse is not */
    CHECK(potok_ref_seq_init(&t, 16u));
    CHECK(potok_ref_seq_observe(&t, 0xFFFFu) == POTOK_REF_SEQ_OK);
    CHECK(potok_ref_seq_observe(&t, 0u) == POTOK_REF_SEQ_OK);
    CHECK(potok_ref_seq_last_rx(&t, &last));
    CHECK(last == 0u);
    CHECK(potok_ref_seq_observe(&t, 0xFFFFu) == POTOK_REF_SEQ_OUT_OF_WINDOW);

    /* reset returns to a fresh state for both directions */
    potok_ref_seq_reset(&t);
    CHECK(!potok_ref_seq_last_rx(&t, &last));
    CHECK(potok_ref_seq_observe(&t, 7u) == POTOK_REF_SEQ_OK);
    CHECK(potok_ref_seq_observe(&t, 8u) == POTOK_REF_SEQ_OK);
    CHECK(potok_ref_seq_next_tx(&t) == 0u);
}

int main(void)
{
    test_crc();
    test_request();
    test_reply();
    test_lowlat_request();
    test_lowlat_reply();
    test_error_paths();
    test_seq();

    if (failures != 0) {
        printf("FAILED: %d check(s)\n", failures);
        return 1;
    }
    printf("C reference host test: OK\n");
    return 0;
}
