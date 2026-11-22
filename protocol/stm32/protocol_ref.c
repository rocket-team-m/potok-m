/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "protocol_ref.h"

#include <string.h>

static void put_u16(uint8_t *buf, size_t off, uint16_t v)
{
    for (size_t i = 0u; i < sizeof(uint16_t); ++i) {
        buf[off + i] = (uint8_t)(v >> (8u * i));
    }
}

static uint16_t get_u16(const uint8_t *buf, size_t off)
{
    uint16_t v = 0u;
    for (size_t i = 0u; i < sizeof(uint16_t); ++i) {
        v |= (uint16_t)((uint16_t)buf[off + i] << (8u * i));
    }
    return v;
}

static void put_u32(uint8_t *buf, size_t off, uint32_t v)
{
    for (size_t i = 0u; i < sizeof(uint32_t); ++i) {
        buf[off + i] = (uint8_t)(v >> (8u * i));
    }
}

static uint32_t get_u32(const uint8_t *buf, size_t off)
{
    uint32_t v = 0u;
    for (size_t i = 0u; i < sizeof(uint32_t); ++i) {
        v |= (uint32_t)((uint32_t)buf[off + i] << (8u * i));
    }
    return v;
}

static void put_u8(uint8_t *buf, size_t off, uint8_t v)
{
    buf[off] = v;
}

static uint8_t get_u8(const uint8_t *buf, size_t off)
{
    return buf[off];
}

static void put_i16(uint8_t *buf, size_t off, int16_t v)
{
    put_u16(buf, off, (uint16_t)v);
}

static int16_t get_i16(const uint8_t *buf, size_t off)
{
    return (int16_t)get_u16(buf, off);
}

static void put_f32(uint8_t *buf, size_t off, float v)
{
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    put_u32(buf, off, bits);
}

static float get_f32(const uint8_t *buf, size_t off)
{
    uint32_t bits = get_u32(buf, off);
    float v;
    memcpy(&v, &bits, sizeof(v));
    return v;
}

uint16_t potok_ref_crc16(const uint8_t *data, size_t len)
{
    uint16_t crc = 0xFFFFu;
    for (size_t i = 0u; i < len; ++i) {
        crc ^= (uint16_t)((uint16_t)data[i] << 8);
        for (int bit = 0; bit < 8; ++bit) {
            crc = (crc & 0x8000u) ? (uint16_t)((uint16_t)(crc << 1) ^ 0x1021u)
                                  : (uint16_t)(crc << 1);
        }
    }
    return (uint16_t)(crc ^ 0x0000u);
}

static int check_header(const uint8_t *buf, size_t len, size_t size,
                        uint16_t magic, size_t crc_off)
{
    if (buf == NULL || len < size) {
        return POTOK_REF_ERR_BUFFER;
    }
    if (get_u16(buf, 0u) != magic) {
        return POTOK_REF_ERR_MAGIC;
    }
    if (get_u16(buf, crc_off) != potok_ref_crc16(buf, crc_off)) {
        return POTOK_REF_ERR_CRC;
    }
    return POTOK_REF_OK;
}

static bool decode_phase(uint8_t raw, PotokRefPhase *phase)
{
    if (raw > (uint8_t)POTOK_REF_PHASE_LANDED) {
        return false;
    }
    *phase = (PotokRefPhase)raw;
    return true;
}

int potok_ref_request_encode(const PotokRefRequest *fr, uint8_t *buf, size_t len)
{
    if (fr == NULL || buf == NULL || len < POTOK_REF_REQUEST_SIZE) {
        return POTOK_REF_ERR_BUFFER;
    }
    put_u16(buf, 0u, (uint16_t)POTOK_REF_REQUEST_MAGIC);
    put_u16(buf, 2u, fr->seq);
    put_u32(buf, 4u, fr->time_us);
    put_u8(buf, 8u, (uint8_t)fr->phase);
    put_u8(buf, 9u, fr->version);
    put_u8(buf, 10u, fr->flags);
    put_u8(buf, 11u, fr->reserved);
    for (size_t i = 0u; i < 3u; ++i) {
        put_f32(buf, 12u + 4u * i, fr->gyro[i]);
    }
    for (size_t i = 0u; i < 3u; ++i) {
        put_f32(buf, 24u + 4u * i, fr->accel[i]);
    }
    for (size_t i = 0u; i < 3u; ++i) {
        put_f32(buf, 36u + 4u * i, fr->mag[i]);
    }
    put_f32(buf, 48u, fr->height);
    put_i16(buf, 52u, fr->pitot_dp);
    put_u16(buf, POTOK_REF_REQUEST_CRC_OFF,
            potok_ref_crc16(buf, POTOK_REF_REQUEST_CRC_OFF));
    return POTOK_REF_OK;
}

int potok_ref_request_decode(const uint8_t *buf, size_t len, PotokRefRequest *fr)
{
    int rc = check_header(buf, len, POTOK_REF_REQUEST_SIZE,
                          (uint16_t)POTOK_REF_REQUEST_MAGIC,
                          POTOK_REF_REQUEST_CRC_OFF);
    if (rc != POTOK_REF_OK) {
        return rc;
    }
    if (fr == NULL) {
        return POTOK_REF_ERR_BUFFER;
    }
    if (!decode_phase(get_u8(buf, 8u), &fr->phase)) {
        return POTOK_REF_ERR_PHASE;
    }
    fr->seq = get_u16(buf, 2u);
    fr->time_us = get_u32(buf, 4u);
    fr->version = get_u8(buf, 9u);
    fr->flags = get_u8(buf, 10u);
    fr->reserved = get_u8(buf, 11u);
    for (size_t i = 0u; i < 3u; ++i) {
        fr->gyro[i] = get_f32(buf, 12u + 4u * i);
    }
    for (size_t i = 0u; i < 3u; ++i) {
        fr->accel[i] = get_f32(buf, 24u + 4u * i);
    }
    for (size_t i = 0u; i < 3u; ++i) {
        fr->mag[i] = get_f32(buf, 36u + 4u * i);
    }
    fr->height = get_f32(buf, 48u);
    fr->pitot_dp = get_i16(buf, 52u);
    if (fr->version != (uint8_t)POTOK_REF_REQUEST_VERSION) {
        return POTOK_REF_ERR_VERSION;
    }
    return POTOK_REF_OK;
}

int potok_ref_reply_encode(const PotokRefReply *fr, uint8_t *buf, size_t len)
{
    if (fr == NULL || buf == NULL || len < POTOK_REF_REPLY_SIZE) {
        return POTOK_REF_ERR_BUFFER;
    }
    put_u16(buf, 0u, (uint16_t)POTOK_REF_REPLY_MAGIC);
    put_u16(buf, 2u, fr->seq);
    put_i16(buf, 4u, fr->g_pitch);
    put_i16(buf, 6u, fr->g_yaw);
    put_u8(buf, 8u, fr->confidence);
    put_u8(buf, 9u, fr->ai_healthy);
    put_u16(buf, 10u, fr->version);
    put_u16(buf, POTOK_REF_REPLY_CRC_OFF,
            potok_ref_crc16(buf, POTOK_REF_REPLY_CRC_OFF));
    return POTOK_REF_OK;
}

int potok_ref_reply_decode(const uint8_t *buf, size_t len, PotokRefReply *fr)
{
    int rc = check_header(buf, len, POTOK_REF_REPLY_SIZE,
                          (uint16_t)POTOK_REF_REPLY_MAGIC,
                          POTOK_REF_REPLY_CRC_OFF);
    if (rc != POTOK_REF_OK) {
        return rc;
    }
    if (fr == NULL) {
        return POTOK_REF_ERR_BUFFER;
    }
    fr->seq = get_u16(buf, 2u);
    fr->g_pitch = get_i16(buf, 4u);
    fr->g_yaw = get_i16(buf, 6u);
    fr->confidence = get_u8(buf, 8u);
    fr->ai_healthy = get_u8(buf, 9u);
    fr->version = get_u16(buf, 10u);
    if (fr->version != (uint16_t)POTOK_REF_REPLY_VERSION) {
        return POTOK_REF_ERR_VERSION;
    }
    return POTOK_REF_OK;
}

int potok_ref_lowlat_request_encode(const PotokRefLowlatRequest *fr,
                                    uint8_t *buf, size_t len)
{
    if (fr == NULL || buf == NULL || len < POTOK_REF_LOWLAT_REQUEST_SIZE) {
        return POTOK_REF_ERR_BUFFER;
    }
    put_u16(buf, 0u, (uint16_t)POTOK_REF_LOWLAT_REQUEST_MAGIC);
    put_u16(buf, 2u, fr->seq);
    put_u32(buf, 4u, fr->time_us);
    put_u8(buf, 8u, (uint8_t)fr->phase);
    put_u8(buf, 9u, fr->version);
    put_u8(buf, 10u, fr->flags);
    put_u8(buf, 11u, fr->reserved);
    for (size_t i = 0u; i < 14u; ++i) {
        put_i16(buf, 12u + 2u * i, fr->features[i]);
    }
    put_u16(buf, POTOK_REF_LOWLAT_REQUEST_CRC_OFF,
            potok_ref_crc16(buf, POTOK_REF_LOWLAT_REQUEST_CRC_OFF));
    return POTOK_REF_OK;
}

int potok_ref_lowlat_request_decode(const uint8_t *buf, size_t len,
                                    PotokRefLowlatRequest *fr)
{
    int rc = check_header(buf, len, POTOK_REF_LOWLAT_REQUEST_SIZE,
                          (uint16_t)POTOK_REF_LOWLAT_REQUEST_MAGIC,
                          POTOK_REF_LOWLAT_REQUEST_CRC_OFF);
    if (rc != POTOK_REF_OK) {
        return rc;
    }
    if (fr == NULL) {
        return POTOK_REF_ERR_BUFFER;
    }
    if (!decode_phase(get_u8(buf, 8u), &fr->phase)) {
        return POTOK_REF_ERR_PHASE;
    }
    fr->seq = get_u16(buf, 2u);
    fr->time_us = get_u32(buf, 4u);
    fr->version = get_u8(buf, 9u);
    fr->flags = get_u8(buf, 10u);
    fr->reserved = get_u8(buf, 11u);
    for (size_t i = 0u; i < 14u; ++i) {
        fr->features[i] = get_i16(buf, 12u + 2u * i);
    }
    if (fr->version != (uint8_t)POTOK_REF_LOWLAT_REQUEST_VERSION) {
        return POTOK_REF_ERR_VERSION;
    }
    return POTOK_REF_OK;
}

int potok_ref_lowlat_reply_encode(const PotokRefLowlatReply *fr,
                                  uint8_t *buf, size_t len)
{
    if (fr == NULL || buf == NULL || len < POTOK_REF_LOWLAT_REPLY_SIZE) {
        return POTOK_REF_ERR_BUFFER;
    }
    put_u16(buf, 0u, (uint16_t)POTOK_REF_LOWLAT_REPLY_MAGIC);
    put_u16(buf, 2u, fr->seq);
    put_i16(buf, 4u, fr->g_pitch);
    put_i16(buf, 6u, fr->g_yaw);
    put_u8(buf, 8u, fr->confidence);
    put_u8(buf, 9u, fr->ai_healthy);
    put_u8(buf, 10u, fr->version);
    put_u8(buf, 11u, fr->compute_us);
    for (size_t i = 0u; i < 2u; ++i) {
        put_u8(buf, 12u + i, fr->reserved[i]);
    }
    put_u16(buf, POTOK_REF_LOWLAT_REPLY_CRC_OFF,
            potok_ref_crc16(buf, POTOK_REF_LOWLAT_REPLY_CRC_OFF));
    return POTOK_REF_OK;
}

int potok_ref_lowlat_reply_decode(const uint8_t *buf, size_t len,
                                  PotokRefLowlatReply *fr)
{
    int rc = check_header(buf, len, POTOK_REF_LOWLAT_REPLY_SIZE,
                          (uint16_t)POTOK_REF_LOWLAT_REPLY_MAGIC,
                          POTOK_REF_LOWLAT_REPLY_CRC_OFF);
    if (rc != POTOK_REF_OK) {
        return rc;
    }
    if (fr == NULL) {
        return POTOK_REF_ERR_BUFFER;
    }
    fr->seq = get_u16(buf, 2u);
    fr->g_pitch = get_i16(buf, 4u);
    fr->g_yaw = get_i16(buf, 6u);
    fr->confidence = get_u8(buf, 8u);
    fr->ai_healthy = get_u8(buf, 9u);
    fr->version = get_u8(buf, 10u);
    fr->compute_us = get_u8(buf, 11u);
    for (size_t i = 0u; i < 2u; ++i) {
        fr->reserved[i] = get_u8(buf, 12u + i);
    }
    if (fr->version != (uint8_t)POTOK_REF_LOWLAT_REPLY_VERSION) {
        return POTOK_REF_ERR_VERSION;
    }
    return POTOK_REF_OK;
}

bool potok_ref_seq_init(PotokRefSeqTracker *tracker, uint16_t window)
{
    if (tracker == NULL || window == 0u || window >= 0x8000u) {
        return false;
    }
    tracker->tx = 0u;
    tracker->rx = 0u;
    tracker->rx_seen = false;
    tracker->window = window;
    return true;
}

uint16_t potok_ref_seq_next_tx(PotokRefSeqTracker *tracker)
{
    uint16_t seq = tracker->tx;
    tracker->tx = (uint16_t)(tracker->tx + 1u);
    return seq;
}

PotokRefSeqStatus potok_ref_seq_observe(PotokRefSeqTracker *tracker, uint16_t seq)
{
    if (!tracker->rx_seen) {
        tracker->rx = seq;
        tracker->rx_seen = true;
        return POTOK_REF_SEQ_OK;
    }
    uint16_t delta = (uint16_t)(seq - tracker->rx);
    if (delta == 0u) {
        return POTOK_REF_SEQ_DUPLICATE;
    }
    if (delta > tracker->window) {
        return POTOK_REF_SEQ_OUT_OF_WINDOW;
    }
    tracker->rx = seq;
    return POTOK_REF_SEQ_OK;
}

bool potok_ref_seq_last_rx(const PotokRefSeqTracker *tracker, uint16_t *out_seq)
{
    if (!tracker->rx_seen) {
        return false;
    }
    if (out_seq != NULL) {
        *out_seq = tracker->rx;
    }
    return true;
}

void potok_ref_seq_reset(PotokRefSeqTracker *tracker)
{
    tracker->tx = 0u;
    tracker->rx = 0u;
    tracker->rx_seen = false;
}
