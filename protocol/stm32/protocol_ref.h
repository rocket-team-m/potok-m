/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * Potok-M UART protocol — STM32 C11 reference.
 *
 * Wire contract is defined in ../../uart.yaml; generated/protocol.h is the
 * generated consumer view. This file is an independent, self-contained
 * reference (it intentionally shares no code with the generator output) so a
 * host test can compare the two byte-for-byte.
 *
 * No dynamic allocation: every buffer is caller-provided and all state lives
 * in caller-owned structs. Only fixed-size arrays are used.
 */
#ifndef POTOK_PROTOCOL_REF_H
#define POTOK_PROTOCOL_REF_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Error codes — same numeric contract as generated/protocol.h. */
enum {
    POTOK_REF_OK = 0,
    POTOK_REF_ERR_BUFFER = -1,
    POTOK_REF_ERR_MAGIC = -2,
    POTOK_REF_ERR_CRC = -3,
    POTOK_REF_ERR_PHASE = -4,
    POTOK_REF_ERR_VERSION = -5
};

/* Frame geometry (bytes). All frames are little-endian and fixed size. */
#define POTOK_REF_REQUEST_SIZE      56u
#define POTOK_REF_REQUEST_MAGIC     0x504Du
#define POTOK_REF_REQUEST_VERSION   1u
#define POTOK_REF_REQUEST_CRC_OFF   54u

#define POTOK_REF_REPLY_SIZE        14u
#define POTOK_REF_REPLY_MAGIC       0x5250u
#define POTOK_REF_REPLY_VERSION     1u
#define POTOK_REF_REPLY_CRC_OFF     12u

#define POTOK_REF_LOWLAT_REQUEST_SIZE    42u
#define POTOK_REF_LOWLAT_REQUEST_MAGIC   0x4C54u
#define POTOK_REF_LOWLAT_REQUEST_VERSION 1u
#define POTOK_REF_LOWLAT_REQUEST_CRC_OFF 40u

#define POTOK_REF_LOWLAT_REPLY_SIZE      16u
#define POTOK_REF_LOWLAT_REPLY_MAGIC     0x4C52u
#define POTOK_REF_LOWLAT_REPLY_VERSION   1u
#define POTOK_REF_LOWLAT_REPLY_CRC_OFF   14u

typedef enum {
    POTOK_REF_PHASE_BURN = 0,
    POTOK_REF_PHASE_COAST = 1,
    POTOK_REF_PHASE_APOGEE = 2,
    POTOK_REF_PHASE_DESCENT = 3,
    POTOK_REF_PHASE_LANDED = 4
} PotokRefPhase;

typedef struct {
    uint16_t seq;
    uint32_t time_us;
    PotokRefPhase phase;
    uint8_t version;
    uint8_t flags;
    uint8_t reserved;
    float gyro[3];
    float accel[3];
    float mag[3];
    float height;
    int16_t pitot_dp;
} PotokRefRequest;

typedef struct {
    uint16_t seq;
    int16_t g_pitch;
    int16_t g_yaw;
    uint8_t confidence;
    uint8_t ai_healthy;
    uint16_t version;
} PotokRefReply;

typedef struct {
    uint16_t seq;
    uint32_t time_us;
    PotokRefPhase phase;
    uint8_t version;
    uint8_t flags;
    uint8_t reserved;
    int16_t features[14];
} PotokRefLowlatRequest;

typedef struct {
    uint16_t seq;
    int16_t g_pitch;
    int16_t g_yaw;
    uint8_t confidence;
    uint8_t ai_healthy;
    uint8_t version;
    uint8_t compute_us;
    uint8_t reserved[2];
} PotokRefLowlatReply;

/* CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, no reflection, xorout 0.
 * Covers every byte before the trailing 2-byte CRC field. */
uint16_t potok_ref_crc16(const uint8_t *data, size_t len);

/* Encode writes exactly SIZE bytes into buf; a larger buffer is fine.
 * Decode validates in the order magic -> CRC -> phase -> version. */
int potok_ref_request_encode(const PotokRefRequest *fr, uint8_t *buf, size_t len);
int potok_ref_request_decode(const uint8_t *buf, size_t len, PotokRefRequest *fr);
int potok_ref_reply_encode(const PotokRefReply *fr, uint8_t *buf, size_t len);
int potok_ref_reply_decode(const uint8_t *buf, size_t len, PotokRefReply *fr);
int potok_ref_lowlat_request_encode(const PotokRefLowlatRequest *fr, uint8_t *buf, size_t len);
int potok_ref_lowlat_request_decode(const uint8_t *buf, size_t len, PotokRefLowlatRequest *fr);
int potok_ref_lowlat_reply_encode(const PotokRefLowlatReply *fr, uint8_t *buf, size_t len);
int potok_ref_lowlat_reply_decode(const uint8_t *buf, size_t len, PotokRefLowlatReply *fr);

/* Wrap-safe sequence tracking, mirroring the Rust SeqTracker. */
#define POTOK_REF_SEQ_DEFAULT_WINDOW 64u

typedef enum {
    POTOK_REF_SEQ_OK = 0,
    POTOK_REF_SEQ_DUPLICATE = 1,
    POTOK_REF_SEQ_OUT_OF_WINDOW = 2
} PotokRefSeqStatus;

typedef struct {
    uint16_t tx;
    uint16_t rx;
    bool rx_seen;
    uint16_t window;
} PotokRefSeqTracker;

/* Returns false (and leaves the tracker unusable) when window is not 1..=0x7fff. */
bool potok_ref_seq_init(PotokRefSeqTracker *tracker, uint16_t window);
uint16_t potok_ref_seq_next_tx(PotokRefSeqTracker *tracker);
PotokRefSeqStatus potok_ref_seq_observe(PotokRefSeqTracker *tracker, uint16_t seq);
bool potok_ref_seq_last_rx(const PotokRefSeqTracker *tracker, uint16_t *out_seq);
void potok_ref_seq_reset(PotokRefSeqTracker *tracker);

#ifdef __cplusplus
}
#endif
#endif /* POTOK_PROTOCOL_REF_H */
