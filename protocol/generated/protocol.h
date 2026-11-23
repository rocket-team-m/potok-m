/* Generated from uart.yaml by `cargo run -p potok-generator`. DO NOT EDIT. */
#ifndef POTOK_POTOK_UART_H
#define POTOK_POTOK_UART_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    POTOK_OK = 0,
    POTOK_ERR_BUFFER = -1,
    POTOK_ERR_MAGIC = -2,
    POTOK_ERR_CRC = -3,
    POTOK_ERR_PHASE = -4,
    POTOK_ERR_VERSION = -5
};

enum PotokPhase {
    POTOK_PHASE_BURN = 0,
    POTOK_PHASE_COAST = 1,
    POTOK_PHASE_APOGEE = 2,
    POTOK_PHASE_DESCENT = 3,
    POTOK_PHASE_LANDED = 4
};

enum PotokFlags {
    POTOK_FLAG_IMU_OK = (1u << 1),
    POTOK_FLAG_BARO_OK = (1u << 2),
    POTOK_FLAG_PITOT_OK = (1u << 3),
    POTOK_FLAG_AI_READY = (1u << 4),
    POTOK_FLAG_CALIBRATED = (1u << 5),
    POTOK_FLAG_MOTOR_ARMED = (1u << 6),
    POTOK_FLAG_FAULT = (1u << 7)
};

#define POTOK_REQUEST_SIZE 56u
#define POTOK_REQUEST_MAGIC 0x504Du
#define POTOK_REQUEST_VERSION ((uint8_t)1)
#define POTOK_REQUEST_OFF_MAGIC 0u
#define POTOK_REQUEST_OFF_SEQ 2u
#define POTOK_REQUEST_OFF_TIME_US 4u
#define POTOK_REQUEST_OFF_PHASE 8u
#define POTOK_REQUEST_OFF_VERSION 9u
#define POTOK_REQUEST_OFF_FLAGS 10u
#define POTOK_REQUEST_OFF_RESERVED 11u
#define POTOK_REQUEST_OFF_GYRO 12u
#define POTOK_REQUEST_OFF_ACCEL 24u
#define POTOK_REQUEST_OFF_MAG 36u
#define POTOK_REQUEST_OFF_HEIGHT 48u
#define POTOK_REQUEST_OFF_PITOT_DP 52u
#define POTOK_REQUEST_OFF_CRC16 54u
#define POTOK_REQUEST_CRC_OFF 54u
#define POTOK_REQUEST_CRC_LEN 2u

#define POTOK_REPLY_SIZE 14u
#define POTOK_REPLY_MAGIC 0x5250u
#define POTOK_REPLY_VERSION ((uint16_t)1)
#define POTOK_REPLY_OFF_MAGIC 0u
#define POTOK_REPLY_OFF_SEQ 2u
#define POTOK_REPLY_OFF_G_PITCH 4u
#define POTOK_REPLY_G_PITCH_SCALE (0.001f)
#define POTOK_REPLY_OFF_G_YAW 6u
#define POTOK_REPLY_G_YAW_SCALE (0.001f)
#define POTOK_REPLY_OFF_CONFIDENCE 8u
#define POTOK_REPLY_OFF_AI_HEALTHY 9u
#define POTOK_REPLY_OFF_VERSION 10u
#define POTOK_REPLY_OFF_CRC16 12u
#define POTOK_REPLY_CRC_OFF 12u
#define POTOK_REPLY_CRC_LEN 2u

#define POTOK_LOWLAT_REQUEST_SIZE 42u
#define POTOK_LOWLAT_REQUEST_MAGIC 0x4C54u
#define POTOK_LOWLAT_REQUEST_VERSION ((uint8_t)1)
#define POTOK_LOWLAT_REQUEST_OFF_MAGIC 0u
#define POTOK_LOWLAT_REQUEST_OFF_SEQ 2u
#define POTOK_LOWLAT_REQUEST_OFF_TIME_US 4u
#define POTOK_LOWLAT_REQUEST_OFF_PHASE 8u
#define POTOK_LOWLAT_REQUEST_OFF_VERSION 9u
#define POTOK_LOWLAT_REQUEST_OFF_FLAGS 10u
#define POTOK_LOWLAT_REQUEST_OFF_RESERVED 11u
#define POTOK_LOWLAT_REQUEST_OFF_FEATURES 12u
static const double POTOK_LOWLAT_REQUEST_FEATURES_SCALES[14] = {0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.0001, 0.0001, 0.0001, 0.01, 1.0, 1.0, 0.1, 1.0};
static const char *const POTOK_LOWLAT_REQUEST_FEATURES_NAMES[14] = {"gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z", "mag_x", "mag_y", "mag_z", "height", "pressure", "pitot_dp", "temperature", "battery_mv"};
#define POTOK_LOWLAT_REQUEST_OFF_CRC16 40u
#define POTOK_LOWLAT_REQUEST_CRC_OFF 40u
#define POTOK_LOWLAT_REQUEST_CRC_LEN 2u

#define POTOK_LOWLAT_REPLY_SIZE 16u
#define POTOK_LOWLAT_REPLY_MAGIC 0x4C52u
#define POTOK_LOWLAT_REPLY_VERSION ((uint8_t)1)
#define POTOK_LOWLAT_REPLY_OFF_MAGIC 0u
#define POTOK_LOWLAT_REPLY_OFF_SEQ 2u
#define POTOK_LOWLAT_REPLY_OFF_G_PITCH 4u
#define POTOK_LOWLAT_REPLY_G_PITCH_SCALE (0.001f)
#define POTOK_LOWLAT_REPLY_OFF_G_YAW 6u
#define POTOK_LOWLAT_REPLY_G_YAW_SCALE (0.001f)
#define POTOK_LOWLAT_REPLY_OFF_CONFIDENCE 8u
#define POTOK_LOWLAT_REPLY_OFF_AI_HEALTHY 9u
#define POTOK_LOWLAT_REPLY_OFF_VERSION 10u
#define POTOK_LOWLAT_REPLY_OFF_COMPUTE_US 11u
#define POTOK_LOWLAT_REPLY_OFF_RESERVED 12u
#define POTOK_LOWLAT_REPLY_OFF_CRC16 14u
#define POTOK_LOWLAT_REPLY_CRC_OFF 14u
#define POTOK_LOWLAT_REPLY_CRC_LEN 2u

typedef struct {
    uint16_t seq;
    uint32_t time_us;
    enum PotokPhase phase;
    uint8_t version;
    uint8_t flags;
    uint8_t reserved;
    float gyro[3];
    float accel[3];
    float mag[3];
    float height;
    int16_t pitot_dp;
} PotokRequest;

typedef struct {
    uint16_t seq;
    int16_t g_pitch;
    int16_t g_yaw;
    uint8_t confidence;
    uint8_t ai_healthy;
    uint16_t version;
} PotokReply;

typedef struct {
    uint16_t seq;
    uint32_t time_us;
    enum PotokPhase phase;
    uint8_t version;
    uint8_t flags;
    uint8_t reserved;
    int16_t features[14];
} PotokLowlatRequest;

typedef struct {
    uint16_t seq;
    int16_t g_pitch;
    int16_t g_yaw;
    uint8_t confidence;
    uint8_t ai_healthy;
    uint8_t version;
    uint8_t compute_us;
    uint8_t reserved[2];
} PotokLowlatReply;

static inline void potok_put_u16(uint8_t *b, size_t o, uint16_t v) {
    for (size_t i = 0; i < sizeof(uint16_t); ++i) b[o + i] = (uint8_t)(v >> (8u * i));
}
static inline uint16_t potok_get_u16(const uint8_t *b, size_t o) {
    uint16_t v = 0;
    for (size_t i = 0; i < sizeof(uint16_t); ++i) v |= (uint16_t)b[o + i] << (8u * i);
    return v;
}
static inline void potok_put_u32(uint8_t *b, size_t o, uint32_t v) {
    for (size_t i = 0; i < sizeof(uint32_t); ++i) b[o + i] = (uint8_t)(v >> (8u * i));
}
static inline uint32_t potok_get_u32(const uint8_t *b, size_t o) {
    uint32_t v = 0;
    for (size_t i = 0; i < sizeof(uint32_t); ++i) v |= (uint32_t)b[o + i] << (8u * i);
    return v;
}
static inline void potok_put_u8(uint8_t *b, size_t o, uint8_t v) {
    for (size_t i = 0; i < sizeof(uint8_t); ++i) b[o + i] = (uint8_t)(v >> (8u * i));
}
static inline uint8_t potok_get_u8(const uint8_t *b, size_t o) {
    uint8_t v = 0;
    for (size_t i = 0; i < sizeof(uint8_t); ++i) v |= (uint8_t)b[o + i] << (8u * i);
    return v;
}
static inline void potok_put_i16(uint8_t *b, size_t o, int16_t v) {
    for (size_t i = 0; i < sizeof(int16_t); ++i) b[o + i] = (uint8_t)(v >> (8u * i));
}
static inline int16_t potok_get_i16(const uint8_t *b, size_t o) {
    int16_t v = 0;
    for (size_t i = 0; i < sizeof(int16_t); ++i) v |= (int16_t)b[o + i] << (8u * i);
    return v;
}
static inline void potok_put_f32(uint8_t *b, size_t o, float v) {
    uint32_t u; memcpy(&u, &v, sizeof(u)); potok_put_u32(b, o, u);
}
static inline float potok_get_f32(const uint8_t *b, size_t o) {
    uint32_t u = potok_get_u32(b, o); float v; memcpy(&v, &u, sizeof(v)); return v;
}

static const uint16_t POTOK_CRC16_TABLE[256] = {
    0x0000u, 0x1021u, 0x2042u, 0x3063u, 0x4084u, 0x50A5u, 0x60C6u, 0x70E7u,
    0x8108u, 0x9129u, 0xA14Au, 0xB16Bu, 0xC18Cu, 0xD1ADu, 0xE1CEu, 0xF1EFu,
    0x1231u, 0x0210u, 0x3273u, 0x2252u, 0x52B5u, 0x4294u, 0x72F7u, 0x62D6u,
    0x9339u, 0x8318u, 0xB37Bu, 0xA35Au, 0xD3BDu, 0xC39Cu, 0xF3FFu, 0xE3DEu,
    0x2462u, 0x3443u, 0x0420u, 0x1401u, 0x64E6u, 0x74C7u, 0x44A4u, 0x5485u,
    0xA56Au, 0xB54Bu, 0x8528u, 0x9509u, 0xE5EEu, 0xF5CFu, 0xC5ACu, 0xD58Du,
    0x3653u, 0x2672u, 0x1611u, 0x0630u, 0x76D7u, 0x66F6u, 0x5695u, 0x46B4u,
    0xB75Bu, 0xA77Au, 0x9719u, 0x8738u, 0xF7DFu, 0xE7FEu, 0xD79Du, 0xC7BCu,
    0x48C4u, 0x58E5u, 0x6886u, 0x78A7u, 0x0840u, 0x1861u, 0x2802u, 0x3823u,
    0xC9CCu, 0xD9EDu, 0xE98Eu, 0xF9AFu, 0x8948u, 0x9969u, 0xA90Au, 0xB92Bu,
    0x5AF5u, 0x4AD4u, 0x7AB7u, 0x6A96u, 0x1A71u, 0x0A50u, 0x3A33u, 0x2A12u,
    0xDBFDu, 0xCBDCu, 0xFBBFu, 0xEB9Eu, 0x9B79u, 0x8B58u, 0xBB3Bu, 0xAB1Au,
    0x6CA6u, 0x7C87u, 0x4CE4u, 0x5CC5u, 0x2C22u, 0x3C03u, 0x0C60u, 0x1C41u,
    0xEDAEu, 0xFD8Fu, 0xCDECu, 0xDDCDu, 0xAD2Au, 0xBD0Bu, 0x8D68u, 0x9D49u,
    0x7E97u, 0x6EB6u, 0x5ED5u, 0x4EF4u, 0x3E13u, 0x2E32u, 0x1E51u, 0x0E70u,
    0xFF9Fu, 0xEFBEu, 0xDFDDu, 0xCFFCu, 0xBF1Bu, 0xAF3Au, 0x9F59u, 0x8F78u,
    0x9188u, 0x81A9u, 0xB1CAu, 0xA1EBu, 0xD10Cu, 0xC12Du, 0xF14Eu, 0xE16Fu,
    0x1080u, 0x00A1u, 0x30C2u, 0x20E3u, 0x5004u, 0x4025u, 0x7046u, 0x6067u,
    0x83B9u, 0x9398u, 0xA3FBu, 0xB3DAu, 0xC33Du, 0xD31Cu, 0xE37Fu, 0xF35Eu,
    0x02B1u, 0x1290u, 0x22F3u, 0x32D2u, 0x4235u, 0x5214u, 0x6277u, 0x7256u,
    0xB5EAu, 0xA5CBu, 0x95A8u, 0x8589u, 0xF56Eu, 0xE54Fu, 0xD52Cu, 0xC50Du,
    0x34E2u, 0x24C3u, 0x14A0u, 0x0481u, 0x7466u, 0x6447u, 0x5424u, 0x4405u,
    0xA7DBu, 0xB7FAu, 0x8799u, 0x97B8u, 0xE75Fu, 0xF77Eu, 0xC71Du, 0xD73Cu,
    0x26D3u, 0x36F2u, 0x0691u, 0x16B0u, 0x6657u, 0x7676u, 0x4615u, 0x5634u,
    0xD94Cu, 0xC96Du, 0xF90Eu, 0xE92Fu, 0x99C8u, 0x89E9u, 0xB98Au, 0xA9ABu,
    0x5844u, 0x4865u, 0x7806u, 0x6827u, 0x18C0u, 0x08E1u, 0x3882u, 0x28A3u,
    0xCB7Du, 0xDB5Cu, 0xEB3Fu, 0xFB1Eu, 0x8BF9u, 0x9BD8u, 0xABBBu, 0xBB9Au,
    0x4A75u, 0x5A54u, 0x6A37u, 0x7A16u, 0x0AF1u, 0x1AD0u, 0x2AB3u, 0x3A92u,
    0xFD2Eu, 0xED0Fu, 0xDD6Cu, 0xCD4Du, 0xBDAAu, 0xAD8Bu, 0x9DE8u, 0x8DC9u,
    0x7C26u, 0x6C07u, 0x5C64u, 0x4C45u, 0x3CA2u, 0x2C83u, 0x1CE0u, 0x0CC1u,
    0xEF1Fu, 0xFF3Eu, 0xCF5Du, 0xDF7Cu, 0xAF9Bu, 0xBFBAu, 0x8FD9u, 0x9FF8u,
    0x6E17u, 0x7E36u, 0x4E55u, 0x5E74u, 0x2E93u, 0x3EB2u, 0x0ED1u, 0x1EF0u,
};

static inline uint16_t potok_crc16(const uint8_t *data, size_t len) {
    uint16_t crc = 0xFFFFu;
    for (size_t i = 0; i < len; ++i) {
        crc = (uint16_t)((crc << 8) ^ POTOK_CRC16_TABLE[(uint8_t)((crc >> 8) ^ data[i])]);
    }
    return (uint16_t)(crc ^ 0x0000u);
}

static inline int potok_flag_has(uint8_t f, uint8_t mask) { return (f & mask) != 0; }
static inline void potok_flag_set(uint8_t *f, uint8_t mask) { *f = (uint8_t)(*f | mask); }
static inline void potok_flag_clear(uint8_t *f, uint8_t mask) { *f = (uint8_t)(*f & (uint8_t)~mask); }

static inline int potok_request_encode(const PotokRequest *fr, uint8_t *buf, size_t len) {
    if (!fr || !buf || len < POTOK_REQUEST_SIZE) return POTOK_ERR_BUFFER;
    potok_put_u16(buf, POTOK_REQUEST_OFF_MAGIC, (uint16_t)POTOK_REQUEST_MAGIC);
    potok_put_u16(buf, 2, (uint16_t)fr->seq);
    potok_put_u32(buf, 4, (uint32_t)fr->time_us);
    potok_put_u8(buf, 8, (uint8_t)fr->phase);
    potok_put_u8(buf, 9, (uint8_t)fr->version);
    potok_put_u8(buf, 10, (uint8_t)fr->flags);
    potok_put_u8(buf, 11, (uint8_t)fr->reserved);
    for (size_t i = 0; i < 3u; ++i) {
        potok_put_f32(buf, 12 + 4u * i, fr->gyro[i]);
    }
    for (size_t i = 0; i < 3u; ++i) {
        potok_put_f32(buf, 24 + 4u * i, fr->accel[i]);
    }
    for (size_t i = 0; i < 3u; ++i) {
        potok_put_f32(buf, 36 + 4u * i, fr->mag[i]);
    }
    potok_put_f32(buf, 48, fr->height);
    potok_put_i16(buf, 52, (int16_t)fr->pitot_dp);
    potok_put_u16(buf, POTOK_REQUEST_CRC_OFF, potok_crc16(buf, POTOK_REQUEST_CRC_OFF));
    return POTOK_OK;
}
static inline int potok_request_decode(const uint8_t *buf, size_t len, PotokRequest *fr) {
    if (!buf || !fr || len < POTOK_REQUEST_SIZE) return POTOK_ERR_BUFFER;
    if (potok_get_u16(buf, 0) != (uint16_t)POTOK_REQUEST_MAGIC) return POTOK_ERR_MAGIC;
    if (potok_get_u16(buf, POTOK_REQUEST_CRC_OFF) != potok_crc16(buf, POTOK_REQUEST_CRC_OFF)) return POTOK_ERR_CRC;
    fr->seq = (uint16_t)potok_get_u16(buf, 2);
    fr->time_us = (uint32_t)potok_get_u32(buf, 4);
    {
        uint8_t raw = potok_get_u8(buf, 8);
        if (raw > 4u) return POTOK_ERR_PHASE;
        fr->phase = (enum PotokPhase)raw;
    }
    fr->version = (uint8_t)potok_get_u8(buf, 9);
    fr->flags = (uint8_t)potok_get_u8(buf, 10);
    fr->reserved = (uint8_t)potok_get_u8(buf, 11);
    for (size_t i = 0; i < 3u; ++i) {
        fr->gyro[i] = potok_get_f32(buf, 12 + 4u * i);
    }
    for (size_t i = 0; i < 3u; ++i) {
        fr->accel[i] = potok_get_f32(buf, 24 + 4u * i);
    }
    for (size_t i = 0; i < 3u; ++i) {
        fr->mag[i] = potok_get_f32(buf, 36 + 4u * i);
    }
    fr->height = potok_get_f32(buf, 48);
    fr->pitot_dp = (int16_t)potok_get_i16(buf, 52);
    if (fr->version != POTOK_REQUEST_VERSION) return POTOK_ERR_VERSION;
    return POTOK_OK;
}

static inline int potok_reply_encode(const PotokReply *fr, uint8_t *buf, size_t len) {
    if (!fr || !buf || len < POTOK_REPLY_SIZE) return POTOK_ERR_BUFFER;
    potok_put_u16(buf, POTOK_REPLY_OFF_MAGIC, (uint16_t)POTOK_REPLY_MAGIC);
    potok_put_u16(buf, 2, (uint16_t)fr->seq);
    potok_put_i16(buf, 4, (int16_t)fr->g_pitch);
    potok_put_i16(buf, 6, (int16_t)fr->g_yaw);
    potok_put_u8(buf, 8, (uint8_t)fr->confidence);
    potok_put_u8(buf, 9, (uint8_t)fr->ai_healthy);
    potok_put_u16(buf, 10, (uint16_t)fr->version);
    potok_put_u16(buf, POTOK_REPLY_CRC_OFF, potok_crc16(buf, POTOK_REPLY_CRC_OFF));
    return POTOK_OK;
}
static inline int potok_reply_decode(const uint8_t *buf, size_t len, PotokReply *fr) {
    if (!buf || !fr || len < POTOK_REPLY_SIZE) return POTOK_ERR_BUFFER;
    if (potok_get_u16(buf, 0) != (uint16_t)POTOK_REPLY_MAGIC) return POTOK_ERR_MAGIC;
    if (potok_get_u16(buf, POTOK_REPLY_CRC_OFF) != potok_crc16(buf, POTOK_REPLY_CRC_OFF)) return POTOK_ERR_CRC;
    fr->seq = (uint16_t)potok_get_u16(buf, 2);
    fr->g_pitch = (int16_t)potok_get_i16(buf, 4);
    fr->g_yaw = (int16_t)potok_get_i16(buf, 6);
    fr->confidence = (uint8_t)potok_get_u8(buf, 8);
    fr->ai_healthy = (uint8_t)potok_get_u8(buf, 9);
    fr->version = (uint16_t)potok_get_u16(buf, 10);
    if (fr->version != POTOK_REPLY_VERSION) return POTOK_ERR_VERSION;
    return POTOK_OK;
}

static inline int potok_lowlat_request_encode(const PotokLowlatRequest *fr, uint8_t *buf, size_t len) {
    if (!fr || !buf || len < POTOK_LOWLAT_REQUEST_SIZE) return POTOK_ERR_BUFFER;
    potok_put_u16(buf, POTOK_LOWLAT_REQUEST_OFF_MAGIC, (uint16_t)POTOK_LOWLAT_REQUEST_MAGIC);
    potok_put_u16(buf, 2, (uint16_t)fr->seq);
    potok_put_u32(buf, 4, (uint32_t)fr->time_us);
    potok_put_u8(buf, 8, (uint8_t)fr->phase);
    potok_put_u8(buf, 9, (uint8_t)fr->version);
    potok_put_u8(buf, 10, (uint8_t)fr->flags);
    potok_put_u8(buf, 11, (uint8_t)fr->reserved);
    for (size_t i = 0; i < 14u; ++i) {
        potok_put_i16(buf, 12 + 2u * i, (int16_t)fr->features[i]);
    }
    potok_put_u16(buf, POTOK_LOWLAT_REQUEST_CRC_OFF, potok_crc16(buf, POTOK_LOWLAT_REQUEST_CRC_OFF));
    return POTOK_OK;
}
static inline int potok_lowlat_request_decode(const uint8_t *buf, size_t len, PotokLowlatRequest *fr) {
    if (!buf || !fr || len < POTOK_LOWLAT_REQUEST_SIZE) return POTOK_ERR_BUFFER;
    if (potok_get_u16(buf, 0) != (uint16_t)POTOK_LOWLAT_REQUEST_MAGIC) return POTOK_ERR_MAGIC;
    if (potok_get_u16(buf, POTOK_LOWLAT_REQUEST_CRC_OFF) != potok_crc16(buf, POTOK_LOWLAT_REQUEST_CRC_OFF)) return POTOK_ERR_CRC;
    fr->seq = (uint16_t)potok_get_u16(buf, 2);
    fr->time_us = (uint32_t)potok_get_u32(buf, 4);
    {
        uint8_t raw = potok_get_u8(buf, 8);
        if (raw > 4u) return POTOK_ERR_PHASE;
        fr->phase = (enum PotokPhase)raw;
    }
    fr->version = (uint8_t)potok_get_u8(buf, 9);
    fr->flags = (uint8_t)potok_get_u8(buf, 10);
    fr->reserved = (uint8_t)potok_get_u8(buf, 11);
    for (size_t i = 0; i < 14u; ++i) {
        fr->features[i] = (int16_t)potok_get_i16(buf, 12 + 2u * i);
    }
    if (fr->version != POTOK_LOWLAT_REQUEST_VERSION) return POTOK_ERR_VERSION;
    return POTOK_OK;
}

static inline int potok_lowlat_reply_encode(const PotokLowlatReply *fr, uint8_t *buf, size_t len) {
    if (!fr || !buf || len < POTOK_LOWLAT_REPLY_SIZE) return POTOK_ERR_BUFFER;
    potok_put_u16(buf, POTOK_LOWLAT_REPLY_OFF_MAGIC, (uint16_t)POTOK_LOWLAT_REPLY_MAGIC);
    potok_put_u16(buf, 2, (uint16_t)fr->seq);
    potok_put_i16(buf, 4, (int16_t)fr->g_pitch);
    potok_put_i16(buf, 6, (int16_t)fr->g_yaw);
    potok_put_u8(buf, 8, (uint8_t)fr->confidence);
    potok_put_u8(buf, 9, (uint8_t)fr->ai_healthy);
    potok_put_u8(buf, 10, (uint8_t)fr->version);
    potok_put_u8(buf, 11, (uint8_t)fr->compute_us);
    for (size_t i = 0; i < 2u; ++i) {
        potok_put_u8(buf, 12 + 1u * i, (uint8_t)fr->reserved[i]);
    }
    potok_put_u16(buf, POTOK_LOWLAT_REPLY_CRC_OFF, potok_crc16(buf, POTOK_LOWLAT_REPLY_CRC_OFF));
    return POTOK_OK;
}
static inline int potok_lowlat_reply_decode(const uint8_t *buf, size_t len, PotokLowlatReply *fr) {
    if (!buf || !fr || len < POTOK_LOWLAT_REPLY_SIZE) return POTOK_ERR_BUFFER;
    if (potok_get_u16(buf, 0) != (uint16_t)POTOK_LOWLAT_REPLY_MAGIC) return POTOK_ERR_MAGIC;
    if (potok_get_u16(buf, POTOK_LOWLAT_REPLY_CRC_OFF) != potok_crc16(buf, POTOK_LOWLAT_REPLY_CRC_OFF)) return POTOK_ERR_CRC;
    fr->seq = (uint16_t)potok_get_u16(buf, 2);
    fr->g_pitch = (int16_t)potok_get_i16(buf, 4);
    fr->g_yaw = (int16_t)potok_get_i16(buf, 6);
    fr->confidence = (uint8_t)potok_get_u8(buf, 8);
    fr->ai_healthy = (uint8_t)potok_get_u8(buf, 9);
    fr->version = (uint8_t)potok_get_u8(buf, 10);
    fr->compute_us = (uint8_t)potok_get_u8(buf, 11);
    for (size_t i = 0; i < 2u; ++i) {
        fr->reserved[i] = (uint8_t)potok_get_u8(buf, 12 + 1u * i);
    }
    if (fr->version != POTOK_LOWLAT_REPLY_VERSION) return POTOK_ERR_VERSION;
    return POTOK_OK;
}

static inline double potok_reply_g_pitch_scaled(int16_t raw) { return (double)raw * POTOK_REPLY_G_PITCH_SCALE; }
static inline int16_t potok_reply_g_pitch_from_scaled(double value) {
    double v = value / POTOK_REPLY_G_PITCH_SCALE;
    double r = (v >= 0.0) ? v + 0.5 : v - 0.5;
    if (r > 32767.0) return (int16_t)32767;
    if (r < -32768.0) return (int16_t)-32768;
    return (int16_t)r;
}
static inline double potok_reply_g_yaw_scaled(int16_t raw) { return (double)raw * POTOK_REPLY_G_YAW_SCALE; }
static inline int16_t potok_reply_g_yaw_from_scaled(double value) {
    double v = value / POTOK_REPLY_G_YAW_SCALE;
    double r = (v >= 0.0) ? v + 0.5 : v - 0.5;
    if (r > 32767.0) return (int16_t)32767;
    if (r < -32768.0) return (int16_t)-32768;
    return (int16_t)r;
}
static inline double potok_lowlat_reply_g_pitch_scaled(int16_t raw) { return (double)raw * POTOK_LOWLAT_REPLY_G_PITCH_SCALE; }
static inline int16_t potok_lowlat_reply_g_pitch_from_scaled(double value) {
    double v = value / POTOK_LOWLAT_REPLY_G_PITCH_SCALE;
    double r = (v >= 0.0) ? v + 0.5 : v - 0.5;
    if (r > 32767.0) return (int16_t)32767;
    if (r < -32768.0) return (int16_t)-32768;
    return (int16_t)r;
}
static inline double potok_lowlat_reply_g_yaw_scaled(int16_t raw) { return (double)raw * POTOK_LOWLAT_REPLY_G_YAW_SCALE; }
static inline int16_t potok_lowlat_reply_g_yaw_from_scaled(double value) {
    double v = value / POTOK_LOWLAT_REPLY_G_YAW_SCALE;
    double r = (v >= 0.0) ? v + 0.5 : v - 0.5;
    if (r > 32767.0) return (int16_t)32767;
    if (r < -32768.0) return (int16_t)-32768;
    return (int16_t)r;
}

#ifdef __cplusplus
}
#endif
#endif /* POTOK_POTOK_UART_H */
