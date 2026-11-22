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
    POTOK_FLAG_GPS_FIX = (1u << 0),
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

static inline uint16_t potok_crc16(const uint8_t *data, size_t len) {
    uint16_t crc = 0xFFFFu;
    for (size_t i = 0; i < len; ++i) {
        crc ^= (uint16_t)((uint16_t)data[i] << 8);
        for (int j = 0; j < 8; ++j) {
            crc = (crc & 0x8000u) ? (uint16_t)((crc << 1) ^ 0x1021u) : (uint16_t)(crc << 1);
        }
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
