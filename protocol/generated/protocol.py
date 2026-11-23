# Generated from uart.yaml by `cargo run -p potok-generator`. DO NOT EDIT.
"""Potok-M UART wire format (generated). All frames are fixed-size."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum

PROTOCOL_NAME = "potok-uart"
SPEC_VERSION = 1
ENDIANNESS = "little"
CRC_POLYNOMIAL = 0x1021
CRC_INIT = 0xFFFF
CRC_XOR_OUT = 0x0000

class ProtocolError(ValueError): pass
class Truncated(ProtocolError): pass
class BadMagic(ProtocolError): pass
class BadCrc(ProtocolError): pass
class BadPhase(ProtocolError): pass
class VersionMismatch(ProtocolError): pass

class Phase(IntEnum):
    BURN = 0
    COAST = 1
    APOGEE = 2
    DESCENT = 3
    LANDED = 4

FLAG_IMU_OK = 1 << 1
FLAG_BARO_OK = 1 << 2
FLAG_PITOT_OK = 1 << 3
FLAG_AI_READY = 1 << 4
FLAG_CALIBRATED = 1 << 5
FLAG_MOTOR_ARMED = 1 << 6
FLAG_FAULT = 1 << 7

def flag_has(flags: int, mask: int) -> bool: return (flags & mask) != 0

def crc16(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc ^ 0x0000

REQUEST_SIZE = 56
REQUEST_MAGIC = 0x504D
REQUEST_VERSION = 1
REQUEST_OFF_MAGIC = 0
REQUEST_OFF_SEQ = 2
REQUEST_OFF_TIME_US = 4
REQUEST_OFF_PHASE = 8
REQUEST_OFF_VERSION = 9
REQUEST_OFF_FLAGS = 10
REQUEST_OFF_RESERVED = 11
REQUEST_OFF_GYRO = 12
REQUEST_OFF_ACCEL = 24
REQUEST_OFF_MAG = 36
REQUEST_OFF_HEIGHT = 48
REQUEST_OFF_PITOT_DP = 52
REQUEST_OFF_CRC16 = 54
REQUEST_CRC_OFF = 54
REQUEST_CRC_LEN = 2

@dataclass
class Request:
    seq: int  # u16
    time_us: int  # u32
    phase: Phase
    version: int  # u8
    flags: int  # u8
    reserved: int  # u8
    gyro: list[float]
    accel: list[float]
    mag: list[float]
    height: float
    pitot_dp: int  # i16

def encode_request(frame: Request) -> bytes:
    buf = bytearray(REQUEST_SIZE)
    struct.pack_into("<" + 'H', buf, REQUEST_OFF_MAGIC, REQUEST_MAGIC)
    struct.pack_into("<" + "H", buf, 2, frame.seq)
    struct.pack_into("<" + "I", buf, 4, frame.time_us)
    struct.pack_into("<" + 'B', buf, 8, int(frame.phase))
    struct.pack_into("<" + "B", buf, 9, frame.version)
    struct.pack_into("<" + "B", buf, 10, frame.flags)
    struct.pack_into("<" + "B", buf, 11, frame.reserved)
    for i, item in enumerate(frame.gyro):
        struct.pack_into("<" + "f", buf, 12 + 4 * i, item)
    for i, item in enumerate(frame.accel):
        struct.pack_into("<" + "f", buf, 24 + 4 * i, item)
    for i, item in enumerate(frame.mag):
        struct.pack_into("<" + "f", buf, 36 + 4 * i, item)
    struct.pack_into("<" + "f", buf, 48, frame.height)
    struct.pack_into("<" + "h", buf, 52, frame.pitot_dp)
    crc = crc16(bytes(buf[:REQUEST_CRC_OFF]))
    struct.pack_into("<" + 'H', buf, REQUEST_CRC_OFF, crc)
    return bytes(buf)

def decode_request(data: bytes) -> Request:
    if len(data) < REQUEST_SIZE: raise Truncated(f'need REQUEST_SIZE bytes, got {len(data)}')
    if struct.unpack_from("<" + 'H', data, REQUEST_OFF_MAGIC)[0] != REQUEST_MAGIC: raise BadMagic('magic mismatch')
    want = crc16(bytes(data[:REQUEST_CRC_OFF]))
    got = struct.unpack_from("<" + 'H', data, REQUEST_CRC_OFF)[0]
    if got != want: raise BadCrc(f'want 0x{want:04X}, got 0x{got:04X}')
    seq_v = struct.unpack_from("<" + "H", data, 2)[0]
    time_us_v = struct.unpack_from("<" + "I", data, 4)[0]
    phase_raw = struct.unpack_from("<" + 'B', data, 8)[0]
    try: phase_v = Phase(phase_raw)
    except ValueError as exc: raise BadPhase(str(exc)) from exc
    version_v = struct.unpack_from("<" + "B", data, 9)[0]
    flags_v = struct.unpack_from("<" + "B", data, 10)[0]
    reserved_v = struct.unpack_from("<" + "B", data, 11)[0]
    gyro_v = list(struct.unpack_from("<" + "f" * 3, data, 12))
    accel_v = list(struct.unpack_from("<" + "f" * 3, data, 24))
    mag_v = list(struct.unpack_from("<" + "f" * 3, data, 36))
    height_v = struct.unpack_from("<" + "f", data, 48)[0]
    pitot_dp_v = struct.unpack_from("<" + "h", data, 52)[0]
    if version_v != REQUEST_VERSION: raise VersionMismatch(f'want 1, got {version_v}')
    return Request(
        seq=seq_v,
        time_us=time_us_v,
        phase=phase_v,
        version=version_v,
        flags=flags_v,
        reserved=reserved_v,
        gyro=gyro_v,
        accel=accel_v,
        mag=mag_v,
        height=height_v,
        pitot_dp=pitot_dp_v,
    )


REPLY_SIZE = 14
REPLY_MAGIC = 0x5250
REPLY_VERSION = 1
REPLY_OFF_MAGIC = 0
REPLY_OFF_SEQ = 2
REPLY_OFF_G_PITCH = 4
REPLY_G_PITCH_SCALE = 0.001
REPLY_OFF_G_YAW = 6
REPLY_G_YAW_SCALE = 0.001
REPLY_OFF_CONFIDENCE = 8
REPLY_OFF_AI_HEALTHY = 9
REPLY_OFF_VERSION = 10
REPLY_OFF_CRC16 = 12
REPLY_CRC_OFF = 12
REPLY_CRC_LEN = 2

@dataclass
class Reply:
    seq: int  # u16
    g_pitch: int  # i16
    g_yaw: int  # i16
    confidence: int  # u8
    ai_healthy: int  # u8
    version: int  # u16

def encode_reply(frame: Reply) -> bytes:
    buf = bytearray(REPLY_SIZE)
    struct.pack_into("<" + 'H', buf, REPLY_OFF_MAGIC, REPLY_MAGIC)
    struct.pack_into("<" + "H", buf, 2, frame.seq)
    struct.pack_into("<" + "h", buf, 4, frame.g_pitch)
    struct.pack_into("<" + "h", buf, 6, frame.g_yaw)
    struct.pack_into("<" + "B", buf, 8, frame.confidence)
    struct.pack_into("<" + "B", buf, 9, frame.ai_healthy)
    struct.pack_into("<" + "H", buf, 10, frame.version)
    crc = crc16(bytes(buf[:REPLY_CRC_OFF]))
    struct.pack_into("<" + 'H', buf, REPLY_CRC_OFF, crc)
    return bytes(buf)

def decode_reply(data: bytes) -> Reply:
    if len(data) < REPLY_SIZE: raise Truncated(f'need REPLY_SIZE bytes, got {len(data)}')
    if struct.unpack_from("<" + 'H', data, REPLY_OFF_MAGIC)[0] != REPLY_MAGIC: raise BadMagic('magic mismatch')
    want = crc16(bytes(data[:REPLY_CRC_OFF]))
    got = struct.unpack_from("<" + 'H', data, REPLY_CRC_OFF)[0]
    if got != want: raise BadCrc(f'want 0x{want:04X}, got 0x{got:04X}')
    seq_v = struct.unpack_from("<" + "H", data, 2)[0]
    g_pitch_v = struct.unpack_from("<" + "h", data, 4)[0]
    g_yaw_v = struct.unpack_from("<" + "h", data, 6)[0]
    confidence_v = struct.unpack_from("<" + "B", data, 8)[0]
    ai_healthy_v = struct.unpack_from("<" + "B", data, 9)[0]
    version_v = struct.unpack_from("<" + "H", data, 10)[0]
    if version_v != REPLY_VERSION: raise VersionMismatch(f'want 1, got {version_v}')
    return Reply(
        seq=seq_v,
        g_pitch=g_pitch_v,
        g_yaw=g_yaw_v,
        confidence=confidence_v,
        ai_healthy=ai_healthy_v,
        version=version_v,
    )

def reply_g_pitch_scaled(raw: int) -> float: return raw * REPLY_G_PITCH_SCALE
def reply_g_pitch_from_scaled(value: float) -> int:
    v = value / REPLY_G_PITCH_SCALE
    r = int(v + 0.5 if v >= 0 else v - 0.5)
    return min(32767, max(-32768, r))
def reply_g_yaw_scaled(raw: int) -> float: return raw * REPLY_G_YAW_SCALE
def reply_g_yaw_from_scaled(value: float) -> int:
    v = value / REPLY_G_YAW_SCALE
    r = int(v + 0.5 if v >= 0 else v - 0.5)
    return min(32767, max(-32768, r))

LOWLAT_REQUEST_SIZE = 42
LOWLAT_REQUEST_MAGIC = 0x4C54
LOWLAT_REQUEST_VERSION = 1
LOWLAT_REQUEST_OFF_MAGIC = 0
LOWLAT_REQUEST_OFF_SEQ = 2
LOWLAT_REQUEST_OFF_TIME_US = 4
LOWLAT_REQUEST_OFF_PHASE = 8
LOWLAT_REQUEST_OFF_VERSION = 9
LOWLAT_REQUEST_OFF_FLAGS = 10
LOWLAT_REQUEST_OFF_RESERVED = 11
LOWLAT_REQUEST_OFF_FEATURES = 12
LOWLAT_REQUEST_FEATURES_SCALES = (0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.0001, 0.0001, 0.0001, 0.01, 1.0, 1.0, 0.1, 1.0,)
LOWLAT_REQUEST_FEATURES_NAMES = ("gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z", "mag_x", "mag_y", "mag_z", "height", "pressure", "pitot_dp", "temperature", "battery_mv",)
LOWLAT_REQUEST_OFF_CRC16 = 40
LOWLAT_REQUEST_CRC_OFF = 40
LOWLAT_REQUEST_CRC_LEN = 2

@dataclass
class LowlatRequest:
    seq: int  # u16
    time_us: int  # u32
    phase: Phase
    version: int  # u8
    flags: int  # u8
    reserved: int  # u8
    features: list[int]  # i16

def encode_lowlat_request(frame: LowlatRequest) -> bytes:
    buf = bytearray(LOWLAT_REQUEST_SIZE)
    struct.pack_into("<" + 'H', buf, LOWLAT_REQUEST_OFF_MAGIC, LOWLAT_REQUEST_MAGIC)
    struct.pack_into("<" + "H", buf, 2, frame.seq)
    struct.pack_into("<" + "I", buf, 4, frame.time_us)
    struct.pack_into("<" + 'B', buf, 8, int(frame.phase))
    struct.pack_into("<" + "B", buf, 9, frame.version)
    struct.pack_into("<" + "B", buf, 10, frame.flags)
    struct.pack_into("<" + "B", buf, 11, frame.reserved)
    for i, item in enumerate(frame.features):
        struct.pack_into("<" + "h", buf, 12 + 2 * i, item)
    crc = crc16(bytes(buf[:LOWLAT_REQUEST_CRC_OFF]))
    struct.pack_into("<" + 'H', buf, LOWLAT_REQUEST_CRC_OFF, crc)
    return bytes(buf)

def decode_lowlat_request(data: bytes) -> LowlatRequest:
    if len(data) < LOWLAT_REQUEST_SIZE: raise Truncated(f'need LOWLAT_REQUEST_SIZE bytes, got {len(data)}')
    if struct.unpack_from("<" + 'H', data, LOWLAT_REQUEST_OFF_MAGIC)[0] != LOWLAT_REQUEST_MAGIC: raise BadMagic('magic mismatch')
    want = crc16(bytes(data[:LOWLAT_REQUEST_CRC_OFF]))
    got = struct.unpack_from("<" + 'H', data, LOWLAT_REQUEST_CRC_OFF)[0]
    if got != want: raise BadCrc(f'want 0x{want:04X}, got 0x{got:04X}')
    seq_v = struct.unpack_from("<" + "H", data, 2)[0]
    time_us_v = struct.unpack_from("<" + "I", data, 4)[0]
    phase_raw = struct.unpack_from("<" + 'B', data, 8)[0]
    try: phase_v = Phase(phase_raw)
    except ValueError as exc: raise BadPhase(str(exc)) from exc
    version_v = struct.unpack_from("<" + "B", data, 9)[0]
    flags_v = struct.unpack_from("<" + "B", data, 10)[0]
    reserved_v = struct.unpack_from("<" + "B", data, 11)[0]
    features_v = list(struct.unpack_from("<" + "h" * 14, data, 12))
    if version_v != LOWLAT_REQUEST_VERSION: raise VersionMismatch(f'want 1, got {version_v}')
    return LowlatRequest(
        seq=seq_v,
        time_us=time_us_v,
        phase=phase_v,
        version=version_v,
        flags=flags_v,
        reserved=reserved_v,
        features=features_v,
    )


LOWLAT_REPLY_SIZE = 16
LOWLAT_REPLY_MAGIC = 0x4C52
LOWLAT_REPLY_VERSION = 1
LOWLAT_REPLY_OFF_MAGIC = 0
LOWLAT_REPLY_OFF_SEQ = 2
LOWLAT_REPLY_OFF_G_PITCH = 4
LOWLAT_REPLY_G_PITCH_SCALE = 0.001
LOWLAT_REPLY_OFF_G_YAW = 6
LOWLAT_REPLY_G_YAW_SCALE = 0.001
LOWLAT_REPLY_OFF_CONFIDENCE = 8
LOWLAT_REPLY_OFF_AI_HEALTHY = 9
LOWLAT_REPLY_OFF_VERSION = 10
LOWLAT_REPLY_OFF_COMPUTE_US = 11
LOWLAT_REPLY_OFF_RESERVED = 12
LOWLAT_REPLY_OFF_CRC16 = 14
LOWLAT_REPLY_CRC_OFF = 14
LOWLAT_REPLY_CRC_LEN = 2

@dataclass
class LowlatReply:
    seq: int  # u16
    g_pitch: int  # i16
    g_yaw: int  # i16
    confidence: int  # u8
    ai_healthy: int  # u8
    version: int  # u8
    compute_us: int  # u8
    reserved: list[int]  # u8

def encode_lowlat_reply(frame: LowlatReply) -> bytes:
    buf = bytearray(LOWLAT_REPLY_SIZE)
    struct.pack_into("<" + 'H', buf, LOWLAT_REPLY_OFF_MAGIC, LOWLAT_REPLY_MAGIC)
    struct.pack_into("<" + "H", buf, 2, frame.seq)
    struct.pack_into("<" + "h", buf, 4, frame.g_pitch)
    struct.pack_into("<" + "h", buf, 6, frame.g_yaw)
    struct.pack_into("<" + "B", buf, 8, frame.confidence)
    struct.pack_into("<" + "B", buf, 9, frame.ai_healthy)
    struct.pack_into("<" + "B", buf, 10, frame.version)
    struct.pack_into("<" + "B", buf, 11, frame.compute_us)
    for i, item in enumerate(frame.reserved):
        struct.pack_into("<" + "B", buf, 12 + 1 * i, item)
    crc = crc16(bytes(buf[:LOWLAT_REPLY_CRC_OFF]))
    struct.pack_into("<" + 'H', buf, LOWLAT_REPLY_CRC_OFF, crc)
    return bytes(buf)

def decode_lowlat_reply(data: bytes) -> LowlatReply:
    if len(data) < LOWLAT_REPLY_SIZE: raise Truncated(f'need LOWLAT_REPLY_SIZE bytes, got {len(data)}')
    if struct.unpack_from("<" + 'H', data, LOWLAT_REPLY_OFF_MAGIC)[0] != LOWLAT_REPLY_MAGIC: raise BadMagic('magic mismatch')
    want = crc16(bytes(data[:LOWLAT_REPLY_CRC_OFF]))
    got = struct.unpack_from("<" + 'H', data, LOWLAT_REPLY_CRC_OFF)[0]
    if got != want: raise BadCrc(f'want 0x{want:04X}, got 0x{got:04X}')
    seq_v = struct.unpack_from("<" + "H", data, 2)[0]
    g_pitch_v = struct.unpack_from("<" + "h", data, 4)[0]
    g_yaw_v = struct.unpack_from("<" + "h", data, 6)[0]
    confidence_v = struct.unpack_from("<" + "B", data, 8)[0]
    ai_healthy_v = struct.unpack_from("<" + "B", data, 9)[0]
    version_v = struct.unpack_from("<" + "B", data, 10)[0]
    compute_us_v = struct.unpack_from("<" + "B", data, 11)[0]
    reserved_v = list(struct.unpack_from("<" + "B" * 2, data, 12))
    if version_v != LOWLAT_REPLY_VERSION: raise VersionMismatch(f'want 1, got {version_v}')
    return LowlatReply(
        seq=seq_v,
        g_pitch=g_pitch_v,
        g_yaw=g_yaw_v,
        confidence=confidence_v,
        ai_healthy=ai_healthy_v,
        version=version_v,
        compute_us=compute_us_v,
        reserved=reserved_v,
    )

def lowlat_reply_g_pitch_scaled(raw: int) -> float: return raw * LOWLAT_REPLY_G_PITCH_SCALE
def lowlat_reply_g_pitch_from_scaled(value: float) -> int:
    v = value / LOWLAT_REPLY_G_PITCH_SCALE
    r = int(v + 0.5 if v >= 0 else v - 0.5)
    return min(32767, max(-32768, r))
def lowlat_reply_g_yaw_scaled(raw: int) -> float: return raw * LOWLAT_REPLY_G_YAW_SCALE
def lowlat_reply_g_yaw_from_scaled(value: float) -> int:
    v = value / LOWLAT_REPLY_G_YAW_SCALE
    r = int(v + 0.5 if v >= 0 else v - 0.5)
    return min(32767, max(-32768, r))

