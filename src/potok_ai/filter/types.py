"""Типы и константы навигационного фильтра."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

GRAVITY = 9.80665

# Индексы вектора ошибки (15 компонент).
DTHETA = slice(0, 3)
DV = slice(3, 6)
DH = 6
DBG = slice(7, 10)
DBA = slice(10, 13)
DBBARO = 13
DW = slice(14, 16)

N_ERROR = 16


class Phase(Enum):
    """Фазы из protocol/flight.yaml. Гейт фильтра работает от фазы, не от невязки."""

    BURN = "burn"
    COAST = "coast"
    APOGEE = "apogee"
    DESCENT = "descent"
    LANDED = "landed"


@dataclass(frozen=True)
class FilterConfig:
    """Шумы от даташитов/Allan. Значения — отправная точка, заменяются замером."""

    dt: float = 0.01
    sigma_gyro: float = 0.0017  # rad/s/sqrt(Hz)
    sigma_accel: float = 0.010  # m/s^2/sqrt(Hz)
    sigma_gyro_bias: float = 1e-5  # rad/s^2
    sigma_accel_bias: float = 1e-4  # m/s^3
    sigma_baro: float = 0.30  # m
    sigma_baro_bias: float = 1e-3  # m/s
    sigma_wind: float = 0.25  # m/s^2/sqrt(s)
    sigma_mag: float = 1.5e-6  # T
    sigma_baro_meas: float = 0.25  # m
    sigma_accel_meas: float = 0.06  # m/s^2

    # Адаптация R по невязке: окно 20 отсчётов (0,2 с при 100 Гц), forgetting 0,97.
    adapt_window: int = 20
    adapt_forgetting: float = 0.97
    adapt_min_scale: float = 0.1
    adapt_max_scale: float = 10.0

    # Huber: порог в единицах σ_MAD, жёсткие границы [1.5, 4].
    huber_c: float = 2.5
    huber_min: float = 1.5
    huber_max: float = 4.0

    # Гейты.
    accel_gate_g: float = 0.35  # допустимое отклонение |a| от g в долях g
    baro_dynamic_scale: float = 30.0  # множитель R_baro при q/p ≈ 0,2–0,26 (170–200 м/с, 3 км)


@dataclass
class Measurement:
    gyro: np.ndarray  # rad/s, связанная система
    accel: np.ndarray  # m/s^2, связанная система (specific force)
    mag: np.ndarray | None = None  # T, связанная система
    baro_height: float | None = None  # m, уже переведённая высота
    phase: Phase = Phase.COAST
    dynamic_pressure_ratio: float = 0.0  # q/p, гейта барометра
    speed: float = 0.0  # m/s, для гейта ветра


@dataclass
class FilterState:
    q: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    v: np.ndarray = field(default_factory=lambda: np.zeros(3))
    h: float = 0.0
    bg: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ba: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bbaro: float = 0.0
    w: np.ndarray = field(default_factory=lambda: np.zeros(2))

    dx: np.ndarray = field(default_factory=lambda: np.zeros(N_ERROR))
    p: np.ndarray = field(default_factory=lambda: np.eye(N_ERROR))

    # Счётчики — каждый fallback должен быть виден в логе (любое ухудшение должно быть видно в логе).
    jitters: int = 0
    rejected_accel: int = 0
    rejected_mag: int = 0
    rejected_baro: int = 0
    adapted_channels: int = 0

    def copy(self) -> FilterState:
        return FilterState(
            q=self.q.copy(),
            v=self.v.copy(),
            h=self.h,
            bg=self.bg.copy(),
            ba=self.ba.copy(),
            bbaro=self.bbaro,
            w=self.w.copy(),
            dx=self.dx.copy(),
            p=self.p.copy(),
            jitters=self.jitters,
            rejected_accel=self.rejected_accel,
            rejected_mag=self.rejected_mag,
            rejected_baro=self.rejected_baro,
            adapted_channels=self.adapted_channels,
        )
