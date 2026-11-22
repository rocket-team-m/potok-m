"""Кватернионы и малые углы. Соглашение Hamilton, мировая система ENU (z вверх)."""

from __future__ import annotations

import numpy as np

IDX3 = np.eye(3)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Нормировка после каждого шага — шаг, которого не было в GSM DELTA."""
    n = np.linalg.norm(q)
    if not np.isfinite(n) or n < 1e-12:
        raise ValueError("некорректный кватернион")
    return q / n


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_from_rotation_vector(v: np.ndarray) -> np.ndarray:
    """exp(v): v — вектор малого угла (радиан). Без sinc-особенностей при v→0."""
    theta2 = float(v @ v)
    if theta2 < 1e-16:
        return np.array([1.0, *(0.5 * v)])
    theta = np.sqrt(theta2)
    s = np.sin(theta / 2) / theta
    return np.array([np.cos(theta / 2), *(s * v)])


def rotation_vector_from_quat(q: np.ndarray) -> np.ndarray:
    """log(q): малый угол. Выбирается полуповорот с неотрицательной скалярной частью."""
    q = quat_normalize(np.asarray(q, dtype=float))
    if q[0] < 0:
        q = -q
    v = q[1:]
    n = np.linalg.norm(v)
    if n < 1e-12:
        return 2 * v
    return 2 * np.arctan2(n, q[0]) * v / n


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Матрица поворота из связанной системы в мировую (body → world)."""
    w, x, y, z = quat_normalize(np.asarray(q, dtype=float))
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def cholesky_psd(p: np.ndarray, jitter: float = 1e-10) -> tuple[np.ndarray, bool]:
    """Нижний множитель Холецкого с COUNTED-подстраховкой от потери ПД.

    Второй элемент результата — флаг: True означает, что пришлось добавить
    джиттер; он идёт в счётчик телеметрии, молчаливого fallback нет.
    """
    scale = max(float(np.max(np.abs(np.diag(p)))), 1e-30)
    for attempt in range(4):
        try:
            return np.linalg.cholesky(p + np.eye(p.shape[0]) * jitter * scale * 10**attempt), True if attempt else False
        except np.linalg.LinAlgError:
            continue
    raise ValueError("ковариация потеряла положительную определённость")
