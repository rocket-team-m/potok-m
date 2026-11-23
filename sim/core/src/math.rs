//! Small deterministic math helpers (no external linear algebra in the hot path).

pub type Vec3 = [f64; 3];
pub type Quat = [f64; 4]; // (w, x, y, z), maps body axes into world axes.

pub fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

pub fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
pub fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
pub fn scale(a: Vec3, s: f64) -> Vec3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
pub fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
pub fn norm(a: Vec3) -> f64 {
    dot(a, a).sqrt()
}
pub fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub fn quat_normalize(q: Quat) -> Quat {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if n < 1e-12 {
        return [1.0, 0.0, 0.0, 0.0];
    }
    [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
}

/// Rotation matrix (row-major 3x3) from a body-to-world quaternion.
pub fn quat_to_matrix(q: Quat) -> [[f64; 3]; 3] {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

/// q ⊗ [0, omega] for the body-frame angular rate.
pub fn quat_rate(q: Quat, w: Vec3) -> Quat {
    let (qw, qx, qy, qz) = (q[0], q[1], q[2], q[3]);
    [
        -0.5 * (qx * w[0] + qy * w[1] + qz * w[2]),
        0.5 * (qw * w[0] + qy * w[2] - qz * w[1]),
        0.5 * (qw * w[1] + qz * w[0] - qx * w[2]),
        0.5 * (qw * w[2] + qx * w[1] - qy * w[0]),
    ]
}

pub fn mat_vec(m: [[f64; 3]; 3], v: Vec3) -> Vec3 {
    [dot(m[0], v), dot(m[1], v), dot(m[2], v)]
}

pub fn mat_t_vec(m: [[f64; 3]; 3], v: Vec3) -> Vec3 {
    [
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
    ]
}

/// Linear interpolation of a sorted table with clamping at both ends.
pub fn lerp_table(table: &[(f64, f64)], x: f64) -> f64 {
    if table.is_empty() {
        return 0.0;
    }
    if x <= table[0].0 {
        return table[0].1;
    }
    let last = table.len() - 1;
    if x >= table[last].0 {
        return table[last].1;
    }
    for i in 1..=last {
        if x <= table[i].0 {
            let (x0, y0) = table[i - 1];
            let (x1, y1) = table[i];
            let t = (x - x0) / (x1 - x0);
            return y0 + t * (y1 - y0);
        }
    }
    table[last].1
}

/// SplitMix64 finalizer: deterministic episode sub-seed, independent of call order.
pub fn episode_seed(root: u64, index: u64) -> u64 {
    let mut z = root.wrapping_add(index.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// CRC-16/CCITT-FALSE, used by the UART-like frame.
pub fn crc16(data: &[u8]) -> u16 {
    let mut crc: u16 = 0xFFFF;
    for &b in data {
        crc ^= (b as u16) << 8;
        for _ in 0..8 {
            crc = if crc & 0x8000 != 0 {
                (crc << 1) ^ 0x1021
            } else {
                crc << 1
            };
        }
    }
    crc
}
