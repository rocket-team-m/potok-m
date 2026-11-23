//! Deterministic wind/gust scenario cells and the train/withheld split.
//!
//! Generalization is tested on *scenarios*, not on rocket types: every episode
//! simulates the same Potok-M airframe. A cell stores fractions of the
//! configured randomization envelopes (`wind.*_range`), so the ranges in
//! `config/default.toml`'s Rust defaults remain the single source of scale.
//! Withheld cells use combinations (all knobs at their extreme plus extra
//! sensor lag) that do not occur in any training cell.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Split {
    Train,
    HeldOut,
}

impl Split {
    pub fn label(self) -> &'static str {
        match self {
            Split::Train => "train",
            Split::HeldOut => "held-out",
        }
    }

    pub fn as_u8(self) -> u8 {
        match self {
            Split::Train => 0,
            Split::HeldOut => 1,
        }
    }

    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Split::Train),
            1 => Some(Split::HeldOut),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Scenario {
    pub id: u16,
    pub split: Split,
    /// Fraction of `wind.v_10_range`.
    pub v10_u: f64,
    /// Fraction of `wind.gust_amplitude_range`; exactly 0 means no gust.
    pub gust_u: f64,
    /// Fraction of `wind.shear_delta_range`.
    pub shear_u: f64,
    /// Fraction of `wind.shear_altitude_range`.
    pub shear_alt_u: f64,
    /// Multiplier on `sensor.lag_tau` (the withheld sensor-delay twist).
    pub lag_scale: f64,
}

impl Scenario {
    /// The four knob values that define a wind/gust scenario. Two cells with
    /// the same signature are the same scenario regardless of episode index.
    pub fn signature(self) -> (f64, f64, f64, f64) {
        (self.v10_u, self.gust_u, self.shear_u, self.shear_alt_u)
    }
}

const CELLS: [Scenario; 20] = [
    Scenario {
        id: 0,
        split: Split::Train,
        v10_u: 0.00,
        gust_u: 0.00,
        shear_u: 0.00,
        shear_alt_u: 0.00,
        lag_scale: 1.0,
    },
    Scenario {
        id: 1,
        split: Split::Train,
        v10_u: 0.20,
        gust_u: 0.00,
        shear_u: 0.25,
        shear_alt_u: 0.20,
        lag_scale: 1.0,
    },
    Scenario {
        id: 2,
        split: Split::Train,
        v10_u: 0.20,
        gust_u: 0.45,
        shear_u: 0.25,
        shear_alt_u: 0.20,
        lag_scale: 1.0,
    },
    Scenario {
        id: 3,
        split: Split::Train,
        v10_u: 0.40,
        gust_u: 0.00,
        shear_u: 0.50,
        shear_alt_u: 0.30,
        lag_scale: 1.0,
    },
    Scenario {
        id: 4,
        split: Split::Train,
        v10_u: 0.40,
        gust_u: 0.60,
        shear_u: 0.50,
        shear_alt_u: 0.30,
        lag_scale: 1.0,
    },
    Scenario {
        id: 5,
        split: Split::Train,
        v10_u: 0.60,
        gust_u: 0.00,
        shear_u: 0.00,
        shear_alt_u: 0.50,
        lag_scale: 1.0,
    },
    Scenario {
        id: 6,
        split: Split::Train,
        v10_u: 0.60,
        gust_u: 0.60,
        shear_u: 0.75,
        shear_alt_u: 0.50,
        lag_scale: 1.0,
    },
    Scenario {
        id: 7,
        split: Split::Train,
        v10_u: 0.80,
        gust_u: 0.00,
        shear_u: 0.25,
        shear_alt_u: 0.60,
        lag_scale: 1.0,
    },
    Scenario {
        id: 8,
        split: Split::Train,
        v10_u: 0.80,
        gust_u: 0.90,
        shear_u: 0.50,
        shear_alt_u: 0.60,
        lag_scale: 1.0,
    },
    Scenario {
        id: 9,
        split: Split::Train,
        v10_u: 1.00,
        gust_u: 0.00,
        shear_u: 0.75,
        shear_alt_u: 0.80,
        lag_scale: 1.0,
    },
    Scenario {
        id: 10,
        split: Split::Train,
        v10_u: 1.00,
        gust_u: 1.00,
        shear_u: 0.00,
        shear_alt_u: 0.80,
        lag_scale: 1.0,
    },
    Scenario {
        id: 11,
        split: Split::Train,
        v10_u: 0.30,
        gust_u: 0.50,
        shear_u: 0.00,
        shear_alt_u: 0.10,
        lag_scale: 1.0,
    },
    Scenario {
        id: 12,
        split: Split::Train,
        v10_u: 0.50,
        gust_u: 0.30,
        shear_u: 1.00,
        shear_alt_u: 0.40,
        lag_scale: 1.0,
    },
    Scenario {
        id: 13,
        split: Split::Train,
        v10_u: 0.70,
        gust_u: 0.75,
        shear_u: 0.50,
        shear_alt_u: 0.70,
        lag_scale: 1.0,
    },
    Scenario {
        id: 14,
        split: Split::Train,
        v10_u: 0.90,
        gust_u: 0.00,
        shear_u: 1.00,
        shear_alt_u: 0.90,
        lag_scale: 1.0,
    },
    Scenario {
        id: 15,
        split: Split::Train,
        v10_u: 0.15,
        gust_u: 1.00,
        shear_u: 0.50,
        shear_alt_u: 0.25,
        lag_scale: 1.0,
    },
    Scenario {
        id: 16,
        split: Split::HeldOut,
        v10_u: 1.00,
        gust_u: 1.00,
        shear_u: 1.00,
        shear_alt_u: 1.00,
        lag_scale: 1.8,
    },
    Scenario {
        id: 17,
        split: Split::HeldOut,
        v10_u: 0.80,
        gust_u: 1.00,
        shear_u: 1.00,
        shear_alt_u: 0.10,
        lag_scale: 2.0,
    },
    Scenario {
        id: 18,
        split: Split::HeldOut,
        v10_u: 1.00,
        gust_u: 0.80,
        shear_u: 1.00,
        shear_alt_u: 0.00,
        lag_scale: 1.6,
    },
    Scenario {
        id: 19,
        split: Split::HeldOut,
        v10_u: 0.90,
        gust_u: 1.00,
        shear_u: 0.90,
        shear_alt_u: 1.00,
        lag_scale: 1.5,
    },
];

pub const SCENARIO_COUNT: usize = CELLS.len();

/// Scenario of an episode index. The mapping is fixed: `0..SCENARIO_COUNT`
/// covers every cell once, then repeats cyclically.
#[inline]
pub fn scenario(index: u64) -> Scenario {
    CELLS[(index % SCENARIO_COUNT as u64) as usize]
}

#[inline]
pub fn is_held_out(index: u64) -> bool {
    scenario(index).split == Split::HeldOut
}
