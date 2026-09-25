//! The 7-tuple of SI exponents.

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

/// The exponent vector of a unit over the seven SI base units.
///
/// Field order follows the SI convention:
///
/// | Field | Unit     | Symbol |
/// |-------|----------|--------|
/// | `m`   | metre    | m      |
/// | `s`   | second   | s      |
/// | `kg`  | kilogram | kg     |
/// | `a`   | ampere   | A      |
/// | `k`   | kelvin   | K      |
/// | `mol` | mole     | mol    |
/// | `cd`  | candela  | cd     |
///
/// Exponents are signed to admit derived units like `m / s` (with
/// `s = -1`) and `m * s^-2` (acceleration).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Exponents {
    /// Metre exponent.
    pub m: i32,
    /// Second exponent.
    pub s: i32,
    /// Kilogram exponent.
    pub kg: i32,
    /// Ampere exponent.
    pub a: i32,
    /// Kelvin exponent.
    pub k: i32,
    /// Mole exponent.
    pub mol: i32,
    /// Candela exponent.
    pub cd: i32,
}

impl Exponents {
    /// The dimensionless unit (all exponents zero).
    pub const DIMENSIONLESS: Self = Self {
        m: 0,
        s: 0,
        kg: 0,
        a: 0,
        k: 0,
        mol: 0,
        cd: 0,
    };

    /// The metre.
    pub const METRE: Self = Self {
        m: 1,
        ..Self::DIMENSIONLESS
    };

    /// The second.
    pub const SECOND: Self = Self {
        s: 1,
        ..Self::DIMENSIONLESS
    };

    /// The kilogram.
    pub const KILOGRAM: Self = Self {
        kg: 1,
        ..Self::DIMENSIONLESS
    };

    /// The ampere.
    pub const AMPERE: Self = Self {
        a: 1,
        ..Self::DIMENSIONLESS
    };

    /// The kelvin.
    pub const KELVIN: Self = Self {
        k: 1,
        ..Self::DIMENSIONLESS
    };

    /// The mole.
    pub const MOLE: Self = Self {
        mol: 1,
        ..Self::DIMENSIONLESS
    };

    /// The candela.
    pub const CANDELA: Self = Self {
        cd: 1,
        ..Self::DIMENSIONLESS
    };

    /// Add the exponents (unit multiplication).
    #[must_use]
    pub const fn mul(self, other: Self) -> Self {
        Self {
            m: self.m + other.m,
            s: self.s + other.s,
            kg: self.kg + other.kg,
            a: self.a + other.a,
            k: self.k + other.k,
            mol: self.mol + other.mol,
            cd: self.cd + other.cd,
        }
    }

    /// Subtract the exponents (unit division).
    #[must_use]
    pub const fn div(self, other: Self) -> Self {
        Self {
            m: self.m - other.m,
            s: self.s - other.s,
            kg: self.kg - other.kg,
            a: self.a - other.a,
            k: self.k - other.k,
            mol: self.mol - other.mol,
            cd: self.cd - other.cd,
        }
    }

    /// Multiply all exponents by `n` (unit exponentiation).
    ///
    /// `n` is an `i32` so that negative exponents are expressible, though
    /// the surface syntax currently accepts only natural-number
    /// exponents.
    #[must_use]
    pub const fn pow(self, n: i32) -> Self {
        Self {
            m: self.m * n,
            s: self.s * n,
            kg: self.kg * n,
            a: self.a * n,
            k: self.k * n,
            mol: self.mol * n,
            cd: self.cd * n,
        }
    }

    /// `true` iff all exponents are zero.
    #[must_use]
    pub const fn is_dimensionless(self) -> bool {
        self.m == 0
            && self.s == 0
            && self.kg == 0
            && self.a == 0
            && self.k == 0
            && self.mol == 0
            && self.cd == 0
    }

    /// The canonical kernel identifier for this exponent vector.
    ///
    /// The name has the fixed shape
    /// `U_<m>_<s>_<kg>_<a>_<k>_<mol>_<cd>`. Negative exponents are
    /// rendered with a leading hyphen (e.g. `U_1_-1_0_0_0_0_0` for
    /// `m / s`).
    #[must_use]
    pub fn canonical_name(self) -> String {
        alloc::format!(
            "U_{}_{}_{}_{}_{}_{}_{}",
            self.m,
            self.s,
            self.kg,
            self.a,
            self.k,
            self.mol,
            self.cd
        )
    }

    /// A human-readable rendering for diagnostics.
    ///
    /// The rendering uses `*` between factors and `^` on non-unit
    /// exponents; zero exponents are dropped. Dimensionless renders as
    /// the literal `dimensionless`.
    #[must_use]
    pub fn render(self) -> String {
        if self.is_dimensionless() {
            return alloc::string::String::from("dimensionless");
        }
        let fields: [(&str, i32); 7] = [
            ("m", self.m),
            ("s", self.s),
            ("kg", self.kg),
            ("A", self.a),
            ("K", self.k),
            ("mol", self.mol),
            ("cd", self.cd),
        ];
        let parts: Vec<String> = fields
            .iter()
            .filter(|(_, e)| *e != 0)
            .map(|(name, e)| {
                if *e == 1 {
                    alloc::string::String::from(*name)
                } else {
                    alloc::format!("{name}^{e}")
                }
            })
            .collect();
        parts.join(" * ")
    }
}

impl fmt::Display for Exponents {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.render())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimensionless_is_all_zero() {
        assert!(Exponents::DIMENSIONLESS.is_dimensionless());
        assert!(!Exponents::METRE.is_dimensionless());
    }

    #[test]
    fn mul_adds_exponents() {
        let m2 = Exponents::METRE.mul(Exponents::METRE);
        assert_eq!(
            m2,
            Exponents {
                m: 2,
                ..Exponents::DIMENSIONLESS
            }
        );
    }

    #[test]
    fn div_subtracts_exponents() {
        let m_over_s = Exponents::METRE.div(Exponents::SECOND);
        assert_eq!(
            m_over_s,
            Exponents {
                m: 1,
                s: -1,
                ..Exponents::DIMENSIONLESS
            }
        );
    }

    #[test]
    fn pow_scales_exponents() {
        let accel = Exponents::METRE.pow(1).div(Exponents::SECOND.pow(2));
        assert_eq!(
            accel,
            Exponents {
                m: 1,
                s: -2,
                ..Exponents::DIMENSIONLESS
            }
        );
    }

    #[test]
    fn canonical_name_format() {
        assert_eq!(Exponents::METRE.canonical_name(), "U_1_0_0_0_0_0_0");
        assert_eq!(
            Exponents {
                m: 2,
                s: -1,
                ..Exponents::DIMENSIONLESS
            }
            .canonical_name(),
            "U_2_-1_0_0_0_0_0"
        );
        assert_eq!(Exponents::DIMENSIONLESS.canonical_name(), "U_0_0_0_0_0_0_0");
    }

    #[test]
    fn render_drops_zero_exponents() {
        let m2_over_s = Exponents {
            m: 2,
            s: -1,
            ..Exponents::DIMENSIONLESS
        };
        assert_eq!(m2_over_s.render(), "m^2 * s^-1");
        assert_eq!(Exponents::METRE.render(), "m");
        assert_eq!(Exponents::DIMENSIONLESS.render(), "dimensionless");
    }

    #[test]
    fn display_matches_render() {
        let e = Exponents {
            kg: 1,
            m: 1,
            s: -2,
            ..Exponents::DIMENSIONLESS
        };
        assert_eq!(alloc::format!("{e}"), e.render());
    }
}
