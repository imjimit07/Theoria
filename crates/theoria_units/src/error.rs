//! Errors produced by the unit subsystem.

use crate::exponents::Exponents;
use alloc::string::String;
use core::fmt;

/// Errors produced by parsing or registering unit expressions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UnitError {
    /// A unit expression referenced a name that is not a base unit.
    UnknownBaseUnit(String),
    /// A unit expression applied an operator not supported by the unit
    /// algebra.
    UnsupportedOperator(String),
    /// A unit expression applied `^` to something other than a natural
    /// number.
    NonLiteralExponent,
    /// A unit expression applied `^` to an exponent outside the `i32`
    /// range.
    ExponentOutOfRange,
    /// A canonical unit constant could not be registered in the kernel
    /// environment.
    Registration(String),
    /// An internal invariant was violated. Indicates a bug in the unit
    /// subsystem.
    Internal(String),
    /// Two operands of a binary operation had incompatible units.
    ///
    /// `+`, `-`, and the comparisons require equal units; the elaborator
    /// produces this error before emitting the corresponding kernel term.
    UnitMismatch {
        /// The left operand's unit.
        lhs: Exponents,
        /// The right operand's unit.
        rhs: Exponents,
    },
    /// A `return` statement's value has a unit that does not match the
    /// declared return type's unit.
    ReturnUnitMismatch {
        /// The declared return type's unit.
        expected: Exponents,
        /// The value's unit.
        found: Exponents,
    },
}

impl fmt::Display for UnitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnitError::UnknownBaseUnit(name) => {
                write!(f, "unknown base unit `{name}`")
            }
            UnitError::UnsupportedOperator(op) => {
                write!(f, "unsupported operator in unit expression: {op}")
            }
            UnitError::NonLiteralExponent => {
                write!(f, "unit exponent must be a literal natural number")
            }
            UnitError::ExponentOutOfRange => {
                write!(f, "unit exponent out of range")
            }
            UnitError::Registration(msg) => write!(f, "cannot register unit: {msg}"),
            UnitError::Internal(msg) => write!(f, "internal unit error: {msg}"),
            UnitError::UnitMismatch { lhs, rhs } => write!(
                f,
                "incompatible units: `{}` vs `{}`",
                lhs.render(),
                rhs.render()
            ),
            UnitError::ReturnUnitMismatch { expected, found } => write!(
                f,
                "return type expects unit `{}`, value has unit `{}`",
                expected.render(),
                found.render()
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for UnitError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_is_nonempty_for_every_variant() {
        let variants = [
            UnitError::UnknownBaseUnit("x".into()),
            UnitError::UnsupportedOperator("+".into()),
            UnitError::NonLiteralExponent,
            UnitError::ExponentOutOfRange,
            UnitError::Registration("boom".into()),
            UnitError::Internal("boom".into()),
            UnitError::UnitMismatch {
                lhs: Exponents::METRE,
                rhs: Exponents::SECOND,
            },
            UnitError::ReturnUnitMismatch {
                expected: Exponents::METRE,
                found: Exponents::SECOND,
            },
        ];
        for v in variants {
            let s = alloc::format!("{v}");
            assert!(!s.is_empty());
        }
    }
}
