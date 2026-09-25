//! Parse a `theoria_syntax::Expr` into [`Exponents`].

use crate::error::UnitError;
use crate::exponents::Exponents;
use alloc::string::ToString;
use theoria_syntax::{BinOp, Expr};

/// Parse a surface expression as a unit expression.
///
/// Only three operators are recognised:
///
/// * `*` — unit multiplication (adds exponent vectors).
/// * `/` — unit division (subtracts exponent vectors).
/// * `^` — unit exponentiation by a literal natural number.
///
/// The leaves are identifiers naming base units: `m`, `s`, `kg`, `A`,
/// `K`, `mol`, `cd`. Identifiers are case-sensitive; `A` (ampere) is a
/// base unit, `a` is not.
///
/// `Paren` nodes are transparent.
///
/// # Errors
///
/// * [`UnitError::UnknownBaseUnit`] for an identifier that is not a base
///   unit.
/// * [`UnitError::UnsupportedOperator`] for any operator other than
///   `*`, `/`, and `^`.
/// * [`UnitError::NonLiteralExponent`] for a `^` whose right-hand side
///   is not a literal natural number.
/// * [`UnitError::ExponentOutOfRange`] for a `^` exponent that does not
///   fit in `i32`.
pub fn parse_unit(expr: &Expr) -> Result<Exponents, UnitError> {
    match expr {
        Expr::Nat { value, .. } if *value == 1 => Ok(Exponents::DIMENSIONLESS),
        Expr::Nat { value, .. } => Err(UnitError::UnsupportedOperator(alloc::format!(
            "unit literal `{value}` (only `1` is a valid unit literal)"
        ))),
        Expr::Ident { text, .. } => base_unit(text),
        Expr::Paren { inner, .. } => parse_unit(inner),
        Expr::BinOp { lhs, op, rhs, .. } => match op {
            BinOp::Mul => Ok(parse_unit(lhs)?.mul(parse_unit(rhs)?)),
            BinOp::Div => Ok(parse_unit(lhs)?.div(parse_unit(rhs)?)),
            BinOp::Pow => {
                let base = parse_unit(lhs)?;
                let exp = literal_exponent(rhs)?;
                Ok(base.pow(exp))
            }
            other => Err(UnitError::UnsupportedOperator(alloc::format!("{other:?}"))),
        },
        _ => Err(UnitError::UnsupportedOperator(
            "only identifiers, `*`, `/`, and `^` are recognised in unit expressions".to_string(),
        )),
    }
}

/// Look up a base unit by name.
fn base_unit(name: &str) -> Result<Exponents, UnitError> {
    match name {
        // SI base units.
        "m" => Ok(Exponents::METRE),
        "s" => Ok(Exponents::SECOND),
        "kg" => Ok(Exponents::KILOGRAM),
        "A" => Ok(Exponents::AMPERE),
        "K" => Ok(Exponents::KELVIN),
        "mol" => Ok(Exponents::MOLE),
        "cd" => Ok(Exponents::CANDELA),
        // The dimensionless unit. Three spellings are accepted for the
        // convenience of the plan's prose, which uses `Unit` in a few
        // places and leaves the intended surface form implicit. All
        // three produce the same exponent vector.
        "dimensionless" | "unitless" | "scalar" => Ok(Exponents::DIMENSIONLESS),
        // SI derived units, as aliases for their base-unit expansions.
        // Each is a pure name-to-exponent-vector rewrite: the resulting
        // kernel constant is the canonical `U_...` for the base
        // expansion, so `Quantity(N)` and `Quantity(kg * m / s^2)` are
        // the same kernel type.
        "N" => Ok(Exponents {
            kg: 1,
            m: 1,
            s: -2,
            ..Exponents::DIMENSIONLESS
        }),
        "Pa" => Ok(Exponents {
            kg: 1,
            m: -1,
            s: -2,
            ..Exponents::DIMENSIONLESS
        }),
        "J" => Ok(Exponents {
            kg: 1,
            m: 2,
            s: -2,
            ..Exponents::DIMENSIONLESS
        }),
        "W" => Ok(Exponents {
            kg: 1,
            m: 2,
            s: -3,
            ..Exponents::DIMENSIONLESS
        }),
        "C" => Ok(Exponents {
            a: 1,
            s: 1,
            ..Exponents::DIMENSIONLESS
        }),
        "V" => Ok(Exponents {
            kg: 1,
            m: 2,
            a: -1,
            s: -3,
            ..Exponents::DIMENSIONLESS
        }),
        "Hz" => Ok(Exponents {
            s: -1,
            ..Exponents::DIMENSIONLESS
        }),
        other => Err(UnitError::UnknownBaseUnit(other.to_string())),
    }
}

/// Extract a literal natural-number exponent.
fn literal_exponent(e: &Expr) -> Result<i32, UnitError> {
    match e {
        Expr::Nat { value, .. } => i32::try_from(*value).map_err(|_| UnitError::ExponentOutOfRange),
        Expr::Paren { inner, .. } => literal_exponent(inner),
        _ => Err(UnitError::NonLiteralExponent),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use theoria_syntax::Span;

    fn ident(name: &str) -> Expr {
        Expr::Ident {
            text: name.to_string(),
            span: Span::empty(0),
        }
    }

    fn nat(n: u64) -> Expr {
        Expr::Nat {
            value: n,
            span: Span::empty(0),
        }
    }

    fn binop(lhs: Expr, op: BinOp, rhs: Expr) -> Expr {
        Expr::BinOp {
            lhs: alloc::boxed::Box::new(lhs),
            op,
            rhs: alloc::boxed::Box::new(rhs),
            span: Span::empty(0),
        }
    }

    fn paren(inner: Expr) -> Expr {
        Expr::Paren {
            inner: alloc::boxed::Box::new(inner),
            span: Span::empty(0),
        }
    }

    #[test]
    fn bare_base_units_parse() {
        assert_eq!(parse_unit(&ident("m")).unwrap(), Exponents::METRE);
        assert_eq!(parse_unit(&ident("s")).unwrap(), Exponents::SECOND);
        assert_eq!(parse_unit(&ident("kg")).unwrap(), Exponents::KILOGRAM);
        assert_eq!(parse_unit(&ident("A")).unwrap(), Exponents::AMPERE);
        assert_eq!(parse_unit(&ident("K")).unwrap(), Exponents::KELVIN);
        assert_eq!(parse_unit(&ident("mol")).unwrap(), Exponents::MOLE);
        assert_eq!(parse_unit(&ident("cd")).unwrap(), Exponents::CANDELA);
    }

    #[test]
    fn power_of_base_unit() {
        let e = parse_unit(&binop(ident("m"), BinOp::Pow, nat(2))).unwrap();
        assert_eq!(
            e,
            Exponents {
                m: 2,
                ..Exponents::DIMENSIONLESS
            }
        );
    }

    #[test]
    fn product_of_units() {
        let e = parse_unit(&binop(ident("kg"), BinOp::Mul, ident("m"))).unwrap();
        assert_eq!(
            e,
            Exponents {
                kg: 1,
                m: 1,
                ..Exponents::DIMENSIONLESS
            }
        );
    }

    #[test]
    fn quotient_of_units() {
        let e = parse_unit(&binop(ident("m"), BinOp::Div, ident("s"))).unwrap();
        assert_eq!(
            e,
            Exponents {
                m: 1,
                s: -1,
                ..Exponents::DIMENSIONLESS
            }
        );
    }

    #[test]
    fn nested_expression() {
        // kg * m / s^2 — acceleration with mass.
        let kg_m = binop(ident("kg"), BinOp::Mul, ident("m"));
        let s_sq = binop(ident("s"), BinOp::Pow, nat(2));
        let e = parse_unit(&binop(kg_m, BinOp::Div, s_sq)).unwrap();
        assert_eq!(
            e,
            Exponents {
                kg: 1,
                m: 1,
                s: -2,
                ..Exponents::DIMENSIONLESS
            }
        );
    }

    #[test]
    fn parens_are_transparent() {
        let m_over_s = binop(ident("m"), BinOp::Div, ident("s"));
        let e = parse_unit(&paren(m_over_s)).unwrap();
        assert_eq!(
            e,
            Exponents {
                m: 1,
                s: -1,
                ..Exponents::DIMENSIONLESS
            }
        );
    }

    #[test]
    fn parens_group_correctly() {
        // (m / s)^2 == m^2 * s^-2
        let m_over_s = binop(ident("m"), BinOp::Div, ident("s"));
        let e = parse_unit(&binop(paren(m_over_s), BinOp::Pow, nat(2))).unwrap();
        assert_eq!(
            e,
            Exponents {
                m: 2,
                s: -2,
                ..Exponents::DIMENSIONLESS
            }
        );
    }

    #[test]
    fn unknown_base_unit_rejected() {
        let err = parse_unit(&ident("zzz")).unwrap_err();
        assert!(matches!(err, UnitError::UnknownBaseUnit(s) if s == "zzz"));
    }

    #[test]
    fn lowercase_a_is_not_a_base_unit() {
        // `A` (ampere) is a base unit; `a` is not.
        assert!(parse_unit(&ident("A")).is_ok());
        assert!(matches!(
            parse_unit(&ident("a")).unwrap_err(),
            UnitError::UnknownBaseUnit(_),
        ));
    }

    #[test]
    fn addition_is_not_a_unit_operator() {
        let err = parse_unit(&binop(ident("m"), BinOp::Add, ident("s"))).unwrap_err();
        assert!(matches!(err, UnitError::UnsupportedOperator(_)));
    }

    #[test]
    fn non_literal_exponent_rejected() {
        // m^x
        let err = parse_unit(&binop(ident("m"), BinOp::Pow, ident("x"))).unwrap_err();
        assert!(matches!(err, UnitError::NonLiteralExponent));
    }

    #[test]
    fn out_of_range_exponent_rejected() {
        let huge = Expr::Nat {
            value: u64::MAX,
            span: Span::empty(0),
        };
        let err = parse_unit(&binop(ident("m"), BinOp::Pow, huge)).unwrap_err();
        assert!(matches!(err, UnitError::ExponentOutOfRange));
    }

    #[test]
    fn dimensionless_keywords_parse() {
        assert_eq!(
            parse_unit(&ident("dimensionless")).unwrap(),
            Exponents::DIMENSIONLESS
        );
        assert_eq!(
            parse_unit(&ident("unitless")).unwrap(),
            Exponents::DIMENSIONLESS
        );
        assert_eq!(
            parse_unit(&ident("scalar")).unwrap(),
            Exponents::DIMENSIONLESS
        );
    }

    #[test]
    fn newton_alias_parses() {
        let n = parse_unit(&ident("N")).unwrap();
        assert_eq!(
            n,
            Exponents {
                kg: 1,
                m: 1,
                s: -2,
                ..Exponents::DIMENSIONLESS
            }
        );
    }

    #[test]
    fn pascal_alias_parses() {
        let p = parse_unit(&ident("Pa")).unwrap();
        assert_eq!(
            p,
            Exponents {
                kg: 1,
                m: -1,
                s: -2,
                ..Exponents::DIMENSIONLESS
            }
        );
    }

    #[test]
    fn joule_alias_parses() {
        let j = parse_unit(&ident("J")).unwrap();
        assert_eq!(
            j,
            Exponents {
                kg: 1,
                m: 2,
                s: -2,
                ..Exponents::DIMENSIONLESS
            }
        );
    }

    #[test]
    fn hertz_alias_parses() {
        let hz = parse_unit(&ident("Hz")).unwrap();
        assert_eq!(
            hz,
            Exponents {
                s: -1,
                ..Exponents::DIMENSIONLESS
            }
        );
    }

    #[test]
    fn derived_alias_matches_base_expansion() {
        // `N` and `kg * m / s^2` must produce the same exponents.
        let n = parse_unit(&ident("N")).unwrap();
        let kg_m = binop(ident("kg"), BinOp::Mul, ident("m"));
        let s_sq = binop(ident("s"), BinOp::Pow, nat(2));
        let expanded = parse_unit(&binop(kg_m, BinOp::Div, s_sq)).unwrap();
        assert_eq!(n, expanded);
    }

    #[test]
    fn volt_alias_parses() {
        let v = parse_unit(&ident("V")).unwrap();
        assert_eq!(
            v,
            Exponents {
                kg: 1,
                m: 2,
                a: -1,
                s: -3,
                ..Exponents::DIMENSIONLESS
            }
        );
    }

    #[test]
    fn numeric_one_is_dimensionless() {
        assert_eq!(parse_unit(&nat(1)).unwrap(), Exponents::DIMENSIONLESS);
    }

    #[test]
    fn other_numeric_literals_rejected() {
        let err = parse_unit(&nat(2)).unwrap_err();
        assert!(matches!(err, UnitError::UnsupportedOperator(_)));
    }
}
