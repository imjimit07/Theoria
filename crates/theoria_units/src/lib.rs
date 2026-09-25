//! Elaboration-time dimensional analysis for Project Theoria.
//!
//! The kernel has no concept of units. Everything unit-related happens at
//! elaboration time: this crate parses unit expressions like `m^2 / s`
//! into a canonical 7-tuple of SI exponents, interns one kernel constant
//! per distinct tuple, and emits `Quantity(u)` where `u` is that constant.
//! The kernel sees ordinary constants and ordinary applications of the
//! `Quantity` type former.
//!
//! ## What is in scope
//!
//! * [`Exponents`] — the 7-tuple of `i32` exponents over the SI base units.
//! * [`parse_unit`] — parse a `theoria_syntax::Expr` into `Exponents`
//!   (unit-algebra operators `*`, `/`, and `^`; base-unit identifiers only).
//! * [`UnitRegistry`] — interns one kernel constant per distinct exponent
//!   vector, with canonical names of the form
//!   `U_<m>_<s>_<kg>_<a>_<k>_<mol>_<cd>`.
//! * Kernel-side installation of `Unit` and `Quantity`.
//!
//! ## What is deferred
//!
//! * **Unit-aware arithmetic.** `r * r` where `r : Quantity(m)` should
//!   produce a value of type `Quantity(m^2)`. This requires the
//!   elaborator to track a unit alongside every local binding and every
//!   expression, which is a whole additional type system layered on top
//!   of the kernel's. It is a later delivery.
//! * **Unit-checked function signatures.** A signature
//!   `f : Quantity(m) -> Quantity(m^2)` is *expressible* (both sides are
//!   ordinary kernel types), but the elaborator does not verify that the
//!   body's unit arithmetic is consistent. The kernel checks only that
//!   the body has the declared type as a whole.
//! * **Values in quantities.** The kernel has no `Real` type, so
//!   `Quantity` is a bare type former with no way to construct values of
//!   it. Once `Real` lands, `Quantity.mk` can be added.
//!
//! ## Example
//!
//! ```ignore
//! use theoria_units::{parse_unit, Exponents, UnitRegistry};
//!
//! // `m^2 / s` parses to the exponent vector `{m: 2, s: -1, ..}`.
//! let e = parse_unit(&expr_for("m^2 / s")).unwrap();
//! assert_eq!(e, Exponents { m: 2, s: -1, ..Exponents::DIMENSIONLESS });
//! ```
//!
//! (The `expr_for` helper is not part of the crate; end-to-end parsing is
//! exercised by the tests in this crate and by the elaborator's tests.)

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

extern crate alloc;

pub mod error;
pub mod exponents;
pub mod parser;
pub mod registry;

pub use error::UnitError;
pub use exponents::Exponents;
pub use parser::parse_unit;
pub use registry::UnitRegistry;
