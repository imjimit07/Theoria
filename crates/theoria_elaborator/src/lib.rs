//! Surface-to-kernel elaboration for Project Theoria.
//!
//! This crate turns the concrete syntax tree produced by `theoria_parser`
//! into kernel [`Expr`](theoria_kernel::Expr)s that the type checker
//! accepts. It performs name resolution (locals, then globals), literal
//! desugaring, and function assembly, then hands each declaration to the
//! kernel for checking before insertion.
//!
//! | [`theoria_units`] | Elaboration-time unit expressions behind `Quantity(...)`. |
//!
//! The crate is `no_std`-compatible (requires only `core` and `alloc`),
//! matching the kernel's discipline.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

extern crate alloc;

pub mod default_levels;
pub mod elaborate;
pub mod error;
pub mod scope;
mod structure;

pub use elaborate::{
    ElaboratedFunction, ElaboratedModule, ElaboratedStructure, ElaboratedTheorem, elaborate_module,
    elaborate_module_with_env,
};
pub use error::{ElaborateError, ElaborateErrorKind};
pub use scope::{LocalBinding, Scope};
