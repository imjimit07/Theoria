//! # Theoria Kernel
//!
//! The trusted logical core of **Project Theoria**. This crate implements the
//! Calculus of Inductive Constructions (CIC) with cumulative universes,
//! definitional equality (β, δ, ι, η), inductive families, universe
//! polymorphism, quotient types, and proof erasure.
//!
//! ## Trust model
//!
//! Everything outside this crate — elaborator, parser, SMT oracles, CAS
//! engines — is *untrusted*. Any proof term that the kernel accepts is
//! considered valid by construction. The kernel's correctness is therefore
//! the single point of failure for the soundness of the whole system.
//!
//! To keep the trusted computing base (TCB) small and auditable:
//!
//! * The crate is `#![forbid(unsafe_code)]`.
//! * The crate has **zero external dependencies**.
//! * The crate is `no_std`-compatible (requires only `core` and `alloc`).
//!
//! ## Module layout
//!
//! | Module  | Responsibility |
//! |---------|----------------|
//! | [`level`] | Universe level algebra: `Zero`, `Succ`, `Max`, `IMax`, `Param`. |
//! | [`name`]  | Interned identifier table used by syntax and environments. |
//! | [`expr`]  | Core syntax tree: `Sort`, `Var`, `Const`, `App`, `Lam`, `Pi`, `Let`. |
//! | [`arena`] | Hash-consed term storage with O(1) `ExprId` equality. |
//! | [`error`] | Kernel error types with source-independent diagnostics. |
//! | [`nbe`]   | Normalization by evaluation: values, environments, conversion. |
//! | [`mod@env`]   | Global declarations and the local typing context. |
//! | [`erasure`] | Proof erasure and Ghost Type Theory extraction. |
//! | [`infer`] | Bidirectional type checker: `infer` and `check`. |
//! | [`inductive`] | Strict and nested positivity checker. |
//! | [`termination`] | Size-Change Termination checker. |
//! | [`prelude`] | The minimal `Nat` standard environment. |
//! | [`quotient`] | Predicative quotient primitives. |
//!
//! ## Roadmap (Phase 1)
//!
//! - [x] `level`  — universe algebra + constraint graph.
//! - [x] `name`   — name interner.
//! - [x] `expr`   — AST skeleton.
//! - [x] `error`  — kernel errors.
//! - [x] `nbe`    — semantic values, de Bruijn levels, readback, conversion
//!         (β/δ/ι/η definitional equality lives here, in `Nbe::convert` and
//!         `Nbe::apply`'s ι/quotient-lift rules; there is no separate
//!         `def_eq` module).
//! - [x] `env`    — declarations, inductives, recursors.
//! - [x] `infer`  — bidirectional type checking and synthesis.
//! - [x] `inductive` — strict and nested positivity checker (incl. carrier
//!         parameters).
//! - [x] `termination` — Size-Change Termination (SCT).
//! - [x] `prelude` — `Nat` inductive, constructors, recursor.
//! - [x] `quotient` — predicative quotients.
//! - [x] `erasure`  — Ghost Type Theory extraction.
//! - [x] `arena` — hash-consed term interner.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

extern crate alloc;

/// The kernel's semantic version, exposed for the CLI's `version` command.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod arena;
pub mod env;
pub mod erasure;
pub mod error;
pub mod expr;
pub mod inductive;
pub mod infer;
pub mod level;
pub mod name;
pub mod nbe;
pub mod prelude;
pub mod quotient;
pub mod termination;

pub use arena::{ExprArena, ExprData, ExprId, InternError};
pub use env::{
    ConstantInfo, ConstantVal, ConstructorVal, DefinitionVal, EnvError, GlobalEnv, InductiveVal,
    LocalContext, LocalDecl, OpaqueVal, QuotientKind, QuotientVal, RecursorRule, RecursorVal,
    TerminationObligation, TheoremVal,
};
pub use erasure::{
    Eraser, ErasureEnv, ErasureSort, GhostSet, LValue, PropOnlyErasureEnv, erase_closed,
};
pub use error::{KernelError, KernelResult};
pub use expr::{BinderInfo, DeBruijnIndex, Expr, Literal};
pub use inductive::PositivityChecker;
pub use infer::{TypeChecker, TypedContext};
pub use level::{Constraint, ConstraintSet, Level, UniverseParamId};
pub use name::{NameId, NameTable};
pub use nbe::{
    ConstDecl, ConstEnv, DbLevel, Head, Nbe, QuotientLiftInfo, RecursorInfo, RecursorRuleInfo,
    Transparency, TransparencyMode, ValRef, Value,
};
pub use prelude::{NatPrelude, Prelude, build_nat_prelude, build_prelude};
pub use quotient::{QuotientPrelude, register_quotient};
pub use termination::{
    CallEdge, CallGraph, Entry, SizeChangeMatrix, TerminationCertificate, TerminationChecker,
    TerminationError,
};
