//! Bytecode VM and Cranelift JIT for erased Theoria terms.
//!
//! This crate consumes [`theoria_kernel::erasure::LValue`] — the untyped
//! λ-calculus result of the kernel's proof-erasure pass — and executes
//! it. Two execution strategies are provided:
//!
//! 1. **Bytecode interpreter** ([`Vm::run`]). Portable, `no_std`-friendly,
//!    the target for the WASM backend.
//! 2. **Cranelift JIT** (`Jit::compile`, feature `jit`). Native code
//!    for the host platform, using Cranelift's SSA IR.
//!
//! Both paths share the [`Program`] representation produced by
//! [`Compiler::compile`].
//!
//! ## What the VM executes
//!
//! The kernel's erased terms reduce to a small untyped language:
//!
//! * `LValue::Star` — a placeholder for erased proofs and types. The VM
//!   treats it as a distinct inert value: it propagates through
//!   applications but never unfolds.
//! * `LValue::Var` — a de Bruijn-indexed local variable.
//! * `LValue::Lam` — an untyped closure.
//! * `LValue::App` — application.
//! * `LValue::Const` — a reference to a kernel constant: a definition
//!   (unfolded on demand), a recursor (interpreted by its ι-rules), or
//!   an inductive constructor (a value).
//! * `LValue::Lit` — a numeric or string literal (unused by the current
//!   kernel prelude, which encodes `Nat` as constructor chains).
//!
//! ## Recursion via recursors
//!
//! The VM has no primitive arithmetic. `Nat.add`, `Nat.mul`,
//! user-defined `Function`s over `Nat`, and every `Structure`-derived
//! eliminator execute the same way: the erased term contains a
//! `Const(Nat.rec)` or `Const(Foo.rec)` application, and the VM walks
//! the recursor's `rules: Vec<RecursorRule>` — the same rules the
//! kernel's `Nbe::try_iota` uses. The VM is therefore *data-driven by
//! the kernel environment*: adding a `Structure` to the elaborator
//! makes it executable with no VM changes.
//!
//! ## What is not yet supported
//!
//! * **Multi-constructor structures beyond `Nat`/`Bool`.** The
//!   recursor's rules are interpreted correctly, but the value
//!   representation recognizes only unary and nullary constructors.
//!   Multi-constructor `Variant` types are a follow-up.
//! * **Arbitrary-precision `Nat`.** Values are `u64`. Overflow saturates
//!   and returns [`VmError::IntegerOverflow`]. Arbitrary-precision
//!   values arrive with the `Integer`/`Rational` kernel types.
//! * **`Real`/`Complex` values.** Their kernel support is out of
//!   Phase 2.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

extern crate alloc;

pub mod bytecode;
pub mod compile;
pub mod env;
pub mod error;
pub mod interp;
pub mod value;

#[cfg(feature = "jit")]
pub mod jit;

pub use bytecode::{Instruction, Program};
pub use compile::Compiler;
pub use env::VmEnv;
pub use error::VmError;
pub use interp::Vm;
pub use value::Value;

#[cfg(feature = "jit")]
pub use jit::Jit;
