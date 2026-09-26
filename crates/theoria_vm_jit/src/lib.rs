//! Cranelift JIT backend for the Theoria VM.
//!
//! This crate exists separately from `theoria_vm` so that the unsafe
//! code required to bridge native code and the interpreter is
//! contained in one crate with its own lint discipline. `theoria_vm`
//! itself forbids unsafe code outright.
//!
//! ## What the JIT compiles
//!
//! See the module documentation on `Jit::compile` for the exact
//! instruction coverage. In short: straight-line `Nat` code
//! (`PushConst`, `PushLocal`, `PushStar`, `Succ`, `Return`) lowers to
//! native code; `MakeClosure`, `Apply`, and `Recurse` are rejected with
//! an `UnsupportedInstruction` error and stay in the bytecode
//! interpreter. Bridging recursors into native code requires partial
//! evaluation or per-call-site specialisation, which is its own
//! delivery.
//!
//! ## Value representation
//!
//! The JIT operates on nan-boxed `u64` values (see the `abi` module).
//! `Nat` values are immediates; closures and multi-field constructs are
//! pointer-tagged and heap-allocated through the bump allocator in the
//! `bridge` module.
//!
//! ## Structure
//!
//! * `abi` — the nan-boxed layout shared by JIT code and the bridge.
//! * `bridge` — the two `extern "C"` symbols JIT code calls:
//!   `jit_alloc` (bump allocator) and `jit_call_interp` (interpreter
//!   fallback for `Recurse`, wired when recursor lowering lands).
//! * `codegen` — the Cranelift lowering for the supported instruction
//!   subset.
//! * `jit` — the `Jit` facade: compile a `Program`, run it with packed
//!   arguments.

#![deny(unsafe_code)]
#![deny(rust_2018_idioms)]

extern crate alloc;

#[allow(unsafe_code)]
pub mod abi;
#[allow(unsafe_code)]
pub mod bridge;
pub mod codegen;
pub mod error;
#[allow(unsafe_code)]
pub mod jit;

pub use error::JitError;
pub use jit::Jit;
