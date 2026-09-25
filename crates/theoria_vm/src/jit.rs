//! Cranelift JIT backend (feature `jit`).
//!
//! Compiles a [`Program`] to native code. Only `Nat`-valued programs
//! are currently supported. The JIT is a follow-up to the bytecode VM
//! delivered in this delivery; this module is a placeholder.
//!
//! ## Design sketch
//!
//! The bytecode's instruction set is close enough to a stack machine
//! that a Cranelift lowering is mechanical:
//!
//! * Stack slots become Cranelift SSA values.
//! * `Value` is represented as a tagged word (`u64` with the low three
//!   bits used as a tag).
//! * Closures allocate a small struct in the JIT's heap area.
//! * Recursors call back into the interpreter, since their rules are
//!   data-driven and not easily compiled ahead of time.
//!
//! The last point matters: without a smart enough compiler to specialise
//! recursor rules per call site, the JIT would need to bridge back to
//! the interpreter for every recursive step. That defeats most of the
//! speedup. The recommended design is therefore to JIT only leaf
//! functions (no recursor applications) and fall back to the interpreter
//! otherwise, until a later delivery adds partial evaluation.

#![cfg(feature = "jit")]

use crate::bytecode::Program;
use crate::error::VmError;
use crate::value::Value;
use alloc::rc::Rc;

/// The native JIT.
///
/// Placeholder: the delivery spec leaves the implementation for a
/// follow-up. The struct exists so that downstream code can refer to it
/// and the feature flag is exercised.
pub struct Jit {
    _private: (),
}

impl Jit {
    /// Attempt to construct a JIT over the given program.
    ///
    /// # Errors
    ///
    /// Currently always returns [`VmError::Internal`]; the real
    /// implementation arrives in the follow-up.
    pub fn compile(_program: &Rc<Program>) -> Result<Self, VmError> {
        Err(VmError::Internal(
            "Cranelift JIT backend is not yet implemented".to_string(),
        ))
    }

    /// Execute the compiled program with the given arguments.
    ///
    /// # Errors
    ///
    /// Currently always returns [`VmError::Internal`].
    pub fn run(&self, _args: &[Value]) -> Result<Value, VmError> {
        Err(VmError::Internal(
            "Cranelift JIT backend is not yet implemented".to_string(),
        ))
    }
}
