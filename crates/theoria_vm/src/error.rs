//! Errors produced by the VM.

use alloc::string::String;
use core::fmt;
use theoria_kernel::name::NameId;

/// Errors the VM can produce.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VmError {
    /// A `Const` referenced a name not present in the runtime
    /// environment.
    UnknownConstant(NameId),
    /// A `Const` referenced a declaration kind the VM cannot execute:
    /// an axiom, an opaque constant, or a quotient primitive.
    UnsupportedConstant(NameId),
    /// An application's head was not a closure or a recursor.
    NotAFunction,
    /// A recursor was applied to a major premise of the wrong shape.
    ///
    /// Produced when, e.g., `Nat.rec` is applied to a value whose head
    /// is not a `Nat.zero` / `Nat.succ` chain.
    BadMajorPremise {
        /// The recursor's name.
        recursor: NameId,
    },
    /// `Nat.succ` applied to a value outside the `u64` range, or
    /// arithmetic that would exceed `u64::MAX`.
    IntegerOverflow,
    /// A de Bruijn index referenced a variable outside the frame's
    /// environment.
    ///
    /// Well-formed erased programs never produce this; it indicates a
    /// bug in the eraser or in the compiler.
    UnboundVariable {
        /// The offending index.
        index: u32,
    },
    /// An internal invariant was violated. Indicates a VM bug.
    Internal(String),
}

impl fmt::Display for VmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VmError::UnknownConstant(id) => write!(f, "unknown constant {id}"),
            VmError::UnsupportedConstant(id) => {
                write!(f, "constant {id} is not executable by the VM")
            }
            VmError::NotAFunction => write!(f, "application of a non-function value"),
            VmError::BadMajorPremise { recursor } => write!(
                f,
                "recursor {recursor} received a major premise that is not a constructor application"
            ),
            VmError::IntegerOverflow => write!(f, "integer overflow"),
            VmError::UnboundVariable { index } => write!(f, "unbound variable #{index}"),
            VmError::Internal(msg) => write!(f, "internal VM error: {msg}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for VmError {}
