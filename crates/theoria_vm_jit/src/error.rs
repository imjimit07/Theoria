//! Errors from the JIT.

use alloc::string::String;
use core::fmt;

/// Errors produced by the JIT.
#[derive(Clone, Debug)]
pub enum JitError {
    /// Cranelift returned an error while building the program.
    Codegen(String),
    /// The host platform is not supported.
    UnsupportedPlatform(String),
    /// The program uses an instruction the JIT does not lower.
    UnsupportedInstruction(String),
    /// A heap allocation exceeded the JIT's bump region.
    OutOfMemory,
    /// An internal invariant was violated.
    Internal(String),
}

impl fmt::Display for JitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JitError::Codegen(msg) => write!(f, "JIT codegen error: {msg}"),
            JitError::UnsupportedPlatform(p) => write!(f, "unsupported platform: {p}"),
            JitError::UnsupportedInstruction(i) => {
                write!(f, "instruction not lowered by the JIT: {i}")
            }
            JitError::OutOfMemory => f.write_str("JIT heap exhausted"),
            JitError::Internal(msg) => write!(f, "internal JIT error: {msg}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for JitError {}
