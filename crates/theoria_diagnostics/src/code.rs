//! Diagnostic codes.
//!
//! Codes are the `error[E0042]` style identifiers. The numeric space is
//! partitioned by prefix:
//!
//! | Prefix | Namespace |
//! |--------|-----------|
//! | `E`    | Parser and elaborator errors |
//! | `W`    | Parser and elaborator warnings |
//!
//! Kernel errors are surfaced as `ElaborateErrorKind::Kernel` and reuse
//! the `E` space; they do not have dedicated codes yet.

use core::fmt;

/// A diagnostic code: a prefix and a number.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DiagnosticCode {
    /// The prefix: `"E"` for errors, `"W"` for warnings.
    pub prefix: &'static str,
    /// The numeric portion.
    pub number: u32,
}

impl DiagnosticCode {
    /// An error code, rendered as `E<number>`.
    #[must_use]
    pub const fn error(number: u32) -> Self {
        DiagnosticCode {
            prefix: "E",
            number,
        }
    }

    /// A warning code, rendered as `W<number>`.
    #[must_use]
    pub const fn warning(number: u32) -> Self {
        DiagnosticCode {
            prefix: "W",
            number,
        }
    }
}

impl fmt::Display for DiagnosticCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{:04}", self.prefix, self.number)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_code_renders_with_padding() {
        assert_eq!(alloc::format!("{}", DiagnosticCode::error(42)), "E0042");
        assert_eq!(alloc::format!("{}", DiagnosticCode::error(1)), "E0001");
        assert_eq!(alloc::format!("{}", DiagnosticCode::error(12345)), "E12345");
    }

    #[test]
    fn warning_code_renders() {
        assert_eq!(alloc::format!("{}", DiagnosticCode::warning(7)), "W0007");
    }
}
