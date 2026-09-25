//! Diagnostic severity.

use core::fmt;

/// The severity of a diagnostic.
///
/// `Error` and `Warning` are the two the CLI produces today. `Note` and
/// `Help` are used for footer lines inside a diagnostic (see
/// [`Diagnostic::notes`](crate::Diagnostic::notes) and
/// [`Diagnostic::help`](crate::Diagnostic::help)).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Severity {
    /// A fatal problem that prevents further compilation.
    Error,
    /// A non-fatal problem the user should review.
    Warning,
    /// Informational context.
    Note,
    /// A suggested fix.
    Help,
}

impl Severity {
    /// The lowercase keyword the renderer emits for this severity.
    #[must_use]
    pub const fn keyword(self) -> &'static str {
        match self {
            Severity::Error => "error",
            Severity::Warning => "warning",
            Severity::Note => "note",
            Severity::Help => "help",
        }
    }
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.keyword())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keyword_is_lowercase() {
        assert_eq!(Severity::Error.keyword(), "error");
        assert_eq!(Severity::Warning.keyword(), "warning");
        assert_eq!(Severity::Note.keyword(), "note");
        assert_eq!(Severity::Help.keyword(), "help");
    }

    #[test]
    fn display_matches_keyword() {
        assert_eq!(alloc::format!("{}", Severity::Error), "error");
        assert_eq!(alloc::format!("{}", Severity::Warning), "warning");
    }
}
