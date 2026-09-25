//! The diagnostic data model.

use crate::code::DiagnosticCode;
use crate::label::Label;
use crate::severity::Severity;
use alloc::string::String;
use alloc::vec::Vec;

/// A compiler diagnostic.
///
/// Built with a fluent interface:
///
/// ```ignore
/// Diagnostic::error("expected `return`")
///     .with_code(DiagnosticCode::error(42))
///     .with_label(Label::primary(0, span))
///     .with_note("the function body must end in `return`")
///     .with_help("add a `return` statement before the closing dedent")
/// ```
#[derive(Clone, Debug)]
pub struct Diagnostic {
    /// How severe the problem is.
    pub severity: Severity,
    /// An optional stable code, rendered as `error[E0042]`.
    pub code: Option<DiagnosticCode>,
    /// The one-line summary shown in the header.
    pub message: String,
    /// Zero or more labelled spans.
    pub labels: Vec<Label>,
    /// Footer notes, rendered as `= note: ...`.
    pub notes: Vec<String>,
    /// An optional suggested fix, rendered as `= help: ...`.
    pub help: Option<String>,
}

impl Diagnostic {
    /// A diagnostic with the given severity and message.
    #[must_use]
    pub fn new(severity: Severity, message: impl Into<String>) -> Self {
        Diagnostic {
            severity,
            code: None,
            message: message.into(),
            labels: Vec::new(),
            notes: Vec::new(),
            help: None,
        }
    }

    /// An error-level diagnostic.
    #[must_use]
    pub fn error(message: impl Into<String>) -> Self {
        Self::new(Severity::Error, message)
    }

    /// A warning-level diagnostic.
    #[must_use]
    pub fn warning(message: impl Into<String>) -> Self {
        Self::new(Severity::Warning, message)
    }

    /// Attach a code.
    #[must_use]
    pub fn with_code(mut self, code: DiagnosticCode) -> Self {
        self.code = Some(code);
        self
    }

    /// Add a labelled span.
    #[must_use]
    pub fn with_label(mut self, label: Label) -> Self {
        self.labels.push(label);
        self
    }

    /// Add a footer note.
    #[must_use]
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    /// Set the suggested fix.
    #[must_use]
    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.help = Some(help.into());
        self
    }

    /// The first primary label, if any. The renderer uses this to place
    /// the header line's `line:col` prefix and to choose the snippet.
    #[must_use]
    pub fn primary_label(&self) -> Option<&Label> {
        self.labels.iter().find(|l| l.is_primary())
    }

    /// `true` iff the diagnostic has no labels.
    #[must_use]
    pub fn is_unlocated(&self) -> bool {
        self.labels.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use theoria_syntax::Span;

    #[test]
    fn error_constructor_sets_severity() {
        let d = Diagnostic::error("boom");
        assert_eq!(d.severity, Severity::Error);
        assert_eq!(d.message, "boom");
        assert!(d.code.is_none());
        assert!(d.labels.is_empty());
        assert!(d.notes.is_empty());
        assert!(d.help.is_none());
    }

    #[test]
    fn warning_constructor_sets_severity() {
        let d = Diagnostic::warning("hm");
        assert_eq!(d.severity, Severity::Warning);
    }

    #[test]
    fn builder_chain_accumulates() {
        let d = Diagnostic::error("oops")
            .with_code(DiagnosticCode::error(42))
            .with_label(Label::primary(0, Span::new(0, 1)))
            .with_label(Label::secondary(0, Span::new(2, 3)))
            .with_note("n1")
            .with_note("n2")
            .with_help("do this");
        assert_eq!(d.code, Some(DiagnosticCode::error(42)));
        assert_eq!(d.labels.len(), 2);
        assert_eq!(d.notes, alloc::vec!["n1".to_string(), "n2".to_string()]);
        assert_eq!(d.help.as_deref(), Some("do this"));
    }

    #[test]
    fn primary_label_finds_first_primary() {
        let d = Diagnostic::error("x")
            .with_label(Label::secondary(0, Span::new(0, 1)))
            .with_label(Label::primary(0, Span::new(2, 3)));
        let p = d.primary_label().unwrap();
        assert_eq!(p.span, Span::new(2, 3));
    }

    #[test]
    fn unlocated_when_no_labels() {
        let d = Diagnostic::error("x");
        assert!(d.is_unlocated());
        let d = d.with_label(Label::primary(0, Span::new(0, 1)));
        assert!(!d.is_unlocated());
    }
}
