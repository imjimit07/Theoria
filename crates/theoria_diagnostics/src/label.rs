//! Labelled source spans.

use crate::severity::Severity;
use alloc::string::String;
use theoria_syntax::Span;

/// A source identifier: an index into the `SourceFile` slice passed to
/// a [`Renderer`](crate::Renderer).
///
/// The CLI has one source today; the type is `u32` so a future LSP
/// server or REPL can address many.
pub type SourceId = u32;

/// Whether a label is the primary annotation or supplementary context.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LabelStyle {
    /// The primary label: the renderer uses it to place the diagnostic's
    /// header line and the main caret.
    Primary,
    /// A secondary label: context for the primary annotation.
    Secondary,
}

/// A single labelled span within a source file.
#[derive(Clone, Debug)]
pub struct Label {
    /// Which source file the span refers to.
    pub source: SourceId,
    /// The byte range.
    pub span: Span,
    /// Whether this is the primary or a secondary annotation.
    pub style: LabelStyle,
    /// Optional label text, printed next to the caret.
    pub message: Option<String>,
}

impl Label {
    /// A primary label.
    #[must_use]
    pub fn primary(source: SourceId, span: Span) -> Self {
        Label {
            source,
            span,
            style: LabelStyle::Primary,
            message: None,
        }
    }

    /// A secondary label.
    #[must_use]
    pub fn secondary(source: SourceId, span: Span) -> Self {
        Label {
            source,
            span,
            style: LabelStyle::Secondary,
            message: None,
        }
    }

    /// Attach a message to this label.
    #[must_use]
    pub fn with_message(mut self, message: impl Into<String>) -> Self {
        self.message = Some(message.into());
        self
    }

    /// `true` iff this is the primary label.
    #[must_use]
    pub const fn is_primary(&self) -> bool {
        matches!(self.style, LabelStyle::Primary)
    }

    /// The severity-appropriate keyword for this label's style, used by
    /// the renderer to annotate secondary lines. Primary labels inherit
    /// the diagnostic's severity; secondary labels render as `note`.
    #[allow(dead_code)]
    #[must_use]
    pub(crate) fn secondary_severity(&self) -> Severity {
        match self.style {
            LabelStyle::Primary => Severity::Error,
            LabelStyle::Secondary => Severity::Note,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primary_is_primary() {
        let l = Label::primary(0, Span::new(0, 3));
        assert!(l.is_primary());
        assert!(l.message.is_none());
    }

    #[test]
    fn secondary_is_not_primary() {
        let l = Label::secondary(0, Span::new(0, 3));
        assert!(!l.is_primary());
    }

    #[test]
    fn with_message_sets_message() {
        let l = Label::primary(0, Span::new(0, 3)).with_message("here");
        assert_eq!(l.message.as_deref(), Some("here"));
    }
}
