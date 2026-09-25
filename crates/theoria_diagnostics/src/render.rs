//! The renderer.
//!
//! Produces the CLI's diagnostic format:
//!
//! ```text
//! file.theoria:5:14: error: expected `return`
//! 5 |     return 1
//!   |     ^^^^^^^^
//!   = note: the function body must end in `return`
//!   = help: add a `return` statement before the closing dedent
//! ```
//!
//! The header line is emitted as `{path}:{line}:{col}: {severity}: {message}`
//! when the diagnostic has a primary label, and as `{severity}: {message}`
//! otherwise. Notes and help are footer lines prefixed with `= note:`
//! and `= help:` and aligned under the caret column.

use crate::diagnostic::Diagnostic;
use crate::label::{Label, LabelStyle};
use crate::source::SourceFile;
use alloc::string::String;
use core::fmt;
use theoria_syntax::Span;

/// When to emit ANSI color escape sequences.
///
/// The renderer currently ignores this setting; the variants exist so
/// that a follow-up can honor them without an API change.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorChoice {
    /// Never emit color.
    Never,
    /// Emit color if the output is a terminal.
    Auto,
    /// Always emit color.
    Always,
}

/// Rendering options.
#[derive(Clone, Debug)]
pub struct RenderOptions {
    /// Whether to emit ANSI color.
    pub color: ColorChoice,
    /// Preferred output width in columns. Currently unused.
    pub width: usize,
    /// Tab width for expanding tabs in source lines. Currently unused;
    /// tabs are not supported by the parser, so they do not appear in
    /// accepted source.
    pub tab_width: usize,
}

impl Default for RenderOptions {
    fn default() -> Self {
        RenderOptions {
            color: ColorChoice::Never,
            width: 100,
            tab_width: 4,
        }
    }
}

/// Renders diagnostics against a set of source files.
pub struct Renderer<'a> {
    sources: &'a [SourceFile],
    options: RenderOptions,
}

impl<'a> Renderer<'a> {
    /// A renderer with default options.
    #[must_use]
    pub fn new(sources: &'a [SourceFile]) -> Self {
        Renderer {
            sources,
            options: RenderOptions::default(),
        }
    }

    /// A renderer with explicit options.
    #[must_use]
    pub fn with_options(sources: &'a [SourceFile], options: RenderOptions) -> Self {
        Renderer { sources, options }
    }

    /// The current options.
    #[must_use]
    pub fn options(&self) -> &RenderOptions {
        &self.options
    }

    /// Render a single diagnostic.
    ///
    /// # Errors
    ///
    /// Propagates `fmt::Error` from the writer.
    pub fn render<W: fmt::Write>(&self, diag: &Diagnostic, w: &mut W) -> fmt::Result {
        self.render_header(diag, w)?;
        if let Some(primary) = diag.primary_label() {
            self.render_snippet(primary, diag.severity, w)?;
            for label in &diag.labels {
                if label.style == LabelStyle::Secondary {
                    self.render_secondary(label, w)?;
                }
            }
            self.render_footer(&diag.notes, diag.help.as_deref(), w)?;
        } else {
            // An unlocated diagnostic: notes and help are still rendered,
            // but no snippet.
            self.render_footer_unlocated(&diag.notes, diag.help.as_deref(), w)?;
        }
        Ok(())
    }

    /// Render several diagnostics in sequence.
    ///
    /// # Errors
    ///
    /// Propagates `fmt::Error` from the writer.
    pub fn render_all<W: fmt::Write>(&self, diags: &[Diagnostic], w: &mut W) -> fmt::Result {
        for (i, d) in diags.iter().enumerate() {
            if i > 0 {
                writeln!(w)?;
            }
            self.render(d, w)?;
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Header
    // ------------------------------------------------------------------

    fn render_header<W: fmt::Write>(&self, diag: &Diagnostic, w: &mut W) -> fmt::Result {
        if let Some(label) = diag.primary_label() {
            if let Some(src) = self.sources.get(label.source as usize) {
                let lc = src.source.line_col(label.span.start);
                write!(w, "{}:{}:{}: ", src.name, lc.line, lc.col)?;
            }
        }
        match diag.code {
            Some(code) => writeln!(w, "{}[{}]: {}", diag.severity, code, diag.message),
            None => writeln!(w, "{}: {}", diag.severity, diag.message),
        }
    }

    // ------------------------------------------------------------------
    // Snippet
    // ------------------------------------------------------------------

    fn render_snippet<W: fmt::Write>(
        &self,
        label: &Label,
        severity: crate::severity::Severity,
        w: &mut W,
    ) -> fmt::Result {
        let Some(src) = self.sources.get(label.source as usize) else {
            return Ok(());
        };
        let start = src.source.line_col(label.span.start);
        let end = src.source.line_col(label.span.end);

        // Bar width: the number of decimal digits of the last line
        // number we will print, plus one for the leading separator.
        let bar = bar_width(end.line);

        if start.line == end.line {
            // Single-line span.
            let line_text = src.source.line_text(start.line);
            let line_chars: alloc::vec::Vec<char> = line_text.chars().collect();
            let col0 = (start.col as usize).saturating_sub(1).min(line_chars.len());
            let span_chars = span_len_within_line(&src.source, label.span);
            let width = span_chars
                .max(1)
                .min(line_chars.len().saturating_sub(col0).max(1));
            writeln!(w, "{:>width$} | {}", start.line, line_text, width = bar)?;
            writeln!(
                w,
                "{:>pad$} | {}{}",
                "",
                " ".repeat(col0),
                "^".repeat(width),
                pad = bar,
            )?;
            if let Some(msg) = &label.message {
                writeln!(w, "{:>pad$} | {}{}", "", " ".repeat(col0), msg, pad = bar,)?;
            }
        } else {
            // Multi-line span: show the first and last lines with an
            // ellipsis between.
            let first_text = src.source.line_text(start.line);
            let last_text = src.source.line_text(end.line);
            let first_chars: alloc::vec::Vec<char> = first_text.chars().collect();
            let last_chars: alloc::vec::Vec<char> = last_text.chars().collect();
            let first_col0 = (start.col as usize)
                .saturating_sub(1)
                .min(first_chars.len());
            let last_end = (end.col as usize).saturating_sub(1).min(last_chars.len());

            writeln!(w, "{:>width$} | {}", start.line, first_text, width = bar)?;
            writeln!(
                w,
                "{:>pad$} | {}{}",
                "",
                " ".repeat(first_col0),
                "^".repeat(first_chars.len().saturating_sub(first_col0).max(1)),
                pad = bar,
            )?;
            writeln!(w, "{:>pad$} | ...", "", pad = bar)?;
            writeln!(w, "{:>width$} | {}", end.line, last_text, width = bar)?;
            writeln!(
                w,
                "{:>pad$} | {}",
                "",
                "^".repeat(last_end.max(1)),
                pad = bar,
            )?;
            if let Some(msg) = &label.message {
                writeln!(w, "{:>pad$} | {}", "", msg, pad = bar)?;
            }
        }
        let _ = severity;
        Ok(())
    }

    fn render_secondary<W: fmt::Write>(&self, label: &Label, w: &mut W) -> fmt::Result {
        let Some(src) = self.sources.get(label.source as usize) else {
            return Ok(());
        };
        let start = src.source.line_col(label.span.start);
        let end = src.source.line_col(label.span.end);
        let bar = bar_width(end.line);
        let msg = label.message.as_deref().unwrap_or("");
        if start.line == end.line {
            let line_text = src.source.line_text(start.line);
            let line_chars: alloc::vec::Vec<char> = line_text.chars().collect();
            let col0 = (start.col as usize).saturating_sub(1).min(line_chars.len());
            let span_chars = span_len_within_line(&src.source, label.span);
            let width = span_chars
                .max(1)
                .min(line_chars.len().saturating_sub(col0).max(1));
            writeln!(w, "{:>width$} | {}", start.line, line_text, width = bar)?;
            writeln!(
                w,
                "{:>pad$} | {}{} {}",
                "",
                " ".repeat(col0),
                "-".repeat(width),
                msg,
                pad = bar,
            )?;
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Footer
    // ------------------------------------------------------------------

    fn render_footer<W: fmt::Write>(
        &self,
        notes: &[String],
        help: Option<&str>,
        w: &mut W,
    ) -> fmt::Result {
        // Align the `=` markers under the caret column, matching the
        // source-line prefix. The exact column depends on the snippet,
        // but two spaces past the bar is a reasonable approximation.
        let pad = 2usize;
        for note in notes {
            writeln!(w, "{:>pad$} = note: {note}", "", pad = pad)?;
        }
        if let Some(h) = help {
            writeln!(w, "{:>pad$} = help: {h}", "", pad = pad)?;
        }
        Ok(())
    }

    fn render_footer_unlocated<W: fmt::Write>(
        &self,
        notes: &[String],
        help: Option<&str>,
        w: &mut W,
    ) -> fmt::Result {
        for note in notes {
            writeln!(w, "  = note: {note}")?;
        }
        if let Some(h) = help {
            writeln!(w, "  = help: {h}")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Free helpers
// ---------------------------------------------------------------------------

/// The width of the source-line bar (`"5 |"` for single-digit line
/// numbers, `"10 |"` for two-digit, ...).
fn bar_width(line: u32) -> usize {
    let mut n = line;
    let mut digits = 1;
    while n >= 10 {
        n /= 10;
        digits += 1;
    }
    digits
}

/// The number of characters the span covers within the line it starts
/// on, capped at the line's remaining length.
fn span_len_within_line(source: &theoria_syntax::SourceMap, span: Span) -> usize {
    let text = source.source();
    let start = span.start as usize;
    let end = span.end as usize;
    let end = end.min(text.len());
    if start >= end {
        return 0;
    }
    text[start..end].chars().take_while(|&c| c != '\n').count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code::DiagnosticCode;
    use crate::label::Label;
    use theoria_syntax::Span;

    fn sources() -> alloc::vec::Vec<SourceFile> {
        alloc::vec![SourceFile::new(
            "file.theoria",
            "Function f() -> Nat:\n    return 1\n",
        )]
    }

    fn render(diag: &Diagnostic, sources: &[SourceFile]) -> String {
        let mut out = String::new();
        let r = Renderer::new(sources);
        r.render(diag, &mut out).unwrap();
        out
    }

    #[test]
    fn single_line_snippet_renders() {
        // Span covers `return 1` on line 2.
        // Line 2 text: "    return 1"
        // "    return 1" is 12 chars; "return 1" starts at column 5.
        let span = Span::new(25, 33);
        let d = Diagnostic::error("expected an expression").with_label(Label::primary(0, span));
        let s = render(&d, &sources());
        assert!(s.contains("file.theoria:2:5: error: expected an expression"));
        assert!(s.contains("2 |     return 1"));
        assert!(s.contains("^"));
    }

    #[test]
    fn header_without_labels_omits_location() {
        let d = Diagnostic::error("internal error");
        let s = render(&d, &sources());
        assert_eq!(s, "error: internal error\n");
    }

    #[test]
    fn code_renders_in_header() {
        let span = Span::new(25, 33);
        let d = Diagnostic::error("nope")
            .with_code(DiagnosticCode::error(42))
            .with_label(Label::primary(0, span));
        let s = render(&d, &sources());
        assert!(s.contains("error[E0042]: nope"));
    }

    #[test]
    fn notes_and_help_render_in_footer() {
        let span = Span::new(25, 33);
        let d = Diagnostic::error("bad")
            .with_label(Label::primary(0, span))
            .with_note("note one")
            .with_note("note two")
            .with_help("try this");
        let s = render(&d, &sources());
        assert!(s.contains("= note: note one"));
        assert!(s.contains("= note: note two"));
        assert!(s.contains("= help: try this"));
    }

    #[test]
    fn multi_line_span_renders_ellipsis() {
        let mut out = String::new();
        let src = SourceFile::new("x", "a\nbb\nccc\ndddd");
        let span = Span::new(0, 10); // crosses several lines
        let d = Diagnostic::error("spans lines").with_label(Label::primary(0, span));
        let r = Renderer::new(core::slice::from_ref(&src));
        r.render(&d, &mut out).unwrap();
        assert!(out.contains("..."));
    }

    #[test]
    fn render_all_separates_with_blank_line() {
        let span = Span::new(25, 33);
        let d1 = Diagnostic::error("one").with_label(Label::primary(0, span));
        let d2 = Diagnostic::error("two").with_label(Label::primary(0, span));
        let srcs = sources();
        let mut out = String::new();
        Renderer::new(&srcs)
            .render_all(&[d1, d2], &mut out)
            .unwrap();
        assert!(out.contains("error: one"));
        assert!(out.contains("error: two"));
        assert!(out.contains("\n\n"));
    }

    #[test]
    fn bar_width_scales_with_line_number() {
        assert_eq!(bar_width(1), 1);
        assert_eq!(bar_width(9), 1);
        assert_eq!(bar_width(10), 2);
        assert_eq!(bar_width(99), 2);
        assert_eq!(bar_width(100), 3);
    }

    #[test]
    fn empty_source_file_renders_without_panic() {
        // A span past the end of the source is clamped; the renderer
        // must not panic.
        let src = SourceFile::new("empty", "");
        let span = Span::new(0, 0);
        let d = Diagnostic::error("empty span").with_label(Label::primary(0, span));
        let mut out = String::new();
        Renderer::new(core::slice::from_ref(&src))
            .render(&d, &mut out)
            .unwrap();
    }

    #[test]
    fn secondary_labels_render() {
        let span = Span::new(25, 33);
        let d = Diagnostic::error("primary")
            .with_label(Label::primary(0, span))
            .with_label(Label::secondary(0, Span::new(11, 19)).with_message("defined here"));
        let s = render(&d, &sources());
        assert!(s.contains("defined here"));
        assert!(s.contains('-'));
    }
}
