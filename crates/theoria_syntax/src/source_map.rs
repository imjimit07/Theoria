//! Source maps: byte offsets to line/column diagnostics.
//!
//! A [`SourceMap`] owns a source file's text together with a precomputed
//! line-start index, so that [`Span`]s can be rendered as `line:col`
//! diagnostics with source snippets.

use crate::span::Span;
use alloc::string::String;
use alloc::vec::Vec;

/// A 1-based line/column position.
///
/// Columns count Unicode characters, not bytes, so that caret snippets
/// line up under multi-byte source text.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct LineCol {
    /// 1-based line number.
    pub line: u32,
    /// 1-based column number, in characters.
    pub col: u32,
}

/// A source file with a precomputed line-start index.
#[derive(Clone, Debug)]
pub struct SourceMap {
    source: String,
    /// Byte offsets of the start of each line. Always non-empty;
    /// `line_starts[0]` is `0`.
    line_starts: Vec<u32>,
}

impl SourceMap {
    /// Build a map for the given source text.
    #[must_use]
    pub fn new(source: String) -> Self {
        let mut line_starts = Vec::new();
        line_starts.push(0);
        for (i, b) in source.bytes().enumerate() {
            if b == b'\n' {
                line_starts.push(i as u32 + 1);
            }
        }
        SourceMap {
            source,
            line_starts,
        }
    }

    /// The underlying source text.
    #[must_use]
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Convert a byte offset to a 1-based line/column.
    ///
    /// Offsets past the end of the source are clamped to the end.
    #[must_use]
    pub fn line_col(&self, offset: u32) -> LineCol {
        let offset = (offset as usize).min(self.source.len());
        // Index of the last line start at or before `offset`.
        let idx = self
            .line_starts
            .partition_point(|&s| (s as usize) <= offset);
        let line = idx as u32; // 1-based: line_starts[0] == 0 always counts
        let start = self.line_starts[idx - 1] as usize;
        let col = self.source[start..offset].chars().count() as u32 + 1;
        LineCol { line, col }
    }

    /// The text of 1-based `line`, without the trailing newline.
    ///
    /// Returns an empty string for out-of-range lines.
    #[must_use]
    pub fn line_text(&self, line: u32) -> &str {
        if line == 0 {
            return "";
        }
        let start = match self.line_starts.get(line as usize - 1) {
            Some(&s) => s as usize,
            None => return "",
        };
        let end = self
            .line_starts
            .get(line as usize)
            .map(|&s| s as usize)
            .unwrap_or(self.source.len());
        let mut text = &self.source[start..end];
        if let Some(stripped) = text.strip_suffix('\n') {
            text = stripped;
        }
        if let Some(stripped) = text.strip_suffix('\r') {
            text = stripped;
        }
        text
    }

    /// Convenience: the line/column of a span's start.
    #[must_use]
    pub fn span_line_col(&self, span: Span) -> LineCol {
        self.line_col(span.start)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_col_first_and_middle_lines() {
        let m = SourceMap::new("ab\ncde\nf".to_string());
        assert_eq!(m.line_col(0), LineCol { line: 1, col: 1 });
        assert_eq!(m.line_col(2), LineCol { line: 1, col: 3 });
        assert_eq!(m.line_col(3), LineCol { line: 2, col: 1 });
        assert_eq!(m.line_col(7), LineCol { line: 3, col: 1 });
    }

    #[test]
    fn line_col_counts_chars_not_bytes() {
        // 'é' is 2 bytes, 1 char.
        let m = SourceMap::new("aé\nxy".to_string());
        assert_eq!(m.line_col(3), LineCol { line: 1, col: 3 });
        assert_eq!(m.line_col(4), LineCol { line: 2, col: 1 });
    }

    #[test]
    fn line_col_last_character_and_end() {
        let m = SourceMap::new("abc".to_string());
        assert_eq!(m.line_col(2), LineCol { line: 1, col: 3 });
        // One past the end clamps to end-of-line.
        assert_eq!(m.line_col(3), LineCol { line: 1, col: 4 });
        assert_eq!(m.line_col(100), LineCol { line: 1, col: 4 });
    }

    #[test]
    fn line_text_first_middle_last() {
        let m = SourceMap::new("one\ntwo\nthree".to_string());
        assert_eq!(m.line_text(1), "one");
        assert_eq!(m.line_text(2), "two");
        assert_eq!(m.line_text(3), "three");
    }

    #[test]
    fn line_text_trailing_newline() {
        let m = SourceMap::new("one\ntwo\n".to_string());
        assert_eq!(m.line_text(1), "one");
        assert_eq!(m.line_text(2), "two");
        assert_eq!(m.line_text(3), "");
    }

    #[test]
    fn line_text_out_of_range_is_empty() {
        let m = SourceMap::new("one".to_string());
        assert_eq!(m.line_text(0), "");
        assert_eq!(m.line_text(2), "");
    }
}
