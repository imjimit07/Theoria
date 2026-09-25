//! A named source file.

use alloc::string::{String, ToString};
use theoria_syntax::SourceMap;

/// A source file: a name plus the parsed source.
///
/// The name is used in the header line (`path:line:col: ...`). It is
/// opaque to the diagnostics crate; the CLI passes the file's path.
#[derive(Clone, Debug)]
pub struct SourceFile {
    /// The file's display name.
    pub name: String,
    /// The parsed source, with the precomputed line index.
    pub source: SourceMap,
}

impl SourceFile {
    /// Build a source file from a name and its text.
    #[must_use]
    pub fn new(name: impl Into<String>, text: &str) -> Self {
        SourceFile {
            name: name.into(),
            source: SourceMap::new(text.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_populates_name_and_source() {
        let f = SourceFile::new("a.theoria", "abc\ndef");
        assert_eq!(f.name, "a.theoria");
        assert_eq!(f.source.source(), "abc\ndef");
    }

    #[test]
    fn line_index_is_built() {
        let f = SourceFile::new("x", "a\nb\nc");
        // Line 2 is "b".
        assert_eq!(f.source.line_text(2), "b");
    }
}
