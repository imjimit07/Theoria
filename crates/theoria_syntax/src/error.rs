//! Syntax errors with source spans.
//!
//! [`ErrorKind`] exists for programmatic matching (e.g. a test asserting
//! "this is an indentation error"); [`SyntaxError::message`] is for
//! humans. The CLI renders both.

use crate::span::Span;
use alloc::string::String;

/// A syntax error with a source span and a human-readable message.
#[derive(Clone, Debug)]
pub struct SyntaxError {
    /// Where the error occurred.
    pub span: Span,
    /// The machine-readable category.
    pub kind: ErrorKind,
    /// Human-readable description.
    pub message: String,
}

impl SyntaxError {
    /// Construct an error from its parts.
    #[must_use]
    pub fn new(span: Span, kind: ErrorKind, message: String) -> Self {
        SyntaxError {
            span,
            kind,
            message,
        }
    }
}

/// The machine-readable category of a [`SyntaxError`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    /// The lexer could not form a token.
    LexError,
    /// The parser saw a token it did not expect.
    UnexpectedToken,
    /// The parser reached end of input mid-construct.
    UnexpectedEof,
    /// The indentation structure was malformed.
    Indentation,
    /// A `(`, `[`, or `{` was not closed.
    UnclosedDelimiter,
    /// An integer or float literal was malformed.
    InvalidLiteral,
    /// A string literal was unterminated or contained a bad escape.
    InvalidString,
    /// A top-level form was recognised but is not yet implemented.
    NotImplemented,
}
