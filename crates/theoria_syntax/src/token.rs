//! Token kinds and the token type.
//!
//! [`Token`] deliberately does **not** store the token text: the parser
//! holds the source and slices it when constructing CST identifiers,
//! numbers, and strings. This keeps tokens `Copy` and small (12 bytes)
//! and avoids allocating a `String` per identifier during lexing.
//!
//! Keywords (`Module`, `Import`, `Function`, `return`) are **not**
//! distinct token kinds. They lex as [`TokenKind::Ident`]; the parser
//! checks the text at keyword positions (contextual keywords).

use crate::span::Span;

/// The kind of a lexical token.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TokenKind {
    // Literals
    /// `42`
    Nat,
    /// `3.14`, `1e-6`
    Float,
    /// `"hello"`
    Str,

    // Identifiers
    /// `foo`, `Nat`, `PI`. Keywords lex as `Ident`.
    Ident,

    // Delimiters
    /// `(`
    LParen,
    /// `)`
    RParen,
    /// `[`
    LBracket,
    /// `]`
    RBracket,
    /// `{`
    LBrace,
    /// `}`
    RBrace,

    // Separators
    /// `,`
    Comma,
    /// `:`
    Colon,
    /// `.`
    Dot,
    /// `;`
    Semicolon,

    // Operators
    /// `+`
    Plus,
    /// `-`
    Minus,
    /// `*`
    Star,
    /// `/`
    Slash,
    /// `^`
    Caret,
    /// `=`
    Eq,
    /// `==`
    EqEq,
    /// `!=`
    BangEq,
    /// `<`
    Lt,
    /// `<=`
    LtEq,
    /// `>`
    Gt,
    /// `>=`
    GtEq,
    /// `->`
    Arrow,
    /// `λ` (U+03BB)
    Lambda,

    // Synthetic (inserted by the indentation post-pass)
    /// Logical end of line.
    Newline,
    /// A block starts here.
    Indent,
    /// A block ends here.
    Dedent,

    // End of input
    /// End of input.
    Eof,
}

/// A lexical token: a kind plus its source span.
///
/// The text is recovered by slicing the source with [`Token::span`].
#[derive(Copy, Clone, Debug)]
pub struct Token {
    /// What kind of token this is.
    pub kind: TokenKind,
    /// Where it appears in the source.
    pub span: Span,
}

impl Token {
    /// Construct a token from a kind and a span.
    #[must_use]
    pub const fn new(kind: TokenKind, span: Span) -> Self {
        Token { kind, span }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::size_of;

    #[test]
    fn token_is_copy_and_small() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<Token>();
        assert_copy::<TokenKind>();
        assert!(
            size_of::<Token>() <= 16,
            "Token is {} bytes",
            size_of::<Token>()
        );
    }
}
