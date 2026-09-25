//! Lexer: source text to tokens, in two passes.
//!
//! **Pass 1** scans left-to-right, emitting raw tokens for literals,
//! identifiers, punctuation, and operators. Newlines inside unclosed
//! brackets do not produce `Newline` tokens (line continuation); newlines
//! at bracket depth 0 do. Comments (`--` to end of line) produce no
//! tokens.
//!
//! **Pass 2** walks the raw stream and inserts the synthetic `Indent`,
//! `Dedent`, and trailing `Newline`/`Eof` tokens that encode the
//! indentation structure. Blank lines and comment-only lines never touch
//! the indentation stack.
//!
//! Tab characters in *leading* whitespace are rejected with
//! [`ErrorKind::Indentation`]; tabs elsewhere are plain whitespace.

use std::cmp::Ordering;
use theoria_syntax::{ErrorKind, Span, SyntaxError, Token, TokenKind};

/// Lex `source` into a token stream ready for the parser.
///
/// The returned stream always ends with `Newline`, zero or more `Dedent`s,
/// and `Eof`.
///
/// # Errors
///
/// Returns a single [`SyntaxError`] on the first lexical problem.
pub fn lex(source: &str) -> Result<Vec<Token>, SyntaxError> {
    let raw = lex_raw(source)?;
    apply_indentation(source, raw)
}

// ---------------------------------------------------------------------------
// Pass 1: raw tokens
// ---------------------------------------------------------------------------

fn lex_raw(source: &str) -> Result<Vec<Token>, SyntaxError> {
    let bytes = source.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    let mut depth = 0usize;

    while i < bytes.len() {
        let c = bytes[i];
        match c {
            b' ' | b'\t' | b'\r' => {
                i += 1;
            }
            b'\n' => {
                if depth == 0 {
                    out.push(Token::new(
                        TokenKind::Newline,
                        Span::new(i as u32, i as u32 + 1),
                    ));
                }
                i += 1;
            }
            b'-' if bytes.get(i + 1) == Some(&b'-') => {
                // Comment: skip to (not including) the newline.
                i += 2;
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
            }
            b'-' if bytes.get(i + 1) == Some(&b'>') => {
                out.push(Token::new(
                    TokenKind::Arrow,
                    Span::new(i as u32, i as u32 + 2),
                ));
                i += 2;
            }
            b'=' if bytes.get(i + 1) == Some(&b'=') => {
                out.push(Token::new(
                    TokenKind::EqEq,
                    Span::new(i as u32, i as u32 + 2),
                ));
                i += 2;
            }
            b'!' if bytes.get(i + 1) == Some(&b'=') => {
                out.push(Token::new(
                    TokenKind::BangEq,
                    Span::new(i as u32, i as u32 + 2),
                ));
                i += 2;
            }
            b'<' if bytes.get(i + 1) == Some(&b'=') => {
                out.push(Token::new(
                    TokenKind::LtEq,
                    Span::new(i as u32, i as u32 + 2),
                ));
                i += 2;
            }
            b'>' if bytes.get(i + 1) == Some(&b'=') => {
                out.push(Token::new(
                    TokenKind::GtEq,
                    Span::new(i as u32, i as u32 + 2),
                ));
                i += 2;
            }
            b'(' | b'[' | b'{' => {
                let kind = match c {
                    b'(' => TokenKind::LParen,
                    b'[' => TokenKind::LBracket,
                    _ => TokenKind::LBrace,
                };
                out.push(Token::new(kind, Span::new(i as u32, i as u32 + 1)));
                depth += 1;
                i += 1;
            }
            b')' | b']' | b'}' => {
                let kind = match c {
                    b')' => TokenKind::RParen,
                    b']' => TokenKind::RBracket,
                    _ => TokenKind::RBrace,
                };
                out.push(Token::new(kind, Span::new(i as u32, i as u32 + 1)));
                depth = depth.saturating_sub(1);
                i += 1;
            }
            b',' => push_single(&mut out, TokenKind::Comma, i, &mut i),
            b':' => push_single(&mut out, TokenKind::Colon, i, &mut i),
            b'.' => push_single(&mut out, TokenKind::Dot, i, &mut i),
            b';' => push_single(&mut out, TokenKind::Semicolon, i, &mut i),
            b'+' => push_single(&mut out, TokenKind::Plus, i, &mut i),
            b'-' => push_single(&mut out, TokenKind::Minus, i, &mut i),
            b'*' => push_single(&mut out, TokenKind::Star, i, &mut i),
            b'/' => push_single(&mut out, TokenKind::Slash, i, &mut i),
            b'^' => push_single(&mut out, TokenKind::Caret, i, &mut i),
            b'=' => push_single(&mut out, TokenKind::Eq, i, &mut i),
            b'<' => push_single(&mut out, TokenKind::Lt, i, &mut i),
            b'>' => push_single(&mut out, TokenKind::Gt, i, &mut i),
            b'!' => {
                return Err(SyntaxError::new(
                    Span::new(i as u32, i as u32 + 1),
                    ErrorKind::LexError,
                    "unexpected `!` (did you mean `!=`?)".to_string(),
                ));
            }
            b'"' => {
                i = lex_string(source, bytes, i, &mut out)?;
            }
            b'0'..=b'9' => {
                i = lex_number(source, bytes, i, &mut out)?;
            }
            b'A'..=b'Z' | b'a'..=b'z' | b'_' => {
                let start = i;
                while i < bytes.len() && is_ident_continue(bytes[i]) {
                    i += 1;
                }
                out.push(Token::new(
                    TokenKind::Ident,
                    Span::new(start as u32, i as u32),
                ));
            }
            // `λ` (U+03BB) is the lambda introducer. Any other non-ASCII
            // byte is rejected: Unicode identifiers arrive in a later
            // delivery.
            0xCE if bytes.get(i + 1) == Some(&0xBB) => {
                out.push(Token::new(
                    TokenKind::Lambda,
                    Span::new(i as u32, i as u32 + 2),
                ));
                i += 2;
            }
            _ => {
                // Multi-byte characters are only legal inside strings and
                // comments, both handled above.
                return Err(SyntaxError::new(
                    Span::new(i as u32, i as u32 + 1),
                    ErrorKind::LexError,
                    "unexpected character".to_string(),
                ));
            }
        }
    }
    Ok(out)
}

fn push_single(out: &mut Vec<Token>, kind: TokenKind, at: usize, i: &mut usize) {
    out.push(Token::new(kind, Span::new(at as u32, at as u32 + 1)));
    *i += 1;
}

fn is_ident_continue(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn lex_number(
    _source: &str,
    bytes: &[u8],
    start: usize,
    out: &mut Vec<Token>,
) -> Result<usize, SyntaxError> {
    let mut i = start;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    let mut is_float = false;
    // Fraction: `.` followed by a digit.
    if bytes.get(i) == Some(&b'.') && bytes.get(i + 1).is_some_and(|b| b.is_ascii_digit()) {
        is_float = true;
        i += 1;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
    }
    // Exponent: `e[+-]?digits`.
    if bytes.get(i) == Some(&b'e') || bytes.get(i) == Some(&b'E') {
        let mut j = i + 1;
        if bytes.get(j) == Some(&b'+') || bytes.get(j) == Some(&b'-') {
            j += 1;
        }
        if bytes.get(j).is_some_and(|b| b.is_ascii_digit()) {
            is_float = true;
            j += 1;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            i = j;
        }
    }
    if bytes.get(i) == Some(&b'_') {
        return Err(SyntaxError::new(
            Span::new(start as u32, i as u32 + 1),
            ErrorKind::InvalidLiteral,
            "numeric literals may not contain underscores".to_string(),
        ));
    }
    let kind = if is_float {
        TokenKind::Float
    } else {
        TokenKind::Nat
    };
    out.push(Token::new(kind, Span::new(start as u32, i as u32)));
    Ok(i)
}

fn lex_string(
    source: &str,
    bytes: &[u8],
    start: usize,
    out: &mut Vec<Token>,
) -> Result<usize, SyntaxError> {
    // `bytes[start]` is the opening quote.
    let mut i = start + 1;
    loop {
        match bytes.get(i) {
            None => {
                return Err(SyntaxError::new(
                    Span::new(start as u32, i as u32),
                    ErrorKind::InvalidString,
                    "unterminated string literal".to_string(),
                ));
            }
            Some(b'"') => {
                i += 1;
                out.push(Token::new(
                    TokenKind::Str,
                    Span::new(start as u32, i as u32),
                ));
                return Ok(i);
            }
            Some(b'\n') => {
                return Err(SyntaxError::new(
                    Span::new(start as u32, i as u32),
                    ErrorKind::InvalidString,
                    "unterminated string literal".to_string(),
                ));
            }
            Some(b'\\') => match bytes.get(i + 1) {
                Some(b'n' | b't' | b'r' | b'\\' | b'"') => {
                    i += 2;
                }
                _ => {
                    return Err(SyntaxError::new(
                        Span::new(start as u32, i as u32 + 2),
                        ErrorKind::InvalidString,
                        "unknown string escape".to_string(),
                    ));
                }
            },
            Some(_) => {
                // Any other character (including multi-byte UTF-8, which
                // is legal inside strings). `i` is always a char
                // boundary here: it starts past the ASCII quote and
                // advances by whole characters below.
                let ch_len = source[i..].chars().next().map_or(1, |c| c.len_utf8());
                i += ch_len;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Pass 2: indentation
// ---------------------------------------------------------------------------

/// Insert synthetic `Indent`/`Dedent`/`Newline`/`Eof` tokens into a raw
/// token stream.
///
/// Blank lines and comment-only lines never touch the indentation stack:
/// they surface as bare `Newline` tokens with no following token, so the
/// pending line start simply carries over to the next real token.
pub fn apply_indentation(source: &str, raw: Vec<Token>) -> Result<Vec<Token>, SyntaxError> {
    let mut out = Vec::new();
    let mut stack: Vec<u32> = vec![0];
    let mut at_line_start = true;

    for t in raw {
        if t.kind == TokenKind::Newline {
            out.push(t);
            at_line_start = true;
            continue;
        }
        if at_line_start {
            let col = leading_column(source, t.span.start)?;
            let top = *stack.last().expect("indent stack is never empty");
            match col.cmp(&top) {
                Ordering::Greater => {
                    stack.push(col);
                    out.push(Token::new(TokenKind::Indent, Span::empty(t.span.start)));
                }
                Ordering::Less => {
                    while *stack.last().expect("indent stack is never empty") > col {
                        stack.pop();
                        out.push(Token::new(TokenKind::Dedent, Span::empty(t.span.start)));
                    }
                    if *stack.last().expect("indent stack is never empty") != col {
                        return Err(SyntaxError::new(
                            Span::empty(t.span.start),
                            ErrorKind::Indentation,
                            "unindent does not match any outer indent level".to_string(),
                        ));
                    }
                }
                Ordering::Equal => {}
            }
            at_line_start = false;
        }
        out.push(t);
    }

    let end = source.len() as u32;
    // Ensure the stream ends with exactly one Newline: files without a
    // trailing newline need a synthetic one, and files with one must not
    // get a duplicate blank line.
    if out.last().is_none_or(|t| t.kind != TokenKind::Newline) {
        out.push(Token::new(TokenKind::Newline, Span::empty(end)));
    }
    while stack.len() > 1 {
        stack.pop();
        out.push(Token::new(TokenKind::Dedent, Span::empty(end)));
    }
    out.push(Token::new(TokenKind::Eof, Span::empty(end)));
    Ok(out)
}

/// Count the leading spaces of the line containing `offset`.
///
/// # Errors
///
/// Returns an [`ErrorKind::Indentation`] error on tab characters in
/// leading whitespace. Tabs elsewhere are plain whitespace and are
/// skipped silently by pass 1.
fn leading_column(source: &str, offset: u32) -> Result<u32, SyntaxError> {
    let offset = offset as usize;
    let bytes = source.as_bytes();
    let line_start = source[..offset].rfind('\n').map_or(0, |i| i + 1);
    let mut col = 0u32;
    for &b in &bytes[line_start..offset] {
        match b {
            b' ' => col += 1,
            b'\t' => {
                return Err(SyntaxError::new(
                    Span::empty(offset as u32),
                    ErrorKind::Indentation,
                    "tab characters are not allowed in leading whitespace; use spaces".to_string(),
                ));
            }
            _ => break,
        }
    }
    Ok(col)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(source: &str) -> Vec<TokenKind> {
        // Raw pass only: indentation is tested separately.
        lex_raw(source)
            .expect("lex should succeed")
            .into_iter()
            .map(|t| t.kind)
            .collect()
    }

    fn lexed(source: &str) -> Vec<Token> {
        lex(source).expect("lex should succeed")
    }

    fn text_of<'a>(source: &'a str, t: &Token) -> &'a str {
        &source[t.span.start as usize..t.span.end as usize]
    }

    // ---- keywords lex as Ident ---------------------------------------

    #[test]
    fn keywords_lex_as_ident_with_text() {
        let src = "Module Function return Import";
        let toks = lex_raw(src).unwrap();
        assert!(toks.iter().all(|t| t.kind == TokenKind::Ident));
        let texts: Vec<&str> = toks.iter().map(|t| text_of(src, t)).collect();
        assert_eq!(texts, vec!["Module", "Function", "return", "Import"]);
    }

    // ---- numbers ------------------------------------------------------

    #[test]
    fn numbers_lex() {
        assert_eq!(kinds("0"), vec![TokenKind::Nat]);
        assert_eq!(kinds("42"), vec![TokenKind::Nat]);
        assert_eq!(kinds("3.14"), vec![TokenKind::Float]);
        assert_eq!(kinds("1e-6"), vec![TokenKind::Float]);
        assert_eq!(kinds("2.5e10"), vec![TokenKind::Float]);
    }

    #[test]
    fn underscores_in_numbers_rejected() {
        let err = lex_raw("1_000").unwrap_err();
        assert_eq!(err.kind, ErrorKind::InvalidLiteral);
    }

    // ---- strings ------------------------------------------------------

    #[test]
    fn strings_lex() {
        assert_eq!(kinds(r#""""#), vec![TokenKind::Str]);
        assert_eq!(kinds(r#""hello""#), vec![TokenKind::Str]);
        assert_eq!(kinds(r#""a\nb\t\"q\"\\""#), vec![TokenKind::Str]);
    }

    #[test]
    fn unterminated_string_errors() {
        let err = lex_raw(r#""hello"#).unwrap_err();
        assert_eq!(err.kind, ErrorKind::InvalidString);
    }

    #[test]
    fn bad_escape_errors() {
        let err = lex_raw(r#""\q""#).unwrap_err();
        assert_eq!(err.kind, ErrorKind::InvalidString);
    }

    #[test]
    fn comment_marker_inside_string_is_not_a_comment() {
        let toks = lex_raw(r#""a -- b""#).unwrap();
        assert_eq!(toks.len(), 1);
        assert_eq!(toks[0].kind, TokenKind::Str);
    }

    // ---- comments -----------------------------------------------------

    #[test]
    fn full_line_comment_produces_no_code_tokens() {
        // The comment vanishes, but its newline still terminates the line.
        assert_eq!(kinds("-- hello\n"), vec![TokenKind::Newline]);
    }

    #[test]
    fn trailing_comment_after_code() {
        // `42 -- foo` then newline: Nat + Newline.
        assert_eq!(
            kinds("42 -- foo\n"),
            vec![TokenKind::Nat, TokenKind::Newline]
        );
    }

    // ---- operators lex greedily ---------------------------------------

    #[test]
    fn multi_char_operators() {
        assert_eq!(kinds("=="), vec![TokenKind::EqEq]);
        assert_eq!(kinds("!="), vec![TokenKind::BangEq]);
        assert_eq!(kinds("->"), vec![TokenKind::Arrow]);
        assert_eq!(kinds("<="), vec![TokenKind::LtEq]);
        assert_eq!(kinds(">="), vec![TokenKind::GtEq]);
        assert_eq!(kinds("="), vec![TokenKind::Eq]);
        assert_eq!(kinds("<"), vec![TokenKind::Lt]);
        assert_eq!(kinds(">"), vec![TokenKind::Gt]);
        assert_eq!(kinds("-"), vec![TokenKind::Minus]);
    }

    #[test]
    fn lone_bang_errors() {
        let err = lex_raw("!").unwrap_err();
        assert_eq!(err.kind, ErrorKind::LexError);
    }

    #[test]
    fn lambda_introducer_lexes() {
        assert_eq!(kinds("λ"), vec![TokenKind::Lambda]);
    }

    #[test]
    fn other_non_ascii_rejected() {
        let err = lex_raw("→").unwrap_err();
        assert_eq!(err.kind, ErrorKind::LexError);
    }

    // ---- indentation --------------------------------------------------

    fn ind_kinds(source: &str) -> Vec<TokenKind> {
        lexed(source).into_iter().map(|t| t.kind).collect()
    }

    #[test]
    fn single_indent_block() {
        use TokenKind::*;
        let toks = ind_kinds("Function f():\n    return 1\n");
        assert_eq!(
            toks,
            vec![
                Ident, Ident, LParen, RParen, Colon, Newline, Indent, Ident, Nat, Newline, Dedent,
                Eof
            ]
        );
    }

    #[test]
    fn dedent_before_next_top_level_item() {
        use TokenKind::*;
        let toks = ind_kinds("Function f():\n    return 1\nFunction g():\n    return 2\n");
        // ... Newline, Dedent, then the next Function at column 0.
        let dedents = toks.iter().filter(|k| **k == Dedent).count();
        assert_eq!(dedents, 2);
        assert_eq!(toks.last(), Some(&Eof));
    }

    #[test]
    fn blank_and_comment_lines_do_not_change_indentation() {
        use TokenKind::*;
        let toks = ind_kinds("Function f():\n    return 1\n\n    -- a comment\n    return 2\n");
        let indents = toks.iter().filter(|k| **k == Indent).count();
        let dedents = toks.iter().filter(|k| **k == Dedent).count();
        assert_eq!(indents, 1);
        assert_eq!(dedents, 1);
    }

    #[test]
    fn tabs_in_leading_whitespace_rejected() {
        let err = lex("Function f():\n\treturn 1\n").unwrap_err();
        assert_eq!(err.kind, ErrorKind::Indentation);
    }

    #[test]
    fn inconsistent_dedent_rejected() {
        let err = lex("Function f():\n    return 1\n   return 2\n").unwrap_err();
        assert_eq!(err.kind, ErrorKind::Indentation);
    }

    #[test]
    fn line_continuation_inside_parens() {
        use TokenKind::*;
        // The continuation line adds no Indent; only the body indents.
        let toks = ind_kinds("Function f(a: Nat,\n b: Nat):\n    return a\n");
        assert_eq!(toks.iter().filter(|k| **k == Indent).count(), 1);
        assert_eq!(toks.iter().filter(|k| **k == Dedent).count(), 1);
    }
}
