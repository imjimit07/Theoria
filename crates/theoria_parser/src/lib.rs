//! Hand-rolled indentation-aware parser for `.theoria` files.
//!
//! The parser is fail-fast (a single error stops parsing) and operates in
//! two stages: [`lexer`] turns source text into tokens with synthetic
//! `Indent`/`Dedent`/`Newline` markers, and [`parser`] turns the token
//! stream into a [`Module`].
//!
//! The crate is `std`-only: it operates on `&str` and will eventually
//! read files directly.

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

pub mod lexer;
pub mod parser;

pub use parser::parse_module;
pub use theoria_syntax::{ErrorKind, Module, SourceMap, Span, SyntaxError};
