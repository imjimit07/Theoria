//! Concrete syntax tree, tokens, and spans for Project Theoria.
//!
//! This crate is pure data: spans, source maps, tokens, CST nodes, and
//! syntax errors. It contains no parsing logic (see `theoria_parser`),
//! no I/O, and no external dependencies.
//!
//! The crate is `no_std`-compatible (requires only `core` and `alloc`)
//! so it can be consumed by the future WASM notebook.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

extern crate alloc;

pub mod cst;
pub mod error;
pub mod source_map;
pub mod span;
pub mod token;

pub use cst::{
    BinOp, Expr, FieldDef, FunctionDef, Hypothesis, IfExpr, ImportDecl, ImportName, LamParam,
    MatchArm, MatchExpr, Module, ModuleDecl, Param, Path, PathSegment, Pattern, PatternArg,
    ProofBlock, ProofStep, ProofStepKind, Stmt, StructureDef, TheoremDef, UnOp,
};
pub use error::{ErrorKind, SyntaxError};
pub use source_map::{LineCol, SourceMap};
pub use span::Span;
pub use token::{Token, TokenKind};
