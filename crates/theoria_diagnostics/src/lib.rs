//! Source-span diagnostics for Project Theoria.
//!
//! One [`Diagnostic`] type carries every error, warning, and note the
//! compiler produces — from the parser's `SyntaxError`, the elaborator's
//! `ElaborateError`, and eventually the kernel's `KernelError`. A single
//! [`Renderer`] turns a slice of diagnostics into the rustc-style output
//! the CLI prints.
//!
//! ## Design
//!
//! * **Structured, not stringly-typed.** A diagnostic is a data
//!   structure with a severity, a code, one or more labelled spans, and
//!   optional notes and help text. It does not commit to a terminal
//!   layout; the renderer does.
//! * **Renderer-independent model.** A future WASM notebook or LSP
//!   server consumes the same [`Diagnostic`] values and produces its
//!   own output (a DOM tree, an LSP `Diagnostic` payload, ...). The
//!   [`Renderer`] in this crate is one consumer among several.
//! * **`no_std`-friendly.** The model uses only `alloc`. The renderer
//!   writes through [`core::fmt::Write`]; it does not require `std::io`.
//!   The `std` feature enables `std::error::Error` impls on the error
//!   types and lets callers use `std::io::Write` adapters.
//! * **Zero dependencies.** The crate depends only on `theoria_syntax`,
//!   which itself has no external dependencies.
//!
//! ## What is deferred
//!
//! * **miette adapter.** PLAN.md §7 names miette. The plan's intent — a
//!   structured diagnostic model with rich rendering — is what this
//!   crate delivers. A follow-up can add `impl Into<miette::Report> for
//!   Diagnostic` behind a `miette` feature flag without touching the
//!   model or the renderer.
//! * **Color output.** [`ColorChoice`] is present, but the renderer
//!   currently ignores it. ANSI escape sequences and terminal detection
//!   are a follow-up.
//! * **Unicode box-drawing.** The renderer uses ASCII (`|`, `-`, `^`).
//!   Unicode (`│`, `─`, `╰─`) is a follow-up.
//! * **Multi-label single-snippet rendering.** Multiple labels on the
//!   same source line are rendered as separate snippets. Emitting them
//!   in one shared snippet (with the caret alignment rustc produces)
//!   is a follow-up.
//! * **Wrapping.** Lines wider than [`RenderOptions::width`] are not
//!   wrapped; the source line is emitted verbatim.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

extern crate alloc;

pub mod code;
pub mod diagnostic;
pub mod label;
pub mod render;
pub mod severity;
pub mod source;

pub use code::DiagnosticCode;
pub use diagnostic::Diagnostic;
pub use label::{Label, LabelStyle, SourceId};
pub use render::{ColorChoice, RenderOptions, Renderer};
pub use severity::Severity;
pub use source::SourceFile;

pub use theoria_syntax::{SourceMap, Span};
