//! The concrete syntax tree.
//!
//! The CST is fully owned (`String` fields) and carries spans on every
//! node. It is not designed for round-tripping — reproducing the exact
//! source requires a formatter, which is a later delivery.
//!
//! Types and expressions share the [`Expr`] type: `Nat -> Nat` in a
//! return-type position is `Expr::BinOp(.., Arrow, ..)`. This mirrors
//! Lean 4 and keeps the elaborator's job simple.

use crate::span::Span;
use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

/// A dotted path, e.g. `Standard.Geometry.Shapes`.
#[derive(Clone, Debug)]
pub struct Path {
    /// The dot-separated segments.
    pub segments: Vec<PathSegment>,
    /// Covers the first segment through the last.
    pub span: Span,
}

/// One segment of a [`Path`].
#[derive(Clone, Debug)]
pub struct PathSegment {
    /// The segment text.
    pub text: String,
    /// Span of the segment text.
    pub span: Span,
}

/// The top-level node. Even an empty file produces a `Module` with all
/// fields empty.
#[derive(Clone, Debug)]
pub struct Module {
    /// The `Module` declaration, if present.
    pub decl: Option<ModuleDecl>,
    /// All `Import` declarations, in source order.
    pub imports: Vec<ImportDecl>,
    /// All `Structure` declarations, in source order.
    pub structures: Vec<StructureDef>,
    /// All `Function` definitions, in source order.
    pub functions: Vec<FunctionDef>,
    /// All `Theorem` declarations, in source order.
    pub theorems: Vec<TheoremDef>,
    /// Covers the whole source.
    pub span: Span,
}

/// The body of a theorem's `Proof:` section.
#[derive(Clone, Debug)]
pub enum ProofBlock {
    /// `Proof: <expr>` on a single line.
    Term(Expr),
    /// `Proof:` followed by an indented block of numbered steps.
    Steps(Vec<ProofStep>),
}

impl ProofBlock {
    /// The span covering the block.
    #[must_use]
    pub fn span(&self) -> Span {
        match self {
            ProofBlock::Term(e) => expr_span(e),
            ProofBlock::Steps(steps) => {
                let first = steps
                    .first()
                    .expect("parser guarantees at least one step")
                    .span;
                let last = steps
                    .last()
                    .expect("parser guarantees at least one step")
                    .span;
                Span::new(first.start, last.end)
            }
        }
    }
}

/// A single numbered step inside a `Proof:` block.
#[derive(Clone, Debug)]
pub struct ProofStep {
    /// The 1-based step number, as written.
    pub number: u32,
    /// Span of the number (used for "expected step N, found M" errors).
    pub number_span: Span,
    /// What the step does.
    pub kind: ProofStepKind,
    /// Optional `[justification]` text. Retained verbatim for a future
    /// tactic engine; has no elaboration effect today.
    pub justification: Option<String>,
    /// Covers the number through the trailing newline.
    pub span: Span,
}

/// What a [`ProofStep`] does.
#[derive(Clone, Debug)]
pub enum ProofStepKind {
    /// `Let name [ : ty ] = value` — introduce a local binding.
    ///
    /// If `ty` is `None`, the elaborator infers it from `value` via the
    /// kernel's type checker.
    Let {
        /// The bound name.
        name: String,
        /// Span of the name.
        name_span: Span,
        /// Optional type annotation.
        ty: Option<Expr>,
        /// The bound value.
        value: Expr,
    },
    /// `Have name : ty = proof` — introduce a local lemma.
    ///
    /// The `proof` is checked against `ty` in the current context.
    Have {
        /// The lemma's name.
        name: String,
        /// Span of the name.
        name_span: Span,
        /// The lemma's type.
        ty: Expr,
        /// The proof term.
        proof: Expr,
    },
    /// `Exact term` — provide the theorem's proof term.
    ///
    /// Must be the last step of the block.
    Exact {
        /// The proof term.
        term: Expr,
    },
    /// `From name` — use a named hypothesis as the goal's proof.
    ///
    /// Reads like the plan's `[from h]` justification when the entire
    /// proof of the current goal is a single hypothesis. Elaborates
    /// identically to `Exact name`; the distinct keyword exists so that
    /// the surface reads naturally.
    From {
        /// The hypothesis's name.
        name: String,
        /// Span of the name.
        name_span: Span,
    },
}

/// Helper to get span of an Expr.
fn expr_span(e: &Expr) -> Span {
    match e {
        Expr::Ident { span, .. }
        | Expr::Nat { span, .. }
        | Expr::Float { span, .. }
        | Expr::Str { span, .. }
        | Expr::Field { span, .. }
        | Expr::Index { span, .. }
        | Expr::Call { span, .. }
        | Expr::BinOp { span, .. }
        | Expr::UnOp { span, .. }
        | Expr::Paren { span, .. }
        | Expr::Lam { span, .. }
        | Expr::If(IfExpr { span, .. })
        | Expr::Match(MatchExpr { span, .. }) => *span,
    }
}

/// A `Theorem name:` block with `Given` / `Assume` / `Show` / `Proof` /
/// `QED` sections.
#[derive(Clone, Debug)]
pub struct TheoremDef {
    /// The theorem's name.
    pub name: String,
    /// Span of the name.
    pub name_span: Span,
    /// The `Given:` binders (parameters of the theorem).
    pub given: Vec<Param>,
    /// The `Assume:` hypotheses.
    pub assume: Vec<Hypothesis>,
    /// The `Show:` goal proposition.
    pub show: Expr,
    /// The `Proof:` block: either a single term-mode expression or a
    /// sequence of numbered steps.
    pub proof: ProofBlock,
    /// Covers from the `Theorem` keyword through `QED`.
    pub span: Span,
}

/// A single `Assume:` hypothesis: `name : ty`.
#[derive(Clone, Debug)]
pub struct Hypothesis {
    /// The hypothesis name.
    pub name: String,
    /// Span of the name.
    pub name_span: Span,
    /// The hypothesis's type (its proposition).
    pub ty: Expr,
    /// Covers from the name through the type.
    pub span: Span,
}

/// A `Module <path>` declaration.
#[derive(Clone, Debug)]
pub struct ModuleDecl {
    /// The module path.
    pub path: Path,
    /// Covers from the `Module` keyword through the path.
    pub span: Span,
}

/// An `Import <path> [(names)]` declaration.
#[derive(Clone, Debug)]
pub struct ImportDecl {
    /// The imported path.
    pub path: Path,
    /// `None` = import every name from the path.
    /// `Some([])` = import nothing (syntactically valid, semantically odd).
    pub names: Option<Vec<ImportName>>,
    /// Covers from the `Import` keyword through the path or name list.
    pub span: Span,
}

/// One name in an import list.
#[derive(Clone, Debug)]
pub struct ImportName {
    /// The imported name.
    pub text: String,
    /// Span of the name.
    pub span: Span,
}

/// A `Function name(params) [-> ret]:` block definition.
#[derive(Clone, Debug)]
pub struct FunctionDef {
    /// The function name.
    pub name: String,
    /// Span of the name.
    pub name_span: Span,
    /// The value parameters.
    pub params: Vec<Param>,
    /// The declared return type, if present.
    pub return_ty: Option<Expr>,
    /// The body statements (at least one).
    pub body: Vec<Stmt>,
    /// Covers from the `Function` keyword to the end of the body's Dedent.
    pub span: Span,
}

/// A `Structure Name(P₁, ..., Pₖ):` block definition.
#[derive(Clone, Debug)]
pub struct StructureDef {
    /// The structure's name, as written in source.
    pub name: String,
    /// Span of the name token.
    pub name_span: Span,
    /// The parameters, in declaration order. Empty for a parameterless
    /// structure.
    pub params: Vec<Param>,
    /// The fields, in declaration order. Guaranteed non-empty by the
    /// parser.
    pub fields: Vec<FieldDef>,
    /// Covers from the `Structure` keyword to the trailing `Dedent`.
    pub span: Span,
}

/// A single field inside a [`StructureDef`] block: `name : ty`.
#[derive(Clone, Debug)]
pub struct FieldDef {
    /// The field's name.
    pub name: String,
    /// Span of the name token.
    pub name_span: Span,
    /// The field's type, as a surface expression.
    pub ty: Expr,
    /// Covers from the field's name token to its trailing `Newline`.
    pub span: Span,
}

/// A single parameter of a `λ`-abstraction: `(name : ty)`.
#[derive(Clone, Debug)]
pub struct LamParam {
    /// The parameter name.
    pub name: String,
    /// Span of the name.
    pub name_span: Span,
    /// The declared type.
    pub ty: Expr,
    /// Covers the name through the type.
    pub span: Span,
}

/// A single value parameter `name : ty`.
#[derive(Clone, Debug)]
pub struct Param {
    /// The parameter name.
    pub name: String,
    /// Span of the name.
    pub name_span: Span,
    /// The declared type.
    pub ty: Expr,
    /// Covers the name through the type.
    pub span: Span,
}

/// A statement inside a function body.
#[derive(Clone, Debug)]
pub enum Stmt {
    /// `let name : ty = value`.
    Let {
        /// The bound name.
        name: String,
        /// Span of the name.
        name_span: Span,
        /// The declared type.
        ty: Expr,
        /// The bound value.
        value: Expr,
        /// Covers from the `let` keyword through the value.
        span: Span,
    },
    /// `return <expr>`.
    Return {
        /// The returned expression.
        value: Expr,
        /// Covers from the `return` keyword through the expression.
        span: Span,
    },
    // Future: If { ... }, Match { ... }.
}

impl Stmt {
    /// The span covering the statement.
    #[must_use]
    pub fn span(&self) -> Span {
        match self {
            Stmt::Let { span, .. } | Stmt::Return { span, .. } => *span,
        }
    }
}

/// A surface expression.
///
/// `Paren` is the only node retained purely for source fidelity; the
/// elaborator strips it, and a future formatter needs it to round-trip
/// parenthesised expressions.
#[derive(Clone, Debug)]
pub enum Expr {
    /// A bare identifier.
    Ident {
        /// The identifier text.
        text: String,
        /// Span of the identifier.
        span: Span,
    },
    /// A natural-number literal.
    Nat {
        /// The parsed value.
        value: u64,
        /// Span of the literal.
        span: Span,
    },
    /// A floating-point literal.
    Float {
        /// The parsed value.
        value: f64,
        /// Span of the literal.
        span: Span,
    },
    /// A string literal (unescaped value).
    Str {
        /// The unescaped string contents.
        value: String,
        /// Span of the literal, including quotes.
        span: Span,
    },
    /// `obj.field`.
    Field {
        /// The object expression.
        obj: Box<Expr>,
        /// The field name.
        field: String,
        /// Span of the field name.
        field_span: Span,
        /// Covers `obj` through the field name.
        span: Span,
    },
    /// `obj[index]`.
    Index {
        /// The object expression.
        obj: Box<Expr>,
        /// The index expression.
        index: Box<Expr>,
        /// Covers `obj` through `]`.
        span: Span,
    },
    /// `func(a, b, ...)`.
    Call {
        /// The callee expression.
        func: Box<Expr>,
        /// The argument expressions.
        args: Vec<Expr>,
        /// Covers `func` through `)`.
        span: Span,
    },
    /// `lhs op rhs`.
    BinOp {
        /// The left-hand side.
        lhs: Box<Expr>,
        /// The operator.
        op: BinOp,
        /// The right-hand side.
        rhs: Box<Expr>,
        /// Covers `lhs` through `rhs`.
        span: Span,
    },
    /// `-operand`.
    UnOp {
        /// The operator (currently only [`UnOp::Neg`]).
        op: UnOp,
        /// The operand.
        operand: Box<Expr>,
        /// Covers the `-` through the operand.
        span: Span,
    },
    /// `(inner)`.
    Paren {
        /// The parenthesised expression.
        inner: Box<Expr>,
        /// Covers `(` through `)`.
        span: Span,
    },
    /// `λ (x : A) (y : B). body` (at least one parameter).
    Lam {
        /// The abstraction parameters, in declaration order.
        params: Vec<LamParam>,
        /// The body expression.
        body: Box<Expr>,
        /// Covers from the `λ` to the end of the body.
        span: Span,
    },
    /// `if cond then a else b`.
    If(IfExpr),
    /// `match scrutinee: case ... => ...`.
    Match(MatchExpr),
}

/// `if cond then a else b` — an expression.
#[derive(Clone, Debug)]
pub struct IfExpr {
    /// The condition expression.
    pub cond: Box<Expr>,
    /// The then-branch expression.
    pub then_branch: Box<Expr>,
    /// The else-branch expression.
    pub else_branch: Box<Expr>,
    /// Covers from the `if` keyword to the end of the `else` branch.
    pub span: Span,
}

/// `match scrutinee: case P₁ => e₁ | ... | case Pₙ => eₙ`.
#[derive(Clone, Debug)]
pub struct MatchExpr {
    /// The scrutinee expression.
    pub scrutinee: Box<Expr>,
    /// The arms, in source order.
    pub arms: Vec<MatchArm>,
    /// Covers from `match` to the end of the last arm.
    pub span: Span,
}

/// One `case` arm of a [`MatchExpr`].
#[derive(Clone, Debug)]
pub struct MatchArm {
    /// The arm's pattern.
    pub pattern: Pattern,
    /// The arm's body expression.
    pub body: Expr,
    /// Covers from `case` to the end of the arm's body.
    pub span: Span,
}

/// A flat match pattern.
#[derive(Clone, Debug)]
pub enum Pattern {
    /// `_` — matches anything, binds nothing.
    Wildcard(Span),
    /// A bare identifier — matches anything, binds the scrutinee.
    Var {
        /// The bound name.
        name: String,
        /// Span of the name.
        name_span: Span,
    },
    /// A constructor pattern, e.g. `Nat.zero` or `Nat.succ(k)`.
    Constructor {
        /// The dotted constructor name as written.
        name: String,
        /// Span of the name.
        name_span: Span,
        /// The field patterns.
        args: Vec<PatternArg>,
        /// Covers the name through the argument list (or just the name).
        span: Span,
    },
}

/// One argument of a constructor pattern.
#[derive(Clone, Debug)]
pub enum PatternArg {
    /// `_` inside a constructor's argument list.
    Wildcard(Span),
    /// A named binder.
    Var {
        /// The bound name.
        name: String,
        /// Span of the name.
        name_span: Span,
    },
}

/// A binary operator.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BinOp {
    /// `+`
    Add,
    /// `-`
    Sub,
    /// `*`
    Mul,
    /// `/`
    Div,
    /// `^` (right-associative)
    Pow,
    /// `==`
    Eq,
    /// `!=`
    Ne,
    /// `<`
    Lt,
    /// `<=`
    Le,
    /// `>`
    Gt,
    /// `>=`
    Ge,
    /// `->` (right-associative)
    Arrow,
}

/// A unary operator.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum UnOp {
    /// Unary `-`.
    Neg,
}
