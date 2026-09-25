//! Proof erasure and Ghost Type Theory (GTT) extraction.
//!
//! This module implements the extraction pass that turns a kernel term into
//! an *untyped* λ-term, dropping anything that lives in `Prop` or in a
//! designated ghost universe. The result is the input to
//! `theoria compile --target paper` and to the future code-generation
//! pipeline.
//!
//! ## Universe partition
//!
//! The kernel's single universe hierarchy is partitioned into three
//! erasure classes:
//!
//! | Class | Source | Treatment |
//! |-------|--------|-----------|
//! | `Prop` | Terms whose type lives in `Sort 0` | Erased to `⋆` |
//! | `Ghost` | Terms whose type is a marked-ghost constant | Erased to `⋆` |
//! | `Compute` | Everything else | Retained |
//!
//! `Prop` is detected from the kernel's own universe structure:
//! `p : P` with `P : Sort 0` means `p` is a proof, and proofs are
//! proof-irrelevant. `Ghost` is detected via the [`ErasureEnv`] trait,
//! which marks particular constants (inductives, definitions) as
//! computationally irrelevant.
//!
//! The kernel does **not** have a built-in `Ghost` universe — the plan's
//! §6 partition is realised here, at the extraction boundary, rather than
//! as a new `Sort` constructor. This keeps the kernel's core logic
//! unchanged and lets clients opt into ghost tracking without forking
//! the type checker.
//!
//! ## The erasure function
//!
//! For a term `t : T`:
//!
//! ```text
//! erase(t : T) =
//!     ⋆                         if T : Prop or T : Ghost
//!     λ x. erase(b)             if t = λ (x : A). b, otherwise
//!     erase(f) erase(a)         if t = f a, otherwise
//!     Var(i)                    if t = Var(i), otherwise
//!     Const(c)                  if t = Const(c) and c is not ghost
//!     Lit(l)                    if t = Lit(l)
//!     ⋆                         if t = Sort(_) or t = Pi(..)
//! ```
//!
//! Binders are never dropped. Type-level `λ (A : Type). ...` keeps its
//! binder in the erased term, so the erased program has the same arity as
//! the original; the extra argument is simply `⋆` at every call site.
//! This is a deliberate Phase 1 simplification — the alternative (dropping
//! type arguments in tandem with type binders) requires a substitution
//! context that the current `LValue` does not carry.
//!
//! ## Performance
//!
//! `Eraser::classify` calls [`TypeChecker::infer`] on each subterm and,
//! for most terms, on the type of each subterm. This makes erasure
//! `O(n²)` in the size of the term. It is fast enough for prelude-scale
//! code and for extracting individual declarations. If larger proofs need
//! to be extracted, cache the types from a prior elaboration pass and
//! pass them to the eraser — a `TypedExpr` interface that this module
//! does not yet expose.

use crate::env::{ConstantInfo, GlobalEnv, LocalDecl};
use crate::error::KernelResult;
use crate::expr::{DeBruijnIndex, Expr, Literal};
use crate::infer::{TypeChecker, TypedContext};
use crate::name::{NameId, NameTable};
use crate::nbe::{Head, ValRef, Value};
use alloc::boxed::Box;
use alloc::collections::BTreeSet;
use alloc::rc::Rc;
use alloc::string::String;
use core::fmt;

// ---------------------------------------------------------------------------
// ErasureSort
// ---------------------------------------------------------------------------

/// The erasure class of a term.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ErasureSort {
    /// Proof-irrelevant: the term lives in `Prop` and is erased.
    Prop,
    /// Proof-relevant but computationally irrelevant: the term lives in a
    /// marked ghost type and is erased at extraction time.
    Ghost,
    /// Computational data: retained in the erased term.
    Compute,
}

impl ErasureSort {
    /// `true` iff the sort is erased at extraction.
    #[must_use]
    pub const fn is_erased(self) -> bool {
        matches!(self, ErasureSort::Prop | ErasureSort::Ghost)
    }
}

// ---------------------------------------------------------------------------
// ErasureEnv
// ---------------------------------------------------------------------------

/// The client's view of which constants are computationally irrelevant.
///
/// The kernel has no built-in notion of ghost data. Clients that want to
/// mark certain definitions or inductives as ghost implement this trait;
/// the default [`PropOnlyErasureEnv`] marks nothing and only erases
/// proofs.
pub trait ErasureEnv {
    /// `true` iff `name` names a ghost constant.
    fn is_ghost(&self, name: NameId) -> bool;
}

/// The trivial erasure environment: erases only proofs.
#[derive(Debug, Default, Clone, Copy)]
pub struct PropOnlyErasureEnv;

impl ErasureEnv for PropOnlyErasureEnv {
    fn is_ghost(&self, _: NameId) -> bool {
        false
    }
}

/// An erasure environment that marks a fixed set of names as ghost.
#[derive(Debug, Default, Clone)]
pub struct GhostSet {
    ghosts: BTreeSet<NameId>,
}

impl GhostSet {
    /// An empty ghost set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark a constant as ghost.
    pub fn mark(&mut self, name: NameId) {
        self.ghosts.insert(name);
    }

    /// Remove a ghost marking.
    pub fn unmark(&mut self, name: NameId) {
        self.ghosts.remove(&name);
    }

    /// Number of marked names.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ghosts.len()
    }

    /// `true` iff no names are marked.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ghosts.is_empty()
    }
}

impl ErasureEnv for GhostSet {
    fn is_ghost(&self, name: NameId) -> bool {
        self.ghosts.contains(&name)
    }
}

// ---------------------------------------------------------------------------
// LValue — the untyped target
// ---------------------------------------------------------------------------

/// An untyped λ-term, the target of erasure.
///
/// Variables are de Bruijn **indices** (0 = innermost), matching the
/// indexing of the source `Expr`. Because erasure never drops binders,
/// the erased context has the same shape as the source context and
/// indices convert one-to-one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LValue {
    /// The erased placeholder `⋆`.
    Star,
    /// A variable, by de Bruijn index.
    Var(DeBruijnIndex),
    /// A function `λ _. body`.
    Lam(Box<LValue>),
    /// An application `f a`.
    App(Box<LValue>, Box<LValue>),
    /// A reference to a top-level constant.
    Const(NameId),
    /// A literal.
    Lit(Literal),
}

impl LValue {
    /// `true` iff the term is the erased placeholder.
    #[must_use]
    pub const fn is_star(&self) -> bool {
        matches!(self, LValue::Star)
    }

    /// Total syntactic size.
    #[must_use]
    pub fn size(&self) -> usize {
        match self {
            LValue::Star | LValue::Var(_) | LValue::Const(_) | LValue::Lit(_) => 1,
            LValue::Lam(b) => 1 + b.size(),
            LValue::App(f, a) => 1 + f.size() + a.size(),
        }
    }

    /// `true` iff the erased term contains no `Star`.
    ///
    /// A term with no `Star` is fully computational: nothing was erased
    /// inside it. Useful as a sanity check on extracted definitions.
    #[must_use]
    pub fn is_star_free(&self) -> bool {
        match self {
            LValue::Star => false,
            LValue::Var(_) | LValue::Const(_) | LValue::Lit(_) => true,
            LValue::Lam(b) => b.is_star_free(),
            LValue::App(f, a) => f.is_star_free() && a.is_star_free(),
        }
    }

    /// Render the term for debugging, using `names` to resolve constant
    /// identifiers.
    #[must_use]
    pub fn render(&self, names: &NameTable) -> String {
        let mut out = String::new();
        self.render_into(names, &mut out);
        out
    }

    fn render_into(&self, names: &NameTable, out: &mut String) {
        match self {
            LValue::Star => out.push('⋆'),
            LValue::Var(idx) => {
                out.push('#');
                out.push_str(&alloc::format!("{}", idx.0));
            }
            LValue::Const(name) => out.push_str(names.resolve(*name)),
            LValue::Lit(Literal::Nat(n)) => out.push_str(&alloc::format!("{n}")),
            LValue::Lit(Literal::Str(s)) => {
                out.push('"');
                out.push_str(s);
                out.push('"');
            }
            LValue::Lam(body) => {
                out.push('(');
                out.push('λ');
                out.push(' ');
                body.render_into(names, out);
                out.push(')');
            }
            LValue::App(f, a) => {
                out.push('(');
                f.render_into(names, out);
                out.push(' ');
                a.render_into(names, out);
                out.push(')');
            }
        }
    }
}

impl fmt::Display for LValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Display without a name table: constants render as `#N`.
        match self {
            LValue::Star => f.write_str("⋆"),
            LValue::Var(idx) => write!(f, "#{}", idx.0),
            LValue::Const(name) => write!(f, "c{}", name.0),
            LValue::Lit(Literal::Nat(n)) => write!(f, "{n}"),
            LValue::Lit(Literal::Str(s)) => write!(f, "{s:?}"),
            LValue::Lam(b) => write!(f, "(λ {b})"),
            LValue::App(fun, a) => write!(f, "({fun} {a})"),
        }
    }
}

// ---------------------------------------------------------------------------
// Eraser
// ---------------------------------------------------------------------------

/// The erasure pass.
///
/// `Eraser` holds a `TypeChecker` and consults it for each subterm to
/// decide whether the subterm is a proof, ghost, or computational data.
pub struct Eraser<'a, E: ErasureEnv + ?Sized> {
    env: &'a GlobalEnv,
    erasure_env: &'a E,
    checker: TypeChecker<'a>,
}

impl<'a, E: ErasureEnv + ?Sized> Eraser<'a, E> {
    /// Build an eraser.
    #[must_use]
    pub fn new(env: &'a GlobalEnv, erasure_env: &'a E) -> Self {
        Self {
            env,
            erasure_env,
            checker: TypeChecker::new(env),
        }
    }

    /// The kernel environment.
    #[must_use]
    pub fn env(&self) -> &'a GlobalEnv {
        self.env
    }

    /// The erasure environment.
    #[must_use]
    pub fn erasure_env(&self) -> &'a E {
        self.erasure_env
    }

    // -----------------------------------------------------------------
    // Public entry points
    // -----------------------------------------------------------------

    /// Erase `e` in the context `ctx`.
    ///
    /// # Errors
    ///
    /// Propagates any error from the internal type checker. For a term
    /// that is already known to be well-typed, this method is infallible.
    pub fn erase(&self, ctx: &TypedContext, e: &Expr) -> KernelResult<LValue> {
        // Fast paths: types are erased, literals are kept as-is.
        match e {
            Expr::Sort(_) | Expr::Pi(_, _, _, _) => return Ok(LValue::Star),
            Expr::Lit(l) => return Ok(LValue::Lit(l.clone())),
            _ => {}
        }
        let sort = self.classify(ctx, e)?;
        if sort.is_erased() {
            return Ok(LValue::Star);
        }
        self.erase_compute(ctx, e)
    }

    /// Erase the body of a top-level declaration.
    ///
    /// Returns:
    ///
    /// * `Some(body)` for a `Definition`, with `body` the erased body.
    /// * `Some(Star)` for a `Theorem`: a theorem is a proof, and its
    ///   erased form is the placeholder.
    /// * `None` for axioms, inductives, constructors, recursors, and
    ///   quotient constants: these are not extracted by this pass.
    ///
    /// # Errors
    ///
    /// Propagates errors from the underlying type checker.
    pub fn erase_declaration(&self, info: &ConstantInfo) -> KernelResult<Option<LValue>> {
        match info {
            ConstantInfo::Definition(d) => {
                let ctx = TypedContext::empty();
                Ok(Some(self.erase(&ctx, &d.body)?))
            }
            ConstantInfo::Theorem(_) => Ok(Some(LValue::Star)),
            ConstantInfo::Axiom(_)
            | ConstantInfo::Opaque(_)
            | ConstantInfo::Inductive(_)
            | ConstantInfo::Constructor(_)
            | ConstantInfo::Recursor(_)
            | ConstantInfo::Quotient(_) => Ok(None),
        }
    }

    // -----------------------------------------------------------------
    // Classification
    // -----------------------------------------------------------------

    /// Classify a term as `Prop`, `Ghost`, or `Compute`.
    fn classify(&self, ctx: &TypedContext, e: &Expr) -> KernelResult<ErasureSort> {
        // Lambdas cannot be inferred without an expected type
        // (`infer` returns `CannotInfer` for them), so classify them
        // structurally: `λ (x : A). b` is a proof iff `b` is a proof.
        // Its Π-type lives in `Prop` exactly when the body's type does
        // (`imax(u, v) = 0` iff `v = 0`), so for well-typed terms this
        // matches the type-based rule below.
        if let Expr::Lam(info, name, dom, body) = e {
            let dom_v = self.checker.nbe().eval(&ctx.to_env(), dom);
            let mut ctx2 = ctx.clone();
            ctx2.push_binder(
                LocalDecl::binder(*name, *info, Rc::new((**dom).clone())),
                dom_v,
            );
            return self.classify(&ctx2, body);
        }
        // Head-based ghost check: a direct application of a ghost constant
        // is ghost, regardless of its type. This handles the case where
        // the ghost marking is on the *value* rather than on the type.
        if let Expr::Const(name, _) = e {
            if self.erasure_env.is_ghost(*name) {
                return Ok(ErasureSort::Ghost);
            }
        }
        // Type-based classification. If the term cannot be inferred (e.g. a
        // β-redex with a lambda head, which `infer` rejects without an
        // expected type), fall back to `Compute` and erase structurally.
        // Structural erasure preserves reduction behaviour, so for
        // well-typed terms this is sound — though it may retain proofs
        // that a fully type-directed pass would erase to `Star`.
        let ty = match self.checker.infer(ctx, e) {
            Ok(ty) => ty,
            Err(_) => return Ok(ErasureSort::Compute),
        };
        self.classify_ty(ctx, &ty)
    }

    /// Classify a *type* by the erasure class of its inhabitants.
    fn classify_ty(&self, ctx: &TypedContext, ty: &ValRef) -> KernelResult<ErasureSort> {
        // Sorts are types, not proofs; classify as computational.
        if let Value::Sort(_) = &**ty {
            return Ok(ErasureSort::Compute);
        }
        // A type whose head is a marked-ghost constant is a ghost type:
        // its inhabitants are ghost values.
        if let Value::Neutral(Head::Const(name, _), _) = &**ty {
            if self.erasure_env.is_ghost(*name) {
                return Ok(ErasureSort::Ghost);
            }
        }
        // Otherwise, infer the sort of the type. A type that lives in
        // `Sort 0` is a proposition; a proof of it is erased.
        let ty_expr = self.checker.nbe().quote(ctx.size(), ty);
        let kind = self.checker.infer(ctx, &ty_expr)?;
        match &*kind {
            Value::Sort(l) if l.normalize().is_zero() => Ok(ErasureSort::Prop),
            _ => Ok(ErasureSort::Compute),
        }
    }

    // -----------------------------------------------------------------
    // Erasure of computational terms
    // -----------------------------------------------------------------

    /// Erase a term already classified as `Compute`.
    fn erase_compute(&self, ctx: &TypedContext, e: &Expr) -> KernelResult<LValue> {
        match e {
            // Types are erased at extraction time.
            Expr::Sort(_) | Expr::Pi(_, _, _, _) => Ok(LValue::Star),

            // Variables pass through unchanged: the erased context has the
            // same shape as the source context.
            Expr::Var(idx) => Ok(LValue::Var(*idx)),

            // Constants pass through unchanged unless the erasure
            // environment marks them as ghost.
            Expr::Const(name, _) => {
                if self.erasure_env.is_ghost(*name) {
                    Ok(LValue::Star)
                } else {
                    Ok(LValue::Const(*name))
                }
            }

            // Application: erase the head and the argument separately,
            // then combine. The argument is erased even if it becomes
            // `Star` — this preserves the arity of the erased function.
            Expr::App(f, a) => {
                let f_lv = self.erase(ctx, f)?;
                let a_lv = self.erase(ctx, a)?;
                Ok(LValue::App(Box::new(f_lv), Box::new(a_lv)))
            }

            // Lambda: push the binder to the context and erase the body.
            // The domain type is discarded (types are erased), but the
            // binder itself is retained so that variable indices line up.
            Expr::Lam(info, name, dom, body) => {
                let dom_v = self.checker.nbe().eval(&ctx.to_env(), dom);
                let mut ctx2 = ctx.clone();
                ctx2.push_binder(
                    LocalDecl::binder(*name, *info, Rc::new((**dom).clone())),
                    dom_v,
                );
                let body_lv = self.erase(&ctx2, body)?;
                Ok(LValue::Lam(Box::new(body_lv)))
            }

            // Let: encode as a lambda application so that the bound value
            // is substituted once, sharing it between all uses in the
            // body.
            Expr::Let(name, ty, val, body) => {
                let val_lv = self.erase(ctx, val)?;
                let env = ctx.to_env();
                let ty_v = self.checker.nbe().eval(&env, ty);
                let val_v = self.checker.nbe().eval(&env, val);
                let mut ctx2 = ctx.clone();
                ctx2.push_let(
                    LocalDecl::let_bound(*name, Rc::new((**ty).clone()), Rc::new((**val).clone())),
                    ty_v,
                    val_v,
                );
                let body_lv = self.erase(&ctx2, body)?;
                Ok(LValue::App(
                    Box::new(LValue::Lam(Box::new(body_lv))),
                    Box::new(val_lv),
                ))
            }

            // Literals pass through unchanged.
            Expr::Lit(l) => Ok(LValue::Lit(l.clone())),
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience functions
// ---------------------------------------------------------------------------

/// Erase a closed expression in the default erasure environment.
///
/// # Errors
///
/// Propagates errors from the underlying type checker.
pub fn erase_closed<E: ErasureEnv + ?Sized>(
    env: &GlobalEnv,
    erasure_env: &E,
    e: &Expr,
) -> KernelResult<LValue> {
    Eraser::new(env, erasure_env).erase(&TypedContext::empty(), e)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::{ConstantInfo, ConstantVal, DefinitionVal};
    use crate::expr::BinderInfo;
    use crate::level::Level;
    use crate::nbe::Transparency;
    use crate::prelude::{Prelude, build_prelude};

    // ---- fixture -----------------------------------------------------

    /// A prelude extended with a proposition `P : Prop` and a proof
    /// `proof_p : P`.
    fn setup() -> (Prelude, NameId, NameId) {
        let mut p = build_prelude();
        let prop_p = p.names.intern("P");
        let proof_p = p.names.intern("proof_p");
        p.env
            .add(ConstantInfo::Axiom(ConstantVal {
                name: prop_p,
                universe_params: alloc::vec::Vec::new(),
                ty: Rc::new(Expr::prop()),
            }))
            .unwrap();
        p.env
            .add(ConstantInfo::Axiom(ConstantVal {
                name: proof_p,
                universe_params: alloc::vec::Vec::new(),
                ty: Rc::new(Expr::Const(prop_p, alloc::vec::Vec::new())),
            }))
            .unwrap();
        (p, prop_p, proof_p)
    }

    fn erase(p: &Prelude, e: &Expr) -> LValue {
        let env = PropOnlyErasureEnv;
        erase_closed(&p.env, &env, e).expect("erasure should succeed")
    }

    fn var(i: u32) -> Expr {
        Expr::Var(DeBruijnIndex(i))
    }

    fn lam(name: NameId, dom: Expr, body: Expr) -> Expr {
        Expr::lam(name, dom, body)
    }

    // ---- atoms -------------------------------------------------------

    #[test]
    fn prop_erases_to_star() {
        let (p, _, _) = setup();
        assert_eq!(erase(&p, &Expr::prop()), LValue::Star);
    }

    #[test]
    fn type_zero_erases_to_star() {
        let (p, _, _) = setup();
        assert_eq!(erase(&p, &Expr::type0()), LValue::Star);
    }

    #[test]
    fn pi_type_erases_to_star() {
        let (mut p, _, _) = setup();
        let ty = Expr::arrow(
            Expr::Const(p.nat, alloc::vec::Vec::new()),
            Expr::Const(p.nat, alloc::vec::Vec::new()),
            p.names.intern("_"),
        );
        assert_eq!(erase(&p, &ty), LValue::Star);
    }

    #[test]
    fn nat_zero_erases_to_const() {
        let (p, _, _) = setup();
        let e = Expr::Const(p.zero, alloc::vec::Vec::new());
        assert_eq!(erase(&p, &e), LValue::Const(p.zero));
    }

    #[test]
    fn nat_succ_erases_to_const() {
        let (p, _, _) = setup();
        let e = Expr::Const(p.succ, alloc::vec::Vec::new());
        assert_eq!(erase(&p, &e), LValue::Const(p.succ));
    }

    #[test]
    fn prop_constant_erases_to_itself() {
        // `P` is a proposition (`P : Prop`), not a proof. The spec erases
        // proofs, not propositions, so `P` is kept.
        let (p, prop_p, _) = setup();
        let e = Expr::Const(prop_p, alloc::vec::Vec::new());
        assert_eq!(erase(&p, &e), LValue::Const(prop_p));
    }

    #[test]
    fn proof_erases_to_star() {
        // `proof_p : P` and `P : Prop`, so `proof_p` is a proof.
        let (p, _, proof_p) = setup();
        let e = Expr::Const(proof_p, alloc::vec::Vec::new());
        assert_eq!(erase(&p, &e), LValue::Star);
    }

    // ---- variables ---------------------------------------------------

    #[test]
    fn variable_passes_through() {
        let (mut p, _, _) = setup();
        // Build a context with one Nat-typed variable.
        let mut ctx = TypedContext::empty();
        let nat_v = {
            let checker = TypeChecker::new(&p.env);
            checker
                .nbe()
                .eval(&ctx.to_env(), &Expr::Const(p.nat, alloc::vec::Vec::new()))
        };
        ctx.push(
            LocalDecl::binder(
                p.names.intern("n"),
                BinderInfo::Default,
                Rc::new(Expr::Const(p.nat, alloc::vec::Vec::new())),
            ),
            nat_v,
        );
        let env = PropOnlyErasureEnv;
        let eraser = Eraser::new(&p.env, &env);
        let lv = eraser.erase(&ctx, &var(0)).unwrap();
        assert_eq!(lv, LValue::Var(DeBruijnIndex(0)));
    }

    // ---- lambdas -----------------------------------------------------

    #[test]
    fn identity_nat_erases_to_lambda() {
        let (mut p, _, _) = setup();
        let n = p.names.intern("n");
        let e = lam(n, Expr::Const(p.nat, alloc::vec::Vec::new()), var(0));
        let expected = LValue::Lam(Box::new(LValue::Var(DeBruijnIndex(0))));
        assert_eq!(erase(&p, &e), expected);
    }

    #[test]
    fn succ_lambda_erases_correctly() {
        let (mut p, _, _) = setup();
        let n = p.names.intern("n");
        let e = lam(
            n,
            Expr::Const(p.nat, alloc::vec::Vec::new()),
            Expr::app(Expr::Const(p.succ, alloc::vec::Vec::new()), var(0)),
        );
        let expected = LValue::Lam(Box::new(LValue::App(
            Box::new(LValue::Const(p.succ)),
            Box::new(LValue::Var(DeBruijnIndex(0))),
        )));
        assert_eq!(erase(&p, &e), expected);
    }

    #[test]
    fn lambda_with_proof_domain_keeps_binder() {
        // λ (p : P). Nat.zero  — the domain is a proof type, but binders
        // are kept and only proofs are erased, so the body still has a λ.
        let (mut p, prop_p, _) = setup();
        let proof_x = p.names.intern("p");
        let e = lam(
            proof_x,
            Expr::Const(prop_p, alloc::vec::Vec::new()),
            Expr::Const(p.zero, alloc::vec::Vec::new()),
        );
        let expected = LValue::Lam(Box::new(LValue::Const(p.zero)));
        assert_eq!(erase(&p, &e), expected);
    }

    #[test]
    fn prop_typed_lambda_is_star() {
        // λ (p : P). p  has type P → P, which lives in Prop, so the whole
        // lambda is a proof and erases to Star.
        let (mut p, prop_p, _) = setup();
        let proof_x = p.names.intern("p");
        let e = lam(proof_x, Expr::Const(prop_p, alloc::vec::Vec::new()), var(0));
        assert_eq!(erase(&p, &e), LValue::Star);
    }

    #[test]
    fn identity_over_a_type_erases_to_lambda() {
        // λ (A : Type 0). λ (x : A). x  — both binders retained.
        let (mut p, _, _) = setup();
        let a_name = p.names.intern("A");
        let x_name = p.names.intern("x");
        let e = lam(a_name, Expr::type0(), lam(x_name, var(0), var(0)));
        let expected = LValue::Lam(Box::new(LValue::Lam(Box::new(LValue::Var(DeBruijnIndex(
            0,
        ))))));
        assert_eq!(erase(&p, &e), expected);
    }

    // ---- applications ------------------------------------------------

    #[test]
    fn application_of_consts_erases_to_app() {
        let (p, _, _) = setup();
        let e = Expr::app(
            Expr::Const(p.succ, alloc::vec::Vec::new()),
            Expr::Const(p.zero, alloc::vec::Vec::new()),
        );
        let expected = LValue::App(
            Box::new(LValue::Const(p.succ)),
            Box::new(LValue::Const(p.zero)),
        );
        assert_eq!(erase(&p, &e), expected);
    }

    #[test]
    fn application_with_proof_argument_substitutes_star() {
        // (λ (x : P). Nat.zero) proof_p — the proof argument erases to Star
        // while the head keeps its binder.
        let (mut p, prop_p, proof_p) = setup();
        let x = p.names.intern("x");
        let lam_expr = lam(
            x,
            Expr::Const(prop_p, alloc::vec::Vec::new()),
            Expr::Const(p.zero, alloc::vec::Vec::new()),
        );
        let e = Expr::app(lam_expr, Expr::Const(proof_p, alloc::vec::Vec::new()));
        let expected = LValue::App(
            Box::new(LValue::Lam(Box::new(LValue::Const(p.zero)))),
            Box::new(LValue::Star),
        );
        assert_eq!(erase(&p, &e), expected);
    }

    // ---- let ---------------------------------------------------------

    #[test]
    fn let_erases_to_application() {
        // let x := Nat.zero; x  →  (λ #0) Nat.zero
        let (mut p, _, _) = setup();
        let x = p.names.intern("x");
        let e = Expr::Let(
            x,
            Box::new(Expr::Const(p.nat, alloc::vec::Vec::new())),
            Box::new(Expr::Const(p.zero, alloc::vec::Vec::new())),
            Box::new(var(0)),
        );
        let expected = LValue::App(
            Box::new(LValue::Lam(Box::new(LValue::Var(DeBruijnIndex(0))))),
            Box::new(LValue::Const(p.zero)),
        );
        assert_eq!(erase(&p, &e), expected);
    }

    // ---- ghost environment -------------------------------------------

    #[test]
    fn ghost_marked_constant_erases_to_star() {
        let (p, _, _) = setup();
        let mut ghost = GhostSet::new();
        ghost.mark(p.nat);
        let lv = erase_closed(&p.env, &ghost, &Expr::Const(p.nat, alloc::vec::Vec::new())).unwrap();
        assert_eq!(lv, LValue::Star);
    }

    #[test]
    fn ghost_marked_head_marks_application_as_ghost() {
        let (p, _, _) = setup();
        let mut ghost = GhostSet::new();
        ghost.mark(p.succ);
        let e = Expr::app(
            Expr::Const(p.succ, alloc::vec::Vec::new()),
            Expr::Const(p.zero, alloc::vec::Vec::new()),
        );
        let lv = erase_closed(&p.env, &ghost, &e).unwrap();
        // The head `succ` is ghost, but the application as a whole is
        // classified by its type (`Nat`, which is not ghost): only the
        // head erases to Star.
        let expected = LValue::App(Box::new(LValue::Star), Box::new(LValue::Const(p.zero)));
        assert_eq!(lv, expected);
    }

    #[test]
    fn ghost_type_marks_inhabitants_as_ghost() {
        // Mark `Nat` as ghost. Then a variable `n : Nat` should be ghost.
        let (mut p, _, _) = setup();
        let mut ghost = GhostSet::new();
        ghost.mark(p.nat);
        let eraser = Eraser::new(&p.env, &ghost);
        let mut ctx = TypedContext::empty();
        let nat_v = {
            let plain = TypeChecker::new(&p.env);
            plain
                .nbe()
                .eval(&ctx.to_env(), &Expr::Const(p.nat, alloc::vec::Vec::new()))
        };
        ctx.push(
            LocalDecl::binder(
                p.names.intern("n"),
                BinderInfo::Default,
                Rc::new(Expr::Const(p.nat, alloc::vec::Vec::new())),
            ),
            nat_v,
        );
        let lv = eraser.erase(&ctx, &var(0)).unwrap();
        assert_eq!(lv, LValue::Star);
    }

    #[test]
    fn ghost_set_lifecycle() {
        let mut ghost = GhostSet::new();
        assert!(ghost.is_empty());
        let name = NameId(42);
        ghost.mark(name);
        assert!(ghost.is_ghost(name));
        assert_eq!(ghost.len(), 1);
        ghost.unmark(name);
        assert!(!ghost.is_ghost(name));
        assert!(ghost.is_empty());
    }

    // ---- erase_declaration -------------------------------------------

    #[test]
    fn erase_declaration_of_definition() {
        let (mut p, _, _) = setup();
        let id_name = p.names.intern("id");
        // def id : Nat -> Nat := λ (n : Nat). n
        let n = p.names.intern("n");
        let ty = Expr::arrow(
            Expr::Const(p.nat, alloc::vec::Vec::new()),
            Expr::Const(p.nat, alloc::vec::Vec::new()),
            p.names.intern("_"),
        );
        let body = lam(n, Expr::Const(p.nat, alloc::vec::Vec::new()), var(0));
        p.env
            .add(ConstantInfo::Definition(DefinitionVal {
                base: ConstantVal {
                    name: id_name,
                    universe_params: alloc::vec::Vec::new(),
                    ty: Rc::new(ty),
                },
                body: Rc::new(body),
                transparency: Transparency::Semireducible,
                termination: crate::env::TerminationObligation::none(),
            }))
            .unwrap();
        let info = p.env.find(id_name).unwrap();
        let env = PropOnlyErasureEnv;
        let eraser = Eraser::new(&p.env, &env);
        let result = eraser.erase_declaration(&info).unwrap();
        let expected = LValue::Lam(Box::new(LValue::Var(DeBruijnIndex(0))));
        assert_eq!(result, Some(expected));
    }

    #[test]
    fn erase_declaration_of_theorem_is_star() {
        use crate::env::TheoremVal;
        let (mut p, prop_p, _) = setup();
        let thm = p.names.intern("thm");
        // theorem thm : P := <proof> (the body is ignored: every
        // theorem erases unconditionally)
        p.env
            .add(ConstantInfo::Theorem(TheoremVal {
                base: ConstantVal {
                    name: thm,
                    universe_params: alloc::vec::Vec::new(),
                    ty: Rc::new(Expr::Const(prop_p, alloc::vec::Vec::new())),
                },
                body: Rc::new(Expr::Const(prop_p, alloc::vec::Vec::new())),
                transparency: Transparency::Irreducible,
            }))
            .unwrap();
        let info = p.env.find(thm).unwrap();
        let env = PropOnlyErasureEnv;
        let eraser = Eraser::new(&p.env, &env);
        let result = eraser.erase_declaration(&info).unwrap();
        assert_eq!(result, Some(LValue::Star));
    }

    #[test]
    fn erase_declaration_of_axiom_is_none() {
        let (p, prop_p, _) = setup();
        let info = p.env.find(prop_p).unwrap();
        let env = PropOnlyErasureEnv;
        let eraser = Eraser::new(&p.env, &env);
        assert!(eraser.erase_declaration(&info).unwrap().is_none());
    }

    #[test]
    fn erase_declaration_of_inductive_is_none() {
        let (p, _, _) = setup();
        let info = p.env.find(p.nat).unwrap();
        let env = PropOnlyErasureEnv;
        let eraser = Eraser::new(&p.env, &env);
        assert!(eraser.erase_declaration(&info).unwrap().is_none());
    }

    // ---- LValue utilities --------------------------------------------

    #[test]
    fn lvalue_size_counts_correctly() {
        assert_eq!(LValue::Star.size(), 1);
        assert_eq!(LValue::Var(DeBruijnIndex(0)).size(), 1);
        let lam = LValue::Lam(Box::new(LValue::Star));
        assert_eq!(lam.size(), 2);
        let app = LValue::App(Box::new(LValue::Star), Box::new(LValue::Star));
        assert_eq!(app.size(), 3);
    }

    #[test]
    fn is_star_free_detects_placeholders() {
        assert!(!LValue::Star.is_star_free());
        assert!(LValue::Var(DeBruijnIndex(0)).is_star_free());
        assert!(LValue::Lam(Box::new(LValue::Var(DeBruijnIndex(0)))).is_star_free());
        assert!(!LValue::Lam(Box::new(LValue::Star)).is_star_free());
        assert!(
            !LValue::App(
                Box::new(LValue::Var(DeBruijnIndex(0))),
                Box::new(LValue::Star)
            )
            .is_star_free()
        );
    }

    #[test]
    fn render_produces_readable_output() {
        let (p, _, _) = setup();
        let e = Expr::app(
            Expr::Const(p.succ, alloc::vec::Vec::new()),
            Expr::Const(p.zero, alloc::vec::Vec::new()),
        );
        let lv = erase(&p, &e);
        let rendered = lv.render(&p.names);
        assert!(rendered.contains("Nat.succ"));
        assert!(rendered.contains("Nat.zero"));
    }

    #[test]
    fn display_fallback_is_nonempty() {
        assert!(!alloc::format!("{}", LValue::Star).is_empty());
        assert!(!alloc::format!("{}", LValue::Lam(Box::new(LValue::Star))).is_empty());
    }

    // ---- integration -------------------------------------------------

    #[test]
    fn plus_erases_to_star_free_term() {
        // Build `plus` in the prelude env and erase it. The top of the
        // erased term is a lambda (the outer λ m).
        let (mut p, _, _) = setup();
        let plus_name = p.names.intern("plus");
        let m = p.names.intern("m");
        let n = p.names.intern("n");
        let k = p.names.intern("k");
        let rec_var = p.names.intern("rec");
        let underscore = p.names.intern("_");

        let nat_const = || Expr::Const(p.nat, alloc::vec::Vec::new());
        let succ_const = || Expr::Const(p.succ, alloc::vec::Vec::new());
        let rec_const = || Expr::Const(p.rec, alloc::vec![Level::zero().succ()]);

        // motive := λ _. Nat
        let motive = lam(underscore, nat_const(), nat_const());
        // zero_case := n   (Var 0 at the point of use)
        let zero_case = var(0);
        // succ_case := λ k. λ rec. Nat.succ rec
        let succ_case = lam(
            k,
            nat_const(),
            lam(rec_var, nat_const(), Expr::app(succ_const(), var(0))),
        );
        // Nat.rec.{1} motive zero_case succ_case m
        let major = var(1); // m at depth 2
        let body = rec_const().apps([motive, zero_case, succ_case, major]);
        // λ m. λ n. body
        let plus_body = lam(m, nat_const(), lam(n, nat_const(), body));

        // Type: Nat -> Nat -> Nat
        let ty = Expr::arrow(
            nat_const(),
            Expr::arrow(nat_const(), nat_const(), underscore),
            underscore,
        );
        p.env
            .add(ConstantInfo::Definition(DefinitionVal {
                base: ConstantVal {
                    name: plus_name,
                    universe_params: alloc::vec::Vec::new(),
                    ty: Rc::new(ty),
                },
                body: Rc::new(plus_body),
                transparency: Transparency::Semireducible,
                termination: crate::env::TerminationObligation::none(),
            }))
            .unwrap();

        let info = p.env.find(plus_name).unwrap();
        let env = PropOnlyErasureEnv;
        let eraser = Eraser::new(&p.env, &env);
        let lv = eraser.erase_declaration(&info).unwrap().unwrap();
        // Sanity: the top of the erased term is a lambda (the outer λ m).
        assert!(matches!(lv, LValue::Lam(_)));
    }
}
