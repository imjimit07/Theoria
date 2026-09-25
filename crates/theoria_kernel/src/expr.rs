//! Core expression syntax tree.
//!
//! The AST here is the *kernel-level* syntax: post-elaboration, fully
//! annotated, and using de Bruijn indices for bound variables. Surface syntax
//! (names, implicit arguments, notation) is desugared by the elaborator before
//! reaching the kernel.
//!
//! ## De Bruijn indices vs. levels
//!
//! The syntactic AST uses **de Bruijn indices** because terms are constructed
//! top-down. The *semantic* domain used by NbE uses **de Bruijn levels**
//! because values are read back bottom-up. The two representations meet at the
//! evaluation boundary in the `nbe` module.

use crate::level::Level;
use crate::name::NameId;
use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

/// A de Bruijn index, counting binders from the innermost outward.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DeBruijnIndex(pub u32);

impl DeBruijnIndex {
    /// The innermost bound variable.
    pub const ZERO: Self = Self(0);

    /// Push the index one binder outward.
    #[must_use]
    pub const fn succ(self) -> Self {
        Self(self.0 + 1)
    }
}

impl fmt::Display for DeBruijnIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}", self.0)
    }
}

/// Binder annotations, mirroring Lean 4's convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinderInfo {
    /// `(x : A)` — an explicit argument.
    Default,
    /// `{x : A}` — an implicit argument, inferred by unification.
    Implicit,
    /// `{{x : A}}` — a strict implicit argument (always elaborated).
    StrictImplicit,
    /// `[x : A]` — a typeclass / instance argument.
    Instance,
}

impl BinderInfo {
    /// `true` for implicits and instances.
    #[must_use]
    pub const fn is_implicit(self) -> bool {
        !matches!(self, BinderInfo::Default)
    }
}

/// Kernel literals.
///
/// Numeric and string literals are *not* opaque: they desugar into the
/// corresponding inductive types (`Nat`, `String`) during elaboration. The AST
/// retains them only as a convenience for the elaborator and for pretty
/// printing.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Literal {
    /// A natural number literal.
    Nat(u64),
    /// A string literal.
    Str(String),
}

/// A kernel-level expression.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Expr {
    /// `Sort(l)` — a universe.
    Sort(Level),

    /// `#i` — a bound variable, by de Bruijn index.
    Var(DeBruijnIndex),

    /// `c.{u1, ..., un}` — a constant applied to universe parameters.
    Const(NameId, Vec<Level>),

    /// `f a` — application.
    App(Box<Expr>, Box<Expr>),

    /// `λ (x : A). b` — lambda abstraction.
    Lam(BinderInfo, NameId, Box<Expr>, Box<Expr>),

    /// `Π (x : A). B` — dependent function type.
    Pi(BinderInfo, NameId, Box<Expr>, Box<Expr>),

    /// `let x : A := v; b` — local definition.
    Let(NameId, Box<Expr>, Box<Expr>, Box<Expr>),

    /// A literal (kept for the elaborator's convenience).
    Lit(Literal),
}

impl Expr {
    /// `Sort(Level::Zero)` — the impredicative sort `Prop`.
    #[must_use]
    pub const fn prop() -> Self {
        Expr::Sort(Level::Zero)
    }

    /// `Sort(Level::Zero.succ())` — the first predicative sort, `Type 0`.
    #[must_use]
    pub fn type0() -> Self {
        Expr::Sort(Level::Zero.succ())
    }

    /// `f a`.
    #[must_use]
    pub fn app(f: Expr, a: Expr) -> Self {
        Expr::App(Box::new(f), Box::new(a))
    }

    /// `Π (_ : A). B` with `Default` binder info.
    #[must_use]
    pub fn arrow(a: Expr, b: Expr, name: NameId) -> Self {
        Expr::Pi(BinderInfo::Default, name, Box::new(a), Box::new(b))
    }

    /// `λ (x : A). b` with `Default` binder info.
    #[must_use]
    pub fn lam(name: NameId, a: Expr, b: Expr) -> Self {
        Expr::Lam(BinderInfo::Default, name, Box::new(a), Box::new(b))
    }

    /// Apply `self` to a sequence of arguments, left-associated.
    #[must_use]
    pub fn apps(self, args: impl IntoIterator<Item = Expr>) -> Self {
        args.into_iter().fold(self, Expr::app)
    }

    /// Total syntactic size (number of AST nodes).
    ///
    /// Used by termination metrics and for kernel sanity checks.
    #[must_use]
    pub fn size(&self) -> usize {
        match self {
            Expr::Sort(_) | Expr::Var(_) | Expr::Lit(_) | Expr::Const(_, _) => 1,
            Expr::App(f, a) => 1 + f.size() + a.size(),
            Expr::Lam(_, _, a, b) | Expr::Pi(_, _, a, b) => 1 + a.size() + b.size(),
            Expr::Let(_, t, v, b) => 1 + t.size() + v.size() + b.size(),
        }
    }

    /// `true` iff the expression is a syntactic universe.
    #[must_use]
    pub const fn is_sort(&self) -> bool {
        matches!(self, Expr::Sort(_))
    }

    /// Number of `Pi` binders at the head of this expression.
    ///
    /// Used by the environment's termination gate to check that a
    /// definition's obligation declares exactly as many parameters as the
    /// definition's type has Π-binders.
    #[must_use]
    pub fn pi_arity(&self) -> usize {
        let mut n = 0;
        let mut cur = self;
        while let Expr::Pi(_, _, _, cod) = cur {
            n += 1;
            cur = cod;
        }
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::name::NameTable;

    fn nid(t: &mut NameTable, s: &str) -> NameId {
        t.intern(s)
    }

    #[test]
    fn size_of_atom_is_one() {
        assert_eq!(Expr::prop().size(), 1);
        assert_eq!(Expr::Var(DeBruijnIndex::ZERO).size(), 1);
    }

    #[test]
    fn apps_left_associates() {
        let mut t = NameTable::new();
        let f = Expr::Const(nid(&mut t, "f"), Vec::new());
        let a = Expr::Const(nid(&mut t, "a"), Vec::new());
        let b = Expr::Const(nid(&mut t, "b"), Vec::new());
        let e = f.apps([a, b]);
        // Expected: App(App(f, a), b)
        match e {
            Expr::App(outer_f, _) => match *outer_f {
                Expr::App(_, _) => {}
                _ => panic!("inner app missing"),
            },
            _ => panic!("outer app missing"),
        }
    }

    #[test]
    fn arrow_constructs_pi() {
        let mut t = NameTable::new();
        let n = nid(&mut t, "_");
        let a = Expr::prop();
        let b = Expr::type0();
        let e = Expr::arrow(a, b, n);
        assert!(matches!(e, Expr::Pi(BinderInfo::Default, _, _, _)));
    }

    #[test]
    fn binder_info_predicate() {
        assert!(!BinderInfo::Default.is_implicit());
        assert!(BinderInfo::Implicit.is_implicit());
        assert!(BinderInfo::StrictImplicit.is_implicit());
        assert!(BinderInfo::Instance.is_implicit());
    }

    #[test]
    fn debruijn_succ() {
        assert_eq!(DeBruijnIndex::ZERO.succ(), DeBruijnIndex(1));
    }

    #[test]
    fn pi_arity_counts_leading_binders() {
        let mut t = NameTable::new();
        let n = nid(&mut t, "_");
        let nat = Expr::Const(nid(&mut t, "Nat"), Vec::new());
        assert_eq!(nat.pi_arity(), 0);
        assert_eq!(Expr::arrow(nat.clone(), nat.clone(), n).pi_arity(), 1);
        assert_eq!(
            Expr::arrow(nat.clone(), Expr::arrow(nat.clone(), nat.clone(), n), n).pi_arity(),
            2
        );
        // A Pi nested under something else does not count.
        let lam = Expr::lam(n, nat.clone(), Expr::arrow(nat.clone(), nat.clone(), n));
        assert_eq!(lam.pi_arity(), 0);
    }
}
