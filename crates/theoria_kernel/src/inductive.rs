//! Strict and nested positivity checker for inductive declarations.
//!
//! This module decides whether an inductive family can be soundly added to
//! the kernel environment. The check is the one that keeps the Calculus of
//! Inductive Constructions free of the classic paradoxes (Russell, Curry,
//! Burali-Forti, Hurkens) that arise from non-well-founded recursive types.
//!
//! ## The rule
//!
//! For each inductive `T` in a (possibly mutually recursive) block, and for
//! each constructor `c : Π (x₁ : A₁) ... (xₙ : Aₙ). T p⃗`, every occurrence of
//! any family member in every `Aᵢ` must be *strictly positive*.
//!
//! An occurrence is strictly positive when it appears:
//!
//! * as the head of an application, or
//! * inside a *carrier* argument of an inductive application whose head
//!   resolves to a strictly positive inductive, or
//! * in the codomain of a `Π`-type whose domain does not mention any family
//!   member at all.
//!
//! In particular, a family member may **never** occur in the domain of a
//! `Π`-type appearing inside a constructor field. This is the rule that
//! rejects the classical unsound inductives:
//!
//! ```text
//! Inductive Bad := mk : (Bad -> X) -> Bad        -- rejected
//! Inductive Bad := mk : ((Bad -> X) -> Bad) -> Bad -- rejected
//! Inductive Bad := mk : (Bad -> Bad) -> Bad      -- rejected
//! ```
//!
//! ## Nested inductives
//!
//! When a constructor field mentions another inductive `G`, the checker
//! recurses into `G`'s arguments treating them as carrier positions. This
//! is sound because every inductive in the environment was itself validated
//! on insertion, so its parameters are known to be strictly positive
//! functors. Consequently
//!
//! ```text
//! Inductive Tree := node : List Tree -> Tree       -- accepted
//! Inductive Rose := rose : List (List Rose) -> Rose -- accepted
//! ```
//!
//! are admitted, while
//!
//! ```text
//! Inductive Bad := mk : List (Bad -> X) -> Bad     -- rejected
//! ```
//!
//! is not.
//!
//! ### Carrier parameters
//!
//! For a parameterised inductive, a parameter whose binder domain is a
//! `Sort` is a *carrier*: instantiating it with a type substitutes that
//! type into every field position where the parameter occurs. To keep this
//! substitution sound, carrier parameters must occur only at strictly
//! positive positions in every constructor field type. Concretely, this
//! rejects
//!
//! ```text
//! Inductive Foo (A : Type) := mk : (A -> A)  -> Foo A   -- rejected
//! Inductive Bar (A : Type) := mk : (A -> Nat) -> Bar A  -- rejected
//! ```
//!
//! while admitting
//!
//! ```text
//! Inductive List (A : Type) := cons : A -> List A -> List A  -- accepted
//! Inductive Baz  (A : Type) := mk   : (Nat -> A) -> Baz A    -- accepted
//! ```
//!
//! Parameters whose domain is not a `Sort` (`n : Nat`, `x : A`, ...) are
//! *indices*. They are terms, not types, and cannot be substituted with a
//! type expression; the checker therefore leaves them alone.
//!
//! ## Known limitations
//!
//! * **`is_unsafe` inductives are treated as opaque.** A nested occurrence of
//!   an unsafe inductive is rejected conservatively (`!occurs`), rather than
//!   trusted via its (unchecked) declaration.
//! * **No caching.** Every call re-walks the constructor types. Fine for the
//!   current scale; a memoized `strict_pos` is a straightforward extension.

use crate::env::{ConstantInfo, ConstructorVal, GlobalEnv, InductiveVal};
use crate::error::{KernelError, KernelResult};
use crate::expr::Expr;
use crate::name::NameId;
use alloc::boxed::Box;
use alloc::vec::Vec;

/// A sentinel [`NameId`] reserved by the positivity checker to represent
/// carrier parameters after substitution. Never interned by
/// [`NameTable`](crate::name::NameTable) — the table allocates sequentially
/// from `0`, so a collision would require interning `u32::MAX` names first.
///
/// The marker is used transiently inside [`PositivityChecker`] and is never
/// stored in a [`GlobalEnv`](crate::env::GlobalEnv).
const CARRIER_MARKER: NameId = NameId(u32::MAX);

// ---------------------------------------------------------------------------
// PositivityChecker
// ---------------------------------------------------------------------------

/// The positivity checker.
///
/// Borrows the [`GlobalEnv`] so it can resolve constants encountered inside
/// constructor field types. The env is read-only during a check.
pub struct PositivityChecker<'a> {
    env: &'a GlobalEnv,
}

impl<'a> PositivityChecker<'a> {
    /// Construct a checker over the given environment.
    #[must_use]
    pub fn new(env: &'a GlobalEnv) -> Self {
        Self { env }
    }

    /// The environment the checker reads from.
    #[must_use]
    pub fn env(&self) -> &'a GlobalEnv {
        self.env
    }

    // -----------------------------------------------------------------
    // Public entry point
    // -----------------------------------------------------------------

    /// Verify that every constructor of `ind` satisfies the strict
    /// positivity condition with respect to the whole family listed in
    /// `ind.all`.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::NonPositiveInductive`] naming the offending
    /// constructor if any field type fails the check.
    pub fn check_inductive(
        &self,
        ind: &InductiveVal,
        ctors: &[ConstructorVal],
    ) -> KernelResult<()> {
        let family: &[NameId] = &ind.all;
        let num_params = ind.num_params as usize;
        for (i, ctor) in ctors.iter().enumerate() {
            if !self.check_constructor_type(family, num_params, &ctor.base.ty) {
                return Err(KernelError::NonPositiveInductive {
                    inductive: ind.base.name,
                    constructor: i,
                });
            }
        }
        Ok(())
    }

    /// Convenience entry point for a whole mutually-recursive block.
    ///
    /// All inductives in `inds` share the same family (the union of their
    /// `all` lists) and all constructors are collected into `ctors`. Each
    /// inductive is checked in turn; the first failure short-circuits.
    ///
    /// # Errors
    ///
    /// As [`PositivityChecker::check_inductive`].
    pub fn check_block(&self, inds: &[InductiveVal], ctors: &[ConstructorVal]) -> KernelResult<()> {
        for ind in inds {
            self.check_inductive(ind, ctors)?;
        }
        Ok(())
    }

    // -----------------------------------------------------------------
    // Constructor traversal
    // -----------------------------------------------------------------

    /// Check every field type of a constructor for strict positivity.
    ///
    /// Three phases:
    ///
    /// 1. **Classify parameters.** Descend through the `num_params` leading
    ///    Π-binders and record, for each, whether its domain is a `Sort`
    ///    (a *carrier* parameter) or something else (an *index*).
    /// 2. **Build the extended family.** The carrier marker is appended to
    ///    the family so that the existing [`Self::strict_pos`] predicate can
    ///    decide carrier occurrences uniformly with family occurrences.
    /// 3. **Check fields.** Descend through the remaining Π-binders. For
    ///    each field type, substitute carrier parameters with the marker
    ///    (respecting the field-binder depth) and run `strict_pos`.
    ///
    /// The carrier classification is syntactic: only a domain that is
    /// literally a `Sort` counts as a carrier. A domain that merely reduces
    /// to a `Sort` is treated as an index (a conservative over-acceptance
    /// that relies on the elaborator supplying β-normal constructor types).
    ///
    /// The final codomain — `T p⃗` — is not inspected: its head is a family
    /// member and its arguments are the parameters, which cannot themselves
    /// introduce negative occurrences.
    fn check_constructor_type(&self, family: &[NameId], num_params: usize, ctor_ty: &Expr) -> bool {
        // --- Phase 1: classify parameters ------------------------------
        let mut is_carrier: Vec<bool> = Vec::with_capacity(num_params);
        let mut current: &Expr = ctor_ty;
        for _ in 0..num_params {
            match current {
                Expr::Pi(_, _, dom, cod) => {
                    // A carrier is a parameter whose domain is itself a
                    // sort. Anything else (`Nat`, `A`, `P -> Q`, ...) is an
                    // index and cannot be substituted with a type.
                    is_carrier.push(matches!(**dom, Expr::Sort(_)));
                    current = cod;
                }
                _ => {
                    // Fewer Π-binders than `num_params`: malformed input.
                    // Reject conservatively.
                    return false;
                }
            }
        }

        // --- Phase 2: extended family ----------------------------------
        let mut extended_family: Vec<NameId> = family.to_vec();
        extended_family.push(CARRIER_MARKER);

        // --- Phase 3: check fields -------------------------------------
        let mut depth: usize = 0;
        loop {
            match current {
                Expr::Pi(_, _, dom, cod) => {
                    let dom_subst =
                        substitute_carriers(dom, depth, num_params, &is_carrier, CARRIER_MARKER);
                    if !self.strict_pos(&extended_family, &dom_subst) {
                        return false;
                    }
                    depth += 1;
                    current = cod;
                }
                _ => return true,
            }
        }
    }

    // -----------------------------------------------------------------
    // Strict positivity
    // -----------------------------------------------------------------

    /// `true` iff every occurrence of a family member in `e` is at a
    /// strictly positive position.
    ///
    /// Rules:
    ///
    /// * Atoms (`Sort`, `Var`, `Lit`, `Const`) — trivially strict. A bare
    ///   `Const(F)` is a positive occurrence of `F`, not a negative one.
    /// * `Π (x : A). B` — strict iff `A` mentions no family member at all
    ///   (any occurrence in a domain is rejected, even under another
    ///   inductive) and `B` is strict.
    /// * `λ (x : A). b` — same as `Π` (defensive: lambdas do not normally
    ///   occur in field types).
    /// * `let x := v; b` — same as `Π` (defensive).
    /// * `head arg₁ ... argₙ` — strict iff `head` is strict and every `argᵢ`
    ///   is strict. The head of a well-formed constructor field is either
    ///   a family member, an inductive constant, or a type-returning
    ///   constant; in all three cases the arguments are carrier positions.
    #[must_use]
    pub fn strict_pos(&self, family: &[NameId], e: &Expr) -> bool {
        match e {
            Expr::Sort(_) | Expr::Var(_) | Expr::Lit(_) | Expr::Const(_, _) => true,

            Expr::Pi(_, _, dom, cod) => !self.occurs(family, dom) && self.strict_pos(family, cod),
            Expr::Lam(_, _, dom, body) => {
                !self.occurs(family, dom) && self.strict_pos(family, body)
            }
            Expr::Let(_, _, val, body) => {
                !self.occurs(family, val) && self.strict_pos(family, body)
            }

            Expr::App(..) => {
                let (head, args) = flatten_app(e);

                // A nested occurrence of an *unsafe* inductive cannot be
                // trusted: its declaration was not checked for positivity,
                // so we fall back to the conservative rule that the family
                // must not occur at all.
                if let Expr::Const(name, _) = head {
                    if self.is_unsafe_inductive(*name) {
                        return !self.occurs(family, e);
                    }
                }

                self.strict_pos(family, head) && args.iter().all(|a| self.strict_pos(family, a))
            }
        }
    }

    /// `true` iff `e` contains at least one occurrence of a family member,
    /// regardless of position.
    // `self` is unused today (`occurs` is purely syntactic), but the
    // receiver is kept for API symmetry with `strict_pos`, which is
    // env-aware, and so future extensions can resolve constants here.
    #[allow(clippy::only_used_in_recursion)]
    #[must_use]
    pub fn occurs(&self, family: &[NameId], e: &Expr) -> bool {
        match e {
            Expr::Sort(_) | Expr::Var(_) | Expr::Lit(_) => false,
            Expr::Const(name, _) => family.contains(name),
            Expr::App(f, a) => self.occurs(family, f) || self.occurs(family, a),
            Expr::Pi(_, _, dom, cod) => self.occurs(family, dom) || self.occurs(family, cod),
            Expr::Lam(_, _, dom, body) => self.occurs(family, dom) || self.occurs(family, body),
            Expr::Let(_, _, val, body) => self.occurs(family, val) || self.occurs(family, body),
        }
    }

    // -----------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------

    /// `true` iff the environment's declaration for `name` is an inductive
    /// marked `unsafe`.
    fn is_unsafe_inductive(&self, name: NameId) -> bool {
        matches!(
            self.env.find(name).as_deref(),
            Some(ConstantInfo::Inductive(v)) if v.is_unsafe
        )
    }
}

// ---------------------------------------------------------------------------
// Free helpers
// ---------------------------------------------------------------------------

/// Flatten an application spine.
///
/// `App(App(App(f, a), b), c)` yields `(f, [a, b, c])`.
fn flatten_app(e: &Expr) -> (&Expr, Vec<&Expr>) {
    let mut args = Vec::new();
    let mut cur = e;
    while let Expr::App(f, a) = cur {
        args.push(a.as_ref());
        cur = f.as_ref();
    }
    args.reverse();
    (cur, args)
}

/// Replace `Var(i)` with `Const(marker)` whenever the variable at index `i`
/// refers to a carrier parameter, given that we have descended `depth`
/// binders past the parameter block.
///
/// ## Index arithmetic
///
/// After descending past all `k` parameter binders, the parameter at
/// 1-indexed position `pos` (where `pos = 1` is the outermost / first
/// parameter) sits at `Var(k - pos)`. Entering `depth` additional binders
/// shifts it to `Var(k - pos + depth)`. Inverting:
///
/// ```text
/// pos_zero_indexed = k + depth - 1 - i
/// ```
///
/// for a variable at index `i` that lies in the parameter block.
fn substitute_carriers(
    e: &Expr,
    depth: usize,
    num_params: usize,
    is_carrier: &[bool],
    marker: NameId,
) -> Expr {
    match e {
        Expr::Var(idx) => {
            let i = idx.0 as usize;
            if i >= depth && i < num_params + depth {
                let param_pos = num_params + depth - 1 - i;
                if is_carrier[param_pos] {
                    return Expr::Const(marker, Vec::new());
                }
            }
            e.clone()
        }
        Expr::App(f, a) => Expr::App(
            Box::new(substitute_carriers(
                f, depth, num_params, is_carrier, marker,
            )),
            Box::new(substitute_carriers(
                a, depth, num_params, is_carrier, marker,
            )),
        ),
        Expr::Lam(info, name, dom, body) => Expr::Lam(
            *info,
            *name,
            Box::new(substitute_carriers(
                dom, depth, num_params, is_carrier, marker,
            )),
            Box::new(substitute_carriers(
                body,
                depth + 1,
                num_params,
                is_carrier,
                marker,
            )),
        ),
        Expr::Pi(info, name, dom, cod) => Expr::Pi(
            *info,
            *name,
            Box::new(substitute_carriers(
                dom, depth, num_params, is_carrier, marker,
            )),
            Box::new(substitute_carriers(
                cod,
                depth + 1,
                num_params,
                is_carrier,
                marker,
            )),
        ),
        Expr::Let(name, ty, val, body) => Expr::Let(
            *name,
            Box::new(substitute_carriers(
                ty, depth, num_params, is_carrier, marker,
            )),
            Box::new(substitute_carriers(
                val, depth, num_params, is_carrier, marker,
            )),
            Box::new(substitute_carriers(
                body,
                depth + 1,
                num_params,
                is_carrier,
                marker,
            )),
        ),
        Expr::Sort(_) | Expr::Const(_, _) | Expr::Lit(_) => e.clone(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::{
        ConstantInfo, ConstantVal, ConstructorVal, EnvError, InductiveVal, RecursorRule,
        RecursorVal,
    };
    use crate::expr::{BinderInfo, DeBruijnIndex, Expr};
    use crate::name::{NameId, NameTable};
    use alloc::boxed::Box;
    use alloc::rc::Rc;
    use alloc::vec;

    // ---- construction helpers ----------------------------------------

    fn var(i: u32) -> Expr {
        Expr::Var(DeBruijnIndex(i))
    }

    fn cst(n: NameId) -> Expr {
        Expr::Const(n, vec![])
    }

    fn app(f: Expr, a: Expr) -> Expr {
        Expr::app(f, a)
    }

    fn apps(f: Expr, args: impl IntoIterator<Item = Expr>) -> Expr {
        f.apps(args)
    }

    fn pi(dom: Expr, cod: Expr) -> Expr {
        Expr::Pi(BinderInfo::Default, NameId(0), Box::new(dom), Box::new(cod))
    }

    fn ctor(name: NameId, ind: NameId, ty: Expr, index: u32) -> ConstructorVal {
        ConstructorVal {
            base: ConstantVal {
                name,
                universe_params: Vec::new(),
                ty: Rc::new(ty),
            },
            inductive: ind,
            index,
            num_params: 0,
            num_fields: 0,
            is_unsafe: false,
        }
    }

    fn ind(name: NameId, all: Vec<NameId>, ctors: Vec<NameId>, num_params: u32) -> InductiveVal {
        InductiveVal {
            base: ConstantVal {
                name,
                universe_params: Vec::new(),
                ty: Rc::new(Expr::type0()),
            },
            num_params,
            num_indices: 0,
            all,
            constructors: ctors,
            is_recursive: true,
            is_nested: false,
            is_unsafe: false,
        }
    }

    /// Register a minimal inductive `name` with the given constructors in
    /// the env so that nested occurrences resolve correctly.
    fn register_inductive(
        env: &mut GlobalEnv,
        name: NameId,
        all: Vec<NameId>,
        ctor_names: Vec<NameId>,
    ) -> Result<(), EnvError> {
        let ind = ind(name, all.clone(), ctor_names.clone(), 0);
        let ctors: Vec<ConstructorVal> = ctor_names
            .iter()
            .enumerate()
            .map(|(i, cn)| ctor(*cn, name, Expr::type0(), i as u32))
            .collect();
        let rec = RecursorVal {
            base: ConstantVal {
                name: NameId(u32::MAX),
                universe_params: Vec::new(),
                ty: Rc::new(Expr::type0()),
            },
            all: all.clone(),
            num_params: 0,
            num_indices: 0,
            num_motives: 1,
            num_minors: 0,
            rules: Vec::<RecursorRule>::new(),
            is_k: false,
            is_unsafe: false,
        };
        env.add_inductive(ind, ctors, rec)
    }

    // ---- direct tests on strict_pos ----------------------------------

    #[test]
    fn strict_pos_atom_is_always_strict() {
        let env = GlobalEnv::new();
        let pc = PositivityChecker::new(&env);
        assert!(pc.strict_pos(&[NameId(1)], &Expr::prop()));
        assert!(pc.strict_pos(&[NameId(1)], &var(0)));
        assert!(pc.strict_pos(&[NameId(1)], &cst(NameId(1))));
        assert!(pc.strict_pos(&[NameId(1)], &cst(NameId(2))));
    }

    #[test]
    fn strict_pos_bare_family_head_is_strict() {
        let env = GlobalEnv::new();
        let pc = PositivityChecker::new(&env);
        let f = NameId(1);
        // F (bare)
        assert!(pc.strict_pos(&[f], &cst(f)));
        // F x
        assert!(pc.strict_pos(&[f], &app(cst(f), var(0))));
        // F x y
        assert!(pc.strict_pos(&[f], &apps(cst(f), [var(0), var(1)])));
    }

    #[test]
    fn strict_pos_family_in_arrow_domain_is_rejected() {
        let env = GlobalEnv::new();
        let pc = PositivityChecker::new(&env);
        let f = NameId(1);
        // F -> Nat
        let e = pi(cst(f), Expr::type0());
        assert!(!pc.strict_pos(&[f], &e));
    }

    #[test]
    fn strict_pos_family_in_arrow_codomain_is_strict() {
        let env = GlobalEnv::new();
        let pc = PositivityChecker::new(&env);
        let f = NameId(1);
        // Nat -> F
        let e = pi(Expr::type0(), cst(f));
        assert!(pc.strict_pos(&[f], &e));
    }

    #[test]
    fn strict_pos_nested_family_at_negative_is_rejected() {
        let env = GlobalEnv::new();
        let pc = PositivityChecker::new(&env);
        let f = NameId(1);
        // (F -> Nat) -> Nat
        let e = pi(pi(cst(f), Expr::type0()), Expr::type0());
        assert!(!pc.strict_pos(&[f], &e));
    }

    #[test]
    fn strict_pos_family_in_app_arg_is_strict() {
        let env = GlobalEnv::new();
        let pc = PositivityChecker::new(&env);
        let f = NameId(1);
        let g = NameId(2);
        // G F
        let e = app(cst(g), cst(f));
        assert!(pc.strict_pos(&[f], &e));
        // G (G F)
        let e = app(cst(g), app(cst(g), cst(f)));
        assert!(pc.strict_pos(&[f], &e));
    }

    #[test]
    fn strict_pos_family_in_pi_inside_app_arg_is_rejected() {
        let env = GlobalEnv::new();
        let pc = PositivityChecker::new(&env);
        let f = NameId(1);
        let g = NameId(2);
        // G (F -> Nat)
        let e = app(cst(g), pi(cst(f), Expr::type0()));
        assert!(!pc.strict_pos(&[f], &e));
    }

    #[test]
    fn occurs_detects_every_position() {
        let env = GlobalEnv::new();
        let pc = PositivityChecker::new(&env);
        let f = NameId(1);
        let g = NameId(2);
        assert!(!pc.occurs(&[f], &Expr::prop()));
        assert!(!pc.occurs(&[f], &cst(g)));
        assert!(pc.occurs(&[f], &cst(f)));
        assert!(pc.occurs(&[f], &pi(cst(f), cst(g))));
        assert!(pc.occurs(&[f], &pi(cst(g), cst(f))));
        assert!(pc.occurs(&[f], &app(cst(g), cst(f))));
    }

    // ---- full check_inductive ----------------------------------------

    #[test]
    fn nat_is_accepted() {
        // Nat : Type := zero : Nat | succ : Nat -> Nat
        let mut names = NameTable::new();
        let nat = names.intern("Nat");
        let zero = names.intern("Nat.zero");
        let succ = names.intern("Nat.succ");

        let mut env = GlobalEnv::new();
        register_inductive(&mut env, nat, vec![nat], vec![zero, succ]).unwrap();

        let ind = ind(nat, vec![nat], vec![zero, succ], 0);
        let ctors = vec![
            ctor(zero, nat, cst(nat), 0),
            ctor(succ, nat, pi(cst(nat), cst(nat)), 1),
        ];

        let pc = PositivityChecker::new(&env);
        pc.check_inductive(&ind, &ctors).unwrap();
    }

    #[test]
    fn bad_in_arrow_domain_is_rejected() {
        // Bad : Type := mk : (Bad -> X) -> Bad
        let mut names = NameTable::new();
        let bad = names.intern("Bad");
        let mk = names.intern("Bad.mk");
        let x = names.intern("X");

        let env = GlobalEnv::new();
        let pc = PositivityChecker::new(&env);

        let ind = ind(bad, vec![bad], vec![mk], 0);
        let bad_ty = pi(pi(cst(bad), cst(x)), cst(bad));
        let ctors = vec![ctor(mk, bad, bad_ty, 0)];

        let err = pc.check_inductive(&ind, &ctors).unwrap_err();
        match err {
            KernelError::NonPositiveInductive {
                inductive,
                constructor,
            } => {
                assert_eq!(inductive, bad);
                assert_eq!(constructor, 0);
            }
            other => panic!("expected NonPositiveInductive, got {other:?}"),
        }
    }

    #[test]
    fn bad_self_arrow_is_rejected() {
        // Bad : Type := mk : (Bad -> Bad) -> Bad
        let mut names = NameTable::new();
        let bad = names.intern("Bad");
        let mk = names.intern("Bad.mk");

        let env = GlobalEnv::new();
        let pc = PositivityChecker::new(&env);

        let ind = ind(bad, vec![bad], vec![mk], 0);
        let ty = pi(pi(cst(bad), cst(bad)), cst(bad));
        let ctors = vec![ctor(mk, bad, ty, 0)];

        assert!(pc.check_inductive(&ind, &ctors).is_err());
    }

    #[test]
    fn bad_nested_in_arrow_domain_is_rejected() {
        // Bad : Type := mk : ((Bad -> X) -> Bad) -> Bad
        let mut names = NameTable::new();
        let bad = names.intern("Bad");
        let mk = names.intern("Bad.mk");
        let x = names.intern("X");

        let env = GlobalEnv::new();
        let pc = PositivityChecker::new(&env);

        let ind = ind(bad, vec![bad], vec![mk], 0);
        // field: (Bad -> X) -> Bad
        let field = pi(pi(cst(bad), cst(x)), cst(bad));
        let ty = pi(field, cst(bad));
        let ctors = vec![ctor(mk, bad, ty, 0)];

        assert!(pc.check_inductive(&ind, &ctors).is_err());
    }

    #[test]
    fn nested_list_occurrence_is_accepted() {
        // List : Type := (abstract inductive)
        // Tree : Type := node : List Tree -> Tree
        let mut names = NameTable::new();
        let list = names.intern("List");
        let nil = names.intern("List.nil");
        let cons = names.intern("List.cons");
        let tree = names.intern("Tree");
        let node = names.intern("Tree.node");

        let mut env = GlobalEnv::new();
        register_inductive(&mut env, list, vec![list], vec![nil, cons]).unwrap();

        let pc = PositivityChecker::new(&env);

        let ind = ind(tree, vec![tree], vec![node], 0);
        // field: List Tree
        let field = app(cst(list), cst(tree));
        let ty = pi(field, cst(tree));
        let ctors = vec![ctor(node, tree, ty, 0)];

        pc.check_inductive(&ind, &ctors).unwrap();
    }

    #[test]
    fn nested_negative_inside_list_is_rejected() {
        // List : Type
        // Bad : Type := mk : List (Bad -> Nat) -> Bad
        let mut names = NameTable::new();
        let list = names.intern("List");
        let nil = names.intern("List.nil");
        let cons = names.intern("List.cons");
        let bad = names.intern("Bad");
        let mk = names.intern("Bad.mk");
        let nat = names.intern("Nat");

        let mut env = GlobalEnv::new();
        register_inductive(&mut env, list, vec![list], vec![nil, cons]).unwrap();

        let pc = PositivityChecker::new(&env);

        let ind = ind(bad, vec![bad], vec![mk], 0);
        // field: List (Bad -> Nat)
        let inner = pi(cst(bad), cst(nat));
        let field = app(cst(list), inner);
        let ty = pi(field, cst(bad));
        let ctors = vec![ctor(mk, bad, ty, 0)];

        assert!(pc.check_inductive(&ind, &ctors).is_err());
    }

    #[test]
    fn deep_nesting_is_accepted() {
        // List : Type
        // Rose : Type := rose : List (List Rose) -> Rose
        let mut names = NameTable::new();
        let list = names.intern("List");
        let nil = names.intern("List.nil");
        let cons = names.intern("List.cons");
        let rose = names.intern("Rose");
        let rmk = names.intern("Rose.rose");

        let mut env = GlobalEnv::new();
        register_inductive(&mut env, list, vec![list], vec![nil, cons]).unwrap();

        let pc = PositivityChecker::new(&env);

        let ind = ind(rose, vec![rose], vec![rmk], 0);
        // field: List (List Rose)
        let inner = app(cst(list), cst(rose));
        let field = app(cst(list), inner);
        let ty = pi(field, cst(rose));
        let ctors = vec![ctor(rmk, rose, ty, 0)];

        pc.check_inductive(&ind, &ctors).unwrap();
    }

    #[test]
    fn mutual_even_odd_is_accepted() {
        // Even : Type := zero : Even | succ_e : Odd -> Even
        // Odd  : Type := succ_o : Even -> Odd
        let mut names = NameTable::new();
        let even = names.intern("Even");
        let odd = names.intern("Odd");
        let zero = names.intern("Even.zero");
        let se = names.intern("Even.succ_e");
        let so = names.intern("Odd.succ_o");

        // Env insertion for the mutual block is currently restricted to a
        // single inductive, so the check runs against the family set passed
        // to `check_inductive` (an empty env suffices: no nested heads need
        // resolving here).
        let empty = GlobalEnv::new();
        let pc = PositivityChecker::new(&empty);
        let ind_e = ind(even, vec![even, odd], vec![zero, se], 0);
        let ctors = vec![
            ctor(zero, even, cst(even), 0),
            ctor(se, even, pi(cst(odd), cst(even)), 1),
        ];
        pc.check_inductive(&ind_e, &ctors).unwrap();

        let ind_o = ind(odd, vec![even, odd], vec![so], 0);
        let ctors_o = vec![ctor(so, odd, pi(cst(even), cst(odd)), 0)];
        pc.check_inductive(&ind_o, &ctors_o).unwrap();
    }

    #[test]
    fn unsafe_nested_inductive_is_rejected_conservatively() {
        // List : unsafe inductive
        // Bad : Type := mk : List Bad -> Bad
        let mut names = NameTable::new();
        let list = names.intern("List");
        let bad = names.intern("Bad");
        let mk = names.intern("Bad.mk");

        let mut env = GlobalEnv::new();
        let mut list_ind = ind(list, vec![list], vec![], 0);
        list_ind.is_unsafe = true;
        // Directly insert the unsafe inductive (bypassing add_inductive
        // since it requires a recursor).
        env.add(ConstantInfo::Inductive(list_ind)).unwrap();

        let pc = PositivityChecker::new(&env);
        let ind = ind(bad, vec![bad], vec![mk], 0);
        // field: List Bad
        let ty = pi(app(cst(list), cst(bad)), cst(bad));
        let ctors = vec![ctor(mk, bad, ty, 0)];

        assert!(pc.check_inductive(&ind, &ctors).is_err());
    }

    #[test]
    fn parameters_are_skipped_in_field_check() {
        // Vec : (A : Type) (n : Nat) -> Type
        //     := nil  : Vec A 0
        //      | cons : A -> Vec A n -> Vec A (n+1)
        let mut names = NameTable::new();
        let vec_n = names.intern("Vec");
        let nil = names.intern("Vec.nil");
        let cons = names.intern("Vec.cons");

        let env = GlobalEnv::new();
        let pc = PositivityChecker::new(&env);

        // Nil has no fields beyond the parameters: Pi (A : Type). Pi (n : _). Vec.
        // The parameter domains only matter for carrier classification here
        // (first is a carrier, second an index); the test's point is that
        // checking stops at the codomain and accepts.
        let ind_v = ind(vec_n, vec![vec_n], vec![nil, cons], 2);
        let nil_ty = pi(Expr::type0(), pi(cst(names_placeholder()), cst(vec_n)));
        let nil_c = ctor(nil, vec_n, nil_ty, 0);

        // Cons: the first two Π-binders are the parameters (skipped); the
        // remaining fields only mention Vars, which are always strict, so
        // the precise de Bruijn shape of the parameter binders is
        // immaterial to the check.
        let field_a = var(0);
        let field_vec = app(cst(vec_n), var(0));
        let ty_cons = pi(
            Expr::type0(),
            pi(
                cst(names_placeholder()),
                pi(field_a, pi(field_vec, cst(vec_n))),
            ),
        );
        let cons_c = ctor(cons, vec_n, ty_cons, 1);

        pc.check_inductive(&ind_v, &[nil_c, cons_c]).unwrap();
    }

    // ---- carrier-parameter positivity --------------------------------

    #[test]
    fn carrier_in_arrow_domain_is_rejected() {
        // Foo (A : Type) := mk : (A -> A) -> Foo A
        let mut names = NameTable::new();
        let foo = names.intern("Foo");
        let mk = names.intern("Foo.mk");
        let a = names.intern("A");

        let env = GlobalEnv::new();
        let pc = PositivityChecker::new(&env);

        let ind = ind(foo, vec![foo], vec![mk], 1);
        // ctor type: Pi (A : Type). Pi (_ : A -> A). Foo A
        let field = pi(var(0), var(0));
        let ctor_ty = Expr::Pi(
            BinderInfo::Default,
            a,
            Box::new(Expr::type0()),
            Box::new(pi(field, cst(foo))),
        );
        let ctors = vec![ctor(mk, foo, ctor_ty, 0)];

        assert!(pc.check_inductive(&ind, &ctors).is_err());
    }

    #[test]
    fn carrier_in_arrow_codomain_is_accepted() {
        // Baz (A : Type) := mk : (Nat -> A) -> Baz A
        let mut names = NameTable::new();
        let baz = names.intern("Baz");
        let mk = names.intern("Baz.mk");
        let a = names.intern("A");
        let nat = names.intern("Nat");

        let mut env = GlobalEnv::new();
        env.add(ConstantInfo::Axiom(ConstantVal {
            name: nat,
            universe_params: Vec::new(),
            ty: Rc::new(Expr::type0()),
        }))
        .unwrap();
        let pc = PositivityChecker::new(&env);

        let ind = ind(baz, vec![baz], vec![mk], 1);
        // field: Nat -> A
        let field = pi(cst(nat), var(0));
        let ctor_ty = Expr::Pi(
            BinderInfo::Default,
            a,
            Box::new(Expr::type0()),
            Box::new(pi(field, cst(baz))),
        );
        let ctors = vec![ctor(mk, baz, ctor_ty, 0)];

        pc.check_inductive(&ind, &ctors).unwrap();
    }

    #[test]
    fn bare_carrier_field_is_accepted() {
        // List (A : Type) := cons : A -> List A -> List A
        let mut names = NameTable::new();
        let list = names.intern("List");
        let cons = names.intern("List.cons");
        let a = names.intern("A");

        let env = GlobalEnv::new();
        let pc = PositivityChecker::new(&env);

        let ind = ind(list, vec![list], vec![cons], 1);
        // ctor type: Pi (A : Type). Pi (_ : A). Pi (_ : List A). List A
        let field1 = var(0); // A
        let field2 = app(cst(list), var(1)); // List A  (A shifted by 1 binder)
        let ctor_ty = Expr::Pi(
            BinderInfo::Default,
            a,
            Box::new(Expr::type0()),
            Box::new(pi(field1, pi(field2, cst(list)))),
        );
        let ctors = vec![ctor(cons, list, ctor_ty, 0)];

        pc.check_inductive(&ind, &ctors).unwrap();
    }

    #[test]
    fn index_parameter_is_not_treated_as_carrier() {
        // Vec (A : Type) (n : Nat) := nil : Vec A n
        // Here `n : Nat` is an index; it should not be required to occur
        // positively anywhere.
        let mut names = NameTable::new();
        let vec_n = names.intern("Vec");
        let nil = names.intern("Vec.nil");
        let a = names.intern("A");
        let n = names.intern("n");
        let nat = names.intern("Nat");

        let mut env = GlobalEnv::new();
        env.add(ConstantInfo::Axiom(ConstantVal {
            name: nat,
            universe_params: Vec::new(),
            ty: Rc::new(Expr::type0()),
        }))
        .unwrap();
        let pc = PositivityChecker::new(&env);

        let ind = ind(vec_n, vec![vec_n], vec![nil], 2);
        // ctor type: Pi (A : Type). Pi (n : Nat). Vec A n
        // No field binders besides params, so this reduces to a bare
        // codomain and should be accepted trivially.
        let ctor_ty = Expr::Pi(
            BinderInfo::Default,
            a,
            Box::new(Expr::type0()),
            Box::new(Expr::Pi(
                BinderInfo::Default,
                n,
                Box::new(cst(nat)),
                Box::new(cst(vec_n)),
            )),
        );
        let ctors = vec![ctor(nil, vec_n, ctor_ty, 0)];

        pc.check_inductive(&ind, &ctors).unwrap();
    }

    #[test]
    fn carrier_applied_negatively_is_rejected() {
        // Foo (A : Type) := mk : (A -> Nat) -> Foo A
        let mut names = NameTable::new();
        let foo = names.intern("Foo");
        let mk = names.intern("Foo.mk");
        let a = names.intern("A");
        let nat = names.intern("Nat");

        let mut env = GlobalEnv::new();
        env.add(ConstantInfo::Axiom(ConstantVal {
            name: nat,
            universe_params: Vec::new(),
            ty: Rc::new(Expr::type0()),
        }))
        .unwrap();
        let pc = PositivityChecker::new(&env);

        let ind = ind(foo, vec![foo], vec![mk], 1);
        let field = pi(var(0), cst(nat));
        let ctor_ty = Expr::Pi(
            BinderInfo::Default,
            a,
            Box::new(Expr::type0()),
            Box::new(pi(field, cst(foo))),
        );
        let ctors = vec![ctor(mk, foo, ctor_ty, 0)];

        assert!(pc.check_inductive(&ind, &ctors).is_err());
    }

    fn names_placeholder() -> NameId {
        NameId(999)
    }
}
