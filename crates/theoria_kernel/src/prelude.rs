//! The minimal standard prelude.
//!
//! Builds a [`GlobalEnv`] containing the four foundational inductives —
//! `Nat`, `Bool`, `List`, and `Option` — together with their constructors
//! and recursors. This is the smallest environment in which the kernel can
//! express and check the basic arithmetic and datatype manipulations the
//! elaborator will build on.
//!
//! ## Contents
//!
//! ```text
//! Nat      : Type 0
//! Nat.zero : Nat
//! Nat.succ : Nat -> Nat
//! Nat.rec.{u} : {motive : Nat -> Sort u}
//!             -> motive Nat.zero
//!             -> ((n : Nat) -> motive n -> motive (Nat.succ n))
//!             -> (t : Nat) -> motive t
//!
//! Bool       : Type 0
//! Bool.true  : Bool
//! Bool.false : Bool
//! Bool.rec.{u} : {motive : Bool -> Sort u}
//!              -> motive Bool.true
//!              -> motive Bool.false
//!              -> (b : Bool) -> motive b
//!
//! List.{u}      : Type u -> Type u
//! List.nil.{u}  : {A : Type u} -> List.{u} A
//! List.cons.{u} : {A : Type u} -> A -> List.{u} A -> List.{u} A
//! List.rec.{u,v} : {A : Type u}
//!                -> {motive : List.{u} A -> Sort v}
//!                -> motive (List.nil.{u} A)
//!                -> ((x : A) -> (xs : List.{u} A) -> motive xs
//!                    -> motive (List.cons.{u} A x xs))
//!                -> (l : List.{u} A) -> motive l
//!
//! Option.{u}      : Type u -> Type u
//! Option.none.{u} : {A : Type u} -> Option.{u} A
//! Option.some.{u} : {A : Type u} -> A -> Option.{u} A
//! Option.rec.{u,v} : {A : Type u}
//!                  -> {motive : Option.{u} A -> Sort v}
//!                  -> motive (Option.none.{u} A)
//!                  -> ((x : A) -> motive (Option.some.{u} A x))
//!                  -> (o : Option.{u} A) -> motive o
//! ```
//!
//! ## Soundness
//!
//! Every inductive is registered through
//! [`GlobalEnv::add_inductive`](crate::env::GlobalEnv::add_inductive), which
//! runs the strict/nested positivity check before committing. The prelude
//! cannot silently gain a non-positive inductive without changing this file.
//!
//! ## ι-rule context layout
//!
//! Each recursor rule's right-hand side is expressed in the context
//! `Nbe::try_iota` builds when the rule fires:
//!
//! ```text
//! [parameters..., motives..., minors..., indices..., constructor-fields...]
//! ```
//!
//! from deepest (highest index) to innermost (index `0`). The comments in
//! the registration helpers spell out the de Bruijn indices at every depth.

use crate::env::{
    ConstantInfo, ConstantVal, ConstructorVal, DefinitionVal, GlobalEnv, InductiveVal,
    RecursorRule, RecursorVal, TerminationObligation,
};
use crate::expr::{BinderInfo, DeBruijnIndex, Expr};
use crate::level::{Level, UniverseParamId};
use crate::name::{NameId, NameTable};
use crate::nbe::Transparency;
use alloc::boxed::Box;
use alloc::rc::Rc;
use alloc::vec;
use alloc::vec::Vec;

/// The names and environment produced by [`build_prelude`].
pub struct Prelude {
    /// The name table used to intern the prelude's identifiers.
    pub names: NameTable,
    /// The environment containing all prelude declarations.
    pub env: GlobalEnv,

    // --- Nat ----------------------------------------------------------
    /// `Nat`'s identifier.
    pub nat: NameId,
    /// `Nat.zero`'s identifier.
    pub zero: NameId,
    /// `Nat.succ`'s identifier.
    pub succ: NameId,
    /// `Nat.rec`'s identifier.
    pub rec: NameId,
    /// `Nat.rec`'s universe parameter.
    pub rec_universe: UniverseParamId,
    /// `Nat.pred`'s identifier.
    pub pred: NameId,
    /// `Nat.add`'s identifier.
    pub add: NameId,
    /// `Nat.mul`'s identifier.
    pub mul: NameId,
    /// `Nat.sub`'s identifier.
    pub sub: NameId,
    /// `Nat.pow`'s identifier.
    pub pow: NameId,

    // --- Bool.not and Nat comparisons ---------------------------------
    /// `Bool.not`'s identifier.
    pub bool_not: NameId,
    /// `Nat.is_zero`'s identifier.
    pub nat_is_zero: NameId,
    /// `Nat.beq_step`'s identifier.
    pub nat_beq_step: NameId,
    /// `Nat.beq`'s identifier.
    pub nat_beq: NameId,
    /// `Nat.bne`'s identifier.
    pub nat_bne: NameId,
    /// `Nat.ble`'s identifier.
    pub nat_ble: NameId,
    /// `Nat.blt`'s identifier.
    pub nat_blt: NameId,
    /// `Nat.bgt`'s identifier.
    pub nat_bgt: NameId,
    /// `Nat.bge`'s identifier.
    pub nat_bge: NameId,

    // --- Bool ---------------------------------------------------------
    /// `Bool`'s identifier.
    pub bool_name: NameId,
    /// `Bool.true`'s identifier.
    pub bool_true: NameId,
    /// `Bool.false`'s identifier.
    pub bool_false: NameId,
    /// `Bool.rec`'s identifier.
    pub bool_rec: NameId,
    /// `Bool.rec`'s universe parameter.
    pub bool_rec_u: UniverseParamId,

    // --- List ---------------------------------------------------------
    /// `List`'s identifier.
    pub list: NameId,
    /// `List.nil`'s identifier.
    pub list_nil: NameId,
    /// `List.cons`'s identifier.
    pub list_cons: NameId,
    /// `List.rec`'s identifier.
    pub list_rec: NameId,
    /// `List`'s universe parameter (shared with `List.rec`).
    pub list_u: UniverseParamId,
    /// `List.rec`'s motive universe parameter.
    pub list_rec_v: UniverseParamId,

    // --- Option -------------------------------------------------------
    /// `Option`'s identifier.
    pub option: NameId,
    /// `Option.none`'s identifier.
    pub option_none: NameId,
    /// `Option.some`'s identifier.
    pub option_some: NameId,
    /// `Option.rec`'s identifier.
    pub option_rec: NameId,
    /// `Option`'s universe parameter (shared with `Option.rec`).
    pub option_u: UniverseParamId,
    /// `Option.rec`'s motive universe parameter.
    pub option_rec_v: UniverseParamId,

    // --- Eq -----------------------------------------------------------
    /// `Eq`'s identifier.
    pub eq: NameId,
    /// `Eq.refl`'s identifier.
    pub eq_refl: NameId,
    /// `Eq.rec`'s identifier.
    pub eq_rec: NameId,
    /// `Eq`'s universe parameter (shared with `Eq.refl` and `Eq.rec`).
    pub eq_u: UniverseParamId,
    /// `Eq.rec`'s motive universe parameter.
    pub eq_rec_v: UniverseParamId,
}

/// Backwards-compatible alias for [`Prelude`].
pub type NatPrelude = Prelude;

/// Build the full prelude: `Nat`, `Bool`, `List`, and `Option`, each with
/// its constructors and recursor.
///
/// # Panics
///
/// Panics if any declaration fails to register. This can only happen if the
/// kernel itself is broken: the prelude is closed, validated by hand, and
/// does not depend on user input.
#[must_use]
pub fn build_prelude() -> Prelude {
    let mut names = NameTable::new();
    let mut env = GlobalEnv::new();

    // Nat.
    let nat = names.intern("Nat");
    let zero = names.intern("Nat.zero");
    let succ = names.intern("Nat.succ");
    let rec = names.intern("Nat.rec");
    let rec_universe = UniverseParamId::fresh();
    let pred = names.intern("Nat.pred");
    let add = names.intern("Nat.add");
    let mul = names.intern("Nat.mul");
    let sub = names.intern("Nat.sub");
    let pow = names.intern("Nat.pow");
    let bool_not = names.intern("Bool.not");
    let nat_is_zero = names.intern("Nat.is_zero");
    let nat_beq_step = names.intern("Nat.beq_step");
    let nat_beq = names.intern("Nat.beq");
    let nat_bne = names.intern("Nat.bne");
    let nat_ble = names.intern("Nat.ble");
    let nat_blt = names.intern("Nat.blt");
    let nat_bgt = names.intern("Nat.bgt");
    let nat_bge = names.intern("Nat.bge");

    // Bool.
    let bool_name = names.intern("Bool");
    let bool_true = names.intern("Bool.true");
    let bool_false = names.intern("Bool.false");
    let bool_rec = names.intern("Bool.rec");
    let bool_rec_u = UniverseParamId::fresh();

    // List.
    let list = names.intern("List");
    let list_nil = names.intern("List.nil");
    let list_cons = names.intern("List.cons");
    let list_rec = names.intern("List.rec");
    let list_u = UniverseParamId::fresh();
    let list_rec_v = UniverseParamId::fresh();

    // Option.
    let option = names.intern("Option");
    let option_none = names.intern("Option.none");
    let option_some = names.intern("Option.some");
    let option_rec = names.intern("Option.rec");
    let option_u = UniverseParamId::fresh();
    let option_rec_v = UniverseParamId::fresh();

    // Eq.
    let eq = names.intern("Eq");
    let eq_refl = names.intern("Eq.refl");
    let eq_rec = names.intern("Eq.rec");
    let eq_u = UniverseParamId::fresh();
    let eq_rec_v = UniverseParamId::fresh();

    register_nat(&mut names, &mut env, nat, zero, succ, rec, rec_universe);
    // Arithmetic, in dependency order: pred and add first, then mul and
    // sub, then pow.
    register_pred(&mut names, &mut env, nat, zero, rec, pred);
    register_add(&mut names, &mut env, nat, succ, rec, add);
    register_mul(&mut names, &mut env, nat, zero, add, rec, mul);
    register_sub(&mut names, &mut env, nat, pred, rec, sub);
    register_pow(&mut names, &mut env, nat, zero, succ, mul, rec, pow);
    // Comparisons, in dependency order: not and is_zero first (no intra
    // dependencies beyond the inductives), then beq_step, beq, and the
    // one-line wrappers bne, ble, blt, bgt, bge. Each helper interns the
    // names it needs itself (interning is idempotent, so the ids match
    // the ones above).
    register_bool_not(&mut names, &mut env);
    register_nat_is_zero(&mut names, &mut env);
    register_nat_beq_step(&mut names, &mut env);
    register_nat_beq(&mut names, &mut env);
    register_nat_bne(&mut names, &mut env);
    register_nat_ble(&mut names, &mut env);
    register_nat_blt(&mut names, &mut env);
    register_nat_bgt(&mut names, &mut env);
    register_nat_bge(&mut names, &mut env);
    register_eq(&mut names, &mut env, eq, eq_refl, eq_rec, eq_u, eq_rec_v);
    register_bool(
        &mut names, &mut env, bool_name, bool_true, bool_false, bool_rec, bool_rec_u,
    );
    register_list(
        &mut names, &mut env, list, list_nil, list_cons, list_rec, list_u, list_rec_v,
    );
    register_option(
        &mut names,
        &mut env,
        option,
        option_none,
        option_some,
        option_rec,
        option_u,
        option_rec_v,
    );

    Prelude {
        names,
        env,
        nat,
        zero,
        succ,
        rec,
        rec_universe,
        pred,
        add,
        mul,
        sub,
        pow,
        bool_not,
        nat_is_zero,
        nat_beq_step,
        nat_beq,
        nat_bne,
        nat_ble,
        nat_blt,
        nat_bgt,
        nat_bge,
        bool_name,
        bool_true,
        bool_false,
        bool_rec,
        bool_rec_u,
        list,
        list_nil,
        list_cons,
        list_rec,
        list_u,
        list_rec_v,
        option,
        option_none,
        option_some,
        option_rec,
        option_u,
        option_rec_v,
        eq,
        eq_refl,
        eq_rec,
        eq_u,
        eq_rec_v,
    }
}

/// Backwards-compatible alias for [`build_prelude`].
#[must_use]
pub fn build_nat_prelude() -> Prelude {
    build_prelude()
}

// ===========================================================================
// Nat
// ===========================================================================

fn register_nat(
    names: &mut NameTable,
    env: &mut GlobalEnv,
    nat: NameId,
    zero: NameId,
    succ: NameId,
    rec: NameId,
    u: UniverseParamId,
) {
    let underscore = names.intern("_");
    let motive = names.intern("motive");
    let zero_case = names.intern("zero_case");
    let succ_case = names.intern("succ_case");
    let n = names.intern("n");
    let h = names.intern("h");
    let major = names.intern("major");

    // Nat : Type 0
    let nat_ty = Expr::type0();
    // Nat.zero : Nat
    let zero_ty = Expr::Const(nat, vec![]);
    // Nat.succ : Nat -> Nat
    let succ_ty = Expr::Pi(
        BinderInfo::Default,
        underscore,
        Box::new(Expr::Const(nat, vec![])),
        Box::new(Expr::Const(nat, vec![])),
    );

    // Nat.rec.{u} : {motive : Nat -> Sort u}
    //             -> motive Nat.zero
    //             -> ((n : Nat) -> motive n -> motive (Nat.succ n))
    //             -> (major : Nat) -> motive major
    //
    // depth 0:  motive_ty    = Π (_ : Nat). Sort u
    // depth 1:  zero_case_ty = motive Nat.zero              (motive = Var(0))
    // depth 2:  succ_case_ty = (n : Nat) -> motive n -> motive (Nat.succ n)
    //   at depth 3 (motive=Var(2), n=Var(0)):  motive n
    //   at depth 4 (motive=Var(3), n=Var(1)):  motive (Nat.succ n)
    // depth 3:  major_ty     = (major : Nat) -> motive major
    //   at depth 4 (motive=Var(3), major=Var(0)): motive major
    let motive_ty = Expr::Pi(
        BinderInfo::Default,
        underscore,
        Box::new(Expr::Const(nat, vec![])),
        Box::new(Expr::Sort(Level::param(u))),
    );
    let zero_case_ty = Expr::app(Expr::Var(DeBruijnIndex(0)), Expr::Const(zero, vec![]));
    let succ_case_ty = Expr::Pi(
        BinderInfo::Default,
        n,
        Box::new(Expr::Const(nat, vec![])),
        Box::new(Expr::Pi(
            BinderInfo::Default,
            h,
            Box::new(Expr::app(
                Expr::Var(DeBruijnIndex(2)),
                Expr::Var(DeBruijnIndex(0)),
            )),
            Box::new(Expr::app(
                Expr::Var(DeBruijnIndex(3)),
                Expr::app(Expr::Const(succ, vec![]), Expr::Var(DeBruijnIndex(1))),
            )),
        )),
    );
    let major_ty = Expr::Pi(
        BinderInfo::Default,
        major,
        Box::new(Expr::Const(nat, vec![])),
        Box::new(Expr::app(
            Expr::Var(DeBruijnIndex(3)),
            Expr::Var(DeBruijnIndex(0)),
        )),
    );
    let rec_ty = Expr::Pi(
        BinderInfo::Implicit,
        motive,
        Box::new(motive_ty),
        Box::new(Expr::Pi(
            BinderInfo::Default,
            zero_case,
            Box::new(zero_case_ty),
            Box::new(Expr::Pi(
                BinderInfo::Default,
                succ_case,
                Box::new(succ_case_ty),
                Box::new(major_ty),
            )),
        )),
    );

    // Zero rule RHS: context [motive, zero_case, succ_case]. Var(1)=zero_case.
    let zero_rule_rhs = Expr::Var(DeBruijnIndex(1));

    // Succ rule RHS: context [motive, zero_case, succ_case, n].
    //   Var(0)=n, Var(1)=succ_case, Var(2)=zero_case, Var(3)=motive.
    //   succ_case n (Nat.rec.{u} motive zero_case succ_case n)
    let recursive_call = Expr::Const(rec, vec![Level::param(u)]).apps([
        Expr::Var(DeBruijnIndex(3)),
        Expr::Var(DeBruijnIndex(2)),
        Expr::Var(DeBruijnIndex(1)),
        Expr::Var(DeBruijnIndex(0)),
    ]);
    let succ_rule_rhs = Expr::app(
        Expr::app(Expr::Var(DeBruijnIndex(1)), Expr::Var(DeBruijnIndex(0))),
        recursive_call,
    );

    let ind = InductiveVal {
        base: ConstantVal {
            name: nat,
            universe_params: Vec::new(),
            ty: Rc::new(nat_ty),
        },
        num_params: 0,
        num_indices: 0,
        all: vec![nat],
        constructors: vec![zero, succ],
        is_recursive: true,
        is_nested: false,
        is_unsafe: false,
    };
    let ctors = vec![
        ConstructorVal {
            base: ConstantVal {
                name: zero,
                universe_params: Vec::new(),
                ty: Rc::new(zero_ty),
            },
            inductive: nat,
            index: 0,
            num_params: 0,
            num_fields: 0,
            is_unsafe: false,
        },
        ConstructorVal {
            base: ConstantVal {
                name: succ,
                universe_params: Vec::new(),
                ty: Rc::new(succ_ty),
            },
            inductive: nat,
            index: 1,
            num_params: 0,
            num_fields: 1,
            is_unsafe: false,
        },
    ];
    let recursor = RecursorVal {
        base: ConstantVal {
            name: rec,
            universe_params: vec![u],
            ty: Rc::new(rec_ty),
        },
        all: vec![nat],
        num_params: 0,
        num_indices: 0,
        num_motives: 1,
        num_minors: 2,
        rules: vec![
            RecursorRule {
                constructor: zero,
                num_fields: 0,
                rhs: Rc::new(zero_rule_rhs),
            },
            RecursorRule {
                constructor: succ,
                num_fields: 1,
                rhs: Rc::new(succ_rule_rhs),
            },
        ],
        is_k: false,
        is_unsafe: false,
    };

    env.add_inductive(ind, ctors, recursor)
        .expect("Nat prelude failed to register");
}

// ===========================================================================
// Nat arithmetic (pred, add, mul, sub, pow)
// ===========================================================================
//
// Each operator is a transparent definition over `Nat.rec.{1}` (the `1`
// because every motive here is `λ _ : Nat. Nat`, which lands in `Sort 1`).
// Bodies never mention their own names, so the termination gate is a
// no-op for each insertion.

/// Insert a transparent, non-recursive operator definition.
///
/// Used for the Nat arithmetic and comparison operators alike; the name
/// is generic because the mechanism (a closed body over earlier
/// definitions) is the same.
fn insert_op(env: &mut GlobalEnv, name: NameId, ty: Expr, body: Expr) {
    env.add(ConstantInfo::Definition(DefinitionVal {
        base: ConstantVal {
            name,
            universe_params: Vec::new(),
            ty: Rc::new(ty),
        },
        body: Rc::new(body),
        transparency: Transparency::Semireducible,
        termination: TerminationObligation::none(),
    }))
    .expect("Nat arithmetic prelude failed to register");
}

/// `Nat.pred : Nat -> Nat := λ n. rec (λ_. Nat) zero (λk. λ_. k) n`.
fn register_pred(
    names: &mut NameTable,
    env: &mut GlobalEnv,
    nat: NameId,
    zero: NameId,
    rec: NameId,
    pred: NameId,
) {
    let underscore = names.intern("_");
    let n = names.intern("n");
    let k = names.intern("k");
    let nat_ty = Expr::Const(nat, Vec::new());

    // motive := λ (_ : Nat). Nat
    let motive = Expr::Lam(
        BinderInfo::Default,
        underscore,
        Box::new(nat_ty.clone()),
        Box::new(nat_ty.clone()),
    );
    // succ_case := λ (k : Nat). λ (_ : Nat). k
    //   inner body at depth 2 (outer n, k, _): k = Var(1).
    let succ_case = Expr::Lam(
        BinderInfo::Default,
        k,
        Box::new(nat_ty.clone()),
        Box::new(Expr::Lam(
            BinderInfo::Default,
            underscore,
            Box::new(nat_ty.clone()),
            Box::new(Expr::Var(DeBruijnIndex(1))),
        )),
    );
    // body at depth 1 (outer n): major = Var(0).
    let body = Expr::Lam(
        BinderInfo::Default,
        n,
        Box::new(nat_ty.clone()),
        Box::new(Expr::Const(rec, vec![Level::Zero.succ()]).apps([
            motive,
            Expr::Const(zero, Vec::new()),
            succ_case,
            Expr::Var(DeBruijnIndex(0)),
        ])),
    );
    let ty = Expr::Pi(
        BinderInfo::Default,
        n,
        Box::new(nat_ty.clone()),
        Box::new(nat_ty),
    );
    insert_op(env, pred, ty, body);
}

/// `Nat.add : Nat -> Nat -> Nat`
///   `:= λ m. λ n. rec (λ_. Nat) n (λk. λih. succ ih) m`.
fn register_add(
    names: &mut NameTable,
    env: &mut GlobalEnv,
    nat: NameId,
    succ: NameId,
    rec: NameId,
    add: NameId,
) {
    let underscore = names.intern("_");
    let m = names.intern("m");
    let n = names.intern("n");
    let k = names.intern("k");
    let ih = names.intern("ih");
    let nat_ty = Expr::Const(nat, Vec::new());

    let motive = Expr::Lam(
        BinderInfo::Default,
        underscore,
        Box::new(nat_ty.clone()),
        Box::new(nat_ty.clone()),
    );
    // succ_case := λ (k : Nat). λ (ih : Nat). Nat.succ ih
    //   inner body at depth 4 (m, n, k, ih): ih = Var(0).
    let succ_case = Expr::Lam(
        BinderInfo::Default,
        k,
        Box::new(nat_ty.clone()),
        Box::new(Expr::Lam(
            BinderInfo::Default,
            ih,
            Box::new(nat_ty.clone()),
            Box::new(Expr::app(
                Expr::Const(succ, Vec::new()),
                Expr::Var(DeBruijnIndex(0)),
            )),
        )),
    );
    // body at depth 2 (m, n): zero_case = n = Var(0), major = m = Var(1).
    let body = Expr::Lam(
        BinderInfo::Default,
        m,
        Box::new(nat_ty.clone()),
        Box::new(Expr::Lam(
            BinderInfo::Default,
            n,
            Box::new(nat_ty.clone()),
            Box::new(Expr::Const(rec, vec![Level::Zero.succ()]).apps([
                motive,
                Expr::Var(DeBruijnIndex(0)),
                succ_case,
                Expr::Var(DeBruijnIndex(1)),
            ])),
        )),
    );
    let ty = Expr::Pi(
        BinderInfo::Default,
        m,
        Box::new(nat_ty.clone()),
        Box::new(Expr::Pi(
            BinderInfo::Default,
            n,
            Box::new(nat_ty.clone()),
            Box::new(nat_ty),
        )),
    );
    insert_op(env, add, ty, body);
}

/// `Nat.mul : Nat -> Nat -> Nat`
///   `:= λ m. λ n. rec (λ_. Nat) zero (λk. λih. add n ih) m`.
fn register_mul(
    names: &mut NameTable,
    env: &mut GlobalEnv,
    nat: NameId,
    zero: NameId,
    add: NameId,
    rec: NameId,
    mul: NameId,
) {
    let underscore = names.intern("_");
    let m = names.intern("m");
    let n = names.intern("n");
    let k = names.intern("k");
    let ih = names.intern("ih");
    let nat_ty = Expr::Const(nat, Vec::new());

    let motive = Expr::Lam(
        BinderInfo::Default,
        underscore,
        Box::new(nat_ty.clone()),
        Box::new(nat_ty.clone()),
    );
    // succ_case body at depth 4 (m, n, k, ih): Nat.add n ih.
    // (m=Var(3), n=Var(2), k=Var(1), ih=Var(0)).
    let succ_case = Expr::Lam(
        BinderInfo::Default,
        k,
        Box::new(nat_ty.clone()),
        Box::new(Expr::Lam(
            BinderInfo::Default,
            ih,
            Box::new(nat_ty.clone()),
            Box::new(
                Expr::Const(add, Vec::new())
                    .apps([Expr::Var(DeBruijnIndex(2)), Expr::Var(DeBruijnIndex(0))]),
            ),
        )),
    );
    // body at depth 2 (m, n): major = m = Var(1).
    let body = Expr::Lam(
        BinderInfo::Default,
        m,
        Box::new(nat_ty.clone()),
        Box::new(Expr::Lam(
            BinderInfo::Default,
            n,
            Box::new(nat_ty.clone()),
            Box::new(Expr::Const(rec, vec![Level::Zero.succ()]).apps([
                motive,
                Expr::Const(zero, Vec::new()),
                succ_case,
                Expr::Var(DeBruijnIndex(1)),
            ])),
        )),
    );
    let ty = Expr::Pi(
        BinderInfo::Default,
        m,
        Box::new(nat_ty.clone()),
        Box::new(Expr::Pi(
            BinderInfo::Default,
            n,
            Box::new(nat_ty.clone()),
            Box::new(nat_ty),
        )),
    );
    insert_op(env, mul, ty, body);
}

/// `Nat.sub : Nat -> Nat -> Nat`
///   `:= λ m. λ n. rec (λ_. Nat) m (λ_. λih. pred ih) n`
/// (truncated subtraction: `sub m 0 = m`, `sub m (succ k) = pred (sub m k)`).
fn register_sub(
    names: &mut NameTable,
    env: &mut GlobalEnv,
    nat: NameId,
    pred: NameId,
    rec: NameId,
    sub: NameId,
) {
    let underscore = names.intern("_");
    let m = names.intern("m");
    let n = names.intern("n");
    let ih = names.intern("ih");
    let nat_ty = Expr::Const(nat, Vec::new());

    let motive = Expr::Lam(
        BinderInfo::Default,
        underscore,
        Box::new(nat_ty.clone()),
        Box::new(nat_ty.clone()),
    );
    // succ_case body at depth 4 (m, n, _, ih): Nat.pred ih.
    let succ_case = Expr::Lam(
        BinderInfo::Default,
        underscore,
        Box::new(nat_ty.clone()),
        Box::new(Expr::Lam(
            BinderInfo::Default,
            ih,
            Box::new(nat_ty.clone()),
            Box::new(Expr::app(
                Expr::Const(pred, Vec::new()),
                Expr::Var(DeBruijnIndex(0)),
            )),
        )),
    );
    // body at depth 2 (m, n): zero_case = m = Var(1), major = n = Var(0).
    let body = Expr::Lam(
        BinderInfo::Default,
        m,
        Box::new(nat_ty.clone()),
        Box::new(Expr::Lam(
            BinderInfo::Default,
            n,
            Box::new(nat_ty.clone()),
            Box::new(Expr::Const(rec, vec![Level::Zero.succ()]).apps([
                motive,
                Expr::Var(DeBruijnIndex(1)),
                succ_case,
                Expr::Var(DeBruijnIndex(0)),
            ])),
        )),
    );
    let ty = Expr::Pi(
        BinderInfo::Default,
        m,
        Box::new(nat_ty.clone()),
        Box::new(Expr::Pi(
            BinderInfo::Default,
            n,
            Box::new(nat_ty.clone()),
            Box::new(nat_ty),
        )),
    );
    insert_op(env, sub, ty, body);
}

/// `Nat.pow : Nat -> Nat -> Nat`
///   `:= λ m. λ n. rec (λ_. Nat) 1 (λ_. λih. mul m ih) n`.
#[allow(clippy::too_many_arguments)]
fn register_pow(
    names: &mut NameTable,
    env: &mut GlobalEnv,
    nat: NameId,
    zero: NameId,
    succ: NameId,
    mul: NameId,
    rec: NameId,
    pow: NameId,
) {
    let underscore = names.intern("_");
    let m = names.intern("m");
    let n = names.intern("n");
    let ih = names.intern("ih");
    let nat_ty = Expr::Const(nat, Vec::new());

    let motive = Expr::Lam(
        BinderInfo::Default,
        underscore,
        Box::new(nat_ty.clone()),
        Box::new(nat_ty.clone()),
    );
    // one := Nat.succ Nat.zero.
    let one = Expr::app(Expr::Const(succ, Vec::new()), Expr::Const(zero, Vec::new()));
    // succ_case body at depth 4 (m, n, _, ih): Nat.mul m ih.
    let succ_case = Expr::Lam(
        BinderInfo::Default,
        underscore,
        Box::new(nat_ty.clone()),
        Box::new(Expr::Lam(
            BinderInfo::Default,
            ih,
            Box::new(nat_ty.clone()),
            Box::new(
                Expr::Const(mul, Vec::new())
                    .apps([Expr::Var(DeBruijnIndex(3)), Expr::Var(DeBruijnIndex(0))]),
            ),
        )),
    );
    // body at depth 2 (m, n): major = n = Var(0).
    let body = Expr::Lam(
        BinderInfo::Default,
        m,
        Box::new(nat_ty.clone()),
        Box::new(Expr::Lam(
            BinderInfo::Default,
            n,
            Box::new(nat_ty.clone()),
            Box::new(Expr::Const(rec, vec![Level::Zero.succ()]).apps([
                motive,
                one,
                succ_case,
                Expr::Var(DeBruijnIndex(0)),
            ])),
        )),
    );
    let ty = Expr::Pi(
        BinderInfo::Default,
        m,
        Box::new(nat_ty.clone()),
        Box::new(Expr::Pi(
            BinderInfo::Default,
            n,
            Box::new(nat_ty.clone()),
            Box::new(nat_ty),
        )),
    );
    insert_op(env, pow, ty, body);
}

// ===========================================================================
// Bool.not and Nat comparisons
// ===========================================================================
//
// Each helper interns the names it needs itself; interning is idempotent,
// so these match the ids in `build_prelude`. Every `Nat.rec` / `Bool.rec`
// uses `Level::Zero.succ()` because every motive here lands in `Sort 1`
// (codomains are `Nat` or `Bool`, both in `Sort 1`).

/// `Bool.not : Bool -> Bool`
///   `:= λ b. Bool.rec.{1} (λ _ : Bool. Bool) Bool.false Bool.true b`.
fn register_bool_not(names: &mut NameTable, env: &mut GlobalEnv) {
    let underscore = names.intern("_");
    let b = names.intern("b");
    let bool_name = names.intern("Bool");
    let bool_true = names.intern("Bool.true");
    let bool_false = names.intern("Bool.false");
    let bool_rec = names.intern("Bool.rec");
    let bool_not = names.intern("Bool.not");
    let bool_ty = Expr::Const(bool_name, Vec::new());

    let motive = Expr::Lam(
        BinderInfo::Default,
        underscore,
        Box::new(bool_ty.clone()),
        Box::new(bool_ty.clone()),
    );
    // body at depth 1 (b): rec-app args in order
    // motive, true_case=false, false_case=true, major=b=Var(0).
    let body = Expr::Lam(
        BinderInfo::Default,
        b,
        Box::new(bool_ty.clone()),
        Box::new(Expr::Const(bool_rec, vec![Level::Zero.succ()]).apps([
            motive,
            Expr::Const(bool_false, Vec::new()),
            Expr::Const(bool_true, Vec::new()),
            Expr::Var(DeBruijnIndex(0)),
        ])),
    );
    let ty = Expr::Pi(
        BinderInfo::Default,
        b,
        Box::new(bool_ty.clone()),
        Box::new(bool_ty),
    );
    insert_op(env, bool_not, ty, body);
}

/// `Nat.is_zero : Nat -> Bool`
///   `:= λ n. Nat.rec.{1} (λ _ : Nat. Bool) Bool.true (λk. λ_ : Bool. Bool.false) n`.
fn register_nat_is_zero(names: &mut NameTable, env: &mut GlobalEnv) {
    let underscore = names.intern("_");
    let n = names.intern("n");
    let k = names.intern("k");
    let nat = names.intern("Nat");
    let rec = names.intern("Nat.rec");
    let bool_name = names.intern("Bool");
    let bool_true = names.intern("Bool.true");
    let bool_false = names.intern("Bool.false");
    let nat_is_zero = names.intern("Nat.is_zero");
    let nat_ty = Expr::Const(nat, Vec::new());
    let bool_ty = Expr::Const(bool_name, Vec::new());

    // motive := λ (_ : Nat). Bool
    let motive = Expr::Lam(
        BinderInfo::Default,
        underscore,
        Box::new(nat_ty.clone()),
        Box::new(bool_ty.clone()),
    );
    // succ_case := λ (k : Nat). λ (_ : Bool). Bool.false (no vars in body).
    let succ_case = Expr::Lam(
        BinderInfo::Default,
        k,
        Box::new(nat_ty.clone()),
        Box::new(Expr::Lam(
            BinderInfo::Default,
            underscore,
            Box::new(bool_ty.clone()),
            Box::new(Expr::Const(bool_false, Vec::new())),
        )),
    );
    // body at depth 1 (n): major = Var(0).
    let body = Expr::Lam(
        BinderInfo::Default,
        n,
        Box::new(nat_ty.clone()),
        Box::new(Expr::Const(rec, vec![Level::Zero.succ()]).apps([
            motive,
            Expr::Const(bool_true, Vec::new()),
            succ_case,
            Expr::Var(DeBruijnIndex(0)),
        ])),
    );
    let ty = Expr::Pi(
        BinderInfo::Default,
        n,
        Box::new(nat_ty.clone()),
        Box::new(bool_ty),
    );
    insert_op(env, nat_is_zero, ty, body);
}

/// `Nat.beq_step : (Nat -> Bool) -> Nat -> Bool`
///   `:= λ ih. λ n. Nat.rec.{1} (λ _ : Nat. Bool) Bool.false (λj. λ_ : Bool. ih j) n`.
///
/// The inner recursion over `n`, parameterised by the outer hypothesis
/// `ih`. Factored out so `Nat.beq` stays readable and testable.
fn register_nat_beq_step(names: &mut NameTable, env: &mut GlobalEnv) {
    let underscore = names.intern("_");
    let ih = names.intern("ih");
    let n = names.intern("n");
    let j = names.intern("j");
    let nat = names.intern("Nat");
    let rec = names.intern("Nat.rec");
    let bool_name = names.intern("Bool");
    let bool_false = names.intern("Bool.false");
    let nat_beq_step = names.intern("Nat.beq_step");
    let nat_ty = Expr::Const(nat, Vec::new());
    let bool_ty = Expr::Const(bool_name, Vec::new());
    let nat_to_bool = Expr::Pi(
        BinderInfo::Default,
        underscore,
        Box::new(nat_ty.clone()),
        Box::new(bool_ty.clone()),
    );

    let motive = Expr::Lam(
        BinderInfo::Default,
        underscore,
        Box::new(nat_ty.clone()),
        Box::new(bool_ty.clone()),
    );
    // succ_case body at depth 4 (ih, n, j, _): ih j = App(Var(3), Var(1)).
    let succ_case = Expr::Lam(
        BinderInfo::Default,
        j,
        Box::new(nat_ty.clone()),
        Box::new(Expr::Lam(
            BinderInfo::Default,
            underscore,
            Box::new(bool_ty.clone()),
            Box::new(Expr::app(
                Expr::Var(DeBruijnIndex(3)),
                Expr::Var(DeBruijnIndex(1)),
            )),
        )),
    );
    // body at depth 2 (ih, n): major = n = Var(0).
    let body = Expr::Lam(
        BinderInfo::Default,
        ih,
        Box::new(nat_to_bool.clone()),
        Box::new(Expr::Lam(
            BinderInfo::Default,
            n,
            Box::new(nat_ty.clone()),
            Box::new(Expr::Const(rec, vec![Level::Zero.succ()]).apps([
                motive,
                Expr::Const(bool_false, Vec::new()),
                succ_case,
                Expr::Var(DeBruijnIndex(0)),
            ])),
        )),
    );
    let ty = Expr::Pi(
        BinderInfo::Default,
        ih,
        Box::new(nat_to_bool.clone()),
        Box::new(Expr::Pi(
            BinderInfo::Default,
            n,
            Box::new(nat_ty.clone()),
            Box::new(bool_ty),
        )),
    );
    insert_op(env, nat_beq_step, ty, body);
}

/// `Nat.beq : Nat -> Nat -> Bool`
///   `:= λ m. λ n. (Nat.rec.{1} (λ _ : Nat. Nat -> Bool) Nat.is_zero (λk. λih : Nat -> Bool. Nat.beq_step ih) m) n`.
///
/// Recurse on `m` with a function-valued motive; apply the result to `n`.
fn register_nat_beq(names: &mut NameTable, env: &mut GlobalEnv) {
    let underscore = names.intern("_");
    let m = names.intern("m");
    let n = names.intern("n");
    let k = names.intern("k");
    let ih = names.intern("ih");
    let nat = names.intern("Nat");
    let rec = names.intern("Nat.rec");
    let bool_name = names.intern("Bool");
    let nat_is_zero = names.intern("Nat.is_zero");
    let nat_beq_step = names.intern("Nat.beq_step");
    let nat_beq = names.intern("Nat.beq");
    let nat_ty = Expr::Const(nat, Vec::new());
    let bool_ty = Expr::Const(bool_name, Vec::new());
    let nat_to_bool = Expr::Pi(
        BinderInfo::Default,
        underscore,
        Box::new(nat_ty.clone()),
        Box::new(bool_ty.clone()),
    );

    // motive := λ (_ : Nat). Nat -> Bool
    let motive = Expr::Lam(
        BinderInfo::Default,
        underscore,
        Box::new(nat_ty.clone()),
        Box::new(nat_to_bool.clone()),
    );
    // succ_case := λ (k : Nat). λ (ih : Nat -> Bool). Nat.beq_step ih
    //   inner body at depth 4 (m, n, k, ih): beq_step applied to Var(0).
    let succ_case = Expr::Lam(
        BinderInfo::Default,
        k,
        Box::new(nat_ty.clone()),
        Box::new(Expr::Lam(
            BinderInfo::Default,
            ih,
            Box::new(nat_to_bool.clone()),
            Box::new(Expr::app(
                Expr::Const(nat_beq_step, Vec::new()),
                Expr::Var(DeBruijnIndex(0)),
            )),
        )),
    );
    // At depth 2 (m, n): (rec motive is_zero succ_case m) n.
    let rec_applied = Expr::Const(rec, vec![Level::Zero.succ()]).apps([
        motive,
        Expr::Const(nat_is_zero, Vec::new()),
        succ_case,
        Expr::Var(DeBruijnIndex(1)),
    ]);
    let body = Expr::Lam(
        BinderInfo::Default,
        m,
        Box::new(nat_ty.clone()),
        Box::new(Expr::Lam(
            BinderInfo::Default,
            n,
            Box::new(nat_ty.clone()),
            Box::new(Expr::app(rec_applied, Expr::Var(DeBruijnIndex(0)))),
        )),
    );
    let ty = Expr::Pi(
        BinderInfo::Default,
        m,
        Box::new(nat_ty.clone()),
        Box::new(Expr::Pi(
            BinderInfo::Default,
            n,
            Box::new(nat_ty.clone()),
            Box::new(bool_ty),
        )),
    );
    insert_op(env, nat_beq, ty, body);
}

/// `Nat.bne : Nat -> Nat -> Bool := λ m. λ n. Bool.not (Nat.beq m n)`.
fn register_nat_bne(names: &mut NameTable, env: &mut GlobalEnv) {
    let m = names.intern("m");
    let n = names.intern("n");
    let nat = names.intern("Nat");
    let bool_name = names.intern("Bool");
    let bool_not = names.intern("Bool.not");
    let nat_beq = names.intern("Nat.beq");
    let nat_bne = names.intern("Nat.bne");
    let nat_ty = Expr::Const(nat, Vec::new());
    let bool_ty = Expr::Const(bool_name, Vec::new());

    // body at depth 2 (m, n): not (beq m n), i.e. not (beq Var(1) Var(0)).
    let beq_app = Expr::Const(nat_beq, Vec::new())
        .apps([Expr::Var(DeBruijnIndex(1)), Expr::Var(DeBruijnIndex(0))]);
    let body = Expr::Lam(
        BinderInfo::Default,
        m,
        Box::new(nat_ty.clone()),
        Box::new(Expr::Lam(
            BinderInfo::Default,
            n,
            Box::new(nat_ty.clone()),
            Box::new(Expr::app(Expr::Const(bool_not, Vec::new()), beq_app)),
        )),
    );
    let ty = Expr::Pi(
        BinderInfo::Default,
        m,
        Box::new(nat_ty.clone()),
        Box::new(Expr::Pi(
            BinderInfo::Default,
            n,
            Box::new(nat_ty.clone()),
            Box::new(bool_ty),
        )),
    );
    insert_op(env, nat_bne, ty, body);
}

/// `Nat.ble : Nat -> Nat -> Bool := λ m. λ n. Nat.beq (Nat.sub m n) Nat.zero`.
///
/// `m ≤ n` iff `m - n = 0` in truncated subtraction: cheaper than a
/// second double recursion, and it reuses `Nat.beq`.
fn register_nat_ble(names: &mut NameTable, env: &mut GlobalEnv) {
    let m = names.intern("m");
    let n = names.intern("n");
    let nat = names.intern("Nat");
    let zero = names.intern("Nat.zero");
    let bool_name = names.intern("Bool");
    let nat_beq = names.intern("Nat.beq");
    let sub = names.intern("Nat.sub");
    let nat_ble = names.intern("Nat.ble");
    let nat_ty = Expr::Const(nat, Vec::new());
    let bool_ty = Expr::Const(bool_name, Vec::new());

    // body at depth 2 (m, n): beq (sub m n) zero.
    let sub_app = Expr::Const(sub, Vec::new())
        .apps([Expr::Var(DeBruijnIndex(1)), Expr::Var(DeBruijnIndex(0))]);
    let beq_app = Expr::Const(nat_beq, Vec::new()).apps([sub_app, Expr::Const(zero, Vec::new())]);
    let body = Expr::Lam(
        BinderInfo::Default,
        m,
        Box::new(nat_ty.clone()),
        Box::new(Expr::Lam(
            BinderInfo::Default,
            n,
            Box::new(nat_ty.clone()),
            Box::new(beq_app),
        )),
    );
    let ty = Expr::Pi(
        BinderInfo::Default,
        m,
        Box::new(nat_ty.clone()),
        Box::new(Expr::Pi(
            BinderInfo::Default,
            n,
            Box::new(nat_ty.clone()),
            Box::new(bool_ty),
        )),
    );
    insert_op(env, nat_ble, ty, body);
}

/// `Nat.blt : Nat -> Nat -> Bool := λ m. λ n. Nat.ble (Nat.succ m) n`.
fn register_nat_blt(names: &mut NameTable, env: &mut GlobalEnv) {
    let m = names.intern("m");
    let n = names.intern("n");
    let nat = names.intern("Nat");
    let succ = names.intern("Nat.succ");
    let bool_name = names.intern("Bool");
    let nat_ble = names.intern("Nat.ble");
    let nat_blt = names.intern("Nat.blt");
    let nat_ty = Expr::Const(nat, Vec::new());
    let bool_ty = Expr::Const(bool_name, Vec::new());

    // body at depth 2 (m, n): ble (succ m) n.
    let succ_m = Expr::app(Expr::Const(succ, Vec::new()), Expr::Var(DeBruijnIndex(1)));
    let ble_app = Expr::Const(nat_ble, Vec::new()).apps([succ_m, Expr::Var(DeBruijnIndex(0))]);
    let body = Expr::Lam(
        BinderInfo::Default,
        m,
        Box::new(nat_ty.clone()),
        Box::new(Expr::Lam(
            BinderInfo::Default,
            n,
            Box::new(nat_ty.clone()),
            Box::new(ble_app),
        )),
    );
    let ty = Expr::Pi(
        BinderInfo::Default,
        m,
        Box::new(nat_ty.clone()),
        Box::new(Expr::Pi(
            BinderInfo::Default,
            n,
            Box::new(nat_ty.clone()),
            Box::new(bool_ty),
        )),
    );
    insert_op(env, nat_blt, ty, body);
}

/// `Nat.bgt : Nat -> Nat -> Bool := λ m. λ n. Nat.blt n m`.
fn register_nat_bgt(names: &mut NameTable, env: &mut GlobalEnv) {
    let m = names.intern("m");
    let n = names.intern("n");
    let nat = names.intern("Nat");
    let bool_name = names.intern("Bool");
    let nat_blt = names.intern("Nat.blt");
    let nat_bgt = names.intern("Nat.bgt");
    let nat_ty = Expr::Const(nat, Vec::new());
    let bool_ty = Expr::Const(bool_name, Vec::new());

    // body at depth 2 (m, n): blt n m.
    let blt_app = Expr::Const(nat_blt, Vec::new())
        .apps([Expr::Var(DeBruijnIndex(0)), Expr::Var(DeBruijnIndex(1))]);
    let body = Expr::Lam(
        BinderInfo::Default,
        m,
        Box::new(nat_ty.clone()),
        Box::new(Expr::Lam(
            BinderInfo::Default,
            n,
            Box::new(nat_ty.clone()),
            Box::new(blt_app),
        )),
    );
    let ty = Expr::Pi(
        BinderInfo::Default,
        m,
        Box::new(nat_ty.clone()),
        Box::new(Expr::Pi(
            BinderInfo::Default,
            n,
            Box::new(nat_ty.clone()),
            Box::new(bool_ty),
        )),
    );
    insert_op(env, nat_bgt, ty, body);
}

/// `Nat.bge : Nat -> Nat -> Bool := λ m. λ n. Nat.ble n m`.
fn register_nat_bge(names: &mut NameTable, env: &mut GlobalEnv) {
    let m = names.intern("m");
    let n = names.intern("n");
    let nat = names.intern("Nat");
    let bool_name = names.intern("Bool");
    let nat_ble = names.intern("Nat.ble");
    let nat_bge = names.intern("Nat.bge");
    let nat_ty = Expr::Const(nat, Vec::new());
    let bool_ty = Expr::Const(bool_name, Vec::new());

    // body at depth 2 (m, n): ble n m.
    let ble_app = Expr::Const(nat_ble, Vec::new())
        .apps([Expr::Var(DeBruijnIndex(0)), Expr::Var(DeBruijnIndex(1))]);
    let body = Expr::Lam(
        BinderInfo::Default,
        m,
        Box::new(nat_ty.clone()),
        Box::new(Expr::Lam(
            BinderInfo::Default,
            n,
            Box::new(nat_ty.clone()),
            Box::new(ble_app),
        )),
    );
    let ty = Expr::Pi(
        BinderInfo::Default,
        m,
        Box::new(nat_ty.clone()),
        Box::new(Expr::Pi(
            BinderInfo::Default,
            n,
            Box::new(nat_ty.clone()),
            Box::new(bool_ty),
        )),
    );
    insert_op(env, nat_bge, ty, body);
}

// ===========================================================================
// Eq
// ===========================================================================

/// Register `Eq`, `Eq.refl`, and `Eq.rec`.
///
/// ```text
/// Eq.{u}      : Π {A : Sort u}. A -> A -> Prop
/// Eq.refl.{u} : Π {A : Sort u} (a : A). Eq A a a
/// Eq.rec.{u, v} : Π {A : Sort u} {a : A}
///                   {motive : Π (b : A). Eq A a b -> Sort v}
///                   (refl_case : motive a (Eq.refl A a))
///                   (b : A) (h : Eq A a b).
///                   motive b h
/// ```
///
/// The family `Eq A a` has two uniform parameters (`A` and `a`); the index
/// is `b`. `Eq.refl` takes no fields beyond the parameters, so its rule
/// needs no field bindings: the right-hand side is just `refl_case`.
#[allow(clippy::too_many_arguments)]
fn register_eq(
    names: &mut NameTable,
    env: &mut GlobalEnv,
    eq: NameId,
    eq_refl: NameId,
    eq_rec: NameId,
    u: UniverseParamId,
    v: UniverseParamId,
) {
    let underscore = names.intern("_");
    let a_name = names.intern("A");
    let x_name = names.intern("x");
    let y_name = names.intern("y");
    let motive = names.intern("motive");
    let refl_case = names.intern("refl_case");
    let b_name = names.intern("b");
    let h_name = names.intern("h");

    // Eq : Π {A : Sort u}. Π (x : A). Π (y : A). Prop
    let eq_ty = Expr::Pi(
        BinderInfo::Implicit,
        a_name,
        Box::new(Expr::Sort(Level::param(u))),
        Box::new(Expr::Pi(
            BinderInfo::Default,
            x_name,
            Box::new(Expr::Var(DeBruijnIndex(0))),
            Box::new(Expr::Pi(
                BinderInfo::Default,
                y_name,
                Box::new(Expr::Var(DeBruijnIndex(1))),
                Box::new(Expr::prop()),
            )),
        )),
    );

    // Eq.refl : Π {A : Sort u} (a : A). Eq A a a
    //   at depth 2 [A, a]: A=Var(1), a=Var(0). Cod = Eq A a a.
    let refl_ty = Expr::Pi(
        BinderInfo::Implicit,
        a_name,
        Box::new(Expr::Sort(Level::param(u))),
        Box::new(Expr::Pi(
            BinderInfo::Default,
            x_name,
            Box::new(Expr::Var(DeBruijnIndex(0))),
            Box::new(Expr::Const(eq, vec![Level::param(u)]).apps([
                Expr::Var(DeBruijnIndex(1)),
                Expr::Var(DeBruijnIndex(0)),
                Expr::Var(DeBruijnIndex(0)),
            ])),
        )),
    );

    // Eq.rec : Π {A : Sort u} {a : A}
    //            {motive : Π (b : A). Eq A a b -> Sort v}
    //            (refl_case : motive a (Eq.refl A a))
    //            (b : A) (h : Eq A a b).
    //            motive b h
    //
    // motive_ty at depth 2 (A=Var(1), a=Var(0)):
    //   Π (b : Var(1)). Π (_ : Eq A a b). Sort v
    //   at depth 3 (b bound): Eq A a b
    //     = App(App(App(Const(Eq,[u]), Var(2)), Var(1)), Var(0))
    let motive_ty = Expr::Pi(
        BinderInfo::Default,
        b_name,
        Box::new(Expr::Var(DeBruijnIndex(1))),
        Box::new(Expr::Pi(
            BinderInfo::Default,
            underscore,
            Box::new(Expr::Const(eq, vec![Level::param(u)]).apps([
                Expr::Var(DeBruijnIndex(2)),
                Expr::Var(DeBruijnIndex(1)),
                Expr::Var(DeBruijnIndex(0)),
            ])),
            Box::new(Expr::Sort(Level::param(v))),
        )),
    );
    // refl_case_ty at depth 3 (A=Var(2), a=Var(1), motive=Var(0)):
    //   App(App(Var(0), Var(1)), App(App(Const(Eq.refl,[u]), Var(2)), Var(1)))
    let refl_case_ty = Expr::app(
        Expr::app(Expr::Var(DeBruijnIndex(0)), Expr::Var(DeBruijnIndex(1))),
        Expr::Const(eq_refl, vec![Level::param(u)])
            .apps([Expr::Var(DeBruijnIndex(2)), Expr::Var(DeBruijnIndex(1))]),
    );
    // b_ty at depth 4 [A, a, motive, refl_case]: A = Var(3).
    let b_ty = Expr::Var(DeBruijnIndex(3));
    // h_ty at depth 4: Eq A a b = App(App(App(Eq, Var(4)), Var(3)), Var(0)).
    let h_ty = Expr::Const(eq, vec![Level::param(u)]).apps([
        Expr::Var(DeBruijnIndex(4)),
        Expr::Var(DeBruijnIndex(3)),
        Expr::Var(DeBruijnIndex(0)),
    ]);
    // codomain at depth 6 [A, a, motive, refl_case, b, h]:
    //   motive=Var(3), b=Var(1), h=Var(0). motive b h.
    let eq_rec_cod = Expr::app(
        Expr::app(Expr::Var(DeBruijnIndex(3)), Expr::Var(DeBruijnIndex(1))),
        Expr::Var(DeBruijnIndex(0)),
    );
    let rec_ty = Expr::Pi(
        BinderInfo::Implicit,
        a_name,
        Box::new(Expr::Sort(Level::param(u))),
        Box::new(Expr::Pi(
            BinderInfo::Implicit,
            underscore,
            Box::new(Expr::Var(DeBruijnIndex(0))),
            Box::new(Expr::Pi(
                BinderInfo::Implicit,
                motive,
                Box::new(motive_ty),
                Box::new(Expr::Pi(
                    BinderInfo::Default,
                    refl_case,
                    Box::new(refl_case_ty),
                    Box::new(Expr::Pi(
                        BinderInfo::Default,
                        b_name,
                        Box::new(b_ty),
                        Box::new(Expr::Pi(
                            BinderInfo::Default,
                            h_name,
                            Box::new(h_ty),
                            Box::new(eq_rec_cod),
                        )),
                    )),
                )),
            )),
        )),
    );

    // Rule: context [A, a, motive, refl_case, b] (no fields).
    //   b=Var(0), refl_case=Var(1), motive=Var(2), a=Var(3), A=Var(4).
    //   RHS = refl_case.
    let rule_rhs = Expr::Var(DeBruijnIndex(1));

    // Inductive metadata.
    //   num_params = 2 (A and a are the uniform parameters of Eq A a).
    //   num_indices = 1 (b).
    let ind = InductiveVal {
        base: ConstantVal {
            name: eq,
            universe_params: vec![u],
            ty: Rc::new(eq_ty),
        },
        num_params: 2,
        num_indices: 1,
        all: vec![eq],
        constructors: vec![eq_refl],
        is_recursive: true,
        is_nested: false,
        is_unsafe: false,
    };
    let ctors = vec![ConstructorVal {
        base: ConstantVal {
            name: eq_refl,
            universe_params: vec![u],
            ty: Rc::new(refl_ty),
        },
        inductive: eq,
        index: 0,
        num_params: 2,
        num_fields: 0,
        is_unsafe: false,
    }];
    let recursor = RecursorVal {
        base: ConstantVal {
            name: eq_rec,
            universe_params: vec![u, v],
            ty: Rc::new(rec_ty),
        },
        all: vec![eq],
        num_params: 2,
        num_indices: 1,
        num_motives: 1,
        num_minors: 1,
        rules: vec![RecursorRule {
            constructor: eq_refl,
            num_fields: 0,
            rhs: Rc::new(rule_rhs),
        }],
        is_k: true,
        is_unsafe: false,
    };

    env.add_inductive(ind, ctors, recursor)
        .expect("Eq prelude failed to register");
}

// ===========================================================================
// Bool
// ===========================================================================

fn register_bool(
    names: &mut NameTable,
    env: &mut GlobalEnv,
    bool_name: NameId,
    bool_true: NameId,
    bool_false: NameId,
    bool_rec: NameId,
    u: UniverseParamId,
) {
    let underscore = names.intern("_");
    let motive = names.intern("motive");
    let true_case = names.intern("true_case");
    let false_case = names.intern("false_case");
    let b = names.intern("b");

    // Bool : Type 0
    let bool_ty = Expr::type0();
    // Bool.true : Bool
    let true_ty = Expr::Const(bool_name, vec![]);
    // Bool.false : Bool
    let false_ty = Expr::Const(bool_name, vec![]);

    // Bool.rec.{u} : {motive : Bool -> Sort u}
    //              -> motive Bool.true
    //              -> motive Bool.false
    //              -> (b : Bool) -> motive b
    //
    // depth 0:  motive_ty     = Π (_ : Bool). Sort u
    // depth 1:  true_case_ty  = motive Bool.true              (motive = Var(0))
    // depth 2:  false_case_ty = motive Bool.false             (motive = Var(1))
    // depth 3:  major_ty     = (b : Bool) -> motive b
    //   at depth 4 (motive=Var(3), b=Var(0)): motive b
    let motive_ty = Expr::Pi(
        BinderInfo::Default,
        underscore,
        Box::new(Expr::Const(bool_name, vec![])),
        Box::new(Expr::Sort(Level::param(u))),
    );
    let true_case_ty = Expr::app(Expr::Var(DeBruijnIndex(0)), Expr::Const(bool_true, vec![]));
    let false_case_ty = Expr::app(Expr::Var(DeBruijnIndex(1)), Expr::Const(bool_false, vec![]));
    let major_ty = Expr::Pi(
        BinderInfo::Default,
        b,
        Box::new(Expr::Const(bool_name, vec![])),
        Box::new(Expr::app(
            Expr::Var(DeBruijnIndex(3)),
            Expr::Var(DeBruijnIndex(0)),
        )),
    );
    let rec_ty = Expr::Pi(
        BinderInfo::Implicit,
        motive,
        Box::new(motive_ty),
        Box::new(Expr::Pi(
            BinderInfo::Default,
            true_case,
            Box::new(true_case_ty),
            Box::new(Expr::Pi(
                BinderInfo::Default,
                false_case,
                Box::new(false_case_ty),
                Box::new(major_ty),
            )),
        )),
    );

    // Rules. Context [motive, true_case, false_case].
    //   Var(0)=false_case, Var(1)=true_case, Var(2)=motive.
    let true_rule_rhs = Expr::Var(DeBruijnIndex(1));
    let false_rule_rhs = Expr::Var(DeBruijnIndex(0));

    let ind = InductiveVal {
        base: ConstantVal {
            name: bool_name,
            universe_params: Vec::new(),
            ty: Rc::new(bool_ty),
        },
        num_params: 0,
        num_indices: 0,
        all: vec![bool_name],
        constructors: vec![bool_true, bool_false],
        is_recursive: false,
        is_nested: false,
        is_unsafe: false,
    };
    let ctors = vec![
        ConstructorVal {
            base: ConstantVal {
                name: bool_true,
                universe_params: Vec::new(),
                ty: Rc::new(true_ty),
            },
            inductive: bool_name,
            index: 0,
            num_params: 0,
            num_fields: 0,
            is_unsafe: false,
        },
        ConstructorVal {
            base: ConstantVal {
                name: bool_false,
                universe_params: Vec::new(),
                ty: Rc::new(false_ty),
            },
            inductive: bool_name,
            index: 1,
            num_params: 0,
            num_fields: 0,
            is_unsafe: false,
        },
    ];
    let recursor = RecursorVal {
        base: ConstantVal {
            name: bool_rec,
            universe_params: vec![u],
            ty: Rc::new(rec_ty),
        },
        all: vec![bool_name],
        num_params: 0,
        num_indices: 0,
        num_motives: 1,
        num_minors: 2,
        rules: vec![
            RecursorRule {
                constructor: bool_true,
                num_fields: 0,
                rhs: Rc::new(true_rule_rhs),
            },
            RecursorRule {
                constructor: bool_false,
                num_fields: 0,
                rhs: Rc::new(false_rule_rhs),
            },
        ],
        is_k: false,
        is_unsafe: false,
    };

    env.add_inductive(ind, ctors, recursor)
        .expect("Bool prelude failed to register");
}

// ===========================================================================
// List
// ===========================================================================

#[allow(clippy::too_many_arguments)]
fn register_list(
    names: &mut NameTable,
    env: &mut GlobalEnv,
    list: NameId,
    list_nil: NameId,
    list_cons: NameId,
    list_rec: NameId,
    list_u: UniverseParamId,
    list_v: UniverseParamId,
) {
    let underscore = names.intern("_");
    let a_name = names.intern("A");
    let motive = names.intern("motive");
    let nil_case = names.intern("nil_case");
    let cons_case = names.intern("cons_case");
    let x_name = names.intern("x");
    let xs_name = names.intern("xs");
    let h = names.intern("h");
    let l = names.intern("l");

    // List.{u} : (A : Type u) -> Type u.
    //
    // The inductive itself is a family over its carrier parameter, so that
    // `List.{u} A` is a well-typed application (and `infer` accepts it).
    let list_ty = Expr::Pi(
        BinderInfo::Implicit,
        a_name,
        Box::new(Expr::Sort(Level::param(list_u))),
        Box::new(Expr::Sort(Level::param(list_u))),
    );

    // List.nil.{u} : {A : Type u} -> List.{u} A
    //   depth 0: Pi {A : Type u}. body
    //   depth 1 [A]: cod = App(Const(List, [u]), Var(0))
    let nil_ty = Expr::Pi(
        BinderInfo::Implicit,
        a_name,
        Box::new(Expr::Sort(Level::param(list_u))),
        Box::new(Expr::app(
            Expr::Const(list, vec![Level::param(list_u)]),
            Expr::Var(DeBruijnIndex(0)),
        )),
    );

    // List.cons.{u} : {A : Type u} -> A -> List.{u} A -> List.{u} A
    //   depth 1 [A]:     x : A             (dom = Var(0))
    //   depth 2 [A, x]:  xs : List.{u} A   (dom = App(List, Var(1)))
    //   depth 3 [A, x, xs]: cod = App(List, Var(2))
    let cons_ty = Expr::Pi(
        BinderInfo::Implicit,
        a_name,
        Box::new(Expr::Sort(Level::param(list_u))),
        Box::new(Expr::Pi(
            BinderInfo::Default,
            x_name,
            Box::new(Expr::Var(DeBruijnIndex(0))),
            Box::new(Expr::Pi(
                BinderInfo::Default,
                xs_name,
                Box::new(Expr::app(
                    Expr::Const(list, vec![Level::param(list_u)]),
                    Expr::Var(DeBruijnIndex(1)),
                )),
                Box::new(Expr::app(
                    Expr::Const(list, vec![Level::param(list_u)]),
                    Expr::Var(DeBruijnIndex(2)),
                )),
            )),
        )),
    );

    // Recursor.
    //   depth 0:  Pi {A : Type u}. body
    //   depth 1 [A]: Pi {motive : List.{u} A -> Sort v}. body
    //     motive_ty = Π (_ : App(List, Var(0))). Sort v
    //   depth 2 [A, motive]: Pi (_ : motive (List.nil.{u} A)). body
    //     nil_case_ty = App(Var(0), App(nil, Var(1)))       (motive=Var(0), A=Var(1))
    //   depth 3 [A, motive, nil_case]: Pi (_ : cons_case_ty). body
    //     cons_case_ty:
    //       Pi (x : Var(2)).
    //         Pi (xs : App(List, Var(3))).
    //           Pi (h : App(Var(3), Var(0))).
    //             App(Var(4), App(App(App(cons, Var(5)), Var(2)), Var(1)))
    //   depth 4 [A, motive, nil_case, cons_case]: major_ty
    //       Pi (l : App(List, Var(3))). App(Var(3), Var(0))
    //       (at depth 4: cons=0, nil=1, motive=2, A=3;
    //        body at depth 5: l=0, cons=1, nil=2, motive=3, A=4)
    let motive_ty = Expr::Pi(
        BinderInfo::Default,
        underscore,
        Box::new(Expr::app(
            Expr::Const(list, vec![Level::param(list_u)]),
            Expr::Var(DeBruijnIndex(0)),
        )),
        Box::new(Expr::Sort(Level::param(list_v))),
    );
    let nil_case_ty = Expr::app(
        Expr::Var(DeBruijnIndex(0)),
        Expr::app(
            Expr::Const(list_nil, vec![Level::param(list_u)]),
            Expr::Var(DeBruijnIndex(1)),
        ),
    );
    let cons_case_ty = Expr::Pi(
        BinderInfo::Default,
        x_name,
        Box::new(Expr::Var(DeBruijnIndex(2))),
        Box::new(Expr::Pi(
            BinderInfo::Default,
            xs_name,
            Box::new(Expr::app(
                Expr::Const(list, vec![Level::param(list_u)]),
                Expr::Var(DeBruijnIndex(3)),
            )),
            Box::new(Expr::Pi(
                BinderInfo::Default,
                h,
                Box::new(Expr::app(
                    Expr::Var(DeBruijnIndex(3)),
                    Expr::Var(DeBruijnIndex(0)),
                )),
                Box::new(Expr::app(
                    Expr::Var(DeBruijnIndex(4)),
                    Expr::Const(list_cons, vec![Level::param(list_u)]).apps([
                        Expr::Var(DeBruijnIndex(5)),
                        Expr::Var(DeBruijnIndex(2)),
                        Expr::Var(DeBruijnIndex(1)),
                    ]),
                )),
            )),
        )),
    );
    let major_ty = Expr::Pi(
        BinderInfo::Default,
        l,
        Box::new(Expr::app(
            Expr::Const(list, vec![Level::param(list_u)]),
            Expr::Var(DeBruijnIndex(3)),
        )),
        Box::new(Expr::app(
            Expr::Var(DeBruijnIndex(3)),
            Expr::Var(DeBruijnIndex(0)),
        )),
    );
    let rec_ty = Expr::Pi(
        BinderInfo::Implicit,
        a_name,
        Box::new(Expr::Sort(Level::param(list_u))),
        Box::new(Expr::Pi(
            BinderInfo::Implicit,
            motive,
            Box::new(motive_ty),
            Box::new(Expr::Pi(
                BinderInfo::Default,
                nil_case,
                Box::new(nil_case_ty),
                Box::new(Expr::Pi(
                    BinderInfo::Default,
                    cons_case,
                    Box::new(cons_case_ty),
                    Box::new(major_ty),
                )),
            )),
        )),
    );

    // Rules. Context [A, motive, nil_case, cons_case, ...fields].
    //   Var(0)=cons_case, Var(1)=nil_case, Var(2)=motive, Var(3)=A.
    let nil_rule_rhs = Expr::Var(DeBruijnIndex(1));

    // Cons rule. Context [A, motive, nil_case, cons_case, x, xs].
    //   Var(0)=xs, Var(1)=x, Var(2)=cons_case, Var(3)=nil_case,
    //   Var(4)=motive, Var(5)=A.
    //   RHS = cons_case x xs (List.rec.{u,v} A motive nil_case cons_case xs)
    let list_rec_const = Expr::Const(list_rec, vec![Level::param(list_u), Level::param(list_v)]);
    let recursive_call = list_rec_const.apps([
        Expr::Var(DeBruijnIndex(5)),
        Expr::Var(DeBruijnIndex(4)),
        Expr::Var(DeBruijnIndex(3)),
        Expr::Var(DeBruijnIndex(2)),
        Expr::Var(DeBruijnIndex(0)),
    ]);
    let cons_rule_rhs = Expr::Var(DeBruijnIndex(2)).apps([
        Expr::Var(DeBruijnIndex(1)),
        Expr::Var(DeBruijnIndex(0)),
        recursive_call,
    ]);

    let ind = InductiveVal {
        base: ConstantVal {
            name: list,
            universe_params: vec![list_u],
            ty: Rc::new(list_ty),
        },
        num_params: 1,
        num_indices: 0,
        all: vec![list],
        constructors: vec![list_nil, list_cons],
        is_recursive: true,
        is_nested: false,
        is_unsafe: false,
    };
    let ctors = vec![
        ConstructorVal {
            base: ConstantVal {
                name: list_nil,
                universe_params: vec![list_u],
                ty: Rc::new(nil_ty),
            },
            inductive: list,
            index: 0,
            num_params: 1,
            num_fields: 0,
            is_unsafe: false,
        },
        ConstructorVal {
            base: ConstantVal {
                name: list_cons,
                universe_params: vec![list_u],
                ty: Rc::new(cons_ty),
            },
            inductive: list,
            index: 1,
            num_params: 1,
            num_fields: 2,
            is_unsafe: false,
        },
    ];
    let recursor = RecursorVal {
        base: ConstantVal {
            name: list_rec,
            universe_params: vec![list_u, list_v],
            ty: Rc::new(rec_ty),
        },
        all: vec![list],
        num_params: 1,
        num_indices: 0,
        num_motives: 1,
        num_minors: 2,
        rules: vec![
            RecursorRule {
                constructor: list_nil,
                num_fields: 0,
                rhs: Rc::new(nil_rule_rhs),
            },
            RecursorRule {
                constructor: list_cons,
                num_fields: 2,
                rhs: Rc::new(cons_rule_rhs),
            },
        ],
        is_k: false,
        is_unsafe: false,
    };

    env.add_inductive(ind, ctors, recursor)
        .expect("List prelude failed to register");
}

// ===========================================================================
// Option
// ===========================================================================

#[allow(clippy::too_many_arguments)]
fn register_option(
    names: &mut NameTable,
    env: &mut GlobalEnv,
    option: NameId,
    option_none: NameId,
    option_some: NameId,
    option_rec: NameId,
    option_u: UniverseParamId,
    option_v: UniverseParamId,
) {
    let underscore = names.intern("_");
    let a_name = names.intern("A");
    let motive = names.intern("motive");
    let none_case = names.intern("none_case");
    let some_case = names.intern("some_case");
    let x_name = names.intern("x");
    let o = names.intern("o");

    // Option.{u} : (A : Type u) -> Type u, so that `Option.{u} A` is a
    // well-typed application.
    let option_ty = Expr::Pi(
        BinderInfo::Implicit,
        a_name,
        Box::new(Expr::Sort(Level::param(option_u))),
        Box::new(Expr::Sort(Level::param(option_u))),
    );

    // Option.none.{u} : {A : Type u} -> Option.{u} A
    let none_ty = Expr::Pi(
        BinderInfo::Implicit,
        a_name,
        Box::new(Expr::Sort(Level::param(option_u))),
        Box::new(Expr::app(
            Expr::Const(option, vec![Level::param(option_u)]),
            Expr::Var(DeBruijnIndex(0)),
        )),
    );

    // Option.some.{u} : {A : Type u} -> A -> Option.{u} A
    let some_ty = Expr::Pi(
        BinderInfo::Implicit,
        a_name,
        Box::new(Expr::Sort(Level::param(option_u))),
        Box::new(Expr::Pi(
            BinderInfo::Default,
            x_name,
            Box::new(Expr::Var(DeBruijnIndex(0))),
            Box::new(Expr::app(
                Expr::Const(option, vec![Level::param(option_u)]),
                Expr::Var(DeBruijnIndex(1)),
            )),
        )),
    );

    // Recursor.
    //   depth 0:  Pi {A : Type u}. body
    //   depth 1 [A]: Pi {motive : Option.{u} A -> Sort v}. body
    //     motive_ty = Π (_ : App(Option, Var(0))). Sort v
    //   depth 2 [A, motive]: Pi (_ : motive (Option.none.{u} A)). body
    //     none_case_ty = App(Var(0), App(none, Var(1)))
    //   depth 3 [A, motive, none_case]: Pi (_ : some_case_ty). body
    //     some_case_ty:
    //       Pi (x : Var(2)).
    //         App(Var(2), App(App(some, Var(3)), Var(0)))
    //       -- some_case_ty = (x : A) -> motive (Option.some.{u} A x)
    //       -- At depth 4: A=Var(3), motive=Var(2), x=Var(0).
    //   depth 4 [A, motive, none_case, some_case]: major_ty
    //       Pi (o : App(Option, Var(3))). App(Var(3), Var(0))
    let motive_ty = Expr::Pi(
        BinderInfo::Default,
        underscore,
        Box::new(Expr::app(
            Expr::Const(option, vec![Level::param(option_u)]),
            Expr::Var(DeBruijnIndex(0)),
        )),
        Box::new(Expr::Sort(Level::param(option_v))),
    );
    let none_case_ty = Expr::app(
        Expr::Var(DeBruijnIndex(0)),
        Expr::app(
            Expr::Const(option_none, vec![Level::param(option_u)]),
            Expr::Var(DeBruijnIndex(1)),
        ),
    );
    let some_case_ty = Expr::Pi(
        BinderInfo::Default,
        x_name,
        Box::new(Expr::Var(DeBruijnIndex(2))),
        Box::new(Expr::app(
            Expr::Var(DeBruijnIndex(2)),
            Expr::Const(option_some, vec![Level::param(option_u)])
                .apps([Expr::Var(DeBruijnIndex(3)), Expr::Var(DeBruijnIndex(0))]),
        )),
    );
    let major_ty = Expr::Pi(
        BinderInfo::Default,
        o,
        Box::new(Expr::app(
            Expr::Const(option, vec![Level::param(option_u)]),
            Expr::Var(DeBruijnIndex(3)),
        )),
        Box::new(Expr::app(
            Expr::Var(DeBruijnIndex(3)),
            Expr::Var(DeBruijnIndex(0)),
        )),
    );
    let rec_ty = Expr::Pi(
        BinderInfo::Implicit,
        a_name,
        Box::new(Expr::Sort(Level::param(option_u))),
        Box::new(Expr::Pi(
            BinderInfo::Implicit,
            motive,
            Box::new(motive_ty),
            Box::new(Expr::Pi(
                BinderInfo::Default,
                none_case,
                Box::new(none_case_ty),
                Box::new(Expr::Pi(
                    BinderInfo::Default,
                    some_case,
                    Box::new(some_case_ty),
                    Box::new(major_ty),
                )),
            )),
        )),
    );

    // Rules. Context [A, motive, none_case, some_case, ...fields].
    //   Var(0)=some_case, Var(1)=none_case, Var(2)=motive, Var(3)=A.
    let none_rule_rhs = Expr::Var(DeBruijnIndex(1));

    // Some rule. Context [A, motive, none_case, some_case, x].
    //   Var(0)=x, Var(1)=some_case, Var(2)=none_case, Var(3)=motive, Var(4)=A.
    //   RHS = some_case x
    let some_rule_rhs = Expr::app(Expr::Var(DeBruijnIndex(1)), Expr::Var(DeBruijnIndex(0)));

    let ind = InductiveVal {
        base: ConstantVal {
            name: option,
            universe_params: vec![option_u],
            ty: Rc::new(option_ty),
        },
        num_params: 1,
        num_indices: 0,
        all: vec![option],
        constructors: vec![option_none, option_some],
        is_recursive: false,
        is_nested: false,
        is_unsafe: false,
    };
    let ctors = vec![
        ConstructorVal {
            base: ConstantVal {
                name: option_none,
                universe_params: vec![option_u],
                ty: Rc::new(none_ty),
            },
            inductive: option,
            index: 0,
            num_params: 1,
            num_fields: 0,
            is_unsafe: false,
        },
        ConstructorVal {
            base: ConstantVal {
                name: option_some,
                universe_params: vec![option_u],
                ty: Rc::new(some_ty),
            },
            inductive: option,
            index: 1,
            num_params: 1,
            num_fields: 1,
            is_unsafe: false,
        },
    ];
    let recursor = RecursorVal {
        base: ConstantVal {
            name: option_rec,
            universe_params: vec![option_u, option_v],
            ty: Rc::new(rec_ty),
        },
        all: vec![option],
        num_params: 1,
        num_indices: 0,
        num_motives: 1,
        num_minors: 2,
        rules: vec![
            RecursorRule {
                constructor: option_none,
                num_fields: 0,
                rhs: Rc::new(none_rule_rhs),
            },
            RecursorRule {
                constructor: option_some,
                num_fields: 1,
                rhs: Rc::new(some_rule_rhs),
            },
        ],
        is_k: false,
        is_unsafe: false,
    };

    env.add_inductive(ind, ctors, recursor)
        .expect("Option prelude failed to register");
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infer::TypeChecker;
    use crate::nbe::{Env as NbeEnv, Nbe, TransparencyMode};

    /// `n` as a closed kernel `Nat` literal chain.
    fn num(p: &Prelude, n: u64) -> Expr {
        let mut e = Expr::Const(p.zero, Vec::new());
        for _ in 0..n {
            e = Expr::app(Expr::Const(p.succ, Vec::new()), e);
        }
        e
    }

    /// Fully-applied operator application.
    fn op2(p: &Prelude, f: NameId, a: u64, b: u64) -> Expr {
        Expr::Const(f, Vec::new()).apps([num(p, a), num(p, b)])
    }

    /// Evaluate a closed term to its βδι-normal form.
    fn normalise(p: &Prelude, e: &Expr) -> Expr {
        let nbe = Nbe::new(&p.env, TransparencyMode::Semireducible);
        let v = nbe.eval(&NbeEnv::empty(), e);
        nbe.quote(0, &v)
    }

    #[test]
    fn all_five_arith_ops_typecheck() {
        let p = build_prelude();
        let tc = TypeChecker::new(&p.env);
        for name in [p.pred, p.add, p.mul, p.sub, p.pow] {
            let info = p.env.find(name).expect("op missing");
            tc.check_declaration(&info)
                .unwrap_or_else(|e| panic!("{name} failed: {e}"));
        }
    }

    #[test]
    fn env_grows_by_five_plus_nine() {
        // 19 declarations from delivery 3 (Nat, Bool, List, Option, Eq),
        // 5 arithmetic operators from delivery 4, and 9 comparison
        // definitions from delivery 6.
        assert_eq!(build_prelude().env.len(), 33);
    }

    #[test]
    fn ops_have_no_universe_params() {
        // Guard against a future regression that universe-polymorphises
        // the operators by mistake.
        let p = build_prelude();
        for name in [p.pred, p.add, p.mul, p.sub, p.pow] {
            let info = p.env.find(name).unwrap();
            assert!(
                info.universe_params().is_empty(),
                "{name} must stay monomorphic"
            );
        }
    }

    #[test]
    fn pred_reduces() {
        let p = build_prelude();
        let pred_of = |n: u64| Expr::app(Expr::Const(p.pred, Vec::new()), num(&p, n));
        // pred (succ zero) = zero
        assert_eq!(normalise(&p, &pred_of(1)), num(&p, 0));
        // pred zero = zero
        assert_eq!(normalise(&p, &pred_of(0)), num(&p, 0));
    }

    #[test]
    fn add_reduces() {
        let p = build_prelude();
        // add 0 1 = 1
        assert_eq!(normalise(&p, &op2(&p, p.add, 0, 1)), num(&p, 1));
        // add 1 0 = 1
        assert_eq!(normalise(&p, &op2(&p, p.add, 1, 0)), num(&p, 1));
        // add 2 1 = 3
        assert_eq!(normalise(&p, &op2(&p, p.add, 2, 1)), num(&p, 3));
    }

    #[test]
    fn mul_reduces() {
        let p = build_prelude();
        // mul 0 2 = 0
        assert_eq!(normalise(&p, &op2(&p, p.mul, 0, 2)), num(&p, 0));
        // mul 1 2 = 2
        assert_eq!(normalise(&p, &op2(&p, p.mul, 1, 2)), num(&p, 2));
        // mul 2 2 = 4
        assert_eq!(normalise(&p, &op2(&p, p.mul, 2, 2)), num(&p, 4));
    }

    #[test]
    fn sub_reduces() {
        let p = build_prelude();
        // sub 2 0 = 2
        assert_eq!(normalise(&p, &op2(&p, p.sub, 2, 0)), num(&p, 2));
        // sub 2 1 = 1
        assert_eq!(normalise(&p, &op2(&p, p.sub, 2, 1)), num(&p, 1));
        // sub 0 1 = 0 (truncated)
        assert_eq!(normalise(&p, &op2(&p, p.sub, 0, 1)), num(&p, 0));
    }

    #[test]
    fn pow_reduces() {
        let p = build_prelude();
        // pow 2 0 = 1
        assert_eq!(normalise(&p, &op2(&p, p.pow, 2, 0)), num(&p, 1));
        // pow 2 1 = 2
        assert_eq!(normalise(&p, &op2(&p, p.pow, 2, 1)), num(&p, 2));
        // pow 2 2 = 4
        assert_eq!(normalise(&p, &op2(&p, p.pow, 2, 2)), num(&p, 4));
    }

    /// Evaluate a closed `Bool`-valued term and compare against the
    /// expected constructor.
    fn check_bool(p: &Prelude, e: &Expr, expected: bool) {
        let want = if expected { p.bool_true } else { p.bool_false };
        assert_eq!(normalise(p, e), Expr::Const(want, Vec::new()));
    }

    /// Fully-applied unary operator application.
    fn op1(f: NameId, a: Expr) -> Expr {
        Expr::app(Expr::Const(f, Vec::new()), a)
    }

    /// `n` as a closed `Bool` constant term.
    fn bool_const(p: &Prelude, b: bool) -> Expr {
        Expr::Const(if b { p.bool_true } else { p.bool_false }, Vec::new())
    }

    #[test]
    fn all_nine_cmp_ops_typecheck() {
        let p = build_prelude();
        let tc = TypeChecker::new(&p.env);
        for name in [
            p.bool_not,
            p.nat_is_zero,
            p.nat_beq_step,
            p.nat_beq,
            p.nat_bne,
            p.nat_ble,
            p.nat_blt,
            p.nat_bgt,
            p.nat_bge,
        ] {
            let info = p.env.find(name).expect("op missing");
            tc.check_declaration(&info)
                .unwrap_or_else(|e| panic!("{name} failed: {e}"));
        }
    }

    #[test]
    fn cmp_ops_have_no_universe_params() {
        // Guard against a future regression that universe-polymorphises
        // the comparison operators by mistake.
        let p = build_prelude();
        for name in [
            p.bool_not,
            p.nat_is_zero,
            p.nat_beq_step,
            p.nat_beq,
            p.nat_bne,
            p.nat_ble,
            p.nat_blt,
            p.nat_bgt,
            p.nat_bge,
        ] {
            let info = p.env.find(name).unwrap();
            assert!(
                info.universe_params().is_empty(),
                "{name} must stay monomorphic"
            );
        }
    }

    #[test]
    fn bool_not_reduces() {
        let p = build_prelude();
        check_bool(&p, &op1(p.bool_not, bool_const(&p, true)), false);
        check_bool(&p, &op1(p.bool_not, bool_const(&p, false)), true);
    }

    #[test]
    fn nat_is_zero_reduces() {
        let p = build_prelude();
        let is_zero_of = |n: u64| op1(p.nat_is_zero, num(&p, n));
        check_bool(&p, &is_zero_of(0), true);
        check_bool(&p, &is_zero_of(1), false);
        check_bool(&p, &is_zero_of(5), false);
    }

    #[test]
    fn nat_beq_reduces() {
        let p = build_prelude();
        let beq = |a: u64, b: u64| op2(&p, p.nat_beq, a, b);
        check_bool(&p, &beq(0, 0), true);
        check_bool(&p, &beq(0, 1), false);
        check_bool(&p, &beq(1, 0), false);
        check_bool(&p, &beq(3, 3), true);
        check_bool(&p, &beq(2, 3), false);
    }

    #[test]
    fn nat_bne_reduces() {
        let p = build_prelude();
        let bne = |a: u64, b: u64| op2(&p, p.nat_bne, a, b);
        check_bool(&p, &bne(0, 0), false);
        check_bool(&p, &bne(0, 1), true);
    }

    #[test]
    fn nat_ble_reduces() {
        let p = build_prelude();
        let ble = |a: u64, b: u64| op2(&p, p.nat_ble, a, b);
        check_bool(&p, &ble(0, 5), true);
        check_bool(&p, &ble(3, 3), true);
        check_bool(&p, &ble(5, 3), false);
    }

    #[test]
    fn nat_blt_reduces() {
        let p = build_prelude();
        let blt = |a: u64, b: u64| op2(&p, p.nat_blt, a, b);
        check_bool(&p, &blt(0, 1), true);
        check_bool(&p, &blt(2, 5), true);
        check_bool(&p, &blt(3, 3), false);
        check_bool(&p, &blt(5, 2), false);
    }

    #[test]
    fn nat_bgt_reduces() {
        let p = build_prelude();
        let bgt = |a: u64, b: u64| op2(&p, p.nat_bgt, a, b);
        check_bool(&p, &bgt(5, 2), true);
        check_bool(&p, &bgt(3, 3), false);
    }

    #[test]
    fn nat_bge_reduces() {
        let p = build_prelude();
        let bge = |a: u64, b: u64| op2(&p, p.nat_bge, a, b);
        check_bool(&p, &bge(5, 5), true);
        check_bool(&p, &bge(2, 5), false);
    }
}
