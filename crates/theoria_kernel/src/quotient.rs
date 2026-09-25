//! Built-in predicative quotient types.
//!
//! This module registers the four quotient primitives in a
//! [`GlobalEnv`]:
//!
//! ```text
//! Quot.{u}      : Π {A : Sort u}. (A → A → Prop) → Sort u
//! Quot.mk.{u}   : Π {A : Sort u} (r : A → A → Prop) (a : A). Quot A r
//! Quot.lift.{u, v} : Π {A : Sort u} {r : A → A → Prop} {B : Sort v}
//!                      (f : A → B)
//!                      (h : Π (a b : A). r a b → Eq.{v} B (f a) (f b))
//!                      (q : Quot A r). B
//! Quot.sound.{u} : Π {A : Sort u} {r : A → A → Prop} {a b : A}
//!                    (h : r a b).
//!                    Eq.{u} (Quot A r) (Quot.mk A r a) (Quot.mk A r b)
//! ```
//!
//! ## Why quotients are axiomatic
//!
//! In the Calculus of Inductive Constructions, a `Quot` type former
//! defined as an ordinary inductive would collapse the universe hierarchy:
//! the *quotient axiom*
//!
//! ```text
//! r a b  →  Quot.mk r a = Quot.mk r b
//! ```
//!
//! is inconsistent with an impredicative `Prop`. The standard escape
//! (used here) is to place `Quot` **predicatively**: `A : Sort u` and
//! `r : A → A → Prop` yield `Quot r : Sort u`, so no universe is
//! collapsed. The four primitives are therefore axioms, not inductives,
//! and the positivity gate does not apply to them.
//!
//! ## Reduction
//!
//! `Quot.lift` carries a β-rule:
//!
//! ```text
//! Quot.lift f h (Quot.mk r a) ≡ f a
//! ```
//!
//! which the NbE engine fires in `Nbe::try_quot_lift` after all six
//! arguments have been supplied. `Quot.mk` is registered as a
//! constructor-like constant so that it appears in values as a
//! `Head::Constructor`, which is what the elimination rule matches on.
//!
//! ## Interaction with the environment
//!
//! Quotients are registered via [`GlobalEnv::add`] (not
//! `add_inductive`), with each constant wrapped in
//! [`ConstantInfo::Quotient`]. The environment projects the quotient
//! metadata into a [`ConstDecl`](crate::nbe::ConstDecl) on demand. The
//! environment's termination gate does not fire because `add`
//! only checks [`ConstantInfo::Definition`]s.

use crate::env::{ConstantInfo, ConstantVal, GlobalEnv, QuotientKind, QuotientVal};
use crate::expr::{BinderInfo, DeBruijnIndex, Expr};
use crate::level::{Level, UniverseParamId};
use crate::name::{NameId, NameTable};
use alloc::boxed::Box;
use alloc::rc::Rc;
use alloc::vec;
use alloc::vec::Vec;

/// Names and identifiers for the quotient system registered by
/// [`register_quotient`].
pub struct QuotientPrelude {
    /// `Quot`'s identifier.
    pub quot: NameId,
    /// `Quot.mk`'s identifier.
    pub mk: NameId,
    /// `Quot.lift`'s identifier.
    pub lift: NameId,
    /// `Quot.sound`'s identifier.
    pub sound: NameId,
    /// `Quot`'s universe parameter (shared with `Quot.mk` and
    /// `Quot.sound`).
    pub quot_u: UniverseParamId,
    /// `Quot.lift`'s codomain universe parameter.
    pub lift_v: UniverseParamId,
}

/// Register the four quotient primitives in `env`.
///
/// `eq_name` must name an `Eq`-like inductive already present in `env`
/// with the standard signature:
///
/// ```text
/// Eq : Π {A : Sort u}. A → A → Prop
/// ```
///
/// The prelude's `Eq` (see [`crate::prelude`]) satisfies this requirement.
///
/// # Panics
///
/// Panics if any of the four constants cannot be inserted, which can only
/// happen if the kernel is broken or if `eq_name` duplicates an existing
/// declaration. This mirrors the prelude's own build-time checks.
#[must_use]
pub fn register_quotient(
    env: &mut GlobalEnv,
    names: &mut NameTable,
    eq_name: NameId,
) -> QuotientPrelude {
    let quot = names.intern("Quot");
    let mk = names.intern("Quot.mk");
    let lift = names.intern("Quot.lift");
    let sound = names.intern("Quot.sound");

    let quot_u = UniverseParamId::fresh();
    let lift_v = UniverseParamId::fresh();

    let a_name = names.intern("A");
    let r_name = names.intern("r");
    let b_name = names.intern("B");
    let underscore = names.intern("_");
    let f_name = names.intern("f");
    let h_name = names.intern("h");
    let q_name = names.intern("q");
    let x_name = names.intern("x");
    let y_name = names.intern("y");

    // --- Quot.{u} : Π {A : Sort u}. (A → A → Prop) → Sort u ------------
    //
    // At depth 1 (A = Var(0)):
    //   domain of r = Π (_ : Var(0)). Π (_ : Var(1)). Prop
    // At depth 2 (A = Var(1), r = Var(0)):
    //   codomain = Sort u.
    let quot_ty = Expr::Pi(
        BinderInfo::Implicit,
        a_name,
        Box::new(Expr::Sort(Level::param(quot_u))),
        Box::new(Expr::Pi(
            BinderInfo::Default,
            r_name,
            Box::new(Expr::Pi(
                BinderInfo::Default,
                underscore,
                Box::new(Expr::Var(DeBruijnIndex(0))),
                Box::new(Expr::Pi(
                    BinderInfo::Default,
                    underscore,
                    Box::new(Expr::Var(DeBruijnIndex(1))),
                    Box::new(Expr::prop()),
                )),
            )),
            Box::new(Expr::Sort(Level::param(quot_u))),
        )),
    );

    // --- Quot.mk.{u} : Π {A : Sort u} (r : A → A → Prop) (a : A).
    //                    Quot.{u} A r
    //
    // At depth 1 (A = Var(0)): domain of r as above.
    // At depth 2 (A = Var(1), r = Var(0)): domain of a = Var(1).
    // At depth 3 (A = Var(2), r = Var(1), a = Var(0)):
    //   codomain = App(App(Const(Quot, [u]), Var(2)), Var(1)).
    let mk_ty = Expr::Pi(
        BinderInfo::Implicit,
        a_name,
        Box::new(Expr::Sort(Level::param(quot_u))),
        Box::new(Expr::Pi(
            BinderInfo::Default,
            r_name,
            Box::new(Expr::Pi(
                BinderInfo::Default,
                underscore,
                Box::new(Expr::Var(DeBruijnIndex(0))),
                Box::new(Expr::Pi(
                    BinderInfo::Default,
                    underscore,
                    Box::new(Expr::Var(DeBruijnIndex(1))),
                    Box::new(Expr::prop()),
                )),
            )),
            Box::new(Expr::Pi(
                BinderInfo::Default,
                x_name,
                Box::new(Expr::Var(DeBruijnIndex(1))),
                Box::new(
                    Expr::Const(quot, vec![Level::param(quot_u)])
                        .apps([Expr::Var(DeBruijnIndex(2)), Expr::Var(DeBruijnIndex(1))]),
                ),
            )),
        )),
    );

    // --- Quot.lift.{u, v} ---------------------------------------------
    //
    // Π {A : Sort u} {r : A → A → Prop} {B : Sort v}
    //     (f : A → B)
    //     (h : Π (a b : A). r a b → Eq.{v} B (f a) (f b))
    //     (q : Quot.{u} A r). B
    //
    // Binder depth map (outermost to innermost):
    //   A at 0 (Var(0) at depth 1)
    //   r at 1 (Var(0) at depth 2, A = Var(1))
    //   B at 2 (Var(0) at depth 3, r = Var(1), A = Var(2))
    //   f at 3 (Var(0) at depth 4, B = Var(1), r = Var(2), A = Var(3))
    //   h at 4 (Var(0) at depth 5, f = Var(1), B = Var(2), r = Var(3), A = Var(4))
    //   q at 5 (Var(0) at depth 6, h = Var(1), f = Var(2), B = Var(3), r = Var(4), A = Var(5))
    //
    // Domain of f at depth 3: Π (_ : Var(2)). Var(1).
    // (The codomain is `B`, which sits at index 1 once the anonymous
    // domain binder is pushed.)
    let lift_f_dom = Expr::Pi(
        BinderInfo::Default,
        underscore,
        Box::new(Expr::Var(DeBruijnIndex(2))),
        Box::new(Expr::Var(DeBruijnIndex(1))),
    );
    // Domain of h at depth 4:
    //   Π (a : Var(3)). Π (b : Var(4)). Π (_ : App(App(Var(4), Var(1)), Var(0))).
    //     App(App(App(Const(Eq, [v]), Var(4)), App(Var(3), Var(2))), App(Var(3), Var(1)))
    let h_dom = Expr::Pi(
        BinderInfo::Default,
        a_name,
        Box::new(Expr::Var(DeBruijnIndex(3))),
        Box::new(Expr::Pi(
            BinderInfo::Default,
            y_name,
            Box::new(Expr::Var(DeBruijnIndex(4))),
            Box::new(Expr::Pi(
                BinderInfo::Default,
                underscore,
                Box::new(
                    Expr::Var(DeBruijnIndex(4))
                        .apps([Expr::Var(DeBruijnIndex(1)), Expr::Var(DeBruijnIndex(0))]),
                ),
                Box::new(Expr::Const(eq_name, vec![Level::param(lift_v)]).apps([
                    Expr::Var(DeBruijnIndex(4)),
                    Expr::app(Expr::Var(DeBruijnIndex(3)), Expr::Var(DeBruijnIndex(2))),
                    Expr::app(Expr::Var(DeBruijnIndex(3)), Expr::Var(DeBruijnIndex(1))),
                ])),
            )),
        )),
    );
    // Domain of q at depth 5: App(App(Const(Quot, [u]), Var(4)), Var(3)).
    let q_dom = Expr::Const(quot, vec![Level::param(quot_u)])
        .apps([Expr::Var(DeBruijnIndex(4)), Expr::Var(DeBruijnIndex(3))]);
    // Codomain at depth 6: B = Var(3).
    let lift_cod = Expr::Var(DeBruijnIndex(3));

    let lift_ty = Expr::Pi(
        BinderInfo::Implicit,
        a_name,
        Box::new(Expr::Sort(Level::param(quot_u))),
        Box::new(Expr::Pi(
            BinderInfo::Implicit,
            r_name,
            Box::new(Expr::Pi(
                BinderInfo::Default,
                underscore,
                Box::new(Expr::Var(DeBruijnIndex(0))),
                Box::new(Expr::Pi(
                    BinderInfo::Default,
                    underscore,
                    Box::new(Expr::Var(DeBruijnIndex(1))),
                    Box::new(Expr::prop()),
                )),
            )),
            Box::new(Expr::Pi(
                BinderInfo::Implicit,
                b_name,
                Box::new(Expr::Sort(Level::param(lift_v))),
                Box::new(Expr::Pi(
                    BinderInfo::Default,
                    f_name,
                    Box::new(lift_f_dom),
                    Box::new(Expr::Pi(
                        BinderInfo::Default,
                        h_name,
                        Box::new(h_dom),
                        Box::new(Expr::Pi(
                            BinderInfo::Default,
                            q_name,
                            Box::new(q_dom),
                            Box::new(lift_cod),
                        )),
                    )),
                )),
            )),
        )),
    );

    // --- Quot.sound.{u} ----------------------------------------------
    //
    // Π {A : Sort u} {r : A → A → Prop} {a b : A}
    //     (h : r a b).
    //     Eq.{u} (Quot.{u} A r) (Quot.mk.{u} A r a) (Quot.mk.{u} A r b)
    //
    // Depth 5 (A=Var(4), r=Var(3), a=Var(2), b=Var(1), h=Var(0)):
    //   Eq (Quot A r) (Quot.mk A r a) (Quot.mk A r b)
    let quot_of_a_r = |a_lvl: u32, r_lvl: u32| {
        Expr::Const(quot, vec![Level::param(quot_u)]).apps([
            Expr::Var(DeBruijnIndex(a_lvl)),
            Expr::Var(DeBruijnIndex(r_lvl)),
        ])
    };
    let mk_of = |a_lvl: u32, r_lvl: u32, x_lvl: u32| {
        Expr::Const(mk, vec![Level::param(quot_u)]).apps([
            Expr::Var(DeBruijnIndex(a_lvl)),
            Expr::Var(DeBruijnIndex(r_lvl)),
            Expr::Var(DeBruijnIndex(x_lvl)),
        ])
    };
    let sound_cod = Expr::Const(eq_name, vec![Level::param(quot_u)]).apps([
        quot_of_a_r(4, 3),
        mk_of(4, 3, 2),
        mk_of(4, 3, 1),
    ]);
    // h : r a b. At depth 4 (A=Var(3), r=Var(2), a=Var(1), b=Var(0)):
    //   App(App(Var(2), Var(1)), Var(0)).
    let sound_h_dom = Expr::Var(DeBruijnIndex(2))
        .apps([Expr::Var(DeBruijnIndex(1)), Expr::Var(DeBruijnIndex(0))]);
    let sound_ty = Expr::Pi(
        BinderInfo::Implicit,
        a_name,
        Box::new(Expr::Sort(Level::param(quot_u))),
        Box::new(Expr::Pi(
            BinderInfo::Implicit,
            r_name,
            Box::new(Expr::Pi(
                BinderInfo::Default,
                underscore,
                Box::new(Expr::Var(DeBruijnIndex(0))),
                Box::new(Expr::Pi(
                    BinderInfo::Default,
                    underscore,
                    Box::new(Expr::Var(DeBruijnIndex(1))),
                    Box::new(Expr::prop()),
                )),
            )),
            Box::new(Expr::Pi(
                BinderInfo::Implicit,
                x_name,
                Box::new(Expr::Var(DeBruijnIndex(1))),
                Box::new(Expr::Pi(
                    BinderInfo::Implicit,
                    y_name,
                    Box::new(Expr::Var(DeBruijnIndex(2))),
                    Box::new(Expr::Pi(
                        BinderInfo::Default,
                        h_name,
                        Box::new(sound_h_dom),
                        Box::new(sound_cod),
                    )),
                )),
            )),
        )),
    );

    // --- Register the four constants. --------------------------------
    let insert = |env: &mut GlobalEnv,
                  name: NameId,
                  universe_params: Vec<UniverseParamId>,
                  ty: Expr,
                  kind: QuotientKind| {
        env.add(ConstantInfo::Quotient(QuotientVal {
            base: ConstantVal {
                name,
                universe_params,
                ty: Rc::new(ty),
            },
            kind,
        }))
        .expect("quotient registration failed");
    };

    insert(env, quot, vec![quot_u], quot_ty, QuotientKind::Type);
    insert(env, mk, vec![quot_u], mk_ty, QuotientKind::Mk);
    insert(
        env,
        lift,
        vec![quot_u, lift_v],
        lift_ty,
        QuotientKind::Lift {
            mk,
            arity: 6,
            func_index: 3,
            major_index: 5,
        },
    );
    insert(env, sound, vec![quot_u], sound_ty, QuotientKind::Sound);

    QuotientPrelude {
        quot,
        mk,
        lift,
        sound,
        quot_u,
        lift_v,
    }
}
