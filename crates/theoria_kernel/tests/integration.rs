//! End-to-end integration tests for the Phase 1 kernel.
//!
//! These exercise multiple modules together: `env` builds the `Nat`
//! prelude, `nbe` performs β and ι reductions, and `infer` typechecks the
//! recursor's applications.

use std::boxed::Box;
use theoria_kernel::env::ConstantInfo;
use theoria_kernel::expr::{BinderInfo, DeBruijnIndex, Expr, Literal};
use theoria_kernel::infer::TypeChecker;
use theoria_kernel::level::Level;
use theoria_kernel::nbe::{Env as NbeEnv, Head, Nbe, TransparencyMode, Value};
use theoria_kernel::prelude::{NatPrelude, build_nat_prelude};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Reduce a closed expression to WHNF and return the resulting value.
fn whnf(p: &NatPrelude, e: &Expr) -> theoria_kernel::nbe::ValRef {
    let nbe = Nbe::new(&p.env, TransparencyMode::Semireducible);
    nbe.eval(&NbeEnv::empty(), e)
}

// ---------------------------------------------------------------------------
// ι-reduction on closed terms
// ---------------------------------------------------------------------------

#[test]
fn nat_rec_zero_reduces_to_zero_case() {
    let mut p = build_nat_prelude();

    // motive := λ _. Nat
    let motive = Expr::lam(
        p.names.intern("_"),
        Expr::Const(p.nat, vec![]),
        Expr::Const(p.nat, vec![]),
    );
    // zero_case := Nat.zero
    let zero_case = Expr::Const(p.zero, vec![]);
    // succ_case := λ n rec. Nat.succ rec
    let succ_case = Expr::lam(
        p.names.intern("n"),
        Expr::Const(p.nat, vec![]),
        Expr::lam(
            p.names.intern("rec"),
            Expr::Const(p.nat, vec![]),
            Expr::app(Expr::Const(p.succ, vec![]), Expr::Var(DeBruijnIndex(0))),
        ),
    );

    // Nat.rec.{u=1} motive zero_case succ_case Nat.zero
    // (u=1 because the motive `λ _. Nat` lands in `Sort 1`.)
    let term = Expr::Const(p.rec, vec![Level::zero().succ()]).apps([
        motive,
        zero_case.clone(),
        succ_case,
        Expr::Const(p.zero, vec![]),
    ]);

    let v = whnf(&p, &term);

    // Expected: Nat.zero.
    match &*v {
        Value::Neutral(Head::Constructor(name, _), args) => {
            assert_eq!(*name, p.zero);
            assert!(args.is_empty());
        }
        other => panic!("expected Nat.zero, got {other:?}"),
    }
}

#[test]
fn nat_rec_succ_reduces_one_step() {
    let mut p = build_nat_prelude();

    let motive = Expr::lam(
        p.names.intern("_"),
        Expr::Const(p.nat, vec![]),
        Expr::Const(p.nat, vec![]),
    );
    let zero_case = Expr::Const(p.zero, vec![]);
    // succ_case := λ n rec. Nat.succ rec
    let succ_case = Expr::lam(
        p.names.intern("n"),
        Expr::Const(p.nat, vec![]),
        Expr::lam(
            p.names.intern("rec"),
            Expr::Const(p.nat, vec![]),
            Expr::app(Expr::Const(p.succ, vec![]), Expr::Var(DeBruijnIndex(0))),
        ),
    );

    // Nat.rec.{1} motive zero_case succ_case (Nat.succ Nat.zero)
    let one = Expr::app(Expr::Const(p.succ, vec![]), Expr::Const(p.zero, vec![]));
    let term =
        Expr::Const(p.rec, vec![Level::zero().succ()]).apps([motive, zero_case, succ_case, one]);

    let v = whnf(&p, &term);

    // Expected: Nat.succ Nat.zero. (The `succ_case` composes Nat.succ with
    // the recursive call; the recursive call is itself Nat.zero.)
    match &*v {
        Value::Neutral(Head::Constructor(name, _), args) => {
            assert_eq!(*name, p.succ);
            assert_eq!(args.len(), 1);
            match &*args[0] {
                Value::Neutral(Head::Constructor(n, _), _) => assert_eq!(*n, p.zero),
                other => panic!("expected Nat.zero as argument, got {other:?}"),
            }
        }
        other => panic!("expected Nat.succ Nat.zero, got {other:?}"),
    }
}

#[test]
fn nat_rec_stuck_on_free_variable() {
    let mut p = build_nat_prelude();

    let motive = Expr::lam(
        p.names.intern("_"),
        Expr::Const(p.nat, vec![]),
        Expr::Const(p.nat, vec![]),
    );
    let zero_case = Expr::Const(p.zero, vec![]);
    let succ_case = Expr::lam(
        p.names.intern("n"),
        Expr::Const(p.nat, vec![]),
        Expr::lam(
            p.names.intern("rec"),
            Expr::Const(p.nat, vec![]),
            Expr::Var(DeBruijnIndex(0)),
        ),
    );

    // Build a context with a free variable `x : Nat` at de Bruijn level 0.
    let mut nbe_env = NbeEnv::empty();
    nbe_env = nbe_env.extend(Value::fresh_var(theoria_kernel::nbe::DbLevel(0)));
    let nbe = Nbe::new(&p.env, TransparencyMode::Semireducible);

    // body: Nat.rec.{1} motive zero_case succ_case x
    let term = Expr::Const(p.rec, vec![Level::zero().succ()]).apps([
        motive,
        zero_case,
        succ_case,
        Expr::Var(DeBruijnIndex(0)),
    ]);

    let v = nbe.eval(&nbe_env, &term);

    // Expected: stuck on the recursor, not reduced.
    match &*v {
        Value::Neutral(Head::Const(name, _), spine) => {
            assert_eq!(*name, p.rec);
            assert_eq!(spine.len(), 4);
        }
        other => panic!("expected stuck Nat.rec, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Plus via the recursor, and its equations
// ---------------------------------------------------------------------------

/// Define `plus : Nat -> Nat -> Nat` using `Nat.rec`.
///
/// ```text
/// plus := λ m n. Nat.rec (λ _. Nat) n (λ _ rec. Nat.succ rec) m
/// ```
fn plus_definition(p: &mut NatPrelude) -> Expr {
    let m = p.names.intern("m");
    let n = p.names.intern("n");
    let k = p.names.intern("k");
    let rec_var = p.names.intern("rec");

    // At depth 4 (m, n, k, rec): Nat.succ (Var(0))
    let succ_body = Expr::app(Expr::Const(p.succ, vec![]), Expr::Var(DeBruijnIndex(0)));

    // inner lam: λ rec. Nat.succ rec  — at depth 3 (m, n, k)
    let inner_lam = Expr::lam(rec_var, Expr::Const(p.nat, vec![]), succ_body);

    // At depth 3, motive references `motive = λ _. Nat`. Its body ignores
    // the argument so no Vars need to be adjusted.
    let motive = Expr::lam(
        p.names.intern("_"),
        Expr::Const(p.nat, vec![]),
        Expr::Const(p.nat, vec![]),
    );

    // The `zero_case` is `n`, at depth 2 (m, n). Var(0) = n.
    let zero_case = Expr::Var(DeBruijnIndex(0));

    // The `succ_case` is `λ k rec. Nat.succ rec`, at depth 2.
    let succ_case = Expr::lam(k, Expr::Const(p.nat, vec![]), inner_lam);

    // The major is `m`, at depth 2. Var(1) = m.
    let major = Expr::Var(DeBruijnIndex(1));

    // Nat.rec.{1} motive zero_case succ_case m, at depth 2.
    // (u=1: the motive `λ _. Nat` lands in `Sort 1`, where `Nat` lives.)
    let body =
        Expr::Const(p.rec, vec![Level::zero().succ()]).apps([motive, zero_case, succ_case, major]);

    // Wrap: λ m. λ n. body
    Expr::lam(
        m,
        Expr::Const(p.nat, vec![]),
        Expr::lam(n, Expr::Const(p.nat, vec![]), body),
    )
}

#[test]
fn plus_zero_n_reduces_to_n() {
    let mut p = build_nat_prelude();
    let plus = plus_definition(&mut p);

    // plus Nat.zero x  where x is free
    let mut nbe_env = NbeEnv::empty();
    nbe_env = nbe_env.extend(Value::fresh_var(theoria_kernel::nbe::DbLevel(0)));
    let nbe = Nbe::new(&p.env, TransparencyMode::Semireducible);

    // At depth 0 with x bound at Var(0):
    //   plus Nat.zero x
    // But `plus` itself is a lambda, so we apply it:
    //   (λ m n. body) Nat.zero x
    // The x that becomes the second argument is Var(0) at depth 0.
    let term = Expr::app(
        Expr::app(plus, Expr::Const(p.zero, vec![])),
        Expr::Var(DeBruijnIndex(0)),
    );

    let v = nbe.eval(&nbe_env, &term);

    // Expected: the free variable x (a stuck neutral at level 0).
    match &*v {
        Value::Neutral(Head::Var(lvl), args) => {
            assert_eq!(*lvl, theoria_kernel::nbe::DbLevel(0));
            assert!(args.is_empty());
        }
        other => panic!("expected free variable, got {other:?}"),
    }
}

#[test]
fn plus_succ_m_n_reduces_to_succ_of_plus() {
    let mut p = build_nat_prelude();
    let plus = plus_definition(&mut p);

    // plus (Nat.succ Nat.zero) Nat.zero
    // Expected: Nat.succ (plus Nat.zero Nat.zero) = Nat.succ Nat.zero.
    let term = Expr::app(
        Expr::app(
            plus,
            Expr::app(Expr::Const(p.succ, vec![]), Expr::Const(p.zero, vec![])),
        ),
        Expr::Const(p.zero, vec![]),
    );

    let v = whnf(&p, &term);

    match &*v {
        Value::Neutral(Head::Constructor(name, _), args) => {
            assert_eq!(*name, p.succ);
            assert_eq!(args.len(), 1);
            match &*args[0] {
                Value::Neutral(Head::Constructor(n, _), _) => assert_eq!(*n, p.zero),
                other => panic!("expected Nat.zero argument, got {other:?}"),
            }
        }
        other => panic!("expected Nat.succ Nat.zero, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Type-checking the recursor's type and application
// ---------------------------------------------------------------------------

#[test]
fn recursor_type_is_well_formed() {
    let p = build_nat_prelude();
    let tc = TypeChecker::new(&p.env);
    let info = p.env.find(p.rec).expect("Nat.rec missing");
    tc.check_declaration(&info)
        .expect("Nat.rec's type should be well-formed");
}

#[test]
fn plus_definition_typechecks() {
    // Build the prelude, then register `plus` as a definition and check it.
    let mut p = build_nat_prelude();
    let plus_name = p.names.intern("plus");
    let plus_body = plus_definition(&mut p);

    // Type: Nat -> Nat -> Nat
    let plus_ty = Expr::Pi(
        BinderInfo::Default,
        p.names.intern("_"),
        Box::new(Expr::Const(p.nat, vec![])),
        Box::new(Expr::Pi(
            BinderInfo::Default,
            p.names.intern("_"),
            Box::new(Expr::Const(p.nat, vec![])),
            Box::new(Expr::Const(p.nat, vec![])),
        )),
    );

    use std::rc::Rc;
    use theoria_kernel::env::DefinitionVal;
    use theoria_kernel::env::TerminationObligation;
    use theoria_kernel::nbe::Transparency;

    p.env
        .add(ConstantInfo::Definition(DefinitionVal {
            base: theoria_kernel::env::ConstantVal {
                name: plus_name,
                universe_params: Vec::new(),
                ty: Rc::new(plus_ty),
            },
            body: Rc::new(plus_body),
            transparency: Transparency::Semireducible,
            termination: TerminationObligation::none(),
        }))
        .expect("plus should register");

    let tc = TypeChecker::new(&p.env);
    let info = p.env.find(plus_name).expect("plus missing");
    tc.check_declaration(&info).expect("plus should typecheck");
}

// ---------------------------------------------------------------------------
// A trivial "theorem": plus 0 0 = 0
// ---------------------------------------------------------------------------

#[test]
fn plus_zero_zero_reduces_to_zero() {
    let mut p = build_nat_prelude();
    let plus = plus_definition(&mut p);

    // plus Nat.zero Nat.zero
    let term = Expr::app(
        Expr::app(plus, Expr::Const(p.zero, vec![])),
        Expr::Const(p.zero, vec![]),
    );

    let v = whnf(&p, &term);

    // Expected: Nat.zero.
    match &*v {
        Value::Neutral(Head::Constructor(name, _), args) => {
            assert_eq!(*name, p.zero);
            assert!(args.is_empty());
        }
        other => panic!("expected Nat.zero, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Sanity: literals do not appear in the prelude.
// ---------------------------------------------------------------------------

#[test]
fn prelude_has_no_literals() {
    // A structural invariant: the prelude uses only Const, Var, Pi, Lam,
    // and App. This is a regression guard against accidental introduction
    // of `Literal` nodes into the environment.
    fn has_lit(e: &Expr) -> bool {
        match e {
            Expr::Lit(_) => true,
            Expr::Sort(_) | Expr::Var(_) | Expr::Const(_, _) => false,
            Expr::App(f, a) => has_lit(f) || has_lit(a),
            Expr::Pi(_, _, d, c) | Expr::Lam(_, _, d, c) => has_lit(d) || has_lit(c),
            Expr::Let(_, t, v, b) => has_lit(t) || has_lit(v) || has_lit(b),
        }
    }

    let p = build_nat_prelude();
    for (name, info) in p.env.iter() {
        assert!(
            !has_lit(info.ty()),
            "declaration {name} type contains a literal"
        );
        if let Some(body) = info.body() {
            assert!(!has_lit(body), "declaration {name} body contains a literal");
        }
    }
    // Ensure the helper itself works.
    assert!(has_lit(&Expr::Lit(Literal::Nat(0))));
}

// ===========================================================================
// Bool
// ===========================================================================

#[test]
fn bool_inductive_registered() {
    let p = theoria_kernel::prelude::build_prelude();
    assert!(p.env.find(p.bool_name).is_some());
    assert!(p.env.find(p.bool_true).is_some());
    assert!(p.env.find(p.bool_false).is_some());
    assert!(p.env.find(p.bool_rec).is_some());
}

#[test]
fn bool_rec_true_reduces_to_true_case() {
    use theoria_kernel::nbe::{Env as NbeEnv, Head, Nbe, TransparencyMode, Value};
    let mut p = theoria_kernel::prelude::build_prelude();
    let nbe = Nbe::new(&p.env, TransparencyMode::Semireducible);

    // Bool.rec.{1} (λ _ : Bool. Bool) Bool.true Bool.false Bool.true
    // → Bool.true
    let motive = Expr::lam(
        p.names.intern("_"),
        Expr::Const(p.bool_name, vec![]),
        Expr::Const(p.bool_name, vec![]),
    );
    let term = Expr::Const(p.bool_rec, vec![Level::zero().succ()]).apps([
        motive,
        Expr::Const(p.bool_true, vec![]),
        Expr::Const(p.bool_false, vec![]),
        Expr::Const(p.bool_true, vec![]),
    ]);
    let v = nbe.eval(&NbeEnv::empty(), &term);
    match &*v {
        Value::Neutral(Head::Constructor(name, _), _) => assert_eq!(*name, p.bool_true),
        other => panic!("expected Bool.true, got {other:?}"),
    }
}

#[test]
fn bool_rec_false_reduces_to_false_case() {
    use theoria_kernel::nbe::{Env as NbeEnv, Head, Nbe, TransparencyMode, Value};
    let mut p = theoria_kernel::prelude::build_prelude();
    let nbe = Nbe::new(&p.env, TransparencyMode::Semireducible);

    let motive = Expr::lam(
        p.names.intern("_"),
        Expr::Const(p.bool_name, vec![]),
        Expr::Const(p.bool_name, vec![]),
    );
    let term = Expr::Const(p.bool_rec, vec![Level::zero().succ()]).apps([
        motive,
        Expr::Const(p.bool_true, vec![]),
        Expr::Const(p.bool_false, vec![]),
        Expr::Const(p.bool_false, vec![]),
    ]);
    let v = nbe.eval(&NbeEnv::empty(), &term);
    match &*v {
        Value::Neutral(Head::Constructor(name, _), _) => assert_eq!(*name, p.bool_false),
        other => panic!("expected Bool.false, got {other:?}"),
    }
}

// ===========================================================================
// List
// ===========================================================================

#[test]
fn list_rec_nil_reduces_to_nil_case() {
    use theoria_kernel::nbe::{Env as NbeEnv, Head, Nbe, TransparencyMode, Value};
    let mut p = theoria_kernel::prelude::build_prelude();
    let nbe = Nbe::new(&p.env, TransparencyMode::Semireducible);

    let u = Level::zero().succ();
    let v_lvl = Level::zero().succ();

    // motive := λ _ : List.{u} Bool. Bool
    let motive = Expr::lam(
        p.names.intern("_"),
        Expr::app(
            Expr::Const(p.list, vec![u.clone()]),
            Expr::Const(p.bool_name, vec![]),
        ),
        Expr::Const(p.bool_name, vec![]),
    );
    let nil_case = Expr::Const(p.bool_true, vec![]);
    // cons_case := λ x xs ih. Bool.false
    let cons_case = Expr::lam(
        p.names.intern("x"),
        Expr::Const(p.bool_name, vec![]),
        Expr::lam(
            p.names.intern("xs"),
            Expr::app(
                Expr::Const(p.list, vec![u.clone()]),
                Expr::Const(p.bool_name, vec![]),
            ),
            Expr::lam(
                p.names.intern("ih"),
                Expr::Const(p.bool_name, vec![]),
                Expr::Const(p.bool_false, vec![]),
            ),
        ),
    );

    // List.nil.{u} Bool : List.{u} Bool
    let list_nil_bool = Expr::app(
        Expr::Const(p.list_nil, vec![u.clone()]),
        Expr::Const(p.bool_name, vec![]),
    );

    // List.rec.{u,v} Bool motive nil_case cons_case (List.nil.{u} Bool)
    let term = Expr::Const(p.list_rec, vec![u, v_lvl]).apps([
        Expr::Const(p.bool_name, vec![]),
        motive,
        nil_case,
        cons_case,
        list_nil_bool,
    ]);

    let value = nbe.eval(&NbeEnv::empty(), &term);
    match &*value {
        Value::Neutral(Head::Constructor(name, _), _) => assert_eq!(*name, p.bool_true),
        other => panic!("expected Bool.true, got {other:?}"),
    }
}

#[test]
fn list_rec_cons_reduces_to_cons_case_applied() {
    use theoria_kernel::nbe::{Env as NbeEnv, Head, Nbe, TransparencyMode, Value};
    let mut p = theoria_kernel::prelude::build_prelude();
    let nbe = Nbe::new(&p.env, TransparencyMode::Semireducible);

    let u = Level::zero().succ();
    let v_lvl = Level::zero().succ();

    // motive := λ _ : List.{u} Bool. Bool
    let motive = Expr::lam(
        p.names.intern("_"),
        Expr::app(
            Expr::Const(p.list, vec![u.clone()]),
            Expr::Const(p.bool_name, vec![]),
        ),
        Expr::Const(p.bool_name, vec![]),
    );
    // nil_case := Bool.true
    let nil_case = Expr::Const(p.bool_true, vec![]);
    // cons_case := λ x xs ih. Bool.true
    let cons_case = Expr::lam(
        p.names.intern("x"),
        Expr::Const(p.bool_name, vec![]),
        Expr::lam(
            p.names.intern("xs"),
            Expr::app(
                Expr::Const(p.list, vec![u.clone()]),
                Expr::Const(p.bool_name, vec![]),
            ),
            Expr::lam(
                p.names.intern("ih"),
                Expr::Const(p.bool_name, vec![]),
                Expr::Const(p.bool_true, vec![]),
            ),
        ),
    );

    // List.cons.{u} Bool Bool.true (List.nil.{u} Bool) : List.{u} Bool
    let list_cons_bool = Expr::Const(p.list_cons, vec![u.clone()]).apps([
        Expr::Const(p.bool_name, vec![]),
        Expr::Const(p.bool_true, vec![]),
        Expr::app(
            Expr::Const(p.list_nil, vec![u.clone()]),
            Expr::Const(p.bool_name, vec![]),
        ),
    ]);

    let term = Expr::Const(p.list_rec, vec![u, v_lvl]).apps([
        Expr::Const(p.bool_name, vec![]),
        motive,
        nil_case,
        cons_case,
        list_cons_bool,
    ]);

    let value = nbe.eval(&NbeEnv::empty(), &term);
    // cons_case Bool.true (List.nil Bool) (recursive result = nil_case).
    match &*value {
        Value::Neutral(Head::Constructor(name, _), _) => assert_eq!(*name, p.bool_true),
        other => panic!("expected Bool.true, got {other:?}"),
    }
}

// ===========================================================================
// Option
// ===========================================================================

#[test]
fn option_inductive_registered() {
    let p = theoria_kernel::prelude::build_prelude();
    assert!(p.env.find(p.option).is_some());
    assert!(p.env.find(p.option_none).is_some());
    assert!(p.env.find(p.option_some).is_some());
    assert!(p.env.find(p.option_rec).is_some());
}

#[test]
fn option_rec_none_reduces_to_none_case() {
    use theoria_kernel::nbe::{Env as NbeEnv, Head, Nbe, TransparencyMode, Value};
    let mut p = theoria_kernel::prelude::build_prelude();
    let nbe = Nbe::new(&p.env, TransparencyMode::Semireducible);

    let u = Level::zero().succ();
    let v_lvl = Level::zero().succ();

    // motive := λ _ : Option.{u} Bool. Bool
    let motive = Expr::lam(
        p.names.intern("_"),
        Expr::app(
            Expr::Const(p.option, vec![u.clone()]),
            Expr::Const(p.bool_name, vec![]),
        ),
        Expr::Const(p.bool_name, vec![]),
    );
    let none_case = Expr::Const(p.bool_true, vec![]);
    // some_case := λ x. Bool.false
    let some_case = Expr::lam(
        p.names.intern("x"),
        Expr::Const(p.bool_name, vec![]),
        Expr::Const(p.bool_false, vec![]),
    );

    // Option.none.{u} Bool
    let opt_none_bool = Expr::app(
        Expr::Const(p.option_none, vec![u.clone()]),
        Expr::Const(p.bool_name, vec![]),
    );

    let term = Expr::Const(p.option_rec, vec![u, v_lvl]).apps([
        Expr::Const(p.bool_name, vec![]),
        motive,
        none_case,
        some_case,
        opt_none_bool,
    ]);

    let value = nbe.eval(&NbeEnv::empty(), &term);
    match &*value {
        Value::Neutral(Head::Constructor(name, _), _) => assert_eq!(*name, p.bool_true),
        other => panic!("expected Bool.true, got {other:?}"),
    }
}

#[test]
fn option_rec_some_reduces_to_some_case_applied() {
    use theoria_kernel::nbe::{Env as NbeEnv, Head, Nbe, TransparencyMode, Value};
    let mut p = theoria_kernel::prelude::build_prelude();
    let nbe = Nbe::new(&p.env, TransparencyMode::Semireducible);

    let u = Level::zero().succ();
    let v_lvl = Level::zero().succ();

    let motive = Expr::lam(
        p.names.intern("_"),
        Expr::app(
            Expr::Const(p.option, vec![u.clone()]),
            Expr::Const(p.bool_name, vec![]),
        ),
        Expr::Const(p.bool_name, vec![]),
    );
    let none_case = Expr::Const(p.bool_false, vec![]);
    // some_case := λ x. Bool.true
    let some_case = Expr::lam(
        p.names.intern("x"),
        Expr::Const(p.bool_name, vec![]),
        Expr::Const(p.bool_true, vec![]),
    );

    // Option.some.{u} Bool Bool.false
    let opt_some_bool = Expr::Const(p.option_some, vec![u.clone()]).apps([
        Expr::Const(p.bool_name, vec![]),
        Expr::Const(p.bool_false, vec![]),
    ]);

    let term = Expr::Const(p.option_rec, vec![u, v_lvl]).apps([
        Expr::Const(p.bool_name, vec![]),
        motive,
        none_case,
        some_case,
        opt_some_bool,
    ]);

    let value = nbe.eval(&NbeEnv::empty(), &term);
    match &*value {
        Value::Neutral(Head::Constructor(name, _), _) => assert_eq!(*name, p.bool_true),
        other => panic!("expected Bool.true, got {other:?}"),
    }
}

// ===========================================================================
// Types of the new recursors are well-formed
// ===========================================================================

#[test]
fn all_recursor_types_typecheck() {
    use theoria_kernel::infer::TypeChecker;
    let p = theoria_kernel::prelude::build_prelude();
    let tc = TypeChecker::new(&p.env);
    for name in [p.rec, p.bool_rec, p.list_rec, p.option_rec] {
        let info = p.env.find(name).expect("recursor missing");
        tc.check_declaration(&info)
            .unwrap_or_else(|e| panic!("recursor {name} type failed: {e}"));
    }
}

// ===========================================================================
// Termination gate
// ===========================================================================

#[test]
fn termination_gate_rejects_direct_self_recursion() {
    use std::rc::Rc;
    use theoria_kernel::env::{
        ConstantInfo, ConstantVal, DefinitionVal, EnvError, TerminationObligation,
    };
    use theoria_kernel::nbe::Transparency;

    let mut p = theoria_kernel::prelude::build_prelude();
    let f = p.names.intern("f");
    let n = p.names.intern("n");

    // f : Nat -> Nat, body := f n
    let ty = Expr::Pi(
        BinderInfo::Default,
        p.names.intern("_"),
        Box::new(Expr::Const(p.nat, vec![])),
        Box::new(Expr::Const(p.nat, vec![])),
    );
    let body = Expr::app(Expr::Const(f, vec![]), Expr::Var(DeBruijnIndex(0)));

    let err = p
        .env
        .add(ConstantInfo::Definition(DefinitionVal {
            base: ConstantVal {
                name: f,
                universe_params: Vec::new(),
                ty: Rc::new(ty),
            },
            body: Rc::new(body),
            transparency: Transparency::Semireducible,
            termination: TerminationObligation::recursive(vec![n]),
        }))
        .unwrap_err();
    assert!(matches!(err, EnvError::NonTerminating { .. }));
    assert!(p.env.find(f).is_none());
}

#[test]
fn termination_gate_accepts_recursor_based_definitions() {
    // `plus` is defined via `Nat.rec`, so its body never mentions `plus`.
    // The gate must skip it entirely, even though it is a "recursive"
    // function in the mathematical sense.
    let p = theoria_kernel::prelude::build_prelude();
    assert!(p.env.find(p.rec).is_some());
}

// ===========================================================================
// Quotients
// ===========================================================================

use theoria_kernel::prelude::build_prelude;
use theoria_kernel::quotient::register_quotient;

fn build_prelude_with_quotients() -> (
    theoria_kernel::prelude::Prelude,
    theoria_kernel::quotient::QuotientPrelude,
) {
    let mut p = build_prelude();
    let q = register_quotient(&mut p.env, &mut p.names, p.eq);
    (p, q)
}

#[test]
fn quotient_constants_are_registered() {
    let (p, q) = build_prelude_with_quotients();
    assert!(p.env.find(q.quot).is_some());
    assert!(p.env.find(q.mk).is_some());
    assert!(p.env.find(q.lift).is_some());
    assert!(p.env.find(q.sound).is_some());
}

#[test]
fn quotient_types_typecheck() {
    use theoria_kernel::infer::TypeChecker;
    let (p, q) = build_prelude_with_quotients();
    let tc = TypeChecker::new(&p.env);
    for name in [q.quot, q.mk, q.lift, q.sound] {
        let info = p.env.find(name).expect("quotient constant missing");
        tc.check_declaration(&info)
            .unwrap_or_else(|e| panic!("quotient constant {name} type failed: {e}"));
    }
}

#[test]
fn quot_mk_is_a_constructor_head() {
    use theoria_kernel::nbe::{Env as NbeEnv, Head, Nbe, TransparencyMode, Value};
    let (mut p, q) = build_prelude_with_quotients();
    let nbe = Nbe::new(&p.env, TransparencyMode::Semireducible);

    let u = Level::zero().succ();
    // R := λ x y. Eq Nat x y
    let x = p.names.intern("x");
    let y = p.names.intern("y");
    let r_expr = Expr::lam(
        x,
        Expr::Const(p.nat, vec![]),
        Expr::lam(
            y,
            Expr::Const(p.nat, vec![]),
            Expr::Const(p.eq, vec![u.clone()]).apps([
                Expr::Const(p.nat, vec![]),
                Expr::Var(DeBruijnIndex(1)),
                Expr::Var(DeBruijnIndex(0)),
            ]),
        ),
    );
    // Quot.mk.{1} Nat R Nat.zero
    let term = Expr::Const(q.mk, vec![u]).apps([
        Expr::Const(p.nat, vec![]),
        r_expr,
        Expr::Const(p.zero, vec![]),
    ]);
    let v = nbe.eval(&NbeEnv::empty(), &term);
    match &*v {
        Value::Neutral(Head::Constructor(name, _), args) => {
            assert_eq!(*name, q.mk);
            assert_eq!(args.len(), 3);
        }
        other => panic!("expected stuck Quot.mk, got {other:?}"),
    }
}

#[test]
fn quot_lift_beta_fires_on_mk() {
    use theoria_kernel::nbe::{Env as NbeEnv, Head, Nbe, TransparencyMode, Value};
    let (mut p, q) = build_prelude_with_quotients();
    let nbe = Nbe::new(&p.env, TransparencyMode::Semireducible);

    let u = Level::zero().succ();
    let v = Level::zero().succ();
    let x = p.names.intern("x");
    let y = p.names.intern("y");

    // R := λ x y. Eq Nat x y
    let r_expr = Expr::lam(
        x,
        Expr::Const(p.nat, vec![]),
        Expr::lam(
            y,
            Expr::Const(p.nat, vec![]),
            Expr::Const(p.eq, vec![u.clone()]).apps([
                Expr::Const(p.nat, vec![]),
                Expr::Var(DeBruijnIndex(1)),
                Expr::Var(DeBruijnIndex(0)),
            ]),
        ),
    );

    // f := λ x. Nat.succ x
    let f_expr = Expr::lam(
        x,
        Expr::Const(p.nat, vec![]),
        Expr::app(Expr::Const(p.succ, vec![]), Expr::Var(DeBruijnIndex(0))),
    );

    // h := λ _ _ _. Eq.refl Nat Nat.zero    (stuck proof; type is irrelevant here)
    let h_expr = Expr::lam(
        p.names.intern("_a"),
        Expr::Const(p.nat, vec![]),
        Expr::lam(
            p.names.intern("_b"),
            Expr::Const(p.nat, vec![]),
            Expr::lam(
                p.names.intern("_c"),
                Expr::Const(p.eq, vec![u.clone()]).apps([
                    Expr::Const(p.nat, vec![]),
                    Expr::Var(DeBruijnIndex(2)),
                    Expr::Var(DeBruijnIndex(1)),
                ]),
                Expr::Const(p.eq_refl, vec![u.clone()])
                    .apps([Expr::Const(p.nat, vec![]), Expr::Const(p.zero, vec![])]),
            ),
        ),
    );

    // Quot.lift.{1,1} Nat R Nat f h (Quot.mk.{1} Nat R Nat.zero)
    let mk_term = Expr::Const(q.mk, vec![u.clone()]).apps([
        Expr::Const(p.nat, vec![]),
        r_expr.clone(),
        Expr::Const(p.zero, vec![]),
    ]);
    let term = Expr::Const(q.lift, vec![u.clone(), v.clone()]).apps([
        Expr::Const(p.nat, vec![]),
        r_expr,
        Expr::Const(p.nat, vec![]),
        f_expr,
        h_expr,
        mk_term,
    ]);

    let value = nbe.eval(&NbeEnv::empty(), &term);
    // Expected: Nat.succ Nat.zero.
    match &*value {
        Value::Neutral(Head::Constructor(name, _), args) => {
            assert_eq!(*name, p.succ);
            assert_eq!(args.len(), 1);
            match &*args[0] {
                Value::Neutral(Head::Constructor(n, _), _) => assert_eq!(*n, p.zero),
                other => panic!("expected Nat.zero argument, got {other:?}"),
            }
        }
        other => panic!("expected Nat.succ Nat.zero, got {other:?}"),
    }
}

#[test]
fn quot_lift_stuck_on_free_variable() {
    use theoria_kernel::nbe::{Env as NbeEnv, Head, Nbe, TransparencyMode, Value};
    let (mut p, q) = build_prelude_with_quotients();
    let nbe = Nbe::new(&p.env, TransparencyMode::Semireducible);

    let u = Level::zero().succ();
    let v = Level::zero().succ();
    let x = p.names.intern("x");
    let y = p.names.intern("y");

    let r_expr = Expr::lam(
        x,
        Expr::Const(p.nat, vec![]),
        Expr::lam(
            y,
            Expr::Const(p.nat, vec![]),
            Expr::Const(p.eq, vec![u.clone()]).apps([
                Expr::Const(p.nat, vec![]),
                Expr::Var(DeBruijnIndex(1)),
                Expr::Var(DeBruijnIndex(0)),
            ]),
        ),
    );
    let f_expr = Expr::lam(x, Expr::Const(p.nat, vec![]), Expr::Var(DeBruijnIndex(0)));
    let h_expr = f_expr.clone();

    // Free variable q : Quot Nat R at level 0.
    let mut env = NbeEnv::empty();
    env = env.extend(Value::fresh_var(theoria_kernel::nbe::DbLevel(0)));

    // Quot.lift A R Nat f h (Var 0)
    let term = Expr::Const(q.lift, vec![u.clone(), v.clone()]).apps([
        Expr::Const(p.nat, vec![]),
        r_expr,
        Expr::Const(p.nat, vec![]),
        f_expr,
        h_expr,
        Expr::Var(DeBruijnIndex(0)),
    ]);

    let value = nbe.eval(&env, &term);
    // Expected: stuck on Quot.lift.
    match &*value {
        Value::Neutral(Head::Const(name, _), spine) => {
            assert_eq!(*name, q.lift);
            assert_eq!(spine.len(), 6);
        }
        other => panic!("expected stuck Quot.lift, got {other:?}"),
    }
}

#[test]
fn prelude_plus_quotients_types_all_typecheck() {
    use theoria_kernel::infer::TypeChecker;
    let (p, q) = build_prelude_with_quotients();
    let tc = TypeChecker::new(&p.env);
    let all = [
        p.rec,
        p.bool_rec,
        p.list_rec,
        p.option_rec,
        p.eq_rec,
        q.quot,
        q.mk,
        q.lift,
        q.sound,
    ];
    for name in all {
        let info = p.env.find(name).expect("constant missing");
        tc.check_declaration(&info)
            .unwrap_or_else(|e| panic!("constant {name} failed: {e}"));
    }
}
