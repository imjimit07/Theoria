//! Performance benchmarks for the kernel.
//!
//! These are not unit tests: nothing here asserts on timing, and the
//! values printed depend on the machine and the runtime load. The point
//! is to have a repeatable measurement of the hot paths that Phase 2's
//! elaborator will exercise, and a baseline to compare against after
//! performance work.
//!
//! Run with `cargo test -p theoria_kernel --release --test bench -- --nocapture`
//! for meaningful numbers. Debug builds still run, only slower.
//!
//! ## Why there is no allocation counter
//!
//! The obvious way to count allocations is a `#[global_allocator]` that
//! wraps `System`. Integration test binaries inherit the package's
//! `unsafe_code = "forbid"` lint, and `forbid` cannot be overridden from
//! source, so the counter cannot live here. It belongs in a separate
//! crate with its own `[lints]` table; that is a follow-up.

use std::rc::Rc;
use std::time::{Duration, Instant};

use theoria_kernel::arena::ExprArena;
use theoria_kernel::env::{ConstantInfo, ConstantVal, DefinitionVal, TerminationObligation};
use theoria_kernel::expr::{BinderInfo, Expr};
use theoria_kernel::infer::{TypeChecker, TypedContext};
use theoria_kernel::nbe::{ConstEnv, Env as NbeEnv, Nbe, Transparency, TransparencyMode};
use theoria_kernel::prelude::{Prelude, build_prelude};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// `n` as a closed kernel `Nat` term (`Nat.succ^n Nat.zero`).
fn num(p: &Prelude, n: u64) -> Expr {
    let mut e = Expr::Const(p.zero, Vec::new());
    for _ in 0..n {
        e = Expr::app(Expr::Const(p.succ, Vec::new()), e);
    }
    e
}

/// Build a `Nat -> Nat` definition: `λ _ : Nat. body`.
///
/// The caller supplies `body` which will be the lambda body.
fn mk_def(p: &mut Prelude, name: &str, body: Expr) -> ConstantInfo {
    let nat = p.nat;
    let id = p.names.intern(name);
    let underscore = p.names.intern("_");
    let ty = Expr::Pi(
        BinderInfo::Default,
        underscore,
        Box::new(Expr::Const(nat, Vec::new())),
        Box::new(Expr::Const(nat, Vec::new())),
    );
    // Wrap body in λ _ : Nat. body
    let lam_body = Expr::Lam(
        BinderInfo::Default,
        underscore,
        Box::new(Expr::Const(nat, Vec::new())),
        Box::new(body),
    );
    ConstantInfo::Definition(DefinitionVal {
        base: ConstantVal {
            name: id,
            universe_params: Vec::new(),
            ty: Rc::new(ty),
        },
        body: Rc::new(lam_body),
        transparency: Transparency::Semireducible,
        termination: TerminationObligation::none(),
    })
}

/// A one-shot timing block that prints its label and elapsed time.
struct Timer {
    label: &'static str,
    start: Instant,
}

impl Timer {
    fn start(label: &'static str) -> Self {
        Timer {
            label,
            start: Instant::now(),
        }
    }

    /// Print the elapsed time and return it.
    ///
    /// The return value is usually discarded by the caller; it is
    /// available so that a caller who wants to compare timings across
    /// variants can do so.
    fn finish(self) -> Duration {
        let elapsed = self.start.elapsed();
        println!("{}: {:?}", self.label, elapsed);
        elapsed
    }
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

/// Chain of `N` definitions: `f₀ = λ _ : Nat. Nat.succ Nat.zero`,
/// `fᵢ = λ _ : Nat. fᵢ₋₁ (Nat.succ Nat.zero)`.
///
/// Each definition is independently typechecked by `TypeChecker` and then
/// inserted into the environment. The timing reflects the incremental cost
/// of `check_declaration` plus environment insertion, which is what the
/// elaborator spends most of its time on.
#[test]
fn bench_elaboration_of_definition_chain() {
    const N: usize = 100;
    let mut p = build_prelude();
    let t = Timer::start("bench_elaboration_of_definition_chain");

    let succ = p.succ;
    let zero = p.zero;

    // f₀ = λ _ : Nat. Nat.succ Nat.zero
    let mut prev_name = p.names.intern("f0_init");
    {
        let info = mk_def(
            &mut p,
            "f0_init",
            Expr::app(Expr::Const(succ, Vec::new()), Expr::Const(zero, Vec::new())),
        );
        p.env.add(info).expect("f0_init should insert");
    }
    let tc = TypeChecker::new(&p.env);
    let info = p.env.find(prev_name).unwrap();
    tc.check_declaration(&info).expect("f0_init should check");

    for i in 1..=N {
        let name = format!("f{i}");
        // f_i = λ _ : Nat. f_{i-1} (Nat.succ Nat.zero)
        let body = Expr::app(
            Expr::Const(prev_name, Vec::new()),
            Expr::app(Expr::Const(succ, Vec::new()), Expr::Const(zero, Vec::new())),
        );
        let info = mk_def(&mut p, &name, body);
        let id = info.name();

        let tc = TypeChecker::new(&p.env);
        tc.check_declaration(&info)
            .expect("benchmark definition should check");

        p.env.add(info).expect("benchmark definition should insert");
        prev_name = id;
    }

    let _ = t.finish();
    // 33 prelude declarations + 1 (f0_init) + N user definitions.
    assert_eq!(p.env.len(), 33 + 1 + N);
}

/// Evaluate a left-nested chain of `N` additions on `Nat`.
///
/// `Nat.add` is a transparent definition over `Nat.rec`, so each step
/// unfolds through the recursor's ι-rule. The timing reflects `Nbe::eval`
/// on a term whose spine length grows with `N`, plus `quote` of the final
/// value.
#[test]
fn bench_eval_of_addition_chain() {
    const N: u64 = 100;
    let p = build_prelude();

    // Nat.add(Nat.add(...Nat.add(0, 1)..., 1), 1)
    let mut e = num(&p, 0);
    for _ in 0..N {
        e = Expr::app(Expr::app(Expr::Const(p.add, Vec::new()), e), num(&p, 1));
    }

    let nbe = Nbe::new(&p.env, TransparencyMode::Semireducible);
    let t = Timer::start("bench_eval_of_addition_chain");

    let v = nbe.eval(&NbeEnv::empty(), &e);
    let back = nbe.quote(0, &v);

    let _ = t.finish();

    // Sanity: the normal form is `N` as a successor chain. This rules
    // out a benchmark that accidentally measures a stuck term.
    assert_eq!(back, num(&p, N));
}

/// Typecheck a deep application: `Nat.succ(Nat.succ(...Nat.zero...))`.
///
/// The kernel infers the type of every nested application and, at each
/// level, performs a defeq check between the argument's inferred type and
/// the expected domain. The timing therefore grows with the depth of the
/// application spine.
#[test]
fn bench_infer_of_a_deep_application() {
    const N: usize = 200;
    let p = build_prelude();

    let mut e = Expr::Const(p.zero, Vec::new());
    for _ in 0..N {
        e = Expr::app(Expr::Const(p.succ, Vec::new()), e);
    }

    let tc = TypeChecker::new(&p.env);
    let ctx = TypedContext::empty();
    let t = Timer::start("bench_infer_of_a_deep_application");

    let ty = tc.infer(&ctx, &e).expect("infer should succeed");

    let _ = t.finish();

    // Sanity: the inferred type must be Nat, not something else.
    let nat_v = tc
        .nbe()
        .eval(&NbeEnv::empty(), &Expr::Const(p.nat, Vec::new()));
    assert!(tc.nbe().convert(0, &ty, &nat_v));
}

/// Intern a moderately large term and re-intern it.
///
/// A well-behaved hash-consed arena should make the second intern cheap:
/// every subterm is already in the dedup index, so the second pass is all
/// hash probes plus structural comparisons, no insertions.
#[test]
fn bench_arena_interning() {
    const N: u64 = 500;
    let p = build_prelude();

    let mut e = num(&p, 0);
    for _ in 0..N {
        e = Expr::app(Expr::Const(p.succ, Vec::new()), e);
    }

    let mut arena = ExprArena::new();
    let t = Timer::start("bench_arena_interning");

    let id1 = arena.intern(&e).expect("first intern");
    let id2 = arena.intern(&e).expect("second intern");

    let _ = t.finish();

    assert_eq!(id1, id2, "arena intern must be idempotent");
    // One node per successor plus `Nat.zero` plus `Nat.succ`: N + 2.
    println!("  arena len: {}", arena.len());
    assert_eq!(arena.len() as u64, N + 2);
}

/// Measure the per-`Const` cost of `ConstEnv::lookup`.
///
/// This is the hot path inside `Nbe::eval_const` and
/// `TypeChecker::infer_const`. With the cached `ConstDecl` on `EnvEntry`,
/// each lookup is a `BTreeMap` probe plus a refcount bump rather than a
/// fresh `ConstDecl` construction.
#[test]
fn bench_env_lookup_through_const_env() {
    const N: usize = 100_000;
    let p = build_prelude();
    let add = p.add;

    let t = Timer::start("bench_env_lookup_through_const_env");

    // First lookup: capture the returned `Rc` so we can pin the cache
    // identity after the loop. `GlobalEnv::lookup` must return the
    // `Rc<ConstDecl>` stored on the `EnvEntry`, not a freshly-built one;
    // if a future regression reintroduces `to_const_decl()` on the hot
    // path, this assertion fires even though the *contents* of the decl
    // would still be correct.
    let first = p
        .env
        .lookup(add)
        .expect("Nat.add must be present in the prelude");

    let mut last = Rc::clone(&first);
    for _ in 1..N {
        last = p.env.lookup(add).expect("Nat.add must remain present");
    }

    let _ = t.finish();

    // Sanity: the name is what we asked for.
    assert_eq!(last.name, add);

    // The actual cache-identity check.
    assert!(
        Rc::ptr_eq(&first, &last),
        "ConstEnv::lookup must return the cached Rc<ConstDecl>, \
         not a freshly-constructed one"
    );
}
