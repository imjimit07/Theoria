//! Property-based differential tests for the kernel.
//!
//! These are not unit tests. They exercise the whole pipeline — parser-less
//! `Expr` construction, arena interning, NbE evaluation, readback, and
//! conversion — against randomly generated terms. Reproducibility comes
//! from a fixed seed; a different seed can be selected via the
//! `THEORIA_FUZZ_SEED` environment variable.
//!
//! ## Properties checked
//!
//! 1. **Arena roundtrip.** `to_expr(intern(e)) == e`.
//! 2. **Arena idempotence.** `intern(e) == intern(e)`.
//! 3. **Arena structural equality.** `e1 == e2` ⇒ `intern(e1) == intern(e2)`.
//! 4. **Well-typed closed terms survive infer.** For randomly generated
//!    `Nat`-valued terms, `infer` succeeds.
//! 5. **Conversion is reflexive.** `convert(eval(t), eval(t))` is true.
//! 6. **Conversion is symmetric.** `convert(a, b) == convert(b, a)`.

use theoria_kernel::arena::ExprArena;
use theoria_kernel::expr::{BinderInfo, DeBruijnIndex, Expr, Literal};
use theoria_kernel::infer::{TypeChecker, TypedContext};
use theoria_kernel::level::Level;
use theoria_kernel::name::{NameId, NameTable};
use theoria_kernel::nbe::{Env as NbeEnv, Nbe, TransparencyMode};
use theoria_kernel::prelude::{Prelude, build_prelude};

// ---------------------------------------------------------------------------
// RNG
// ---------------------------------------------------------------------------

/// A tiny xorshift64 generator. Deterministic, no dependencies.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        // Seed must be non-zero for xorshift.
        Rng(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn below(&mut self, n: u64) -> u64 {
        assert!(n > 0, "Rng::below(0)");
        self.next_u64() % n
    }
}

fn seed() -> u64 {
    std::env::var("THEORIA_FUZZ_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0x00C0_FFEE_5EED)
}

// ---------------------------------------------------------------------------
// Generators
// ---------------------------------------------------------------------------

/// Generate a closed `Expr` of bounded depth.
///
/// `ctx` tracks the number of locally bound variables available; `Var(i)`
/// is only emitted for `i < ctx`.
fn gen_closed_expr(rng: &mut Rng, names: &[NameId], depth: u32, ctx: u32) -> Expr {
    if depth == 0 {
        return gen_leaf(rng, names, ctx);
    }
    match rng.below(6) {
        0 => gen_leaf(rng, names, ctx),
        1 => {
            let f = gen_closed_expr(rng, names, depth - 1, ctx);
            let a = gen_closed_expr(rng, names, depth - 1, ctx);
            Expr::app(f, a)
        }
        2 => {
            let dom = gen_closed_expr(rng, names, depth - 1, ctx);
            let body = gen_closed_expr(rng, names, depth - 1, ctx + 1);
            Expr::lam(NameId(0), dom, body)
        }
        3 => {
            let dom = gen_closed_expr(rng, names, depth - 1, ctx);
            let cod = gen_closed_expr(rng, names, depth - 1, ctx + 1);
            Expr::Pi(BinderInfo::Default, NameId(0), Box::new(dom), Box::new(cod))
        }
        4 => {
            let ty = gen_closed_expr(rng, names, depth - 1, ctx);
            let val = gen_closed_expr(rng, names, depth - 1, ctx);
            let body = gen_closed_expr(rng, names, depth - 1, ctx + 1);
            Expr::Let(NameId(0), Box::new(ty), Box::new(val), Box::new(body))
        }
        _ => {
            // Random constant with random (possibly empty) universe levels.
            let n = names[rng.below(names.len() as u64) as usize];
            let n_levels = rng.below(3) as usize;
            let levels: Vec<Level> = (0..n_levels)
                .map(|_| Level::param(theoria_kernel::UniverseParamId(rng.next_u64() as u32)))
                .collect();
            Expr::Const(n, levels)
        }
    }
}

fn gen_leaf(rng: &mut Rng, names: &[NameId], ctx: u32) -> Expr {
    // 3 fixed leaves (Sort, Lit, Const) + ctx free variables.
    let choices = 3 + u64::from(ctx);
    match rng.below(choices) {
        0 => Expr::prop(),
        1 => Expr::Lit(Literal::Nat(rng.next_u64())),
        2 => Expr::Const(names[rng.below(names.len() as u64) as usize], Vec::new()),
        i => Expr::Var(DeBruijnIndex((i - 3) as u32)),
    }
}

/// Generate a closed, well-typed `Nat`-valued term.
fn gen_nat_term(rng: &mut Rng, p: &Prelude, depth: u32) -> Expr {
    if depth == 0 || rng.below(3) == 0 {
        return Expr::Const(p.zero, Vec::new());
    }
    // `Nat.succ` applied to a recursive argument.
    Expr::app(
        Expr::Const(p.succ, Vec::new()),
        gen_nat_term(rng, p, depth - 1),
    )
}

// ---------------------------------------------------------------------------
// Property 1 & 2 & 3: arena
// ---------------------------------------------------------------------------

#[test]
fn fuzz_arena_roundtrip() {
    let mut rng = Rng::new(seed());
    let mut names = NameTable::new();
    let nat = names.intern("Nat");
    let succ = names.intern("Nat.succ");
    let zero = names.intern("Nat.zero");
    let name_ids = vec![nat, succ, zero];

    for trial in 0..300 {
        let e = gen_closed_expr(&mut rng, &name_ids, 6, 0);
        let mut arena = ExprArena::new();
        let id = match arena.intern(&e) {
            Ok(id) => id,
            Err(err) => panic!("trial {trial}: intern failed: {err}"),
        };
        let back = arena.to_expr(id);
        assert_eq!(back, e, "trial {trial}: roundtrip mismatch");
    }
}

#[test]
fn fuzz_arena_idempotent() {
    let mut rng = Rng::new(seed() ^ 1);
    let mut names = NameTable::new();
    let nat = names.intern("Nat");
    let succ = names.intern("Nat.succ");
    let name_ids = vec![nat, succ];

    for trial in 0..200 {
        let e = gen_closed_expr(&mut rng, &name_ids, 5, 0);
        let mut arena = ExprArena::new();
        let a = arena.intern(&e).unwrap();
        let b = arena.intern(&e).unwrap();
        assert_eq!(a, b, "trial {trial}: idempotence violated");
    }
}

#[test]
fn fuzz_arena_structural_equality_implies_id_equality() {
    let mut rng = Rng::new(seed() ^ 2);
    let mut names = NameTable::new();
    let nat = names.intern("Nat");
    let zero = names.intern("Nat.zero");
    let name_ids = vec![nat, zero];

    for trial in 0..200 {
        let e = gen_closed_expr(&mut rng, &name_ids, 4, 0);
        // Two independent clones must be structurally equal.
        let e2 = e.clone();
        let mut arena = ExprArena::new();
        let a = arena.intern(&e).unwrap();
        let b = arena.intern(&e2).unwrap();
        assert_eq!(
            a, b,
            "trial {trial}: structural-equal terms got distinct ids"
        );
    }
}

// ---------------------------------------------------------------------------
// Property 4 & 5: well-typed closed terms
// ---------------------------------------------------------------------------

#[test]
fn fuzz_well_typed_nat_terms_infer_and_eval() {
    let p = build_prelude();
    let mut rng = Rng::new(seed() ^ 3);
    let checker = TypeChecker::new(&p.env);
    let nbe = Nbe::new(&p.env, TransparencyMode::Semireducible);
    let env = NbeEnv::empty();
    let ctx = TypedContext::empty();

    for trial in 0..200 {
        let e = gen_nat_term(&mut rng, &p, 8);
        // Inference must succeed.
        let ty = checker
            .infer(&ctx, &e)
            .unwrap_or_else(|err| panic!("trial {trial}: infer failed: {err}"));
        // The inferred type must be defeq to the *value* of Nat.
        let nat_value = nbe.eval(&env, &Expr::Const(p.nat, Vec::new()));
        assert!(
            nbe.convert(ctx.size(), &ty, &nat_value),
            "trial {trial}: expected Nat, got {ty:?}"
        );
        // Evaluation must succeed.
        let _ = nbe.eval(&env, &e);
    }
}

#[test]
fn fuzz_conversion_reflexive() {
    let p = build_prelude();
    let mut rng = Rng::new(seed() ^ 4);
    let nbe = Nbe::new(&p.env, TransparencyMode::Semireducible);
    let env = NbeEnv::empty();

    for trial in 0..200 {
        let e = gen_nat_term(&mut rng, &p, 6);
        let v = nbe.eval(&env, &e);
        assert!(
            nbe.convert(0, &v, &v),
            "trial {trial}: conversion not reflexive"
        );
    }
}

#[test]
fn fuzz_conversion_symmetric() {
    let p = build_prelude();
    let mut rng = Rng::new(seed() ^ 5);
    let nbe = Nbe::new(&p.env, TransparencyMode::Semireducible);
    let env = NbeEnv::empty();

    for trial in 0..100 {
        let a = gen_nat_term(&mut rng, &p, 5);
        let b = gen_nat_term(&mut rng, &p, 5);
        let va = nbe.eval(&env, &a);
        let vb = nbe.eval(&env, &b);
        let ab = nbe.convert(0, &va, &vb);
        let ba = nbe.convert(0, &vb, &va);
        assert_eq!(ab, ba, "trial {trial}: conversion asymmetry");
    }
}

// ---------------------------------------------------------------------------
// Property 6: arena shares structure with itself
// ---------------------------------------------------------------------------

#[test]
fn fuzz_arena_shares_repeated_subterms() {
    let mut rng = Rng::new(seed() ^ 6);
    let mut names = NameTable::new();
    let nat = names.intern("Nat");
    let succ = names.intern("Nat.succ");
    let name_ids = vec![nat, succ];

    for trial in 0..100 {
        // Build a term by explicit sharing: succ applied to itself twice.
        let mut arena = ExprArena::new();
        let z = arena.const_(theoria_kernel::NameId(2), Vec::new());
        let s = arena.const_(succ, Vec::new());
        let sz = arena.app(s, z);
        let ssz = arena.app(s, sz);
        let sssz = arena.app(s, ssz);
        // Interning the same flat expression should reuse every subterm.
        let flat = arena.to_expr(sssz);
        let id = arena.intern(&flat).unwrap();
        assert_eq!(id, sssz, "trial {trial}: no structural sharing");
        // Also fuzz unrelated terms to keep the arena warm.
        let _ = gen_closed_expr(&mut rng, &name_ids, 3, 0);
    }
}

// ---------------------------------------------------------------------------
// Ensure we don't silently pass on empty inputs
// ---------------------------------------------------------------------------

#[test]
fn fuzz_generator_produces_nontrivial_terms() {
    let mut rng = Rng::new(seed() ^ 7);
    let mut names = NameTable::new();
    let nat = names.intern("Nat");
    let succ = names.intern("Nat.succ");
    let name_ids = vec![nat, succ];

    let mut total_size = 0usize;
    let mut max_depth = 0usize;
    for _ in 0..200 {
        let e = gen_closed_expr(&mut rng, &name_ids, 6, 0);
        total_size += e.size();
        max_depth = max_depth.max(term_depth(&e));
    }
    assert!(
        total_size > 200,
        "generator produced only trivial terms (total size {total_size})"
    );
    assert!(max_depth >= 3, "generator depth too shallow ({max_depth})");
}

fn term_depth(e: &Expr) -> usize {
    match e {
        Expr::Sort(_) | Expr::Var(_) | Expr::Const(_, _) | Expr::Lit(_) => 1,
        Expr::App(f, a) => 1 + term_depth(f).max(term_depth(a)),
        Expr::Lam(_, _, d, b) | Expr::Pi(_, _, d, b) => 1 + term_depth(d).max(term_depth(b)),
        Expr::Let(_, t, v, b) => 1 + term_depth(t).max(term_depth(v)).max(term_depth(b)),
    }
}
