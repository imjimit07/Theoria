//! Normalization by Evaluation (NbE) engine.
//!
//! This module is the operational heart of the kernel. It provides three
//! mutually-recursive services:
//!
//! 1. [`Nbe::eval`] — evaluates a syntactic [`Expr`] in a semantic
//!    [`Env`] into a [`Value`].
//! 2. [`Nbe::quote`] — reads back a [`Value`] into a canonical, βη-normal
//!    [`Expr`] at a given context size.
//! 3. [`Nbe::convert`] — decides definitional equality of two [`Value`]s
//!    up to β, δ, ι, and η.
//!
//! ## Design notes
//!
//! * **De Bruijn levels, not indices, in values.** `Value` uses
//!   [`DbLevel`] because values are built bottom-up (the outermost binder
//!   has level `0`); `Expr` uses [`DeBruijnIndex`]
//!   because terms are read top-down. The two representations meet at
//!   [`Nbe::eval`] / [`Nbe::quote`].
//! * **Defunctionalized closures.** A [`Closure`] is a pair of a syntactic
//!   body and a semantic environment — never a native Rust closure. This
//!   keeps values structurally inspectable (needed by `quote` and
//!   `convert`) and prevents leaking the interpreter's control flow into
//!   the semantic domain.
//! * **Persistent environments.** [`Env`] is a persistent linked list
//!   behind `Rc`, so extending it during evaluation is O(1) and old
//!   environments are never mutated.
//! * **Shared subtrees.** [`ValRef`] is `Rc<Value>`; sharing keeps
//!   conversion checking cheap on large types and lets `Rc::ptr_eq` serve
//!   as a fast path.
//! * **Total on well-typed inputs.** `eval` and `apply` are total on
//!   well-typed terms; `apply` panics on a non-function value, which is
//!   unreachable for any term the elaborator accepts.
//!
//! ## Universe polymorphism
//!
//! When a constant is unfolded (`δ`-reduction), its universe parameters
//! are substituted by the actual levels from the `Const` node. See
//! [`substitute_levels`].

use crate::expr::{BinderInfo, DeBruijnIndex, Expr, Literal};
use crate::level::{Level, UniverseParamId};
use crate::name::NameId;
use alloc::boxed::Box;
use alloc::rc::Rc;
use alloc::vec::Vec;
use core::fmt;

// ---------------------------------------------------------------------------
// De Bruijn levels
// ---------------------------------------------------------------------------

/// A de Bruijn *level*: counts binders from the outermost.
///
/// Level `0` is the outermost binder in the current context; level
/// `ctx_size - 1` is the innermost. This is the representation used inside
/// [`Value`], because semantic values are constructed bottom-up.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DbLevel(pub u32);

impl DbLevel {
    /// The outermost level.
    #[must_use]
    pub const fn zero() -> Self {
        DbLevel(0)
    }

    /// The next level inward.
    #[must_use]
    pub const fn succ(self) -> Self {
        DbLevel(self.0 + 1)
    }

    /// Convert to `usize`.
    #[must_use]
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for DbLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ℓ{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// Transparency
// ---------------------------------------------------------------------------

/// The transparency marker on a definition. Controls when δ-unfolding is
/// permitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Transparency {
    /// `@[reducible]`: unfolded even in the strictest mode.
    Reducible,
    /// An ordinary `def`: unfolded in `Semireducible` and `All` modes.
    Semireducible,
    /// `@[irreducible]`: only unfolded in `All` mode.
    Irreducible,
}

/// The current transparency *mode* of a conversion check.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TransparencyMode {
    /// Only `@[reducible]` definitions are unfolded.
    Reducible,
    /// The default. Unfolds everything except `@[irreducible]`.
    Semireducible,
    /// Unfolds everything, including `@[irreducible]`.
    All,
}

impl TransparencyMode {
    /// Whether a definition with the given transparency may be unfolded in
    /// this mode.
    #[must_use]
    pub const fn permits(self, def: Transparency) -> bool {
        match self {
            TransparencyMode::Reducible => matches!(def, Transparency::Reducible),
            TransparencyMode::Semireducible => !matches!(def, Transparency::Irreducible),
            TransparencyMode::All => true,
        }
    }
}

// ---------------------------------------------------------------------------
// Semantic values
// ---------------------------------------------------------------------------

/// A shared reference to a semantic value.
pub type ValRef = Rc<Value>;

/// The head of a neutral term: what evaluation got stuck on.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Head {
    /// A local variable, identified by its de Bruijn level.
    Var(DbLevel),
    /// A constant that was not unfolded, applied to universe arguments.
    Const(NameId, Vec<Level>),
    /// A constructor, applied to universe arguments.
    ///
    /// Distinguished from `Const` so that the ι-reduction rule in
    /// [`Nbe::apply`] can identify the major premise of a recursor
    /// application without a second environment lookup.
    Constructor(NameId, Vec<Level>),
}

/// A semantic value.
///
/// Applications of non-lambda values (i.e. stuck applications) are
/// represented as `Neutral(head, spine)`.
#[derive(Clone, Debug)]
pub enum Value {
    /// A universe `Sort(l)`.
    Sort(Level),
    /// A dependent function type `Π (x : A). B`.
    Pi(BinderInfo, NameId, ValRef, Closure),
    /// A lambda `λ (x : A). b`.
    Lam(BinderInfo, NameId, ValRef, Closure),
    /// A literal.
    Lit(Literal),
    /// A stuck term: `head` applied to a (possibly empty) spine of arguments.
    Neutral(Head, Vec<ValRef>),
}

impl Value {
    /// A fresh neutral variable at the given level.
    #[must_use]
    pub fn fresh_var(level: DbLevel) -> ValRef {
        Rc::new(Value::Neutral(Head::Var(level), Vec::new()))
    }
}

/// A defunctionalized closure: a syntactic body paired with a semantic
/// environment in which to evaluate it when the closure is applied.
#[derive(Clone, Debug)]
pub struct Closure(Rc<ClosureInner>);

#[derive(Debug)]
struct ClosureInner {
    body: Expr,
    env: Env,
}

impl Closure {
    /// Build a closure from a body and its captured environment.
    #[must_use]
    pub fn new(body: Expr, env: Env) -> Self {
        Closure(Rc::new(ClosureInner { body, env }))
    }

    /// The syntactic body of the closure.
    #[must_use]
    pub fn body(&self) -> &Expr {
        &self.0.body
    }

    /// The captured environment.
    #[must_use]
    pub fn env(&self) -> &Env {
        &self.0.env
    }
}

// ---------------------------------------------------------------------------
// Persistent environment
// ---------------------------------------------------------------------------

/// A persistent, immutable environment mapping de Bruijn *indices* to
/// values.
///
/// The head of the list is the most recently pushed (innermost) binding;
/// index `0` therefore corresponds to the head. Extending is O(1).
#[derive(Clone)]
pub struct Env(Rc<EnvNode>);

#[derive(Debug)]
enum EnvNode {
    Empty,
    Cons(ValRef, Env),
}

impl Env {
    /// The empty environment.
    #[must_use]
    pub fn empty() -> Self {
        Env(Rc::new(EnvNode::Empty))
    }

    /// Push a binding onto the front (innermost position).
    #[must_use]
    pub fn extend(&self, v: ValRef) -> Self {
        Env(Rc::new(EnvNode::Cons(v, self.clone())))
    }

    /// Number of bindings.
    #[must_use]
    pub fn len(&self) -> usize {
        let mut n = 0;
        let mut cur = Rc::clone(&self.0);
        loop {
            match &*cur {
                EnvNode::Empty => return n,
                EnvNode::Cons(_, rest) => {
                    n += 1;
                    cur = Rc::clone(&rest.0);
                }
            }
        }
    }

    /// `true` iff the environment has no bindings.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        matches!(&*self.0, EnvNode::Empty)
    }

    /// Look up the value bound at the given de Bruijn index.
    ///
    /// # Panics
    ///
    /// Panics if `idx` is out of range. This is a programming error:
    /// well-typed terms never reference an unbound variable.
    #[must_use]
    pub fn lookup(&self, idx: DeBruijnIndex) -> ValRef {
        let mut cur = Rc::clone(&self.0);
        let mut i = idx.0 as usize;
        loop {
            match &*cur {
                EnvNode::Empty => {
                    panic!("unbound de Bruijn index {idx} (env size exceeded)")
                }
                EnvNode::Cons(v, rest) => {
                    if i == 0 {
                        return Rc::clone(v);
                    }
                    i -= 1;
                    cur = Rc::clone(&rest.0);
                }
            }
        }
    }

    /// Build the environment for a context of size `n`: fresh neutral
    /// variables at levels `0 .. n`, with the innermost (highest level) at
    /// the head.
    #[must_use]
    pub fn of_context(n: usize) -> Self {
        let mut env = Env::empty();
        for i in 0..n {
            env = env.extend(Value::fresh_var(DbLevel(i as u32)));
        }
        env
    }
}

impl fmt::Debug for Env {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Env(len={})", self.len())
    }
}

// ---------------------------------------------------------------------------
// Constant environment
// ---------------------------------------------------------------------------

/// A view of a constant declaration, provided by the caller's environment.
#[derive(Clone, Debug)]
pub struct ConstDecl {
    /// The constant's name.
    pub name: NameId,
    /// Universe parameters introduced by the declaration, in order.
    pub universe_params: Vec<UniverseParamId>,
    /// The constant's type.
    pub ty: Rc<Expr>,
    /// The constant's body, if it is a definition (rather than an axiom or
    /// an opaque declaration).
    pub body: Option<Rc<Expr>>,
    /// The definition's transparency marker.
    pub transparency: Transparency,
    /// `true` iff this constant is a constructor of an inductive family.
    pub is_constructor: bool,
    /// Present iff this constant is a recursor.
    pub recursor_info: Option<RecursorInfo>,
    /// When set, this constant has a `Quot.lift`-style β-rule that fires
    /// on `Quot.mk`-headed arguments.
    pub quotient_lift: Option<QuotientLiftInfo>,
}

/// A lightweight view of a recursor, sufficient to drive ι-reduction.
#[derive(Clone, Debug)]
pub struct RecursorInfo {
    /// Number of uniform parameters.
    pub num_params: u32,
    /// Number of indices.
    pub num_indices: u32,
    /// Number of motives (usually `1`).
    pub num_motives: u32,
    /// Number of minor premises.
    pub num_minors: u32,
    /// The ι-reduction rules, one per constructor.
    pub rules: Vec<RecursorRuleInfo>,
}

/// A single ι-reduction rule.
#[derive(Clone, Debug)]
pub struct RecursorRuleInfo {
    /// The constructor this rule fires for.
    pub constructor: NameId,
    /// Number of fields the rule binds.
    pub num_fields: u32,
    /// The right-hand side of the rule. Its context is, from outermost to
    /// innermost: parameters, motives, minors, indices, constructor fields.
    pub rhs: Rc<Expr>,
}

/// Information about a `Quot.lift`-style eliminator.
#[derive(Clone, Debug)]
pub struct QuotientLiftInfo {
    /// The `Quot.mk` recognised by this eliminator.
    pub mk: NameId,
    /// Total number of arguments the eliminator expects.
    pub arity: u32,
    /// Index (0-based) of the function argument.
    pub func_index: u32,
    /// Index (0-based) of the quotient-value argument.
    pub major_index: u32,
}

/// A trait for resolving constants to their declarations.
pub trait ConstEnv {
    /// Look up a constant declaration by name.
    fn lookup(&self, name: NameId) -> Option<Rc<ConstDecl>>;
}

/// A constant environment containing no declarations. Useful for tests and
/// for pure-λ-calculus experiments.
#[derive(Debug, Default, Clone, Copy)]
pub struct EmptyConstEnv;

impl ConstEnv for EmptyConstEnv {
    fn lookup(&self, _: NameId) -> Option<Rc<ConstDecl>> {
        None
    }
}

// ---------------------------------------------------------------------------
// The NbE engine
// ---------------------------------------------------------------------------

/// The NbE engine: bundles a constant environment and a transparency mode.
#[derive(Clone, Copy)]
pub struct Nbe<'a> {
    consts: &'a dyn ConstEnv,
    mode: TransparencyMode,
}

impl<'a> Nbe<'a> {
    /// Construct an NbE engine.
    #[must_use]
    pub fn new(consts: &'a dyn ConstEnv, mode: TransparencyMode) -> Self {
        Nbe { consts, mode }
    }

    /// The current transparency mode.
    #[must_use]
    pub fn mode(&self) -> TransparencyMode {
        self.mode
    }

    // -----------------------------------------------------------------
    // Evaluation
    // -----------------------------------------------------------------

    /// Evaluate `expr` in `env`, producing a semantic value.
    #[must_use]
    pub fn eval(&self, env: &Env, expr: &Expr) -> ValRef {
        match expr {
            Expr::Sort(l) => Rc::new(Value::Sort(l.normalize())),
            Expr::Var(idx) => env.lookup(*idx),
            Expr::Const(name, levels) => self.eval_const(*name, levels),
            Expr::App(f, a) => {
                let fv = self.eval(env, f);
                let av = self.eval(env, a);
                self.apply(fv, av)
            }
            Expr::Lam(info, name, dom, body) => {
                let dom_v = self.eval(env, dom);
                let clo = Closure::new((**body).clone(), env.clone());
                Rc::new(Value::Lam(*info, *name, dom_v, clo))
            }
            Expr::Pi(info, name, dom, cod) => {
                let dom_v = self.eval(env, dom);
                let clo = Closure::new((**cod).clone(), env.clone());
                Rc::new(Value::Pi(*info, *name, dom_v, clo))
            }
            Expr::Let(_, _ty, val, body) => {
                // The bound value is inlined into the body. Its type is
                // irrelevant for evaluation.
                let v = self.eval(env, val);
                let env2 = env.extend(v);
                self.eval(&env2, body)
            }
            Expr::Lit(l) => Rc::new(Value::Lit(l.clone())),
        }
    }

    fn eval_const(&self, name: NameId, levels: &[Level]) -> ValRef {
        let normalized: Vec<Level> = levels.iter().map(Level::normalize).collect();

        let Some(decl) = self.consts.lookup(name) else {
            return Rc::new(Value::Neutral(Head::Const(name, normalized), Vec::new()));
        };
        let head = if decl.is_constructor {
            Head::Constructor(name, normalized.clone())
        } else {
            Head::Const(name, normalized.clone())
        };
        if !self.mode.permits(decl.transparency) {
            return Rc::new(Value::Neutral(head, Vec::new()));
        }
        let Some(body) = decl.body.as_ref() else {
            return Rc::new(Value::Neutral(head, Vec::new()));
        };

        let body_expr: Rc<Expr> = if decl.universe_params.is_empty() {
            Rc::clone(body)
        } else {
            Rc::new(substitute_levels(body, &decl.universe_params, &normalized))
        };
        // A constant's body is closed: evaluate in the empty environment.
        self.eval(&Env::empty(), &body_expr)
    }

    /// Apply a function value to an argument value.
    ///
    /// # Panics
    ///
    /// Panics if `f` is not a function (i.e. not a lambda and not neutral).
    /// This is unreachable for well-typed terms.
    #[must_use]
    pub fn apply(&self, f: ValRef, a: ValRef) -> ValRef {
        match &*f {
            Value::Lam(_, _, _, clo) => self.closure_apply(clo, a),
            Value::Neutral(head, spine) => {
                let mut new_spine: Vec<ValRef> = Vec::with_capacity(spine.len() + 1);
                new_spine.extend(spine.iter().cloned());
                new_spine.push(a);

                if let Some(reduced) = self.try_iota(head, &new_spine) {
                    return reduced;
                }
                if let Some(reduced) = self.try_quot_lift(head, &new_spine) {
                    return reduced;
                }
                Rc::new(Value::Neutral(head.clone(), new_spine))
            }
            other => panic!("nbe::apply: cannot apply non-function value: {other:?}"),
        }
    }

    /// Attempt ι-reduction on a fully-applied recursor application.
    ///
    /// Returns `Some(v)` if the spine's final argument is a constructor
    /// application and a rule fires; `None` otherwise (leaving the application
    /// stuck).
    fn try_iota(&self, head: &Head, spine: &[ValRef]) -> Option<ValRef> {
        let (rec_name, actual_levels) = match head {
            Head::Const(n, ls) => (*n, ls.as_slice()),
            Head::Constructor(_, _) | Head::Var(_) => return None,
        };
        let decl = self.consts.lookup(rec_name)?;
        let info = decl.recursor_info.as_ref()?;

        let total_arity =
            (info.num_params + info.num_motives + info.num_minors + info.num_indices + 1) as usize;
        if spine.len() != total_arity {
            return None;
        }

        // The major premise is the final argument.
        let major = &spine[total_arity - 1];
        let (ctor_name, ctor_args) = match &**major {
            Value::Neutral(Head::Constructor(n, _), args) => (*n, args.clone()),
            _ => return None,
        };

        let rule = info.rules.iter().find(|r| r.constructor == ctor_name)?;

        // Build the environment for the RHS. Layout (bottom to top):
        //   parameters, motives, minors, indices, fields.
        // The spine (excluding the major) supplies the first four groups.
        let np = info.num_params as usize;
        let nm = info.num_motives as usize;
        let nmin = info.num_minors as usize;
        let ni = info.num_indices as usize;

        // The constructor's first `num_params` arguments are the uniform
        // parameters, already bound above; only the remaining arguments are
        // the rule's fields. Anything else (under- or over-applied major)
        // is malformed: leave the application stuck rather than run the
        // rule with a shifted environment, which would produce a wrong term.
        if ctor_args.len() != np + rule.num_fields as usize {
            return None;
        }

        let mut env = Env::empty();
        for v in spine.iter().take(np) {
            env = env.extend(Rc::clone(v));
        }
        for v in spine.iter().skip(np).take(nm) {
            env = env.extend(Rc::clone(v));
        }
        for v in spine.iter().skip(np + nm).take(nmin) {
            env = env.extend(Rc::clone(v));
        }
        for v in spine.iter().skip(np + nm + nmin).take(ni) {
            env = env.extend(Rc::clone(v));
        }
        // Skip the uniform parameters: they are already bound above.
        for v in ctor_args.iter().skip(np) {
            env = env.extend(Rc::clone(v));
        }

        // Substitute the recursor's universe parameters into the RHS.
        let rhs: Rc<Expr> = if decl.universe_params.is_empty() {
            Rc::clone(&rule.rhs)
        } else {
            Rc::new(substitute_levels(
                &rule.rhs,
                &decl.universe_params,
                actual_levels,
            ))
        };

        Some(self.eval(&env, &rhs))
    }

    /// Attempt `Quot.lift` β-reduction on a fully-applied eliminator.
    ///
    /// Fires when the eliminator's major argument is `Quot.mk`-headed; then
    /// `Quot.lift A R B f h (Quot.mk A R a) ≡ f a`. The function argument is
    /// applied to the *last* argument of the `Quot.mk` application (the
    /// wrapped value, past the type and relation arguments).
    fn try_quot_lift(&self, head: &Head, spine: &[ValRef]) -> Option<ValRef> {
        let name = match head {
            Head::Const(n, _) => *n,
            Head::Constructor(_, _) | Head::Var(_) => return None,
        };
        let decl = self.consts.lookup(name)?;
        let info = decl.quotient_lift.as_ref()?;
        if spine.len() != info.arity as usize {
            return None;
        }
        let major = &spine[info.major_index as usize];
        let a = match &**major {
            Value::Neutral(Head::Constructor(mk_name, _), args)
                if *mk_name == info.mk && !args.is_empty() =>
            {
                Rc::clone(args.last().unwrap())
            }
            _ => return None,
        };
        let f = Rc::clone(&spine[info.func_index as usize]);
        Some(self.apply(f, a))
    }

    fn closure_apply(&self, clo: &Closure, arg: ValRef) -> ValRef {
        let new_env = clo.0.env.extend(arg);
        self.eval(&new_env, &clo.0.body)
    }

    /// Apply a semantic closure to an argument.
    ///
    /// The public counterpart of the internal closure-application routine,
    /// exposed so the type checker can apply the codomain closure of a `Pi`
    /// value without first wrapping it back into a `Value`.
    #[must_use]
    pub fn apply_closure(&self, clo: &Closure, arg: ValRef) -> ValRef {
        self.closure_apply(clo, arg)
    }

    // -----------------------------------------------------------------
    // Quotation / readback
    // -----------------------------------------------------------------

    /// Read back a value into a canonical syntactic expression, at the given
    /// context size.
    ///
    /// The returned expression is in βη-normal form and uses de Bruijn
    /// indices relative to the context of size `level`.
    #[must_use]
    pub fn quote(&self, level: usize, v: &ValRef) -> Expr {
        match &**v {
            Value::Sort(l) => Expr::Sort(l.clone()),
            Value::Lit(l) => Expr::Lit(l.clone()),
            Value::Pi(info, name, dom, clo) => {
                let dom_e = self.quote(level, dom);
                let body_e = self.quote_closure(level, clo);
                Expr::Pi(*info, *name, Box::new(dom_e), Box::new(body_e))
            }
            Value::Lam(info, name, dom, clo) => {
                let dom_e = self.quote(level, dom);
                let body_e = self.quote_closure(level, clo);
                Expr::Lam(*info, *name, Box::new(dom_e), Box::new(body_e))
            }
            Value::Neutral(head, spine) => {
                let head_e = match head {
                    Head::Var(lvl) => {
                        // level is the current context size.
                        // The variable at de Bruijn *level* L has index
                        // (context_size - L - 1).
                        let idx = (level as u32)
                            .checked_sub(lvl.0)
                            .and_then(|x| x.checked_sub(1))
                            .unwrap_or_else(|| {
                                panic!(
                                    "nbe::quote: level {lvl} out of range at context size {level}"
                                )
                            });
                        Expr::Var(DeBruijnIndex(idx))
                    }
                    Head::Const(name, levels) | Head::Constructor(name, levels) => {
                        Expr::Const(*name, levels.clone())
                    }
                };
                spine
                    .iter()
                    .fold(head_e, |acc, a| Expr::app(acc, self.quote(level, a)))
            }
        }
    }

    /// Quote the body of a closure by applying it to a fresh variable.
    fn quote_closure(&self, level: usize, clo: &Closure) -> Expr {
        let v = Value::fresh_var(DbLevel(level as u32));
        let body = self.closure_apply(clo, v);
        self.quote(level + 1, &body)
    }

    // -----------------------------------------------------------------
    // Conversion
    // -----------------------------------------------------------------

    /// Decide definitional equality of two values at the given context size.
    ///
    /// Implements βη-conversion on already-evaluated values. δ- and
    /// ι-reduction are handled by [`Nbe::eval`], which produces the
    /// fully-unfolded values in the first place.
    #[must_use]
    pub fn convert(&self, level: usize, a: &ValRef, b: &ValRef) -> bool {
        // Fast path: shared pointers are trivially equal.
        if Rc::ptr_eq(a, b) {
            return true;
        }
        match (&**a, &**b) {
            // Universe sorts: compare levels up to normalization.
            (Value::Sort(l1), Value::Sort(l2)) => l1.normalize() == l2.normalize(),

            // Literals: structural equality.
            (Value::Lit(l1), Value::Lit(l2)) => l1 == l2,

            // Pi types: binder info and components must match.
            (Value::Pi(i1, _, d1, c1), Value::Pi(i2, _, d2, c2)) => {
                i1 == i2 && self.convert(level, d1, d2) && self.convert_closure(level, c1, c2)
            }

            // Lambdas: same shape, defeq domain, defeq body.
            (Value::Lam(i1, _, d1, c1), Value::Lam(i2, _, d2, c2)) => {
                i1 == i2 && self.convert(level, d1, d2) && self.convert_closure(level, c1, c2)
            }

            // η for functions: a lambda is equal to a non-lambda iff their
            // applications to a fresh variable are equal.
            (Value::Lam(_, _, _, c1), _) => {
                let v = Value::fresh_var(DbLevel(level as u32));
                let lhs = self.closure_apply(c1, Rc::clone(&v));
                let rhs = self.apply(Rc::clone(b), v);
                self.convert(level + 1, &lhs, &rhs)
            }
            (_, Value::Lam(_, _, _, c2)) => {
                let v = Value::fresh_var(DbLevel(level as u32));
                let lhs = self.apply(Rc::clone(a), Rc::clone(&v));
                let rhs = self.closure_apply(c2, v);
                self.convert(level + 1, &lhs, &rhs)
            }

            // Neutral terms: same head and pointwise-equal spines.
            (Value::Neutral(h1, s1), Value::Neutral(h2, s2)) => {
                h1 == h2
                    && s1.len() == s2.len()
                    && s1
                        .iter()
                        .zip(s2.iter())
                        .all(|(x, y)| self.convert(level, x, y))
            }

            _ => false,
        }
    }

    fn convert_closure(&self, level: usize, c1: &Closure, c2: &Closure) -> bool {
        let v = Value::fresh_var(DbLevel(level as u32));
        let lhs = self.closure_apply(c1, Rc::clone(&v));
        let rhs = self.closure_apply(c2, v);
        self.convert(level + 1, &lhs, &rhs)
    }
}

// ---------------------------------------------------------------------------
// Universe substitution
// ---------------------------------------------------------------------------

/// Substitute `actuals[i]` for `params[i]` throughout `expr`.
///
/// Parameters not present in `params` are left untouched. If `params` is
/// shorter than `actuals`, the excess actuals are ignored; if an actual is
/// missing, the parameter is left in place.
#[must_use]
pub fn substitute_levels(expr: &Expr, params: &[UniverseParamId], actuals: &[Level]) -> Expr {
    match expr {
        Expr::Sort(l) => Expr::Sort(subst_level(l, params, actuals)),
        Expr::Var(idx) => Expr::Var(*idx),
        Expr::Const(name, levels) => Expr::Const(
            *name,
            levels
                .iter()
                .map(|l| subst_level(l, params, actuals))
                .collect(),
        ),
        Expr::App(f, a) => Expr::App(
            Box::new(substitute_levels(f, params, actuals)),
            Box::new(substitute_levels(a, params, actuals)),
        ),
        Expr::Lam(info, name, dom, body) => Expr::Lam(
            *info,
            *name,
            Box::new(substitute_levels(dom, params, actuals)),
            Box::new(substitute_levels(body, params, actuals)),
        ),
        Expr::Pi(info, name, dom, cod) => Expr::Pi(
            *info,
            *name,
            Box::new(substitute_levels(dom, params, actuals)),
            Box::new(substitute_levels(cod, params, actuals)),
        ),
        Expr::Let(name, ty, val, body) => Expr::Let(
            *name,
            Box::new(substitute_levels(ty, params, actuals)),
            Box::new(substitute_levels(val, params, actuals)),
            Box::new(substitute_levels(body, params, actuals)),
        ),
        Expr::Lit(l) => Expr::Lit(l.clone()),
    }
}

/// Substitute `actuals[i]` for `params[i]` in a single level.
fn subst_level(level: &Level, params: &[UniverseParamId], actuals: &[Level]) -> Level {
    match level {
        Level::Zero => Level::Zero,
        Level::Param(p) => params
            .iter()
            .position(|q| q == p)
            .and_then(|i| actuals.get(i).cloned())
            .unwrap_or(Level::Param(*p)),
        Level::Succ(l) => subst_level(l, params, actuals).succ(),
        Level::Max(a, b) => Level::max(
            subst_level(a, params, actuals),
            subst_level(b, params, actuals),
        ),
        Level::IMax(a, b) => Level::imax(
            subst_level(a, params, actuals),
            subst_level(b, params, actuals),
        ),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::level::UniverseParamId;
    use crate::name::NameTable;
    use alloc::vec;

    /// A minimal [`ConstEnv`] backed by a linear list. Adequate for tests.
    #[derive(Default)]
    struct TestConsts {
        decls: Vec<Rc<ConstDecl>>,
    }

    impl TestConsts {
        fn insert(&mut self, decl: ConstDecl) -> NameId {
            let name = decl.name;
            self.decls.push(Rc::new(decl));
            name
        }
    }

    impl ConstEnv for TestConsts {
        fn lookup(&self, name: NameId) -> Option<Rc<ConstDecl>> {
            self.decls.iter().find(|d| d.name == name).cloned()
        }
    }

    fn engine(consts: &dyn ConstEnv, mode: TransparencyMode) -> Nbe<'_> {
        Nbe::new(consts, mode)
    }

    fn var(idx: u32) -> Expr {
        Expr::Var(DeBruijnIndex(idx))
    }

    /// The polymorphic identity body `λ (A : Sort(u)). λ (a : #0). #0`,
    /// using de Bruijn indices (`#1` is `A` under the inner binder).
    fn poly_id_body(u: UniverseParamId) -> Expr {
        let mut t = NameTable::new();
        let a_ty = t.intern("A");
        let a_tm = t.intern("a");
        let sort_u = Expr::Sort(Level::Param(u));
        // λ (A : Sort u). λ (a : #0). #0
        Expr::Lam(
            BinderInfo::Default,
            a_ty,
            Box::new(sort_u),
            Box::new(Expr::Lam(
                BinderInfo::Default,
                a_tm,
                Box::new(var(0)),
                Box::new(var(0)),
            )),
        )
    }

    #[test]
    fn dblevel_basics() {
        assert_eq!(DbLevel::zero(), DbLevel(0));
        assert_eq!(DbLevel::zero().succ(), DbLevel(1));
        assert_eq!(DbLevel(3).as_usize(), 3);
        assert_eq!(alloc::format!("{}", DbLevel(2)), "ℓ2");
    }

    #[test]
    fn transparency_permits() {
        use Transparency::{Irreducible, Reducible, Semireducible};
        use TransparencyMode as M;
        assert!(M::Reducible.permits(Reducible));
        assert!(!M::Reducible.permits(Semireducible));
        assert!(!M::Reducible.permits(Irreducible));
        assert!(M::Semireducible.permits(Reducible));
        assert!(M::Semireducible.permits(Semireducible));
        assert!(!M::Semireducible.permits(Irreducible));
        assert!(M::All.permits(Irreducible));
    }

    #[test]
    fn env_extend_lookup_len() {
        let env = Env::of_context(2);
        assert_eq!(env.len(), 2);
        assert!(!env.is_empty());
        assert!(Env::empty().is_empty());
        // Index 0 is the innermost binding: fresh var at level 1.
        let v0 = env.lookup(DeBruijnIndex(0));
        let v1 = env.lookup(DeBruijnIndex(1));
        assert!(matches!(&*v0, Value::Neutral(Head::Var(DbLevel(1)), _)));
        assert!(matches!(&*v1, Value::Neutral(Head::Var(DbLevel(0)), _)));
    }

    #[test]
    fn eval_beta_reduces() {
        let consts = EmptyConstEnv;
        let nbe = engine(&consts, TransparencyMode::Semireducible);
        let mut names = NameTable::new();
        let x = names.intern("x");
        // (λ x : Prop. x) applied to a free variable `#0` in a size-1 ctx.
        let id = Expr::lam(x, Expr::prop(), var(0));
        let app = Expr::app(id, var(0));
        let env = Env::of_context(1);
        let v = nbe.eval(&env, &app);
        // β: result is the argument, i.e. neutral var at level 0.
        assert!(matches!(&*v, Value::Neutral(Head::Var(DbLevel(0)), _)));
    }

    #[test]
    fn eval_let_inlines() {
        let consts = EmptyConstEnv;
        let nbe = engine(&consts, TransparencyMode::Semireducible);
        let mut names = NameTable::new();
        let x = names.intern("x");
        // let x : Prop := Prop; x  ~~>  Prop
        let e = Expr::Let(
            x,
            Box::new(Expr::prop()),
            Box::new(Expr::prop()),
            Box::new(var(0)),
        );
        let v = nbe.eval(&Env::empty(), &e);
        assert!(matches!(&*v, Value::Sort(_)));
    }

    #[test]
    fn eval_sort_normalizes() {
        let consts = EmptyConstEnv;
        let nbe = engine(&consts, TransparencyMode::Semireducible);
        let l = Level::max(Level::zero(), Level::zero().succ());
        let v = nbe.eval(&Env::empty(), &Expr::Sort(l));
        match &*v {
            Value::Sort(got) => assert_eq!(*got, Level::zero().succ()),
            other => panic!("expected sort, got {other:?}"),
        }
    }

    #[test]
    fn quote_roundtrip_closed_id() {
        let consts = EmptyConstEnv;
        let nbe = engine(&consts, TransparencyMode::Semireducible);
        let mut names = NameTable::new();
        let x = names.intern("x");
        let id = Expr::lam(x, Expr::prop(), var(0));
        let v = nbe.eval(&Env::empty(), &id);
        let back = nbe.quote(0, &v);
        assert_eq!(back, id);
    }

    #[test]
    fn quote_neutral_var_index_math() {
        let consts = EmptyConstEnv;
        let nbe = engine(&consts, TransparencyMode::Semireducible);
        // In a context of size 2, level 0 is the outermost: index 1.
        let v = Value::fresh_var(DbLevel(0));
        assert_eq!(nbe.quote(2, &v), var(1));
        let v = Value::fresh_var(DbLevel(1));
        assert_eq!(nbe.quote(2, &v), var(0));
    }

    #[test]
    fn convert_beta_equal() {
        let consts = EmptyConstEnv;
        let nbe = engine(&consts, TransparencyMode::Semireducible);
        let mut names = NameTable::new();
        let x = names.intern("x");
        let id = Expr::lam(x, Expr::prop(), var(0));
        let env = Env::of_context(1);
        // `(λx.x) #0` vs `#0`: β-equal.
        let a = nbe.eval(&env, &Expr::app(id, var(0)));
        let b = nbe.eval(&env, &var(0));
        assert!(nbe.convert(1, &a, &b));
    }

    #[test]
    fn convert_eta() {
        let consts = EmptyConstEnv;
        let nbe = engine(&consts, TransparencyMode::Semireducible);
        let mut names = NameTable::new();
        let x = names.intern("x");
        let env = Env::of_context(1);
        // `f` (neutral) vs `λ x. f x`: η-equal.
        let f = nbe.eval(&env, &var(0));
        let eta = Expr::lam(x, Expr::prop(), Expr::app(var(1), var(0)));
        let g = nbe.eval(&env, &eta);
        assert!(nbe.convert(1, &f, &g));
        assert!(nbe.convert(1, &g, &f));
    }

    #[test]
    fn convert_distinguishes() {
        let consts = EmptyConstEnv;
        let nbe = engine(&consts, TransparencyMode::Semireducible);
        let env = Env::of_context(2);
        let a = nbe.eval(&env, &var(0));
        let b = nbe.eval(&env, &var(1));
        assert!(!nbe.convert(2, &a, &b));
        // Sorts at different levels are not equal.
        let s0 = nbe.eval(&Env::empty(), &Expr::prop());
        let s1 = nbe.eval(&Env::empty(), &Expr::type0());
        assert!(!nbe.convert(0, &s0, &s1));
        assert!(nbe.convert(0, &s0, &s0));
    }

    #[test]
    fn convert_sorts_up_to_normalization() {
        let consts = EmptyConstEnv;
        let nbe = engine(&consts, TransparencyMode::Semireducible);
        let l = Level::max(Level::zero(), Level::zero().succ());
        let a = Rc::new(Value::Sort(l));
        let b = Rc::new(Value::Sort(Level::zero().succ()));
        assert!(nbe.convert(0, &a, &b));
    }

    #[test]
    fn convert_ptr_eq_fast_path() {
        let consts = EmptyConstEnv;
        let nbe = engine(&consts, TransparencyMode::Semireducible);
        let v = Value::fresh_var(DbLevel(0));
        assert!(nbe.convert(1, &v, &v));
    }

    #[test]
    fn delta_unfolds_definition() {
        let mut names = NameTable::new();
        let c = names.intern("c");
        let mut consts = TestConsts::default();
        consts.insert(ConstDecl {
            name: c,
            universe_params: vec![],
            ty: Rc::new(Expr::prop()),
            body: Some(Rc::new(Expr::prop())),
            transparency: Transparency::Semireducible,
            is_constructor: false,
            recursor_info: None,
            quotient_lift: None,
        });
        let nbe = engine(&consts, TransparencyMode::Semireducible);
        let v = nbe.eval(&Env::empty(), &Expr::Const(c, vec![]));
        assert!(matches!(&*v, Value::Sort(_)));
    }

    #[test]
    fn delta_respects_transparency() {
        let mut names = NameTable::new();
        let c = names.intern("c");
        let mut consts = TestConsts::default();
        consts.insert(ConstDecl {
            name: c,
            universe_params: vec![],
            ty: Rc::new(Expr::prop()),
            body: Some(Rc::new(Expr::prop())),
            transparency: Transparency::Irreducible,
            is_constructor: false,
            recursor_info: None,
            quotient_lift: None,
        });
        // Default mode: no unfold — stuck neutral.
        let nbe = engine(&consts, TransparencyMode::Semireducible);
        let v = nbe.eval(&Env::empty(), &Expr::Const(c, vec![]));
        assert!(matches!(&*v, Value::Neutral(Head::Const(_, _), _)));
        // All mode: unfolds.
        let nbe_all = engine(&consts, TransparencyMode::All);
        let v2 = nbe_all.eval(&Env::empty(), &Expr::Const(c, vec![]));
        assert!(matches!(&*v2, Value::Sort(_)));
        // Reducible mode does not unfold a semireducible def.
        let mut names2 = NameTable::new();
        let d = names2.intern("d");
        let mut consts2 = TestConsts::default();
        consts2.insert(ConstDecl {
            name: d,
            universe_params: vec![],
            ty: Rc::new(Expr::prop()),
            body: Some(Rc::new(Expr::prop())),
            transparency: Transparency::Semireducible,
            is_constructor: false,
            recursor_info: None,
            quotient_lift: None,
        });
        let nbe_red = engine(&consts2, TransparencyMode::Reducible);
        let v3 = nbe_red.eval(&Env::empty(), &Expr::Const(d, vec![]));
        assert!(matches!(&*v3, Value::Neutral(Head::Const(_, _), _)));
    }

    #[test]
    fn delta_unknown_const_is_neutral() {
        let consts = EmptyConstEnv;
        let nbe = engine(&consts, TransparencyMode::All);
        let mut names = NameTable::new();
        let c = names.intern("missing");
        let v = nbe.eval(&Env::empty(), &Expr::Const(c, vec![]));
        match &*v {
            Value::Neutral(Head::Const(name, _), spine) => {
                assert_eq!(*name, c);
                assert!(spine.is_empty());
            }
            other => panic!("expected neutral const, got {other:?}"),
        }
    }

    #[test]
    fn delta_axiom_is_neutral() {
        let mut names = NameTable::new();
        let ax = names.intern("ax");
        let mut consts = TestConsts::default();
        consts.insert(ConstDecl {
            name: ax,
            universe_params: vec![],
            ty: Rc::new(Expr::prop()),
            body: None,
            transparency: Transparency::Semireducible,
            is_constructor: false,
            recursor_info: None,
            quotient_lift: None,
        });
        let nbe = engine(&consts, TransparencyMode::All);
        let v = nbe.eval(&Env::empty(), &Expr::Const(ax, vec![]));
        assert!(matches!(&*v, Value::Neutral(Head::Const(_, _), _)));
    }

    #[test]
    fn delta_instantiates_universe_params() {
        let u = UniverseParamId(41);
        let mut names = NameTable::new();
        let id = names.intern("id");
        let mut consts = TestConsts::default();
        consts.insert(ConstDecl {
            name: id,
            universe_params: vec![u],
            ty: Rc::new(Expr::prop()),
            body: Some(Rc::new(poly_id_body(u))),
            transparency: Transparency::Semireducible,
            is_constructor: false,
            recursor_info: None,
            quotient_lift: None,
        });
        let nbe = engine(&consts, TransparencyMode::Semireducible);
        // Instantiate `id.{0}`: the domain `Sort(u)` must become `Sort(0)`.
        let v = nbe.eval(&Env::empty(), &Expr::Const(id, vec![Level::zero()]));
        let back = nbe.quote(0, &v);
        match back {
            Expr::Lam(_, _, dom, _) => assert_eq!(*dom, Expr::Sort(Level::zero())),
            other => panic!("expected lam, got {other:?}"),
        }
    }

    #[test]
    fn substitute_levels_replaces_params() {
        let u = UniverseParamId(51);
        let v = UniverseParamId(52);
        let e = Expr::Sort(Level::max(Level::Param(u), Level::Param(v).succ()));
        let got = substitute_levels(&e, &[u], &[Level::zero()]);
        assert_eq!(
            got,
            Expr::Sort(Level::max(Level::zero(), Level::Param(v).succ()))
        );
    }

    #[test]
    fn substitute_levels_missing_actual_keeps_param() {
        let u = UniverseParamId(53);
        let e = Expr::Sort(Level::Param(u));
        // No actuals: parameter survives.
        assert_eq!(substitute_levels(&e, &[u], &[]), e);
        // Unknown parameter untouched.
        let w = UniverseParamId(54);
        let e2 = Expr::Sort(Level::Param(w));
        assert_eq!(substitute_levels(&e2, &[u], &[Level::zero()]), e2);
    }

    #[test]
    fn substitute_levels_descends_into_const() {
        let u = UniverseParamId(55);
        let mut names = NameTable::new();
        let c = names.intern("c");
        let e = Expr::Const(c, vec![Level::Param(u)]);
        let got = substitute_levels(&e, &[u], &[Level::zero().succ()]);
        assert_eq!(got, Expr::Const(c, vec![Level::zero().succ()]));
    }

    #[test]
    fn stuck_application_spine_quotes() {
        let consts = EmptyConstEnv;
        let nbe = engine(&consts, TransparencyMode::Semireducible);
        let env = Env::of_context(2);
        // `#1 #0` with `#1, #0` the two context vars.
        let e = Expr::app(var(1), var(0));
        let v = nbe.eval(&env, &e);
        let back = nbe.quote(2, &v);
        assert_eq!(back, e);
    }
}
