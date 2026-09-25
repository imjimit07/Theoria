//! Universe level algebra.
//!
//! Implements the canonical level algebra used by CIC-style kernels. A universe
//! level is an expression built from:
//!
//! * `Zero`   — the bottom level.
//! * `Succ`   — the successor of a level.
//! * `Max`    — the pointwise maximum of two levels.
//! * `IMax`   — the *impredicative* maximum: `IMax(a, b)` is `0` when `b = 0`,
//!   and `Max(a, b)` otherwise. This operation encodes the impredicativity of
//!   `Prop`.
//! * `Param`  — a universally quantified universe parameter.
//!
//! Levels are not required to be in normal form; [`Level::normalize`] produces
//! a canonical representative using the standard algebraic laws.
//!
//! ## Soundness note
//!
//! The type checker relies on the *normalized* form of a level to compare
//! universe instantiations. Any bug in [`Level::normalize`] or in the
//! constraint graph can potentially admit Girard's paradox. This module is
//! therefore treated as part of the TCB and must remain small and total.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::fmt;
use core::sync::atomic::{AtomicU32, Ordering};

/// Identifier for a universe parameter variable.
///
/// Universe parameters are introduced by polymorphic declarations (e.g.
/// `def id.{u} (A : Type u) (a : A) : A := a`) and instantiated by callers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct UniverseParamId(pub u32);

impl UniverseParamId {
    /// Allocate a globally fresh universe parameter identifier.
    ///
    /// The generator is process-wide and monotonic. It is intentionally not
    /// reset between elaboration sessions to keep identifiers unique across
    /// incremental compilation steps.
    #[must_use]
    pub fn fresh() -> Self {
        static NEXT: AtomicU32 = AtomicU32::new(1);
        Self(NEXT.fetch_add(1, Ordering::Relaxed))
    }
}

impl fmt::Display for UniverseParamId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Render as `u0`, `u1`, ... to keep diagnostics readable.
        write!(f, "u{}", self.0)
    }
}

/// A universe level expression.
///
/// The tree is intentionally explicit (no sharing/arena indirection yet) so
/// that the algebra is easy to audit. Arena representation will arrive when
/// the NbE engine requires pointer-stable terms.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Level {
    /// The bottom universe, `Prop`/`Type 0`.
    Zero,
    /// `l + 1`.
    Succ(Box<Level>),
    /// `max(l, r)`.
    Max(Box<Level>, Box<Level>),
    /// `imax(l, r)` — impredicative maximum.
    IMax(Box<Level>, Box<Level>),
    /// A universe parameter.
    Param(UniverseParamId),
}

impl Level {
    /// Construct `Zero`.
    #[must_use]
    pub const fn zero() -> Self {
        Level::Zero
    }

    /// Construct `self + 1`.
    #[must_use]
    pub fn succ(self) -> Self {
        Level::Succ(Box::new(self))
    }

    /// Construct `max(a, b)`.
    #[must_use]
    pub fn max(a: Self, b: Self) -> Self {
        Level::Max(Box::new(a), Box::new(b))
    }

    /// Construct `imax(a, b)`.
    #[must_use]
    pub fn imax(a: Self, b: Self) -> Self {
        Level::IMax(Box::new(a), Box::new(b))
    }

    /// Construct a parameter reference.
    #[must_use]
    pub const fn param(id: UniverseParamId) -> Self {
        Level::Param(id)
    }

    /// `true` iff the level is syntactically `Zero`.
    #[must_use]
    pub const fn is_zero(&self) -> bool {
        matches!(self, Level::Zero)
    }

    /// Return the predecessor if the level is a syntactic successor.
    #[must_use]
    pub fn pred(&self) -> Option<&Level> {
        match self {
            Level::Succ(l) => Some(l),
            _ => None,
        }
    }

    /// Normalize the level into a canonical form.
    ///
    /// Rewrite rules:
    ///
    /// ```text
    /// max(0, x)        = x
    /// max(x, 0)        = x
    /// max(x, x)        = x
    /// imax(x, 0)       = 0
    /// imax(0, x)       = x
    /// imax(x, x)       = x
    /// ```
    ///
    /// The normalization is idempotent and total.
    #[must_use]
    pub fn normalize(&self) -> Level {
        match self {
            Level::Zero => Level::Zero,
            Level::Param(p) => Level::Param(*p),
            Level::Succ(l) => l.normalize().succ(),
            Level::Max(a, b) => {
                let a = a.normalize();
                let b = b.normalize();
                match (&a, &b) {
                    (Level::Zero, _) => b,
                    (_, Level::Zero) => a,
                    _ if a == b => a,
                    _ => Level::Max(Box::new(a), Box::new(b)),
                }
            }
            Level::IMax(a, b) => {
                let a = a.normalize();
                let b = b.normalize();
                // impredicativity: imax(a, 0) = 0
                if b.is_zero() {
                    return Level::Zero;
                }
                match (&a, &b) {
                    (Level::Zero, _) => b,
                    (_, Level::Zero) => Level::Zero,
                    _ if a == b => a,
                    _ => Level::IMax(Box::new(a), Box::new(b)),
                }
            }
        }
    }

    /// Free universe parameters occurring in this level, in ascending order.
    #[must_use]
    pub fn params(&self) -> Vec<UniverseParamId> {
        let mut out = Vec::new();
        self.collect_params(&mut out);
        out.sort_unstable();
        out.dedup();
        out
    }

    fn collect_params(&self, out: &mut Vec<UniverseParamId>) {
        match self {
            Level::Zero => {}
            Level::Param(p) => out.push(*p),
            Level::Succ(l) => l.collect_params(out),
            Level::Max(a, b) | Level::IMax(a, b) => {
                a.collect_params(out);
                b.collect_params(out);
            }
        }
    }
}

impl fmt::Display for Level {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Level::Zero => f.write_str("0"),
            Level::Param(p) => write!(f, "{p}"),
            Level::Succ(l) => write!(f, "({l} + 1)"),
            Level::Max(a, b) => write!(f, "max({a}, {b})"),
            Level::IMax(a, b) => write!(f, "imax({a}, {b})"),
        }
    }
}

// ---------------------------------------------------------------------------
// Constraint graph
// ---------------------------------------------------------------------------

/// A single order constraint: `lhs ≤ rhs`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Constraint {
    /// Left-hand side.
    pub lhs: Level,
    /// Right-hand side.
    pub rhs: Level,
}

/// A set of accumulated universe inequalities.
///
/// The solver is deliberately *sound but incomplete*: it only discharges
/// constraints that follow immediately from the algebraic laws. Undischarged
/// constraints are retained so that the caller can decide whether to fail or
/// to generalize them into fresh parameters (the "typical ambiguity"
/// behaviour of Coq and Lean).
#[derive(Clone, Debug, Default)]
pub struct ConstraintSet {
    constraints: Vec<Constraint>,
}

impl ConstraintSet {
    /// Create an empty constraint set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record `lhs ≤ rhs`.
    pub fn add(&mut self, lhs: Level, rhs: Level) {
        self.constraints.push(Constraint { lhs, rhs });
    }

    /// Number of recorded constraints.
    #[must_use]
    pub fn len(&self) -> usize {
        self.constraints.len()
    }

    /// `true` iff no constraints have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.constraints.is_empty()
    }

    /// Iterate over the recorded constraints.
    pub fn iter(&self) -> impl Iterator<Item = &Constraint> {
        self.constraints.iter()
    }

    /// Try to discharge all constraints.
    ///
    /// Returns `Ok(())` if every constraint is entailed by the algebraic laws,
    /// otherwise returns the first constraint that could not be discharged.
    pub fn discharge(&self) -> Result<(), Constraint> {
        for c in &self.constraints {
            if !leq(&c.lhs, &c.rhs) {
                return Err(c.clone());
            }
        }
        Ok(())
    }
}

/// Sound (but incomplete) decision procedure for `a ≤ b`.
///
/// The procedure is total and terminating: every recursive call strictly
/// decreases the total syntactic size of the two levels.
#[must_use]
pub fn leq(a: &Level, b: &Level) -> bool {
    let a = a.normalize();
    let b = b.normalize();
    leq_nf(&a, &b)
}

fn leq_nf(a: &Level, b: &Level) -> bool {
    if a == b {
        return true;
    }
    // 0 ≤ anything.
    if matches!(a, Level::Zero) {
        return true;
    }
    // From here `a ≠ 0` and `a ≠ b`. Decompose only the right-hand side:
    // every rule below is sound, and each recursive call strictly shrinks
    // `b`, so the procedure terminates. (Decomposing the left-hand side
    // with rules like `succ(a) ≤ b if a ≤ b` would be *unsound* — e.g. it
    // proves `2 ≤ 1` from `1 ≤ 1` — so left `Succ`/`Max`/`IMax` splitting
    // is deliberately absent. The solver is incomplete instead: e.g.
    // `max(u,v) ≤ max(u,v,w)` returns `false`.)
    match b {
        // `a ≤ 0` with `a ≠ 0` never holds.
        Level::Zero => false,
        // `a ≤ p` with `a ≠ p`, `a ≠ 0`: not entailed without constraints.
        Level::Param(_) => false,
        // `a ≤ y+1` if `a ≤ y` (right weakening; sound).
        Level::Succ(y) => leq_nf(a, y),
        // `a ≤ max(x, y)` if `a ≤ x` or `a ≤ y` (sound).
        Level::Max(x, y) => leq_nf(a, x) || leq_nf(a, y),
        // `a ≤ imax(x, y)` with `a ≠ 0`: must return `false`. Reducing to
        // `a ≤ x` would be unsound because `y` may instantiate to `0`,
        // collapsing `imax(x, y)` to `0` (e.g. `u ≤ imax(u, v)` fails for
        // `v = 0`). Only `0 ≤ imax` holds, handled above.
        Level::IMax(_, _) => false,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_is_zero() {
        assert!(Level::zero().is_zero());
        assert!(!Level::zero().succ().is_zero());
    }

    #[test]
    fn succ_normalizes() {
        let l = Level::zero().succ().succ();
        assert_eq!(
            l.normalize(),
            Level::Succ(Box::new(Level::Succ(Box::new(Level::Zero))))
        );
    }

    #[test]
    fn max_with_zero_is_identity() {
        let u = Level::param(UniverseParamId(7));
        assert_eq!(Level::max(Level::zero(), u.clone()).normalize(), u);
        assert_eq!(Level::max(u.clone(), Level::zero()).normalize(), u);
    }

    #[test]
    fn max_idempotent() {
        let u = Level::param(UniverseParamId(1));
        assert_eq!(Level::max(u.clone(), u.clone()).normalize(), u);
    }

    #[test]
    fn imax_annihilates_at_zero() {
        let u = Level::param(UniverseParamId(2));
        assert_eq!(
            Level::imax(u.clone(), Level::zero()).normalize(),
            Level::Zero
        );
    }

    #[test]
    fn imax_with_zero_left() {
        let u = Level::param(UniverseParamId(3));
        assert_eq!(Level::imax(Level::zero(), u.clone()).normalize(), u);
    }

    #[test]
    fn normalize_is_idempotent() {
        let u = Level::param(UniverseParamId(4));
        let v = Level::param(UniverseParamId(5));
        let l = Level::max(
            Level::imax(u.clone(), v.clone()),
            Level::max(Level::zero(), u.clone()),
        );
        let n1 = l.normalize();
        let n2 = n1.normalize();
        assert_eq!(n1, n2);
    }

    #[test]
    fn leq_reflexive() {
        let u = Level::param(UniverseParamId(6));
        assert!(leq(&u, &u));
    }

    #[test]
    fn leq_zero_bottom() {
        let u = Level::param(UniverseParamId(8));
        assert!(leq(&Level::zero(), &u));
        assert!(!leq(&u, &Level::zero()));
    }

    #[test]
    fn leq_succ_congruent() {
        let u = Level::param(UniverseParamId(9));
        let v = Level::param(UniverseParamId(10));
        // The solver is incomplete: u+1 ≤ v+1 requires u ≤ v, which cannot be
        // decided for distinct parameters without extra information.
        assert!(!leq(&u.clone().succ(), &v.clone().succ()));
        // But reflexive successors do discharge.
        assert!(leq(&u.clone().succ(), &u.succ()));
    }

    #[test]
    fn constraint_set_discharge() {
        let u = Level::param(UniverseParamId(11));
        let mut cs = ConstraintSet::new();
        cs.add(Level::zero(), u.clone());
        cs.add(u.clone(), u.clone());
        assert!(cs.discharge().is_ok());

        cs.add(u.clone(), Level::zero());
        assert!(cs.discharge().is_err());
    }
}
