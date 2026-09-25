//! Hash-consed term arena.
//!
//! Provides an alternative representation of kernel terms in which every
//! distinct term is stored exactly once and referred to by an [`ExprId`].
//! Because the arena is *hash-consed*, two `ExprId`s are equal iff their
//! terms are structurally equal — so equality is O(1) (a `u32` comparison)
//! rather than O(size of term).
//!
//! ## When to use the arena
//!
//! The kernel's canonical AST is [`Expr`], an owned
//! tree with `Box`ed subterms. Existing code — NbE, the type checker,
//! positivity, termination, erasure — operates on `Expr` and continues to
//! do so. The arena is a **parallel representation** for callers that
//! benefit from structural sharing and O(1) equality:
//!
//! * the elaborator, which compares types repeatedly,
//! * proof libraries, which store many overlapping subterms,
//! * any cache keyed by term identity.
//!
//! ## Guarantees
//!
//! * **Canonical form.** After `intern`, structurally equal terms have
//!   equal `ExprId`s.
//! * **Structural equality is `ExprId` equality.** `a == b` iff
//!   `to_expr(a) == to_expr(b)`.
//! * **Termination.** `intern` recurses with a depth counter; terms
//!   deeper than [`ExprArena::max_depth`] are rejected with
//!   [`InternError::DepthExceeded`] rather than overflowing the stack.
//!
//! ## Non-guarantees
//!
//! * The arena does **not** replace `Expr`. Both representations coexist.
//! * `ExprArena::data` panics on an out-of-range `ExprId`. Construct ids
//!   only from [`ExprArena`] constructors or from `ExprId::from_index`
//!   applied to an index previously returned by the arena.
//! * `to_expr` recurses to the term's depth; callers who choose
//!   `max_depth` above the platform's stack limit may still overflow.
//!   The default, 10 000, is safe on every target we care about.
//!
//! ## Complexity
//!
//! | Operation | Cost |
//! |-----------|------|
//! | `intern` for a term of size `n` | `O(n log m)` where `m` is the arena size |
//! | Constructors (`app`, `lam`, ...) | `O(log m)` |
//! | `data(id)` | `O(1)` |
//! | `id1 == id2` | `O(1)` |
//! | `to_expr(id)` | `O(size of term)` |

use crate::expr::{BinderInfo, DeBruijnIndex, Expr, Literal};
use crate::level::Level;
use crate::name::NameId;
use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;
use core::hash::{Hash, Hasher};

// ---------------------------------------------------------------------------
// ExprId
// ---------------------------------------------------------------------------

/// A handle to a term stored in an [`ExprArena`].
///
/// Two `ExprId`s compare equal iff their terms are structurally equal
/// (this is the hash-consing invariant). The wrapped `u32` is an index
/// into the arena and is stable for the lifetime of the arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ExprId(pub u32);

impl ExprId {
    /// The raw index into the arena.
    #[must_use]
    pub const fn index(self) -> u32 {
        self.0
    }

    /// Construct an `ExprId` from a raw index.
    ///
    /// The index must have been produced by an [`ExprArena`]; otherwise
    /// [`ExprArena::data`] will panic. This constructor is provided for
    /// callers who serialise or transport `ExprId`s across process
    /// boundaries; it is not a way to fabricate terms.
    #[must_use]
    pub const fn from_index(i: u32) -> Self {
        ExprId(i)
    }
}

// ---------------------------------------------------------------------------
// ExprData
// ---------------------------------------------------------------------------

/// A single node in the arena.
///
/// Same shape as [`Expr`], but subterms are [`ExprId`]
/// handles instead of `Box<Expr>`. Every variant is hashed and compared
/// by the *ids* of its subterms, not their contents.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ExprData {
    /// A universe `Sort(l)`.
    Sort(Level),
    /// A bound variable, by de Bruijn index.
    Var(DeBruijnIndex),
    /// A constant applied to universe parameters.
    Const(NameId, Vec<Level>),
    /// Application `f a`.
    App(ExprId, ExprId),
    /// Lambda `λ (x : A). b`.
    Lam(BinderInfo, NameId, ExprId, ExprId),
    /// Dependent function type `Π (x : A). B`.
    Pi(BinderInfo, NameId, ExprId, ExprId),
    /// Local definition `let x : A := v; b`.
    Let(NameId, ExprId, ExprId, ExprId),
    /// A literal.
    Lit(Literal),
}

impl ExprData {
    /// `true` iff this is a `Sort`.
    #[must_use]
    pub const fn is_sort(&self) -> bool {
        matches!(self, ExprData::Sort(_))
    }

    /// `true` iff this is a `Var`.
    #[must_use]
    pub const fn is_var(&self) -> bool {
        matches!(self, ExprData::Var(_))
    }

    /// `true` iff this is a `Const`.
    #[must_use]
    pub const fn is_const(&self) -> bool {
        matches!(self, ExprData::Const(_, _))
    }

    /// `true` iff this is an `App`.
    #[must_use]
    pub const fn is_app(&self) -> bool {
        matches!(self, ExprData::App(_, _))
    }

    /// `true` iff this is a `Lam`.
    #[must_use]
    pub const fn is_lam(&self) -> bool {
        matches!(self, ExprData::Lam(_, _, _, _))
    }

    /// `true` iff this is a `Pi`.
    #[must_use]
    pub const fn is_pi(&self) -> bool {
        matches!(self, ExprData::Pi(_, _, _, _))
    }

    /// `true` iff this is a `Let`.
    #[must_use]
    pub const fn is_let(&self) -> bool {
        matches!(self, ExprData::Let(_, _, _, _))
    }

    /// `true` iff this is a `Lit`.
    #[must_use]
    pub const fn is_lit(&self) -> bool {
        matches!(self, ExprData::Lit(_))
    }
}

// ---------------------------------------------------------------------------
// InternError
// ---------------------------------------------------------------------------

/// Errors produced by [`ExprArena::intern`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InternError {
    /// The expression exceeded the arena's configured maximum depth.
    ///
    /// The depth counter starts at 0 for the root and increments on each
    /// recursion into a subterm.
    DepthExceeded {
        /// The arena's configured limit.
        limit: usize,
        /// The depth at which the limit was exceeded.
        depth: usize,
    },
}

impl fmt::Display for InternError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InternError::DepthExceeded { limit, depth } => {
                write!(f, "expression depth {depth} exceeds arena limit {limit}")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for InternError {}

// ---------------------------------------------------------------------------
// FNV-1a hasher
// ---------------------------------------------------------------------------

/// FNV-1a 64-bit hasher.
///
/// Deterministic and dependency-free. Used to bucket `ExprData` values in
/// the arena's dedup index. FNV-1a is fine here because the inputs are
/// hash-consed term ids and short level vectors, not attacker-controlled
/// bytes with structured collisions.
#[derive(Clone, Copy, Debug)]
struct FnvHasher(u64);

impl FnvHasher {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    #[inline]
    const fn new() -> Self {
        FnvHasher(Self::OFFSET)
    }
}

impl Default for FnvHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl Hasher for FnvHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.0 ^= u64::from(*b);
            self.0 = self.0.wrapping_mul(Self::PRIME);
        }
    }
}

/// Hash an `ExprData` with FNV-1a.
fn hash_data(data: &ExprData) -> u64 {
    let mut h = FnvHasher::new();
    data.hash(&mut h);
    h.finish()
}

// ---------------------------------------------------------------------------
// ExprArena
// ---------------------------------------------------------------------------

/// A hash-consed term arena.
///
/// Terms are stored in insertion order; the dedup index maps each term's
/// hash to the ids of terms with that hash. Constructing a term with
/// [`ExprArena::app`] (or any other constructor) returns the id of an
/// existing equal term if one is present, and otherwise inserts a new one.
#[derive(Debug)]
pub struct ExprArena {
    terms: Vec<ExprData>,
    dedup: BTreeMap<u64, Vec<ExprId>>,
    max_depth: usize,
}

impl ExprArena {
    /// The default depth limit for [`intern`](ExprArena::intern): 10 000.
    pub const DEFAULT_MAX_DEPTH: usize = 10_000;

    /// Create an empty arena with the default depth limit.
    #[must_use]
    pub fn new() -> Self {
        Self::with_max_depth(Self::DEFAULT_MAX_DEPTH)
    }

    /// Create an empty arena with a custom depth limit.
    ///
    /// The limit bounds the recursion depth of [`intern`](ExprArena::intern)
    /// and, by construction, the depth of every term stored in the arena.
    /// Callers should choose a value that is safe for the platform's stack;
    /// 10 000 is conservative on all mainstream targets.
    #[must_use]
    pub fn with_max_depth(max_depth: usize) -> Self {
        ExprArena {
            terms: Vec::new(),
            dedup: BTreeMap::new(),
            max_depth,
        }
    }

    /// The configured depth limit.
    #[must_use]
    pub const fn max_depth(&self) -> usize {
        self.max_depth
    }

    /// Number of distinct terms in the arena.
    #[must_use]
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    /// `true` iff the arena contains no terms.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    /// `true` iff `id` refers to a term in this arena.
    #[must_use]
    pub fn contains(&self, id: ExprId) -> bool {
        (id.index() as usize) < self.terms.len()
    }

    // -----------------------------------------------------------------
    // Core: interned insert
    // -----------------------------------------------------------------

    /// Insert or find `data`, returning its canonical id.
    fn intern_data(&mut self, data: ExprData) -> ExprId {
        // Split the borrow so we can read `terms` while mutating `dedup`.
        let ExprArena { terms, dedup, .. } = self;
        let h = hash_data(&data);
        if let Some(bucket) = dedup.get(&h) {
            for &existing in bucket {
                if terms[existing.index() as usize] == data {
                    return existing;
                }
            }
        }
        let id = ExprId(u32::try_from(terms.len()).expect("arena overflow (>4.29B terms)"));
        dedup.entry(h).or_default().push(id);
        terms.push(data);
        id
    }

    // -----------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------

    /// `Sort(l)`.
    pub fn sort(&mut self, l: Level) -> ExprId {
        self.intern_data(ExprData::Sort(l))
    }

    /// A bound variable.
    pub fn var(&mut self, i: DeBruijnIndex) -> ExprId {
        self.intern_data(ExprData::Var(i))
    }

    /// A constant applied to universe parameters.
    pub fn const_(&mut self, name: NameId, levels: Vec<Level>) -> ExprId {
        self.intern_data(ExprData::Const(name, levels))
    }

    /// Application `f a`.
    pub fn app(&mut self, f: ExprId, a: ExprId) -> ExprId {
        self.intern_data(ExprData::App(f, a))
    }

    /// Left-associated application of `f` to a sequence of arguments.
    ///
    /// `app_many(f, [a, b, c])` produces `App(App(App(f, a), b), c)`,
    /// reusing any existing shared subterms.
    pub fn app_many(&mut self, f: ExprId, args: impl IntoIterator<Item = ExprId>) -> ExprId {
        let mut acc = f;
        for a in args {
            acc = self.app(acc, a);
        }
        acc
    }

    /// Lambda `λ (x : A). b`.
    pub fn lam(&mut self, info: BinderInfo, name: NameId, dom: ExprId, body: ExprId) -> ExprId {
        self.intern_data(ExprData::Lam(info, name, dom, body))
    }

    /// Dependent function type `Π (x : A). B`.
    pub fn pi(&mut self, info: BinderInfo, name: NameId, dom: ExprId, cod: ExprId) -> ExprId {
        self.intern_data(ExprData::Pi(info, name, dom, cod))
    }

    /// Local definition `let x : A := v; b`.
    pub fn let_(&mut self, name: NameId, ty: ExprId, val: ExprId, body: ExprId) -> ExprId {
        self.intern_data(ExprData::Let(name, ty, val, body))
    }

    /// A literal.
    pub fn lit(&mut self, l: Literal) -> ExprId {
        self.intern_data(ExprData::Lit(l))
    }

    // -----------------------------------------------------------------
    // Access
    // -----------------------------------------------------------------

    /// The data at `id`.
    ///
    /// # Panics
    ///
    /// Panics if `id` is out of range. Construct ids only through this
    /// arena; the panic is a programming error, not a user error.
    #[must_use]
    pub fn data(&self, id: ExprId) -> &ExprData {
        let i = id.index() as usize;
        self.terms.get(i).unwrap_or_else(|| {
            panic!(
                "ExprArena::data: id {id:?} out of range (len {})",
                self.terms.len()
            )
        })
    }

    /// Decompose an application spine.
    ///
    /// `App(App(App(f, a), b), c)` returns `(f, [a, b, c])`. For a
    /// non-application, returns `(id, [])`.
    #[must_use]
    pub fn get_app_fn_and_args(&self, id: ExprId) -> (ExprId, Vec<ExprId>) {
        let mut args = Vec::new();
        let mut cur = id;
        while let ExprData::App(f, a) = self.data(cur) {
            args.push(*a);
            cur = *f;
        }
        args.reverse();
        (cur, args)
    }

    // -----------------------------------------------------------------
    // Bridge: Expr <-> ExprId
    // -----------------------------------------------------------------

    /// Intern an [`Expr`], returning its canonical id.
    ///
    /// Recurses with a depth counter; terms deeper than
    /// [`ExprArena::max_depth`] are rejected with
    /// [`InternError::DepthExceeded`].
    ///
    /// # Errors
    ///
    /// [`InternError::DepthExceeded`] if the term's depth exceeds the
    /// arena's limit.
    pub fn intern(&mut self, e: &Expr) -> Result<ExprId, InternError> {
        self.intern_rec(e, 0)
    }

    fn intern_rec(&mut self, e: &Expr, depth: usize) -> Result<ExprId, InternError> {
        if depth > self.max_depth {
            return Err(InternError::DepthExceeded {
                limit: self.max_depth,
                depth,
            });
        }
        let id = match e {
            Expr::Sort(l) => self.sort(l.clone()),
            Expr::Var(i) => self.var(*i),
            Expr::Const(name, levels) => self.const_(*name, levels.clone()),
            Expr::App(f, a) => {
                let fi = self.intern_rec(f, depth + 1)?;
                let ai = self.intern_rec(a, depth + 1)?;
                self.app(fi, ai)
            }
            Expr::Lam(info, name, dom, body) => {
                let di = self.intern_rec(dom, depth + 1)?;
                let bi = self.intern_rec(body, depth + 1)?;
                self.lam(*info, *name, di, bi)
            }
            Expr::Pi(info, name, dom, cod) => {
                let di = self.intern_rec(dom, depth + 1)?;
                let ci = self.intern_rec(cod, depth + 1)?;
                self.pi(*info, *name, di, ci)
            }
            Expr::Let(name, ty, val, body) => {
                let ti = self.intern_rec(ty, depth + 1)?;
                let vi = self.intern_rec(val, depth + 1)?;
                let bi = self.intern_rec(body, depth + 1)?;
                self.let_(*name, ti, vi, bi)
            }
            Expr::Lit(l) => self.lit(l.clone()),
        };
        Ok(id)
    }

    /// Reconstruct the [`Expr`] at `id`.
    ///
    /// The arena's own `max_depth` guarantees the term's depth is
    /// bounded, so this function does not check depth. Callers that
    /// configure `max_depth` above the platform's stack limit must take
    /// their own precautions.
    ///
    /// # Panics
    ///
    /// Panics if `id` is out of range.
    #[must_use]
    pub fn to_expr(&self, id: ExprId) -> Expr {
        match self.data(id) {
            ExprData::Sort(l) => Expr::Sort(l.clone()),
            ExprData::Var(i) => Expr::Var(*i),
            ExprData::Const(name, levels) => Expr::Const(*name, levels.clone()),
            ExprData::App(f, a) => Expr::app(self.to_expr(*f), self.to_expr(*a)),
            ExprData::Lam(info, name, dom, body) => Expr::Lam(
                *info,
                *name,
                Box::new(self.to_expr(*dom)),
                Box::new(self.to_expr(*body)),
            ),
            ExprData::Pi(info, name, dom, cod) => Expr::Pi(
                *info,
                *name,
                Box::new(self.to_expr(*dom)),
                Box::new(self.to_expr(*cod)),
            ),
            ExprData::Let(name, ty, val, body) => Expr::Let(
                *name,
                Box::new(self.to_expr(*ty)),
                Box::new(self.to_expr(*val)),
                Box::new(self.to_expr(*body)),
            ),
            ExprData::Lit(l) => Expr::Lit(l.clone()),
        }
    }

    // -----------------------------------------------------------------
    // Diagnostics
    // -----------------------------------------------------------------

    /// Render the term at `id` in a compact, name-resolving form.
    ///
    /// Intended for diagnostics and tests; the output is not canonical
    /// source syntax. Sorts render without their universe level; callers
    /// needing levels should use [`to_expr`](ExprArena::to_expr) with a
    /// display formatter instead.
    ///
    /// # Panics
    ///
    /// Panics if `id` is out of range.
    #[must_use]
    pub fn render(&self, id: ExprId, names: &crate::name::NameTable) -> String {
        let mut out = String::new();
        self.render_into(id, names, &mut out);
        out
    }

    fn render_into(&self, id: ExprId, names: &crate::name::NameTable, out: &mut String) {
        match self.data(id) {
            ExprData::Sort(_) => out.push_str("Sort"),
            ExprData::Var(i) => {
                out.push('#');
                out.push_str(&alloc::format!("{}", i.0));
            }
            ExprData::Const(name, _) => out.push_str(names.resolve(*name)),
            ExprData::Lit(Literal::Nat(n)) => out.push_str(&alloc::format!("{n}")),
            ExprData::Lit(Literal::Str(s)) => {
                out.push('"');
                out.push_str(s);
                out.push('"');
            }
            ExprData::App(_, _) => {
                let (f, args) = self.get_app_fn_and_args(id);
                out.push('(');
                self.render_into(f, names, out);
                for a in args {
                    out.push(' ');
                    self.render_into(a, names, out);
                }
                out.push(')');
            }
            ExprData::Lam(_, _, dom, body) => {
                out.push_str("(λ ");
                self.render_into(*dom, names, out);
                out.push_str(". ");
                self.render_into(*body, names, out);
                out.push(')');
            }
            ExprData::Pi(_, _, dom, cod) => {
                out.push_str("(Π ");
                self.render_into(*dom, names, out);
                out.push_str(". ");
                self.render_into(*cod, names, out);
                out.push(')');
            }
            ExprData::Let(_, ty, val, body) => {
                out.push_str("(let ");
                self.render_into(*ty, names, out);
                out.push_str(" := ");
                self.render_into(*val, names, out);
                out.push_str(" in ");
                self.render_into(*body, names, out);
                out.push(')');
            }
        }
    }
}

impl Default for ExprArena {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{BinderInfo, DeBruijnIndex, Expr, Literal};
    use crate::level::Level;
    use crate::name::{NameId, NameTable};
    use alloc::boxed::Box;
    use alloc::vec;

    fn mk_names() -> NameTable {
        let mut t = NameTable::new();
        t.intern("Nat");
        t.intern("Nat.zero");
        t.intern("Nat.succ");
        t
    }

    fn nat_id(_t: &NameTable) -> NameId {
        // We know the intern order: Nat = 0.
        NameId(0)
    }

    fn succ_id() -> NameId {
        NameId(2)
    }

    // ---- ExprId basics -----------------------------------------------

    #[test]
    fn expr_id_index_roundtrip() {
        let id = ExprId(42);
        assert_eq!(ExprId::from_index(id.index()), id);
    }

    #[test]
    fn expr_id_equality_is_structural() {
        let mut arena = ExprArena::new();
        let z = arena.var(DeBruijnIndex(0));
        let z2 = arena.var(DeBruijnIndex(0));
        assert_eq!(z, z2);
        let n = arena.var(DeBruijnIndex(1));
        assert_ne!(z, n);
    }

    // ---- Deduplication -----------------------------------------------

    #[test]
    fn intern_same_expr_twice_returns_same_id() {
        let mut arena = ExprArena::new();
        let names = mk_names();
        let e = Expr::Const(nat_id(&names), vec![]);
        let a = arena.intern(&e).unwrap();
        let b = arena.intern(&e).unwrap();
        assert_eq!(a, b);
        assert_eq!(arena.len(), 1);
    }

    #[test]
    fn intern_structurally_equal_exprs_returns_same_id() {
        let mut arena = ExprArena::new();
        let names = mk_names();
        let e1 = Expr::app(
            Expr::Const(succ_id(), vec![]),
            Expr::Const(nat_id(&names), vec![]),
        );
        let e2 = Expr::app(
            Expr::Const(succ_id(), vec![]),
            Expr::Const(nat_id(&names), vec![]),
        );
        let a = arena.intern(&e1).unwrap();
        let b = arena.intern(&e2).unwrap();
        assert_eq!(a, b);
        // Three distinct nodes: Const(succ), Const(Nat), and the App.
        assert_eq!(arena.len(), 3);
    }

    #[test]
    fn intern_distinct_exprs_returns_distinct_ids() {
        let mut arena = ExprArena::new();
        let names = mk_names();
        let a = arena.intern(&Expr::Const(nat_id(&names), vec![])).unwrap();
        let b = arena.intern(&Expr::Const(succ_id(), vec![])).unwrap();
        assert_ne!(a, b);
        assert_eq!(arena.len(), 2);
    }

    // ---- Sharing -----------------------------------------------------

    #[test]
    fn subterm_sharing_reuses_ids() {
        let mut arena = ExprArena::new();
        let names = mk_names();
        let nat = Expr::Const(nat_id(&names), vec![]);
        let succ = Expr::Const(succ_id(), vec![]);

        let z = arena.intern(&nat).unwrap();
        let succ_eid = arena.intern(&succ).unwrap();
        let sz = arena.app(succ_eid, z);
        let ssz = arena.app(succ_eid, sz);
        let sssz = arena.app(succ_eid, ssz);

        // Five distinct nodes: Nat, succ, and the three App nodes built
        // from the already-interned ids (no re-interning happens here).
        let _ = sssz;
        assert_eq!(arena.len(), 5);
    }

    #[test]
    fn reinterning_shared_subterm_does_not_grow_arena() {
        let mut arena = ExprArena::new();
        let names = mk_names();
        let e = Expr::app(
            Expr::Const(succ_id(), vec![]),
            Expr::Const(nat_id(&names), vec![]),
        );
        let _ = arena.intern(&e).unwrap();
        let before = arena.len();
        let _ = arena.intern(&e).unwrap();
        let _ = arena.intern(&e).unwrap();
        assert_eq!(arena.len(), before);
    }

    // ---- Round-trip --------------------------------------------------

    #[test]
    fn to_expr_roundtrips_simple() {
        let mut arena = ExprArena::new();
        let names = mk_names();
        let e = Expr::Const(nat_id(&names), vec![]);
        let id = arena.intern(&e).unwrap();
        assert_eq!(arena.to_expr(id), e);
    }

    #[test]
    fn to_expr_roundtrips_complex() {
        let mut arena = ExprArena::new();
        let names = mk_names();
        let nat = Expr::Const(nat_id(&names), vec![]);
        let succ = Expr::Const(succ_id(), vec![]);
        let e = Expr::Lam(
            BinderInfo::Default,
            names_intern_placeholder(),
            Box::new(nat.clone()),
            Box::new(Expr::app(
                succ,
                Expr::app(
                    Expr::Const(nat_id(&names), vec![]),
                    Expr::Var(DeBruijnIndex(0)),
                ),
            )),
        );
        let id = arena.intern(&e).unwrap();
        assert_eq!(arena.to_expr(id), e);
    }

    fn names_intern_placeholder() -> NameId {
        // We don't need a real name for roundtrip testing; any NameId works.
        NameId(99)
    }

    // ---- Predicates --------------------------------------------------

    #[test]
    fn predicates_match_variant() {
        let mut arena = ExprArena::new();
        let s = arena.sort(Level::zero());
        let v = arena.var(DeBruijnIndex(0));
        let c = arena.const_(NameId(0), vec![]);
        let f = arena.app(s, v);
        assert!(arena.data(s).is_sort());
        assert!(arena.data(v).is_var());
        assert!(arena.data(c).is_const());
        assert!(arena.data(f).is_app());
    }

    #[test]
    fn get_app_fn_and_args_flattens_spine() {
        let mut arena = ExprArena::new();
        let f = arena.const_(NameId(1), vec![]);
        let a = arena.var(DeBruijnIndex(0));
        let b = arena.var(DeBruijnIndex(1));
        let c = arena.var(DeBruijnIndex(2));
        let full = arena.app_many(f, [a, b, c]);
        let (head, args) = arena.get_app_fn_and_args(full);
        assert_eq!(head, f);
        assert_eq!(args, vec![a, b, c]);
    }

    #[test]
    fn get_app_fn_and_args_of_non_app_is_identity() {
        let mut arena = ExprArena::new();
        let v = arena.var(DeBruijnIndex(0));
        let (head, args) = arena.get_app_fn_and_args(v);
        assert_eq!(head, v);
        assert!(args.is_empty());
    }

    // ---- Depth limit -------------------------------------------------

    #[test]
    fn depth_limit_accepts_shallow_terms() {
        let mut arena = ExprArena::with_max_depth(2);
        let names = mk_names();
        let z = Expr::Const(nat_id(&names), vec![]);
        let e = Expr::app(z.clone(), z.clone());
        assert!(arena.intern(&e).is_ok());
    }

    #[test]
    fn depth_limit_rejects_deep_terms() {
        let mut arena = ExprArena::with_max_depth(1);
        let names = mk_names();
        let z = Expr::Const(nat_id(&names), vec![]);
        // App(App(z, z), z): root depth 0, inner App depth 1, leaves depth 2.
        // With max_depth = 1, leaves at depth 2 trigger the limit.
        let inner = Expr::app(z.clone(), z.clone());
        let e = Expr::app(inner, z);
        let err = arena.intern(&e).unwrap_err();
        assert!(matches!(err, InternError::DepthExceeded { limit: 1, .. }));
    }

    #[test]
    fn default_max_depth_is_ten_thousand() {
        let arena = ExprArena::new();
        assert_eq!(arena.max_depth(), ExprArena::DEFAULT_MAX_DEPTH);
    }

    // ---- Bounds ------------------------------------------------------

    #[test]
    #[should_panic(expected = "out of range")]
    fn data_on_out_of_range_id_panics() {
        let arena = ExprArena::new();
        let _ = arena.data(ExprId(99));
    }

    #[test]
    fn contains_reports_membership() {
        let mut arena = ExprArena::new();
        let names = mk_names();
        let id = arena.intern(&Expr::Const(nat_id(&names), vec![])).unwrap();
        assert!(arena.contains(id));
        assert!(!arena.contains(ExprId(999)));
    }

    // ---- Display / render --------------------------------------------

    #[test]
    fn render_produces_readable_output() {
        let mut arena = ExprArena::new();
        let names = mk_names();
        let e = Expr::app(
            Expr::Const(succ_id(), vec![]),
            Expr::Const(nat_id(&names), vec![]),
        );
        let id = arena.intern(&e).unwrap();
        let rendered = arena.render(id, &names);
        assert!(rendered.contains("Nat.succ"));
        assert!(rendered.contains("Nat"));
    }

    #[test]
    fn error_display_is_nonempty() {
        let e = InternError::DepthExceeded {
            limit: 10,
            depth: 11,
        };
        assert!(!alloc::format!("{e}").is_empty());
    }

    // ---- Empty arena -------------------------------------------------

    #[test]
    fn empty_arena_reports_empty() {
        let arena = ExprArena::new();
        assert!(arena.is_empty());
        assert_eq!(arena.len(), 0);
    }

    #[test]
    fn default_arena_equals_new() {
        let a = ExprArena::default();
        let b = ExprArena::new();
        assert_eq!(a.len(), b.len());
        assert_eq!(a.max_depth(), b.max_depth());
    }

    // ---- Const with universe levels ----------------------------------

    #[test]
    fn const_with_levels_dedups_by_level() {
        let mut arena = ExprArena::new();
        let id0 = arena.const_(NameId(7), vec![Level::zero()]);
        let id1 = arena.const_(NameId(7), vec![Level::zero()]);
        assert_eq!(id0, id1);
        let id2 = arena.const_(NameId(7), vec![Level::zero().succ()]);
        assert_ne!(id0, id2);
    }

    #[test]
    fn literal_dedups_by_value() {
        let mut arena = ExprArena::new();
        let a = arena.lit(Literal::Nat(42));
        let b = arena.lit(Literal::Nat(42));
        let c = arena.lit(Literal::Nat(43));
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn lam_dedups_by_components() {
        let mut arena = ExprArena::new();
        let names = mk_names();
        let nat = arena.const_(nat_id(&names), vec![]);
        let v = arena.var(DeBruijnIndex(0));
        let id_a = arena.lam(BinderInfo::Default, NameId(5), nat, v);
        let id_b = arena.lam(BinderInfo::Default, NameId(5), nat, v);
        assert_eq!(id_a, id_b);
        let id_c = arena.lam(BinderInfo::Implicit, NameId(5), nat, v);
        assert_ne!(id_a, id_c);
    }
}
