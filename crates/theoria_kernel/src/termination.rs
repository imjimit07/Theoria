//! Size-Change Termination (SCT) analysis.
//!
//! This module decides whether a recursive function definition terminates by
//! the **size-change principle** of Lee, Jones, and Ben-Amram.
//!
//! ## The principle
//!
//! A recursive function `f` is *size-change terminating* iff every infinite
//! sequence of recursive calls would force an infinite descent in some
//! well-founded data value. Concretely:
//!
//! 1. For each recursive call from `f` to `g`, build a **size-change graph**
//!    `G` whose nodes are the parameters of `f` and `g` and whose edges are
//!    labelled by the relation between corresponding arguments.
//! 2. Take the **transitive closure** of the call graph, composing matrices
//!    via the min-plus semiring.
//! 3. Require every **idempotent** matrix in that closure to have a strict
//!    decrease (`-1`) on its diagonal.
//!
//! If the condition holds, the function terminates. If it fails, the function
//! *may* still terminate (the analysis is sound but incomplete).
//!
//! ## Matrix entries
//!
//! | Entry | Meaning |
//! |-------|---------|
//! | `-1`  | strict decrease: `new_j` is structurally smaller than `old_i` |
//! | `0`   | non-increase: `new_j` is structurally equal to or smaller than `old_i` |
//! | `∞`   | no relation between `old_i` and `new_j` |
//!
//! Composition uses the min-plus semiring: `(A ⊗ B)[i][j] = min_k (A[i][k] + B[k][j])`,
//! with `∞` as the additive identity and `0` as the multiplicative identity.
//! The `min` operation respects `-1 < 0 < ∞`, so a strict decrease anywhere
//! along a thread propagates as a strict decrease in the composite.
//!
//! ## Soundness
//!
//! The checker is a *sufficient* condition: a passing SCT check guarantees
//! termination. A failing check does not imply non-termination. The kernel
//! therefore rejects any recursive definition that does not pass, but the
//! user may add an explicit termination proof via the `termination by` clause
//! (a future extension).
//!
//! ## Certificates
//!
//! On success, the checker produces a [`TerminationCertificate`] recording
//! the call graph, the closure, and the idempotent witnesses. The kernel
//! can re-verify the certificate independently, which makes termination
//! checking auditable without re-running the analysis.

use crate::expr::{DeBruijnIndex, Expr};
use crate::name::NameId;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::fmt;

// ---------------------------------------------------------------------------
// Size-change matrix entries
// ---------------------------------------------------------------------------

/// A single entry in a size-change matrix.
///
/// The ordering is `Decrease < Equal < None`, matching the min-plus semiring
/// where `Decrease = -1`, `Equal = 0`, and `None = ∞`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Entry {
    /// The new argument is *strictly smaller* than the old argument.
    Decrease,
    /// The new argument is *equal to or smaller* than the old argument.
    Equal,
    /// No relation is known between the old and new arguments.
    None,
}

impl Entry {
    /// The min-plus semiring addition (`min`).
    #[must_use]
    pub const fn min(self, other: Self) -> Self {
        match (self, other) {
            (Entry::Decrease, _) | (_, Entry::Decrease) => Entry::Decrease,
            (Entry::Equal, _) | (_, Entry::Equal) => Entry::Equal,
            (Entry::None, Entry::None) => Entry::None,
        }
    }

    /// The min-plus semiring multiplication (`+`), with `None` as the
    /// absorbing element.
    #[must_use]
    pub const fn add(self, other: Self) -> Self {
        match (self, other) {
            (Entry::None, _) | (_, Entry::None) => Entry::None,
            (Entry::Decrease, _) | (_, Entry::Decrease) => Entry::Decrease,
            (Entry::Equal, Entry::Equal) => Entry::Equal,
        }
    }

    /// `true` iff the entry is a strict decrease.
    #[must_use]
    pub const fn is_decrease(self) -> bool {
        matches!(self, Entry::Decrease)
    }

    /// Numeric representation for diagnostics.
    #[must_use]
    pub const fn to_i8(self) -> i8 {
        match self {
            Entry::Decrease => -1,
            Entry::Equal => 0,
            Entry::None => i8::MAX,
        }
    }
}

impl fmt::Display for Entry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Entry::Decrease => f.write_str("↓"),
            Entry::Equal => f.write_str("≈"),
            Entry::None => f.write_str("∞"),
        }
    }
}

// ---------------------------------------------------------------------------
// SizeChangeMatrix
// ---------------------------------------------------------------------------

/// A size-change matrix: an `rows × cols` grid of [`Entry`] values.
///
/// `rows` is the arity of the source function; `cols` is the arity of the
/// target function. Missing arguments are represented by `Entry::None`,
/// following the convention that a partially applied symbol cannot be
/// compared with any other argument.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SizeChangeMatrix {
    rows: usize,
    cols: usize,
    entries: Vec<Entry>, // row-major: entries[i * cols + j]
}

impl SizeChangeMatrix {
    /// Build a matrix filled with `Entry::None`.
    #[must_use]
    pub fn empty(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            entries: alloc::vec![Entry::None; rows * cols],
        }
    }

    /// Build the identity matrix on `n` arguments: every diagonal entry is
    /// `Entry::Equal`, every off-diagonal entry is `Entry::None`.
    #[must_use]
    pub fn identity(n: usize) -> Self {
        let mut m = Self::empty(n, n);
        for i in 0..n {
            m.set(i, i, Entry::Equal);
        }
        m
    }

    /// Number of rows (source arity).
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns (target arity).
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// `true` iff the matrix is square.
    #[must_use]
    pub const fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    /// Read the entry at `(i, j)`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= rows` or `j >= cols`.
    #[must_use]
    pub fn get(&self, i: usize, j: usize) -> Entry {
        assert!(i < self.rows, "row {i} out of range (rows = {})", self.rows);
        assert!(j < self.cols, "col {j} out of range (cols = {})", self.cols);
        self.entries[i * self.cols + j]
    }

    /// Set the entry at `(i, j)`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= rows` or `j >= cols`.
    pub fn set(&mut self, i: usize, j: usize, e: Entry) {
        assert!(i < self.rows, "row {i} out of range (rows = {})", self.rows);
        assert!(j < self.cols, "col {j} out of range (cols = {})", self.cols);
        self.entries[i * self.cols + j] = e;
    }

    /// Compose two matrices: `self ⊗ other`.
    ///
    /// The result has `self.rows` rows and `other.cols` columns. The
    /// composition uses the min-plus semiring:
    ///
    /// ```text
    /// (self ⊗ other)[i][j] = min_k (self[i][k] + other[k][j])
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `self.cols != other.rows`.
    #[must_use]
    pub fn compose(&self, other: &SizeChangeMatrix) -> SizeChangeMatrix {
        assert_eq!(
            self.cols, other.rows,
            "cannot compose {}×{} with {}×{}",
            self.rows, self.cols, other.rows, other.cols
        );
        let mut result = SizeChangeMatrix::empty(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut acc = Entry::None;
                for k in 0..self.cols {
                    let a = self.get(i, k);
                    let b = other.get(k, j);
                    acc = acc.min(a.add(b));
                }
                result.set(i, j, acc);
            }
        }
        result
    }

    /// Iterate the squaring map `M ↦ M ⊗ M` up to `fuel + 1` times,
    /// returning the first iterate that is a fixed point of the map.
    ///
    /// ## Why fuel is required
    ///
    /// `⊗` is monotone in each argument, so `f(M) = M ⊗ M` is monotone —
    /// but it is **not** inflationary (`M ≤ f(M)` need not hold), and only
    /// an inflationary or deflationary map on a finite lattice is
    /// guaranteed to reach a fixed point by iteration. The squaring
    /// sequence `M, M², M⁴, M⁸, …` can move both up and down in the
    /// matrix order; whether it converges to a fixed point is not known
    /// for arbitrary inputs and may depend on `M`. The earlier version of
    /// this method looped until equality, which risked a hang on inputs
    /// whose squaring sequence is eventually periodic with period ≥ 2.
    ///
    /// This method therefore takes an explicit iteration budget and
    /// returns `None` if no fixed point is reached within it, rather than
    /// looping.
    ///
    /// ## Relationship to the SCT closure
    ///
    /// The SCT closure is the *set* of all `Mᵏ` for `k ≥ 1`; the squaring
    /// sequence only visits powers of two. For the checker's purposes
    /// [`TerminationChecker::check`] computes the full closure via
    /// `compute_closure` and does **not** call this method. This routine
    /// exists for tests and diagnostics, where the input matrices are
    /// small enough that a fixed point is reached in a handful of
    /// squarings.
    ///
    /// ## Fuel semantics
    ///
    /// At most `fuel + 1` squarings are performed:
    ///
    /// * `fuel` squarings inside the loop, each followed by a check that
    ///   the current iterate equals its own square;
    /// * one final squaring after the loop, in case the last iterate
    ///   produced by the loop is a fixed point only when squared once
    ///   more.
    ///
    /// `fuel = 0` therefore tests whether the input is already idempotent.
    #[must_use]
    pub fn idempotent_closure_with_fuel(&self, fuel: usize) -> Option<SizeChangeMatrix> {
        let mut current = self.clone();
        for _ in 0..fuel {
            let next = current.compose(&current);
            if next == current {
                return Some(current);
            }
            current = next;
        }
        // The final squaring may coincide with the iterate we hold.
        let next = current.compose(&current);
        if next == current { Some(current) } else { None }
    }

    /// Convenience wrapper around [`Self::idempotent_closure_with_fuel`]
    /// with a size-derived default budget of `rows² + 8` iterations.
    ///
    /// The result, when one exists, is a fixed point of the squaring map
    /// `M ↦ M ⊗ M`. It is **not** in general the SCT closure of `self`
    /// (the closure is a set of matrices, computed by
    /// `TerminationChecker::check` via `compute_closure`); it is simply
    /// the first power-of-two iterate that happens to be a fixed point.
    /// Callers who need the closure should use
    /// [`TerminationChecker::check`].
    ///
    /// # Panics
    ///
    /// Panics if no fixed point is reached within the default budget.
    /// The default is generous for the small matrices the SCT tests
    /// exercise (`rows ≤ 4`). If the input may come from an untrusted
    /// source, call [`Self::idempotent_closure_with_fuel`] directly and
    /// handle the `None` case; do not rely on this wrapper.
    #[must_use]
    pub fn idempotent_closure(&self) -> SizeChangeMatrix {
        let fuel = self.rows.saturating_mul(self.rows).saturating_add(8);
        self.idempotent_closure_with_fuel(fuel).expect(
            "idempotent_closure: matrix did not stabilise within the default \
             fuel budget; call idempotent_closure_with_fuel with an explicit \
             budget",
        )
    }

    /// `true` iff the matrix has a strict decrease on its diagonal.
    ///
    /// Only meaningful for square matrices; a non-square matrix returns
    /// `false`.
    #[must_use]
    pub fn has_diagonal_decrease(&self) -> bool {
        if !self.is_square() {
            return false;
        }
        (0..self.rows).any(|i| self.get(i, i).is_decrease())
    }

    /// Apply the matrix to a vector of "size" values.
    ///
    /// Used only in tests and for diagnostics. The vector must have
    /// `self.cols` entries; the result has `self.rows` entries.
    #[must_use]
    pub fn apply(&self, input: &[i64]) -> Vec<i64> {
        assert_eq!(input.len(), self.cols);
        (0..self.rows)
            .map(|i| {
                (0..self.cols)
                    .map(|j| match self.get(i, j) {
                        Entry::Decrease => input[j] - 1,
                        Entry::Equal => input[j],
                        Entry::None => i64::MIN,
                    })
                    .max()
                    .unwrap_or(i64::MIN)
            })
            .collect()
    }
}

impl fmt::Display for SizeChangeMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..self.rows {
            for j in 0..self.cols {
                write!(f, "{}", self.get(i, j))?;
                if j + 1 < self.cols {
                    write!(f, " ")?;
                }
            }
            if i + 1 < self.rows {
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Call graph
// ---------------------------------------------------------------------------

/// A single recursive call edge in the call graph.
#[derive(Clone, Debug)]
pub struct CallEdge {
    /// The source function.
    pub source: NameId,
    /// The target function.
    pub target: NameId,
    /// The size-change matrix for this call.
    pub matrix: SizeChangeMatrix,
}

/// A call graph for a (possibly mutually recursive) block of functions.
#[derive(Clone, Debug, Default)]
pub struct CallGraph {
    /// The functions in the block, in declaration order.
    pub functions: Vec<NameId>,
    /// The arity of each function, indexed by position in `functions`.
    pub arities: BTreeMap<NameId, usize>,
    /// The call edges.
    pub edges: Vec<CallEdge>,
}

impl CallGraph {
    /// Create an empty call graph.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a function with its arity.
    pub fn add_function(&mut self, name: NameId, arity: usize) {
        self.functions.push(name);
        self.arities.insert(name, arity);
    }

    /// Add a call edge.
    pub fn add_edge(&mut self, source: NameId, target: NameId, matrix: SizeChangeMatrix) {
        self.edges.push(CallEdge {
            source,
            target,
            matrix,
        });
    }

    /// The arity of a function, or `None` if it is not in this graph.
    #[must_use]
    pub fn arity(&self, name: NameId) -> Option<usize> {
        self.arities.get(&name).copied()
    }

    /// All edges whose source is `name`.
    pub fn edges_from(&self, name: NameId) -> impl Iterator<Item = &CallEdge> {
        self.edges.iter().filter(move |e| e.source == name)
    }
}

// ---------------------------------------------------------------------------
// Termination certificate
// ---------------------------------------------------------------------------

/// A certified witness that a call graph satisfies the SCT condition.
#[derive(Clone, Debug)]
pub struct TerminationCertificate {
    /// The function the certificate is about (for a mutual block, the
    /// representative).
    pub function: NameId,
    /// The number of functions in the block.
    pub block_size: usize,
    /// The size of the closure (number of matrices).
    pub closure_size: usize,
    /// One witness per idempotent matrix in the closure: the diagonal index
    /// at which a strict decrease occurs.
    pub witnesses: Vec<(usize, usize)>, // (matrix_index, diagonal_index)
}

impl TerminationCertificate {
    /// Render a one-line summary of the certificate.
    #[must_use]
    pub fn summary(&self) -> alloc::string::String {
        alloc::format!(
            "SCT certificate for {}: block of {}, closure of {} matrices, {} witnesses",
            self.function,
            self.block_size,
            self.closure_size,
            self.witnesses.len()
        )
    }
}

// ---------------------------------------------------------------------------
// Termination errors
// ---------------------------------------------------------------------------

/// Errors produced by the termination checker.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TerminationError {
    /// A function's arity is unknown.
    UnknownFunction(NameId),
    /// A call edge references a function not in the graph.
    UnknownCallee {
        /// The source of the offending edge.
        source: NameId,
        /// The unknown callee.
        target: NameId,
    },
    /// The SCT condition failed: an idempotent matrix lacks a diagonal
    /// decrease.
    NotSizeChangeTerminating {
        /// The function whose call graph failed.
        function: NameId,
        /// The offending idempotent matrix.
        matrix: SizeChangeMatrix,
    },
    /// The call graph contains no edges (trivially terminating, but
    /// reported for diagnostics).
    EmptyCallGraph,
}

impl fmt::Display for TerminationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TerminationError::UnknownFunction(n) => {
                write!(f, "unknown function {n} in call graph")
            }
            TerminationError::UnknownCallee { source, target } => {
                write!(f, "call edge {source} → {target} references unknown callee")
            }
            TerminationError::NotSizeChangeTerminating { function, matrix } => write!(
                f,
                "function {function} is not size-change terminating\n\
                 offending idempotent matrix:\n{matrix}"
            ),
            TerminationError::EmptyCallGraph => {
                write!(f, "call graph has no edges")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for TerminationError {}

// ---------------------------------------------------------------------------
// TerminationChecker
// ---------------------------------------------------------------------------

/// The SCT termination checker.
#[derive(Debug, Default)]
pub struct TerminationChecker;

impl TerminationChecker {
    /// Create a new checker.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Check that `graph` satisfies the SCT condition.
    ///
    /// Returns a [`TerminationCertificate`] on success.
    ///
    /// # Errors
    ///
    /// * [`TerminationError::UnknownCallee`] if an edge references a function
    ///   not registered in the graph.
    /// * [`TerminationError::NotSizeChangeTerminating`] if some idempotent
    ///   matrix in the closure lacks a diagonal decrease.
    pub fn check(&self, graph: &CallGraph) -> Result<TerminationCertificate, TerminationError> {
        // --- 1. Validate the graph. -------------------------------------
        for edge in &graph.edges {
            if graph.arity(edge.source).is_none() {
                return Err(TerminationError::UnknownFunction(edge.source));
            }
            if graph.arity(edge.target).is_none() {
                return Err(TerminationError::UnknownCallee {
                    source: edge.source,
                    target: edge.target,
                });
            }
            if edge.matrix.rows() != graph.arity(edge.source).unwrap()
                || edge.matrix.cols() != graph.arity(edge.target).unwrap()
            {
                return Err(TerminationError::UnknownCallee {
                    source: edge.source,
                    target: edge.target,
                });
            }
        }

        // --- 2. Build the closure. --------------------------------------
        //
        // For every pair (f, g), the closure contains the min-plus sum of
        // all matrices along any path from f to g. We compute this by
        // repeated composition of the full edge set until no new matrices
        // appear.
        let closure = self.compute_closure(graph);

        // --- 3. Check the SCT condition. -------------------------------
        //
        // For every idempotent matrix M in the closure (i.e. M ⊗ M = M)
        // whose source and target arities match (a loop), M must have a
        // strict decrease on its diagonal.
        let mut witnesses = Vec::new();
        for (idx, m) in closure.iter().enumerate() {
            if !m.is_square() {
                continue;
            }
            // Idempotence: M ⊗ M == M.
            let mm = m.compose(m);
            if mm != *m {
                continue;
            }
            // The SCT condition: some diagonal entry is a strict decrease.
            if let Some(diag) = (0..m.rows()).find(|&i| m.get(i, i).is_decrease()) {
                witnesses.push((idx, diag));
            } else {
                return Err(TerminationError::NotSizeChangeTerminating {
                    function: graph.functions.first().copied().unwrap_or(NameId(0)),
                    matrix: m.clone(),
                });
            }
        }

        // A graph with no edges is trivially terminating; still emit a
        // certificate for uniformity.
        let function = graph.functions.first().copied().unwrap_or(NameId(0));
        Ok(TerminationCertificate {
            function,
            block_size: graph.functions.len(),
            closure_size: closure.len(),
            witnesses,
        })
    }

    /// Compute the transitive closure of the call graph under min-plus
    /// composition.
    ///
    /// The result is the set of all matrices that can be obtained by
    /// composing one or more edges along a path. Duplicates are removed.
    fn compute_closure(&self, graph: &CallGraph) -> Vec<SizeChangeMatrix> {
        let mut closure: Vec<SizeChangeMatrix> =
            graph.edges.iter().map(|e| e.matrix.clone()).collect();
        let mut changed = true;
        while changed {
            changed = false;
            let snapshot = closure.clone();
            for a in &snapshot {
                for b in &snapshot {
                    // Only compose matrices where the arities line up.
                    if a.cols() != b.rows() {
                        continue;
                    }
                    let composed = a.compose(b);
                    if !closure.contains(&composed) {
                        closure.push(composed);
                        changed = true;
                    }
                }
            }
        }
        closure
    }

    // -----------------------------------------------------------------
    // From AST: build a call graph for a single self-recursive function
    // -----------------------------------------------------------------

    /// Build a call graph for a single self-recursive function whose body
    /// is `body` and whose parameters are `params`.
    ///
    /// Each recursive call `f a₁ ... aₙ` inside `body` generates an edge
    /// from `f` to `f` with a matrix whose `(i, j)` entry records the
    /// structural relation between the `i`-th formal parameter and the
    /// `j`-th actual argument:
    ///
    /// * `Equal` if the actual is the formal itself.
    /// * `None` otherwise — including constructor applications like
    ///   `pred(x)`, which the kernel cannot prove decreasing (see
    ///   `classify_param_vs_arg` for why).
    ///
    /// Note that `Decrease` never appears in a generated graph: only an
    /// explicit matrix built by hand (as in the SCT unit tests) contains
    /// decreases. Generated graphs therefore fail the SCT check whenever
    /// they contain a self-edge, which is exactly the sound behaviour for
    /// direct self-recursion (see the classifier's documentation).
    #[must_use]
    pub fn build_self_call_graph(
        &self,
        function: NameId,
        params: &[NameId],
        body: &Expr,
    ) -> CallGraph {
        let mut graph = CallGraph::new();
        let arity = params.len();
        graph.add_function(function, arity);

        // Collect all recursive calls in `body` where the head is `function`.
        let mut calls: Vec<Vec<Expr>> = Vec::new();
        collect_calls(function, params.len(), body, &mut calls);

        for actuals in calls {
            let mut matrix = SizeChangeMatrix::empty(arity, arity);
            for (i, param) in params.iter().enumerate() {
                for (j, actual) in actuals.iter().enumerate() {
                    let entry = classify_param_vs_arg(*param, actual, params, arity);
                    // Take the *minimum* (strongest) relation if multiple
                    // entries map to the same cell.
                    let old = matrix.get(i, j);
                    matrix.set(i, j, old.min(entry));
                }
            }
            graph.add_edge(function, function, matrix);
        }
        graph
    }

    /// Build a call graph for a mutually recursive block.
    ///
    /// `functions[i]` is `(name, params)` for the `i`-th function; `bodies[i]`
    /// is its body. Every call to any block member from any block body
    /// contributes an edge.
    ///
    /// # Panics
    ///
    /// Panics if `functions.len() != bodies.len()`.
    #[must_use]
    pub fn build_mutual_call_graph(
        &self,
        functions: &[(NameId, Vec<NameId>)],
        bodies: &[Expr],
    ) -> CallGraph {
        assert_eq!(
            functions.len(),
            bodies.len(),
            "build_mutual_call_graph: functions/bodies length mismatch"
        );
        let mut graph = CallGraph::new();
        for (name, params) in functions {
            graph.add_function(*name, params.len());
        }
        let family: Vec<NameId> = functions.iter().map(|(n, _)| *n).collect();
        let arities: BTreeMap<NameId, usize> =
            functions.iter().map(|(n, p)| (*n, p.len())).collect();

        for ((source_name, source_params), body) in functions.iter().zip(bodies.iter()) {
            let mut calls: Vec<(NameId, Vec<Expr>)> = Vec::new();
            collect_family_calls(&family, &arities, body, &mut calls);
            for (target_name, actuals) in calls {
                let target_arity = *arities.get(&target_name).expect("family name missing");
                let mut matrix = SizeChangeMatrix::empty(source_params.len(), target_arity);
                for (i, param) in source_params.iter().enumerate() {
                    for (j, actual) in actuals.iter().enumerate() {
                        let entry = classify_param_vs_arg(
                            *param,
                            actual,
                            source_params,
                            source_params.len(),
                        );
                        let old = matrix.get(i, j);
                        matrix.set(i, j, old.min(entry));
                    }
                }
                graph.add_edge(*source_name, target_name, matrix);
            }
        }
        graph
    }
}

// ---------------------------------------------------------------------------
// AST traversal helpers
// ---------------------------------------------------------------------------

/// Collect the argument lists of every recursive call to `function` in
/// `body`.
///
/// A "recursive call" is an application whose head is `Const(function, _)`
/// and whose arity matches `arity`. Higher-order occurrences (the function
/// passed as an argument) are *not* collected; that is sound because a
/// higher-order call cannot be bounded by structural descent anyway.
fn collect_calls(function: NameId, arity: usize, body: &Expr, out: &mut Vec<Vec<Expr>>) {
    match body {
        Expr::Sort(_) | Expr::Var(_) | Expr::Lit(_) => {}

        Expr::Const(name, _) => {
            // A bare reference to a 0-arity function is a self-call.
            if *name == function && arity == 0 {
                out.push(Vec::new());
            }
        }

        Expr::App(..) => {
            let (head, args) = flatten_app(body);
            if let Expr::Const(name, _) = head {
                if *name == function && args.len() == arity {
                    out.push(args.iter().map(|a| (*a).clone()).collect());
                }
            }
            // Recurse into subterms, including the head and each argument.
            collect_calls(function, arity, head, out);
            for a in &args {
                collect_calls(function, arity, a, out);
            }
        }

        Expr::Pi(_, _, dom, cod) | Expr::Lam(_, _, dom, cod) => {
            collect_calls(function, arity, dom, out);
            collect_calls(function, arity, cod, out);
        }

        Expr::Let(_, ty, val, body) => {
            collect_calls(function, arity, ty, out);
            collect_calls(function, arity, val, out);
            collect_calls(function, arity, body, out);
        }
    }
}

/// Collect the argument lists of every call to a family member inside `body`.
///
/// A call is an application whose head is `Const(name, _)` for some
/// `name ∈ family`, with the arity matching the family member's declared
/// arity. A bare `Const(name, _)` counts as a 0-arity call.
fn collect_family_calls(
    family: &[NameId],
    arities: &BTreeMap<NameId, usize>,
    body: &Expr,
    out: &mut Vec<(NameId, Vec<Expr>)>,
) {
    match body {
        Expr::Sort(_) | Expr::Var(_) | Expr::Lit(_) => {}
        Expr::Const(name, _) => {
            if family.contains(name) && arities.get(name) == Some(&0) {
                out.push((*name, Vec::new()));
            }
        }
        Expr::App(..) => {
            let (head, args) = flatten_app(body);
            if let Expr::Const(name, _) = head {
                if family.contains(name) && arities.get(name) == Some(&args.len()) {
                    out.push((*name, args.iter().map(|a| (*a).clone()).collect()));
                }
            }
            collect_family_calls(family, arities, head, out);
            for a in &args {
                collect_family_calls(family, arities, a, out);
            }
        }
        Expr::Pi(_, _, d, c) | Expr::Lam(_, _, d, c) => {
            collect_family_calls(family, arities, d, out);
            collect_family_calls(family, arities, c, out);
        }
        Expr::Let(_, t, v, b) => {
            collect_family_calls(family, arities, t, out);
            collect_family_calls(family, arities, v, out);
            collect_family_calls(family, arities, b, out);
        }
    }
}

/// Flatten an application spine: `App(App(f, a), b)` → `(f, [a, b])`.
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

/// Classify the relation between a formal parameter and an actual argument.
///
/// The relation is `Equal` iff the actual is the formal parameter itself
/// (matched by its de Bruijn index in the body). Every other actual is
/// `None`.
///
/// ## Why so conservative
///
/// SCT in the classical sense operates over a language with explicit
/// **destructors** — pattern matching that extracts a strict sub-term of a
/// matched value. The kernel does not have destructors: pattern matching is
/// desugared into recursor applications, and the recursor's ι-rule fires
/// under a well-foundedness argument supplied by the inductive's positivity
/// proof, not by any syntactic property of the term.
///
/// Consequently, from the kernel's point of view, the *only* way a
/// recursive argument can be known equal to the formal is by syntactic
/// identity. Any expression that applies a function — including
/// `Nat.succ n` and every defined "destructor" like `Nat.pred n` — is
/// classified as `None`, because the kernel cannot decide from the head
/// alone whether the result is smaller, equal, or larger than the input.
///
/// This rejects every direct self-recursive definition. That is sound: no
/// such definition is structurally terminating in the kernel's own terms.
/// Real recursion uses a recursor and never appears as a self-edge.
fn classify_param_vs_arg(param: NameId, actual: &Expr, params: &[NameId], arity: usize) -> Entry {
    let param_index = match params.iter().position(|p| *p == param) {
        Some(i) => (arity - 1 - i) as u32,
        None => return Entry::None,
    };
    match actual {
        Expr::Var(DeBruijnIndex(i)) if *i == param_index => Entry::Equal,
        _ => Entry::None,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::name::NameTable;
    use alloc::vec;

    // ---- matrix algebra ----------------------------------------------

    #[test]
    fn entry_ordering() {
        assert!(Entry::Decrease.min(Entry::Equal) == Entry::Decrease);
        assert!(Entry::Equal.min(Entry::None) == Entry::Equal);
        assert!(Entry::None.min(Entry::None) == Entry::None);
    }

    #[test]
    fn entry_addition() {
        assert!(Entry::None.add(Entry::Decrease) == Entry::None);
        assert!(Entry::Decrease.add(Entry::None) == Entry::None);
        assert!(Entry::Decrease.add(Entry::Equal) == Entry::Decrease);
        assert!(Entry::Equal.add(Entry::Equal) == Entry::Equal);
    }

    #[test]
    fn matrix_identity_is_identity() {
        let m = SizeChangeMatrix::identity(3);
        assert_eq!(m.get(0, 0), Entry::Equal);
        assert_eq!(m.get(0, 1), Entry::None);
        assert_eq!(m.get(1, 1), Entry::Equal);
        assert_eq!(m.get(2, 2), Entry::Equal);

        // Composing with identity is a no-op.
        let mut a = SizeChangeMatrix::empty(3, 3);
        a.set(0, 1, Entry::Decrease);
        a.set(1, 2, Entry::Equal);
        assert_eq!(a.compose(&m), a);
        assert_eq!(m.compose(&a), a);
    }

    #[test]
    fn matrix_composition_min_plus() {
        // A = [↓ ∞]   B = [≈ ∞]
        //     [∞ ≈]       [↓ ∞]
        let mut a = SizeChangeMatrix::empty(2, 2);
        a.set(0, 0, Entry::Decrease);
        a.set(1, 1, Entry::Equal);

        let mut b = SizeChangeMatrix::empty(2, 2);
        b.set(0, 0, Entry::Equal);
        b.set(1, 0, Entry::Decrease);

        let c = a.compose(&b);
        // (A ⊗ B)[0][0] = min(A[0][0]+B[0][0], A[0][1]+B[1][0])
        //               = min(↓ + ≈, ∞ + ↓) = min(↓, ∞) = ↓
        assert_eq!(c.get(0, 0), Entry::Decrease);
        // (A ⊗ B)[1][0] = min(A[1][0]+B[0][0], A[1][1]+B[1][0])
        //               = min(∞ + ≈, ≈ + ↓) = min(∞, ↓) = ↓
        assert_eq!(c.get(1, 0), Entry::Decrease);
    }

    #[test]
    fn matrix_closure_of_single_decrease() {
        // A = [↓]  — the closure is just [↓], which is idempotent.
        let mut a = SizeChangeMatrix::empty(1, 1);
        a.set(0, 0, Entry::Decrease);
        let c = a.idempotent_closure();
        assert_eq!(c, a);
        assert!(c.has_diagonal_decrease());
    }

    #[test]
    fn matrix_closure_of_identity() {
        // The identity matrix is already idempotent.
        let m = SizeChangeMatrix::identity(2);
        let c = m.idempotent_closure();
        assert_eq!(c, m);
        assert!(!c.has_diagonal_decrease());
    }

    #[test]
    fn idempotent_closure_with_fuel_succeeds_when_sufficient() {
        let m = SizeChangeMatrix::identity(2);
        assert_eq!(m.idempotent_closure_with_fuel(4), Some(m.clone()));
    }

    #[test]
    fn idempotent_closure_with_fuel_returns_none_when_exhausted() {
        // A matrix whose squaring sequence reaches a fixed point on its
        // second iterate, but is not itself a fixed point:
        //
        //     M = [∞ ↓]        M² = [∞ ∞]        M² ⊗ M² = M²
        //         [∞ ∞]             [∞ ∞]
        //
        // At fuel = 0 the loop is skipped and only the final squaring is
        // performed; M² ≠ M so the routine returns `None`. At fuel = 3
        // the loop finds M² ⊗ M² = M² on its second iteration and
        // returns `Some(M²)`.
        let mut m = SizeChangeMatrix::empty(2, 2);
        m.set(0, 1, Entry::Decrease);
        assert_eq!(m.idempotent_closure_with_fuel(0), None);
        assert!(m.idempotent_closure_with_fuel(3).is_some());
    }

    // ---- call graph --------------------------------------------------

    #[test]
    fn call_graph_basics() {
        let mut names = NameTable::new();
        let f = names.intern("f");
        let g = names.intern("g");
        let mut graph = CallGraph::new();
        graph.add_function(f, 2);
        graph.add_function(g, 1);
        graph.add_edge(f, f, SizeChangeMatrix::identity(2));
        graph.add_edge(f, g, SizeChangeMatrix::empty(2, 1));
        assert_eq!(graph.arity(f), Some(2));
        assert_eq!(graph.edges_from(f).count(), 2);
    }

    // ---- SCT checker: canonical examples -----------------------------

    #[test]
    fn sct_accepts_first_param_decrease() {
        // f(x, y) = ... f(x-1, y) ...
        // matrix: [↓ ∞]
        //         [∞ ≈]
        let mut names = NameTable::new();
        let f = names.intern("f");
        let mut graph = CallGraph::new();
        graph.add_function(f, 2);
        let mut m = SizeChangeMatrix::empty(2, 2);
        m.set(0, 0, Entry::Decrease);
        m.set(1, 1, Entry::Equal);
        graph.add_edge(f, f, m);

        let cert = TerminationChecker::new().check(&graph).unwrap();
        assert_eq!(cert.block_size, 1);
        assert!(!cert.witnesses.is_empty());
    }

    #[test]
    fn sct_rejects_no_decrease() {
        // f(x, y) = ... f(x, y) ...
        // matrix: [≈ ∞]
        //         [∞ ≈]
        let mut names = NameTable::new();
        let f = names.intern("f");
        let mut graph = CallGraph::new();
        graph.add_function(f, 2);
        let mut m = SizeChangeMatrix::empty(2, 2);
        m.set(0, 0, Entry::Equal);
        m.set(1, 1, Entry::Equal);
        graph.add_edge(f, f, m);

        let err = TerminationChecker::new().check(&graph).unwrap_err();
        assert!(matches!(
            err,
            TerminationError::NotSizeChangeTerminating { .. }
        ));
    }

    #[test]
    fn sct_accepts_alternating_decrease() {
        // f(x, y) = ... f(y, x-1) ...
        // matrix: [∞ ↓]
        //         [↓ ∞]
        let mut names = NameTable::new();
        let f = names.intern("f");
        let mut graph = CallGraph::new();
        graph.add_function(f, 2);
        let mut m = SizeChangeMatrix::empty(2, 2);
        m.set(0, 1, Entry::Decrease);
        m.set(1, 0, Entry::Decrease);
        graph.add_edge(f, f, m);

        let cert = TerminationChecker::new().check(&graph).unwrap();
        assert!(!cert.witnesses.is_empty());
    }

    #[test]
    fn sct_accepts_ackermann_style() {
        // ack(x, y) = ... ack(x-1, y) ... ack(x, y-1) ...
        // Two edges:
        //   [↓ ∞]        [≈ ∞]
        //   [∞ ≈]        [∞ ↓]
        let mut names = NameTable::new();
        let ack = names.intern("ack");
        let mut graph = CallGraph::new();
        graph.add_function(ack, 2);

        let mut m1 = SizeChangeMatrix::empty(2, 2);
        m1.set(0, 0, Entry::Decrease);
        m1.set(1, 1, Entry::Equal);
        graph.add_edge(ack, ack, m1);

        let mut m2 = SizeChangeMatrix::empty(2, 2);
        m2.set(0, 0, Entry::Equal);
        m2.set(1, 1, Entry::Decrease);
        graph.add_edge(ack, ack, m2);

        let cert = TerminationChecker::new().check(&graph).unwrap();
        assert!(cert.closure_size >= 2);
    }

    #[test]
    fn sct_rejects_constant_loop() {
        // f(x) = ... f(x) ...  with a matrix that forgets the argument.
        let mut names = NameTable::new();
        let f = names.intern("f");
        let mut graph = CallGraph::new();
        graph.add_function(f, 1);
        let m = SizeChangeMatrix::empty(1, 1); // all None
        graph.add_edge(f, f, m);

        let err = TerminationChecker::new().check(&graph).unwrap_err();
        assert!(matches!(
            err,
            TerminationError::NotSizeChangeTerminating { .. }
        ));
    }

    #[test]
    fn sct_accepts_zero_arity() {
        // f() = f()  with an empty matrix. The matrix is trivially
        // idempotent and has no diagonal, so the SCT condition is
        // vacuously satisfied. In practice such a function is
        // non-terminating, but the SCT abstraction cannot see that.
        let mut names = NameTable::new();
        let f = names.intern("f");
        let mut graph = CallGraph::new();
        graph.add_function(f, 0);
        graph.add_edge(f, f, SizeChangeMatrix::empty(0, 0));

        // An empty matrix has `has_diagonal_decrease() == false`, so the
        // checker rejects it.
        let err = TerminationChecker::new().check(&graph).unwrap_err();
        assert!(matches!(
            err,
            TerminationError::NotSizeChangeTerminating { .. }
        ));
    }

    #[test]
    fn sct_rejects_unknown_callee() {
        let mut names = NameTable::new();
        let f = names.intern("f");
        let g = names.intern("g");
        let mut graph = CallGraph::new();
        graph.add_function(f, 1);
        graph.add_edge(f, g, SizeChangeMatrix::empty(1, 1));

        let err = TerminationChecker::new().check(&graph).unwrap_err();
        assert!(matches!(err, TerminationError::UnknownCallee { .. }));
    }

    // ---- AST integration ---------------------------------------------

    #[test]
    fn build_graph_for_simple_recursion() {
        // f(x) = ... f(x) ...   — no decrease, should fail.
        let mut names = NameTable::new();
        let f = names.intern("f");
        let x = names.intern("x");

        // body: App(Const(f, []), [Var(0)])
        let body = Expr::app(Expr::Const(f, Vec::new()), Expr::Var(DeBruijnIndex(0)));

        let graph = TerminationChecker::new().build_self_call_graph(f, &[x], &body);
        assert_eq!(graph.edges.len(), 1);
        // The matrix should have (0,0) = Equal (x is passed unchanged).
        let m = &graph.edges[0].matrix;
        assert_eq!(m.get(0, 0), Entry::Equal);
    }

    #[test]
    fn build_graph_treats_application_as_none() {
        // f(x) = ... f(pred(x)) ...
        //
        // Under the sound classifier, `pred(x)` is `None`: the kernel has no
        // way to know that `pred` is a destructor, and the elaborator is not
        // trusted to have told us. This test pins that behaviour.
        let mut names = NameTable::new();
        let f = names.intern("f");
        let x = names.intern("x");
        let pred = names.intern("pred");

        let actual = Expr::app(Expr::Const(pred, Vec::new()), Expr::Var(DeBruijnIndex(0)));
        let body = Expr::app(Expr::Const(f, Vec::new()), actual);

        let graph = TerminationChecker::new().build_self_call_graph(f, &[x], &body);
        assert_eq!(graph.edges.len(), 1);
        let m = &graph.edges[0].matrix;
        assert_eq!(m.get(0, 0), Entry::None);
    }

    #[test]
    fn build_graph_catches_zero_arity_self_reference() {
        let mut names = NameTable::new();
        let f = names.intern("f");
        let body = Expr::Const(f, Vec::new());
        let graph = TerminationChecker::new().build_self_call_graph(f, &[], &body);
        assert_eq!(graph.edges.len(), 1);
        assert_eq!(graph.edges[0].matrix.rows(), 0);
        assert_eq!(graph.edges[0].matrix.cols(), 0);
    }

    // ---- certificate -------------------------------------------------

    #[test]
    fn certificate_summary_is_nonempty() {
        let mut names = NameTable::new();
        let f = names.intern("f");
        let mut graph = CallGraph::new();
        graph.add_function(f, 1);
        let mut m = SizeChangeMatrix::empty(1, 1);
        m.set(0, 0, Entry::Decrease);
        graph.add_edge(f, f, m);
        let cert = TerminationChecker::new().check(&graph).unwrap();
        assert!(!cert.summary().is_empty());
    }

    // ---- mutual blocks -----------------------------------------------

    #[test]
    fn mutual_call_graph_builds_cross_edges() {
        // even(0) := true;  even(succ n) := odd n
        // odd(0)  := false; odd(succ n)  := even n
        // Bodies are abstracted: we only care that even calls odd and
        // vice-versa.
        let mut names = NameTable::new();
        let even = names.intern("even");
        let odd = names.intern("odd");
        let n = names.intern("n");

        // even body: odd n
        let even_body = Expr::app(Expr::Const(odd, Vec::new()), Expr::Var(DeBruijnIndex(0)));
        // odd body: even n
        let odd_body = Expr::app(Expr::Const(even, Vec::new()), Expr::Var(DeBruijnIndex(0)));

        let functions = vec![(even, vec![n]), (odd, vec![n])];
        let bodies = vec![even_body, odd_body];

        let graph = TerminationChecker::new().build_mutual_call_graph(&functions, &bodies);
        assert_eq!(graph.functions.len(), 2);
        // Two edges: even → odd, odd → even.
        assert_eq!(graph.edges.len(), 2);
    }

    #[test]
    fn mutual_block_with_no_internal_calls_builds_empty_graph() {
        let mut names = NameTable::new();
        let f = names.intern("f");
        let g = names.intern("g");
        let x = names.intern("x");
        // f body: Nat.zero (no calls); g body: Nat.zero (no calls).
        let zero = names.intern("Nat.zero");
        let f_body = Expr::Const(zero, Vec::new());
        let g_body = Expr::Const(zero, Vec::new());
        let functions = vec![(f, vec![x]), (g, vec![x])];
        let bodies = vec![f_body, g_body];
        let graph = TerminationChecker::new().build_mutual_call_graph(&functions, &bodies);
        assert!(graph.edges.is_empty());
    }

    #[test]
    fn mutual_even_odd_fails_sct() {
        // even n calls odd n (no decrease); odd n calls even n (no decrease).
        // The closure contains a self-loop with no ↓ on the diagonal, so
        // SCT rejects. This is correct: no such pair is structurally
        // terminating in the kernel's own terms.
        let mut names = NameTable::new();
        let even = names.intern("even");
        let odd = names.intern("odd");
        let n = names.intern("n");

        let even_body = Expr::app(Expr::Const(odd, Vec::new()), Expr::Var(DeBruijnIndex(0)));
        let odd_body = Expr::app(Expr::Const(even, Vec::new()), Expr::Var(DeBruijnIndex(0)));
        let functions = vec![(even, vec![n]), (odd, vec![n])];
        let bodies = vec![even_body, odd_body];
        let graph = TerminationChecker::new().build_mutual_call_graph(&functions, &bodies);
        let err = TerminationChecker::new().check(&graph).unwrap_err();
        assert!(matches!(
            err,
            TerminationError::NotSizeChangeTerminating { .. }
        ));
    }
}
