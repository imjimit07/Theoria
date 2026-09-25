//! Global environment and local typing context.
//!
//! The [`GlobalEnv`] stores all top-level declarations: axioms, definitions,
//! theorems, opaque constants, inductive families, their constructors, and
//! their recursors. It is the single source of truth that the type checker,
//! the NbE engine, and the termination checker all consult.
//!
//! The [`LocalContext`] models the local binders that are in scope at any
//! point during elaboration: lambda parameters, Pi parameters, and let-bound
//! variables. It is a *persistent* linked list so that extending it during
//! type checking is O(1) and requires no mutation.
//!
//! ## Design notes
//!
//! * **Atomic inductive insertion.** An inductive family, its constructors,
//!   and its recursor are inserted together by [`GlobalEnv::add_inductive`]
//!   or not at all. Partial insertion would leave the environment in an
//!   inconsistent state where a constructor exists without its inductive.
//! * **`ConstEnv` bridge.** The NbE engine consumes a light-weight
//!   [`ConstDecl`] view rather than the full
//!   [`ConstantInfo`]. The view is cached at insertion time in every
//!   entry, so [`GlobalEnv`]'s [`ConstEnv`] implementation is
//!   O(log n) plus one cheap `Rc` clone.
//! * **Opaque-by-default theorems.** Theorem bodies are retained (for
//!   auditing and for `All`-mode conversion) but their transparency is
//!   [`Irreducible`](crate::nbe::Transparency::Irreducible), so δ-unfolding
//!   never expands a proof term during ordinary type checking.
//!
//! ## Not yet implemented
//!
//! * Attribute tracking (`@[simp]`, `@[reducible]`, ...) beyond transparency.
//! * Incremental snapshot/rollback of the environment.
//! * Serialization of the environment to a stable on-disk format.

use crate::error::KernelError;
use crate::expr::{BinderInfo, DeBruijnIndex, Expr};
use crate::inductive::PositivityChecker;
use crate::level::UniverseParamId;
use crate::name::NameId;
use crate::nbe::{
    ConstDecl, ConstEnv, DbLevel, Env as NbeEnv, QuotientLiftInfo, RecursorInfo, RecursorRuleInfo,
    Transparency, Value,
};
use crate::termination::TerminationChecker;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::format;
use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

// ---------------------------------------------------------------------------
// ConstantVal — the shared name / params / type triple
// ---------------------------------------------------------------------------

/// The header shared by every declared constant.
///
/// Every declaration — axiom, definition, theorem, inductive, constructor,
/// or recursor — has a name, a (possibly empty) list of universe parameters,
/// and a type. This struct factors out that common data.
#[derive(Clone, Debug)]
pub struct ConstantVal {
    /// The constant's name.
    pub name: NameId,
    /// Universe parameters introduced by the declaration, in declaration
    /// order.
    pub universe_params: Vec<UniverseParamId>,
    /// The constant's type.
    pub ty: Rc<Expr>,
}

// ---------------------------------------------------------------------------
// Definitions, theorems, opaque constants
// ---------------------------------------------------------------------------

/// A transparent (or user-controlled-transparency) definition.
#[derive(Clone, Debug)]
pub struct DefinitionVal {
    /// The declaration header.
    pub base: ConstantVal,
    /// The definition's body.
    pub body: Rc<Expr>,
    /// When δ-unfolding is permitted for this definition.
    pub transparency: Transparency,
    /// The termination obligation, checked by [`GlobalEnv::add`].
    pub termination: TerminationObligation,
}

/// A termination obligation attached to a definition.
///
/// A definition that is not syntactically self-recursive is not required to
/// carry parameters; use [`TerminationObligation::default`] (empty params).
/// A definition whose body mentions its own name **must** supply the
/// parameter names in declaration order, otherwise
/// [`GlobalEnv::add`] fails with
/// [`EnvError::TerminationArityMismatch`].
#[derive(Clone, Debug, Default)]
pub struct TerminationObligation {
    /// The parameter names, in declaration order (outermost first).
    pub params: Vec<NameId>,
}

impl TerminationObligation {
    /// No obligation (a non-recursive definition with no parameters).
    #[must_use]
    pub fn none() -> Self {
        Self::default()
    }

    /// An obligation for a definition whose body may self-recurse.
    #[must_use]
    pub fn recursive(params: Vec<NameId>) -> Self {
        Self { params }
    }

    /// `true` iff no parameters are recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }
}

/// A theorem: a definition whose body is a proof term.
///
/// Theorems are opaque by default. Their bodies are retained so that
/// transparency mode [`All`](crate::nbe::TransparencyMode::All) can unfold
/// them for auditing and so that `theoria compile --target paper` can
/// reproduce the proof text.
#[derive(Clone, Debug)]
pub struct TheoremVal {
    /// The declaration header.
    pub base: ConstantVal,
    /// The proof term.
    pub body: Rc<Expr>,
    /// When δ-unfolding is permitted. Defaults to `Irreducible`.
    pub transparency: Transparency,
}

/// An opaque constant: has a body, but is never unfolded.
#[derive(Clone, Debug)]
pub struct OpaqueVal {
    /// The declaration header.
    pub base: ConstantVal,
    /// The body, retained for auditing.
    pub body: Rc<Expr>,
}

// ---------------------------------------------------------------------------
// Inductive families, constructors, recursors
// ---------------------------------------------------------------------------

/// An inductive family declaration.
#[derive(Clone, Debug)]
pub struct InductiveVal {
    /// The declaration header.
    pub base: ConstantVal,
    /// Number of uniform parameters (the prefix of arguments that are the
    /// same across every constructor).
    pub num_params: u32,
    /// Number of indices (the suffix of arguments that vary per value).
    pub num_indices: u32,
    /// All inductives declared in the same mutually-recursive block,
    /// including this one. Order is the declaration order.
    pub all: Vec<NameId>,
    /// Constructors of this inductive, in declaration order.
    pub constructors: Vec<NameId>,
    /// `true` iff the inductive appears in any of its own constructors.
    pub is_recursive: bool,
    /// `true` iff the inductive appears nested inside another inductive
    /// (e.g. `Tree := Node (List Tree)`).
    pub is_nested: bool,
    /// `true` iff this declaration was marked `unsafe`.
    pub is_unsafe: bool,
}

/// A constructor of an inductive family.
#[derive(Clone, Debug)]
pub struct ConstructorVal {
    /// The declaration header.
    pub base: ConstantVal,
    /// The inductive this constructor belongs to.
    pub inductive: NameId,
    /// Zero-based position of this constructor within its inductive's
    /// declaration order.
    pub index: u32,
    /// Number of parameters copied from the inductive.
    pub num_params: u32,
    /// Number of fields (constructor arguments past the parameters and
    /// indices).
    pub num_fields: u32,
    /// `true` iff the containing inductive was marked `unsafe`.
    pub is_unsafe: bool,
}

/// A single ι-reduction rule for a recursor.
#[derive(Clone, Debug)]
pub struct RecursorRule {
    /// The constructor this rule fires for.
    pub constructor: NameId,
    /// Number of fields the rule binds.
    pub num_fields: u32,
    /// The right-hand side of the rule, expressed in a context that begins
    /// with the motive(s), the minor premise(s), the parameters, the
    /// indices, and finally the constructor's fields.
    pub rhs: Rc<Expr>,
}

/// A recursor (eliminator) for an inductive family.
#[derive(Clone, Debug)]
pub struct RecursorVal {
    /// The declaration header.
    pub base: ConstantVal,
    /// The inductives eliminated by this recursor, in declaration order.
    pub all: Vec<NameId>,
    /// Number of parameters.
    pub num_params: u32,
    /// Number of indices.
    pub num_indices: u32,
    /// Number of motives. Always 1 in this kernel (no mutual recursion via
    /// a single recursor); kept explicit for future extension.
    pub num_motives: u32,
    /// Total number of minor premises (sum of constructors across `all`).
    pub num_minors: u32,
    /// The ι-reduction rules.
    pub rules: Vec<RecursorRule>,
    /// `true` for K-like eliminators, which are restricted to `Prop`.
    pub is_k: bool,
    /// `true` iff the containing inductive was marked `unsafe`.
    pub is_unsafe: bool,
}

/// The kind of a quotient constant.
#[derive(Clone, Debug)]
pub enum QuotientKind {
    /// `Quot A R` — the type former.
    Type,
    /// `Quot.mk A R a` — the constructor.
    Mk,
    /// `Quot.lift A R B f h q` — the eliminator with a β-rule.
    Lift {
        /// The `Quot.mk` this eliminator recognises.
        mk: NameId,
        /// Total number of explicit arguments.
        arity: u32,
        /// Zero-based index of the function argument.
        func_index: u32,
        /// Zero-based index of the quotient-value argument.
        major_index: u32,
    },
    /// `Quot.sound h` — the soundness axiom.
    Sound,
}

/// A quotient-related constant.
#[derive(Clone, Debug)]
pub struct QuotientVal {
    /// The declaration header.
    pub base: ConstantVal,
    /// Which quotient primitive this is.
    pub kind: QuotientKind,
}

// ---------------------------------------------------------------------------
// ConstantInfo
// ---------------------------------------------------------------------------

/// The full information attached to a declared constant.
#[allow(clippy::large_enum_variant)] // Measured: largest variant < 256 bytes.
#[derive(Debug)]
pub enum ConstantInfo {
    /// A trusted, unproven constant.
    Axiom(ConstantVal),
    /// A transparent definition.
    Definition(DefinitionVal),
    /// A theorem (proof term).
    Theorem(TheoremVal),
    /// An opaque constant with a body.
    Opaque(OpaqueVal),
    /// An inductive family.
    Inductive(InductiveVal),
    /// A constructor.
    Constructor(ConstructorVal),
    /// A recursor.
    Recursor(RecursorVal),
    /// A quotient-related constant.
    Quotient(QuotientVal),
}

impl ConstantInfo {
    /// The declaration header.
    #[must_use]
    pub fn base(&self) -> &ConstantVal {
        match self {
            ConstantInfo::Axiom(v) => v,
            ConstantInfo::Definition(v) => &v.base,
            ConstantInfo::Theorem(v) => &v.base,
            ConstantInfo::Opaque(v) => &v.base,
            ConstantInfo::Inductive(v) => &v.base,
            ConstantInfo::Constructor(v) => &v.base,
            ConstantInfo::Recursor(v) => &v.base,
            ConstantInfo::Quotient(v) => &v.base,
        }
    }

    /// The constant's name.
    #[must_use]
    pub fn name(&self) -> NameId {
        self.base().name
    }

    /// The constant's type.
    #[must_use]
    pub fn ty(&self) -> &Rc<Expr> {
        &self.base().ty
    }

    /// The constant's universe parameters.
    #[must_use]
    pub fn universe_params(&self) -> &[UniverseParamId] {
        &self.base().universe_params
    }

    /// The definition body, if any.
    ///
    /// Axioms, inductives, constructors, recursors, and quotient constants
    /// have no unfoldable body; for definitions, theorems, and opaque
    /// constants it is `Some`.
    #[must_use]
    pub fn body(&self) -> Option<&Rc<Expr>> {
        match self {
            ConstantInfo::Definition(v) => Some(&v.body),
            ConstantInfo::Theorem(v) => Some(&v.body),
            ConstantInfo::Opaque(v) => Some(&v.body),
            _ => None,
        }
    }

    /// The transparency marker that governs δ-unfolding.
    ///
    /// Axioms, inductives, constructors, recursors, and quotient constants
    /// are `Irreducible` because they have no unfoldable body. Theorems and
    /// opaque constants are `Irreducible` by default so that proofs are
    /// never unfolded during ordinary type checking.
    #[must_use]
    pub fn transparency(&self) -> Transparency {
        match self {
            ConstantInfo::Definition(v) => v.transparency,
            ConstantInfo::Theorem(v) => v.transparency,
            _ => Transparency::Irreducible,
        }
    }

    /// Project into a lightweight [`ConstDecl`] view for the NbE engine.
    #[must_use]
    pub fn to_const_decl(&self) -> ConstDecl {
        let base = self.base();
        let (is_constructor, quotient_lift) = match self {
            ConstantInfo::Constructor(_) => (true, None),
            ConstantInfo::Quotient(v) => match &v.kind {
                QuotientKind::Mk => (true, None),
                QuotientKind::Lift {
                    mk,
                    arity,
                    func_index,
                    major_index,
                } => (
                    false,
                    Some(QuotientLiftInfo {
                        mk: *mk,
                        arity: *arity,
                        func_index: *func_index,
                        major_index: *major_index,
                    }),
                ),
                _ => (false, None),
            },
            _ => (false, None),
        };
        ConstDecl {
            name: base.name,
            universe_params: base.universe_params.clone(),
            ty: Rc::clone(&base.ty),
            body: self.body().cloned(),
            transparency: self.transparency(),
            is_constructor,
            recursor_info: match self {
                ConstantInfo::Recursor(v) => Some(project_recursor_info(v)),
                _ => None,
            },
            quotient_lift,
        }
    }

    /// `true` iff this is an inductive family.
    #[must_use]
    pub const fn is_inductive(&self) -> bool {
        matches!(self, ConstantInfo::Inductive(_))
    }

    /// `true` iff this is a constructor.
    ///
    /// Both inductive constructors and `Quot.mk` count: the latter is
    /// constructor-like for the purposes of elimination (see
    /// [`Nbe::try_quot_lift`](crate::nbe::Nbe)).
    #[must_use]
    pub const fn is_constructor(&self) -> bool {
        match self {
            ConstantInfo::Constructor(_) => true,
            ConstantInfo::Quotient(v) => matches!(v.kind, QuotientKind::Mk),
            _ => false,
        }
    }

    /// `true` iff this is a recursor.
    #[must_use]
    pub const fn is_recursor(&self) -> bool {
        matches!(self, ConstantInfo::Recursor(_))
    }

    /// `true` iff this is a quotient-related constant.
    #[must_use]
    pub const fn is_quotient(&self) -> bool {
        matches!(self, ConstantInfo::Quotient(_))
    }
}

/// Project a [`RecursorVal`] into the lightweight view consumed by the
/// NbE engine.
fn project_recursor_info(v: &RecursorVal) -> RecursorInfo {
    RecursorInfo {
        num_params: v.num_params,
        num_indices: v.num_indices,
        num_motives: v.num_motives,
        num_minors: v.num_minors,
        rules: v
            .rules
            .iter()
            .map(|r| RecursorRuleInfo {
                constructor: r.constructor,
                num_fields: r.num_fields,
                rhs: Rc::clone(&r.rhs),
            })
            .collect(),
    }
}

// ---------------------------------------------------------------------------
// Environment errors
// ---------------------------------------------------------------------------

/// Errors that arise when mutating the environment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EnvError {
    /// A declaration with the same name already exists.
    DuplicateDeclaration(NameId),
    /// A constructor references an inductive that is not part of the block
    /// being inserted.
    ConstructorBelongsToUnknownInductive {
        /// The constructor being inserted.
        constructor: NameId,
        /// The inductive the constructor claims to belong to.
        inductive: NameId,
    },
    /// A recursor references an inductive that is not part of the block
    /// being inserted.
    RecursorReferencesUnknownInductive {
        /// The recursor being inserted.
        recursor: NameId,
        /// The offending inductive.
        inductive: NameId,
    },
    /// A constructor index is not in `0 .. num_constructors`.
    ConstructorIndexOutOfRange {
        /// The constructor being inserted.
        constructor: NameId,
        /// The out-of-range index.
        index: u32,
        /// The number of constructors declared for the inductive.
        count: u32,
    },
    /// An inductive declaration failed the positivity check.
    NonPositiveInductive {
        /// The offending inductive.
        inductive: NameId,
        /// Zero-based index of the offending constructor.
        constructor: usize,
    },
    /// An internal invariant was violated. Indicates a kernel bug.
    Internal(String),
    /// A definition's termination obligation was not satisfied.
    NonTerminating {
        /// The offending definition.
        function: NameId,
        /// A rendered explanation from the SCT checker.
        reason: String,
    },
    /// A definition's termination obligation did not match its arity.
    TerminationArityMismatch {
        /// The offending definition.
        function: NameId,
        /// The arity implied by the definition's type (number of Π binders).
        expected: usize,
        /// The number of parameter names supplied in the obligation.
        actual: usize,
    },
}

impl fmt::Display for EnvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EnvError::DuplicateDeclaration(n) => write!(f, "duplicate declaration for {n}"),
            EnvError::ConstructorBelongsToUnknownInductive {
                constructor,
                inductive,
            } => write!(
                f,
                "constructor {constructor} references unknown inductive {inductive}"
            ),
            EnvError::RecursorReferencesUnknownInductive {
                recursor,
                inductive,
            } => write!(
                f,
                "recursor {recursor} references unknown inductive {inductive}"
            ),
            EnvError::ConstructorIndexOutOfRange {
                constructor,
                index,
                count,
            } => write!(
                f,
                "constructor {constructor} has index {index} but inductive has {count} constructors"
            ),
            EnvError::NonPositiveInductive {
                inductive,
                constructor,
            } => write!(
                f,
                "inductive {inductive} fails positivity at constructor #{constructor}"
            ),
            EnvError::Internal(msg) => write!(f, "internal environment error: {msg}"),
            EnvError::NonTerminating { function, reason } => write!(
                f,
                "definition {function} failed the termination check: {reason}"
            ),
            EnvError::TerminationArityMismatch {
                function,
                expected,
                actual,
            } => write!(
                f,
                "definition {function} declares {actual} termination parameter(s) \
                 but its type has {expected} Π binder(s)"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for EnvError {}

// ---------------------------------------------------------------------------
// GlobalEnv
// ---------------------------------------------------------------------------

/// A declaration paired with the cached [`ConstDecl`] view consumed by the
/// NbE engine.
#[derive(Clone, Debug)]
struct EnvEntry {
    /// The full declaration information.
    info: Rc<ConstantInfo>,
    /// The cached light-weight view, computed once at insertion time.
    decl: Rc<ConstDecl>,
}

/// The global environment: all top-level declarations.
#[derive(Default, Debug)]
pub struct GlobalEnv {
    constants: BTreeMap<NameId, EnvEntry>,
}

impl GlobalEnv {
    /// An empty environment.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a single constant.
    ///
    /// Delegates to [`GlobalEnv::add_block`] with a singleton block.
    ///
    /// # Errors
    ///
    /// See [`GlobalEnv::add_block`].
    pub fn add(&mut self, info: ConstantInfo) -> Result<(), EnvError> {
        self.add_block(alloc::vec![info])
    }

    /// Insert a block of constants atomically.
    ///
    /// All declarations are validated together; on failure the environment
    /// is left unchanged.
    ///
    /// ## Termination gate
    ///
    /// If any `Definition` in the block mentions any other member of the
    /// block (including itself), the whole block is treated as a mutually
    /// recursive family and subjected to Size-Change Termination
    /// analysis. Every `Definition` in the block whose body mentions a
    /// block member must supply a termination obligation whose parameter
    /// list matches its type's Π-arity.
    ///
    /// # Errors
    ///
    /// * [`EnvError::DuplicateDeclaration`] for a name repeated within the
    ///   block or already present in the environment.
    /// * [`EnvError::TerminationArityMismatch`] if a recursive definition's
    ///   obligation parameter count disagrees with its type.
    /// * [`EnvError::NonTerminating`] if the joint SCT check fails.
    pub fn add_block(&mut self, infos: Vec<ConstantInfo>) -> Result<(), EnvError> {
        if infos.is_empty() {
            return Ok(());
        }

        // 1. Duplicates within the block.
        let mut seen: BTreeSet<NameId> = BTreeSet::new();
        for info in &infos {
            let name = info.name();
            if !seen.insert(name) {
                return Err(EnvError::DuplicateDeclaration(name));
            }
        }

        // 2. Duplicates against the environment.
        for info in &infos {
            let name = info.name();
            if self.constants.contains_key(&name) {
                return Err(EnvError::DuplicateDeclaration(name));
            }
        }

        // 3. Joint termination check on any Definition group.
        check_termination_block(&infos)?;

        // 4. Commit.
        for info in infos {
            let name = info.name();
            let decl = Rc::new(info.to_const_decl());
            self.constants.insert(
                name,
                EnvEntry {
                    info: Rc::new(info),
                    decl,
                },
            );
        }
        Ok(())
    }

    /// Insert an entire inductive family — the inductive, all of its
    /// constructors, and its recursor — atomically.
    ///
    /// # Errors
    ///
    /// * [`EnvError::DuplicateDeclaration`] if any of the new names is
    ///   already in the environment.
    /// * [`EnvError::ConstructorBelongsToUnknownInductive`] if a constructor
    ///   does not reference the inductive being inserted.
    /// * [`EnvError::ConstructorIndexOutOfRange`] if a constructor's index
    ///   is not in `0 .. num_constructors`.
    /// * [`EnvError::RecursorReferencesUnknownInductive`] if the recursor's
    ///   `all` list contains a name other than the inductive being inserted.
    /// * [`EnvError::NonPositiveInductive`] if a constructor field type
    ///   fails the strict-positivity check.
    ///
    /// On error the environment is left unchanged.
    pub fn add_inductive(
        &mut self,
        ind: InductiveVal,
        ctors: Vec<ConstructorVal>,
        rec: RecursorVal,
    ) -> Result<(), EnvError> {
        let ind_name = ind.base.name;
        let ctor_count =
            u32::try_from(ctors.len()).map_err(|_| EnvError::ConstructorIndexOutOfRange {
                constructor: ind_name,
                index: u32::MAX,
                count: u32::MAX,
            })?;

        // --- Validate -----------------------------------------------------
        for c in &ctors {
            if c.inductive != ind_name {
                return Err(EnvError::ConstructorBelongsToUnknownInductive {
                    constructor: c.base.name,
                    inductive: c.inductive,
                });
            }
            if c.index >= ctor_count {
                return Err(EnvError::ConstructorIndexOutOfRange {
                    constructor: c.base.name,
                    index: c.index,
                    count: ctor_count,
                });
            }
        }
        for name in &rec.all {
            if *name != ind_name {
                return Err(EnvError::RecursorReferencesUnknownInductive {
                    recursor: rec.base.name,
                    inductive: *name,
                });
            }
        }

        // --- Check for duplicates ----------------------------------------
        if self.constants.contains_key(&ind_name) {
            return Err(EnvError::DuplicateDeclaration(ind_name));
        }
        let mut seen_ctor_names: BTreeSet<NameId> = BTreeSet::new();
        for c in &ctors {
            if !seen_ctor_names.insert(c.base.name) {
                return Err(EnvError::DuplicateDeclaration(c.base.name));
            }
        }
        for c in &ctors {
            if self.constants.contains_key(&c.base.name) {
                return Err(EnvError::DuplicateDeclaration(c.base.name));
            }
        }
        if self.constants.contains_key(&rec.base.name) {
            return Err(EnvError::DuplicateDeclaration(rec.base.name));
        }

        // --- Positivity ----------------------------------------------------
        //
        // Read-only check against the current environment. Any inductive
        // inserted by this call is not yet visible, which is correct: the
        // family is identified by name, not by env lookup.
        {
            let checker = PositivityChecker::new(self);
            match checker.check_inductive(&ind, &ctors) {
                Ok(()) => {}
                Err(KernelError::NonPositiveInductive {
                    inductive,
                    constructor,
                }) => {
                    return Err(EnvError::NonPositiveInductive {
                        inductive,
                        constructor,
                    });
                }
                Err(other) => {
                    // `check_inductive` is documented to return only
                    // `NonPositiveInductive`; anything else is a bug.
                    return Err(EnvError::Internal(format!(
                        "positivity check returned unexpected error: {other}"
                    )));
                }
            }
        }
        // The immutable borrow of `self` ends here; mutation below is safe.

        // --- Commit -------------------------------------------------------
        let ind_info = ConstantInfo::Inductive(ind);
        let ind_decl = Rc::new(ind_info.to_const_decl());
        self.constants.insert(
            ind_name,
            EnvEntry {
                info: Rc::new(ind_info),
                decl: ind_decl,
            },
        );
        for c in ctors {
            let name = c.base.name;
            let info = ConstantInfo::Constructor(c);
            let decl = Rc::new(info.to_const_decl());
            self.constants.insert(
                name,
                EnvEntry {
                    info: Rc::new(info),
                    decl,
                },
            );
        }
        let rec_name = rec.base.name;
        let rec_info = ConstantInfo::Recursor(rec);
        let rec_decl = Rc::new(rec_info.to_const_decl());
        self.constants.insert(
            rec_name,
            EnvEntry {
                info: Rc::new(rec_info),
                decl: rec_decl,
            },
        );
        Ok(())
    }

    /// Look up a constant by name.
    #[must_use]
    pub fn find(&self, name: NameId) -> Option<Rc<ConstantInfo>> {
        self.constants.get(&name).map(|e| Rc::clone(&e.info))
    }

    /// `true` iff the environment contains a declaration for `name`.
    #[must_use]
    pub fn contains(&self, name: NameId) -> bool {
        self.constants.contains_key(&name)
    }

    /// Number of declarations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.constants.len()
    }

    /// `true` iff the environment is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.constants.is_empty()
    }

    /// Iterate over all declarations in name order.
    pub fn iter(&self) -> impl Iterator<Item = (&NameId, &Rc<ConstantInfo>)> {
        self.constants
            .iter()
            .map(|(name, entry)| (name, &entry.info))
    }
}

impl ConstEnv for GlobalEnv {
    fn lookup(&self, name: NameId) -> Option<Rc<ConstDecl>> {
        self.constants.get(&name).map(|e| Rc::clone(&e.decl))
    }
}

// ---------------------------------------------------------------------------
// LocalContext
// ---------------------------------------------------------------------------

/// A single local binder.
#[derive(Clone, Debug)]
pub struct LocalDecl {
    /// The binder's user-visible name.
    pub name: NameId,
    /// Whether the binder is explicit, implicit, strict-implicit, or an
    /// instance argument.
    pub binder_info: BinderInfo,
    /// The binder's type, expressed in the context *before* the binder is
    /// pushed.
    pub ty: Rc<Expr>,
    /// For let-bound variables, the bound value. `None` for λ- and
    /// Π-bound variables.
    pub value: Option<Rc<Expr>>,
}

impl LocalDecl {
    /// Construct a λ- or Π-binder.
    #[must_use]
    pub fn binder(name: NameId, binder_info: BinderInfo, ty: Rc<Expr>) -> Self {
        LocalDecl {
            name,
            binder_info,
            ty,
            value: None,
        }
    }

    /// Construct a let-bound declaration.
    #[must_use]
    pub fn let_bound(name: NameId, ty: Rc<Expr>, value: Rc<Expr>) -> Self {
        LocalDecl {
            name,
            binder_info: BinderInfo::Default,
            ty,
            value: Some(value),
        }
    }

    /// `true` iff this declaration is let-bound.
    #[must_use]
    pub const fn is_let(&self) -> bool {
        self.value.is_some()
    }
}

/// A persistent, immutable local typing context.
///
/// The head of the linked list is the *innermost* binder, so de Bruijn index
/// `0` refers to the head. Extending is O(1); lookup by index is O(index).
#[derive(Clone)]
pub struct LocalContext(Rc<LocalContextNode>);

#[derive(Debug)]
enum LocalContextNode {
    Empty,
    Cons(LocalDecl, LocalContext),
}

impl LocalContext {
    /// The empty context.
    #[must_use]
    pub fn empty() -> Self {
        LocalContext(Rc::new(LocalContextNode::Empty))
    }

    /// Push a declaration onto the front (innermost position).
    #[must_use]
    pub fn push(&self, decl: LocalDecl) -> Self {
        LocalContext(Rc::new(LocalContextNode::Cons(decl, self.clone())))
    }

    /// Number of bindings.
    #[must_use]
    pub fn len(&self) -> usize {
        let mut n = 0;
        let mut cur = &self.0;
        loop {
            match &**cur {
                LocalContextNode::Empty => return n,
                LocalContextNode::Cons(_, rest) => {
                    n += 1;
                    cur = &rest.0;
                }
            }
        }
    }

    /// `true` iff there are no bindings.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        matches!(&*self.0, LocalContextNode::Empty)
    }

    /// Look up the declaration at the given de Bruijn index.
    ///
    /// # Panics
    ///
    /// Panics if `idx` is out of range. This is a programming error: a
    /// well-typed term never references an unbound variable.
    #[must_use]
    pub fn lookup(&self, idx: DeBruijnIndex) -> &LocalDecl {
        let mut cur = &self.0;
        let mut i = idx.0 as usize;
        loop {
            match &**cur {
                LocalContextNode::Empty => {
                    panic!(
                        "LocalContext::lookup: index {idx} out of range (context has size {})",
                        self.len()
                    )
                }
                LocalContextNode::Cons(decl, rest) => {
                    if i == 0 {
                        return decl;
                    }
                    i -= 1;
                    cur = &rest.0;
                }
            }
        }
    }

    /// Iterate over the bindings, innermost first.
    #[must_use]
    pub fn iter(&self) -> LocalContextIter<'_> {
        LocalContextIter {
            current: Some(&self.0),
        }
    }

    /// Build the semantic environment of fresh neutral variables that
    /// corresponds to this context.
    ///
    /// The variable at de Bruijn *index* `i` receives de Bruijn *level*
    /// `n - 1 - i`, where `n` is [`LocalContext::len`]. This is the
    /// representation that [`Nbe`](crate::nbe::Nbe) expects.
    #[must_use]
    pub fn to_env(&self) -> NbeEnv {
        let n = self.len();
        let mut env = NbeEnv::empty();
        // Extend in order from outermost (level 0) to innermost (level n-1).
        // Since `extend` prepends, the innermost ends up at the head of the
        // list — matching index 0.
        for i in 0..n {
            env = env.extend(Value::fresh_var(DbLevel(i as u32)));
        }
        env
    }
}

impl fmt::Debug for LocalContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LocalContext(len={})", self.len())
    }
}

/// Iterator over the bindings of a [`LocalContext`], innermost first.
pub struct LocalContextIter<'a> {
    current: Option<&'a LocalContextNode>,
}

impl<'a> Iterator for LocalContextIter<'a> {
    type Item = &'a LocalDecl;

    fn next(&mut self) -> Option<Self::Item> {
        match self.current? {
            LocalContextNode::Empty => None,
            LocalContextNode::Cons(decl, rest) => {
                self.current = Some(&rest.0);
                Some(decl)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Termination gate
// ---------------------------------------------------------------------------

/// Termination gate for a block of definitions.
///
/// * If no definition in the block mentions any block member, the check
///   is a no-op.
/// * Otherwise every such definition must carry an obligation whose
///   parameter list matches its Π-arity, and the joint call graph must
///   pass SCT.
fn check_termination_block(infos: &[ConstantInfo]) -> Result<(), EnvError> {
    let defs: Vec<&DefinitionVal> = infos
        .iter()
        .filter_map(|i| match i {
            ConstantInfo::Definition(d) => Some(d),
            _ => None,
        })
        .collect();
    if defs.is_empty() {
        return Ok(());
    }

    // Fast path: no block member mentions any block member.
    let block_names: Vec<NameId> = defs.iter().map(|d| d.base.name).collect();
    let mut any_mentions = false;
    'outer: for d in &defs {
        for name in &block_names {
            if body_mentions_constant(&d.body, *name) {
                any_mentions = true;
                break 'outer;
            }
        }
    }
    if !any_mentions {
        return Ok(());
    }

    // Arity check: every definition whose body mentions a block member
    // must declare exactly `pi_arity` parameters.
    for d in &defs {
        let mentions = block_names
            .iter()
            .any(|n| body_mentions_constant(&d.body, *n));
        if !mentions {
            continue;
        }
        let expected = d.base.ty.pi_arity();
        let actual = d.termination.params.len();
        if expected != actual {
            return Err(EnvError::TerminationArityMismatch {
                function: d.base.name,
                expected,
                actual,
            });
        }
    }

    // Joint SCT.
    let functions: Vec<(NameId, Vec<NameId>)> = defs
        .iter()
        .map(|d| (d.base.name, d.termination.params.clone()))
        .collect();
    let bodies: Vec<Expr> = defs.iter().map(|d| (*d.body).clone()).collect();
    let checker = TerminationChecker::new();
    let graph = checker.build_mutual_call_graph(&functions, &bodies);
    match checker.check(&graph) {
        Ok(_) => Ok(()),
        Err(e) => Err(EnvError::NonTerminating {
            function: defs[0].base.name,
            reason: alloc::format!("{e}"),
        }),
    }
}

/// `true` iff `body` mentions `name` as a constant.
fn body_mentions_constant(body: &Expr, name: NameId) -> bool {
    match body {
        Expr::Const(n, _) => *n == name,
        Expr::Sort(_) | Expr::Var(_) | Expr::Lit(_) => false,
        Expr::App(f, a) => body_mentions_constant(f, name) || body_mentions_constant(a, name),
        Expr::Pi(_, _, d, c) | Expr::Lam(_, _, d, c) => {
            body_mentions_constant(d, name) || body_mentions_constant(c, name)
        }
        Expr::Let(_, t, v, b) => {
            body_mentions_constant(t, name)
                || body_mentions_constant(v, name)
                || body_mentions_constant(b, name)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{BinderInfo, Expr};
    use crate::level::{Level, UniverseParamId};
    use crate::name::NameTable;
    use crate::nbe::{ConstEnv, Head, Nbe, TransparencyMode};
    use alloc::boxed::Box;
    use alloc::vec;

    // ---- helpers -----------------------------------------------------

    fn cv(name: NameId, ty: Expr) -> ConstantVal {
        ConstantVal {
            name,
            universe_params: Vec::new(),
            ty: Rc::new(ty),
        }
    }

    fn axiom(name: NameId, ty: Expr) -> ConstantInfo {
        ConstantInfo::Axiom(cv(name, ty))
    }

    fn definition(name: NameId, ty: Expr, body: Expr, transparency: Transparency) -> ConstantInfo {
        ConstantInfo::Definition(DefinitionVal {
            base: cv(name, ty),
            body: Rc::new(body),
            transparency,
            termination: TerminationObligation::none(),
        })
    }

    fn theorem(name: NameId, ty: Expr, body: Expr) -> ConstantInfo {
        ConstantInfo::Theorem(TheoremVal {
            base: cv(name, ty),
            body: Rc::new(body),
            transparency: Transparency::Irreducible,
        })
    }

    fn var(idx: u32) -> Expr {
        Expr::Var(DeBruijnIndex(idx))
    }

    // ---- GlobalEnv basics --------------------------------------------

    #[test]
    fn empty_env_is_empty() {
        let env = GlobalEnv::new();
        assert!(env.is_empty());
        assert_eq!(env.len(), 0);
    }

    #[test]
    fn add_and_find_axiom() {
        let mut names = NameTable::new();
        let n = names.intern("P");
        let mut env = GlobalEnv::new();
        env.add(axiom(n, Expr::prop())).unwrap();
        let info = env.find(n).unwrap();
        assert!(matches!(&*info, ConstantInfo::Axiom(_)));
        assert_eq!(info.name(), n);
    }

    #[test]
    fn duplicate_declaration_is_rejected() {
        let mut names = NameTable::new();
        let n = names.intern("P");
        let mut env = GlobalEnv::new();
        env.add(axiom(n, Expr::prop())).unwrap();
        let err = env.add(axiom(n, Expr::prop())).unwrap_err();
        assert_eq!(err, EnvError::DuplicateDeclaration(n));
        // Environment unchanged.
        assert_eq!(env.len(), 1);
    }

    #[test]
    fn definition_body_is_stored() {
        let mut names = NameTable::new();
        let n = names.intern("c");
        let body = Expr::Sort(Level::zero());
        let mut env = GlobalEnv::new();
        env.add(definition(
            n,
            Expr::Sort(Level::zero().succ()),
            body.clone(),
            Transparency::Semireducible,
        ))
        .unwrap();
        let info = env.find(n).unwrap();
        assert_eq!(info.body().map(|b| (**b).clone()), Some(body));
    }

    #[test]
    fn theorem_is_opaque() {
        let mut names = NameTable::new();
        let n = names.intern("thm");
        let body = Expr::Sort(Level::zero());
        let mut env = GlobalEnv::new();
        env.add(theorem(n, Expr::prop(), body)).unwrap();
        let info = env.find(n).unwrap();
        assert_eq!(info.transparency(), Transparency::Irreducible);
    }

    // ---- Inductive insertion -----------------------------------------

    fn make_inductive_block(
        names: &mut NameTable,
    ) -> (InductiveVal, Vec<ConstructorVal>, RecursorVal) {
        let nat = names.intern("Nat");
        let zero = names.intern("Nat.zero");
        let succ = names.intern("Nat.succ");
        let rec = names.intern("Nat.rec");

        let ind = InductiveVal {
            base: cv(nat, Expr::type0()),
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
                base: cv(zero, Expr::type0()),
                inductive: nat,
                index: 0,
                num_params: 0,
                num_fields: 0,
                is_unsafe: false,
            },
            ConstructorVal {
                base: cv(succ, Expr::type0()),
                inductive: nat,
                index: 1,
                num_params: 0,
                num_fields: 1,
                is_unsafe: false,
            },
        ];
        let recursor = RecursorVal {
            base: cv(rec, Expr::type0()),
            all: vec![nat],
            num_params: 0,
            num_indices: 0,
            num_motives: 1,
            num_minors: 2,
            rules: vec![
                RecursorRule {
                    constructor: zero,
                    num_fields: 0,
                    rhs: Rc::new(var(0)),
                },
                RecursorRule {
                    constructor: succ,
                    num_fields: 1,
                    rhs: Rc::new(Expr::app(var(0), var(3))),
                },
            ],
            is_k: false,
            is_unsafe: false,
        };
        (ind, ctors, recursor)
    }

    #[test]
    fn add_inductive_family_atomically() {
        let mut names = NameTable::new();
        let (ind, ctors, rec) = make_inductive_block(&mut names);
        let nat = ind.base.name;
        let rec_name = rec.base.name;
        let mut env = GlobalEnv::new();
        env.add_inductive(ind, ctors, rec).unwrap();
        assert!(env.find(nat).unwrap().is_inductive());
        assert!(env.find(rec_name).unwrap().is_recursor());
        assert_eq!(env.len(), 4); // Nat, Nat.zero, Nat.succ, Nat.rec
    }

    #[test]
    fn constructor_with_wrong_inductive_is_rejected() {
        let mut names = NameTable::new();
        let (ind, mut ctors, rec) = make_inductive_block(&mut names);
        let bogus = names.intern("Bogus");
        ctors[0].inductive = bogus;
        let mut env = GlobalEnv::new();
        let err = env.add_inductive(ind, ctors, rec).unwrap_err();
        assert!(matches!(
            err,
            EnvError::ConstructorBelongsToUnknownInductive { .. }
        ));
        assert!(env.is_empty());
    }

    #[test]
    fn constructor_index_out_of_range_is_rejected() {
        let mut names = NameTable::new();
        let (ind, mut ctors, rec) = make_inductive_block(&mut names);
        ctors[0].index = 7;
        let mut env = GlobalEnv::new();
        let err = env.add_inductive(ind, ctors, rec).unwrap_err();
        assert!(matches!(err, EnvError::ConstructorIndexOutOfRange { .. }));
        assert!(env.is_empty());
    }

    #[test]
    fn add_inductive_rejects_non_positive() {
        let mut names = NameTable::new();
        let bad = names.intern("Bad");
        let mk = names.intern("Bad.mk");
        let x = names.intern("X");

        let mut env = GlobalEnv::new();
        env.add(axiom(x, Expr::type0())).unwrap();

        let ind = InductiveVal {
            base: cv(bad, Expr::type0()),
            num_params: 0,
            num_indices: 0,
            all: vec![bad],
            constructors: vec![mk],
            is_recursive: true,
            is_nested: false,
            is_unsafe: false,
        };
        // field : (Bad -> X) -> Bad
        let field = Expr::Pi(
            BinderInfo::Default,
            NameId(0),
            Box::new(Expr::Const(bad, Vec::new())),
            Box::new(Expr::Const(x, Vec::new())),
        );
        let ty = Expr::Pi(
            BinderInfo::Default,
            NameId(0),
            Box::new(field),
            Box::new(Expr::Const(bad, Vec::new())),
        );
        let ctor = ConstructorVal {
            base: cv(mk, ty),
            inductive: bad,
            index: 0,
            num_params: 0,
            num_fields: 1,
            is_unsafe: false,
        };
        let rec = RecursorVal {
            base: cv(names.intern("Bad.rec"), Expr::type0()),
            all: vec![bad],
            num_params: 0,
            num_indices: 0,
            num_motives: 1,
            num_minors: 0,
            rules: Vec::new(),
            is_k: false,
            is_unsafe: false,
        };

        let err = env.add_inductive(ind, vec![ctor], rec).unwrap_err();
        assert!(matches!(
            err,
            EnvError::NonPositiveInductive {
                inductive,
                constructor
            } if inductive == bad && constructor == 0
        ));
        // Nothing was committed.
        assert!(env.find(bad).is_none());
    }

    #[test]
    fn add_inductive_accepts_positive() {
        let mut names = NameTable::new();
        let (ind, ctors, rec) = make_inductive_block(&mut names);
        let mut env = GlobalEnv::new();
        env.add_inductive(ind, ctors, rec).unwrap();
        // Nat, Nat.zero, Nat.succ, Nat.rec
        assert_eq!(env.len(), 4);
    }

    #[test]
    fn add_inductive_rejects_duplicate_constructor_names() {
        let mut names = NameTable::new();
        let (ind, mut ctors, rec) = make_inductive_block(&mut names);
        let zero_name = ctors[0].base.name;
        ctors[1].base.name = zero_name;
        let mut env = GlobalEnv::new();
        let err = env.add_inductive(ind, ctors, rec).unwrap_err();
        assert_eq!(err, EnvError::DuplicateDeclaration(zero_name));
        assert!(env.is_empty());
    }

    #[test]
    fn const_env_lookup_serves_cached_decl() {
        let mut names = NameTable::new();
        let (ind, ctors, rec) = make_inductive_block(&mut names);
        let succ_name = ctors[1].base.name;
        let mut env = GlobalEnv::new();
        env.add_inductive(ind, ctors, rec).unwrap();
        let decl = env.lookup(succ_name).expect("constructor must be visible");
        assert_eq!(decl.name, succ_name);
        assert!(decl.is_constructor);
        assert!(decl.body.is_none());
    }

    // ---- Termination gate --------------------------------------------

    #[test]
    fn add_rejects_self_recursive_non_terminating() {
        let mut names = NameTable::new();
        let f = names.intern("f");
        let n = names.intern("n");

        // f : Nat -> Nat, body := f n
        let nat = names.intern("Nat");
        let ty = Expr::Pi(
            BinderInfo::Default,
            names.intern("_"),
            Box::new(Expr::Const(nat, Vec::new())),
            Box::new(Expr::Const(nat, Vec::new())),
        );
        let body = Expr::app(
            Expr::Const(f, Vec::new()),
            Expr::Var(crate::DeBruijnIndex(0)),
        );
        let mut env = GlobalEnv::new();
        // Nat must exist for the body to typecheck later, but `add` itself
        // only needs the names.
        env.add(axiom(nat, Expr::type0())).unwrap();

        let err = env
            .add(ConstantInfo::Definition(DefinitionVal {
                base: cv(f, ty),
                body: Rc::new(body),
                transparency: Transparency::Semireducible,
                termination: TerminationObligation::recursive(vec![n]),
            }))
            .unwrap_err();
        assert!(matches!(err, EnvError::NonTerminating { .. }));
        assert!(env.find(f).is_none());
    }

    #[test]
    fn add_accepts_non_recursive_without_params() {
        let mut names = NameTable::new();
        let nat = names.intern("Nat");
        let id = names.intern("id");
        let x = names.intern("x");

        // def id : Nat -> Nat := λ x. x
        let ty = Expr::Pi(
            BinderInfo::Default,
            names.intern("_"),
            Box::new(Expr::Const(nat, Vec::new())),
            Box::new(Expr::Const(nat, Vec::new())),
        );
        let body = Expr::lam(
            x,
            Expr::Const(nat, Vec::new()),
            Expr::Var(crate::DeBruijnIndex(0)),
        );

        let mut env = GlobalEnv::new();
        env.add(axiom(nat, Expr::type0())).unwrap();
        env.add(ConstantInfo::Definition(DefinitionVal {
            base: cv(id, ty),
            body: Rc::new(body),
            transparency: Transparency::Semireducible,
            termination: TerminationObligation::none(),
        }))
        .unwrap();
        assert!(env.find(id).is_some());
    }

    #[test]
    fn add_rejects_recursive_with_wrong_arity() {
        let mut names = NameTable::new();
        let nat = names.intern("Nat");
        let f = names.intern("f");

        // f : Nat -> Nat -> Nat, body mentions f but obligation has 0 params.
        let ty = Expr::Pi(
            BinderInfo::Default,
            names.intern("_"),
            Box::new(Expr::Const(nat, Vec::new())),
            Box::new(Expr::Pi(
                BinderInfo::Default,
                names.intern("_"),
                Box::new(Expr::Const(nat, Vec::new())),
                Box::new(Expr::Const(nat, Vec::new())),
            )),
        );
        // Body: f (Var 0) (Var 0) — a self-reference sufficient to trip the
        // recursion detector.
        let body = Expr::Const(f, Vec::new()).apps([
            Expr::Var(crate::DeBruijnIndex(0)),
            Expr::Var(crate::DeBruijnIndex(0)),
        ]);

        let mut env = GlobalEnv::new();
        env.add(axiom(nat, Expr::type0())).unwrap();
        let err = env
            .add(ConstantInfo::Definition(DefinitionVal {
                base: cv(f, ty),
                body: Rc::new(body),
                transparency: Transparency::Semireducible,
                termination: TerminationObligation::none(), // 0 params
            }))
            .unwrap_err();
        assert!(matches!(
            err,
            EnvError::TerminationArityMismatch {
                expected: 2,
                actual: 0,
                ..
            }
        ));
        assert!(env.find(f).is_none());
    }

    // ---- Mutual blocks -------------------------------------------------

    #[test]
    fn add_block_rejects_mutual_recursion() {
        let mut names = NameTable::new();
        let nat = names.intern("Nat");
        let even = names.intern("even");
        let odd = names.intern("odd");
        let n = names.intern("n");

        let mut env = GlobalEnv::new();
        env.add(axiom(nat, Expr::type0())).unwrap();

        let ty = Expr::Pi(
            BinderInfo::Default,
            names.intern("_"),
            Box::new(Expr::Const(nat, Vec::new())),
            Box::new(Expr::Const(nat, Vec::new())),
        );
        let even_body = Expr::app(
            Expr::Const(odd, Vec::new()),
            Expr::Var(crate::DeBruijnIndex(0)),
        );
        let odd_body = Expr::app(
            Expr::Const(even, Vec::new()),
            Expr::Var(crate::DeBruijnIndex(0)),
        );

        let err = env
            .add_block(vec![
                ConstantInfo::Definition(DefinitionVal {
                    base: cv(even, ty.clone()),
                    body: Rc::new(even_body),
                    transparency: Transparency::Semireducible,
                    termination: TerminationObligation::recursive(vec![n]),
                }),
                ConstantInfo::Definition(DefinitionVal {
                    base: cv(odd, ty),
                    body: Rc::new(odd_body),
                    transparency: Transparency::Semireducible,
                    termination: TerminationObligation::recursive(vec![n]),
                }),
            ])
            .unwrap_err();
        assert!(matches!(err, EnvError::NonTerminating { .. }));
        // Nothing committed.
        assert!(env.find(even).is_none());
        assert!(env.find(odd).is_none());
    }

    #[test]
    fn add_block_accepts_independent_definitions() {
        let mut names = NameTable::new();
        let nat = names.intern("Nat");
        let f = names.intern("f");
        let g = names.intern("g");

        let mut env = GlobalEnv::new();
        env.add(axiom(nat, Expr::type0())).unwrap();
        let ty = Expr::Pi(
            BinderInfo::Default,
            names.intern("_"),
            Box::new(Expr::Const(nat, Vec::new())),
            Box::new(Expr::Const(nat, Vec::new())),
        );
        // Bodies don't mention f or g.
        let body = Expr::Var(crate::DeBruijnIndex(0));

        env.add_block(vec![
            ConstantInfo::Definition(DefinitionVal {
                base: cv(f, ty.clone()),
                body: Rc::new(body.clone()),
                transparency: Transparency::Semireducible,
                termination: TerminationObligation::none(),
            }),
            ConstantInfo::Definition(DefinitionVal {
                base: cv(g, ty),
                body: Rc::new(body),
                transparency: Transparency::Semireducible,
                termination: TerminationObligation::none(),
            }),
        ])
        .unwrap();
        assert!(env.find(f).is_some());
        assert!(env.find(g).is_some());
    }

    #[test]
    fn add_block_rejects_internal_duplicate_names() {
        let mut names = NameTable::new();
        let nat = names.intern("Nat");
        let f = names.intern("f");
        let mut env = GlobalEnv::new();
        env.add(axiom(nat, Expr::type0())).unwrap();
        let err = env
            .add_block(vec![axiom(f, Expr::type0()), axiom(f, Expr::type0())])
            .unwrap_err();
        assert_eq!(err, EnvError::DuplicateDeclaration(f));
    }

    // ---- ConstEnv bridge ---------------------------------------------

    #[test]
    fn const_env_projects_axiom() {
        let mut names = NameTable::new();
        let n = names.intern("P");
        let mut env = GlobalEnv::new();
        env.add(axiom(n, Expr::prop())).unwrap();
        let decl = env.lookup(n).unwrap();
        assert_eq!(decl.name, n);
        assert!(decl.body.is_none());
        assert_eq!(decl.transparency, Transparency::Irreducible);
    }

    #[test]
    fn const_env_projects_definition_body() {
        let mut names = NameTable::new();
        let n = names.intern("c");
        let body = Expr::Sort(Level::zero());
        let mut env = GlobalEnv::new();
        env.add(definition(
            n,
            Expr::Sort(Level::zero().succ()),
            body.clone(),
            Transparency::Semireducible,
        ))
        .unwrap();
        let decl = env.lookup(n).unwrap();
        assert_eq!(decl.body.as_ref().map(|b| (**b).clone()), Some(body));
        assert_eq!(decl.transparency, Transparency::Semireducible);
    }

    #[test]
    fn const_env_projects_theorem_as_opaque() {
        let mut names = NameTable::new();
        let n = names.intern("thm");
        let mut env = GlobalEnv::new();
        env.add(theorem(n, Expr::prop(), Expr::prop())).unwrap();
        let decl = env.lookup(n).unwrap();
        assert!(decl.body.is_some());
        assert_eq!(decl.transparency, Transparency::Irreducible);
    }

    // ---- Local context -----------------------------------------------

    #[test]
    fn local_context_push_and_lookup() {
        let mut names = NameTable::new();
        let a = names.intern("a");
        let b = names.intern("b");
        let mut ctx = LocalContext::empty();
        ctx = ctx.push(LocalDecl::binder(
            a,
            BinderInfo::Default,
            Rc::new(Expr::prop()),
        ));
        ctx = ctx.push(LocalDecl::binder(
            b,
            BinderInfo::Default,
            Rc::new(Expr::prop()),
        ));
        assert_eq!(ctx.len(), 2);
        assert_eq!(ctx.lookup(DeBruijnIndex(0)).name, b); // innermost
        assert_eq!(ctx.lookup(DeBruijnIndex(1)).name, a); // outermost
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn local_context_lookup_out_of_range_panics() {
        let ctx = LocalContext::empty();
        let _ = ctx.lookup(DeBruijnIndex(0));
    }

    #[test]
    fn local_context_iter_is_innermost_first() {
        let mut names = NameTable::new();
        let a = names.intern("a");
        let b = names.intern("b");
        let c = names.intern("c");
        let mut ctx = LocalContext::empty();
        ctx = ctx.push(LocalDecl::binder(
            a,
            BinderInfo::Default,
            Rc::new(Expr::prop()),
        ));
        ctx = ctx.push(LocalDecl::binder(
            b,
            BinderInfo::Default,
            Rc::new(Expr::prop()),
        ));
        ctx = ctx.push(LocalDecl::binder(
            c,
            BinderInfo::Default,
            Rc::new(Expr::prop()),
        ));
        let order: Vec<NameId> = ctx.iter().map(|d| d.name).collect();
        assert_eq!(order, vec![c, b, a]);
    }

    #[test]
    fn local_context_to_env_assigns_levels() {
        let mut names = NameTable::new();
        let a = names.intern("a");
        let b = names.intern("b");
        let mut ctx = LocalContext::empty();
        ctx = ctx.push(LocalDecl::binder(
            a,
            BinderInfo::Default,
            Rc::new(Expr::prop()),
        ));
        ctx = ctx.push(LocalDecl::binder(
            b,
            BinderInfo::Default,
            Rc::new(Expr::prop()),
        ));
        let env = ctx.to_env();
        // Index 0 (innermost, b) → level 1
        // Index 1 (outermost, a) → level 0
        match &*env.lookup(DeBruijnIndex(0)) {
            Value::Neutral(Head::Var(lvl), spine) => {
                assert_eq!(*lvl, DbLevel(1));
                assert!(spine.is_empty());
            }
            other => panic!("expected neutral var, got {other:?}"),
        }
        match &*env.lookup(DeBruijnIndex(1)) {
            Value::Neutral(Head::Var(lvl), _) => assert_eq!(*lvl, DbLevel(0)),
            other => panic!("expected neutral var, got {other:?}"),
        }
    }

    #[test]
    fn local_context_let_bound() {
        let mut names = NameTable::new();
        let x = names.intern("x");
        let ctx = LocalContext::empty().push(LocalDecl::let_bound(
            x,
            Rc::new(Expr::prop()),
            Rc::new(Expr::prop()),
        ));
        assert!(ctx.lookup(DeBruijnIndex(0)).is_let());
    }

    // ---- End-to-end with NbE ----------------------------------------

    #[test]
    fn nbe_unfolds_definition_from_env() {
        let mut names = NameTable::new();
        let c = names.intern("c");
        // def c : Sort(1) := Sort(0)
        let mut env = GlobalEnv::new();
        env.add(definition(
            c,
            Expr::Sort(Level::zero().succ()),
            Expr::Sort(Level::zero()),
            Transparency::Semireducible,
        ))
        .unwrap();

        let nbe = Nbe::new(&env, TransparencyMode::Semireducible);
        let term = Expr::Const(c, Vec::new());
        let value = nbe.eval(&NbeEnv::empty(), &term);
        match &*value {
            Value::Sort(l) => assert_eq!(l.normalize(), Level::zero()),
            other => panic!("expected Sort(0), got {other:?}"),
        }
    }

    #[test]
    fn nbe_does_not_unfold_theorem() {
        let mut names = NameTable::new();
        let t = names.intern("t");
        let mut env = GlobalEnv::new();
        env.add(theorem(t, Expr::prop(), Expr::prop())).unwrap();

        let nbe = Nbe::new(&env, TransparencyMode::Semireducible);
        let term = Expr::Const(t, Vec::new());
        let value = nbe.eval(&NbeEnv::empty(), &term);
        // Theorem is opaque: evaluation is stuck on the constant.
        match &*value {
            Value::Neutral(Head::Const(name, levels), spine) => {
                assert_eq!(*name, t);
                assert!(levels.is_empty());
                assert!(spine.is_empty());
            }
            other => panic!("expected stuck const, got {other:?}"),
        }
    }

    #[test]
    fn nbe_does_not_unfold_axiom() {
        let mut names = NameTable::new();
        let a = names.intern("A");
        let mut env = GlobalEnv::new();
        env.add(axiom(a, Expr::prop())).unwrap();

        let nbe = Nbe::new(&env, TransparencyMode::All);
        let term = Expr::Const(a, Vec::new());
        let value = nbe.eval(&NbeEnv::empty(), &term);
        // Even in `All` mode, an axiom has no body to unfold.
        match &*value {
            Value::Neutral(Head::Const(name, _), _) => assert_eq!(*name, a),
            other => panic!("expected stuck axiom, got {other:?}"),
        }
    }

    #[test]
    fn nbe_universe_polymorphic_definition() {
        // def id.{u} : Π (A : Sort(u)). Π (x : A). A
        //   := λ (A : Sort(u)). λ (x : A). x
        let mut names = NameTable::new();
        let id = names.intern("id");
        let a_name = names.intern("A");
        let x_name = names.intern("x");
        let u = UniverseParamId(42);

        // Body: λ A. λ x. x
        //   outer dom: Sort(u)
        //   inner dom: Var(0)  (A)
        //   inner body: Var(0) (x)
        let body = Expr::Lam(
            BinderInfo::Default,
            a_name,
            Box::new(Expr::Sort(Level::param(u))),
            Box::new(Expr::Lam(
                BinderInfo::Default,
                x_name,
                Box::new(var(0)),
                Box::new(var(0)),
            )),
        );
        // Type: Π A. Π x. A
        let ty = Expr::Pi(
            BinderInfo::Default,
            a_name,
            Box::new(Expr::Sort(Level::param(u))),
            Box::new(Expr::Pi(
                BinderInfo::Default,
                x_name,
                Box::new(var(0)),
                Box::new(var(1)),
            )),
        );
        let mut env = GlobalEnv::new();
        env.add(ConstantInfo::Definition(DefinitionVal {
            base: ConstantVal {
                name: id,
                universe_params: vec![u],
                ty: Rc::new(ty),
            },
            body: Rc::new(body),
            transparency: Transparency::Semireducible,
            termination: TerminationObligation::none(),
        }))
        .unwrap();

        // Instantiate id.{0} — should unfold to a lambda.
        let nbe = Nbe::new(&env, TransparencyMode::Semireducible);
        let term = Expr::Const(id, vec![Level::zero()]);
        let value = nbe.eval(&NbeEnv::empty(), &term);
        assert!(matches!(&*value, Value::Lam(_, _, _, _)));
    }

    #[test]
    fn nbe_uses_transparency_mode() {
        let mut names = NameTable::new();
        let c = names.intern("c");
        let mut env = GlobalEnv::new();
        env.add(definition(
            c,
            Expr::Sort(Level::zero().succ()),
            Expr::Sort(Level::zero()),
            Transparency::Semireducible,
        ))
        .unwrap();

        // In Reducible-only mode, a Semireducible definition must not unfold.
        let nbe = Nbe::new(&env, TransparencyMode::Reducible);
        let term = Expr::Const(c, Vec::new());
        let value = nbe.eval(&NbeEnv::empty(), &term);
        assert!(matches!(&*value, Value::Neutral(Head::Const(_, _), _)));
    }
}
