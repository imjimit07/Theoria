//! Elaboration errors.
//!
//! [`ElaborateErrorKind`] exists for programmatic matching;
//! [`ElaborateError::message`] is for humans. The CLI renders both with
//! source spans.

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;
use theoria_kernel::KernelError;
use theoria_syntax::Span;

/// A surface-level elaboration error with a source span.
#[derive(Clone, Debug)]
pub struct ElaborateError {
    /// Where the error occurred.
    pub span: Span,
    /// The machine-readable category.
    pub kind: ElaborateErrorKind,
    /// Human-readable description.
    pub message: String,
}

impl ElaborateError {
    /// Construct an error from its parts.
    #[must_use]
    pub fn new(span: Span, kind: ElaborateErrorKind, message: String) -> Self {
        ElaborateError {
            span,
            kind,
            message,
        }
    }

    /// Convenience constructor for [`ElaborateErrorKind::NotImplemented`].
    #[must_use]
    pub fn not_implemented(span: Span, message: impl Into<String>) -> Self {
        ElaborateError::new(span, ElaborateErrorKind::NotImplemented, message.into())
    }

    /// Convenience constructor for [`ElaborateErrorKind::UnsupportedScrutinee`].
    #[must_use]
    pub fn unsupported_scrutinee(span: Span, message: impl Into<String>) -> Self {
        ElaborateError::new(
            span,
            ElaborateErrorKind::UnsupportedScrutinee,
            message.into(),
        )
    }
}

impl fmt::Display for ElaborateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

/// The machine-readable category of an [`ElaborateError`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ElaborateErrorKind {
    /// A name was not found in local or global scope.
    UnknownIdentifier,
    /// A prelude constant exists but is not supported by this delivery.
    UnsupportedConstant,
    /// An import path named a module that is not on the whitelist.
    UnknownModule,
    /// A parameter name shadows a constant in the global environment.
    ///
    /// Parameter shadowing of locals is permitted (inner wins), but a
    /// parameter whose name matches a global constant produces ill-scoped
    /// types in the return position. Until proper namespacing arrives, this
    /// is rejected at declaration time.
    ShadowedGlobal {
        /// The parameter name that shadowed the global.
        name: String,
        /// The name of the global that was shadowed.
        shadowed: String,
    },
    /// A surface feature is parsed but not elaborated.
    NotImplemented,
    /// A function body does not end in a `return` statement: either the
    /// final statement is a `let`, or a `return` appears in the middle.
    BodyNotReturning,
    /// A `match` arm is missing for some constructors, with no catch-all
    /// to cover them.
    NonExhaustiveMatch {
        /// The uncovered constructors, as dotted source names.
        missing: Vec<String>,
    },
    /// A `match` arm covers a constructor (or catch-all position) that
    /// an earlier arm already covers.
    RedundantArm {
        /// The offending pattern, as written.
        name: String,
    },
    /// A `match` arm names something that is not a constructor of the
    /// scrutinee's inductive.
    UnknownConstructor {
        /// The offending name, as written.
        name: String,
    },
    /// A `match` scrutinee's type is not a supported inductive.
    UnsupportedScrutinee,
    /// A structure's parameter has a type that is not a `Sort`.
    NonSortParameter,
    /// A field's type mentions the structure in a non-positive or
    /// non-direct-recursion position.
    NonPositiveField,
    /// A structure's name collides with an existing declaration.
    DuplicateStructure,
    /// Two fields of the same structure share a name.
    DuplicateField,
    /// A structure's recursion pattern is not supported in this delivery
    /// (nested or higher-order recursion).
    NestedRecursion,
    /// A structure declares no fields. (Caught at parse time; kept here
    /// for hand-built CST.)
    EmptyStructure,
    /// Field access on a value whose type the elaborator cannot infer.
    CannotInferFieldReceiver,
    /// Field access naming a field that does not exist on the receiver's
    /// structure, or on a non-structure value.
    UnknownField,
    /// The kernel rejected the elaborated declaration.
    Kernel(KernelError),
    /// A `Quantity(...)` argument failed to elaborate as a unit
    /// expression, or a canonical unit constant could not be registered.
    UnitError(theoria_units::UnitError),
    /// A `Theorem`'s `Show:` goal is not a proposition.
    ///
    /// The kernel accepts any `Sort`-inhabited type as the theorem's
    /// type, so a goal like `Nat` would otherwise register as a theorem
    /// whose type lives in `Sort 1`. The elaborator enforces the plan's
    /// requirement that a proof obligation is a proposition.
    TheoremGoalNotAProposition {
        /// The sort the goal lives in, if it is a type at all.
        ///
        /// `None` means the goal is not a type (its inference did not
        /// yield a `Sort`); `Some(l)` means the goal is a type at
        /// universe level `l`, which is not `Sort 0`.
        level: Option<theoria_kernel::Level>,
    },
    /// A `Proof:` block's step sequence is malformed in a way the
    /// parser cannot detect: `Exact` is missing, `Exact` is not last, or
    /// a step's binding name collides with one already in scope.
    InvalidProofStep {
        /// The 1-based step number the error refers to.
        step: u32,
    },
    /// A local proof binding (an `Assume:` hypothesis or a `Have:` step
    /// lemma) is not a proposition.
    ///
    /// Symmetric to `TheoremGoalNotAProposition`: a proof obligation
    /// must live in `Prop`.
    LocalBindingNotAProposition {
        /// The binding's name.
        name: theoria_kernel::NameId,
        /// The sort the binding's type lives in, if it is a type at all.
        level: Option<theoria_kernel::Level>,
    },
}
