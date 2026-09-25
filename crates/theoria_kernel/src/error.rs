//! Kernel error types.
//!
//! Errors are deliberately source-independent: the kernel has no access to
//! spans, file paths, or the surface syntax. The elaborator is responsible for
//! re-attaching source locations when it propagates these errors to the user.

use crate::expr::DeBruijnIndex;
use crate::level::UniverseParamId;
use crate::name::NameId;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

/// Result alias for kernel operations.
pub type KernelResult<T> = Result<T, KernelError>;

/// All errors the kernel can produce.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KernelError {
    /// Universe constraints formed a cycle `u < u`.
    UniverseInconsistency {
        /// The parameters participating in the cycle.
        cycle: Vec<UniverseParamId>,
    },

    /// A universe constraint could not be discharged.
    UnsolvedUniverseConstraint {
        /// Human-readable rendering of the constraint.
        message: String,
    },

    /// A constant was referenced but not found in the environment.
    UnknownConstant(NameId),

    /// A name was referenced that is not present in the environment.
    UnknownName(NameId),

    /// A bound variable referenced an index outside the local context.
    UnboundVariable {
        /// The offending index.
        index: DeBruijnIndex,
        /// The size of the context at the point of failure.
        context_size: usize,
    },

    /// Two types were expected to be definitionally equal but were not.
    TypeMismatch {
        /// Pretty-printed expected type.
        expected: String,
        /// Pretty-printed actual type.
        actual: String,
    },

    /// Application of a non-function.
    NotAFunction {
        /// Pretty-printed type of the applied term.
        found: String,
    },

    /// A `Prop`-valued elimination was attempted on a `Type`-valued inductive.
    ///
    /// This is the *singleton elimination* restriction that keeps `Prop`
    /// impredicative and proof-irrelevant.
    InvalidElimination {
        /// The inductive name.
        inductive: NameId,
    },

    /// An inductive declaration failed the positivity check.
    NonPositiveInductive {
        /// The offending inductive.
        inductive: NameId,
        /// The index of the constructor that failed.
        constructor: usize,
    },

    /// A recursive definition failed the termination check.
    NonTerminating {
        /// The offending function.
        function: NameId,
    },

    /// A proof term was rejected by the kernel.
    ProofRejected {
        /// Human-readable reason.
        reason: String,
    },

    /// A constant was applied to the wrong number of universe arguments.
    UniverseArityMismatch {
        /// The constant's name.
        name: NameId,
        /// Number of universe parameters expected.
        expected: usize,
        /// Number provided.
        actual: usize,
    },

    /// A term could not be inferred and requires an explicit annotation.
    CannotInfer {
        /// Why the term cannot be inferred.
        reason: String,
    },

    /// An expression that was expected to be a type turned out not to be a
    /// sort.
    NotASort {
        /// Pretty-printed representation of the offending expression.
        found: String,
    },

    /// An internal invariant was violated. This should never occur; if it does,
    /// it indicates a kernel bug and should be reported.
    Internal(String),
}

impl fmt::Display for KernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KernelError::UniverseInconsistency { cycle } => {
                write!(f, "universe inconsistency: cycle")?;
                for p in cycle {
                    write!(f, " → {p}")?;
                }
                Ok(())
            }
            KernelError::UnsolvedUniverseConstraint { message } => {
                write!(f, "unsolved universe constraint: {message}")
            }
            KernelError::UnknownConstant(id) => write!(f, "unknown constant {id}"),
            KernelError::UnknownName(id) => write!(f, "unknown name {id}"),
            KernelError::UnboundVariable {
                index,
                context_size,
            } => write!(
                f,
                "unbound variable {index} (context has {context_size} binders)"
            ),
            KernelError::TypeMismatch { expected, actual } => write!(
                f,
                "type mismatch:\n  expected: {expected}\n  actual:   {actual}"
            ),
            KernelError::NotAFunction { found } => {
                write!(f, "expected a function, found: {found}")
            }
            KernelError::InvalidElimination { inductive } => write!(
                f,
                "cannot eliminate inductive {inductive} into a non-Prop sort"
            ),
            KernelError::NonPositiveInductive {
                inductive,
                constructor,
            } => write!(
                f,
                "inductive {inductive} fails positivity at constructor #{constructor}"
            ),
            KernelError::NonTerminating { function } => {
                write!(
                    f,
                    "recursive function {function} fails the termination check"
                )
            }
            KernelError::ProofRejected { reason } => write!(f, "proof rejected: {reason}"),
            KernelError::UniverseArityMismatch {
                name,
                expected,
                actual,
            } => write!(
                f,
                "constant {name} expects {expected} universe argument(s), got {actual}"
            ),
            KernelError::CannotInfer { reason } => write!(f, "cannot infer type: {reason}"),
            KernelError::NotASort { found } => write!(f, "expected a sort, found: {found}"),
            KernelError::Internal(msg) => write!(f, "internal kernel error: {msg}"),
        }
    }
}

// In no_std builds we cannot implement `std::error::Error`; gate it.
#[cfg(feature = "std")]
impl std::error::Error for KernelError {}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    #[test]
    fn display_is_nonempty_for_every_variant() {
        let variants = [
            KernelError::UniverseInconsistency {
                cycle: alloc::vec![],
            },
            KernelError::UnsolvedUniverseConstraint {
                message: "u ≤ v".to_string(),
            },
            KernelError::UnknownConstant(crate::NameId(0)),
            KernelError::UnknownName(crate::NameId(0)),
            KernelError::UnboundVariable {
                index: DeBruijnIndex(3),
                context_size: 2,
            },
            KernelError::TypeMismatch {
                expected: "Nat".into(),
                actual: "Bool".into(),
            },
            KernelError::NotAFunction {
                found: "Nat".into(),
            },
            KernelError::InvalidElimination {
                inductive: crate::NameId(0),
            },
            KernelError::NonPositiveInductive {
                inductive: crate::NameId(0),
                constructor: 0,
            },
            KernelError::NonTerminating {
                function: crate::NameId(0),
            },
            KernelError::ProofRejected {
                reason: "bad".into(),
            },
            KernelError::UniverseArityMismatch {
                name: crate::NameId(0),
                expected: 1,
                actual: 2,
            },
            KernelError::CannotInfer {
                reason: "λ".into()
            },
            KernelError::NotASort { found: "x".into() },
            KernelError::Internal("boom".into()),
        ];
        for v in variants {
            let s = alloc::format!("{v}");
            assert!(!s.is_empty());
        }
    }
}
