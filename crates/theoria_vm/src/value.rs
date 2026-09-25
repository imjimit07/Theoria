//! Runtime values.

use alloc::rc::Rc;
use alloc::vec::Vec;
use core::fmt;
use theoria_kernel::name::NameId;

use crate::bytecode::Program;

/// A runtime value.
///
/// `Nat` and `Bool` are recognized directly rather than as generic
/// constructor applications because they are the only inductives the
/// current kernel fragment produces. Other single-constructor
/// inductives fall into the [`Value::Construct`] variant.
#[derive(Clone)]
pub enum Value {
    /// The erased placeholder. Introduced by the erasure pass for
    /// proofs and types; propagated through applications but never
    /// unfolded.
    Star,
    /// A natural number.
    Nat(u64),
    /// A unary constructor applied to a value.
    ///
    /// `Nat.succ` produces this variant when its argument is not
    /// representable as a `u64` — which cannot happen in the current
    /// fragment, but the variant exists for uniformity with user
    /// structures.
    Unary(NameId, Rc<Value>),
    /// A nullary constructor.
    Nullary(NameId),
    /// A multi-field constructor. Field order matches the constructor's
    /// declaration order; erased fields (types, proofs) appear as
    /// [`Value::Star`].
    Construct(NameId, Vec<Value>),
    /// A closure captured by the compiler.
    Closure(Rc<ClosureBody>),
    /// A stuck term: an application whose head is not reducible. Kept
    /// for symmetry with the kernel's neutrals; the VM's fragment
    /// rarely produces these.
    Stuck(NameId, Vec<Value>),
}

/// The compiled body of a lambda.
#[derive(Clone)]
pub struct ClosureBody {
    /// The compiled program the closure's code belongs to.
    pub program: Rc<Program>,
    /// Byte offset into `program.instructions`.
    pub entry: u32,
    /// The number of formal parameters.
    pub arity: u32,
    /// The captured environment. Index `0` is the innermost capture,
    /// matching the runtime frame layout.
    pub captured: Vec<Value>,
}

impl Value {
    /// `true` iff the value is the erased placeholder.
    #[must_use]
    pub const fn is_star(&self) -> bool {
        matches!(self, Value::Star)
    }

    /// Extract the `u64` from a `Nat`, if this is one.
    #[must_use]
    pub const fn as_nat(&self) -> Option<u64> {
        match self {
            Value::Nat(n) => Some(*n),
            _ => None,
        }
    }

    /// `true` iff this is a `Nat` or a constructor value.
    ///
    /// Used by the interpreter's `Apply` to distinguish a partially
    /// applied constructor (which should be extended) from an
    /// application that cannot proceed.
    #[must_use]
    pub fn is_constructor(&self) -> bool {
        matches!(
            self,
            Value::Nat(_) | Value::Unary(_, _) | Value::Nullary(_) | Value::Construct(_, _)
        )
    }

    /// If this is a partially applied constructor, extend it with `arg`
    /// and return `Some(extended)`. Otherwise return `None`.
    ///
    /// The interpretation is deliberately name-agnostic: the caller
    /// checks arity against the environment's `ctor_arity` after the
    /// extension.
    #[must_use]
    pub fn extend_construct(self, arg: Value) -> Option<Value> {
        match self {
            Value::Construct(name, mut fields) => {
                fields.push(arg);
                Some(Value::Construct(name, fields))
            }
            Value::Nullary(name) => Some(Value::Unary(name, Rc::new(arg))),
            _ => None,
        }
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Star => f.write_str("⋆"),
            Value::Nat(n) => write!(f, "{n}"),
            Value::Unary(name, v) => write!(f, "({name} {v:?})"),
            Value::Nullary(name) => write!(f, "{name}"),
            Value::Construct(name, vs) => {
                write!(f, "({name}")?;
                for v in vs {
                    write!(f, " {v:?}")?;
                }
                f.write_str(")")
            }
            Value::Closure(c) => write!(f, "<closure arity={}>", c.arity),
            Value::Stuck(name, vs) => {
                write!(f, "(stuck {name}")?;
                for v in vs {
                    write!(f, " {v:?}")?;
                }
                f.write_str(")")
            }
        }
    }
}
