//! Local scopes for elaboration.
//!
//! A [`Scope`] is the elaborator's view of the binders currently in
//! scope: function parameters and `λ`-abstraction parameters. The top of
//! the stack is the **innermost** binder, matching the kernel's de Bruijn
//! convention (`Var(0)` = innermost).

use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;
use theoria_kernel::DeBruijnIndex;
use theoria_kernel::Expr as KernelExpr;
use theoria_units::Exponents;

/// A single binder in scope. The type is already elaborated to a kernel
/// [`Expr`](theoria_kernel::Expr), and the unit is the dimensional tag of
/// that type, if any.
#[derive(Clone, Debug)]
pub struct LocalBinding {
    /// The source-level name as written.
    pub source_name: String,
    /// The elaborated kernel type of the binder.
    pub ty: Rc<KernelExpr>,
    /// The unit of the binder's type, if the type is a `Quantity(u)`.
    ///
    /// `None` means the binder's type is not a quantity: `Nat`, `Bool`,
    /// a function type, a sort, etc. `Some(e)` means the type is
    /// `Quantity(u)` where `u`'s exponent vector is `e`.
    pub unit: Option<Exponents>,
}

impl LocalBinding {
    /// A binder whose type is not a quantity.
    #[must_use]
    pub fn bare(source_name: String, ty: Rc<KernelExpr>) -> Self {
        LocalBinding {
            source_name,
            ty,
            unit: None,
        }
    }

    /// A binder whose type is `Quantity(u)`.
    #[must_use]
    pub fn with_unit(source_name: String, ty: Rc<KernelExpr>, unit: Exponents) -> Self {
        LocalBinding {
            source_name,
            ty,
            unit: Some(unit),
        }
    }
}

/// The local scope during elaboration of a function or lambda.
///
/// The top of the stack (the end of the internal vector) is the
/// innermost binder.
#[derive(Clone, Debug, Default)]
pub struct Scope {
    bindings: Vec<LocalBinding>,
}

impl Scope {
    /// An empty scope.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a binder onto the innermost position.
    pub fn push(&mut self, b: LocalBinding) {
        self.bindings.push(b);
    }

    /// Pop the innermost binder.
    ///
    /// # Panics
    ///
    /// Panics if the scope is empty. This is a programming error:
    /// pushes and pops are balanced by construction.
    #[must_use]
    pub fn pop(&mut self) -> LocalBinding {
        self.bindings.pop().expect("Scope::pop on empty scope")
    }

    /// Number of binders.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    /// `true` iff there are no binders.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    /// Find a name and return its de Bruijn index (distance from the top
    /// of the stack).
    ///
    /// Scans from the innermost binder outward and returns the first
    /// match: the innermost-wins rule of the surface language.
    #[must_use]
    pub fn lookup(&self, name: &str) -> Option<DeBruijnIndex> {
        self.bindings
            .iter()
            .rev()
            .position(|b| b.source_name == name)
            .map(|i| DeBruijnIndex(i as u32))
    }

    /// Look up the binding at a de Bruijn index (0 = innermost).
    ///
    /// Returns `None` if the index is out of range.
    #[must_use]
    pub fn lookup_index(&self, idx: DeBruijnIndex) -> Option<&LocalBinding> {
        let i = idx.0 as usize;
        self.bindings.iter().rev().nth(i)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binding(name: &str) -> LocalBinding {
        LocalBinding::bare(name.to_string(), Rc::new(KernelExpr::prop()))
    }

    #[test]
    fn lookup_innermost_wins() {
        let mut s = Scope::new();
        assert!(s.is_empty());
        s.push(binding("x"));
        s.push(binding("y"));
        s.push(binding("x"));
        assert_eq!(s.len(), 3);
        // Innermost `x` is at index 0, `y` at 1, outer `x` shadowed.
        assert_eq!(s.lookup("x"), Some(DeBruijnIndex(0)));
        assert_eq!(s.lookup("y"), Some(DeBruijnIndex(1)));
        assert_eq!(s.lookup("z"), None);
    }

    #[test]
    fn pop_returns_innermost() {
        let mut s = Scope::new();
        s.push(binding("a"));
        s.push(binding("b"));
        assert_eq!(s.pop().source_name, "b");
        assert_eq!(s.pop().source_name, "a");
        assert!(s.is_empty());
    }

    #[test]
    #[should_panic(expected = "empty scope")]
    fn pop_empty_panics() {
        let mut s = Scope::new();
        let _ = s.pop();
    }
}
