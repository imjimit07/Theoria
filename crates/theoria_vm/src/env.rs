//! The runtime environment: erased definitions and recursor metadata.

use alloc::collections::BTreeMap;
use alloc::rc::Rc;
use theoria_kernel::env::{ConstantInfo, GlobalEnv};
use theoria_kernel::erasure::LValue;
use theoria_kernel::name::{NameId, NameTable};

use crate::error::VmError;

/// The runtime environment consumed by the compiler and interpreter.
///
/// Built once from a [`GlobalEnv`]. Erased definitions and constructor
/// / recursor metadata are cached; the original environment is kept
/// behind an `Rc` so that lazy evaluation of `Const` unfolding can
/// consult it if needed.
pub struct VmEnv<'a> {
    /// The kernel environment.
    kernel: &'a GlobalEnv,
    /// The name table used to render constants in diagnostics.
    names: &'a NameTable,
    /// Erased definitions, keyed by name.
    erased: BTreeMap<NameId, Rc<LValue>>,
    /// Prelude identifiers the VM recognizes by structure.
    ///
    /// `nat_zero` and `nat_succ` let the interpreter classify a
    /// `Value::Nullary` / `Value::Unary` as the zero / successor
    /// constructors without comparing name strings at runtime. The
    /// `nat_rec` identifier lets the compiler recognize the recursor
    /// when it appears as a bare constant.
    nat_zero: NameId,
    nat_succ: NameId,
    nat_rec: NameId,
    /// Constructor arity, keyed by constructor `NameId`. Populated for
    /// every constructor in the environment at build time so that
    /// `Vm::apply` can decide when a `Construct` marker is fully
    /// saturated.
    ctor_arity: BTreeMap<NameId, u32>,
}

impl<'a> VmEnv<'a> {
    /// Build the environment by erasing every `Definition` in
    /// `kernel`.
    ///
    /// Theorems are erased to `Star` and are not stored: referencing a
    /// theorem at runtime is not a runtime operation.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`theoria_kernel::erasure::Eraser`] if any
    /// definition fails to erase. In practice this cannot happen for a
    /// well-formed environment, since the eraser only classifies types
    /// already checked by the kernel.
    pub fn build(
        kernel: &'a GlobalEnv,
        names: &'a mut NameTable,
    ) -> Result<Self, theoria_kernel::error::KernelError> {
        let nat_zero = names.intern("Nat.zero");
        let nat_succ = names.intern("Nat.succ");
        let nat_rec = names.intern("Nat.rec");

        let erasure = theoria_kernel::erasure::PropOnlyErasureEnv;
        let eraser = theoria_kernel::erasure::Eraser::new(kernel, &erasure);
        let mut erased = BTreeMap::new();
        let mut ctor_arity = BTreeMap::new();
        for (id, info) in kernel.iter() {
            match &**info {
                ConstantInfo::Definition(_) => {
                    if let Some(lv) = eraser.erase_declaration(info)? {
                        erased.insert(*id, Rc::new(lv));
                    }
                }
                ConstantInfo::Constructor(c) => {
                    ctor_arity.insert(*id, c.num_fields);
                }
                _ => {}
            }
        }

        Ok(VmEnv {
            kernel,
            names,
            erased,
            nat_zero,
            nat_succ,
            nat_rec,
            ctor_arity,
        })
    }

    /// The `Nat.zero` identifier.
    #[must_use]
    pub fn nat_zero(&self) -> NameId {
        self.nat_zero
    }

    /// The `Nat.succ` identifier.
    #[must_use]
    pub fn nat_succ(&self) -> NameId {
        self.nat_succ
    }

    /// The `Nat.rec` identifier.
    #[must_use]
    pub fn nat_rec(&self) -> NameId {
        self.nat_rec
    }

    /// The declared arity of a constructor, or `None` if the name is not
    /// a constructor in this environment.
    #[must_use]
    pub fn ctor_arity(&self, id: NameId) -> Option<u32> {
        self.ctor_arity.get(&id).copied()
    }

    /// The underlying kernel environment.
    #[must_use]
    pub fn kernel(&self) -> &'a GlobalEnv {
        self.kernel
    }

    /// The name table.
    #[must_use]
    pub fn names(&self) -> &'a NameTable {
        self.names
    }

    /// The erased body of a definition, if one has been cached.
    #[must_use]
    pub fn erased_body(&self, id: NameId) -> Option<Rc<LValue>> {
        self.erased.get(&id).cloned()
    }

    /// Look up a constant in the kernel environment.
    ///
    /// Returns an error for names not present. Callers decide whether
    /// the kind (Definition / Recursor / Constructor / Axiom / ...) is
    /// executable.
    pub fn lookup(&self, id: NameId) -> Result<Rc<ConstantInfo>, VmError> {
        self.kernel.find(id).ok_or(VmError::UnknownConstant(id))
    }

    /// `true` iff `id` is `Nat.zero`.
    #[must_use]
    pub fn is_zero(&self, id: NameId) -> bool {
        id == self.nat_zero
    }

    /// `true` iff `id` is `Nat.succ`.
    #[must_use]
    pub fn is_succ(&self, id: NameId) -> bool {
        id == self.nat_succ
    }

    /// `true` iff `id` is `Nat.rec`.
    #[must_use]
    pub fn is_nat_rec(&self, id: NameId) -> bool {
        id == self.nat_rec
    }

    /// The `NameId` for `Nat.zero`, if present (always `Some` after `build`).
    #[must_use]
    pub fn zero_name(&self) -> Option<NameId> {
        Some(self.nat_zero)
    }

    /// The `NameId` for `Nat.succ`, if present.
    #[must_use]
    pub fn succ_name(&self) -> Option<NameId> {
        Some(self.nat_succ)
    }
}
