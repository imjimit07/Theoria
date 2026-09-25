//! Interning canonical unit constants in the kernel environment.

use crate::error::UnitError;
use crate::exponents::Exponents;
use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::rc::Rc;
use alloc::string::ToString;
use alloc::vec::Vec;
use theoria_kernel::TerminationObligation;
use theoria_kernel::env::DefinitionVal;
use theoria_kernel::env::{ConstantInfo, ConstantVal, GlobalEnv};
use theoria_kernel::expr::{BinderInfo, Expr};
use theoria_kernel::level::Level;
use theoria_kernel::name::{NameId, NameTable};
use theoria_kernel::nbe::Transparency;

/// Tracks the kernel constants that back the unit system.
///
/// The registry is per-elaboration. It records which canonical unit
/// constants have been interned so that repeated `Quantity(m^2)` type
/// expressions produce the same `NameId` without re-querying the
/// environment.
pub struct UnitRegistry {
    /// Cache from exponent vector to the kernel constant's `NameId`.
    cache: BTreeMap<Exponents, NameId>,
    /// Reverse of `cache`: from a canonical constant's `NameId` back to
    /// its exponent vector. Consulted by the elaborator's type classifier
    /// to recover a unit from an elaborated `Quantity(u)` type.
    reverse: BTreeMap<NameId, Exponents>,
    /// The `NameId` of the `Unit` type former, if installed.
    unit: Option<NameId>,
    /// The `NameId` of the `Quantity` type former, if installed.
    quantity: Option<NameId>,
}

impl Default for UnitRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl UnitRegistry {
    /// A fresh, empty registry.
    #[must_use]
    pub fn new() -> Self {
        UnitRegistry {
            cache: BTreeMap::new(),
            reverse: BTreeMap::new(),
            unit: None,
            quantity: None,
        }
    }

    /// Install the kernel-side declarations the unit system needs.
    ///
    /// Registers `Unit : Sort 1` and `Quantity : Unit -> Type 0` as a
    /// definition `λ _ : Unit. Nat`. Idempotent: if the names are already
    /// present in the environment, the existing `NameId`s are remembered and
    /// nothing is inserted.
    ///
    /// ## Caller responsibility
    ///
    /// If the environment already contains a `Unit` or `Quantity`
    /// declaration with a different type, the registry silently uses the
    /// existing one. This can produce nonsense types (e.g. a user's
    /// `Quantity` that is not a `Unit -> Sort u` function) and is not
    /// detected. It is the caller's responsibility to start from a
    /// `GlobalEnv` that is either empty of these names or has them with
    /// the expected types.
    ///
    /// # Errors
    ///
    /// Returns [`UnitError::Registration`] if a declaration cannot be
    /// inserted (a pre-existing declaration is not an error; a
    /// different failure is).
    pub fn install(&mut self, env: &mut GlobalEnv, names: &mut NameTable) -> Result<(), UnitError> {
        if self.unit.is_none() {
            let id = names.intern("Unit");
            if env.find(id).is_none() {
                env.add(ConstantInfo::Axiom(ConstantVal {
                    name: id,
                    universe_params: Vec::new(),
                    ty: Rc::new(Expr::Sort(Level::Zero.succ())),
                }))
                .map_err(|e| UnitError::Registration(format!("`Unit`: {e}")))?;
            }
            self.unit = Some(id);
        }
        if self.quantity.is_none() {
            let id = names.intern("Quantity");
            if env.find(id).is_none() {
                let unit_id = self.unit.expect("Unit was installed just above");
                let nat_id = names.intern("Nat");
                if env.find(nat_id).is_none() {
                    return Err(UnitError::Registration(
                        "the unit system requires `Nat` in the environment; install `build_prelude` before `UnitRegistry::install`"
                            .to_string(),
                    ));
                }
                let underscore = names.intern("_");
                // def Quantity : Unit -> Type 0 := λ _ : Unit. Nat
                let ty = Expr::Pi(
                    BinderInfo::Default,
                    underscore,
                    Box::new(Expr::Const(unit_id, Vec::new())),
                    Box::new(Expr::Sort(Level::Zero.succ())),
                );
                let body = Expr::Lam(
                    BinderInfo::Default,
                    underscore,
                    Box::new(Expr::Const(unit_id, Vec::new())),
                    Box::new(Expr::Const(nat_id, Vec::new())),
                );
                env.add(ConstantInfo::Definition(DefinitionVal {
                    base: ConstantVal {
                        name: id,
                        universe_params: Vec::new(),
                        ty: Rc::new(ty),
                    },
                    body: Rc::new(body),
                    transparency: Transparency::Semireducible,
                    termination: TerminationObligation::none(),
                }))
                .map_err(|e| UnitError::Registration(format!("`Quantity`: {e}")))?;
            }
            self.quantity = Some(id);
        }
        Ok(())
    }

    /// The `NameId` of the `Unit` type former.
    ///
    /// # Panics
    ///
    /// Panics if [`UnitRegistry::install`] has not been called.
    #[must_use]
    pub fn unit(&self) -> NameId {
        self.unit.expect("UnitRegistry::unit called before install")
    }

    /// The `NameId` of the `Quantity` type former.
    ///
    /// # Panics
    ///
    /// Panics if [`UnitRegistry::install`] has not been called.
    #[must_use]
    pub fn quantity(&self) -> NameId {
        self.quantity
            .expect("UnitRegistry::quantity called before install")
    }

    /// Look up a canonical unit constant without creating one.
    #[must_use]
    pub fn lookup(&self, exp: &Exponents) -> Option<NameId> {
        self.cache.get(exp).copied()
    }

    /// Recover the exponent vector of a canonical unit constant.
    ///
    /// Returns `None` for any `NameId` the registry did not intern.
    #[must_use]
    pub fn exponents_of(&self, id: NameId) -> Option<Exponents> {
        self.reverse.get(&id).copied()
    }

    /// Ensure the canonical kernel constant for `exp` exists; return its
    /// `NameId`.
    ///
    /// If the constant is already cached, returns the cached `NameId`. If
    /// it exists in the environment but is not cached, interns it and
    /// caches. Otherwise, inserts a fresh axiom `U_<m>_... : Unit` and
    /// caches.
    ///
    /// # Errors
    ///
    /// * [`UnitError::Internal`] if `install` has not been called.
    /// * [`UnitError::Registration`] if insertion into the environment
    ///   fails.
    pub fn ensure(
        &mut self,
        exp: Exponents,
        env: &mut GlobalEnv,
        names: &mut NameTable,
    ) -> Result<NameId, UnitError> {
        if let Some(id) = self.cache.get(&exp) {
            return Ok(*id);
        }
        let unit = self.unit.ok_or_else(|| {
            UnitError::Internal("UnitRegistry::ensure called before install".to_string())
        })?;
        let name_str = exp.canonical_name();
        let id = names.intern(&name_str);
        if env.find(id).is_none() {
            env.add(ConstantInfo::Axiom(ConstantVal {
                name: id,
                universe_params: Vec::new(),
                ty: Rc::new(Expr::Const(unit, Vec::new())),
            }))
            .map_err(|e| UnitError::Registration(format!("`{name_str}`: {e}")))?;
        }
        self.cache.insert(exp, id);
        self.reverse.insert(id, exp);
        Ok(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use theoria_kernel::prelude::build_prelude;

    #[test]
    fn install_registers_unit_and_quantity() {
        let mut p = build_prelude();
        let mut reg = UnitRegistry::new();
        reg.install(&mut p.env, &mut p.names).unwrap();
        assert!(p.env.find(reg.unit()).is_some());
        assert!(p.env.find(reg.quantity()).is_some());
    }

    #[test]
    fn install_is_idempotent() {
        let mut p = build_prelude();
        let mut reg = UnitRegistry::new();
        reg.install(&mut p.env, &mut p.names).unwrap();
        let len_after_first = p.env.len();
        reg.install(&mut p.env, &mut p.names).unwrap();
        assert_eq!(p.env.len(), len_after_first);
    }

    #[test]
    fn install_across_registries_is_idempotent_on_env() {
        let mut p = build_prelude();
        UnitRegistry::new()
            .install(&mut p.env, &mut p.names)
            .unwrap();
        let len_after_first = p.env.len();
        // A second, independent registry reuses the same names.
        UnitRegistry::new()
            .install(&mut p.env, &mut p.names)
            .unwrap();
        assert_eq!(p.env.len(), len_after_first);
    }

    #[test]
    fn ensure_inserts_and_caches() {
        let mut p = build_prelude();
        let mut reg = UnitRegistry::new();
        reg.install(&mut p.env, &mut p.names).unwrap();

        assert!(reg.lookup(&Exponents::METRE).is_none());
        let id = reg
            .ensure(Exponents::METRE, &mut p.env, &mut p.names)
            .unwrap();
        assert_eq!(reg.lookup(&Exponents::METRE), Some(id));
        assert!(p.env.find(id).is_some());
        // Name matches the canonical form.
        assert_eq!(p.names.resolve(id), "U_1_0_0_0_0_0_0");
    }

    #[test]
    fn ensure_is_idempotent() {
        let mut p = build_prelude();
        let mut reg = UnitRegistry::new();
        reg.install(&mut p.env, &mut p.names).unwrap();
        let id1 = reg
            .ensure(Exponents::METRE, &mut p.env, &mut p.names)
            .unwrap();
        let len = p.env.len();
        let id2 = reg
            .ensure(Exponents::METRE, &mut p.env, &mut p.names)
            .unwrap();
        assert_eq!(id1, id2);
        assert_eq!(p.env.len(), len);
    }

    #[test]
    fn ensure_before_install_errors() {
        let mut p = build_prelude();
        let mut reg = UnitRegistry::new();
        let err = reg
            .ensure(Exponents::METRE, &mut p.env, &mut p.names)
            .unwrap_err();
        assert!(matches!(err, UnitError::Internal(_)));
    }

    #[test]
    fn ensure_compound_unit_has_canonical_name() {
        let mut p = build_prelude();
        let mut reg = UnitRegistry::new();
        reg.install(&mut p.env, &mut p.names).unwrap();
        let e = Exponents {
            m: 2,
            s: -1,
            ..Exponents::DIMENSIONLESS
        };
        let id = reg.ensure(e, &mut p.env, &mut p.names).unwrap();
        assert_eq!(p.names.resolve(id), "U_2_-1_0_0_0_0_0");
    }

    #[test]
    fn ensured_constant_has_type_unit() {
        let mut p = build_prelude();
        let mut reg = UnitRegistry::new();
        reg.install(&mut p.env, &mut p.names).unwrap();
        let id = reg
            .ensure(Exponents::KILOGRAM, &mut p.env, &mut p.names)
            .unwrap();
        let info = p.env.find(id).unwrap();
        let unit_id = reg.unit();
        match &**info.ty() {
            Expr::Const(name, args) => {
                assert_eq!(*name, unit_id);
                assert!(args.is_empty());
            }
            other => panic!("expected `Unit`, got {other:?}"),
        }
    }

    #[test]
    fn quantity_is_a_definition_of_nat() {
        // `Quantity(m)` must be defeq to `Nat`: this is what lets the
        // kernel accept `Nat.add r s` when `r, s : Quantity(m)`.
        let mut p = build_prelude();
        let mut reg = UnitRegistry::new();
        reg.install(&mut p.env, &mut p.names).unwrap();
        let quantity_id = reg.quantity();
        let info = p.env.find(quantity_id).unwrap();
        assert!(
            info.body().is_some(),
            "`Quantity` must be a definition, not an axiom"
        );
    }

    #[test]
    fn exponents_of_recovers_the_canonical_vector() {
        let mut p = build_prelude();
        let mut reg = UnitRegistry::new();
        reg.install(&mut p.env, &mut p.names).unwrap();
        let e = Exponents {
            m: 2,
            s: -1,
            ..Exponents::DIMENSIONLESS
        };
        let id = reg.ensure(e, &mut p.env, &mut p.names).unwrap();
        assert_eq!(reg.exponents_of(id), Some(e));
    }

    #[test]
    fn exponents_of_unknown_returns_none() {
        let reg = UnitRegistry::new();
        assert_eq!(reg.exponents_of(NameId(9999)), None);
    }
}
