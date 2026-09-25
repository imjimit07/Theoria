//! Default universe levels for supported constants.
//!
//! Delivery 2 has no universe inference. Every `Const(name, levels)` must
//! carry a `Vec<Level>` whose length equals the constant's declared
//! universe parameter count. This module implements the temporary rule:
//!
//! | Universe parameter count | Default levels |
//! |---|---|
//! | 0 | `[]` |
//! | 1 | `[Level::Zero.succ()]` — a `Type 0`-sized universe |
//! | ≥ 2 | rejected (`None`) — requires universe inference |
//!
//! The 1-parameter default admits every single-universe prelude constant
//! (`Nat.rec`, `Bool.rec`, `Eq`, `Eq.refl`, `List`, `Option`, `Quot`, …)
//! at `Type 0`. A Prop-valued instantiation (e.g. a `Nat.rec` motive into
//! `Prop`) would need `u = 0` and is rejected; it waits on inference.
//! Constants with two universe parameters (`Eq.rec`, `List.rec`,
//! `Option.rec`, `Quot.lift`) are rejected for the same reason.
//!
//! // TODO(universe-inference): replace with metavariable-based
//! // elaboration that unifies levels during checking.

use alloc::vec::Vec;
use theoria_kernel::{ConstantInfo, Level, NameId, NameTable};

/// Default universe levels for a resolved constant.
///
/// Returns `Some(levels)` with `levels.len()` equal to the constant's
/// parameter count, per the table above, and `None` for constants with
/// two or more universe parameters.
#[must_use]
pub fn default_levels(
    _name: NameId,
    info: &ConstantInfo,
    _names: &NameTable,
) -> Option<Vec<Level>> {
    match info.universe_params().len() {
        0 => Some(Vec::new()),
        1 => Some(alloc::vec![Level::Zero.succ()]),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::rc::Rc;
    use theoria_kernel::Expr as KernelExpr;
    use theoria_kernel::{ConstantVal, GlobalEnv};

    fn axiom(env: &mut GlobalEnv, names: &mut NameTable, text: &str, params: usize) -> NameId {
        let id = names.intern(text);
        let ups: Vec<theoria_kernel::UniverseParamId> = (0..params as u32)
            .map(theoria_kernel::UniverseParamId)
            .collect();
        env.add(ConstantInfo::Axiom(ConstantVal {
            name: id,
            universe_params: ups,
            ty: Rc::new(KernelExpr::prop()),
        }))
        .unwrap();
        id
    }

    #[test]
    fn monomorphic_constants_default_to_empty() {
        let mut env = GlobalEnv::new();
        let mut names = NameTable::new();
        let id = axiom(&mut env, &mut names, "Nat", 0);
        let info = env.find(id).unwrap();
        assert_eq!(default_levels(id, &info, &names), Some(Vec::new()));
    }

    #[test]
    fn nat_rec_defaults_to_level_one() {
        let mut env = GlobalEnv::new();
        let mut names = NameTable::new();
        let id = axiom(&mut env, &mut names, "Nat.rec", 1);
        let info = env.find(id).unwrap();
        assert_eq!(
            default_levels(id, &info, &names),
            Some(alloc::vec![Level::Zero.succ()])
        );
    }

    #[test]
    fn polymorphic_unknown_constants_are_unsupported() {
        let mut env = GlobalEnv::new();
        let mut names = NameTable::new();
        let id = axiom(&mut env, &mut names, "List.rec", 2);
        let info = env.find(id).unwrap();
        assert_eq!(default_levels(id, &info, &names), None);
    }

    #[test]
    fn single_param_unknown_constants_default_to_level_one() {
        let mut env = GlobalEnv::new();
        let mut names = NameTable::new();
        let id = axiom(&mut env, &mut names, "Foo.rec", 1);
        let info = env.find(id).unwrap();
        assert_eq!(
            default_levels(id, &info, &names),
            Some(alloc::vec![Level::Zero.succ()])
        );
    }
}
