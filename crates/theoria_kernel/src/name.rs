//! Name interning.
//!
//! Source-level names are interned into a [`NameId`]. The table is owned by the
//! kernel environment (or by the elaborator) so that the kernel itself remains
//! free of global mutable state, which would be incompatible with `no_std` and
//! with deterministic replay of proof checking.

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

/// A compact identifier for an interned name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NameId(pub u32);

impl fmt::Display for NameId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}", self.0)
    }
}

/// Interns strings into [`NameId`]s.
///
/// Names are never garbage-collected: the interner is expected to live for the
/// duration of an elaboration session, which matches how environments are used
/// in practice.
#[derive(Default)]
pub struct NameTable {
    names: Vec<String>,
    lookup: BTreeMap<String, NameId>,
}

impl NameTable {
    /// Create an empty table.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern a string, returning its stable identifier.
    pub fn intern(&mut self, s: &str) -> NameId {
        if let Some(id) = self.lookup.get(s) {
            return *id;
        }
        let id = NameId(u32::try_from(self.names.len()).expect("name table overflow"));
        let owned = s.to_string();
        self.names.push(owned.clone());
        self.lookup.insert(owned, id);
        id
    }

    /// Resolve a [`NameId`] back to its string.
    ///
    /// # Panics
    ///
    /// Panics if the identifier was not produced by this table. This is a
    /// programming error: identifiers must never cross table boundaries.
    #[must_use]
    pub fn resolve(&self, id: NameId) -> &str {
        &self.names[id.0 as usize]
    }

    /// Number of interned names.
    #[must_use]
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// `true` iff the table is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

impl fmt::Debug for NameTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NameTable")
            .field("len", &self.names.len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interning_is_stable() {
        let mut t = NameTable::new();
        let a = t.intern("Nat");
        let b = t.intern("Nat");
        let c = t.intern("Bool");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn resolve_roundtrip() {
        let mut t = NameTable::new();
        let id = t.intern("theoria");
        assert_eq!(t.resolve(id), "theoria");
    }

    #[test]
    fn distinct_names_are_distinct() {
        let mut t = NameTable::new();
        let ids: alloc::vec::Vec<_> = (0..100).map(|i| t.intern(&i.to_string())).collect();
        for (i, a) in ids.iter().enumerate() {
            for (j, b) in ids.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }
    }
}
