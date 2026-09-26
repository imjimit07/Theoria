//! Nan-boxed value representation shared between JIT code and the bridge.
//!
//! The JIT operates on raw `u64` values. Immediates (`Nat`, `Star`)
//! never touch the heap; pointer-tagged values (closures, multi-field
//! constructs) live in the bump allocator in the `bridge` module. When
//! JIT code calls back into the interpreter, operands cross the
//! boundary through [`pack`] and [`unpack`]; this module is the sole
//! source of truth for the layout.

use alloc::vec::Vec;
use theoria_kernel::name::NameId;
use theoria_vm::Value;

/// Tag bits for immediate values.
const TAG_MASK: u64 = 0b111;
/// Tag for a small `Nat` immediate.
const TAG_NAT: u64 = 0b000;
/// Tag for a pointer to a heap object.
const TAG_PTR: u64 = 0b001;
/// Tag for the erased `Star` placeholder.
const TAG_STAR: u64 = 0b010;

/// The largest `Nat` representable as an immediate.
pub const NAT_MAX: u64 = (1 << 61) - 1;

/// Pack a `Nat` value into an immediate.
///
/// # Panics
///
/// Panics if `n > NAT_MAX`. The JIT's arithmetic stays within small
/// literals; larger values remain interpreter-only until heap-boxed
/// `Nat` support lands.
#[must_use]
pub fn pack_nat(n: u64) -> u64 {
    assert!(
        n <= NAT_MAX,
        "Nat value exceeds 2^61 - 1; heap boxing not yet supported"
    );
    (n << 3) | TAG_NAT
}

/// Pack a pointer to a heap object.
///
/// The pointer must be 8-byte aligned; the bump allocator guarantees
/// this. Only the low 61 bits of the address are stored, so this
/// scheme assumes a 64-bit host with 8-byte-aligned allocations.
#[must_use]
pub fn pack_ptr(ptr: *const u8) -> u64 {
    let p = ptr as u64;
    debug_assert!(p & TAG_MASK == 0, "pointer is not 8-byte aligned");
    (p << 3) | TAG_PTR
}

/// The canonical `Star` immediate.
#[must_use]
pub const fn pack_star() -> u64 {
    TAG_STAR
}

/// The tag of a packed value.
#[must_use]
pub const fn tag_of(bits: u64) -> u64 {
    bits & TAG_MASK
}

/// `true` iff the packed value is a `Nat` immediate.
#[must_use]
pub const fn is_nat(bits: u64) -> bool {
    tag_of(bits) == TAG_NAT
}

/// `true` iff the packed value is a heap pointer.
#[must_use]
pub const fn is_ptr(bits: u64) -> bool {
    tag_of(bits) == TAG_PTR
}

/// `true` iff the packed value is `Star`.
#[must_use]
pub const fn is_star(bits: u64) -> bool {
    tag_of(bits) == TAG_STAR
}

/// Unpack a `Nat` immediate.
///
/// # Panics
///
/// Panics if the value is not a `Nat` immediate.
#[must_use]
pub const fn unpack_nat(bits: u64) -> u64 {
    assert!(is_nat(bits), "expected a Nat immediate");
    bits >> 3
}

/// Unpack a heap pointer.
///
/// # Panics
///
/// Panics if the value is not a pointer.
#[must_use]
pub fn unpack_ptr(bits: u64) -> *const u8 {
    assert!(is_ptr(bits), "expected a pointer");
    ((bits >> 3) << 3) as *const u8
}

/// The C-ABI heap layout of a construct value.
///
/// Stored as `[tag: u32, arity: u32, fields: u64[arity]]`, where each
/// field is itself a packed value.
#[repr(C)]
pub struct ConstructHeader {
    /// The kernel `NameId` of the constructor.
    pub tag: u32,
    /// Number of fields.
    pub arity: u32,
}

/// The C-ABI heap layout of a closure.
///
/// Stored as `[entry: u32, arity: u32, program_id: u32, pad: u32,
/// captures: u64[n]]`.
#[repr(C)]
pub struct ClosureHeader {
    /// The instruction index of the closure's body in the owning program.
    pub entry: u32,
    /// The closure's arity.
    pub arity: u32,
    /// The owning program's identifier in the JIT registry.
    pub program_id: u32,
    /// Padding so that `captures` is 8-byte aligned.
    pub _pad: u32,
}

/// Convert a [`Value`] into its packed representation.
///
/// Only immediates (`Nat`, `Star`) are supported: pointer-tagged values
/// are allocated by the JIT's bump allocator, and a Rust-side pointer
/// would not be compatible with it. Used by the JIT entry point when
/// packing call arguments.
///
/// # Panics
///
/// Panics for values the JIT does not support (closures, constructs,
/// stuck terms).
#[must_use]
pub fn pack(value: &Value) -> u64 {
    match value {
        Value::Star => pack_star(),
        Value::Nat(n) => pack_nat(*n),
        other => {
            let _ = other;
            unimplemented!("packing a pointer value from the Rust side is not supported")
        }
    }
}

/// Convert a packed representation into a [`Value`].
///
/// Pointer-tagged values are reconstructed from the heap layout written
/// by the JIT's bump allocator. In this delivery only immediates cross
/// the boundary back into Rust (the JIT rejects every program that
/// could produce a pointer), so the pointer branch exists for the
/// recursor-bridging delivery that follows.
///
/// # Safety
///
/// If `bits` is pointer-tagged, it must point to a valid heap object
/// with a well-formed header, as produced by the JIT's allocator.
#[must_use]
pub fn unpack(bits: u64) -> Value {
    if is_nat(bits) {
        Value::Nat(unpack_nat(bits))
    } else if is_star(bits) {
        Value::Star
    } else if is_ptr(bits) {
        // Reconstruct a multi-field construct from the C-ABI heap
        // layout. Closures never cross the boundary in this direction:
        // the major premise of a recursor is always a construct.
        let ptr = unpack_ptr(bits);
        let header = ptr as *const ConstructHeader;
        // SAFETY: the caller guarantees the pointer came from the JIT's
        // allocator with a valid header.
        let tag = unsafe { (*header).tag };
        let arity = unsafe { (*header).arity } as usize;
        let fields_ptr = unsafe { ptr.add(core::mem::size_of::<ConstructHeader>()) as *const u64 };
        let mut fields = Vec::with_capacity(arity);
        for i in 0..arity {
            // SAFETY: same guarantee as above; the header promises
            // `arity` packed fields follow it.
            let field_bits = unsafe { *fields_ptr.add(i) };
            fields.push(unpack(field_bits));
        }
        Value::Construct(NameId(tag), fields)
    } else {
        panic!("invalid nan-boxed value: {bits:#x}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nat_roundtrip() {
        for n in [0u64, 1, 42, 1 << 20, NAT_MAX] {
            let bits = pack_nat(n);
            assert!(is_nat(bits));
            assert_eq!(unpack_nat(bits), n);
        }
    }

    #[test]
    fn star_is_tagged() {
        assert!(is_star(pack_star()));
        assert!(!is_nat(pack_star()));
        assert!(!is_ptr(pack_star()));
    }

    #[test]
    #[should_panic(expected = "Nat value exceeds")]
    fn nat_overflow_panics() {
        let _ = pack_nat(NAT_MAX + 1);
    }

    #[test]
    fn tag_extraction() {
        assert_eq!(tag_of(pack_nat(0)), TAG_NAT);
        assert_eq!(tag_of(pack_star()), TAG_STAR);
    }
}
