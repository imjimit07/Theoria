//! The native-to-interpreter bridge.
//!
//! Two `extern "C"` symbols are exposed for JIT-compiled code:
//!
//! * `jit_alloc` — a bump allocator for closures and constructs.
//! * `jit_call_interp` — invokes the bytecode interpreter with a packed
//!   frame and returns a packed result.
//!
//! `jit_alloc` is fully implemented. `jit_call_interp` is a stub until
//! the program-ID stitching delivery lands: codegen currently rejects
//! every `Recurse`, so no compiled program can reach it.

use alloc::vec::Vec;
use core::cell::Cell;
use theoria_vm::Value;

thread_local! {
    /// The bump region as `(base pointer, remaining bytes)`. A null
    /// base means uninitialized.
    static HEAP: Cell<(*mut u8, usize)> = const { Cell::new((core::ptr::null_mut(), 0)) };
}

/// A bump allocation of `size` bytes, 8-byte aligned.
///
/// Returns a null pointer on exhaustion; JIT code checks for null and
/// traps. Fresh 64 KiB chunks come from the Rust allocator; growth by
/// doubling is a follow-up.
#[unsafe(no_mangle)]
pub extern "C" fn jit_alloc(size: usize) -> *mut u8 {
    let size = size.next_multiple_of(8);
    HEAP.with(|heap| {
        let (ptr, remaining) = heap.get();
        if !ptr.is_null() && remaining >= size {
            // SAFETY: `ptr` is the live bump cursor and `remaining`
            // tracks exactly the bytes after it.
            heap.set((unsafe { ptr.add(size) }, remaining - size));
            return ptr;
        }
        let chunk_size = 64 * 1024;
        let layout = core::alloc::Layout::from_size_align(chunk_size, 8).expect("valid layout");
        // SAFETY: the layout is non-zero and 8-byte aligned.
        let base = unsafe { alloc::alloc::alloc(layout) };
        if base.is_null() {
            return core::ptr::null_mut();
        }
        // SAFETY: `base` is fresh chunk memory; carving `size` bytes
        // off the front leaves a valid cursor.
        heap.set((unsafe { base.add(size) }, chunk_size - size));
        base
    })
}

/// Invoke the bytecode interpreter and return a packed result.
///
/// Wired when program-ID stitching lands: the `program_id` selects a
/// registered program, `entry` is the rule-body offset inside it, and
/// `frame`/`n` is the packed operand frame.
///
/// # Safety
///
/// `frame` must point to at least `n` initialized `u64` values, each a
/// valid nan-boxed value.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_call_interp(
    program_id: u32,
    entry: u32,
    frame: *const u64,
    n: u32,
) -> u64 {
    let _ = (program_id, entry, frame, n);
    unimplemented!("jit_call_interp bridges once Recurse lowering lands (delivery 25)")
}

/// Reconstruct interpreter [`Value`]s from a packed frame.
///
/// Used by the bridging delivery; kept here so the frame-walking logic
/// lives next to the allocator that produced the frame.
///
/// # Safety
///
/// `frame` must point to at least `n` initialized packed values.
pub unsafe fn unpack_frame(frame: *const u64, n: u32) -> Vec<Value> {
    let mut locals = Vec::with_capacity(n as usize);
    for i in 0..n as usize {
        // SAFETY: guaranteed by the caller.
        let bits = unsafe { *frame.add(i) };
        locals.push(crate::abi::unpack(bits));
    }
    locals
}
