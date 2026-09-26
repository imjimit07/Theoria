//! Cranelift lowering for the supported bytecode subset.
//!
//! Every JIT-compiled function has the C ABI
//! `extern "C" fn(frame: *mut u64, n: u32) -> u64`, where `frame` points
//! to an array of `n` nan-boxed values and the return value is packed.
//!
//! Lowered instructions:
//!
//! * `PushConst` of a `Nat` or `Star` — an `iconst` of the packed bits.
//! * `PushLocal` — a trusted 8-byte load from `frame[slot]`.
//! * `PushStar` — an `iconst` of the canonical `Star` bits.
//! * `Succ` — integer addition of `1 << 3` on the packed immediate.
//!   `Nat(n)` is `(n << 3) | 0`, so the successor is simply `v + 8`.
//!   Overflow wraps in this delivery; matching the interpreter's
//!   `IntegerOverflow` error is deferred with overflow-checked
//!   arithmetic.
//! * `Return` — returns the top of the virtual stack.
//!
//! Rejected with `UnsupportedInstruction`:
//!
//! * `MakeClosure` and `Apply` — heap-allocated closures need the bump
//!   allocator wired into codegen (delivery 25).
//! * `Recurse` — bridges back into the interpreter via
//!   `jit_call_interp` once program-ID stitching lands (delivery 25).
//!
//! The bytecode's value stack is modelled as a Rust `Vec` of SSA values
//! during lowering; each push/pop pair lines up structurally.

use crate::abi;
use crate::error::JitError;
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags, Signature, Value as ClValue, types};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::{FuncId, Linkage, Module};
use theoria_vm::Value;
use theoria_vm::bytecode::{Instruction, Program};

/// The Cranelift type used for every packed value.
const TY_U64: types::Type = types::I64;

/// Lower a whole [`Program`] into a Cranelift function.
///
/// The function has signature
/// `extern "C" fn(frame: *mut u64, n: u32) -> u64`. The caller supplies
/// the initial frame as a packed array; the JIT returns the result
/// packed.
///
/// # Errors
///
/// * [`JitError::UnsupportedInstruction`] if the program contains an
///   instruction outside the supported subset, or a `PushConst` of a
///   non-immediate value.
/// * [`JitError::Codegen`] for any Cranelift error.
/// * [`JitError::Internal`] if the value stack underflows or the
///   program does not end in `Return`.
pub fn lower_program(
    program: &Program,
    module: &mut dyn Module,
    func_ctx: &mut FunctionBuilderContext,
) -> Result<FuncId, JitError> {
    let mut sig = Signature::new(module.isa().default_call_conv());
    sig.params.push(AbiParam::new(types::I64));
    sig.params.push(AbiParam::new(types::I32));
    sig.returns.push(AbiParam::new(types::I64));

    let func_id = module
        .declare_function("theoria_jit_entry", Linkage::Local, &sig)
        .map_err(|e| JitError::Codegen(e.to_string()))?;
    let mut ctx = module.make_context();
    ctx.func.signature = sig;
    let mut builder = FunctionBuilder::new(&mut ctx.func, func_ctx);

    let entry_block = builder.create_block();
    builder.append_block_params_for_function_params(entry_block);
    builder.switch_to_block(entry_block);
    builder.seal_block(entry_block);

    let frame_ptr = builder.block_params(entry_block)[0];
    let _frame_len = builder.block_params(entry_block)[1];

    let mut vstack: Vec<ClValue> = Vec::new();
    let mut returned = false;
    for instr in &program.instructions {
        match instr {
            Instruction::PushConst(v) => {
                let bits = encode_const(v)?;
                let clv = builder.ins().iconst(TY_U64, bits as i64);
                vstack.push(clv);
            }
            Instruction::PushLocal(slot) => {
                let offset = i32::try_from(slot.checked_mul(8).ok_or_else(|| {
                    JitError::UnsupportedInstruction("frame slot out of range".into())
                })?)
                .map_err(|_| JitError::UnsupportedInstruction("frame slot out of range".into()))?;
                let clv = builder
                    .ins()
                    .load(TY_U64, MemFlags::trusted(), frame_ptr, offset);
                vstack.push(clv);
            }
            Instruction::PushStar => {
                let clv = builder.ins().iconst(TY_U64, abi::pack_star() as i64);
                vstack.push(clv);
            }
            Instruction::Succ(_) => {
                let v = vstack
                    .pop()
                    .ok_or_else(|| JitError::Internal("value-stack underflow at Succ".into()))?;
                // `Nat(n)` packs as `(n << 3)`; the successor adds one
                // unit of `1 << 3`.
                let next = builder.ins().iadd_imm(v, 8);
                vstack.push(next);
            }
            Instruction::MakeClosure { .. } => {
                return Err(JitError::UnsupportedInstruction(
                    "MakeClosure needs the bump allocator wired into codegen".into(),
                ));
            }
            Instruction::Apply => {
                return Err(JitError::UnsupportedInstruction(
                    "Apply needs closure dispatch (delivery 25)".into(),
                ));
            }
            Instruction::Recurse(entry) => {
                return Err(JitError::UnsupportedInstruction(alloc::format!(
                    "Recurse on {} bridges to the interpreter (delivery 25)",
                    entry.name
                )));
            }
            Instruction::Return => {
                let v = vstack
                    .pop()
                    .ok_or_else(|| JitError::Internal("value-stack underflow at Return".into()))?;
                builder.ins().return_(&[v]);
                returned = true;
                break;
            }
        }
    }

    if !returned {
        return Err(JitError::Internal("program does not end in Return".into()));
    }

    builder.seal_all_blocks();
    builder.finalize();

    module
        .define_function(func_id, &mut ctx)
        .map_err(|e| JitError::Codegen(e.to_string()))?;

    Ok(func_id)
}

/// Encode an immediately-packable constant.
fn encode_const(v: &Value) -> Result<u64, JitError> {
    match v {
        Value::Nat(n) => Ok(abi::pack_nat(*n)),
        Value::Star => Ok(abi::pack_star()),
        _ => Err(JitError::UnsupportedInstruction(
            "non-immediate PushConst".into(),
        )),
    }
}
