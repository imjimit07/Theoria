//! The JIT facade: compile a [`Program`] to native code and run it.

use crate::abi;
use crate::codegen::lower_program;
use crate::error::JitError;
use alloc::rc::Rc;
use alloc::vec::Vec;
use cranelift_jit::{JITBuilder, JITModule};
use theoria_vm::{Program, Value, VmEnv};

/// A JIT-compiled program.
///
/// Owns its Cranelift module; the entry point stays valid as long as
/// `self` lives.
pub struct Jit {
    module: JITModule,
    entry_id: cranelift_module::FuncId,
}

impl Jit {
    /// Compile a program to native code.
    ///
    /// The `env` parameter is currently unused: every program the JIT
    /// accepts is closed over immediates, so no environment lookup is
    /// needed. It is kept so that the recursor-bridging delivery can
    /// thread the environment through without changing the signature.
    ///
    /// # Errors
    ///
    /// * [`JitError::UnsupportedPlatform`] if the host has no Cranelift
    ///   backend.
    /// * [`JitError::Codegen`] for any Cranelift error.
    /// * [`JitError::UnsupportedInstruction`] if the program uses
    ///   `MakeClosure`, `Apply`, or `Recurse`.
    pub fn compile(program: Rc<Program>, _env: Rc<VmEnv<'_>>) -> Result<Self, JitError> {
        let isa_builder = cranelift_native::builder()
            .map_err(|e: &str| JitError::UnsupportedPlatform(e.into()))?;
        let flags = cranelift_codegen::settings::Flags::new(cranelift_codegen::settings::builder());
        let isa = isa_builder
            .finish(flags)
            .map_err(|e| JitError::Codegen(e.to_string()))?;

        let mut module = JITModule::new(JITBuilder::with_isa(
            isa,
            cranelift_module::default_libcall_names(),
        ));

        let mut func_ctx = cranelift_frontend::FunctionBuilderContext::new();
        let entry_id = lower_program(&program, &mut module, &mut func_ctx)?;

        module
            .finalize_definitions()
            .map_err(|e| JitError::Codegen(e.to_string()))?;

        Ok(Jit { module, entry_id })
    }

    /// Execute the compiled program with the given packed arguments.
    ///
    /// Only `Nat` and `Star` arguments are supported; anything else is
    /// rejected rather than miscompiled.
    ///
    /// # Errors
    ///
    /// * [`JitError::UnsupportedInstruction`] if an argument is not a
    ///   `Nat` or `Star`.
    pub fn run(&self, args: &[Value]) -> Result<Value, JitError> {
        let mut frame: Vec<u64> = Vec::with_capacity(args.len());
        for v in args {
            match v {
                Value::Nat(n) => frame.push(abi::pack_nat(*n)),
                Value::Star => frame.push(abi::pack_star()),
                _ => {
                    return Err(JitError::UnsupportedInstruction(
                        "JIT arguments must be Nat or Star".into(),
                    ));
                }
            }
        }

        let ptr = self.module.get_finalized_function(self.entry_id);
        // SAFETY: the JIT compiled the entry with exactly the signature
        // declared in `lower_program`: `extern "C" fn(*mut u64, u32) ->
        // u64`.
        let f: extern "C" fn(*mut u64, u32) -> u64 = unsafe { core::mem::transmute(ptr) };
        let bits = f(frame.as_mut_ptr(), frame.len() as u32);
        Ok(abi::unpack(bits))
    }
}
