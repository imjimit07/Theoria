//! Bytecode interpreter.

use alloc::format;
use alloc::rc::Rc;
use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;

use crate::bytecode::{Instruction, Program, RecursorEntry};
use crate::env::VmEnv;
use crate::error::VmError;
use crate::value::{ClosureBody, Value};

/// The bytecode interpreter.
pub struct Vm<'a> {
    env: &'a VmEnv<'a>,
    /// Maximum number of frame pushes. Prevents runaway recursion from
    /// hanging the host; erased programs are well-founded by
    /// construction, but user bugs can still produce surprising terms.
    max_depth: u32,
}

impl<'a> Vm<'a> {
    /// A new interpreter with a conservative recursion depth limit.
    #[must_use]
    pub fn new(env: &'a VmEnv<'a>) -> Self {
        Vm {
            env,
            max_depth: 1 << 16,
        }
    }

    /// Execute a compiled program with no arguments.
    ///
    /// # Errors
    ///
    /// Returns any [`VmError`] produced during execution.
    pub fn run(&self, program: &Rc<Program>) -> Result<Value, VmError> {
        self.run_from(program, 0, Vec::new())
    }

    /// Execute a program's entry point with a frame of locals.
    fn run_from(
        &self,
        program: &Rc<Program>,
        entry: u32,
        locals: Vec<Value>,
    ) -> Result<Value, VmError> {
        self.run_with_depth(program, entry, locals, 0)
    }

    /// Execute a program starting at an instruction index with the given
    /// locals.
    ///
    /// Exposed for the JIT bridge, which jumps into individual rule
    /// bodies; the CLI uses [`Vm::run`] instead.
    ///
    /// # Errors
    ///
    /// Returns any [`VmError`] produced during execution, as [`Vm::run`]
    /// does.
    pub fn run_entry(
        &self,
        program: &Rc<Program>,
        entry: u32,
        locals: Vec<Value>,
    ) -> Result<Value, VmError> {
        self.run_with_depth(program, entry, locals, 0)
    }

    fn run_with_depth(
        &self,
        program: &Rc<Program>,
        mut pc: u32,
        mut locals: Vec<Value>,
        depth: u32,
    ) -> Result<Value, VmError> {
        if depth > self.max_depth {
            return Err(VmError::Internal(
                "recursion depth limit exceeded".to_string(),
            ));
        }
        let mut stack: Vec<Value> = Vec::new();

        loop {
            let instr = program
                .instructions
                .get(pc as usize)
                .ok_or_else(|| VmError::Internal(format!("pc {pc} out of range")))?;

            match instr {
                Instruction::PushConst(v) => {
                    stack.push((**v).clone());
                    pc += 1;
                }
                Instruction::PushLocal(slot) => {
                    let v = locals
                        .get(*slot as usize)
                        .cloned()
                        .ok_or(VmError::UnboundVariable { index: *slot })?;
                    stack.push(v);
                    pc += 1;
                }
                Instruction::PushStar => {
                    stack.push(Value::Star);
                    pc += 1;
                }
                Instruction::MakeClosure { entry, arity } => {
                    let captured = locals.clone();
                    let prog_clone = Rc::clone(program);
                    stack.push(Value::Closure(Rc::new(ClosureBody {
                        program: prog_clone,
                        entry: *entry,
                        arity: *arity,
                        captured,
                    })));
                    pc += 1;
                }
                Instruction::Apply => {
                    let arg = stack
                        .pop()
                        .ok_or_else(|| VmError::Internal("stack underflow".to_string()))?;
                    let f = stack
                        .pop()
                        .ok_or_else(|| VmError::Internal("stack underflow".to_string()))?;
                    let result = self.apply(f, arg, depth)?;
                    stack.push(result);
                    pc += 1;
                }
                Instruction::Succ(name) => {
                    let v = stack
                        .pop()
                        .ok_or_else(|| VmError::Internal("stack underflow".to_string()))?;
                    let result = match v {
                        Value::Nat(n) => {
                            Value::Nat(n.checked_add(1).ok_or(VmError::IntegerOverflow)?)
                        }
                        other => Value::Unary(*name, Rc::new(other)),
                    };
                    stack.push(result);
                    pc += 1;
                }
                Instruction::Recurse(entry) => {
                    let result = self.run_recursor(program, &mut stack, entry, depth)?;
                    stack.push(result);
                    pc += 1;
                }
                Instruction::Return => {
                    let v = stack
                        .pop()
                        .ok_or_else(|| VmError::Internal("stack underflow".to_string()))?;
                    return Ok(v);
                }
            }
            let _ = &mut locals;
        }
    }

    /// Apply a value to an argument.
    fn apply(&self, f: Value, arg: Value, depth: u32) -> Result<Value, VmError> {
        match f {
            Value::Closure(c) => {
                let mut new_locals = c.captured.clone();
                new_locals.push(arg);
                self.run_with_depth(&c.program, c.entry, new_locals, depth + 1)
            }
            Value::Stuck(name, mut args) => {
                // Stuck application: collect args
                args.push(arg);
                Ok(Value::Stuck(name, args))
            }
            _ => Err(VmError::NotAFunction),
        }
    }

    /// Execute a recursor with the arguments on the stack.
    ///
    /// The compiler emits the arguments in order: parameters, motives,
    /// minors, indices, then the major premise. The stack has them in
    /// the same order with the major premise on top.
    fn run_recursor(
        &self,
        program: &Rc<Program>,
        stack: &mut Vec<Value>,
        entry: &RecursorEntry,
        depth: u32,
    ) -> Result<Value, VmError> {
        let total_args =
            (entry.num_params + entry.num_motives + entry.num_minors + entry.num_indices + 1)
                as usize;
        if stack.len() < total_args {
            return Err(VmError::Internal("stack underflow in recursor".to_string()));
        }
        let major_idx = stack.len() - 1;
        let major = stack[major_idx].clone();

        // Determine the constructor and fields for the major premise.
        // `Nat` is special-cased as `Nat(u64)`.
        let (ctor_name, fields) =
            match &major {
                Value::Nat(0) => {
                    let zero = self.env.zero_name().ok_or_else(|| {
                        VmError::Internal("Nat.zero not found in VmEnv".to_string())
                    })?;
                    (zero, Vec::new())
                }
                Value::Nat(n) => {
                    let succ = self.env.succ_name().ok_or_else(|| {
                        VmError::Internal("Nat.succ not found in VmEnv".to_string())
                    })?;
                    // `Nat(n)` where n>0 is `Nat.succ` applied to `Nat(n-1)`
                    (succ, vec![Value::Nat(n - 1)])
                }
                Value::Nullary(name) => (*name, Vec::new()),
                Value::Unary(name, v) => (*name, vec![(**v).clone()]),
                Value::Construct(name, vs) => (*name, vs.clone()),
                _ => {
                    return Err(VmError::BadMajorPremise {
                        recursor: entry.name,
                    });
                }
            };

        // Find the rule.
        let rule = entry
            .rules
            .iter()
            .find(|r| r.constructor == ctor_name)
            .ok_or(VmError::BadMajorPremise {
                recursor: entry.name,
            })?;

        // Build the RHS frame: parameters, motives, minors, indices are
        // below the major premise on the stack.
        let mut locals: Vec<Value> = stack[..major_idx].to_vec();
        let is_nat_succ = self.env.is_succ(ctor_name) && rule.num_fields == 1;
        if is_nat_succ {
            // Nat.succ case: fields = [Nat(k)], need to provide `n` and `ih`
            // where `ih` is `Nat.rec ... Nat(k)`.
            let pred = fields[0].clone();
            locals.push(pred.clone());
            let ih = {
                let mut rec_stack: Vec<Value> = stack[..major_idx].to_vec();
                rec_stack.push(pred);
                let mut tmp_stack = rec_stack;
                self.run_recursor(program, &mut tmp_stack, entry, depth + 1)?
            };
            locals.push(ih);
        } else {
            for f in &fields {
                locals.push(f.clone());
            }
        }

        // Remove the consumed arguments from the stack.
        stack.truncate(stack.len() - total_args);

        // Jump into the RHS.
        self.run_with_depth(program, rule.rhs_entry, locals, depth + 1)
    }
}
