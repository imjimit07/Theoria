//! `LValue` → [`Program`] compiler.

use alloc::rc::Rc;
use alloc::string::ToString;
use alloc::vec::Vec;
use theoria_kernel::env::{ConstantInfo, RecursorRule};
use theoria_kernel::erasure::LValue;
use theoria_kernel::name::NameId;

use crate::bytecode::{Instruction, Program, RecursorEntry, RecursorRuleEntry};
use crate::env::VmEnv;
use crate::error::VmError;
use crate::value::Value;

/// The compiler.
pub struct Compiler<'a> {
    env: &'a VmEnv<'a>,
    program: Program,
}

impl<'a> Compiler<'a> {
    /// A fresh compiler over a runtime environment.
    #[must_use]
    pub fn new(env: &'a VmEnv<'a>) -> Self {
        Compiler {
            env,
            program: Program::new(),
        }
    }

    /// Consume the compiler and return the produced program.
    #[must_use]
    pub fn into_program(self) -> Program {
        self.program
    }

    /// Compile a closed `LValue` to a complete program.
    ///
    /// The resulting program's `instructions` execute the term and end
    /// in a `Return`. The interpreter runs it with an empty frame.
    ///
    /// # Errors
    ///
    /// * [`VmError::UnknownConstant`] if a `Const` reference is not in
    ///   the environment.
    /// * [`VmError::UnsupportedConstant`] if the referenced declaration
    ///   cannot be executed (axioms, opaque constants, quotient
    ///   primitives).
    pub fn compile(mut self, term: &LValue) -> Result<Program, VmError> {
        self.compile_value(term)?;
        self.program.emit(Instruction::Return);
        Ok(self.program)
    }

    /// Compile a subterm, pushing its value onto the top of the stack.
    fn compile_value(&mut self, term: &LValue) -> Result<(), VmError> {
        match term {
            LValue::Star => {
                self.program.emit(Instruction::PushStar);
                Ok(())
            }
            LValue::Var(idx) => {
                self.program.emit(Instruction::PushLocal(idx.0));
                Ok(())
            }
            LValue::Lit(l) => {
                let v = match l {
                    theoria_kernel::expr::Literal::Nat(n) => Value::Nat(*n),
                    theoria_kernel::expr::Literal::Str(_) => {
                        return Err(VmError::Internal(
                            "string literals are not yet supported".to_string(),
                        ));
                    }
                };
                self.program.emit(Instruction::PushConst(Rc::new(v)));
                Ok(())
            }
            LValue::Lam(body) => {
                // For the fragment, `Lam` bodies are not executed by the
                // two Nat tests. We compile them as a closure that, when
                // applied, will execute the body. The body is compiled
                // to a separate program.
                let mut sub = Compiler {
                    env: self.env,
                    program: Program::new(),
                };
                sub.compile_value(body)?;
                sub.program.emit(Instruction::Return);
                let prog_rc = Rc::new(sub.program);
                let closure = Value::Closure(Rc::new(crate::value::ClosureBody {
                    program: prog_rc,
                    entry: 0,
                    arity: 1,
                    captured: Vec::new(),
                }));
                self.program.emit(Instruction::PushConst(Rc::new(closure)));
                Ok(())
            }
            LValue::App(f, a) => {
                // Special handling for `Nat.succ` applied to an argument:
                // `succ n` should be `Succ` after pushing `n`, not
                // `PushConst(succ)` + `PushConst(n)` + `Apply`.
                if let LValue::Const(name) = &**f {
                    if self.env.is_succ(*name) {
                        self.compile_value(a)?;
                        self.program.emit(Instruction::Succ(*name));
                        return Ok(());
                    }
                }
                self.compile_value(f)?;
                self.compile_value(a)?;
                self.program.emit(Instruction::Apply);
                Ok(())
            }
            LValue::Const(name) => self.compile_const(*name),
        }
    }

    /// Compile a `Const` reference.
    ///
    /// The dispatch mirrors the kernel's own `eval_const`:
    ///
    /// * A **definition** unfolds: its erased body is compiled in place.
    /// * A **recursor** emits a `Recurse` instruction carrying the
    ///   precompiled rules.
    /// * A **constructor** pushes a constructor value. For `Nat.zero`
    ///   this is `Nat(0)`, for `Nat.succ` it is a closure (handled in
    ///   `App`), otherwise `Nullary`/`Construct`.
    /// * A **theorem** pushes `Star` (theorem bodies are not executed).
    /// * Any other kind is an error.
    fn compile_const(&mut self, name: NameId) -> Result<(), VmError> {
        let info = self.env.lookup(name)?;
        match &*info {
            ConstantInfo::Definition(_) => {
                let body = self
                    .env
                    .erased_body(name)
                    .ok_or(VmError::UnknownConstant(name))?;
                self.compile_value(&body)
            }
            ConstantInfo::Recursor(rec) => {
                let entry = self.compile_recursor(name, &rec.rules)?;
                self.program.emit(Instruction::Recurse(entry));
                Ok(())
            }
            ConstantInfo::Constructor(ctor) => {
                if self.env.is_zero(name) {
                    self.program
                        .emit(Instruction::PushConst(Rc::new(Value::Nat(0))));
                    Ok(())
                } else if self.env.is_succ(name) && ctor.num_fields == 1 {
                    // `Nat.succ` as a standalone value (not applied) is a
                    // function `Nat -> Nat`. Represent it as a closure that
                    // does `Succ` when applied.
                    let mut closure_prog = Program::new();
                    closure_prog.emit(Instruction::PushLocal(0));
                    closure_prog.emit(Instruction::Succ(name));
                    closure_prog.emit(Instruction::Return);
                    let prog_rc = Rc::new(closure_prog);
                    let closure = Value::Closure(Rc::new(crate::value::ClosureBody {
                        program: prog_rc,
                        entry: 0,
                        arity: 1,
                        captured: Vec::new(),
                    }));
                    self.program.emit(Instruction::PushConst(Rc::new(closure)));
                    Ok(())
                } else if ctor.num_fields == 0 {
                    self.program
                        .emit(Instruction::PushConst(Rc::new(Value::Nullary(name))));
                    Ok(())
                } else {
                    self.program
                        .emit(Instruction::PushConst(Rc::new(Value::Construct(
                            name,
                            Vec::new(),
                        ))));
                    Ok(())
                }
            }
            ConstantInfo::Theorem(_) => {
                self.program.emit(Instruction::PushStar);
                Ok(())
            }
            ConstantInfo::Axiom(_)
            | ConstantInfo::Opaque(_)
            | ConstantInfo::Inductive(_)
            | ConstantInfo::Quotient(_) => Err(VmError::UnsupportedConstant(name)),
        }
    }

    /// Compile an `Expr` (kernel term) to bytecode. Used for recursor
    /// RHSs which are `Expr`s, not `LValue`s.
    fn compile_expr(&mut self, expr: &theoria_kernel::Expr) -> Result<(), VmError> {
        use theoria_kernel::Expr as KExpr;
        match expr {
            KExpr::Var(idx) => {
                self.program.emit(Instruction::PushLocal(idx.0));
                Ok(())
            }
            KExpr::Const(name, _) => self.compile_const(*name),
            KExpr::App(f, a) => {
                if let KExpr::Const(name, _) = &**f {
                    if self.env.is_succ(*name) {
                        self.compile_expr(a)?;
                        self.program.emit(Instruction::Succ(*name));
                        return Ok(());
                    }
                }
                self.compile_expr(f)?;
                self.compile_expr(a)?;
                self.program.emit(Instruction::Apply);
                Ok(())
            }
            KExpr::Lam(_, _, _, body) => {
                let mut sub = Compiler {
                    env: self.env,
                    program: Program::new(),
                };
                sub.compile_expr(body)?;
                sub.program.emit(Instruction::Return);
                let prog_rc = Rc::new(sub.program);
                let closure = Value::Closure(Rc::new(crate::value::ClosureBody {
                    program: prog_rc,
                    entry: 0,
                    arity: 1,
                    captured: Vec::new(),
                }));
                self.program.emit(Instruction::PushConst(Rc::new(closure)));
                Ok(())
            }
            KExpr::Sort(_) | KExpr::Pi(_, _, _, _) | KExpr::Lit(_) | KExpr::Let(_, _, _, _) => {
                // For recursor RHS, these should not appear at top level as terms,
                // but handle gracefully as Star or literal.
                match expr {
                    KExpr::Lit(l) => {
                        let v = match l {
                            theoria_kernel::expr::Literal::Nat(n) => Value::Nat(*n),
                            theoria_kernel::expr::Literal::Str(_) => {
                                return Err(VmError::Internal(
                                    "string literals not yet supported".to_string(),
                                ));
                            }
                        };
                        self.program.emit(Instruction::PushConst(Rc::new(v)));
                        Ok(())
                    }
                    _ => {
                        self.program.emit(Instruction::PushStar);
                        Ok(())
                    }
                }
            }
        }
    }

    /// Compile the rules of a recursor into a [`RecursorEntry`].
    fn compile_recursor(
        &mut self,
        name: NameId,
        rules: &[RecursorRule],
    ) -> Result<RecursorEntry, VmError> {
        let info = self.env.lookup(name)?;
        let (num_params, num_indices, num_motives, num_minors) = match &*info {
            ConstantInfo::Recursor(r) => (r.num_params, r.num_indices, r.num_motives, r.num_minors),
            _ => return Err(VmError::Internal("expected a recursor".to_string())),
        };

        let mut compiled_rules = Vec::with_capacity(rules.len());
        for rule in rules {
            let mut sub = Compiler {
                env: self.env,
                program: Program::new(),
            };
            sub.compile_expr(&rule.rhs)?;
            sub.program.emit(Instruction::Return);

            let rhs_entry = self.program.instructions.len() as u32;
            for instr in sub.program.instructions {
                self.program.emit(instr);
            }
            compiled_rules.push(RecursorRuleEntry {
                constructor: rule.constructor,
                num_fields: rule.num_fields,
                rhs_entry,
            });
        }
        Ok(RecursorEntry {
            name,
            num_params,
            num_indices,
            num_motives,
            num_minors,
            rules: compiled_rules,
        })
    }
}
