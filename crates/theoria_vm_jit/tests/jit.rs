//! JIT end-to-end tests: straight-line `Nat` programs.

use std::rc::Rc;
use theoria_kernel::name::NameId;
use theoria_kernel::prelude::build_prelude;
use theoria_vm::bytecode::{Instruction, Program};
use theoria_vm::value::Value;
use theoria_vm::{Vm, VmEnv};
use theoria_vm_jit::Jit;

fn simple_env() -> (
    theoria_kernel::env::GlobalEnv,
    theoria_kernel::name::NameTable,
) {
    let p = build_prelude();
    (p.env, p.names)
}

#[test]
fn jit_nat_literal() {
    // A hand-built program: `PushConst(Nat(42)); Return`.
    let mut prog = Program::new();
    prog.emit(Instruction::PushConst(Rc::new(Value::Nat(42))));
    prog.emit(Instruction::Return);
    let program = Rc::new(prog);

    let (env, mut names) = simple_env();
    let vm_env = Rc::new(VmEnv::build(&env, &mut names).unwrap());
    let jit = Jit::compile(Rc::clone(&program), vm_env).unwrap();
    let result = jit.run(&[]).unwrap();
    assert_eq!(result.as_nat(), Some(42));
}

#[test]
fn jit_succ_chain() {
    // `Succ(Succ(PushConst(Nat(0))))`.
    let mut prog = Program::new();
    prog.emit(Instruction::PushConst(Rc::new(Value::Nat(0))));
    prog.emit(Instruction::Succ(NameId(0)));
    prog.emit(Instruction::Succ(NameId(0)));
    prog.emit(Instruction::Return);
    let program = Rc::new(prog);

    let (env, mut names) = simple_env();
    let vm_env = Rc::new(VmEnv::build(&env, &mut names).unwrap());
    let jit = Jit::compile(Rc::clone(&program), vm_env).unwrap();
    let result = jit.run(&[]).unwrap();
    assert_eq!(result.as_nat(), Some(2));
}

#[test]
fn jit_agrees_with_interpreter_on_succ_chains() {
    let mut prog = Program::new();
    prog.emit(Instruction::PushConst(Rc::new(Value::Nat(0))));
    for _ in 0..20 {
        prog.emit(Instruction::Succ(NameId(0)));
    }
    prog.emit(Instruction::Return);
    let program = Rc::new(prog);

    let (env, mut names) = simple_env();
    let vm_env = VmEnv::build(&env, &mut names).unwrap();

    let interp = Vm::new(&vm_env).run(&program).unwrap();
    // Also exercise the new `run_entry` accessor on the same program.
    let via_entry = Vm::new(&vm_env).run_entry(&program, 0, Vec::new()).unwrap();
    assert_eq!(interp.as_nat(), via_entry.as_nat());

    let vm_env_rc = Rc::new(vm_env);
    let jit = Jit::compile(Rc::clone(&program), vm_env_rc).unwrap();
    let jitted = jit.run(&[]).unwrap();

    assert_eq!(interp.as_nat(), jitted.as_nat());
}
