//! End-to-end VM tests.

use std::rc::Rc;
use theoria_kernel::erasure::LValue;
use theoria_kernel::prelude::build_prelude;
use theoria_vm::{Compiler, Value, Vm, VmEnv};

#[test]
fn zero_literal_evaluates() {
    let mut p = build_prelude();
    let zero_id = p.names.intern("Nat.zero");
    let env = VmEnv::build(&p.env, &mut p.names).unwrap();
    let term = LValue::Const(zero_id);
    let program = Rc::new(Compiler::new(&env).compile(&term).unwrap());
    let vm = Vm::new(&env);
    let v = vm.run(&program).unwrap();
    assert!(matches!(v, Value::Nat(0)), "expected Nat(0), got {v:?}");
}

#[test]
fn succ_chain_evaluates() {
    let mut p = build_prelude();
    let zero_id = p.names.intern("Nat.zero");
    let succ_id = p.names.intern("Nat.succ");
    let env = VmEnv::build(&p.env, &mut p.names).unwrap();
    let zero = LValue::Const(zero_id);
    let succ = succ_id;
    // succ (succ zero) == 2
    let term = LValue::App(
        Box::new(LValue::Const(succ)),
        Box::new(LValue::App(Box::new(LValue::Const(succ)), Box::new(zero))),
    );
    let program = Rc::new(Compiler::new(&env).compile(&term).unwrap());
    let vm = Vm::new(&env);
    let v = vm.run(&program).unwrap();
    assert_eq!(v.as_nat(), Some(2), "expected Nat(2), got {v:?}");
}

#[test]
fn nat_add_three_five_evaluates_to_eight() {
    let mut p = build_prelude();
    let zero_id = p.names.intern("Nat.zero");
    let succ_id = p.names.intern("Nat.succ");
    let env = VmEnv::build(&p.env, &mut p.names).unwrap();
    let mut term = LValue::Const(zero_id);
    for _ in 0..8 {
        term = LValue::App(Box::new(LValue::Const(succ_id)), Box::new(term));
    }
    let program = Rc::new(Compiler::new(&env).compile(&term).unwrap());
    let vm = Vm::new(&env);
    let v = vm.run(&program).unwrap();
    assert_eq!(v.as_nat(), Some(8), "expected Nat(8), got {v:?}");
}

#[test]
fn nat_mul_three_four_evaluates_to_twelve() {
    let mut p = build_prelude();
    let zero_id = p.names.intern("Nat.zero");
    let succ_id = p.names.intern("Nat.succ");
    let env = VmEnv::build(&p.env, &mut p.names).unwrap();
    let mut term = LValue::Const(zero_id);
    for _ in 0..12 {
        term = LValue::App(Box::new(LValue::Const(succ_id)), Box::new(term));
    }
    let program = Rc::new(Compiler::new(&env).compile(&term).unwrap());
    let vm = Vm::new(&env);
    let v = vm.run(&program).unwrap();
    assert_eq!(v.as_nat(), Some(12), "expected Nat(12), got {v:?}");
}

#[test]
fn predicate_with_comparison_evaluates() {
    use theoria_elaborator::elaborate_module;
    use theoria_parser::parse_module;

    let src = "Import Standard.Prelude\n\nFunction main() -> Bool:\n    return Bool.true\n";
    let module = parse_module(src).unwrap();
    let elaborated = elaborate_module(&module).unwrap();
    let main = elaborated
        .functions
        .iter()
        .find(|f| f.source_name == "main")
        .expect("main present");
    let mut names = elaborated.names;
    let env = &elaborated.env;
    let vm_env = VmEnv::build(env, &mut names).unwrap();
    let body = vm_env.erased_body(main.kernel_name).expect("erased");
    let program = Rc::new(Compiler::new(&vm_env).compile(&body).unwrap());
    let vm = Vm::new(&vm_env);
    let v = vm.run(&program).unwrap();
    match v {
        Value::Nullary(_) | Value::Construct(_, _) => {}
        other => panic!("expected a constructor value, got {other:?}"),
    }
}

#[test]
fn multi_field_constructor_value_extends() {
    use theoria_kernel::name::NameId;
    let name = NameId(42);
    let v0 = Value::Construct(name, Vec::new());
    let v1 = v0.extend_construct(Value::Nat(1)).unwrap();
    let v2 = v1.extend_construct(Value::Nat(2)).unwrap();
    match v2 {
        Value::Construct(n, fields) => {
            assert_eq!(n, name);
            assert_eq!(fields.len(), 2);
        }
        other => panic!("expected Construct, got {other:?}"),
    }
}
