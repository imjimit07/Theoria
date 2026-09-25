//! Bytecode instruction set.

use alloc::rc::Rc;
use alloc::vec::Vec;
use theoria_kernel::name::NameId;

/// A single bytecode instruction.
///
/// The VM is a stack machine: most instructions pop their arguments
/// from the top of the stack and push their result. This is the
/// simplest design that also compiles straightforwardly to WASM.
#[derive(Clone, Debug)]
pub enum Instruction {
    /// Push the constant value `c` onto the stack.
    PushConst(Rc<crate::value::Value>),
    /// Push the value of local variable `slot` from the current frame.
    PushLocal(u32),
    /// Pop `n` values, package them into a closure whose code entry is
    /// `entry` and whose arity is `n`, and push the closure.
    MakeClosure {
        /// Byte offset of the closure's body.
        entry: u32,
        /// Number of parameters.
        arity: u32,
    },
    /// Pop one value (the argument) and one closure; apply.
    Apply,
    /// Pop one value; if it is `Nat(n)`, push `Nat(n + 1)`; if it is
    /// any other constructor, wrap it in `Unary(succ_name, v)`.
    ///
    /// `succ_name` is the kernel `NameId` for `Nat.succ`; it is a field
    /// of the instruction so the VM does not need an environment lookup
    /// per call.
    Succ(NameId),
    /// Pop the major premise and execute the given recursor.
    ///
    /// The remaining arguments (motive, minors, parameters, indices) are
    /// on the stack under the major premise in the order the compiler
    /// emitted them. `RecursorEntry` records where to find each.
    Recurse(RecursorEntry),
    /// Return the top of the stack as the result of the current frame.
    Return,
    /// Push `Value::Star`.
    PushStar,
}

/// Recursor execution metadata baked into the compiled program.
///
/// `theoria_kernel::env::RecursorVal` is looked up at compile time and
/// flattened into this struct so the interpreter does not need to
/// consult the environment at run time.
#[derive(Clone, Debug)]
pub struct RecursorEntry {
    /// The recursor's name, for diagnostics.
    pub name: NameId,
    /// Number of parameters.
    pub num_params: u32,
    /// Number of indices (currently always 0 for prelude recursors).
    pub num_indices: u32,
    /// Number of motives.
    pub num_motives: u32,
    /// Number of minors.
    pub num_minors: u32,
    /// Rules, one per constructor.
    pub rules: Vec<RecursorRuleEntry>,
}

/// One ι-rule, precompiled into the program.
#[derive(Clone, Debug)]
pub struct RecursorRuleEntry {
    /// The constructor this rule fires for.
    pub constructor: NameId,
    /// Number of fields the constructor binds.
    pub num_fields: u32,
    /// Where the right-hand side lives in the program.
    pub rhs_entry: u32,
}

/// A compiled program.
#[derive(Clone, Debug)]
pub struct Program {
    /// The instruction stream.
    pub instructions: Vec<Instruction>,
    /// Human-readable labels for diagnostics, parallel to a subset of
    /// `instructions`. Not used by the interpreter.
    pub labels: Vec<(u32, alloc::string::String)>,
}

impl Program {
    /// A fresh, empty program.
    #[must_use]
    pub fn new() -> Self {
        Program {
            instructions: Vec::new(),
            labels: Vec::new(),
        }
    }

    /// Append an instruction, returning its index.
    pub fn emit(&mut self, i: Instruction) -> u32 {
        let idx = u32::try_from(self.instructions.len()).expect("program size overflow");
        self.instructions.push(i);
        idx
    }

    /// Attach a diagnostic label to the next instruction to be emitted.
    pub fn label(&mut self, name: impl Into<alloc::string::String>) {
        let idx = u32::try_from(self.instructions.len()).expect("program size overflow");
        self.labels.push((idx, name.into()));
    }
}

impl Default for Program {
    fn default() -> Self {
        Self::new()
    }
}
