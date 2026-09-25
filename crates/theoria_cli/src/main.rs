//! `theoria` — command-line interface for Project Theoria.
//!
//! Currently a Phase-1 stub. The three commands that work today are:
//!
//! ```text
//! theoria version      print version
//! theoria prelude      describe the kernel's built-in prelude
//! theoria check FILE   read a `.theoria` file and report status
//! ```
//!
//! `check` does not yet parse. It reads the file, prints its size, and
//! reports that the parser will arrive in Phase 2. Exit code is `0` if
//! the file is readable, `1` if it is not.

use std::env;
use std::fs;
use std::io::{self, Write};
use std::process::ExitCode;

use theoria_diagnostics::{Diagnostic, Label, Renderer, SourceFile};
use theoria_kernel::env::ConstantInfo;
use theoria_kernel::prelude::build_prelude;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        None | Some("help") | Some("--help") | Some("-h") => {
            print_help();
            ExitCode::SUCCESS
        }
        Some("version") | Some("--version") | Some("-V") => {
            print_version();
            ExitCode::SUCCESS
        }
        Some("prelude") => cmd_prelude(),
        Some("check") => cmd_check(args.get(1).map(String::as_str)),
        Some("run") => cmd_run(args.get(1).map(String::as_str)),
        Some(other) => {
            eprintln!("theoria: unknown command: {other}");
            eprintln!("Try `theoria help`.");
            ExitCode::from(2)
        }
    }
}

fn print_help() {
    println!(
        "theoria {} — Project Theoria command-line interface\n\
         \n\
         USAGE:\n\
         \x20   theoria <COMMAND> [ARGS]\n\
         \n\
          COMMANDS:\n\
         \x20   version         Print the version.\n\
         \x20   prelude         Describe the kernel's built-in prelude.\n\
         \x20   check <FILE>    Read a .theoria file and report status.\n\
         \x20   run <FILE>      Elaborate and execute a .theoria file.\n\
         \x20   help            Print this help.\n\
         \n\
         The parser, elaborator, and REPL arrive in Phase 2.",
        env!("CARGO_PKG_VERSION"),
    );
}

fn print_version() {
    println!("theoria {}", env!("CARGO_PKG_VERSION"));
    println!("kernel: {}", theoria_kernel::VERSION);
}

fn cmd_prelude() -> ExitCode {
    let p = build_prelude();
    let mut out = io::stdout().lock();

    let _ = writeln!(out, "Prelude — {} declarations\n", p.env.len());

    let mut inductives = Vec::new();
    let mut recursors = Vec::new();
    let mut constructors = Vec::new();
    let mut axioms = Vec::new();
    let mut others = Vec::new();

    for (name, info) in p.env.iter() {
        match &**info {
            ConstantInfo::Inductive(_) => inductives.push(*name),
            ConstantInfo::Recursor(_) => recursors.push(*name),
            ConstantInfo::Constructor(_) => constructors.push(*name),
            ConstantInfo::Axiom(_) => axioms.push(*name),
            _ => others.push(*name),
        }
    }

    print_group(&mut out, "Inductives", &inductives, &p.names);
    print_group(&mut out, "Constructors", &constructors, &p.names);
    print_group(&mut out, "Recursors", &recursors, &p.names);
    print_group(&mut out, "Axioms", &axioms, &p.names);
    if !others.is_empty() {
        print_group(&mut out, "Other", &others, &p.names);
    }
    ExitCode::SUCCESS
}

fn print_group(
    out: &mut io::StdoutLock<'_>,
    label: &str,
    names: &[theoria_kernel::NameId],
    table: &theoria_kernel::NameTable,
) {
    if names.is_empty() {
        return;
    }
    let _ = writeln!(out, "{label} ({}):", names.len());
    let mut sorted: Vec<&str> = names.iter().map(|n| table.resolve(*n)).collect();
    sorted.sort_unstable();
    for name in sorted {
        let _ = writeln!(out, "  {name}");
    }
    let _ = writeln!(out);
}

fn cmd_check(path: Option<&str>) -> ExitCode {
    let Some(path) = path else {
        eprintln!("theoria: `check` requires a file argument");
        return ExitCode::from(2);
    };
    let contents = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("theoria: cannot read {path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    match theoria_parser::parse_module(&contents) {
        Ok(module) => match theoria_elaborator::elaborate_module(&module) {
            Ok(elaborated) => {
                print_parse_summary(path, &module, &elaborated);
                ExitCode::SUCCESS
            }
            Err(errors) => {
                render_elaborate_errors(path, &contents, &errors);
                ExitCode::FAILURE
            }
        },
        Err(errors) => {
            render_syntax_errors(path, &contents, &errors);
            ExitCode::FAILURE
        }
    }
}

fn cmd_run(path: Option<&str>) -> ExitCode {
    let Some(path) = path else {
        eprintln!("theoria: `run` requires a file argument");
        return ExitCode::from(2);
    };
    let contents = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("theoria: cannot read {path}: {e}");
            return ExitCode::FAILURE;
        }
    };

    let module = match theoria_parser::parse_module(&contents) {
        Ok(m) => m,
        Err(errors) => {
            render_syntax_errors(path, &contents, &errors);
            return ExitCode::FAILURE;
        }
    };

    let mut elaborated = match theoria_elaborator::elaborate_module(&module) {
        Ok(e) => e,
        Err(errors) => {
            render_elaborate_errors(path, &contents, &errors);
            return ExitCode::FAILURE;
        }
    };

    // Build the VM environment. Errors here are kernel errors, not
    // surface errors; report them plainly.
    let vm_env = match theoria_vm::VmEnv::build(&elaborated.env, &mut elaborated.names) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("theoria: cannot construct VM environment: {e}");
            return ExitCode::FAILURE;
        }
    };

    // Execute the last zero-arity function declared in the module, if
    // any. If the module has no zero-arity function, report that the
    // module has nothing to run.
    let entry = elaborated.functions.iter().rev().find(|f| f.arity == 0);
    let Some(entry) = entry else {
        eprintln!(
            "theoria: no zero-arity function to run in `{path}`; declare one with `Function main() -> …`"
        );
        return ExitCode::FAILURE;
    };

    let body = match vm_env.erased_body(entry.kernel_name) {
        Some(b) => b,
        None => {
            eprintln!(
                "theoria: function `{}` has no erased body (not a definition?)",
                entry.source_name,
            );
            return ExitCode::FAILURE;
        }
    };

    let program = match theoria_vm::Compiler::new(&vm_env).compile(&body) {
        Ok(p) => std::rc::Rc::new(p),
        Err(e) => {
            eprintln!("theoria: cannot compile `{}`: {e}", entry.source_name);
            return ExitCode::FAILURE;
        }
    };

    let vm = theoria_vm::Vm::new(&vm_env);
    match vm.run(&program) {
        Ok(v) => {
            println!("theoria: {} → {:?}", entry.source_name, v);
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("theoria: runtime error in `{}`: {e}", entry.source_name);
            ExitCode::FAILURE
        }
    }
}

/// Print a short summary of a successfully parsed and elaborated module.
fn print_parse_summary(
    path: &str,
    module: &theoria_parser::Module,
    elaborated: &theoria_elaborator::ElaboratedModule,
) {
    println!("theoria: {path}: parsed OK");
    match &module.decl {
        Some(decl) => {
            let dotted: Vec<&str> = decl.path.segments.iter().map(|s| s.text.as_str()).collect();
            println!("  module:    {}", dotted.join("."));
        }
        None => println!("  module:    (none)"),
    }
    println!("  imports:   {}", module.imports.len());
    println!("  structures: {}", elaborated.structures.len());
    for s in &elaborated.structures {
        let p_plural = if s.num_params == 1 { "" } else { "s" };
        let f_plural = if s.num_fields == 1 { "" } else { "s" };
        println!(
            "    {} ({} parameter{p_plural}, {} field{f_plural})",
            s.source_name, s.num_params, s.num_fields
        );
    }
    println!("  functions: {}", elaborated.functions.len());
    for f in &elaborated.functions {
        let plural = if f.arity == 1 { "" } else { "s" };
        println!("    {} ({} parameter{plural})", f.source_name, f.arity);
    }
    println!("  theorems:  {}", elaborated.theorems.len());
    for t in &elaborated.theorems {
        println!(
            "    {} ({} given, {} assume)",
            t.source_name, t.num_given, t.num_assume
        );
    }
}

fn render_syntax_errors(path: &str, contents: &str, errors: &[theoria_parser::SyntaxError]) {
    let source = SourceFile::new(path, contents);
    let sources = [source];
    let renderer = Renderer::new(&sources);
    let mut out = String::new();
    for e in errors {
        let diag = Diagnostic::error(e.message.clone()).with_label(Label::primary(0, e.span));
        let _ = renderer.render(&diag, &mut out);
    }
    eprint!("{out}");
}

fn render_elaborate_errors(
    path: &str,
    contents: &str,
    errors: &[theoria_elaborator::ElaborateError],
) {
    let source = SourceFile::new(path, contents);
    let sources = [source];
    let renderer = Renderer::new(&sources);
    let mut out = String::new();
    for e in errors {
        let diag = Diagnostic::error(e.message.clone()).with_label(Label::primary(0, e.span));
        let _ = renderer.render(&diag, &mut out);
    }
    eprint!("{out}");
}
