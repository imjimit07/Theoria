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

/// Render syntax errors rustc-style: one `path:line:col: error:` line
/// plus a caret snippet per error.
fn render_syntax_errors(path: &str, contents: &str, errors: &[theoria_parser::SyntaxError]) {
    let map = theoria_parser::SourceMap::new(contents.to_string());
    for e in errors {
        render_diagnostic(path, &map, contents, e.span.start, e.span.end, &e.message);
    }
}

/// Render elaboration errors in the same shape as syntax errors.
fn render_elaborate_errors(
    path: &str,
    contents: &str,
    errors: &[theoria_elaborator::ElaborateError],
) {
    let map = theoria_parser::SourceMap::new(contents.to_string());
    for e in errors {
        render_diagnostic(path, &map, contents, e.span.start, e.span.end, &e.message);
    }
}

/// One `path:line:col: error:` line plus a caret snippet.
fn render_diagnostic(
    path: &str,
    map: &theoria_parser::SourceMap,
    contents: &str,
    start: u32,
    end: u32,
    message: &str,
) {
    let lc = map.line_col(start);
    eprintln!("{path}:{}:{}: error: {message}", lc.line, lc.col);
    let line_text = map.line_text(lc.line);
    let line_chars: Vec<char> = line_text.chars().collect();
    let col0 = (lc.col as usize).saturating_sub(1).min(line_chars.len());
    let span_chars = contents
        .get(start as usize..end as usize)
        .map_or(0, |s| s.chars().take_while(|&c| c != '\n').count())
        .max(1);
    let width = span_chars.min(line_chars.len().saturating_sub(col0).max(1));
    eprintln!("{} | {}", lc.line, line_text);
    eprintln!(
        "{}| {}{}",
        " ".repeat(lc.line.to_string().len()),
        " ".repeat(col0),
        "^".repeat(width)
    );
}
