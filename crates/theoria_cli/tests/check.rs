//! CLI integration tests: `theoria check` on fixture files.
//!
//! The binary is located via `CARGO_BIN_EXE_theoria`; fixtures live in
//! `tests/fixtures/` next to this file. Cargo runs integration tests
//! with the crate root as the working directory, so the fixture paths
//! below resolve relative to `crates/theoria_cli/`.

use std::process::Command;

fn theoria() -> Command {
    Command::new(env!("CARGO_BIN_EXE_theoria"))
}

#[test]
fn check_valid_file_exits_zero_and_names_functions() {
    let out = theoria()
        .arg("check")
        .arg("tests/fixtures/valid.theoria")
        .output()
        .expect("failed to run theoria");
    assert!(
        out.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("parsed OK"), "stdout was:\n{stdout}");
    assert!(stdout.contains("zero"), "stdout was:\n{stdout}");
    assert!(stdout.contains("one"), "stdout was:\n{stdout}");
    assert!(stdout.contains("id (1 parameter)"), "stdout was:\n{stdout}");
}

#[test]
fn check_invalid_indent_exits_one_with_diagnostic() {
    let path = "tests/fixtures/invalid_indent.theoria";
    let out = theoria()
        .arg("check")
        .arg(path)
        .output()
        .expect("failed to run theoria");
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("error: "), "stderr was:\n{stderr}");
    assert!(stderr.contains(path), "stderr was:\n{stderr}");
}

#[test]
fn check_unknown_keyword_reports_not_implemented() {
    let out = theoria()
        .arg("check")
        .arg("tests/fixtures/unknown_keyword.theoria")
        .output()
        .expect("failed to run theoria");
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("not yet implemented"),
        "stderr was:\n{stderr}"
    );
}

#[test]
fn check_elaboration_error_names_identifier() {
    let out = theoria()
        .arg("check")
        .arg("tests/fixtures/elaboration_error.theoria")
        .output()
        .expect("failed to run theoria");
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("unknown identifier"),
        "stderr was:\n{stderr}"
    );
}

#[test]
fn check_type_error_reports_function_span() {
    let path = "tests/fixtures/type_error.theoria";
    let out = theoria()
        .arg("check")
        .arg(path)
        .output()
        .expect("failed to run theoria");
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("error:"), "stderr was:\n{stderr}");
    // The `bad` function starts on line 3, column 1.
    assert!(stderr.contains(":3:1:"), "stderr was:\n{stderr}");
}

#[test]
fn check_nat_rec_plus_elaborates() {
    // The first real program: `plus` via `Nat.rec`, exercising
    // qualified-name resolution, universe defaulting, and the kernel's
    // recursor check end to end through the binary.
    let out = theoria()
        .arg("check")
        .arg("tests/fixtures/valid_nat_rec.theoria")
        .output()
        .expect("failed to run theoria");
    assert!(
        out.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("plus (2 parameters)"),
        "stdout was:\n{stdout}"
    );
}

#[test]
fn check_arith_fixture_elaborates() {
    let out = theoria()
        .arg("check")
        .arg("tests/fixtures/arith.theoria")
        .output()
        .expect("failed to run theoria");
    assert!(
        out.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    for name in ["double", "square", "quad"] {
        assert!(stdout.contains(name), "stdout was:\n{stdout}");
    }
}

#[test]
fn check_comparisons_fixture_elaborates() {
    let out = theoria()
        .arg("check")
        .arg("tests/fixtures/comparisons.theoria")
        .output()
        .expect("failed to run theoria");
    assert!(
        out.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    for name in ["is_zero", "less_than", "if_nat"] {
        assert!(stdout.contains(name), "stdout was:\n{stdout}");
    }
}

#[test]
fn check_shadow_error_names_parameter() {
    let path = "tests/fixtures/shadow_error.theoria";
    let out = theoria()
        .arg("check")
        .arg(path)
        .output()
        .expect("failed to run theoria");
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("shadows the global constant"),
        "stderr was:\n{stderr}"
    );
    // The `Nat` parameter starts on line 5 at column 14.
    assert!(stderr.contains(":5:14:"), "stderr was:\n{stderr}");
}

#[test]
fn check_control_fixture_elaborates() {
    let out = theoria()
        .arg("check")
        .arg("tests/fixtures/control.theoria")
        .output()
        .expect("failed to run theoria");
    assert!(
        out.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    for name in ["min", "abs_diff", "double_square"] {
        assert!(stdout.contains(name), "stdout was:\n{stdout}");
    }
}

#[test]
fn check_bad_if_reports_type_mismatch() {
    let out = theoria()
        .arg("check")
        .arg("tests/fixtures/bad_if.theoria")
        .output()
        .expect("failed to run theoria");
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("type mismatch"), "stderr was:\n{stderr}");
}

#[test]
fn check_if_in_arg_requires_expected_type() {
    let out = theoria()
        .arg("check")
        .arg("tests/fixtures/if_in_arg.theoria")
        .output()
        .expect("failed to run theoria");
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("`if` requires an expected type"),
        "stderr was:\n{stderr}"
    );
}

#[test]
fn check_match_fixture_elaborates() {
    // `match` on `Nat`/`Bool` desugars to recursor applications end to
    // end through the binary: constructor arms, catch-alls, and an
    // application scrutinee.
    let out = theoria()
        .arg("check")
        .arg("tests/fixtures/match.theoria")
        .output()
        .expect("failed to run theoria");
    assert!(
        out.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    for name in ["is_zero", "pred", "neg", "min", "pred_or_zero"] {
        assert!(stdout.contains(name), "stdout was:\n{stdout}");
    }
}

#[test]
fn check_structures_fixture_elaborates() {
    let out = theoria()
        .arg("check")
        .arg("tests/fixtures/structures.theoria")
        .output()
        .expect("failed to run theoria");
    assert!(
        out.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    for name in [
        "Point2D (0 parameters, 2 fields)",
        "Box (1 parameter, 1 field)",
        "origin",
        "unwrap_box",
    ] {
        assert!(stdout.contains(name), "stdout was:\n{stdout}");
    }
}

#[test]
fn check_recursive_structure_fixture_elaborates() {
    let out = theoria()
        .arg("check")
        .arg("tests/fixtures/recursive_structure.theoria")
        .output()
        .expect("failed to run theoria");
    assert!(
        out.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    for name in ["Wrapper", "peek_value"] {
        assert!(stdout.contains(name), "stdout was:\n{stdout}");
    }
}

#[test]
fn check_bad_recursion_reports_positivity() {
    let out = theoria()
        .arg("check")
        .arg("tests/fixtures/bad_recursion.theoria")
        .output()
        .expect("failed to run theoria");
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("arrow"), "stderr was:\n{stderr}");
    assert!(stderr.contains("negative"), "stderr was:\n{stderr}");
}

#[test]
fn check_theorems_fixture_elaborates() {
    let out = theoria()
        .arg("check")
        .arg("tests/fixtures/theorems.theoria")
        .output()
        .expect("failed to run theoria");
    assert!(
        out.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("theorems:  4"), "stdout was:\n{stdout}");
    assert!(stdout.contains("zero_eq_zero"), "stdout was:\n{stdout}");
    assert!(stdout.contains("refl_nat"), "stdout was:\n{stdout}");
    assert!(stdout.contains("h_used"), "stdout was:\n{stdout}");
    assert!(stdout.contains("steps_demo"), "stdout was:\n{stdout}");
}

#[test]
fn check_non_prop_theorem_goal_reports_error() {
    let out = theoria()
        .arg("check")
        .arg("tests/fixtures/bad_prop.theoria")
        .output()
        .expect("failed to run theoria");
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("proposition"),
        "stderr should mention `proposition`, was:\n{stderr}"
    );
}

#[test]
fn check_steps_prop_fixture_elaborates() {
    let out = theoria()
        .arg("check")
        .arg("tests/fixtures/steps_prop.theoria")
        .output()
        .expect("failed to run theoria");
    assert!(
        out.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("theorems:  2"), "stdout was:\n{stdout}");
    assert!(stdout.contains("assume_prop"), "stdout was:\n{stdout}");
    assert!(stdout.contains("let_inferred"), "stdout was:\n{stdout}");
}
