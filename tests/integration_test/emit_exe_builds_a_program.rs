//===- emit_exe_builds_a_program.rs - Vx Compiler ---------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// `--action emit-exe` builds a program that runs on its own, without vxc.
//
//===----------------------------------------------------------------------===//

use std::path::{Path, PathBuf};
use std::process::Command;

const PROGRAM: &str = r#"fn main() -> i32 {
    print!("built by emit-exe\n");
    return 7;
}
"#;

fn vxc() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_vxc"));
    cmd.current_dir(env!("CARGO_MANIFEST_DIR"));
    cmd
}

/// A fresh directory holding `name.vx` with PROGRAM in it.
fn program_in(dir_name: &str) -> (PathBuf, PathBuf) {
    let dir = std::env::temp_dir().join(dir_name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("seven.vx");
    std::fs::write(&source, PROGRAM).unwrap();
    (dir, source)
}

/// Run the executable at `exe` and check it prints the line and exits with the program's value.
fn runs_on_its_own(exe: &Path) {
    assert!(exe.is_file(), "no executable at {}", exe.display());
    let out = Command::new(exe)
        .output()
        .unwrap_or_else(|e| panic!("could not run {}: {e}", exe.display()));
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "built by emit-exe\n",
        "the program's output"
    );
    assert_eq!(out.status.code(), Some(7), "the program's exit code");
}

#[test]
fn emit_exe_writes_the_executable_named_by_o() {
    let (dir, source) = program_in("vx_emit_exe_named");
    let exe = dir.join("my_program");
    let status = vxc()
        .arg(&source)
        .args(["--action", "emit-exe", "-o"])
        .arg(&exe)
        .status()
        .unwrap();
    assert!(status.success(), "vxc failed");
    runs_on_its_own(&exe);
}

#[test]
fn emit_exe_without_o_names_the_executable_after_the_source() {
    let (dir, source) = program_in("vx_emit_exe_default_name");
    let status = vxc()
        .arg(&source)
        .args(["--action", "emit-exe"])
        .status()
        .unwrap();
    assert!(status.success(), "vxc failed");
    runs_on_its_own(&dir.join("seven"));
}

#[test]
fn emit_exe_refuses_to_overwrite_a_source_with_no_extension() {
    let dir = std::env::temp_dir().join("vx_emit_exe_no_extension");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("seven");
    std::fs::write(&source, PROGRAM).unwrap();
    let out = vxc()
        .arg(&source)
        .args(["--action", "emit-exe"])
        .output()
        .unwrap();
    assert!(!out.status.success(), "vxc should have refused");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("name the output with -o"),
        "the error should say what to do, got:\n{stderr}"
    );
    assert_eq!(
        std::fs::read_to_string(&source).unwrap(),
        PROGRAM,
        "the source must be left as it was"
    );
}
