//===- emit_mlir_and_emit_llvm_write_to_o.rs - Vx Compiler ------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// `--action emit-mlir -o FILE` and `--action emit-llvm -o FILE` write to FILE, and
// print nothing. Without `-o` they print, as before.
//
//===----------------------------------------------------------------------===//

use std::path::PathBuf;
use std::process::Command;

const PROGRAM: &str = "fn main() -> i32 {\n    return 4;\n}\n";

fn vxc() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_vxc"));
    cmd.current_dir(env!("CARGO_MANIFEST_DIR"));
    cmd
}

/// A fresh directory holding PROGRAM as `four.vx`.
fn program_in(dir_name: &str) -> (PathBuf, PathBuf) {
    let dir = std::env::temp_dir().join(dir_name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("four.vx");
    std::fs::write(&source, PROGRAM).unwrap();
    (dir, source)
}

/// Both actions print a module with `@main` in it, whatever dialect it is in.
fn check_writes_to_o(action: &str, dir_name: &str) {
    let (dir, source) = program_in(dir_name);
    let out_file = dir.join("out.txt");
    let out = vxc()
        .arg(&source)
        .args(["--action", action, "-o"])
        .arg(&out_file)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "vxc failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let written = std::fs::read_to_string(&out_file)
        .unwrap_or_else(|e| panic!("{action} wrote nothing to {}: {e}", out_file.display()));
    assert!(
        written.contains("@main"),
        "the file should hold the module, got:\n{written}"
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        !stdout.contains("@main"),
        "with -o the module should not also be printed, got:\n{stdout}"
    );
}

#[test]
fn emit_mlir_writes_to_o() {
    check_writes_to_o("emit-mlir", "vx_emit_mlir_to_o");
}

#[test]
fn emit_llvm_writes_to_o() {
    check_writes_to_o("emit-llvm", "vx_emit_llvm_to_o");
}

#[test]
fn without_o_the_module_is_printed() {
    let (_dir, source) = program_in("vx_emit_mlir_to_stdout");
    let out = vxc()
        .arg(&source)
        .args(["--action", "emit-mlir"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("@main"), "got:\n{stdout}");
}

/// `-o -` is standard output, as with a C compiler. The fixtures under tests/frontend use it.
#[test]
fn o_dash_prints_and_writes_no_file() {
    let (dir, source) = program_in("vx_emit_mlir_to_dash");
    let out = vxc()
        .current_dir(&dir)
        .arg(&source)
        .args(["--action", "emit-mlir", "-o", "-"])
        .output()
        .unwrap();
    assert!(out.status.success());
    assert!(
        !dir.join("-").exists(),
        "a file named `-` was written next to the source"
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("@main"), "got:\n{stdout}");
}

#[test]
fn several_inputs_with_o_naming_a_directory_write_one_file_each() {
    let (dir, four) = program_in("vx_emit_mlir_several");
    let five = dir.join("five.vx");
    std::fs::write(&five, PROGRAM.replace('4', "5")).unwrap();
    let out_dir = dir.join("out");
    std::fs::create_dir_all(&out_dir).unwrap();
    let out = vxc()
        .arg(&four)
        .arg(&five)
        .args(["--action", "emit-mlir", "-o"])
        .arg(&out_dir)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "vxc failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    for name in ["four.mlir", "five.mlir"] {
        let written = std::fs::read_to_string(out_dir.join(name))
            .unwrap_or_else(|e| panic!("no {name} in the output directory: {e}"));
        assert!(written.contains("@main"), "{name} should hold a module");
    }
}
