//===- link_a_c_library.rs - Vx Compiler ------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// `-l NAME` links a C library into the program, for `--run` and for `emit-exe`.
// The program calls zlib's `crc32`, which every machine that builds Vx has.
//
//===----------------------------------------------------------------------===//

use std::path::PathBuf;
use std::process::Command;

const PROGRAM: &str = r#"extern {
    fn crc32(crc : u64, buf : *const i8, len : u32) -> u64;
}

fn main() -> i32 {
    let c = unsafe { crc32(0u64, "123456789", 9u32) };
    print(c);
    print!("\n");
    return 0;
}
"#;

/// crc32 of "123456789", the check value zlib's own tests use.
const CRC_OF_123456789: &str = "3421780262";

fn vxc() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_vxc"));
    cmd.current_dir(env!("CARGO_MANIFEST_DIR"));
    cmd
}

/// A fresh directory holding the program as `crc.vx`.
fn program_in(dir_name: &str) -> (PathBuf, PathBuf) {
    let dir = std::env::temp_dir().join(dir_name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("crc.vx");
    std::fs::write(&source, PROGRAM).unwrap();
    (dir, source)
}

#[test]
fn emit_exe_links_the_library_l_names() {
    let (dir, source) = program_in("vx_link_zlib_exe");
    let exe = dir.join("crc");
    let out = vxc()
        .arg(&source)
        .args(["--action", "emit-exe", "-l", "z", "-o"])
        .arg(&exe)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "vxc failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let run = Command::new(&exe).output().unwrap();
    assert_eq!(
        String::from_utf8_lossy(&run.stdout).trim(),
        CRC_OF_123456789,
        "the crc the program printed"
    );
}

#[test]
fn run_links_the_library_l_names() {
    let (_dir, source) = program_in("vx_link_zlib_run");
    let out = vxc()
        .arg(&source)
        .args(["--run", "-l", "z"])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "vxc failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        String::from_utf8_lossy(&out.stdout).contains(CRC_OF_123456789),
        "the crc the program printed"
    );
}

/// `-L DIR` finds a library that is only in DIR, and the built program finds it there too when
/// it runs. The library is built here with clang, which vxc needs anyway.
#[test]
fn capital_l_names_a_directory_the_program_also_searches_at_run_time() {
    let dir = std::env::temp_dir().join("vx_link_own_library");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("answer.c"),
        "long vx_test_answer(void) { return 42; }\n",
    )
    .unwrap();
    let lib_name = format!("libvxanswer{}", std::env::consts::DLL_SUFFIX);
    let clang = std::env::var("CLANG_PATH").unwrap_or_else(|_| "clang".to_string());
    let mut build_lib = Command::new(clang);
    build_lib
        .current_dir(&dir)
        .args(["-shared", "answer.c", "-o", &lib_name]);
    if cfg!(target_os = "macos") {
        // A dylib records the path it expects to be loaded from. `@rpath/` makes it the
        // directory the program's rpath names, which is what `-L` is claimed to add.
        build_lib.arg(format!("-Wl,-install_name,@rpath/{lib_name}"));
    }
    let built = build_lib.output().unwrap();
    assert!(
        built.status.success(),
        "clang could not build the test library:\n{}",
        String::from_utf8_lossy(&built.stderr)
    );

    let source = dir.join("answer.vx");
    std::fs::write(
        &source,
        r#"extern {
    fn vx_test_answer() -> i64;
}

fn main() -> i32 {
    print(unsafe { vx_test_answer() });
    print!("\n");
    return 0;
}
"#,
    )
    .unwrap();
    let exe = dir.join("answer");
    let out = vxc()
        .arg(&source)
        .args(["--action", "emit-exe", "-l", "vxanswer", "-L"])
        .arg(&dir)
        .arg("-o")
        .arg(&exe)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "vxc failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let run = Command::new(&exe).output().unwrap();
    assert!(
        run.status.success(),
        "the program should run from where it was built, but:\n{}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "42");
}

#[test]
fn without_l_the_link_fails_and_names_the_missing_function() {
    let (dir, source) = program_in("vx_link_zlib_missing");
    let out = vxc()
        .arg(&source)
        .args(["--action", "emit-exe", "-o"])
        .arg(dir.join("crc"))
        .output()
        .unwrap();
    assert!(!out.status.success(), "the link should have failed");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("crc32"),
        "the error should name the function nothing defines, got:\n{stderr}"
    );
}

#[test]
fn a_library_that_does_not_exist_is_an_error_that_names_it() {
    let (dir, source) = program_in("vx_link_no_such_library");
    let out = vxc()
        .arg(&source)
        .args(["--action", "emit-exe", "-l", "vx_no_such_library", "-o"])
        .arg(dir.join("crc"))
        .output()
        .unwrap();
    assert!(!out.status.success(), "the link should have failed");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("vx_no_such_library"),
        "the error should name the library, got:\n{stderr}"
    );
}
