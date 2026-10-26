//===- build_and_run_packages.rs - Vx Package Tool -------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Runs the real `vex` program with the real `vxc` compiler on the packages in `tests/packages`:
// `app` depends on `shapes`, and `shapes` depends on `numbers`.
//
// These tests need the compiler and its runtime library built from this tree: `cargo build`.
//
//===----------------------------------------------------------------------===//
//! End-to-end tests of `vex` with the real `vxc` compiler.
#![allow(
    clippy::missing_panics_doc,
    reason = "a test reports a failure by panicking"
)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// The repository root, where the compiler finds `stdlib/`.
fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the vex crate is one directory below the repository root")
        .to_path_buf()
}

/// The `vxc` that cargo built into the same directory as `vex`.
fn compiler() -> PathBuf {
    let vex = Path::new(env!("CARGO_BIN_EXE_vex"));
    let compiler = vex
        .parent()
        .expect("the vex binary is in a build directory")
        .join("vxc");
    assert!(
        compiler.is_file(),
        "no compiler at `{}`. Build it first: cargo build",
        compiler.display()
    );
    compiler
}

/// Copy the test packages to a fresh directory, so a test can write a `target` directory without
/// touching the repository, and so the packages are not below the directory `vex` runs in.
fn copy_packages_to(destination: &Path) {
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/packages");
    for package in ["app", "shapes", "numbers"] {
        let package_destination = destination.join(package);
        fs::create_dir_all(&package_destination).expect("create a package directory");
        for entry in fs::read_dir(source.join(package)).expect("list a test package") {
            let file = entry.expect("read a directory entry").path();
            let name = file.file_name().expect("a listed file has a name");
            fs::copy(&file, package_destination.join(name)).expect("copy a test package file");
        }
    }
}

fn vex(args: &[&str], manifest: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_vex"))
        .arg("--manifest-path")
        .arg(manifest)
        .args(args)
        .env("VXC", compiler())
        .current_dir(repository_root())
        .output()
        .expect("start vex")
}

/// What the run printed, on either stream.
fn printed(output: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

#[test]
fn run_compiles_a_package_with_the_packages_it_depends_on() {
    let packages = tempfile::tempdir().expect("create a temporary directory");
    copy_packages_to(packages.path());

    let output = vex(&["run"], &packages.path().join("app/vex.toml"));
    let printed = printed(&output);
    assert_eq!(output.status.code(), Some(0), "{printed}");
    // 18 is twice(3 * 3): `shapes::area` ran and called `numbers::twice`.
    assert!(printed.contains("area 18"), "{printed}");
    // 4 comes from `app::sides`, a second file of the root package.
    assert!(printed.contains("sides 4"), "{printed}");
}

#[test]
fn build_writes_an_object_file_in_the_package() {
    let packages = tempfile::tempdir().expect("create a temporary directory");
    copy_packages_to(packages.path());

    let output = vex(&["build"], &packages.path().join("app/vex.toml"));
    assert_eq!(output.status.code(), Some(0), "{}", printed(&output));
    let object_file = packages.path().join("app/target/app.o");
    let size = fs::metadata(&object_file)
        .unwrap_or_else(|error| panic!("no `{}`: {error}", object_file.display()))
        .len();
    assert!(size > 0, "`{}` is empty", object_file.display());
}

#[test]
fn a_library_package_cannot_be_run() {
    let packages = tempfile::tempdir().expect("create a temporary directory");
    copy_packages_to(packages.path());

    let output = vex(&["run"], &packages.path().join("numbers/vex.toml"));
    assert_eq!(output.status.code(), Some(1));
    let printed = printed(&output);
    assert!(
        printed.contains("error: package `numbers` has no `main.vx`"),
        "{printed}"
    );
}

#[test]
fn a_directory_without_vex_toml_is_an_error() {
    let empty = tempfile::tempdir().expect("create a temporary directory");

    let output = Command::new(env!("CARGO_BIN_EXE_vex"))
        .arg("run")
        .current_dir(empty.path())
        .output()
        .expect("start vex");
    assert_eq!(output.status.code(), Some(1));
    let printed = printed(&output);
    assert!(printed.contains("error: no `vex.toml` in"), "{printed}");
}
