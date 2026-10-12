//===- vx-opt.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Standalone Vx optimizer tool that interfaces with the MLIR backend.
//
//===----------------------------------------------------------------------===//

//! The `vx-opt` command line program: it hands its arguments to the C++ MLIR optimizer.

use std::ffi::CString;
use std::os::raw::{c_char, c_int};

// Force Cargo to pull in melior's transitive dependencies (MLIR libraries)
#[allow(
    unused_imports,
    clippy::single_component_path_imports,
    reason = "the import exists only to make cargo link the MLIR libraries"
)]
use melior;

#[allow(
    clippy::duplicated_attributes,
    reason = "each link attribute names a different static library"
)]
#[link(name = "vx_dialect", kind = "static")]
#[link(name = "plugin_loader", kind = "static")]
extern "C" {
    fn run_vx_opt(argc: c_int, argv: *const *const c_char) -> c_int;
}

use std::os::unix::ffi::OsStrExt;

/// Converts the arguments to C strings, runs the optimizer, and exits with its status.
///
/// # Panics
///
/// Panics when there are more arguments than a C `int` can count.
#[allow(unsafe_code, reason = "the optimizer is a C function")]
fn main() {
    let c_args: Vec<CString> = std::env::args_os()
        .map(|arg| CString::new(arg.as_bytes()).unwrap_or_else(|_| c"invalid_arg".to_owned()))
        .collect();

    let c_ptrs: Vec<*const c_char> = c_args.iter().map(|a| a.as_ptr()).collect();
    let argc = c_int::try_from(c_ptrs.len()).expect("the argument count fits in a C int");

    // SAFETY: `c_ptrs` holds `argc` pointers to NUL-terminated strings that `c_args` keeps alive
    // for the whole call.
    let status = unsafe { run_vx_opt(argc, c_ptrs.as_ptr()) };

    std::process::exit(status);
}
