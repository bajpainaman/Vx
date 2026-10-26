//===- lib.rs - Vx Compiler ------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is the root of the statically linked Vx standard library core.
// It aggregates all the low-level FFI implementations (math, networking, I/O,
// collections) that are natively compiled and linked against lowered Vx MLIR code.
//
//===----------------------------------------------------------------------===//

//! The Vx runtime library: the functions a compiled Vx program calls through the C ABI.

#![allow(
    unsafe_code,
    reason = "the whole crate is a C ABI boundary: every export is `#[no_mangle]` and works on raw pointers"
)]
#![allow(
    clippy::not_unsafe_ptr_arg_deref,
    reason = "the exports are called from generated code, not from Rust, so marking them `unsafe` protects no caller"
)]

pub mod collections;
/// Command line arguments of the running Vx program.
pub mod env;
pub mod ffi;
/// Equality checks that Vx tests call to compare a value with the expected one.
pub mod googletest;
pub mod io;
pub mod net;
/// Arithmetic on groups of four `f32` values at a time.
pub mod simd;
