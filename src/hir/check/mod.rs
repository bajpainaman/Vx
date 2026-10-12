//===- check/mod.rs - Vx Compiler -------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Expression type-checking, split out of the former 5000-line `hir/expr.rs` along the
// `check_expr_type_flag` dispatch seam. Each submodule
// holds the checks for one family of expressions as additional `impl TypeChecker` blocks; the dispatch and
// the shared type helpers (`is_assignable`, `lower_to_type_id`) stay in `hir/expr.rs`. Moves only — zero
// logic change.
//
//===----------------------------------------------------------------------===//

/// Type checks for names, fields, indexes, borrows (`&x`) and dereferences (`*x`).
pub mod access;
/// Type checks for `grad`, `vjp` and `jvp`, and the check that a function can be differentiated.
pub mod autodiff;
/// Type checks for function calls, method calls and calls through a function value.
pub mod calls;
/// The memory capacity check across calls: adds up what each function keeps in a memory space.
pub mod capacity_fold;
/// Type checks for `if`, `match`, ranges, closures, and `comptime` and `unsafe` blocks.
pub mod control;
/// Finds where each tensor a function owns is dropped, and writes the drops into the program.
pub mod drops;
/// Refuses a value that points into a function's stack frame and would outlive the function.
pub mod escape;
/// Reports a type error in an imported function only when the program uses that function.
pub mod imported_errors;
/// Type checks for number and array literals, enum variants and struct literals.
pub mod literals;
/// Type checks for `as` casts and for unary, binary, comparison and logical operators.
pub mod operators;
/// Resolves `I::Item`, the associated type of a generic parameter.
pub mod projection;
/// Type checks for the `raw::` primitives that an `impl transfer` is written with.
pub mod raw;
/// Counts what a `spawn on(...)` region reads and writes in each memory space.
pub mod region_traffic;
/// Type checks for `transfer(...)` and `spawn on(...)`, and the memory capacity check.
pub mod transfer;
/// The warning for a function in the program's own file that nothing calls.
pub mod unused_functions;
/// Treats a tensor row or a tensor field as a borrow of the tensor that owns its memory.
pub mod views;
