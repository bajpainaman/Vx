//===- mod.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Abstract Syntax Tree module for the Vx compiler. Defines the core representation of the parsed program.
//
//===----------------------------------------------------------------------===//

/// The declarations of a program: functions, structs, enums, traits, impls and the rest.
pub mod decl;
/// Expressions, such as operators, calls and literals.
pub mod expr;
/// Macro definitions and the rules they are made of.
pub mod macro_rules;
/// Name resolution: attaches a global id to each type name in the tree.
pub mod resolve;
/// Statements, such as `let`, assignment, `if` and loops.
pub mod stmt;
/// Types as they are written in source, and the helpers that work on them.
pub mod types;

pub use decl::*;
pub use expr::*;
pub use macro_rules::*;
pub use stmt::*;
pub use types::*;
