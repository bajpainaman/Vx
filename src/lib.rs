//===- lib.rs - Vx Compiler ------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is the primary library entry point for the Vx compiler crate.
// It exports all public modules, encompassing the lexer, parser, semantic analyzer,
// borrow checker, and the MLIR code generation backend, allowing the compiler to
// be embedded or tested modularly.
//
//===----------------------------------------------------------------------===//
//! The Vx compiler as a library: the lexer, parser, type checker, borrow checker and the MLIR
//! code generator.

/// The memory spaces and devices of a machine, and what it costs to move data between them.
pub mod arch;
/// The borrow checker: rejects code that breaks the aliasing and mutability rules.
pub mod borrow;
/// The flat intermediate representation: a function as a list of simple instructions.
pub mod bytecode;
/// Code generation: turns a type-checked program into MLIR.
pub mod codegen;
/// Settings that say how a compile runs, as opposed to what it compiles.
pub mod config;
/// The reasons the flat code generator can give for not handling a function.
pub mod decline;
/// Compiler errors and warnings, with their codes and source locations.
pub mod diagnostic;
/// Writes the result of a compile as JSON for `--diagnostics-json`.
pub mod diagnostics_json;
/// The compiler driver: reads files, runs each compiler stage, and writes the output.
pub mod driver;
/// The compiler error type, and how an error is printed with its source line.
pub mod error;
/// The Vx source code formatter.
pub mod formatter;
/// Global identifiers: how a type and its properties are packed into 64-bit numbers.
pub mod gid;
/// Hashes that give the same result on every build and machine.
pub mod hash;
/// Semantic analysis: the type checker and the checks that run on a typed program.
pub mod hir;
/// Editor support: analyses a file that is open in an editor.
pub mod ide;
/// How a generic instantiation gets its identity, and timing hooks for the parallel compiler.
pub mod intern_mode;
/// Compiles MLIR to machine code in memory and runs it.
pub mod jit;
/// Computes the size, alignment and field offsets of structs and enums.
pub mod layout;
/// The lexer: turns source text into tokens.
pub mod lexer;
/// Reads and writes the type and function information stored in compiled files.
pub mod metadata;
/// How a Vx type is written in MLIR.
pub mod mlir_ty;
/// Finds the source files a program imports, reads them and parses them.
pub mod module_loader;
/// Renames a top-level name that two modules of one compile both declare.
pub mod name_clashes;
/// Checks that the parallel parts of the compiler do not use locks.
pub mod parallel_architecture_verifier;
/// The parser: turns tokens into a syntax tree.
pub mod parser;
/// Runs the compiler stages in order for a whole project.
pub mod pipeline;
/// Plugins that let the compiler target hardware accelerators.
pub mod plugin;
/// The registry of loaded modules and the functions and types they declare.
pub mod registry;
/// What the memory analysis found: where data lives, how it moves and what that costs.
pub mod report;
/// Name resolution: finds the declaration each name refers to.
pub mod resolver;
/// The state of one compile, shared between its worker threads.
pub mod session;
/// "Did you mean?" suggestions for a name that was not found.
pub mod suggest;
/// `Symbol`, a shared string that is cheap to copy and compare.
pub mod symbol;
/// The syntax tree a parsed program is stored in.
pub mod syntax;
/// Prints a syntax tree back as Vx source text.
pub mod syntax_printer;
/// Byte units such as `KB` and `GiB`, and their exact conversion to bytes.
pub mod units;

/// Convenience API for parsing a string representation of a module into an `VxModule` (AST).
/// Useful for unit testing and interactive REPLs.
///
/// # Errors
///
/// Returns the formatted syntax error when `source` does not parse.
pub fn parse_module(source: &str) -> Result<syntax::VxModule, crate::error::Error> {
    let mut lexer = lexer::Lexer::new(source);
    let tokens = lexer.tokenize();
    let mut parser = parser::Parser::new(&tokens, source);
    parser
        .parse()
        .map_err(|e| crate::error::Error(e.format(source)))
}
