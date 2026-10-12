//===- macro_rules.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions for macros and compile-time procedural metaprogramming.
//
//===----------------------------------------------------------------------===//

use crate::lexer::OwnedToken;

/// A piece of the tokens a macro is given or expands to.
#[derive(Debug, Clone, PartialEq)]
pub enum TokenTree {
    /// One token.
    Token(OwnedToken),
    /// Several trees in a row, with no brackets around them.
    Group(Vec<TokenTree>),
    /// The trees between a pair of brackets, and which brackets they are.
    Delimited(Delimiter, Vec<TokenTree>),
}

/// The pair of brackets around a [`TokenTree::Delimited`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Delimiter {
    /// Round brackets: `( )`.
    Parenthesis,
    /// Curly brackets: `{ }`.
    Brace,
    /// Square brackets: `[ ]`.
    Bracket,
}
