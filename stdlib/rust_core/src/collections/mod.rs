//===- mod.rs - Vx Compiler ------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file aggregates the standard library collection implementations.
// It exposes the FFI bindings for Vec, HashMap, HashSet, String, and other core
// data structures to the Vx runtime environment.
//
//===----------------------------------------------------------------------===//

//! The collection types of the Vx standard library, exported through the C ABI.

/// Hash maps with integer keys and values.
pub mod hash_map;
/// Hash sets of integers.
pub mod hash_set;
/// Optional values of the number types.
pub mod option;
/// Results whose success and error values are numbers.
pub mod result;
/// Growable UTF-8 strings.
pub mod string;
/// Growable arrays, and the byte allocator behind the Vx `Vec<T>`.
pub mod vec;
