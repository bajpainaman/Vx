//===- vx-format.rs - Vx Compiler ------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the standalone Vx code formatter.
// It parses raw Vx source code using the compiler's lexer and outputs a standardized,
// cleanly indented version of the code, enforcing consistent styling conventions
// across all Vx projects.
//
//===----------------------------------------------------------------------===//

//! The `vx-format` command line program: it rewrites Vx source files in the standard layout.
use std::fs;
use std::path::Path;

use vxc::formatter::format_file;

use clap::Parser;
use rayon::prelude::*;

#[derive(Parser)]
#[command(author, version, about = "Vx Code Formatter")]
struct Cli {
    /// Number of spaces for indentation
    #[arg(long, default_value_t = 2)]
    indent: usize,

    /// Files to format
    #[arg(required = true)]
    files: Vec<String>,
}

/// Formats one file in place and prints whether it changed.
///
/// # Errors
///
/// Returns an error when the file does not exist, cannot be read, or cannot be written.
fn process_file(file_path: &str, indent_spaces: usize) -> anyhow::Result<()> {
    let path = Path::new(file_path);
    if !path.exists() {
        anyhow::bail!("File not found: {file_path}");
    }

    let content = fs::read_to_string(path)?;
    let formatted = format_file(&content, indent_spaces);
    if formatted == content {
        println!("Unchanged {file_path}");
    } else {
        fs::write(path, formatted)?;
        println!("Formatted {file_path}");
    }
    Ok(())
}

fn main() {
    let cli = Cli::parse();

    cli.files.par_iter().for_each(|file_path| {
        if let Err(e) = process_file(file_path, cli.indent) {
            eprintln!("Error processing {file_path}: {e:?}");
        }
    });
}
