//===- main.rs - Vx Package Tool -------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// `vex` builds a Vx package. It reads `vex.toml`, finds the packages that one depends on, and
// runs the `vxc` compiler with a `--package` flag for each of them.
//
//===----------------------------------------------------------------------===//
//! `vex` builds a Vx package and the packages it depends on.

mod compiler;
mod graph;
mod manifest;

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::{self, Command};

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand};

use compiler::{Action, CompileOptions};
use graph::PackageGraph;
use manifest::{Manifest, MANIFEST_FILE_NAME};

#[derive(Parser)]
#[command(version, about = "Builds a Vx package and the packages it depends on")]
struct Cli {
    /// The vex.toml to use. Without it, vex looks in this directory and then in each one above it
    #[arg(long = "manifest-path", value_name = "FILE", global = true)]
    manifest_path: Option<PathBuf>,

    #[command(subcommand)]
    command: CliCommand,
}

#[derive(Subcommand)]
enum CliCommand {
    /// Compile the package to an object file in its `target` directory
    Build(CompileFlags),
    /// Compile the package and run it. The exit code is the program's own
    Run {
        #[command(flatten)]
        flags: CompileFlags,

        /// Arguments for the program, after `--`
        #[arg(last = true)]
        program_args: Vec<String>,
    },
}

#[derive(Args)]
struct CompileFlags {
    /// Compile against this machine file, as `vxc --machine` does
    #[arg(long = "machine", value_name = "FILE")]
    machine: Option<PathBuf>,

    /// The host the program runs on, as `vxc --host` takes it: a file, or `default`
    #[arg(long = "host", value_name = "FILE|default")]
    host: Option<String>,
}

fn main() {
    match run(Cli::parse()) {
        Ok(exit_code) => process::exit(exit_code),
        Err(error) => {
            eprintln!("error: {error:#}");
            process::exit(1);
        }
    }
}

/// Returns the exit code of the compiler, which for `vex run` is the program's own.
///
/// # Errors
///
/// If the package or one of its dependencies is not usable, or the compiler cannot be run.
fn run(cli: Cli) -> Result<i32> {
    let manifest_path = match cli.manifest_path {
        Some(path) => path,
        None => find_manifest()?,
    };
    let graph = PackageGraph::resolve(Manifest::load(&manifest_path)?)?;

    let (action, options) = match cli.command {
        CliCommand::Build(flags) => (
            Action::Build,
            CompileOptions {
                machine: flags.machine,
                host: flags.host,
                program_args: Vec::new(),
            },
        ),
        CliCommand::Run {
            flags,
            program_args,
        } => (
            Action::Run,
            CompileOptions {
                machine: flags.machine,
                host: flags.host,
                program_args,
            },
        ),
    };

    let args = compiler::compiler_args(&graph, action, &options)?;
    if action == Action::Build {
        let output_dir = compiler::output_dir(&graph);
        fs::create_dir_all(&output_dir)
            .with_context(|| format!("cannot create `{}`", output_dir.display()))?;
    }

    let compiler_path = compiler::locate_compiler();
    let status = Command::new(&compiler_path)
        .args(&args)
        .status()
        .with_context(|| {
            format!(
                "cannot run the compiler `{}`. Install Vx, or set VXC to the path of `vxc`",
                compiler_path.display()
            )
        })?;
    match status.code() {
        Some(code) => Ok(code),
        None => bail!("the compiler was stopped by a signal: {status}"),
    }
}

/// The `vex.toml` in the current directory, or in the nearest directory above it.
///
/// # Errors
///
/// If the current directory cannot be read, or no `vex.toml` is found.
fn find_manifest() -> Result<PathBuf> {
    let start = env::current_dir().context("cannot read the current directory")?;
    for dir in start.ancestors() {
        let candidate = dir.join(MANIFEST_FILE_NAME);
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    bail!(
        "no `{MANIFEST_FILE_NAME}` in `{}` or in any directory above it",
        start.display()
    )
}
