//===- compiler.rs - Vx Package Tool ---------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Finds the `vxc` compiler and works out the command line that compiles a package.
//
//===----------------------------------------------------------------------===//
use std::env;
use std::ffi::OsString;
use std::path::PathBuf;

use anyhow::{bail, Result};

use crate::graph::PackageGraph;

const COMPILER_NAME: &str = "vxc";

/// Set this to the path of a `vxc` to use it instead of the one `vex` would find.
const COMPILER_ENV_VAR: &str = "VXC";

/// Where `vex build` writes, inside the package directory.
const OUTPUT_DIR_NAME: &str = "target";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Compile the package to an object file.
    Build,
    /// Compile the package and run it.
    Run,
}

/// Compiler flags that `vex` hands on unchanged.
#[derive(Debug, Default)]
pub struct CompileOptions {
    pub machine: Option<PathBuf>,
    pub host: Option<String>,
    /// Arguments for the program itself. Only used by `Action::Run`.
    pub program_args: Vec<String>,
}

/// `$VXC`, else the `vxc` installed beside this program, else whichever `vxc` is on `PATH`.
pub fn locate_compiler() -> PathBuf {
    if let Some(path) = env::var_os(COMPILER_ENV_VAR) {
        return PathBuf::from(path);
    }
    let beside_this_program = env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|dir| dir.join(COMPILER_NAME)))
        .filter(|path| path.is_file());
    beside_this_program.unwrap_or_else(|| PathBuf::from(COMPILER_NAME))
}

/// The object file `vex build` writes for this package.
pub fn output_path(graph: &PackageGraph) -> PathBuf {
    output_dir(graph).join(format!("{}.o", graph.root.name))
}

/// The directory `vex build` writes into.
pub fn output_dir(graph: &PackageGraph) -> PathBuf {
    graph.root.dir.join(OUTPUT_DIR_NAME)
}

/// The arguments to give `vxc` so that it compiles the root package of `graph`.
///
/// # Errors
///
/// If the root package has no main file.
pub fn compiler_args(
    graph: &PackageGraph,
    action: Action,
    options: &CompileOptions,
) -> Result<Vec<OsString>> {
    let Some(main) = &graph.root.main else {
        bail!(
            "package `{}` has no `main.vx`, so there is nothing to build or run. Add one, or name \
             the file with `main = \"..\"` under `[package]`",
            graph.root.name
        );
    };

    let mut args: Vec<OsString> = Vec::new();
    // Every package in the build is named, not only the root's direct dependencies: the compiler
    // also has to find what those dependencies import.
    for (name, dir) in &graph.packages {
        let mut package = OsString::from(format!("{name}="));
        package.push(dir);
        args.push("--package".into());
        args.push(package);
    }
    if let Some(machine) = &options.machine {
        args.push("--machine".into());
        args.push(machine.into());
    }
    if let Some(host) = &options.host {
        args.push("--host".into());
        args.push(host.into());
    }
    match action {
        Action::Build => {
            args.push("-c".into());
            args.push(main.into());
            args.push("-o".into());
            args.push(output_path(graph).into());
        }
        Action::Run => {
            args.push("--run".into());
            args.push(main.into());
            if !options.program_args.is_empty() {
                args.push("--".into());
                args.extend(options.program_args.iter().map(OsString::from));
            }
        }
    }
    Ok(args)
}

#[cfg(test)]
#[allow(
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    reason = "a test reports a failure by panicking, and its helpers pass errors on"
)]
mod tests {
    use super::*;
    use crate::manifest::{Manifest, Version};
    use std::collections::BTreeMap;

    fn graph_of_app_and_shapes(main: Option<&str>) -> PackageGraph {
        let root = Manifest {
            name: "app".to_string(),
            version: Version {
                major: 1,
                minor: 0,
                patch: 0,
            },
            dir: PathBuf::from("/work/app"),
            main: main.map(PathBuf::from),
            dependencies: BTreeMap::from([("shapes".to_string(), PathBuf::from("/work/shapes"))]),
        };
        let packages = BTreeMap::from([
            ("app".to_string(), PathBuf::from("/work/app")),
            ("shapes".to_string(), PathBuf::from("/work/shapes")),
        ]);
        PackageGraph { root, packages }
    }

    fn args_as_text(graph: &PackageGraph, action: Action, options: &CompileOptions) -> Vec<String> {
        compiler_args(graph, action, options)
            .expect("a package with a main file")
            .into_iter()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect()
    }

    #[test]
    fn run_names_every_package_then_the_main_file() {
        let graph = graph_of_app_and_shapes(Some("/work/app/main.vx"));
        assert_eq!(
            args_as_text(&graph, Action::Run, &CompileOptions::default()),
            [
                "--package",
                "app=/work/app",
                "--package",
                "shapes=/work/shapes",
                "--run",
                "/work/app/main.vx"
            ]
        );
    }

    #[test]
    fn build_writes_an_object_file_named_after_the_package() {
        let graph = graph_of_app_and_shapes(Some("/work/app/main.vx"));
        let args = args_as_text(&graph, Action::Build, &CompileOptions::default());
        assert_eq!(
            args[4..],
            ["-c", "/work/app/main.vx", "-o", "/work/app/target/app.o"]
        );
    }

    #[test]
    fn machine_host_and_program_arguments_are_handed_on() {
        let graph = graph_of_app_and_shapes(Some("/work/app/main.vx"));
        let options = CompileOptions {
            machine: Some(PathBuf::from("fleet/h100-sxm.vx")),
            host: Some("default".to_string()),
            program_args: vec!["--steps".to_string(), "3".to_string()],
        };
        let args = args_as_text(&graph, Action::Run, &options);
        assert_eq!(
            args[4..],
            [
                "--machine",
                "fleet/h100-sxm.vx",
                "--host",
                "default",
                "--run",
                "/work/app/main.vx",
                "--",
                "--steps",
                "3"
            ]
        );
    }

    #[test]
    fn a_library_cannot_be_built_or_run() {
        let graph = graph_of_app_and_shapes(None);
        let error = compiler_args(&graph, Action::Run, &CompileOptions::default())
            .expect_err("a package with no main file");
        assert!(error.to_string().contains("has no `main.vx`"), "{error}");
    }
}
