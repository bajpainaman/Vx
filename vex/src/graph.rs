//===- graph.rs - Vx Package Tool ------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Follows `[dependencies]` from a package to every package it needs, directly or through
// another package.
//
//===----------------------------------------------------------------------===//
use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};

use crate::manifest::{Manifest, MANIFEST_FILE_NAME};

/// A package and everything it needs to compile.
#[derive(Debug)]
pub struct PackageGraph {
    pub root: Manifest,
    /// The directory of every package in the build, the root included, by package name.
    ///
    /// One name maps to one directory because the compiler's `--package name=dir` does, so a build
    /// holds one copy of each package.
    pub packages: BTreeMap<String, PathBuf>,
}

impl PackageGraph {
    /// Find every package `root` needs.
    ///
    /// # Errors
    ///
    /// If a dependency's `vex.toml` cannot be read, two packages share a name, or packages
    /// depend on each other in a circle.
    pub fn resolve(root: Manifest) -> Result<Self> {
        let mut packages = BTreeMap::new();
        packages.insert(root.name.clone(), root.dir.clone());
        let mut chain = vec![root.name.clone()];
        visit(&root, &mut chain, &mut packages)?;
        Ok(Self { root, packages })
    }
}

/// `chain` is the list of packages from the root down to `manifest`, used to report a cycle.
///
/// # Errors
///
/// The same as [`PackageGraph::resolve`].
fn visit(
    manifest: &Manifest,
    chain: &mut Vec<String>,
    packages: &mut BTreeMap<String, PathBuf>,
) -> Result<()> {
    for (name, dir) in &manifest.dependencies {
        if let Some(known_dir) = packages.get(name) {
            if known_dir != dir {
                bail!(
                    "two different packages are both called `{name}`: `{}` and `{}`. A build can \
                     hold only one package of each name",
                    known_dir.display(),
                    dir.display()
                );
            }
            if let Some(start) = chain.iter().position(|seen| seen == name) {
                bail!(
                    "packages depend on each other in a circle: {} -> {name}",
                    chain[start..].join(" -> ")
                );
            }
            continue;
        }

        let dependency = Manifest::load(&dir.join(MANIFEST_FILE_NAME))
            .with_context(|| format!("while reading dependency `{name}` of `{}`", manifest.name))?;
        if dependency.name != *name {
            bail!(
                "`{}` depends on `{name}`, but the package in `{}` is named `{}`. The name in \
                 `[dependencies]` must be the package's own name, because that is the name `import` \
                 uses",
                manifest.name,
                dir.display(),
                dependency.name
            );
        }

        packages.insert(name.clone(), dir.clone());
        chain.push(name.clone());
        visit(&dependency, chain, packages)?;
        chain.pop();
    }
    Ok(())
}

#[cfg(test)]
#[allow(
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    reason = "a test reports a failure by panicking, and its helpers pass errors on"
)]
mod tests {
    use super::*;
    use std::fmt::Write as _;
    use std::fs;
    use std::path::Path;

    /// Write `<root>/<name>/vex.toml` for a package that depends on `dependencies`, each given as
    /// (name in `[dependencies]`, directory next to this package).
    fn write_package(
        root: &Path,
        dir_name: &str,
        package_name: &str,
        dependencies: &[(&str, &str)],
    ) {
        let dir = root.join(dir_name);
        fs::create_dir_all(&dir).expect("create the package directory");
        let mut text = format!(
            "[package]\nname = \"{package_name}\"\nversion = \"1.0.0\"\n\n[dependencies]\n"
        );
        for (name, sibling_dir) in dependencies {
            writeln!(text, "{name} = {{ path = \"../{sibling_dir}\" }}")
                .expect("writing to a String cannot fail");
        }
        fs::write(dir.join(MANIFEST_FILE_NAME), text).expect("write vex.toml");
    }

    fn resolve_package(root: &Path, dir_name: &str) -> Result<PackageGraph> {
        PackageGraph::resolve(Manifest::load(
            &root.join(dir_name).join(MANIFEST_FILE_NAME),
        )?)
    }

    fn error_of(root: &Path, dir_name: &str) -> String {
        format!(
            "{:#}",
            resolve_package(root, dir_name).expect_err("this build should be refused")
        )
    }

    #[test]
    fn a_dependency_of_a_dependency_is_part_of_the_build() {
        let root = tempfile::tempdir().expect("create a temporary directory");
        write_package(root.path(), "app", "app", &[("shapes", "shapes")]);
        write_package(root.path(), "shapes", "shapes", &[("numbers", "numbers")]);
        write_package(root.path(), "numbers", "numbers", &[]);

        let graph = resolve_package(root.path(), "app").expect("three packages in a line");
        let names: Vec<&str> = graph.packages.keys().map(String::as_str).collect();
        assert_eq!(names, ["app", "numbers", "shapes"]);
        assert_eq!(
            graph.packages["numbers"],
            root.path()
                .join("numbers")
                .canonicalize()
                .expect("resolve numbers")
        );
    }

    #[test]
    fn two_packages_may_share_a_dependency() {
        let root = tempfile::tempdir().expect("create a temporary directory");
        write_package(
            root.path(),
            "app",
            "app",
            &[("shapes", "shapes"), ("text", "text")],
        );
        write_package(root.path(), "shapes", "shapes", &[("numbers", "numbers")]);
        write_package(root.path(), "text", "text", &[("numbers", "numbers")]);
        write_package(root.path(), "numbers", "numbers", &[]);

        let graph = resolve_package(root.path(), "app").expect("a shared dependency");
        assert_eq!(graph.packages.len(), 4);
    }

    #[test]
    fn two_packages_with_one_name_are_refused() {
        let root = tempfile::tempdir().expect("create a temporary directory");
        write_package(
            root.path(),
            "app",
            "app",
            &[("shapes", "shapes"), ("numbers", "numbers_a")],
        );
        write_package(root.path(), "shapes", "shapes", &[("numbers", "numbers_b")]);
        write_package(root.path(), "numbers_a", "numbers", &[]);
        write_package(root.path(), "numbers_b", "numbers", &[]);

        let error = error_of(root.path(), "app");
        assert!(
            error.contains("two different packages are both called `numbers`"),
            "{error}"
        );
    }

    #[test]
    fn packages_that_depend_on_each_other_are_refused() {
        let root = tempfile::tempdir().expect("create a temporary directory");
        write_package(root.path(), "app", "app", &[("shapes", "shapes")]);
        write_package(root.path(), "shapes", "shapes", &[("numbers", "numbers")]);
        write_package(root.path(), "numbers", "numbers", &[("shapes", "shapes")]);

        let error = error_of(root.path(), "app");
        assert!(
            error.contains("in a circle: shapes -> numbers -> shapes"),
            "{error}"
        );
    }

    #[test]
    fn a_dependency_must_be_listed_under_its_own_name() {
        let root = tempfile::tempdir().expect("create a temporary directory");
        write_package(root.path(), "app", "app", &[("geometry", "shapes")]);
        write_package(root.path(), "shapes", "shapes", &[]);

        let error = error_of(root.path(), "app");
        assert!(error.contains("is named `shapes`"), "{error}");
    }

    #[test]
    fn a_dependency_without_a_vx_toml_is_refused() {
        let root = tempfile::tempdir().expect("create a temporary directory");
        write_package(root.path(), "app", "app", &[("shapes", "shapes")]);
        fs::create_dir_all(root.path().join("shapes")).expect("create shapes");

        let error = error_of(root.path(), "app");
        assert!(
            error.contains("while reading dependency `shapes` of `app`"),
            "{error}"
        );
    }
}
