//===- manifest.rs - Vx Package Tool ---------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Reads and checks `vex.toml`, the file that names a package and the packages it depends on.
//
//===----------------------------------------------------------------------===//
use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::Deserialize;

pub const MANIFEST_FILE_NAME: &str = "vex.toml";

/// The file `vex run` compiles when `[package]` does not name one.
const DEFAULT_MAIN_FILE: &str = "main.vx";

/// `import std::..` and `import core::..` always mean the standard library, so a package with
/// one of these names could never be imported.
const RESERVED_PACKAGE_NAMES: [&str; 2] = ["std", "core"];

/// A version is written `MAJOR.MINOR.PATCH`.
const VERSION_PART_COUNT: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Version {
    pub major: u64,
    pub minor: u64,
    pub patch: u64,
}

impl Version {
    /// Read a version written as `MAJOR.MINOR.PATCH`.
    ///
    /// # Errors
    ///
    /// If `text` is not three numbers with dots between them.
    fn parse(text: &str) -> Result<Self> {
        let parts: Vec<&str> = text.split('.').collect();
        if parts.len() != VERSION_PART_COUNT {
            bail!("version `{text}` must be three numbers, like `1.2.3`");
        }
        let number = |part: &str| -> Result<u64> {
            // `u64::from_str` accepts a leading `+`, which is not part of a version.
            if part.is_empty() || !part.chars().all(|c| c.is_ascii_digit()) {
                bail!("version `{text}` must be three numbers, like `1.2.3`");
            }
            part.parse::<u64>()
                .with_context(|| format!("version `{text}` has a number that is too large"))
        };
        Ok(Self {
            major: number(parts[0])?,
            minor: number(parts[1])?,
            patch: number(parts[2])?,
        })
    }
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// A checked `vex.toml`. Every path in it is absolute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    pub name: String,
    pub version: Version,
    /// The directory that holds `vex.toml`. `import name::a::b` reads `dir/a/b.vx`.
    pub dir: PathBuf,
    /// The file with `fn main`. A package without one is a library.
    pub main: Option<PathBuf>,
    /// The directory of each package this one depends on, by package name.
    pub dependencies: BTreeMap<String, PathBuf>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawManifest {
    package: RawPackage,
    #[serde(default)]
    dependencies: BTreeMap<String, toml::Value>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPackage {
    name: String,
    version: String,
    main: Option<PathBuf>,
}

impl Manifest {
    /// Read the `vex.toml` at `manifest_path`.
    ///
    /// # Errors
    ///
    /// If the file cannot be read, or [`Manifest::parse`] refuses it.
    pub fn load(manifest_path: &Path) -> Result<Self> {
        let text = fs::read_to_string(manifest_path)
            .with_context(|| format!("cannot read `{}`", manifest_path.display()))?;
        let manifest_path = manifest_path
            .canonicalize()
            .with_context(|| format!("cannot resolve `{}`", manifest_path.display()))?;
        let dir = manifest_path
            .parent()
            .with_context(|| format!("`{}` is not inside a directory", manifest_path.display()))?;
        Self::parse(&text, dir).with_context(|| format!("in `{}`", manifest_path.display()))
    }

    /// Check the text of a `vex.toml` that lives in `dir`.
    ///
    /// # Errors
    ///
    /// If the text is not valid TOML, has a key `vex` does not know, or names a package, version,
    /// main file or dependency that is not usable.
    fn parse(text: &str, dir: &Path) -> Result<Self> {
        let raw: RawManifest = toml::from_str(text)?;
        check_package_name(&raw.package.name)?;
        let version = Version::parse(&raw.package.version)?;

        let main = match raw.package.main {
            Some(file) => {
                let path = dir.join(&file);
                if !path.is_file() {
                    bail!(
                        "`main = \"{}\"` names a file that does not exist",
                        file.display()
                    );
                }
                Some(path)
            }
            None => Some(dir.join(DEFAULT_MAIN_FILE)).filter(|path| path.is_file()),
        };

        let mut dependencies = BTreeMap::new();
        for (name, value) in &raw.dependencies {
            check_package_name(name)?;
            if *name == raw.package.name {
                bail!("package `{name}` lists itself as a dependency");
            }
            dependencies.insert(name.clone(), dependency_dir(name, value, dir)?);
        }

        Ok(Self {
            name: raw.package.name,
            version,
            dir: dir.to_path_buf(),
            main,
            dependencies,
        })
    }
}

/// A package name is used in `import name::..`, so it has to be a Vx identifier.
///
/// # Errors
///
/// If `name` is not an identifier, or is a standard library name.
fn check_package_name(name: &str) -> Result<()> {
    let starts_like_an_identifier = name
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_');
    if !starts_like_an_identifier || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        bail!(
            "`{name}` is not a package name: use letters, digits and `_`, and do not start with a digit"
        );
    }
    if RESERVED_PACKAGE_NAMES.contains(&name) {
        bail!("`{name}` is part of the standard library and cannot be a package name");
    }
    Ok(())
}

/// The directory a `[dependencies]` entry points at. Only `{ path = ".." }` exists so far.
///
/// # Errors
///
/// If the entry is not `{ path = ".." }`, or the directory does not exist.
fn dependency_dir(name: &str, value: &toml::Value, package_dir: &Path) -> Result<PathBuf> {
    let path = match value {
        toml::Value::String(version) => bail!(
            "dependency `{name} = \"{version}\"` asks for a version from a registry, which vex does \
             not have yet; write `{name} = {{ path = \"..\" }}`"
        ),
        toml::Value::Table(table) => match (table.len(), table.get("path")) {
            (1, Some(toml::Value::String(path))) => path,
            _ => bail!(
                "dependency `{name}` must be written `{name} = {{ path = \"..\" }}`; no other kind \
                 of dependency exists yet"
            ),
        },
        _ => bail!("dependency `{name}` must be written `{name} = {{ path = \"..\" }}`"),
    };
    let dir = package_dir.join(path);
    dir.canonicalize()
        .with_context(|| format!("dependency `{name}`: `{}` does not exist", dir.display()))
}

#[cfg(test)]
#[allow(
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    reason = "a test reports a failure by panicking, and its helpers pass errors on"
)]
mod tests {
    use super::*;

    fn parse_in_empty_dir(text: &str) -> Result<Manifest> {
        let dir = tempfile::tempdir().expect("create a temporary directory");
        Manifest::parse(text, dir.path())
    }

    fn error_of(text: &str) -> String {
        format!(
            "{:#}",
            parse_in_empty_dir(text).expect_err("this vex.toml should be refused")
        )
    }

    #[test]
    fn a_package_with_a_name_and_a_version_is_accepted() {
        let manifest = parse_in_empty_dir("[package]\nname = \"shapes\"\nversion = \"1.2.3\"\n")
            .expect("a minimal vex.toml");
        assert_eq!(manifest.name, "shapes");
        assert_eq!(
            manifest.version,
            Version {
                major: 1,
                minor: 2,
                patch: 3
            }
        );
        assert_eq!(manifest.version.to_string(), "1.2.3");
        assert!(manifest.dependencies.is_empty());
    }

    #[test]
    fn a_package_without_main_vx_is_a_library() {
        let manifest = parse_in_empty_dir("[package]\nname = \"shapes\"\nversion = \"1.0.0\"\n")
            .expect("a minimal vex.toml");
        assert_eq!(manifest.main, None);
    }

    #[test]
    fn main_vx_is_found_without_being_named() {
        let dir = tempfile::tempdir().expect("create a temporary directory");
        fs::write(
            dir.path().join("main.vx"),
            "fn main() -> i32 { return 0; }\n",
        )
        .expect("write main.vx");
        let manifest = Manifest::parse(
            "[package]\nname = \"app\"\nversion = \"1.0.0\"\n",
            dir.path(),
        )
        .expect("a minimal vex.toml");
        assert_eq!(manifest.main, Some(dir.path().join("main.vx")));
    }

    #[test]
    fn a_named_main_file_must_exist() {
        let error =
            error_of("[package]\nname = \"app\"\nversion = \"1.0.0\"\nmain = \"start.vx\"\n");
        assert!(
            error.contains("names a file that does not exist"),
            "{error}"
        );
    }

    #[test]
    fn a_version_must_be_three_numbers() {
        for version in ["1.2", "1.2.3.4", "1.x.3", "1..3", "+1.2.3", ""] {
            let error = error_of(&format!(
                "[package]\nname = \"app\"\nversion = \"{version}\"\n"
            ));
            assert!(
                error.contains("must be three numbers"),
                "{version}: {error}"
            );
        }
    }

    #[test]
    fn a_package_name_must_be_an_identifier() {
        for name in ["my-package", "2fast", "a b", ""] {
            let error = error_of(&format!(
                "[package]\nname = \"{name}\"\nversion = \"1.0.0\"\n"
            ));
            assert!(error.contains("is not a package name"), "{name}: {error}");
        }
    }

    #[test]
    fn a_package_cannot_take_a_standard_library_name() {
        for name in ["std", "core"] {
            let error = error_of(&format!(
                "[package]\nname = \"{name}\"\nversion = \"1.0.0\"\n"
            ));
            assert!(
                error.contains("part of the standard library"),
                "{name}: {error}"
            );
        }
    }

    #[test]
    fn an_unknown_key_is_refused() {
        let error = error_of("[package]\nname = \"app\"\nversion = \"1.0.0\"\nauthor = \"me\"\n");
        assert!(error.contains("unknown field `author`"), "{error}");
    }

    #[test]
    fn a_path_dependency_becomes_an_absolute_directory() {
        let root = tempfile::tempdir().expect("create a temporary directory");
        let app = root.path().join("app");
        let geometry = root.path().join("geometry");
        fs::create_dir_all(&app).expect("create app");
        fs::create_dir_all(&geometry).expect("create geometry");
        let manifest = Manifest::parse(
            "[package]\nname = \"app\"\nversion = \"1.0.0\"\n\n[dependencies]\ngeometry = { path = \"../geometry\" }\n",
            &app,
        )
        .expect("a vex.toml with one dependency");
        assert_eq!(
            manifest.dependencies.get("geometry"),
            Some(&geometry.canonicalize().expect("resolve geometry"))
        );
    }

    #[test]
    fn a_dependency_directory_must_exist() {
        let error = error_of(
            "[package]\nname = \"app\"\nversion = \"1.0.0\"\n\n[dependencies]\ngeometry = { path = \"../nowhere\" }\n",
        );
        assert!(error.contains("dependency `geometry`"), "{error}");
        assert!(error.contains("does not exist"), "{error}");
    }

    #[test]
    fn a_version_dependency_is_refused_until_there_is_a_registry() {
        let error = error_of(
            "[package]\nname = \"app\"\nversion = \"1.0.0\"\n\n[dependencies]\ngeometry = \"1.0.0\"\n",
        );
        assert!(error.contains("does not have yet"), "{error}");
    }

    #[test]
    fn a_dependency_with_an_extra_key_is_refused() {
        let error = error_of(
            "[package]\nname = \"app\"\nversion = \"1.0.0\"\n\n[dependencies]\ngeometry = { path = \".\", rev = \"abc\" }\n",
        );
        assert!(
            error.contains("no other kind of dependency exists yet"),
            "{error}"
        );
    }

    #[test]
    fn a_package_cannot_depend_on_itself() {
        let error = error_of(
            "[package]\nname = \"app\"\nversion = \"1.0.0\"\n\n[dependencies]\napp = { path = \".\" }\n",
        );
        assert!(error.contains("lists itself as a dependency"), "{error}");
    }
}
