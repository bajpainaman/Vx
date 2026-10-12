//===- build.rs - Vx Compiler ----------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//! Vx Build Script (`build.rs`)
//!
//! This script is automatically executed by Cargo before compiling the `vxc` compiler.
//!
//! # Why does Vx need this?
//! Vx supports two execution modes: JIT Execution and AOT (Ahead-of-Time) Compilation.
//!
//! 1. **JIT Execution**: Managed dynamically by `src/jit.rs`, which shells out to `clang++` at
//!    runtime to build `.dylib` files for `lli`.
//! 2. **AOT Compilation**: If a user uses `vxc` to compile their Vx code into a standalone
//!    executable binary, the linker needs a static version of the Objective-C++ hardware dispatcher.
//!
//! This script ensures that `libnpu_dispatch.a` is pre-compiled into Cargo's `OUT_DIR` so that
//! the standalone AOT linker can statically bundle the Apple Accelerate AMX hardware dispatcher
//! directly into the final application.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

// Square f16 matmul sizes to build Neural Engine primitives for.
// Adding one is a change here and nowhere else: the dispatcher looks the
// model up by shape and falls back when there is none (Vx#173).
const VX_ANE_DIMS: &str = "512,1024";

/// The C++ compiler, the archiver and the LLVM compile flags, shared by the plugin loader and
/// the dialect.
struct Toolchain {
    cxx: String,
    ar: String,
    arflags: Vec<String>,
    llvm_cxxflags: Vec<String>,
}

impl Toolchain {
    fn from_env() -> Self {
        let arflags = env::var("ARFLAGS").unwrap_or_else(|_| "rcs".to_string());
        Self {
            cxx: env::var("CXX").unwrap_or_else(|_| "clang++".to_string()),
            ar: env::var("AR").unwrap_or_else(|_| "ar".to_string()),
            arflags: arflags.split_whitespace().map(str::to_string).collect(),
            llvm_cxxflags: llvm_config("--cxxflags")
                .split_whitespace()
                .map(str::to_string)
                .collect(),
        }
    }
}

/// What `llvm-config <flag>` prints, without the newline at the end.
///
/// # Panics
///
/// If `llvm-config` cannot be run.
fn llvm_config(flag: &str) -> String {
    let output = Command::new("llvm-config")
        .arg(flag)
        .output()
        .unwrap_or_else(|_| panic!("Failed to get llvm-config {flag}"));
    String::from_utf8_lossy(&output.stdout).trim().to_string()
}

/// Where a usable CUDA toolkit lives, as (root, library directory), or `None`
/// to build the portable host shim instead.
///
/// A GPU is not required here and deliberately not looked for: the build host
/// and the run host are different machines in this project's workflow, and the
/// backend decides at run time whether a device is present (see
/// `cuda_available()` in `runtime/cuda_dispatch.cpp`). What must be present to
/// build is the toolkit -- headers plus libcudart and libcublas.
///
/// Set `VX_DISABLE_CUDA` to build the host shim on a machine that has CUDA.
fn cuda_root() -> Option<(PathBuf, PathBuf)> {
    if env::var_os("VX_DISABLE_CUDA").is_some() {
        return None;
    }

    let mut candidates: Vec<PathBuf> = Vec::new();
    for var in ["CUDA_HOME", "CUDA_PATH"] {
        if let Some(dir) = env::var_os(var) {
            candidates.push(PathBuf::from(dir));
        }
    }
    candidates.push(PathBuf::from("/usr/local/cuda"));
    candidates.push(PathBuf::from("/usr"));

    for root in candidates {
        if !root.join("include/cuda_runtime.h").exists() {
            continue;
        }
        for lib in ["lib64", "lib/x86_64-linux-gnu", "lib"] {
            let libdir = root.join(lib);
            if libdir.join("libcudart.so").exists() {
                return Some((root, libdir));
            }
        }
    }

    None
}

fn main() {
    require_llvm_config();

    // The fleet headers, and the two the dispatch plan is built from. Every
    // backend compiles these -- npu_dispatch.mm as much as cuda_dispatch.cpp --
    // so they are listed here, above the platform split, rather than inside one
    // arm of it.
    //
    // They were listed only in the `else`. On macOS that made editing
    // vx_remote_client.h rebuild nothing: cargo reported "Finished" in a tenth
    // of a second, the JIT linked the previous dispatch library, and the change
    // appeared to have no effect. Which is an hour spent debugging a binary
    // that does not contain the fix, with no sign of it anywhere -- and it
    // happened twice, because the first fix was written into whichever branch
    // was open at the time (#348).
    println!("cargo:rerun-if-changed=runtime/vx_mlir_shims.c");

    let out_dir = env::var("OUT_DIR").unwrap();
    build_mlir_shims(&out_dir);

    println!("cargo:rerun-if-changed=runtime/vx_remote_routing.h");
    println!("cargo:rerun-if-changed=runtime/vx_remote_client.h");
    println!("cargo:rerun-if-changed=runtime/vx_remote_region.h");
    println!("cargo:rerun-if-changed=runtime/vx_manifest.h");
    println!("cargo:rerun-if-changed=runtime/vx_transport.h");
    println!("cargo:rerun-if-changed=runtime/vx_wire.h");
    println!("cargo:rerun-if-changed=runtime/vx_agent.h");
    println!("cargo:rerun-if-changed=runtime/vx_device_pool.h");
    println!("cargo:rerun-if-changed=runtime/vx_dispatch_plan.h");
    println!("cargo:rerun-if-changed=runtime/vx_host_call.h");
    println!("cargo:rerun-if-changed=include/vx_hardware_runtime.h");

    if cfg!(target_os = "macos") {
        build_apple_dispatch(&out_dir);
    } else {
        build_host_dispatch(&out_dir);
    }

    let tools = Toolchain::from_env();
    build_plugin_loader(&tools, &out_dir);
    build_vx_dialect(&tools, &out_dir);
}

/// Stop the build unless an `llvm-config` is on `PATH`: the `melior` and `tblgen` dependencies
/// require it.
///
/// # Panics
///
/// If no `llvm-config` is found.
fn require_llvm_config() {
    let llvm_configs = [
        "llvm-config-22",
        "llvm-config-21",
        "llvm-config-20",
        "llvm-config-19",
        "llvm-config-18",
        "llvm-config-17",
        "llvm-config-16",
        "llvm-config-15",
        "llvm-config",
    ];
    let found = llvm_configs
        .iter()
        .any(|cfg| Command::new(cfg).arg("--version").output().is_ok());

    if !found {
        println!(
            "cargo:warning=⚠️  LLVM is not in PATH! `melior` and `tblgen` will fail to build."
        );
        println!("cargo:warning=On macOS with Homebrew, run: export PATH=\"/opt/homebrew/opt/llvm/bin:$PATH\"");
        println!("cargo:warning=On Ubuntu, run: sudo apt-get install llvm-22 llvm-22-dev");
        panic!("llvm-config not found in PATH. Make sure LLVM 15+ is installed.");
    }
}

/// The half-precision memref printers MLIR exports only in packed form. Built here rather than
/// into `vx_std_core`, which has no MLIR dependency and should not gain one; see the source.
///
/// # Panics
///
/// If `out_dir` is not valid UTF-8.
fn build_mlir_shims(out_dir: &str) {
    let shim_path = PathBuf::from(out_dir).join(format!(
        "{}vx_mlir_shims{}",
        env::consts::DLL_PREFIX,
        env::consts::DLL_SUFFIX
    ));
    let cc = env::var("CC").unwrap_or_else(|_| "clang".to_string());
    let mut cmd = Command::new(cc);
    cmd.args(["-shared", "-fPIC", "runtime/vx_mlir_shims.c"]);
    // The forwarded `_mlir_ciface_*` symbols resolve from MLIR's runner utils, which is loaded
    // alongside this. ELF shared objects allow that by default; Mach-O has to be told.
    if cfg!(target_os = "macos") {
        cmd.args(["-undefined", "dynamic_lookup"]);
        // Otherwise the library records this OUT_DIR as the path it lives at, and every
        // program linked against it looks for the build machine's directory.
        cmd.arg("-Wl,-install_name,@rpath/libvx_mlir_shims.dylib");
    }
    cmd.args(["-o", shim_path.to_str().unwrap()]);
    match cmd.status() {
        Ok(st) if st.success() => {
            println!("cargo:rustc-env=VX_MLIR_SHIMS_PATH={}", shim_path.display());
        }
        // Not fatal: only half-precision printing depends on it, and a compiler that cannot
        // build one shim should still build.
        _ => {
            println!("cargo:rustc-env=VX_MLIR_SHIMS_PATH=");
            println!(
                "cargo:warning=Could not build runtime/vx_mlir_shims.c; printing a bf16 or f16 \
                 tensor will fail to link."
            );
        }
    }
}

/// The Python that can build the Neural Engine models, if this machine has one.
fn find_ane_python() -> Option<String> {
    // The interpreter has to be one that can actually *build* a model, which
    // the first `python3` on PATH generally cannot: this checkout keeps
    // coremltools in its own venv. Picking by name rather than by capability
    // meant the models were never generated on a machine provisioned for
    // them, and the warning was the same one a machine with no coremltools at
    // all prints.
    //
    // The probe imports `libmilstoragepython`, not just `coremltools`.
    // coremltools is part pure Python and part compiled extension, and only
    // the pure half installs on an interpreter it ships no wheel for -- so
    // `import coremltools` succeeds on a Python too new for it and the model
    // build then dies much later with `RuntimeError: BlobWriter not loaded`.
    // Probing the half that is missing is what makes the fallback work.
    let python = env::var("VX_PYTHON")
        .ok()
        .into_iter()
        .chain(
            [
                // The checkout keeps a coremltools that can build a model in
                // `venv-ane`; without this a machine provisioned for the ANE
                // still reports that no interpreter can build the models.
                "venv-ane/bin/python3",
                "venv/bin/python3",
                "python3",
                "python3.13",
                "python3.12",
                "python3.11",
            ]
            .into_iter()
            .map(str::to_string),
        )
        .find(|p| {
            Command::new(p)
                .args(["-c", "import coremltools, coremltools.libmilstoragepython"])
                .status()
                .is_ok_and(|s| s.success())
        });

    match &python {
        Some(p) => eprintln!("ANE models: generating with {p}"),
        None => println!(
            "cargo:warning=No interpreter can build the ANE models: none of $VX_PYTHON, \
             venv/bin/python3, python3, python3.13, python3.12 or python3.11 has a working \
             coremltools (the compiled half, libmilstoragepython, is what is checked). \
             Dispatch will use the CPU shim. Set VX_PYTHON to one that does."
        ),
    }
    python
}

/// Generate the Neural Engine primitive models and compile each one with `coremlc`.
fn build_ane_models(out_dir: &str) {
    eprintln!("Building ANE primitive models via CoreML...");

    let python = find_ane_python();
    let py_status = python.as_ref().map(|p| {
        Command::new(p)
            .args([
                "scripts/tools/generate_ane_primitives.py",
                "--out-dir",
                out_dir,
                "--dim",
                "4",
                "--ane-dims",
                VX_ANE_DIMS,
            ])
            .status()
    });

    match py_status {
        Some(Ok(status)) if status.success() => {
            // Compile the .mlpackage into .mlmodelc
            // The fp16 512x512 is the one the Neural Engine will actually take:
            // CoreML prefers the CPU for every fp32 matmul at every size, and
            // for fp16 below 512. The 4x4 pair stays because the dispatcher's
            // fp32 matmul and affine routes are written against it; no test
            // covers the affine route.
            for model_name in [
                "matmul_4x4",
                "affine_4",
                "matmul_512x512_fp16",
                "matmul_1024x1024_fp16",
            ] {
                compile_ane_model(model_name, out_dir);
            }
        }
        Some(Ok(_)) => println!("cargo:warning=Python script failed to generate ANE models."),
        Some(Err(_)) => println!(
            "cargo:warning=Failed to invoke the ANE model generator; \
             dispatch will use the CPU shim"
        ),
        None => {}
    }
}

/// Compile one generated `.mlpackage` and copy the result into `ane-primitives/`.
///
/// # Panics
///
/// If the current directory cannot be read, or a path is not valid UTF-8.
fn compile_ane_model(model_name: &str, out_dir: &str) {
    let pkg_path = PathBuf::from(out_dir).join(format!("{model_name}.mlpackage"));
    let modelc_path = PathBuf::from(out_dir).join(format!("{model_name}.mlmodelc"));

    // coremlc compile <pkg> <out_dir>
    //
    // `coremlc` ships with Xcode and not with the Command Line
    // Tools, so `xcrun` cannot find it when `xcode-select` points
    // at the CLT -- which is the default on a machine that has
    // both. Pointing DEVELOPER_DIR at Xcode for this one call
    // fixes that without `sudo xcode-select -s`, which is not a
    // build script's business.
    let mut coremlc = Command::new("xcrun");
    let xcrun_finds_coremlc = Command::new("xcrun")
        .args(["--find", "coremlc"])
        .output()
        .is_ok_and(|o| o.status.success());
    if !xcrun_finds_coremlc {
        let xcode = "/Applications/Xcode.app/Contents/Developer";
        if PathBuf::from(xcode).exists() {
            coremlc.env("DEVELOPER_DIR", xcode);
        }
    }
    let coremlc_status = coremlc
        .args(["coremlc", "compile", pkg_path.to_str().unwrap(), out_dir])
        .status();

    let Ok(compiled) = coremlc_status else {
        return;
    };
    if !compiled.success() {
        println!("cargo:warning=Failed to compile {model_name} with coremlc");
        return;
    }

    // One directory for every primitive, rather than five
    // bundles loose in the checkout root. The dispatcher
    // looks here; see vx_find_ane_model in
    // runtime/npu_dispatch.mm for the full search order.
    let root_dir = env::current_dir().unwrap();
    let dest_dir = root_dir.join("ane-primitives");
    let _ = std::fs::create_dir_all(&dest_dir);
    let target_modelc = dest_dir.join(format!("{model_name}.mlmodelc"));
    // Delete old one if exists
    let _ = std::fs::remove_dir_all(&target_modelc);
    // A copy left in the root by an older build would
    // still be found by the search path's last candidate,
    // so a stale bundle could answer for a fresh one.
    let _ = std::fs::remove_dir_all(root_dir.join(format!("{model_name}.mlmodelc")));
    let cp_status = Command::new("cp")
        .args([
            "-R",
            modelc_path.to_str().unwrap(),
            target_modelc.to_str().unwrap(),
        ])
        .status();

    if cp_status.is_ok_and(|s| s.success()) {
        eprintln!("Successfully compiled {model_name}.mlmodelc");
    }
}

/// The macOS dispatch backend: the Neural Engine models, then `runtime/npu_dispatch.mm` as a
/// static archive for AOT and as a shared library for the JIT.
fn build_apple_dispatch(out_dir: &str) {
    println!("cargo:rerun-if-changed=runtime/npu_dispatch.mm");
    println!("cargo:rerun-if-changed=runtime/npu_dispatch.h");
    println!("cargo:rerun-if-changed=scripts/tools/generate_ane_primitives.py");

    build_ane_models(out_dir);

    // Determine compiler and flags
    let cxx = env::var("CXX").unwrap_or_else(|_| "clang++".to_string());
    let cxxflags_env =
        env::var("CXXFLAGS").unwrap_or_else(|_| "-O3 -Wno-deprecated-declarations".to_string());
    let mut cxxflags: Vec<&str> = cxxflags_env.split_whitespace().collect();
    // Fall back to running the outlined kernel on the CPU (via libffi) when the ANE/CoreML
    // path is unavailable -- no accelerator, or the CoreML models were not built (no
    // coremltools/coremlc). Without this the dispatcher aborts on any unhandled kernel.
    cxxflags.push("-DVX_ENABLE_CPU_FALLBACK");

    build_npu_static_archive(out_dir, &cxx, &cxxflags);
    let lib_shared_path = build_npu_shared_library(out_dir, &cxx, &cxxflags);

    // Pass the path to the shared library to the Rust compiler
    println!(
        "cargo:rustc-env=NPU_SHARED_LIB_PATH={}",
        lib_shared_path.display()
    );

    // Tell cargo to link against the generated static library
    println!("cargo:rustc-link-search=native={out_dir}");
    println!("cargo:rustc-link-lib=static=npu_dispatch");

    // Link required Apple frameworks and C++ standard library
    println!("cargo:rustc-link-lib=dylib=c++");
    // libffi: used by vx_plugin_dispatch_async to call JIT kernels per the C ABI
    println!("cargo:rustc-link-lib=dylib=ffi");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Accelerate");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
    println!("cargo:rustc-link-lib=framework=CoreML");
}

/// Compile `runtime/npu_dispatch.mm` into `libnpu_dispatch.a`, which AOT programs link.
///
/// # Panics
///
/// If the C++ compiler or the archiver cannot be run, or either one fails.
fn build_npu_static_archive(out_dir: &str, cxx: &str, cxxflags: &[&str]) {
    let obj_path = PathBuf::from(out_dir).join("npu_dispatch.o");
    let lib_path = PathBuf::from(out_dir).join("libnpu_dispatch.a");

    // Compile the Objective-C++ runtime file for AOT (Static Archive)
    let mut clang_cmd = Command::new(cxx);
    clang_cmd.args([
        "-c",
        "runtime/npu_dispatch.mm",
        "-o",
        obj_path.to_str().unwrap(),
        "-fobjc-arc",
    ]);
    clang_cmd.args(cxxflags);

    let status = clang_cmd
        .status()
        .unwrap_or_else(|_| panic!("Failed to execute {cxx}"));

    assert!(status.success(), "clang++ compilation failed");

    // Determine archiver and flags
    let ar = env::var("AR").unwrap_or_else(|_| "ar".to_string());
    let arflags_env = env::var("ARFLAGS").unwrap_or_else(|_| "rcs".to_string());
    let arflags: Vec<&str> = arflags_env.split_whitespace().collect();

    // Create the static archive
    let mut ar_cmd = Command::new(&ar);
    ar_cmd.args(&arflags);
    ar_cmd.args([lib_path.to_str().unwrap(), obj_path.to_str().unwrap()]);

    let status = ar_cmd
        .status()
        .unwrap_or_else(|_| panic!("Failed to execute {ar}"));

    assert!(status.success(), "{ar} archiving failed");
}

/// Compile `runtime/npu_dispatch.mm` into the shared library the JIT loads, and return its path.
///
/// # Panics
///
/// If the C++ compiler cannot be run, or fails.
fn build_npu_shared_library(out_dir: &str, cxx: &str, cxxflags: &[&str]) -> PathBuf {
    // Named like the other platforms' backends, so the release packaging finds it.
    let lib_shared_path = PathBuf::from(out_dir).join("libvx_npu_dispatch.dylib");
    let mut clang_shared_cmd = Command::new(cxx);
    clang_shared_cmd.args([
        "-shared",
        "-fPIC",
        "-fobjc-arc",
        "runtime/npu_dispatch.mm",
        "-framework",
        "Accelerate",
        "-framework",
        "Foundation",
        "-framework",
        "Metal",
        "-framework",
        "MetalPerformanceShaders",
        "-framework",
        "CoreML",
        "-lffi",
        // Otherwise every program linked against it looks for it in this OUT_DIR.
        "-Wl,-install_name,@rpath/libvx_npu_dispatch.dylib",
        "-o",
        lib_shared_path.to_str().unwrap(),
    ]);
    clang_shared_cmd.args(cxxflags);

    let status = clang_shared_cmd
        .status()
        .unwrap_or_else(|_| panic!("Failed to execute clang++ for shared library"));

    assert!(
        status.success(),
        "clang++ compilation failed for shared library"
    );
    lib_shared_path
}

/// Add the flags that build the dispatch backend against the CUDA toolkit in `root`.
fn add_cuda_flags(clang_shared_cmd: &mut Command, root: &Path, libdir: &Path) {
    // A program's own `extern` may name a cuBLAS or CUDA runtime entry point, so the
    // native link needs the toolkit's library directory too (src/jit.rs).
    println!("cargo:rustc-env=VX_CUDA_LIBDIR={}", libdir.display());
    // The rpath matters because this library is loaded by the programs
    // vxc links, not by vxc itself: without it a program would have to
    // be run with the toolkit on LD_LIBRARY_PATH.
    clang_shared_cmd.arg(format!("-I{}/include", root.display()));
    clang_shared_cmd.arg(format!("-L{}", libdir.display()));
    clang_shared_cmd.arg(format!("-Wl,-rpath,{}", libdir.display()));
    // `-lcuda` is the driver API, for loading a device image the
    // compiler emitted (#251). It is linked from the toolkit's stubs
    // and resolved at run time against the real driver -- which is why
    // the stub directory is on -L and deliberately not on the rpath: an
    // rpath to it would find the stub first, and the stub's entry
    // points do nothing.
    clang_shared_cmd.arg(format!("-L{}/stubs", libdir.display()));
    clang_shared_cmd.args(["-lcudart", "-lcublas", "-lcuda"]);
}

/// The dispatch backend for every platform that is not macOS.
///
/// # Panics
///
/// If the C++ compiler cannot be run, or fails.
fn build_host_dispatch(out_dir: &str) {
    // A program containing a non-CPU `spawn on` needs a provider for the
    // vx_plugin_* ABI or it will not link, so one is always built. Which
    // one depends on what the machine has: the CUDA backend where a toolkit
    // is installed (runtime/cuda_dispatch.cpp), and otherwise the portable
    // shim that runs outlined kernels on the CPU through libffi
    // (runtime/host_dispatch.cpp). Both fall back to the host for anything
    // they cannot route, so the choice affects speed, not results.
    println!("cargo:rerun-if-changed=runtime/host_dispatch.cpp");
    println!("cargo:rerun-if-changed=runtime/host_dispatch_common.h");
    println!("cargo:rerun-if-changed=runtime/x86_dispatch.cpp");
    println!("cargo:rerun-if-changed=runtime/arm64_dispatch.cpp");
    println!("cargo:rerun-if-changed=runtime/cuda_dispatch.cpp");

    // Which backend answers the plugin ABI. CUDA wins where a toolkit is
    // installed, since it is the only one of these that can reach an
    // accelerator; otherwise the target's own CPU backend, and the portable
    // one for an architecture nobody has looked at yet.
    //
    // Every target gets a backend, which is what lets the compiler emit
    // allocate/transfer/free unconditionally and never name a vendor.
    let cuda = cuda_root();
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let (source, lib_stem) = match (&cuda, arch.as_str()) {
        (Some(_), _) => ("runtime/cuda_dispatch.cpp", "vx_cuda_dispatch"),
        (None, "x86_64") => ("runtime/x86_dispatch.cpp", "vx_x86_dispatch"),
        (None, "aarch64") => ("runtime/arm64_dispatch.cpp", "vx_arm64_dispatch"),
        (None, _) => ("runtime/host_dispatch.cpp", "vx_host_dispatch"),
    };

    let lib_shared_path = PathBuf::from(out_dir).join(format!(
        "{}{}{}",
        env::consts::DLL_PREFIX,
        lib_stem,
        env::consts::DLL_SUFFIX
    ));

    let cxx = env::var("CXX").unwrap_or_else(|_| "clang++".to_string());
    let cxxflags_env = env::var("CXXFLAGS").unwrap_or_else(|_| "-O3".to_string());
    let cxxflags: Vec<&str> = cxxflags_env.split_whitespace().collect();

    let mut clang_shared_cmd = Command::new(&cxx);
    clang_shared_cmd.args([
        "-shared",
        "-fPIC",
        source,
        "-lffi",
        "-o",
        lib_shared_path.to_str().unwrap(),
    ]);
    clang_shared_cmd.args(&cxxflags);

    if let Some((root, libdir)) = &cuda {
        add_cuda_flags(&mut clang_shared_cmd, root, libdir);
    }

    let status = clang_shared_cmd
        .status()
        .unwrap_or_else(|_| panic!("Failed to execute {cxx} for {source}"));
    assert!(status.success(), "{cxx} failed to build {source}");

    println!(
        "cargo:rustc-env=NPU_SHARED_LIB_PATH={}",
        lib_shared_path.display()
    );
    match &cuda {
        Some((root, _)) => eprintln!(
            "Built the CUDA dispatch backend against {}; \
             recognised matmuls run on the GPU, everything else on the CPU.",
            root.display()
        ),
        None => {
            eprintln!("Built the {source} dispatch backend; kernels run on the CPU via libffi.");
        }
    }
}

/// Compile the MLIR pass plugin loader wrapper into `libplugin_loader.a`.
///
/// # Panics
///
/// If the C++ compiler or the archiver cannot be run, or either one fails.
fn build_plugin_loader(tools: &Toolchain, out_dir: &str) {
    println!("cargo:rerun-if-changed=src/plugin_loader.cpp");

    let plugin_obj_path = PathBuf::from(out_dir).join("plugin_loader.o");
    let plugin_lib_path = PathBuf::from(out_dir).join("libplugin_loader.a");

    let mut plugin_cmd = Command::new(&tools.cxx);
    plugin_cmd.args([
        "-c",
        "src/plugin_loader.cpp",
        "-o",
        plugin_obj_path.to_str().unwrap(),
        "-std=c++17",
    ]);
    plugin_cmd.args(&tools.llvm_cxxflags);

    let status = plugin_cmd
        .status()
        .unwrap_or_else(|_| panic!("Failed to execute {} for plugin_loader", tools.cxx));
    assert!(
        status.success(),
        "clang++ compilation failed for plugin_loader"
    );

    let mut ar_cmd = Command::new(&tools.ar);
    ar_cmd.args(&tools.arflags);
    ar_cmd.args([
        plugin_lib_path.to_str().unwrap(),
        plugin_obj_path.to_str().unwrap(),
    ]);
    let status = ar_cmd
        .status()
        .unwrap_or_else(|_| panic!("Failed to archive libplugin_loader.a"));
    assert!(status.success(), "ar failed for libplugin_loader");

    println!("cargo:rustc-link-search=native={out_dir}");
    println!("cargo:rustc-link-lib=static=plugin_loader");
}

/// Generate the Vx MLIR dialect's tables with `mlir-tblgen`, then compile the dialect into
/// `libvx_dialect.a`.
///
/// # Panics
///
/// If `mlir-tblgen` or the archiver cannot be run, or either one fails.
fn build_vx_dialect(tools: &Toolchain, out_dir: &str) {
    println!("cargo:rerun-if-changed=include/VxDialect.td");
    println!("cargo:rerun-if-changed=include/VxDialect.h");
    println!("cargo:rerun-if-changed=src/dialect/VxDialect.cpp");
    println!("cargo:rerun-if-changed=src/dialect/VxLowering.cpp");
    println!("cargo:rerun-if-changed=src/dialect/vx-opt.cpp");

    let tblgen = PathBuf::from(llvm_config("--bindir")).join("mlir-tblgen");
    let llvm_include = llvm_config("--includedir");

    // Each row is the generator flag, the file it writes, and its name in an error message.
    for (generator, output, what) in [
        ("-gen-dialect-decls", "VxDialect.h.inc", "dialect decls"),
        ("-gen-dialect-defs", "VxDialect.cpp.inc", "dialect defs"),
        ("-gen-op-decls", "VxOps.h.inc", "op decls"),
        ("-gen-op-defs", "VxOps.cpp.inc", "op defs"),
    ] {
        let status = Command::new(&tblgen)
            .args([
                generator,
                "-I",
                &llvm_include,
                "include/VxDialect.td",
                "-o",
                &format!("{out_dir}/{output}"),
            ])
            .status()
            .unwrap_or_else(|_| panic!("Failed to run mlir-tblgen for {what}"));
        assert!(status.success(), "mlir-tblgen failed");
    }

    let dialect_obj_path =
        compile_dialect_source(tools, out_dir, "src/dialect/VxDialect.cpp", "VxDialect");
    let lowering_obj_path =
        compile_dialect_source(tools, out_dir, "src/dialect/VxLowering.cpp", "VxLowering");
    let vx_opt_obj_path =
        compile_dialect_source(tools, out_dir, "src/dialect/vx-opt.cpp", "vx-opt");
    let dialect_lib_path = PathBuf::from(out_dir).join("libvx_dialect.a");

    let mut ar_cmd = Command::new(&tools.ar);
    ar_cmd.args(&tools.arflags);
    ar_cmd.args([
        dialect_lib_path.to_str().unwrap(),
        dialect_obj_path.to_str().unwrap(),
        lowering_obj_path.to_str().unwrap(),
        vx_opt_obj_path.to_str().unwrap(),
    ]);
    let status = ar_cmd.status().expect("Failed to archive libvx_dialect.a");
    assert!(status.success(), "ar failed for libvx_dialect");

    println!("cargo:rustc-link-lib=static=vx_dialect");
}

/// Compile one C++ file of the dialect to `<out_dir>/<name>.o` and return that path.
///
/// # Panics
///
/// If the C++ compiler cannot be run, or fails.
fn compile_dialect_source(tools: &Toolchain, out_dir: &str, source: &str, name: &str) -> PathBuf {
    let object_path = Path::new(out_dir).join(format!("{name}.o"));
    let mut cmd = Command::new(&tools.cxx);
    cmd.args([
        "-c",
        source,
        "-o",
        object_path.to_str().unwrap(),
        "-std=c++17",
        &format!("-I{out_dir}"), // to find the generated .inc files
        "-Iinclude",             // to find VxDialect.h
    ]);
    cmd.args(&tools.llvm_cxxflags);
    let status = cmd
        .status()
        .unwrap_or_else(|_| panic!("Failed to execute cxx for {name}"));
    assert!(status.success(), "clang++ compilation failed for {name}");
    object_path
}
