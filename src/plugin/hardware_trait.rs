//===- hardware_trait.rs - Vx Compiler -------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the core traits and interfaces for hardware accelerators.
// It establishes the standard contract that all external accelerator plugins must
// implement to interact with the Vx compiler's dispatch and memory management
// systems.
//
//===----------------------------------------------------------------------===//
use std::borrow::Cow;
use std::fmt;

// Mock MLIR types since Vx currently uses string-based codegen instead of a Rust MLIR crate.
/// Stand-ins for MLIR types, holding MLIR as text.
pub mod mlir {
    /// One MLIR operation.
    pub struct Operation {
        /// The operation written as MLIR text.
        pub text: String,
    }

    /// The list of passes that will run on a module.
    pub struct PassManager;

    /// One MLIR module.
    pub struct Module {
        /// The module written as MLIR text.
        pub text: String,
    }
}

/// The number that identifies one kind of hardware target.
pub type TopologyID = u32;

/// The order a plugin wants the dimensions of a tensor stored in.
pub enum TensorLayout {
    /// Batch, height, width, channels.
    NHWC,
    /// Batch, channels, height, width.
    NCHW,
    /// Any order, as long as the elements are stored without gaps.
    Contiguous,
}

/// Why a plugin could not produce code.
#[derive(Debug)]
pub enum PluginError {
    /// The plugin could not turn the module into its binary form; the text says why.
    LoweringFailed(String),
    /// The module has an operation the hardware cannot run; the text names it.
    UnsupportedOperation(String),
}

impl fmt::Display for PluginError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PluginError::LoweringFailed(msg) => write!(f, "Lowering failed: {msg}"),
            PluginError::UnsupportedOperation(msg) => write!(f, "Unsupported operation: {msg}"),
        }
    }
}

impl std::error::Error for PluginError {}

/// What a hardware accelerator plugin must provide so the compiler can generate code for it.
pub trait VxHardwarePlugin: Send + Sync {
    /// The vendor's identifier. Cow allows for static or dynamic strings.
    fn plugin_name(&self) -> Cow<'static, str>;

    /// The topology this plugin claims responsibility for
    fn target_topology(&self) -> TopologyID;

    /// Buffer Layout Constraints
    fn preferred_tensor_layout(&self) -> TensorLayout;
    /// The alignment, in bytes, the hardware needs for a tensor buffer.
    fn required_alignment(&self) -> usize;

    /// Verification Contract: Returns true if the hardware can execute this specific operation.
    /// Default implementation accepts all ops.
    fn is_op_supported(&self, _op: &mlir::Operation) -> bool {
        true
    }

    /// Compile-Time Escape Hatch: Allows the plugin to attach hardware-specific metadata
    /// or attributes to the operation before the lowering phase.
    fn annotate_operation(&self, _op: &mut mlir::Operation) {}

    /// The Pass Pipeline
    fn register_passes(&self, _pass_manager: &mut mlir::PassManager) {}

    /// Final Lowering
    /// Takes the optimized MLIR module and emits the final hardware-specific payload.
    ///
    /// # Errors
    ///
    /// Returns a [`PluginError`] when the plugin cannot turn the module into a binary.
    fn lower_to_binary(&self, module: mlir::Module) -> Result<Vec<u8>, PluginError>;
}
