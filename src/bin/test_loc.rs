//! A scratch program that checks MLIR accepts the debug-info location attributes the compiler
//! generates.
use anyhow::{Context as AnyhowContext, Result};
use melior::ir::operation::OperationLike;
use melior::ir::BlockLike;
use melior::Context;

/// Parses a small module with a debug-info location and prints that location.
///
/// # Errors
///
/// Returns an error when MLIR cannot parse the module, or the module has no operation.
fn main() -> Result<()> {
    let context = Context::new();
    melior::dialect::DialectHandle::llvm().register_dialect(&context);
    context.set_allow_unregistered_dialects(true);

    let di_file = r#"#llvm.di_file<"test.vx" in "">"#;
    let di_cu = format!(
        r#"#llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_C, file = {di_file}, producer = "vx", isOptimized = false, emissionKind = Full>"#
    );
    let di_subp = format!(
        r#"#llvm.di_subprogram<id = distinct[1]<>, compileUnit = {di_cu}, scope = {di_file}, name = "test_fn", file = {di_file}, subprogramFlags = Definition, type = #llvm.di_subroutine_type<>>"#
    );

    let mlir_str = format!(
        r#"
    #di_file = {di_file}
    #di_cu = {di_cu}
    #di_subp = {di_subp}
    module {{
        "dummy.op"() : () -> () loc(fused<#di_subp>["test.vx":1:1])
    }}
    "#
    );

    let module = melior::ir::Module::parse(&context, &mlir_str)
        .context("Failed to parse MLIR string. Check DI attribute formatting")?;
    let op = module
        .body()
        .first_operation()
        .context("Module has no operations")?;
    let loc = op.location();
    println!("Got loc: {loc}");
    Ok(())
}
