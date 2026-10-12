//! Printing values and strings, and aborting with a message.
//!
//! One `impl` block on the per-function emitter; the dispatch that reaches these
//! lives in `step`, next to the struct.

use super::super::{
    memref_elem, mlir_scalar, Decline, ElementType, FnEmit, HirInstruction, Lowered,
};
// Writing into a `String` cannot fail, so the results of `writeln!` below are dropped.
use std::fmt::Write as _;

impl FnEmit<'_> {
    /// Print a value (no result). A tensor is `memref.cast`'d to an unranked memref and passed
    /// to the `printMemref*` runtime helper; a scalar goes to `print_*`. These are the same
    /// helpers the AST path calls; `emit_module_mlir` prepends their `private` declarations.
    ///
    /// # Errors
    ///
    /// Declines when the value has an element type no print helper takes, or when the emitter
    /// has no record of the value.
    pub(crate) fn op_print(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let arg = self
            .names
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        if let Some(memty) = self
            .mem_of
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone()
        {
            self.print_tensor(idx, ins, &arg, &memty)?;
        } else if let Some(e) = self.elem_at(ins.operand1.0) {
            self.print_scalar(idx, arg, &e)?;
        } else if self
            .ptr_of
            .get(ins.operand1.0 as usize)
            .copied()
            .unwrap_or(false)
        {
            // A pointer value is a C string: the `print_str` helper, as for a literal.
            let n = format!("%v{idx}");
            let _ = writeln!(
                self.body,
                "  {n} = func.call @print_str({arg}) : (!llvm.ptr) -> i32"
            );
        } else {
            return Err(crate::emitter_gap!());
        }
        Ok(())
    }

    /// Print the tensor `arg`, whose memref type is `memty`, with the `printMemref*` helper
    /// for its element type.
    ///
    /// # Errors
    ///
    /// Declines when no `printMemref*` helper takes the tensor's element type.
    fn print_tensor(
        &mut self,
        idx: usize,
        ins: &HirInstruction,
        arg: &str,
        memty: &str,
    ) -> Lowered<()> {
        let et = memref_elem(memty).ok_or(crate::emitter_gap!())?;
        // MLIR integers carry no sign, so the tensor's own element type says whether an
        // `i32` memref holds `u32`s.
        let unsigned = self
            .types
            .get(ins.type_idx.0 as usize)
            .and_then(|gid| self.ctx.tensors.get(gid))
            .is_some_and(|(elem, _)| matches!(elem, ElementType::U32 | ElementType::U64));
        let helper = match (et, unsigned) {
            ("f32", _) => "printMemrefF32",
            ("f64", _) => "printMemrefF64",
            ("i32", true) => "printMemrefU32",
            ("i64", true) => "printMemrefU64",
            ("i32", false) => "printMemrefI32",
            ("i64", false) => "printMemrefI64",
            ("bf16", _) => "printMemrefBF16",
            _ => return Err(crate::emitter_gap!()),
        };
        let c = format!("%pc{idx}");
        let _ = writeln!(
            self.body,
            "  {c} = memref.cast {arg} : {memty} to memref<*x{et}>"
        );
        let _ = writeln!(
            self.body,
            "  func.call @{helper}({c}) : (memref<*x{et}>) -> ()"
        );
        Ok(())
    }

    /// Print the scalar `arg`, whose element type is `e`, with the `print_*` helper for it.
    ///
    /// # Errors
    ///
    /// Declines when no `print_*` helper takes the element type.
    fn print_scalar(&mut self, idx: usize, arg: String, e: &ElementType) -> Lowered<()> {
        let et = mlir_scalar(e).ok_or(crate::emitter_gap!())?;
        // Each unsigned width has a helper of its own, so the element type alone decides
        // which one, and nothing has to reason about where a value still fits.
        let unsigned_helper = match e {
            ElementType::U8 => Some("print_u8"),
            ElementType::U16 => Some("print_u16"),
            ElementType::U32 => Some("print_u32"),
            ElementType::U64 => Some("print_u64"),
            _ => None,
        };
        // The narrow *signed* scalars have none, so they widen to one that does -- the same
        // widening the AST path applies, so a program prints the same text either way.
        let (arg, et) = match et {
            // A `bool` goes to `print_bool` as an `i32`, the width C passes reliably.
            "i1" => {
                let w = format!("%pw{idx}");
                let _ = writeln!(self.body, "  {w} = arith.extui {arg} : i1 to i32");
                let n = format!("%v{idx}");
                let _ = writeln!(
                    self.body,
                    "  {n} = func.call @print_bool({w}) : (i32) -> i32"
                );
                return Ok(());
            }
            "f16" | "bf16" => {
                let w = format!("%pw{idx}");
                let _ = writeln!(self.body, "  {w} = arith.extf {arg} : {et} to f32");
                (w, "f32")
            }
            "i8" | "i16" if unsigned_helper.is_none() => {
                let w = format!("%pw{idx}");
                let _ = writeln!(self.body, "  {w} = arith.extsi {arg} : {et} to i32");
                (w, "i32")
            }
            _ => (arg, et),
        };
        let helper = match unsigned_helper {
            Some(h) => h,
            None => match et {
                "f32" => "print_f32",
                "f64" => "print_f64",
                "i32" => "print_i32",
                "i64" => "print_i64",
                _ => {
                    return Err(Decline::TypeNotModelled {
                        what: "a print of an element type with no helper",
                    })
                }
            },
        };
        let n = format!("%v{idx}");
        let _ = writeln!(
            self.body,
            "  {n} = func.call @{helper}({arg}) : ({et}) -> i32"
        );
        Ok(())
    }

    /// Stop the program with a message when a condition is false. The message is the string
    /// the instruction names, or "assertion failed" when it names none.
    ///
    /// # Errors
    ///
    /// Declines when the emitter has no record of the condition value.
    pub(crate) fn op_abort(&mut self, _idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let cond = self
            .names
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let msg = usize::try_from(ins.imm)
            .ok()
            .and_then(|i| self.strings.get(i))
            .map_or("assertion failed", std::string::String::as_str);
        let escaped = msg.replace('\\', "\\\\").replace('"', "\\\"");
        let _ = writeln!(self.body, "  cf.assert {cond}, \"{escaped}\"");
        Ok(())
    }

    /// The number in the name of the module-level global (`@".str.<n>"`) that holds the string
    /// literal an instruction names.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "`imm` is an index into the function's string table, which is held in memory"
    )]
    fn string_global_number(&self, ins: &HirInstruction) -> usize {
        self.str_base + ins.imm as usize
    }

    /// Print a string literal with the `print_str` runtime helper.
    ///
    /// # Errors
    ///
    /// Never fails. It returns a `Lowered` so that every `op_*` function has the same shape.
    #[allow(
        clippy::unnecessary_wraps,
        reason = "the caller is in another file and expects a `Lowered` from every `op_*` function"
    )]
    pub(crate) fn op_print_str(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let n = self.string_global_number(ins);
        let p = format!("%pstrp{idx}");
        let r = format!("%pstr{idx}");
        let _ = writeln!(
            self.body,
            "  {p} = llvm.mlir.addressof @\".str.{n}\" : !llvm.ptr"
        );
        let _ = writeln!(
            self.body,
            "  {r} = func.call @print_str({p}) : (!llvm.ptr) -> i32"
        );
        Ok(())
    }

    /// A string literal in value position: take the address of the module-level global
    /// (`@".str.<n>"`, `n = str_base + imm`, the same numbering as `PrintStr`) as a first-class
    /// `!llvm.ptr` value — what a `let s = "…"` binds or a string argument passes. The global's
    /// bytes are emitted by `emit_module_mlir` from the string side table.
    ///
    /// # Errors
    ///
    /// Never fails. It returns a `Lowered` so that every `op_*` function has the same shape.
    #[allow(
        clippy::unnecessary_wraps,
        reason = "the caller is in another file and expects a `Lowered` from every `op_*` function"
    )]
    pub(crate) fn op_string_const(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let n = self.string_global_number(ins);
        let p = format!("%v{idx}");
        let _ = writeln!(
            self.body,
            "  {p} = llvm.mlir.addressof @\".str.{n}\" : !llvm.ptr"
        );
        self.names[idx] = p;
        self.ptr_of[idx] = true;
        Ok(())
    }

    /// A `const` table: its read-only global, as a memref that indexing reads like any tensor's.
    /// `emit_module_mlir` writes the global itself.
    ///
    /// # Errors
    ///
    /// Declines when the instruction names a table the module does not have.
    pub(crate) fn op_const_table(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let (symbol, memty) = usize::try_from(ins.imm)
            .ok()
            .and_then(|i| self.ctx.const_tables.get(i))
            .ok_or(crate::emitter_gap!())?
            .clone();
        let n = format!("%v{idx}");
        let _ = writeln!(self.body, "  {n} = memref.get_global @{symbol} : {memty}");
        self.names[idx] = n;
        self.mem_of[idx] = Some(memty);
        Ok(())
    }
}
