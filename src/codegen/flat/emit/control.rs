//! Block labels, branches, and returning from the function.
//!
//! One `impl` block on the per-function emitter; the dispatch that reaches these
//! lives in `step`, next to the struct.

use super::super::{
    cast_op, elem_of_gid, mlir_scalar, peel_wrappers, ptr_gid, tensor_memref_of_type,
    tensor_memref_ty, vector_align_attr, ElementType, FnEmit, HirInstruction, Lowered, Type,
    DYN_DIM,
};
// Writing into a `String` cannot fail, so the results of `writeln!` below are dropped.
use std::fmt::Write as _;

impl FnEmit<'_> {
    /// Block markers → MLIR blocks. Block 0 is the func's entry block (implicit; it carries the
    /// params), so it gets no label; every other id opens `^bbN:`.
    ///
    /// # Errors
    ///
    /// Never fails. It returns a `Result` so that every opcode handler has the same signature.
    #[allow(
        clippy::unnecessary_wraps,
        reason = "callers are in other files; every opcode handler returns `Lowered` so `step` can dispatch them alike"
    )]
    pub(crate) fn op_block_start(&mut self, _idx: usize, ins: &HirInstruction) -> Lowered<()> {
        if ins.imm != 0 {
            let _ = writeln!(self.body, "^bb{}:", ins.imm);
        }
        self.terminated = false;
        Ok(())
    }

    /// Writes an unconditional branch to the block named by `imm`.
    ///
    /// # Errors
    ///
    /// Never fails. It returns a `Result` so that every opcode handler has the same signature.
    #[allow(
        clippy::unnecessary_wraps,
        reason = "callers are in other files; every opcode handler returns `Lowered` so `step` can dispatch them alike"
    )]
    pub(crate) fn op_br(&mut self, _idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let _ = writeln!(self.body, "  cf.br ^bb{}", ins.imm);
        self.terminated = true;
        Ok(())
    }

    /// `imm` packs the two targets as `then | (else << 32)` (see `flatten::pack_targets`).
    ///
    /// # Errors
    ///
    /// Fails when the condition register has no name.
    pub(crate) fn op_cond_br(&mut self, _idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let cond = self
            .names
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?;
        let then_b = ins.imm & 0xffff_ffff;
        let else_b = ins.imm >> 32;
        let _ = writeln!(self.body, "  cf.cond_br {cond}, ^bb{then_b}, ^bb{else_b}");
        self.terminated = true;
        Ok(())
    }

    /// Writes the `func.return` for a `return`, first moving the value to where the caller
    /// expects it: a cast scalar, the caller's result buffer, or a loaded struct.
    ///
    /// # Errors
    ///
    /// Fails when the returned value has no name or type on record, or has a type this emitter
    /// cannot return.
    pub(crate) fn op_ret(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        // A bare `return;`: emitted as an effect, so it carries the no-type sentinel and there is
        // no operand to name. Only a `void` function may write one, which the checker enforces.
        if ins.type_idx.0 == u32::MAX {
            self.body += "  func.return\n";
            self.terminated = true;
            return Ok(());
        }
        let gid = *self
            .types
            .get(ins.type_idx.0 as usize)
            .ok_or(crate::emitter_gap!())?;
        let a = self
            .names
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        if let Some(e) = elem_of_gid(gid) {
            self.ret_scalar(idx, &a, &e)?;
        } else if gid == ptr_gid() {
            let _ = writeln!(self.body, "  func.return {a} : !llvm.ptr"); // a pointer return (#235)
        } else if let Some(tensor) = self.ctx.tensors.get(&gid) {
            self.ret_tensor(idx, ins, &a, tensor)?;
        } else if let Some(agg) = self.ctx.aggs.get(&gid) {
            // A struct return (#215). The operand is either a slot pointer (a constructed
            // struct) -> load the value; or already a struct value (a returned call result) ->
            // return it directly.
            if self
                .agg_of
                .get(ins.operand1.0 as usize)
                .copied()
                .flatten()
                .is_some()
            {
                let rv = format!("%rv{idx}");
                let _ = writeln!(
                    self.body,
                    "  {rv} = llvm.load {a} : !llvm.ptr -> {}",
                    agg.struct_ty
                );
                let _ = writeln!(self.body, "  func.return {rv} : {}", agg.struct_ty);
            } else {
                let _ = writeln!(self.body, "  func.return {a} : {}", agg.struct_ty);
            }
        } else if let Some(vecty) = self
            .vec_of
            .get(ins.operand1.0 as usize)
            .cloned()
            .flatten()
            .filter(|_| matches!(peel_wrappers(&self.func.return_type), Type::Simd(..)))
        {
            // A `<N x T>` return: the register already holds the vector value.
            let _ = writeln!(self.body, "  func.return {a} : {vecty}");
        } else {
            return Err(crate::emitter_gap!());
        }
        self.terminated = true;
        Ok(())
    }

    /// Returns the scalar `a` of element type `e`, casting it first when the function declares
    /// a different return type.
    ///
    /// # Errors
    ///
    /// Fails when either element type has no MLIR spelling, or no cast exists between them.
    fn ret_scalar(&mut self, idx: usize, a: &str, e: &ElementType) -> Lowered<()> {
        // Coerce the returned scalar to the function's declared return type if they differ
        // (e.g. `return 10` — a default-`i32` literal — from an `-> i64` function ->
        // `arith.extsi`), mirroring the AST's `coerce_type` at return. Otherwise the
        // `func.return` type contradicts the signature.
        let target = self.ret_elem.clone().unwrap_or_else(|| e.clone());
        if mlir_scalar(e).ok_or(crate::emitter_gap!())?
            == mlir_scalar(&target).ok_or(crate::emitter_gap!())?
        {
            let _ = writeln!(
                self.body,
                "  func.return {a} : {}",
                mlir_scalar(e).ok_or(crate::emitter_gap!())?
            );
        } else {
            match cast_op(e, &target).ok_or(crate::emitter_gap!())? {
                "" => {
                    let _ = writeln!(
                        self.body,
                        "  func.return {a} : {}",
                        mlir_scalar(&target).ok_or(crate::emitter_gap!())?
                    );
                }
                op => {
                    let c = format!("%rc{idx}");
                    let _ = writeln!(
                        self.body,
                        "  {c} = {op} {a} : {} to {}",
                        mlir_scalar(e).ok_or(crate::emitter_gap!())?,
                        mlir_scalar(&target).ok_or(crate::emitter_gap!())?
                    );
                    let _ = writeln!(
                        self.body,
                        "  func.return {c} : {}",
                        mlir_scalar(&target).ok_or(crate::emitter_gap!())?
                    );
                }
            }
        }
        Ok(())
    }

    /// Returns the tensor `a`, whose element type and shape are `tensor`.
    ///
    /// # Errors
    ///
    /// Fails when the tensor type has no memref spelling.
    fn ret_tensor(
        &mut self,
        idx: usize,
        ins: &HirInstruction,
        a: &str,
        tensor: &(ElementType, Vec<String>),
    ) -> Lowered<()> {
        let (elem, shape) = tensor;
        let slot = self.ret_slot.clone();
        if let Some(vecty) = self.vec_of.get(ins.operand1.0 as usize).cloned().flatten() {
            // An elementwise result is a vector register; the result is a buffer. Store the
            // vector into the caller's slot, or into a fresh one when there is no slot (a
            // `?`-shaped return, where the caller could not have sized it).
            let memty = tensor_memref_ty(elem, shape).ok_or(crate::emitter_gap!())?;
            let al = vector_align_attr(&vecty);
            let rc = format!("%rc{idx}");
            let rt = if let Some((s, _)) = &slot {
                s.clone()
            } else {
                let rt = format!("%rt{idx}");
                let _ = writeln!(self.body, "  {rt} = memref.alloc() : {memty}");
                rt
            };
            let _ = writeln!(self.body, "  {rc} = arith.constant 0 : index");
            let _ = writeln!(
                self.body,
                "  vector.store {a}, {rt}[{rc}]{al} : {memty}, {vecty}"
            );
            match &slot {
                Some(_) => self.body += "  func.return\n",
                None => {
                    let _ = writeln!(self.body, "  func.return {rt} : {memty}");
                }
            }
        } else {
            // A memref value: its tracked type is authoritative (a strided row differs from
            // the plain spelling); fall back to the GID's shape.
            let memty = match self.mem_of.get(ins.operand1.0 as usize).cloned().flatten() {
                Some(m) => m,
                None => tensor_memref_ty(elem, shape).ok_or(crate::emitter_gap!())?,
            };
            match &slot {
                // Under NRVO the value already is the slot -- the body has been writing into
                // the caller's buffer all along, so there is nothing left to do.
                Some((s, _)) if *s == a => self.body += "  func.return\n",
                // Otherwise the result lives somewhere else (a parameter, a call result) and
                // has to be copied into the caller's buffer before the frame goes away.
                Some((s, slot_ty)) => {
                    let src = if &memty == slot_ty {
                        a.to_string()
                    } else {
                        // A strided row or a `?`-spelled view reaching a statically shaped
                        // return: `memref.copy` wants both sides spelled the same.
                        let c = format!("%rcast{idx}");
                        let _ =
                            writeln!(self.body, "  {c} = memref.cast {a} : {memty} to {slot_ty}");
                        c
                    };
                    let _ = writeln!(
                        self.body,
                        "  memref.copy {src}, {s} : {slot_ty} to {slot_ty}"
                    );
                    // The copy moved the result into the caller's buffer. What it was copied
                    // from is freed when this function owned it, and left alone when it is a
                    // view of something else.
                    let _ = writeln!(self.body, "  vx.drop {a} {{owned_only}} : {memty}");

                    self.body += "  func.return\n";
                }
                None => self.ret_tensor_without_slot(idx, a, &memty, shape)?,
            }
        }
        Ok(())
    }

    /// Returns the tensor `a` of memref type `memty` from a function that has no result buffer
    /// from its caller: `a` itself when its type matches the declared one, a fresh copy otherwise.
    ///
    /// # Errors
    ///
    /// Fails when the function's declared return type has no memref spelling.
    fn ret_tensor_without_slot(
        &mut self,
        idx: usize,
        a: &str,
        memty: &str,
        shape: &[String],
    ) -> Lowered<()> {
        let declared =
            tensor_memref_of_type(&self.func.return_type).ok_or(crate::emitter_gap!())?;
        if declared == memty {
            let _ = writeln!(self.body, "  func.return {a} : {memty}");
        } else {
            // A row is a strided view, and a `?`-shaped result is a buffer of
            // its own: return a copy.
            let mut sizes = Vec::new();
            for (k, d) in shape.iter().enumerate() {
                if d == DYN_DIM {
                    let (c, v) = (format!("%rk{idx}_{k}"), format!("%rd{idx}_{k}"));
                    let _ = writeln!(self.body, "  {c} = arith.constant {k} : index");
                    let _ = writeln!(self.body, "  {v} = memref.dim {a}, {c} : {memty}");
                    sizes.push(v);
                }
            }
            let rt = format!("%rt{idx}");
            let _ = writeln!(
                self.body,
                "  {rt} = memref.alloc({}) : {declared}",
                sizes.join(", ")
            );
            let _ = writeln!(self.body, "  memref.copy {a}, {rt} : {memty} to {declared}");
            let _ = writeln!(self.body, "  func.return {rt} : {declared}");
        }
        Ok(())
    }
}
