//! Calls: argument collection, direct calls, function constants and indirect calls.
//!
//! One `impl` block on the per-function emitter; the dispatch that reaches these
//! lives in `step`, next to the struct.

use super::super::{
    elem_of_gid, mlir_scalar, sym_ref, tensor_memref_ty, Callee, Decline, FnEmit, HirInstruction,
    Lowered, DYN_DIM,
};
use std::fmt::Write as _;

/// The argument count an instruction carries in `imm`. A count too large for `usize` becomes
/// `usize::MAX`: no list of pending arguments is that long, so the caller declines.
fn arg_count(imm: u64) -> usize {
    usize::try_from(imm).unwrap_or(usize::MAX)
}

impl FnEmit<'_> {
    /// Whether register `r` is passed as a bare `!llvm.ptr`: a pointer value, a pointer local's
    /// slot handed out by `&mut p` (the slot's address is the pointer the callee wants), or an
    /// aggregate slot passed by reference.
    ///
    /// # Errors
    ///
    /// Declines when register `r` is not one this function has.
    fn is_pointer_value(&self, r: usize) -> Lowered<bool> {
        Ok(*self.ptr_of.get(r).ok_or(crate::emitter_gap!())?
            || self.pslot_of.get(r).copied().unwrap_or(false)
            || self.agg_of.get(r).copied().flatten().is_some())
    }

    /// One argument of the following `Call`: record its value register (no op emitted).
    ///
    /// # Errors
    ///
    /// Never fails. It returns `Lowered` so the dispatch calls every instruction the same way.
    #[allow(
        clippy::unnecessary_wraps,
        reason = "the caller is the dispatch in another file, where every arm returns `Lowered`"
    )]
    pub(crate) fn op_arg(&mut self, _idx: usize, ins: &HirInstruction) -> Lowered<()> {
        self.pending_args.push(ins.operand1.0);
        Ok(())
    }

    /// The MLIR spelling of what `callee` returns: a scalar, an `!llvm.struct` by value for a
    /// struct-returning callee (#215), a pointer, or `()` for a void callee (a `&mut` mutator
    /// called in statement position, #230).
    ///
    /// # Errors
    ///
    /// Declines when the return type has no MLIR spelling, or names a struct the module does not
    /// have a layout for.
    fn call_return_type(&self, callee: &Callee) -> Lowered<String> {
        Ok(if let Some(e) = &callee.ret {
            mlir_scalar(e).ok_or(crate::emitter_gap!())?.to_string()
        } else if let Some(agg_gid) = callee.ret_agg {
            self.ctx
                .aggs
                .get(&agg_gid)
                .ok_or(crate::emitter_gap!())?
                .struct_ty
                .clone()
        } else if let Some(mt) = &callee.ret_tensor {
            mt.clone() // a `?`-shaped tensor return: the result is a memref value the callee made
        } else if callee.ret_ptr {
            "!llvm.ptr".to_string() // an FFI pointer-returning callee (#235)
        } else if callee.ret_void {
            "()".to_string()
        } else {
            return Err(Decline::TypeNotModelled {
                what: "a callee whose return type has no MLIR spelling",
            });
        })
    }

    /// Takes the last `n` pending arguments, and gives the name and the MLIR type of each.
    ///
    /// # Errors
    ///
    /// Declines when fewer than `n` arguments are pending, or when an argument has no name or
    /// no type on record.
    fn take_call_args(&mut self, n: usize) -> Lowered<(Vec<String>, Vec<String>)> {
        if self.pending_args.len() < n {
            return Err(crate::emitter_gap!());
        }
        let args = self.pending_args.split_off(self.pending_args.len() - n);
        let mut arg_names = Vec::with_capacity(n);
        let mut arg_types: Vec<String> = Vec::with_capacity(n);
        for a in &args {
            arg_names.push(
                self.names
                    .get(*a as usize)
                    .ok_or(crate::emitter_gap!())?
                    .clone(),
            );
            // A scalar arg is its element type; a pointer arg (a string value / FFI pointer, or
            // an aggregate *slot* passed by reference — `&v` / a `self` pointer, #242) is
            // `!llvm.ptr`; a tensor arg is its memref type.
            let at = if let Some(e) = self.elem_at(*a) {
                mlir_scalar(&e).ok_or(crate::emitter_gap!())?.to_string()
            } else if let Some(agg_gid) = self.agg_val_of.get(*a as usize).copied().flatten() {
                // A by-value aggregate argument (`push(&outer, a)` passing `a : Vec<i32>` by
                // value into a `Vec<Vec<T>>::push`) — an `!llvm.struct` value (#242).
                self.ctx
                    .aggs
                    .get(&agg_gid)
                    .ok_or(crate::emitter_gap!())?
                    .struct_ty
                    .clone()
            } else if self.is_pointer_value(*a as usize)? {
                "!llvm.ptr".to_string()
            } else {
                self.mem_of
                    .get(*a as usize)
                    .ok_or(crate::emitter_gap!())?
                    .clone()
                    .ok_or(crate::emitter_gap!())?
            };
            arg_types.push(at);
        }
        Ok((arg_names, arg_types))
    }

    /// Casts each tensor argument whose memref spelling differs from the one the callee
    /// declares. A row is a strided view, and the callee takes a plain memref: pass the same
    /// memory under the callee's spelling. `declared` is the callee's `param_tensors`.
    ///
    /// # Errors
    ///
    /// Declines when a cast between the two memref types cannot be written.
    fn cast_args_to_declared(
        &mut self,
        idx: usize,
        declared: &[Option<String>],
        arg_names: &mut [String],
        arg_types: &mut [String],
    ) -> Lowered<()> {
        let n = arg_names.len();
        for (k, declared) in declared.iter().enumerate().take(n) {
            if let Some(declared) = declared {
                if arg_types[k].starts_with("memref<") && arg_types[k] != *declared {
                    arg_names[k] = self.cast_memref_value(
                        &format!("pc{idx}_{k}"),
                        &arg_names[k],
                        &arg_types[k],
                        declared,
                    )?;
                    arg_types[k].clone_from(declared);
                }
            }
        }
        Ok(())
    }

    /// Records what kind of value the call at `idx` returned, so later instructions can use it.
    fn record_call_result(&mut self, idx: usize, callee: &Callee) {
        if let Some(e) = &callee.ret {
            self.etypes[idx] = Some(e.clone());
        } else if callee.ret_ptr {
            self.ptr_of[idx] = true; // the call result is a pointer value (#235)
            if let Some(pointee) = callee.ret_pointee {
                self.agg_of[idx] = Some(pointee);
            }
        } else if let Some(agg_gid) = callee.ret_agg {
            // A struct-returning call result is a struct *value*; tracked so it can be
            // spilled to a slot (`Store`), returned (`Ret`), or passed by value to another
            // call (#242).
            self.agg_val_of[idx] = Some(agg_gid);
        } else if let Some(mt) = &callee.ret_tensor {
            self.mem_of[idx] = Some(mt.clone()); // a tensor result: downstream ops index it
        }
    }

    /// A fixed-arity call. `type_idx` is the callee's GID (resolved to name + return type via
    /// `ctx.callees`); `imm` is the arg count, taken from the tail of `pending_args`. Emit
    /// `%r = func.call @name(%a, %b) : (Ta, Tb) -> Tret`.
    ///
    /// # Errors
    ///
    /// Declines when the callee is not known, when its return type or an argument's type has no
    /// MLIR spelling, or when fewer arguments are pending than the call takes.
    pub(crate) fn op_call(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let gid = *self
            .types
            .get(ins.type_idx.0 as usize)
            .ok_or(crate::emitter_gap!())?;
        let callee = self.ctx.callees.get(&gid).ok_or(crate::emitter_gap!())?;
        let rt = self.call_return_type(callee)?;
        let (mut arg_names, mut arg_types) = self.take_call_args(arg_count(ins.imm))?;
        let param_tensors = callee.param_tensors.clone();
        self.cast_args_to_declared(idx, &param_tensors, &mut arg_names, &mut arg_types)?;
        // A callee returning a statically shaped tensor writes into a buffer we hand it, so
        // allocate one and pass it as the first argument (#643). The slot goes in our entry block:
        // a call in a loop then reuses one buffer rather than taking a fresh one per iteration,
        // which is right because the result is consumed before the next iteration overwrites it.
        let ret_slot_ty = callee
            .ret_tensor
            .as_ref()
            .filter(|mt| !mt.contains(DYN_DIM) && !callee.ret_placed)
            .cloned();
        if let Some(slot_ty) = &ret_slot_ty {
            let slot = format!("%rs{idx}");
            self.emit_slot(&format!("  {slot} = memref.alloca() : {slot_ty}\n"));
            arg_names.insert(0, slot.clone());
            arg_types.insert(0, slot_ty.clone());
            self.names[idx] = slot;
            self.mem_of[idx] = Some(slot_ty.clone()); // downstream ops index the buffer
        }
        if callee.ret_void || ret_slot_ty.is_some() {
            // A void call binds no result register (MLIR forbids `%v = func.call ... -> ()`);
            // the call is a pure effect (mutation through a `&mut` arg). The private extern decl
            // records an empty return (no `->`) so a void `extern` declares as `(args)`. (#230)
            let _ = writeln!(
                self.body,
                "  func.call {}({}) : ({}) -> ()",
                sym_ref(&callee.name),
                arg_names.join(", "),
                arg_types.join(", "),
            );
            self.calls
                .push((callee.name.clone(), arg_types.clone(), String::new()));
        } else {
            let nm = format!("%v{idx}");
            let _ = writeln!(
                self.body,
                "  {nm} = func.call {}({}) : ({}) -> {rt}",
                sym_ref(&callee.name),
                arg_names.join(", "),
                arg_types.join(", "),
            );
            // Record the callee's signature so the module emitter can declare it if it is a
            // called-but-undefined symbol (an `extern`): the private decl's signature is taken
            // from the emitted call, so they match by construction.
            self.calls
                .push((callee.name.clone(), arg_types.clone(), rt.clone()));
            self.names[idx] = nm;
            self.record_call_result(idx, callee);
        }
        Ok(())
    }

    /// The operands of the call to an inline `mlir!` block's wrapper: the last pending arguments,
    /// one for each of the block's inputs. A memref input is cast to the declared type when the
    /// spellings differ.
    ///
    /// # Errors
    ///
    /// Declines when fewer arguments are pending than the block has inputs, or when an argument
    /// has no name or no type on record.
    fn inline_mlir_operands(
        &mut self,
        idx: usize,
        blk: &crate::hir::flatten::InlineBlock,
    ) -> Lowered<Vec<String>> {
        let n = blk.inputs.len();
        if self.pending_args.len() < n {
            return Err(crate::emitter_gap!());
        }
        let args = self.pending_args.split_off(self.pending_args.len() - n);
        let mut operands = Vec::with_capacity(n);
        for (k, (a, (_, declared))) in args.iter().zip(&blk.inputs).enumerate() {
            let r = *a as usize;
            let name = self.names.get(r).ok_or(crate::emitter_gap!())?.clone();
            let actual = if let Some(e) = self.elem_at(*a) {
                mlir_scalar(&e).ok_or(crate::emitter_gap!())?.to_string()
            } else if let Some(Some(m)) = self.mem_of.get(r) {
                m.clone()
            } else if self.is_pointer_value(r)? {
                "!llvm.ptr".to_string()
            } else {
                return Err(crate::emitter_gap!());
            };
            let same = actual.replace(' ', "") == declared.replace(' ', "");
            if !same && actual.starts_with("memref<") {
                let c =
                    self.cast_memref_value(&format!("imc{idx}_{k}"), &name, &actual, declared)?;
                operands.push(c);
            } else {
                operands.push(name);
            }
        }
        Ok(operands)
    }

    /// The MLIR type of the value an inline `mlir!` block returns, taken from the instruction's
    /// `type_idx`. Also records what kind of value register `idx` now holds.
    ///
    /// # Errors
    ///
    /// Declines when the type is not a scalar, a tensor or a struct that the emitter models.
    fn inline_mlir_result_type(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<String> {
        let gid = *self
            .types
            .get(ins.type_idx.0 as usize)
            .ok_or(crate::emitter_gap!())?;
        if let Some(e) = elem_of_gid(gid) {
            let t = mlir_scalar(&e).ok_or(crate::emitter_gap!())?.to_string();
            self.etypes[idx] = Some(e);
            Ok(t)
        } else if let Some((e, shape)) = self.ctx.tensors.get(&gid) {
            let t = tensor_memref_ty(e, shape).ok_or(crate::emitter_gap!())?;
            self.mem_of[idx] = Some(t.clone());
            Ok(t)
        } else if let Some(agg) = self.ctx.aggs.get(&gid) {
            self.agg_val_of[idx] = Some(gid);
            Ok(agg.struct_ty.clone())
        } else {
            Err(crate::emitter_gap!())
        }
    }

    /// An inline `mlir!` block: a call to a private wrapper whose parameters are the block's
    /// named inputs and whose body is its text, the way the oracle wraps one. The wrapper goes
    /// after this function's text. A memref input is cast to the declared type when the
    /// spellings differ (a `?` extent against a static one, a strided row against a plain one).
    ///
    /// # Errors
    ///
    /// Declines when the instruction names a block the function does not have, when fewer
    /// arguments are pending than the block has inputs, or when an input or the result has a
    /// type the emitter does not model.
    pub(crate) fn op_inline_mlir(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let Ok(block_idx) = usize::try_from(ins.imm) else {
            return Err(crate::emitter_gap!());
        };
        let blk = self
            .inline_blocks
            .get(block_idx)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let operands = self.inline_mlir_operands(idx, &blk)?;
        let fname = format!("vx_flat_mlir_{}", self.inline_base + block_idx);
        let tys: Vec<&str> = blk.inputs.iter().map(|(_, t)| t.as_str()).collect();
        let params: Vec<String> = blk
            .inputs
            .iter()
            .map(|(nm, t)| format!("{nm}: {t}"))
            .collect();
        let ret = if blk.void {
            None
        } else {
            Some(self.inline_mlir_result_type(idx, ins)?)
        };
        match &ret {
            Some(rt) => {
                let nm = format!("%v{idx}");
                let _ = writeln!(
                    self.body,
                    "  {nm} = func.call @{fname}({}) : ({}) -> {rt}",
                    operands.join(", "),
                    tys.join(", ")
                );
                self.names[idx] = nm;
            }
            None => {
                let _ = writeln!(
                    self.body,
                    "  func.call @{fname}({}) : ({}) -> ()",
                    operands.join(", "),
                    tys.join(", ")
                );
            }
        }
        let ret_sig = ret.map(|rt| format!(" -> {rt}")).unwrap_or_default();
        let _ = writeln!(
            self.wrappers,
            "func.func private @{fname}({}){ret_sig} {{\n{}\n}}",
            params.join(", "),
            blk.body.replace("macro.yield", "return")
        );
        Ok(())
    }

    /// Materialize a function pointer for a named function: `type_idx` is the target's GID
    /// (name via `ctx.callees`, signature via `ctx.func_sigs`). Emit `func.constant @name : sig`
    /// then cast the `FunctionType` value to an opaque `!llvm.ptr` (the ABI of a fn pointer),
    /// tracked in `ptr_of`. (#242)
    ///
    /// # Errors
    ///
    /// Declines when the function or its signature is not known.
    pub(crate) fn op_func_const(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let gid = *self
            .types
            .get(ins.type_idx.0 as usize)
            .ok_or(crate::emitter_gap!())?;
        let callee = self.ctx.callees.get(&gid).ok_or(crate::emitter_gap!())?;
        let (params, ret) = self.ctx.func_sigs.get(&gid).ok_or(crate::emitter_gap!())?;
        let fnty = format!("({}) -> {}", params.join(", "), ret);
        let fc = format!("%fc{idx}");
        let nm = format!("%v{idx}");
        let _ = writeln!(
            self.body,
            "  {fc} = func.constant {} : {fnty}",
            sym_ref(&callee.name)
        );
        let _ = writeln!(
            self.body,
            "  {nm} = builtin.unrealized_conversion_cast {fc} : {fnty} to !llvm.ptr"
        );
        self.names[idx] = nm;
        self.ptr_of[idx] = true;
        Ok(())
    }

    /// A differentiated call (`grad`/`vjp`/`jvp`). `type_idx` is the TARGET's GID, not a callee's:
    /// the target is materialized as a function constant and handed to Enzyme's wrapper as its
    /// first argument, which is the shape Enzyme recognizes. `imm` packs the argument count with
    /// the mode. The wrapper itself is never defined, so it declares like any other extern.
    ///
    /// # Errors
    ///
    /// Declines when the target or its signature is not known, when the target does not return a
    /// scalar, when an argument is not a scalar, or when fewer arguments are pending than `imm`
    /// counts.
    pub(crate) fn op_autodiff(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let gid = *self
            .types
            .get(ins.type_idx.0 as usize)
            .ok_or(crate::emitter_gap!())?;
        let target = self.ctx.callees.get(&gid).ok_or(crate::emitter_gap!())?;
        let (params, ret) = self.ctx.func_sigs.get(&gid).ok_or(crate::emitter_gap!())?;
        let ret_elem = target.ret.clone().ok_or(Decline::TypeNotModelled {
            what: "an autodiff target whose return type has no MLIR spelling",
        })?;
        let fnty = format!("({}) -> {}", params.join(", "), ret);
        let forward =
            (ins.imm >> crate::bytecode::AUTODIFF_MODE_SHIFT) == crate::bytecode::AUTODIFF_FORWARD;
        let n = (ins.imm & 0xffff_ffff) as usize;
        if self.pending_args.len() < n {
            return Err(crate::emitter_gap!());
        }
        let args = self.pending_args.split_off(self.pending_args.len() - n);
        // The function constant leads, so the wrapper's signature starts with the target's type.
        let mut arg_names = vec![format!("%fc{idx}")];
        let mut arg_types = vec![fnty.clone()];
        for a in &args {
            arg_names.push(
                self.names
                    .get(*a as usize)
                    .ok_or(crate::emitter_gap!())?
                    .clone(),
            );
            let e = self.elem_at(*a).ok_or(Decline::TypeNotModelled {
                what: "an autodiff argument that is not a scalar",
            })?;
            arg_types.push(mlir_scalar(&e).ok_or(crate::emitter_gap!())?.to_string());
        }
        let wrapper = crate::codegen::enzyme_wrapper_name(forward, &target.name);
        let nm = format!("%v{idx}");
        let _ = writeln!(
            self.body,
            "  %fc{idx} = func.constant {} : {fnty}",
            sym_ref(&target.name)
        );
        let _ = writeln!(
            self.body,
            "  {nm} = func.call {}({}) : ({}) -> {ret}",
            sym_ref(&wrapper),
            arg_names.join(", "),
            arg_types.join(", "),
        );
        self.calls.push((wrapper, arg_types, ret.clone()));
        self.names[idx] = nm;
        self.etypes[idx] = Some(ret_elem);
        Ok(())
    }

    /// An indirect call through a function pointer. `operand1` is the callee `!llvm.ptr`, `imm`
    /// the arg count (the tail of `pending_args`, like `Call`), and this instruction's `type_idx`
    /// the scalar return type. Reconstruct the function type `(arg types)->ret` from the actual
    /// args, cast the pointer to it, and `func.call_indirect`. (#242)
    ///
    /// # Errors
    ///
    /// Declines when the return type is neither a scalar nor a pointer, when fewer arguments are
    /// pending than the call takes, or when the callee or an argument has no name or no type on
    /// record.
    pub(crate) fn op_call_indirect(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        // The result is a scalar, or a pointer when the type is the pointer GID.
        let ret_gid = *self
            .types
            .get(ins.type_idx.0 as usize)
            .ok_or(crate::emitter_gap!())?;
        let ret_ptr = ret_gid == crate::hir::flatten::ptr_gid();
        let ret_elem = if ret_ptr {
            None
        } else {
            self.ty_at(ins.type_idx.0)
        };
        let rt = match &ret_elem {
            Some(e) => mlir_scalar(e).ok_or(crate::emitter_gap!())?.to_string(),
            None if ret_ptr => "!llvm.ptr".to_string(),
            None => return Err(crate::emitter_gap!()),
        };
        let (arg_names, arg_types) = self.take_call_args(arg_count(ins.imm))?;
        let fnty = format!("({}) -> {rt}", arg_types.join(", "));
        let fnptr = self
            .names
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let fc = format!("%fc{idx}");
        let nm = format!("%v{idx}");
        let _ = writeln!(
            self.body,
            "  {fc} = builtin.unrealized_conversion_cast {fnptr} : !llvm.ptr to {fnty}"
        );
        let _ = writeln!(
            self.body,
            "  {nm} = func.call_indirect {fc}({}) : {fnty}",
            arg_names.join(", ")
        );
        self.names[idx] = nm;
        if ret_ptr {
            self.ptr_of[idx] = true;
        } else {
            self.etypes[idx] = ret_elem;
        }
        Ok(())
    }
}
