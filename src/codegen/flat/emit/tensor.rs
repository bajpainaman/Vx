//! Tensors: allocation, indexing, stores, and the reduction and matmul kernels.
//!
//! One `impl` block on the per-function emitter; the dispatch that reaches these
//! lives in `step`, next to the struct.

use super::super::{
    cast_op, elem_from_mlir_scalar, elem_of_gid, join_i64, memref_descriptor_ty, memref_elem,
    memref_lead_dim, memref_lead_dims_and_elem, mlir_scalar, sym_ref, tensor_memref_ty,
    vector_align_attr, Decline, ElementType, FnEmit, HirInstruction, Lowered, TypeId, DYN_DIM,
};
use std::fmt::Write as _;

/// The element type of a memref spelling: `memref<8x16xf32>` -> `f32`.
///
/// The element is the segment after the last `x`, shorn of the closing `>` and of a memory-space
/// suffix (`memref<4x4xf32, 3>`). A rank-0 memref has no `x` and is all element.
fn memref_element(memty: &str) -> Option<&str> {
    let inner = memty.strip_prefix("memref<")?;
    let inner = inner.split(',').next()?.trim_end_matches('>');
    Some(inner.rsplit_once('x').map_or(inner, |(_, el)| el))
}

/// The leading dimensions (`4x8x`), element type and rank of the strided view type `src_ty`,
/// once its layout is known to be contiguous with an offset of `?` or 0.
///
/// # Errors
///
/// Declines when the offset is something else, when the type does not spell one stride per
/// dimension, or when a literal stride is not the product of the sizes after it.
fn contiguous_view_layout(src_ty: &str) -> Lowered<(&str, &str, usize)> {
    if !src_ty.ends_with("offset: ?>>") && !src_ty.ends_with("offset: 0>>") {
        return Err(crate::emitter_gap!());
    }
    let (dims_x, et) = memref_lead_dims_and_elem(src_ty).ok_or(crate::emitter_gap!())?;
    let dims: Vec<&str> = dims_x.split('x').filter(|d| !d.is_empty()).collect();
    let strides: Vec<&str> = src_ty
        .split("strided<[")
        .nth(1)
        .and_then(|s| s.split(']').next())
        .ok_or(crate::emitter_gap!())?
        .split(',')
        .map(str::trim)
        .collect();
    if dims.is_empty() || strides.len() != dims.len() {
        return Err(crate::emitter_gap!());
    }
    // Each stride must be the product of the sizes after it. A `?` cannot be checked here;
    // only indexing a row out of a tensor makes one, and a row of a tensor is contiguous.
    let mut expect = Some(1i64);
    for k in (0..dims.len()).rev() {
        if let (Some(e), Ok(s)) = (expect, strides[k].parse::<i64>()) {
            if s != e {
                return Err(crate::emitter_gap!());
            }
        }
        expect = expect.zip(dims[k].parse::<i64>().ok()).map(|(e, d)| e * d);
    }
    Ok((dims_x, et, dims.len()))
}

impl FnEmit<'_> {
    /// Give `val`, of memref type `src_ty`, the type `dst_ty`, and return the new value's name.
    ///
    /// A `memref.cast` is enough unless `val` is a strided view (a row, `memref<4xf32,
    /// strided<[1], offset: ?>>`) and `dst_ty` has no layout. A plain memref's offset is 0 by its
    /// type, so the code using it never reads the descriptor's offset, and a cast would hand it
    /// the start of the tensor rather than the row. Instead the descriptor's data pointer is
    /// moved to the row and its offset set to 0. The view must be contiguous: a plain memref has
    /// row-major strides. A view in a GPU memory space is refused, as its pointers are not `ptr`.
    ///
    /// # Errors
    ///
    /// Declines when a strided view has to become a plain memref and is not contiguous, or is in
    /// a GPU memory space.
    pub(crate) fn cast_memref_value(
        &mut self,
        tag: &str,
        val: &str,
        src_ty: &str,
        dst_ty: &str,
    ) -> Lowered<String> {
        if src_ty == dst_ty {
            return Ok(val.to_string());
        }
        if !src_ty.contains("strided<") || dst_ty.contains("strided<") {
            let c = format!("%{tag}");
            let _ = writeln!(
                self.body,
                "  {c} = memref.cast {val} : {src_ty} to {dst_ty}"
            );
            return Ok(c);
        }
        let (dims_x, et, rank) = contiguous_view_layout(src_ty)?;
        let desc = memref_descriptor_ty(rank);
        let plain = format!("memref<{dims_x}{et}>");
        let d = format!("%{tag}_d");
        let al = format!("%{tag}_al");
        let off = format!("%{tag}_off");
        let ptr = format!("%{tag}_p");
        let z = format!("%{tag}_z");
        let d1 = format!("%{tag}_d1");
        let d2 = format!("%{tag}_d2");
        let r = format!("%{tag}_r");
        let _ = writeln!(
            self.body,
            "  {d} = builtin.unrealized_conversion_cast {val} : {src_ty} to {desc}"
        );
        let _ = writeln!(self.body, "  {al} = llvm.extractvalue {d}[1] : {desc}");
        let _ = writeln!(self.body, "  {off} = llvm.extractvalue {d}[2] : {desc}");
        let _ = writeln!(
            self.body,
            "  {ptr} = llvm.getelementptr {al}[{off}] : (!llvm.ptr, i64) -> !llvm.ptr, {et}"
        );
        let _ = writeln!(self.body, "  {z} = llvm.mlir.constant(0 : i64) : i64");
        let _ = writeln!(
            self.body,
            "  {d1} = llvm.insertvalue {ptr}, {d}[1] : {desc}"
        );
        let _ = writeln!(self.body, "  {d2} = llvm.insertvalue {z}, {d1}[2] : {desc}");
        let _ = writeln!(
            self.body,
            "  {r} = builtin.unrealized_conversion_cast {d2} : {desc} to {plain}"
        );
        if plain == dst_ty {
            return Ok(r);
        }
        let c = format!("%{tag}");
        let _ = writeln!(self.body, "  {c} = memref.cast {r} : {plain} to {dst_ty}");
        Ok(c)
    }

    /// `t.as_ptr()`: the address of the first element, as a bare `!llvm.ptr`. The descriptor's
    /// aligned pointer plus its offset in bytes, so a row view answers with the row's start and
    /// not the buffer's. A tensor in GPU memory is an ordinary descriptor whose pointer is the
    /// device address, so the same code hands a C function what cuBLAS expects.
    ///
    /// # Errors
    ///
    /// Declines when the operand is not a tensor the emitter knows, or its element type has no
    /// known size in bits.
    pub(crate) fn op_tensor_data_ptr(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let s = ins.operand1.0 as usize;
        let src = self.names.get(s).ok_or(crate::emitter_gap!())?.clone();
        let src_mem = self
            .mem_of
            .get(s)
            .cloned()
            .flatten()
            .ok_or(crate::emitter_gap!())?;
        let (dims_x, et) = memref_lead_dims_and_elem(&src_mem).ok_or(crate::emitter_gap!())?;
        let rank = dims_x.split('x').filter(|d| !d.is_empty()).count();
        let bytes = crate::codegen::generator::scalar_type_bits(et)
            .ok_or(crate::emitter_gap!())?
            .div_ceil(8);
        let space_sfx = if src_mem.ends_with(", 3>") { ", 3" } else { "" };
        let buf = format!("%dp{idx}_b");
        let off = format!("%dp{idx}_o");
        let rest: Vec<String> = (0..2 * rank).map(|k| format!("%dp{idx}_r{k}")).collect();
        let results = [vec![buf, off.clone()], rest].concat().join(", ");
        let index_tys = vec!["index"; 1 + 2 * rank].join(", ");
        let _ = writeln!(self.body,
            "  {results} = memref.extract_strided_metadata {src} : {src_mem} -> memref<{et}{space_sfx}>, {index_tys}"
        );
        let _ = writeln!(
            self.body,
            "  %dp{idx}_a = memref.extract_aligned_pointer_as_index {src} : {src_mem} -> index"
        );
        let _ = writeln!(self.body, "  %dp{idx}_e = arith.constant {bytes} : index");
        let _ = writeln!(
            self.body,
            "  %dp{idx}_m = arith.muli {off}, %dp{idx}_e : index"
        );
        let _ = writeln!(
            self.body,
            "  %dp{idx}_s = arith.addi %dp{idx}_a, %dp{idx}_m : index"
        );
        let _ = writeln!(
            self.body,
            "  %dp{idx}_i = arith.index_cast %dp{idx}_s : index to i64"
        );
        let n = format!("%v{idx}");
        let _ = writeln!(
            self.body,
            "  {n} = llvm.inttoptr %dp{idx}_i : i64 to !llvm.ptr"
        );
        self.names[idx] = n;
        self.ptr_of[idx] = true;
        Ok(())
    }

    /// `vx.drop`: convert-vx-to-standard frees the buffer when it is on the heap.
    ///
    /// # Errors
    ///
    /// Declines when the dropped value, or the flag that says it was moved, is not a value the
    /// emitter knows.
    pub(crate) fn op_tensor_drop(&mut self, _idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let r = ins.operand1.0 as usize;
        // A small elementwise result held in a vector register has no buffer.
        if self.vec_of.get(r).cloned().flatten().is_some() {
            return Ok(());
        }
        let name = self.names.get(r).ok_or(crate::emitter_gap!())?.clone();
        let ty = self
            .mem_of
            .get(r)
            .cloned()
            .flatten()
            .ok_or(crate::emitter_gap!())?;
        let unless = if ins.imm & crate::bytecode::TENSOR_DROP_FLAG != 0 {
            let moved = self
                .names
                .get(ins.operand2.0 as usize)
                .ok_or(crate::emitter_gap!())?;
            format!(" unless {moved}")
        } else {
            String::new()
        };
        let field = if ins.imm & crate::bytecode::TENSOR_DROP_FIELD != 0 {
            " {field}"
        } else {
            ""
        };
        let _ = writeln!(self.body, "  vx.drop {name}{unless}{field} : {ty}");
        Ok(())
    }

    /// `t.clone()`: a new buffer of the source's shape and a `memref.copy` into it. The copy reads
    /// the source through its own layout, so cloning a row copies that row.
    ///
    /// # Errors
    ///
    /// Declines when the result type or the source is not a tensor the emitter knows.
    pub(crate) fn op_tensor_clone(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let gid = *self
            .types
            .get(ins.type_idx.0 as usize)
            .ok_or(crate::emitter_gap!())?;
        let (elem, shape) = self
            .ctx
            .tensors
            .get(&gid)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let memty = tensor_memref_ty(&elem, &shape).ok_or(crate::emitter_gap!())?;
        let s = ins.operand1.0 as usize;
        let src = self.names.get(s).ok_or(crate::emitter_gap!())?.clone();
        let src_mem = self
            .mem_of
            .get(s)
            .cloned()
            .flatten()
            .ok_or(crate::emitter_gap!())?;
        let n = if let Some(slot) = self.nrvo_slot(idx, &memty) {
            slot
        } else {
            let mut sizes = Vec::new();
            for (k, d) in shape.iter().enumerate() {
                if d == DYN_DIM {
                    let c = format!("%tcl{idx}_c{k}");
                    let v = format!("%tcl{idx}_d{k}");
                    let _ = writeln!(self.body, "  {c} = arith.constant {k} : index");
                    let _ = writeln!(self.body, "  {v} = memref.dim {src}, {c} : {src_mem}");
                    sizes.push(v);
                }
            }
            let n = format!("%v{idx}");
            let _ = writeln!(
                self.body,
                "  {n} = memref.alloc({}) : {memty}",
                sizes.join(", ")
            );
            n
        };
        let _ = writeln!(self.body, "  memref.copy {src}, {n} : {src_mem} to {memty}");
        self.names[idx] = n;
        self.mem_of[idx] = Some(memty);
        Ok(())
    }

    /// The run-time sizes of a tensor being allocated, as `index` values joined by commas. They
    /// are the last `dyn_count` `Arg`s before the instruction, one per `?` dimension in order.
    ///
    /// # Errors
    ///
    /// Declines when fewer than `dyn_count` `Arg`s are waiting, or one is not a scalar the
    /// emitter knows.
    fn alloc_extent_operands(&mut self, idx: usize, dyn_count: usize) -> Lowered<String> {
        if self.pending_args.len() < dyn_count {
            return Err(crate::emitter_gap!());
        }
        let extents = self
            .pending_args
            .split_off(self.pending_args.len() - dyn_count);
        let mut sizes = Vec::with_capacity(dyn_count);
        for (k, r) in extents.iter().enumerate() {
            let v = self
                .names
                .get(*r as usize)
                .ok_or(crate::emitter_gap!())?
                .clone();
            let vt = mlir_scalar(&self.elem_at(*r).ok_or(crate::emitter_gap!())?)
                .ok_or(crate::emitter_gap!())?;
            let ix = format!("%tai{idx}_{k}");
            let _ = writeln!(self.body, "  {ix} = arith.index_cast {v} : {vt} to index");
            sizes.push(ix);
        }
        Ok(sizes.join(", "))
    }

    /// Allocate a tensor buffer (`Tensor<T>([..])`): a static `memref` of the shape recovered
    /// from the side table by GID. Its register is tracked in `mem_of` for later index/store.
    ///
    /// # Errors
    ///
    /// Declines when the type is not a tensor the emitter knows, when a run-time size is
    /// missing, or when a tensor of run-time size is placed in shared memory.
    pub(crate) fn op_tensor_alloc(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let gid = *self
            .types
            .get(ins.type_idx.0 as usize)
            .ok_or(crate::emitter_gap!())?;
        let (elem, shape) = self.ctx.tensors.get(&gid).ok_or(crate::emitter_gap!())?;
        let memty = tensor_memref_ty(elem, shape).ok_or(crate::emitter_gap!())?;
        // This is the buffer the function returns, and the caller already allocated it: build in
        // place and allocate nothing. Named return value optimization, the same move C++ and Rust
        // make. `op_ret` then returns without copying, because the value already *is* the slot.
        if let Some(slot) = self.nrvo_slot(idx, &memty) {
            self.names[idx] = slot;
            self.mem_of[idx] = Some(memty);
            return Ok(());
        }
        let n = format!("%v{idx}");
        // A `?` dimension's extent is the `Arg` before this instruction, one per `?` in order:
        // `memref.alloc(%d0, %d1)` takes them as indices.
        let dyn_count = shape.iter().filter(|d| *d == DYN_DIM).count();
        let sizes = self.alloc_extent_operands(idx, dyn_count)?;
        // `operand2` may carry a memory-space dispatch id from the placement in the type (Vx#379
        // stage B). A `scope: sm` space becomes a space-3 ALLOCA: on the host that is a
        // stack slot like any other, and in the device clone materializeGpuKernels turns
        // a static space-3 alloca into `.shared` storage -- the machinery #352 built.
        // Any other space stays a plain allocation; the annotation was advisory.
        let sm = ins.operand2.0 != 0
            && self
                .ctx
                .subspaces
                .get(&u64::from(ins.operand2.0))
                .and_then(|s| s.scope.as_deref())
                == Some("sm");
        if sm {
            // Shared storage is packed by the driver from static sizes; a run-time one has none.
            if dyn_count > 0 {
                return Err(Decline::TypeNotModelled {
                    what: "a tile of run-time size in shared memory",
                });
            }
            let smty = format!(
                "{}, 3>",
                memty.strip_suffix('>').ok_or(crate::emitter_gap!())?
            );
            // alignment 16, explicitly. The alloca becomes a `.shared` global under
            // convert-gpu-to-nvvm, and the DRIVER packs those globals by their declared
            // alignment -- an unannotated 4-byte-aligned f32 array landed at offset
            // 0x204, and LLVM's loop vectorizer (which assumes natural vector alignment
            // when it widens a serial walk over the tile) issued a 16-byte
            // `ld.shared.v4` into it: "misaligned address", device-fatal, found by
            // compute-sanitizer on the block-per-row softmax (Vx#379 R3).
            // Entry block: a tensor declared inside a loop would otherwise take a
            // fresh stack slot per iteration and never give one back.
            self.emit_slot(&format!(
                "  {n} = memref.alloca() {{alignment = 16 : i64}} : {smty}\n"
            ));
            self.names[idx] = n;
            self.mem_of[idx] = Some(smty);
        } else {
            // A tensor placed on a device is allocated there. One that is filled is filled on
            // the host and copied there, since the host cannot write the device's memory.
            match super::super::placed_device(u64::from(ins.operand2.0), self.ctx) {
                Some(device) if !self.filled.contains(&idx) => {
                    let _ =
                        writeln!(self.body,
                        "  {n} = vx.alloc({sizes}) {{target_topology = {device} : i32}} : {memty}"
                    );
                }
                device => {
                    if let Some(device) = device {
                        self.copy_after_fill.insert(idx, device);
                    }
                    let _ = writeln!(self.body, "  {n} = memref.alloc({sizes}) : {memty}");
                }
            }
            self.names[idx] = n;
            self.mem_of[idx] = Some(memty);
        }
        Ok(())
    }

    /// After the fill of a tensor placed on a device: copy it there, and let the copy stand for it.
    fn copy_filled_to_device(&mut self, reg: usize, memty: &str) {
        let Some(device) = self.copy_after_fill.remove(&reg) else {
            return;
        };
        let host = self.names[reg].clone();
        let placed = format!("{host}_placed");
        let _ = writeln!(self.body,
            "  {placed} = \"vx.transfer\"({host}) <{{target_topology = {device} : i32}}> : ({memty}) -> {memty}"
        );
        self.names[reg] = placed;
    }

    /// The result tensor of a view-producing op, and the source's name, memref type and static
    /// sizes; a source with a run-time extent or a strided layout is not reinterpreted here.
    ///
    /// # Errors
    ///
    /// Declines when the result type or the source is not a tensor the emitter knows, or the
    /// source is a strided view.
    fn view_operands(&self, ins: &HirInstruction) -> Lowered<ViewSource> {
        let gid = *self
            .types
            .get(ins.type_idx.0 as usize)
            .ok_or(crate::emitter_gap!())?;
        let (elem, shape) = self
            .ctx
            .tensors
            .get(&gid)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let s = ins.operand1.0 as usize;
        let src = self.names.get(s).ok_or(crate::emitter_gap!())?.clone();
        let src_mem = self
            .mem_of
            .get(s)
            .cloned()
            .flatten()
            .ok_or(crate::emitter_gap!())?;
        if src_mem.contains("strided") {
            return Err(crate::emitter_gap!());
        }
        let sizes: Vec<i64> = src_mem
            .trim_start_matches("memref<")
            .split('x')
            .map_while(|t| t.parse().ok())
            .collect();
        Ok(ViewSource {
            elem,
            shape,
            src,
            src_mem,
            src_sizes: sizes,
        })
    }

    /// `TensorReshape`: the source buffer reinterpreted with the result's sizes and contiguous
    /// strides, the way the oracle lowers `reshape`.
    ///
    /// # Errors
    ///
    /// Declines when the source cannot be viewed (see `view_operands`) or the result has a
    /// run-time size.
    pub(crate) fn op_tensor_reshape(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let ViewSource {
            elem,
            shape,
            src,
            src_mem,
            ..
        } = self.view_operands(ins)?;
        let sizes: Vec<i64> = shape
            .iter()
            // The parse error is dropped: a size that is not a number is a run-time size.
            .map(|d| d.parse().ok().ok_or(crate::emitter_gap!()))
            .collect::<Lowered<_>>()?;
        let memty = tensor_memref_ty(&elem, &shape).ok_or(crate::emitter_gap!())?;
        let n = format!("%v{idx}");
        let _ = writeln!(self.body,
            "  {n} = memref.reinterpret_cast {src} to offset: [0], sizes: [{}], strides: [{}] : {src_mem} to {memty}",
            i64_list(&sizes),
            i64_list(&contiguous_strides(&sizes))
        );
        self.names[idx] = n;
        self.mem_of[idx] = Some(memty);
        Ok(())
    }

    /// `TensorTranspose`: a strided view with the axes permuted, copied into a fresh contiguous
    /// buffer of the result shape, the way the oracle lowers `transpose`.
    ///
    /// # Errors
    ///
    /// Declines when the source cannot be viewed (see `view_operands`), when its rank differs
    /// from the result's, or when the permutation names an axis the tensor does not have.
    pub(crate) fn op_tensor_transpose(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let ViewSource {
            elem,
            shape,
            src,
            src_mem,
            src_sizes,
        } = self.view_operands(ins)?;
        let rank = shape.len();
        if src_sizes.len() != rank {
            return Err(crate::emitter_gap!());
        }
        let src_strides = contiguous_strides(&src_sizes);
        let perm: Vec<usize> = (0..rank)
            .map(|i| ((ins.imm >> (4 * i)) & 0xF) as usize)
            .collect();
        if perm.iter().any(|&p| p >= rank) {
            return Err(crate::emitter_gap!());
        }
        let view_sizes: Vec<i64> = perm.iter().map(|&p| src_sizes[p]).collect();
        let view_strides: Vec<i64> = perm.iter().map(|&p| src_strides[p]).collect();
        let el = mlir_scalar(&elem).ok_or(crate::emitter_gap!())?;
        let dims: Vec<String> = view_sizes
            .iter()
            .map(std::string::ToString::to_string)
            .collect();
        let view_ty = format!(
            "memref<{}x{el}, strided<[{}], offset: 0>>",
            dims.join("x"),
            i64_list(&view_strides)
        );
        let memty = tensor_memref_ty(&elem, &shape).ok_or(crate::emitter_gap!())?;
        let v = format!("%tvw{idx}");
        let n = format!("%v{idx}");
        let _ = writeln!(self.body,
            "  {v} = memref.reinterpret_cast {src} to offset: [0], sizes: [{}], strides: [{}] : {src_mem} to {view_ty}",
            i64_list(&view_sizes),
            i64_list(&view_strides)
        );
        // A returned transpose copies out of the permuted view into the caller's buffer, which is
        // the one copy this op always needed -- it just no longer needs a buffer to make it into.
        let n = if let Some(slot) = self.nrvo_slot(idx, &memty) {
            slot
        } else {
            let _ = writeln!(self.body, "  {n} = memref.alloc() : {memty}");
            n
        };
        let _ = writeln!(self.body, "  memref.copy {v}, {n} : {view_ty} to {memty}");
        self.names[idx] = n;
        self.mem_of[idx] = Some(memty);
        Ok(())
    }

    /// The buffer an op's result goes in: the caller's slot when the result is returned straight
    /// into it, otherwise a fresh `memref.alloc` of `memty` named `%v{idx}`, with `sizes` as the
    /// operands for its run-time extents.
    fn result_buffer(&mut self, idx: usize, memty: &str, sizes: &[String]) -> String {
        if let Some(slot) = self.nrvo_slot(idx, memty) {
            return slot;
        }
        let n = format!("%v{idx}");
        let _ = writeln!(
            self.body,
            "  {n} = memref.alloc({}) : {memty}",
            sizes.join(", ")
        );
        n
    }

    /// The SSA name and memref type of the tensor in register `r`.
    ///
    /// # Errors
    ///
    /// Declines when `r` is not a register the emitter knows, or holds a value that is not a
    /// tensor.
    fn operand_memref(&self, r: usize) -> Lowered<(String, String)> {
        let name = self.names.get(r).ok_or(crate::emitter_gap!())?.clone();
        let memty = self
            .mem_of
            .get(r)
            .cloned()
            .flatten()
            .ok_or(crate::emitter_gap!())?;
        Ok((name, memty))
    }

    /// The target of `TensorPad`'s copy when each axis is padded: a view of the result buffer
    /// `dst` (its name and memref type) with the source's sizes and the result's strides, so
    /// the copy lands in the result's leading corner. Returns the view's name and type.
    ///
    /// # Errors
    ///
    /// Declines when the result has a run-time size or a different rank from the source.
    fn pad_each_axis_target(
        &mut self,
        idx: usize,
        view: &ViewSource,
        dst: (&str, &str),
        el: &str,
    ) -> Lowered<(String, String)> {
        let (dst, memty) = dst;
        let tgt_sizes: Vec<i64> = view
            .shape
            .iter()
            // The parse error is dropped: a size that is not a number is a run-time size.
            .map(|d| d.parse().ok().ok_or(crate::emitter_gap!()))
            .collect::<Lowered<_>>()?;
        if tgt_sizes.len() != view.src_sizes.len() {
            return Err(crate::emitter_gap!());
        }
        let strides = contiguous_strides(&tgt_sizes);
        let dims: Vec<String> = view
            .src_sizes
            .iter()
            .map(std::string::ToString::to_string)
            .collect();
        let corner_ty = format!(
            "memref<{}x{el}, strided<[{}], offset: 0>>",
            dims.join("x"),
            i64_list(&strides)
        );
        let _ = writeln!(self.body,
            "  %tpd{idx} = memref.reinterpret_cast {dst} to offset: [0], sizes: [{}], strides: [{}] : {memty} to {corner_ty}",
            i64_list(&view.src_sizes),
            i64_list(&strides)
        );
        Ok((format!("%tpd{idx}"), corner_ty))
    }

    /// `TensorPad`: a fresh buffer of the result shape filled with zeros, with the source copied
    /// into it, the way the oracle lowers a `PadMode::Pad` reshape and `pad`.
    ///
    /// # Errors
    ///
    /// Declines when the source cannot be viewed (see `view_operands`), or when padding each
    /// axis and the result has a run-time size or a different rank.
    pub(crate) fn op_tensor_pad(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let view = self.view_operands(ins)?;
        let n: i64 = view.src_sizes.iter().product();
        let el = mlir_scalar(&view.elem).ok_or(crate::emitter_gap!())?;
        let memty = tensor_memref_ty(&view.elem, &view.shape).ok_or(crate::emitter_gap!())?;
        let dst = self.result_buffer(idx, &memty, &[]);
        let zero = zero_literal(el);
        // The copy goes from `from` to `to`. Flat: both seen as `n` elements in a row. Each
        // axis: the source as it is, into a view of the new buffer with the source's sizes and
        // the new buffer's strides.
        let (from, from_ty, to, to_ty) = if ins.imm & crate::bytecode::TENSOR_PAD_EACH_AXIS != 0 {
            let (to, to_ty) = self.pad_each_axis_target(idx, &view, (&dst, &memty), el)?;
            (view.src, view.src_mem, to, to_ty)
        } else {
            let ViewSource { src, src_mem, .. } = view;
            let flat_ty = format!("memref<{n}x{el}>");
            let _ = writeln!(self.body,
                "  %tps{idx} = memref.reinterpret_cast {src} to offset: [0], sizes: [{n}], strides: [1] : {src_mem} to {flat_ty}"
            );
            let _ = writeln!(self.body,
                "  %tpd{idx} = memref.reinterpret_cast {dst} to offset: [0], sizes: [{n}], strides: [1] : {memty} to {flat_ty}"
            );
            (
                format!("%tps{idx}"),
                flat_ty.clone(),
                format!("%tpd{idx}"),
                flat_ty,
            )
        };
        let _ = writeln!(self.body, "  %z{idx} = arith.constant {zero} : {el}");
        let _ = writeln!(
            self.body,
            "  linalg.fill ins(%z{idx} : {el}) outs({dst} : {memty})"
        );
        let _ = writeln!(
            self.body,
            "  memref.copy {from}, {to} : {from_ty} to {to_ty}"
        );
        self.names[idx] = dst;
        self.mem_of[idx] = Some(memty);
        Ok(())
    }

    /// The closure a `TensorMap` or `TensorReduce` calls for each element: the value holding
    /// what the closure captured (`operand2`), and the name and signature of the function that
    /// runs it (the type table entry `imm` points at).
    ///
    /// # Errors
    ///
    /// Declines when the captured value or the function is not one the emitter knows.
    fn closure_adapter(&self, ins: &HirInstruction) -> Lowered<ClosureAdapter> {
        let env = self
            .names
            .get(ins.operand2.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let callee_gid = *usize::try_from(ins.imm)
            .ok()
            .and_then(|i| self.types.get(i))
            .ok_or(crate::emitter_gap!())?;
        let callee = self
            .ctx
            .callees
            .get(&callee_gid)
            .ok_or(crate::emitter_gap!())?;
        let (params, ret) = self
            .ctx
            .func_sigs
            .get(&callee_gid)
            .ok_or(crate::emitter_gap!())?;
        Ok(ClosureAdapter {
            env,
            name: callee.name.clone(),
            params: params.join(", "),
            ret: ret.clone(),
        })
    }

    /// `TensorMap`: a fresh buffer of the source's shape, each element the closure adapter
    /// applied to the source's, as a `linalg.generic` over the two, the way the oracle lowers
    /// `map`. A `?` extent is read off the source.
    ///
    /// # Errors
    ///
    /// Declines when the result type, the source or the closure is not one the emitter knows.
    pub(crate) fn op_tensor_map(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let gid = *self
            .types
            .get(ins.type_idx.0 as usize)
            .ok_or(crate::emitter_gap!())?;
        let (elem, shape) = self
            .ctx
            .tensors
            .get(&gid)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let memty = tensor_memref_ty(&elem, &shape).ok_or(crate::emitter_gap!())?;
        let el = mlir_scalar(&elem).ok_or(crate::emitter_gap!())?;
        let (src, src_mem) = self.operand_memref(ins.operand1.0 as usize)?;
        let ClosureAdapter {
            env,
            name,
            params,
            ret,
        } = self.closure_adapter(ins)?;
        let mut sizes: Vec<String> = Vec::new();
        for (k, d) in shape.iter().enumerate() {
            if d == DYN_DIM {
                let c = format!("%tmc{idx}_{k}");
                let sz = format!("%tms{idx}_{k}");
                let _ = writeln!(self.body, "  {c} = arith.constant {k} : index");
                let _ = writeln!(self.body, "  {sz} = memref.dim {src}, {c} : {src_mem}");
                sizes.push(sz);
            }
        }
        // A returned map writes its elements straight into the caller's buffer.
        let n = self.result_buffer(idx, &memty, &sizes);
        let dims: Vec<String> = (0..shape.len()).map(|i| format!("d{i}")).collect();
        let map = format!("affine_map<({0}) -> ({0})>", dims.join(", "));
        let iters: Vec<&str> = shape.iter().map(|_| "\"parallel\"").collect();
        let _ = writeln!(self.body,
            "  linalg.generic {{indexing_maps = [{map}, {map}], iterator_types = [{}]}} ins({src} : {src_mem}) outs({n} : {memty}) {{",
            iters.join(", ")
        );
        let _ = writeln!(self.body, "  ^bb0(%tmi{idx}: {el}, %tmo{idx}: {el}):");
        let _ = writeln!(
            self.body,
            "    %tmr{idx} = func.call {}({env}, %tmi{idx}) : ({params}) -> {ret}",
            sym_ref(&name)
        );
        let _ = write!(self.body, "    linalg.yield %tmr{idx} : {el}\n  }}\n");
        self.names[idx] = n;
        self.mem_of[idx] = Some(memty);
        Ok(())
    }

    /// `TensorReduce`: every element combined with the closure adapter into one scalar, starting
    /// from the preceding `Arg`. The accumulator is a rank-0 buffer the `linalg.generic` reduces
    /// into. `vx.reassoc` marks it for `vx-reorderable-reductions`, which lets the closure's
    /// arithmetic happen in any order, as `reduce` allows.
    ///
    /// # Errors
    ///
    /// Declines when the result is not a scalar, when the starting value is missing, or when
    /// the source or the closure is not one the emitter knows.
    pub(crate) fn op_tensor_reduce(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let result_gid = *self
            .types
            .get(ins.type_idx.0 as usize)
            .ok_or(crate::emitter_gap!())?;
        let elem = elem_of_gid(result_gid).ok_or(crate::emitter_gap!())?;
        let el = mlir_scalar(&elem).ok_or(crate::emitter_gap!())?;
        let init_reg = self.pending_args.pop().ok_or(crate::emitter_gap!())?;
        let init = self
            .names
            .get(init_reg as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let s = ins.operand1.0 as usize;
        let src = self.names.get(s).ok_or(crate::emitter_gap!())?.clone();
        let src_mem = self
            .mem_of
            .get(s)
            .cloned()
            .flatten()
            .ok_or(crate::emitter_gap!())?;
        let rank = super::aggregate::memref_rank(&src_mem).ok_or(crate::emitter_gap!())?;
        let ClosureAdapter {
            env,
            name,
            params,
            ret,
        } = self.closure_adapter(ins)?;
        let acc = format!("%tra{idx}");
        let _ = writeln!(self.body, "  {acc} = memref.alloca() : memref<{el}>");
        let _ = writeln!(self.body, "  memref.store {init}, {acc}[] : memref<{el}>");
        let dims: Vec<String> = (0..rank).map(|i| format!("d{i}")).collect();
        let in_map = format!("affine_map<({0}) -> ({0})>", dims.join(", "));
        let out_map = format!("affine_map<({}) -> ()>", dims.join(", "));
        let iters: Vec<&str> = (0..rank).map(|_| "\"reduction\"").collect();
        let _ = writeln!(self.body,
            "  linalg.generic {{indexing_maps = [{in_map}, {out_map}], iterator_types = [{}]}} ins({src} : {src_mem}) outs({acc} : memref<{el}>) attrs = {{vx.reassoc}} {{",
            iters.join(", ")
        );
        let _ = writeln!(self.body, "  ^bb0(%tri{idx}: {el}, %tro{idx}: {el}):");
        let _ = writeln!(
            self.body,
            "    %trr{idx} = func.call {}({env}, %tro{idx}, %tri{idx}) : ({params}) -> {ret}",
            sym_ref(&name)
        );
        let _ = write!(self.body, "    linalg.yield %trr{idx} : {el}\n  }}\n");
        let n = format!("%v{idx}");
        let _ = writeln!(self.body, "  {n} = memref.load {acc}[] : memref<{el}>");
        self.names[idx] = n;
        self.etypes[idx] = Some(elem);
        Ok(())
    }

    /// The sizes of a view over the caller's memory, one `i64` value per dimension of `shape`.
    /// A `?` size is the next waiting `Arg`, widened to `i64`; a static one is a constant.
    ///
    /// # Errors
    ///
    /// Declines when there are fewer waiting `Arg`s than `?` sizes, or one is not a scalar the
    /// emitter knows.
    fn view_extents(&mut self, idx: usize, shape: &[String]) -> Lowered<Vec<String>> {
        let dyn_count = shape.iter().filter(|d| *d == DYN_DIM).count();
        if self.pending_args.len() < dyn_count {
            return Err(crate::emitter_gap!());
        }
        let extents = self
            .pending_args
            .split_off(self.pending_args.len() - dyn_count);
        let mut ext = extents.into_iter();
        let mut dims_i64: Vec<String> = Vec::with_capacity(2);
        for (k, d) in shape.iter().enumerate() {
            let v = format!("%tv{idx}_{k}");
            if d == DYN_DIM {
                let reg = ext.next().ok_or(crate::emitter_gap!())?;
                let arg = self
                    .names
                    .get(reg as usize)
                    .ok_or(crate::emitter_gap!())?
                    .clone();
                let at = mlir_scalar(&self.elem_at(reg).ok_or(crate::emitter_gap!())?)
                    .ok_or(crate::emitter_gap!())?;
                if at == "i64" {
                    dims_i64.push(arg);
                    continue;
                }
                let _ = writeln!(self.body, "  {v} = arith.extsi {arg} : {at} to i64");
            } else {
                let _ = writeln!(self.body, "  {v} = arith.constant {d} : i64");
            }
            dims_i64.push(v);
        }
        Ok(dims_i64)
    }

    /// A rank-2 view over memory the caller owns (`tensor_view_2d(ptr, rows, cols)`): the memref
    /// descriptor built by hand over `operand1`'s pointer -- both pointers at the storage, a zero
    /// offset, the extents as sizes, row-major strides -- and cast into memref-typed IR the way
    /// the memref-to-LLVM conversion materializes one. A `?` extent is the `Arg` before this
    /// instruction; a static one is the type's. Nothing here checks the extents against the
    /// memory: that is the caller's claim, which is why the surface form needs `unsafe`.
    ///
    /// # Errors
    ///
    /// Declines when the result is not a rank-2 tensor the emitter knows, when `operand1` is not
    /// a pointer, or when a run-time size is missing.
    pub(crate) fn op_tensor_view(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let gid = *self
            .types
            .get(ins.type_idx.0 as usize)
            .ok_or(crate::emitter_gap!())?;
        let (elem, shape) = self
            .ctx
            .tensors
            .get(&gid)
            .ok_or(crate::emitter_gap!())?
            .clone();
        if shape.len() != 2 {
            return Err(crate::emitter_gap!());
        }
        let memty = tensor_memref_ty(&elem, &shape).ok_or(crate::emitter_gap!())?;
        let p = ins.operand1.0 as usize;
        if !self.ptr_of.get(p).copied().unwrap_or(false) {
            return Err(crate::emitter_gap!());
        }
        let ptr = self.names.get(p).ok_or(crate::emitter_gap!())?.clone();
        let dims_i64 = self.view_extents(idx, &shape)?;
        let zero = format!("%tvz{idx}");
        let one = format!("%tvo{idx}");
        let _ = writeln!(self.body, "  {zero} = arith.constant 0 : i64");
        let _ = writeln!(self.body, "  {one} = arith.constant 1 : i64");
        let dty = "!llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>";
        let mut d = format!("%tvd{idx}_0");
        let _ = writeln!(self.body, "  {d} = llvm.mlir.undef : {dty}");
        // Fields in order: allocated, aligned, offset, sizes[0..1], strides[0..1]. Row-major, so
        // the row stride is the column count and the column stride 1.
        let fields: [(&str, &str); 7] = [
            (&ptr, "0"),
            (&ptr, "1"),
            (&zero, "2"),
            (&dims_i64[0], "3, 0"),
            (&dims_i64[1], "3, 1"),
            (&dims_i64[1], "4, 0"),
            (&one, "4, 1"),
        ];
        for (k, (val, pos)) in fields.iter().enumerate() {
            let nd = format!("%tvd{idx}_{}", k + 1);
            let _ = writeln!(
                self.body,
                "  {nd} = llvm.insertvalue {val}, {d}[{pos}] : {dty}"
            );
            d = nd;
        }
        let n = format!("%v{idx}");
        let _ = writeln!(
            self.body,
            "  {n} = builtin.unrealized_conversion_cast {d} : {dty} to {memty}"
        );
        self.names[idx] = n;
        self.mem_of[idx] = Some(memty);
        Ok(())
    }

    /// Zero a freshly allocated tensor (`Tensor<T, [..]>::new()`): `linalg.fill` with a zero of
    /// the element type, which is the fill `MatmulInto` already emits before it accumulates.
    ///
    /// Integer elements are spelled with an integer zero rather than declined -- `linalg.fill`
    /// writes the value it is given, so there is no integer semantics to improvise here, unlike
    /// the multiply-accumulate `MatmulInto` guards against.
    ///
    /// # Errors
    ///
    /// Declines when `operand1` is not a tensor the emitter knows.
    pub(crate) fn op_tensor_zero(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let t = self
            .names
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let memty = self
            .mem_of
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone()
            .ok_or(crate::emitter_gap!())?;
        let et = memref_element(&memty).ok_or(crate::emitter_gap!())?;
        let zero = zero_literal(et);
        let _ = writeln!(self.body, "  %z{idx} = arith.constant {zero} : {et}");
        let _ = writeln!(
            self.body,
            "  linalg.fill ins(%z{idx} : {et}) outs({t} : {memty})"
        );
        self.copy_filled_to_device(ins.operand1.0 as usize, &memty);
        Ok(())
    }

    /// Fill a freshly allocated tensor with a value (`Tensor<T, [..]>::fill(v)`): `linalg.fill`
    /// with the value the source wrote, where `TensorZero` supplies a zero of its own.
    ///
    /// # Errors
    ///
    /// Declines when the tensor or the fill value is not one the emitter knows.
    pub(crate) fn op_tensor_fill(&mut self, _idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let t = self
            .names
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let memty = self
            .mem_of
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone()
            .ok_or(crate::emitter_gap!())?;
        let et = memref_element(&memty).ok_or(crate::emitter_gap!())?;
        let v = self
            .names
            .get(ins.operand2.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let _ = writeln!(
            self.body,
            "  linalg.fill ins({v} : {et}) outs({t} : {memty})"
        );
        self.copy_filled_to_device(ins.operand1.0 as usize, &memty);
        Ok(())
    }

    /// The runtime extent of one dimension: `memref.dim %t, %k` (`t.shape[k]`). The index
    /// operand arrives as a scalar and is cast to `index`; the result is cast back to `i32`,
    /// the type the checker gives the expression.
    ///
    /// # Errors
    ///
    /// Declines when the tensor or the index is not one the emitter knows.
    pub(crate) fn op_tensor_dim(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let t = self
            .names
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let memty = self
            .mem_of
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone()
            .ok_or(crate::emitter_gap!())?;
        let k = self
            .names
            .get(ins.operand2.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let kt = mlir_scalar(&self.elem_at(ins.operand2.0).ok_or(crate::emitter_gap!())?)
            .ok_or(crate::emitter_gap!())?;
        let ki = format!("%tdi{idx}");
        let d = format!("%tdd{idx}");
        let n = format!("%v{idx}");
        let _ = writeln!(self.body, "  {ki} = arith.index_cast {k} : {kt} to index");
        let _ = writeln!(self.body, "  {d} = memref.dim {t}, {ki} : {memty}");
        let _ = writeln!(self.body, "  {n} = arith.index_cast {d} : index to i32");
        self.names[idx] = n;
        self.etypes[idx] = Some(ElementType::I32);
        Ok(())
    }

    /// Read a rank-0 tensor's element: `memref.load %t[]`. Only rank-0 bases emit this
    /// (flatten's `read_rank0`); anything shaped is a wrong stream and declines.
    ///
    /// # Errors
    ///
    /// Declines when `operand1` is not a rank-0 tensor the emitter knows, or the result is not
    /// a scalar.
    pub(crate) fn op_tensor_load(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let src = self
            .names
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let memty = self
            .mem_of
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone()
            .ok_or(crate::emitter_gap!())?;
        let rank0 = memref_lead_dims_and_elem(&memty).is_some_and(|(d, _)| d.is_empty());
        if !rank0 {
            return Err(crate::emitter_gap!());
        }
        let e = elem_of_gid(
            *self
                .types
                .get(ins.type_idx.0 as usize)
                .ok_or(crate::emitter_gap!())?,
        )
        .ok_or(crate::emitter_gap!())?;
        let n = format!("%v{idx}");
        let _ = writeln!(self.body, "  {n} = memref.load {src}[] : {memty}");
        self.names[idx] = n;
        self.etypes[idx] = Some(e);
        Ok(())
    }

    /// Index a tensor along its outermost dimension. `operand1` is the base tensor (memref),
    /// `operand2` the index (`arith.index_cast` to `index`). A scalar-element result
    /// (`type_idx` is a scalar GID) is a value read (`imm = 0` → `memref.load`) or an element
    /// *place* (`imm = 1` → recorded for the following `TensorStore`). A sub-view result (a
    /// tensor GID) rank-reduces the base to a row via `memref.reinterpret_cast`.
    ///
    /// # Errors
    ///
    /// Declines when the base or the index is not a value the emitter knows, or when the row
    /// cannot be viewed (see `row_view`).
    pub(crate) fn op_tensor_index(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let result_gid = *self
            .types
            .get(ins.type_idx.0 as usize)
            .ok_or(crate::emitter_gap!())?;
        let base = self
            .names
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        if self.vector_element(idx, ins, result_gid, &base)? {
            return Ok(());
        }
        let base_memty = self
            .mem_of
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone()
            .ok_or(crate::emitter_gap!())?;
        let imt = mlir_scalar(&self.elem_at(ins.operand2.0).ok_or(crate::emitter_gap!())?)
            .ok_or(crate::emitter_gap!())?;
        let iname = self
            .names
            .get(ins.operand2.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let ic = format!("%ic{idx}");
        let _ = writeln!(
            self.body,
            "  {ic} = arith.index_cast {iname} : {imt} to index"
        );

        if let Some(e) = elem_of_gid(result_gid) {
            // Scalar element: a value read or a store place (works on a contiguous or a
            // strided-row base — `memref.load`/`store` handle both).
            if ins.imm == 1 {
                self.place_of[idx] = Some((base, ic, base_memty));
            } else {
                let n = format!("%v{idx}");
                let _ = writeln!(self.body, "  {n} = memref.load {base}[{ic}] : {base_memty}");
                self.names[idx] = n;
                self.etypes[idx] = Some(e);
            }
            return Ok(());
        }
        let at = IndexedBase {
            base: &base,
            memty: &base_memty,
            index: &ic,
        };
        self.row_view(idx, &at, result_gid)
    }

    /// Read one element out of a rank-1 elementwise result, which is a vector value and not a
    /// buffer. Returns `false`, having emitted nothing, when `operand1` is not such a vector,
    /// the result is not a scalar, or the instruction asks for a place to store into.
    ///
    /// # Errors
    ///
    /// Declines when the index is not a scalar the emitter knows.
    fn vector_element(
        &mut self,
        idx: usize,
        ins: &HirInstruction,
        result_gid: TypeId,
        base: &str,
    ) -> Lowered<bool> {
        // A rank-1 elementwise result is a vector value, not a buffer: read an element
        // straight out of it. There is nothing to store into.
        let (Some(Some(vecty)), Some(e), false) = (
            self.vec_of.get(ins.operand1.0 as usize).cloned(),
            elem_of_gid(result_gid),
            ins.imm == 1,
        ) else {
            return Ok(false);
        };
        let imt = mlir_scalar(&self.elem_at(ins.operand2.0).ok_or(crate::emitter_gap!())?)
            .ok_or(crate::emitter_gap!())?;
        let et = mlir_scalar(&e).ok_or(crate::emitter_gap!())?;
        let iname = self
            .names
            .get(ins.operand2.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let ic = format!("%ic{idx}");
        let n = format!("%v{idx}");
        let _ = writeln!(
            self.body,
            "  {ic} = arith.index_cast {iname} : {imt} to index"
        );
        let _ = writeln!(
            self.body,
            "  {n} = vector.extract {base}[{ic}] : {et} from {vecty}"
        );
        self.names[idx] = n;
        self.etypes[idx] = Some(e);
        Ok(true)
    }

    /// The row of a tensor at index `at.index`, as a view of the base: the contiguous base
    /// reinterpreted as the row at flat offset `index * product(row dims)`, with row-major
    /// strides over the remaining dims. `result_gid` is the row's own tensor type.
    ///
    /// # Errors
    ///
    /// Declines when the row's type is not a tensor the emitter knows, when a size is neither
    /// a number nor `?`, or when a strided base cannot be indexed (see `strided_row`).
    fn row_view(&mut self, idx: usize, at: &IndexedBase<'_>, result_gid: TypeId) -> Lowered<()> {
        let IndexedBase {
            base,
            memty: base_memty,
            index: ic,
        } = *at;
        // A row of a row: the base is already a strided view, so its own offset, sizes and
        // strides come back through `memref.extract_strided_metadata`, and the next row is
        // that offset plus the index times the leading stride.
        if base_memty.contains("strided") {
            return self.strided_row(idx, base, base_memty, ic);
        }
        let (elem, shape) = self
            .ctx
            .tensors
            .get(&result_gid)
            .ok_or(crate::emitter_gap!())?;
        let et = mlir_scalar(elem).ok_or(crate::emitter_gap!())?;
        // A row with a run-time extent anywhere: each `?` size is read off the base with
        // `memref.dim`, the strides are the row-major products, and the offset is the index
        // times the row's element count. A static position stays a literal in the type.
        if shape.iter().any(|d| d == DYN_DIM) {
            let et = et.to_string();
            let shape = shape.clone();
            self.dynamic_row(idx, at, &et, &shape);
            return Ok(());
        }
        let dims: Vec<i64> = shape
            .iter()
            .map(|d| d.parse::<i64>().ok())
            .collect::<Option<_>>()
            .ok_or(Decline::TypeNotModelled {
                what: "a row whose extents are not all literals",
            })?;
        let row_len: i64 = dims.iter().product();
        let strides = contiguous_strides(&dims);
        let off = if row_len == 1 {
            ic.to_string()
        } else {
            let cst = format!("%cs{idx}");
            let o = format!("%off{idx}");
            let _ = writeln!(self.body, "  {cst} = arith.constant {row_len} : index");
            let _ = writeln!(self.body, "  {o} = arith.muli {ic}, {cst} : index");
            o
        };
        let sizes_s = join_i64(&dims);
        let strides_s = join_i64(&strides);
        let dims_x = dims_prefix(&dims);
        // A sub-view of shared storage stays in its space: dropping the `, 3` here
        // would make the row a generic pointer and the PTX would address `.shared`
        // data with global loads.
        let space_sfx = if base_memty.ends_with(", 3>") {
            ", 3"
        } else {
            ""
        };
        let result_ty =
            format!("memref<{dims_x}{et}, strided<[{strides_s}], offset: ?>{space_sfx}>");
        let n = format!("%v{idx}");
        let _ = writeln!(
            self.body,
            "  {n} = memref.reinterpret_cast {base} to offset: [{off}], sizes: [{sizes_s}], strides: [{strides_s}] : {base_memty} to {result_ty}"
        );
        self.names[idx] = n;
        self.mem_of[idx] = Some(result_ty);
        Ok(())
    }

    /// The row of a dynamically shaped tensor. A `?` extent is not a number to compute with,
    /// so it is read off the base with `memref.dim` and the flat offset is computed against
    /// that value. `et` is the element type and `shape` the row's sizes.
    fn dynamic_row(&mut self, idx: usize, at: &IndexedBase<'_>, et: &str, shape: &[String]) {
        let IndexedBase {
            base,
            memty: base_memty,
            index: ic,
        } = *at;
        // Sizes: a literal stays one; a `?` is `memref.dim` of the base at that position, which
        // is one deeper than in the row (the outermost dimension is the one being indexed).
        let mut sizes: Vec<String> = Vec::with_capacity(shape.len());
        for (k, d) in shape.iter().enumerate() {
            if d == DYN_DIM {
                let c = format!("%rdc{idx}_{k}");
                let s = format!("%rds{idx}_{k}");
                let _ = writeln!(self.body, "  {c} = arith.constant {} : index", k + 1);
                let _ = writeln!(self.body, "  {s} = memref.dim {base}, {c} : {base_memty}");
                sizes.push(s);
            } else {
                sizes.push(d.clone());
            }
        }
        // Strides, row-major from the back: the last is 1, each other is the next stride
        // times the next size. A product of literals stays a literal.
        let r = shape.len();
        let mut strides: Vec<String> = vec!["1".to_string(); r];
        for k in (0..r.saturating_sub(1)).rev() {
            let (a, b) = (strides[k + 1].clone(), sizes[k + 1].clone());
            strides[k] = self.mul_extent(&a, &b, &format!("%rdt{idx}_{k}"));
        }
        // Offset: the index times the row's element count.
        let count = self.mul_extent(&strides[0], &sizes[0], &format!("%rdn{idx}"));
        let off = self.mul_extent(ic, &count, &format!("%rdo{idx}"));
        let as_ty = |v: &String| {
            if v.starts_with('%') {
                "?".to_string()
            } else {
                v.clone()
            }
        };
        let dims = dims_prefix(sizes.iter().map(as_ty));
        let stride_tys: Vec<String> = strides.iter().map(as_ty).collect();
        let result_ty = format!(
            "memref<{dims}{et}, strided<[{}], offset: ?>>",
            stride_tys.join(", ")
        );
        let n = format!("%v{idx}");
        let _ = writeln!(
            self.body,
            "  {n} = memref.reinterpret_cast {base} to offset: [{off}], sizes: [{}], strides: [{}] : {base_memty} to {result_ty}",
            sizes.join(", "),
            strides.join(", ")
        );
        self.names[idx] = n;
        self.mem_of[idx] = Some(result_ty);
    }

    /// Index a strided row along its outermost dimension. The base type spells its sizes and
    /// strides (`memref<AxBx..xet, strided<[s0, s1, ..], offset: ?>>`), so the result's are
    /// those with the first dropped, a literal staying a literal and a `?` taking the value
    /// `extract_strided_metadata` hands back; the offset is the base's plus `index * s0`.
    ///
    /// # Errors
    ///
    /// Declines when the base type does not spell at least two dimensions and one stride for
    /// each.
    fn strided_row(&mut self, idx: usize, base: &str, base_memty: &str, ic: &str) -> Lowered<()> {
        let (dims_x, et) = memref_lead_dims_and_elem(base_memty).ok_or(crate::emitter_gap!())?;
        let dims: Vec<&str> = dims_x.split('x').filter(|d| !d.is_empty()).collect();
        let strides_txt = base_memty
            .split("strided<[")
            .nth(1)
            .and_then(|s| s.split(']').next())
            .ok_or(crate::emitter_gap!())?;
        let strides: Vec<&str> = strides_txt.split(',').map(str::trim).collect();
        let rank = dims.len();
        if rank < 2 || strides.len() != rank {
            return Err(crate::emitter_gap!());
        }
        let space_sfx = if base_memty.ends_with(", 3>") {
            ", 3"
        } else {
            ""
        };
        let buf = format!("%srb{idx}");
        let off = format!("%sro{idx}");
        let sizes: Vec<String> = (0..rank).map(|k| format!("%srs{idx}_{k}")).collect();
        let strs: Vec<String> = (0..rank).map(|k| format!("%srt{idx}_{k}")).collect();
        let results = [vec![buf.clone(), off.clone()], sizes.clone(), strs.clone()]
            .concat()
            .join(", ");
        let index_tys = vec!["index"; 1 + 2 * rank].join(", ");
        let _ = writeln!(self.body,
            "  {results} = memref.extract_strided_metadata {base} : {base_memty} -> memref<{et}{space_sfx}>, {index_tys}"
        );
        let step = format!("%srm{idx}");
        let new_off = format!("%srn{idx}");
        let _ = writeln!(self.body, "  {step} = arith.muli {ic}, {} : index", strs[0]);
        let _ = writeln!(self.body, "  {new_off} = arith.addi {off}, {step} : index");
        let pick = |txt: &str, ssa: &String| {
            if txt == "?" {
                ssa.clone()
            } else {
                txt.to_string()
            }
        };
        let size_ops: Vec<String> = (1..rank).map(|k| pick(dims[k], &sizes[k])).collect();
        let stride_ops: Vec<String> = (1..rank).map(|k| pick(strides[k], &strs[k])).collect();
        let result_ty = format!(
            "memref<{}{et}, strided<[{}], offset: ?>{space_sfx}>",
            dims_prefix(&dims[1..]),
            strides[1..].join(", ")
        );
        let n = format!("%v{idx}");
        let _ = writeln!(self.body,
            "  {n} = memref.reinterpret_cast {buf} to offset: [{new_off}], sizes: [{}], strides: [{}] : memref<{et}{space_sfx}> to {result_ty}",
            size_ops.join(", "),
            stride_ops.join(", ")
        );
        self.names[idx] = n;
        self.mem_of[idx] = Some(result_ty);
        Ok(())
    }

    /// The product of two extents, each a literal or an `index` SSA name: a literal when both
    /// are, otherwise an `arith.muli` with any literal operand materialized first.
    fn mul_extent(&mut self, a: &str, b: &str, name: &str) -> String {
        if let (Ok(x), Ok(y)) = (a.parse::<i64>(), b.parse::<i64>()) {
            return (x * y).to_string();
        }
        // A unit factor contributes nothing; a rank-1 row's element count is its width.
        if a == "1" {
            return b.to_string();
        }
        if b == "1" {
            return a.to_string();
        }
        let operand = |v: &str, tag: &str, body: &mut String| -> String {
            if v.starts_with('%') {
                v.to_string()
            } else {
                let c = format!("{name}{tag}");
                let _ = writeln!(body, "  {c} = arith.constant {v} : index");
                c
            }
        };
        let a = operand(a, "a", &mut self.body);
        let b = operand(b, "b", &mut self.body);
        let _ = writeln!(self.body, "  {name} = arith.muli {a}, {b} : index");
        name.to_string()
    }

    /// `a @ b`: a matmul producing a fresh tensor, as against `MatmulInto`'s write into a buffer
    /// the caller owns. `type_idx` is the result's own GID -- the flattener sizes it `[m, n]` from
    /// the two operands -- so the destination is allocated here and then filled by the same pair of
    /// ops `MatmulInto` emits.
    ///
    /// # Errors
    ///
    /// Declines when an operand or the result type is not a tensor the emitter knows, or the
    /// element type is not a float.
    pub(crate) fn op_matmul(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let (a, ma) = self.operand_memref(ins.operand1.0 as usize)?;
        let (b, mb) = self.operand_memref(ins.operand2.0 as usize)?;
        let gid = *self
            .types
            .get(ins.type_idx.0 as usize)
            .ok_or(crate::emitter_gap!())?;
        let (elem, shape) = self.ctx.tensors.get(&gid).ok_or(crate::emitter_gap!())?;
        let (elem, shape) = (elem.clone(), shape.clone());
        let md = tensor_memref_ty(&elem, &shape).ok_or(crate::emitter_gap!())?;
        // Floats only, on the same terms as `MatmulInto`: linalg's integer semantics are not
        // improvised here.
        let et = mlir_scalar(&elem).ok_or(crate::emitter_gap!())?;
        if !matches!(et, "f32" | "f64" | "f16" | "bf16") {
            return Err(crate::emitter_gap!());
        }
        // A `?` extent of the result is read off the operand it comes from: the rows of the
        // left operand, the columns of the right.
        let mut sizes: Vec<String> = Vec::new();
        for (k, d) in shape.iter().enumerate() {
            if d == DYN_DIM {
                let (src, sm) = if k == 0 { (&a, &ma) } else { (&b, &mb) };
                let c = format!("%mmc{idx}_{k}");
                let sz = format!("%mms{idx}_{k}");
                let _ = writeln!(self.body, "  {c} = arith.constant {k} : index");
                let _ = writeln!(self.body, "  {sz} = memref.dim {src}, {c} : {sm}");
                sizes.push(sz);
            }
        }
        // A returned product accumulates straight into the caller's buffer -- the biggest of the
        // buffers this saves, since a matmul result is the whole output matrix.
        let n = self.result_buffer(idx, &md, &sizes);
        let _ = writeln!(self.body, "  %mz{idx} = arith.constant 0.0 : {et}");
        let _ = writeln!(
            self.body,
            "  linalg.fill ins(%mz{idx} : {et}) outs({n} : {md})"
        );
        let _ = writeln!(
            self.body,
            "  linalg.matmul ins({a}, {b} : {ma}, {mb}) outs({n} : {md})"
        );
        self.names[idx] = n;
        self.mem_of[idx] = Some(md);
        Ok(())
    }

    /// `matmul_into(&mut dst, &a, &b)` (no result): `linalg.fill` + `linalg.matmul` on the
    /// three whole-tensor memrefs, the same pair the AST path builds -- and the exact shape
    /// `kernelKindOf` classifies, so a spawn whose whole job is this op still routes to
    /// cuBLAS. The destination register rides the imm (see `Opcode::MatmulInto`).
    ///
    /// # Errors
    ///
    /// Declines when an operand or the destination is not a tensor the emitter knows, or the
    /// element type is not a float.
    pub(crate) fn op_matmul_into(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let a = self
            .names
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let ma = self
            .mem_of
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone()
            .ok_or(crate::emitter_gap!())?;
        let b = self
            .names
            .get(ins.operand2.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let mb = self
            .mem_of
            .get(ins.operand2.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone()
            .ok_or(crate::emitter_gap!())?;
        // The destination's register, which the instruction carries in `imm`.
        let dst_reg = usize::try_from(ins.imm).ok().ok_or(crate::emitter_gap!())?;
        let dst = self
            .names
            .get(dst_reg)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let md = self
            .mem_of
            .get(dst_reg)
            .ok_or(crate::emitter_gap!())?
            .clone()
            .ok_or(crate::emitter_gap!())?;
        // Floats only -- an int matmul declines the program to the AST path rather than
        // improvising linalg's integer semantics here. The half-precision pair is in (Vx#320):
        // the routed backend runs them through cublasGemmEx with f32 accumulation, and the host
        // fallback's linalg lowers them like any float -- restricting to f32 here silently
        // evicted every f16 program from the flat path, prover and all.
        let et = memref_element(&md).ok_or(crate::emitter_gap!())?;
        if et != "f32" && et != "f64" && et != "f16" && et != "bf16" {
            return Err(crate::emitter_gap!());
        }
        let _ = writeln!(self.body, "  %mz{idx} = arith.constant 0.0 : {et}");
        let _ = writeln!(
            self.body,
            "  linalg.fill ins(%mz{idx} : {et}) outs({dst} : {md})"
        );
        let _ = writeln!(
            self.body,
            "  linalg.matmul ins({a}, {b} : {ma}, {mb}) outs({dst} : {md})"
        );
        Ok(())
    }

    /// Store into a tensor place (no result). A scalar-element place (an `imm = 1`
    /// `TensorIndex`) → `memref.store`; a row/sub-view place (an `imm = 0` `TensorIndex`, a row
    /// memref in `mem_of`) takes an elementwise vector value → `vector.store`.
    ///
    /// # Errors
    ///
    /// Declines when `operand1` is neither an element place nor a tensor, or when the stored
    /// value does not fit it (see `store_element` and `store_row`).
    pub(crate) fn op_tensor_store(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        if let Some(place) = self
            .place_of
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone()
        {
            self.store_element(idx, ins, place)
        } else if let Some(rowty) = self
            .mem_of
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone()
        {
            self.store_row(idx, ins, &rowty)
        } else {
            Err(crate::emitter_gap!())
        }
    }

    /// Store the scalar `operand2` into one element of a tensor. `place` is the tensor's name,
    /// the index and the tensor's memref type, as `TensorIndex` recorded them.
    ///
    /// # Errors
    ///
    /// Declines when the value is not one the emitter knows, or has no modelled conversion to
    /// the tensor's element type.
    fn store_element(
        &mut self,
        idx: usize,
        ins: &HirInstruction,
        place: (String, String, String),
    ) -> Lowered<()> {
        let (base, ic, memty) = place;
        let vreg = ins.operand2.0;
        let mut val = self
            .names
            .get(vreg as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        // Coerce the stored scalar to the tensor's element type when they differ — a
        // default-`f32` float literal `1.0` stored into a `bf16` tensor becomes
        // `arith.truncf`, `f32 -> f64` becomes `arith.extf`, etc. The AST path does the
        // same via `coerce_type` before its `memref.store`; without it the store is
        // ill-typed (`f32` value into a `memref<..xbf16>`).
        if let (Some(src_e), Some(tgt_s)) = (self.elem_at(vreg), memref_elem(&memty)) {
            if let Some(tgt_e) = elem_from_mlir_scalar(tgt_s) {
                match cast_op(&src_e, &tgt_e) {
                    Some("") => {} // same MLIR type: no conversion
                    Some(op) => {
                        let c = format!("%tsc{idx}");
                        let _ = writeln!(
                            self.body,
                            "  {c} = {op} {val} : {} to {tgt_s}",
                            mlir_scalar(&src_e).ok_or(crate::emitter_gap!())?
                        );
                        val = c;
                    }
                    None => return Err(crate::emitter_gap!()), // unmodelled conversion -> decline (AST oracle)
                }
            }
        }
        let _ = writeln!(self.body, "  memref.store {val}, {base}[{ic}] : {memty}");
        Ok(())
    }

    /// Store the scalar in register `vreg` into the rank-0 tensor `dst` of memref type `rowty`.
    ///
    /// # Errors
    ///
    /// Declines when the scalar is not a value the emitter knows, or its type differs from the
    /// tensor's element type.
    fn store_rank0(&mut self, vreg: u32, dst: &str, rowty: &str) -> Lowered<()> {
        let val = self
            .names
            .get(vreg as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let same_elem = match (self.elem_at(vreg), memref_elem(rowty)) {
            (Some(src_e), Some(tgt_s)) => mlir_scalar(&src_e) == Some(tgt_s),
            _ => false,
        };
        if !same_elem {
            return Err(crate::emitter_gap!());
        }
        let _ = writeln!(self.body, "  memref.store {val}, {dst}[] : {rowty}");
        Ok(())
    }

    /// Store `operand2` into the whole tensor `operand1`, whose memref type is `rowty`: a
    /// scalar into a rank-0 tensor, or a vector or a rank-1 tensor into a row.
    ///
    /// # Errors
    ///
    /// Declines when the value is not one the emitter knows, when a scalar's type differs from
    /// the rank-0 tensor's element, or when the value is neither a vector nor a tensor.
    fn store_row(&mut self, idx: usize, ins: &HirInstruction, rowty: &str) -> Lowered<()> {
        let dst = self
            .names
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        // A rank-0 tensor (`memref<el>`, no dims): the stored value is the scalar itself,
        // written with an empty index list. Elements are identical by the checker's
        // scalar-into-tensor rule (Vx#396); anything else declines.
        if memref_lead_dims_and_elem(rowty).is_some_and(|(d, _)| d.is_empty()) {
            return self.store_rank0(ins.operand2.0, &dst, rowty);
        }
        let vecname = self
            .names
            .get(ins.operand2.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let vec_rec = self
            .vec_of
            .get(ins.operand2.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let c0 = format!("%sc{idx}");
        if let Some(vecty) = vec_rec {
            let al = vector_align_attr(&vecty);
            let _ = writeln!(self.body, "  {c0} = arith.constant 0 : index");
            let _ = writeln!(
                self.body,
                "  vector.store {vecname}, {dst}[{c0}]{al} : {rowty}, {vecty}"
            );
        } else if let Some(srcty) = self
            .mem_of
            .get(ins.operand2.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone()
        {
            // The stored value is a rank-1 tensor, not a vector -- `m[0] = [10.0, ..]`, a row
            // written from an array literal's buffer. Read the whole source row as a vector
            // and store it; the checker already matched the two lengths.
            let d = memref_lead_dim(&srcty).ok_or(crate::emitter_gap!())?;
            let et = memref_elem(&srcty).ok_or(crate::emitter_gap!())?;
            let vecty = format!("vector<{d}x{et}>");
            let al = vector_align_attr(&vecty);
            let v = format!("%svl{idx}");
            let _ = writeln!(self.body, "  {c0} = arith.constant 0 : index");
            let _ = writeln!(
                self.body,
                "  {v} = vector.load {vecname}[{c0}]{al} : {srcty}, {vecty}"
            );
            let _ = writeln!(
                self.body,
                "  vector.store {v}, {dst}[{c0}]{al} : {rowty}, {vecty}"
            );
        } else {
            return Err(crate::emitter_gap!());
        }
        Ok(())
    }
}

/// What a view-producing op reinterprets: the result tensor, and the source's name, memref type
/// and static sizes.
struct ViewSource {
    elem: ElementType,
    shape: Vec<String>,
    src: String,
    src_mem: String,
    src_sizes: Vec<i64>,
}

/// A tensor being indexed along its outermost dimension.
struct IndexedBase<'a> {
    /// The SSA name of the tensor.
    base: &'a str,
    /// The tensor's memref type.
    memty: &'a str,
    /// The SSA name of the index, already cast to `index`.
    index: &'a str,
}

/// The sizes as a memref type spells them, each followed by `x`: `4x8x`.
fn dims_prefix<T: std::fmt::Display>(dims: impl IntoIterator<Item = T>) -> String {
    dims.into_iter().fold(String::new(), |mut acc, d| {
        let _ = write!(acc, "{d}x");
        acc
    })
}

/// The closure a `TensorMap` or `TensorReduce` calls for each element.
struct ClosureAdapter {
    /// The SSA name of the value holding what the closure captured.
    env: String,
    /// The name of the function that runs the closure.
    name: String,
    /// That function's parameter types, joined by commas.
    params: String,
    /// That function's return type.
    ret: String,
}

/// How MLIR spells zero for the scalar type `el`: `0.0` for a float, `0` for an integer.
fn zero_literal(el: &str) -> &'static str {
    if el.starts_with('f') || el.starts_with("bf") {
        "0.0"
    } else {
        "0"
    }
}

/// Row-major strides for static sizes: the last axis is unit.
fn contiguous_strides(sizes: &[i64]) -> Vec<i64> {
    let mut strides = vec![1i64; sizes.len()];
    for i in (0..sizes.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * sizes[i + 1];
    }
    strides
}

fn i64_list(v: &[i64]) -> String {
    v.iter()
        .map(std::string::ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}
