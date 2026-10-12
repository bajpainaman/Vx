//! The parallel constructs: memory transfers, spawn regions and barriers.
//!
//! One `impl` block on the per-function emitter; the dispatch that reaches these
//! lives in `step`, next to the struct.

use super::super::{
    mlir_scalar, static_tile_bytes, tensor_memref_ty, ElementType, FnEmit, HirInstruction, Lowered,
};
use std::fmt::Write as _;

impl FnEmit<'_> {
    /// Move a tensor to a memory space: `operand1` is the source, the low half of `imm` the target
    /// space's dispatch id and the high half the source's topology, `type_idx` the result tensor
    /// (same element + shape). Emits `vx.transfer`
    /// (generic form) with `target_topology`; the vx→standard lowering turns it into an
    /// alloc + `memref.copy` (it ignores the source's layout suffix, so the result is a plain
    /// `memref<NxT>`). When the target space declares a sub-space descriptor, re-attach the
    /// scheduling attrs (`space`/`within`/`granule`/`capacity`/`scope` + a bump-allocated
    /// `offset`/`slots`) the AST path emits — a device backend needs them to place the tile
    /// into VMEM/TMEM, and they are dropped otherwise.
    ///
    /// # Errors
    ///
    /// Declines when the source is not a value the emitter knows, or the result type is not a
    /// tensor it knows.
    pub(crate) fn op_transfer(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let src = self
            .names
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        // A scalar operand: the op restates its type, and lowering folds it to the operand.
        // A tensor: the result is the transferred tile's memref.
        let (srcty, dstty, elem, shape) = if let Some(e) = self.elem_at(ins.operand1.0) {
            let t = mlir_scalar(&e).ok_or(crate::emitter_gap!())?.to_string();
            self.etypes[idx] = Some(e);
            (t.clone(), t, None, None)
        } else {
            let result_gid = *self
                .types
                .get(ins.type_idx.0 as usize)
                .ok_or(crate::emitter_gap!())?;
            let (elem, shape) = self
                .ctx
                .tensors
                .get(&result_gid)
                .ok_or(crate::emitter_gap!())?;
            let dstty = tensor_memref_ty(elem, shape).ok_or(crate::emitter_gap!())?;
            let srcty = self
                .mem_of
                .get(ins.operand1.0 as usize)
                .ok_or(crate::emitter_gap!())?
                .clone()
                .ok_or(crate::emitter_gap!())?;
            (srcty, dstty, Some(elem.clone()), Some(shape.clone()))
        };
        let n = format!("%v{idx}");
        let attrs = self.transfer_attrs(ins.imm, elem.as_ref().zip(shape.as_deref()));
        let _ = writeln!(
            self.body,
            "  {n} = \"vx.transfer\"({src}) {{{attrs}}} : ({srcty}) -> {dstty}"
        );
        self.names[idx] = n;
        if elem.is_some() {
            self.mem_of[idx] = Some(dstty);
        }
        Ok(())
    }

    /// The attributes of a `vx.transfer`. The low half of `imm` is the target space's dispatch
    /// id and the high half the source's topology. `tile` is the element type and shape of a
    /// transferred tensor; a tile of static size in a space with a granule claims the next
    /// free offset in that space.
    fn transfer_attrs(&mut self, imm: u64, tile: Option<(&ElementType, &[String])>) -> String {
        let space_id = imm & 0xffff_ffff;
        let source = imm >> 32;
        // A space declaring `node: N` is dispatched by its NUMA band id rather than by the hash
        // of its name. The hash says which space; the backend has to know which NODE, and cannot
        // recover a node number from a hash. Copied out rather than held, so the descriptor's
        // borrow ends before the bump allocation below takes `&mut self`.
        let numa_target = self
            .ctx
            .subspaces
            .get(&space_id)
            .and_then(|d| d.numa_node)
            .and_then(crate::arch::numa_dispatch_id);
        // A NUMA dispatch id is never negative, so `unsigned_abs` is the id itself.
        let target = numa_target.map_or(space_id, |i| u64::from(i.unsigned_abs()));
        let mut attrs = format!("target_topology = {target} : i32");
        if source != 0 {
            let _ = write!(attrs, ", source_topology = {source} : i32");
        }
        if let Some(desc) = self.ctx.subspaces.get(&space_id) {
            // Descriptor attrs, in the AST path's emission order (MLIR sorts on print, so the
            // final parsed form is byte-identical regardless of the order emitted here).
            let _ = write!(attrs, ", space = \"{}\"", desc.name);
            if let Some(w) = &desc.within {
                let _ = write!(attrs, ", within = \"{w}\"");
            }
            if let Some(g) = desc.granule {
                let _ = write!(attrs, ", granule = {g} : i64");
            }
            if let Some(c) = desc.capacity {
                let _ = write!(attrs, ", capacity = {c} : i64");
            }
            if let Some(s) = &desc.scope {
                let _ = write!(attrs, ", scope = \"{s}\"");
            }
            if let Some(m) = &desc.managed {
                let _ = write!(attrs, ", managed = \"{m}\"");
            }
            // SS2 bump allocation: a statically-shaped tile into a granule'd space claims the
            // next granule-rounded `offset`; `slots` is the granule count it occupies.
            let tile_bytes = tile.and_then(|(e, s)| static_tile_bytes(e, s));
            if let (Some(granule), Some(bytes)) = (desc.granule, tile_bytes) {
                if granule > 0 {
                    let rounded = bytes.div_ceil(granule) * granule;
                    let offset = *self.subspace_offsets.entry(space_id).or_insert(0);
                    self.subspace_offsets.insert(space_id, offset + rounded);
                    let _ = write!(attrs, ", offset = {offset} : i64");
                    let _ = write!(attrs, ", slots = {} : i64", rounded / granule);
                }
            }
        } else if let Some(name) = crate::arch::builtin_memory_space_name(space_id) {
            // A built-in space declares nothing, but its name still goes on the op.
            let _ = write!(attrs, ", space = \"{name}\"");
        }
        attrs
    }

    /// Open a `vx.spawn` region (generic form). The op is inline in the enclosing block, which
    /// continues after it; the instructions up to the matching `SpawnEnd` form the region body.
    /// The ops immediately after `Spawn` (a control-flow body's setup, before its first explicit
    /// block) go in the region's entry block, so open a label for it. The low half of `imm` is the
    /// topology dispatch id (the AST path's `topology` attribute), the high half its name's index.
    ///
    /// # Errors
    ///
    /// Never fails. It returns a `Result` so that every opcode handler has the same type.
    #[allow(
        clippy::unnecessary_wraps,
        reason = "callers are in other files; every opcode handler returns `Lowered<()>`"
    )]
    pub(crate) fn op_spawn(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        #[allow(
            clippy::cast_possible_truncation,
            reason = "the topology id is the low half of `imm`; the high half is read separately"
        )]
        let topo = ins.imm as u32;
        self.spawn_stack
            .push((i64::from(topo), self.body.len(), ins.imm >> 32));
        let _ = write!(self.body, "  \"vx.spawn\"() ({{\n^bbspawn{idx}:\n");
        self.terminated = false;
        Ok(())
    }

    /// Close the `vx.spawn` region: terminate its last block with `vx.yield` (unless a body
    /// terminator already ended it), stamp the `topology` attribute, and resume emitting into
    /// the enclosing block (which the spawn op did not terminate).
    ///
    /// # Errors
    ///
    /// Declines when no region is open, when the topology's name is not in the string table,
    /// or when the value the region yields is not one the emitter knows.
    pub(crate) fn op_spawn_end(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let (topo, open_at, name) = self.spawn_stack.pop().ok_or(crate::emitter_gap!())?;
        // The topology's name as the program wrote it, from the function's string table.
        let name = match name.checked_sub(1) {
            Some(i) => {
                let s = usize::try_from(i)
                    .ok()
                    .and_then(|i| self.strings.get(i))
                    .ok_or(crate::emitter_gap!())?;
                format!(
                    ", topology_name = \"{}\"",
                    s.replace('\\', "\\\\").replace('"', "\\\"")
                )
            }
            None => String::new(),
        };
        // A region with a value yields `operand1`, and the op that opened at `open_at` gets
        // the result name now that the value's type is known. A scalar or a tensor.
        let mut result_ty: Option<String> = None;
        if self.types.get(ins.type_idx.0 as usize).is_some() {
            let r = ins.operand1.0 as usize;
            let v = self.names.get(r).ok_or(crate::emitter_gap!())?.clone();
            let ty = if let Some(Some(m)) = self.mem_of.get(r) {
                m.clone()
            } else {
                mlir_scalar(&self.elem_at(ins.operand1.0).ok_or(crate::emitter_gap!())?)
                    .ok_or(crate::emitter_gap!())?
                    .to_string()
            };
            if !self.terminated {
                let _ = writeln!(self.body, "  \"vx.yield\"({v}) : ({ty}) -> ()");
            }
            let n = format!("%v{idx}");
            self.body.insert_str(open_at + 2, &format!("{n} = "));
            self.names[idx] = n;
            if self.mem_of.get(r).is_some_and(std::option::Option::is_some) {
                self.mem_of[idx] = Some(ty.clone());
            } else {
                self.etypes[idx] = self.elem_at(ins.operand1.0);
            }
            result_ty = Some(ty);
        } else if !self.terminated {
            self.body += "  \"vx.yield\"() : () -> ()\n";
        }
        let result_ty = result_ty.unwrap_or_default();
        let trip = spawn_trip_attrs(ins.imm);
        // A topology that declared an `arch:` sends it along, so the device pipeline can
        // gate on the declaration instead of the dispatch-id band (Vx#352). Discardable
        // attribute on the generic form -- no dialect change involved.
        if let Some(arch) = self.ctx.topo_archs.get(&topo) {
            let _ = writeln!(
                self.body,
                "  }}) {{arch = \"{arch}\", topology = {topo} : i32{name}{trip}}} : () -> ({result_ty})"
            );
        } else {
            let _ = writeln!(
                self.body,
                "  }}) {{topology = {topo} : i32{name}{trip}}} : () -> ({result_ty})"
            );
        }
        self.terminated = false;
        Ok(())
    }

    /// Block-level sync (Vx#379). A registered op with no results and no Pure trait, so
    /// the greedy folder cannot erase it before the device clone rewrites it to
    /// `gpu.barrier`; the host lowering erases it instead (a serial loop IS the barrier).
    ///
    /// # Errors
    ///
    /// Never fails. It returns a `Result` so that every opcode handler has the same type.
    #[allow(
        clippy::unnecessary_wraps,
        reason = "callers are in other files; every opcode handler returns `Lowered<()>`"
    )]
    pub(crate) fn op_barrier(&mut self, _idx: usize, _ins: &HirInstruction) -> Lowered<()> {
        self.body += "  \"vx.barrier\"() : () -> ()\n";
        Ok(())
    }
}

/// The attributes a `vx.spawn` carries about its parallel loop, read from `SpawnEnd`'s `imm`.
///
/// A nonzero low half is the trip count `parallel_outer_for` proved for the region's
/// outermost loop: `vx_parallel_trip` tells the device pipeline the loop is safe to
/// grid-stride and how wide the work is. Empty when the low half is zero.
fn spawn_trip_attrs(imm: u64) -> String {
    // The `SPAWN_TWO_LEVEL` bit (Vx#379) marks the trip as a BLOCK count instead, and
    // `vx_parallel_two_level` rides along so the launch is sized as blocks x threads.
    let two_level = imm & crate::hir::flatten::SPAWN_TWO_LEVEL != 0;
    // The `SPAWN_COOP` bit (Vx#379 stage C): the barriers are INSIDE the thread
    // loop, so the region has no serial schedule and the host must refuse it --
    // `vx_parallel_coop` travels to the launch payload for exactly that refusal.
    let coop = imm & crate::hir::flatten::SPAWN_COOP != 0;
    let trip_count = imm & 0xffff_ffff;
    // Bits 32..47: the widest thread-loop trip, i.e. the block shape the launch
    // should use (Vx#379). Zero (a pre-two-level stream) falls back to 128.
    let threads = (imm >> 32) & 0xffff;
    if trip_count > 0 {
        format!(
            ", vx_parallel_trip = {trip_count} : i64{}",
            if two_level {
                format!(
                    ", vx_parallel_two_level, vx_parallel_threads = {} : i64{}",
                    if threads > 0 { threads } else { 128 },
                    if coop { ", vx_parallel_coop" } else { "" }
                )
            } else {
                String::new()
            }
        )
    } else {
        String::new()
    }
}
