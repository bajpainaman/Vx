use super::{
    emit_enzyme_decl, topology_to_i32, BlockLike, LowerError, LowerToMelior, MeliorGenerator,
    RegionLike,
};
use crate::syntax;
use crate::syntax::{ElementType, GradExpr, JvpExpr, Topology, VjpExpr};
use melior::ir::{
    attribute::{FlatSymbolRefAttribute, IntegerAttribute, StringAttribute},
    operation::{OperationBuilder, OperationLike},
    Identifier, Type, Value,
};

/// What lowering an expression gives back: its value, the value's type, and the block that
/// later code is appended to.
type LoweredExpr<'c> = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;

impl<'c> LowerToMelior<'c> for syntax::SpawnOnExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    /// A block: a `let` in it hides an outer variable of the same name only until it ends.
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let depth = gen.open_block();
        let lowered = lower_spawn_region(self, gen, block);
        gen.close_blocks_to(depth);
        lowered
    }
}

/// Lowers a `spawn on(...) { ... }` expression to a `vx.spawn` op whose region holds the body.
///
/// # Errors
///
/// Returns an error when a statement or the tail expression of the body cannot be lowered, or
/// when MLIR refuses one of the ops built here.
///
/// # Panics
///
/// Panics if MLIR refuses to build the `vx.yield` op that ends the region.
fn lower_spawn_region<'c>(
    this: &syntax::SpawnOnExpr,
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
) -> LoweredExpr<'c> {
    let location = gen.loc();
    let region = melior::ir::Region::new();
    let body_block = region.append_block(melior::ir::Block::new(&[]));
    let prev_in_spawn = gen.in_spawn;
    gen.in_spawn = true;

    let (body_block, tail) = lower_spawn_body(this, gen, body_block)?;
    let mut result_types = vec![];
    let mut ret_val = None;
    if let Some((val, ty)) = tail {
        result_types.push(ty);
        ret_val = Some(val);
    }

    let mut needs_yield = true;
    if let Some(syntax::Statement::Return(_)) = this.stmts.last() {
        needs_yield = false;
    }

    if needs_yield {
        let mut yield_builder = OperationBuilder::new("vx.yield", location);
        if let Some(v) = ret_val {
            yield_builder = yield_builder.add_operands(&[v]);
        }
        let yield_op = yield_builder
            .build()
            .expect("Failed to build vx.yield operation");
        body_block.append_operation(yield_op);
    }

    gen.in_spawn = prev_in_spawn;

    let mut spawn_builder = spawn_op_builder(this, gen, region, location);

    if !result_types.is_empty() {
        spawn_builder = spawn_builder.add_results(&result_types);
    }

    let spawn_op = spawn_builder.build()?;
    let spawn_ref = block.append_operation(spawn_op);

    if !needs_yield {
        let mut ret_builder = OperationBuilder::new("func.return", location);
        if !result_types.is_empty() {
            ret_builder = ret_builder.add_operands(&[spawn_ref.result(0)?.into()]);
        }
        let ret_op = ret_builder.build()?;
        block.append_operation(ret_op);
    }

    if result_types.is_empty() {
        let dummy_op = OperationBuilder::new("arith.constant", location)
            .add_attributes(&[(
                Identifier::new(gen.context, "value"),
                IntegerAttribute::new(Type::index(gen.context), 0).into(),
            )])
            .add_results(&[Type::index(gen.context)])
            .build()?;
        let dummy_ref = block.append_operation(dummy_op);
        Ok((dummy_ref.result(0)?.into(), Type::index(gen.context), block))
    } else {
        Ok((spawn_ref.result(0)?.into(), result_types[0], block))
    }
}

/// The body of a spawn region: where its last op is to be appended, and the value and type of
/// its tail expression when something is waiting for that value.
type SpawnBody<'c> = (
    melior::ir::BlockRef<'c, 'c>,
    Option<(Value<'c, 'c>, Type<'c>)>,
);

/// Lowers the statements and the tail expression of a spawn region into `body_block`.
///
/// # Errors
///
/// Returns an error when a statement or the tail expression cannot be lowered.
fn lower_spawn_body<'c>(
    this: &syntax::SpawnOnExpr,
    gen: &mut MeliorGenerator<'c>,
    mut body_block: melior::ir::BlockRef<'c, 'c>,
) -> Result<SpawnBody<'c>, LowerError> {
    // Whether anything is waiting on a value from this region. A spawn in
    // statement position is asked for `none`, and its tail expression is
    // then the last thing the region does rather than something it returns
    // -- which is what a trailing `if` is. Reading it as a result gave the
    // region a `vx.yield` of the condition against a declared result type of
    // `none`, a mismatch nothing looked at because the result went unused.
    let wants_value = gen.expected_type != Some(gen.none_ty);

    // Transport host-proven `assert` facts into the device kernel as
    // `llvm.intr.assume` certificates before the body is lowered, so they dominate
    // the guard they let the device backend fold. Host targets get nothing (the
    // relation never crossed a launch boundary). Off unless `--emit-seam-certs`.
    let is_device = !matches!(
        this.top,
        Topology::CPU | Topology::CpuAvx512 | Topology::CpuNeon | Topology::Current
    );
    if gen.emit_seam_certs && is_device {
        body_block =
            super::seam_cert::emit_seam_certificates(gen, &this.stmts, &this.ret, body_block)?;
    }

    for stmt in &this.stmts {
        if let Some(b) = gen.generate_statement(stmt, body_block)? {
            body_block = b;
        }
    }

    // The region's tail expression -- its value is the region's value. The
    // block it hands back is where the terminator has to go: an `if` (or a
    // `match`, or anything else that branches) leaves the insertion point in
    // a fresh merge block, and appending `vx.yield` to the block we started
    // in would put it after that block's `cf.cond_br`. That is invalid IR
    // rather than a wrong answer, and it is why a spawn whose last statement
    // is an `if` did not compile: the parser reads a trailing `if` as the
    // tail expression whether or not a semicolon follows it, so
    // `spawn on(...) { ...; if c { o[0][0] = 1.0; } }` came through here.
    let mut tail = None;
    if let Some(r) = &this.ret {
        let (val, ty, tail_block) = gen.generate_expr(r, body_block)?;
        body_block = tail_block;
        if wants_value {
            tail = Some((val, ty));
        }
    }
    Ok((body_block, tail))
}

/// Starts the `vx.spawn` op for `this`: the region, the topology's id and name, the declared
/// arch of a custom topology, and the plugin that claims the topology.
fn spawn_op_builder<'c>(
    this: &syntax::SpawnOnExpr,
    gen: &MeliorGenerator<'c>,
    region: melior::ir::Region<'c>,
    location: melior::ir::Location<'c>,
) -> OperationBuilder<'c> {
    let topology_id = topology_to_i32(&this.top);
    let top_attr = IntegerAttribute::new(gen.i32_ty, i64::from(topology_id)).into();

    // The topology's declared spelling, alongside its id.
    //
    // The id is a hash for a name declared in a machine file --
    // `1000 + fnv32(name) % 1000` -- so it is one-way and only 1000 wide. A plugin holding
    // `topo=1113` cannot recover `DecodeWorker`, so it cannot look the worker up in a fleet
    // manifest and discover where it lives, which is the whole of resolving a placement to a
    // machine (#348). Two names can also collide onto one id, and nothing would notice.
    //
    // Carrying the name makes it the identity and the id an optimisation. It costs a few bytes
    // per kernel and lets the compiler stay out of the business of knowing endpoints: the
    // program names a role, the machine file says what the role is, and the plugin maps the
    // name to an address.
    let name_attr =
        melior::ir::attribute::StringAttribute::new(gen.context, &this.top.display_name()).into();

    let mut spawn_builder = OperationBuilder::new("vx.spawn", location)
        .add_attributes(&[
            (Identifier::new(gen.context, "topology"), top_attr),
            (Identifier::new(gen.context, "topology_name"), name_attr),
        ])
        .add_regions([region]);

    // The declared arch travels on the spawn here exactly as on the flat path (Vx#352):
    // without it, the same source lost its device image whenever flat codegen declined and
    // this path took over -- silently, since the eligibility gate saw no attribute and a
    // custom topology's id cannot enter the band fallback. Custom topologies only: built-ins
    // declare no arch and ride the band, same asymmetry as the flat emitter.
    if let syntax::Topology::Custom(name) = &this.top {
        if let Some(desc) = gen.topologies.get(name) {
            if let Some(arch) = &desc.arch {
                let arch_attr =
                    melior::ir::attribute::StringAttribute::new(gen.context, arch.as_ref()).into();
                spawn_builder = spawn_builder
                    .add_attributes(&[(Identifier::new(gen.context, "arch"), arch_attr)]);
            }
        }
    }

    // Topology → plugin selection: if a hardware plugin claims this topology, record its
    // identity on the op so the emitted IR reflects which backend owns the region (a
    // later lowering / the runtime dispatcher can route on it). This is the point where
    // the `VxHardwarePlugin` trait is actually consulted during compilation.
    // A dispatch id is never negative, so `cast_unsigned` changes only the type.
    if let Some(plugin) = crate::plugin::plugin_for(topology_id.cast_unsigned()) {
        let plugin_attr =
            melior::ir::attribute::StringAttribute::new(gen.context, &plugin.plugin_name()).into();
        spawn_builder =
            spawn_builder.add_attributes(&[(Identifier::new(gen.context, "plugin"), plugin_attr)]);
    }
    spawn_builder
}

impl<'c> LowerToMelior<'c> for syntax::TransferExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let (src_val, src_ty, block) = gen.generate_expr(&self.expr, block)?;
        let location = gen.loc();

        // Static byte size of the transferred tile, for the SS2 sub-space bump allocator.
        let tile_bytes = static_tile_bytes(gen, &self.expr);
        let (target_topology_id, source_topology_id) = transfer_topology_ids(self, gen);

        let mut target_ty = dynamic_target_type(gen, src_ty);
        let static_target = static_target_type(gen, &self.expr);
        if let Some(t) = static_target {
            target_ty = t;
        }

        // Reconcile the operand with the static result: `memref.cast` is exactly the
        // dynamic-to-static reconciliation, and the AST type is the authority for the
        // shape it asserts.
        let mut src_val = src_val;
        if let Some(t) = static_target {
            if src_ty != t {
                src_val = crate::codegen::lower::cast_memref(gen, &block, src_val, src_ty, t)?;
            }
        }

        let transfer_builder = transfer_builder_with_topologies(
            self,
            gen,
            location,
            src_val,
            (target_topology_id, source_topology_id),
        );

        let (transfer_builder, granule_bytes) = add_space_attrs(self, gen, transfer_builder);
        let mut transfer_builder = schedule_tile(
            gen,
            &self.space,
            transfer_builder,
            granule_bytes,
            tile_bytes,
        );

        // #353: a site sema matched to a user lowering keeps the builtin's
        // allocation half -- the alloca, the offsets, the space propagation all hang
        // off the vx.transfer op -- and skips its copy and barrier: `user_lowered`
        // is the C++ side's signal. The body is then inlined right here with its
        // (src, dst) parameters bound to the source value and the placed tile.
        // Inlined, not called: a func.call inside the kernel region silently costs
        // the kernel its device twin (`isDeviceLowerableDialect` excludes func).
        let user_lowering = user_lowering_of(self, gen);
        if user_lowering.is_some() {
            transfer_builder = transfer_builder.add_attributes(&[(
                Identifier::new(gen.context, "user_lowered"),
                melior::ir::attribute::Attribute::unit(gen.context),
            )]);
        }

        let transfer_op = transfer_builder
            .add_results(&[target_ty])
            .build()
            .expect("Failed to build vx.transfer operation");

        let result_val: Value<'c, 'c> = transfer_op.result(0)?.into();
        block.append_operation(transfer_op);

        let block = match user_lowering {
            Some(body_fn) => inline_user_lowering(
                gen,
                &body_fn,
                block,
                (src_val, src_ty),
                (result_val, target_ty),
            )?,
            None => block,
        };

        Ok((result_val, target_ty, block))
    }
}

/// A `vx.transfer` of `src_val` with its target topology and, when they are known, its source
/// topology and the cost of the move as attributes. The two ids are the target's and the
/// source's, in that order; a source of 0 is unknown and gets no attribute.
fn transfer_builder_with_topologies<'c>(
    this: &syntax::TransferExpr,
    gen: &MeliorGenerator<'c>,
    location: melior::ir::Location<'c>,
    src_val: Value<'c, 'c>,
    (target_topology_id, source_topology_id): (i32, i32),
) -> OperationBuilder<'c> {
    let top_attr = IntegerAttribute::new(gen.i32_ty, i64::from(target_topology_id)).into();
    let mut transfer_builder = OperationBuilder::new("vx.transfer", location)
        .add_operands(&[src_val])
        .add_attributes(&[(Identifier::new(gen.context, "target_topology"), top_attr)]);

    if source_topology_id != 0 {
        let src_attr = IntegerAttribute::new(gen.i32_ty, i64::from(source_topology_id)).into();
        transfer_builder = transfer_builder
            .add_attributes(&[(Identifier::new(gen.context, "source_topology"), src_attr)]);
    }

    // The bandwidth-derived roofline cost (set by sema when the memory hierarchy declares
    // bandwidths). Emitting it makes the paper's data-movement cost visible in the IR.
    // i64, not i32: a picosecond-resolution time cost overflows i32 at 4.3 ms, which any
    // multi-gigabyte host transfer exceeds.
    if let Some(cost) = this.cost {
        let cost_attr = IntegerAttribute::new(gen.i64_ty, cost.cast_signed()).into();
        transfer_builder =
            transfer_builder.add_attributes(&[(Identifier::new(gen.context, "cost"), cost_attr)]);
    }
    transfer_builder
}

/// The function a user wrote to lower this transfer, when sema matched the site to one and the
/// generator knows that function's body.
fn user_lowering_of(
    this: &syntax::TransferExpr,
    gen: &MeliorGenerator<'_>,
) -> Option<syntax::Function> {
    this.lowering.as_ref().and_then(|(f, t, topo)| {
        gen.transfer_impls
            .get(&(f.name(), t.name(), topo.clone()))
            .cloned()
    })
}

/// The size in bytes of the tensor `expr` evaluates to, when its shape is known while
/// compiling. `None` for any other type or a shape with an unknown extent.
fn static_tile_bytes(gen: &MeliorGenerator<'_>, expr: &syntax::Expr) -> Option<u64> {
    gen.infer_ast_type(expr).and_then(|ty| {
        let inner = match ty {
            syntax::Type::Pinned(b, _) | syntax::Type::Ref(b, _) => *b,
            other => other,
        };
        match inner {
            syntax::Type::Tensor(e, d, _) => crate::hir::memory::static_tensor_bytes(&e, &d),
            _ => None,
        }
    })
}

/// The dispatch ids a `vx.transfer` carries: where the data is going, then where it is now
/// (0 when the source is not a placed tensor).
fn transfer_topology_ids(this: &syntax::TransferExpr, gen: &MeliorGenerator<'_>) -> (i32, i32) {
    // Map memory space to its canonical topology's dispatch id (single source of truth
    // in `arch`, adjacent to `topology_dispatch_id` so the two mappings stay in sync).
    //
    // A space that declares `node: N` is given its NUMA band id instead. The name-derived
    // id says which space; the backend needs to know which NODE, and cannot recover that
    // from an FNV hash. This is the one place the two differ, and it is deliberate: the
    // hash identifies a space, and a NUMA domain has to be acted on rather than identified.
    //
    // A node too large for the band keeps the hash, so the allocation lands wherever
    // first-touch puts it -- the behaviour before any of this, rather than an id that would
    // alias onto a slice.
    let target_topology_id = gen
        .memories
        .get(&this.space)
        .and_then(|d| d.numa_node)
        .and_then(crate::arch::numa_dispatch_id)
        .unwrap_or_else(|| crate::arch::memory_space_dispatch_id(&this.space));

    // Where the data is *now*, which is a different question from where it is going and one
    // only this stage can answer cheaply. A placed tensor's type is `Pinned(_, topology)`, so
    // the checker has already decided this; the backend would otherwise have to re-derive it
    // by walking use-def edges back to the transfer that did the placing, and that walk ends
    // at the first variable the value was bound to -- every `let` is an `alloca` and a store,
    // and the read is a load whose operand is the slot. The way home was compiled into a
    // `memcpy` from device memory for exactly that reason (#321, #348).
    let source_topology_id = match gen.infer_ast_type(&this.expr) {
        Some(syntax::Type::Pinned(_, top)) => crate::arch::topology_dispatch_id(&top),
        _ => 0,
    };
    (target_topology_id, source_topology_id)
}

/// The result type of a transfer whose shape is not known while compiling: the source memref
/// type with any layout suffix (the part after `, `) removed. Any other type is kept as it is.
fn dynamic_target_type<'c>(gen: &MeliorGenerator<'c>, src_ty: Type<'c>) -> Type<'c> {
    let src_ty_str = src_ty.to_string();
    if src_ty_str.starts_with("memref<") && src_ty_str.ends_with('>') {
        let inner_str = match src_ty_str.rfind(", ") {
            Some(suffix_at) => src_ty_str[7..suffix_at].to_string(),
            None => src_ty_str[7..src_ty_str.len() - 1].to_string(),
        };
        let target_ty_str = format!("memref<{inner_str}>");
        return Type::parse(gen.context, &target_ty_str).unwrap_or(src_ty);
    }
    src_ty
}

/// The statically shaped memref type of the tensor `expr` evaluates to, when every extent is a
/// number and the element type has an MLIR spelling.
///
/// The result is the placed TILE, and the tile's shape is a static fact of the
/// AST type even though this path's memrefs are dynamically shaped (`?x?`). A
/// static result type is what makes the SMEM alloca static downstream, and a
/// static space-3 alloca is what the device clone can promote to real `.shared`
/// storage -- the dynamic one is refused outright (#353; the A100
/// `ILLEGAL_ADDRESS` fault is why the refusal exists). Shapes that stay genuinely
/// dynamic keep the dynamic type and keep the refusal.
///
/// The SOURCE is cast to match. Making the result static alone types the
/// lowering's own `memref.copy` with a dynamic source and a static destination,
/// which is not a copy MLIR accepts -- `cpu_vector_transfer.vx` stopped lowering
/// on exactly that, a host transfer of a locally allocated `Tensor<f32>([4])`.
fn static_target_type<'c>(gen: &MeliorGenerator<'c>, expr: &syntax::Expr) -> Option<Type<'c>> {
    let inner = match gen.infer_ast_type(expr)? {
        syntax::Type::Pinned(b, _) | syntax::Type::Ref(b, _) => *b,
        other => other,
    };
    let syntax::Type::Tensor(e, dims, _) = inner else {
        return None;
    };
    let static_dims: Vec<u64> = dims
        .iter()
        .filter_map(|d| d.literal().and_then(|v| v.parse().ok()))
        .collect();
    if static_dims.len() != dims.len() || dims.is_empty() {
        return None;
    }
    let elem = match e {
        ElementType::F16 => "f16",
        ElementType::F32 => "f32",
        ElementType::F64 => "f64",
        ElementType::BF16 => "bf16",
        ElementType::I8 | ElementType::U8 => "i8",
        ElementType::I16 | ElementType::U16 => "i16",
        ElementType::I32 | ElementType::U32 => "i32",
        ElementType::I64 | ElementType::U64 | ElementType::USize => "i64",
        _ => return None,
    };
    let shape = static_dims
        .iter()
        .map(std::string::ToString::to_string)
        .collect::<Vec<_>>()
        .join("x");
    Type::parse(gen.context, &format!("memref<{shape}x{elem}>"))
}

/// Adds the description of the target memory space to a `vx.transfer`: its name and, when the
/// space is declared, what it sits within, its granule, capacity and scope, and whether the
/// host can reach it. Also returns the declared granule, in bytes.
fn add_space_attrs<'c>(
    this: &syntax::TransferExpr,
    gen: &MeliorGenerator<'c>,
    mut transfer_builder: OperationBuilder<'c>,
) -> (OperationBuilder<'c>, Option<u64>) {
    // Sub-space descriptor as IR metadata (SS1, subspace_scheduling.md): the target space name
    // plus, when declared, its containment/granule/capacity/scope. A later vx.* pass reads this
    // to schedule tiles into sub-spaces (TMEM/SMEM); it does not affect lowering.
    transfer_builder = transfer_builder.add_attributes(&[(
        Identifier::new(gen.context, "space"),
        StringAttribute::new(gen.context, &this.space.name()).into(),
    )]);
    let mut granule_bytes: Option<u64> = None;
    if let Some(decl) = gen.memories.get(&this.space) {
        if let Some(parent) = &decl.parent {
            transfer_builder = transfer_builder.add_attributes(&[(
                Identifier::new(gen.context, "within"),
                StringAttribute::new(gen.context, &parent.name()).into(),
            )]);
        }
        if let Some(g) = &decl.granule {
            granule_bytes = Some(g.0);
            transfer_builder = transfer_builder.add_attributes(&[(
                Identifier::new(gen.context, "granule"),
                IntegerAttribute::new(gen.i64_ty, g.0.cast_signed()).into(),
            )]);
        }
        if let Some(c) = &decl.capacity {
            transfer_builder = transfer_builder.add_attributes(&[(
                Identifier::new(gen.context, "capacity"),
                IntegerAttribute::new(gen.i64_ty, c.0.cast_signed()).into(),
            )]);
        }
        if let Some(scope) = &decl.scope {
            let scope_str = match scope {
                syntax::Scope::Device => "device",
                syntax::Scope::Sm => "sm",
                syntax::Scope::Cta => "cta",
                syntax::Scope::Thread => "thread",
            };
            transfer_builder = transfer_builder.add_attributes(&[(
                Identifier::new(gen.context, "scope"),
                StringAttribute::new(gen.context, scope_str).into(),
            )]);
        }

        // Whether the host can reach this space without being told to move
        // anything. `explicit` means it cannot: the bytes are somewhere only
        // an explicit transfer reaches, which on a discrete GPU is device
        // memory the CPU cannot load from at all.
        //
        // Every other field here is descriptive, and this one is not -- it
        // decides whether a kernel that fails to route may fall back to the
        // host. Without it in the IR, `vx.spawn` lowering could see that a
        // region was placed on a device and that its kernel was
        // unclassifiable, and still not know that running it here would
        // dereference device memory. That combination segmentation-faults
        // inside the outlined kernel on a real GPU (#251, #348), and the
        // fact needed to reject it at compile time was being dropped one
        // stage before the place that needed it.
        let managed_str = match decl.managed {
            syntax::Management::Explicit => "explicit",
            syntax::Management::Cached => "cached",
        };
        transfer_builder = transfer_builder.add_attributes(&[(
            Identifier::new(gen.context, "managed"),
            StringAttribute::new(gen.context, managed_str).into(),
        )]);
    }
    (transfer_builder, granule_bytes)
}

/// SS2 — schedule the tile into the sub-space: a granule-rounded bump allocation. When the
/// target sub-space declares a `granule` and the tile size is statically known, assign the
/// next free `offset` (bytes) and `slots` (granule count) within the space and advance the
/// per-function cursor. Later passes / a device backend read these to place the tile.
fn schedule_tile<'c>(
    gen: &mut MeliorGenerator<'c>,
    space: &syntax::MemorySpace,
    mut transfer_builder: OperationBuilder<'c>,
    granule_bytes: Option<u64>,
    tile_bytes: Option<u64>,
) -> OperationBuilder<'c> {
    if let (Some(granule), Some(bytes)) = (granule_bytes, tile_bytes) {
        if granule > 0 {
            let rounded = bytes.div_ceil(granule) * granule;
            let offset = *gen.subspace_offsets.entry(space.clone()).or_insert(0);
            gen.subspace_offsets.insert(space.clone(), offset + rounded);
            transfer_builder = transfer_builder.add_attributes(&[
                (
                    Identifier::new(gen.context, "offset"),
                    IntegerAttribute::new(gen.i64_ty, offset.cast_signed()).into(),
                ),
                (
                    Identifier::new(gen.context, "slots"),
                    IntegerAttribute::new(gen.i64_ty, (rounded / granule).cast_signed()).into(),
                ),
            ]);
        }
    }
    transfer_builder
}

/// Lowers the body of a user-written `impl transfer` in place of a call to it, with its two
/// parameters bound to `src` (the source value) and `dst` (the placed tile), each a value and
/// its type. Returns the block that later code is appended to.
///
/// # Errors
///
/// Returns an error when a statement of the body cannot be lowered.
fn inline_user_lowering<'c>(
    gen: &mut MeliorGenerator<'c>,
    body_fn: &syntax::Function,
    mut block: melior::ir::BlockRef<'c, 'c>,
    src: (Value<'c, 'c>, Type<'c>),
    dst: (Value<'c, 'c>, Type<'c>),
) -> Result<melior::ir::BlockRef<'c, 'c>, LowerError> {
    // The body is another function's code pasted into this one's scope, and the
    // generator's name maps are FLAT -- no scoping, no shadowing, no unwind.
    // Snapshot all three and restore them wholesale rather than hand-saving the
    // two parameter names: a `let` in the body must not outlive it, the body's
    // loop variable must not clobber the site's, and `allocs` must not go on
    // claiming a name is behind an alloca after that name means something else
    // again. Every one of those was reproduced before this snapshot existed --
    // a silently wrong answer, an out-of-bounds store, and an ICE respectively.
    let saved_env = gen.env.clone();
    let saved_ast_env = gen.ast_env.clone();
    let saved_allocs = gen.allocs.clone();
    let saved_returned = gen.has_returned;
    // An `assert` in the body is a fact about the body, not about the rest of the caller.
    let saved_facts = gen.assert_facts.len();

    let src_sym = body_fn.params[0].0.clone();
    let dst_sym = body_fn.params[1].0.clone();
    gen.env.insert(src_sym.clone(), src);
    gen.env.insert(dst_sym.clone(), dst);
    gen.ast_env
        .insert(src_sym.clone(), body_fn.params[0].1.clone());
    gen.ast_env
        .insert(dst_sym.clone(), body_fn.params[1].1.clone());
    // The tiles are values here, not alloca slots. A stale `allocs` entry from
    // a same-named binding at the site makes every read of the tile dereference
    // it.
    gen.allocs.remove(src_sym.as_ref());
    gen.allocs.remove(dst_sym.as_ref());

    for stmt in &body_fn.body {
        // The trailing `return` exists because every Vx function returns; an
        // inlined lowering returns nothing (the placed tile is the transfer's
        // own result). A return anywhere else would return from the CALLER --
        // refused in the checker, not here (E6021).
        if matches!(stmt, syntax::Statement::Return(_)) {
            continue;
        }
        if let Some(next) = gen.generate_statement(stmt, block)? {
            block = next;
        }
    }

    gen.env = saved_env;
    gen.ast_env = saved_ast_env;
    gen.allocs = saved_allocs;
    gen.has_returned = saved_returned;
    gen.assert_facts.truncate(saved_facts);
    Ok(block)
}

impl<'c> LowerToMelior<'c> for GradExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let GradExpr {
            target_fn,
            args,
            span: _,
        } = self;
        let (ret_ty, orig_arg_types) = gen
            .functions
            .get(target_fn)
            .cloned()
            .expect("Function not found");

        let fn_ty = melior::ir::r#type::FunctionType::new(gen.context, &orig_arg_types, &[ret_ty]);
        let const_op = OperationBuilder::new("func.constant", gen.loc())
            .add_attributes(&[(
                Identifier::new(gen.context, "value"),
                FlatSymbolRefAttribute::new(gen.context, target_fn).into(),
            )])
            .add_results(&[fn_ty.into()])
            .build()?;
        let const_ref = block.append_operation(const_op);
        let target_fn_val = const_ref.result(0)?.into();

        let mut arg_vals = vec![target_fn_val];
        let mut enzyme_arg_types = vec![fn_ty.into()];

        let mut current_b = block;
        for arg in args {
            let (v, ty, new_b) = gen.generate_expr(arg, current_b)?;
            arg_vals.push(v);
            enzyme_arg_types.push(ty);
            current_b = new_b;
        }

        let enzyme_name = emit_enzyme_decl(gen, "grad", target_fn, &enzyme_arg_types, ret_ty);

        let name_attr = FlatSymbolRefAttribute::new(gen.context, &enzyme_name);
        let call_op = OperationBuilder::new("func.call", gen.loc())
            .add_operands(&arg_vals)
            .add_results(&[ret_ty])
            .add_attributes(&[(Identifier::new(gen.context, "callee"), name_attr.into())])
            .build()?;

        let call_ref = current_b.append_operation(call_op);
        Ok((call_ref.result(0)?.into(), ret_ty, current_b))
    }
}

impl<'c> LowerToMelior<'c> for VjpExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let VjpExpr {
            target_fn,
            args,
            cotangent,
            span: _,
        } = self;
        let (ret_ty, orig_arg_types) = gen
            .functions
            .get(target_fn)
            .cloned()
            .expect("Function not found");

        let fn_ty = melior::ir::r#type::FunctionType::new(gen.context, &orig_arg_types, &[ret_ty]);
        let const_op = OperationBuilder::new("func.constant", gen.loc())
            .add_attributes(&[(
                Identifier::new(gen.context, "value"),
                FlatSymbolRefAttribute::new(gen.context, target_fn).into(),
            )])
            .add_results(&[fn_ty.into()])
            .build()?;
        let const_ref = block.append_operation(const_op);
        let target_fn_val = const_ref.result(0)?.into();

        let mut arg_vals = vec![target_fn_val];
        let mut enzyme_arg_types = vec![fn_ty.into()];

        let mut current_b = block;
        for arg in args {
            let (v, ty, new_b) = gen.generate_expr(arg, current_b)?;
            arg_vals.push(v);
            enzyme_arg_types.push(ty);
            current_b = new_b;
        }

        // For a scalar VJP in Enzyme, we just compute the gradient (implicitly seed=1.0)
        // and then multiply by the cotangent seed.
        let enzyme_name = emit_enzyme_decl(gen, "grad", target_fn, &enzyme_arg_types, ret_ty);

        let name_attr = FlatSymbolRefAttribute::new(gen.context, &enzyme_name);
        let call_op = OperationBuilder::new("func.call", gen.loc())
            .add_operands(&arg_vals)
            .add_results(&[ret_ty])
            .add_attributes(&[(Identifier::new(gen.context, "callee"), name_attr.into())])
            .build()?;

        let call_ref = current_b.append_operation(call_op);
        let grad_val = call_ref.result(0)?.into();

        let (c_val, _, current_b) = gen.generate_expr(cotangent, current_b)?;

        // Multiply grad by cotangent
        let is_float = ret_ty.to_string().contains("f32")
            || ret_ty.to_string().contains("f64")
            || ret_ty.to_string().contains("f16")
            || ret_ty.to_string().contains("bf16");
        let op_name = if is_float { "arith.mulf" } else { "arith.muli" };
        let mul_op = OperationBuilder::new(op_name, gen.loc())
            .add_operands(&[grad_val, c_val])
            .add_results(&[ret_ty])
            .build()?;
        let mul_ref = current_b.append_operation(mul_op);

        Ok((mul_ref.result(0)?.into(), ret_ty, current_b))
    }
}

impl<'c> LowerToMelior<'c> for JvpExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let JvpExpr {
            target_fn,
            args,
            tangent,
            span: _,
        } = self;
        let (ret_ty, orig_arg_types) = gen
            .functions
            .get(target_fn)
            .cloned()
            .expect("Function not found");

        let fn_ty = melior::ir::r#type::FunctionType::new(gen.context, &orig_arg_types, &[ret_ty]);
        let const_op = OperationBuilder::new("func.constant", gen.loc())
            .add_attributes(&[(
                Identifier::new(gen.context, "value"),
                FlatSymbolRefAttribute::new(gen.context, target_fn).into(),
            )])
            .add_results(&[fn_ty.into()])
            .build()?;
        let const_ref = block.append_operation(const_op);
        let target_fn_val = const_ref.result(0)?.into();

        let mut arg_vals = vec![target_fn_val];
        let mut enzyme_arg_types = vec![fn_ty.into()];

        let mut current_b = block;
        for arg in args {
            let (v, ty, new_b) = gen.generate_expr(arg, current_b)?;
            arg_vals.push(v);
            enzyme_arg_types.push(ty);
            current_b = new_b;
        }

        let (t_val, t_ty, current_b) = gen.generate_expr(tangent, current_b)?;
        arg_vals.push(t_val);
        enzyme_arg_types.push(t_ty);

        // Enzyme intercepts `__enzyme_fwddiff` for forward mode.
        let enzyme_name = emit_enzyme_decl(gen, "fwddiff", target_fn, &enzyme_arg_types, ret_ty);

        let name_attr = FlatSymbolRefAttribute::new(gen.context, &enzyme_name);
        let call_op = OperationBuilder::new("func.call", gen.loc())
            .add_operands(&arg_vals)
            .add_results(&[ret_ty])
            .add_attributes(&[(Identifier::new(gen.context, "callee"), name_attr.into())])
            .build()?;

        let call_ref = current_b.append_operation(call_op);
        Ok((call_ref.result(0)?.into(), ret_ty, current_b))
    }
}
