use super::{
    extract_mlir_element_type, generate_match_chain, lower_map_call, lower_print_call,
    lower_raw_primitive, lower_reduce_call, returns_through_slot, ArrayAttribute, BlockLike,
    LowerError, LowerToMelior, MeliorGenerator, MeliorOpInfo, RegionLike, ValueLike,
};
use crate::syntax;
use crate::syntax::{
    ArrayExpr, BinaryOp, BinaryOpExpr, BorrowExpr, ClosureExpr, ComptimeBlockExpr, DereferenceExpr,
    EnumVariantExpr, Expr, FunctionCallExpr, IdentifierExpr, IndirectCallExpr, InlineMlirExpr,
    LogicalOpExpr, MatchExpr, MemberAccessExpr, MemorySpaceExpr, MethodCallExpr, NumberExpr,
    RelationalOpExpr, Span, StringLiteralExpr, StructInitExpr, TopologyExpr, UnsafeBlockExpr,
    VecMacroExpr,
};
use melior::ir::{
    attribute::{
        DenseI32ArrayAttribute, DenseI64ArrayAttribute, FlatSymbolRefAttribute, FloatAttribute,
        IntegerAttribute, StringAttribute, TypeAttribute,
    },
    operation::OperationBuilder,
    Attribute, Identifier, Region, Type, Value,
};

/// What lowering an expression gives back: its value, its MLIR type, and the block the next
/// operation goes in.
type Lowered<'c> = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;

/// A lowered value with its MLIR type.
type Typed<'c> = (Value<'c, 'c>, Type<'c>);

/// A position, a count or a size in bytes, as the `i64` an MLIR attribute takes.
///
/// # Panics
///
/// Panics if `n` does not fit in an `i64`, which no position, count or size here does.
fn count_i64(n: usize) -> i64 {
    i64::try_from(n).expect("a position, a count or a size fits in an i64")
}

/// A count, as the `i32` an MLIR attribute takes.
///
/// # Panics
///
/// Panics if `n` does not fit in an `i32`, which no count of operands does.
fn count_i32(n: usize) -> i32 {
    i32::try_from(n).expect("a count of operands fits in an i32")
}

/// Argument `n` of a block that is still being built.
///
/// # Errors
///
/// Returns an error if the block has no argument `n`.
fn block_arg<'c, 'a>(
    block: &'a melior::ir::Block<'c>,
    n: usize,
) -> Result<Value<'c, 'a>, LowerError> {
    // The lookup's own error only repeats the number, which this message already has.
    let Ok(arg) = block.argument(n) else {
        return Err(LowerError::from(format!("Missing argument {n} block")));
    };
    Ok(arg.into())
}

/// One arithmetic operation, `name`, on two `index` values.
///
/// # Errors
///
/// Returns an error if MLIR refuses an operation built here.
fn index_arith<'c>(
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    name: &str,
    a: Value<'c, 'c>,
    b: Value<'c, 'c>,
) -> Result<Value<'c, 'c>, LowerError> {
    let op = OperationBuilder::new(name, gen.loc())
        .add_operands(&[a, b])
        .add_results(&[Type::index(gen.context)])
        .build()?;
    Ok(block.append_operation(op).result(0)?.into())
}

/// The element type named by `el_ty_str`.
///
/// # Errors
///
/// Returns an error if the text is not an MLIR type.
fn parse_element_type<'c>(
    gen: &MeliorGenerator<'c>,
    el_ty_str: &str,
) -> Result<Type<'c>, LowerError> {
    Type::parse(gen.context, el_ty_str).ok_or_else(|| {
        crate::codegen::lower::LowerError::ParseType("Type::parse failed".to_string())
    })
}

/// A new block that takes `count` arguments of the element type `el_ty_str`.
///
/// # Errors
///
/// Returns an error if the text is not an MLIR type.
fn element_block<'c>(
    gen: &MeliorGenerator<'c>,
    el_ty_str: &str,
    count: usize,
) -> Result<melior::ir::Block<'c>, LowerError> {
    let mut args = Vec::with_capacity(count);
    for _ in 0..count {
        args.push((parse_element_type(gen, el_ty_str)?, gen.loc()));
    }
    Ok(melior::ir::Block::new(&args))
}

/// Whether this name holds a reference rather than a value of its own.
///
/// A local bound to `&mut x` carries `x`'s scalar-`memref` descriptor, which is the same shape
/// as the slot a mutable scalar local is stored in. The two are told apart by what reading them
/// means: reading a slot loads the value out of it, and reading a reference hands the reference
/// back. Without this, `let p : *mut i32 = &mut cell;` loaded `cell` and passed the integer
/// where a pointer was wanted, which reached LLVM translation as a cast from `i32` to a pointer.
fn holds_a_reference(gen: &MeliorGenerator<'_>, name: &crate::symbol::Symbol) -> bool {
    matches!(
        gen.ast_env.get(name),
        Some(syntax::Type::Pointer(..) | syntax::Type::Borrow { .. } | syntax::Type::Ref(..))
    )
}

/// The one value of a rank-0 tensor, for an operation on scalars; any other value as it is.
///
/// # Errors
///
/// Returns an error if the tensor's element type does not parse, or if MLIR refuses the load.
fn load_rank_zero<'c>(
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    val: Value<'c, 'c>,
    ty: Type<'c>,
) -> Result<(Value<'c, 'c>, Type<'c>), LowerError> {
    if !gen.is_scalar_memref(&ty) {
        return Ok((val, ty));
    }
    let text = ty.to_string();
    let elem = Type::parse(gen.context, &text[7..text.len() - 1])
        .ok_or_else(|| LowerError::ParseType(text.clone()))?;
    let load = OperationBuilder::new("memref.load", gen.loc())
        .add_operands(&[val])
        .add_results(&[elem])
        .build()?;
    Ok((block.append_operation(load).result(0)?.into(), elem))
}

/// A `Tensor<T, []>` local: its `memref<T>` is the tensor, not a cell holding a scalar.
fn is_rank_zero_tensor(gen: &MeliorGenerator<'_>, name: &crate::symbol::Symbol) -> bool {
    matches!(gen.ast_env.get(name), Some(syntax::Type::Tensor(_, dims, _)) if dims.is_empty())
}

/// Reads a local that has a stack slot: its value, or the slot's address when an address is
/// wanted. `None` when the slot's type is not one that is read here.
///
/// # Errors
///
/// Returns an error if MLIR refuses the load.
///
/// # Panics
///
/// Panics if the slot's element type does not parse, which means the slot was made wrongly.
fn read_stack_slot<'c>(
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    name: &crate::symbol::Symbol,
    val: Value<'c, 'c>,
    ty: Type<'c>,
) -> Result<Option<Typed<'c>>, LowerError> {
    let ty_str = ty.to_string();
    if ty_str.starts_with("memref<") {
        let inner_ty_str = &ty_str[7..ty_str.len() - 1];
        let inner_ty = Type::parse(gen.context, inner_ty_str)
            .unwrap_or_else(|| panic!("failed to parse {inner_ty_str:?} for variable {name:?}"));
        let load_op = OperationBuilder::new("memref.load", gen.loc())
            .add_operands(&[val])
            .add_results(&[inner_ty])
            .build()?;
        let load_ref = block.append_operation(load_op);
        return Ok(Some((load_ref.result(0)?.into(), inner_ty)));
    } else if ty_str.starts_with("!llvm.ptr")
        || ty_str.starts_with("!llvm.struct")
        || ty_str.starts_with('i')
        || ty_str.starts_with('u')
        || ty_str.starts_with('f')
    {
        if gen.is_lvalue_context {
            let ptr_ty = gen.ptr_ty;
            return Ok(Some((val, ptr_ty)));
        }
        let elem_ty = ty;
        let load_op = OperationBuilder::new("llvm.load", gen.loc())
            .add_operands(&[val])
            .add_results(&[elem_ty])
            .build()?;
        let load_ref = block.append_operation(load_op);
        return Ok(Some((load_ref.result(0)?.into(), elem_ty)));
    }
    Ok(None)
}

/// The value of a local variable: what its slot holds, or the variable itself when it has none.
///
/// # Errors
///
/// Returns an error if a slot's element type does not parse, or if MLIR refuses the load.
fn read_local<'c>(
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    name: &crate::symbol::Symbol,
    val: Value<'c, 'c>,
    ty: Type<'c>,
) -> Result<Typed<'c>, LowerError> {
    let ty_str = ty.to_string();
    if gen.allocs.contains(name.as_ref()) {
        if let Some(read) = read_stack_slot(gen, block, name, val, ty)? {
            return Ok(read);
        }
    }
    if ty_str.starts_with("memref<memref<") {
        let inner_ty_str = &ty_str[7..ty_str.len() - 1];
        let inner_ty = Type::parse(gen.context, inner_ty_str).ok_or_else(|| {
            crate::codegen::lower::LowerError::ParseType("Type::parse failed".to_string())
        })?;
        let load_op = OperationBuilder::new("memref.load", gen.loc())
            .add_operands(&[val])
            .add_results(&[inner_ty])
            .build()?;
        let load_ref = block.append_operation(load_op);
        Ok((load_ref.result(0)?.into(), inner_ty))
    } else if ty_str.starts_with("memref<")
        && !ty_str.contains('x')
        && !holds_a_reference(gen, name)
        && !is_rank_zero_tensor(gen, name)
    {
        let inner_ty_str = &ty_str[7..ty_str.len() - 1];
        let inner_ty = Type::parse(gen.context, inner_ty_str).ok_or_else(|| {
            crate::codegen::lower::LowerError::ParseType("Type::parse failed".to_string())
        })?;
        let load_op = OperationBuilder::new("memref.load", gen.loc())
            .add_operands(&[val])
            .add_results(&[inner_ty])
            .build()?;
        let load_ref = block.append_operation(load_op);
        Ok((load_ref.result(0)?.into(), inner_ty))
    } else {
        Ok((val, ty))
    }
}

/// A name that is not a local: the address of a function, or a `const` table. `None` when the
/// name is neither.
///
/// # Errors
///
/// Returns an error if MLIR refuses an operation built here.
///
/// # Panics
///
/// Panics if a `const` table's memref type does not parse, which means the table was made wrongly.
fn named_global<'c>(
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    name: &crate::symbol::Symbol,
) -> Result<Option<Typed<'c>>, LowerError> {
    if let Some((ret_ty, arg_tys)) = gen.functions.get(name) {
        // The address of a function has to name the signature it was emitted with, return
        // slot and all, or an indirect call through it disagrees with the body.
        let (abi_ret, abi_args) = gen.abi_signature(*ret_ty, arg_tys);
        let abi_rets: Vec<_> = abi_ret.into_iter().collect();
        let func_ty = melior::ir::r#type::FunctionType::new(gen.context, &abi_args, &abi_rets);
        let const_op = OperationBuilder::new("func.constant", gen.loc())
            .add_attributes(&[(
                Identifier::new(gen.context, "value"),
                FlatSymbolRefAttribute::new(gen.context, name).into(),
            )])
            .add_results(&[func_ty.into()])
            .build()?;
        let const_ref = block.append_operation(const_op);

        let ptr_ty = gen.ptr_ty;
        let cast_op = OperationBuilder::new("builtin.unrealized_conversion_cast", gen.loc())
            .add_operands(&[const_ref.result(0)?.into()])
            .add_results(&[ptr_ty])
            .build()?;
        let cast_ref = block.append_operation(cast_op);

        Ok(Some((cast_ref.result(0)?.into(), ptr_ty)))
    } else if let Some(table) = gen.const_tables.get(name) {
        // A `const` table: its read-only global, which `generate_module` added.
        let memty =
            Type::parse(gen.context, &table.memref_type()).expect("a table's memref type parses");
        let get = OperationBuilder::new("memref.get_global", gen.loc())
            .add_attributes(&[(
                Identifier::new(gen.context, "name"),
                FlatSymbolRefAttribute::new(gen.context, &table.symbol()).into(),
            )])
            .add_results(&[memty])
            .build()?;
        let get_ref = block.append_operation(get);
        Ok(Some((get_ref.result(0)?.into(), memty)))
    } else {
        Ok(None)
    }
}

impl<'c> LowerToMelior<'c> for IdentifierExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let IdentifierExpr { name, span: _ } = self;
        if name.as_ref() == "true" || **name == *"false" {
            let i1_ty = gen.i1_ty;
            let val = i64::from(name.as_ref() == "true");
            let const_op = OperationBuilder::new("arith.constant", gen.loc())
                .add_results(&[i1_ty])
                .add_attributes(&[(
                    Identifier::new(gen.context, "value"),
                    IntegerAttribute::new(i1_ty, val).into(),
                )])
                .build()?;
            let const_ref = block.append_operation(const_op);
            return Ok((const_ref.result(0)?.into(), i1_ty, block));
        }
        if let Some((val, ty)) = gen.env.get(name).map(|(v, t)| (*v, *t)) {
            let (val, ty) = read_local(gen, block, name, val, ty)?;
            Ok((val, ty, block))
        } else if let Some((val, ty)) = named_global(gen, block, name)? {
            Ok((val, ty, block))
        } else {
            panic!("Undefined variable: {name}");
        }
    }
}

/// The aligned data pointer of a scalar memref, as a bare `!llvm.ptr`.
///
/// # Errors
///
/// Returns an error if MLIR refuses an operation built here.
fn memref_data_pointer<'c>(
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    val: Value<'c, 'c>,
) -> Result<Value<'c, 'c>, LowerError> {
    let idx_ty = Type::index(gen.context);
    let raw_idx = block
        .append_operation(
            OperationBuilder::new("memref.extract_aligned_pointer_as_index", gen.loc())
                .add_operands(&[val])
                .add_results(&[idx_ty])
                .build()?,
        )
        .result(0)?
        .into();
    let raw_i64 = block
        .append_operation(
            OperationBuilder::new("arith.index_cast", gen.loc())
                .add_operands(&[raw_idx])
                .add_results(&[gen.i64_ty])
                .build()?,
        )
        .result(0)?
        .into();
    Ok(block
        .append_operation(
            OperationBuilder::new("llvm.inttoptr", gen.loc())
                .add_operands(&[raw_i64])
                .add_results(&[gen.ptr_ty])
                .build()?,
        )
        .result(0)?
        .into())
}

/// The element type of a tensor's memref type, as text, and its size in bits.
///
/// # Errors
///
/// Returns an error if `base_ty` is not a tensor's memref type, if its elements have no fixed
/// size, or if the tensor is in another memory space.
fn tensor_element_bits(base_ty: &str) -> Result<(&str, u32), LowerError> {
    let inner = base_ty
        .strip_prefix("memref<")
        .and_then(|s| s.split([',', '>']).next())
        .ok_or_else(|| {
            format!("an element address of something that is not a tensor: {base_ty}")
        })?;
    let elem = inner.rsplit('x').next().unwrap_or(inner);
    let bits = crate::codegen::generator::scalar_type_bits(elem)
        .ok_or_else(|| format!("an element address of a tensor of {elem}"))?;
    if base_ty
        .split(['<', '>'])
        .any(|p| p.trim_start_matches(", ").parse::<u32>().is_ok())
    {
        return Err(LowerError::from(format!(
            "an element address of a tensor in another memory space: {base_ty}"
        )));
    }
    Ok((elem, bits))
}

/// Where an element sits in a tensor's buffer, counted in elements: the tensor's offset plus
/// each index times the stride of its dimension, read from the descriptor.
///
/// # Errors
///
/// Returns an error if the buffer's type does not parse, or if MLIR refuses an operation.
fn element_position<'c>(
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    base: Value<'c, 'c>,
    elem: &str,
    indices: &[Value<'c, 'c>],
) -> Result<Value<'c, 'c>, LowerError> {
    let index_ty = Type::index(gen.context);
    let buf_ty = Type::parse(gen.context, &format!("memref<{elem}>"))
        .ok_or_else(|| LowerError::ParseType(format!("memref<{elem}>")))?;
    let rank = indices.len();
    let mut result_tys = vec![buf_ty];
    result_tys.extend(std::iter::repeat_n(index_ty, 1 + 2 * rank));
    let meta = block.append_operation(
        OperationBuilder::new("memref.extract_strided_metadata", gen.loc())
            .add_operands(&[base])
            .add_results(&result_tys)
            .build()?,
    );
    let mut linear: Value<'c, 'c> = meta.result(1)?.into();
    for (k, idx) in indices.iter().enumerate() {
        let stride: Value<'c, 'c> = meta.result(2 + rank + k)?.into();
        let step = index_arith(gen, block, "arith.muli", *idx, stride)?;
        linear = index_arith(gen, block, "arith.addi", linear, step)?;
    }
    Ok(linear)
}

/// The address of one element of a tensor, as a bare `!llvm.ptr`: the buffer's aligned pointer
/// plus `(offset + Σ index·stride) · element size`, read from the descriptor. `base` is the
/// tensor's memref and `indices` one `index` value per dimension.
///
/// # Errors
///
/// Returns an error if `base_ty` is not the type of a tensor whose elements have an address
/// (see [`tensor_element_bits`]), or if MLIR refuses an operation.
fn tensor_element_pointer<'c>(
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    base: Value<'c, 'c>,
    base_ty: &str,
    indices: &[Value<'c, 'c>],
) -> Result<Value<'c, 'c>, LowerError> {
    let (elem, bits) = tensor_element_bits(base_ty)?;
    let index_ty = Type::index(gen.context);
    let linear = element_position(gen, block, base, elem, indices)?;
    let aligned: Value<'c, 'c> = block
        .append_operation(
            OperationBuilder::new("memref.extract_aligned_pointer_as_index", gen.loc())
                .add_operands(&[base])
                .add_results(&[index_ty])
                .build()?,
        )
        .result(0)?
        .into();
    let size: Value<'c, 'c> = block
        .append_operation(
            OperationBuilder::new("arith.constant", gen.loc())
                .add_attributes(&[(
                    Identifier::new(gen.context, "value"),
                    IntegerAttribute::new(index_ty, (i64::from(bits) + 7) / 8).into(),
                )])
                .add_results(&[index_ty])
                .build()?,
        )
        .result(0)?
        .into();
    let bytes = index_arith(gen, block, "arith.muli", linear, size)?;
    let addr = index_arith(gen, block, "arith.addi", aligned, bytes)?;
    let addr_i64: Value<'c, 'c> = block
        .append_operation(
            OperationBuilder::new("arith.index_cast", gen.loc())
                .add_operands(&[addr])
                .add_results(&[gen.i64_ty])
                .build()?,
        )
        .result(0)?
        .into();
    Ok(block
        .append_operation(
            OperationBuilder::new("llvm.inttoptr", gen.loc())
                .add_operands(&[addr_i64])
                .add_results(&[gen.ptr_ty])
                .build()?,
        )
        .result(0)?
        .into())
}

/// `&name` where `name` is a local variable: the address of its slot, or a new slot for a
/// reference that has none. `None` when the general path has to handle it.
///
/// # Errors
///
/// Returns an error if MLIR refuses an operation built here.
fn borrow_of_local<'c>(
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    id: &IdentifierExpr,
) -> Result<Option<Typed<'c>>, LowerError> {
    if gen.allocs.contains(&*id.name) {
        if let Some((val, val_ty)) = gen.env.get(&*id.name) {
            let ptr_ty = gen.ptr_ty;
            if gen.is_memref(val_ty) {
                return Ok(Some((*val, *val_ty)));
            }
            // The slot's address. Typing it as the slot's contents made `(*p).f`
            // in a place context extract from a pointer as if it were a struct.
            return Ok(Some((*val, ptr_ty)));
        }
    } else if let Some((val, val_ty)) = gen.env.get(&*id.name).map(|(v, t)| (*v, *t)) {
        // `&r` where `r` is a non-`mut` reference *local* (not in `allocs`): it was bound to a
        // reference value with no backing storage of its own, so there is no address to hand
        // back. Materialize a slot holding a bare pointer to what `r` refers to, and return the
        // slot — a real `&&T`. Otherwise `&r` would alias `r`'s pointee and `**rr` reads garbage
        // / segfaults (#278). Two shapes of reference local reach here:
        //   - an immutable `let r = &x`, bound to a bare SSA `!llvm.ptr` (Defect 1): store it;
        //   - a mutable `let r = &mut x`, bound to `x`'s `memref<T>` *descriptor* (Defect 2):
        //     extract the memref's aligned data pointer as a bare `!llvm.ptr` and store that.
        //     Only the raw pointer is kept, so the nested deref chain stays llvm-pointer-based
        //     (matching `DereferenceExpr`, which llvm-loads and cannot chase a memref).
        // A non-pointer rvalue (`let x = 5`) or a ranked/tensor memref falls through to the
        // general materialization below.
        let ptr_ty = gen.ptr_ty;
        let val_ty_str = val_ty.to_string();
        let is_bare_ptr = val.r#type() == ptr_ty;
        let is_scalar_memref = val_ty_str.starts_with("memref<")
            && !val_ty_str.contains('x')
            && !val_ty_str.starts_with("memref<memref<");
        if is_bare_ptr || is_scalar_memref {
            // The bare `!llvm.ptr` to store: `r`'s value directly, or the memref's data pointer.
            let ptr_val = if is_bare_ptr {
                val
            } else {
                memref_data_pointer(gen, block, val)?
            };
            let i32_ty = gen.i32_ty;
            let c1 = block
                .append_operation(
                    OperationBuilder::new("llvm.mlir.constant", gen.loc())
                        .add_results(&[i32_ty])
                        .add_attributes(&[(
                            Identifier::new(gen.context, "value"),
                            IntegerAttribute::new(i32_ty, 1).into(),
                        )])
                        .build()?,
                )
                .result(0)?
                .into();
            let slot = block
                .append_operation(
                    OperationBuilder::new("llvm.alloca", gen.loc())
                        .add_operands(&[c1])
                        .add_results(&[ptr_ty])
                        .add_attributes(&[(
                            Identifier::new(gen.context, "elem_type"),
                            TypeAttribute::new(ptr_ty).into(),
                        )])
                        .build()?,
                )
                .result(0)?
                .into();
            block.append_operation(
                OperationBuilder::new("llvm.store", gen.loc())
                    .add_operands(&[ptr_val, slot])
                    .build()?,
            );
            return Ok(Some((slot, ptr_ty)));
        }
    }
    Ok(None)
}

/// `&mut c[i][j]`, an element of a tensor variable: the element's address in the tensor.
/// Lowered as a value, the element would be loaded and the borrow would point at a copy, so a
/// write through it would be lost. `None` when `expr` is not an element of a tensor variable.
///
/// # Errors
///
/// Returns an error if the indices cannot be lowered, or if the element has no address (see
/// [`tensor_element_pointer`]).
fn borrow_of_tensor_element<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    expr: &Expr,
) -> Result<Option<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>)>, LowerError> {
    let mut root = expr;
    let mut indices = 0;
    while let Expr::IndexAccess(ix) = root {
        root = &ix.base;
        indices += 1;
    }
    let element_of_a_tensor = indices > 0
        && matches!(root, Expr::Identifier(_))
        && match gen.infer_ast_type(root) {
            Some(syntax::Type::Tensor(_, dims, _)) => dims.len() == indices,
            Some(syntax::Type::Borrow { inner, .. }) => {
                matches!(&*inner, syntax::Type::Tensor(_, dims, _) if dims.len() == indices)
            }
            // A `let` bound to a row has no inferred type here; its slot's memref type
            // says the rank.
            None => match root {
                Expr::Identifier(id) => gen
                    .env
                    .get(&*id.name)
                    .map(|(_, t)| t.to_string())
                    .and_then(|t| {
                        let t = t.strip_prefix("memref<memref<")?;
                        let dims = t.split([',', '>']).next()?;
                        Some(dims.split('x').count() - 1 == indices)
                    })
                    .unwrap_or(false),
                _ => false,
            },
            _ => false,
        };
    if !element_of_a_tensor {
        return Ok(None);
    }
    let prev_lvalue = gen.is_lvalue_context;
    gen.is_lvalue_context = false;
    let flat = gen.flatten_indices(expr, block);
    gen.is_lvalue_context = prev_lvalue;
    let (base, mut base_ty, idx_vals, block) = flat
        .ok_or_else(|| LowerError::from("the indices of a borrowed tensor element".to_string()))?;
    let base = load_tensor_slot(gen, &block, base, &mut base_ty)?;
    let ptr = tensor_element_pointer(gen, block, base, &base_ty.to_string(), &idx_vals)?;
    Ok(Some((ptr, gen.ptr_ty, block)))
}

/// Puts a value in a new stack slot and gives back the slot's address and the address's type.
///
/// # Errors
///
/// Returns an error if MLIR refuses an operation built here.
///
/// # Panics
///
/// Panics if the type of a slot for a memref does not parse, which no memref type does.
fn store_in_new_slot<'c>(
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    val: Value<'c, 'c>,
    ty: Type<'c>,
) -> Result<Typed<'c>, LowerError> {
    let ptr_ty = gen.ptr_ty;
    if gen.is_memref(&ty) {
        // Allocate a pointer to the memref
        let alloca_op = block.append_operation(
            OperationBuilder::new("memref.alloca", gen.loc())
                .add_results(&[Type::parse(gen.context, &format!("memref<{ty}>")).unwrap()])
                .build()?,
        );
        let ptr = alloca_op.result(0)?.into();

        block.append_operation(
            OperationBuilder::new("memref.store", gen.loc())
                .add_operands(&[val, ptr])
                .build()?,
        );
        return Ok((
            ptr,
            Type::parse(gen.context, &format!("memref<{ty}>")).unwrap(),
        ));
    }

    let i32_ty = gen.i32_ty;
    let c1_op = block.append_operation(
        OperationBuilder::new("llvm.mlir.constant", gen.loc())
            .add_results(&[i32_ty])
            .add_attributes(&[(
                Identifier::new(gen.context, "value"),
                IntegerAttribute::new(i32_ty, 1).into(),
            )])
            .build()?,
    );
    let c1 = c1_op.result(0)?.into();

    let alloca_op = block.append_operation(
        OperationBuilder::new("llvm.alloca", gen.loc())
            .add_operands(&[c1])
            .add_results(&[ptr_ty])
            .add_attributes(&[(
                Identifier::new(gen.context, "elem_type"),
                TypeAttribute::new(ty).into(),
            )])
            .build()?,
    );
    let ptr = alloca_op.result(0)?.into();

    block.append_operation(
        OperationBuilder::new("llvm.store", gen.loc())
            .add_operands(&[val, ptr])
            .build()?,
    );

    Ok((ptr, ptr_ty))
}

impl<'c> LowerToMelior<'c> for BorrowExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let BorrowExpr { expr, .. } = self;
        let local = match &**expr {
            Expr::Identifier(id) => borrow_of_local(gen, block, id)?,
            _ => None,
        };
        if let Some((val, ty)) = local {
            return Ok((val, ty, block));
        }
        if let Some(lowered) = borrow_of_tensor_element(gen, block, expr)? {
            return Ok(lowered);
        }
        // Only a place is lowered to its address. Anything else, `&(*p / 2)`, is a value that
        // gets a slot below, and inside it `*p` is a load like any other.
        let is_place = matches!(
            &**expr,
            Expr::Identifier(_)
                | Expr::MemberAccess(_)
                | Expr::IndexAccess(_)
                | Expr::Dereference(_)
        );
        let prev_lvalue = gen.is_lvalue_context;
        gen.is_lvalue_context = is_place;
        let (val, ty, block) = gen.generate_expr(expr, block)?;
        gen.is_lvalue_context = prev_lvalue;
        let ptr_ty = gen.ptr_ty;
        // A place lowered to its address. A value that is itself a pointer, `&a` in `&(&a)`,
        // gets a slot below like any other value, or the outer `&` would be lost.
        if ty == ptr_ty && is_place {
            return Ok((val, ptr_ty, block));
        }
        if ty.to_string().starts_with("memref<memref<") {
            return Ok((val, ty, block));
        }
        // `&(&mut b)`: the inner `&mut` of a local is that local's scalar memref. Keep only its
        // data pointer, as `&r` does above, so `**` reads through plain pointers.
        let (val, ty) = if matches!(&**expr, Expr::Borrow(_))
            && ty.to_string().starts_with("memref<")
            && !ty.to_string().contains('x')
        {
            (memref_data_pointer(gen, block, val)?, ptr_ty)
        } else {
            (val, ty)
        };
        let (slot, slot_ty) = store_in_new_slot(gen, block, val, ty)?;
        Ok((slot, slot_ty, block))
    }
}

impl<'c> LowerToMelior<'c> for StringLiteralExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let str_val = format!("{}\0", self.value);
        let str_name = format!(".str.{}", gen.string_counter);
        gen.string_counter += 1;

        let module_body = gen.module.body();
        let array_ty =
            Type::parse(gen.context, &format!("!llvm.array<{} x i8>", str_val.len())).unwrap();
        let ptr_ty = gen.ptr_ty;

        let global_op = OperationBuilder::new("llvm.mlir.global", gen.loc())
            .add_attributes(&[
                (
                    Identifier::new(gen.context, "sym_name"),
                    StringAttribute::new(gen.context, &str_name).into(),
                ),
                (
                    Identifier::new(gen.context, "global_type"),
                    TypeAttribute::new(array_ty).into(),
                ),
                (
                    Identifier::new(gen.context, "constant"),
                    Attribute::parse(gen.context, "unit").unwrap(),
                ),
                (
                    Identifier::new(gen.context, "linkage"),
                    Attribute::parse(gen.context, "#llvm.linkage<internal>").unwrap(),
                ),
                (
                    Identifier::new(gen.context, "value"),
                    StringAttribute::new(gen.context, &str_val).into(),
                ),
            ])
            .add_regions([Region::new()])
            .build()?;
        module_body.append_operation(global_op);

        let addressof_op = OperationBuilder::new("llvm.mlir.addressof", gen.loc())
            .add_attributes(&[(
                Identifier::new(gen.context, "global_name"),
                FlatSymbolRefAttribute::new(gen.context, &str_name).into(),
            )])
            .add_results(&[ptr_ty])
            .build()?;
        let addressof_ref = block.append_operation(addressof_op);

        Ok((addressof_ref.result(0)?.into(), ptr_ty, block))
    }
}

impl<'c> LowerToMelior<'c> for ComptimeBlockExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    /// A block: a `let` in it hides an outer variable of the same name only until it ends.
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let depth = gen.open_block();
        let lowered = lower_comptime_block(self, gen, block);
        gen.close_blocks_to(depth);
        lowered
    }
}

/// The statements of a `comptime` block in order, then its value.
///
/// # Errors
///
/// Returns an error if a statement or the final expression cannot be lowered.
fn lower_comptime_block<'c>(
    this: &ComptimeBlockExpr,
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
) -> Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError> {
    // Same rule as everywhere else a block of statements is spliced inline:
    // the block a statement hands back is the one the next statement goes
    // in. Dropping it left `comptime { for ... }` appending past the loop's
    // own branch, which the verifier rejects outright.
    let mut cur = block;
    for stmt in &this.stmts {
        if let Some(b) = gen.generate_statement(stmt, cur)? {
            cur = b;
        } else {
            cur = super::dead_continuation(cur);
            break;
        }
    }
    if let Some(ret_expr) = &this.ret {
        let (val, ty, cur) = gen.generate_expr(ret_expr, cur)?;
        let cur = gen.run_queued_drops(cur)?;
        Ok((val, ty, cur))
    } else {
        let none_ty = gen.none_ty;
        let dummy_val = OperationBuilder::new("arith.constant", gen.loc())
            .add_attributes(&[(
                Identifier::new(gen.context, "value"),
                IntegerAttribute::new(Type::index(gen.context), 0).into(),
            )])
            .add_results(&[Type::index(gen.context)])
            .build()?;
        let dummy_ref = cur.append_operation(dummy_val);
        Ok((dummy_ref.result(0)?.into(), none_ty, cur))
    }
}

/// The pointee AST type of a pointer/borrow/ref (`*mut i32` -> `i32`), for recovering a dereference's
/// loaded type. `None` for a non-pointer. (#242)
fn deref_pointee_type(ty: &syntax::Type) -> Option<syntax::Type> {
    match ty {
        syntax::Type::Pointer(inner, ..)
        | syntax::Type::Borrow { inner, .. }
        | syntax::Type::Ref(inner, ..) => Some((**inner).clone()),
        _ => None,
    }
}

impl<'c> LowerToMelior<'c> for DereferenceExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let (ptr_val, ptr_ty, block) = gen.generate_expr(&self.expr, block)?;
        let ptr_ty_str = ptr_ty.to_string();

        // The pointee (loaded) type: the checker-set `self.ty` when present; else the pointee recovered
        // from the pointer expression's AST type (`p : *mut i32` -> `i32`) — MLIR opaque pointers carry
        // no pointee, so the old `!llvm.ptr<…>` string-parse never fired and fell through to a wrong
        // `f32` default (a bare `*p` on an i32 pointer returned garbage). Only if inference fails too do
        // we fall back to the string-parse/`f32` path. (#242)
        let inner_ty = if let Some(t) = &self.ty {
            gen.lower_type(t)?
        } else if let Some(inner) = gen
            .infer_ast_type(&self.expr)
            .and_then(|t| deref_pointee_type(&t))
        {
            gen.lower_type(&inner)?
        } else {
            let inner_ty_str = if ptr_ty_str.starts_with("!llvm.ptr<") {
                ptr_ty_str[10..ptr_ty_str.len() - 1].to_string()
            } else {
                "f32".to_string()
            };
            Type::parse(gen.context, &inner_ty_str).ok_or_else(|| {
                crate::codegen::lower::LowerError::ParseType("Type::parse failed".to_string())
            })?
        };

        if gen.is_lvalue_context {
            return Ok((ptr_val, ptr_ty, block));
        }

        // A `&mut` to a mutable local holds that local's scalar memref, not a bare pointer.
        if ptr_ty_str.starts_with("memref<") && !ptr_ty_str.contains('x') {
            let load_op = OperationBuilder::new("memref.load", gen.loc())
                .add_operands(&[ptr_val])
                .add_results(&[inner_ty])
                .build()?;
            return Ok((
                block.append_operation(load_op).result(0)?.into(),
                inner_ty,
                block,
            ));
        }

        let load_op = OperationBuilder::new("llvm.load", gen.loc())
            .add_operands(&[ptr_val])
            .add_results(&[inner_ty])
            .build()?;
        let load_ref = block.append_operation(load_op);
        Ok((load_ref.result(0)?.into(), inner_ty, block))
    }
}

/// `t.extent(k)`, which the checker rewrote to the indexed-member form `t.$extent[k]`: the
/// run-time extent of dimension k of a tensor, as an `i32`. `None` when `this` is anything else.
///
/// # Errors
///
/// Returns an error if the tensor or the dimension number cannot be lowered.
fn extent_of_dimension<'c>(
    this: &syntax::IndexAccessExpr,
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
) -> Result<Option<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>)>, LowerError> {
    let syntax::Expr::MemberAccess(ma) = this.base.as_ref() else {
        return Ok(None);
    };
    if ma.member.as_ref() != "$extent" {
        return Ok(None);
    }
    let (t_val, t_ty, block) = gen.generate_expr(&ma.base, block)?;
    if !t_ty.to_string().starts_with("memref<") {
        return Ok(None);
    }
    let (i_val, i_ty, block) = gen.generate_expr(&this.index, block)?;
    let index_ty = Type::index(gen.context);
    let idx_val = if i_ty == index_ty {
        i_val
    } else {
        let cast_op = OperationBuilder::new("arith.index_cast", gen.loc())
            .add_operands(&[i_val])
            .add_results(&[index_ty])
            .build()?;
        block.append_operation(cast_op).result(0)?.into()
    };
    let dim_op = OperationBuilder::new("memref.dim", gen.loc())
        .add_operands(&[t_val, idx_val])
        .add_results(&[index_ty])
        .build()?;
    let dim_val: Value<'c, 'c> = block.append_operation(dim_op).result(0)?.into();
    let i32_ty = gen.i32_ty;
    let out_op = OperationBuilder::new("arith.index_cast", gen.loc())
        .add_operands(&[dim_val])
        .add_results(&[i32_ty])
        .build()?;
    let out = block.append_operation(out_op).result(0)?.into();
    Ok(Some((out, i32_ty, block)))
}

/// `p[i]` where `p` is a raw pointer: the element's address when an address is wanted,
/// otherwise the element loaded from it.
///
/// # Errors
///
/// Returns an error if the element type cannot be lowered, or if MLIR refuses an operation.
///
/// # Panics
///
/// Panics if the element type's text does not parse as an MLIR type.
fn index_through_pointer<'c>(
    this: &syntax::IndexAccessExpr,
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    base: Typed<'c>,
    index: Value<'c, 'c>,
) -> Lowered<'c> {
    let (base_val, base_ty) = base;
    let base_ty_str = base_ty.to_string();
    let mut inferred_el_ty_str = None;
    if let Some(syntax::Type::Pointer(inner, _, _)) = gen.infer_ast_type(this.base.as_ref()) {
        inferred_el_ty_str = Some(gen.lower_type_str(&inner)?);
    }

    let inner_ty_str = if let Some(open) = base_ty_str.find('<') {
        base_ty_str[open + 1..base_ty_str.len() - 1].to_string()
    } else if let Some(el_str) = inferred_el_ty_str {
        el_str
    } else if let Some(expected) = gen.expected_type {
        expected.to_string()
    } else {
        "f32".to_string()
    };
    let inner_ty = Type::parse(gen.context, &inner_ty_str)
        .unwrap_or_else(|| panic!("Failed to parse inner ptr type: {inner_ty_str}"));

    let i64_ty = gen.i64_ty;
    let cast_op = OperationBuilder::new("arith.index_cast", gen.loc())
        .add_operands(&[index])
        .add_results(&[i64_ty])
        .build()?;
    let idx_i64 = block.append_operation(cast_op).result(0)?.into();

    let gep_op = OperationBuilder::new("llvm.getelementptr", gen.loc())
        .add_attributes(&[
            (
                Identifier::new(gen.context, "rawConstantIndices"),
                DenseI32ArrayAttribute::new(gen.context, &[i32::MIN]).into(),
            ),
            (
                Identifier::new(gen.context, "elem_type"),
                TypeAttribute::new(inner_ty).into(),
            ),
        ])
        .add_operands(&[base_val, idx_i64])
        .add_results(&[base_ty])
        .build()?;

    let gep_ref = block.append_operation(gep_op);
    let ptr_val = gep_ref.result(0)?.into();

    if gen.is_lvalue_context {
        let ptr_ty = gen.ptr_ty;
        return Ok((ptr_val, ptr_ty, block));
    }

    let load_op = OperationBuilder::new("llvm.load", gen.loc())
        .add_operands(&[ptr_val])
        .add_results(&[inner_ty])
        .build()?;

    let load_ref = block.append_operation(load_op);
    Ok((load_ref.result(0)?.into(), inner_ty, block))
}

/// The element type of a tensor's memref type, as text; `f32` for anything else.
fn memref_element_text(base_ty_str: &str) -> String {
    if !base_ty_str.starts_with("memref<") {
        return "f32".to_string();
    }
    let inner = base_ty_str.replace("memref<", "").replace('>', "");
    // A split always gives at least one part, so neither default is ever used.
    let last_part = inner.rsplit('x').next().unwrap_or("");
    last_part.split(',').next().unwrap_or("").trim().to_string()
}

/// Each extent of the tensor that an index expression counts from (`a` in `a[1][2]`), or `None`
/// for an extent only known when the program runs. Empty when it is not a tensor.
fn indexed_tensor_extents(
    this: &syntax::IndexAccessExpr,
    gen: &MeliorGenerator<'_>,
) -> Vec<Option<i64>> {
    // The tensor the indices count from: `a` in `a[1][2]`, where `self.base` is `a[1]`.
    let mut root = this.base.as_ref();
    while let syntax::Expr::IndexAccess(inner) = root {
        root = inner.base.as_ref();
    }
    let base_ty = gen.infer_ast_type(root);
    let base_ty = match &base_ty {
        Some(syntax::Type::Borrow { inner, .. }) => Some(&**inner),
        other => other.as_ref(),
    };
    // Each extent, or `None` for one only known when the program runs.
    match base_ty {
        Some(syntax::Type::Tensor(_, dims, _)) => dims
            .iter()
            .map(|d| d.literal().and_then(|v| v.parse::<i64>().ok()))
            .collect(),
        _ => Vec::new(),
    }
}

/// The stride of one dimension, in elements: a number when it is known while compiling,
/// otherwise a value worked out when the program runs.
type Stride<'c> = (Option<i64>, Option<Value<'c, 'c>>);

/// The stride of each dimension of a tensor: the product of the extents after it. Known while
/// compiling when all of those are; otherwise built at run time, with each unknown extent read
/// from the buffer itself.
///
/// # Errors
///
/// Returns an error if MLIR refuses an operation built here.
///
/// # Panics
///
/// Panics if a stride has neither a known number nor a run-time value, which the loop never
/// leaves behind.
fn row_strides<'c>(
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    base_val: Value<'c, 'c>,
    base_dims: &[Option<i64>],
) -> Result<Vec<Stride<'c>>, LowerError> {
    let rank = base_dims.len();
    let index_ty = Type::index(gen.context);
    let mut strides: Vec<Stride<'c>> = vec![(Some(1), None); rank];
    for d in (0..rank - 1).rev() {
        strides[d] = match (strides[d + 1], base_dims[d + 1]) {
            ((Some(a), _), Some(b)) => (Some(a * b), None),
            ((next_static, next_val), extent) => {
                let next = match next_val {
                    Some(v) => v,
                    None => index_const(gen, block, next_static.expect("a stride"))?,
                };
                let extent = if let Some(n) = extent {
                    index_const(gen, block, n)?
                } else {
                    let axis = index_const(gen, block, count_i64(d) + 1)?;
                    let op = OperationBuilder::new("memref.dim", gen.loc())
                        .add_operands(&[base_val, axis])
                        .add_results(&[index_ty])
                        .build()?;
                    block.append_operation(op).result(0)?.into()
                };
                (
                    None,
                    Some(index_arith(gen, block, "arith.muli", next, extent)?),
                )
            }
        };
    }
    Ok(strides)
}

/// Where a row starts in the tensor's buffer: the sum of each leading index times the stride
/// of its dimension.
///
/// # Errors
///
/// Returns an error if MLIR refuses an operation built here.
///
/// # Panics
///
/// Panics if there is no index, or if a stride has neither a known number nor a run-time value.
fn row_offset<'c>(
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    indices: &[Value<'c, 'c>],
    strides: &[Stride<'c>],
) -> Result<Value<'c, 'c>, LowerError> {
    let index_ty = Type::index(gen.context);
    let mut offset_val: Option<Value<'c, 'c>> = None;
    for (m, idx) in indices.iter().enumerate() {
        let term = match strides[m] {
            (Some(1), _) => *idx,
            (Some(k), _) => {
                let k = index_const(gen, block, k)?;
                index_arith(gen, block, "arith.muli", *idx, k)?
            }
            (None, Some(v)) => index_arith(gen, block, "arith.muli", *idx, v)?,
            (None, None) => unreachable!("a run-time stride has a value"),
        };
        offset_val = Some(match offset_val {
            None => term,
            Some(acc) => {
                let add_op = OperationBuilder::new("arith.addi", gen.loc())
                    .add_operands(&[acc, term])
                    .add_results(&[index_ty])
                    .build()?;
                block.append_operation(add_op).result(0)?.into()
            }
        });
    }
    Ok(offset_val.expect("partial index has >= 1 index"))
}

/// A tensor indexed with fewer indices than it has dimensions: what is needed to make a view
/// of the rows that are left.
#[derive(Clone, Copy)]
struct RowView<'a, 'c> {
    /// The whole tensor's memref.
    base: Value<'c, 'c>,
    /// Each extent of the whole tensor, or `None` for one only known when the program runs.
    dims: &'a [Option<i64>],
    /// The stride of each dimension of the whole tensor.
    strides: &'a [Stride<'c>],
    /// How many leading dimensions the indices used up.
    taken: usize,
}

impl RowView<'_, '_> {
    /// The dimensions the view keeps.
    fn kept(&self) -> std::ops::Range<usize> {
        self.taken..self.dims.len()
    }
}

/// The memref type of a row view whose elements have the type `inner_ty_str`.
///
/// # Errors
///
/// Returns an error if the type's text does not parse.
fn row_view_type<'c>(
    gen: &MeliorGenerator<'c>,
    view: &RowView<'_, 'c>,
    inner_ty_str: &str,
) -> Result<Type<'c>, LowerError> {
    let kept = view.kept();
    let text = |v: Option<i64>| v.map_or("?".to_string(), |n| n.to_string());
    let dims_str = kept
        .clone()
        .map(|d| text(view.dims[d]))
        .collect::<Vec<_>>()
        .join("x");
    let strides_str = kept
        .clone()
        .map(|d| text(view.strides[d].0))
        .collect::<Vec<_>>()
        .join(", ");
    let result_ty_str =
        format!("memref<{dims_str}x{inner_ty_str}, strided<[{strides_str}], offset: ?>>");
    let result_ty = Type::parse(gen.context, &result_ty_str)
        .ok_or_else(|| crate::codegen::lower::LowerError::ParseType(result_ty_str.clone()))?;
    Ok(result_ty)
}

/// The run-time size of each kept dimension whose extent is not known while compiling.
///
/// # Errors
///
/// Returns an error if MLIR refuses an operation built here.
fn row_dynamic_sizes<'c>(
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    view: &RowView<'_, 'c>,
) -> Result<Vec<Value<'c, 'c>>, LowerError> {
    let index_ty = Type::index(gen.context);
    let mut dynamic_sizes: Vec<Value<'c, 'c>> = Vec::new();
    for d in view.kept().filter(|d| view.dims[*d].is_none()) {
        let axis = index_const(gen, block, count_i64(d))?;
        let op = OperationBuilder::new("memref.dim", gen.loc())
            .add_operands(&[view.base, axis])
            .add_results(&[index_ty])
            .build()?;
        dynamic_sizes.push(block.append_operation(op).result(0)?.into());
    }
    Ok(dynamic_sizes)
}

/// The view itself: a `memref.reinterpret_cast` of the whole tensor that starts at
/// `offset_val` and has the kept dimensions. Gives back the view and its type.
///
/// # Errors
///
/// Returns an error if the view's type does not parse, or if MLIR refuses an operation.
fn emit_row_view<'c>(
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    view: &RowView<'_, 'c>,
    offset_val: Value<'c, 'c>,
    inner_ty_str: &str,
) -> Result<Typed<'c>, LowerError> {
    let kept = view.kept();
    let dynamic = i64::MIN; // ShapedType::kDynamic marker
    let result_ty = row_view_type(gen, view, inner_ty_str)?;
    let static_sizes: Vec<i64> = kept
        .clone()
        .map(|d| view.dims[d].unwrap_or(dynamic))
        .collect();
    let static_strides: Vec<i64> = kept
        .clone()
        .map(|d| view.strides[d].0.unwrap_or(dynamic))
        .collect();
    let dynamic_sizes = row_dynamic_sizes(gen, block, view)?;
    let dynamic_strides: Vec<Value<'c, 'c>> = kept.filter_map(|d| view.strides[d].1).collect();
    let mut operands = vec![view.base, offset_val];
    operands.extend(&dynamic_sizes);
    operands.extend(&dynamic_strides);
    let reinterp = OperationBuilder::new("memref.reinterpret_cast", gen.loc())
        .add_operands(&operands)
        .add_attributes(&[
            (
                Identifier::new(gen.context, "operandSegmentSizes"),
                DenseI32ArrayAttribute::new(
                    gen.context,
                    &[
                        1,
                        1,
                        count_i32(dynamic_sizes.len()),
                        count_i32(dynamic_strides.len()),
                    ],
                )
                .into(),
            ),
            (
                Identifier::new(gen.context, "static_offsets"),
                DenseI64ArrayAttribute::new(gen.context, &[dynamic]).into(),
            ),
            (
                Identifier::new(gen.context, "static_sizes"),
                DenseI64ArrayAttribute::new(gen.context, &static_sizes).into(),
            ),
            (
                Identifier::new(gen.context, "static_strides"),
                DenseI64ArrayAttribute::new(gen.context, &static_strides).into(),
            ),
        ])
        .add_results(&[result_ty])
        .build()?;
    let r = block.append_operation(reinterp).result(0)?.into();
    Ok((r, result_ty))
}

impl<'c> LowerToMelior<'c> for syntax::IndexAccessExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        // `t.extent(k)`, which the checker rewrote to this indexed-member form: the runtime
        // extent of dimension k of a tensor -- `memref.dim`. The bare member has no
        // materialization; only the indexed form is a value, so it is one construct here.
        if let Some(extent) = extent_of_dimension(self, gen, block)? {
            return Ok(extent);
        }
        let (base_val, base_ty, indices, block) = gen
            .flatten_indices(&syntax::Expr::IndexAccess(self.clone()), block)
            .expect("Failed to flatten indices for IndexAccess");

        let base_ty_str = base_ty.to_string();
        let is_ptr = base_ty_str.starts_with("!llvm.ptr");

        if is_ptr {
            return index_through_pointer(self, gen, block, (base_val, base_ty), indices[0]);
        }
        let inner_ty_str = memref_element_text(&base_ty_str);

        // Slice indexing (S1): fewer indices than the tensor's rank -> a rank-reduced view
        // of the remaining dimensions (e.g. `q[i]` on Tensor<f32,[N,D]> is row i, a
        // Tensor<f32,[D]>). The memref lowers to `?x?` but the Vx type carries the static
        // shape, so we emit a `memref.reinterpret_cast` with a static row size at the flat
        // offset `sum_m idx_m * stride_m`. This is what lets a row be used as a tensor of its own.
        let base_dims = indexed_tensor_extents(self, gen);
        let rank = base_dims.len();
        if rank >= 2 && indices.len() < rank {
            let strides = row_strides(gen, block, base_val, &base_dims)?;
            let offset = row_offset(gen, block, &indices, &strides)?;
            let view = RowView {
                base: base_val,
                dims: &base_dims,
                strides: &strides,
                taken: indices.len(),
            };
            let (row, row_ty) = emit_row_view(gen, block, &view, offset, &inner_ty_str)?;
            return Ok((row, row_ty, block));
        }

        let inner_ty = Type::parse(gen.context, &inner_ty_str).ok_or_else(|| {
            crate::codegen::lower::LowerError::ParseType("Type::parse failed".to_string())
        })?;
        let mut load_builder =
            OperationBuilder::new("memref.load", gen.loc()).add_operands(&[base_val]);

        for idx in indices {
            load_builder = load_builder.add_operands(&[idx]);
        }

        let load_op = load_builder.add_results(&[inner_ty]).build()?;

        let load_ref = block.append_operation(load_op);
        Ok((load_ref.result(0)?.into(), inner_ty, block))
    }
}

/// The two lowered operands of a binary operator, each with its MLIR type.
#[derive(Clone, Copy)]
struct Operands<'c> {
    /// The left operand.
    lhs: Value<'c, 'c>,
    /// The MLIR type of the left operand.
    lhs_ty: Type<'c>,
    /// The right operand.
    rhs: Value<'c, 'c>,
    /// The MLIR type of the right operand.
    rhs_ty: Type<'c>,
}

/// `+ - * /` where an operand is a rank-1 `f32` slice or vector: one SIMD operation on
/// `vector<Dxf32>` values. The result goes to a `vector.store` where it is assigned (see
/// stmt.rs). `None` for any other operator or operands.
///
/// # Errors
///
/// Returns an error if an operand cannot be made a vector, or if MLIR refuses the operation.
fn slice_elementwise<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    op: &BinaryOp,
    ops: Operands<'c>,
) -> Result<Option<Typed<'c>>, LowerError> {
    let Operands {
        lhs: lhs_val,
        lhs_ty,
        rhs: rhs_val,
        rhs_ty,
    } = ops;
    if matches!(
        op,
        BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div
    ) {
        let lhs_ty_s = lhs_ty.to_string();
        let rhs_ty_s = rhs_ty.to_string();
        let is_slice = is_slice_operand(&lhs_ty_s) || is_slice_operand(&rhs_ty_s);
        if let Some(d) = is_slice
            .then(|| slice_vec_len(&lhs_ty_s).or_else(|| slice_vec_len(&rhs_ty_s)))
            .flatten()
        {
            let va = to_vector(gen, lhs_val, &lhs_ty_s, d, block)?;
            let vb = to_vector(gen, rhs_val, &rhs_ty_s, d, block)?;
            let vec_ty = Type::parse(gen.context, &format!("vector<{d}xf32>"))
                .ok_or_else(|| LowerError::ParseType(format!("vector<{d}xf32>")))?;
            let op_name = match op {
                BinaryOp::Add => "arith.addf",
                BinaryOp::Sub => "arith.subf",
                BinaryOp::Mul => "arith.mulf",
                BinaryOp::Div => "arith.divf",
                // Both are excluded by the `matches!` guard above: `@` is a matmul, and
                // `%` has no vectorized slice form.
                BinaryOp::MatMul
                | BinaryOp::Rem
                | BinaryOp::BitAnd
                | BinaryOp::BitOr
                | BinaryOp::BitXor
                | BinaryOp::Shl
                | BinaryOp::Shr => unreachable!(),
            };
            let arith_op = OperationBuilder::new(op_name, gen.loc())
                .add_operands(&[va, vb])
                .add_results(&[vec_ty])
                .build()?;
            let res: Value<'c, 'c> = block.append_operation(arith_op).result(0)?.into();
            return Ok(Some((res, vec_ty)));
        }
    }
    Ok(None)
}

/// Two tensors that a binary operator works on, with their memref types split at each `x`
/// (`4x?xf32` gives `4`, `?`, `f32`).
#[derive(Clone, Copy)]
struct TensorPair<'a, 'c> {
    /// The left tensor.
    lhs: Value<'c, 'c>,
    /// The right tensor.
    rhs: Value<'c, 'c>,
    /// The left tensor's memref type, as text.
    lhs_ty_str: &'a str,
    /// The parts of the left tensor's type.
    lhs_parts: &'a [&'a str],
    /// The parts of the right tensor's type.
    rhs_parts: &'a [&'a str],
    /// The caller's buffer for the result, when this result is what the function returns.
    nrvo_slot: Option<Value<'c, 'c>>,
}

/// The run-time extent of dimension `axis` of a tensor.
///
/// # Errors
///
/// Returns an error if MLIR refuses an operation built here.
fn extent_at_run_time<'c>(
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    tensor: Value<'c, 'c>,
    axis: i64,
) -> Result<Value<'c, 'c>, LowerError> {
    let index_ty = gen.index_ty;
    let idx_attr = IntegerAttribute::new(Type::index(gen.context), axis).into();
    let cst_op = OperationBuilder::new("arith.constant", gen.loc())
        .add_results(&[index_ty])
        .add_attributes(&[(Identifier::new(gen.context, "value"), idx_attr)])
        .build()?;
    let idx_val = block.append_operation(cst_op).result(0)?.into();

    let dim_op = OperationBuilder::new("memref.dim", gen.loc())
        .add_operands(&[tensor, idx_val])
        .add_results(&[index_ty])
        .build()?;
    Ok(block.append_operation(dim_op).result(0)?.into())
}

/// The buffer a tensor result is written to: the caller's when this result is what the
/// function returns and the types agree, otherwise a new one. `alloc_operands` holds the
/// run-time extents a new buffer needs.
///
/// # Errors
///
/// Returns an error if MLIR refuses an operation built here.
fn result_buffer<'c>(
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    nrvo_slot: Option<Value<'c, 'c>>,
    out_ty: Type<'c>,
    alloc_operands: &[Value<'c, 'c>],
) -> Result<Value<'c, 'c>, LowerError> {
    if let Some(slot) = nrvo_slot.filter(|s| s.r#type() == out_ty) {
        return Ok(slot);
    }
    let alloc_op = OperationBuilder::new("memref.alloc", gen.loc())
        .add_operands(alloc_operands)
        .add_attributes(&[(
            Identifier::new(gen.context, "operandSegmentSizes"),
            DenseI32ArrayAttribute::new(gen.context, &[count_i32(alloc_operands.len()), 0]).into(),
        )])
        .add_results(&[out_ty])
        .build()?;
    Ok(block.append_operation(alloc_op).result(0)?.into())
}

/// The constant zero of the element type `el_ty_str`.
///
/// # Errors
///
/// Returns an error if the element type does not parse, or if MLIR refuses the constant.
fn zero_of_element_type<'c>(
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    el_ty_str: &str,
) -> Result<Value<'c, 'c>, LowerError> {
    let el_ty = parse_element_type(gen, el_ty_str)?;
    let zero_attr = if el_ty_str.starts_with('i') {
        IntegerAttribute::new(el_ty, 0).into()
    } else {
        FloatAttribute::new(gen.context, el_ty, 0.0).into()
    };
    let zero_op = OperationBuilder::new("arith.constant", gen.loc())
        .add_results(&[el_ty])
        .add_attributes(&[(Identifier::new(gen.context, "value"), zero_attr)])
        .build()?;
    Ok(block.append_operation(zero_op).result(0)?.into())
}

/// Fills the tensor `out_val` with `fill_val`, using `linalg.fill`.
///
/// # Errors
///
/// Returns an error if the element type does not parse, or if MLIR refuses an operation.
fn emit_linalg_fill<'c>(
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    el_ty_str: &str,
    fill_val: Value<'c, 'c>,
    out_val: Value<'c, 'c>,
) -> Result<(), LowerError> {
    let region_fill = Region::new();
    let block_fill = element_block(gen, el_ty_str, 2)?;
    let yield_fill = OperationBuilder::new("linalg.yield", gen.loc())
        .add_operands(&[block_arg(&block_fill, 0)?])
        .build()?;
    block_fill.append_operation(yield_fill);
    region_fill.append_block(block_fill);

    let linalg_fill = OperationBuilder::new("linalg.fill", gen.loc())
        .add_operands(&[fill_val, out_val])
        .add_attributes(&[(
            Identifier::new(gen.context, "operandSegmentSizes"),
            DenseI32ArrayAttribute::new(gen.context, &[1, 1]).into(),
        )])
        .add_regions([region_fill])
        .build()?;
    block.append_operation(linalg_fill);
    Ok(())
}

/// The body of a `linalg.matmul`: `out + a * b` on one element of each tensor.
///
/// # Errors
///
/// Returns an error if the element type does not parse, or if MLIR refuses an operation.
fn matmul_region<'c>(gen: &MeliorGenerator<'c>, el_ty_str: &str) -> Result<Region<'c>, LowerError> {
    let region_matmul = Region::new();
    let block_matmul = element_block(gen, el_ty_str, 3)?;
    let el_ty = parse_element_type(gen, el_ty_str)?;

    let is_float = el_ty_str.contains("f32")
        || el_ty_str.contains("f64")
        || el_ty_str.contains("f16")
        || el_ty_str.contains("bf16");
    let mul_op_name = if is_float { "arith.mulf" } else { "arith.muli" };
    let add_op_name = if is_float { "arith.addf" } else { "arith.addi" };

    let mul_op = OperationBuilder::new(mul_op_name, gen.loc())
        .add_operands(&[block_arg(&block_matmul, 0)?, block_arg(&block_matmul, 1)?])
        .add_results(&[el_ty])
        .build()?;
    let mul_val = block_matmul.append_operation(mul_op).result(0)?.into();

    let add_op = OperationBuilder::new(add_op_name, gen.loc())
        .add_operands(&[block_arg(&block_matmul, 2)?, mul_val])
        .add_results(&[el_ty])
        .build()?;
    let add_val = block_matmul.append_operation(add_op).result(0)?.into();

    let yield_matmul = OperationBuilder::new("linalg.yield", gen.loc())
        .add_operands(&[add_val])
        .build()?;
    block_matmul.append_operation(yield_matmul);
    region_matmul.append_block(block_matmul);
    Ok(region_matmul)
}

/// `a @ b` on two rank-2 tensors: a zeroed result buffer, then `linalg.matmul` into it.
///
/// # Errors
///
/// Returns an error if a type does not parse, or if MLIR refuses an operation.
fn lower_matmul<'c>(
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    pair: &TensorPair<'_, 'c>,
) -> Result<Typed<'c>, LowerError> {
    let m_str = pair.lhs_parts[0];
    let n_str = pair.rhs_parts[1];
    let el_part = pair.lhs_parts[2];
    let el_ty_str = el_part
        .split_once(',')
        .map_or(el_part, |(head, _)| head)
        .trim();

    let out_ty_str = format!("memref<{m_str}x{n_str}x{el_ty_str}>");
    let out_ty = Type::parse(gen.context, &out_ty_str).ok_or_else(|| {
        crate::codegen::lower::LowerError::ParseType("Type::parse failed".to_string())
    })?;

    // Determine dynamic dimensions for alloc
    let mut alloc_operands = Vec::new();
    if m_str == "?" {
        alloc_operands.push(extent_at_run_time(gen, block, pair.lhs, 0)?);
    }
    if n_str == "?" {
        alloc_operands.push(extent_at_run_time(gen, block, pair.rhs, 1)?);
    }
    let out_val = result_buffer(gen, block, pair.nrvo_slot, out_ty, &alloc_operands)?;

    // Zero initialize the output buffer since matmul accumulates!
    let zero_val = zero_of_element_type(gen, block, el_ty_str)?;
    emit_linalg_fill(gen, block, el_ty_str, zero_val, out_val)?;

    // Execute linalg.matmul
    let region_matmul = matmul_region(gen, el_ty_str)?;
    let matmul_op = OperationBuilder::new("linalg.matmul", gen.loc())
        .add_operands(&[pair.lhs, pair.rhs, out_val])
        .add_attributes(&[(
            Identifier::new(gen.context, "operandSegmentSizes"),
            DenseI32ArrayAttribute::new(gen.context, &[2, 1]).into(),
        )])
        .add_regions([region_matmul])
        .build()?;
    block.append_operation(matmul_op);

    Ok((out_val, out_ty))
}

/// The named `linalg` operation for an element-by-element operator on tensors.
///
/// # Panics
///
/// Panics for `@`, which is lowered before this is asked, and for an operator the checker
/// refuses on tensors.
fn linalg_op_name(op: &BinaryOp) -> &'static str {
    match op {
        BinaryOp::Add => "linalg.add",
        BinaryOp::Sub => "linalg.sub",
        BinaryOp::Mul => "linalg.mul",
        BinaryOp::MatMul => panic!("MatMul must have been handled by is_matmul branch"),
        BinaryOp::Div => "linalg.div",
        // The checker refuses a tensor operand to `%` (E3030): there is no named
        // `linalg` remainder to lower it to.
        BinaryOp::Rem
        | BinaryOp::BitAnd
        | BinaryOp::BitOr
        | BinaryOp::BitXor
        | BinaryOp::Shl
        | BinaryOp::Shr => {
            panic!("this operator on tensors must have been refused by the checker")
        }
    }
}

/// The body of an element-by-element `linalg` operation: `arith_op_name` on one element of
/// each tensor.
///
/// # Errors
///
/// Returns an error if the element type does not parse, or if MLIR refuses an operation.
fn elementwise_region<'c>(
    gen: &MeliorGenerator<'c>,
    arith_op_name: &str,
    el_ty_str: &str,
) -> Result<Region<'c>, LowerError> {
    let region = Region::new();
    let block_inner = element_block(gen, el_ty_str, 3)?;

    let arith_op = OperationBuilder::new(arith_op_name, gen.loc())
        .add_operands(&[block_arg(&block_inner, 0)?, block_arg(&block_inner, 1)?])
        .add_results(&[parse_element_type(gen, el_ty_str)?])
        .build()?;

    let arith_val = block_inner.append_operation(arith_op).result(0)?.into();

    let yield_op = OperationBuilder::new("linalg.yield", gen.loc())
        .add_operands(&[arith_val])
        .build()?;

    block_inner.append_operation(yield_op);
    region.append_block(block_inner);
    Ok(region)
}

/// `+ - * /` on two tensors of the same shape: a result buffer, then the named `linalg`
/// operation into it.
///
/// # Errors
///
/// Returns an error if a type does not parse, or if MLIR refuses an operation.
///
/// # Panics
///
/// Panics if the left type has no parts (a split always gives one), or for an operator that
/// [`linalg_op_name`] refuses.
fn lower_tensor_elementwise<'c>(
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    op: &BinaryOp,
    pair: &TensorPair<'_, 'c>,
) -> Result<Typed<'c>, LowerError> {
    let out_ty = Type::parse(gen.context, pair.lhs_ty_str).ok_or_else(|| {
        crate::codegen::lower::LowerError::ParseType("Type::parse failed".to_string())
    })?;

    let mut alloc_operands = Vec::new();
    let rank = pair.lhs_parts.len() - 1; // Last part is element type
    let el_ty_str = pair.lhs_parts.last().unwrap();
    for (i, dim_str) in pair.lhs_parts.iter().take(rank).enumerate() {
        if *dim_str == "?" {
            alloc_operands.push(extent_at_run_time(gen, block, pair.lhs, count_i64(i))?);
        }
    }
    let out_val = result_buffer(gen, block, pair.nrvo_slot, out_ty, &alloc_operands)?;

    let op_name = linalg_op_name(op);

    let is_float = el_ty_str.contains("f32")
        || el_ty_str.contains("f64")
        || el_ty_str.contains("f16")
        || el_ty_str.contains("bf16");
    let arith_op_name = op.get_op_name(is_float, false);

    let region = elementwise_region(gen, arith_op_name, el_ty_str)?;

    let linalg_op = OperationBuilder::new(op_name, gen.loc())
        .add_operands(&[pair.lhs, pair.rhs, out_val])
        .add_attributes(&[(
            Identifier::new(gen.context, "operandSegmentSizes"),
            DenseI32ArrayAttribute::new(gen.context, &[2, 1]).into(),
        )])
        .add_regions([region])
        .build()?;
    block.append_operation(linalg_op);

    Ok((out_val, out_ty))
}

/// A binary operator on two tensors: `@` as a matrix product, the others element by element.
/// `None` when the operands are not both tensors.
///
/// # Errors
///
/// Returns an error if a type does not parse, or if MLIR refuses an operation.
fn lower_tensor_binary<'c>(
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    op: &BinaryOp,
    ops: Operands<'c>,
    nrvo_slot: Option<Value<'c, 'c>>,
) -> Result<Option<Typed<'c>>, LowerError> {
    if !(gen.is_memref(&ops.lhs_ty) && gen.is_memref(&ops.rhs_ty)) {
        return Ok(None);
    }
    let lhs_ty_str = ops.lhs_ty.to_string();
    let rhs_ty_str = ops.rhs_ty.to_string();

    let lhs_inner = lhs_ty_str.replace("memref<", "").replace('>', "");
    let rhs_inner = rhs_ty_str.replace("memref<", "").replace('>', "");
    let lhs_parts: Vec<&str> = lhs_inner.split('x').collect();
    let rhs_parts: Vec<&str> = rhs_inner.split('x').collect();
    let is_matmul = if op == &BinaryOp::MatMul && lhs_parts.len() == 3 && rhs_parts.len() == 3 {
        let lhs_dim1 = lhs_parts[1];
        let rhs_dim0 = rhs_parts[0];
        lhs_dim1 == rhs_dim0 || lhs_dim1 == "?" || rhs_dim0 == "?"
    } else {
        false
    };
    let pair = TensorPair {
        lhs: ops.lhs,
        rhs: ops.rhs,
        lhs_ty_str: &lhs_ty_str,
        lhs_parts: &lhs_parts,
        rhs_parts: &rhs_parts,
        nrvo_slot,
    };
    if is_matmul {
        lower_matmul(gen, block, &pair).map(Some)
    } else {
        // Element-wise Linalg Lowering (Add, Sub, Mul, Div)
        lower_tensor_elementwise(gen, block, op, &pair).map(Some)
    }
}

/// A binary operator on two scalars: one `arith` operation, or a comparison that gives an `i1`.
///
/// # Errors
///
/// Returns an error if MLIR refuses an operation built here.
fn lower_scalar_binary<'c>(
    this: &BinaryOpExpr,
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    ops: Operands<'c>,
) -> Result<Typed<'c>, LowerError> {
    let Operands {
        lhs: mut lhs_val,
        lhs_ty,
        rhs: mut rhs_val,
        rhs_ty,
    } = ops;
    let BinaryOpExpr { op, operand_ty, .. } = this;
    let mut final_ty = lhs_ty;
    let lhs_ty_str = lhs_ty.to_string();
    let rhs_ty_str = rhs_ty.to_string();

    if lhs_ty != rhs_ty
        && ((lhs_ty_str == "index" && rhs_ty_str == "i32")
            || (lhs_ty_str == "i32" && rhs_ty_str == "index"))
    {
        if lhs_ty_str == "index" && rhs_ty_str == "i32" {
            let cast_op = OperationBuilder::new("arith.index_cast", gen.loc())
                .add_operands(&[rhs_val])
                .add_results(&[lhs_ty])
                .build()?;
            rhs_val = block.append_operation(cast_op).result(0)?.into();
        } else {
            let cast_op = OperationBuilder::new("arith.index_cast", gen.loc())
                .add_operands(&[lhs_val])
                .add_results(&[rhs_ty])
                .build()?;
            lhs_val = block.append_operation(cast_op).result(0)?.into();
            final_ty = rhs_ty;
        }
    }

    let is_float = final_ty.to_string().contains("f32")
        || final_ty.to_string().contains("f64")
        || final_ty.to_string().contains("f16")
        || final_ty.to_string().contains("bf16");

    // The checker recorded the operands' element type. An MLIR integer has no
    // sign, so this is the only thing that tells `/`, `%` and `>>` which form to use.
    let is_unsigned = matches!(operand_ty, Some(e) if !e.is_float() && !e.is_signed_int());
    if *operand_ty == Some(syntax::ElementType::USize) {
        if let Some(v) =
            crate::codegen::lower::lower_usize_checked(gen, &block, op, lhs_val, rhs_val)?
        {
            return Ok((v, final_ty));
        }
    }
    let mut builder = OperationBuilder::new(op.get_op_name(is_float, is_unsigned), gen.loc());
    builder = builder.add_operands(&[lhs_val, rhs_val]);

    let ret_ty = if let Some(pred_val) = op.get_predicate(is_float, is_unsigned) {
        let i1_ty = gen.i1_ty;
        let i64_ty = gen.i64_ty;
        builder = builder.add_results(&[i1_ty]).add_attributes(&[(
            Identifier::new(gen.context, "predicate"),
            IntegerAttribute::new(i64_ty, pred_val).into(),
        )]);
        i1_ty
    } else {
        builder = builder.add_results(&[final_ty]);
        final_ty
    };

    let bin_op = builder.build()?;
    let bin_ref = block.append_operation(bin_op);
    Ok((bin_ref.result(0)?.into(), ret_ty))
}

impl<'c> LowerToMelior<'c> for BinaryOpExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let BinaryOpExpr { lhs, op, rhs, .. } = self;
        // Claim the return slot before lowering the operands, so a nested operator inside them
        // does not take it and build its intermediate in the caller's buffer.
        let nrvo_slot = gen.nrvo_slot.take();
        let (mut lhs_val, mut lhs_ty, block) = gen.generate_expr(lhs, block)?;
        let prev_expected = gen.expected_type;
        gen.expected_type = Some(lhs_ty);
        let (mut rhs_val, mut rhs_ty, block) = gen.generate_expr(rhs, block)?;
        gen.expected_type = prev_expected;

        // Slice elementwise (S3): if either operand is a rank-1 f32 slice/vector, lower to
        // `vector.load`/`vector.broadcast` + `arith.{mulf,addf,subf,divf}` -> `vector<Dxf32>`
        // (SIMD). The result flows to a `vector.store` at the assignment site (see stmt.rs).
        let ops = Operands {
            lhs: lhs_val,
            lhs_ty,
            rhs: rhs_val,
            rhs_ty,
        };
        if let Some((val, ty)) = slice_elementwise(gen, block, op, ops)? {
            return Ok((val, ty, block));
        }

        // A borrowed tensor arrives as the slot holding it --
        // `memref<memref<?x?xf32>>` -- because a borrow is the address and not
        // the value. `&a @ &b` is the non-consuming spelling of `a @ b` (#335)
        // and has to lower to the same operation, so load the descriptor out.
        //
        // Without this the operand reached `linalg.matmul` as a memref of
        // memref. The shape check below reads the type as a string and would
        // have accepted it -- `memref<memref<?x?xf32>>` still yields three parts
        // once the wrappers are stripped -- so the mistake surfaced as MLIR
        // verification failing rather than as a rejected program.
        if op == &BinaryOp::MatMul {
            lhs_val = load_tensor_slot(gen, &block, lhs_val, &mut lhs_ty)?;
            rhs_val = load_tensor_slot(gen, &block, rhs_val, &mut rhs_ty)?;
        }

        if lhs_ty != rhs_ty && op != &BinaryOp::MatMul {
            // Priority coercion: f64 > f32 > i64 > i32
            // To simplify, we'll cast rhs to lhs for now.
            rhs_val = gen.coerce_type(&block, rhs_val, rhs_ty, lhs_ty)?;
            rhs_ty = lhs_ty;
        }

        let ops = Operands {
            lhs: lhs_val,
            lhs_ty,
            rhs: rhs_val,
            rhs_ty,
        };
        if let Some((val, ty)) = lower_tensor_binary(gen, block, op, ops, nrvo_slot)? {
            return Ok((val, ty, block));
        }
        let (val, ty) = lower_scalar_binary(self, gen, block, ops)?;
        Ok((val, ty, block))
    }
}

impl<'c> LowerToMelior<'c> for RelationalOpExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let RelationalOpExpr {
            lhs,
            op,
            rhs,
            span: _,
            operand_ty,
        } = self;
        let (lhs_val, lhs_ty, block) = gen.generate_expr(lhs, block)?;
        let (lhs_val, lhs_ty) = load_rank_zero(gen, block, lhs_val, lhs_ty)?;
        let prev_expected = gen.expected_type;
        gen.expected_type = Some(lhs_ty);
        let (rhs_val, rhs_ty, block) = gen.generate_expr(rhs, block)?;
        let (mut rhs_val, rhs_ty) = load_rank_zero(gen, block, rhs_val, rhs_ty)?;
        gen.expected_type = prev_expected;

        let _lhs_ty_str = lhs_ty.to_string();
        let _rhs_ty_str = rhs_ty.to_string();

        let mut final_ty = lhs_ty;

        if lhs_ty != rhs_ty {
            // Prioritize standard coercion depending on which type is more generic (e.g. f64 > f32 > i64 > i32)
            // For simplicity, just cast rhs to lhs for now.
            rhs_val = gen.coerce_type(&block, rhs_val, rhs_ty, lhs_ty)?;
            _ = lhs_ty;
            final_ty = lhs_ty;
        }

        let is_float = final_ty.to_string().contains("f32")
            || final_ty.to_string().contains("f64")
            || final_ty.to_string().contains("f16")
            || final_ty.to_string().contains("bf16");

        // `arith.cmpi` takes only integers. Two raw pointers compare their addresses, as
        // unsigned numbers.
        if final_ty == gen.ptr_ty {
            let pred = op
                .get_predicate(false, true)
                .expect("every relational operator has a predicate");
            let i1_ty = gen.i1_ty;
            let cmp = OperationBuilder::new("llvm.icmp", gen.loc())
                .add_operands(&[lhs_val, rhs_val])
                .add_results(&[i1_ty])
                .add_attributes(&[(
                    Identifier::new(gen.context, "predicate"),
                    IntegerAttribute::new(gen.i64_ty, pred).into(),
                )])
                .build()?;
            let cmp_ref = block.append_operation(cmp);
            return Ok((cmp_ref.result(0)?.into(), i1_ty, block));
        }

        let is_unsigned = matches!(operand_ty, Some(e) if !e.is_float() && !e.is_signed_int());
        let mut builder = OperationBuilder::new(op.get_op_name(is_float, false), gen.loc());
        builder = builder.add_operands(&[lhs_val, rhs_val]);

        let ret_ty = if let Some(pred_val) = op.get_predicate(is_float, is_unsigned) {
            let i1_ty = gen.i1_ty;
            let i64_ty = gen.i64_ty;
            builder = builder.add_results(&[i1_ty]).add_attributes(&[(
                Identifier::new(gen.context, "predicate"),
                IntegerAttribute::new(i64_ty, pred_val).into(),
            )]);
            i1_ty
        } else {
            builder = builder.add_results(&[final_ty]);
            final_ty
        };

        let bin_op = builder.build()?;
        let bin_ref = block.append_operation(bin_op);
        Ok((bin_ref.result(0)?.into(), ret_ty, block))
    }
}

impl<'c> LowerToMelior<'c> for LogicalOpExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let LogicalOpExpr {
            lhs,
            op,
            rhs,
            span: _,
        } = self;
        // The right side runs only when the left side does not decide the result, so a guard
        // like `i > 0 && v.get(i - 1) == 0` never reads index -1.
        let (lhs_val, _lhs_ty, block) = gen.generate_expr(lhs, block)?;
        let final_ty = gen.i1_ty;
        let parent_region = block.parent_region().unwrap();
        let rhs_block = parent_region.append_block(melior::ir::Block::new(&[]));
        let merge_block =
            parent_region.append_block(melior::ir::Block::new(&[(final_ty, gen.loc())]));

        // `a && b` is `a` when `a` is false; `a || b` is `a` when `a` is true.
        let (true_dest, false_dest, true_args, false_args): (
            _,
            _,
            &[Value<'_, '_>],
            &[Value<'_, '_>],
        ) = match op {
            syntax::LogicalOp::And => (&*rhs_block, &*merge_block, &[], &[lhs_val]),
            syntax::LogicalOp::Or => (&*merge_block, &*rhs_block, &[lhs_val], &[]),
        };
        let operands: Vec<Value<'_, '_>> = std::iter::once(lhs_val)
            .chain(true_args.iter().copied())
            .chain(false_args.iter().copied())
            .collect();
        block.append_operation(
            OperationBuilder::new("cf.cond_br", gen.loc())
                .add_operands(&operands)
                .add_successors(&[true_dest, false_dest])
                .add_attributes(&[(
                    Identifier::new(gen.context, "operandSegmentSizes"),
                    melior::ir::attribute::DenseI32ArrayAttribute::new(
                        gen.context,
                        &[1, count_i32(true_args.len()), count_i32(false_args.len())],
                    )
                    .into(),
                )])
                .build()?,
        );

        let (rhs_val, _rhs_ty, rhs_end) = gen.generate_expr(rhs, rhs_block)?;
        rhs_end.append_operation(
            OperationBuilder::new("cf.br", gen.loc())
                .add_operands(&[rhs_val])
                .add_successors(&[&*merge_block])
                .build()?,
        );

        Ok((merge_block.argument(0)?.into(), final_ty, merge_block))
    }
}

impl<'c> LowerToMelior<'c> for syntax::UnaryOpExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let syntax::UnaryOpExpr { op, expr, span: _ } = self;
        let (val, ty, block) = gen.generate_expr(expr, block)?;
        match op {
            syntax::UnaryOp::Not => {
                let true_val_op = OperationBuilder::new("arith.constant", gen.loc())
                    .add_results(&[ty])
                    .add_attributes(&[(
                        Identifier::new(gen.context, "value"),
                        // All ones at the operand's width: `true` for a `bool`, the bitwise
                        // complement for an integer. `1` is all ones only at one bit.
                        IntegerAttribute::new(ty, -1).into(),
                    )])
                    .build()?;
                let true_val_ref = block.append_operation(true_val_op);

                let not_op = OperationBuilder::new("arith.xori", gen.loc())
                    .add_operands(&[val, true_val_ref.result(0)?.into()])
                    .add_results(&[ty])
                    .build()?;

                let not_ref = block.append_operation(not_op);
                Ok((not_ref.result(0)?.into(), ty, block))
            }
            syntax::UnaryOp::Neg if gen.is_memref(&ty) => lower_tensor_negate(gen, block, val, ty),
            syntax::UnaryOp::Neg => {
                let is_float = ty.to_string().contains("f32")
                    || ty.to_string().contains("f64")
                    || ty.to_string().contains("f16")
                    || ty.to_string().contains("bf16");
                if is_float {
                    let neg_op = OperationBuilder::new("arith.negf", gen.loc())
                        .add_operands(&[val])
                        .add_results(&[ty])
                        .build()?;
                    let neg_ref = block.append_operation(neg_op);
                    Ok((neg_ref.result(0)?.into(), ty, block))
                } else {
                    let zero_op = OperationBuilder::new("arith.constant", gen.loc())
                        .add_results(&[ty])
                        .add_attributes(&[(
                            Identifier::new(gen.context, "value"),
                            IntegerAttribute::new(ty, 0).into(),
                        )])
                        .build()?;
                    let zero_ref = block.append_operation(zero_op);

                    let neg_op = OperationBuilder::new("arith.subi", gen.loc())
                        .add_operands(&[zero_ref.result(0)?.into(), val])
                        .add_results(&[ty])
                        .build()?;
                    let neg_ref = block.append_operation(neg_op);
                    Ok((neg_ref.result(0)?.into(), ty, block))
                }
            }
        }
    }
}

/// Which type each generic parameter of a struct stands for.
type TypeArgs = std::collections::HashMap<crate::symbol::Symbol, syntax::Type>;

/// The type arguments written in a struct name like `Pair<i32, f32>`. `None` when the name
/// has none.
fn written_type_args(name: &str) -> Option<Vec<syntax::Type>> {
    let open = name.find('<')?;
    if !name.ends_with('>') {
        return None;
    }
    let inner_ty_str = &name[open + 1..name.len() - 1];
    let mut inner_tys = Vec::new();
    for ty_arg in crate::syntax::split_type_args(inner_ty_str) {
        inner_tys.push(crate::codegen::lower::parse_type_arg(ty_arg));
    }
    Some(inner_tys)
}

/// Pairs each generic parameter of a struct with the type argument at the same position.
fn generic_mapping(struct_decl: &syntax::StructDecl, inner_tys: &[syntax::Type]) -> TypeArgs {
    let mut mapping = TypeArgs::new();
    for (i, param) in struct_decl.generics.iter().enumerate() {
        if i < inner_tys.len() {
            mapping.insert(param.name().into(), inner_tys[i].clone());
        }
    }
    mapping
}

/// The value given to one field of a struct literal, made ready to be stored in a field of
/// the Vx type `sub_ty`. Gives back the value and the block the next operation goes in.
///
/// # Errors
///
/// Returns an error if the field's type or its value cannot be lowered.
fn lower_field_value<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    f_expr: &Expr,
    sub_ty: &syntax::Type,
) -> Result<(Value<'c, 'c>, melior::ir::BlockRef<'c, 'c>), LowerError> {
    let field_ty = gen.lower_type(sub_ty)?;
    let prev_expected = gen.expected_type;
    gen.expected_type = Some(field_ty);
    let (mut field_val, expr_ty, block) = gen.generate_expr(f_expr, block)?;
    gen.expected_type = prev_expected;

    if expr_ty != field_ty
        && ((expr_ty.to_string() == "index" && field_ty.to_string() == "i32")
            || (expr_ty.to_string() == "i32" && field_ty.to_string() == "index"))
    {
        let cast_op = OperationBuilder::new("arith.index_cast", gen.loc())
            .add_operands(&[field_val])
            .add_results(&[field_ty])
            .build()?;
        field_val = block.append_operation(cast_op).result(0)?.into();
    } else if expr_ty != field_ty {
        // A closure literal given to a `ClosureK` field: build the `{env, func}` pair,
        // as a function argument of that type already gets.
        if let Some(adapted) = gen.adapt_closure_to_nominal(&block, field_val, expr_ty, field_ty)? {
            field_val = adapted;
        }
    }

    if let Some(rank) = crate::codegen::generator::tensor_field_rank(sub_ty) {
        field_val = tensor_into_field(gen, &block, field_val, expr_ty, field_ty, rank)?;
    }
    Ok((field_val, block))
}

impl<'c> LowerToMelior<'c> for StructInitExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let StructInitExpr {
            name,
            fields,
            type_id: _,
            span: _,
        } = self;
        let base_name = name.split('<').next().unwrap_or(name).to_string();
        let struct_decl = gen
            .structs
            .get(base_name.as_str())
            .unwrap_or_else(|| {
                panic!(
                    "Struct {} not found in gen.structs! Available: {:?}",
                    base_name,
                    gen.structs.keys().collect::<Vec<_>>()
                )
            })
            .clone();

        let inner_tys = written_type_args(name);
        let mapping = generic_mapping(&struct_decl, inner_tys.as_deref().unwrap_or(&[]));
        let struct_ty = if let Some(inner_tys) = inner_tys {
            gen.lower_type(&syntax::Type::GenericInstance(
                Box::new(syntax::Type::Struct(base_name.clone().into(), None)),
                inner_tys,
            ))?
        } else {
            gen.lower_type(&syntax::Type::Struct(name.clone(), None))?
        };

        let undef_op = OperationBuilder::new("llvm.mlir.undef", gen.loc())
            .add_results(&[struct_ty])
            .build()?;
        let mut current_struct = block.append_operation(undef_op).result(0)?.into();

        for (field_name, f_expr) in fields {
            let field_idx = struct_decl
                .fields
                .iter()
                .position(|(n, _)| n == field_name)
                .unwrap();
            let sub_ty = struct_decl.fields[field_idx].1.substitute(&mapping);

            let (field_val, block) = lower_field_value(gen, block, f_expr, &sub_ty)?;

            let pos_attr = melior::ir::attribute::DenseI64ArrayAttribute::new(
                gen.context,
                &[count_i64(field_idx)],
            );

            let insert_op = OperationBuilder::new("llvm.insertvalue", gen.loc())
                .add_operands(&[current_struct, field_val])
                .add_attributes(&[(Identifier::new(gen.context, "position"), pos_attr.into())])
                .add_results(&[struct_ty])
                .build()?;
            current_struct = block.append_operation(insert_op).result(0)?.into();
        }
        Ok((current_struct, struct_ty, block))
    }
}

impl<'c> LowerToMelior<'c> for UnsafeBlockExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    /// A block: a `let` in it hides an outer variable of the same name only until it ends.
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let depth = gen.open_block();
        let lowered = lower_unsafe_block(self, gen, block);
        gen.close_blocks_to(depth);
        lowered
    }
}

/// The statements of an `unsafe` block in order, then its value.
///
/// # Errors
///
/// Returns an error if a statement or the final expression cannot be lowered.
fn lower_unsafe_block<'c>(
    this: &UnsafeBlockExpr,
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
) -> Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError> {
    let mut current_block = block;
    for stmt in &this.stmts {
        if let Some(b) = gen.generate_statement(stmt, current_block)? {
            current_block = b;
        } else {
            // Block was terminated (e.g., by break/continue/return). The
            // tail expression and the placeholder below still have to be
            // emitted somewhere, and it cannot be after that terminator.
            current_block = super::dead_continuation(current_block);
            break;
        }
    }
    if let Some(ret_expr) = &this.ret {
        let (val, ty, cur) = gen.generate_expr(ret_expr, current_block)?;
        let cur = gen.run_queued_drops(cur)?;
        Ok((val, ty, cur))
    } else {
        // Return an i32 0 or something empty if no return type is expected.
        let i32_ty = gen.i32_ty;
        let zero_attr = IntegerAttribute::new(i32_ty, 0).into();
        let zero_op = OperationBuilder::new("arith.constant", gen.loc())
            .add_results(&[i32_ty])
            .add_attributes(&[(Identifier::new(gen.context, "value"), zero_attr)])
            .build()?;
        let zero_val = current_block.append_operation(zero_op).result(0)?.into();
        Ok((zero_val, i32_ty, current_block))
    }
}

/// The struct a member access reads from: the name the checker recorded, or else the name in
/// quotes in the base's MLIR type.
fn struct_name_of_base(
    struct_name: Option<&crate::symbol::Symbol>,
    base_ty_str: &str,
) -> Option<crate::symbol::Symbol> {
    if let Some(name) = struct_name {
        return Some(name.clone());
    }
    let start_idx = base_ty_str.find('\"')?;
    let end_idx = base_ty_str[start_idx + 1..].find('"')?;
    Some(
        base_ty_str[start_idx + 1..start_idx + 1 + end_idx]
            .to_string()
            .into(),
    )
}

/// A struct name like `Pair<i32, f32>` split into the struct's own name and the type each of
/// its generic parameters stands for.
fn struct_and_type_args(
    gen: &MeliorGenerator<'_>,
    resolved_struct_name: &str,
) -> (String, TypeArgs) {
    let base_name = resolved_struct_name
        .split('<')
        .next()
        .unwrap_or(resolved_struct_name)
        .to_string();
    let mapping = match (
        written_type_args(resolved_struct_name),
        gen.structs.get(base_name.as_str()),
    ) {
        (Some(inner_tys), Some(struct_decl)) => generic_mapping(struct_decl, &inner_tys),
        _ => TypeArgs::new(),
    };
    (base_name, mapping)
}

/// A struct field that a member access reads.
#[derive(Clone, Copy)]
struct FieldRead<'a, 'c> {
    /// The field's position in the struct.
    idx: usize,
    /// The field's Vx type, with the struct's type arguments filled in.
    ast_ty: &'a syntax::Type,
    /// The field's MLIR type.
    ty: Type<'c>,
}

/// A field read through a pointer to its struct: the field's address when an address is
/// wanted, otherwise the field loaded from it.
///
/// # Errors
///
/// Returns an error if the type the field is stored as does not parse.
///
/// # Panics
///
/// Panics if MLIR refuses the `getelementptr` or the load.
fn field_through_pointer<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    base_val: Value<'c, 'c>,
    struct_llvm_ty: Type<'c>,
    field: &FieldRead<'_, 'c>,
) -> Lowered<'c> {
    let ptr_ty = gen.ptr_ty;
    let gep_op = OperationBuilder::new("llvm.getelementptr", gen.loc())
        .add_operands(&[base_val])
        .add_attributes(&[
            (
                Identifier::new(gen.context, "rawConstantIndices"),
                DenseI32ArrayAttribute::new(gen.context, &[0, count_i32(field.idx)]).into(),
            ),
            (
                Identifier::new(gen.context, "elem_type"),
                TypeAttribute::new(struct_llvm_ty).into(),
            ),
        ])
        .add_results(&[ptr_ty])
        .build()
        .unwrap();
    let gep_ref = block.append_operation(gep_op);
    let field_ptr = gep_ref.result(0)?.into();

    if gen.is_lvalue_context {
        return Ok((field_ptr, ptr_ty, block));
    }

    let stored_ty = field_storage_ty(gen, field.ast_ty, field.ty)?;
    let load_op = OperationBuilder::new("llvm.load", gen.loc())
        .add_operands(&[field_ptr])
        .add_results(&[stored_ty])
        .build()
        .unwrap();
    let load_ref = block.append_operation(load_op);
    let loaded = load_ref.result(0)?.into();
    let val = tensor_from_field(gen, block, loaded, stored_ty, field.ty)?;
    Ok((val, field.ty, block))
}

/// A field read out of a struct value.
///
/// # Errors
///
/// Returns an error if the type the field is stored as does not parse.
///
/// # Panics
///
/// Panics if MLIR refuses the `extractvalue`.
fn field_of_value<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    base_val: Value<'c, 'c>,
    field: &FieldRead<'_, 'c>,
) -> Lowered<'c> {
    let pos_attr =
        melior::ir::attribute::DenseI64ArrayAttribute::new(gen.context, &[count_i64(field.idx)]);

    let stored_ty = field_storage_ty(gen, field.ast_ty, field.ty)?;
    let ext_op = OperationBuilder::new("llvm.extractvalue", gen.loc())
        .add_operands(&[base_val])
        .add_attributes(&[(Identifier::new(gen.context, "position"), pos_attr.into())])
        .add_results(&[stored_ty])
        .build()
        .unwrap();
    let ext_ref = block.append_operation(ext_op);
    let extracted = ext_ref.result(0)?.into();
    let val = tensor_from_field(gen, block, extracted, stored_ty, field.ty)?;
    Ok((val, field.ty, block))
}

impl<'c> LowerToMelior<'c> for MemberAccessExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let MemberAccessExpr {
            base,
            member,
            struct_name,
            span: _,
        } = self;
        let (base_val, base_ty, block) = gen.generate_expr(base, block)?;
        let base_ty_str = base_ty.to_string();

        let struct_name_opt = struct_name_of_base(struct_name.as_ref(), &base_ty_str);

        let is_ptr = base_ty_str.starts_with("!llvm.ptr");

        if let Some(ref resolved_struct_name) = struct_name_opt {
            let (base_name, mapping) = struct_and_type_args(gen, resolved_struct_name);

            if let Some(struct_decl) = gen.structs.get(base_name.as_str()).cloned() {
                if let Some(field_idx) = struct_decl.fields.iter().position(|(n, _)| n == member) {
                    let sub_ty = struct_decl.fields[field_idx].1.substitute(&mapping);
                    let field_ty = gen.lower_type(&sub_ty)?;

                    let field = FieldRead {
                        idx: field_idx,
                        ast_ty: &sub_ty,
                        ty: field_ty,
                    };
                    if is_ptr {
                        let struct_llvm_ty = crate::codegen::lower::stmt::struct_instance_ty(
                            gen,
                            &base_name.as_str().into(),
                            &struct_decl,
                            &mapping,
                        )?;
                        return field_through_pointer(gen, block, base_val, struct_llvm_ty, &field);
                    }
                    return field_of_value(gen, block, base_val, &field);
                }
            }
        }
        println!(
            "DEBUG PANIC: member={member}, struct_name_opt={struct_name_opt:?}, base_ty_str={base_ty_str}, base_val={base_val:?}"
        );
        if let Some(name) = &struct_name_opt {
            let base_name = name.split('<').next().unwrap_or(name);
            println!(
                "DEBUG PANIC: gen.structs.get({:?}) = {:?}",
                base_name,
                gen.structs.get(base_name)
            );
        }
        panic!("Cannot resolve member access {member}");
    }
}

/// A new buffer for a tensor initializer list of this shape, and the buffer's type.
///
/// # Errors
///
/// Returns an error if the buffer's type does not parse, or if MLIR refuses an operation.
fn alloc_initializer_buffer<'c>(
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    elem_ty_str: &str,
    shape: &[usize],
) -> Result<Typed<'c>, LowerError> {
    let rank = shape.len();
    // A dynamic `memref<?x..xT>` sized by constants -- matches how plain `Tensor<T>([n, m])`
    // buffers are typed, so transfers / slices downstream are unaffected.
    let ty_str = format!("memref<{}{}>", "?x".repeat(rank), elem_ty_str);
    let tensor_ty =
        Type::parse(gen.context, &ty_str).ok_or_else(|| LowerError::ParseType(ty_str.clone()))?;
    let mut size_vals = Vec::with_capacity(rank);
    for &d in shape {
        size_vals.push(index_const(gen, block, count_i64(d))?);
    }
    let alloc_op = OperationBuilder::new("memref.alloc", gen.loc())
        .add_operands(&size_vals)
        .add_attributes(&[(
            Identifier::new(gen.context, "operandSegmentSizes"),
            DenseI32ArrayAttribute::new(gen.context, &[count_i32(size_vals.len()), 0]).into(),
        )])
        .add_results(&[tensor_ty])
        .build()?;
    let buf: Value<'c, 'c> = block.append_operation(alloc_op).result(0)?.into();
    Ok((buf, tensor_ty))
}

/// Lower a tensor initializer list (`Tensor<T>([[..],[..]])`): allocate a buffer of the shape
/// inferred from the nesting and store each constant element in row-major order. The buffer keeps
/// the dynamic `memref<?x..xT>` type (with constant sizes) that the rest of codegen expects.
///
/// # Errors
///
/// Returns an error if a type does not parse, or if an element cannot be lowered.
fn lower_tensor_initializer<'c>(
    gen: &mut MeliorGenerator<'c>,
    elem_ty_str: &str,
    shape: &[usize],
    arr: &ArrayExpr,
    block: melior::ir::BlockRef<'c, 'c>,
) -> Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError> {
    let elem_ty = Type::parse(gen.context, elem_ty_str)
        .ok_or_else(|| LowerError::ParseType(elem_ty_str.to_string()))?;
    let rank = shape.len();
    let (buf, tensor_ty) = alloc_initializer_buffer(gen, block, elem_ty_str, shape)?;

    let values = arr.initializer_values();
    let mut cur = block;
    for (flat, val_expr) in values.iter().enumerate() {
        let (v, vty, nb) = gen.generate_expr(val_expr, cur)?;
        cur = nb;
        let v = gen.coerce_type(&cur, v, vty, elem_ty)?;
        // Row-major flat index -> per-dimension coordinates.
        let mut rem = flat;
        let mut coords = vec![0usize; rank];
        for dim in (0..rank).rev() {
            coords[dim] = rem % shape[dim];
            rem /= shape[dim];
        }
        let mut store = OperationBuilder::new("memref.store", gen.loc()).add_operands(&[v, buf]);
        let idx_vals: Vec<Value<'c, 'c>> = coords
            .iter()
            .map(|&c| index_const(gen, cur, count_i64(c)))
            .collect::<Result<_, _>>()?;
        for iv in &idx_vals {
            store = store.add_operands(std::slice::from_ref(iv));
        }
        cur.append_operation(store.build()?);
    }
    Ok((buf, tensor_ty, cur))
}

/// Static length D of a rank-1 f32 slice value, from its melior type string. Matches a
/// `vector<Dxf32>` or a rank-1 `memref<Dxf32, ...>` (the S1 row view); returns `None` for
/// scalars, dynamic dims, and higher-rank memrefs (which are not slice operands).
fn slice_vec_len(ty_str: &str) -> Option<i64> {
    let inner = ty_str
        .strip_prefix("vector<")
        .or_else(|| ty_str.strip_prefix("memref<"))?;
    // Shape is the text before any `, strided<...>` layout, e.g. `4xf32`; require exactly one
    // `x` (rank 1) and an f32 element (the slice ops emit vector<Dxf32>).
    let shape = inner.split(',').next()?.trim_end_matches('>');
    let parts: Vec<&str> = shape.split('x').collect();
    if parts.len() == 2 && parts[1] == "f32" {
        return parts[0].parse().ok();
    }
    None
}

/// Emit `linalg.fill(0)` over an existing buffer.
///
/// The zero a `Tensor<T, [..]>::new()` arrives holding, and the beta = 0 a GEMM starts from. One
/// emission, so the two cannot come to disagree about what zero is for an element type.
///
/// # Errors
///
/// Returns an error if the element type does not parse, or if MLIR refuses an operation.
pub(super) fn emit_zero_fill<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    dst_val: Value<'c, 'c>,
    el_ty_str: &str,
) -> Result<(), LowerError> {
    let el_ty = Type::parse(gen.context, el_ty_str)
        .ok_or_else(|| LowerError::ParseType(el_ty_str.to_string()))?;
    let is_float = el_ty_str.starts_with('f') || el_ty_str.starts_with("bf");

    let zero_attr: melior::ir::Attribute<'_> = if is_float {
        FloatAttribute::new(gen.context, el_ty, 0.0).into()
    } else {
        IntegerAttribute::new(el_ty, 0).into()
    };
    let zero_op = OperationBuilder::new("arith.constant", gen.loc())
        .add_results(&[el_ty])
        .add_attributes(&[(Identifier::new(gen.context, "value"), zero_attr)])
        .build()?;
    let zero_val: Value<'_, '_> = block.append_operation(zero_op).result(0)?.into();
    emit_value_fill(gen, block, dst_val, zero_val, el_ty_str)
}

/// Emit `linalg.fill(v)` over a whole buffer.
///
/// `::new()` fills with a zero it makes itself; `::fill(v)` fills with a value the caller already
/// lowered. One fill either way, so a tensor gets its contents one way.
///
/// # Errors
///
/// Returns an error if the element type does not parse, or if MLIR refuses an operation.
pub(super) fn emit_value_fill<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    dst_val: Value<'c, 'c>,
    fill_val: Value<'c, 'c>,
    el_ty_str: &str,
) -> Result<(), LowerError> {
    let el_ty = Type::parse(gen.context, el_ty_str)
        .ok_or_else(|| LowerError::ParseType(el_ty_str.to_string()))?;
    let zero_val = fill_val;

    // linalg.fill: one block argument (the value), yielded straight out.
    let region_fill = Region::new();
    let block_fill = melior::ir::Block::new(&[(el_ty, gen.loc()), (el_ty, gen.loc())]);
    // The lookup's own error adds nothing to this message.
    let Ok(fill_arg) = block_fill.argument(0) else {
        return Err(LowerError::from("Missing fill block argument".to_string()));
    };
    let yield_fill = OperationBuilder::new("linalg.yield", gen.loc())
        .add_operands(&[fill_arg.into()])
        .build()?;
    block_fill.append_operation(yield_fill);
    region_fill.append_block(block_fill);

    let fill_op = OperationBuilder::new("linalg.fill", gen.loc())
        .add_operands(&[zero_val, dst_val])
        .add_attributes(&[(
            Identifier::new(gen.context, "operandSegmentSizes"),
            DenseI32ArrayAttribute::new(gen.context, &[1, 1]).into(),
        )])
        .add_regions([region_fill])
        .build()?;
    block.append_operation(fill_op);
    Ok(())
}

/// Emit `linalg.fill(0)` then `linalg.matmul` into an existing buffer.
///
/// The pair is what a GEMM with beta = 0 is, and it is the shape `kernelKindOf`
/// recognises when deciding whether an outlined region can be handed to a
/// vendor kernel (#325). `dst` is written in place, so no result is produced
/// and nothing is allocated.
///
/// # Errors
///
/// Returns an error if the element type does not parse, or if MLIR refuses an operation.
#[allow(
    clippy::too_many_arguments,
    clippy::trivially_copy_pass_by_ref,
    reason = "a caller is in another file; it passes the block by reference and each tensor on its own"
)]
pub(super) fn emit_matmul_into<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: &melior::ir::BlockRef<'c, 'c>,
    lhs_val: Value<'c, 'c>,
    rhs_val: Value<'c, 'c>,
    dst_val: Value<'c, 'c>,
    el_ty_str: &str,
) -> Result<(), LowerError> {
    let el_ty = Type::parse(gen.context, el_ty_str)
        .ok_or_else(|| LowerError::ParseType(el_ty_str.to_string()))?;
    let is_float = el_ty_str.starts_with('f');

    emit_zero_fill(gen, *block, dst_val, el_ty_str)?;

    // linalg.matmul: acc = acc + lhs * rhs, per element.
    let region_matmul = Region::new();
    let block_matmul =
        melior::ir::Block::new(&[(el_ty, gen.loc()), (el_ty, gen.loc()), (el_ty, gen.loc())]);
    let arg = |i: usize| -> Result<Value<'c, 'c>, LowerError> {
        // The lookup's own error adds nothing to this message.
        let Ok(arg) = block_matmul.argument(i) else {
            return Err(LowerError::from(format!(
                "Missing matmul block argument {i}"
            )));
        };
        Ok(arg.into())
    };
    let (mul_name, add_name) = if is_float {
        ("arith.mulf", "arith.addf")
    } else {
        ("arith.muli", "arith.addi")
    };
    let mul_op = OperationBuilder::new(mul_name, gen.loc())
        .add_operands(&[arg(0)?, arg(1)?])
        .add_results(&[el_ty])
        .build()?;
    let mul_val: Value<'_, '_> = block_matmul.append_operation(mul_op).result(0)?.into();
    let add_op = OperationBuilder::new(add_name, gen.loc())
        .add_operands(&[arg(2)?, mul_val])
        .add_results(&[el_ty])
        .build()?;
    let add_val: Value<'_, '_> = block_matmul.append_operation(add_op).result(0)?.into();
    let yield_matmul = OperationBuilder::new("linalg.yield", gen.loc())
        .add_operands(&[add_val])
        .build()?;
    block_matmul.append_operation(yield_matmul);
    region_matmul.append_block(block_matmul);

    let matmul_op = OperationBuilder::new("linalg.matmul", gen.loc())
        .add_operands(&[lhs_val, rhs_val, dst_val])
        .add_attributes(&[(
            Identifier::new(gen.context, "operandSegmentSizes"),
            DenseI32ArrayAttribute::new(gen.context, &[2, 1]).into(),
        )])
        .add_regions([region_matmul])
        .build()?;
    block.append_operation(matmul_op);
    Ok(())
}

/// Lower `matmul_into(&mut dst, &a, &b)`: a matmul that fills a buffer it was
/// given rather than allocating one.
///
/// `dst = &a @ &b` cannot express this. The assignment rebinds `dst` to a fresh
/// allocation, so a view over a caller's buffer is left untouched -- which is
/// exactly what a model's `matmul(xout, x, w, ...)` needs to do. It also
/// allocates a result per call, and a decoder does several matmuls per layer per
/// token.
///
/// This is also the only producer of `outkind=buffer`, which the dispatch
/// decoder and both backends have implemented and never seen.
///
/// # Errors
///
/// Returns an error if there are not three arguments, if one cannot be lowered, or if the
/// destination is not a tensor.
fn lower_matmul_into<'c>(
    gen: &mut MeliorGenerator<'c>,
    args: &[Expr],
    block: melior::ir::BlockRef<'c, 'c>,
) -> Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError> {
    if args.len() != 3 {
        return Err(LowerError::from(
            "matmul_into expects (dst, a, b)".to_string(),
        ));
    }

    let (dst_val, mut dst_ty, block) = gen.generate_expr(&args[0], block)?;
    let (lhs_val, mut lhs_ty, block) = gen.generate_expr(&args[1], block)?;
    let (rhs_val, mut rhs_ty, block) = gen.generate_expr(&args[2], block)?;

    // Each operand may be a borrow, which is the address of the slot holding
    // the descriptor rather than the descriptor itself.
    let dst_val = load_tensor_slot(gen, &block, dst_val, &mut dst_ty)?;
    let lhs_val = load_tensor_slot(gen, &block, lhs_val, &mut lhs_ty)?;
    let rhs_val = load_tensor_slot(gen, &block, rhs_val, &mut rhs_ty)?;

    let dst_ty_str = dst_ty.to_string();
    let el_ty_str = dst_ty_str
        .rsplit('x')
        .next()
        .and_then(|s| s.strip_suffix('>'))
        .ok_or_else(|| LowerError::from(format!("matmul_into: not a memref: {dst_ty_str}")))?
        .to_string();

    emit_matmul_into(gen, &block, lhs_val, rhs_val, dst_val, &el_ty_str)?;

    // Writes in place, so there is no value to hand back. The unit stand-in
    // matches what other void builtins return here.
    let zero_op = OperationBuilder::new("arith.constant", gen.loc())
        .add_results(&[gen.i32_ty])
        .add_attributes(&[(
            Identifier::new(gen.context, "value"),
            IntegerAttribute::new(gen.i32_ty, 0).into(),
        )])
        .build()?;
    let unit: Value<'_, '_> = block.append_operation(zero_op).result(0)?.into();
    Ok((unit, gen.i32_ty, block))
}

/// The descriptor of a rank-2 tensor over the memory at `ptr_val` with these extents:
/// `{allocated, aligned, offset, sizes, strides}`.
///
/// # Errors
///
/// Returns an error if a type does not parse, or if MLIR refuses an operation.
fn view_2d_descriptor<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    ptr_val: Value<'c, 'c>,
    rows_i64: Value<'c, 'c>,
    cols_i64: Value<'c, 'c>,
) -> Result<Value<'c, 'c>, LowerError> {
    let i64_ty =
        Type::parse(gen.context, "i64").ok_or_else(|| LowerError::ParseType("i64".to_string()))?;
    let const_i64 = |gen: &mut MeliorGenerator<'c>,
                     n: i64,
                     b: &melior::ir::BlockRef<'c, 'c>|
     -> Result<Value<'c, 'c>, LowerError> {
        let op = OperationBuilder::new("arith.constant", gen.loc())
            .add_results(&[i64_ty])
            .add_attributes(&[(
                Identifier::new(gen.context, "value"),
                IntegerAttribute::new(i64_ty, n).into(),
            )])
            .build()?;
        Ok(b.append_operation(op).result(0)?.into())
    };
    let zero = const_i64(gen, 0, &block)?;
    let one = const_i64(gen, 1, &block)?;

    let desc_ty_str = "!llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>".to_string();
    let desc_ty = Type::parse(gen.context, &desc_ty_str)
        .ok_or_else(|| LowerError::ParseType(desc_ty_str.clone()))?;

    let undef = OperationBuilder::new("llvm.mlir.undef", gen.loc())
        .add_results(&[desc_ty])
        .build()?;
    let mut desc: Value<'_, '_> = block.append_operation(undef).result(0)?.into();

    // Fields in order: allocated, aligned, offset, sizes[0..1], strides[0..1].
    // Row-major, so the row stride is the column count and the column stride 1.
    let fields: [(Value<'_, '_>, &[i64]); 7] = [
        (ptr_val, &[0]),
        (ptr_val, &[1]),
        (zero, &[2]),
        (rows_i64, &[3, 0]),
        (cols_i64, &[3, 1]),
        (cols_i64, &[4, 0]),
        (one, &[4, 1]),
    ];
    for (val, position) in fields {
        let pos_attr = melior::ir::attribute::DenseI64ArrayAttribute::new(gen.context, position);
        let insert = OperationBuilder::new("llvm.insertvalue", gen.loc())
            .add_operands(&[desc, val])
            .add_attributes(&[(Identifier::new(gen.context, "position"), pos_attr.into())])
            .add_results(&[desc_ty])
            .build()?;
        desc = block.append_operation(insert).result(0)?.into();
    }
    Ok(desc)
}

/// Lower `tensor_view_2d(ptr, rows, cols)` to a rank-2 memref descriptor built
/// over `ptr`, with no allocation and no copy (#336).
///
/// The descriptor MLIR passes for a ranked memref is
/// `{allocated, aligned, offset, sizes[rank], strides[rank]}`, so a view is
/// that struct with both pointers aimed at the caller's storage, a zero offset,
/// the given extents, and row-major strides. `unrealized_conversion_cast` is
/// how a built descriptor re-enters memref-typed IR; it is the same
/// materialization the memref-to-LLVM conversion uses.
///
/// Nothing here checks that the extents describe the memory truthfully -- that
/// is the caller's claim, which is why the surface form requires `unsafe`.
///
/// # Errors
///
/// Returns an error if there are not three arguments, if one cannot be lowered, or if a type
/// does not parse.
fn lower_tensor_view_2d<'c>(
    gen: &mut MeliorGenerator<'c>,
    args: &[Expr],
    type_args: Option<&[syntax::Type]>,
    block: melior::ir::BlockRef<'c, 'c>,
) -> Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError> {
    if args.len() != 3 {
        return Err(LowerError::from(
            "tensor_view_2d expects (ptr, rows, cols)".to_string(),
        ));
    }

    let elem_str = match type_args {
        Some(tys) if !tys.is_empty() => gen.lower_type_str(&tys[0])?,
        _ => "f32".to_string(),
    };

    let (ptr_val, _ptr_ty, block) = gen.generate_expr(&args[0], block)?;
    let (rows_val, rows_ty, block) = gen.generate_expr(&args[1], block)?;
    let (cols_val, cols_ty, block) = gen.generate_expr(&args[2], block)?;

    let i64_ty =
        Type::parse(gen.context, "i64").ok_or_else(|| LowerError::ParseType("i64".to_string()))?;
    let to_i64 = |gen: &mut MeliorGenerator<'c>,
                  v: Value<'c, 'c>,
                  t: Type<'c>,
                  b: &melior::ir::BlockRef<'c, 'c>|
     -> Result<Value<'c, 'c>, LowerError> {
        if t == i64_ty {
            return Ok(v);
        }
        let ext = OperationBuilder::new("arith.extsi", gen.loc())
            .add_operands(&[v])
            .add_results(&[i64_ty])
            .build()?;
        Ok(b.append_operation(ext).result(0)?.into())
    };
    let rows_i64 = to_i64(gen, rows_val, rows_ty, &block)?;
    let cols_i64 = to_i64(gen, cols_val, cols_ty, &block)?;

    let desc = view_2d_descriptor(gen, block, ptr_val, rows_i64, cols_i64)?;

    let memref_ty_str = format!("memref<?x?x{elem_str}>");
    let memref_ty = Type::parse(gen.context, &memref_ty_str)
        .ok_or_else(|| LowerError::ParseType(memref_ty_str.clone()))?;
    let cast = OperationBuilder::new("builtin.unrealized_conversion_cast", gen.loc())
        .add_operands(&[desc])
        .add_results(&[memref_ty])
        .build()?;
    let view: Value<'_, '_> = block.append_operation(cast).result(0)?.into();

    Ok((view, memref_ty, block))
}

/// Load a tensor descriptor out of the slot holding it, if that is what the
/// operand is.
///
/// `&a` lowers to the address of `a`, which for a tensor is the rank-0 memref
/// its descriptor lives in: `memref<memref<?x?xf32>>`. A matmul needs the
/// descriptor, so it is loaded here and the operand's type updated to match.
/// Anything else passes through untouched, so the by-value spelling is
/// unaffected.
///
/// # Errors
///
/// Returns an error if the descriptor's type does not parse, or if MLIR refuses the load.
#[allow(
    clippy::trivially_copy_pass_by_ref,
    reason = "callers are in other files; they pass the block by reference"
)]
pub(super) fn load_tensor_slot<'c>(
    gen: &MeliorGenerator<'c>,
    block: &melior::ir::BlockRef<'c, 'c>,
    val: Value<'c, 'c>,
    ty: &mut Type<'c>,
) -> Result<Value<'c, 'c>, LowerError> {
    let ty_str = ty.to_string();
    if !ty_str.starts_with("memref<memref<") {
        return Ok(val);
    }

    // Strip one `memref<` ... `>` layer to name the descriptor's own type.
    let inner_str = &ty_str["memref<".len()..ty_str.len() - 1];
    let inner_ty = Type::parse(gen.context, inner_str)
        .ok_or_else(|| LowerError::ParseType(inner_str.to_string()))?;

    let load_op = OperationBuilder::new("memref.load", gen.loc())
        .add_operands(&[val])
        .add_results(&[inner_ty])
        .build()?;
    let loaded: Value<'_, '_> = block.append_operation(load_op).result(0)?.into();
    *ty = inner_ty;
    Ok(loaded)
}

/// A reference to a tensor, `memref<memref<A>>`, passed where `memref<memref<B>>` is expected.
///
/// A `memref.cast` cannot change what a memref holds, so the descriptor is loaded, given type
/// `B`, and stored in a new slot. It still points at the same data, so writes through a `&mut`
/// reach the tensor. A row (`memref<4xf32, strided<[1], offset: ?>>`) passed as a plain memref
/// has its data pointer moved to the row: a plain memref's offset is 0 by its type, so the code
/// using it never reads the offset, and a cast would hand it row 0.
///
/// # Errors
///
/// Returns an error if a type does not parse, if a view that is not contiguous is passed as a
/// plain memref, or if MLIR refuses an operation.
fn retype_tensor_reference<'c>(
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    val: Value<'c, 'c>,
    from: Type<'c>,
    to: Type<'c>,
) -> Result<Value<'c, 'c>, LowerError> {
    let mut src_ty = from;
    let desc = load_tensor_slot(gen, &block, val, &mut src_ty)?;
    let to_str = to.to_string();
    let dst_str = &to_str["memref<".len()..to_str.len() - 1];
    let dst_ty =
        Type::parse(gen.context, dst_str).ok_or_else(|| LowerError::ParseType(dst_str.into()))?;
    let src_str = src_ty.to_string();
    let mut desc = desc;
    let mut desc_ty = src_ty;
    if src_str.contains("strided<") && !dst_str.contains("strided<") {
        let plain = contiguous_view_as_plain(&src_str).ok_or_else(|| {
            LowerError::from(format!("cannot pass a {src_str} view as a {dst_str}"))
        })?;
        desc_ty = Type::parse(gen.context, &plain).ok_or(LowerError::ParseType(plain))?;
        desc = move_view_pointer(gen, block, desc, src_ty, desc_ty)?;
    }
    if desc_ty != dst_ty {
        let cast = OperationBuilder::new("memref.cast", gen.loc())
            .add_operands(&[desc])
            .add_results(&[dst_ty])
            .build()?;
        desc = block.append_operation(cast).result(0)?.into();
    }
    let slot: Value<'_, '_> = block
        .append_operation(
            OperationBuilder::new("memref.alloca", gen.loc())
                .add_attributes(&[(
                    Identifier::new(gen.context, "operandSegmentSizes"),
                    DenseI32ArrayAttribute::new(gen.context, &[0, 0]).into(),
                )])
                .add_results(&[to])
                .build()?,
        )
        .result(0)?
        .into();
    block.append_operation(
        OperationBuilder::new("memref.store", gen.loc())
            .add_operands(&[desc, slot])
            .build()?,
    );
    Ok(slot)
}

/// The plain memref type with the same shape as a contiguous strided view, or `None` when the
/// view's strides are not row-major or it is in a GPU memory space. `memref<4xf32, strided<[1],
/// offset: ?>>` gives `memref<4xf32>`. A `?` stride cannot be checked here; only indexing a row
/// out of a tensor makes one, and a row of a tensor is contiguous.
fn contiguous_view_as_plain(memty: &str) -> Option<String> {
    if !memty.ends_with("offset: ?>>") && !memty.ends_with("offset: 0>>") {
        return None;
    }
    let (shape, layout) = memty.strip_prefix("memref<")?.split_once(", strided<[")?;
    let dims: Vec<&str> = shape.split('x').collect();
    let dims = &dims[..dims.len() - 1];
    let strides: Vec<&str> = layout
        .split(']')
        .next()?
        .split(',')
        .map(str::trim)
        .collect();
    if dims.is_empty() || strides.len() != dims.len() {
        return None;
    }
    let mut expect = Some(1i64);
    for k in (0..dims.len()).rev() {
        if let (Some(e), Ok(s)) = (expect, strides[k].parse::<i64>()) {
            if s != e {
                return None;
            }
        }
        expect = expect.zip(dims[k].parse::<i64>().ok()).map(|(e, d)| e * d);
    }
    Some(format!("memref<{shape}>"))
}

/// The number of dimensions of a view's memref type, and its element type as text.
///
/// # Errors
///
/// Returns an error if `view_str` is not a memref type.
fn view_rank_and_element(view_str: &str) -> Result<(usize, &str), LowerError> {
    let shape = view_str
        .strip_prefix("memref<")
        .and_then(|s| s.split(',').next())
        .ok_or_else(|| LowerError::ParseType(view_str.to_string()))?;
    let (dims, elem) = shape.rsplit_once('x').unwrap_or(("", shape));
    let rank = if dims.is_empty() {
        0
    } else {
        dims.split('x').count()
    };
    Ok((rank, elem))
}

/// A view's descriptor with its data pointer moved forward by its offset and the offset set to
/// 0, retyped as `plain`, the same shape with no layout.
///
/// # Errors
///
/// Returns an error if `view_ty` is not a memref type, if a type does not parse, or if MLIR
/// refuses an operation.
fn move_view_pointer<'c>(
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    view: Value<'c, 'c>,
    view_ty: Type<'c>,
    plain: Type<'c>,
) -> Result<Value<'c, 'c>, LowerError> {
    let view_str = view_ty.to_string();
    let (rank, elem) = view_rank_and_element(&view_str)?;
    let desc_str = crate::codegen::flat::memref_descriptor_ty(rank);
    let desc_ty = Type::parse(gen.context, &desc_str).ok_or(LowerError::ParseType(desc_str))?;
    let elem_ty =
        Type::parse(gen.context, elem).ok_or_else(|| LowerError::ParseType(elem.into()))?;
    let op = |b: OperationBuilder<'c>| -> Result<Value<'c, 'c>, LowerError> {
        Ok(block.append_operation(b.build()?).result(0)?.into())
    };
    let position = |n: i64| {
        (
            Identifier::new(gen.context, "position"),
            melior::ir::attribute::DenseI64ArrayAttribute::new(gen.context, &[n]).into(),
        )
    };
    let desc = op(
        OperationBuilder::new("builtin.unrealized_conversion_cast", gen.loc())
            .add_operands(&[view])
            .add_results(&[desc_ty]),
    )?;
    let aligned = op(OperationBuilder::new("llvm.extractvalue", gen.loc())
        .add_operands(&[desc])
        .add_attributes(&[position(1)])
        .add_results(&[gen.ptr_ty]))?;
    let offset = op(OperationBuilder::new("llvm.extractvalue", gen.loc())
        .add_operands(&[desc])
        .add_attributes(&[position(2)])
        .add_results(&[gen.i64_ty]))?;
    let moved = op(OperationBuilder::new("llvm.getelementptr", gen.loc())
        .add_operands(&[aligned, offset])
        .add_attributes(&[
            (
                Identifier::new(gen.context, "rawConstantIndices"),
                DenseI32ArrayAttribute::new(gen.context, &[i32::MIN]).into(),
            ),
            (
                Identifier::new(gen.context, "elem_type"),
                TypeAttribute::new(elem_ty).into(),
            ),
        ])
        .add_results(&[gen.ptr_ty]))?;
    let zero = op(OperationBuilder::new("llvm.mlir.constant", gen.loc())
        .add_attributes(&[(
            Identifier::new(gen.context, "value"),
            IntegerAttribute::new(gen.i64_ty, 0).into(),
        )])
        .add_results(&[gen.i64_ty]))?;
    let desc = op(OperationBuilder::new("llvm.insertvalue", gen.loc())
        .add_operands(&[desc, moved])
        .add_attributes(&[position(1)])
        .add_results(&[desc_ty]))?;
    let desc = op(OperationBuilder::new("llvm.insertvalue", gen.loc())
        .add_operands(&[desc, zero])
        .add_attributes(&[position(2)])
        .add_results(&[desc_ty]))?;
    op(
        OperationBuilder::new("builtin.unrealized_conversion_cast", gen.loc())
            .add_operands(&[desc])
            .add_results(&[plain]),
    )
}

/// Whether an operand should drive the slice-elementwise (S3) vector path: a `vector<...>`
/// value or a strided slice view (`memref<..., strided<...>>`, as produced by S1). A plain
/// contiguous `memref<Nxf32>` (a whole 1-D tensor) is deliberately excluded -- those keep the
/// loop-based lowering that the optimization-pass tests exercise (scf-to-cf, loop unroll).
fn is_slice_operand(ty_str: &str) -> bool {
    ty_str.starts_with("vector<") || (ty_str.starts_with("memref<") && ty_str.contains("strided"))
}

/// Coerce a slice-elementwise operand to `vector<Dxf32>`: a vector passes through, a rank-1
/// memref is `vector.load`ed, and a scalar is `vector.broadcast`ed to the slice width.
///
/// # Errors
///
/// Returns an error if the vector type does not parse, or if MLIR refuses an operation.
fn to_vector<'c>(
    gen: &mut MeliorGenerator<'c>,
    val: Value<'c, 'c>,
    ty_str: &str,
    d: i64,
    block: melior::ir::BlockRef<'c, 'c>,
) -> Result<Value<'c, 'c>, LowerError> {
    let vec_ty = Type::parse(gen.context, &format!("vector<{d}xf32>"))
        .ok_or_else(|| LowerError::ParseType(format!("vector<{d}xf32>")))?;
    if ty_str.starts_with("vector<") {
        return Ok(val);
    }
    if ty_str.starts_with("memref<") {
        let index_ty = Type::index(gen.context);
        let c0_op = OperationBuilder::new("arith.constant", gen.loc())
            .add_attributes(&[(
                Identifier::new(gen.context, "value"),
                IntegerAttribute::new(index_ty, 0).into(),
            )])
            .add_results(&[index_ty])
            .build()?;
        let c0: Value<'_, '_> = block.append_operation(c0_op).result(0)?.into();
        let load_op = OperationBuilder::new("vector.load", gen.loc())
            .add_operands(&[val, c0])
            .add_results(&[vec_ty])
            .build()?;
        return Ok(block.append_operation(load_op).result(0)?.into());
    }
    // Scalar: broadcast to the slice width.
    let bcast_op = OperationBuilder::new("vector.broadcast", gen.loc())
        .add_operands(&[val])
        .add_results(&[vec_ty])
        .build()?;
    Ok(block.append_operation(bcast_op).result(0)?.into())
}

/// The element type and the shape arguments of a tensor constructor call.
///
/// `Tensor<T, [d0, d1]>::uninit()` writes its element and shape in the type; the older
/// `Tensor<T>([d0, d1])` splits them between the generic argument and the call arguments.
/// The first is turned into the second's form -- one array argument of extents -- so that
/// there is a single lowering rather than two that must agree.
///
/// # Panics
///
/// Panics if the call has no type argument.
fn tensor_element_and_shape(
    written: Option<&syntax::Type>,
    args: &[Expr],
) -> (syntax::Type, Vec<Expr>) {
    match written {
        // A shape with a run-time extent is not in the type, so it stays where it was
        // written; a static one is rebuilt from the type as the one array argument.
        Some(syntax::Type::Tensor(el, dims, _))
            if dims.iter().any(|d| matches!(d, syntax::Dim::Dyn)) =>
        {
            (syntax::Type::Scalar(el.clone()), args.to_vec())
        }
        Some(syntax::Type::Tensor(el, dims, _)) => (
            syntax::Type::Scalar(el.clone()),
            vec![Expr::Array(syntax::ArrayExpr::new(
                dims.iter().filter_map(|d| d.as_static().cloned()).collect(),
                syntax::Span::default(),
            ))],
        ),
        Some(scalar) => (scalar.clone(), args.to_vec()),
        None => panic!("Tensor initialization requires an explicit generic type argument"),
    }
}

/// The extents written in a tensor constructor call, each as an `index` value: the elements
/// of a single array argument, or else each argument. Also gives back the block the last
/// extent ended in.
///
/// # Errors
///
/// Returns an error if an extent cannot be lowered.
///
/// # Panics
///
/// Panics if MLIR refuses the cast of an extent to `index`.
fn lower_extents<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    args: &[Expr],
) -> Result<(Vec<Value<'c, 'c>>, melior::ir::BlockRef<'c, 'c>), LowerError> {
    let mut dynamic_sizes = Vec::new();
    let mut current_b = block;
    if let [Expr::Array(arr)] = args {
        for el in &arr.elements {
            let (mut val, ty, new_b) = gen.generate_expr(el, current_b)?;
            current_b = new_b;
            if ty.to_string() != "index" {
                let cast_op = OperationBuilder::new("arith.index_cast", gen.loc())
                    .add_operands(&[val])
                    .add_results(&[Type::index(gen.context)])
                    .build()
                    .unwrap();
                val = current_b.append_operation(cast_op).result(0)?.into();
            }
            dynamic_sizes.push(val);
        }
    } else if !args.is_empty() {
        for el in args {
            let (mut val, ty, new_b) = gen.generate_expr(el, current_b)?;
            current_b = new_b;
            if ty.to_string() != "index" {
                let cast_op = OperationBuilder::new("arith.index_cast", gen.loc())
                    .add_operands(&[val])
                    .add_results(&[Type::index(gen.context)])
                    .build()
                    .unwrap();
                val = current_b.append_operation(cast_op).result(0)?.into();
            }
            dynamic_sizes.push(val);
        }
    }
    Ok((dynamic_sizes, current_b))
}

/// The attribute that names the device an operation works on.
fn target_topology_attr<'c>(gen: &MeliorGenerator<'c>, t: i32) -> (Identifier<'c>, Attribute<'c>) {
    (
        Identifier::new(gen.context, "target_topology"),
        IntegerAttribute::new(gen.i32_ty, i64::from(t)).into(),
    )
}

/// The device whose memory a tensor's written type places it in. `None` for the host.
fn placed_device(gen: &MeliorGenerator<'_>, written: Option<&syntax::Type>) -> Option<i32> {
    match written {
        Some(syntax::Type::Tensor(_, _, Some(p))) => gen.placed_topology(p).filter(|t| *t != 0),
        _ => None,
    }
}

/// The operation that allocates a tensor's buffer: in the memory of the device `on_device`
/// when there is one, otherwise on the host.
///
/// # Errors
///
/// Returns an error if MLIR refuses the operation.
fn tensor_alloc_op<'c>(
    gen: &MeliorGenerator<'c>,
    on_device: Option<i32>,
    dynamic_sizes: &[Value<'c, 'c>],
    tensor_ty: Type<'c>,
) -> Result<melior::ir::Operation<'c>, LowerError> {
    Ok(match on_device {
        Some(t) => OperationBuilder::new("vx.alloc", gen.loc())
            .add_operands(dynamic_sizes)
            .add_attributes(&[target_topology_attr(gen, t)])
            .add_results(&[tensor_ty])
            .build()?,
        None => OperationBuilder::new("memref.alloc", gen.loc())
            .add_operands(dynamic_sizes)
            .add_attributes(&[(
                Identifier::new(gen.context, "operandSegmentSizes"),
                DenseI32ArrayAttribute::new(gen.context, &[count_i32(dynamic_sizes.len()), 0])
                    .into(),
            )])
            .add_results(&[tensor_ty])
            .build()?,
    })
}

/// `Tensor<T>(..)`, `Tensor::new()`, `Tensor::uninit()` and `Tensor::fill(v)`: a new tensor,
/// zeroed, left as it is, or filled with `v`.
///
/// # Errors
///
/// Returns an error if the element type, an extent or the fill value cannot be lowered, or if
/// MLIR refuses an operation.
fn lower_tensor_constructor<'c>(
    this: &FunctionCallExpr,
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
) -> Lowered<'c> {
    let FunctionCallExpr {
        name,
        args,
        type_args,
        ..
    } = this;
    // `Tensor<T, [d0, d1]>::fill(v)` takes its shape from the type like `::new()` does, so
    // the call argument is the fill value rather than an extent. Captured before `args` is
    // rebound to the normalised shape below.
    let fill_expr: Option<Expr> = if name.as_ref() == "Tensor::fill" {
        args.first().cloned()
    } else {
        None
    };
    // `Tensor<T, [d0, d1]>::uninit()` writes its element and shape in the type; the older
    // `Tensor<T>([d0, d1])` splits them between the generic argument and the call
    // arguments. Normalising the first into the second's shape -- one array argument of
    // extents -- keeps a single lowering rather than two that must agree.
    let written = type_args.as_ref().and_then(|t| t.first());
    let (elem, shape_args) = tensor_element_and_shape(written, args);
    let mlir_ty_str = gen.lower_type_str(&elem)?;
    let args: &[Expr] = &shape_args;
    // Initializer list: `Tensor<T>([[..],[..]])` allocates a shaped buffer and stores the
    // constant values in place, so no fill loop is needed.
    if let Some(Expr::Array(arr)) = args.first() {
        if let Some(shape) = arr.initializer_shape() {
            return lower_tensor_initializer(gen, &mlir_ty_str, &shape, arr, block);
        }
    }
    let (dynamic_sizes, current_b) = lower_extents(gen, block, args)?;
    // Number of dynamic dimensions; the type below uses one `?` per dim,
    // so this must equal the number of size operands collected into
    // `dynamic_sizes`. With no shape args (e.g. `Tensor<f32>()`) it stays
    // 0, yielding a valid rank-0 `memref<f32>` rather than a `?x?` memref
    // with no sizes (invalid IR). See GitHub #147.
    let dims_count = dynamic_sizes.len();

    let mut shape_str = String::new();
    for _ in 0..dims_count {
        shape_str.push_str("?x");
    }
    let tensor_ty_str = format!("memref<{shape_str}{mlir_ty_str}>");
    let tensor_ty = Type::parse(gen.context, &tensor_ty_str).ok_or_else(|| {
        crate::codegen::lower::LowerError::ParseType("Type::parse failed".to_string())
    })?;

    // A tensor placed in a device's memory. `uninit()` is allocated there; one that is
    // filled is filled on the host and copied there, since the host cannot write it.
    let device = placed_device(gen, written);
    let uninit = name.as_ref().ends_with("::uninit");
    let alloc_op = tensor_alloc_op(gen, device.filter(|_| uninit), &dynamic_sizes, tensor_ty)?;
    let copy_to_device = device.filter(|_| !name.as_ref().ends_with("::uninit"));
    // A size can end the block it started in (`[if c { 3 } else { 2 }]` lands in the
    // `if`'s merge block), so what follows goes where the sizes left off.
    let block = current_b;
    let alloc_ref = block.append_operation(alloc_op);
    let alloc_val: Value<'c, 'c> = alloc_ref.result(0)?.into();
    // `::new()` zeroes what `::uninit()` leaves as it was found. The same fill a matmul
    // emits before it accumulates, so there is one answer to what zero is per element.
    if name.as_ref().ends_with("::new") {
        emit_zero_fill(gen, block, alloc_val, &mlir_ty_str)?;
    }
    let block = if let Some(fe) = &fill_expr {
        let (fill_val, _fill_ty, block) = gen.generate_expr(fe, block)?;
        emit_value_fill(gen, block, alloc_val, fill_val, &mlir_ty_str)?;
        block
    } else {
        block
    };
    if let Some(t) = copy_to_device {
        let transfer = OperationBuilder::new("vx.transfer", gen.loc())
            .add_operands(&[alloc_val])
            .add_attributes(&[target_topology_attr(gen, t)])
            .add_results(&[tensor_ty])
            .build()?;
        let placed: Value<'c, 'c> = block.append_operation(transfer).result(0)?.into();
        return Ok((placed, tensor_ty, block));
    }
    Ok((alloc_val, tensor_ty, block))
}

/// Row-major (contiguous) strides for a shape.
fn row_major_strides(dims: &[i64]) -> Vec<i64> {
    let mut s = vec![1i64; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * dims[i + 1];
    }
    s
}

/// The shape and the element type of a memref as its type writes them: `2x3xf32`.
fn dims_text(dims: &[i64], el_ty_str: &str) -> String {
    let mut s = String::new();
    for d in dims {
        s.push_str(&d.to_string());
        s.push('x');
    }
    s.push_str(el_ty_str);
    s
}

/// The numbers with a comma between each two.
fn i64_list(v: &[i64]) -> String {
    v.iter()
        .map(std::string::ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

/// A view of a tensor whose sizes and strides are all known while compiling.
#[derive(Clone, Copy)]
struct StaticView<'a, 'c> {
    /// The size of each dimension.
    sizes: &'a [i64],
    /// The stride of each dimension, in elements.
    strides: &'a [i64],
    /// The view's memref type.
    ty: Type<'c>,
}

/// `memref.reinterpret_cast` of `src` at offset 0 as `view`.
///
/// # Panics
///
/// Panics if MLIR refuses the operation.
fn reinterpret_static<'c>(
    gen: &MeliorGenerator<'c>,
    blk: melior::ir::BlockRef<'c, 'c>,
    src: Value<'c, 'c>,
    view: StaticView<'_, 'c>,
) -> Value<'c, 'c> {
    let StaticView {
        sizes,
        strides,
        ty: result_ty,
    } = view;
    let op = OperationBuilder::new("memref.reinterpret_cast", gen.loc())
        .add_operands(&[src])
        .add_attributes(&[
            (
                Identifier::new(gen.context, "operandSegmentSizes"),
                DenseI32ArrayAttribute::new(gen.context, &[1, 0, 0, 0]).into(),
            ),
            (
                Identifier::new(gen.context, "static_offsets"),
                DenseI64ArrayAttribute::new(gen.context, &[0]).into(),
            ),
            (
                Identifier::new(gen.context, "static_sizes"),
                DenseI64ArrayAttribute::new(gen.context, sizes).into(),
            ),
            (
                Identifier::new(gen.context, "static_strides"),
                DenseI64ArrayAttribute::new(gen.context, strides).into(),
            ),
        ])
        .add_results(&[result_ty])
        .build()
        .expect("Failed to build reinterpret_cast");
    blk.append_operation(op)
        .result(0)
        .expect("Failed to get result")
        .into()
}

/// A tensor argument, with the static extents its Vx type knows put back on its memref type.
///
/// A `let mut` tensor is a slot holding the tensor, and the slot keeps only `?` extents.
///
/// # Errors
///
/// Returns an error if the argument cannot be lowered, or if MLIR refuses an operation.
///
/// # Panics
///
/// Panics if the argument's element type cannot be read from its memref type.
fn tensor_with_static_extents<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    arg: &Expr,
) -> Lowered<'c> {
    let (arg_val, mut expr_ty, block) = gen.generate_expr(arg, block)?;
    // A `let mut` tensor is a slot holding the tensor, and the slot keeps only `?`
    // extents. The shape below is read off the memref type, so put the static extents
    // the Vx type still knows back on it.
    let mut arg_val = load_tensor_slot(gen, &block, arg_val, &mut expr_ty)?;
    if expr_ty.to_string().contains('?') {
        if let Some(syntax::Type::Tensor(_, dims, _)) = gen.infer_ast_type(arg) {
            let extents: Option<Vec<&str>> = dims.iter().map(|d| d.literal()).collect();
            let el =
                extract_mlir_element_type(&expr_ty.to_string()).unwrap_or_else(|e| panic!("{}", e));
            if let Some(extents) = extents {
                let static_str = format!("memref<{}x{}>", extents.join("x"), el);
                let static_ty = Type::parse(gen.context, &static_str)
                    .ok_or_else(|| LowerError::ParseType(static_str.clone()))?;
                let cast = OperationBuilder::new("memref.cast", gen.loc())
                    .add_operands(&[arg_val])
                    .add_results(&[static_ty])
                    .build()?;
                arg_val = block.append_operation(cast).result(0)?.into();
                expr_ty = static_ty;
            }
        }
    }
    Ok((arg_val, expr_ty, block))
}

/// The integers written in an array literal; empty for anything else.
fn integer_array(expr: &Expr) -> Vec<i64> {
    match expr {
        Expr::Array(arr) => arr
            .elements
            .iter()
            .filter_map(|el| match el {
                Expr::Number(num) => num.value.parse::<i64>().ok(),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

/// A tensor whose sizes are all known while compiling.
#[derive(Clone, Copy)]
struct StaticTensor<'a, 'c> {
    /// The tensor's memref.
    val: Value<'c, 'c>,
    /// The size of each dimension.
    dims: &'a [i64],
    /// The element type, as MLIR writes it.
    el_ty_str: &'a str,
}

/// `transpose`: a view of the source with its strides in the order `perm` gives, copied into
/// a new contiguous buffer so that later operations see row-major data.
///
/// # Errors
///
/// Returns an error if MLIR refuses an operation.
///
/// # Panics
///
/// Panics if an axis in `perm` is negative or past the source's rank, or if a type built here
/// does not parse.
fn lower_transpose<'c>(
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    src: StaticTensor<'_, 'c>,
    perm: &[i64],
) -> Lowered<'c> {
    let axis = |p: i64| usize::try_from(p).expect("a transpose axis is not negative");
    let src_strides = row_major_strides(src.dims);
    let view_dims: Vec<i64> = perm.iter().map(|&p| src.dims[axis(p)]).collect();
    let view_strides: Vec<i64> = perm.iter().map(|&p| src_strides[axis(p)]).collect();

    let view_ty = Type::parse(
        gen.context,
        &format!(
            "memref<{}, strided<[{}], offset: 0>>",
            dims_text(&view_dims, src.el_ty_str),
            i64_list(&view_strides)
        ),
    )
    .unwrap();
    let view = StaticView {
        sizes: &view_dims,
        strides: &view_strides,
        ty: view_ty,
    };
    let view_val = reinterpret_static(gen, block, src.val, view);

    let dst_ty = Type::parse(
        gen.context,
        &format!("memref<{}>", dims_text(&view_dims, src.el_ty_str)),
    )
    .unwrap();
    let alloc = OperationBuilder::new("memref.alloc", gen.loc())
        .add_attributes(&[(
            Identifier::new(gen.context, "operandSegmentSizes"),
            DenseI32ArrayAttribute::new(gen.context, &[0, 0]).into(),
        )])
        .add_results(&[dst_ty])
        .build()?;
    let dst_val = block.append_operation(alloc).result(0)?.into();

    let copy = OperationBuilder::new("memref.copy", gen.loc())
        .add_operands(&[view_val, dst_val])
        .build()?;
    block.append_operation(copy);

    Ok((dst_val, dst_ty, block))
}

/// `reshape`, `transpose` and `pad` on a tensor.
///
/// `memref.cast` cannot reshape or permute data (it only changes static/dynamic/ranked
/// info), so these are modelled as buffer views:
///   - reshape: reinterpret the contiguous source as the target shape (a row-major view;
///     also handles a smaller "trim" target).
///   - transpose: a permuted-stride view, copied into a new contiguous buffer.
///
/// # Errors
///
/// Returns an error if the tensor cannot be lowered, or if MLIR refuses an operation.
///
/// # Panics
///
/// Panics if the tensor's element type cannot be read from its memref type, or if a type built
/// here does not parse.
fn lower_reshape_like<'c>(
    this: &FunctionCallExpr,
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
) -> Lowered<'c> {
    let FunctionCallExpr { name, args, .. } = this;
    let is_transpose = **name == *"transpose";
    let (arg_val, expr_ty, block) = tensor_with_static_extents(gen, block, &args[0])?;
    let expr_ty_str = expr_ty.to_string();
    let el_ty_str = extract_mlir_element_type(&expr_ty_str).unwrap_or_else(|e| panic!("{}", e));

    // Static dimensions of the source, parsed from "memref<AxBx...xT>".
    let src_dims: Vec<i64> = expr_ty_str
        .trim_start_matches("memref<")
        .split('x')
        .map_while(|t| t.parse::<i64>().ok())
        .collect();
    // The second argument is an integer array: the target shape for
    // `reshape`, or the dimension permutation for `transpose`.
    let idx_vals = integer_array(&args[1]);

    if is_transpose {
        let src = StaticTensor {
            val: arg_val,
            dims: &src_dims,
            el_ty_str,
        };
        return lower_transpose(gen, block, src, &idx_vals);
    }

    if **name == *"pad" {
        let target = PadTarget {
            dims: &idx_vals,
            each_axis: true,
        };
        return lower_padded_copy(gen, block, arg_val, &expr_ty_str, target);
    }

    // reshape
    let tgt_dims = idx_vals;
    let tgt_strides = row_major_strides(&tgt_dims);
    let tgt_ty = Type::parse(
        gen.context,
        &format!("memref<{}>", dims_text(&tgt_dims, el_ty_str)),
    )
    .unwrap();
    if args.get(2).is_some_and(crate::syntax::is_pad_mode_pad) {
        let target = PadTarget {
            dims: &tgt_dims,
            each_axis: false,
        };
        return lower_padded_copy(gen, block, arg_val, &expr_ty_str, target);
    }
    let view = StaticView {
        sizes: &tgt_dims,
        strides: &tgt_strides,
        ty: tgt_ty,
    };
    let out_val = reinterpret_static(gen, block, arg_val, view);
    Ok((out_val, tgt_ty, block))
}

/// `t.as_ptr()` / `t.as_mut_ptr()` on a tensor: its data pointer as a bare `!llvm.ptr`.
///
/// # Errors
///
/// Returns an error if the argument cannot be lowered, is not a tensor, or is a view of one.
fn lower_tensor_data_pointer<'c>(
    this: &FunctionCallExpr,
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
) -> Lowered<'c> {
    let FunctionCallExpr { name, args, .. } = this;
    let (val, val_ty, block) = gen.generate_expr(&args[0], block)?;
    let ty_text = val_ty.to_string();
    // `(&mut c[0][0]).as_mut_ptr()`: a borrow is already the element's address.
    if val_ty == gen.ptr_ty {
        return Ok((val, gen.ptr_ty, block));
    }
    if !ty_text.starts_with("memref<") || ty_text.starts_with("memref<memref<") {
        return Err(LowerError::from(format!(
            "`{name}()` on something that is not a tensor: {ty_text}"
        )));
    }
    // The helper answers with the buffer's start. A view (a row, `q[i]`) starts at an
    // offset into it, which this path does not add, so it is refused rather than
    // answered wrongly. The flat path handles it.
    if ty_text.contains("strided<") {
        return Err(LowerError::from(format!(
            "`{name}()` on a tensor view is not lowered by the legacy code generator"
        )));
    }
    let ptr = memref_data_pointer(gen, block, val)?;
    Ok((ptr, gen.ptr_ty, block))
}

/// Declares the C function `printf` in the module, the first time it is called.
///
/// # Errors
///
/// Returns an error if MLIR refuses the declaration.
fn declare_printf<'c>(
    gen: &mut MeliorGenerator<'c>,
    printf_func_ty: TypeAttribute<'c>,
) -> Result<(), LowerError> {
    // Iterate over operations to check if printf is already declared
    // We can't iterate easily over body in melior, but we can just use a separate set for llvm_funcs.
    // Wait, we can just check a boolean in gen or insert "llvm_printf" into gen.functions.
    // Let's just forcefully insert it if we haven't seen it in gen.functions, BUT wait,
    // gen.functions has "printf" from the AST extern. So let's use a unique key for the llvm decl.
    if !gen.functions.contains_key("llvm_printf_decl") {
        let printf_decl = OperationBuilder::new("llvm.func", gen.loc())
            .add_attributes(&[
                (
                    Identifier::new(gen.context, "sym_name"),
                    StringAttribute::new(gen.context, "printf").into(),
                ),
                (
                    Identifier::new(gen.context, "linkage_name"),
                    StringAttribute::new(gen.context, "printf").into(),
                ),
                (
                    Identifier::new(gen.context, "function_type"),
                    printf_func_ty.into(),
                ),
            ])
            .add_regions([melior::ir::Region::new()])
            .build()?;

        gen.module.body().append_operation(printf_decl);
        gen.functions.insert(
            "llvm_printf_decl".to_string().into(),
            (gen.i32_ty, vec![gen.ptr_ty]),
        );
    }
    Ok(())
}

/// `printf(..)`: a call to the C function with these arguments.
///
/// # Errors
///
/// Returns an error if an argument cannot be lowered, or if MLIR refuses an operation.
///
/// # Panics
///
/// Panics if the type of `printf` does not parse.
fn lower_printf_call<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    args: &[Expr],
) -> Lowered<'c> {
    let mut arg_vals = Vec::new();
    let mut current_b = block;
    for arg in args {
        let (arg_val, _arg_ty, new_b) = gen.generate_expr(arg, current_b)?;
        arg_vals.push(arg_val);
        current_b = new_b;
    }

    let printf_func_ty =
        TypeAttribute::new(Type::parse(gen.context, "!llvm.func<i32 (!llvm.ptr, ...)>").unwrap());
    declare_printf(gen, printf_func_ty)?;

    let call_op = current_b.append_operation(
        OperationBuilder::new("llvm.call", gen.loc())
            .add_operands(&arg_vals)
            .add_attributes(&[
                (
                    Identifier::new(gen.context, "callee"),
                    FlatSymbolRefAttribute::new(gen.context, "printf").into(),
                ),
                (
                    Identifier::new(gen.context, "var_callee_type"),
                    printf_func_ty.into(),
                ),
                (
                    Identifier::new(gen.context, "operandSegmentSizes"),
                    DenseI32ArrayAttribute::new(gen.context, &[count_i32(arg_vals.len()), 0])
                        .into(),
                ),
                (
                    Identifier::new(gen.context, "op_bundle_sizes"),
                    DenseI32ArrayAttribute::new(gen.context, &[]).into(),
                ),
            ])
            .add_results(&[gen.i32_ty])
            .build()?,
    );
    Ok((call_op.result(0)?.into(), gen.i32_ty, current_b))
}

/// An argument made to fit the type the function takes: a reference to a tensor is given the
/// expected type, a memref is cast, a closure literal becomes the `{env, func}` pair of a
/// nominal closure struct, and anything else is coerced.
///
/// # Errors
///
/// Returns an error if the value cannot be made to fit.
fn fit_argument<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    arg_val: Value<'c, 'c>,
    expr_ty: Type<'c>,
    field_ty: Type<'c>,
) -> Result<Value<'c, 'c>, LowerError> {
    if expr_ty == field_ty {
        return Ok(arg_val);
    }
    if expr_ty.to_string().starts_with("memref<memref<")
        && field_ty.to_string().starts_with("memref<memref<")
    {
        retype_tensor_reference(gen, block, arg_val, expr_ty, field_ty)
    } else if gen.is_memref(&expr_ty) && gen.is_memref(&field_ty) {
        crate::codegen::lower::cast_memref(gen, &block, arg_val, expr_ty, field_ty)
    } else if let Some(adapted) =
        gen.adapt_closure_to_nominal(&block, arg_val, expr_ty, field_ty)?
    {
        // A closure literal (`Closure_N` env) passed where a nominal `ClosureK`
        // struct is expected (e.g. `.map(|x| ..)`): build the `{env, func}` value.
        Ok(adapted)
    } else {
        gen.coerce_type(&block, arg_val, expr_ty, field_ty)
    }
}

/// Ends a call: gives the call its result type, appends it, and gives back its value. A call
/// to a function that returns nothing gives back a placeholder that is never read.
///
/// # Errors
///
/// Returns an error if MLIR refuses the call.
fn finish_call<'c>(
    gen: &MeliorGenerator<'c>,
    current_b: melior::ir::BlockRef<'c, 'c>,
    builder: OperationBuilder<'c>,
    ret_ty: Type<'c>,
) -> Lowered<'c> {
    let mut builder = builder;
    if ret_ty.to_string() == "none" {
        let call_op = builder.build()?;
        current_b.append_operation(call_op);
        let none_ty = gen.none_ty;
        // this value shouldn't be used
        let dummy_op = OperationBuilder::new("llvm.mlir.constant", gen.loc())
            .add_results(&[gen.i32_ty])
            .add_attributes(&[(
                Identifier::new(gen.context, "value"),
                IntegerAttribute::new(gen.i32_ty, 0).into(),
            )])
            .build()?;
        Ok((
            current_b.append_operation(dummy_op).result(0)?.into(),
            none_ty,
            current_b,
        ))
    } else {
        builder = builder.add_results(&[ret_ty]);
        let call_op = builder.build()?;
        let call_ref = current_b.append_operation(call_op);
        Ok((call_ref.result(0)?.into(), ret_ty, current_b))
    }
}

/// A call to a function the program declares, by name.
///
/// # Errors
///
/// Returns an error if an argument cannot be lowered or made to fit, or if MLIR refuses an
/// operation.
fn lower_direct_call<'c>(
    this: &FunctionCallExpr,
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    ret_ty: Type<'c>,
    arg_tys: &[Type<'c>],
) -> Lowered<'c> {
    let FunctionCallExpr { name, args, .. } = this;
    // A callee returning a statically shaped tensor writes into a buffer we allocate and
    // pass as its first argument, and the call itself produces nothing -- the result is
    // that buffer (#643).
    let ret_slot = if returns_through_slot(&ret_ty.to_string()) {
        let alloc_op = OperationBuilder::new("memref.alloca", gen.loc())
            .add_attributes(&[(
                Identifier::new(gen.context, "operandSegmentSizes"),
                DenseI32ArrayAttribute::new(gen.context, &[0, 0]).into(),
            )])
            .add_results(&[ret_ty])
            .build()?;
        Some(block.append_operation(alloc_op).result(0)?.into())
    } else {
        None
    };
    let mut arg_vals: Vec<Value<'c, 'c>> = ret_slot.into_iter().collect();
    let mut current_b = block;
    for (i, arg) in args.iter().enumerate() {
        let field_ty = arg_tys[i];
        let prev_expected = gen.expected_type;
        gen.expected_type = Some(field_ty);
        let (arg_val, expr_ty, new_b) = gen.generate_expr(arg, current_b)?;
        current_b = new_b;
        gen.expected_type = prev_expected;
        arg_vals.push(fit_argument(gen, current_b, arg_val, expr_ty, field_ty)?);
    }

    let name_attr = FlatSymbolRefAttribute::new(gen.context, name);
    let builder = OperationBuilder::new("func.call", gen.loc())
        .add_operands(&arg_vals)
        .add_attributes(&[(Identifier::new(gen.context, "callee"), name_attr.into())]);

    if let Some(slot) = ret_slot {
        current_b.append_operation(builder.build()?);
        return Ok((slot, ret_ty, current_b));
    }
    finish_call(gen, current_b, builder, ret_ty)
}

/// The MLIR function type of the function pointer or closure held in the variable `name`,
/// whose value has the type `func_ty`.
///
/// # Errors
///
/// Returns an error if a type in the signature cannot be lowered.
///
/// # Panics
///
/// Panics if the variable's signature was not recorded.
fn called_value_type<'c>(
    gen: &MeliorGenerator<'c>,
    name: &crate::symbol::Symbol,
    func_ty: Type<'c>,
) -> Result<Type<'c>, LowerError> {
    let mut actual_func_ty = func_ty;
    let is_closure = func_ty.to_string() == "!llvm.struct<(ptr, ptr)>";
    if func_ty.to_string() == "!llvm.ptr" {
        if let Some(syntax::Type::Function(func_args, ret, _)) = gen.ast_env.get(name) {
            let r = gen.lower_type(ret.as_ref())?;
            let mut a: Vec<_> = Vec::new();
            for t in func_args {
                a.push(gen.lower_type(t)?);
            }
            actual_func_ty = function_type(gen, &a, r);
        } else {
            panic!("Missing signature for function pointer '{name}'");
        }
    } else if is_closure {
        if let Some(syntax::Type::Closure(func_args, ret)) = gen.ast_env.get(name) {
            let r = gen.lower_type(ret.as_ref())?;
            let mut a: Vec<_> = vec![gen.ptr_ty];
            for t in func_args {
                a.push(gen.lower_type(t)?);
            }
            actual_func_ty = function_type(gen, &a, r);
        } else {
            panic!("Missing signature for closure '{name}'");
        }
    }
    Ok(actual_func_ty)
}

/// The two halves of a closure value: the pointer to what it captured, and the pointer to
/// its function.
///
/// # Errors
///
/// Returns an error if MLIR refuses an operation.
fn closure_parts<'c>(
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    ptr_val: Value<'c, 'c>,
) -> Result<(Value<'c, 'c>, Value<'c, 'c>), LowerError> {
    // Extract env_ptr
    let extract_env_op = OperationBuilder::new("llvm.extractvalue", gen.loc())
        .add_operands(&[ptr_val])
        .add_attributes(&[(
            Identifier::new(gen.context, "position"),
            melior::ir::attribute::DenseI64ArrayAttribute::new(gen.context, &[0]).into(),
        )])
        .add_results(&[gen.ptr_ty])
        .build()?;
    let extract_env_ref = block.append_operation(extract_env_op);
    let env_ptr = extract_env_ref.result(0)?.into();

    // Extract func_ptr
    let extract_func_op = OperationBuilder::new("llvm.extractvalue", gen.loc())
        .add_operands(&[ptr_val])
        .add_attributes(&[(
            Identifier::new(gen.context, "position"),
            melior::ir::attribute::DenseI64ArrayAttribute::new(gen.context, &[1]).into(),
        )])
        .add_results(&[gen.ptr_ty])
        .build()?;
    let extract_func_ref = block.append_operation(extract_func_op);
    Ok((env_ptr, extract_func_ref.result(0)?.into()))
}

/// A call through a variable that holds a function pointer or a closure.
///
/// # Errors
///
/// Returns an error if an argument cannot be lowered, or if MLIR refuses an operation.
///
/// # Panics
///
/// Panics if the variable has no function type, if the call passes more arguments than that
/// type takes, or if MLIR refuses the cast to it.
fn lower_call_through_value<'c>(
    this: &FunctionCallExpr,
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    ptr_val: Value<'c, 'c>,
    func_ty: Type<'c>,
) -> Lowered<'c> {
    let FunctionCallExpr { name, args, .. } = this;
    let is_closure = func_ty.to_string() == "!llvm.struct<(ptr, ptr)>";
    let actual_func_ty = called_value_type(gen, name, func_ty)?;
    let Ok(mlir_func_ty) = melior::ir::r#type::FunctionType::try_from(actual_func_ty) else {
        panic!("Function pointer {name} missing type info. func_ty={func_ty}");
    };
    let ret_ty = mlir_func_ty.result(0).unwrap_or(gen.none_ty);
    let mut arg_vals = Vec::new();

    let mut arg_offset = 0;
    let mut env_ptr = None;
    let mut actual_ptr_val = ptr_val;

    if is_closure {
        arg_offset = 1;
        let (env, func) = closure_parts(gen, block, ptr_val)?;
        env_ptr = Some(env);
        actual_ptr_val = func;
    }

    let mut current_b = block;
    for (i, arg) in args.iter().enumerate() {
        let (mut arg_val, expr_ty, new_b) = gen.generate_expr(arg, current_b)?;
        current_b = new_b;
        let field_ty = mlir_func_ty.input(i + arg_offset).unwrap();
        if expr_ty != field_ty {
            arg_val = gen.coerce_type(&current_b, arg_val, expr_ty, field_ty)?;
        }
        arg_vals.push(arg_val);
    }

    if func_ty.to_string() == "!llvm.ptr" || is_closure {
        let cast_op = OperationBuilder::new("builtin.unrealized_conversion_cast", gen.loc())
            .add_operands(&[actual_ptr_val])
            .add_results(&[actual_func_ty])
            .build()
            .unwrap();
        let cast_ref = current_b.append_operation(cast_op);
        actual_ptr_val = cast_ref.result(0)?.into();
    }

    let mut builder =
        OperationBuilder::new("func.call_indirect", gen.loc()).add_operands(&[actual_ptr_val]);

    if let Some(ep) = env_ptr {
        builder = builder.add_operands(&[ep]);
    }

    for a in &arg_vals {
        builder = builder.add_operands(&[*a]);
    }
    finish_call(gen, current_b, builder, ret_ty)
}

impl<'c> LowerToMelior<'c> for FunctionCallExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let FunctionCallExpr {
            name,
            args,
            type_args,
            span: _,
        } = self;
        if name.as_ref() == "Verified" {
            return gen.generate_expr(&args[0], block);
        }
        // The eight raw:: transfer-lowering primitives (#353). They reach codegen
        // only inside an inlined `impl transfer` body -- the checker refuses them
        // anywhere else (E6017) -- and they lower to loads/stores/gpu ops directly in
        // place, never to calls: a func.call inside a kernel region silently costs the
        // kernel its device twin (`isDeviceLowerableDialect` excludes func).
        if let Some(prim) = name.as_ref().strip_prefix("raw::") {
            return lower_raw_primitive(gen, block, prim, args);
        }
        // `tensor_view_2d(ptr, rows, cols)` -- a rank-2 tensor over memory it
        // does not own, with no copy (#336).
        //
        // The alternative is `Tensor::from_ptr_2d`, which allocates and copies
        // element by element. That is fine once at startup and impossible per
        // token, and a scalar copy loop is not something a later pass removes.
        // A memref *is* a descriptor, so pointing one at existing storage is
        // the natural operation; it just had no spelling.
        //
        // Built here rather than in the stdlib because it has to construct the
        // descriptor `{allocated, aligned, offset, sizes, strides}` directly:
        // `*mut f32` lowers to `!llvm.ptr`, and no memref op takes a bare
        // pointer as its source.
        if name.as_ref() == "matmul_into" {
            return lower_matmul_into(gen, args, block);
        }

        if name.as_ref() == "tensor_view_2d" {
            return lower_tensor_view_2d(gen, args, type_args.as_deref(), block);
        }

        if matches!(
            name.as_ref(),
            "Tensor" | "Tensor::new" | "Tensor::uninit" | "Tensor::fill"
        ) {
            return lower_tensor_constructor(self, gen, block);
        }

        if name.as_ref() == "reshape"
            || **name == *"transpose"
            || (**name == *"pad" && args.len() == 2 && matches!(args[1], Expr::Array(_)))
        {
            return lower_reshape_like(self, gen, block);
        }

        if name.as_ref() == "clone" && args.len() == 1 {
            return lower_clone_call(gen, block, &args[0]);
        }

        // `t.as_ptr()` / `t.as_mut_ptr()` on a tensor: its data pointer as a bare `!llvm.ptr`.
        // A method of that name on anything else was renamed by the checker before it got here,
        // so a bare `as_ptr` with one argument is always the tensor intrinsic.
        if matches!(name.as_ref(), "as_ptr" | "as_mut_ptr") && args.len() == 1 {
            return lower_tensor_data_pointer(self, gen, block);
        }

        if name.as_ref() == "map" {
            return lower_map_call(gen, block, args);
        }

        if name.as_ref() == "reduce" && args.len() == 3 {
            return lower_reduce_call(gen, block, args);
        }

        if name.as_ref() == "print" {
            return lower_print_call(gen, block, args);
        }

        if name.as_ref() == "printf" || **name == *"vx_internal_printf" {
            return lower_printf_call(gen, block, args);
        }

        if let Some((ret_ty, arg_tys)) = gen.functions.get(name).cloned() {
            lower_direct_call(self, gen, block, ret_ty, &arg_tys)
        } else if let Some((ptr_val, func_ty)) = gen.env.get(name).copied() {
            lower_call_through_value(self, gen, block, ptr_val, func_ty)
        } else {
            panic!("Function {name} not found");
        }
    }
}

impl<'c> LowerToMelior<'c> for IndirectCallExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let IndirectCallExpr {
            callee,
            args,
            target_func_ty,
            span: _,
        } = self;

        let (callee_val, callee_ty, block) = gen.generate_expr(callee, block)?;

        if callee_ty.to_string() == "!llvm.struct<(ptr, ptr)>" {
            let (env_ptr, mut actual_ptr_val) = closure_parts(gen, block, callee_val)?;

            let mut arg_vals = Vec::new();
            let mut current_b = block;
            for arg in args {
                let (arg_val, _, new_b) = gen.generate_expr(arg, current_b)?;
                arg_vals.push(arg_val);
                current_b = new_b;
            }

            // In our current implementation we assume all functions taking closure
            // arguments are dynamically typed correctly via Sema. We don't have
            // the exact MLIR FunctionType statically available here without
            // looking it up, but `func.call_indirect` requires the operand to
            // match the callee type exactly. Since we don't have the explicit
            // func signature here, we construct it from the generated arg types!
            // To figure out the return type, we use `target_func_ty` from Sema.
            let target_func_ty = target_func_ty
                .as_ref()
                .expect("IndirectCallExpr MLIR lowering needs explicit target_func_ty from Sema");
            let syntax::Type::Closure(func_args, ret) = target_func_ty else {
                panic!("Expected Type::Closure for indirect call fat pointer target");
            };

            let r = gen.lower_type(ret)?;
            let mut a: Vec<_> = vec![gen.ptr_ty];
            for t in func_args {
                a.push(gen.lower_type(t)?);
            }
            let actual_func_ty = function_type(gen, &a, r);
            let ret_ty = r;

            // Cast the raw func ptr to the actual function signature
            let cast_op = OperationBuilder::new("builtin.unrealized_conversion_cast", gen.loc())
                .add_operands(&[actual_ptr_val])
                .add_results(&[actual_func_ty])
                .build()?;
            let cast_ref = current_b.append_operation(cast_op);
            actual_ptr_val = cast_ref.result(0)?.into();

            let mut builder = OperationBuilder::new("func.call_indirect", gen.loc())
                .add_operands(&[actual_ptr_val, env_ptr]);

            for a_val in &arg_vals {
                builder = builder.add_operands(&[*a_val]);
            }

            finish_call(gen, current_b, builder, ret_ty)
        } else {
            panic!("Unsupported callee type for indirect call: {callee_ty}");
        }
    }
}

impl<'c> LowerToMelior<'c> for MethodCallExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let MethodCallExpr {
            base,
            method_name,
            args,
            type_args: _,
            span,
        } = self;
        let mut new_args = vec![*base.clone()];
        new_args.extend(args.clone());
        gen.generate_expr(
            &Expr::FunctionCall(FunctionCallExpr {
                name: method_name.clone(),
                type_args: None,
                args: new_args,
                span: *span,
            }),
            block,
        )
    }
}

/// The values given to an `mlir!` block, and each input written as an MLIR function
/// parameter, `name: type`.
///
/// # Errors
///
/// Returns an error if an input cannot be lowered, or if its type does not parse.
fn lower_mlir_inputs<'c>(
    this: &InlineMlirExpr,
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
) -> Result<(Vec<Value<'c, 'c>>, Vec<String>), LowerError> {
    let mut mlir_args = Vec::new();
    let mut input_types_str = Vec::new();

    for (name, expr, ty_str) in &this.inputs {
        let (mut val, val_ty, block) = gen.generate_expr(expr, block)?;
        if val_ty.to_string().starts_with("memref<memref<") && ty_str.starts_with("memref<") {
            let load_op = OperationBuilder::new("memref.load", gen.loc())
                .add_operands(&[val])
                .add_results(&[Type::parse(gen.context, ty_str).ok_or_else(|| {
                    crate::codegen::lower::LowerError::ParseType("Type::parse failed".to_string())
                })?])
                .build()?;
            val = block.append_operation(load_op).result(0)?.into();
        }
        mlir_args.push(val);
        input_types_str.push(format!("{name}: {ty_str}"));
    }
    Ok((mlir_args, input_types_str))
}

/// Adds the body of an `mlir!` block to the module as a private function with these
/// parameters and this result, and gives back the function's name.
///
/// # Errors
///
/// Returns an error if the block does not parse as MLIR.
///
/// # Panics
///
/// Panics if the parsed module holds no function, which a module that parsed always does.
fn add_mlir_block_function(
    this: &InlineMlirExpr,
    gen: &mut MeliorGenerator<'_>,
    args_str: &str,
    ret_str: &str,
) -> Result<String, LowerError> {
    let unique_id = format!(
        "vx_macro_mlir_L{}_C{}_{}",
        this.span.line, this.span.column, gen.mlir_block_counter
    );
    gen.mlir_block_counter += 1;

    let block_str = this.block_str.replace("macro.yield", "return");

    let mlir_source = format!(
        "module {{\n    func.func private @{unique_id}({args_str}) {ret_str} {{\n{block_str}\n    }}\n}}"
    );

    gen.context.set_allow_unregistered_dialects(true);
    // A block a programmer can write wrong is a diagnostic, not a crash (Vx#414).
    let parsed_module = melior::ir::Module::parse(gen.context, &mlir_source).ok_or_else(|| {
        LowerError::ParseType(format!(
            "mlir! block at line {}, column {} does not parse:\n{}",
            this.span.line, this.span.column, mlir_source
        ))
    })?;

    let func_op = parsed_module.body().first_operation().unwrap();
    let cloned_func = (*func_op).clone();

    gen.module.body().append_operation(cloned_func);
    Ok(unique_id)
}

impl<'c> LowerToMelior<'c> for InlineMlirExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;

    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let (mlir_args, input_types_str) = lower_mlir_inputs(self, gen, block)?;

        let args_str = input_types_str.join(", ");

        let mut ret_str = String::new();
        let mut call_ret_tys = Vec::new();
        if let Some(ret_ty) = &self.returns {
            let mlir_ret_ty = gen.lower_type(ret_ty)?;
            if mlir_ret_ty.to_string() != "void" {
                ret_str = format!("-> {mlir_ret_ty}");
            }
            call_ret_tys.push(mlir_ret_ty);
        }

        let unique_id = add_mlir_block_function(self, gen, &args_str, &ret_str)?;

        let call_op = OperationBuilder::new("func.call", gen.loc())
            .add_attributes(&[(
                Identifier::new(gen.context, "callee"),
                FlatSymbolRefAttribute::new(gen.context, &unique_id).into(),
            )])
            .add_operands(&mlir_args)
            .add_results(&call_ret_tys)
            .build()?;

        let op = block.append_operation(call_op);

        if let Some(ret_ty) = &self.returns {
            Ok((op.result(0)?.into(), gen.lower_type(ret_ty)?, block))
        } else {
            let dummy_val = OperationBuilder::new("arith.constant", gen.loc())
                .add_attributes(&[(
                    Identifier::new(gen.context, "value"),
                    IntegerAttribute::new(Type::index(gen.context), 0).into(),
                )])
                .add_results(&[Type::index(gen.context)])
                .build()?;
            Ok((
                block.append_operation(dummy_val).result(0)?.into(),
                Type::index(gen.context),
                block,
            ))
        }
    }
}

impl<'c> LowerToMelior<'c> for ArrayExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let ArrayExpr { elements, span: _ } = self;
        // These three used to be a `panic!` and two `unwrap`s, so an array literal the checker
        // let through killed the compiler instead of reporting anything (Vx#354). E3018 now
        // rejects both shapes with a span before codegen runs; this is the second line of
        // defence, because an internal error is the one outcome that tells the programmer
        // nothing at all.
        if elements.is_empty() {
            return Err(LowerError::from(
                "an empty array literal has no element type to lower".to_string(),
            ));
        }
        let mut vals = Vec::new();
        let mut el_ty = None;
        let mut current_b = block;
        for el in elements {
            let (v, t, new_b) = gen.generate_expr(el, current_b)?;
            vals.push(v);
            current_b = new_b;
            if el_ty.is_none() {
                el_ty = Some(t);
            }
        }
        let el_ty = el_ty.ok_or_else(|| {
            LowerError::from("an array literal produced no element type".to_string())
        })?;
        let num_elements = elements.len();

        // A memref, like every other tensor value this backend produces. `tensor.from_elements`
        // gave the literal a `tensor<Nx...>` that nothing downstream reads -- indexing emits
        // `memref.load`, and no bufferization runs between the two (Vx#422).
        //
        // A non-scalar element has no spelling here, and unlike `tensor<Nx tensor<..>>` a
        // `memref<Nx memref<..>>` parses -- so the check has to be its own rather than a failed
        // parse, else a nested literal reaches the verifier instead of a diagnostic.
        let el_str = el_ty.to_string();
        let scalar_element = !el_str.contains('<') && !el_str.starts_with('!');
        let memref_ty = Type::parse(gen.context, &format!("memref<{num_elements}x{el_ty}>"))
            .filter(|_| scalar_element)
            .ok_or_else(|| {
                LowerError::from(format!(
                    "an array literal's elements must be scalars; `{el_ty}` cannot be the \
                     element type of a tensor"
                ))
            })?;

        let alloc = OperationBuilder::new("memref.alloc", gen.loc())
            .add_attributes(&[(
                Identifier::new(gen.context, "operandSegmentSizes"),
                DenseI32ArrayAttribute::new(gen.context, &[0, 0]).into(),
            )])
            .add_results(&[memref_ty])
            .build()?;
        let buf: Value<'_, '_> = current_b.append_operation(alloc).result(0)?.into();

        let index_ty = Type::index(gen.context);
        for (i, v) in vals.into_iter().enumerate() {
            let idx_op = OperationBuilder::new("arith.constant", gen.loc())
                .add_attributes(&[(
                    Identifier::new(gen.context, "value"),
                    IntegerAttribute::new(index_ty, count_i64(i)).into(),
                )])
                .add_results(&[index_ty])
                .build()?;
            let idx = current_b.append_operation(idx_op).result(0)?.into();
            let store = OperationBuilder::new("memref.store", gen.loc())
                .add_operands(&[v, buf, idx])
                .build()?;
            current_b.append_operation(store);
        }

        Ok((buf, memref_ty, current_b))
    }
}

impl<'c> LowerToMelior<'c> for MemorySpaceExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        _gen: &mut MeliorGenerator<'c>,
        _block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        panic!("Should not be evaluated directly")
    }
}

impl<'c> LowerToMelior<'c> for TopologyExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        // A topology used as a value lowers to its stable runtime dispatch id
        // (an i32 discriminant, `arch::topology_dispatch_id`). This makes
        // `Topology::GPU` storable (e.g. in `Vec<Topology>`) and comparable at
        // runtime. Comptime placement comparisons are folded before codegen, so
        // a topology reaching here is genuinely being used as a value.
        let i32_ty = gen.i32_ty;
        let id = i64::from(crate::arch::topology_dispatch_id(&self.top));
        let op = OperationBuilder::new("arith.constant", gen.loc())
            .add_results(&[i32_ty])
            .add_attributes(&[(
                Identifier::new(gen.context, "value"),
                IntegerAttribute::new(i32_ty, id).into(),
            )])
            .build()?;
        let val = block.append_operation(op).result(0)?.into();
        Ok((val, i32_ty, block))
    }
}

impl<'c> LowerToMelior<'c> for NumberExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let NumberExpr {
            value: val_str,
            ty: ast_ty_opt,
            span: _,
        } = self;
        let ty = if let Some(ast_ty) = ast_ty_opt {
            gen.lower_type(&syntax::Type::Scalar(ast_ty.clone()))
                .unwrap()
        } else if val_str.contains('.') {
            gen.f32_ty
        } else {
            gen.i32_ty
        };
        let ty_str = ty.to_string();
        if ty_str.contains("f32")
            || ty_str.contains("f64")
            || ty_str.contains("f16")
            || ty_str.contains("bf16")
        {
            let op = OperationBuilder::new("arith.constant", gen.loc())
                .add_results(&[ty])
                .add_attributes(&[(
                    Identifier::new(gen.context, "value"),
                    FloatAttribute::new(gen.context, ty, val_str.parse::<f64>().unwrap()).into(),
                )])
                .build()?;
            let op_ref = block.append_operation(op);
            Ok((op_ref.result(0)?.into(), ty, block))
        } else {
            // A `u64` above `i64::MAX` is a legitimate literal that does not parse as one:
            // MLIR's integer types are signless, so it goes in as the same 64 bits read as
            // signed. The flat path's `encode_imm` falls back the same way, and the two
            // backends have to agree on a constant.
            let bits = val_str
                .parse::<i64>()
                .or_else(|_| val_str.parse::<u64>().map(u64::cast_signed))
                .unwrap_or_else(|_| panic!("integer literal out of range: {val_str}"));
            let op = OperationBuilder::new("arith.constant", gen.loc())
                .add_results(&[ty])
                .add_attributes(&[(
                    Identifier::new(gen.context, "value"),
                    IntegerAttribute::new(ty, bits).into(),
                )])
                .build()?;
            let op_ref = block.append_operation(op);
            Ok((op_ref.result(0)?.into(), ty, block))
        }
    }
}

impl<'c> LowerToMelior<'c> for MatchExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        // Read before the scrutinee is generated, which may set it for its own operands.
        let outer_expected = gen.expected_type;
        let (match_val, match_ty, block) = gen.generate_expr(&self.expr, block)?;

        // The match yields a value exactly when an arm ends in a tail expression -- the same
        // rule the checker uses to type it. The merge block then takes that value as a block
        // argument and each arm branches with its own, which is what makes the match evaluate
        // to what its arms say. The type comes from the first such arm rather than from
        // `expected_type`, which is unset for an unannotated `let`.
        let tail_expr = self.arms.iter().find_map(|arm| match arm.body.last() {
            Some(syntax::Statement::ExprStmt(syntax::stmt::ExprStmtStmt {
                expr,
                has_semi: false,
                ..
            })) => Some(expr),
            _ => None,
        });
        // The fallback to `expected_type` applies only when there *is* a tail expression whose
        // type could not be inferred. Applying it to a match with no tail expression at all
        // would make a statement-position match look value-producing, and every arm would then
        // have to yield something it never had.
        // A statement `match` produces no value, whatever its arms end in: a trailing `if` or
        // `unsafe { }` there is a statement too.
        let in_statement = outer_expected == Some(gen.none_ty);
        let value_ty = tail_expr.filter(|_| !in_statement).and_then(|e| {
            match e {
                // `infer_ast_type` has no arm for a bare literal, and an arm body is usually
                // exactly that. Read the literal's own type, which the checker stamps, and fall
                // back to how the parser would have typed it.
                Expr::Number(n) => {
                    let elem = n.ty.clone().unwrap_or_else(|| {
                        crate::parser::expr::default_number_elem(n.value.as_ref())
                    });
                    gen.lower_type(&syntax::Type::Scalar(elem)).ok()
                }
                other => gen
                    .infer_ast_type(other)
                    .and_then(|t| gen.lower_type(&t).ok()),
            }
            .or(gen.expected_type)
        });

        let parent_region = block.parent_region().unwrap();
        let merge_block = match value_ty {
            Some(ty) => parent_region.append_block(melior::ir::Block::new(&[(ty, gen.loc())])),
            None => parent_region.append_block(melior::ir::Block::new(&[])),
        };
        generate_match_chain(
            gen,
            &self.arms,
            match_val,
            match_ty,
            block,
            merge_block,
            value_ty,
        )?;
        let block = merge_block;

        if let Some(ty) = value_ty {
            return Ok((block.argument(0)?.into(), ty, block));
        }

        // No arm ends in a tail expression, so the match produces nothing of its own. There
        // are two ways to arrive here and they need different answers.
        //
        // A match written as a statement is the ordinary one: nothing reads the result, and a
        // placeholder is the whole story.
        //
        // The other is a match whose arms each `return`. The parser turns a trailing match into
        // `return match ...`, so the caller is waiting for a value of the function's return
        // type -- and the merge block it would come from is unreachable, because every arm has
        // already left. An `i32` placeholder handed back there was coerced to whatever the
        // caller wanted, which for an aggregate meant an unrealized conversion cast that no
        // backend can translate. An undef of the awaited type says the same thing and says it
        // in a type the caller can return: this edge is not taken.
        // A `void` function's return type lowers to MLIR's `none`, which is not a value any
        // operation can produce -- and a void caller has nothing to receive anyway, so the
        // placeholder below is the right answer there.
        if let Some(ty) = outer_expected.filter(|t| t.to_string() != "none") {
            let op = OperationBuilder::new("llvm.mlir.undef", gen.loc())
                .add_results(&[ty])
                .build()?;
            let op_ref = block.append_operation(op);
            return Ok((op_ref.result(0)?.into(), ty, block));
        }
        let ty = melior::ir::r#type::IntegerType::new(gen.context, 32).into();
        let op = OperationBuilder::new("arith.constant", gen.loc())
            .add_results(&[ty])
            .add_attributes(&[(
                Identifier::new(gen.context, "value"),
                IntegerAttribute::new(ty, 0).into(),
            )])
            .build()?;
        let op_ref = block.append_operation(op);
        Ok((op_ref.result(0)?.into(), ty, block))
    }
}

/// Whether `want` is a strictly narrower integer type than `slot`, so a value read out of
/// the slot has to be truncated to it. Floats are left out: reinterpreting one is the
/// denormal hazard `enum_payload_per_variant.vx` records.
pub(crate) fn is_narrower_int(slot: &str, want: &str) -> bool {
    let bits = |t: &str| t.strip_prefix('i').and_then(|r| r.parse::<u32>().ok());
    matches!((bits(slot), bits(want)), (Some(s), Some(w)) if w < s)
}

/// The payload slots of an enum's `{tag, slot, slot, ..}` struct type, one per payload
/// position, split at the commas that are not inside a nested type.
pub(crate) fn enum_payload_slot_texts(struct_ty_text: &str) -> Vec<String> {
    let Some(start) = struct_ty_text.find("(i32, ").map(|i| i + 6) else {
        return Vec::new();
    };
    let Some(end) = struct_ty_text.rfind(')') else {
        return Vec::new();
    };
    if end <= start {
        return Vec::new();
    }
    let mut slots = Vec::new();
    let (mut depth, mut from) = (0i32, start);
    for (i, c) in struct_ty_text[start..end].char_indices() {
        match c {
            '<' | '(' => depth += 1,
            '>' | ')' => depth -= 1,
            ',' if depth == 0 => {
                slots.push(struct_ty_text[from..start + i].trim().to_string());
                from = start + i + 1;
            }
            _ => {}
        }
    }
    slots.push(struct_ty_text[from..end].trim().to_string());
    slots
}

/// Resizes an enum payload to the width of the slot that holds it.
///
/// A slot is sized for the widest variant, so a narrower variant is zero-extended into
/// it. The AST path then binds a matched payload at the slot type rather than at its own
/// variant's type, so the value coming back is too wide and is truncated here. Either
/// direction round-trips the bits, so the signedness this path no longer knows does not
/// matter. A genuinely too-wide value cannot reach this point: E3008 rejects it.
///
/// # Errors
///
/// Returns an error if a type does not parse, or if MLIR refuses an operation.
pub(crate) fn fit_payload_to_slot_text<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    payload_val: Value<'c, 'c>,
    payload_ty: Type<'c>,
    slot_text: &str,
) -> Result<Value<'c, 'c>, LowerError> {
    let slot_text = slot_text.to_string();
    // Integers only. Reinterpreting an integer into a float slot would make a denormal
    // that a copy may flush to zero, which is the hazard
    // `tests/optimizations/pass/enum_payload_per_variant.vx` records.
    let int_bits = |t: &str| t.strip_prefix('i').and_then(|r| r.parse::<u32>().ok());
    // A loop variable is an `index` in MLIR and an integer in Vx, and `index` is not a type a
    // struct can hold, so it is cast to the slot's integer type.
    if payload_ty == Type::index(gen.context) && int_bits(&slot_text).is_some() {
        let slot_ty = Type::parse(gen.context, &slot_text)
            .ok_or_else(|| LowerError::ParseType("Type::parse failed".to_string()))?;
        let cast = OperationBuilder::new("arith.index_cast", gen.loc())
            .add_operands(&[payload_val])
            .add_results(&[slot_ty])
            .build()?;
        return Ok(block.append_operation(cast).result(0)?.into());
    }
    // A float in an integer slot goes in as its bits.
    let mut payload_val = payload_val;
    let mut payload_ty = payload_ty;
    let payload_text = payload_ty.to_string();
    if crate::codegen::generator::is_float_text(&payload_text) && int_bits(&slot_text).is_some() {
        let bits = crate::codegen::generator::scalar_type_bits(&payload_text)
            .ok_or_else(|| LowerError::from(format!("a float of no width: {payload_text}")))?;
        let as_int = Type::parse(gen.context, &format!("i{bits}"))
            .ok_or_else(|| LowerError::ParseType("Type::parse failed".to_string()))?;
        let cast = OperationBuilder::new("arith.bitcast", gen.loc())
            .add_operands(&[payload_val])
            .add_results(&[as_int])
            .build()?;
        payload_val = block.append_operation(cast).result(0)?.into();
        payload_ty = as_int;
    }
    let (Some(have), Some(want)) = (int_bits(&payload_ty.to_string()), int_bits(&slot_text)) else {
        return Ok(payload_val);
    };
    if have == want {
        return Ok(payload_val);
    }
    let slot_ty = Type::parse(gen.context, &slot_text)
        .ok_or_else(|| LowerError::ParseType("Type::parse failed".to_string()))?;
    let op = if have < want {
        "arith.extui"
    } else {
        "arith.trunci"
    };
    let cast = OperationBuilder::new(op, gen.loc())
        .add_operands(&[payload_val])
        .add_results(&[slot_ty])
        .build()?;
    Ok(block.append_operation(cast).result(0)?.into())
}

/// The tag of an enum variant, the MLIR type of the enum as text, and whether that type has
/// room for a payload. An enum with no payload is the bare tag, an `i32`.
///
/// # Errors
///
/// Returns an error if the enum's type cannot be lowered.
fn enum_tag_and_type(
    this: &EnumVariantExpr,
    gen: &MeliorGenerator<'_>,
) -> Result<(i64, String, bool), LowerError> {
    let EnumVariantExpr {
        enum_name,
        variant_name,
        ..
    } = this;
    let mut tag_val = 0;
    let mut enum_ty_str = "i32".to_string();
    let mut has_payload = false;

    let actual_enum_name = if let Some(idx) = enum_name.find('<') {
        &enum_name[0..idx]
    } else {
        enum_name
    };

    if let Some(enum_def) = gen.enums.get(actual_enum_name) {
        for (i, v) in enum_def.iter().enumerate() {
            if *v.0 == **variant_name {
                tag_val = count_i64(i);
                break;
            }
        }
        // An instantiated generic enum lowers to a `{tag, payload}` struct, and is told
        // apart by its name carrying the arguments it was instantiated with. This used to
        // read `starts_with("Option<")` -- recognising one enum by name. Every other
        // generic enum fell through to the bare-tag branch below and lost its payload; the
        // `i32` then reached whatever wanted the struct, which coerced it with an
        // unrealized conversion cast: valid MLIR that no backend can translate.
        //
        // The turbofish arguments arrive re-serialized into the name, so the payload type
        // comes back out through the type parser.
        if enum_name.contains('<') {
            if let Some(t) = crate::parser::types::parse_type_text(enum_name) {
                enum_ty_str = gen.lower_type_str(&t)?;
            }
            has_payload = true;
        } else if let Some(ty) = gen.enum_payload_type_str(actual_enum_name) {
            // A non-generic enum with a payload has the same `{tag, payload}` shape.
            enum_ty_str = ty;
            has_payload = true;
        }
    }
    Ok((tag_val, enum_ty_str, has_payload))
}

/// Puts each value of a variant's payload in the slot for its position in the enum's struct.
/// Gives back the struct and the block the next operation goes in.
///
/// # Errors
///
/// Returns an error if a payload value cannot be lowered or made to fit its slot, or if there
/// are more values than slots.
fn insert_enum_payload<'c>(
    this: &EnumVariantExpr,
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    struct_val: Value<'c, 'c>,
    struct_ty: Type<'c>,
) -> Result<(Value<'c, 'c>, melior::ir::BlockRef<'c, 'c>), LowerError> {
    let EnumVariantExpr {
        enum_name,
        variant_name,
        payload,
        ..
    } = this;
    let mut struct_val = struct_val;
    let mut block = block;
    if let Some(payload_exprs) = payload {
        // Each element of the payload goes in the slot for its position.
        let slots = enum_payload_slot_texts(&struct_ty.to_string());
        for (i, payload_expr) in payload_exprs.iter().enumerate() {
            let (payload_val, payload_val_ty, next) = gen.generate_expr(payload_expr, block)?;
            block = next;
            let slot = slots.get(i).ok_or_else(|| {
                crate::codegen::lower::LowerError::from(format!(
                    "enum variant {enum_name}::{variant_name} has more payload values than \
                     its type has slots"
                ))
            })?;
            let payload_val =
                fit_payload_to_slot_text(gen, block, payload_val, payload_val_ty, slot)?;
            let insert_payload_op = OperationBuilder::new("llvm.insertvalue", gen.loc())
                .add_operands(&[struct_val, payload_val])
                .add_results(&[struct_ty])
                .add_attributes(&[(
                    Identifier::new(gen.context, "position"),
                    melior::ir::attribute::DenseI64ArrayAttribute::new(
                        gen.context,
                        &[count_i64(i) + 1],
                    )
                    .into(),
                )])
                .build()?;
            struct_val = block.append_operation(insert_payload_op).result(0)?.into();
        }
    }

    Ok((struct_val, block))
}

impl<'c> LowerToMelior<'c> for EnumVariantExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let (tag_val, enum_ty_str, has_payload) = enum_tag_and_type(self, gen)?;

        let i32_ty = gen.i32_ty;
        let tag_op = OperationBuilder::new("arith.constant", gen.loc())
            .add_results(&[i32_ty])
            .add_attributes(&[(
                Identifier::new(gen.context, "value"),
                IntegerAttribute::new(i32_ty, tag_val).into(),
            )])
            .build()?;
        let tag_val = block.append_operation(tag_op).result(0)?.into();

        if !has_payload {
            return Ok((tag_val, i32_ty, block));
        }

        // We have an Option<T> struct
        let struct_ty = Type::parse(gen.context, &enum_ty_str).ok_or_else(|| {
            crate::codegen::lower::LowerError::ParseType("Type::parse failed".to_string())
        })?;
        let undef_op = OperationBuilder::new("llvm.mlir.undef", gen.loc())
            .add_results(&[struct_ty])
            .build()?;
        let undef_val = block.append_operation(undef_op).result(0)?.into();

        let insert_tag_op = OperationBuilder::new("llvm.insertvalue", gen.loc())
            .add_operands(&[undef_val, tag_val])
            .add_results(&[struct_ty])
            .add_attributes(&[(
                Identifier::new(gen.context, "position"),
                melior::ir::attribute::DenseI64ArrayAttribute::new(gen.context, &[0]).into(),
            )])
            .build()?;
        let struct_val = block.append_operation(insert_tag_op).result(0)?.into();

        let (struct_val, block) = insert_enum_payload(self, gen, block, struct_val, struct_ty)?;
        Ok((struct_val, struct_ty, block))
    }
}

/// The end of the name of the `Vec` type a `vec![..]` makes, taken from its first element:
/// `i32` for `Vec_i32`.
fn vec_macro_suffix(this: &VecMacroExpr, gen: &MeliorGenerator<'_>) -> &'static str {
    let mut el_ty = syntax::ElementType::F32;
    if !this.elements.is_empty() {
        if let Some(syntax::Type::Scalar(t)) = gen.infer_ast_type(&this.elements[0]) {
            el_ty = t;
        } else if let Some(syntax::Type::Struct(s, _)) = gen.infer_ast_type(&this.elements[0]) {
            if s == "String".into() {
                // String is equivalent to pointer, but generic instantiation requires element type
            }
        }
    }

    match el_ty {
        syntax::ElementType::I32 => "i32",
        syntax::ElementType::F32 => "f32",
        syntax::ElementType::I64 => "i64",
        syntax::ElementType::F64 => "f64",
        syntax::ElementType::Bool => "Bool",
        _ => {
            if let Some(syntax::Type::Struct(s, _)) =
                gen.infer_ast_type(this.elements.first().unwrap_or(&Expr::Number(NumberExpr {
                    value: "0".into(),
                    ty: None,
                    span: Span::default(),
                })))
            {
                if s == "String".into() {
                    "String"
                } else {
                    "f32"
                }
            } else {
                "f32"
            }
        }
    }
}

/// Puts a new `Vec` in a stack slot, so that it can be borrowed, and gives back the slot.
///
/// # Errors
///
/// Returns an error if MLIR refuses an operation built here.
fn spill_vec<'c>(
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    vec_val: Value<'c, 'c>,
    vec_ty: Type<'c>,
) -> Result<Value<'c, 'c>, LowerError> {
    let i32_ty = gen.i32_ty;
    let c1_op = block.append_operation(
        OperationBuilder::new("llvm.mlir.constant", gen.loc())
            .add_results(&[i32_ty])
            .add_attributes(&[(
                Identifier::new(gen.context, "value"),
                IntegerAttribute::new(i32_ty, 1).into(),
            )])
            .build()?,
    );
    let c1 = c1_op.result(0)?.into();

    let ptr_ty = gen.ptr_ty;
    let alloca_op = block.append_operation(
        OperationBuilder::new("llvm.alloca", gen.loc())
            .add_operands(&[c1])
            .add_results(&[ptr_ty])
            .add_attributes(&[(
                Identifier::new(gen.context, "elem_type"),
                TypeAttribute::new(vec_ty).into(),
            )])
            .build()?,
    );
    let ptr_val = alloca_op.result(0)?.into();

    block.append_operation(
        OperationBuilder::new("llvm.store", gen.loc())
            .add_operands(&[vec_val, ptr_val])
            .build()?,
    );
    Ok(ptr_val)
}

/// Pushes each element of a `vec![..]` onto the `Vec` held in the variable `tmp_vec_name`.
///
/// # Errors
///
/// Returns an error if a `push` call cannot be lowered.
fn push_vec_elements<'c>(
    this: &VecMacroExpr,
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    type_suffix: &str,
    tmp_vec_name: &str,
) -> Result<(), LowerError> {
    for el in &this.elements {
        let push_call = Expr::FunctionCall(FunctionCallExpr {
            name: format!("Vec_{type_suffix}::push").into(),
            args: vec![
                Expr::Borrow(BorrowExpr {
                    expr: Box::new(Expr::Identifier(IdentifierExpr {
                        name: tmp_vec_name.to_string().into(),
                        span: Span::default(),
                    })),
                    is_mut: true,
                    span: Span::default(),
                }),
                el.clone(),
            ],
            type_args: None,
            span: this.span,
        });
        gen.generate_expr(&push_call, block)?;
    }
    Ok(())
}

impl<'c> LowerToMelior<'c> for VecMacroExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let type_suffix = vec_macro_suffix(self, gen);

        let new_call = Expr::FunctionCall(FunctionCallExpr {
            name: format!("Vec_{type_suffix}::new").into(),
            args: vec![],
            type_args: None,
            span: self.span,
        });

        let (vec_val, vec_ty, block) = gen.generate_expr(&new_call, block)?;

        let ptr_val = spill_vec(gen, block, vec_val, vec_ty)?;

        let tmp_vec_name = format!("__vec_ptr_{}", gen.string_counter);
        gen.string_counter += 1;
        gen.env
            .insert(tmp_vec_name.clone().into(), (ptr_val, vec_ty));
        gen.allocs.insert(tmp_vec_name.clone());

        push_vec_elements(self, gen, block, type_suffix, &tmp_vec_name)?;

        let load_op = block.append_operation(
            OperationBuilder::new("llvm.load", gen.loc())
                .add_operands(&[ptr_val])
                .add_results(&[vec_ty])
                .build()?,
        );
        Ok((load_op.result(0)?.into(), vec_ty, block))
    }
}

impl<'c> LowerToMelior<'c> for ClosureExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;

    fn lower(
        &self,
        _gen: &mut MeliorGenerator<'c>,
        _block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        unreachable!("ClosureExpr should be transformed to StructInitExpr by Sema")
    }
}

impl<'c> LowerToMelior<'c> for syntax::expr::PrintExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        for arg in &self.args {
            let (arg_val, arg_ty, block) = gen.generate_expr(arg, block)?;
            if let Some((fn_name, arg_val, arg_ty_str)) =
                super::pick_scalar_print(gen, block, arg, arg_val, arg_ty)?
            {
                super::call_scalar_print(gen, block, fn_name, arg_val, &arg_ty_str)?;
                continue;
            }

            let arg_ty_str = arg_ty.to_string();
            if arg_ty_str == r#"!llvm.struct<"String", (ptr)>"# {
                // A `String` prints through a NUL-terminated copy of its contents, freed after.
                let handle = block
                    .append_operation(
                        OperationBuilder::new("llvm.extractvalue", gen.loc())
                            .add_operands(&[arg_val])
                            .add_results(&[gen.ptr_ty])
                            .add_attributes(&[(
                                Identifier::new(gen.context, "position"),
                                Attribute::parse(gen.context, "array<i64: 0>").unwrap(),
                            )])
                            .build()?,
                    )
                    .result(0)?
                    .into();
                let c_str = super::call_runtime_fn(
                    gen,
                    block,
                    "vx_string_as_c_str",
                    &[(handle, "!llvm.ptr")],
                    "!llvm.ptr",
                )?;
                super::call_runtime_fn(gen, block, "print_str", &[(c_str, "!llvm.ptr")], "i32")?;
                super::call_runtime_fn(
                    gen,
                    block,
                    "vx_string_free_c_str",
                    &[(c_str, "!llvm.ptr")],
                    "i32",
                )?;
                continue;
            }
            if !matches!(arg_ty_str.as_str(), "!llvm.ptr" | "!llvm.ptr<i8>") {
                return Err(LowerError::from(format!(
                    "print cannot print a value of type {arg_ty}"
                )));
            }
            super::call_runtime_fn(gen, block, "print_str", &[(arg_val, "!llvm.ptr")], "i32")?;
        }

        let dummy_op = OperationBuilder::new("llvm.mlir.constant", gen.loc())
            .add_results(&[gen.i32_ty])
            .add_attributes(&[(
                Identifier::new(gen.context, "value"),
                IntegerAttribute::new(gen.i32_ty, 0).into(),
            )])
            .build()?;

        Ok((
            block.append_operation(dummy_op).result(0)?.into(),
            gen.i32_ty,
            block,
        ))
    }
}

impl<'c> LowerToMelior<'c> for syntax::expr::PrintlnExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        // First, reuse PrintExpr logic for arguments
        if !self.args.is_empty() {
            let print_expr = syntax::expr::PrintExpr {
                args: self.args.clone(),
                span: self.span,
            };
            print_expr.lower(gen, block)?;
        }

        // Then emit a call to println()
        if !gen.functions.contains_key("println") {
            let func_ty = Type::parse(gen.context, "() -> i32").unwrap();

            let func_decl = OperationBuilder::new("func.func", gen.loc())
                .add_attributes(&[
                    (
                        Identifier::new(gen.context, "sym_name"),
                        StringAttribute::new(gen.context, "println").into(),
                    ),
                    (
                        Identifier::new(gen.context, "function_type"),
                        TypeAttribute::new(func_ty).into(),
                    ),
                    (
                        Identifier::new(gen.context, "sym_visibility"),
                        StringAttribute::new(gen.context, "private").into(),
                    ),
                ])
                .add_regions([melior::ir::Region::new()])
                .build()?;

            gen.module.body().append_operation(func_decl);
            gen.functions
                .insert("println".to_string().into(), (gen.i32_ty, vec![]));
        }

        let name_attr = FlatSymbolRefAttribute::new(gen.context, "println");
        let call_op = OperationBuilder::new("func.call", gen.loc())
            .add_results(&[gen.i32_ty])
            .add_attributes(&[(Identifier::new(gen.context, "callee"), name_attr.into())])
            .build()?;

        Ok((
            block.append_operation(call_op).result(0)?.into(),
            gen.i32_ty,
            block,
        ))
    }
}

impl<'c> LowerToMelior<'c> for syntax::expr::SizeOfExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;

    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let size: i64 = match &self.target_ty {
            // Scalar byte size from the single width source (`scalar_size_align` -> `ElementType::bits`),
            // rather than a fourth hand-maintained per-type table; also sizes sub-byte scalars (`i4` = 1)
            // that the old match dropped to the `_ => 8` fallback.
            syntax::Type::Scalar(e) => {
                crate::layout::scalar_size_align(e).map_or(8, |(s, _)| count_i64(s))
            }
            // A nominal struct (or a monomorphized generic instance of one) gets its *real* layout
            // size, not the old hardcoded `8` — otherwise a `Vec<struct>`'s element buffer sized by
            // `sizeof<T>()` under-allocates and its element stores run out of bounds (a `Vec<Vec<T>>`
            // UB). Falls back to `8` for a type the layout pass doesn't model (tensor, closure). (#242)
            ty @ (syntax::Type::Struct(..) | syntax::Type::GenericInstance(..)) => {
                gen.type_size_align(ty, 0).map_or(8, |(s, _)| count_i64(s))
            }
            // An unbound type parameter has no size, and the checker refuses the call that
            // would bring one here, so this is an internal error rather than a case for 8.
            ty @ syntax::Type::Generic(..) => {
                return Err(LowerError::from(format!(
                    "internal: sizeof of a generic type reached codegen (should have been \
                     monomorphized): {ty:?}"
                )));
            }
            // A pointer or a reference is 8 bytes, and so is any type whose layout is not
            // worked out here.
            _ => 8,
        };

        let size_ty = gen.i64_ty;
        let const_op = OperationBuilder::new("arith.constant", gen.loc())
            .add_attributes(&[(
                Identifier::new(gen.context, "value"),
                IntegerAttribute::new(size_ty, size).into(),
            )])
            .add_results(&[size_ty])
            .build()?;

        let const_op = block.append_operation(const_op);
        Ok((const_op.result(0)?.into(), size_ty, block))
    }
}

/// The name of the `Closure_N` struct that a cast to a closure type starts from.
///
/// # Panics
///
/// Panics if the cast's source is not a `Closure_N` struct or a borrow of one.
fn cast_source_closure_name(source_ty: Option<&syntax::Type>) -> crate::symbol::Symbol {
    match source_ty {
        Some(syntax::Type::Struct(name, _)) => name.clone(),
        Some(syntax::Type::Borrow { inner, .. }) => {
            if let syntax::Type::Struct(name, _) = &**inner {
                name.clone()
            } else {
                panic!("Expected Closure_N struct, got {inner:?}");
            }
        }
        Some(t) => panic!("Expected Closure_N struct, got {t:?}"),
        None => panic!("Missing source_ty for AsCast expression"),
    }
}

/// The address of what a closure captured. A by-value env struct is spilled to a stack
/// slot; a borrowed env is already a pointer and is used as it is.
///
/// # Errors
///
/// Returns an error if MLIR refuses an operation built here.
fn closure_env_pointer<'c>(
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    source: Typed<'c>,
) -> Result<Value<'c, 'c>, LowerError> {
    let (source_val, source_mlir_ty) = source;
    let ptr_ty = gen.ptr_ty;
    let mut env_ptr_val = source_val;
    if source_mlir_ty != ptr_ty {
        let i32_ty = gen.i32_ty;
        let c1 = block
            .append_operation(
                OperationBuilder::new("llvm.mlir.constant", gen.loc())
                    .add_results(&[i32_ty])
                    .add_attributes(&[(
                        Identifier::new(gen.context, "value"),
                        IntegerAttribute::new(i32_ty, 1).into(),
                    )])
                    .build()?,
            )
            .result(0)?
            .into();
        let slot: Value<'c, 'c> = block
            .append_operation(
                OperationBuilder::new("llvm.alloca", gen.loc())
                    .add_operands(&[c1])
                    .add_results(&[ptr_ty])
                    .add_attributes(&[(
                        Identifier::new(gen.context, "elem_type"),
                        TypeAttribute::new(source_mlir_ty).into(),
                    )])
                    .build()?,
            )
            .result(0)?
            .into();
        block.append_operation(
            OperationBuilder::new("llvm.store", gen.loc())
                .add_operands(&[env_ptr_val, slot])
                .build()?,
        );
        env_ptr_val = slot;
    }
    Ok(env_ptr_val)
}

/// A `Closure_N` value cast to a closure type: the `{ env, func }` pair that every call
/// through a closure takes apart.
///
/// # Errors
///
/// Returns an error if MLIR refuses an operation built here.
///
/// # Panics
///
/// Panics if the cast's source is not a `Closure_N` struct, if that struct has no call
/// function, or if the pair's type does not parse.
fn cast_closure_to_fat_pointer<'c>(
    this: &syntax::expr::AsCastExpr,
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    source: Typed<'c>,
) -> Lowered<'c> {
    let closure_struct_name = cast_source_closure_name(this.source_ty.as_ref());

    let call_fn_name = format!("{closure_struct_name}_call");
    let (ret_ty, orig_arg_types) = gen
        .functions
        .get(&*call_fn_name)
        .cloned()
        .expect("Closure call function not found");

    let fn_ty = melior::ir::r#type::FunctionType::new(gen.context, &orig_arg_types, &[ret_ty]);
    let const_op = OperationBuilder::new("func.constant", gen.loc())
        .add_attributes(&[(
            Identifier::new(gen.context, "value"),
            FlatSymbolRefAttribute::new(gen.context, &call_fn_name).into(),
        )])
        .add_results(&[fn_ty.into()])
        .build()?;
    let const_ref = block.append_operation(const_op);
    let mut fn_ptr_val: Value<'c, 'c> = const_ref.result(0)?.into();

    let ptr_ty = gen.ptr_ty;
    let bitcast_op = OperationBuilder::new("builtin.unrealized_conversion_cast", gen.loc())
        .add_operands(&[fn_ptr_val])
        .add_results(&[ptr_ty])
        .build()?;
    let bitcast_ref = block.append_operation(bitcast_op);
    fn_ptr_val = bitcast_ref.result(0)?.into();

    // Fat-pointer layout is { env, func }, the ClosureK convention every call site extracts.
    let fat_ptr_ty = Type::parse(gen.context, "!llvm.struct<(ptr, ptr)>").unwrap();
    let undef_op = OperationBuilder::new("llvm.mlir.undef", gen.loc())
        .add_results(&[fat_ptr_ty])
        .build()?;
    let undef_ref = block.append_operation(undef_op);
    let mut fat_ptr_val: Value<'c, 'c> = undef_ref.result(0)?.into();

    let insert_fn_op = OperationBuilder::new("llvm.insertvalue", gen.loc())
        .add_operands(&[fat_ptr_val, fn_ptr_val])
        .add_attributes(&[(
            Identifier::new(gen.context, "position"),
            melior::ir::attribute::DenseI64ArrayAttribute::new(gen.context, &[1]).into(),
        )])
        .add_results(&[fat_ptr_ty])
        .build()?;
    let insert_fn_ref = block.append_operation(insert_fn_op);
    fat_ptr_val = insert_fn_ref.result(0)?.into();
    let env_ptr_val = closure_env_pointer(gen, block, source)?;

    let insert_env_op = OperationBuilder::new("llvm.insertvalue", gen.loc())
        .add_operands(&[fat_ptr_val, env_ptr_val])
        .add_attributes(&[(
            Identifier::new(gen.context, "position"),
            melior::ir::attribute::DenseI64ArrayAttribute::new(gen.context, &[0]).into(),
        )])
        .add_results(&[fat_ptr_ty])
        .build()?;
    let insert_env_ref = block.append_operation(insert_env_op);
    fat_ptr_val = insert_env_ref.result(0)?.into();

    Ok((fat_ptr_val, fat_ptr_ty, block))
}

/// A cast to the scalar type `target_elem`.
///
/// Widening an unsigned number brings in zeroes, not copies of its top bit, and an unsigned
/// number converts to and from a float with the unsigned operations. `coerce_type` cannot
/// tell: it sees only the printed MLIR type, which has no sign. The cast does know, because
/// it kept the type it started from.
///
/// # Errors
///
/// Returns an error if the target type cannot be lowered, or if the value cannot be converted.
fn cast_to_scalar<'c>(
    this: &syntax::expr::AsCastExpr,
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    source: Typed<'c>,
    target_elem: &syntax::ElementType,
) -> Lowered<'c> {
    let (source_val, source_mlir_ty) = source;
    let target_ty_mlir = gen.lower_type(&this.target_ty)?;
    // Widening an unsigned number brings in zeroes, not copies of its top bit.
    // `coerce_type` cannot tell: it sees only the printed MLIR type, which has no
    // sign. The cast does know, because it kept the type it started from.
    if let Some(syntax::Type::Scalar(source_elem)) = this.source_ty.as_ref() {
        let widening = matches!(
            (source_elem.int_bits(), target_elem.int_bits()),
            (Some(from), Some(to)) if from < to
        );
        if widening && !source_elem.is_signed_int() {
            let ext = OperationBuilder::new("arith.extui", gen.loc())
                .add_operands(&[source_val])
                .add_results(&[target_ty_mlir])
                .build()?;
            let widened = block.append_operation(ext).result(0)?.into();
            return Ok((widened, target_ty_mlir, block));
        }
        // For the same reason, an unsigned number converts to and from a float with
        // the unsigned ops: `3000000000u32 as f64` is not negative.
        let unsigned_float_op = if source_elem.int_bits().is_some()
            && !source_elem.is_signed_int()
            && target_elem.is_float()
            && melior::ir::TypeLike::is_integer(&source_mlir_ty)
        {
            Some("arith.uitofp")
        } else if source_elem.is_float()
            && target_elem.int_bits().is_some()
            && !target_elem.is_signed_int()
            && melior::ir::TypeLike::is_integer(&target_ty_mlir)
        {
            Some("arith.fptoui")
        } else {
            None
        };
        if let Some(op_name) = unsigned_float_op {
            let op = OperationBuilder::new(op_name, gen.loc())
                .add_operands(&[source_val])
                .add_results(&[target_ty_mlir])
                .build()?;
            let converted = block.append_operation(op).result(0)?.into();
            return Ok((converted, target_ty_mlir, block));
        }
    }
    let coerced_val = gen.coerce_type(&block, source_val, source_mlir_ty, target_ty_mlir)?;
    Ok((coerced_val, target_ty_mlir, block))
}

/// A cast to a raw pointer, from a borrow, another raw pointer or an integer. `None` for any
/// other source.
///
/// # Errors
///
/// Returns an error if a borrow is neither a pointer nor a scalar's slot, or if MLIR refuses an
/// operation.
fn cast_to_pointer<'c>(
    this: &syntax::expr::AsCastExpr,
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    source: Typed<'c>,
) -> Result<Option<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>)>, LowerError> {
    let (source_val, source_mlir_ty) = source;
    // `&mut x as *mut T`: the borrow is a pointer already, or a scalar's slot whose data
    // pointer is the address.
    if let Some(syntax::Type::Borrow { .. }) = this.source_ty.as_ref() {
        let ty_text = source_val.r#type().to_string();
        if source_val.r#type() == gen.ptr_ty {
            return Ok(Some((source_val, gen.ptr_ty, block)));
        }
        if ty_text.starts_with("memref<") && !ty_text.starts_with("memref<memref<") {
            let ptr = memref_data_pointer(gen, block, source_val)?;
            return Ok(Some((ptr, gen.ptr_ty, block)));
        }
        return Err(LowerError::from(format!(
            "a borrow cast to a raw pointer, from {ty_text}"
        )));
    }
    // Between `*const T` and `*mut T` the address is unchanged.
    if let Some(syntax::Type::Pointer(..)) = this.source_ty.as_ref() {
        return Ok(Some((source_val, source_mlir_ty, block)));
    }
    if let Some(syntax::Type::Scalar(_)) = this.source_ty.as_ref() {
        let ptr_ty = gen.ptr_ty;
        let cast_op = OperationBuilder::new("llvm.inttoptr", gen.loc())
            .add_operands(&[source_val])
            .add_results(&[ptr_ty])
            .build()?;
        let cast_ref = block.append_operation(cast_op);
        return Ok(Some((cast_ref.result(0)?.into(), ptr_ty, block)));
    }
    Ok(None)
}

impl<'c> LowerToMelior<'c> for syntax::expr::AsCastExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;

    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        // A slice `as` is a vector conversion, which only the flat path emits. Say so:
        // without this the legacy walk tried to build a scalar conversion from a vector
        // and failed much later with `ParseType`, naming nothing.
        if let Some(syntax::Type::Tensor(_, dims, _)) = self.source_ty.as_ref() {
            assert!(
                !(dims.len() == 1 && matches!(&self.target_ty, syntax::Type::Scalar(_))),
                "an elementwise cast over a slice has no legacy (AST) lowering; it \
                 requires the flat path (drop --legacy-codegen)"
            );
        }
        let (source_val, source_mlir_ty, block) = gen.generate_expr(&self.expr, block)?;
        let source = (source_val, source_mlir_ty);
        if let syntax::Type::Closure(_, _) = &self.target_ty {
            return cast_closure_to_fat_pointer(self, gen, block, source);
        } else if let syntax::Type::Scalar(target_elem) = &self.target_ty {
            return cast_to_scalar(self, gen, block, source, target_elem);
        } else if let syntax::Type::Pointer(..) = &self.target_ty {
            if let Some(lowered) = cast_to_pointer(self, gen, block, source)? {
                return Ok(lowered);
            }
        }

        panic!("Unsupported cast operation in codegen");
    }
}

/// One size of a tensor: known while compiling, or read at run time.
#[derive(Clone, Copy)]
enum Size<'c> {
    Known(i64),
    AtRunTime(Value<'c, 'c>),
}

/// The constant `v` as an `index` value.
///
/// # Errors
///
/// Returns an error if MLIR refuses an operation built here.
fn index_const<'c>(
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    v: i64,
) -> Result<Value<'c, 'c>, LowerError> {
    let index_ty = Type::index(gen.context);
    let op = OperationBuilder::new("arith.constant", gen.loc())
        .add_results(&[index_ty])
        .add_attributes(&[(
            Identifier::new(gen.context, "value"),
            IntegerAttribute::new(index_ty, v).into(),
        )])
        .build()?;
    Ok(block.append_operation(op).result(0)?.into())
}

/// The sizes and element type of the tensor `src`, whose type is `src_str`
/// ("memref<2x?xf32, ..>"). Each `?` is read with `memref.dim`.
///
/// # Errors
///
/// Returns an error if `src_str` is not a tensor's memref type, or if MLIR refuses an operation.
fn tensor_sizes<'c>(
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    src: Value<'c, 'c>,
    src_str: &str,
) -> Result<(Vec<Size<'c>>, String), LowerError> {
    let inner = src_str
        .strip_prefix("memref<")
        .and_then(|s| s.split([',', '>']).next())
        .ok_or_else(|| format!("sizes of something that is not a tensor: {src_str}"))?;
    let mut parts: Vec<&str> = inner.split('x').collect();
    let elem = parts
        .pop()
        .ok_or_else(|| format!("a tensor with no element type: {src_str}"))?
        .to_string();
    let mut sizes = Vec::new();
    for (k, d) in parts.iter().enumerate() {
        if *d == "?" {
            let c = index_const(gen, block, count_i64(k))?;
            let dim = OperationBuilder::new("memref.dim", gen.loc())
                .add_operands(&[src, c])
                .add_results(&[Type::index(gen.context)])
                .build()?;
            sizes.push(Size::AtRunTime(
                block.append_operation(dim).result(0)?.into(),
            ));
        } else {
            // The parse error says only that the text is not a number, as this message does.
            let Ok(n) = d.parse() else {
                return Err(LowerError::from(format!(
                    "a tensor size that is not a number: {src_str}"
                )));
            };
            sizes.push(Size::Known(n));
        }
    }
    Ok((sizes, elem))
}

/// The number of elements of a tensor with these sizes.
///
/// # Errors
///
/// Returns an error if MLIR refuses an operation built here.
fn element_count<'c>(
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    sizes: &[Size<'c>],
) -> Result<Size<'c>, LowerError> {
    let mut known = 1i64;
    let mut at_run_time: Option<Value<'c, 'c>> = None;
    for s in sizes {
        match s {
            Size::Known(n) => known *= n,
            Size::AtRunTime(v) => {
                at_run_time = Some(match at_run_time {
                    None => *v,
                    Some(acc) => {
                        let mul = OperationBuilder::new("arith.muli", gen.loc())
                            .add_operands(&[acc, *v])
                            .add_results(&[Type::index(gen.context)])
                            .build()?;
                        block.append_operation(mul).result(0)?.into()
                    }
                });
            }
        }
    }
    let Some(v) = at_run_time else {
        return Ok(Size::Known(known));
    };
    let c = index_const(gen, block, known)?;
    let mul = OperationBuilder::new("arith.muli", gen.loc())
        .add_operands(&[v, c])
        .add_results(&[Type::index(gen.context)])
        .build()?;
    Ok(Size::AtRunTime(
        block.append_operation(mul).result(0)?.into(),
    ))
}

/// The sizes and the strides of a view of a tensor, one of each per dimension.
#[derive(Clone, Copy)]
struct ViewShape<'a, 'c> {
    /// The size of each dimension.
    sizes: &'a [Size<'c>],
    /// The stride of each dimension, in elements.
    strides: &'a [i64],
}

/// `memref.reinterpret_cast` of `src` at offset 0 with these sizes and strides. A size read
/// at run time is an operand, and a `?` in the result type.
///
/// # Errors
///
/// Returns an error if the view's type does not parse, or if MLIR refuses the operation.
fn reinterpret_with_sizes<'c>(
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    src: Value<'c, 'c>,
    shape: ViewShape<'_, 'c>,
    elem: &str,
) -> Result<(Value<'c, 'c>, Type<'c>), LowerError> {
    let ViewShape { sizes, strides } = shape;
    let dims: Vec<String> = sizes
        .iter()
        .map(|s| match s {
            Size::Known(n) => n.to_string(),
            Size::AtRunTime(_) => "?".to_string(),
        })
        .collect();
    let contiguous = sizes.len() == 1 && strides == [1];
    let layout = if contiguous {
        String::new()
    } else {
        let s: Vec<String> = strides
            .iter()
            .map(std::string::ToString::to_string)
            .collect();
        format!(", strided<[{}], offset: 0>", s.join(", "))
    };
    let ty_str = format!("memref<{}x{elem}{layout}>", dims.join("x"));
    let ty = Type::parse(gen.context, &ty_str).ok_or(LowerError::ParseType(ty_str))?;
    let dynamic: Vec<Value<'_, '_>> = sizes
        .iter()
        .filter_map(|s| match s {
            Size::AtRunTime(v) => Some(*v),
            Size::Known(_) => None,
        })
        .collect();
    let static_sizes: Vec<i64> = sizes
        .iter()
        .map(|s| match s {
            Size::Known(n) => *n,
            Size::AtRunTime(_) => i64::MIN,
        })
        .collect();
    let mut operands = vec![src];
    operands.extend(&dynamic);
    let op = OperationBuilder::new("memref.reinterpret_cast", gen.loc())
        .add_operands(&operands)
        .add_attributes(&[
            (
                Identifier::new(gen.context, "operandSegmentSizes"),
                DenseI32ArrayAttribute::new(gen.context, &[1, 0, count_i32(dynamic.len()), 0])
                    .into(),
            ),
            (
                Identifier::new(gen.context, "static_offsets"),
                DenseI64ArrayAttribute::new(gen.context, &[0]).into(),
            ),
            (
                Identifier::new(gen.context, "static_sizes"),
                DenseI64ArrayAttribute::new(gen.context, &static_sizes).into(),
            ),
            (
                Identifier::new(gen.context, "static_strides"),
                DenseI64ArrayAttribute::new(gen.context, strides).into(),
            ),
        ])
        .add_results(&[ty])
        .build()?;
    Ok((block.append_operation(op).result(0)?.into(), ty))
}

/// The shape a padded copy fills, and how the source is laid out in it.
#[derive(Clone, Copy)]
struct PadTarget<'a> {
    /// The sizes of the new buffer.
    dims: &'a [i64],
    /// `true` for `t.pad(shape)`, which keeps each element at its own indices; `false` for
    /// `t.reshape(shape, PadMode::Pad)`, which keeps the elements in row order.
    each_axis: bool,
}

/// A new buffer of `tgt_dims` filled with zeros, with the source copied into it.
/// - `t.reshape(shape, PadMode::Pad)`: into its first elements, both seen as flat, so the copy
///   keeps row order.
/// - `t.pad(shape)` (`each_axis`): into a view of the new buffer with the source's sizes and
///   the new buffer's strides, so element `[i][j]` stays at `[i][j]`.
///
/// A source size known only at run time is read with `memref.dim`.
///
/// # Errors
///
/// Returns an error if `src_str` is not a tensor's memref type, if the new buffer's type does
/// not parse, or if MLIR refuses an operation.
///
/// # Panics
///
/// Panics if `pad` changes the number of axes, or if `PadMode::Pad` is given a smaller shape.
/// The checker refuses both.
fn lower_padded_copy<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    src: Value<'c, 'c>,
    src_str: &str,
    target: PadTarget<'_>,
) -> Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError> {
    let PadTarget {
        dims: tgt_dims,
        each_axis,
    } = target;
    let (src_sizes, elem) = tensor_sizes(gen, block, src, src_str)?;
    let dims: Vec<String> = tgt_dims
        .iter()
        .map(std::string::ToString::to_string)
        .collect();
    let tgt_str = format!("memref<{}x{elem}>", dims.join("x"));
    let tgt_ty = Type::parse(gen.context, &tgt_str).ok_or(LowerError::ParseType(tgt_str))?;
    let alloc = OperationBuilder::new("memref.alloc", gen.loc())
        .add_attributes(&[(
            Identifier::new(gen.context, "operandSegmentSizes"),
            DenseI32ArrayAttribute::new(gen.context, &[0, 0]).into(),
        )])
        .add_results(&[tgt_ty])
        .build()?;
    let dst: Value<'_, '_> = block.append_operation(alloc).result(0)?.into();
    if each_axis {
        assert_eq!(
            src_sizes.len(),
            tgt_dims.len(),
            "the checker keeps pad's number of axes"
        );
        let mut strides = vec![1i64; tgt_dims.len()];
        for i in (0..tgt_dims.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * tgt_dims[i + 1];
        }
        let shape = ViewShape {
            sizes: &src_sizes,
            strides: &strides,
        };
        let (corner, _) = reinterpret_with_sizes(gen, block, dst, shape, &elem)?;
        emit_zero_fill(gen, block, dst, &elem)?;
        let copy = OperationBuilder::new("memref.copy", gen.loc())
            .add_operands(&[src, corner])
            .build()?;
        block.append_operation(copy);
        return Ok((dst, tgt_ty, block));
    }
    let n = element_count(gen, block, &src_sizes)?;
    if let Size::Known(n) = n {
        assert!(
            n <= tgt_dims.iter().product(),
            "the checker refuses PadMode::Pad to a smaller shape"
        );
    }
    let flat = ViewShape {
        sizes: &[n],
        strides: &[1],
    };
    let (src_flat, _) = reinterpret_with_sizes(gen, block, src, flat, &elem)?;
    let (dst_prefix, _) = reinterpret_with_sizes(gen, block, dst, flat, &elem)?;
    emit_zero_fill(gen, block, dst, &elem)?;
    let copy = OperationBuilder::new("memref.copy", gen.loc())
        .add_operands(&[src_flat, dst_prefix])
        .build()?;
    block.append_operation(copy);
    Ok((dst, tgt_ty, block))
}

/// The run-time size of each dimension that a tensor's type writes as `?`. `parts` holds the
/// dimensions of the type of `src`, in order.
///
/// # Errors
///
/// Returns an error if MLIR refuses an operation built here.
fn run_time_sizes<'c>(
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    src: Value<'c, 'c>,
    parts: &[&str],
) -> Result<Vec<Value<'c, 'c>>, LowerError> {
    let index_ty = Type::index(gen.context);
    let mut sizes = Vec::new();
    for (k, d) in parts.iter().enumerate() {
        if *d == "?" {
            let attr = IntegerAttribute::new(index_ty, count_i64(k)).into();
            let c = OperationBuilder::new("arith.constant", gen.loc())
                .add_results(&[index_ty])
                .add_attributes(&[(Identifier::new(gen.context, "value"), attr)])
                .build()?;
            let c: Value<'c, 'c> = block.append_operation(c).result(0)?.into();
            let dim = OperationBuilder::new("memref.dim", gen.loc())
                .add_operands(&[src, c])
                .add_results(&[index_ty])
                .build()?;
            sizes.push(block.append_operation(dim).result(0)?.into());
        }
    }
    Ok(sizes)
}

/// `t.clone()`: a new buffer of `t`'s shape and a `memref.copy` into it. The copy reads `t`
/// through its own layout, so cloning a row copies that row.
///
/// # Errors
///
/// Returns an error if the source cannot be lowered or is not a tensor, or if MLIR refuses an
/// operation.
fn lower_clone_call<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    src_expr: &Expr,
) -> Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError> {
    // A returned clone copies into the caller's buffer; taken before the source is lowered, so
    // nothing inside it claims the buffer.
    let nrvo_slot = gen.nrvo_slot.take();
    let (src, src_ty, block) = gen.generate_expr(src_expr, block)?;
    let src_str = src_ty.to_string();
    // "memref<2x?xf32, strided<..>>" -> sizes ["2", "?"] and element "f32".
    let inner = src_str
        .strip_prefix("memref<")
        .and_then(|s| s.split([',', '>']).next())
        .ok_or_else(|| format!("clone of something that is not a tensor: {src_str}"))?;
    let mut parts: Vec<&str> = inner.split('x').collect();
    let elem = parts
        .pop()
        .ok_or_else(|| format!("clone of a tensor with no element type: {src_str}"))?;
    let sizes = run_time_sizes(gen, block, src, &parts)?;
    let dst_ty = Type::parse(gen.context, &format!("memref<{}x{elem}>", parts.join("x")))
        .ok_or_else(|| format!("clone of a tensor with no memref type: {src_str}"))?;
    if let Some(slot) = nrvo_slot.filter(|s| s.r#type() == dst_ty) {
        let copy = OperationBuilder::new("memref.copy", gen.loc())
            .add_operands(&[src, slot])
            .build()?;
        block.append_operation(copy);
        return Ok((slot, dst_ty, block));
    }
    let alloc = OperationBuilder::new("memref.alloc", gen.loc())
        .add_operands(&sizes)
        .add_attributes(&[(
            Identifier::new(gen.context, "operandSegmentSizes"),
            DenseI32ArrayAttribute::new(gen.context, &[count_i32(sizes.len()), 0]).into(),
        )])
        .add_results(&[dst_ty])
        .build()?;
    let dst: Value<'_, '_> = block.append_operation(alloc).result(0)?.into();
    let copy = OperationBuilder::new("memref.copy", gen.loc())
        .add_operands(&[src, dst])
        .build()?;
    block.append_operation(copy);
    Ok((dst, dst_ty, block))
}

/// The type a struct field is stored as: a tensor's descriptor, or the field's own type.
///
/// # Errors
///
/// Returns an error if a tensor's descriptor type does not parse.
fn field_storage_ty<'c>(
    gen: &MeliorGenerator<'c>,
    ty: &syntax::Type,
    field_ty: Type<'c>,
) -> Result<Type<'c>, LowerError> {
    match crate::codegen::generator::tensor_field_rank(ty) {
        Some(rank) => {
            let desc = crate::codegen::generator::tensor_descriptor_str(rank);
            Ok(Type::parse(gen.context, &desc).ok_or(LowerError::ParseType(desc))?)
        }
        None => Ok(field_ty),
    }
}

/// A tensor going into a struct field: a row is rebased to a plain memref first, so its
/// offset is in the data pointer, and it is stored as its descriptor.
///
/// # Errors
///
/// Returns an error if the tensor is a view that cannot be stored in a field, if a type does not
/// parse, or if MLIR refuses an operation.
#[allow(
    clippy::too_many_arguments,
    clippy::trivially_copy_pass_by_ref,
    reason = "callers are in other files; they pass the block by reference and each value on its own"
)]
pub(crate) fn tensor_into_field<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: &melior::ir::BlockRef<'c, 'c>,
    val: Value<'c, 'c>,
    val_ty: Type<'c>,
    field_ty: Type<'c>,
    rank: usize,
) -> Result<Value<'c, 'c>, LowerError> {
    let mut val = val;
    let mut val_ty = val_ty;
    let val_str = val_ty.to_string();
    if val_str.contains("strided<") {
        let plain = contiguous_view_as_plain(&val_str)
            .ok_or_else(|| LowerError::from(format!("cannot store a {val_str} view in a field")))?;
        let plain_ty = Type::parse(gen.context, &plain).ok_or(LowerError::ParseType(plain))?;
        val = move_view_pointer(gen, *block, val, val_ty, plain_ty)?;
        val_ty = plain_ty;
    }
    if val_ty != field_ty {
        val = gen.coerce_type(block, val, val_ty, field_ty)?;
    }
    let desc = crate::codegen::generator::tensor_descriptor_str(rank);
    let desc_ty = Type::parse(gen.context, &desc).ok_or(LowerError::ParseType(desc))?;
    let cast = OperationBuilder::new("builtin.unrealized_conversion_cast", gen.loc())
        .add_operands(&[val])
        .add_results(&[desc_ty])
        .build()?;
    Ok(block.append_operation(cast).result(0)?.into())
}

/// A struct field read back: a tensor's descriptor becomes the memref again.
///
/// # Errors
///
/// Returns an error if MLIR refuses an operation built here.
fn tensor_from_field<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    val: Value<'c, 'c>,
    stored_ty: Type<'c>,
    field_ty: Type<'c>,
) -> Result<Value<'c, 'c>, LowerError> {
    if stored_ty == field_ty {
        return Ok(val);
    }
    let cast = OperationBuilder::new("builtin.unrealized_conversion_cast", gen.loc())
        .add_operands(&[val])
        .add_results(&[field_ty])
        .build()?;
    Ok(block.append_operation(cast).result(0)?.into())
}

/// The body of a tensor negation: one element negated. A float uses `arith.negf`; an integer
/// is `0 - x`.
///
/// # Errors
///
/// Returns an error if the element type `elem` does not parse, or if MLIR refuses an operation.
fn negate_region<'c>(gen: &MeliorGenerator<'c>, elem: &str) -> Result<Region<'c>, LowerError> {
    let el_ty =
        Type::parse(gen.context, elem).ok_or_else(|| format!("negation of a tensor of {elem}"))?;
    let body = melior::ir::Block::new(&[(el_ty, gen.loc()), (el_ty, gen.loc())]);
    let x: Value<'_, '_> = body.argument(0)?.into();
    let negated: Value<'_, '_> = if elem.starts_with('f') || elem == "bf16" {
        let op = OperationBuilder::new("arith.negf", gen.loc())
            .add_operands(&[x])
            .add_results(&[el_ty])
            .build()?;
        body.append_operation(op).result(0)?.into()
    } else {
        let zero = OperationBuilder::new("arith.constant", gen.loc())
            .add_results(&[el_ty])
            .add_attributes(&[(
                Identifier::new(gen.context, "value"),
                IntegerAttribute::new(el_ty, 0).into(),
            )])
            .build()?;
        let zero: Value<'_, '_> = body.append_operation(zero).result(0)?.into();
        let op = OperationBuilder::new("arith.subi", gen.loc())
            .add_operands(&[zero, x])
            .add_results(&[el_ty])
            .build()?;
        body.append_operation(op).result(0)?.into()
    };
    body.append_operation(
        OperationBuilder::new("linalg.yield", gen.loc())
            .add_operands(&[negated])
            .build()?,
    );
    let region = Region::new();
    region.append_block(body);
    Ok(region)
}

/// `-t` on a tensor: a new tensor of each element negated, by a `linalg.generic`. A float uses
/// `arith.negf`, which keeps the sign of zero right; an integer is `0 - x`.
///
/// # Errors
///
/// Returns an error if `src_ty` is not a tensor's memref type, if a type or an attribute does
/// not parse, or if MLIR refuses an operation.
fn lower_tensor_negate<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    src: Value<'c, 'c>,
    src_ty: Type<'c>,
) -> Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError> {
    let src_str = src_ty.to_string();
    let inner = src_str
        .strip_prefix("memref<")
        .and_then(|s| s.split([',', '>']).next())
        .ok_or_else(|| format!("negation of something that is not a tensor: {src_str}"))?;
    let mut parts: Vec<&str> = inner.split('x').collect();
    let elem = parts
        .pop()
        .ok_or_else(|| format!("negation of a tensor with no element type: {src_str}"))?;
    let rank = parts.len();
    let sizes = run_time_sizes(gen, block, src, &parts)?;
    let out_ty = Type::parse(gen.context, &format!("memref<{}x{elem}>", parts.join("x")))
        .ok_or_else(|| format!("negation of a tensor with no memref type: {src_str}"))?;
    let alloc = OperationBuilder::new("memref.alloc", gen.loc())
        .add_operands(&sizes)
        .add_attributes(&[(
            Identifier::new(gen.context, "operandSegmentSizes"),
            DenseI32ArrayAttribute::new(gen.context, &[count_i32(sizes.len()), 0]).into(),
        )])
        .add_results(&[out_ty])
        .build()?;
    let out: Value<'_, '_> = block.append_operation(alloc).result(0)?.into();
    let region = negate_region(gen, elem)?;

    let dims: Vec<String> = (0..rank).map(|k| format!("d{k}")).collect();
    let map = Attribute::parse(
        gen.context,
        &format!("affine_map<({0}) -> ({0})>", dims.join(", ")),
    )
    .ok_or_else(|| "negation: an affine map that does not parse".to_string())?;
    let parallel = Attribute::parse(gen.context, "#linalg.iterator_type<parallel>")
        .ok_or_else(|| "negation: an iterator type that does not parse".to_string())?;
    let generic = OperationBuilder::new("linalg.generic", gen.loc())
        .add_operands(&[src, out])
        .add_attributes(&[
            (
                Identifier::new(gen.context, "operandSegmentSizes"),
                DenseI32ArrayAttribute::new(gen.context, &[1, 1]).into(),
            ),
            (
                Identifier::new(gen.context, "indexing_maps"),
                ArrayAttribute::new(gen.context, &[map, map]).into(),
            ),
            (
                Identifier::new(gen.context, "iterator_types"),
                ArrayAttribute::new(gen.context, &vec![parallel; rank]).into(),
            ),
        ])
        .add_regions([region])
        .build()?;
    block.append_operation(generic);
    Ok((out, out_ty, block))
}

/// The MLIR type of a function taking `args` and returning `ret`. A function returning `void`
/// lowers with no results, so its type must have none either: `() -> none` is a different type
/// from `() -> ()`, and `func.call_indirect` refuses a callee whose type does not match.
fn function_type<'c>(gen: &MeliorGenerator<'c>, args: &[Type<'c>], ret: Type<'c>) -> Type<'c> {
    let results: &[Type<'c>] = if ret == gen.none_ty { &[] } else { &[ret] };
    melior::ir::r#type::FunctionType::new(gen.context, args, results).into()
}
