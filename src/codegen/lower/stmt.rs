use super::{BlockLike, LowerError, LowerToMelior, MeliorGenerator, MeliorOpInfo, Value};
use crate::syntax;
use crate::syntax::{
    AssignStmt, BinaryOp, BinaryOpExpr, CompoundAssignStmt, Expr, ExprStmtStmt, IdentifierExpr,
    LetDeclStmt, MemberAccessExpr, ReturnStmt, Span,
};
use melior::ir::{
    attribute::{DenseI32ArrayAttribute, IntegerAttribute, TypeAttribute},
    operation::OperationBuilder,
    Identifier, Type,
};

/// The block that code is being added to.
type Block<'c> = melior::ir::BlockRef<'c, 'c>;

/// A lowered value together with its MLIR type.
type Typed<'c> = (Value<'c, 'c>, Type<'c>);

/// One field of a struct that is being assigned, and the value to store in it.
struct FieldStore<'c> {
    /// The field's position in the struct declaration.
    index: usize,
    /// The value to store, already converted to the field's type.
    val: Value<'c, 'c>,
}

impl<'c> LowerToMelior<'c> for ReturnStmt {
    type Output = Result<Option<melior::ir::BlockRef<'c, 'c>>, LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let ReturnStmt { expr, span: _ } = self;
        // A bare `return;` -- only a `void` function may write one, which the checker enforces.
        // Nothing to compute and nothing to convert, so emit the terminator and stop.
        let Some(expr) = expr else {
            let block = gen.run_value_drops(block)?;
            block.append_operation(OperationBuilder::new("func.return", gen.loc()).build()?);
            return Ok(None);
        };
        gen.expected_type = gen.current_return_type;
        // Offer the caller's buffer to the expression being returned, so a tensor operator builds
        // the answer there instead of in a buffer of its own that we then copy out of. Only an
        // operator is offered it: it takes the slot before lowering its own operands, which is
        // what keeps a nested operator from claiming it. A returned `.clone()` copies straight in.
        let is_clone = |e: &Expr| match e {
            Expr::MethodCall(mc) => mc.method_name.as_ref() == "clone" && mc.args.is_empty(),
            Expr::FunctionCall(fc) => fc.name.as_ref() == "clone" && fc.args.len() == 1,
            _ => false,
        };
        if matches!(expr, Expr::BinaryOp(_)) || is_clone(expr) {
            gen.nrvo_slot = gen.current_return_slot.filter(|_| !gen.in_spawn);
        }
        let (val, expr_ty, block) = gen.generate_expr(expr, block)?;
        gen.nrvo_slot = None;
        gen.expected_type = None;
        // `return f();` in a function returning `void`, such as `apply<T>` with `T = void`:
        // the call has run, and there is no value to return.
        if !gen.in_spawn && gen.current_return_type == Some(gen.none_ty) {
            let block = gen.run_value_drops(block)?;
            block.append_operation(OperationBuilder::new("func.return", gen.loc()).build()?);
            gen.has_returned = true;
            return Ok(None);
        }
        let val = convert_returned_value(gen, block, val, expr_ty)?;
        emit_return(gen, block, val)?;
        Ok(None)
    }
}

/// The returned value `val`, of type `expr_ty`, converted to the function's return type.
///
/// # Errors
/// When a type cannot be parsed, MLIR refuses a conversion operation, or the two types have no
/// conversion.
fn convert_returned_value<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: Block<'c>,
    val: Value<'c, 'c>,
    expr_ty: Type<'c>,
) -> Result<Value<'c, 'c>, LowerError> {
    let Some(ret_ty) = gen.current_return_type else {
        return Ok(val);
    };
    // A row of a tensor is a strided view of the tensor's memory, and the function returns a
    // tensor of its own. A cast cannot change the layout, so the row is copied: into the
    // caller's return slot, which the copy in `emit_return` does, or into a new buffer.
    let returns_a_view = gen.is_memref(&expr_ty)
        && expr_ty.to_string().contains("strided<")
        && !ret_ty.to_string().contains("strided<");
    if returns_a_view {
        if gen
            .current_return_slot
            .as_ref()
            .is_none_or(|_| gen.in_spawn)
        {
            return copy_into_new_buffer(gen, block, val, ret_ty);
        }
        return Ok(val);
    }
    if expr_ty == ret_ty {
        return Ok(val);
    }
    if gen.is_memref(&expr_ty) && gen.is_memref(&ret_ty) {
        cast_returned_memref(gen, block, val, expr_ty, ret_ty)
    } else if ret_ty.to_string() == "i32" && gen.is_memref(&expr_ty) {
        let zero_op = OperationBuilder::new("arith.constant", gen.loc())
            .add_results(&[ret_ty])
            .add_attributes(&[(
                Identifier::new(gen.context, "value"),
                IntegerAttribute::new(ret_ty, 0).into(),
            )])
            .build()?;
        Ok(block.append_operation(zero_op).result(0)?.into())
    } else {
        gen.coerce_type(&block, val, expr_ty, ret_ty)
    }
}

/// `val`, a buffer of type `expr_ty`, cast to `ret_ty`, the buffer type the function returns.
///
/// # Errors
/// When a type cannot be parsed or MLIR refuses a cast operation.
fn cast_returned_memref<'c>(
    gen: &MeliorGenerator<'c>,
    block: Block<'c>,
    mut val: Value<'c, 'c>,
    expr_ty: Type<'c>,
    ret_ty: Type<'c>,
) -> Result<Value<'c, 'c>, LowerError> {
    let expr_parts = expr_ty.to_string();
    let ret_parts = ret_ty.to_string();
    let space_differs = expr_parts.matches(',').count() != ret_parts.matches(',').count();
    // `memref.memory_space_cast` changes the space and nothing else, and
    // `memref.cast` cannot change the space at all. A return whose type differs in
    // both -- a placed tensor with a written shape, returned from a body that
    // allocated it dynamically -- needs one of each, shape first (Vx#429).
    if space_differs {
        if let Some(no_space) =
            crate::codegen::generator::strip_memref_space(&ret_parts).filter(|s| *s != expr_parts)
        {
            let shaped = Type::parse(gen.context, &no_space)
                .ok_or_else(|| LowerError::ParseType(no_space.clone()))?;
            let shape_op = OperationBuilder::new("memref.cast", gen.loc())
                .add_operands(&[val])
                .add_results(&[shaped])
                .build()?;
            val = block.append_operation(shape_op).result(0)?.into();
        }
    }
    let cast_op_name = if space_differs {
        "memref.memory_space_cast"
    } else {
        "memref.cast"
    };
    let cast_op = OperationBuilder::new(cast_op_name, gen.loc())
        .add_operands(&[val])
        .add_results(&[ret_ty])
        .build()?;
    Ok(block.append_operation(cast_op).result(0)?.into())
}

/// Emits the return of `val`: the drops that are waiting, then the terminator. When the caller
/// passed a buffer for the result, `val` is copied into it and nothing is returned.
///
/// # Errors
/// When a waiting drop cannot be emitted or MLIR refuses an operation.
fn emit_return<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: Block<'c>,
    val: Value<'c, 'c>,
) -> Result<(), LowerError> {
    let op_name = if gen.in_spawn {
        "vx.return"
    } else {
        "func.return"
    };
    // Returning through a buffer the caller allocated: copy the result in and return nothing.
    // A `vx.return` is a spawn region's yield rather than the function's, so it is untouched.
    if let Some(slot) = gen.current_return_slot.filter(|_| !gen.in_spawn) {
        if val != slot {
            let copy_op = OperationBuilder::new("memref.copy", gen.loc())
                .add_operands(&[val, slot])
                .build()?;
            block.append_operation(copy_op);
            // The copy moved the result into the caller's buffer. What it was copied from
            // is freed when this function owned it, and left alone when it is a view.
            let drop = OperationBuilder::new("vx.drop", gen.loc())
                .add_operands(&[val])
                .add_attributes(&[(
                    Identifier::new(gen.context, "owned_only"),
                    melior::ir::attribute::Attribute::unit(gen.context),
                )])
                .build()?;
            block.append_operation(drop);
        }
        let block = gen.run_value_drops(block)?;
        block.append_operation(OperationBuilder::new(op_name, gen.loc()).build()?);
        gen.has_returned = true;
        return Ok(());
    }
    let block = gen.run_value_drops(block)?;
    let ret_op = OperationBuilder::new(op_name, gen.loc())
        .add_operands(&[val])
        .build()?;
    block.append_operation(ret_op);
    gen.has_returned = true;
    Ok(())
}

impl<'c> LowerToMelior<'c> for LetDeclStmt {
    type Output = Result<Option<melior::ir::BlockRef<'c, 'c>>, LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let LetDeclStmt {
            name,
            is_mut: _,
            ty_ann,
            expr,
            span: _,
        } = self;
        gen.note_shadow(name.as_ref());
        // The initializer is expected to be the annotated type, or whatever it is. Not the
        // type the enclosing code expected: inside a statement `if`, that is "no value", and an
        // `if` initializer would then drop its value.
        let prev_expected = gen.expected_type;
        gen.expected_type = ty_ann.as_ref().and_then(|ann| gen.lower_type(ann).ok());
        let (val, ty, block) = gen.generate_expr(expr, block)?;
        if let Expr::Closure(c) = expr {
            let func_args = c.params.iter().map(|(_, t)| t.clone()).collect();
            let ret_ty = c
                .ret_ty
                .clone()
                .unwrap_or(syntax::Type::Scalar(syntax::ElementType::I32));
            gen.ast_env.insert(
                name.clone(),
                syntax::Type::Closure(func_args, Box::new(ret_ty)),
            );
        }
        gen.expected_type = prev_expected;

        let ast_ty = ty_ann.clone().or_else(|| gen.infer_ast_type(expr));
        if let Some(ref t) = ast_ty {
            gen.ast_env.insert(name.to_string().into(), t.clone());
        }
        if syntax::is_tensor_construction(expr) {
            gen.owned_tensors.insert(name.to_string().into());
        }

        // A local declared `*mut T` holds a pointer, but `&mut x` on a scalar local lowers to
        // `x`'s `memref` descriptor. Turn it into the raw pointer here, at the binding, so every
        // later read of the name sees the same `!llvm.ptr` that a pointer *parameter* carries.
        // Without this the descriptor reached each consumer instead, and each one got it wrong in
        // its own way: a call argument loaded the pointee, and `p[0] = v` stored through a rank-0
        // memref with an index.
        let (val, ty) =
            if matches!(ast_ty, Some(syntax::Type::Pointer(..))) && gen.is_scalar_memref(&ty) {
                let ptr_ty = gen.ptr_ty;
                (gen.coerce_type(&block, val, ty, ptr_ty)?, ptr_ty)
            } else {
                (val, ty)
            };

        // A loop variable is an `index` in MLIR and an integer in Vx. A local made from one
        // holds the integer -- the declared type, or `i32` -- so `&mut` of it is a pointer to
        // that integer, as the function it is passed to expects.
        let (val, ty) = if ty == gen.index_ty {
            index_to_integer(gen, block, val, ty_ann.as_ref())?
        } else {
            (val, ty)
        };

        // `let t : Tensor<f32, []> = 1.0`: a rank-0 tensor is a buffer holding one value, so the
        // scalar the initializer produced is stored into one.
        let (val, ty) = match ty_ann {
            Some(syntax::Type::Tensor(_, dims, _)) if dims.is_empty() && !gen.is_memref(&ty) => {
                let buffer = scalar_into_new_buffer(gen, block, val, ty)?;
                gen.owned_tensors.insert(name.to_string().into());
                buffer
            }
            _ => (val, ty),
        };

        bind_local(gen, block, self, val, ty)?;
        Ok(Some(block))
    }
}

/// The loop index `val` as an integer of the declared type `ty_ann`, or `i32` when the local
/// has no annotation.
///
/// # Errors
/// When the annotation has no MLIR type or MLIR refuses the cast.
fn index_to_integer<'c>(
    gen: &MeliorGenerator<'c>,
    block: Block<'c>,
    val: Value<'c, 'c>,
    ty_ann: Option<&syntax::Type>,
) -> Result<Typed<'c>, LowerError> {
    let int_ty = match ty_ann {
        Some(ann) => gen.lower_type(ann)?,
        None => gen.i32_ty,
    };
    let cast = OperationBuilder::new("arith.index_cast", gen.loc())
        .add_operands(&[val])
        .add_results(&[int_ty])
        .build()?;
    Ok((block.append_operation(cast).result(0)?.into(), int_ty))
}

/// A new buffer that holds the one scalar `val`, of type `ty`. Returns the buffer and its type.
///
/// # Errors
/// When the buffer type cannot be parsed or MLIR refuses an operation.
fn scalar_into_new_buffer<'c>(
    gen: &MeliorGenerator<'c>,
    block: Block<'c>,
    val: Value<'c, 'c>,
    ty: Type<'c>,
) -> Result<Typed<'c>, LowerError> {
    let buffer_ty = Type::parse(gen.context, &format!("memref<{ty}>"))
        .ok_or_else(|| crate::codegen::lower::LowerError::ParseType(format!("memref<{ty}>")))?;
    let alloc = OperationBuilder::new("memref.alloc", gen.loc())
        .add_attributes(&[(
            Identifier::new(gen.context, "operandSegmentSizes"),
            DenseI32ArrayAttribute::new(gen.context, &[0, 0]).into(),
        )])
        .add_results(&[buffer_ty])
        .build()?;
    let buffer: Value<'_, '_> = block.append_operation(alloc).result(0)?.into();
    let store = OperationBuilder::new("memref.store", gen.loc())
        .add_operands(&[val, buffer])
        .build()?;
    block.append_operation(store);
    Ok((buffer, buffer_ty))
}

/// Binds the name of the local `decl` to `val`: in a stack slot when the local is `mut` or has
/// its address taken, and as the plain value otherwise.
///
/// # Errors
/// When a slot type cannot be parsed or MLIR refuses an operation.
fn bind_local<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: Block<'c>,
    decl: &LetDeclStmt,
    val: Value<'c, 'c>,
    ty: Type<'c>,
) -> Result<(), LowerError> {
    let name = &decl.name;
    let is_mut = decl.is_mut;
    // A borrowed value needs one address for all its borrows, so it lives in a slot as a
    // `let mut` does. Without one, each `&x` copied `x` to a new place.
    let borrowed_value =
        gen.address_taken.contains(name) && !gen.is_memref(&ty) && ty != gen.ptr_ty;
    if is_mut || borrowed_value {
        let ty_str = ty.to_string();
        // A borrowed value is never assigned, so a plain slot will do, and its address is
        // the bare pointer an `&T` parameter takes.
        if ty_str.contains("!llvm.struct") || ty_str.contains("!llvm.ptr") || !is_mut {
            bind_in_llvm_slot(gen, block, name, val, ty)?;
        } else {
            bind_in_memref_slot(gen, block, name, val, ty)?;
        }
    } else {
        gen.env.insert(name.to_string().into(), (val, ty));
        // A value, not a stack slot, even if an outer `let mut` of this name had one.
        gen.allocs.remove(name.as_ref());
    }
    Ok(())
}

/// Stores `val` in a new `llvm.alloca` slot and binds `name` to the slot's pointer.
///
/// # Errors
/// When MLIR refuses an operation.
fn bind_in_llvm_slot<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: Block<'c>,
    name: &crate::symbol::Symbol,
    val: Value<'c, 'c>,
    ty: Type<'c>,
) -> Result<(), LowerError> {
    let ptr_ty = gen.ptr_ty;
    let i32_ty = gen.i32_ty;
    let one_attr = IntegerAttribute::new(i32_ty, 1).into();
    let const_op = OperationBuilder::new("llvm.mlir.constant", gen.loc())
        .add_results(&[i32_ty])
        .add_attributes(&[(Identifier::new(gen.context, "value"), one_attr)])
        .build()?;
    let one_val = block.append_operation(const_op).result(0)?.into();

    let alloca_op = OperationBuilder::new("llvm.alloca", gen.loc())
        .add_operands(&[one_val])
        .add_results(&[ptr_ty])
        .add_attributes(&[(
            Identifier::new(gen.context, "elem_type"),
            TypeAttribute::new(ty).into(),
        )])
        .build()?;
    let alloca_ref = block.append_operation(alloca_op);
    let alloca_val = alloca_ref.result(0)?.into();

    let store_op = OperationBuilder::new("llvm.store", gen.loc())
        .add_operands(&[val, alloca_val])
        .build()?;
    block.append_operation(store_op);

    gen.env.insert(name.to_string().into(), (alloca_val, ty));
    gen.allocs.insert(name.to_string());
    Ok(())
}

/// Stores `val` in a new `memref.alloca` slot and binds `name` to the slot.
///
/// # Errors
/// When the slot type cannot be parsed or MLIR refuses an operation.
fn bind_in_memref_slot<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: Block<'c>,
    name: &crate::symbol::Symbol,
    val: Value<'c, 'c>,
    ty: Type<'c>,
) -> Result<(), LowerError> {
    let memref_ty = format!("memref<{ty}>");
    let parsed_memref_ty = Type::parse(gen.context, &memref_ty).ok_or_else(|| {
        crate::codegen::lower::LowerError::ParseType("Type::parse failed".to_string())
    })?;
    let alloca_op = OperationBuilder::new("memref.alloca", gen.loc())
        .add_results(&[parsed_memref_ty])
        .build()?;
    let alloca_ref = block.append_operation(alloca_op);
    let alloca_val = alloca_ref.result(0)?.into();

    let store_op = OperationBuilder::new("memref.store", gen.loc())
        .add_operands(&[val, alloca_val])
        .build()?;
    block.append_operation(store_op);
    gen.env
        .insert(name.to_string().into(), (alloca_val, parsed_memref_ty));
    gen.allocs.insert(name.to_string());
    Ok(())
}

/// A new buffer of type `ty` holding a copy of `val`, a view with the same extents in another
/// layout. A `?` extent of `ty` is read off `val`.
///
/// # Errors
/// When `ty` is not a `memref` type or MLIR refuses an operation.
fn copy_into_new_buffer<'c>(
    gen: &MeliorGenerator<'c>,
    block: Block<'c>,
    val: Value<'c, 'c>,
    ty: Type<'c>,
) -> Result<Value<'c, 'c>, LowerError> {
    let ty_str = ty.to_string();
    let extents = ty_str
        .strip_prefix("memref<")
        .ok_or_else(|| LowerError::from(format!("not a memref: {ty_str}")))?
        .split('x');
    let mut dynamic = Vec::new();
    let mut dynamic_count: i32 = 0;
    for (axis, extent) in (0_i64..).zip(extents) {
        if extent != "?" {
            continue;
        }
        let axis_val = block
            .append_operation(
                OperationBuilder::new("arith.constant", gen.loc())
                    .add_results(&[gen.index_ty])
                    .add_attributes(&[(
                        Identifier::new(gen.context, "value"),
                        IntegerAttribute::new(gen.index_ty, axis).into(),
                    )])
                    .build()?,
            )
            .result(0)?
            .into();
        let dim = block
            .append_operation(
                OperationBuilder::new("memref.dim", gen.loc())
                    .add_operands(&[val, axis_val])
                    .add_results(&[gen.index_ty])
                    .build()?,
            )
            .result(0)?
            .into();
        dynamic.push(dim);
        dynamic_count += 1;
    }
    let buffer = block
        .append_operation(
            OperationBuilder::new("memref.alloc", gen.loc())
                .add_operands(&dynamic)
                .add_attributes(&[(
                    Identifier::new(gen.context, "operandSegmentSizes"),
                    melior::ir::attribute::DenseI32ArrayAttribute::new(
                        gen.context,
                        &[dynamic_count, 0],
                    )
                    .into(),
                )])
                .add_results(&[ty])
                .build()?,
        )
        .result(0)?
        .into();
    block.append_operation(
        OperationBuilder::new("memref.copy", gen.loc())
            .add_operands(&[val, buffer])
            .build()?,
    );
    Ok(buffer)
}

/// The nominal struct/enum name a type ultimately names, peeling reference/pointer wrappers and
/// generic instantiation. `&Vec<i32>` -> `Vec`. Used to recover a struct name for field assignment
/// when the base lowers to a bare `!llvm.ptr` (which carries no struct identity) and sema did not
/// stamp `struct_name` on the assignment target (#205).
fn nominal_struct_name(ty: &syntax::Type) -> Option<crate::symbol::Symbol> {
    use syntax::Type;
    match ty {
        Type::Struct(n, _) | Type::Enum(n, _) => Some(n.clone()),
        Type::GenericInstance(base, _) => nominal_struct_name(base),
        Type::Ref(inner, _)
        | Type::Pointer(inner, _, _)
        | Type::Verified(inner)
        | Type::Pinned(inner, _)
        | Type::Borrow { inner, .. } => nominal_struct_name(inner),
        _ => None,
    }
}

/// For `place.member = rhs` where `place` is itself a field (`a.b`), a struct literal that
/// copies every field of `place` but `member`, which is `rhs`. `None` when `place` is not a
/// field of a variable path, or is a generic struct.
fn struct_with_one_field_replaced(
    gen: &MeliorGenerator<'_>,
    place: &Expr,
    member: &crate::symbol::Symbol,
    rhs: &Expr,
) -> Option<Expr> {
    let Expr::MemberAccess(_) = place else {
        return None;
    };
    // `place` is read once per field, so it must be a plain path such as `a.b.c`.
    let mut root = place;
    while let Expr::MemberAccess(m) = root {
        root = &m.base;
    }
    if !matches!(root, Expr::Identifier(_)) {
        return None;
    }
    let syntax::Type::Struct(name, _) = gen.infer_ast_type(place)? else {
        return None;
    };
    let decl = gen.structs.get(&name)?;
    if !decl.generics.is_empty() {
        return None;
    }
    let fields = decl
        .fields
        .iter()
        .map(|(field, _)| {
            let value = if field == member {
                rhs.clone()
            } else {
                Expr::MemberAccess(MemberAccessExpr {
                    base: Box::new(place.clone()),
                    member: field.clone(),
                    struct_name: Some(name.clone()),
                    span: place.span(),
                })
            };
            (field.clone(), value)
        })
        .collect();
    Some(Expr::StructInit(syntax::StructInitExpr::new(
        name,
        fields,
        place.span(),
    )))
}

/// Substitution `struct generic param -> concrete type arg`, parsed from a monomorphized struct name
/// like `Vec<i32>` and the struct's declared generics. Mirrors the read-path logic so a generic
/// field type (`data: *mut T`) lowers correctly on the *assignment* path (#205), instead of a raw
/// `T` reaching codegen.
fn generic_arg_mapping(
    resolved_struct_name: &str,
    generics: &[syntax::GenericParam],
) -> std::collections::HashMap<crate::symbol::Symbol, syntax::Type> {
    use syntax::Type;
    let mut mapping = std::collections::HashMap::new();
    let (Some(lt), true) = (
        resolved_struct_name.find('<'),
        resolved_struct_name.ends_with('>'),
    ) else {
        return mapping;
    };
    let inner = &resolved_struct_name[lt + 1..resolved_struct_name.len() - 1];
    let inner_tys: Vec<Type> = crate::syntax::split_type_args(inner)
        .into_iter()
        .map(crate::codegen::lower::parse_type_arg)
        .collect();
    for (i, param) in generics.iter().enumerate() {
        if let Some(ty) = inner_tys.get(i) {
            mapping.insert(param.name().into(), ty.clone());
        }
    }
    mapping
}

/// The LLVM type of struct `base` instantiated by `mapping`, under the name every other use of
/// that instance gives it. Spelled with the bare `base`, two instances of one generic struct
/// were two bodies under one name, which MLIR refuses.
///
/// # Errors
/// When `mapping` has no type argument for one of the struct's generic parameters, or the
/// struct has no MLIR type.
pub(crate) fn struct_instance_ty<'c>(
    gen: &mut MeliorGenerator<'c>,
    base: &crate::symbol::Symbol,
    decl: &syntax::StructDecl,
    mapping: &std::collections::HashMap<crate::symbol::Symbol, syntax::Type>,
) -> Result<melior::ir::Type<'c>, crate::codegen::lower::LowerError> {
    let nominal = syntax::Type::Struct(base.clone(), None);
    if decl.generics.is_empty() {
        return gen.lower_type(&nominal);
    }
    let args = decl
        .generics
        .iter()
        .map(|g| {
            mapping.get(g.name()).cloned().ok_or_else(|| {
                crate::codegen::lower::LowerError::from(format!(
                    "no type argument for `{}` of `{}`",
                    g.name(),
                    base
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    gen.lower_type(&syntax::Type::GenericInstance(Box::new(nominal), args))
}

/// Do `dst = left @ right`'s three tensors have statically agreeing rank-2 shapes?
///
/// Only then is the destination known to be the size of the product, which is what makes
/// filling it in place safe. A dynamic operand has no product shape for the checker to have
/// compared `dst` against, and keeps the allocate-and-publish form.
fn matmul_assign_shapes_agree(
    gen: &MeliorGenerator<'_>,
    dst: &Expr,
    left: &Expr,
    right: &Expr,
) -> bool {
    // Filling `dst` in place zeroes it before the multiply reads its inputs, so an operand
    // naming the same buffer would read zeros. Each of the three has to be a local declared a
    // tensor -- a name bound to a borrow denotes whatever it points at, and following that is
    // the analysis this check exists to avoid.
    // A destination whose pointer is taken can be read under another name, through a view.
    let root = |e: &Expr| -> Option<crate::symbol::Symbol> {
        let r = syntax::matmul_operand_root(e)?;
        let named = Expr::Identifier(IdentifierExpr {
            name: r.clone(),
            span: Span::default(),
        });
        // A name declared a borrow or a pointer denotes whatever it points at, which is the
        // alias the destination's own name does not reveal.
        match gen.infer_ast_type(&named) {
            Some(syntax::Type::Borrow { .. } | syntax::Type::Pointer(..)) => None,
            _ => Some(r.clone()),
        }
    };
    let (Some(dst_root), Some(left_root), Some(right_root)) = (root(dst), root(left), root(right))
    else {
        return false;
    };
    if !gen.owned_tensors.contains(&dst_root)
        || gen.pointer_taken.contains(&dst_root)
        || dst_root == left_root
        || dst_root == right_root
    {
        return false;
    }

    let dims = |e: &Expr| -> Option<Vec<i64>> {
        let e = match e {
            Expr::Borrow(borrow) => borrow.expr.as_ref(),
            other => other,
        };
        match gen.infer_ast_type(e)? {
            syntax::Type::Tensor(_, dims, _) => dims
                .iter()
                .map(|d| d.literal().and_then(|v| v.parse::<i64>().ok()))
                .collect(),
            _ => None,
        }
    };
    let (Some(dst_dims), Some(left_dims), Some(right_dims)) = (dims(dst), dims(left), dims(right))
    else {
        return false;
    };
    dst_dims.len() == 2
        && left_dims.len() == 2
        && right_dims.len() == 2
        && left_dims[1] == right_dims[0]
        && dst_dims[0] == left_dims[0]
        && dst_dims[1] == right_dims[1]
}

impl<'c> LowerToMelior<'c> for AssignStmt {
    type Output = Result<Option<melior::ir::BlockRef<'c, 'c>>, LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let AssignStmt { lhs, rhs, span } = self;

        // `a.b.c = v` is lowered as `a.b = B { c: v, d: a.b.d, ... }`. The store below handles
        // only a field of a variable, so each level of nesting peels off one field.
        if let Expr::MemberAccess(m) = lhs {
            if let Some(rebuilt) = struct_with_one_field_replaced(gen, &m.base, &m.member, rhs) {
                let assign = AssignStmt {
                    lhs: (*m.base).clone(),
                    rhs: rebuilt,
                    span: *span,
                };
                return assign.lower(gen, block);
            }
        }

        // Slice/row initializer: `o[i] = [a, b, c, d]`. Lower the LHS partial index to its S1 row
        // view and store each element. This is cold setup code, so scalar stores are fine.
        if let (Expr::IndexAccess(_), Expr::Array(arr)) = (lhs, rhs) {
            if let Some(done) = store_row_initializer(gen, block, lhs, arr)? {
                return Ok(Some(done));
            }
        }

        // A row copied into a row: `p[0] = q[1]`. Both sides are views, and the elements are
        // copied from one into the other.
        // Only a tensor variable or a row on the right: an expression such as `a[i] + b[i]` is
        // lowered further down, into a store of its result.
        if lhs_is_a_row(gen, lhs) && matches!(rhs, Expr::Identifier(_) | Expr::IndexAccess(_)) {
            return copy_row_into_row(gen, block, lhs, rhs).map(Some);
        }

        // `c = a @ b` where `c` already names a buffer: fill it, rather than allocating a second
        // one and rebinding `c` to it. The checker has compared `c`'s declared shape against the
        // product, so the destination is the right size (Vx#391).
        if let Some(done) = matmul_into_destination(gen, block, lhs, rhs)? {
            return Ok(Some(done));
        }

        let expected_ty = assignment_expected_type(gen, lhs)?;
        let prev_expected = gen.expected_type;
        if expected_ty.is_some() {
            gen.expected_type = expected_ty;
        }
        let (rhs_val, rhs_ty, block) = gen.generate_expr(rhs, block)?;
        gen.expected_type = prev_expected;

        // Slice store (S3): `o[i] = <slice>` where the RHS is a vector<Dxf32> (from a slice
        // elementwise op). Lower the LHS partial index to its S1 row view (a reinterpret_cast'd
        // rank-1 memref aliasing `o`) and `vector.store` the result back into it.
        if rhs_ty.to_string().starts_with("vector<") {
            return store_vector_into_row(gen, block, lhs, rhs_val).map(Some);
        }

        // The old value's drops run now that the new value is computed, before it is stored.
        let block = if matches!(lhs, Expr::Identifier(_)) {
            gen.run_value_drops(block)?
        } else {
            block
        };
        match lhs {
            Expr::Identifier(IdentifierExpr { name, span: _ }) => {
                store_to_variable(gen, block, name, (rhs_val, rhs_ty))?;
                Ok(Some(block))
            }
            Expr::IndexAccess(access) => {
                let stored = store_to_element(gen, block, access, (rhs_val, rhs_ty))?;
                Ok(Some(stored.unwrap_or(block)))
            }
            // `*p = val`: store the RHS *through* the pointer. Without this case a dereference
            // assignment fell through to the no-op tail below and was **silently dropped** (`Box`'s
            // `*p = val` never wrote, so a read-back saw uninitialized memory). The store type comes
            // from the RHS value, so there's no pointee-type guesswork. (#242)
            Expr::Dereference(d) => {
                store_through_dereference(gen, block, d.expr.as_ref(), rhs_val).map(Some)
            }
            Expr::MemberAccess(access) => {
                store_to_field(gen, block, access, (rhs_val, rhs_ty)).map(Some)
            }
            _ => Ok(Some(block)),
        }
    }
}

/// Lowers `o[i] = [a, b, c, d]`: stores each element of the array literal into the row that
/// `lhs` names. Returns `None`, with nothing stored, when `lhs` does not lower to a buffer.
///
/// # Errors
/// When `lhs` or an element cannot be lowered, or MLIR refuses an operation.
fn store_row_initializer<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: Block<'c>,
    lhs: &Expr,
    arr: &syntax::ArrayExpr,
) -> Result<Option<Block<'c>>, LowerError> {
    let (row, row_ty, mut b) = gen.generate_expr(lhs, block)?;
    let row_ty_str = row_ty.to_string();
    if !row_ty_str.starts_with("memref<") {
        return Ok(None);
    }
    let elem_ty_str = row_ty_str
        .strip_prefix("memref<")
        .and_then(|s| s.split(',').next())
        .and_then(|s| s.rsplit('x').next())
        .unwrap_or("f32");
    let elem_ty = Type::parse(gen.context, elem_ty_str)
        .ok_or_else(|| crate::codegen::lower::LowerError::ParseType(elem_ty_str.to_string()))?;
    let index_ty = Type::index(gen.context);
    for (k, el) in (0_i64..).zip(arr.elements.iter()) {
        let (v, vty, nb) = gen.generate_expr(el, b)?;
        b = nb;
        let v = gen.coerce_type(&b, v, vty, elem_ty)?;
        let idx_op = OperationBuilder::new("arith.constant", gen.loc())
            .add_attributes(&[(
                Identifier::new(gen.context, "value"),
                IntegerAttribute::new(index_ty, k).into(),
            )])
            .add_results(&[index_ty])
            .build()?;
        let idx = b.append_operation(idx_op).result(0)?.into();
        let store = OperationBuilder::new("memref.store", gen.loc())
            .add_operands(&[v, row, idx])
            .build()?;
        b.append_operation(store);
    }
    Ok(Some(b))
}

/// Whether `lhs` indexes a tensor with fewer indices than the tensor has dimensions, so that it
/// names a row and not one element.
fn lhs_is_a_row(gen: &MeliorGenerator<'_>, lhs: &Expr) -> bool {
    let mut base = lhs;
    let mut indices = 0;
    while let Expr::IndexAccess(ix) = base {
        base = &ix.base;
        indices += 1;
    }
    indices > 0
        && matches!(
            gen.infer_ast_type(base),
            Some(syntax::Type::Tensor(_, dims, _)) if dims.len() > indices
        )
}

/// Lowers `p[0] = q[1]`: copies the elements of the row or tensor `rhs` into the row `lhs`.
///
/// # Errors
/// When a side cannot be lowered, or one of the two is not a view of a tensor's memory.
fn copy_row_into_row<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: Block<'c>,
    lhs: &Expr,
    rhs: &Expr,
) -> Result<Block<'c>, LowerError> {
    let (dst, dst_ty, b) = gen.generate_expr(lhs, block)?;
    let (src, mut src_ty, b) = gen.generate_expr(rhs, b)?;
    let src = super::expr::load_tensor_slot(gen, &b, src, &mut src_ty)?;
    let is_view = |t: &Type<'_>| {
        let t = t.to_string();
        t.starts_with("memref<") && !t.starts_with("memref<memref<")
    };
    if is_view(&dst_ty) && is_view(&src_ty) {
        b.append_operation(
            OperationBuilder::new("memref.copy", gen.loc())
                .add_operands(&[src, dst])
                .build()?,
        );
        return Ok(b);
    }
    Err(LowerError::from(format!(
        "a tensor assigned to an element of another: {src_ty} into {dst_ty}"
    )))
}

/// Lowers `lhs = a @ b` by filling the buffer `lhs` already names, when the three shapes are
/// known to agree. Returns `None`, with nothing lowered, for any other assignment.
///
/// # Errors
/// When an operand cannot be lowered or the multiply cannot be emitted.
fn matmul_into_destination<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: Block<'c>,
    lhs: &Expr,
    rhs: &Expr,
) -> Result<Option<Block<'c>>, LowerError> {
    let Expr::BinaryOp(BinaryOpExpr {
        lhs: ml,
        op: BinaryOp::MatMul,
        rhs: mr,
        ..
    }) = rhs
    else {
        return Ok(None);
    };
    if !matmul_assign_shapes_agree(gen, lhs, ml, mr) {
        return Ok(None);
    }
    fill_matmul_destination(gen, block, lhs, ml, mr).map(Some)
}

/// Lowers `dst = left @ right` by writing the product straight into the buffer `dst` names.
///
/// # Errors
/// When an operand cannot be lowered, `dst` is not a buffer, or the multiply cannot be emitted.
fn fill_matmul_destination<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: Block<'c>,
    dst: &Expr,
    left: &Expr,
    right: &Expr,
) -> Result<Block<'c>, LowerError> {
    let (dst_val, mut dst_ty, block) = gen.generate_expr(dst, block)?;
    let (a_val, mut a_ty, block) = gen.generate_expr(left, block)?;
    let (b_val, mut b_ty, block) = gen.generate_expr(right, block)?;
    let dst_val = super::expr::load_tensor_slot(gen, &block, dst_val, &mut dst_ty)?;
    let a_val = super::expr::load_tensor_slot(gen, &block, a_val, &mut a_ty)?;
    let b_val = super::expr::load_tensor_slot(gen, &block, b_val, &mut b_ty)?;
    let dst_ty_str = dst_ty.to_string();
    let el_ty_str = dst_ty_str
        .rsplit('x')
        .next()
        .and_then(|t| t.strip_suffix('>'))
        .ok_or_else(|| LowerError::from(format!("not a memref: {dst_ty_str}")))?
        .to_string();
    super::expr::emit_matmul_into(gen, &block, a_val, b_val, dst_val, &el_ty_str)?;
    Ok(block)
}

/// The MLIR type the right-hand side of an assignment to `lhs` is expected to have: what the
/// variable's slot holds, else the lowered form of the type `lhs` has in the source. `None`
/// when neither is known.
///
/// # Errors
/// When the type inside a variable's slot type cannot be parsed.
fn assignment_expected_type<'c>(
    gen: &MeliorGenerator<'c>,
    lhs: &Expr,
) -> Result<Option<Type<'c>>, LowerError> {
    let mut expected_ty = None;
    if let Expr::Identifier(IdentifierExpr { name, span: _ }) = lhs {
        if let Some((_, mem_ty)) = gen.env.get(name) {
            let mem_ty_str = mem_ty.to_string();
            if mem_ty_str.starts_with("memref<") {
                let inner_ty_str = &mem_ty_str[7..mem_ty_str.len() - 1];
                expected_ty = Some(Type::parse(gen.context, inner_ty_str).ok_or_else(|| {
                    crate::codegen::lower::LowerError::ParseType("Type::parse failed".to_string())
                })?);
            } else {
                expected_ty = Some(*mem_ty);
            }
        }
    }

    if expected_ty.is_none() {
        if let Some(ast_ty) = gen.infer_ast_type(lhs) {
            expected_ty = gen.lower_type(&ast_ty).ok();
        }
    }
    Ok(expected_ty)
}

/// Lowers `o[i] = <slice>` where the right-hand side is the vector `rhs_val`: stores the vector
/// into the row that `lhs` names.
///
/// # Errors
/// When `lhs` cannot be lowered or MLIR refuses an operation.
fn store_vector_into_row<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: Block<'c>,
    lhs: &Expr,
    rhs_val: Value<'c, 'c>,
) -> Result<Block<'c>, LowerError> {
    let (dst, _dst_ty, block) = gen.generate_expr(lhs, block)?;
    let index_ty = Type::index(gen.context);
    let c0_op = OperationBuilder::new("arith.constant", gen.loc())
        .add_attributes(&[(
            Identifier::new(gen.context, "value"),
            IntegerAttribute::new(index_ty, 0).into(),
        )])
        .add_results(&[index_ty])
        .build()?;
    let c0 = block.append_operation(c0_op).result(0)?.into();
    let store_op = OperationBuilder::new("vector.store", gen.loc())
        .add_operands(&[rhs_val, dst, c0])
        .build()?;
    block.append_operation(store_op);
    Ok(block)
}

/// Lowers `name = rhs` for a variable: stores into the variable's slot when it has one, and
/// rebinds the name to the new value otherwise. A name that is not bound is left alone.
///
/// # Errors
/// When the slot's element type cannot be parsed, the value cannot be converted to it, or MLIR
/// refuses an operation.
fn store_to_variable<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: Block<'c>,
    name: &crate::symbol::Symbol,
    rhs: Typed<'c>,
) -> Result<(), LowerError> {
    let (rhs_val, rhs_ty) = rhs;
    if let Some((mem_val, mem_ty)) = gen.env.get(name).copied() {
        let mem_ty_str = mem_ty.to_string();
        if mem_ty_str.starts_with("memref<") {
            let mut store_val = rhs_val;
            let inner_ty_str = &mem_ty_str[7..mem_ty_str.len() - 1];
            let inner_ty = Type::parse(gen.context, inner_ty_str).ok_or_else(|| {
                crate::codegen::lower::LowerError::ParseType("Type::parse failed".to_string())
            })?;
            // Coerce the value to the slot's element type. An annotated `let` now sizes its
            // slot from the annotation and types its literal to match (#240), so a later
            // assignment of a default-`i32` literal to a wider slot (`let mut r: i64; r = 5`)
            // must widen; `coerce_type` also still handles the `index`↔`i32` loop case.
            if rhs_ty != inner_ty {
                store_val = gen.coerce_type(&block, rhs_val, rhs_ty, inner_ty)?;
            }

            let store_op = OperationBuilder::new("memref.store", gen.loc())
                .add_operands(&[store_val, mem_val])
                .build()?;
            block.append_operation(store_op);
        } else if gen.allocs.contains(name.as_ref()) {
            let store_op = OperationBuilder::new("llvm.store", gen.loc())
                .add_operands(&[rhs_val, mem_val])
                .build()?;
            block.append_operation(store_op);
        } else {
            gen.env.insert(name.to_string().into(), (rhs_val, rhs_ty));
        }
    }
    Ok(())
}

/// Lowers `base[i] = rhs`. Returns the block to continue in, or `None`, with nothing stored,
/// when the indexed place cannot be reduced to a buffer and its indices.
///
/// # Errors
/// When the element type cannot be parsed, the value cannot be converted to it, or MLIR refuses
/// an operation.
fn store_to_element<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: Block<'c>,
    access: &syntax::IndexAccessExpr,
    rhs: Typed<'c>,
) -> Result<Option<Block<'c>>, LowerError> {
    let target = syntax::Expr::IndexAccess(syntax::IndexAccessExpr {
        base: access.base.clone(),
        index: access.index.clone(),
        span: Span::default(),
    });
    let Some((base_val, base_ty, indices, new_b)) = gen.flatten_indices(&target, block) else {
        return Ok(None);
    };
    if base_ty.to_string().starts_with("!llvm.ptr") {
        store_to_pointer_element(gen, new_b, (base_val, base_ty), indices[0], rhs)?;
    } else {
        store_to_buffer_element(gen, new_b, (base_val, base_ty), &indices, rhs)?;
    }
    Ok(Some(new_b))
}

/// Stores `rhs` at offset `index` from the raw pointer `base`.
///
/// # Errors
/// When MLIR refuses an operation.
fn store_to_pointer_element<'c>(
    gen: &MeliorGenerator<'c>,
    block: Block<'c>,
    base: Typed<'c>,
    index: Value<'c, 'c>,
    rhs: Typed<'c>,
) -> Result<(), LowerError> {
    let (base_val, base_ty) = base;
    let (rhs_val, rhs_ty) = rhs;
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
                TypeAttribute::new(rhs_ty).into(),
            ),
        ])
        .add_operands(&[base_val, idx_i64])
        .add_results(&[base_ty])
        .build()?;

    let gep_ref = block.append_operation(gep_op);
    let ptr_val = gep_ref.result(0)?.into();

    let store_op = OperationBuilder::new("llvm.store", gen.loc())
        .add_operands(&[rhs_val, ptr_val])
        .build()?;

    block.append_operation(store_op);
    Ok(())
}

/// Stores `rhs` into the buffer `base` at `indices`, converting it to the buffer's element type.
///
/// # Errors
/// When the element type cannot be parsed, the value cannot be converted to it, or MLIR refuses
/// an operation.
fn store_to_buffer_element<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: Block<'c>,
    base: Typed<'c>,
    indices: &[Value<'c, 'c>],
    rhs: Typed<'c>,
) -> Result<(), LowerError> {
    let (base_val, base_ty) = base;
    let (rhs_val, rhs_ty) = rhs;
    let base_ty_str = base_ty.to_string();
    let mut inner_ty_str = String::new();
    if let Some(start) = base_ty_str.find('<') {
        let inner = &base_ty_str[start + 1..base_ty_str.len() - 1];
        let parts: Vec<&str> = inner.split(',').collect();
        let shape_type = parts[0].trim();
        if let Some(last_x) = shape_type.rfind('x') {
            inner_ty_str = shape_type[last_x + 1..].to_string();
        } else {
            inner_ty_str = shape_type.to_string();
        }
    }

    let mut store_val = rhs_val;
    if !inner_ty_str.is_empty() {
        let inner_ty = melior::ir::Type::parse(gen.context, &inner_ty_str).ok_or_else(|| {
            crate::codegen::lower::LowerError::ParseType("Type::parse failed".to_string())
        })?;
        store_val = gen.coerce_type(&block, store_val, rhs_ty, inner_ty)?;
    }

    let mut store_builder =
        OperationBuilder::new("memref.store", gen.loc()).add_operands(&[store_val, base_val]);

    for idx in indices {
        store_builder = store_builder.add_operands(&[*idx]);
    }

    let store_op = store_builder.build()?;
    block.append_operation(store_op);
    Ok(())
}

/// Lowers `*pointer = rhs_val`: stores the value through the pointer.
///
/// # Errors
/// When `pointer` cannot be lowered or MLIR refuses the store.
fn store_through_dereference<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: Block<'c>,
    pointer: &Expr,
    rhs_val: Value<'c, 'c>,
) -> Result<Block<'c>, LowerError> {
    let (ptr_val, ptr_ty, new_b) = gen.generate_expr(pointer, block)?;
    // A `&mut` to a mutable local holds that local's scalar memref.
    let ptr_ty_str = ptr_ty.to_string();
    let op_name = if ptr_ty_str.starts_with("memref<") && !ptr_ty_str.contains('x') {
        "memref.store"
    } else {
        "llvm.store"
    };
    let store_op = OperationBuilder::new(op_name, gen.loc())
        .add_operands(&[rhs_val, ptr_val])
        .build()?;
    new_b.append_operation(store_op);
    Ok(new_b)
}

/// Lowers `base.member = rhs` where `base` is a variable. Nothing is stored when the struct or
/// the field cannot be found.
///
/// # Errors
/// When `base` or a type cannot be lowered, or MLIR refuses an operation.
///
/// # Panics
/// When `base` is not a plain variable. A deeper path such as `a.b.c = v` is rewritten into an
/// assignment to a variable's field before it gets here.
fn store_to_field<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: Block<'c>,
    access: &MemberAccessExpr,
    rhs: Typed<'c>,
) -> Result<Block<'c>, LowerError> {
    let MemberAccessExpr {
        base,
        member,
        struct_name,
        span: _,
    } = access;
    let Expr::Identifier(IdentifierExpr {
        name: base_name,
        span: _,
    }) = &**base
    else {
        panic!("Complex struct assignment lhs not supported");
    };
    let (base_val, base_ty, new_b) = gen.generate_expr(base, block)?;
    let base_ty_str = base_ty.to_string();
    let is_ptr = base_ty_str.starts_with("!llvm.ptr");

    let Some(resolved_struct_name) =
        assigned_struct_name(gen, struct_name.as_ref(), base_name, &base_ty_str)
    else {
        return Ok(new_b);
    };
    // A monomorphized generic carries its instantiation in the name ("Vec<i32>"),
    // but `gen.structs` and the emitted LLVM struct type are keyed by the *base*
    // name ("Vec"). Strip the args before looking the declaration up — without this
    // the lookup missed and the field store was silently dropped (#205, e.g.
    // `Vec::push`'s `self.len = self.len + 1`). The field order (hence offset) is the
    // same for every instantiation, so the generic declaration is the right one.
    let base_struct: crate::symbol::Symbol = resolved_struct_name
        .split('<')
        .next()
        .unwrap_or(resolved_struct_name.as_ref())
        .into();
    let Some(struct_decl) = gen.structs.get(&base_struct).cloned() else {
        return Ok(new_b);
    };
    let Some(field_idx) = struct_decl.fields.iter().position(|(n, _)| n == member) else {
        return Ok(new_b);
    };
    // Substitute the struct's generic params (T -> i32, …) so a generic field
    // type like `data: *mut T` lowers instead of a raw `T` reaching codegen.
    let mapping = generic_arg_mapping(&resolved_struct_name, &struct_decl.generics);
    let field_ast_ty = &struct_decl.fields[field_idx].1;
    let field_ty = gen.lower_type(&field_ast_ty.substitute(&mapping))?;
    let field = FieldStore {
        index: field_idx,
        val: value_for_field(gen, new_b, field_ast_ty, rhs, field_ty)?,
    };

    if is_ptr {
        let struct_llvm_ty = struct_instance_ty(gen, &base_struct, &struct_decl, &mapping)?;
        store_field_through_pointer(gen, new_b, base_val, struct_llvm_ty, &field)?;
    } else {
        store_field_in_struct_value(gen, new_b, base_name, (base_val, base_ty), &field)?;
    }
    Ok(new_b)
}

/// The name of the struct whose field is being assigned: the one the checker recorded, else the
/// one written in the base's LLVM type, else the one the base variable's source type names.
fn assigned_struct_name(
    gen: &MeliorGenerator<'_>,
    struct_name: Option<&crate::symbol::Symbol>,
    base_name: &crate::symbol::Symbol,
    base_ty_str: &str,
) -> Option<crate::symbol::Symbol> {
    let mut struct_name_opt = struct_name.cloned();
    if struct_name_opt.is_none() {
        if let Some(start_idx) = base_ty_str.find('"') {
            if let Some(end_idx) = base_ty_str[start_idx + 1..].find('"') {
                struct_name_opt = Some(
                    base_ty_str[start_idx + 1..start_idx + 1 + end_idx]
                        .to_string()
                        .into(),
                );
            }
        }
    }
    // #205: a `&Struct` base lowers to a bare `!llvm.ptr` with no embedded struct name,
    // and sema does not always stamp `struct_name` on an assignment target. Without the
    // name we cannot find the field to store to, and the write is silently dropped (the
    // bug that broke `Vec::push`'s `self.len = self.len + 1`). Recover it from the base's
    // AST type.
    if struct_name_opt.is_none() {
        if let Some(ast_ty) = gen.ast_env.get(base_name) {
            struct_name_opt = nominal_struct_name(ast_ty);
        }
    }
    struct_name_opt
}

/// The value `rhs` in the form a field of type `field_ty` stores: an `index` and an `i32` are
/// cast to each other, and a tensor is put in the layout a tensor field has.
///
/// # Errors
/// When MLIR refuses the cast or the tensor cannot be converted.
///
/// # Panics
/// When MLIR cannot build the `arith.index_cast` operation, which would be a bug in the
/// compiler.
fn value_for_field<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: Block<'c>,
    field_ast_ty: &syntax::Type,
    rhs: Typed<'c>,
    field_ty: Type<'c>,
) -> Result<Value<'c, 'c>, LowerError> {
    let (rhs_val, rhs_ty) = rhs;
    let mut field_val = rhs_val;

    if rhs_ty != field_ty
        && ((rhs_ty.to_string() == "index" && field_ty.to_string() == "i32")
            || (rhs_ty.to_string() == "i32" && field_ty.to_string() == "index"))
    {
        let cast_op = OperationBuilder::new("arith.index_cast", gen.loc())
            .add_operands(&[rhs_val])
            .add_results(&[field_ty])
            .build()
            .unwrap();
        field_val = block.append_operation(cast_op).result(0)?.into();
    }
    if let Some(rank) = crate::codegen::generator::tensor_field_rank(field_ast_ty) {
        field_val = super::expr::tensor_into_field(gen, &block, field_val, rhs_ty, field_ty, rank)?;
    }
    Ok(field_val)
}

/// Stores a field through `base_val`, a pointer to a struct of LLVM type `struct_llvm_ty`.
///
/// # Errors
/// When the field's position does not fit an `i32`, or MLIR refuses an operation's result.
///
/// # Panics
/// When MLIR cannot build the `llvm.getelementptr` or `llvm.store` operation, which would be a
/// bug in the compiler.
fn store_field_through_pointer<'c>(
    gen: &MeliorGenerator<'c>,
    block: Block<'c>,
    base_val: Value<'c, 'c>,
    struct_llvm_ty: Type<'c>,
    field: &FieldStore<'c>,
) -> Result<(), LowerError> {
    let ptr_ty = gen.ptr_ty;
    let position = i32::try_from(field.index).map_err(|e| {
        LowerError::from(format!("field position {} is too large: {e}", field.index))
    })?;

    let gep_op = OperationBuilder::new("llvm.getelementptr", gen.loc())
        .add_attributes(&[
            (
                Identifier::new(gen.context, "rawConstantIndices"),
                DenseI32ArrayAttribute::new(gen.context, &[0, position]).into(),
            ),
            (
                Identifier::new(gen.context, "elem_type"),
                TypeAttribute::new(struct_llvm_ty).into(),
            ),
        ])
        .add_operands(&[base_val])
        .add_results(&[ptr_ty])
        .build()
        .unwrap();

    let gep_ref = block.append_operation(gep_op);
    let ptr_val = gep_ref.result(0)?.into();

    let store_op = OperationBuilder::new("llvm.store", gen.loc())
        .add_operands(&[field.val, ptr_val])
        .build()
        .unwrap();
    block.append_operation(store_op);
    Ok(())
}

/// Stores a field of the struct value `base`, which the variable `base_name` holds: makes a
/// copy of the struct with the field replaced, and puts the copy back where the variable lives.
///
/// # Errors
/// When the field's position does not fit an `i64`, or MLIR refuses an operation's result.
///
/// # Panics
/// When MLIR cannot build the `llvm.insertvalue` operation or the store that follows it, which
/// would be a bug in the compiler.
fn store_field_in_struct_value<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: Block<'c>,
    base_name: &crate::symbol::Symbol,
    base: Typed<'c>,
    field: &FieldStore<'c>,
) -> Result<(), LowerError> {
    let (base_val, base_ty) = base;
    let position = i64::try_from(field.index).map_err(|e| {
        LowerError::from(format!("field position {} is too large: {e}", field.index))
    })?;
    let pos_attr = melior::ir::attribute::DenseI64ArrayAttribute::new(gen.context, &[position]);
    let insert_op = OperationBuilder::new("llvm.insertvalue", gen.loc())
        .add_operands(&[base_val, field.val])
        .add_attributes(&[(Identifier::new(gen.context, "position"), pos_attr.into())])
        .add_results(&[base_ty])
        .build()
        .unwrap();

    let new_struct_val = block.append_operation(insert_op).result(0)?.into();

    if let Some((mem_val, mem_ty)) = gen.env.get(base_name).copied() {
        let mem_ty_str = mem_ty.to_string();
        if mem_ty_str.starts_with("memref<") {
            let store_op = OperationBuilder::new("memref.store", gen.loc())
                .add_operands(&[new_struct_val, mem_val])
                .build()
                .unwrap();
            block.append_operation(store_op);
        } else if gen.allocs.contains(base_name.as_ref()) {
            // A `let mut` struct lives behind a pointer. Store the
            // updated struct through it, as a plain `x = v` does;
            // replacing the pointer with the value made the next
            // read load from a struct.
            let store_op = OperationBuilder::new("llvm.store", gen.loc())
                .add_operands(&[new_struct_val, mem_val])
                .build()
                .unwrap();
            block.append_operation(store_op);
        } else {
            gen.env
                .insert(base_name.to_string().into(), (new_struct_val, base_ty));
        }
    }
    Ok(())
}

impl<'c> LowerToMelior<'c> for CompoundAssignStmt {
    type Output = Result<Option<melior::ir::BlockRef<'c, 'c>>, LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let CompoundAssignStmt {
            lhs,
            op,
            rhs,
            span,
            operand_ty,
        } = self;
        // `*p op= v`, `s.f op= v` and `p[i] op= v` through a raw pointer are lowered as
        // `lhs = lhs op v`, which stores through the place the way a plain assignment does.
        // Reading such a place twice has no effect.
        let through_raw_pointer = match lhs {
            Expr::IndexAccess(i) => {
                matches!(gen.infer_ast_type(&i.base), Some(syntax::Type::Pointer(..)))
            }
            _ => false,
        };
        if matches!(lhs, Expr::Dereference(_) | Expr::MemberAccess(_)) || through_raw_pointer {
            let assign = AssignStmt {
                lhs: lhs.clone(),
                rhs: Expr::BinaryOp(syntax::BinaryOpExpr {
                    lhs: Box::new(lhs.clone()),
                    op: op.clone(),
                    rhs: Box::new(rhs.clone()),
                    span: *span,
                    operand_ty: operand_ty.clone(),
                }),
                span: *span,
            };
            return assign.lower(gen, block);
        }
        let (lhs_val, ty, block) = gen.generate_expr(lhs, block)?;
        let prev_expected = gen.expected_type;
        gen.expected_type = Some(ty);
        let (rhs_val, rhs_ty, block) = gen.generate_expr(rhs, block)?;
        gen.expected_type = prev_expected;

        // Coerce the operand to the accumulator's type before the arithmetic. Now that an
        // annotated `let` types its literal at the annotation (#240), a wider accumulator
        // (`let mut i: i64`) legitimately combines with a default-`i32` literal (`i += 5`); the
        // general `coerce_type` widens/narrows it (and still handles the `index`↔`i32` loop case).
        let mut actual_rhs = rhs_val;
        if rhs_ty != ty {
            actual_rhs = gen.coerce_type(&block, rhs_val, rhs_ty, ty)?;
        }

        let result_val = apply_compound_op(gen, block, self, (lhs_val, ty), actual_rhs)?;
        if let Some(done) = store_compound_result(gen, block, lhs, (result_val, ty))? {
            return Ok(Some(done));
        }
        // Computing the new value and dropping it is how `*p += 1` used to do nothing.
        Err(LowerError::from(format!(
            "a compound assignment to this place is not stored: {lhs:?}"
        )))
    }
}

/// Emits the arithmetic of the compound assignment `stmt` on `lhs` and `rhs_val`, which already
/// has the type of `lhs`, and returns the result.
///
/// # Errors
/// When MLIR refuses the operation.
fn apply_compound_op<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: Block<'c>,
    stmt: &CompoundAssignStmt,
    lhs: Typed<'c>,
    rhs_val: Value<'c, 'c>,
) -> Result<Value<'c, 'c>, LowerError> {
    let (lhs_val, ty) = lhs;
    let op = &stmt.op;
    let operand_ty = &stmt.operand_ty;
    let is_float = ty.to_string().contains("f32")
        || ty.to_string().contains("f64")
        || ty.to_string().contains("f16")
        || ty.to_string().contains("bf16");
    let is_unsigned = matches!(operand_ty, Some(e) if !e.is_float() && !e.is_signed_int());
    let checked = if *operand_ty == Some(syntax::ElementType::USize) {
        crate::codegen::lower::lower_usize_checked(gen, &block, op, lhs_val, rhs_val)?
    } else {
        None
    };
    if let Some(v) = checked {
        return Ok(v);
    }
    let bin_op = OperationBuilder::new(op.get_op_name(is_float, is_unsigned), gen.loc())
        .add_operands(&[lhs_val, rhs_val])
        .add_results(&[ty])
        .build()?;
    Ok(block.append_operation(bin_op).result(0)?.into())
}

/// Stores the result of a compound assignment back into `lhs`, a variable or an element of a
/// buffer. Returns the block to continue in, or `None` when `lhs` is a place this cannot store
/// to.
///
/// # Errors
/// When MLIR refuses the store.
fn store_compound_result<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: Block<'c>,
    lhs: &Expr,
    result: Typed<'c>,
) -> Result<Option<Block<'c>>, LowerError> {
    let (result_val, ty) = result;
    if let Expr::Identifier(IdentifierExpr { name, span: _ }) = lhs {
        if let Some((mem_val, mem_ty)) = gen.env.get(name).copied() {
            let mem_ty_str = mem_ty.to_string();
            if mem_ty_str.starts_with("memref<") {
                let store_op = OperationBuilder::new("memref.store", gen.loc())
                    .add_operands(&[result_val, mem_val])
                    .build()?;
                block.append_operation(store_op);
            } else {
                gen.env.insert(name.to_string().into(), (result_val, ty));
            }
            return Ok(Some(block));
        }
    } else if let Expr::IndexAccess(_) = lhs {
        if let Some((mem_val, mem_ty, indices, new_b)) = gen.flatten_indices(lhs, block) {
            let mem_ty_str = mem_ty.to_string();
            if mem_ty_str.starts_with("memref<") {
                let mut operands = vec![result_val, mem_val];
                operands.extend(indices);
                let store_op = OperationBuilder::new("memref.store", gen.loc())
                    .add_operands(&operands)
                    .build()?;
                new_b.append_operation(store_op);
                return Ok(Some(new_b));
            }
        }
    }
    Ok(None)
}

impl<'c> LowerToMelior<'c> for ExprStmtStmt {
    type Output = Result<Option<melior::ir::BlockRef<'c, 'c>>, LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let ExprStmtStmt {
            expr,
            has_semi: _,
            span: _,
        } = self;
        let prev = gen.expected_type;
        gen.expected_type = Some(gen.none_ty);
        let (_, _, updated_block) = gen.generate_expr(expr, block)?;
        gen.expected_type = prev;
        Ok(Some(updated_block))
    }
}
