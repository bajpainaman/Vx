use super::{BlockLike, LowerError, LowerToMelior, MeliorGenerator, RegionLike};
use crate::syntax;
use crate::syntax::{
    BorrowExpr, BreakStmt, ContinueStmt, Expr, ForLoopStmt, FunctionCallExpr, IdentifierExpr,
    IfExpr, LoopStmt, Span,
};
use melior::ir::{
    attribute::{FloatAttribute, IntegerAttribute},
    operation::OperationBuilder,
    Identifier, Type, Value,
};

/// A value together with its MLIR type.
type TypedValue<'c> = (Value<'c, 'c>, Type<'c>);

impl<'c> LowerToMelior<'c> for IfExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let IfExpr {
            is_comptime,
            cond,
            then_block,
            else_block: else_block_opt,
            span: _,
        } = self;

        if *is_comptime {
            return lower_comptime_if(gen, block, then_block, else_block_opt.as_deref());
        }

        let (cond_val, _, block) = gen.generate_expr(cond, block)?;

        let expected = gen.expected_type;
        let has_ret = expected.is_none_or(|t| {
            let t = t.to_string();
            t != "none" && t != "void"
        });
        let parent_region = block.parent_region().unwrap();
        let then_b = parent_region.append_block(melior::ir::Block::new(&[]));
        let else_b = parent_region.append_block(melior::ir::Block::new(&[]));
        // The merge block's argument is added once both branches are lowered, because with no
        // expected type the value's type is whatever the branches produce.
        let merge_b = parent_region.append_block(melior::ir::Block::new(&[]));

        append_cond_br(gen, block, cond_val, then_b, else_b)?;

        let empty = Vec::new();
        let (then_end, then_val) = lower_branch(gen, then_block, then_b, has_ret)?;
        let (else_end, else_val) = lower_branch(
            gen,
            else_block_opt.as_ref().unwrap_or(&empty),
            else_b,
            has_ret,
        )?;

        // With no expected type and no value from either branch, the `if` is `void`, as in
        // `let u = if c { print(..); };`. There is no value to merge.
        let has_ret = has_ret && (expected.is_some() || then_val.is_some() || else_val.is_some());

        // Without an expected type, the branches decide. A loop variable or an extent is an
        // `index` in MLIR but an `i32` in Vx, so an `index` value is given the Vx type, and a
        // branch that yields an `i32` and one that yields an `index` agree.
        let ret_ty = match expected {
            Some(t) => t,
            None => match then_val.or(else_val).map(|(_, t)| t) {
                Some(t) if t == gen.index_ty => gen.i32_ty,
                Some(t) => t,
                None => gen.f32_ty,
            },
        };
        let merge_arg = has_ret.then(|| merge_b.add_argument(ret_ty, gen.loc()));

        let ends = [(then_end, then_val), (else_end, else_val)];
        join_branches(gen, ends, merge_b, has_ret, ret_ty)?;

        let both_ended = then_end.is_none() && else_end.is_none();
        let next = finish_merge_block(gen, merge_b, both_ended)?;

        match merge_arg {
            Some(res) => Ok((res, ret_ty, next)),
            None => Ok((cond_val, ret_ty, next)),
        }
    }
}

/// Lower an `if comptime`: only the branch chosen at compile time is lowered, straight into
/// `block`. Returns the value of the `if`, its type, and the block that lowering continues in.
///
/// # Errors
///
/// Returns an error when a statement of the chosen branch cannot be lowered, or when MLIR
/// rejects an operation.
fn lower_comptime_if<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    then_block: &[syntax::Statement],
    else_block: Option<&[syntax::Statement]>,
) -> Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError> {
    let depth = gen.open_block();
    let target_block = if then_block.is_empty() {
        else_block
    } else {
        Some(then_block)
    };

    let (cur, last_val) = match target_block {
        Some(tb) => lower_comptime_statements(gen, tb, block)?,
        None => (block, None),
    };
    gen.close_blocks_to(depth);

    if let Some((val, ty)) = last_val {
        return Ok((val, ty, cur));
    }
    comptime_if_placeholder(gen, cur)
}

/// Lower the statements of the branch an `if comptime` keeps. Returns the block that lowering
/// continues in, and the value of the last expression when it has no `;` after it.
///
/// # Errors
///
/// Returns an error when a statement or expression of the branch cannot be lowered.
fn lower_comptime_statements<'c>(
    gen: &mut MeliorGenerator<'c>,
    tb: &[syntax::Statement],
    block: melior::ir::BlockRef<'c, 'c>,
) -> Result<(melior::ir::BlockRef<'c, 'c>, Option<TypedValue<'c>>), LowerError> {
    let mut last_val = None;
    // The surviving branch is spliced straight into the enclosing block,
    // with no branch of its own -- which means every statement in it can
    // move the insertion point, and the branch as a whole can close the
    // block outright. A `for` opens three blocks and lands in a fourth;
    // an `if` lands in its merge block; a `return` ends the block for
    // good. Lowering the rest of the function into the block we started
    // in appends past a terminator, so `if comptime <taken> { for ... }`
    // did not compile at all.
    let mut cur = block;
    let mut terminated = false;
    for (i, stmt) in tb.iter().enumerate() {
        let is_last = i == tb.len() - 1;
        if is_last {
            if let syntax::Statement::ExprStmt(syntax::stmt::ExprStmtStmt {
                expr, has_semi, ..
            }) = stmt
            {
                let (val, ty, tail_block) = gen.generate_expr(expr, cur)?;
                cur = tail_block;
                cur = gen.run_queued_drops(cur)?;
                if !has_semi {
                    last_val = Some((val, ty));
                }
                continue;
            }
        }
        if let Some(b) = gen.generate_statement(stmt, cur)? {
            cur = b
        } else {
            terminated = true;
            break;
        }
    }

    if terminated {
        cur = super::dead_continuation(cur);
    }
    Ok((cur, last_val))
}

/// Build the value of an `if comptime` whose chosen branch gives no value: a zero of the type
/// the surrounding code expects, appended to `cur`.
///
/// # Errors
///
/// Returns an error when MLIR rejects the constant.
fn comptime_if_placeholder<'c>(
    gen: &MeliorGenerator<'c>,
    cur: melior::ir::BlockRef<'c, 'c>,
) -> Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError> {
    let ret_ty = gen.expected_type.unwrap_or(gen.f32_ty);
    let dummy_op = if ret_ty.to_string() == "f32" {
        OperationBuilder::new("arith.constant", gen.loc())
            .add_attributes(&[(
                Identifier::new(gen.context, "value"),
                FloatAttribute::new(gen.context, ret_ty, 0.0).into(),
            )])
            .add_results(&[ret_ty])
            .build()?
    } else if ret_ty.to_string() == "i1" {
        OperationBuilder::new("arith.constant", gen.loc())
            .add_attributes(&[(
                Identifier::new(gen.context, "value"),
                IntegerAttribute::new(ret_ty, 0).into(),
            )])
            .add_results(&[ret_ty])
            .build()?
    } else {
        OperationBuilder::new("arith.constant", gen.loc())
            .add_attributes(&[(
                Identifier::new(gen.context, "value"),
                IntegerAttribute::new(ret_ty, 0).into(),
            )])
            .add_results(&[ret_ty])
            .build()?
    };
    Ok((
        cur.append_operation(dummy_op).result(0)?.into(),
        ret_ty,
        cur,
    ))
}

/// Append a conditional branch to `block`: go to `on_true` when `cond_val` is true, and to
/// `on_false` otherwise.
///
/// # Errors
///
/// Returns an error when MLIR rejects the branch.
fn append_cond_br<'c>(
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    cond_val: Value<'c, 'c>,
    on_true: melior::ir::BlockRef<'c, 'c>,
    on_false: melior::ir::BlockRef<'c, 'c>,
) -> Result<(), LowerError> {
    block.append_operation(
        OperationBuilder::new("cf.cond_br", gen.loc())
            .add_operands(&[cond_val])
            .add_successors(&[&*on_true, &*on_false])
            .add_attributes(&[(
                Identifier::new(gen.context, "operandSegmentSizes"),
                melior::ir::attribute::DenseI32ArrayAttribute::new(gen.context, &[1, 0, 0]).into(),
            )])
            .build()?,
    );
    Ok(())
}

/// End each branch of an `if` with a jump to `merge_b`. When the `if` has a value (`has_ret`),
/// the jump carries the branch's value as a `ret_ty`, or a zero when the branch gave none.
/// A branch whose end block is `None` already ended in a `return` and is skipped.
///
/// # Errors
///
/// Returns an error when a branch's value cannot be converted to `ret_ty`, or when MLIR rejects
/// an operation.
fn join_branches<'c>(
    gen: &mut MeliorGenerator<'c>,
    ends: [(Option<melior::ir::BlockRef<'c, 'c>>, Option<TypedValue<'c>>); 2],
    merge_b: melior::ir::BlockRef<'c, 'c>,
    has_ret: bool,
    ret_ty: Type<'c>,
) -> Result<(), LowerError> {
    for (end, val) in ends {
        let Some(end) = end else { continue };
        let mut yield_operands = vec![];
        if has_ret {
            let v = if let Some((v, t)) = val {
                gen.coerce_type(&end, v, t, ret_ty)?
            } else {
                let dummy_op = OperationBuilder::new("arith.constant", gen.loc())
                    .add_attributes(&[(
                        Identifier::new(gen.context, "value"),
                        melior::ir::attribute::IntegerAttribute::new(ret_ty, 0).into(),
                    )])
                    .build()?;
                end.append_operation(dummy_op).result(0)?.into()
            };
            yield_operands.push(v);
        }
        end.append_operation(
            OperationBuilder::new("cf.br", gen.loc())
                .add_operands(&yield_operands)
                .add_successors(&[&*merge_b])
                .build()?,
        );
    }
    Ok(())
}

/// Give the block after an `if` a terminator when both branches ended in one of their own
/// (`both_ended`), so nothing jumps to it. Returns the block that lowering continues in.
///
/// # Errors
///
/// Returns an error when MLIR rejects the `llvm.unreachable` operation.
fn finish_merge_block<'c>(
    gen: &MeliorGenerator<'c>,
    merge_b: melior::ir::BlockRef<'c, 'c>,
    both_ended: bool,
) -> Result<melior::ir::BlockRef<'c, 'c>, LowerError> {
    // Both arms ended in a terminator, so nothing branches to the merge block -- but it still
    // needs a terminator of its own, because a block without one fails the MLIR verifier. That
    // is what a body ending in `if c { return a; } else { return b; }` produced: an empty block
    // the verifier rejected, after the frontend had accepted the program.
    // What follows the `if` then goes in a block of its own, since nothing may come after the
    // terminator: `while true { ..; break; }` lowers to an `if` both of whose arms `break`,
    // and the loop's back-edge used to land after the `llvm.unreachable`.
    if both_ended {
        merge_b.append_operation(OperationBuilder::new("llvm.unreachable", gen.loc()).build()?);
        Ok(super::dead_continuation(merge_b))
    } else {
        Ok(merge_b)
    }
}

/// Lower one branch of an `if` into `b`. Returns the block the branch ends in, or `None` when it
/// ended in a terminator of its own (a `return`), and the value of its last expression when the
/// `if` produces one.
///
/// # Errors
///
/// Returns an error when a statement or expression of the branch cannot be lowered.
#[allow(
    clippy::type_complexity,
    reason = "the result is only handed to the one caller, which takes it apart at once"
)]
fn lower_branch<'c>(
    gen: &mut MeliorGenerator<'c>,
    stmts: &[syntax::Statement],
    b: melior::ir::BlockRef<'c, 'c>,
    has_ret: bool,
) -> Result<
    (
        Option<melior::ir::BlockRef<'c, 'c>>,
        Option<(Value<'c, 'c>, Type<'c>)>,
    ),
    LowerError,
> {
    let depth = gen.open_block();
    let lowered = lower_branch_statements(gen, stmts, b, has_ret);
    gen.close_blocks_to(depth);
    lowered
}

/// Lower the statements of one branch of an `if`, starting in `b`. Returns the same pair as
/// `lower_branch`.
///
/// # Errors
///
/// Returns an error when a statement or expression of the branch cannot be lowered.
#[allow(
    clippy::type_complexity,
    reason = "the result is only handed to `lower_branch`, which returns it unchanged"
)]
fn lower_branch_statements<'c>(
    gen: &mut MeliorGenerator<'c>,
    stmts: &[syntax::Statement],
    mut b: melior::ir::BlockRef<'c, 'c>,
    has_ret: bool,
) -> Result<
    (
        Option<melior::ir::BlockRef<'c, 'c>>,
        Option<(Value<'c, 'c>, Type<'c>)>,
    ),
    LowerError,
> {
    let mut val = None;
    for (i, stmt) in stmts.iter().enumerate() {
        if i == stmts.len() - 1 && has_ret {
            if let syntax::Statement::ExprStmt(syntax::stmt::ExprStmtStmt {
                expr,
                has_semi: false,
                ..
            }) = stmt
            {
                let (v, t, next) = gen.generate_expr(expr, b)?;
                b = next;
                b = gen.run_queued_drops(b)?;
                val = Some((v, t));
                continue;
            }
        }
        match gen.generate_statement(stmt, b)? {
            Some(next) => b = next,
            None => return Ok((None, None)),
        }
    }
    Ok((Some(b), val))
}

impl<'c> LowerToMelior<'c> for ForLoopStmt {
    type Output = Result<Option<melior::ir::BlockRef<'c, 'c>>, LowerError>;
    /// The loop variable is a value, not a stack slot, so a `let mut` of the same name earlier
    /// in the function must not make the body read it through that slot. What the name meant
    /// before the loop is put back after it.
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let depth = gen.open_block();
        gen.note_shadow(self.iter.as_str());
        gen.allocs.remove(self.iter.as_str());
        let result = lower_for_loop(self, gen, block);
        gen.close_blocks_to(depth);
        result
    }
}

/// Lower a `for` loop. A loop over a range and a loop over a tensor count with an index; any
/// other loop calls the iterator's `next` function until it returns no value. Returns the block
/// that runs after the loop.
///
/// # Errors
///
/// Returns an error when the thing looped over or a statement of the body cannot be lowered, or
/// when MLIR rejects an operation.
fn lower_for_loop<'c>(
    for_loop: &ForLoopStmt,
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
) -> Result<Option<melior::ir::BlockRef<'c, 'c>>, LowerError> {
    if let Expr::Range(range) = &*for_loop.iterable {
        return lower_range_loop(gen, block, range, for_loop);
    }

    let (iter_val, iter_ty, block) = gen.generate_expr(&for_loop.iterable, block)?;
    let iter_ty_str = iter_ty.to_string();

    if iter_ty_str.starts_with("tensor<") {
        return lower_tensor_loop(gen, block, for_loop, iter_val, &iter_ty_str);
    }

    lower_iterator_loop(gen, block, for_loop, iter_val, iter_ty)
}

/// The blocks of a loop that counts with an index, and the index on the current turn.
struct CountedLoop<'c> {
    /// The region the loop's blocks are in.
    region: melior::ir::RegionRef<'c, 'c>,
    /// The block that compares the index with the end. The index is its argument.
    cond_block: melior::ir::BlockRef<'c, 'c>,
    /// The block the loop body starts in.
    body_block: melior::ir::BlockRef<'c, 'c>,
    /// The block that runs after the loop.
    merge_block: melior::ir::BlockRef<'c, 'c>,
    /// The index on the current turn of the loop.
    current_idx: Value<'c, 'c>,
}

/// Lower `for i in start..end`: the loop variable is the index itself.
///
/// # Errors
///
/// Returns an error when `start`, `end` or a statement of the body cannot be lowered, or when
/// MLIR rejects an operation.
///
/// # Panics
///
/// Panics if `block` is not inside a region, which would be a bug in the caller.
fn lower_range_loop<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    range: &syntax::expr::RangeExpr,
    for_loop: &ForLoopStmt,
) -> Result<Option<melior::ir::BlockRef<'c, 'c>>, LowerError> {
    let (start_val, start_ty, block) = gen.generate_expr(&range.start, block)?;
    let (end_val, end_ty, block) = gen.generate_expr(&range.end, block)?;

    let ty_index = gen.index_ty;

    let start_idx = cast_to_index(gen, block, start_val, start_ty)?;
    let end_idx = cast_to_index(gen, block, end_val, end_ty)?;
    let step_idx = index_constant(gen, block, 1)?;

    let counted = open_counted_loop(gen, block, start_idx, end_idx)?;

    gen.env.insert(
        for_loop.iter.clone().into(),
        (counted.current_idx, ty_index),
    );

    close_counted_loop(gen, &for_loop.body, &counted, step_idx)
}

/// Lower a `for` loop over a tensor whose length is in its type: the loop variable is the
/// element at the index.
///
/// # Errors
///
/// Returns an error when the element type cannot be parsed, when a statement of the body cannot
/// be lowered, or when MLIR rejects an operation.
///
/// # Panics
///
/// Panics if `block` is not inside a region, or if `iter_ty_str` is not a one-dimensional tensor
/// type with a length written as a number, such as `tensor<4xf32>`.
fn lower_tensor_loop<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    for_loop: &ForLoopStmt,
    iter_val: Value<'c, 'c>,
    iter_ty_str: &str,
) -> Result<Option<melior::ir::BlockRef<'c, 'c>>, LowerError> {
    let start_idx = index_constant(gen, block, 0)?;

    let len_str = iter_ty_str
        .split('x')
        .next()
        .unwrap()
        .split('<')
        .nth(1)
        .unwrap();
    let len: i64 = len_str.parse().unwrap();

    let end_idx = index_constant(gen, block, len)?;
    let step_idx = index_constant(gen, block, 1)?;

    let counted = open_counted_loop(gen, block, start_idx, end_idx)?;

    let el_ty_str = iter_ty_str.split('x').nth(1).unwrap().trim_end_matches('>');
    let el_ty = Type::parse(gen.context, el_ty_str).ok_or_else(|| {
        crate::codegen::lower::LowerError::ParseType("Type::parse failed".to_string())
    })?;

    let extract_op = OperationBuilder::new("tensor.extract", gen.loc())
        .add_operands(&[iter_val, counted.current_idx])
        .add_results(&[el_ty])
        .build()?;

    let el_val = counted
        .body_block
        .append_operation(extract_op)
        .result(0)?
        .into();

    gen.env
        .insert(for_loop.iter.clone().into(), (el_val, el_ty));

    close_counted_loop(gen, &for_loop.body, &counted, step_idx)
}

/// Give back `val` as an `index`, appending a cast to `block` when its type `ty` is not `index`
/// already.
///
/// # Errors
///
/// Returns an error when MLIR rejects the cast.
fn cast_to_index<'c>(
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    val: Value<'c, 'c>,
    ty: Type<'c>,
) -> Result<Value<'c, 'c>, LowerError> {
    let ty_index = gen.index_ty;
    if ty == ty_index {
        return Ok(val);
    }
    let cast_op = OperationBuilder::new("arith.index_cast", gen.loc())
        .add_operands(&[val])
        .add_results(&[ty_index])
        .build()?;
    Ok(block.append_operation(cast_op).result(0)?.into())
}

/// Append an `index` constant with the given value to `block`.
///
/// # Errors
///
/// Returns an error when MLIR rejects the constant.
fn index_constant<'c>(
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    value: i64,
) -> Result<Value<'c, 'c>, LowerError> {
    let ty_index = gen.index_ty;
    let op = block.append_operation(
        OperationBuilder::new("arith.constant", gen.loc())
            .add_results(&[ty_index])
            .add_attributes(&[(
                Identifier::new(gen.context, "value"),
                IntegerAttribute::new(ty_index, value).into(),
            )])
            .build()?,
    );
    Ok(op.result(0)?.into())
}

/// Start a loop that counts from `start_idx` up to, but not including, `end_idx`. Creates the
/// loop's blocks, makes `block` jump to the condition block, and fills the condition block.
///
/// # Errors
///
/// Returns an error when MLIR rejects an operation.
///
/// # Panics
///
/// Panics if `block` is not inside a region, which would be a bug in the caller.
fn open_counted_loop<'c>(
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    start_idx: Value<'c, 'c>,
    end_idx: Value<'c, 'c>,
) -> Result<CountedLoop<'c>, LowerError> {
    let ty_index = gen.index_ty;
    let parent_region = block.parent_region().unwrap();
    let cond_block = parent_region.append_block(melior::ir::Block::new(&[(ty_index, gen.loc())]));
    let body_block = parent_region.append_block(melior::ir::Block::new(&[]));
    let merge_block = parent_region.append_block(melior::ir::Block::new(&[]));

    block.append_operation(
        OperationBuilder::new("cf.br", gen.loc())
            .add_operands(&[start_idx])
            .add_successors(&[&*cond_block])
            .build()?,
    );

    // The error from melior is dropped: it could only say that the block has no argument 0,
    // which the message below already says.
    let current_idx = cond_block
        .argument(0)
        .ok()
        .ok_or_else(|| {
            crate::codegen::lower::LowerError::from(format!("Missing argument {} block", 0))
        })?
        .into();

    let cmp_op = cond_block.append_operation(
        OperationBuilder::new("arith.cmpi", gen.loc())
            .add_operands(&[current_idx, end_idx])
            .add_results(&[gen.i1_ty])
            .add_attributes(&[(
                Identifier::new(gen.context, "predicate"),
                IntegerAttribute::new(
                    gen.i64_ty,
                    melior::dialect::arith::CmpiPredicate::Slt as i64,
                )
                .into(),
            )])
            .build()?,
    );
    let cond_val = cmp_op.result(0)?.into();

    append_cond_br(gen, cond_block, cond_val, body_block, merge_block)?;

    Ok(CountedLoop {
        region: parent_region,
        cond_block,
        body_block,
        merge_block,
        current_idx,
    })
}

/// Lower the body of a counted loop and add the block that moves the index on by `step_idx`.
/// Returns the block that runs after the loop.
///
/// # Errors
///
/// Returns an error when a statement of the body cannot be lowered, or when MLIR rejects an
/// operation.
fn close_counted_loop<'c>(
    gen: &mut MeliorGenerator<'c>,
    body: &[syntax::Statement],
    counted: &CountedLoop<'c>,
    step_idx: Value<'c, 'c>,
) -> Result<Option<melior::ir::BlockRef<'c, 'c>>, LowerError> {
    // `continue` must still advance the loop index, so it targets a
    // dedicated latch block rather than branching to the condition
    // block directly (which would skip the increment and, since the
    // condition block takes the index as an argument, emit invalid IR).
    let latch_block = counted.region.append_block(melior::ir::Block::new(&[]));
    let merge_block = counted.merge_block;
    lower_loop_statements(gen, body, counted.body_block, merge_block, latch_block)?;

    // Latch: increment the index and branch back to the condition.
    // Both normal fall-through and `continue` route through here.
    let next_idx_op = latch_block.append_operation(
        OperationBuilder::new("arith.addi", gen.loc())
            .add_operands(&[counted.current_idx, step_idx])
            .add_results(&[gen.index_ty])
            .build()?,
    );
    let next_idx = next_idx_op.result(0)?.into();
    latch_block.append_operation(
        OperationBuilder::new("cf.br", gen.loc())
            .add_operands(&[next_idx])
            .add_successors(&[&*counted.cond_block])
            .build()?,
    );

    Ok(Some(merge_block))
}

/// Lower the statements of a `for` loop's body, starting in `body_block`. While they are
/// lowered, a `break` jumps to `merge_block` and a `continue` jumps to `continue_block`. A body
/// that does not end in a terminator of its own also ends with a jump to `continue_block`.
///
/// # Errors
///
/// Returns an error when a statement of the body cannot be lowered, or when MLIR rejects the
/// jump.
fn lower_loop_statements<'c>(
    gen: &mut MeliorGenerator<'c>,
    body: &[syntax::Statement],
    mut body_block: melior::ir::BlockRef<'c, 'c>,
    merge_block: melior::ir::BlockRef<'c, 'c>,
    continue_block: melior::ir::BlockRef<'c, 'c>,
) -> Result<(), LowerError> {
    // These pointers point at the two parameters above, which live until this function returns.
    // They are popped again below, before that happens.
    gen.break_blocks.push(&raw const *merge_block);
    gen.continue_blocks.push(&raw const *continue_block);

    let mut body_terminated = false;
    for stmt in body {
        if let Some(b) = gen.generate_statement(stmt, body_block)? {
            body_block = b;
        } else {
            body_terminated = true;
            break;
        }
    }

    gen.break_blocks.pop();
    gen.continue_blocks.pop();

    if !body_terminated {
        body_block.append_operation(
            OperationBuilder::new("cf.br", gen.loc())
                .add_successors(&[&*continue_block])
                .build()?,
        );
    }
    Ok(())
}

/// Lower a `for` loop over an iterator: each turn calls the iterator's `next` function, and the
/// loop ends when `next` returns no value.
///
/// # Errors
///
/// Returns an error when the call to `next` or a statement of the body cannot be lowered, when
/// the type of the value `next` returns cannot be parsed, or when MLIR rejects an operation.
///
/// # Panics
///
/// Panics if `block` is not inside a region, which would be a bug in the caller.
fn lower_iterator_loop<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    for_loop: &ForLoopStmt,
    iter_val: Value<'c, 'c>,
    iter_ty: Type<'c>,
) -> Result<Option<melior::ir::BlockRef<'c, 'c>>, LowerError> {
    // Generic Iterator loop via cf
    let actual_next_name = resolve_next_fn(gen, for_loop, &iter_ty.to_string());
    let ptr_val = iterator_slot(gen, block, &for_loop.iterable, iter_val, iter_ty)?;

    let tmp_iter_name = format!("__iter_ptr_{}", gen.string_counter);
    gen.string_counter += 1;
    gen.env
        .insert(tmp_iter_name.clone().into(), (ptr_val, iter_ty));
    gen.allocs.insert(tmp_iter_name.clone());

    let parent_region = block.parent_region().unwrap();
    let cond_block = parent_region.append_block(melior::ir::Block::new(&[]));
    let body_block = parent_region.append_block(melior::ir::Block::new(&[]));
    let merge_block = parent_region.append_block(melior::ir::Block::new(&[]));

    block.append_operation(
        OperationBuilder::new("cf.br", gen.loc())
            .add_successors(&[&*cond_block])
            .build()?,
    );

    let next_call = next_call_expr(actual_next_name, tmp_iter_name);
    let (opt_val, opt_ty, cond_block_end) = gen.generate_expr(&next_call, cond_block)?;
    branch_if_some(gen, cond_block_end, opt_val, body_block, merge_block)?;

    let payload_ty = option_payload_type(gen, opt_ty)?;

    let extract_payload_op = OperationBuilder::new("llvm.extractvalue", gen.loc())
        .add_operands(&[opt_val])
        .add_results(&[payload_ty])
        .add_attributes(&[(
            Identifier::new(gen.context, "position"),
            melior::ir::attribute::DenseI64ArrayAttribute::new(gen.context, &[1]).into(),
        )])
        .build()?;
    let payload_val = body_block
        .append_operation(extract_payload_op)
        .result(0)?
        .into();

    gen.env
        .insert(for_loop.iter.clone().into(), (payload_val, payload_ty));
    lower_loop_statements(gen, &for_loop.body, body_block, merge_block, cond_block)?;

    Ok(Some(merge_block))
}

/// Find the name of the `next` function to call for a loop over an iterator whose MLIR type
/// prints as `iter_ty_str`. Gives back plain `next` when the program has no such function.
fn resolve_next_fn(gen: &MeliorGenerator<'_>, for_loop: &ForLoopStmt, iter_ty_str: &str) -> String {
    // Resolve the monomorphized `next` for *this* iterator. The mangler spells it with `$`
    // (`VecIter$i32$next$i32`), which the old `_next_`/`_next` patterns never matched (hence the
    // `Function next not found` panic). Prefer a `next` whose name shares the iterator's struct-name
    // prefix (from its MLIR type `!llvm.struct<"VecIter_i32", …>` -> `VecIter`), so the right
    // iterator's `next` is chosen when several are in scope. (#242)
    let iter_base: String = iter_ty_str
        .find('"')
        .and_then(|q| {
            iter_ty_str[q + 1..]
                .find('"')
                .map(|e| &iter_ty_str[q + 1..q + 1 + e])
        })
        .unwrap_or("")
        .chars()
        .take_while(|c| c.is_alphabetic())
        .collect();
    let is_next = |name: &str| {
        name.contains("_next_")
            || name.ends_with("_next")
            || name.contains("$next$")
            || name.ends_with("$next")
    };
    let mut actual_next_name = "next".to_string();
    if let Some(n) = for_loop
        .next_fn
        .as_ref()
        .filter(|n| gen.functions.contains_key(n.as_ref()))
    {
        actual_next_name = n.to_string();
    }
    for (name, (_, _args)) in gen.functions.iter().filter(|_| actual_next_name == "next") {
        if is_next(name) && (iter_base.is_empty() || name.starts_with(&iter_base)) {
            actual_next_name = name.to_string();
            break;
        }
    }
    // Fallback: any `next` (a single-iterator program, or an unnamed iterator type).
    if actual_next_name == "next" {
        for (name, (_, _args)) in &gen.functions {
            if is_next(name) {
                actual_next_name = name.to_string();
                break;
            }
        }
    }
    actual_next_name
}

/// Give back a pointer to the stack slot that holds the iterator of a `for` loop. When the loop
/// is over a variable that already lives in a slot, that slot is used; otherwise a new slot is
/// made in `block` and `iter_val` is stored in it.
///
/// # Errors
///
/// Returns an error when MLIR rejects an operation.
fn iterator_slot<'c>(
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    iterable: &Expr,
    iter_val: Value<'c, 'c>,
    iter_ty: Type<'c>,
) -> Result<Value<'c, 'c>, LowerError> {
    let ptr_ty = gen.ptr_ty;
    let i32_ty = gen.i32_ty;
    // A loop over a variable advances that variable, so what it has not handed out is
    // still there for its drop after the loop. Any other iterator gets a slot of its own.
    let in_place = match iterable {
        Expr::Identifier(id) if gen.allocs.contains(id.name.as_ref()) => {
            gen.env.get(&id.name).map(|(slot, _)| *slot)
        }
        _ => None,
    };
    let ptr_val = if let Some(slot) = in_place {
        slot
    } else {
        let c1_op_alloc = block.append_operation(
            OperationBuilder::new("llvm.mlir.constant", gen.loc())
                .add_results(&[i32_ty])
                .add_attributes(&[(
                    Identifier::new(gen.context, "value"),
                    IntegerAttribute::new(i32_ty, 1).into(),
                )])
                .build()?,
        );
        let c1_val_alloc = c1_op_alloc.result(0)?.into();
        let alloca_op = block.append_operation(
            OperationBuilder::new("llvm.alloca", gen.loc())
                .add_operands(&[c1_val_alloc])
                .add_results(&[ptr_ty])
                .add_attributes(&[(
                    Identifier::new(gen.context, "elem_type"),
                    melior::ir::attribute::TypeAttribute::new(iter_ty).into(),
                )])
                .build()?,
        );
        let ptr_val = alloca_op.result(0)?.into();
        block.append_operation(
            OperationBuilder::new("llvm.store", gen.loc())
                .add_operands(&[iter_val, ptr_val])
                .build()?,
        );
        ptr_val
    };
    Ok(ptr_val)
}

/// Build the call `next_name(&mut slot)`, where `tmp_iter_name` is the name the iterator's
/// stack slot is known by.
fn next_call_expr(next_name: String, tmp_iter_name: String) -> Expr {
    Expr::FunctionCall(FunctionCallExpr {
        name: next_name.into(),
        type_args: None,
        args: vec![Expr::Borrow(BorrowExpr {
            expr: Box::new(Expr::Identifier(IdentifierExpr {
                name: tmp_iter_name.into(),
                span: Span::default(),
            })),
            is_mut: true,
            span: Span::default(),
        })],
        span: Span::default(),
    })
}

/// End the condition block of an iterator loop: go to `body_block` when the option `opt_val`
/// that `next` returned has tag 0, and to `merge_block` otherwise.
///
/// # Errors
///
/// Returns an error when MLIR rejects an operation.
fn branch_if_some<'c>(
    gen: &MeliorGenerator<'c>,
    cond_block_end: melior::ir::BlockRef<'c, 'c>,
    opt_val: Value<'c, 'c>,
    body_block: melior::ir::BlockRef<'c, 'c>,
    merge_block: melior::ir::BlockRef<'c, 'c>,
) -> Result<(), LowerError> {
    let i32_ty = gen.i32_ty;
    let extract_tag_op = OperationBuilder::new("llvm.extractvalue", gen.loc())
        .add_operands(&[opt_val])
        .add_results(&[i32_ty])
        .add_attributes(&[(
            Identifier::new(gen.context, "position"),
            melior::ir::attribute::DenseI64ArrayAttribute::new(gen.context, &[0]).into(),
        )])
        .build()?;
    let tag_val = cond_block_end
        .append_operation(extract_tag_op)
        .result(0)?
        .into();

    let c1_op = cond_block_end.append_operation(
        OperationBuilder::new("arith.constant", gen.loc())
            .add_results(&[i32_ty])
            .add_attributes(&[(
                Identifier::new(gen.context, "value"),
                IntegerAttribute::new(i32_ty, 0).into(),
            )])
            .build()?,
    );
    let c1_val = c1_op.result(0)?.into();

    let cmpi_op = cond_block_end.append_operation(
        OperationBuilder::new("arith.cmpi", gen.loc())
            .add_operands(&[tag_val, c1_val])
            .add_results(&[gen.i1_ty])
            .add_attributes(&[(
                Identifier::new(gen.context, "predicate"),
                IntegerAttribute::new(gen.i64_ty, melior::dialect::arith::CmpiPredicate::Eq as i64)
                    .into(),
            )])
            .build()?,
    );
    let cond_val = cmpi_op.result(0)?.into();

    append_cond_br(gen, cond_block_end, cond_val, body_block, merge_block)
}

/// Work out the type of the value inside the option that `next` returns, from the option's
/// MLIR type `opt_ty`. An option with no value type in it is taken to hold an `i32`.
///
/// # Errors
///
/// Returns an error when the text of the value's type does not parse as an MLIR type.
///
/// # Panics
///
/// Panics if the text of `opt_ty` contains `(i32, ` but no `)`, which no option type does.
fn option_payload_type<'c>(
    gen: &MeliorGenerator<'c>,
    opt_ty: Type<'c>,
) -> Result<Type<'c>, LowerError> {
    let opt_ty_str = opt_ty.to_string();
    let mut payload_ty_str = if opt_ty_str.contains("(i32, ") {
        let start = opt_ty_str.find("(i32, ").unwrap() + 6;
        let end = opt_ty_str.rfind(')').unwrap();
        opt_ty_str[start..end].to_string()
    } else {
        "i32".to_string()
    };
    // A struct payload prints inside the option's type without its dialect prefix, and
    // parses only with it, as `match` lowering also finds.
    if ["struct", "ptr", "array"]
        .iter()
        .any(|p| payload_ty_str.starts_with(p))
    {
        payload_ty_str = format!("!llvm.{payload_ty_str}");
    }
    Type::parse(gen.context, &payload_ty_str).ok_or_else(|| {
        crate::codegen::lower::LowerError::ParseType("Type::parse failed".to_string())
    })
}

impl<'c> LowerToMelior<'c> for LoopStmt {
    type Output = Result<Option<melior::ir::BlockRef<'c, 'c>>, LowerError>;
    /// A block: a `let` in it hides an outer variable of the same name only until it ends.
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let depth = gen.open_block();
        let lowered = lower_loop_body(self, gen, block);
        gen.close_blocks_to(depth);
        lowered
    }
}

/// Lower a `loop { .. }`: the body runs again and again until a `break` or a `return`. Returns
/// the block that runs after the loop.
///
/// # Errors
///
/// Returns an error when a statement of the body cannot be lowered, or when MLIR rejects a jump.
///
/// # Panics
///
/// Panics if `block` is not inside a region, which would be a bug in the caller.
fn lower_loop_body<'c>(
    this: &LoopStmt,
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
) -> Result<Option<melior::ir::BlockRef<'c, 'c>>, LowerError> {
    let LoopStmt {
        invariants: _,
        body,
        span: _,
    } = this;

    let parent_region = block.parent_region().unwrap();
    let mut body_block = parent_region.append_block(melior::ir::Block::new(&[]));
    let loop_entry_block_ref = body_block; // separate variable
    let merge_block = parent_region.append_block(melior::ir::Block::new(&[]));

    let loop_entry_block = &raw const *loop_entry_block_ref;

    block.append_operation(
        OperationBuilder::new("cf.br", gen.loc())
            .add_successors(&[&*body_block])
            .build()?,
    );

    gen.break_blocks.push(&raw const *merge_block);
    gen.continue_blocks.push(loop_entry_block);

    let mut body_terminated = false;
    for stmt in body {
        if let Some(b) = gen.generate_statement(stmt, body_block)? {
            body_block = b;
        } else {
            body_terminated = true;
            break;
        }
    }

    gen.break_blocks.pop();
    gen.continue_blocks.pop();

    if !body_terminated {
        #[allow(
            unsafe_code,
            reason = "the block a `continue` jumps to is kept as a raw pointer, to match the generator's list"
        )]
        // SAFETY: the pointer was made above from `loop_entry_block_ref`, a local of this
        // function that is still alive, and nothing has written to it since.
        let entry_b: &melior::ir::Block<'c> = unsafe { &*loop_entry_block };
        body_block.append_operation(
            OperationBuilder::new("cf.br", gen.loc())
                .add_successors(&[entry_b])
                .build()?,
        );
    }

    Ok(Some(merge_block))
}

impl<'c> LowerToMelior<'c> for BreakStmt {
    type Output = Result<Option<melior::ir::BlockRef<'c, 'c>>, LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        if let Some(&break_ptr) = gen.break_blocks.last() {
            #[allow(
                unsafe_code,
                reason = "the generator keeps the block a `break` jumps to as a raw pointer"
            )]
            // SAFETY: the loop being lowered pushed this pointer, and it keeps the block the
            // pointer points at alive until it has lowered its whole body, this `break` included.
            let break_block: &melior::ir::Block<'c> = unsafe { &*break_ptr };
            block.append_operation(
                OperationBuilder::new("cf.br", gen.loc())
                    .add_successors(&[break_block])
                    .build()?,
            );
        } else {
            panic!("break outside of a loop");
        }
        Ok(None)
    }
}

impl<'c> LowerToMelior<'c> for ContinueStmt {
    type Output = Result<Option<melior::ir::BlockRef<'c, 'c>>, LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        if let Some(&continue_ptr) = gen.continue_blocks.last() {
            #[allow(
                unsafe_code,
                reason = "the generator keeps the block a `continue` jumps to as a raw pointer"
            )]
            // SAFETY: the loop being lowered pushed this pointer, and it keeps the block the
            // pointer points at alive until it has lowered its whole body, this `continue`
            // included.
            let continue_block: &melior::ir::Block<'c> = unsafe { &*continue_ptr };
            block.append_operation(
                OperationBuilder::new("cf.br", gen.loc())
                    .add_successors(&[continue_block])
                    .build()?,
            );
        } else {
            panic!("continue outside of a loop");
        }
        Ok(None)
    }
}
