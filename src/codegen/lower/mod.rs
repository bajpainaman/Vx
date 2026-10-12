//===- lower.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lowering logic translating Vx AST into MLIR operations using the Melior API.
//
//===----------------------------------------------------------------------===//

use super::{
    returns_through_slot, BinaryOp, BlockLike, ElementType, Expr, LogicalOp, MatchArm,
    MeliorGenerator, Pattern, Region, RegionLike, RelationalOp, Type, Value, ValueLike,
};
use melior::ir::{
    attribute::{
        ArrayAttribute, Attribute, DenseI32ArrayAttribute, FlatSymbolRefAttribute,
        IntegerAttribute, StringAttribute, TypeAttribute,
    },
    operation::OperationBuilder,
    Identifier,
};

use crate::syntax;
mod control_flow;
mod expr;
mod seam_cert;
mod stmt;
mod tensors;

/// A piece of Vx syntax that can generate MLIR code for itself.
pub trait LowerToMelior<'c> {
    /// What generating the code gives back, for example a value and the block to continue in.
    type Output;
    /// Generates the MLIR code for `self`, adding its operations to `block`.
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output;
}

/// Tells which MLIR `arith` operation implements a Vx operator.
pub trait MeliorOpInfo {
    /// The name of the MLIR operation for this operator. It depends on whether the operands
    /// are floats and whether they are unsigned.
    fn get_op_name(&self, is_float: bool, is_unsigned: bool) -> &'static str;
    /// The comparison code that `arith.cmpi` or `arith.cmpf` takes, or `None` when the
    /// operator is not a comparison.
    fn get_predicate(&self, is_float: bool, is_unsigned: bool) -> Option<i64>;
}

impl MeliorOpInfo for BinaryOp {
    fn get_op_name(&self, is_float: bool, is_unsigned: bool) -> &'static str {
        match self {
            BinaryOp::Add => {
                if is_float {
                    "arith.addf"
                } else {
                    "arith.addi"
                }
            }
            BinaryOp::Sub => {
                if is_float {
                    "arith.subf"
                } else {
                    "arith.subi"
                }
            }
            BinaryOp::Mul | BinaryOp::MatMul => {
                if is_float {
                    "arith.mulf"
                } else {
                    "arith.muli"
                }
            }
            BinaryOp::Div => {
                if is_float {
                    "arith.divf"
                } else if is_unsigned {
                    "arith.divui"
                } else {
                    "arith.divsi"
                }
            }
            BinaryOp::Rem => {
                if is_float {
                    "arith.remf"
                } else if is_unsigned {
                    "arith.remui"
                } else {
                    "arith.remsi"
                }
            }
            // One name each: these are bit patterns, so neither signedness nor floatness
            // enters into the choice. A float operand is refused by the checker (E3030).
            BinaryOp::BitAnd => "arith.andi",
            BinaryOp::BitOr => "arith.ori",
            BinaryOp::BitXor => "arith.xori",
            BinaryOp::Shl => "arith.shli",
            // Arithmetic for a signed operand and logical for an unsigned one: the sign bit
            // is copied only when there is a sign bit to copy.
            BinaryOp::Shr => {
                if is_unsigned {
                    "arith.shrui"
                } else {
                    "arith.shrsi"
                }
            }
        }
    }

    fn get_predicate(&self, _is_float: bool, _is_unsigned: bool) -> Option<i64> {
        None
    }
}

impl MeliorOpInfo for RelationalOp {
    fn get_op_name(&self, is_float: bool, _is_unsigned: bool) -> &'static str {
        if is_float {
            "arith.cmpf"
        } else {
            "arith.cmpi"
        }
    }

    fn get_predicate(&self, is_float: bool, is_unsigned: bool) -> Option<i64> {
        use melior::dialect::arith::{CmpfPredicate as F, CmpiPredicate as I};
        Some(if is_float {
            (match self {
                RelationalOp::Eq => F::Oeq,
                RelationalOp::Gt => F::Ogt,
                RelationalOp::Ge => F::Oge,
                RelationalOp::Lt => F::Olt,
                RelationalOp::Le => F::Ole,
                // Unordered: `!=` is the negation of `==`, so a NaN on either side makes it true.
                RelationalOp::NotEq => F::Une,
            }) as i64
        } else if is_unsigned {
            (match self {
                RelationalOp::Eq => I::Eq,
                RelationalOp::NotEq => I::Ne,
                RelationalOp::Lt => I::Ult,
                RelationalOp::Le => I::Ule,
                RelationalOp::Gt => I::Ugt,
                RelationalOp::Ge => I::Uge,
            }) as i64
        } else {
            (match self {
                RelationalOp::Eq => I::Eq,
                RelationalOp::NotEq => I::Ne,
                RelationalOp::Lt => I::Slt,
                RelationalOp::Le => I::Sle,
                RelationalOp::Gt => I::Sgt,
                RelationalOp::Ge => I::Sge,
            }) as i64
        })
    }
}

impl MeliorOpInfo for LogicalOp {
    fn get_op_name(&self, _is_float: bool, _is_unsigned: bool) -> &'static str {
        match self {
            LogicalOp::And => "arith.andi",
            LogicalOp::Or => "arith.ori",
        }
    }

    fn get_predicate(&self, _is_float: bool, _is_unsigned: bool) -> Option<i64> {
        None
    }
}

/// A block to keep appending to after a statement closed the current one.
///
/// `return`, `break` and `continue` terminate their block, and MLIR admits no
/// operation after a terminator. Whatever follows them in the source is
/// unreachable rather than illegal, so it gets a block of its own: the verifier
/// checks dominance only for blocks reachable from the entry, and a block with
/// no predecessors is not one. The alternative -- piling the rest of the
/// function on after a `func.return` -- does not compile at all.
///
/// # Panics
///
/// Panics if `block` is not inside a region. Every block that code is generated into is.
pub(crate) fn dead_continuation<'c>(
    block: melior::ir::BlockRef<'c, 'c>,
) -> melior::ir::BlockRef<'c, 'c> {
    block
        .parent_region()
        .expect("a block being lowered into always sits in a region")
        .append_block(melior::ir::Block::new(&[]))
}

/// Runtime dispatch id for a topology. Thin delegate to the single source of truth in
/// `arch`, which co-locates this with the memory-space mapping so the two cannot diverge.
pub(crate) fn topology_to_i32(top: &syntax::Topology) -> i32 {
    crate::arch::topology_dispatch_id(top)
}

/// The element type of a `memref` or `tensor` type written as MLIR text: `f32` for
/// `memref<4x4xf32>`.
///
/// # Errors
///
/// Returns an error if the text is not a `memref<..>` or `tensor<..>` type, or if its
/// element type is not one of `bf16`, `f16`, `f32`, `f64`, `i32`, `i64` and `i1`.
pub(crate) fn extract_mlir_element_type(ty_str: &str) -> Result<&'static str, String> {
    let mut inner = ty_str;
    if inner.starts_with("memref<") && inner.ends_with('>') {
        inner = &inner[7..inner.len() - 1];
    } else if inner.starts_with("tensor<") && inner.ends_with('>') {
        inner = &inner[7..inner.len() - 1];
    } else {
        return Err(format!("Unsupported MLIR element type in: {ty_str}"));
    }

    if let Some(idx) = inner.find(',') {
        inner = &inner[..idx];
    }

    if let Some(idx) = inner.rfind('x') {
        inner = &inner[idx + 1..];
    }

    inner = inner.trim();

    match inner {
        "bf16" => Ok("bf16"),
        "f16" => Ok("f16"),
        "f32" => Ok("f32"),
        "f64" => Ok("f64"),
        "i32" => Ok("i32"),
        "i64" => Ok("i64"),
        "i1" => Ok("i1"),
        _ => Err(format!("Unsupported MLIR element type in: {ty_str}")),
    }
}

/// Declares the Enzyme wrapper function that differentiates `target_fn`, the first time it is
/// needed, and returns its name. `base_name` says which kind of derivative is wanted.
///
/// # Panics
///
/// Panics if MLIR refuses to build the `func.func` declaration, which would be a compiler bug.
pub(crate) fn emit_enzyme_decl<'c>(
    gen: &mut MeliorGenerator<'c>,
    base_name: &str,
    target_fn: &str,
    arg_tys: &[Type<'c>],
    ret_ty: Type<'c>,
) -> String {
    let enzyme_name = crate::codegen::enzyme_wrapper_name(base_name == "fwddiff", target_fn);
    if !gen.functions.contains_key(&*enzyme_name) && gen.enzyme_decls.insert(enzyme_name.clone()) {
        let func_type = melior::ir::r#type::FunctionType::new(gen.context, arg_tys, &[ret_ty]);
        let _name_attr = StringAttribute::new(gen.context, &enzyme_name);
        let _type_attr = TypeAttribute::new(func_type.into());

        let region = melior::ir::Region::new();
        let func_op = OperationBuilder::new("func.func", gen.loc())
            .add_attributes(&[
                (
                    Identifier::new(gen.context, "sym_name"),
                    StringAttribute::new(gen.context, &enzyme_name).into(),
                ),
                (
                    Identifier::new(gen.context, "function_type"),
                    TypeAttribute::new(func_type.into()).into(),
                ),
                (
                    Identifier::new(gen.context, "sym_visibility"),
                    StringAttribute::new(gen.context, "private").into(),
                ),
            ])
            .add_regions([region])
            .build()
            .expect("Failed to build enzyme decl");

        gen.module.body().append_operation(func_op);
    }
    enzyme_name
}

/// Lower one arm's body into `b` and branch to the merge block, carrying the arm's tail value
/// when the match is in value position.
///
/// `value_ty` is `Some` exactly when the merge block takes an argument. An arm that diverges
/// (its body ends in `return`) branches nowhere and contributes no value, which is what lets a
/// match whose arms all return type-check and lower.
///
/// # Errors
///
/// Returns an error if a statement of the body fails to generate code, if the match needs a
/// value and the arm ends in a statement, or if the arm's value has a different type from the
/// one the match gives.
fn lower_match_arm_body<'c>(
    gen: &mut MeliorGenerator<'c>,
    body: &[syntax::Statement],
    mut b: melior::ir::BlockRef<'c, 'c>,
    merge_block: melior::ir::BlockRef<'c, 'c>,
    value_ty: Option<melior::ir::Type<'c>>,
) -> Result<(), LowerError> {
    let mut terminated = false;
    let mut tail = None;
    for (i, stmt) in body.iter().enumerate() {
        let is_last = i == body.len() - 1;
        if is_last && value_ty.is_some() {
            if let syntax::Statement::ExprStmt(syntax::stmt::ExprStmtStmt {
                expr,
                has_semi: false,
                ..
            }) = stmt
            {
                let (val, ty, nb) = gen.generate_expr(expr, b)?;
                b = nb;
                b = gen.run_queued_drops(b)?;
                tail = Some((val, ty));
                continue;
            }
        }
        if let Some(nb) = gen.generate_statement(stmt, b)? {
            b = nb
        } else {
            terminated = true;
            break;
        }
    }
    if terminated {
        return Ok(());
    }

    let mut operands = Vec::new();
    if let Some(want) = value_ty {
        // Refuse rather than emit a branch whose operands do not match the block's arguments.
        // This used to be a hard-coded `arith.constant 0` returned in place of the match, so a
        // value-producing match silently evaluated to zero.
        let Some((val, got)) = tail else {
            return Err(LowerError::ParseType(
                "a `match` arm in value position ends in a statement, so it produces no value"
                    .to_string(),
            ));
        };
        if got != want {
            return Err(LowerError::ParseType(format!(
                "`match` arms disagree on value type: expected {want}, this arm yields {got}"
            )));
        }
        operands.push(val);
    }
    b.append_operation(
        OperationBuilder::new("cf.br", gen.loc())
            .add_operands(&operands)
            .add_successors(&[&*merge_block])
            .build()?,
    );
    Ok(())
}

/// The lowered type a matched variant actually declares for its payload.
///
/// The slot is sized for the widest variant, so a match that bound its payload at the
/// slot type handed back a value too wide: an `i8` -3 extended into an `i64` slot reads
/// as 253 unless it is cut back to the width the variant declares.
///
/// The pattern carries the enum's parameter names (`Result<T, E>`) and the lowered struct
/// carries the arguments in its mangled name (`Result_i64_i8`), which together give the
/// substitution. That name is split on `_`, so the mapping is used only when the pieces
/// come out as the expected number of plain scalar types. Anything else -- a nested
/// generic, a struct whose own name has an underscore -- returns `None`, and the caller
/// keeps the slot type it used before.
fn declared_payload_ty(
    gen: &MeliorGenerator<'_>,
    pattern_enum: &str,
    variant: &str,
    position: usize,
    match_ty_text: &str,
) -> Option<String> {
    let (base, params) = match pattern_enum.split_once('<') {
        Some((b, rest)) => (
            b,
            rest.trim_end_matches('>')
                .split(',')
                .map(|p| p.trim().to_string())
                .collect::<Vec<_>>(),
        ),
        None => (pattern_enum, Vec::new()),
    };

    let declared = gen
        .enums
        .get(&crate::symbol::Symbol::from(base))?
        .iter()
        .find(|(v, _)| v.as_ref() == variant)?
        .1
        .as_ref()?
        .get(position)?
        .clone();

    let mut mapping = std::collections::HashMap::new();
    if !params.is_empty() {
        let quoted = match_ty_text.split('"').nth(1)?;
        let args: Vec<&str> = quoted
            .strip_prefix(&format!("{base}_"))?
            .split('_')
            .collect();
        if args.len() != params.len() || !args.iter().all(|a| is_plain_scalar(a)) {
            return None;
        }
        for (p, a) in params.iter().zip(args) {
            mapping.insert(
                crate::symbol::Symbol::from(p.as_str()),
                parse_scalar_type(a)?,
            );
        }
    }

    gen.lower_type_str(&declared.substitute(&mapping)).ok()
}

/// A scalar type spelled plainly, so a mangled name can be split on `_` without guessing.
fn is_plain_scalar(text: &str) -> bool {
    parse_scalar_type(text).is_some()
}

/// What the steps of `lower_usize_checked` share: the block they add operations to, and the
/// location and types every one of those operations uses.
struct UsizeCheck<'c, 'a> {
    block: &'a melior::ir::BlockRef<'c, 'c>,
    loc: melior::ir::Location<'c>,
    ctx: &'c melior::Context,
    i64_ty: melior::ir::Type<'c>,
    i1_ty: melior::ir::Type<'c>,
}

impl<'c> UsizeCheck<'c, '_> {
    /// Adds `op` to the block and returns its first result.
    ///
    /// # Errors
    ///
    /// Returns an error if `op` has no result.
    fn append(
        &self,
        op: melior::ir::Operation<'c>,
    ) -> Result<melior::ir::Value<'c, 'c>, LowerError> {
        Ok(self.block.append_operation(op).result(0)?.into())
    }

    /// The `i64` constant `v`.
    ///
    /// # Errors
    ///
    /// Returns an error if MLIR refuses to build the constant.
    fn constant(&self, v: i64) -> Result<melior::ir::Value<'c, 'c>, LowerError> {
        self.append(
            OperationBuilder::new("arith.constant", self.loc)
                .add_results(&[self.i64_ty])
                .add_attributes(&[(
                    Identifier::new(self.ctx, "value"),
                    IntegerAttribute::new(self.i64_ty, v).into(),
                )])
                .build()?,
        )
    }

    /// The two-operand operation `name` on `x` and `y`, with a result of type `ty`.
    ///
    /// # Errors
    ///
    /// Returns an error if MLIR refuses to build the operation.
    fn binary(
        &self,
        name: &str,
        x: melior::ir::Value<'c, 'c>,
        y: melior::ir::Value<'c, 'c>,
        ty: melior::ir::Type<'c>,
    ) -> Result<melior::ir::Value<'c, 'c>, LowerError> {
        self.append(
            OperationBuilder::new(name, self.loc)
                .add_operands(&[x, y])
                .add_results(&[ty])
                .build()?,
        )
    }

    /// An `arith.cmpi` of `x` and `y`. The predicates used here are: eq = 0, sge = 5, ult = 6,
    /// ule = 7, uge = 9.
    ///
    /// # Errors
    ///
    /// Returns an error if MLIR refuses to build the comparison.
    fn compare(
        &self,
        pred: i64,
        x: melior::ir::Value<'c, 'c>,
        y: melior::ir::Value<'c, 'c>,
    ) -> Result<melior::ir::Value<'c, 'c>, LowerError> {
        self.append(
            OperationBuilder::new("arith.cmpi", self.loc)
                .add_operands(&[x, y])
                .add_results(&[self.i1_ty])
                .add_attributes(&[(
                    Identifier::new(self.ctx, "predicate"),
                    IntegerAttribute::new(self.i64_ty, pred).into(),
                )])
                .build()?,
        )
    }

    /// An `i1` value that is true when `a op b` stays inside the range of `usize`. `op` is one
    /// of `+`, `-`, `*` and `<<`; any other operator is checked as `<<`.
    ///
    /// # Errors
    ///
    /// Returns an error if MLIR refuses to build one of the operations of the check.
    fn operands_fit(
        &self,
        op: &BinaryOp,
        a: melior::ir::Value<'c, 'c>,
        b: melior::ir::Value<'c, 'c>,
    ) -> Result<melior::ir::Value<'c, 'c>, LowerError> {
        let (i64_ty, i1_ty) = (self.i64_ty, self.i1_ty);
        Ok(match op {
            // a + b fits when a <= MAX - b.
            BinaryOp::Add => {
                let room = self.binary("arith.subi", self.constant(i64::MAX)?, b, i64_ty)?;
                self.compare(7, a, room)?
            }
            BinaryOp::Sub => self.compare(9, a, b)?,
            // The full product has no high word, and its low word has the top bit clear.
            BinaryOp::Mul => {
                let product = self.block.append_operation(
                    OperationBuilder::new("arith.mului_extended", self.loc)
                        .add_operands(&[a, b])
                        .add_results(&[i64_ty, i64_ty])
                        .build()?,
                );
                let (low, high) = (product.result(0)?.into(), product.result(1)?.into());
                let zero = self.constant(0)?;
                let no_high = self.compare(0, high, zero)?;
                let low_fits = self.compare(5, low, zero)?;
                self.binary("arith.andi", no_high, low_fits, i1_ty)?
            }
            // a << b fits when b < 63 and a >> (63 - b) is 0. The shift used for the check is
            // clamped to 63, so the check itself never shifts by the width or more.
            _ => {
                let width = self.constant(63)?;
                let shift_fits = self.compare(6, b, width)?;
                let clamped = self.binary("arith.minui", b, width, i64_ty)?;
                let back = self.binary("arith.subi", width, clamped, i64_ty)?;
                let lost = self.binary("arith.shrui", a, back, i64_ty)?;
                let nothing_lost = self.compare(0, lost, self.constant(0)?)?;
                self.binary("arith.andi", shift_fits, nothing_lost, i1_ty)?
            }
        })
    }
}

/// `a op b` on `usize`, for the operators that can leave its range: `+`, `-`, `*` and `<<`.
/// The operands are checked first and a failed check traps through `cf.assert`; the operation
/// itself then never wraps and carries `overflow<nsw, nuw>`. Checking the result instead would
/// not work: an overflowing flagged operation gives poison, and LLVM may delete a branch on it.
/// `None` for any other operator, which cannot leave the range.
///
/// # Errors
///
/// Returns an error if MLIR refuses to build one of the operations, or if the overflow flags
/// do not parse.
#[allow(
    clippy::trivially_copy_pass_by_ref,
    reason = "callers are in other files; they pass the block by reference"
)]
pub(crate) fn lower_usize_checked<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: &melior::ir::BlockRef<'c, 'c>,
    op: &BinaryOp,
    a: melior::ir::Value<'c, 'c>,
    b: melior::ir::Value<'c, 'c>,
) -> Result<Option<melior::ir::Value<'c, 'c>>, LowerError> {
    let (name, failure) = match op {
        BinaryOp::Add => ("arith.addi", "usize addition overflowed"),
        BinaryOp::Sub => ("arith.subi", "usize subtraction went below zero"),
        BinaryOp::Mul => ("arith.muli", "usize multiplication overflowed"),
        BinaryOp::Shl => ("arith.shli", "usize left shift overflowed"),
        _ => return Ok(None),
    };
    let i64_ty = gen.i64_ty;
    let ctx = gen.context;
    let loc = gen.loc();
    let check = UsizeCheck {
        block,
        loc,
        ctx,
        i64_ty,
        i1_ty: gen.i1_ty,
    };
    let fits = check.operands_fit(op, a, b)?;
    block.append_operation(
        OperationBuilder::new("cf.assert", loc)
            .add_operands(&[fits])
            .add_attributes(&[(
                Identifier::new(ctx, "msg"),
                StringAttribute::new(ctx, failure).into(),
            )])
            .build()?,
    );
    let flags = Attribute::parse(ctx, "#arith.overflow<nsw, nuw>")
        .ok_or_else(|| LowerError::from("the overflow flags did not parse".to_string()))?;
    let result = check.append(
        OperationBuilder::new(name, loc)
            .add_operands(&[a, b])
            .add_results(&[i64_ty])
            .add_attributes(&[(Identifier::new(ctx, "overflowFlags"), flags)])
            .build()?,
    )?;
    Ok(Some(result))
}

/// One type argument of a monomorphized name, `usize` in `Vec<usize>`: a number is a
/// const-generic argument, and anything else is read by the type parser.
pub(crate) fn parse_type_arg(text: &str) -> syntax::Type {
    let text = text.trim();
    if !text.is_empty() && text.chars().all(|c| c.is_ascii_digit()) {
        return syntax::Type::Const(Box::new(syntax::Expr::Number(
            syntax::expr::NumberExpr::new(text.to_string(), None, syntax::Span::default()),
        )));
    }
    let tokens = crate::lexer::Lexer::new(text).tokenize();
    crate::parser::Parser::new(&tokens, text)
        .parse_type()
        .unwrap_or_else(|_| syntax::Type::Struct(text.to_string().into(), None))
}

/// The syntax type a plain scalar name stands for.
fn parse_scalar_type(text: &str) -> Option<syntax::Type> {
    let mut lexer = crate::lexer::Lexer::new(text);
    let tokens = lexer.tokenize();
    let mut parser = crate::parser::Parser::new(&tokens, text);
    match parser.parse_type() {
        Ok(t @ syntax::Type::Scalar(_)) => Some(t),
        _ => None,
    }
}

/// Ends a chain of `match` arms. `block` is where control lands when no arm matched, and it
/// branches to the merge block.
///
/// # Errors
///
/// Returns an error if MLIR refuses to build one of the operations.
fn end_match_chain<'c>(
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    merge_block: melior::ir::BlockRef<'c, 'c>,
    value_ty: Option<melior::ir::Type<'c>>,
) -> Result<(), LowerError> {
    // The chain always ends in a fall-through block. In value position the checker has
    // already established that some arm matches (a wildcard, or every variant of the
    // scrutinee's enum), so this edge is unreachable -- but it still has to satisfy the
    // merge block's argument, and an undef says "unreachable" where a zero would be
    // indistinguishable from a real result.
    let mut operands = Vec::new();
    if let Some(ty) = value_ty {
        let undef = block.append_operation(
            OperationBuilder::new("llvm.mlir.undef", gen.loc())
                .add_results(&[ty])
                .build()?,
        );
        operands.push(undef.result(0)?.into());
    }
    block.append_operation(
        OperationBuilder::new("cf.br", gen.loc())
            .add_operands(&operands)
            .add_successors(&[&*merge_block])
            .build()?,
    );
    Ok(())
}

/// An `i1` value that is true when the matched enum value holds the variant `variant_name`.
/// The tag is the first field of the enum value, or the value itself when the enum is a
/// plain `i32`.
///
/// # Errors
///
/// Returns an error if MLIR refuses to build one of the operations.
///
/// # Panics
///
/// Panics if a variant's position in its enum does not fit in an `i64`, which cannot happen
/// for an enum a program can declare.
fn variant_tag_condition<'c>(
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    variant_name: &str,
    match_val: melior::ir::Value<'c, 'c>,
    match_ty: melior::ir::Type<'c>,
) -> Result<melior::ir::Value<'c, 'c>, LowerError> {
    let i32_ty = melior::ir::r#type::IntegerType::new(gen.context, 32).into();
    let mut tag_val = 0;
    for enum_def in gen.enums.values() {
        for (i, v) in enum_def.iter().enumerate() {
            if *v.0 == *variant_name {
                tag_val = i64::try_from(i).expect("an enum has far fewer variants than i64 holds");
                break;
            }
        }
    }

    let tag_op = block.append_operation(
        OperationBuilder::new("arith.constant", gen.loc())
            .add_results(&[i32_ty])
            .add_attributes(&[(
                Identifier::new(gen.context, "value"),
                IntegerAttribute::new(i32_ty, tag_val).into(),
            )])
            .build()?,
    );
    let tag = tag_op.result(0)?.into();

    let actual_tag = if match_ty.to_string() == "i32" {
        match_val
    } else {
        let extract_tag_op = block.append_operation(
            OperationBuilder::new("llvm.extractvalue", gen.loc())
                .add_operands(&[match_val])
                .add_results(&[i32_ty])
                .add_attributes(&[(
                    Identifier::new(gen.context, "position"),
                    melior::ir::attribute::DenseI64ArrayAttribute::new(gen.context, &[0]).into(),
                )])
                .build()?,
        );
        extract_tag_op.result(0)?.into()
    };

    let cmp_op = block.append_operation(
        OperationBuilder::new("arith.cmpi", gen.loc())
            .add_operands(&[actual_tag, tag])
            .add_results(&[melior::ir::r#type::IntegerType::new(gen.context, 1).into()])
            .add_attributes(&[(
                Identifier::new(gen.context, "predicate"),
                IntegerAttribute::new(
                    melior::ir::r#type::IntegerType::new(gen.context, 64).into(),
                    0, // eq
                )
                .into(),
            )])
            .build()?,
    );
    Ok(cmp_op.result(0)?.into())
}

/// An `i1` value that is true when the matched integer equals the number `lit`. It is the
/// same `arith.cmpi eq` as for an enum variant, but against `match_val` itself, with no tag
/// to read first.
///
/// # Errors
///
/// Returns an error if MLIR refuses to build one of the operations.
///
/// # Panics
///
/// Panics if `lit` is not a number.
fn literal_condition<'c>(
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    lit: &Expr,
    match_val: melior::ir::Value<'c, 'c>,
    match_ty: melior::ir::Type<'c>,
) -> Result<melior::ir::Value<'c, 'c>, LowerError> {
    let lit_val = match lit {
        Expr::Number(n) => n.value.as_ref().parse::<i64>().unwrap_or(0),
        other => panic!("Unsupported literal pattern in codegen: {other:?}"),
    };
    let const_op = block.append_operation(
        OperationBuilder::new("arith.constant", gen.loc())
            .add_results(&[match_ty])
            .add_attributes(&[(
                Identifier::new(gen.context, "value"),
                IntegerAttribute::new(match_ty, lit_val).into(),
            )])
            .build()?,
    );
    let const_val = const_op.result(0)?.into();
    let cmp_op = block.append_operation(
        OperationBuilder::new("arith.cmpi", gen.loc())
            .add_operands(&[match_val, const_val])
            .add_results(&[melior::ir::r#type::IntegerType::new(gen.context, 1).into()])
            .add_attributes(&[(
                Identifier::new(gen.context, "predicate"),
                IntegerAttribute::new(
                    melior::ir::r#type::IntegerType::new(gen.context, 64).into(),
                    0, // eq
                )
                .into(),
            )])
            .build()?,
    );
    Ok(cmp_op.result(0)?.into())
}

/// Reads the payload at `position` out of the matched enum value. Returns the text of the
/// slot's type, the value read, and its type.
///
/// # Errors
///
/// Returns an error if the slot's type does not parse or MLIR refuses to build the operation.
///
/// # Panics
///
/// Panics if `position` does not fit in an `i64`, which cannot happen for a variant a program
/// can declare.
fn extract_payload_slot<'c>(
    gen: &MeliorGenerator<'c>,
    then_block: melior::ir::BlockRef<'c, 'c>,
    match_val: melior::ir::Value<'c, 'c>,
    position: usize,
    slot_text: &str,
) -> Result<(String, melior::ir::Value<'c, 'c>, melior::ir::Type<'c>), LowerError> {
    let mut payload_ty_str = slot_text.to_string();
    if payload_ty_str.starts_with("struct<")
        || payload_ty_str.starts_with("ptr")
        || payload_ty_str.starts_with("func")
        || payload_ty_str.starts_with("array")
    {
        payload_ty_str = format!("!llvm.{payload_ty_str}");
    }
    let payload_ty = melior::ir::Type::parse(gen.context, &payload_ty_str).ok_or_else(|| {
        crate::codegen::lower::LowerError::ParseType("Type::parse failed".to_string())
    })?;
    // Field 0 of the enum value is the tag, so the payloads start at field 1.
    let field = i64::try_from(position).expect("a variant has far fewer fields than i64 holds") + 1;
    let extract_payload_op = OperationBuilder::new("llvm.extractvalue", gen.loc())
        .add_operands(&[match_val])
        .add_results(&[payload_ty])
        .add_attributes(&[(
            Identifier::new(gen.context, "position"),
            melior::ir::attribute::DenseI64ArrayAttribute::new(gen.context, &[field]).into(),
        )])
        .build()?;
    let payload_val: melior::ir::Value<'_, '_> = then_block
        .append_operation(extract_payload_op)
        .result(0)?
        .into();
    Ok((payload_ty_str, payload_val, payload_ty))
}

/// Cuts a payload back to the integer type `narrower` when there is one, and returns the
/// payload unchanged otherwise.
///
/// # Errors
///
/// Returns an error if `narrower` does not parse or MLIR refuses to build the operation.
fn truncate_payload<'c>(
    gen: &MeliorGenerator<'c>,
    then_block: melior::ir::BlockRef<'c, 'c>,
    narrower: Option<String>,
    (payload_val, payload_ty): (melior::ir::Value<'c, 'c>, melior::ir::Type<'c>),
) -> Result<(melior::ir::Value<'c, 'c>, melior::ir::Type<'c>), LowerError> {
    Ok(match narrower {
        Some(text) => {
            let ty = melior::ir::Type::parse(gen.context, &text).ok_or_else(|| {
                crate::codegen::lower::LowerError::ParseType("Type::parse failed".to_string())
            })?;
            let trunc = OperationBuilder::new("arith.trunci", gen.loc())
                .add_operands(&[payload_val])
                .add_results(&[ty])
                .build()?;
            (then_block.append_operation(trunc).result(0)?.into(), ty)
        }
        None => (payload_val, payload_ty),
    })
}

/// Turns a payload kept as integer bits back into the float type `declared_float` when there
/// is one, and returns the payload unchanged otherwise.
///
/// # Errors
///
/// Returns an error if the float type has no known width, a type does not parse, or MLIR
/// refuses to build an operation.
fn payload_float_from_bits<'c>(
    gen: &MeliorGenerator<'c>,
    then_block: melior::ir::BlockRef<'c, 'c>,
    declared_float: Option<String>,
    (payload_val, payload_ty): (melior::ir::Value<'c, 'c>, melior::ir::Type<'c>),
) -> Result<(melior::ir::Value<'c, 'c>, melior::ir::Type<'c>), LowerError> {
    Ok(match declared_float {
        Some(text) => {
            let bits = crate::codegen::generator::scalar_type_bits(&text)
                .ok_or_else(|| LowerError::from(format!("a float of no width: {text}")))?;
            let mut val = payload_val;
            if payload_ty.to_string() != format!("i{bits}") {
                let as_int = melior::ir::Type::parse(gen.context, &format!("i{bits}"))
                    .ok_or_else(|| LowerError::ParseType("Type::parse failed".into()))?;
                let trunc = OperationBuilder::new("arith.trunci", gen.loc())
                    .add_operands(&[val])
                    .add_results(&[as_int])
                    .build()?;
                val = then_block.append_operation(trunc).result(0)?.into();
            }
            let ty = melior::ir::Type::parse(gen.context, &text)
                .ok_or_else(|| LowerError::ParseType("Type::parse failed".into()))?;
            let cast = OperationBuilder::new("arith.bitcast", gen.loc())
                .add_operands(&[val])
                .add_results(&[ty])
                .build()?;
            (then_block.append_operation(cast).result(0)?.into(), ty)
        }
        None => (payload_val, payload_ty),
    })
}

/// Binds each name in an enum-variant pattern to the payload value at its position, read from
/// that position's slot in `then_block`. `_` binds nothing, and a pattern that is not an enum
/// variant with payloads binds nothing at all.
///
/// # Errors
///
/// Returns an error if a payload's type does not parse or MLIR refuses to build an operation.
fn bind_variant_payloads<'c>(
    gen: &mut MeliorGenerator<'c>,
    then_block: melior::ir::BlockRef<'c, 'c>,
    pattern: &Pattern,
    match_val: melior::ir::Value<'c, 'c>,
    match_ty: melior::ir::Type<'c>,
) -> Result<(), LowerError> {
    if let Pattern::EnumVariant(en, vn, Some(payloads)) = pattern {
        let opt_ty_str = match_ty.to_string();
        let slots = crate::codegen::lower::expr::enum_payload_slot_texts(&opt_ty_str);
        for (position, pattern) in payloads.iter().enumerate() {
            let Pattern::Identifier(name) = pattern else {
                continue;
            };
            let Some(slot_text) = slots.get(position) else {
                continue;
            };
            let (payload_ty_str, payload_val, payload_ty) =
                extract_payload_slot(gen, then_block, match_val, position, slot_text)?;

            // The slot may be wider than this variant declares, so cut the value back
            // to its own type before the arm body sees it.
            let narrower =
                declared_payload_ty(gen, en.as_ref(), vn.as_ref(), position, &opt_ty_str).filter(
                    |text| crate::codegen::lower::expr::is_narrower_int(&payload_ty_str, text),
                );
            let payload = truncate_payload(gen, then_block, narrower, (payload_val, payload_ty))?;
            // A float kept in an integer slot comes back out from its bits.
            let declared_float =
                declared_payload_ty(gen, en.as_ref(), vn.as_ref(), position, &opt_ty_str).filter(
                    |text| {
                        crate::codegen::generator::is_float_text(text)
                            && payload_ty_str.starts_with('i')
                    },
                );
            let (payload_val, payload_ty) =
                payload_float_from_bits(gen, then_block, declared_float, payload)?;
            gen.note_shadow(name.as_ref());
            gen.env
                .insert(name.to_string().into(), (payload_val, payload_ty));
        }
    }
    Ok(())
}

/// Generates the code for the arms of a `match`, one after another. Each arm tests the matched
/// value, runs its body when the test passes, and goes on to the next arm otherwise. Every arm
/// ends by branching to `merge_block`, with the arm's value when `value_ty` is `Some`.
/// Returns `block`.
///
/// # Errors
///
/// Returns an error if an arm's body fails to generate code, a payload's type does not parse,
/// or MLIR refuses to build an operation.
///
/// # Panics
///
/// Panics if `block` is not inside a region, or if an arm's pattern is not a wildcard, an enum
/// variant or a number.
#[allow(
    clippy::too_many_arguments,
    reason = "callers are in other files; they pass these seven values one by one"
)]
pub fn generate_match_chain<'c>(
    gen: &mut MeliorGenerator<'c>,
    arms: &[MatchArm],
    match_val: melior::ir::Value<'c, 'c>,
    match_ty: melior::ir::Type<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    merge_block: melior::ir::BlockRef<'c, 'c>,
    value_ty: Option<melior::ir::Type<'c>>,
) -> Result<melior::ir::BlockRef<'c, 'c>, LowerError> {
    if arms.is_empty() {
        end_match_chain(gen, block, merge_block, value_ty)?;
        return Ok(block);
    }

    let arm = &arms[0];

    // Each arm is a block: its pattern's names and its `let`s end with it.
    let depth = gen.open_block();
    if let Pattern::Wildcard = arm.pattern {
        lower_match_arm_body(gen, &arm.body, block, merge_block, value_ty)?;
        gen.close_blocks_to(depth);
        return Ok(block);
    }

    let parent_region = block.parent_region().unwrap();
    let then_block = parent_region.append_block(melior::ir::Block::new(&[]));
    let else_block = parent_region.append_block(melior::ir::Block::new(&[]));

    let cond_val = match &arm.pattern {
        Pattern::EnumVariant(_, variant_name, _) => {
            variant_tag_condition(gen, block, variant_name.as_ref(), match_val, match_ty)?
        }
        // A scalar literal arm (`0 => …`, `1 => …`) over an integer scrutinee.
        Pattern::Literal(lit) => literal_condition(gen, block, lit, match_val, match_ty)?,
        _ => panic!("Unsupported pattern in codegen"),
    };

    block.append_operation(
        OperationBuilder::new("cf.cond_br", gen.loc())
            .add_operands(&[cond_val])
            .add_successors(&[&*then_block, &*else_block])
            .add_attributes(&[(
                Identifier::new(gen.context, "operandSegmentSizes"),
                melior::ir::attribute::DenseI32ArrayAttribute::new(gen.context, &[1, 0, 0]).into(),
            )])
            .build()?,
    );

    bind_variant_payloads(gen, then_block, &arm.pattern, match_val, match_ty)?;

    lower_match_arm_body(gen, &arm.body, then_block, merge_block, value_ty)?;
    gen.close_blocks_to(depth);

    generate_match_chain(
        gen,
        &arms[1..],
        match_val,
        match_ty,
        else_block,
        merge_block,
        value_ty,
    )?;

    Ok(block)
}

/// Puts a closure's captured values on the stack, so that the function that runs the closure
/// can take a pointer to them. Returns that pointer and the name of that function.
///
/// # Errors
///
/// Returns an error if MLIR refuses to build one of the operations.
///
/// # Panics
///
/// Panics if splitting the closure's type name gives no first piece, which a split never does.
fn spill_closure_env<'c>(
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    closure_val: Value<'c, 'c>,
    closure_ty: Type<'c>,
) -> Result<(Value<'c, 'c>, String), LowerError> {
    let ptr_ty = gen.ptr_ty;
    let i32_ty = gen.i32_ty;
    let c1_op = OperationBuilder::new("arith.constant", gen.loc())
        .add_results(&[i32_ty])
        .add_attributes(&[(
            Identifier::new(gen.context, "value"),
            IntegerAttribute::new(i32_ty, 1).into(),
        )])
        .build()?;
    let c1 = block.append_operation(c1_op).result(0)?.into();

    let stack_slot_op = OperationBuilder::new("llvm.alloca", gen.loc())
        .add_operands(&[c1])
        .add_results(&[ptr_ty])
        .add_attributes(&[(
            Identifier::new(gen.context, "elem_type"),
            TypeAttribute::new(closure_ty).into(),
        )])
        .build()?;
    let alloca_ptr = block.append_operation(stack_slot_op).result(0)?.into();

    let store_op = OperationBuilder::new("llvm.store", gen.loc())
        .add_operands(&[closure_val, alloca_ptr])
        .build()?;
    block.append_operation(store_op);

    // Extract invoke method name
    let ty_str = closure_ty.to_string();
    let struct_name = ty_str
        .strip_prefix("!llvm.struct<\"")
        .unwrap_or(&ty_str)
        .split('"')
        .next()
        .unwrap();
    Ok((alloca_ptr, format!("{struct_name}_call")))
}

/// Allocates the tensor that `map` writes its results into. It has the type of the input
/// tensor, and each dimension written `?` in `dims` takes its size from the input.
///
/// # Errors
///
/// Returns an error if MLIR refuses to build one of the operations.
///
/// # Panics
///
/// Panics if the number of dimensions does not fit in an `i32`, which cannot happen for a
/// tensor a program can declare.
fn alloc_map_output<'c>(
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    tensor_val: Value<'c, 'c>,
    tensor_ty: Type<'c>,
    dims: &[&str],
) -> Result<Value<'c, 'c>, LowerError> {
    let mut alloc_operands = Vec::new();
    let index_ty = gen.index_ty;
    for (i, dim_str) in dims.iter().enumerate() {
        if *dim_str == "?" {
            let idx_attr = IntegerAttribute::new(
                Type::index(gen.context),
                i64::try_from(i).expect("a tensor has far fewer dimensions than i64 holds"),
            )
            .into();
            let cst_op = OperationBuilder::new("arith.constant", gen.loc())
                .add_results(&[index_ty])
                .add_attributes(&[(Identifier::new(gen.context, "value"), idx_attr)])
                .build()?;
            let idx_val = block.append_operation(cst_op).result(0)?.into();

            let dim_op = OperationBuilder::new("memref.dim", gen.loc())
                .add_operands(&[tensor_val, idx_val])
                .add_results(&[index_ty])
                .build()?;
            alloc_operands.push(block.append_operation(dim_op).result(0)?.into());
        }
    }

    let dynamic_dims = i32::try_from(alloc_operands.len())
        .expect("a tensor has far fewer dimensions than i32 holds");
    let alloc_op = OperationBuilder::new("memref.alloc", gen.loc())
        .add_operands(&alloc_operands)
        .add_attributes(&[(
            Identifier::new(gen.context, "operandSegmentSizes"),
            DenseI32ArrayAttribute::new(gen.context, &[dynamic_dims, 0]).into(),
        )])
        .add_results(&[tensor_ty])
        .build()?;
    Ok(block.append_operation(alloc_op).result(0)?.into())
}

/// Fills the body of the `linalg.generic` of `map`: it calls the closure on one element and
/// gives back the result. Returns the region that holds that body.
///
/// # Errors
///
/// Returns an error if the element type does not parse or MLIR refuses to build an operation.
fn map_body_region<'c>(
    gen: &MeliorGenerator<'c>,
    block_generic: melior::ir::Block<'c>,
    env_ptr: Value<'c, 'c>,
    invoke_method: &str,
    el_ty_str: &str,
) -> Result<melior::ir::Region<'c>, LowerError> {
    let region = Region::new();
    let call_op = OperationBuilder::new("func.call", gen.loc())
        .add_operands(&[
            env_ptr,
            block_generic
                .argument(0)
                // The message says all there is to say: the block has no first argument.
                .map_err(|_no_such_argument| {
                    crate::codegen::lower::LowerError::from(format!("Missing argument {} block", 0))
                })?
                .into(),
        ])
        .add_attributes(&[(
            Identifier::new(gen.context, "callee"),
            FlatSymbolRefAttribute::new(gen.context, invoke_method).into(),
        )])
        .add_results(&[Type::parse(gen.context, el_ty_str).ok_or_else(|| {
            crate::codegen::lower::LowerError::ParseType("Type::parse failed".to_string())
        })?])
        .build()?;

    let call_res = block_generic.append_operation(call_op).result(0)?.into();

    let yield_op = OperationBuilder::new("linalg.yield", gen.loc())
        .add_operands(&[call_res])
        .build()?;
    block_generic.append_operation(yield_op);
    region.append_block(block_generic);
    Ok(region)
}

/// Adds the `linalg.generic` of `map` to `block`. `operands` are the input tensor and the
/// output tensor, and `region` is the body that runs for each element.
///
/// # Errors
///
/// Returns an error if MLIR refuses to build the operation.
///
/// # Panics
///
/// Panics if MLIR does not parse the affine map or the iterator type written here, which
/// would be a compiler bug.
fn append_map_generic<'c>(
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    operands: &[Value<'c, 'c>],
    rank: usize,
    region: melior::ir::Region<'c>,
) -> Result<(), LowerError> {
    use std::fmt::Write as _;
    // Affine maps: both input and output are identity maps
    let mut more_dims = String::new();
    for i in 1..rank {
        write!(more_dims, ", d{i}").expect("writing to a String cannot fail");
    }
    let affine_map = format!("affine_map<(d0{more_dims}) -> (d0{more_dims})>");
    // For rank 0 it's affine_map<() -> ()>
    let affine_map_attr = if rank == 0 {
        Attribute::parse(gen.context, "affine_map<() -> ()>").unwrap()
    } else {
        Attribute::parse(gen.context, &affine_map).unwrap()
    };

    let linalg_generic = OperationBuilder::new("linalg.generic", gen.loc())
        .add_operands(operands)
        .add_attributes(&[
            (
                Identifier::new(gen.context, "operandSegmentSizes"),
                DenseI32ArrayAttribute::new(gen.context, &[1, 1]).into(),
            ),
            (
                Identifier::new(gen.context, "indexing_maps"),
                ArrayAttribute::new(gen.context, &[affine_map_attr, affine_map_attr]).into(),
            ),
            (
                Identifier::new(gen.context, "iterator_types"),
                ArrayAttribute::new(
                    gen.context,
                    &vec![
                        Attribute::parse(gen.context, "#linalg.iterator_type<parallel>").unwrap();
                        rank
                    ],
                )
                .into(),
            ),
        ])
        .add_regions([region])
        .build()?;
    block.append_operation(linalg_generic);
    Ok(())
}

/// `t.map(|x| ..)`: `args` is `[t, closure]`. Allocates a tensor of the same type as `t`, and
/// a `linalg.generic` writes into it the result of calling the closure on each element.
///
/// # Errors
///
/// Returns an error if an argument fails to generate code, the element type does not parse,
/// or MLIR refuses to build an operation.
///
/// # Panics
///
/// Panics if `t` is not a `memref`.
pub(crate) fn lower_map_call<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    args: &[Expr],
) -> Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError> {
    let (tensor_val, tensor_ty, block) = gen.generate_expr(&args[0], block)?;
    // args[1] is the closure
    // We need to fetch the closure's function and invoke it inside a linalg.generic.
    let tensor_ty_str = tensor_ty.to_string();
    if tensor_ty_str.starts_with("memref<") {
        let parts: Vec<&str> = tensor_ty_str
            .trim_start_matches("memref<")
            .trim_end_matches('>')
            .split('x')
            .collect();
        let el_ty_str = parts.last().unwrap_or(&"f32").trim();
        let rank = parts.len() - 1;

        // 1. Allocate output memref
        let out_val = alloc_map_output(gen, block, tensor_val, tensor_ty, &parts[..rank])?;

        // 2. Generate linalg.generic
        let block_generic = melior::ir::Block::new(&[
            (
                Type::parse(gen.context, el_ty_str).ok_or_else(|| {
                    crate::codegen::lower::LowerError::ParseType("Type::parse failed".to_string())
                })?,
                gen.loc(),
            ),
            (
                Type::parse(gen.context, el_ty_str).ok_or_else(|| {
                    crate::codegen::lower::LowerError::ParseType("Type::parse failed".to_string())
                })?,
                gen.loc(),
            ),
        ]);

        // We need to call the closure!
        // args[1] is the closure expression (StructInitExpr for Closure_N).
        let (closure_val, closure_ty, block) = gen.generate_expr(&args[1], block)?;

        // Allocate it on stack to get a pointer
        let (alloca_ptr, invoke_method) = spill_closure_env(gen, block, closure_val, closure_ty)?;

        let region = map_body_region(gen, block_generic, alloca_ptr, &invoke_method, el_ty_str)?;
        append_map_generic(gen, block, &[tensor_val, out_val], rank, region)?;

        return Ok((out_val, tensor_ty, block));
    }
    panic!("map called on unsupported tensor type: {tensor_ty_str}");
}

/// Makes the rank-0 buffer that `reduce` keeps its running value in, and stores `init_val`
/// in it.
///
/// # Errors
///
/// Returns an error if MLIR refuses to build one of the operations.
fn reduce_accumulator<'c>(
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    init_val: Value<'c, 'c>,
    acc_ty: Type<'c>,
) -> Result<Value<'c, 'c>, LowerError> {
    let alloca_acc = OperationBuilder::new("memref.alloca", gen.loc())
        .add_attributes(&[(
            Identifier::new(gen.context, "operandSegmentSizes"),
            DenseI32ArrayAttribute::new(gen.context, &[0, 0]).into(),
        )])
        .add_results(&[acc_ty])
        .build()?;
    let acc_val = block.append_operation(alloca_acc).result(0)?.into();
    let store_init = OperationBuilder::new("memref.store", gen.loc())
        .add_operands(&[init_val, acc_val])
        .build()?;
    block.append_operation(store_init);
    Ok(acc_val)
}

/// Builds the body of the `linalg.generic` of `reduce`: `acc = closure(env, acc, x)` for each
/// element `x`. Returns the region that holds that body.
///
/// # Errors
///
/// Returns an error if MLIR refuses to build one of the operations.
fn reduce_body_region<'c>(
    gen: &MeliorGenerator<'c>,
    el_ty: Type<'c>,
    env_ptr: Value<'c, 'c>,
    invoke_method: &str,
) -> Result<melior::ir::Region<'c>, LowerError> {
    let region = Region::new();
    let body = melior::ir::Block::new(&[(el_ty, gen.loc()), (el_ty, gen.loc())]);
    // The messages say all there is to say: the block has no such argument.
    let element = body
        .argument(0)
        .map_err(|_no_such_argument| LowerError::from("missing reduce element".to_string()))?;
    let running = body
        .argument(1)
        .map_err(|_no_such_argument| LowerError::from("missing reduce accumulator".to_string()))?;
    let call_op = OperationBuilder::new("func.call", gen.loc())
        .add_operands(&[env_ptr, running.into(), element.into()])
        .add_attributes(&[(
            Identifier::new(gen.context, "callee"),
            FlatSymbolRefAttribute::new(gen.context, invoke_method).into(),
        )])
        .add_results(&[el_ty])
        .build()?;
    let call_res = body.append_operation(call_op).result(0)?.into();
    let yield_op = OperationBuilder::new("linalg.yield", gen.loc())
        .add_operands(&[call_res])
        .build()?;
    body.append_operation(yield_op);
    region.append_block(body);
    Ok(region)
}

/// Adds the `linalg.generic` of `reduce` to `block`. `operands` are the input tensor and the
/// accumulator, and `region` is the body that runs for each element. Every dimension is a
/// reduction, and `vx.reassoc` marks the operation for `vx-reorderable-reductions`.
///
/// # Errors
///
/// Returns an error if an attribute does not parse or MLIR refuses to build the operation.
fn append_reduce_generic<'c>(
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    operands: &[Value<'c, 'c>],
    rank: usize,
    region: melior::ir::Region<'c>,
) -> Result<(), LowerError> {
    let dims = (0..rank)
        .map(|i| format!("d{i}"))
        .collect::<Vec<_>>()
        .join(", ");
    let in_map = Attribute::parse(gen.context, &format!("affine_map<({dims}) -> ({dims})>"))
        .ok_or_else(|| LowerError::from("could not parse the reduce input map".to_string()))?;
    let out_map = Attribute::parse(gen.context, &format!("affine_map<({dims}) -> ()>"))
        .ok_or_else(|| LowerError::from("could not parse the reduce output map".to_string()))?;
    let reduction =
        Attribute::parse(gen.context, "#linalg.iterator_type<reduction>").ok_or_else(|| {
            LowerError::from("could not parse the reduction iterator type".to_string())
        })?;
    let unit = Attribute::parse(gen.context, "unit")
        .ok_or_else(|| LowerError::from("could not parse a unit attribute".to_string()))?;
    let linalg_generic = OperationBuilder::new("linalg.generic", gen.loc())
        .add_operands(operands)
        .add_attributes(&[
            (
                Identifier::new(gen.context, "operandSegmentSizes"),
                DenseI32ArrayAttribute::new(gen.context, &[1, 1]).into(),
            ),
            (
                Identifier::new(gen.context, "indexing_maps"),
                ArrayAttribute::new(gen.context, &[in_map, out_map]).into(),
            ),
            (
                Identifier::new(gen.context, "iterator_types"),
                ArrayAttribute::new(gen.context, &vec![reduction; rank]).into(),
            ),
            (Identifier::new(gen.context, "vx.reassoc"), unit),
        ])
        .add_regions([region])
        .build()?;
    block.append_operation(linalg_generic);
    Ok(())
}

/// `t.reduce(init, |acc, x| ..)`: `args` is `[t, init, closure]`. A rank-0 accumulator holds
/// `init`, and a `linalg.generic` with every dimension a reduction folds each element into it
/// by calling the closure, as `lower_map_call` does. `vx.reassoc` marks it for
/// `vx-reorderable-reductions`.
///
/// # Errors
///
/// Returns an error if an argument fails to generate code, a type or an attribute does not
/// parse, or MLIR refuses to build an operation.
///
/// # Panics
///
/// Panics if `t` is not a `memref`.
pub(crate) fn lower_reduce_call<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    args: &[Expr],
) -> Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError> {
    let (tensor_val, tensor_ty, block) = gen.generate_expr(&args[0], block)?;
    let tensor_ty_str = tensor_ty.to_string();
    let Some(inner) = tensor_ty_str.strip_prefix("memref<") else {
        panic!("reduce called on unsupported tensor type: {tensor_ty_str}");
    };
    let parts: Vec<&str> = inner
        .trim_end_matches('>')
        .split(',')
        .next()
        .unwrap_or("")
        .split('x')
        .collect();
    let el_ty_str = parts.last().unwrap_or(&"f32").trim();
    let rank = parts.len() - 1;
    let parse = |s: &str| {
        Type::parse(gen.context, s).ok_or_else(|| {
            crate::codegen::lower::LowerError::ParseType(format!("Type::parse failed on {s}"))
        })
    };
    let el_ty = parse(el_ty_str)?;
    let acc_ty = parse(&format!("memref<{el_ty_str}>"))?;

    let (init_val, _, block) = gen.generate_expr(&args[1], block)?;
    let acc_val = reduce_accumulator(gen, block, init_val, acc_ty)?;

    // The closure's environment, spilled to the stack for its adapter, as in `lower_map_call`.
    let (closure_val, closure_ty, block) = gen.generate_expr(&args[2], block)?;
    let (env_ptr, invoke_method) = spill_closure_env(gen, block, closure_val, closure_ty)?;

    let region = reduce_body_region(gen, el_ty, env_ptr, &invoke_method)?;
    append_reduce_generic(gen, block, &[tensor_val, acc_val], rank, region)?;

    let load_op = OperationBuilder::new("memref.load", gen.loc())
        .add_operands(&[acc_val])
        .add_results(&[el_ty])
        .build()?;
    let result = block.append_operation(load_op).result(0)?.into();
    Ok((result, el_ty, block))
}

/// Why generating MLIR code for a piece of Vx syntax failed.
#[derive(Debug)]
pub enum LowerError {
    /// A tensor has an element type that the operation cannot handle. Holds the type's name.
    UnsupportedElementType(String),
    /// MLIR refused to build an operation, or a value asked for does not exist.
    Melior(melior::Error),
    /// A type did not parse. Also used for any other failure that only has a message.
    ParseType(String),
}

impl From<melior::Error> for LowerError {
    fn from(e: melior::Error) -> Self {
        LowerError::Melior(e)
    }
}

impl From<String> for LowerError {
    fn from(s: String) -> Self {
        LowerError::ParseType(s)
    }
}

/// The expression that `print` really prints, with a borrow and the checker's cast taken off,
/// and whether that expression is a tensor of `u32` or `u64`.
fn print_operand(args: &[Expr]) -> (&Expr, bool) {
    let mut print_arg = &args[0];
    if let Expr::Borrow(borrow) = print_arg {
        print_arg = &borrow.expr;
    }
    // The checker marks a tensor of `u32` or `u64` with a cast to its own type: MLIR integers
    // carry no sign, and its runtime printer reads them signed.
    let mut unsigned_tensor = false;
    if let Expr::AsCast(cast) = print_arg {
        if let syntax::Type::Tensor(e, _, _) = &cast.target_ty {
            unsigned_tensor = matches!(
                e,
                syntax::types::ElementType::U32 | syntax::types::ElementType::U64
            );
            print_arg = &cast.expr;
        }
    }
    (print_arg, unsigned_tensor)
}

/// The runtime function that prints a tensor with elements of type `el_ty_str`.
///
/// # Errors
///
/// Returns an error if the runtime has no function that prints this element type.
fn tensor_print_fn_name(
    el_ty_str: &'static str,
    unsigned_tensor: bool,
) -> Result<&'static str, LowerError> {
    Ok(match (el_ty_str, unsigned_tensor) {
        ("f32", _) => "printMemrefF32",
        ("f64", _) => "printMemrefF64",
        ("i32", true) => "printMemrefU32",
        ("i64", true) => "printMemrefU64",
        ("i32", false) => "printMemrefI32",
        ("i64", false) => "printMemrefI64",
        ("bf16", _) => "printMemrefBF16",
        _ => return Err(LowerError::UnsupportedElementType(el_ty_str.to_string())),
    })
}

/// Declares the runtime function `print_fn_name`, which takes one tensor of type
/// `unranked_memref_ty`. It does nothing if the function was declared before.
///
/// # Errors
///
/// Returns an error if the function type does not parse or MLIR refuses to build the
/// declaration.
fn declare_tensor_print<'c>(
    gen: &mut MeliorGenerator<'c>,
    print_fn_name: &'static str,
    unranked_memref_ty: Type<'c>,
) -> Result<(), LowerError> {
    // Declared the first time a tensor of this element type is printed.
    if gen.declared_tensor_prints.insert(print_fn_name) {
        let func_ty = Type::parse(gen.context, &format!("({unranked_memref_ty}) -> ()"))
            .ok_or_else(|| LowerError::ParseType(format!("({unranked_memref_ty}) -> ()")))?;
        let decl = OperationBuilder::new("func.func", gen.loc())
            .add_attributes(&[
                (
                    Identifier::new(gen.context, "sym_name"),
                    StringAttribute::new(gen.context, print_fn_name).into(),
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
        gen.module.body().append_operation(decl);
    }
    Ok(())
}

/// `print(x)`: `args` is `[x]`. A number, a `bool` or a string goes to one of the runtime's
/// `print_*` functions, and a tensor goes to the `printMemref*` function for its element type.
///
/// # Errors
///
/// Returns an error if the argument fails to generate code, the runtime cannot print its
/// type, a type does not parse, or MLIR refuses to build an operation.
pub(crate) fn lower_print_call<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    args: &[Expr],
) -> Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError> {
    let (print_arg, unsigned_tensor) = print_operand(args);

    let (mut arg_val, arg_ty, block) = gen.generate_expr(print_arg, block)?;

    // Scalar fast-path (#185): `printMemref*` only accepts ranked memrefs/tensors, so a
    // bare scalar must route through the scalar `print_*` runtime helpers instead.
    if let Some((fn_name, arg_val, arg_ty_str)) =
        pick_scalar_print(gen, block, print_arg, arg_val, arg_ty)?
    {
        let result = call_scalar_print(gen, block, fn_name, arg_val, &arg_ty_str)?;
        return Ok((result, gen.none_ty, block));
    }

    // A string is a pointer to its bytes: the `print_str` helper, as `print!` uses.
    if arg_ty.to_string() == "!llvm.ptr" {
        let result = call_scalar_print(gen, block, "print_str", arg_val, "!llvm.ptr")?;
        return Ok((result, gen.none_ty, block));
    }

    let el_ty_str = extract_mlir_element_type(&arg_ty.to_string())?;

    let print_fn_name = tensor_print_fn_name(el_ty_str, unsigned_tensor)?;

    if arg_ty.to_string().contains(", ") {
        // Strip the space and nothing else -- `memref.memory_space_cast` cannot change the shape,
        // so naming a dynamic one here made the cast invalid for any statically shaped placed
        // tensor (Vx#429). Falls back to the old spelling where there is no space to strip.
        let stripped = crate::codegen::generator::strip_memref_space(&arg_ty.to_string())
            .unwrap_or_else(|| format!("memref<?x?x{el_ty_str}>"));
        let stripped_ty = Type::parse(gen.context, &stripped)
            .ok_or_else(|| LowerError::ParseType(stripped.clone()))?;
        let mcast_op = block.append_operation(
            OperationBuilder::new("memref.memory_space_cast", gen.loc())
                .add_operands(&[arg_val])
                .add_results(&[stripped_ty])
                .build()?,
        );
        arg_val = mcast_op.result(0)?.into();
    }

    let unranked_memref_ty = Type::parse(gen.context, &format!("memref<*x{el_ty_str}>"))
        .ok_or_else(|| LowerError::ParseType(format!("memref<*x{el_ty_str}>")))?;
    declare_tensor_print(gen, print_fn_name, unranked_memref_ty)?;
    let cast_op = block.append_operation(
        OperationBuilder::new("memref.cast", gen.loc())
            .add_operands(&[arg_val])
            .add_results(&[unranked_memref_ty])
            .build()?,
    );
    let cast_val: Value<'_, '_> = cast_op.result(0)?.into();

    block.append_operation(
        OperationBuilder::new("func.call", gen.loc())
            .add_operands(&[cast_val])
            .add_attributes(&[(
                Identifier::new(gen.context, "callee"),
                FlatSymbolRefAttribute::new(gen.context, print_fn_name).into(),
            )])
            .build()?,
    );

    Ok((
        cast_val, // Dummy return value, caller ignores it
        Some(gen.none_ty).ok_or_else(|| LowerError::ParseType("none".to_string()))?,
        block,
    ))
}

/// How to print a scalar: the name of the runtime function, the value to pass to it, and the
/// MLIR type of that value as text.
type ScalarPrint<'c> = (&'static str, Value<'c, 'c>, String);

/// Picks the runtime `print_*` function for a scalar and returns it with the value to pass
/// and that value's MLIR type. MLIR integers carry no sign, so an unsigned value is known by
/// the `x as u64` cast the checker wraps it in. Narrow signed integers and `f16`/`bf16` have
/// no helper of their own and are widened (#320). `None` for anything that is not a scalar.
///
/// # Errors
///
/// Returns an error if the wider type does not parse or MLIR refuses to build the operation
/// that widens the value.
pub(crate) fn pick_scalar_print<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    arg: &Expr,
    arg_val: Value<'c, 'c>,
    arg_ty: Type<'c>,
) -> Result<Option<ScalarPrint<'c>>, LowerError> {
    use crate::syntax::types::{ElementType, Type as VxType};
    let arg = match arg {
        Expr::Borrow(b) => &*b.expr,
        other => other,
    };
    let unsigned = match arg {
        Expr::AsCast(c) => match &c.target_ty {
            VxType::Scalar(e) => matches!(
                e,
                ElementType::U8 | ElementType::U16 | ElementType::U32 | ElementType::U64
            ),
            _ => false,
        },
        _ => false,
    };
    let ty_str = arg_ty.to_string();
    let (fn_name, widen): (&'static str, Option<(&str, &str)>) = match (ty_str.as_str(), unsigned) {
        ("i8", true) => ("print_u8", None),
        ("i16", true) => ("print_u16", None),
        ("i32", true) => ("print_u32", None),
        ("i64", true) => ("print_u64", None),
        ("i1", _) => ("print_bool", Some(("arith.extui", "i32"))),
        ("i8" | "i16", false) => ("print_i32", Some(("arith.extsi", "i32"))),
        ("i32", false) => ("print_i32", None),
        ("i64", false) => ("print_i64", None),
        ("f16" | "bf16", _) => ("print_f32", Some(("arith.extf", "f32"))),
        ("f32", _) => ("print_f32", None),
        ("f64", _) => ("print_f64", None),
        _ => return Ok(None),
    };
    let Some((op_name, wide_ty_str)) = widen else {
        return Ok(Some((fn_name, arg_val, ty_str)));
    };
    let wide_ty = Type::parse(gen.context, wide_ty_str)
        .ok_or_else(|| LowerError::ParseType(wide_ty_str.to_string()))?;
    let cast_op = OperationBuilder::new(op_name, gen.loc())
        .add_operands(&[arg_val])
        .add_results(&[wide_ty])
        .build()?;
    let wide_val = block.append_operation(cast_op).result(0)?.into();
    Ok(Some((fn_name, wide_val, wide_ty_str.to_string())))
}

/// Declares the runtime print helper `fn_name` on first use and calls it with `arg_val`.
///
/// # Errors
///
/// Returns an error if `arg_ty_str` does not parse or MLIR refuses to build an operation.
pub(crate) fn call_scalar_print<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    fn_name: &str,
    arg_val: Value<'c, 'c>,
    arg_ty_str: &str,
) -> Result<Value<'c, 'c>, LowerError> {
    call_runtime_fn(gen, block, fn_name, &[(arg_val, arg_ty_str)], "i32")
}

/// Call a runtime function, declaring it first if the module does not have it yet. The
/// declaration takes the given argument types, not the types of the values passed, so a later
/// call is checked against the signature the declaration really has.
///
/// # Errors
///
/// Returns an error if one of the types given as text does not parse or MLIR refuses to
/// build an operation.
pub(crate) fn call_runtime_fn<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    fn_name: &str,
    args: &[(Value<'c, 'c>, &str)],
    ret_ty_str: &str,
) -> Result<Value<'c, 'c>, LowerError> {
    let parse = |text: &str| {
        Type::parse(gen.context, text).ok_or_else(|| LowerError::ParseType(text.to_string()))
    };
    let ret_ty = parse(ret_ty_str)?;
    if !gen.functions.contains_key(fn_name) {
        let arg_tys: Vec<&str> = args.iter().map(|(_, ty)| *ty).collect();
        let func_ty = parse(&format!("({}) -> {ret_ty_str}", arg_tys.join(", ")))?;
        let recorded_tys = arg_tys
            .iter()
            .map(|ty| parse(ty))
            .collect::<Result<Vec<_>, _>>()?;
        let func_decl = OperationBuilder::new("func.func", gen.loc())
            .add_attributes(&[
                (
                    Identifier::new(gen.context, "sym_name"),
                    StringAttribute::new(gen.context, fn_name).into(),
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
            .insert(fn_name.to_string().into(), (ret_ty, recorded_tys));
    }
    let operands: Vec<Value<'c, 'c>> = args.iter().map(|(v, _)| *v).collect();
    let call_op = block.append_operation(
        OperationBuilder::new("func.call", gen.loc())
            .add_operands(&operands)
            .add_results(&[ret_ty])
            .add_attributes(&[(
                Identifier::new(gen.context, "callee"),
                FlatSymbolRefAttribute::new(gen.context, fn_name).into(),
            )])
            .build()?,
    );
    Ok(call_op.result(0)?.into())
}

/// The eight `raw::` transfer-lowering primitives (#353), emitted in place.
///
/// These are reachable only from an `impl transfer` body inlined at a transfer
/// site -- the checker refuses `raw::` anywhere else (E6017) and discharges the
/// bounds, barrier-shape, and async obligations before codegen ever runs. The
/// lowering here is deliberately direct: loads and stores against the tile
/// memrefs, `gpu.barrier` for the fence, thread identity for the work split --
/// never a `func.call`, which would silently cost the enclosing kernel its
/// device twin (`isDeviceLowerableDialect` excludes func).
///
/// Tiles are FLAT-indexed at the surface (`raw::load(t, i)` with
/// `i < raw::extent(t)`) while the memrefs are rank-N, so a linear index is
/// delinearized here with the tile's static dims (row-major div/mod chain).
/// Static dims are a checked fact: the prover obligations only close for
/// statically shaped tiles, and the device path refuses dynamic shared tiles.
///
/// `raw::async_copy` lowers to its synchronous fallback (an element load+store)
/// and `raw::async_wait` to nothing: the copy-engine capability is declared and
/// gated (E6020), but the engine itself is not driven until the nvgpu route
/// lands -- the fallback preserves the contract's semantics exactly, one
/// element per call, ordered before the wait.
///
/// # Errors
///
/// Returns an error if an argument fails to generate code, a tile is not a tensor with
/// dimensions known at compile time, its element type has no lowering, `prim` is not one of
/// the eight primitives, or MLIR refuses to build an operation.
pub(crate) fn lower_raw_primitive<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    prim: &str,
    args: &[Expr],
) -> Result<Lowered<'c>, LowerError> {
    match prim {
        "extent" => raw_extent(gen, block, args),
        "lane" | "lanes" => raw_lane(gen, block, prim),
        "load" => raw_load(gen, block, args),
        "store" => raw_store(gen, block, args),
        "barrier" => {
            let op = OperationBuilder::new("gpu.barrier", gen.loc()).build()?;
            block.append_operation(op);
            raw_unit(gen, block)
        }
        "async_copy" => raw_async_copy(gen, block, args),
        "async_wait" => raw_unit(gen, block),
        other => Err(LowerError::from(format!(
            "raw::{other} has no lowering; the checker should have refused it"
        ))),
    }
}

/// What generating the code for an expression gives back: its value, the type of that value,
/// and the block to continue in.
type Lowered<'c> = (Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>);

/// The tile's static dims, read off the AST type the checker validated.
///
/// # Errors
///
/// Returns an error if the type of `e` is not known, is not a tensor, or has a dimension that
/// is not a whole number written in the source.
fn raw_tile_dims(gen: &MeliorGenerator<'_>, e: &Expr) -> Result<Vec<u64>, LowerError> {
    let ty = gen
        .infer_ast_type(e)
        .ok_or_else(|| LowerError::from("raw:: tile has no inferable type".to_string()))?;
    let mut inner = &ty;
    loop {
        match inner {
            syntax::Type::Tensor(_, dims, _) => {
                let mut out = Vec::new();
                for d in dims {
                    let Some(n) = d.literal() else {
                        return Err(LowerError::from(
                            "raw:: tile has a non-static dim; the device path \
                             refuses dynamic shared tiles (#353)"
                                .to_string(),
                        ));
                    };
                    // The message says all there is to say: the text is not a whole number.
                    out.push(n.parse::<u64>().map_err(|_not_a_number| {
                        LowerError::from("raw:: tile dim is not an integer".to_string())
                    })?);
                }
                return Ok(out);
            }
            syntax::Type::Borrow { inner: b, .. }
            | syntax::Type::Pointer(b, _, _)
            | syntax::Type::Pinned(b, _)
            | syntax::Type::Ref(b, _) => inner = b,
            _ => {
                return Err(LowerError::from(
                    "raw:: tile argument is not tensor-typed".to_string(),
                ))
            }
        }
    }
}

/// The MLIR type of one element of the tile `e`.
///
/// # Errors
///
/// Returns an error if the type of `e` is not known, is not a tensor, or has an element type
/// that `raw::` cannot load and store.
///
/// # Panics
///
/// Panics if MLIR does not parse one of the element type names written here, which would be a
/// compiler bug.
fn raw_elem_type<'c>(gen: &MeliorGenerator<'c>, e: &Expr) -> Result<Type<'c>, LowerError> {
    let ty = gen
        .infer_ast_type(e)
        .ok_or_else(|| LowerError::from("raw:: tile has no inferable type".to_string()))?;
    let mut inner = &ty;
    loop {
        match inner {
            syntax::Type::Tensor(el, _, _) => {
                let s = match el {
                    ElementType::F16 => "f16",
                    ElementType::F32 => "f32",
                    ElementType::F64 => "f64",
                    ElementType::BF16 => "bf16",
                    ElementType::I8 | ElementType::U8 => "i8",
                    ElementType::I16 | ElementType::U16 => "i16",
                    ElementType::I32 | ElementType::U32 => "i32",
                    ElementType::I64 | ElementType::U64 | ElementType::USize => "i64",
                    other => {
                        return Err(LowerError::from(format!(
                            "raw:: has no lowering for {other:?} tiles"
                        )))
                    }
                };
                return Ok(Type::parse(gen.context, s).unwrap());
            }
            syntax::Type::Borrow { inner: b, .. }
            | syntax::Type::Pointer(b, _, _)
            | syntax::Type::Pinned(b, _)
            | syntax::Type::Ref(b, _) => inner = b,
            _ => {
                return Err(LowerError::from(
                    "raw:: tile argument is not tensor-typed".to_string(),
                ))
            }
        }
    }
}

/// A zero-result primitive still returns a value to the expression walk: the
/// same index-0 dummy the block-expression lowering uses.
///
/// # Errors
///
/// Returns an error if MLIR refuses to build the constant.
fn raw_unit<'c>(
    gen: &MeliorGenerator<'c>,
    b: melior::ir::BlockRef<'c, 'c>,
) -> Result<Lowered<'c>, LowerError> {
    let dummy = OperationBuilder::new("arith.constant", gen.loc())
        .add_attributes(&[(
            Identifier::new(gen.context, "value"),
            IntegerAttribute::new(Type::index(gen.context), 0).into(),
        )])
        .add_results(&[Type::index(gen.context)])
        .build()?;
    let r = b.append_operation(dummy);
    Ok((r.result(0)?.into(), gen.none_ty, b))
}

/// The `index` constant `v`.
///
/// # Errors
///
/// Returns an error if MLIR refuses to build the constant.
///
/// # Panics
///
/// Panics if `v` does not fit in an `i64`. `v` is a size taken from a tile's dimensions, and a
/// tile that large cannot exist.
fn raw_const_index<'c>(
    gen: &MeliorGenerator<'c>,
    b: melior::ir::BlockRef<'c, 'c>,
    v: u64,
) -> Result<Value<'c, 'c>, LowerError> {
    let index_ty = Type::index(gen.context);
    let v = i64::try_from(v).expect("a tile dimension fits in an i64");
    let op = OperationBuilder::new("arith.constant", gen.loc())
        .add_attributes(&[(
            Identifier::new(gen.context, "value"),
            IntegerAttribute::new(index_ty, v).into(),
        )])
        .add_results(&[index_ty])
        .build()?;
    Ok(b.append_operation(op).result(0)?.into())
}

/// Flat i64 index -> per-dim index values, row-major: `idx_k = (i / stride_k) % d_k`.
///
/// # Errors
///
/// Returns an error if MLIR refuses to build one of the operations.
fn raw_delinearize<'c>(
    gen: &MeliorGenerator<'c>,
    b: melior::ir::BlockRef<'c, 'c>,
    lin_in: Value<'c, 'c>,
    lin_ty: Type<'c>,
    dims: &[u64],
) -> Result<Vec<Value<'c, 'c>>, LowerError> {
    let index_ty = Type::index(gen.context);
    // The index expression arrives as i64 from raw::extent/lane arithmetic,
    // but a for-loop induction variable is already `index` on this path --
    // an index->index cast is invalid IR, so cast only when needed.
    let lin: Value<'_, '_> = if lin_ty == index_ty {
        lin_in
    } else {
        let cast = OperationBuilder::new("arith.index_cast", gen.loc())
            .add_operands(&[lin_in])
            .add_results(&[index_ty])
            .build()?;
        b.append_operation(cast).result(0)?.into()
    };
    if dims.len() <= 1 {
        return Ok(vec![lin]);
    }
    let mut out = Vec::new();
    let mut stride: u64 = dims.iter().product();
    for (k, d) in dims.iter().enumerate() {
        stride /= d;
        let sv = raw_const_index(gen, b, stride)?;
        let div = OperationBuilder::new("arith.divui", gen.loc())
            .add_operands(&[lin, sv])
            .add_results(&[index_ty])
            .build()?;
        let q: Value<'_, '_> = b.append_operation(div).result(0)?.into();
        let idx = if k == 0 {
            // The leading digit needs no mod: the checked bound i < extent
            // already caps it below dims[0].
            q
        } else {
            let dv = raw_const_index(gen, b, *d)?;
            let rem = OperationBuilder::new("arith.remui", gen.loc())
                .add_operands(&[q, dv])
                .add_results(&[index_ty])
                .build()?;
            b.append_operation(rem).result(0)?.into()
        };
        out.push(idx);
    }
    Ok(out)
}

/// `raw::extent(t)`: the number of elements in the tile, as an `i64` constant.
///
/// # Errors
///
/// Returns an error if the tile's dimensions are not known or MLIR refuses to build the
/// constant.
///
/// # Panics
///
/// Panics if the number of elements does not fit in an `i64`. A tile that large cannot exist.
fn raw_extent<'c>(
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    args: &[Expr],
) -> Result<Lowered<'c>, LowerError> {
    let i64_ty = gen.i64_ty;
    let dims = raw_tile_dims(gen, &args[0])?;
    let n: u64 = dims.iter().product();
    let n = i64::try_from(n).expect("the number of elements in a tile fits in an i64");
    let op = OperationBuilder::new("arith.constant", gen.loc())
        .add_attributes(&[(
            Identifier::new(gen.context, "value"),
            IntegerAttribute::new(i64_ty, n).into(),
        )])
        .add_results(&[i64_ty])
        .build()?;
    let r = block.append_operation(op);
    Ok((r.result(0)?.into(), i64_ty, block))
}

/// `raw::lane()` and `raw::lanes()`: the number of the running thread, and how many threads
/// there are, each as an `i64`. `prim` is `lane` or `lanes`.
///
/// # Errors
///
/// Returns an error if the `#gpu<dim x>` attribute does not parse or MLIR refuses to build an
/// operation.
fn raw_lane<'c>(
    gen: &MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    prim: &str,
) -> Result<Lowered<'c>, LowerError> {
    let index_ty = Type::index(gen.context);
    let i64_ty = gen.i64_ty;
    // Thread identity, not a constant: correct under any launch geometry,
    // and the 1x1x1 launches of today read 0 and 1 (#251). The host clone
    // of the kernel body gets the same ops; the vx-to-llvm stage folds
    // them to 0/1 there, because the CPU fallback runs one thread.
    let op_name = if prim == "lane" {
        "gpu.thread_id"
    } else {
        "gpu.block_dim"
    };
    let dim_attr = melior::ir::Attribute::parse(gen.context, "#gpu<dim x>")
        .ok_or_else(|| LowerError::from("cannot parse #gpu<dim x>".to_string()))?;
    let op = OperationBuilder::new(op_name, gen.loc())
        .add_attributes(&[(Identifier::new(gen.context, "dimension"), dim_attr)])
        .add_results(&[index_ty])
        .build()?;
    let idx: Value<'_, '_> = block.append_operation(op).result(0)?.into();
    let cast = OperationBuilder::new("arith.index_cast", gen.loc())
        .add_operands(&[idx])
        .add_results(&[i64_ty])
        .build()?;
    let r = block.append_operation(cast);
    Ok((r.result(0)?.into(), i64_ty, block))
}

/// `raw::load(t, i)`: reads element `i` of the tile `t`.
///
/// # Errors
///
/// Returns an error if an argument fails to generate code, the tile's dimensions or element
/// type are not known, or MLIR refuses to build an operation.
fn raw_load<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    args: &[Expr],
) -> Result<Lowered<'c>, LowerError> {
    let (tile, _tty, block) = gen.generate_expr(&args[0], block)?;
    let (lin, lin_ty, block) = gen.generate_expr(&args[1], block)?;
    let dims = raw_tile_dims(gen, &args[0])?;
    let idx = raw_delinearize(gen, block, lin, lin_ty, &dims)?;
    let elem = raw_elem_type(gen, &args[0])?;
    let mut operands = vec![tile];
    operands.extend(idx);
    let op = OperationBuilder::new("memref.load", gen.loc())
        .add_operands(&operands)
        .add_results(&[elem])
        .build()?;
    let r = block.append_operation(op);
    Ok((r.result(0)?.into(), elem, block))
}

/// `raw::store(t, i, v)`: writes `v` to element `i` of the tile `t`.
///
/// # Errors
///
/// Returns an error if an argument fails to generate code, the tile's dimensions are not
/// known, or MLIR refuses to build an operation.
fn raw_store<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    args: &[Expr],
) -> Result<Lowered<'c>, LowerError> {
    let (tile, _tty, block) = gen.generate_expr(&args[0], block)?;
    let (lin, lin_ty, block) = gen.generate_expr(&args[1], block)?;
    let (val, _vty, block) = gen.generate_expr(&args[2], block)?;
    let dims = raw_tile_dims(gen, &args[0])?;
    let idx = raw_delinearize(gen, block, lin, lin_ty, &dims)?;
    let mut operands = vec![val, tile];
    operands.extend(idx);
    let op = OperationBuilder::new("memref.store", gen.loc())
        .add_operands(&operands)
        .build()?;
    block.append_operation(op);
    raw_unit(gen, block)
}

/// `raw::async_copy(dst, src, i)`: copies element `i` of the tile `src` to the tile `dst`.
///
/// # Errors
///
/// Returns an error if an argument fails to generate code, a tile's dimensions or element
/// type are not known, or MLIR refuses to build an operation.
fn raw_async_copy<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    args: &[Expr],
) -> Result<Lowered<'c>, LowerError> {
    // Synchronous fallback: dst[i] = src[i]. The capability is declared
    // (E6020 checked it); the engine is not driven yet -- when the nvgpu
    // route lands this becomes nvgpu.device_async_copy.
    let (dst, _dty, block) = gen.generate_expr(&args[0], block)?;
    let (src, _sty, block) = gen.generate_expr(&args[1], block)?;
    let (lin, lin_ty, block) = gen.generate_expr(&args[2], block)?;
    let sdims = raw_tile_dims(gen, &args[1])?;
    let ddims = raw_tile_dims(gen, &args[0])?;
    let elem = raw_elem_type(gen, &args[1])?;
    let sidx = raw_delinearize(gen, block, lin, lin_ty, &sdims)?;
    let mut load_ops = vec![src];
    load_ops.extend(sidx);
    let load = OperationBuilder::new("memref.load", gen.loc())
        .add_operands(&load_ops)
        .add_results(&[elem])
        .build()?;
    let v: Value<'_, '_> = block.append_operation(load).result(0)?.into();
    let didx = raw_delinearize(gen, block, lin, lin_ty, &ddims)?;
    let mut store_ops = vec![v, dst];
    store_ops.extend(didx);
    let store = OperationBuilder::new("memref.store", gen.loc())
        .add_operands(&store_ops)
        .build()?;
    block.append_operation(store);
    raw_unit(gen, block)
}

/// The memory space a memref type's text ends with, `1` in `memref<4xf32, 1>`; `None` for host
/// memory, which carries none.
fn memref_space(ty: &str) -> Option<&str> {
    let body = ty.strip_prefix("memref<")?.strip_suffix('>')?;
    let last = body.rsplit(", ").next()?;
    (last != body && last.chars().all(|c| c.is_ascii_digit())).then_some(last)
}

/// `memref.cast` from `from` to `to`, through a `memref.memory_space_cast` when the two are in
/// different memory spaces: a placed tensor in a type with its address space, and the same
/// buffer where a plain memref is wanted. A plain cast cannot change the space.
///
/// # Errors
///
/// Returns an error if the two types are in different memory spaces and `from` is not a
/// `memref`, if the type in between does not parse, or if MLIR refuses to build a cast.
#[allow(
    clippy::trivially_copy_pass_by_ref,
    reason = "callers are in other files; they pass the block by reference"
)]
pub(crate) fn cast_memref<'c>(
    gen: &MeliorGenerator<'c>,
    block: &melior::ir::BlockRef<'c, 'c>,
    val: melior::ir::Value<'c, 'c>,
    from: melior::ir::Type<'c>,
    to: melior::ir::Type<'c>,
) -> Result<melior::ir::Value<'c, 'c>, LowerError> {
    let (from_s, to_s) = (from.to_string(), to.to_string());
    let mut val = val;
    let mut current = from;
    if memref_space(&from_s) != memref_space(&to_s) {
        let body = from_s
            .strip_prefix("memref<")
            .and_then(|b| b.strip_suffix('>'))
            .ok_or_else(|| LowerError::from(format!("not a memref: {from_s}")))?;
        let body = match memref_space(&from_s) {
            Some(space) => &body[..body.len() - space.len() - 2],
            None => body,
        };
        let spaced = match memref_space(&to_s) {
            Some(space) => format!("memref<{body}, {space}>"),
            None => format!("memref<{body}>"),
        };
        let spaced_ty = melior::ir::Type::parse(gen.context, &spaced)
            .ok_or_else(|| LowerError::ParseType(spaced.clone()))?;
        val = block
            .append_operation(
                OperationBuilder::new("memref.memory_space_cast", gen.loc())
                    .add_operands(&[val])
                    .add_results(&[spaced_ty])
                    .build()?,
            )
            .result(0)?
            .into();
        current = spaced_ty;
    }
    if current != to {
        val = block
            .append_operation(
                OperationBuilder::new("memref.cast", gen.loc())
                    .add_operands(&[val])
                    .add_results(&[to])
                    .build()?,
            )
            .result(0)?
            .into();
    }
    Ok(val)
}

#[cfg(test)]
#[allow(
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    reason = "a test reports a failure by panicking"
)]
mod tests {
    use super::*;
    use syntax::{Expr, NumberExpr, Span, Topology};

    fn make_num_expr(val: &str) -> Box<Expr> {
        Box::new(Expr::Number(NumberExpr::new(
            val.to_string(),
            None,
            Span::default(),
        )))
    }

    #[test]
    fn test_topology_to_i32_all_variants() {
        assert_eq!(topology_to_i32(&Topology::CPU), 0);
        assert_eq!(topology_to_i32(&Topology::NPU(make_num_expr("0"))), 100);
        assert_eq!(topology_to_i32(&Topology::NPU(make_num_expr("3"))), 103);
        assert_eq!(topology_to_i32(&Topology::AccCore(make_num_expr("0"))), 200);
        assert_eq!(topology_to_i32(&Topology::AccCore(make_num_expr("5"))), 205);
        assert_eq!(topology_to_i32(&Topology::AMX), 300);
        assert_eq!(topology_to_i32(&Topology::ANE), 400);
        assert_eq!(topology_to_i32(&Topology::gpu(0)), 500);
        assert_eq!(topology_to_i32(&Topology::CpuAvx512), 600);
        assert_eq!(topology_to_i32(&Topology::CpuNeon), 700);
        assert_eq!(topology_to_i32(&Topology::Current), 0);
    }

    #[test]
    fn test_topology_to_i32_slice() {
        // #253: a slice's id derives from (base, start, end) in the dedicated 2000..2999
        // band — stable across calls, distinct for a different extent or base, no longer the
        // single constant every slice used to collapse onto.
        let slice = |base: &str, start: &str, end: &str| {
            Topology::Slice(
                Box::new(Topology::NPU(make_num_expr(base))),
                make_num_expr(start),
                make_num_expr(end),
            )
        };
        let nvl144 = topology_to_i32(&slice("0", "0", "144"));
        let nvl72 = topology_to_i32(&slice("0", "0", "72"));
        let other_base = topology_to_i32(&slice("1", "0", "144"));
        assert!((2000..3000).contains(&nvl144), "banded: {nvl144}");
        assert!((2000..3000).contains(&nvl72), "banded: {nvl72}");
        assert_ne!(nvl144, nvl72, "distinct extents get distinct ids");
        assert_ne!(nvl144, other_base, "distinct bases get distinct ids");
        assert_eq!(
            nvl144,
            topology_to_i32(&slice("0", "0", "144")),
            "stable across calls"
        );
    }

    #[test]
    fn test_topology_to_i32_npu_non_numeric_falls_back() {
        // NPU with a non-numeric expr should fall back to 100
        let ident_expr = Box::new(Expr::Identifier(syntax::IdentifierExpr {
            name: "i".into(),
            span: Span::default(),
        }));
        assert_eq!(topology_to_i32(&Topology::NPU(ident_expr)), 100);
    }
}
