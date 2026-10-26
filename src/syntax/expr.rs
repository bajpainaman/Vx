//===- expr.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Abstract Syntax Tree nodes for Vx expressions, including mathematical operators and function calls.
//
//===----------------------------------------------------------------------===//

use super::{ElementType, ExprStmtStmt, MemorySpace, Span, Statement, TokenTree, Topology, Type};

use crate::symbol::Symbol;
use crate::syntax;
/// An arithmetic or bitwise operator that takes two operands.
#[derive(Debug, PartialEq, Clone)]
pub enum BinaryOp {
    /// `a + b`.
    Add,
    /// `a - b`.
    Sub,
    /// `a * b`.
    Mul,
    /// `a @ b`: matrix multiplication.
    MatMul,
    /// `a / b`.
    Div,
    /// `a % b`: the remainder of a division.
    Rem,
    /// `a & b`: bitwise and.
    BitAnd,
    /// `a | b`: bitwise or.
    BitOr,
    /// `a ^ b`: bitwise exclusive or.
    BitXor,
    /// `a << b`: shift left.
    Shl,
    /// `a >> b`: shift right.
    Shr,
}

/// A comparison operator. Its result is a boolean.
#[derive(Debug, PartialEq, Clone)]
pub enum RelationalOp {
    /// `a == b`.
    Eq,
    /// `a != b`.
    NotEq,
    /// `a < b`.
    Lt,
    /// `a > b`.
    Gt,
    /// `a <= b`.
    Le,
    /// `a >= b`.
    Ge,
}

/// A boolean operator that takes two operands.
#[derive(Debug, PartialEq, Clone)]
pub enum LogicalOp {
    /// `a && b`.
    And,
    /// `a || b`.
    Or,
}

/// An operator that takes one operand.
#[derive(Debug, PartialEq, Clone)]
pub enum UnaryOp {
    /// `!a`.
    Not,
    /// `-a`.
    Neg,
}

/// A name used as a value, such as a variable.
#[derive(Debug, PartialEq, Clone)]
pub struct IdentifierExpr {
    /// The name as written.
    pub name: Symbol,
    /// Where the name is in the source.
    pub span: Span,
}
impl IdentifierExpr {
    /// Creates the node from its fields.
    #[must_use]
    pub fn new(name: Symbol, span: Span) -> Self {
        Self { name, span }
    }
}

/// An enum variant used as a value, such as `Option::Some(x)`.
#[derive(Debug, PartialEq, Clone)]
pub struct EnumVariantExpr {
    /// The enum the variant belongs to.
    pub enum_name: Symbol,
    /// The variant inside that enum.
    pub variant_name: Symbol,
    /// The values in parentheses after the variant, if it carries any.
    pub payload: Option<Vec<Expr>>,
    /// Where the expression is in the source.
    pub span: Span,
}
impl EnumVariantExpr {
    /// Creates the node from its fields.
    #[must_use]
    pub fn new(
        enum_name: Symbol,
        variant_name: Symbol,
        payload: Option<Vec<Expr>>,
        span: Span,
    ) -> Self {
        Self {
            enum_name,
            variant_name,
            payload,
            span,
        }
    }
}

/// A number literal.
#[derive(Debug, PartialEq, Clone)]
pub struct NumberExpr {
    /// The number as text.
    pub value: Symbol,
    /// The number's type, when it is known.
    pub ty: Option<ElementType>,
    /// Where the literal is in the source.
    pub span: Span,
}
impl NumberExpr {
    /// Creates the node from its fields.
    #[must_use]
    pub fn new(value: String, ty: Option<ElementType>, span: Span) -> Self {
        Self {
            value: value.into(),
            ty,
            span,
        }
    }
}

/// A string literal.
#[derive(Debug, PartialEq, Clone)]
pub struct StringLiteralExpr {
    /// The text of the string.
    pub value: Symbol,
    /// Where the literal is in the source.
    pub span: Span,
}
impl StringLiteralExpr {
    /// Creates the node from its fields.
    #[must_use]
    pub fn new(value: String, span: Span) -> Self {
        Self {
            value: value.into(),
            span,
        }
    }
}

/// A `spawn on(..) { .. }` block: code that runs on the named device.
#[derive(Debug, PartialEq, Clone)]
pub struct SpawnOnExpr {
    /// The device the block runs on.
    pub top: Topology,
    /// The statements in the block.
    pub stmts: Vec<Statement>,
    /// The expression at the end of the block that gives its value, if there is one.
    pub ret: Option<Box<Expr>>,
    /// Where the block is in the source.
    pub span: Span,
}
impl SpawnOnExpr {
    /// Creates the node from its fields.
    #[must_use]
    pub fn new(top: Topology, stmts: Vec<Statement>, ret: Option<Box<Expr>>, span: Span) -> Self {
        Self {
            top,
            stmts,
            ret,
            span,
        }
    }
}

/// A value moved into another memory space.
#[derive(Debug, PartialEq, Clone)]
pub struct TransferExpr {
    /// The value being moved.
    pub expr: Box<Expr>,
    /// The memory space it is moved to.
    pub space: MemorySpace,
    /// The bandwidth-derived roofline cost, filled in by sema. Cycles for a `B/cyc` hop,
    /// picoseconds for a `B/s` one — `u64` because picoseconds overflow `u32` at 4.3 ms.
    pub cost: Option<u64>,
    /// The `impl Transfer` lowering sema matched for this site, when one exists and is
    /// emittable: codegen inlines that lowering's body in place of the builtin copy.
    /// Recorded as the (from, to, topology) key -- the lowering itself is looked up again
    /// at emission, where the bodies live.
    ///
    /// The topology is part of the key because the edge alone no longer identifies a
    /// lowering: two machines may implement `Memory::L2 -> Memory::SMEM` differently, and
    /// the one that runs here is the one sema picked, not whichever the lookup finds first.
    pub lowering: Option<(MemorySpace, MemorySpace, String)>,
    /// Where the expression is in the source.
    pub span: Span,
}

/// `Reachable<A, B>` used as a value: a compile-time boolean, true iff a transfer path exists
/// from `from` to `to` in the cost graph. Evaluated at comptime (e.g. inside
/// `comptime { if Reachable<A, B> { … } }`); topology variables are substituted during
/// monomorphization first.
///
/// Its arguments are TOPOLOGIES, and it names no particular lowering -- which is why it is no
/// longer spelled `Transfer<A, B>`. That name now belongs to the edge lowering
/// (`impl Transfer<Memory::A, Memory::B> for Topology::X`), whose arguments are memory spaces.
#[derive(Debug, PartialEq, Clone)]
pub struct TransferPredicateExpr {
    /// The topology the path starts at.
    pub from: Topology,
    /// The topology the path ends at.
    pub to: Topology,
    /// Where the expression is in the source.
    pub span: Span,
}
impl TransferExpr {
    /// Creates the node. The cost and the lowering stay empty until the checker fills them in.
    #[must_use]
    pub fn new(expr: Box<Expr>, space: MemorySpace, span: Span) -> Self {
        Self {
            expr,
            space,
            cost: None,
            lowering: None,
            span,
        }
    }
}

/// A call to a function by its name, such as `f(a, b)`.
#[derive(Debug, PartialEq, Clone)]
pub struct FunctionCallExpr {
    /// The function being called. A static call keeps its owner, as in `Tensor::new`.
    pub name: Symbol,
    /// The type arguments written between `<` and `>` at the call, if any.
    pub type_args: Option<Vec<Type>>,
    /// The arguments, in the order written.
    pub args: Vec<Expr>,
    /// Where the call is in the source.
    pub span: Span,
}
impl FunctionCallExpr {
    /// Creates the node from its fields.
    #[must_use]
    pub fn new(name: Symbol, type_args: Option<Vec<Type>>, args: Vec<Expr>, span: Span) -> Self {
        Self {
            name,
            type_args,
            args,
            span,
        }
    }
}

/// A cast written `expr as Type`.
#[derive(Debug, PartialEq, Clone)]
pub struct AsCastExpr {
    /// The value being cast.
    pub expr: Box<Expr>,
    /// The type written after `as`.
    pub target_ty: Type,
    /// The type of `expr`. `None` until the checker fills it in.
    pub source_ty: Option<Type>, // Added by sema
    /// Where the cast is in the source.
    pub span: Span,
}

/// A call where the thing being called is an expression, not a function name.
#[derive(Debug, PartialEq, Clone)]
pub struct IndirectCallExpr {
    /// The expression that gives the function to call.
    pub callee: Box<Expr>,
    /// The arguments, in the order written.
    pub args: Vec<Expr>,
    /// The function type of `callee`. `None` until the checker fills it in.
    pub target_func_ty: Option<syntax::Type>,
    /// Where the call is in the source.
    pub span: Span,
}
impl IndirectCallExpr {
    /// Creates the node. The function type stays empty until the checker fills it in.
    #[must_use]
    pub fn new(callee: Box<Expr>, args: Vec<Expr>, span: Span) -> Self {
        Self {
            callee,
            args,
            target_func_ty: None,
            span,
        }
    }
}

/// An array literal, such as `[1, 2, 3]`.
#[derive(Debug, PartialEq, Clone)]
pub struct ArrayExpr {
    /// The elements, in the order written.
    pub elements: Vec<Expr>,
    /// Where the literal is in the source.
    pub span: Span,
}
impl ArrayExpr {
    /// Creates the node from its fields.
    #[must_use]
    pub fn new(elements: Vec<Expr>, span: Span) -> Self {
        Self { elements, span }
    }

    /// If this is a nested initializer list (its first element is itself an array), return the
    /// inferred shape: this level's length followed by the sub-shape. A *flat* array returns
    /// `None` -- the `Tensor` constructor reads a flat array as an explicit dims list (a shape),
    /// so `Tensor<f32>([2, 4])` stays a 2x4 shape while `Tensor<f32>([[..],[..]])` is data.
    #[must_use]
    pub fn initializer_shape(&self) -> Option<Vec<usize>> {
        match self.elements.first() {
            Some(Expr::Array(inner)) => {
                let mut shape = vec![self.elements.len()];
                match inner.initializer_shape() {
                    Some(sub) => shape.extend(sub),
                    None => shape.push(inner.elements.len()),
                }
                Some(shape)
            }
            _ => None,
        }
    }

    /// Row-major flatten of a nested initializer list into its scalar element expressions,
    /// e.g. `[[a, b], [c, d]]` -> `[a, b, c, d]`.
    #[must_use]
    pub fn initializer_values(&self) -> Vec<&Expr> {
        let mut out = Vec::new();
        for el in &self.elements {
            match el {
                Expr::Array(inner) => out.extend(inner.initializer_values()),
                other => out.push(other),
            }
        }
        out
    }
}

/// A field read, such as `point.x`.
#[derive(Debug, PartialEq, Clone)]
pub struct MemberAccessExpr {
    /// The value whose field is read.
    pub base: Box<Expr>,
    /// The field being read.
    pub member: Symbol,
    /// The struct that `base` is. `None` until a later pass fills it in.
    pub struct_name: Option<Symbol>,
    /// Where the expression is in the source.
    pub span: Span,
}
impl MemberAccessExpr {
    /// Creates the node. The struct name stays empty until a later pass fills it in.
    #[must_use]
    pub fn new(base: Box<Expr>, member: Symbol, span: Span) -> Self {
        Self {
            base,
            member,
            struct_name: None,
            span,
        }
    }
}

/// An index expression, such as `a[i]`.
#[derive(Debug, PartialEq, Clone)]
pub struct IndexAccessExpr {
    /// The value being indexed.
    pub base: Box<Expr>,
    /// The index.
    pub index: Box<Expr>,
    /// Where the expression is in the source.
    pub span: Span,
}
impl IndexAccessExpr {
    /// Creates the node from its fields.
    #[must_use]
    pub fn new(base: Box<Expr>, index: Box<Expr>, span: Span) -> Self {
        Self { base, index, span }
    }
}

/// A method call, such as `v.push(x)`.
#[derive(Debug, PartialEq, Clone)]
pub struct MethodCallExpr {
    /// The value the method is called on.
    pub base: Box<Expr>,
    /// The method being called.
    pub method_name: Symbol,
    /// The type arguments written between `<` and `>` at the call, if any.
    pub type_args: Option<Vec<Type>>,
    /// The arguments, in the order written. The base is not one of them.
    pub args: Vec<Expr>,
    /// Where the call is in the source.
    pub span: Span,
}
impl MethodCallExpr {
    /// Creates the node from its fields.
    #[must_use]
    pub fn new(
        base: Box<Expr>,
        method_name: Symbol,
        type_args: Option<Vec<Type>>,
        args: Vec<Expr>,
        span: Span,
    ) -> Self {
        Self {
            base,
            method_name,
            type_args,
            args,
            span,
        }
    }

    /// `t.reshape(shape, PadMode::Pad)`: a reshape that makes a new tensor rather than a view.
    pub fn is_padding_reshape(&self) -> bool {
        self.method_name.as_ref() == "reshape" && self.args.get(1).is_some_and(is_pad_mode_pad)
    }
}

/// Whether `e` is `PadMode::Pad`.
#[must_use]
pub fn is_pad_mode_pad(e: &Expr) -> bool {
    matches!(e, Expr::EnumVariant(v)
        if v.enum_name.as_ref() == "PadMode" && v.variant_name.as_ref() == "Pad")
}

/// An expression with a left operand, an operator and a right operand.
pub trait BinaryOperator {
    /// The left operand.
    fn lhs(&self) -> &Expr;
    /// The right operand.
    fn rhs(&self) -> &Expr;
    /// Where the whole expression is in the source.
    fn span(&self) -> &Span;
}

/// Two operands joined by an arithmetic or bitwise operator, such as `a + b`.
#[derive(Debug, PartialEq, Clone)]
pub struct BinaryOpExpr {
    /// The left operand.
    pub lhs: Box<Expr>,
    /// The operator between the operands.
    pub op: BinaryOp,
    /// The right operand.
    pub rhs: Box<Expr>,
    /// Where the expression is in the source.
    pub span: Span,
    /// The operands' element type, filled in by the checker. An MLIR integer carries no
    /// sign, so `/`, `%` and `>>` cannot be lowered without it. `None` until the checker
    /// has run, and for an operand that is not a number.
    pub operand_ty: Option<ElementType>,
}
impl BinaryOpExpr {
    /// Creates the node. The operand type stays empty until the checker fills it in.
    #[must_use]
    pub fn new(lhs: Box<Expr>, op: BinaryOp, rhs: Box<Expr>, span: Span) -> Self {
        Self {
            lhs,
            op,
            rhs,
            span,
            operand_ty: None,
        }
    }
}

impl BinaryOperator for BinaryOpExpr {
    fn lhs(&self) -> &Expr {
        &self.lhs
    }
    fn rhs(&self) -> &Expr {
        &self.rhs
    }
    fn span(&self) -> &Span {
        &self.span
    }
}

/// A comparison of two operands, such as `a < b`.
#[derive(Debug, PartialEq, Clone)]
pub struct RelationalOpExpr {
    /// The left operand.
    pub lhs: Box<Expr>,
    /// The comparison between the operands.
    pub op: RelationalOp,
    /// The right operand.
    pub rhs: Box<Expr>,
    /// Where the expression is in the source.
    pub span: Span,
    /// The operands' element type, filled in by the checker, as on `BinaryOpExpr`: `<` on
    /// an unsigned number needs the unsigned predicate. `None` until the checker has run.
    pub operand_ty: Option<ElementType>,
}
impl RelationalOpExpr {
    /// Creates the node. The operand type stays empty until the checker fills it in.
    #[must_use]
    pub fn new(lhs: Box<Expr>, op: RelationalOp, rhs: Box<Expr>, span: Span) -> Self {
        Self {
            lhs,
            op,
            rhs,
            span,
            operand_ty: None,
        }
    }
}

impl BinaryOperator for RelationalOpExpr {
    fn lhs(&self) -> &Expr {
        &self.lhs
    }
    fn rhs(&self) -> &Expr {
        &self.rhs
    }
    fn span(&self) -> &Span {
        &self.span
    }
}

/// Two boolean operands joined by `&&` or `||`.
#[derive(Debug, PartialEq, Clone)]
pub struct LogicalOpExpr {
    /// The left operand.
    pub lhs: Box<Expr>,
    /// The operator between the operands.
    pub op: LogicalOp,
    /// The right operand.
    pub rhs: Box<Expr>,
    /// Where the expression is in the source.
    pub span: Span,
}
impl LogicalOpExpr {
    /// Creates the node from its fields.
    #[must_use]
    pub fn new(lhs: Box<Expr>, op: LogicalOp, rhs: Box<Expr>, span: Span) -> Self {
        Self { lhs, op, rhs, span }
    }
}

impl BinaryOperator for LogicalOpExpr {
    fn lhs(&self) -> &Expr {
        &self.lhs
    }
    fn rhs(&self) -> &Expr {
        &self.rhs
    }
    fn span(&self) -> &Span {
        &self.span
    }
}

/// An operator applied to one operand, such as `-a` or `!a`.
#[derive(Debug, PartialEq, Clone)]
pub struct UnaryOpExpr {
    /// The operator in front of the operand.
    pub op: UnaryOp,
    /// The operand.
    pub expr: Box<Expr>,
    /// Where the expression is in the source.
    pub span: Span,
}
impl UnaryOpExpr {
    /// Creates the node from its fields.
    #[must_use]
    pub fn new(op: UnaryOp, expr: Box<Expr>, span: Span) -> Self {
        Self { op, expr, span }
    }
}

/// A borrow: `&x` or `&mut x`.
#[derive(Debug, PartialEq, Clone)]
pub struct BorrowExpr {
    /// The value being borrowed.
    pub expr: Box<Expr>,
    /// Whether the borrow is `&mut`.
    pub is_mut: bool,
    /// Where the expression is in the source.
    pub span: Span,
}
impl BorrowExpr {
    /// Creates the node from its fields.
    #[must_use]
    pub fn new(expr: Box<Expr>, is_mut: bool, span: Span) -> Self {
        Self { expr, is_mut, span }
    }
}

/// A dereference: `*p`.
#[derive(Debug, PartialEq, Clone)]
pub struct DereferenceExpr {
    /// The pointer or reference being followed.
    pub expr: Box<Expr>,
    /// The type of the value that `expr` points to. `None` until the checker fills it in.
    pub ty: Option<syntax::Type>,
    /// Where the expression is in the source.
    pub span: Span,
}
impl DereferenceExpr {
    /// Creates the node. The type stays empty until the checker fills it in.
    #[must_use]
    pub fn new(expr: Box<Expr>, span: Span) -> Self {
        Self {
            expr,
            ty: None,
            span,
        }
    }
}

/// An `unsafe { .. }` block.
#[derive(Debug, PartialEq, Clone)]
pub struct UnsafeBlockExpr {
    /// The statements in the block.
    pub stmts: Vec<Statement>,
    /// The expression at the end of the block that gives its value, if there is one.
    pub ret: Option<Box<Expr>>,
    /// Where the block is in the source.
    pub span: Span,
}
impl UnsafeBlockExpr {
    /// Creates the node from its fields.
    #[must_use]
    pub fn new(stmts: Vec<Statement>, ret: Option<Box<Expr>>, span: Span) -> Self {
        Self { stmts, ret, span }
    }
}

/// A `comptime { .. }` block: code the compiler runs while it compiles the program.
#[derive(Debug, PartialEq, Clone)]
pub struct ComptimeBlockExpr {
    /// The statements in the block.
    pub stmts: Vec<Statement>,
    /// The expression at the end of the block that gives its value, if there is one.
    pub ret: Option<Box<Expr>>,
    /// Where the block is in the source.
    pub span: Span,
}
impl ComptimeBlockExpr {
    /// Creates the node from its fields.
    #[must_use]
    pub fn new(stmts: Vec<Statement>, ret: Option<Box<Expr>>, span: Span) -> Self {
        Self { stmts, ret, span }
    }
}

/// A struct literal, such as `Point { x: 1, y: 2 }`.
#[derive(Debug, PartialEq, Clone)]
pub struct StructInitExpr {
    /// The struct being built.
    pub name: Symbol,
    /// Each field with the value given to it, in the order written.
    pub fields: Vec<(Symbol, Expr)>,
    /// The resolved GID of the struct being initialized, attached by the type checker
    /// (`check_structinit_expr`). `None` until type-checked, or for a monomorphized/generated struct
    /// whose identity isn't a plain module symbol. Lets consumers (e.g. the flat-HIR lowerer) map the
    /// construction to its registry layout without re-resolving the name (#199).
    pub type_id: Option<crate::gid::TypeId>,
    /// Where the literal is in the source.
    pub span: Span,
}
impl StructInitExpr {
    /// Creates the node. The type id stays empty until the checker fills it in.
    #[must_use]
    pub fn new(name: Symbol, fields: Vec<(Symbol, Expr)>, span: Span) -> Self {
        Self {
            name,
            fields,
            type_id: None,
            span,
        }
    }
}

/// A memory space used as a value.
#[derive(Debug, PartialEq, Clone)]
pub struct MemorySpaceExpr {
    /// The memory space that was named.
    pub space: MemorySpace,
    /// Where the expression is in the source.
    pub span: Span,
}
impl MemorySpaceExpr {
    /// Creates the node from its fields.
    #[must_use]
    pub fn new(space: MemorySpace, span: Span) -> Self {
        Self { space, span }
    }
}

/// A topology used as a value.
#[derive(Debug, PartialEq, Clone)]
pub struct TopologyExpr {
    /// The topology that was named.
    pub top: Topology,
    /// Where the expression is in the source.
    pub span: Span,
}
impl TopologyExpr {
    /// Creates the node from its fields.
    #[must_use]
    pub fn new(top: Topology, span: Span) -> Self {
        Self { top, span }
    }
}

/// The gradient of a function at the given arguments.
#[derive(Debug, PartialEq, Clone)]
pub struct GradExpr {
    /// The function to differentiate.
    pub target_fn: Symbol,
    /// The arguments at which the gradient is taken.
    pub args: Vec<Expr>,
    /// Where the expression is in the source.
    pub span: Span,
}
impl GradExpr {
    /// Creates the node from its fields.
    #[must_use]
    pub fn new(target_fn: Symbol, args: Vec<Expr>, span: Span) -> Self {
        Self {
            target_fn,
            args,
            span,
        }
    }
}

/// A vector-Jacobian product: reverse-mode differentiation of a function.
#[derive(Debug, PartialEq, Clone)]
pub struct VjpExpr {
    /// The function to differentiate.
    pub target_fn: Symbol,
    /// The arguments at which the function is differentiated.
    pub args: Vec<Expr>,
    /// The vector that multiplies the Jacobian from the left.
    pub cotangent: Box<Expr>,
    /// Where the expression is in the source.
    pub span: Span,
}
impl VjpExpr {
    /// Creates the node from its fields.
    #[must_use]
    pub fn new(target_fn: Symbol, args: Vec<Expr>, cotangent: Box<Expr>, span: Span) -> Self {
        Self {
            target_fn,
            args,
            cotangent,
            span,
        }
    }
}

/// A Jacobian-vector product: forward-mode differentiation of a function.
#[derive(Debug, PartialEq, Clone)]
pub struct JvpExpr {
    /// The function to differentiate.
    pub target_fn: Symbol,
    /// The arguments at which the function is differentiated.
    pub args: Vec<Expr>,
    /// The vector that the Jacobian multiplies.
    pub tangent: Box<Expr>,
    /// Where the expression is in the source.
    pub span: Span,
}
impl JvpExpr {
    /// Creates the node from its fields.
    #[must_use]
    pub fn new(target_fn: Symbol, args: Vec<Expr>, tangent: Box<Expr>, span: Span) -> Self {
        Self {
            target_fn,
            args,
            tangent,
            span,
        }
    }
}

/// Whether any arm of a `match` ends in a tail expression, making the match evaluate to a value.
///
/// This is the rule the AST lowering already uses to decide whether its merge block takes a block
/// argument, and the flat emitter needs the same answer for the opposite reason: it has no value
/// form for `match`, so it must decline rather than drop the value.
///
/// The question is *not* "does every arm return". `Option::unwrap`'s `None` arm ends in an
/// `assert`, which aborts rather than returning, so control falls through to the merge block --
/// and a default return there is correct, because nothing arrives carrying a value. What breaks is
/// an arm ending in a bare expression: `match x { 0 => { 7 }, _ => { 9 } }` computed 7 and 9 and
/// then returned the default, silently.
#[must_use]
pub fn match_yields_a_value(m: &MatchExpr) -> bool {
    m.arms.iter().any(|a| {
        matches!(
            a.body.last(),
            Some(Statement::ExprStmt(ExprStmtStmt {
                has_semi: false,
                ..
            }))
        )
    })
}

/// Whether control can reach the end of `stmts`, or whether every path out of it returns.
///
/// A function with a non-void return type whose body can complete is missing a return. Today that
/// is not a frontend error: it reaches codegen, the emitter leaves a block with no terminator, and
/// the MLIR verifier reports it as `block with no terminator` naming an arith op -- with no source
/// location and no statement of what is wrong.
///
/// `abort()` and `panic(msg)` count as exits: they are primitives that end the process, so no path
/// continues past them. `assert` deliberately does *not* -- it only aborts when its condition is false, so control
/// reaches the next statement in general.
///
/// `never(name)` says whether the function `name` is declared `-> !`; a call to one is an exit too.
pub fn block_always_exits(stmts: &[Statement], never: &dyn Fn(&str) -> bool) -> bool {
    stmts.iter().any(|s| statement_always_exits(s, never))
}

/// Whether this single statement ends control flow on every path through it.
pub fn statement_always_exits(stmt: &Statement, never: &dyn Fn(&str) -> bool) -> bool {
    match stmt {
        Statement::Return(_) => true,
        Statement::ExprStmt(ExprStmtStmt { expr, .. }) => match expr {
            Expr::If(IfExpr {
                then_block,
                else_block: Some(else_block),
                ..
            }) => block_always_exits(then_block, never) && block_always_exits(else_block, never),
            Expr::Match(MatchExpr { arms, .. }) => {
                !arms.is_empty() && arms.iter().all(|a| block_always_exits(&a.body, never))
            }
            Expr::FunctionCall(c) => is_abort_or_panic(c) || never(c.name.as_ref()),
            Expr::UnsafeBlock(UnsafeBlockExpr { stmts, .. }) => block_always_exits(stmts, never),
            _ => false,
        },
        _ => false,
    }
}

/// Whether this call is `abort()` or `panic(msg)`, the two built-ins that end the program.
#[must_use]
pub fn is_abort_or_panic(c: &FunctionCallExpr) -> bool {
    match c.name.as_ref() {
        "abort" => c.args.is_empty(),
        "panic" => c.args.len() == 1,
        _ => false,
    }
}

/// Whether this expression cannot be a value, so a trailing one is a statement rather than the
/// block's result.
///
/// The parser rewrites a trailing semicolon-less expression to `return <expr>`, which is right for
/// `if c { 1 } else { 2 }` and wrong for anything that produces nothing: the rewrite then returns a
/// void from a function that promised a value, reporting a type mismatch the source never wrote --
/// or, in a `void` function, emitting `func.return %v : i32` and failing the MLIR verifier.
///
/// Two shapes qualify. An `if` with no `else` has a path that yields nothing. An `unsafe { .. }`
/// with no trailing expression is a block of statements.
#[must_use]
pub fn yields_no_value(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::If(IfExpr {
            else_block: None,
            ..
        }) | Expr::UnsafeBlock(UnsafeBlockExpr { ret: None, .. })
    )
}

/// Whether an `if`/`match` returns on every path, and so is a statement rather than a value.
///
/// Two callers need the same answer. The parser rewrites a trailing semicolon-less expression to
/// `return <expr>`, which is right for `if c { 1 } else { 2 }` and wrong for
/// `if c { return 1; } else { return 2; }` -- the second yields nothing, and rewriting it produced
/// `return <void>` and a type error naming a void the source never wrote. The emitter needs it to
/// know that control does not reach past such a statement, so it does not branch to a merge block
/// that nothing arrives at and no terminator closes.
///
/// Only the case where *every* path returns counts. A construct where one arm returns and another
/// yields a value is still a value.
#[must_use]
pub fn diverges_on_every_path(expr: &Expr) -> bool {
    // A branch may also end in a `match` or an `unsafe` block that exits on every path; that is a
    // statement too.
    fn block_returns(stmts: &[Statement]) -> bool {
        match stmts.last() {
            Some(Statement::Return(_)) => true,
            Some(last @ Statement::ExprStmt(ExprStmtStmt { expr, .. })) => {
                diverges_on_every_path(expr)
                    || (matches!(expr, Expr::Match(_) | Expr::UnsafeBlock(_))
                        && statement_always_exits(last, &|_| false))
            }
            _ => false,
        }
    }
    match expr {
        // An `if` with no `else` always has a path that falls through.
        Expr::If(IfExpr {
            then_block,
            else_block: Some(else_block),
            ..
        }) => block_returns(then_block) && block_returns(else_block),
        // Deliberately not `match`. A `match` whose arms all return already compiled correctly --
        // the rewrite to `return <match>` is harmless there, because the arm blocks carry their own
        // terminators and the chain never falls through to the merge block. Treating it as
        // diverging broke that working case, so the rule covers only the construct that failed.
        _ => false,
    }
}

/// An `if` expression, with or without an `else`.
#[derive(Debug, PartialEq, Clone)]
pub struct IfExpr {
    /// Whether this is a `comptime if`, which the compiler decides while it compiles the program.
    pub is_comptime: bool,
    /// The condition.
    pub cond: Box<Expr>,
    /// The statements that run when the condition is true.
    pub then_block: Vec<Statement>,
    /// The statements that run when the condition is false, if there is an `else`.
    pub else_block: Option<Vec<Statement>>,
    /// Where the expression is in the source.
    pub span: Span,
}
impl IfExpr {
    /// Creates the node from its fields.
    #[must_use]
    pub fn new(
        is_comptime: bool,
        cond: Box<Expr>,
        then_block: Vec<Statement>,
        else_block: Option<Vec<Statement>>,
        span: Span,
    ) -> Self {
        Self {
            is_comptime,
            cond,
            then_block,
            else_block,
            span,
        }
    }
}

/// A range written `start..end`.
#[derive(Debug, PartialEq, Clone)]
pub struct RangeExpr {
    /// The expression before `..`.
    pub start: Box<Expr>,
    /// The expression after `..`.
    pub end: Box<Expr>,
    /// Where the range is in the source.
    pub span: Span,
}
impl RangeExpr {
    /// Creates the node from its fields.
    #[must_use]
    pub fn new(start: Box<Expr>, end: Box<Expr>, span: Span) -> Self {
        Self { start, end, span }
    }
}

/// What one arm of a `match` compares the value against.
#[derive(Debug, PartialEq, Clone)]
pub enum Pattern {
    /// `_`: matches any value.
    Wildcard,
    /// A literal such as a number or a string: matches a value equal to it.
    Literal(Expr), // e.g. Number, StringLiteral
    /// A name: matches any value and gives it that name.
    Identifier(Symbol),
    /// An enum variant: the enum, the variant, and one pattern for each value the variant carries.
    EnumVariant(Symbol, Symbol, Option<Vec<Pattern>>), // enum_name, variant_name, payload patterns
}

/// One arm of a `match`: a pattern and the code that runs when the pattern matches.
#[derive(Debug, PartialEq, Clone)]
pub struct MatchArm {
    /// What the value is compared against.
    pub pattern: Pattern,
    /// The statements that run when the pattern matches.
    pub body: Vec<Statement>,
    /// The arm's first token. `Span::default()` for an arm the compiler wrote itself.
    pub span: Span,
}

/// A `match` expression.
#[derive(Debug, PartialEq, Clone)]
pub struct MatchExpr {
    /// The value being matched.
    pub expr: Box<Expr>,
    /// The arms, in the order written.
    pub arms: Vec<MatchArm>,
    /// Where the expression is in the source.
    pub span: Span,
}
impl MatchExpr {
    /// Creates the node from its fields.
    #[must_use]
    pub fn new(expr: Box<Expr>, arms: Vec<MatchArm>, span: Span) -> Self {
        Self { expr, arms, span }
    }
}

/// A `vec![..]` expression.
#[derive(Debug, PartialEq, Clone)]
pub struct VecMacroExpr {
    /// The elements, in the order written.
    pub elements: Vec<Expr>,
    /// Where the expression is in the source.
    pub span: Span,
}
impl VecMacroExpr {
    /// Creates the node from its fields.
    #[must_use]
    pub fn new(elements: Vec<Expr>, span: Span) -> Self {
        Self { elements, span }
    }
}

/// A closure: a function written inline as an expression.
#[derive(Debug, PartialEq, Clone)]
pub struct ClosureExpr {
    /// Each parameter with its type.
    pub params: Vec<(Symbol, Type)>,
    /// The expression the closure evaluates when it is called.
    pub body: Box<Expr>,
    /// The outer variables the body uses, with their types. Empty until a later pass fills it in.
    pub captures: Vec<(Symbol, Type)>,
    /// The type the closure returns. `None` until a later pass fills it in.
    pub ret_ty: Option<Type>,
    /// Where the closure is in the source.
    pub span: Span,
}
impl ClosureExpr {
    /// Creates the node. The captures and the return type stay empty until a later pass fills them in.
    #[must_use]
    pub fn new(params: Vec<(Symbol, Type)>, body: Box<Expr>, span: Span) -> Self {
        Self {
            params,
            body,
            captures: Vec::new(),
            ret_ty: None,
            span,
        }
    }
}

/// A macro call that has not been expanded yet.
#[derive(Debug, PartialEq, Clone)]
pub struct MacroCallExpr {
    /// The macro being called.
    pub name: Symbol,
    /// The tokens passed to the macro.
    pub token_tree: TokenTree,
    /// The tokens of a block that follows the call, if there is one.
    pub block_tree: Option<TokenTree>,
    /// Where the call is in the source.
    pub span: Span,
}
impl MacroCallExpr {
    /// Creates the node from its fields.
    #[must_use]
    pub fn new(
        name: String,
        token_tree: TokenTree,
        block_tree: Option<TokenTree>,
        span: Span,
    ) -> Self {
        Self {
            name: name.into(),
            token_tree,
            block_tree,
            span,
        }
    }
}

/// A `print` call.
#[derive(Debug, PartialEq, Clone)]
pub struct PrintExpr {
    /// The values to print, in the order written.
    pub args: Vec<Expr>,
    /// Where the call is in the source.
    pub span: Span,
}
impl PrintExpr {
    /// Creates the node from its fields.
    #[must_use]
    pub fn new(args: Vec<Expr>, span: Span) -> Self {
        Self { args, span }
    }
}

/// A `println` call.
#[derive(Debug, PartialEq, Clone)]
pub struct PrintlnExpr {
    /// The values to print, in the order written.
    pub args: Vec<Expr>,
    /// Where the call is in the source.
    pub span: Span,
}
impl PrintlnExpr {
    /// Creates the node from its fields.
    #[must_use]
    pub fn new(args: Vec<Expr>, span: Span) -> Self {
        Self { args, span }
    }
}

/// A `sizeof<T>()` expression.
#[derive(Debug, PartialEq, Clone)]
pub struct SizeOfExpr {
    /// The type whose size is asked for.
    pub target_ty: Type,
    /// Where the expression is in the source.
    pub span: Span,
}
impl SizeOfExpr {
    /// Creates the node from its fields.
    #[must_use]
    pub fn new(target_ty: Type, span: Span) -> Self {
        Self { target_ty, span }
    }
}

/// An `mlir!` block: MLIR text written directly inside a Vx program.
#[derive(Debug, PartialEq, Clone)]
pub struct InlineMlirExpr {
    /// Each input: its `%argN` name in the MLIR text, the Vx expression passed for it, and its
    /// MLIR type as text.
    pub inputs: Vec<(Symbol, Expr, String)>, // "%argN", expr, mlir_type_str
    /// The values the block writes to.
    pub clobbers: Vec<Expr>,
    /// The type the block returns. `None` when it returns nothing.
    pub returns: Option<Type>, // None for void
    /// The entries of the block's `dialects` list.
    pub dialects: Vec<String>,
    /// The MLIR text of the block.
    pub block_str: String,
    /// Where the block is in the source.
    pub span: Span,
}

/// Any Vx expression.
#[derive(Debug, PartialEq, Clone)]
pub enum Expr {
    /// A name used as a value.
    Identifier(IdentifierExpr),
    /// An enum variant used as a value.
    EnumVariant(EnumVariantExpr),
    /// A number literal.
    Number(NumberExpr),
    /// A string literal.
    StringLiteral(StringLiteralExpr),
    /// A value moved into another memory space.
    Transfer(TransferExpr),
    /// `Reachable<A, B>` used as a value.
    TransferPredicate(TransferPredicateExpr),
    /// A call to a function by its name.
    FunctionCall(FunctionCallExpr),
    /// A call where the thing being called is an expression.
    IndirectCall(IndirectCallExpr),
    /// An array literal.
    Array(ArrayExpr),
    /// A field read, such as `point.x`.
    MemberAccess(MemberAccessExpr),
    /// An index expression, such as `a[i]`.
    IndexAccess(IndexAccessExpr),
    /// A method call.
    MethodCall(MethodCallExpr),
    /// An arithmetic or bitwise operator with two operands.
    BinaryOp(BinaryOpExpr),
    /// A comparison of two operands.
    RelationalOp(RelationalOpExpr),
    /// `&&` or `||` with its two operands.
    LogicalOp(LogicalOpExpr),
    /// An operator with one operand.
    UnaryOp(UnaryOpExpr),
    /// A borrow: `&x` or `&mut x`.
    Borrow(BorrowExpr),
    /// A dereference: `*p`.
    Dereference(DereferenceExpr),
    /// An `unsafe { .. }` block.
    UnsafeBlock(UnsafeBlockExpr),
    /// A `comptime { .. }` block.
    ComptimeBlock(ComptimeBlockExpr),
    /// A struct literal.
    StructInit(StructInitExpr),
    /// A memory space used as a value.
    MemorySpace(MemorySpaceExpr),
    /// A topology used as a value.
    Topology(TopologyExpr),
    /// An `if` expression.
    If(IfExpr),
    /// A range written `start..end`.
    Range(RangeExpr),
    /// A `match` expression.
    Match(MatchExpr),
    /// The gradient of a function.
    Grad(GradExpr),
    /// A vector-Jacobian product.
    Vjp(VjpExpr),
    /// A Jacobian-vector product.
    Jvp(JvpExpr),
    /// A `spawn on(..) { .. }` block.
    SpawnOn(SpawnOnExpr),
    /// A `vec![..]` expression.
    VecMacro(VecMacroExpr),
    /// A closure.
    Closure(ClosureExpr),
    /// A macro call that has not been expanded yet.
    MacroCall(MacroCallExpr),
    /// A cast written `expr as Type`.
    AsCast(AsCastExpr),
    /// A `print` call.
    Print(PrintExpr),
    /// A `println` call.
    Println(PrintlnExpr),
    /// A `sizeof<T>()` expression.
    SizeOf(SizeOfExpr),
    /// An `mlir!` block.
    InlineMlir(InlineMlirExpr),
}

macro_rules! delegate_expr {
    ($self:ident, $method:ident) => {
        match $self {
            Expr::Identifier(e) => e.$method.clone(),
            Expr::EnumVariant(e) => e.$method.clone(),
            Expr::Number(e) => e.$method.clone(),
            Expr::StringLiteral(e) => e.$method.clone(),
            Expr::Transfer(e) => e.$method.clone(),
            Expr::TransferPredicate(e) => e.$method.clone(),
            Expr::FunctionCall(e) => e.$method.clone(),
            Expr::IndirectCall(e) => e.$method.clone(),
            Expr::Array(e) => e.$method.clone(),
            Expr::MemberAccess(e) => e.$method.clone(),
            Expr::IndexAccess(e) => e.$method.clone(),
            Expr::MethodCall(e) => e.$method.clone(),
            Expr::BinaryOp(e) => e.$method.clone(),
            Expr::RelationalOp(e) => e.$method.clone(),
            Expr::LogicalOp(e) => e.$method.clone(),
            Expr::UnaryOp(e) => e.$method.clone(),
            Expr::Borrow(e) => e.$method.clone(),
            Expr::Dereference(e) => e.$method.clone(),
            Expr::UnsafeBlock(e) => e.$method.clone(),
            Expr::ComptimeBlock(e) => e.$method.clone(),
            Expr::StructInit(e) => e.$method.clone(),
            Expr::MemorySpace(e) => e.$method.clone(),
            Expr::Topology(e) => e.$method.clone(),
            Expr::If(e) => e.$method.clone(),
            Expr::Range(e) => e.$method.clone(),
            Expr::Match(e) => e.$method.clone(),
            Expr::Grad(e) => e.$method.clone(),
            Expr::Vjp(e) => e.$method.clone(),
            Expr::Jvp(e) => e.$method.clone(),
            Expr::SpawnOn(e) => e.$method.clone(),
            Expr::VecMacro(e) => e.$method.clone(),
            Expr::Closure(e) => e.$method.clone(),
            Expr::MacroCall(e) => e.$method.clone(),
            Expr::AsCast(e) => e.$method.clone(),
            Expr::Print(e) => e.$method.clone(),
            Expr::Println(e) => e.$method.clone(),
            Expr::SizeOf(e) => e.$method.clone(),
            Expr::InlineMlir(e) => e.$method.clone(),
        }
    };
}

/// Replace whole identifiers in a fragment of MLIR text with their bound types. Only complete
/// words are replaced: the `N` in `memref<N x M x f32>` becomes the bound extent, the one
/// inside `NxMxf32` does not.
fn substitute_words(text: &str, mapping: &std::collections::HashMap<Symbol, Type>) -> String {
    if mapping.is_empty() || !mapping.keys().any(|k| text.contains(&**k)) {
        return text.to_string();
    }
    let mut result = String::with_capacity(text.len());
    let mut word = String::new();
    let flush = |word: &mut String, result: &mut String| {
        if word.is_empty() {
            return;
        }
        match mapping.get(word.as_str()) {
            Some(ty) => result.push_str(&ty.to_string()),
            None => result.push_str(word),
        }
        word.clear();
    };
    for c in text.chars() {
        if c.is_alphanumeric() || c == '_' {
            word.push(c);
        } else {
            flush(&mut word, &mut result);
            result.push(c);
        }
    }
    flush(&mut word, &mut result);
    result
}

/// Is this a tensor construction -- storage the compiler allocated?
///
/// A view built with `tensor_view_2d` has the same type and a shape the checker enforces, but
/// names memory the compiler does not control: two views over one pointer are two names for one
/// buffer, which no rule about names can see.
#[must_use]
pub fn is_tensor_construction(e: &Expr) -> bool {
    match e {
        Expr::FunctionCall(fc) => {
            let n = fc.name.as_ref();
            if matches!(n, "Tensor::new" | "Tensor::uninit" | "Tensor::fill") {
                return true;
            }
            n.starts_with("Tensor") && !n.contains("::") && !n.contains('$') && !n.contains("__")
        }
        _ => false,
    }
}

/// The local an `@` operand reads, when the operand is a bare name or a borrow of one.
///
/// `None` for anything else, which is the answer a caller wants: `dst = a @ b` fills `dst` in
/// place, and that is only correct while `dst` is disjoint from both operands. A name, and the
/// declared type behind it, is as far as the caller can settle that without alias analysis.
#[must_use]
pub fn matmul_operand_root(e: &Expr) -> Option<&Symbol> {
    let e = match e {
        Expr::Borrow(b) => b.expr.as_ref(),
        other => other,
    };
    match e {
        Expr::Identifier(id) => Some(&id.name),
        _ => None,
    }
}

/// The locals in `stmts` whose element pointer is taken with `as_ptr()` or `as_mut_ptr()`.
/// Through that pointer a view can share the tensor's memory under another name.
#[must_use]
pub fn locals_with_pointer_taken(stmts: &[Statement]) -> std::collections::HashSet<Symbol> {
    let mut names = std::collections::HashSet::new();
    visit_exprs(stmts, &mut |e| {
        if let Expr::MethodCall(m) = e {
            if matches!(m.method_name.as_ref(), "as_ptr" | "as_mut_ptr") {
                if let Some(root) = matmul_operand_root(&m.base) {
                    names.insert(root.clone());
                }
            }
        }
    });
    names
}

/// The locals in `stmts` whose address is taken with `&x` or `&mut x`.
#[must_use]
pub fn locals_with_address_taken(stmts: &[Statement]) -> std::collections::HashSet<Symbol> {
    let mut names = std::collections::HashSet::new();
    visit_exprs(stmts, &mut |e| {
        if let Expr::Borrow(b) = e {
            if let Expr::Identifier(id) = &*b.expr {
                names.insert(id.name.clone());
            }
        }
    });
    names
}

/// Calls `f` on every expression in `stmts`, each one before the expressions inside it. The
/// tokens of a macro that was not expanded are not expressions, and are not visited.
pub fn visit_exprs(stmts: &[Statement], f: &mut dyn FnMut(&Expr)) {
    for stmt in stmts {
        match stmt {
            Statement::LetDecl(l) => visit_expr(&l.expr, f),
            Statement::Return(r) => {
                if let Some(e) = &r.expr {
                    visit_expr(e, f);
                }
            }
            Statement::ExprStmt(e) => visit_expr(&e.expr, f),
            Statement::ForLoop(l) => {
                visit_expr(&l.iterable, f);
                l.invariants.iter().for_each(|e| visit_expr(e, f));
                visit_exprs(&l.body, f);
            }
            Statement::Assign(a) => {
                visit_expr(&a.lhs, f);
                visit_expr(&a.rhs, f);
            }
            Statement::CompoundAssign(a) => {
                visit_expr(&a.lhs, f);
                visit_expr(&a.rhs, f);
            }
            Statement::Assert(a) => visit_expr(&a.expr, f),
            Statement::Loop(l) => {
                l.invariants.iter().for_each(|e| visit_expr(e, f));
                visit_exprs(&l.body, f);
            }
            Statement::Break(_)
            | Statement::Continue(_)
            | Statement::MacroCall(_)
            | Statement::Drop(_)
            | Statement::Error(_) => {}
        }
    }
}

/// Calls [`visit_expr`] on each expression of `es`, in order.
fn visit_each<'a>(es: impl IntoIterator<Item = &'a Expr>, f: &mut dyn FnMut(&Expr)) {
    es.into_iter().for_each(|e| visit_expr(e, f));
}

/// Visits the statements of a block and then the expression at its end, if it has one.
fn visit_block(stmts: &[Statement], ret: Option<&Expr>, f: &mut dyn FnMut(&Expr)) {
    visit_exprs(stmts, f);
    if let Some(r) = ret {
        visit_expr(r, f);
    }
}

/// Calls `f` on `e` and then on every expression inside it; see [`visit_exprs`].
pub fn visit_expr(e: &Expr, f: &mut dyn FnMut(&Expr)) {
    f(e);
    match e {
        Expr::Identifier(_)
        | Expr::Number(_)
        | Expr::StringLiteral(_)
        | Expr::TransferPredicate(_)
        | Expr::MemorySpace(_)
        | Expr::Topology(_)
        | Expr::MacroCall(_)
        | Expr::SizeOf(_) => {}
        Expr::EnumVariant(v) => visit_each(v.payload.iter().flatten(), f),
        Expr::Transfer(t) => visit_expr(&t.expr, f),
        Expr::FunctionCall(c) => visit_each(&c.args, f),
        Expr::IndirectCall(c) => visit_each([&*c.callee].into_iter().chain(&c.args), f),
        Expr::Array(a) => visit_each(&a.elements, f),
        Expr::MemberAccess(m) => visit_expr(&m.base, f),
        Expr::IndexAccess(i) => visit_each([&*i.base, &*i.index], f),
        Expr::MethodCall(m) => visit_each([&*m.base].into_iter().chain(&m.args), f),
        Expr::BinaryOp(b) => visit_each([&*b.lhs, &*b.rhs], f),
        Expr::RelationalOp(r) => visit_each([&*r.lhs, &*r.rhs], f),
        Expr::LogicalOp(l) => visit_each([&*l.lhs, &*l.rhs], f),
        Expr::UnaryOp(u) => visit_expr(&u.expr, f),
        Expr::Borrow(b) => visit_expr(&b.expr, f),
        Expr::Dereference(d) => visit_expr(&d.expr, f),
        Expr::UnsafeBlock(b) => visit_block(&b.stmts, b.ret.as_deref(), f),
        Expr::ComptimeBlock(b) => visit_block(&b.stmts, b.ret.as_deref(), f),
        Expr::SpawnOn(b) => visit_block(&b.stmts, b.ret.as_deref(), f),
        Expr::StructInit(s) => visit_each(s.fields.iter().map(|(_, e)| e), f),
        Expr::If(i) => {
            visit_expr(&i.cond, f);
            visit_exprs(&i.then_block, f);
            if let Some(b) = &i.else_block {
                visit_exprs(b, f);
            }
        }
        Expr::Range(r) => visit_each([&*r.start, &*r.end], f),
        Expr::Match(m) => {
            visit_expr(&m.expr, f);
            m.arms.iter().for_each(|arm| visit_exprs(&arm.body, f));
        }
        Expr::Grad(g) => visit_each(&g.args, f),
        Expr::Vjp(v) => visit_each(v.args.iter().chain([&*v.cotangent]), f),
        Expr::Jvp(j) => visit_each(j.args.iter().chain([&*j.tangent]), f),
        Expr::VecMacro(v) => visit_each(&v.elements, f),
        Expr::Closure(c) => visit_expr(&c.body, f),
        Expr::AsCast(c) => visit_expr(&c.expr, f),
        Expr::Print(p) => visit_each(&p.args, f),
        Expr::Println(p) => visit_each(&p.args, f),
        Expr::InlineMlir(m) => {
            visit_each(m.inputs.iter().map(|(_, e, _)| e), f);
            visit_each(&m.clobbers, f);
        }
    }
}

/// The type each type parameter stands for, by the parameter's name.
type TypeBindings = std::collections::HashMap<Symbol, Type>;

impl Expr {
    /// Where this expression is in the source.
    #[must_use]
    pub fn span(&self) -> Span {
        delegate_expr!(self, span)
    }

    /// Whether this is an operator with two operands: arithmetic, bitwise, comparison, `&&` or `||`.
    #[must_use]
    pub fn is_binary_operator(&self) -> bool {
        matches!(
            self,
            Expr::BinaryOp(_) | Expr::RelationalOp(_) | Expr::LogicalOp(_)
        )
    }

    /// The left and right operands, when this is an operator with two operands.
    #[must_use]
    pub fn get_binary_operands(&self) -> Option<(&Expr, &Expr)> {
        match self {
            Expr::BinaryOp(e) => Some((&e.lhs, &e.rhs)),
            Expr::RelationalOp(e) => Some((&e.lhs, &e.rhs)),
            Expr::LogicalOp(e) => Some((&e.lhs, &e.rhs)),
            _ => None,
        }
    }

    /// Returns a copy of this expression in which every type parameter named in `mapping` is
    /// replaced by the type bound to it.
    #[must_use]
    pub fn substitute(&self, mapping: &std::collections::HashMap<Symbol, Type>) -> Expr {
        if mapping.is_empty() {
            return self.clone();
        }
        match self {
            Expr::Transfer(e) => substitute_transfer(e, mapping),
            // Type substitution does not touch topologies; topology substitution is done
            // separately during monomorphization.
            Expr::TransferPredicate(e) => Expr::TransferPredicate(e.clone()),
            Expr::ComptimeBlock(e) => substitute_comptime_block(e, mapping),
            Expr::FunctionCall(e) => substitute_function_call(e, mapping),
            Expr::IndirectCall(e) => substitute_indirect_call(e, mapping),
            Expr::Array(e) => substitute_array(e, mapping),
            Expr::MemberAccess(e) => substitute_member_access(e, mapping),
            Expr::IndexAccess(e) => substitute_index_access(e, mapping),
            Expr::MethodCall(e) => substitute_method_call(e, mapping),
            Expr::BinaryOp(e) => substitute_binary_op(e, mapping),
            Expr::RelationalOp(e) => substitute_relational_op(e, mapping),
            Expr::LogicalOp(e) => substitute_logical_op(e, mapping),
            Expr::UnaryOp(e) => substitute_unary_op(e, mapping),
            Expr::Borrow(e) => substitute_borrow(e, mapping),
            Expr::Dereference(e) => substitute_dereference(e, mapping),
            Expr::UnsafeBlock(e) => substitute_unsafe_block(e, mapping),
            Expr::StructInit(e) => substitute_struct_init(e, mapping),
            Expr::AsCast(e) => substitute_as_cast(e, mapping),
            Expr::If(e) => substitute_if(e, mapping),
            Expr::Range(e) => substitute_range(e, mapping),
            Expr::Match(e) => substitute_match(e, mapping),
            Expr::Grad(e) => substitute_grad(e, mapping),
            Expr::Vjp(e) => substitute_vjp(e, mapping),
            Expr::Jvp(e) => substitute_jvp(e, mapping),
            Expr::SpawnOn(e) => substitute_spawn_on(e, mapping),
            Expr::Identifier(id) => substitute_identifier(id, mapping),
            Expr::EnumVariant(e) => substitute_enum_variant(e, mapping),
            Expr::VecMacro(e) => substitute_vec_macro(e, mapping),
            Expr::Closure(e) => substitute_closure(e, mapping),
            Expr::MacroCall(e) => substitute_macro_call(e),
            Expr::Print(e) => substitute_print(e, mapping),
            Expr::Println(e) => substitute_println(e, mapping),
            Expr::SizeOf(e) => substitute_size_of(e, mapping),
            Expr::InlineMlir(e) => substitute_inline_mlir(e, mapping),
            Expr::Number(_) | Expr::StringLiteral(_) | Expr::MemorySpace(_) | Expr::Topology(_) => {
                self.clone()
            }
        }
    }
}

// The functions below are the cases of `Expr::substitute`, one for each kind of expression that
// can contain a type parameter.

/// Substitutes in the value being moved. The cost and the lowering are kept.
fn substitute_transfer(e: &TransferExpr, mapping: &TypeBindings) -> Expr {
    Expr::Transfer(TransferExpr {
        expr: Box::new(e.expr.substitute(mapping)),
        space: e.space.clone(),
        cost: e.cost,
        lowering: e.lowering.clone(),
        span: e.span,
    })
}

/// Substitutes in the statements and the final expression of a `comptime` block.
fn substitute_comptime_block(e: &ComptimeBlockExpr, mapping: &TypeBindings) -> Expr {
    Expr::ComptimeBlock(ComptimeBlockExpr {
        stmts: e.stmts.iter().map(|s| s.substitute(mapping)).collect(),
        ret: e.ret.as_ref().map(|r| Box::new(r.substitute(mapping))),
        span: e.span,
    })
}

/// Substitutes in a call: in the function's name when it names a type parameter, in the type
/// arguments and in the arguments.
fn substitute_function_call(e: &FunctionCallExpr, mapping: &TypeBindings) -> Expr {
    let mut new_name = e.name.clone();
    if let Some(t_name) = new_name.strip_prefix("Tensor_") {
        if let Some(concrete_el) = mapping.get(t_name) {
            new_name = format!("Tensor_{concrete_el}").into();
        }
    } else if new_name.contains('<') {
        new_name = crate::syntax::substitute_type_name(&new_name, mapping).into();
    } else if let Some(ty) = new_name
        .rsplit_once("::")
        .and_then(|(owner, method)| Some((mapping.get(owner)?, method)))
        .map(|(ty, method)| format!("{ty}::{method}"))
    {
        // `T::default()`, `Self::Item::default()`: a static call on a type variable
        // is a static call on what the variable stands for.
        new_name = ty.into();
    }
    let substituted_type_args = e
        .type_args
        .as_ref()
        .map(|tys| tys.iter().map(|ty| ty.substitute(mapping)).collect());
    Expr::FunctionCall(FunctionCallExpr {
        name: new_name,
        type_args: substituted_type_args,
        args: e.args.iter().map(|a| a.substitute(mapping)).collect(),
        span: e.span,
    })
}

/// Substitutes in the callee and the arguments. The recorded function type is kept as it is.
fn substitute_indirect_call(e: &IndirectCallExpr, mapping: &TypeBindings) -> Expr {
    Expr::IndirectCall(IndirectCallExpr {
        callee: Box::new(e.callee.substitute(mapping)),
        args: e.args.iter().map(|arg| arg.substitute(mapping)).collect(),
        target_func_ty: e.target_func_ty.clone(),
        span: e.span,
    })
}

/// Substitutes in each element of an array literal.
fn substitute_array(e: &ArrayExpr, mapping: &TypeBindings) -> Expr {
    Expr::Array(ArrayExpr {
        elements: e.elements.iter().map(|a| a.substitute(mapping)).collect(),
        span: e.span,
    })
}

/// Substitutes in the value whose field is read.
fn substitute_member_access(e: &MemberAccessExpr, mapping: &TypeBindings) -> Expr {
    Expr::MemberAccess(MemberAccessExpr {
        base: Box::new(e.base.substitute(mapping)),
        member: e.member.clone(),
        struct_name: e.struct_name.clone(),
        span: e.span,
    })
}

/// Substitutes in the indexed value and in the index.
fn substitute_index_access(e: &IndexAccessExpr, mapping: &TypeBindings) -> Expr {
    Expr::IndexAccess(IndexAccessExpr {
        base: Box::new(e.base.substitute(mapping)),
        index: Box::new(e.index.substitute(mapping)),
        span: e.span,
    })
}

/// Substitutes in the base, the type arguments and the arguments of a method call.
fn substitute_method_call(e: &MethodCallExpr, mapping: &TypeBindings) -> Expr {
    let substituted_type_args = e
        .type_args
        .as_ref()
        .map(|tys| tys.iter().map(|ty| ty.substitute(mapping)).collect());
    Expr::MethodCall(MethodCallExpr {
        base: Box::new(e.base.substitute(mapping)),
        method_name: e.method_name.clone(),
        type_args: substituted_type_args,
        args: e.args.iter().map(|a| a.substitute(mapping)).collect(),
        span: e.span,
    })
}

/// Substitutes in both operands. The operand type is cleared, so the checker fills it in again.
fn substitute_binary_op(e: &BinaryOpExpr, mapping: &TypeBindings) -> Expr {
    Expr::BinaryOp(BinaryOpExpr {
        lhs: Box::new(e.lhs.substitute(mapping)),
        op: e.op.clone(),
        rhs: Box::new(e.rhs.substitute(mapping)),
        span: e.span,
        operand_ty: None,
    })
}

/// Substitutes in both operands. The operand type is cleared, so the checker fills it in again.
fn substitute_relational_op(e: &RelationalOpExpr, mapping: &TypeBindings) -> Expr {
    Expr::RelationalOp(RelationalOpExpr {
        lhs: Box::new(e.lhs.substitute(mapping)),
        op: e.op.clone(),
        rhs: Box::new(e.rhs.substitute(mapping)),
        span: e.span,
        operand_ty: None,
    })
}

/// Substitutes in both operands of `&&` or `||`.
fn substitute_logical_op(e: &LogicalOpExpr, mapping: &TypeBindings) -> Expr {
    Expr::LogicalOp(LogicalOpExpr {
        lhs: Box::new(e.lhs.substitute(mapping)),
        op: e.op.clone(),
        rhs: Box::new(e.rhs.substitute(mapping)),
        span: e.span,
    })
}

/// Substitutes in the operand.
fn substitute_unary_op(e: &UnaryOpExpr, mapping: &TypeBindings) -> Expr {
    Expr::UnaryOp(UnaryOpExpr {
        op: e.op.clone(),
        expr: Box::new(e.expr.substitute(mapping)),
        span: e.span,
    })
}

/// Substitutes in the borrowed value.
fn substitute_borrow(e: &BorrowExpr, mapping: &TypeBindings) -> Expr {
    Expr::Borrow(BorrowExpr {
        expr: Box::new(e.expr.substitute(mapping)),
        is_mut: e.is_mut,
        span: e.span,
    })
}

/// Substitutes in the dereferenced value and in its recorded type.
fn substitute_dereference(e: &DereferenceExpr, mapping: &TypeBindings) -> Expr {
    Expr::Dereference(DereferenceExpr {
        expr: Box::new(e.expr.substitute(mapping)),
        ty: e.ty.as_ref().map(|t| t.substitute(mapping)),
        span: e.span,
    })
}

/// Substitutes in the statements and the final expression of an `unsafe` block.
fn substitute_unsafe_block(e: &UnsafeBlockExpr, mapping: &TypeBindings) -> Expr {
    Expr::UnsafeBlock(UnsafeBlockExpr {
        stmts: e.stmts.iter().map(|s| s.substitute(mapping)).collect(),
        ret: e.ret.as_ref().map(|r| Box::new(r.substitute(mapping))),
        span: e.span,
    })
}

/// Substitutes in the struct's name, when it has type arguments, and in each field's value.
fn substitute_struct_init(e: &StructInitExpr, mapping: &TypeBindings) -> Expr {
    let mut new_name = e.name.clone();
    if new_name.contains('<') {
        new_name = crate::syntax::substitute_type_name(&new_name, mapping).into();
    }
    Expr::StructInit(StructInitExpr {
        name: new_name,
        fields: e
            .fields
            .iter()
            .map(|(n, ex)| (n.clone(), ex.substitute(mapping)))
            .collect(),
        type_id: e.type_id,
        span: e.span,
    })
}

/// Substitutes in the value being cast and in both of the cast's types.
fn substitute_as_cast(e: &AsCastExpr, mapping: &TypeBindings) -> Expr {
    Expr::AsCast(AsCastExpr {
        expr: Box::new(e.expr.substitute(mapping)),
        target_ty: e.target_ty.substitute(mapping),
        source_ty: e.source_ty.as_ref().map(|t| t.substitute(mapping)),
        span: e.span,
    })
}

/// Substitutes in the condition and in both branches of an `if`.
fn substitute_if(e: &IfExpr, mapping: &TypeBindings) -> Expr {
    Expr::If(IfExpr {
        is_comptime: e.is_comptime,
        cond: Box::new(e.cond.substitute(mapping)),
        then_block: e.then_block.iter().map(|s| s.substitute(mapping)).collect(),
        else_block: e
            .else_block
            .as_ref()
            .map(|b| b.iter().map(|s| s.substitute(mapping)).collect()),
        span: e.span,
    })
}

/// Substitutes in both ends of a range.
fn substitute_range(e: &RangeExpr, mapping: &TypeBindings) -> Expr {
    Expr::Range(RangeExpr {
        start: Box::new(e.start.substitute(mapping)),
        end: Box::new(e.end.substitute(mapping)),
        span: e.span,
    })
}

/// Substitutes in the matched value and in the body of each arm. Patterns are kept as they are.
fn substitute_match(e: &MatchExpr, mapping: &TypeBindings) -> Expr {
    Expr::Match(MatchExpr {
        expr: Box::new(e.expr.substitute(mapping)),
        arms: e
            .arms
            .iter()
            .map(|arm| MatchArm {
                pattern: arm.pattern.clone(), // Patterns don't currently have types to substitute
                body: arm.body.iter().map(|s| s.substitute(mapping)).collect(),
                span: arm.span,
            })
            .collect(),
        span: e.span,
    })
}

/// Substitutes in the arguments of a gradient expression.
fn substitute_grad(e: &GradExpr, mapping: &TypeBindings) -> Expr {
    Expr::Grad(GradExpr {
        target_fn: e.target_fn.clone(),
        args: e.args.iter().map(|a| a.substitute(mapping)).collect(),
        span: e.span,
    })
}

/// Substitutes in the arguments and the cotangent of a vector-Jacobian product.
fn substitute_vjp(e: &VjpExpr, mapping: &TypeBindings) -> Expr {
    Expr::Vjp(VjpExpr {
        target_fn: e.target_fn.clone(),
        args: e.args.iter().map(|a| a.substitute(mapping)).collect(),
        cotangent: Box::new(e.cotangent.substitute(mapping)),
        span: e.span,
    })
}

/// Substitutes in the arguments and the tangent of a Jacobian-vector product.
fn substitute_jvp(e: &JvpExpr, mapping: &TypeBindings) -> Expr {
    Expr::Jvp(JvpExpr {
        target_fn: e.target_fn.clone(),
        args: e.args.iter().map(|a| a.substitute(mapping)).collect(),
        tangent: Box::new(e.tangent.substitute(mapping)),
        span: e.span,
    })
}

/// Substitutes in the device, the statements and the final expression of a `spawn on` block.
fn substitute_spawn_on(e: &SpawnOnExpr, mapping: &TypeBindings) -> Expr {
    Expr::SpawnOn(SpawnOnExpr {
        // The device index is substituted too: `spawn on(NPU[R])` in a generic body must
        // become `NPU[5]` at `f<5>`, or dispatch silently falls back to device 0.
        top: e.top.substitute(mapping),
        stmts: e.stmts.iter().map(|s| s.substitute(mapping)).collect(),
        ret: e.ret.as_ref().map(|r| Box::new(r.substitute(mapping))),
        span: e.span,
    })
}

/// Replaces a name that is a parameter bound to a value: a number becomes a number literal, a
/// constant expression is copied in, and any other value becomes a name. Other names are kept.
fn substitute_identifier(id: &IdentifierExpr, mapping: &TypeBindings) -> Expr {
    if let Some(mapped_ty) = mapping.get(&*id.name) {
        if let Type::Generic(val_str, _) = mapped_ty {
            if val_str.parse::<f64>().is_ok() {
                return Expr::Number(syntax::NumberExpr {
                    value: val_str.to_string().into(),
                    ty: None,
                    span: id.span,
                });
            }
            return Expr::Identifier(syntax::IdentifierExpr {
                name: val_str.clone(),
                span: id.span,
            });
        } else if let Type::Const(expr) = mapped_ty {
            return *expr.clone();
        }
    }
    Expr::Identifier(id.clone())
}

/// Substitutes in the enum's name, when it has type arguments, and in the variant's values.
fn substitute_enum_variant(e: &EnumVariantExpr, mapping: &TypeBindings) -> Expr {
    let mut new_name = e.enum_name.clone();
    if new_name.contains('<') {
        new_name = crate::syntax::substitute_type_name(&new_name, mapping).into();
    }
    Expr::EnumVariant(EnumVariantExpr {
        enum_name: new_name,
        variant_name: e.variant_name.clone(),
        payload: e
            .payload
            .as_ref()
            .map(|p| p.iter().map(|ex| ex.substitute(mapping)).collect()),
        span: e.span,
    })
}

/// Substitutes in each element of a `vec![..]`.
fn substitute_vec_macro(e: &VecMacroExpr, mapping: &TypeBindings) -> Expr {
    Expr::VecMacro(VecMacroExpr {
        elements: e.elements.iter().map(|ex| ex.substitute(mapping)).collect(),
        span: e.span,
    })
}

/// Substitutes in a closure's parameter types, body, capture types and return type.
fn substitute_closure(e: &ClosureExpr, mapping: &TypeBindings) -> Expr {
    Expr::Closure(ClosureExpr {
        params: e
            .params
            .iter()
            .map(|(n, t)| (n.clone(), t.substitute(mapping)))
            .collect(),
        body: Box::new(e.body.substitute(mapping)),
        captures: e
            .captures
            .iter()
            .map(|(n, t)| (n.clone(), t.substitute(mapping)))
            .collect(),
        ret_ty: e.ret_ty.as_ref().map(|t| t.substitute(mapping)),
        span: e.span,
    })
}

/// Copies a macro call. Its tokens are not expressions yet, so nothing is substituted in them.
fn substitute_macro_call(e: &MacroCallExpr) -> Expr {
    Expr::MacroCall(MacroCallExpr {
        name: e.name.clone(),
        token_tree: e.token_tree.clone(),
        block_tree: e.block_tree.clone(),
        span: e.span,
    })
}

/// Substitutes in each value a `print` call prints.
fn substitute_print(e: &PrintExpr, mapping: &TypeBindings) -> Expr {
    Expr::Print(PrintExpr {
        args: e.args.iter().map(|ex| ex.substitute(mapping)).collect(),
        span: e.span,
    })
}

/// Substitutes in each value a `println` call prints.
fn substitute_println(e: &PrintlnExpr, mapping: &TypeBindings) -> Expr {
    Expr::Println(PrintlnExpr {
        args: e.args.iter().map(|ex| ex.substitute(mapping)).collect(),
        span: e.span,
    })
}

/// Substitutes in the type whose size is asked for.
fn substitute_size_of(e: &SizeOfExpr, mapping: &TypeBindings) -> Expr {
    Expr::SizeOf(SizeOfExpr {
        target_ty: e.target_ty.substitute(mapping),
        span: e.span,
    })
}

/// Substitutes in an `mlir!` block. The block is text, so its type parameters are replaced word
/// by word. The declared input types are the same text and are replaced the same way: if they
/// were left alone, the generated wrapper would have a `memref<NxMxf32>` signature over a body
/// that had already become `2x2`.
fn substitute_inline_mlir(e: &InlineMlirExpr, mapping: &TypeBindings) -> Expr {
    Expr::InlineMlir(InlineMlirExpr {
        inputs: e
            .inputs
            .iter()
            .map(|(n, ex, t)| {
                (
                    n.clone(),
                    ex.substitute(mapping),
                    substitute_words(t, mapping),
                )
            })
            .collect(),
        clobbers: e.clobbers.iter().map(|ex| ex.substitute(mapping)).collect(),
        returns: e.returns.as_ref().map(|t| t.substitute(mapping)),
        dialects: e.dialects.clone(),
        block_str: substitute_words(&e.block_str, mapping),
        span: e.span,
    })
}
