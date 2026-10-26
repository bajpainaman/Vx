//===- stmt.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Abstract Syntax Tree nodes for Vx statements.
//
//===----------------------------------------------------------------------===//

use super::{BinaryOp, Expr, IdentifierExpr, IfExpr, Span, TokenTree, Type, UnaryOp, UnaryOpExpr};
use crate::symbol::Symbol;

/// `let name = expr;`: declares a variable and gives it its first value.
#[derive(Debug, PartialEq, Clone)]
pub struct LetDeclStmt {
    /// The name of the variable.
    pub name: Symbol,
    /// Whether the variable is declared `let mut`, so it can be assigned again.
    pub is_mut: bool,
    /// The type written after the name, if any.
    pub ty_ann: Option<Type>,
    /// The first value of the variable.
    pub expr: Expr,
    /// Where the statement is in the source file.
    pub span: Span,
}
impl LetDeclStmt {
    /// Builds a `let` statement.
    #[must_use]
    pub fn new(name: String, is_mut: bool, ty_ann: Option<Type>, expr: Expr, span: Span) -> Self {
        Self {
            name: name.into(),
            is_mut,
            ty_ann,
            expr,
            span,
        }
    }
}

/// `return expr;` or a bare `return;`.
#[derive(Debug, PartialEq, Clone)]
pub struct ReturnStmt {
    /// `None` for a bare `return;`, which only a `void` function may write. Modelled as an
    /// absence rather than a synthesized zero: a `void` function has no value to return, and
    /// inventing one would have to be typed as something.
    pub expr: Option<Expr>,
    /// Where the statement is in the source file.
    pub span: Span,
}
impl ReturnStmt {
    /// Builds a `return` statement.
    #[must_use]
    pub fn new(expr: Option<Expr>, span: Span) -> Self {
        Self { expr, span }
    }
}

/// An expression used as a statement, such as a call.
#[derive(Debug, PartialEq, Clone)]
pub struct ExprStmtStmt {
    /// The expression.
    pub expr: Expr,
    /// Whether a `;` follows the expression. The last expression of a block with no `;` is the
    /// value of the block.
    pub has_semi: bool,
    /// Where the statement is in the source file.
    pub span: Span,
}
impl ExprStmtStmt {
    /// Builds an expression statement.
    #[must_use]
    pub fn new(expr: Expr, has_semi: bool, span: Span) -> Self {
        Self {
            expr,
            has_semi,
            span,
        }
    }
}

/// `for iter in iterable { body }`.
#[derive(Debug, PartialEq, Clone)]
pub struct ForLoopStmt {
    /// The name of the loop variable.
    pub iter: String,
    /// The value the loop goes over, such as a range.
    pub iterable: Box<Expr>,
    /// The conditions the programmer says hold at the start of every pass.
    pub invariants: Vec<Expr>,
    /// The statements that run on every pass.
    pub body: Vec<Statement>,
    /// Where the statement is in the source file.
    pub span: Span,
    /// The `next` the checker resolved for this loop's iterator, by its instance name. Two
    /// instances of one generic iterator have two `next`s, and only the checker knows which.
    pub next_fn: Option<crate::symbol::Symbol>,
}
impl ForLoopStmt {
    /// Builds a `for` loop. The checker fills in `next_fn` later.
    #[must_use]
    pub fn new(
        iter: String,
        iterable: Box<Expr>,
        invariants: Vec<Expr>,
        body: Vec<Statement>,
        span: Span,
    ) -> Self {
        Self {
            iter,
            iterable,
            invariants,
            body,
            span,
            next_fn: None,
        }
    }
}

/// `lhs = rhs;`.
#[derive(Debug, PartialEq, Clone)]
pub struct AssignStmt {
    /// The place that is assigned to.
    pub lhs: Expr,
    /// The value that is assigned.
    pub rhs: Expr,
    /// Where the statement is in the source file.
    pub span: Span,
}
impl AssignStmt {
    /// Builds an assignment.
    #[must_use]
    pub fn new(lhs: Expr, rhs: Expr, span: Span) -> Self {
        Self { lhs, rhs, span }
    }
}

/// An assignment with an operator, such as `lhs += rhs;`.
#[derive(Debug, PartialEq, Clone)]
pub struct CompoundAssignStmt {
    /// The place that is read and then assigned to.
    pub lhs: Expr,
    /// The operator: `Add` for `+=`.
    pub op: BinaryOp,
    /// The right operand.
    pub rhs: Expr,
    /// Where the statement is in the source file.
    pub span: Span,
    /// The accumulator's element type, filled in by the checker, for the same reason
    /// `BinaryOpExpr` carries one: `/=`, `%=` and `>>=` need the sign.
    pub operand_ty: Option<crate::syntax::ElementType>,
}
impl CompoundAssignStmt {
    /// Builds an assignment with an operator. The checker fills in `operand_ty` later.
    #[must_use]
    pub fn new(lhs: Expr, op: BinaryOp, rhs: Expr, span: Span) -> Self {
        Self {
            lhs,
            op,
            rhs,
            span,
            operand_ty: None,
        }
    }
}

/// `assert(expr);` or `assert(expr, "message");`.
#[derive(Debug, PartialEq, Clone)]
pub struct AssertStmt {
    /// The condition that must be true.
    pub expr: Box<Expr>,
    /// The message to show when the condition is false, if one was written.
    pub msg: Option<String>,
    /// Where the statement is in the source file.
    pub span: Span,
}
impl AssertStmt {
    /// Builds an `assert` statement.
    #[must_use]
    pub fn new(expr: Box<Expr>, msg: Option<String>, span: Span) -> Self {
        Self { expr, msg, span }
    }
}

/// `loop { body }`: runs until a `break` or a `return`.
#[derive(Debug, PartialEq, Clone)]
pub struct LoopStmt {
    /// The conditions the programmer says hold at the start of every pass.
    pub invariants: Vec<Expr>,
    /// The statements that run on every pass.
    pub body: Vec<Statement>,
    /// Where the statement is in the source file.
    pub span: Span,
}
impl LoopStmt {
    /// Builds a `loop` statement.
    #[must_use]
    pub fn new(invariants: Vec<Expr>, body: Vec<Statement>, span: Span) -> Self {
        Self {
            invariants,
            body,
            span,
        }
    }
}

/// `break;`: leaves the nearest loop.
#[derive(Debug, PartialEq, Clone)]
pub struct BreakStmt {
    /// Where the statement is in the source file.
    pub span: Span,
}
impl BreakStmt {
    /// Builds a `break` statement.
    #[must_use]
    pub fn new(span: Span) -> Self {
        Self { span }
    }
}

/// `continue;`: starts the next pass of the nearest loop.
#[derive(Debug, PartialEq, Clone)]
pub struct ContinueStmt {
    /// Where the statement is in the source file.
    pub span: Span,
}
impl ContinueStmt {
    /// Builds a `continue` statement.
    #[must_use]
    pub fn new(span: Span) -> Self {
        Self { span }
    }
}

/// A macro call used as a statement, such as `name!(tokens);`, before the macro is expanded.
#[derive(Debug, PartialEq, Clone)]
pub struct MacroCallStmt {
    /// The name of the macro.
    pub name: Symbol,
    /// The tokens between the delimiters of the call.
    pub token_tree: TokenTree,
    /// The tokens of a block written after the call, if there is one.
    pub block_tree: Option<TokenTree>,
    /// Whether a `;` follows the call.
    pub has_semi: bool,
    /// Where the statement is in the source file.
    pub span: Span,
}
impl MacroCallStmt {
    /// Builds a macro call statement.
    #[must_use]
    pub fn new(
        name: String,
        token_tree: TokenTree,
        block_tree: Option<TokenTree>,
        has_semi: bool,
        span: Span,
    ) -> Self {
        Self {
            name: name.into(),
            token_tree,
            block_tree,
            has_semi,
            span,
        }
    }
}

/// `Statement::Drop`: the tensor named here is freed at this point.
#[derive(Debug, PartialEq, Clone)]
pub struct DropStmt {
    /// The variable that owns what is freed.
    pub name: Symbol,
    /// The fields to follow from `name` to the tensor freed: `["inner", "t"]` for
    /// `name.inner.t`. Empty when `name` is the tensor.
    pub path: Vec<Symbol>,
    /// The `drop` of a type that implements `Drop`, by its mangled name: then this drop calls
    /// it with `&mut name.path` instead of freeing a tensor.
    pub call: Option<Symbol>,
    /// What the code generators lower, built by the checker: the call `call(&mut name.path)`,
    /// or the place `name.path` of the tensor to free. `None` when `name` is the tensor.
    pub expr: Option<Box<super::Expr>>,
    /// A `bool` local that is true once the tensor was moved: then nothing is freed.
    pub flag: Option<Symbol>,
    /// Freed only once the next statement has computed its value: a `return`'s, a block's last
    /// expression, or an assignment's right-hand side, which may still read the tensor.
    pub after_value: bool,
    /// Where the statement is in the source file.
    pub span: Span,
}

impl DropStmt {
    /// For a drop that calls a `Drop` impl, or drops an enum's payload: the statement that
    /// does it, inside `if !moved { .. }` when the value may have been moved.
    ///
    /// # Panics
    ///
    /// Panics if this drop calls a `Drop` impl but `expr` is `None`. The checker fills in `expr`
    /// for every such drop.
    #[must_use]
    pub fn call_statement(&self) -> Option<Statement> {
        let payload = matches!(self.expr.as_deref(), Some(Expr::Match(_)));
        if self.call.is_none() && !payload {
            return None;
        }
        let call = self
            .expr
            .as_deref()
            .expect("a drop that runs code has it")
            .clone();
        let call = Statement::ExprStmt(ExprStmtStmt::new(call, true, Span::default()));
        let Some(flag) = &self.flag else {
            return Some(call);
        };
        let not_moved = Expr::UnaryOp(UnaryOpExpr::new(
            UnaryOp::Not,
            Box::new(Expr::Identifier(IdentifierExpr::new(
                flag.clone(),
                Span::default(),
            ))),
            Span::default(),
        ));
        Some(Statement::ExprStmt(ExprStmtStmt::new(
            Expr::If(IfExpr::new(
                false,
                Box::new(not_moved),
                vec![call],
                None,
                Span::default(),
            )),
            true,
            Span::default(),
        )))
    }
}

/// One statement of a function body or a block.
#[derive(Debug, PartialEq, Clone)]
pub enum Statement {
    /// `let name = expr;`
    LetDecl(LetDeclStmt),
    /// `return expr;`
    Return(ReturnStmt),
    /// An expression used as a statement.
    ExprStmt(ExprStmtStmt),
    /// `for iter in iterable { .. }`
    ForLoop(ForLoopStmt),
    /// `lhs = rhs;`
    Assign(AssignStmt),
    /// An assignment with an operator, such as `lhs += rhs;`
    CompoundAssign(CompoundAssignStmt),
    /// `assert(expr);`
    Assert(AssertStmt),
    /// `loop { .. }`
    Loop(LoopStmt),
    /// `break;`
    Break(BreakStmt),
    /// `continue;`
    Continue(ContinueStmt),
    /// A macro call that is not expanded yet.
    MacroCall(MacroCallStmt),
    /// Free what the tensor `name` owns. Never written by a programmer: the checker puts it
    /// where the tensor's drop point is.
    Drop(DropStmt),
    /// A statement the parser could not read. It holds where the bad text is.
    Error(Span),
}

macro_rules! delegate_stmt {
    ($self:ident, $inner:ident => $expr:expr) => {
        match $self {
            Statement::LetDecl($inner) => $expr,
            Statement::Return($inner) => $expr,
            Statement::ExprStmt($inner) => $expr,
            Statement::ForLoop($inner) => $expr,
            Statement::Assign($inner) => $expr,
            Statement::CompoundAssign($inner) => $expr,
            Statement::Assert($inner) => $expr,
            Statement::Loop($inner) => $expr,
            Statement::Break($inner) => $expr,
            Statement::Continue($inner) => $expr,
            Statement::MacroCall($inner) => $expr,
            Statement::Drop($inner) => $expr,
            Statement::Error(_) => unreachable!(),
        }
    };
}

impl ForLoopStmt {
    /// A copy of the loop with each type name in `mapping` replaced by its type.
    fn substitute(&self, mapping: &std::collections::HashMap<crate::symbol::Symbol, Type>) -> Self {
        let e = self;
        ForLoopStmt {
            iter: e.iter.clone(),
            iterable: Box::new(e.iterable.substitute(mapping)),
            invariants: e
                .invariants
                .iter()
                .map(|expr| expr.substitute(mapping))
                .collect(),
            body: e.body.iter().map(|s| s.substitute(mapping)).collect(),
            span: e.span,
            next_fn: e.next_fn.clone(),
        }
    }
}

impl LoopStmt {
    /// A copy of the loop with each type name in `mapping` replaced by its type.
    fn substitute(&self, mapping: &std::collections::HashMap<crate::symbol::Symbol, Type>) -> Self {
        let e = self;
        LoopStmt {
            invariants: e
                .invariants
                .iter()
                .map(|expr| expr.substitute(mapping))
                .collect(),
            body: e.body.iter().map(|s| s.substitute(mapping)).collect(),
            span: e.span,
        }
    }
}

impl Statement {
    /// A copy of the statement with each type name in `mapping` replaced by its type. This is
    /// how the body of a generic function gets its type arguments.
    #[must_use]
    pub fn substitute(
        &self,
        mapping: &std::collections::HashMap<crate::symbol::Symbol, Type>,
    ) -> Statement {
        match self {
            Statement::LetDecl(e) => Statement::LetDecl(LetDeclStmt {
                name: e.name.clone(),
                is_mut: e.is_mut,
                ty_ann: e.ty_ann.as_ref().map(|t| t.substitute(mapping)),
                expr: e.expr.substitute(mapping),
                span: e.span,
            }),
            Statement::Return(e) => Statement::Return(ReturnStmt {
                expr: e.expr.as_ref().map(|x| x.substitute(mapping)),
                span: e.span,
            }),
            Statement::ExprStmt(e) => Statement::ExprStmt(ExprStmtStmt {
                expr: e.expr.substitute(mapping),
                has_semi: e.has_semi,
                span: e.span,
            }),
            Statement::ForLoop(e) => Statement::ForLoop(e.substitute(mapping)),
            Statement::Assign(e) => Statement::Assign(AssignStmt {
                lhs: e.lhs.substitute(mapping),
                rhs: e.rhs.substitute(mapping),
                span: e.span,
            }),
            Statement::CompoundAssign(e) => Statement::CompoundAssign(CompoundAssignStmt {
                lhs: e.lhs.substitute(mapping),
                op: e.op.clone(),
                rhs: e.rhs.substitute(mapping),
                span: e.span,
                operand_ty: e.operand_ty.clone(),
            }),
            Statement::Assert(e) => Statement::Assert(AssertStmt {
                expr: Box::new(e.expr.substitute(mapping)),
                msg: e.msg.clone(),
                span: e.span,
            }),
            Statement::Loop(e) => Statement::Loop(e.substitute(mapping)),
            Statement::Break(e) => Statement::Break(e.clone()),
            Statement::Continue(e) => Statement::Continue(e.clone()),
            Statement::MacroCall(e) => Statement::MacroCall(MacroCallStmt {
                name: e.name.clone(),
                token_tree: e.token_tree.clone(),
                block_tree: e.block_tree.clone(),
                has_semi: e.has_semi,
                span: e.span,
            }),
            Statement::Drop(d) => Statement::Drop(d.clone()),
            Statement::Error(s) => Statement::Error(*s),
        }
    }

    /// Where the statement is in the source file. Not for `Statement::Error`.
    #[must_use]
    pub fn span(&self) -> Span {
        delegate_stmt!(self, s => s.span)
    }
}
