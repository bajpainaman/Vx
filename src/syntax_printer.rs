//===- ast_printer.rs - Vx Compiler ----------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides utility functions for pretty-printing the Abstract Syntax Tree.
// It translates parsed Vx AST nodes back into human-readable string representations,
// which is primarily used for debugging, diagnostic messages, and testing the parser's
// fidelity.
//
//===----------------------------------------------------------------------===//
use crate::syntax::{
    ArrayExpr, AssignStmt, BinaryOpExpr, ComptimeBlockExpr, DropStmt, EnumVariantExpr, Expr,
    ExprStmtStmt, FunctionCallExpr, IdentifierExpr, LetDeclStmt, MethodCallExpr, NumberExpr,
    Program, ReturnStmt, Statement, StringLiteralExpr, Type,
};
use std::io::Write;

/// Prints a parsed program as a tree of its declarations, statements and expressions.
pub struct AstPrinter;

/// How deep a line of the tree is, and which of the levels above it have no more siblings to
/// draw a line down to.
#[derive(Clone, Copy)]
pub struct Indent {
    depth: usize,
    last_mask: u64,
}

impl Default for Indent {
    fn default() -> Self {
        Self::new()
    }
}

impl Indent {
    /// The indent of the root, which has nothing in front of it.
    #[must_use]
    pub fn new() -> Self {
        Self {
            depth: 0,
            last_mask: 0,
        }
    }

    /// The indent one level deeper. `is_last` says the node at this level is the last of its
    /// siblings, so no line is drawn below it.
    #[must_use]
    pub fn child(&self, is_last: bool) -> Self {
        let mut mask = self.last_mask;
        if is_last {
            mask |= 1 << self.depth;
        }
        Self {
            depth: self.depth + 1,
            last_mask: mask,
        }
    }

    /// Writes the leading part of a line: a `│` under each level that still has siblings.
    ///
    /// # Errors
    ///
    /// Returns the error from `w` when a write fails.
    pub fn print(&self, w: &mut impl Write) -> std::io::Result<()> {
        for i in 0..self.depth {
            if (self.last_mask & (1 << i)) != 0 {
                write!(w, "   ")?;
            } else {
                write!(w, "│  ")?;
            }
        }
        Ok(())
    }
}

impl AstPrinter {
    /// Calls `print_fn` on each item with the branch to draw in front of it and whether it is
    /// the last one.
    ///
    /// # Errors
    ///
    /// Returns the first error `print_fn` returns.
    fn print_list<T, F>(items: &[T], mut print_fn: F) -> std::io::Result<()>
    where
        F: FnMut(&T, &str, bool) -> std::io::Result<()>,
    {
        for (i, item) in items.iter().enumerate() {
            let is_last = i == items.len() - 1;
            let prefix = if is_last { "└─ " } else { "├─ " };
            print_fn(item, prefix, is_last)?;
        }
        Ok(())
    }

    /// Writes the imports, structs and functions of `program` as a tree.
    ///
    /// # Errors
    ///
    /// Returns the error from `w` when a write fails.
    pub fn print_program(program: &Program, w: &mut impl Write) -> std::io::Result<()> {
        writeln!(w, "Program")?;

        for (i, imp) in program.imports.iter().enumerate() {
            let is_last = i == program.imports.len() - 1
                && program.structs.is_empty()
                && program.functions.is_empty()
                && program.externs.is_empty();
            let prefix = if is_last { "└─ " } else { "├─ " };
            writeln!(w, "{}Import: {}", prefix, imp.path.join("::"))?;
        }

        for (i, struc) in program.structs.iter().enumerate() {
            let is_last = i == program.structs.len() - 1
                && program.functions.is_empty()
                && program.externs.is_empty();
            let prefix = if is_last { "└─ " } else { "├─ " };
            writeln!(w, "{}Struct: {}", prefix, struc.name)?;
            Self::print_fields(w, &struc.fields, &Indent::new().child(is_last))?;
        }

        for (i, func) in program.functions.iter().enumerate() {
            let is_last = i == program.functions.len() - 1 && program.externs.is_empty();
            let prefix = if is_last { "└─ " } else { "├─ " };
            writeln!(w, "{}Function: {}", prefix, func.name)?;
            Self::print_statements(w, &func.body, &Indent::new().child(is_last))?;
        }
        Ok(())
    }

    /// Writes one line for each field of a struct.
    ///
    /// # Errors
    ///
    /// Returns the error from `w` when a write fails.
    fn print_fields(
        w: &mut impl Write,
        fields: &[(crate::symbol::Symbol, Type)],
        indent: &Indent,
    ) -> std::io::Result<()> {
        for (i, (name, ty)) in fields.iter().enumerate() {
            let is_last = i == fields.len() - 1;
            let prefix = if is_last { "└─ " } else { "├─ " };
            indent.print(w)?;
            writeln!(w, "{prefix}{name}: {ty:?}")?;
        }
        Ok(())
    }

    /// Writes each statement of a block, with the expressions inside it below it.
    ///
    /// # Errors
    ///
    /// Returns the error from `w` when a write fails.
    fn print_statements(
        w: &mut impl Write,
        stmts: &[Statement],
        indent: &Indent,
    ) -> std::io::Result<()> {
        Self::print_list(stmts, |stmt, prefix, is_last| {
            Self::print_statement(w, stmt, prefix, indent, is_last)
        })
    }

    /// Writes one statement after the branch `prefix`, with the expressions inside it below it.
    ///
    /// # Errors
    ///
    /// Returns the error from `w` when a write fails.
    fn print_statement(
        w: &mut impl Write,
        stmt: &Statement,
        prefix: &str,
        indent: &Indent,
        is_last: bool,
    ) -> std::io::Result<()> {
        match stmt {
            Statement::LetDecl(LetDeclStmt {
                name,
                is_mut,
                ty_ann: ty,
                expr,
                span: _,
            }) => {
                indent.print(w)?;
                writeln!(
                    w,
                    "{}Let {}{}{} = ",
                    prefix,
                    if *is_mut { "mut " } else { "" },
                    name,
                    if ty.is_some() { " (typed)" } else { "" }
                )?;
                Self::print_expr(w, expr, &indent.child(is_last), true)?;
            }
            Statement::Assign(AssignStmt { lhs, rhs, span: _ }) => {
                indent.print(w)?;
                writeln!(w, "{prefix}Assign")?;
                Self::print_expr(w, lhs, &indent.child(is_last), false)?;
                Self::print_expr(w, rhs, &indent.child(is_last), true)?;
            }
            Statement::Drop(drop) => Self::print_drop(w, drop, prefix, indent)?,
            Statement::Return(ReturnStmt { expr, span: _ }) => {
                indent.print(w)?;
                writeln!(w, "{prefix}Return")?;
                if let Some(e) = expr {
                    Self::print_expr(w, e, &indent.child(is_last), true)?;
                }
            }
            Statement::ExprStmt(ExprStmtStmt {
                expr,
                has_semi: _,
                span: _,
            }) => {
                indent.print(w)?;
                writeln!(w, "{prefix}ExprStmt")?;
                Self::print_expr(w, expr, &indent.child(is_last), true)?;
            }
            _ => {
                indent.print(w)?;
                writeln!(w, "{prefix}{stmt:?}")?; // Fallback for other statements
            }
        }
        Ok(())
    }

    /// Writes a `drop` on one line: what is dropped, how, and under which condition.
    ///
    /// # Errors
    ///
    /// Returns the error from `w` when a write fails.
    fn print_drop(
        w: &mut impl Write,
        drop: &DropStmt,
        prefix: &str,
        indent: &Indent,
    ) -> std::io::Result<()> {
        let DropStmt {
            name,
            path,
            call,
            expr,
            flag,
            after_value,
            span: _,
        } = drop;
        indent.print(w)?;
        write!(w, "{prefix}Drop {name}")?;
        for field in path {
            write!(w, ".{field}")?;
        }
        if let Some(f) = call {
            write!(w, " by calling {f}")?;
        }
        if matches!(expr.as_deref(), Some(Expr::Match(_))) {
            write!(w, " by dropping its payload")?;
        }
        if let Some(f) = flag {
            write!(w, " if {f} is false")?;
        }
        if *after_value {
            write!(w, ", after the value")?;
        }
        writeln!(w)?;
        Ok(())
    }

    /// Writes one expression, with the expressions inside it below it. `is_last` says it is the
    /// last child of its parent.
    ///
    /// # Errors
    ///
    /// Returns the error from `w` when a write fails.
    fn print_expr(
        w: &mut impl Write,
        expr: &Expr,
        indent: &Indent,
        is_last: bool,
    ) -> std::io::Result<()> {
        let prefix = if is_last { "└─ " } else { "├─ " };
        match expr {
            Expr::Identifier(IdentifierExpr { name, span: _ }) => {
                indent.print(w)?;
                writeln!(w, "{prefix}Identifier({name})")?;
            }
            Expr::Number(NumberExpr {
                value: val,
                ty: el_ty,
                span: _,
            }) => {
                indent.print(w)?;
                writeln!(w, "{prefix}Number({val}{el_ty:?})")?;
            }
            Expr::StringLiteral(StringLiteralExpr { value: s, span: _ }) => {
                indent.print(w)?;
                writeln!(w, "{prefix}String(\"{s}\")")?;
            }
            Expr::BinaryOp(e) => Self::print_binary_op(w, e, prefix, indent, is_last)?,
            Expr::FunctionCall(e) => Self::print_function_call(w, e, prefix, indent, is_last)?,
            Expr::MethodCall(e) => Self::print_method_call(w, e, prefix, indent, is_last)?,
            Expr::Array(ArrayExpr {
                elements: items,
                span: _,
            }) => {
                indent.print(w)?;
                writeln!(w, "{prefix}Array")?;
                let new_indent = indent.child(is_last);
                for (i, item) in items.iter().enumerate() {
                    Self::print_expr(w, item, &new_indent, i == items.len() - 1)?;
                }
            }
            Expr::EnumVariant(EnumVariantExpr {
                enum_name,
                variant_name: variant,
                payload: _,
                span: _,
            }) => {
                indent.print(w)?;
                writeln!(w, "{prefix}Enum({enum_name}::{variant})")?;
            }
            Expr::ComptimeBlock(e) => Self::print_comptime_block(w, e, prefix, indent, is_last)?,
            _ => {
                indent.print(w)?;
                writeln!(w, "{prefix}{expr:?}")?;
            }
        }
        Ok(())
    }

    /// Writes a binary operation, with its two operands below it.
    ///
    /// # Errors
    ///
    /// Returns the error from `w` when a write fails.
    fn print_binary_op(
        w: &mut impl Write,
        e: &BinaryOpExpr,
        prefix: &str,
        indent: &Indent,
        is_last: bool,
    ) -> std::io::Result<()> {
        let BinaryOpExpr {
            lhs,
            op,
            rhs,
            span: _,
            operand_ty: _,
        } = e;
        indent.print(w)?;
        writeln!(w, "{prefix}BinaryOp({op:?})")?;
        let new_indent = indent.child(is_last);
        Self::print_expr(w, lhs, &new_indent, false)?;
        Self::print_expr(w, rhs, &new_indent, true)?;
        Ok(())
    }

    /// Writes a function call, with its arguments below it.
    ///
    /// # Errors
    ///
    /// Returns the error from `w` when a write fails.
    fn print_function_call(
        w: &mut impl Write,
        e: &FunctionCallExpr,
        prefix: &str,
        indent: &Indent,
        is_last: bool,
    ) -> std::io::Result<()> {
        let FunctionCallExpr {
            name,
            type_args: _,
            args,
            span: _,
        } = e;
        indent.print(w)?;
        writeln!(w, "{prefix}Call({name})")?;
        let new_indent = indent.child(is_last);
        for (i, arg) in args.iter().enumerate() {
            Self::print_expr(w, arg, &new_indent, i == args.len() - 1)?;
        }
        Ok(())
    }

    /// Writes a method call, with the receiver and then the arguments below it.
    ///
    /// # Errors
    ///
    /// Returns the error from `w` when a write fails.
    fn print_method_call(
        w: &mut impl Write,
        e: &MethodCallExpr,
        prefix: &str,
        indent: &Indent,
        is_last: bool,
    ) -> std::io::Result<()> {
        let MethodCallExpr {
            base: expr,
            method_name: method,
            type_args: _,
            args,
            span: _,
        } = e;
        indent.print(w)?;
        writeln!(w, "{prefix}MethodCall(.'{method}')")?;
        let new_indent = indent.child(is_last);
        Self::print_expr(w, expr, &new_indent, args.is_empty())?;
        for (i, arg) in args.iter().enumerate() {
            Self::print_expr(w, arg, &new_indent, i == args.len() - 1)?;
        }
        Ok(())
    }

    /// Writes a `comptime` block, with its statements and its final value below it.
    ///
    /// # Errors
    ///
    /// Returns the error from `w` when a write fails.
    fn print_comptime_block(
        w: &mut impl Write,
        e: &ComptimeBlockExpr,
        prefix: &str,
        indent: &Indent,
        is_last: bool,
    ) -> std::io::Result<()> {
        let ComptimeBlockExpr {
            stmts,
            ret,
            span: _,
        } = e;
        indent.print(w)?;
        writeln!(w, "{prefix}ComptimeBlock")?;
        let new_indent = indent.child(is_last);
        Self::print_statements(w, stmts, &new_indent)?;
        if let Some(r) = ret {
            Self::print_expr(w, r, &new_indent, true)?;
        }
        Ok(())
    }
}
