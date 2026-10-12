//===- check submodule - Vx Compiler ----------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// One expression family's type checks, split out of the former 5000-line `hir/expr.rs` along the
// `check_expr_type_flag` dispatch seam. An additional
// `impl TypeChecker` block; moves only, zero logic change. Reaches the shared types and helpers via
// `use super::super::*`, exactly as `expr.rs` uses `use super::*`.
//
//===----------------------------------------------------------------------===//

use super::super::{
    decl, ArrayExpr, ComptimeBlockExpr, ElementType, Expr, ExprStmtStmt, IdentifierExpr,
    LetDeclStmt, MatchExpr, MemberAccessExpr, NumberExpr, Pattern, RangeExpr, ReturnStmt, Span,
    Statement, StringLiteralExpr, StructInitExpr, Type, TypeChecker, UnsafeBlockExpr, Value,
};
use crate::hir::check_state::ComptimeEvalFlow;
use crate::syntax::expr::{ClosureExpr, IfExpr, MatchArm};
use std::collections::HashMap;

/// The value a `match` looks at, and how the `match` itself is used.
struct MatchedValue<'a> {
    /// The expression being matched.
    expr: &'a Expr,
    /// Its type.
    ty: &'a Type,
    /// Whether it is a variable or what a reference points at, which matching only reads.
    place: bool,
    /// Whether nothing uses the value of the `match`.
    unused: bool,
    /// Whether the value of each arm is moved out of it.
    consume: bool,
}

/// What the arms of a `match` checked so far have shown.
struct ArmsSeen {
    /// The type of the first arm that ends in a value.
    match_ty: Option<Type>,
    /// The enum variants the arms name.
    covered: Vec<crate::symbol::Symbol>,
    /// Whether an arm matches every value: `_`, or a name.
    has_wildcard: bool,
    /// The moved variables at the end of the arms that reach the code after the `match`.
    joined: Option<Vec<std::collections::HashSet<String>>>,
    /// Whether every arm leaves early, so the code after the `match` is never reached.
    every_arm_ends: bool,
}

/// What running a `comptime` block produced.
enum ComptimeFold {
    /// A value, which replaces the block. Boxed because it dwarfs the other two.
    Folded(Box<Expr>),
    /// It ran and left no value, so the block goes away entirely.
    NoValue,
    /// It could not be run. Reported already, unless this is a closure body -- which is not
    /// asked to fold where it is written.
    Refused,
}

impl TypeChecker<'_> {
    /// The constant environment as one map, the innermost scope winning.
    pub(crate) fn consteval_snapshot(&self) -> HashMap<crate::symbol::Symbol, Value> {
        let mut env = HashMap::new();
        for scope in &self.consteval.env {
            for (k, v) in scope {
                env.insert(k.clone(), v.clone());
            }
        }
        env
    }

    /// Run a `comptime` block and answer the constant it produced, if it has one.
    ///
    /// A block that cannot be run is reported here. It exists to run during compilation and
    /// leave nothing behind, so there is no such thing as one that half-ran: until now those
    /// were emitted as ordinary run-time code, which is how a loop or a call the evaluator
    /// skipped stayed invisible.
    fn fold_comptime_block(
        &mut self,
        stmts: &[Statement],
        ret: Option<&Expr>,
        block_span: Span,
        before: &HashMap<crate::symbol::Symbol, Value>,
    ) -> ComptimeFold {
        let interpretation = self.observe_comptime_block(stmts, ret, before);
        // At the block's last expression; at the `comptime` keyword when there is none, or when
        // it has no position of its own, as a number literal does.
        let span = ret
            .map(crate::syntax::expr::Expr::span)
            .filter(|span| *span != Span::default())
            .unwrap_or(block_span);

        // Anything it writes that outlives it would have to survive, and the block does not.
        if let Some(name) = interpretation.escaping_write {
            self.report_comptime_block_failure(
                &format!(
                    "it writes to '{name}', which is declared outside it -- the block disappears, \
                     so the write would have to disappear with it"
                ),
                &span,
            );
            return ComptimeFold::Refused;
        }
        if interpretation.call_depth_exceeded {
            self.report_comptime_depth_exceeded(&span);
            return ComptimeFold::Refused;
        }
        if interpretation.outcome.requires_refusal || !interpretation.outcome.support.is_supported()
        {
            self.report_comptime_block_failure(
                "it holds a statement the evaluator cannot run",
                &span,
            );
            return ComptimeFold::Refused;
        }
        // A `return` inside the block, where the block is a closure or function body, is
        // that body's value -- `|| comptime { ..; return x; }` is how the closure fixtures
        // are written. Answer with it, the same as a trailing expression.
        if interpretation.outcome.flow == ComptimeEvalFlow::Return {
            return self.fold_value(interpretation.outcome.value.concrete, &span);
        }
        if interpretation.outcome.flow != ComptimeEvalFlow::Normal {
            self.report_comptime_block_failure(
                "its control flow does not complete normally",
                &span,
            );
            return ComptimeFold::Refused;
        }
        if ret.is_none() {
            return ComptimeFold::NoValue;
        }
        self.fold_value(interpretation.outcome.value.concrete, &span)
    }

    fn report_comptime_depth_exceeded(&mut self, span: &Span) {
        if self.speculating
            || self.consteval.closure_body_depth > 0
            || self
                .errors
                .has_error(crate::diagnostic::DiagnosticCode::E8004)
        {
            return;
        }
        self.errors.error_with_code(
            crate::diagnostic::DiagnosticCode::E8004,
            format!(
                "compile-time evaluation went more than {} calls deep and was stopped. A \
                 recursive function whose base case is never reached is the usual cause.",
                crate::hir::check_state::MAX_CALL_DEPTH
            ),
            Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
        );
    }

    fn fold_value(&mut self, value: Option<Value>, span: &Span) -> ComptimeFold {
        let Some(value) = value else {
            self.report_comptime_block_failure("its value cannot be worked out", span);
            return ComptimeFold::Refused;
        };
        if let Some(expr) = Self::value_to_expr(&value, span) {
            ComptimeFold::Folded(Box::new(expr))
        } else {
            self.report_comptime_block_failure(
                "its value is not one that can be written as a constant",
                span,
            );
            ComptimeFold::Refused
        }
    }

    fn report_comptime_block_failure(&mut self, why: &str, span: &Span) {
        // A closure body is not asked to fold where it is written -- its parameters have no
        // values yet. The call is what has to fold, and that is reported at the call.
        if self.speculating || self.consteval.closure_body_depth > 0 {
            return;
        }
        self.errors.error_with_code(
            crate::diagnostic::DiagnosticCode::E3033,
            format!("this `comptime` block cannot be evaluated: {why}"),
            Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
        );
    }

    /// Whether a statement binds a lambda whose body is a `comptime` block.
    ///
    /// Every call to one has folded by now, so nothing is left to call and the binding is
    /// dropped. Keeping it would emit the closure and its generated body, which is exactly
    /// the run-time code a `comptime` block must not leave behind.
    pub(crate) fn binds_a_comptime_lambda(stmt: &Statement) -> bool {
        let Statement::LetDecl(decl) = stmt else {
            return false;
        };
        let Expr::Closure(closure) = &decl.expr else {
            return false;
        };
        matches!(&*closure.body, Expr::ComptimeBlock(_))
    }

    /// Write a computed value back as a constant expression.
    ///
    /// Only immutable values: numbers, booleans and arrays of them. Anything else has no
    /// spelling that survives to run time on its own.
    pub(crate) fn value_to_expr(value: &Value, span: &Span) -> Option<Expr> {
        match value {
            Value::Int(i) => Some(Expr::Number(NumberExpr {
                value: i.to_string().into(),
                ty: None,
                span: *span,
            })),
            Value::Number(n) => Some(Expr::Number(NumberExpr {
                value: n.to_string().into(),
                ty: None,
                span: *span,
            })),
            Value::CStr(text) => Some(Expr::StringLiteral(StringLiteralExpr {
                value: text.as_str().into(),
                span: *span,
            })),
            Value::Bool(b) => Some(Expr::Identifier(IdentifierExpr {
                name: if *b { "true".into() } else { "false".into() },
                span: *span,
            })),
            Value::Array(items) => {
                let mut elements = Vec::with_capacity(items.len());
                for item in items {
                    elements.push(Self::value_to_expr(item, span)?);
                }
                Some(Expr::Array(ArrayExpr {
                    elements,
                    span: *span,
                }))
            }
            _ => None,
        }
    }

    /// Type check a `comptime` block, run it, and replace it with the value it worked out.
    /// Returns the type of that value.
    ///
    /// # Panics
    ///
    /// If `expr` is not a `comptime` block.
    pub(crate) fn check_comptimeblock_expr(&mut self, expr: &mut Expr, consume: bool) -> Type {
        let unused = std::mem::replace(&mut self.value_unused, false);
        match expr {
            Expr::ComptimeBlock(ComptimeBlockExpr {
                stmts,
                ret,
                span: block_span,
            }) => {
                let block_span = *block_span;
                // What is known before the block runs. Its own `let`s live in the scope
                // pushed below and are gone by the time the fold needs them, so the block
                // is run again from here rather than read out of the checker's environment.
                // One `comptime` inside another asks for nothing the outer one does not
                // already do, and it is what put a closure's body out of the evaluator's
                // reach: the inner block's value needed a call to a closure declared in the
                // outer block.
                if self.consteval.comptime_depth > 0 && !self.speculating {
                    self.errors.error_with_code(
                        crate::diagnostic::DiagnosticCode::E3034,
                        "a `comptime` block inside another one: the outer block already runs \
                         while compiling, so remove the inner `comptime`"
                            .to_string(),
                        Some(crate::diagnostic::SourceSpan::from_ast_span(&block_span)),
                    );
                }
                let before = self.consteval_snapshot();
                self.push_scope();
                self.consteval.comptime_depth += 1;
                self.value_unused = unused;
                let mut ret_ty = self.check_expr_block(stmts, ret.as_deref(), consume);
                if let Some(r) = ret {
                    ret_ty = self.check_expr_type(r);
                }
                self.consteval.comptime_depth -= 1;
                self.pop_scope();
                let folded = self.fold_comptime_block(stmts, ret.as_deref(), block_span, &before);
                // An assertion the placement fold answered `true` is discharged here: nothing
                // at run time holds a placement, so nothing is left to check. (A false one
                // was reported by the assert check.) Asserts on anything else stay, as the
                // program wrote them.
                stmts.retain(|s| {
                    !matches!(s, Statement::Assert(a)
                        if matches!(&*a.expr, Expr::Identifier(id) if id.name.as_ref() == "true"))
                });
                stmts.retain(|s| !Self::binds_a_comptime_lambda(s));
                // The block ran while compiling and must leave nothing behind, so what it
                // worked out replaces it. A block with no value becomes an empty one, which
                // the statement walk then drops.
                match folded {
                    // The value it worked out replaces it, as a literal of the block's type:
                    // an untyped literal is an `i32` to the legacy code generator, which cut
                    // an `i64` like 20! down to its low 32 bits.
                    ComptimeFold::Folded(mut value) => {
                        if let (Expr::Number(n), Type::Scalar(elem)) = (&mut *value, &ret_ty) {
                            if n.ty.is_none() {
                                n.ty = Some(elem.clone());
                            }
                        }
                        *expr = *value;
                    }
                    // It ran and produced nothing, so nothing is left to emit.
                    ComptimeFold::NoValue => {
                        *expr = Expr::ComptimeBlock(ComptimeBlockExpr {
                            stmts: Vec::new(),
                            ret: None,
                            span: block_span,
                        });
                    }
                    // Left as written: either the error above stops the build, or this is a
                    // closure body, which folds at its call rather than here.
                    ComptimeFold::Refused => {}
                }
                ret_ty
            }

            _ => panic!("Expected IndexAccess, got {expr:?}"),
        }
    }

    /// Whether a block leaves the function, so that nothing after the `if` it belongs to is
    /// reached along this path.
    ///
    /// Only `return` counts. `break` and `continue` also stop the code right after the `if`
    /// from running, but they go somewhere else in the same function -- the statement after
    /// the loop, or its next turn -- and a value moved on the way there is just as moved when
    /// it arrives. Treating them like `return` would drop a mark that is still true.
    ///
    /// A block that ends some other way is answered `false`, including one whose last
    /// statement is an `if` both of whose arms return. That is the conservative direction:
    /// the mark is kept and the program refused, rather than a moved value let through.
    /// The moves at the end of a branch that was just checked, when it reaches the code after
    /// it. A branch that ends in `break` or `continue` instead hands its moves to the loop it
    /// leaves or restarts; one that ends in `return` hands them to nobody.
    pub(crate) fn branch_end(
        &mut self,
        stmts: &[Statement],
    ) -> Option<Vec<std::collections::HashSet<String>>> {
        let exit = stmts.iter().find_map(|s| match s {
            Statement::Return(_) => Some(0),
            Statement::Break(_) => Some(1),
            Statement::Continue(_) => Some(2),
            _ => None,
        });
        let marks = self.moved_snapshot();
        let slot = match (exit, self.borrow.loop_exits.last_mut()) {
            (None, _) => return Some(marks),
            (Some(1), Some(l)) => &mut l.at_break,
            (Some(2), Some(l)) => &mut l.at_continue,
            _ => return None,
        };
        *slot = Some(match slot.take() {
            Some(j) => Self::union_moved(j, marks),
            None => marks,
        });
        None
    }

    /// Whether control never reaches the end of `stmts`: it returns, breaks, continues, panics,
    /// aborts or calls a function declared `-> !` on every path.
    fn branch_ends_early(&self, stmts: &[Statement]) -> bool {
        let functions = &self.env.functions;
        let never = |name: &str| {
            functions
                .get(name)
                .is_some_and(|f| crate::syntax::is_never_ty(&f.0))
        };
        stmts.iter().any(|s| {
            matches!(
                s,
                Statement::Return(_) | Statement::Break(_) | Statement::Continue(_)
            ) || crate::syntax::expr::statement_always_exits(s, &never)
        })
    }

    /// Decide a `comptime if` while compiling: only the branch its condition selects is kept.
    fn fold_comptime_if(&mut self, if_expr: &mut IfExpr) {
        let tmp_env = self.consteval_snapshot();
        if let Some(Value::Bool(b)) = self.eval_expr(&if_expr.cond, &tmp_env) {
            if b {
                if_expr.else_block = None;
            } else {
                if_expr.then_block.clear();
            }
        } else {
            self.errors
                .push("Cannot statically evaluate comptime if condition".to_string());
        }
    }

    /// Type check the `then` branch of an `if`, in a scope of its own. Returns its type.
    fn check_then_branch(&mut self, if_expr: &mut IfExpr, unused: bool, consume: bool) -> Type {
        self.push_releasing_scope();
        let mut then_ty = Type::Struct("void".into(), None);
        if !self.speculating && !if_expr.then_block.is_empty() {
            self.value_unused = unused;
            then_ty = self.check_expr_block(&mut if_expr.then_block, None, consume);
        }
        self.pop_scope();
        then_ty
    }

    /// Type check a non-empty `else` branch, in a scope of its own. Returns its type, and the
    /// moves at its end as [`Self::branch_end`] gives them.
    fn check_else_branch(
        &mut self,
        else_b: &mut Vec<Statement>,
        unused: bool,
        consume: bool,
    ) -> (Type, Option<Vec<std::collections::HashSet<String>>>) {
        let mut else_ty = Type::Struct("void".into(), None);
        self.push_releasing_scope();
        if !self.speculating {
            self.value_unused = unused;
            else_ty = self.check_expr_block(else_b, None, consume);
        }
        self.pop_scope();
        let marks_after_else = self.branch_end(else_b);
        (else_ty, marks_after_else)
    }

    /// Make the two branches of an `if` agree on one type, and report an error when they
    /// cannot.
    fn reconcile_branch_types(&mut self, if_expr: &IfExpr, then_ty: &mut Type, else_ty: &mut Type) {
        // A branch that never finishes, `todo()`, has no value: the other branch decides.
        if crate::syntax::is_never_ty(then_ty) {
            *then_ty = else_ty.clone();
        } else if crate::syntax::is_never_ty(else_ty) {
            *else_ty = then_ty.clone();
        }
        if !if_expr.is_comptime && then_ty != else_ty {
            self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E3007,
                format!(
                    "If expression branches have incompatible types: {then_ty:?} and {else_ty:?}"
                ),
                Some(crate::diagnostic::SourceSpan::from_ast_span(&if_expr.span)),
            );
        }
    }

    /// An `if` with no `else`: it has no value, which is an error unless nothing uses the value.
    fn check_if_without_else(&mut self, if_expr: &IfExpr, unused: bool, then_ty: &mut Type) {
        // After the error the `if` keeps its `then` branch's type, so that one mistake
        // gives one error.
        if unused {
            *then_ty = Type::Struct("void".into(), None);
        } else {
            self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E3048,
                "this `if` has no `else`, so it has no value when its condition is false; \
                 add an `else`, or use the `if` as a statement"
                    .to_string(),
                Some(crate::diagnostic::SourceSpan::from_ast_span(&if_expr.span)),
            );
        }
    }

    /// Keep, for the code after an `if`, the facts the prover may still use there.
    fn keep_facts_after_if(
        &mut self,
        if_expr: &IfExpr,
        then_facts: Vec<Expr>,
        else_facts: Vec<Expr>,
    ) {
        // When one branch ends early, the code after the `if` is reached only through the other,
        // so what that branch knew at its end still holds. When both end early, it is not reached.
        let then_ends = self.branch_ends_early(&if_expr.then_block);
        let else_ends = if_expr
            .else_block
            .as_ref()
            .is_some_and(|b| self.branch_ends_early(b));
        match (then_ends, else_ends) {
            (true, false) => self.consteval.constraints.extend(else_facts),
            (false, true) => self.consteval.constraints.extend(then_facts),
            (true, true) => self
                .consteval
                .constraints
                .push(crate::hir::prover::unreachable_fact()),
            (false, false) => {}
        }
    }

    /// Type check an `if`: its condition, both branches, and which variables are moved after
    /// it. Returns the type of the value the `if` produces.
    ///
    /// # Panics
    ///
    /// If `expr` is not an `if`.
    pub(crate) fn check_if_expr(&mut self, expr: &mut Expr, consume: bool) -> Type {
        let unused = std::mem::replace(&mut self.value_unused, false);
        let Expr::If(if_expr) = expr else {
            panic!("Expected IndexAccess, got {expr:?}");
        };

        let cond_ty = self.check_expr_type(&mut if_expr.cond);
        if cond_ty != Type::Scalar(ElementType::Bool) {
            self.errors
                .push("Condition in if expression must be of type bool (i1)".to_string());
        }

        if if_expr.is_comptime {
            self.fold_comptime_if(if_expr);
        }

        // Each branch starts from the moves made before the `if`. Afterwards a variable is
        // moved if a branch that reaches the code after the `if` moved it; a branch that
        // returns or breaks does not count.
        let marks_before = self.moved_snapshot();
        // The prover's facts follow the paths: the `then` branch knows the condition and the
        // `else` branch its negation, and what a branch learns ends with it.
        let facts_before = self.consteval.constraints.len();
        let cond_fact = (!if_expr.is_comptime && crate::hir::prover::is_modelled(&if_expr.cond))
            .then(|| (*if_expr.cond).clone());
        if let Some(c) = &cond_fact {
            self.consteval.constraints.push(c.clone());
        }
        let mut then_ty = self.check_then_branch(if_expr, unused, consume);
        let then_facts = self.consteval.constraints.split_off(facts_before);
        if let Some(c) = &cond_fact {
            self.consteval
                .constraints
                .push(crate::hir::prover::negate(c));
        }
        let marks_after_then = self.branch_end(&if_expr.then_block);
        self.restore_moved(marks_before.clone());
        let mut marks_after_else = Some(marks_before.clone());

        let mut else_ty = Type::Struct("void".into(), None);
        if let Some(else_b) = if_expr.else_block.as_mut() {
            if !else_b.is_empty() {
                (else_ty, marks_after_else) = self.check_else_branch(else_b, unused, consume);
                self.reconcile_branch_types(if_expr, &mut then_ty, &mut else_ty);
            }
        } else if !if_expr.is_comptime {
            self.check_if_without_else(if_expr, unused, &mut then_ty);
        }
        let else_facts = self.consteval.constraints.split_off(facts_before);
        self.keep_facts_after_if(if_expr, then_facts, else_facts);
        let joined = match (marks_after_then, marks_after_else) {
            (Some(a), Some(b)) => Self::union_moved(a, b),
            (Some(a), None) | (None, Some(a)) => a,
            (None, None) => marks_before,
        };
        self.restore_moved(joined);

        // If it was comptime evaluated to false, the return type should just be the else block type
        if if_expr.is_comptime {
            if if_expr.then_block.is_empty() {
                return else_ty;
            }
            return then_ty;
        }

        then_ty
    }

    /// Type check an `unsafe` block. Returns the type of the value it produces.
    ///
    /// # Panics
    ///
    /// If `expr` is not an `unsafe` block.
    pub(crate) fn check_unsafeblock_expr(&mut self, expr: &mut Expr, consume: bool) -> Type {
        let unused = std::mem::replace(&mut self.value_unused, false);
        match expr {
            Expr::UnsafeBlock(UnsafeBlockExpr {
                stmts,
                ret: ret_expr,
                span: _,
            }) => {
                let prev_unsafe = self.in_unsafe_block;
                self.in_unsafe_block = true;
                // `unsafe { let v = ..; v }`, as `vec![..]` writes: the block's drops are placed
                // when its statements are checked, so the value is moved out of `v` by one of
                // them, into a `$value` that owns nothing.
                if let Some(Expr::Identifier(id)) = ret_expr.as_deref() {
                    if !id.name.starts_with("$value") && !stmts.is_empty() {
                        let value = Statement::LetDecl(crate::syntax::stmt::LetDeclStmt {
                            name: "$value".into(),
                            is_mut: false,
                            ty_ann: None,
                            expr: Expr::Identifier(id.clone()),
                            span: id.span,
                        });
                        stmts.push(value);
                        *ret_expr = Some(Box::new(Expr::Identifier(IdentifierExpr {
                            name: "$value".into(),
                            span: id.span,
                        })));
                    }
                }
                self.push_scope();
                self.value_unused = unused;
                let mut ret_ty = self.check_expr_block(stmts, ret_expr.as_deref(), consume);
                if let Some(r) = ret_expr {
                    self.value_unused = unused;
                    ret_ty = self.check_expr_type_flag(r, consume);
                }
                self.pop_scope();
                self.in_unsafe_block = prev_unsafe;
                ret_ty
            }
            _ => panic!("Expected IndexAccess, got {expr:?}"),
        }
    }

    /// Type check a range such as `0..n`. Returns the type of its two ends.
    ///
    /// # Panics
    ///
    /// If `expr` is not a range.
    pub(crate) fn check_range_expr(&mut self, expr: &mut Expr) -> Type {
        match expr {
            Expr::Range(RangeExpr {
                start,
                end,
                span: _,
            }) => {
                // Reconcile the bounds so an untyped literal adopts the other bound's type
                // (`0..n` with `n: i64` → `0` becomes i64), mirroring binary-operand inference (#240).
                let (start_ty, end_ty) = self.check_operand_pair(start, end, true);
                if start_ty != end_ty {
                    self.errors.push(format!(
                        "Range start and end types must match, got {start_ty:?} and {end_ty:?}"
                    ));
                }
                start_ty
            }
            _ => panic!("Expected IndexAccess, got {expr:?}"),
        }
    }

    /// The variant names of `ty`'s enum declaration, or `None` when it is not an enum this
    /// compilation declares (so exhaustiveness cannot be decided by enumeration).
    fn enum_variants_of(&self, ty: &Type) -> Option<Vec<crate::symbol::Symbol>> {
        let name = match ty {
            Type::Enum(n, _) | Type::Struct(n, _) => n.clone(),
            Type::GenericInstance(inner, _) => match &**inner {
                Type::Enum(n, _) | Type::Struct(n, _) => n.clone(),
                _ => return None,
            },
            _ => return None,
        };
        let base = match name.find('<') {
            Some(i) => crate::symbol::Symbol::from(&name[..i]),
            None => name,
        };
        self.env
            .enums
            .get(base.as_ref())
            .map(|d| d.variants.iter().map(|v| v.0.clone()).collect())
    }

    /// Refuse a `match` arm whose integer literal cannot be represented in the scrutinee's type.
    ///
    /// Such an arm can never be selected. Codegen parses the literal with a zero fallback, which
    /// silently turns it into a comparison against 0 -- so the arm fires for scrutinee 0 instead
    /// of never. Reported here, where the scrutinee's type is known.
    pub(crate) fn check_literal_pattern_range(&mut self, pattern: &Pattern, expr_ty: &Type) {
        if self.speculating {
            return;
        }
        let Pattern::Literal(Expr::Number(n)) = pattern else {
            return;
        };
        let Type::Scalar(elem) = expr_ty else {
            return;
        };
        if elem.accepts_integer_literal(n.value.as_ref()) == Some(false) {
            self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E3019,
                format!(
                    "match arm literal '{}' is not representable in the scrutinee's type '{}', \
                     so the arm can never be selected",
                    n.value, elem
                ),
                Some(crate::diagnostic::SourceSpan::from_ast_span(&n.span)),
            );
        }
    }

    /// Whether `pattern`, just bound, takes a value that has to move: a name bound to
    /// something linear that is not `Copy`.
    fn pattern_moves(&mut self, pattern: &Pattern) -> bool {
        fn names(p: &Pattern, out: &mut Vec<crate::symbol::Symbol>) {
            match p {
                Pattern::Identifier(n) => out.push(n.clone()),
                Pattern::EnumVariant(_, _, Some(parts)) => {
                    for part in parts {
                        names(part, out);
                    }
                }
                _ => {}
            }
        }
        let mut bound = Vec::new();
        names(pattern, &mut bound);
        bound.iter().any(|n| {
            let ty = self.lookup(n).map(|(t, _)| t.clone());
            ty.is_some_and(|t| t.is_linear() && !self.is_copy(&t))
        })
    }

    /// An arm of `match scrutinee` took its value: the variable is moved, and what a
    /// reference points at cannot be.
    fn match_moves(&mut self, scrutinee: &Expr) {
        match scrutinee {
            Expr::Identifier(id) => self.consume(&id.name),
            Expr::Dereference(d) => {
                let inner = self.check_expr_type_flag(&mut (*d.expr).clone(), false);
                if let Type::Borrow { inner, .. } = inner {
                    if !self.is_copy(&inner) {
                        self.errors.error_with_code(
                            crate::diagnostic::DiagnosticCode::E4008,
                            format!(
                                "this moves a value of type `{inner}` out from behind a \
                                 reference, which does not own it; copy or clone it instead"
                            ),
                            Some(crate::diagnostic::SourceSpan::from_ast_span(&d.span)),
                        );
                    }
                }
            }
            _ => {}
        }
    }

    pub(crate) fn bind_pattern_variables(&mut self, pattern: &Pattern, expr_ty: &Type) {
        match pattern {
            Pattern::Identifier(name) => {
                self.insert(name.to_string(), expr_ty.clone());
            }
            Pattern::EnumVariant(enum_name, variant_name, Some(payloads)) => {
                let mut base_name = enum_name.clone();
                if let Some(idx) = enum_name.find('<') {
                    base_name = enum_name[..idx].to_string().into();
                }
                if let Some(enum_decl) = self.env.enums.get(base_name.as_ref()) {
                    if let Some(variant) = enum_decl.variants.iter().find(|v| v.0 == *variant_name)
                    {
                        if let Some(payload_types) = &variant.1 {
                            let mut mapping = std::collections::HashMap::new();
                            if let Type::GenericInstance(_, args) = expr_ty {
                                for (i, param) in enum_decl.generics.iter().enumerate() {
                                    if i < args.len() {
                                        mapping.insert(param.name().into(), args[i].clone());
                                    }
                                }
                            }
                            for (i, p) in payloads.iter().enumerate() {
                                if let Pattern::Identifier(name) = p {
                                    if i < payload_types.len() {
                                        let p_ty = payload_types[i].substitute(&mapping);
                                        self.insert(name.to_string(), p_ty);
                                    } else {
                                        self.insert(name.to_string(), Type::Unknown);
                                    }
                                }
                            }
                        }
                    }
                } else {
                    for p in payloads {
                        if let Pattern::Identifier(name) = p {
                            self.insert(name.to_string(), Type::Unknown);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    /// Type check one arm of a `match`, and add what it shows to `seen`.
    fn check_match_arm(&mut self, arm: &mut MatchArm, on: &MatchedValue<'_>, seen: &mut ArmsSeen) {
        match &arm.pattern {
            Pattern::Wildcard | Pattern::Identifier(_) => seen.has_wildcard = true,
            Pattern::EnumVariant(_, variant, _) => seen.covered.push(variant.clone()),
            Pattern::Literal(_) => {}
        }
        // `_ => {}` and an empty enum variant arm are there to make the match cover
        // every case, so they stay quiet. Every arm the compiler writes is one of them.
        if arm.body.is_empty()
            && !matches!(arm.pattern, Pattern::Wildcard | Pattern::EnumVariant(..))
            && !self.speculating
        {
            self.errors.warn(
                crate::diagnostic::DiagnosticCode::W1008,
                "this match arm's body is empty, so this case does nothing",
                Some(crate::diagnostic::SourceSpan::from_ast_span(&arm.span)),
            );
        }
        self.push_scope();
        self.check_literal_pattern_range(&arm.pattern, on.ty);
        self.bind_pattern_variables(&arm.pattern, on.ty);
        if on.place && !self.speculating && self.pattern_moves(&arm.pattern) {
            // Moved before the arm's own names exist, which may hide it.
            self.pop_scope();
            self.match_moves(on.expr);
            self.push_scope();
            self.bind_pattern_variables(&arm.pattern, on.ty);
        }

        let arm_ty = if self.speculating {
            Type::Struct("void".into(), None)
        } else {
            self.value_unused = on.unused;
            self.check_expr_block(&mut arm.body, None, on.consume)
        };
        self.pop_scope();
        seen.every_arm_ends &= self.branch_ends_early(&arm.body);
        if let Some(after) = self.branch_end(&arm.body) {
            seen.joined = Some(match seen.joined.take() {
                Some(j) => Self::union_moved(j, after),
                None => after,
            });
        }

        // An arm contributes to the match's value type only if its block
        // ends in a tail expression (a non-semicolon `ExprStmt`). Arms that
        // end in `return`/`break`/`continue`, or in a statement like `assert`,
        // diverge or yield unit and do not constrain the match value — this is
        // what lets `Option::unwrap` (Some arm `return v`, None arm asserts)
        // type-check as `-> T`.
        let yields_value = matches!(
            arm.body.last(),
            Some(Statement::ExprStmt(ExprStmtStmt {
                has_semi: false,
                ..
            }))
        );
        if yields_value && seen.match_ty.is_none() {
            seen.match_ty = Some(arm_ty);
        }
    }

    /// Report a `match` with no arm that matches every value, when it leaves a value uncovered.
    ///
    /// An enum scrutinee can be enumerated, so a missing variant is a fact
    /// about the match wherever it sits: the uncovered value falls through, and if
    /// every written arm returns, the function falls off its end carrying whatever
    /// was in the return slot. A scrutinee that cannot be enumerated has no such
    /// fact, so asking it for a wildcard stays limited to value position, where the
    /// fall-through edge would otherwise have no value to carry.
    fn report_uncovered_match(
        &mut self,
        expr_ty: &Type,
        covered: &[crate::symbol::Symbol],
        produces_value: bool,
    ) {
        match self.enum_variants_of(expr_ty) {
            Some(all) => {
                let missing: Vec<String> = all
                    .iter()
                    .filter(|v| !covered.contains(v))
                    .map(std::string::ToString::to_string)
                    .collect();
                if !missing.is_empty() {
                    self.errors.error_with_code(
                        crate::diagnostic::DiagnosticCode::E3020,
                        format!(
                            "this `match` does not cover {}; add an arm for each, \
                             or a `_` arm",
                            missing.join(", ")
                        ),
                        None,
                    );
                }
            }
            None => {
                if produces_value {
                    self.errors.error_with_code(
                        crate::diagnostic::DiagnosticCode::E3020,
                        "this `match` produces a value but has no arm matching every \
                         value; add a `_` arm so every path yields one"
                            .to_string(),
                        None,
                    );
                }
            }
        }
    }

    /// Type check a `match`: each arm, which variables the arms move, and whether the arms
    /// cover every value. Returns the type of the value the `match` produces.
    ///
    /// # Panics
    ///
    /// If `expr` is not a `match`.
    pub(crate) fn check_match_expr(&mut self, expr: &mut Expr, consume: bool) -> Type {
        let unused = std::mem::replace(&mut self.value_unused, false);
        match expr {
            Expr::Match(MatchExpr {
                expr: match_expr,
                arms,
                span: _,
            }) => {
                // Matching a variable, or what a reference points at, reads it: only an arm
                // that binds a value which has to move moves it (below). Any other scrutinee
                // is a value the match takes.
                let place = matches!(**match_expr, Expr::Identifier(_) | Expr::Dereference(_));
                let expr_ty = if place {
                    self.check_expr_type_flag(match_expr, false)
                } else {
                    self.check_expr_type(match_expr)
                };
                let on = MatchedValue {
                    expr: match_expr,
                    ty: &expr_ty,
                    place,
                    unused,
                    consume,
                };

                let mut seen = ArmsSeen {
                    match_ty: None,
                    covered: Vec::new(),
                    has_wildcard: false,
                    joined: None,
                    every_arm_ends: !arms.is_empty(),
                };
                // Each arm starts from the moves made before the `match`; afterwards a
                // variable is moved if an arm that reaches the following code moved it.
                let marks_before = self.moved_snapshot();
                // What an arm learns ends with it: the prover does not model the patterns, so
                // it cannot say which arm the code after the `match` came through.
                let facts_before = self.consteval.constraints.len();
                for arm in arms {
                    self.consteval.constraints.truncate(facts_before);
                    self.restore_moved(marks_before.clone());
                    self.check_match_arm(arm, &on, &mut seen);
                }
                self.restore_moved(seen.joined.unwrap_or(marks_before));
                self.consteval.constraints.truncate(facts_before);
                if seen.every_arm_ends {
                    self.consteval
                        .constraints
                        .push(crate::hir::prover::unreachable_fact());
                }

                if !seen.has_wildcard && !self.speculating {
                    self.report_uncovered_match(&expr_ty, &seen.covered, seen.match_ty.is_some());
                }

                // When no arm yields a value the match sits in diverging/statement
                // position (every arm returns or aborts). Type it as the expected return
                // type so an implicit `return match { ... }` type-checks; fall back to the
                // historical placeholder when there is no expected return type. A match
                // whose value nothing uses is a statement, and has no value at all.
                if unused && seen.match_ty.is_none() {
                    return Type::Struct("void".into(), None);
                }
                seen.match_ty.unwrap_or_else(|| {
                    self.current_return_type.clone().unwrap_or(Type::Tensor(
                        ElementType::F32,
                        vec![],
                        None,
                    ))
                })
            }
            _ => panic!("Expected IndexAccess, got {expr:?}"),
        }
    }

    /// Type check the body of the closure `e` as a function of its own. Returns the checked
    /// body and the type it returns.
    fn check_closure_body(&mut self, e: &ClosureExpr) -> (Box<Expr>, Type) {
        let closure_depth = self.scopes.len();
        self.mono.closure_depths.push(closure_depth);
        self.mono.closure_captures_stack.push(HashMap::new());

        let old_ret = self.current_return_type.take();
        self.current_return_type = Some(Type::Unknown);

        self.push_scope();
        for (name, ty) in &e.params {
            self.insert(name.to_string(), ty.clone());
        }
        let mut b = e.body.clone();
        // A closure is a function, so its body starts a fresh compile-time context.
        // `comptime` directly inside `comptime` is the thing being refused; a lambda
        // whose body is a `comptime` block, written inside another one, is two
        // separate functions and is ordinary.
        let outer_comptime = std::mem::take(&mut self.consteval.comptime_depth);
        self.consteval.closure_body_depth += 1;
        // A `return` in the body leaves the closure, not the function around it.
        self.drops_enter_function(Vec::new(), &[]);
        let expr_ret_ty = self.check_expr_type(&mut b);
        self.drops_exit_function();
        self.consteval.closure_body_depth -= 1;
        self.consteval.comptime_depth = outer_comptime;

        let mut ret_ty = expr_ret_ty;
        if let Some(inferred) = &self.current_return_type {
            if *inferred != Type::Unknown {
                ret_ty = inferred.clone();
            }
        }

        self.pop_scope();
        self.current_return_type = old_ret;
        (b, ret_ty)
    }

    /// The outer variables that the closure just checked uses, sorted by name. A captured
    /// struct or tensor is moved into the closure.
    fn take_closure_captures(&mut self) -> Vec<(crate::symbol::Symbol, Type)> {
        let captured_vars_map = self.mono.closure_captures_stack.pop().unwrap_or_default();
        self.mono.closure_depths.pop();

        let mut captured_vars: Vec<(crate::symbol::Symbol, Type)> =
            captured_vars_map.into_iter().collect();
        captured_vars.sort_by(|a, b| a.0.cmp(&b.0)); // Stable layout

        if !self.speculating {
            // Consume captured variables in the outer scope if they are linear
            for (name, ty) in &captured_vars {
                if matches!(ty, Type::Struct(_, _) | Type::Tensor(_, _, _)) {
                    self.consume(name.as_ref());
                }
            }
        }
        captured_vars
    }

    /// The parameters of the function generated for a closure: a pointer to the struct that
    /// holds the captured variables, then the closure's own parameters.
    fn closure_call_params(
        struct_name: &str,
        params: &[(crate::symbol::Symbol, Type)],
    ) -> Vec<(crate::symbol::Symbol, Type)> {
        let mut env_params: Vec<(crate::symbol::Symbol, Type)> = vec![(
            "_env".to_string().into(),
            Type::Pointer(Box::new(Type::Struct(struct_name.into(), None)), None, true), // &mut env
        )];

        for (name, ty) in params {
            env_params.push((name.clone(), ty.clone()));
        }
        env_params
    }

    /// The body of the function generated for a closure: a `let` that reads each captured
    /// variable out of the struct, then a `return` of the closure's own body.
    fn closure_call_fn_body(
        struct_name: &str,
        captured_vars: &[(crate::symbol::Symbol, Type)],
        body: Expr,
        span: Span,
    ) -> Vec<Statement> {
        let mut body_stmts = Vec::new();
        for (cap_name, cap_ty) in captured_vars {
            let env_access = Expr::MemberAccess(MemberAccessExpr {
                base: Box::new(Expr::Identifier(IdentifierExpr::new(
                    "_env".to_string().into(),
                    span,
                ))),
                member: cap_name.clone(),
                struct_name: Some(struct_name.into()),
                span,
            });
            body_stmts.push(Statement::LetDecl(LetDeclStmt {
                name: cap_name.clone(),
                is_mut: true,
                ty_ann: Some(cap_ty.clone()),
                expr: env_access,
                span,
            }));
        }

        body_stmts.push(Statement::Return(ReturnStmt {
            expr: Some(body),
            span,
        }));
        body_stmts
    }

    /// The struct literal that replaces a closure: each captured variable, stored in the field
    /// that has its name.
    fn closure_env_init(
        struct_name: &str,
        captured_vars: &[(crate::symbol::Symbol, Type)],
        span: Span,
    ) -> Expr {
        let mut fields = Vec::new();
        for (cap_name, _) in captured_vars {
            fields.push((
                cap_name.clone(),
                Expr::Identifier(IdentifierExpr::new(cap_name.clone(), span)),
            ));
        }

        Expr::StructInit(StructInitExpr {
            name: struct_name.into(),
            fields,
            type_id: None,
            span,
        })
    }

    /// Type check a closure and replace it with the value that holds what it captures: a
    /// generated struct with one field for each captured variable. The closure's body becomes
    /// a generated function that takes that struct.
    ///
    /// # Panics
    ///
    /// If `expr` is not a closure.
    pub(crate) fn check_closure_expr(&mut self, expr: &mut Expr, _consume: bool) -> Type {
        match expr {
            Expr::Closure(e) => {
                let struct_name = format!("Closure_{}", self.next_id);
                let func_name = format!("{struct_name}_call");
                self.next_id += 1;

                let cloned_params = e.params.clone();
                let (b, ret_ty) = self.check_closure_body(e);
                let captured_vars = self.take_closure_captures();

                // Create StructDecl for the environment
                let struct_decl = decl::StructDecl {
                    name: struct_name.clone().into(),
                    generics: vec![],
                    fields: captured_vars.clone(),
                    doc_comment: None,
                };
                self.mono.generated_structs.push(struct_decl);

                // Create the Function for calling the closure
                let call_func = decl::Function {
                    is_unsafe: false,
                    name: func_name.into(),
                    generics: vec![],
                    params: Self::closure_call_params(&struct_name, &cloned_params),
                    mut_params: Vec::new(),
                    topology: self.active_topology.clone(),
                    return_type: ret_ty.clone(),
                    requires: vec![],
                    ensures: vec![],
                    where_transfers: vec![],
                    body: Self::closure_call_fn_body(&struct_name, &captured_vars, *b, e.span),
                    doc_comment: None,
                };

                self.mono.functions.push((call_func, 0));

                *expr = Self::closure_env_init(&struct_name, &captured_vars, e.span);

                // Record the closure's call signature so a later `unify_types` can recover it:
                // `Struct("Closure_N")` erases the args/ret, but matching it against a
                // `ClosureK<Args.., Ret>` parameter (e.g. `.map`'s `Closure1<T, NewItem>`) needs
                // them to bind the method's generics. Keyed by the generated struct name.
                let param_tys: Vec<Type> = cloned_params.iter().map(|(_, ty)| ty.clone()).collect();
                self.mono
                    .closure_signatures
                    .insert(struct_name.clone().into(), (param_tys, ret_ty));

                Type::Struct(struct_name.into(), None)
            }
            _ => panic!("Expected IndexAccess, got {expr:?}"),
        }
    }
}
