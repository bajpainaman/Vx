//===- expr.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Semantic analysis for expressions. Enforces type checking, const generics, and memory topology rules.
//
//===----------------------------------------------------------------------===//

use super::{
    Dim, ElementType, EnumVariantExpr, Expr, ExprStmtStmt, MemorySpace, MemorySpaceExpr,
    NumberExpr, Span, Statement, StringLiteralExpr, Topology, TopologyExpr, Type, TypeChecker,
    UnaryOp, UnaryOpExpr,
};

/// Build an `i32` numeric dimension expression (used to synthesize a tensor shape from an
/// initializer list's nesting).
pub(crate) fn int_dim_expr(n: usize) -> Expr {
    Expr::Number(NumberExpr {
        value: n.to_string().into(),
        ty: Some(ElementType::I32),
        span: Span::default(),
    })
}

/// The scalar element an untyped numeric literal adopts from its expected type, when compatible:
/// an integer literal takes an integer expected type, a float literal a float expected type. A
/// `bool`, generic, or non-scalar expected type — or a literal/expected-kind mismatch such as
/// `let x: f64 = 5` — yields `None`, leaving the literal to its spelling default (#240).
pub(crate) fn expected_numeric_elem(expected: &Type, value: &str) -> Option<ElementType> {
    let Type::Scalar(el) = expected else {
        return None;
    };
    if matches!(el, ElementType::Bool | ElementType::Generic(_)) {
        return None;
    }
    let is_float_lit = value.contains('.');
    (el.is_float() == is_float_lit).then(|| el.clone())
}

/// The numeric literal an expression *is*, seen through a unary minus.
///
/// `-1` is a negation applied to the literal `1`, so a position that types a literal from its
/// context has to look one level in to find the literal that adopts the type. Without this a
/// negative number is typed by its spelling alone, and every context that accepts `1` refuses
/// `-1`.
pub(crate) fn numeric_literal_mut(e: &mut Expr) -> Option<&mut NumberExpr> {
    match e {
        Expr::Number(n) => Some(n),
        Expr::UnaryOp(UnaryOpExpr {
            op: UnaryOp::Neg,
            expr: inner,
            ..
        }) => numeric_literal_mut(inner),
        _ => None,
    }
}

/// Whether an expression is a numeric literal that has no type of its own yet -- the one that
/// takes its type from the other operand of a binary op, or from the position it sits in.
pub(crate) fn is_untyped_numeric_literal(e: &Expr) -> bool {
    match e {
        Expr::Number(n) => n.ty.is_none(),
        Expr::UnaryOp(UnaryOpExpr {
            op: UnaryOp::Neg,
            expr: inner,
            ..
        }) => is_untyped_numeric_literal(inner),
        _ => false,
    }
}

/// Forget the type an untyped numeric literal was given when it was checked, so it can be
/// checked again against another expected type.
pub(crate) fn clear_literal_type(e: &mut Expr) {
    match e {
        Expr::Number(n) => n.ty = None,
        Expr::UnaryOp(UnaryOpExpr {
            op: UnaryOp::Neg,
            expr: inner,
            ..
        }) => clear_literal_type(inner),
        _ => {}
    }
}

impl TypeChecker<'_> {
    /// Checks `expr` and returns its type. This is a real check, not a trial one, so its errors
    /// are reported even when a trial check is running further up the call stack.
    pub fn check_expr_type(&mut self, expr: &mut Expr) -> Type {
        // A "fresh", non-speculative check. Force `speculating` off for the duration so a probe
        // higher on the stack (the methodcall return-type probe in `check_methodcall_expr`) can't
        // leak into this independent subtree — reproducing the old hard-coded `silent = false`
        // argument this call used to pass (#279 R3).
        let saved = self.speculating;
        self.speculating = false;
        let ty = self.check_expr_type_flag(expr, true);
        self.speculating = saved;
        ty
    }

    /// As [`Self::check_expr_type`], with `expected` as the type an untyped numeric literal in
    /// `expr` takes.
    pub(crate) fn check_expr_type_expecting(&mut self, expr: &mut Expr, expected: Type) -> Type {
        let saved = self.speculating;
        self.speculating = false;
        let ty = self.check_expr_expecting(expr, Some(expected), true);
        self.speculating = saved;
        ty
    }

    /// Checks a statement of a block that is one expression. `value_unused` says whether anything
    /// reads the value of the expression. Returns the type of the expression when it has no `;`,
    /// which makes it the value of the block.
    fn check_block_expr_stmt(
        &mut self,
        expr: &mut Expr,
        has_semi: &mut bool,
        value_unused: bool,
        consume: bool,
    ) -> Option<Type> {
        // See the same call in `check_statement`: the values a mutable borrow is
        // about to drop have to be read before the arguments are checked.
        let before = self.consteval_snapshot();
        let scopes = self.consteval_scopes();
        let saved_borrows = self.borrow.snapshot();
        self.value_unused = value_unused;
        let ty = self.check_expr_type_flag(expr, consume);
        let mut block_value = None;
        if !*has_semi {
            // A last expression that never finishes, `todo()`, gives the block the type
            // `!` but no value: it is a statement, after which nothing runs.
            if crate::syntax::is_never_ty(&ty) {
                *has_semi = true;
            }
            block_value = Some(ty);
        }
        self.borrow.restore(saved_borrows);
        self.settle_mut_borrow_call(expr, &before, &scopes);
        block_value
    }

    /// Checks the statements of an `if` or `match` arm, or of an `unsafe`, `comptime` or `spawn`
    /// block. `tail` is the value such a block ends with after its statements, which the caller
    /// checks; it is passed here so a borrow it uses lasts until it.
    pub(crate) fn check_expr_block(
        &mut self,
        stmts: &mut Vec<Statement>,
        tail: Option<&Expr>,
        consume: bool,
    ) -> Type {
        Self::name_loop_literals(stmts);
        self.hoist_spawn_allocations(stmts);
        let block_unused = std::mem::replace(&mut self.value_unused, false);
        let mut ret_ty = Type::Struct("void".into(), None);
        let mut terminated = false;
        let mut terminated_at = None;
        let last = stmts.len().saturating_sub(1);
        // An `if` or `match` arm is a block of its own: what it declares is dropped inside it,
        // and a borrow made in it ends at its last use there.
        self.drops_enter_block(stmts);
        self.borrow.enter_block(Self::block_uses(stmts, tail));

        let mut declared = std::collections::HashMap::new();
        for (i, s) in stmts.iter_mut().enumerate() {
            self.borrow.set_stmt(i);
            self.drops_set_stmt(i);
            if terminated && !self.speculating {
                let stmt_span = s.span();
                self.errors.warn(
                    crate::diagnostic::DiagnosticCode::W1003,
                    "Unreachable code after return, break, or continue",
                    Some(crate::diagnostic::SourceSpan::from_ast_span(&stmt_span)),
                );
                break;
            }
            self.warn_if_declared_twice(s, &mut declared);

            // Only a last expression with no `;` gives the block its value.
            ret_ty = Type::Struct("void".into(), None);
            if let Statement::ExprStmt(ExprStmtStmt {
                ref mut expr,
                has_semi,
                span: _,
            }) = s
            {
                let unused = *has_semi || i != last || block_unused;
                if let Some(ty) = self.check_block_expr_stmt(expr, has_semi, unused, consume) {
                    ret_ty = ty;
                }
                // `check_statement` asks this of every other statement.
                self.check_frame_escape(s);
            } else {
                let expected_ret = self.current_return_type.clone().unwrap_or(Type::Tensor(
                    ElementType::F32,
                    vec![],
                    None,
                ));
                self.check_statement(s, &expected_ret, consume);
            }

            match s {
                Statement::Return(_) | Statement::Break(_) | Statement::Continue(_) => {
                    terminated = true;
                    terminated_at = Some(i);
                }
                _ => {}
            }
        }
        self.drops_exit_block(stmts, terminated_at);
        self.borrow.exit_block();
        ret_ty
    }

    /// Rewrites a call spelled `Enum::Variant(args)` into the enum variant expression it is.
    /// A call to anything else is left alone.
    fn rewrite_enum_variant_call(&self, expr: &mut Expr) {
        let mut is_enum_variant = false;
        if let Expr::FunctionCall(fc) = expr {
            if let Some((enum_name, variant)) = fc.name.split_once("::") {
                let actual_enum_name = if let Some(idx) = enum_name.find('<') {
                    &enum_name[0..idx]
                } else {
                    enum_name
                };
                // Only a name the enum declares is a variant. `Option<i32>::default()` names a
                // trait's static method on the enum, which has to reach the static call path
                // rather than be reported as a variant the enum does not have.
                if let Some(decl) = self.env.enums.get(actual_enum_name) {
                    if decl.variants.iter().any(|(v, _)| v.as_ref() == variant) {
                        is_enum_variant = true;
                    }
                }
            }
        }

        if is_enum_variant {
            if let Expr::FunctionCall(fc) = expr {
                if let Some((enum_name, variant)) = fc.name.split_once("::") {
                    let mut payload = None;
                    if !fc.args.is_empty() {
                        let mut args = Vec::new();
                        std::mem::swap(&mut args, &mut fc.args);
                        payload = Some(args);
                    }
                    *expr = Expr::EnumVariant(EnumVariantExpr {
                        enum_name: enum_name.into(),
                        variant_name: variant.to_string().into(),
                        payload,
                        span: fc.span,
                    });
                }
            }
        }
    }

    /// Whether `expr` is a block or a branch, whose value comes from the last expression inside
    /// it. Such an expression passes on whether its own value is read.
    fn is_block_like(expr: &Expr) -> bool {
        matches!(
            expr,
            Expr::ComptimeBlock(..)
                | Expr::SpawnOn(..)
                | Expr::If(..)
                | Expr::UnsafeBlock(..)
                | Expr::Match(..)
        )
    }

    /// Checks a function call, then folds it when it calls a comptime lambda.
    fn check_functioncall_and_fold(&mut self, expr: &mut Expr) -> Type {
        let ty = self.check_functioncall_expr(expr);
        // A comptime lambda runs while compiling, so its calls fold here. With every
        // call folded the lambda is unused, and the `let` that bound it is dropped.
        let span = expr.span();
        self.fold_comptime_lambda_call(expr, &span);
        ty
    }

    /// Checks a field access. With `consume`, refuses to move the field out of its struct.
    fn check_memberaccess_value(&mut self, expr: &mut Expr, consume: bool) -> Type {
        let ty = self.check_memberaccess_expr(expr);
        if consume {
            self.refuse_field_move(expr, &ty);
        }
        ty
    }

    /// Reports `Memory::X` used where a value is expected.
    ///
    /// A memory space is not a value: `transfer(a, Memory::X)` destructures it at the
    /// call site, and every other use carries it in a type. Answering with a dims-less
    /// tensor let `let t : Tensor<f32, [2, 2]> = Memory::GPU_HBM` type-check, because a
    /// source with no dims skips the comparison against the annotation.
    fn refuse_memory_space_value(&mut self, space: &MemorySpace) -> Type {
        self.errors.push(format!(
            "`Memory::{}` names a memory space, which is not a value; it belongs in a \
             type or as the destination of a `transfer`",
            space.name()
        ));
        Type::Unknown
    }

    /// The type of a topology used as a value. `Topology::Current` is first replaced by the
    /// topology the code runs on.
    fn check_topology_value(&self, top: &mut Topology) -> Type {
        if matches!(top, Topology::Current) {
            *top = self.active_topology.clone();
        }
        // A topology used as a *value* (assigned, pushed into a `Vec<Topology>`,
        // compared) is its runtime dispatch id: an i32 discriminant. Placement
        // uses (`Pinned<T, Topology::X>`, `spawn on`) carry the topology in the
        // type/statement, not through this expression's value type.
        Type::Scalar(ElementType::I32)
    }

    /// Checks the arguments of a `print` or `println`. The result is the `i32` the C function
    /// returns.
    fn check_print_args(&mut self, args: &mut [Expr], consume: bool) -> Type {
        for arg in args {
            let ty = self.check_expr_type_flag(arg, consume);
            Self::prepare_print_operand(arg, &ty);
        }
        Type::Scalar(ElementType::I32)
    }

    /// Checks the inputs and the clobbers of an inline MLIR block. The type is the one the block
    /// declares, or unknown when it declares none.
    fn check_inline_mlir_expr(
        &mut self,
        e: &mut crate::syntax::InlineMlirExpr,
        consume: bool,
    ) -> Type {
        // Typecheck inputs
        for (_, arg_expr, _) in &mut e.inputs {
            self.check_expr_type_flag(arg_expr, consume);
        }
        // Typecheck clobbers. The block writes to each of them, so each one changes.
        for clobber in &mut e.clobbers {
            self.record_change(clobber);
            self.check_expr_type_flag(clobber, consume);
        }

        // Return the specified type or Unknown if void
        e.returns.clone().unwrap_or(Type::Unknown)
    }

    /// Checks `expr` and returns its type. `consume` says the value is moved out of where it
    /// lives, so a value that cannot be copied is marked as moved.
    pub fn check_expr_type_flag(&mut self, expr: &mut Expr, consume: bool) -> Type {
        let value_unused = std::mem::replace(&mut self.value_unused, false);
        self.rewrite_enum_variant_call(expr);
        if Self::is_block_like(expr) {
            self.value_unused = value_unused;
        }

        match expr {
            Expr::Identifier(..) => self.check_identifier_expr(expr, consume),
            Expr::EnumVariant(..) => self.check_enumvariant_expr(expr, consume),
            Expr::Number(n) => self.check_number_literal(n),
            Expr::StringLiteral(StringLiteralExpr { .. }) => Type::Pointer(
                Box::new(Type::Scalar(ElementType::I8)),
                None,
                false, // const
            ),
            Expr::Transfer(..) => self.check_transfer_expr(expr, consume),
            // `Reachable<A, B>` is a comptime boolean.
            Expr::TransferPredicate(..) => Type::Scalar(ElementType::Bool),
            Expr::ComptimeBlock(..) => self.check_comptimeblock_expr(expr, consume),
            Expr::SpawnOn(..) => self.check_spawnon_expr(expr, consume),
            Expr::If(..) => self.check_if_expr(expr, consume),
            Expr::SizeOf(..) => Type::Scalar(ElementType::I64),
            Expr::FunctionCall(..) => self.check_functioncall_and_fold(expr),
            Expr::IndirectCall(..) => self.check_indirectcall_expr(expr, consume),
            Expr::Array(..) => self.check_array_expr(expr),
            Expr::MemberAccess(..) => self.check_memberaccess_value(expr, consume),
            Expr::IndexAccess(..) => self.check_indexaccess_expr(expr, consume),
            Expr::MethodCall(..) => self.check_methodcall_expr(expr, consume),
            Expr::BinaryOp(..) => self.check_binaryop_expr(expr, consume),
            Expr::RelationalOp(..) => self.check_relationalop_expr(expr),
            Expr::LogicalOp(..) => self.check_logicalop_expr(expr),
            Expr::MemorySpace(MemorySpaceExpr { space, .. }) => {
                self.refuse_memory_space_value(space)
            }
            Expr::Topology(TopologyExpr { top, span: _ }) => self.check_topology_value(top),
            Expr::UnaryOp(..) => self.check_unaryop_expr(expr),
            Expr::Borrow(..) => self.check_borrow_expr(expr),
            Expr::Dereference(..) => self.check_dereference_expr(expr, consume),
            Expr::UnsafeBlock(..) => self.check_unsafeblock_expr(expr, consume),
            Expr::StructInit(..) => self.check_structinit_expr(expr, consume),
            Expr::Grad(..) => self.check_grad_expr(expr),
            Expr::Vjp(..) => self.check_vjp_expr(expr),
            Expr::Jvp(..) => self.check_jvp_expr(expr),
            Expr::Range(..) => self.check_range_expr(expr),
            Expr::Match(..) => self.check_match_expr(expr, consume),
            Expr::VecMacro(..) => self.check_vecmacro_expr(expr),
            Expr::Closure(..) => self.check_closure_expr(expr, consume),
            Expr::AsCast(e) => self.check_ascast_expr(e, consume),
            // `print` returns 0 as an `i32`, for C compatibility.
            Expr::Print(p) => self.check_print_args(&mut p.args, consume),
            Expr::Println(p) => self.check_print_args(&mut p.args, consume),
            Expr::InlineMlir(e) => self.check_inline_mlir_expr(e, consume),
            Expr::MacroCall(_) => {
                self.errors.push("Macro failed to expand".to_string());
                crate::syntax::Type::Scalar(crate::syntax::types::ElementType::I32)
                // Fallback type
            }
        }
    }

    /// Lowers an AST `Type` to a globally resolved `TypeId` structure.
    /// This integrates the AST semantic boundary with the hardware-level
    /// 256-bit `FastPath` borrow checking rules.
    pub fn lower_to_type_id(&self, ty: &Type) -> crate::gid::TypeId {
        // We use a dummy symbol_hash for local types, as we are only concerned
        // with the Lifetime Signature (Word 2) for borrow checking right now.
        let mut id = crate::gid::TypeId::new(0, 0, 0, 0);

        match ty {
            Type::Borrow {
                region_id: region, ..
            } => {
                // The lifetime of the borrow itself is Covariant (even for mutable borrows,
                // which allows reborrowing for shorter lifetimes during function calls).
                // (The inner type T would be invariant for mutable borrows, but we are
                // only hashing the outer lifetime here).
                let variance: u8 = 0x1;

                // Pack the region and variance directly into Param 0 (the return slot) of the
                // FastPath hash! Slot 0 reserves its top 3 region bits for the return-provenance code
                // (#265), so its region is 9 bits: `region_for_depth_slot0` preserves the "unset"
                // sentinel, clamps a real depth below it, and keeps the value inside the 9-bit field
                // so it can neither collide with the sentinel (#267) nor spill into the code bits.
                #[allow(
                    clippy::cast_possible_truncation,
                    reason = "`region_for_depth_slot0` keeps the value inside the 9-bit region field"
                )]
                let region = crate::borrow::region_for_depth_slot0(*region as u64) as u16;
                let _ = id.try_set_fast_param(0, region, variance);
            }
            Type::Pointer(_inner, _mem, is_mut) => {
                let variance: u8 = u8::from(!*is_mut);
                // Pointers don't have safe lifetimes, so we assign 'static (0)
                // which represents the unconstrained lifetime.
                let _ = id.try_set_fast_param(0, 0, variance);
            }
            // For other types, we just return the raw un-initialized hash
            _ => {}
        }
        id
    }

    /// A located tensor split into what it holds and where it is, for either
    /// spelling. `None` for a type that says nothing about where it lives.
    fn located_parts(t: &Type) -> Option<(Type, Topology)> {
        match t {
            Type::Pinned(inner, top) => Some(((**inner).clone(), top.clone())),
            Type::Tensor(e, d, Some(p)) => {
                Some((Type::Tensor(e.clone(), d.clone(), None), p.topology.clone()))
            }
            _ => None,
        }
    }

    /// Compares two struct or enum types. They are assignable when the names are equal and, if
    /// both carry an id, the ids are equal too. `None` when the names differ or either type is
    /// not a struct or an enum.
    fn named_types_assignable(target: &Type, source: &Type) -> Option<bool> {
        if let Type::Struct(n_target, id_target) = target {
            if let Type::Struct(n_source, id_source) = source {
                if n_target == n_source {
                    if id_target.is_some() && id_source.is_some() {
                        return Some(id_target == id_source);
                    }
                    return Some(true);
                }
            }
        }

        if let Type::Enum(n_target, id_target) = target {
            if let Type::Enum(n_source, id_source) = source {
                if n_target == n_source {
                    if id_target.is_some() && id_source.is_some() {
                        return Some(id_target == id_source);
                    }
                    return Some(true);
                }
            }
        }

        if let Type::Struct(n_target, id_target) = target {
            if let Type::Enum(n_source, id_source) = source {
                if n_target == n_source {
                    if id_target.is_some() && id_source.is_some() {
                        return Some(id_target == id_source);
                    }
                    return Some(true);
                }
            }
        }

        if let Type::Enum(n_target, id_target) = target {
            if let Type::Struct(n_source, id_source) = source {
                if n_target == n_source {
                    if id_target.is_some() && id_source.is_some() {
                        return Some(id_target == id_source);
                    }
                    return Some(true);
                }
            }
        }
        None
    }

    /// Whether two instances of a generic type are assignable: the generic type and every type
    /// argument must be. False when either type is not a generic instance.
    fn generic_instances_assignable(&self, target: &Type, source: &Type) -> bool {
        if let Type::GenericInstance(inner_target, args_target) = target {
            if let Type::GenericInstance(inner_source, args_source) = source {
                if self.is_assignable(inner_target, inner_source)
                    && args_target.len() == args_source.len()
                {
                    let mut all_match = true;
                    for (at, asrc) in args_target.iter().zip(args_source.iter()) {
                        if !self.is_assignable(at, asrc) {
                            all_match = false;
                            break;
                        }
                    }
                    if all_match {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Compares two tensor types: the element, the place and every extent. `None` when either
    /// type is not a tensor.
    fn tensors_assignable(&self, target: &Type, source: &Type) -> Option<bool> {
        // Allow numeric coercions for scalar literals (mock behavior for now)
        if let Type::Tensor(t_target, dims_target, top_target) = target {
            if let Type::Tensor(t_source, dims_source, top_source) = &source {
                let mut el_match = false;
                if *t_target == *t_source {
                    el_match = true;
                }

                if !el_match {
                    return Some(false);
                }

                if top_target.is_some() && top_target != top_source {
                    return Some(false);
                }

                // An annotation that names no place asks for host data, so a value
                // sitting on a device does not satisfy it -- passing one to
                // `fn f(t : Tensor<f32, [4]>)` is exactly the mistake `transfer`
                // exists to make visible. A value placed in CPU_DRAM does satisfy
                // it: that is where an unplaced tensor lives.
                if top_target.is_none() {
                    if let Some(p) = top_source {
                        if p.space != MemorySpace::CPUDRAM {
                            return Some(false);
                        }
                    }
                }

                // Rank is static and no cast changes it. Per dimension, a `?` in the target
                // accepts any extent and a static extent accepts only itself: `[512, 8]` widens
                // to `[?, 8]`, and `[?, 8]` does not narrow to `[512, 8]` without saying so.
                if dims_target.len() != dims_source.len() {
                    return Some(false);
                }
                let empty_env = std::collections::HashMap::new();
                for (dt, ds) in dims_target.iter().zip(dims_source.iter()) {
                    let Some(et) = dt.as_static() else { continue };
                    let Some(es) = ds.as_static() else {
                        return Some(false);
                    };
                    let vt = self.eval_expr(et, &empty_env);
                    let vs = self.eval_expr(es, &empty_env);
                    if vt.is_some() && vs.is_some() {
                        if vt != vs {
                            return Some(false);
                        }
                    } else if et != es {
                        return Some(false);
                    }
                }
                return Some(true);
            }
        }
        None
    }

    /// Compares two function types, or two closure types: the parameters and the return type
    /// must be assignable, and an unsafe function never fits a safe one. `None` when the check
    /// does not settle the answer.
    fn callables_assignable(&self, target: &Type, source: &Type) -> Option<bool> {
        if let Type::Function(p_target, r_target, target_unsafe) = target {
            if let Type::Function(p_source, r_source, source_unsafe) = source {
                // A safe function honours an unsafe function's contract vacuously, so it fits
                // where one is expected. An unsafe function does not fit where a safe one is:
                // that is how the obligation would be lost.
                if *source_unsafe && !*target_unsafe {
                    return Some(false);
                }
                if p_target.len() == p_source.len() && self.is_assignable(r_target, r_source) {
                    let mut all_match = true;
                    for (pt, ps) in p_target.iter().zip(p_source.iter()) {
                        if !self.is_assignable(pt, ps) {
                            all_match = false;
                        }
                    }
                    if all_match {
                        return Some(true);
                    }
                }
            }
        }

        if let Type::Closure(p_target, r_target) = target {
            if let Type::Closure(p_source, r_source) = source {
                if p_target.len() == p_source.len() && self.is_assignable(r_target, r_source) {
                    let mut all_match = true;
                    for (pt, ps) in p_target.iter().zip(p_source.iter()) {
                        if !self.is_assignable(pt, ps) {
                            all_match = false;
                        }
                    }
                    if all_match {
                        return Some(true);
                    }
                }
            }
        }
        None
    }

    /// Compares the struct of a closure literal against a `ClosureK<Args.., Ret>` target. `None`
    /// when the two types are not that pair, or the literal has no recorded signature.
    fn closure_literal_assignable(&self, target: &Type, source: &Type) -> Option<bool> {
        // A closure literal checks to `Closure_N`, a struct with no call signature in its
        // type. The signature is recorded when the literal is checked, so compare that
        // against `ClosureK<Args.., Ret>`, as inference already does. Without this a
        // literal was accepted as a call argument but refused as a struct field's value.
        if let (Type::GenericInstance(inner, args), Type::Struct(cn, _)) = (target, source) {
            if cn.starts_with("Closure_")
                && matches!(&**inner, Type::Struct(n, _) if n.starts_with("Closure"))
            {
                if let Some((params, ret)) = self.mono.closure_signatures.get(cn) {
                    return Some(
                        args.len() == params.len() + 1
                            && args
                                .iter()
                                .zip(params.iter())
                                .all(|(a, p)| self.is_assignable(a, p))
                            && self.is_assignable(&args[args.len() - 1], ret),
                    );
                }
            }
        }
        None
    }

    /// Whether a closure type is assignable to a `ClosureN<Args.., Ret>` struct target.
    fn closure_type_assignable(&self, target: &Type, source: &Type) -> bool {
        // Allow Closure to map to ClosureN struct (if tests use it)
        if let Type::GenericInstance(inner, args) = target {
            if let Type::Struct(name, _) = &**inner {
                if name.starts_with("Closure") {
                    if let Type::Closure(p_source, r_source) = source {
                        if args.len() == p_source.len() + 1 {
                            let mut all_match = true;
                            for (i, ps) in p_source.iter().enumerate() {
                                if !self.is_assignable(&args[i], ps) {
                                    all_match = false;
                                }
                            }
                            if !self.is_assignable(&args[args.len() - 1], r_source) {
                                all_match = false;
                            }
                            if all_match {
                                return true;
                            }
                        }
                    }
                }
            }
        }
        false
    }

    /// Compares a scalar source against a scalar or a tensor target. `None` when the source is
    /// not a scalar, or the target is neither.
    fn scalar_source_assignable(target: &Type, source: &Type) -> Option<bool> {
        if let Type::Scalar(t_target) = target {
            if let Type::Scalar(t_source) = &source {
                // Scalars are assignable only when identical (#240): Vx has no implicit numeric
                // conversion. An untyped literal has already adopted its context's type in the
                // checking positions (`let`/`return`/assignment/operands/call args), so what reaches
                // here mismatched is a genuine typed-value conversion — the programmer writes `as`.
                return Some(*t_target == *t_source);
            }
        }

        // A scalar is assignable to a RANK-0 tensor of the identical element, which wraps it.
        // A shaped tensor is not: that spelling was a splat, and it allocated and filled a whole
        // buffer from something that reads as an assignment. The two codegen paths did not even
        // agree on it -- one emitted the allocation and the fill, the other kept the bare
        // constant. `Tensor<T, [..]>::fill(v)` says it instead. The any-to-any arm went with the
        // rest of implicit numeric conversion (#240, Vx#396).
        if let Type::Tensor(t_target, dims_target, _) = target {
            if let Type::Scalar(t_source) = &source {
                return Some(dims_target.is_empty() && *t_target == *t_source);
            }
        }
        None
    }

    /// Whether a `Verified<U>` is assignable to a `Verified<T>`.
    fn verified_assignable(&self, target: &Type, source: &Type) -> bool {
        // Semantic coercion rule: Verified<T> can only be assigned from another Verified<U> where is_assignable(T, U)
        if let Type::Verified(inner_target) = target {
            if let Type::Verified(inner_source) = source {
                if self.is_assignable(inner_target, inner_source) {
                    return true;
                }
            }
        }
        false
    }

    /// Whether a borrow is assignable to a pointer or to another borrow.
    fn borrow_source_assignable(&self, target: &Type, source: &Type) -> bool {
        // Allow coercing Borrow to Pointer (e.g. &mut T to *mut T)
        if let Type::Pointer(target_inner, target_mem, target_mut) = target {
            if let Type::Borrow {
                inner: source_inner,
                mem_space: source_mem,
                is_mut: source_mut,
                ..
            } = source
            {
                if target_mem.as_ref().map(|p| &p.space) == source_mem.as_ref()
                    && (!*target_mut || *source_mut)
                    && self.is_assignable(target_inner, source_inner)
                {
                    return true;
                }
            }
        }

        if let Type::Borrow {
            inner: target_inner,
            mem_space: target_mem,
            is_mut: target_mut,
            ..
        } = target
        {
            if let Type::Borrow {
                inner: source_inner,
                mem_space: source_mem,
                is_mut: source_mut,
                ..
            } = source
            {
                if target_mem == source_mem
                    && (!*target_mut || *source_mut)
                    && self.is_assignable(target_inner, source_inner)
                {
                    // Hook up 256-bit FastPath Borrow Checker algorithm from src/borrow.rs
                    let id_target = self.lower_to_type_id(target);
                    let id_source = self.lower_to_type_id(source);
                    if crate::borrow::verify_subtyping_bounds(&id_source, &id_target, self.worker) {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Whether a pointer is assignable to another pointer.
    fn pointers_assignable(&self, target: &Type, source: &Type) -> bool {
        if let Type::Pointer(target_inner, target_mem, target_mut) = target {
            if let Type::Pointer(source_inner, source_mem, source_mut) = source {
                if target_mem == source_mem && (!*target_mut || *source_mut) {
                    if let Type::Scalar(ElementType::I8) = &**source_inner {
                        return true; // allow casting *mut i8 (void*) to any pointer
                    }
                    if let Type::Scalar(ElementType::I8) = &**target_inner {
                        return true; // allow casting any pointer to *mut i8 (void*)
                    }
                    if self.is_assignable(target_inner, source_inner) {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Whether a value of type `source` may be used where a `target` is expected.
    pub(crate) fn is_assignable(&self, target: &Type, source: &Type) -> bool {
        // A value of type `!` never exists, since the expression never finishes, so it fits
        // wherever a value is expected.
        if target == source || crate::syntax::is_never_ty(source) {
            return true;
        }

        // `Pinned<T, D>` and a `T` carrying a placement on `D` are two spellings of
        // one fact: this value is on that device. Both are real -- a signature can
        // write `Pinned<Tensor<bf16, [4, 4]>, Topology::NPU[0]>`, and the checker
        // wraps a topology-bound return the same way, while `transfer` and a
        // `Memory::` annotation produce the placed tensor. Comparing them by
        // constructor made a transferred tensor unassignable to a parameter that
        // named the same device.
        if let (Some((t_inner, t_top)), Some((s_inner, s_top))) =
            (Self::located_parts(target), Self::located_parts(source))
        {
            return t_top == s_top && self.is_assignable(&t_inner, &s_inner);
        }

        if let Some(answer) = Self::named_types_assignable(target, source) {
            return answer;
        }
        if self.generic_instances_assignable(target, source) {
            return true;
        }

        // Explicit Memory transfer enforcement:
        // We no longer allow implicit unwrapping of Ref<T> or Pinned<T> to T.
        // Users must use `transfer(expr, Memory::Space)` or `.to_host()` / `.to_device()`
        // to move data across memory boundaries.

        if let Some(answer) = self.tensors_assignable(target, source) {
            return answer;
        }
        if let Some(answer) = self.callables_assignable(target, source) {
            return answer;
        }
        if let Some(answer) = self.closure_literal_assignable(target, source) {
            return answer;
        }
        if self.closure_type_assignable(target, source) {
            return true;
        }
        if let Some(answer) = Self::scalar_source_assignable(target, source) {
            return answer;
        }
        if self.verified_assignable(target, source) {
            return true;
        }

        // Note: Verified<T> should NOT implicitly coerce to T if the user strictly expected T in tests,
        // or perhaps we shouldn't strip it here. Let's revert this coercion so type_mismatch fails again.

        self.borrow_source_assignable(target, source) || self.pointers_assignable(target, source)
    }

    /// The `(element, dims)` of the tensor at the core of a (possibly wrapped) type, if any.
    pub(crate) fn tensor_of(ty: &Type) -> Option<(&ElementType, &[Dim])> {
        match ty {
            Type::Tensor(e, d, _) => Some((e, d.as_slice())),
            Type::Ref(inner, _) | Type::Pinned(inner, _) | Type::Verified(inner) => {
                Self::tensor_of(inner)
            }
            _ => None,
        }
    }
}
