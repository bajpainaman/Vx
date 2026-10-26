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
    decl, ArrayExpr, BorrowExpr, BorrowRecord, Dim, ElementType, EnumVariantExpr, Expr, Function,
    FunctionCallExpr, IdentifierExpr, IndexAccessExpr, Mangle, MemberAccessExpr, MemorySpace,
    MethodCallExpr, NumberExpr, Placement, Span, Statement, Topology, TransferExpr, Type,
    TypeChecker, Value,
};
use crate::hir::expr::{expected_numeric_elem, int_dim_expr};
use crate::symbol::Symbol;
use std::collections::HashMap;

/// Everything `check_functioncall_expr` computes *before* the argument loop so it can decide,
/// *after* it, which reference-argument reborrows persist past the call (#243). Built by
/// `prepare_reference_arg_reborrows`, consumed by `commit_reference_arg_reborrows`.
struct ReborrowPlan {
    /// Callee `(param types, return type)` if resolvable and not speculating; `None` disables all
    /// reborrow bookkeeping.
    callee_sig: Option<(Vec<Type>, Type)>,
    /// The callee returns a reference, or a value that can hold one (only then can an argument
    /// borrow outlive the call).
    ret_is_ref: bool,
    /// The result is a value holding a reference, `View { r : &x }`, rather than a reference: a
    /// kept borrow then takes the mutability of the parameter it came through.
    ret_holds_ref: bool,
    /// The returned reference is mutable (the reborrow's mutability is the result's, not the param's).
    result_is_mut: bool,
    /// Parameter slots the returned reference derives from (in-compilation summary).
    return_prov: crate::hir::provenance::ReturnProvenance,
    /// Same, for an imported callee read from the registry inline code (`None` = use `return_prov`).
    imported_prov: Option<u8>,
    /// Each reference argument of the call.
    ref_args: Vec<RefArg>,
    /// Pre-call borrow list of each distinct reference-argument base, for the selective revert.
    base_snapshots: BaseSnapshots,
    /// By-value arguments that can hold a reference, `s` after `let s = View { r : &x }`.
    held_args: Vec<String>,
}

/// One reference argument of a call: its position, the variable it borrows, the field path into
/// that variable, whether the parameter is `&mut`, and whether the argument is written as `&x`.
type RefArg = (usize, String, Vec<String>, bool, bool);

/// The borrows each variable had before a call's arguments were checked, by variable name.
type BaseSnapshots = HashMap<String, Option<Vec<BorrowRecord>>>;

/// A place a borrow reaches: a variable and a field path into it.
type Place = (String, Vec<String>);

/// What checking a call gives back: the call's type and, when the check rewrites the call, the
/// expression to put in its place.
type Rewrite = (Type, Option<Expr>);

/// A trait's method that a method call could mean: the trait, the method, its impl block, and
/// the types the impl block's parameters take for this receiver.
type TraitMethod = (Symbol, Function, decl::ImplBlock, HashMap<Symbol, Type>);

/// The arguments of one call.
pub(crate) struct CallArgs<'a> {
    /// The argument expressions, in the order written.
    pub(crate) exprs: &'a mut [Expr],
    /// The type each argument was checked to have.
    pub(crate) types: &'a [Type],
    /// Where the call is in the source.
    pub(crate) span: &'a Span,
}

/// The generic function a call names.
pub(crate) struct GenericCallee<'a> {
    /// The generic function's declaration.
    pub(crate) func: &'a Function,
    /// The hash of the module that declares it.
    pub(crate) origin_hash: u64,
    /// The type arguments the call wrote out.
    pub(crate) explicit_args: &'a [Type],
}

/// The names a call to a function goes by.
struct CalleeName<'a> {
    /// The name as the call wrote it, type arguments included.
    resolved: &'a Symbol,
    /// The name without type arguments.
    base: &'a Symbol,
    /// The type arguments the call wrote out.
    explicit_generic_args: &'a [Type],
}

/// What a static call, `Type::method(..)` or `Trait::method(..)`, names.
struct StaticTarget<'t> {
    /// The type whose impl blocks are searched.
    struct_name: &'t str,
    /// The method's name, without type arguments.
    method_name: &'t str,
    /// The type arguments written at the call, as text. Empty when there are none.
    explicit_types: &'t str,
    /// The trait, when the call is written `Trait::method(..)`.
    only_trait: Option<&'t str>,
}

/// The impl blocks that have the method a static call names.
struct StaticCandidates<'a> {
    /// Blocks that implement no trait.
    inherent: Vec<&'a decl::ImplBlock>,
    /// Blocks that implement a trait, each with the trait's name.
    from_traits: Vec<(&'a Symbol, &'a decl::ImplBlock)>,
}

/// The method a static call resolved to, before an instance of it is made.
struct FoundStaticMethod {
    /// The method's declaration.
    func: Function,
    /// The type parameters of the impl block and of the method, each name once.
    generics: Vec<decl::GenericParam>,
    /// The type parameters the call's written type arguments bind.
    mapping: HashMap<Symbol, Type>,
    /// The name the instance is made under: the type, the trait when the method comes from a
    /// trait's impl block, and the method.
    instance_name: String,
}

/// A method call `obj.method(args)` after its receiver and arguments were checked.
struct MethodCallSite<'a> {
    /// The receiver, `obj`.
    obj: &'a Expr,
    /// The method's name.
    method: &'a Symbol,
    /// The arguments, without the receiver.
    args: &'a mut [Expr],
    /// The receiver's type.
    base_ty: &'a Type,
    /// The type each argument was checked to have.
    arg_types: &'a [Type],
    /// Whether this check moves the values the call takes by value.
    consume: bool,
    /// Where the call is in the source.
    span: Span,
}

/// What was recorded about a method call's borrows before its arguments were checked.
struct MethodBorrows {
    /// The borrows of each variable the receiver or an argument reaches.
    snapshots: BaseSnapshots,
    /// How many errors there were, so an error the arguments reported can be told apart.
    errors_before_args: usize,
    /// The variable that changes if the method changes its receiver.
    receiver_root: Option<String>,
}

/// Could this method take `arg_types` as written?
///
/// Used to narrow several traits' same-named methods down to the one the call means. Exact types
/// only: Vx has no implicit scalar conversion (#240), so a `u8` argument does not reach an `i32`
/// parameter and the two impls really are distinguishable. A generic parameter matches anything --
/// it has not been substituted yet, and excluding it would drop a candidate that does apply.
fn params_accept(m: &Function, arg_types: &[Type]) -> bool {
    if m.params.len() != arg_types.len() {
        return false;
    }
    m.params
        .iter()
        .zip(arg_types)
        .all(|((_, p), a)| matches!(p, Type::Generic(..)) || p == a)
}

/// How an impl block is named in a diagnostic: the trait with its arguments, so two impls of
/// one trait for one type (`From<i32>`, `From<u8>`) are told apart.
fn impl_display(trait_name: &crate::symbol::Symbol, ib: &decl::ImplBlock) -> String {
    if ib.trait_args.is_empty() {
        return trait_name.to_string();
    }
    let args: Vec<String> = ib
        .trait_args
        .iter()
        .map(std::string::ToString::to_string)
        .collect();
    format!("{}<{}>", trait_name, args.join(", "))
}

impl TypeChecker<'_> {
    /// Checks a call whose callee is an expression, such as `(pick())(1, 2)`, and gives the type
    /// of its result. A call of a closure value is rewritten to a call of the function that
    /// holds the closure's body.
    ///
    /// # Panics
    ///
    /// Panics if `expr` is not an indirect call.
    pub(crate) fn check_indirectcall_expr(&mut self, expr: &mut Expr, consume: bool) -> Type {
        let Expr::IndirectCall(c) = expr else {
            panic!("Expected IndexAccess, got {expr:?}")
        };
        let callee_ty = self.check_expr_type_flag(&mut c.callee, consume);
        // As for a call by name: the arguments are moved whatever is done with the result.
        let mut arg_types = Vec::new();
        for arg in &mut c.args {
            arg_types.push(self.check_expr_type_flag(arg, true));
        }

        if let Type::Struct(struct_name, _) = &callee_ty {
            if let Some((closure_ret, call)) = self.call_closure_struct(c, struct_name, &arg_types)
            {
                *expr = call;
                return closure_ret;
            }
        } else if let Type::Closure(param_types, ret_ty) = &callee_ty {
            self.check_closure_pointer_args(param_types, &arg_types, c.args.len());
            c.target_func_ty = Some(callee_ty.clone());

            return *ret_ty.clone();
        } else if let Type::Function(..) = callee_ty {
            if !self.speculating {
                self.errors.push(
                    "Function pointers are not natively callable yet; use closure interfaces."
                        .to_string(),
                );
            }
        } else if !self.speculating {
            self.errors
                .push(format!("Cannot call expression of type {callee_ty}"));
        }

        Type::Unknown
    }

    /// Checks a call of a closure value, whose type is the struct `struct_name` that holds what
    /// the closure captured. Gives the result type and the call of the closure's function to put
    /// in place of the indirect call, or `None` after reporting why the value cannot be called.
    fn call_closure_struct(
        &mut self,
        c: &crate::syntax::expr::IndirectCallExpr,
        struct_name: &str,
        arg_types: &[Type],
    ) -> Option<(Type, Expr)> {
        if struct_name.starts_with("Closure_") {
            let call_name = format!("{struct_name}_call");
            if let Some((param_types, closure_ret)) = self
                .mono
                .functions
                .iter()
                .find(|f| {
                    f.0.name == <std::string::String as Clone>::clone(&call_name.clone()).into()
                })
                .map(|func| {
                    (
                        func.0
                            .params
                            .iter()
                            .skip(1)
                            .map(|(_, t)| t.clone())
                            .collect::<Vec<Type>>(),
                        func.0.return_type.clone(),
                    )
                })
            {
                self.record_call_edge(&call_name, &c.span);
                self.check_closure_struct_args(&param_types, arg_types, c.args.len());

                let mut new_args = vec![Expr::Borrow(BorrowExpr {
                    expr: c.callee.clone(),
                    is_mut: true,
                    span: Span::default(),
                })];
                new_args.extend(c.args.clone());

                let call = Expr::FunctionCall(FunctionCallExpr {
                    name: call_name.into(),
                    type_args: None,
                    args: new_args,
                    span: Span::default(),
                });

                return Some((closure_ret, call));
            } else if !self.speculating {
                self.errors.push(format!(
                    "Missing call method for closure struct '{struct_name}'"
                ));
            }
        } else if !self.speculating {
            self.errors
                .push(format!("Cannot call non-closure struct '{struct_name}'"));
        }
        None
    }

    /// Checks the arguments of a call of a closure value against the closure's parameters:
    /// the count first, then each type.
    fn check_closure_struct_args(
        &mut self,
        param_types: &[Type],
        arg_types: &[Type],
        arg_count: usize,
    ) {
        if arg_count == param_types.len() {
            for (i, param_ty) in param_types.iter().enumerate() {
                let arg_ty = &arg_types[i];
                if !self.is_assignable(param_ty, arg_ty) && !self.speculating {
                    self.errors.push(format!(
                        "Type mismatch in argument {} for closure. Expected {}, got {}",
                        i + 1,
                        param_ty,
                        arg_ty
                    ));
                }
            }
        } else if !self.speculating {
            self.errors.push(format!(
                "Closure expects {} arguments, got {}",
                param_types.len(),
                arg_count
            ));
        }
    }

    /// Checks the arguments of a call through a closure pointer against the parameters in the
    /// pointer's type: the count first, then each type.
    fn check_closure_pointer_args(
        &mut self,
        param_types: &[Type],
        arg_types: &[Type],
        arg_count: usize,
    ) {
        if arg_count == param_types.len() {
            for (i, param_ty) in param_types.iter().enumerate() {
                let arg_ty = &arg_types[i];
                if !self.is_assignable(param_ty, arg_ty) && !self.speculating {
                    self.errors.push(format!(
                        "Type mismatch in argument {} for closure fat pointer. Expected {}, got {}",
                        i + 1,
                        param_ty,
                        arg_ty
                    ));
                }
            }
        } else if !self.speculating {
            self.errors.push(format!(
                "Closure fat pointer expects {} arguments, got {}",
                param_types.len(),
                arg_count
            ));
        }
    }

    /// Type-check `e` under an explicit expected type, restoring the previous one after. The
    /// mechanism by which an untyped numeric literal in a *checking position* adopts that
    /// position's type (#240): only [`Self::check_number_literal`] consults `expected_type`, so
    /// this affects a bare literal (and literals inside a nested checking position) and nothing
    /// else. The expected type does not leak past `e`.
    pub(crate) fn check_expr_expecting(
        &mut self,
        e: &mut Expr,
        expected: Option<Type>,
        consume: bool,
    ) -> Type {
        let prev = self.expected_type.take();
        self.expected_type = expected;
        let ty = self.check_expr_type_flag(e, consume);
        self.expected_type = prev;
        ty
    }

    /// Refine a call argument's type against its parameter, letting an untyped numeric literal
    /// adopt the parameter's scalar type — the call-argument checking position (#240). Arguments
    /// are type-checked eagerly (before the callee is resolved), so a literal has already taken its
    /// spelling default by the time the parameter type is known; this re-types it in place to the
    /// parameter (when kind-compatible: an integer literal to an integer parameter, a float literal
    /// to a float parameter). A non-literal argument is returned unchanged, so a genuine typed-value
    /// mismatch (`f(some_i64)` into an `i32` parameter) is left for the caller's `is_assignable`
    /// check to reject — the programmer writes an explicit `as`. Returns the argument's type after
    /// refinement.
    pub(crate) fn refine_literal_arg(
        &mut self,
        arg: &mut Expr,
        param_ty: &Type,
        arg_ty: &Type,
    ) -> Type {
        if let Some(n) = crate::hir::expr::numeric_literal_mut(arg) {
            if let Some(elem) = expected_numeric_elem(param_ty, &n.value) {
                n.ty = Some(elem.clone());
                let n = n.clone();
                self.check_usize_literal(&n, &elem);
                return Type::Scalar(elem);
            }
        }
        arg_ty.clone()
    }

    /// Compute, *before* the argument loop, everything the post-call reborrow revert needs (#243):
    /// the callee signature, whether it returns a (mutable) reference, the parameter slots the
    /// return derives from (in-compilation summary + imported inline code), and — per reference
    /// argument — its base/path plus a snapshot of that base's pre-call borrow list. Consumed by
    /// `commit_reference_arg_reborrows`. Split out of `check_functioncall_expr` (R4, #279).
    fn prepare_reference_arg_reborrows(
        &mut self,
        resolved_name: &Symbol,
        args: &[Expr],
    ) -> ReborrowPlan {
        // Reborrow tracking (#243). Passing a reference to a reference parameter reborrows
        // the underlying storage; whether that reborrow *persists past the call* is decided
        // per argument by the callee's return-provenance summary — only an argument the
        // returned reference actually derives from stays borrowed once the call returns
        // (`pick(&x, &y)` returning from `b` must keep `y` borrowed but release `x`). We
        // snapshot each reference-argument base *before* the arguments are checked, let the
        // arguments record their borrows (so intra-call conflicts like `f(&mut x, &x)` still
        // fire), then revert the non-deriving bases to their pre-call state.
        let callee_sig = if self.speculating {
            None
        } else {
            self.resolve_callee_ref_signature(resolved_name)
        };
        let return_prov = self.env.return_provenance_of(resolved_name.as_ref());
        let imported_prov = self.imported_return_provenance(resolved_name);
        #[cfg(debug_assertions)]
        Self::debug_check_inline_provenance(return_prov, resolved_name);
        // A reference argument's borrow can only outlive the call if the callee actually
        // returns a reference. Gating on the return *type* (not just the summary) keeps
        // void/value-returning callees correct even when their summary is absent — e.g. an
        // impl method, which `build` does not summarize (it defaults to the conservative
        // `AnyParam`). Without this, `foo(&mut x)` on a void method would wrongly persist a
        // mutable borrow of `x` and fire a spurious `E4004` at the next use.
        let ret_holds_ref = callee_sig
            .as_ref()
            .is_some_and(|(_, ret)| !Self::is_ref_type(ret) && self.result_can_hold_reference(ret));
        let ret_is_ref = ret_holds_ref
            || callee_sig
                .as_ref()
                .is_some_and(|(_, ret)| Self::is_ref_type(ret));
        // A reborrow's mutability is the *result* reference's, not the parameter's: `found =
        // probe_mut(m)` where `probe_mut(m: &mut Map) -> &i32` yields a *shared* alias of
        // `*m`, so the reborrow is shared. Keying on the return type (rather than the param)
        // is both more correct and what keeps a shared-returning generic from marking its
        // argument mutably borrowed — which would collide with the callee body's own reads
        // of a same-named parameter during instantiation (#268).
        let result_is_mut = callee_sig
            .as_ref()
            .is_some_and(|(_, ret)| Self::is_mut_ref(ret));
        let is_extern = self
            .env
            .externs
            .iter()
            .any(|e| e.name.as_ref() == resolved_name.as_ref());
        let mut plan = ReborrowPlan {
            callee_sig,
            ret_is_ref,
            ret_holds_ref,
            result_is_mut,
            return_prov,
            imported_prov,
            ref_args: Vec::new(),
            base_snapshots: HashMap::new(),
            held_args: Vec::new(),
        };
        self.collect_reference_args(&mut plan, is_extern, args);
        plan
    }

    /// The return provenance of a callee that comes from an imported library: which parameters
    /// its returned reference may come from, as the library recorded it. `None` for a function
    /// this compile has its own summary of.
    fn imported_return_provenance(&self, resolved_name: &Symbol) -> Option<u8> {
        // Cross-module refinement (#265 step 7): an *imported* callee has no AST body, so it is
        // absent from the `return_provenances` summary map and `return_provenance_of` falls to
        // the conservative `AnyParam`. Its per-parameter provenance instead travels in the
        // frozen registry's `FnSig.ret_prov` (populated by a `.vxlib` deserialize). Read it
        // only on a genuine summary *miss*, so a local definition's summary always wins; a hit
        // (any in-compilation function, even one that is legitimately `AnyParam`) is untouched.
        if self
            .env
            .return_provenances
            .contains_key(resolved_name.as_ref())
        {
            None
        } else {
            self.worker
                .global
                .registry
                .fn_sigs
                .get(&crate::symbol::Symbol::from(resolved_name.as_ref()))
                .map(|sig| sig.ret_prov)
        }
    }

    /// Debug builds only: checks that the return provenance of the function `resolved_name`
    /// survives being packed into a type id, and that the packed form still names every
    /// parameter the summary names.
    #[cfg(debug_assertions)]
    fn debug_check_inline_provenance(
        return_prov: crate::hir::provenance::ReturnProvenance,
        resolved_name: &str,
    ) {
        // (#265) The same summary lowers into the return type's inline slot-0 provenance code,
        // which the cross-module path (step 7, `.vxlib`) will read straight from the `TypeId`
        // with no side table. Intra-compilation the side table above still drives the persist
        // decision; here we populate and *check* the inline form on every real call — proving
        // it round-trips through the `TypeId` API and stays a conservative refinement of the
        // summary it will replace, so wiring the consumer later can never silently read a
        // narrower (unsound) alias set. Debug-only: the release hot path is untouched.
        let code = crate::hir::provenance::encode_return_provenance(&return_prov);
        let mut sig = crate::gid::TypeId::new(0, 0, 0, 0);
        sig.set_return_provenance(code);
        debug_assert_eq!(
            sig.extract_return_provenance(),
            code,
            "slot-0 provenance code must round-trip through the TypeId encoding"
        );
        debug_assert!(
            (0..32).all(|slot| {
                !return_prov.includes(slot)
                    || crate::hir::provenance::inline_prov_includes(
                        sig.extract_return_provenance(),
                        slot,
                    )
            }),
            "inline provenance must conservatively refine the summary for '{resolved_name}'"
        );
    }

    /// Fills `plan` with the call's reference arguments, the borrows each borrowed variable had
    /// before the call, and the by-value arguments that can hold a reference.
    fn collect_reference_args(&self, plan: &mut ReborrowPlan, is_extern: bool, args: &[Expr]) {
        let Some((param_types, _)) = &plan.callee_sig else {
            return;
        };
        for (i, arg) in args.iter().enumerate() {
            match param_types.get(i) {
                Some(pty) if Self::is_ref_type(pty) => {
                    // `(&mut b) as *mut u8` passes a borrow of `b`, as `&mut b` does.
                    let mut arg = arg;
                    while let Expr::AsCast(cast) = arg {
                        arg = &cast.expr;
                    }
                    // The checker cannot see an `extern` function's body, so it takes the
                    // result to come from every argument. A raw pointer passed by value is
                    // a copy and borrows nothing; only a borrow written at the call counts.
                    if is_extern
                        && matches!(pty, Type::Pointer(..))
                        && !matches!(arg, Expr::Borrow(_))
                    {
                        continue;
                    }
                    if let Some((base, path)) = Self::arg_reborrow_base(arg) {
                        plan.base_snapshots
                            .entry(base.clone())
                            .or_insert_with(|| self.borrow.snapshot_base(base.as_str()));
                        plan.ref_args.push((
                            i,
                            base,
                            path,
                            Self::is_mut_ref(pty),
                            matches!(arg, Expr::Borrow(_)),
                        ));
                    }
                }
                Some(pty) if self.result_can_hold_reference(pty) => {
                    if let Expr::Identifier(id) = arg {
                        plan.held_args.push(id.name.to_string());
                    }
                }
                _ => {}
            }
        }
    }

    /// Two `&mut` arguments of one call, or a `&mut` and a `&`, must not reach the same variable,
    /// as in `f(&mut x, &mut x)` or `f(&mut h, &h.n)`: the callee would see one value through two
    /// names that it may assume are separate. Raw pointer parameters (`*mut`) may alias, as in Rust.
    fn check_overlapping_reference_args(
        &mut self,
        plan: &ReborrowPlan,
        errors_before_args: usize,
        span: &Span,
    ) {
        let Some((param_types, _)) = &plan.callee_sig else {
            return;
        };
        let mut borrows = plan
            .ref_args
            .iter()
            .filter(|(i, ..)| matches!(param_types.get(*i), Some(Type::Borrow { .. })))
            .map(|(_, base, path, is_mut, _)| (base.clone(), path.clone(), *is_mut))
            .collect::<Vec<_>>();
        Self::add_held_borrows(&mut borrows, &plan.held_args, &plan.base_snapshots);
        self.check_overlapping_borrows(&borrows, &plan.base_snapshots, errors_before_args, span);
    }

    /// A by-value argument that holds a borrow takes part in the overlap check too, as mutable
    /// as the borrow it holds: `f(&mut x, s)`, where `s` holds `&x`, gives the callee `x` twice.
    /// The other argument's snapshot of `x`, taken before the arguments were checked, still has
    /// `s`'s borrow, which may have ended at this call since.
    fn add_held_borrows(
        borrows: &mut Vec<(String, Vec<String>, bool)>,
        held: &[String],
        snapshots: &HashMap<String, Option<Vec<BorrowRecord>>>,
    ) {
        for var in held {
            let is_mut = snapshots.values().flatten().flatten().any(|record| {
                record.borrower_name.as_deref() == Some(var.as_str()) && record.is_mut
            });
            borrows.push((var.clone(), Vec::new(), is_mut));
        }
    }

    /// The shared part of the check above, also used for a method's receiver and arguments.
    /// `borrows` holds each borrowed argument's variable, field path and whether it is `&mut`;
    /// `snapshots` holds the borrows of those variables from before the arguments were checked.
    fn check_overlapping_borrows(
        &mut self,
        borrows: &[(String, Vec<String>, bool)],
        snapshots: &HashMap<String, Option<Vec<BorrowRecord>>>,
        errors_before_args: usize,
        span: &Span,
    ) {
        if self.speculating {
            return;
        }
        let borrows = Self::places_reached(borrows, snapshots);
        for (i, (places, is_mut)) in borrows.iter().enumerate() {
            for (other_places, other_is_mut) in &borrows[i + 1..] {
                let shared = places.iter().find(|(base, path)| {
                    other_places.iter().any(|(other_base, other_path)| {
                        base == other_base && crate::hir::places::paths_may_alias(path, other_path)
                    })
                });
                let Some((base, _)) = shared else {
                    continue;
                };
                let (code, message) = if *is_mut && *other_is_mut {
                    (
                        crate::diagnostic::DiagnosticCode::E4004,
                        format!("Cannot borrow '{base}' as mutable twice in one call."),
                    )
                } else if *is_mut || *other_is_mut {
                    (
                        crate::diagnostic::DiagnosticCode::E4003,
                        format!("Cannot borrow '{base}' as mutable and as immutable in one call."),
                    )
                } else {
                    continue;
                };
                // A borrow still held after this statement is already reported by the argument's
                // own borrow check.
                let quoted = format!("'{base}'");
                let already_reported = self.errors.inner[errors_before_args..].iter().any(|d| {
                    matches!(
                        d.code,
                        Some(
                            crate::diagnostic::DiagnosticCode::E4003
                                | crate::diagnostic::DiagnosticCode::E4004
                        )
                    ) && d.message.contains(&quoted)
                });
                if already_reported {
                    continue;
                }
                self.errors.error_with_code(
                    code,
                    message,
                    Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
                );
            }
        }
    }

    /// Every place each borrowed argument of a call reaches, with whether the argument is
    /// `&mut`. `borrows` and `snapshots` are those of `check_overlapping_borrows`.
    fn places_reached(
        borrows: &[(String, Vec<String>, bool)],
        snapshots: &BaseSnapshots,
    ) -> Vec<(Vec<Place>, bool)> {
        borrows
            .iter()
            .map(|(base, path, is_mut)| {
                // An argument that is a reference variable, `a` after `let a = &mut x`, also
                // reaches `x`: the call's other arguments must not borrow `x` either.
                let mut places = vec![(base.clone(), path.clone())];
                for (borrowed, records) in snapshots {
                    for record in records.iter().flatten() {
                        if record.borrower_name.as_deref() == Some(base.as_str()) {
                            let mut through = record.path.clone();
                            through.extend(path.iter().cloned());
                            places.push((borrowed.clone(), through));
                        }
                    }
                }
                (places, *is_mut)
            })
            .collect::<Vec<_>>()
    }

    /// After the argument loop, decide which reference-argument reborrows persist past the call
    /// and revert the rest (#243). A reborrow outlives the call iff the callee returns a reference
    /// the argument's slot derives from; every other base is restored to the borrows it had before
    /// the call (releasing the `&x` literals `check_borrow_expr` over-recorded). No-op while
    /// speculating or when the callee signature was unresolved. Split out of
    /// `check_functioncall_expr` (R4, #279).
    fn commit_reference_arg_reborrows(
        &mut self,
        plan: &ReborrowPlan,
        arg_types: &[Type],
        span: &Span,
    ) {
        if self.speculating || plan.callee_sig.is_none() {
            return;
        }
        // An argument's reborrow outlives the call iff the callee returns a reference
        // and its result derives from that argument's parameter slot. For an imported
        // callee the summary comes from the registry's inline code (#265 step 7); for an
        // in-compilation callee, from the AST `return_provenances` summary.
        let persists = |i: usize| {
            plan.ret_is_ref
                && match plan.imported_prov {
                    Some(code) => crate::hir::provenance::inline_prov_includes(code, i),
                    None => plan.return_prov.includes(i),
                }
        };
        // (a) Record reborrows for bare-reference arguments (`foo(m)`); `&x` literals
        //     were already recorded by `check_borrow_expr` during the arg loop. A
        //     non-deriving argument still runs the conflict check but records nothing —
        //     so passing the same reference to two parameters of a non-reference-
        //     returning call (`rmsnorm(x, x, ..)`, an in-place reborrow) does not
        //     self-conflict, matching the pre-#243 behaviour.
        for (i, base, path, is_mut_param, is_borrow_lit) in &plan.ref_args {
            if *is_borrow_lit || !Self::is_ref_type(&arg_types[*i]) {
                continue;
            }
            self.track_reference_arg_borrow(
                base,
                path.clone(),
                *is_mut_param,
                plan.result_is_mut || (plan.ret_holds_ref && *is_mut_param),
                persists(*i),
                span,
            );
            // `let r = keep(m)`: `r` may point where `m` points, so it holds every borrow
            // `m` holds, as it would after `let r = m`.
            if persists(*i) {
                if let Some(target) = self.current_assignment_target.clone() {
                    self.borrow.copy_borrows(base, &target, self.scopes.len());
                }
            }
        }
        // (b) Selective revert: a base keeps its borrow past the call iff at least one
        //     of its argument positions is one the return derives from. Non-deriving
        //     bases are restored to their pre-call state (call-duration borrow only).
        //     This is what releases the `&x` literals `check_borrow_expr` over-recorded.
        let deriving: std::collections::HashSet<&str> = plan
            .ref_args
            .iter()
            .filter(|(i, _, _, _, _)| persists(*i))
            .map(|(_, base, _, _, _)| base.as_str())
            .collect();
        for (base, snap) in &plan.base_snapshots {
            if deriving.contains(base.as_str()) {
                continue;
            }
            // Drop only the borrows *this call* added; keep exactly the records that
            // were present pre-call and still are. Restoring the raw snapshot instead
            // would resurrect borrows the arg loop legitimately NLL-released, firing
            // spurious conflicts later.
            let prev: &[BorrowRecord] = snap.as_deref().unwrap_or(&[]);
            self.borrow.retain_present(base.as_str(), prev);
        }
    }

    /// E8006: a call must meet the called function's `requires`. Each one, with the call's
    /// arguments in place of the parameters, must follow from what the prover knows at the call.
    /// An argument the prover cannot model, such as another call, is an unknown value.
    fn check_call_requires(&mut self, name: &str, args: &[Expr], span: &crate::syntax::Span) {
        if self.speculating {
            return;
        }
        let Some(func) = self.env.syntax_functions.get(name) else {
            return;
        };
        if func.requires.is_empty() {
            return;
        }
        let mut names = HashMap::new();
        for (i, ((param, _), arg)) in func.params.iter().zip(args).enumerate() {
            let value = if crate::hir::prover::is_modelled(arg) {
                arg.clone()
            } else {
                Expr::Identifier(IdentifierExpr {
                    name: format!("$arg_{}_{}_{i}", span.line, span.column).into(),
                    span: *span,
                })
            };
            names.insert(param.clone(), value);
        }
        for req in &func.requires {
            let at_call = crate::hir::prover::substitute_names(req, &names);
            if !self.prove_expr(&at_call) {
                let condition = crate::hir::prover::to_source(req)
                    .map(|c| format!(" `{c}`"))
                    .unwrap_or_default();
                self.errors.error_with_code(
                    crate::diagnostic::DiagnosticCode::E8006,
                    format!(
                        "this call to '{name}' may not meet its precondition{condition}: \
                         the compiler cannot prove it from what is known here"
                    ),
                    Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
                );
            }
        }
    }

    /// Checks a call to a function by name, `f(a, b)`, and gives the type of its result. The
    /// name can be a built-in function, a local variable that holds something callable, a
    /// declared function, a generic function, a static method or a function from an imported
    /// library. A call of a generic function or of a closure value is rewritten to a call of
    /// the function that is made for it.
    ///
    /// # Panics
    ///
    /// Panics if `expr` is not a call to a function by name.
    pub(crate) fn check_functioncall_expr(&mut self, expr: &mut Expr) -> Type {
        if let Some(ty) = self.fold_needs_drop_call(expr) {
            return ty;
        }
        let Expr::FunctionCall(FunctionCallExpr {
            name,
            type_args,
            args,
            span,
        }) = expr
        else {
            panic!("Expected IndexAccess, got {expr:?}")
        };
        let resolved_name = name.clone();
        // Calling a local, `p(env, x)` with `p` a closure's function, uses it.
        if self.lookup(&resolved_name).is_some() {
            self.used_vars.insert(resolved_name.clone());
        }
        let (base_name, explicit_generic_args) =
            self.split_generic_call_name(&resolved_name, type_args.as_deref());
        let arg_types =
            self.check_function_call_arguments(&resolved_name, args, &explicit_generic_args, span);
        let mut call = CallArgs {
            exprs: args.as_mut_slice(),
            types: &arg_types,
            span: &*span,
        };
        if let Some(intrinsic_ty) =
            self.resolve_intrinsic_function(&resolved_name, &call, &explicit_generic_args)
        {
            return intrinsic_ty;
        }
        if let Some((ty, replacement)) = self.check_call_of_local(&resolved_name, &call) {
            if let Some(new_call) = replacement {
                *expr = new_call;
            }
            return ty;
        }
        let callee = CalleeName {
            resolved: &resolved_name,
            base: &base_name,
            explicit_generic_args: &explicit_generic_args,
        };
        self.check_call_by_name(&callee, name, &mut call)
    }

    /// Rewrites `core::mem::needs_drop<T>()` to `true` or `false` and gives its type. Whether
    /// dropping a `T` runs anything is known only to the checker, so the call is answered
    /// here. Gives `None`, and changes nothing, for any other expression.
    fn fold_needs_drop_call(&mut self, expr: &mut Expr) -> Option<Type> {
        if let Expr::FunctionCall(fc) = expr {
            if fc.name.as_ref() == "needs_drop"
                && fc.args.is_empty()
                && self.env.generic_functions.contains_key("needs_drop")
            {
                if let Some([ty]) = fc.type_args.as_deref() {
                    if !ty.has_generic_params() {
                        let answer = !self.drops_glue(ty).is_empty();
                        *expr = Expr::Identifier(IdentifierExpr {
                            name: if answer { "true" } else { "false" }.into(),
                            span: fc.span,
                        });
                        return Some(Type::Scalar(ElementType::Bool));
                    }
                }
            }
        }
        None
    }

    /// Splits the name a call wrote into the function's name and the type arguments. The type
    /// arguments are written after the name (`type_args`) or inside it, as in `pick<i32>`. A
    /// constant type argument is reduced to the number it works out to.
    fn split_generic_call_name(
        &mut self,
        resolved_name: &Symbol,
        type_args: Option<&[Type]>,
    ) -> (Symbol, Vec<Type>) {
        let mut base_name = resolved_name.clone();
        let mut explicit_generic_args = type_args.map(<[Type]>::to_vec).unwrap_or_default();
        if explicit_generic_args.is_empty() {
            if let Some(start_idx) = resolved_name.find('<') {
                if let Some(end_idx) = resolved_name.rfind('>') {
                    base_name = format!(
                        "{}{}",
                        &resolved_name[..start_idx],
                        &resolved_name[end_idx + 1..]
                    )
                    .into();
                    let args_str = &resolved_name[start_idx + 1..end_idx];
                    explicit_generic_args = crate::syntax::split_type_args(args_str)
                        .into_iter()
                        .map(|s| self.parse_ty_str(s))
                        .collect();
                }
            }
        } else if let Some(idx) = resolved_name.find('<') {
            base_name = resolved_name[..idx].to_string().into();
        }
        self.fold_const_generic_args(&mut explicit_generic_args);
        (base_name, explicit_generic_args)
    }

    /// Checks each argument of a call to `resolved_name` and gives their types. Also checks the
    /// borrows the arguments make against each other, and ends the ones that do not last past
    /// the call.
    fn check_function_call_arguments(
        &mut self,
        resolved_name: &Symbol,
        args: &mut [Expr],
        explicit_generic_args: &[Type],
        span: &Span,
    ) -> Vec<Type> {
        // Mocking built-ins
        let mut arg_types = Vec::new();
        // A view reads the pointer it is handed and aliases the
        // storage behind it; it takes nothing (#336). `Verified(t)` is `t` itself,
        // wrapped, so it moves `t`.
        let is_builtin_ref = *resolved_name == "print".into()
            || *resolved_name == "tensor_view_2d".into()
            || *resolved_name == "matmul_into".into();
        // A call moves what it takes by value, whatever happens to its own result: inside
        // `print(take(t))`, `t` is moved into `take` though `print` takes nothing.
        let arg_consume = !is_builtin_ref;

        // Reference-argument reborrow bookkeeping (#243): snapshot each base before the
        // args are checked so the post-call revert can tell a call-duration borrow from
        // one the return keeps alive.
        let reborrow_plan = self.prepare_reference_arg_reborrows(resolved_name, args);
        let errors_before_args = self.errors.inner.len();
        for arg in args.iter_mut() {
            arg_types.push(self.check_expr_type_flag(arg, arg_consume));
        }
        // A number literal given to `Tensor<T, ..>::fill` takes the element type `T`.
        if *resolved_name == "Tensor::fill".into() {
            if let (Some(Type::Tensor(elem, _, _)), Some(value)) =
                (explicit_generic_args.first(), args.last_mut())
            {
                if crate::hir::expr::numeric_literal_mut(value).is_some() {
                    crate::hir::expr::clear_literal_type(value);
                    let ty = self.check_expr_expecting(
                        value,
                        Some(Type::Scalar(elem.clone())),
                        arg_consume,
                    );
                    if let Some(last) = arg_types.last_mut() {
                        *last = ty;
                    }
                }
            }
        }

        self.check_overlapping_reference_args(&reborrow_plan, errors_before_args, span);
        if let Some((params, _)) = &reborrow_plan.callee_sig {
            let params = params.clone();
            self.check_views_passed_by_value(resolved_name.as_ref(), &params, args);
        }
        self.commit_reference_arg_reborrows(&reborrow_plan, &arg_types, span);
        if *resolved_name == "print".into() && args.len() == 1 {
            Self::prepare_print_operand(&mut args[0], &arg_types[0]);
        }
        arg_types
    }

    /// Reports a call to an `unsafe` function made outside an `unsafe` function or block
    /// (E5001).
    fn require_unsafe_context(&mut self, callee_unsafe: bool, resolved_name: &str, span: &Span) {
        if callee_unsafe && !self.in_unsafe_block && !self.speculating {
            self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E5001,
                format!(
                    "Call to unsafe function '{resolved_name}' is unsafe and requires unsafe \
                     function or block"
                ),
                Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
            );
        }
    }

    /// Reports a call to a function that runs on another device than the code that calls it
    /// (E6001). `required` is where the function `callee` runs.
    fn check_call_topology(&mut self, callee: &str, required: &Topology, span: &Span) {
        if !required.same_device(&self.active_topology) && !self.speculating {
            self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E6001,
                format!(
                    "Type error: Function '{}' requires topology '{}', but is called from '{}'",
                    callee,
                    required.display_name(),
                    self.active_topology.display_name()
                ),
                Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
            );
        }
    }

    /// Checks a call whose name is a local variable that holds something callable: a function
    /// pointer, a closure pointer or a closure value. Gives `None` when the name is not such a
    /// variable.
    fn check_call_of_local(
        &mut self,
        resolved_name: &Symbol,
        call: &CallArgs<'_>,
    ) -> Option<Rewrite> {
        if let Some((Type::Function(param_types, ret_ty, callee_unsafe), _)) =
            self.lookup(resolved_name).cloned()
        {
            self.check_function_pointer_call(resolved_name, &param_types, callee_unsafe, call);
            Some((*ret_ty, None))
        } else if let Some((Type::Closure(param_types, ret_ty), _)) =
            self.lookup(resolved_name).cloned()
        {
            self.check_closure_pointer_variable_call(resolved_name, &param_types, call);
            Some((*ret_ty, None))
        } else if let Some((Type::Struct(struct_name, _), _)) = self.lookup(resolved_name).cloned()
        {
            Some(self.check_closure_variable_call(resolved_name, &struct_name, call))
        } else {
            None
        }
    }

    /// Checks a call through the local `resolved_name`, a function pointer: the call needs an
    /// `unsafe` context when the pointer's type says so, and the arguments must fit the
    /// parameters in the pointer's type.
    fn check_function_pointer_call(
        &mut self,
        resolved_name: &Symbol,
        param_types: &[Type],
        callee_unsafe: bool,
        call: &CallArgs<'_>,
    ) {
        let (args, arg_types, span) = (&*call.exprs, call.types, call.span);
        // The contract rides on the type, so a call through a name honours it just
        // as a call through the declaration does.
        self.require_unsafe_context(callee_unsafe, resolved_name, span);
        if args.len() != param_types.len() && !self.speculating {
            self.errors.push(format!(
                "Function pointer '{}' expects {} arguments, got {}",
                resolved_name,
                param_types.len(),
                args.len()
            ));
        } else {
            for (i, param_ty) in param_types.iter().enumerate() {
                let arg_ty = &arg_types[i];
                if !self.is_assignable(param_ty, arg_ty) && !self.speculating {
                    self.errors.push(format!(
                        "Type mismatch in argument {} for function pointer '{}'. Expected {}, got {}",
                        i + 1, resolved_name, param_ty, arg_ty
                    ));
                }
            }
        }
    }

    /// Checks the arguments of a call through the local `resolved_name`, a closure pointer,
    /// against the parameters in the pointer's type.
    fn check_closure_pointer_variable_call(
        &mut self,
        resolved_name: &Symbol,
        param_types: &[Type],
        call: &CallArgs<'_>,
    ) {
        let (args, arg_types) = (&*call.exprs, call.types);
        if args.len() != param_types.len() && !self.speculating {
            self.errors.push(format!(
                "Closure '{}' expects {} arguments, got {}",
                resolved_name,
                param_types.len(),
                args.len()
            ));
        } else {
            for (i, param_ty) in param_types.iter().enumerate() {
                let arg_ty = &arg_types[i];
                if !self.is_assignable(param_ty, arg_ty) && !self.speculating {
                    self.errors.push(format!(
                        "Type mismatch in argument {} for closure '{}'. Expected {}, got {}",
                        i + 1,
                        resolved_name,
                        param_ty,
                        arg_ty
                    ));
                }
            }
        }
    }

    /// Checks a call of the local `resolved_name`, a closure value whose type is the struct
    /// `struct_name`. Gives the result type and the call of the closure's function to put in
    /// place of this call, or the `Unknown` type after reporting why the value cannot be called.
    fn check_closure_variable_call(
        &mut self,
        resolved_name: &Symbol,
        struct_name: &str,
        call: &CallArgs<'_>,
    ) -> Rewrite {
        if struct_name.starts_with("Closure_") {
            let call_name = format!("{struct_name}_call");
            if let Some(func) = self
                .mono
                .functions
                .iter()
                .find(|f| f.0.name == call_name.as_str().into())
            {
                let param_types: Vec<Type> = func
                    .0
                    .params
                    .iter()
                    .skip(1)
                    .map(|(_, t)| t.clone())
                    .collect();
                let return_type = func.0.return_type.clone();
                self.check_closure_variable_args(resolved_name, &param_types, call);

                let mut new_args = vec![Expr::Borrow(BorrowExpr {
                    expr: Box::new(Expr::Identifier(IdentifierExpr::new(
                        resolved_name.clone(),
                        Span::default(),
                    ))),
                    is_mut: true,
                    span: Span::default(),
                })];
                new_args.extend(call.exprs.to_vec());

                let new_call = Expr::FunctionCall(FunctionCallExpr {
                    name: call_name.into(),
                    type_args: None,
                    args: new_args,
                    span: Span::default(),
                });

                (return_type, Some(new_call))
            } else {
                if !self.speculating {
                    self.errors.push(format!(
                        "Missing call method for closure struct '{struct_name}'"
                    ));
                }
                (Type::Unknown, None)
            }
        } else {
            if !self.speculating {
                self.errors
                    .push(format!("Cannot call non-closure struct '{resolved_name}'"));
            }
            (Type::Unknown, None)
        }
    }

    /// Checks the arguments of a call of the local `resolved_name`, a closure value, against
    /// the closure's parameters: the count first, then each type.
    fn check_closure_variable_args(
        &mut self,
        resolved_name: &Symbol,
        param_types: &[Type],
        call: &CallArgs<'_>,
    ) {
        let (args, arg_types) = (&*call.exprs, call.types);
        if args.len() == param_types.len() {
            for (i, param_ty) in param_types.iter().enumerate() {
                let arg_ty = &arg_types[i];
                if !self.is_assignable(param_ty, arg_ty) && !self.speculating {
                    self.errors.push(format!(
                        "Type mismatch in argument {} for closure '{}'. Expected {}, got {}",
                        i + 1,
                        resolved_name,
                        param_ty,
                        arg_ty
                    ));
                }
            }
        } else if !self.speculating {
            self.errors.push(format!(
                "Closure '{}' expects {} arguments, got {}",
                resolved_name,
                param_types.len(),
                args.len()
            ));
        }
    }

    /// Checks a call whose name is not a built-in function or a local variable. The name is
    /// looked up as a declared function, then as an instance of a generic function, then as a
    /// generic function, then as a static method (a name with `::`), and last as a function
    /// from an imported library.
    fn check_call_by_name(
        &mut self,
        callee: &CalleeName<'_>,
        name: &mut Symbol,
        call: &mut CallArgs<'_>,
    ) -> Type {
        let resolved_name = callee.resolved;
        let span = call.span;
        if let Some((ret_ty, is_unsafe, param_types, req_topology, _, _)) =
            self.env.functions.get(resolved_name.as_ref())
        {
            self.record_call_edge(resolved_name.as_ref(), span);
            self.check_call_topology(resolved_name, req_topology, span);
            self.require_unsafe_context(*is_unsafe, resolved_name, span);
            self.check_call_args(resolved_name.as_ref(), param_types, call);
            self.check_call_requires(resolved_name.as_ref(), call.exprs, span);
            ret_ty.clone()
        } else if let Some((mono_topology, param_types, mono_ret)) = self
            .mono
            .functions
            .iter()
            .find(|f| f.0.name == *resolved_name)
            .map(|func| {
                (
                    func.0.topology.clone(),
                    func.0
                        .params
                        .iter()
                        .map(|(_, t)| t.clone())
                        .collect::<Vec<Type>>(),
                    func.0.return_type.clone(),
                )
            })
        {
            self.record_call_edge(resolved_name.as_ref(), span);
            self.check_call_topology(resolved_name, &mono_topology, span);
            self.check_call_args(resolved_name.as_ref(), &param_types, call);
            mono_ret
        } else if let Some((generic_func, origin_hash)) = self
            .env
            .generic_functions
            .get(callee.base.as_ref())
            .copied()
        {
            // A generic `unsafe fn` resolves here rather than through `env.functions`,
            // so the rule is repeated: being generic does not discharge a contract.
            self.require_unsafe_context(generic_func.is_unsafe, resolved_name, span);
            let generic = GenericCallee {
                func: generic_func,
                origin_hash,
                explicit_args: callee.explicit_generic_args,
            };
            self.instantiate_generic_function_call(&generic, resolved_name, name, call)
                .unwrap_or(Type::Unknown)
        } else if resolved_name.contains("::") {
            self.check_static_method_call(resolved_name, name, call)
        } else {
            self.check_imported_or_undefined_call(resolved_name, call)
        }
    }

    /// Checks a call to a function this compile has no declaration of. A function from an
    /// imported library is checked against the signature the library gives. Any other name is
    /// reported as undefined, or as ambiguous when two imported libraries define it (E2002).
    fn check_imported_or_undefined_call(
        &mut self,
        resolved_name: &Symbol,
        call: &mut CallArgs<'_>,
    ) -> Type {
        let span = call.span;
        if let Some(sig) = self
            .worker
            .global
            .registry
            .fn_sigs
            .get(&crate::symbol::Symbol::from(resolved_name.as_ref()))
            .cloned()
        {
            self.record_call_edge(resolved_name.as_ref(), span);
            // Imported callee resolved from a merged `.vxlib` interface (#219 flip, phase 2):
            // its AST is absent from this compile, so type-check the call against the registry
            // `FnSig` — arg count + per-argument assignability against `params`, result type is
            // `ret_ty`. The flat codegen links the body separately via `body_of`. In a normal
            // compile the registry is empty, so this arm is inert and the call falls to E2002.
            // The signature carries the callee's unsafe-ness precisely so an importing
            // module can refuse the call without seeing the body. Read it here, or the
            // flag crosses the boundary and means nothing.
            self.require_unsafe_context(sig.is_unsafe, resolved_name, span);
            self.check_call_args(resolved_name.as_ref(), &sig.params, call);
            sig.ret_ty.clone()
        } else if self
            .worker
            .global
            .registry
            .poisoned_reason(&crate::symbol::Symbol::from(resolved_name.as_ref()))
            .is_some()
        {
            // Not undefined — defined in *more than one* imported artifact and tombstoned
            // by the order-independent merge (#291). "Undefined" would send the reader
            // hunting for a missing import when the cause is a duplicate one; report the
            // ambiguity, in the same wording as the struct path (#294). `Unknown` is the
            // poison result type, so the return check doesn't add a noise mismatch.
            if !self.speculating {
                self.errors.error_with_code(
                    crate::diagnostic::DiagnosticCode::E2002,
                    crate::registry::ambiguous_import_message("Function", resolved_name),
                    Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
                );
            }
            Type::Unknown
        } else {
            let mono_names: Vec<crate::symbol::Symbol> = self
                .mono
                .functions
                .iter()
                .map(|(f, _)| f.name.clone())
                .collect();
            if !self.speculating {
                self.errors.error_with_code(
                    crate::diagnostic::DiagnosticCode::E2002,
                    format!(
                        "Undefined function '{resolved_name}'. Available monos: {mono_names:?}"
                    ),
                    Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
                );
            }
            Type::Unknown
        }
    }

    /// Check a call's arguments against the callee's parameter list: the count first (E3010),
    /// then each argument's type against its parameter (E3003). A literal argument is refined to
    /// the parameter's type before the comparison, so `f(7)` fits an `i64` parameter. Every
    /// resolution path in `check_functioncall_expr` runs this same check, so it lives here once.
    pub(crate) fn check_call_args(
        &mut self,
        callee: &str,
        params: &[Type],
        call: &mut CallArgs<'_>,
    ) {
        let span = call.span;
        if call.exprs.len() != params.len() {
            if !self.speculating {
                self.errors.error_with_code(
                    crate::diagnostic::DiagnosticCode::E3010,
                    format!(
                        "Function '{}' expects {} arguments, got {}",
                        callee,
                        params.len(),
                        call.exprs.len()
                    ),
                    Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
                );
            }
            return;
        }
        for (i, param_ty) in params.iter().enumerate() {
            let arg_ty = self.refine_literal_arg(&mut call.exprs[i], param_ty, &call.types[i]);
            // An `Unknown` argument already has its error, a use of a moved value say.
            if !self.is_assignable(param_ty, &arg_ty)
                && arg_ty != Type::Unknown
                && !self.speculating
            {
                if self.report_pointer_in_wrong_memory(callee, i, (param_ty, &arg_ty), span) {
                    continue;
                }
                self.errors.error_with_code(
                    crate::diagnostic::DiagnosticCode::E3003,
                    format!(
                        "Type mismatch in argument {} for function '{}'. Expected {}, got {}{}",
                        i + 1,
                        callee,
                        param_ty,
                        arg_ty,
                        Self::rank_note(param_ty, &arg_ty)
                    ),
                    Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
                );
            }
        }
    }

    /// Reports argument `i` of a call to `callee` when it points to the right thing in the wrong
    /// memory (E6029), and gives `true`. `types` is the parameter's type, then the argument's.
    /// Gives `false`, and reports nothing, for any other mismatch.
    fn report_pointer_in_wrong_memory(
        &mut self,
        callee: &str,
        i: usize,
        types: (&Type, &Type),
        span: &Span,
    ) -> bool {
        // Two pointers to the same thing in different memories: say so, rather than
        // printing two types that differ only after `in`.
        if let (Type::Pointer(p_inner, p_place, _), Type::Pointer(a_inner, a_place, _)) = types {
            if p_place != a_place && self.is_assignable(p_inner, a_inner) {
                let memory = |place: &Option<crate::syntax::Placement>| match place {
                    Some(p) => p.as_written(),
                    None => "host memory".to_string(),
                };
                self.errors.error_with_code(
                    crate::diagnostic::DiagnosticCode::E6029,
                    format!(
                        "argument {} of '{}' is a pointer into {}, but the parameter \
                         wants a pointer into {}: a pointer is only passed where its \
                         memory is expected",
                        i + 1,
                        callee,
                        memory(a_place),
                        memory(p_place),
                    ),
                    Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
                );
                return true;
            }
        }
        false
    }

    /// Refuse a call whose mapping leaves one of the callee's type parameters unbound (E3016,
    /// once per parameter). Left alone, the parameter travels into the instance's symbol name
    /// as its bare letter and `sizeof<T>()` in the body folds to a default. Const and topology
    /// parameters bind through their own bookkeeping and are not looked at here.
    pub(crate) fn require_type_params_bound(
        &mut self,
        callee: &str,
        generics: &[decl::GenericParam],
        mapping: &HashMap<crate::symbol::Symbol, Type>,
        span: &crate::syntax::Span,
    ) -> bool {
        let mut all_bound = true;
        for param in generics {
            let decl::GenericParam::Type { name, bounds } = param else {
                continue;
            };
            if bounds.iter().any(|b| b.trait_name.as_ref() == "Topology") {
                continue;
            }
            // A dimension parameter (`<N>` in `Tensor<f32, [N]>`) binds to its extent as a
            // `Type::Generic` holding the name or literal, so presence is all that is asked.
            if mapping.contains_key(name) {
                continue;
            }
            all_bound = false;
            if !self.speculating {
                self.errors.error_with_code(
                    crate::diagnostic::DiagnosticCode::E3016,
                    format!(
                        "Type parameter '{name}' of '{callee}' is not determined by this call: no argument \
                         or declared result type binds it. Spell it out as a type argument, or \
                         give the result a type."
                    ),
                    Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
                );
            }
        }
        all_bound
    }

    /// Fold a call to a comptime lambda, or refuse it.
    ///
    /// A comptime lambda is a function that runs while compiling, so every call has to work
    /// out to a value: with all its calls folded the lambda itself is unused and goes away,
    /// which is what keeps a `comptime` block from leaving anything behind. A call it cannot
    /// fold -- an argument only known at run time, say -- has nowhere to go and is refused.
    /// That is stricter than a C++ `constexpr`, which would fall back to a run-time call.
    pub(crate) fn fold_comptime_lambda_call(
        &mut self,
        expr: &mut Expr,
        span: &crate::syntax::Span,
    ) {
        let Expr::FunctionCall(call) = expr else {
            return;
        };
        if !Self::is_closure_body(call.name.as_ref()) {
            if self
                .callee_body(call.name.as_ref())
                .is_some_and(Self::is_comptime_fn)
            {
                self.fold_comptime_fn_call(expr, span);
            }
            return;
        }
        if !self.callee_is_comptime(call.name.as_ref()) {
            return;
        }
        // The enclosing comptime interpreter must observe the call before it is replaced with a
        // constant. Folding it here erases a captured write from that outer observation.
        if self.consteval.comptime_depth > 0 {
            return;
        }
        let env = self.consteval_snapshot();
        let folded = self
            .eval_expr(expr, &env)
            .and_then(|value| Self::value_to_expr(&value, span));
        match folded {
            Some(constant) => *expr = constant,
            None => {
                if !self.speculating {
                    self.errors.error_with_code(
                        crate::diagnostic::DiagnosticCode::E3033,
                        "this call to a `comptime` lambda cannot be evaluated: a comptime \
                         lambda runs while compiling, so every argument has to be known then"
                            .to_string(),
                        Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
                    );
                }
            }
        }
    }

    /// Fold a call to a comptime function, or refuse it, as a comptime lambda's call is.
    ///
    /// The call runs in the `comptime` interpreter, so the function's body can use what a
    /// `comptime` block can: byte buffers made from string literals, among them.
    fn fold_comptime_fn_call(&mut self, expr: &mut Expr, span: &crate::syntax::Span) {
        // The enclosing comptime interpreter must observe the call before it is replaced.
        if self.consteval.comptime_depth > 0 {
            return;
        }
        let (ret_ty, takes_params) = match &*expr {
            Expr::FunctionCall(call) => match self.callee_body(call.name.as_ref()) {
                Some(func) => (Some(func.return_type.clone()), !func.params.is_empty()),
                None => (None, true),
            },
            _ => (None, true),
        };
        let before = self.consteval_snapshot();
        let observed = self.observe_comptime_block(&[], Some(&*expr), &before);
        let outcome = &observed.outcome;
        let folded = (observed.escaping_write.is_none()
            && !observed.call_depth_exceeded
            && !outcome.requires_refusal
            && outcome.support.is_supported())
        .then_some(outcome.value.concrete.as_ref())
        .flatten()
        .and_then(|value| Self::value_to_expr(value, span));
        match folded {
            Some(mut constant) => {
                // The literal takes the function's return type. Left bare it is an `i32`.
                if let (Expr::Number(n), Some(Type::Scalar(elem))) = (&mut constant, &ret_ty) {
                    n.ty.get_or_insert_with(|| elem.clone());
                }
                *expr = constant;
            }
            // Without parameters the function's block was worked out where it is written, and
            // a failure was reported there.
            None => {
                if !self.speculating && takes_params {
                    self.errors.error_with_code(
                        crate::diagnostic::DiagnosticCode::E3033,
                        "this call to a `comptime` function cannot be evaluated: it runs \
                         while compiling, so every argument has to be known then"
                            .to_string(),
                        Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
                    );
                }
            }
        }
    }

    /// Whether a function is a comptime function: its whole body is `return comptime { .. };`.
    /// It runs only while compiling, so every call folds and the function is never emitted.
    #[must_use]
    pub fn is_comptime_fn(func: &Function) -> bool {
        !Self::is_closure_body(func.name.as_ref())
            && matches!(func.body.as_slice(),
                [Statement::Return(r)] if matches!(&r.expr, Some(Expr::ComptimeBlock(_))))
    }

    /// Whether a function is the generated body of a comptime lambda.
    #[must_use]
    pub fn is_comptime_lambda_body(func: &Function) -> bool {
        Self::is_closure_body(func.name.as_ref())
            && func.body.iter().any(|stmt| match stmt {
                Statement::Return(r) => matches!(&r.expr, Some(Expr::ComptimeBlock(_))),
                Statement::ExprStmt(e) => matches!(&e.expr, Expr::ComptimeBlock(_)),
                _ => false,
            })
    }

    /// Whether a generated closure body is a `comptime` one.
    fn callee_is_comptime(&self, name: &str) -> bool {
        let Some(func) = self.comptime_closure_body(name) else {
            return false;
        };
        func.body.iter().any(|stmt| match stmt {
            Statement::Return(r) => {
                matches!(&r.expr, Some(Expr::ComptimeBlock(_)))
            }
            Statement::ExprStmt(e) => matches!(&e.expr, Expr::ComptimeBlock(_)),
            _ => false,
        })
    }

    /// Reduce each constant generic argument to the number it works out to.
    ///
    /// `countdown<N - 1>()` inside `countdown<3>` has to become `countdown<2>`, not
    /// `countdown<3 - 1>`. The instance is named after its arguments, so an unfolded one
    /// names a different instance every time round and the recursion never meets a base
    /// case -- it took the compiler's stack down rather than terminating.
    ///
    /// An argument the evaluator cannot work out is left as written, which is what it did
    /// before this.
    fn fold_const_generic_args(&mut self, args: &mut [Type]) {
        for arg in args.iter_mut() {
            let Type::Const(expr) = arg else {
                continue;
            };
            if matches!(&**expr, Expr::Number(_)) {
                continue;
            }
            let env = self.consteval_snapshot();
            let Some(value) = self.eval_expr(expr, &env).and_then(|v| match v {
                Value::Int(i) => Some(i.to_string()),
                #[allow(
                    clippy::cast_possible_truncation,
                    reason = "the guard lets only a whole number through, and a constant generic \
                              argument is far inside the range of an i64"
                )]
                Value::Number(n) if n.fract() == 0.0 => Some((n as i64).to_string()),
                _ => None,
            }) else {
                continue;
            };
            **expr = Expr::Number(NumberExpr {
                value: value.into(),
                ty: None,
                span: Span::default(),
            });
        }
    }

    /// Resolve and instantiate a `Struct::method(...)` static call (an inherent-impl method
    /// named through its type). Parses any explicit type args on the struct or method, finds the
    /// matching `_inherent` impl method, deduces what those left open from the arguments and
    /// from the type declared for the result, instantiates it, checks the arguments against the
    /// instance, rewrites `name` to the mangled instance and checks that instance once. Returns
    /// the instance return type, or `Unknown` after an undefined-method error or when a type
    /// parameter is left unbound. Split out of `check_functioncall_expr`.
    fn check_static_method_call(
        &mut self,
        resolved_name: &Symbol,
        name: &mut Symbol,
        call: &mut CallArgs<'_>,
    ) -> Type {
        let (mut struct_name, method_name, explicit_ty_str) =
            Self::split_static_call_name(resolved_name);

        // `Trait::method(&x)`: the trait chooses the impl, and the first argument's type is the
        // type that impl is for. This is how a call picks between two traits' same-named methods.
        let mut only_trait: Option<String> = None;
        if self.env.traits.contains_key(struct_name.as_str()) {
            let Some(self_name) =
                self.trait_call_receiver_name(resolved_name, &struct_name, call.types)
            else {
                return Type::Unknown;
            };
            only_trait = Some(std::mem::replace(&mut struct_name, self_name));
        }

        let target = StaticTarget {
            struct_name: &struct_name,
            method_name: &method_name,
            explicit_types: &explicit_ty_str,
            only_trait: only_trait.as_deref(),
        };
        let Some((chosen, chosen_seg)) = self.choose_static_impl(&target, call) else {
            return Type::Unknown;
        };
        let found = chosen.and_then(|ib| self.find_static_method(ib, &target, &chosen_seg));

        if let Some(found) = found {
            self.instantiate_static_method(found, resolved_name, name, call)
        } else {
            if !self.speculating {
                match &only_trait {
                    Some(t) => self.errors.push(format!(
                        "'{struct_name}' has no impl of '{t}' with a method '{method_name}'."
                    )),
                    None => self
                        .errors
                        .push(format!("Undefined static method '{resolved_name}'.")),
                }
            }
            Type::Unknown
        }
    }

    /// Splits the name of a static call, `Type<A>::method<B>`, into the type's name, the
    /// method's name, and the type arguments written on either of them, as text.
    ///
    /// # Panics
    ///
    /// Panics if `resolved_name` has no `::`. The caller checks for it first.
    fn split_static_call_name(resolved_name: &str) -> (String, String, String) {
        let idx = resolved_name.find("::").expect("caller guards on `::`");
        let mut struct_name = resolved_name[..idx].to_string();
        let mut method_name = resolved_name[idx + 2..].to_string();
        let mut explicit_ty_str = String::new();

        if let Some(lt) = struct_name.find('<') {
            if struct_name.ends_with('>') {
                explicit_ty_str = struct_name[lt + 1..struct_name.len() - 1].to_string();
                struct_name = struct_name[..lt].to_string();
            }
        }

        if let Some(lt) = method_name.find('<') {
            if method_name.ends_with('>') {
                let method_ty_str = method_name[lt + 1..method_name.len() - 1].to_string();
                if explicit_ty_str.is_empty() {
                    explicit_ty_str = method_ty_str;
                } else {
                    explicit_ty_str = format!("{explicit_ty_str}, {method_ty_str}");
                }
                method_name = method_name[..lt].to_string();
            }
        }
        (struct_name, method_name, explicit_ty_str)
    }

    /// The name of the type whose impl a call written `Trait::method(x, ..)` reaches: the type
    /// of the first argument, read through a reference. `struct_name` is the trait's name.
    /// Reports the call and gives `None` when there is no first argument or its type has no
    /// name.
    fn trait_call_receiver_name(
        &mut self,
        resolved_name: &Symbol,
        struct_name: &str,
        arg_types: &[Type],
    ) -> Option<String> {
        let self_ty = arg_types.first().map(|t| match t {
            Type::Borrow { inner, .. } | Type::Ref(inner, _) => inner.as_ref(),
            other => other,
        });
        let self_name = match self_ty {
            Some(Type::Struct(n, _) | Type::Enum(n, _)) => Some(n.to_string()),
            Some(Type::GenericInstance(inner, _)) => match inner.as_ref() {
                Type::Struct(n, _) | Type::Enum(n, _) => Some(n.to_string()),
                _ => None,
            },
            Some(Type::Scalar(el)) => Some(el.to_string()),
            _ => None,
        };
        let Some(self_name) = self_name else {
            if !self.speculating {
                self.errors.push(format!(
                    "'{resolved_name}' needs the value whose '{struct_name}' impl it calls as its first argument."
                ));
            }
            return None;
        };
        Some(self_name)
    }

    /// Reports a static call that the impls of two or more traits accept (E3035).
    fn report_ambiguous_static_method(
        &mut self,
        target: &StaticTarget<'_>,
        from_traits: &[(&Symbol, &decl::ImplBlock)],
        span: &Span,
    ) {
        if !self.speculating {
            let mut names: Vec<String> = from_traits
                .iter()
                .map(|(t, ib)| impl_display(t, ib))
                .collect();
            names.sort();
            let mut traits: Vec<&str> = from_traits.iter().map(|(t, _)| t.as_ref()).collect();
            traits.sort_unstable();
            self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E3035,
                format!(
                    "Method '{}' on '{}' is defined by more than one impl ({}); name the \
                     trait to choose one, as in '{}::{}(..)'",
                    target.method_name,
                    target.struct_name,
                    names.join(", "),
                    traits[0],
                    target.method_name
                ),
                Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
            );
        }
    }

    /// Reports a static call whose arguments no impl accepts, and names the impls that exist
    /// (E3036).
    fn report_no_accepting_impl(
        &mut self,
        target: &StaticTarget<'_>,
        candidate_impls: &[(&Symbol, &decl::ImplBlock)],
        call: &CallArgs<'_>,
    ) {
        if !self.speculating {
            let mut names: Vec<String> = candidate_impls
                .iter()
                .map(|(t, ib)| impl_display(t, ib))
                .collect();
            names.sort();
            let got: Vec<String> = call
                .types
                .iter()
                .map(std::string::ToString::to_string)
                .collect();
            self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E3036,
                format!(
                    "No impl of '{}' on '{}' takes ({}); '{}' implements {}",
                    target.method_name,
                    target.struct_name,
                    got.join(", "),
                    target.struct_name,
                    names.join(", ")
                ),
                Some(crate::diagnostic::SourceSpan::from_ast_span(call.span)),
            );
        }
    }

    /// The method `target` names in the impl block `ib`, with the type parameters it has to
    /// bind and the ones the call's written type arguments already bind. `chosen_seg` is the
    /// trait part of the instance's name. Gives `None` when the block has no such method.
    fn find_static_method(
        &self,
        ib: &decl::ImplBlock,
        target: &StaticTarget<'_>,
        chosen_seg: &str,
    ) -> Option<FoundStaticMethod> {
        let method_name = target.method_name;
        let m = ib.methods.iter().find(|m| m.name == method_name.into())?;
        // A method's own list repeats the impl block's parameters, so
        // take each name once or an unbound one is reported twice.
        let mut found_generics = ib.generics.clone();
        for g in &m.generics {
            if !found_generics.iter().any(|f| f.name() == g.name()) {
                found_generics.push(g.clone());
            }
        }
        let mut found_mapping: HashMap<Symbol, Type> = HashMap::new();
        if !target.explicit_types.is_empty() {
            let explicit_args = self.parse_type_list(target.explicit_types);
            for (i, parsed_ty) in explicit_args.into_iter().enumerate() {
                if i < ib.generics.len() {
                    found_mapping.insert(ib.generics[i].name().into(), parsed_ty);
                }
            }
        }
        Some(FoundStaticMethod {
            func: m.clone(),
            generics: found_generics,
            mapping: found_mapping,
            instance_name: format!("{}::{}{}", target.struct_name, chosen_seg, method_name),
        })
    }

    /// Parses a list of types separated by commas, such as `i32, Vec<u8>`. A comma inside
    /// `<..>` belongs to the type it is in.
    fn parse_type_list(&self, explicit_ty_str: &str) -> Vec<Type> {
        let mut explicit_args = Vec::new();
        let mut depth = 0;
        let mut current = String::new();
        for c in explicit_ty_str.chars() {
            if c == '<' {
                depth += 1;
                current.push(c);
            } else if c == '>' {
                depth -= 1;
                current.push(c);
            } else if c == ',' && depth == 0 {
                explicit_args.push(self.parse_ty_str(&current));
                current.clear();
            } else {
                current.push(c);
            }
        }
        if !current.trim().is_empty() {
            explicit_args.push(self.parse_ty_str(&current));
        }
        explicit_args
    }

    /// Makes the instance of the method a static call resolved to. The type parameters the
    /// written type arguments left open are worked out from the arguments and from the type
    /// declared for the result. The arguments are checked against the instance, `name` is
    /// rewritten to the instance's name, and the instance is checked once. Gives the
    /// instance's return type, or `Unknown` when a type parameter is left unbound.
    fn instantiate_static_method(
        &mut self,
        found: FoundStaticMethod,
        resolved_name: &Symbol,
        name: &mut Symbol,
        call: &mut CallArgs<'_>,
    ) -> Type {
        let FoundStaticMethod {
            func: generic_func,
            generics: found_generics,
            mapping: mut found_mapping,
            instance_name,
        } = found;
        let (arg_types, span) = (call.types, call.span);
        // Whatever the explicit type arguments left open, the arguments may fix, the way a
        // generic free function's call does: `Holder::make(7i32)` binds `T` without `<i32>`.
        // Walk only as far as both lists reach; a count mismatch is reported below.
        for ((_, param_ty), arg_ty) in generic_func.params.iter().zip(arg_types.iter()) {
            self.unify_types(param_ty, arg_ty, &mut found_mapping);
        }
        // What the arguments leave open, the type declared for the result may still fix:
        // `let v : Vec<i32> = Vec::new();` binds `T` from the annotation, as a generic
        // free function's call already does.
        if let Some(expected) = self.expected_type.clone() {
            let _ = self.unify_types(&generic_func.return_type, &expected, &mut found_mapping);
        }
        if !self.require_type_params_bound(
            resolved_name.as_ref(),
            &found_generics,
            &found_mapping,
            span,
        ) {
            return Type::Unknown;
        }

        let mut modified_func = generic_func.clone();
        modified_func.name = instance_name.into();
        modified_func.generics = found_mapping
            .keys()
            .map(|k| decl::GenericParam::Type {
                name: k.clone(),
                bounds: Vec::new(),
            })
            .collect();

        let mut inst_func = self.instantiate_function(
            &modified_func,
            &found_mapping,
            &std::collections::HashMap::new(),
        );
        let inst_ret = inst_func.return_type.clone();
        let inst_name = inst_func.name.clone();
        let inst_params: Vec<Type> = inst_func.params.iter().map(|(_, t)| t.clone()).collect();

        *name = inst_name.clone();
        self.record_call_edge(inst_name.as_ref(), span);
        self.check_call_args(resolved_name.as_ref(), &inst_params, call);

        if !self.env.functions.contains_key(inst_name.as_ref())
            && !self.mono.functions.iter().any(|(f, _)| f.name == inst_name)
            && self.mono.being_checked.insert(inst_name.clone())
        {
            let saved_edge = self.seam.lowering_edge.take();
            self.check_instance(&generic_func, &mut inst_func);
            self.seam.lowering_edge = saved_edge;
            self.mono.being_checked.remove(&inst_name);
            self.mono.functions.push((inst_func, 0));
        }

        inst_ret
    }

    /// Makes the instance of a generic function that a call needs: works out the type each
    /// type parameter takes, checks the instance once, and rewrites `name` to the instance's
    /// name. Gives the instance's return type, or `None` after reporting an error. A chain of
    /// generic calls more than `MAX_INSTANTIATION_DEPTH` deep is stopped with E3032.
    pub(crate) fn instantiate_generic_function_call(
        &mut self,
        generic: &GenericCallee<'_>,
        resolved_name: &str,
        name: &mut Symbol,
        call: &CallArgs<'_>,
    ) -> Option<Type> {
        // A recursion through the generic arguments themselves makes a fresh instance every
        // time round, each checked inside the last. Stop, and say so, rather than running
        // out of stack.
        if self.mono.instantiation_depth >= crate::hir::check_state::MAX_INSTANTIATION_DEPTH {
            if !self.speculating {
                self.errors.error_with_code(
                    crate::diagnostic::DiagnosticCode::E3032,
                    format!(
                        "instantiating '{}' went more than {} generic calls deep and was \
                         stopped. A recursive generic whose base case is never reached is the \
                         usual cause.",
                        resolved_name,
                        crate::hir::check_state::MAX_INSTANTIATION_DEPTH
                    ),
                    Some(crate::diagnostic::SourceSpan::from_ast_span(call.span)),
                );
            }
            return None;
        }
        self.mono.instantiation_depth += 1;
        let result =
            self.instantiate_generic_function_call_inner(generic, resolved_name, name, call);
        self.mono.instantiation_depth -= 1;
        result
    }

    /// The work of `instantiate_generic_function_call`, without the check of how deep the
    /// chain of generic calls is.
    fn instantiate_generic_function_call_inner(
        &mut self,
        generic: &GenericCallee<'_>,
        resolved_name: &str,
        name: &mut Symbol,
        call: &CallArgs<'_>,
    ) -> Option<Type> {
        let generic_func = generic.func;
        let span = call.span;
        let mut mapping: std::collections::HashMap<crate::symbol::Symbol, Type> = HashMap::new();
        // Topology generic parameters (`<D: Topology>`): make `unify_types` bind a
        // `Pinned<_, D>` param's topology to the argument's concrete topology.
        self.mono.pending_topo_vars = generic_func
            .generics
            .iter()
            .filter_map(|g| match g {
                decl::GenericParam::Type { name, bounds }
                    if bounds.iter().any(|b| b.trait_name.as_ref() == "Topology") =>
                {
                    Some(name.clone())
                }
                _ => None,
            })
            .collect();
        self.mono.pending_topo_bindings.clear();
        let mut success = if call.exprs.len() == generic_func.params.len() {
            self.unify_generic_call_args(generic, name, call, &mut mapping)
        } else {
            if !self.speculating {
                self.errors.push(format!(
                    "Generic function '{}' expects {} arguments, got {}",
                    resolved_name,
                    generic_func.params.len(),
                    call.exprs.len()
                ));
            }
            false
        };

        if success
            && !self.require_type_params_bound(
                resolved_name,
                &generic_func.generics,
                &mapping,
                span,
            )
        {
            success = false;
        }

        if success {
            success = self.generic_bounds_hold(generic_func, &mapping);
        }

        if success {
            let params: Vec<&Type> = generic_func.params.iter().map(|(_, t)| t).collect();
            success = self.resolve_projections_for(
                resolved_name,
                &params,
                &generic_func.return_type,
                &mut mapping,
                Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
            );
        }

        if success {
            self.check_generic_instance(generic, name, &mapping, span)
        } else {
            None
        }
    }

    /// Works out the type each type parameter of a generic function takes at one call: first
    /// from the type arguments the call wrote out, then from the argument types, then from the
    /// type the call's result is expected to have. Fills `mapping` and gives `false` when an
    /// argument's type does not fit its parameter. `name` is the function's name as the call
    /// has it.
    fn unify_generic_call_args(
        &mut self,
        generic: &GenericCallee<'_>,
        name: &Symbol,
        call: &CallArgs<'_>,
        mapping: &mut HashMap<Symbol, Type>,
    ) -> bool {
        let generic_func = generic.func;
        let explicit_generic_args = generic.explicit_args;
        let arg_types = call.types;
        let mut success = true;
        for (i, param) in generic_func.generics.iter().enumerate() {
            if i < explicit_generic_args.len() {
                mapping.insert(param.name().into(), explicit_generic_args[i].clone());
            }
        }
        // Which argument first bound each generic name, so a later conflict can say where
        // the binding came from. Names the call spelled out explicitly stay out of the map,
        // because no argument bound them.
        let explicit: Vec<crate::symbol::Symbol> = mapping.keys().cloned().collect();
        let mut bound_by: HashMap<crate::symbol::Symbol, usize> = HashMap::new();
        for (i, _arg) in call.exprs.iter().enumerate() {
            let arg_ty = arg_types[i].clone();
            let param_ty = &generic_func.params[i].1;
            if self.unify_types(param_ty, &arg_ty, mapping) {
                for bound_name in mapping.keys() {
                    if !bound_by.contains_key(bound_name) && !explicit.contains(bound_name) {
                        bound_by.insert(bound_name.clone(), i + 1);
                    }
                }
            } else {
                if !self.speculating {
                    self.errors.push(format!(
                        "Failed to deduce types for generic function '{}': Expected {}, got {}{}{}",
                        name,
                        param_ty,
                        arg_ty,
                        Self::const_extent_note(param_ty, &arg_ty),
                        Self::dim_conflict_note(param_ty, &arg_ty, mapping, &bound_by, i + 1)
                    ));
                }
                success = false;
            }
        }
        // Deduce return-only variables (a topology `D` in `-> Pinned<T, D>`, or a
        // return-only type generic) from the call's expected type, if known. Best-effort:
        // the return-type unify binds topology variables into `pending_topo_bindings`
        // (via the `Pinned` arm) and type generics into `mapping`.
        if let Some(expected) = self.expected_type.clone() {
            let ret_ty = generic_func.return_type.clone();
            let _ = self.unify_types(&ret_ty, &expected, mapping);
        }
        success
    }

    /// Checks that the type each type parameter of `generic_func` takes implements every trait
    /// the parameter asks for. Reports each one that does not, and gives `false` if there is
    /// any.
    fn generic_bounds_hold(
        &mut self,
        generic_func: &Function,
        mapping: &HashMap<Symbol, Type>,
    ) -> bool {
        let mut success = true;
        for param in &generic_func.generics {
            let g_name = param.name();
            let bounds = match param {
                decl::GenericParam::Type { bounds, .. } => bounds.clone(),
                decl::GenericParam::Const { .. } => Vec::new(),
            };
            // Every bound has to hold, and every failing one is reported: told about
            // only the first, a caller fixes it and is handed the next.
            for bound in bounds {
                if let Some(concrete_ty) = mapping.get(g_name).cloned() {
                    let bound = self.substitute_bound(&bound, mapping);
                    if !self.bound_holds(&bound, &concrete_ty) {
                        if !self.speculating {
                            self.errors.push(format!(
                                "Type '{concrete_ty}' does not implement trait '{bound}' required by parameter '{g_name}'"
                            ));
                        }
                        success = false;
                    }
                }
            }
        }
        success
    }

    /// Checks each `where Reachable<A, B>` of `generic_func` now that its topology parameters
    /// are bound: a transfer path from A's memory to B's must exist. Reports the first one
    /// that has no path and gives `false`. `name` is the function's name as the call has it.
    fn where_transfers_hold(
        &mut self,
        generic_func: &Function,
        topo_mapping: &HashMap<Symbol, Topology>,
        name: &Symbol,
    ) -> bool {
        for (a, b) in &generic_func.where_transfers {
            if let (Some(ta), Some(tb)) = (topo_mapping.get(a), topo_mapping.get(b)) {
                let ma = self.transfer_cost_graph.default_memory_for(ta);
                let mb = self.transfer_cost_graph.default_memory_for(tb);
                if self.transfer_cost_graph.transfer_path(&ma, &mb).is_none() {
                    if !self.speculating {
                        self.errors.push(format!(
                            "unsatisfied `where Reachable<{a}, {b}>` in call to '{name}': no transfer \
                             path from {ta:?} to {tb:?}"
                        ));
                    }
                    return false;
                }
            }
        }
        true
    }

    /// Makes the instance of a generic function for the types in `mapping`, rewrites `name` to
    /// the instance's name, and checks the instance unless that was done before. Gives the
    /// instance's return type, or `None` when a `where Reachable<..>` of the function does not
    /// hold.
    fn check_generic_instance(
        &mut self,
        generic: &GenericCallee<'_>,
        name: &mut Symbol,
        mapping: &HashMap<Symbol, Type>,
        span: &Span,
    ) -> Option<Type> {
        let generic_func = generic.func;
        let topo_mapping = std::mem::take(&mut self.mono.pending_topo_bindings);
        self.mono.pending_topo_vars.clear();

        // Discharge `where Reachable<A, B>` now that the topology variables are bound:
        // a transfer path from A's memory to B's must exist in the cost graph.
        if !self.where_transfers_hold(generic_func, &topo_mapping, name) {
            return None;
        }

        let mut inst_func = self.instantiate_function(generic_func, mapping, &topo_mapping);
        let inst_ret = inst_func.return_type.clone();
        let inst_name = inst_func.name.clone();

        *name = inst_name.clone();
        self.record_call_edge(inst_name.as_ref(), span);

        // The instance carries the placement, not the template: a generic bound to a device is
        // a device function for every argument it is instantiated with. The two concrete call
        // paths check this and this one must too -- the call site names the template, so it
        // never reaches the `mono.functions` arm that would have caught it, and without this
        // an `on Topology::..` on a generic was accepted from anywhere.
        self.check_call_topology(generic_func.name.as_ref(), &inst_func.topology, span);

        if !self.env.functions.contains_key(inst_name.as_ref())
            && !self.mono.functions.iter().any(|(f, _)| f.name == inst_name)
            && self.mono.being_checked.insert(inst_name.clone())
        {
            // `check_function` checks the body in a borrow table of its own (#268).
            // An instantiated generic is ordinary code even when the call site sits in
            // a transfer lowering; the raw:: primitives must not resolve inside it.
            let saved_edge = self.seam.lowering_edge.take();
            self.check_instance(generic_func, &mut inst_func);
            self.seam.lowering_edge = saved_edge;
            self.mono.being_checked.remove(&inst_name);
            self.mono.functions.push((inst_func, generic.origin_hash));
        }
        Some(inst_ret)
    }

    fn deref_number_operand_unless_borrow(arg: &mut Expr, ty: &Type) -> Option<Type> {
        if matches!(arg, Expr::Borrow(_)) {
            return None;
        }
        Self::deref_number_operand(arg, ty)
    }

    /// Get a `print` operand ready for code generation.
    ///
    /// A reference to a number prints the number, as in Rust: `x` becomes `*x`. A written
    /// `&x` is left alone, since `print` already looks through it.
    ///
    /// MLIR integers carry no sign, so the AST code generator cannot tell a `u64` from an
    /// `i64` when it picks the runtime print function. An unsigned operand is wrapped in a
    /// cast to its own type (`x as u64`), which changes no value and tells it which to pick.
    pub(crate) fn prepare_print_operand(arg: &mut Expr, ty: &Type) {
        let read_through;
        let ty = match Self::deref_number_operand_unless_borrow(arg, ty) {
            Some(t) => {
                read_through = t;
                &read_through
            }
            None => ty,
        };
        let (arg, ty) = match (arg, ty) {
            (Expr::Borrow(b), Type::Borrow { inner, .. }) => (&mut *b.expr, &**inner),
            (arg, ty) => (arg, ty),
        };
        // A tensor of `u32` or `u64` is marked the same way, for its runtime printer.
        let unsigned = match ty {
            Type::Scalar(e) => matches!(
                e,
                ElementType::U8 | ElementType::U16 | ElementType::U32 | ElementType::U64
            ),
            Type::Tensor(e, _, _) => matches!(e, ElementType::U32 | ElementType::U64),
            _ => false,
        };
        if !unsigned {
            return;
        }
        if matches!(arg, Expr::AsCast(c) if c.target_ty == *ty) {
            return;
        }
        let span = arg.span();
        *arg = Expr::AsCast(crate::syntax::expr::AsCastExpr {
            expr: Box::new(arg.clone()),
            target_ty: ty.clone(),
            source_ty: Some(ty.clone()),
            span,
        });
    }

    /// `print` reads its operand where the program is running, so a value the active topology
    /// cannot see is refused -- the same question `o[0][0]` is refused on.
    ///
    /// It used to be asked only of an identifier, so `print(o)` on a binding was refused while
    /// `print(mk())` on a call returning the same placed type was accepted. That one lowers to
    /// MLIR's `printMemrefF32`, which walks the memref on the host: with a device present the
    /// data pointer is device memory and the walk faults.
    fn check_print_operand_is_visible(&mut self, ty: Option<&Type>, span: crate::syntax::Span) {
        if self.speculating || self.allow_cross_topology {
            return;
        }
        let Some(ty) = ty else { return };
        // A reference to a placed tensor is read where the tensor is.
        let ty = match ty {
            Type::Borrow { inner, .. } => inner.as_ref(),
            other => other,
        };
        let Some(p) = ty.placement() else { return };
        let owner = self.transfer_cost_graph.placement_topology(p);
        let src = self.transfer_cost_graph.placement_space(p);
        // Same admission the identifier rule applies: a space declared `managed: cached` is
        // hardware-coherent, so reading it across the boundary is legal when a path exists. Only
        // an `explicit` space obliges a transfer -- refusing a cached one here would refuse what
        // the model permits everywhere else.
        let reach = self
            .transfer_cost_graph
            .reachable(&self.active_topology, &owner, ty);
        if matches!(reach, crate::arch::Reachability::Visible)
            || (matches!(reach, crate::arch::Reachability::NeedsSeam { .. })
                && self.space_is_cached(&src))
        {
            return;
        }
        let dst = self
            .transfer_cost_graph
            .default_memory_for(&self.active_topology);
        let visible = self
            .transfer_cost_graph
            .descriptor(&self.active_topology.kind())
            .map(|d| {
                d.visibility
                    .iter()
                    .map(crate::syntax::types::MemorySpace::name)
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .unwrap_or_default();
        self.errors.error_with_code(
            crate::diagnostic::DiagnosticCode::E6003,
            format!(
                "`print` reads its argument on {}, which sees only [{}], but the value \
                 lives in {}; bring it home first with `transfer(.., Memory::{})`",
                self.active_topology.display_name(),
                visible,
                src.name(),
                dst.name()
            ),
            Some(crate::diagnostic::SourceSpan::from_ast_span(&span)),
        );
    }

    /// Checks a call to a function the compiler provides itself, such as `print`, `panic`, a
    /// `Tensor` constructor or a `raw::` primitive, and gives the call's type. Gives `None`
    /// when `resolved_name` is not one of them.
    pub(crate) fn resolve_intrinsic_function(
        &mut self,
        resolved_name: &str,
        call: &CallArgs<'_>,
        explicit_generic_args: &[Type],
    ) -> Option<Type> {
        let (args, arg_types, call_span) = (&*call.exprs, call.types, *call.span);
        // The `raw::` prefix is reserved for the transfer-lowering primitives (Vx#353).
        // Intercepted before every other lookup so the namespace cannot be shadowed by user
        // code, and so a use outside an `impl transfer` body gets a targeted error instead
        // of "Undefined static method".
        if let Some(prim) = resolved_name.strip_prefix("raw::") {
            return Some(self.check_raw_primitive(prim, args, arg_types, call_span));
        }
        // `Tensor<T, [..], P>::uninit()` and the run-time-extent form: the type is written, so it is
        // read off the call rather than rebuilt from the call's name and arguments. That
        // reconstruction is what the older spelling forces on three separate places, and it cannot
        // carry a placement at all -- there is nowhere in `Tensor<f32>([4, 4])` to put one.
        if matches!(
            resolved_name,
            "Tensor::new" | "Tensor::uninit" | "Tensor::fill"
        ) {
            return Some(self.check_tensor_constructor(resolved_name, call, explicit_generic_args));
        }
        if resolved_name == "Verified" {
            if args.len() != 1 {
                self.errors.push(format!(
                    "Function 'Verified' expects 1 argument, got {}",
                    args.len()
                ));
            }
            let inner_ty = arg_types[0].clone();
            Some(Type::Verified(Box::new(inner_ty)))
        // The older spelling, `Tensor<f32>([2, 3])`, names the constructor by its prefix. A name
        // holding `::` is a path -- the four constructors above are matched exactly, so anything
        // else is a method nothing defines, and reporting it undefined beats building a tensor
        // out of whatever its arguments were.
        } else if resolved_name.starts_with("Tensor")
            && !resolved_name.contains("::")
            && !resolved_name.contains('$')
            && !resolved_name.contains("__")
        {
            Some(self.check_old_tensor_constructor(args, explicit_generic_args))
        } else if resolved_name.starts_with("Math::") {
            if args.len() != 1 {
                self.errors.push(format!(
                    "Function '{}' expects 1 argument, got {}",
                    resolved_name,
                    args.len()
                ));
            }
            let inner_ty = arg_types[0].clone();
            if inner_ty != Type::Scalar(ElementType::F32) {
                self.errors.push(format!(
                    "Function '{resolved_name}' expects f32 argument, got {inner_ty:?}"
                ));
            }
            Some(Type::Scalar(ElementType::F32))
        } else {
            self.resolve_builtin_function(resolved_name, call)
        }
    }

    /// Checks a call to `Tensor<..>::new`, `::uninit` or `::fill`, whose first type argument is
    /// the tensor type to build, and gives that type.
    fn check_tensor_constructor(
        &mut self,
        resolved_name: &str,
        call: &CallArgs<'_>,
        explicit_generic_args: &[Type],
    ) -> Type {
        let (args, arg_types, call_span) = (&*call.exprs, call.types, *call.span);
        let Some(ty) = explicit_generic_args.first() else {
            self.errors
                .push(format!("'{resolved_name}' has no type to build"));
            return Type::Unknown;
        };
        // A static shape lives in the type and a dynamic one in the argument, so each form
        // takes exactly the arguments the other cannot. `::fill(v)` adds one on top: the
        // value, which no other constructor takes.
        let fills = resolved_name == "Tensor::fill";
        let dynamic =
            matches!(ty, Type::Tensor(_, dims, _) if dims.iter().any(|d| matches!(d, Dim::Dyn)));
        let wanted = usize::from(dynamic) + usize::from(fills);
        if args.len() != wanted {
            let why = if fills {
                "`::fill` takes the value to write into every element"
            } else if wanted == 0 {
                "a Tensor's shape is part of its type, so the constructor takes no arguments"
            } else {
                "a run-time extent is not in the type, so the constructor takes the shape: \
                 `::new([n, m])`"
            };
            self.errors.push(format!(
                "'{resolved_name}' expects {wanted} argument(s): {why}"
            ));
        }
        // `unsafe` is where a program says it writes every element of `::uninit()` storage
        // before reading it.
        if resolved_name == "Tensor::uninit" && !self.in_unsafe_block && !self.speculating {
            self.errors.warn(
                crate::diagnostic::DiagnosticCode::W1032,
                "`::uninit()` storage is not written: reading an element before writing it \
                 reads whatever the memory held. Call it inside `unsafe { .. }` to say every \
                 element is written first, or use `::new()` for zeroed storage. This will \
                 become an error",
                Some(crate::diagnostic::SourceSpan::from_ast_span(&call_span)),
            );
        }
        // The fill value has the tensor's element type: a number literal was given it when
        // the arguments were checked, and a typed value needs `as`.
        if let (true, Type::Tensor(elem, _, _), Some(Type::Scalar(got)), Some(value)) =
            (fills, ty, arg_types.last(), args.last())
        {
            if got != elem {
                self.errors.error_with_code(
                    crate::diagnostic::DiagnosticCode::E3003,
                    format!(
                        "'{resolved_name}' fills a tensor of {elem}, but the value is {got}; \
                         convert it with `as {elem}`"
                    ),
                    Some(crate::diagnostic::SourceSpan::from_ast_span(&value.span())),
                );
            }
        }
        if let (false, Type::Tensor(_, dims, _)) = (dynamic, ty) {
            self.check_static_extents_known(dims);
        }
        ty.clone()
    }

    /// Reports a tensor type whose shape has a size the compiler cannot work out. `dims` is
    /// the shape of a type that was written with every size given.
    fn check_static_extents_known(&mut self, dims: &[Dim]) {
        // A `Tensor`'s extents are part of its type, so an extent nothing can evaluate is not
        // a static extent (Vx#399). Refused rather than quietly answered with a `?`:
        // the type was written down, and handing back a different one is how a shape nobody
        // can read comes to be trusted by a later check. A const generic does evaluate.
        let mut env = HashMap::new();
        for scope in &self.consteval.env {
            for (k, v) in scope {
                env.insert(k.clone(), v.clone());
            }
        }
        if !dims.iter().all(|d| {
            d.as_static()
                .is_some_and(|e| self.eval_expr(e, &env).is_some())
        }) {
            self.errors.push(
                "a Tensor's shape is part of its type, so every extent has to be known \
                 at compile time; write `Tensor<T, [?, ?]>::uninit([..])` for a shape \
                 that is not"
                    .to_string(),
            );
        }
    }

    /// Checks a tensor built with the older spelling, `Tensor<f32>([2, 3])`, where the shape
    /// is an argument, and gives the tensor's type.
    fn check_old_tensor_constructor(
        &mut self,
        args: &[Expr],
        explicit_generic_args: &[Type],
    ) -> Type {
        let el_ty = if explicit_generic_args.is_empty() {
            self.errors
                .push("Missing generic argument for Tensor initialization.".to_string());
            ElementType::F32
        } else if let Type::Scalar(el) = &explicit_generic_args[0] {
            el.clone()
        } else {
            self.errors
                .push("Generic argument to Tensor must be a scalar type.".to_string());
            ElementType::F32
        };
        let mut dims = Vec::new();
        if !args.is_empty() {
            if let Expr::Array(arr) = &args[0] {
                if let Some(shape) = arr.initializer_shape() {
                    // Initializer list `Tensor<T>([[..],[..]])`: shape from the nesting.
                    dims = shape.into_iter().map(int_dim_expr).collect();
                } else {
                    dims.clone_from(&arr.elements);
                }
            } else {
                dims = args.to_vec();
            }
        }
        // A dimension the compiler cannot evaluate is a run-time value, so that position is
        // `?`: recording `[n, 3]` as static would let later checks read an extent that does
        // not exist yet, which is what W1029 warns about. A const-generic dimension does
        // evaluate, so `Tensor<f32, [N, N]>` stays statically shaped. (Vx#399)
        let mut env = HashMap::new();
        for scope in &self.consteval.env {
            for (k, v) in scope {
                env.insert(k.clone(), v.clone());
            }
        }
        let dims = dims
            .into_iter()
            .map(|d| {
                if self.eval_expr(&d, &env).is_some() {
                    Dim::Static(d)
                } else {
                    Dim::Dyn
                }
            })
            .collect();
        Type::Tensor(el_ty, dims, None)
    }

    /// Checks a call to one of the small built-in functions: `abort`, `panic`, `print`,
    /// `printf`, `Some`, `matmul_into`, `tensor_view_2d` and `barrier`. Gives the call's type,
    /// or `None` when `resolved_name` is not one of them.
    fn resolve_builtin_function(
        &mut self,
        resolved_name: &str,
        call: &CallArgs<'_>,
    ) -> Option<Type> {
        let (args, arg_types, call_span) = (&*call.exprs, call.types, *call.span);
        if resolved_name == "abort" {
            // Terminating the program is SAFE: it ends the process rather than violating any
            // memory-safety property (the reason `std::process::abort` needs no `unsafe` in
            // Rust). So it is a builtin here, not an `extern` -- routing it through an FFI
            // declaration would drag in E5001 for a construct that earns none of it.
            if !args.is_empty() {
                self.errors
                    .push("Function 'abort' expects no arguments".to_string());
            }
            Some(crate::syntax::never_ty())
        } else if resolved_name == "panic" {
            // `abort()` with a message. The message is a C string, so it can be a parameter that
            // a caller filled in, as `expect(msg)` needs, not only a literal.
            let c_string = Type::Pointer(Box::new(Type::Scalar(ElementType::I8)), None, false);
            if args.len() != 1 {
                self.errors
                    .push("Function 'panic' expects 1 argument, the message".to_string());
            } else if arg_types[0] != c_string {
                self.errors.push(format!(
                    "Function 'panic' expects a string message, got {}",
                    arg_types[0]
                ));
            }
            Some(crate::syntax::never_ty())
        } else if resolved_name == "print" {
            if args.len() != 1 {
                self.errors
                    .push("Function 'print' expects 1 argument".to_string());
            }
            self.check_print_operand_is_visible(arg_types.first(), call_span);
            Some(Type::Struct("void".into(), None))
        } else if resolved_name == "printf" || resolved_name == "vx_internal_printf" {
            if args.is_empty() {
                self.errors
                    .push("Function 'printf' expects at least 1 argument".to_string());
            }
            Some(Type::Scalar(ElementType::I32))
        } else if resolved_name == "Some" || resolved_name == "Option::Some" {
            if args.len() != 1 {
                self.errors
                    .push(format!("Function '{resolved_name}' expects 1 argument"));
            }
            Some(Type::Struct("Option".into(), None))
        } else if resolved_name == "matmul_into" {
            Some(self.check_matmul_into(args, arg_types))
        } else if resolved_name == "tensor_view_2d" {
            Some(self.check_tensor_view_2d(args, arg_types))
        } else if resolved_name == "barrier" {
            // Block-level synchronization inside a spawn region (Vx#379): all threads of a
            // block reach it before any proceeds. Serial execution already provides that
            // ordering, so on the host path it is a no-op; the device pipeline rewrites it to
            // `gpu.barrier`. Takes nothing, returns nothing worth keeping.
            if !args.is_empty() {
                self.errors
                    .push("Function 'barrier' takes no arguments".to_string());
            }
            Some(Type::Scalar(ElementType::I32))
        } else {
            None
        }
    }

    /// Checks a call to `matmul_into(&mut dst, &a, &b)`, which writes the matrix product of
    /// `a` and `b` into `dst`, and gives the call's type.
    fn check_matmul_into(&mut self, args: &[Expr], arg_types: &[Type]) -> Type {
        // `dst = &a @ &b` rebinds dst to a fresh allocation, so a view over
        // a caller's buffer is left untouched -- and a model's
        // `matmul(xout, ...)` exists to fill exactly such a buffer. This
        // writes in place instead, and allocates nothing.
        if args.len() != 3 {
            self.errors
                .push("Function 'matmul_into' expects (&mut dst, &a, &b)".to_string());
        }
        match arg_types.first() {
            Some(Type::Borrow { is_mut: true, .. }) => {}
            _ => {
                if !self.speculating {
                    self.errors.push(
                        "matmul_into writes its first argument, which must be a mutable borrow (&mut dst)"
                            .to_string(),
                    );
                }
            }
        }
        // A view over a caller's buffer has run-time extents, and filling exactly such a buffer is
        // what this exists for -- so the operands need to be tensors, not statically shaped
        // ones (Vx#399).
        for t in arg_types.iter().take(3) {
            if Self::tensor_operand_elem(t).is_none() && !self.speculating {
                self.errors
                    .push(format!("Function 'matmul_into' expects tensors, got {t:?}"));
            }
        }
        // `void` is spelled as a struct name in this type system.
        Type::Struct("void".into(), None)
    }

    /// Checks a call to `tensor_view_2d(ptr, rows, cols)`, which gives a rank-2 tensor over
    /// memory the tensor does not own, and gives the tensor's type.
    fn check_tensor_view_2d(&mut self, args: &[Expr], arg_types: &[Type]) -> Type {
        // A rank-2 tensor over memory it does not own, with no copy (#336).
        // The element type comes from the pointer, so the view cannot
        // reinterpret f32 storage as f64 by accident.
        //
        // Unsafe because nothing checks that the extents describe the
        // memory truthfully -- the same claim `*mut T` indexing already
        // makes, and the reason the view exists is to wrap foreign buffers.
        if args.len() != 3 {
            self.errors
                .push("Function 'tensor_view_2d' expects (ptr, rows, cols)".to_string());
        }
        if !self.in_unsafe_block && !self.speculating {
            self.errors
                .push("Viewing raw memory as a tensor requires an unsafe block".to_string());
        }
        let elem = match arg_types.first() {
            Some(Type::Pointer(inner, _, _)) => match inner.as_ref() {
                Type::Scalar(e) => Some(e.clone()),
                _ => None,
            },
            _ => None,
        };
        let Some(e) = elem else {
            if !self.speculating {
                self.errors.push(format!(
                    "Function 'tensor_view_2d' expects a pointer to a scalar, got {:?}",
                    arg_types.first()
                ));
            }
            return Type::Unknown;
        };
        // The extents are ordinary arguments, so the same rule the constructor gets
        // applies: a view whose rows and cols evaluate at compile time is statically
        // shaped, and one over run-time extents has `?` there (Vx#399).
        let mut env = HashMap::new();
        for scope in &self.consteval.env {
            for (k, v) in scope {
                env.insert(k.clone(), v.clone());
            }
        }
        let dims = match args.get(1..3) {
            Some(ext) if ext.len() == 2 => ext
                .iter()
                .map(|a| {
                    if self.eval_expr(a, &env).is_some() {
                        Dim::Static(a.clone())
                    } else {
                        Dim::Dyn
                    }
                })
                .collect(),
            _ => vec![Dim::Dyn, Dim::Dyn],
        };
        Type::Tensor(e, dims, None)
    }

    /// Resolve `base_ty`'s method `method` by walking every impl block (unifying the receiver,
    /// peeling references), filling `mapping` with the deduced impl-level generic bindings. Returns
    /// the matching method + its impl block, or `None`. In debug builds also asserts the frozen
    /// registry's `ModuleInterface` resolves the same concrete `(receiver GID, method)` — the #219
    /// keep-green parity gate. Split out of `check_methodcall_expr` (R4, #279).
    pub(crate) fn resolve_method_in_impls(
        &mut self,
        base_ty: &Type,
        method: &crate::symbol::Symbol,
        mapping: &mut std::collections::HashMap<crate::symbol::Symbol, Type>,
    ) -> Option<(Function, decl::ImplBlock)> {
        let mut check_ty = base_ty.clone();
        while let Type::Borrow { inner, .. } | Type::Pointer(inner, _, _) | Type::Ref(inner, _) =
            &check_ty
        {
            check_ty = *inner.clone();
        }

        // Every impl block the receiver unifies with that has the method, with the mapping
        // that unification produced. An inherent method shadows a trait's, as in Rust; two
        // traits' is refused rather than chosen, because the blocks live in a hash map and
        // which body a call reached used to change from one run of the compiler to the next.
        let mut inherent = None;
        let mut from_traits: Vec<TraitMethod> = Vec::new();
        for (trait_key, impl_blocks) in &self.env.impls {
            for ib in impl_blocks {
                let mut candidate_mapping = HashMap::new();
                if !self.impl_applies(ib, &check_ty, &mut candidate_mapping) {
                    continue;
                }
                let Some(m) = ib.methods.iter().find(|m| m.name == *method) else {
                    continue;
                };
                if trait_key.as_ref() == "_inherent" {
                    if inherent.is_none() {
                        inherent = Some((m.clone(), (*ib).clone(), candidate_mapping));
                    }
                } else {
                    from_traits.push((
                        trait_key.clone(),
                        m.clone(),
                        (*ib).clone(),
                        candidate_mapping,
                    ));
                }
            }
        }
        let found_method = if let Some((m, ib, chosen_mapping)) = inherent {
            *mapping = chosen_mapping;
            Some((m, ib))
        } else if from_traits.len() > 1 {
            self.report_ambiguous_method(method, &check_ty, &from_traits);
            mapping.clear();
            None
        } else if let Some((_, m, ib, chosen_mapping)) = from_traits.pop() {
            *mapping = chosen_mapping;
            Some((m, ib))
        } else {
            mapping.clear();
            None
        };

        #[cfg(debug_assertions)]
        if let Some((ref m, ref ib)) = found_method {
            self.debug_check_registry_has_method(base_ty, m, ib);
        }
        found_method
    }

    /// Reports a method call that the impls of two or more traits could mean (E3035).
    /// `check_ty` is the receiver's type and `from_traits` the methods found.
    fn report_ambiguous_method(
        &mut self,
        method: &Symbol,
        check_ty: &Type,
        from_traits: &[TraitMethod],
    ) {
        if !self.speculating {
            let mut names: Vec<String> = from_traits
                .iter()
                .map(|(t, _, ib, _)| impl_display(t, ib))
                .collect();
            names.sort();
            let mut traits: Vec<&str> = from_traits.iter().map(|(t, _, _, _)| t.as_ref()).collect();
            traits.sort_unstable();
            self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E3035,
                format!(
                    "Method '{}' on '{}' is defined by more than one impl ({}); name the \
                     trait to choose one, as in '{}::{}(..)'",
                    method,
                    check_ty,
                    names.join(", "),
                    traits[0],
                    method
                ),
                None,
            );
        }
    }

    /// Debug builds only: checks that the table of methods built from imported libraries has
    /// the method `m` of the impl block `ib` that the search of the impl blocks found for a
    /// receiver of type `base_ty`.
    #[cfg(debug_assertions)]
    fn debug_check_registry_has_method(&self, base_ty: &Type, m: &Function, ib: &decl::ImplBlock) {
        // Dual-run parity gate for the stdlib<->compiler decoupling (#219): when the AST
        // impl-walk resolves a method on a *concrete* receiver via a non-generic impl,
        // the registry-backed `ModuleInterface` must resolve the same `(receiver GID, method)`.
        // This proves the frozen registry is a sufficient oracle at real resolution sites --
        // the keep-green gate before imported-symbol resolution stops consulting the borrowed
        // AST env. Generic impls and non-nominal / generic receivers are outside the registry
        // method table's scope (#218), so they are skipped rather than asserted. The gate only
        // runs when a frozen registry is actually in use: an *empty* method table means this
        // compilation never built one (the sequential driver / legacy AST-only harnesses use
        // `GlobalSession::new`), so there is nothing to dual-run against.
        if !self.worker.global.registry.methods.is_empty()
            && ib.generics.is_empty()
            && m.generics.is_empty()
        {
            let mut recv = base_ty.clone();
            while let Type::Borrow { inner, .. }
            | Type::Pointer(inner, _, _)
            | Type::Ref(inner, _) = &recv
            {
                recv = (**inner).clone();
            }
            let recv_gid = match &recv {
                Type::Scalar(ElementType::Generic(_)) => None,
                Type::Scalar(e) => Some(crate::hir::flatten::scalar_gid(e)),
                Type::Struct(_, Some(id)) | Type::Enum(_, Some(id)) => Some(*id),
                _ => None,
            };
            if let Some(gid) = recv_gid {
                let mi: &dyn crate::registry::ModuleInterface = &*self.worker.global.registry;
                debug_assert!(
                    mi.resolve_method(gid, &m.name).is_some(),
                    "ModuleInterface missing a method the AST resolved: {}.{}",
                    recv.mangle(),
                    m.name
                );
            }
        }
    }

    /// Names a generic function's body by the function's name and where the body starts. Two
    /// generic methods can share a name, such as two types' `new`, but not a body.
    fn generic_body_key(template: &Function) -> String {
        format!(
            "{}@{:?}",
            template.name,
            template
                .body
                .first()
                .map(crate::syntax::stmt::Statement::span)
        )
    }

    /// Checks one instance of the generic function `template`. Every instance checks the same
    /// body, so a warning an earlier instance already gave is not given again.
    fn check_instance(&mut self, template: &Function, inst: &mut Function) {
        let start = self.errors.len();
        self.check_function(inst);
        let given = self
            .mono
            .warnings_given
            .entry(Self::generic_body_key(template))
            .or_default();
        self.errors.drop_warnings_already_given(start, given);
    }

    /// `generic_method` of `ib`, instantiated at `mapping` for the receiver type `base_ty`,
    /// type-checked and registered unless it already is. Returns it, renamed to its mangled
    /// name, and that name.
    pub(crate) fn instantiate_impl_method(
        &mut self,
        generic_method: &Function,
        mapping: &std::collections::HashMap<crate::symbol::Symbol, Type>,
        base_ty: &Type,
        ib: &decl::ImplBlock,
    ) -> (Function, String) {
        // Provide generic mapping to the method itself by copying impl block generics
        let mut modified_func = generic_method.clone();
        modified_func.generics = mapping
            .keys()
            .map(|k| decl::GenericParam::Type {
                name: k.clone(),
                bounds: Vec::new(),
            })
            .collect();
        let mut method_func =
            self.instantiate_function(&modified_func, mapping, &std::collections::HashMap::new());

        // The name the definition was minted under: receiver, then the trait and its arguments
        // when the method came from a trait impl. Shared with the definition site, so a call and
        // the body it resolves to cannot drift apart.
        let mangled_name = crate::syntax::types::mangle_method(
            base_ty,
            ib.trait_name.as_ref(),
            &ib.trait_args,
            method_func.name.as_ref(),
        );

        method_func.name = mangled_name.clone().into();

        let mangled_sym = crate::symbol::Symbol::from(mangled_name.as_str());
        if !self.env.functions.contains_key(&*mangled_name)
            && !self
                .mono
                .functions
                .iter()
                .any(|(f, _)| f.name == mangled_sym)
            && self.mono.being_checked.insert(mangled_sym.clone())
        {
            // Type check the instantiated method
            let mut func_to_check = method_func.clone();
            let saved_edge = self.seam.lowering_edge.take();
            // The body's own `let`s overwrite it, and the caller's `let` still needs it.
            let saved_target = self.current_assignment_target.take();
            self.check_instance(generic_method, &mut func_to_check);
            self.current_assignment_target = saved_target;
            self.seam.lowering_edge = saved_edge;
            self.mono.being_checked.remove(&mangled_sym);
            self.mono.functions.push((func_to_check, 0)); // 0 will fall back to caller_module_idx
        }
        (method_func, mangled_name)
    }

    /// Instantiate the resolved generic method, register + type-check the monomorphized function,
    /// and build the `FunctionCall` node the method call rewrites to (prepending the receiver,
    /// borrowed to match a `&self`/`&mut self` first parameter). Probes the synthesized call
    /// speculatively (R3) to recover its return type without double-reporting. Returns `(return
    /// type, replacement node)`; the caller performs the `*expr = …` rewrite. Split out of
    /// `check_methodcall_expr` (R4, #279).
    fn instantiate_method_call_rewrite(
        &mut self,
        generic_method: &Function,
        mut mapping: std::collections::HashMap<crate::symbol::Symbol, Type>,
        ib: &decl::ImplBlock,
        site: &MethodCallSite<'_>,
    ) -> (Type, Expr) {
        let (base_ty, checked_arg_types, span) = (site.base_ty, site.arg_types, site.span);
        self.deduce_method_generics(generic_method, &mut mapping, checked_arg_types);

        let (method_func, mangled_name) =
            self.instantiate_impl_method(generic_method, &mapping, base_ty, ib);

        // Rewrite AST from MethodCall to FunctionCall
        let mut call_args = vec![self.receiver_argument(generic_method, &method_func, site)];

        for a in &*site.args {
            call_args.push(a.clone());
        }

        // The arguments after the receiver, against the instance's parameters. The call below
        // is only probed speculatively, which reports nothing, so this is where they are asked.
        let params: Vec<Type> = method_func
            .params
            .iter()
            .skip(1)
            .map(|(_, t)| t.clone())
            .collect();
        let callee = format!("{}::{}", base_ty, generic_method.name);
        let mut call = CallArgs {
            exprs: &mut call_args[1..],
            types: checked_arg_types,
            span: &span,
        };
        self.check_call_args(&callee, &params, &mut call);

        let mut func_call = Expr::FunctionCall(FunctionCallExpr {
            name: crate::symbol::Symbol::from(mangled_name.as_str()),
            type_args: None,
            args: call_args,
            span,
        });
        // Probe the synthesized call *speculatively* to recover its return type without
        // emitting diagnostics or committing borrow/move side effects: the method-call
        // node is only now being rewritten into this call, so a second (real) check
        // would double-report. This is the sole site that turns `speculating` on; the
        // "fresh check" entry points (`check_expr_type`, `check_block`) force it back off
        // for independent subtrees. Replaces the old `silent = true` argument (#279 R3).
        let saved_speculating = self.speculating;
        self.speculating = true;
        let ret_ty = self.check_expr_type_flag(&mut func_call, site.consume);
        self.speculating = saved_speculating;
        // A call to a comptime method has to fold. The probe tried with diagnostics off, so
        // try again here, where a call that cannot fold is reported.
        let comptime_callee = matches!(&func_call, Expr::FunctionCall(call)
            if self.callee_body(call.name.as_ref()).is_some_and(Self::is_comptime_fn));
        if comptime_callee {
            self.fold_comptime_fn_call(&mut func_call, &span);
        }
        // A method calling itself is not registered until its body is checked, so the probe
        // cannot find it. Its signature says what it returns.
        let ret_ty =
            if ret_ty == Type::Unknown && self.mono.being_checked.contains(mangled_name.as_str()) {
                method_func.return_type.clone()
            } else {
                ret_ty
            };
        (ret_ty, func_call)
    }

    /// Works out the type parameters of a method that the receiver did not bind: from the
    /// types of the arguments, then from the type the call's result is expected to have.
    /// Fills `mapping`.
    fn deduce_method_generics(
        &mut self,
        generic_method: &Function,
        mapping: &mut HashMap<Symbol, Type>,
        checked_arg_types: &[Type],
    ) {
        // Infer method-level generics from argument types. Reuse the types from the
        // single check above: re-checking here would re-consume linear args (a closure
        // struct passed to `.map`) and yield `Unknown`, defeating the deduction.
        for (i, arg_ty) in checked_arg_types.iter().enumerate() {
            if i + 1 < generic_method.params.len() {
                let expected_param = &generic_method.params[i + 1].1;
                self.unify_types(expected_param, arg_ty, mapping);
            }
        }
        // A method parameter only the return type names, `collect<B>`, from what the call is
        // assigned to, as a generic function's is.
        if let Some(expected) = self.expected_type.clone() {
            self.unify_types(&generic_method.return_type, &expected, mapping);
        }
        let params: Vec<&Type> = generic_method.params.iter().map(|(_, t)| t).collect();
        self.resolve_projections_for(
            generic_method.name.as_ref(),
            &params,
            &generic_method.return_type,
            mapping,
            None,
        );
    }

    /// The first argument of the function call a method call is rewritten to: the receiver,
    /// borrowed when the method takes `&self` or `&mut self` and the receiver is not a
    /// reference already. Reports a `&mut self` method called on a receiver that cannot be
    /// changed. `method_func` is the instance of `generic_method` the call reaches.
    fn receiver_argument(
        &mut self,
        generic_method: &Function,
        method_func: &Function,
        site: &MethodCallSite<'_>,
    ) -> Expr {
        let (obj, base_ty, span) = (site.obj, site.base_ty, site.span);
        let Some(first_param) = method_func.params.first() else {
            return obj.clone();
        };
        let param_is_ref = matches!(first_param.1, Type::Borrow { .. } | Type::Pointer(_, _, _));
        let is_mut = match &first_param.1 {
            Type::Borrow { is_mut: m, .. } | Type::Pointer(_, _, m) => *m,
            _ => false,
        };

        let obj_is_ref = matches!(base_ty, Type::Borrow { .. } | Type::Pointer(_, _, _));
        if param_is_ref && is_mut {
            let method = generic_method.name.to_string();
            if !obj_is_ref {
                if let Some(root) = self.read_only_root(obj) {
                    let what = format!("`{method}`, which takes `&mut self`, cannot change it");
                    self.report_read_only(&root, &what, &span);
                }
            } else if matches!(base_ty, Type::Borrow { is_mut: false, .. }) && !self.speculating {
                self.errors.error_with_code(
                    crate::diagnostic::DiagnosticCode::E4006,
                    format!(
                        "`{method}` takes `&mut self`, and this calls it through a shared \
                         reference (`&`), which only allows reading"
                    ),
                    Some(crate::diagnostic::SourceSpan::from_ast_span(&span)),
                );
            }
        }

        if param_is_ref && !obj_is_ref {
            Expr::Borrow(BorrowExpr {
                expr: Box::new(obj.clone()),
                is_mut,
                span: Span::default(),
            })
        } else {
            obj.clone()
        }
    }

    /// Checks a method call `obj.method(args)` and gives the type of its result. A call of a
    /// method from an impl block is rewritten to a call of the function made for that method,
    /// with the receiver as its first argument. A few built-in methods are rewritten to the
    /// expression they stand for.
    ///
    /// # Panics
    ///
    /// Panics if `expr` is not a method call.
    pub(crate) fn check_methodcall_expr(&mut self, expr: &mut Expr, consume: bool) -> Type {
        let Expr::MethodCall(MethodCallExpr {
            base: obj,
            method_name: method,
            args,
            span: method_span,
            ..
        }) = expr
        else {
            panic!("Expected IndexAccess, got {expr:?}")
        };
        let method_span = *method_span;
        let (base_ty, checked_arg_types, borrows) = self.check_method_operands(obj, method, args);

        let site = MethodCallSite {
            obj,
            method: &*method,
            args: args.as_mut_slice(),
            base_ty: &base_ty,
            arg_types: &checked_arg_types,
            consume,
            span: method_span,
        };
        if let Some((ty, replacement)) = self.rewrite_special_method_call(&site) {
            if let Some(new_expr) = replacement {
                *expr = new_expr;
            }
            return ty;
        }
        self.report_unfolded_placement_query(&site);
        if let Some((ty, replace_with_obj)) =
            self.resolve_intrinsic_method(site.base_ty, site.method, site.args, &method_span)
        {
            if replace_with_obj {
                *expr = site.obj.clone();
            }
            return ty;
        }

        // Dynamic Method Resolution
        let mut mapping = HashMap::new();
        let found_method = self.resolve_method_in_impls(&base_ty, site.method, &mut mapping);
        if let Some(found) = found_method {
            let (ret_ty, func_call) =
                self.check_resolved_method_call(&found, mapping, &site, &borrows);
            *expr = func_call;
            return ret_ty;
        }

        // Fallback for hardcoded mock methods
        if let Some((transfer, is_relaxed)) = Self::device_transfer_rewrite(&site) {
            *expr = transfer;
            // Mark it so the per-seam obligation in `check_transfer_expr` sends
            // published payloads to TOP (a stale read).
            self.seam.pending_relaxed = is_relaxed;
            return self.check_transfer_expr(expr, consume);
        }
        let (ty, replacement) =
            self.check_builtin_method_fallback(&site, borrows.receiver_root.as_ref());
        if let Some(new_expr) = replacement {
            *expr = new_expr;
        }
        ty
    }

    /// Checks the receiver and the arguments of a call of the method `method`, each once.
    /// Gives the receiver's type, the arguments' types, and what was recorded about the
    /// borrows before the arguments were checked.
    fn check_method_operands(
        &mut self,
        obj: &mut Expr,
        method: &str,
        args: &mut [Expr],
    ) -> (Type, Vec<Type>, MethodBorrows) {
        let (base_ty, receiver_root) = self.check_method_receiver(obj, method);

        // Pre-infer closure argument types for specific intrinsics before type-checking them
        let reduce_init_ty = Self::preinfer_closure_params(&base_ty, method, args);

        // Check each argument once, keeping its type. A linear arg (e.g. a closure
        // struct passed to `.map`) is consumed by this pass, so re-checking it later for
        // generic deduction would see it moved and yield `Unknown` — reuse these instead.
        let snapshots = self.snapshot_borrowed_operands(obj, args);
        let errors_before_args = self.errors.inner.len();
        let checked_arg_types = self.check_method_arg_types(args, reduce_init_ty.as_ref());
        let borrows = MethodBorrows {
            snapshots,
            errors_before_args,
            receiver_root,
        };
        (base_ty, checked_arg_types, borrows)
    }

    /// Checks the receiver of a call of the method `method`. Gives the receiver's type, and
    /// the variable that changes if the method changes its receiver.
    fn check_method_receiver(&mut self, obj: &mut Expr, method: &str) -> (Type, Option<String>) {
        // Taking the address of a tensor does not read it, so `on_gpu.as_ptr()` is
        // allowed from host code even though a host read of `on_gpu` is not. The
        // pointer it gives carries the tensor's placement, and reading through that
        // is checked where the read happens.
        let takes_an_address = matches!(method, "as_ptr" | "as_mut_ptr");
        if takes_an_address && !self.speculating {
            self.drops_note_raw_pointer(obj);
        }
        let prev_allow = self.allow_cross_topology;
        if takes_an_address {
            self.allow_cross_topology = true;
        }
        // Found before the check rewrites `v[i]` into a call: the variable that
        // changes if the method changes its receiver.
        let receiver_root = self.changed_root(obj);
        let base_ty = self.check_expr_type_flag(obj, false);
        self.allow_cross_topology = prev_allow;
        (base_ty, receiver_root)
    }

    /// Gives the closure parameters of a tensor's `map` or `reduce` call the tensor's element
    /// type when they are written without a type. Gives the type the start value of `reduce`
    /// is expected to have, or `None` when the call is not a tensor's `reduce`.
    fn preinfer_closure_params(base_ty: &Type, method: &str, args: &mut [Expr]) -> Option<Type> {
        let mut reduce_init_ty = None;
        let tensor_ty = match base_ty {
            Type::Borrow { inner, .. } => inner.as_ref(),
            ty => ty,
        };
        if let Type::Tensor(el_ty, _, _) = tensor_ty {
            if method == "map" && args.len() == 1 {
                if let Expr::Closure(c) = &mut args[0] {
                    if c.params.len() == 1 && c.params[0].1 == Type::Unknown {
                        c.params[0].1 = Type::Scalar(el_ty.clone());
                    }
                }
            }
            // `t.reduce(init, |acc, x| ..)`: the running value and each element both
            // have the element type, and a bare literal `init` takes it too.
            if method == "reduce" && args.len() == 2 {
                if let Expr::Closure(c) = &mut args[1] {
                    for (_, ty) in &mut c.params {
                        if *ty == Type::Unknown {
                            *ty = Type::Scalar(el_ty.clone());
                        }
                    }
                }
                reduce_init_ty = Some(Type::Scalar(el_ty.clone()));
            }
        }
        reduce_init_ty
    }

    /// The borrows of each variable that the receiver or an argument of a method call reaches,
    /// as they are before the arguments are checked.
    fn snapshot_borrowed_operands(&self, obj: &Expr, args: &[Expr]) -> BaseSnapshots {
        let mut borrowed_snapshots = HashMap::new();
        for operand in std::iter::once(obj).chain(args.iter()) {
            if let Some((base, _)) = Self::arg_reborrow_base(operand) {
                borrowed_snapshots
                    .entry(base.clone())
                    .or_insert_with(|| self.borrow.snapshot_base(base.as_str()));
            }
        }
        borrowed_snapshots
    }

    /// Checks each argument of a method call once and gives their types. `reduce_init_ty` is
    /// the type the first argument is expected to have when the call is a tensor's `reduce`.
    fn check_method_arg_types(
        &mut self,
        args: &mut [Expr],
        reduce_init_ty: Option<&Type>,
    ) -> Vec<Type> {
        let mut checked_arg_types: Vec<Type> = Vec::with_capacity(args.len());
        for (i, arg) in args.iter_mut().enumerate() {
            // Only `reduce`'s start value gets an expected type here; every other
            // argument keeps the one it inherits.
            let Some(init_ty) = reduce_init_ty.filter(|_| i == 0).cloned() else {
                checked_arg_types.push(self.check_expr_type(arg));
                continue;
            };
            let prev = self.expected_type.replace(init_ty);
            checked_arg_types.push(self.check_expr_type(arg));
            self.expected_type = prev;
        }
        checked_arg_types
    }

    /// Checks the method calls that are rewritten before any method is looked up: a call of a
    /// module's function written `module.f(..)`, and a tensor's `extent(i)` and `len()`. Gives
    /// `None` when the call is none of them.
    fn rewrite_special_method_call(&mut self, site: &MethodCallSite<'_>) -> Option<Rewrite> {
        if let Some(rewrite) = self.rewrite_module_call(site) {
            return Some(rewrite);
        }
        if let Some(rewrite) = self.rewrite_extent_call(site) {
            return Some(rewrite);
        }
        self.rewrite_len_call(site)
    }

    /// Rewrites `module.f(args)` to a call of the function `f` that the module exports. Gives
    /// `None` when the receiver is not a module.
    fn rewrite_module_call(&mut self, site: &MethodCallSite<'_>) -> Option<Rewrite> {
        let Type::Module(path, exports) = site.base_ty else {
            return None;
        };
        let method = site.method;
        if let Some(exported_ty) = exports.get(method) {
            let prefix = TypeChecker::mangle_path(path);
            let mangled_name = format!("{prefix}_{method}");
            self.record_call_edge(&mangled_name, &site.span);
            let func_call = Expr::FunctionCall(FunctionCallExpr {
                name: crate::symbol::Symbol::from(mangled_name.as_str()),
                type_args: None,
                args: site.args.to_vec(),
                span: Span::default(),
            });
            return Some((exported_ty.clone(), Some(func_call)));
        }
        self.errors.push(format!(
            "Module '{path}' does not export function '{method}'"
        ));
        Some((Type::Unknown, None))
    }

    /// Rewrites `t.extent(i)` on a tensor: the run-time size of dimension `i`, an `i32`. Gives
    /// `None` when the call is not `extent` on a tensor.
    fn rewrite_extent_call(&mut self, site: &MethodCallSite<'_>) -> Option<Rewrite> {
        let method_span = site.span;
        // `t.extent(i)`: the run-time extent of dimension `i`, an i32. Rewritten to the
        // indexed-member form both lowerings carry, under a member name no source can
        // spell, so a written `.shape` is refused while this is not.
        if site.method.as_ref() != "extent" {
            return None;
        }
        let (_, dims, _) = Self::as_tensor_operand(site.base_ty)?;
        let rank = dims.len();
        let index = match &*site.args {
            [Expr::Number(n)] => n.value.as_ref().parse::<usize>().ok(),
            _ => None,
        };
        let Some(i) = index else {
            self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E3025,
                "`extent(i)` takes one literal dimension index".to_string(),
                Some(crate::diagnostic::SourceSpan::from_ast_span(&method_span)),
            );
            return Some((Type::Scalar(ElementType::I32), None));
        };
        if i >= rank {
            self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E3025,
                format!("`extent({i})` is out of range: the tensor has rank {rank}"),
                Some(crate::diagnostic::SourceSpan::from_ast_span(&method_span)),
            );
            return Some((Type::Scalar(ElementType::I32), None));
        }
        let index = site.args[0].clone();
        let rewritten = Expr::IndexAccess(IndexAccessExpr::new(
            Box::new(Expr::MemberAccess(MemberAccessExpr::new(
                Box::new(site.obj.clone()),
                crate::symbol::Symbol::from("$extent"),
                method_span,
            ))),
            Box::new(index),
            method_span,
        ));
        Some((Type::Scalar(ElementType::I32), Some(rewritten)))
    }

    /// Rewrites `t.len()` on a tensor: the size of the outermost axis, an `i32`. Gives `None`
    /// when the call is not `len` on a tensor.
    fn rewrite_len_call(&mut self, site: &MethodCallSite<'_>) -> Option<Rewrite> {
        let method_span = site.span;
        // `t.len()`: the extent of the outermost axis, the same read as `t.extent(0)`,
        // so `for i in 0..t.len() { t[i] }` visits exactly the positions `t[i]` accepts.
        // Typed `i32` like `extent` and like `Vec::len`. A rank-0 tensor holds one value
        // and has no axis to count, so it is refused rather than answered.
        if site.method.as_ref() != "len" {
            return None;
        }
        let (_, dims, _) = Self::as_tensor_operand(site.base_ty)?;
        if !site.args.is_empty() {
            self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E3025,
                "`len()` takes no arguments; `extent(i)` reads one axis".to_string(),
                Some(crate::diagnostic::SourceSpan::from_ast_span(&method_span)),
            );
            return Some((Type::Scalar(ElementType::I32), None));
        }
        if dims.is_empty() {
            self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E3025,
                "`len()` on a rank-0 tensor: a scalar has no outermost axis to count".to_string(),
                Some(crate::diagnostic::SourceSpan::from_ast_span(&method_span)),
            );
            return Some((Type::Scalar(ElementType::I32), None));
        }
        let rewritten = Expr::IndexAccess(IndexAccessExpr::new(
            Box::new(Expr::MemberAccess(MemberAccessExpr::new(
                Box::new(site.obj.clone()),
                crate::symbol::Symbol::from("$extent"),
                method_span,
            ))),
            Box::new(Expr::Number(NumberExpr::new(
                "0".to_string(),
                None,
                method_span,
            ))),
            method_span,
        ));
        Some((Type::Scalar(ElementType::I32), Some(rewritten)))
    }

    /// Reports `x.topology()` used as a value (E3026). The answer is known only while
    /// compiling, so the call has a meaning only where a comparison with it was worked out
    /// then.
    fn report_unfolded_placement_query(&mut self, site: &MethodCallSite<'_>) {
        // A placement query no comparison folded: nothing at run time holds a
        // placement, so it has no value here.
        if site.method.as_ref() == "topology"
            && matches!(site.base_ty, Type::Pinned(..) | Type::Tensor(..))
        {
            self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E3026,
                "a placement query is decided at compile time: compare `.topology()` \
                 with `Some(Topology::..)` or `None`"
                    .to_string(),
                Some(crate::diagnostic::SourceSpan::from_ast_span(&site.span)),
            );
        }
    }

    /// Checks a call of a method that was found in an impl block. `found` is the method and
    /// its impl block, and `mapping` the types the impl block's parameters take. Gives the
    /// result type and the function call to put in place of the method call.
    fn check_resolved_method_call(
        &mut self,
        found: &(Function, decl::ImplBlock),
        mapping: HashMap<Symbol, Type>,
        site: &MethodCallSite<'_>,
        borrows: &MethodBorrows,
    ) -> (Type, Expr) {
        let (generic_method, ib) = found;
        let (obj, method, base_ty) = (site.obj, site.method, site.base_ty);
        // An `unsafe fn` method needs an unsafe context at its call, exactly as a
        // free function does. Only the resolution differs, so the rule is repeated
        // here rather than shared: a method never reaches `env.functions`.
        if generic_method.is_unsafe && !self.in_unsafe_block && !self.speculating {
            self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E5001,
                format!(
                    "Call to unsafe method '{method}' is unsafe and requires unsafe \
                     function or block"
                ),
                Some(crate::diagnostic::SourceSpan::from_ast_span(&site.span)),
            );
        }
        self.check_method_operand_overlap(generic_method, site, borrows);
        // `w.peek()` passes `&w`, which the rewrite adds and checks only
        // speculatively, so it is checked here, as a written `&w` argument would be.
        let receiver_borrow = match generic_method.params.first() {
            Some((_, Type::Borrow { is_mut, .. }))
                if !matches!(base_ty, Type::Borrow { .. } | Type::Pointer(..)) =>
            {
                Self::arg_reborrow_base(obj).map(|(base, path)| (base, path, *is_mut))
            }
            _ => None,
        };
        self.note_receiver_effects(found, site, borrows.receiver_root.as_ref());
        let result = self.instantiate_method_call_rewrite(generic_method, mapping, ib, site);
        self.settle_method_call_borrows(site, borrows, &result, receiver_borrow);
        result
    }

    /// Reports two operands of a method call that borrow the same variable in a way a call
    /// may not, as `check_overlapping_borrows` does for a function call. The receiver counts
    /// as the first borrowed argument when `self` is a reference.
    fn check_method_operand_overlap(
        &mut self,
        generic_method: &Function,
        site: &MethodCallSite<'_>,
        method_borrows: &MethodBorrows,
    ) {
        let borrowed_snapshots = &method_borrows.snapshots;
        let mut borrows = std::iter::once(site.obj)
            .chain(site.args.iter())
            .zip(generic_method.params.iter())
            .filter_map(|(operand, (_, param_ty))| match param_ty {
                Type::Borrow { is_mut, .. } => {
                    Self::arg_reborrow_base(operand).map(|(base, path)| (base, path, *is_mut))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        let held = std::iter::once(site.obj)
            .chain(site.args.iter())
            .zip(generic_method.params.iter())
            .filter_map(|(operand, (_, param_ty))| match operand {
                Expr::Identifier(id)
                    if !Self::is_ref_type(param_ty) && self.result_can_hold_reference(param_ty) =>
                {
                    Some(id.name.to_string())
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        Self::add_held_borrows(&mut borrows, &held, borrowed_snapshots);
        self.check_overlapping_borrows(
            &borrows,
            borrowed_snapshots,
            method_borrows.errors_before_args,
            &site.span,
        );
    }

    /// Records what calling the method in `found` does to the receiver: a `&mut self` method
    /// changes it and a method that takes `self` by value moves it. Reports a call of `drop`
    /// by hand (E4012). `receiver_root` is the variable that changes with the receiver.
    fn note_receiver_effects(
        &mut self,
        found: &(Function, decl::ImplBlock),
        site: &MethodCallSite<'_>,
        receiver_root: Option<&String>,
    ) {
        let (generic_method, ib) = found;
        let base_ty = site.base_ty;
        // A method that takes `&mut self` changes a receiver that is not a reference.
        let takes_mut_self = matches!(
            generic_method.params.first(),
            Some((
                _,
                Type::Borrow { is_mut: true, .. } | Type::Pointer(_, _, true)
            ))
        );
        if takes_mut_self && !matches!(base_ty, Type::Borrow { .. } | Type::Pointer(..)) {
            self.changed_vars.extend(receiver_root.cloned());
        }
        // `x.drop()` would run `drop`, and `x` would be dropped again when its owner
        // gives it up, as in Rust.
        if ib.trait_name.as_deref() == Some("Drop") && !self.speculating {
            self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E4012,
                "`drop` is called for you when a value is dropped, and cannot be \
                 called by hand. To drop a value early, write `drop(x)` (from \
                 `core::mem`)",
                Some(crate::diagnostic::SourceSpan::from_ast_span(&site.span)),
            );
        }
        // A method that takes `self` by value moves its receiver: `w.into_iter()`.
        let self_by_value = generic_method
            .params
            .first()
            .is_some_and(|(_, t)| !matches!(t, Type::Borrow { .. } | Type::Pointer(..)));
        if site.consume
            && self_by_value
            && !matches!(base_ty, Type::Borrow { .. } | Type::Pointer(..))
            && base_ty.is_linear()
            && !self.is_copy(base_ty)
        {
            if let Expr::Identifier(id) = site.obj {
                self.consume(&id.name);
            }
        }
    }

    /// Ends or keeps the borrows a method call made, now that the type of its result is
    /// known. `result` is the result type and the function call the method call is rewritten
    /// to. `receiver_borrow` is the borrow of the receiver that the rewrite adds: the
    /// variable, the field path into it and whether it is `&mut`.
    fn settle_method_call_borrows(
        &mut self,
        site: &MethodCallSite<'_>,
        borrows: &MethodBorrows,
        result: &(Type, Expr),
        receiver_borrow: Option<(String, Vec<String>, bool)>,
    ) {
        let (ret_ty, func_call) = result;
        let (obj, base_ty, method_span) = (site.obj, site.base_ty, site.span);
        // A number or `bool` result holds no reference, so the borrows its
        // arguments made end with the call, as they do for a function call:
        // `let same = a.compare(&b)` leaves `b` free to assign.
        let holds_no_reference =
            matches!(ret_ty, Type::Scalar(el) if !matches!(el, ElementType::Generic(_)));
        if holds_no_reference && !self.speculating {
            for (base, snap) in &borrows.snapshots {
                let prev: &[BorrowRecord] = snap.as_deref().unwrap_or(&[]);
                self.borrow.retain_present(base.as_str(), prev);
            }
        }
        // A returned reference keeps the receiver borrowed while it is used.
        let holds_ref = !Self::is_ref_type(ret_ty) && self.result_can_hold_reference(ret_ty);
        let keeps_borrow = (matches!(ret_ty, Type::Borrow { .. }) || holds_ref)
            && match func_call {
                Expr::FunctionCall(fc) => {
                    self.env.return_provenance_of(fc.name.as_ref()).includes(0)
                }
                _ => true,
            };
        if let Some((base, path, is_mut)) = receiver_borrow {
            self.track_reference_arg_borrow(
                &base,
                path,
                is_mut,
                Self::is_mut_ref(ret_ty) || (holds_ref && is_mut),
                keeps_borrow,
                &method_span,
            );
        }
        // A receiver that is itself a reference: after `let r = m.get_mut()` with
        // `m : &mut S`, `r` may point where `m` points, so it holds every borrow
        // `m` holds.
        if keeps_borrow && matches!(base_ty, Type::Borrow { .. }) && !self.speculating {
            if let (Some((receiver, _)), Some(target)) = (
                Self::arg_reborrow_base(obj),
                self.current_assignment_target.clone(),
            ) {
                self.borrow
                    .copy_borrows(&receiver, &target, self.scopes.len());
            }
        }
    }

    /// The transfer expression that `obj.to_device()`, `to_sram()`, `to_gpu()` or one of their
    /// `_relaxed` forms stands for, and whether the form is a relaxed one. Gives `None` for
    /// any other method.
    fn device_transfer_rewrite(site: &MethodCallSite<'_>) -> Option<(Expr, bool)> {
        if !matches!(
            site.method.as_ref(),
            "to_device"
                | "to_device_relaxed"
                | "to_sram"
                | "to_sram_relaxed"
                | "to_gpu"
                | "to_gpu_relaxed"
        ) {
            return None;
        }
        // Device-placement transfers. The `_relaxed` variants are the escape
        // hatch that omits the synchronizing release / DMA-completion wait.
        let m: &str = site.method.as_ref();
        let is_relaxed = m.ends_with("_relaxed");
        let target_mem = if m.starts_with("to_sram") {
            MemorySpace::LocalSRAM // accelerator-core scratchpad
        } else if m.starts_with("to_gpu") {
            MemorySpace::GpuHbm // discrete-GPU device memory (NVPTX side)
        } else {
            MemorySpace::NPUHBM // to_device: default NPU[0]
        };
        let transfer = Expr::Transfer(TransferExpr {
            expr: Box::new(site.obj.clone()),
            space: target_mem,
            cost: None,
            lowering: None,
            span: site.span,
        });
        Some((transfer, is_relaxed))
    }

    /// Checks a call of a method that no impl block has: `to_host`, `as_ptr`, `as_mut_ptr`
    /// and `len` are built in, and any other name is reported as not found. Gives the call's
    /// type and, for `to_host`, the transfer expression to put in its place.
    fn check_builtin_method_fallback(
        &mut self,
        site: &MethodCallSite<'_>,
        receiver_root: Option<&String>,
    ) -> Rewrite {
        let (method, base_ty) = (site.method, site.base_ty);
        if method.as_ref() == "to_host" {
            let target_mem = MemorySpace::CPUDRAM;
            let pinned_ty = Type::Pinned(Box::new(base_ty.clone()), Topology::CPU);
            let transfer = Expr::Transfer(TransferExpr {
                expr: Box::new(site.obj.clone()),
                space: target_mem,
                cost: None,
                lowering: None,
                span: Span::default(),
            });
            (pinned_ty, Some(transfer))
        } else if method.as_ref() == "as_ptr" || **method == *"as_mut_ptr" {
            (self.check_as_ptr_call(site, receiver_root), None)
        } else if method.as_ref() == "len" {
            match base_ty {
                // A tensor's `len()` was rewritten to `extent(0)` above, so only a
                // borrow or pointer of something else still lands here.
                Type::Borrow { .. } | Type::Pointer(_, _, _) => {
                    (Type::Scalar(ElementType::I64), None)
                }
                _ => {
                    self.errors.push(format!("Cannot call len on {base_ty}"));
                    (base_ty.clone(), None)
                }
            }
        } else {
            // A method that is not found might have changed its receiver; assume so,
            // so the error does not also bring a warning that its `mut` is not needed.
            self.changed_vars.extend(receiver_root.cloned());
            if !self.speculating {
                self.errors
                    .push(format!("Method '{method}' not found on type {base_ty}"));
            }
            (base_ty.clone(), None)
        }
    }

    /// Checks `x.as_ptr()` or `x.as_mut_ptr()` and gives the pointer type. After an error the
    /// type is the receiver's own.
    fn check_as_ptr_call(
        &mut self,
        site: &MethodCallSite<'_>,
        receiver_root: Option<&String>,
    ) -> Type {
        let (method, base_ty) = (site.method, site.base_ty);
        let is_mut = method.as_ref() == "as_mut_ptr";
        // A tensor's `as_ptr()` is a pointer to its first element, placed where the
        // tensor is: `Tensor<f32, [4], Memory::GPU_HBM>` gives
        // `*const f32 in Memory::GPU_HBM`. Taking the address does not read the
        // shape, so both tensor spellings answer here (Vx#399), and a borrowed
        // tensor answers the same as the tensor.
        let element_pointer = |is_mut: bool, el: &ElementType, place: &Option<Placement>| {
            Type::Pointer(Box::new(Type::Scalar(el.clone())), place.clone(), is_mut)
        };
        match base_ty {
            Type::Tensor(el, _, place) => {
                // The tensor can be written through the pointer, so it changes.
                if is_mut {
                    self.changed_vars.extend(receiver_root.cloned());
                }
                element_pointer(is_mut, el, place)
            }
            Type::Borrow {
                inner,
                mem_space: mem,
                is_mut: mutability,
                ..
            } => {
                if is_mut && !mutability {
                    self.errors
                        .push("Cannot get mutable pointer from immutable borrow".to_string());
                }
                if let Type::Tensor(el, _, place) = inner.as_ref() {
                    element_pointer(is_mut, el, place)
                } else {
                    let place = mem
                        .as_ref()
                        .map(|m| crate::syntax::Placement::at(m.clone()));
                    Type::Pointer(inner.clone(), place, is_mut)
                }
            }
            Type::Pointer(_, _, _) => {
                self.errors.push("Already a pointer".to_string());
                base_ty.clone()
            }
            _ => {
                self.errors
                    .push(format!("Cannot call {method} on {base_ty}"));
                base_ty.clone()
            }
        }
    }

    /// Can a call's result, of type `ret`, hold a reference to one of the call's arguments? A
    /// type parameter or an unknown type says yes to `type_can_hold_reference`, which would keep
    /// every argument of `fn id<T>(x : &T) -> T` borrowed, so neither counts here.
    fn result_can_hold_reference(&self, ret: &Type) -> bool {
        !matches!(ret, Type::Generic(..) | Type::Unknown) && self.type_can_hold_reference(ret)
    }

    /// Resolve a callee's `(param types, return type)` for the reborrow analysis (#243, #268). The
    /// return type's *reference shape* (is it a reference? which parameters are references?) is all
    /// the reborrow decision needs; the summary (`return_provenance_of`) supplies which parameter
    /// the return derives from, defaulting to `AnyParam` where unknown.
    ///
    /// **Generics resolve to their declared signature** without instantiation: params/return may
    /// carry type variables, but `&T`/`&mut T` are still references, so a reborrow through a generic
    /// callee (bc9 through a generic, #268) is tracked conservatively instead of leaking untracked.
    /// A generic returning a non-reference has a non-reference declared return, so nothing persists —
    /// no false positive. Only genuinely unresolvable callees (closures resolved via other rules,
    /// `dyn`, intrinsics) still return `None`.
    ///
    /// **Function-pointer / closure *values*** (a parameter or local of `fn(..)->..` type, #268)
    /// carry their signature in their type, so a reborrow through `f(m)` is tracked even though which
    /// function `f` holds is unknown — the aliasing depends only on the signature. Checked first so a
    /// shadowing local wins over a same-named global function.
    pub(crate) fn resolve_callee_ref_signature(
        &self,
        resolved_name: &str,
    ) -> Option<(Vec<Type>, Type)> {
        if let Some((Type::Function(params, ret, _) | Type::Closure(params, ret), _)) =
            self.lookup(resolved_name)
        {
            return Some((params.clone(), (**ret).clone()));
        }
        if let Some(signature) = self.closure_value_signature(resolved_name) {
            return Some(signature);
        }
        if let Some(f) = self
            .mono
            .functions
            .iter()
            .find(|f| f.0.name.as_ref() == resolved_name)
        {
            return Some((
                f.0.params.iter().map(|(_, t)| t.clone()).collect(),
                f.0.return_type.clone(),
            ));
        }
        if let Some(f) = self.env.syntax_functions.get(resolved_name) {
            if f.generics.is_empty() {
                return Some((
                    f.params.iter().map(|(_, t)| t.clone()).collect(),
                    f.return_type.clone(),
                ));
            }
        }
        // Generic callee: its declared signature (the name may carry explicit type args, e.g.
        // `pass<i32>`, so strip them to the base name the generic table is keyed by).
        let base = resolved_name.split('<').next().unwrap_or(resolved_name);
        if let Some((gf, _)) = self.env.generic_functions.get(base) {
            return Some((
                gf.params.iter().map(|(_, t)| t.clone()).collect(),
                gf.return_type.clone(),
            ));
        }
        // A function from an `extern` block. Without its signature, none of the borrows its
        // arguments make would end when the call returns.
        if let Some(ext) = self
            .env
            .externs
            .iter()
            .find(|e| e.name.as_ref() == resolved_name)
        {
            return Some((
                ext.params.iter().map(|(_, t)| t.clone()).collect(),
                ext.return_type.clone(),
            ));
        }
        // An *imported* callee resolved from a merged `.vxlib` interface (#219 flip, phase 2) has no
        // AST in this compile, so it is absent from the env tables above — but its signature lives in
        // the frozen registry's `fn_sigs`. Resolving it there is what lets the borrow checker
        // reborrow-track (and, via `ret_prov`, provenance-refine) a cross-module reference call.
        // Checked last so a local definition always shadows a same-named import.
        if let Some(sig) = self
            .worker
            .global
            .registry
            .fn_sigs
            .get(&crate::symbol::Symbol::from(resolved_name))
        {
            return Some((sig.params.clone(), sig.ret_ty.clone()));
        }
        None
    }

    /// The parameter types and return type of the local `resolved_name` when it is a closure
    /// value: those of the function that holds the closure's body, without the first
    /// parameter, which is what the closure captured. `None` when the local is not a closure
    /// value or that function is not known yet.
    fn closure_value_signature(&self, resolved_name: &str) -> Option<(Vec<Type>, Type)> {
        // A closure *value* (`f` typed `Closure_N`): resolve its generated `Closure_N_call` and drop
        // the synthetic environment parameter (the leading `skip(1)`), so its reference arguments
        // line up with the call's arguments. Without this, a reborrow through a closure (`let r =
        // f(m); insert(m, ..)`) was tracked by nothing — the escape rule caught the unsound case only
        // incidentally, and over-rejected the sound one (#269).
        let closure_call: Option<String> = match self.lookup(resolved_name) {
            Some((Type::Struct(sname, _), _)) if sname.starts_with("Closure_") => {
                Some(format!("{sname}_call"))
            }
            _ => None,
        };
        let call_name = closure_call?;
        if let Some(f) = self
            .mono
            .functions
            .iter()
            .find(|f| f.0.name.as_ref() == call_name)
        {
            return Some((
                f.0.params.iter().skip(1).map(|(_, t)| t.clone()).collect(),
                f.0.return_type.clone(),
            ));
        }
        self.env.syntax_functions.get(call_name.as_str()).map(|f| {
            (
                f.params.iter().skip(1).map(|(_, t)| t.clone()).collect(),
                f.return_type.clone(),
            )
        })
    }

    /// Reports an error in a `reshape` or `pad` call (E3047).
    fn report_shape_error(&mut self, msg: String, span: &crate::syntax::Span) {
        self.errors.error_with_code(
            crate::diagnostic::DiagnosticCode::E3047,
            msg,
            Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
        );
    }

    /// `t.reshape([d0, d1, ..])`, optionally with `PadMode::Pad` or `PadMode::Trim`. Every
    /// error here is E3047. After one, the call's type is the shape it asked for when that is
    /// known, and `Unknown` otherwise, so a `let` that names the same shape does not report a
    /// second mismatch.
    fn check_reshape(&mut self, base_ty: &Type, args: &[Expr], span: &crate::syntax::Span) -> Type {
        let Type::Tensor(el_ty, dims, top) = base_ty else {
            unreachable!("reshape is only checked on a tensor");
        };
        if args.is_empty() || args.len() > 2 {
            self.report_shape_error(
                "reshape takes the new shape, and optionally PadMode::Pad or PadMode::Trim"
                    .to_string(),
                span,
            );
            return Type::Unknown;
        }

        // `None` is an exact reshape; otherwise "Pad" or "Trim".
        let mode_name = self.reshape_mode(args, span);

        let Expr::Array(ArrayExpr {
            elements: new_dims, ..
        }) = &args[0]
        else {
            self.report_shape_error(
                "reshape takes the new shape as an array of sizes, such as [4, 4]".to_string(),
                span,
            );
            return Type::Unknown;
        };
        let target = Type::Tensor(
            el_ty.clone(),
            new_dims.iter().cloned().map(Dim::Static).collect(),
            top.clone(),
        );

        let Some(src_elements) = self.element_count(dims.iter().map(|d| d.as_static()).collect())
        else {
            self.report_shape_error(
                format!(
                    "reshape needs a tensor whose sizes are known while compiling, not {base_ty}"
                ),
                span,
            );
            return target;
        };
        let Some(target_elements) = self.element_count(new_dims.iter().map(Some).collect()) else {
            self.report_shape_error(
                "reshape needs a new shape whose sizes are known while compiling".to_string(),
                span,
            );
            return Type::Unknown;
        };

        let refusal =
            Self::reshape_refusal(base_ty, mode_name.as_deref(), src_elements, target_elements);
        if let Some(msg) = refusal {
            self.report_shape_error(msg, span);
        }
        target
    }

    /// The mode a `reshape` call names in its second argument: "Pad" or "Trim", or `None` when
    /// there is no second argument. Reports a second argument that is neither.
    fn reshape_mode(&mut self, args: &[Expr], span: &crate::syntax::Span) -> Option<String> {
        let mut mode_name = None;
        if let Some(mode) = args.get(1) {
            match mode {
                Expr::EnumVariant(EnumVariantExpr {
                    enum_name,
                    variant_name,
                    ..
                }) if enum_name.as_ref() == "PadMode"
                    && matches!(variant_name.as_ref(), "Pad" | "Trim") =>
                {
                    mode_name = Some(variant_name.as_ref().to_string());
                }
                _ => self.report_shape_error(
                    "the second argument of reshape must be PadMode::Pad or PadMode::Trim"
                        .to_string(),
                    span,
                ),
            }
        }
        mode_name
    }

    /// The number of elements a shape holds: the product of its sizes. `None` when a size is
    /// missing or cannot be worked out while compiling.
    fn element_count(&self, sizes: Vec<Option<&Expr>>) -> Option<f64> {
        let empty_env = HashMap::new();
        let mut n = 1.0;
        for e in sizes {
            n *= e
                .and_then(|e| self.eval_expr(e, &empty_env))
                .and_then(|v| v.as_f64())?;
        }
        Some(n)
    }

    /// Why a reshape of a tensor of type `base_ty`, which holds `src_elements` elements, to a
    /// shape that holds `target_elements` is refused, or `None` when it is allowed.
    /// `mode_name` is "Pad", "Trim", or `None` for a reshape that keeps every element.
    fn reshape_refusal(
        base_ty: &Type,
        mode_name: Option<&str>,
        src_elements: f64,
        target_elements: f64,
    ) -> Option<String> {
        let placed = matches!(base_ty, Type::Tensor(_, _, Some(_)));
        // Without `Pad` a reshape is a view of the source's buffer, so it can keep fewer elements
        // but never add any. `Pad` copies into a new buffer, which can be larger.
        let counts = format!("{base_ty} has {src_elements}, the new shape has {target_elements}");
        if mode_name == Some("Pad") && placed {
            Some(format!(
                "PadMode::Pad of a tensor placed in a memory space is not supported yet: \
                 {base_ty}"
            ))
        } else if (src_elements - target_elements).abs() <= 1e-6 {
            None
        } else {
            match (mode_name, target_elements > src_elements) {
                (None, _) => Some(format!(
                    "reshape cannot change the number of elements: {counts}"
                )),
                (Some("Trim"), true) => Some(format!(
                    "PadMode::Trim keeps the first elements, so it cannot make a larger tensor: \
                     {counts}"
                )),
                (Some("Pad"), false) => Some(format!(
                    "PadMode::Pad cannot make a smaller tensor: {counts}; use PadMode::Trim to \
                     keep the first elements"
                )),
                (Some("Pad"), true) | (Some("Trim"), false) => None,
                (Some(other), _) => unreachable!("reshape mode {other} was refused above"),
            }
        }
    }

    /// `t.pad([d0, d1, ..])`: a new tensor of that shape, with `t` at the start of every axis
    /// and zeros after it. Every error here is E3047, like `reshape`'s.
    fn check_pad(&mut self, base_ty: &Type, args: &[Expr], span: &crate::syntax::Span) -> Type {
        let Type::Tensor(el_ty, dims, top) = base_ty else {
            unreachable!("pad is only checked on a tensor");
        };
        let [Expr::Array(ArrayExpr {
            elements: new_dims, ..
        })] = args
        else {
            self.report_shape_error(
                "pad takes the new shape as an array of sizes, such as [4, 4]".to_string(),
                span,
            );
            return Type::Unknown;
        };
        let target = Type::Tensor(
            el_ty.clone(),
            new_dims.iter().cloned().map(Dim::Static).collect(),
            top.clone(),
        );
        if top.is_some() {
            self.report_shape_error(
                format!("pad of a tensor placed in a memory space is not supported yet: {base_ty}"),
                span,
            );
            return target;
        }
        if new_dims.len() != dims.len() {
            self.report_shape_error(
                format!(
                    "pad keeps the number of axes: {base_ty} has {}, the new shape has {}",
                    dims.len(),
                    new_dims.len()
                ),
                span,
            );
            return target;
        }
        self.check_pad_axes(base_ty, dims, new_dims, span);
        target
    }

    /// Checks each axis of a `pad` call: the old size `dims` and the new size `new_dims` must
    /// both be known while compiling, and the new one must not be smaller. Stops at the first
    /// size that is not known.
    fn check_pad_axes(
        &mut self,
        base_ty: &Type,
        dims: &[Dim],
        new_dims: &[Expr],
        span: &crate::syntax::Span,
    ) {
        let empty_env = HashMap::new();
        for (axis, (old, new)) in dims.iter().zip(new_dims).enumerate() {
            let old = old
                .as_static()
                .and_then(|e| self.eval_expr(e, &empty_env))
                .and_then(|v| v.as_f64());
            let new = self.eval_expr(new, &empty_env).and_then(|v| v.as_f64());
            let Some(old) = old else {
                self.report_shape_error(
                    format!(
                        "pad needs a tensor whose sizes are known while compiling, not {base_ty}"
                    ),
                    span,
                );
                return;
            };
            let Some(new) = new else {
                self.report_shape_error(
                    "pad needs a new shape whose sizes are known while compiling".to_string(),
                    span,
                );
                return;
            };
            if new < old {
                self.report_shape_error(
                    format!(
                        "pad cannot make an axis smaller: axis {axis} of {base_ty} has {old}, \
                         the new shape has {new}"
                    ),
                    span,
                );
            }
        }
    }

    /// Checks a call of a method the compiler provides itself on a tensor or a pinned value,
    /// such as `t.map(..)` or `t.reshape(..)`. Gives the call's type, and whether the call is to
    /// be replaced by its receiver (`t.iter()` is `t`). Gives `None` when `method` is not one of
    /// these methods.
    pub(crate) fn resolve_intrinsic_method(
        &mut self,
        base_ty: &Type,
        method: &str,
        args: &mut [Expr],
        span: &crate::syntax::Span,
    ) -> Option<(Type, bool)> {
        // `a.reduce(..)` with `a : &Tensor<..>` reads the tensor it points to.
        if let Type::Borrow { inner, .. } = base_ty {
            if method == "reduce" && args.len() == 2 && matches!(**inner, Type::Tensor(..)) {
                return self.resolve_intrinsic_method(inner, method, args, span);
            }
        }
        if let Type::Pinned(_inner, _top) = base_ty {
            if method == "topology" {
                if !args.is_empty() {
                    self.errors
                        .push("topology requires 0 arguments".to_string());
                }
                return Some((Type::Struct("Option".into(), None), false));
            }
        }

        if let Type::Tensor(el_ty, dims, top) = base_ty {
            if method == "topology" {
                if !args.is_empty() {
                    self.errors
                        .push("topology requires 0 arguments".to_string());
                }
                return Some((Type::Struct("Option".into(), None), false));
            } else if method == "clone" {
                if !args.is_empty() {
                    self.errors.push("clone takes no arguments".to_string());
                }
                return Some((
                    Type::Tensor(el_ty.clone(), dims.clone(), top.clone()),
                    false,
                ));
            } else if method == "reshape" {
                return Some((self.check_reshape(base_ty, args, span), false));
            } else if method == "pad" {
                return Some((self.check_pad(base_ty, args, span), false));
            } else if method == "iter" {
                if !args.is_empty() {
                    self.errors.push("iter requires 0 arguments".to_string());
                }
                return Some((base_ty.clone(), true));
            } else if method == "map" {
                return Some(self.check_tensor_map(base_ty, args));
            } else if method == "reduce" && args.len() == 2 {
                return Some(self.check_tensor_reduce(el_ty, args));
            } else if method == "transpose" {
                return Some(self.check_tensor_transpose(base_ty, args));
            }
        }
        None
    }

    /// Checks `t.map(closure)` on a tensor of type `base_ty`. Gives the call's type, and
    /// `false`: the call is not replaced by its receiver.
    fn check_tensor_map(&mut self, base_ty: &Type, args: &mut [Expr]) -> (Type, bool) {
        if args.len() != 1 {
            self.errors
                .push("map requires exactly 1 argument (the closure)".to_string());
            return (base_ty.clone(), false);
        }

        let arg_ty = self.check_expr_type(&mut args[0]);
        if let Type::Struct(name, _) = &arg_ty {
            if !name.starts_with("Closure_") {
                self.errors
                    .push(format!("map expects a closure, got {arg_ty}"));
            }
        } else {
            self.errors
                .push(format!("map expects a closure, got {arg_ty}"));
        }
        (base_ty.clone(), false)
    }

    /// Checks `t.reduce(init, closure)` on a tensor whose elements have type `el_ty`. Gives the
    /// call's type, and `false`: the call is not replaced by its receiver.
    fn check_tensor_reduce(&mut self, el_ty: &ElementType, args: &mut [Expr]) -> (Type, bool) {
        // The one-argument form is `Iterator::reduce`, which `t.iter()` (rewritten to `t`)
        // also reaches; only this two-argument form is the tensor operation.
        let el_ty = Type::Scalar(el_ty.clone());
        let init_ty = self.check_expr_expecting(&mut args[0], Some(el_ty.clone()), false);
        if init_ty != Type::Unknown && !self.is_assignable(&el_ty, &init_ty) {
            self.errors.push(format!(
                "reduce starts from a value of the element type {el_ty}, got {init_ty}"
            ));
        }
        let closure_ty = self.check_expr_type(&mut args[1]);
        match &closure_ty {
            Type::Struct(name, _) if name.starts_with("Closure_") => {
                let returns = self
                    .mono
                    .closure_signatures
                    .get(name)
                    .map(|(params, ret)| (params.len(), ret.clone()));
                if let Some((count, ret)) = returns {
                    if count != 2 || (ret != Type::Unknown && ret != el_ty) {
                        self.errors.push(format!(
                            "reduce combines two values of type {el_ty} into one: its \
                             closure takes |acc, x| and returns {el_ty}"
                        ));
                    }
                }
            }
            _ => self
                .errors
                .push(format!("reduce expects a closure, got {closure_ty}")),
        }
        (el_ty, false)
    }

    /// Checks `t.transpose([..])` on a tensor of type `base_ty`. Gives the type with the axes in
    /// the new order, or `base_ty` after an error, and `false`: the call is not replaced by its
    /// receiver.
    fn check_tensor_transpose(&mut self, base_ty: &Type, args: &[Expr]) -> (Type, bool) {
        let Type::Tensor(el_ty, dims, top) = base_ty else {
            unreachable!("transpose is only checked on a tensor");
        };
        if args.len() != 1 {
            self.errors.push(
                "transpose requires exactly 1 argument (an array of permutation indices)"
                    .to_string(),
            );
            return (base_ty.clone(), false);
        }
        if let Expr::Array(ArrayExpr {
            elements: perm,
            span: _,
        }) = &args[0]
        {
            if perm.len() != dims.len() {
                self.errors
                    .push("transpose permutation map length must match tensor rank".to_string());
                return (base_ty.clone(), false);
            }
            let Some(new_dims) = self.permuted_dims(dims, perm) else {
                return (base_ty.clone(), false);
            };
            return (Type::Tensor(el_ty.clone(), new_dims, top.clone()), false);
        }
        self.errors
            .push("transpose requires an array of permutation indices".to_string());
        (base_ty.clone(), false)
    }

    /// The sizes `dims` in the order the permutation `perm` asks for. `perm` has one index for
    /// each of `dims`. Reports an index that cannot be worked out while compiling, is out of
    /// range or is given twice, and gives `None`.
    fn permuted_dims(&mut self, dims: &[Dim], perm: &[Expr]) -> Option<Vec<Dim>> {
        let empty_env = HashMap::new();
        let mut new_dims = vec![
            Dim::Static(Expr::Number(NumberExpr {
                value: "0".into(),
                ty: Some(ElementType::I32),
                span: Span::default()
            }));
            dims.len()
        ];
        let mut seen = vec![false; dims.len()];
        for (i, p) in perm.iter().enumerate() {
            if let Some(v) = self.eval_expr(p, &empty_env).and_then(|v| v.as_f64()) {
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "an index is a small whole number. The cast drops a fraction and \
                              turns a negative number into 0, and the range check below \
                              refuses a number that is too large"
                )]
                let v = v as usize;
                if v >= dims.len() {
                    self.errors
                        .push("transpose index out of bounds".to_string());
                    return None;
                }
                if seen[v] {
                    self.errors
                        .push("transpose permutation map must not contain duplicates".to_string());
                    return None;
                }
                seen[v] = true;
                new_dims[i] = dims[v].clone();
            } else {
                self.errors
                    .push("Cannot statically evaluate transpose permutation index".to_string());
                return None;
            }
        }
        Some(new_dims)
    }
}

impl<'a> TypeChecker<'a> {
    /// Every impl block that targets the type a static call names and has the method: the
    /// inherent one and each trait's. When the call names a trait, only that trait's blocks
    /// are looked at.
    fn static_method_candidates(&self, target: &StaticTarget<'_>) -> StaticCandidates<'a> {
        let struct_name = target.struct_name;
        let method_name = target.method_name;
        // A scalar target (`impl u64 { .. }`, `impl Default for i32`) is
        // matched by its spelling, which a name-only match over structs and enums never was.
        let target_matches = |ib: &decl::ImplBlock| -> bool {
            match &ib.target_type {
                Type::Struct(n, _) | Type::Enum(n, _) | Type::Generic(n, _) => *struct_name == **n,
                Type::GenericInstance(inner, _) => match &**inner {
                    Type::Struct(n, _) | Type::Enum(n, _) => *struct_name == **n,
                    _ => false,
                },
                Type::Scalar(el) => struct_name == el.to_string(),
                _ => false,
            }
        };
        let mut inherent: Vec<&'a decl::ImplBlock> = Vec::new();
        let mut from_traits: Vec<(&'a Symbol, &'a decl::ImplBlock)> = Vec::new();
        for (trait_key, impl_blocks) in &self.env.impls {
            if target.only_trait.is_some_and(|t| t != &**trait_key) {
                continue;
            }
            for ib in impl_blocks {
                if !target_matches(ib) || !ib.methods.iter().any(|m| m.name == method_name.into()) {
                    continue;
                }
                if trait_key.as_ref() == "_inherent" {
                    inherent.push(*ib);
                } else {
                    from_traits.push((trait_key, *ib));
                }
            }
        }
        StaticCandidates {
            inherent,
            from_traits,
        }
    }

    /// Picks the impl block a static call means. An inherent method is chosen before a
    /// trait's. Among the methods of several traits, the types of the arguments choose. Gives
    /// the block, which is `None` when no impl has the method, and the trait part of the
    /// instance's name. Gives `None` after reporting a call that more than one impl accepts
    /// (E3035) or that no impl accepts (E3036).
    fn choose_static_impl(
        &mut self,
        target: &StaticTarget<'_>,
        call: &CallArgs<'_>,
    ) -> Option<(Option<&'a decl::ImplBlock>, String)> {
        let StaticCandidates {
            inherent,
            mut from_traits,
        } = self.static_method_candidates(target);
        let method_name = target.method_name;
        let arg_types = call.types;
        // An inherent method shadows a trait's, as in Rust. Two traits' is a refusal rather
        // than a choice: the impls live in a hash map, and which one a call reached used to
        // change between runs of the compiler.
        // Kept before narrowing, so a call that no impl accepts can still name them all.
        let candidate_impls = from_traits.clone();
        // Several traits supplying this method name is not yet an ambiguity: the arguments
        // usually say which one is meant. Rust settles `i64::from(x)` the same way -- it does not
        // choose between impls at all, it infers the argument's type and then exactly one impl
        // applies. An untyped integer literal is already `i32` by the time it reaches here, which
        // is Rust's integer fallback arriving by a different route.
        if inherent.is_empty() && from_traits.len() > 1 {
            from_traits.retain(|(_, ib)| {
                ib.methods
                    .iter()
                    .find(|m| m.name == method_name.into())
                    .is_some_and(|m| params_accept(m, arg_types))
            });
        }
        let chosen: Option<&'a decl::ImplBlock> = if let Some(ib) = inherent.first() {
            Some(*ib)
        } else if from_traits.len() > 1 {
            // Still more than one after the arguments have had their say: two traits whose
            // methods take the same types. Nothing at the call can decide, so refuse.
            self.report_ambiguous_static_method(target, &from_traits, call.span);
            return None;
        } else if from_traits.is_empty() && !candidate_impls.is_empty() {
            // Every candidate was ruled out by the arguments. Name what IS implemented rather
            // than reporting an ambiguity, which is not what went wrong: the same shape in Rust
            // reports the bound that failed and lists the impls that exist.
            self.report_no_accepting_impl(target, &candidate_impls, call);
            return None;
        } else {
            from_traits.first().map(|(_, ib)| *ib)
        };
        let chosen_seg = Self::trait_name_segment(&inherent, &from_traits, target);
        Some((chosen, chosen_seg))
    }

    /// The trait part of the name of the instance a static call makes. It is empty for an
    /// inherent impl, which keeps the bare `Type::method` spelling.
    fn trait_name_segment(
        inherent: &[&decl::ImplBlock],
        from_traits: &[(&Symbol, &decl::ImplBlock)],
        target: &StaticTarget<'_>,
    ) -> String {
        // The trait and its arguments belong in the instantiated name for the same reason they
        // belong in instance dispatch's: `impl MyFrom<i32> for i64` and `impl MyFrom<u8> for i64`
        // are both `i64::my_from` without them, and whichever is instantiated first answers for
        // both. Empty for an inherent impl, which cannot collide.
        if inherent.is_empty() {
            from_traits
                .first()
                .map(|(tr, ib)| {
                    let seg = crate::syntax::types::trait_segment(Some(*tr), &ib.trait_args);
                    // A trait with no arguments adds nothing to the name, so `A::f` and `B::f`
                    // on one type would both be `i32::f`. A trait-qualified call names the trait.
                    if seg.is_empty() && target.only_trait.is_some() {
                        format!("{tr}$")
                    } else {
                        seg
                    }
                })
                .unwrap_or_default()
        } else {
            String::new()
        }
    }
}
