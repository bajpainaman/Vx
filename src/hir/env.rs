//===- sema.rs - Vx Compiler -----------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Semantic Analyzer for the Vx compiler.
// It is responsible for type checking, resolving operator overloading (e.g., tensor
// matrix multiplication), verifying memory topology constraints, and constructing
// the global AST environment for subsequent lowering phases.
//
// The Semantic Analyzer also includes the Lexical Borrow Checker, which handles local variable
// lifetimes and Strict Aliasing (Shared XOR Mutable).
//
// DESIGN NOTE: Speculative checking is carried by the `speculating` field (see its doc on
// `TypeChecker`), not a threaded `silent` parameter. It prevents duplicate compiler errors:
// an AST node is sometimes checked twice — once to probe a type (e.g. the method-call →
// function-call return-type probe), then again for real — so the probe runs with `speculating`
// on to suppress redundant emissions and side effects. Retired the former `silent: bool`
// parameter that threaded through ~36 checker signatures (#279 R3).
//
//===----------------------------------------------------------------------===//
use crate::syntax::{
    Dim, ElementType, EnumDecl, Expr, Function, ImplBlock, Mangle, MemoryDecl, MemorySpace,
    Program, Statement, StructDecl, Topology, TraitDecl, Type,
};
use std::collections::HashMap;

/// A value the compiler has worked out while compiling, such as the result of a `comptime`
/// block or a constant.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// `true` or `false`.
    Bool(bool),
    /// A floating-point number.
    Number(f64),
    /// An integer, kept as one. Sending it through `Number` would lose every value past
    /// 2^53 and would compute `3 / 2` as 1.5, which is not what the program does.
    Int(i64),
    /// A place code can run, such as `Topology::GPU(0)`.
    Topology(Topology),
    /// A fixed-size array of scalars, known at compile time. The length is set when the
    /// array is built and never changes, so an index past the end is a compile error.
    Array(Vec<Value>),
    /// A place in a byte buffer that a `comptime` block made from a string literal. It only
    /// exists while the block runs; every copy of the pointer sees the same bytes.
    Ptr {
        /// Which buffer the pointer points into.
        block: usize,
        /// How many bytes past the start of the buffer it points.
        offset: i64,
    },
    /// The text a `comptime` block's pointer result points at, up to its NUL. Written into the
    /// program as a string literal.
    CStr(String),
}

impl Value {
    /// This value as a float, whichever kind of number it is. For a caller that wants a
    /// magnitude and does not care how it was stored, such as a tensor dimension.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        reason = "the caller wants a magnitude; an integer past 2^53 becomes the nearest float"
    )]
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Int(i) => Some(*i as f64),
            Value::Number(n) => Some(*n),
            _ => None,
        }
    }
}

/// One `Memory`/`Topology` name declared by two modules with *different* declarations — see
/// [`GlobalAstEnv::duplicate_decls`].
#[derive(Debug, Clone, PartialEq)]
pub struct DuplicateDecl {
    /// What was re-declared, for the message: `"memory space"` or `"topology"`.
    pub kind: &'static str,
    /// The name that was declared twice.
    pub name: crate::symbol::Symbol,
    /// The module whose declaration collided with one already indexed.
    pub module: crate::symbol::Symbol,
}

/// Every declaration of one compilation, from all of its modules, looked up by name. The type
/// checker reads it and never changes it.
pub struct GlobalAstEnv<'a> {
    /// Every `struct`, by name.
    pub structs: HashMap<crate::symbol::Symbol, &'a StructDecl>,
    /// Every `enum`, by name.
    pub enums: HashMap<crate::symbol::Symbol, &'a EnumDecl>,
    /// Every `trait`, by name.
    pub traits: HashMap<crate::symbol::Symbol, &'a TraitDecl>,
    /// Every `impl` block, by the name of the trait it implements. A block that implements no
    /// trait is under `_inherent`.
    pub impls: HashMap<crate::symbol::Symbol, Vec<&'a ImplBlock>>,
    /// The signature of every function that is not generic, and of every `extern`, by name:
    /// return type, whether a call needs `unsafe`, parameter types, where it runs, and its
    /// `requires` and `ensures` conditions.
    #[allow(
        clippy::type_complexity,
        reason = "code in other files reads the parts of this tuple by position"
    )]
    pub functions:
        HashMap<crate::symbol::Symbol, (Type, bool, Vec<Type>, Topology, Vec<Expr>, Vec<Expr>)>,
    /// The declaration of every function that is not generic, by name.
    pub syntax_functions: HashMap<crate::symbol::Symbol, &'a Function>,
    /// Bodies the compile-time evaluator can run, owned rather than borrowed.
    ///
    /// The resolution env is built from signature-stripped modules, so a non-generic function
    /// defined in the module being compiled has an empty body in `syntax_functions` and the
    /// evaluator walks nothing. These are refilled from the full modules, which cannot be
    /// borrowed from because they are type-checked in place. Only functions a compile-time
    /// context could reach are copied, so this is not a second copy of the program.
    pub comptime_bodies: HashMap<crate::symbol::Symbol, Function>,
    /// Every generic function, by name, with the hash of the module that declares it.
    pub generic_functions: HashMap<crate::symbol::Symbol, (&'a Function, u64)>, // (func, origin_module_hash)
    /// User-defined memory spaces (`Memory <Name> { ... }`), indexed by name. Populated from
    /// `Program.memories` — the per-compilation home for memory descriptors (no global registry).
    pub memories: HashMap<crate::symbol::Symbol, &'a MemoryDecl>,
    /// Transfer lowerings (`impl transfer A -> B { ... }`) across every module of this
    /// compilation, in module order. A Vec rather than an edge-keyed map on purpose: the
    /// duplicate-edge check (E6015) needs to SEE both declarations to report one, and a map
    /// would have silently kept whichever was inserted last.
    pub transfer_impls: Vec<&'a crate::syntax::TransferImplDecl>,
    /// Every `extern` declaration in this compilation, for the checks that are about the
    /// signature rather than a call. `env.functions` cannot serve: it merges externs with Vx
    /// functions and keeps no way to tell them apart.
    pub externs: Vec<&'a crate::syntax::ExternDecl>,
    /// User-defined topologies (`Topology <Name> { ... }`), indexed by name. Populated from
    /// `Program.topologies`; the per-compilation home for topology descriptors, seeded into each
    /// `TransferCostGraph` — no global registry (see `docs/parallel_compiler_architecture.md`).
    pub topologies: HashMap<crate::symbol::Symbol, &'a crate::arch::TopologyDecl>,
    /// `Memory`/`Topology` names declared by more than one module in this compilation unit, with
    /// the modules that declared them. The tables above are name-keyed, so a duplicate would
    /// silently resolve to whichever module was indexed last -- and module order is a `HashMap`
    /// iteration, so the machine model actually in force would vary between runs. Recorded here
    /// and reported as E6012 by `check_declaration_conflicts` (#281).
    pub duplicate_decls: Vec<DuplicateDecl>,
    /// Per-function return-provenance summary (#243): which parameter slot(s) a reference-returning
    /// function's result roots in. A read-only, precomputed artifact of the immutable env, consulted
    /// at call sites to make the reborrow-persistence decision per argument. Filled from *present*
    /// bodies here (covers full-AST callers); production entry points that strip bodies before
    /// building refill it from the full modules via [`Self::annotate_return_provenances`]. A missing
    /// entry means `AnyParam` — today's conservative behaviour. See `crate::hir::provenance`.
    pub return_provenances:
        HashMap<crate::symbol::Symbol, crate::hir::provenance::ReturnProvenance>,
    /// This compilation's transfer-cost graph: the built-in memory-space edges plus every declared
    /// topology's, with the all-pairs shortest-path matrix precomputed.
    ///
    /// Built once here, at the end of `build_from_refs`, and shared by `&` with every
    /// `TypeChecker`. It used to be built inside `TypeChecker::new`, i.e. once per *function*,
    /// running the all-pairs precompute twice each time — which a profile put at the top of the
    /// whole compiler by self time. Every use in the checker is a `&self` read, so there is nothing
    /// per-function about it.
    pub transfer_cost_graph: crate::arch::TransferCostGraph,
    /// Top-level `const` items of every module, by name. A use of one is replaced by its value
    /// where the checker resolves the name.
    pub consts: HashMap<crate::symbol::Symbol, &'a crate::syntax::ConstDecl>,
}

impl<'a> GlobalAstEnv<'a> {
    /// Indexes the declarations of `modules`.
    #[must_use]
    pub fn build(modules: &'a [Program]) -> Self {
        let refs: Vec<&'a Program> = modules.iter().collect();
        Self::build_from_refs(&refs)
    }

    /// Indexes the declarations of `modules`. The same as [`Self::build`], for a caller that
    /// holds references to its modules.
    #[must_use]
    pub fn build_from_refs(modules: &[&'a Program]) -> Self {
        let mut env = Self {
            structs: HashMap::new(),
            enums: HashMap::new(),
            traits: HashMap::new(),
            impls: HashMap::new(),
            functions: HashMap::new(),
            syntax_functions: HashMap::new(),
            comptime_bodies: HashMap::new(),
            generic_functions: HashMap::new(),
            memories: HashMap::new(),
            topologies: HashMap::new(),
            duplicate_decls: Vec::new(),
            transfer_impls: Vec::new(),
            externs: Vec::new(),
            return_provenances: HashMap::new(),
            transfer_cost_graph: crate::arch::TransferCostGraph::default(),
            consts: HashMap::new(),
        };

        for &module in modules {
            env.index_types_and_machines(module);
            env.index_functions(module);
        }
        let funcs: Vec<&Function> = env.syntax_functions.values().copied().collect();
        crate::hir::provenance::refine_through_calls(&funcs, &mut env.return_provenances);
        // Build the transfer-cost graph once, now that `topologies` is populated. It used to be
        // built inside `TypeChecker::new` -- that is, once per *function* -- where it ran the
        // all-pairs shortest-path precompute twice (once in `default()`, once in
        // `seed_from_topologies`) and cloned every topology declaration in the compilation. A
        // profile of a 6,000-module corpus put `TransferCostGraph::transfer_path` at the top of the
        // whole compiler by self time, on a corpus that declares no topologies at all.
        //
        // The graph depends only on this compilation's topology declarations, and every one of its
        // ~31 uses in the checker is a `&self` read, so one per compilation is not merely an
        // optimisation -- it is what the comment on `seed_from_topologies` already claimed it was:
        // "the per-compilation, lock-free carrier the parallel pipeline shares by `&`".
        env.transfer_cost_graph = {
            let mut g = crate::arch::TransferCostGraph::default();
            let decls: Vec<crate::arch::TopologyDecl> =
                env.topologies.values().map(|&d| d.clone()).collect();
            // Edges only. `resolve_derived_route_costs` below rewrites edge costs and then runs
            // the shortest-path sweep itself, so a sweep here would be computed and immediately
            // overwritten -- O(spaces^2) searches thrown away, on the serial spine.
            g.add_topology_edges(&decls);
            // Routing minimises predicted cost, and a containment hop's cost lives
            // in the memory declarations rather than the topology's. Without this the on-die edges
            // -- which is every hop inside a device on every fleet SKU -- stay unpriced and the
            // router treats them as the last resort they are not.
            g.resolve_derived_route_costs(env.memories.values().copied());
            g
        };
        env
    }

    /// Indexes everything `module` declares except its functions: structs, enums, traits,
    /// impl blocks, constants, memory spaces, topologies and transfer lowerings.
    fn index_types_and_machines(&mut self, module: &'a Program) {
        let env = self;
        for s in &module.structs {
            env.structs.insert(s.name.clone(), s);
        }
        for m in &module.memories {
            if let Some(prev) = env.memories.insert(m.name.clone(), m) {
                // Re-declaring an *identical* space in two modules is harmless (the same
                // machine file reached twice); only a genuine disagreement is ambiguous.
                if prev != m {
                    env.duplicate_decls.push(DuplicateDecl {
                        kind: "memory space",
                        name: m.name.clone(),
                        module: module.module_path.clone(),
                    });
                }
            }
        }
        for t in &module.transfer_impls {
            env.transfer_impls.push(t);
        }
        for t in &module.topologies {
            if let Some(prev) = env.topologies.insert(t.name.clone(), t) {
                if prev != t {
                    env.duplicate_decls.push(DuplicateDecl {
                        kind: "topology",
                        name: t.name.clone(),
                        module: module.module_path.clone(),
                    });
                }
            }
        }
        for e in &module.enums {
            env.enums.insert(e.name.clone(), e);
        }
        for c in &module.consts {
            if let Some(prev) = env.consts.insert(c.name.clone(), c) {
                if prev != c {
                    env.duplicate_decls.push(DuplicateDecl {
                        kind: "const",
                        name: c.name.clone(),
                        module: module.module_path.clone(),
                    });
                }
            }
        }
        for t in &module.traits {
            env.traits.insert(t.name.clone(), t);
        }
        for i in &module.impls {
            let trait_name = match &i.trait_name {
                Some(name) => name.clone(),
                None => "_inherent".to_string().into(),
            };
            env.impls.entry(trait_name).or_default().push(i);
        }
    }

    /// Indexes the `extern` declarations and the functions of `module`.
    fn index_functions(&mut self, module: &'a Program) {
        let env = self;
        let module_hash = crate::hash::compute_module_hash(&module.module_path);
        for ext in &module.externs {
            env.externs.push(ext);
            let param_types: Vec<Type> = ext.params.iter().map(|(_, t)| t.clone()).collect();
            env.functions.insert(
                ext.name.clone(),
                (
                    ext.return_type.clone(),
                    !ext.is_safe,
                    param_types,
                    Topology::CPU,
                    Vec::new(),
                    Vec::new(),
                ),
            );
        }
        for func in &module.functions {
            if func.generics.is_empty() {
                let param_types: Vec<Type> = func.params.iter().map(|(_, t)| t.clone()).collect();
                env.functions.insert(
                    func.name.clone(),
                    (
                        func.return_type.clone(),
                        func.is_unsafe,
                        param_types,
                        func.topology.clone(),
                        func.requires.clone(),
                        func.ensures.clone(),
                    ),
                );
                env.syntax_functions.insert(func.name.clone(), func);
                // Summarize the return provenance while the body is present. Signature-stripped
                // callers (production) leave this empty and refill via
                // `annotate_return_provenances` from the full modules.
                if !func.body.is_empty() {
                    env.return_provenances.insert(
                        func.name.clone(),
                        crate::hir::provenance::compute_return_provenance(func),
                    );
                }
            } else {
                env.generic_functions
                    .insert(func.name.clone(), (func, module_hash));
            }
        }
    }

    /// Refill `return_provenances` from modules that still carry function bodies (#243). Production
    /// pipelines build the env from *signature-stripped* modules, so `build` cannot summarize their
    /// free functions; the entry points call this with the full pre-strip modules to restore
    /// precision. Idempotent — overwrites any existing entry. A function whose summary cannot be
    /// computed (no body) is left absent, i.e. conservative `AnyParam` at lookup.
    pub fn annotate_return_provenances(&mut self, modules: &[Program]) {
        for module in modules {
            for func in &module.functions {
                if func.generics.is_empty() && !func.body.is_empty() {
                    self.return_provenances.insert(
                        func.name.clone(),
                        crate::hir::provenance::compute_return_provenance(func),
                    );
                }
            }
        }
        let funcs: Vec<&Function> = modules.iter().flat_map(|m| m.functions.iter()).collect();
        crate::hir::provenance::refine_through_calls(&funcs, &mut self.return_provenances);
    }

    /// Fill [`GlobalAstEnv::comptime_bodies`] from modules that still carry bodies.
    ///
    /// Only what compile-time evaluation could reach is copied: the names mentioned in an
    /// `assert` or inside a `comptime` block, then whatever those functions call, and so on.
    /// A program that does no compile-time work copies nothing. Idempotent.
    pub fn annotate_comptime_bodies(&mut self, modules: &[Program]) {
        let mut with_bodies: HashMap<&str, &Function> = HashMap::new();
        for module in modules {
            for func in &module.functions {
                if !func.body.is_empty() {
                    with_bodies.insert(func.name.as_ref(), func);
                }
            }
        }

        let mut pending: Vec<String> = Vec::new();
        for module in modules {
            for func in &module.functions {
                for stmt in &func.body {
                    collect_comptime_seeds(stmt, &mut pending);
                }
                // A comptime function runs only while compiling, wherever it is called from.
                if TypeChecker::is_comptime_fn(func) {
                    pending.push(func.name.to_string());
                }
            }
        }

        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        while let Some(name) = pending.pop() {
            if !seen.insert(name.clone()) {
                continue;
            }
            let Some(func) = with_bodies.get(name.as_str()) else {
                continue;
            };
            // Whatever this one calls can be reached from a compile-time context too.
            let mut uses = std::collections::HashSet::new();
            for stmt in &func.body {
                TypeChecker::extract_uses_stmt(stmt, &mut uses);
            }
            pending.extend(uses);
            self.comptime_bodies
                .insert(func.name.clone(), (*func).clone());
        }
    }

    /// The return-provenance summary for a callee, or the conservative default when unknown.
    #[must_use]
    pub fn return_provenance_of(&self, name: &str) -> crate::hir::provenance::ReturnProvenance {
        self.return_provenances
            .get(name)
            .copied()
            .unwrap_or(crate::hir::provenance::ReturnProvenance::AnyParam)
    }
}

/// One borrow that is still in use: a `&` or `&mut` taken of a variable or of a part of it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BorrowRecord {
    /// Whether it is a `&mut` borrow.
    pub is_mut: bool,
    /// How deeply nested the block is that the borrow was taken in. The borrow ends when that
    /// block does.
    pub scope_depth: usize,
    /// The variable the reference was stored in, when it was stored in one.
    pub borrower_name: Option<String>,
    /// The part of the variable that is borrowed, as the fields taken to reach it. Empty when
    /// the whole variable is borrowed.
    pub path: Vec<String>,
}

/// Where a reference value ultimately points, for return-escape analysis (#243).
/// A reference may be returned iff its provenance is `External`; returning a `Local`
/// reference would leave it dangling once the function's stack frame is gone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefProvenance {
    /// Roots in caller-owned memory: a reference *parameter* (or `'static`). Safe to return.
    External,
    /// Roots in a function-local slot: a `let` binding, a by-value parameter, or a
    /// temporary. Returning it dangles. The number is the scope depth of the innermost block
    /// it points into: the value is gone once that block ends.
    Local(usize),
}

impl RefProvenance {
    /// Of two values, the one gone soonest: local beats external, a deeper block beats a
    /// shallower one.
    pub(crate) fn join(a: Self, b: Self) -> Self {
        match (a, b) {
            (Self::Local(x), Self::Local(y)) => Self::Local(x.max(y)),
            (Self::Local(x), _) | (_, Self::Local(x)) => Self::Local(x),
            _ => Self::External,
        }
    }
}

/// Checks the types, borrows and memory placements of function bodies, one function at a time.
#[allow(
    clippy::struct_excessive_bools,
    reason = "code in other files sets these flags by name; each one is a separate yes or no about the code being checked"
)]
pub struct TypeChecker<'a> {
    /// This worker thread's own state.
    pub worker: &'a mut crate::session::LocalWorkerState,
    /// Every declaration of the compilation, by name.
    pub env: &'a GlobalAstEnv<'a>,
    pub(crate) scopes: Vec<HashMap<crate::symbol::Symbol, (Type, Topology)>>,
    /// Whether a value may be used on a topology other than the one it lives on.
    pub allow_cross_topology: bool,
    /// The errors and warnings found so far.
    pub errors: crate::diagnostic::DiagnosticsVec,
    pub(crate) in_unsafe_block: bool,
    pub(crate) active_topology: Topology,
    pub(crate) active_memory: MemorySpace,
    /// The cost of moving data between memory spaces. The same graph as
    /// `env.transfer_cost_graph`.
    pub transfer_cost_graph: &'a crate::arch::TransferCostGraph,
    /// Borrow-checking state (active records + NLL liveness), encapsulated so a conflict-check read
    /// cannot bypass the dead-borrow sweep. Replaces the former `active_borrows` /
    /// `block_liveness` / `current_stmt_idx` fields.
    pub(crate) borrow: crate::hir::borrow_cx::BorrowCx,
    /// Speculative-check mode. When set, expression checking is a *probe* — diagnostics are
    /// suppressed, borrow/move side effects and unreachable-code warnings are skipped, and
    /// `check_if_expr`/`check_match_expr` return a placeholder without descending into the block.
    /// Replaces the `silent: bool` parameter that used to thread through ~36 checker signatures.
    /// The only site that turns it on is the
    /// method-call → function-call return-type probe in `check_methodcall_expr`; the two
    /// "fresh check" entry points (`check_expr_type`, `check_block`) force it back off, so the
    /// field's dynamic scope reproduces the old parameter's exactly.
    pub(crate) speculating: bool,
    /// Identifiers already reported as unreadable from the active topology, so the indexing of
    /// the same read (`c[0][0]`) does not report it again.
    pub(crate) unreadable_reported: std::collections::HashSet<crate::syntax::Span>,
    /// The name being indexed while the base of `name[i]` is checked: the one place a `const`
    /// table may be named.
    pub(crate) indexed_name: Option<crate::symbol::Symbol>,
    /// Set just before checking an expression whose value nothing uses: an expression statement,
    /// or the last expression of a block whose own value is unused. Blocks, `if`, `match` and
    /// `unsafe`/`comptime` blocks pass it on to the blocks inside them; `match` reads it, so a
    /// statement `match` whose arms yield nothing has no value rather than the function's
    /// return type. Every expression check takes it back to `false` on entry.
    pub(crate) value_unused: bool,
    /// Set while checking the left-hand side of an assignment (plain or compound): a container
    /// index there is a *store* place, so the `v[i]` -> `v.get(i)` read rewrite must not fire.
    pub(crate) checking_assign_lhs: bool,
    pub(crate) next_id: u32,
    /// The return type of the function being checked.
    pub current_return_type: Option<Type>,
    /// Name of the function being checked, so a record produced deep inside a body can say
    /// where it came from (a program may have several `spawn` regions in different
    /// functions). Saved and restored around `check_function` like `current_return_type`.
    pub(crate) current_function: String,
    pub(crate) current_assignment_target: Option<String>,
    /// Tracks which variables have been read during the current function check.
    pub(crate) used_vars: std::collections::HashSet<crate::symbol::Symbol>,
    /// Tracks declared variables with their spans (for unused variable warnings).
    pub(crate) declared_vars: Vec<(crate::symbol::Symbol, crate::syntax::Span)>,
    /// The current function's `let mut` bindings, for the warning about a `mut` never needed.
    pub(crate) declared_mut_vars: Vec<(crate::symbol::Symbol, crate::syntax::Span)>,
    /// The variables the current function changes: assigns, borrows `&mut`, or calls a
    /// `&mut self` method on.
    pub(crate) changed_vars: std::collections::HashSet<String>,
    /// Expected type of the expression currently being checked, from a `let x: T = …` or a
    /// `return` in a typed function. Lets a generic call deduce a *return-only* topology (or
    /// type) variable — e.g. `D` in `-> Pinned<T, D>` — from the call's context.
    pub(crate) expected_type: Option<Type>,
    /// Compile-time evaluation state. See [`crate::hir::check_state::ConstEvalState`].
    pub consteval: crate::hir::check_state::ConstEvalState,
    /// Monomorphization state and its output. See [`crate::hir::check_state::MonoState`].
    pub mono: crate::hir::check_state::MonoState,
    /// The memory algebra's seam obligations. See [`crate::hir::check_state::SeamState`].
    pub seam: crate::hir::check_state::SeamState,
    /// Traffic and capacity accounting. See [`crate::hir::check_state::TrafficState`].
    pub traffic: crate::hir::check_state::TrafficState,
}

impl<'a> TypeChecker<'a> {
    /// A checker with one empty scope, ready to check a function on the CPU.
    pub fn new(
        env: &'a GlobalAstEnv<'a>,
        worker: &'a mut crate::session::LocalWorkerState,
    ) -> Self {
        // Build the per-compilation cost graph first: fold in the topologies this compilation
        // declared (carried on the AST, indexed by `env.topologies`) so it holds both their
        // descriptors and transfer edges. No global registry -- the graph is the per-compilation,
        // lock-free carrier the parallel pipeline shares by `&`.
        let transfer_cost_graph = &env.transfer_cost_graph;
        let active_memory = transfer_cost_graph.default_memory_for(&Topology::CPU);
        Self {
            env,
            worker,
            scopes: vec![HashMap::new()],
            errors: crate::diagnostic::DiagnosticsVec::new(),
            in_unsafe_block: false,
            allow_cross_topology: false,
            active_topology: Topology::CPU,
            active_memory,
            transfer_cost_graph,
            borrow: crate::hir::borrow_cx::BorrowCx::default(),
            speculating: false,
            unreadable_reported: std::collections::HashSet::default(),
            indexed_name: None,
            value_unused: false,
            checking_assign_lhs: false,
            next_id: 1,
            current_return_type: None,
            current_function: String::new(),
            current_assignment_target: None,
            used_vars: std::collections::HashSet::new(),
            declared_vars: Vec::new(),
            declared_mut_vars: Vec::new(),
            changed_vars: std::collections::HashSet::new(),
            expected_type: None,
            consteval: crate::hir::check_state::ConstEvalState::default(),
            mono: crate::hir::check_state::MonoState::default(),
            seam: crate::hir::check_state::SeamState::default(),
            traffic: crate::hir::check_state::TrafficState::default(),
        }
    }

    /// Opens a block: names declared from here on are gone when the block is closed.
    pub fn push_scope(&mut self) {
        // Conservative by default: the tiles placed in here are treated as still resident
        // outside it. Only `push_releasing_scope` claims otherwise, and only for the scopes
        // whose end the lowering actually frees at.
        self.traffic.scope_releases.push(false);
        self.scopes.push(std::collections::HashMap::new());
        self.borrow
            .moved_vars
            .push(std::collections::HashSet::new());
        self.borrow
            .mutable_vars
            .push(std::collections::HashSet::new());
        self.consteval.env.push(std::collections::HashMap::new());
    }

    /// A scope the lowering frees at the end of: an `if` arm, a loop body. A tile placed inside
    /// one is gone once it closes, so it does not count against what is placed afterwards.
    ///
    /// Verified against the emitted IR rather than assumed, because the rule the lowering follows
    /// is dominance rather than nesting -- a scope whose block dominates the function's exits has
    /// its frees hoisted there instead. `if` arms and loop bodies do not dominate; a `spawn`
    /// region does, which is why it uses the plain `push_scope`.
    pub fn push_releasing_scope(&mut self) {
        self.traffic.next_scope_id += 1;
        let scope_id = self.traffic.next_scope_id;
        self.traffic.scope_chain.push(scope_id);
        self.scopes.push(std::collections::HashMap::new());
        self.traffic.scope_releases.push(true);
        self.borrow
            .moved_vars
            .push(std::collections::HashSet::new());
        self.borrow
            .mutable_vars
            .push(std::collections::HashSet::new());
        self.consteval.env.push(std::collections::HashMap::new());
    }

    /// Closes the innermost block, and ends the borrows taken in it.
    pub fn pop_scope(&mut self) {
        if self.traffic.scope_releases.pop().unwrap_or(false) {
            self.traffic.scope_chain.pop();
        }
        let depth = self.scopes.len();
        self.scopes.pop();
        self.borrow.moved_vars.pop();
        self.borrow.mutable_vars.pop();
        self.consteval.env.pop();

        // Lexical Lifetime cleanup: Remove borrows originating in this scope
        self.borrow.retain_scope(depth);
    }

    /// Declares `name` with type `ty` in the innermost block, on the topology being checked.
    pub fn insert(&mut self, name: String, ty: Type) {
        let current_top = self.active_topology.clone();
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.into(), (ty, current_top));
        }
    }

    /// Adds every name the statement mentions to `uses`.
    pub fn extract_uses_stmt(stmt: &Statement, uses: &mut std::collections::HashSet<String>) {
        match stmt {
            Statement::ExprStmt(e) => Self::extract_uses_expr(&e.expr, uses),
            Statement::Return(r) => {
                if let Some(e) = &r.expr {
                    Self::extract_uses_expr(e, uses);
                }
            }
            Statement::Assign(a) => {
                Self::extract_uses_expr(&a.lhs, uses);
                Self::extract_uses_expr(&a.rhs, uses);
            }
            Statement::CompoundAssign(ca) => {
                Self::extract_uses_expr(&ca.lhs, uses);
                Self::extract_uses_expr(&ca.rhs, uses);
            }
            Statement::LetDecl(l) => Self::extract_uses_expr(&l.expr, uses),
            Statement::ForLoop(f) => {
                Self::extract_uses_expr(&f.iterable, uses);
                for inv in &f.invariants {
                    Self::extract_uses_expr(inv, uses);
                }
                for s in &f.body {
                    Self::extract_uses_stmt(s, uses);
                }
            }
            Statement::Loop(l) => {
                for inv in &l.invariants {
                    Self::extract_uses_expr(inv, uses);
                }
                for s in &l.body {
                    Self::extract_uses_stmt(s, uses);
                }
            }
            Statement::Assert(a) => Self::extract_uses_expr(&a.expr, uses),
            // A drop uses what it frees, and leaves a mark a body already rewritten is known by.
            Statement::Drop(d) => {
                uses.insert(d.name.to_string());
                uses.insert(crate::hir::check::drops::DROP_MARK.to_string());
            }
            _ => {}
        }
    }

    /// Adds every name the expression mentions to `uses`.
    pub fn extract_uses_expr(expr: &Expr, uses: &mut std::collections::HashSet<String>) {
        match expr {
            Expr::Identifier(id) => {
                uses.insert(id.name.to_string());
            }
            Expr::MemberAccess(m) => Self::extract_uses_expr(&m.base, uses),
            // Two operands: an operator, an index, a range.
            Expr::BinaryOp(_)
            | Expr::RelationalOp(_)
            | Expr::LogicalOp(_)
            | Expr::IndexAccess(_)
            | Expr::Range(_) => Self::extract_uses_pair(expr, uses),
            Expr::UnaryOp(u) => Self::extract_uses_expr(&u.expr, uses),
            Expr::Borrow(b) => Self::extract_uses_expr(&b.expr, uses),
            // `transfer(x, Memory::W)` reads `x`. Without this the block's liveness map has no
            // entry for a value whose only reader is a transfer, which makes a live tile look
            // dead -- and the same map is what the dead-borrow sweep consults.
            Expr::Transfer(t) => Self::extract_uses_expr(&t.expr, uses),
            Expr::TransferPredicate(predicate) => {
                Self::extract_uses_topology(&predicate.from, uses);
                Self::extract_uses_topology(&predicate.to, uses);
            }
            Expr::Topology(topology_expr) => Self::extract_uses_topology(&topology_expr.top, uses),
            Expr::Dereference(d) => Self::extract_uses_expr(&d.expr, uses),
            Expr::AsCast(c) => Self::extract_uses_expr(&c.expr, uses),
            Expr::Closure(c) => {
                Self::extract_uses_expr(&c.body, uses);
            }
            // No names inside. A macro call is expanded before the checker runs.
            Expr::Number(_)
            | Expr::StringLiteral(_)
            | Expr::MemorySpace(_)
            | Expr::SizeOf(_)
            | Expr::MacroCall(_) => {}
            // Calls, blocks and lists of expressions.
            Expr::MethodCall(_)
            | Expr::FunctionCall(_)
            | Expr::IndirectCall(_)
            | Expr::Grad(_)
            | Expr::Jvp(_)
            | Expr::Vjp(_)
            | Expr::Print(_)
            | Expr::Println(_)
            | Expr::StructInit(_)
            | Expr::Array(_)
            | Expr::VecMacro(_)
            | Expr::EnumVariant(_)
            | Expr::InlineMlir(_)
            | Expr::SpawnOn(_)
            | Expr::Match(_)
            | Expr::If(_)
            | Expr::UnsafeBlock(_)
            | Expr::ComptimeBlock(_) => Self::extract_uses_nested(expr, uses),
        }
    }

    /// The part of [`Self::extract_uses_expr`] for an expression made of two others: a binary,
    /// relational or logical operator, an index access, a range.
    fn extract_uses_pair(expr: &Expr, uses: &mut std::collections::HashSet<String>) {
        match expr {
            Expr::BinaryOp(b) => {
                Self::extract_uses_expr(&b.lhs, uses);
                Self::extract_uses_expr(&b.rhs, uses);
            }
            Expr::RelationalOp(r) => {
                Self::extract_uses_expr(&r.lhs, uses);
                Self::extract_uses_expr(&r.rhs, uses);
            }
            Expr::LogicalOp(l) => {
                Self::extract_uses_expr(&l.lhs, uses);
                Self::extract_uses_expr(&l.rhs, uses);
            }
            Expr::IndexAccess(i) => {
                Self::extract_uses_expr(&i.base, uses);
                Self::extract_uses_expr(&i.index, uses);
            }
            Expr::Range(r) => {
                Self::extract_uses_expr(&r.start, uses);
                Self::extract_uses_expr(&r.end, uses);
            }
            // Every other kind of expression is handled by `extract_uses_expr`, which has
            // no catch-all, so a new kind of expression cannot be missed.
            _ => {}
        }
    }

    /// The part of [`Self::extract_uses_expr`] for an expression that holds a list of
    /// expressions or a block of statements: a call, a literal, an `if`, a `match`, a block.
    fn extract_uses_nested(expr: &Expr, uses: &mut std::collections::HashSet<String>) {
        match expr {
            Expr::MethodCall(m) => {
                Self::extract_uses_expr(&m.base, uses);
                Self::extract_uses_all(&m.args, uses);
            }
            Expr::FunctionCall(f) => {
                uses.insert(f.name.to_string());
                Self::extract_uses_all(&f.args, uses);
            }
            // A `spawn` region is where a placed tensor can legally be read, so the reads
            // that matter most for residency are inside one. Missing them made two tiles
            // held across a region look like one, which the space then had room for.
            Expr::SpawnOn(s) => {
                Self::extract_uses_topology(&s.top, uses);
                Self::extract_uses_block(&s.stmts, s.ret.as_deref(), uses);
            }
            Expr::Print(pr) => Self::extract_uses_all(&pr.args, uses),
            Expr::Println(pr) => Self::extract_uses_all(&pr.args, uses),
            Expr::Match(m) => {
                Self::extract_uses_expr(&m.expr, uses);
                for arm in &m.arms {
                    Self::extract_uses_block(&arm.body, None, uses);
                }
            }
            Expr::StructInit(s) => Self::extract_uses_all(s.fields.iter().map(|f| &f.1), uses),
            Expr::Array(a) => Self::extract_uses_all(&a.elements, uses),
            Expr::If(i) => {
                Self::extract_uses_expr(&i.cond, uses);
                Self::extract_uses_block(&i.then_block, None, uses);
                if let Some(eb) = &i.else_block {
                    Self::extract_uses_block(eb, None, uses);
                }
            }
            Expr::UnsafeBlock(u) => Self::extract_uses_block(&u.stmts, u.ret.as_deref(), uses),
            Expr::ComptimeBlock(c) => Self::extract_uses_block(&c.stmts, c.ret.as_deref(), uses),
            Expr::Grad(g) => Self::extract_uses_all(&g.args, uses),
            Expr::EnumVariant(v) => Self::extract_uses_all(v.payload.iter().flatten(), uses),
            Expr::IndirectCall(c) => {
                Self::extract_uses_expr(&c.callee, uses);
                Self::extract_uses_all(&c.args, uses);
            }
            Expr::VecMacro(v) => Self::extract_uses_all(&v.elements, uses),
            Expr::InlineMlir(m) => {
                Self::extract_uses_all(m.inputs.iter().map(|(_, e, _)| e), uses);
                Self::extract_uses_all(&m.clobbers, uses);
            }
            Expr::Jvp(j) => {
                Self::extract_uses_all(&j.args, uses);
                Self::extract_uses_expr(&j.tangent, uses);
            }
            Expr::Vjp(v) => {
                Self::extract_uses_all(&v.args, uses);
                Self::extract_uses_expr(&v.cotangent, uses);
            }
            // Every other kind of expression is handled by `extract_uses_expr`, which has
            // no catch-all, so a new kind of expression cannot be missed.
            _ => {}
        }
    }

    /// Adds every name that each of `exprs` mentions to `uses`.
    fn extract_uses_all<'e>(
        exprs: impl IntoIterator<Item = &'e Expr>,
        uses: &mut std::collections::HashSet<String>,
    ) {
        for e in exprs {
            Self::extract_uses_expr(e, uses);
        }
    }

    /// Adds every name mentioned in a block, and in the value it ends with, to `uses`.
    fn extract_uses_block(
        stmts: &[Statement],
        ret: Option<&Expr>,
        uses: &mut std::collections::HashSet<String>,
    ) {
        for stmt in stmts {
            Self::extract_uses_stmt(stmt, uses);
        }
        if let Some(ret) = ret {
            Self::extract_uses_expr(ret, uses);
        }
    }

    /// Adds every name mentioned in the index or the bounds of a topology to `uses`.
    fn extract_uses_topology(value: &Topology, uses: &mut std::collections::HashSet<String>) {
        match value {
            Topology::NPU(index) | Topology::AccCore(index) | Topology::GPU(index) => {
                TypeChecker::extract_uses_expr(index, uses);
            }
            Topology::Slice(base, start, end) => {
                Self::extract_uses_topology(base, uses);
                TypeChecker::extract_uses_expr(start, uses);
                TypeChecker::extract_uses_expr(end, uses);
            }
            Topology::CPU
            | Topology::AMX
            | Topology::ANE
            | Topology::CpuAvx512
            | Topology::CpuNeon
            | Topology::Current
            | Topology::Custom(_) => {}
        }
    }

    /// Record that a linear variable has been moved out of.
    ///
    /// The binding stays. Marking rather than deleting is what lets the variable be given a
    /// new value afterwards -- an assignment is a write, so whatever the name held before is
    /// beside the point, and there has to be something left to write to.
    ///
    /// The mark goes in the same scope as the binding, so the two live and die together.
    /// Putting it in the innermost scope instead made them come apart: the deletion reached
    /// an outer scope and stayed, while the mark was thrown away the moment the block
    /// closed. What was left was a name that was neither defined nor moved, reported as an
    /// undefined variable two lines after its own declaration.
    pub fn consume(&mut self, name: &str) {
        let Some(scope) = self.scope_of(name) else {
            return;
        };
        self.drops_note_move(name, false);
        if let Some(moved) = self.borrow.moved_vars.get_mut(scope) {
            moved.insert(name.to_string());
        }
    }

    /// Undo a move, because the variable has been given a value again.
    pub fn unconsume(&mut self, name: &str) {
        let Some(scope) = self.scope_of(name) else {
            return;
        };
        self.drops_note_move(name, true);
        if let Some(moved) = self.borrow.moved_vars.get_mut(scope) {
            moved.remove(name);
        }
    }

    /// The moved marks as they stand, for putting back after a branch whose moves do not
    /// reach the code that follows it.
    pub fn moved_snapshot(&self) -> Vec<std::collections::HashSet<String>> {
        self.borrow.moved_vars.clone()
    }

    /// Put the marks back as they were. Restores scope by scope rather than wholesale, so a
    /// mismatch in depth cannot silently resize the stack.
    pub fn restore_moved(&mut self, snapshot: Vec<std::collections::HashSet<String>>) {
        for (scope, marks) in snapshot.into_iter().enumerate() {
            if let Some(current) = self.borrow.moved_vars.get_mut(scope) {
                *current = marks;
            }
        }
    }

    /// The moves of two paths joined: moved if either path moved it.
    #[must_use]
    pub fn union_moved(
        mut a: Vec<std::collections::HashSet<String>>,
        b: Vec<std::collections::HashSet<String>>,
    ) -> Vec<std::collections::HashSet<String>> {
        for (scope, marks) in a.iter_mut().zip(b) {
            scope.extend(marks);
        }
        a
    }

    /// Record whether the binding of `name` now in scope was declared `mut`.
    pub(crate) fn set_mutable(&mut self, name: &str, is_mut: bool) {
        let Some(scope) = self.scope_of(name) else {
            return;
        };
        if let Some(set) = self.borrow.mutable_vars.get_mut(scope) {
            if is_mut {
                set.insert(name.to_string());
            } else {
                set.remove(name);
            }
        }
    }

    /// Whether the binding of `name` in scope was declared `mut`. A name with no binding here
    /// is not this check's business, so it answers yes.
    pub(crate) fn is_mutable(&self, name: &str) -> bool {
        match self.scope_of(name) {
            Some(scope) => self.borrow.mutable_vars[scope].contains(name),
            None => true,
        }
    }

    /// The innermost scope binding `name`. `moved_vars` is pushed and popped alongside
    /// `scopes`, so the index means the same thing in both.
    fn scope_of(&self, name: &str) -> Option<usize> {
        self.scopes
            .iter()
            .rposition(|scope| scope.contains_key(name))
    }

    /// Whether the variable `name` has been moved out of and not given a value since.
    pub fn is_moved(&self, name: &str) -> bool {
        // Only the binding that is actually in scope. A `let` of the same name in an inner
        // block is a different variable, and an outer one's mark says nothing about it.
        match self.scope_of(name) {
            Some(scope) => self.borrow.moved_vars[scope].contains(name),
            None => false,
        }
    }

    /// The type of the variable `name` and the topology it was declared on, from the
    /// innermost block that declares it.
    pub fn lookup(&self, name: &str) -> Option<&(Type, Topology)> {
        for scope in self.scopes.iter().rev() {
            if let Some(ty) = scope.get(name) {
                return Some(ty);
            }
        }
        None
    }

    /// The same as [`Self::lookup`], with the depth of the block that declares the name. The
    /// outermost block has depth 0.
    pub fn lookup_with_depth(&self, name: &str) -> Option<(&Type, &Topology, usize)> {
        for (i, scope) in self.scopes.iter().enumerate().rev() {
            if let Some((ty, top)) = scope.get(name) {
                return Some((ty, top, i));
            }
        }
        None
    }

    /// Matches the pattern `generic_ty` against `concrete_ty`, and records in `mapping` what
    /// each type variable of the pattern stands for. Returns false, and leaves `mapping` as it
    /// was, when they do not match.
    pub fn unify_types(
        &mut self,
        generic_ty: &Type,
        concrete_ty: &Type,
        mapping: &mut std::collections::HashMap<crate::symbol::Symbol, Type>,
    ) -> bool {
        let mut temp_mapping = mapping.clone();
        if self.unify_types_internal(generic_ty, concrete_ty, &mut temp_mapping) {
            *mapping = temp_mapping;
            true
        } else {
            false
        }
    }

    /// Unify a tensor pattern's element against a concrete one, binding a generic element
    /// (`Tensor<T>` against `Tensor<f32, [?, ?]>`) into `mapping`.
    fn unify_tensor_elem(
        e1: &ElementType,
        e2: &ElementType,
        mapping: &mut std::collections::HashMap<crate::symbol::Symbol, Type>,
    ) -> bool {
        if let ElementType::Generic(name) = e1 {
            if let Some(existing) = mapping.get(name) {
                return existing == &Type::Scalar(e2.clone());
            }
            mapping.insert(name.clone(), Type::Scalar(e2.clone()));
            return true;
        }
        e1 == e2
    }

    /// Bind a dimension name in a pattern (the `N` of `Tensor<f32, [N]>`) to the extent the
    /// argument supplies. A name that is already bound has to agree: `N` shared by two
    /// parameters says the two arguments have the same extent, so the second argument cannot
    /// silently rebind it. This matches what `unify_tensor_elem` and the `Type::Generic` arm
    /// already do for a shared element type and a shared type variable.
    fn bind_dim(
        name: &crate::symbol::Symbol,
        extent: Type,
        mapping: &mut std::collections::HashMap<crate::symbol::Symbol, Type>,
    ) -> bool {
        if let Some(existing) = mapping.get(name) {
            return match (Self::extent_of(existing), Self::extent_of(&extent)) {
                (Some(a), Some(b)) => a == b,
                _ => *existing == extent,
            };
        }
        mapping.insert(name.clone(), extent);
        true
    }

    /// The extent a binding stands for, however it was spelled. Unification binds a dimension
    /// as a `Type::Generic` holding the extent, while a call that spells the argument out
    /// (`same_len<4>(x, y)`) binds a `Type::Const` holding the literal. Both mean the same
    /// extent, so they have to compare equal.
    pub(crate) fn extent_of(ty: &Type) -> Option<&str> {
        match ty {
            Type::Generic(name, _) => Some(name.as_ref()),
            Type::Const(expr) => match expr.as_ref() {
                Expr::Number(n) => Some(n.value.as_ref()),
                Expr::Identifier(id) => Some(id.name.as_ref()),
                _ => None,
            },
            _ => None,
        }
    }

    /// Matches a tensor pattern against a concrete tensor type. Each side is the element
    /// type, the dimensions and the placement of one `Type::Tensor`.
    fn unify_tensor(
        (e1, d1, t1): (&ElementType, &[Dim], &Option<crate::syntax::Placement>),
        (e2, d2, t2): (&ElementType, &[Dim], &Option<crate::syntax::Placement>),
        mapping: &mut std::collections::HashMap<crate::symbol::Symbol, Type>,
    ) -> bool {
        // A pattern that names no place applies wherever the receiver lives; one that
        // names a place applies only there.
        let e1_match = Self::unify_tensor_elem(e1, e2, mapping);
        if !e1_match || (t1.is_some() && t1 != t2) {
            return false;
        }
        // Rank is static, so the two lists have to be the same length. Per dimension:
        // `?` in the pattern matches any extent; a name binds to a compile-time extent
        // and refuses a run-time one, since a `const` parameter cannot stand for a
        // value that does not exist until then; anything else matches itself.
        if d1.len() != d2.len() {
            return false;
        }
        for (dim1, dim2) in d1.iter().zip(d2.iter()) {
            match (dim1, dim2) {
                (Dim::Dyn, _) => {}
                (Dim::Static(Expr::Identifier(id)), Dim::Static(Expr::Number(n))) => {
                    let extent = Type::Generic(n.value.clone(), None);
                    if !Self::bind_dim(&id.name, extent, mapping) {
                        return false;
                    }
                }
                (Dim::Static(Expr::Identifier(id)), Dim::Static(Expr::Identifier(id2))) => {
                    let extent = Type::Generic(id2.name.clone(), None);
                    if !Self::bind_dim(&id.name, extent, mapping) {
                        return false;
                    }
                }
                (Dim::Static(Expr::Identifier(_)), Dim::Dyn) => return false,
                _ if dim1 != dim2 => return false,
                _ => {}
            }
        }
        true
    }

    /// Matches each pattern in `generic` against the type at the same position in `concrete`.
    /// The caller has checked that the two lists are the same length.
    fn unify_type_lists(
        &mut self,
        generic: &[Type],
        concrete: &[Type],
        mapping: &mut std::collections::HashMap<crate::symbol::Symbol, Type>,
    ) -> bool {
        for (a1, a2) in generic.iter().zip(concrete.iter()) {
            if !self.unify_types_internal(a1, a2, mapping) {
                return false;
            }
        }
        true
    }

    /// Matches a `ClosureK<Args.., Ret>` pattern, whose type arguments are `gi_args`, against
    /// the closure literal named `cn`.
    fn unify_closure_literal(
        &mut self,
        gi_args: &[Type],
        cn: &crate::symbol::Symbol,
        mapping: &mut std::collections::HashMap<crate::symbol::Symbol, Type>,
    ) -> bool {
        if let Some((params, ret)) = self.mono.closure_signatures.get(cn).cloned() {
            if gi_args.len() != params.len() + 1 {
                return false;
            }
            for (a, p) in gi_args.iter().zip(params.iter()) {
                if !self.unify_types_internal(a, p, mapping) {
                    return false;
                }
            }
            self.unify_types_internal(&gi_args[gi_args.len() - 1], &ret, mapping)
        } else {
            false
        }
    }

    /// Matches a `Pinned<T, Top>` pattern against a concrete one. Each side is the payload
    /// type and the topology.
    fn unify_pinned(
        &mut self,
        (t1, top1): (&Type, &Topology),
        (t2, top2): (&Type, &Topology),
        mapping: &mut std::collections::HashMap<crate::symbol::Symbol, Type>,
    ) -> bool {
        // A topology variable (`Pinned<_, D>` with D a `<D: Topology>` param) binds
        // to the argument's concrete topology; a concrete topology must match. Then
        // unify the payload (which may itself carry type variables).
        let tops_ok = match top1 {
            Topology::Custom(name) if self.mono.pending_topo_vars.contains(name) => {
                self.mono
                    .pending_topo_bindings
                    .insert(name.clone(), top2.clone());
                true
            }
            _ => top1 == top2,
        };
        tops_ok && self.unify_types_internal(t1, t2, mapping)
    }

    /// The matching behind [`Self::unify_types`]. It writes to `mapping` as it goes, so
    /// `mapping` can hold bindings from a match that failed part of the way through.
    fn unify_types_internal(
        &mut self,
        generic_ty: &Type,
        concrete_ty: &Type,
        mapping: &mut std::collections::HashMap<crate::symbol::Symbol, Type>,
    ) -> bool {
        match (generic_ty, concrete_ty) {
            (Type::Generic(name, _), _) => {
                if let Some(existing) = mapping.get(name) {
                    existing == concrete_ty
                } else {
                    mapping.insert(name.clone(), concrete_ty.clone());
                    true
                }
            }
            (Type::Tensor(e1, d1, t1), Type::Tensor(e2, d2, t2)) => {
                Self::unify_tensor((e1, d1, t1), (e2, d2, t2), mapping)
            }
            (Type::Pointer(t1, m1, mut1), Type::Pointer(t2, m2, mut2)) => {
                m1 == m2 && mut1 == mut2 && self.unify_types_internal(t1, t2, mapping)
            }
            (
                Type::Borrow {
                    inner: t1,
                    mem_space: m1,
                    is_mut: mut1,
                    ..
                },
                Type::Borrow {
                    inner: t2,
                    mem_space: m2,
                    is_mut: mut2,
                    ..
                },
            ) => m1 == m2 && mut1 == mut2 && self.unify_types_internal(t1, t2, mapping),
            (Type::Ref(t1, m1), Type::Ref(t2, m2)) => {
                m1 == m2 && self.unify_types_internal(t1, t2, mapping)
            }
            (Type::GenericInstance(b1, args1), Type::GenericInstance(b2, args2)) => {
                args1.len() == args2.len()
                    && self.unify_types_internal(b1, b2, mapping)
                    && self.unify_type_lists(args1, args2, mapping)
            }
            (Type::Function(p1, r1, u1), Type::Function(p2, r2, u2)) => {
                p1.len() == p2.len()
                    && u1 == u2
                    && self.unify_types_internal(r1, r2, mapping)
                    && self.unify_type_lists(p1, p2, mapping)
            }
            // A closure literal checks to `Struct("Closure_N")`, erasing its call signature.
            // Matching it against a `ClosureK<Args.., Ret>` parameter (e.g. `.map`'s
            // `Closure1<T, NewItem>`) recovers the args/ret from `closure_signatures` and unifies
            // them into the mapping so return-type generics like `NewItem` get bound. The generic
            // instance carries `[Args.., Ret]`; the recorded signature carries `(Args.., Ret)`.
            (Type::GenericInstance(inner, gi_args), Type::Struct(cn, _))
                if cn.starts_with("Closure_")
                    && matches!(&**inner, Type::Struct(n, _) if n.starts_with("Closure")) =>
            {
                self.unify_closure_literal(gi_args, cn, mapping)
            }
            (Type::Struct(n1, _), Type::Struct(n2, _)) => n1 == n2,
            (Type::Pinned(t1, top1), Type::Pinned(t2, top2)) => {
                self.unify_pinned((t1, top1), (t2, top2), mapping)
            }
            (t1, t2) => t1 == t2,
        }
    }

    /// Replace a topology variable (`Custom(name)` with `name` in `topo_mapping`) by its
    /// bound concrete topology; leave everything else unchanged.
    fn substitute_topology(
        top: Topology,
        topo_mapping: &std::collections::HashMap<crate::symbol::Symbol, Topology>,
    ) -> Topology {
        if let Topology::Custom(name) = &top {
            if let Some(bound) = topo_mapping.get(name) {
                return bound.clone();
            }
        }
        top
    }

    /// Apply `substitute_topology` to the topology component of every located type
    /// (`Pinned`/`Ref`) inside `ty`.
    fn substitute_topology_in_type(
        ty: Type,
        topo_mapping: &std::collections::HashMap<crate::symbol::Symbol, Topology>,
    ) -> Type {
        match ty {
            Type::Pinned(inner, top) => Type::Pinned(
                Box::new(Self::substitute_topology_in_type(*inner, topo_mapping)),
                Self::substitute_topology(top, topo_mapping),
            ),
            Type::Ref(inner, mem) => Type::Ref(
                Box::new(Self::substitute_topology_in_type(*inner, topo_mapping)),
                mem,
            ),
            other => other,
        }
    }

    /// In-place topology-variable substitution over a body statement (and its sub-
    /// expressions): specializes an explicit `spawn on(D)` and `Pinned<_, D>` annotation
    /// *inside* a generic body. The signature is handled separately in `instantiate_function`.
    fn subst_topo_in_stmt(
        stmt: &mut crate::syntax::Statement,
        tm: &std::collections::HashMap<crate::symbol::Symbol, Topology>,
    ) {
        use crate::syntax::Statement as S;
        match stmt {
            S::LetDecl(l) => {
                if let Some(ty) = l.ty_ann.take() {
                    l.ty_ann = Some(Self::substitute_topology_in_type(ty, tm));
                }
                Self::subst_topo_in_expr(&mut l.expr, tm);
            }
            S::ExprStmt(e) => Self::subst_topo_in_expr(&mut e.expr, tm),
            S::Return(r) => {
                if let Some(e) = &mut r.expr {
                    Self::subst_topo_in_expr(e, tm);
                }
            }
            S::Assign(a) => {
                Self::subst_topo_in_expr(&mut a.lhs, tm);
                Self::subst_topo_in_expr(&mut a.rhs, tm);
            }
            S::CompoundAssign(a) => {
                Self::subst_topo_in_expr(&mut a.lhs, tm);
                Self::subst_topo_in_expr(&mut a.rhs, tm);
            }
            S::ForLoop(f) => {
                Self::subst_topo_in_expr(&mut f.iterable, tm);
                for s in &mut f.body {
                    Self::subst_topo_in_stmt(s, tm);
                }
            }
            S::Loop(l) => {
                for s in &mut l.body {
                    Self::subst_topo_in_stmt(s, tm);
                }
            }
            _ => {}
        }
    }

    /// The same substitution as `subst_topo_in_stmt`, over the statements of a block and the
    /// value it ends with.
    fn subst_topo_in_block(
        stmts: &mut [crate::syntax::Statement],
        ret: Option<&mut Expr>,
        tm: &std::collections::HashMap<crate::symbol::Symbol, Topology>,
    ) {
        for s in stmts {
            Self::subst_topo_in_stmt(s, tm);
        }
        if let Some(r) = ret {
            Self::subst_topo_in_expr(r, tm);
        }
    }

    /// The same substitution as `subst_topo_in_stmt`, over one expression.
    fn subst_topo_in_expr(
        expr: &mut Expr,
        tm: &std::collections::HashMap<crate::symbol::Symbol, Topology>,
    ) {
        use crate::syntax::Expr as E;
        match expr {
            E::SpawnOn(e) => {
                e.top = Self::substitute_topology(e.top.clone(), tm);
                Self::subst_topo_in_block(&mut e.stmts, e.ret.as_deref_mut(), tm);
            }
            E::If(e) => {
                Self::subst_topo_in_expr(&mut e.cond, tm);
                Self::subst_topo_in_block(&mut e.then_block, None, tm);
                if let Some(eb) = &mut e.else_block {
                    Self::subst_topo_in_block(eb, None, tm);
                }
            }
            E::UnsafeBlock(e) => Self::subst_topo_in_block(&mut e.stmts, e.ret.as_deref_mut(), tm),
            E::ComptimeBlock(e) => {
                Self::subst_topo_in_block(&mut e.stmts, e.ret.as_deref_mut(), tm);
            }
            E::BinaryOp(e) => {
                Self::subst_topo_in_expr(&mut e.lhs, tm);
                Self::subst_topo_in_expr(&mut e.rhs, tm);
            }
            E::RelationalOp(e) => {
                Self::subst_topo_in_expr(&mut e.lhs, tm);
                Self::subst_topo_in_expr(&mut e.rhs, tm);
            }
            E::LogicalOp(e) => {
                Self::subst_topo_in_expr(&mut e.lhs, tm);
                Self::subst_topo_in_expr(&mut e.rhs, tm);
            }
            E::UnaryOp(e) => Self::subst_topo_in_expr(&mut e.expr, tm),
            E::Dereference(e) => Self::subst_topo_in_expr(&mut e.expr, tm),
            E::Borrow(e) => Self::subst_topo_in_expr(&mut e.expr, tm),
            E::AsCast(e) => {
                e.target_ty = Self::substitute_topology_in_type(e.target_ty.clone(), tm);
                Self::subst_topo_in_expr(&mut e.expr, tm);
            }
            E::Transfer(e) => Self::subst_topo_in_expr(&mut e.expr, tm),
            E::TransferPredicate(e) => {
                e.from = Self::substitute_topology(e.from.clone(), tm);
                e.to = Self::substitute_topology(e.to.clone(), tm);
            }
            E::IndexAccess(e) => {
                Self::subst_topo_in_expr(&mut e.base, tm);
                Self::subst_topo_in_expr(&mut e.index, tm);
            }
            E::MemberAccess(e) => Self::subst_topo_in_expr(&mut e.base, tm),
            E::FunctionCall(e) => {
                for a in &mut e.args {
                    Self::subst_topo_in_expr(a, tm);
                }
            }
            E::MethodCall(e) => {
                Self::subst_topo_in_expr(&mut e.base, tm);
                for a in &mut e.args {
                    Self::subst_topo_in_expr(a, tm);
                }
            }
            _ => {}
        }
    }

    /// The name of the copy of `generic_func` made for the types in `mapping` and the
    /// topologies in `topo_mapping`: the function's name, then `$` and each one in turn.
    fn mangled_instance_name(
        generic_func: &Function,
        mapping: &std::collections::HashMap<crate::symbol::Symbol, Type>,
        topo_mapping: &std::collections::HashMap<crate::symbol::Symbol, Topology>,
    ) -> String {
        // Writing into a `String` cannot fail, so the results of `write!` are dropped.
        use std::fmt::Write as _;
        let mut mangled_name = generic_func.name.to_string();
        // `T::Item` follows from `T`, so it adds nothing to the name.
        let mut sorted_keys: Vec<&crate::symbol::Symbol> = mapping
            .keys()
            .filter(|k| !k.as_ref().contains("::"))
            .collect();
        sorted_keys.sort();
        for g_name in sorted_keys {
            if let Some(ty) = mapping.get(g_name) {
                let _ = write!(mangled_name, "${}", ty.mangle());
            }
        }
        // Distinguish topology instantiations (`f$GPU` vs `f$NPU`).
        let mut topo_keys: Vec<&crate::symbol::Symbol> = topo_mapping.keys().collect();
        topo_keys.sort();
        for g_name in topo_keys {
            if let Some(top) = topo_mapping.get(g_name) {
                let _ = write!(mangled_name, "${:?}", top.kind());
            }
        }
        mangled_name
    }

    /// Makes the copy of the generic function `generic_func` in which each type variable is
    /// replaced by the type `mapping` gives it, and each topology variable by the topology
    /// `topo_mapping` gives it.
    pub fn instantiate_function(
        &mut self,
        generic_func: &Function,
        mapping: &std::collections::HashMap<crate::symbol::Symbol, Type>,
        topo_mapping: &std::collections::HashMap<crate::symbol::Symbol, Topology>,
    ) -> Function {
        let mangled_name = Self::mangled_instance_name(generic_func, mapping, topo_mapping);

        let new_params = generic_func
            .params
            .iter()
            .map(|(n, t)| {
                let substituted =
                    Self::substitute_topology_in_type(t.substitute(mapping), topo_mapping);
                (n.clone(), substituted)
            })
            .collect();
        let new_ret = Self::substitute_topology_in_type(
            generic_func.return_type.substitute(mapping),
            topo_mapping,
        );

        let new_body = generic_func
            .body
            .iter()
            .map(|s| {
                let mut s = s.substitute(mapping);
                if !topo_mapping.is_empty() {
                    Self::subst_topo_in_stmt(&mut s, topo_mapping);
                }
                s
            })
            .collect();

        Function {
            is_unsafe: false,
            name: mangled_name.into(),
            generics: Vec::new(),
            params: new_params,
            mut_params: generic_func.mut_params.clone(),
            topology: Self::substitute_topology(generic_func.topology.clone(), topo_mapping),
            return_type: new_ret,
            requires: generic_func
                .requires
                .iter()
                .map(|e| e.substitute(mapping))
                .collect(),
            ensures: generic_func
                .ensures
                .iter()
                .map(|e| e.substitute(mapping))
                .collect(),
            // Constraints were discharged at the call site before instantiation.
            where_transfers: Vec::new(),
            body: new_body,
            doc_comment: generic_func.doc_comment.clone(),
        }
    }

    /// `path` with every `/` and `.` replaced by `_`, so it can be part of a symbol name.
    #[must_use]
    pub fn mangle_path(path: &str) -> String {
        path.replace(['/', '.'], "_")
    }

    /// Checks one function: its signature, its body, and what the body leaves unused. A
    /// generic function is skipped, because its copies are checked where they are made.
    pub fn check_function(&mut self, func: &mut Function) {
        if !func.generics.is_empty() {
            return;
        }

        // `main` is the C entry point: its return value is the process exit code,
        // so it must return i32 (#210). A non-i32 `main` is a hard error rather
        // than a silently-discarded value.
        if func.name.as_ref() == "main" && func.return_type != Type::Scalar(ElementType::I32) {
            self.errors.push(format!(
                "`main` must return i32 (the process exit code), found {}",
                func.return_type
            ));
        }
        self.check_signature_types_are_declared(func);
        let exits = self.check_body_returns(func);

        let mut saved = self.enter_function(func);
        self.bind_parameters(func);
        self.check_function_body(func);

        // Each `return` checked the `ensures` where it stood. A body that can also reach its end
        // without one is checked here, from what is known at the end.
        if !exits && !self.speculating {
            for ens in &func.ensures {
                if !self.prove_expr(ens) {
                    self.errors.error_with_code(
                        crate::diagnostic::DiagnosticCode::E8001,
                        format!(
                            "Function '{}' cannot prove postcondition (ensures) at compile time",
                            func.name
                        ),
                        None,
                    );
                }
            }
        }
        self.consteval.current_ensures = std::mem::take(&mut saved.ensures);

        self.warn_unused_names(func);

        // The whole function has been checked; verify the working set of each memory space
        // it places tiles into fits (or is `overcommit`). Clears the per-function placement map.
        self.check_cumulative_capacity();

        self.pop_scope();
        self.leave_function(saved);
    }

    /// Reports a nominal type in the signature that names no declaration. An unknown name in type
    /// position parses as a user nominal and nothing downstream asks whether it was ever
    /// declared, so `Wibble<Frobnicate>` compiled and did nothing -- and an obsolete
    /// constructor kept "working" in the documentation long after it was removed.
    ///
    /// `Struct(name, None)` is a legitimate state on its own: a cross-module generic carries an
    /// unresolved base that later code resolves by name. What is checked here is stronger --
    /// that no declaration of that name exists anywhere the checker can see.
    fn check_signature_types_are_declared(&mut self, func: &Function) {
        let mut names = Vec::new();
        for (_, t) in &func.params {
            unresolved_nominals(t, &mut names);
        }
        unresolved_nominals(&func.return_type, &mut names);
        for n in names {
            // `void`/`none`/`!` are spelled as nominals but are builtins, and a `Closure_N` is
            // synthesized by the compiler rather than written by anyone.
            if n == "void" || n == "none" || n == "!" || n.starts_with("Closure_") {
                continue;
            }
            let sym = crate::symbol::Symbol::from(n.as_str());
            if !self.env.structs.contains_key(&sym) && !self.env.enums.contains_key(&sym) {
                self.errors.error_with_code(
                    crate::diagnostic::DiagnosticCode::E3029,
                    format!("no type named '{n}' is declared or imported"),
                    func.body
                        .first()
                        .map(|st| crate::diagnostic::SourceSpan::from_ast_span(&st.span())),
                );
            }
        }
    }

    /// Reports a non-void function whose body can complete without returning. Left to codegen this
    /// surfaced as `block with no terminator` naming an arith op, with no source location and
    /// no statement of what was wrong -- unreadable for the most ordinary mistake there is.
    ///
    /// Returns whether every path through the body ends in a `return` or never finishes.
    fn check_body_returns(&mut self, func: &Function) -> bool {
        let functions = &self.env.functions;
        let never = |name: &str| {
            functions
                .get(name)
                .is_some_and(|f| crate::syntax::is_never_ty(&f.0))
        };
        let exits = crate::syntax::expr::block_always_exits(&func.body, &never);
        if crate::syntax::is_never_ty(&func.return_type) && !exits {
            self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E3028,
                format!(
                    "'{}' is declared `-> !`, so it must never return, but its body can finish",
                    func.name
                ),
                func.body
                    .last()
                    .map(|st| crate::diagnostic::SourceSpan::from_ast_span(&st.span())),
            );
        } else if !crate::syntax::is_void_ty(&func.return_type) && !exits {
            self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E3028,
                format!(
                    "'{}' returns {} but its body can finish without returning a value",
                    func.name, func.return_type
                ),
                // The last statement, which is where the missing `return` belongs. A `Function`
                // carries no span of its own; an empty body has nothing to point at.
                func.body
                    .last()
                    .map(|st| crate::diagnostic::SourceSpan::from_ast_span(&st.span())),
            );
        }
        exits
    }

    /// Sets the checker up for the body of `func`, and returns what it held for the function
    /// around it. A function can be checked from inside the body of another, so everything
    /// that is kept per function is swapped out here and put back by `leave_function`.
    fn enter_function(&mut self, func: &Function) -> SavedFunctionState {
        let prev_constraints = self.consteval.constraints.clone();
        let prev_ensures =
            std::mem::replace(&mut self.consteval.current_ensures, func.ensures.clone());
        let prev_ret_ty = self.current_return_type.clone();
        let prev_fn = std::mem::replace(&mut self.current_function, func.name.as_ref().to_string());
        self.current_return_type = Some(func.return_type.clone());
        // A generic instantiation is checked from inside its caller's body, and the placement
        // walk is per-function: without a fresh slate the caller's tiles placed so far land in
        // the callee's cumulative check (mis-attributed and then lost to the caller, since that
        // check takes the map). Swap the per-function traffic out, restore it on exit.
        let prev_placements = std::mem::take(&mut self.traffic.memory_placements);
        let prev_call_sites = std::mem::take(&mut self.traffic.call_sites);
        let prev_scope_chain = std::mem::take(&mut self.traffic.scope_chain);
        let prev_scope_releases = std::mem::take(&mut self.traffic.scope_releases);
        // The unused-name sets are per-function for the same reason, and used to be cleared
        // on the way out instead of restored. A callee checked from inside a caller's body
        // wiped the names the caller had used so far, so the caller then warned that its own
        // parameters were unused -- which is every parameter read before the body's first
        // method call.
        let prev_used_vars = std::mem::take(&mut self.used_vars);
        let prev_declared_vars = std::mem::take(&mut self.declared_vars);
        let prev_declared_mut_vars = std::mem::take(&mut self.declared_mut_vars);
        let prev_changed_vars = std::mem::take(&mut self.changed_vars);
        self.push_scope();

        let prev_top = self.active_topology.clone();
        let prev_mem = self.active_memory.clone();
        self.active_topology = func.topology.clone();
        self.active_memory = self
            .transfer_cost_graph
            .default_memory_for(&self.active_topology);

        // Record the parameter set and clear per-function provenance state for the
        // return-escape analysis (#243). Saved/restored so nested checks (closures) don't
        // clobber the enclosing function's view.
        let prev_params = std::mem::take(&mut self.borrow.current_params);
        let prev_provenance = std::mem::take(&mut self.borrow.ref_provenance);
        let prev_views = std::mem::take(&mut self.borrow.views);
        // The borrow table too. A function's body can be checked from inside another's: a
        // generic method is checked when a call first instantiates it. The caller's borrows are
        // not the callee's, and a parameter of the same name (`self` in both) would otherwise
        // make the callee's uses conflict with the caller's live borrows, reported in the
        // callee's source.
        let prev_borrows = self.borrow.take();
        let prev_borrowers = self.borrow.take_borrowers();
        SavedFunctionState {
            constraints: prev_constraints,
            ensures: prev_ensures,
            ret_ty: prev_ret_ty,
            function: prev_fn,
            placements: prev_placements,
            call_sites: prev_call_sites,
            scope_chain: prev_scope_chain,
            scope_releases: prev_scope_releases,
            used_vars: prev_used_vars,
            declared_vars: prev_declared_vars,
            declared_mut_vars: prev_declared_mut_vars,
            changed_vars: prev_changed_vars,
            top: prev_top,
            mem: prev_mem,
            params: prev_params,
            provenance: prev_provenance,
            views: prev_views,
            borrows: prev_borrows,
            borrowers: prev_borrowers,
        }
    }

    /// Puts back what `enter_function` swapped out, except the `ensures` conditions, which
    /// `check_function` puts back earlier.
    fn leave_function(&mut self, saved: SavedFunctionState) {
        self.traffic.memory_placements = saved.placements;
        self.traffic.call_sites = saved.call_sites;
        self.traffic.scope_chain = saved.scope_chain;
        self.traffic.scope_releases = saved.scope_releases;
        self.current_return_type = saved.ret_ty;
        self.current_function = saved.function;
        self.consteval.constraints = saved.constraints;
        self.active_topology = saved.top;
        self.active_memory = saved.mem;
        self.borrow.current_params = saved.params;
        self.borrow.ref_provenance = saved.provenance;
        self.borrow.views = saved.views;
        self.borrow.restore(saved.borrows);
        self.borrow.restore_borrowers(saved.borrowers);
        self.used_vars = saved.used_vars;
        self.declared_vars = saved.declared_vars;
        self.declared_mut_vars = saved.declared_mut_vars;
        self.changed_vars = saved.changed_vars;
    }

    /// Declares the parameters of `func` in the scope of its body, and adds its `requires`
    /// conditions to what is known to hold.
    fn bind_parameters(&mut self, func: &Function) {
        for (name, ty) in &func.params {
            self.insert(name.to_string(), ty.clone());
            self.borrow.current_params.insert(name.clone(), ty.clone());
            self.set_mutable(name.as_ref(), func.mut_params.contains(name));
        }
        // Inside a transfer lowering, the raw:: primitives may only touch tiles the
        // transfer was given -- record the parameter names they are allowed to name.
        if self.seam.lowering_edge.is_some() {
            self.seam.lowering_params = func.params.iter().map(|(n, _)| n.clone()).collect();
        }

        // Add preconditions (requires) to our constraints
        for req in &func.requires {
            self.consteval.constraints.push(req.clone());
        }
    }

    /// Checks the statements of the body of `func`.
    fn check_function_body(&mut self, func: &mut Function) {
        // Pre-scan: gather the value contracts the consumer's asserts impose, so a
        // transfer seam (checked before the consumer's spawn body) can consult them.
        // Only when seam verification is enabled (otherwise we pay nothing for it).
        let prev_contracts = std::mem::take(&mut self.seam.contracts);
        if self.seam.verify {
            Self::collect_assert_contracts(&func.body, &mut self.seam.contracts);
        }

        let owned_params = func
            .params
            .iter()
            .map(|(n, t)| (n.to_string(), t.clone()))
            .collect();
        self.drops_enter_function(owned_params, &func.body);
        // A comptime function with parameters folds at each call, where they have values, so
        // its block is not folded here, the same as a comptime lambda's body. Without
        // parameters it folds here as well, so an error in it is reported even with no call.
        let comptime_fn = Self::is_comptime_fn(func) && !func.params.is_empty();
        let outer_comptime = if comptime_fn {
            self.consteval.closure_body_depth += 1;
            std::mem::take(&mut self.consteval.comptime_depth)
        } else {
            self.consteval.comptime_depth
        };
        self.check_block(&mut func.body, &func.return_type.clone());
        if comptime_fn {
            self.consteval.closure_body_depth -= 1;
            self.consteval.comptime_depth = outer_comptime;
        }
        self.drops_exit_function();
        Self::drop_spent_comptime_lambdas(&mut func.body);

        self.seam.contracts = prev_contracts;
    }

    /// Warns about each parameter and variable of `func` that is never read, and each
    /// `let mut` variable that never changes.
    fn warn_unused_names(&mut self, func: &Function) {
        // W1009: Unused function parameters
        for (param_name, _) in &func.params {
            let name_str: &str = param_name.as_ref();
            if !name_str.starts_with('_')
                && name_str != "self"
                && !self.used_vars.contains(param_name.as_ref())
            {
                self.errors.warn(
                    crate::diagnostic::DiagnosticCode::W1009,
                    format!("Unused function parameter '{param_name}'"),
                    None,
                );
            }
        }

        // W1001: Unused variable bindings
        for (var_name, var_span) in std::mem::take(&mut self.declared_vars) {
            let name_str: &str = var_name.as_ref();
            if !name_str.starts_with('_') && !self.used_vars.contains(&var_name) {
                let diag = self.errors.warn(
                    crate::diagnostic::DiagnosticCode::W1001,
                    format!("Unused variable '{var_name}'"),
                    Some(crate::diagnostic::SourceSpan::from_ast_span(&var_span)),
                );
                diag.fix_its.push(crate::diagnostic::FixIt {
                    message: "prefix with underscore to suppress".into(),
                    span: crate::diagnostic::SourceSpan::from_ast_span(&var_span),
                    replacement: format!("_{var_name}").into(),
                });
            }
        }

        // W1004: a `let mut` binding that never changes. An unused one already has W1001. An
        // instance of a generic function is skipped: `it.next()` changes `it` for some types
        // and not for a reference, so the `mut` may be needed by another instance.
        let is_instance = self.mono.being_checked.contains(&func.name);
        for (var_name, var_span) in std::mem::take(&mut self.declared_mut_vars) {
            let name_str: &str = var_name.as_ref();
            if !self.speculating
                && !is_instance
                && !name_str.starts_with(['_', '$'])
                && self.used_vars.contains(&var_name)
                && !self.changed_vars.contains(name_str)
            {
                self.errors.warn(
                    crate::diagnostic::DiagnosticCode::W1004,
                    format!(
                        "`{var_name}` is declared `mut` but never changes, so the `mut` can go"
                    ),
                    Some(crate::diagnostic::SourceSpan::from_ast_span(&var_span)),
                );
            }
        }
    }

    /// Parses the text of a type, such as `Vec<i32>`. Text that is not a type but is an
    /// expression, such as `4`, becomes a `Type::Const`; anything else is `Type::Unknown`.
    pub fn parse_ty_str(&self, s: &str) -> Type {
        let mut lexer = crate::lexer::Lexer::new(s);
        let tokens = lexer.tokenize();

        let mut parser = crate::parser::Parser::new(&tokens, s);
        if let Ok(ty) = parser.parse_type() {
            if parser.check(&crate::lexer::TokenType::Eof) {
                return self.resolve_parsed_type(ty);
            }
        }

        let mut expr_parser = crate::parser::Parser::new(&tokens, s);
        if let Ok(expr) = expr_parser.parse_primary_expr() {
            return Type::Const(Box::new(expr));
        }

        Type::Unknown
    }

    fn resolve_parsed_type(&self, ty: Type) -> Type {
        match ty {
            Type::Struct(name, id) => {
                // A nominal type name is a struct *or* an enum (both are represented
                // as `Type::Struct` nominally). Only demote to a generic type
                // parameter when the name matches neither — otherwise an enum type
                // argument like `Option<i32>` keeps a `Generic("Option")` head and
                // fails to unify against `impl<T> Option<T>` during method resolution.
                if self.env.structs.contains_key(&name)
                    || self.env.enums.contains_key(&name)
                    || self.mono.generated_structs.iter().any(|s| s.name == name)
                {
                    Type::Struct(name, id)
                } else {
                    Type::Generic(name, id)
                }
            }
            Type::GenericInstance(base, args) => {
                let resolved_base = Box::new(self.resolve_parsed_type(*base));
                let resolved_args = args
                    .into_iter()
                    .map(|a| self.resolve_parsed_type(a))
                    .collect();
                Type::GenericInstance(resolved_base, resolved_args)
            }
            Type::Pointer(inner, mem, mut_flag) => {
                Type::Pointer(Box::new(self.resolve_parsed_type(*inner)), mem, mut_flag)
            }
            Type::Ref(inner, mem) => Type::Ref(Box::new(self.resolve_parsed_type(*inner)), mem),
            Type::Borrow {
                inner,
                mem_space: mem,
                is_mut: mut_flag,
                region_id: r,
            } => Type::Borrow {
                inner: Box::new(self.resolve_parsed_type(*inner)),
                mem_space: mem.clone(),
                is_mut: mut_flag,
                region_id: r,
            },
            Type::Pinned(inner, top) => {
                Type::Pinned(Box::new(self.resolve_parsed_type(*inner)), top)
            }
            Type::Function(args, ret, unsafe_fn) => {
                let resolved_args = args
                    .into_iter()
                    .map(|a| self.resolve_parsed_type(a))
                    .collect();
                Type::Function(
                    resolved_args,
                    Box::new(self.resolve_parsed_type(*ret)),
                    unsafe_fn,
                )
            }
            Type::Closure(args, ret) => {
                let resolved_args = args
                    .into_iter()
                    .map(|a| self.resolve_parsed_type(a))
                    .collect();
                Type::Closure(resolved_args, Box::new(self.resolve_parsed_type(*ret)))
            }
            _ => ty,
        }
    }
}

/// What the checker held for the function around the one being checked. `enter_function`
/// fills it and `leave_function` puts it back. Each field is the saved value of the
/// `TypeChecker` state it is named after.
struct SavedFunctionState {
    constraints: Vec<Expr>,
    ensures: Vec<Expr>,
    ret_ty: Option<Type>,
    function: String,
    placements: HashMap<MemorySpace, HashMap<String, crate::hir::check_state::Placement>>,
    call_sites: HashMap<String, crate::hir::check_state::CallSite>,
    scope_chain: Vec<u32>,
    scope_releases: Vec<bool>,
    used_vars: std::collections::HashSet<crate::symbol::Symbol>,
    declared_vars: Vec<(crate::symbol::Symbol, crate::syntax::Span)>,
    declared_mut_vars: Vec<(crate::symbol::Symbol, crate::syntax::Span)>,
    changed_vars: std::collections::HashSet<String>,
    top: Topology,
    mem: MemorySpace,
    params: HashMap<crate::symbol::Symbol, Type>,
    provenance: HashMap<crate::symbol::Symbol, RefProvenance>,
    views: HashMap<crate::symbol::Symbol, crate::hir::check::views::View>,
    borrows: HashMap<crate::symbol::Symbol, Vec<BorrowRecord>>,
    borrowers: HashMap<String, std::collections::HashSet<String>>,
}

/// Adds to `out` the name of every struct or enum type inside `t` that has not been matched
/// to a declaration yet.
fn unresolved_nominals(t: &Type, out: &mut Vec<String>) {
    match t {
        Type::Struct(n, None) | Type::Enum(n, None) => out.push(n.to_string()),
        Type::Pointer(i, ..)
        | Type::Ref(i, _)
        | Type::Pinned(i, _)
        | Type::Verified(i)
        | Type::Borrow { inner: i, .. } => unresolved_nominals(i, out),
        Type::GenericInstance(b, args) => {
            unresolved_nominals(b, out);
            for a in args {
                unresolved_nominals(a, out);
            }
        }
        Type::Function(ps, r, _) | Type::Closure(ps, r) => {
            for a in ps {
                unresolved_nominals(a, out);
            }
            unresolved_nominals(r, out);
        }
        // A struct or enum that is already matched to its declaration, or a type with no
        // other type inside it.
        _ => {}
    }
}

/// Names mentioned where compile-time evaluation can start: an `assert` condition, and
/// anything inside a `comptime` block or an `if comptime` condition. Ordinary control flow in
/// between is walked through to reach them.
fn collect_comptime_seeds(stmt: &Statement, out: &mut Vec<String>) {
    match stmt {
        Statement::Assert(a) => push_uses_expr(&a.expr, out),
        Statement::LetDecl(l) => seeds_in_expr(&l.expr, out),
        Statement::Assign(a) => {
            seeds_in_expr(&a.lhs, out);
            seeds_in_expr(&a.rhs, out);
        }
        Statement::CompoundAssign(a) => {
            seeds_in_expr(&a.lhs, out);
            seeds_in_expr(&a.rhs, out);
        }
        Statement::Return(r) => {
            if let Some(e) = &r.expr {
                seeds_in_expr(e, out);
            }
        }
        Statement::ExprStmt(e) => seeds_in_expr(&e.expr, out),
        Statement::ForLoop(f) => {
            for s in &f.body {
                collect_comptime_seeds(s, out);
            }
        }
        Statement::Loop(l) => {
            for s in &l.body {
                collect_comptime_seeds(s, out);
            }
        }
        _ => {}
    }
}

/// The same walk over an expression. A `comptime` block seeds every name it mentions; an
/// ordinary block is only descended into, since nothing in it is evaluated on its own.
fn seeds_in_expr(expr: &Expr, out: &mut Vec<String>) {
    match expr {
        Expr::ComptimeBlock(c) => {
            for s in &c.stmts {
                push_uses_stmt(s, out);
            }
            if let Some(r) = &c.ret {
                push_uses_expr(r, out);
            }
        }
        Expr::If(i) => {
            if i.is_comptime {
                push_uses_expr(&i.cond, out);
            } else {
                seeds_in_expr(&i.cond, out);
            }
            for s in &i.then_block {
                collect_comptime_seeds(s, out);
            }
            if let Some(else_block) = &i.else_block {
                for s in else_block {
                    collect_comptime_seeds(s, out);
                }
            }
        }
        Expr::UnsafeBlock(u) => {
            for s in &u.stmts {
                collect_comptime_seeds(s, out);
            }
        }
        Expr::SpawnOn(s) => {
            for stmt in &s.stmts {
                collect_comptime_seeds(stmt, out);
            }
        }
        Expr::Match(m) => {
            for arm in &m.arms {
                for s in &arm.body {
                    collect_comptime_seeds(s, out);
                }
            }
        }
        _ => {}
    }
}

fn push_uses_expr(expr: &Expr, out: &mut Vec<String>) {
    let mut uses = std::collections::HashSet::new();
    TypeChecker::extract_uses_expr(expr, &mut uses);
    out.extend(uses);
}

fn push_uses_stmt(stmt: &Statement, out: &mut Vec<String>) {
    let mut uses = std::collections::HashSet::new();
    TypeChecker::extract_uses_stmt(stmt, &mut uses);
    out.extend(uses);
}

#[cfg(test)]
#[allow(
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    reason = "a test reports a failure by panicking"
)]
mod build_cost_tests {
    use crate::hir::GlobalAstEnv;

    fn machine(i: usize) -> String {
        format!(
            "Memory CPU_DRAM {{}}\n\
             Memory H{i} {{ within: Memory::CPU_DRAM, capacity: 8 GiB, bandwidth: 1 TB/s }}\n\
             Topology T{i} {{\n\
             \x20 memory: Memory::H{i},\n\
             \x20 visible: [Memory::CPU_DRAM, Memory::H{i}],\n\
             \x20 transfer Memory::CPU_DRAM -> Memory::H{i} : 63 GB/s\n\
             }}\n"
        )
    }

    fn parse(src: &str) -> crate::syntax::Program {
        let mut lexer = crate::lexer::Lexer::new(src);
        let tokens = lexer.tokenize();
        let mut parser = crate::parser::Parser::new(&tokens, src);
        parser.parse().expect("the machine corpus must parse")
    }

    /// Building the environment sweeps the transfer graph for shortest paths EXACTLY ONCE.
    ///
    /// The sweep is `O(spaces^2)` Dijkstra searches and it is essentially all of `env_build`,
    /// which runs once per compilation on the serial spine. A redundant sweep changes no answer --
    /// the later one overwrites the earlier -- so it is invisible except as every compile of a
    /// program that declares machines getting slower.
    ///
    /// It has gone wrong twice. The graph was first built inside `TypeChecker::new`, so the sweep
    /// ran twice per FUNCTION; hoisting it to one per compilation fixed that and left a second
    /// sweep behind, once in `seed_from_topologies` and again in `resolve_derived_route_costs`.
    /// At 800 declared machines that second sweep was 396 ms of a 778 ms phase, and the phase was
    /// half the compile (Vx#380).
    ///
    /// So this counts sweeps rather than timing them: a count is exact, is the same on every
    /// machine, and names the defect instead of reporting that something got slower.
    #[test]
    fn building_the_env_sweeps_the_transfer_graph_once() {
        const MACHINES: usize = 12;
        let mut src: String = (0..MACHINES).map(machine).collect::<Vec<_>>().join("\n");
        src.push_str("\nfn main() -> i32 { return 0; }\n");
        let program = parse(&src);
        let programs = vec![program];

        crate::arch::reset_sweep_sizes();
        let env = GlobalAstEnv::build(&programs);
        let sizes = crate::arch::sweep_sizes();

        assert_eq!(
            env.topologies.len(),
            MACHINES,
            "the corpus must actually declare the machines this measures"
        );

        // `TransferCostGraph::default()` sweeps the six built-in spaces before any declaration is
        // folded in, and that one is cheap and expected. What must not repeat is a sweep over the
        // DECLARED set, which is the one that grows with the machine file.
        let over_declared: Vec<usize> = sizes.iter().copied().filter(|&n| n > 8).collect();
        assert_eq!(
            over_declared.len(),
            1,
            "the enlarged transfer graph must be swept exactly once per compilation; \
             saw sweeps over {sizes:?} spaces. A second sweep is invisible in the output and \
             costs O(spaces^2) searches on the serial spine (Vx#380)."
        );
        assert!(
            over_declared[0] >= MACHINES,
            "the one real sweep must cover every declared space, saw {} for {MACHINES} machines",
            over_declared[0]
        );
    }

    /// The interning strategy is a per-compilation fact carried on the frozen session, not a
    /// process-global a worker reaches into (Vx#381).
    #[test]
    fn the_intern_strategy_is_carried_on_the_session() {
        use crate::intern_mode::InternMode;
        use crate::session::GlobalSession;

        let deferred = GlobalSession::new(1);
        assert_eq!(
            deferred.intern_mode,
            InternMode::Deferred,
            "a session must default to the shipped strategy rather than to whatever ran last"
        );

        let content = GlobalSession::new(1).in_intern_mode(InternMode::Content);
        assert_eq!(content.intern_mode, InternMode::Content);
        assert_eq!(
            deferred.intern_mode,
            InternMode::Deferred,
            "one session's choice must not be visible from another -- that is the whole point of \
             moving it off a global"
        );
    }
}
