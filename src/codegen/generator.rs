//===- generator.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Code generator module orchestrating the compilation pipeline from AST to MLIR to binary.
//
//===----------------------------------------------------------------------===//
use super::{
    Block, BlockLike, Context, DialectRegistry, ElementType, Expr, Function, HashMap, Location,
    Module, NumberExpr, Program, RegionLike, Statement, Type, Value, ValueLike,
};

use crate::codegen::lower::{LowerError, LowerToMelior};
use crate::syntax;
use std::fmt::Write as _;

/// Extract the quoted identifier from an LLVM named-struct MLIR type string, e.g.
/// `!llvm.struct<"Closure_1", ()>` -> `Closure_1`. Returns `None` for anonymous structs or
/// non-struct types. Used to recognize closure-env vs nominal-closure structs during coercion.
fn parse_llvm_struct_name(ty_str: &str) -> Option<String> {
    let rest = ty_str.strip_prefix("!llvm.struct<\"")?;
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// A name a `let` in a block shadows, with everything the generator knew about it before.
pub(crate) struct Shadowed<'c> {
    name: crate::symbol::Symbol,
    value: Option<(Value<'c, 'c>, Type<'c>)>,
    in_a_slot: bool,
    ast_ty: Option<syntax::Type>,
    owned: bool,
}

/// The code generator that walks the syntax tree and builds the MLIR module for a program.
#[allow(
    clippy::struct_excessive_bools,
    reason = "callers are in other files; each flag is read and set on its own by the lowering code"
)]
pub struct MeliorGenerator<'c> {
    pub(crate) context: &'c Context,
    pub(crate) module: Module<'c>,
    pub(crate) env: HashMap<crate::symbol::Symbol, (Value<'c, 'c>, Type<'c>)>,
    pub(crate) ast_env: HashMap<crate::symbol::Symbol, syntax::Type>,
    /// Locals bound to a tensor the compiler allocated, as against a view over memory it does
    /// not control. Only these can be filled in place by `c = a @ b` (Vx#391).
    pub(crate) owned_tensors: std::collections::HashSet<crate::symbol::Symbol>,
    /// The `printMemref*` helpers declared so far; each is declared on its first use.
    pub(crate) declared_tensor_prints: std::collections::HashSet<&'static str>,
    /// Locals whose element pointer the function takes: a view through it can share their memory.
    pub(crate) pointer_taken: std::collections::HashSet<crate::symbol::Symbol>,
    /// The locals of the function being lowered that are borrowed with `&x`. Each gets a stack
    /// slot, so every borrow of it is the same address.
    pub(crate) address_taken: std::collections::HashSet<crate::symbol::Symbol>,
    /// Declared memory spaces, keyed by space, so a `transfer` can emit the sub-space descriptor
    /// (granule/capacity/scope/parent) as IR metadata for later passes (see `subspace_scheduling.md`).
    pub(crate) memories: HashMap<syntax::MemorySpace, syntax::MemoryDecl>,
    /// Transfer lowerings by (from, to, topology) -- `impl Transfer<Memory::From,
    /// Memory::To> for Topology::X` -- the single method's body each. Sema records the
    /// key it chose on the `TransferExpr`; the body is inlined at the site in place of
    /// the builtin copy.
    ///
    /// The topology is in the key because the edge alone does not identify a lowering:
    /// an Ampere part and a Hopper part both declare `Memory::L2 -> Memory::SMEM` and
    /// move it with different instructions.
    pub(crate) transfer_impls: HashMap<(String, String, String), syntax::Function>,
    /// Declared topologies by name, so a `Pinned`/tensor value on a declared topology resolves the
    /// memory that topology actually names (`Topology SmemDev { memory: Memory::SMEM }`) when
    /// picking its address space, instead of guessing a like-named space (#258).
    pub(crate) topologies: HashMap<crate::symbol::Symbol, crate::arch::TopologyDescriptor>,
    /// Next free byte per granule'd sub-space — a bump allocator that assigns each tile a
    /// granule-rounded `offset` within its sub-space (SS2). Reset at each function boundary, since
    /// a sub-space (e.g. per-SM SMEM) is reused across kernels.
    pub(crate) subspace_offsets: HashMap<syntax::MemorySpace, u64>,
    pub(crate) structs: HashMap<crate::symbol::Symbol, syntax::StructDecl>,
    #[allow(
        clippy::type_complexity,
        reason = "the map mirrors how an enum is declared: each variant with its optional payload"
    )]
    pub(crate) enums:
        HashMap<crate::symbol::Symbol, Vec<(crate::symbol::Symbol, Option<Vec<syntax::Type>>)>>,
    /// Each generic enum's type parameter names, in order, so a use like `Pair<i64, i8>`
    /// can bind every one of them. `enums` keeps only the variants.
    pub(crate) enum_generics: HashMap<crate::symbol::Symbol, Vec<crate::symbol::Symbol>>,
    pub(crate) functions: HashMap<crate::symbol::Symbol, (Type<'c>, Vec<Type<'c>>)>,
    pub(crate) syntax_functions: HashMap<crate::symbol::Symbol, syntax::Function>,
    pub(crate) enzyme_decls: std::collections::HashSet<String>,
    /// A number that goes up each time a unique name is needed, such as for a string global.
    pub string_counter: usize,
    /// The program's `const` tables by name; `generate_module` adds each one's global.
    pub(crate) const_tables: HashMap<crate::symbol::Symbol, crate::registry::ConstTable>,
    /// The return type of the function being lowered. `None` outside a function.
    pub current_return_type: Option<Type<'c>>,
    /// The buffer the caller allocated for this function's tensor result, when it returns through
    /// one. A `return` writes its value in here and returns nothing. `None` for every other return
    /// type, which still returns by value. See `returns_through_slot`.
    pub current_return_slot: Option<melior::ir::Value<'c, 'c>>,
    /// The return slot offered to the expression currently being returned, so an op that would
    /// allocate a fresh result buffer writes into the caller's instead -- the named return value
    /// optimization, matching what the flat path does through `FnEmit::nrvo_slot`.
    ///
    /// Set by `return` around the expression it returns, and taken by the outermost op that can
    /// use it. A consumer takes it before lowering its own operands, so a nested op never mistakes
    /// the slot for its own scratch.
    pub(crate) nrvo_slot: Option<melior::ir::Value<'c, 'c>>,
    /// Drops marked `after_value` in the block being lowered, waiting for the statement they
    /// precede: the buffer the name held at the drop, and the name of its moved flag.
    pub(crate) queued_drops: Vec<WaitingDrop<'c>>,
    /// The `queued_drops` of the statement being lowered: they run once its value is computed.
    pub(crate) value_drops: Vec<WaitingDrop<'c>>,
    /// The type the surrounding code wants the expression being lowered to have, when it has
    /// a preference: for example the type of the left side of an assignment.
    pub expected_type: Option<Type<'c>>,
    /// Whether the code being lowered is the body of a `spawn`, which runs on a device.
    pub in_spawn: bool,
    /// For each loop being lowered, innermost last, the block a `break` jumps to.
    pub break_blocks: Vec<*const melior::ir::Block<'c>>,
    /// For each loop being lowered, innermost last, the block a `continue` jumps to.
    pub continue_blocks: Vec<*const melior::ir::Block<'c>>,
    /// The locals that live in a stack slot, so a read of one is a load from its slot.
    pub allocs: std::collections::HashSet<String>,
    /// One frame per block being lowered: what each name a `let` in the block shadows meant
    /// before it, put back when the block ends.
    pub(crate) blocks: Vec<Vec<Shadowed<'c>>>,
    pub(crate) is_lvalue_context: bool,
    pub(crate) mlir_block_counter: usize,
    pub(crate) has_returned: bool,
    pub(crate) current_filename: String,
    pub(crate) di_compile_unit: Option<melior::ir::Attribute<'c>>,
    pub(crate) di_subprogram: Option<melior::ir::Attribute<'c>>,
    /// Where in the source the statement or expression being lowered is.
    pub current_span: syntax::types::Span,
    pub(crate) index_ty: Type<'c>,
    pub(crate) i32_ty: Type<'c>,
    pub(crate) i64_ty: Type<'c>,
    pub(crate) f32_ty: Type<'c>,
    pub(crate) f64_ty: Type<'c>,
    pub(crate) f16_ty: Type<'c>,
    pub(crate) bf16_ty: Type<'c>,
    pub(crate) i1_ty: Type<'c>,
    pub(crate) ptr_ty: Type<'c>,
    pub(crate) none_ty: Type<'c>,
    /// Whether to transport host-proven `assert` facts across a device `spawn` seam as
    /// `llvm.intr.assume` certificates (`vxc --emit-seam-certs`). Off by default so
    /// ordinary lowering is unchanged. See `crate::codegen::lower::seam_cert`.
    pub emit_seam_certs: bool,
    /// Host-proven facts (the conditions of `assert`s lowered so far in the current
    /// function) available for transport into a device kernel body. Reset per function.
    pub(crate) assert_facts: Vec<syntax::Expr>,
}

/// `memref<2x3xf32>` -> `memref<?x?xf32>`, keeping the rank and element type.
///
/// Anything carrying a layout or memory space is returned unchanged: there is no shape to erase
/// there without losing the rest of the type with it.
fn erase_memref_extents(s: &str) -> String {
    let Some(inner) = s.strip_prefix("memref<").and_then(|t| t.strip_suffix('>')) else {
        return s.to_string();
    };
    if inner.contains(',') {
        return s.to_string();
    }
    let parts: Vec<&str> = inner.split('x').collect();
    let Some((elem, dims)) = parts.split_last() else {
        return s.to_string();
    };
    if dims.is_empty() || !dims.iter().all(|d| *d == "?" || d.parse::<i64>().is_ok()) {
        return s.to_string();
    }
    let mut out = vec!["?"; dims.len()];
    out.push(elem);
    format!("memref<{}>", out.join("x"))
}

/// `memref<4x4xf64, 1>` -> `memref<4x4xf64>`, keeping the shape and element type.
///
/// `memref.memory_space_cast` changes the space and nothing else, so a cast that also has to
/// change the shape needs the intermediate type this produces. Returns `None` where there is no
/// space to strip, or where the trailing component is a layout rather than a space.
pub(crate) fn strip_memref_space(s: &str) -> Option<String> {
    let inner = s.strip_prefix("memref<")?.strip_suffix('>')?;
    let (head, tail) = inner.rsplit_once(',')?;
    tail.trim().parse::<u32>().ok()?;
    Some(format!("memref<{head}>"))
}

/// The functions in `operations` without the imported ones (the first `imported`) that nothing
/// reaches: an imported module's function is emitted only when the program calls it, directly
/// or through another function it calls. A function is reached by any `@name` in a reached
/// function's code. Also returns every name the kept functions refer to.
///
/// # Panics
///
/// Never in practice: the pattern that finds `@name` is fixed and valid.
fn keep_reached_functions(
    operations: Vec<melior::ir::Operation<'_>>,
    imported: usize,
) -> (
    Vec<melior::ir::Operation<'_>>,
    std::collections::HashSet<String>,
) {
    use melior::ir::operation::OperationLike;
    let symbol = regex::Regex::new(r#"@(?:"((?:[^"\\]|\\.)*)"|([A-Za-z0-9_$.]+))"#)
        .expect("a valid pattern");
    let texts: Vec<String> = operations
        .iter()
        .map(std::string::ToString::to_string)
        .collect();
    let names: HashMap<String, usize> = operations
        .iter()
        .enumerate()
        .filter_map(|(i, op)| {
            let name = op.attribute("sym_name").ok()?;
            let name = melior::ir::attribute::StringAttribute::try_from(name).ok()?;
            Some((name.value().to_string(), i))
        })
        .collect();
    let mut reached: Vec<bool> = (0..operations.len()).map(|i| i >= imported).collect();
    let mut pending: Vec<usize> = (imported..operations.len()).collect();
    let mut referenced = std::collections::HashSet::new();
    while let Some(i) = pending.pop() {
        for found in symbol.captures_iter(&texts[i]) {
            let name = found
                .get(1)
                .or_else(|| found.get(2))
                .map_or("", |m| m.as_str());
            referenced.insert(name.to_string());
            if let Some(&j) = names.get(name) {
                if !reached[j] {
                    reached[j] = true;
                    pending.push(j);
                }
            }
        }
    }
    let kept = operations
        .into_iter()
        .zip(reached)
        .filter_map(|(op, keep)| keep.then_some(op))
        .collect();
    (kept, referenced)
}

/// Whether a lowered return type travels through a buffer the caller allocated, rather than
/// coming back by value: a tensor whose extents are all known at compile time.
///
/// The same rule the flat path applies in `flat::returns_through_slot`, and the two have to agree
/// -- every fixture runs on both back ends against one set of CHECK lines, and a caller on one
/// path can link a body compiled by the other. A `?` extent gives the caller no size to reserve,
/// and a memory space means the buffer is not the caller's to allocate, so both keep returning a
/// descriptor. (#643)
pub(crate) fn returns_through_slot(ty_text: &str) -> bool {
    ty_text.starts_with("memref<") && !ty_text.contains('?') && !ty_text.contains(',')
}

/// The width in bits of a scalar MLIR type, so `i8` gives 8. `None` for anything with no
/// width of its own, such as a pointer or an aggregate.
pub(crate) fn scalar_type_bits(ty_text: &str) -> Option<u32> {
    if let Some(bits) = ty_text.strip_prefix('i').and_then(|r| r.parse().ok()) {
        return Some(bits);
    }
    match ty_text {
        "f16" | "bf16" => Some(16),
        "f32" => Some(32),
        "f64" => Some(64),
        _ => None,
    }
}

/// Whether an MLIR type spelling is a float type.
pub(crate) fn is_float_text(ty_text: &str) -> bool {
    matches!(ty_text, "f16" | "bf16" | "f32" | "f64")
}

/// Register every MLIR dialect with `context` and load them, so their types and operations
/// can be parsed and built.
fn load_all_dialects(context: &Context) {
    let registry = DialectRegistry::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
}

/// The `extern` declarations of the program and then of each module, without the repeats:
/// two declarations with the same name and the same parameters count as one.
fn unique_externs(program: &Program, sorted_modules: &[&Program]) -> Vec<syntax::ExternDecl> {
    let mut all_externs = program.externs.clone();
    for module_prog in sorted_modules.iter().copied() {
        all_externs.extend(module_prog.externs.clone());
    }

    let mut seen_externs = std::collections::HashSet::new();
    let mut unique_externs = Vec::new();
    for ext in all_externs {
        let sig = format!("{}: {:?}", ext.name, ext.params);
        if seen_externs.insert(sig) {
            unique_externs.push(ext);
        }
    }
    unique_externs
}

/// A drop waiting for the value of the statement it precedes.
pub(crate) enum WaitingDrop<'c> {
    /// The buffer to free, read where the drop was: the owner, or (`field`) a tensor field of
    /// a struct being dropped.
    Tensor {
        buffer: melior::ir::Value<'c, 'c>,
        flag: Option<crate::symbol::Symbol>,
        field: bool,
    },
    /// The statement that calls a `Drop` impl's `drop`.
    Call(Box<Statement>),
}

impl<'c> MeliorGenerator<'c> {
    /// Start a block: a `let` in it hides an outer variable of the same name only until the
    /// block ends, as in Rust. Returns the depth `close_blocks_to` takes it back to.
    pub(crate) fn open_block(&mut self) -> usize {
        self.blocks.push(Vec::new());
        self.blocks.len() - 1
    }

    /// End every block opened above `depth`, restoring what the names its `let`s shadowed meant.
    ///
    /// # Panics
    ///
    /// Never in practice: a frame is only taken while more than `depth` of them are open.
    pub(crate) fn close_blocks_to(&mut self, depth: usize) {
        while self.blocks.len() > depth {
            let frame = self.blocks.pop().expect("a frame above depth");
            for s in frame {
                match s.value {
                    Some(v) => self.env.insert(s.name.clone(), v),
                    None => self.env.remove(&s.name),
                };
                match s.ast_ty {
                    Some(t) => self.ast_env.insert(s.name.clone(), t),
                    None => self.ast_env.remove(&s.name),
                };
                if s.in_a_slot {
                    self.allocs.insert(s.name.to_string());
                } else {
                    self.allocs.remove(s.name.as_ref());
                }
                if s.owned {
                    self.owned_tensors.insert(s.name);
                } else {
                    self.owned_tensors.remove(&s.name);
                }
            }
        }
    }

    /// Called before a `let` or a pattern binds `name`: the first time in a block, remember what
    /// the name meant, for `close_blocks_to` to put back.
    ///
    /// # Panics
    ///
    /// Never in practice: the function returns early when no block is open.
    pub(crate) fn note_shadow(&mut self, name: &str) {
        let Some(frame) = self.blocks.last() else {
            return;
        };
        if frame.iter().any(|s| s.name.as_ref() == name) {
            return;
        }
        let name: crate::symbol::Symbol = name.into();
        let before = Shadowed {
            value: self.env.get(&name).copied(),
            in_a_slot: self.allocs.contains(name.as_ref()),
            ast_ty: self.ast_env.get(&name).cloned(),
            owned: self.owned_tensors.contains(&name),
            name,
        };
        self.blocks
            .last_mut()
            .expect("a frame, checked above")
            .push(before);
    }

    /// The MLIR location of the statement or expression being lowered: file, line and column.
    #[must_use]
    pub fn loc(&self) -> melior::ir::Location<'c> {
        melior::ir::Location::new(
            self.context,
            &self.current_filename,
            self.current_span.line,
            self.current_span.column,
        )
    }

    /// The MLIR signature a Vx function is emitted with, given its declared one.
    ///
    /// A statically shaped tensor result is not a result at all: it is a buffer the caller
    /// allocated and passed as the first parameter, so the function returns nothing. Every
    /// consumer -- the definition, a call, a function constant -- goes through here, because a
    /// caller that disagreed with the definition about this would build a call MLIR rejects.
    pub(crate) fn abi_signature(
        &self,
        ret_ty: Type<'c>,
        arg_tys: &[Type<'c>],
    ) -> (Option<Type<'c>>, Vec<Type<'c>>) {
        // A `void` function has no result, both in its definition and in the type of its address.
        if ret_ty == self.none_ty {
            return (None, arg_tys.to_vec());
        }
        if !returns_through_slot(&ret_ty.to_string()) {
            return (Some(ret_ty), arg_tys.to_vec());
        }
        let mut with_slot = Vec::with_capacity(arg_tys.len() + 1);
        with_slot.push(ret_ty);
        with_slot.extend_from_slice(arg_tys);
        (None, with_slot)
    }

    /// Whether an MLIR type is a `memref`, which is how a tensor or a mutable local is held.
    #[must_use]
    pub fn is_memref(&self, ty: &Type<'c>) -> bool {
        let ty_str = ty.to_string();
        ty_str.starts_with("memref<")
    }

    /// A rank-0 `memref` over a single scalar (`memref<i32>`), which is how a mutable scalar
    /// local's storage and a borrow of one are both spelled. Ranked memrefs (`memref<4xf32>`)
    /// and nested ones (`memref<memref<i32>>`) are different shapes and are excluded.
    #[must_use]
    pub fn is_scalar_memref(&self, ty: &Type<'c>) -> bool {
        let ty_str = ty.to_string();
        ty_str.starts_with("memref<")
            && !ty_str.contains('x')
            && !ty_str.starts_with("memref<memref<")
    }

    /// Adapt a closure-literal environment struct (`Closure_N`, the by-value capture record) to a
    /// nominal stdlib closure struct (`Closure0/1/2/3<Args.., Ret>`, laid out `{env: *mut i8, func}`).
    /// A closure passed where an API takes a `ClosureK` (e.g. `VecIter::map`'s `f: Closure1<T,NewItem>`)
    /// reaches codegen as a `Closure_N` value; the two share the `{ptr, ptr}` shape but not the
    /// fields, so we materialize `{env: &spilled_env, func: &Closure_N_call}` here. Returns `None`
    /// when the pair isn't a `Closure_N` -> `ClosureK` adaptation, so the caller falls back to its
    /// normal coercion. See `check_closure_expr` (produces `Closure_N`) and closure.vx (`ClosureK`).
    ///
    /// # Errors
    ///
    /// Fails when the closure's call function is not registered, or when MLIR rejects one of
    /// the operations built here.
    pub(crate) fn adapt_closure_to_nominal(
        &mut self,
        block: &melior::ir::Block<'c>,
        env_val: Value<'c, 'c>,
        env_ty: Type<'c>,
        target_ty: Type<'c>,
    ) -> Result<Option<Value<'c, 'c>>, crate::codegen::lower::LowerError> {
        let env_str = env_ty.to_string();
        let target_str = target_ty.to_string();
        // Source must be a generated closure-env struct; target a nominal ClosureK (digit after
        // "Closure"). Both are `!llvm.struct<"NAME", (...)>`.
        let env_name = match parse_llvm_struct_name(&env_str) {
            Some(n) if n.starts_with("Closure_") => n,
            _ => return Ok(None),
        };
        let is_nominal_closure = parse_llvm_struct_name(&target_str).is_some_and(|n| {
            n.starts_with("Closure") && n[7..].chars().next().is_some_and(|c| c.is_ascii_digit())
        });
        if !is_nominal_closure {
            return Ok(None);
        }

        let fn_ptr = self.closure_call_fn_ptr(block, &env_name)?;
        let env_slot = self.spill_to_stack(block, env_val, env_ty)?;

        // Build the nominal `{env, func}` struct: field 0 = env ptr, field 1 = func ptr.
        let undef_op = melior::ir::operation::OperationBuilder::new("llvm.mlir.undef", self.loc())
            .add_results(&[target_ty])
            .build()?;
        let mut agg: Value<'c, 'c> = block.append_operation(undef_op).result(0)?.into();
        for (pos, field_val) in [(0_i64, env_slot), (1, fn_ptr)] {
            let ins_op =
                melior::ir::operation::OperationBuilder::new("llvm.insertvalue", self.loc())
                    .add_operands(&[agg, field_val])
                    .add_attributes(&[(
                        melior::ir::Identifier::new(self.context, "position"),
                        melior::ir::attribute::DenseI64ArrayAttribute::new(self.context, &[pos])
                            .into(),
                    )])
                    .add_results(&[target_ty])
                    .build()?;
            agg = block.append_operation(ins_op).result(0)?.into();
        }
        Ok(Some(agg))
    }

    /// The address of a closure's call function, `Closure_N_call(_env, args..) -> ret`, as an
    /// opaque pointer that can be stored in a struct field.
    ///
    /// # Errors
    ///
    /// Fails when the call function is not registered, or when MLIR rejects an operation.
    fn closure_call_fn_ptr(
        &self,
        block: &melior::ir::Block<'c>,
        env_name: &str,
    ) -> Result<Value<'c, 'c>, crate::codegen::lower::LowerError> {
        let call_fn_name = format!("{env_name}_call");
        let (ret_ty, arg_types) = self
            .functions
            .get(call_fn_name.as_str())
            .cloned()
            .ok_or_else(|| format!("closure call fn {call_fn_name} not found"))?;
        let fn_ty = melior::ir::r#type::FunctionType::new(self.context, &arg_types, &[ret_ty]);
        let const_op = melior::ir::operation::OperationBuilder::new("func.constant", self.loc())
            .add_attributes(&[(
                melior::ir::Identifier::new(self.context, "value"),
                melior::ir::attribute::FlatSymbolRefAttribute::new(self.context, &call_fn_name)
                    .into(),
            )])
            .add_results(&[fn_ty.into()])
            .build()?;
        let fn_val: Value<'c, 'c> = block.append_operation(const_op).result(0)?.into();
        // Function value -> opaque ptr for storage in the struct's `func` field.
        let fn_ptr_op = melior::ir::operation::OperationBuilder::new(
            "builtin.unrealized_conversion_cast",
            self.loc(),
        )
        .add_operands(&[fn_val])
        .add_results(&[self.ptr_ty])
        .build()?;
        Ok(block.append_operation(fn_ptr_op).result(0)?.into())
    }

    /// Spill a struct value to a new stack slot and return the slot's address. A closure's
    /// captured values live there.
    ///
    /// # Errors
    ///
    /// Fails when MLIR rejects one of the operations built here.
    fn spill_to_stack(
        &self,
        block: &melior::ir::Block<'c>,
        env_val: Value<'c, 'c>,
        env_ty: Type<'c>,
    ) -> Result<Value<'c, 'c>, crate::codegen::lower::LowerError> {
        let ptr_ty = self.ptr_ty;
        let i32_ty = self.i32_ty;
        let one_op = melior::ir::operation::OperationBuilder::new("llvm.mlir.constant", self.loc())
            .add_results(&[i32_ty])
            .add_attributes(&[(
                melior::ir::Identifier::new(self.context, "value"),
                melior::ir::attribute::IntegerAttribute::new(i32_ty, 1).into(),
            )])
            .build()?;
        let one: Value<'c, 'c> = block.append_operation(one_op).result(0)?.into();
        let alloca_op = melior::ir::operation::OperationBuilder::new("llvm.alloca", self.loc())
            .add_operands(&[one])
            .add_results(&[ptr_ty])
            .add_attributes(&[(
                melior::ir::Identifier::new(self.context, "elem_type"),
                melior::ir::attribute::TypeAttribute::new(env_ty).into(),
            )])
            .build()?;
        let slot: Value<'c, 'c> = block.append_operation(alloca_op).result(0)?.into();
        block.append_operation(
            melior::ir::operation::OperationBuilder::new("llvm.store", self.loc())
                .add_operands(&[env_val, slot])
                .build()?,
        );
        Ok(slot)
    }

    /// Whether an MLIR type is one of the float types: `f16`, `bf16`, `f32` or `f64`.
    fn is_float_ty(&self, t: Type<'c>) -> bool {
        t == self.f32_ty || t == self.f64_ty || t == self.f16_ty || t == self.bf16_ty
    }

    /// Append an operation named `op_name` that takes `val` and gives one result of type
    /// `to_ty`, and return that result.
    ///
    /// # Errors
    ///
    /// Fails when the appended operation has no result to read.
    ///
    /// # Panics
    ///
    /// Panics when MLIR cannot build the operation, which means `op_name` is not a real one.
    fn emit_cast(
        &self,
        block: &melior::ir::Block<'c>,
        op_name: &str,
        val: Value<'c, 'c>,
        to_ty: Type<'c>,
    ) -> Result<Value<'c, 'c>, crate::codegen::lower::LowerError> {
        let cast_op = melior::ir::operation::OperationBuilder::new(op_name, self.loc())
            .add_operands(&[val])
            .add_results(&[to_ty])
            .build()
            .unwrap();
        Ok(block.append_operation(cast_op).result(0)?.into())
    }

    /// The `operandSegmentSizes` attribute, which tells MLIR how many operands are in each
    /// group of an operation that takes several groups.
    fn operand_segment_sizes(
        &self,
        sizes: &[i32],
    ) -> (melior::ir::Identifier<'c>, melior::ir::Attribute<'c>) {
        (
            melior::ir::Identifier::new(self.context, "operandSegmentSizes"),
            melior::ir::attribute::DenseI32ArrayAttribute::new(self.context, sizes).into(),
        )
    }

    /// Scalar -> tensor: broadcast the scalar across a fresh buffer (fill), rather than
    /// emitting an invalid f32->memref bitcast. Handles static, identity-layout memrefs whose
    /// element type matches the scalar. Returns `None` for any other pair of types.
    ///
    /// # Panics
    ///
    /// Panics when MLIR cannot build the `memref.alloc` or `linalg.fill` operation, which
    /// means the types checked above do not fit those operations.
    fn fill_tensor_with_scalar(
        &self,
        block: &melior::ir::Block<'c>,
        val: Value<'c, 'c>,
        from_ty: Type<'c>,
        to_ty: Type<'c>,
    ) -> Option<Value<'c, 'c>> {
        if !self.is_memref(&to_ty) || self.is_memref(&from_ty) {
            return None;
        }
        let to_str = to_ty.to_string();
        let el = to_str
            .trim_start_matches("memref<")
            .split('x')
            .next_back()
            .unwrap_or("")
            .trim_end_matches('>');
        if !to_str.starts_with("memref<")
            || to_str.contains('?')
            || to_str.contains('*')
            || el != from_ty.to_string()
        {
            return None;
        }
        let alloc_op = melior::ir::operation::OperationBuilder::new("memref.alloc", self.loc())
            .add_attributes(&[self.operand_segment_sizes(&[0, 0])])
            .add_results(&[to_ty])
            .build()
            .unwrap();
        let dst: Value<'c, 'c> = block.append_operation(alloc_op).result(0).unwrap().into();

        let region = melior::ir::Region::new();
        let fblock = melior::ir::Block::new(&[(from_ty, self.loc()), (from_ty, self.loc())]);
        let yield_op = melior::ir::operation::OperationBuilder::new("linalg.yield", self.loc())
            .add_operands(&[fblock.argument(0).unwrap().into()])
            .build()
            .unwrap();
        fblock.append_operation(yield_op);
        region.append_block(fblock);

        let fill_op = melior::ir::operation::OperationBuilder::new("linalg.fill", self.loc())
            .add_operands(&[val, dst])
            .add_attributes(&[self.operand_segment_sizes(&[1, 1])])
            .add_regions([region])
            .build()
            .unwrap();
        block.append_operation(fill_op);
        Some(dst)
    }

    /// The single operation that converts a number of type `from_ty` to `to_ty`, for the
    /// pairs of `i32`, `i64` and the float types that have one. `None` for any other pair.
    fn number_cast_op(&self, from_ty: Type<'c>, to_ty: Type<'c>) -> Option<&'static str> {
        let half = |t: Type<'c>| t == self.bf16_ty || t == self.f16_ty;
        if from_ty == self.i32_ty && to_ty == self.i64_ty {
            return Some("arith.extsi");
        }
        if from_ty == self.i64_ty && to_ty == self.i32_ty {
            return Some("arith.trunci");
        }
        if from_ty == self.f32_ty && to_ty == self.f64_ty {
            return Some("arith.extf");
        }
        if from_ty == self.f32_ty && half(to_ty) {
            return Some("arith.truncf");
        }
        if half(from_ty) && to_ty == self.f32_ty {
            return Some("arith.extf");
        }
        if (from_ty == self.f64_ty || from_ty == self.f32_ty)
            && (to_ty == self.f32_ty || half(to_ty))
        {
            return Some("arith.truncf");
        }
        if half(from_ty) && (to_ty == self.f32_ty || to_ty == self.f64_ty) {
            return Some("arith.extf");
        }
        if (from_ty == self.i32_ty || from_ty == self.i64_ty) && self.is_float_ty(to_ty) {
            return Some("arith.sitofp");
        }
        if self.is_float_ty(from_ty) && (to_ty == self.i32_ty || to_ty == self.i64_ty) {
            return Some("arith.fptosi");
        }
        None
    }

    /// The single operation that converts between two integer types, or between an integer
    /// type and `index`, going by how the types are spelled. `None` when either type is not
    /// an integer, or when a width cannot be read from its spelling.
    fn integer_cast_op(&self, from_ty: Type<'c>, to_ty: Type<'c>) -> Option<&'static str> {
        let from_str = from_ty.to_string();
        let to_str = to_ty.to_string();
        let is_int = |s: &str| s.starts_with('i') || s.starts_with('u');
        if (from_ty == self.index_ty && is_int(&to_str))
            || (is_int(&from_str) && to_ty == self.index_ty)
        {
            return Some("arith.index_cast");
        }
        if !is_int(&from_str) || !is_int(&to_str) {
            return None;
        }
        let from_width: u32 = from_str[1..].parse().unwrap_or(0);
        let to_width: u32 = to_str[1..].parse().unwrap_or(0);
        if from_width == 0 || to_width == 0 {
            return None;
        }
        Some(if from_width > to_width {
            "arith.trunci"
        } else if from_str.starts_with('u') {
            "arith.extui"
        } else {
            "arith.extsi"
        })
    }

    /// Convert `val` from the MLIR type `from_ty` to `to_ty`, appending the operations that
    /// do it to `block`. Returns `val` itself when it already has the wanted type.
    ///
    /// A pair of types with no known conversion falls back to
    /// `builtin.unrealized_conversion_cast`.
    ///
    /// # Errors
    ///
    /// Fails when an appended operation has no result to read.
    ///
    /// # Panics
    ///
    /// Panics when MLIR cannot build a conversion operation, which means the two types do
    /// not fit the operation chosen for them.
    pub fn coerce_type(
        &mut self,
        block: &melior::ir::Block<'c>,
        val: Value<'c, 'c>,
        from_ty: Type<'c>,
        to_ty: Type<'c>,
    ) -> Result<Value<'c, 'c>, crate::codegen::lower::LowerError> {
        if from_ty == to_ty {
            return Ok(val);
        }

        if let Some(filled) = self.fill_tensor_with_scalar(block, val, from_ty, to_ty) {
            return Ok(filled);
        }
        if let Some(op_name) = self.number_cast_op(from_ty, to_ty) {
            return self.emit_cast(block, op_name, val, to_ty);
        }

        // `index` has no direct fp conversion, so route through i64. This
        // is the loop-index-in-float-context path: `for i in 0..n { i as f32 }`,
        // where `i` is lowered as MLIR `index`. Without this the fallback emits a
        // bitcast (unrealized_conversion_cast index->f32) that survives LLVM
        // lowering as an unreconciled `i64 to index` cast and fails translation.
        if from_ty == self.index_ty && self.is_float_ty(to_ty) {
            let i64_val = self.emit_cast(block, "arith.index_cast", val, self.i64_ty)?;
            return self.emit_cast(block, "arith.sitofp", i64_val, to_ty);
        }
        if self.is_float_ty(from_ty) && to_ty == self.index_ty {
            let i64_val = self.emit_cast(block, "arith.fptosi", val, self.i64_ty)?;
            return self.emit_cast(block, "arith.index_cast", i64_val, to_ty);
        }
        if let Some(op_name) = self.integer_cast_op(from_ty, to_ty) {
            return self.emit_cast(block, op_name, val, to_ty);
        }

        if val.r#type() == to_ty {
            return Ok(val);
        }

        let from_str = from_ty.to_string();
        let to_str = to_ty.to_string();
        // A scalar-`memref` reference coerced to a bare `!llvm.ptr` — passing `&x` for a *mutable* scalar
        // local (a `memref<i32>` descriptor) to a `&i32` parameter (which lowers to `!llvm.ptr`). A raw
        // bitcast of the multi-word descriptor to a single pointer produces an `unrealized_conversion_cast`
        // that fails LLVM translation. Extract the memref's aligned data pointer instead, matching how a
        // nested `&mut r` is lowered — the resulting `!llvm.ptr` aliases the same storage.
        if to_ty == self.ptr_ty
            && from_str.starts_with("memref<")
            && !from_str.contains('x')
            && !from_str.starts_with("memref<memref<")
        {
            let aligned = "memref.extract_aligned_pointer_as_index";
            let raw_idx = self.emit_cast(block, aligned, val, self.index_ty)?;
            let raw_i64 = self.emit_cast(block, "arith.index_cast", raw_idx, self.i64_ty)?;
            return self.emit_cast(block, "llvm.inttoptr", raw_i64, self.ptr_ty);
        }

        if cfg!(debug_assertions) {
            println!(
                "Warning: Falling back to bitcast from {} to {}\nBacktrace:\n{:?}",
                from_str,
                to_str,
                std::backtrace::Backtrace::force_capture()
            );
        }
        // Default to unrealized_conversion_cast if nothing matches but we need a cast
        self.emit_cast(block, "builtin.unrealized_conversion_cast", val, to_ty)
    }

    /// A generator with an empty module, for the source file `filename`. Loads every MLIR
    /// dialect into `context` first.
    ///
    /// # Panics
    ///
    /// Panics when MLIR cannot parse the name of a built-in type such as `i32`, which means
    /// the MLIR libraries are broken.
    #[must_use]
    pub fn new(context: &'c Context, filename: String) -> Self {
        let builtin = |name: &str| Type::parse(context, name).unwrap();
        load_all_dialects(context);

        Self {
            context,
            module: Module::new(Location::unknown(context)),
            env: HashMap::new(),
            ast_env: HashMap::new(),
            owned_tensors: std::collections::HashSet::new(),
            declared_tensor_prints: std::collections::HashSet::new(),
            pointer_taken: std::collections::HashSet::new(),
            address_taken: std::collections::HashSet::new(),
            memories: HashMap::new(),
            transfer_impls: HashMap::new(),
            topologies: HashMap::new(),
            subspace_offsets: HashMap::new(),
            structs: HashMap::new(),
            enums: HashMap::new(),
            enum_generics: HashMap::new(),
            functions: HashMap::new(),
            syntax_functions: HashMap::new(),
            enzyme_decls: std::collections::HashSet::new(),
            string_counter: 0,
            const_tables: HashMap::new(),
            current_return_type: None,
            current_return_slot: None,
            nrvo_slot: None,
            queued_drops: Vec::new(),
            value_drops: Vec::new(),
            expected_type: None,
            in_spawn: false,
            break_blocks: Vec::new(),
            continue_blocks: Vec::new(),
            allocs: std::collections::HashSet::new(),
            blocks: Vec::new(),
            is_lvalue_context: false,
            mlir_block_counter: 0,
            has_returned: false,
            current_filename: filename,
            di_compile_unit: None,
            di_subprogram: None,
            current_span: syntax::types::Span {
                line: 1,
                column: 1,
                length: 1,
            },
            index_ty: builtin("index"),
            i32_ty: builtin("i32"),
            i64_ty: builtin("i64"),
            f32_ty: builtin("f32"),
            f64_ty: builtin("f64"),
            f16_ty: builtin("f16"),
            bf16_ty: builtin("bf16"),
            i1_ty: builtin("i1"),
            ptr_ty: builtin("!llvm.ptr"),
            none_ty: builtin("none"),
            emit_seam_certs: false,
            assert_facts: Vec::new(),
        }
    }

    /// Give up the generator and keep the MLIR module it built.
    #[must_use]
    pub fn into_module(self) -> Module<'c> {
        self.module
    }

    /// Generate the MLIR for `program` and the modules it imports, and return it as text.
    ///
    /// # Errors
    ///
    /// Fails when a type or a statement of the program cannot be lowered to MLIR.
    ///
    /// # Panics
    ///
    /// Panics when the list of local devices does not parse as an MLIR attribute, which
    /// means `local_devices_attr` wrote it wrongly.
    pub fn generate(
        &mut self,
        program: &Program,
        modules: &HashMap<crate::symbol::Symbol, Program>,
    ) -> Result<String, LowerError> {
        use melior::ir::operation::OperationMutLike;
        let location = self.loc();
        self.module = melior::ir::Module::new(location);

        let di_file_str = format!("#llvm.di_file<\"{}\" in \"\">", self.current_filename);
        let di_cu_str = format!(
            "#llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_C, file = {di_file_str}, producer = \"vx\", isOptimized = false, emissionKind = Full>"
        );
        self.di_compile_unit = melior::ir::Attribute::parse(self.context, &di_cu_str);

        // Add llvm.module_flags
        let module_flags_attr = melior::ir::Attribute::parse(
            self.context,
            "{\"Dwarf Version\" = 4 : i32, \"Debug Info Version\" = 3 : i32}",
        );
        if let Some(flags) = module_flags_attr {
            self.module
                .as_operation_mut()
                .set_attribute("llvm.module_flags", flags);
        }
        // The program frees its tensors at their drops; see the flat code generator.
        self.module
            .as_operation_mut()
            .set_attribute("vx.drops", melior::ir::Attribute::unit(self.context));

        self.generate_module(program, modules)?;

        // After `generate_module`, which is what fills `self.topologies`.
        let local_devices = crate::arch::local_device_aliases(self.topologies.iter());
        let attr = crate::codegen::flat::local_devices_attr(&local_devices);
        if let Some(value) = attr.strip_prefix(", vx.local_devices = ") {
            let value = melior::ir::Attribute::parse(self.context, value)
                .expect("vx.local_devices is a valid array attribute");
            self.module
                .as_operation_mut()
                .set_attribute("vx.local_devices", value);
        }

        let op = self.module.as_operation();
        let s = format!("{op}");
        Ok(s)
    }

    /// A read-only global for each `const` table in `programs`, in name order.
    ///
    /// # Errors
    ///
    /// Fails when MLIR rejects the `memref.global` operation of a table.
    ///
    /// # Panics
    ///
    /// Panics when a table's type or values do not parse as MLIR, which means the table
    /// was built from a declaration the checker should have rejected.
    fn add_const_tables<'p>(
        &mut self,
        programs: impl Iterator<Item = &'p Program>,
    ) -> Result<(), LowerError> {
        use melior::ir::attribute::{Attribute, StringAttribute, TypeAttribute};
        use melior::ir::operation::OperationBuilder;
        use melior::ir::Identifier;
        for c in programs.flat_map(|p| &p.consts) {
            if let Ok(Some(table)) = crate::registry::ConstTable::from_decl(c) {
                self.const_tables.insert(c.name.clone(), table);
            }
        }
        let mut tables: Vec<_> = self.const_tables.values().collect();
        tables.sort_by(|a, b| a.name.cmp(&b.name));
        for table in tables {
            let memty = Type::parse(self.context, &table.memref_type())
                .expect("a table's memref type parses");
            let elem = crate::mlir_ty::mlir_scalar(&table.elem)
                .expect("a table's element has an MLIR type");
            let init = format!(
                "dense<[{}]> : tensor<{}x{}>",
                table.values.join(", "),
                table.len,
                elem
            );
            let attr =
                |name: &str, value: Attribute<'c>| (Identifier::new(self.context, name), value);
            let global = OperationBuilder::new("memref.global", self.loc())
                .add_attributes(&[
                    attr(
                        "sym_name",
                        StringAttribute::new(self.context, &table.symbol()).into(),
                    ),
                    attr(
                        "sym_visibility",
                        StringAttribute::new(self.context, "private").into(),
                    ),
                    attr("type", TypeAttribute::new(memty).into()),
                    attr(
                        "initial_value",
                        Attribute::parse(self.context, &init).expect("a table's values parse"),
                    ),
                    attr("constant", Attribute::parse(self.context, "unit").unwrap()),
                ])
                .build()?;
            self.module.body().append_operation(global);
        }
        Ok(())
    }

    /// Record what `program` declares that later lowering looks up by name: its structs,
    /// enums, memories, topologies and transfer lowerings.
    fn register_declarations(&mut self, program: &Program) {
        for s in &program.structs {
            self.structs.insert(s.name.clone(), s.clone());
        }
        for e in &program.enums {
            self.enums.insert(e.name.clone(), e.variants.clone());
            self.enum_generics.insert(
                e.name.clone(),
                e.generics.iter().map(|g| g.name().into()).collect(),
            );
        }
        for m in &program.memories {
            self.memories
                .insert(syntax::MemorySpace::from_name(m.name.as_ref()), m.clone());
        }
        for t in &program.topologies {
            self.topologies.insert(t.name.clone(), t.descriptor.clone());
        }
        for li in &program.transfer_impls {
            if li.methods.len() == 1 {
                self.transfer_impls.insert(
                    (
                        li.from.name(),
                        li.to.name(),
                        li.topology.display_name().clone(),
                    ),
                    li.methods[0].clone(),
                );
            }
        }
    }

    /// Record the lowered signature of each `extern` function `program` declares.
    ///
    /// # Errors
    ///
    /// Fails when a parameter type or the return type of an `extern` cannot be lowered.
    fn register_externs(&mut self, program: &Program) -> Result<(), LowerError> {
        for ext in &program.externs {
            let ret_ty = self.lower_type(&ext.return_type)?;
            let mut arg_tys = Vec::new();
            for (_, ty) in &ext.params {
                arg_tys.push(self.lower_type(ty)?);
            }
            self.functions.insert(ext.name.clone(), (ret_ty, arg_tys));
        }
        Ok(())
    }

    /// Record the lowered signature of one function and keep its declaration, so a call to
    /// it can be lowered before its body is.
    ///
    /// # Errors
    ///
    /// Fails when a parameter type or the return type of the function cannot be lowered.
    fn register_function(&mut self, func: &Function) -> Result<(), LowerError> {
        let ret_ty = self.lower_type(&func.return_type)?;
        let mut arg_tys = Vec::new();
        for (_, ty) in &func.params {
            arg_tys.push(self.lower_type(ty)?);
        }
        self.functions.insert(func.name.clone(), (ret_ty, arg_tys));
        self.syntax_functions
            .insert(func.name.clone(), func.clone());
        Ok(())
    }

    /// Declare `vx_init_signals`, the runtime function that `main` calls first.
    ///
    /// # Errors
    ///
    /// Fails when MLIR rejects the declaration.
    fn declare_init_signals(&self) -> Result<(), LowerError> {
        let sig_init_ty = melior::ir::r#type::FunctionType::new(self.context, &[], &[]);
        let sig_init_decl = melior::ir::operation::OperationBuilder::new("func.func", self.loc())
            .add_attributes(&[
                (
                    melior::ir::Identifier::new(self.context, "sym_name"),
                    melior::ir::attribute::StringAttribute::new(self.context, "vx_init_signals")
                        .into(),
                ),
                (
                    melior::ir::Identifier::new(self.context, "function_type"),
                    melior::ir::attribute::TypeAttribute::new(sig_init_ty.into()).into(),
                ),
                (
                    melior::ir::Identifier::new(self.context, "sym_visibility"),
                    melior::ir::attribute::StringAttribute::new(self.context, "private").into(),
                ),
            ])
            .add_regions([melior::ir::Region::new()])
            .build()?;
        self.module.body().append_operation(sig_init_decl);
        Ok(())
    }

    /// Declare the `extern` functions the generated code can call. `referenced` holds every
    /// name the kept functions refer to.
    ///
    /// # Errors
    ///
    /// Fails when an `extern` has no recorded signature, or when MLIR rejects a declaration.
    fn declare_externs(
        &self,
        program: &Program,
        sorted_modules: &[&Program],
        referenced: &std::collections::HashSet<String>,
    ) -> Result<(), LowerError> {
        for ext in &unique_externs(program, sorted_modules) {
            let name = &ext.name;
            if name.as_ref() == "printf" || **name == *"vx_internal_printf" {
                continue;
            }
            // An imported module's extern is declared only when a function calls it.
            let own = program.externs.iter().any(|e| e.name == *name);
            if !own && !referenced.contains(name.as_ref()) {
                continue;
            }
            self.declare_extern(name)?;
        }
        Ok(())
    }

    /// Declare one `extern` function in the module, with the signature recorded for `name`.
    ///
    /// # Errors
    ///
    /// Fails when no signature is recorded for `name`, or when MLIR rejects the declaration.
    ///
    /// # Panics
    ///
    /// Panics when the `extern` returns a tensor with a fixed shape by value: the C rule for
    /// returning such a value is not modelled.
    fn declare_extern(&self, name: &crate::symbol::Symbol) -> Result<(), LowerError> {
        let (ret_ty, arg_tys) = self.functions.get(name).ok_or_else(|| {
            crate::codegen::lower::LowerError::from(format!("Function not found: {name}"))
        })?;

        // An `extern` keeps the C convention: its return type is whatever the foreign
        // function actually returns, never a slot we invented. Nothing in the stdlib returns
        // an aggregate from C today, and guessing at the platform's rule for one would
        // produce a call that links and is wrong.
        assert!(
            !returns_through_slot(&ret_ty.to_string()),
            "extern '{name}' returns a tensor by value; the C return convention for an \
             aggregate is not modelled (#643)"
        );
        let mut actual_ret_tys = Vec::new();
        if ret_ty.to_string() != "none" {
            actual_ret_tys.push(*ret_ty);
        }

        // FunctionType::new takes arg_tys and ret_tys
        let func_type =
            melior::ir::r#type::FunctionType::new(self.context, arg_tys, &actual_ret_tys);

        // Define the string attribute for the function name
        let name_attr = melior::ir::attribute::StringAttribute::new(self.context, name);
        let type_attr = melior::ir::attribute::TypeAttribute::new(func_type.into());

        let builder = melior::ir::operation::OperationBuilder::new("func.func", self.loc())
            .add_attributes(&[
                (
                    melior::ir::Identifier::new(self.context, "sym_name"),
                    name_attr.into(),
                ),
                (
                    melior::ir::Identifier::new(self.context, "function_type"),
                    type_attr.into(),
                ),
                (
                    melior::ir::Identifier::new(self.context, "sym_visibility"),
                    melior::ir::attribute::StringAttribute::new(self.context, "private").into(),
                ),
            ]);

        let region = melior::ir::Region::new();
        let func_op = builder.add_regions([region]).build()?;

        self.module.body().append_operation(func_op);
        Ok(())
    }

    /// Build the whole module: record every declaration of `program` and of the modules it
    /// imports, generate each function, and declare the `extern` functions they call.
    ///
    /// # Errors
    ///
    /// Fails when a type, a function or a declaration cannot be lowered to MLIR.
    ///
    /// # Panics
    ///
    /// Panics when a `const` table does not parse as MLIR, or when an `extern` returns a
    /// tensor with a fixed shape by value.
    pub(crate) fn generate_module(
        &mut self,
        program: &Program,
        modules: &HashMap<crate::symbol::Symbol, Program>,
    ) -> Result<(), LowerError> {
        // Walked in name order rather than straight off `modules.values()`. A HashMap iterates in
        // an order that depends on the process's random hash seed, and the loops below decide the
        // order function bodies and extern declarations are emitted in -- so the same program
        // compiled twice came out with its functions in a different order.
        let sorted_modules: Vec<&Program> = {
            let mut names: Vec<&crate::symbol::Symbol> = modules.keys().collect();
            names.sort();
            names.into_iter().map(|n| &modules[n]).collect()
        };
        self.add_const_tables(std::iter::once(program).chain(sorted_modules.iter().copied()))?;
        self.register_declarations(program);
        // An imported module's memories and topologies are recorded too. `--machine` loads
        // the SKU as a peer *module* (driver.rs), so taking these from the main program alone
        // meant a machine model contributed nothing here: a `vx.transfer` came out carrying
        // `space` and a cost but no `capacity` and no `scope`, because the lookup that adds
        // them found no declaration.
        //
        // Nothing failed. The attributes a device backend reads to place a tile were simply
        // absent, and a lowering keyed on `scope` did the right thing for a program compiled
        // on the flat path -- which does merge them -- and quietly nothing for the same
        // program on the AST path.
        for module in sorted_modules.iter().copied() {
            self.register_declarations(module);
        }
        self.register_externs(program)?;
        self.declare_init_signals()?;
        for module_prog in sorted_modules.iter().copied() {
            self.register_externs(module_prog)?;
        }

        let mut operations = Vec::new();

        for module_prog in sorted_modules.iter().copied() {
            // Skip generic *templates* — only their concrete instantiations (in `program.functions`,
            // collected as monomorphizations) are codegen'd. Imported modules now retain their
            // generic free functions so the env can instantiate cross-module generic calls (#204).
            for func in module_prog
                .functions
                .iter()
                .filter(|f| f.generics.is_empty())
            {
                self.register_function(func)?;
            }
        }

        for func in &program.functions {
            self.register_function(func)?;
        }

        // Emit module functions (concrete only; generic templates are skipped — see above).
        for module_prog in sorted_modules.iter().copied() {
            for func in module_prog
                .functions
                .iter()
                .filter(|f| f.generics.is_empty())
            {
                operations.push(self.generate_function(func)?);
            }
        }
        let imported = operations.len();

        for func in &program.functions {
            operations.push(self.generate_function(func)?);
        }
        let (operations, referenced) = keep_reached_functions(operations, imported);

        self.declare_externs(program, &sorted_modules, &referenced)?;

        let body = self.module.body();
        for op in operations {
            body.append_operation(op);
        }
        Ok(())
    }

    /// The MLIR location of a function's definition. It carries the debug entry (the
    /// `DISubprogram`) that a debugger uses to name the function. Also records that entry as
    /// the one of the function being lowered.
    ///
    /// # Errors
    ///
    /// Fails when MLIR cannot parse the small module the location is read from.
    fn function_debug_location(
        &mut self,
        func: &Function,
    ) -> Result<melior::ir::Location<'c>, LowerError> {
        use melior::ir::operation::OperationLike;
        use melior::ir::BlockLike;
        let di_file_str = format!("#llvm.di_file<\"{}\" in \"\">", self.current_filename);
        let di_cu_str = format!(
            "#llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_C, file = {di_file_str}, producer = \"vx\", isOptimized = false, emissionKind = Full>"
        );
        let di_subp_str = format!(
            "#llvm.di_subprogram<id = distinct[1]<>, compileUnit = {}, scope = {}, name = \"{}\", file = {}, subprogramFlags = Definition, type = #llvm.di_subroutine_type<>>",
            di_cu_str, di_file_str, func.name, di_file_str
        );
        self.di_subprogram = melior::ir::Attribute::parse(self.context, &di_subp_str);

        let mlir_str = format!(
            "module {{ func.func private @dummy() loc(fused<{}>[\"{}\":1:1]) }}",
            di_subp_str, self.current_filename
        );
        let dummy_module = melior::ir::Module::parse(self.context, &mlir_str).ok_or_else(|| {
            crate::codegen::lower::LowerError::from(format!("Failed to parse module: {mlir_str}"))
        })?;
        let dummy_op = dummy_module.body().first_operation().ok_or_else(|| {
            crate::codegen::lower::LowerError::from("No first operation".to_string())
        })?;
        Ok(dummy_op.location())
    }

    /// Bind each parameter of `func` to its block argument, and record the return slot when
    /// the function has one. `slot_args` is how many block arguments come before the first
    /// declared parameter.
    ///
    /// # Errors
    ///
    /// Fails when the block has fewer arguments than the function's signature needs.
    fn bind_parameters(
        &mut self,
        func: &Function,
        current_block: melior::ir::BlockRef<'c, 'c>,
        arg_tys: &[Type<'c>],
        slot_args: usize,
    ) -> Result<(), LowerError> {
        // Map arguments into the environment, past the return slot when there is one.
        for (i, (name, ast_ty)) in func.params.iter().enumerate() {
            let arg_val = current_block.argument(i + slot_args)?.into();
            self.env
                .insert(name.to_string().into(), (arg_val, arg_tys[i + slot_args]));
            self.ast_env.insert(name.to_string().into(), ast_ty.clone());
        }
        // The caller's buffer is block argument 0; `return` writes into it. Cleared with
        // `current_return_type` at the end of the function.
        self.current_return_slot = match slot_args {
            0 => None,
            _ => Some(current_block.argument(0)?.into()),
        };
        Ok(())
    }

    /// Append the call to `vx_init_signals` that `main` starts with.
    ///
    /// # Errors
    ///
    /// Fails when MLIR rejects the call.
    fn call_init_signals(&self, block: melior::ir::BlockRef<'c, 'c>) -> Result<(), LowerError> {
        let sig_init_call = melior::ir::operation::OperationBuilder::new("func.call", self.loc())
            .add_attributes(&[(
                melior::ir::Identifier::new(self.context, "callee"),
                melior::ir::attribute::FlatSymbolRefAttribute::new(self.context, "vx_init_signals")
                    .into(),
            )])
            .add_results(&[])
            .build()?;
        block.append_operation(sig_init_call);
        Ok(())
    }

    /// Lower the statements of a function's body into `current_block`, then end the last
    /// block when the body does not end it itself.
    ///
    /// # Errors
    ///
    /// Fails when a statement cannot be lowered.
    fn lower_function_body(
        &mut self,
        func: &Function,
        mut current_block: melior::ir::BlockRef<'c, 'c>,
        is_main: bool,
    ) -> Result<(), LowerError> {
        // Facts are scoped to the function being lowered so a proven `assert` cannot be
        // transported into a device kernel in an unrelated function (which could name the
        // same variable but bind a different, non-dominating SSA value).
        self.assert_facts.clear();
        let mut terminated = false;
        for stmt in &func.body {
            if let Some(b) = self.generate_statement(stmt, current_block)? {
                current_block = b;
            } else {
                terminated = true;
                break;
            }
        }

        // Append a fall-through terminator only when control actually reaches the
        // end of the body (no explicit `return`/`break` terminated it first).
        if terminated {
            return Ok(());
        }
        if is_main {
            // `main` is the C entry point and always lowers to `-> i32`. An
            // explicit `return <expr>` in main is honored like any other
            // function -- its i32 value becomes the process exit
            // code -- so we only synthesize `return 0` when the body has no
            // trailing return (e.g. a main that just prints).
            let i32_ty = self.i32_ty;
            let c0_op = current_block.append_operation(
                melior::ir::operation::OperationBuilder::new("arith.constant", self.loc())
                    .add_results(&[i32_ty])
                    .add_attributes(&[(
                        melior::ir::Identifier::new(self.context, "value"),
                        melior::ir::attribute::IntegerAttribute::new(i32_ty, 0).into(),
                    )])
                    .build()?,
            );
            let c0 = c0_op.result(0)?.into();
            current_block.append_operation(
                melior::ir::operation::OperationBuilder::new("func.return", self.loc())
                    .add_operands(&[c0])
                    .build()?,
            );
        } else if let syntax::Type::Struct(name, _) = &func.return_type {
            if syntax::is_void_ty(&func.return_type) && name.as_ref() != "none" {
                current_block.append_operation(
                    melior::ir::operation::OperationBuilder::new("func.return", self.loc())
                        .build()?,
                );
            }
        }
        Ok(())
    }

    /// The attributes of a function's `func.func` operation: its name and type, who frees
    /// each tensor argument and a placed result, and `noreturn` for a function declared
    /// `-> !`. `slot_args` is how many block arguments come before the first declared
    /// parameter.
    ///
    /// # Errors
    ///
    /// Fails when one of the attribute texts built here does not parse as MLIR.
    fn function_attributes(
        &self,
        func: &Function,
        slot_args: usize,
        name_attr: melior::ir::attribute::StringAttribute<'c>,
        type_attr: melior::ir::attribute::TypeAttribute<'c>,
    ) -> Result<Vec<(melior::ir::Identifier<'c>, melior::ir::Attribute<'c>)>, LowerError> {
        let mut func_attributes = vec![
            (
                melior::ir::Identifier::new(self.context, "sym_name"),
                name_attr.into(),
            ),
            (
                melior::ir::Identifier::new(self.context, "function_type"),
                type_attr.into(),
            ),
        ];
        // A tensor taken by value belongs to the function, which frees it. One placed in
        // another memory names the topology its plugin frees it on, and so does a placed
        // result, which the caller then frees.
        let mut per_arg: Vec<String> = vec!["{}".to_string(); slot_args];
        per_arg.extend(func.params.iter().map(|(_, ty)| match owned_tensor(ty) {
            Some(None) => "{vx.owned}".to_string(),
            Some(Some(p)) => self.placed_attr(&p),
            None => "{}".to_string(),
        }));
        let text = format!("[{}]", per_arg.join(", "));
        let attr = melior::ir::Attribute::parse(self.context, &text)
            .ok_or_else(|| LowerError::from(format!("bad argument attributes {text}")))?;
        func_attributes.push((melior::ir::Identifier::new(self.context, "arg_attrs"), attr));
        if let (0, Some(Some(p))) = (slot_args, owned_tensor(&func.return_type)) {
            let text = format!("[{}]", self.placed_attr(&p));
            let attr = melior::ir::Attribute::parse(self.context, &text)
                .ok_or_else(|| LowerError::from(format!("bad result attributes {text}")))?;
            func_attributes.push((melior::ir::Identifier::new(self.context, "res_attrs"), attr));
        }

        // `main` deliberately carries no `llvm.emit_c_interface`.
        //
        // The attribute makes `convert-func-to-llvm` synthesize an `_mlir_ciface_main` wrapper and
        // copy this function's location onto it -- including the `distinct !DISubprogram` that
        // location fuses, once debug emission was turned on. A distinct DISubprogram may be
        // attached to one function only, so LLVM's verifier rejected the module, the execution
        // engine returned null, and `--action emit-obj` dereferenced it and died with SIGSEGV for
        // every program that reached this path.
        //
        // Nothing wanted the wrapper. The ciface convention is how the dispatcher finds an
        // *outlined kernel* -- `runtime/vx_host_call.h` builds `_mlir_ciface_<kernel>` and the
        // attribute is set for that in `KernelOpLowering` -- and `main` is entered through its
        // ordinary symbol.

        // A function declared `-> !` is `noreturn` to LLVM.
        if syntax::is_never_ty(&func.return_type) {
            let attr = melior::ir::Attribute::parse(self.context, "[\"noreturn\"]")
                .ok_or_else(|| LowerError::from("bad noreturn attribute".to_string()))?;
            func_attributes.push((
                melior::ir::Identifier::new(self.context, "passthrough"),
                attr,
            ));
        }
        Ok(func_attributes)
    }

    /// Generate the `func.func` operation of one function: its signature, its body and its
    /// attributes.
    ///
    /// # Errors
    ///
    /// Fails when a type in the signature or a statement in the body cannot be lowered.
    pub(crate) fn generate_function(
        &mut self,
        func: &Function,
    ) -> Result<melior::ir::Operation<'c>, LowerError> {
        self.env.clear();
        self.allocs.clear();
        self.subspace_offsets.clear(); // per-function sub-space bump allocator (SS2)
        self.pointer_taken = syntax::locals_with_pointer_taken(&func.body);
        self.address_taken = syntax::locals_with_address_taken(&func.body);
        let is_main = func.name.as_ref() == "main";
        let true_ret_ty = self.lower_type(&func.return_type)?;
        let ret_ty = if is_main { self.i32_ty } else { true_ret_ty };

        let mut declared_tys = Vec::new();
        for (_, ty) in &func.params {
            declared_tys.push(self.lower_type(ty)?);
        }
        // The return slot, when there is one, becomes the first parameter and the result goes
        // away. `arg_tys` is what the block arguments are built from below, so a declared
        // parameter `i` is block argument `i + slot_args`.
        let (abi_ret, arg_tys) = self.abi_signature(ret_ty, &declared_tys);
        let slot_args = arg_tys.len() - declared_tys.len();

        let mut actual_ret_tys = Vec::new();
        if let Some(rt) = abi_ret.filter(|rt| rt.to_string() != "none") {
            actual_ret_tys.push(rt);
        }
        let func_type =
            melior::ir::r#type::FunctionType::new(self.context, &arg_tys, &actual_ret_tys);
        let name_attr = melior::ir::attribute::StringAttribute::new(self.context, &func.name);
        let type_attr = melior::ir::attribute::TypeAttribute::new(func_type.into());
        let func_loc = self.function_debug_location(func)?;

        let mut block_args = Vec::new();
        for ty in &arg_tys {
            block_args.push((*ty, self.loc()));
        }
        let region = melior::ir::Region::new();
        let current_block = region.append_block(Block::new(&block_args));
        self.bind_parameters(func, current_block, &arg_tys, slot_args)?;

        if is_main {
            self.call_init_signals(current_block)?;
        }

        self.current_return_type = Some(ret_ty);
        self.lower_function_body(func, current_block, is_main)?;
        self.current_return_type = None;
        self.current_return_slot = None;

        let func_attributes = self.function_attributes(func, slot_args, name_attr, type_attr)?;
        let func_op = melior::ir::operation::OperationBuilder::new("func.func", func_loc)
            .add_attributes(&func_attributes)
            .add_regions([region])
            .build()?;

        Ok(func_op)
    }

    /// Emit the runtime check for `assert(cond)` / `assert(cond, "msg")` (Vx#361).
    ///
    /// `cf.assert` rather than a hand-rolled branch, because MLIR lowers it for whichever
    /// target the code reaches and both passes are already in the pipelines: on the host
    /// `convert-cf-to-llvm` expands it to a branch onto `puts` + `abort` + `unreachable`, and
    /// INSIDE A KERNEL `convert-gpu-to-nvvm` expands it to `__assertfail` with the message,
    /// file, line and a `noreturn` attribute. `cf` is in `isDeviceLowerableDialect`, so the op
    /// also passes the kernel's device-ready gate.
    ///
    /// That portability is why there is no kernel guard here. An earlier version skipped
    /// emission under `in_spawn`, reasoning that the op would lower to a call to the HOST's
    /// `abort` in PTX -- which was the host pipeline's behaviour applied to device code, and
    /// wrong. A kernel assertion is a real assertion (Vx#362).
    ///
    /// A condition the checker already folded to `true` is dropped rather than emitted: there
    /// is nothing to test at run time, and E8002 has already rejected the false ones.
    ///
    /// # Errors
    ///
    /// Fails when the condition cannot be lowered, or when MLIR rejects the `cf.assert`.
    fn emit_runtime_assert(
        &mut self,
        s: &syntax::AssertStmt,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Result<Option<melior::ir::BlockRef<'c, 'c>>, LowerError> {
        // A literal `assert(true)` is not worth a branch. Anything less obvious is left to
        // the optimiser, which sees the same constant the checker did; a statically-FALSE
        // condition never reaches codegen at all (E8002 rejects it).
        if matches!(
            &*s.expr,
            syntax::Expr::Identifier(id) if id.name.as_ref() == "true"
        ) {
            return Ok(Some(block));
        }
        let (cond, _ty, block) = self.generate_expr(&s.expr, block)?;
        let msg = s
            .msg
            .clone()
            .unwrap_or_else(|| "assertion failed".to_string());
        let op = melior::ir::operation::OperationBuilder::new("cf.assert", self.loc())
            .add_operands(&[cond])
            .add_attributes(&[(
                melior::ir::Identifier::new(self.context, "msg"),
                melior::ir::attribute::StringAttribute::new(self.context, &msg).into(),
            )])
            .build()?;
        block.append_operation(op);
        Ok(Some(block))
    }

    /// Lower one statement into `block`, then run the drops that were waiting for its value.
    /// Returns the block to go on in, or `None` when the statement ends its block.
    ///
    /// # Errors
    ///
    /// Fails when the statement, or a drop that runs after it, cannot be lowered.
    pub(crate) fn generate_statement(
        &mut self,
        stmt: &Statement,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Result<Option<melior::ir::BlockRef<'c, 'c>>, LowerError> {
        if matches!(stmt, Statement::Drop(_)) {
            return self.generate_statement_inner(stmt, block);
        }
        let mine = std::mem::take(&mut self.queued_drops);
        let outer = std::mem::replace(&mut self.value_drops, mine);
        let lowered = self.generate_statement_inner(stmt, block);
        let left = std::mem::replace(&mut self.value_drops, outer);
        let next = lowered?;
        let Some(mut b) = next else {
            return Ok(None);
        };
        for d in left {
            b = self.emit_waiting_drop(d, b)?;
        }
        Ok(Some(b))
    }

    /// `Statement::Drop`: free the owner's buffer now, or once the next statement has its
    /// value. The buffer is read now, before an assignment can give the name a new one.
    ///
    /// # Errors
    ///
    /// Fails when the dropped value, or the call to its `Drop` impl, cannot be lowered.
    fn lower_drop(
        &mut self,
        d: &syntax::stmt::DropStmt,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Result<Option<melior::ir::BlockRef<'c, 'c>>, LowerError> {
        if let Some(stmt) = d.call_statement() {
            if d.after_value {
                self.queued_drops.push(WaitingDrop::Call(Box::new(stmt)));
                return Ok(Some(block));
            }
            return self.generate_statement_inner(&stmt, block);
        }
        // The tensor itself, or a tensor field of a struct being dropped.
        let place = d.expr.as_deref().cloned().unwrap_or_else(|| {
            Expr::Identifier(syntax::IdentifierExpr::new(d.name.clone(), d.span))
        });
        let (buffer, ty, block) = self.generate_expr(&place, block)?;
        // A rank-0 tensor is a plain number here: there is nothing to free.
        if !self.is_memref(&ty) {
            return Ok(Some(block));
        }
        let drop = WaitingDrop::Tensor {
            buffer,
            flag: d.flag.clone(),
            field: d.expr.is_some(),
        };
        if d.after_value {
            self.queued_drops.push(drop);
            return Ok(Some(block));
        }
        self.emit_waiting_drop(drop, block).map(Some)
    }

    /// Run one waiting drop in `block`. Returns the block to go on in: calling a `Drop` impl
    /// under a moved flag branches.
    ///
    /// # Errors
    ///
    /// Fails when the drop cannot be lowered, or when a call to a `Drop` impl ends its block.
    fn emit_waiting_drop(
        &mut self,
        d: WaitingDrop<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Result<melior::ir::BlockRef<'c, 'c>, LowerError> {
        match d {
            WaitingDrop::Tensor {
                buffer,
                flag,
                field,
            } => {
                self.emit_drop(buffer, flag, field, block)?;
                Ok(block)
            }
            WaitingDrop::Call(stmt) => self
                .generate_statement_inner(&stmt, block)?
                .ok_or_else(|| LowerError::from("a drop ended its block".to_string())),
        }
    }

    /// `vx.drop %buffer`, or `vx.drop %buffer unless %moved` when the tensor has a moved flag.
    /// `field`: the buffer is a tensor field of a struct being dropped.
    ///
    /// # Errors
    ///
    /// Fails when the moved flag cannot be read, or when MLIR rejects the `vx.drop`.
    pub(crate) fn emit_drop(
        &mut self,
        buffer: melior::ir::Value<'c, 'c>,
        flag: Option<crate::symbol::Symbol>,
        field: bool,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Result<(), LowerError> {
        let mut operands = vec![buffer];
        if let Some(flag) = flag {
            let name = Expr::Identifier(syntax::IdentifierExpr::new(flag, self.current_span));
            let (moved, _, _) = self.generate_expr(&name, block)?;
            operands.push(moved);
        }
        let mut op = melior::ir::operation::OperationBuilder::new("vx.drop", self.loc())
            .add_operands(&operands);
        if field {
            op = op.add_attributes(&[(
                melior::ir::Identifier::new(self.context, "field"),
                melior::ir::Attribute::unit(self.context),
            )]);
        }
        block.append_operation(op.build()?);
        Ok(())
    }

    /// Run the drops waiting for the value of the statement being lowered: once a `return` has
    /// its value, before it leaves.
    ///
    /// # Errors
    ///
    /// Fails when one of the drops cannot be lowered.
    pub(crate) fn run_value_drops(
        &mut self,
        mut block: melior::ir::BlockRef<'c, 'c>,
    ) -> Result<melior::ir::BlockRef<'c, 'c>, LowerError> {
        for d in std::mem::take(&mut self.value_drops) {
            block = self.emit_waiting_drop(d, block)?;
        }
        Ok(block)
    }

    /// Run the drops waiting for a block's last expression, which is lowered without
    /// `generate_statement`.
    ///
    /// # Errors
    ///
    /// Fails when one of the drops cannot be lowered.
    pub(crate) fn run_queued_drops(
        &mut self,
        mut block: melior::ir::BlockRef<'c, 'c>,
    ) -> Result<melior::ir::BlockRef<'c, 'c>, LowerError> {
        for d in std::mem::take(&mut self.queued_drops) {
            block = self.emit_waiting_drop(d, block)?;
        }
        Ok(block)
    }

    /// Lower one statement into `block`, without running the drops waiting for its value.
    /// Returns the block to go on in, or `None` when the statement ends its block.
    ///
    /// # Errors
    ///
    /// Fails when the statement cannot be lowered.
    ///
    /// # Panics
    ///
    /// Panics on a macro call: macros are expanded before code generation.
    fn generate_statement_inner(
        &mut self,
        stmt: &Statement,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Result<Option<melior::ir::BlockRef<'c, 'c>>, LowerError> {
        self.current_span = stmt.span();
        match stmt {
            Statement::Return(s) => LowerToMelior::lower(s, self, block),
            Statement::LetDecl(s) => LowerToMelior::lower(s, self, block),
            Statement::Assign(s) => LowerToMelior::lower(s, self, block),
            Statement::CompoundAssign(s) => LowerToMelior::lower(s, self, block),
            // An `if`/`match` whose every path returns is a statement that control does not
            // leave. Lower it, then report termination: without this the enclosing block branches
            // to a merge block nothing reaches, which the MLIR verifier rejects as a block with
            // no terminator. The same predicate stops the parser rewriting it to `return <expr>`.
            Statement::ExprStmt(s) => {
                // `abort()`, `panic(msg)` and a call to a `-> !` function end the block they are
                // in: nothing after them runs.
                if let Expr::FunctionCall(c) = &s.expr {
                    let never = self
                        .syntax_functions
                        .get(&c.name)
                        .is_some_and(|f| syntax::is_never_ty(&f.return_type));
                    if crate::syntax::is_abort_or_panic(c) || never {
                        let out = LowerToMelior::lower(s, self, block)?;
                        if let Some(b) = out {
                            b.append_operation(
                                melior::ir::operation::OperationBuilder::new(
                                    "llvm.unreachable",
                                    self.loc(),
                                )
                                .build()?,
                            );
                        }
                        return Ok(None);
                    }
                }
                let diverges = crate::syntax::expr::diverges_on_every_path(&s.expr);
                let out = LowerToMelior::lower(s, self, block)?;
                if !diverges {
                    return Ok(out);
                }
                // The block the statement hands back is never reached, and nothing will be
                // lowered into it, but it still needs a terminator.
                if let Some(b) = out.filter(|b| b.first_operation().is_none()) {
                    b.append_operation(
                        melior::ir::operation::OperationBuilder::new(
                            "llvm.unreachable",
                            self.loc(),
                        )
                        .build()?,
                    );
                }
                Ok(None)
            }
            Statement::ForLoop(s) => LowerToMelior::lower(s, self, block),
            Statement::Assert(s) => {
                // The condition is also a fact the host proves; record it so it can be
                // transported across a device `spawn` seam as an `llvm.intr.assume`
                // certificate (see `crate::codegen::lower::seam_cert`).
                if self.emit_seam_certs {
                    self.assert_facts.push((*s.expr).clone());
                }
                self.emit_runtime_assert(s, block)
            }
            Statement::Loop(s) => LowerToMelior::lower(s, self, block),
            Statement::Break(s) => LowerToMelior::lower(s, self, block),
            Statement::Continue(s) => LowerToMelior::lower(s, self, block),
            Statement::MacroCall(_) => panic!("Macros should be expanded before codegen"),
            Statement::Drop(d) => self.lower_drop(d, block),
            Statement::Error(_) => Ok(None),
        }
    }

    /// `abort()` -- terminate unconditionally, expressed as a conditional abort whose
    /// condition is `false`. One form serves both `abort` and `assert`, and it inherits
    /// the same target portability: `abort` + `unreachable` on the host, `__assertfail`
    /// inside a kernel. Handled here rather than in the general call lowering because it
    /// has no Vx-level definition to resolve -- it is a primitive, like `print`. Safe to
    /// call; ending a process breaks no memory-safety property.
    ///
    /// `panic(msg)` prints `panic: <msg>` first, through `print!`, and then aborts with an
    /// empty message: `cf.assert` prints it with `puts`, which ends the line. A kernel cannot
    /// print, so there it only aborts.
    ///
    /// # Errors
    ///
    /// Fails when the message of a `panic` cannot be printed, or when MLIR rejects an
    /// operation built here.
    fn lower_abort_or_panic(
        &mut self,
        e: &syntax::FunctionCallExpr,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError> {
        let mut block = block;
        let mut abort_msg = "abort() called";
        if e.name.as_ref() == "panic" {
            abort_msg = "panic";
            if !self.in_spawn {
                let print = Expr::Print(syntax::PrintExpr::new(
                    vec![
                        Expr::StringLiteral(syntax::StringLiteralExpr::new(
                            "panic: ".to_string(),
                            e.span,
                        )),
                        e.args[0].clone(),
                    ],
                    e.span,
                ));
                block = self.generate_expr(&print, block)?.2;
                abort_msg = "";
            }
        }
        self.emit_abort(abort_msg, block)
    }

    /// Append an assertion that always fails, with `abort_msg` as its message. Returns an
    /// `i32` zero as the value of the call, which no code after it reads.
    ///
    /// # Errors
    ///
    /// Fails when MLIR rejects an operation built here.
    fn emit_abort(
        &self,
        abort_msg: &str,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError> {
        let never = melior::ir::operation::OperationBuilder::new("arith.constant", self.loc())
            .add_attributes(&[(
                melior::ir::Identifier::new(self.context, "value"),
                melior::ir::attribute::IntegerAttribute::new(
                    melior::ir::r#type::IntegerType::new(self.context, 1).into(),
                    0,
                )
                .into(),
            )])
            .add_results(&[melior::ir::r#type::IntegerType::new(self.context, 1).into()])
            .build()?;
        let never_v = block.append_operation(never).result(0)?.into();
        let assert_op = melior::ir::operation::OperationBuilder::new("cf.assert", self.loc())
            .add_operands(&[never_v])
            .add_attributes(&[(
                melior::ir::Identifier::new(self.context, "msg"),
                melior::ir::attribute::StringAttribute::new(self.context, abort_msg).into(),
            )])
            .build()?;
        block.append_operation(assert_op);
        let zero = melior::ir::operation::OperationBuilder::new("arith.constant", self.loc())
            .add_attributes(&[(
                melior::ir::Identifier::new(self.context, "value"),
                melior::ir::attribute::IntegerAttribute::new(self.i32_ty, 0).into(),
            )])
            .add_results(&[self.i32_ty])
            .build()?;
        Ok((
            block.append_operation(zero).result(0)?.into(),
            self.i32_ty,
            block,
        ))
    }

    /// Lower one expression into `block`. Returns its value, the MLIR type of the value, and
    /// the block to go on in.
    ///
    /// # Errors
    ///
    /// Fails when the expression cannot be lowered, or is one that has no value at run time,
    /// such as a range or a memory space.
    pub(crate) fn generate_expr(
        &mut self,
        expr: &Expr,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError> {
        self.current_span = expr.span();
        match expr {
            Expr::Identifier(e) => LowerToMelior::lower(e, self, block),
            Expr::BinaryOp(e) => LowerToMelior::lower(e, self, block),
            Expr::RelationalOp(e) => LowerToMelior::lower(e, self, block),
            Expr::LogicalOp(e) => LowerToMelior::lower(e, self, block),
            Expr::UnaryOp(e) => LowerToMelior::lower(e, self, block),
            Expr::StructInit(e) => LowerToMelior::lower(e, self, block),
            Expr::MemberAccess(e) => LowerToMelior::lower(e, self, block),
            Expr::IndexAccess(e) => LowerToMelior::lower(e, self, block),
            Expr::FunctionCall(e) if syntax::is_abort_or_panic(e) => {
                self.lower_abort_or_panic(e, block)
            }
            Expr::FunctionCall(e) => LowerToMelior::lower(e, self, block),
            Expr::MethodCall(e) => LowerToMelior::lower(e, self, block),
            Expr::SpawnOn(e) => LowerToMelior::lower(e, self, block),
            Expr::Array(e) => LowerToMelior::lower(e, self, block),
            Expr::If(e) => LowerToMelior::lower(e, self, block),
            Expr::EnumVariant(e) => LowerToMelior::lower(e, self, block),
            Expr::Match(e) => LowerToMelior::lower(e, self, block),
            Expr::Number(e) => LowerToMelior::lower(e, self, block),
            Expr::UnsafeBlock(e) => LowerToMelior::lower(e, self, block),
            Expr::Grad(e) => LowerToMelior::lower(e, self, block),
            Expr::Vjp(e) => LowerToMelior::lower(e, self, block),
            Expr::Jvp(e) => LowerToMelior::lower(e, self, block),
            Expr::Transfer(e) => LowerToMelior::lower(e, self, block),
            Expr::Borrow(e) => LowerToMelior::lower(e, self, block),
            Expr::StringLiteral(e) => LowerToMelior::lower(e, self, block),
            Expr::Closure(e) => LowerToMelior::lower(e, self, block),
            Expr::ComptimeBlock(e) => LowerToMelior::lower(e, self, block),
            Expr::Dereference(e) => LowerToMelior::lower(e, self, block),
            Expr::AsCast(e) => LowerToMelior::lower(e, self, block),
            Expr::IndirectCall(e) => LowerToMelior::lower(e, self, block),
            Expr::Print(e) => LowerToMelior::lower(e, self, block),
            Expr::Println(e) => LowerToMelior::lower(e, self, block),
            Expr::InlineMlir(e) => LowerToMelior::lower(e, self, block),
            Expr::Topology(e) => LowerToMelior::lower(e, self, block),
            Expr::SizeOf(e) => LowerToMelior::lower(e, self, block),
            // Constructs with no runtime lowering: report a diagnostic instead of a
            // `todo!()` panic. (Match is exhaustive so a new Expr variant forces a decision.)
            Expr::Range(_) => Err(LowerError::from(
                "a range (`a..b`) is only valid as a `for`-loop bound, not as a value".to_string(),
            )),
            Expr::MemorySpace(_) => Err(LowerError::from(
                "a memory space (`Memory::...`) is not a runtime value".to_string(),
            )),
            Expr::VecMacro(_) => Err(LowerError::from(
                "`vec![...]` is not supported at this codegen position".to_string(),
            )),
            Expr::TransferPredicate(_) => Err(LowerError::from(
                "`Reachable<A, B>` is a comptime predicate; it may only appear as an \
                 `if comptime` condition"
                    .to_string(),
            )),
            Expr::MacroCall(_) => Err(LowerError::from(
                "internal error: macro call reached codegen (macros must be expanded first)"
                    .to_string(),
            )),
        }
    }

    /// Byte size + alignment of a type for `sizeof<T>()`, computed the way the layout pass does
    /// (`crate::layout::LayoutComputer::struct_layout`): a scalar by its element, any pointer/borrow
    /// 8, and a nominal struct (or a monomorphized generic instance of one) as the aligned sum of its
    /// recursively-sized fields. Name-keyed, because the generator holds struct decls by name, not
    /// GID. `None` for an unmodelled type (tensor, closure, generic scalar, data-carrying enum) — the
    /// caller falls back to a conservative default. This replaces `SizeOfExpr::lower`'s old hardcoded
    /// `8` for aggregates, which under-sized a `Vec<struct>`'s element buffer (a `Vec<Vec<T>>` UB). (#242)
    pub(crate) fn type_size_align(&self, ty: &syntax::Type, depth: u32) -> Option<(usize, usize)> {
        use crate::layout::{align_up, scalar_size_align};
        if depth > 32 {
            return None; // guard against a pathological/cyclic nesting
        }
        match ty {
            syntax::Type::Scalar(e) => scalar_size_align(e),
            syntax::Type::Pointer(..) | syntax::Type::Borrow { .. } | syntax::Type::Ref(..) => {
                Some((8, 8))
            }
            syntax::Type::Pinned(inner, _) | syntax::Type::Verified(inner) => {
                self.type_size_align(inner, depth)
            }
            syntax::Type::GenericInstance(base, _) => self.type_size_align(base, depth),
            syntax::Type::Struct(name, _) => {
                let base = name.as_ref().split('<').next().unwrap_or(name.as_ref());
                let decl = self.structs.get(base)?;
                let mut offset = 0usize;
                let mut align = 1usize;
                for (_, fty) in &decl.fields {
                    let (fsize, falign) = self.type_size_align(fty, depth + 1)?;
                    offset = align_up(offset, falign) + fsize;
                    align = align.max(falign);
                }
                Some((align_up(offset, align), align))
            }
            _ => None,
        }
    }

    ///
    /// The type of an enum's single payload slot.
    ///
    /// One slot holds every variant, so it is sized for the widest of them. Taking the
    /// first variant that had a payload made the slot too narrow for all the others, and
    /// the value was cut down to fit. `mapping` substitutes the enum's type parameters
    /// when the caller has them.
    ///
    /// A payload with no width of its own -- a pointer, or a nested aggregate -- keeps
    /// the old rule and takes the slot only if nothing else has claimed it.
    ///
    /// # Errors
    ///
    /// Fails when a payload type cannot be lowered, or when two variants put a struct and
    /// something else, or two different structs, in the same payload position.
    fn enum_payload_slot_ty(
        &self,
        enum_def: &[(crate::symbol::Symbol, Option<Vec<syntax::Type>>)],
        mapping: Option<&std::collections::HashMap<crate::symbol::Symbol, syntax::Type>>,
    ) -> Result<String, LowerError> {
        // One slot per payload position, each holding what any variant puts there: numbers of
        // any width fit the widest, but a struct needs its slot to itself.
        let arity = enum_def
            .iter()
            .filter_map(|(_, p)| p.as_ref().map(Vec::len))
            .max()
            .unwrap_or(0);
        if arity == 0 {
            return Ok("none".to_string());
        }
        let mut slots = Vec::with_capacity(arity);
        for position in 0..arity {
            let mut payload_ty_str = "none".to_string();
            let mut widest = 0u32;
            // Numbers of different kinds, such as `i32` and `f32`, share an integer slot. A float
            // goes in as its bits; an integer in a float slot could read back as a denormal.
            let mut scalars: Vec<String> = Vec::new();
            for (_v_name, payload) in enum_def {
                let Some(types) = payload else { continue };
                let Some(ty) = types.get(position) else {
                    continue;
                };
                let ty = match mapping {
                    Some(m) => ty.substitute(m),
                    None => ty.clone(),
                };
                let mut lowered = self.lower_type_str(&ty)?;
                if lowered.starts_with("memref<") {
                    lowered = "!llvm.ptr".to_string();
                }
                let is_struct = lowered.starts_with("!llvm.struct");
                if payload_ty_str != "none"
                    && is_struct != payload_ty_str.starts_with("!llvm.struct")
                    || is_struct && payload_ty_str != "none" && payload_ty_str != lowered
                {
                    return Err(LowerError::from(format!(
                        "an enum whose variants carry different kinds of payload \
                         ({payload_ty_str} and {lowered}) cannot be lowered by the legacy code \
                         generator"
                    )));
                }
                if scalar_type_bits(&lowered).is_some() && !scalars.contains(&lowered) {
                    scalars.push(lowered.clone());
                }
                match scalar_type_bits(&lowered) {
                    Some(b) if b > widest => {
                        widest = b;
                        payload_ty_str = lowered;
                    }
                    None if payload_ty_str == "none" => payload_ty_str = lowered,
                    _ => {}
                }
            }
            if scalars.len() > 1 && scalars.iter().any(|t| is_float_text(t)) {
                payload_ty_str = format!("i{widest}");
            }
            slots.push(payload_ty_str);
        }
        Ok(slots.join(", "))
    }

    /// How a struct field of type `ty` is laid out. A tensor is its memref descriptor, the
    /// layout the default code generator uses too, so its shape and offset travel with it.
    /// Any other memref (a reference to a scalar) is a bare pointer.
    ///
    /// # Errors
    ///
    /// Fails when the field's type cannot be lowered.
    pub(crate) fn field_type_str(
        &self,
        ty: &syntax::Type,
    ) -> Result<String, crate::codegen::lower::LowerError> {
        if let Some(rank) = tensor_field_rank(ty) {
            return Ok(tensor_descriptor_str(rank));
        }
        let lowered = self.lower_type_str(ty)?;
        Ok(if lowered.starts_with("memref<") {
            "!llvm.ptr".to_string()
        } else {
            lowered
        })
    }

    /// The `vx.placed` mark of a placed parameter or result. A placement written as a device
    /// holds that device's memory; on-chip memory, by the address space its type is given, is
    /// scratch nothing frees.
    fn placed_attr(&self, p: &syntax::Placement) -> String {
        match self.placed_topology(p) {
            Some(topology) => format!("{{vx.placed = {topology} : i32}}"),
            None => "{vx.placed}".to_string(),
        }
    }

    /// The topology a placed tensor's memory belongs to, which frees it; `None` for on-chip
    /// scratch, which nothing frees.
    pub(crate) fn placed_topology(&self, p: &syntax::Placement) -> Option<i32> {
        if on_chip(crate::arch::topology_address_space(
            &p.topology,
            &self.memories,
            &self.topologies,
        )) {
            return None;
        }
        let space = match p.written() {
            syntax::Written::Device => {
                crate::arch::topology_default_space(&p.topology, &self.topologies)
            }
            syntax::Written::Space => p.space.clone(),
        };
        Some(crate::arch::memory_space_dispatch_id(&space))
    }

    pub(crate) fn enum_payload_type_str(&self, name: &str) -> Option<String> {
        let enum_def = self.enums.get(name)?;
        if !enum_has_payload(enum_def) {
            return None;
        }
        self.generic_enum_type(name, enum_def)
            .ok()
            .map(|t| t.to_string())
    }

    /// The MLIR type of an enum that carries a payload: a struct named `name` that holds the
    /// variant's number and the payload slots.
    ///
    /// # Errors
    ///
    /// Fails when the payload slots cannot be worked out, or the struct type does not parse.
    fn generic_enum_type(
        &self,
        name: &str,
        enum_def: &[(crate::symbol::Symbol, Option<Vec<syntax::Type>>)],
    ) -> Result<Type<'c>, LowerError> {
        let payload_ty_str = self.enum_payload_slot_ty(enum_def, None)?;
        Type::parse(
            self.context,
            &format!("!llvm.struct<\"{name}\", (i32, {payload_ty_str})>"),
        )
        .ok_or_else(|| LowerError::from("failed to parse Option enum layout".to_string()))
    }

    /// The type arguments of a generic instance, joined into a name that is safe inside a
    /// struct name: `[i32, f32]` becomes `i32_f32`.
    ///
    /// # Errors
    ///
    /// Fails when one of the type arguments cannot be lowered.
    fn type_args_name(&self, args: &[syntax::Type]) -> Result<String, LowerError> {
        let names = args
            .iter()
            .map(|a| {
                let lowered = self.lower_type_str(a)?;
                Ok(lowered
                    .replace('!', "")
                    .replace(['<', '>', ' ', ','], "_")
                    .replace('"', "")
                    .replace(['(', ')', '.'], "_"))
            })
            .collect::<Result<Vec<_>, LowerError>>()?;
        Ok(names.join("_"))
    }

    /// Parse the text of a lowered type into an MLIR type.
    ///
    /// # Errors
    ///
    /// Fails when MLIR cannot parse the text.
    fn parse_lowered(&self, ty_str: &str) -> Result<Type<'c>, LowerError> {
        Type::parse(self.context, ty_str).ok_or_else(|| {
            LowerError::from(format!("failed to parse lowered MLIR type `{ty_str}`"))
        })
    }

    /// The MLIR type of a borrow of, or a raw pointer to, a value of type `inner`. `mem` is
    /// the memory space written on a borrow.
    ///
    /// # Errors
    ///
    /// Fails when `inner` cannot be lowered, or when `mem` has no address space on this
    /// target.
    fn lower_pointer_type(
        &self,
        inner: &syntax::Type,
        mem: Option<&syntax::MemorySpace>,
    ) -> Result<Type<'c>, LowerError> {
        let inner_str = self.lower_type_str(inner)?;
        let ty_str = if inner_str.starts_with("memref<") {
            // A borrowed tensor is the slot holding its descriptor, and every tensor
            // this backend allocates is dynamically shaped. Naming the static shape
            // here left the call site casting between two memref element types, which
            // `memref.cast` cannot do (Vx#417).
            format!("memref<{}>", erase_memref_extents(&inner_str))
        } else {
            // A space with no honest target mapping (an undeclared custom space, or one
            // whose declaration gives no `scope:`) is an error, not a silent fallback --
            // the old code answered "4", NVPTX read-only constant memory (#258).
            let addr_space = match mem {
                None => crate::arch::AddressSpace::Host,
                Some(m) => crate::arch::declared_address_space(m, self.memories.get(m))
                    .ok_or_else(|| LowerError::from(unmappable_space_message(m)))?,
            };
            format!("!llvm.ptr<{}>", addr_space.nvptx_addrspace())
        };
        self.parse_lowered(&ty_str)
    }

    /// The MLIR type of a type written as a plain name: an enum, a struct, `void`, `!`, or a
    /// generic instance whose name was flattened to text such as `Vec<i32>`.
    ///
    /// # Errors
    ///
    /// Fails when a field of the struct cannot be lowered, or the result does not parse.
    fn lower_named_type(&self, name: &crate::symbol::Symbol) -> Result<Type<'c>, LowerError> {
        if let Some(enum_def) = self.enums.get(name) {
            // Any instantiated generic enum, and whichever of its variants carries
            // the payload -- not a type called `Option` with a variant called `Some`.
            // A non-generic enum with a payload has the same `{tag, payload}` shape.
            if name.contains('<') || enum_has_payload(enum_def) {
                return self.generic_enum_type(name, enum_def);
            }
            return Ok(self.i32_ty);
        }
        let ty_str = if let Some(decl) = self.structs.get(name).cloned() {
            let mut field_types = Vec::new();
            for (_, ty) in &decl.fields {
                field_types.push(self.field_type_str(ty)?);
            }
            format!("!llvm.struct<\"{}\", ({})>", name, field_types.join(","))
        } else if name.as_ref() == "void" || name.as_ref() == "!" {
            "none".to_string()
        } else if name.contains('<') {
            // A generic instance that was flattened to a bracketed nominal
            // name during monomorphization (e.g. "Vec<i32>", "Option<i32>")
            // — such names arise when a monomorphized value type is
            // stringified and re-stored as a plain `Struct`. Re-parse it
            // into a proper `GenericInstance` and lower that, which resolves
            // to the concrete monomorphized struct body.
            let mut lexer = crate::lexer::Lexer::new(name);
            let tokens = lexer.tokenize();
            let mut parser = crate::parser::Parser::new(&tokens, name);
            match parser.parse_type() {
                Ok(parsed @ syntax::Type::GenericInstance(..)) => {
                    return self.lower_type(&parsed);
                }
                _ => format!("!llvm.struct<\"{name}\">"),
            }
        } else {
            format!("!llvm.struct<\"{name}\">")
        };
        self.parse_lowered(&ty_str)
    }

    /// The text of the MLIR struct type of a generic struct used with the type arguments
    /// `args`, such as `Vec<i32>`.
    ///
    /// # Errors
    ///
    /// Fails when there are fewer type arguments than the struct has type parameters, or
    /// when a field or a type argument cannot be lowered.
    fn generic_struct_type_str(
        &self,
        name: &crate::symbol::Symbol,
        decl: &syntax::StructDecl,
        args: &[syntax::Type],
    ) -> Result<String, LowerError> {
        let mut field_types = Vec::new();
        let mut mapping: std::collections::HashMap<crate::symbol::Symbol, syntax::Type> =
            std::collections::HashMap::new();
        for (i, param) in decl.generics.iter().enumerate() {
            if i >= args.len() {
                return Err(LowerError::from(format!(
                    "generic type `{}` expects {} type argument(s), got {}",
                    name,
                    decl.generics.len(),
                    args.len()
                )));
            }
            mapping.insert(param.name().into(), args[i].clone());
        }
        for (_, ty) in &decl.fields {
            field_types.push(self.field_type_str(&ty.substitute(&mapping))?);
        }
        let args_str = self.type_args_name(args)?;
        Ok(format!(
            "!llvm.struct<\"{}_{}\", ({})>",
            name,
            args_str,
            field_types.join(", ")
        ))
    }

    /// The text of the MLIR struct type of a generic enum used with the type arguments
    /// `args`, such as `Option<i32>`.
    ///
    /// # Errors
    ///
    /// Fails when there is no type argument, or when a payload or a type argument cannot
    /// be lowered.
    fn generic_enum_type_str(
        &self,
        name: &crate::symbol::Symbol,
        enum_def: &[(crate::symbol::Symbol, Option<Vec<syntax::Type>>)],
        args: &[syntax::Type],
    ) -> Result<String, LowerError> {
        let ty_arg = args.first().ok_or_else(|| {
            LowerError::from(format!("generic enum `{name}` used with no type argument"))
        })?;
        // Every parameter the declaration names, paired with the argument
        // in the same position. Binding only a parameter literally called
        // `T` left `Pair<T, E>`'s second one generic, and the payload of
        // whichever variant carried it reached codegen unsubstituted.
        let mut mapping: std::collections::HashMap<crate::symbol::Symbol, syntax::Type> =
            std::collections::HashMap::new();
        match self.enum_generics.get(name) {
            Some(params) if !params.is_empty() => {
                for (p, a) in params.iter().zip(args.iter()) {
                    mapping.insert(p.clone(), a.clone());
                }
            }
            // A declaration this generator never saw: the old guess is
            // still better than nothing for the common one-parameter enum.
            _ => {
                mapping.insert("T".into(), ty_arg.clone());
            }
        }
        let payload_ty_str = self.enum_payload_slot_ty(enum_def, Some(&mapping))?;

        let args_str = self.type_args_name(args)?;

        Ok(format!(
            "!llvm.struct<\"{name}_{args_str}\", (i32, {payload_ty_str})>"
        ))
    }

    /// The MLIR type of a generic struct or enum used with type arguments. `base` is the
    /// type the arguments are applied to.
    ///
    /// # Errors
    ///
    /// Fails when `base` is not the name of a declared struct or enum, or when the
    /// instance cannot be lowered.
    fn lower_generic_instance(
        &self,
        base: &syntax::Type,
        args: &[syntax::Type],
    ) -> Result<Type<'c>, LowerError> {
        let syntax::Type::Struct(name, _) = base else {
            return Err(LowerError::from(
                "internal: generic-instance base is not a struct".to_string(),
            ));
        };
        let ty_str = if let Some(decl) = self.structs.get(name).cloned() {
            self.generic_struct_type_str(name, &decl, args)?
        } else if let Some(enum_def) = self.enums.get(name).cloned() {
            self.generic_enum_type_str(name, &enum_def, args)?
        } else {
            return Err(LowerError::from(format!(
                "generic struct or enum `{name}` not found during codegen"
            )));
        };
        self.parse_lowered(&ty_str)
    }

    /// The MLIR type of a type the checker resolved to the enum `name`: `{tag, payload}`
    /// when the enum carries a payload, and a plain `i32` tag when it does not.
    ///
    /// # Errors
    ///
    /// Fails when the payload slots of the enum cannot be worked out.
    fn lower_enum_type(&self, name: &crate::symbol::Symbol) -> Result<Type<'c>, LowerError> {
        if let Some(enum_def) = self.enums.get(name) {
            // As for a struct-spelled enum in `lower_named_type`.
            if name.contains('<') || enum_has_payload(enum_def) {
                return self.generic_enum_type(name, enum_def);
            }
        }
        self.parse_lowered("i32")
    }

    /// The MLIR type a Vx type is lowered to.
    ///
    /// # Errors
    ///
    /// Fails for a type with no MLIR form: `Matrix`, a generic type that was not
    /// instantiated, a const generic argument, a memory space with no address space on
    /// this target, or a name that is not a declared generic struct or enum.
    ///
    /// # Panics
    ///
    /// Panics when MLIR cannot parse the fixed type of a closure, which means the MLIR
    /// libraries are broken.
    pub(crate) fn lower_type(
        &self,
        ty: &syntax::Type,
    ) -> Result<Type<'c>, crate::codegen::lower::LowerError> {
        match ty {
            syntax::Type::Tensor(el_ty, dims, top) => {
                self.lower_tensor_type(el_ty, dims, top.as_ref())
            }
            syntax::Type::Scalar(el_ty) => {
                let name = element_mlir(el_ty)?;
                Type::parse(self.context, name).ok_or_else(|| {
                    LowerError::from(format!("failed to parse MLIR element type `{name}`"))
                })
            }
            syntax::Type::Matrix => Err(LowerError::from(
                "codegen does not support the `Matrix` type".to_string(),
            )),
            syntax::Type::Ref(inner, _) | syntax::Type::Verified(inner) => self.lower_type(inner),
            syntax::Type::Pinned(inner, _top) => {
                self.parse_lowered(&self.lower_type(inner)?.to_string())
            }
            syntax::Type::Borrow {
                inner, mem_space, ..
            } => self.lower_pointer_type(inner, mem_space.as_ref()),
            // A raw pointer's placement is a fact for the type checker, not for the C ABI:
            // a C function takes one pointer type whatever memory it addresses, so the
            // placement is erased here and the pointer is a plain `!llvm.ptr`.
            syntax::Type::Pointer(inner, _, _) => self.lower_pointer_type(inner, None),
            syntax::Type::Struct(name, _) => self.lower_named_type(name),
            syntax::Type::GenericInstance(base, args) => self.lower_generic_instance(base, args),
            syntax::Type::Generic(_, _) => Err(LowerError::from(format!(
                "internal: generic type reached codegen (should have been monomorphized): {ty:?}"
            ))),
            syntax::Type::Simd(el_ty, n) => {
                self.parse_lowered(&format!("vector<{}x{}>", n, element_mlir(el_ty)?))
            }
            syntax::Type::Enum(name, _) => self.lower_enum_type(name),
            syntax::Type::Function(..) => Ok(self.ptr_ty),
            syntax::Type::Closure(_, _) => {
                Ok(Type::parse(self.context, "!llvm.struct<(ptr, ptr)>").unwrap())
            }
            syntax::Type::Module(..) => self.parse_lowered("none"),
            syntax::Type::Const(expr) => Err(LowerError::from(format!(
                "cannot lower a const generic argument to an MLIR type: {expr:?}"
            ))),
            syntax::Type::Unknown => self.parse_lowered("unknown"),
        }
    }

    /// The text of the MLIR type a Vx type is lowered to. A const generic argument, which
    /// has no MLIR type, gives its value as text that can be part of a struct name.
    ///
    /// # Errors
    ///
    /// Fails when the type cannot be lowered; see `lower_type`.
    pub(crate) fn lower_type_str(
        &self,
        ty: &syntax::Type,
    ) -> Result<String, crate::codegen::lower::LowerError> {
        if let syntax::Type::Function(..) = ty {
            return Ok("!llvm.ptr".to_string());
        }
        if let syntax::Type::Const(expr) = ty {
            if let syntax::Expr::Number(n) = &**expr {
                return Ok(n.value.to_string());
            } else if let syntax::Expr::StringLiteral(s) = &**expr {
                return Ok(s.value.to_string());
            }
            return Ok(format!("{expr:?}")
                .replace(' ', "_")
                .replace('"', "")
                .replace(['(', ')'], "_"));
        }
        let t = self.lower_type(ty)?;
        Ok(t.to_string())
    }

    /// Lowers a Vx `Tensor(ElementType, dims, topology)` to an MLIR `memref<...>` type.
    ///
    /// # Errors
    ///
    /// Fails when the element type has no MLIR form, when the topology's memory has no
    /// address space on this target, or when the `memref` type does not parse.
    fn lower_tensor_type(
        &self,
        el_ty: &ElementType,
        dims: &[syntax::Dim],
        top: Option<&syntax::Placement>,
    ) -> Result<Type<'c>, crate::codegen::lower::LowerError> {
        let ty_str = element_mlir(el_ty)?;

        // Rank is the list's length: `Tensor<f32, []>` is `memref<f32>`, and a `?` dimension
        // is a `?` in the memref, so a rank-1 dynamic tensor is `memref<?xf32>` and not the
        // rank 2 every run-time shape used to be assumed to have (Vx#404).
        let mut shape_str = String::new();
        if !dims.is_empty() {
            for (i, dim) in dims.iter().enumerate() {
                if let syntax::Dim::Static(syntax::Expr::Number(NumberExpr {
                    value: n_str,
                    ty: _,
                    span: _,
                })) = dim
                {
                    if let Ok(n) = n_str.parse::<f64>() {
                        #[allow(
                            clippy::cast_possible_truncation,
                            reason = "an extent is a whole number; one written as `4.0` is cut to `4`"
                        )]
                        let extent = n as i64;
                        let _ = write!(shape_str, "{extent}");
                    }
                } else {
                    shape_str.push('?');
                }
                if i < dims.len() - 1 {
                    shape_str.push('x');
                }
            }
        }

        if !shape_str.is_empty() && !shape_str.ends_with('x') {
            shape_str.push('x');
        }

        // Address space follows the topology's default memory space (single source of truth
        // in `arch`), so on-`Topology` and in-`MemorySpace` values of one buffer agree. A
        // topology whose memory has no honest target mapping is an error rather than a silent
        // fallback to constant memory (#258).
        let addr_space = match top {
            None => crate::arch::AddressSpace::Host,
            Some(p) => {
                crate::arch::topology_address_space(&p.topology, &self.memories, &self.topologies)
                    .ok_or_else(|| {
                    LowerError::from(format!(
                    "topology '{}' has no memory space that maps to this target's address spaces; \
                     declare its memory with a `scope:` (device/sm/cta/thread)",
                    p.topology.display_name()
                ))
                })?
            }
        };

        let memref_str = if addr_space == crate::arch::AddressSpace::Host {
            format!("memref<{shape_str}{ty_str}>")
        } else {
            format!(
                "memref<{}{}, {}>",
                shape_str,
                ty_str,
                addr_space.nvptx_addrspace()
            )
        };

        Type::parse(self.context, &memref_str).ok_or_else(|| {
            LowerError::from(format!(
                "failed to parse lowered memref type `{memref_str}`"
            ))
        })
    }

    /// The type of a field read `base.member`, with the struct's type parameters replaced
    /// by the type arguments of `base`. `None` when the type of `base` is not a known struct
    /// or the struct has no such field.
    fn infer_member_type(&self, ma: &syntax::MemberAccessExpr) -> Option<syntax::Type> {
        let mut base_ty = self.infer_ast_type(&ma.base)?;
        if let syntax::Type::Borrow { inner, .. } = base_ty {
            base_ty = *inner;
        }
        let mut generic_args = None;
        if let syntax::Type::GenericInstance(inner, args) = base_ty {
            base_ty = *inner;
            generic_args = Some(args);
        }
        if let syntax::Type::Struct(s_name, _) = base_ty {
            if let Some(decl) = self.structs.get(&s_name) {
                for (n, t) in &decl.fields {
                    if n == &ma.member {
                        let mut resolved_ty = t.clone();
                        if let Some(args) = generic_args {
                            let mut mapping: std::collections::HashMap<
                                crate::symbol::Symbol,
                                syntax::Type,
                            > = std::collections::HashMap::new();
                            for (i, param) in decl.generics.iter().enumerate() {
                                if i < args.len() {
                                    mapping.insert(param.name().into(), args[i].clone());
                                }
                            }
                            resolved_ty = resolved_ty.substitute(&mapping);
                        }
                        return Some(resolved_ty);
                    }
                }
            }
        }
        None
    }

    /// The type of `tensor_view_2d(p, rows, cols)` when both extents are number literals: a
    /// tensor of that shape over the element type `p` points to. `None` when an extent is
    /// only known at run time, or `p` is not a pointer to a scalar.
    fn tensor_view_type(&self, fc: &syntax::FunctionCallExpr) -> Option<syntax::Type> {
        let dims: Vec<Expr> = fc.args[1..3]
            .iter()
            .filter(|a| matches!(a, Expr::Number(_)))
            .cloned()
            .collect();
        if dims.len() == 2 {
            let el_ty = match self.infer_ast_type(&fc.args[0]) {
                Some(syntax::Type::Pointer(inner, _, _)) => match *inner {
                    syntax::Type::Scalar(e) => Some(e),
                    _ => None,
                },
                _ => None,
            };
            if let Some(el_ty) = el_ty {
                let dims = dims.into_iter().map(syntax::Dim::Static).collect();
                return Some(syntax::Type::Tensor(el_ty, dims, None));
            }
        }
        None
    }

    /// The type of the value a function call gives: the tensor a tensor constructor builds,
    /// or the declared return type of the function. `None` for a function that is not known.
    fn infer_call_type(&self, fc: &syntax::FunctionCallExpr) -> Option<syntax::Type> {
        let name = fc.name.to_string();
        // `Tensor<T>([d0, d1, ...])` constructor: recover the static shape so slice
        // indexing / vectorized reductions know the tile dims (mirrors sema in hir/expr).
        // The newer spelling writes the whole type, placement included, so there is
        // nothing to recover -- hand it back as written.
        if matches!(
            name.as_str(),
            "Tensor::new" | "Tensor::uninit" | "Tensor::fill"
        ) {
            return fc.type_args.as_ref().and_then(|a| a.first()).cloned();
        }
        if name.starts_with("Tensor")
            && !name.contains("::")
            && !name.contains('$')
            && !name.contains("__")
        {
            return Some(tensor_constructor_type(fc));
        }
        // `tensor_view_2d(p, rows, cols)` over literal extents is a statically shaped
        // view, which is how the checker types it. Recovering it here is what keeps an
        // unannotated binding and a redundantly annotated one generating the same code;
        // run-time extents stay unshaped, as they are for the checker too.
        if name == "tensor_view_2d" && fc.args.len() == 3 {
            if let Some(view) = self.tensor_view_type(fc) {
                return Some(view);
            }
        }
        self.syntax_functions
            .get(&fc.name)
            .map(|decl| decl.return_type.clone())
    }

    /// The type of the value a method call gives: a pointer for `as_ptr` / `as_mut_ptr` on
    /// a tensor, or the declared return type of the struct's method. `None` when the type
    /// of the receiver or the method is not known.
    fn infer_method_call_type(&self, mc: &syntax::MethodCallExpr) -> Option<syntax::Type> {
        let mut base_ty = self.infer_ast_type(&mc.base)?;
        if let syntax::Type::Borrow { inner, .. } = base_ty {
            base_ty = *inner;
        }
        // `t.as_ptr()` on a tensor: a pointer to its elements, in the tensor's memory,
        // as the checker types it.
        if let syntax::Type::Tensor(el, _, place) = &base_ty {
            if matches!(mc.method_name.as_ref(), "as_ptr" | "as_mut_ptr") {
                return Some(syntax::Type::Pointer(
                    Box::new(syntax::Type::Scalar(el.clone())),
                    place.clone(),
                    mc.method_name.as_ref() == "as_mut_ptr",
                ));
            }
        }
        if let syntax::Type::GenericInstance(inner, _) = base_ty {
            base_ty = *inner;
        }
        if let syntax::Type::Struct(s_name, _) = base_ty {
            let mangled = format!("{}_{}", s_name, mc.method_name);
            self.syntax_functions
                .get(mangled.as_str())
                .map(|decl| decl.return_type.clone())
        } else {
            None
        }
    }

    /// The Vx type of an expression, worked out again from the expression itself and the
    /// types recorded for the names it uses. `None` when it cannot be told this way.
    ///
    /// # Panics
    ///
    /// Never in practice: the number that marks a borrow's region as not yet set is small
    /// enough for a `usize`.
    #[must_use]
    pub fn infer_ast_type(&self, expr: &Expr) -> Option<syntax::Type> {
        match expr {
            // A cast's value has the target type by definition (`my_closure as ||->i32`), and the
            // indirect-call path needs it in `ast_env` to rebuild the callee signature.
            Expr::AsCast(c) => Some(c.target_ty.clone()),
            // An `unsafe { .. }` block has the type of its tail expression. Without this arm an
            // unannotated `let` over one records nothing in `ast_env`, so every consumer is blind
            // to a type the checker already computed -- and a redundant annotation then decides
            // which code is generated. The flat path's inference already sees through it.
            Expr::UnsafeBlock(u) => self.infer_ast_type(u.ret.as_deref()?),
            // `let p = &mut x` binds a reference. Recording that is what tells a later read of
            // `p` to hand the reference back rather than load through it; an unannotated binding
            // recorded nothing, so the read could not tell `p` from a scalar local of its own.
            Expr::Borrow(b) => Some(syntax::Type::Borrow {
                inner: Box::new(self.infer_ast_type(&b.expr)?),
                mem_space: None,
                is_mut: b.is_mut,
                region_id: usize::try_from(syntax::REGION_UNSET)
                    .expect("the unset region is a 12-bit number"),
            }),
            // A generic struct's literal does not spell its type arguments, so only a plain
            // struct is known from the literal alone.
            Expr::StructInit(s) => self
                .structs
                .get(&s.name)
                .filter(|decl| decl.generics.is_empty())
                .map(|_| syntax::Type::Struct(s.name.clone(), s.type_id)),
            Expr::Identifier(id) => self.ast_env.get(&id.name).cloned().or_else(|| {
                // A bare function name used as a *value* (`let f = probe`) is a function pointer;
                // recover its signature from the function registry so a later indirect call `f(..)`
                // can build the callee type (#268). The frontend already types this via `self.lookup`;
                // this is the codegen counterpart, keeping `ast_env` populated so `lower/expr.rs`'s
                // indirect-call path does not hit "Missing signature for function pointer".
                self.syntax_functions.get(&id.name).map(|f| {
                    syntax::Type::Function(
                        f.params.iter().map(|(_, t)| t.clone()).collect(),
                        Box::new(f.return_type.clone()),
                        f.is_unsafe,
                    )
                })
            }),
            Expr::MemberAccess(ma) => self.infer_member_type(ma),
            Expr::FunctionCall(fc) => self.infer_call_type(fc),
            // `transfer(x, space)` relocates x but preserves its shape, so a slice of a
            // transferred tensor (`q[i]` inside a spawn) can still recover its static dims.
            Expr::Transfer(e) => self.infer_ast_type(&e.expr),
            Expr::Dereference(d) => match self.infer_ast_type(&d.expr)? {
                syntax::Type::Pointer(inner, ..)
                | syntax::Type::Borrow { inner, .. }
                | syntax::Type::Ref(inner, ..) => Some(*inner),
                _ => None,
            },
            Expr::MethodCall(mc) => self.infer_method_call_type(mc),
            _ => None,
        }
    }

    /// Lower a chain of index reads such as `a[i][j]` into the value being indexed, its MLIR
    /// type and the list of indices, outermost first, each cast to `index`. An expression
    /// that is not an index read gives itself and no indices. `None` when a part of the
    /// chain cannot be lowered.
    pub fn flatten_indices(
        &mut self,
        expr: &Expr,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Option<(
        Value<'c, 'c>,
        Type<'c>,
        Vec<Value<'c, 'c>>,
        melior::ir::BlockRef<'c, 'c>,
    )> {
        if let Expr::IndexAccess(syntax::IndexAccessExpr {
            base,
            index: idx,
            span: _,
        }) = expr
        {
            let (base_val, base_ty, mut indices, block) = self.flatten_indices(base, block)?;
            // The index is a value even when the element is wanted as a place: `&a[*i]`.
            let prev_lvalue = std::mem::replace(&mut self.is_lvalue_context, false);
            let lowered = self.generate_expr(idx, block);
            self.is_lvalue_context = prev_lvalue;
            let (idx_val, _, block) = lowered.ok()?;

            let idx_ty_str = idx_val.r#type().to_string();
            let actual_idx = if idx_ty_str == "index" {
                idx_val
            } else {
                let cast_op =
                    melior::ir::operation::OperationBuilder::new("arith.index_cast", self.loc())
                        .add_operands(&[idx_val])
                        .add_results(&[self.index_ty])
                        .build()
                        .ok()?;
                block.append_operation(cast_op).result(0).ok()?.into()
            };

            indices.push(actual_idx);
            Some((base_val, base_ty, indices, block))
        } else {
            // A raw pointer is indexed through its value, so `&(*p).data[i]` loads `data`
            // even when the element is wanted as a place.
            let is_raw_ptr = matches!(self.infer_ast_type(expr), Some(syntax::Type::Pointer(..)));
            let prev_lvalue = self.is_lvalue_context;
            self.is_lvalue_context &= !is_raw_ptr;
            let lowered = self.generate_expr(expr, block);
            self.is_lvalue_context = prev_lvalue;
            let (val, ty, block) = lowered.ok()?;
            Some((val, ty, Vec::new(), block))
        }
    }
}

/// The diagnostic for a memory space that cannot be mapped to a target address space: an
/// undeclared custom space, or a declaration carrying no `scope:`. Reported instead of silently
/// lowering into NVPTX constant (read-only) memory, which is what every declared space used to
/// get (#258).
fn unmappable_space_message(mem: &syntax::MemorySpace) -> String {
    format!(
        "memory space '{}' has no address space on this target; declare it with a `scope:` \
         (device/sm/cta/thread) so it maps to global/shared/private memory",
        mem.name()
    )
}

/// The LLVM struct a memref of `rank` lowers to: allocated and aligned pointers, offset, sizes
/// and strides.
pub(crate) fn tensor_descriptor_str(rank: usize) -> String {
    if rank == 0 {
        "!llvm.struct<(ptr, ptr, i64)>".to_string()
    } else {
        format!("!llvm.struct<(ptr, ptr, i64, array<{rank} x i64>, array<{rank} x i64>)>")
    }
}

/// The rank of a tensor a struct field holds as a descriptor. `None` for any other type,
/// including a reference to a tensor, which this code generator keeps as a cell holding the
/// descriptor rather than as the descriptor.
pub(crate) fn tensor_field_rank(ty: &syntax::Type) -> Option<usize> {
    match ty {
        syntax::Type::Tensor(_, dims, _) => Some(dims.len()),
        _ => None,
    }
}

/// The tensor type an older-style constructor call `Tensor<T>([d0, d1, ...])` builds. The
/// element type is the type argument, or `f32` when none is written. The shape is the list
/// of extents, or the nesting of an initializer list such as `Tensor<T>([[..], [..]])`.
fn tensor_constructor_type(fc: &syntax::FunctionCallExpr) -> syntax::Type {
    let el_ty = match fc.type_args.as_ref().and_then(|a| a.first()) {
        Some(syntax::Type::Scalar(el)) => el.clone(),
        _ => syntax::ElementType::F32,
    };
    let dims = match fc.args.first() {
        Some(Expr::Array(arr)) => match arr.initializer_shape() {
            // Initializer list `Tensor<T>([[..],[..]])`: shape from the nesting.
            Some(shape) => shape
                .into_iter()
                .map(|n| {
                    Expr::Number(syntax::NumberExpr {
                        value: n.to_string().into(),
                        ty: Some(syntax::ElementType::I32),
                        span: syntax::Span::default(),
                    })
                })
                .collect(),
            None => arr.elements.clone(),
        },
        _ => fc.args.clone(),
    };
    let dims = dims.into_iter().map(syntax::Dim::Static).collect();
    syntax::Type::Tensor(el_ty, dims, None)
}

/// The MLIR name of an element type, with an error for the ones codegen cannot lower.
///
/// # Errors
///
/// Fails for a generic element type, and for a float type narrower than 8 bits.
fn element_mlir(el_ty: &ElementType) -> Result<&'static str, LowerError> {
    if let Some(name) = crate::mlir_ty::mlir_scalar(el_ty) {
        return Ok(name);
    }
    Err(LowerError::from(match el_ty {
        ElementType::Generic(_) => "internal: generic element type reached codegen \
                                    (monomorphization should have instantiated it)"
            .to_string(),
        _ => "sub-8-bit float element types are capacity/declaration-only; \
              their codegen is not implemented"
            .to_string(),
    }))
}

/// Whether some variant of the enum carries a payload, so its values need room for it.
pub(crate) fn enum_has_payload(
    enum_def: &[(crate::symbol::Symbol, Option<Vec<syntax::Type>>)],
) -> bool {
    enum_def
        .iter()
        .any(|(_, p)| p.as_ref().is_some_and(|p| !p.is_empty()))
}

/// `{vx.placed = 500 : i32}`: a placed parameter or result, and the topology the plugin that
/// allocated it frees it on, the number a transfer to that memory carries. On-chip memory, shared
/// or per-thread, is scratch nothing frees: it is `{vx.placed}` with no topology.
pub(crate) fn placed_attr(space: &syntax::MemorySpace, on_chip: bool) -> String {
    if on_chip {
        return "{vx.placed}".to_string();
    }
    format!(
        "{{vx.placed = {} : i32}}",
        crate::arch::memory_space_dispatch_id(space)
    )
}

/// The tensor a value of type `ty` is, seen through `Verified<..>` and `Pinned<..>`, which add a
/// proof or a location to the same buffer: `Some(None)` for host memory, `Some(Some(p))` for one
/// placed by `p`, `None` for a type that is not a tensor.
#[allow(
    clippy::option_option,
    reason = "callers are in other files; they match on the three cases this returns"
)]
pub(crate) fn owned_tensor(ty: &syntax::Type) -> Option<Option<syntax::Placement>> {
    match ty {
        syntax::Type::Tensor(_, _, placement) => Some(placement.clone()),
        syntax::Type::Verified(inner) => owned_tensor(inner),
        syntax::Type::Pinned(inner, top) => {
            if let Some(p) = owned_tensor(inner)? {
                Some(Some(p))
            } else {
                let p = syntax::Placement::on(top.clone());
                Some((p.space != syntax::MemorySpace::CPUDRAM).then_some(p))
            }
        }
        _ => None,
    }
}

/// The placement written in a tensor type, seen through `Verified<..>` and `Pinned<..>`. Only
/// this one gives the tensor's memref a memory space, which keeps it out of a return slot.
pub(crate) fn written_placement(ty: &syntax::Type) -> Option<&syntax::Placement> {
    match ty {
        syntax::Type::Tensor(_, _, placement) => placement.as_ref(),
        syntax::Type::Verified(inner) | syntax::Type::Pinned(inner, _) => written_placement(inner),
        _ => None,
    }
}

/// Whether an address space is on-chip scratch: shared memory or per-thread memory.
pub(crate) fn on_chip(space: Option<crate::arch::AddressSpace>) -> bool {
    matches!(
        space,
        Some(crate::arch::AddressSpace::Workgroup | crate::arch::AddressSpace::Private)
    )
}
