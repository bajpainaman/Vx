//===- decl.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Abstract Syntax Tree nodes for declarations, including traits, structs, and functions.
//
//===----------------------------------------------------------------------===//

use super::{Expr, MemorySpace, Span, Statement, TokenTree, Topology, Type};
use crate::symbol::Symbol;

/// One generic parameter of a function, struct, enum, trait or impl: a type or a constant.
#[derive(Debug, PartialEq, Clone)]
pub enum GenericParam {
    /// A type parameter and the traits it is constrained by. Several are written
    /// `T : A + B`, and all of them have to hold; an unconstrained parameter has none.
    Type {
        /// The parameter's name, such as `T`.
        name: Symbol,
        /// The traits the type must implement.
        bounds: Vec<TraitBound>,
    },
    /// A constant parameter, written `const N : usize`.
    Const {
        /// The parameter's name, such as `N`.
        name: Symbol,
        /// The type of the constant.
        ty: Type,
    },
}

/// One bound on a type parameter: the trait, with the type arguments it is given and what its
/// associated types must be. `Sum<i64>` has one argument, `Iterator<Item = i64>` one binding,
/// and a plain `Ord` neither.
#[derive(Debug, PartialEq, Clone)]
pub struct TraitBound {
    /// The trait the type must implement.
    pub trait_name: Symbol,
    /// The type arguments given to the trait, as in `Sum<i64>`.
    pub args: Vec<Type>,
    /// What each named associated type must be, as in `Iterator<Item = i64>`.
    pub bindings: Vec<(Symbol, Type)>,
}

impl TraitBound {
    /// A bound that names a trait and nothing more.
    #[must_use]
    pub fn named(trait_name: &str) -> Self {
        Self {
            trait_name: trait_name.into(),
            args: Vec::new(),
            bindings: Vec::new(),
        }
    }
}

impl std::fmt::Display for TraitBound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.trait_name)?;
        if self.args.is_empty() && self.bindings.is_empty() {
            return Ok(());
        }
        let args = self.args.iter().map(std::string::ToString::to_string);
        let bindings = self.bindings.iter().map(|(n, t)| format!("{n} = {t}"));
        write!(
            f,
            "<{}>",
            args.chain(bindings).collect::<Vec<_>>().join(", ")
        )
    }
}

impl GenericParam {
    /// The parameter's name.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            GenericParam::Type { name, .. } | GenericParam::Const { name, .. } => name,
        }
    }
}

/// A function: its signature, its contracts and its body.
#[derive(Debug, PartialEq, Clone)]
pub struct Function {
    /// The function's name.
    pub name: Symbol,
    /// The generic parameters written after the name.
    pub generics: Vec<GenericParam>,
    /// Each parameter's name and type, in the order written.
    pub params: Vec<(Symbol, Type)>,
    /// The parameters written `mut x : T`. The others, like a `let` without `mut`, cannot be
    /// assigned or borrowed `&mut`.
    pub mut_params: Vec<Symbol>,
    /// The hardware the function runs on, from its `on ...` clause. The CPU when there is none.
    pub topology: Topology,
    /// The type of the value the function returns.
    pub return_type: Type,
    /// The `requires` conditions: what must hold when the function is called.
    pub requires: Vec<Expr>,
    /// The `ensures` conditions: what holds when the function returns.
    pub ensures: Vec<Expr>,
    /// `where Reachable<S, D>` constraints: pairs of topology names (generic topology
    /// variables or concrete topologies) that must have a transfer path in the cost
    /// graph. Discharged at each generic call once the variables are bound.
    pub where_transfers: Vec<(Symbol, Symbol)>,
    /// `unsafe fn`: the function states a contract its caller must discharge, so a call needs an
    /// unsafe context. The body is not itself one (Rust 2024): an unsafe operation inside still
    /// needs its own `unsafe` block.
    pub is_unsafe: bool,
    /// The statements of the body, in order.
    pub body: Vec<Statement>,
    /// The `///` lines above the function.
    pub doc_comment: Option<String>,
}

impl Function {
    /// A copy of the function that always has its signature and contracts. The body is copied
    /// only when `preserve_body` is set or the function is generic, and is empty otherwise.
    #[must_use]
    pub fn clone_signature(&self, preserve_body: bool) -> Self {
        Self {
            name: self.name.clone(),
            generics: self.generics.clone(),
            params: self.params.clone(),
            mut_params: self.mut_params.clone(),
            topology: self.topology.clone(),
            return_type: self.return_type.clone(),
            requires: self.requires.clone(),
            ensures: self.ensures.clone(),
            where_transfers: self.where_transfers.clone(),
            is_unsafe: self.is_unsafe,
            body: if preserve_body || !self.generics.is_empty() {
                self.body.clone()
            } else {
                Vec::new()
            },
            doc_comment: self.doc_comment.clone(),
        }
    }
}

/// A `struct` declaration.
#[derive(Debug, PartialEq, Clone)]
pub struct StructDecl {
    /// The struct's name.
    pub name: Symbol,
    /// The generic parameters written after the name.
    pub generics: Vec<GenericParam>,
    /// Each field's name and type, in the order written.
    pub fields: Vec<(Symbol, Type)>,
    /// The `///` lines above the struct.
    pub doc_comment: Option<String>,
}

/// An `enum` declaration.
#[derive(Debug, PartialEq, Clone)]
pub struct EnumDecl {
    /// The enum's name.
    pub name: Symbol,
    /// The generic parameters written after the name.
    pub generics: Vec<GenericParam>,
    /// Each variant's name and the types of its payload, or `None` for a variant without one.
    pub variants: Vec<(Symbol, Option<Vec<Type>>)>,
    /// The `///` lines above the enum.
    pub doc_comment: Option<String>,
}

/// An `extern` function declaration: a function defined outside Vx, with no body here.
#[derive(Debug, PartialEq, Clone)]
pub struct ExternDecl {
    /// The function's name.
    pub name: Symbol,
    /// Whether the declaration is written with the `safe` keyword.
    pub is_safe: bool,
    /// Each parameter's name and type, in the order written.
    pub params: Vec<(Symbol, Type)>,
    /// The type of the value the function returns.
    pub return_type: Type,
    /// Where the declaration was written, so a diagnostic about the signature can point at it.
    pub span: Span,
    /// The `///` lines above the declaration. Six library modules are nothing but an extern
    /// block, so without this they had no way to say what their functions do.
    pub doc_comment: Option<String>,
}

/// One method declared in a trait.
#[derive(Debug, PartialEq, Clone)]
pub struct MethodSignature {
    /// The method's name.
    pub name: Symbol,
    /// The method's own type parameters, as in `fn fold<B>(..)`. Separate from the trait's.
    pub generics: Vec<GenericParam>,
    /// Each parameter's name and type, in the order written.
    pub params: Vec<(Symbol, Type)>,
    /// The parameters written `mut`, as for a [`Function`].
    pub mut_params: Vec<Symbol>,
    /// The type of the value the method returns.
    pub return_type: Type,
    /// The default body, when the trait writes one instead of a `;`. An impl that does not
    /// provide the method gets a copy of this, with `Self` replaced by the type it is
    /// implemented for. `None` means the method is required.
    pub default_body: Option<Vec<crate::syntax::Statement>>,
    /// The `///` lines above the signature. A trait's methods are the public surface of the
    /// types that implement it, so this is where most of the library's documentation belongs.
    pub doc_comment: Option<String>,
}

/// A `trait` declaration.
#[derive(Debug, PartialEq, Clone)]
pub struct TraitDecl {
    /// The trait's name.
    pub name: Symbol,
    /// The generic parameters written after the name.
    pub generics: Vec<GenericParam>,
    /// The methods the trait declares, in the order written.
    pub methods: Vec<MethodSignature>,
    /// `type Item;`. A name the trait's signatures may write as `Self::Item`, which every
    /// impl binds to a type of its own.
    pub assoc_types: Vec<Symbol>,
}

/// An `impl` block: methods for a type, and the trait they implement when there is one.
#[derive(Debug, PartialEq, Clone)]
pub struct ImplBlock {
    /// The generic parameters written after `impl`.
    pub generics: Vec<GenericParam>,
    /// The trait being implemented, or `None` for an impl that adds methods to the type itself.
    pub trait_name: Option<Symbol>,
    /// The trait's own type arguments: `[i64]` for `impl Iterator<i64> for Range`. Needed
    /// to substitute the trait's parameters into a default body's signature.
    pub trait_args: Vec<Type>,
    /// The type the methods are for.
    pub target_type: Type,
    /// The methods, including any copied from the trait's default bodies.
    pub methods: Vec<Function>,
    /// `type Item = i64;`. What this impl binds each of the trait's associated types to.
    pub assoc_bindings: Vec<(Symbol, Type)>,
    /// The methods in `methods` that were copied from the trait's default bodies rather than
    /// written in this impl. They are checked only when something calls them, as Rust checks a
    /// default's bounds only at the call: `Iterator::sum` needs `Sum` on the item type, and an
    /// iterator over structs must still compile when it never calls `sum`.
    pub copied_defaults: Vec<Symbol>,
}

impl ImplBlock {
    /// A copy of the impl block. Method bodies are kept.
    #[must_use]
    pub fn clone_signature(&self) -> Self {
        Self {
            generics: self.generics.clone(),
            trait_name: self.trait_name.clone(),
            trait_args: self.trait_args.clone(),
            target_type: self.target_type.clone(),
            assoc_bindings: self.assoc_bindings.clone(),
            copied_defaults: self.copied_defaults.clone(),
            // Always preserve method bodies: methods only exist in impl blocks, so
            // method-call monomorphization clones the body from the type-check env
            // (env.impls). Dropping it (as the signature clone does for free
            // functions, whose bodies come from `program.functions`) yields an
            // empty method body and invalid IR. See GitHub #146.
            methods: self
                .methods
                .iter()
                .map(|m| m.clone_signature(true))
                .collect(),
        }
    }
}

/// An `import` of another module.
#[derive(Debug, PartialEq, Clone)]
pub struct ImportDecl {
    /// The parts of the module's path, in order.
    pub path: Vec<Symbol>,
}

/// A macro definition: its name and the rules that say what a call expands to.
#[derive(Debug, PartialEq, Clone)]
pub struct MacroDefDecl {
    /// The macro's name.
    pub name: Symbol,
    /// The rules, tried in the order written.
    pub rules: Vec<MacroRule>,
    /// Where the definition is in the source.
    pub span: Span,
}

/// One rule of a macro: a pattern, and what a call that matches it expands to.
#[derive(Debug, PartialEq, Clone)]
pub struct MacroRule {
    /// The pattern a call's tokens are matched against.
    pub matcher: Vec<TokenTree>,
    /// The tokens the call is replaced with.
    pub transcriber: Vec<TokenTree>,
}

/// A byte quantity, e.g. `256 KB`, normalized to bytes (binary multipliers: `KB` = 1024).
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct ByteSize(pub u64);

/// The denominator of a bandwidth rate.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum RatePer {
    /// `/s`
    Second,
    /// `/cyc`
    Cycle,
}

/// A bandwidth, e.g. `8 TB/s` or `128 B/cyc`; the numerator is normalized to bytes.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Bandwidth {
    /// How many bytes move in one unit of time.
    pub bytes: u64,
    /// The unit of time: a second or a clock cycle.
    pub per: RatePer,
}

/// How a memory space is managed (the transfer checker uses this to decide the obligation).
#[derive(Debug, PartialEq, Eq, Clone, Copy, Default)]
pub enum Management {
    /// Programmer-managed: movement in/out requires an explicit `transfer`.
    Explicit,
    /// Hardware-cached: movement may be implicit. The default.
    #[default]
    Cached,
}

/// How data physically crosses into a space from its parent, which decides how a multi-hop walk
/// composes.
///
/// Measured on two architectures, and the law tracks this property rather than the machine:
///
///   H100  HBM->L2->SMEM   streamed (cp.async)     bottleneck fits  -13.9%
///   H100  L1->REG->SMEM   sequenced               sum fits          -9.3%
///   M4    L2->REG->SMEM   sequenced               sum fits         +13.2%
///   M4    HBM->REG->SMEM  sequenced               sum fits          +2.7%
///
/// `within:` says one space contains another; it does not say whether crossing that boundary is a
/// hardware path or a pair of instructions. Without this distinction any single global law is
/// wrong on half the routes -- summing overstates a streamed route by ~67%, and taking the
/// bottleneck understates a sequenced one by ~49%. Same mistake, opposite directions.
#[derive(Debug, PartialEq, Eq, Clone, Copy, Default)]
pub enum Crossing {
    /// A load into a register followed by a store back out. The instructions genuinely happen one
    /// after the other, so the legs genuinely add.
    ///
    /// The default, deliberately: it is what the algebra has always done, so introducing this
    /// property changes no prediction anywhere and the frozen cells stay byte-identical. Declaring
    /// `streamed` is then an explicit, dated machine-file edit with a localized effect, rather than
    /// a silent model change that moves 240 cells at once.
    #[default]
    Sequenced,
    /// A hardware engine fills the destination without passing through registers (Hopper's
    /// `cp.async`, a DMA). Nothing stages, so the narrowest leg alone sets the rate.
    Streamed,
}

/// The execution level at which a memory space is private / replicated. Ordered broadest to
/// narrowest — locality *narrows* going down a `within:` hierarchy (a per-SM space sits inside
/// a per-device space, never the reverse). Lets `capacity` be read as a per-scope budget:
/// a `Sm`-scoped 256 KB is one SM's TMEM, not a global pool.
#[derive(Debug, PartialEq, Eq, Clone, Copy, PartialOrd, Ord)]
pub enum Scope {
    /// The whole device (HBM, L2).
    Device,
    /// One streaming multiprocessor (SMEM, TMEM).
    Sm,
    /// One cooperative thread array / thread block.
    Cta,
    /// One thread (registers).
    Thread,
}

/// A first-class memory-space declaration:
/// `Memory <Name> { within:, capacity:, bandwidth:, managed:, granule: }`.
///
/// Carried on the AST (`Program.memories`) and indexed by `GlobalAstEnv` for semantic
/// analysis; codegen reads it from the `Program`. Deliberately *not* registered in a
/// process-global registry (unlike topologies), so declarations cannot leak between
/// compilations.
#[derive(Debug, PartialEq, Clone)]
pub struct MemoryDecl {
    /// The memory space's name.
    pub name: Symbol,
    /// `within: Memory::X` — the parent space in the hierarchy tree; `None` for a root.
    pub parent: Option<MemorySpace>,
    /// `capacity:`, how many bytes the space holds, when declared.
    pub capacity: Option<ByteSize>,
    /// `bandwidth:`, how fast data moves into this space, when declared.
    pub bandwidth: Option<Bandwidth>,
    /// `clock: 1.98 GHz` — the clock a `B/cyc` bandwidth on this space is denominated in, in Hz.
    ///
    /// Required to convert this space's rate against a `B/s` one. Without it a path that mixes
    /// `B/cyc` and `B/s` is not derivable, because inventing a clock is exactly the silent
    /// conversion PREDICTIONS.md decision 5 forbids.
    pub clock_hz: Option<u64>,
    /// `managed:`, whether moving data in and out needs an explicit `transfer`.
    pub managed: Management,
    /// `granule:`, the unit a placement's size is rounded up to, when declared.
    pub granule: Option<ByteSize>,
    /// The execution level this space is private to (`scope: sm` etc.); `None` = unscoped.
    pub scope: Option<Scope>,
    /// `replicas: 132` — how many instances of this space the device contains.
    ///
    /// A scoped space's `bandwidth:` is the rate of ONE instance, while an enclosing device-scoped
    /// space quotes an aggregate. Composing the two without this count silently adds a per-SM rate
    /// to a device-wide one, which is how `L2->SMEM` stayed at -87% even after the containment fix:
    /// L2's 12 TB/s is 6060 B/cyc device-wide against SMEM's 128 per SM, so the larger term simply
    /// vanished.
    pub replicas: Option<u64>,
    /// `overcommit`: opt out of the *cumulative* budget error — the working set placed here may
    /// exceed `capacity` (downgraded to a warning). The programmer asserts the tiles do not all
    /// coexist, so the conservative sum should not block them.
    pub overcommit: bool,
    /// `crossing: streamed` -- how data enters this space from its parent, which decides whether a
    /// walk through it sums its legs or takes the slowest. Defaults to `sequenced`,
    /// which is the pre-existing behaviour.
    pub crossing: Crossing,
    /// `node: 1` -- the NUMA node this space's memory physically lives on.
    ///
    /// The one field here a backend has to ACT on rather than check against. Every other
    /// property describes the space so the compiler can admit or refuse a placement; this one
    /// says where to put the bytes, and a runtime that ignores it hands back memory on
    /// whichever node happened to fault first. The space name cannot carry it: a name reaches
    /// the backend as a hash, and two nodes of one machine are otherwise indistinguishable.
    ///
    /// `None` means the space is not a NUMA domain, which is every space on every accelerator
    /// in fleet/ -- a GPU has one memory and nothing to choose between.
    pub numa_node: Option<u64>,
    /// The `///` lines above the declaration.
    pub doc_comment: Option<String>,
}

/// `impl Transfer<Memory::A, Memory::B> for Topology::X { fn ... }` — a transfer lowering: the
/// code a movement across this edge emits, written in Vx and compiled by our own front end
/// (`docs/custom_transfer_contract.md`).
///
/// Distinct from the two existing `transfer` forms on purpose. The topology clause
/// (`transfer Memory::A -> Memory::B : cost`) declares that an edge EXISTS and what it costs; the
/// expression (`transfer(x, Memory::B)`) asks for a movement. This declares HOW the movement is
/// performed — and its `relaxed|sync` grade, its aliasing and its overhead are eventually read off
/// the body rather than declared, which is what supersedes the edge clause's declared marker.
///
/// The `for Topology::X` clause is required. Without it a lowering was keyed on the edge pair
/// alone, across the whole compilation — so an A100 and an H100 could not both implement
/// `Memory::L2 -> Memory::SMEM` even though `cp.async` and TMA are different instructions and
/// each machine file declares that edge for itself. The edge clause always lived inside a
/// `Topology { ... }` block; only the `impl` had escaped its machine.
#[derive(Debug, PartialEq, Clone)]
pub struct TransferImplDecl {
    /// The edge this lowering implements, `from -> to`. Checked against the declared edges of
    /// `topology`: a lowering for an edge that topology does not declare is meaningless.
    pub from: MemorySpace,
    /// The space the movement ends in.
    pub to: MemorySpace,
    /// The machine this lowering is for. A lowering is code for one part's edge, so it names
    /// that part; two topologies may implement the same edge differently.
    pub topology: crate::syntax::Topology,
    /// The lowering's functions, `fn move(...)` by convention. Parsed as ordinary Vx functions,
    /// and treated as such downstream: macro-expanded, structurally checked (E6015), and
    /// type-checked like impl methods (#353). Not yet emitted as code, and the `raw::`
    /// primitive obligations are not yet enforced. Which shapes are legal (copy fills a `dst`;
    /// an alias returns a view) is sema's question, not the parser's, so the parser accepts any
    /// `fn` items here.
    pub methods: Vec<Function>,
    /// The `///` lines above the declaration.
    pub doc_comment: Option<String>,
}

/// A top-level `const NAME : T = value;`. Its value is known while compiling, and each use
/// of `NAME` is replaced by it, as a literal of type `T`.
#[derive(Debug, PartialEq, Clone)]
pub struct ConstDecl {
    /// The constant's name.
    pub name: Symbol,
    /// The declared type.
    pub ty: Type,
    /// The value, as written.
    pub expr: Expr,
    /// Where the declaration is in the source.
    pub span: Span,
    /// The `///` lines above the declaration.
    pub doc_comment: Option<String>,
}

/// One parsed source file: every item it declares.
#[derive(Debug, PartialEq, Clone)]
pub struct Program {
    /// Macro calls written where an item goes, rather than inside a function body. They are
    /// held here between parsing and macro expansion, which replaces each one with the items
    /// it produced and empties this list.
    pub item_macros: Vec<crate::syntax::stmt::MacroCallStmt>,
    /// The module's name. The pipeline sets it after parsing, from the source file's name.
    pub module_path: Symbol,
    /// The `import` lines.
    pub imports: Vec<ImportDecl>,
    /// The macro definitions.
    pub macros: Vec<MacroDefDecl>,
    /// The `extern` function declarations.
    pub externs: Vec<ExternDecl>,
    /// The `struct` declarations.
    pub structs: Vec<StructDecl>,
    /// The `enum` declarations.
    pub enums: Vec<EnumDecl>,
    /// The `trait` declarations.
    pub traits: Vec<TraitDecl>,
    /// The `impl` blocks.
    pub impls: Vec<ImplBlock>,
    /// The functions written at the top level.
    pub functions: Vec<Function>,
    /// Top-level `const` items.
    pub consts: Vec<ConstDecl>,
    /// User-defined topologies declared in this program (`Topology <Name> { ... }`). Like
    /// `memories`, the full descriptors live here on the AST — *not* a process-global registry —
    /// so declarations never leak between compilations and the parallel pipeline needs no lock.
    /// Sema indexes them via `GlobalAstEnv` and seeds the per-compilation transfer cost graph.
    pub topologies: Vec<crate::arch::TopologyDecl>,
    /// User-defined memory spaces declared in this program (`Memory <Name> { ... }`). Unlike
    /// topologies, the full descriptors live here on the AST (not a global registry); sema
    /// indexes them via `GlobalAstEnv` and codegen reads them from the `Program`.
    pub memories: Vec<MemoryDecl>,
    /// Transfer lowerings declared in this program (`impl transfer A -> B { ... }`). Parsed and
    /// carried; nothing consumes them yet — the checks in `docs/custom_transfer_contract.md` land
    /// against this field.
    pub transfer_impls: Vec<TransferImplDecl>,
}

/// An older name for [`Program`].
pub type VxModule = Program;
/// An older name for [`Function`].
pub type VxFunction = Function;

impl Program {
    /// Adds a top-level function to the program.
    pub fn add(&mut self, func: VxFunction) {
        self.functions.push(func);
    }

    /// A copy of the program for sharing between compile workers: the bodies of functions that are
    /// not generic are left out, and everything else is kept.
    #[must_use]
    pub fn clone_signature(&self) -> Self {
        Self {
            module_path: self.module_path.clone(),
            // Not carried: a signature-only copy is taken after expansion, when the list is
            // already empty, and carrying it would expand the same call a second time.
            item_macros: Vec::new(),
            imports: self.imports.clone(),
            macros: self.macros.clone(),
            externs: self.externs.clone(),
            structs: self.structs.clone(),
            enums: self.enums.clone(),
            traits: self.traits.clone(),
            impls: self.impls.iter().map(ImplBlock::clone_signature).collect(),
            functions: self
                .functions
                .iter()
                .map(|f| f.clone_signature(false))
                .collect(),
            consts: self.consts.clone(),
            topologies: self.topologies.clone(),
            memories: self.memories.clone(),
            // Lowering bodies are KEPT, unlike `functions`/`impls`. They used to be stripped
            // for the same reason as everything else -- the signature clone exists so the
            // parallel pipeline can share a light Program -- but a lowering's body is now a
            // fact the checker reads at every transfer site: it is what the traffic count is
            // derived FROM (#353). Stripped, the count came back as zero bytes with no
            // indication anything was missing, which is the failure a derived figure must
            // never have. They are copy loops, a few statements each; the weight argument
            // does not survive contact with needing them.
            transfer_impls: self
                .transfer_impls
                .iter()
                .map(|t| TransferImplDecl {
                    from: t.from.clone(),
                    to: t.to.clone(),
                    topology: t.topology.clone(),
                    methods: t.methods.iter().map(|f| f.clone_signature(true)).collect(),
                    doc_comment: t.doc_comment.clone(),
                })
                .collect(),
        }
    }
}
