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
    ArrayExpr, BorrowExpr, ElementType, EnumVariantExpr, Expr, ExprStmtStmt, FunctionCallExpr,
    IdentifierExpr, LetDeclStmt, MethodCallExpr, NumberExpr, Placement, Span, Statement,
    StructInitExpr, Type, TypeChecker, UnsafeBlockExpr, VecMacroExpr,
};
use crate::hir::expr::expected_numeric_elem;
use std::collections::HashMap;

/// The parts of a tensor type: its element type, its dimensions and where it is placed.
type TensorParts<'a> = (
    &'a ElementType,
    &'a Vec<crate::syntax::Dim>,
    &'a Option<Placement>,
);

/// A struct's generic parameter names and its fields with their declared types.
type StructGenericsAndFields = (Vec<String>, Vec<(crate::symbol::Symbol, Type)>);

/// One payload argument of an enum variant expression, as the error messages name it.
struct EnumPayloadArg<'a> {
    /// The argument's position, counted from zero.
    index: usize,
    /// The enum's name without its type arguments.
    enum_name: &'a str,
    /// The variant the argument is passed to.
    variant: &'a crate::symbol::Symbol,
    /// Where the whole variant expression is in the source.
    span: &'a Span,
}

impl TypeChecker<'_> {
    /// An enum expression's name read back as a type. The parser re-serializes the turbofish
    /// arguments into the name (`Option<*mut i8>`), so recovering them needs the type parser
    /// (Vx#415). `None` when the name carries no arguments.
    fn enum_name_as_type(&self, enum_name: &str) -> Option<Type> {
        match crate::parser::types::parse_type_text(enum_name)? {
            Type::GenericInstance(base, args) => Some(Type::GenericInstance(
                base,
                args.into_iter()
                    .map(|a| self.as_scoped_generic(a))
                    .collect(),
            )),
            _ => None,
        }
    }

    /// The type arguments that name carries, empty when it carries none.
    fn enum_name_type_args(&self, enum_name: &str) -> Vec<Type> {
        match self.enum_name_as_type(enum_name) {
            Some(Type::GenericInstance(_, args)) => args,
            _ => Vec::new(),
        }
    }

    /// A bare name that is not a declared nominal is a generic parameter still in scope, not a
    /// struct: `Option<T>::Some(v: T)` has to match its own `Generic("T")` payload (#242).
    /// `void` is built in, so it is never declared, and stays `void`.
    fn as_scoped_generic(&self, ty: Type) -> Type {
        match ty {
            Type::Struct(name, None)
                if !self.env.structs.contains_key(&name)
                    && !self.env.enums.contains_key(&name)
                    && !crate::syntax::is_void_ty(&Type::Struct(name.clone(), None)) =>
            {
                Type::Generic(name, None)
            }
            other => other,
        }
    }

    /// Checks an enum variant expression such as `Option<i32>::Some(1)` and returns the enum's
    /// type.
    ///
    /// # Panics
    /// When `expr` is not an enum variant expression. The caller dispatches on the expression's
    /// kind, so that is a bug in the checker.
    pub(crate) fn check_enumvariant_expr(&mut self, expr: &mut Expr, consume: bool) -> Type {
        match expr {
            Expr::EnumVariant(variant_expr) => {
                self.check_enum_variant_payload(variant_expr, consume);

                let enum_name = &variant_expr.enum_name;
                if let Some(ty) = self.enum_name_as_type(enum_name) {
                    return ty;
                }
                Type::Enum(enum_name.clone(), None)
            }
            _ => panic!("Expected IndexAccess, got {expr:?}"),
        }
    }

    /// Reports what is wrong with an enum variant expression: an enum or a variant that does not
    /// exist, a payload that is missing, not wanted or of the wrong length, and payload arguments
    /// of the wrong type.
    fn check_enum_variant_payload(&mut self, variant_expr: &mut EnumVariantExpr, consume: bool) {
        let enum_name = &variant_expr.enum_name;
        let variant = &variant_expr.variant_name;
        let span = &variant_expr.span;
        let actual_enum_name = Self::enum_base_name(enum_name);

        let Some(enum_decl) = self.env.enums.get(actual_enum_name) else {
            self.enum_variant_error(crate::diagnostic::DiagnosticCode::E2003, span, || {
                format!("Unknown enum {enum_name}")
            });
            return;
        };
        let Some((_, expected_payload)) = enum_decl.variants.iter().find(|(n, _)| n == variant)
        else {
            self.enum_variant_error(crate::diagnostic::DiagnosticCode::E2004, span, || {
                format!("Enum {actual_enum_name} does not have variant {variant}")
            });
            return;
        };
        self.check_enum_payload(
            variant_expr,
            enum_decl,
            expected_payload.as_deref(),
            consume,
        );
    }

    /// An enum expression's name without its type arguments: `Option` for `Option<i32>`.
    fn enum_base_name(enum_name: &str) -> &str {
        if let Some(idx) = enum_name.find('<') {
            &enum_name[..idx]
        } else {
            enum_name
        }
    }

    /// Checks the payload of an enum variant expression against `expected_payload`: the payload
    /// types the variant declares, or `None` for a variant that takes no payload.
    fn check_enum_payload(
        &mut self,
        variant_expr: &mut EnumVariantExpr,
        enum_decl: &crate::syntax::EnumDecl,
        expected_payload: Option<&[Type]>,
        consume: bool,
    ) {
        let EnumVariantExpr {
            enum_name,
            variant_name: variant,
            payload,
            span,
        } = variant_expr;
        let actual_enum_name = Self::enum_base_name(enum_name);
        let Some(expr_payload) = payload else {
            if expected_payload.is_some() {
                self.enum_variant_error(crate::diagnostic::DiagnosticCode::E3009, span, || {
                    format!("Enum variant {actual_enum_name}::{variant} expects a payload")
                });
            }
            return;
        };
        let Some(exp_types) = expected_payload else {
            self.enum_variant_error(crate::diagnostic::DiagnosticCode::E3009, span, || {
                format!("Enum variant {actual_enum_name}::{variant} does not take a payload")
            });
            return;
        };
        if expr_payload.len() != exp_types.len() {
            self.enum_variant_error(crate::diagnostic::DiagnosticCode::E3009, span, || {
                format!(
                    "Enum variant {}::{} expects {} payload arguments, got {}",
                    actual_enum_name,
                    variant,
                    exp_types.len(),
                    expr_payload.len()
                )
            });
            return;
        }

        let mapping = self.enum_generic_mapping(enum_decl, enum_name);
        for (i, expr) in expr_payload.iter_mut().enumerate() {
            // An untyped literal takes the payload's type:
            // `Pair<i32, f64>::Of(1, 2.5)` holds an `f64`.
            let expected_ty = exp_types[i].substitute(&mapping);
            let arg = EnumPayloadArg {
                index: i,
                enum_name: actual_enum_name,
                variant,
                span,
            };
            self.check_enum_payload_arg(expr, &expected_ty, &arg, consume);
        }
    }

    /// Reports an error at an enum variant expression, unless the checker is only trying a
    /// guess. The message is built only when it is reported.
    fn enum_variant_error(
        &mut self,
        code: crate::diagnostic::DiagnosticCode,
        span: &Span,
        message: impl FnOnce() -> String,
    ) {
        if !self.speculating {
            self.errors.error_with_code(
                code,
                message(),
                Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
            );
        }
    }

    /// The enum's generic parameters paired with the type arguments written in `enum_name`:
    /// for `Option<i32>`, `T` maps to `i32`.
    fn enum_generic_mapping(
        &self,
        enum_decl: &crate::syntax::EnumDecl,
        enum_name: &str,
    ) -> HashMap<crate::symbol::Symbol, Type> {
        let mut mapping = HashMap::new();
        let ty_args = self.enum_name_type_args(enum_name);
        for (param, ty_arg) in enum_decl.generics.iter().zip(ty_args.iter()) {
            mapping.insert(param.name().into(), ty_arg.clone());
        }
        mapping
    }

    /// Checks one payload argument of an enum variant against the type the variant declares for
    /// it.
    fn check_enum_payload_arg(
        &mut self,
        expr: &mut Expr,
        expected_ty: &Type,
        arg: &EnumPayloadArg<'_>,
        consume: bool,
    ) {
        let expr_ty = if matches!(
            expected_ty,
            Type::Generic(..) | Type::Scalar(ElementType::Generic(_))
        ) {
            self.check_expr_type_flag(expr, consume)
        } else {
            self.check_expr_expecting(expr, Some(expected_ty.clone()), consume)
        };
        // A tensor payload has no lowering: the variant's slot is
        // built with `llvm.insertvalue`, which takes primitives,
        // and the AST path emitted the tag and dropped the tensor
        // without saying so.
        if Self::is_tensor_payload(expected_ty) {
            if !self.speculating {
                self.errors.error_with_code(
                    crate::diagnostic::DiagnosticCode::E3021,
                    format!(
                        "payload argument {} of {}::{} is {}; an \
                         enum variant cannot carry a tensor",
                        arg.index + 1,
                        arg.enum_name,
                        arg.variant,
                        Self::short_type_name(expected_ty)
                    ),
                    Some(crate::diagnostic::SourceSpan::from_ast_span(arg.span)),
                );
            }
            return;
        }
        if !self.is_assignable(expected_ty, &expr_ty)
            && !matches!(expected_ty, Type::Generic(_, _))
            && !self.speculating
        {
            self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E3008,
                format!(
                    "Type mismatch in payload argument {} for {}::{}: expected {}, got {}",
                    arg.index + 1,
                    arg.enum_name,
                    arg.variant,
                    expected_ty,
                    expr_ty
                ),
                Some(crate::diagnostic::SourceSpan::from_ast_span(arg.span)),
            );
        }
    }

    /// Type an untyped numeric literal (#240). An unsuffixed literal reaches the checker with
    /// `ty: None`; it adopts the *expected* scalar type of its context when that is a compatible
    /// numeric type (integer literal → integer type, float literal → float type), and otherwise
    /// falls back to the spelling-based default. The inferred type is written back onto the node so
    /// every downstream consumer — the flat lowerer and the AST codegen alike — sees the concrete
    /// type. A suffixed literal already carries its type and is returned unchanged.
    pub(crate) fn check_number_literal(&mut self, n: &mut NumberExpr) -> Type {
        if let Some(el) = &n.ty {
            let el = el.clone();
            self.check_usize_literal(n, &el);
            return Type::Scalar(el);
        }
        let elem = self
            .expected_type
            .as_ref()
            .and_then(|t| expected_numeric_elem(t, &n.value))
            .unwrap_or_else(|| crate::parser::expr::default_number_elem(&n.value));
        n.ty = Some(elem.clone());
        self.check_usize_literal(n, &elem);
        Type::Scalar(elem)
    }

    /// A `usize` never has its top bit set, which is what lets its arithmetic be marked as
    /// never wrapping, so a literal past the largest `i64` is refused rather than stored.
    pub(crate) fn check_usize_literal(&mut self, n: &NumberExpr, elem: &ElementType) {
        if *elem != ElementType::USize
            || self.speculating
            || elem.accepts_integer_literal(n.value.as_ref()) != Some(false)
        {
            return;
        }
        self.errors.error_with_code(
            crate::diagnostic::DiagnosticCode::E3045,
            format!(
                "{} does not fit `usize`, which holds 0 up to {}",
                n.value,
                i64::MAX
            ),
            // A number literal has no span of its own: spans take part in comparing types,
            // and a literal is also a tensor dimension.
            None,
        );
    }

    /// The tensor an operand denotes, seen through the wrappers that do not
    /// change what it is: a borrow, a pointer, a placement, a reference.
    ///
    /// `&a @ &b` is the non-consuming spelling of `a @ b` -- both are matmuls
    /// over the same two tensors, and one reads its operands where the other
    /// moves them (#335). The wrappers are peeled in a loop rather than once,
    /// so a borrow of a placed tensor (`&Ref<Tensor<f32, [?, ?]>, Memory::GPU_HBM>`,
    /// which is what a resident weight is) resolves too.
    pub(crate) fn as_tensor_operand(t: &Type) -> Option<TensorParts<'_>> {
        let mut inner = t;
        loop {
            match inner {
                Type::Tensor(elem, dims, top) => return Some((elem, dims, top)),
                Type::Borrow { inner: i, .. }
                | Type::Pointer(i, _, _)
                | Type::Pinned(i, _)
                | Type::Ref(i, _) => inner = i.as_ref(),
                _ => return None,
            }
        }
    }

    /// A note for a mismatch between two tensors of different rank: no cast changes a rank,
    /// and the Debug rendering of the two types buries the one number that differs.
    pub(crate) fn rank_note(expected: &Type, got: &Type) -> String {
        match (
            Self::as_tensor_operand(expected),
            Self::as_tensor_operand(got),
        ) {
            (Some((_, ed, _)), Some((_, gd, _))) if ed.len() != gd.len() => format!(
                "; rank {} does not match rank {}, and rank is static",
                gd.len(),
                ed.len()
            ),
            _ => String::new(),
        }
    }

    /// A note when a `const` dimension in a parameter met a `?` in the argument: the name
    /// stands for a compile-time extent, and the argument has none in that position.
    pub(crate) fn const_extent_note(param: &Type, arg: &Type) -> String {
        let (Some((_, pd, _)), Some((_, ad, _))) =
            (Self::as_tensor_operand(param), Self::as_tensor_operand(arg))
        else {
            return String::new();
        };
        for (i, (p, a)) in pd.iter().zip(ad.iter()).enumerate() {
            if let (
                crate::syntax::Dim::Static(crate::syntax::Expr::Identifier(id)),
                crate::syntax::Dim::Dyn,
            ) = (p, a)
            {
                return format!(
                    "; '{}' is a compile-time extent, but dimension {i} of the argument has a \
                     run-time extent",
                    id.name
                );
            }
        }
        String::new()
    }

    /// A note when a dimension name shared by two parameters met two different extents. The
    /// useful message names the conflict rather than one argument, so it says which argument
    /// bound the name and what this one asked for instead.
    pub(crate) fn dim_conflict_note(
        param: &Type,
        arg: &Type,
        mapping: &HashMap<crate::symbol::Symbol, Type>,
        bound_by: &HashMap<crate::symbol::Symbol, usize>,
        arg_index: usize,
    ) -> String {
        let (Some((_, pd, _)), Some((_, ad, _))) =
            (Self::as_tensor_operand(param), Self::as_tensor_operand(arg))
        else {
            return String::new();
        };
        for (p, a) in pd.iter().zip(ad.iter()) {
            let crate::syntax::Dim::Static(crate::syntax::Expr::Identifier(id)) = p else {
                continue;
            };
            let Some(bound) = mapping.get(&id.name).and_then(Self::extent_of) else {
                continue;
            };
            // What this argument puts in that position: an extent, or another dimension name.
            let actual = match a {
                crate::syntax::Dim::Static(crate::syntax::Expr::Number(n)) => n.value.clone(),
                crate::syntax::Dim::Static(crate::syntax::Expr::Identifier(id2)) => {
                    id2.name.clone()
                }
                _ => continue,
            };
            if bound == actual.as_ref() {
                continue;
            }
            let bound_where = match bound_by.get(&id.name) {
                Some(earlier) => format!("argument {earlier}"),
                None => "an explicit generic argument".to_string(),
            };
            return format!(
                "; '{}' is bound to {bound} by {bound_where}, but argument {arg_index} has \
                 extent {actual}",
                id.name
            );
        }
        String::new()
    }

    /// The element type of any tensor operand, static or dynamic, under the checker-level
    /// wrappers. `as_tensor_operand` answers only for a statically shaped tensor, because it
    /// hands back the dimensions; a caller that needs to know *whether* it has a tensor, or only
    /// what it is made of, asks here and accepts run-time extents too (Vx#399).
    pub(crate) fn tensor_operand_elem(t: &Type) -> Option<&ElementType> {
        let mut inner = t;
        loop {
            match inner {
                Type::Tensor(elem, _, _) => return Some(elem),
                Type::Borrow { inner: i, .. }
                | Type::Pointer(i, _, _)
                | Type::Pinned(i, _)
                | Type::Ref(i, _) => inner = i.as_ref(),
                _ => return None,
            }
        }
    }

    /// A rank-1 f32 tensor slice, as produced by `q[i]` (S1).
    ///
    /// The rank is part of the question. Every contract downstream of this reads the slice with
    /// a single-index `vector.load`, so a higher-rank tensor answered `true` here reaches codegen
    /// as an op MLIR rejects, with no diagnostic of its own to explain it.
    pub(crate) fn is_f32_slice(t: &Type) -> bool {
        let inner = match t {
            Type::Borrow { inner, .. }
            | Type::Pointer(inner, _, _)
            | Type::Pinned(inner, _)
            | Type::Ref(inner, _) => inner.as_ref(),
            other => other,
        };
        matches!(inner, Type::Tensor(ElementType::F32, dims, _) if dims.len() == 1)
    }

    /// A rank-1 f16/bf16 tensor slice: half-precision STORAGE that every slice contract widens
    /// to f32 on load (Vx#320). Storage precision and arithmetic precision are separate
    /// decisions -- a wide-slice op computes in f32 and its result IS f32; only an explicit
    /// store back into a half tensor narrows again.
    pub(crate) fn is_half_slice(t: &Type) -> bool {
        let inner = match t {
            Type::Borrow { inner, .. }
            | Type::Pointer(inner, _, _)
            | Type::Pinned(inner, _)
            | Type::Ref(inner, _) => inner.as_ref(),
            other => other,
        };
        matches!(
            inner,
            Type::Tensor(ElementType::F16 | ElementType::BF16, dims, _) if dims.len() == 1
        )
    }

    /// Whether a payload type is a tensor, through the wrappers that restate where it lives
    /// without changing how it is represented.
    fn is_tensor_payload(t: &Type) -> bool {
        match t {
            Type::Tensor(..) => true,
            Type::Verified(inner) | Type::Pinned(inner, _) | Type::Ref(inner, _) => {
                Self::is_tensor_payload(inner)
            }
            _ => false,
        }
    }

    /// A short human name for a type, for diagnostics that only need to say what KIND of thing
    /// the programmer wrote. The `{:?}` rendering used elsewhere prints the whole AST of every
    /// dimension expression, which buries the one word the reader needs.
    pub(crate) fn short_type_name(t: &Type) -> String {
        match t {
            Type::Scalar(e) => format!("a scalar {e:?}"),
            // A placed tensor says where it is, the same as the `Pinned` arm below --
            // they are two spellings of one fact, and a diagnostic that named it for
            // only one of them lost the detail as soon as `transfer` started
            // producing the other.
            Type::Tensor(e, _, Some(p)) => format!(
                "a tensor of {e:?} placed on Topology::{}",
                p.topology.display_name()
            ),
            Type::Tensor(e, _, _) => format!("a tensor of {e:?}"),
            Type::Pinned(inner, top) => format!(
                "{} placed on Topology::{}",
                Self::short_type_name(inner),
                top.display_name()
            ),
            Type::Ref(inner, mem) => {
                format!("{} in Memory::{}", Self::short_type_name(inner), mem.name())
            }
            Type::Struct(name, _) => format!("a struct {name}"),
            other => format!("{other:?}"),
        }
    }

    /// Checks an array literal such as `[1, 2, 3]` and returns its tensor type, or
    /// `Type::Unknown` after reporting why it has none.
    ///
    /// # Panics
    /// When `expr` is not an array literal. The caller dispatches on the expression's kind, so
    /// that is a bug in the checker.
    pub(crate) fn check_array_expr(&mut self, expr: &mut Expr) -> Type {
        match expr {
            Expr::Array(ArrayExpr { elements, span }) => {
                // The array's element type is its first element's — an integer array literal
                // (`[10, 20, 30]`) is `Tensor<i32, [?, ?]>`, not `Tensor<f32, [?, ?]>` (#240). Later elements are
                // checked expecting that type, so untyped literals adopt it.
                let span = *span;
                // An array literal lowers to `tensor.from_elements`, so its elements have to be
                // scalars. A non-scalar first element used to leave `elem_ty` at its `f32`
                // default and report `Tensor<f32, [?, ?]>` for something that is nothing of the sort,
                // and codegen then died building `tensor<Nx tensor<...>>` — an internal error on
                // a two-line program, where the checker had the type in its hand all along
                // (Vx#354). Empty has no element type to report at all.
                if elements.is_empty() {
                    self.errors.error_with_code(
                        crate::diagnostic::DiagnosticCode::E3018,
                        "an empty array literal has no element type; annotate the binding or \
                         give it at least one element"
                            .to_string(),
                        Some(crate::diagnostic::SourceSpan::from_ast_span(&span)),
                    );
                    return Type::Unknown;
                }
                let Some(elem_ty) = self.check_array_elements(elements, span) else {
                    return Type::Unknown;
                };
                Self::array_literal_type(elements, elem_ty, span)
            }
            _ => panic!("Expected IndexAccess, got {expr:?}"),
        }
    }

    /// Checks every element of a non-empty array literal and returns the element type they
    /// share, or `None` after reporting an element that cannot be in an array literal.
    fn check_array_elements(&mut self, elements: &mut [Expr], span: Span) -> Option<ElementType> {
        // A NESTED literal is how a multi-dimensional initializer is written
        // (`Tensor<f32>([[1.0, 2.0], [3.0, 4.0]])`), and those rows are consumed as a
        // shape by `initializer_shape` rather than lowered through
        // `tensor.from_elements`. So the restriction is on non-scalar VALUES -- a
        // tensor variable -- not on nested literals, which stay legal.
        let nested = matches!(elements.first(), Some(Expr::Array(_)));
        // The element type comes from the first element that is not a bare number
        // literal, so `[1.0, d]` takes `d`'s type. That element is checked first.
        let lead = if nested {
            0
        } else {
            elements
                .iter()
                .position(|e| !crate::hir::expr::is_untyped_numeric_literal(e))
                .unwrap_or(0)
        };
        let order = std::iter::once(lead).chain((0..elements.len()).filter(|&i| i != lead));
        let mut elem_ty = ElementType::F32;
        for i in order {
            let el = &mut elements[i];
            if i == lead {
                let first = self.check_expr_type(el);
                match (first, nested) {
                    // A nested row reports `Tensor<e, [?, ?]>`; the initializer's element type
                    // is that inner `e`, which is also more accurate than the `f32`
                    // default this arm used to fall through to.
                    (Type::Scalar(e), false) | (Type::Tensor(e, _, _), true) => elem_ty = e,
                    (other, _) => {
                        self.errors.error_with_code(
                            crate::diagnostic::DiagnosticCode::E3018,
                            format!(
                                "an array literal's elements must be scalars, but this \
                                 one holds {}; an array of tensors has no lowering \
                                 (`tensor.from_elements` takes scalar elements)",
                                Self::short_type_name(&other)
                            ),
                            Some(crate::diagnostic::SourceSpan::from_ast_span(&span)),
                        );
                        return None;
                    }
                }
            } else if nested {
                // Every row of a nested initializer must itself be a literal row: a
                // mixed `[[1.0], x]` has no shape, and letting it through would put the
                // ragged case back on codegen.
                if !matches!(el, Expr::Array(_)) {
                    self.errors.error_with_code(
                        crate::diagnostic::DiagnosticCode::E3018,
                        "a nested array literal's rows must all be literals; this one \
                         mixes a row with something else"
                            .to_string(),
                        Some(crate::diagnostic::SourceSpan::from_ast_span(&span)),
                    );
                    return None;
                }
                self.check_expr_type(el);
            } else {
                self.check_later_array_element(el, &elem_ty);
            }
        }
        Some(elem_ty)
    }

    /// Checks an element that comes after the one that set the array's element type, and
    /// reports it when its type is a different scalar.
    fn check_later_array_element(&mut self, el: &mut Expr, elem_ty: &ElementType) {
        let ty = self.check_expr_expecting(el, Some(Type::Scalar(elem_ty.clone())), true);
        if let Type::Scalar(e) = &ty {
            if e != elem_ty {
                self.errors.error_with_code(
                    crate::diagnostic::DiagnosticCode::E3018,
                    format!(
                        "this array literal's elements are {elem_ty}, but one is {e}; \
                         convert it with `as {elem_ty}`"
                    ),
                    Some(crate::diagnostic::SourceSpan::from_ast_span(&el.span())),
                );
            }
        }
    }

    /// The tensor type of an array literal whose elements are `elem_ty`, or `Type::Unknown`
    /// when its nested rows have no single shape.
    fn array_literal_type(elements: &[Expr], elem_ty: ElementType, span: Span) -> Type {
        // A literal knows its shape: `[1, 2, 3]` is `Tensor<i32, [3]>`, and
        // `[[1, 2], [3, 4]]` is `Tensor<i32, [2, 2]>`.
        let nested = elements.iter().any(|el| matches!(el, Expr::Array(_)));
        let shape = if nested {
            ArrayExpr::new(elements.to_vec(), span).initializer_shape()
        } else {
            Some(vec![elements.len()])
        };
        let Some(shape) = shape else {
            return Type::Unknown;
        };
        let dims = shape
            .iter()
            .map(|n| {
                crate::syntax::Dim::Static(Expr::Number(crate::syntax::expr::NumberExpr::new(
                    n.to_string(),
                    Some(ElementType::I32),
                    span,
                )))
            })
            .collect();
        Type::Tensor(elem_ty, dims, None)
    }

    /// Checks a struct literal such as `Pair { a : 1, b : true }` and returns the struct's type.
    ///
    /// # Panics
    /// When `expr` is not a struct literal. The caller dispatches on the expression's kind, so
    /// that is a bug in the checker.
    pub(crate) fn check_structinit_expr(&mut self, expr: &mut Expr, consume: bool) -> Type {
        match expr {
            Expr::StructInit(StructInitExpr {
                name,
                fields,
                type_id,
                span: _,
            }) => {
                let resolved_name = name.clone();
                let (base_name, mut generic_args) = self.struct_name_and_type_args(&resolved_name);

                // Attach the struct's resolved GID so downstream consumers (the flat-HIR lowerer)
                // reach its registry layout without re-resolving the name (#199). Resolved through the
                // registry-backed `ModuleInterface` -- the AST-free import oracle -- rather than a
                // borrowed-AST side map (#219). A generic instance's monomorph has a distinct identity,
                // so its GID is left `None` here; so is the empty-registry path (the sequential driver /
                // legacy AST harnesses), where only the AST codegen runs and never reads this field.
                let resolved_gid = if generic_args.is_empty() {
                    let mi: &dyn crate::registry::ModuleInterface = &*self.worker.global.registry;
                    mi.resolve_unique_nominal(&base_name)
                } else {
                    None
                };
                *type_id = resolved_gid;

                let struct_info = self.struct_generics_and_fields(&base_name, resolved_gid);

                if let Some((generic_names, struct_fields)) = struct_info {
                    // `Pair { a : 1i64, b : true }` with no type arguments written: they come from
                    // what the literal is expected to be, else from its field values, as in Rust.
                    // The fields are checked for real below, so the probe here is speculative.
                    if generic_args.is_empty() && !generic_names.is_empty() {
                        if let Some(inferred) = self.infer_struct_generic_args(
                            &base_name,
                            &generic_names,
                            &struct_fields,
                            fields,
                        ) {
                            generic_args = inferred;
                            let spelled: Vec<String> = generic_args
                                .iter()
                                .map(std::string::ToString::to_string)
                                .collect();
                            *name = format!("{}<{}>", base_name, spelled.join(", ")).into();
                            *type_id = None;
                        }
                    }
                    let expected_fields =
                        Self::substituted_fields(&generic_names, &generic_args, &struct_fields);
                    self.check_struct_init_fields(
                        &resolved_name,
                        &expected_fields,
                        fields,
                        consume,
                    );
                } else {
                    if !self.speculating {
                        self.report_unknown_struct(&base_name, &resolved_name);
                    }
                    for (_, f_expr) in fields.iter_mut() {
                        self.check_expr_type_flag(f_expr, consume);
                    }
                }

                if generic_args.is_empty() {
                    // The *returned* type keeps GID `None` (as before) so struct type-identity
                    // comparisons are unchanged; the resolved GID rides on the `type_id` field for
                    // the flat-HIR lowerer to consume (#199).
                    Type::Struct(resolved_name, None)
                } else {
                    Type::GenericInstance(Box::new(Type::Struct(base_name, None)), generic_args)
                }
            }
            _ => panic!("Expected IndexAccess, got {expr:?}"),
        }
    }

    /// The struct's own name and the type arguments written after it: `Pair<i32, f64>` gives
    /// `Pair` and `[i32, f64]`. A name with no type arguments is returned as it is.
    fn struct_name_and_type_args(
        &self,
        name: &crate::symbol::Symbol,
    ) -> (crate::symbol::Symbol, Vec<Type>) {
        let mut base_name = name.clone();
        let mut generic_args = Vec::new();

        if let Type::GenericInstance(inner, args) = self.parse_ty_str(name) {
            if let Type::Struct(s, _) = *inner {
                base_name = s;
            }
            generic_args = args;
        }
        (base_name, generic_args)
    }

    /// The struct's fields with each generic parameter replaced by its type argument.
    fn substituted_fields(
        generic_names: &[String],
        generic_args: &[Type],
        struct_fields: &[(crate::symbol::Symbol, Type)],
    ) -> Vec<(crate::symbol::Symbol, Type)> {
        let mut mapping = std::collections::HashMap::new();
        for (i, param) in generic_names.iter().enumerate() {
            if i < generic_args.len() {
                mapping.insert(param.as_str().into(), generic_args[i].clone());
            }
        }
        struct_fields
            .iter()
            .map(|(field, ty)| (field.clone(), ty.substitute(&mapping)))
            .collect()
    }

    /// The struct's generic parameter names + declared field types: a local AST struct (or
    /// a generated one) first, else an *imported* struct from the frozen registry's
    /// `structs` table (#219) — so constructing an imported struct type-checks with no AST.
    /// A local definition shadows a same-named import.
    fn struct_generics_and_fields(
        &self,
        base_name: &crate::symbol::Symbol,
        resolved_gid: Option<crate::gid::TypeId>,
    ) -> Option<StructGenericsAndFields> {
        self.env
            .structs
            .get(base_name.as_ref())
            .map(|s| {
                (
                    s.generics.iter().map(|g| g.name().to_string()).collect(),
                    s.fields.clone(),
                )
            })
            .or_else(|| {
                self.mono
                    .generated_structs
                    .iter()
                    .find(|s| s.name == *base_name)
                    .map(|s| {
                        (
                            s.generics.iter().map(|g| g.name().to_string()).collect(),
                            s.fields.clone(),
                        )
                    })
            })
            .or_else(|| {
                // GID-keyed since #291: resolve the base name to its unique GID (the
                // instance-annotation `resolved_gid` when it was computed, else a fresh
                // bare-name resolution for a generic instance's base). An ambiguous name
                // declines to "unknown struct" rather than picking an arbitrary module's.
                let reg = &self.worker.global.registry;
                let gid = resolved_gid.or_else(|| reg.resolve_unique_nominal(base_name))?;
                reg.structs.get(&gid).map(|sf| {
                    (
                        sf.generics
                            .iter()
                            .map(std::string::ToString::to_string)
                            .collect(),
                        sf.fields.clone(),
                    )
                })
            })
    }

    /// Works out the type arguments of a generic struct literal that does not write them: from
    /// the type the literal is expected to have, else from the values given to its fields.
    /// `None` when some type parameter is still unknown after both.
    fn infer_struct_generic_args(
        &mut self,
        base_name: &crate::symbol::Symbol,
        generic_names: &[String],
        struct_fields: &[(crate::symbol::Symbol, Type)],
        fields: &[(crate::symbol::Symbol, Expr)],
    ) -> Option<Vec<Type>> {
        let mut inferred: HashMap<crate::symbol::Symbol, Type> = HashMap::new();
        if let Some(Type::GenericInstance(b, a)) = self.expected_type.clone() {
            if matches!(*b, Type::Struct(ref n, _) if n == base_name)
                && a.len() == generic_names.len()
            {
                for (g, t) in generic_names.iter().zip(a) {
                    inferred.insert(g.as_str().into(), t);
                }
            }
        }
        if inferred.len() < generic_names.len() {
            let saved = self.speculating;
            self.speculating = true;
            for (f_name, f_ty) in struct_fields {
                if let Some((_, f_expr)) = fields.iter().find(|(n, _)| n == f_name) {
                    let mut probe = f_expr.clone();
                    let actual = self.check_expr_type_flag(&mut probe, false);
                    self.unify_types(f_ty, &actual, &mut inferred);
                }
            }
            self.speculating = saved;
        }
        if generic_names
            .iter()
            .all(|g| inferred.contains_key(g.as_str()))
        {
            Some(
                generic_names
                    .iter()
                    .map(|g| inferred[g.as_str()].clone())
                    .collect(),
            )
        } else {
            None
        }
    }

    /// Checks the values in a struct literal against the struct's fields: every declared field
    /// must be given a value of its type, and no other field may be named.
    fn check_struct_init_fields(
        &mut self,
        resolved_name: &crate::symbol::Symbol,
        expected_fields: &[(crate::symbol::Symbol, Type)],
        fields: &mut [(crate::symbol::Symbol, Expr)],
        consume: bool,
    ) {
        // Check missing fields and type mismatch
        for (expected_name, expected_type) in expected_fields {
            let mut found = false;
            for (f_name, f_expr) in fields.iter_mut() {
                if f_name == expected_name {
                    found = true;
                    // The field's type is the expected type of what initializes it,
                    // so `Counter { n : 0 }` writes a `u64` zero into a `u64` field
                    // rather than an `i32` one the field then refuses.
                    let f_type =
                        self.check_expr_expecting(f_expr, Some(expected_type.clone()), consume);
                    self.check_view_stored(
                        &format!("the field `{expected_name}`"),
                        f_expr,
                        &f_type,
                    );
                    if !self.is_assignable(expected_type, &f_type) && !self.speculating {
                        self.errors.push(format!(
                            "Type mismatch in struct initialization for field '{expected_name}'. Expected {expected_type}, got {f_type}"
                        ));
                    }
                    break;
                }
            }
            if !found && !self.speculating {
                self.errors.push(format!(
                    "Missing field '{expected_name}' in initialization of struct '{resolved_name}'"
                ));
            }
        }
        // Check extra fields
        for (f_name, f_expr) in fields.iter_mut() {
            if !expected_fields.iter().any(|(n, _)| n == f_name) {
                if !self.speculating {
                    self.errors
                        .push(format!("Struct '{resolved_name}' has no field '{f_name}'"));
                }
                self.check_expr_type_flag(f_expr, consume); // evaluate to find errors
            }
        }
    }

    /// Reports a struct literal whose struct is not known.
    fn report_unknown_struct(
        &mut self,
        base_name: &crate::symbol::Symbol,
        resolved_name: &crate::symbol::Symbol,
    ) {
        // Distinguish "no such struct" from "two imported modules both define it":
        // the latter used to resolve by coin flip before the bare name learned to
        // decline, and "unknown" would send the user hunting the wrong bug (#291).
        if self.worker.global.registry.is_ambiguous_nominal(base_name) {
            self.errors.push(crate::registry::ambiguous_import_message(
                "Struct",
                resolved_name,
            ));
        } else {
            self.errors
                .push(format!("Unknown struct {resolved_name} (expr.rs:2175)"));
        }
    }

    /// Rewrites a `vec![...]` expression into a block that makes a `Vec` and pushes each element,
    /// then checks that block and returns its type.
    ///
    /// # Panics
    /// When `expr` is not a `vec![...]` expression. The caller dispatches on the expression's
    /// kind, so that is a bug in the checker.
    pub(crate) fn check_vecmacro_expr(&mut self, expr: &mut Expr) -> Type {
        match expr {
            Expr::VecMacro(VecMacroExpr { elements, span }) => {
                let mut element_type = Type::Scalar(ElementType::I32); // Default
                if !elements.is_empty() {
                    let mut first = elements[0].clone();
                    element_type = self.check_expr_type(&mut first);
                }

                let var_name = format!("_vec_macro_tmp_{}", self.next_id);
                self.next_id += 1;

                let new_call = Expr::FunctionCall(FunctionCallExpr::new(
                    format!("Vec<{element_type}>::new").into(),
                    None,
                    vec![],
                    *span,
                ));

                let decl = Statement::LetDecl(LetDeclStmt::new(
                    var_name.clone(),
                    true,
                    None,
                    new_call,
                    *span,
                ));

                let mut stmts = vec![decl];

                for el in elements.clone() {
                    let push_call = Expr::MethodCall(MethodCallExpr::new(
                        Box::new(Expr::Borrow(BorrowExpr {
                            expr: Box::new(Expr::Identifier(IdentifierExpr::new(
                                var_name.clone().into(),
                                Span::default(),
                            ))),
                            is_mut: true,
                            span: Span::default(),
                        })),
                        "push".to_string().into(),
                        None,
                        vec![el],
                        *span,
                    ));
                    stmts.push(Statement::ExprStmt(ExprStmtStmt::new(
                        push_call, true, *span,
                    )));
                }

                let ret_expr =
                    Expr::Identifier(IdentifierExpr::new(var_name.clone().into(), *span));

                let block =
                    Expr::UnsafeBlock(UnsafeBlockExpr::new(stmts, Some(Box::new(ret_expr)), *span));

                *expr = block;
                self.check_expr_type(expr)
            }
            _ => panic!("Expected IndexAccess, got {expr:?}"),
        }
    }
}
