//===- macro_expand.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// AST transformations for expanding macros into inline code during compilation.
//
// This lives under parser/ rather than syntax/ because expanding a macro means parsing the
// tokens it produced back into AST, so the expander needs the parser.
//
//===----------------------------------------------------------------------===//

use crate::lexer::OwnedTokenType;
use crate::syntax::{expr, stmt, Function, MacroRule, TokenTree, VxModule};
use crate::syntax::{Delimiter, Span};
use std::collections::HashMap;

/// Replaces every macro call in a module with the code the macro stands for.
pub struct MacroExpander<'a> {
    /// The rules of every `macro_rules!` macro, by macro name.
    pub macros: &'a HashMap<crate::symbol::Symbol, Vec<MacroRule>>,
    /// The trait methods that carry a default body, across every module. Held here because
    /// filling them in is the last thing expansion does, and it has to be the last thing:
    /// an impl a macro produced is entitled to the defaults, and nothing before expansion
    /// can see it.
    pub trait_defaults: &'a crate::resolver::TraitDefaults,
}

/// Move an expression out of its place, leaving the number `0` there until the caller writes
/// the expanded expression back.
fn take_expr(expr: &mut expr::Expr) -> expr::Expr {
    std::mem::replace(
        expr,
        expr::Expr::Number(expr::NumberExpr::new(
            "0".into(),
            None,
            crate::syntax::Span::default(),
        )),
    )
}
/// Split a format string at its `{}` placeholders and put the next argument in each place.
/// Returns the pieces in the order they print, and how many placeholders there were.
/// `exprs` is the whole argument list of the macro, the format string first.
///
/// # Errors
///
/// Fails when the format string has a placeholder other than `{}`, a `}` with no `{` before
/// it, or more `{}` than there are arguments.
fn split_at_placeholders(
    text: &str,
    fmt_span: Span,
    exprs: &[expr::Expr],
    macro_name: &str,
) -> Result<(Vec<expr::Expr>, usize), String> {
    let mut pieces: Vec<expr::Expr> = Vec::new();
    let mut lit = String::new();
    let mut args = exprs.iter().skip(1);
    let mut used = 0usize;
    let mut chars = text.chars().peekable();

    while let Some(c) = chars.next() {
        match c {
            '{' if chars.peek() == Some(&'{') => {
                chars.next();
                lit.push('{');
            }
            '}' if chars.peek() == Some(&'}') => {
                chars.next();
                lit.push('}');
            }
            '{' => {
                if chars.peek() != Some(&'}') {
                    return Err(format!(
                        "{macro_name}: only `{{}}` is supported in a format string; write the value as its own argument"
                    ));
                }
                chars.next();
                let Some(arg) = args.next() else {
                    return Err(format!(
                        "{macro_name}: the format string has more `{{}}` than arguments"
                    ));
                };
                used += 1;
                if !lit.is_empty() {
                    pieces.push(expr::Expr::StringLiteral(expr::StringLiteralExpr::new(
                        std::mem::take(&mut lit),
                        fmt_span,
                    )));
                }
                pieces.push(arg.clone());
            }
            '}' => return Err(format!("{macro_name}: unmatched `}}` in the format string")),
            _ => lit.push(c),
        }
    }
    if !lit.is_empty() {
        pieces.push(expr::Expr::StringLiteral(expr::StringLiteralExpr::new(
            lit, fmt_span,
        )));
    }
    Ok((pieces, used))
}

/// Split a `print!`/`println!` format string at its `{}` placeholders and interleave the
/// arguments, so `println!("a={} b={}", x, y)` becomes the four pieces `"a="`, `x`, `" b="`, `y`.
///
/// Both lowering paths print a macro's arguments in sequence, so doing this here is the whole
/// implementation of formatting -- and it fixes both paths at once. Without it the format string
/// was printed verbatim, placeholders included, and the arguments were appended after it:
/// `println!("value={}", x)` wrote `value={}1.5`.
///
/// `{{` and `}}` are the escapes for a literal brace, as in Rust. A format spec (`{:.3}`, `{name}`)
/// is refused rather than silently printed, so the gap is visible where it is written.
///
/// A leading string with no `{}` at all is left alone: it is a label, and `println!("count: ", v)`
/// -- print the arguments in sequence -- is how the corpus already spells that.
///
/// # Errors
///
/// Fails when the format string has a placeholder other than `{}`, a `}` with no `{` before
/// it, or a number of `{}` different from the number of arguments.
fn interleave_format_args(
    exprs: Vec<expr::Expr>,
    macro_name: &str,
) -> Result<Vec<expr::Expr>, String> {
    // Only a leading string literal is a format string. `print!(x)` prints a value and has none.
    let Some(expr::Expr::StringLiteral(fmt)) = exprs.first() else {
        return Ok(exprs);
    };
    let (text, fmt_span) = (fmt.value.as_ref().to_string(), fmt.span);
    let (pieces, used) = split_at_placeholders(&text, fmt_span, &exprs, macro_name)?;
    let supplied = exprs.len() - 1;
    // A leading string with no placeholders is a label, not a format string: Vx prints a macro's
    // arguments in sequence, and `println!("count: ", v)` is the established spelling for that.
    // Only a string that uses `{}` is held to matching its arguments.
    if used == 0 {
        // No placeholders: a label, printed with the arguments after it. Rebuild only when the
        // escape pass actually changed the text, so `{{` still reaches the output as `{`.
        let unescaped = match pieces.first() {
            Some(expr::Expr::StringLiteral(p)) => p.value.as_ref() != text,
            _ => false,
        };
        if unescaped {
            let mut out = pieces;
            out.extend(exprs.iter().skip(1).cloned());
            return Ok(out);
        }
        return Ok(exprs);
    }
    if used != supplied {
        return Err(format!(
            "{macro_name}: the format string has {used} `{{}}` but {supplied} argument(s) were given"
        ));
    }
    Ok(pieces)
}

impl<'a> MacroExpander<'a> {
    /// An expander that knows the given macros and trait default methods.
    #[must_use]
    pub fn new(
        macros: &'a HashMap<crate::symbol::Symbol, Vec<MacroRule>>,
        trait_defaults: &'a crate::resolver::TraitDefaults,
    ) -> Self {
        Self {
            macros,
            trait_defaults,
        }
    }

    /// Parse the tokens a macro produced as one expression.
    ///
    /// # Errors
    ///
    /// Fails when the tokens are not an expression.
    fn parse_expanded_expr(tokens: &[crate::lexer::OwnedToken]) -> Result<expr::Expr, String> {
        let lexed_tokens: Vec<_> = tokens.iter().map(|t| t.as_token()).collect();
        let mut parser = crate::parser::Parser::new(&lexed_tokens, "");
        parser.parse_expr().map_err(|e| format!("{e:?}"))
    }

    /// Parse the tokens a macro call was given as a list of expressions with commas between.
    ///
    /// # Errors
    ///
    /// Fails when one of the list's entries is not an expression.
    fn parse_expanded_exprs(
        tokens: &[crate::lexer::OwnedToken],
    ) -> Result<Vec<expr::Expr>, String> {
        let lexed_tokens: Vec<_> = tokens.iter().map(|t| t.as_token()).collect();
        let mut parser = crate::parser::Parser::new(&lexed_tokens, "");
        let mut exprs = Vec::new();
        while !parser.check(&crate::lexer::TokenType::Eof) {
            exprs.push(parser.parse_expr().map_err(|e| format!("{e:?}"))?);
            if !parser.match_token(&crate::lexer::TokenType::Comma) {
                break;
            }
        }
        Ok(exprs)
    }

    /// Refuse a rule whose matcher names a fragment kind that does not exist.
    ///
    /// Checked before any rule is tried, rather than while matching, because a rule that
    /// fails to match is not an error -- the next rule is tried, and only when all of them
    /// fail is anything reported. An unknown kind raised there came out as "no matching rule
    /// found", which points at the call rather than at the typo in the rule.
    ///
    /// # Errors
    ///
    /// Fails when a fragment in the matcher has a kind other than `expr`, `ty` or `ident`.
    fn validate_matcher(name: &str, matcher: &[TokenTree]) -> Result<(), String> {
        let mut tokens = Vec::new();
        for tt in matcher {
            tokens.extend(Self::flatten_tt(tt));
        }
        for (i, tok) in tokens.iter().enumerate() {
            if tok.kind != OwnedTokenType::Dollar || i + 3 >= tokens.len() {
                continue;
            }
            if tokens[i + 2].kind != OwnedTokenType::Colon {
                continue;
            }
            if let OwnedTokenType::Identifier(kind) = &tokens[i + 3].kind {
                if !matches!(kind.as_ref(), "expr" | "ty" | "ident") {
                    return Err(format!(
                        "unknown macro fragment kind '{kind}' in macro {name}: the kinds that exist \
                         are `expr`, `ty` and `ident`"
                    ));
                }
            }
        }
        Ok(())
    }

    /// Transcribe one item-position macro call and parse what came out as items.
    ///
    /// The transcribed tokens are handed to a fresh parser's whole-program entry point, so a
    /// macro may produce anything that can be written at the top level -- an `impl`, a `fn`,
    /// a `struct` -- and each is parsed by the same code that parses one written by hand.
    ///
    /// # Errors
    ///
    /// Fails when the macro is not defined, when one of its rules names a fragment kind that
    /// does not exist, when no rule matches the call, or when the result does not parse as
    /// items.
    fn expand_item_macro(
        &self,
        call: &stmt::MacroCallStmt,
    ) -> Result<crate::syntax::Program, String> {
        let rules = self
            .macros
            .get(call.name.as_ref())
            .ok_or_else(|| format!("Macro {} not found", call.name))?;
        for rule in rules {
            Self::validate_matcher(call.name.as_ref(), &rule.matcher)?;
        }
        let input_tokens = match &call.token_tree {
            TokenTree::Delimited(_, inner) => {
                let mut tokens = Vec::new();
                for i in inner {
                    tokens.extend(Self::flatten_tt(i));
                }
                tokens
            }
            other => Self::flatten_tt(other),
        };
        for rule in rules {
            if let Ok(captures) = Self::match_rule(&rule.matcher, &input_tokens) {
                let mut transcribed = Self::transcribe(&rule.transcriber, &captures);
                transcribed.push(crate::lexer::OwnedToken {
                    kind: OwnedTokenType::Eof,
                    line: 0,
                    column: 0,
                    length: 0,
                });
                let lexed: Vec<_> = transcribed.iter().map(|t| t.as_token()).collect();
                let mut parser = crate::parser::Parser::new(&lexed, "");
                return parser
                    .parse()
                    .map_err(|e| format!("macro {} did not expand to items: {:?}", call.name, e));
            }
        }
        Err(format!("No matching rule found for macro {}", call.name))
    }

    /// Expand every macro call in `module`, then give each impl the default trait methods it
    /// did not write.
    ///
    /// # Errors
    ///
    /// Fails when a macro call cannot be expanded, or when macro calls in item position still
    /// produce more macro calls after 64 rounds.
    pub fn expand_module(&self, module: &mut VxModule) -> Result<(), String> {
        // Item-position macro calls first, so the items they produce are expanded below like
        // any other. A produced item may call another item macro, so this drains a worklist
        // rather than walking the list once; the depth limit is what a macro that expands to
        // itself runs into, instead of the compiler running out of stack.
        let mut pending = std::mem::take(&mut module.item_macros);
        let mut rounds = 0;
        while !pending.is_empty() {
            rounds += 1;
            if rounds > 64 {
                return Err(format!(
                    "macro expansion in item position did not settle after {} rounds; a macro \
                     that expands to a call of itself is the usual cause",
                    rounds - 1
                ));
            }
            let mut next = Vec::new();
            for call in pending {
                let produced = self.expand_item_macro(&call)?;
                module.structs.extend(produced.structs);
                module.enums.extend(produced.enums);
                module.traits.extend(produced.traits);
                module.impls.extend(produced.impls);
                module.functions.extend(produced.functions);
                module.consts.extend(produced.consts);
                module.externs.extend(produced.externs);
                module.macros.extend(produced.macros);
                module.transfer_impls.extend(produced.transfer_impls);
                next.extend(produced.item_macros);
            }
            pending = next;
        }

        // Every impl the module now has, including the ones the macros above produced, gets
        // the default method bodies it did not write. This is why it happens here and not
        // where the modules are first loaded: before expansion a macro-produced impl does
        // not exist yet, and filling defaults then left it without them.
        crate::resolver::fill_trait_defaults_in(module, self.trait_defaults);

        // After the copy, so a default's signature gets the same treatment a hand-written
        // method's does. Both say `Self::Item`; from here on both say what the impl bound it to.
        crate::resolver::bind_associated_types_in(module, self.trait_defaults);

        // Expand top level decls
        for func in &mut module.functions {
            self.expand_function(func)?;
        }
        for impl_block in &mut module.impls {
            for func in &mut impl_block.methods {
                self.expand_function(func)?;
            }
        }
        // Transfer lowerings carry ordinary function bodies, so a macro call inside one must
        // expand like a macro call anywhere else -- skipping them would leave a MacroCall node
        // behind for every downstream pass to trip on (Vx#352 review finding).
        for t in &mut module.transfer_impls {
            for func in &mut t.methods {
                self.expand_function(func)?;
            }
        }
        Ok(())
    }

    /// Expand the macro calls in a function's body.
    ///
    /// # Errors
    ///
    /// Fails when a macro call in the body cannot be expanded.
    fn expand_function(&self, func: &mut Function) -> Result<(), String> {
        self.expand_block(&mut func.body)
    }

    /// Expand the macro calls in a list of statements, in place.
    ///
    /// # Errors
    ///
    /// Fails when a macro call in one of the statements cannot be expanded.
    fn expand_block(&self, block: &mut Vec<stmt::Statement>) -> Result<(), String> {
        let body = std::mem::take(block);
        let mut new_body = Vec::with_capacity(body.len());
        for s in body {
            new_body.extend(self.expand_stmt(s)?);
        }
        *block = new_body;
        Ok(())
    }

    /// Expand one statement. A statement that is a macro call becomes the expression the macro
    /// stands for.
    ///
    /// # Errors
    ///
    /// Fails when a macro call in the statement cannot be expanded.
    fn expand_stmt(&self, mut stmt: stmt::Statement) -> Result<Vec<stmt::Statement>, String> {
        if let stmt::Statement::MacroCall(call) = stmt {
            let expanded_expr =
                self.expand_macro_call(&call.name, &call.token_tree, call.block_tree.as_ref())?;
            let recursively_expanded = self.expand_expr(expanded_expr)?;
            return Ok(vec![stmt::Statement::ExprStmt(stmt::ExprStmtStmt {
                expr: recursively_expanded,
                has_semi: call.has_semi,
                span: call.span,
            })]);
        }
        // Recurse into children
        self.expand_stmt_children(&mut stmt)?;
        Ok(vec![stmt])
    }

    /// Expand the macro calls in the expressions and statements a statement holds.
    ///
    /// # Errors
    ///
    /// Fails when one of those macro calls cannot be expanded.
    fn expand_stmt_children(&self, stmt: &mut stmt::Statement) -> Result<(), String> {
        match stmt {
            stmt::Statement::ExprStmt(e) => {
                e.expr = self.expand_expr(take_expr(&mut e.expr))?;
            }
            stmt::Statement::LetDecl(l) => {
                l.expr = self.expand_expr(take_expr(&mut l.expr))?;
            }
            stmt::Statement::Assign(a) => {
                a.lhs = self.expand_expr(take_expr(&mut a.lhs))?;
                a.rhs = self.expand_expr(take_expr(&mut a.rhs))?;
            }
            stmt::Statement::CompoundAssign(a) => {
                a.lhs = self.expand_expr(take_expr(&mut a.lhs))?;
                a.rhs = self.expand_expr(take_expr(&mut a.rhs))?;
            }
            stmt::Statement::Return(r) => {
                if let Some(e) = r.expr.as_mut() {
                    r.expr = Some(self.expand_expr(take_expr(e))?);
                }
            }
            stmt::Statement::Assert(a) => {
                *a.expr = self.expand_expr(take_expr(&mut a.expr))?;
            }
            stmt::Statement::Loop(l) => {
                self.expand_block(&mut l.body)?;
            }
            stmt::Statement::ForLoop(f) => {
                *f.iterable = self.expand_expr(take_expr(&mut f.iterable))?;
                self.expand_block(&mut f.body)?;
            }
            _ => {}
        }
        Ok(())
    }

    /// Expand an expression: a macro call becomes what the macro stands for, and every other
    /// expression has the macro calls inside it expanded.
    ///
    /// # Errors
    ///
    /// Fails when a macro call in the expression cannot be expanded.
    fn expand_expr(&self, mut expr: expr::Expr) -> Result<expr::Expr, String> {
        if let expr::Expr::MacroCall(call) = expr {
            let expanded =
                self.expand_macro_call(&call.name, &call.token_tree, call.block_tree.as_ref())?;
            return self.expand_expr(expanded);
        }
        // Traverse and expand
        self.expand_operands(&mut expr)?;
        Ok(expr)
    }

    /// Expand the macro calls in the operands of an operator, a range, a field access or an
    /// index. Any other kind of expression is passed on to `expand_call_arguments`.
    ///
    /// # Errors
    ///
    /// Fails when a macro call in the expression cannot be expanded.
    fn expand_operands(&self, expr: &mut expr::Expr) -> Result<(), String> {
        match expr {
            expr::Expr::BinaryOp(b) => {
                *b.lhs = self.expand_expr(take_expr(&mut b.lhs))?;
                *b.rhs = self.expand_expr(take_expr(&mut b.rhs))?;
            }
            expr::Expr::RelationalOp(b) => {
                *b.lhs = self.expand_expr(take_expr(&mut b.lhs))?;
                *b.rhs = self.expand_expr(take_expr(&mut b.rhs))?;
            }
            expr::Expr::LogicalOp(b) => {
                *b.lhs = self.expand_expr(take_expr(&mut b.lhs))?;
                *b.rhs = self.expand_expr(take_expr(&mut b.rhs))?;
            }
            expr::Expr::Range(b) => {
                *b.start = self.expand_expr(take_expr(&mut b.start))?;
                *b.end = self.expand_expr(take_expr(&mut b.end))?;
            }
            expr::Expr::UnaryOp(u) => {
                *u.expr = self.expand_expr(take_expr(&mut u.expr))?;
            }
            expr::Expr::Borrow(u) => {
                *u.expr = self.expand_expr(take_expr(&mut u.expr))?;
            }
            expr::Expr::Dereference(u) => {
                *u.expr = self.expand_expr(take_expr(&mut u.expr))?;
            }
            expr::Expr::MemberAccess(m) => {
                *m.base = self.expand_expr(take_expr(&mut m.base))?;
            }
            expr::Expr::IndexAccess(m) => {
                *m.base = self.expand_expr(take_expr(&mut m.base))?;
                *m.index = self.expand_expr(take_expr(&mut m.index))?;
            }
            _ => self.expand_call_arguments(expr)?,
        }
        Ok(())
    }

    /// Expand the macro calls in the bodies of a `match`, an `if`, a closure, and an `unsafe`,
    /// `comptime` or `spawn` block. Any other kind of expression has nothing left to expand.
    ///
    /// # Errors
    ///
    /// Fails when a macro call in the expression cannot be expanded.
    fn expand_nested_blocks(&self, expr: &mut expr::Expr) -> Result<(), String> {
        match expr {
            expr::Expr::Match(m) => {
                *m.expr = self.expand_expr(take_expr(&mut m.expr))?;
                for arm in &mut m.arms {
                    self.expand_block(&mut arm.body)?;
                }
            }
            expr::Expr::If(i) => {
                *i.cond = self.expand_expr(take_expr(&mut i.cond))?;
                self.expand_block(&mut i.then_block)?;
                if let Some(else_b) = &mut i.else_block {
                    self.expand_block(else_b)?;
                }
            }
            expr::Expr::UnsafeBlock(u) => {
                if let Some(ret) = &mut u.ret {
                    **ret = self.expand_expr(take_expr(ret))?;
                }
                self.expand_block(&mut u.stmts)?;
            }
            expr::Expr::ComptimeBlock(u) => {
                if let Some(ret) = &mut u.ret {
                    **ret = self.expand_expr(take_expr(ret))?;
                }
                self.expand_block(&mut u.stmts)?;
            }
            expr::Expr::Closure(c) => {
                *c.body = self.expand_expr(take_expr(&mut c.body))?;
            }
            expr::Expr::SpawnOn(s) => {
                if let Some(ret) = &mut s.ret {
                    **ret = self.expand_expr(take_expr(ret))?;
                }
                self.expand_block(&mut s.stmts)?;
            }
            _ => {}
        }
        Ok(())
    }

    /// Expand the macro calls in the arguments of a call, the fields of a struct literal and
    /// the elements of an array or `vec!`. Any other kind of expression is passed on to
    /// `expand_nested_blocks`.
    ///
    /// # Errors
    ///
    /// Fails when a macro call in the expression cannot be expanded.
    fn expand_call_arguments(&self, expr: &mut expr::Expr) -> Result<(), String> {
        match expr {
            expr::Expr::FunctionCall(f) => {
                for arg in &mut f.args {
                    *arg = self.expand_expr(take_expr(arg))?;
                }
            }
            expr::Expr::MethodCall(m) => {
                *m.base = self.expand_expr(take_expr(&mut m.base))?;
                for arg in &mut m.args {
                    *arg = self.expand_expr(take_expr(arg))?;
                }
            }
            expr::Expr::StructInit(s) => {
                for (_, e) in &mut s.fields {
                    *e = self.expand_expr(take_expr(e))?;
                }
            }
            expr::Expr::Array(a) => {
                for e in &mut a.elements {
                    *e = self.expand_expr(take_expr(e))?;
                }
            }
            expr::Expr::VecMacro(a) => {
                for e in &mut a.elements {
                    *e = self.expand_expr(take_expr(e))?;
                }
            }
            expr::Expr::Grad(g) => {
                for arg in &mut g.args {
                    *arg = self.expand_expr(take_expr(arg))?;
                }
            }
            expr::Expr::Vjp(v) => {
                for arg in &mut v.args {
                    *arg = self.expand_expr(take_expr(arg))?;
                }
                *v.cotangent = self.expand_expr(take_expr(&mut v.cotangent))?;
            }
            expr::Expr::Jvp(j_expr) => {
                for arg in &mut j_expr.args {
                    *arg = self.expand_expr(take_expr(arg))?;
                }
                *j_expr.tangent = self.expand_expr(take_expr(&mut j_expr.tangent))?;
            }
            _ => self.expand_nested_blocks(expr)?,
        }
        Ok(())
    }

    /// The expression one macro call stands for. `mlir!`, `vec!`, `print!` and `println!` are
    /// built in; any other name is looked up in the `macro_rules!` macros.
    ///
    /// # Errors
    ///
    /// Fails when the macro is not defined, when one of its rules names a fragment kind that
    /// does not exist, when no rule matches the call, when the result does not parse as an
    /// expression, or when a built-in macro is given arguments it cannot read.
    fn expand_macro_call(
        &self,
        name: &str,
        tt: &TokenTree,
        block_tree: Option<&TokenTree>,
    ) -> Result<expr::Expr, String> {
        if name == "mlir" {
            return Self::expand_mlir_macro(tt, block_tree);
        }

        if name == "vec" {
            return Self::expand_vec_macro(tt);
        }
        if name == "print" {
            return Self::expand_print_macro(tt);
        }
        if name == "println" {
            return Self::expand_println_macro(tt);
        }

        let rules = self
            .macros
            .get(name)
            .ok_or_else(|| format!("Macro {name} not found"))?;
        for rule in rules {
            Self::validate_matcher(name, &rule.matcher)?;
        }

        let input_tokens = match tt {
            TokenTree::Delimited(_, inner) => {
                let mut tokens = Vec::new();
                for i in inner {
                    tokens.extend(Self::flatten_tt(i));
                }
                tokens
            }
            _ => Self::flatten_tt(tt),
        };

        for rule in rules {
            if let Ok(captures) = Self::match_rule(&rule.matcher, &input_tokens) {
                let mut transcribed = Self::transcribe(&rule.transcriber, &captures);
                // IMPORTANT: The parser expects an EOF token at the end!
                transcribed.push(crate::lexer::OwnedToken {
                    kind: OwnedTokenType::Eof,
                    line: 0,
                    column: 0,
                    length: 0,
                });
                return Self::parse_expanded_expr(&transcribed);
            }
        }

        Err(format!("No matching rule found for macro {name}"))
    }

    /// The tokens of a token tree in source order, with the brackets of a delimited tree
    /// written out as tokens.
    fn flatten_tt(tt: &TokenTree) -> Vec<crate::lexer::OwnedToken> {
        let mut tokens = Vec::new();
        match tt {
            TokenTree::Token(t) => tokens.push(t.clone()),
            TokenTree::Group(inner) => {
                for i in inner {
                    tokens.extend(Self::flatten_tt(i));
                }
            }
            TokenTree::Delimited(delim, inner) => {
                let (open, close) = match delim {
                    Delimiter::Parenthesis => {
                        (OwnedTokenType::LeftParen, OwnedTokenType::RightParen)
                    }
                    Delimiter::Brace => (OwnedTokenType::LeftBrace, OwnedTokenType::RightBrace),
                    Delimiter::Bracket => {
                        (OwnedTokenType::LeftBracket, OwnedTokenType::RightBracket)
                    }
                };
                tokens.push(crate::lexer::OwnedToken {
                    kind: open,
                    line: 0,
                    column: 0,
                    length: 0,
                });
                for i in inner {
                    tokens.extend(Self::flatten_tt(i));
                }
                tokens.push(crate::lexer::OwnedToken {
                    kind: close,
                    line: 0,
                    column: 0,
                    length: 0,
                });
            }
        }
        tokens
    }

    /// The name and kind of the fragment `$name:kind` whose `$` is the matcher token at `at`.
    /// `None` when the tokens there are not a fragment.
    fn fragment_at(
        matcher_tokens: &[crate::lexer::OwnedToken],
        at: usize,
    ) -> Option<(&crate::symbol::Symbol, &str)> {
        if matcher_tokens[at].kind != OwnedTokenType::Dollar || at + 3 >= matcher_tokens.len() {
            return None;
        }
        let OwnedTokenType::Identifier(name) = &matcher_tokens[at + 1].kind else {
            return None;
        };
        if matcher_tokens[at + 2].kind != OwnedTokenType::Colon {
            return None;
        }
        let OwnedTokenType::Identifier(kind) = &matcher_tokens[at + 3].kind else {
            return None;
        };
        Some((name, kind.as_ref()))
    }

    /// Match the tokens of a macro call against one rule's matcher. Returns the tokens each
    /// fragment of the matcher took, by fragment name.
    ///
    /// # Errors
    ///
    /// Fails when the call does not fit the matcher: a token differs, the call ends too early,
    /// or the call has tokens left over.
    fn match_rule(
        matcher: &[TokenTree],
        input: &[crate::lexer::OwnedToken],
    ) -> Result<HashMap<crate::symbol::Symbol, Vec<crate::lexer::OwnedToken>>, String> {
        let mut captures = HashMap::new();
        let mut matcher_tokens = Vec::new();
        for tt in matcher {
            matcher_tokens.extend(Self::flatten_tt(tt));
        }

        let mut i = 0; // input index
        let mut j = 0; // matcher index

        while j < matcher_tokens.len() {
            if let Some((name, kind)) = Self::fragment_at(&matcher_tokens, j) {
                // Match a meta-variable. `ident` takes exactly one token; `expr`
                // and `ty` take everything up to the next token the matcher names,
                // which is what lets `$t : ty` capture `Tensor<f32, [4]>` as well
                // as `i32`. They are not told apart beyond that: nothing here
                // parses the capture to check it really is a type or an
                // expression, and the error surfaces where the capture is used.
                if kind == "ident" {
                    if i >= input.len() {
                        return Err("Input ended unexpectedly".to_string());
                    }
                    captures.insert(name.clone(), vec![input[i].clone()]);
                    i += 1;
                    j += 4;
                    continue;
                }
                if kind == "expr" || kind == "ty" {
                    let mut taken = Vec::new();
                    if j + 4 < matcher_tokens.len() {
                        let next_m_tok = &matcher_tokens[j + 4];
                        while i < input.len() && input[i].kind != next_m_tok.kind {
                            taken.push(input[i].clone());
                            i += 1;
                        }
                    } else {
                        while i < input.len() {
                            taken.push(input[i].clone());
                            i += 1;
                        }
                    }
                    captures.insert(name.clone(), taken);
                    j += 4;
                    continue;
                }
            }

            // Literal match
            if i >= input.len() {
                return Err("Input ended unexpectedly".to_string());
            }
            if matcher_tokens[j].kind != input[i].kind {
                return Err(format!(
                    "Token mismatch: expected {:?}, got {:?}",
                    matcher_tokens[j].kind, input[i].kind
                ));
            }
            i += 1;
            j += 1;
        }

        if i < input.len() {
            return Err("Trailing input tokens".to_string());
        }

        Ok(captures)
    }

    /// Write out a rule's transcriber, with each `$name` replaced by the tokens that fragment
    /// took from the call. The result ends with an end-of-file token.
    fn transcribe(
        transcriber: &[TokenTree],
        captures: &HashMap<crate::symbol::Symbol, Vec<crate::lexer::OwnedToken>>,
    ) -> Vec<crate::lexer::OwnedToken> {
        let mut tokens = Vec::new();
        let mut transcriber_tokens = Vec::new();
        for tt in transcriber {
            transcriber_tokens.extend(Self::flatten_tt(tt));
        }

        let mut j = 0;
        while j < transcriber_tokens.len() {
            let m_tok = &transcriber_tokens[j];
            if m_tok.kind == OwnedTokenType::Dollar && j + 1 < transcriber_tokens.len() {
                let name_tok = &transcriber_tokens[j + 1];
                if let OwnedTokenType::Identifier(name) = &name_tok.kind {
                    if let Some(captured) = captures.get(name) {
                        // Move the capture to where `$name` stood, keeping the tokens'
                        // offsets from each other. An `mlir!` block is rebuilt from token
                        // positions, so a capture still carrying the call site's line
                        // lands away from the rest of the block; and adjacency within the
                        // capture matters, since a nested `add_one!(x)` is a call only
                        // when the `!` touches the name.
                        let anchor = captured.first().map(|t| (t.line, t.column));
                        for tok in captured {
                            let mut placed = tok.clone();
                            if let Some((base_line, base_col)) = anchor {
                                // A synthesized token carries no position and is left alone.
                                if placed.line != 0 {
                                    let line_off = placed.line.saturating_sub(base_line);
                                    placed.line = m_tok.line + line_off;
                                    if line_off == 0 {
                                        placed.column =
                                            m_tok.column + placed.column.saturating_sub(base_col);
                                    }
                                }
                            }
                            tokens.push(placed);
                        }
                        j += 2;
                        continue;
                    }
                }
            }
            tokens.push(m_tok.clone());
            j += 1;
        }

        // Append EOF to ensure parser completes
        tokens.push(crate::lexer::OwnedToken {
            kind: OwnedTokenType::Eof,
            line: 0,
            column: 0,
            length: 0,
        });

        tokens
    }

    /// Turn a `vec!` call into the list of its elements.
    ///
    /// # Errors
    ///
    /// Fails when the arguments are not in brackets, or when one of them is not an
    /// expression.
    fn expand_vec_macro(tt: &TokenTree) -> Result<expr::Expr, String> {
        let TokenTree::Delimited(_, elements) = tt else {
            return Err("Expected delimited token tree for vec!".to_string());
        };
        let mut tokens = Vec::new();
        for t in elements {
            tokens.extend(Self::flatten_tt(t));
        }
        tokens.push(crate::lexer::OwnedToken {
            kind: OwnedTokenType::Eof,
            line: 0,
            column: 0,
            length: 0,
        });
        let exprs = Self::parse_expanded_exprs(&tokens)?;
        Ok(expr::Expr::VecMacro(expr::VecMacroExpr {
            elements: exprs,
            span: Span::default(),
        }))
    }

    /// Turn a `print!` call into an expression that prints its arguments.
    ///
    /// # Errors
    ///
    /// Fails when the arguments are not in brackets, when one of them is not an
    /// expression, or when the format string and the arguments do not agree.
    fn expand_print_macro(tt: &TokenTree) -> Result<expr::Expr, String> {
        let TokenTree::Delimited(_, elements) = tt else {
            return Err("Expected delimited token tree for print!".to_string());
        };
        let mut tokens = Vec::new();
        for t in elements {
            tokens.extend(Self::flatten_tt(t));
        }
        tokens.push(crate::lexer::OwnedToken {
            kind: OwnedTokenType::Eof,
            line: 0,
            column: 0,
            length: 0,
        });
        let exprs = Self::parse_expanded_exprs(&tokens)?;
        Ok(expr::Expr::Print(expr::PrintExpr {
            args: interleave_format_args(exprs, "print!")?,
            span: Span::default(),
        }))
    }

    /// Turn a `println!` call into an expression that prints its arguments and ends the line.
    ///
    /// # Errors
    ///
    /// Fails when the arguments are not in brackets, when one of them is not an
    /// expression, or when the format string and the arguments do not agree.
    fn expand_println_macro(tt: &TokenTree) -> Result<expr::Expr, String> {
        let TokenTree::Delimited(_, elements) = tt else {
            return Err("Expected delimited token tree for println!".to_string());
        };
        let mut tokens = Vec::new();
        for t in elements {
            tokens.extend(Self::flatten_tt(t));
        }
        tokens.push(crate::lexer::OwnedToken {
            kind: OwnedTokenType::Eof,
            line: 0,
            column: 0,
            length: 0,
        });
        let exprs = Self::parse_expanded_exprs(&tokens)?;
        Ok(expr::Expr::Println(expr::PrintlnExpr {
            args: interleave_format_args(exprs, "println!")?,
            span: Span::default(),
        }))
    }

    /// The tokens inside the brackets of an `mlir!` call, with an end-of-file token after them.
    ///
    /// # Errors
    ///
    /// Fails when the arguments of the call are not in brackets.
    fn mlir_header_tokens(tt: &TokenTree) -> Result<Vec<crate::lexer::OwnedToken>, String> {
        match tt {
            TokenTree::Delimited(_, inner) => {
                let mut t = Vec::new();
                for i in inner {
                    t.extend(Self::flatten_tt(i));
                }
                t.push(crate::lexer::OwnedToken {
                    kind: crate::lexer::OwnedTokenType::Eof,
                    line: 0,
                    column: 0,
                    length: 0,
                });
                Ok(t)
            }
            _ => Err("Expected delimited token tree for mlir!".to_string()),
        }
    }

    /// Parse the `inputs` field of an `mlir!` call, `(%a = x: type, ...)`, and add each entry
    /// to `inputs`.
    ///
    /// # Errors
    ///
    /// Fails when the list is not in `(` and `)`, or when an entry is not a name, `=`, an
    /// expression, `:` and a type.
    fn parse_mlir_inputs(
        parser: &mut crate::parser::Parser<'_>,
        inputs: &mut Vec<(crate::symbol::Symbol, expr::Expr, String)>,
    ) -> Result<(), String> {
        parser
            .consume(&crate::lexer::TokenType::LeftParen, "Expected '('")
            .map_err(|e| e.format(""))?;
        if !parser.check(&crate::lexer::TokenType::RightParen) {
            loop {
                // An MLIR value name: `%arg`. Until `%` was the remainder
                // operator nothing else in the language spelled it, so it arrived
                // here as `Unknown('%')`; it is an ordinary token now.
                let is_percent = match parser.peek().kind {
                    crate::lexer::TokenType::Percent => {
                        parser.advance();
                        true
                    }
                    _ => false,
                };
                let arg_name = match &parser.advance().kind {
                    crate::lexer::TokenType::Identifier(ident) => {
                        if is_percent {
                            format!("%{ident}")
                        } else {
                            ident.to_string()
                        }
                    }
                    _ => return Err("Expected identifier in inputs".to_string()),
                };
                parser
                    .consume(&crate::lexer::TokenType::Equals, "Expected '='")
                    .map_err(|e| e.format(""))?;
                let expr = parser.parse_expr().map_err(|e| e.format(""))?;
                parser
                    .consume(&crate::lexer::TokenType::Colon, "Expected ':'")
                    .map_err(|e| e.format(""))?;

                let ty_str = Self::parse_mlir_type_text(parser);
                inputs.push((arg_name.into(), expr, ty_str));

                if !parser.match_token(&crate::lexer::TokenType::Comma) {
                    break;
                }
            }
        }
        parser
            .consume(&crate::lexer::TokenType::RightParen, "Expected ')'")
            .map_err(|e| e.format(""))?;
        Ok(())
    }

    /// The MLIR type of one `inputs` entry as text: every token up to the `,` or `)` that ends
    /// the entry, with a space wherever the source had one.
    fn parse_mlir_type_text(parser: &mut crate::parser::Parser<'_>) -> String {
        let mut ty_str = String::new();
        let mut angle_depth = 0;
        // Keep the gaps the source had. Gluing the tokens together turned
        // `memref<N x M x f32>` into one word, and a type parameter that is
        // not a word of its own cannot be substituted (Vx#414).
        let mut prev_end = 0;
        while !parser.check(&crate::lexer::TokenType::Eof) {
            if angle_depth == 0
                && (parser.check(&crate::lexer::TokenType::Comma)
                    || parser.check(&crate::lexer::TokenType::RightParen))
            {
                break;
            }
            let tok = parser.advance();
            if tok.kind == crate::lexer::TokenType::LeftAngle {
                angle_depth += 1;
            } else if tok.kind == crate::lexer::TokenType::RightAngle {
                angle_depth -= 1;
            }
            if prev_end > 0 && tok.column > prev_end {
                ty_str.push(' ');
            }
            prev_end = tok.column + tok.length;
            ty_str.push_str(&tok.kind.to_string());
        }
        ty_str
    }

    /// Parse the `clobbers` field of an `mlir!` call, `[a, b]`, and add each expression to
    /// `clobbers`.
    ///
    /// # Errors
    ///
    /// Fails when the list is not in `[` and `]`, or when an entry is not an expression.
    fn parse_mlir_clobbers(
        parser: &mut crate::parser::Parser<'_>,
        clobbers: &mut Vec<expr::Expr>,
    ) -> Result<(), String> {
        parser
            .consume(&crate::lexer::TokenType::LeftBracket, "Expected '['")
            .map_err(|e| e.format(""))?;
        if !parser.check(&crate::lexer::TokenType::RightBracket) {
            loop {
                clobbers.push(parser.parse_expr().map_err(|e| e.format(""))?);
                if !parser.match_token(&crate::lexer::TokenType::Comma) {
                    break;
                }
            }
        }
        parser
            .consume(&crate::lexer::TokenType::RightBracket, "Expected ']'")
            .map_err(|e| e.format(""))?;
        Ok(())
    }

    /// Parse the `dialects` field of an `mlir!` call, `["a", "b"]`, and add each name to
    /// `dialects`.
    ///
    /// # Errors
    ///
    /// Fails when the list is not in `[` and `]`, or when an entry is not a string.
    fn parse_mlir_dialects(
        parser: &mut crate::parser::Parser<'_>,
        dialects: &mut Vec<String>,
    ) -> Result<(), String> {
        parser
            .consume(&crate::lexer::TokenType::LeftBracket, "Expected '['")
            .map_err(|e| e.format(""))?;
        if !parser.check(&crate::lexer::TokenType::RightBracket) {
            loop {
                match &parser.advance().kind {
                    crate::lexer::TokenType::StringLiteral(lit) => {
                        dialects.push(lit.to_string());
                    }
                    _ => return Err("Expected string literal in dialects".to_string()),
                }
                if !parser.match_token(&crate::lexer::TokenType::Comma) {
                    break;
                }
            }
        }
        parser
            .consume(&crate::lexer::TokenType::RightBracket, "Expected ']'")
            .map_err(|e| e.format(""))?;
        Ok(())
    }

    /// The text of the block after an `mlir!` call, laid out by the line and column of each
    /// token so it reads as it did in the source.
    ///
    /// # Errors
    ///
    /// Fails when the call has no block after it.
    fn mlir_block_text(block_tree: Option<&TokenTree>) -> Result<String, String> {
        let Some(TokenTree::Delimited(_, inner)) = block_tree else {
            return Err("mlir! macro requires a trailing block".to_string());
        };
        let mut t = Vec::new();
        for i in inner {
            t.extend(Self::flatten_tt(i));
        }

        let mut s = String::new();
        let mut current_line = 0;
        let mut current_col = 0;

        for tok in t {
            let text = tok.kind.to_string();
            // A delimiter synthesized when the token tree was built carries no position.
            // Laying one out by column drove the rest of its line hundreds of columns to
            // the right, and the block stopped parsing (Vx#414).
            if tok.line == 0 {
                s.push_str(&text);
                current_col += text.chars().count();
                continue;
            }

            if current_line == 0 {
                current_line = tok.line;
                current_col = tok.column;
            }

            if tok.line > current_line {
                for _ in 0..(tok.line - current_line) {
                    s.push('\n');
                }
                current_col = 1;
                current_line = tok.line;
            }

            if tok.column > current_col {
                for _ in 0..(tok.column - current_col) {
                    s.push(' ');
                }
            }

            s.push_str(&text);
            current_col = tok.column + tok.length;
        }
        Ok(s)
    }

    /// Turn an `mlir!` call into an inline MLIR expression: the fields in the brackets, then
    /// the block of MLIR text.
    ///
    /// # Errors
    ///
    /// Fails when the arguments are not in brackets, when a field is not `inputs`, `clobbers`,
    /// `returns` or `dialects`, when a field's value does not parse, or when the call has no
    /// block after it.
    fn expand_mlir_macro(
        tt: &TokenTree,
        block_tree: Option<&TokenTree>,
    ) -> Result<expr::Expr, String> {
        let mut inputs = Vec::new();
        let mut clobbers = Vec::new();
        let mut returns = None;
        let mut dialects = Vec::new();

        let tokens = Self::mlir_header_tokens(tt)?;

        let lexed_tokens: Vec<_> = tokens.iter().map(|t| t.as_token()).collect();
        let mut parser = crate::parser::Parser::new(&lexed_tokens, "");

        while !parser.check(&crate::lexer::TokenType::Eof) {
            let field_name = match &parser.advance().kind {
                crate::lexer::TokenType::Identifier(ident) => ident.to_string(),
                _ => {
                    return Err(
                        "Expected 'inputs', 'clobbers', 'returns', or 'dialects'".to_string()
                    )
                }
            };
            parser
                .consume(&crate::lexer::TokenType::Colon, "Expected ':'")
                .map_err(|e| e.format(""))?;

            match field_name.as_ref() {
                "inputs" => Self::parse_mlir_inputs(&mut parser, &mut inputs)?,
                "clobbers" => Self::parse_mlir_clobbers(&mut parser, &mut clobbers)?,
                "returns" => {
                    if parser.match_token(&crate::lexer::TokenType::Identifier("void")) {
                        returns = None;
                    } else {
                        returns = Some(parser.parse_type().map_err(|e| e.format(""))?);
                    }
                }
                "dialects" => Self::parse_mlir_dialects(&mut parser, &mut dialects)?,
                _ => return Err(format!("Unknown field '{field_name}' in mlir! macro")),
            }

            parser.match_token(&crate::lexer::TokenType::Comma);
        }

        let block_str = Self::mlir_block_text(block_tree)?;

        // The macro's own position, so a block that fails to parse can be pointed at. A
        // default span reported every one of them at 0:0 (Vx#414).
        let span = tokens
            .first()
            .map(|t| Span {
                line: t.line,
                column: t.column,
                length: t.length,
            })
            .unwrap_or_default();

        Ok(expr::Expr::InlineMlir(expr::InlineMlirExpr {
            inputs,
            clobbers,
            returns,
            dialects,
            block_str,
            span,
        }))
    }
}
