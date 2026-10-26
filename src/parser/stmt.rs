//===- stmt.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Parser for Vx statements, including let bindings, loops, and return statements.
//
//===----------------------------------------------------------------------===//

use super::{
    AssertStmt, AssignStmt, BinaryOp, BreakStmt, CompoundAssignStmt, ComptimeBlockExpr,
    ContinueStmt, Expr, ExprStmtStmt, ForLoopStmt, IdentifierExpr, IfExpr, LetDeclStmt, LoopStmt,
    MacroCallStmt, MatchExpr, MemberAccessExpr, ParseResult, Parser, ReturnStmt, Span, SpawnOnExpr,
    Statement, Token, TokenType, UnsafeBlockExpr,
};

impl<'a> Parser<'a> {
    /// The compound assignment at the cursor, consumed, or `None`. Each one means the
    /// binary operator of the same name applied to the two sides, so the statement they
    /// build is checked and lowered by the same rules as `a = a OP b` -- there is no
    /// second implementation of what `%` or `>>` does.
    fn compound_assign_op(&mut self) -> Option<BinaryOp> {
        // Two tokens, so it has to be tried before the single-token spellings: `>>=`
        // begins with a `>` that nothing else here would claim, but `>=` would.
        if let Some(op) = self.shift_assign_at() {
            self.advance();
            self.advance();
            return Some(op);
        }
        let op = match self.peek().kind {
            TokenType::PlusEquals => BinaryOp::Add,
            TokenType::MinusEquals => BinaryOp::Sub,
            TokenType::StarEquals => BinaryOp::Mul,
            TokenType::SlashEquals => BinaryOp::Div,
            TokenType::PercentEquals => BinaryOp::Rem,
            TokenType::AmpersandEquals => BinaryOp::BitAnd,
            TokenType::PipeEquals => BinaryOp::BitOr,
            TokenType::CaretEquals => BinaryOp::BitXor,
            _ => return None,
        };
        self.advance();
        Some(op)
    }

    /// The rest of a statement that starts with the expression `expr`: an assignment, a compound
    /// assignment, or the expression on its own. `span` is the statement's first token.
    ///
    /// # Errors
    ///
    /// Fails when the right side does not parse, when a `;` is missing, or when a tuple
    /// assignment has a different number of elements on each side.
    fn parse_expr_or_assign_stmt(&mut self, expr: Expr, span: Span) -> ParseResult<'a, Statement> {
        if self.match_token(&TokenType::Equals) {
            let start = self.peek().clone();
            let rhs = self.parse_expr()?;
            self.consume(&TokenType::Semicolon, "Expected ';'")?;
            // `(a, b) = e` assigns each element, as in Rust. Parsed as a tuple literal, the
            // left side was a new value that the assignment wrote into and threw away.
            if Self::is_tuple_literal(&expr) {
                let mut out = Vec::new();
                let mut lowering = TupleLowering {
                    span,
                    start: &start,
                    out: &mut out,
                };
                self.lower_tuple_assign(expr, rhs, &mut lowering)?;
                let first = out.remove(0);
                self.pending_stmts.extend(out);
                return Ok(first);
            }
            Ok(Statement::Assign(AssignStmt {
                lhs: expr,
                rhs,
                span,
            }))
        } else if let Some(op) = self.compound_assign_op() {
            let rhs = self.parse_expr()?;
            self.consume(&TokenType::Semicolon, "Expected ';'")?;
            Ok(Statement::CompoundAssign(CompoundAssignStmt {
                lhs: expr,
                op,
                rhs,
                span,
                operand_ty: None,
            }))
        } else {
            let mut has_semicolon = true;
            match &expr {
                Expr::UnsafeBlock(UnsafeBlockExpr { .. })
                | Expr::ComptimeBlock(ComptimeBlockExpr { .. })
                | Expr::SpawnOn(SpawnOnExpr { .. })
                | Expr::If(IfExpr { .. })
                | Expr::Match(MatchExpr { .. }) => {
                    has_semicolon = self.match_token(&TokenType::Semicolon);
                }
                _ => {
                    if self.check(&TokenType::RightBrace) {
                        has_semicolon = false; // Optional at end of block
                    } else {
                        self.consume(&TokenType::Semicolon, "Expected ';'")?;
                    }
                }
            }
            Ok(Statement::ExprStmt(ExprStmtStmt {
                expr,
                has_semi: has_semicolon,
                span,
            }))
        }
    }

    /// Parses one statement.
    ///
    /// # Errors
    ///
    /// Fails when the tokens at the cursor are not a statement.
    pub(crate) fn parse_statement(&mut self) -> ParseResult<'a, Statement> {
        let token = self.peek();
        let token_span = Span {
            line: token.line,
            column: token.column,
            length: token.length,
        };

        match &token.kind {
            TokenType::Let => self.parse_let_stmt(token_span),
            TokenType::Comptime => self.parse_comptime_stmt(),
            TokenType::Assert => self.parse_assert_stmt(),
            TokenType::Return => self.parse_return_stmt(token_span),
            TokenType::Loop => self.parse_loop_stmt(),
            TokenType::While => self.parse_while_stmt(token_span),
            TokenType::Break => {
                self.advance();
                self.consume(&TokenType::Semicolon, "Expected ';'")?;
                Ok(Statement::Break(BreakStmt { span: token_span }))
            }
            TokenType::Continue => {
                self.advance();
                self.consume(&TokenType::Semicolon, "Expected ';'")?;
                Ok(Statement::Continue(ContinueStmt { span: token_span }))
            }
            TokenType::For => self.parse_for_stmt(token_span),
            TokenType::Identifier(s) => self.parse_identifier_stmt(s, token_span),
            // An `if`, `match`, `unsafe` or `spawn` block that starts a statement ends at its
            // closing brace, as in Rust. Read as one expression, `if c { .. }` followed by
            // `*x = 1;` would be `(if c { .. }) * x = 1`.
            TokenType::If | TokenType::Match | TokenType::Unsafe | TokenType::Spawn => {
                let expr = self.parse_primary_expr()?;
                self.parse_expr_or_assign_stmt(expr, token_span)
            }
            _ => {
                let expr = self.parse_expr()?;
                self.parse_expr_or_assign_stmt(expr, token_span)
            }
        }
    }

    /// A `let` statement. `token_span` is the span of the `let` keyword at the cursor.
    ///
    /// # Errors
    ///
    /// Fails when the name, the type, the `=`, the value or the `;` is missing or wrong.
    fn parse_let_stmt(&mut self, token_span: Span) -> ParseResult<'a, Statement> {
        self.advance();
        let mut is_mut = false;
        if self.match_token(&TokenType::Mut) {
            is_mut = true;
        }
        if !is_mut && self.check(&TokenType::LeftParen) {
            return self.parse_tuple_let(token_span);
        }
        let name = match &self.advance().kind {
            TokenType::Identifier(s) => s.to_string(),
            _ => return Err(self.error("Expected identifier after let")),
        };
        let mut type_annotation = None;
        if self.match_token(&TokenType::Colon) {
            type_annotation = Some(self.parse_type()?);
        }
        self.consume(&TokenType::Equals, "Expected '='")?;
        let expr = self.parse_expr()?;
        self.consume(&TokenType::Semicolon, "Expected ';'")?;
        Ok(Statement::LetDecl(LetDeclStmt {
            // The `let` keyword's own token: a diagnostic about the binding (a type
            // mismatch, an unused variable) points at the declaration, not at 0:0.
            name: name.into(),
            is_mut,
            ty_ann: type_annotation,
            expr,
            span: token_span,
        }))
    }

    /// A `comptime { .. }` block used as a statement.
    ///
    /// # Errors
    ///
    /// Fails when a brace is missing or a statement inside the block does not parse.
    ///
    /// # Panics
    ///
    /// Does not panic in practice: the statement taken off the list was just seen to be there.
    fn parse_comptime_stmt(&mut self) -> ParseResult<'a, Statement> {
        let keyword = self.advance().clone();
        self.consume(&TokenType::LeftBrace, "Expected '{' after 'comptime'")?;
        let mut stmts = Vec::new();
        while self.peek().kind != TokenType::RightBrace && self.peek().kind != TokenType::Eof {
            self.parse_statement_into(&mut stmts)?;
        }
        let mut ret = None;
        if let Some(Statement::ExprStmt(stmt)) = stmts.last() {
            if !stmt.has_semi {
                let last = stmts.pop().unwrap();
                if let Statement::ExprStmt(expr_stmt) = last {
                    ret = Some(Box::new(expr_stmt.expr));
                }
            }
        }
        self.consume(&TokenType::RightBrace, "Expected '}' after comptime block")?;
        Ok(Statement::ExprStmt(ExprStmtStmt {
            expr: Expr::ComptimeBlock(ComptimeBlockExpr {
                stmts,
                ret,
                span: Span {
                    line: keyword.line,
                    column: keyword.column,
                    length: keyword.length,
                },
            }),
            has_semi: true,
            span: Span::default(),
        }))
    }

    /// An `assert(condition);` or `assert(condition, "message");` statement.
    ///
    /// # Errors
    ///
    /// Fails when the condition does not parse, the message is not a string literal, or a
    /// parenthesis or the `;` is missing.
    fn parse_assert_stmt(&mut self) -> ParseResult<'a, Statement> {
        self.advance();
        self.consume(&TokenType::LeftParen, "Expected '(' after 'assert'")?;
        let expr = self.parse_expr()?;
        let mut msg = None;
        if self.match_token(&TokenType::Comma) {
            if let TokenType::StringLiteral(s) = &self.peek().kind {
                msg = Some(s.to_string());
                self.advance();
            } else {
                return Err(self.error("Expected string literal message after comma in assert"));
            }
        }
        self.consume(
            &TokenType::RightParen,
            "Expected ')' after assert condition",
        )?;
        self.consume(&TokenType::Semicolon, "Expected ';' after assert statement")?;
        Ok(Statement::Assert(AssertStmt {
            expr: Box::new(expr),
            msg,
            span: Span::default(),
        }))
    }

    /// A `return;` or `return value;` statement. `token_span` is the span of the `return` keyword.
    ///
    /// # Errors
    ///
    /// Fails when the value does not parse or the `;` is missing.
    fn parse_return_stmt(&mut self, token_span: Span) -> ParseResult<'a, Statement> {
        self.advance();
        // `return;` -- an early exit from a `void` function. Without this the parser
        // demanded an expression and reported `Expected expression, found Semicolon`,
        // so a void function had no way to return before its last statement.
        let expr = if self.check(&TokenType::Semicolon) {
            None
        } else {
            Some(self.parse_expr()?)
        };
        self.consume(&TokenType::Semicolon, "Expected ';'")?;
        Ok(Statement::Return(ReturnStmt {
            expr,
            span: token_span,
        }))
    }

    /// A `loop { .. }` statement, with any `invariant` clauses before its body.
    ///
    /// # Errors
    ///
    /// Fails when an invariant or a statement in the body does not parse, or a brace is missing.
    fn parse_loop_stmt(&mut self) -> ParseResult<'a, Statement> {
        self.advance();
        let mut invariants = Vec::new();
        while self.match_token(&TokenType::Invariant) {
            invariants.push(self.parse_expr()?);
        }
        self.consume(&TokenType::LeftBrace, "Expected '{' after loop")?;
        let mut body = Vec::new();
        while !self.check(&TokenType::RightBrace) && !self.check(&TokenType::Eof) {
            self.parse_statement_into(&mut body)?;
        }
        self.consume(&TokenType::RightBrace, "Expected '}'")?;
        Ok(Statement::Loop(LoopStmt {
            invariants,
            body,
            span: Span::default(),
        }))
    }

    /// A `while` statement, built as a `loop` whose body is `if cond { .. } else { break }`.
    /// `token_span` is the span of the `while` keyword.
    ///
    /// # Errors
    ///
    /// Fails when the condition, an invariant or a statement in the body does not parse, or a
    /// brace is missing.
    fn parse_while_stmt(&mut self, token_span: Span) -> ParseResult<'a, Statement> {
        let while_span = token_span;
        self.advance();
        let cond = self.parse_expr()?;
        let mut invariants = Vec::new();
        while self.match_token(&TokenType::Invariant) {
            invariants.push(self.parse_expr()?);
        }
        self.consume(&TokenType::LeftBrace, "Expected '{' after while condition")?;
        let mut while_body = Vec::new();
        while !self.check(&TokenType::RightBrace) && !self.check(&TokenType::Eof) {
            self.parse_statement_into(&mut while_body)?;
        }
        self.consume(&TokenType::RightBrace, "Expected '}'")?;

        let else_block = Some(vec![Statement::Break(BreakStmt { span: while_span })]);

        let if_expr = Expr::If(IfExpr {
            is_comptime: false,
            cond: Box::new(cond),
            then_block: while_body,
            else_block,
            span: while_span,
        });

        let body = vec![Statement::ExprStmt(ExprStmtStmt {
            expr: if_expr,
            has_semi: false,
            span: while_span,
        })];

        Ok(Statement::Loop(LoopStmt {
            invariants,
            body,
            span: while_span,
        }))
    }

    /// A `for name in iterable { .. }` statement. `token_span` is the span of the `for` keyword.
    ///
    /// # Errors
    ///
    /// Fails when the loop variable, the `in`, the iterable, an invariant or a statement in the
    /// body does not parse, or a brace is missing.
    fn parse_for_stmt(&mut self, token_span: Span) -> ParseResult<'a, Statement> {
        let for_span = token_span;
        self.advance();
        let iter = match &self.advance().kind {
            TokenType::Identifier(s) => s.to_string(),
            _ => return Err(self.error("Expected identifier after 'for'")),
        };
        self.consume(&TokenType::In, "Expected 'in' after for iterator")?;
        let iterable = self.parse_expr()?;
        let mut invariants = Vec::new();
        while self.match_token(&TokenType::Invariant) {
            invariants.push(self.parse_expr()?);
        }
        self.consume(&TokenType::LeftBrace, "Expected '{'")?;
        let mut stmts = Vec::new();
        while !self.check(&TokenType::RightBrace) && !self.check(&TokenType::Eof) {
            self.parse_statement_into(&mut stmts)?;
        }
        self.consume(&TokenType::RightBrace, "Expected '}'")?;
        Ok(Statement::ForLoop(ForLoopStmt {
            iter,
            iterable: Box::new(iterable),
            invariants,
            body: stmts,
            span: for_span,
            next_fn: None,
        }))
    }

    /// A statement that starts with the identifier `name`: a macro call such as `name!(..)`, or
    /// else an expression or an assignment. `token_span` is the span of the identifier.
    ///
    /// # Errors
    ///
    /// Fails when there is a space between a macro name and its `!`, or when the macro call or
    /// the expression does not parse.
    fn parse_identifier_stmt(
        &mut self,
        name: &str,
        token_span: Span,
    ) -> ParseResult<'a, Statement> {
        let ident_line = token_span.line;
        let ident_end_col = token_span.column + token_span.length;
        let next_token = self.peek_n(1);

        let mut is_macro = false;
        if next_token.kind == TokenType::Bang {
            if next_token.line == ident_line && next_token.column == ident_end_col {
                is_macro = true;
            } else {
                return Err(self.error(&format!("Macro invocations must not have spaces between the macro name and '!'. Did you mean `{name}!`?")));
            }
        }

        if is_macro {
            let name = name.to_string();
            self.advance(); // consume identifier
            self.advance(); // consume '!'
            let token_tree = self.parse_token_tree()?;
            let mut block_tree = None;
            if self.check(&TokenType::LeftBrace) {
                block_tree = Some(self.parse_token_tree()?);
            }
            let has_semi = self.match_token(&TokenType::Semicolon); // optional semicolon for statement macros
            return Ok(Statement::MacroCall(MacroCallStmt {
                name: name.into(),
                token_tree,
                block_tree,
                has_semi,
                span: Span::default(),
            }));
        }
        // fallback to expression parsing
        let expr = self.parse_expr()?;
        self.parse_expr_or_assign_stmt(expr, token_span)
    }

    /// `let (a, mut b, _) = e;`: a `let` of `e` under a name no source can spell, then one `let`
    /// per name, of the element in its place. Nested tuples recurse the same way.
    ///
    /// The first `let` is returned and the rest wait in `pending_stmts` for
    /// `parse_statement_into`, since a statement parses to one statement.
    ///
    /// # Errors
    ///
    /// Fails when the pattern, the type, the `=`, the value or the `;` is missing or wrong, or
    /// when the pattern and a tuple literal have a different number of elements.
    fn parse_tuple_let(&mut self, span: Span) -> ParseResult<'a, Statement> {
        let start = self.peek().clone();
        let pattern = self.parse_tuple_pattern()?;
        let mut type_annotation = None;
        if self.match_token(&TokenType::Colon) {
            type_annotation = Some(self.parse_type()?);
        }
        self.consume(&TokenType::Equals, "Expected '='")?;
        let expr = self.parse_expr()?;
        self.consume(&TokenType::Semicolon, "Expected ';'")?;
        let mut out = Vec::new();
        let mut lowering = TupleLowering {
            span,
            start: &start,
            out: &mut out,
        };
        self.lower_tuple_pattern(pattern, type_annotation, expr, &mut lowering)?;
        let first = out.remove(0);
        self.pending_stmts.extend(out);
        Ok(first)
    }

    /// The pattern of a tuple `let`, such as `(a, mut b, _)`. A pattern can hold another pattern.
    ///
    /// # Errors
    ///
    /// Fails when the tokens are not a list of names in parentheses, or when the list does not
    /// have 2 to `MAX_TUPLE` names.
    fn parse_tuple_pattern(&mut self) -> ParseResult<'a, TuplePattern> {
        self.consume(&TokenType::LeftParen, "Expected '('")?;
        let mut elems = Vec::new();
        while !self.check(&TokenType::RightParen) {
            if self.check(&TokenType::LeftParen) {
                elems.push(self.parse_tuple_pattern()?);
            } else {
                let is_mut = self.match_token(&TokenType::Mut);
                let name = self.expect_identifier("Expected a name in a tuple pattern")?;
                elems.push(if name == "_" && !is_mut {
                    TuplePattern::Ignore
                } else {
                    TuplePattern::Bind(name, is_mut)
                });
            }
            if !self.match_token(&TokenType::Comma) {
                break;
            }
        }
        self.consume(&TokenType::RightParen, "Expected ')' after a tuple pattern")?;
        self.tuple_struct(elems.len())?;
        Ok(TuplePattern::Tuple(elems))
    }

    /// Turns the tuple `let` of `pattern` from `expr` into plain `let` statements, added to
    /// `lowering.out`.
    ///
    /// # Errors
    ///
    /// Fails when `expr` is a tuple literal whose number of elements differs from the pattern's.
    fn lower_tuple_pattern(
        &mut self,
        pattern: TuplePattern,
        ty_ann: Option<crate::syntax::Type>,
        expr: Expr,
        lowering: &mut TupleLowering<'_, 'a>,
    ) -> ParseResult<'a, ()> {
        let span = lowering.span;
        match pattern {
            TuplePattern::Ignore => {}
            TuplePattern::Bind(name, is_mut) => {
                lowering.out.push(Statement::LetDecl(LetDeclStmt {
                    name: name.into(),
                    is_mut,
                    ty_ann,
                    expr,
                    span,
                }));
            }
            TuplePattern::Tuple(elems) => {
                // The number of elements must match: counted here for a tuple literal, and
                // otherwise checked by `core::tuple`'s `has_N_elements`.
                let literal_size = match &expr {
                    Expr::StructInit(lit) if lit.name.as_ref().starts_with("Tuple") => {
                        Some(lit.fields.len())
                    }
                    _ => None,
                };
                if literal_size.is_some_and(|n| n != elems.len()) {
                    return Err(self.error_at(
                        lowering.start,
                        &format!(
                            "the pattern has {} elements, but the tuple has {}",
                            elems.len(),
                            literal_size.unwrap_or_default()
                        ),
                    ));
                }
                let temp = format!("$tuple{}", self.tuple_lets);
                self.tuple_lets += 1;
                lowering.out.push(Statement::LetDecl(LetDeclStmt {
                    name: temp.clone().into(),
                    is_mut: false,
                    ty_ann,
                    expr,
                    span,
                }));
                let temp_expr = Expr::Identifier(IdentifierExpr {
                    name: temp.clone().into(),
                    span,
                });
                if literal_size.is_none() {
                    lowering.out.push(Statement::LetDecl(LetDeclStmt {
                        name: format!("_{temp}_size").into(),
                        is_mut: false,
                        ty_ann: None,
                        expr: super::tuple_match::has_elements(elems.len(), &temp_expr),
                        span,
                    }));
                }
                for (i, elem) in elems.into_iter().enumerate() {
                    let element = tuple_element(&temp, i, span);
                    self.lower_tuple_pattern(elem, None, element, lowering)?;
                }
            }
        }
        Ok(())
    }
}

impl<'a> Parser<'a> {
    /// Whether `e` is a tuple written out, such as `(a, b)`.
    fn is_tuple_literal(e: &Expr) -> bool {
        matches!(e, Expr::StructInit(lit) if lit.name.as_ref().starts_with("Tuple"))
    }

    /// `(a, b) = rhs`: `rhs` is held in a `$tuple` local first, so `(a, b) = (b, a)` swaps,
    /// then each place is assigned its element. `_` assigns nothing, and a nested tuple
    /// recurses.
    ///
    /// # Errors
    ///
    /// Fails when `rhs` is a tuple literal whose number of elements differs from the left side's.
    fn lower_tuple_assign(
        &mut self,
        lhs: Expr,
        rhs: Expr,
        lowering: &mut TupleLowering<'_, 'a>,
    ) -> ParseResult<'a, ()> {
        let span = lowering.span;
        let Expr::StructInit(lit) = lhs else {
            lowering
                .out
                .push(Statement::Assign(AssignStmt { lhs, rhs, span }));
            return Ok(());
        };
        if let Expr::StructInit(r) = &rhs {
            if r.name.as_ref().starts_with("Tuple") && r.fields.len() != lit.fields.len() {
                return Err(self.error_at(
                    lowering.start,
                    &format!(
                        "the left side has {} elements, but the tuple has {}",
                        lit.fields.len(),
                        r.fields.len()
                    ),
                ));
            }
        }
        let temp = format!("$tuple{}", self.tuple_lets);
        self.tuple_lets += 1;
        lowering.out.push(Statement::LetDecl(LetDeclStmt {
            name: temp.clone().into(),
            is_mut: false,
            ty_ann: None,
            expr: rhs,
            span,
        }));
        for (i, (_, place)) in lit.fields.into_iter().enumerate() {
            if matches!(&place, Expr::Identifier(id) if id.name.as_ref() == "_") {
                continue;
            }
            let element = tuple_element(&temp, i, span);
            if Self::is_tuple_literal(&place) {
                self.lower_tuple_assign(place, element, lowering)?;
            } else {
                lowering.out.push(Statement::Assign(AssignStmt {
                    lhs: place,
                    rhs: element,
                    span,
                }));
            }
        }
        Ok(())
    }
}

/// The expression `temp._i`: element `i` of the tuple held in the local variable `temp`.
fn tuple_element(temp: &str, i: usize, span: Span) -> Expr {
    Expr::MemberAccess(MemberAccessExpr {
        base: Box::new(Expr::Identifier(IdentifierExpr {
            name: temp.to_string().into(),
            span,
        })),
        member: format!("_{i}").into(),
        struct_name: None,
        span,
    })
}

/// What lowering a tuple `let` or a tuple assignment needs at every level of nesting.
struct TupleLowering<'t, 'a> {
    /// The span given to every statement that is generated.
    span: Span,
    /// The first token of the statement's tuple, which an error message points at.
    start: &'t Token<'a>,
    /// The generated statements, in order.
    out: &'t mut Vec<Statement>,
}

/// The left of a tuple `let`.
enum TuplePattern {
    /// `_`: the element is not bound.
    Ignore,
    /// A name, and whether it was written `mut`.
    Bind(String, bool),
    /// A pattern in parentheses, with one pattern for each element.
    Tuple(Vec<TuplePattern>),
}
