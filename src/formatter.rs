//===- formatter.rs - Vx Compiler ------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides the core logic for the Vx code formatter.
// It iterates over token streams produced by the lexer and applies intelligent
// spacing, indentation, and line-wrapping rules to generate idiomatic and
// readable Vx source code.
//
//===----------------------------------------------------------------------===//
use crate::lexer::{Lexer, Token, TokenType};
use std::collections::HashMap;
use std::fmt::Write as _;

/// The longest line an `unsafe { ... }` block may take and still be put on one line.
const MAX_ONE_LINE_BLOCK_WIDTH: usize = 80;

/// Formats Vx source text, indenting each block by `indent_spaces` spaces.
///
/// A file whose first token is a `//! vx-format: OFF` comment is returned unchanged.
#[must_use]
pub fn format_file(content: &str, indent_spaces: usize) -> String {
    let mut lexer = Lexer::new_with_comments(content);
    let tokens = lexer.tokenize();

    if let Some(token) = tokens.first() {
        if let TokenType::Comment(c) | TokenType::DocComment(c) = &token.kind {
            if c.contains("//! vx-format: OFF") {
                return content.to_string();
            }
        }
    }

    let tokens = normalize_and_expand_blocks(&tokens);
    let tokens = adjust_spacing(tokens);
    emit_formatted_string(tokens, indent_spaces, content.len())
}

/// A whitespace token that is not in the source: the formatter adds it.
fn added_whitespace(text: &'static str, line: usize, column: usize) -> Token<'static> {
    Token {
        kind: TokenType::Whitespace(text),
        line,
        column,
        length: 1,
    }
}

/// Whether `token` is whitespace or a comment, which is to say not code.
fn is_whitespace_or_comment(token: &Token<'_>) -> bool {
    matches!(
        token.kind,
        TokenType::Whitespace(_) | TokenType::Comment(_) | TokenType::DocComment(_)
    )
}

/// Whether any whitespace token in `tokens` holds a line break.
fn has_newline(tokens: &[Token<'_>]) -> bool {
    tokens.iter().any(|t| {
        if let TokenType::Whitespace(ws) = t.kind {
            ws.contains('\n')
        } else {
            false
        }
    })
}

/// Whether `tokens` holds anything other than whitespace and comments.
fn has_code(tokens: &[Token<'_>]) -> bool {
    tokens.iter().any(|t| !is_whitespace_or_comment(t))
}

/// Puts a space before every `{` that does not already follow whitespace.
fn space_before_left_braces<'a>(tokens: &[Token<'a>]) -> Vec<Token<'a>> {
    let mut spaced_tokens: Vec<Token<'_>> = Vec::with_capacity(tokens.len() + 10);
    for token in tokens {
        if let TokenType::LeftBrace = token.kind {
            if spaced_tokens
                .last()
                .is_some_and(|last| !matches!(last.kind, TokenType::Whitespace(_)))
            {
                spaced_tokens.push(added_whitespace(" ", token.line, token.column));
            }
        }
        spaced_tokens.push(token.clone());
    }
    spaced_tokens
}

/// Pairs each `{` with its `}` in one pass over `tokens`, using a stack.
///
/// Returns two maps of token positions: from each `{` to its `}`, and from each `}` to its `{`.
fn match_braces(tokens: &[Token<'_>]) -> (HashMap<usize, usize>, HashMap<usize, usize>) {
    let mut stack = Vec::new();
    let mut left_to_right = HashMap::new();
    let mut right_to_left = HashMap::new();
    for (idx, token) in tokens.iter().enumerate() {
        match token.kind {
            TokenType::LeftBrace => stack.push(idx),
            TokenType::RightBrace => {
                if let Some(lb_idx) = stack.pop() {
                    left_to_right.insert(lb_idx, idx);
                    right_to_left.insert(idx, lb_idx);
                }
            }
            _ => {}
        }
    }
    (left_to_right, right_to_left)
}

/// Whether the last token of `new_tokens` that is not whitespace or a comment is `unsafe`.
fn ends_with_unsafe(new_tokens: &[Token<'_>]) -> bool {
    for token in new_tokens.iter().rev() {
        match token.kind {
            TokenType::Whitespace(_) | TokenType::Comment(_) | TokenType::DocComment(_) => {}
            TokenType::Unsafe => return true,
            _ => return false,
        }
    }
    false
}

/// How long the line would be if `block` (from `{` to `}`) were put on the current line.
///
/// `new_tokens` is what has been written so far; its last line is the start of that line.
fn collapsed_line_length(new_tokens: &[Token<'_>], block: &[Token<'_>]) -> usize {
    let mut collapsed_len = 0;
    for token in new_tokens.iter().rev() {
        if let TokenType::Whitespace(ws) = token.kind {
            if let Some(pos) = ws.rfind('\n') {
                collapsed_len += ws.len() - pos - 1;
                break;
            }
            collapsed_len += token.length;
        } else {
            collapsed_len += token.length;
        }
    }
    for t in block {
        if let TokenType::Whitespace(ws) = t.kind {
            if ws.contains('\n') {
                collapsed_len += 1;
            } else {
                collapsed_len += t.length;
            }
        } else {
            collapsed_len += t.length;
        }
    }
    collapsed_len
}

/// Writes the block from `tokens[lb_idx]` to `tokens[rb_idx]` on one line: every line break
/// inside becomes a space, and the braces get a space on their inner side.
fn push_block_on_one_line<'a>(
    new_tokens: &mut Vec<Token<'a>>,
    tokens: &[Token<'a>],
    lb_idx: usize,
    rb_idx: usize,
) {
    new_tokens.push(tokens[lb_idx].clone());

    if !matches!(tokens[lb_idx + 1].kind, TokenType::Whitespace(_)) {
        new_tokens.push(added_whitespace(
            " ",
            tokens[lb_idx].line,
            tokens[lb_idx].column,
        ));
    }

    for token in &tokens[(lb_idx + 1)..rb_idx] {
        let mut t = token.clone();
        if let TokenType::Whitespace(ref mut ws) = t.kind {
            if ws.contains('\n') {
                *ws = " ";
            }
        }
        new_tokens.push(t);
    }

    if !new_tokens
        .last()
        .is_some_and(|last| matches!(last.kind, TokenType::Whitespace(_)))
    {
        new_tokens.push(added_whitespace(
            " ",
            tokens[rb_idx].line,
            tokens[rb_idx].column,
        ));
    }

    new_tokens.push(tokens[rb_idx].clone());
}

/// Handles the `{` at `lb_idx`, whose `}` is at `rb_idx`.
///
/// A short `unsafe` block is put on one line. Any other block that is written on one line and
/// holds code gets a line break after its `{`. Returns the position to continue from when the
/// `{` was written here, and `None` when the caller should copy it as it is.
fn rewrite_block_start<'a>(
    new_tokens: &mut Vec<Token<'a>>,
    tokens: &[Token<'a>],
    lb_idx: usize,
    rb_idx: usize,
) -> Option<usize> {
    let inside = &tokens[(lb_idx + 1)..rb_idx];

    // A line comment inside runs to the end of its line, so a block holding one is
    // never put on one line: the comment would swallow the code after it.
    let holds_comment = tokens[lb_idx..=rb_idx]
        .iter()
        .any(|t| matches!(t.kind, TokenType::Comment(_) | TokenType::DocComment(_)));
    if ends_with_unsafe(new_tokens)
        && !holds_comment
        && collapsed_line_length(new_tokens, &tokens[lb_idx..=rb_idx]) + 2
            <= MAX_ONE_LINE_BLOCK_WIDTH
    {
        push_block_on_one_line(new_tokens, tokens, lb_idx, rb_idx);
        return Some(rb_idx + 1);
    }

    if has_newline(inside) || !has_code(inside) {
        return None;
    }
    new_tokens.push(tokens[lb_idx].clone());

    if matches!(tokens[lb_idx + 1].kind, TokenType::Whitespace(_)) {
        let mut t = tokens[lb_idx + 1].clone();
        t.kind = TokenType::Whitespace("\n");
        new_tokens.push(t);
        Some(lb_idx + 2)
    } else {
        new_tokens.push(added_whitespace("\n", 0, 0));
        Some(lb_idx + 1)
    }
}

/// Makes sure a `}` that closes a one-line block holding code starts on a new line.
///
/// `inside` is the tokens between the braces. The `}` itself is written by the caller.
fn break_line_before_block_end(new_tokens: &mut Vec<Token<'_>>, inside: &[Token<'_>]) {
    if has_newline(inside) || !has_code(inside) {
        return;
    }
    let needs_newline = match new_tokens.last() {
        Some(t) => {
            if let TokenType::Whitespace(ws) = t.kind {
                !ws.contains('\n')
            } else {
                true
            }
        }
        None => true,
    };

    if needs_newline {
        if let Some(t) = new_tokens.last_mut() {
            if let TokenType::Whitespace(_) = t.kind {
                t.kind = TokenType::Whitespace("\n");
            } else {
                new_tokens.push(added_whitespace("\n", 0, 0));
            }
        }
    }
}

/// Puts a space before each `{`, puts short `unsafe` blocks on one line, and breaks every other
/// one-line block that holds code over several lines.
fn normalize_and_expand_blocks<'a>(tokens: &[Token<'a>]) -> Vec<Token<'a>> {
    // Pass 1a: Ensure spaces before `{`
    let tokens = space_before_left_braces(tokens);

    // Pass 1b: Match braces in O(N) using a stack
    let (left_to_right, right_to_left) = match_braces(&tokens);

    // Pass 1c: Process blocks (collapsing `unsafe` or expanding others)
    let mut new_tokens: Vec<Token<'_>> = Vec::with_capacity(tokens.len() + 20);
    let mut i = 0;
    while i < tokens.len() {
        if let TokenType::LeftBrace = tokens[i].kind {
            if let Some(&rb_idx) = left_to_right.get(&i) {
                if let Some(next) = rewrite_block_start(&mut new_tokens, &tokens, i, rb_idx) {
                    i = next;
                    continue;
                }
            }
        } else if let TokenType::RightBrace = tokens[i].kind {
            if let Some(&lb_idx) = right_to_left.get(&i) {
                break_line_before_block_end(&mut new_tokens, &tokens[(lb_idx + 1)..i]);
            }
        }

        new_tokens.push(tokens[i].clone());
        i += 1;
    }

    new_tokens
}

/// Replaces line breaks with a space where the code must stay on one line: after `for`, `if` and
/// `while`, before a comparison or `&&`/`||`, and around `else`.
fn adjust_spacing(tokens: Vec<Token<'_>>) -> Vec<Token<'_>> {
    let mut new_tokens = Vec::with_capacity(tokens.len());
    let mut last_non_ws: Option<TokenType<'_>> = None;

    let mut next_non_ws_arr = vec![None; tokens.len()];
    let mut curr_next = None;
    for (i, token) in tokens.iter().enumerate().rev() {
        next_non_ws_arr[i].clone_from(&curr_next);
        if !is_whitespace_or_comment(token) {
            curr_next = Some(token.kind.clone());
        }
    }

    for (i, mut token) in tokens.into_iter().enumerate() {
        if !is_whitespace_or_comment(&token) {
            last_non_ws = Some(token.kind.clone());
            new_tokens.push(token);
            continue;
        }

        if let TokenType::Whitespace(ref mut ws) = token.kind {
            let next_non_ws = &next_non_ws_arr[i];

            if ws.contains('\n') {
                if let Some(prev) = &last_non_ws {
                    if matches!(prev, TokenType::For | TokenType::If | TokenType::While) {
                        *ws = " ";
                    }
                }
                if let Some(next) = next_non_ws {
                    if matches!(
                        next,
                        TokenType::EqEq
                            | TokenType::NotEq
                            | TokenType::LessEq
                            | TokenType::GreaterEq
                            | TokenType::LeftAngle
                            | TokenType::RightAngle
                            | TokenType::AndAnd
                            | TokenType::OrOr
                    ) {
                        *ws = " ";
                    }
                }
            }

            if let (Some(TokenType::RightBrace), Some(TokenType::Else)) =
                (&last_non_ws, next_non_ws)
            {
                *ws = " ";
            }
            if let (Some(TokenType::Else), Some(TokenType::LeftBrace)) = (&last_non_ws, next_non_ws)
            {
                *ws = " ";
            }
        }

        new_tokens.push(token);
    }

    new_tokens
}

/// Writes the indentation for a line that is `level` blocks deep.
fn write_indent(out: &mut String, level: isize, spaces: usize) {
    if let Ok(level) = usize::try_from(level) {
        for _ in 0..(level * spaces) {
            out.push(' ');
        }
    }
}

/// Writes one whitespace token. With formatting on, a run of line breaks is kept and any other
/// whitespace at the start of a line is dropped, because the indentation is written instead.
fn emit_whitespace(formatted: &mut String, ws: &str, format_enabled: bool, is_new_line: &mut bool) {
    if format_enabled {
        if ws.contains('\n') {
            let newlines = ws.chars().filter(|&c| c == '\n').count();
            for _ in 0..newlines {
                formatted.push('\n');
            }
            *is_new_line = true;
        } else if !*is_new_line {
            formatted.push_str(ws);
        }
    } else {
        formatted.push_str(ws);
        if ws.contains('\n') {
            *is_new_line = true;
        }
    }
}

/// `text` with the spaces and tabs at the end of every line removed.
fn strip_trailing_whitespace(text: &str) -> String {
    let mut final_out = String::with_capacity(text.len());
    let mut iter = text.split('\n').peekable();
    while let Some(line) = iter.next() {
        final_out.push_str(line.trim_end_matches([' ', '\t']));
        if iter.peek().is_some() {
            final_out.push('\n');
        }
    }
    final_out
}

/// Turns the tokens back into text, indenting each line by its block depth.
///
/// Text between a `vx-format-begin: OFF` comment and a `vx-format-begin: ON` or
/// `vx-format-end: OFF` comment is copied as it is.
fn emit_formatted_string(
    tokens: Vec<Token<'_>>,
    indent_spaces: usize,
    original_len: usize,
) -> String {
    let mut formatted = String::with_capacity(original_len + original_len / 10);
    let mut indent_level: isize = 0;

    let mut is_new_line = true;
    let mut format_enabled = true;

    for token in tokens {
        if token.kind == TokenType::Eof {
            break;
        }

        match token.kind {
            TokenType::Whitespace(ws) => {
                emit_whitespace(&mut formatted, ws, format_enabled, &mut is_new_line);
            }
            TokenType::RightBrace => {
                indent_level -= 1;
                if indent_level < 0 {
                    indent_level = 0;
                }

                if format_enabled && is_new_line {
                    write_indent(&mut formatted, indent_level, indent_spaces);
                }
                is_new_line = false;
                formatted.push('}');
            }
            TokenType::LeftBrace => {
                if format_enabled && is_new_line {
                    write_indent(&mut formatted, indent_level, indent_spaces);
                }
                is_new_line = false;
                formatted.push('{');
                indent_level += 1;
            }
            TokenType::Comment(c) | TokenType::DocComment(c) => {
                let mut turning_off = false;
                if c.contains("vx-format-begin: OFF") {
                    format_enabled = false;
                    turning_off = true;
                } else if c.contains("vx-format-begin: ON") || c.contains("vx-format-end: OFF") {
                    format_enabled = true;
                }

                if (format_enabled || turning_off) && is_new_line {
                    write_indent(&mut formatted, indent_level, indent_spaces);
                }
                is_new_line = false;
                formatted.push_str(c);
            }
            other => {
                if format_enabled && is_new_line {
                    write_indent(&mut formatted, indent_level, indent_spaces);
                }
                is_new_line = false;
                write!(&mut formatted, "{other}").expect("writing to a String cannot fail");
            }
        }
    }

    // Strip trailing whitespace from all lines
    strip_trailing_whitespace(&formatted)
}

#[cfg(test)]
#[allow(
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    reason = "a test reports a failure by panicking"
)]
mod tests {
    use super::*;

    #[test]
    fn test_format_off_top_level() {
        let input = "//! vx-format: OFF\nfn   foo()   {\nlet  x=1;\n}";
        let formatted = format_file(input, 2);
        assert_eq!(formatted, input);
    }

    #[test]
    fn a_short_block_with_a_comment_stays_on_several_lines() {
        let input = "fn main() -> i32 {\n  let mut x = 1;\n  unsafe {\n    // set x\n    x = 2;\n  }\n  return x;\n}";
        assert_eq!(format_file(input, 2), input);
    }

    #[test]
    fn test_format_off_block() {
        let input = "fn foo() {\n// vx-format-begin: OFF\n      let   x = 1;\n// vx-format-end: OFF\nlet y = 2;\n}";
        let expected = "fn foo() {\n  // vx-format-begin: OFF\n      let   x = 1;\n  // vx-format-end: OFF\n  let y = 2;\n}";
        let formatted = format_file(input, 2);
        assert_eq!(formatted, expected);
    }

    #[test]
    fn test_format_newline_stripping() {
        let input = "for \n i in 0..10 {\n  if sum_idx\n == 0 { a = 1; }\n}";
        let expected = "for i in 0..10 {\n  if sum_idx == 0 {\n    a = 1;\n  }\n}";
        let formatted = format_file(input, 2);
        assert_eq!(formatted, expected);
    }

    #[test]
    fn test_format_while_loop() {
        let input = "while \n keep_going { step(); }";
        let expected = "while keep_going {\n  step();\n}";
        let formatted = format_file(input, 2);
        assert_eq!(formatted, expected);
        assert_eq!(format_file(&formatted, 2), formatted);
    }

    #[test]
    fn test_format_else_single_line_expansion() {
        let input = "if a > 0 { x = 1; } else {\n  x = 0;\n}";
        let expected = "if a > 0 {\n  x = 1;\n} else {\n  x = 0;\n}";
        let formatted = format_file(input, 2);
        assert_eq!(formatted, expected);
    }

    #[test]
    fn test_format_dangling_else() {
        let input = "if a > 0 {\n  x = 1;\n}\n else {\n  x = 0;\n}";
        let expected = "if a > 0 {\n  x = 1;\n} else {\n  x = 0;\n}";
        let formatted = format_file(input, 2);
        assert_eq!(formatted, expected);
    }
}
