//===- lexer.rs - Vx Compiler ----------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Lexical Analyzer (Scanner) for the Vx language.
// It processes raw source text and converts it into a stream of discrete tokens
// (keywords, identifiers, literals, symbols), handling whitespace, comments, and
// basic syntax validation.
//
//===----------------------------------------------------------------------===//

/// What kind of token a piece of source text is. `S` is how the text of a name, a number or
/// a comment is held, and `C` is how the text of a string literal is held.
#[derive(Debug, PartialEq, Clone, Eq)]
pub enum TokenTypeBase<S, C> {
    // Keywords
    /// The keyword `fn`.
    Fn,
    /// The keyword `let`.
    Let,
    /// The keyword `mut`.
    Mut,
    /// The keyword `for`.
    For,
    /// The keyword `in`.
    In,
    /// The keyword `if`.
    If,
    /// The keyword `else`.
    Else,
    /// The keyword `loop`.
    Loop,
    /// The keyword `while`.
    While,
    /// The keyword `break`.
    Break,
    /// The keyword `continue`.
    Continue,
    /// The keyword `return`.
    Return,
    /// The keyword `spawn`.
    Spawn,
    /// The keyword `on`.
    On,
    /// The keyword `where`.
    Where,
    /// The keyword `transfer`.
    Transfer,
    /// The keyword `unroll`.
    Unroll,
    /// The keyword `across`.
    Across,
    /// The keyword `match`.
    Match,
    /// The keyword `struct`.
    Struct,
    /// The keyword `unsafe`.
    Unsafe,
    /// The keyword `safe`.
    Safe,
    /// The keyword `extern`.
    Extern,
    /// The keyword `trait`.
    Trait,
    /// `type Item;` in a trait, `type Item = i64;` in an impl.
    TypeKw,
    /// The keyword `impl`.
    Impl,
    /// The keyword `comptime`.
    Comptime,
    /// The keyword `import`.
    Import,
    /// The keyword `assert`.
    Assert,
    /// The keyword `enum`.
    Enum,
    /// The keyword `as`.
    As,
    /// The keyword `grad`.
    Grad,
    /// The keyword `vjp`.
    Vjp,
    /// The keyword `jvp`.
    Jvp,
    /// The keyword `requires`.
    Requires,
    /// The keyword `ensures`.
    Ensures,
    /// The keyword `invariant`.
    Invariant,
    /// The keyword `macro_rules`, together with the `!` after it when there is one.
    MacroRules,

    // Types & Topology
    /// The built-in name `Topology`.
    Topology,
    /// The built-in name `Memory`.
    Memory,
    /// The built-in name `Ref`.
    Ref,
    /// The built-in name `Verified`.
    Verified,
    /// The built-in name `Pinned`.
    Pinned,
    /// The built-in name `HardwareState`.
    HardwareState,

    // Identifiers & Literals
    /// A name, such as `count`. Holds the name.
    Identifier(S),
    /// A number, such as `42` or `1.5f32`. Holds its text as written.
    Number(S),
    /// A string in double quotes. Holds its text, with each escape such as `\n` replaced
    /// by the character it stands for.
    StringLiteral(C),

    // Symbols
    /// The symbol `(`.
    LeftParen,
    /// The symbol `)`.
    RightParen,
    /// The symbol `{`.
    LeftBrace,
    /// The symbol `}`.
    RightBrace,
    /// The symbol `[`.
    LeftBracket,
    /// The symbol `]`.
    RightBracket,
    /// The symbol `<`.
    LeftAngle,
    /// The symbol `>`.
    RightAngle,
    /// The symbol `:`.
    Colon,
    /// The symbol `::`.
    DoubleColon,
    /// The symbol `;`.
    Semicolon,
    /// The symbol `,`.
    Comma,
    /// The symbol `=`.
    Equals,
    /// The symbol `+=`.
    PlusEquals,
    /// The symbol `-=`.
    MinusEquals,
    /// The symbol `*=`.
    StarEquals,
    /// The symbol `/=`.
    SlashEquals,
    /// The symbol `%=`.
    PercentEquals,
    /// The symbol `&=`.
    AmpersandEquals,
    /// The symbol `|=`.
    PipeEquals,
    /// The symbol `^=`.
    CaretEquals,
    /// The symbol `->`.
    Arrow,
    /// The symbol `=>`.
    FatArrow,
    /// The symbol `+`.
    Plus,
    /// The symbol `-`.
    Minus,
    /// The symbol `*`.
    Star,
    /// The symbol `/`.
    Slash,
    /// The symbol `%`.
    Percent,
    /// The symbol `.`.
    Dot,
    /// The symbol `..`.
    DoubleDot,
    /// The symbol `&`.
    Ampersand,
    /// The symbol `^`.
    Caret,
    /// The symbol `@`.
    At,
    /// The symbol `$`.
    Dollar,
    /// The symbol `?`.
    Question,

    // Logical & Relational
    /// The symbol `==`.
    EqEq,
    /// The symbol `!=`.
    NotEq,
    /// The symbol `<=`.
    LessEq,
    /// The symbol `>=`.
    GreaterEq,
    /// The symbol `&&`.
    AndAnd,
    /// The symbol `||`.
    OrOr,
    /// The symbol `!`.
    Bang,
    /// The symbol `|`.
    Pipe,

    // Special
    /// The end of the source.
    Eof,
    /// A character that starts no token, or the first escape in a string literal that the
    /// lexer does not know.
    Unknown(char),
    /// A `//` comment. Holds the whole comment. Only produced by a lexer that keeps comments.
    Comment(S),
    /// A `///` comment. Holds the whole comment.
    DocComment(S),
    /// A run of spaces, tabs and newlines. Holds that text. Only produced by a lexer that
    /// keeps comments.
    Whitespace(S),
}

/// A token kind whose text is borrowed from the source being lexed.
pub type TokenType<'a> = TokenTypeBase<&'a str, std::borrow::Cow<'a, str>>;
/// A token kind that owns its text, so it can outlive the source.
pub type OwnedTokenType = TokenTypeBase<crate::symbol::Symbol, String>;

/// Writes the code that needs one line for every token kind that holds no text: how a kind
/// is printed, and how a kind is copied into a `TokenTypeBase` that holds its text in another
/// way. Each entry below is such a kind and how it is written in source.
///
/// The kinds that do hold text are written by hand in each `match`. A kind that is in neither
/// place is a compile error, so a new kind cannot be forgotten.
macro_rules! plain_token_kinds {
    ($(($kind:ident, $text:literal)),* $(,)?) => {
        impl<S: std::fmt::Display, C: std::fmt::Display> std::fmt::Display for TokenTypeBase<S, C> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    $(TokenTypeBase::$kind => f.write_str($text),)*
                    TokenTypeBase::Identifier(s)
                    | TokenTypeBase::Number(s)
                    | TokenTypeBase::Comment(s)
                    | TokenTypeBase::DocComment(s)
                    | TokenTypeBase::Whitespace(s) => write!(f, "{s}"),
                    TokenTypeBase::StringLiteral(s) => write!(f, "\"{s}\""),
                    TokenTypeBase::Unknown(c) => write!(f, "{c}"),
                }
            }
        }

        impl<S, C> TokenTypeBase<S, C> {
            /// This kind, with the text it holds converted by `text`, or by `string` when it
            /// is a string literal.
            fn map_text<S2, C2>(
                self,
                text: impl FnOnce(S) -> S2,
                string: impl FnOnce(C) -> C2,
            ) -> TokenTypeBase<S2, C2> {
                match self {
                    $(TokenTypeBase::$kind => TokenTypeBase::$kind,)*
                    TokenTypeBase::Identifier(s) => TokenTypeBase::Identifier(text(s)),
                    TokenTypeBase::Number(s) => TokenTypeBase::Number(text(s)),
                    TokenTypeBase::StringLiteral(s) => TokenTypeBase::StringLiteral(string(s)),
                    TokenTypeBase::Unknown(c) => TokenTypeBase::Unknown(c),
                    TokenTypeBase::Comment(s) => TokenTypeBase::Comment(text(s)),
                    TokenTypeBase::DocComment(s) => TokenTypeBase::DocComment(text(s)),
                    TokenTypeBase::Whitespace(s) => TokenTypeBase::Whitespace(text(s)),
                }
            }

            /// The same as `map_text`, for a kind that is only borrowed.
            fn map_text_ref<'t, S2, C2>(
                &'t self,
                text: impl FnOnce(&'t S) -> S2,
                string: impl FnOnce(&'t C) -> C2,
            ) -> TokenTypeBase<S2, C2> {
                match self {
                    $(TokenTypeBase::$kind => TokenTypeBase::$kind,)*
                    TokenTypeBase::Identifier(s) => TokenTypeBase::Identifier(text(s)),
                    TokenTypeBase::Number(s) => TokenTypeBase::Number(text(s)),
                    TokenTypeBase::StringLiteral(s) => TokenTypeBase::StringLiteral(string(s)),
                    TokenTypeBase::Unknown(c) => TokenTypeBase::Unknown(*c),
                    TokenTypeBase::Comment(s) => TokenTypeBase::Comment(text(s)),
                    TokenTypeBase::DocComment(s) => TokenTypeBase::DocComment(text(s)),
                    TokenTypeBase::Whitespace(s) => TokenTypeBase::Whitespace(text(s)),
                }
            }
        }
    };
}

plain_token_kinds! {
    (Fn, "fn"),
    (Let, "let"),
    (Mut, "mut"),
    (For, "for"),
    (In, "in"),
    (If, "if"),
    (Else, "else"),
    (Loop, "loop"),
    (While, "while"),
    (Break, "break"),
    (Continue, "continue"),
    (Return, "return"),
    (Spawn, "spawn"),
    (On, "on"),
    (Where, "where"),
    (Transfer, "transfer"),
    (Unroll, "unroll"),
    (Across, "across"),
    (Match, "match"),
    (Struct, "struct"),
    (Unsafe, "unsafe"),
    (Safe, "safe"),
    (Extern, "extern"),
    (Trait, "trait"),
    (TypeKw, "type"),
    (Impl, "impl"),
    (Comptime, "comptime"),
    (Import, "import"),
    (Assert, "assert"),
    (Enum, "enum"),
    (As, "as"),
    (Grad, "grad"),
    (Vjp, "vjp"),
    (Jvp, "jvp"),
    (Requires, "requires"),
    (Ensures, "ensures"),
    (Invariant, "invariant"),
    (MacroRules, "macro_rules"),
    (Topology, "Topology"),
    (Memory, "Memory"),
    (Ref, "Ref"),
    (Verified, "Verified"),
    (Pinned, "Pinned"),
    (HardwareState, "HardwareState"),
    (LeftParen, "("),
    (RightParen, ")"),
    (LeftBrace, "{"),
    (RightBrace, "}"),
    (LeftBracket, "["),
    (RightBracket, "]"),
    (LeftAngle, "<"),
    (RightAngle, ">"),
    (Colon, ":"),
    (DoubleColon, "::"),
    (Semicolon, ";"),
    (Comma, ","),
    (Equals, "="),
    (PlusEquals, "+="),
    (MinusEquals, "-="),
    (StarEquals, "*="),
    (SlashEquals, "/="),
    (PercentEquals, "%="),
    (AmpersandEquals, "&="),
    (PipeEquals, "|="),
    (CaretEquals, "^="),
    (Arrow, "->"),
    (FatArrow, "=>"),
    (Plus, "+"),
    (Minus, "-"),
    (Star, "*"),
    (Slash, "/"),
    (Percent, "%"),
    (Dot, "."),
    (DoubleDot, ".."),
    (Ampersand, "&"),
    (Caret, "^"),
    (At, "@"),
    (Dollar, "$"),
    (Question, "?"),
    (EqEq, "=="),
    (NotEq, "!="),
    (LessEq, "<="),
    (GreaterEq, ">="),
    (AndAnd, "&&"),
    (OrOr, "||"),
    (Bang, "!"),
    (Pipe, "|"),
    (Eof, ""),
}

/// One token, and where it is in the source.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenBase<S, C> {
    /// What kind of token it is, with its text when the kind holds any.
    pub kind: TokenTypeBase<S, C>,
    /// The line the lexer was on when it finished the token. The first line is 1.
    pub line: usize,
    /// The column the token starts at. The first column is 1.
    pub column: usize,
    /// How many bytes of source the token covers.
    pub length: usize,
}

/// A token whose text is borrowed from the source being lexed.
pub type Token<'a> = TokenBase<&'a str, std::borrow::Cow<'a, str>>;
/// A token that owns its text, so it can outlive the source.
pub type OwnedToken = TokenBase<crate::symbol::Symbol, String>;

use rustc_hash::FxHashMap;

static KEYWORDS: std::sync::LazyLock<FxHashMap<&'static str, TokenType<'static>>> =
    std::sync::LazyLock::new(|| {
        let mut m = FxHashMap::default();
        m.insert("fn", TokenTypeBase::Fn);
        m.insert("let", TokenTypeBase::Let);
        m.insert("mut", TokenTypeBase::Mut);
        m.insert("for", TokenTypeBase::For);
        m.insert("in", TokenTypeBase::In);
        m.insert("if", TokenTypeBase::If);
        m.insert("else", TokenTypeBase::Else);
        m.insert("loop", TokenTypeBase::Loop);
        m.insert("while", TokenTypeBase::While);
        m.insert("break", TokenTypeBase::Break);
        m.insert("continue", TokenTypeBase::Continue);
        m.insert("return", TokenTypeBase::Return);
        m.insert("spawn", TokenTypeBase::Spawn);
        m.insert("on", TokenTypeBase::On);
        m.insert("where", TokenTypeBase::Where);
        m.insert("transfer", TokenTypeBase::Transfer);
        m.insert("unroll", TokenTypeBase::Unroll);
        m.insert("across", TokenTypeBase::Across);
        m.insert("match", TokenTypeBase::Match);
        m.insert("Topology", TokenTypeBase::Topology);
        m.insert("Memory", TokenTypeBase::Memory);
        m.insert("Ref", TokenTypeBase::Ref);
        m.insert("Verified", TokenTypeBase::Verified);
        m.insert("Pinned", TokenTypeBase::Pinned);
        m.insert("HardwareState", TokenTypeBase::HardwareState);
        m.insert("struct", TokenTypeBase::Struct);
        m.insert("unsafe", TokenTypeBase::Unsafe);
        m.insert("safe", TokenTypeBase::Safe);
        m.insert("extern", TokenTypeBase::Extern);
        m.insert("trait", TokenTypeBase::Trait);
        m.insert("type", TokenTypeBase::TypeKw);
        m.insert("impl", TokenTypeBase::Impl);
        m.insert("comptime", TokenTypeBase::Comptime);
        m.insert("import", TokenTypeBase::Import);
        m.insert("assert", TokenTypeBase::Assert);
        m.insert("enum", TokenTypeBase::Enum);
        m.insert("as", TokenTypeBase::As);
        m.insert("grad", TokenTypeBase::Grad);
        m.insert("vjp", TokenTypeBase::Vjp);
        m.insert("jvp", TokenTypeBase::Jvp);
        m.insert("requires", TokenTypeBase::Requires);
        m.insert("ensures", TokenTypeBase::Ensures);
        m.insert("invariant", TokenTypeBase::Invariant);
        m
    });

/// Splits source text into tokens, one at a time.
pub struct Lexer<'a> {
    source: &'a str,
    iter: std::iter::Peekable<std::str::CharIndices<'a>>,
    line: usize,
    column: usize,
    /// Whether `//` comments and whitespace come out as tokens. When false they are skipped.
    pub preserve_comments: bool,
}

impl<'a> Lexer<'a> {
    /// A lexer over `source` that skips `//` comments and whitespace.
    #[must_use]
    pub fn new(source: &'a str) -> Self {
        Self {
            source,
            iter: source.char_indices().peekable(),
            line: 1,
            column: 1,
            preserve_comments: false,
        }
    }

    /// A lexer over `source` that gives `//` comments and whitespace back as tokens, for a
    /// tool that has to keep them, such as the formatter.
    #[must_use]
    pub fn new_with_comments(source: &'a str) -> Self {
        Self {
            source,
            iter: source.char_indices().peekable(),
            line: 1,
            column: 1,
            preserve_comments: true,
        }
    }

    fn advance(&mut self) -> Option<(usize, char)> {
        let res = self.iter.next()?;
        if res.1 == '\n' {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }
        Some(res)
    }

    fn peek(&mut self) -> Option<&(usize, char)> {
        self.iter.peek()
    }

    fn peek_char(&mut self) -> Option<char> {
        self.iter.peek().map(|&(_, c)| c)
    }

    fn current_byte_offset(&mut self) -> usize {
        let end = self.source.len();
        self.peek().map_or(end, |&(idx, _)| idx)
    }

    fn skip_whitespace(&mut self) {
        loop {
            let offset = self.current_byte_offset();
            if offset >= self.source.len() {
                break;
            }

            let mut chars = self.source[offset..].chars();
            if let Some(c) = chars.next() {
                if c.is_whitespace() {
                    self.advance();
                } else if self.source[offset..].starts_with("//") {
                    let rest = &self.source[offset..];
                    if rest.starts_with("///") && !rest.starts_with("////") {
                        break; // doc comment
                    }
                    // Line comment
                    self.advance(); // consume first '/'
                    self.advance(); // consume second '/'
                    while let Some((_, c)) = self.advance() {
                        if c == '\n' {
                            break;
                        }
                    }
                } else {
                    break;
                }
            } else {
                break;
            }
        }
    }

    fn identifier_or_keyword(&mut self, start_byte: usize, start_col: usize) -> Token<'a> {
        while let Some(c) = self.peek_char() {
            if c.is_alphanumeric() || c == '_' {
                self.advance();
            } else {
                break;
            }
        }

        let end_byte = self.current_byte_offset();
        let text = &self.source[start_byte..end_byte];

        let kind = if let Some(k) = KEYWORDS.get(text) {
            k.clone()
        } else if text == "macro_rules" {
            if self.peek_char() == Some('!') {
                self.advance(); // consume '!'
            }
            TokenTypeBase::MacroRules
        } else {
            TokenTypeBase::Identifier(text)
        };

        TokenBase {
            kind,
            line: self.line,
            column: start_col,
            length: self.current_byte_offset() - start_byte,
        }
    }

    fn number(&mut self, start_byte: usize, start_col: usize) -> Token<'a> {
        while let Some(c) = self.peek_char() {
            if c.is_ascii_digit() || c.is_alphabetic() || c == '_' {
                self.advance();
            } else if c == '.' {
                let offset = self.current_byte_offset();
                let mut chars = self.source[offset + 1..].chars();
                if chars.next() == Some('.') {
                    break;
                }
                self.advance();
            } else {
                break;
            }
        }

        let end_byte = self.current_byte_offset();
        let text = &self.source[start_byte..end_byte];

        TokenBase {
            kind: TokenTypeBase::Number(text),
            line: self.line,
            column: start_col,
            length: self.current_byte_offset() - start_byte,
        }
    }

    fn string_literal(&mut self, start_byte: usize, start_col: usize) -> Token<'a> {
        let mut text: Option<String> = None;
        let mut raw_len = 0;
        let mut unknown_escape = None;
        while let Some(next_c) = self.peek_char() {
            if next_c == '"' {
                self.advance();
                break;
            }
            self.advance();
            let mut char_to_push = next_c;
            if char_to_push == '\\' {
                if text.is_none() {
                    let mut initial_text = String::with_capacity(raw_len + 4);
                    initial_text.push_str(&self.source[start_byte + 1..start_byte + 1 + raw_len]);
                    text = Some(initial_text);
                }
                if let Some(esc_c) = self.peek_char() {
                    self.advance();
                    if let Some(value) = Self::escape_value(esc_c) {
                        char_to_push = value;
                    } else {
                        char_to_push = esc_c;
                        if unknown_escape.is_none() {
                            unknown_escape = Some(esc_c);
                        }
                    }
                }
            } else if text.is_none() {
                raw_len += char_to_push.len_utf8();
            }
            if let Some(ref mut t) = text {
                t.push(char_to_push);
            }
        }

        let end_byte = self.current_byte_offset();

        if let Some(c) = unknown_escape {
            return TokenBase {
                kind: TokenTypeBase::Unknown(c),
                line: self.line,
                column: start_col,
                length: self.current_byte_offset() - start_byte,
            };
        }

        let literal = if let Some(t) = text {
            std::borrow::Cow::Owned(t)
        } else {
            std::borrow::Cow::Borrowed(&self.source[start_byte + 1..end_byte - 1])
        };

        TokenBase {
            kind: TokenTypeBase::StringLiteral(literal),
            line: self.line,
            column: start_col,
            length: self.current_byte_offset() - start_byte,
        }
    }

    /// The character that a backslash followed by `esc_c` stands for in a string literal, or
    /// `None` when there is no such escape.
    fn escape_value(esc_c: char) -> Option<char> {
        match esc_c {
            'n' => Some('\n'),
            't' => Some('\t'),
            'r' => Some('\r'),
            '"' => Some('"'),
            '\\' => Some('\\'),
            _ => None,
        }
    }

    fn lex_comment(&mut self, start_byte: usize, start_col: usize, is_doc: bool) -> Token<'a> {
        while let Some(next_c) = self.peek_char() {
            if next_c == '\n' {
                break;
            }
            self.advance();
        }
        let end_byte = self.current_byte_offset();
        let comment = &self.source[start_byte..end_byte];
        TokenBase {
            kind: if is_doc {
                TokenTypeBase::DocComment(comment)
            } else {
                TokenTypeBase::Comment(comment)
            },
            line: self.line,
            column: start_col,
            length: end_byte - start_byte,
        }
    }

    fn lex_whitespace(&mut self, start_byte: usize, start_col: usize) -> Token<'a> {
        while let Some(next_c) = self.peek_char() {
            if next_c.is_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
        let end_byte = self.current_byte_offset();
        let ws = &self.source[start_byte..end_byte];
        TokenBase {
            kind: TokenTypeBase::Whitespace(ws),
            line: self.line,
            column: start_col,
            length: end_byte - start_byte,
        }
    }

    fn lex_slash_or_comment(&mut self, start_byte: usize, start_col: usize) -> Option<Token<'a>> {
        let rest = &self.source[self.current_byte_offset()..];
        if rest.starts_with("///") && !rest.starts_with("////") {
            self.advance(); // consume first '/'
            self.advance(); // consume second '/'
            self.advance(); // consume third '/'
            return Some(self.lex_comment(start_byte, start_col, true));
        }

        if self.preserve_comments && rest.starts_with("//") {
            self.advance(); // consume first '/'
            self.advance(); // consume second '/'
            return Some(self.lex_comment(start_byte, start_col, false));
        }

        None
    }

    /// If the next character is `second`, takes it and returns `two`. Otherwise returns `one`.
    /// For a symbol that is one character long or two, such as `+` and `+=`.
    fn one_or_two(
        &mut self,
        second: char,
        one: TokenType<'a>,
        two: TokenType<'a>,
    ) -> TokenType<'a> {
        if self.peek_char() == Some(second) {
            self.advance();
            two
        } else {
            one
        }
    }

    /// The symbol that starts with `c`, which has just been taken from the source. Takes the
    /// second character too when the two together are one symbol, such as `->`.
    fn symbol(&mut self, c: char) -> TokenType<'a> {
        match c {
            '(' => TokenTypeBase::LeftParen,
            ')' => TokenTypeBase::RightParen,
            '{' => TokenTypeBase::LeftBrace,
            '}' => TokenTypeBase::RightBrace,
            '[' => TokenTypeBase::LeftBracket,
            ']' => TokenTypeBase::RightBracket,

            ';' => TokenTypeBase::Semicolon,
            ',' => TokenTypeBase::Comma,
            '+' => self.one_or_two('=', TokenTypeBase::Plus, TokenTypeBase::PlusEquals),
            '*' => self.one_or_two('=', TokenTypeBase::Star, TokenTypeBase::StarEquals),
            '@' => TokenTypeBase::At,
            '$' => TokenTypeBase::Dollar,
            '?' => TokenTypeBase::Question,
            '/' => self.one_or_two('=', TokenTypeBase::Slash, TokenTypeBase::SlashEquals),
            '%' => self.one_or_two('=', TokenTypeBase::Percent, TokenTypeBase::PercentEquals),
            '^' => self.one_or_two('=', TokenTypeBase::Caret, TokenTypeBase::CaretEquals),
            '=' => {
                if self.peek_char() == Some('=') {
                    self.advance();
                    TokenTypeBase::EqEq
                } else {
                    self.one_or_two('>', TokenTypeBase::Equals, TokenTypeBase::FatArrow)
                }
            }
            '!' => self.one_or_two('=', TokenTypeBase::Bang, TokenTypeBase::NotEq),
            '<' => self.one_or_two('=', TokenTypeBase::LeftAngle, TokenTypeBase::LessEq),
            '>' => self.one_or_two('=', TokenTypeBase::RightAngle, TokenTypeBase::GreaterEq),
            '&' => {
                if self.peek_char() == Some('&') {
                    self.advance();
                    TokenTypeBase::AndAnd
                } else {
                    self.one_or_two(
                        '=',
                        TokenTypeBase::Ampersand,
                        TokenTypeBase::AmpersandEquals,
                    )
                }
            }
            '|' => {
                if self.peek_char() == Some('|') {
                    self.advance();
                    TokenTypeBase::OrOr
                } else {
                    self.one_or_two('=', TokenTypeBase::Pipe, TokenTypeBase::PipeEquals)
                }
            }
            '.' => self.one_or_two('.', TokenTypeBase::Dot, TokenTypeBase::DoubleDot),
            '-' => {
                if self.peek_char() == Some('>') {
                    self.advance();
                    TokenTypeBase::Arrow
                } else {
                    self.one_or_two('=', TokenTypeBase::Minus, TokenTypeBase::MinusEquals)
                }
            }
            ':' => self.one_or_two(':', TokenTypeBase::Colon, TokenTypeBase::DoubleColon),
            _ => TokenTypeBase::Unknown(c),
        }
    }

    /// Takes the next token from the source. At the end of the source it returns an `Eof`
    /// token, and keeps returning one if it is called again.
    pub fn next_token(&mut self) -> Token<'a> {
        if !self.preserve_comments {
            self.skip_whitespace();
        }

        let start_col = self.column;

        let Some(&(start_byte, c)) = self.peek() else {
            return TokenBase {
                kind: TokenTypeBase::Eof,
                line: self.line,
                column: self.column,
                length: 0,
            };
        };

        if self.preserve_comments && c.is_whitespace() {
            return self.lex_whitespace(start_byte, start_col);
        }

        if c == '/' {
            if let Some(token) = self.lex_slash_or_comment(start_byte, start_col) {
                return token;
            }
        }

        self.advance(); // consume c

        if c.is_alphabetic() || c == '_' {
            return self.identifier_or_keyword(start_byte, start_col);
        }

        if c.is_ascii_digit() {
            return self.number(start_byte, start_col);
        }

        if c == '"' {
            return self.string_literal(start_byte, start_col);
        }

        let kind = self.symbol(c);

        TokenBase {
            kind,
            line: self.line,
            column: start_col,
            length: self.current_byte_offset() - start_byte,
        }
    }

    /// Every token from here to the end of the source. The last one is the `Eof` token.
    pub fn tokenize(&mut self) -> Vec<Token<'a>> {
        let mut tokens = Vec::new();
        loop {
            let t = self.next_token();
            let is_eof = t.kind == TokenTypeBase::Eof;
            tokens.push(t);
            if is_eof {
                break;
            }
        }
        tokens
    }
}

impl Token<'_> {
    /// The same token with its own copy of its text, so it can outlive the source.
    #[must_use]
    pub fn into_owned(self) -> OwnedToken {
        let kind = self
            .kind
            .map_text(crate::symbol::Symbol::from, std::borrow::Cow::into_owned);
        OwnedToken {
            kind,
            line: self.line,
            column: self.column,
            length: self.length,
        }
    }
}

impl OwnedToken {
    /// The same token with its text borrowed from this one.
    #[must_use]
    pub fn as_token(&self) -> Token<'_> {
        let kind = self
            .kind
            .map_text_ref(|s| &**s, |s| std::borrow::Cow::Borrowed(s.as_str()));
        Token {
            kind,
            line: self.line,
            column: self.column,
            length: self.length,
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    reason = "a test reports a failure by panicking"
)]
mod tests {
    use super::*;

    fn lex(input: &str) -> Vec<TokenType<'_>> {
        let mut lexer = Lexer::new(input);
        lexer
            .tokenize()
            .into_iter()
            .map(|t| t.kind)
            .filter(|k| *k != TokenTypeBase::Eof)
            .collect()
    }

    fn lex_first(input: &str) -> TokenType<'_> {
        let mut lexer = Lexer::new(input);
        lexer.tokenize().into_iter().next().unwrap().kind
    }

    #[test]
    fn test_lex_keywords() {
        assert_eq!(lex_first("fn"), TokenTypeBase::Fn);
        assert_eq!(lex_first("let"), TokenTypeBase::Let);
        assert_eq!(lex_first("mut"), TokenTypeBase::Mut);
        assert_eq!(lex_first("for"), TokenTypeBase::For);
        assert_eq!(lex_first("in"), TokenTypeBase::In);
        assert_eq!(lex_first("if"), TokenTypeBase::If);
        assert_eq!(lex_first("else"), TokenTypeBase::Else);
        assert_eq!(lex_first("return"), TokenTypeBase::Return);
        assert_eq!(lex_first("match"), TokenTypeBase::Match);
        assert_eq!(lex_first("struct"), TokenTypeBase::Struct);
        assert_eq!(lex_first("enum"), TokenTypeBase::Enum);
        assert_eq!(lex_first("unsafe"), TokenTypeBase::Unsafe);
        assert_eq!(lex_first("extern"), TokenTypeBase::Extern);
        assert_eq!(lex_first("trait"), TokenTypeBase::Trait);
        assert_eq!(lex_first("impl"), TokenTypeBase::Impl);
        assert_eq!(lex_first("import"), TokenTypeBase::Import);
        assert_eq!(lex_first("grad"), TokenTypeBase::Grad);
        assert_eq!(lex_first("vjp"), TokenTypeBase::Vjp);
        assert_eq!(lex_first("jvp"), TokenTypeBase::Jvp);
    }

    #[test]
    fn test_lex_operators() {
        assert_eq!(lex_first("+"), TokenTypeBase::Plus);
        assert_eq!(lex_first("-"), TokenTypeBase::Minus);
        assert_eq!(lex_first("*"), TokenTypeBase::Star);
        assert_eq!(lex_first("/"), TokenTypeBase::Slash);
        assert_eq!(lex_first("%"), TokenTypeBase::Percent);
        assert_eq!(lex_first("^"), TokenTypeBase::Caret);
        assert_eq!(lex_first("@"), TokenTypeBase::At);
        assert_eq!(lex_first("("), TokenTypeBase::LeftParen);
        assert_eq!(lex_first(")"), TokenTypeBase::RightParen);
        assert_eq!(lex_first("{"), TokenTypeBase::LeftBrace);
        assert_eq!(lex_first("}"), TokenTypeBase::RightBrace);
        assert_eq!(lex_first("["), TokenTypeBase::LeftBracket);
        assert_eq!(lex_first("]"), TokenTypeBase::RightBracket);
        assert_eq!(lex_first(";"), TokenTypeBase::Semicolon);
        assert_eq!(lex_first(","), TokenTypeBase::Comma);
        assert_eq!(lex_first("&"), TokenTypeBase::Ampersand);
        assert_eq!(lex_first("!"), TokenTypeBase::Bang);
        assert_eq!(lex_first("."), TokenTypeBase::Dot);
        assert_eq!(lex_first(":"), TokenTypeBase::Colon);
    }

    #[test]
    fn test_lex_multi_char_operators() {
        assert_eq!(lex_first("=="), TokenTypeBase::EqEq);
        assert_eq!(lex_first("!="), TokenTypeBase::NotEq);
        assert_eq!(lex_first("<="), TokenTypeBase::LessEq);
        assert_eq!(lex_first(">="), TokenTypeBase::GreaterEq);
        assert_eq!(lex_first("&&"), TokenTypeBase::AndAnd);
        assert_eq!(lex_first("||"), TokenTypeBase::OrOr);
        assert_eq!(lex_first("::"), TokenTypeBase::DoubleColon);
        assert_eq!(lex_first(".."), TokenTypeBase::DoubleDot);
        assert_eq!(lex_first("->"), TokenTypeBase::Arrow);
        assert_eq!(lex_first("=>"), TokenTypeBase::FatArrow);
        assert_eq!(lex_first("+="), TokenTypeBase::PlusEquals);
        assert_eq!(lex_first("-="), TokenTypeBase::MinusEquals);
        assert_eq!(lex_first("*="), TokenTypeBase::StarEquals);
        assert_eq!(lex_first("/="), TokenTypeBase::SlashEquals);
        assert_eq!(lex_first("%="), TokenTypeBase::PercentEquals);
        assert_eq!(lex_first("&="), TokenTypeBase::AmpersandEquals);
        assert_eq!(lex_first("|="), TokenTypeBase::PipeEquals);
        assert_eq!(lex_first("^="), TokenTypeBase::CaretEquals);
    }

    #[test]
    fn test_lex_number_integer() {
        assert_eq!(lex_first("42"), TokenTypeBase::Number("42"));
    }

    #[test]
    fn test_lex_number_float() {
        assert_eq!(lex_first("3.14"), TokenTypeBase::Number("3.14"));
    }

    #[test]
    fn test_lex_number_with_suffix() {
        assert_eq!(lex_first("42i64"), TokenTypeBase::Number("42i64"));
    }

    #[test]
    fn test_lex_string_literal() {
        let kind = lex_first("\"hello\"");
        if let TokenTypeBase::StringLiteral(s) = kind {
            assert_eq!(s.as_ref(), "hello");
        } else {
            panic!("Expected StringLiteral, got {kind:?}");
        }
    }

    #[test]
    fn test_lex_identifier() {
        assert_eq!(lex_first("my_var"), TokenTypeBase::Identifier("my_var"));
    }

    #[test]
    fn test_lex_identifier_vs_keyword() {
        // "fn" is a keyword, "fn_name" is an identifier
        assert_eq!(lex_first("fn"), TokenTypeBase::Fn);
        assert_eq!(lex_first("fn_name"), TokenTypeBase::Identifier("fn_name"));
    }

    #[test]
    fn test_lex_empty_input() {
        let mut lexer = Lexer::new("");
        let tokens = lexer.tokenize();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, TokenTypeBase::Eof);
    }

    #[test]
    fn test_lex_line_tracking() {
        let input = "a\nb";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        // Filter out Eof
        let non_eof: Vec<_> = tokens
            .iter()
            .filter(|t| t.kind != TokenTypeBase::Eof)
            .collect();
        assert_eq!(non_eof.len(), 2);
        assert_eq!(non_eof[0].line, 1);
        assert_eq!(non_eof[1].line, 2);
    }

    #[test]
    fn test_lex_comment_skipped_by_default() {
        let input = "a // comment\nb";
        let kinds = lex(input);
        // Comments should be skipped by default
        assert_eq!(kinds.len(), 2);
        assert_eq!(kinds[0], TokenTypeBase::Identifier("a"));
        assert_eq!(kinds[1], TokenTypeBase::Identifier("b"));
    }

    #[test]
    fn test_lex_comment_preserved() {
        let input = "a // comment";
        let mut lexer = Lexer::new_with_comments(input);
        let tokens = lexer.tokenize();
        let has_comment = tokens
            .iter()
            .any(|t| matches!(&t.kind, TokenTypeBase::Comment(_)));
        assert!(
            has_comment,
            "Expected comment token when preserve_comments is true"
        );
    }

    #[test]
    fn test_lex_multiple_expressions() {
        let kinds = lex("a + b * c");
        assert_eq!(kinds.len(), 5);
        assert_eq!(kinds[0], TokenTypeBase::Identifier("a"));
        assert_eq!(kinds[1], TokenTypeBase::Plus);
        assert_eq!(kinds[2], TokenTypeBase::Identifier("b"));
        assert_eq!(kinds[3], TokenTypeBase::Star);
        assert_eq!(kinds[4], TokenTypeBase::Identifier("c"));
    }

    #[test]
    fn test_lex_function_signature() {
        let kinds = lex("fn foo(x: i32) -> f32");
        assert_eq!(kinds[0], TokenTypeBase::Fn);
        assert_eq!(kinds[1], TokenTypeBase::Identifier("foo"));
        assert_eq!(kinds[2], TokenTypeBase::LeftParen);
        assert_eq!(kinds[3], TokenTypeBase::Identifier("x"));
        assert_eq!(kinds[4], TokenTypeBase::Colon);
        assert_eq!(kinds[5], TokenTypeBase::Identifier("i32"));
        assert_eq!(kinds[6], TokenTypeBase::RightParen);
        assert_eq!(kinds[7], TokenTypeBase::Arrow);
        assert_eq!(kinds[8], TokenTypeBase::Identifier("f32"));
    }
}
