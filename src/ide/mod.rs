//! What an editor asks the compiler about a file being edited: its errors, what a name is,
//! and where a name is defined.

use crate::hir::{GlobalAstEnv, TypeChecker};
use crate::module_loader::ModuleLoader;
use crate::parser::MacroExpander;
use crate::session::{GlobalSession, LocalWorkerState};
use crate::syntax::decl::Program;
use std::collections::HashMap;
// Writing into a `String` cannot fail, so the results of `write!` below are dropped.
use std::fmt::Write as _;

/// One error to underline in the editor. Lines and columns count from zero.
#[derive(Debug, Clone)]
pub struct IdeDiagnostic {
    /// The line the underline starts on.
    pub line_start: usize,
    /// The column the underline starts at.
    pub col_start: usize,
    /// The line the underline ends on.
    pub line_end: usize,
    /// The column just after the end of the underline.
    pub col_end: usize,
    /// The error text to show.
    pub message: String,
}

/// The text to show when the mouse rests on a word, and where that word is. Lines and columns
/// count from zero.
#[derive(Debug, Clone)]
pub struct HoverInfo {
    /// The text to show, as Markdown.
    pub value: String,
    /// The line the word is on.
    pub line_start: usize,
    /// The column the word starts at.
    pub col_start: usize,
    /// The line the word is on.
    pub line_end: usize,
    /// The column just after the end of the word.
    pub col_end: usize,
}

/// Where a name is defined. Lines and columns count from zero.
#[derive(Debug, Clone)]
pub struct DefinitionLocation {
    /// The file the definition is in.
    pub uri: String,
    /// The line the defined name is on.
    pub line_start: usize,
    /// The column the defined name starts at.
    pub col_start: usize,
    /// The line the defined name is on.
    pub line_end: usize,
    /// The column just after the end of the defined name.
    pub col_end: usize,
}

/// The files open in the editor, with the text the editor currently shows for each.
#[derive(Default)]
pub struct AnalysisHost {
    /// File URI to the current text of that file.
    pub files: HashMap<String, String>,
}

impl AnalysisHost {
    /// A host with no open files.
    #[must_use]
    pub fn new() -> Self {
        Self {
            files: HashMap::new(),
        }
    }

    /// Records that the file `uri` now has the text `text`.
    pub fn apply_change(&mut self, uri: String, text: String) {
        self.files.insert(uri, text);
    }

    /// A copy of the open files to answer questions from, so later edits do not change the
    /// answers.
    #[must_use]
    pub fn snapshot(&self) -> Analysis {
        Analysis {
            files: self.files.clone(),
        }
    }
}

/// A copy of the open files at one moment, which the editor's questions are answered from.
pub struct Analysis {
    /// File URI to the text of that file.
    pub files: HashMap<String, String>,
}

/// Stage the in-memory buffer where the module loader can read it.
///
/// The loader takes a path, not text. These paths used to be absolute and machine-specific,
/// so on any other machine the write failed, the failure was discarded, and analysis ran
/// against a file that was stale or absent. Keyed by process id so two servers cannot race
/// on one scratch file.
fn scratch_file(stem: &str, text: &str) -> Option<String> {
    let path = std::env::temp_dir().join(format!("vx-{stem}-{}.vx", std::process::id()));
    std::fs::write(&path, text).ok()?;
    Some(path.to_string_lossy().into_owned())
}

/// The diagnostic for a file that failed to load. The position is read out of the message,
/// which contains `Error at <line>:<column>:` when the loader knows it; otherwise the
/// diagnostic is put at the start of the file.
fn load_error_diagnostic(err_str: String) -> IdeDiagnostic {
    let mut line = 1;
    let mut col = 1;

    if let Some(idx) = err_str.find("Error at ") {
        let rest = &err_str[idx + 9..];
        if let Some(colon_idx) = rest.find(':') {
            if let Ok(l) = rest[..colon_idx].parse::<usize>() {
                line = l;
                let rest_after_col = &rest[colon_idx + 1..];
                if let Some(second_colon) = rest_after_col.find(':') {
                    if let Ok(c) = rest_after_col[..second_colon].parse::<usize>() {
                        col = c;
                    }
                }
            }
        }
    }

    IdeDiagnostic {
        line_start: line.saturating_sub(1),
        col_start: col.saturating_sub(1),
        line_end: line.saturating_sub(1),
        col_end: col,
        message: err_str,
    }
}

/// Expands macros and resolves names in `modules`, the two steps that come before type
/// checking.
fn expand_and_resolve(modules: &mut [Program]) {
    // 1. Macro Expansion
    let mut global_macros = std::collections::HashMap::new();
    for m in modules.iter() {
        for mac in &m.macros {
            global_macros.insert(mac.name.clone(), mac.rules.clone());
        }
    }
    let trait_defaults = crate::resolver::collect_trait_defaults(modules.iter());
    let expander = MacroExpander::new(&global_macros, &trait_defaults);
    for m in modules.iter_mut() {
        let _ = expander.expand_module(m); // Ignore expansion errors for IDE
    }

    // 2. Name Resolution
    let symbol_map = crate::resolver::build_symbol_map(modules);
    let topologies = crate::resolver::collect_topologies(modules);
    for m in modules.iter_mut() {
        m.resolve_names(&symbol_map, &topologies);
    }
}

/// Type checks every function in `modules` and adds one diagnostic per error to
/// `diagnostics`.
fn type_check_diagnostics(
    modules: &mut [Program],
    global_env: &GlobalAstEnv<'_>,
    diagnostics: &mut Vec<IdeDiagnostic>,
) {
    let session = std::sync::Arc::new(GlobalSession::new(0));

    for p in modules {
        for func in &mut p.functions {
            let mut worker_state = LocalWorkerState::new(session.clone());
            let mut checker = TypeChecker::new(global_env, &mut worker_state);
            checker.check_function(func);

            for d in checker.errors {
                let (ls, cs) = match &d.source_span {
                    Some(s) => (s.line.saturating_sub(1), s.column.saturating_sub(1)),
                    None => (0, 0),
                };
                let (le, ce) = match &d.source_span {
                    Some(s) => (
                        s.line.saturating_sub(1),
                        s.column.saturating_sub(1) + s.length.max(1),
                    ),
                    None => (0, 0),
                };
                diagnostics.push(IdeDiagnostic {
                    line_start: ls,
                    col_start: cs,
                    line_end: le,
                    col_end: ce,
                    message: d.message.to_string(),
                });
            }
        }
    }
}

/// The word under the cursor at `line` and `col` of `text`, with the columns it starts at and
/// ends before. `None` when the cursor is past the end of the line or not on a word.
fn word_at(text: &str, line: usize, col: usize) -> Option<(String, usize, usize)> {
    let target_line = text.lines().nth(line)?;

    if col >= target_line.len() {
        return None;
    }

    let mut start_col = col;
    let mut end_col = col;
    let chars: Vec<char> = target_line.chars().collect();

    while start_col > 0 && (chars[start_col - 1].is_alphanumeric() || chars[start_col - 1] == '_') {
        start_col -= 1;
    }

    while end_col < chars.len() && (chars[end_col].is_alphanumeric() || chars[end_col] == '_') {
        end_col += 1;
    }

    if start_col == end_col {
        return None; // Not a word
    }

    Some((
        chars[start_col..end_col].iter().collect(),
        start_col,
        end_col,
    ))
}

/// The declaration of `word` as Vx code, and its doc comment when it has one. The code is
/// empty when `word` names no function, struct or enum.
fn declaration_text(global_env: &GlobalAstEnv<'_>, word: &str) -> (String, Option<String>) {
    let word_sym: crate::symbol::Symbol = word.into();
    let mut hover_text = String::new();

    let mut doc = None;

    if let Some((ty, is_unsafe, params, top, _, _)) = global_env.functions.get(&word_sym) {
        let _ = write!(hover_text, "fn {word}(");
        let params_str = params
            .iter()
            .map(|t| format!("{t}"))
            .collect::<Vec<_>>()
            .join(", ");
        hover_text.push_str(&params_str);
        let _ = write!(hover_text, ") -> {ty}");
        if *is_unsafe {
            hover_text = format!("unsafe {hover_text}");
        }
        let _ = write!(hover_text, " [Topology: {:?}]", top.kind());

        if let Some(f) = global_env.syntax_functions.get(&word_sym) {
            doc.clone_from(&f.doc_comment);
        }
    } else if let Some(struct_decl) = global_env.structs.get(&word_sym) {
        let _ = writeln!(hover_text, "struct {word} {{");
        for (name, ty) in &struct_decl.fields {
            let _ = writeln!(hover_text, "    {name}: {ty},");
        }
        hover_text.push('}');
        doc.clone_from(&struct_decl.doc_comment);
    } else if let Some(enum_decl) = global_env.enums.get(&word_sym) {
        let _ = writeln!(hover_text, "enum {word} {{");
        for var in &enum_decl.variants {
            let _ = writeln!(hover_text, "    {},", var.0);
        }
        hover_text.push('}');
        doc.clone_from(&enum_decl.doc_comment);
    }

    (hover_text, doc)
}

impl Analysis {
    /// The errors in the file `uri`: the load error when the file does not parse, otherwise
    /// every type error in it and in the modules it imports.
    #[must_use]
    pub fn diagnostics(&self, uri: &str) -> Vec<IdeDiagnostic> {
        let Some(text) = self.files.get(uri) else {
            return vec![];
        };

        let mut diagnostics = Vec::new();

        let Some(temp_path) = scratch_file("diagnostics", text) else {
            return diagnostics;
        };

        let mut loader = ModuleLoader::new();
        if let Err(e) = loader.load_main(&temp_path) {
            diagnostics.push(load_error_diagnostic(format!("{e}")));

            return diagnostics;
        }

        let mut modules: Vec<_> = loader.loaded_modules.values().cloned().collect();
        expand_and_resolve(&mut modules);

        let global_env_modules: Vec<_> = modules.iter().map(Program::clone_signature).collect();
        let mut global_env = GlobalAstEnv::build(&global_env_modules);
        global_env.annotate_return_provenances(&modules);
        type_check_diagnostics(&mut modules, &global_env, &mut diagnostics);

        diagnostics
    }

    /// What to show for the word at `line` and `col` of the file `uri`: the declaration of
    /// the function, struct or enum it names, under its doc comment.
    #[must_use]
    pub fn hover(&self, uri: &str, line: usize, col: usize) -> Option<HoverInfo> {
        let text = self.files.get(uri)?;

        // 1. Find the word under the cursor
        let (word, start_col, end_col) = word_at(text, line, col)?;

        // 2. Load the environment to resolve the word
        let temp_path = scratch_file("hover", text)?;
        let mut loader = ModuleLoader::new();
        // A module that fails to load still resolves whatever parsed, so hover proceeds.
        let _ = loader.load_main(&temp_path);

        let modules: Vec<_> = loader.loaded_modules.values().cloned().collect();
        let global_env_modules: Vec<_> = modules.iter().map(Program::clone_signature).collect();
        let mut global_env = GlobalAstEnv::build(&global_env_modules);
        global_env.annotate_return_provenances(&modules);

        // 3. Look up the word in the environment
        let (mut hover_text, doc) = declaration_text(&global_env, &word);

        if let Some(doc_str) = doc {
            let clean_doc = doc_str
                .lines()
                .map(|line| {
                    let s = line.trim_start();
                    if let Some(stripped) = s.strip_prefix("/// ") {
                        stripped
                    } else if let Some(stripped) = s.strip_prefix("///") {
                        stripped
                    } else {
                        s
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");
            hover_text = format!("{clean_doc}\n\n```vx\n{hover_text}\n```");
        } else {
            hover_text = format!("```vx\n{hover_text}\n```");
        }

        if hover_text.is_empty() {
            return None;
        }

        Some(HoverInfo {
            value: hover_text,
            line_start: line,
            col_start: start_col,
            line_end: line,
            col_end: end_col,
        })
    }

    /// Where the word at `line` and `col` of the file `uri` is defined: the first place in
    /// any open file where it follows `fn`, `struct`, `enum`, `let` or `mut`.
    #[must_use]
    pub fn goto_definition(
        &self,
        uri: &str,
        line: usize,
        col: usize,
    ) -> Option<DefinitionLocation> {
        let text = self.files.get(uri)?;

        let (word, _, _) = word_at(text, line, col)?;

        // Simple heuristic: search all files for definition
        for (search_uri, search_text) in &self.files {
            let mut lexer = crate::lexer::Lexer::new(search_text);
            let tokens = lexer.tokenize();

            for i in 0..tokens.len() {
                // If it's `fn`, `struct`, `enum`, `let`, `mut` followed by the word
                match tokens[i].kind {
                    crate::lexer::TokenTypeBase::Fn
                    | crate::lexer::TokenTypeBase::Struct
                    | crate::lexer::TokenTypeBase::Enum
                    | crate::lexer::TokenTypeBase::Let
                    | crate::lexer::TokenTypeBase::Mut
                        if i + 1 < tokens.len() =>
                    {
                        if let crate::lexer::TokenTypeBase::Identifier(id) = tokens[i + 1].kind {
                            if id == word {
                                return Some(DefinitionLocation {
                                    uri: search_uri.clone(),
                                    line_start: tokens[i + 1].line.saturating_sub(1),
                                    col_start: tokens[i + 1].column.saturating_sub(1),
                                    line_end: tokens[i + 1].line.saturating_sub(1),
                                    col_end: tokens[i + 1].column.saturating_sub(1)
                                        + tokens[i + 1].length,
                                });
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        None
    }
}
