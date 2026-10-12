//! The Vx language server: answers hover, go-to-definition and formatting requests from an
//! editor over the Language Server Protocol, and reports type errors as diagnostics.

use lsp_server::{Connection, Message, Notification, Request, RequestId, Response};
use lsp_types::{
    Diagnostic, DiagnosticSeverity, DidChangeTextDocumentParams, DidOpenTextDocumentParams, Hover,
    HoverContents, HoverProviderCapability, Position, PublishDiagnosticsParams, Range,
    ServerCapabilities, TextDocumentSyncCapability, TextDocumentSyncKind, Url,
};
use serde_json::Value;
use std::error::Error;
use vxc::ide::{AnalysisHost, IdeDiagnostic};

/// What every step of the server returns: any error ends the server.
type LspResult<T> = Result<T, Box<dyn Error + Sync + Send>>;

/// The debug log, when `VX_ANALYZER_LOG` named one.
type LogSink = Option<std::fs::File>;

/// Debug log sink, opt-in via `VX_ANALYZER_LOG=<path>`.
///
/// This used to be an unconditional `OpenOptions::open(<an absolute path>).unwrap()`.
/// `create(true)` makes the file, never its parent directory, so on any machine without
/// that directory the server panicked before reading its first message.
fn log_sink() -> LogSink {
    let path = std::env::var_os("VX_ANALYZER_LOG")?;
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .ok()
}

/// Write a line to the sink if there is one. Never fatal: a language server that cannot
/// write its log must still serve.
macro_rules! log_line {
    ($sink:expr, $($arg:tt)*) => {
        if let Some(f) = $sink.as_mut() {
            use std::io::Write;
            let _ = writeln!(f, $($arg)*);
        }
    };
}

/// Turns a line or column count into the 32-bit number the protocol uses. A count too
/// large for 32 bits is clamped: no editor can address such a position anyway.
fn lsp_u32(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

/// Starts the server on stdin and stdout and serves until the editor asks it to shut down.
///
/// # Errors
///
/// Returns an error when the editor's first message is not a valid initialize request, when
/// a message cannot be read or sent, or when the stdin and stdout threads fail.
///
/// # Panics
///
/// Panics if the server's capability list cannot be turned into JSON, which cannot happen
/// for the value built here.
fn main() -> LspResult<()> {
    let mut log_file = log_sink();
    log_line!(log_file, "--- vx-analyzer started (lsp-server) ---");

    let (connection, io_threads) = Connection::stdio();

    let server_capabilities = serde_json::to_value(&ServerCapabilities {
        text_document_sync: Some(TextDocumentSyncCapability::Kind(TextDocumentSyncKind::FULL)),
        hover_provider: Some(HoverProviderCapability::Simple(true)),
        definition_provider: Some(lsp_types::OneOf::Left(true)),
        references_provider: Some(lsp_types::OneOf::Left(true)),
        document_formatting_provider: Some(lsp_types::OneOf::Left(true)),
        ..Default::default()
    })
    .unwrap();
    let _initialization_params = connection.initialize(server_capabilities)?;

    log_line!(log_file, "Initialized");

    main_loop(&connection)?;
    io_threads.join()?;

    log_line!(log_file, "Shut down");
    Ok(())
}

/// Serves every message from the editor until the shutdown request arrives.
///
/// # Errors
///
/// Returns an error when a request carries the wrong parameters for its method, when a
/// notification's parameters cannot be read, or when a reply cannot be sent.
fn main_loop(connection: &Connection) -> LspResult<()> {
    let mut log_file = log_sink();

    let mut host = AnalysisHost::new();

    for msg in &connection.receiver {
        log_line!(log_file, "Received msg: {:?}", msg);
        match msg {
            Message::Request(req) => {
                if connection.handle_shutdown(&req)? {
                    return Ok(());
                }
                log_line!(log_file, "got request: {:?}", req);
                handle_request(connection, &host, req, &mut log_file)?;
            }
            Message::Response(resp) => {
                log_line!(log_file, "got response: {:?}", resp);
            }
            Message::Notification(not) => {
                log_line!(log_file, "got notification: {:?}", not);
                handle_notification(connection, &mut host, not)?;
            }
        }
    }
    Ok(())
}

/// Answers one request from the editor. A method the server does not support gets no reply.
///
/// # Errors
///
/// Returns an error when the request's parameters do not match its method, or when the
/// reply cannot be sent.
///
/// # Panics
///
/// Panics if a reply cannot be turned into JSON, which cannot happen for the protocol
/// types built here.
fn handle_request(
    connection: &Connection,
    host: &AnalysisHost,
    req: Request,
    log_file: &mut LogSink,
) -> LspResult<()> {
    match req.method.as_str() {
        "textDocument/hover" => {
            let (id, params) = cast::<lsp_types::request::HoverRequest>(req)?;
            let result = serde_json::to_value(hover(host, &params)).unwrap();
            let resp = Response {
                id,
                result: Some(result),
                error: None,
            };
            log_line!(log_file, "sending hover response: {:?}", resp);
            connection.sender.send(Message::Response(resp))?;
        }
        "textDocument/definition" => {
            let (id, params) = cast::<lsp_types::request::GotoDefinition>(req)?;
            let result = serde_json::to_value(definition(host, &params)).unwrap();
            let resp = Response {
                id,
                result: Some(result),
                error: None,
            };
            connection.sender.send(Message::Response(resp))?;
        }
        "textDocument/formatting" => {
            let (id, params) = cast::<lsp_types::request::Formatting>(req)?;
            let result = match formatting(host, &params) {
                Some(edit) => serde_json::to_value(vec![edit]).unwrap(),
                None => Value::Null,
            };
            let resp = Response {
                id,
                result: Some(result),
                error: None,
            };
            connection.sender.send(Message::Response(resp))?;
        }
        _ => {}
    }
    Ok(())
}

/// The hover text for the position in `params`, or `None` when nothing is under the cursor.
fn hover(host: &AnalysisHost, params: &lsp_types::HoverParams) -> Option<Hover> {
    let analysis = host.snapshot();
    let position = &params.text_document_position_params;
    let uri = position.text_document.uri.as_str();
    let line = position.position.line as usize;
    let character = position.position.character as usize;

    let hover = analysis.hover(uri, line, character)?;
    Some(Hover {
        contents: HoverContents::Markup(lsp_types::MarkupContent {
            kind: lsp_types::MarkupKind::Markdown,
            value: hover.value,
        }),
        range: Some(Range {
            start: Position::new(lsp_u32(hover.line_start), lsp_u32(hover.col_start)),
            end: Position::new(lsp_u32(hover.line_end), lsp_u32(hover.col_end)),
        }),
    })
}

/// Where the name under the cursor in `params` is defined, or `None` when there is none.
fn definition(
    host: &AnalysisHost,
    params: &lsp_types::GotoDefinitionParams,
) -> Option<lsp_types::GotoDefinitionResponse> {
    let analysis = host.snapshot();
    let position = &params.text_document_position_params;
    let uri = position.text_document.uri.as_str();
    let line = position.position.line as usize;
    let character = position.position.character as usize;

    let def_loc = analysis.goto_definition(uri, line, character)?;
    let loc = lsp_types::Location {
        uri: Url::parse(&def_loc.uri).unwrap_or_else(|_| position.text_document.uri.clone()),
        range: Range {
            start: Position::new(lsp_u32(def_loc.line_start), lsp_u32(def_loc.col_start)),
            end: Position::new(lsp_u32(def_loc.line_end), lsp_u32(def_loc.col_end)),
        },
    };
    Some(lsp_types::GotoDefinitionResponse::Scalar(loc))
}

/// One edit that replaces the whole file with its formatted text, or `None` when the server
/// has not seen the file.
fn formatting(
    host: &AnalysisHost,
    params: &lsp_types::DocumentFormattingParams,
) -> Option<lsp_types::TextEdit> {
    let analysis = host.snapshot();
    let text = analysis.files.get(params.text_document.uri.as_str())?;
    let formatted = vxc::formatter::format_file(text, params.options.tab_size as usize);
    let lines: Vec<&str> = text.lines().collect();
    let last_line = lsp_u32(lines.len().saturating_sub(1));
    let last_col = lines.last().map_or(0, |s| lsp_u32(s.len()));

    Some(lsp_types::TextEdit {
        range: Range {
            start: Position::new(0, 0),
            end: Position::new(last_line, last_col),
        },
        new_text: formatted,
    })
}

/// Applies a notification from the editor. An opened or changed file is checked again and
/// its diagnostics sent back; other notifications are ignored.
///
/// # Errors
///
/// Returns an error when the notification's parameters cannot be read, or when the
/// diagnostics cannot be sent.
fn handle_notification(
    connection: &Connection,
    host: &mut AnalysisHost,
    not: Notification,
) -> LspResult<()> {
    match not.method.as_str() {
        "textDocument/didOpen" => {
            let params: DidOpenTextDocumentParams = serde_json::from_value(not.params)?;
            let uri = params.text_document.uri.as_str().to_string();
            let text = params.text_document.text;
            host.apply_change(uri.clone(), text);

            let analysis = host.snapshot();
            let diagnostics = analysis.diagnostics(&uri);
            send_diagnostics(connection, params.text_document.uri, diagnostics)?;
        }
        "textDocument/didChange" => {
            let mut params: DidChangeTextDocumentParams = serde_json::from_value(not.params)?;
            let uri = params.text_document.uri.as_str().to_string();
            if let Some(change) = params.content_changes.pop() {
                let text = change.text;
                host.apply_change(uri.clone(), text);

                let analysis = host.snapshot();
                let diagnostics = analysis.diagnostics(&uri);
                send_diagnostics(connection, params.text_document.uri, diagnostics)?;
            }
        }
        _ => {}
    }
    Ok(())
}

/// Splits a request into its id and its typed parameters.
///
/// # Errors
///
/// Returns an error when the request is for another method, or when its parameters do not
/// have the shape `R` expects.
fn cast<R>(req: Request) -> LspResult<(RequestId, R::Params)>
where
    R: lsp_types::request::Request,
    R::Params: serde::de::DeserializeOwned,
{
    req.extract(R::METHOD)
        .map_err(|e| format!("cast failed: {e:?}").into())
}

/// Sends the editor the diagnostics for one file, replacing the ones it showed before.
///
/// # Errors
///
/// Returns an error when the notification cannot be sent to the editor.
fn send_diagnostics(
    connection: &Connection,
    uri: Url,
    diagnostics: Vec<IdeDiagnostic>,
) -> LspResult<()> {
    let diags: Vec<Diagnostic> = diagnostics
        .into_iter()
        .map(|d| Diagnostic {
            range: Range {
                start: Position::new(lsp_u32(d.line_start), lsp_u32(d.col_start)),
                end: Position::new(lsp_u32(d.line_end), lsp_u32(d.col_end)),
            },
            severity: Some(DiagnosticSeverity::ERROR),
            code: None,
            code_description: None,
            source: Some("vxc".to_string()),
            message: d.message,
            related_information: None,
            tags: None,
            data: None,
        })
        .collect();

    let params = PublishDiagnosticsParams {
        uri,
        diagnostics: diags,
        version: None,
    };

    let not = Notification::new("textDocument/publishDiagnostics".to_string(), params);
    connection.sender.send(Message::Notification(not))?;
    Ok(())
}
