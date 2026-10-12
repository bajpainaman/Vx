//! A scratch program that loads one Vx file into the IDE analysis host and prints the hover
//! and go-to-definition answers for a fixed position.
use vxc::ide::AnalysisHost;

/// Reads the file named on the command line and prints two IDE answers for it.
///
/// # Panics
///
/// Panics when the file cannot be read.
fn main() {
    let mut host = AnalysisHost::new();
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: test_analysis <file.vx> [line] [col]");
        std::process::exit(2);
    });
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    let uri = format!(
        "file://{}",
        std::fs::canonicalize(&path)
            .unwrap_or_else(|_| path.clone().into())
            .display()
    );
    let uri = uri.as_str();

    host.apply_change(uri.to_string(), text);
    let analysis = host.snapshot();
    let hover = analysis.hover(uri, 14, 48);
    println!("Hover result 48: {hover:?}");

    let def = analysis.goto_definition(uri, 14, 48);
    println!("Definition 48: {def:?}");
}
