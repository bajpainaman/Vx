//! LLVM IR for a Solana program, for `--emit-sbf`.
//!
//! Solana runs SBF code. The LLVM that vxc builds with has no SBF target, but the Solana tools
//! carry a build of LLVM that has one. So vxc writes LLVM IR, and the Solana `clang` and `ld.lld`
//! make the program file from it, as the Metal compiler does for `--emit-msl`.
//!
//! A Solana program has one entry point. This module writes that function. It reads which
//! `pub fn` the caller wants from the first 8 bytes of the instruction data, reads the arguments
//! that follow, calls the function, and gives the result back as the return data.
//!
//! The layout is the one that Anchor programs use, so that a client for an Anchor program can
//! call a Vx program:
//! - The first 8 bytes are the first 8 bytes of the SHA-256 of `global:<name>`.
//! - Each argument follows in the order of the parameters, as little-endian bytes with no
//!   padding. A `bool` is one byte.
//! - The result is the return data, in the same form.
//!
//! A call with other instruction data, or with an account, ends with the error "invalid
//! instruction data". A failed `assert` or division check writes its message to the log of the
//! program and stops the program.

use std::fmt::Write as _;

/// The data layout and the target that the Solana `clang` gives a module for `-target sbf`.
pub const DATA_LAYOUT: &str = "e-m:e-p:64:64-i64:64-n32:64-S128";
pub const TARGET_TRIPLE: &str = "sbf";

/// `ProgramError::InvalidInstructionData`, as a Solana program returns it.
const INVALID_INSTRUCTION_DATA: u64 = 3 << 32;

/// The function that writes the message of a failed check to the log of the program. A program
/// cannot have a function of its own with one of these names.
const LOG_FUNCTION: &str = "vx_sbf_log";
const RESERVED: [&str; 5] = [
    "entrypoint",
    "abort",
    LOG_FUNCTION,
    "sol_log_",
    "sol_set_return_data",
];

/// One `pub fn`: its name, and the Vx types of its parameters and result as the source spells
/// them. `result` is `None` for a function that returns nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublicFunction {
    pub name: String,
    pub params: Vec<String>,
    pub result: Option<String>,
}

/// Reads the lines that the driver writes for the public functions, each as
/// `name(u64,i64)->u64`. `void` is a function that returns nothing.
pub fn parse_public_functions(lines: &str) -> Result<Vec<PublicFunction>, String> {
    let mut functions = Vec::new();
    for line in lines.lines().map(str::trim).filter(|l| !l.is_empty()) {
        let malformed = || format!("'{line}' is not `name(types)->type`");
        let (name, rest) = line.split_once('(').ok_or_else(malformed)?;
        let (params, result) = rest.split_once(")->").ok_or_else(malformed)?;
        functions.push(PublicFunction {
            name: name.to_string(),
            params: params
                .split(',')
                .map(str::trim)
                .filter(|p| !p.is_empty())
                .map(str::to_string)
                .collect(),
            result: (result != "void").then(|| result.to_string()),
        });
    }
    Ok(functions)
}

/// The LLVM type that holds a Vx scalar type, and its size in bytes in the instruction data.
/// `None` for a type that the entry point cannot read or return yet.
fn wire(ty: &str) -> Option<(&'static str, u64)> {
    Some(match ty {
        "i8" | "u8" => ("i8", 1),
        "i16" | "u16" => ("i16", 2),
        "i32" | "u32" => ("i32", 4),
        "i64" | "u64" | "usize" => ("i64", 8),
        "bool" => ("i1", 1),
        _ => return None,
    })
}

/// The first 8 bytes of the instruction data of a call to the public function `name`.
pub fn discriminator(name: &str) -> [u8; 8] {
    let digest = sha256(format!("global:{name}").as_bytes());
    let mut first = [0u8; 8];
    first.copy_from_slice(&digest[..8]);
    first
}

/// SHA-256 (FIPS 180-4). The compiler needs one hash for the entry point, and has no other use
/// for a hashing crate.
fn sha256(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut state: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut message = data.to_vec();
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&((data.len() as u64) * 8).to_be_bytes());

    for block in message.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choice = (e & f) ^ (!e & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(choice)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (slot, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *slot = slot.wrapping_add(value);
        }
    }

    let mut digest = [0u8; 32];
    for (bytes, word) in digest.chunks_exact_mut(4).zip(state) {
        bytes.copy_from_slice(&word.to_be_bytes());
    }
    digest
}

/// The LLVM IR of the entry point of a program with the public functions `functions`, and of the
/// function that writes a message to the log.
fn entry_point(functions: &[PublicFunction]) -> Result<String, String> {
    let mut ir = String::new();
    // `write!` to a `String` cannot fail.
    let _ = writeln!(
        ir,
        "\n; The entry point of the Solana program. vxc wrote it from the `pub` functions.\n\
         declare void @sol_set_return_data(ptr, i64)\n\
         declare void @sol_log_(ptr, i64)\n\n\
         define internal void @{LOG_FUNCTION}(ptr %message) {{\n\
         entry:\n  br label %scan\n\
         scan:\n\
         \x20 %length = phi i64 [ 0, %entry ], [ %next, %scan ]\n\
         \x20 %at = getelementptr i8, ptr %message, i64 %length\n\
         \x20 %byte = load i8, ptr %at, align 1\n\
         \x20 %next = add i64 %length, 1\n\
         \x20 %end = icmp eq i8 %byte, 0\n\
         \x20 br i1 %end, label %done, label %scan\n\
         done:\n\
         \x20 call void @sol_log_(ptr %message, i64 %length)\n\
         \x20 ret void\n\
         }}\n"
    );
    let _ = writeln!(
        ir,
        "define i64 @entrypoint(ptr %input) {{\n\
         entry:\n\
         \x20 %out = alloca [8 x i8], align 8\n\
         \x20 %accounts = load i64, ptr %input, align 1\n\
         \x20 %no_accounts = icmp eq i64 %accounts, 0\n\
         \x20 br i1 %no_accounts, label %header, label %bad\n\
         header:\n\
         \x20 %length_at = getelementptr i8, ptr %input, i64 8\n\
         \x20 %length = load i64, ptr %length_at, align 1\n\
         \x20 %data = getelementptr i8, ptr %input, i64 16\n\
         \x20 %has_name = icmp uge i64 %length, 8\n\
         \x20 br i1 %has_name, label %pick, label %bad\n\
         pick:\n\
         \x20 %name = load i64, ptr %data, align 1\n\
         \x20 switch i64 %name, label %bad ["
    );
    let mut seen = std::collections::HashMap::new();
    for (index, function) in functions.iter().enumerate() {
        let first = discriminator(&function.name);
        if let Some(other) = seen.insert(first, &function.name) {
            return Err(format!(
                "Public functions '{other}' and '{}' start with the same 8 bytes of instruction \
                 data. Give one of them another name.",
                function.name
            ));
        }
        let _ = writeln!(ir, "    i64 {}, label %f{index}", i64::from_le_bytes(first));
    }
    ir.push_str("  ]\n");

    for (index, function) in functions.iter().enumerate() {
        let name = &function.name;
        let refuse = |what: &str, ty: &str| {
            format!(
                "Cannot write public function '{name}' for Solana. Its {what} has type `{ty}`, \
                 and the entry point takes only integers of up to 64 bits and `bool`."
            )
        };
        let mut offset = 8u64;
        let mut loads = String::new();
        let mut arguments = Vec::new();
        for (position, ty) in function.params.iter().enumerate() {
            let (llvm, size) = wire(ty).ok_or_else(|| refuse("parameter", ty))?;
            let at = format!("%f{index}.at{position}");
            let value = format!("%f{index}.a{position}");
            let _ = writeln!(loads, "  {at} = getelementptr i8, ptr %data, i64 {offset}");
            if llvm == "i1" {
                let _ = writeln!(loads, "  {value}.byte = load i8, ptr {at}, align 1");
                let _ = writeln!(loads, "  {value} = icmp ne i8 {value}.byte, 0");
            } else {
                let _ = writeln!(loads, "  {value} = load {llvm}, ptr {at}, align 1");
            }
            arguments.push(format!("{llvm} {value}"));
            offset += size;
        }
        let _ = writeln!(
            ir,
            "f{index}:\n\
             \x20 %f{index}.fits = icmp eq i64 %length, {offset}\n\
             \x20 br i1 %f{index}.fits, label %f{index}.call, label %bad\n\
             f{index}.call:"
        );
        ir.push_str(&loads);
        let call = format!("@{name}({})", arguments.join(", "));
        match &function.result {
            None => {
                let _ = writeln!(ir, "  call void {call}");
            }
            Some(ty) => {
                let (llvm, size) = wire(ty).ok_or_else(|| refuse("result", ty))?;
                let _ = writeln!(ir, "  %f{index}.r = call {llvm} {call}");
                if llvm == "i1" {
                    let _ = writeln!(ir, "  %f{index}.byte = zext i1 %f{index}.r to i8");
                    let _ = writeln!(ir, "  store i8 %f{index}.byte, ptr %out, align 1");
                } else {
                    let _ = writeln!(ir, "  store {llvm} %f{index}.r, ptr %out, align 1");
                }
                let _ = writeln!(ir, "  call void @sol_set_return_data(ptr %out, i64 {size})");
            }
        }
        ir.push_str("  ret i64 0\n");
    }
    let _ = writeln!(ir, "bad:\n  ret i64 {INVALID_INSTRUCTION_DATA}\n}}");
    Ok(ir)
}

/// The LLVM IR of the Solana program: the IR that `mlir-translate` made from the module, and
/// the entry point for `functions`.
///
/// Only the entry point is visible outside the program, so every function of the module becomes
/// internal. A failed check calls `puts` and then `abort`. Solana has `abort` and has no `puts`,
/// so a call to `puts` becomes a call to the function that writes to the log.
pub fn program_ir(module_ir: &str, functions: &[PublicFunction]) -> Result<String, String> {
    if functions.is_empty() {
        return Err(
            "The program has no `pub fn`. `--emit-sbf` writes one entry for each `pub fn`."
                .to_string(),
        );
    }
    for reserved in RESERVED {
        if module_ir.contains(&format!("define i64 @{reserved}("))
            || module_ir.contains(&format!("define void @{reserved}("))
            || functions.iter().any(|f| f.name == reserved)
        {
            return Err(format!(
                "The program has a function '{reserved}'. A Solana program that vxc writes has \
                 a function of that name already. Give the function another name."
            ));
        }
    }

    let mut ir = String::new();
    for line in module_ir.lines() {
        if line.starts_with("declare ") && line.contains("@puts(") {
            continue;
        }
        let line = line.replace("@puts(", &format!("@{LOG_FUNCTION}("));
        match line.strip_prefix("define ") {
            Some(rest) => {
                ir.push_str("define internal ");
                ir.push_str(rest);
            }
            None => ir.push_str(&line),
        }
        ir.push('\n');
    }
    ir.push_str(&entry_point(functions)?);
    Ok(ir)
}

/// The LLVM IR that `mlir-translate` makes from `mlir`, a module in the LLVM dialect. The tool is
/// the one that `run-jit` uses, and `MLIR_TRANSLATE_PATH` names another one in the same way.
pub fn translate(mlir: &str) -> Result<String, String> {
    let directory = tempfile::tempdir().map_err(|e| e.to_string())?;
    let input = directory.path().join("input.mlir");
    std::fs::write(&input, mlir)
        .map_err(|e| format!("could not write '{}': {e}", input.display()))?;
    let tool =
        std::env::var("MLIR_TRANSLATE_PATH").unwrap_or_else(|_| "mlir-translate".to_string());
    let output = std::process::Command::new(&tool)
        .arg("--mlir-to-llvmir")
        .arg(&input)
        .output()
        .map_err(|e| format!("could not run {tool}: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "{tool} failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    String::from_utf8(output.stdout).map_err(|e| e.to_string())
}

/// What a client needs to call the program: for each public function, the first 8 bytes of the
/// instruction data, and the types of its parameters and result. JSON.
pub fn manifest(functions: &[PublicFunction]) -> String {
    let mut text = String::from("{\n  \"target\": \"sbf\",\n  \"functions\": [\n");
    for (index, function) in functions.iter().enumerate() {
        let first: Vec<String> = discriminator(&function.name)
            .iter()
            .map(|b| b.to_string())
            .collect();
        let params: Vec<String> = function.params.iter().map(|p| format!("\"{p}\"")).collect();
        let result = match &function.result {
            Some(ty) => format!("\"{ty}\""),
            None => "null".to_string(),
        };
        let _ = write!(
            text,
            "    {{\"name\": \"{}\", \"discriminator\": [{}], \"params\": [{}], \"returns\": {}}}",
            function.name,
            first.join(", "),
            params.join(", "),
            result
        );
        text.push_str(if index + 1 == functions.len() {
            "\n"
        } else {
            ",\n"
        });
    }
    text.push_str("  ]\n}\n");
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn sha256_gives_the_digests_of_the_standard() {
        assert_eq!(
            hex(&sha256(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex(&sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        // 56 bytes: the length does not fit the first block, so the padding takes a second one.
        assert_eq!(
            hex(&sha256(
                b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
            )),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn the_lines_of_the_driver_give_the_public_functions() {
        let functions = parse_public_functions("add(i64,i64)->i64\nreset()->void\n").unwrap();
        assert_eq!(
            functions,
            vec![
                PublicFunction {
                    name: "add".to_string(),
                    params: vec!["i64".to_string(), "i64".to_string()],
                    result: Some("i64".to_string()),
                },
                PublicFunction {
                    name: "reset".to_string(),
                    params: vec![],
                    result: None,
                },
            ]
        );
        assert!(parse_public_functions("add i64").is_err());
    }

    #[test]
    fn the_entry_point_reads_each_argument_after_the_one_before() {
        let functions = parse_public_functions("mix(i32,bool,u64)->bool").unwrap();
        let ir = program_ir("define i1 @mix(i32 %0, i1 %1, i64 %2) {\n}\n", &functions).unwrap();
        // 8 bytes of name, then 4 + 1 + 8 bytes of arguments.
        assert!(ir.contains("%f0.fits = icmp eq i64 %length, 21"));
        assert!(ir.contains("%f0.at1 = getelementptr i8, ptr %data, i64 12"));
        assert!(ir.contains("%f0.at2 = getelementptr i8, ptr %data, i64 13"));
        assert!(ir.contains("call void @sol_set_return_data(ptr %out, i64 1)"));
        assert!(ir.contains("define internal i1 @mix("));
    }

    #[test]
    fn a_wide_parameter_and_a_reserved_name_are_refused() {
        let wide = parse_public_functions("big(i128)->i64").unwrap();
        assert!(program_ir("", &wide).unwrap_err().contains("`i128`"));
        let reserved = parse_public_functions("entrypoint()->void").unwrap();
        assert!(program_ir("", &reserved)
            .unwrap_err()
            .contains("'entrypoint'"));
        assert!(program_ir("", &[]).unwrap_err().contains("no `pub fn`"));
    }
}
