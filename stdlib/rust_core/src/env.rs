use std::ffi::CString;

/// The index of the first argument that belongs to the Vx program.
///
/// When executing via the JIT engine (lli), the process name is `lli` and the script
/// is passed as a `.ll` file. We filter these out.
/// When running as a standalone compiled executable, `args[0]` is the binary itself.
#[allow(
    clippy::case_sensitive_file_extension_comparisons,
    reason = "the compiler always names the file it gives to lli with a lowercase `.ll`"
)]
fn first_program_arg(args: &[String]) -> usize {
    args.iter()
        .position(|arg| arg.ends_with(".ll"))
        .map_or(0, |i| i + 1)
}

/// The number of command line arguments the Vx program can read.
#[no_mangle]
pub extern "C" fn vx_env_args_count() -> i32 {
    let args: Vec<String> = std::env::args().collect();
    let start_idx = first_program_arg(&args);

    i32::try_from(args.len().saturating_sub(start_idx)).unwrap_or(i32::MAX)
}

/// The argument at `idx`, as a C string the caller may keep. `c_char`, not `i8`: that is what
/// `CString::into_raw` returns, and it is `u8` on aarch64 Linux. Returns null when there is
/// no argument at `idx`.
///
/// # Panics
/// Panics if the argument has a NUL byte in it, which the operating system does not allow.
#[no_mangle]
pub extern "C" fn vx_env_arg(idx: i32) -> *mut std::ffi::c_char {
    let args: Vec<String> = std::env::args().collect();
    let start_idx = first_program_arg(&args);

    let Ok(offset) = usize::try_from(idx) else {
        return std::ptr::null_mut();
    };
    let actual_idx = start_idx + offset;
    if actual_idx >= args.len() {
        return std::ptr::null_mut();
    }

    // We intentionally leak the memory here so that the C caller can access the string
    let c_str = CString::new(args[actual_idx].clone()).unwrap();
    c_str.into_raw()
}
