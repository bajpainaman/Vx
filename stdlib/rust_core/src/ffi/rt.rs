//===- rt.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the core runtime initialization routines for Vx.
// It manages standard library state, threading infrastructure, and any required
// setup or teardown steps that must occur before and after the execution of
// a compiled Vx program.
//
//===----------------------------------------------------------------------===//

use std::ffi::{c_char, CStr};
use std::ptr;
use std::sync::{Mutex, OnceLock, PoisonError};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
// ============================================================================
// Math & Core Functions
// ============================================================================

/// The handler for a bad memory access: prints which signal arrived and a backtrace,
/// then stops the program.
#[no_mangle]
pub extern "C" fn vx_sigsegv_handler(sig: libc::c_int) {
    // A wild memory access can surface as either SIGSEGV (unmapped page) or
    // SIGBUS (e.g. a far out-of-range address on arm64); catch both so the crash
    // is always reported with a backtrace rather than a silent signal death.
    if sig == libc::SIGBUS {
        println!("Caught SIGBUS: Bus Error!");
    } else {
        println!("Caught SIGSEGV: Segmentation Fault!");
    }
    println!(
        "Backtrace:
{:#?}",
        std::backtrace::Backtrace::force_capture()
    );
    std::process::abort();
}

/// A failed check prints its message with `puts` and then calls `abort`. Standard output is
/// buffered when it is a pipe, and `abort` does not flush it, so the message was lost. Flush,
/// then let the abort go on.
extern "C" fn vx_sigabrt_handler(_sig: libc::c_int) {
    // SAFETY: these are plain libc calls with valid arguments: a null stream flushes
    // every open stream, and `SIG_DFL` is a valid handler for `SIGABRT`.
    unsafe {
        libc::fflush(ptr::null_mut());
        libc::signal(libc::SIGABRT, libc::SIG_DFL);
        libc::raise(libc::SIGABRT);
    }
}

/// Installs the handlers that report an abort or a bad memory access before the
/// program stops.
#[no_mangle]
pub extern "C" fn vx_init_signals() {
    // SAFETY: both handlers are `extern "C"` functions that take the signal number,
    // which is the shape `signal` expects.
    unsafe {
        libc::signal(
            libc::SIGABRT,
            vx_sigabrt_handler as *const () as libc::sighandler_t,
        );
        libc::signal(
            libc::SIGSEGV,
            vx_sigsegv_handler as *const () as libc::sighandler_t,
        );
        libc::signal(
            libc::SIGBUS,
            vx_sigsegv_handler as *const () as libc::sighandler_t,
        );
    }
}

/// The square root of `x`.
#[no_mangle]
pub extern "C" fn vx_sqrtf(x: f32) -> f32 {
    x.sqrt()
}

/// The number e raised to the power `x`.
#[no_mangle]
pub extern "C" fn vx_expf(x: f32) -> f32 {
    x.exp()
}

/// The cosine of `x`, where `x` is in radians.
#[no_mangle]
pub extern "C" fn vx_cosf(x: f32) -> f32 {
    x.cos()
}

/// The sine of `x`, where `x` is in radians.
#[no_mangle]
pub extern "C" fn vx_sinf(x: f32) -> f32 {
    x.sin()
}

/// The rotation angle that rotary position embedding (`RoPE`) uses for token position
/// `pos` and pair number `i`, in a head with `head_size` values.
#[no_mangle]
#[allow(
    clippy::cast_precision_loss,
    reason = "positions and head sizes are small numbers, and a rounded angle is fine for a larger one"
)]
pub extern "C" fn vx_get_rope_freq(pos: i32, i: i32, head_size: i32) -> f32 {
    let freq = 10000.0f32.powf(-(i as f32) / (head_size as f32));
    (pos as f32) * freq
}

// ============================================================================
// Timing & Benchmarking
// ============================================================================

/// When `start_benchmark` was last called.
static BENCHMARK_START: Mutex<Option<Instant>> = Mutex::new(None);
/// When `vx_get_time` was first called.
static GLOBAL_START: OnceLock<Instant> = OnceLock::new();

/// Starts the benchmark timer, and returns 0.
#[no_mangle]
pub extern "C" fn start_benchmark() -> f32 {
    *BENCHMARK_START
        .lock()
        .unwrap_or_else(PoisonError::into_inner) = Some(Instant::now());
    0.0
}

/// Prints the seconds since `start_benchmark` was called. Prints nothing when it was
/// never called.
#[no_mangle]
pub extern "C" fn end_benchmark() {
    let start = *BENCHMARK_START
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    if let Some(start) = start {
        let elapsed = start.elapsed().as_secs_f32();
        println!("{elapsed}");
    }
}

/// Report one benchmark measurement as a single machine-readable line.
///
/// Printed from Rust because `println!` in Vx emits its format string literally and appends the
/// arguments after it (Vx#450), so a Vx-side `println!("{} {}", name, value)` cannot produce a
/// parseable record. The line is what `cargo vx-bench` reads and what CI gates the presence of;
/// the runner adds the commit and the machine, which a program cannot know about itself.
///
/// Shape: `vx-bench <name> <unit> <value>` -- one space-separated record, no interpolation, so it
/// survives being mixed into JIT chatter on stdout. `name` and `unit` must not contain spaces.
#[no_mangle]
pub extern "C" fn vx_bench_report(name: *const c_char, unit: *const c_char, value: f32) -> i32 {
    if name.is_null() || unit.is_null() {
        eprintln!("vx_bench_report: null name or unit");
        return 1;
    }
    // SAFETY: the caller passes a NUL-terminated C string that stays alive for this call.
    let name = unsafe { CStr::from_ptr(name) }.to_string_lossy();
    // SAFETY: the caller passes a NUL-terminated C string that stays alive for this call.
    let unit = unsafe { CStr::from_ptr(unit) }.to_string_lossy();
    if name.contains(char::is_whitespace) || unit.contains(char::is_whitespace) {
        eprintln!("vx_bench_report: name and unit must not contain whitespace");
        return 1;
    }
    println!("vx-bench {name} {unit} {value}");
    0
}

/// The seconds since the first call to this function, so the first call returns about 0.
#[no_mangle]
pub extern "C" fn vx_get_time() -> f32 {
    GLOBAL_START
        .get_or_init(Instant::now)
        .elapsed()
        .as_secs_f32()
}

/// Pauses the calling thread for `seconds`, and returns 0.
///
/// # Panics
/// Panics if `seconds` is negative, not a number, or too large for a `Duration`.
#[no_mangle]
pub extern "C" fn vx_sleep(seconds: f32) -> i32 {
    thread::sleep(Duration::from_secs_f32(seconds));
    0
}

/// The seconds since 1 January 1970, or 0 when the system clock is set before that day.
#[no_mangle]
pub extern "C" fn vx_unix_timestamp() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_secs_f64()
}
/// Prints the line that marks the start of a traced region.
#[no_mangle]
pub extern "C" fn trace_start() {
    println!("[TRACE START] Event ID: 100");
}

/// Prints the line that marks the end of a traced region.
#[no_mangle]
pub extern "C" fn trace_end() {
    println!("[TRACE END] Event ID: 100");
}

// ============================================================================
// Memory & Pointers
// ============================================================================

/// A zeroed buffer of `n` f32s, for code that then hands the pointer to
/// something expecting foreign memory -- a tensor view over it, say (#336).
///
/// The runtime already produces such buffers (`vx_load_weights` mmaps a model
/// and returns `*mut f32`); this is the same thing without a file behind it, so
/// the pointer-facing paths can be exercised on their own.
#[no_mangle]
pub extern "C" fn vx_alloc_f32(n: i32) -> *mut f32 {
    let len = match usize::try_from(n) {
        Ok(len) if len > 0 => len,
        _ => return ptr::null_mut(),
    };
    Box::into_raw(vec![0.0f32; len].into_boxed_slice()).cast::<f32>()
}

/// Free a buffer from `vx_alloc_f32`. The length is required because the
/// allocation is a boxed slice, and reconstructing one needs the length it was
/// created with.
#[no_mangle]
pub extern "C" fn vx_free_f32(p: *mut f32, n: i32) {
    let len = match usize::try_from(n) {
        Ok(len) if len > 0 && !p.is_null() => len,
        _ => return,
    };
    // SAFETY: the caller passes a pointer from `vx_alloc_f32` and the same `n`, so this
    // rebuilds the boxed slice that was leaked there.
    unsafe {
        drop(Box::from_raw(ptr::slice_from_raw_parts_mut(p, len)));
    }
}

/// The address `offset` elements after `p`, or before it when `offset` is negative.
/// The result must stay inside the buffer that `p` points into.
#[no_mangle]
pub extern "C" fn vx_advance_ptr(p: *mut f32, offset: i32) -> *mut f32 {
    // SAFETY: the caller promises the result stays inside the same buffer as `p`.
    unsafe { p.offset(offset as isize) }
}

/// The same as `vx_advance_ptr`, for a pointer to read-only memory.
#[no_mangle]
pub extern "C" fn vx_advance_ptr_const(p: *const f32, offset: i32) -> *const f32 {
    // SAFETY: the caller promises the result stays inside the same buffer as `p`.
    unsafe { p.offset(offset as isize) }
}

/// The same as `vx_advance_ptr`. Both names are in use by compiled programs.
#[no_mangle]
pub extern "C" fn vx_advance_ptr_f32(p: *mut f32, offset: i32) -> *mut f32 {
    // SAFETY: the caller promises the result stays inside the same buffer as `p`.
    unsafe { p.offset(offset as isize) }
}

/// Copies `num_bytes` bytes from `src` to `dest`, and returns 0. The two buffers must
/// not overlap and each must be at least `num_bytes` long.
///
/// # Panics
/// Panics if `num_bytes` is negative.
#[no_mangle]
pub extern "C" fn vx_memcpy(dest: *mut f32, src: *const f32, num_bytes: i32) -> i32 {
    let len = usize::try_from(num_bytes).expect("vx_memcpy: negative byte count");
    // SAFETY: the caller promises both buffers hold `num_bytes` bytes and do not overlap.
    unsafe {
        ptr::copy_nonoverlapping(src.cast::<u8>(), dest.cast::<u8>(), len);
    }
    0
}

/// Prints a message and panics, so that a Vx program can test `vx_catch_unwind`.
///
/// # Panics
/// Always panics. That is its job.
#[no_mangle]
pub extern "C-unwind" fn vx_panic() -> i32 {
    println!("Vx runtime panic occurred!");
    panic!("vx_panic");
}

/// Calls `f` and returns its result. When `f` panics, prints a message and returns 1.
#[no_mangle]
pub extern "C-unwind" fn vx_catch_unwind(f: extern "C-unwind" fn() -> i32) -> i32 {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f()));
    if let Ok(v) = result {
        v
    } else {
        println!("Caught panic!");
        1
    }
}

// ============================================================================
// Pseudo-random numbers
// ============================================================================
//
// SplitMix64, which is a multiply-xor-shift over a 64-bit counter. Cheap on
// purpose: this exists so a test can fill a matrix with values that are not all
// zero, and a zero-filled matrix hides indexing and transposition errors that
// varied input catches immediately.
//
// Deterministic and seedable, which matters more here than statistical quality:
// a differential test compares two execution paths on the *same* inputs, so the
// sequence has to repeat exactly. `vx_rand_seed` sets it; the default seed is
// fixed rather than taken from the clock, so a run is reproducible without
// asking for it.
//
// Not cryptographic, and not a substitute for a real generator if distribution
// quality ever matters. SplitMix64 passes BigCrush, which is far more than
// filling test matrices needs.

/// The generator's state. An atomic because a Vx program may fill buffers from
/// more than one thread, and a torn read here would be a data race rather than
/// merely a worse sequence.
static VX_RNG_STATE: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0x9E37_79B9_7F4A_7C15);

/// One `SplitMix64` step: advance the counter, then scramble the value taken.
fn vx_next_u64() -> u64 {
    use std::sync::atomic::Ordering;
    let z = VX_RNG_STATE
        .fetch_add(0x9E37_79B9_7F4A_7C15, Ordering::Relaxed)
        .wrapping_add(0x9E37_79B9_7F4A_7C15);
    let z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Set the seed, so a differential run sees the same sequence twice.
#[no_mangle]
pub extern "C" fn vx_rand_seed(seed: u64) {
    VX_RNG_STATE.store(seed, std::sync::atomic::Ordering::Relaxed);
}

/// A uniform `u64` over the whole range.
#[no_mangle]
pub extern "C" fn vx_rand_u64() -> u64 {
    vx_next_u64()
}

/// A uniform `i32` over the whole range, negatives included.
#[no_mangle]
pub extern "C" fn vx_rand_i32() -> i32 {
    ((vx_next_u64() >> 32) as u32).cast_signed()
}

/// A uniform `f32` in [0, 1).
///
/// Built from the top 24 bits rather than by dividing the integer, because f32
/// has 24 bits of mantissa: taking more would round and could return exactly
/// 1.0, which a caller scaling into a half-open range does not expect.
#[no_mangle]
#[allow(
    clippy::cast_precision_loss,
    reason = "the value has 24 bits, which an f32 holds exactly"
)]
pub extern "C" fn vx_rand_f32() -> f32 {
    ((vx_next_u64() >> 40) as f32) * (1.0 / 16_777_216.0)
}

/// A uniform `f64` in [0, 1), on the same terms with 53 bits.
#[no_mangle]
#[allow(
    clippy::cast_precision_loss,
    reason = "the value has 53 bits, which an f64 holds exactly"
)]
pub extern "C" fn vx_rand_f64() -> f64 {
    ((vx_next_u64() >> 11) as f64) * (1.0 / 9_007_199_254_740_992.0)
}

/// A uniform `f32` in [lo, hi), which is what filling a test matrix wants.
#[no_mangle]
pub extern "C" fn vx_rand_range_f32(lo: f32, hi: f32) -> f32 {
    lo + (hi - lo) * vx_rand_f32()
}
