//===- macros.rs - Vx Compiler ---------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides macros used to simplify FFI definitions.
// It contains helper macros that automatically generate the boilerplate required
// to safely expose Rust functions to the Vx compiler's MLIR execution engine.
//
//===----------------------------------------------------------------------===//

//! Macros that generate the C ABI functions for each standard library type.

/// The length of a caller's buffer as a `usize`. `None` when the length is zero or is too
/// long for a Rust slice, which are the two cases where the caller has nothing to do.
#[must_use]
pub fn buffer_len(len: u64) -> Option<usize> {
    if len == 0 {
        return None;
    }
    usize::try_from(len)
        .ok()
        .filter(|&len| isize::try_from(len).is_ok())
}

/// Macro to instantiate C-ABI compatible FFI wrappers for `Vec<T>`.
///
/// This generates `vx_vec_new_<name>`, `vx_vec_push_<name>`,
/// `vx_vec_len_<name>`, and `vx_vec_drop_<name>` functions
/// that operate on an opaque `*mut std::ffi::c_void`.
#[macro_export]
macro_rules! instantiate_vec_ffi {
    ($type_name:ident, $type:ty) => {
        paste::paste! {
            /// Makes an empty vector and returns a pointer that owns it.
            #[no_mangle]
            pub extern "C" fn [<vx_vec_new_ $type_name>]() -> *mut std::ffi::c_void {
                let vec: Box<Vec<$type>> = Box::default();
                Box::into_raw(vec).cast::<std::ffi::c_void>()
            }

            /// Adds `val` to the end of the vector. Does nothing when `ptr` is null.
            #[no_mangle]
            pub extern "C" fn [<vx_vec_push_ $type_name>](ptr: *mut std::ffi::c_void, val: $type) {
                if ptr.is_null() {
                    return;
                }
                // SAFETY: the caller passes a live pointer that this library returned, and nothing
                // else uses it during this call.
                let vec = unsafe { &mut *ptr.cast::<Vec<$type>>() };
                vec.push(val);
            }

            /// The number of elements in the vector, or 0 when `ptr` is null.
            #[no_mangle]
            pub extern "C" fn [<vx_vec_len_ $type_name>](ptr: *mut std::ffi::c_void) -> usize {
                if ptr.is_null() {
                    return 0;
                }
                // SAFETY: the caller passes a live pointer that this library returned.
                let vec = unsafe { &*ptr.cast::<Vec<$type>>() };
                vec.len()
            }

            /// Frees the vector. Does nothing when `ptr` is null.
            #[no_mangle]
            pub extern "C" fn [<vx_vec_drop_ $type_name>](ptr: *mut std::ffi::c_void) {
                if !ptr.is_null() {
                    // SAFETY: the pointer came from `Box::into_raw` in this library and the caller
                    // gives up ownership here.
                    let _ = unsafe { Box::from_raw(ptr.cast::<Vec<$type>>()) };
                }
            }
        }
    };
}

/// Macro to instantiate C-ABI compatible FFI wrappers for `Option<T>`.
#[macro_export]
macro_rules! instantiate_option_ffi {
    ($type_name:ident, $type:ty) => {
        paste::paste! {
            /// Makes an option that holds `val` and returns a pointer that owns it.
            #[no_mangle]
            pub extern "C" fn [<vx_option_new_some_ $type_name>](val: $type) -> *mut std::ffi::c_void {
                let opt: Box<Option<$type>> = Box::new(Some(val));
                Box::into_raw(opt).cast::<std::ffi::c_void>()
            }

            /// Makes an empty option and returns a pointer that owns it.
            #[no_mangle]
            pub extern "C" fn [<vx_option_new_none_ $type_name>]() -> *mut std::ffi::c_void {
                let opt: Box<Option<$type>> = Box::new(None);
                Box::into_raw(opt).cast::<std::ffi::c_void>()
            }

            /// Whether the option holds a value. A null `ptr` counts as empty.
            #[no_mangle]
            pub extern "C" fn [<vx_option_is_some_ $type_name>](ptr: *mut std::ffi::c_void) -> bool {
                if ptr.is_null() { return false; }
                // SAFETY: the caller passes a live pointer that this library returned.
                let opt = unsafe { &*ptr.cast::<Option<$type>>() };
                opt.is_some()
            }

            /// Whether the option is empty. A null `ptr` counts as empty.
            #[no_mangle]
            pub extern "C" fn [<vx_option_is_none_ $type_name>](ptr: *mut std::ffi::c_void) -> bool {
                if ptr.is_null() { return true; }
                // SAFETY: the caller passes a live pointer that this library returned.
                let opt = unsafe { &*ptr.cast::<Option<$type>>() };
                opt.is_none()
            }

            /// A copy of the value inside the option. `ptr` must not be null.
            ///
            /// # Panics
            /// Panics when the option is empty.
            #[no_mangle]
            pub extern "C" fn [<vx_option_unwrap_ $type_name>](ptr: *mut std::ffi::c_void) -> $type {
                // SAFETY: the caller passes a live pointer that this library returned.
                let opt = unsafe { &*ptr.cast::<Option<$type>>() };
                opt.clone().unwrap()
            }

            /// Frees the option. Does nothing when `ptr` is null.
            #[no_mangle]
            pub extern "C" fn [<vx_option_drop_ $type_name>](ptr: *mut std::ffi::c_void) {
                if !ptr.is_null() {
                    // SAFETY: the pointer came from `Box::into_raw` in this library and the caller
                    // gives up ownership here.
                    let _ = unsafe { Box::from_raw(ptr.cast::<Option<$type>>()) };
                }
            }
        }
    };
}

/// Macro to instantiate C-ABI compatible FFI wrappers for `Result<T, E>`.
#[macro_export]
macro_rules! instantiate_result_ffi {
    ($t_name:ident, $t_type:ty, $e_name:ident, $e_type:ty) => {
        paste::paste! {
            /// Makes a successful result that holds `val` and returns a pointer that owns it.
            #[no_mangle]
            pub extern "C" fn [<vx_result_new_ok_ $t_name _ $e_name>](val: $t_type) -> *mut std::ffi::c_void {
                let res: Box<Result<$t_type, $e_type>> = Box::new(Ok(val));
                Box::into_raw(res).cast::<std::ffi::c_void>()
            }

            /// Makes a failed result that holds `err` and returns a pointer that owns it.
            #[no_mangle]
            pub extern "C" fn [<vx_result_new_err_ $t_name _ $e_name>](err: $e_type) -> *mut std::ffi::c_void {
                let res: Box<Result<$t_type, $e_type>> = Box::new(Err(err));
                Box::into_raw(res).cast::<std::ffi::c_void>()
            }

            /// Whether the result is a success. False when `ptr` is null.
            #[no_mangle]
            pub extern "C" fn [<vx_result_is_ok_ $t_name _ $e_name>](ptr: *mut std::ffi::c_void) -> bool {
                if ptr.is_null() { return false; }
                // SAFETY: the caller passes a live pointer that this library returned.
                let res = unsafe { &*ptr.cast::<Result<$t_type, $e_type>>() };
                res.is_ok()
            }

            /// Whether the result is a failure. False when `ptr` is null.
            #[no_mangle]
            pub extern "C" fn [<vx_result_is_err_ $t_name _ $e_name>](ptr: *mut std::ffi::c_void) -> bool {
                if ptr.is_null() { return false; }
                // SAFETY: the caller passes a live pointer that this library returned.
                let res = unsafe { &*ptr.cast::<Result<$t_type, $e_type>>() };
                res.is_err()
            }

            /// A copy of the success value inside the result. `ptr` must not be null.
            ///
            /// # Panics
            /// Panics when the result is a failure.
            #[no_mangle]
            pub extern "C" fn [<vx_result_unwrap_ $t_name _ $e_name>](ptr: *mut std::ffi::c_void) -> $t_type {
                // SAFETY: the caller passes a live pointer that this library returned.
                let res = unsafe { &*ptr.cast::<Result<$t_type, $e_type>>() };
                res.clone().unwrap()
            }

            /// Frees the result. Does nothing when `ptr` is null.
            #[no_mangle]
            pub extern "C" fn [<vx_result_drop_ $t_name _ $e_name>](ptr: *mut std::ffi::c_void) {
                if !ptr.is_null() {
                    // SAFETY: the pointer came from `Box::into_raw` in this library and the caller
                    // gives up ownership here.
                    let _ = unsafe { Box::from_raw(ptr.cast::<Result<$t_type, $e_type>>()) };
                }
            }
        }
    };
}

/// Macro to instantiate C-ABI compatible FFI wrappers for `HashMap<K, V>`.
#[macro_export]
macro_rules! instantiate_hash_map_ffi {
    ($name:ident, $k:ty, $v:ty) => {
        paste::paste! {
            /// Makes an empty map and returns a pointer that owns it.
            #[no_mangle]
            pub extern "C" fn [<vx_hash_map_new_ $name>]() -> *mut std::ffi::c_void {
                let map: Box<std::collections::HashMap<$k, $v>> = Box::default();
                Box::into_raw(map).cast::<std::ffi::c_void>()
            }

            /// Stores `val` under `key`, replacing any older value. Does nothing when `ptr` is null.
            #[no_mangle]
            pub extern "C" fn [<vx_hash_map_insert_ $name>](ptr: *mut std::ffi::c_void, key: $k, val: $v) {
                if !ptr.is_null() {
                    // SAFETY: the caller passes a live pointer that this library returned, and
                    // nothing else uses it during this call.
                    let map = unsafe { &mut *ptr.cast::<std::collections::HashMap<$k, $v>>() };
                    map.insert(key, val);
                }
            }

            /// Looks up `key` and returns a new option that the caller must free with the
            /// matching `vx_option_drop_*` function. Returns null when `ptr` is null.
            #[no_mangle]
            pub extern "C" fn [<vx_hash_map_get_ $name>](ptr: *mut std::ffi::c_void, key: $k) -> *mut std::ffi::c_void {
                if ptr.is_null() { return std::ptr::null_mut(); }
                // SAFETY: the caller passes a live pointer that this library returned.
                let map = unsafe { &*ptr.cast::<std::collections::HashMap<$k, $v>>() };
                let opt: Box<Option<$v>> = Box::new(map.get(&key).copied());
                Box::into_raw(opt).cast::<std::ffi::c_void>()
            }

            /// Whether the map has a value for `key`. False when `ptr` is null.
            #[no_mangle]
            pub extern "C" fn [<vx_hash_map_contains_key_ $name>](ptr: *mut std::ffi::c_void, key: $k) -> bool {
                if ptr.is_null() { return false; }
                // SAFETY: the caller passes a live pointer that this library returned.
                let map = unsafe { &*ptr.cast::<std::collections::HashMap<$k, $v>>() };
                map.contains_key(&key)
            }

            /// The number of entries in the map, or 0 when `ptr` is null.
            #[no_mangle]
            pub extern "C" fn [<vx_hash_map_len_ $name>](ptr: *mut std::ffi::c_void) -> usize {
                if ptr.is_null() { return 0; }
                // SAFETY: the caller passes a live pointer that this library returned.
                let map = unsafe { &*ptr.cast::<std::collections::HashMap<$k, $v>>() };
                map.len()
            }

            /// Frees the map. Does nothing when `ptr` is null.
            #[no_mangle]
            pub extern "C" fn [<vx_hash_map_drop_ $name>](ptr: *mut std::ffi::c_void) {
                if !ptr.is_null() {
                    // SAFETY: the pointer came from `Box::into_raw` in this library and the caller
                    // gives up ownership here.
                    let _ = unsafe { Box::from_raw(ptr.cast::<std::collections::HashMap<$k, $v>>()) };
                }
            }
        }
    };
}

/// Macro to instantiate C-ABI compatible FFI wrappers for `HashSet<T>`.
#[macro_export]
macro_rules! instantiate_hash_set_ffi {
    ($name:ident, $t:ty) => {
        paste::paste! {
            /// Makes an empty set and returns a pointer that owns it.
            #[no_mangle]
            pub extern "C" fn [<vx_hash_set_new_ $name>]() -> *mut std::ffi::c_void {
                let set: Box<std::collections::HashSet<$t>> = Box::default();
                Box::into_raw(set).cast::<std::ffi::c_void>()
            }

            /// Adds `val` to the set. Does nothing when `ptr` is null.
            #[no_mangle]
            pub extern "C" fn [<vx_hash_set_insert_ $name>](ptr: *mut std::ffi::c_void, val: $t) {
                if !ptr.is_null() {
                    // SAFETY: the caller passes a live pointer that this library returned, and
                    // nothing else uses it during this call.
                    let set = unsafe { &mut *ptr.cast::<std::collections::HashSet<$t>>() };
                    set.insert(val);
                }
            }

            /// Whether the set holds `val`. False when `ptr` is null.
            #[no_mangle]
            pub extern "C" fn [<vx_hash_set_contains_ $name>](ptr: *mut std::ffi::c_void, val: $t) -> bool {
                if ptr.is_null() { return false; }
                // SAFETY: the caller passes a live pointer that this library returned.
                let set = unsafe { &*ptr.cast::<std::collections::HashSet<$t>>() };
                set.contains(&val)
            }

            /// The number of values in the set, or 0 when `ptr` is null.
            #[no_mangle]
            pub extern "C" fn [<vx_hash_set_len_ $name>](ptr: *mut std::ffi::c_void) -> usize {
                if ptr.is_null() { return 0; }
                // SAFETY: the caller passes a live pointer that this library returned.
                let set = unsafe { &*ptr.cast::<std::collections::HashSet<$t>>() };
                set.len()
            }

            /// Frees the set. Does nothing when `ptr` is null.
            #[no_mangle]
            pub extern "C" fn [<vx_hash_set_drop_ $name>](ptr: *mut std::ffi::c_void) {
                if !ptr.is_null() {
                    // SAFETY: the pointer came from `Box::into_raw` in this library and the caller
                    // gives up ownership here.
                    let _ = unsafe { Box::from_raw(ptr.cast::<std::collections::HashSet<$t>>()) };
                }
            }
        }
    };
}

/// Macro to instantiate C-ABI compatible FFI wrappers for `String`.
#[macro_export]
macro_rules! instantiate_string_ffi {
    () => {
        /// Makes an empty string and returns a pointer that owns it.
        #[no_mangle]
        pub extern "C" fn vx_string_new() -> *mut std::ffi::c_void {
            let s: Box<String> = Box::default();
            Box::into_raw(s).cast::<std::ffi::c_void>()
        }

        /// A string of `len` NUL bytes, for the caller to overwrite through
        /// `vx_string_as_mut_ptr` with the same number of ASCII bytes.
        ///
        /// # Panics
        /// Panics when `len` is negative.
        #[no_mangle]
        pub extern "C" fn vx_string_with_len(len: i64) -> *mut std::ffi::c_void {
            let byte_count = usize::try_from(len)
                .unwrap_or_else(|_| panic!("vx_string_with_len: negative length {len}"));
            // SAFETY: NUL bytes are valid UTF-8, so the unchecked conversion holds.
            let zeroed = unsafe { String::from_utf8_unchecked(vec![0u8; byte_count]) };
            let s: Box<String> = Box::new(zeroed);
            Box::into_raw(s).cast::<std::ffi::c_void>()
        }

        /// The string's bytes, writable. Writing anything but ASCII breaks the UTF-8
        /// invariant, which is why no Vx caller outside `std::string` should see this.
        ///
        /// # Panics
        /// Panics when `ptr` is null.
        #[no_mangle]
        pub extern "C" fn vx_string_as_mut_ptr(ptr: *mut std::ffi::c_void) -> *mut u8 {
            assert!(!ptr.is_null(), "vx_string_as_mut_ptr: null string");
            // SAFETY: the caller passes a live pointer that this library returned, and nothing else
            // uses it during this call.
            let s = unsafe { &mut *ptr.cast::<String>() };
            // SAFETY: this only hands out the address. The caller promises to write ASCII
            // bytes through it, which keeps the string valid UTF-8.
            unsafe { s.as_mut_vec().as_mut_ptr() }
        }

        /// Copies a C string into a new string. Bytes that are not valid UTF-8 become the
        /// replacement character, and a null `c_str` gives an empty string.
        #[no_mangle]
        pub extern "C" fn vx_string_from_c_str(
            c_str: *const std::ffi::c_char,
        ) -> *mut std::ffi::c_void {
            if c_str.is_null() {
                return vx_string_new();
            }
            // SAFETY: the caller passes a NUL-terminated C string that stays alive for this call.
            let c_str = unsafe { std::ffi::CStr::from_ptr(c_str) };
            let s: Box<String> = Box::new(c_str.to_string_lossy().into_owned());
            Box::into_raw(s).cast::<std::ffi::c_void>()
        }

        /// Appends a C string to the string. Does nothing when either pointer is null.
        #[no_mangle]
        pub extern "C" fn vx_string_push_c_str(
            ptr: *mut std::ffi::c_void,
            c_str: *const std::ffi::c_char,
        ) {
            if ptr.is_null() || c_str.is_null() {
                return;
            }
            // SAFETY: the caller passes a live pointer that this library returned, and nothing else
            // uses it during this call.
            let s = unsafe { &mut *ptr.cast::<String>() };
            // SAFETY: the caller passes a NUL-terminated C string that stays alive for this call.
            let c_str = unsafe { std::ffi::CStr::from_ptr(c_str) };
            s.push_str(&c_str.to_string_lossy());
        }

        /// The length of the string in bytes, or 0 when `ptr` is null.
        #[no_mangle]
        pub extern "C" fn vx_string_len(ptr: *mut std::ffi::c_void) -> usize {
            if ptr.is_null() {
                return 0;
            }
            // SAFETY: the caller passes a live pointer that this library returned.
            let s = unsafe { &*ptr.cast::<String>() };
            s.len()
        }

        /// 1 when the two strings hold the same bytes, else 0. A null string is empty.
        #[no_mangle]
        pub extern "C" fn vx_string_eq(a: *mut std::ffi::c_void, b: *mut std::ffi::c_void) -> i32 {
            let bytes = |p: *mut std::ffi::c_void| {
                if p.is_null() {
                    ""
                } else {
                    // SAFETY: a non-null string pointer came from this library and is still alive.
                    unsafe { (*p.cast::<String>()).as_str() }
                }
            };
            i32::from(bytes(a) == bytes(b))
        }

        /// Copies the string into a new C string, which the caller must free with
        /// `vx_string_free_c_str`. Returns null when `ptr` is null.
        ///
        /// # Panics
        /// Panics when the string has a NUL byte in it, because a C string cannot hold one.
        #[no_mangle]
        pub extern "C" fn vx_string_as_c_str(
            ptr: *mut std::ffi::c_void,
        ) -> *const std::ffi::c_char {
            if ptr.is_null() {
                return std::ptr::null();
            }
            // Note: This forces an allocation of a CString, which we must leak or store.
            // For simplicity and safety in a quick binding, we could return a leaked ptr,
            // but that's a memory leak.
            // A better approach is to modify the string in place to append a null byte if it doesn't have one,
            // or just use `as_ptr()` if we know it won't be used past its lifetime, but it needs a null terminator.
            // For now, let's leak a CString.
            // SAFETY: the caller passes a live pointer that this library returned.
            let s = unsafe { &*ptr.cast::<String>() };
            let c_string = std::ffi::CString::new(s.clone()).unwrap();
            c_string.into_raw()
        }

        /// Frees a C string that `vx_string_as_c_str` returned. Does nothing when it is null.
        #[no_mangle]
        pub extern "C" fn vx_string_free_c_str(c_str: *mut std::ffi::c_char) {
            if !c_str.is_null() {
                // SAFETY: the pointer came from `CString::into_raw` in this library and the caller
                // gives up ownership here.
                let _ = unsafe { std::ffi::CString::from_raw(c_str) };
            }
        }

        /// Frees the string. Does nothing when `ptr` is null.
        #[no_mangle]
        pub extern "C" fn vx_string_drop(ptr: *mut std::ffi::c_void) {
            if !ptr.is_null() {
                // SAFETY: the pointer came from `Box::into_raw` in this library and the caller
                // gives up ownership here.
                let _ = unsafe { Box::from_raw(ptr.cast::<String>()) };
            }
        }
    };
}

/// Macro to instantiate C-ABI compatible FFI wrappers for `std::fs::File`.
#[macro_export]
macro_rules! instantiate_file_ffi {
    () => {
        use std::io::{Read, Seek, SeekFrom, Write};

        // Private to the bridge, but part of its i32 ABI. Keep these values in sync with
        // the table in docs/implementation_plans/io_error_contract.md.
        const VX_FILE_STATUS_OK: i32 = 0;
        const VX_FILE_STATUS_INVALID_ARGUMENT: i32 = -1;
        const VX_FILE_ERROR_NOT_FOUND: i32 = 1;
        const VX_FILE_ERROR_PERMISSION_DENIED: i32 = 2;
        const VX_FILE_ERROR_ALREADY_EXISTS: i32 = 3;
        const VX_FILE_ERROR_INVALID_INPUT: i32 = 4;
        const VX_FILE_ERROR_INTERRUPTED: i32 = 5;
        const VX_FILE_ERROR_WOULD_BLOCK: i32 = 6;
        const VX_FILE_ERROR_UNEXPECTED_EOF: i32 = 7;
        const VX_FILE_ERROR_WRITE_ZERO: i32 = 8;
        const VX_FILE_ERROR_OUT_OF_MEMORY: i32 = 9;
        const VX_FILE_ERROR_LIMIT_EXCEEDED: i32 = 10;
        const VX_FILE_ERROR_UNSUPPORTED: i32 = 11;
        const VX_FILE_ERROR_OTHER: i32 = 12;

        /// The portable error number that the Vx side sees for an operating system error.
        fn vx_file_error_kind(kind: std::io::ErrorKind) -> i32 {
            match kind {
                std::io::ErrorKind::NotFound => VX_FILE_ERROR_NOT_FOUND,
                std::io::ErrorKind::PermissionDenied | std::io::ErrorKind::ReadOnlyFilesystem => {
                    VX_FILE_ERROR_PERMISSION_DENIED
                }
                std::io::ErrorKind::AlreadyExists => VX_FILE_ERROR_ALREADY_EXISTS,
                std::io::ErrorKind::InvalidInput => VX_FILE_ERROR_INVALID_INPUT,
                std::io::ErrorKind::Interrupted => VX_FILE_ERROR_INTERRUPTED,
                std::io::ErrorKind::WouldBlock => VX_FILE_ERROR_WOULD_BLOCK,
                std::io::ErrorKind::UnexpectedEof => VX_FILE_ERROR_UNEXPECTED_EOF,
                std::io::ErrorKind::WriteZero => VX_FILE_ERROR_WRITE_ZERO,
                std::io::ErrorKind::OutOfMemory => VX_FILE_ERROR_OUT_OF_MEMORY,
                std::io::ErrorKind::StorageFull
                | std::io::ErrorKind::QuotaExceeded
                | std::io::ErrorKind::FileTooLarge => VX_FILE_ERROR_LIMIT_EXCEEDED,
                std::io::ErrorKind::Unsupported => VX_FILE_ERROR_UNSUPPORTED,
                _ => VX_FILE_ERROR_OTHER,
            }
        }

        /// The three places a `vx_file_try_*` call reports its result through.
        struct VxFileTransferOutputs {
            /// Receives the number of bytes that were read or written.
            count: *mut u64,
            /// Receives the operating system's error code, when it gave one.
            native_code: *mut i32,
            /// Receives whether `native_code` holds a real error code.
            has_native_code: *mut bool,
        }

        /// Sets every output that is not null to its "nothing happened" value.
        ///
        /// # Safety
        /// Every non-null output must be aligned, writable, and disjoint from the others.
        unsafe fn vx_file_clear_transfer_outputs(outputs: &VxFileTransferOutputs) {
            // Clear every provided output even when another output pointer is null.
            // SAFETY: each pointer is written only when it is not null, and the caller
            // promises a non-null output is aligned and writable.
            unsafe {
                if !outputs.count.is_null() {
                    *outputs.count = 0;
                }
                if !outputs.native_code.is_null() {
                    *outputs.native_code = 0;
                }
                if !outputs.has_native_code.is_null() {
                    *outputs.has_native_code = false;
                }
            }
        }

        /// Writes the operating system's error code to the outputs and returns the
        /// portable error number for `error`.
        ///
        /// # Safety
        /// Both output pointers must be non-null, aligned, writable, and disjoint.
        unsafe fn vx_file_store_error(
            error: &std::io::Error,
            native_code: *mut i32,
            has_native_code: *mut bool,
        ) -> i32 {
            let captured_code = error.raw_os_error();
            // SAFETY: the caller promises both pointers are non-null, aligned and writable.
            unsafe {
                *has_native_code = captured_code.is_some();
                *native_code = captured_code.unwrap_or(0);
            }
            vx_file_error_kind(error.kind())
        }

        /// The shared body of `vx_file_try_read` and `vx_file_try_write`: checks the
        /// arguments, runs `transfer` with the file and the buffer length, and reports
        /// the result through `outputs`.
        ///
        /// # Safety
        /// The handle, buffer, and output pointers must meet the calling function's
        /// safety requirements. `transfer` may use the buffer only during this call.
        unsafe fn vx_file_try_transfer(
            ptr: *mut std::ffi::c_void,
            buffer: *const u8,
            len: u64,
            outputs: &VxFileTransferOutputs,
            transfer: impl FnOnce(&mut std::fs::File, usize) -> std::io::Result<usize>,
        ) -> i32 {
            // SAFETY: the caller promises every non-null output is aligned and writable.
            unsafe { vx_file_clear_transfer_outputs(outputs) };
            if outputs.count.is_null()
                || outputs.native_code.is_null()
                || outputs.has_native_code.is_null()
            {
                return VX_FILE_STATUS_INVALID_ARGUMENT;
            }
            if ptr.is_null() || (len != 0 && buffer.is_null()) {
                return VX_FILE_STATUS_INVALID_ARGUMENT;
            }
            if len == 0 {
                return VX_FILE_STATUS_OK;
            }
            // A length that is not zero and still has no `usize` form is too long for a slice.
            let Some(len) = $crate::ffi::macros::buffer_len(len) else {
                return VX_FILE_STATUS_INVALID_ARGUMENT;
            };
            // SAFETY: the caller passes a live file that this library returned, and nothing
            // else uses it during this call.
            let file = unsafe { &mut *ptr.cast::<std::fs::File>() };
            match transfer(file, len) {
                Ok(transferred) => {
                    // SAFETY: `count` was checked to be non-null above, and the caller
                    // promises it is aligned and writable.
                    unsafe { *outputs.count = transferred as u64 };
                    VX_FILE_STATUS_OK
                }
                // SAFETY: both pointers were checked to be non-null above, and the caller
                // promises they are aligned and writable.
                Err(error) => unsafe {
                    vx_file_store_error(&error, outputs.native_code, outputs.has_native_code)
                },
            }
        }

        /// Opens the file at `c_path`. Mode 0 reads, mode 1 writes and empties the file, and
        /// mode 2 reads and writes. Returns null when the path is null or the file cannot be
        /// opened, and stops the program on any other mode.
        #[no_mangle]
        pub extern "C" fn vx_file_open(
            c_path: *const std::ffi::c_char,
            mode: i32,
        ) -> *mut std::ffi::c_void {
            // The Vx OpenMode variants map to these three values.
            let mut opts = std::fs::OpenOptions::new();
            match mode {
                0 => {
                    opts.read(true);
                }
                1 => {
                    opts.write(true).create(true).truncate(true);
                }
                2 => {
                    opts.read(true).write(true).create(true);
                }
                _ => {
                    eprintln!("vx_file_open received unknown mode {mode}");
                    std::process::abort();
                }
            }

            if c_path.is_null() {
                return std::ptr::null_mut();
            }
            // SAFETY: the caller passes a NUL-terminated C string that stays alive for this call.
            let c_path = unsafe { std::ffi::CStr::from_ptr(c_path) };
            #[cfg(unix)]
            let path = {
                use std::os::unix::ffi::OsStrExt;
                std::ffi::OsStr::from_bytes(c_path.to_bytes())
            };
            #[cfg(not(unix))]
            let path = match c_path.to_str() {
                Ok(path) => path,
                Err(_) => return std::ptr::null_mut(),
            };

            if let Ok(file) = opts.open(path) {
                let boxed: Box<std::fs::File> = Box::new(file);
                Box::into_raw(boxed).cast::<std::ffi::c_void>()
            } else {
                std::ptr::null_mut()
            }
        }

        /// Reads up to `len` bytes into `buffer` and returns how many were read. Returns 0 on
        /// an error, a null pointer, or a zero length.
        #[no_mangle]
        pub extern "C" fn vx_file_read(
            ptr: *mut std::ffi::c_void,
            buffer: *mut u8,
            len: u64,
        ) -> u64 {
            if ptr.is_null() || buffer.is_null() {
                return 0;
            }
            let Some(len) = $crate::ffi::macros::buffer_len(len) else {
                return 0;
            };
            // SAFETY: the caller passes a live pointer that this library returned, and nothing else
            // uses it during this call.
            let file = unsafe { &mut *ptr.cast::<std::fs::File>() };
            // SAFETY: the caller passes a buffer of at least `len` bytes that stays alive for this
            // call.
            let buf_slice = unsafe { std::slice::from_raw_parts_mut(buffer, len) };
            file.read(buf_slice).unwrap_or(0) as u64
        }

        /// Writes up to `len` bytes from `buffer` and returns how many were written. Returns 0
        /// on an error, a null pointer, or a zero length.
        #[no_mangle]
        pub extern "C" fn vx_file_write(
            ptr: *mut std::ffi::c_void,
            buffer: *const u8,
            len: u64,
        ) -> u64 {
            if ptr.is_null() || buffer.is_null() {
                return 0;
            }
            let Some(len) = $crate::ffi::macros::buffer_len(len) else {
                return 0;
            };
            // SAFETY: the caller passes a live pointer that this library returned, and nothing else
            // uses it during this call.
            let file = unsafe { &mut *ptr.cast::<std::fs::File>() };
            // SAFETY: the caller passes a buffer of at least `len` bytes that stays alive for this
            // call.
            let buf_slice = unsafe { std::slice::from_raw_parts(buffer, len) };
            file.write(buf_slice).unwrap_or(0) as u64
        }

        /// Zero is success, -1 is invalid arguments, and 1..=12 is a portable
        /// OS error category. Every non-null output is initialized on every path:
        /// `count` is zero except on success; `native_code` is meaningful only when
        /// `has_native_code` is true. See `io_error_contract.md` for the ABI table.
        /// The native code is copied before classification or other work.
        /// On success, only the first `count` buffer bytes contain data read from
        /// the file. Bytes in `buffer[count..len]` may also change, including at
        /// EOF; callers must not rely on their previous contents being preserved.
        ///
        /// # Safety
        /// A null `ptr` or `buffer` is accepted and reports invalid arguments. A
        /// non-null `ptr` must point to a live bridge-owned `std::fs::File`, with no
        /// concurrent access or drop. When `ptr` and `buffer` are non-null and
        /// `0 < len <= isize::MAX`, `buffer` must point to `len` writable
        /// bytes. They need not be initialized. Every non-null output pointer
        /// must be aligned and writable, even if another argument is invalid.
        /// The file, buffer, and outputs must not overlap during the call.
        #[no_mangle]
        pub unsafe extern "C" fn vx_file_try_read(
            ptr: *mut std::ffi::c_void,
            buffer: *mut u8,
            len: u64,
            count: *mut u64,
            native_code: *mut i32,
            has_native_code: *mut bool,
        ) -> i32 {
            let outputs = VxFileTransferOutputs {
                count,
                native_code,
                has_native_code,
            };
            let read = |file: &mut std::fs::File, len: usize| {
                // Read::read requires an initialized slice, even when the
                // caller provides writable memory without initial values.
                // SAFETY: the helper calls this only with a non-null `buffer`, which
                // the caller promises has `len` writable bytes.
                let bytes = unsafe {
                    std::ptr::write_bytes(buffer, 0, len);
                    std::slice::from_raw_parts_mut(buffer, len)
                };
                file.read(bytes)
            };
            // SAFETY: the caller meets this function's safety rules, which are the ones
            // `vx_file_try_transfer` asks for.
            unsafe { vx_file_try_transfer(ptr, buffer, len, &outputs, read) }
        }

        /// Uses the same status and output contract as `vx_file_try_read`.
        ///
        /// # Safety
        /// The handle, outputs, null arguments, and non-overlap rules are the same
        /// as for `vx_file_try_read`. For a valid positive length, `buffer` must
        /// point to `len` initialized, readable bytes instead of writable bytes.
        #[no_mangle]
        pub unsafe extern "C" fn vx_file_try_write(
            ptr: *mut std::ffi::c_void,
            buffer: *const u8,
            len: u64,
            count: *mut u64,
            native_code: *mut i32,
            has_native_code: *mut bool,
        ) -> i32 {
            let outputs = VxFileTransferOutputs {
                count,
                native_code,
                has_native_code,
            };
            let write = |file: &mut std::fs::File, len: usize| {
                // SAFETY: the helper calls this only with a non-null `buffer`, which
                // the caller promises has `len` readable bytes.
                let bytes = unsafe { std::slice::from_raw_parts(buffer, len) };
                file.write(bytes)
            };
            // SAFETY: the caller meets this function's safety rules, which are the ones
            // `vx_file_try_transfer` asks for.
            unsafe { vx_file_try_transfer(ptr, buffer, len, &outputs, write) }
        }

        /// Moves the file position and returns the new position, or -1 on an error. `whence`
        /// 0 counts from the start, 1 from the current position, and 2 from the end.
        #[no_mangle]
        pub extern "C" fn vx_file_seek(
            ptr: *mut std::ffi::c_void,
            offset: i64,
            whence: i32,
        ) -> i64 {
            if ptr.is_null() {
                return -1;
            }
            // SAFETY: the caller passes a live pointer that this library returned, and nothing else
            // uses it during this call.
            let file = unsafe { &mut *ptr.cast::<std::fs::File>() };
            let seek_from = match whence {
                0 => SeekFrom::Start(offset.cast_unsigned()),
                1 => SeekFrom::Current(offset),
                2 => SeekFrom::End(offset),
                _ => return -1,
            };
            file.seek(seek_from).map(u64::cast_signed).unwrap_or(-1)
        }

        /// The path of `c_filename` inside the system's temporary directory, as a new C
        /// string that the caller must free with `vx_free_temp_file`. `c_filename` must not be null.
        ///
        /// # Panics
        /// Panics when the temporary directory's path has a NUL byte in it.
        #[no_mangle]
        pub extern "C" fn vx_get_temp_file(
            c_filename: *const std::ffi::c_char,
        ) -> *mut std::ffi::c_char {
            // SAFETY: the caller passes a NUL-terminated C string that stays alive for this call.
            let filename = unsafe { std::ffi::CStr::from_ptr(c_filename) }.to_string_lossy();
            let mut path = std::env::temp_dir();
            path.push(filename.as_ref());
            let path_str = path.to_string_lossy().into_owned();
            let c_string = std::ffi::CString::new(path_str).unwrap();
            c_string.into_raw()
        }

        /// Frees a path that `vx_get_temp_file` returned, and returns 0. A null `ptr` is allowed.
        #[no_mangle]
        pub extern "C" fn vx_free_temp_file(ptr: *mut std::ffi::c_char) -> i32 {
            if !ptr.is_null() {
                // SAFETY: the pointer came from `CString::into_raw` in this library and the caller
                // gives up ownership here.
                let _ = unsafe { std::ffi::CString::from_raw(ptr) };
            }
            0
        }

        /// Closes the file and frees it. Does nothing when `ptr` is null.
        #[no_mangle]
        pub extern "C" fn vx_file_drop(ptr: *mut std::ffi::c_void) {
            if !ptr.is_null() {
                // SAFETY: the pointer came from `Box::into_raw` in this library and the caller
                // gives up ownership here.
                let _ = unsafe { Box::from_raw(ptr.cast::<std::fs::File>()) };
            }
        }
    };
}

/// Macro to instantiate C-ABI compatible FFI wrappers for `std::net::TcpStream`.
#[macro_export]
macro_rules! instantiate_tcp_stream_ffi {
    () => {
        use std::io::{Read, Write};

        /// Connects to the address in `c_addr`, such as `127.0.0.1:80`. Returns null when the
        /// address is null or the connection fails.
        #[no_mangle]
        pub extern "C" fn vx_tcp_stream_connect(
            c_addr: *const std::ffi::c_char,
        ) -> *mut std::ffi::c_void {
            if c_addr.is_null() {
                return std::ptr::null_mut();
            }
            // SAFETY: the caller passes a NUL-terminated C string that stays alive for this call.
            let addr_str = unsafe { std::ffi::CStr::from_ptr(c_addr) }.to_string_lossy();
            if let Ok(stream) = std::net::TcpStream::connect(addr_str.as_ref()) {
                let boxed: Box<std::net::TcpStream> = Box::new(stream);
                Box::into_raw(boxed).cast::<std::ffi::c_void>()
            } else {
                std::ptr::null_mut()
            }
        }

        /// Reads up to `len` bytes into `buffer` and returns how many were read. Returns 0 on
        /// an error, a null pointer, or a zero length.
        #[no_mangle]
        pub extern "C" fn vx_tcp_stream_read(
            ptr: *mut std::ffi::c_void,
            buffer: *mut u8,
            len: u64,
        ) -> u64 {
            if ptr.is_null() || buffer.is_null() {
                return 0;
            }
            let Some(len) = $crate::ffi::macros::buffer_len(len) else {
                return 0;
            };
            // SAFETY: the caller passes a live pointer that this library returned, and nothing else
            // uses it during this call.
            let stream = unsafe { &mut *ptr.cast::<std::net::TcpStream>() };
            // SAFETY: the caller passes a buffer of at least `len` bytes that stays alive for this
            // call.
            let buf_slice = unsafe { std::slice::from_raw_parts_mut(buffer, len) };
            stream.read(buf_slice).unwrap_or(0) as u64
        }

        /// Writes up to `len` bytes from `buffer` and returns how many were written. Returns 0
        /// on an error, a null pointer, or a zero length.
        #[no_mangle]
        pub extern "C" fn vx_tcp_stream_write(
            ptr: *mut std::ffi::c_void,
            buffer: *const u8,
            len: u64,
        ) -> u64 {
            if ptr.is_null() || buffer.is_null() {
                return 0;
            }
            let Some(len) = $crate::ffi::macros::buffer_len(len) else {
                return 0;
            };
            // SAFETY: the caller passes a live pointer that this library returned, and nothing else
            // uses it during this call.
            let stream = unsafe { &mut *ptr.cast::<std::net::TcpStream>() };
            // SAFETY: the caller passes a buffer of at least `len` bytes that stays alive for this
            // call.
            let buf_slice = unsafe { std::slice::from_raw_parts(buffer, len) };
            stream.write(buf_slice).unwrap_or(0) as u64
        }

        /// Closes the connection and frees it. Does nothing when `ptr` is null.
        #[no_mangle]
        pub extern "C" fn vx_tcp_stream_drop(ptr: *mut std::ffi::c_void) {
            if !ptr.is_null() {
                // SAFETY: the pointer came from `Box::into_raw` in this library and the caller
                // gives up ownership here.
                let _ = unsafe { Box::from_raw(ptr.cast::<std::net::TcpStream>()) };
            }
        }
    };
}

/// Macro to instantiate C-ABI compatible FFI wrappers for `std::net::UdpSocket`.
#[macro_export]
macro_rules! instantiate_udp_socket_ffi {
    () => {
        /// Opens a UDP socket on the address in `c_addr`. Returns null when the address is
        /// null or the socket cannot be opened.
        #[no_mangle]
        pub extern "C" fn vx_udp_socket_bind(
            c_addr: *const std::ffi::c_char,
        ) -> *mut std::ffi::c_void {
            if c_addr.is_null() {
                return std::ptr::null_mut();
            }
            // SAFETY: the caller passes a NUL-terminated C string that stays alive for this call.
            let addr_str = unsafe { std::ffi::CStr::from_ptr(c_addr) }.to_string_lossy();
            if let Ok(socket) = std::net::UdpSocket::bind(addr_str.as_ref()) {
                let boxed: Box<std::net::UdpSocket> = Box::new(socket);
                Box::into_raw(boxed).cast::<std::ffi::c_void>()
            } else {
                std::ptr::null_mut()
            }
        }

        /// Waits for one datagram, copies up to `len` bytes of it into `buffer`, and returns
        /// how many bytes were copied. Returns 0 on an error, a null pointer, or a zero length.
        #[no_mangle]
        pub extern "C" fn vx_udp_socket_recv(
            ptr: *mut std::ffi::c_void,
            buffer: *mut u8,
            len: u64,
        ) -> u64 {
            if ptr.is_null() || buffer.is_null() {
                return 0;
            }
            let Some(len) = $crate::ffi::macros::buffer_len(len) else {
                return 0;
            };
            // SAFETY: the caller passes a live pointer that this library returned, and nothing else
            // uses it during this call.
            let socket = unsafe { &mut *ptr.cast::<std::net::UdpSocket>() };
            // SAFETY: the caller passes a buffer of at least `len` bytes that stays alive for this
            // call.
            let buf_slice = unsafe { std::slice::from_raw_parts_mut(buffer, len) };
            // Note: We ignore the peer address for simplicity in the FFI.
            socket.recv(buf_slice).unwrap_or(0) as u64
        }

        /// Sends `len` bytes from `buffer` to the address in `c_addr` and returns how many
        /// were sent. Returns 0 on an error, a null pointer, or a zero length.
        #[no_mangle]
        pub extern "C" fn vx_udp_socket_send_to(
            ptr: *mut std::ffi::c_void,
            buffer: *const u8,
            len: u64,
            c_addr: *const std::ffi::c_char,
        ) -> u64 {
            if ptr.is_null() || buffer.is_null() || c_addr.is_null() {
                return 0;
            }
            let Some(len) = $crate::ffi::macros::buffer_len(len) else {
                return 0;
            };
            // SAFETY: the caller passes a live pointer that this library returned, and nothing else
            // uses it during this call.
            let socket = unsafe { &mut *ptr.cast::<std::net::UdpSocket>() };
            // SAFETY: the caller passes a buffer of at least `len` bytes that stays alive for this
            // call.
            let buf_slice = unsafe { std::slice::from_raw_parts(buffer, len) };
            // SAFETY: the caller passes a NUL-terminated C string that stays alive for this call.
            let addr_str = unsafe { std::ffi::CStr::from_ptr(c_addr) }.to_string_lossy();
            socket.send_to(buf_slice, addr_str.as_ref()).unwrap_or(0) as u64
        }

        /// Closes the socket and frees it. Does nothing when `ptr` is null.
        #[no_mangle]
        pub extern "C" fn vx_udp_socket_drop(ptr: *mut std::ffi::c_void) {
            if !ptr.is_null() {
                // SAFETY: the pointer came from `Box::into_raw` in this library and the caller
                // gives up ownership here.
                let _ = unsafe { Box::from_raw(ptr.cast::<std::net::UdpSocket>()) };
            }
        }
    };
}

/// Macro to instantiate C-ABI compatible FFI wrappers for `std::net::TcpListener`.
#[macro_export]
macro_rules! instantiate_tcp_listener_ffi {
    () => {
        /// Starts listening for TCP connections on the address in `c_addr`. Returns null when
        /// the address is null or the listener cannot be opened.
        #[no_mangle]
        pub extern "C" fn vx_tcp_listener_bind(
            c_addr: *const std::ffi::c_char,
        ) -> *mut std::ffi::c_void {
            if c_addr.is_null() {
                return std::ptr::null_mut();
            }
            // SAFETY: the caller passes a NUL-terminated C string that stays alive for this call.
            let addr_str = unsafe { std::ffi::CStr::from_ptr(c_addr) }.to_string_lossy();
            if let Ok(listener) = std::net::TcpListener::bind(addr_str.as_ref()) {
                let boxed: Box<std::net::TcpListener> = Box::new(listener);
                Box::into_raw(boxed).cast::<std::ffi::c_void>()
            } else {
                std::ptr::null_mut()
            }
        }

        /// Waits for the next connection and returns it as a TCP stream, which the caller
        /// must free with `vx_tcp_stream_drop`. Returns null on an error or a null `ptr`.
        #[no_mangle]
        pub extern "C" fn vx_tcp_listener_accept(
            ptr: *mut std::ffi::c_void,
        ) -> *mut std::ffi::c_void {
            if ptr.is_null() {
                return std::ptr::null_mut();
            }
            // SAFETY: the caller passes a live pointer that this library returned, and nothing else
            // uses it during this call.
            let listener = unsafe { &mut *ptr.cast::<std::net::TcpListener>() };
            if let Ok((stream, _)) = listener.accept() {
                let boxed: Box<std::net::TcpStream> = Box::new(stream);
                Box::into_raw(boxed).cast::<std::ffi::c_void>()
            } else {
                std::ptr::null_mut()
            }
        }

        /// Stops listening and frees the listener. Does nothing when `ptr` is null.
        #[no_mangle]
        pub extern "C" fn vx_tcp_listener_drop(ptr: *mut std::ffi::c_void) {
            if !ptr.is_null() {
                // SAFETY: the pointer came from `Box::into_raw` in this library and the caller
                // gives up ownership here.
                let _ = unsafe { Box::from_raw(ptr.cast::<std::net::TcpListener>()) };
            }
        }
    };
}

/// Macro to instantiate C-ABI compatible FFI wrappers for Standard I/O Streams.
#[macro_export]
macro_rules! instantiate_stdio_ffi {
    () => {
        /// Writes up to `len` bytes from `buffer` to standard output and returns how many
        /// were written. Returns 0 on an error, a null `buffer`, or a zero length.
        #[no_mangle]
        pub extern "C" fn vx_stdout_write(buffer: *const u8, len: u64) -> u64 {
            if buffer.is_null() {
                return 0;
            }
            let Some(len) = $crate::ffi::macros::buffer_len(len) else {
                return 0;
            };
            // SAFETY: the caller passes a buffer of at least `len` bytes that stays alive for this
            // call.
            let buf_slice = unsafe { std::slice::from_raw_parts(buffer, len) };
            std::io::Write::write(&mut std::io::stdout(), buf_slice).unwrap_or(0) as u64
        }

        /// Writes up to `len` bytes from `buffer` to standard error and returns how many
        /// were written. Returns 0 on an error, a null `buffer`, or a zero length.
        #[no_mangle]
        pub extern "C" fn vx_stderr_write(buffer: *const u8, len: u64) -> u64 {
            if buffer.is_null() {
                return 0;
            }
            let Some(len) = $crate::ffi::macros::buffer_len(len) else {
                return 0;
            };
            // SAFETY: the caller passes a buffer of at least `len` bytes that stays alive for this
            // call.
            let buf_slice = unsafe { std::slice::from_raw_parts(buffer, len) };
            std::io::Write::write(&mut std::io::stderr(), buf_slice).unwrap_or(0) as u64
        }

        /// Reads up to `len` bytes from standard input into `buffer` and returns how many
        /// were read. Returns 0 on an error, a null `buffer`, or a zero length.
        #[no_mangle]
        pub extern "C" fn vx_stdin_read(buffer: *mut u8, len: u64) -> u64 {
            if buffer.is_null() {
                return 0;
            }
            let Some(len) = $crate::ffi::macros::buffer_len(len) else {
                return 0;
            };
            // SAFETY: the caller passes a buffer of at least `len` bytes that stays alive for this
            // call.
            let buf_slice = unsafe { std::slice::from_raw_parts_mut(buffer, len) };
            std::io::Read::read(&mut std::io::stdin(), buf_slice).unwrap_or(0) as u64
        }
    };
}
