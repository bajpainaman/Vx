use std::ffi::{c_char, c_void, CStr};
use std::fs::File;
use std::io::Read;
use std::ptr;

// ============================================================================
// Llama2 Model Loading Helpers
// ============================================================================

/// Reads the seven `i32` values at the start of a model file. Returns a pointer to them
/// that is never freed, or null when the path is null or the file cannot be read.
#[no_mangle]
pub extern "C" fn vx_load_config(filepath: *const c_char) -> *mut i32 {
    if filepath.is_null() {
        return ptr::null_mut();
    }
    // SAFETY: the caller passes a NUL-terminated C string that stays alive for this call.
    let path = unsafe { CStr::from_ptr(filepath) }.to_string_lossy();
    if let Ok(mut f) = File::open(path.as_ref()) {
        let mut bytes = [0u8; 28]; // 7 * 4 bytes
        if f.read_exact(&mut bytes).is_ok() {
            let mut config = vec![0i32; 7];
            // SAFETY: `bytes` and `config` are both 28 bytes long and do not overlap.
            unsafe {
                ptr::copy_nonoverlapping(bytes.as_ptr(), config.as_mut_ptr().cast::<u8>(), 28);
            }
            return config.leak().as_mut_ptr();
        }
    }
    ptr::null_mut()
}

/// Reads a whole model file into memory and returns a pointer to the weights, which
/// start after the seven `i32` values of the header. The memory is never freed. Returns
/// null when the path is null, the file cannot be read, or there is no memory left.
#[no_mangle]
#[allow(
    clippy::cast_ptr_alignment,
    reason = "`malloc` returns memory that is aligned for every number type"
)]
pub extern "C" fn vx_load_weights(filepath: *const c_char) -> *mut f32 {
    if filepath.is_null() {
        return ptr::null_mut();
    }
    // SAFETY: the caller passes a NUL-terminated C string that stays alive for this call.
    let path = unsafe { CStr::from_ptr(filepath) }.to_string_lossy();
    if let Ok(bytes) = std::fs::read(path.as_ref()) {
        // SAFETY: the copy stays inside the `bytes.len()` bytes that `malloc` returned. The
        // caller passes a model file, which starts with a 28-byte header, so the pointer
        // after the header is still inside the buffer.
        unsafe {
            let ptr = libc::malloc(bytes.len()).cast::<u8>();
            if ptr.is_null() {
                return ptr::null_mut();
            }
            ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
            return ptr.cast::<f32>().add(7);
        }
    }
    ptr::null_mut()
}

/// The environment variable `name` as a number. Returns `default_val` when `name` is
/// null, or when the variable is not set or is not a number.
#[no_mangle]
pub extern "C" fn vx_get_env_int(name: *const c_char, default_val: i32) -> i32 {
    if name.is_null() {
        return default_val;
    }
    // SAFETY: the caller passes a NUL-terminated C string that stays alive for this call.
    let key = unsafe { CStr::from_ptr(name) }.to_string_lossy();
    std::env::var(key.as_ref())
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default_val)
}

#[no_mangle]
/// `[max_tokens, steps, disaggregate]`.
///
/// The third element asks whether decode should run on a second device.
/// It is here rather than behind `vx_get_env_int` so the program reads it the
/// same way it reads the other two, and so a run's whole configuration comes
/// from one call.
pub extern "C" fn vx_get_llama_config() -> *mut i32 {
    let mut config = vec![1000i32, 1i32, 0i32];
    if let Ok(val) = std::env::var("LLAMA_TOKENS_CONFIG") {
        let parts: Vec<&str> = val.split(';').collect();
        if parts.len() == 2 {
            if let Ok(max) = parts[0].parse::<i32>() {
                config[0] = max;
            }
            if let Ok(steps) = parts[1].parse::<i32>() {
                config[1] = steps;
            }
        }
    }
    // Anything unparseable, or absent, means "one device" -- the mode that runs
    // everywhere. Asking for two on a box that has one aborts in the plugin
    // with the device count, which is the right place to find out.
    if let Ok(val) = std::env::var("VX_LLAMA_DISAGG") {
        if let Ok(flag) = val.parse::<i32>() {
            config[2] = flag;
        }
    }
    config.leak().as_mut_ptr()
}

// ============================================================================
// Llama2 Tokenizer
// ============================================================================

/// The vocabulary of a language model: the text of every token, by token number.
#[allow(
    dead_code,
    reason = "the scores, the longest token length and the one-byte strings are loaded for an encoder that does not read them yet"
)]
pub struct Tokenizer {
    vocab: Vec<String>,
    vocab_scores: Vec<f32>,
    vocab_size: i32,
    max_token_length: i32,
    byte_pieces: Vec<String>,
}

/// Loads a tokenizer with `vocab_size` tokens from a vocabulary file. Returns a pointer
/// that is never freed, or null when the path is null or the file cannot be opened. A
/// file that ends early gives a tokenizer with only the tokens that were read.
///
/// # Panics
/// Panics if `vocab_size` is negative, or if the file gives a negative length for a token.
#[no_mangle]
pub extern "C" fn vx_build_tokenizer(filepath: *const c_char, vocab_size: i32) -> *mut c_void {
    if filepath.is_null() {
        return ptr::null_mut();
    }
    // SAFETY: the caller passes a NUL-terminated C string that stays alive for this call.
    let path = unsafe { CStr::from_ptr(filepath) }.to_string_lossy();
    let Ok(mut file) = File::open(path.as_ref()) else {
        return ptr::null_mut();
    };

    let mut byte_pieces = Vec::new();
    for i in 0..256u32 {
        let mut s = String::new();
        s.push(char::from_u32(i).unwrap_or('?'));
        byte_pieces.push(s);
    }

    let mut max_token_length_buf = [0u8; 4];
    if file.read_exact(&mut max_token_length_buf).is_err() {
        return ptr::null_mut();
    }
    let max_token_length = i32::from_le_bytes(max_token_length_buf);

    let capacity =
        usize::try_from(vocab_size).expect("vx_build_tokenizer: negative vocabulary size");
    let mut vocab = Vec::with_capacity(capacity);
    let mut vocab_scores = Vec::with_capacity(capacity);

    for _ in 0..vocab_size {
        let mut score_buf = [0u8; 4];
        if file.read_exact(&mut score_buf).is_err() {
            break;
        }
        vocab_scores.push(f32::from_le_bytes(score_buf));

        let mut len_buf = [0u8; 4];
        if file.read_exact(&mut len_buf).is_err() {
            break;
        }
        let len = usize::try_from(i32::from_le_bytes(len_buf))
            .expect("vx_build_tokenizer: negative token length in the vocabulary file");

        let mut string_buf = vec![0u8; len];
        if file.read_exact(&mut string_buf).is_err() {
            break;
        }

        let s = String::from_utf8_lossy(&string_buf).into_owned();
        vocab.push(s);
    }

    let tokenizer = Box::new(Tokenizer {
        vocab,
        vocab_scores,
        vocab_size,
        max_token_length,
        byte_pieces,
    });

    Box::into_raw(tokenizer).cast::<c_void>()
}

/// The text of `token` as a new C string that is never freed. `prev_token` is the token
/// before it: a leading space is dropped right after the start-of-text token. Returns
/// null when the tokenizer is null or the token number is out of range.
///
/// # Panics
/// Panics if the vocabulary file had fewer tokens than the tokenizer was built for and
/// `token` is one of the missing ones, or if the token's text has a NUL byte in it.
#[no_mangle]
pub extern "C" fn vx_decode_token(
    tokenizer_ptr: *mut c_void,
    prev_token: i32,
    token: i32,
) -> *mut c_char {
    if tokenizer_ptr.is_null() {
        return ptr::null_mut();
    }
    // SAFETY: the caller passes a tokenizer that `vx_build_tokenizer` returned.
    let t = unsafe { &*tokenizer_ptr.cast::<Tokenizer>() };
    let Ok(index) = usize::try_from(token) else {
        return ptr::null_mut();
    };
    if token >= t.vocab_size {
        return ptr::null_mut();
    }
    let mut piece = t.vocab[index].clone();
    if prev_token == 1 && piece.starts_with(' ') {
        piece = piece.chars().skip(1).collect();
    }

    // Hex decoding fallback `<0xXX>`
    if piece.starts_with("<0x") && piece.ends_with('>') && piece.len() == 6 {
        if let Ok(byte_val) = u8::from_str_radix(&piece[3..5], 16) {
            piece = String::new();
            piece.push(byte_val as char);
        }
    } else if piece.is_empty() {
        // Tokens 3 to 258 stand for the single bytes 0 to 255.
        if let Ok(byte_val) = u8::try_from(token - 3) {
            piece = String::new();
            piece.push(byte_val as char);
        }
    }

    let c_str = std::ffi::CString::new(piece).unwrap();
    c_str.into_raw()
}

/// Turns text into token numbers. Returns an array that is never freed: its first
/// element is how many tokens follow, and the first token is always 1, the
/// start-of-text token. A null `text_ptr` gives just that one token. When `text_ptr` is
/// not null, `tokenizer_ptr` must be a tokenizer from `vx_build_tokenizer`.
///
/// # Panics
/// Panics if the text has more tokens than fit in an `i32`.
#[no_mangle]
pub extern "C" fn vx_encode_prompt(
    tokenizer_ptr: *mut c_void,
    text_ptr: *const c_char,
) -> *mut i32 {
    if text_ptr.is_null() {
        let tokens = vec![1i32, 1i32]; // length, BOS
        return tokens.leak().as_mut_ptr();
    }
    // SAFETY: the caller passes a NUL-terminated C string that stays alive for this call.
    let text = unsafe { CStr::from_ptr(text_ptr) }.to_string_lossy();
    let mut tokens = vec![0i32; 1]; // Reserve index 0 for length
    tokens.push(1); // BOS

    // SAFETY: the caller passes a tokenizer that `vx_build_tokenizer` returned.
    let t = unsafe { &*tokenizer_ptr.cast::<Tokenizer>() };

    for c in text.chars() {
        let single = c.to_string();
        if let Some(pos) = t.vocab.iter().position(|v| v == &single) {
            // The vocabulary never has more tokens than its `i32` size.
            tokens.push(i32::try_from(pos).expect("vx_encode_prompt: token number over i32"));
        } else {
            let mut byte_buf = [0u8; 4];
            let bytes = c.encode_utf8(&mut byte_buf).as_bytes();
            for &b in bytes {
                tokens.push(i32::from(b) + 3);
            }
        }
    }

    tokens[0] = i32::try_from(tokens.len()).expect("vx_encode_prompt: too many tokens") - 1;
    tokens.leak().as_mut_ptr()
}

/// Reads a whole text file into a new C string that is never freed. Returns null when
/// the path is null or the file cannot be read as UTF-8 text.
///
/// # Panics
/// Panics if the file has a NUL byte in it, because a C string cannot hold one.
#[no_mangle]
pub extern "C" fn vx_read_prompt_file(filepath: *const c_char) -> *mut c_char {
    if filepath.is_null() {
        return ptr::null_mut();
    }
    // SAFETY: the caller passes a NUL-terminated C string that stays alive for this call.
    let path = unsafe { CStr::from_ptr(filepath) }.to_string_lossy();
    if let Ok(text) = std::fs::read_to_string(path.as_ref()) {
        let c_str = std::ffi::CString::new(text).unwrap();
        return c_str.into_raw();
    }
    ptr::null_mut()
}
