use googletest::prelude::*;

/// Checks that two `f32` values are equal, and returns 0 when they are.
///
/// # Panics
/// Panics with a message that shows both values when they differ.
#[no_mangle]
pub extern "C" fn vx_googletest_expect_eq_f32(actual: f32, expected: f32) -> i32 {
    assert_that!(actual, eq(expected));
    0
}

/// Checks that two `i32` values are equal, and returns 0 when they are.
///
/// # Panics
/// Panics with a message that shows both values when they differ.
#[no_mangle]
pub extern "C" fn vx_googletest_expect_eq_i32(actual: i32, expected: i32) -> i32 {
    assert_that!(actual, eq(expected));
    0
}
