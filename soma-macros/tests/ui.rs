//! Compile-fail tests for the derives.
//!
//! The crate had no tests of any kind, while its parser silently ignored
//! anything it did not recognise — so a misspelled attribute produced a
//! filter with no search space and nothing anywhere said so. These lock
//! in that such input is a compile error, and that the message points at
//! the offending token.

#[test]
fn malformed_attributes_are_compile_errors() {
    trybuild::TestCases::new().compile_fail("tests/ui/*.rs");
}
