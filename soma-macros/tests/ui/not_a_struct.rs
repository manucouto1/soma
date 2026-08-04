//! The error names the offending item, not the derive.
use somatize_core::SomaFilter;

#[derive(SomaFilter, serde::Serialize)]
enum Mode {
    Fast,
    Slow,
}

fn main() {}
