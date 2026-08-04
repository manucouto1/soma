//! An unknown argument inside `search(...)` must not compile either.
use somatize_core::SomaFilter;

#[derive(SomaFilter, serde::Serialize)]
struct Scaler {
    #[soma(search(lo = 0.1, high = 2.0))]
    factor: f64,
}

fn main() {}
