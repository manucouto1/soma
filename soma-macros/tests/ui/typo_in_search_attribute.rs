//! A misspelled `#[soma(...)]` must not compile.
//!
//! This is the whole point of the strict parser: `serach` used to fall
//! through every `is_ident` branch and return `Ok(())`, so the filter
//! compiled with no search dimension at all. A sweep over it then
//! explored nothing and still reported a best trial.
use somatize_core::SomaFilter;

#[derive(SomaFilter, serde::Serialize)]
struct Scaler {
    #[soma(serach(low = 0.1, high = 2.0))]
    factor: f64,
}

fn main() {}
