//! El binario de tests unitarios. Un `mod` por módulo de `src/`.
//!
//! Los tests viven fuera de `src/` a propósito: son otro crate, así que solo
//! ven la API pública y no pueden apoyarse en lo privado para pasar.

mod dobles;
mod execution;
mod graph;
mod plan;
mod value;
