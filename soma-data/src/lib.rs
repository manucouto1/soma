//! Where the data comes from, and what it is once it arrives.
//!
//! **A source is a node**, and there is no `Source` trait: a source takes
//! something and answers with something, which is
//! [`Node`](somatize_core::Node), and a second trait with a method that does
//! what `forward` does is both a hole with one tenant and the `E0034` the rules
//! warn about. Being a node is the point — `.at()`, `.cached()`, `.mapped()`,
//! the record and the figure all reach a dataset with nothing written here.
//!
//! What the graph is handed is a **coordinate** and not a batch, because the
//! input is the one value a cache hashes by content. Measured on 24 August 2026,
//! release build, one node returning a constant so only the input grows:
//!
//! | what is handed to `forward` | with a store behind it |
//! |---|---|
//! | 1 MB of tensor | 6,1 ms |
//! | 19 MB of tensor (32×3×224×224) | **121 ms**, every step, hit or miss |
//! | a [`Span`] | **0,027 ms** |
//!
//! Nothing there is pathological — `torch.save` is 1 ms/MB and sha256 2 ms/MB —
//! which is why the answer is not a faster hash.
//!
//! The other half of the name is the dataset's version, and a source has to
//! state it **without reading itself**. Against a
//! [`Store`](somatize_store::Store) that is free: a name resolves to a digest
//! and the digest **is** the content hash, so [`Parquet::version`] costs one
//! `resolve` and no bytes.
//!
//! Not here: **SQL**, decided rather than delayed, since every Rust driver worth
//! using carries a runtime and `Store` is synchronous on purpose; and **ranged
//! reads**, which need the store to learn to read a range first.

mod frame;
mod ipc;
mod parquet;
mod span;

pub use frame::{Frame, FrameError};
pub use ipc::Ipc;
pub use parquet::{Parquet, ParquetError};
pub use span::{Span, SpanError};
