//! Where the data comes from, and what it is once it arrives.
//!
//! # A source is a node, and that is the whole design
//!
//! There is no `Source` trait here, and there was nearly one. A source takes
//! something and answers with something, which is [`Node`](somatize_core::Node)
//! — and a second trait with a method that does what `forward` does is both a
//! hole with one tenant and the `E0034` this project's rules warn about.
//!
//! Being a node is not a saving, it is the point: a source gets
//! `.at()` — read where the data is —, `.cached()`, `.mapped()`, the record and
//! the figure, without one line here.
//!
//! # What a span buys, measured
//!
//! The graph's input is the one value hashed **by its content**, so a cache has
//! to look at all of it before it can say whether it already has the answer.
//! Measured on 24 August 2026, release build, one node that returns a constant
//! so that only the input grows:
//!
//! | what is handed to `forward` | with a store behind it |
//! |---|---|
//! | 1 MB of tensor | 6,1 ms |
//! | 19 MB of tensor (32×3×224×224) | **121 ms**, on every step, hit or miss |
//! | a [`Span`] | **0,027 ms** |
//!
//! Nothing there is pathological — `torch.save` is 1 ms/MB and sha256 is 2 ms/MB
//! — which is why the answer is not a faster hash. It is to stop handing the
//! graph an anonymous heap of numbers and hand it **a reference** instead: the
//! rows are named by where they came from, and where they came from is a
//! sentence, not a batch.
//!
//! # Which leaves the version, and the store had already computed it
//!
//! The other half of the name is the dataset's own: two runs against different
//! data must not share a key. A source has to state its identity **without
//! reading itself**, or it does the very work the cache exists to avoid.
//!
//! Against a [`Store`](somatize_store::Store) that is free: a name resolves to
//! a digest, and the digest **is** the content hash. So
//! [`Parquet::version`] costs one `resolve` and no bytes, and it goes where the
//! digest of settled weights goes — `Memory::freeze(id, Some(version))`, the
//! call that is made twice on purpose.
//!
//! # What is not here
//!
//! - **SQL**, and it is a decision rather than a delay: every Rust driver worth
//!   using carries a runtime, and `store/Cargo.toml` already says what that
//!   costs — *«an SDK with a runtime inside would have made the trait async, and
//!   that is the objection that has kept a bus out of this repo twice»*. When
//!   SQL arrives it arrives synchronous.
//! - **Ranged reads.** [`Store::get`](somatize_store::Store::get) answers with
//!   every byte of a blob, so a file is read once and held. A dataset that does
//!   not fit in memory needs the store to learn to read a range first.

mod frame;
mod ipc;
mod parquet;
mod span;

pub use frame::{Frame, FrameError};
pub use ipc::Ipc;
pub use parquet::{Parquet, ParquetError};
pub use span::{Span, SpanError};
